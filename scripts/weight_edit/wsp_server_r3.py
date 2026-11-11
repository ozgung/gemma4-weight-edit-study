"""WP11 forward-pass server, harness version 3 (round 3): one MLX process serving budgeted forward passes.  Owner: Claude.

    outputs/small_model/mlx_venv/bin/python -I scripts/weight_edit/wsp_server_r3.py serve [--once]
    outputs/small_model/mlx_venv/bin/python -I scripts/weight_edit/wsp_server_r3.py init RUN_ID --proposer opus \\
        --model claude-opus-5-5 --effort max --budget 25 --fullcall

Runs live in outputs/weight_edit/proposers/<run_id>/ with state.json, requests/, edit/ (wsp_kernel), work/.
A proposer writes requests/fwd_<k>.json; the server claims it (renames to .claimed), runs it, writes
requests/fwd_<k>.done.json and counts one pass if it succeeded. A malformed or refused request does not count.
This server only serves runs initialised for harness 3.

Callable set: the packet's search/calls.jsonl, 1,050 calls. Each call has set = "train" (730 calls, the four training
repositories) or "dev" (320 calls, other repositories, from the round-2 judging set, now open), is_retry and failed_set.
failed_set: the old_strings of the earlier recorded not_found calls in the same run and task. A retry call has a
non-empty failed_set; a first-attempt call has an empty one. A generation is a repeat when it is byte-identical to a
string in the call's failed_set.

Request types (limits are frozen for all proposers; see LIMITS):
  Optional in any request: "expect_edit": {file name: sha256} pins the edit/ snapshot the pass must apply (the
  request fails, uncounted, on a mismatch); every .done.json records "edit_snapshot". Requests of a run are served
  in numeric order of k.
  Harness 3 has no raw rendering: every prompt is the served one, exactly as the agent saw it. A request with
  "rendering": "raw" fails, uncounted.
  {"type": "gen", "arms": ["base", "edit"], "calls": [keys] | {"sample": N, "seed": S}, "draws": 1}
      Regenerate old_string on search-set replay calls (served prompt + the model's own prefix) and score each
      with the harness's own edit function (swegemma.edit.apply_replacement on the true file state). Draw 0 is greedy,
      draw 1 is sampled at the served settings (T 0.6, top_p 0.95, top_k 64); seeds are per key and draw, the same in all
      arms. Returns per arm: accepted / not_found / multiple rates, exact copies, compile failures among accepted
      edits, escape_artefact (share of generations whose old_string contains a literal backslash-n or backslash-quote
      that the true file does not contain), per recorded stratum, and the paired base-vs-edit tables for acceptance and
      artefacts.
      Harness 3: every arm also has "by_set", with "train" and "dev" (a set is left out if the request has no call in it):
        n_calls, n_gens
        overall_acceptance                 accepted generations / generations
        compile_fail_among_accepted        compile failures / accepted generations (null if none accepted)
        retry: n_calls, n_gens, repeats, repeat_rate, accepted, acceptance, repeats_accepted
        first: n_calls, n_gens, accepted, acceptance, identical_to_base, identical_to_base_share
        served_ok_lost_per_draw: {draw: {lost, base_accepted_served_ok}}: calls whose recorded outcome is ok and that
                                 base accepts, and that this arm does not accept, for each draw
      identical_to_base compares the generated text with base's at the same key and draw. An "edit" arm always runs with
      "base" (added if the request leaves it out), so these figures exist for every edit arm. Base is free once cached.
      Per-call rows (with generated text) are saved in work/fwd_<k>/rows.jsonl. Base generations are cached (same
      seeds), so "base" is free after the first time.
  {"type": "read", "arm": "base" | "edit", "texts": [{"key": K, "continuation": "recorded" | "<text>"}],
   "layers": [ints], "capture": ["resid", "mlp", "attn", "logprobs"], "positions": "continuation" | "prompt_last"}
      Teacher-forced pass over served prompt + prefix + continuation. Saves to work/fwd_<k>/text_<i>.npz:
      resid [n_layers_req, P, 2560] (residual stream after each requested layer), mlp [.., P, d_ff]
      (geglu activations entering down_proj), attn [.., heads, P, L] (attention probabilities of the requested
      layers for the P positions), logprobs [P] of the continuation tokens and the top-5 alternatives.
      logprobs / top-5 come from the model's native cached forward (the same code path as generation);
      logprobs_manual [P] are the same targets from the capture forward, which agrees with the native forward
      to about 0.5 nats on long prompts (resid / mlp / attn come from the capture forward).
  Full-call option, for runs initialised with --fullcall (state.json "fullcall": true): every gen request also runs a
      full-call replay of the same calls and draws (unless the request says "fullcall": false). The served prompt + the
      model's own text up to `new_string:<|"|>` is the prefix; the arm generates new_string until <|"|>, then
      `<|"|>,old_string:<|"|>` is teacher-forced and the arm generates old_string, so both arguments come from the same
      arm (the judge's compile guard uses this replay). The pair goes through apply_replacement on the true file state
      and the result is compiled. Returned under "fullcall": per arm n, accepted, not_found, exact_copy, escape_artefact
      (old_string), escape_artefact_new, unclosed_new, compile_fail_among_accepted, by_recorded_stratum, paired base-vs-edit
      tables, and "by_set" as above, where a repeat is a generated old_string in failed_set and identical_to_base means
      both generated arguments equal base's. "skipped" lists calls whose prefix cannot be cut at new_string (they are left
      out of the full-call figures only). Per-call rows in work/fwd_<k>/rows_fullcall.jsonl (gen_new, gen). Base
      full-call generations are cached. A pass takes about two to three times as long with the full call.
Edit arms apply the run's current edit (edit/*.npz, W' = W + B @ A) for the request only.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/weight_edit"))
import wsp_server as V2  # noqa: E402

V2_SHA256 = "4ed63ba7b39fcbd02216c2593c8a2da9ca05cbe8ab56a23c6fda0fc8eb6fe5d2"
PROPOSERS = ROOT / "outputs/weight_edit/proposers"
SEARCH_DIR = ROOT / "outputs/weight_edit/shared/search_r3"
CACHE = ROOT / "outputs/weight_edit/wsp_cache_r3"  # the v2 caches in wsp_cache/ stay untouched
HARNESS_VERSION = 3  # v3 (2026-11-07): dev set in the callable set, retry and first-attempt figures by set, no raw rendering
LIMITS = {"gen_calls": 128, "gen_draws": 2, "read_texts": 32, "read_layers": 8, "read_positions": 256,
          "attn_layers": 2, "continuation_tokens": 512}

assert LIMITS == V2.LIMITS, "harness 3 keeps the frozen limits"
# The v2 methods read these module globals at call time, so v3 only has to rebind them (and never touches wsp_cache/).
V2.SEARCH_DIR, V2.REPLAY, V2.CACHE, V2.HARNESS_VERSION = SEARCH_DIR, SEARCH_DIR, CACHE, HARNESS_VERSION


def check_v2_source() -> None:
    got = hashlib.sha256(Path(V2.__file__).read_bytes()).hexdigest()
    if got != V2_SHA256:
        raise SystemExit(f"wsp_server.py changed: {got} (harness 3 is built on {V2_SHA256})")


def read_rows(path: Path) -> dict[str, dict]:
    """{arm: {(key, draw): row}} from rows.jsonl or rows_fullcall.jsonl; the last record of a (key, draw) wins."""
    out: dict[str, dict] = collections.defaultdict(dict)
    for r in map(json.loads, path.open()):
        out[r["arm"]][(r["key"], r["draw"])] = r
    return out


def set_metrics(rows: dict, base: dict | None, calls: dict, fields: tuple[str, ...] = ("gen",)) -> dict:
    """The by_set block of one arm. rows and base are {(key, draw): row}; fields are compared for identical_to_base."""
    out = {}
    for name in ("train", "dev"):
        ks = [k for k in rows if calls[k[0]]["set"] == name]
        if not ks:
            continue
        acc = [k for k in ks if rows[k]["accepted"]]
        ret = [k for k in ks if calls[k[0]]["is_retry"]]
        fst = [k for k in ks if not calls[k[0]]["is_retry"]]
        rep = [k for k in ret if rows[k]["gen"] in calls[k[0]]["failed_set"]]
        same = None
        if base is not None:
            same = sum(all(rows[k][f] == base[k][f] for f in fields) for k in fst)
        lost = {}
        for d in sorted({k[1] for k in ks}):
            okb = [k for k in ks if k[1] == d and base is not None and base[k]["recorded_kind"] == "ok" and base[k]["accepted"]]
            lost[str(d)] = {"lost": sum(not rows[k]["accepted"] for k in okb), "base_accepted_served_ok": len(okb)}
        out[name] = {
            "n_calls": len({k[0] for k in ks}), "n_gens": len(ks), "overall_acceptance": len(acc) / len(ks),
            "compile_fail_among_accepted": sum(rows[k]["compiles"] == 0 for k in acc) / len(acc) if acc else None,
            "retry": {"n_calls": len({k[0] for k in ret}), "n_gens": len(ret), "repeats": len(rep),
                      "repeat_rate": len(rep) / len(ret) if ret else None,
                      "accepted": sum(rows[k]["accepted"] for k in ret),
                      "acceptance": sum(rows[k]["accepted"] for k in ret) / len(ret) if ret else None,
                      "repeats_accepted": sum(rows[k]["accepted"] for k in rep)},
            "first": {"n_calls": len({k[0] for k in fst}), "n_gens": len(fst), "accepted": sum(rows[k]["accepted"] for k in fst),
                      "acceptance": sum(rows[k]["accepted"] for k in fst) / len(fst) if fst else None,
                      "identical_to_base": same,
                      "identical_to_base_share": same / len(fst) if (fst and same is not None) else None},
            "served_ok_lost_per_draw": lost,
        }
    return out


class ServerR3(V2.Server):
    def __init__(self):
        import mlx.core as mx
        from mlx.utils import tree_flatten
        from mlx_lm.utils import load_model
        from tokenizers import Tokenizer

        check_v2_source()
        self.mx = mx
        self.tok = Tokenizer.from_file(str(V2.MODEL / "tokenizer.json"))
        self.q_id = self.tok.token_to_id(V2.Q)
        self.sep_ids = self.tok.encode(V2.SEP, add_special_tokens=False).ids
        assert self.sep_ids[0] == self.q_id and self.sep_ids[-1] == self.q_id, self.sep_ids
        t = time.time()
        self.model, _ = load_model(V2.MODEL, lazy=False, strict=True)
        self.params = dict(tree_flatten(self.model.parameters()))
        self.load_tables()
        print(f"[{V2.now()}] server v3 up: model {time.time() - t:.0f}s, search calls {len(self.calls)}, "
              f"cached base gens {len(self.base_cache)}, full-call {len(self.base_fc)}", flush=True)

    def load_tables(self) -> None:
        """Calls and base caches (split from __init__ so tests can run without a model)."""
        self.calls = V2.search_calls()
        self.raw_prompts = {}
        self._files = {}
        V2.CACHE.mkdir(parents=True, exist_ok=True)
        self.base_cache, self.base_cache_raw, self.base_fc, self.base_fc_raw = {}, {}, {}, {}
        for d, name in ((self.base_cache, "base_gen.jsonl"), (self.base_fc, "base_fullcall.jsonl")):
            if (V2.CACHE / name).exists():
                for g in map(json.loads, (V2.CACHE / name).open()):
                    d[(g["key"], g["draw"])] = g

    def call(self, key: str, rendering: str) -> dict:
        if rendering != "served":
            raise ValueError("harness 3 has no raw rendering; every prompt is the served one")
        return self.calls[key]

    def score(self, gens: list[dict], work: Path, tag: str) -> list[dict]:
        p = work / f"gen_{tag}.jsonl"
        with p.open("w") as f:
            for g in gens:
                f.write(json.dumps(g) + "\n")
        subprocess.run([str(V2.VENV_PY), "-I", str(ROOT / "scripts/weight_edit/replay_score.py"), str(p), str(SEARCH_DIR)],
                       check=True, capture_output=True, text=True)
        return [json.loads(l) for l in p.with_suffix(".scored.jsonl").open()]

    def do_gen(self, req: dict, run_dir: Path, work: Path, fullcall: bool = False) -> dict:
        arms = list(req.get("arms", ["base", "edit"]))
        if "edit" in arms and "base" not in arms:
            arms = ["base", *arms]  # the comparisons against base must exist for every edit arm
        res = super().do_gen({**req, "arms": arms}, run_dir, work, fullcall)
        rows = read_rows(work / "rows.jsonl")
        for arm, st in res["arms"].items():
            st["by_set"] = set_metrics(rows[arm], rows.get("base"), self.calls)
        fc = res.get("fullcall")
        if fc and fc["arms"]:
            rows = read_rows(work / "rows_fullcall.jsonl")
            for arm, st in fc["arms"].items():
                st["by_set"] = set_metrics(rows[arm], rows.get("base"), self.calls, ("gen_new", "gen"))
        return res

    def loop(self, once: bool = False, proposers: Path = PROPOSERS) -> None:
        def harness3(p: Path) -> bool:
            s = p.parent.parent / "state.json"
            return s.exists() and json.loads(s.read_text()).get("harness_version") == HARNESS_VERSION

        while True:
            pending = sorted((p for p in proposers.glob("*/requests/fwd_*.json") if not p.name.endswith(".done.json") and harness3(p)),
                             key=lambda p: (p.parent.parent.name, int(p.stem.split("_")[1])))
            for p in pending:
                self.handle(p)
            if once:
                return
            time.sleep(2)


def init_run(a) -> None:
    run = a.proposers / a.run_id
    if run.exists():
        sys.exit(f"{run} exists")
    for d in ("requests", "edit", "work"):
        (run / d).mkdir(parents=True)
    state = {"run_id": a.run_id, "proposer": a.proposer, "model": a.model, "effort": a.effort, "budget": a.budget,
             "used": 0, "created_utc": V2.now(), "limits": LIMITS, "harness_version": HARNESS_VERSION, "fullcall": bool(a.fullcall),
             "search_calls_sha256": hashlib.sha256((SEARCH_DIR / "calls.jsonl").read_bytes()).hexdigest()}
    (run / "state.json").write_text(json.dumps(state, indent=1))
    (run / "notebook.md").write_text(f"# Notebook of run {a.run_id}\n\n")
    print(run)


def seed_cache() -> None:
    """Copy the v2 base caches' rows for calls still callable into the v3 cache (same keys, seeds and generation code)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    keys = set(V2.search_calls())
    for name in ("base_gen.jsonl", "base_fullcall.jsonl"):
        dst = CACHE / name
        if dst.exists():
            sys.exit(f"{dst} exists")
        seen = {}
        for g in map(json.loads, (ROOT / "outputs/weight_edit/wsp_cache" / name).open()):
            if g["key"] in keys:
                seen.setdefault((g["key"], g["draw"]), g)
        with dst.open("w") as f:
            for g in seen.values():
                f.write(json.dumps(g) + "\n")
        print(f"{name}: {len(seen)} generations")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve")
    s.add_argument("--once", action="store_true")
    s.add_argument("--proposers", type=Path, default=PROPOSERS, help="run folders (tests only)")
    s.add_argument("--cache", type=Path, default=CACHE, help="base cache (tests only)")
    i = sub.add_parser("init")
    i.add_argument("run_id")
    i.add_argument("--proposer", required=True)
    i.add_argument("--model", required=True)
    i.add_argument("--effort", required=True)
    i.add_argument("--budget", type=int, default=25)
    i.add_argument("--fullcall", action="store_true", help="gen requests also run the full-call replay")
    i.add_argument("--proposers", type=Path, default=PROPOSERS, help="run folders (tests only)")
    sub.add_parser("seed-cache")
    a = ap.parse_args()
    if a.cmd == "init":
        init_run(a)
    elif a.cmd == "seed-cache":
        seed_cache()
    else:
        V2.CACHE = a.cache
        ServerR3().loop(once=a.once, proposers=a.proposers)


if __name__ == "__main__":
    main()
