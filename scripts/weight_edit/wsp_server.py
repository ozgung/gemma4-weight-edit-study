"""WP11 forward-pass server: one MLX process serving budgeted forward passes to proposer runs.  Owner: Claude.

    outputs/small_model/mlx_venv/bin/python -I scripts/weight_edit/wsp_server.py [--once]
    outputs/small_model/mlx_venv/bin/python -I scripts/weight_edit/wsp_server.py init RUN_ID --proposer opus \\
        --model claude-opus-5-5 --effort max --budget 10

Runs live in outputs/weight_edit/proposers/<run_id>/ with state.json, requests/, edit/ (wsp_kernel), work/.
A proposer writes requests/fwd_<k>.json; the server claims it (renames to .claimed), runs it, writes
requests/fwd_<k>.done.json and counts one pass if it succeeded. A malformed or refused request does not count.

Request types (limits are frozen for all proposers; see LIMITS):
  Optional in any request: "expect_edit": {file name: sha256} pins the edit/ snapshot the pass must apply (the
  request fails, uncounted, on a mismatch); every .done.json records "edit_snapshot". Requests of a run are served
  in numeric order of k.
  Optional in gen and read: "rendering": "served" (default; the prompt exactly as the agent saw it, tool responses
  JSON-escaped inside a string) | "raw" (the same prompt with every tool response re-rendered in the chat template's
  native mapping form, file text raw; search/calls_raw.jsonl). The model's own prefix is the same in both.
  {"type": "gen", "arms": ["base", "edit"], "calls": [keys] | {"sample": N, "seed": S}, "draws": 1}
      Regenerate old_string on search-set replay calls (served prompt + the model's own prefix) and score each
      with the harness's own edit function (swegemma.edit.apply_replacement on the true file state). Returns
      per arm: accepted / not_found / multiple rates, exact copies, compile failures among accepted edits, per
      recorded stratum, and the paired base-vs-edit table. v2: also escape_artefact (share of generations whose
      old_string contains a literal backslash-n or backslash-quote that the true file does not contain) per arm and
      stratum, and the paired base-vs-edit artefact table. Per-call rows (with generated text) are saved in
      work/fwd_<k>/rows.jsonl. Base generations are cached (same seeds), so "base" is free after the first time.
  {"type": "read", "arm": "base" | "edit", "texts": [{"key": K, "continuation": "recorded" | "<text>"}],
   "layers": [ints], "capture": ["resid", "mlp", "attn", "logprobs"], "positions": "continuation" | "prompt_last"}
      Teacher-forced pass over served prompt + prefix + continuation. Saves to work/fwd_<k>/text_<i>.npz:
      resid [n_layers_req, P, 2560] (residual stream after each requested layer), mlp [.., P, d_ff]
      (geglu activations entering down_proj), attn [.., heads, P, L] (attention probabilities of the requested
      layers for the P positions), logprobs [P] of the continuation tokens and the top-5 alternatives.
      v2: logprobs / top-5 come from the model's native cached forward (the same code path as generation);
      logprobs_manual [P] are the same targets from the capture forward, which agrees with the native forward
      to about 0.5 nats on long prompts (resid / mlp / attn come from the capture forward).
  v2c (2026-11-05), full-call option, only for runs initialised with --fullcall (state.json "fullcall": true):
      every gen request also runs a full-call replay of the same calls and draws (unless the request says
      "fullcall": false). The served prompt + the model's own text up to `new_string:<|"|>` is the prefix; the arm
      generates new_string until <|"|>, then `<|"|>,old_string:<|"|>` is teacher-forced and the arm generates
      old_string, so both arguments come from the same arm (the gate's compile guard of record uses this replay).
      The pair goes through apply_replacement on the true file state and the result is compiled. Returned under
      "fullcall": per arm n, accepted, not_found, exact_copy, escape_artefact (old_string), escape_artefact_new,
      unclosed_new, compile_fail_among_accepted, by_recorded_stratum, and paired base-vs-edit tables; "skipped"
      lists calls whose prefix cannot be cut at new_string (they are left out of the full-call figures only).
      Per-call rows in work/fwd_<k>/rows_fullcall.jsonl (gen_new, gen). Base full-call generations are cached.
      About twice the time of the frozen replay.
Edit arms apply the run's current edit (edit/*.npz, W' = W + B @ A) for the request only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/weight_edit"))
sys.path.insert(0, str(ROOT / "scripts/mlx_lora"))
PROPOSERS = ROOT / "outputs/weight_edit/proposers"
REPLAY = ROOT / "outputs/weight_edit/claude/replay"
SPLIT = ROOT / "outputs/weight_edit/shared/split.json"
CACHE = ROOT / "outputs/weight_edit/wsp_cache"
MODEL = ROOT / "data/models/gemma-4-e4b-it-qat-q4_0-unquantized"
VENV_PY = ROOT / ".venv/bin/python"
REPLAY = ROOT / "outputs/weight_edit/claude/replay"  # replay_score.py's default replay dir (file states)
Q = '<|"|>'
# Harness v2 (round 2, 2026-11-04): gen_calls 64 -> 128; callable set = shared/search_r2 (730 calls: the round-1
# search tasks plus the round-1 selection tasks); numeric request order; expect_edit pinning; per-pass edit snapshot.
HARNESS_VERSION = 2  # v2b: rendering option, escape_artefact metric
SEARCH_DIR = ROOT / "outputs/weight_edit/shared/search_r2"
SEP = Q + ",old_string:" + Q


def split_prefix(r: dict) -> str:
    """Full-call prefix: the model's own text up to and including `new_string:<|"|>` (as replay_fullcall.py)."""
    mark = "new_string:" + Q
    i = r["prefix"].rfind(mark)
    if i < 0 or r["prefix"][i + len(mark):] != (r.get("new_string") or "") + SEP:
        raise ValueError(f"prefix of {r['key']} cannot be cut at new_string")
    return r["prefix"][: i + len(mark)]


LIMITS = {"gen_calls": 128, "gen_draws": 2, "read_texts": 32, "read_layers": 8, "read_positions": 256,
          "attn_layers": 2, "continuation_tokens": 512}


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def search_calls() -> dict[str, dict]:
    """v2: every usable call of SEARCH_DIR (round 2: search plus former selection tasks; all training repos)."""
    out = {}
    for line in (SEARCH_DIR / "calls.jsonl").open():
        r = json.loads(line)
        if r["verified"] and not r["state_uncertain"] and not r["file_missing"] and r.get("old_string"):
            out[r["key"]] = r
    return out


def seed_of(key: str, draw: int) -> int:
    return int(hashlib.sha1(f"{key}|{draw}".encode()).hexdigest()[:8], 16)


class Server:
    def __init__(self):
        import mlx.core as mx
        from mlx_lm.utils import load_model
        from tokenizers import Tokenizer
        from mlx.utils import tree_flatten

        self.mx = mx
        self.tok = Tokenizer.from_file(str(MODEL / "tokenizer.json"))
        self.q_id = self.tok.token_to_id(Q)
        self.sep_ids = self.tok.encode(SEP, add_special_tokens=False).ids
        assert self.sep_ids[0] == self.q_id and self.sep_ids[-1] == self.q_id, self.sep_ids
        t = time.time()
        self.model, _ = load_model(MODEL, lazy=False, strict=True)
        self.params = dict(tree_flatten(self.model.parameters()))
        self.calls = search_calls()
        self.raw_prompts = {}
        for line in (SEARCH_DIR / "calls_raw.jsonl").open():
            r = json.loads(line)
            if r["key"] in self.calls:
                self.raw_prompts[r["key"]] = r["prompt"]
        assert set(self.raw_prompts) == set(self.calls), "calls_raw.jsonl does not cover the callable set"
        self._files = {}
        CACHE.mkdir(parents=True, exist_ok=True)
        self.base_cache = {}
        self.base_cache_raw = {}
        self.base_fc = {}
        self.base_fc_raw = {}
        for cache_d, name in ((self.base_cache, "base_gen.jsonl"), (self.base_cache_raw, "base_gen_raw.jsonl"),
                              (self.base_fc, "base_fullcall.jsonl"), (self.base_fc_raw, "base_fullcall_raw.jsonl")):
            if (CACHE / name).exists():
                for line in (CACHE / name).open():
                    g = json.loads(line)
                    cache_d[(g["key"], g["draw"])] = g
        print(f"[{now()}] server up: model {time.time() - t:.0f}s, search calls {len(self.calls)}, "
              f"cached base gens {len(self.base_cache)}", flush=True)

    # ---- edits -------------------------------------------------------------------------------------------
    def apply(self, run_dir: Path) -> dict:
        import numpy as np
        mx = self.mx
        from mlx.utils import tree_unflatten
        originals = {}
        for f in sorted((run_dir / "edit").glob("*.npz")):
            z = np.load(f, allow_pickle=False)
            name = str(z["module"])
            key = "language_model.model." + name.removeprefix("model.language_model.").removesuffix(".weight") + ".weight"
            W = self.params[key]
            D = mx.array(z["B"].astype(np.float32)) @ mx.array(z["A"].astype(np.float32))
            if D.shape != W.shape:
                raise ValueError(f"edit shape {D.shape} != {W.shape} for {name}")
            originals[key] = W
            self.model.update(tree_unflatten([(key, (W.astype(mx.float32) + D).astype(W.dtype))]))
        mx.eval(self.model.parameters())
        return originals

    def restore(self, originals: dict) -> None:
        from mlx.utils import tree_unflatten
        if originals:
            self.model.update(tree_unflatten(list(originals.items())))
            self.mx.eval(self.model.parameters())

    # ---- generation --------------------------------------------------------------------------------------
    def generate(self, r: dict, draws: int) -> list[dict]:
        import copy
        mx = self.mx
        from mlx_lm.generate import generate_step
        from mlx_lm.models.cache import make_prompt_cache
        from mlx_lm.sample_utils import make_sampler
        ids = self.tok.encode(r["prompt"] + r["prefix"], add_special_tokens=False).ids
        ref = len(self.tok.encode(r["old_string"], add_special_tokens=False).ids)
        max_new = min(1024, 2 * ref + 64)
        cache = make_prompt_cache(self.model)
        for s in range(0, len(ids) - 1, 2048):
            self.model(mx.array(ids[s:min(s + 2048, len(ids) - 1)])[None], cache=cache)
            mx.eval([c.state for c in cache])
        outs = []
        for d in range(draws):
            sampler = make_sampler(temp=0.0) if d == 0 else make_sampler(temp=0.6, top_p=0.95, top_k=64)
            mx.random.seed(seed_of(r["key"], d))
            c2 = copy.deepcopy(cache)
            gen, closed = [], 0
            for tid, _ in generate_step(mx.array(ids[-1:]), self.model, max_tokens=max_new, sampler=sampler, prompt_cache=c2):
                tid = int(tid)
                if tid == self.q_id:
                    closed = 1
                    break
                gen.append(tid)
            outs.append({"key": r["key"], "draw": d, "gen": self.tok.decode(gen, skip_special_tokens=False),
                         "closed": closed, "n_gen": len(gen)})
        return outs

    def score(self, gens: list[dict], work: Path, tag: str) -> list[dict]:
        p = work / f"gen_{tag}.jsonl"
        with p.open("w") as f:
            for g in gens:
                f.write(json.dumps(g) + "\n")
        subprocess.run([str(VENV_PY), "-I", str(ROOT / "scripts/weight_edit/replay_score.py"), str(p)],
                       check=True, capture_output=True, text=True)
        return [json.loads(l) for l in p.with_suffix(".scored.jsonl").open()]

    def generate_fullcall(self, r: dict, draws: int) -> list[dict]:
        """new_string then old_string from the same arm (replay_fullcall.py's procedure; seeds as generate())."""
        import copy
        mx = self.mx
        from mlx_lm.models.cache import make_prompt_cache
        from mlx_lm.sample_utils import make_sampler
        ids = self.tok.encode(r["prompt"] + split_prefix(r), add_special_tokens=False).ids
        ref_new = len(self.tok.encode(r.get("new_string") or "", add_special_tokens=False).ids)
        ref_old = len(self.tok.encode(r["old_string"], add_special_tokens=False).ids)
        cap_new, cap_old = min(2048, 2 * ref_new + 64), min(1024, 2 * ref_old + 64)
        cache = make_prompt_cache(self.model)
        for s in range(0, len(ids) - 1, 2048):
            self.model(mx.array(ids[s:min(s + 2048, len(ids) - 1)])[None], cache=cache)
            mx.eval([c.state for c in cache])

        def decode(c, first, sampler, max_new):
            logits = self.model(mx.array(first)[None], cache=c)[:, -1, :]
            out = []
            for _ in range(max_new):
                t = int(sampler(logits - mx.logsumexp(logits, axis=-1, keepdims=True)).item())
                if t == self.q_id:
                    return out, 1
                out.append(t)
                logits = self.model(mx.array([[t]]), cache=c)[:, -1, :]
            return out, 0

        outs = []
        for d in range(draws):
            sampler = make_sampler(temp=0.0) if d == 0 else make_sampler(temp=0.6, top_p=0.95, top_k=64)
            mx.random.seed(seed_of(r["key"], d))
            c2 = copy.deepcopy(cache)
            new, closed_new = decode(c2, ids[-1:], sampler, cap_new)
            old, closed = decode(c2, self.sep_ids, sampler, cap_old) if closed_new else ([], 0)
            outs.append({"key": r["key"], "draw": d, "gen_new": self.tok.decode(new, skip_special_tokens=False),
                         "gen": self.tok.decode(old, skip_special_tokens=False), "closed_new": closed_new,
                         "closed": closed, "n_gen_new": len(new), "n_gen": len(old)})
        return outs

    def score_fullcall(self, gens: list[dict], work: Path, tag: str) -> list[dict]:
        p = work / f"gen_fc_{tag}.jsonl"
        with p.open("w") as f:
            for g in gens:
                f.write(json.dumps(g) + "\n")
        subprocess.run([str(VENV_PY), "-I", str(ROOT / "scripts/weight_edit/replay_score_fullcall.py"), str(p), str(REPLAY)],
                       check=True, capture_output=True, text=True)
        return [json.loads(l) for l in p.with_suffix(".scored.jsonl").open()]

    def do_fullcall(self, keys: list[str], draws: int, arms: list[str], rendering: str, run_dir: Path, work: Path) -> dict:
        usable, skipped = [], []
        for k in keys:
            try:
                split_prefix(self.calls[k])
                usable.append(k)
            except ValueError:
                skipped.append(k)
        out = {"n_calls": len(usable), "skipped": skipped, "arms": {}}
        if not usable:
            return out
        bcache, bfile = (self.base_fc, "base_fullcall.jsonl") if rendering == "served" else (self.base_fc_raw, "base_fullcall_raw.jsonl")
        scored = {}
        for arm in arms:
            originals = self.apply(run_dir) if arm == "edit" else {}
            try:
                gens = []
                for k in usable:
                    if arm == "base" and all((k, d) in bcache for d in range(draws)):
                        gens += [bcache[(k, d)] for d in range(draws)]
                        continue
                    g = self.generate_fullcall(self.call(k, rendering), draws)
                    gens += g
                    if arm == "base":
                        with (CACHE / bfile).open("a") as f:
                            for x in g:
                                f.write(json.dumps(x) + "\n")
                                bcache[(x["key"], x["draw"])] = x
            finally:
                self.restore(originals)
            rows = self.score_fullcall(gens, work, arm)
            for x, g in zip(rows, gens):
                assert (x["key"], x["draw"]) == (g["key"], g["draw"])
            scored[arm] = {(x["key"], x["draw"]): x for x in rows}
            n = len(rows)
            acc = [x for x in rows if x["accepted"]]
            strata = {}
            for x in rows:
                st = x["recorded_kind"] + ("/" + str(x["recorded_strategy"]) if x["recorded_kind"] == "ok" else "")
                s = strata.setdefault(st, {"n": 0, "accepted": 0, "not_found": 0, "escape_artefact": 0, "compile_fail": 0})
                s["n"] += 1
                s["accepted"] += x["accepted"]
                s["not_found"] += int(x["outcome"] == "not_found")
                s["escape_artefact"] += x["artefact_old"]
                s["compile_fail"] += int(bool(x["accepted"]) and x["compiles"] == 0)
            out["arms"][arm] = {
                "n": n, "accepted": len(acc) / n,
                "not_found": sum(x["outcome"] == "not_found" for x in rows) / n,
                "exact_copy": sum(x["exact_copy"] for x in rows) / n,
                "escape_artefact": sum(x["artefact_old"] for x in rows) / n,
                "escape_artefact_new": sum(x["artefact_new"] for x in rows) / n,
                "unclosed_new": sum(1 - x["closed_new"] for x in rows) / n,
                "compile_fail_among_accepted": (sum(1 for x in acc if x["compiles"] == 0) / len(acc)) if acc else None,
                "by_recorded_stratum": strata,
            }
            with (work / "rows_fullcall.jsonl").open("a") as f:
                for x, g in zip(rows, gens):
                    f.write(json.dumps({"arm": arm, **x, "gen_new": g["gen_new"], "gen": g["gen"]}) + "\n")
        if set(arms) == {"base", "edit"}:
            t = {"both_accepted": 0, "only_base": 0, "only_edit": 0, "neither": 0}
            ta = {"both": 0, "only_base": 0, "only_edit": 0, "neither": 0}
            for k, b in scored["base"].items():
                e = scored["edit"][k]
                t["both_accepted" if b["accepted"] and e["accepted"] else "only_base" if b["accepted"] else
                  "only_edit" if e["accepted"] else "neither"] += 1
                ta["both" if b["artefact_old"] and e["artefact_old"] else "only_base" if b["artefact_old"] else
                   "only_edit" if e["artefact_old"] else "neither"] += 1
            out["paired_accepted"] = t
            out["paired_escape_artefact"] = ta
        return out

    def call(self, key: str, rendering: str) -> dict:
        if rendering not in ("served", "raw"):
            raise ValueError("rendering must be served or raw")
        r = self.calls[key]
        return dict(r, prompt=self.raw_prompts[key]) if rendering == "raw" else r

    def artefact(self, key: str, gen: str) -> int:
        sha = self.calls[key]["file_sha"]
        if sha not in self._files:
            self._files[sha] = (SEARCH_DIR / "files" / f"{sha}.txt").read_text()
        return int(any(e in gen and e not in self._files[sha] for e in ("\\n", '\\"')))

    def do_gen(self, req: dict, run_dir: Path, work: Path, fullcall: bool = False) -> dict:
        import random
        arms = req.get("arms", ["base", "edit"])
        draws = int(req.get("draws", 1))
        if not set(arms) <= {"base", "edit"} or not 1 <= draws <= LIMITS["gen_draws"]:
            raise ValueError("arms must be base/edit; draws 1..%d" % LIMITS["gen_draws"])
        sel = req.get("calls")
        if isinstance(sel, dict):
            keys = sorted(self.calls)
            random.Random(int(sel.get("seed", 0))).shuffle(keys)
            keys = keys[: int(sel["sample"])]
        else:
            keys = list(sel)
        if not keys or len(keys) > LIMITS["gen_calls"]:
            raise ValueError(f"1..{LIMITS['gen_calls']} calls per pass")
        bad = [k for k in keys if k not in self.calls]
        if bad:
            raise ValueError(f"not search-set calls: {bad[:3]}")
        rendering = req.get("rendering", "served")
        self.call(keys[0], rendering)  # validates
        bcache, bfile = (self.base_cache, "base_gen.jsonl") if rendering == "served" else (self.base_cache_raw, "base_gen_raw.jsonl")
        result = {"n_calls": len(keys), "draws": draws, "rendering": rendering, "arms": {}}
        scored = {}
        for arm in arms:
            originals = self.apply(run_dir) if arm == "edit" else {}
            if arm == "edit" and not originals:
                raise ValueError("arm 'edit' but the run has no saved edit")
            try:
                gens = []
                for k in keys:
                    if arm == "base" and all((k, d) in bcache for d in range(draws)):
                        gens += [bcache[(k, d)] for d in range(draws)]
                        continue
                    g = self.generate(self.call(k, rendering), draws)
                    gens += g
                    if arm == "base":
                        with (CACHE / bfile).open("a") as f:
                            for x in g:
                                f.write(json.dumps(x) + "\n")
                                bcache[(x["key"], x["draw"])] = x
            finally:
                self.restore(originals)
            rows = self.score(gens, work, arm)
            for x, g in zip(rows, gens):
                assert (x["key"], x["draw"]) == (g["key"], g["draw"])
                x["escape_artefact"] = self.artefact(g["key"], g["gen"])
            scored[arm] = {(x["key"], x["draw"]): x for x in rows}
            n = len(rows)
            acc = [x for x in rows if x["accepted"]]
            strata = {}
            for x in rows:
                st = x["recorded_kind"] + ("/" + str(x["recorded_strategy"]) if x["recorded_kind"] == "ok" else "")
                s = strata.setdefault(st, {"n": 0, "accepted": 0, "not_found": 0, "escape_artefact": 0})
                s["n"] += 1
                s["escape_artefact"] += x["escape_artefact"]
                s["accepted"] += x["accepted"]
                s["not_found"] += int(x["outcome"] == "not_found")
            result["arms"][arm] = {
                "n": n, "accepted": sum(x["accepted"] for x in rows) / n,
                "not_found": sum(x["outcome"] == "not_found" for x in rows) / n,
                "multiple": sum(x["outcome"] == "multiple" for x in rows) / n,
                "exact_copy": sum(x["exact_copy"] for x in rows) / n,
                "escape_artefact": sum(x["escape_artefact"] for x in rows) / n,
                "unclosed": sum(1 - x["closed"] for x in rows) / n,
                "compile_fail_among_accepted": (sum(1 for x in acc if x["compiles"] == 0) / len(acc)) if acc else None,
                "by_recorded_stratum": strata,
            }
            with (work / "rows.jsonl").open("a") as f:
                for x, g in zip(rows, gens):
                    f.write(json.dumps({"arm": arm, **x, "gen": g["gen"]}) + "\n")
        if set(arms) == {"base", "edit"}:
            t = {"both_accepted": 0, "only_base": 0, "only_edit": 0, "neither": 0}
            for k, b in scored["base"].items():
                e = scored["edit"][k]
                t["both_accepted" if b["accepted"] and e["accepted"] else "only_base" if b["accepted"] else
                  "only_edit" if e["accepted"] else "neither"] += 1
            result["paired_accepted"] = t
            ta = {"both": 0, "only_base": 0, "only_edit": 0, "neither": 0}
            for k, b in scored["base"].items():
                e = scored["edit"][k]
                ta["both" if b["escape_artefact"] and e["escape_artefact"] else "only_base" if b["escape_artefact"] else
                   "only_edit" if e["escape_artefact"] else "neither"] += 1
            result["paired_escape_artefact"] = ta
        if fullcall and req.get("fullcall", True):
            result["fullcall"] = self.do_fullcall(keys, draws, arms, rendering, run_dir, work)
        return result

    # ---- teacher-forced read -----------------------------------------------------------------------------
    def forward_capture(self, ids: list[int], positions: list[int], layers: set[int], capture: set[str], attn_layers: set[int]):
        import numpy as np
        mx = self.mx
        from common import attn_forward, text_model
        from mlx_lm.models.gemma4_text import geglu
        import mlx.nn as nn
        tm = text_model(self.model)
        x_ids = mx.array(ids)[None]
        emb = tm.embed_tokens(x_ids)
        h = emb * tm.embed_scale
        pli_all = tm._project_per_layer_inputs(h, tm._get_per_layer_inputs(x_ids, emb)) if tm.hidden_size_per_layer_input else None
        prev = list(tm.previous_kvs)
        kvs = {}
        pos = mx.array(positions)
        out = {"resid": [], "mlp": [], "attn": []}
        W = tm.window_size
        for i, layer in enumerate(tm.layers):
            pli = pli_all[:, :, i, :] if pli_all is not None else None
            shared = kvs.get(prev[i]) if prev[i] != i else None
            residual = h
            xa = layer.input_layernorm(h)
            a_out, kv = attn_forward(layer.self_attn, xa, shared, W, 2048)
            if i in attn_layers and "attn" in capture:
                att = layer.self_attn
                q = att.q_norm(att.q_proj(xa).reshape(1, len(ids), att.n_heads, att.head_dim)).transpose(0, 2, 1, 3)
                q = att.rope(q, offset=0)[:, :, pos, :]
                k = kv[0]
                rep = att.n_heads // k.shape[1]
                k = mx.repeat(k, rep, axis=1)
                s = (q.astype(mx.float32) @ k.astype(mx.float32).transpose(0, 1, 3, 2)) * att.scale
                j = mx.arange(len(ids))[None, :]
                p_ = pos[:, None]
                allowed = (j <= p_) & ((j > p_ - W) if att.is_sliding else (j >= 0))
                s = mx.where(allowed[None, None], s, -1e30)
                out["attn"].append(np.array(mx.softmax(s, axis=-1)[0].astype(mx.float16)))
            if prev[i] == i:
                kvs[i] = kv
            h = residual + layer.post_attention_layernorm(a_out)
            residual = h
            xm = layer.pre_feedforward_layernorm(h)
            act = geglu(layer.mlp.gate_proj(xm), layer.mlp.up_proj(xm))
            if i in layers and "mlp" in capture:
                out["mlp"].append(np.array(act[0, pos].astype(mx.float16)))
            h = residual + layer.post_feedforward_layernorm(layer.mlp.down_proj(act))
            if layer.per_layer_input_gate is not None and pli is not None:
                residual = h
                g = nn.gelu_approx(layer.per_layer_input_gate(h)) * pli
                h = residual + layer.post_per_layer_input_norm(layer.per_layer_projection(g))
            if layer.layer_scalar is not None:
                h = h * layer.layer_scalar
            if i in layers and "resid" in capture:
                out["resid"].append(np.array(h[0, pos].astype(mx.float32)))
            mx.eval(h)
        logits = None
        if "logprobs" in capture:
            hn = tm.norm(h[0, pos])
            lg = tm.embed_tokens.as_linear(hn).astype(mx.float32)
            sc = self.model.language_model.final_logit_softcapping
            if sc:
                lg = mx.tanh(lg / sc) * sc
            logits = lg - mx.logsumexp(lg, axis=-1, keepdims=True)
        return out, logits

    def native_logprobs(self, ids: list[int], positions: list[int]):
        """v2: log-probs at contiguous `positions` from mlx_lm's own forward with a KV cache (the generation path)."""
        mx = self.mx
        from mlx_lm.models.cache import make_prompt_cache
        p0, n = positions[0], len(positions)
        assert positions == list(range(p0, p0 + n))
        cache = make_prompt_cache(self.model)
        for s in range(0, p0, 2048):
            self.model(mx.array(ids[s:min(s + 2048, p0)])[None], cache=cache)
            mx.eval([c.state for c in cache])
        lg = self.model(mx.array(ids[p0:p0 + n])[None], cache=cache)[0].astype(mx.float32)
        return lg - mx.logsumexp(lg, axis=-1, keepdims=True)

    def do_read(self, req: dict, run_dir: Path, work: Path) -> dict:
        import numpy as np
        mx = self.mx
        texts = req.get("texts", [])
        layers = set(int(l) for l in req.get("layers", []))
        capture = set(req.get("capture", ["resid", "logprobs"]))
        attn_layers = set(int(l) for l in req.get("attn_layers", [])) if "attn" in capture else set()
        if not 1 <= len(texts) <= LIMITS["read_texts"] or len(layers) > LIMITS["read_layers"] or len(attn_layers) > LIMITS["attn_layers"]:
            raise ValueError(f"limits: texts 1..{LIMITS['read_texts']}, layers <= {LIMITS['read_layers']}, attn_layers <= {LIMITS['attn_layers']}")
        arm = req.get("arm", "base")
        rendering = req.get("rendering", "served")
        originals = self.apply(run_dir) if arm == "edit" else {}
        summaries = []
        try:
            for ti, t in enumerate(texts):
                if t.get("key") not in self.calls:
                    raise ValueError(f"text {ti}: key must be a search-set call")
                r = self.call(t["key"], rendering)
                cont = r["old_string"] if t.get("continuation", "recorded") == "recorded" else str(t["continuation"])
                p_ids = self.tok.encode(r["prompt"] + r["prefix"], add_special_tokens=False).ids
                c_ids = self.tok.encode(cont, add_special_tokens=False).ids[: LIMITS["continuation_tokens"]]
                ids = p_ids + c_ids
                if req.get("positions", "continuation") == "prompt_last":
                    positions = [len(p_ids) - 1]
                else:  # the positions that predict each continuation token (and the closing quote)
                    positions = list(range(len(p_ids) - 1, len(ids)))[: LIMITS["read_positions"]]
                caps, logp_manual = self.forward_capture(ids, positions, layers, capture, attn_layers)
                logp = self.native_logprobs(ids, positions) if logp_manual is not None else None
                arrays = {"positions": np.array(positions), "token_ids": np.array(ids[min(positions):max(positions) + 2])}
                for k in ("resid", "mlp", "attn"):
                    if caps[k]:
                        arrays[k] = np.stack(caps[k])
                summary = {"key": r["key"], "recorded_kind": r["error_kind"], "prompt_tokens": len(p_ids),
                           "continuation_tokens": len(c_ids), "n_positions": len(positions)}
                if logp is not None:
                    targets = (c_ids + [self.q_id])[: len(positions)]
                    lp = logp[mx.arange(len(targets)), mx.array(targets)]
                    top = mx.argsort(-logp, axis=-1)[:, :5]
                    arrays["logprobs"] = np.array(lp)  # v2: native forward
                    arrays["logprobs_manual"] = np.array(logp_manual[mx.arange(len(targets)), mx.array(targets)])
                    arrays["top5_ids"] = np.array(top)
                    arrays["top5_logprobs"] = np.array(mx.take_along_axis(logp, top, axis=-1))
                    summary["sum_logprob"] = float(np.array(lp).sum())
                    summary["min_logprob"] = float(np.array(lp).min())
                    summary["first_below_-2"] = int(np.argmax(np.array(lp) < -2)) if (np.array(lp) < -2).any() else None
                np.savez_compressed(work / f"text_{ti}.npz", **arrays)
                summaries.append(summary)
        finally:
            self.restore(originals)
        return {"arm": arm, "rendering": rendering, "layers": sorted(layers), "attn_layers": sorted(attn_layers), "capture": sorted(capture),
                "texts": summaries, "files": f"work/{work.name}/text_<i>.npz"}

    # ---- loop --------------------------------------------------------------------------------------------
    def handle(self, req_path: Path) -> None:
        run_dir = req_path.parent.parent
        state_p = run_dir / "state.json"
        claimed = req_path.with_suffix(".json.claimed")
        req_path.rename(claimed)
        done = req_path.with_name(req_path.stem + ".done.json")
        state = json.loads(state_p.read_text())
        t0 = time.time()
        rec = {"request": req_path.name, "received_utc": now()}
        try:
            req = json.loads(claimed.read_text())
            # v2: record (and optionally pin) the exact edit files applied by this pass
            edit_snap = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted((run_dir / "edit").glob("*.npz"))}
            if req.get("expect_edit") is not None and req["expect_edit"] != edit_snap:
                raise ValueError("expect_edit does not match the run's current edit/ (pinned snapshot mismatch)")
            rec["edit_snapshot"] = edit_snap
            if state["used"] >= state["budget"]:
                rec.update(status="refused", reason=f"budget exhausted ({state['used']}/{state['budget']})")
            else:
                work = run_dir / "work" / req_path.stem
                work.mkdir(parents=True, exist_ok=True)
                if req.get("type") == "gen":
                    res = self.do_gen(req, run_dir, work, fullcall=bool(state.get("fullcall")))
                elif req.get("type") == "read":
                    res = self.do_read(req, run_dir, work)
                else:
                    raise ValueError("type must be gen or read")
                state["used"] += 1
                state.setdefault("passes", []).append({"request": req_path.name, "type": req["type"], "utc": now(),
                                                       "secs": round(time.time() - t0, 1),
                                                       "start_utc": rec["received_utc"], "edit_snapshot": edit_snap})
                state_p.write_text(json.dumps(state, indent=1))
                rec.update(status="ok", result=res)
        except Exception as e:  # noqa: BLE001
            rec.update(status="failed (not counted)", error=f"{type(e).__name__}: {e}", trace=traceback.format_exc()[-1500:])
        rec.update(secs=round(time.time() - t0, 1), passes_used=state["used"], budget=state["budget"])
        done.write_text(json.dumps(rec, indent=1))
        print(f"[{now()}] {run_dir.name}/{req_path.name}: {rec['status']} ({rec['secs']}s, {state['used']}/{state['budget']})", flush=True)

    def loop(self, once: bool = False) -> None:
        while True:
            pending = sorted((p for p in PROPOSERS.glob("*/requests/fwd_*.json") if not p.name.endswith(".done.json")),
                             key=lambda p: (p.parent.parent.name, int(p.stem.split("_")[1])))  # numeric k (v2 fix)
            for p in pending:
                self.handle(p)
            if once:
                return
            time.sleep(2)


def init_run(a) -> None:
    run = PROPOSERS / a.run_id
    if run.exists():
        sys.exit(f"{run} exists")
    for d in ("requests", "edit", "work"):
        (run / d).mkdir(parents=True)
    state = {"run_id": a.run_id, "proposer": a.proposer, "model": a.model, "effort": a.effort, "budget": a.budget,
             "used": 0, "created_utc": now(), "limits": LIMITS, "split_sha256": hashlib.sha256(SPLIT.read_bytes()).hexdigest(),
             "harness_version": HARNESS_VERSION, "fullcall": bool(getattr(a, "fullcall", False)),
             "search_calls_sha256": hashlib.sha256((SEARCH_DIR / "calls.jsonl").read_bytes()).hexdigest()}
    (run / "state.json").write_text(json.dumps(state, indent=1))
    (run / "notebook.md").write_text(f"# Notebook of run {a.run_id}\n\n")
    print(run)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    s = sub.add_parser("serve")
    s.add_argument("--once", action="store_true")
    i = sub.add_parser("init")
    i.add_argument("run_id")
    i.add_argument("--proposer", required=True)
    i.add_argument("--model", required=True)
    i.add_argument("--effort", required=True)
    i.add_argument("--budget", type=int, default=10)
    i.add_argument("--fullcall", action="store_true", help="v2c: gen requests also run the full-call replay")
    a = ap.parse_args()
    if a.cmd == "init":
        init_run(a)
    else:
        Server().loop(once=getattr(a, "once", False))


if __name__ == "__main__":
    main()
