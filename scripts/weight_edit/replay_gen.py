"""Regenerate old_string from replayed discovery prefixes with local E4B (bf16, MLX).  Owner: Claude.

    outputs/small_model/mlx_venv/bin/python -I scripts/weight_edit/replay_gen.py \\
        --set fidelity --sample 120 --draws 3 --out outputs/weight_edit/claude/replay/gen_fidelity_base.jsonl
    (then score with the harness venv: scripts/weight_edit/replay_score.py)

Input: outputs/weight_edit/claude/replay/calls.jsonl (replay_set.py). Usable calls only: verified prompt,
certain file state, file present. The prompt is `prompt + prefix` (the served prompt plus the model's own
generated text up to `old_string:<|"|>`); generation stops at the closing <|"|> token.
Draw 0 is greedy, draws 1.. are sampled at the served settings (T 0.6, top_p 0.95, top_k 64), each with its own
seed. The prompt is prefilled once per call and the KV cache is copied for each draw.
--edit DIR applies a LoRA-style edit (DIR/*.npz with B [out,r], A [r,in], module name) to the weights first.
--sample N: a fixed stratified sample (seed 20261102) over error_kind/strategy strata, task-grouped.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "data/models/gemma-4-e4b-it-qat-q4_0-unquantized"
REPLAY = ROOT / "outputs/weight_edit/claude/replay"
Q = '<|"|>'
STRATA = {"ok_exact": 0.35, "ok_unescaped": 0.15, "not_found": 0.40, "multiple": 0.10}


def stratum(r: dict) -> str | None:
    if r["error_kind"] == "ok":
        return "ok_exact" if r["strategy"] == "exact" else ("ok_unescaped" if r["strategy"] == "unescaped" else None)
    return r["error_kind"] if r["error_kind"] in ("not_found", "multiple") else None


def usable(path: Path = REPLAY / "calls.jsonl") -> list[dict]:
    out = []
    for line in path.open():
        r = json.loads(line)
        if r["verified"] and not r["state_uncertain"] and not r["file_missing"] and r.get("old_string"):
            out.append(r)
    return out


def sample(rows: list[dict], n: int, seed: int = 20261102) -> list[dict]:
    rng = random.Random(seed)
    by: dict[str, list] = {}
    for r in rows:
        s = stratum(r)
        if s:
            by.setdefault(s, []).append(r)
    picked = []
    for s, frac in STRATA.items():
        pool = sorted(by.get(s, []), key=lambda r: r["key"])
        rng.shuffle(pool)
        k = round(n * frac)
        # at most 2 calls per task per stratum, so a few long traces do not dominate
        per_task: dict[str, int] = {}
        for r in pool:
            if len([p for p in picked if stratum(p) == s]) >= k:
                break
            if per_task.get(r["instance_id"], 0) >= 2:
                continue
            per_task[r["instance_id"]] = per_task.get(r["instance_id"], 0) + 1
            picked.append(r)
    return picked


def apply_edit(model, edit_dir: Path) -> list[str]:
    """W += B @ A for each module in edit_dir (npz files with keys module, B, A). Returns the modules changed."""
    import mlx.core as mx
    import numpy as np
    from mlx.utils import tree_flatten

    params = dict(tree_flatten(model.parameters()))
    changed = []
    for f in sorted(edit_dir.glob("*.npz")):
        z = np.load(f, allow_pickle=False)
        name = str(z["module"])
        key = "language_model.model." + name.removeprefix("model.language_model.").removesuffix(".weight") + ".weight"
        W = params[key]
        delta = mx.array(z["B"].astype(np.float32)) @ mx.array(z["A"].astype(np.float32))
        assert delta.shape == W.shape, (name, delta.shape, W.shape)
        new = (W.astype(mx.float32) + delta).astype(W.dtype)
        model.update(_nest({key: new}))
        changed.append(key)
    mx.eval(model.parameters())
    return changed


def _nest(flat: dict) -> dict:
    from mlx.utils import tree_unflatten
    return tree_unflatten(list(flat.items()))


def main() -> None:
    import mlx.core as mx
    from mlx_lm.generate import generate_step
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.sample_utils import make_sampler
    from mlx_lm.utils import load_model
    from tokenizers import Tokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=120)
    ap.add_argument("--keys", type=Path, help="file with one replay key per line (overrides --sample)")
    ap.add_argument("--draws", type=int, default=3, help="1 greedy + (draws-1) sampled")
    ap.add_argument("--edit", type=Path)
    ap.add_argument("--max-new-cap", type=int, default=1024)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--calls", type=Path, default=REPLAY / "calls.jsonl",
                    help="replay calls file (WP12: outputs/weight_edit/claude/wp12/calls_raw.jsonl)")
    a = ap.parse_args()
    rows = usable(a.calls)
    if a.keys:
        want = [l.strip() for l in a.keys.read_text().splitlines() if l.strip()]
        idx = {r["key"]: r for r in rows}
        rows = [idx[k] for k in want if k in idx]
    else:
        rows = sample(rows, a.sample)
    tok = Tokenizer.from_file(str(MODEL / "tokenizer.json"))
    q_id = tok.token_to_id(Q)
    t0 = time.time()
    model, _ = load_model(MODEL, lazy=False, strict=True)
    changed = apply_edit(model, a.edit) if a.edit else []
    print(f"model loaded {time.time() - t0:.0f}s; edit modules {len(changed)}; calls {len(rows)}", flush=True)
    done = set()
    if a.out.exists():
        done = {(json.loads(l)["key"], json.loads(l)["draw"]) for l in a.out.open()}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("a") as out:
        for ci, r in enumerate(rows):
            if all((r["key"], d) in done for d in range(a.draws)):
                continue
            ids = tok.encode(r["prompt"] + r["prefix"], add_special_tokens=False).ids
            ref = len(tok.encode(r["old_string"], add_special_tokens=False).ids)
            max_new = min(a.max_new_cap, 2 * ref + 64)
            t1 = time.time()
            cache = make_prompt_cache(model)
            # prefill all but the last prompt token once
            for start in range(0, len(ids) - 1, 2048):
                model(mx.array(ids[start:min(start + 2048, len(ids) - 1)])[None], cache=cache)
                mx.eval([c.state for c in cache])
            t_prefill = time.time() - t1
            for d in range(a.draws):
                if (r["key"], d) in done:
                    continue
                sampler = make_sampler(temp=0.0) if d == 0 else make_sampler(temp=0.6, top_p=0.95, top_k=64)
                mx.random.seed(7919 * ci + d)
                c2 = copy.deepcopy(cache)
                t2 = time.time()
                gen, closed = [], 0
                for tid, _ in generate_step(mx.array(ids[-1:]), model, max_tokens=max_new, sampler=sampler, prompt_cache=c2):
                    tid = int(tid)
                    if tid == q_id:
                        closed = 1
                        break
                    gen.append(tid)
                text = tok.decode(gen, skip_special_tokens=False)
                out.write(json.dumps({"key": r["key"], "draw": d, "gen": text, "closed": closed, "n_gen": len(gen),
                                      "prompt_tokens": len(ids), "prefill_s": round(t_prefill, 1),
                                      "gen_s": round(time.time() - t2, 1), "edit": str(a.edit) if a.edit else None}) + "\n")
                out.flush()
            print(f"[{ci + 1}/{len(rows)}] {r['key']} {r['error_kind']}/{r['strategy']} prompt {len(ids)} "
                  f"prefill {t_prefill:.1f}s", flush=True)
    print(f"done {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
