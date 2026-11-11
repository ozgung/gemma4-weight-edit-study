"""Residual-stream features for the retry go/no-go (MLX, bf16, base model only).  Owner: Claude.

    outputs/small_model/mlx_venv/bin/python -I scripts/weight_edit/retry_gonogo_probe.py

For each key in keys_{heldout,train}_{retry,first}.txt: one forward pass over prompt + prefix + the first 12 tokens of the
recorded old_string, residual stream after layers 35, 38, 40, 41 at three positions:
  turn  = last token of the prompt (start of the model's turn)
  open  = last token of the prefix (the position that predicts the first old_string token)
  mid   = 8 tokens into old_string (or its last available token)
Resumable: one npz per set. Output: features_<set>.npz with keys, X [N, layers, positions, 2560].
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/weight_edit"))
from wsp_server import Server  # noqa: E402

R = ROOT / "outputs/weight_edit/claude"
OUT = R / "retry_gonogo"
LAYERS = {35, 38, 40, 41}


def main() -> None:
    calls = {}
    for p in (R / "replay_ood/calls.jsonl", R / "replay/calls.jsonl"):
        for line in p.open():
            r = json.loads(line)
            calls[r["key"]] = r
    S = Server()
    S.mx.set_cache_limit(4 << 30)
    t0 = time.time()
    for name in ("heldout_retry", "heldout_first", "train_retry", "train_first"):
        out = OUT / f"features_{name}.npz"
        if out.exists():
            continue
        keys = [l.strip() for l in (OUT / f"keys_{name}.txt").read_text().splitlines() if l.strip()]
        xs = []
        for i, k in enumerate(keys):
            r = calls[k]
            n_prompt = len(S.tok.encode(r["prompt"], add_special_tokens=False).ids)
            p_ids = S.tok.encode(r["prompt"] + r["prefix"], add_special_tokens=False).ids
            c_ids = S.tok.encode(r["old_string"], add_special_tokens=False).ids[:12]
            ids = p_ids + c_ids
            pos = [n_prompt - 1, len(p_ids) - 1, len(p_ids) - 1 + min(8, len(c_ids))]
            caps, _ = S.forward_capture(ids, pos, LAYERS, {"resid"}, set())
            xs.append(np.stack(caps["resid"]).astype(np.float32))
            S.mx.clear_cache()  # amendment 2026-11-07 21:3x: Metal buffer cache reached about 36 GB; no logic change
            if i % 20 == 0:
                print(f"{name} {i + 1}/{len(keys)} {time.time() - t0:.0f}s", flush=True)
        np.savez_compressed(out, keys=np.array(keys), X=np.stack(xs))
    print("probe done", flush=True)


if __name__ == "__main__":
    main()
