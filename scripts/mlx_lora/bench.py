"""Throughput/peak-memory of one LoRA training micro-step (forward+backward) at given lengths.

Usage: python -I scripts/mlx_lora/bench.py --lengths 8192,16384 --loss-frac 0.25 --reps 2
Optionally estimate one epoch of a dataset: --data data/sft/<name> (reads n_tokens / input_ids).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="data/models/gemma-4-e4b-it-qat-q4_0-unquantized")
    p.add_argument("--lengths", default="8192,16384")
    p.add_argument("--loss-frac", type=float, default=0.25)
    p.add_argument("--reps", type=int, default=2)
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--chunk", type=int, default=2048)
    p.add_argument("--ce-chunk", type=int, default=1024)
    p.add_argument("--no-checkpoint", action="store_true")
    p.add_argument("--mode", default="layerwise", choices=["layerwise", "autograd"])
    p.add_argument("--out", default="outputs/small_model/bench/bench.json")
    a = p.parse_args()

    model, cfg, mdir = common.load_lm(a.model)
    common.apply_lora(model, rank=a.rank, alpha=float(a.rank))
    fwd = common.TrainForward(model, checkpoint=not a.no_checkpoint, chunk=a.chunk)
    lw = common.LayerwiseTrainer(model, chunk=a.chunk, ce_chunk=a.ce_chunk)
    base_mem = mx.get_active_memory() / 1e9
    print(f"weights+lora active {base_mem:.1f}GB", flush=True)

    def loss_fn(model, x, pos, tgt):
        return common.masked_nll_sum(model, fwd(x), pos, tgt, chunk=a.ce_chunk)

    vg = nn.value_and_grad(model, loss_fn)
    res = []
    rng = np.random.default_rng(0)
    for L in [int(v) for v in a.lengths.split(",")]:
        for r in range(a.reps + 1):  # first rep = warm-up
            ids = rng.integers(1000, 200000, size=L).astype(np.int32)
            m = np.zeros(L, np.int8)
            # contiguous assistant-like spans
            n_loss = int(a.loss_frac * L)
            span = 512
            starts = rng.choice(np.arange(1, L - span), size=max(1, n_loss // span), replace=False)
            for s0 in starts:
                m[s0:s0 + span] = 1
            pos, tgt = common.loss_positions(ids, m)
            mx.clear_cache()
            mx.reset_peak_memory()
            t = time.time()
            if a.mode == "layerwise":
                l, g = lw.loss_and_grads(mx.array(ids)[None], mx.array(pos), mx.array(tgt))
            else:
                l, g = vg(model, mx.array(ids)[None], mx.array(pos), mx.array(tgt))
            mx.eval(g)
            dtv = time.time() - t
            peak = mx.get_peak_memory() / 1e9
            row = {"L": L, "rep": r, "loss_tokens": int(len(pos)), "sec": dtv, "tok_s": L / dtv, "peak_gb": peak}
            print(json.dumps(row), flush=True)
            if r > 0:
                res.append(row)
            del g, l
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"rows": res, "weights_gb": base_mem, "checkpoint": not a.no_checkpoint, "mode": a.mode, "chunk": a.chunk, "ce_chunk": a.ce_chunk}, indent=1))


if __name__ == "__main__":
    main()
