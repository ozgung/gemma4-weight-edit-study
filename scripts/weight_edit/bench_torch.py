"""Benchmark + validation of the PyTorch port (wsp_torch.py) on a GPU VM.  Owner: Claude.

    python -I bench_torch.py MODEL_DIR CALLS.jsonl KEYS.txt OUT_DIR [--draws 2] [--read 8]

For each key (in file order): regenerate old_string (greedy + sampled, seeds 7919*index+draw as replay_gen.py), time
prefill and generation; write OUT_DIR/gen_torch.jsonl in replay_gen's format (score it on the Mac with
replay_score.py against the same replay dir). For the first --read keys: native log-probs of the recorded
old_string (+ closing quote) at the continuation positions -> OUT_DIR/logprobs_torch.npz (compare with MLX on the Mac).
Also a capture smoke (resid + mlp at 2 layers on one key) and an apply/restore round trip of a random rank-1 edit.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wsp_torch import TorchBackend  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("calls", type=Path)
    ap.add_argument("keys", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--draws", type=int, default=2)
    ap.add_argument("--read", type=int, default=8)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    calls = {r["key"]: r for r in map(json.loads, a.calls.open())}
    keys = [k.strip() for k in a.keys.read_text().splitlines() if k.strip()]
    be = TorchBackend(a.model_dir)
    info = {"gpu": torch.cuda.get_device_name(0), "load_s": round(be.load_s, 1), "torch": torch.__version__}
    import transformers
    info["transformers"] = transformers.__version__
    print(json.dumps(info), flush=True)
    t_all = time.time()
    with (a.out / "gen_torch.jsonl").open("w") as f:
        for ci, k in enumerate(keys):
            r = calls[k]
            ids = be.tok.encode(r["prompt"] + r["prefix"], add_special_tokens=False).ids
            ref = len(be.tok.encode(r["old_string"], add_special_tokens=False).ids)
            outs = be.generate(ids, min(1024, 2 * ref + 64), a.draws, [7919 * ci + d for d in range(a.draws)])
            for o in outs:
                f.write(json.dumps({"key": k, "prompt_tokens": len(ids), **o, "edit": None, "backend": "torch"}) + "\n")
            f.flush()
            print(f"[{ci + 1}/{len(keys)}] {k} prompt {len(ids)} prefill {outs[0]['prefill_s']}s gen "
                  f"{[o['gen_s'] for o in outs]} n_gen {[o['n_gen'] for o in outs]}", flush=True)
    info["gen_total_s"] = round(time.time() - t_all, 1)
    # teacher-forced log-probs of the recorded old_string
    lps = {}
    t = time.time()
    for k in keys[: a.read]:
        r = calls[k]
        p_ids = be.tok.encode(r["prompt"] + r["prefix"], add_special_tokens=False).ids
        c_ids = be.tok.encode(r["old_string"], add_special_tokens=False).ids[:512]
        ids = p_ids + c_ids
        positions = list(range(len(p_ids) - 1, len(ids)))[:256]
        targets = (c_ids + [be.q_id])[: len(positions)]
        lp = be.native_logprobs(ids, positions)
        lps[k] = lp[np.arange(len(targets)), targets]
    info["read_s_per_text"] = round((time.time() - t) / max(1, len(lps)), 2)
    np.savez(a.out / "logprobs_torch.npz", keys=np.array(list(lps)), **{f"lp_{i}": v for i, v in enumerate(lps.values())})
    # capture smoke
    r = calls[keys[0]]
    ids = be.tok.encode(r["prompt"] + r["prefix"] + r["old_string"], add_special_tokens=False).ids
    t = time.time()
    cap = be.capture(ids, list(range(len(ids) - 32, len(ids))), {10, 30}, {"resid", "mlp"})
    info["capture_s"] = round(time.time() - t, 2)
    info["capture_shapes"] = {k: list(v.shape) for k, v in cap.items()}
    # apply / restore round trip
    import tempfile
    name = "model.language_model.layers.12.mlp.down_proj.weight"
    p = be._param(name)
    w0 = p.data.clone()
    with tempfile.TemporaryDirectory() as td:
        rng = np.random.default_rng(0)
        np.savez(Path(td) / "e.npz", module=np.array(name), B=rng.standard_normal((p.shape[0], 1)).astype(np.float32) * 0.02,
                 A=rng.standard_normal((1, p.shape[1])).astype(np.float32) * 0.02)
        orig = be.apply(Path(td))
        info["edit_changed"] = bool((p.data != w0).any())
        be.restore(orig)
        info["edit_restored"] = bool(torch.equal(p.data, w0))
    info["max_mem_gb"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
    (a.out / "bench_info.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info), flush=True)


if __name__ == "__main__":
    main()
