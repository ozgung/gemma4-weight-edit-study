"""Convert an MLX LoRA adapter (scripts/mlx_lora/train.py) into the gate's edit format.  Owner: Claude.

    outputs/small_model/mlx_venv/bin/python -I scripts/weight_edit/lora_to_edit.py ADAPTER.safetensors RANK ALPHA OUT_DIR

MLX keys language_model.model.layers.<i>.<part>.<name>.lora_a [in, r] and .lora_b [r, out]. Output per module:
OUT_DIR/edit/layers_<i>_<part>_<name>.npz with module = model.language_model.layers.<i>.<part>.<name>.weight,
B [out, r] = (alpha / rank) * lora_b^T and A [r, in] = lora_a^T, so W' = W + B A (replay_gen.apply_edit).
Prints the number of modules and the Frobenius norm of each delta (and the total).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np


def main() -> None:
    src, rank, alpha, out = Path(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), Path(sys.argv[4])
    w = {k: np.array(v.astype(mx.float32)) for k, v in mx.load(str(src)).items()}
    scale = alpha / rank
    mods = sorted({k.rsplit(".", 1)[0] for k in w})
    (out / "edit").mkdir(parents=True, exist_ok=True)
    norms = {}
    for m in mods:
        a, b = w[m + ".lora_a"], w[m + ".lora_b"]
        B, A = (scale * b.T).astype(np.float32), a.T.astype(np.float32)
        tail = m.removeprefix("language_model.model.")
        name = "model.language_model." + tail + ".weight"
        np.savez(out / "edit" / (tail.replace(".", "_") + ".npz"), module=np.array(name), B=B, A=A,
                 meta=np.array(json.dumps({"src": str(src), "rank": rank, "alpha": alpha})))
        norms[tail] = float(np.linalg.norm(B @ A))
    tot = float(np.sqrt(sum(v * v for v in norms.values())))
    (out / "edit_norms.json").write_text(json.dumps({"modules": len(mods), "total_fro": tot, "per_module": norms}, indent=1))
    print(f"modules {len(mods)} total_fro {tot:.4f} max_module {max(norms.values()):.4f}")


if __name__ == "__main__":
    main()
