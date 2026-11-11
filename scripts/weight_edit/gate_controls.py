"""Matched random controls for a frozen edit (WP7 gate, pre-registered).  Owner: Claude.

    outputs/small_model/mlx_venv/bin/python -I scripts/weight_edit/gate_controls.py SRC_RUN_DIR OUT_DIR --seed S

For every module of the source edit (B [out, r], A [r, in]): random Gaussian B', A' of the same shape, scaled
so that ||B' A'||_F equals the source module's ||B A||_F. Same modules, same rank, same per-module norm.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--support-matched", action="store_true",
                    help="restrict the random B to the rows where the source B is non-zero (sensitivity control)")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    out = Path(a.out) / "edit"
    out.mkdir(parents=True, exist_ok=True)
    rec = []
    for f in sorted((Path(a.src) / "edit").glob("*.npz")):
        z = np.load(f, allow_pickle=False)
        B, A = z["B"].astype(np.float32), z["A"].astype(np.float32)
        target = float(np.linalg.norm(B @ A))
        Br = rng.standard_normal(B.shape).astype(np.float32)
        if a.support_matched:
            Br[np.abs(B).sum(1) == 0] = 0.0
        Ar = rng.standard_normal(A.shape).astype(np.float32)
        k = np.sqrt(target / float(np.linalg.norm(Br @ Ar)))
        Br *= k
        Ar *= k
        got = float(np.linalg.norm(Br @ Ar))
        meta = {"module": str(z["module"]), "rank": int(B.shape[1]), "target_fro": target, "control_fro": got,
                "seed": a.seed, "source": str(f), "support_matched": bool(a.support_matched),
                "nonzero_rows": int((np.abs(Br).sum(1) > 0).sum())}
        np.savez(out / f.name, module=z["module"], B=Br, A=Ar, meta=np.array(json.dumps(meta)))
        rec.append(meta)
    (Path(a.out) / "control.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec))


if __name__ == "__main__":
    main()
