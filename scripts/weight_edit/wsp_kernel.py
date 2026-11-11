"""Weight kernel for WP11 proposers: read E4B tensors, save LoRA-expressible edits.  Owner: Claude.

Use from Python started with the MLX venv (outputs/small_model/mlx_venv/bin/python -I):

    import sys; sys.path.insert(0, "scripts/weight_edit")
    import wsp_kernel as K
    K.names(r"layers\\.15\\..*mlp")          # tensor names matching a regex
    W = K.load("model.language_model.layers.15.mlp.down_proj.weight")   # numpy float32, [out, in]
    K.save_edit(RUN_DIR, "model.language_model.layers.15.mlp.down_proj.weight", B, A, note="why")
    K.list_edits(RUN_DIR); K.remove_edit(RUN_DIR, name)

An edit on a module is W' = W + B @ A with B [out, r] and A [r, in], r <= 128 (one entry per module;
saving again replaces it). Allowed modules: the language-model linears that the competition's vLLM LoRA path
serves (q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj of the 42 decoder layers; layers 24-41
have no k_proj/v_proj): 258 modules. Refused: embeddings, per-layer embeddings and projections, norms,
layer scalars, lm_head, vision/audio towers, and any B @ A that is a multiple of W (whole-tensor rescale).
Rules (from the task brief): this module reads weights only. Do not run the model here; forward passes go
through the forward-pass server (requests/fwd_<k>.json). No gradients or training of any kind.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MODEL_FILE = ROOT / "data/models/gemma-4-e4b-it-qat-q4_0-unquantized/model.safetensors"
ALLOWED = re.compile(r"^model\.language_model\.layers\.(\d+)\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)\.weight$")
MAX_RANK = 128
_W = None


def _weights():
    global _W
    if _W is None:
        import mlx.core as mx
        _W = mx.load(str(MODEL_FILE))  # lazy: tensors are read when used
    return _W


def names(pattern: str = ".*") -> list[str]:
    rx = re.compile(pattern)
    return sorted(k for k in _weights() if rx.search(k))


def shape(name: str) -> tuple[int, ...]:
    return tuple(_weights()[name].shape)


def load(name: str) -> np.ndarray:
    import mlx.core as mx
    a = _weights()[name].astype(mx.float32)
    mx.eval(a)
    return np.array(a)


def allowed(name: str) -> bool:
    return bool(ALLOWED.match(name)) and name in _weights()


def _edit_dir(run_dir: str | Path) -> Path:
    d = Path(run_dir) / "edit"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _fname(name: str) -> str:
    return name.removeprefix("model.language_model.").removesuffix(".weight").replace(".", "_") + ".npz"


def save_edit(run_dir: str | Path, name: str, B: np.ndarray, A: np.ndarray, note: str = "") -> dict:
    if not allowed(name):
        raise ValueError(f"module not editable: {name}")
    out, inn = shape(name)
    B = np.asarray(B, dtype=np.float32)
    A = np.asarray(A, dtype=np.float32)
    if B.ndim != 2 or A.ndim != 2 or B.shape[0] != out or A.shape[1] != inn or B.shape[1] != A.shape[0]:
        raise ValueError(f"shapes: need B [{out}, r] and A [r, {inn}], got {B.shape} and {A.shape}")
    r = B.shape[1]
    if not 1 <= r <= MAX_RANK:
        raise ValueError(f"rank {r} not in 1..{MAX_RANK}")
    if not (np.isfinite(B).all() and np.isfinite(A).all()):
        raise ValueError("non-finite values")
    W = load(name)
    D = B @ A
    wn = float(np.linalg.norm(W))
    c = float((W * D).sum() / (wn ** 2))
    if np.linalg.norm(D - c * W) <= 1e-4 * max(1e-12, np.linalg.norm(D)):
        raise ValueError("refused: the edit is a multiple of the whole tensor (rescale)")
    meta = {"module": name, "rank": r, "delta_fro": float(np.linalg.norm(D)), "weight_fro": wn,
            "rel_change": float(np.linalg.norm(D) / wn), "note": note, "saved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    f = _edit_dir(run_dir) / _fname(name)
    np.savez(f, module=np.array(name), B=B, A=A, meta=np.array(json.dumps(meta)))
    _log(run_dir, {"op": "save_edit", **meta})
    return meta


def remove_edit(run_dir: str | Path, name: str) -> bool:
    f = _edit_dir(run_dir) / _fname(name)
    if f.exists():
        f.unlink()
        _log(run_dir, {"op": "remove_edit", "module": name, "removed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        return True
    return False


def list_edits(run_dir: str | Path) -> list[dict]:
    out = []
    for f in sorted(_edit_dir(run_dir).glob("*.npz")):
        z = np.load(f, allow_pickle=False)
        out.append(json.loads(str(z["meta"])))
    return out


def _log(run_dir, rec: dict) -> None:
    with (Path(run_dir) / "kernel_log.jsonl").open("a") as f:
        f.write(json.dumps(rec) + "\n")
