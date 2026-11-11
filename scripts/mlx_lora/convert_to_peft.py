"""Convert an MLX LoRA adapter (this lane's naming) to a PEFT LoRA directory loadable by
transformers+PEFT and vLLM 0.19.1 on Gemma4ForConditionalGeneration.

MLX:  language_model.model.layers.{i}.{self_attn|mlp}.{proj}.lora_a  [in, r]
      language_model.model.layers.{i}.{self_attn|mlp}.{proj}.lora_b  [r, out]
      forward: y += scale * (x @ lora_a) @ lora_b, scale = alpha / r
PEFT: base_model.model.model.language_model.layers.{i}.{...}.{proj}.lora_A.weight [r, in]
      base_model.model.model.language_model.layers.{i}.{...}.{proj}.lora_B.weight [out, r]
      forward: y += (alpha / r) * lora_B(lora_A(x))

Usage: python -I convert_to_peft.py --adapter adapter.safetensors --rank 32 --alpha 32 --out DIR
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

KEY_RE = re.compile(r"^language_model\.model\.layers\.(\d+)\.(self_attn|mlp)\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)\.lora_(a|b)$")
ORDER = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def load_mlx_adapter(path) -> dict[str, np.ndarray]:
    import mlx.core as mx

    w = mx.load(str(path))
    return {k: np.array(v.astype(mx.float32)) for k, v in w.items()}


def convert(weights: dict[str, np.ndarray], rank: int, alpha: float, out_dir, dtype: str = "bfloat16",
            base_name: str = "gemma-4-e4b-it-qat-w4a16-ct") -> dict:
    import torch
    from safetensors.torch import save_file

    tdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[dtype]
    out = {}
    seen = set()
    for k, v in weights.items():
        m = KEY_RE.match(k)
        if not m:
            raise ValueError(f"unexpected adapter key {k}")
        li, parent, proj, ab = m.groups()
        seen.add(proj)
        if ab == "a":
            assert v.shape[1] == rank, (k, v.shape)
            name = f"base_model.model.model.language_model.layers.{li}.{parent}.{proj}.lora_A.weight"
        else:
            assert v.shape[0] == rank, (k, v.shape)
            name = f"base_model.model.model.language_model.layers.{li}.{parent}.{proj}.lora_B.weight"
        out[name] = torch.from_numpy(np.ascontiguousarray(v.T)).to(tdtype).contiguous()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_file(out, str(out_dir / "adapter_model.safetensors"), metadata={"format": "pt"})
    cfg = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "r": rank,
        "lora_alpha": alpha,
        "lora_dropout": 0.0,
        "bias": "none",
        "use_rslora": False,
        "target_modules": [p for p in ORDER if p in seen],
        "fan_in_fan_out": False,
        "base_model_name_or_path": base_name,
        "modules_to_save": None,
    }
    (out_dir / "adapter_config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    return {"n_tensors": len(out), "targets": cfg["target_modules"],
            "n_params": int(sum(t.numel() for t in out.values()))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", required=True)
    p.add_argument("--rank", type=int, required=True)
    p.add_argument("--alpha", type=float, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    a = p.parse_args()
    info = convert(load_mlx_adapter(a.adapter), a.rank, a.alpha, a.out, a.dtype)
    print(json.dumps(info))


if __name__ == "__main__":
    main()
