"""transformers (CPU) side of the MLX<->HF equivalence check.

Reads outputs of `verify_mlx.py logits` (mlx_logits.npz, mlx_random_adapter.safetensors),
converts the MLX adapter to PEFT format with convert_to_peft.py, runs
Gemma4ForConditionalGeneration (CPU) without and with the PEFT adapter on the same ids,
and reports max/mean |logit diff| vs MLX.
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
import convert_to_peft  # noqa: E402
from common import find_model_dir  # noqa: E402

LM_REGEX = r".*\.language_model\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))"


def stats(a, b):
    d = np.abs(a - b)
    return {"max": float(d.max()), "mean": float(d.mean()),
            "argmax_agree": float((a.argmax(-1) == b.argmax(-1)).mean())}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="data/models/gemma-4-e4b-it-qat-q4_0-unquantized")
    p.add_argument("--dir", default="outputs/small_model/verify")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    p.add_argument("--attn", default="eager")
    p.add_argument("--peft-dir", default="", help="an existing PEFT adapter dir (default: convert the random one)")
    a = p.parse_args()
    d = Path(a.dir)
    z = np.load(d / "mlx_logits.npz")
    ids = z["ids"].astype(np.int64)
    meta = json.loads((d / "mlx_random_adapter.json").read_text())
    if a.peft_dir:
        peft_dir = Path(a.peft_dir)
    else:
        peft_dir = d / "peft_random_adapter"
        info = convert_to_peft.convert(convert_to_peft.load_mlx_adapter(d / "mlx_random_adapter.safetensors"),
                                       meta["rank"], meta["alpha"], peft_dir)
        print("converted:", info)

    from transformers import AutoConfig, Gemma4ForConditionalGeneration

    mdir = find_model_dir(a.model)
    dtype = getattr(torch, a.dtype)
    t = time.time()
    model = Gemma4ForConditionalGeneration.from_pretrained(mdir, dtype=dtype, attn_implementation=a.attn)
    model.eval()
    print(f"loaded HF model in {time.time()-t:.0f}s", flush=True)
    x = torch.tensor(ids)[None]
    with torch.no_grad():
        base = model(input_ids=x).logits[0].float().numpy()
    res = {"n_tokens": int(len(ids)), "hf_dtype": a.dtype, "attn": a.attn}
    res["base_mlx_vs_hf"] = stats(z["base"], base)

    from peft import PeftModel, LoraConfig

    loaded_with = "adapter_config.json as written"
    try:
        pm = PeftModel.from_pretrained(model, str(peft_dir))
    except Exception as e:  # tower wrappers (Gemma4ClippableLinear) may break suffix matching
        print("PEFT load with the as-written config failed:", type(e).__name__, str(e)[:300])
        cfg = json.loads((peft_dir / "adapter_config.json").read_text())
        override = LoraConfig(r=cfg["r"], lora_alpha=cfg["lora_alpha"], lora_dropout=0.0, bias="none",
                              use_rslora=False, target_modules=LM_REGEX, task_type="CAUSAL_LM")
        pm = PeftModel.from_pretrained(model, str(peft_dir), config=override)
        loaded_with = "regex target_modules override (language_model only)"
    res["peft_loaded_with"] = loaded_with
    # check every adapter tensor was consumed
    sd = {k for k in pm.state_dict() if "lora_" in k}
    from safetensors.torch import load_file
    w = load_file(str(peft_dir / "adapter_model.safetensors"))
    exp = {k.replace(".lora_A.weight", ".lora_A.default.weight").replace(".lora_B.weight", ".lora_B.default.weight") for k in w}
    res["adapter_tensors"] = len(w)
    res["adapter_tensors_missing_in_model"] = len(exp - sd)
    res["model_lora_tensors_not_in_adapter"] = len(sd - exp)
    # verify loaded values equal the file values
    psd = pm.state_dict()
    maxdiff = 0.0
    for k, v in w.items():
        kk = k.replace(".lora_A.weight", ".lora_A.default.weight").replace(".lora_B.weight", ".lora_B.default.weight")
        maxdiff = max(maxdiff, float((psd[kk].float() - v.float()).abs().max()))
    res["loaded_weight_maxdiff"] = maxdiff
    with torch.no_grad():
        lora = pm(input_ids=x).logits[0].float().numpy()
    np.savez(d / "hf_logits.npz", base=base, lora=lora)
    res["lora_mlx_vs_hf"] = stats(z["lora"], lora)
    res["lora_trainfwd_mlx_vs_hf"] = stats(z["lora_trainfwd"], lora)
    res["lora_effect_hf"] = stats(lora, base)
    res["lora_effect_mlx"] = stats(z["lora"], z["base"])
    print(json.dumps(res, indent=2))
    (d / "equivalence.json").write_text(json.dumps(res, indent=2) + "\n")


if __name__ == "__main__":
    main()
