"""Package a finished (or partial) MLX LoRA run as a PEFT adapter directory with README.txt.

Usage:
  python -I scripts/mlx_lora/finalize.py --run outputs/small_model/runs/mlx_sa_nothink_v1 \
      --name mlx_sa_nothink_v1 [--from final|latest]
Writes outputs/small_model/adapters/<name>/{adapter_config.json,adapter_model.safetensors,README.txt}
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import convert_to_peft  # noqa: E402

KEY = re.compile(r"^base_model\.model\.model\.language_model\.layers\.(\d+)\.(self_attn|mlp)\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)\.lora_(A|B)\.weight$")


def check_peft_dir(d: Path, rank: int) -> dict:
    from safetensors import safe_open

    n, layers, bad = 0, set(), []
    with safe_open(str(d / "adapter_model.safetensors"), "pt") as f:
        for k in f.keys():
            m = KEY.match(k)
            if not m:
                bad.append(k)
                continue
            t = f.get_tensor(k)
            li, _, proj, ab = m.groups()
            layers.add(int(li))
            if ab == "A" and t.shape[0] != rank or ab == "B" and t.shape[1] != rank:
                bad.append(f"{k} {tuple(t.shape)}")
            if not bool(t.float().isfinite().all()):
                bad.append(f"{k} non-finite")
            if int(li) >= 24 and proj in ("k_proj", "v_proj"):
                bad.append(f"{k} on a KV-shared layer")
            n += 1
    return {"tensors": n, "layers": len(layers), "problems": bad[:10]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--from", dest="src", default="final", choices=["final", "latest"])
    p.add_argument("--root", default="outputs/small_model/adapters")
    p.add_argument("--note", default="")
    a = p.parse_args()
    run = Path(a.run)
    log_lines = (run / "train.log").read_text().splitlines()
    args_line = next(l for l in log_lines if " args {" in l)
    args = json.loads(args_line.split(" args ", 1)[1])
    if a.src == "final":
        src = run / "final" / "adapter.safetensors"
        state = json.loads((run / "summary.json").read_text())
        steps_done, total = state["steps_done"], state["total_steps"]
        hist = state["hist"]
    else:
        ck = sorted((run / "ckpt").glob("step_*[0-9]"))[-1]
        src = ck / "adapter.safetensors"
        st = json.loads((ck / "state.json").read_text())
        steps_done, total, hist = st["step"], st["total"], st["hist"]
    out = Path(a.root) / a.name
    info = convert_to_peft.convert(convert_to_peft.load_mlx_adapter(src), args["rank"], args["alpha"], out)
    chk = check_peft_dir(out, args["rank"])
    assert chk["tensors"] == 516 and not chk["problems"], chk

    meta_p = Path(args["data"]) / "meta.json"
    meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
    tr = hist.get("train", [])
    va = hist.get("val", [])
    tl = [x[1] for x in tr]
    try:
        rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception:
        rev = "?"
    sched = [l for l in log_lines if "steps_per_epoch=" in l]
    m_w = re.search(r"warmup=(\d+)", sched[-1]) if sched else None
    warm_eff = m_w.group(1) if m_w else str(args["warmup"])
    fits = [l.split("fit-deadline: ", 1)[1] for l in log_lines if "fit-deadline:" in l]
    step_lines = [l for l in log_lines if re.search(r" step \d+/\d+ ep ", l)]
    secs = [float(m.group(1)) for l in step_lines if (m := re.search(r" (\d+)s \d+ tok/s", l))]
    lines = [
        f"Adapter: {a.name}",
        f"Source run: {run} ({a.src}: {src})",
        f"Base model (training): data/models/gemma-4-e4b-it-qat-q4_0-unquantized (bf16, Gemma4ForConditionalGeneration; towers unused)",
        f"Serve base: gemma-4-e4b-it-qat-w4a16-ct (vLLM 0.19.1); PEFT LoRA, keys base_model.model.model.language_model.layers.N.*",
        "",
        "DATA",
        f"  dataset: {args['data']} (meta: created {meta.get('created_utc', '?')}, mode {meta.get('mode', '?')}, "
        f"script {meta.get('script', '?')})",
        f"  train samples {meta.get('counts', {}).get('samples_train', '?')}, tokens {meta.get('counts', {}).get('tokens', {}).get('train', '?')}, "
        f"loss tokens {meta.get('counts', {}).get('loss_tokens', {}).get('train', '?')}; val samples {meta.get('counts', {}).get('samples_val', '?')}",
        f"  split: {meta.get('split', {}).get('rule', '?')}",
        f"  export target submission: {meta.get('args', {}).get('target_submission', '?')}; arms {meta.get('args', {}).get('arms', '?')}",
        "",
        "TRAINING (MLX on Apple M4 Max, scripts/mlx_lora/train.py)",
        f"  LoRA r={args['rank']} alpha={args['alpha']} (scale {args['alpha']/args['rank']:g}), dropout 0, targets q/k/v/o/gate/up/down "
        f"on language_model.layers (258 modules; k/v absent on KV-shared layers 24-41), {info['n_params']/1e6:.1f}M params",
        f"  steps done {steps_done}/{total} (planned epochs {args['epochs']}, accumulation {args['accum']} sequences/step, batch 1)",
        f"  optimizer AdamW(0.9, 0.999, eps 1e-8, wd {args['weight_decay']}), peak lr {args['lr']}, warmup {warm_eff} steps, "
        f"cosine to {args['min_lr_ratio']}x, grad clip {args['grad_clip']}, seed {args['seed']}"
        + (f", init adapter {args.get('init_adapter')}" if args.get('init_adapter') else ""),
        "  loss: next-token NLL on loss_mask=1 tokens, final-logit softcap 30, token-weighted mean over each accumulation group",
        f"  kernels: layer-wise explicit backprop (recompute per layer), blocked 512-window attention, global attention query chunk "
        f"{args['chunk']}, CE chunk {args['ce_chunk']}, LoRA matmuls bf16 with fp32 master weights",
        f"  wall time per step: mean {np.mean(secs):.0f}s over {len(secs)} logged steps" if secs else "",
        f"  schedule re-planned to fit the deadline: {fits[-1]}" if fits else "",
        f"  git revision {rev}",
        "",
        "LOSSES (nats/token)",
        f"  train: first {tl[0]:.4f}, last {tl[-1]:.4f}, mean of first 5 {np.mean(tl[:5]):.4f}, mean of last 5 {np.mean(tl[-5:]):.4f}" if tl else "  train: none",
        "  val: " + ", ".join(f"step {s}: {v:.4f}" for s, v in va),
        "",
        "VERIFICATION (scripts/mlx_lora/verify_*.py, outputs/small_model/verify/)",
        "  MLX vs transformers+PEFT on a 64-token input with a random r32 adapter: differences at bf16 noise level",
        "  (max 2.0 / mean 0.11 logits, vs an adapter effect of mean 0.45 / max 10.5; vs an fp32 reference both bf16 paths",
        "  deviate equally). Converter key pattern/shapes match smoke_adapters/rand8, which loads in the competition vLLM.",
        f"  this adapter: {chk['tensors']} tensors over {chk['layers']} layers, all finite, no k/v on shared layers",
    ]
    if a.note:
        lines += ["", "NOTE", "  " + a.note]
    (out / "README.txt").write_text("\n".join(l for l in lines if l is not None) + "\n")
    print(json.dumps({"out": str(out), "steps": f"{steps_done}/{total}", "check": chk, "val": va}, indent=1))


if __name__ == "__main__":
    main()
