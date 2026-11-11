"""MLX-side verification for the E4B LoRA lane.

Subcommands:
  gen        greedy generation from a short chat prompt (coherence check)
  trainfwd   training forward (blocked sliding / chunked global attention, checkpointed)
             vs. the stock mlx-lm forward: logits, masked NLL and LoRA gradients
  logits     write logits for a fixed 64-token input (base and with a random adapter)
             to an .npz for comparison with transformers (see verify_hf.py)
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
from mlx.utils import tree_flatten

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

DEFAULT_MODEL = "data/models/gemma-4-e4b-it-qat-q4_0-unquantized"


def cmd_gen(args):
    from mlx_lm import generate
    from mlx_lm.utils import load_tokenizer

    model, cfg, mdir = common.load_lm(args.model)
    tok = load_tokenizer(mdir)
    msgs = [{"role": "user", "content": "Write a Python function that checks whether a string is a palindrome, then explain it in one sentence."}]
    prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    print("PROMPT:", repr(prompt[-200:]))
    t = time.time()
    out = generate(model, tok, prompt=prompt, max_tokens=args.max_tokens, verbose=False)
    print(f"GEN ({time.time()-t:.1f}s):\n{out}")
    print("peak GB", mx.get_peak_memory() / 1e9)


def random_input(tok_vocab, L, seed):
    rng = np.random.default_rng(seed)
    # avoid special-token range: draw from common ids
    return rng.integers(1000, 200000, size=L).astype(np.int32)


def set_random_lora(model, seed, b_std):
    """Give every LoRA module a non-trivial random (A, B)."""
    rng = np.random.default_rng(seed)
    upd = []
    for k, v in tree_flatten(model.trainable_parameters()):
        if k.endswith("lora_a"):
            fan_in = v.shape[0]
            arr = rng.uniform(-1 / np.sqrt(fan_in), 1 / np.sqrt(fan_in), size=v.shape)
        else:
            arr = rng.normal(0.0, b_std, size=v.shape)
        upd.append((k, mx.array(arr.astype(np.float32))))
    model.load_weights(upd, strict=False)
    mx.eval(model.parameters())


def cmd_trainfwd(args):
    model, cfg, mdir = common.load_lm(args.model)
    tc = cfg.get("text_config", cfg)
    print("layer_types:", tc.get("layer_types"))
    common.apply_lora(model, rank=args.rank, alpha=args.alpha)
    set_random_lora(model, 0, args.b_std)
    L = args.length
    ids = random_input(None, L, 1)
    rng = np.random.default_rng(2)
    lm = (rng.random(L) < 0.3).astype(np.int8)
    pos, tgt = common.loss_positions(ids, lm)
    x = mx.array(ids)[None]
    posm, tgtm = mx.array(pos), mx.array(tgt)

    # Reference: stock mlx-lm forward with full logits
    def ref_loss(model):
        logits = model(x)[0, :-1].astype(mx.float32)
        lse = mx.logsumexp(logits[posm], axis=-1)
        t = mx.take_along_axis(logits[posm], tgtm[:, None], axis=-1)[:, 0]
        return (lse - t).sum()

    fwd = common.TrainForward(model, checkpoint=True, chunk=args.chunk)

    def new_loss(model):
        h = fwd(x)
        return common.masked_nll_sum(model, h, posm, tgtm, chunk=256)

    # logits comparison (forward only)
    logits_ref = model(x)[0].astype(mx.float32)
    h = fwd(x)
    W = common.lm_head_weight(model)
    logits_new = (h[0] @ W.T).astype(mx.float32)
    sc = model.language_model.final_logit_softcapping
    logits_new = mx.tanh(logits_new / sc) * sc
    d = mx.abs(logits_ref - logits_new)
    mx.eval(d)
    print(f"L={L} train-forward vs stock logits: max|d|={d.max().item():.4g} mean|d|={d.mean().item():.4g} "
          f"argmax agree={float((logits_ref.argmax(-1) == logits_new.argmax(-1)).mean().item()):.4f}")
    del logits_ref, logits_new, d, h

    t = time.time()
    lr_, g_ref = nn.value_and_grad(model, ref_loss)(model)
    mx.eval(lr_, g_ref)
    t_ref = time.time() - t
    mx.reset_peak_memory()
    t = time.time()
    ln_, g_new = nn.value_and_grad(model, new_loss)(model)
    mx.eval(ln_, g_new)
    t_new = time.time() - t
    print(f"loss ref={lr_.item()/len(pos):.6f} new={ln_.item()/len(pos):.6f} (per token, N={len(pos)}) "
          f"time ref={t_ref:.1f}s new={t_new:.1f}s peak(new)={mx.get_peak_memory()/1e9:.1f}GB")
    fr = dict(tree_flatten(g_ref))
    fn_ = dict(tree_flatten(g_new))
    worst = 0.0
    rels = []
    for k in fr:
        a, b = fr[k].astype(mx.float32), fn_[k].astype(mx.float32)
        rel = (mx.linalg.norm(a - b) / (mx.linalg.norm(a) + 1e-12)).item()
        rels.append(rel)
        worst = max(worst, rel)
    print(f"LoRA grads: {len(rels)} tensors, rel L2 diff median={np.median(rels):.3g} max={worst:.3g}")


def cmd_logits(args):
    model, cfg, mdir = common.load_lm(args.model)
    ids = np.load(args.ids) if args.ids else random_input(None, 64, 7)
    x = mx.array(ids.astype(np.int32))[None]
    base = model(x)[0].astype(mx.float32)
    mx.eval(base)
    base = np.array(base)
    common.apply_lora(model, rank=args.rank, alpha=args.alpha)
    if args.adapter:  # a trained MLX adapter instead of a random one
        model.load_weights(list(mx.load(args.adapter).items()), strict=False)
        mx.eval(model.parameters())
    else:
        set_random_lora(model, args.seed, args.b_std)
    lo = model(x)[0].astype(mx.float32)
    mx.eval(lo)
    lo = np.array(lo)
    # the training forward too (should equal stock forward)
    fwd = common.TrainForward(model, checkpoint=False)
    h = fwd(x)
    W = common.lm_head_weight(model)
    sc = model.language_model.final_logit_softcapping
    tr = mx.tanh((h[0] @ W.T).astype(mx.float32) / sc) * sc
    mx.eval(tr)
    tr = np.array(tr)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "mlx_logits.npz", ids=ids, base=base, lora=lo, lora_trainfwd=tr)
    adapter = {k: v for k, v in tree_flatten(model.trainable_parameters())}
    mx.save_safetensors(str(out / "mlx_random_adapter.safetensors"), adapter)
    meta = {"rank": args.rank, "alpha": args.alpha, "seed": args.seed, "b_std": args.b_std}
    (out / "mlx_random_adapter.json").write_text(json.dumps(meta))
    print("saved", out, "| lora effect max|d|", float(np.abs(lo - base).max()), "mean|d|", float(np.abs(lo - base).mean()),
          "| trainfwd vs stock max|d|", float(np.abs(tr - lo).max()))
    print("peak GB", mx.get_peak_memory() / 1e9)


def cmd_logits32(args):
    """fp32 reference in MLX: every weight in fp32 except the per-layer embedding table
    (a bf16 lookup whose values are exact in bf16). Same ids and random adapter as `logits`."""
    from mlx.utils import tree_map_with_path
    model, cfg, mdir = common.load_lm(args.model, lazy=True)

    def cast(path, x):
        return x if path.endswith("embed_tokens_per_layer.weight") else x.astype(mx.float32)

    model.update(tree_map_with_path(cast, model.parameters()))
    mx.eval(model.parameters())
    print("fp32 model active GB", mx.get_active_memory() / 1e9, flush=True)
    out = Path(args.out)
    z = np.load(out / "mlx_logits.npz")
    x = mx.array(z["ids"].astype(np.int32))[None]
    base = model(x)[0].astype(mx.float32)
    mx.eval(base)
    base = np.array(base)
    meta = json.loads((out / "mlx_random_adapter.json").read_text())
    common.apply_lora(model, rank=meta["rank"], alpha=meta["alpha"])
    model.load_weights(list(mx.load(str(out / "mlx_random_adapter.safetensors")).items()), strict=False)
    lo = model(x)[0].astype(mx.float32)
    mx.eval(lo)
    np.savez(out / "mlx32_logits.npz", base=base, lora=np.array(lo))
    print("saved fp32 logits; peak GB", mx.get_peak_memory() / 1e9)


def cmd_layerwise(args):
    """LayerwiseTrainer (explicit per-layer VJP) vs autograd over TrainForward: loss + LoRA grads."""
    model, cfg, mdir = common.load_lm(args.model)
    common.apply_lora(model, rank=args.rank, alpha=args.alpha)
    set_random_lora(model, 0, args.b_std)
    L = args.length
    ids = random_input(None, L, 1)
    lm = (np.random.default_rng(2).random(L) < 0.3).astype(np.int8)
    pos, tgt = common.loss_positions(ids, lm)
    x, posm, tgtm = mx.array(ids)[None], mx.array(pos), mx.array(tgt)
    fwd = common.TrainForward(model, checkpoint=False, chunk=args.chunk)
    t = time.time()
    l_ref, g_ref = nn.value_and_grad(model, lambda m: common.masked_nll_sum(m, fwd(x), posm, tgtm, chunk=256))(model)
    mx.eval(l_ref, g_ref)
    t_ref = time.time() - t
    lw = common.LayerwiseTrainer(model, chunk=args.chunk, ce_chunk=256)
    mx.reset_peak_memory()
    t = time.time()
    l_new, g_new = lw.loss_and_grads(x, posm, tgtm)
    t_new = time.time() - t
    fr, fn_ = dict(tree_flatten(g_ref)), dict(tree_flatten(g_new))
    assert set(fr) == set(fn_), (len(fr), len(fn_), list(set(fr) ^ set(fn_))[:5])
    rels = [(mx.linalg.norm(fr[k].astype(mx.float32) - fn_[k].astype(mx.float32)) /
             (mx.linalg.norm(fr[k].astype(mx.float32)) + 1e-12)).item() for k in fr]
    print(f"L={L} loss/token autograd={l_ref.item()/len(pos):.6f} layerwise={l_new/len(pos):.6f} | "
          f"grads {len(rels)} tensors rel L2 diff median={np.median(rels):.3g} max={max(rels):.3g} | "
          f"time {t_ref:.1f}s vs {t_new:.1f}s, layerwise peak {mx.get_peak_memory()/1e9:.1f}GB")
    tp = dict(tree_flatten(model.trainable_parameters()))
    assert set(tp) == set(fn_), "grad tree differs from trainable_parameters"
    print("grad tree matches trainable_parameters:", len(tp))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["gen", "trainfwd", "logits", "logits32", "layerwise"])
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--max-tokens", type=int, default=120)
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--alpha", type=float, default=32.0)
    p.add_argument("--b-std", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--length", type=int, default=1536)
    p.add_argument("--chunk", type=int, default=512)
    p.add_argument("--ids", default=None)
    p.add_argument("--adapter", default="", help="trained MLX adapter.safetensors for `logits`")
    p.add_argument("--out", default="outputs/small_model/verify")
    args = p.parse_args()
    {"gen": cmd_gen, "trainfwd": cmd_trainfwd, "logits": cmd_logits, "logits32": cmd_logits32, "layerwise": cmd_layerwise}[args.cmd](args)


if __name__ == "__main__":
    main()
