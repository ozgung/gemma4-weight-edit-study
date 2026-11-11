"""LoRA SFT of Gemma 4 E4B (language model only) with MLX on Apple silicon.

Data: DIR/{train,val}.jsonl, one object per line with input_ids and loss_mask
(loss on tokens with loss_mask=1; target t predicted from position t-1). Batch 1,
gradient accumulation with a token-weighted mean over the accumulation group.

Resumable: OUT/ckpt/step_XXXXX/{adapter.safetensors,optimizer.safetensors,state.json}.
At the end (or at --deadline) the adapter is saved and converted to PEFT in OUT/peft.

Example:
  python -I scripts/mlx_lora/train.py --data data/sft/sa_nothink_v1 --out outputs/small_model/runs/x
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import shutil
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_map, tree_unflatten

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import convert_to_peft  # noqa: E402


def utcnow():
    return dt.datetime.now(dt.timezone.utc)


def log(fh, msg):
    line = f"{utcnow().strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    fh.write(line + "\n")
    fh.flush()


def load_split(path, max_len):
    rows = []
    n_trunc = 0
    for r in common.read_jsonl(path):
        ids = np.asarray(r["input_ids"], dtype=np.int32)
        m = np.asarray(r["loss_mask"], dtype=np.int8)
        assert len(ids) == len(m), r.get("id")
        if len(ids) > max_len:
            ids, m = ids[:max_len], m[:max_len]
            n_trunc += 1
        pos, tgt = common.loss_positions(ids, m)
        if len(pos) == 0:
            continue
        rows.append({"id": r.get("id"), "instance_id": r.get("instance_id"), "repo": r.get("repo"),
                     "ids": ids, "pos": pos, "tgt": tgt})
    return rows, n_trunc


def fro_cap(model, cap, scale):
    """Per LoRA module: if ||scale * A B||_F > cap, rescale lora_b so that it equals cap (A [in,r], B [r,out])."""
    from mlx_lm.tuner.lora import LoRALinear
    for _, m in model.named_modules():
        if isinstance(m, LoRALinear):
            n = float(mx.sqrt(((scale * (m.lora_a @ m.lora_b)) ** 2).sum()))
            if n > cap:
                m.lora_b = m.lora_b * (cap / n)
    mx.eval(model.trainable_parameters())


def lr_at(step, total, peak, warmup, min_ratio):
    """step is 0-based index of the optimizer update about to happen."""
    if step < warmup:
        return peak * (step + 1) / warmup
    prog = (step - warmup) / max(1, total - warmup)
    prog = min(1.0, prog)
    return peak * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * prog)))


def save_ckpt(out, step, model, opt, state, keep=2):
    d = out / "ckpt" / f"step_{step:05d}"
    tmp = d.with_suffix(".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    mx.save_safetensors(str(tmp / "adapter.safetensors"), dict(tree_flatten(model.trainable_parameters())))
    mx.save_safetensors(str(tmp / "optimizer.safetensors"), dict(tree_flatten(opt.state)))
    (tmp / "state.json").write_text(json.dumps(state, indent=1))
    if d.exists():
        shutil.rmtree(d)
    tmp.rename(d)
    ck = sorted((out / "ckpt").glob("step_*[0-9]"))
    for old in ck[:-keep]:
        if old.name.startswith("step_") and old.parent == out / "ckpt":
            shutil.rmtree(old)
    return d


def latest_ckpt(out):
    ck = sorted((out / "ckpt").glob("step_*[0-9]")) if (out / "ckpt").exists() else []
    for d in reversed(ck):
        if (d / "state.json").exists() and (d / "adapter.safetensors").exists():
            return d
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--model", default="data/models/gemma-4-e4b-it-qat-q4_0-unquantized")
    p.add_argument("--out", required=True)
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--alpha", type=float, default=32.0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    p.add_argument("--warmup", type=int, default=-1, help="warmup steps; -1 = 5%% of total (min 3)")
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--accum", type=int, default=8)
    p.add_argument("--max-len", type=int, default=16384)
    p.add_argument("--max-steps", type=int, default=0, help="stop after this many updates (smoke tests)")
    p.add_argument("--val-every", type=int, default=10)
    p.add_argument("--val-max", type=int, default=24)
    p.add_argument("--save-every", type=int, default=5)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--chunk", type=int, default=1024, help="query chunk for head_dim>256 global attention")
    p.add_argument("--ce-chunk", type=int, default=512)
    p.add_argument("--lora-compute", default="bf16", choices=["bf16", "fp32"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--targets", default="", help="comma list of q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj (default all)")
    p.add_argument("--layers", default="", help="comma list of layer indices to wrap (default all)")
    p.add_argument("--max-fro", type=float, default=0.0,
                   help="after each update, rescale lora_b so that ||scale * A B||_F <= this, per module (0 = off)")
    p.add_argument("--deadline", default="", help="UTC ISO time; save+convert and stop when an update would cross it")
    p.add_argument("--cache-limit-gb", type=float, default=4.0)
    p.add_argument("--name", default="")
    p.add_argument("--init-adapter", default="", help="MLX adapter.safetensors to start from (fresh optimizer)")
    p.add_argument("--fit-deadline", action="store_true",
                   help="after --replan-at updates, shrink total steps so the schedule ends before --deadline")
    p.add_argument("--replan-at", type=int, default=2)
    p.add_argument("--skip-val0", action="store_true", help="no validation before the first update")
    p.add_argument("--val-secs-hint", type=float, default=0.0, help="val pass seconds for --fit-deadline if none measured")
    p.add_argument("--deadline-margin-min", type=float, default=10.0)
    a = p.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fh = open(out / "train.log", "a")
    log(fh, f"args {json.dumps(vars(a))}")
    mx.set_cache_limit(int(a.cache_limit_gb * 1e9))
    deadline = dt.datetime.fromisoformat(a.deadline.replace("Z", "+00:00")) if a.deadline else None

    data = Path(a.data)
    train, ntr = load_split(data / "train.jsonl", a.max_len)
    val, nva = load_split(data / "val.jsonl", a.max_len) if (data / "val.jsonl").exists() else ([], 0)
    rng_val = np.random.default_rng(1234)
    if len(val) > a.val_max:
        val = [val[i] for i in sorted(rng_val.choice(len(val), a.val_max, replace=False))]
    n_tok_train = int(sum(len(r["ids"]) for r in train))
    n_loss_train = int(sum(len(r["pos"]) for r in train))
    log(fh, f"data train={len(train)} (truncated {ntr}) tokens={n_tok_train} loss_tokens={n_loss_train} | val={len(val)} (truncated {nva})")

    spe = math.ceil(len(train) / a.accum)
    total = max(1, int(round(a.epochs * spe)))
    if a.max_steps:
        total = min(total, a.max_steps)
    warmup = a.warmup if a.warmup >= 0 else max(3, int(round(0.05 * total)))
    log(fh, f"steps_per_epoch={spe} total_steps={total} warmup={warmup}")

    model, cfg, mdir = common.load_lm(a.model)
    tg = tuple(x for x in a.targets.split(",") if x) or common.ALL_TARGETS
    ly = {int(x) for x in a.layers.split(",") if x} or None
    wrapped = common.apply_lora(model, rank=a.rank, alpha=a.alpha, compute=a.lora_compute, targets=tg, layers=ly)
    assert wrapped, "no LoRA modules selected"
    n_lora = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
    log(fh, f"model {mdir} | LoRA modules={len(wrapped)} params={n_lora/1e6:.1f}M | active mem {mx.get_active_memory()/1e9:.1f}GB")
    lw = common.LayerwiseTrainer(model, chunk=a.chunk, ce_chunk=a.ce_chunk)

    if a.init_adapter:
        w = list(mx.load(a.init_adapter).items())
        names = {k for k, _ in tree_flatten(model.trainable_parameters())}
        assert {k for k, _ in w} == names, "init adapter keys differ from the LoRA parameters"
        model.load_weights(w, strict=False)
        mx.eval(model.parameters())
        log(fh, f"initialised LoRA from {a.init_adapter} ({len(w)} tensors)")
    opt = optim.AdamW(learning_rate=a.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=a.weight_decay)
    opt.init(model.trainable_parameters())

    start_step = 0
    hist = {"train": [], "val": []}
    ck = latest_ckpt(out)
    if ck is not None:
        st = json.loads((ck / "state.json").read_text())
        model.load_weights(list(mx.load(str(ck / "adapter.safetensors")).items()), strict=False)
        opt.state = tree_unflatten(list(mx.load(str(ck / "optimizer.safetensors")).items()))
        start_step = st["step"]
        hist = st.get("hist", hist)
        if st.get("total") and st["total"] < total:
            total = st["total"]  # keep a schedule that was shrunk by --fit-deadline
        mx.eval(model.parameters(), opt.state)
        log(fh, f"resumed from {ck} at step {start_step}")

    def order_for_epoch(e):
        return np.random.default_rng(a.seed + 1000 * e).permutation(len(train))

    def batch_for_step(s):
        e, k = divmod(s, spe)
        o = order_for_epoch(e)
        return [int(i) for i in o[k * a.accum:(k + 1) * a.accum]], e

    def run_val():
        if not val:
            return None
        tot, n = 0.0, 0
        for r in val:
            x = mx.array(r["ids"])[None]
            tot += lw.loss(x, mx.array(r["pos"]), mx.array(r["tgt"]))
            n += len(r["pos"])
            mx.clear_cache()
        return tot / n

    val_secs = []
    if start_step == 0 and not a.skip_val0:
        tv = time.time()
        v0 = run_val()
        val_secs.append(time.time() - tv)
        if v0 is not None:
            hist["val"].append([0, v0])
            log(fh, f"val step 0 loss {v0:.4f} ({val_secs[-1]:.0f}s)")

    t_run = time.time()
    tok_run = 0
    step_times = []
    stopped_early = False
    s = start_step
    while s < total:
        idx, epoch = batch_for_step(s)
        t0 = time.time()
        acc = None
        lsum, nl, ntok = 0.0, 0, 0
        for i in idx:
            r = train[i]
            x = mx.array(r["ids"])[None]
            l, g = lw.loss_and_grads(x, mx.array(r["pos"]), mx.array(r["tgt"]))
            acc = g if acc is None else tree_map(lambda u, w: u + w, acc, g)
            mx.eval(acc)
            lsum += l
            nl += len(r["pos"])
            ntok += len(r["ids"])
            mx.clear_cache()
        grads = tree_map(lambda u: u / nl, acc)
        if a.grad_clip > 0:
            grads, gnorm = optim.clip_grad_norm(grads, a.grad_clip)
        else:
            gnorm = mx.sqrt(sum((u * u).sum() for _, u in tree_flatten(grads)))
        lr = lr_at(s, total, a.lr, warmup, a.min_lr_ratio)
        opt.learning_rate = lr
        opt.update(model, grads)
        mx.eval(model.trainable_parameters(), opt.state)
        if a.max_fro > 0:
            fro_cap(model, a.max_fro, a.alpha / a.rank)
        del acc, grads
        mx.clear_cache()
        s += 1
        dt_s = time.time() - t0
        step_times.append(dt_s)
        tok_run += ntok
        tl = lsum / nl
        hist["train"].append([s, tl])
        rate = tok_run / (time.time() - t_run)
        eta = (total - s) * float(np.mean(step_times[-10:]))
        log(fh, f"step {s}/{total} ep {epoch} loss {tl:.4f} gnorm {float(gnorm):.3f} lr {lr:.2e} "
                f"tok {ntok} loss_tok {nl} {dt_s:.0f}s {ntok/dt_s:.0f} tok/s (run {rate:.0f}) "
                f"peak {mx.get_peak_memory()/1e9:.1f}GB eta {eta/60:.0f}min")
        mx.reset_peak_memory()
        if a.fit_deadline and deadline is not None and s == start_step + a.replan_at:
            per = float(np.mean(step_times))
            vt = float(np.mean(val_secs)) if val_secs else a.val_secs_hint
            left = (deadline - utcnow()).total_seconds() - 60 * a.deadline_margin_min
            fit = s
            while fit < total:
                n_val = (sum(1 for j in range(s + 1, fit + 2) if a.val_every and j % a.val_every == 0) + 1) if val else 0
                if (fit + 1 - s) * per + n_val * vt > left:
                    break
                fit += 1
            if fit < total:
                log(fh, f"fit-deadline: {per:.0f}s/step, val {vt:.0f}s -> total steps {total} -> {fit}")
                total = fit
            else:
                log(fh, f"fit-deadline: {per:.0f}s/step, val {vt:.0f}s -> all {total} steps fit")
        if a.val_every and (s % a.val_every == 0 or s == total):
            tv = time.time()
            v = run_val()
            val_secs.append(time.time() - tv)
            if v is not None:
                hist["val"].append([s, v])
                log(fh, f"val step {s} loss {v:.4f}")
        state = {"step": s, "total": total, "epoch": epoch, "hist": hist, "args": vars(a),
                 "time": utcnow().isoformat()}
        if s % a.save_every == 0 or s == total:
            d = save_ckpt(out, s, model, opt, state)
            log(fh, f"saved {d}")
        if deadline is not None and s < total:
            nxt = utcnow() + dt.timedelta(seconds=1.3 * float(np.mean(step_times[-5:])) + 120)
            if nxt > deadline:
                d = save_ckpt(out, s, model, opt, state)
                log(fh, f"deadline {a.deadline} would be crossed; stopping at step {s}/{total}; saved {d}")
                stopped_early = True
                break

    # final adapter
    final = out / "final"
    final.mkdir(exist_ok=True)
    mx.save_safetensors(str(final / "adapter.safetensors"), dict(tree_flatten(model.trainable_parameters())))
    info = convert_to_peft.convert(convert_to_peft.load_mlx_adapter(final / "adapter.safetensors"),
                                   a.rank, a.alpha, out / "peft")
    summary = {"steps_done": s, "total_steps": total, "stopped_early": stopped_early, "hist": hist,
               "peft": info, "train_examples": len(train), "train_tokens": n_tok_train,
               "train_loss_tokens": n_loss_train, "val_examples": len(val),
               "wall_s": time.time() - t_run, "args": vars(a), "lora_modules": len(wrapped)}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    log(fh, f"done: steps {s}/{total} early={stopped_early} peft={info}")


if __name__ == "__main__":
    main()
