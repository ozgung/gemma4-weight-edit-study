"""Shared pieces of the MLX LoRA lane for Gemma 4 E4B.

- load_lm(): loads the language model (towers dropped by mlx-lm's sanitize) in bf16.
- apply_lora(): wraps the language-model linears (q,k,v,o,gate,up,down) with LoRALinear.
- hidden_states(): memory-lean training forward (per-layer gradient checkpointing,
  blocked sliding-window attention, query-chunked global attention).
- masked_nll_sum(): sum of token NLL on loss positions only (logits never materialised
  for the whole sequence; chunked and checkpointed).

Model files are treated as data: only config/safetensors/tokenizer are read.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten
from mlx_lm.tuner.lora import LoRALinear

ATTN_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP_TARGETS = ("gate_proj", "up_proj", "down_proj")
ALL_TARGETS = ATTN_TARGETS + MLP_TARGETS


def find_model_dir(root: str | Path) -> Path:
    root = Path(root)
    cands = sorted(p.parent for p in root.rglob("config.json"))
    for c in cands:
        if list(c.glob("*.safetensors")):
            return c
    raise FileNotFoundError(f"no config.json with safetensors under {root}")


def load_lm(model_root: str | Path, lazy: bool = False):
    """Return (model, config, model_dir). model is mlx_lm gemma4.Model (text only)."""
    from mlx_lm.utils import load_model

    model_dir = find_model_dir(model_root)
    model, config = load_model(model_dir, lazy=True, strict=True)
    # keep everything in bf16 (the checkpoint is bf16; norms/scalars too)
    if not lazy:
        mx.eval(model.parameters())
    return model, config, model_dir


def text_model(model):
    """The Gemma4TextModel inside mlx_lm gemma4.Model."""
    return model.language_model.model


class LoRALinearBF16(LoRALinear):
    """LoRA with fp32 master weights (lora_a [in,r], lora_b [r,out]) computed in the
    activation dtype (bf16), like bf16 autocast training and vLLM's bf16 LoRA kernels."""

    def __call__(self, x):
        y = self.linear(x)
        z = (x @ self.lora_a.astype(x.dtype)) @ self.lora_b.astype(x.dtype)
        if self.scale != 1.0:
            z = z * self.scale
        return y + z


def apply_lora(model, rank: int = 32, alpha: float = 32.0, targets=ALL_TARGETS, dropout: float = 0.0,
               compute: str = "bf16", layers=None):
    """Freeze the model and wrap language-model linears with LoRA. Returns list of wrapped paths.
    compute='bf16' (default): LoRA matmuls in bf16 with fp32 master weights; 'fp32': mlx-lm LoRALinear.
    layers: optional set of layer indices to wrap (default: all)."""
    model.freeze()
    scale = alpha / rank  # PEFT scaling (use_rslora=False)
    cls = LoRALinearBF16 if compute == "bf16" else LoRALinear
    wrapped = []
    tm = text_model(model)
    for i, layer in enumerate(tm.layers):
        if layers is not None and i not in layers:
            continue
        for parent_name, names in (("self_attn", ATTN_TARGETS), ("mlp", MLP_TARGETS)):
            parent = getattr(layer, parent_name)
            for n in names:
                if n not in targets or n not in parent:  # k/v absent on KV-shared layers
                    continue
                lin = parent[n]
                if not isinstance(lin, nn.Linear):
                    raise TypeError(f"layer {i} {parent_name}.{n} is {type(lin)}")
                lora = LoRALinear.from_base(lin, r=rank, dropout=dropout, scale=scale)
                if cls is not LoRALinear:
                    lora.__class__ = cls
                setattr(parent, n, lora)
                wrapped.append(f"language_model.model.layers.{i}.{parent_name}.{n}")
    return wrapped


def lora_state(model) -> dict:
    """Flat dict of LoRA arrays (MLX names)."""
    return {k: v for k, v in tree_flatten(model.trainable_parameters())}


# ----------------------------------------------------------------------------------
# Attention for training
# ----------------------------------------------------------------------------------

def _sdpa(q, k, v, mask):
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0, mask=mask)


_ckpt_causal = mx.checkpoint(lambda q, k, v: _sdpa(q, k, v, "causal"))


def _ckpt_masked(q, k, v, m):
    # mx.checkpoint computes VJPs for every argument; keep the boolean mask out of the
    # argument list (closure constant) so no gradient is requested for it.
    return mx.checkpoint(lambda q_, k_, v_: _sdpa(q_, k_, v_, m))(q, k, v)


def sliding_attention(q, k, v, window: int, block: int | None = None):
    """Causal attention restricted to the last `window` keys (k > p - window, k <= p).

    Queries are processed in blocks of size `window`; each block attends to its own
    block plus the previous one with a band mask. Memory O(L * 2W) instead of O(L^2).
    q: [B, H, L, D], k/v: [B, KVH, L, D].
    """
    B, H, L, D = q.shape
    KVH = k.shape[1]
    W = window
    if L <= W:
        return _sdpa(q, k, v, "causal")
    nb = -(-L // W)
    Lp = nb * W
    pad = Lp - L
    if pad:
        pw = [(0, 0), (0, 0), (0, pad), (0, 0)]
        q, k, v = mx.pad(q, pw), mx.pad(k, pw), mx.pad(v, pw)
    qb = q.reshape(B, H, nb, W, D).transpose(0, 2, 1, 3, 4).reshape(B * nb, H, W, D)

    def two_blocks(x):
        xb = x.reshape(B, KVH, nb, W, D)
        prev = mx.concatenate([mx.zeros_like(xb[:, :, :1]), xb[:, :, :-1]], axis=2)
        xx = mx.concatenate([prev, xb], axis=3)  # [B, KVH, nb, 2W, D]
        return xx.transpose(0, 2, 1, 3, 4).reshape(B * nb, KVH, 2 * W, D)

    kk, vv = two_blocks(k), two_blocks(v)
    a = mx.arange(W)[:, None]
    c = mx.arange(2 * W)[None]
    band = (c > a) & (c <= a + W)  # local key c is at abs pos (i-1)W + c, query at iW + a
    first = band & (c >= W)  # block 0 has no previous block
    mask = mx.concatenate([first[None], mx.broadcast_to(band[None], (nb - 1, W, 2 * W))], axis=0)
    mask = mx.tile(mask[:, None], (B, 1, 1, 1)) if B > 1 else mask[:, None]  # [B*nb, 1, W, 2W]
    o = _ckpt_masked(qb, kk, vv, mask)
    o = o.reshape(B, nb, H, W, D).transpose(0, 2, 1, 3, 4).reshape(B, H, Lp, D)
    return o[:, :, :L] if pad else o


def global_attention(q, k, v, chunk: int = 2048, fused_bwd_max_dim: int = 256):
    """Full causal attention. MLX's fused SDPA has a fused backward only for head_dim<=256
    (measured); for larger head dims the queries are chunked and each chunk checkpointed
    so the unfused backward only materialises [chunk x keys] scores."""
    L, D = q.shape[2], q.shape[3]
    if D <= fused_bwd_max_dim or L <= chunk:
        return _sdpa(q, k, v, "causal")
    outs = []
    for s in range(0, L, chunk):
        e = min(s + chunk, L)
        outs.append(_ckpt_causal(q[:, :, s:e], k[:, :, :e], v[:, :, :e]))
    return mx.concatenate(outs, axis=2)


def attn_forward(attn, x, shared_kv, window, chunk):
    B, L, _ = x.shape
    q = attn.q_proj(x).reshape(B, L, attn.n_heads, attn.head_dim)
    q = attn.q_norm(q)
    if shared_kv is not None:
        k, v = shared_kv
    else:
        k = attn.k_proj(x).reshape(B, L, attn.n_kv_heads, attn.head_dim)
        v = k if attn.use_k_eq_v else attn.v_proj(x).reshape(B, L, attn.n_kv_heads, attn.head_dim)
        k = attn.k_norm(k).transpose(0, 2, 1, 3)
        k = attn.rope(k, offset=0)
        v = attn.v_norm(v).transpose(0, 2, 1, 3)
    q = q.transpose(0, 2, 1, 3)
    q = attn.rope(q, offset=0)
    if attn.is_sliding:
        o = sliding_attention(q, k, v, window)
    else:
        o = global_attention(q, k, v, chunk)
    o = o.transpose(0, 2, 1, 3).reshape(B, L, -1)
    return attn.o_proj(o), (k, v)


def layer_forward(layer, h, per_layer_input, shared_kv, window, chunk):
    residual = h
    x = layer.input_layernorm(h)
    x, kv = attn_forward(layer.self_attn, x, shared_kv, window, chunk)
    x = layer.post_attention_layernorm(x)
    h = residual + x
    residual = h
    if layer.enable_moe:
        raise NotImplementedError("MoE layers not supported in this lane")
    x = layer.pre_feedforward_layernorm(h)
    x = layer.mlp(x)
    x = layer.post_feedforward_layernorm(x)
    h = residual + x
    if layer.per_layer_input_gate is not None and per_layer_input is not None:
        residual = h
        g = nn.gelu_approx(layer.per_layer_input_gate(h))
        g = g * per_layer_input
        g = layer.per_layer_projection(g)
        g = layer.post_per_layer_input_norm(g)
        h = residual + g
    if layer.layer_scalar is not None:
        h = h * layer.layer_scalar
    return h, kv


class TrainForward:
    """Memory-lean forward over Gemma4TextModel returning final-normed hidden states."""

    def __init__(self, model, checkpoint: bool = True, window: int | None = None, chunk: int = 2048):
        self.model = model
        self.tm = text_model(model)
        self.window = window or self.tm.window_size
        self.chunk = chunk
        self.checkpoint = checkpoint
        n = len(self.tm.layers)
        self.prev = list(self.tm.previous_kvs)
        self.kv_sources = {self.prev[j] for j in range(n) if self.prev[j] != j}
        self._fns = [self._make(i) for i in range(n)]

    def _make(self, i):
        layer = self.tm.layers[i]
        shared = self.prev[i] != i
        returns_kv = i in self.kv_sources
        W, C = self.window, self.chunk

        if shared:
            def inner(params, h, pli, k, v):
                layer.update(params)
                out, _ = layer_forward(layer, h, pli, (k, v), W, C)
                return out
        elif returns_kv:
            def inner(params, h, pli):
                layer.update(params)
                out, (k, v) = layer_forward(layer, h, pli, None, W, C)
                return out, k, v
        else:
            def inner(params, h, pli):
                layer.update(params)
                out, _ = layer_forward(layer, h, pli, None, W, C)
                return out

        fn = mx.checkpoint(inner) if self.checkpoint else inner

        def call(h, pli, kv):
            params = layer.trainable_parameters()
            if shared:
                return fn(params, h, pli, *kv), None
            if returns_kv:
                out, k, v = fn(params, h, pli)
                return out, (k, v)
            return fn(params, h, pli), None

        return call

    def __call__(self, input_ids):
        tm = self.tm
        emb = tm.embed_tokens(input_ids)
        h = emb * tm.embed_scale  # python float: same weak-typed bf16 multiply as mlx-lm
        pli_all = None
        if tm.hidden_size_per_layer_input:
            pli_all = tm._get_per_layer_inputs(input_ids, emb)
            pli_all = tm._project_per_layer_inputs(h, pli_all)
        kvs = {}
        for i, fn in enumerate(self._fns):
            pli = pli_all[:, :, i, :] if pli_all is not None else None
            kv_in = kvs.get(self.prev[i]) if self.prev[i] != i else None
            h, kv = fn(h, pli, kv_in)
            if kv is not None:
                kvs[i] = kv
        return tm.norm(h)


def lm_head_weight(model):
    lm = model.language_model
    if lm.tie_word_embeddings:
        return lm.model.embed_tokens.weight
    return lm.lm_head.weight


def masked_nll_sum(model, hidden, positions, targets, chunk: int = 1024):
    """hidden [1, L, D]; positions int32 [N] (rows of hidden predicting targets); targets int32 [N].
    Returns sum of NLL over the N targets (fp32). Logits chunked + checkpointed."""
    W = lm_head_weight(model)
    softcap = model.language_model.final_logit_softcapping

    def make_ce(y):
        y = mx.stop_gradient(y)

        def ce(h):  # targets captured as a constant: checkpoint VJPs only w.r.t. h
            logits = (h @ W.T).astype(mx.float32)
            if softcap is not None:
                logits = mx.tanh(logits / softcap) * softcap
            lse = mx.logsumexp(logits, axis=-1)
            tgt = mx.take_along_axis(logits, y[:, None], axis=-1)[:, 0]
            return (lse - tgt).sum()

        return mx.checkpoint(ce)

    hs = hidden[0][mx.stop_gradient(positions)]
    total = mx.array(0.0, dtype=mx.float32)
    N = positions.shape[0]
    for s in range(0, N, chunk):
        total = total + make_ce(targets[s:s + chunk])(hs[s:s + chunk])
    return total


def loss_positions(input_ids: np.ndarray, loss_mask: np.ndarray):
    """Target token t (t>=1) with loss_mask[t]==1 is predicted from hidden row t-1."""
    m = np.asarray(loss_mask[1:], dtype=bool)
    pos = np.nonzero(m)[0].astype(np.int32)
    tgt = np.asarray(input_ids[1:], dtype=np.int32)[pos]
    return pos, tgt


def read_jsonl(path):
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def mem_gb():
    return mx.get_active_memory() / 1e9, mx.get_peak_memory() / 1e9


# ----------------------------------------------------------------------------------
# Layer-wise backprop with explicit evaluation (bounded memory by construction)
# ----------------------------------------------------------------------------------

class LayerwiseTrainer:
    """loss_and_grads(ids, pos, tgt) -> (nll_sum, grads tree matching model.trainable_parameters()).

    1. no-grad forward, storing each layer's input h_i (bf16) and the K/V of KV-source layers;
    2. loss head chunk by chunk (logits only at loss positions), giving dL/dh_42;
    3. for i = 41..0: one mx.vjp over layer i (recompute + backward), eval, keep dL/dh_i.
    MLX's lazy graph never spans more than one layer, so peak = weights + stored h's + one layer.
    """

    def __init__(self, model, window: int | None = None, chunk: int = 2048, ce_chunk: int = 1024):
        self.model = model
        self.tm = text_model(model)
        self.window = window or self.tm.window_size
        self.chunk = chunk
        self.ce_chunk = ce_chunk
        n = len(self.tm.layers)
        self.n = n
        self.prev = list(self.tm.previous_kvs)
        self.kv_sources = {self.prev[j] for j in range(n) if self.prev[j] != j}
        W = lm_head_weight(model)
        self.W = W
        self.softcap = model.language_model.final_logit_softcapping

    def _layer_fn(self, i, names, pli, shared):
        layer = self.tm.layers[i]
        W, C = self.window, self.chunk
        returns_kv = i in self.kv_sources

        def f(*args):
            nP = len(names)
            params = tree_unflatten(list(zip(names, args[:nP])))
            h = args[nP]
            layer.update(params)
            if shared:
                k, v = args[nP + 1], args[nP + 2]
                out, _ = layer_forward(layer, h, pli, (k, v), W, C)
                return [out]
            out, (k, v) = layer_forward(layer, h, pli, None, W, C)
            return [out, k, v] if returns_kv else [out]

        return f

    def forward_states(self, x):
        tm = self.tm
        emb = tm.embed_tokens(x)
        h = emb * tm.embed_scale
        pli_all = None
        if tm.hidden_size_per_layer_input:
            pli_all = tm._project_per_layer_inputs(h, tm._get_per_layer_inputs(x, emb))
        mx.eval(h, pli_all)
        hs = [h]
        kvs = {}
        for i, layer in enumerate(tm.layers):
            pli = pli_all[:, :, i, :] if pli_all is not None else None
            shared = self.prev[i] != i
            out, kv = layer_forward(layer, h, pli, kvs[self.prev[i]] if shared else None, self.window, self.chunk)
            if i in self.kv_sources:
                kvs[i] = kv
                mx.eval(out, kv)
            else:
                mx.eval(out)
            h = out
            hs.append(h)
        return hs, kvs, pli_all

    def head(self, h_last, pos, tgt):
        """returns (nll_sum, dL/dh_last)."""
        tm = self.tm
        W, sc = self.W, self.softcap

        def sel(h):
            return tm.norm(h)[0][pos]

        hsel = sel(h_last)
        mx.eval(hsel)

        def ce(hc, y):
            logits = (hc @ W.T).astype(mx.float32)
            if sc is not None:
                logits = mx.tanh(logits / sc) * sc
            lse = mx.logsumexp(logits, axis=-1)
            t = mx.take_along_axis(logits, y[:, None], axis=-1)[:, 0]
            return (lse - t).sum()

        vg = mx.value_and_grad(ce, argnums=0)
        total = 0.0
        dsel = []
        N = int(pos.shape[0])
        for s in range(0, N, self.ce_chunk):
            l, g = vg(hsel[s:s + self.ce_chunk], tgt[s:s + self.ce_chunk])
            mx.eval(l, g)
            total += l.item()
            dsel.append(g)
        dsel = mx.concatenate(dsel, axis=0) if len(dsel) > 1 else dsel[0]
        _, (dh,) = mx.vjp(lambda h: [sel(h)], [h_last], [dsel])
        mx.eval(dh)
        return total, dh

    def loss(self, x, pos, tgt):
        """Forward-only NLL sum (validation). Keeps only the current hidden state."""
        tm = self.tm
        emb = tm.embed_tokens(x)
        h = emb * tm.embed_scale
        pli_all = None
        if tm.hidden_size_per_layer_input:
            pli_all = tm._project_per_layer_inputs(h, tm._get_per_layer_inputs(x, emb))
        mx.eval(h, pli_all)
        kvs = {}
        for i, layer in enumerate(tm.layers):
            pli = pli_all[:, :, i, :] if pli_all is not None else None
            shared = self.prev[i] != i
            h, kv = layer_forward(layer, h, pli, kvs[self.prev[i]] if shared else None, self.window, self.chunk)
            if i in self.kv_sources:
                kvs[i] = kv
                mx.eval(h, kv)
            else:
                mx.eval(h)
        hsel = tm.norm(h)[0][pos]
        W, sc = self.W, self.softcap
        total = 0.0
        for s in range(0, int(pos.shape[0]), self.ce_chunk):
            logits = (hsel[s:s + self.ce_chunk] @ W.T).astype(mx.float32)
            if sc is not None:
                logits = mx.tanh(logits / sc) * sc
            y = tgt[s:s + self.ce_chunk]
            l = (mx.logsumexp(logits, axis=-1) - mx.take_along_axis(logits, y[:, None], axis=-1)[:, 0]).sum()
            total += l.item()
        return total

    def loss_and_grads(self, x, pos, tgt):
        hs, kvs, pli_all = self.forward_states(x)
        total, dh = self.head(hs[-1], pos, tgt)
        hs[-1] = None
        dkv = {}
        layer_grads = [None] * self.n
        for i in reversed(range(self.n)):
            layer = self.tm.layers[i]
            flat = tree_flatten(layer.trainable_parameters())
            names = [k for k, _ in flat]
            orig = [v for _, v in flat]
            pli = pli_all[:, :, i, :] if pli_all is not None else None
            shared = self.prev[i] != i
            f = self._layer_fn(i, names, pli, shared)
            if shared:
                k, v = kvs[self.prev[i]]
                outs, vjps = mx.vjp(f, orig + [hs[i], k, v], [dh])
                gk, gv = vjps[-2].astype(mx.float32), vjps[-1].astype(mx.float32)
                src = self.prev[i]
                if src in dkv:
                    dkv[src] = (dkv[src][0] + gk, dkv[src][1] + gv)
                else:
                    dkv[src] = (gk, gv)
                mx.eval(dkv[src])
                dh_new = vjps[len(names)]
            elif i in self.kv_sources:
                k, v = kvs[i]
                dk, dv = dkv.pop(i, (mx.zeros_like(k), mx.zeros_like(v)))
                outs, vjps = mx.vjp(f, orig + [hs[i]], [dh, dk.astype(k.dtype), dv.astype(v.dtype)])
                dh_new = vjps[len(names)]
            else:
                outs, vjps = mx.vjp(f, orig + [hs[i]], [dh])
                dh_new = vjps[len(names)]
            g = vjps[:len(names)]
            mx.eval(g, dh_new)
            layer.update(tree_unflatten(list(zip(names, orig))))
            layer_grads[i] = tree_unflatten(list(zip(names, g)))
            dh = dh_new
            hs[i] = None
            del outs, vjps
        grads = {"language_model": {"model": {"layers": layer_grads}}}
        return total, grads
