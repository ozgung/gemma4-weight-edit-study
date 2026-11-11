"""Microbenchmark: memory/time of MLX SDPA forward+backward at long sequence lengths."""
import sys, time
import mlx.core as mx

def run(L, mask_kind, H=8, KVH=2, D=256, window=512):
    mx.random.seed(0)
    q = mx.random.normal((1, H, L, D)).astype(mx.bfloat16)
    k = mx.random.normal((1, KVH, L, D)).astype(mx.bfloat16)
    v = mx.random.normal((1, KVH, L, D)).astype(mx.bfloat16)
    g = mx.random.normal((1, H, L, D)).astype(mx.bfloat16)
    if mask_kind == "causal":
        mask = "causal"
    else:
        r = mx.arange(L)
        mask = (r[:, None] >= r[None]) & (r[:, None] < r[None] + window)
    mx.eval(q, k, v, g)
    def f(q, k, v):
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0, mask=mask)
        return (o.astype(mx.float32) * g.astype(mx.float32)).sum()
    vg = mx.value_and_grad(f, argnums=(0, 1, 2))
    mx.clear_cache(); mx.reset_peak_memory()
    base = mx.get_active_memory()
    t = time.time()
    val, grads = vg(q, k, v)
    mx.eval(val, grads)
    dt = time.time() - t
    peak = (mx.get_peak_memory() - base) / 1e9
    # forward only
    mx.clear_cache(); mx.reset_peak_memory()
    t = time.time()
    o = mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0, mask=mask); mx.eval(o)
    dtf = time.time() - t
    peakf = (mx.get_peak_memory() - base) / 1e9
    print(f"L={L} mask={mask_kind}: fwd+bwd {dt:.2f}s peak+{peak:.2f}GB | fwd {dtf:.3f}s peak+{peakf:.2f}GB", flush=True)

if __name__ == "__main__":
    print("mlx", mx.__version__)
    for L in [int(x) for x in sys.argv[1].split(",")]:
        for mk in sys.argv[2].split(","):
            run(L, mk)
