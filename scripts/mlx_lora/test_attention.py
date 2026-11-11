"""fp32 unit test: blocked sliding-window and chunked global attention vs reference masked SDPA
(values and gradients)."""
import sys
from pathlib import Path
import mlx.core as mx
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common

def ref(q, k, v, window=None):
    L = q.shape[2]
    r = mx.arange(L)
    m = r[:, None] >= r[None]
    if window:
        m = m & (r[:, None] < r[None] + window)
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0, mask=m)

mx.random.seed(0)
worst = 0.0
for (L, D, kind) in [(1500, 256, "sliding"), (1024, 256, "sliding"), (513, 256, "sliding"), (2100, 512, "global"), (4096, 512, "global")]:
    q = mx.random.normal((1, 8, L, D)) * 0.1
    k = mx.random.normal((1, 2, L, D)) * 0.1
    v = mx.random.normal((1, 2, L, D))
    g = mx.random.normal((1, 8, L, D))
    if kind == "sliding":
        f_new = lambda q, k, v: common.sliding_attention(q, k, v, 512)
        f_ref = lambda q, k, v: ref(q, k, v, 512)
    else:
        f_new = lambda q, k, v: common.global_attention(q, k, v, chunk=512)
        f_ref = lambda q, k, v: ref(q, k, v)
    o1, o2 = f_new(q, k, v), f_ref(q, k, v)
    gn = mx.grad(lambda q, k, v: (f_new(q, k, v) * g).sum(), argnums=(0, 1, 2))(q, k, v)
    gr = mx.grad(lambda q, k, v: (f_ref(q, k, v) * g).sum(), argnums=(0, 1, 2))(q, k, v)
    d = mx.abs(o1 - o2).max().item()
    dg = max((mx.abs(a - b).max() / (mx.abs(b).max() + 1e-12)).item() for a, b in zip(gn, gr))
    worst = max(worst, d, dg)
    print(f"{kind:8s} L={L} D={D}: max|out diff|={d:.2e} max rel grad diff={dg:.2e}")
print("PASS" if worst < 1e-4 else "FAIL", worst)
