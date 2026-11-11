"""Peak memory of attention pieces (fwd+bwd) in isolation, bf16."""
import sys, time
from pathlib import Path
import mlx.core as mx
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common

def measure(name, f, *xs):
    g = mx.random.normal(f(*xs).shape).astype(mx.bfloat16); mx.eval(g)
    vg = mx.grad(lambda *a: (f(*a).astype(mx.float32) * g).sum(), argnums=tuple(range(len(xs))))
    mx.clear_cache(); mx.reset_peak_memory(); base = mx.get_active_memory()
    t = time.time(); out = vg(*xs); mx.eval(out); dt = time.time() - t
    print(f"{name}: {dt:.2f}s peak+{(mx.get_peak_memory()-base)/1e9:.2f}GB", flush=True)

L = int(sys.argv[1])
for D, kind in ((512, "global"), (256, "sliding")):
    q = (mx.random.normal((1, 8, L, D)) * 0.1).astype(mx.bfloat16)
    k = (mx.random.normal((1, 2, L, D)) * 0.1).astype(mx.bfloat16)
    v = mx.random.normal((1, 2, L, D)).astype(mx.bfloat16)
    mx.eval(q, k, v)
    if kind == "global":
        for C in (1024, 2048, 4096):
            measure(f"global D=512 L={L} chunk={C}", lambda q, k, v: common.global_attention(q, k, v, chunk=C), q, k, v)
    else:
        measure(f"sliding D=256 L={L}", lambda q, k, v: common.sliding_attention(q, k, v, 512), q, k, v)
