"""WP4 recipe arm: a scripted difference-of-means edit through the same WP11 harness.  Owner: Claude.

    outputs/small_model/mlx_venv/bin/python -I scripts/weight_edit/recipe.py RUN_DIR

Pre-specified algorithm (written 2026-11-02 UTC before any of its passes; the author had seen the dossier facts
and aggregate base-model fidelity numbers, not any edit outcome):
1. Calls (search set only, fixed seed): 16 not-found calls and 16 accepted-exact calls whose old_string spans
   at least 2 lines.
2. For each not-found call, the divergence token = the first token of the recorded old_string after its
   longest prefix that occurs in the true file state. "Fail" positions = the position predicting that token.
   "Correct" positions = positions predicting tokens inside the matched prefix of the same call (at most 8,
   the last before divergence) and positions predicting old_string tokens of accepted calls at the same
   relative depth (at most 8 per call).
3. Pass 1 (read, base): residual stream at layers 6, 12, 18, 24, 30, 36, 40, 41 for all 32 calls.
   Pass 2 (read, base): the same for 32 further calls (16 + 16) to have a held-out half.
4. Per layer: v = mean(h_fail) - mean(h_correct) on pass-1 calls; separation = AUC of projection onto v on
   pass-2 calls. Pick the layer with the best held-out AUC.
5. Edit: at the chosen layer l, remove the direction from what attention and the MLP write into the residual
   stream. Each writer's output passes through a post-norm with weight g before it is added, so the removed
   output-space direction is u = normalise(v / g_post) for that writer; delta W = -alpha u (u^T W) on o_proj and
   down_proj of layer l (rank 1 each).
6. Passes 3-6 (gen, paired base/edit, 64 search calls, 1 draw): alpha in {1, 2} at the best layer, then at the
   second-best layer with the better alpha, plus one 2-layer combination (best two layers, better alpha).
   Pick the candidate with the largest paired gain in accepted calls (ties: smaller norm). Passes 7-10 unused
   unless a pass failed. The final edit is written to RUN_DIR/edit/.
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/weight_edit"))
import wsp_kernel as K  # noqa: E402

SEARCH = ROOT / "outputs/weight_edit/shared/search"
LAYERS = [6, 12, 18, 24, 30, 36, 40, 41]


def log(run: Path, msg: str) -> None:
    with (run / "notebook.md").open("a") as f:
        f.write(f"- {time.strftime('%H:%M:%SZ', time.gmtime())} {msg}\n")
    print(msg, flush=True)


def request(run: Path, k: int, req: dict) -> dict:
    p = run / "requests" / f"fwd_{k}.json"
    p.write_text(json.dumps(req))
    done = run / "requests" / f"fwd_{k}.done.json"
    while not done.exists():
        time.sleep(10)
    time.sleep(1)
    d = json.loads(done.read_text())
    if d["status"] != "ok":
        raise RuntimeError(f"pass {k}: {d['status']} {d.get('error') or d.get('reason')}")
    return d


def tok():
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(ROOT / "data/models/gemma-4-e4b-it-qat-q4_0-unquantized/tokenizer.json"))


def divergence(old: str, content: str, T) -> int:
    """Token index of the first old_string token after the longest prefix found in the file."""
    lo, hi = 0, len(old)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if old[:mid] in content:
            lo = mid
        else:
            hi = mid - 1
    ids = T.encode(old, add_special_tokens=False).ids
    n = 0
    for i in range(len(ids)):
        if len(T.decode(ids[: i + 1], skip_special_tokens=False)) > lo:
            return i
        n = i
    return n


def pick(calls, rng, n_nf, n_ok, exclude=()):
    nf = [c for c in calls if c["error_kind"] == "not_found" and c["key"] not in exclude]
    ok = [c for c in calls if c["error_kind"] == "ok" and c["strategy"] == "exact" and c["old_string"].count("\n") >= 1
          and c["key"] not in exclude]
    rng.shuffle(nf)
    rng.shuffle(ok)
    return nf[:n_nf] + ok[:n_ok]


def positions_for(c, T, content):
    """(fail_rel, correct_rel): indices into the read pass's continuation positions (0 predicts token 0)."""
    n_tok = len(T.encode(c["old_string"], add_special_tokens=False).ids)
    if c["error_kind"] == "not_found":
        d = divergence(c["old_string"], content, T)
        return [d] if d < n_tok else [], list(range(max(0, d - 8), d))
    return [], list(range(min(n_tok, 8)))


def collect(run: Path, k: int, chosen, T):
    d = request(run, k, {"type": "read", "arm": "base", "texts": [{"key": c["key"], "continuation": "recorded"} for c in chosen],
                         "layers": LAYERS, "capture": ["resid"]})
    fails, oks = {l: [] for l in LAYERS}, {l: [] for l in LAYERS}
    for i, c in enumerate(chosen):
        z = np.load(run / "work" / f"fwd_{k}" / f"text_{i}.npz")
        content = (SEARCH / "files" / f"{c['file_sha']}.txt").read_text()
        fr, cr = positions_for(c, T, content)
        n = z["resid"].shape[1]
        for li, l in enumerate(LAYERS):
            fails[l] += [z["resid"][li, p] for p in fr if p < n]
            oks[l] += [z["resid"][li, p] for p in cr if p < n]
    return {l: (np.array(fails[l]), np.array(oks[l])) for l in LAYERS}


def auc(pos, neg):
    s = np.concatenate([pos, neg])
    r = s.argsort().argsort() + 1
    return (r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def build_edit(run: Path, layer_alphas: list[tuple[int, float]], dirs: dict) -> None:
    for f in (run / "edit").glob("*.npz"):
        f.unlink()
    for l, alpha in layer_alphas:
        v = dirs[l] / np.linalg.norm(dirs[l])
        for writer, norm in (("self_attn.o_proj", "post_attention_layernorm"), ("mlp.down_proj", "post_feedforward_layernorm")):
            name = f"model.language_model.layers.{l}.{writer}.weight"
            g = K.load(f"model.language_model.layers.{l}.{norm}.weight")
            u = v / np.where(np.abs(g) < 1e-3, 1e-3, g)
            u = (u / np.linalg.norm(u)).astype(np.float32)[:, None]
            W = K.load(name)
            K.save_edit(run, name, -alpha * u, u.T @ W, note=f"recipe: remove dom direction, layer {l}, alpha {alpha}")


def main() -> None:
    run = Path(sys.argv[1]).resolve()
    T = tok()
    calls = [json.loads(l) for l in (SEARCH / "calls.jsonl").open()]
    rng = random.Random(20261102)
    a = pick(calls, rng, 16, 16)
    b = pick(calls, rng, 16, 16, exclude={c["key"] for c in a})
    log(run, f"recipe start: pass-1 calls {len(a)}, pass-2 calls {len(b)}")
    A = collect(run, 1, a, T)
    B = collect(run, 2, b, T)
    dirs, aucs = {}, {}
    for l in LAYERS:
        (fa, oa), (fb, ob) = A[l], B[l]
        if len(fa) < 3 or len(oa) < 3 or len(fb) < 3 or len(ob) < 3:
            continue
        v = fa.mean(0) - oa.mean(0)
        dirs[l] = v
        aucs[l] = float(auc(fb @ v, ob @ v))
    log(run, "held-out AUC per layer: " + json.dumps({l: round(x, 3) for l, x in aucs.items()}))
    order = sorted(aucs, key=lambda l: -aucs[l])
    best, second = order[0], order[1]
    gen_keys = sorted(c["key"] for c in calls)
    random.Random(7).shuffle(gen_keys)
    gen_keys = gen_keys[:64]
    cands = [[(best, 1.0)], [(best, 2.0)]]
    results = []
    k = 3
    for cand in cands + [None, None]:
        if cand is None:  # derived candidates after the alpha comparison
            ba = max(results[:2], key=lambda r: r[1])[0][0][1]
            cand = [(second, ba)] if len(results) == 2 else [(best, ba), (second, ba)]
        build_edit(run, cand, dirs)
        d = request(run, k, {"type": "gen", "arms": ["base", "edit"], "calls": gen_keys, "draws": 1})
        p = d["result"]["paired_accepted"]
        gain = p["only_edit"] - p["only_base"]
        norm = sum(e["delta_fro"] for e in K.list_edits(run))
        results.append((cand, gain, norm, p, d["result"]["arms"]["edit"]["accepted"], d["result"]["arms"]["base"]["accepted"]))
        log(run, f"pass {k}: candidate {cand} paired {p} gain {gain} edit acc {results[-1][4]:.3f} base acc {results[-1][5]:.3f}")
        k += 1
    final = max(results, key=lambda r: (r[1], -r[2]))
    build_edit(run, final[0], dirs)
    (run / "final_answer.md").write_text(
        "# Recipe arm final answer\n\n"
        f"Final edit: {final[0]} (layer, alpha); rank-1 removal of the difference-of-means direction (first wrong copy "
        "token vs correct copy tokens) from o_proj and down_proj outputs of the layer(s), mapped through the post-norm "
        f"weights. Held-out AUC per layer: {json.dumps({l: round(x, 3) for l, x in aucs.items()})}.\n\n"
        "Per-pass record:\n\n" + "\n".join(f"- {r[0]}: paired {r[3]}, gain {r[1]}, edit acc {r[4]:.3f}, base acc {r[5]:.3f}, "
                                           f"sum |dW| {r[2]:.3f}" for r in results) + "\n")
    log(run, f"final: {final[0]} gain {final[1]}")


if __name__ == "__main__":
    main()
