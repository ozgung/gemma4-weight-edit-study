"""What an edit does at each token: layer-41 MLP activations on the round-2 gate calls, and the exact effect of
same-size changes of one matrix on the next-token distribution.  Owner: Claude (review session), 2026-11-12.

    outputs/small_model/mlx_venv/bin/python -I scripts/paper/activation_capture.py [--n 120] [--out DIR]

DESCRIPTIVE, POST HOC (for a paper figure; no registered verdict depends on it). Local E4B, bf16, MLX, through the
weightprog backend (capture and exact tail recomputation, validated on E4B: weightprog docs/real-model-validation.md,
checks 3 and 4).

Changes compared, all on model.language_model.layers.41.mlp.down_proj.weight, rank 2, Frobenius norm 5.06:
  fable   the frozen Fable round-2 edit (authored in closed form; served live)
  trained G-yer-39, the gradient-trained adapter at the same module, rank and norm
  rand1, rand2  the two size-matched random controls of the gate

For each selected call the base model's own greedy old text (gate_r2/gen_base.jsonl, draw 0) is appended to the frozen
prompt and the model is run once (teacher forcing). At every position that predicts a token of the call's own text
(old text, and the prefix after the last tool response) we take the activations entering down_proj (10,240 values) and
recompute the tail of the last layer with and without each change. Because layer 41 is the last layer, a change of
down_proj acts only on the position's own next-token distribution, and the recomputation is exact up to bf16 rounding.

Per position and change: the two rank components A h ("detector" values), |dW h| / |W h|, KL(base || changed) over the
whole vocabulary, the change in log-probability of the token the base model wrote, and whether the top token changes.
Position labels (old text only): `escape` = the predicted token starts a literal \\n or \\" that the file does not
contain anywhere (the gate's artefact rule, applied per token); `other` = every other old-text position.
Also: the changed model's top token at every position, and the mean change of probability and of log-probability
over escape positions for every vocabulary entry (which tokens each change moves there).

Calls: every usable gate call whose base greedy old text carries an artefact, plus a seeded sample of the others
(at most 2 per task) up to --n. Output: positions.npz, tokens.jsonl (old-text tokens with their values), summary.json.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT.parent / "weightprog" / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "weight_edit"))

from weightprog.construct.logit_steer import TailState, edit_delta, gate_values, rms_norm  # noqa: E402
from weightprog.edits import EditSet  # noqa: E402
from weightprog.evaluate import open_model  # noqa: E402

MODEL = ROOT / "data/models/gemma-4-e4b-it-qat-q4_0-unquantized"
R = ROOT / "outputs/weight_edit/claude/replay_ood"
G2 = ROOT / "outputs/weight_edit/claude/gate_r2"
EDITS = {
    "fable": ROOT / "outputs/weight_edit/proposers/fable_r2/edit/layers_41_mlp_down_proj.npz",
    "trained": ROOT / "outputs/weight_edit/claude/edit_vs_train/arms/G-yer-39/edit/layers_41_mlp_down_proj.npz",
    "rand1": G2 / "fable_r2_rand1/edit/layers_41_mlp_down_proj.npz",
    "rand2": G2 / "fable_r2_rand2/edit/layers_41_mlp_down_proj.npz",
}
LAYER = 41
ESC = ("\\n", '\\"')
PRE = "model.language_model."


def task_of(key: str) -> str:
    return key.split("|")[1]


def escape_tokens(gen: str, offsets: list, file_text: str) -> list[int]:
    """Indices of tokens that hold the backslash of a literal \\n or \\" whose two-character form is absent from the file."""
    starts = []
    for e in ESC:
        if e in file_text:
            continue
        i = gen.find(e)
        while i != -1:
            starts.append(i)
            i = gen.find(e, i + 1)
    return sorted({t for t, (a, b) in enumerate(offsets) for s in starts if a <= s < b})


def log_softmax(x: np.ndarray) -> np.ndarray:
    m = x.max(-1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(-1, keepdims=True))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--seed", type=int, default=20261112)
    ap.add_argument("--max-prefix", type=int, default=400, help="at most this many prefix positions per call")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/paper/figures/activations")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    want = [l.strip() for l in (R / "usable_keys.txt").read_text().splitlines() if l.strip()]
    calls = {r["key"]: r for r in map(json.loads, (R / "calls.jsonl").open())}
    base = {r["key"]: r["gen"] for r in map(json.loads, (G2 / "gen_base.jsonl").open()) if r["draw"] == 0}
    files = {}

    def file_text(k: str) -> str:
        sha = calls[k]["file_sha"]
        if sha not in files:
            files[sha] = (R / "files" / f"{sha}.txt").read_text()
        return files[sha]

    art = [k for k in want if any(e in base[k] and e not in file_text(k) for e in ESC)]
    rest = sorted(k for k in want if k not in set(art))
    random.Random(a.seed).shuffle(rest)
    per_task: dict[str, int] = {}
    picked = list(art)
    for k in rest:
        if len(picked) >= a.n:
            break
        if per_task.get(task_of(k), 0) < 2:
            per_task[task_of(k)] = per_task.get(task_of(k), 0) + 1
            picked.append(k)
    print(f"calls: {len(picked)} ({len(art)} with an artefact in the base greedy old text, {len(picked) - len(art)} others)", flush=True)

    t0 = time.time()
    be = open_model(MODEL, backend="mlx", dtype="bfloat16")
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(MODEL / "tokenizer.json"))
    edits = {n: next(iter(EditSet.load(p))) for n, p in EDITS.items()}
    for n, e in edits.items():
        assert e.module.endswith(f"layers.{LAYER}.mlp.down_proj.weight"), e.module
        print(f"  {n}: rank {e.A.shape[0]}, |BA| = {np.linalg.norm(e.B.astype(np.float64) @ e.A.astype(np.float64)):.3f}", flush=True)
    w = TailState.weights_from_backend(be, LAYER)
    E = np.asarray(be.read_weight(PRE + "embed_tokens.weight"), dtype=np.float32)       # [V, H], tied unembedding
    cap_c = be.spec.softcap
    print(f"model and weights loaded {time.time() - t0:.0f}s; vocabulary {E.shape[0]}, softcap {cap_c}", flush=True)

    def full_logits(st: TailState, delta=None) -> np.ndarray:
        hn = rms_norm(st.final_hidden(delta), st.path.w_final, st.path.eps).astype(np.float32)
        lg = hn @ E.T
        return np.tanh(lg / cap_c) * cap_c if cap_c else lg

    cols = {k: [] for k in ("call", "pos_kind", "label", "tok_index", "target", "base_top1", "base_lp_target")}
    per = {n: {k: [] for k in ("gate", "rel", "kl", "dlp_target", "top1_changed", "top1_new")} for n in edits}
    shift_sum = {n: np.zeros(E.shape[0], dtype=np.float64) for n in edits}
    pshift_sum = {n: np.zeros(E.shape[0], dtype=np.float64) for n in edits}
    p0_sum = np.zeros(E.shape[0], dtype=np.float64)
    n_escape = 0
    tokens_out = (a.out / "tokens.jsonl").open("w")
    q_id = tok.token_to_id('<|"|>')

    for ci, k in enumerate(picked):
        r = calls[k]
        t1 = time.time()
        pre_ids = tok.encode(r["prompt"] + r["prefix"], add_special_tokens=False).ids
        enc = tok.encode(base[k], add_special_tokens=False)
        gen_ids, offs = enc.ids, enc.offsets
        if not gen_ids:
            continue
        ids = pre_ids + gen_ids
        n0 = len(pre_ids)
        esc = set(escape_tokens(base[k], offs, file_text(k)))
        # prefix positions: the call's own text after the last tool response (the model wrote it in this turn)
        n_pref = len(tok.encode(r["prefix"], add_special_tokens=False).ids)
        pref_lo = max(n0 - n_pref, n0 - a.max_prefix)
        pos_pref = list(range(pref_lo, n0 - 1))                 # predicts ids[p + 1], all inside the prefix
        pos_old = [n0 - 1 + i for i in range(len(gen_ids))]     # predicts gen_ids[i]
        pos = pos_pref + pos_old
        st = TailState.from_backend(be, ids, pos, LAYER, weights=w)
        targets = np.array([ids[p + 1] for p in pos])
        kinds = ["prefix"] * len(pos_pref) + ["old"] * len(pos_old)
        labels = ["prefix"] * len(pos_pref) + [("escape" if i in esc else "other") for i in range(len(gen_ids))]
        mo = np.linalg.norm(st.mlp_out(), axis=-1)
        rows = {n: {} for n in edits}
        lp0_all, top0_all = [], []
        for s in range(0, len(pos), 96):
            sl = slice(s, s + 96)
            sub = TailState(st.layer, st.is_last, st.resid_mid[sl], st.act[sl], None if st.pli is None else st.pli[sl],
                            st.W_down, st.path, st.pl_gate, st.pl_proj, st.pl_norm)
            lp0 = log_softmax(full_logits(sub))
            p0 = np.exp(lp0)
            tg = targets[sl]
            ar = np.arange(len(tg))
            lp0_all.append(lp0[ar, tg])
            top0 = lp0.argmax(-1)
            top0_all.append(top0)
            lab = np.array(labels[sl])
            if (lab == "escape").any():
                p0_sum += p0[lab == "escape"].sum(0)
            for n, e in edits.items():
                d = edit_delta(sub, e)
                lg1 = full_logits(sub, d)
                lp1 = log_softmax(lg1)
                o = rows[n]
                o.setdefault("gate", []).append(gate_values(sub, e))
                o.setdefault("rel", []).append(np.linalg.norm(d, axis=-1) / mo[sl])
                o.setdefault("kl", []).append((p0 * (lp0 - lp1)).sum(-1))
                o.setdefault("dlp_target", []).append(lp1[ar, tg] - lp0[ar, tg])
                top1 = lp1.argmax(-1)
                o.setdefault("top1_changed", []).append(top1 != top0)
                o.setdefault("top1_new", []).append(top1)
                if (lab == "escape").any():
                    shift_sum[n] += (lp1[lab == "escape"] - lp0[lab == "escape"]).sum(0)
                    pshift_sum[n] += (np.exp(lp1[lab == "escape"]) - p0[lab == "escape"]).sum(0)
            n_escape += int((lab == "escape").sum())
        lp0_t, top0 = np.concatenate(lp0_all), np.concatenate(top0_all)
        for n in edits:
            for kk, v in rows[n].items():
                per[n][kk].append(np.concatenate(v))
        cols["call"] += [ci] * len(pos)
        cols["pos_kind"] += kinds
        cols["label"] += labels
        cols["tok_index"] += list(range(len(pos_pref))) + list(range(len(gen_ids)))
        cols["target"] += targets.tolist()
        cols["base_top1"] += top0.tolist()
        cols["base_lp_target"] += lp0_t.tolist()
        no = len(pos_pref)
        tokens_out.write(json.dumps({
            "key": k, "call": ci, "n_escape": len(esc),
            "tokens": [base[k][a_:b_] for a_, b_ in offs], "label": labels[no:],
            **{f"kl_{n}": np.round(np.concatenate(rows[n]["kl"])[no:], 5).tolist() for n in edits},
            **{f"gate_{n}": np.round(np.concatenate(rows[n]["gate"])[no:], 4).tolist() for n in edits},
            **{f"top1_{n}": [tok.id_to_token(int(v)) for v in np.concatenate(rows[n]["top1_new"])[no:]] for n in ("fable", "trained")},
            "base_top1_is_target": (top0[no:] == targets[no:]).tolist()}) + "\n")
        tokens_out.flush()
        print(f"[{ci + 1}/{len(picked)}] {k} prompt {n0} old {len(gen_ids)} prefix {len(pos_pref)} escape {len(esc)} "
              f"{time.time() - t1:.1f}s", flush=True)
    tokens_out.close()

    label = np.array(cols["label"])
    out = {"call": np.array(cols["call"]), "label": label, "pos_kind": np.array(cols["pos_kind"]),
           "tok_index": np.array(cols["tok_index"]), "target": np.array(cols["target"]),
           "base_top1": np.array(cols["base_top1"]), "base_lp_target": np.array(cols["base_lp_target"]),
           "keys": np.array(picked)}
    for n in edits:
        for kk, v in per[n].items():
            out[f"{n}_{kk}"] = np.concatenate(v)
        out[f"{n}_mean_shift_escape"] = (shift_sum[n] / max(1, n_escape)).astype(np.float32)
        out[f"{n}_mean_pshift_escape"] = (pshift_sum[n] / max(1, n_escape)).astype(np.float32)
    out["base_mean_p_escape"] = (p0_sum / max(1, n_escape)).astype(np.float32)
    np.savez_compressed(a.out / "positions.npz", **out)

    old = out["pos_kind"] == "old"
    summ = {"note": "descriptive, post hoc; see scripts/paper/activation_capture.py", "calls": len(picked),
            "calls_with_base_artefact": len(art), "positions": {s: int((label == s).sum()) for s in ("escape", "other", "prefix")},
            "base_top1_equals_written_token_old_text": float((out["base_top1"][old] == out["target"][old]).mean()),
            "closing_quote_id": q_id, "changes": {}}
    for n in edits:
        c = {}
        for s in ("escape", "other", "prefix"):
            m = label == s
            kl = out[f"{n}_kl"][m]
            c[s] = {"n": int(m.sum()), "kl_median": float(np.median(kl)), "kl_mean": float(kl.mean()),
                    "kl_p90": float(np.quantile(kl, 0.9)), "kl_over_0.1": float((kl > 0.1).mean()),
                    "top1_changed": float(out[f"{n}_top1_changed"][m].mean()),
                    "rel_write_median": float(np.median(out[f"{n}_rel"][m])),
                    "dlp_written_token_median": float(np.median(out[f"{n}_dlp_target"][m]))}
        sh = out[f"{n}_mean_pshift_escape"]
        order = np.argsort(sh)
        c["prob_down"] = [[tok.id_to_token(int(i)), round(float(sh[i]), 4), round(float(out["base_mean_p_escape"][i]), 4)] for i in order[:10]]
        c["prob_up"] = [[tok.id_to_token(int(i)), round(float(sh[i]), 4), round(float(out["base_mean_p_escape"][i]), 4)] for i in order[::-1][:10]]
        m = label == "escape"
        vals, cnt = np.unique(out[f"{n}_top1_new"][m], return_counts=True)
        c["top1_at_escape"] = [[tok.id_to_token(int(v)), int(k)] for v, k in sorted(zip(vals, cnt), key=lambda z: -z[1])[:8]]
        summ["changes"][n] = c
    (a.out / "summary.json").write_text(json.dumps(summ, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in summ.items() if k != "changes"}, indent=1))
    for n, c in summ["changes"].items():
        print(n, {s: {kk: round(vv, 4) for kk, vv in c[s].items()} for s in ("escape", "other", "prefix")})
    print(f"done {time.time() - t0:.0f}s -> {a.out}")


if __name__ == "__main__":
    main()
