"""Go/no-go analysis for the retry target (rule frozen in the weight-edit log, go/no-go entry of 2026-11-07).  Owner: Claude.

    outputs/small_model/mlx_venv/bin/python -I scripts/weight_edit/retry_gonogo_analyze.py

G1 (held-out retry calls, 8 draws: greedy + 7 sampled at the served settings):
  stuck call = at least 4 of 8 draws are byte-identical to a string in the call's failed set F(c);
  escapable  = stuck call with at least one draw outside F(c).
  value = mean over tasks of (mean over the task's escapable calls of acceptance among the draws outside F(c)).
  Pass: value >= 0.40 over at least 8 tasks.
G2 (linear probe): fit on training calls only, retry against first attempt, residual stream at the `open` position, per layer
  (35, 38, 40, 41): standardise, PCA to 64, ridge on +/-1 (lambda = n). Test on held-out calls.
  Pass: best AUC of the four layers >= 0.80. The best-of-four choice is disclosed; this is a go/no-go, not a claim.
Go = G1 and G2. Everything else printed here is descriptive.
"""
from __future__ import annotations

import collections
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/weight_edit/claude/retry_gonogo"
LAYERS = [35, 38, 40, 41]
POS = ["turn", "open", "mid"]


def g1(tag: str) -> dict:
    gen_p, sc_p = OUT / f"gen_{tag}_retry.jsonl", OUT / f"gen_{tag}_retry.scored.jsonl"
    if not sc_p.exists():
        return {"missing": str(sc_p)}
    failed = json.load(open(OUT / "failed_sets.json"))
    gens = {(g["key"], g["draw"]): g for g in map(json.loads, gen_p.open())}
    sc = collections.defaultdict(list)
    for x in map(json.loads, sc_p.open()):
        g = gens[(x["key"], x["draw"])]
        sc[x["key"]].append((g["gen"] in failed[x["key"]], x["accepted"], x["compiles"], x["draw"]))
    per_task = collections.defaultdict(list)
    per_task_comp = collections.defaultdict(list)
    n_stuck = n_esc = n_call = 0
    all_id = all_n = acc_non = n_non = 0
    greedy_id = 0
    for k, rows in sc.items():
        n_call += 1
        ids = sum(r[0] for r in rows)
        all_id += ids
        all_n += len(rows)
        greedy_id += sum(r[0] for r in rows if r[3] == 0)
        non = [r for r in rows if not r[0]]
        acc_non += sum(r[1] for r in non)
        n_non += len(non)
        if ids >= 4:
            n_stuck += 1
            if non:
                n_esc += 1
                t = k.split("|")[1]
                per_task[t].append(sum(r[1] for r in non) / len(non))
                per_task_comp[t].append(sum(1 for r in non if r[1] and r[2] != 0) / len(non))
    vals = [sum(v) / len(v) for v in per_task.values()]
    comp = [sum(v) / len(v) for v in per_task_comp.values()]
    return {
        "calls": n_call, "draws": all_n, "identical_share": all_id / all_n, "greedy_identical_share": greedy_id / n_call,
        "stuck_calls": n_stuck, "escapable_calls": n_esc, "tasks": len(vals),
        "value": float(np.mean(vals)) if vals else None,
        "value_accepted_and_compiling": float(np.mean(comp)) if comp else None,
        "all_non_identical_acceptance": acc_non / n_non if n_non else None,
    }


def auc(s: np.ndarray, y: np.ndarray) -> float:
    pos, neg = s[y == 1], s[y == 0]
    r = np.argsort(np.argsort(np.r_[pos, neg])) + 1.0
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def probe() -> dict:
    d = {k: np.load(OUT / f"features_{k}.npz") for k in ("train_retry", "train_first", "heldout_retry", "heldout_first")}
    Xtr = np.concatenate([d["train_retry"]["X"], d["train_first"]["X"]])
    ytr = np.r_[np.ones(len(d["train_retry"]["X"])), np.zeros(len(d["train_first"]["X"]))]
    Xte = np.concatenate([d["heldout_retry"]["X"], d["heldout_first"]["X"]])
    yte = np.r_[np.ones(len(d["heldout_retry"]["X"])), np.zeros(len(d["heldout_first"]["X"]))]
    res = {}
    for li, l in enumerate(LAYERS):
        for pi, p in enumerate(POS):
            a, b = Xtr[:, li, pi], Xte[:, li, pi]
            m, s = a.mean(0), a.std(0) + 1e-6
            a, b = (a - m) / s, (b - m) / s
            u, sv, vt = np.linalg.svd(a, full_matrices=False)
            v = vt[:64].T
            pa, pb = a @ v, b @ v
            w = np.linalg.solve(pa.T @ pa + len(pa) * np.eye(64), pa.T @ (2 * ytr - 1))
            res[f"L{l}_{p}"] = {"train_auc": auc(pa @ w, ytr), "heldout_auc": auc(pb @ w, yte)}
    return res


def main() -> None:
    held = g1("heldout")
    train = g1("train")
    report = {"g1_heldout": held, "g1_train_exploratory": train}
    try:
        pr = probe()
        report["probe"] = pr
        best = max(pr[f"L{l}_open"]["heldout_auc"] for l in LAYERS)
    except FileNotFoundError as e:
        report["probe_missing"] = str(e)
        best = None
    g1_ok = bool(held.get("value") is not None and held["value"] >= 0.40 and held["tasks"] >= 8)
    g2_ok = bool(best is not None and best >= 0.80)
    report.update({"g2_best_open_auc": best, "G1": g1_ok, "G2": g2_ok, "GO": g1_ok and g2_ok})
    (OUT / "gonogo_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
