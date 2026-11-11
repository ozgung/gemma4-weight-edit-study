"""Native-forward check of the proposers' teacher-forced log-prob claims.  Owner: Claude.

    outputs/small_model/mlx_venv/bin/python -I scripts/weight_edit/native_check.py RUN_DIR PASS [--out OUT.json]

The proposers' read passes used the forward-pass server's manual forward (wsp_server.forward_capture), which was
validated against the native forward on the base model only. This script takes the texts of one read pass that ran
under the run's FINAL edit and computes, for each text, the log-probs at the same positions four ways: manual and
native forward, base and edited weights. Native = mlx_lm's own model call with a KV cache (chunked prefill, then the
scored positions in one chunk), the same code path the generation runs used.

Reported per text: target log-prob agreement manual vs native (base and edit), top-1 agreement, full-row agreement
over tokens with native log-prob > -10 and > -25, and the edit-minus-base shift of the target log-prob under each
forward (overall and at the position of the lowest base target log-prob, i.e. the divergence / switch point).
Also: manual edit log-probs recomputed here vs those stored by the pass (checks that the final edit is the one that
pass used).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/weight_edit"))
sys.path.insert(0, str(ROOT / "scripts/mlx_lora"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("pass_", type=int)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--max-texts", type=int, default=0, help="smoke test: only the first N texts")
    a = ap.parse_args()
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache
    import wsp_server as S

    run = a.run.resolve()
    req = json.loads((run / "requests" / f"fwd_{a.pass_}.json.claimed").read_text())
    assert req["type"] == "read" and req.get("arm") == "edit", req.get("type")
    srv = S.Server()
    texts = []
    for ti, t in enumerate(req["texts"][: a.max_texts or None]):
        r = srv.calls[t["key"]]
        cont = r["old_string"] if t.get("continuation", "recorded") == "recorded" else str(t["continuation"])
        p_ids = srv.tok.encode(r["prompt"] + r["prefix"], add_special_tokens=False).ids
        c_ids = srv.tok.encode(cont, add_special_tokens=False).ids[: S.LIMITS["continuation_tokens"]]
        ids = p_ids + c_ids
        positions = list(range(len(p_ids) - 1, len(ids)))[: S.LIMITS["read_positions"]]
        targets = (c_ids + [srv.q_id])[: len(positions)]
        stored = np.load(run / "work" / f"fwd_{a.pass_}" / f"text_{ti}.npz")
        texts.append({"ti": ti, "key": t["key"], "ids": ids, "p0": len(p_ids), "positions": positions,
                      "targets": targets, "stored_lp": stored["logprobs"] if "logprobs" in stored else None})

    def manual(t):
        _, lp = srv.forward_capture(t["ids"], t["positions"], set(), {"logprobs"}, set())
        return np.array(lp.astype(mx.float32))

    def native(t):
        ids, p0, n = t["ids"], t["p0"], len(t["positions"])
        cache = make_prompt_cache(srv.model)
        for s in range(0, p0 - 1, 2048):
            srv.model(mx.array(ids[s:min(s + 2048, p0 - 1)])[None], cache=cache)
            mx.eval([c.state for c in cache])
        lg = srv.model(mx.array(ids[p0 - 1:p0 - 1 + n])[None], cache=cache)[0].astype(mx.float32)
        lp = lg - mx.logsumexp(lg, axis=-1, keepdims=True)
        return np.array(lp)

    res = {}
    for arm in ("base", "edit"):
        originals = srv.apply(run) if arm == "edit" else {}
        try:
            for t in texts:
                t0 = time.time()
                m, n = manual(t), native(t)
                idx = np.arange(len(t["targets"]))
                tg = np.array(t["targets"])
                t[f"{arm}_man"] = m[idx, tg]
                t[f"{arm}_nat"] = n[idx, tg]
                rowdiff = np.abs(m - n)
                t[f"{arm}_row10"] = float(rowdiff[n > -10].max())
                t[f"{arm}_row25"] = float(rowdiff[n > -25].max())
                t[f"{arm}_top1"] = float((m.argmax(-1) == n.argmax(-1)).mean())
                print(f"{arm} text {t['ti']}: {time.time() - t0:.1f}s, target max|man-nat| "
                      f"{np.abs(t[f'{arm}_man'] - t[f'{arm}_nat']).max():.3f}", flush=True)
        finally:
            srv.restore(originals)
    per = []
    for t in texts:
        dm, dn = t["edit_man"] - t["base_man"], t["edit_nat"] - t["base_nat"]
        j = int(np.argmin(t["base_nat"]))
        per.append({
            "ti": t["ti"], "key": t["key"], "n_positions": len(t["positions"]),
            "target_maxabs_man_vs_nat": {"base": float(np.abs(t["base_man"] - t["base_nat"]).max()),
                                         "edit": float(np.abs(t["edit_man"] - t["edit_nat"]).max())},
            "row_maxabs_lp_gt_-10": {"base": t["base_row10"], "edit": t["edit_row10"]},
            "row_maxabs_lp_gt_-25": {"base": t["base_row25"], "edit": t["edit_row25"]},
            "top1_agree": {"base": t["base_top1"], "edit": t["edit_top1"]},
            "shift_sum": {"manual": float(dm.sum()), "native": float(dn.sum())},
            "shift_maxabs_man_vs_nat": float(np.abs(dm - dn).max()),
            "divergence_pos": j, "base_lp_at_div": {"manual": float(t["base_man"][j]), "native": float(t["base_nat"][j])},
            "shift_at_div": {"manual": float(dm[j]), "native": float(dn[j])},
            "stored_vs_recomputed_manual_edit_maxabs": (float(np.abs(t["stored_lp"] - t["edit_man"]).max())
                                                        if t["stored_lp"] is not None else None),
        })
    agg = {
        "texts": len(per),
        "max_target_diff_base": max(p["target_maxabs_man_vs_nat"]["base"] for p in per),
        "max_target_diff_edit": max(p["target_maxabs_man_vs_nat"]["edit"] for p in per),
        "max_shift_diff": max(p["shift_maxabs_man_vs_nat"] for p in per),
        "mean_shift_at_div": {"manual": float(np.mean([p["shift_at_div"]["manual"] for p in per])),
                              "native": float(np.mean([p["shift_at_div"]["native"] for p in per]))},
        "max_stored_vs_recomputed": max((p["stored_vs_recomputed_manual_edit_maxabs"] or 0) for p in per),
    }
    out = a.out or (ROOT / "outputs/weight_edit/claude/native_check" / f"{run.name}_fwd{a.pass_}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"run": str(run), "pass": a.pass_, "aggregate": agg, "per_text": per}, indent=1))
    print(json.dumps(agg))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
