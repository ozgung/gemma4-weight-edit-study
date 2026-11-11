"""Native re-measure of Opus's line-end escape-margin claim (ledger D4).  Owner: Claude.

    outputs/small_model/mlx_venv/bin/python -I scripts/weight_edit/native_margin_check.py [--max-texts N]

Criterion fixed before the run: docs/experiments/weight-edit-collab-20261102.md, entry 2026-11-07 11:01 UTC.
Texts: opus_p10_1 pass 9 (32 texts, the final edit's read). Positions: Opus's pass-1 switch points
(scratch/pass1_meta.json "div", as used by its scratch/anal3.py). Margin m = lp(escape token) - lp(right token);
shrinkage dm = m_base - m_edit. Forwards: native (mlx_lm with a KV cache, as in native_check.py) and manual
(wsp_server.forward_capture), base weights and the run's final edit.
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
RUN = ROOT / "outputs/weight_edit/proposers/opus_p10_1"
LINE_END = [0, 1, 4, 5, 9, 10, 11, 13, 16]
QUOTE = [2, 3, 6, 7, 8, 14, 15, 17, 18]
CTRL = list(range(19, 26))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-texts", type=int, default=0, help="smoke test: only the first N texts")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/weight_edit/claude/native_check/opus_p10_1_margin.json")
    a = ap.parse_args()
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache
    import wsp_server as S

    req = json.loads((RUN / "requests" / "fwd_9.json.claimed").read_text())
    assert req["type"] == "read" and req.get("arm") == "edit"
    m3 = json.loads((RUN / "scratch" / "pass3_meta.json").read_text())
    m1 = {m["i"]: m for m in json.loads((RUN / "scratch" / "pass1_meta.json").read_text())}
    assert len(m3) == len(req["texts"]) and all(m["key"] == t["key"] for m, t in zip(m3, req["texts"])
                                                if m["grp"] != "C_syn"), "pass-9 texts differ from pass3_meta"
    srv = S.Server()
    tok = srv.tok
    texts = []
    for ti, t in enumerate(req["texts"][: a.max_texts or None]):
        r = srv.calls[t["key"]]
        cont = r["old_string"] if t.get("continuation", "recorded") == "recorded" else str(t["continuation"])
        p_ids = tok.encode(r["prompt"] + r["prefix"], add_special_tokens=False).ids
        c_ids = tok.encode(cont, add_special_tokens=False).ids[: S.LIMITS["continuation_tokens"]]
        ids = p_ids + c_ids
        positions = list(range(len(p_ids) - 1, len(ids)))[: S.LIMITS["read_positions"]]
        targets = (c_ids + [srv.q_id])[: len(positions)]
        meta = m3[ti]
        info = {"ti": ti, "key": t["key"], "grp": meta["grp"], "ids": ids, "p0": len(p_ids), "positions": positions,
                "targets": targets, "stored_edit": np.load(RUN / "work" / "fwd_9" / f"text_{ti}.npz")["logprobs"]}
        if meta["src"] == "p1":
            mm = m1[meta["i"]]
            info["stored_base"] = np.load(RUN / "work" / "fwd_1" / f"text_{meta['i']}.npz")["logprobs"]
            if mm["div"] is not None:
                d = mm["div"]
                info["div"], info["right_tok"], info["wrong_tok"] = d, mm["right_tok"], mm["wrong_tok"]
                info["right_id"], info["wrong_id"] = targets[d], tok.token_to_id(mm["wrong_tok"])
                assert tok.id_to_token(targets[d]) == mm["right_tok"], (ti, tok.id_to_token(targets[d]), mm["right_tok"])
                assert info["wrong_id"] is not None, mm["wrong_tok"]
        texts.append(info)

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
        return np.array(lg - mx.logsumexp(lg, axis=-1, keepdims=True))

    for arm in ("base", "edit"):
        originals = srv.apply(RUN) if arm == "edit" else {}
        try:
            for t in texts:
                t0 = time.time()
                idx, tg = np.arange(len(t["targets"])), np.array(t["targets"])
                for fw, fn in (("man", manual), ("nat", native)):
                    row = fn(t)
                    t[f"{arm}_{fw}_tgt"] = row[idx, tg]
                    if "div" in t:
                        d = t["div"]
                        t[f"{arm}_{fw}_right"] = float(row[d, t["right_id"]])
                        t[f"{arm}_{fw}_wrong"] = float(row[d, t["wrong_id"]])
                        t[f"{arm}_{fw}_top1"] = tok.id_to_token(int(row[d].argmax()))
                print(f"{arm} text {t['ti']}: {time.time() - t0:.1f}s", flush=True)
        finally:
            srv.restore(originals)

    per = []
    for t in texts:
        p = {"ti": t["ti"], "key": t["key"], "grp": t["grp"],
             "stored_vs_recomputed_manual": {
                 "edit": float(np.abs(t["stored_edit"] - t["edit_man_tgt"]).max()),
                 "base": float(np.abs(t["stored_base"] - t["base_man_tgt"]).max()) if "stored_base" in t else None},
             "dsum_lp": {fw: float(t[f"edit_{fw}_tgt"].sum() - t[f"base_{fw}_tgt"].sum()) for fw in ("man", "nat")}}
        if "div" in t:
            p.update({"div": t["div"], "right_tok": t["right_tok"], "wrong_tok": t["wrong_tok"]})
            for fw in ("man", "nat"):
                mb = t[f"base_{fw}_wrong"] - t[f"base_{fw}_right"]
                me = t[f"edit_{fw}_wrong"] - t[f"edit_{fw}_right"]
                p[fw] = {"lp_right": [t[f"base_{fw}_right"], t[f"edit_{fw}_right"]],
                         "lp_wrong": [t[f"base_{fw}_wrong"], t[f"edit_{fw}_wrong"]],
                         "margin": [mb, me], "dm": mb - me,
                         "top1": [t[f"base_{fw}_top1"], t[f"edit_{fw}_top1"]]}
        per.append(p)

    def stat(ids, fw):
        v = [q[fw]["dm"] for q in per if q["ti"] in ids and fw in q]
        return {"n": len(v), "mean": float(np.mean(v)) if v else None, "median": float(np.median(v)) if v else None,
                "min": float(np.min(v)) if v else None, "max": float(np.max(v)) if v else None}

    le = {fw: stat(LINE_END, fw) for fw in ("man", "nat")}
    verdict = None
    if le["nat"]["n"] == len(LINE_END):
        nm, mm_ = le["nat"]["mean"], le["man"]["mean"]
        within = np.sign(nm) == np.sign(mm_) and abs(nm - mm_) <= 0.25 * abs(mm_)
        if 2.5 <= le["nat"]["median"] <= 6 and within:
            verdict = "reproduced as stated"
        elif nm > 0.6:
            verdict = "direction only"
        else:
            verdict = "not reproduced"
    agg = {
        "line_end_dm": le, "quote_dm": {fw: stat(QUOTE, fw) for fw in ("man", "nat")},
        "verdict": verdict,
        "paren_flip_native": next(([q["nat"]["top1"], q["right_tok"]] for q in per if q["ti"] == 1 and "nat" in q), None),
        "ctrl_max_abs_dsum_native": max((abs(q["dsum_lp"]["nat"]) for q in per if q["ti"] in CTRL), default=None),
        "ctrl_max_abs_dsum_manual": max((abs(q["dsum_lp"]["man"]) for q in per if q["ti"] in CTRL), default=None),
        "max_stored_vs_recomputed": {
            "edit": max(q["stored_vs_recomputed_manual"]["edit"] for q in per),
            "base": max((q["stored_vs_recomputed_manual"]["base"] or 0) for q in per)},
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"run": str(RUN), "pass": 9, "aggregate": agg, "per_text": per}, indent=1))
    print(json.dumps(agg))
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
