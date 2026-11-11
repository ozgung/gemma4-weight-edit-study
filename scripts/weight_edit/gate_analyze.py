"""WP7 gate analysis on the selection set, exactly as pre-registered (doc Thread, 2026-11-02 UTC).  Owner: Claude.

    python3 -I scripts/weight_edit/gate_analyze.py CAND [CAND...]      (family for Holm = all CANDs given)

Inputs in outputs/weight_edit/claude/gate/: gen_<name>.scored.jsonl for base, each candidate <c>, and its
controls <c>_rand1, <c>_rand2. Written before any selection result was inspected.

Rule:
1. Per selection task, mean acceptance over its fixed calls x draws in each arm; paired per-task differences.
   Required against base and each control: positive task-macro mean difference AND one-sided exact sign test
   on the non-zero task differences, p < 0.05.
2. p_candidate = max(p_base, p_rand1, p_rand2); Holm across the family of candidates given on the command line.
3. Served-ok loss rule, per draw: among calls recorded as served-ok, at most 2 calls accepted by base and
   rejected by the edit (affected tasks reported). A missing generation counts as rejected.
4. Compile guard: compile-fail rate among accepted edits <= base + 2 percentage points; also reported: the
   fixed-denominator rate of accepted-and-compiling edits; unknown compile states reported separately.
5. (Export check: PEFT export verifies and loads on the W4A16 vLLM smoke; done separately for passing edits.)
Call-micro effects are reported alongside.
"""

from __future__ import annotations

import collections
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
G = ROOT / "outputs/weight_edit/claude/gate"


def load(name: str) -> dict:
    out = {}
    for line in (G / f"gen_{name}.scored.jsonl").open():
        r = json.loads(line)
        out[(r["key"], r["draw"])] = r
    return out


def sign_test_one_sided(diffs: list[float]) -> tuple[float, int, int]:
    pos = sum(1 for d in diffs if d > 0)
    neg = sum(1 for d in diffs if d < 0)
    n = pos + neg
    if n == 0:
        return 1.0, pos, neg
    p = sum(math.comb(n, k) for k in range(pos, n + 1)) / 2 ** n
    return p, pos, neg


def task_means(arm: dict, keys: list) -> dict:
    by = collections.defaultdict(list)
    for k in keys:
        r = arm.get(k)
        by[k[0].split("|")[1]].append(int(bool(r and r["accepted"])))
    return {t: sum(v) / len(v) for t, v in by.items()}


def compare(edit: dict, ref: dict, keys: list) -> dict:
    te, tr = task_means(edit, keys), task_means(ref, keys)
    diffs = [te[t] - tr[t] for t in sorted(te)]
    p, pos, neg = sign_test_one_sided(diffs)
    micro_e = sum(int(bool(edit.get(k) and edit[k]["accepted"])) for k in keys)
    micro_r = sum(int(bool(ref.get(k) and ref[k]["accepted"])) for k in keys)
    return {"task_macro_mean_diff": sum(diffs) / len(diffs), "tasks_up": pos, "tasks_down": neg,
            "tasks_zero": len(diffs) - pos - neg, "p_sign_one_sided": p,
            "pass": (sum(diffs) / len(diffs) > 0) and p < 0.05,
            "call_micro_accepted": [micro_e, micro_r, len(keys)]}


def main() -> None:
    cands = sys.argv[1:]
    base = load("base")
    keys = sorted(base)
    report = {"n_calls_x_draws": len(keys), "n_tasks": len({k[0].split("|")[1] for k in keys}), "candidates": {}}
    for c in cands:
        e = load(c)
        res = {"vs_base": compare(e, base, keys)}
        for rr in ("rand1", "rand2"):
            ctrl = load(f"{c}_{rr}")
            res[f"vs_{rr}"] = compare(e, ctrl, keys)
            res[f"{rr}_vs_base"] = compare(ctrl, base, keys)
        res["p_candidate"] = max(res["vs_base"]["p_sign_one_sided"], res["vs_rand1"]["p_sign_one_sided"],
                                 res["vs_rand2"]["p_sign_one_sided"])
        res["positive_means"] = all(res[x]["task_macro_mean_diff"] > 0 for x in ("vs_base", "vs_rand1", "vs_rand2"))
        # served-ok loss rule, per draw
        loss = {}
        for d in (0, 1):
            lost = [k for k in keys if k[1] == d and base[k]["recorded_kind"] == "ok" and base[k]["accepted"]
                    and not (e.get(k) and e[k]["accepted"])]
            loss[f"draw{d}"] = {"n_lost": len(lost), "tasks": sorted({k[0].split("|")[1] for k in lost})}
        res["served_ok_loss"] = loss
        res["served_ok_rule_pass"] = all(v["n_lost"] <= 2 for v in loss.values())
        # compile guard
        def comp(arm):
            acc = [arm[k] for k in keys if arm.get(k) and arm[k]["accepted"]]
            fail = sum(1 for r in acc if r["compiles"] == 0)
            unknown = sum(1 for r in acc if r["compiles"] is None)
            ok_comp = sum(1 for r in acc if r["compiles"] == 1)
            return {"accepted": len(acc), "compile_fail_rate_among_accepted": fail / len(acc) if acc else None,
                    "accepted_and_compiling_rate": ok_comp / len(keys), "unknown_compile": unknown}
        cb, ce = comp(base), comp(e)
        res["compile"] = {"base": cb, "edit": ce}
        res["compile_guard_pass"] = (ce["compile_fail_rate_among_accepted"] or 0) <= (cb["compile_fail_rate_among_accepted"] or 0) + 0.02
        report["candidates"][c] = res
    # Holm across the family
    order = sorted(cands, key=lambda c: report["candidates"][c]["p_candidate"])
    m = len(order)
    still = True
    for i, c in enumerate(order):
        r = report["candidates"][c]
        thr = 0.05 / (m - i)
        r["holm_threshold"] = thr
        r["holm_pass"] = still and r["p_candidate"] < thr
        still = r["holm_pass"]
        r["gate_pass_excluding_export"] = (r["holm_pass"] and r["positive_means"] and r["served_ok_rule_pass"]
                                           and r["compile_guard_pass"])
    out = G / "gate_report.json"
    out.write_text(json.dumps(report, indent=1))
    for c in cands:
        r = report["candidates"][c]
        print(f"{c}: macro diff vs base {r['vs_base']['task_macro_mean_diff']:+.4f} (up {r['vs_base']['tasks_up']} / down "
              f"{r['vs_base']['tasks_down']}), p_cand {r['p_candidate']:.4f}, holm {r['holm_pass']}, served-ok rule "
              f"{r['served_ok_rule_pass']}, compile guard {r['compile_guard_pass']} -> GATE {r['gate_pass_excluding_export']}")
    print(f"report -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
