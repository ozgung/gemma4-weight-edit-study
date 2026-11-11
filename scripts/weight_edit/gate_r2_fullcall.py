"""Round-2 gate, amended compile guard (full-call replay).  Owner: Claude.

    python3 -I scripts/weight_edit/gate_r2_fullcall.py --report gate_r2_report.json [--out NAME] CAND [CAND...]

Amendment (2026-11-05 ~03:10 UTC, before any round-2 candidate edit was generated on the gate set; doc Thread
2026-11-05). The registered compile guard (gate_r2_analyze.py: compile-fail rate among accepted edits <= base + 2
points) is computed by replay_score.py with the served model's recorded new_string, which sits in the frozen prefix
because the model writes new_string before old_string. An edit that turns an escaped old_string into an exact copy
moves the harness from its unescape fallback (which also unescapes new_string) to an exact match that inserts the
frozen, escaped new_string verbatim, so the file stops compiling whatever the edited model would have written for
new_string. On the search set every artefact-reducing pass of all three proposers raised that rate (Astra pass 4:
37 of 67 compile failures are exactly this case, base 2 of 31).

Guard of record from now on: the same rule (edit compile-fail rate among accepted edits <= base + 0.02) computed on the
full-call replay (replay_fullcall.py: the arm generates new_string, then old_string; the pair goes through
apply_replacement; compile()). Everything else in the registered rule is unchanged: primary metric, sign tests, Holm,
acceptance non-inferiority, served-ok preservation, fixed-sequence secondary. Amended primary pass = registered primary
pass with compile_guard_pass replaced by fullcall_compile_guard_pass; the secondary is then tested (fixed sequence) for
amended-primary passes and Holm-adjusted over the same family size. The registered verdicts stay in the registered
report and are printed next to the amended ones.

Inputs: gate_r2/<report> (from gate_r2_analyze.py) and gate_r2/fullcall/gen_{base,CAND}.scored.jsonl
(replay_score_fullcall.py). BLOCK (exit 1, no output) if any arm's (key, draw) set differs from usable_keys x {0, 1}
or has duplicates, or an edit arm's generations do not name its frozen edit dir.
Output: gate_r2/<out> (default: the report name with _amended), one line per candidate.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
G = Path(os.environ.get("WSP_GATE_DIR", ROOT / "outputs/weight_edit/claude/gate_r2"))
R = ROOT / "outputs/weight_edit/claude/replay_ood"
DRAWS = 2


def load(arm: str, universe: set) -> dict:
    p = G / "fullcall" / f"gen_{arm}.scored.jsonl"
    g = G / "fullcall" / f"gen_{arm}.jsonl"
    if not p.exists() or not g.exists():
        sys.exit(f"BLOCK: {p} or {g} missing")
    rows = [json.loads(l) for l in p.open()]
    keys = [(r["key"], r["draw"]) for r in rows]
    if len(keys) != len(set(keys)) or set(keys) != universe:
        sys.exit(f"BLOCK: {arm}: {len(keys)} records, {len(set(keys))} unique, universe {len(universe)}, "
                 f"missing {len(universe - set(keys))}, extra {len(set(keys) - universe)}")
    edits = {json.loads(l).get("edit") for l in g.open()}
    if arm == "base" and edits != {None}:
        sys.exit("BLOCK: base full-call records carry an edit")
    if arm != "base" and edits != {str(Path("outputs/weight_edit/claude/gate_r2") / arm / "edit")}:
        sys.exit(f"BLOCK: {arm}: edit field {edits}")
    return {(r["key"], r["draw"]): r for r in rows}


def summary(arm: dict, keys: list) -> dict:
    acc = [arm[k] for k in keys if arm[k]["accepted"]]
    n = len(keys)
    return {"n": n, "accepted": len(acc) / n,
            "compile_fail_rate_among_accepted": (sum(1 for r in acc if r["compiles"] == 0) / len(acc)) if acc else None,
            "accepted_and_compiling": sum(1 for r in acc if r["compiles"] == 1),
            "compile_fail_rate_among_accepted_frozen_new": (sum(1 for r in acc if r["compiles_frozen_new"] == 0) / len(acc)) if acc else None,
            "artefact_old": sum(arm[k]["artefact_old"] for k in keys) / n,
            "artefact_new": sum(arm[k]["artefact_new"] for k in keys) / n,
            "exact_copy": sum(arm[k]["exact_copy"] for k in keys) / n,
            "new_unclosed": sum(1 - arm[k]["closed_new"] for k in keys) / n}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True)
    ap.add_argument("--out")
    ap.add_argument("cands", nargs="+")
    a = ap.parse_args()
    rep = json.loads((G / a.report).read_text())
    universe = {(k.strip(), d) for k in (R / "usable_keys.txt").read_text().splitlines() if k.strip() for d in range(DRAWS)}
    keys = sorted(universe)
    base = load("base", universe)
    m = rep["holm_m"]
    out = {"amendment": "full-call compile guard, 2026-11-05 (see module docstring)", "registered_report": a.report,
           "holm_m": m, "base": summary(base, keys), "candidates": {}}
    cands = list(a.cands)
    if set(cands) != set(rep["candidates"]):
        sys.exit(f"BLOCK: candidates {sorted(cands)} differ from the registered report's {sorted(rep['candidates'])}")
    for c in cands:
        r = rep["candidates"][c]
        e = load(c, universe)
        se = summary(e, keys)
        cb, ce = out["base"]["compile_fail_rate_among_accepted"] or 0, se["compile_fail_rate_among_accepted"] or 0
        both = [k for k in keys if base[k]["accepted"] and e[k]["accepted"]]
        res = {"fullcall": se, "fullcall_compile_guard_pass": ce <= cb + 0.02,
               "paired_accepted_and_compiling": {
                   "both": sum(1 for k in both if base[k]["compiles"] == 1 and e[k]["compiles"] == 1),
                   "only_base": sum(1 for k in keys if base[k]["accepted"] and base[k]["compiles"] == 1 and not (e[k]["accepted"] and e[k]["compiles"] == 1)),
                   "only_edit": sum(1 for k in keys if e[k]["accepted"] and e[k]["compiles"] == 1 and not (base[k]["accepted"] and base[k]["compiles"] == 1))},
               "registered": {x: r[x] for x in ("primary_pass", "secondary_pass", "compile_guard_pass", "acceptance_noninferior",
                                                "served_ok_rule_pass", "primary_holm_pass")}}
        res["amended_primary_pass"] = (r["primary_holm_pass"] and r["primary_escape_artefact"]["negative_means"]
                                       and r["acceptance_noninferior"] and r["served_ok_rule_pass"]
                                       and res["fullcall_compile_guard_pass"])
        res["secondary_p"] = r["secondary_acceptance"]["p_raw"] if res["amended_primary_pass"] else 1.0
        out["candidates"][c] = res
    # Holm over the same family size as the registered report (empty runs hold p = 1 slots)
    order = sorted(cands, key=lambda c: out["candidates"][c]["secondary_p"])
    still = True
    for i, c in enumerate(order):
        x = out["candidates"][c]
        thr = 0.05 / (m - i)
        ok = still and x["secondary_p"] < thr
        x["secondary_holm_threshold"], x["secondary_holm_pass"] = thr, ok
        still = ok
        x["amended_secondary_pass"] = (x["amended_primary_pass"] and ok
                                       and rep["candidates"][c]["secondary_acceptance"]["positive_means"])
    name = a.out or a.report.replace(".json", "_amended.json")
    (G / name).write_text(json.dumps(out, indent=1))
    b = out["base"]
    print(f"base full-call: accepted {b['accepted']:.3f}, compile fail {b['compile_fail_rate_among_accepted']:.3f}, "
          f"artefacts old {b['artefact_old']:.3f} new {b['artefact_new']:.3f}")
    for c in cands:
        x = out["candidates"][c]
        f = x["fullcall"]
        print(f"{c}: full-call accepted {f['accepted']:.3f}, compile fail {f['compile_fail_rate_among_accepted']:.3f} "
              f"(guard {x['fullcall_compile_guard_pass']}; registered guard {x['registered']['compile_guard_pass']}), "
              f"artefacts old {f['artefact_old']:.3f} new {f['artefact_new']:.3f} -> AMENDED PRIMARY {x['amended_primary_pass']}, "
              f"SECONDARY {x['amended_secondary_pass']} (registered: {x['registered']['primary_pass']}, {x['registered']['secondary_pass']})")
    print(f"report -> {G / name}")


if __name__ == "__main__":
    main()
