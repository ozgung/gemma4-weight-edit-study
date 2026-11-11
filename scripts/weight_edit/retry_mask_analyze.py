"""Masked-copy check (go/no-go 3b) analysis; rule frozen in the weight-edit log, entry 2026-11-07 23:13 UTC.  Owner: Claude.

    python3 -I scripts/weight_edit/retry_mask_analyze.py

Stuck calls are fixed by the base run already read (gen_heldout_retry: at least 4 of 8 draws byte-identical to a string
in F(c)); 39 calls in 18 tasks. Masked run: mask/gen.jsonl, mask/gen.scored.jsonl (8 draws per call, same seeds).
G1b value = mean over tasks of (mean over the task's stuck calls of acceptance over all 8 masked draws).
Pass: value >= 0.40 over at least 8 tasks. Everything else is descriptive. Writes mask/report.json.
"""

from __future__ import annotations

import collections
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
O = ROOT / "outputs/weight_edit/claude/retry_gonogo"


def load(gen_p: Path, sc_p: Path, failed: dict) -> dict:
    gens = {(g["key"], g["draw"]): g for g in map(json.loads, gen_p.open())}
    by = collections.defaultdict(list)
    for x in map(json.loads, sc_p.open()):
        g = gens[(x["key"], x["draw"])]
        by[x["key"]].append({"rep": g["gen"] in failed[x["key"]], "acc": bool(x["accepted"]), "closed": g["closed"],
                             "comp": x["compiles"], "draw": x["draw"]})
    return by


def macro(by: dict, keys: list[str], f) -> tuple[float | None, int]:
    per = collections.defaultdict(list)
    for k in keys:
        rows = by.get(k, [])
        if rows:
            per[k.split("|")[1]].append(sum(f(r) for r in rows) / len(rows))
    vals = [sum(v) / len(v) for v in per.values()]
    return (sum(vals) / len(vals) if vals else None), len(vals)


def main() -> None:
    failed = json.load(open(O / "failed_sets.json"))
    base = load(O / "gen_heldout_retry.jsonl", O / "gen_heldout_retry.scored.jsonl", failed)
    msk = load(O / "mask/gen.jsonl", O / "mask/gen.scored.jsonl", failed)
    stuck = sorted(k for k, v in base.items() if sum(r["rep"] for r in v) >= 4)
    free = sorted(k for k in base if k not in stuck)
    rep = {}
    for name, ks in (("stuck", stuck), ("not_stuck", free)):
        for arm, by in (("base", base), ("masked", msk)):
            acc, nt = macro(by, ks, lambda r: r["acc"])
            rp, _ = macro(by, ks, lambda r: r["rep"])
            cl, _ = macro(by, ks, lambda r: r["closed"])
            ac, _ = macro(by, ks, lambda r: r["acc"] and r["comp"] != 0)
            rep[f"{name}_{arm}"] = {"calls": len(ks), "calls_with_draws": sum(1 for k in ks if by.get(k)),
                                    "tasks": nt, "acceptance": acc, "accepted_and_compiling": ac,
                                    "repeat_share": rp, "closed_share": cl}
    g1b, nt = rep["stuck_masked"]["acceptance"], rep["stuck_masked"]["tasks"]
    rep["G1b_value"], rep["G1b_tasks"] = g1b, nt
    rep["G1b"] = bool(g1b is not None and g1b >= 0.40 and nt >= 8)
    rep["GO_3b"] = rep["G1b"]  # G2 passed in the first go/no-go (held-out AUC 0.959) and is not re-run
    (O / "mask/report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
