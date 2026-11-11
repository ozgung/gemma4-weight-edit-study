"""Edit-vs-training gate analysis (plan docs/plans/edit-vs-train-20261108.md, frozen 17:08 UTC).  Owner: Claude.

    python3 -I scripts/weight_edit/gate_g_analyze.py [ARM ...]      (default: G-yer-39 G-tam-39 G-tam-411)

Inputs: base and fable_r2 from outputs/weight_edit/claude/gate_r2/ (gen_*.jsonl, gen_*.scored.jsonl, fullcall/gen_*),
reused unchanged; each G arm from outputs/weight_edit/claude/edit_vs_train/gate/ (same file names), generated with
--edit outputs/weight_edit/claude/edit_vs_train/arms/<ARM>/edit. Universe: replay_ood/usable_keys.txt x draws {0, 1}.
Completeness checks are gate_r2_analyze.load's (BLOCK on any missing, extra or duplicate record, or a wrong edit dir).

Rule per G arm (round-2 rule as amended 2026-11-05, reference set {base} only, Holm over m = 3; an arm without
generations holds a p = 1 slot): PRIMARY = negative task-macro artefact diff vs base and one-sided exact sign test
p < Holm threshold; GUARDS = acceptance non-inferiority (task-macro diff vs base >= -0.01), served-ok preservation per
draw (lost <= max(2, floor(1 % of base-accepted served-ok calls))), full-call compile guard (fail rate among accepted
<= base + 0.02). Pass = primary and all guards.
Head-to-head vs fable_r2 (no pass/fail): per-task artefact rate and acceptance, two-sided exact sign test; "better" /
"worse" at p < 0.05, else "tie"; plus fable_r2's own guard metrics side by side. Bootstrap CIs: 10,000, seed 20261108.
Writes outputs/weight_edit/claude/edit_vs_train/gate_g_report.json.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("g2", ROOT / "scripts/weight_edit/gate_r2_analyze.py")
g2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g2)
GR2 = ROOT / "outputs/weight_edit/claude/gate_r2"
GE = ROOT / "outputs/weight_edit/claude/edit_vs_train/gate"
ARMS = ["G-yer-39", "G-tam-39", "G-tam-411"]
M = 3


def load_plain(arm: str, keys: list, gdir: Path, edit: bool) -> dict:
    g2.G = gdir
    return g2.load(arm, keys, edit)


def load_full(arm: str, keys: list, gdir: Path, edit_dir: str | None) -> dict:
    p, g = gdir / "fullcall" / f"gen_{arm}.scored.jsonl", gdir / "fullcall" / f"gen_{arm}.jsonl"
    rows = [json.loads(l) for l in p.open()]
    ks = [(r["key"], r["draw"]) for r in rows]
    if len(ks) != len(set(ks)) or set(ks) != set(keys):
        g2.block(f"{arm} full call: {len(ks)} records, {len(set(ks))} unique, universe {len(keys)}")
    edits = {json.loads(l).get("edit") for l in g.open()}
    if edits != {edit_dir}:
        g2.block(f"{arm} full call: edit field {edits}, expected {edit_dir}")
    return {(r["key"], r["draw"]): r for r in rows}


def comp_rate(arm: dict, keys: list) -> float:
    acc = [arm[k] for k in keys if arm[k]["accepted"]]
    return (sum(1 for r in acc if r["compiles"] == 0) / len(acc)) if acc else 0.0


def served_ok(e: dict, base: dict, keys: list) -> dict:
    out = {}
    for d in range(g2.DRAWS):
        okb = [k for k in keys if k[1] == d and base[k]["recorded_kind"] == "ok" and base[k]["accepted"]]
        lost = sum(1 for k in okb if not e[k]["accepted"])
        out[f"draw{d}"] = {"n_lost": lost, "limit": max(2, math.floor(0.01 * len(okb)))}
    return out


def two_sided(e: dict, r: dict, keys: list, metric: str, lower_better: bool) -> dict:
    c = g2.compare(e, r, keys, metric, lower_better)
    b, w = c["tasks_better"], c["tasks_worse"]
    n = b + w
    tail = lambda x: sum(math.comb(n, i) for i in range(x, n + 1)) / 2 ** n if n else 1.0  # noqa: E731
    p = min(1.0, 2 * min(tail(b), tail(w))) if n else 1.0
    verdict = "better" if p < 0.05 and b > w else "worse" if p < 0.05 and w > b else "tie"
    return {"task_macro_mean_diff": c["task_macro_mean_diff"], "tasks_better": b, "tasks_worse": w,
            "p_sign_two_sided": p, "verdict": verdict, "call_micro": c["call_micro"]}


def main() -> None:
    arms = sys.argv[1:] or ARMS
    want = [l.strip() for l in (g2.R / "usable_keys.txt").read_text().splitlines() if l.strip()]
    keys = sorted((k, d) for k in want for d in range(g2.DRAWS))
    g2.CALLS.update({r["key"]: r for r in map(json.loads, (g2.R / "calls.jsonl").open())})
    repo_of = {g2.task((k, 0)): g2.CALLS[k]["repo"] for k in want}
    base = load_plain("base", keys, GR2, False)
    fab = load_plain("fable_r2", keys, GR2, True)
    base_f = load_full("base", keys, GR2, None)
    fab_f = load_full("fable_r2", keys, GR2, "outputs/weight_edit/claude/gate_r2/fable_r2/edit")
    rep = {"plan": "docs/plans/edit-vs-train-20261108.md", "holm_m": M, "n_calls_x_draws": len(keys),
           "base": {"artefacts": sum(base[k]["escape_artefact"] for k in keys), "fullcall_compile_fail": comp_rate(base_f, keys)},
           "fable_r2": {"artefacts": sum(fab[k]["escape_artefact"] for k in keys),
                        "vs_base_artefact": g2.compare(fab, base, keys, "escape_artefact", True),
                        "vs_base_accepted": g2.compare(fab, base, keys, "accepted"),
                        "served_ok": served_ok(fab, base, keys), "fullcall_compile_fail": comp_rate(fab_f, keys)},
           "arms": {}}
    present = [a for a in arms if (GE / f"gen_{a}.scored.jsonl").exists()]
    for a in present:
        ed = f"outputs/weight_edit/claude/edit_vs_train/arms/{a}/edit"
        e = load_plain(a, keys, GE, True)
        ef = load_full(a, keys, GE, ed)
        pri = g2.compare(e, base, keys, "escape_artefact", True)
        acc = g2.compare(e, base, keys, "accepted")
        so = served_ok(e, base, keys)
        cf, cb = comp_rate(ef, keys), comp_rate(base_f, keys)
        rep["arms"][a] = {
            "artefacts": sum(e[k]["escape_artefact"] for k in keys),
            "primary_vs_base": pri, "acceptance_vs_base": acc,
            "acceptance_noninferior": acc["task_macro_mean_diff"] >= -0.01,
            "served_ok": so, "served_ok_rule_pass": all(v["n_lost"] <= v["limit"] for v in so.values()),
            "fullcall_compile_fail": cf, "fullcall_compile_guard_pass": cf <= cb + 0.02,
            "head_to_head_vs_fable": {"escape_artefact": two_sided(e, fab, keys, "escape_artefact", True),
                                      "accepted": two_sided(e, fab, keys, "accepted", False)},
            "bootstrap": {"artefact_vs_base": g2.boots(e, base, keys, repo_of, "escape_artefact", seed=20261108),
                          "artefact_vs_fable": g2.boots(e, fab, keys, repo_of, "escape_artefact", seed=20261108),
                          "accepted_vs_base": g2.boots(e, base, keys, repo_of, "accepted", seed=20261108)}}
    ps = {a: (rep["arms"][a]["primary_vs_base"]["p_sign_one_sided"] if a in rep["arms"] else 1.0) for a in arms}
    still = True
    for i, a in enumerate(sorted(arms, key=lambda x: ps[x])):
        thr = 0.05 / (M - i)
        ok = still and ps[a] < thr
        still = ok
        if a in rep["arms"]:
            r = rep["arms"][a]
            r["holm_threshold"], r["primary_holm_pass"] = thr, ok
            r["pass"] = (ok and r["primary_vs_base"]["task_macro_mean_diff"] < 0 and r["acceptance_noninferior"]
                         and r["served_ok_rule_pass"] and r["fullcall_compile_guard_pass"])
    out = GE.parent / "gate_g_report.json"
    out.write_text(json.dumps(rep, indent=1))
    print(f"base artefacts {rep['base']['artefacts']}; fable_r2 {rep['fable_r2']['artefacts']}")
    for a, r in rep["arms"].items():
        h = r["head_to_head_vs_fable"]
        print(f"{a}: artefacts {r['artefacts']} (vs base {r['primary_vs_base']['task_macro_mean_diff']:+.4f}, "
              f"p {r['primary_vs_base']['p_sign_one_sided']:.2g}); guards noninf {r['acceptance_noninferior']} "
              f"served-ok {r['served_ok_rule_pass']} compile {r['fullcall_compile_guard_pass']} -> PASS {r['pass']}; "
              f"vs fable: artefacts {h['escape_artefact']['verdict']} (p {h['escape_artefact']['p_sign_two_sided']:.2g}), "
              f"acceptance {h['accepted']['verdict']}")
    print(f"report -> {out}")


if __name__ == "__main__":
    main()
