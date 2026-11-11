"""Repair arm (fable52rep_r2) against its starting adapter G-tam-39: secondary, descriptive.  Owner: Claude.

    python3 -I scripts/weight_edit/rep_vs_start.py [RUN]        (default RUN fable52rep_r2)

Plan: docs/plans/fable52-repair-20261109.md, written and hashed before any pass of the repair arm. The pass/fail
verdict is NOT computed here: it is the round-2 rule from run_gate_r2.sh (m = 1) with the full-call compile guard
of record (gate_r2_fullcall.py, gate_r2_fable52rep_report_amended.json). This script only sets the final edit beside
the adapter it started from, on the same gate set and draws:
  - per-task escape-artefact rate and acceptance, final edit against G-tam-39, two-sided exact sign test
    ("better" / "worse" at p < 0.05, else "tie"), with task- and repository-clustered bootstrap CIs (10,000, seed
    20261108);
  - the three guard metrics of both, side by side: acceptance change against base (task-macro), served-ok calls lost
    per draw (limit max(2, floor(1 %))), full-call compile-fail rate among accepted edits (limit base + 0.02);
  - "repaired" (descriptive label, fixed in the plan): the final edit passes served-ok and the compile guard, which
    G-tam-39 failed, and is not worse than G-tam-39 on artefacts by the two-sided test.
Inputs: base and the final edit from outputs/weight_edit/claude/gate_r2/ (gen_<RUN>*.jsonl, fullcall/gen_<RUN>*),
G-tam-39 from outputs/weight_edit/claude/edit_vs_train/gate/; universe and completeness checks as gate_g_analyze.py.
Writes outputs/weight_edit/claude/gate_r2/rep_vs_start_<RUN>.json.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("gg", ROOT / "scripts/weight_edit/gate_g_analyze.py")
gg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gg)
g2 = gg.g2
START = "G-tam-39"


def guards(e: dict, ef: dict, base: dict, base_f: dict, keys: list) -> dict:
    acc = g2.compare(e, base, keys, "accepted")
    so = gg.served_ok(e, base, keys)
    cf, cb = gg.comp_rate(ef, keys), gg.comp_rate(base_f, keys)
    return {"artefacts": sum(e[k]["escape_artefact"] for k in keys),
            "acceptance_change_vs_base": acc["task_macro_mean_diff"], "acceptance_noninferior": acc["task_macro_mean_diff"] >= -0.01,
            "served_ok": so, "served_ok_rule_pass": all(v["n_lost"] <= v["limit"] for v in so.values()),
            "fullcall_compile_fail": cf, "fullcall_compile_guard_pass": cf <= cb + 0.02}


def main() -> None:
    run = sys.argv[1] if len(sys.argv) > 1 else "fable52rep_r2"
    want = [l.strip() for l in (g2.R / "usable_keys.txt").read_text().splitlines() if l.strip()]
    keys = sorted((k, d) for k in want for d in range(g2.DRAWS))
    g2.CALLS.update({r["key"]: r for r in map(json.loads, (g2.R / "calls.jsonl").open())})
    repo_of = {g2.task((k, 0)): g2.CALLS[k]["repo"] for k in want}
    base = gg.load_plain("base", keys, gg.GR2, False)
    base_f = gg.load_full("base", keys, gg.GR2, None)
    e = gg.load_plain(run, keys, gg.GR2, True)
    ef = gg.load_full(run, keys, gg.GR2, f"outputs/weight_edit/claude/gate_r2/{run}/edit")
    s = gg.load_plain(START, keys, gg.GE, True)
    sf = gg.load_full(START, keys, gg.GE, f"outputs/weight_edit/claude/edit_vs_train/arms/{START}/edit")
    ge, gs = guards(e, ef, base, base_f, keys), guards(s, sf, base, base_f, keys)
    art = gg.two_sided(e, s, keys, "escape_artefact", True)
    rep = {"plan": "docs/plans/fable52-repair-20261109.md", "run": run, "start": START, "n_calls_x_draws": len(keys),
           "base_fullcall_compile_fail": gg.comp_rate(base_f, keys), "final": ge, "start_adapter": gs,
           "head_to_head_vs_start": {"escape_artefact": art, "accepted": gg.two_sided(e, s, keys, "accepted", False)},
           "bootstrap_vs_start": {"escape_artefact": g2.boots(e, s, keys, repo_of, "escape_artefact", seed=20261108),
                                  "accepted": g2.boots(e, s, keys, repo_of, "accepted", seed=20261108)},
           "repaired": bool(ge["served_ok_rule_pass"] and ge["fullcall_compile_guard_pass"] and art["verdict"] != "worse")}
    out = gg.GR2 / f"rep_vs_start_{run}.json"
    out.write_text(json.dumps(rep, indent=1))
    for name, g in (("final", ge), ("start", gs)):
        print(f"{name:5} artefacts {g['artefacts']:3}  acc {g['acceptance_change_vs_base']:+.4f}  served-ok lost "
              f"{[v['n_lost'] for v in g['served_ok'].values()]}  compile {g['fullcall_compile_fail']:.3f}")
    h = rep["head_to_head_vs_start"]
    print(f"vs start: artefacts {h['escape_artefact']['verdict']} (p {h['escape_artefact']['p_sign_two_sided']:.2g}), "
          f"acceptance {h['accepted']['verdict']} (p {h['accepted']['p_sign_two_sided']:.2g}); repaired {rep['repaired']}")
    print(f"report -> {out}")


if __name__ == "__main__":
    main()
