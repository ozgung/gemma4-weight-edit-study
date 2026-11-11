"""Round-2 gate generations re-scored with the harness's unescape fallback switched off.  Owner: Claude (review session).

    .venv/bin/python -I scripts/weight_edit/gate_r2_nofallback.py [--extra NAME=GEN.jsonl[,FULLCALL_GEN.jsonl] ...]

POST HOC AND DESCRIPTIVE (weight-edit log, 2026-11-11). Written after the registered verdicts and after a scratch
recount of the same question had been read; it changes no registered verdict. Nothing is regenerated: the frozen
generations of the round-2 gate (gate_r2/gen_<arm>.jsonl, edit_vs_train/gate/gen_<arm>.jsonl and their full-call
replays) go through swegemma.edit.apply_replacement again, once as the harness ships it and once with the fourth
matching tier (adk_eval_core.editing.edit._apply_unescaped_match, the JSON-unescape fallback) returning None.
Tiers 1-3 (exact, flexible, regex) are untouched, so the question is: what would the same gate have shown in a
harness that does not repair escaped arguments?

Checks (BLOCK = exit 1, no report):
  - every arm holds exactly usable_keys x {0, 1}, no duplicates;
  - with the fallback on, every recomputed outcome equals the stored .scored.jsonl outcome (old-text and full-call).

Per arm, under each rule ("harness" = as shipped, "no_fallback" = tier 4 off), with the registered definitions of
gate_r2_analyze.py (imported, not copied): task-macro acceptance difference against the base with the one-sided sign
test and bootstrap intervals; against both size-matched random controls where the arm has them; served-ok preservation
per draw (calls recorded served-ok that the base accepts under the rule and the arm rejects, limit max(2, 1 %));
the compile guard of record on the full-call replay (compile-fail rate among accepted <= base + 0.02). The escape-
artefact count does not depend on the rule and is copied from the generations for reference.
Also reported: calls newly accepted per draw; the served-ok count when the recorded call itself must have succeeded
without the fallback (recorded strategy not 'unescaped'); the exact-only variant (tiers 2-4 off) by outcome label.

--extra adds an arm that is not a weight edit (for example the native-rendering arm); it is compared with the base
on the same keys and has no controls. For such an arm the report also holds the escape-artefact comparison per task
(lower is better) and the sign test in the worse direction, since the arm is read descriptively in both directions.
For every arm with a scored full-call replay: escape artefacts in the generated old and new text (stored flags).
Output: gate_r2/nofallback_report.json and one line per arm. With --extra the report goes to
gate_r2/nofallback_report_<names>.json, so the report of record (no extra arm) is not overwritten.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gate_r2_analyze as A  # noqa: E402  registered definitions: task, compare, boots, artefact
from replay_score import compiles, outcome  # noqa: E402

ROOT = HERE.parents[1]
G2 = ROOT / "outputs/weight_edit/claude/gate_r2"
GT = ROOT / "outputs/weight_edit/claude/edit_vs_train/gate"
R = ROOT / "outputs/weight_edit/claude/replay_ood"
DRAWS = 2
EDITS = ("opus_r2", "fable_r2", "opusinf_r2", "fable52inf_v6")  # the last one is the withdrawn, off-record candidate
TRAINED = ("G-yer-39", "G-tam-39", "G-tam-411")


def block(msg: str) -> None:
    raise SystemExit(f"BLOCKED: {msg}")


def read(p: Path, universe: set) -> dict:
    rows = [json.loads(l) for l in p.open()]
    keys = [(r["key"], r["draw"]) for r in rows]
    if len(keys) != len(set(keys)) or set(keys) != universe:
        block(f"{p}: {len(keys)} records, {len(set(keys))} unique, missing {len(universe - set(keys))}, extra {len(set(keys) - universe)}")
    return dict(zip(keys, rows))


def set_fallback(on: bool, _orig=[]) -> None:
    import adk_eval_core.editing.edit as core

    if not _orig:
        _orig.append(core._apply_unescaped_match)
    core._apply_unescaped_match = _orig[0] if on else (lambda *a, **k: None)


def score(gens: dict, calls: dict, files: dict, new_key: str | None) -> dict:
    """Outcome and compile flag per (key, draw) under the current fallback setting.
    new_key None: the recorded new_string (old-text replay); 'gen_new': the generated one (full-call replay)."""
    out = {}
    for k, g in gens.items():
        r = calls[k[0]]
        new = g[new_key] if new_key else (r.get("new_string") or "")
        oc, new_content = outcome(files[r["file_sha"]], g["gen"], new, r["filepath"], r["allow_multiple"])
        out[k] = {"outcome": oc, "accepted": int(oc.startswith("applied")),
                  "compiles": compiles(new_content, r["filepath"]) if new_content is not None else None,
                  "recorded_kind": r["error_kind"], "recorded_strategy": r["strategy"]}
    return out


def served_ok(e: dict, base: dict, keys: list, strict_recorded: bool = False) -> dict:
    loss = {}
    for d in range(DRAWS):
        okb = [k for k in keys if k[1] == d and base[k]["recorded_kind"] == "ok" and base[k]["accepted"]
               and not (strict_recorded and base[k]["recorded_strategy"] == "unescaped")]
        lost = [k for k in okb if not e[k]["accepted"]]
        loss[f"draw{d}"] = {"n_lost": len(lost), "limit": max(2, math.floor(0.01 * len(okb))),
                            "base_accepted_served_ok": len(okb), "tasks": sorted({A.task(k) for k in lost})}
    return loss


def compile_summary(arm: dict, keys: list) -> dict:
    acc = [arm[k] for k in keys if arm[k]["accepted"]]
    return {"accepted": len(acc),
            "compile_fail_rate_among_accepted": (sum(1 for r in acc if r["compiles"] == 0) / len(acc)) if acc else None,
            "accepted_and_compiling": sum(1 for r in acc if r["compiles"] == 1)}


def main() -> None:
    extras = {}
    args = sys.argv[1:]
    while args[:1] == ["--extra"]:
        name, paths = args[1].split("=", 1)
        extras[name] = [Path(p) for p in paths.split(",")]
        args = args[2:]
    if args:
        block(f"unknown arguments {args}")
    want = [l.strip() for l in (R / "usable_keys.txt").read_text().splitlines() if l.strip()]
    universe = {(k, d) for k in want for d in range(DRAWS)}
    keys = sorted(universe)
    calls = {r["key"]: r for r in map(json.loads, (R / "calls.jsonl").open())}
    files = {sha: (R / "files" / f"{sha}.txt").read_text() for sha in {calls[k]["file_sha"] for k in want}}
    A.CALLS.update(calls)
    repo_of = {A.task((k, 0)): calls[k]["repo"] for k in want}

    # arm -> (old-text generations, full-call generations or None, stored scored files or None)
    src = {"base": (G2 / "gen_base.jsonl", G2 / "fullcall/gen_base.jsonl")}
    for c in EDITS:
        src[c] = (G2 / f"gen_{c}.jsonl", G2 / f"fullcall/gen_{c}.jsonl")
        for x in ("rand1", "rand2"):
            src[f"{c}_{x}"] = (G2 / f"gen_{c}_{x}.jsonl", None)
    for c in TRAINED:
        src[c] = (GT / f"gen_{c}.jsonl", GT / f"fullcall/gen_{c}.jsonl")
    for name, ps in extras.items():
        src[name] = (ps[0], ps[1] if len(ps) > 1 else None)

    gens = {a: read(p, universe) for a, (p, _) in src.items()}
    fgens = {a: read(p, universe) for a, (_, p) in src.items() if p is not None and p.exists()}
    art = {a: {k: A.artefact(k[0], g[k]["gen"]) for k in keys} for a, g in gens.items()}

    scored, fscored = {}, {}
    for rule, on in (("harness", True), ("no_fallback", False)):
        set_fallback(on)
        scored[rule] = {a: score(g, calls, files, None) for a, g in gens.items()}
        fscored[rule] = {a: score(g, calls, files, "gen_new") for a, g in fgens.items()}
    set_fallback(True)

    # the recomputation with the fallback on must reproduce the stored scoring
    n_checked = 0
    for a, (p, fp) in src.items():
        for path, mine in ((p, scored["harness"][a]), (fp, fscored["harness"].get(a))):
            if path is None or mine is None:
                continue
            stored_p = path.with_suffix(".scored.jsonl")
            if not stored_p.exists():
                continue
            stored = read(stored_p, universe)
            bad = [k for k in keys if stored[k]["outcome"] != mine[k]["outcome"] or stored[k]["compiles"] != mine[k]["compiles"]]
            if bad:
                block(f"{stored_p}: {len(bad)} recomputed outcomes differ from the stored scoring, first {bad[0]}")
            n_checked += len(keys)

    # exact-only variant: tiers 2-4 off; a tier-1 match is labelled applied_exact by the harness as shipped
    scored["exact_only"] = {a: {k: dict(v, accepted=int(v["outcome"] == "applied_exact")) for k, v in s.items()}
                            for a, s in scored["harness"].items()}
    fscored["exact_only"] = {a: {k: dict(v, accepted=int(v["outcome"] == "applied_exact")) for k, v in s.items()}
                             for a, s in fscored["harness"].items()}

    rep = {"note": "post hoc, descriptive; frozen generations re-scored; see module docstring",
           "n_calls_x_draws": len(keys), "n_tasks": len({A.task(k) for k in keys}),
           "stored_outcomes_reproduced": n_checked, "rules": {}}
    for rule in ("harness", "no_fallback", "exact_only"):
        S, F = scored[rule], fscored[rule]
        base = S["base"]
        out = {"base": {"accepted": sum(base[k]["accepted"] for k in keys), "artefacts": sum(art["base"].values()),
                        "fullcall": compile_summary(F["base"], keys)}, "arms": {}}
        for a in src:
            if a == "base":
                continue
            e = S[a]
            res = {"accepted": sum(e[k]["accepted"] for k in keys), "artefacts": sum(art[a].values()),
                   "acceptance_vs_base": A.compare(e, base, keys, "accepted"),
                   "acceptance_bootstrap_vs_base": A.boots(e, base, keys, repo_of, "accepted"),
                   "served_ok_loss": served_ok(e, base, keys),
                   "served_ok_loss_recorded_without_fallback": served_ok(e, base, keys, strict_recorded=True),
                   "newly_accepted": {f"draw{d}": sum(1 for k in keys if k[1] == d and not base[k]["accepted"] and e[k]["accepted"])
                                      for d in range(DRAWS)}}
            res["acceptance_noninferior"] = res["acceptance_vs_base"]["task_macro_mean_diff"] >= -0.01
            res["served_ok_rule_pass"] = all(v["n_lost"] <= v["limit"] for v in res["served_ok_loss"].values())
            for x in ("rand1", "rand2"):
                if f"{a}_{x}" in S:
                    res[f"acceptance_vs_{x}"] = A.compare(e, S[f"{a}_{x}"], keys, "accepted")
            if a in extras:
                ae, ab = ({k: {"artefact": art[x][k]} for k in keys} for x in (a, "base"))
                res["artefact_vs_base"] = A.compare(ae, ab, keys, "artefact", lower_better=True)
                res["artefact_bootstrap_vs_base"] = A.boots(ae, ab, keys, repo_of, "artefact")
                res["acceptance_worse_side"] = A.compare(base, e, keys, "accepted")  # one-sided p that the arm is worse
            if a in F:
                fb, fe = compile_summary(F["base"], keys), compile_summary(F[a], keys)
                res["fullcall"] = fe
                res["fullcall_compile_guard_pass"] = (fe["compile_fail_rate_among_accepted"] or 0) <= (fb["compile_fail_rate_among_accepted"] or 0) + 0.02
                res["guards_pass"] = res["acceptance_noninferior"] and res["served_ok_rule_pass"] and res["fullcall_compile_guard_pass"]
            out["arms"][a] = res
        rep["rules"][rule] = out

    # escape artefacts in the full-call replay, from the stored scoring (they do not depend on the rule)
    rep["fullcall_artefacts"] = {}
    for a, (_, fp) in src.items():
        if a in fgens and fp.with_suffix(".scored.jsonl").exists():
            st = read(fp.with_suffix(".scored.jsonl"), universe)
            rep["fullcall_artefacts"][a] = {"old": sum(int(bool(st[k]["artefact_old"])) for k in keys),
                                            "new": sum(int(bool(st[k]["artefact_new"])) for k in keys)}

    dst = G2 / ("nofallback_report.json" if not extras else "nofallback_report_" + "_".join(extras) + ".json")
    dst.write_text(json.dumps(rep, indent=1))
    print(f"stored outcomes reproduced with the fallback on: {n_checked} records")
    for rule in ("harness", "no_fallback", "exact_only"):
        o = rep["rules"][rule]
        b = o["base"]
        print(f"\n== {rule}: base accepted {b['accepted']}/{len(keys)}, artefacts {b['artefacts']}, "
              f"full-call compile fail {b['fullcall']['compile_fail_rate_among_accepted']:.3f} of {b['fullcall']['accepted']}")
        for a, r in o["arms"].items():
            c, ci = r["acceptance_vs_base"], r["acceptance_bootstrap_vs_base"]["repo_clustered_ci95"]
            lo = r["served_ok_loss"]
            line = (f"  {a:20s} art {r['artefacts']:3d} | accepted {r['accepted']} ({r['accepted'] - b['accepted']:+d}) | task macro "
                    f"{100 * c['task_macro_mean_diff']:+.1f} pts [{100 * ci[0]:+.1f}, {100 * ci[1]:+.1f}], better/worse "
                    f"{c['tasks_better']}/{c['tasks_worse']}, p {c['p_sign_one_sided']:.2g} | served-ok lost "
                    f"{lo['draw0']['n_lost']},{lo['draw1']['n_lost']} of {lo['draw0']['base_accepted_served_ok']},{lo['draw1']['base_accepted_served_ok']} | "
                    f"new {r['newly_accepted']['draw0']},{r['newly_accepted']['draw1']}")
            if "acceptance_vs_rand1" in r:
                line += f" | vs controls p {max(r['acceptance_vs_rand1']['p_sign_one_sided'], r['acceptance_vs_rand2']['p_sign_one_sided']):.2g}"
            if "fullcall" in r:
                line += (f" | full-call compile fail {r['fullcall']['compile_fail_rate_among_accepted']:.3f} of {r['fullcall']['accepted']} "
                         f"(guard {r['fullcall_compile_guard_pass']}) -> guards {r['guards_pass']}")
            print(line)
            if "artefact_vs_base" in r:
                c, w = r["artefact_vs_base"], r["acceptance_worse_side"]
                print(f"  {'':20s} artefacts vs base: task macro {100 * c['task_macro_mean_diff']:+.1f} pts, fewer/more "
                      f"{c['tasks_better']}/{c['tasks_worse']}, p {c['p_sign_one_sided']:.2g} | acceptance, worse side p {w['p_sign_one_sided']:.2g}")
    print("\nfull-call artefacts old/new: " + ", ".join(f"{a} {v['old']}/{v['new']}" for a, v in rep["fullcall_artefacts"].items()))
    print(f"\nreport -> {dst}")


if __name__ == "__main__":
    main()
