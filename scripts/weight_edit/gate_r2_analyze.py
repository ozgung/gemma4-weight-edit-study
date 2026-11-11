"""Round-2 gate analysis on the fresh held-out-repository set (pre-registered, doc Thread 2026-11-04).  Owner: Claude.

    python3 -I scripts/weight_edit/gate_r2_analyze.py [--m 2] [--report gate_r2_report.json] CAND [CAND...]
    (the Astra arm, added 2026-11-04, is its own family: --m 1 --report gate_r2_astra_report.json astra_r2)
    Holm family size m = max(--m, number of CANDs); default m 2 = {opus_r2, fable_r2}. A run whose final edit/ is empty
    is not a candidate and is not generated; it holds the lowest-priority Holm slot (p = 1).

Inputs in outputs/weight_edit/claude/gate_r2/: gen_<arm>.jsonl and gen_<arm>.scored.jsonl for base, each candidate
<c> and its controls <c>_rand1, <c>_rand2. Universe: replay_ood/usable_keys.txt (320 calls, 101 tasks) x draws
{0 greedy, 1 sampled T 0.6}. Written before any round-2 proposer pass.

Completeness (BLOCK = exit without a report): duplicate records; any arm missing or extra (key, draw) records in
generation or scoring; an edit arm whose records do not name its edit dir; a base record with an edit.

Escape artefact (per generation): the generated old_string contains a literal backslash-n or backslash-quote that
the true file state does not contain (the WP12 definition; dossier rev 3).

Rule (revised 2026-11-04 before any round-2 pass: escape artefacts primary, acceptance key secondary):
1. PRIMARY. Per task (instance id; the four OOD-43 runs of a task are pooled), mean artefact rate over its calls x
   draws; paired per-task differences edit - ref. Against base and each control: negative task-macro mean AND
   one-sided exact sign test (tasks lower vs higher) p < 0.05. p_primary = max of the three; Holm across the family.
2. GUARDS (all required with 1): acceptance non-inferiority, task-macro acceptance diff vs base >= -0.01;
   served-ok preservation per draw (calls recorded served-ok that base accepts and the edit rejects
   <= max(2, floor(1 % of base-accepted served-ok calls in that draw))); compile-fail rate among accepted edits
   <= base + 2 points.  Primary pass = 1 and 2.
3. KEY SECONDARY, fixed sequence: tested only for a candidate whose primary passes (otherwise p = 1). Task-macro
   acceptance against base and each control: positive mean AND one-sided sign test p < 0.05; p = max of the
   three; Holm across the same family size.
Reported, not part of the rule: task and repository-clustered bootstrap CIs (10,000 resamples, seed 20261104) of
both task-macro diffs vs base, OOD-43 / OOD-B separately, by recorded kind, the outcome under the edit of base
artefact generations, accepted-and-compiling counts.
"""

from __future__ import annotations

import collections
import json
import math
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
G = Path(os.environ.get("WSP_GATE_DIR", ROOT / "outputs/weight_edit/claude/gate_r2"))  # override for tests only
R = ROOT / "outputs/weight_edit/claude/replay_ood"
DRAWS = 2


def block(msg: str) -> None:
    raise SystemExit(f"BLOCKED: {msg}")


def _read(p: Path) -> dict:
    out = {}
    for r in map(json.loads, p.open()):
        k = (r["key"], r["draw"])
        if k in out:
            block(f"duplicate {k} in {p.name}")
        out[k] = r
    return out


CALLS = {}
FILES = {}


def artefact(key: str, gen: str) -> int:
    sha = CALLS[key]["file_sha"]
    if sha not in FILES:
        FILES[sha] = (R / "files" / f"{sha}.txt").read_text()
    return int(any(e in gen and e not in FILES[sha] for e in ("\\n", '\\"')))


def load(arm: str, keys: list, edit_expected: bool) -> dict:
    g, s = _read(G / f"gen_{arm}.jsonl"), _read(G / f"gen_{arm}.scored.jsonl")
    for name, d in (("gen", g), ("scored", s)):
        miss, extra = set(keys) - set(d), set(d) - set(keys)
        if miss or extra:
            block(f"{arm} {name}: missing {len(miss)} / extra {len(extra)}")
    for k in keys:
        e = g[k].get("edit")
        if edit_expected and not (e and Path(e).parent.name == arm):
            block(f"{arm}: record {k} not generated with {arm}/edit (edit={e})")
        if not edit_expected and e:
            block(f"base record {k} has an edit")
        s[k]["escape_artefact"] = artefact(k[0], g[k]["gen"])
    return s


def task(k) -> str:
    return k[0].split("|")[1]


def task_means(arm: dict, keys: list, metric: str = "accepted") -> dict:
    by = collections.defaultdict(list)
    for k in keys:
        by[task(k)].append(int(bool(arm[k][metric])))
    return {t: sum(v) / len(v) for t, v in by.items()}


def sign_p(diffs) -> tuple[float, int, int]:
    pos, neg = sum(d > 0 for d in diffs), sum(d < 0 for d in diffs)
    n = pos + neg
    return (sum(math.comb(n, i) for i in range(pos, n + 1)) / 2 ** n if n else 1.0), pos, neg


def compare(e: dict, r: dict, keys: list, metric: str = "accepted", lower_better: bool = False) -> dict:
    te, tr = task_means(e, keys, metric), task_means(r, keys, metric)
    diffs = [te[t] - tr[t] for t in sorted(te)]
    p, up, down = sign_p([-d for d in diffs] if lower_better else diffs)
    mean = sum(diffs) / len(diffs)
    better, worse = (up, down)
    return {"metric": metric, "task_macro_mean_diff": mean, "tasks_better": better, "tasks_worse": worse,
            "tasks_zero": len(diffs) - better - worse, "p_sign_one_sided": p,
            "pass": ((mean < 0) if lower_better else (mean > 0)) and p < 0.05,
            "call_micro": [sum(int(bool(e[k][metric])) for k in keys), sum(int(bool(r[k][metric])) for k in keys), len(keys)]}


def boots(e: dict, r: dict, keys: list, repo_of: dict, metric: str = "accepted", n=10000, seed=20261104) -> dict:
    te, tr = task_means(e, keys, metric), task_means(r, keys, metric)
    d = {t: te[t] - tr[t] for t in te}
    ts = sorted(d)
    rng = random.Random(seed)
    m = sorted(sum(d[rng.choice(ts)] for _ in ts) / len(ts) for _ in range(n))
    by_repo = collections.defaultdict(list)
    for t in ts:
        by_repo[repo_of[t]].append(d[t])
    repos = sorted(by_repo)
    rng = random.Random(seed + 1)
    mc = []
    for _ in range(n):
        vals = [x for _ in repos for x in by_repo[rng.choice(repos)]]
        mc.append(sum(vals) / len(vals))
    mc.sort()
    ci = lambda v: [v[int(0.025 * n)], v[int(0.975 * n) - 1]]  # noqa: E731
    return {"task_ci95": ci(m), "repo_clustered_ci95": ci(mc), "n_tasks": len(ts), "n_repos": len(repos)}


def main() -> None:
    args = sys.argv[1:]
    m_family = 2  # pre-registered Holm family {opus_r2, fable_r2}; a run with an empty edit is a non-candidate (p = 1)
    report_name = "gate_r2_report.json"
    while args[:1] in (["--m"], ["--report"]):
        if args[0] == "--m":
            m_family = int(args[1])
        else:
            report_name = args[1]
        args = args[2:]
    cands = args
    want = [l.strip() for l in (R / "usable_keys.txt").read_text().splitlines() if l.strip()]
    if len(set(want)) != len(want):
        block("duplicate keys in usable_keys.txt")
    keys = sorted((k, d) for k in want for d in range(DRAWS))
    CALLS.update({r["key"]: r for r in map(json.loads, (R / "calls.jsonl").open())})
    repo_of = {task((k, 0)): CALLS[k]["repo"] for k in want}
    base = load("base", keys, False)
    arms = {c: load(c, keys, True) for c in cands}
    ctrls = {f"{c}_{x}": load(f"{c}_{x}", keys, True) for c in cands for x in ("rand1", "rand2")}
    srcs = {"OOD-43": [k for k in keys if k[0].startswith("o43")], "OOD-B": [k for k in keys if k[0].startswith("oodb")]}
    refs = ("base", "rand1", "rand2")
    rep = {"n_calls_x_draws": len(keys), "n_tasks": len({task(k) for k in keys}), "holm_m": max(m_family, len(cands)),
           "base_escape_artefacts": sum(base[k]["escape_artefact"] for k in keys), "candidates": {}}
    for c in cands:
        e = arms[c]
        ref = {"base": base, "rand1": ctrls[f"{c}_rand1"], "rand2": ctrls[f"{c}_rand2"]}
        pri = {f"vs_{x}": compare(e, ref[x], keys, "escape_artefact", True) for x in refs}
        pri.update({f"{x}_vs_base": compare(ref[x], base, keys, "escape_artefact", True) for x in ("rand1", "rand2")})
        pri["p"] = max(pri[f"vs_{x}"]["p_sign_one_sided"] for x in refs)
        pri["negative_means"] = all(pri[f"vs_{x}"]["task_macro_mean_diff"] < 0 for x in refs)
        sec = {f"vs_{x}": compare(e, ref[x], keys, "accepted") for x in refs}
        sec.update({f"{x}_vs_base": compare(ref[x], base, keys, "accepted") for x in ("rand1", "rand2")})
        sec["p_raw"] = max(sec[f"vs_{x}"]["p_sign_one_sided"] for x in refs)
        sec["positive_means"] = all(sec[f"vs_{x}"]["task_macro_mean_diff"] > 0 for x in refs)
        res = {"primary_escape_artefact": pri, "secondary_acceptance": sec}
        res["acceptance_noninferior"] = sec["vs_base"]["task_macro_mean_diff"] >= -0.01
        loss = {}
        for d in range(DRAWS):
            okb = [k for k in keys if k[1] == d and base[k]["recorded_kind"] == "ok" and base[k]["accepted"]]
            lost = [k for k in okb if not e[k]["accepted"]]
            lim = max(2, math.floor(0.01 * len(okb)))
            loss[f"draw{d}"] = {"n_lost": len(lost), "limit": lim, "base_accepted_served_ok": len(okb),
                                "tasks": sorted({task(k) for k in lost})}
        res["served_ok_loss"] = loss
        res["served_ok_rule_pass"] = all(v["n_lost"] <= v["limit"] for v in loss.values())

        def comp(arm):
            acc = [arm[k] for k in keys if arm[k]["accepted"]]
            fail = sum(1 for r in acc if r["compiles"] == 0)
            return {"accepted": len(acc), "compile_fail_rate_among_accepted": fail / len(acc) if acc else None,
                    "accepted_and_compiling": sum(1 for r in acc if r["compiles"] == 1),
                    "unknown_compile": sum(1 for r in acc if r["compiles"] is None)}
        cb, ce = comp(base), comp(e)
        res["compile"] = {"base": cb, "edit": ce}
        res["compile_guard_pass"] = (ce["compile_fail_rate_among_accepted"] or 0) <= (cb["compile_fail_rate_among_accepted"] or 0) + 0.02
        base_art = [k for k in keys if base[k]["escape_artefact"]]
        res["reported"] = {
            "bootstrap_vs_base": {"escape_artefact": boots(e, base, keys, repo_of, "escape_artefact"),
                                  "accepted": boots(e, base, keys, repo_of, "accepted")},
            "by_source": {s_: {"escape_artefact": compare(e, base, ks, "escape_artefact", True),
                               "accepted": compare(e, base, ks, "accepted")} for s_, ks in srcs.items()},
            "by_recorded_kind": {kind: {"accepted": [sum(e[k]["accepted"] for k in ks), sum(base[k]["accepted"] for k in ks)],
                                        "escape_artefact": [sum(e[k]["escape_artefact"] for k in ks), sum(base[k]["escape_artefact"] for k in ks)],
                                        "n": len(ks)}
                                 for kind in sorted({base[k]["recorded_kind"] for k in keys})
                                 for ks in [[k for k in keys if base[k]["recorded_kind"] == kind]]},
            "base_artefact_generations_under_edit": {
                "n": len(base_art), "still_artefact": sum(e[k]["escape_artefact"] for k in base_art),
                "base_accepted": sum(base[k]["accepted"] for k in base_art), "edit_accepted": sum(e[k]["accepted"] for k in base_art)},
        }
        rep["candidates"][c] = res
    m = rep["holm_m"]

    def holm(pkey):
        order = sorted(cands, key=pkey)
        still, out = True, {}
        for i, c in enumerate(order):
            thr = 0.05 / (m - i)
            ok = still and pkey(c) < thr
            out[c] = (thr, ok)
            still = ok
        return out

    hp = holm(lambda c: rep["candidates"][c]["primary_escape_artefact"]["p"])
    for c in cands:
        r = rep["candidates"][c]
        r["primary_holm_threshold"], r["primary_holm_pass"] = hp[c]
        r["primary_pass"] = (r["primary_holm_pass"] and r["primary_escape_artefact"]["negative_means"]
                             and r["acceptance_noninferior"] and r["served_ok_rule_pass"] and r["compile_guard_pass"])
        r["secondary_acceptance"]["p"] = r["secondary_acceptance"]["p_raw"] if r["primary_pass"] else 1.0
    hs = holm(lambda c: rep["candidates"][c]["secondary_acceptance"]["p"])
    for c in cands:
        r = rep["candidates"][c]
        r["secondary_holm_threshold"], r["secondary_holm_pass"] = hs[c]
        r["secondary_pass"] = r["primary_pass"] and r["secondary_holm_pass"] and r["secondary_acceptance"]["positive_means"]
    out = G / report_name
    out.write_text(json.dumps(rep, indent=1))
    for c in cands:
        r = rep["candidates"][c]
        pa, sa = r["primary_escape_artefact"], r["secondary_acceptance"]
        print(f"{c}: artefacts vs base {pa['vs_base']['task_macro_mean_diff']:+.4f} (better {pa['vs_base']['tasks_better']} / "
              f"worse {pa['vs_base']['tasks_worse']}), p {pa['p']:.4f}; acceptance vs base {sa['vs_base']['task_macro_mean_diff']:+.4f}; "
              f"guards: noninf {r['acceptance_noninferior']} served-ok {r['served_ok_rule_pass']} compile {r['compile_guard_pass']} "
              f"-> PRIMARY {r['primary_pass']}, SECONDARY {r['secondary_pass']}")
    print(f"report -> {out}")


if __name__ == "__main__":
    main()
