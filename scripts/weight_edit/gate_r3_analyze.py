"""Round-3 gate analysis: the retry target on a held-out replay set.  Owner: Claude.

    python3 -I scripts/weight_edit/gate_r3_analyze.py --replay REPLAY_DIR --gen-dir GEN_DIR --prepare [--editor-calls CALLS.jsonl]
    python3 -I scripts/weight_edit/gate_r3_analyze.py --replay REPLAY_DIR --gen-dir GEN_DIR [--m 2] [--alpha 0.05] \\
        [--out REPORT.json] CAND [CAND...]

REPLAY_DIR: built by replay_set_ood.py: calls.jsonl (every recorded call), files/, optionally usable_keys.txt.
GEN_DIR: gen_<arm>.jsonl and gen_<arm>.scored.jsonl (replay_gen.py, replay_score.py) for base, each candidate c and its
controls c_rand1 and c_rand2; fullcall/gen_<arm>.jsonl and .scored.jsonl (replay_fullcall.py, replay_score_fullcall.py)
for base and each candidate. --prepare writes keys_usable.txt and keys_fullcall.txt to GEN_DIR for the generators, and with
--editor-calls refuses a set that shares a repository with what the editors could read.
Report: --out, default GEN_DIR/gate_r3_report.json. Nothing is written to REPLAY_DIR.
Holm family size m = max(--m, number of CANDs). A run whose final edit is empty is not a candidate and is not generated;
it holds the lowest-priority Holm slot, so pass --m for the registered family.

Terms. Usable call: verified prompt, certain file state, file present, old_string present. failed_set(c): the old_strings
of the earlier recorded not_found calls in the same run and task, from ALL recorded calls of the task, usable or not,
ordered by (item, tc). Retry call: a usable call with a non-empty failed_set. First-attempt call: any other usable call.
Repeat: a generation byte-identical to a string in its call's failed_set. Task: instance_id, pooled over runs. Universe:
usable calls x draws {0 greedy, 1 sampled at T 0.6, top_p 0.95, top_k 64}, the same seeds across arms.

Completeness (BLOCK = exit without a report): a duplicate record; any arm missing or extra (key, draw) records in generation
or scoring; an edit arm whose records do not name its edit dir; a base record with an edit; usable_keys.txt that differs
from the usable rule.

Rule.
1. PRIMARY. Per-task repeat rate on retry calls, among tasks with at least one retry call: mean over the task's retry
   calls x draws. Paired per-task differences candidate - reference, against base, rand1 and rand2: negative task-macro
   mean AND one-sided exact sign test (tasks lower vs higher, ties dropped) below alpha, against all three. p = the largest
   of the three; Holm over the registered candidates.
2. GUARDS, on all usable calls, all required:
   (a) overall task-macro acceptance, candidate - base >= -0.01 (plain replay);
   (b) served-ok calls lost per draw <= floor(2 * n / 320), n = usable calls, in each draw: calls recorded ok that base
       accepts and the candidate does not (plain replay);
   (c) full-call compile failure among accepted <= base + 0.02 (full-call replay; a call that cannot be cut at new_string
       is left out for both arms and counted in the report).
   Pass = 1 and 2.
3. KEY SECONDARY, tested only for a candidate that passes (otherwise p = 1). Per-task acceptance on retry calls against the
   same three references: positive task-macro mean AND one-sided sign test; p = the largest; Holm over the same family.
Reported, not tested: first-attempt acceptance and share byte-identical to base; repository-clustered bootstrap CIs (10,000
resamples, seed 20261107) of the differences against base; per-source and per-run splits; counts per arm, including near
repeats: retry generations that are not repeats but have difflib ratio >= 0.95 against a string in the failed set, since a repeat
changed by a few bytes lowers the primary without being a different copy.
Under 60 tasks with retry calls: the report is still written, marked reportable = false, and flagged on stdout. Such a set
is too small to report a verdict from; the run does not stop.
"""

from __future__ import annotations

import argparse
import collections
import difflib
import json
import math
import random
import sys
from pathlib import Path

DRAWS = 2
MIN_RETRY_TASKS = 60
SEP = '<|"|>,old_string:<|"|>'
MARK = 'new_string:<|"|>'


def block(msg: str) -> None:
    raise SystemExit(f"BLOCKED: {msg}")


def is_usable(r: dict) -> bool:
    return bool(r["verified"] and not r["state_uncertain"] and not r["file_missing"] and r.get("old_string"))


def cuttable(r: dict) -> bool:
    """replay_fullcall.py's split_prefix conditions, which it asserts."""
    i = r["prefix"].rfind(MARK)
    tail = r["prefix"][i + len(MARK):]
    return i >= 0 and tail.startswith(r.get("new_string") or "") and tail.endswith(SEP)


def failed_sets(calls: list[dict]) -> dict[str, list[str]]:
    by = collections.defaultdict(list)
    for c in calls:
        by[(c["run"], c["instance_id"])].append(c)
    out = {}
    for cs in by.values():
        cs.sort(key=lambda c: (c["item"], c["tc"]))
        f: list[str] = []
        for c in cs:
            out[c["key"]] = list(f)
            if c["error_kind"] == "not_found" and c.get("old_string"):
                f.append(c["old_string"])
    return out


def usable_keys(replay: Path, calls: list[dict]) -> list[str]:
    rule = [c["key"] for c in calls if is_usable(c)]
    p = replay / "usable_keys.txt"
    if not p.exists():
        return rule
    want = [l.strip() for l in p.read_text().splitlines() if l.strip()]
    if len(set(want)) != len(want):
        block("duplicate keys in usable_keys.txt")
    if set(want) != set(rule):
        block(f"usable_keys.txt differs from the usable rule: {len(set(want) - set(rule))} extra, {len(set(rule) - set(want))} missing")
    return want


def read_records(p: Path) -> dict:
    out = {}
    for r in map(json.loads, p.open()):
        k = (r["key"], r["draw"])
        if k in out:
            block(f"duplicate {k} in {p.name}")
        out[k] = r
    return out


def load(d: Path, arm: str, keys: list, edit_expected: bool, failed: dict) -> dict:
    g, s = read_records(d / f"gen_{arm}.jsonl"), read_records(d / f"gen_{arm}.scored.jsonl")
    for name, rec in (("gen", g), ("scored", s)):
        miss, extra = set(keys) - set(rec), set(rec) - set(keys)
        if miss or extra:
            block(f"{d.name}/{arm} {name}: missing {len(miss)} / extra {len(extra)}")
    for k in keys:
        e = g[k].get("edit")
        if edit_expected and not (e and Path(e).parent.name == arm):
            block(f"{arm}: record {k} not generated with {arm}/edit (edit={e})")
        if not edit_expected and e:
            block(f"{arm}: base record {k} has an edit")
        s[k]["gen"] = g[k]["gen"]
        s[k]["repeat"] = int(g[k]["gen"] in failed[k[0]])
    return s


class Gate:
    def __init__(self, calls: list[dict], keys: list[str]):
        self.calls = {c["key"]: c for c in calls}
        self.keys = sorted((k, d) for k in keys for d in range(DRAWS))
        self.task = {k: self.calls[k]["instance_id"] for k in keys}
        self.repo_of = {self.calls[k]["instance_id"]: self.calls[k]["repo"] for k in keys}

    def task_means(self, arm: dict, keys: list, metric: str) -> dict:
        by = collections.defaultdict(list)
        for k in keys:
            by[self.task[k[0]]].append(int(bool(arm[k][metric])))
        return {t: sum(v) / len(v) for t, v in by.items()}

    def compare(self, e: dict, r: dict, keys: list, metric: str, lower_better: bool, alpha: float) -> dict:
        te, tr = self.task_means(e, keys, metric), self.task_means(r, keys, metric)
        diffs = [te[t] - tr[t] for t in sorted(te)]
        p, better, worse = sign_p([-d for d in diffs] if lower_better else diffs)
        mean = sum(diffs) / len(diffs) if diffs else 0.0
        return {"metric": metric, "n_tasks": len(diffs), "task_macro_mean_diff": mean, "tasks_better": better, "tasks_worse": worse,
                "tasks_zero": len(diffs) - better - worse, "p_sign_one_sided": p,
                "pass": ((mean < 0) if lower_better else (mean > 0)) and p < alpha,
                "call_micro": [sum(int(bool(e[k][metric])) for k in keys), sum(int(bool(r[k][metric])) for k in keys), len(keys)]}

    def boots(self, e: dict, r: dict, keys: list, metric: str, n: int = 10000, seed: int = 20261107) -> dict:
        te, tr = self.task_means(e, keys, metric), self.task_means(r, keys, metric)
        d = {t: te[t] - tr[t] for t in te}
        by_repo = collections.defaultdict(list)
        for t in sorted(d):
            by_repo[self.repo_of[t]].append(d[t])
        repos = sorted(by_repo)
        rng = random.Random(seed)
        ms = []
        for _ in range(n):
            vals = [x for _ in repos for x in by_repo[rng.choice(repos)]]
            ms.append(sum(vals) / len(vals))
        ms.sort()
        return {"task_macro_mean_diff": sum(d.values()) / len(d) if d else None, "repo_clustered_ci95": [ms[int(0.025 * n)], ms[int(0.975 * n) - 1]],
                "n_tasks": len(d), "n_repos": len(repos)}


def sign_p(diffs) -> tuple[float, int, int]:
    pos, neg = sum(d > 0 for d in diffs), sum(d < 0 for d in diffs)
    n = pos + neg
    return (sum(math.comb(n, i) for i in range(pos, n + 1)) / 2 ** n if n else 1.0), pos, neg


def holm(cands: list[str], p_of, m: int, alpha: float) -> dict:
    order = sorted(cands, key=p_of)
    still, out = True, {}
    for i, c in enumerate(order):
        thr = alpha / (m - i)
        ok = still and p_of(c) < thr
        out[c] = (thr, ok)
        still = ok
    return out


def served_ok_limit(n: int) -> int:
    """floor(2 * n / 320): the same rate as round 2's limit of 2 on 320 calls."""
    return (2 * n) // 320


def compile_stats(arm: dict, keys: list) -> dict:
    acc = [arm[k] for k in keys if arm[k]["accepted"]]
    return {"n": len(keys), "accepted": len(acc), "compile_fail": sum(1 for r in acc if r["compiles"] == 0),
            "compile_fail_rate_among_accepted": sum(1 for r in acc if r["compiles"] == 0) / len(acc) if acc else None}


def near_repeat(gen: str, failed: list[str]) -> bool:
    return any(difflib.SequenceMatcher(None, gen, f, autojunk=False).ratio() >= 0.95 for f in set(failed))


def counts(arm: dict, base: dict, retry: list, first: list, failed: dict) -> dict:
    rep = [k for k in retry if arm[k]["repeat"]]
    near = [k for k in retry if not arm[k]["repeat"] and near_repeat(arm[k]["gen"], failed[k[0]])]
    return {"retry_gens": len(retry), "repeats": len(rep), "retry_accepted": sum(arm[k]["accepted"] for k in retry),
            "repeats_accepted": sum(arm[k]["accepted"] for k in rep),
            "near_repeats": len(near), "near_repeats_accepted": sum(arm[k]["accepted"] for k in near),
            "first_gens": len(first), "first_accepted": sum(arm[k]["accepted"] for k in first),
            "first_identical_to_base": sum(arm[k]["gen"] == base[k]["gen"] for k in first),
            "first_identical_to_base_share": sum(arm[k]["gen"] == base[k]["gen"] for k in first) / len(first) if first else None}


def main() -> None:
    ap = argparse.ArgumentParser(description="Round-3 gate analysis; the rule is in the module docstring.")
    ap.add_argument("--replay", type=Path, required=True)
    ap.add_argument("--gen-dir", type=Path, required=True)
    ap.add_argument("--m", type=int, default=2)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--editor-calls", type=Path, help="with --prepare: the editors' calls.jsonl; BLOCK if a repository overlaps")
    ap.add_argument("cands", nargs="*")
    a = ap.parse_args()
    calls = [json.loads(l) for l in (a.replay / "calls.jsonl").open()]
    idx = {c["key"]: c for c in calls}
    if len(idx) != len(calls):
        block("duplicate keys in calls.jsonl")
    want = usable_keys(a.replay, calls)
    fc_want = [k for k in want if cuttable(idx[k])]
    if a.prepare:
        if a.editor_calls:
            seen = {json.loads(l)["repo"] for l in a.editor_calls.open()}
            hit = sorted(seen & {idx[k]["repo"] for k in want})
            if hit:
                block(f"{len(hit)} repositories of this set were open to the editors: {hit[:5]}")
        a.gen_dir.mkdir(parents=True, exist_ok=True)
        (a.gen_dir / "keys_usable.txt").write_text("\n".join(want) + "\n")
        (a.gen_dir / "keys_fullcall.txt").write_text("\n".join(fc_want) + "\n")
        print(f"usable {len(want)}, full-call {len(fc_want)}")
        return
    if not a.cands:
        block("no candidates")
    failed = failed_sets(calls)
    gate = Gate(calls, want)
    keys = gate.keys
    retry = [k for k in keys if failed[k[0]]]
    first = [k for k in keys if not failed[k[0]]]
    fc_keys = sorted((k, d) for k in fc_want for d in range(DRAWS))
    n = len(want)
    ok_limit = served_ok_limit(n)
    tasks_retry = {gate.task[k[0]] for k in retry}
    m = max(a.m, len(a.cands))
    base = load(a.gen_dir, "base", keys, False, failed)
    fc_base = load(a.gen_dir / "fullcall", "base", fc_keys, False, failed)
    refs = ("base", "rand1", "rand2")
    rep = {"replay": str(a.replay), "gen_dir": str(a.gen_dir), "alpha": a.alpha, "holm_m": m, "n_calls": n, "n_calls_x_draws": len(keys),
           "n_tasks": len({gate.task[k[0]] for k in keys}), "n_repos": len({gate.repo_of[gate.task[k[0]]] for k in keys}),
           "retry_calls": len({k[0] for k in retry}), "first_calls": len({k[0] for k in first}), "tasks_with_retry": len(tasks_retry),
           "min_tasks_with_retry": MIN_RETRY_TASKS, "reportable": len(tasks_retry) >= MIN_RETRY_TASKS, "served_ok_limit_per_draw": ok_limit,
           "fullcall_calls": len(fc_want), "fullcall_excluded": sorted(set(want) - set(fc_want)),
           "base": counts(base, base, retry, first, failed), "base_fullcall": compile_stats(fc_base, fc_keys), "candidates": {}}
    ok_rows = {d: [k for k in keys if k[1] == d and base[k]["recorded_kind"] == "ok" and base[k]["accepted"]] for d in range(DRAWS)}
    for c in a.cands:
        e = load(a.gen_dir, c, keys, True, failed)
        ctrl = {x: load(a.gen_dir, f"{c}_{x}", keys, True, failed) for x in ("rand1", "rand2")}
        fc_e = load(a.gen_dir / "fullcall", c, fc_keys, True, failed)
        ref = {"base": base, **ctrl}
        pri = {f"vs_{x}": gate.compare(e, ref[x], retry, "repeat", True, a.alpha) for x in refs}
        pri.update({f"{x}_vs_base": gate.compare(ctrl[x], base, retry, "repeat", True, a.alpha) for x in ctrl})
        pri["p"] = max(pri[f"vs_{x}"]["p_sign_one_sided"] for x in refs)
        pri["negative_means"] = all(pri[f"vs_{x}"]["task_macro_mean_diff"] < 0 for x in refs)
        sec = {f"vs_{x}": gate.compare(e, ref[x], retry, "accepted", False, a.alpha) for x in refs}
        sec.update({f"{x}_vs_base": gate.compare(ctrl[x], base, retry, "accepted", False, a.alpha) for x in ctrl})
        sec["p_raw"] = max(sec[f"vs_{x}"]["p_sign_one_sided"] for x in refs)
        sec["positive_means"] = all(sec[f"vs_{x}"]["task_macro_mean_diff"] > 0 for x in refs)
        acc_all = gate.compare(e, base, keys, "accepted", False, a.alpha)
        loss = {}
        for d, okb in ok_rows.items():
            lost = [k for k in okb if not e[k]["accepted"]]
            loss[f"draw{d}"] = {"n_lost": len(lost), "limit": ok_limit, "base_accepted_served_ok": len(okb),
                                "tasks": sorted({gate.task[k[0]] for k in lost})}
        cb, ce = rep["base_fullcall"], compile_stats(fc_e, fc_keys)
        g_comp = (ce["compile_fail_rate_among_accepted"] or 0) <= (cb["compile_fail_rate_among_accepted"] or 0) + 0.02
        res = {"primary_repeat_rate": pri, "secondary_retry_acceptance": sec,
               "guard_a_acceptance": {"vs_base": acc_all, "pass": acc_all["task_macro_mean_diff"] >= -0.01},
               "guard_b_served_ok": {"per_draw": loss, "pass": all(v["n_lost"] <= v["limit"] for v in loss.values())},
               "guard_c_fullcall_compile": {"base": cb, "candidate": ce, "pass": g_comp}}
        by_src = collections.defaultdict(list)
        by_run = collections.defaultdict(list)
        for k in keys:
            by_src[idx[k[0]].get("suite") or idx[k[0]]["run"]].append(k)
            by_run[idx[k[0]]["run"]].append(k)

        def split(groups: dict) -> dict:
            out = {}
            for s, ks in sorted(groups.items()):
                rk = [k for k in ks if failed[k[0]]]
                out[s] = {"n_gens": len(ks), "retry_gens": len(rk), "acceptance_all": gate.compare(e, base, ks, "accepted", False, a.alpha)}
                if rk:
                    out[s]["repeat"] = gate.compare(e, base, rk, "repeat", True, a.alpha)
                    out[s]["retry_acceptance"] = gate.compare(e, base, rk, "accepted", False, a.alpha)
            return out

        res["reported"] = {
            "counts": {"candidate": counts(e, base, retry, first, failed), "rand1": counts(ctrl["rand1"], base, retry, first, failed),
                       "rand2": counts(ctrl["rand2"], base, retry, first, failed)},
            "first_attempt_acceptance_vs_base": gate.compare(e, base, first, "accepted", False, a.alpha),
            "bootstrap_vs_base": {"retry_repeat": gate.boots(e, base, retry, "repeat"), "retry_acceptance": gate.boots(e, base, retry, "accepted"),
                                  "overall_acceptance": gate.boots(e, base, keys, "accepted"), "first_acceptance": gate.boots(e, base, first, "accepted")},
            "by_source": split(by_src), "by_run": split(by_run),
        }
        rep["candidates"][c] = res
    cands = list(a.cands)
    hp = holm(cands, lambda c: rep["candidates"][c]["primary_repeat_rate"]["p"], m, a.alpha)
    for c in cands:
        r = rep["candidates"][c]
        r["primary_holm_threshold"], r["primary_holm_pass"] = hp[c]
        r["guards_pass"] = r["guard_a_acceptance"]["pass"] and r["guard_b_served_ok"]["pass"] and r["guard_c_fullcall_compile"]["pass"]
        r["passes"] = bool(r["primary_holm_pass"] and r["primary_repeat_rate"]["negative_means"] and r["guards_pass"])
        r["secondary_retry_acceptance"]["p"] = r["secondary_retry_acceptance"]["p_raw"] if r["passes"] else 1.0
    hs = holm(cands, lambda c: rep["candidates"][c]["secondary_retry_acceptance"]["p"], m, a.alpha)
    for c in cands:
        r = rep["candidates"][c]
        r["secondary_holm_threshold"], r["secondary_holm_pass"] = hs[c]
        r["secondary_passes"] = bool(r["passes"] and r["secondary_holm_pass"] and r["secondary_retry_acceptance"]["positive_means"])
    out = a.out or a.gen_dir / "gate_r3_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=1))
    warn = None
    if not rep["reportable"]:
        warn = (f"FLAG: only {len(tasks_retry)} tasks with retry calls, under {MIN_RETRY_TASKS}: this set is too small to report a verdict from; "
                "the figures below are written and marked reportable = false")
        print(warn, file=sys.stderr)
    b = rep["base"]
    print(f"set: {n} usable calls, {rep['n_tasks']} tasks, {rep['n_repos']} repos; {rep['retry_calls']} retry calls in {len(tasks_retry)} tasks; "
          f"served-ok limit {ok_limit} per draw; Holm m {m}")
    print(f"base: repeats {b['repeats']} of {b['retry_gens']}, near repeats {b['near_repeats']}, retry accepted {b['retry_accepted']} of {b['retry_gens']}, "
          f"first accepted {b['first_accepted']} of {b['first_gens']}, full-call compile fail {rep['base_fullcall']['compile_fail_rate_among_accepted']:.4f}")
    for c in cands:
        r = rep["candidates"][c]
        pr, ct = r["primary_repeat_rate"], r["reported"]["counts"]["candidate"]
        print(f"{c}: repeats {ct['repeats']}, near repeats {ct['near_repeats']}, retry accepted {ct['retry_accepted']}, first accepted {ct['first_accepted']}, "
              f"first identical {ct['first_identical_to_base_share']:.3f}; repeat vs base {pr['vs_base']['task_macro_mean_diff']:+.4f} "
              f"({pr['vs_base']['tasks_better']} better / {pr['vs_base']['tasks_worse']} worse), p {pr['p']:.4g} (threshold {r['primary_holm_threshold']:.4g}); "
              f"guards: acceptance {r['guard_a_acceptance']['vs_base']['task_macro_mean_diff']:+.4f} {r['guard_a_acceptance']['pass']}, "
              f"served-ok {[v['n_lost'] for v in r['guard_b_served_ok']['per_draw'].values()]} {r['guard_b_served_ok']['pass']}, "
              f"compile {r['guard_c_fullcall_compile']['candidate']['compile_fail_rate_among_accepted']:.4f} {r['guard_c_fullcall_compile']['pass']} "
              f"-> PASS {r['passes']}, SECONDARY {r['secondary_passes']}")
    if warn:
        print(warn)
    print(f"report -> {out}")


if __name__ == "__main__":
    main()
