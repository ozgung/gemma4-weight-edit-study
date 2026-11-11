"""WP7 stage 2: live escape artefacts in agent runs, edit vs base and vs random (exploratory).  Owner: Claude.

    .venv/bin/python -I scripts/weight_edit/wp7_stage2_analyze.py [--set ood --set oodb] [--dir SET=PREFIX ...]
        [--out outputs/weight_edit/claude/wp7/stage2_report.json]

Plan: docs/plans/wp7-agent-test-20261107.md (stage 2). Written and committed before any stage-2 trace was read.
Runs: results/<session>/results/wp7_<set>_<arm>_cap12/ with session wp7a for ood (OOD-43, suite ood45) and wp7b for
oodb (OOD-B-71); WP7R adds session wp7r for oodc (OOD-C, suite oodc), run with --set oodc; --dir SET=PREFIX replaces results/<session>/results/wp7_<set>_ (tests only).
WP7T adds session wp7t, set oodct (OOD-C, suite oodc, thinking on: arms e001tt, e001tte, e001ttx), run alone with --set oodct.
Arms: base = e001nt, edit = e001nte (fable_r2 as LoRA), rand = e001ntx (fable_r2_rand1 as LoRA).

Per edit_file call (traces/trace_<iid>.json, agent steps, as edit_calls.py reads them): the true file state is the
snapshot file (data/ext/ood_files/wp7/<suite>/<iid>/<path> from wp7_extract_files.py, else the earlier
data/ext/ood_files/oodfiles/) with every earlier successful edit_file re-applied by the harness's apply_replacement and
write_file content set; a file named by a modifying shell command (replay_set_ood.py's MODIFY_RE) is uncertain from
then on, and every file is if the command names none. Escape artefact (the gate's definition, applied to both
strings as in the full-call replay): old_string or new_string contains a literal backslash-n or backslash-quote that
the file state does not contain. Calls with an unknown or uncertain state are not counted (reported).

PRIMARY (exploratory): per task, artefact rate over its counted calls; tasks paired across two arms when both have at
least one counted call; edit - ref per task; task-macro mean and one-sided exact sign test (edit lower), against base
and against random, both sets pooled (no task is in both). Also per set.
DESCRIPTIVE: calls per task, not-found rate, accepted strategies (unescaped among them), resolved, no patch, errors,
paired discordant resolves (from task_results.jsonl as graded on the VM; a Docker regrade may follow).
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FILES = [ROOT / "data/ext/ood_files/wp7", ROOT / "data/ext/ood_files/oodfiles"]
SETS = {"ood": ("wp7a", "ood45"), "oodb": ("wp7b", "oodb"), "oodc": ("wp7r", "oodc"),  # oodc: WP7R (docs/plans/wp7r-oodc-run-20261108.md)
        "oodct": ("wp7t", "oodc")}  # oodct: WP7T, thinking on (docs/plans/wp7t-think-oodc-20261108.md)
ARMS = {"base": "e001nt_baseline_nothink_t06_e4b", "edit": "e001nte_nothink_t06_fableedit_e4b",
        "rand": "e001ntx_nothink_t06_randedit_e4b"}
ARMS_THINK = {"base": "e001tt_think_t06_e4b", "edit": "e001tte_think_t06_fableedit_e4b",
              "rand": "e001ttx_think_t06_randedit_e4b"}  # WP7T arms (set oodct only)
MODIFY_RE = re.compile(r"sed\s+-i|>\s*\S+\.py|\btee\b|git\s+(checkout|apply|restore|stash|reset)|\bpatch\b|\bmv\b|\bcp\b|\brm\b")
LIT = ("\\n", '\\"')


def parse_obs(step: dict) -> dict:
    try:
        o = json.loads((step.get("observation") or {}).get("content") or "{}")
        return o if isinstance(o, dict) else {}
    except ValueError:
        return {}


def kind(o: dict) -> str:
    if o.get("status") == "ok":
        return "ok"
    msg = str(o.get("error_message") or o.get("error") or "")
    return ("not_found" if "not found" in msg else "multiple" if "occurs" in msg and "times" in msg else
            "missing_param" if "mandatory input parameters" in msg else "other")


def snapshot(suite: str, iid: str, rel: str) -> str | None:
    for base in FILES:
        f = base / suite / iid / rel.removeprefix("./")
        if f.is_file():
            return f.read_text(encoding="utf-8", errors="replace")
    return None


def artefact(text: str, state: str) -> int:
    return int(any(e in text and e not in state for e in LIT))


def calls(trace: Path, suite: str, iid: str) -> list[dict]:
    from swegemma.edit import apply_replacement
    files: dict[str, str | None] = {}
    uncertain: set[str] = set()
    all_uncertain = False
    out = []
    for s in json.loads(trace.read_text()).get("steps", []):
        if s.get("source") != "agent":
            continue
        o = parse_obs(s)
        for tc in s.get("tool_calls") or []:
            name, a = tc.get("function_name"), tc.get("arguments") or {}
            fp = str(a.get("filepath") or "").removeprefix("./")
            if name == "edit_file" and fp:
                if fp not in files:
                    files[fp] = snapshot(suite, iid, fp)
                state = files[fp]
                old, new = str(a.get("old_string") or ""), str(a.get("new_string") or "")
                known = state is not None and not all_uncertain and fp not in uncertain
                k = kind(o)
                out.append({"filepath": fp, "kind": k, "strategy": o.get("strategy") if k == "ok" else None,
                            "counted": int(known),
                            "art_old": artefact(old, state) if known else None,
                            "art_new": artefact(new, state) if known else None})
                if known:
                    out[-1]["art"] = int(out[-1]["art_old"] or out[-1]["art_new"])
                if k == "ok" and state is not None and old:
                    try:
                        res = apply_replacement(state, old, new, filepath=fp, allow_multiple=bool(a.get("allow_multiple", False)))
                        files[fp] = res.new_content if hasattr(res, "new_content") else res[0]
                    except Exception:  # noqa: BLE001
                        uncertain.add(fp)
            elif name == "write_file" and fp and o.get("status") == "ok":
                files[fp] = str(a.get("content", ""))
                uncertain.discard(fp)
            elif name == "run_command" and MODIFY_RE.search(str(a.get("command", ""))):
                hit = [f for f in files if f in str(a.get("command", ""))]
                if hit:
                    uncertain.update(hit)
                else:
                    all_uncertain = True
    return out


def sign_p(lower: int, higher: int) -> float:
    n = lower + higher
    return sum(math.comb(n, i) for i in range(lower, n + 1)) / 2 ** n if n else 1.0


def compare(e: dict, r: dict) -> dict:
    ts = sorted(t for t in e if t in r and e[t]["counted"] and r[t]["counted"])
    d = [e[t]["rate"] - r[t]["rate"] for t in ts]
    lo, hi = sum(x < 0 for x in d), sum(x > 0 for x in d)
    return {"tasks": len(ts), "task_macro_mean_diff": sum(d) / len(d) if d else None, "tasks_lower": lo,
            "tasks_higher": hi, "tasks_equal": len(d) - lo - hi, "p_sign_one_sided": sign_p(lo, hi),
            "artefacts_on_paired_tasks": [sum(e[t]["art"] for t in ts), sum(r[t]["art"] for t in ts)],
            "counted_calls_on_paired_tasks": [sum(e[t]["counted"] for t in ts), sum(r[t]["counted"] for t in ts)]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", action="append", choices=list(SETS))
    ap.add_argument("--dir", action="append", default=[], help="SET=PREFIX (tests)")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/weight_edit/claude/wp7/stage2_report.json")
    a = ap.parse_args()
    sets = a.set or list(SETS)
    prefix = {s: ROOT / f"results/{SETS[s][0]}/results/wp7_{s}_" for s in sets}
    prefix.update({k: Path(v) for k, v in (x.split("=", 1) for x in a.dir)})
    task, res, desc = {}, {}, {}
    for s in sets:
        suite = SETS[s][1]
        for arm, sub in (ARMS_THINK if s == "oodct" else ARMS).items():
            d = Path(f"{prefix[s]}{sub}_cap12")
            rows = {r["instance_id"]: r for r in map(json.loads, (d / "task_results.jsonl").read_text().splitlines()) if r}
            st = collections.Counter()
            for iid in rows:
                tr = d / "traces" / f"trace_{iid}.json"
                cs = calls(tr, suite, iid) if tr.exists() else []
                c = [x for x in cs if x["counted"]]
                task.setdefault(arm, {})[iid] = {"set": s, "calls": len(cs), "counted": len(c), "art": sum(x["art"] for x in c),
                                                 "rate": sum(x["art"] for x in c) / len(c) if c else None}
                res.setdefault(arm, {})[iid] = rows[iid]
                st["tasks"] += 1
                st["traces_missing"] += int(not tr.exists())
                st["edit_calls"] += len(cs)
                st["counted_calls"] += len(c)
                st["artefact_calls"] += sum(x["art"] for x in c)
                st["art_old"] += sum(x["art_old"] for x in c)
                st["art_new"] += sum(x["art_new"] for x in c)
                st["not_found"] += sum(x["kind"] == "not_found" for x in cs)
                st["accepted"] += sum(x["kind"] == "ok" for x in cs)
                for x in cs:
                    if x["strategy"]:
                        st[f"strategy_{x['strategy']}"] += 1
                st["resolved"] += int(bool(rows[iid].get("resolved")))
                st["no_patch"] += int(not rows[iid].get("agent_patch_size"))
                st["error"] += int(bool(rows[iid].get("error")))
            desc[f"{s}/{arm}"] = dict(st)
    rep = {"descriptive": desc, "primary": {}, "resolves": {}}
    for scope in ["pooled", *sets]:
        pick = lambda arm: {t: v for t, v in task[arm].items() if scope == "pooled" or v["set"] == scope}  # noqa: E731
        rep["primary"][scope] = {"edit_vs_base": compare(pick("edit"), pick("base")),
                                 "edit_vs_rand": compare(pick("edit"), pick("rand")),
                                 "rand_vs_base": compare(pick("rand"), pick("base"))}
    for ref in ("base", "rand"):
        ts = sorted(set(res["edit"]) & set(res[ref]))
        e, r = ([int(bool(res[x][t].get("resolved"))) for t in ts] for x in ("edit", ref))
        rep["resolves"][f"edit_vs_{ref}"] = {"tasks": len(ts), "resolved": [sum(e), sum(r)],
                                             "edit_only": sum(x > y for x, y in zip(e, r)), "ref_only": sum(x < y for x, y in zip(e, r))}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rep, indent=1))
    for k, v in desc.items():
        print(f"{k:10} tasks {v['tasks']:3}  edit calls {v['edit_calls']:4}  counted {v['counted_calls']:4}  artefact calls "
              f"{v['artefact_calls']:3} (old {v['art_old']}, new {v['art_new']})  not found {v['not_found']:3}  resolved {v['resolved']:3}")
    for scope, comps in rep["primary"].items():
        for name, c in comps.items():
            m = c["task_macro_mean_diff"]
            print(f"{scope:6} {name:12} tasks {c['tasks']:3}  lower/higher/equal {c['tasks_lower']}/{c['tasks_higher']}/{c['tasks_equal']}"
                  f"  mean diff {m if m is None else round(m, 4)}  p {c['p_sign_one_sided']:.3g}  artefacts {c['artefacts_on_paired_tasks']}")
    print("resolves:", rep["resolves"])


if __name__ == "__main__":
    main()
