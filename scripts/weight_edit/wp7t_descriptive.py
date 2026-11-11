"""WP7T descriptive extras, copied from wp7r_descriptive.py with paths and arm table changed (plan docs/plans/wp7t-think-oodc-20261108.md, "Descriptive"). Written after the primary was read;
nothing here changes the primary.  Owner: Claude.

    .venv/bin/python -I scripts/weight_edit/wp7r_descriptive.py GRADABLE_CP312_TXT TASKS_JSONL

Per-task artefact rates come from wp7_stage2_analyze.calls() (same counting). Adds: task-macro mean rate per arm on the
paired tasks and their ratio; a repository-clustered bootstrap (10,000 resamples of repositories, seed 20261108) of the
task-macro difference; resolved counts and discordant pairs on the cp312-gradable subset (as graded on the VM).
Writes outputs/weight_edit/claude/wp7t/descriptive.json.
"""

from __future__ import annotations

import importlib.util
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("s2", ROOT / "scripts/weight_edit/wp7_stage2_analyze.py")
s2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s2)
PFX = ROOT / "results/wp7t/results/wp7_oodct_"


def main() -> None:
    gradable = {l.strip() for l in Path(sys.argv[1]).read_text().splitlines() if l.strip()}
    repo = {r["instance_id"]: r["repo"] for r in map(json.loads, Path(sys.argv[2]).read_text().splitlines()) if r}
    rate, res = {}, {}
    for arm, sub in s2.ARMS_THINK.items():
        d = Path(f"{PFX}{sub}_cap12")
        rows = {r["instance_id"]: r for r in map(json.loads, (d / "task_results.jsonl").read_text().splitlines()) if r}
        for iid in rows:
            tr = d / "traces" / f"trace_{iid}.json"
            c = [x for x in (s2.calls(tr, "oodc", iid) if tr.exists() else []) if x["counted"]]
            rate.setdefault(arm, {})[iid] = sum(x["art"] for x in c) / len(c) if c else None
            res.setdefault(arm, {})[iid] = int(bool(rows[iid].get("resolved")))
    out = {}
    rng = random.Random(20261108)
    for e, r in (("edit", "base"), ("edit", "rand"), ("rand", "base")):
        ts = sorted(t for t in rate[e] if rate[e][t] is not None and rate[r].get(t) is not None)
        me = sum(rate[e][t] for t in ts) / len(ts)
        mr = sum(rate[r][t] for t in ts) / len(ts)
        by = {}
        for t in ts:
            by.setdefault(repo[t], []).append(rate[e][t] - rate[r][t])
        keys = sorted(by)
        boots = []
        for _ in range(10000):
            pick = [by[rng.choice(keys)] for _ in keys]
            flat = [x for g in pick for x in g]
            boots.append(sum(flat) / len(flat))
        boots.sort()
        out[f"{e}_vs_{r}"] = {"paired_tasks": len(ts), "repos": len(keys), "macro_rate": [me, mr],
                              "macro_ratio": me / mr if mr else None, "macro_diff": me - mr,
                              "repo_bootstrap_95": [boots[249], boots[9749]]}
    for e, r in (("edit", "base"), ("edit", "rand")):
        ts = sorted(gradable & set(res[e]) & set(res[r]))
        a, b = [res[e][t] for t in ts], [res[r][t] for t in ts]
        out[f"gradable_resolved_{e}_vs_{r}"] = {"tasks": len(ts), "resolved": [sum(a), sum(b)],
                                                f"{e}_only": sum(x > y for x, y in zip(a, b)),
                                                f"{r}_only": sum(x < y for x, y in zip(a, b))}
    p = ROOT / "outputs/weight_edit/claude/wp7t/descriptive.json"
    p.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
