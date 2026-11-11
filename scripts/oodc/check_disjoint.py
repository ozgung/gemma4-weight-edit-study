"""Check that a task file's repositories are disjoint from everything an agent of ours has seen. Reads only small JSON files.

    python3 -I scripts/oodc/check_disjoint.py data/ext/oodc/tasks.jsonl [--repo-meta data/ext/oodc/repo_meta.jsonl]

Compares on the name as written, the GitHub canonical name, the fork parent and source, and the bare repository name (all
lower case), against: the training repos (data/raw/tasks.jsonl), OOD-45 (scripts/oodc/select_candidates.py and
data/ext/ood45/tasks.jsonl), OOD-B's kept tasks (data/ext/oodb/tasks.jsonl, research/raw4/oodb/tasks.jsonl), the replay set
(outputs/weight_edit/claude/replay_ood/calls.jsonl) and every local agent run (rows of results/**/task_results.jsonl and
outputs/**/task_results.jsonl with tool or model calls; gold and empty-patch gates have none). Exit status 1 on any overlap.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRAIN_NAMES = {"fastapi", "rich", "requests", "httpx"}
OOD45 = {
    "apache/iceberg-python", "caronc/apprise", "feast-dev/feast", "fsspec/filesystem_spec", "has2k1/plotnine",
    "hgrecco/pint", "kedro-org/kedro", "kedro-org/kedro-viz", "pdm-project/pdm", "pola-rs/polars",
    "posit-dev/great-tables", "posit-dev/py-shiny", "pyjanitor-devs/pyjanitor", "tkem/cachetools", "tox-dev/tox",
    "urllib3/urllib3", "wntrblm/nox",
}


def repos_in(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {json.loads(l)["repo"] for l in path.read_text().splitlines() if l.strip()}


def agent_run_repos() -> tuple[set[str], int, int]:
    repos, files, rows = set(), 0, 0
    for top in ("results", "outputs"):
        p = subprocess.run(["find", str(ROOT / top), "-name", "task_results.jsonl", "-size", "+0"], capture_output=True,
                           text=True)
        for f in p.stdout.split():
            files += 1
            for line in Path(f).read_text(errors="replace").splitlines():
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if r.get("repo") and ((r.get("tool_calls") or 0) > 0 or (r.get("total_llm_calls") or 0) > 0):
                    repos.add(r["repo"])
                    rows += 1
    return repos, files, rows


def forms(repo: str, meta: dict) -> set[str]:
    m = meta.get(repo) or {}
    full = {repo.lower()} | {str(m[k]).lower() for k in ("full_name", "parent", "source") if m.get(k)}
    return full | {x.split("/", 1)[1] for x in full}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tasks", type=Path)
    ap.add_argument("--repo-meta", type=Path, default=ROOT / "data/ext/oodc/repo_meta.jsonl")
    a = ap.parse_args()
    meta = {}
    for line in a.repo_meta.read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            meta[d["asked"]] = d
    mine = repos_in(a.tasks)
    runs, nfiles, nrows = agent_run_repos()
    sets = {
        "training (data/raw/tasks.jsonl)": repos_in(ROOT / "data/raw/tasks.jsonl"),
        "OOD-45": OOD45 | repos_in(ROOT / "data/ext/ood45/tasks.jsonl"),
        "OOD-B kept (data/ext/oodb)": repos_in(ROOT / "data/ext/oodb/tasks.jsonl"),
        "OOD-B kept (research/raw4)": repos_in(ROOT / "research/raw4/oodb/tasks.jsonl"),
        "replay_ood calls.jsonl": repos_in(ROOT / "outputs/weight_edit/claude/replay_ood/calls.jsonl"),
        f"local agent runs ({nfiles} result files, {nrows} rows)": runs,
    }
    print(f"{a.tasks}: {sum(1 for l in a.tasks.read_text().splitlines() if l.strip())} tasks in {len(mine)} repos")
    missing = sorted(r for r in mine if r not in meta)
    print(f"repos without GitHub metadata (matched on the written name only): {len(missing)} {missing[:10]}")
    bad = 0
    mine_forms = {r: forms(r, meta) for r in mine}
    for name, other in sets.items():
        keys = set(TRAIN_NAMES) if name.startswith("training") else set()
        for r in other:
            keys |= forms(r, meta)
        hit = sorted(r for r, f in mine_forms.items() if f & keys)
        print(f"  {name}: {len(other)} repos, overlap {len(hit)} {hit[:10]}")
        bad += len(hit)
    names = {}
    for r in mine:
        names.setdefault(r.split("/", 1)[1].lower(), []).append(r)
    dup = {k: v for k, v in names.items() if len(v) > 1}
    print(f"  same bare name, two owners inside the set: {len(dup)} {dup}")
    print("DISJOINT: OK" if not bad and not dup else f"DISJOINT: FAILED ({bad} overlaps)")
    sys.exit(1 if bad or dup else 0)


if __name__ == "__main__":
    main()
