"""Build a held-out replay set for the round-3 gate: replay_set_ood.py with its paths as arguments.  Owner: Claude.

    .venv/bin/python -I scripts/weight_edit/replay_set_r3.py --run LABEL=RUN_DIR:SUITE [--run ...] --tasks SUITE=TASKS.jsonl [--tasks ...] \\
        --files FILES_DIR --out REPLAY_DIR [--limit-traces N]

RUN_DIR holds traces/trace_<iid>.json of an E4B e001nt run. FILES_DIR holds <suite>/<iid>/<path> base files of the snapshots. replay_set_ood.py is imported
unchanged, only its RUNS, TASK_FILES, FILES and OUT are replaced, so the method is the one the round-2 gate set was built with. Output in REPLAY_DIR:
calls.jsonl, files/ and, added here, usable_keys.txt and summary.json. The result is what run_gate_r3.sh takes.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/weight_edit"))
import replay_set_ood as M  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, metavar="LABEL=RUN_DIR:SUITE")
    ap.add_argument("--tasks", action="append", required=True, metavar="SUITE=TASKS.jsonl")
    ap.add_argument("--files", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit-traces", type=int, default=0)
    a = ap.parse_args()
    runs = {}
    for s in a.run:
        label, rest = s.split("=", 1)
        rdir, suite = rest.rsplit(":", 1)
        runs[label] = (rdir, suite)
    M.RUNS = runs
    M.TASK_FILES = {s.split("=", 1)[0]: Path(s.split("=", 1)[1]).resolve() for s in a.tasks}
    M.FILES = a.files.resolve()
    M.OUT = a.out.resolve()
    M.OUT.mkdir(parents=True, exist_ok=True)
    sys.argv = ["replay_set_ood", "--runs", ",".join(runs)] + (["--limit-traces", str(a.limit_traces)] if a.limit_traces else [])
    M.main()
    rows = [json.loads(l) for l in (M.OUT / "calls.jsonl").open()]
    use = [r for r in rows if r["verified"] and not r["state_uncertain"] and not r["file_missing"] and r.get("old_string")]
    (M.OUT / "usable_keys.txt").write_text("\n".join(r["key"] for r in use) + "\n")
    summary = {"calls": len(rows), "verified": sum(r["verified"] for r in rows), "usable": len(use),
               "usable_kinds": dict(collections.Counter(r["error_kind"] for r in use)),
               "usable_tasks": len({r["instance_id"] for r in use}), "usable_repos": len({r["repo"] for r in use}),
               "usable_by_run": dict(collections.Counter(r["run"] for r in use))}
    (M.OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
