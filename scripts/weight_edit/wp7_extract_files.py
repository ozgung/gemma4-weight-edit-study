"""WP7 stage 2, on the Colab VM: copy out of the task snapshots every file the WP7 traces read, edit or write.  Owner: Claude.

    python3 -I /content/g4/scripts/weight_edit/wp7_extract_files.py --suite ood45 --snapshots /content/ood45/snapshots \
        --out /content/wp7_files /content/results/wp7_ood_*_cap12
    (--suite oodb --snapshots /content/oodb/snapshots for OOD-B), then tar /content/wp7_files and download it to
    data/ext/ood_files/wp7/ (git-ignored data), where wp7_stage2_analyze.py reads base file states.

Stdlib only. Output: <out>/<suite>/<iid>/<path> plus <out>/<suite>/report.json (files found and missing per task).
"""

from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path


def touched(trace: Path) -> set[str]:
    out = set()
    for s in json.loads(trace.read_text()).get("steps", []):
        for tc in s.get("tool_calls") or []:
            fp = (tc.get("arguments") or {}).get("filepath")
            if tc.get("function_name") in ("read_file", "edit_file", "write_file") and fp:
                out.add(str(fp).removeprefix("./"))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True)
    ap.add_argument("--snapshots", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("runs", nargs="+", type=Path)
    a = ap.parse_args()
    want: dict[str, set[str]] = {}
    for run in a.runs:
        for t in (run / "traces").glob("trace_*.json"):
            want.setdefault(t.stem.removeprefix("trace_"), set()).update(touched(t))
    report = {}
    for iid, rels in sorted(want.items()):
        tgz = a.snapshots / f"{iid}.tgz"
        got, missing = 0, []
        if not tgz.exists():
            report[iid] = {"got": 0, "missing": sorted(rels), "no_snapshot": True}
            continue
        with tarfile.open(tgz) as tf:
            members = {m.name.removeprefix("./"): m for m in tf.getmembers() if m.isfile()}
            for rel in sorted(rels):
                cand = [k for k in members if k.endswith("/" + rel)]  # archives with a top-level directory
                m = members.get(rel) or (members[min(cand, key=len)] if cand else None)
                if m is None:
                    missing.append(rel)
                    continue
                dst = a.out / a.suite / iid / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(tf.extractfile(m).read())
                got += 1
        report[iid] = {"got": got, "missing": missing}
    (a.out / a.suite).mkdir(parents=True, exist_ok=True)
    (a.out / a.suite / "report.json").write_text(json.dumps(report, indent=1))
    print(f"{a.suite}: {len(want)} tasks, {sum(r['got'] for r in report.values())} files, "
          f"{sum(len(r['missing']) for r in report.values())} missing")


if __name__ == "__main__":
    main()
