"""Final step before publishing OOD-C: fold the Python 3.12 check into the layout. Reads JSON only; runs no repository code.

    python3 -I scripts/oodc/finalize_tasks.py --ood <layout dir> --gate312 <gate312/gate.json>

Drops tasks the agent path could not run on Python 3.12 (Colab's interpreter), records final_gold_cp312 per task, and writes
validation/final_gate_cp312.json, validation/gradable_cp313.txt and validation/gradable_cp312.txt. A task counts as gradable on
a host when it was gradable on its shard (gold twice, empty not) and gold resolved again on the merged layout on that host.
"""

from __future__ import annotations

import argparse
import collections
import json
import shutil
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ood", type=Path, required=True)
    ap.add_argument("--gate312", type=Path, required=True)
    a = ap.parse_args()
    rows = [json.loads(l) for l in (a.ood / "tasks.jsonl").read_text().splitlines() if l.strip()]
    g = json.loads(a.gate312.read_text())
    keep, drop = [], []
    for r in rows:
        v = g.get(r["instance_id"], {})
        if v.get("usable"):
            r["final_gold_cp312"] = bool(v.get("gold1"))
            keep.append(r)
        else:
            drop.append((r["instance_id"], str(v.get("agent_error"))[:120]))
            (a.ood / "snapshots" / f"{r['instance_id']}.tgz").unlink(missing_ok=True)
    (a.ood / "tasks.jsonl").write_text("".join(json.dumps(r) + "\n" for r in keep))
    shutil.copy2(a.gate312, a.ood / "validation" / "final_gate_cp312.json")
    g313 = sorted(r["instance_id"] for r in keep if r["gradable"] and r.get("final_gold_cp313"))
    g312 = sorted(r["instance_id"] for r in keep if r["gradable"] and r.get("final_gold_cp312"))
    (a.ood / "validation" / "gradable_cp313.txt").write_text("\n".join(g313) + "\n")
    (a.ood / "validation" / "gradable_cp312.txt").write_text("\n".join(g312) + "\n")
    print(f"kept {len(keep)} tasks in {len({r['repo'] for r in keep})} repos; dropped on cp312: {drop}")
    print(f"gradable: shard {sum(bool(r['gradable']) for r in keep)}, cp313 {len(g313)}, cp312 {len(g312)}")
    print("gold_gate:", collections.Counter(r.get("gold_gate") for r in keep))
    print("licences:", collections.Counter(r.get("license") for r in keep).most_common())
    print("per repo:", collections.Counter(collections.Counter(r["repo"] for r in keep).values()))


if __name__ == "__main__":
    main()
