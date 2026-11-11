"""Key lists for the retry go/no-go (rule frozen in the weight-edit log, go/no-go entry of 2026-11-07).  Owner: Claude.

    python3 -I scripts/weight_edit/retry_gonogo_keys.py

Retry call: a usable recorded edit_file call in a task run with an earlier recorded not_found.
Failed set F(c): the old_strings of those earlier not_found calls.
Writes keys_{heldout,train}_{retry,first}.txt and failed_sets.json to outputs/weight_edit/claude/retry_gonogo/.
"""
from __future__ import annotations

import collections
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "outputs/weight_edit/claude"
OUT = R / "retry_gonogo"
SEED = 20261107


def scan(path: Path):
    rows = [json.loads(l) for l in path.open()]
    by = collections.defaultdict(list)
    for c in rows:
        by[(c["run"], c["instance_id"])].append(c)
    retry, first, failed = [], [], {}
    for cs in by.values():
        cs.sort(key=lambda c: (c["item"], c["tc"]))
        f: list[str] = []
        for c in cs:
            ok = c["verified"] and not c["state_uncertain"] and not c["file_missing"] and c.get("old_string")
            if ok:
                (retry if f else first).append(c)
                if f:
                    failed[c["key"]] = list(f)
            if c["error_kind"] == "not_found":
                f.append(c["old_string"])
    return retry, first, failed


def sample(rows: list[dict], n: int, per_task: int) -> list[str]:
    rng = random.Random(SEED)
    pool = sorted(rows, key=lambda r: r["key"])
    rng.shuffle(pool)
    seen = collections.Counter()
    out = []
    for r in pool:
        if len(out) >= n:
            break
        if seen[r["instance_id"]] >= per_task:
            continue
        seen[r["instance_id"]] += 1
        out.append(r["key"])
    return out


def main() -> None:
    failed_all = {}
    for name, p in (("heldout", R / "replay_ood/calls.jsonl"), ("train", R / "replay/calls.jsonl")):
        retry, first, failed = scan(p)
        failed_all.update(failed)
        if name == "heldout":
            keys_r = sorted(r["key"] for r in retry)
            keys_f = sample(first, len(keys_r), 2)
        else:
            keys_r = sample(retry, 120, 3)
            keys_f = sample(first, 120, 3)
        (OUT / f"keys_{name}_retry.txt").write_text("\n".join(keys_r) + "\n")
        (OUT / f"keys_{name}_first.txt").write_text("\n".join(keys_f) + "\n")
        print(name, "retry", len(keys_r), "first", len(keys_f))
    (OUT / "failed_sets.json").write_text(json.dumps(failed_all))


if __name__ == "__main__":
    main()
