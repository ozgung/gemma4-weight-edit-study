"""Build search set v3 for round 3 (retry target).  Owner: Claude.

    python3 -I scripts/weight_edit/build_search_r3.py

Union of shared/search_r2 (730 calls, training repositories, set "train") and the 320 usable calls of the used round-2
gate set (claude/replay_ood, keys in usable_keys.txt, set "dev"). Added per call:
    set         "train" | "dev"
    failed_set  old_strings of earlier recorded not_found calls in the same run and task, in call order, repeats kept.
                Computed from ALL recorded calls of the task, usable or not (claude/replay/calls.jsonl and
                claude/replay_ood/calls.jsonl), ordered by (item, tc), as retry_gonogo_keys.py does.
    is_retry    failed_set is not empty
Output: outputs/weight_edit/shared/search_r3/ with calls.jsonl, files/<sha>.txt (one directory, deduplicated by content
hash) and MANIFEST.txt. Checks: 730 + 320 calls, no key overlap, 71 dev retry calls as in the census, and agreement with
retry_gonogo/failed_sets.json on every call.
"""
from __future__ import annotations

import collections
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SHARED = ROOT / "outputs/weight_edit/shared"
C = ROOT / "outputs/weight_edit/claude"
OUT = SHARED / "search_r3"


def failed_sets(all_calls: Path) -> dict[str, list[str]]:
    by = collections.defaultdict(list)
    for r in map(json.loads, all_calls.open()):
        by[(r["run"], r["instance_id"])].append({k: r[k] for k in ("key", "item", "tc", "error_kind", "old_string")})
    out = {}
    for cs in by.values():
        cs.sort(key=lambda c: (c["item"], c["tc"]))
        f: list[str] = []
        for c in cs:
            out[c["key"]] = list(f)
            if c["error_kind"] == "not_found":
                assert c["old_string"], c["key"]  # none in either set; a recorded not_found always carries a string
                f.append(c["old_string"])
    return out


def main() -> None:
    (OUT / "files").mkdir(parents=True, exist_ok=True)
    train = [json.loads(l) for l in (SHARED / "search_r2/calls.jsonl").open()]
    want = [l.strip() for l in (C / "replay_ood/usable_keys.txt").read_text().splitlines() if l.strip()]
    ood = {r["key"]: r for r in map(json.loads, (C / "replay_ood/calls.jsonl").open())}
    dev = [ood[k] for k in want]
    assert len(train) == 730 and len(dev) == 320, (len(train), len(dev))
    assert not {r["key"] for r in train} & {r["key"] for r in dev}
    assert not {r["repo"] for r in train} & {r["repo"] for r in dev}
    fs = failed_sets(C / "replay/calls.jsonl")
    fs.update(failed_sets(C / "replay_ood/calls.jsonl"))
    shas = set()
    with (OUT / "calls.jsonl").open("w") as f:
        for name, rows in (("train", train), ("dev", dev)):
            for r in rows:
                assert r["verified"] and not r["state_uncertain"] and not r["file_missing"] and r.get("old_string"), r["key"]
                failed = fs[r["key"]]
                f.write(json.dumps({**r, "set": name, "is_retry": bool(failed), "failed_set": failed}) + "\n")
                shas.add((name, r["file_sha"]))
    src = {"train": SHARED / "search_r2/files", "dev": C / "replay_ood/files"}
    for name, sha in sorted(shas):
        s, d = src[name] / f"{sha}.txt", OUT / "files" / f"{sha}.txt"
        if d.exists():
            assert d.read_bytes() == s.read_bytes(), sha
        else:
            shutil.copyfile(s, d)
    rows = [json.loads(l) for l in (OUT / "calls.jsonl").open()]
    gonogo = json.load((C / "retry_gonogo/failed_sets.json").open())
    for r in rows:
        assert gonogo.get(r["key"], []) == r["failed_set"], r["key"]
    stats = {}
    for name in ("train", "dev"):
        rs = [r for r in rows if r["set"] == name]
        ret = [r for r in rs if r["is_retry"]]
        stats[name] = {
            "calls": len(rs), "tasks": len({r["instance_id"] for r in rs}), "repos": len({r["repo"] for r in rs}),
            "retry_calls": len(ret), "tasks_with_retry": len({r["instance_id"] for r in ret}),
            "first_calls": len(rs) - len(ret),
            "outcomes": dict(collections.Counter(r["error_kind"] for r in rs)),
            "retry_outcomes": dict(collections.Counter(r["error_kind"] for r in ret)),
            "retry_recorded_old_string_in_failed_set": sum(r["old_string"] in r["failed_set"] for r in ret),
        }
    assert stats["dev"]["retry_calls"] == 71 and stats["dev"]["first_calls"] == 249, stats["dev"]
    files = sorted((OUT / "files").glob("*.txt"))
    sha_calls = hashlib.sha256((OUT / "calls.jsonl").read_bytes()).hexdigest()
    lines = [
        f"Round-3 search set manifest (author Claude, 2026-11-07 UTC). {len(rows)} calls, {len(files)} file states.",
        "set train: the 730 calls of the training repositories (fastapi, rich, requests, httpx). set dev: 320 calls from other "
        "repositories, from the round-2 judging set, which is used up and is open to you now.",
        "Fields added to each call: set, is_retry, failed_set.",
        json.dumps(stats),
        f"{sha_calls}  calls.jsonl",
    ]
    lines += [f"{hashlib.sha256(p.read_bytes()).hexdigest()}  files/{p.name}" for p in files]
    (OUT / "MANIFEST.txt").write_text("\n".join(lines) + "\n")
    print(json.dumps(stats, indent=1))
    print("calls.jsonl sha256", sha_calls, "files", len(files))


if __name__ == "__main__":
    main()
