"""Summarise OOD-C build and gate results from the small files of the kernel outputs (no repo code runs).

    kaggle kernels output ozgung/g4-oodc-build-s0 -p outputs/oodc/build-s0 --page-size 200 \
        --file-pattern '(build_report\\.jsonl|gate/gate\\.json|oodc/tasks\\.jsonl|kernel\\.log|gate/.*\\.log)$'
    python3 -I scripts/oodc/report.py --candidates data/ext/oodc/candidates.jsonl outputs/oodc/build-s*

Prints the per-task table (build stage, times, sizes, gold/empty), the failure buckets and totals as Markdown.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path


def gold_cause(log: str, secs) -> str:
    """Coarse cause of a gold-patch failure from the pytest output of the first gold run."""
    if secs and secs >= 295:
        return "timeout (300 s)"
    if "_pytest.subtests" in log or "requires pytest" in log.lower():
        return "needs a newer pytest than the host's 8.4.2"
    if "async def functions are not natively supported" in log:
        return "async tests without a loaded plugin (anyio is disabled by the harness)"
    if re.search(r"ModuleNotFoundError: No module named", log):
        return "missing module in the wheel set"
    if re.search(r"cannot import name|unexpected keyword argument|has no attribute|no validator found", log):
        return "dependency version mismatch"
    if "ImportError while loading conftest" in log or "error during collection" in log:
        return "collection/conftest error"
    if re.search(r"\d+ failed|\d+ error", log):
        return "other tests in the files fail"
    return "other"


def bucket(rec: dict, gate: dict | None, log: str = "") -> str:
    if rec.get("status") != "ok":
        e = rec.get("error") or ""
        st = rec.get("stage")
        if "heavy dependencies" in e:
            return "build: heavy deps (torch/tf/jax)"
        if "no Python project at the repository root" in e:
            return "build: project not at repo root"
        if "wheel set" in e and "MB >" in e:
            return "build: wheel set too large"
        if "no wheel for" in e:
            return "build: no wheel for a pin"
        if ("import" in e and "fails in the build venv" in e) or "no importable package" in e \
                or "no import package found" in e:
            return "build: package not importable"
        if "snapshot tree differs" in e or "file changed as we read it" in e:
            return "build: snapshot (tree mismatch / tar race)"
        if st == "env":
            return "build: install failed"
        return f"build: {st}"
    if gate is None:
        return "gate: not run"
    if not gate.get("usable"):
        return "gate: agent path failed"
    if gate["gradable"]:
        return "kept, gradable"
    if not gate["gold1"] or not gate["gold2"]:
        err = gate.get("gold_error") or ""
        if gate["gold1"] and not gate["gold2"]:
            return "kept, not gradable: gold flaky (1 of 2)"
        if "Required test node did not pass" in err:
            return "kept, not gradable: gold fails F2P node"
        if "zero or no passing tests" in err:
            return "kept, not gradable: harness sees no passing tests"
        return "kept, not gradable: gold fails: " + gold_cause(log, gate.get("gold1_s"))
    if gate["empty"]:
        return "kept, not gradable: empty patch passes"
    return "kept, not gradable: other"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", type=Path, nargs="+")
    ap.add_argument("--candidates", type=Path)
    a = ap.parse_args()
    cand = {}
    if a.candidates:
        cand = {json.loads(l)["instance_id"]: json.loads(l) for l in a.candidates.read_text().splitlines() if l}
    recs, gates, logs = {}, {}, {}
    for d in a.dirs:
        for f in d.rglob("gold1/test_outputs/*.log"):
            logs[f.stem] = f.read_text(errors="replace")
        for f in d.rglob("build_report.jsonl"):
            for line in f.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    recs[r["instance_id"]] = r
        for f in d.rglob("gate.json"):
            gates.update(json.loads(f.read_text()))
    rows = []
    for iid, r in sorted(recs.items()):
        g = gates.get(iid)
        c = cand.get(iid, {})
        b_ = bucket(r, g, logs.get(iid, ""))
        t = r.get("t", {})
        rows.append({
            "id": iid, "repo": r["repo"], "created": (c.get("created_at") or "")[:10], "bucket": b_,
            "build_s": t.get("total_s"), "snap_mb": round((r.get("snapshot_bytes") or 0) / 2**20, 1),
            "wheels_mb": round((r.get("wheels_bytes") or 0) / 2**20, 1), "wheels_n": r.get("wheels_n"),
            "history": (r.get("history") or "").split(";")[0][:20], "gold": "" if g is None else
            f"{int(bool(g['gold1']))}{int(bool(g['gold2']))}", "empty": "" if g is None else int(bool(g["empty"])),
            "gold_s": (g or {}).get("gold1_s"), "env_mode": r.get("env_mode", ""),
            "agent_s": (g or {}).get("agent_s"), "calls": (g or {}).get("agent_tool_calls"),
            "err": re.sub(r"\s+", " ", (r.get("error") or (g or {}).get("agent_error")
                                         or (g or {}).get("gold_error") or ""))[:90],
        })
    print("| task | repo | created | result | build s | snapshot MB | wheels MB (n) | history | gold | empty | "
          "gold s | agent s | tool calls | note |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for x in rows:
        print(f"| {x['id']} | {x['repo']} | {x['created']} | {x['bucket']} | {x['build_s']} | {x['snap_mb']} | "
              f"{x['wheels_mb']} ({x['wheels_n']}) | {x['history']} | {x['gold']} | {x['empty']} | {x['gold_s']} | "
              f"{x['agent_s']} | {x['calls']} | {x['err'].replace('|', '/')} |")
    cnt = collections.Counter(x["bucket"] for x in rows)
    print("\n| bucket | tasks |\n|---|---|")
    for k, v in cnt.most_common():
        print(f"| {k} | {v} |")
    kept = [x for x in rows if x["bucket"].startswith("kept")]
    built = [x for x in rows if not x["bucket"].startswith("build")]
    gradable = sum(x["bucket"] == "kept, gradable" for x in rows)
    print(f"\ncandidates {len(rows)}, built {len(built)}, gated {sum(1 for x in rows if x['gold'])}, kept {len(kept)} "
          f"in {len({x['repo'] for x in kept})} repos, gradable {gradable}")
    if built:
        bs = sorted(x["build_s"] or 0 for x in rows)
        print(f"build time per candidate: median {bs[len(bs) // 2]} s, sum {sum(bs) / 3600:.1f} h; "
              f"kept snapshot MB sum {sum(x['snap_mb'] for x in kept):.0f}, "
              f"kept wheels MB sum (per task, before dedup) {sum(x['wheels_mb'] for x in kept):.0f}")


if __name__ == "__main__":
    main()
