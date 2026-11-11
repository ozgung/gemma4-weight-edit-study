"""WP1 failure census for e001nt (E4B-NoThink-T0.6) on training-repo runs only.  Owner: Claude.

Labels every task-run with a primary outcome and behaviour flags, from the ATIF traces, patches,
result rows and test logs. Discovery data only: OOD traces are deliberately not read (see
docs/experiments/weight-edit-collab-20261102.md, Thread 20:15).

    python3 -I scripts/weight_edit/census.py --out outputs/weight_edit/claude/census.csv

Primary outcome (one per run):
    resolved
    np_budget      no patch, stopped by the tool-call or time cap
    np_edit_failed no patch, edits attempted, none succeeded
    np_reverted    no patch, an edit succeeded but the diff is empty at the end
    np_no_edit     no patch, never attempted an edit (stopped while exploring)
    p_broken       patch, test run errored at collection/import (pytest exit 2, or SyntaxError/ImportError)
    p_wrong        patch, tests ran and failed
    p_other        patch, unresolved for another reason (exit code missing etc.)
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS = {  # run label -> (results dir, optional regrade file with authoritative resolved/exit code)
    "rft1": ("results/e4r/results/rft1_e001nt_baseline_nothink_t06_e4b_cap12", None),
    "rft3": ("results/e4r/results/rft3_e001nt_baseline_nothink_t06_e4b_cap12", None),
    "rft5": ("results/e4r/results/rft5_e001nt_baseline_nothink_t06_e4b_cap12", None),
    "l4": ("outputs/l4run_e001nt/results", "outputs/l4run_e001nt/regraded.jsonl"),
    "w91a": ("results/e4a/results/w91_e001nt_baseline_nothink_t06_e4b_cap12", None),
}
EDIT_TOOLS = {"edit_file", "write_file"}
BREAK_RE = re.compile(r"ERROR collecting|SyntaxError|IndentationError|ImportError|ModuleNotFoundError|NameError: name")


def truthy(v) -> bool:
    return v is True or str(v) == "True"


def obs_status(step: dict) -> str | None:
    content = (step.get("observation") or {}).get("content")
    if not content:
        return None
    try:
        o = json.loads(content)
    except ValueError:
        return None
    if not isinstance(o, dict):
        return None
    # ADK argument-validation failures ("mandatory input parameters are not present") carry only an
    # "error" key and no "status"; count them as errors (fixed 2026-11-02 after Astra's reconciliation).
    return o.get("status") or ("error" if "error" in o else None)


def obs_text(step: dict) -> str:
    return str((step.get("observation") or {}).get("content") or "")


def patch_files(patch: str) -> list[str]:
    return re.findall(r"^\+\+\+ b/(\S+)", patch, flags=re.M)


def analyse(trace_path: Path) -> dict:
    t = json.loads(trace_path.read_text())
    agent = [s for s in t["steps"] if s.get("source") == "agent"]
    calls = []  # (step index, name, args json, status)
    text_only = 0
    comp_tokens = []
    for i, s in enumerate(agent):
        tcs = s.get("tool_calls") or []
        if not tcs and s.get("extra", {}).get("event_type") != "final":
            text_only += 1
        if s.get("metrics"):
            comp_tokens.append(s["metrics"].get("completion_tokens") or 0)
        for tc in tcs:
            calls.append((i, tc.get("function_name"), json.dumps(tc.get("arguments"), sort_keys=True), obs_status(s), s))
    names = [c[1] for c in calls]
    sig = collections.Counter((c[1], c[2]) for c in calls)
    max_repeat = max(sig.values(), default=0)
    edits = [c for c in calls if c[1] in EDIT_TOOLS]
    edit_ok = [c for c in edits if c[3] == "ok"]
    edit_err = [c for c in edits if c[3] not in (None, "ok")]
    reads = {json.loads(c[2]).get("filepath") for c in calls if c[1] == "read_file" and c[2] != "null"}
    edited_paths = {json.loads(c[2]).get("filepath") for c in edits if c[2] != "null"}
    pytest_idx = [c[0] for c in calls if c[1] == "run_command" and "pytest" in c[2]]
    last_edit = max((c[0] for c in edit_ok), default=-1)
    first_edit = min((c[0] for c in edits), default=-1)
    submit_ok = any(c[1] == "submit_patch" and c[3] == "ok" for c in calls)
    budget_warned = any("budget_warning" in obs_text(c[4]) for c in calls)
    final = next((s for s in reversed(agent) if s.get("extra", {}).get("event_type") == "final"), None)
    return {
        "n_agent_steps": len(agent),
        "n_calls": len(calls),
        "n_text_only_turns": text_only,
        "mean_completion_tokens": round(sum(comp_tokens) / len(comp_tokens), 1) if comp_tokens else 0,
        "max_completion_tokens": max(comp_tokens, default=0),
        "n_read": names.count("read_file"),
        "n_cmd": names.count("run_command"),
        "n_edit_attempts": len(edits),
        "n_edit_ok": len(edit_ok),
        "n_edit_err": len(edit_err),
        "n_write_file": names.count("write_file"),
        "n_graph_tools": sum(names.count(n) for n in ("get_code_neighbors", "search_similar_code", "get_code_subgraph")),
        "n_tool_err": sum(1 for c in calls if c[3] not in (None, "ok")),
        "max_identical_call_repeat": max_repeat,
        "first_edit_step": first_edit,
        "edited_unread_file": int(any(p not in reads for p in edited_paths if p)),
        "n_pytest": len(pytest_idx),
        "tested_after_last_edit": int(any(i > last_edit for i in pytest_idx)) if last_edit >= 0 else 0,
        "submitted": int(submit_ok),
        "budget_warned": int(budget_warned),
        "final_text_chars": len(str(final.get("message", ""))) if final else 0,
    }


def primary(row: dict) -> str:
    if row["resolved"]:
        return "resolved"
    if row["patch_size"] == 0:
        if row["budget_stop"]:
            return "np_budget"
        if row["n_edit_attempts"] and not row["n_edit_ok"]:
            return "np_edit_failed"
        if row["n_edit_ok"]:
            return "np_reverted"
        return "np_no_edit"
    if row["broken"]:
        return "p_broken"
    if row["test_exit_code"] == 1:
        return "p_wrong"
    return "p_other"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/weight_edit/claude/census.csv")
    args = ap.parse_args()
    rows = []
    for label, (rdir, regrade) in RUNS.items():
        rdir = ROOT / rdir
        regr = {}
        if regrade:
            for line in (ROOT / regrade).read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    regr[r["iid"]] = r
        for line in (rdir / "task_results.jsonl").read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            iid = r["instance_id"]
            trace = rdir / "traces" / f"trace_{iid}.json"
            if not trace.exists():
                continue
            patch_p = rdir / "patches" / f"{iid}.patch"
            patch = patch_p.read_text() if patch_p.exists() else ""
            g = regr.get(iid, r)
            exit_code = g.get("test_exit_code")
            exit_code = int(exit_code) if str(exit_code) not in ("None", "") else None
            log_p = rdir / "test_outputs" / f"{iid}.log"
            log = log_p.read_text(errors="replace") if log_p.exists() else ""
            files = patch_files(patch)
            err = str(r.get("error"))
            row = {
                "run": label,
                "instance_id": iid,
                "repo": iid.rsplit("_", 1)[0],
                "resolved": int(truthy(g.get("resolved"))),
                "patch_size": int(r.get("agent_patch_size") or 0) if str(r.get("agent_patch_size")) != "None" else 0,
                "patch_files": len(files),
                "patch_touches_tests_only": int(bool(files) and all("test" in f.split("/")[0] or "/tests/" in f or f.startswith("test") for f in files)),
                "test_exit_code": exit_code,
                "broken": int(exit_code == 2 or (exit_code not in (0, None) and bool(BREAK_RE.search(log[:20000])) and not regrade)),
                "budget_stop": int("budget" in err or "timeout" in err),
                "error": "" if err == "None" else err[:60],
                "duration_s": float(r.get("duration_seconds") or 0),
                "tool_calls": int(r.get("tool_calls") or 0),
                "llm_calls": int(r.get("total_llm_calls") or 0),
            }
            row.update(analyse(trace))
            row["primary"] = primary(row)
            rows.append(row)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} task-runs -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
