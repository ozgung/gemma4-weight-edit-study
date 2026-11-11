"""Recount for Appendix A (read-only): edit_file outcomes and matching strategies in the training runs. Run: python3 -I scripts/paper/appendix_checks/appA_strategy_count.py"""
import json, collections, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
RUNS = {
    "rft1": ("results/e4r/results/rft1_e001nt_baseline_nothink_t06_e4b_cap12", None),
    "rft3": ("results/e4r/results/rft3_e001nt_baseline_nothink_t06_e4b_cap12", None),
    "rft5": ("results/e4r/results/rft5_e001nt_baseline_nothink_t06_e4b_cap12", None),
    "l4": ("outputs/l4run_e001nt/results", "outputs/l4run_e001nt/regraded.jsonl"),
    "w91a": ("results/e4a/results/w91_e001nt_baseline_nothink_t06_e4b_cap12", None),
}
def truthy(v):
    return v in (True, 1, "1", "true", "True")
strat = collections.Counter(); n_edit = 0; runs = 0
lit_ok = collections.Counter()
for label, (rdir, regrade) in RUNS.items():
    rdir = ROOT / rdir
    src = (ROOT / regrade) if regrade else rdir / "task_results.jsonl"
    resolved = set()
    for line in src.read_text().splitlines():
        if line.strip():
            r = json.loads(line); resolved.add(r.get("iid") or r.get("instance_id"))
    for trace in sorted((rdir / "traces").glob("trace_*.json")):
        iid = trace.stem.removeprefix("trace_")
        if iid not in resolved: continue
        runs += 1
        for s in json.loads(trace.read_text())["steps"]:
            if s.get("source") != "agent": continue
            try:
                o = json.loads((s.get("observation") or {}).get("content") or "{}")
            except ValueError:
                o = {}
            if not isinstance(o, dict): o = {}
            for tc in s.get("tool_calls") or []:
                if tc.get("function_name") != "edit_file": continue
                n_edit += 1
                if o.get("status") == "ok":
                    st = o.get("strategy")
                    strat[st] += 1
                    old = str((tc.get("arguments") or {}).get("old_string", ""))
                    if "\\n" in old:
                        lit_ok[st] += 1
print("runs", runs, "edit_file", n_edit, "ok", sum(strat.values()), dict(strat))
print("ok with literal \\n in old_string by strategy", dict(lit_ok), sum(lit_ok.values()))
