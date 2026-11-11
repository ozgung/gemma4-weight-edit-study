"""Call-level table of every edit_file / write_file call in the WP1 discovery runs.  Owner: Claude.

    python3 -I scripts/weight_edit/edit_calls.py --out outputs/weight_edit/claude/edit_calls.csv

Discovery data only (training repos; the same runs as census.py). One row per call, with the outcome
the tool reported and, for `old_string not found`, a subtype computed against the text of that file the
agent had seen *before* this call (read_file output, plus run_command stdout whose command names the
file), reset after each successful edit of the file.

error_kind: ok | not_found | multiple (target occurs N times) | missing_param (ADK argument validation,
            observation has only an "error" key) | other
subtype (not_found only), first match wins:
    not_seen        no text of the file seen before the call
    in_seen         old_string occurs verbatim in the seen text (stale view, or the file changed)
    escape          not in seen text, but unescaping \\n \\t \\" \\' \\\\ makes it occur (genuine extra
                    escaping; a literal backslash sequence that is in the source does not trigger this)
    whitespace      occurs after normalising whitespace per line
    escape_ws       occurs after both
    noncontiguous   every non-blank line occurs in the seen text, but not as one block
    partial         at least half of the non-blank lines occur
    invented        under half occur
Arguments in the trace are already JSON-decoded, so a literal backslash-n in old_string means the model
emitted an escaped backslash (caveat: assumes the vLLM gemma4 tool parser passes arguments through).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from census import ROOT, RUNS, truthy  # noqa: E402

UNESC = [("\\\\", "\x00"), ("\\n", "\n"), ("\\t", "\t"), ('\\"', '"'), ("\\'", "'"), ("\x00", "\\")]


def unescape(s: str) -> str:
    for a, b in UNESC:
        s = s.replace(a, b)
    return s


def norm_ws(s: str) -> str:
    return "\n".join(" ".join(l.split()) for l in s.splitlines() if l.strip())


def parse_obs(step: dict) -> dict:
    try:
        o = json.loads((step.get("observation") or {}).get("content") or "{}")
        return o if isinstance(o, dict) else {}
    except ValueError:
        return {}


def error_kind(o: dict) -> str:
    if o.get("status") == "ok":
        return "ok"
    msg = str(o.get("error_message") or o.get("error") or "")
    if "not found" in msg:
        return "not_found"
    if "occurs" in msg and "times" in msg:
        return "multiple"
    if "mandatory input parameters" in msg:
        return "missing_param"
    return "other"


def subtype(old: str, seen: str) -> tuple[str, float]:
    lines = [l.strip() for l in old.splitlines() if l.strip()]
    seen_lines = {l.strip() for l in seen.splitlines()}
    cov = sum(1 for l in lines if l in seen_lines) / len(lines) if lines else 0.0
    if not seen:
        return "not_seen", cov
    if old and old in seen:
        return "in_seen", cov
    u = unescape(old)
    if u != old and u in seen:
        return "escape", cov
    if norm_ws(old) and norm_ws(old) in norm_ws(seen):
        return "whitespace", cov
    if u != old and norm_ws(u) and norm_ws(u) in norm_ws(seen):
        return "escape_ws", cov
    if cov == 1:
        return "noncontiguous", cov
    return ("partial" if cov >= 0.5 else "invented"), cov


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/weight_edit/claude/edit_calls.csv")
    args = ap.parse_args()
    rows = []
    for label, (rdir, regrade) in RUNS.items():
        rdir = ROOT / rdir
        resolved = {}
        src = (ROOT / regrade) if regrade else rdir / "task_results.jsonl"
        for line in src.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                resolved[r.get("iid") or r.get("instance_id")] = int(truthy(r.get("resolved")))
        for trace in sorted((rdir / "traces").glob("trace_*.json")):
            iid = trace.stem.removeprefix("trace_")
            if iid not in resolved:
                continue
            steps = json.loads(trace.read_text())["steps"]
            seen: dict[str, str] = {}
            edit_idx = 0
            agent_idx = -1
            for s in steps:
                if s.get("source") != "agent":
                    continue
                agent_idx += 1
                o = parse_obs(s)
                for tc in s.get("tool_calls") or []:
                    name = tc.get("function_name")
                    a = tc.get("arguments") or {}
                    fp = a.get("filepath")
                    if name == "read_file" and o.get("status") == "ok" and fp:
                        seen[fp] = seen.get(fp, "") + "\n" + str(o.get("content", ""))
                    elif name == "run_command" and o.get("status") == "ok":
                        cmd = str(a.get("command", ""))
                        for f in list(seen) + re.findall(r"[\w./-]+\.py", cmd):
                            if f and f in cmd:
                                seen[f] = seen.get(f, "") + "\n" + str(o.get("stdout", ""))
                    if name not in ("edit_file", "write_file"):
                        continue
                    kind = error_kind(o)
                    old = str(a.get("old_string", "")) if name == "edit_file" else ""
                    new = str(a.get("new_string", a.get("content", "")))
                    st, cov = subtype(old, seen.get(fp, "")) if kind == "not_found" else ("", "")
                    m = s.get("metrics") or {}
                    rows.append({
                        "run": label, "instance_id": iid, "run_resolved": resolved[iid],
                        "agent_step": agent_idx, "edit_idx": edit_idx, "tool": name, "filepath": fp,
                        "error_kind": kind, "subtype": st, "line_coverage": round(cov, 2) if cov != "" else "",
                        "read_before": int(bool(seen.get(fp))),
                        "old_chars": len(old), "old_lines": old.count("\n") + 1 if old else 0,
                        "new_chars": len(new),
                        "old_lit_bs_n": int("\\n" in old), "old_lit_bs_quote": int('\\"' in old or "\\'" in old),
                        "seen_lit_bs_n": int("\\n" in seen.get(fp, "")),
                        "old_max_indent": max((len(l) - len(l.lstrip()) for l in old.splitlines() if l.strip()), default=0),
                        "prompt_tokens": m.get("prompt_tokens", ""), "completion_tokens": m.get("completion_tokens", ""),
                    })
                    edit_idx += 1
                    if kind == "ok" and fp:
                        seen[fp] = ""  # the file changed; earlier views are stale
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} edit calls -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
