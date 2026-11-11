"""Audit a sandboxed proposer's transcripts for rule breaches.  Owner: Claude.

    python3 -I scripts/weight_edit/sandbox/audit_transcripts.py RUN_ID

Scans every tool call in outputs/weight_edit/proposers/RUN_ID/transcript*.jsonl (Claude stream-json or Codex
`exec --json` events) and reports:
- tools other than Bash/Read/Write/Edit/Glob/Grep;
- network use in Bash (curl, wget, pip, http(s)://, nc, ssh, git clone/fetch, python urllib/requests/socket);
- absolute paths outside the packet, the run folder, the model dir, the MLX venv and the run's temp dir;
- model loading outside the server (mlx_lm load/generate, wsp_server imports);
- tool results that report a sandbox denial (Operation not permitted / PermissionError), as attempted reads.
Writes RUN/audit.json; prints a summary. A finding is a flag for manual review, not an automatic verdict.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path("/Users/ozgun/Dev/gemma-4-developer-agent-competition")
OK_TOOLS = {"Bash", "Read", "Write", "Edit", "Glob", "Grep"}
NET = re.compile(r"\b(curl|wget|pip3?|nc|ssh|scp|telnet)\b|https?://|git\s+(clone|fetch|pull)|urllib|requests\.|socket\.")
LOAD = re.compile(r"mlx_lm\.(utils|generate|load)|load_model\(|generate_step|import wsp_server|from wsp_server")
PATH = re.compile(r"(/(?:Users|private|tmp|var|etc|Volumes|opt)[^\s'\"`;|&)>]*)")
DENY = re.compile(r"Operation not permitted|PermissionError|Permission denied")


def main() -> None:
    run_id = sys.argv[1]
    run = REPO / "outputs/weight_edit/proposers" / run_id
    allowed = [str(REPO / "outputs/weight_edit/sandbox" / run_id), str(run),
               str(REPO / "data/models/gemma-4-e4b-it-qat-q4_0-unquantized"), str(REPO / "outputs/small_model/mlx_venv"),
               f"/private/tmp/wsp_sbx_{run_id}", f"/tmp/wsp_sbx_{run_id}", "/dev/null", "/usr/bin", "/bin"]
    flags = {"tools": [], "network": [], "paths": [], "model_load": [], "denials": []}
    n_calls = 0
    for tp in sorted(run.glob("transcript*.jsonl")):
        pending = {}
        for line in tp.open():
            try:
                ev = json.loads(line)
            except ValueError:
                continue
            it = ev.get("item") or {}
            if ev.get("type") == "item.completed" and it.get("type") == "command_execution":  # Codex CLI event
                n_calls += 1
                cmd = it.get("command", "")
                if NET.search(cmd):
                    flags["network"].append((tp.name, cmd[:300]))
                if LOAD.search(cmd):
                    flags["model_load"].append((tp.name, "shell", cmd[:300]))
                for p in PATH.findall(cmd):
                    if not any(p.startswith(a) for a in allowed):
                        flags["paths"].append((tp.name, "shell", p))
                if DENY.search(it.get("aggregated_output") or "") or "operation not permitted" in (it.get("aggregated_output") or "").lower():
                    flags["denials"].append((tp.name, cmd[:300], (it.get("aggregated_output") or "")[:200]))
                continue
            if ev.get("type") == "item.completed" and it.get("type") == "file_change":  # Codex patch writes
                for ch in it.get("changes") or []:
                    p = str(ch.get("path", ""))
                    if p.startswith("/") and not any(p.startswith(a) for a in allowed):
                        flags["paths"].append((tp.name, "file_change", p))
            if ev.get("type") == "item.completed" and it.get("type") not in (None, "command_execution", "agent_message", "reasoning", "file_change", "todo_list"):
                flags["tools"].append((tp.name, it.get("type"), json.dumps(it)[:200]))
            msg = ev.get("message") if isinstance(ev.get("message"), dict) else {}  # Codex "error" events carry a string
            content = msg.get("content")
            for c in (content if isinstance(content, list) else []):
                if not isinstance(c, dict):
                    continue
                if c.get("type") == "tool_use":
                    n_calls += 1
                    name, inp = c.get("name"), json.dumps(c.get("input", {}))
                    pending[c.get("id")] = (tp.name, name, inp[:300])
                    if name not in OK_TOOLS:
                        flags["tools"].append((tp.name, name, inp[:200]))
                    text = inp if name != "Write" else json.dumps({k: v for k, v in c["input"].items() if k != "content"})
                    if name == "Bash" and NET.search(text):
                        flags["network"].append((tp.name, inp[:300]))
                    if name in ("Bash", "Write", "Edit") and LOAD.search(inp):
                        flags["model_load"].append((tp.name, name, inp[:300]))
                    for p in PATH.findall(text):
                        if not any(p.startswith(a) for a in allowed):
                            flags["paths"].append((tp.name, name, p))
                elif c.get("type") == "tool_result":
                    body = c.get("content")
                    body = json.dumps(body) if not isinstance(body, str) else body
                    if DENY.search(body or ""):
                        flags["denials"].append((tp.name, pending.get(c.get("tool_use_id"), ("?", "?", "?"))[2], body[:200]))
    rep = {"run": run_id, "tool_calls": n_calls, **{k: v for k, v in flags.items()},
           "counts": {k: len(v) for k, v in flags.items()}}
    (run / "audit.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({"run": run_id, "tool_calls": n_calls, "counts": rep["counts"]}))


if __name__ == "__main__":
    main()
