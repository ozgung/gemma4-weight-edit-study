"""Stub OpenAI-compatible chat endpoint for the OOD-C harness check (stdlib only, no model).

    python stub_llm.py --port 8123

It lets `run_ood45.py` drive the real agent path (sandbox setup, ADK loop, tool execution, patch capture, verification)
without a model. Script per conversation, chosen by the number of tool results already in the request:
run_command, then edit_file with an old_string that cannot match, then submit_patch, then a final text. The agent
therefore ends with an empty patch and the run records tool_calls == 3 when the tool path works.
"""

from __future__ import annotations

import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

STEPS = [
    ("run_command", {"command": "git log --oneline -1; ls | head -5; python3 -c 'import sys; print(sys.version)'"}),
    ("edit_file", {"filepath": "setup.py", "old_string": "__oodc_stub_no_such_string__", "new_string": "x"}),
    ("submit_patch", {}),
]


def reply(body: dict) -> dict:
    done = sum(m.get("role") == "tool" for m in body.get("messages", []))
    names = {t.get("function", {}).get("name") for t in body.get("tools") or []}
    msg: dict = {"role": "assistant", "content": ""}
    finish = "stop"
    if done < len(STEPS) and STEPS[done][0] in names:
        name, args = STEPS[done]
        msg["tool_calls"] = [{"id": f"call_{done}", "type": "function",
                              "function": {"name": name, "arguments": json.dumps(args)}}]
        finish = "tool_calls"
    else:
        msg["content"] = "Stub run finished."
    return {"id": f"stub-{time.time_ns()}", "object": "chat.completion", "created": int(time.time()),
            "model": body.get("model", "stub"), "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        data = json.dumps({"object": "list", "data": [{"id": "stub", "object": "model"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
        r = reply(body)
        if body.get("stream"):
            c = r["choices"][0]
            chunk = {**r, "object": "chat.completion.chunk",
                     "choices": [{"index": 0, "delta": c["message"], "finish_reason": c["finish_reason"]}]}
            if chunk["choices"][0]["delta"].get("tool_calls"):
                chunk["choices"][0]["delta"]["tool_calls"][0]["index"] = 0
            data = f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n".encode()
            ctype = "text/event-stream"
        else:
            data = json.dumps(r).encode()
            ctype = "application/json"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8123)
    a = ap.parse_args()
    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
