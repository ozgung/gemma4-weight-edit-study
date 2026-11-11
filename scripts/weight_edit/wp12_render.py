"""WP12 rendering discriminator: re-render tool responses in Gemma 4's native mapping form.  Owner: Claude.

    .venv/bin/python -I scripts/weight_edit/wp12_render.py [--out outputs/weight_edit/claude/wp12/calls_raw.jsonl]

Served path (ADK LiteLlm -> vLLM OpenAI API): every tool result dict is json.dumps'ed into a string, and the Gemma 4
chat template wraps that string as response:<tool>{value:<|"|>{"status": "ok", "content": "line1\\nline2"}<|"|>},
so file text appears JSON-escaped (newline as the two characters \\n, quote as \\").
The same template has a native branch for a mapping response, which the served path never reaches:
response:<tool>{content:<|"|>line1
line2<|"|>,filepath:<|"|>...<|"|>,status:<|"|>ok<|"|>} (keys dictsorted, string values raw).

This script rewrites every tool-response block of each replay prompt from the first form into the second (a Python
port of the template's format_tool_response_block / format_argument mapping branches). Everything else in the prompt,
and the model's own prefix up to old_string, is unchanged. Blocks whose value is not a JSON object stay as served.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPLAY = ROOT / "outputs/weight_edit/claude/replay"
Q = '<|"|>'
BLOCK = re.compile(r'<\|tool_response>response:([^{<]+)\{value:<\|"\|>(.*?)<\|"\|>\}<tool_response\|>', re.S)


def fmt(v, escape_keys: bool = False) -> str:
    """Port of the template's format_argument (escape_keys=False inside tool responses)."""
    if isinstance(v, str):
        return Q + v + Q
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, dict):
        items = sorted(v.items(), key=lambda kv: kv[0].lower())  # jinja dictsort: case-insensitive by key
        return "{" + ",".join((Q + k + Q if escape_keys else k) + ":" + fmt(x, escape_keys) for k, x in items) + "}"
    if isinstance(v, list):
        return "[" + ",".join(fmt(x, escape_keys) for x in v) + "]"
    if v is None:
        return "None"
    return str(v)


def native_block(tool: str, obj: dict) -> str:
    items = sorted(obj.items(), key=lambda kv: kv[0].lower())
    return "<|tool_response>response:" + tool + "{" + ",".join(k + ":" + fmt(x) for k, x in items) + "}<tool_response|>"


def rewrite(prompt: str) -> tuple[str, dict]:
    st = {"blocks": 0, "rewritten": 0, "not_object": 0}

    def sub(m: re.Match) -> str:
        st["blocks"] += 1
        try:
            obj = json.loads(m.group(2))
        except ValueError:
            st["not_object"] += 1
            return m.group(0)
        if not isinstance(obj, dict):
            st["not_object"] += 1
            return m.group(0)
        st["rewritten"] += 1
        return native_block(m.group(1), obj)

    return BLOCK.sub(sub, prompt), st


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=REPLAY / "calls.jsonl")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/weight_edit/claude/wp12/calls_raw.jsonl")
    a = ap.parse_args()
    sys.path.insert(0, str(ROOT / "scripts"))
    import sft_export as X  # noqa: E402
    rnd = X.Renderer(X.DEFAULT_TOKENIZER, X.DEFAULT_TEMPLATE)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    tot = {"calls": 0, "blocks": 0, "rewritten": 0, "not_object": 0, "tok_served": 0, "tok_raw": 0}
    with a.out.open("w") as f:
        for line in a.src.open():
            r = json.loads(line)
            if r.get("prompt"):
                new, st = rewrite(r["prompt"])
                r["prompt_served_tokens"] = len(rnd.encode(r["prompt"]))
                r["prompt"] = new
                r["prompt_raw_tokens"] = len(rnd.encode(new))
                r["wp12_render"] = st
                tot["calls"] += 1
                for k in ("blocks", "rewritten", "not_object"):
                    tot[k] += st[k]
                tot["tok_served"] += r["prompt_served_tokens"]
                tot["tok_raw"] += r["prompt_raw_tokens"]
            f.write(json.dumps(r) + "\n")
    print(json.dumps(tot))


if __name__ == "__main__":
    main()
