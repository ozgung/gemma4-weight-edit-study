"""Masked-copy check for the retry target (go/no-go 3b; rule in the weight-edit log, entry 2026-11-07 23:13 UTC).  Owner: Claude.

    python3 -I scripts/weight_edit/retry_mask_calls.py

For each of the 71 held-out retry calls (keys_heldout_retry.txt), copies the replay row from replay_ood/calls.jsonl and
deletes every occurrence of every string in the call's failed set F(c) from the model-authored parts of the prompt and
of the current-turn prefix: model text and tool-call arguments. Environment text is never touched: system and user
turns and tool responses (file views, command output, error messages) are copied byte for byte. Longer strings first.
The not-found error responses therefore stay, so the call is still a retry; only the copy source is gone.
Output: outputs/weight_edit/claude/retry_gonogo/mask/calls.jsonl (same keys, same fields, plus mask counts).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
O = ROOT / "outputs/weight_edit/claude/retry_gonogo"
SRC = ROOT / "outputs/weight_edit/claude/replay_ood/calls.jsonl"
ENV = re.compile(r"<\|tool_response>.*?<tool_response\|>|<\|turn>(?:user|system)\n.*?<turn\|>", re.S)


def mask(text: str, strings: list[str]) -> tuple[str, int]:
    """Delete the strings outside environment spans; return the new text and the number of deletions."""
    out, n, last = [], 0, 0
    spans = [(m.start(), m.end()) for m in ENV.finditer(text)] + [(len(text), len(text))]
    for a, b in spans:
        seg = text[last:a]
        for s in strings:
            n += seg.count(s)
            seg = seg.replace(s, "")
        out.append(seg)
        out.append(text[a:b])
        last = b
    return "".join(out), n


def main() -> None:
    keys = [l.strip() for l in (O / "keys_heldout_retry.txt").read_text().splitlines() if l.strip()]
    failed = json.load(open(O / "failed_sets.json"))
    want = set(keys)
    (O / "mask").mkdir(exist_ok=True)
    n_rows = n_prompt = n_prefix = 0
    with open(O / "mask/calls.jsonl", "w") as f:
        for line in SRC.open():
            r = json.loads(line)
            if r["key"] not in want:
                continue
            strings = sorted({s for s in failed[r["key"]] if s}, key=len, reverse=True)
            env_before = ENV.findall(r["prompt"])
            r["prompt"], a = mask(r["prompt"], strings)
            r["prefix"], b = mask(r["prefix"], strings)
            assert ENV.findall(r["prompt"]) == env_before, r["key"]
            r["mask_prompt_deletions"], r["mask_prefix_deletions"] = a, b
            f.write(json.dumps(r) + "\n")
            n_rows += 1
            n_prompt += a
            n_prefix += b
    print(f"rows {n_rows} of {len(keys)}; deletions prompt {n_prompt}, prefix {n_prefix}")


if __name__ == "__main__":
    main()
