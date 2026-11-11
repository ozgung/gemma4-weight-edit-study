"""Score regenerated old_strings with the harness's own edit function.  Owner: Claude.

    .venv/bin/python -I scripts/weight_edit/replay_score.py outputs/weight_edit/claude/replay/gen_<name>.jsonl [REPLAY_DIR]

For every generation (key, draw, gen) the (true file state, generated old_string, recorded new_string,
allow_multiple) triple goes through swegemma.edit.apply_replacement, as the edit_file tool would do. Outcomes:
    applied_exact | applied_unescaped | applied_flexible | applied_regex   tool accepts (strategy)
    not_found | multiple | empty | other_error                            tool refuses
plus: exact_copy (gen occurs verbatim in the file state), same_as_recorded (gen == the served model's
old_string), recorded outcome, and for accepted edits whether the resulting file still compiles
(compile() of the new content for .py files) and whether it equals the result of the recorded call
(same_result; only defined when the recorded call was accepted).
Output: <input>.scored.jsonl and a summary printed per recorded stratum.
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPLAY = ROOT / "outputs/weight_edit/claude/replay"


def outcome(content: str, old: str, new: str, fp: str, allow_multiple: bool):
    from swegemma.edit import apply_replacement

    if not old:
        return "empty", None
    try:
        res = apply_replacement(content, old, new, filepath=fp, allow_multiple=allow_multiple)
    except ValueError as e:
        m = str(e)
        return ("not_found" if "not found" in m else "multiple" if "occurs" in m or "multiple" in m else "other_error"), None
    except Exception:  # noqa: BLE001
        return "other_error", None
    return f"applied_{res.strategy}", res.new_content


def compiles(text: str, fp: str) -> int | None:
    if not fp.endswith(".py"):
        return None
    try:
        compile(text, fp, "exec")
        return 1
    except (SyntaxError, ValueError):
        return 0


def main() -> None:
    src = Path(sys.argv[1])
    replay = Path(sys.argv[2]) if len(sys.argv) > 2 else REPLAY  # optional replay dir (round 2: replay_ood)
    calls = {}
    for line in (replay / "calls.jsonl").open():
        r = json.loads(line)
        calls[r["key"]] = r
    gens = [json.loads(l) for l in src.open()]
    out_p = src.with_suffix(".scored.jsonl")
    summary = collections.defaultdict(collections.Counter)
    with out_p.open("w") as out:
        for g in gens:
            r = calls[g["key"]]
            content = (replay / "files" / f"{r['file_sha']}.txt").read_text()
            new = r.get("new_string") or ""
            oc, new_content = outcome(content, g["gen"], new, r["filepath"], r["allow_multiple"])
            rec_oc, rec_content = outcome(content, r["old_string"], new, r["filepath"], r["allow_multiple"])
            row = {
                "key": g["key"], "draw": g["draw"], "recorded_kind": r["error_kind"], "recorded_strategy": r["strategy"],
                "recorded_replayed": rec_oc, "outcome": oc, "accepted": int(oc.startswith("applied")),
                "exact_copy": int(bool(g["gen"]) and g["gen"] in content),
                "same_as_recorded": int(g["gen"] == r["old_string"]), "closed": g["closed"],
                "compiles": compiles(new_content, r["filepath"]) if new_content is not None else None,
                "same_result": (int(new_content == rec_content) if (new_content is not None and rec_content is not None) else None),
            }
            out.write(json.dumps(row) + "\n")
            stratum = r["error_kind"] + ("/" + str(r["strategy"]) if r["error_kind"] == "ok" else "")
            d = "greedy" if g["draw"] == 0 else "sampled"
            s = summary[(stratum, d)]
            s["n"] += 1
            s["accepted"] += row["accepted"]
            s["exact_copy"] += row["exact_copy"]
            s["same_as_recorded"] += row["same_as_recorded"]
            s["replay_matches_recorded_outcome"] += int((rec_oc.startswith("applied")) == (r["error_kind"] == "ok"))
    print(f"scored {len(gens)} -> {out_p}")
    for (st, d), c in sorted(summary.items()):
        n = c["n"]
        print(f"  {st:20} {d:7} n={n:3}  accepted {c['accepted']/n:.2f}  exact_copy {c['exact_copy']/n:.2f}  "
              f"same_as_served {c['same_as_recorded']/n:.2f}  (recorded-call replay agrees {c['replay_matches_recorded_outcome']}/{n})")


if __name__ == "__main__":
    main()
