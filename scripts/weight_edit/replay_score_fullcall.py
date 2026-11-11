"""Score full-call generations (replay_fullcall.py) with the harness's own edit function.  Owner: Claude.

    .venv/bin/python -I scripts/weight_edit/replay_score_fullcall.py GEN.jsonl REPLAY_DIR

Per (key, draw): the generated pair (old_string = gen, new_string = gen_new) goes through
swegemma.edit.apply_replacement on the true file state, as the edit_file tool would apply it. Fields:
    outcome / accepted / exact_copy           as replay_score.py
    compiles                                  compile() of the resulting .py file (None: not .py or not accepted)
    artefact_old / artefact_new               generated text has a literal \\n or \\" that the file lacks
    compiles_frozen_new                       the registered definition: generated old_string + the served model's
                                              recorded new_string (what replay_score.py measures)
Output: GEN.scored.jsonl next to the input, and a one-line summary.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from replay_score import compiles, outcome  # noqa: E402


def artefact(text: str, content: str) -> int:
    return int(any(e in text and e not in content for e in ("\\n", '\\"')))


def main() -> None:
    src, replay = Path(sys.argv[1]), Path(sys.argv[2])
    calls = {r["key"]: r for r in map(json.loads, (replay / "calls.jsonl").open())}
    gens = [json.loads(l) for l in src.open()]
    out_p = src.with_suffix(".scored.jsonl")
    n = acc = cf = 0
    with out_p.open("w") as out:
        for g in gens:
            r = calls[g["key"]]
            content = (replay / "files" / f"{r['file_sha']}.txt").read_text()
            oc, new_content = outcome(content, g["gen"], g["gen_new"], r["filepath"], r["allow_multiple"])
            oc_f, new_content_f = outcome(content, g["gen"], r.get("new_string") or "", r["filepath"], r["allow_multiple"])
            row = {"key": g["key"], "draw": g["draw"], "recorded_kind": r["error_kind"], "recorded_strategy": r["strategy"],
                   "outcome": oc, "accepted": int(oc.startswith("applied")),
                   "exact_copy": int(bool(g["gen"]) and g["gen"] in content),
                   "closed_new": g["closed_new"], "closed": g["closed"],
                   "compiles": compiles(new_content, r["filepath"]) if new_content is not None else None,
                   "artefact_old": artefact(g["gen"], content), "artefact_new": artefact(g["gen_new"], content),
                   "same_new_as_recorded": int(g["gen_new"] == (r.get("new_string") or "")),
                   "compiles_frozen_new": compiles(new_content_f, r["filepath"]) if new_content_f is not None else None}
            out.write(json.dumps(row) + "\n")
            n += 1
            acc += row["accepted"]
            cf += int(row["accepted"] and row["compiles"] == 0)
    print(f"scored {n} -> {out_p}: accepted {acc / n:.3f}, compile fail among accepted {cf / max(1, acc):.3f}")


if __name__ == "__main__":
    main()
