"""Census of the unedited Gemma 4 E4B on the pydantic cases (base model only).  Owner: Claude (fork, libtest).

    <weightprog venv>/bin/python -I scripts/weight_edit/libtest/census.py CASES OUT_DIR

Writes OUT_DIR/rows.jsonl (one row per case and draw) and OUT_DIR/summary.json, and prints bad / good / other
rates by role, API and style, plus timing. Uses weightprog.evaluate.run (draw 0 greedy, draw 1 sampled at the served
settings, seeds from the case id).
"""

from __future__ import annotations

import collections
import json
import sys
import time
from pathlib import Path

from weightprog.evaluate import open_model, run, write_rows
from weightprog.target import load_cases

MODEL = Path("data/models/gemma-4-e4b-it-qat-q4_0-unquantized")


def main():
    cases_path, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    cases = load_cases(cases_path)
    t0 = time.time()
    be = open_model(MODEL)
    t_load = time.time() - t0
    t1 = time.time()
    rows = run(be, cases, {"base": None}, draws=2, model_dir=MODEL,
               progress=lambda m: print(m, file=sys.stderr, flush=True))
    t_run = time.time() - t1
    write_rows(rows, out / "rows.jsonl")
    meta = {c.id: c.meta for c in cases}
    agg = collections.defaultdict(collections.Counter)
    for r in rows:
        m = meta[r["id"]]
        key = (r["role"], m.get("api", "-"), m.get("style", "-"), str(m.get("pydantic", "-")))
        agg[key][r["label"] or "unchanged"] += 1
        agg[key]["tokens"] += r["n_tokens"]
    table = []
    for key in sorted(agg):
        c = agg[key]
        n = sum(v for k, v in c.items() if k != "tokens")
        table.append({"role": key[0], "api": key[1], "style": key[2], "pydantic": key[3], "n": n,
                      **{lab: c[lab] / n for lab in ("bad", "good", "other", "unchanged") if c[lab]},
                      "mean_tokens": c["tokens"] / n})
    tok = sum(r["n_tokens"] for r in rows)
    summ = {"cases": str(cases_path), "n_cases": len(cases), "n_rows": len(rows), "load_s": t_load, "run_s": t_run,
            "tokens": tok, "s_per_token": t_run / max(1, tok), "table": table}
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    for t in table:
        print(f"{t['role']:6} {t['api']:13} {t['style']:5} v{t['pydantic']}  n {t['n']:3}  "
              + "  ".join(f"{k} {t[k]:.2f}" for k in ("bad", "good", "other", "unchanged") if k in t)
              + f"  tok {t['mean_tokens']:.0f}")
    print(f"load {t_load:.0f} s, run {t_run:.0f} s, {tok} tokens, {t_run / max(1, tok) * 1000:.0f} ms per token")


if __name__ == "__main__":
    main()
