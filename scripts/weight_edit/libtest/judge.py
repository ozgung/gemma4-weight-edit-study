"""Judge the usability-test arms on the held-out pydantic cases.  Owner: Claude (fork, libtest).

    <judge venv>/bin/python -I scripts/weight_edit/libtest/judge.py --cases HOLDOUT.jsonl --texts judge_texts.json \
        --criteria criteria.json --seed-base 20261110 --out OUT_DIR  lib=FROZEN_EDIT_DIR nolib=FROZEN_EDIT_DIR

The rule is the one in the plan and the brief (weightprog.evaluate.compare with the frozen Criteria): target bad rate
lower than base and both matched random controls (one-sided sign test over cases, p < alpha each), keep cases broken
per draw at most keep_max_broken, rule-less keep cases changed at most keep_max_changed_frac, judge-text KL and NLL
within limits. Base runs once; each arm's controls are matched to its own edit (seed = seed_base + 10 * i + k, i = arm
index from --first-index, default 1, in the order given, k = 1, 2). An empty edit directory is reported as "no edit"; an edit that fails
validation (module, shape, rank <= 128, finite, no whole-tensor rescale) is reported as invalid. Also scans each
arm's transcripts for gradient-training calls (descriptive flag for the audit; the plan says how it is used).
Writes OUT_DIR/rows_<arm>.jsonl, OUT_DIR/report_<arm>.json, OUT_DIR/judge_summary.json.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

from weightprog.controls import matched_control
from weightprog.edits import EditSet
from weightprog.evaluate import Criteria, compare, open_model, run, side_effects, write_rows
from weightprog.io.safetensors_np import SafetensorsReader
from weightprog.names import ModelSpec
from weightprog.target import load_cases

MODEL = Path("data/models/gemma-4-e4b-it-qat-q4_0-unquantized")
TOOL_LINE = re.compile(r'"tool_use"|"tool_?[cC]all|"function_?[cC]all')  # Claude Code and agy stream-json
GRAD = re.compile(r"value_and_grad|mx\.grad\b|mlx\.optimizers|optim\.(SGD|Adam)|\.backward\(|torch\.autograd|jax\.grad")


def grad_hits(run_dir: Path) -> list[str]:
    hits = []
    for t in sorted(run_dir.glob("transcript_*.jsonl")):
        for line in t.read_text(errors="replace").splitlines():
            if TOOL_LINE.search(line) and GRAD.search(line):
                hits.append(f"{t.name}: {GRAD.search(line).group(0)}")
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True); ap.add_argument("--texts", required=True)
    ap.add_argument("--criteria", required=True); ap.add_argument("--seed-base", type=int, required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--model-dir", default=str(MODEL))
    ap.add_argument("--limit", type=int, help="dry runs only"); ap.add_argument("--backend-kw", default="{}")
    ap.add_argument("--first-index", type=int, default=1, help="control-seed index of the first arm given")
    ap.add_argument("arms", nargs="+")
    a = ap.parse_args()
    model = Path(a.model_dir)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cases = load_cases(a.cases)[: a.limit] if a.limit else load_cases(a.cases)
    texts = json.loads(Path(a.texts).read_text())["texts"]
    crit = Criteria(**json.loads(Path(a.criteria).read_text()))
    log = lambda m: print(f"[{time.strftime('%H:%M:%S', time.gmtime())}] {m}", file=sys.stderr, flush=True)  # noqa: E731
    be = open_model(model, **json.loads(a.backend_kw))
    spec = ModelSpec.from_config(model)
    log("base")
    base_rows = run(be, cases, {"base": None}, draws=2, model_dir=model)
    write_rows(base_rows, out / "rows_base.jsonl")
    summary = {"cases": a.cases, "criteria": json.loads(Path(a.criteria).read_text()), "model_dir": str(model),
               "seed_base": a.seed_base, "first_index": a.first_index, "arms": {}}
    for i, spec_arg in enumerate(a.arms, a.first_index):
        name, d = spec_arg.split("=", 1)
        d = Path(d)
        entry = {"edit_dir": str(d)}
        run_dir = d.parent.parent if d.name == "edit" and d.parent.name == "frozen" else d.parent
        es = EditSet.load_dir(d) if d.is_dir() else EditSet()
        if not len(es):
            entry["verdict"] = "no edit"
            summary["arms"][name] = entry
            log(f"{name}: no edit")
            continue
        try:
            with SafetensorsReader(model) as r:
                entry["validation"] = es.validate(spec, r)
        except Exception as e:  # noqa: BLE001
            entry["verdict"] = f"invalid edit: {e}"
            summary["arms"][name] = entry
            log(f"{name}: invalid: {e}")
            continue
        arms = {"edit": es, "rand1": matched_control(es, seed=a.seed_base + 10 * i + 1),
                "rand2": matched_control(es, seed=a.seed_base + 10 * i + 2)}
        log(f"{name}: generating edit + 2 controls ({len(es)} modules)")
        rows = run(be, cases, arms, draws=2, model_dir=model)
        write_rows(rows, out / f"rows_{name}.jsonl")
        log(f"{name}: side effects")
        side = side_effects(be, {"base": None, **arms}, texts)
        rep = compare(base_rows + rows, side, arm="edit", controls=["rand1", "rand2"], criteria=crit)
        rep["arm"] = name
        rep["edit_dir"] = str(d)
        rep["grad_training_hits"] = grad_hits(run_dir)
        (out / f"report_{name}.json").write_text(json.dumps(rep, indent=1))
        entry.update({"verdict": "PASS" if rep["pass"] else "FAIL", "target_pass": rep["target"]["pass"],
                      "keep_pass": rep["keep"]["pass"], "side_effects_pass": rep["side_effects"]["pass"],
                      "rates": rep["rates"], "grad_training_hits": rep["grad_training_hits"]})
        summary["arms"][name] = entry
        print(f"\n##### {name}\n" + "\n".join(rep["summary"]), flush=True)
    (out / "judge_summary.json").write_text(json.dumps(summary, indent=1))
    log("done")


if __name__ == "__main__":
    main()
