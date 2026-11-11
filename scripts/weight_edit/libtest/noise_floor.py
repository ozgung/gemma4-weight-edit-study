"""Noise floor of the usability test's rule-less keep check (post hoc, descriptive).  Owner: Claude (paper session).

    outputs/weight_edit/libtest/envs/lib/bin/python -I scripts/weight_edit/libtest/noise_floor.py \
        --out outputs/weight_edit/libtest/noise_floor --k 20 [--base-only]

Question: how often does a size-matched random edit change the greedy output of unrelated plain-code prompts, and
where do the three frozen edits (lib, nolib, libfable) fall in that distribution? The registered verdicts stand; this
only measures how sensitive the four-prompt check was. Registered in docs/experiments/weight-edit-collab-20261102.md
(2026-11-11) before any edited generation.

Prompts: the 8 rule-less plain-code cases of the test (4 holdout, 4 tune; same template) plus the 40 below, written
for this measurement before any run. Greedy only (draw 0), max_new 160, stop at ```.
Arms per frozen edit i (lib=1, nolib=2, libfable=3, the judge's indices): the edit; the judge's two controls
(seed 20261110 + 10 i + k, k = 1, 2: a reproduction check of the registered counts); K new matched controls
(seed 31111100 + 100 i + k, k = 1..K; same weightprog.controls.matched_control, same modules, ranks and norms).
Output: rows_<name>.jsonl and summary.json (per configuration: prompts whose greedy text differs from base, over
all 48 and over the 4 holdout prompts of the registered check).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from weightprog.controls import matched_control
from weightprog.edits import EditSet
from weightprog.evaluate import open_model, run, write_rows
from weightprog.target import Case, load_cases

D = Path("outputs/weight_edit/libtest")
MODEL = Path("data/models/gemma-4-e4b-it-qat-q4_0-unquantized")
ARMS = [("lib", 1), ("nolib", 2), ("libfable", 3)]
HOLDOUT4 = ["h-plain-slug", "h-plain-chunks", "h-plain-pal", "h-plain-dur"]

NEW = [  # (name, task); prompt = "Write a function `<task>. Reply with the code only."
    ("reverse-words", "reverse_words(s: str) -> str` that reverses the order of the words in a sentence"),
    ("vowels", "count_vowels(s: str) -> int` that counts vowels, ignoring case"),
    ("flatten", "flatten(xs)` that flattens a nested list of any depth"),
    ("prime", "is_prime(n: int) -> bool`"),
    ("gcd", "gcd(a: int, b: int) -> int` using Euclid's algorithm"),
    ("rle", "rle_encode(s: str) -> str` that run-length encodes a string, so 'aaab' becomes 'a3b1'"),
    ("rotate", "rotate(xs, k)` that rotates a list to the right by k places"),
    ("dedupe", "dedupe(xs)` that removes duplicates from a list while keeping the first occurrence order"),
    ("camel-snake", "camel_to_snake(name: str) -> str` that converts CamelCase to snake_case"),
    ("snake-camel", "snake_to_camel(name: str) -> str` that converts snake_case to camelCase"),
    ("leap", "is_leap_year(year: int) -> bool`"),
    ("transpose", "transpose(matrix)` that transposes a list of lists"),
    ("bsearch", "binary_search(xs, target) -> int` that returns the index of target in a sorted list, or -1"),
    ("roman-int", "roman_to_int(s: str) -> int`"),
    ("int-roman", "int_to_roman(n: int) -> str`"),
    ("capwords", "capitalize_words(s: str) -> str` that capitalizes the first letter of every word"),
    ("median", "median(xs) -> float` that returns the median of a list of numbers"),
    ("anagram", "are_anagrams(a: str, b: str) -> bool` that ignores case and spaces"),
    ("fizzbuzz", "fizzbuzz(n: int) -> list[str]` for the numbers 1 to n"),
    ("bytes-human", "bytes_to_human(n: int) -> str` so that 1536 becomes '1.5 KB'"),
    ("query", "parse_query(qs: str) -> dict` that parses 'a=1&b=2' into {'a': '1', 'b': '2'}"),
    ("chunk-str", "chunk_string(s: str, n: int) -> list[str]` that splits a string into pieces of length n"),
    ("kadane", "max_subarray(xs) -> int` that returns the largest sum of a contiguous sublist"),
    ("parens", "balanced(s: str) -> bool` that checks that (), [] and {} are balanced"),
    ("deep-merge", "deep_merge(a: dict, b: dict) -> dict` that merges nested dicts, with b winning on conflicts"),
    ("two-sum", "two_sum(xs, target)` that returns the indices of two numbers that add up to target"),
    ("c-to-f", "celsius_to_fahrenheit(c: float) -> float`"),
    ("truncate", "truncate(text: str, n: int) -> str` that cuts text to n characters and adds '...' if it was longer"),
    ("ipv4", "is_valid_ipv4(s: str) -> bool` for dotted-quad addresses"),
    ("matmul", "matmul(a, b)` that multiplies two matrices given as lists of lists"),
    ("group-by", "group_by(xs, key)` that returns a dict mapping key(x) to the list of matching items"),
    ("levenshtein", "levenshtein(a: str, b: str) -> int` that returns the edit distance"),
    ("sum-digits", "sum_digits(n: int) -> int` that adds up the decimal digits of n"),
    ("intervals", "merge_intervals(intervals)` that merges overlapping [start, end] pairs"),
    ("strip-html", "strip_tags(s: str) -> str` that removes HTML tags with a regular expression"),
    ("days", "days_between(a: str, b: str) -> int` for two ISO dates like '2024-01-31'"),
    ("whitespace", "normalize_whitespace(s: str) -> str` that collapses runs of whitespace into one space"),
    ("top-k", "top_k(xs, k)` that returns the k largest items in descending order"),
    ("hex-rgb", "hex_to_rgb(h: str) -> tuple[int, int, int]` for colours like '#1a2b3c'"),
    ("primes-upto", "primes_up_to(n: int) -> list[int]` using the sieve of Eratosthenes"),
]


def cases() -> list[Case]:
    old = [c for f in ("cases_holdout.jsonl", "cases_tune.jsonl") for c in load_cases(D / "sets" / f)
           if "plain" in c.id]
    assert len(old) == 8 and set(HOLDOUT4) <= {c.id for c in old}, [c.id for c in old]
    new = [Case(id=f"n-{n}", messages=[{"role": "user", "content": f"Write a function `{t}. Reply with the code only."}],
                prefill="```python\n", role="keep", split="holdout", group=f"noise-{n}", max_new=160, stop=["```"],
                meta={"api": "none"}) for n, t in NEW]
    assert len(new) == 40 and len({c.id for c in new}) == 40
    return old + new


def greedy(rows: list[dict]) -> dict:
    out: dict = {}
    for r in rows:
        if r["draw"] == 0:
            out.setdefault(r["arm"], {})[r["id"]] = r["text"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--model-dir", default=str(MODEL)); ap.add_argument("--base-only", action="store_true")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    log = lambda m: print(f"[{time.strftime('%H:%M:%S', time.gmtime())}] {m}", file=sys.stderr, flush=True)  # noqa: E731
    cs = cases()
    be = open_model(Path(a.model_dir))
    t0 = time.time()
    base_path = out / "rows_base.jsonl"
    if base_path.exists():
        base_rows = [json.loads(l) for l in base_path.read_text().splitlines() if l.strip()]
    else:
        base_rows = run(be, cs, {"base": None}, draws=1, model_dir=a.model_dir)
        write_rows(base_rows, base_path)
        log(f"base: {len(cs)} prompts in {time.time() - t0:.0f} s, "
            f"{sum(r['n_tokens'] for r in base_rows)} tokens")
    base = greedy(base_rows)["base"]
    judge_base = greedy([json.loads(l) for l in (D / "judge" / "rows_base.jsonl").read_text().splitlines()])["base"]
    repro_base = {c: base[c] == judge_base[c] for c in HOLDOUT4}
    log(f"base greedy equals the judge's on the 4 holdout prompts: {repro_base}")
    summary = {"k": a.k, "prompts": len(cs), "holdout4": HOLDOUT4, "repro_base": repro_base, "arms": {}}
    if a.base_only:
        (out / "summary_base.json").write_text(json.dumps(summary, indent=1))
        return
    for name, i in ARMS:
        es = EditSet.load_dir(D / "runs" / name / "frozen" / "edit")
        arms = {"edit": es,
                **{f"judge{k}": matched_control(es, seed=20261110 + 10 * i + k) for k in (1, 2)},
                **{f"r{k:02d}": matched_control(es, seed=31111100 + 100 * i + k) for k in range(1, a.k + 1)}}
        t1 = time.time()
        rp = out / f"rows_{name}.jsonl"
        if rp.exists():  # resume: an arm whose rows were written is not rerun
            rows = [json.loads(l) for l in rp.read_text().splitlines() if l.strip()]
        else:
            rows = run(be, cs, arms, draws=1, model_dir=a.model_dir)
            write_rows(rows, rp)
        g = greedy(rows)
        per = {}
        for arm, texts in g.items():
            ch = sorted(c for c in texts if texts[c] != base[c])
            per[arm] = {"changed_all": len(ch), "changed_holdout4": sum(c in ch for c in HOLDOUT4),
                        "changed_ids": ch}
        ctrl = [per[f"r{k:02d}"] for k in range(1, a.k + 1)]
        summary["arms"][name] = {
            "per_config": per,
            "edit_changed_all": per["edit"]["changed_all"], "edit_changed_holdout4": per["edit"]["changed_holdout4"],
            "controls_changed_all": [c["changed_all"] for c in ctrl],
            "controls_fail_holdout4": sum(c["changed_holdout4"] >= 1 for c in ctrl),
            "judge_controls_changed_holdout4": [per["judge1"]["changed_holdout4"], per["judge2"]["changed_holdout4"]],
            "seconds": round(time.time() - t1)}
        log(f"{name}: edit changed {per['edit']['changed_all']}/{len(cs)} (holdout4 {per['edit']['changed_holdout4']}); "
            f"controls {sorted(c['changed_all'] for c in ctrl)}; fail holdout4 "
            f"{summary['arms'][name]['controls_fail_holdout4']}/{a.k}; judge controls "
            f"{summary['arms'][name]['judge_controls_changed_holdout4']}; {time.time() - t1:.0f} s")
        (out / "summary.json").write_text(json.dumps(summary, indent=1))
    log("done")


if __name__ == "__main__":
    main()
