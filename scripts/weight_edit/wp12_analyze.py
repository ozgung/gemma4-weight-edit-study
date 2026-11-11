"""WP12 analysis (pre-registered, doc Thread 2026-11-03 UTC; written before any selection-set raw output existed).
Owner: Claude.

    .venv/bin/python -I scripts/weight_edit/wp12_analyze.py SERVED_GEN RAW_GEN --keys KEYS [--draws 2] [--label NAME]

Revised 2026-11-03 17:3x UTC before any WP12 generation (Astra 13:11): the expected universe comes from the key
manifest x draws; duplicates, missing or extra records in either arm, unpaired generation order or an applied edit
BLOCK the analysis (exit, no report) instead of counting as rejections.

SERVED_GEN / RAW_GEN: replay_gen outputs (gen_*.jsonl) with their replay_score files (*.scored.jsonl) next to them.
Same keys, same draws (0 greedy, 1 sampled at T 0.6), same seeds; the only difference is the rendering of tool
responses in the prompt (served JSON string vs Gemma 4 native mapping, scripts/weight_edit/wp12_render.py).

Primary (one test, no family): per task, mean acceptance over its calls x draws; d_t = raw - served.
  H1 supported iff mean(d_t) > 0 AND one-sided exact sign test over tasks with d_t != 0 gives p < 0.05.
Reported with it: 95 % task bootstrap CI of mean(d_t) (10,000 resamples, seed 20261103), call-micro counts.
Secondary (descriptive, no test): accepted-and-compiling rate (fixed denominator), exact-copy rate, same_result among
calls accepted in both arms, breakdown by the recorded served outcome (not_found / multiple / ok), escape artefacts
(generated old_string containing a literal backslash-n or backslash-quote that does not occur in the true file),
generations changed between arms.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FILES = ROOT / "outputs/weight_edit/claude/replay/files"
CALLS = ROOT / "outputs/weight_edit/claude/replay/calls.jsonl"


def _read(path: Path) -> tuple[dict, list]:
    out, order = {}, []
    for r in map(json.loads, path.open()):
        k = (r["key"], r["draw"])
        if k in out:
            raise SystemExit(f"BLOCKED: duplicate record {k} in {path}")
        out[k] = r
        order.append(k)
    return out, order


def load(gen: Path) -> tuple[dict, dict, list]:
    g, order = _read(gen)
    s, _ = _read(gen.with_suffix(".scored.jsonl"))
    return g, s, order


def task(k) -> str:
    return k[0].split("|")[1]


def macro(sc: dict, keys: list, f) -> dict:
    by = collections.defaultdict(list)
    for k in keys:
        by[task(k)].append(f(sc.get(k)))
    return {t: sum(v) / len(v) for t, v in by.items()}


def sign_p(diffs) -> tuple[float, int, int]:
    pos, neg = sum(d > 0 for d in diffs), sum(d < 0 for d in diffs)
    n = pos + neg
    return (sum(math.comb(n, i) for i in range(pos, n + 1)) / 2 ** n if n else 1.0), pos, neg


def boot(diffs, n=10000, seed=20261103) -> list[float]:
    rng = random.Random(seed)
    m = sorted(sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(n))
    return [m[int(0.025 * n)], m[int(0.975 * n) - 1]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("served", type=Path)
    ap.add_argument("raw", type=Path)
    ap.add_argument("--label", default="wp12")
    ap.add_argument("--keys", type=Path, required=True, help="frozen key manifest (one replay key per line)")
    ap.add_argument("--draws", type=int, default=2)
    a = ap.parse_args()
    want = [l.strip() for l in a.keys.read_text().splitlines() if l.strip()]
    if len(set(want)) != len(want):
        raise SystemExit("BLOCKED: duplicate keys in the manifest")
    keys = sorted((k, d) for k in want for d in range(a.draws))
    gs, ss, os_ = load(a.served)
    gr, sr, or_ = load(a.raw)
    # completeness: both arms must cover exactly the expected universe in generation and scoring
    for name, d in (("served gen", gs), ("served scored", ss), ("raw gen", gr), ("raw scored", sr)):
        miss, extra = set(keys) - set(d), set(d) - set(keys)
        if miss or extra:
            raise SystemExit(f"BLOCKED: {name} missing {len(miss)} / extra {len(extra)} records (infrastructure gap, "
                             "recover before the primary analysis)")
    # pairing: same key order in both generation files (seeds are 7919*index+draw in key order), no edit applied
    if [k for k in os_] != [k for k in or_]:
        raise SystemExit("BLOCKED: generation order differs between arms (seeds would not be paired)")
    if any(gs[k].get("edit") or gr[k].get("edit") for k in keys):
        raise SystemExit("BLOCKED: an arm was generated with a weight edit")
    missing_raw = []
    acc = lambda r: int(bool(r["accepted"]))  # noqa: E731  a completed generation with an invalid edit is a rejection
    accc = lambda r: int(bool(r["accepted"] and r["compiles"] == 1))  # noqa: E731
    ms, mr = macro(ss, keys, acc), macro(sr, keys, acc)
    diffs = [mr[t] - ms[t] for t in sorted(ms)]
    p, up, down = sign_p(diffs)
    mean = sum(diffs) / len(diffs)
    calls = {r["key"]: r for r in map(json.loads, CALLS.open())}

    def escape_art(g, k):
        if not g:
            return 0
        txt = g["gen"]  # decoded old_string value as generated (raw text between the <|"|> delimiters)
        content = (FILES / f"{calls[k[0]]['file_sha']}.txt").read_text()
        return int(any(e in txt and e not in content for e in ("\\n", '\\"')))

    def summ(sc, gg, sub):
        n = len(sub)
        return {"n": n, "accepted": sum(acc(sc.get(k)) for k in sub),
                "accepted_and_compiling": sum(accc(sc.get(k)) for k in sub),
                "exact_copy": sum(int(bool(sc.get(k) and sc[k]["exact_copy"])) for k in sub),
                "escape_artefact": sum(escape_art(gg.get(k), k) for k in sub)}

    by_kind = collections.defaultdict(list)
    for k in keys:
        by_kind[ss[k]["recorded_kind"]].append(k)
    both = [k for k in keys if acc(ss.get(k)) and acc(sr.get(k))]
    rep = {
        "label": a.label, "served": str(a.served), "raw": str(a.raw), "n_calls_x_draws": len(keys),
        "n_tasks": len(ms), "missing_raw": len(missing_raw),
        "primary": {"task_macro_mean_diff": mean, "ci95": boot(diffs), "tasks_up": up, "tasks_down": down,
                    "tasks_zero": len(diffs) - up - down, "p_sign_one_sided": p, "h1_supported": bool(mean > 0 and p < 0.05)},
        "served": summ(ss, gs, keys), "raw": summ(sr, gr, keys),
        "same_result_both_accepted": {
            "n": len(both),
            "served": sum(int(ss[k]["same_result"] == 1) for k in both),
            "raw": sum(int(sr[k]["same_result"] == 1) for k in both)},
        "by_recorded_kind": {kind: {"served": summ(ss, gs, sub), "raw": summ(sr, gr, sub)} for kind, sub in sorted(by_kind.items())},
        "changed_generations": sum(int(gs.get(k, {}).get("gen") != gr.get(k, {}).get("gen")) for k in keys),
        "by_draw": {d: {"served": summ(ss, gs, [k for k in keys if k[1] == d]), "raw": summ(sr, gr, [k for k in keys if k[1] == d])}
                    for d in sorted({k[1] for k in keys})},
    }
    out = a.raw.parent / f"{a.label}_report.json"
    out.write_text(json.dumps(rep, indent=1))
    pr = rep["primary"]
    print(f"{a.label}: tasks {rep['n_tasks']}, calls x draws {len(keys)}; accepted served {rep['served']['accepted']} "
          f"raw {rep['raw']['accepted']}; task-macro diff {mean:+.4f} CI [{pr['ci95'][0]:+.4f}, {pr['ci95'][1]:+.4f}] "
          f"up {up} / down {down}, p {p:.4f} -> H1 {pr['h1_supported']}")
    print(f"report -> {out}")


if __name__ == "__main__":
    main()
