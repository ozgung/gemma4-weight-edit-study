"""WP7 stage 1: the W4A16 vLLM replay against the go rule fixed in docs/plans/wp7-agent-test-20261107.md.  Owner: Claude.

    .venv/bin/python -I scripts/weight_edit/replay_score.py <D>/gen_<arm>.jsonl outputs/weight_edit/claude/replay_ood
        (for arm in base edit rand), then
    .venv/bin/python -I scripts/weight_edit/wp7_stage1_score.py <D>     (D: the pulled results/wp7a/results/wp7_w4a16)

Go rule for stage 2: edit escape artefacts <= 0.75 x the W4A16 base count, and the random adapter within +-10 % of it.
Artefact, served-ok loss and compile follow gate_r2_analyze.py (imported). Reported beside the rule: the same figures for
the bf16 MLX gate arms (base, fable_r2, fable_r2_rand1), the gate's primary test (task-macro sign test) on W4A16, and
greedy agreement between the W4A16 and bf16 base generations (the serving gap itself). Writes <D>/stage1_report.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gate_r2_analyze as GA  # noqa: E402

ARMS = {"base": "base", "edit": "fable_r2", "rand": "fable_r2_rand1"}  # W4A16 arm -> bf16 gate arm


def load(gen: Path, scored: Path, keys: list) -> dict:
    g, s = GA._read(gen), GA._read(scored)
    for name, d in (("gen", g), ("scored", s)):
        if set(d) != set(keys):
            GA.block(f"{name} {gen.name}: {len(set(keys) - set(d))} missing, {len(set(d) - set(keys))} extra")
    for k in keys:
        s[k]["escape_artefact"] = GA.artefact(k[0], g[k]["gen"])
        s[k]["gen"] = g[k]["gen"]
    return s


def figures(e: dict, base: dict, keys: list) -> dict:
    out = {"artefacts": sum(e[k]["escape_artefact"] for k in keys), "accepted": sum(e[k]["accepted"] for k in keys),
           "closed": sum(e[k]["closed"] for k in keys)}
    acc = [e[k] for k in keys if e[k]["accepted"]]
    out["accepted_and_compiling"] = sum(1 for r in acc if r["compiles"] == 1)
    out["compile_fail_rate_among_accepted"] = sum(1 for r in acc if r["compiles"] == 0) / len(acc) if acc else None
    if e is not base:
        out["served_ok_lost_per_draw"] = [sum(1 for k in keys if k[1] == d and base[k]["recorded_kind"] == "ok"
                                              and base[k]["accepted"] and not e[k]["accepted"]) for d in range(GA.DRAWS)]
        out["primary_vs_base"] = GA.compare(e, base, keys, "escape_artefact", lower_better=True)
        out["acceptance_vs_base"] = GA.compare(e, base, keys, "accepted")
    return out


def main() -> None:
    d = Path(sys.argv[1])
    want = [l.strip() for l in (GA.R / "usable_keys.txt").read_text().splitlines() if l.strip()]
    GA.CALLS.update({r["key"]: r for r in map(json.loads, (GA.R / "calls.jsonl").open())})
    keys = [(k, dr) for k in want for dr in range(GA.DRAWS)]
    w4 = {a: load(d / f"gen_{a}.jsonl", d / f"gen_{a}.scored.jsonl", keys) for a in ARMS}
    bf = {a: load(GA.G / f"gen_{b}.jsonl", GA.G / f"gen_{b}.scored.jsonl", keys) for a, b in ARMS.items()}
    rep = {"w4a16": {a: figures(w4[a], w4["base"], keys) for a in ARMS},
           "bf16": {a: figures(bf[a], bf["base"], keys) for a in ARMS}}
    b, e, r = (rep["w4a16"][a]["artefacts"] for a in ARMS)
    rep["go_rule"] = {"edit_over_base": e / b if b else None, "rand_over_base": r / b if b else None,
                      "edit_ok": b > 0 and e <= 0.75 * b, "rand_ok": b > 0 and abs(r - b) <= 0.10 * b}
    rep["go_rule"]["go"] = rep["go_rule"]["edit_ok"] and rep["go_rule"]["rand_ok"]
    g0 = [k for k in keys if k[1] == 0]
    rep["greedy_same_text_w4a16_vs_bf16"] = {a: [sum(w4[a][k]["gen"] == bf[a][k]["gen"] for k in g0), len(g0)] for a in ARMS}
    (d / "stage1_report.json").write_text(json.dumps(rep, indent=1))
    for prec in ("w4a16", "bf16"):
        for a in ARMS:
            f = rep[prec][a]
            extra = (f"  served-ok lost {f['served_ok_lost_per_draw']}  primary p {f['primary_vs_base']['p_sign_one_sided']:.2g} "
                     f"({f['primary_vs_base']['tasks_better']}/{f['primary_vs_base']['tasks_worse']})") if a != "base" else ""
            print(f"{prec:6} {a:5} artefacts {f['artefacts']:4}  accepted {f['accepted']:4}  acc+compiling "
                  f"{f['accepted_and_compiling']:4}  closed {f['closed']:4}{extra}")
    print("greedy same text W4A16 vs bf16:", rep["greedy_same_text_w4a16_vs_bf16"])
    print("go rule:", rep["go_rule"])


if __name__ == "__main__":
    main()
