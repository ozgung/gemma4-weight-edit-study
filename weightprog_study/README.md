# weightprog_study

The experiment kit of the Gemma 4 E4B escape-artefact study, built on the general
`weightprog` library (its own repository). It holds the parts that only make sense for that study:

| Module | What it does |
|---|---|
| `weightprog_study.replay` | call records, the Gemma 4 tool-call codec (`<\|"\|>` strings, sorted keys), frozen-prefix and full-call replay of `edit_file` calls |
| `weightprog_study.scoring` | escape-artefact, exact-copy and compile metrics; pluggable replacement function (exact match, or the competition harness's `swegemma.edit.apply_replacement`) |
| `weightprog_study.gate` | the registered gate: `GateSpec` + `analyze()` (paired task-macro sign tests vs base and controls, Holm, guards, fixed-sequence secondary, bootstrap CIs), completeness checks that block, the full-call compile-guard amendment, a pre-registration template |
| `weightprog_study.proposer` | the forward-pass budget server (file queue), the proposer-side numpy kernel, brief template, packet builder, macOS Seatbelt launchers for Claude Code and Codex, transcript audit |
| `skills/edit-gate` | how to pre-register and run the gate |
| `docs/case-study-escape-artefacts.md` | the round-2 case study |
| `scripts/validate_*.py` | real-model validation of weightprog's backends against the study's recorded generations |

```bash
pip install -e /path/to/weightprog -e weightprog_study
weightprog-study --help
pytest -q weightprog_study/tests
WEIGHTPROG_GATE_DIR=outputs/weight_edit/claude/gate_r2 WEIGHTPROG_REPLAY_DIR=outputs/weight_edit/claude/replay_ood \
  pytest -q weightprog_study/tests/test_gate_regression.py   # reproduces the registered gate report exactly
```

The study's own scripts (the ones that produced the registered results) are in `scripts/weight_edit/`; this package
is their tested re-implementation, and the regression test checks that both give the same report.
