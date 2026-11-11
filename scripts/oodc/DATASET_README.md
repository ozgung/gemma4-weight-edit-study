# OOD-C: held-out Python bug-fix tasks in the OOD-45 layout

Held-out validation set built for the Gemma 4 Developer Agent competition (team ozgung). Real GitHub issues and their merged fixes
from Python repositories that no agent of ours had run on when the set was built: not in the competition's training set (fastapi, rich, requests,
httpx), not in OOD-45 (`rodrigospalma/ood45-bugfix-split`), not in OOD-B (`ozgung/g4-oodb-pilot`). At most two tasks per
repository. Built for measurements on the agent's `edit_file` calls, so a task is kept when the agent can run on it; whether
it can also be graded is recorded per task and is not a condition.

## Contents

186 tasks in 148 repositories (189 kept by the build; 3 dropped by the Python 3.12 gate: python_bigquery_2135,
python_bigquery_2218, python_pubsub_1318). `gradable` on the shard: 60; of those, the gold patch also passes on the merged
layout for 58 on cp313 and 53 on cp312. Licences: MIT 78, Apache-2.0 50, BSD-3-Clause 39, AGPL-3.0 6, BSD-2-Clause 5,
Unlicense 3, MIT-0 2, MPL-2.0 1, UPL-1.0 1, GPL-3.0 1 (186; counted from the task rows). Used for the WP7R agent-level test in our paper-1 writeup (three arms, one run each).

## Source and licences

- Tasks: the SWE-rebench family on Hugging Face, all licensed **CC-BY-4.0**, issues from 2025-01 on:
  `nebius/SWE-rebench-leaderboard` (monthly splits 2025-01 to 2026-03; 109 tasks), `nebius/SWE-rebench-V2` (72) and
  `nebius/SWE-rebench` V1 (5). Per-task `source` and `source_instance_id` give the original row.
  Attribution: Badertdinov et al., *SWE-rebench: An Automated Pipeline for Task Collection and Decontaminated Evaluation of
  Software Engineering Agents*, NeurIPS 2025, arXiv:2505.20411; Badertdinov, Nekrashevich, Shevtsov, Golubev,
  *SWE-rebench V2: Language-Agnostic SWE Task Collection at Scale*, arXiv:2602.23866.
- Code: each snapshot, patch and test patch keeps its repository's licence (`license` field, SPDX id from the GitHub API on
  2026-11-07; only OSI-approved licences were kept). Wheels keep their own licences.
- Built by `tools/build_env.py` (our code). `tools/ood45_harness.py` is OOD-B's copy of OOD-45's harness (MIT), unchanged.

## Layout (same as OOD-B and OOD-45)

| Path | What |
|---|---|
| `tasks.jsonl` | swegemma task fields plus provenance. `problem_statement` = `problem_statement_original` = the issue text as mined. `hints_text` empty. `PASS_TO_PASS` empty. Extra fields: `gradable`, `gold_gate`, `final_gold_cp313`, `final_gold_cp312`, `tests_strict`, `oodb_candidate` |
| `snapshots/<id>.tgz` | Repository at `base_commit`: `git fast-export` of the history into a fresh repo on `main` (squashed to one commit when the history is over 150 MB; submodules dropped) |
| `wheels/house/` + `wheels/x86_64-cp313/manifest.json`, `wheels/x86_64-cp312/manifest.json` | Offline per-repo environments. cp313: Kaggle notebook and scorer image. cp312: Colab |
| `tools/` | `ood45_harness.py`, `oodb_meta.json` (import name per repo; the harness looks for this file name), `run_ood45.py`, `build_env.py`, `stub_llm.py` |
| `validation/` | `build_report.jsonl` (every candidate: stage, timings, sizes, error), `shard_gate.json`, `final_gate_cp313.json`, `final_gate_cp312.json` |

## Run

```bash
python scripts/colab/run_ood45.py --ood <this dir> --submission-dir <submission> --api-base http://127.0.0.1:8000/v1 \
    --concurrency 4 --results-dir <results>          # add --gold / --baseline to grade the reference / empty patch
```

The host needs swegemma 0.2.11 (or 0.2.7) and pytest 8.4.2, as for OOD-45. `--statement g1` and `g2` are the same text.

## Gate

`usable` (every kept task): `run_ood45.py` drove the real agent path against a stub model (`tools/stub_llm.py`: a shell
command, a failing `edit_file`, `submit_patch`, a final message) with no harness error, on the shard and on this merged layout,
in the harness's subprocess sandbox on Kaggle CPU kernels. `gradable`: gold resolved twice and the empty patch did not, on
the shard. Solved counts are only meaningful on `gradable` tasks.

## Limitations

- Statements are the raw issue text: some name the file or function to change, or include the fix. Not audited.
- Created 2025-01 to 2026-05; `created_at` is per task. Gemma 4's stated pre-training cutoff is January 2025.
- The subprocess sandbox sees the host's site-packages behind the task wheels.
- The subprocess sandbox does not confine shell commands, so an agent can read this directory (gold patch included).
