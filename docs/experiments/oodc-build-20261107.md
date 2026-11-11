# OOD-C build: a new held-out task set (2026-11-07)

**Status (2026-11-07 21:35 UTC): the four build shards are running (pushed 20:46 UTC, still RUNNING at 21:33). Not done: no shard result read yet, no merge, no dataset. The local watcher was stopped by Claude Code for low memory on the Mac and not restarted; the next step is in "How to resume".**

**Why.** A pre-registered weight-edit gate and a pre-registered live-agent replication for paper 1 need tasks from repositories
no agent of ours has run on. The primary endpoints are measured on the agent's `edit_file` calls, so a task is kept when the
agent can run on it; test grading is recorded per task, not required.

**Acceptance (set before the build).** At least 120 usable tasks (150+ if cheap); every repository disjoint from the training
repos, the 17 OOD-45 repos, OOD-B's 70 kept repos and every repo in `outputs/weight_edit/claude/replay_ood/calls.jsonl`, checked
by script; the harness runs the agent on each kept task; OSI-approved licences only, recorded per task.

## Sources

| Source | Used | Licence | Notes |
|---|---|---|---|
| `nebius/SWE-rebench-leaderboard` (monthly splits 2025-01 .. 2026-03, 860 rows) | yes | CC-BY-4.0 | The rows behind OOD-B. No newer monthly split exists: on 2026-11-07 the Hugging Face tree ends at `2026_03`, and the `test` parquet holds the same 860 rows |
| `nebius/SWE-rebench-V2` (429 MB parquet, 1,074 Python rows from 2024-12 on) | yes, from 2025-01 | CC-BY-4.0 | Not in the brief's list. Added because the leaderboard alone gave 159 candidates under the strict filters. Filtered on a Kaggle kernel so the parquet stayed off the Mac. Kept only rows with the dataset's clean code `A` and a `*_bug` PR category. All rows use one image, `python_base_310` |
| `nebius/SWE-rebench` V1 test split (211 MB; 978 rows from 2024-12 on) | yes, from 2025-01 | CC-BY-4.0 | Same `install_config` format as the leaderboard; 10 candidates after dedupe |
| `nebius/SWE-rebench` V1 `filtered` split (34 MB) | subset of the above for these dates | CC-BY-4.0 | Checked: of 353 rows from 2025, 265 are already leaderboard rows, 6 survive the filters |

## Selection (static, `scripts/oodc/select_candidates.py`)

No repository code runs on the Mac; selection reads parquet and JSON rows only.

```
  2192   707  source rows (distinct instance ids, 2025-01 on)
  1686   649  V2 rows: clean code A and a *_bug PR category
  1296   567  repo not training, OOD-45, OOD-B kept or replay set (canonical name, bare name, fork parent)
   930   423  python 3.10-3.13
   828   395  no conda env, no compiled build
   794   380  apt pre-install only compilers/-dev
   589   323  gold: 1-3 .py files, 2-120 lines, no non-py code
   561   303  issue text >= 250 chars
   439   246  repo licence OSI-approved (GitHub API, 2026-11-07)
   426   238  no `interface` field (rows that add a new API)
   381   227  tests: 1-3 files, F2P in them, <=400 tests (grading only)
   332   197  not a known build or sandbox failure of the OOD-B builder (task or repo)
   258   197  selected (<= 2 per repo; prefer no known gold failure, newest)
months: {'2025-01': 28, '2025-02': 20, '2025-03': 36, '2025-04': 25, '2025-05': 23, '2025-06': 27, '2025-07': 12, '2025-08': 24, '2025-09': 21, '2025-10': 15, '2025-11': 4, '2025-12': 4, '2026-01': 6, '2026-02': 6, '2026-03': 2, '2026-04': 4, '2026-05': 1}
sources: [('nebius/SWE-rebench-leaderboard', 149), ('nebius/SWE-rebench-V2', 99), ('nebius/SWE-rebench', 10)]
interface rows: 0
known gold failures in OOD-B: 25
licences: [('MIT', 107), ('Apache-2.0', 79), ('BSD-3-Clause', 52), ('BSD-2-Clause', 6), ('AGPL-3.0', 6), ('Unlicense', 3), ('MIT-0', 2), ('UPL-1.0', 1), ('GPL-3.0', 1), ('MPL-2.0', 1)]
```

Differences from OOD-B's selection:

- Repositories compared on the name as written, the GitHub canonical name (28 renames found among the repos looked up, for example
  `omni-us/jsonargparse` to `mauvilsa/jsonargparse`), the fork parent and source, and the bare repository name. This caught
  `jawah/niquests`, a fork of `psf/requests`, which a name match would have let through.
- Licences re-read from the GitHub API on 2026-11-07 (`data/ext/oodc/repo_meta.jsonl`); only OSI ids, as in OOD-B. NOASSERTION and
  no-licence repos are dropped.
- Leaderboard rows with an `interface` field (they add a new API, not a bug fix) are dropped, which OOD-B did not do.
- The grading-only test-shape filter was kept, since the pool was large enough without relaxing it. `tests_strict` is true for
  every candidate. Per-repo limit stays 2.
- Tasks and repos that OOD-B's builder showed cannot run are not rebuilt (heavy dependencies, no project at the root, not
  importable, wheel set too large, broken in the sandbox), nor repos that need torch or JAX by design (a short hand list).
- Rank: no known gold failure first, then leaderboard rows, then newest. 25 candidates are OOD-B candidates whose repos have no kept
  task and which failed only the gold gate; they are expected to be usable but not gradable.

## Build and gate (Kaggle CPU kernels, `scripts/oodc/build_env.py`)

Same builder as OOD-B (`scripts/oodb/build_env.py`, copied; OOD-B's scripts are unchanged), with these differences:

1. **History squash at 150 MB instead of 300 MB.** A kernel output holds 20 GB (`df` on `/kaggle/working`), and snapshots were
   over half of OOD-B's 9.2 GB.
2. **Gate.** gold on every task; a second gold and an empty patch only on tasks gold1 resolved; then **a real agent pass** with
   `run_ood45.py` against `scripts/oodc/stub_llm.py`, a stub OpenAI endpoint that scripts `run_command`, a failing `edit_file`,
   `submit_patch` and a final message, using the E4B submission `e001tt_think_t06_e4b`. This runs the whole agent path: sandbox
   setup, ADK loop, tool execution, patch capture, verification.
   - `usable` = agent pass ended with at least 2 counted tool calls and no harness error.
   - `gradable` = gold resolved twice and the empty patch did not.
3. **Merge** keeps `usable` tasks, records `gradable` and `gold_gate` per task, and stops at a 17 GB cap, gradable tasks first, then
   smallest. Tasks over the cap are listed in `validation/over_size_cap.json`.
4. **Host pins** as in `scripts/colab/setup_vm.py` (`litellm==1.101.6`, `mcp<2`, `starlette<1`, `pytest==8.4.2`), since the agent pass
   imports LiteLlm; OOD-B's gate did not need that.

The dataset ships OOD-B's `tools/ood45_harness.py` byte for byte and an unchanged `run_ood45.py`, so the runner needs no flags.

**Smoke** (`ozgung/g4-oodc-smoke` v1, 7 tasks, 12 min, Python 3.13.18): 7 of 7 built, 7 of 7 usable, 3 gradable (3 of the 4 tasks new to
this builder; `confluence_markdown_exporter_92` failed gold, and the 3 known OOD-B gold failures failed gold again). Checked one trace by hand (`twyn_317`): the stub drove `run_command` (real output), `edit_file` (a real
`FileEditError`), `submit_patch` (empty) and a final message. The harness counts 2 tool calls for that script, since `submit_patch` is
not counted.

## Kernels

| Kernel | Version | What | State |
|---|---|---|---|
| `ozgung/g4-oodc-v2filter` | 1 | filter V2 and V1 rows on Kaggle | complete |
| `ozgung/g4-oodc-smoke` | 1 | 7-task smoke | complete |
| `ozgung/g4-oodc-build-s0` .. `-s3` | 1 | 65, 72, 63 and 58 tasks; pushed 20:46 UTC; build deadline 5 h, then the gate | complete by 22:55: 198 built, 189 kept in 150 repos, 60 gradable (`outputs/oodc/report_shards.txt`) |
| `ozgung/g4-oodc-assemble` | 1 | generated in `kernels/oodc_assemble` (attaches the four shards) | complete 01:05 UTC 11-08: 189 tasks in 150 repos, 15 GB, none over the cap |
| `ozgung/g4-oodc-check312` | 1 | generated in `kernels/oodc_check312` (attaches the assemble output) | pushed 01:06 UTC 11-08 |

## How to resume

```bash
# status
for i in 0 1 2 3; do kaggle kernels status ozgung/g4-oodc-build-s$i; done
# small outputs of a finished shard
kaggle kernels output ozgung/g4-oodc-build-s0 -p outputs/oodc/build-s0 --page-size 200 \
    --file-pattern '(build_report\.jsonl|gate/gate\.json|oodc/tasks\.jsonl|kernel\.log|gate/.*\.log|gate/gold1/test_outputs/.*\.log)$'
# merge and check on Python 3.13, then Python 3.12
python3 -I scripts/oodc/make_kernel.py --candidates data/ext/oodc/candidates.jsonl --name assemble \
    --assemble g4-oodc-build-s0,g4-oodc-build-s1,g4-oodc-build-s2,g4-oodc-build-s3
kaggle kernels push -p kernels/oodc_assemble
python3 -I scripts/oodc/make_kernel.py --candidates data/ext/oodc/candidates.jsonl --name check312 --check312 g4-oodc-assemble
kaggle kernels push -p kernels/oodc_check312
# publish from a Colab CPU VM: scripts/oodc/publish_colab.sh (header says what to upload)
```


**2026-11-08 01:1x UTC, disjointness on the final tasks:** `check_disjoint.py outputs/oodc/assemble/oodc_set/tasks.jsonl` finds 0 overlaps with training, OOD-45, OOD-B kept and `replay_ood`. It flags 7 repos under local agent runs, and those are exactly the 7 smoke tasks (`outputs/oodc/smoke/gate/agent`, stub endpoint, no model). The flag is expected: the checker counts any row with tool calls, and the stub makes 2.
