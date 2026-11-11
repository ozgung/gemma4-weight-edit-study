# WP7R: pre-registered replication of the agent test on OOD-C

*2026-11-07, Claude (session 534a). Status: **frozen** by the weight-edit log entry of 2026-11-07 23:16 UTC (sha256 of this file), before any OOD-C agent run. Approved in outline by the user on 2026-11-07 about 20:10 UTC (Colab L4 or other GPUs). WP7 itself: `docs/plans/wp7-agent-test-20261107.md`.*

## Question

Served as a LoRA adapter on the competition's W4A16 model in vLLM, does the frozen edit `fable_r2` (Fable-r2) lower the escape-artefact rate of live `edit_file` calls on repositories no agent of ours has run on, compared with the base and a size-matched random adapter?

## Why it is worth running

WP7 found 18 % → 4 % (random 20 %), but as an exploratory result: the edit was chosen after the gate, on the gate's repositories, one run per arm. This run has a fixed edit, a fixed analysis and new repositories, so it can carry a confirmatory claim. It cannot change any round-2 verdict.

## Design

- **Task set:** OOD-C (`docs/experiments/oodc-build-20261107.md`), repositories disjoint from training, OOD-43 and OOD-B. All tasks of the final layout, in its order: the `g4-oodc-assemble` output after `finalize_tasks.py` folds in the Python 3.12 check (`g4-oodc-check312`), whether read from a published dataset or pulled from the kernel output by the chain. At least 100 tasks, or the run does not start and the user is told. Gradability does not matter for the primary.
- **Arms, one run each, one L4 session, one LoRA-enabled vLLM, side by side:** e001nt E4B-NoThink-T0.6 (base), e001nte E4B-NoThink-T0.6-FableEdit, e001ntx E4B-NoThink-T0.6-RandEdit. Adapters sha256 0d5688d7… and bfcdbdb9…. 4 streams per arm, 12-minute cap, hard stop 7 h per arm. Chain: `scripts/colab/chain_20261108_oodc.sh`, role a.
- **Same as WP7 stage 2:** arms, serving, sampling, cap. Only the task set changes.
- **No role b.** The extra base runs (role b) were only for the round-3 gate set; round 3 on the retry target was cancelled by its go/no-go (weight-edit log, 2026-11-07 23:00 UTC), so role b is dropped.

## Analysis

- **Code:** the logic of `scripts/weight_edit/wp7_stage2_analyze.py`, unchanged since 9f14a41. Only the set table gains `"oodc": ("wp7r", "oodc")` (session wp7r, suite oodc) and its docstring one sentence; run as `--set oodc`, so "pooled" equals OOD-C. Snapshot files for the true file state: `wp7_extract_files.py --suite oodc` on the VM, unpacked to `data/ext/ood_files/wp7/oodc/`.
- **Primary:** per-task live escape-artefact rate over counted `edit_file` calls; tasks paired when both arms have a counted call; task-macro difference; one-sided exact sign test, edit lower, against base and against random. Pass: both differences negative and both p below 0.05.
- **Descriptive:** not-found per call; acceptance through the unescape fallback; calls per task; no patch; resolved on the gradable subset, with a difference under about 7 of 115 read as a tie; repository-clustered bootstrap interval.
- **Reading:** a pass confirms that the edit changes what the live agent writes on unseen repositories. It says nothing about resolved tasks, and it does not lift the guard failures of the replay gate.

## Expectation, written before the run

The artefact ratio edit/base lands between 0.2 and 0.6; the random arm stays within noise of the base; not-found per call does not move by more than noise.

## Cost

About 2.5 task runs per minute per L4 session (WP7 stage 2). Role a on about 200 tasks: about 4 h. L4 only, about 6-7 units.
