# Gemma 4 E4B weight-edit study

This repository holds the code, protocols, records and agent configurations behind two Kaggle writeups:

- Paper 1, *Frontier Models as Weight Editors for Gemma 4 E4B*: `docs/paper/writeup-3000.md`, its full version
  `docs/paper/full/paper-full.md`, and the records behind every number in `docs/paper/full/records.md`.
- Paper 2, the resource writeup on `weightprog`, OOD-C and the replay set: `docs/paper2/writeup-resource.md`.

## Layout

| Path | What it holds |
|---|---|
| `docs/paper/`, `docs/paper2/` | The two papers and Paper 1's records. |
| `docs/experiments/weight-edit-collab-20261102.md` | The study log. The records cite its entries by UTC time. |
| `docs/experiments/oodc-build-20261107.md` | How OOD-C was built. |
| `docs/plans/` | The plans written before each test: the live agent tests (WP7, WP7R, WP7T), edit against training, the repair arm and the usability test. |
| `weightprog_study/` | The study kit, a tested package on top of `weightprog`: call replay and scoring, the registered gate, the forward-pass budget server, sandbox launchers and transcript audit. See its own README. |
| `scripts/weight_edit/` | The scripts that produced the registered results: replay set, generation, scoring, gates, the editors' server and sandbox, the retry checks, and the usability test (`libtest/`). |
| `scripts/mlx_lora/` | MLX LoRA training for Gemma 4 E4B, used for the gradient-trained arms (Paper 1, Appendix J), and the model helpers the editors' server imports. |
| `scripts/oodc/` | The OOD-C build: candidate selection, environments, the disjointness check and the report. |
| `scripts/colab/run_ood45.py` | The runner for the live agent tests on OOD task sets. OOD-C ships an unchanged copy. |
| `scripts/paper/` | Figure scripts, the full-paper build and the word count. |
| `kernels/wp7t_files_extract/` | The Kaggle kernel that recovered the files touched in the thinking-on live run. |
| `submissions/` | The three agent configurations used in the live runs: e001nt (thinking off), e001tt (thinking on) and the serving manifest of the thinking-on run. |
| `figures/` | The paper figures as PNG and PDF: `v4/` for Paper 1 (English and Turkish), `resource/` for Paper 2. |
| `paper/` | PDFs: Paper 1's writeup and full paper, Paper 2's writeup. |

The scripts keep the paths of the original project (for example `outputs/weight_edit/claude/...`). Many of them read
run outputs, agent traces, model weights or the competition harness, which are not part of this repository.
`scripts/weight_edit/replay_set.py` and `replay_set_ood.py` import the prompt-rendering module `scripts/sft_export.py`
from the competition pipeline. They are kept as the record of how the replay set was built; the set itself is in the
Kaggle dataset below.

## Related resources

- `weightprog`, the editing library, is a separate repository: github.com/ozgung/weightprog.
- The replay set (320 held-out `edit_file` calls with prompts and file states), the frozen gate reports and the
  usability-test records are a Kaggle dataset: [link].
- OOD-C, 186 tasks for offline agent runs, is a Kaggle dataset: [link].

## Commit hashes

Paper 1's Appendix D cites commit hashes, for example the commit that registered the round-2 rule. They refer to the
project's private development history, which will be published after the competition ends on 2026-12-02. This
repository is a snapshot of the study's files in a single commit.

## Running the study kit's tests

The kit needs Python 3.10 or later, `weightprog` 0.2 and numpy. The unit tests build a tiny random Gemma 4 checkpoint;
the tests for each backend run when `mlx` or `torch` is installed and are skipped otherwise.

```bash
pip install -e /path/to/weightprog -e "weightprog_study[dev]"
cd weightprog_study
pytest -q tests
```

The gate regression recomputes the registered round-2 report from the replay dataset and checks that it is identical,
floats included. It is skipped unless both folders are given:

```bash
WEIGHTPROG_GATE_DIR=/path/to/replay-dataset/gate_r2 WEIGHTPROG_REPLAY_DIR=/path/to/replay-dataset/replay_ood \
  pytest -q tests/test_gate_regression.py
```

## Licence

Apache-2.0, see `LICENSE`. The agent configurations target Gemma 4, which Google releases under Apache-2.0.
