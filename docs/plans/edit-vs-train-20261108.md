# Edit vs training: gradient LoRA baselines against Fable's round-2 edit (pre-registration)

Written 2026-11-08 16:3x UTC by Claude (session deneyler 05c039), at the user's request ("ön kaydı da yaz"), before any
training data is built and before any trained adapter is generated on the gate set. Status: **draft until the freeze line
at the end is filled in**; the freeze happens before the first training step.

## Question

Does direct weight editing by a frontier model (Fable, round 2, frozen edit `fable_r2`) have any measurable advantage
over ordinary gradient training on the same target? The likely answer is "not in compute" (see the ledger below). The
candidate advantages tested here are **data efficiency** (Fable fitted its edit on teacher-forced reads of 57 calls)
and **where vs what**: given Fable's location and size, does gradient descent find better or worse values than Fable's
closed-form reasoning? "No advantage" is a valid outcome and will be reported as such.

## Arms

All arms are LoRA adapters (W + scale · B A) trained with `scripts/mlx_lora/train.py` on the same bf16 weights the
replay gate uses (`data/models/gemma-4-e4b-it-qat-q4_0-unquantized`), seed 0, then converted to the gate's edit format.

| Arm | Modules | Rank / alpha | Data | Epochs | lr (cosine to 0.1x, warmup 5 %, min 3 steps) | Accum |
|---|---|---|---|---|---|---|
| **G-tam-411** (practitioner baseline) | all 258 language-model linears | 8 / 16 | 411 labelled calls (4.56 M tokens) | 2 | 1e-4 | 8 |
| **G-tam-39** (data-matched) | all 258 linears | 8 / 16 | the 39 labelled calls among Fable's reads (444 k tokens) | 3 | chosen from {1e-4, 3e-4, 1e-3} | 4 |
| **G-yer-39** (location-, size- and data-matched) | `layers.41.mlp.down_proj` only (25,600 parameters, as Fable's) | 2 / 2 | same 39 calls | 3 | chosen from {1e-3, 3e-3, 1e-2} | 4 |

- **G-yer norm cap:** after every optimizer step, if ||scale · B A||_F > 5.06 (Fable's edit norm), B is rescaled so that it
  equals 5.06. Same module and rank as Fable's edit, never larger.
- **No tuning on the gate set.** For the two 39-call arms the lr is chosen from the grid above by the lowest validation
  NLL at the last step (the 32 validation calls below; training repositories only); the other runs of the grid are
  reported. G-tam-411 is too long for a grid (about 7 h per run) and uses lr 1e-4, the trainer's standard; if its
  training NLL on loss tokens (mean of the last 3 steps against step 1) falls by less than 25 %, it is rerun once at 3e-4
  and the rerun is the arm of record; both are reported.
- **Validation (lr choice for the 39-call arms):** NLL on loss tokens of 32 labelled calls drawn (seed 20261108) from
  training-repository tasks outside Fable's 32 tasks. G-tam-411 trains on all 411 labelled calls and has no validation.
- `--max-len 20000` (17 of the 411 calls exceed 16,384 tokens, the longest has 19,916; truncation would cut the label).

## Data and labels

- **Calls:** the 730 round-2 search calls the editors saw (`outputs/weight_edit/sandbox/fable_r2/search/calls.jsonl`,
  four training repositories, 107 tasks among the labelled ones). The gate set is never read for training.
- **Fable's calls:** the unique keys of Fable's teacher-forced read passes 1, 2 and 5 (88 texts over 57 calls), the
  passes its final answer names as the fit data for its gates. 39 of the 57 have a label (32 tasks, 385 k tokens).
- **Label (target `old_string`):** the first of (recorded `old_string`, its JSON-unescaped form) that occurs exactly once in
  the true file state. Calls with no such string are dropped (319 of 730: mostly not-found calls whose text is not
  recoverable by unescaping). Of the 411 labelled calls, 341 keep the recorded string and 70 take the unescaped one.
- **Input:** tokens of `prompt + prefix` (the replay gate's exact input, ending at `old_string:<|"|>`), then the label's
  tokens and the closing `<|"|>` token. Loss on the label tokens and the closing token only.

## Evaluation (round-2 gate set, round-2 rule)

- **Set:** `replay_ood/usable_keys.txt`, 320 calls in 101 tasks from 69 repositories outside training, draws {0 greedy, 1 at
  T 0.6}, the same per-(call, draw) seeds as every round-2 arm. The cached base generations (`gate_r2/gen_base*`) and
  Fable's (`gate_r2/gen_fable_r2*`, `gate_r2/fullcall/gen_fable_r2*`) are reused; nothing of theirs is regenerated.
- **Per arm:** `replay_gen.py --edit` (plain), `replay_score.py`, then the full call (`replay_fullcall.py`,
  `replay_score_fullcall.py`) for the compile guard of record.
- **Rule for each G arm (pass/fail, Holm over m = 3):** the round-2 rule as amended on 2026-11-05 (artefact primary,
  acceptance non-inferiority, served-ok preservation, full-call compile guard), with the reference set **{base}** only.
  No matched random controls: the question is edit against training, not training against noise. Implemented by a new
  `scripts/weight_edit/gate_g_analyze.py` that reuses the round-2 functions; its sha256 goes in the log before any G arm
  is generated on the gate set.
- **Head-to-head against Fable (the comparison of interest, no pass/fail):** per-task artefact rate and per-task
  acceptance, G arm against `fable_r2`, two-sided exact sign test; "better" / "worse" at p < 0.05, otherwise "tie".
  Also the three guard metrics side by side, and task- and repository-clustered bootstrap CIs (10,000, seed 20261108).

## Readings, fixed now

| Outcome | Reading |
|---|---|
| G-tam-39 **and** G-yer-39 both worse than Fable on artefacts, G-tam-411 tie or better | editing has a **data-efficiency** advantage on this target |
| G-tam-39 or G-yer-39 tie or better than Fable | **no data-efficiency advantage**: gradient training on Fable's own calls does as well |
| G-yer-39 better / tie / worse than Fable | given Fable's location, size and calls, gradient values are better / as good / worse than Fable's reasoned values |
| A G arm passes the full rule (primary and all guards) | training reaches what no editor reached in round 2 (Fable failed the served-ok and acceptance guards) |
| All G arms fail the guards too | collateral loss is shared by both methods at this target |

Agent-level confirmation on OOD-C (WP7R protocol, Colab L4) is not part of this registration; it would be a separate
registration, run only with the user's go and only for an arm whose result makes it informative.

## Compute ledger (reported in the paper next to the result)

- **Gemma side, same Mac and framework:** server GPU seconds of `fable_r2`'s 25 passes (10.6 h; 19 gen, 6 read) and the
  forward tokens they processed (exact count from the request logs and `prompt_tokens_ours`), against each G arm's
  measured training seconds and trained tokens (forward-and-backward counted as about 3 forward tokens), grid runs
  included. Smoke test 2026-11-08 17:03-17:07 UTC (G-yer, 2 steps): about 420 tok/s, peak 24.3 GB. Expected: G-tam-411
  2 epochs 9.1 M tokens, about 6-7 h; each 39-call run 1.33 M tokens, about 1 h.
- **Theoretical floor for G-yer:** because layer 41 is the last block and down_proj's input does not depend on the edit,
  the same arm could be trained on cached activations after one forward pass; reported as a calculation, not run.
- **Frontier side:** wall time and API token counts of the editor (from the transcripts, checked before citing); not
  converted to FLOPs, since the model sizes are not public.

## Order and files

1. Build `data/edit_vs_train/{all411,fable39}/{train,val}.jsonl` with a new `scripts/weight_edit/build_g_data.py`
   (sha in the log); extend `train.py` with `--layers`, `--targets` and `--max-fro` (default behaviour unchanged);
   converter `scripts/weight_edit/lora_to_edit.py` (MLX adapter to gate `edit/*.npz`, scale folded into B).
2. Train in this order: the G-yer-39 grid, the G-tam-39 grid, G-tam-411 (resumable checkpoints; the Mac sleeps during
   daytime moves). Runner `outputs/weight_edit/claude/edit_vs_train/run_train.sh`.
3. Gate each arm as soon as it is trained (one model process at a time on the Mac), then `gate_g_analyze.py`.
4. Results go in the weight-edit log; outputs in `outputs/weight_edit/claude/edit_vs_train/`.

Freeze: **2026-11-08 17:08 UTC**, before the first training step of any arm and before any G arm touches the gate set.
Done before the freeze, all on training-repository data only: the data build (`MANIFEST.json` 8b4056afe239) and a 2-step
smoke run of the G-yer configuration at lr 1e-3 (17:03-17:07 UTC; output `edit_vs_train/smoke/`, not an arm). The lr
grid and `--max-len 20000` were added after the smoke run and before this freeze. Code at freeze (sha256 prefixes):
`build_g_data.py` d671f79a8c6e, `lora_to_edit.py` 4cd3e0a26954, `train.py` bba3d6edb4db (adds `--targets`, `--layers`,
`--max-fro`; default behaviour unchanged), `common.py` 46bf6de78aee (`apply_lora(layers=...)`), `run_train.sh` cf6c88c2ebaf.
`gate_g_analyze.py` is written and hashed in the weight-edit log before the first gate generation of a G arm.
