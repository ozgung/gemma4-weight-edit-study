# WP7: agent-level test of a round-2 edit (proposal, exploratory)

*2026-11-07, Claude. Status: **approved by the user at 09:37 UTC** (option 1: stage 1 now, stage 2 in tonight's window if stage 1 passes its go rule; L4 only, about 14 units in all). This file is frozen by the commit that adds it, before the first run; any change after that is logged as an amendment.*

## Why this is outside the gate rule

The round-1 gate (weight-edit log, 2026-11-02 21:34 UTC, point 5) says: if nothing passes, the null result stands, with no automatic extra GPU evaluation, and any further run needs a stated question and a compute justification. No round-2 edit passed: Opus (served-ok guard), Fable (served-ok and acceptance), Informed-Opus (acceptance, served-ok and plain compile). Any result here is therefore **exploratory**, reported in the paper under that label, and does not change a verdict.

## Question

Served as a LoRA adapter on the competition's W4A16 model in vLLM, does a frozen round-2 edit lower the escape-artefact rate of live `edit_file` calls in full agent runs on held-out repositories, compared with the base and a size-matched random adapter? Resolve counts are reported, not tested.

## Compute justification

- It removes a listed limitation: draft-v3 §8 says "No edit was loaded in vLLM".
- It tests whether the full-call replay, the closest proxy we have, predicts live behaviour.
- It is the step the round-2 pre-registration names after a pass ("only a gate-passing edit goes to WP7"). Running it on a non-passing edit answers a narrower question, and only under the exploratory label.

## Which edit

**`fable_r2`** (Fable-r2), by this rule: the best full-call accepted-and-compiling count among the three frozen round-2 edits. The full call regenerates both `old_string` and `new_string`, as the live agent does.

| Edit | Full-call accepted and compiling (base 407) | `old_string` artefacts (base 13.9 %) | `new_string` artefacts (base 10.5 %) | Form |
|---|---|---|---|---|
| `fable_r2` | 408 (paired 398 / 9 / 10) | 5.3 % | 2.7 % | layer 41 `down_proj`, rank 2 |
| `opus_r2` | 404 (395 / 12 / 9) | 10.2 % | 7.3 % | layer 40 `down_proj` rank 1, plus layer 41 gate/up/down rank 40 |
| `opusinf_r2` | 402 (387 / 20 / 15) | 3.6 % | 2.3 % | layer 41 gate/up/down, rank 80 |

- **Practical reason:** `fable_r2` is one module with one rank. `export_peft` writes per-module ranks through `rank_pattern`, which vLLM's LoRA loader may not read. `opus_r2` mixes ranks 1 and 40, so it would need zero-padding to one rank first. `opusinf_r2` has one rank (80) on three modules, so it is servable as it is; it ranks last on the rule.
- **Disclosed:** the choice was made after seeing gate outcomes on the same 101 tasks the agent test reuses (OOD-43 plus OOD-B). The selection is therefore optimistic, and the paper says so.
- **Control:** `fable_r2_rand1`, the gate's first matched random edit (same module, rank and per-module Frobenius norm). Its full-call outcome was not used to choose anything.

## Stage 0: Mac, CPU, under 30 minutes, no cost

1. Export `fable_r2` and `fable_r2_rand1` with `weightprog.edits.export_peft` (float32, `lora_alpha = r`, so the delta is exactly B·A). Tensor names follow the working E4B adapter, `base_model.model.model.language_model.layers.41.mlp.down_proj.lora_{A,B}.weight`.
2. Check the delta with `scripts/verify_lora_adapter.py`, and the bf16 rounding of B·A against the npz.
3. Make two arms, each one factor from e001nt (E4B-NoThink-T0.6): e001nt plus `adapter:`.
   - `e001nte` **E4B-NoThink-T0.6-FableEdit**
   - `e001ntx` **E4B-NoThink-T0.6-RandEdit**
4. `uv run python scripts/submission.py validate` on both.
5. Write `scripts/weight_edit/replay_vllm.py`. It mirrors `replay_gen.py`:
   - input: `prompt + prefix` token ids;
   - draw 0 greedy, draw 1 at T 0.6, top_p 0.95, top_k 64, seeded;
   - stops at the closing `<|"|>`; `max_new = min(1024, 2·ref + 64)`;
   - uses the vLLM offline `LLM` with a `LoRARequest` per arm;
   - writes the same jsonl, so `replay_score.py` scores it unchanged.

## Stage 1: Colab L4, about 1 h (about 2 units). W4A16 replay

- **Setup:** one vLLM with the W4A16 base, `--enable-lora --max-lora-rank 8`. Three arms (base, `fable_r2`, `fable_r2_rand1`) on the 320 gate calls (`replay_ood/usable_keys.txt`) × 2 draws, frozen prefix.
- **Scoring:** the gate's artefact and acceptance functions, scored on the Mac afterwards.
- **Go rule for stage 2, fixed now:**
  - the edit's artefact count is at most 0.75 × the W4A16 base count (the bf16 effect was 0.47 ×);
  - the random adapter is within ±10 % of the base.
  - Otherwise stop and report that the edit's effect does not survive serving. That outcome is a result for the paper too.
- **Reported:** acceptance and served-ok losses per draw against the bf16 figures (W4A16 base vs bf16 base gives the serving gap itself).

## Stage 2: Colab L4, about 5-6 h (about 11-12 units), night window

- **Arms:** base e001nt, e001nte and e001ntx side by side on one LoRA-enabled vLLM, 4 streams each, 12-min cap. This is the e4v/e4t pattern (`scripts/colab/run_eval.py`, LoRA mounting as in `chain_20261029_e4b_d.sh`).
- **Order:** OOD-43 (44), then OOD-B-71 (`configs/tasks_oodb71_colab.txt`).
- **Limits:** 1 run per arm; hard stop at 7 h. A partial set is paired on the common tasks.
- **Primary, exploratory:** the per-task live escape-artefact rate.
  - What it counts: `edit_file` calls whose `old_string` or `new_string` carries a literal-escape artefact, by the gate's artefact function.
  - How it is tested: task macro, edit against base and against random, one-sided sign test, with tasks without an `edit_file` call omitted.
  - Expected information: in the base replays 30 of 101 tasks had any artefact, and live trajectories diverge, so expect about 30 informative tasks per comparison.
- **Descriptive:**
  - the `old_string` not-found rate;
  - acceptance through the unescape fallback (`applied_unescaped`);
  - `edit_file` calls per task;
  - resolved (Docker regrade), no-patch and timeouts;
  - paired discordant resolves.
  - Resolve noise: e001nt resolves about 27 of 115 per run, and same-arm repeats move 3-7 tasks per 44. A resolve difference under about 7 is a tie.
- **Thinking:** not used. The edits were built on e001nt calls, which have no thinking. The E4B base is e001tt (thinking), so a transfer to e001tt is a separate question.

## Timeline and cost

- **Stage 0:** done 2026-11-07 (local).
- **Stage 1:** 2026-11-07 from about 09:45 UTC, by the user's choice, on its own L4 session (`scripts/colab/chain_20261107_wp7s1.sh`), stopped when done. The same chain also records two serving facts for stage 2: the LoRA ranks vLLM accepts and how the harness sets `max_lora_rank`.
- **Stage 2:** only if stage 1 passes its go rule; a new L4 session in the 2026-11-07 17:00 to 11-08 09:00 UTC window. Results 11-08 morning; §7 and §8 updated by 11-10 within the word limit.
- **Cost:** L4 only, about 2 units per hour, so about 14 units for both stages. No A100.
- **Clash check:** the E4B track runs on Kaggle; STATUS shows no Colab session. Check the E4B session's plans before starting.

## Decision for the user

1. Stages 0-2 as above, with stage 2 conditional on stage 1. **Recommended:** it answers the question at the lowest cost that can answer it.
2. Stage 0 and 1 only (about 2 units). This removes the "not loaded in vLLM" limitation but makes no agent-level claim.
3. None. The paper reports the gate results and the full-call replay only.

## Amendments (after the freeze at 1c8d19f)

1. **2026-11-07 10:15 UTC, stage 1 engine settings.** The first run failed before any generation: vLLM ran out of memory on the L4 while warming up the sampler with its default 256 sequences over the 262k-token vocabulary. `replay_vllm.py` now passes `max_num_seqs=32` and `gpu_memory_utilization=0.85` (KV cache room for about 9 of the longest requests at once). Inputs, arms, sampling and scoring are unchanged.
2. **Serving fact for stage 2, recorded by the same run:** vLLM 0.19.1 accepts `max_lora_rank` only from {1, 8, 16, 32, 64, 128, 256, 320, 512}, and the edit and its control are rank 2. The harness rounds the largest detected rank up to the next accepted value (`_round_up_vllm_lora_rank` in `adk_submission/server.py`, 0.2.13), so it serves both at `max_lora_rank` 8 and no padding is needed.
