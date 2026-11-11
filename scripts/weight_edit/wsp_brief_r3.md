# Task brief for a weight-editing proposer, round 3. Identical for every proposer except the values in this header.

You are the editor in a weight-editing experiment. Your run folder is `{RUN_DIR}`. Your forward-pass budget is
**{BUDGET} passes**. You are proposer `{PROPOSER}` ({MODEL}, effort {EFFORT}). The current directory is your
read-only packet: `brief.md` (this file), `dossier.md`, `server_spec.md`, `search/`, `prior/`, `wsp_kernel.py`, and a
writable `scratch/`.

## Goal

Gemma 4 E4B (instruction-tuned, QAT checkpoint, run here in bf16) is the model inside a coding agent. When it
calls the `edit_file` tool it must quote the exact current text to replace as `old_string`. About a third of its
`edit_file` calls fail with "old_string not found". Most of those failures come after an earlier failure in the same task,
and the model often writes the very string that has just failed: about half of its generations at such a point are
byte-identical to a string that already failed, and those are almost never accepted.

Your task: **after an `old_string not found`, the model repeats the failed string. Change the model's weights, without any
training, so that it writes that repeated string far less often and its retry edits land more often, while first attempts stay
as they are.** A first attempt is an `edit_file` call in a task run with no earlier not-found. You choose what to look at, what
hypothesis to test and what edit to make.

This is round 3. Read `dossier.md` first: it records rounds 1 and 2, the census of retries, and what the earlier editors found and
missed. Two lessons matter most. In round 2 every edit cut its target on unseen repositories and every edit also broke calls that the
base had copied correctly and that it had changed. And guard tuning on the four training repositories did not carry over to other
repositories, which is why your packet now has a second set from other repositories, `dev`. Writing something different is not the goal
in itself: a stable copy error may not be fixed by simply diverging from the failed string.

## How your edit is judged

Your final edit is judged on a set you cannot see: replayed `edit_file` calls from E4B runs on repositories that are in neither
`train` nor `dev`. No editor has seen them. The replay method is the one your server uses: the exact served prompt, the model's own prefix,
two draws per call (greedy, then T 0.6, top_p 0.95, top_k 64), the harness's edit function on the true file state, the same seeds in every
arm. The unit is the task. Besides base, two random edits matched to yours in modules, rank and per-module Frobenius norm are references.

1. **Main test: repeats.** For each task with at least one retry call, take the share of its retry generations that repeat a string in the
   call's failed set. Your edit must have a lower mean over tasks than base and than each random edit, and a one-sided sign test over the
   tasks (tasks lower against tasks higher) must be below 0.05 against all three. The largest of the three p-values is the one that counts,
   with a correction for the number of editors.
2. **Guards, all required, on all calls.**
   (a) Overall acceptance, averaged over tasks, at most 1 point below base.
   (b) In each draw, at most 2 lost calls per 320 calls in the set, rounded down. A lost call is one the agent's tool had accepted, base
   accepts and your edit does not.
   (c) When your edit writes both `new_string` and `old_string` (the full-call replay), compile failures among accepted edits at most 2
   points above base.
   An edit passes only if 1 and 2 both hold.
3. **Then, only for an edit that passes: retry acceptance.** Per-task acceptance on retry calls, higher than base and each random edit, by the
   same kind of test and correction.

Also reported, without a test: first-attempt acceptance and the share of first-attempt generations byte-identical to base. An edit that passes
may then be tested in the full agent.

## What you have

1. **Weights (read-only) and edit saving**, from Python started with
   `outputs/small_model/mlx_venv/bin/python -I` given by its absolute path below (numpy, mlx available):
   ```python
   import sys; sys.path.insert(0, "{PACKET}"); import wsp_kernel as K
   K.names(regex); K.shape(name); W = K.load(name)            # float32 numpy, [out, in]
   K.save_edit("{RUN_DIR}", name, B, A, note="...")         # W' = W + B @ A, B [out,r], A [r,in], r <= 128
   K.list_edits("{RUN_DIR}"); K.remove_edit("{RUN_DIR}", name)
   ```
   Python: `{VENV}/bin/python -I`. Editable: `q_proj k_proj v_proj o_proj gate_proj up_proj down_proj` of the 42
   decoder layers (layers 24-41 have no k/v projections; they reuse earlier layers' keys and values): 258 matrices.
   Architecture: `{MODEL_DIR}/config.json`. Hidden 2560; 8 query heads and 2 KV heads; head dim 256 on the
   sliding-window layers (window 512) and 512 on the global layers 5, 11, 17, 23, 29, 35, 41, where RoPE rotates
   only the first quarter of each head's dimensions; a post-norm (RMSNorm with weight) after attention and after the
   MLP; per-layer input embeddings; a per-layer scalar; tied output embedding with logit softcap 30. Model code:
   `gemma4_text.py` of mlx_lm under `{VENV}/lib/`. At most **600 Python runs** for the whole run; keep scripts and
   intermediate arrays in `{RUN_DIR}/scratch/` or the packet's `scratch/`.
2. **Forward-pass server** (the only way to run the model). Write a request file
   `{RUN_DIR}/requests/fwd_<k>.json` (k = 1, 2, ...) and wait for `fwd_<k>.done.json`. A pass takes about 1-25
   minutes (a 128-call `gen` pass with an edit arm about 15-25, about an hour with the full call); another proposer's requests may be queued
   first, and a request's first use of a call also generates its base outputs.
   To wait, run a shell command in the background that ends when the file exists, e.g.
   `until [ -f {RUN_DIR}/requests/fwd_<k>.done.json ]; do sleep 20; done`, and continue when it finishes; do not
   busy-poll in the foreground. One request = one pass, whatever its size within the limits; a failed or refused
   request does not count. Full specification and limits: `server_spec.md`. In short:
   - `gen`: the model, base and/or with your current edit, regenerates `old_string` at up to **128** recorded agent
     edit calls (the exact prompt the agent saw plus the model's own text up to `old_string:<|"|>`), 1 or 2 draws
     (greedy, then T 0.6); every result is scored by the harness's own edit function against the true file state.
     You get the round-2 tables (acceptance, not-found, exact-copy and escape-artefact rates per arm, per recorded outcome, paired
     base-vs-edit tables) and, per arm and split into `train` and `dev`: retry calls (n, repeat rate, acceptance), first-attempt calls (n, acceptance,
     share byte-identical to base), overall acceptance, served-ok calls lost per draw against base, compile failure among accepted. Per-call rows with
     the generated text are in `work/fwd_<k>/rows.jsonl`. Base generations are cached, and an edit arm always comes with base.
   - **Full call:** your run is set up so that every `gen` request also regenerates `new_string` and `old_string` from the same arm on the same calls and
     draws, with the same figures under `"fullcall"` in the result (rows in `work/fwd_<k>/rows_fullcall.jsonl`). It takes about two to three times as long; add
     `"fullcall": false` to a request to skip it. The judge's compile guard uses these figures, not the plain `gen` compile figure.
   - `read`: teacher-forced, up to 32 texts (a recorded call plus a continuation: the recorded `old_string` or
     any text you give), returning per-token log-probs and top-5 alternatives (native forward), and for up to 8
     layers the residual stream and MLP activations, and for up to 2 layers attention probabilities, at the
     continuation positions; arrays in `work/fwd_<k>/text_<i>.npz`.
   - Add `"expect_edit": {file name: sha256}` to a request to make sure it runs with exactly the edit you mean;
     every `.done.json` records the edit snapshot it used.
3. **Data:**
   - `search/calls.jsonl`: 1,050 calls, see `search/MANIFEST.txt`: `set` = `train` (730 calls, 118 tasks, the four training repositories) or `dev`
     (320 calls, 101 tasks, 69 other repositories, the used round-2 judging set). Each row: key, task, repository, file, the recorded
     `old_string`/`new_string`, the recorded tool outcome (`error_kind`, `strategy`), the full prompt and prefix, and `is_retry`, `failed_set` (the
     `old_string`s of earlier recorded not-found calls in the same run and task). File states: `search/files/<file_sha>.txt`. Only these keys are
     accepted by the server. Use `dev` as you see fit; nothing from it is held back from you, and it is not the judging set.
   - `dossier.md`: facts established before your run.
4. **Earlier editors' work (read-only):** `prior/` holds the final answers and submitted edit tensors of the four round-2 editors, who worked on
   another target (escape artefacts); see `prior/README.md`. You may reuse their findings and edits, combine them, or start fresh. Check claims you rely on:
   their answers reflect what each believed at the time. Their saved edits are not in your `edit/`: to use one, load it from `prior/<run>/edit/` with numpy and
   save it with `K.save_edit`.

## Rules

- **No training of any kind:** no gradients, autograd, backward passes, optimisers or iterative fitting of
  weights to a loss. Closed-form computations from weights and activations (means, projections, SVD, least
  squares in closed form, thresholds you choose by reasoning) are allowed.
- **Run the model only through the forward-pass server.** Do not load the model for inference anywhere else.
- **Read only** the packet, your run folder, the model's config and tokenizer, and the mlx_lm source in the venv.
  **Do not use the internet** (no curl, wget, pip or any network access). Do not try to reach other folders or
  processes. Every action you take is logged and audited; a breach invalidates the run.
- Before each code run or request, write one or two plain sentences in `{RUN_DIR}/notebook.md`: what you do and
  why. Keep the notebook as your working memory: hypotheses, evidence, decisions, current edit, open questions.

## How the run ends

Continue until all {BUDGET} passes are used, then make your final choice: the tensors saved in
`{RUN_DIR}/edit/` at that moment are your submission (an empty `edit/` is a valid choice if you believe no edit
helps). Write `{RUN_DIR}/final_answer.md`: the edit (modules, rank, formula), the mechanism you believe it
addresses, the evidence for and against, what you would test next, and the per-pass record. If your conversation
must end before the budget is used, first bring the notebook up to date (state, current edit, next step), so a
fresh conversation can continue from it.
