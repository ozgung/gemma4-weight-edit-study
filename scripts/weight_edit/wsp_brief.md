# Task brief for a weight-editing proposer (WP11). Identical for every proposer; placeholders in {braces}.

You are the editor in a weight-editing experiment. Your run folder is `{RUN_DIR}`. Your forward-pass budget is
**{BUDGET} passes**; {USED} are used. You are proposer `{PROPOSER}` ({MODEL}, effort {EFFORT}). This is
conversation {CONV} of the run{HANDOFF_NOTE}.

## Goal

Gemma 4 E4B (instruction-tuned, QAT checkpoint, run here in bf16) is the model inside a coding agent. When it
calls the `edit_file` tool it must quote the exact current text to replace as `old_string`. About a third of its
`edit_file` calls fail because `old_string` does not occur in the file ("old_string not found"); a further share
is accepted only because the tool silently repairs JSON-style escaping. Your task: **change the model's weights,
without any training, so that more of its `edit_file` calls are accepted by the tool and produce the intended
edit**, without making the model worse at anything else. You choose what to look at, what hypothesis to test
and what edit to make. You will be judged later, on tasks and repositories you never see, by the end-to-end
resolution rate of the agent with your edit, against the unedited model and against random edits of the same
size.

## What you have

1. **Weights (read-only) and edit saving**, from Python started with
   `outputs/small_model/mlx_venv/bin/python -I` (numpy, mlx available):
   ```python
   import sys; sys.path.insert(0, "scripts/weight_edit"); import wsp_kernel as K
   K.names(regex); K.shape(name); W = K.load(name)            # float32 numpy, [out, in]
   K.save_edit("{RUN_DIR}", name, B, A, note="...")         # W' = W + B @ A, B [out,r], A [r,in], r <= 128
   K.list_edits("{RUN_DIR}"); K.remove_edit("{RUN_DIR}", name)
   ```
   Editable: `q_proj k_proj v_proj o_proj gate_proj up_proj down_proj` of the 42 decoder layers (layers 24-41
   have no k/v projections; they reuse earlier layers' keys and values): 258 matrices. Architecture facts:
   `data/models/gemma-4-e4b-it-qat-q4_0-unquantized/config.json`; hidden 2560, 8 query heads and 2 KV heads of
   dim 256, MLP width 10240, sliding-window attention (512) on most layers with periodic global layers, a
   post-norm (RMSNorm with weight) after attention and after the MLP, per-layer input embeddings, a per-layer
   scalar, tied output embedding with logit softcap 30. Model code: the mlx_lm `gemma4_text.py` in the venv.
   At most **300 Python runs** for the whole run; keep scripts and intermediate arrays in `{RUN_DIR}/scratch/`.
2. **Forward-pass server** (the only way to run the model). Write a request file
   `{RUN_DIR}/requests/fwd_<k>.json` (k = 1, 2, ...) and wait for `fwd_<k>.done.json` (a pass takes about
   1-15 minutes; other runs' requests may be queued first). To wait, run a shell command in the background
   that ends when the file exists, e.g. `until [ -f {RUN_DIR}/requests/fwd_<k>.done.json ]; do sleep 15; done`,
   and continue when it finishes; do not busy-poll in the foreground. One request = one pass, whatever its size within the limits; a failed or
   refused request does not count. Full specification and limits: the docstring of
   `scripts/weight_edit/wsp_server.py`. In short:
   - `gen`: the model, base and/or with your current edit, regenerates `old_string` at up to 64 recorded agent
     edit calls (the exact prompt the agent saw plus the model's own text up to `old_string:<|"|>`); every
     result is scored by the harness's own edit function against the true file state. You get acceptance,
     not-found and exact-copy rates per arm, per recorded outcome, and paired base-vs-edit counts; per-call rows
     with the generated text are in `work/fwd_<k>/rows.jsonl`. Base generations are cached.
   - `read`: teacher-forced, up to 32 texts (a recorded call plus a continuation: the recorded `old_string` or
     any text you give), returning per-token log-probs and top-5 alternatives, and for up to 8 layers the
     residual stream and MLP activations, and for up to 2 layers attention probabilities, at the continuation
     positions; arrays in `work/fwd_<k>/text_<i>.npz`.
3. **Data** (all from the agent's runs on training repositories; nothing else):
   - The callable set: `outputs/weight_edit/shared/search/calls.jsonl` (505 calls from the 83 search tasks;
     see `MANIFEST.txt` there). Each row: key, task, file, the recorded `old_string`/`new_string`, the recorded
     tool outcome (`error_kind`, `strategy`), the full prompt and prefix. File states:
     `outputs/weight_edit/shared/search/files/<file_sha>.txt`. Only these keys are accepted by the server.
   - The common dossier: `outputs/weight_edit/shared/dossier.md` (facts established before your run).

## Rules

- **No training of any kind:** no gradients, autograd, backward passes, optimisers or iterative fitting of
  weights to a loss. Closed-form computations from weights and activations (means, projections, SVD, least
  squares in closed form, thresholds you choose by reasoning) are allowed.
- **Run the model only through the forward-pass server.** Do not load the model for inference anywhere else.
- **Read only:** this brief, the dossier, `outputs/weight_edit/shared/search/`, the model config and code, `scripts/` source,
  and your own run folder. Do not open anything else under `outputs/` (in particular not `outputs/weight_edit/claude/`), `results/`, `data/` (other than the model
  config/tokenizer and the files named above), `docs/`, `research/`, `STATUS.md`, other runs' folders, or other
  proposers' notes. Do not use the internet.
- Every action you take is logged; a breach invalidates the run.
- Before each code run or request, write one or two plain sentences in `{RUN_DIR}/notebook.md`: what you do and
  why. Keep the notebook as your working memory: hypotheses, evidence, decisions, current edit, open questions.

## How the run ends

Continue until all {BUDGET} passes are used, then make your final choice: the tensors saved in
`{RUN_DIR}/edit/` at that moment are your submission. Write `{RUN_DIR}/final_answer.md`: the edit (modules,
rank, formula), the mechanism you believe it addresses, the evidence for and against, what you would test next,
and the per-pass record. If your conversation must end before the budget is used, first bring the notebook up to
date (state, current edit, next step), so a fresh conversation can continue from it.
