# Brief: edit Gemma 4 E4B's weights so it keeps the pydantic 1 API in pydantic 1 projects

You are working alone, in a sandbox, on a Mac (Apple M4 Max, 48 GB unified memory, Metal GPU). Your working
directory is this packet (read-only except `scratch/`). Your run folder is `{RUN}` (read/write). Nobody will answer
questions during the run; decide yourself and write down why.

## The task

Gemma 4 E4B (instruction-tuned, 42 decoder layers) is often used as a coding assistant. Pydantic 2 renamed much of
the pydantic 1 API (`Model.schema()` became `Model.model_json_schema()`, `Model.__fields__` became
`Model.model_fields`, `Model.parse_raw(s)` became `Model.model_validate_json(s)`, and so on). Many projects are still
pinned to pydantic 1, where the new names do not exist: on a pydantic 1 model, `Model.model_json_schema()` raises
`AttributeError`. In projects pinned to pydantic 1, E4B often writes the pydantic 2 API anyway.

**Change the model's weights so that, when the prompt shows a pydantic 1 project, the model writes the pydantic 1
API instead of the pydantic 2 API**, while:

- in a project that uses pydantic 2, it keeps writing the pydantic 2 API;
- in code that does not use pydantic (dataclasses, plain functions), nothing changes;
- the model does not get worse at anything else.

**How you may change the weights.** Only by adding low-rank terms `W' = W + B @ A` to decoder-layer linear weights
(`q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` of layers 0-41; layers 24-41 have no `k_proj` and
`v_proj`), rank at most 128 per module, computed **in closed form** from the weights and from activations you
capture: linear algebra such as least squares, ridge regression, projections, eigen- or singular-value
decompositions. **No gradient-based training**: no optimiser, no backpropagation or autograd, no loss minimised by
iteration. Choosing a few scalars (a strength, a regularisation constant) by trying a handful of values and
measuring is allowed.

## What you have

- The model: `{MODEL_DIR}` (`config.json`, `model.safetensors` in bfloat16, `tokenizer.json`, `chat_template.jinja`).
  Canonical weight names: `model.language_model.layers.{{i}}.self_attn.{{q,k,v,o}}_proj.weight`,
  `model.language_model.layers.{{i}}.mlp.{{gate,up,down}}_proj.weight` (shapes `[out, in]`).
- `cases_tune.jsonl`: {N_TUNE} cases (format below). A held-out set with the same format, other domains and other
  request wordings, is kept by the operator and is used only for the final judgement.
- A Python environment (on `PATH`) with numpy, mlx {MLX}, mlx-lm {MLX_LM}, tokenizers, jinja2, safetensors and
  transformers (no PyTorch).{LIBRARY}
- Time: **{HOURS} hours of wall-clock time** from the start of this conversation. At the limit the operator stops
  the run; whatever is in `{RUN}/edit/` then is your submission. Loading the model takes tens of seconds and
  generation about 50 ms per token; keep one long-lived Python process if you can, and only one process holding the
  model at a time (memory).

## The cases

One JSON object per line. Each is a chat prompt (`messages`, rendered with the model's chat template and the usual
generation prompt `<|turn>model\n`), then `prefill` (text appended after the generation prompt, as if the model had
already written it), then up to `max_new` generated tokens. `role` is `target` (pydantic 1 project: the behaviour to
change) or `keep` (pydantic 2 project, non-pydantic code, unrelated prompts: must not change). `good` / `bad` are
regular expressions on the generated text: a generation is `bad` if any `bad` pattern matches, else `good` if any
`good` pattern matches, else `other`. A `keep` case without `good`/`bad` must produce the same greedy text as the
unedited model. `group` is the domain (the held-out set uses other domains).

## How the submission is judged (fixed before the run)

On the held-out set, with draw 0 greedy and draw 1 sampled (temperature 0.6, top-p 0.95, top-k 64), the operator runs
the unedited model, your edit, and two random edits with the same modules, ranks and per-module norm (Gaussian `B`,
`A` rescaled to the same Frobenius norm of `B @ A`). Your edit passes if all of these hold:

1. **Target:** on `target` cases, the per-case bad rate (over the two draws) is lower under your edit than under the
   unedited model and than under each random edit: one-sided exact sign test over cases, p < 0.05 for each of the
   three comparisons. Also, against the unedited model: the 95 % bootstrap interval of the mean improvement,
   resampling domains (`group`), lies above 0; and the good rate rises by at least half of the drop in the bad rate
   (turning bad outputs into something that is neither is not a fix).
2. **Keep:** at most {KEEP_MAX} `keep` cases per draw go from `good` (unedited) to not `good` (edit), and at most 10 %
   of the rule-less `keep` cases change their greedy text.
3. **Side effects:** on a fixed set of short English prose and Python code texts (none about pydantic), the mean
   KL divergence per token from the unedited model's next-token distribution to the edited one's is at most 0.05 nats,
   and the mean increase of the per-token negative log-likelihood of the text is at most 0.02 nats.

The operator also reports the target's size (bad and good rates) and every keep case your edit breaks.

## What to deliver (in `{RUN}`)

- `edit/`: one `.npz` file per edited module, named freely, each holding `module` (the canonical weight name, a
  string), `B` (float32, `[out, r]`) and `A` (float32, `[r, in]`). An empty `edit/` is allowed and means "no edit".
- `notebook.md`: before every substantial step, what you are about to do and why; after it, what came out. Keep it
  current: the run may be stopped at any time.
- `final_answer.md`: the edit (modules, ranks, how it was built), what it should do and why, your own measurements
  on the tune cases, the evidence against it, and what you would try next.

Work only inside this packet, `scratch/` and your run folder. Do not try to reach the held-out set, other files on
this machine, or the network beyond your own API connection.
