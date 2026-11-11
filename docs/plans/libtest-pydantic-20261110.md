# weightprog usability test: an agent edits Gemma 4 E4B toward the pydantic 2 API, with and without the library (pre-registration)

*2026-11-10, Claude Opus 5.5 (fork of session deneyler 05c039, Herdr terminal `opus-weightprog`). Status: **frozen at 18:52 UTC 11-10** by the weight-edit log entry of that time (`docs/experiments/weight-edit-collab-20261102.md`); target flipped after the census (18:46 UTC).
It is frozen by a weight-edit log entry that records this file's sha256 and the sha256 of every input listed below.
That entry is written after the census and before either arm starts. Each arm starts only on the user's go, since
the arms spend Claude tokens.*

*User decisions (11-10, about 12:50 UTC):*
- *target pydantic v1→v2 (flipped at 18:46 UTC, see below);*
- *two arms, with and without the library, both Opus 5.5;*
- *one paper, with weightprog in a resources section.*

## Question

Does weightprog help a coding agent produce a working closed-form weight edit for a **new** target on Gemma 4 E4B?
The comparison is the same agent with only standard tools (numpy, mlx, mlx-lm) and the same time. "Working" means
passing a fixed held-out judgement: the target moves beyond the base model and two matched random edits, the
keep cases survive, and the side effects stay small.

## Target

**Flipped after the census (user, 18:46 UTC 11-10).** The user wrote: "daha iyi bir hedef bulunabilir mi, yoksa
hedefi çevirerek devam et" ("can a better target be found, or else continue with the target flipped"). I kept
the flip. Another candidate (for example pandas 2's removed `DataFrame.append`) would need a new case builder and
census, about an hour. Choosing among several censuses would also be target shopping. The flipped target is real,
and the census has already measured it.

**Registered target (flipped):** in a project pinned to pydantic 1, write the pydantic 1 API instead of the pydantic
2 API, which does not exist there. The APIs are `.schema()` (not `.model_json_schema()`), `.__fields__` (not
`.model_fields`) and `.parse_raw()` (not `.model_validate_json()`).

Keep:
- the v2 API in projects that use pydantic 2 (the hard negative; base good rate 1.00);
- dataclass code without pydantic methods;
- unrelated coding prompts unchanged.

The census section below gives the rule and the numbers. The original wording follows, unchanged.

*Original target (before the census):* in a project that uses pydantic 2, write the pydantic 2 API instead of the deprecated pydantic 1 API:
- `.model_dump()` for `.dict()`;
- `.model_dump_json()` for `.json()`;
- `.model_validate()` for `.parse_obj()`;
- `.model_validate_json()` for `.parse_raw()`;
- `.model_copy()` for `.copy()`;
- `.model_fields` for `.__fields__`;
- `.model_json_schema()` for `.schema()`.

The final list is the APIs the census shows E4B getting wrong (filled in below).

Keep:
- the v1 API in projects pinned to pydantic 1 (the natural hard negative);
- dataclass code without pydantic methods;
- unrelated coding prompts unchanged.

Not the escape artefact: this target was never worked on in the study.

## Cases (`scripts/weight_edit/libtest/make_pydantic_cases.py`)

- **Format.** Chat prompts in the style of an agent request: a `models.py` with one model, then a one-function
  request. They are rendered with the checkpoint's template plus a prefill, and every case stops at the closing code
  fence.
- **Two styles.**
  - `tight`: the prefill ends right before the API token, `max_new` 16.
  - `free`: the model writes the whole function, `max_new` 240.
- **How v2 is signalled**, rotated:
  - the version stated in the request;
  - v2-only imports and `model_config = ConfigDict(...)` in the file;
  - "we are on pydantic v2".
- **v1 signals:**
  - a stated pin to 1.10;
  - a `validator` import and `class Config:` in the file;
  - "still on v1".
- **Tune split:** 8 domains (user, order, event, settings, post, address, ticket, recipe) with request wording A.
- **Holdout split:** the other 8 domains (invoice, product, reading, payment, employee, flight, repo, book) with
  wording B, which uses other function names and phrasing.
- Non-pydantic keep cases alternate between the splits by group. No group is shared between splits.
- **Census:** all 16 domains, wording A, base model only. It is used to choose the API list and to report base
  rates. Its prompts in the tune domains are the tune cases. The holdout wording is never run before the judge.

## Arms

| | `lib` | `nolib` |
|---|---|---|
| Agent | Claude Code 2.1.322, `claude-opus-5-5`, effort max | same |
| Brief | `brief_template.md` filled by `build_packets.sh` | identical except one paragraph naming the library |
| Python env | `envs/lib`: 38 packages (mlx 0.32.4, mlx-lm 0.32.2, numpy 2.5.4, tokenizers, jinja2, safetensors, transformers; no torch) **plus weightprog 0.2.0** from the packet snapshot | `envs/nolib`: the same 38 packages, no weightprog |
| Packet | brief, `cases_tune.jsonl`, `weightprog/` (snapshot of the repository: code, AGENTS.md, skills, recipes, tests) | brief, `cases_tune.jsonl` |
| Time | 4 h wall clock (waits after a usage limit not counted) | same |
| Sandbox | Seatbelt profile of the study (`proposer_claude.sb`): home unreadable except packet, run folder, model, env; writes only to run folder, scratch, private temp | same |

- **Order:** lib, then nolib, one at a time (one model process on the Mac).
- **Runner:** `run_libtest.sh`. It stops an arm at its deadline by killing the process tree and sweeping orphans,
  then freezes `edit/` and runs the judge after both arms.
- **Checks already done:**
  - Inside the sandbox, MLX runs on the GPU, the model is readable, and this repository, the weightprog repository
    and the case files outside the packet are not readable.
  - The packet is read-only and the run folder writable (test at 13:31 UTC).
- **A possible added arm (on hold; user, 11-10 16:40 UTC).** Whether to run one is decided after the two arms'
  verdict. The first candidate is Fable 5.2 in the lib packet: same harness, only the model changes. Gemini 4 Carbon
  through the Antigravity CLI is the alternative:
  - prepared: `launch_arm_agy.sh`, `proposer_agy.sb` and an isolated, signed-in agy home;
  - sandbox checked at 16:31 UTC; no model called.

  If an added arm runs, it is registered in its own log entry before launch. It reuses this brief, the lib packet's
  library snapshot, the same judgement and seed base (controls seeded as arm index 3), and it is judged on its own.
  It is reported as a sequential follow-up chosen after the first result, not as part of this comparison.

## Judgement (fixed now)

`judge.py` with weightprog's `evaluate.compare` on `cases_holdout.jsonl`, `criteria.json` and `judge_texts.json`:
- **Draws:** 0 greedy, 1 sampled (T 0.6, top-p 0.95, top-k 64). Seeds come from the case id, so all arms are paired.
- **Controls:** each arm's edit gets two matched random edits (same modules, ranks and per-module norm), seed
  20261110 + 10 i + k.
- **Pass** requires all three:
  1. **Target:** the bad rate is lower than base and than each control. This is a one-sided exact sign test over
     target cases, p < 0.05 for each of the three. In addition, against base, the domain-clustered bootstrap 95 % CI
     of the mean improvement must lie above 0, and the good rate must rise by at least half the drop in the bad
     rate (`min_good_share` 0.5).
  2. **Keep:** at most 2 keep cases broken per draw, and at most 10 % of rule-less keep cases changed.
  3. **Side effects:** on 16 judge-only texts (new prose and code, not in the library, no pydantic), mean
     KL(base‖edit) ≤ 0.05 nats per token and mean NLL increase ≤ 0.02 nats per token.
- **Other outcomes:** an empty `edit/` is "no edit", and an edit that fails validation is "invalid".
- **Audit flag:** the transcripts are scanned for gradient-training calls (`value_and_grad`, `mx.grad`, optimisers,
  `.backward(`). A hit makes the arm "rule broken" whatever the numbers say, after I read the context and confirm it
  was used to compute the edit.

## Readings, fixed now

| lib | nolib | Reading |
|---|---|---|
| PASS | not PASS | The library made the difference in this pair: a working edit on a new target in 4 h with the library, none without. One run each: an existence result for the library, not an effect size. |
| PASS | PASS | Both agents can do this target in 4 h. The library's value then shows only in the descriptive secondaries (time to the first evaluated edit, size of the target effect, keep damage). |
| not PASS | PASS | No evidence the library helps on this target; the agent did better without it. Reported with what the lib arm spent its time on. |
| not PASS | not PASS | Neither succeeded in 4 h. Descriptive secondaries only; a failed target joins the escape-artefact study as a limit of the method or of the time, and we say which if the evidence allows. |

## Secondary (descriptive, no pass/fail)

- **Size:** bad and good rates per arm and API; keep cases broken; KL and NLL.
- **Speed:** time from start to the arm's first own evaluation of an edit on generations (from transcript
  timestamps), and the number of distinct edits it evaluated.
- **Self-assessment:** whether the arm's own tune measurement predicted the holdout result.
- **Tune-set judgement:** the same rule on `cases_tune.jsonl`, to separate overfitting from failure.

## Disclosed

- I wrote the library documents (AGENTS.md, the workflow skill, the recipes) on 11-10, after the target was chosen.
  To avoid writing the solution into them, every worked example uses an unrelated target: US vs British spelling,
  answering with a list, a word preference. The docs say nothing about pydantic. The `context-token-choice` recipe
  has the same shape as this task (a context-dependent token choice). That is what the library is for, and it is
  the main way it could help.
- I wrote the cases and the judge, and I chose the API list after the census.
- Before the snapshot, a Claude Fable 5.2 subagent reviewed weightprog 0.2.0 for fitness to purpose (11-10, about
  14:15 UTC). It was not told the target. I fixed its blocking and important findings, each a general change with no
  target-specific content:
  - memory use of `collect_tail` on the real model;
  - an INCOMPLETE verdict;
  - the good-rate and group-CI conditions;
  - separate fit and eval reference texts;
  - base-row reuse;
  - fit/check splitting and `cv_gate`;
  - harvesting decision rows from generations;
  - `EditSet.merge`;
  - the covariance update `kv_update`;
  - residual-direction writes;
  - a layer sweep;
  - the bf16 realised fraction.

  The same rule changes apply to the judge.
- After a methods-table review against the literature (11-10, about 15:00 UTC, `docs/plans/methods-table-20261110.md`),
  three general tools were added before the snapshot: activation patching (`probe.patch_effect` / `patch_sweep`), unit
  patching (`construct.unit_patch`), and null-space restriction (AlphaEdit: `null_space_basis`, `project_edit`,
  `kv_update(null_basis=)`).
- The spelling worked example was run on real E4B for the first time at about 18:00 UTC 11-10
  (`examples/spelling/run_e4b.py`, out `outputs/weight_edit/libtest/spelling_example/`). The recipe code ran end to
  end: 17 s for the recipe, gated and kv edits built, each judged against two controls with side effects measured
  (KL about 0.0001). But E4B never writes the word in those prompts: base, edit and controls were all `other`. A
  variant whose prompts contain the word failed the same way (48 cases, scratch census at 18:06 UTC). Before the
  snapshot I added a note saying so to `examples/spelling/README.md` and one clause to `README.md`. There are no
  other library changes after commit 810116f.
- The library arm can see weightprog's built-in reference texts. The judge uses different texts.
- The time budget (4 h) and the effort (max) are the same for both arms. Opus usage limits may pause an arm, and
  pauses are not counted.

## Limits written now

- One run per arm. The agent's own variance is unknown, so a single difference can be chance.
- A single target, chosen by us.
- The same author built the library and ran the test.

## Inputs (sha256 recorded in the freezing log entry)

- `make_pydantic_cases.py`, `sets/cases_tune.jsonl`, `sets/cases_holdout.jsonl`, `criteria.json`, `judge_texts.json`.
- `brief_template.md`, `build_packets.sh`, `run_libtest.sh`, `launch_arm.sh`, `judge.py`.
- `packets/MANIFEST_lib.sha256`, `packets/MANIFEST_nolib.sha256`.
- The weightprog snapshot as installed in `envs/lib`.

## Census (filled in after the run, before freezing)

**API selection rule.** Written at 17:50 UTC 11-10, while the census was running and before any of its output had
been read:
- For each API, take the base model's bad rate on the v2 target cases, pooled over both styles and all 16 domains
  (wording A, both draws, greedy and sampled, as in the judgement).
- Keep an API if that rate is at least 0.25.
- If fewer than 3 APIs pass, keep the 3 with the highest rates, provided each is above 0.10.
- If fewer than 2 APIs have a rate above 0.10, the target is too weak for E4B: stop and report to the user before
  freezing.
- The v1 keep cases are built for the kept APIs. Their base rates are reported, not used for selection.

**Result** (17:49-17:59 UTC 11-10): 236 cases, two draws, load 5 s, run 613 s, 22,799 tokens, 27 ms per token.
Out: `outputs/weight_edit/libtest/census/` (`summary.json`, `rows.jsonl`). Bad rate = share of the 16 generations
per cell (8 cases × 2 draws):

| API | v2 project, tight | v2 project, free | v1 project, tight | v1 project, free |
|---|---|---|---|---|
| dump | 0 (good 1.00) | 0 (good 1.00) | 0 (good 1.00) | 0 (good 1.00) |
| dump_json | 0 (1.00) | 0 (1.00) | **0.19** (good 0.81) | 0.06 (0.94) |
| validate | 0 (1.00) | 0 (other 1.00) | 0.06 (0.94) | 0 (other 1.00) |
| validate_json | 0 (1.00) | 0 (other 1.00) | **0.31** (0.69) | 0 (other 1.00) |
| copy | 0 (1.00) | 0 (1.00) | 0 (1.00) | 0 (good 0.75, other 0.25) |
| fields | 0 (1.00) | 0 (1.00) | **0.94** (other 0.06) | 0.12 (0.88) |
| schema | 0 (1.00) | 0 (1.00) | **0.62** (good 0.38) | **1.00** |

Non-pydantic keep cases: dataclass good 1.00; unrelated prompts unchanged 1.00 (as the reference).

**What it shows.**
- The registered target does not exist on E4B: in pydantic 2 projects the base model writes the v2 API every time,
  bad rate 0 for all 7 APIs. By the rule above (fewer than 2 APIs above 0.10), the test stops here and goes back to
  the user before freezing (18:02 UTC).
- The error runs the other way. In projects pinned to pydantic 1, E4B writes the v2 API, which does not exist there.
  Example: `return self.model_json_schema()` inside a model with `class Config:` and a `validator` import.
- "other" in the free style of validate and validate_json is the version-neutral `Model(**data)`, not an error.

**If the target is flipped** (option A, put to the user at 18:02 UTC): the same rule applied to the v1-project cases
keeps schema (0.81), fields (0.53) and validate_json (0.16, the third-highest; above 0.10). No new census is needed,
because the cases are the same prompts with the roles swapped (`make_pydantic_cases.py full DIR APIS 1`, checked:
same ids and rules, roles only).

**Sets built (18:50 UTC)** with `make_pydantic_cases.py full outputs/weight_edit/libtest/sets schema,fields,validate_json 1`:
- tune 54 cases and holdout 54 cases;
- each split has 24 targets (v1 projects: 12 tight, 12 free), 24 v2 keep cases and 6 non-pydantic keep cases;
- no group is shared between splits.

The style rotation depends on the list of domains and APIs. As a result, only 13 of the 24 tune target cases are
the very census cases; the other 11 are the same prompts in the other style. In those 13 the census recorded 19 bad
generations out of 26.

## Cost and timing

- **Mac only:** census about 20 min, arms 2 × 4 h, judge about 1-1.5 h.
- **Tokens:** Claude subscription, Opus 5.5.
- **Earliest go:** after the census (the GPU is free after the parent session's v6 gate, about 17:00 UTC). A verdict
  would then come about 11-11 03:00-05:00 UTC.
