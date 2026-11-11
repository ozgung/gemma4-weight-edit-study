<!-- Long-form records behind Paper 1 (not part of the paper). Moved out of docs/paper/full/paper-full.md on 2026-11-10 when the author decided the paper should hold only the summary and main findings (the 45-page version is commit 6c12b3b). Drafted from the frozen reports, plans, logs and code; the comment after each paragraph names its sources. Section letters are kept for traceability; the paper no longer cites them. -->

# Paper 1: records behind the numbers

These notes give, for each test in the paper, the protocol, when it was fixed, the full results and the run records, with the source file of every number. They are for readers and agents who want to check or extend the work; the paper itself is `docs/paper/full/paper-full.md`.

## Appendix A. The agent, the harness and the edit tool

### A.1 The agent and its recorded runs

Every recorded edit call in this study comes from one agent configuration, e001nt (E4B-NoThink-T0.6). It runs Gemma 4 E4B as the 4-bit W4A16 checkpoint in vLLM, with thinking off and the sampling settings temperature 0.6, top_p 0.95 and top_k 64. It is capped at 28 tool calls and uses the harness's own tools: `run_command`, `read_file`, `edit_file`, `write_file`, `get_status`, `submit_patch` and three code-graph tools. The editors worked on 412 task runs on the four training repositories (fastapi, rich, requests, httpx): three runs of the 91 "winnable" training tasks, one run of 123 training tasks on Kaggle L4x4, and a partial run of 16 tasks.

<!-- src: submissions/e001nt_baseline_nothink_t06_e4b/agent.yaml (model gemma-4-e4b-it-qat-w4a16-ct, tools), configs/sampling.yaml (temperature 0.6, top_p 0.95, top_k 64, thinking_budget 0), eval_config.yaml (max_tool_calls 28; also in results/e4r/results/rft1_e001nt_baseline_nothink_t06_e4b_cap12/run_meta.json, which has max_time_minutes 12); run list scripts/weight_edit/census.py:30-35; outputs/weight_edit/claude/census.csv (412 rows: rft1, rft3, rft5 91 each, l4 123, w91a 16); log 2026-11-02 20:10 UTC -->

### A.2 How the agent calls the edit tool

The model writes a tool call as plain tokens in the Gemma 4 format:

```
<|tool_call>call:edit_file{filepath:<|"|>…<|"|>,
  new_string:<|"|>…<|"|>,old_string:<|"|>…<|"|>}<tool_call|>
```

Each string value is raw text between two `<|"|>` delimiter tokens. The model writes the arguments in alphabetical order, so it writes the new text before the old text. In the recorded training calls the order was `filepath, new_string, old_string` in 785 calls and `allow_multiple, filepath, new_string, old_string` in 355; six calls put `allow_multiple` last, which shows that the parser does not sort the keys. The new text comes first in 727 of the 730 training calls used for search and in all 320 held-out test calls.

<!-- src: data/models/gemma-4-e4b-it-qat-q4_0-unquantized/chat_template.jinja:245-257 (call format, keys unquoted, dictsort); log 2026-11-02 20:26 UTC (785, 355, 6); scripts/weight_edit/replay_fullcall.py docstring lines 8-9 and log 2026-11-05 03:1x UTC (727/730, 320/320) -->

The harness's `edit_file` tool copies the file out of the sandbox, calls `apply_replacement(content, old_string, new_string, allow_multiple)`, writes the result back and returns a dictionary with the status, the number of occurrences, the matching strategy and a diff. When the old text cannot be matched, the tool returns a `FileEditError` with the message "Failed to replace: old_string not found. Ensure you're not escaping content incorrectly and check whitespace, indentation, and context."

<!-- src: data/harness_src/swegemma/tools/workspace.py:217-293 (edit_file), :338-361 (edit_file_tool returns a dict via parse_tool_response); data/harness_src/swegemma/edit.py:28-75; data/harness_src/adk_eval_core/editing/edit.py:449-453 (message). data/harness_src copies are identical to the installed .venv packages (diff -q), swegemma 0.2.11. -->

### A.3 Why tool results arrive escaped and the model's own arguments do not

The escaping comes from the serving path, not from the model. We confirmed each step by reading the code:

1. Tools return Python dictionaries; `read_file` puts the file text in a `content` field (`swegemma/tools/workspace.py:88-167`, `tools/base.py:34-44`).
2. ADK's LiteLlm adapter turns every tool result that is not already a string into a string with `json.dumps(obj, ensure_ascii=False)` (`google/adk/models/lite_llm.py:546-560`, called at `:803-809`). In that string a line break becomes the two characters `\n` and a quote becomes `\"`.
3. The Gemma 4 chat template renders a tool response in one of two ways (`chat_template.jinja:160-173`). A dictionary is shown field by field, with each string raw between `<|"|>` delimiters. Anything else is shown as `value:<|"|>…<|"|>` with the string inserted unchanged. Because the content arrives as a JSON string (`:288-289`), the second branch is always taken, and the model sees `response:read_file{value:<|"|>{"status": "ok", "content": "line1\nline2 …"}<|"|>}`.
4. The model's own past calls arrive differently. vLLM's message post-processing parses each call's argument string back into a dictionary, and the template then renders each argument raw through `format_argument` (`chat_template.jinja:118-120, 247-252`). We read this vLLM step in our copy of it (`scripts/sft_export.py:426-440`, mirroring vLLM 0.19.1), not in vLLM's own source.

<!-- src: .venv/lib/python3.12/site-packages/google/adk/models/lite_llm.py (google-adk 1.36.1, the version pinned in pyproject.toml) lines 546-560 and 803-809; data/models/gemma-4-e4b-it-qat-q4_0-unquantized/chat_template.jinja lines 118-120, 160-173, 247-252, 288-289; scripts/sft_export.py:20-35, 426-440; log 2026-11-02 20:26 UTC (rendering checked as an identical json.dumps round trip) and 2026-11-03 13:10 UTC (lite_llm.py about line 808) -->

So to copy a line from a file it has read, the model must copy and also unescape: every line break in its context is `\n` and every quote is `\"`, while the old text must hold the real characters. The prompt rebuild of Appendix B follows this same path. Its prompts matched the served token count in 751 of 1,191 training calls and 330 of 545 held-out calls, and in every mismatch the rebuilt prompt was longer, as history compaction would cause. In a small exploratory copy pilot (8 spans from 4 training files, synthetic prompts), escape artefacts appeared only when the file was shown JSON-escaped: in 10 of 32 greedy copies, and in none of 32 when it was shown raw.

<!-- src: log 2026-11-02 20:47 UTC (copy pilot; outputs/weight_edit/claude/copy_pilot/results.jsonl); Appendix B for the token-count check -->

### A.4 What the fallback does

`apply_replacement` tries four matching tiers in order: exact, flexible (lines compared after stripping surrounding whitespace), regex (a token pattern with flexible whitespace between tokens), and last the unescape fallback (`adk_eval_core/editing/edit.py:401-447`). The fallback (`:353-398`) runs only if the old text contains a backslash. It then tries five transforms in turn:

1. `\uXXXX` codes only;
2. one layer of inline JSON escapes (`\"`, `\\`, `\uXXXX`), keeping `\n`, `\r` and `\t`;
3. two such layers;
4. one full layer of JSON escapes, including `\n`, `\r`, `\t`, `\b`, `\f` and `\/`;
5. two full layers.

For each transform it skips a candidate old text that is empty or already tried. It applies the same transform to the new text, and skips the candidate if the transformed old and new texts are equal. It then tries an exact and then a flexible match. The first match is returned with strategy `"unescaped"`. An error from the exact or flexible tier, such as several occurrences, is also returned, and no later transform is tried.

<!-- src: data/harness_src/adk_eval_core/editing/edit.py:307-398 (escape regexes, transforms, _apply_unescaped_match), 401-447 (tier order); log 2026-11-02 20:47 UTC and Astra 20:51 UTC (edit.py:353-430 read independently) -->

Three consequences follow, and they recur through the paper:

- An old text that differs from the file only by escape codes is accepted. Only copies that mix escaping with other errors reach "not found", which is why the strict escape subtype in Table A.2 is small.
- The fallback also unescapes the new text. The harness does not check whether that gives correct code.
- When an edited model writes the old text raw, exact matching succeeds and the fallback no longer runs, so an escaped new text is inserted verbatim. This is why the compile guard uses the full-call replay (Appendix B.5).

### A.5 Failure census on the four training repositories

Failed edits are frequent: 405 of the 1,191 `edit_file` calls in the 412 training runs returned "not found" (Table A.1). The 18 `write_file` calls are left out.

**Table A.1.** Outcomes of `edit_file` calls in the 412 training runs.

| Outcome | Calls | Share |
|---|---|---|
| accepted | 614 | 52 % |
| old text not found | 405 | 34 % |
| old text occurs several times | 122 | 10 % |
| `old_string` argument missing | 41 | 3 % |
| other error | 9 | 1 % |

<!-- src: outputs/weight_edit/claude/edit_calls.csv (1,209 rows; tool == edit_file 1,191; error_kind counts), built by scripts/weight_edit/edit_calls.py (sha256 267ba3a0e058); log 2026-11-02 20:18 UTC -->

Most failed copies are bad copies, not pure escaping (Table A.2). The types are heuristic. Each failed old text is compared with the text of that file the agent had seen earlier in the same run (through `read_file`, or `run_command` output that names the path), and that seen text is reset after each successful edit of the file.

**Table A.2.** Types of the 405 not-found old texts.

| Type | Calls | Share |
|---|---|---|
| partial copy: at least half of its lines were seen | 142 | 35 % |
| mostly invented: under half of its lines were seen | 124 | 31 % |
| file not seen before the call | 93 | 23 % |
| every line seen, but not contiguously | 36 | 9 % |
| differs only by escape codes | 7 | 2 % |
| differs only by whitespace | 3 | 1 % |

"File not seen" undercounts views through shell commands that do not name the path.

<!-- src: outputs/weight_edit/claude/edit_calls.csv, column subtype for error_kind == not_found (recount matches); definitions in scripts/weight_edit/edit_calls.py:41-89; log 2026-11-02 20:18 UTC -->

Failure rises with the length of the old text and with the length of the prompt (Table A.3). Here the rate is not-found calls over accepted plus not-found calls; calls whose old text occurred several times or was missing are left out. Reading the file first barely separates the outcomes: 484 of 614 accepted calls (79 %) and 312 of 405 not-found calls (77 %) came after a read of that file.

**Table A.3.** Not-found rate by old-text length and prompt length.

| Old text, lines | 1 | 2–3 | 4–8 | 9–20 | 21+ |
|---|---|---|---|---|---|
| not found / (accepted + not found) | 74/250 (30 %) | 94/240 (39 %) | 104/297 (35 %) | 94/172 (55 %) | 39/60 (65 %) |

| Prompt, tokens | under 8k | 8–12k | 12–16k | 16k+ |
|---|---|---|---|---|
| not found / (accepted + not found) | 82/284 (29 %) | 183/449 (41 %) | 133/275 (48 %) | 7/11 (64 %) |

<!-- src: recount from outputs/weight_edit/claude/edit_calls.csv (columns old_lines, prompt_tokens, read_before, error_kind); the percentages equal those in log 2026-11-02 20:18 UTC, whose denominator is not stated there -->

The fallback hides much of the escaping. Of the 614 accepted calls, 498 matched exactly, 108 (18 %) only through the fallback, 5 through regex and 3 through flexible matching. Of the 65 accepted calls whose old text held a literal `\n`, 59 went through the fallback.

<!-- src: recount from the traces of the five run directories in scripts/weight_edit/census.py:30-35 (412 runs, 1,191 edit_file calls, 614 ok: unescaped 108, exact 498, flexible 3, regex 5; literal \n in old_string among ok calls: unescaped 59, exact 6); log 2026-11-02 20:47 UTC -->

In replay, the unedited model writes an escape artefact (a literal `\n` or `\"` in the old text that the file does not contain) in 42 of 450 generations on the 35 round-1 selection tasks, 194 of 1,010 on the 83 search tasks, and 100 of 640 (16 %) on the held-out test calls. Of those 100, the fallback accepted 61 and 39 were not found.

<!-- src: outputs/weight_edit/claude/wp12/wp12_primary_report.json served.escape_artefact 42 (n 450); wp12_search_exploratory_report.json served.escape_artefact 194 (n 1010); outputs/weight_edit/claude/gate_r2/gate_r2_report.json base_escape_artefacts 100 (n 640); recount from gate_r2/gen_base.jsonl and gen_base.scored.jsonl: 61 applied_unescaped, 39 not_found (matches log 2026-11-04 14:xx UTC) -->

Edit failures go with unresolved runs, but this is an association. A not-found call occurred in 124 of 335 unresolved runs (37 %) and 7 of 77 resolved runs (9 %). Counting any edit error, the shares are 153 of 335 (46 %) and 12 of 77 (16 %).

<!-- src: recount joining outputs/weight_edit/claude/census.csv (resolved) with edit_calls.csv (error_kind). draft-v3 §1 says 44 % and 9 %; those match "any error except a missing old_string" (148/335, 7/77) and predate the 20:18 UTC fix that counts missing-parameter errors. See [CHECK] A3. -->

## Appendix B. The replay

### B.1 What is rebuilt

A replay regenerates one argument of one recorded edit call from the exact situation in which the agent wrote it. For each recorded `edit_file` call we rebuild three things.

- **The prompt.** We replay the trace through the harness's own code path: ADK's request preprocessing, the submission's LiteLLM patch, a copy of vLLM 0.19.1's message parsing, and the model's chat template. The rebuilt prompt is kept only if its token count equals the trace's `prompt_tokens` for that call.
- **The prefix.** This is the model's own output for that call up to `old_string:<|"|>`: its thought, any visible text, earlier parallel tool calls, the call header and the arguments written before the old text, including the new text.
- **The file state.** We take the task's snapshot file and re-apply every earlier successful `edit_file` or `write_file` of the same run with the harness's `apply_replacement`. A file named by a shell command that may change it (`sed -i`, redirection, `tee`, `git checkout|apply|restore|stash|reset`, `patch`, `mv`, `cp`, `rm`) is marked uncertain from then on. A modifying command that names no file makes every file in the run uncertain.

<!-- src: scripts/weight_edit/replay_set.py docstring and MODIFY_RE (sha256 3cb9b109b9f8); scripts/sft_export.py:20-35; log 2026-11-02 20:50 and 20:57 UTC; held-out builder scripts/weight_edit/replay_set_ood.py (sha256 811b8645fce2), outputs/weight_edit/claude/replay_ood/README.md -->

For the held-out calls, the snapshot files came from the Kaggle snapshot archives of OOD-43 and OOD-B, extracted on a Colab CPU runtime (308 paths, 2.3 MB). The rebuilt file states pass a sanity check on both sets. On training calls, all 341 recorded exact-match old texts occur in the rebuilt file states, and none of the 228 recorded not-found ones do. On held-out calls, re-scoring the recorded old texts reproduces every recorded outcome: 213 of 213 successes and 107 of 107 rejections.

<!-- src: outputs/weight_edit/claude/replay_ood/README.md ("Base files", "Sanity"); log 2026-11-02 20:57 UTC (341 exact-ok occur, 228 not-found do not), 2026-11-03 18:5x UTC -->

### B.2 Which calls are usable

A call is usable when its prompt is verified, its file exists and its state is certain, and it has an old text. About a third of the calls fail the prompt check, and all of them for the same reason (Table B.1). In every such call our rebuilt prompt is longer than the served one: by 3,465 to 37,808 tokens on training calls (median 9,557) and by 3,899 to 27,544 on held-out calls (median 9,574). This matches the harness's history compaction in long served runs, which the rebuild does not reproduce. The usable calls are therefore biased toward calls made before compaction, earlier in a run. A token-count match after the same rendering path is strong evidence but not proof of identical prompts: these runs left no vLLM token logs to compare token IDs against.

**Table B.1.** From recorded calls to usable replay calls.

| | Training repositories | Held-out repositories |
|---|---|---|
| recorded `edit_file` calls | 1,191 | 545 |
| prompt token count verified | 751 (63 %) | 330 (61 %) |
| dropped: file state uncertain | 2 | 2 |
| dropped: file missing from snapshot | 0 | 2 |
| dropped: no old text in the call | 19 | 6 |
| **usable calls** | **730** | **320** |
| tasks / repositories | 118 / 4 | 101 / 69 |
| recorded outcome: accepted / not found / several occurrences / other | 426 / 228 / 75 / 1 | 213 / 103 / 3 / 1 |

<!-- src: recount from outputs/weight_edit/claude/replay/calls.jsonl and replay_ood/calls.jsonl (fields verified, file_missing, state_uncertain, old_string, prompt_tokens_ours - prompt_tokens_trace over unverified calls: training 440 calls, min 3,465, median 9,557, max 37,808; held-out 215 calls, min 3,899, median 9,574, max 27,544); replay/build.log; replay_ood/summary.json (calls 545, verified 330, usable 320, kinds, tasks 101, repos 69); log 2026-11-02 20:57 UTC ("749 usable", which includes the 19 calls without an old text) -->

The held-out calls come from five runs of the same configuration on the two held-out sets: four runs on OOD-43 (209 usable calls, 43 tasks) and one on OOD-B (111 calls, 58 tasks). They share no repository with the training set or with each other, and no editor saw any of them. Among usable calls, 31 % of training calls and 32 % of held-out calls were recorded failures. The training calls were split by task within each repository (a fixed sha1 order, the first 30 % of each repository's tasks to selection): 83 search tasks with 505 calls, and 35 selection tasks with 225 calls. Round 1 searched on the 505 and was judged on the 225. Round 2 searched on all 730 and was judged on the 320 held-out calls.

<!-- src: replay_ood/summary.json (usable_by_suite ood45 209, oodb 111; tasks 43, 58; training_repo_overlap [], suite_overlap_repos []); outputs/weight_edit/shared/split.json (rule, created 2026-11-02T20:54:04Z, 83/35 tasks); log 2026-11-02 20:57 UTC (505/225 calls), 2026-11-03 18:25 UTC (round-2 search data, 730 calls) -->

### B.3 Generation, draws and scoring

The model regenerates only the old text, on the bf16 weights of E4B in MLX on a Mac, from the rebuilt prompt plus prefix, and stops at the closing `<|"|>` token, with a limit of min(1,024, 2 × the recorded old text's length in tokens + 64) new tokens. Each call gets several *draws*. Draw 0 is greedy, and later draws sample at the served settings (temperature 0.6, top_p 0.95, top_k 64). The seed is 7919 × i + d for call index i and draw d, so every arm sees the same seeds in the same key order. The fidelity check used three draws per call; the round-1 and round-2 tests used two.

<!-- src: scripts/weight_edit/replay_gen.py (sha256 e090d2500559) docstring lines 1-13 and lines 109-111, 139, 150-151 -->

Each generation is scored by the harness's own function. The triple of file state, generated old text and recorded new text, with the recorded `allow_multiple`, goes through `swegemma.edit.apply_replacement`, exactly as the tool would apply it. The outcome is accepted (with its strategy) or refused (not found, several occurrences, empty, other error). We also record whether the generation is an exact copy, whether it equals the recorded old text, and, for accepted Python files, whether the resulting file still compiles.

<!-- src: scripts/weight_edit/replay_score.py (sha256 bc2a19433c2e) docstring and outcome() -->

### B.4 Fidelity

The bf16 replay reproduces most recorded failures and every recorded success (Table B.2). The recorded calls were made by the served W4A16 model, sampling at temperature 0.6, so exact agreement is not expected.

**Table B.2.** Replay outcome against the recorded outcome (greedy draw unless stated).

| | Training sample (120 calls) | Held-out test set (320 calls) |
|---|---|---|
| recorded not found, still not found | 43/48 (90 %; 78–96) | 95/103 (92 %; 85–96) |
| recorded success, still accepted | 60/60 | 213/213 |
| recorded success, still accepted, sampled draws | 120/120 | 213/213 |
| recorded not found, still refused, sampled draws | 87/96 | 96/103 |
| greedy old text identical to the recorded one | 108/120 (90 %) | 279/320 (87 %) |
| recorded old text re-scored gives the recorded outcome | 120/120 | 320/320 |

Ranges are Wilson 95 % intervals. The training sample is stratified (ok-exact, ok-fallback, not found, several occurrences) and mixes search and selection tasks. In the training greedy draw, one of the 48 recorded failures became an "other error" and four were accepted; in the held-out greedy draw, eight were accepted. Regenerating the round-1 base arm later reproduced all 450 earlier generations, so the replay is deterministic for a fixed seed.

<!-- src: outputs/weight_edit/claude/replay/gen_fidelity_base.scored.jsonl (360 rows = 120 calls x 3 draws; greedy not_found 43, other_error 1, applied 4 of 48; ok 60/60; sampled ok 120/120; sampled not-found unaccepted 87/96; same_as_recorded 108/120; recorded_replayed agrees on all rows); outputs/weight_edit/claude/gate_r2/gen_base.scored.jsonl (draw 0: not_found 95, applied 8 of 103; ok 213/213; same_as_recorded 279/320; draw 1: not_found 96/103, ok 213/213) and gate_r2/score_base.log ("recorded-call replay agrees"); Wilson intervals recomputed; WP12 served arm identical to gate/gen_base.jsonl in 450/450 (log 2026-11-03 19:0x UTC). Log 2026-11-02 21:23 UTC gives "131/144" for sampled not-found calls; that figure counts all three draws (44 + 87). See [CHECK] B1. -->

### B.5 Old-text replay and full-call replay

The two replays differ in which arguments the model writes. The *replay* keeps the recorded new text in the prefix and regenerates only the old text. The *full-call replay* cuts the prefix just after `new_string:<|"|>`, generates the new text, teacher-forces `<|"|>,old_string:<|"|>`, and then generates the old text. It scores the generated pair, with the same seeds and draws as the replay. Its code is `replay_fullcall.py`, `replay_score_fullcall.py` and `gate_r2_fullcall.py`, all first committed at 03:12 UTC on 2026-11-05.

<!-- src: scripts/weight_edit/replay_fullcall.py docstring (sha256 e213adcd… in the log); log 2026-11-05 03:1x UTC; git commit ee0b34d 2026-11-05 03:12:55 UTC (first commit of all three scripts) -->

The old-text replay misjudges one thing: compile failures after an edit. Because the new text is frozen, an edit that makes the old text exact also turns off the fallback, which would have unescaped the recorded, still-escaped new text. The escaped new text then goes into the file verbatim, and the file stops compiling, whatever the edited model would have written. On the search set, every pass that cut artefacts raised the old-text replay's compile-failure rate: 5 of 5 of Opus's passes, 10 of 10 of Fable's and 1 of 1 of Astra's, by 2 to 21 points. In Astra's pass 4, 37 of the edit's 67 compile failures were exact matches with an escaped recorded new text, against 2 of 31 for the base model. So the compile guard was moved to the full-call replay on 2026-11-05, before any round-2 edit was generated on the test calls; both verdicts are reported in Appendix D. The acceptance and working-call guards stayed on the old-text replay, because exact matching ignores the new text.

<!-- src: log 2026-11-05 03:1x UTC ("Defect", "Evidence", "Not a defect", "Guard of record") -->

**Table B.3.** The base model under the two replays (held-out test set, 320 calls × 2 draws).

| Base model, 640 generations | Old-text replay | Full-call replay |
|---|---|---|
| old text accepted | 443 | 444 |
| compile failures among accepted | 9.0 % | 7.9 % |
| accepted and compiling | 401 | 407 |
| escape artefacts in the old text | 100 | 89 |
| escape artefacts in the new text | (recorded text) | 67 (10.5 %) |

<!-- src: outputs/weight_edit/claude/gate_r2/gate_r2_report.json candidates.fable_r2.compile.base (accepted 443, compile_fail_rate_among_accepted 0.0903, accepted_and_compiling 401) and base_escape_artefacts 100; gate_r2_report_amended.json base (accepted 0.69375 = 444/640, compile_fail_rate_among_accepted 0.0788, accepted_and_compiling 407, artefact_old 0.1390625 = 89/640, artefact_new 0.1046875 = 67/640) -->

For the base model the two replays agree closely, and they differ mainly when the regenerated new text differs from the recorded one. In the greedy draw, the full-call new text equals the recorded new text in 186 of 320 calls. In 184 of those 186, the full-call old text is identical to the old-text replay's. This matches the smoke test logged with the amendment.

<!-- src: recount comparing outputs/weight_edit/claude/gate_r2/fullcall/gen_base.jsonl (gen_new, gen) with replay_ood/calls.jsonl (new_string) and gate_r2/gen_base.jsonl (gen), draw 0; smoke test in log 2026-11-05 03:1x UTC ("Code" bullet) -->

## Appendix C. The editors: sandbox, budgets and what each edit changes

### C.1 Sandbox, kernel and forward-pass server

Each editor ran as a fresh agent with no access to our log, our status file or the test data. The Claude models ran in Claude Code and Astra in the Codex CLI, all at the highest reasoning effort. A macOS sandbox profile was the boundary. The editor could read a read-only packet, the model's config and tokenizer, and the mlx_lm source, and could write only its own run folder. The packet held the brief, a dossier on round 1 (revision 3, with no test-set statistics), the server specification, the search data and a small weight kernel.
<!-- src: log 2026-11-04 13:2x UTC ("Isolation", "Inputs") and 17:1x UTC ("Dossier rev 3"); log 2026-11-04 21:0x UTC (Codex isolation, CODEX_HOME); outputs/weight_edit/sandbox/fable_r2/brief.md "Rules" -->

The editors worked through two interfaces:

- **Kernel** (`wsp_kernel.py`): read any weight as a float32 array, and save an edit W′ = W + BA of rank at most 128 on any of 258 matrices. These are the q, k, v, o, gate, up and down projections of the 42 decoder layers; layers 24 to 41 have no k or v projections, because they reuse the keys and values of earlier layers. Embeddings, norms and the output layer could not be edited.
- **Forward-pass server** (`wsp_server.py`, MLX, bf16, one process on the Mac): the only way to run the model. An editor wrote a request file. The server ran it and counted one pass if it succeeded; a refused or failed request did not count. A request could pin the edit snapshot by its hash, and every finished pass recorded the snapshot it used.
- **`gen` request**: regenerate the old text at up to 128 recorded search calls, in one or two draws (greedy, then temperature 0.6), for the base model, the current edit or both. Each output was scored by the harness's own edit function against the true file state. The result gave acceptance, not-found, exact-copy, compile and escape-artefact rates, paired base-against-edit tables and the generated text per call.
- **`read` request**: a teacher-forced pass over up to 32 texts, at up to 8 layers, 256 positions, 2 attention layers and 512 continuation tokens. It returned residuals, the GeGLU activations entering the down projection, attention probabilities and log-probabilities with the top-5 alternatives.
- Both request types could use the served view of tool responses or a raw view. The edit was judged under the served view only.
- Each run could make at most 600 Python runs and use at most 4 counted conversations.
<!-- src: outputs/weight_edit/sandbox/fable_r2/server_spec.md (request types; LIMITS gen_calls 128, gen_draws 2, read_texts 32, read_layers 8, read_positions 256, attn_layers 2, continuation_tokens 512); sandbox/fable_r2/brief.md "What you have" (258 matrices, r <= 128, 600 Python runs); log 2026-11-04 13:2x UTC (harness v2) and 17:1x UTC (harness v2b, raw rendering) -->

The search data were 730 recorded edit calls from 118 tasks in the four training repositories: fastapi 376, rich 275, requests 71 and httpx 8. Each call came with its exact prompt and file state, and with a raw-view copy. Every run's state file records the same split. The editors divided these calls themselves into tuning and check sets. Fable, for example, tuned on an enriched set of 128 calls and checked on three natural sets of 128, 128 and 123 calls; Astra split tasks into discovery and validation halves before seeing any edit result.
<!-- src: sandbox/fable_r2/search/calls.jsonl (730 rows, 118 instance ids; repo counts fastapi/fastapi 376, Textualize/rich 275, psf/requests 71, encode/httpx 8; sha256 4a96fd0654cb); search/calls_raw.jsonl sha256 9125bdc1a6b7; state.json split_sha256 903e9abaf555 in every run; fable_r2/final_answer.md §3; astra_r2/final_answer.md (seed 4217, 71 + 47 tasks) -->

The informed arms had two additions. Their packet held a read-only `prior/` folder with the three blind runs' notebooks, final answers, state, search-set results, edits and small code files (428 files, 9.2 MB). And each of their `gen` passes also ran the full-call replay, so they saw the compile figure that the amended guard uses. The repair arm's `prior/` added the gradient-trained adapter G-tam-39 (Section 3.2 of the main text) as its starting point.
<!-- src: log 2026-11-05 16:10 and 16:54 UTC (prior/ contents, server v2c, MANIFEST fc5aa561e44a, verified with shasum); log 2026-11-09 19:05 UTC (fable52inf_r2 uses the same prior_opusinf) and 19:40 UTC (prior_rep MANIFEST ca699fa57c1c, verified) -->

### C.2 Arms, budgets and cost

Table C.1 lists every second-round arm that ran. Opus, Fable and Astra shared one server. Informed Opus at first shared the GPU with the round-2 gate; the Fable 5.2 arms ran alone.

**Table C.1.** Second-round editor runs.

| Arm (run id) | Model | Passes used of budget (gen / read) | Conversations | Run (UTC) | Server time, h | Tool calls | Output tokens | Input tokens, incl. cached | Submitted |
|---|---|---|---|---|---|---|---|---|---|
| Opus (`opus_r2`) | Claude Opus 5.5 | 25 of 25 (15 / 10) | 1 | 11-04 17:11 to 11-05 16:05 | 8.5 | 193 | 453k | 80.3M | an edit |
| Fable (`fable_r2`) | Claude Fable 5.1 | 25 of 25 (19 / 6) | 1 | 11-04 17:12 to 11-05 13:44 | 10.6 | 154 | 359k | 46.4M | an edit |
| Astra (`astra_r2`) | GPT-6 Astra | 10 of 10 (7 / 3) | 5 | 11-04 20:56 to 11-05 14:14 | 3.5 | 251 | not recorded | not recorded | nothing |
| Informed Opus (`opusinf_r2`) | Claude Opus 5.5 | 10 of 10 (8 / 2) | 2 | 11-05 16:55 to 11-06 21:29 | 14.8 * | 181 | 478k | 61.6M | an edit |
| Informed Fable 5.2 (`fable52inf_r2`) | Claude Fable 5.2 | 10 of 10 (6 / 4) | 1 | 11-09 19:15 to 11-10 04:05 | 6.5 | 289 | 1,024k | 133.1M | nothing |
| Repair Fable 5.2 (`fable52rep_r2`) | Claude Fable 5.2 | 10 of 10 (7 / 3) | 1 | 11-10 04:08 to 12:14 | 7.1 | 145 | 474k | 64.9M | nothing |

Server time sums the seconds the server spent on each pass; arms that ran at the same time shared the GPU. \* Informed Opus's first pass took 6.5 h because the server was paused for gate priority from 17:36 UTC on 11-05. Tokens are summed over the result events of each Claude Code transcript. Astra's Codex transcripts recorded usage for one turn only (722,580 input and 10,243 output tokens), so its total is not known.
<!-- src: proposers/<arm>/state.json (budget, used, passes[].type, passes[].secs, start_utc, utc); claude/round2/round2.log (17:11:08 and 17:12:08 starts; exits 13:43:34 and 16:04:54), round2/astra.log (20:55:49 start, 14:13:33 exit), proposers/<arm>/launcher.log (exit times); proposers/<arm>/audit.json "tool_calls"; tokens: sum of "usage" over type=="result" events in transcript*.jsonl (output_tokens; input_tokens + cache_read_input_tokens + cache_creation_input_tokens); Astra: the only turn.completed event in astra_r2/transcript*.jsonl; pause: log 2026-11-05 17:36 UTC -->

How the runs went, in brief:

- **Opus and Fable** each finished in one conversation, with all 25 passes used.
- **Astra** was added at the user's request with 10 passes, the smallest budget of the weight-space programming blog [1]. Its run spanned five Codex conversations. The first ended on a usage limit, the second and third on operator errors (the third with no editor turn), and the fourth on a provider capacity error after 9 of 10 passes. The fifth was not in the registration and was added, with the user's approval, with the budget unchanged.
- **Informed Opus**'s first conversation ran against the paused server and was not counted. A scheduling fault on our side then delayed its second conversation to 09:40 UTC on 11-06. That conversation's prompt carried an operator note with infrastructure facts only.
- **Fable 5.2, informed** replaced a registered informed Fable 5.1 arm (`fableinf_r2`). That arm was stopped after one pass when Fable 5.2 became available and has no result. The two Fable 5.2 arms each ran in one conversation.
- Two more arms were registered and never ran, Astra 6.2 (`astra62_r2`) and an informed GPT-6.2-sol arm (`solinf_r2`). Each was to start only once its model was offered in the Codex CLI. The watcher for the second was stopped on the user's word on 11-10.
<!-- src: log 2026-11-04 21:0x UTC (Astra budget "the blog's smallest"); log 2026-11-05 13:55 UTC (five conversations, wording); round2/astra.log (conversation exits rc 1, 143, 143, 1, 0); log 2026-11-05 21:01 UTC and 2026-11-06 09:33, 09:41 UTC (pause, scheduling fault, operator note); log 2026-11-09 19:05 UTC (fableinf_r2 stopped after 1 pass); log 2026-11-05 14:05 UTC ("runs only if GPT-6.2 Astra is offered by the Codex CLI"); log 2026-11-08 22:35, 22:40 UTC; round2/solinf/solinf.log (launch refused, model not in catalog) and watch_sol.log ("watcher stopped by the user's word" 2026-11-10T14:17:05Z); no proposers/astra62_r2 or solinf_r2 folder -->

### C.3 What "gradient-free" covered, and how we checked it

The brief stated the rule as: "no gradients, autograd, backward passes, optimisers or iterative fitting of weights to a loss. Closed-form computations from weights and activations (means, projections, SVD, least squares in closed form, thresholds you choose by reasoning) are allowed." The weight kernel offered no gradient function, though the MLX library in the editors' Python could have computed one.
<!-- src: sandbox/fable_r2/brief.md "Rules" (identical across arms except the header); sandbox/fable_r2/wsp_kernel.py functions: names, shape, load, allowed, save_edit, remove_edit, list_edits; fable_r2/scratch/build_v5.py imports mlx.core -->

We checked each run in three ways:

- **Kernel log.** Each `kernel_log.jsonl` records only `save_edit` and `remove_edit` events, with the module, rank, Frobenius norm and the editor's note. It shows what was saved and when, not how it was computed.
- **Code and transcripts.** We searched every transcript and every saved Python script for gradient and optimiser calls (`value_and_grad`, `mx.grad(`, `.backward(`, `requires_grad`, `torch.optim`, `mlx.optimizers`, `jax.grad`). There were no hits in any of the six runs (121, 50, 21, 97, 99 and 43 scripts). The word "autograd" appears only in the brief's rule text and in editors' notes quoting it.
- **Audit** (`audit.json`, Section C.4).
<!-- src: proposers/<arm>/kernel_log.jsonl op counts (opus_r2 67 save_edit; fable_r2 11 save_edit; astra_r2 7 save + 2 remove; opusinf_r2 16 save; fable52inf_r2 21 save + 3 remove; fable52rep_r2 2,065 save + 515 remove); grep over proposers/<arm>/transcript*.jsonl and <arm>/{scratch,work}/**/*.py, run 2026-11-10 -->

What each editor actually used:

- **Fable**: only `np.linalg.solve`. Its two gates are weighted ridge regressions (ridge strength 3) on layer-41 GeGLU activations from its read passes 1, 2 and 5, with targets of 1 at escape rows of the gate's type and 0 elsewhere, and legitimate-backslash rows weighted 5. Each gate uses the 60 activations most correlated with its label. Its two output directions are minimum-norm least-squares solutions through the post-feed-forward norm weight, the final norm weight and the tied unembedding, with a 1e-6 ridge term. The target logit changes were set by Fable (−1 on 49 escape tokens, +0.5 on the gate's own raw family, −0.3 on newline tokens, 0 on raw counterparts and the 300 most frequent code tokens) and are linear in the logits, without the softmax. No eigendecomposition appears in its code or transcript.
- **Opus**: ridge solves and eigendecompositions in closed form ("weights came from closed-form ridge and eigen solves"). It also recovered layer-41 MLP inputs from captured outputs with an iterative Gauss-Newton fit. That fit solves for activations, not weights.
- **Informed Opus** used the same Gauss-Newton inversion. It also chose gate thresholds and gains with small grid searches and one coordinate search over a simulated score on an offline simulator. It flagged both itself as close to the rule's limit and left the judgement to us. Its output vectors were closed form.
- **Astra**: one closed-form linear solve per variant.
- **Fable 5.2 (informed)**: ridge solves, constrained least squares, and thresholds chosen by evaluating the last layer's equations on captured activations, with leave-one-task-out refits.
<!-- src: fable_r2/scratch/gate_fit_v3.py (np.linalg.solve with 3.0 * I, 60 candidates by correlation, LEGIT_W 5.0) and scratch/build_v5.py (G = E[ids] * (w_final * w_post); G.T @ solve(G G.T + 1e-6 I, tgt)); fable_r2/final_answer.md §1; grep -w eigh over fable_r2 transcript and scratch: 0 hits, linalg.solve 22 in transcript; opus_r2 transcript: linalg.eigh 4, linalg.solve 84; opus_r2/final_answer.md §1; opus_r2/notebook.md runs 20-23 (Gauss-Newton); opusinf_r2/final_answer.md §7 "Conduct notes"; astra_r2/final_answer.md; fable52inf_r2/final_answer.md §1 -->

Fable's version and strength were chosen within its 25 passes. It built versions v1 to v5 one after another, each after reading the previous pass on its search sets. It then ran v5 at three strengths, 6.5, 8 and 9.5, on natural sets C and D. Before its pass 24 it wrote a rule in its notebook for the last choice: "choose alpha 9.5 if it is not worse on lost calls and acceptance and at least as good on tasks down; otherwise restore alpha 8". On set B, strength 9.5 gave 9 tasks down against 8, the same lost calls (3 and 2 per draw) and acceptance −0.95 points against −1.58, so 9.5 was kept. The saved file (sha256 9302714d09f0) is the one the gate froze.
<!-- src: fable_r2/notebook.md (runs 48-67; "Decision rule for pass 24 vs pass 11"; "FINAL DECISION"); fable_r2/final_answer.md §3 table (passes 11, 17-20, 24) and §6; fable_r2/kernel_log.jsonl (alpha 6.5, 8.0, 9.5 saves); claude/gate_r2/fable_r2/FROZEN.txt -->

### C.4 Audits

Every transcript was passed through `audit_transcripts.py`, which flags tool use outside the allowed set, network commands, paths outside the sandbox, model loads and sandbox denials. Table C.2 gives the counts. No run loaded the model outside the server or used another tool.

**Table C.2.** Audit flags per run.

| Run | Tools | Network | Paths | Model loads | Denials |
|---|---|---|---|---|---|
| `opus_r2` | 0 | 3 | 2 | 0 | 1 |
| `fable_r2` | 0 | 0 | 0 | 0 | 0 |
| `astra_r2` | 0 | 1 | 9 | 0 | 14 |
| `opusinf_r2` | 0 | 2 | 2 | 0 | 3 |
| `fable52inf_r2` | 0 | 4 | 2 | 0 | 2 |
| `fable52rep_r2` | 0 | 0 | 1 | 0 | 1 |
<!-- src: proposers/<arm>/audit.json "counts"; round2/round2.log audit lines; log 2026-11-10 04:10 and 12:20 UTC -->

We read every flag. All network flags are notebook text that the keyword match caught; no command made a network call. The path and denial flags have four sources:

- Opus tried to list, and both Fable 5.2 runs tried to read, the operator's Claude memory folder. The sandbox refused each attempt and returned nothing.
- Astra listed the folders above its run folder and tried to read the repository's AGENTS.md, which was refused.
- The sandbox refused temporary files for some of Astra's shell here-documents.
- The rest are reads of notebooks whose text mentions a denial: Astra's own notebook, and the prior runs' notebooks in the informed packets. Informed Opus also read one of its own saved tool outputs from its Claude Code session folder.

The editors disclosed their own slips in their final answers. Opus listed file names in the mlx package; informed Opus ran one `grep` on an mlx file outside the allowed list and three stray system-Python commands; Astra recorded its refused AGENTS.md read.
<!-- src: audit.json "network", "paths", "denials" entries, all read 2026-11-10; transcript tool results for the MEMORY.md reads (EPERM in fable52inf_r2 and fable52rep_r2); log 2026-11-05 13:55 UTC (Astra flags); opus_r2/final_answer.md §6; opusinf_r2/final_answer.md §7; astra_r2/final_answer.md last paragraph -->

### C.5 The submitted edits

Table C.3 describes the three submitted edits, the withdrawn v6 that was later gated off the record (Appendix F.6), and Astra's best unsubmitted candidate. The designs are the editors' own accounts; Appendix E reports which parts independent checks confirmed.

**Table C.3.** What each edit changes. Frobenius norms are of the change BA; the share is of the original matrix W. "Total" is the root of the summed squares.

| Edit | Matrices | Rank | Frobenius norm of change (share of W) | Design, in the editor's words, shortened |
|---|---|---|---|---|
| Opus | layer 40 down; layer 41 gate, up, down | 1; 40, 40, 40 | 2.82 (5.5 %); 27.90 (28.1 %), 5.91 (6.0 %), 5.49 (11.4 %); total 29.17 | 40 little-used layer-41 feed-forward units rewired as one detector per escape token, firing only where a layer-40 "safe to clean" flag says the new text is clean; each unit lowers its escape token and lifts the raw counterpart |
| Fable | layer 41 down | 2 | 5.06 (10.5 %); total 5.06 | two closed-form gates over 118 layer-41 activations (escaped punctuation, escaped quote) times two least-squares directions that lower escape tokens and raise raw ones; strength 9.5 |
| Informed Opus | layer 41 gate, up, down | 80, 80, 80 | 45.33 (45.7 %), 8.59 (8.7 %), 3.80 (7.9 %); total 46.29 | 80 little-used units, one per escape token (Opus's 40 plus 40 line-end and quote tokens), gated by tied-embedding evidence with per-family thresholds and gains; no flag |
| v6, withdrawn (Fable 5.2, informed) | layer 41 gate, up, down | 86, 86, 86 | 38.57 (38.9 %), 20.26 (20.4 %), 8.15 (16.9 %); total 44.32 | 43 pairs of units relabel 40 escape tokens and 3 triple-quote starts to their raw twin, gated by closed-form ridge read-outs that separate artefact rows from legitimate escapes |
| Astra, not submitted | layer 41 down | 13 | 1.09 (2.3 %) at half strength | ridge keys on late MLP states with directions from raw-minus-escaped token embeddings; failed on Astra's own validation tasks |
<!-- src: npz files in proposers/{opus_r2,fable_r2,opusinf_r2,fable52inf_v6}/edit/ (B, A shapes; Frobenius computed as sqrt(trace((B^T B)(A A^T))), equal to each file's meta delta_fro and rel_change); unit counts: nonzero columns of A in layer-41 down_proj (40, 80, 86) and nonzero gate rows (Fable 60 + 60, union 118); hashes equal claude/gate_r2/<arm>/FROZEN.txt and proposers/fable52inf_v6/sha_v6.json; designs: opus_r2/final_answer.md §1, fable_r2/final_answer.md §1, opusinf_r2/final_answer.md §1, fable52inf_r2/final_answer.md §1, astra_r2/final_answer.md -->

All submitted edits sit in the last two layers. Layers 24 to 41 read keys and values computed at layers 22 and 23, so a change in layer 40 or 41 cannot alter what other positions attend to. It changes only the next-token logits at the position being generated, and later positions only through the tokens written. This follows from the architecture in the brief; Opus and informed Opus used it as a design principle.
<!-- src: sandbox/fable_r2/brief.md (architecture paragraph); opus_r2/final_answer.md §1; opusinf_r2/final_answer.md §1 -->

The two Fable 5.2 runs ended with an empty edit. The informed arm used 10 passes in one conversation (289 tool calls) and removed its tensors from `edit/` at 04:02 UTC on 11-10, keeping v6 with recorded hashes. The repair arm started from G-tam-39, used 10 passes (7 gen, 3 read, 145 tool calls) on 128 calls from the 86 tasks without any of the adapter's training calls, and tried four repair families. Its runner logged "empty edit, not a candidate" at 12:14 UTC. Appendix F.6 gives what each found.
<!-- src: log 2026-11-10 04:10 UTC and 12:20 UTC; round2/fable52inf and fable52rep runner logs ("empty edit, not a candidate" 04:05:16 and 12:14:18 UTC); fable52inf_r2/kernel_log.jsonl (3 remove_edit); fable52rep_r2/final_answer.md §1-2 -->

## Appendix D. Rules fixed in advance, changes and timing

### D.1 What the round-2 rule fixed

The round-2 rule was written in our log and committed to git as commit 606ce41. It fixed the following:

- **Test data**: the 320 usable replay calls (101 tasks, 69 held-out repositories from OOD-43 and OOD-B), each generated in two draws, greedy and temperature 0.6. The base arm was the existing base generation file.
- **Edits and controls**: each candidate is frozen by copy, with its hashes, before any generation. Each faces two random controls with the same modules, rank and Frobenius norm per module and random directions, with fixed seeds.
- **Primary measure**: the per-task escape-artefact rate. An escape artefact is a literal `\n` or `\"` in the generated old text that the true file does not contain. The edit must show a negative task-macro mean and a one-sided exact sign test (tasks lower against higher) with p < 0.05 against the base and against each control. Its p is the largest of the three.
- **Holm correction across candidates**: the blind family was {Opus, Fable} with m = 2. Every later arm was registered as its own family with m = 1, and families are not pooled. An empty edit is not a candidate.
- **Three guards**, all required: task-macro acceptance at least the base's minus 1 point; at most 2 working calls lost in each draw (the larger of 2 and 1 % of the 213 working calls, rounded down); compile failures among accepted calls at most 2 points above the base.
- **Key secondary**: acceptance against all three references, tested only if the primary passes.
- **Completeness**: the analysis stops on any missing or duplicate record or a record without its edit.
- **Reported, not part of the rule**: task and repository-clustered bootstrap intervals, OOD-43 and OOD-B separately, results by recorded outcome, and what the edit does to the base model's artefact generations.
- **After the gate**: only a passing edit would go on, to an agent-level test registered separately. Otherwise round 2 would be reported as a null.
<!-- src: log 2026-11-04 13:2x UTC ("Gate", "After the gate") and 17:1x UTC ("Primary", "Guards", "Key secondary", "Reported"); git show 606ce41:scripts/weight_edit/gate_r2_analyze.py docstring (sha256 96e49cd06ba5, as logged); replay_ood/usable_keys.txt sha256 19385ab9629a (320 lines); gate_r2/gen_base.jsonl 23feddb57f0d and gen_base.scored.jsonl 60e1844cb3c8; scripts/weight_edit/gate_controls.py c7adb2bd1fcf; seeds 20261104 + 10·i + k (blind), 20261200 (Astra), 20261400 (informed Opus), 20261800 (Fable 5.2 informed), 20261900 (repair), 20262000 (v6 off-record) -->

The analysis code changed once after 606ce41, when the Astra arm was added: version 961f315f77ca gained a `--report` option for separate families, with the defaults unchanged and re-tested on synthetic arms. It is the version in the repository now.
<!-- src: log 2026-11-04 21:0x UTC; git show 9c3b51a:scripts/weight_edit/gate_r2_analyze.py sha256 961f315f77ca; working-tree sha256 961f315f77ca -->

### D.2 When each rule was fixed and published

Our log records each rule before its run. Git shows when each entry was committed locally, and the local reflog of `origin/main` shows when it was pushed to GitHub. Only two pushes fall inside the study: 18:40:15 UTC on 11-04 (e5aae94) and 20:55:44 UTC on 11-09 (93117d8). Table D.1 gives the times.

**Table D.1.** Registrations and their timing (UTC, 2026).

| Item | Log entry | Local commit | Reached GitHub | The run it governs |
|---|---|---|---|---|
| First round-2 freeze (acceptance primary; superseded) | 11-04 "13:2x" | 2daf233, 11-04 13:03:45 | 11-04 18:40:15 | none |
| Base-only artefact count on the test calls | 11-04 14:xx | c8fea6f, 11-04 16:54:23 | 11-04 18:40:15 | none (a disclosed look, D.3) |
| Round-2 rule, artefacts primary | 11-04 17:1x | 606ce41, 11-04 17:11:08 | 11-04 18:40:15 | first forward pass 17:17:24; Fable's first saved edit 17:45:05 |
| Astra arm, own family | 11-04 21:0x | 9c3b51a, 11-04 20:55:34 | 11-09 20:55:44 | run created 20:55:49; first pass 21:08:28 |
| Compile-guard amendment | 11-05 03:1x | ee0b34d, 11-05 03:12:55 | 11-09 20:55:44 | first edit generated on the test calls 11-05 16:04:54 |
| Informed Opus arm, own family | 11-05 16:10 | 9e9eaf4, 11-05 16:09:12 | 11-09 20:55:44 | run created 16:55:20; first pass 17:03:20 |
| Informed Fable 5.1 arm | 11-06 21:26 | b60fad2, 11-07 11:03:54 | 11-09 20:55:44 | one pass on 11-09 18:49, then stopped |
| Informed Fable 5.2 arm | 11-09 19:05 | 93117d8, 11-09 20:55:42 | 11-09 20:55:44 | conversation from 19:15:17; first pass 19:36:14 |
| Repair arm plan | 11-09 19:40 | 93117d8, 11-09 20:55:42 | 11-09 20:55:44 | conversation from 11-10 04:07:53 |
| v6 off-record gate and its expectation | 11-10 11:06 | 69c866b, 11-10 13:19:29 | not pushed | gate generation from 12:15:52 |
<!-- src: TZ=UTC git show -s for each commit; git log -S"<entry heading>" -- docs/experiments/weight-edit-collab-20261102.md for the first commit holding each entry; TZ=UTC git reflog show --date=iso-local origin/main (e5aae94 2026-11-04 18:40:15 +0000 and 93117d8 2026-11-09 20:55:44 +0000, "update by push"); git merge-base --is-ancestor (606ce41 in e5aae94; 9c3b51a, ee0b34d, 9e9eaf4, b60fad2 in 93117d8 only; 69c866b in neither); run start and pass times: round2/round2.log, round2/astra.log, proposers/<arm>/state.json passes[0].start_utc, fable_r2/kernel_log.jsonl first save_edit; transcript first timestamps for fable52inf_r2 (19:15:17) and fable52rep_r2 (04:07:55); gate start: round2/round2.log "gate start: opus_r2 fable_r2" 2026-11-05T16:04:54Z; v6: round2/v6gate/v6gate.log -->

So the round-2 rule was frozen locally before the first editor pass and published while the search was under way. The commit came at 17:11:08 UTC, the same second the launcher started the server and Opus's conversation, and six minutes before the first forward pass. By the push at 18:40:15 UTC, Fable had saved its first edit and run its first generation pass with it. The amendment and the informed Opus registration were committed before the runs they govern but reached GitHub only on 11-09. The informed Fable 5.2 registration was logged, with the hashes of its brief and runner, before that arm's first pass, but committed about 1 h 40 min after its conversation began. The live-replication and training plans are covered in Appendices H and J.
<!-- src: Table D.1 sources; log 2026-11-09 19:05 UTC (brief 993e9bca0988, runner 5c464bb8b045; both verified with shasum 2026-11-10) -->

### D.3 The disclosed look at the test set and the choice of target

We chose the target after counting artefacts in the base model's test calls. The first round-2 freeze (2daf233) kept acceptance as the primary measure. The base model's generations on the 320 test calls already existed, because they had served as the replay fidelity check and were fixed as the gate's base arm. Later on 11-04 (log entry 14:xx UTC, committed at 16:54 UTC) we counted escape artefacts in them, with no edit run: 100 of 640 generations (15.6 %; 48 greedy, 52 sampled), in 30 of the 101 tasks and 23 of the 69 repositories. Of these, 61 were accepted through the fallback and 39 were not found. The entry worked out that a pass would need about 7 tasks lower and 1 higher against every reference.
<!-- src: log 2026-11-04 13:2x UTC ("Fidelity on the fresh set ... this is also the gate's base arm") and 14:xx UTC ("unedited escape-artefact rate on the fresh gate set", all figures) -->

At 17:1x UTC the rule was revised and re-frozen with escape artefacts as the primary measure and acceptance as the key secondary. The target was narrowed after two looks, both listed in the same entry: the base-only count above, and a check of the round-1 edits' artefacts on a reused set. No edit had been run on the test calls. The editors' dossier received no test-set statistics.
<!-- src: log 2026-11-04 17:1x UTC ("Disclosed looks before this choice", "No edit has been run on the fresh set", user "go with (a) with the raw pairs, then launch"); log 2026-11-04 14:xx UTC (post-hoc round-1 check; plan change "asks for a positive weight-space result"); dossier: log 17:1x UTC "No fresh-set statistics" -->

### D.4 The compile-guard amendment and both verdicts

The registered compile guard was confounded. The model writes the new text before the old text in 727 of 730 search calls and in all 320 test calls, so the replay prefix already holds the base model's new text, and only the old text is regenerated. When an edit turns an escaped old text into an exact copy, the harness no longer takes its fallback, which would also have unescaped the new text. The exact match then inserts the frozen, still-escaped new text, and the file stops compiling, whatever the edited model would have written. On the editors' search passes, every artefact-reducing pass raised the measured compile-failure rate: Opus 5 of 5, Fable 10 of 10 and Astra 1 of 1, by 2 to 21 points.
<!-- src: log 2026-11-05 03:1x UTC ("Defect", "Evidence") -->

At 03:1x UTC on 11-05 we amended the guard. The amended guard keeps the same limit, base plus 2 points of compile failures among accepted calls, but measures it on a full-call replay. There the edited model generates the new text and then, after a teacher-forced separator, the old text, and the pair goes through the harness's replacement and `compile()`. Nothing else in the rule changed. The registered verdicts are still computed and reported. The amendment came before any candidate edit was generated on the test calls: only the base arm existed there, and the first candidate generation started at 16:04:54 UTC that day. It was motivated by the editors' search-set passes, which we could see, not by test data. All three blind editors had also described the same trap in their own notes (Appendix F.6).
<!-- src: log 2026-11-05 03:1x UTC ("Guard of record from now on", "Code", "Scheduling", "Wording for the write-up"); git ee0b34d 03:12:55 UTC; round2/round2.log gate start 16:04:54Z; fullcall/fullcall.log "fullcall base start edit=none" 03:09:06Z; scripts sha256 replay_fullcall.py e213adcd8813, replay_score_fullcall.py 250176b7e592, gate_r2_fullcall.py 649ff4106000 (verified 2026-11-10, equal to the logged prefixes) -->

Table D.2 gives both verdicts for every candidate. The amendment changes the compile guard from fail to pass for all three submitted edits, but it changes no overall verdict, because every edit also fails the working-call guard. For the off-record v6 both forms of the compile guard fail.

**Table D.2.** Guards and verdicts under the registered and the amended compile guard.

| Edit (family m) | Primary p (Holm threshold) | Acceptance, points (≥ −1) | Working calls lost per draw (≤ 2) | Compile, registered: plain replay (base 9.0 %, limit 11.0 %) | Compile, amended: full call (base 7.9 %, limit 9.9 %) | Verdict, registered / amended |
|---|---|---|---|---|---|---|
| Opus (2) | 0.011 (0.05): pass | +0.39: pass | 2, 3: fail | 12.5 %: fail | 8.4 %: pass | fail / fail |
| Fable (2) | 9.5 × 10⁻⁷ (0.025): pass | −1.01: fail | 5, 5: fail | 15.3 %: fail | 9.1 %: pass | fail / fail |
| Informed Opus (1) | 4.0 × 10⁻⁷ (0.05): pass | −4.17: fail | 11, 11: fail | 15.9 %: fail | 8.4 %: pass | fail / fail |
| v6, off the record (1) | 2.0 × 10⁻⁴ (0.05): pass | −1.27: fail | 3, 3: fail | 15.1 %: fail | 11.6 %: fail | fail / fail |

Astra and the two Fable 5.2 arms submitted empty edits and are not candidates. Because no primary passed with all guards, the key secondary was not tested for any candidate (p = 1).
<!-- src: gate_r2_report.json (opus_r2, fable_r2: primary_escape_artefact.p, primary_holm_threshold, secondary_acceptance.vs_base.task_macro_mean_diff, served_ok_loss, compile.*.compile_fail_rate_among_accepted, primary_pass); gate_r2_report_amended.json (fullcall.compile_fail_rate_among_accepted, fullcall_compile_guard_pass, amended_primary_pass); gate_r2_opusinf_report.json and _amended.json; gate_r2_fable52inf_v6_report.json; v6 _amended.json not yet written at 2026-11-10 16:38 UTC (full call at row 484 of 640) -->

### D.5 Other changes in conduct

These changed how runs were carried out, not the rule:

- Astra's fifth conversation, approved by the user after a provider capacity error, with the budget unchanged (Appendix C.2).
- Gate priority for the informed Opus arm: its server was paused while gate models ran, as its registration allowed. Its first conversation read the pause as a stall; we stopped counting conversations that ended during the pause and gave later conversations an infrastructure note.
- A schedule change, approved by the user, that ran the informed Opus arm before the record-only full-call replays of Opus and Fable, after the registered verdicts were in.
- The working-call guard was not loosened after it failed. On 11-06 the user decided to keep the rule as registered.
<!-- src: log 2026-11-05 13:55, 17:36, 21:01 UTC; log 2026-11-06 09:41 UTC ("Informed arm first"); log 2026-11-06 22:4x UTC (user: "kurali degistirmeyelim") -->

## Appendix E. Round 1, the tool-side test, and checks of the editors' explanations

### E.1 Round 1: target, editors and edits

Round 1 asked whether a weight edit raises acceptance, that is, whether the old text is found at all. This is a broad target: it mixes partial, invented and escaped copies (Table A.2). Two editors and one scripted recipe each produced one frozen edit, with 10 forward-pass requests each. A `gen` request covered at most 64 search calls × 2 draws, and a `read` request at most 32 texts. Claude Opus 5.5 (`opus_p10_1`) and Claude Fable 5.1 (`fable_p10_1`) ran as Claude Code subagents at maximum effort, with the same brief and dossier. Opus used about 524k tokens in 3.5 h, and Fable about 388k in 3.9 h. Round 1 had no sandbox: isolation was by instruction, checked afterwards from each run's logged tool calls. A first Opus launch read a file that mixed search and selection rows; it was stopped before any forward pass and voided. The recipe (`recipe_p10_1`) is our own difference-of-means procedure, specified in its docstring before its first pass. It ran through the same server and used 6 of its 10 passes, as specified.

<!-- src: log 2026-11-02 20:57 UTC (LIMITS), 21:34 UTC (void run, frozen brief/dossier hashes), 21:53 UTC (recipe pre-specification), 2026-11-03 01:05 UTC (Opus cost, audit), 03:50 UTC (Fable cost, audit), 2026-11-02 23:34 UTC (recipe 6 of 10) -->

**Table E.1.** The three round-1 edits.

| Edit | Where | Rank and size | What it does (editor's description) |
|---|---|---|---|
| Opus | layer 23 `self_attn.v_proj`, rows of KV head 0 only | rank 64; ‖ΔW‖/‖W‖ 6.5 % | ridge-fitted remap of the value vectors of 41 fused line-end escape tokens (such as `,\` or `)\`) toward those of their raw counterparts |
| Fable | `q_proj` and `k_proj` in layers 11, 17 and 23 | six rank-1 terms; ‖ΔW‖/‖W‖ 0.04–0.21 per module | gated terms that make old-text queries attend more to raw code quoted in the model's thoughts and less to the new text |
| Recipe | layer 24 `self_attn.o_proj` and `mlp.down_proj` | rank 1 each, α = 2 | removes a difference-of-means direction (first wrong copy token against correct copy tokens) from both modules' outputs |

<!-- src: outputs/weight_edit/proposers/opus_p10_1/final_answer.md ("The edit"); log 2026-11-03 01:05 UTC (Opus), 03:50 UTC (Fable E6, module hashes), 2026-11-02 23:34 UTC and outputs/weight_edit/proposers/recipe_p10_1/final_answer.md (recipe) -->

### E.2 The gate and its timing

The gate rule was written in the log at 21:23 UTC on 2026-11-02, before the first proposer run. It was amended at 21:34 and 21:41 UTC after Astra's review, before any output on the selection set existed. Under the amended rule, each edit was replayed on the 225 selection calls (35 tasks from fastapi, rich and requests; none from httpx) with two draws, giving 450 generations per arm. To pass, an edit needed all of the following:

- a positive per-task mean gain in acceptance, with a one-sided exact sign test over tasks giving p < 0.05, against the base model and against each of two random controls with the same modules, rank and per-module Frobenius norm (p is the largest of the three);
- the Holm correction across the three edits;
- at most 2 working calls lost per draw: recorded successes that the base replay accepted (126 and 127 of 128 in the two draws) and the edit's replay refused;
- compile failures among accepted calls at most 2 points above the base model's.

The analysis script `gate_analyze.py` (sha256 7cfb11454bcd) was frozen at 01:29 UTC on 2026-11-03, before any selection output was read. The log and the script entered git only at 13:02 UTC on 2026-11-04 (commit 5db4e17), after the result, so the timing rests on the log itself.

<!-- src: log 2026-11-02 21:23, 21:32 (Astra), 21:34, 21:41 UTC; 2026-11-03 01:29 UTC; outputs/weight_edit/shared/split.json and outputs/weight_edit/claude/gate/selection_keys.txt (225 keys; tasks fastapi 19, rich 13, requests 3); gate/gen_base.scored.jsonl (recorded ok 128 per draw, base accepted 126 and 127); git log: 5db4e17 2026-11-04 13:02:53 UTC first commit of the log, gate_analyze.py, wp12_analyze.py, native_check.py -->

### E.3 Result: no edit passed

No round-1 edit raised acceptance (Table E.2). All three per-task mean differences were slightly negative, no p-value came near 0.05, and Fable's edit also failed the working-call guard.

**Table E.2.** Round-1 gate on the selection set (450 generations per arm; base 264 accepted).

| Edit | Accepted | Per-task Δ acceptance, points [95 % CI] | Tasks up / down | p (largest of 3) | Working calls lost (draw 0, 1) | Accepted and compiling (base 52.7 %) |
|---|---|---|---|---|---|---|
| Opus | 264 | −0.09 [−0.43, +0.15] | 1 / 1 | 0.75 | 0, 0 | 53.1 % |
| Fable | 264 | −0.22 [−2.70, +1.67] | 4 / 4 | 0.86 | 2, **4** | 51.8 % |
| Recipe | 262 | −0.22 [−0.91, +0.31] | 1 / 2 | 0.88 | 0, 2 | 52.2 % |

Intervals are task-level bootstraps (10,000 resamples over 35 tasks). Fable's four lost calls in draw 1 were in fastapi_13920, fastapi_14605 and fastapi_14962.

<!-- src: outputs/weight_edit/claude/gate/gate_report.json (candidates.*.vs_base, p_candidate 0.75 / 0.855 / 0.875, served_ok_loss, compile.*.accepted_and_compiling_rate 0.5267 / 0.5311 / 0.5178 / 0.5222); gate/gate_effect_intervals.json ("<edit> vs base"); log 2026-11-03 09:06 and 09:11 UTC -->

Random edits of the same size moved acceptance as much as the real edits did. The six random controls accepted 263 to 267 of 450, and their per-task differences from the base model ranged from −0.14 to +0.46 points. Opus's edit touches only the 512 rows of KV head 0, while its gate controls spread over the whole matrix. Two extra controls on the same 512 rows, a sensitivity check specified before the gate result and outside the rule, accepted 264 and 265, and Opus's edit against them gave p = 0.75 both times. The Opus and recipe intervals exclude per-task gains above about +0.5 points; Fable's interval is wider, about ±2.5 points, because its edit changed more generations in both directions. Under the wording agreed at the time, this shows no demonstrated benefit for these three edits under this budget and protocol. It is not proof that weight edits cannot work, and not a test of equivalence to random edits.

<!-- src: gate/gate_report.json (rand1_vs_base, rand2_vs_base for each candidate: call_micro 265, 264, 267, 266, 263, 265; task_macro_mean_diff −0.0014 to +0.0046); gate/sensitivity_support_matched.json (supp1 264, supp2 265; p 0.75, 0.75); log 2026-11-03 09:11 UTC (interval reading; Astra's wording), 09:37 UTC -->

### E.4 Search-set results and the recipe

On their own search calls, before the gate, no edit had improved correct edits either. Opus's edit raised acceptance from 222 to 226 of 384 paired generations, but the four new acceptances were old texts that closed too early, and accepted-and-compiling calls fell from 201 to 198. Fable's edit gained 2 and lost 2 calls on each of two 64-call sets and left all 108 accepted exact copies unchanged. The recipe gained 1 of 64.

<!-- src: log 2026-11-03 01:05 UTC (Opus), 03:50 UTC (Fable), 2026-11-02 23:34 UTC (recipe); Astra 2026-11-03 02:41 UTC on reading Opus's 222 → 226 -->

The recipe shows that a direction can predict the failure without causing it. A difference-of-means direction separated the first wrong token of a copy from correct copy tokens with a held-out AUC of 0.88 at layer 24 (Table E.3). Removing it from that layer's two writing modules changed at most one of 64 generations, and on the gate the edit accepted 262 against 264. The removal takes the direction out of those two modules' outputs only; other layers still write it into the residual stream. Here, decodable did not mean causal.

**Table E.3.** Recipe: held-out AUC of the direction by layer (read 1 fit, read 2 scored) and paired generations on 64 search calls (base 37 accepted).

| Layer | 6 | 12 | 18 | 24 | 30 | 36 | 40 | 41 |
|---|---|---|---|---|---|---|---|---|
| AUC | 0.79 | 0.86 | 0.80 | **0.88** | 0.86 | 0.80 | 0.72 | 0.73 |

| Candidate | Only the edit accepted | Only the base accepted |
|---|---|---|
| layer 24, α 1 | 0 | 0 |
| layer 24, α 2 (frozen) | 1 | 0 |
| layer 30, α 2 | 0 | 0 |
| layers 24 + 30, α 2 | 1 | 0 |

<!-- src: outputs/weight_edit/proposers/recipe_p10_1/final_answer.md (AUC per layer; per-pass paired counts); log 2026-11-02 23:34 UTC; gate/gate_report.json recipe_p10_1 call_micro 262 vs 264 -->

Round 1 also left escape artefacts almost unchanged. In a post-hoc look at the gate generations, the base model wrote 42 in 450, in 11 of the 35 tasks. The three edits wrote 41, 39 and 40, and the controls 42 to 44, with no difference near significance (task-level p 0.25 to 0.50). This look, together with a disclosed base-only count on the held-out calls (100 of 640), led to the narrower round-2 target.

<!-- src: log 2026-11-04 14:xx UTC (both entries; exploratory, motivation only) -->

### E.5 The tool-side test, planned in advance

Changing how tool results are shown raised acceptance on the same calls where the weight edits gained nothing. The hypothesis, H1: with the model's prefix fixed, showing every tool response in the template's native form raises old-text acceptance compared with the served JSON-string form. It was written in the log at 13:10 UTC on 2026-11-03, before any generation for this test. Astra reviewed it at 13:11 UTC, and its corrections were adopted at 17:27 UTC, again before any generation. The main changes were completeness rules that block the analysis on missing or duplicate records, and the description of the change as a bundle. The frozen hashes were `wp12_analyze.py` 5c8dff0ba329, `wp12_render.py` 734225b90967, `replay_gen.py` e090d2500559 and `replay_score.py` ce5df47c4b56. Like the round-1 gate, the plan was timestamped only in the log until the 2026-11-04 commit.

<!-- src: log 2026-11-03 13:10 UTC (pre-registration), 13:11 UTC (Astra), 17:27 UTC (corrections, frozen hashes); git commit 5db4e17 (2026-11-04 13:02:53 UTC) -->

**Design.** `wp12_render.py` rewrote every tool response in every replay prompt (10,468 blocks in 1,191 prompts) from the served `value:<|"|>JSON<|"|>` form into the template's dictionary branch, where each field is shown raw (Appendix A.3). Prompts became shorter, with a median of 0.971 times the served length (range 0.915 to 0.999). Nothing else changed; in particular, the prefix the model had written under the escaped view was kept. The data were the 225 round-1 selection calls × 2 draws, with the same seeds and key order as the round-1 gate. The served arm was regenerated and was identical to the gate's base arm in 450 of 450 generations. The test ran on the bf16 model in MLX on the Mac, from 17:30 UTC on 2026-11-03. The single test was a positive per-task mean difference with a one-sided exact sign test over tasks, p < 0.05.

<!-- src: log 2026-11-03 13:10 UTC (Intervention, Data and arms, Primary test, Schedule), 19:0x UTC (completeness, identity with gate base 450/450) -->

**Table E.4.** The tool-side test.

| Set | Per-task Δ acceptance, points [95 % CI] | Tasks up / down | p | Accepted |
|---|---|---|---|---|
| Selection, 35 tasks (the test) | **+1.98 [+0.33, +4.42]** | 5 / 0 | **0.031** | 264 → 273 of 450 |
| Search, 83 tasks (exploratory; the hypothesis was formed here) | +0.78 [−0.32, +2.33] | 6 / 3 | 0.25 | 607 → 611 of 1,010 |

<!-- src: outputs/weight_edit/claude/wp12/wp12_primary_report.json (primary.task_macro_mean_diff 0.01981, ci95 [0.00335, 0.04417], tasks 5/0/30, p 0.03125, h1_supported true; served/raw accepted 264/273); wp12_search_exploratory_report.json (0.00779, [−0.00324, 0.02330], 6/3/74, p 0.254; 607/611) -->

H1 was supported, but the effect is small and sits in five tasks. All 9 changed call-draw outcomes favoured the native view; they were 6 distinct calls in 5 tasks. Accepted-and-compiling calls rose from 237 to 245 (52.7 % to 54.4 %), and exact copies from 271 to 279. Among recorded failures, accepted call-draws rose from 11 to 17 of 136. Escape artefacts barely moved, from 42 to 39. A plausible reason is that the fixed prefix (thought and new text) was written under the escaped view, but this test does not establish it.

<!-- src: wp12/wp12_primary_report.json (accepted_and_compiling 237/245, exact_copy 271/279, escape_artefact 42/39, by_recorded_kind.not_found accepted 11/17 of 136); log 2026-11-03 19:0x UTC (9 flips, 6 calls, 5 tasks) -->

The test has a narrow scope:

- The change is a bundle: escaping, field layout and about 3 % shorter prompts change together. It does not isolate escaping.
- The selection set had already been used for the round-1 gate, so this is a separately planned follow-up on a reused set, not a fresh hold-out.
- One task moving the other way would have given p = 0.11.
- The difference between the rendering change and the weight edits was not tested directly. All three edits' upper bounds (+0.15, +1.67 and +0.31 points) lie below the +1.98-point gain.
- The test measures one call at a fixed prefix, not the whole agent. The competition's serving path cannot be changed by a submission, so the result informs the paper, not a submission setting.

<!-- src: log 2026-11-03 13:10 UTC ("Where the escaping comes from"), 17:27 UTC (bundle, reuse), 19:0x UTC; one-sided sign test P(X ≥ 5 | n = 6) = 7/64 = 0.109; gate/gate_effect_intervals.json (upper bounds) -->

### E.6 Checks of the editors' explanations

Each round-1 editor explained how its edit worked, and we checked these explanations with an independent forward pass. Opus said its edit narrows the model's preference for escape tokens at line ends. Fable said the model copies the old text from raw text it wrote itself (the new text, and code quoted in its thoughts), and that its edit moves attention toward the thought quotes in a little-used direction. The two accounts are not independent: both editors read a dossier that stated the JSON rendering and the new-text-first order.

<!-- src: log 2026-11-03 01:05 UTC (Opus mechanism), 03:50 UTC (Fable mechanism), 03:51 UTC (Astra; revised synthesis: not independent confirmation); draft-v3 §5 -->

The editors' own measurements used the server's "manual" forward, a hand-written pass that returns activations. The checks used the model's native forward (mlx_lm with a KV cache), with criteria written before each run. The two forwards disagree by up to 0.60 nats on the base model alone (Opus's texts; 0.47 on Fable's), with top-1 agreement of at least 0.93. The stored log-probs of the editors' passes equal those recomputed under the final edits, so the passes checked are the ones that used the final edits.

<!-- src: outputs/weight_edit/claude/native_check/opus_p10_1_fwd9.json and fable_p10_1_fwd6.json (aggregate.max_target_diff_base 0.598 / 0.467; max_stored_vs_recomputed 0; per_text top1_agree min 0.978 / 0.929); log 2026-11-03 18:14 UTC (criteria), 23:10 UTC (results) -->

**Table E.5.** The editors' explanations and what the independent checks found.

| Claim (editor's own figure, search set) | Check and criterion | Result |
|---|---|---|
| Opus: line-end escape margins shrink by about 2.5–6 nats | 9 line-end switch points of pass 9; native median shrinkage in [2.5, 6] and native mean within 25 % of manual (rule fixed 2026-11-07 11:01 UTC) | **Reproduced as stated**: median 2.75 nats native (2.76 manual), mean 2.61 (2.68), range 0–6.1 |
| Opus: the edit flips the choice at `(` | native top-1 at that point, base and edit | **Not a flip**: the base model already ranked `(` first; no greedy choice changes at any of the 9 points |
| Opus: quote escapes unchanged; control texts move ≤ 0.07 nats | 9 quote points; 7 control texts | Quote margins unchanged (−0.25 to +0.31); controls move up to 0.44 nats natively (0.10 manual) |
| Fable: the true token gains about 1.4 nats at divergences, no flips | 28 texts of pass 6; per-position agreement within 0.5 nats and mean within 25 % (rule fixed 2026-11-03 18:14 UTC) | **Mean reproduced**: +1.34 manual, +1.26 native; the per-position bar fails in 3 of 28 texts (largest gap 0.70 nats) |
| Fable: the edit uses an unused, unrotated head direction | 8 search calls Fable never used; four criteria fixed before the run | **Partly**: no energy in RoPE-rotated dimensions; but the direction ranks 70–150 of 384 by variance (0.49–0.75 × the median), not the least used; ordinary queries' attention in the edited heads moves (mean total variation 0.13–0.21) |
| Recipe: the failure is decodable mid-depth | held-out AUC | Decodable (0.88), but removal changed nothing (E.4) |

<!-- src: native_check/opus_p10_1_margin.json (aggregate.line_end_dm nat median 2.75, mean 2.611, min 0, max 6.0625; man median 2.755, mean 2.681; verdict "reproduced as stated"; paren_flip_native; quote_dm nat min −0.25 max 0.3125; ctrl_max_abs_dsum_native 0.4393, manual 0.1040); log 2026-11-07 11:01 and 11:23 UTC (script native_margin_check.py sha256 0f7c116680ff, committed in b60fad2 at 11:03:54 UTC before the full run); native_check/fable_p10_1_fwd6.json (mean_shift_at_div manual 1.340, native 1.255; max_shift_diff 0.698; 3 texts over 0.5 by recount); native_check/audit_u_fable.json (C1 0.0 in all edited heads; C2 rank_of_u_among_eigs 70–150, ratio_to_median 0.49–0.75, ratio_to_min 1.70–3.19; C3 CV 0; C4 natural edited-head tv_mean per layer 0.21 / 0.13 / 0.21 by recount); log 2026-11-03 23:10 UTC; ledger check docs/paper/review/ledger-v3-check-20261107.md D3, D4 -->

The Opus row needed a second, targeted check. The first native check, on 2026-11-03, measured the shift of the target token at each text's lowest-probability position, not at Opus's line-end switch points. By its stated rule it did not reproduce (mean shift +0.04 nats manual against +0.10 native). Because that measured a different quantity from the claim, we re-measured the claim itself at its own 9 positions on 2026-11-07, under a rule written before the run (Table E.5). Both results are kept.

<!-- src: log 2026-11-03 23:10 UTC (first check, "not reproduced by the stated criterion"), 2026-11-07 11:01 UTC ("Why"), 11:23 UTC; native_check/opus_p10_1_fwd9.json mean_shift_at_div manual 0.0416, native 0.0958 -->

So two explanations held up and a third held in part: Opus's margin claim (in full), Fable's mean shift (in the mean, not at every position) and Fable's direction claim (in part). The audit of Fable's direction gives more detail. In the edited heads, old-text queries move strongly, with a mean total variation of 0.44–0.74, so the effect is selective. But ordinary prompt queries in the same heads also move: their mean total variation is 0.13–0.21 and the 99th percentile of the score change is 2.1–5.1. About a sixth of the attention mass of those heads shifts on ordinary text. This fits Fable's loss of working calls at the gate, though that link was not tested.

<!-- src: native_check/audit_u_fable.json C4 (recount over 8 calls: old_string tv_mean 0.74 / 0.44 / 0.72; natural tv_mean 0.21 / 0.13 / 0.21, abs_dscore_p99 5.1 / 2.1 / 4.3); log 2026-11-03 23:10 UTC; Astra 2026-11-04 12:46 UTC (link to served-ok losses plausible, not demonstrated) -->

The confirmed effects were far too small to change what the model wrote. At Opus's points, seven of the nine base margins were 6.5 to 12.5 nats in favour of the escape token. One was 2.5 nats and ended at 0.75, and at the last the base model already preferred the right token. The escape token stayed first at all eight points where it had been first. At Fable's divergences, the true token sat at log-probabilities of −7 to −25 while the copied new-text token was near 0 (Fable's own pass-1 figure, manual forward, imprecise below about −10). A mean gain of 1.3 nats cannot move the top choice. Even the largest native shift in the check, +7.65 nats, left the true token at about −2.4. These checks verify parts of what the editors said; they do not show that either explanation is the cause of the failure.

<!-- src: native_check/opus_p10_1_margin.json per_text nat margins base → edit: 9.06→3.00, −1.00→−2.62, 12.50→9.50, 2.50→0.75, 11.81→9.56, 9.94→7.06, 8.62→5.88, 11.50→8.31, 6.50→6.50; outputs/weight_edit/proposers/fable_p10_1/final_answer.md lines 25-36 (−7..−25), dossier note on manual-forward precision (log 2026-11-02 21:34 UTC); native_check/fable_p10_1_fwd6.json per_text ti 11 (base −10.0 native, shift +7.65); Fable's final answer pass 6 ("the raw-quote case reaches −2.4") -->

## Appendix F. Full round-2 results

All figures in this appendix come from the registered gate reports and the frozen generation files on the 320 test calls, two draws each (640 generations per arm). The base model has 100 escape artefacts and 443 accepted generations. The v6 rows are descriptive and off the record (F.6).

### F.1 Effects with task and repository-clustered intervals

**Table F.1.** Effects against the base model. Changes are task-macro, in points; intervals are 95 % bootstrap intervals over tasks and over the 69 repositories.

| Edit | Artefacts of 640 | Tasks lower / higher | Artefact change [task CI] [repository CI] | Accepted of 640 | Acceptance change [task CI] [repository CI] |
|---|---|---|---|---|---|
| Opus | 77 | 9 / 0 | −4.0 [−7.2, −1.4] [−7.7, −1.2] | 449 | +0.4 [−0.7, +1.6] [−0.6, +1.7] |
| Fable | 47 | 20 / 0 | −9.6 [−14.4, −5.3] [−15.4, −5.1] | 445 | −1.0 [−3.9, +1.4] [−4.0, +1.3] |
| Informed Opus | 25 | 25 / 1 | −14.3 [−20.6, −8.4] [−22.0, −8.1] | 427 | −4.2 [−7.7, −1.2] [−7.9, −1.4] |
| v6 (off the record) | 43 | 17 / 1 | −10.4 [−16.1, −5.3] [−17.5, −4.7] | 444 | −1.3 [−4.0, +0.8] [−4.1, +0.7] |
<!-- src: gate_r2_report.json, gate_r2_opusinf_report.json, gate_r2_fable52inf_v6_report.json: candidates.<arm>.primary_escape_artefact.vs_base (call_micro, tasks_better, tasks_worse, task_macro_mean_diff), secondary_acceptance.vs_base, reported.bootstrap_vs_base (task_ci95, repo_clustered_ci95; 10,000 resamples, seed 20261104) -->

**Table F.2.** Primary comparisons per reference (one-sided exact sign test; tasks lower / higher), and the split by source.

| Edit | Against base | Against control 1 | Against control 2 | OOD-43: change, tasks lower / higher | OOD-B: change, tasks lower / higher |
|---|---|---|---|---|---|
| Opus | 9 / 0, p 0.0020 | 9 / 1, p 0.011 | 8 / 0, p 0.0039 | −4.9, 5 / 0 | −3.3, 4 / 0 |
| Fable | 20 / 0, p 9.5 × 10⁻⁷ | 20 / 0, p 9.5 × 10⁻⁷ | 20 / 0, p 9.5 × 10⁻⁷ | −8.0, 9 / 0 | −10.8, 11 / 0 |
| Informed Opus | 25 / 1, p 4.0 × 10⁻⁷ | 27 / 0, p 7.5 × 10⁻⁹ | 26 / 1, p 2.1 × 10⁻⁷ | −11.4, 12 / 1 | −16.4, 13 / 0 |
| v6 (off the record) | 17 / 1, p 7.2 × 10⁻⁵ | 18 / 2, p 2.0 × 10⁻⁴ | 18 / 1, p 3.8 × 10⁻⁵ | −8.8, 7 / 1 | −11.6, 10 / 0 |
<!-- src: same reports, primary_escape_artefact.vs_base / vs_rand1 / vs_rand2 and reported.by_source.<OOD-43|OOD-B>.escape_artefact -->

By recorded outcome, every edit gained acceptance on calls that had failed in the recorded run and lost it on calls that had succeeded. On the 206 recorded not-found generations, the base accepted 15; Opus accepted 26, Fable 27, informed Opus 21 and v6 22. On the 426 recorded-ok generations, the base accepted 426; Opus accepted 421, Fable 416, informed Opus 404 and v6 420.
<!-- src: reports, reported.by_recorded_kind.not_found.accepted and .ok.accepted ([edit, base]) -->

### F.2 Random controls

The random controls left artefacts and acceptance at the base level and lost almost no working calls (Table F.3). So the losses in Table D.2 come from what the edits do, not from perturbing those matrices by that amount.

**Table F.3.** Matched random controls (same modules, rank and Frobenius norm; random directions).

| Controls of | Artefacts of 640 (control 1, 2) | Accepted of 640 | Working calls lost per draw |
|---|---|---|---|
| Opus | 99, 99 | 443, 443 | 0, 0 and 0, 0 |
| Fable | 100, 100 | 443, 442 | 0, 0 and 0, 0 |
| Informed Opus | 101, 102 | 443, 441 | 0, 0 and 0, 0 |
| v6 (off the record) | 99, 101 | 440, 442 | 1, 1 and 1, 1 |
<!-- src: reports, primary_escape_artefact.rand1_vs_base / rand2_vs_base call_micro and secondary_acceptance.rand*_vs_base call_micro; working calls: appF_checks.py section 3 (recorded ok, base-accepted, control rejects, per draw), matching log 2026-11-06 22:4x UTC for the Opus and Fable controls -->

### F.3 What happened to the escaped copies the fallback had rescued

The base model wrote 84 old texts that the harness accepted only through its fallback. Table F.4 shows what each edit made of them. A control is shown for scale.

**Table F.4.** Outcome transitions, base to edit, over 640 generations.

| Arm | Fallback-rescued (84): now exact | now other accepted | still escaped and rescued | now not found | Base not-found (191) now accepted |
|---|---|---|---|---|---|
| Opus | 18 | 1 | 60 | 5 | 11 |
| Fable | 31 | 2 | 39 | 12 | 14 |
| Informed Opus | 39 | 4 | 17 | 24 | 8 |
| v6 (off the record) | 35 | 2 | 40 | 7 | 8 |
| Informed Opus, control 1 | 1 | 1 | 82 | 0 | 0 |
<!-- src: log 2026-11-07 01:00 UTC transition table (Opus, Fable, informed Opus, control); recomputed with appF_checks.py section 1 from gen_<arm>.scored.jsonl "outcome", identical; v6 row new, same script. Range over all eight controls: 0 to 3 left the fallback path, 0 to 2 lost -->

The three submitted edits moved 24, 45 and 67 of these generations off the fallback path, and 5, 12 and 24 of those were then not found. The stronger the edit, the more it moved and the more it lost. The lost generations are not all fully raw. In 9 of Fable's 10 working-call losses, the edit removed most escapes but left one to three, so the mixed string matched neither exactly nor after unescaping. The informed Opus losses include dropped characters (`)))` to `))`), changed brackets and mangled regular expressions.
<!-- src: Table F.4 (exact + other accepted + not found = 24, 45, 67); log 2026-11-06 22:4x UTC ("How Fable's 10 served-ok losses arise"); log 2026-11-07 01:00 UTC ("Reading, descriptive" examples) -->

### F.4 Full-call figures

When the edited model also writes the new text, the edits remove artefacts in both arguments, and accepted-and-compiling calls stay close to the base (Table F.5). These figures were read after the registered verdicts and do not change them. The working-call guard is a per-call rule that the full call does not replace.

**Table F.5.** Full-call replay, 640 generations per arm.

| Arm | Accepted | Compile failures among accepted | Accepted and compiling | Paired: only base / only edit | Old-text artefacts | New-text artefacts | Exact copy | Compile failures with the frozen new text |
|---|---|---|---|---|---|---|---|---|
| Base | 69.4 % | 7.9 % | 407 | | 13.9 % | 10.5 % | 55.6 % | 17.8 % |
| Opus | 69.2 % | 8.4 % | 404 | 12 / 9 | 10.2 % | 7.3 % | 59.5 % | 19.0 % |
| Fable | 70.5 % | 9.1 % | 408 | 9 / 10 | 5.3 % | 2.7 % | 62.7 % | 23.5 % |
| Informed Opus | 68.9 % | 8.4 % | 402 | 20 / 15 | 3.6 % | 2.3 % | 64.7 % | 22.7 % |
| v6 (off the record) | 70.3 % | 11.6 % | 396 | 17 / 6 | 6.9 % | 5.8 % | 61.3 % | 22.4 % |
<!-- src: gate_r2_report_amended.json (base, opus_r2, fable_r2: fullcall.accepted, compile_fail_rate_among_accepted, accepted_and_compiling, paired_accepted_and_compiling, artefact_old, artefact_new, exact_copy, compile_fail_rate_among_accepted_frozen_new); gate_r2_opusinf_report_amended.json; log 2026-11-06 22:04 UTC and 2026-11-07 02:47 UTC -->

The base model's old-text artefact rate is lower here (13.9 %) than in the plain replay (100 of 640, 15.6 %), because the base model also regenerates its new text, and that changes what follows. Acceptance in the full call is close to the base for all three edits, while the plain replay of record gave the informed edit 16 fewer accepted calls.
<!-- src: gate_r2_report_amended.json base.artefact_old 0.1390625; gate_r2_report.json base_escape_artefacts 100; log 2026-11-07 02:47 UTC ("−3 calls" full call, "−16 calls" plain replay) -->

### F.5 Where the effects differ: escaped new text

Opus and Fable made different design choices, and the gap between their effects sits where the recorded new text is escaped. Opus gated its edit with a "safe to clean" flag and acted only where the new text was clean, because on its search data cleaning the old text under an escaped new text broke files. Fable acted at every escape decision. We split the test generations by whether the recorded new text holds a literal `\n` or `\"` that the file does not (Table F.6).

**Table F.6.** Artefacts and acceptance changes by whether the recorded new text is escaped.

| Arm | Artefacts, escaped new text (76 generations) | Artefacts, clean new text (564) | Acceptance gains: clean / escaped | Acceptance losses: clean / escaped |
|---|---|---|---|---|
| Base | 71 | 29 | | |
| Opus | 58 | 19 | 10 / 1 | 4 / 1 |
| Fable | 30 | 17 | 13 / 1 | 6 / 6 |
| Informed Opus | 14 | 11 | 8 / 0 | 13 / 11 |
| v6 (off the record) | 32 | 11 | 7 / 1 | 1 / 6 |
<!-- src: log 2026-11-06 23:2x UTC (Opus and Fable rows: 76 / 564, 71 / 29, 58 / 19, 30 / 17; gains Opus 10 of 11, Fable 13 of 14); recomputed with appF_checks.py section 2, identical; informed Opus and v6 rows new, same script; losses per group also new -->

On clean new text, the edits removed 10, 12, 18 and 18 artefacts. On escaped new text, Opus removed 13, Fable 41, informed Opus 57 and v6 39. Almost all acceptance gains came on clean new text. On escaped new text the fallback already accepts most fully escaped copies, so removing their artefacts changes the artefact count far more than it changes which edits apply. The primary measure rewards a wider scope; acceptance barely does.
<!-- src: Table F.6 differences; log 2026-11-06 23:2x UTC ("Reading") -->

### F.6 The editors' decisions and the off-record gate of v6

**Informed Opus** submitted its edit while naming the working-call guard as its main risk. On its own search data, three natural sets of 768 generations from 108 tasks, artefacts fell from 73 to 13 (22 tasks lower, none higher) and accepted generations went from 458 to 457. But it lost more than 2 calls in a draw on two of the three sets (2 and 3 on set B, 3 and 5 on set D). It wrote that guard 2 was "more likely to fail than pass on a set of 200 or more calls". It also noted that set B was not clean, because it had tuned on it, and that random controls at its edit's size were untested. On the test calls its edit lost 11 working calls in each draw and 4.2 points of acceptance. Its artefact effect transferred (82 % fewer on its search sets, 75 % on the test calls); its guard figures did not.
<!-- src: opusinf_r2/final_answer.md (opening "Short verdict"; §3 table; §4 "Guard 2 is the main risk", "Set B is not clean", "Random references are untested at these norms"); gate_r2_opusinf_report.json; log 2026-11-07 01:00 UTC ("Generalisation, descriptive") and 2026-11-10 15:23 UTC -->

**Astra** submitted no edit. On its discovery tasks a rank-13 edit at half strength cut artefacts from 128 to 102 of 256, but on its 47 set-aside validation tasks it changed artefacts from 106 to 105, with one task lower and none higher (p = 0.5 against the base and both of its own random controls). Astra also found the compile trap on its own. Its notebook records, before its first pass had finished (21:13 UTC on 11-04), 43 recorded accepted artefact calls whose corrected old text would match exactly while leaving a new text that does not compile. After its pass 4 it attributed 34 of the 36 extra compile failures to this group. This was about six hours before our amendment, which also cited Astra's pass-4 figures. Opus (its pass 5) and Fable (its final answer, "trap cases") described the same mechanism on their own search data.
<!-- src: astra_r2/final_answer.md (discovery and validation tables; compile paragraph "34 of the 36"); astra_r2/notebook.md lines 23-27 and 59 (finding recorded before "Pass 1 completed"); astra_r2/state.json pass 1 end 2026-11-04T21:13:27Z; git ee0b34d 2026-11-05 03:12:55 UTC; log 2026-11-05 03:1x UTC (cites Astra pass 4); opus_r2/final_answer.md §2.3; fable_r2/final_answer.md §3 "Compile guard" -->

**Informed Fable 5.2** withdrew its best candidate, v6. On fresh search calls that no edit had been run on before (gen128b), v6 cut artefacts from 116 to 39 (21 tasks lower, none higher), but it lost 7 base-accepted calls per draw and gained 7, and full-call compile failures among accepted calls rose by 7.2 points. It corrupted legitimate escapes, such as `\n\n` in new code in 2 of 6 calls and a backslash in a raw regular expression. Re-weighted to the natural mix, its own estimates, with 90 % intervals over tasks, were −0.3 points of acceptance (−1.5 to +1.0), 1.6 to 1.75 lost calls per 128 calls (0.55 to 2.9), and +2.8 to +3.0 compile points (+0.7 to +5.7). It removed the tensors and kept v6 with hashes. A drop rule it had written before its pass 10 was not triggered; it withdrew on the broken legitimate escapes, which that rule had not foreseen.
<!-- src: fable52inf_r2/final_answer.md ("Submission and bottom line", table and "Why I withdrew it"; departure from its pre-pass-10 rule); log 2026-11-10 04:10 UTC -->

**Repair Fable 5.2** withdrew too. On its set of 128 calls from tasks the adapter had not trained on, the starting adapter G-tam-39 cut artefacts from 77 to 9 of 256 (21 tasks lower, none higher), but it made the model stop the quoted span early: 49 old texts were cut short, 3 working calls were lost per draw, and full-call compile failures rose by 5.1 points. Four repair families failed: projecting the closing-quote and newline read-outs out of the late outputs; the adapter at half strength with Fable's line-end term and a rank-1 penalty on the closing quote; the adapter at 0.15 strength with that line-end term; and the adapter scaled to 0.35 or 0.20. In the full-call replay every candidate left fewer accepted-and-compiling edits than the base model's 137 of 256. Its stated reason is the fallback: in its set, all 50 fallback-accepted artefact calls compile, so removing the artefact can only lose there.
<!-- src: fable52rep_r2/final_answer.md §1 (accepted-and-compiling table: base 137, x0.20 133, L15 131, x1.0 130, x0.35 130, v5 125, H1 111) and §2 (candidate table: G1 9 artefacts, 21 / 0, 3, 3 lost, 49 cut short, +5.1 compile); log 2026-11-10 12:20 UTC -->

**The off-record gate of v6.** At the user's request we ran the withdrawn v6 through the same gate afterwards. It is descriptive and off the record: the informed Fable 5.2 arm's registered outcome stays "empty edit, not a candidate". We wrote an expectation in the log at 11:06 UTC on 11-10, before any v6 generation: the primary would probably pass, and the working-call and compile guards would probably fail, as the editor had estimated. That entry was committed at 13:19 UTC, while the gate was running.
<!-- src: log 2026-11-10 11:06 UTC (off-record, adaptive, expectation, runner run_v6gate.sh sha256 8a8219cfe444 verified, SEED_BASE 20262000, v6 hashes d9749f8f, da879d41, a9cb9e7d verified against proposers/fable52inf_v6/edit and gate_r2/fable52inf_v6/FROZEN.txt); git 69c866b 2026-11-10 13:19:29 UTC; round2/v6gate/v6gate.log (gate start 12:15:52Z, done 15:19:22Z) -->

**Table F.7.** v6 on the test calls, off the record, beside the blind Fable edit.

| | Artefacts of 640 | Tasks lower / higher | Acceptance, points | Working calls lost per draw | Compile, plain replay | Compile, full call |
|---|---|---|---|---|---|---|
| Fable (`fable_r2`) | 47 | 20 / 0 | −1.01 | 5, 5 | 15.3 % | 9.1 % |
| v6 | 43 | 17 / 1 | −1.27 | 3, 3 | 15.1 % | 11.6 % |

The primary passed (p = 2.0 × 10⁻⁴, the largest of three). Acceptance fell 1.27 points against a limit of 1, with the call count level (444 against 443): v6 gained 7 accepts on recorded not-found calls and lost 6 on recorded-ok calls. It lost 3 working calls in each draw, in the same three tasks in both draws. Unlike the three submitted edits, v6 also failed the full-call compile guard: 11.6 % of accepted calls failed to compile, against the base's 7.9 % and a limit of 9.9 %. The editor's estimates bracket the gate: 0.55 to 2.9 lost calls per 128 calls scale to about 1.4 to 7.3 per 320, and the gate gave 3; its +0.7 to +5.7 compile points bracket the +3.7 measured. Ten informed passes by Fable 5.2 ended about where the blind Fable edit did, with two fewer calls lost per draw.
<!-- src: gate_r2_fable52inf_v6_report.json (call_micro 43 / 100; p 0.000201; secondary_acceptance.vs_base task_macro_mean_diff −0.0127, call_micro 444 / 443; by_recorded_kind not_found 22 / 15, ok 420 / 426; served_ok_loss draw0 and draw1 3, tasks nwb2bids_347_interface, sqlfluff_7622, tctrack_119; compile.edit 0.1509); gate_r2_report.json (fable_r2); gate_r2_report_amended.json (fable_r2 0.0909); log 2026-11-10 15:23 UTC; v6 full call: gate_r2/gate_r2_fable52inf_v6_report_amended.json (fullcall.compile_fail_rate_among_accepted 0.1156, base 0.0788; fullcall_compile_guard_pass false), written about 17:45 UTC on 2026-11-10 -->

## Appendix G. 4-bit serving in vLLM and the competition stack

The competition serves models as 4-bit weights with 16-bit activations (W4A16) in vLLM and loads edits as LoRA adapters. Every replay result in the main text used the 16-bit weights on the Mac. This appendix checks that Fable's edit keeps its effect, and its cost, when served the competition's way, and that the competition's own stack loads it.

**Adapters.** We exported Fable's edit and its first random control (`fable_r2_rand1`, same module, rank and size) with `weightprog.edits.export_peft`, with `lora_alpha` equal to the rank so that the adapter's change is exactly B·A. The exported change matches the edit exactly in float32; rounding it to bf16 changes its Frobenius norm by 0.23 %. Both adapters are rank 2 on layer 41's feed-forward output (`mlp.down_proj`).
<!-- src: log 2026-11-07 02:59 UTC "Stage 0, done"; adapter sha256 0d5688d730a30191… (edit) and bfcdbdb95e94922b… (random), corrected in log 2026-11-07 10:20 UTC "Inputs on the VM"; outputs/weight_edit/claude/wp7/adapter_fable_r2/, adapter_fable_r2_rand1/ -->

**Replay setup.** We replayed the 320 test calls, two draws each, through the vLLM offline engine on the W4A16 model, with one LoRA request per arm: base, Fable's edit and the random adapter. The input was the same prompt and prefix as the 16-bit replay; every request's prompt token count matched the 16-bit base arm's. Draw 0 was greedy and draw 1 used temperature 0.6, top-p 0.95 and top-k 64 with the replay's seeds. Generation stopped at the closing quote token, with at most min(1024, 2 × reference length + 64) new tokens. The generations were scored on the Mac with the replay's own scorer. The run used Colab session `wp7a` (one L4), vLLM 0.19.1 and `replay_vllm.py` at commit 8ab880b, from 10:16 to 10:48 UTC on 2026-11-07 (engine start 162 s; arms 546, 609 and 601 s).
<!-- src: docs/plans/wp7-agent-test-20261107.md "Stage 0" item 5 and "Stage 1"; log 2026-11-07 02:59 UTC (dry run: prompt counts equal the MLX base arm's prompt_tokens); log 2026-11-07 10:52 UTC (run record) -->

**Go rule, fixed before the run.** The plan, committed at 1c8d19f (09:38 UTC on 2026-11-07, before the first attempt at 09:57 UTC), set two conditions for going on to live agent runs: the edit's artefacts at most 0.75 times the 4-bit base, and the random adapter within ±10 % of the base. The first attempt ran out of GPU memory before any generation. One amendment (`max_num_seqs` 32, `gpu_memory_utilization` 0.85) changed engine settings only, and the run restarted at 10:16 UTC. The edit's ratio was 0.451 and the random adapter's 0.980, so the rule said go.
<!-- src: results/wp7a/wp7_w4a16/stage1_report.json go_rule {edit_over_base 0.4510, rand_over_base 0.9804, go true}; docs/plans/wp7-agent-test-20261107.md "Amendments" 1; git 1c8d19f 2026-11-07 09:38:37 UTC; log 2026-11-07 10:20 UTC "Stage 1, first try" -->

**Table G.1.** Old-text replay of the 320 test calls (640 generations per arm), 4-bit served against 16-bit. "Tasks lower / higher" compares per-task artefact rates with the base of the same precision (one-sided sign test). Accepted-and-compiling counts come from this old-text replay with the recorded new text; the full-call compile guard was not run on 4-bit.

| Arm | Escape artefacts ↓ | Ratio to base | Tasks lower / higher (p) | Accepted | Acceptance change, points | Working calls lost (draw 0, draw 1) | Accepted and compiling |
|---|---|---|---|---|---|---|---|
| 4-bit base | 102 | | | 439 | | | 394 |
| 4-bit, Fable's edit | 46 | 0.45 | 22 / 0 (2.4 × 10⁻⁷) | 433 | −2.8 | 7, 11 | 369 |
| 4-bit, random adapter | 100 | 0.98 | 2 / 0 (0.25) | 438 | −0.2 | 0, 0 | 393 |
| 16-bit base | 100 | | | 443 | | | 401 |
| 16-bit, Fable's edit | 47 | 0.47 | 20 / 0 (9.5 × 10⁻⁷) | 445 | −1.0 | 5, 5 | 375 |
| 16-bit, random control | 100 | 1.00 | 0 / 0 (1.0) | 443 | 0.0 | 0, 0 | 401 |

<!-- src: results/wp7a/wp7_w4a16/stage1_report.json keys w4a16.{base,edit,rand} and bf16.{base,edit,rand}: artefacts, accepted, accepted_and_compiling, served_ok_lost_per_draw, primary_vs_base.{tasks_better,tasks_worse,p_sign_one_sided}, acceptance_vs_base.task_macro_mean_diff (−0.0283, −0.0025, −0.0101, 0.0) -->

The effect survived 4-bit serving, and so did its cost. Artefacts fell to 0.45 times the base, against 0.47 in 16-bit. The edit lost 7 and 11 working calls per draw, against 5 and 5 in 16-bit, and its acceptance fell 2.8 points, against 1.0. The random adapter changed almost nothing. This step had only the go rule; we did not judge the 4-bit replay by the round-2 rule. The 4-bit model is not the 16-bit model: greedy draws gave the same text as 16-bit in 289 of 320 calls for the base, 283 for the edit and 292 for the random adapter.
<!-- src: stage1_report.json greedy_same_text_w4a16_vs_bf16 {base [289,320], edit [283,320], rand [292,320]}; log 2026-11-07 10:52 UTC "Serving gap", "Costs of the edit are larger when served" -->

**Rehearsal in the competition stack.** vLLM 0.19.1 accepts `max_lora_rank` only from {1, 8, 16, 32, …}. The competition harness rounds the largest adapter rank up to the next accepted value, so the rank-2 adapter was served with `max_lora_rank` 8 and no padding. We packed the thinking-off agent with the edit (e001nte, E4B-NoThink-T0.6-FableEdit: e001nt plus `adapter: edit`) and rehearsed it in a Kaggle notebook on four L4 GPUs with internet off, on 12 tasks from the training repositories, one at a time. The pass rule, written before the output, asked for a complete kernel, the adapter mounted and serving requests, no harness error, at least 9 of 12 non-empty patches, and no task over the 8-minute cap beyond the harness's grace. The vLLM log shows `lora_modules` with `edit`, `max_lora_rank` 8 and "Loaded new LoRA adapter: name 'edit'". The server's end metrics show `running_lora_adapters="edit"` and 223 finished requests (222 stopped, 1 at the length limit), with 0 errors. Eleven of 12 patches were non-empty (fastapi_14873 was empty), and the longest task took 349 s of the 480 s cap. A local Docker regrade solved 2 of 12; we read this as a check only. The thinking-on agent (e001tte) passed the same rule in its own rehearsal: adapter `fableedit` mounted with `max_lora_rank` 8, 208 requests, 0 errors, and 11 of 12 non-empty patches (fastapi_11355 empty).
<!-- src: docs/plans/wp7-agent-test-20261107.md "Amendments" 2; outputs/e001nte/rehearsal_v1/vllm.log, metrics_end.txt (vllm:request_success_total stop 222, length 1, error 0), timing.json (wheelhouse 53.2 s, vLLM start 393.5 s, tasks 1,854.8 s, 4 × NVIDIA L4), results/patches/ (11 files), regrade.log ("docker resolved 2/12"); STATUS.md 2026-11-07 20:30 UTC (pass rule, longest task 349 s) and 2026-11-09 21:46 UTC; outputs/e001tte/rehearsal_v1/vllm.log, metrics_end.txt (stop 208, error 0) -->

**Table G.2.** The two competition entries with Fable's edit, and the comparable E4B agents without an adapter that scored on the public board. Both edit entries used the same adapter file (sha256 0d5688d7…). The thinking-off base e001nt was never submitted.

| Entry (ref) | Agent | Thinking | Zip sha256 | Submitted (UTC) | Public score |
|---|---|---|---|---|---|
| e001nte (58130056) | E4B-NoThink-T0.6-FableEdit: e001nt plus Fable's edit | off | 461250bb | 2026-11-08 13:51 | 0.05 |
| e001tte (58175971) | E4B-Think-T0.6-FableEdit: e001tt plus Fable's edit | on (4,096 tokens) | 355b6f98 | 2026-11-09 21:45 | 0.05 |
| e001tt (57995556) | E4B-Think-T0.6, no adapter | on | 25709533 | 2026-11-04 18:56 | 0.06 |
| e005 (58023985) | Verify-E4B, no adapter | on | 5243b88d | 2026-11-05 13:14 | 0.08 |
| e010b (58053361) | BestOf-E4B, no adapter | mixed (4,096 or 0 by sub-agent) | 9c45fed3 | 2026-11-06 09:32 | 0.05 |

<!-- src: STATUS.md "Submissions log" rows 10, 11, 12, 14, 15; thinking_budget in submissions/e001tt_think_t06_e4b/configs/sampling.yaml (4096), submissions/e005_verify_e4b/configs/sampling.yaml (4096), submissions/e010_bestof_e4b/sub_agents/{sampling,sampling_fast}.yaml (4096, 0); outputs/zips/e001nte_nothink_t06_fableedit_e4b.zip and e001tte_think_t06_fableedit_e4b.zip sha256 recomputed (461250bb17a98cb0…, 355b6f98ce17972b…), equal to the submit kernels' outputs (outputs/e001nte/submit_v1/submission.zip, outputs/e001tte/submit_kernel_v1/submission.zip); outputs/e001nte/submit_poll.log (COMPLETE 0.05 at 20:54 UTC 11-08); STATUS.md 2026-11-07 20:30 UTC ("e001nt never on the board"). Row 13 (e011 Staged-E4B, 0.01) is left out of "comparable": a nine-stage design with two known defects (STATUS 2026-11-07 20:28 UTC). -->

**What the board can and cannot show.** The reading rule for each entry was written before its submission: a complete run scoring 0.03 or above means the edit deploys in the scorer and ties with the base by design; a score of 0.02 or below, or an error, means investigate the serving. Both entries completed without an error and scored 0.05. So the board shows that the edited agent loads, runs and finishes within the scorer's limits on the hidden repositories, with thinking off and on. It cannot show what the edit changes. The board reports no artefact counts. It floors scores to two decimals, and from the distinct scores we infer about 58 public tasks, so 0.05 is about 3 tasks and one task moves the score by about 0.017. A gap between 0.05 and 0.06 is one task in one run and is read as a tie.
<!-- src: STATUS.md rows 14 and 15 (reading rules written 2026-11-07 20:30 UTC and 2026-11-09 21:03 UTC, before each submit); STATUS.md 2026-11-07 20:28 UTC ("the board floors to two decimals and 58 is the only public size from 50 to 70 that fits all nine distinct scores"; inferred, not published); AGENTS.md (one task about 0.017) -->

## Appendix H. Live agent tests

A replay holds everything before the old text fixed. Live runs let the agent read, edit and test as it likes, so they show whether the edit changes what the whole agent writes. We ran three live tests of Fable's edit, each with one run per arm: an exploratory first run on OOD-43 and OOD-B, a replication on OOD-C with thinking off, and a second replication on OOD-C with thinking on.

### H.1 Common design

**Arms and serving.** Each test had three arms on one LoRA-enabled vLLM serving the W4A16 E4B model: the base agent, the same agent with Fable's edit, and the same agent with the random adapter (`fable_r2_rand1`). The adapters were byte-identical across all three tests (sha256 0d5688d7… and bfcdbdb9…). Each arm ran 4 tasks at a time, at temperature 0.6, with a 12-minute cap per task. The base agents were e001nt (thinking off) and e001tt (thinking on, budget 4,096 tokens); each edited arm differs from its base by one `adapter:` line.
<!-- src: docs/plans/wp7-agent-test-20261107.md "Stage 2"; docs/plans/wp7r-oodc-run-20261108.md "Design"; docs/plans/wp7t-think-oodc-20261108.md "Design"; log 2026-11-07 10:52 UTC "Stage 2", 2026-11-08 06:32 UTC "Run record", 2026-11-09 03:12 UTC "Run record" -->

**What counts as an artefact.** For each edit call in a trace, the analysis rebuilds the true file state: the task's snapshot file with every earlier successful edit re-applied by the harness's own code. A call counts as an artefact call if its old text *or* its new text contains a literal `\n` or `\"` that the file state does not contain. This differs from the old-text replay, which scores the old text only; Table H.2 also gives old-text-only rates. A file named by a shell command that may modify it becomes uncertain from then on, and every file does if the command names none. Calls on files with an unknown or uncertain state are not counted, and their numbers are reported.
<!-- src: scripts/weight_edit/wp7_stage2_analyze.py docstring (lines 12-22) -->

**Primary test.** Per task, the artefact rate over its counted calls. Two arms are paired on a task when both have at least one counted call. We report the mean of the per-task differences and a one-sided exact sign test that the edit is lower, against the base and against the random adapter. For the two replications the pass rule, fixed in advance, was: both differences negative and both p below 0.05. The first run had no pass rule.
<!-- src: wp7_stage2_analyze.py docstring "PRIMARY"; wp7r and wp7t plans "Analysis" -->

### H.2 Provenance

**Table H.1.** Plans, freezes, commits and run records. Times are UTC. "Frozen" is the time the plan's sha256 was written into the weight-edit log, a local file. For both replications the plan, its log entry and the analysis script were first committed to git after the run had finished, so no git record predates their data; the log entry's time and the recorded hash are the only evidence of the freeze.

| Test | Plan file (sha256 prefix) | Frozen | First git commit of the plan | Run | Hardware |
|---|---|---|---|---|---|
| First run, exploratory | `docs/plans/wp7-agent-test-20261107.md` (7467200cc970 at the freeze; 806c3abbed46 after two amendments) | in git, commit 1c8d19f, 2026-11-07 09:38; amendments at 8ab880b, 10:50 | 1c8d19f, before the run | 2026-11-07: OOD-43 10:55–11:55; OOD-B-71 11:38–13:04 | Colab, two L4 sessions (`wp7a`, `wp7b`), about 4.6 L4 hours |
| Replication, thinking off | `docs/plans/wp7r-oodc-run-20261108.md` (b175eff406b9) | log 2026-11-07 23:16 | 4359669, 2026-11-08 16:16, after the run | 2026-11-08 01:07–06:30 (arms 03:02–06:25) | Colab, one L4 (`wp7r`), about 5.4 L4 hours, 1.5 h of it fetching the task set |
| Replication, thinking on | `docs/plans/wp7t-think-oodc-20261108.md` (496b7a0f9288) | log 2026-11-08 22:35 | 93117d8, 2026-11-09 20:55, after the run | 2026-11-08 22:35 to 2026-11-09 02:26 (3.7 h) | Kaggle `ozgung/g4-wp7t-think-oodc` v1, four L4 GPUs (tensor parallel 4), internet off |

<!-- src: shasum -a 256 of the three plans (current files) and of `git show 1c8d19f:docs/plans/wp7-agent-test-20261107.md` (7467200cc970); `TZ=UTC git log` for each plan (1c8d19f 2026-11-07 09:38:37, 8ab880b 10:50:15, 4359669 2026-11-08 16:16:07, 93117d8 2026-11-09 20:55:42); `git log -S` shows b175eff406b9 first enters the weight-edit log in 4359669 and 496b7a0f9288 in 93117d8; log 2026-11-07 10:52, 11:56, 13:08 UTC; 2026-11-07 23:16 UTC; 2026-11-08 06:32 UTC; 2026-11-08 22:35 UTC; 2026-11-09 03:12 UTC; outputs/wp7t/status_poll.log (RUNNING 22:35, COMPLETE 02:26) -->

**Analysis code.** The first run's analysis, `wp7_stage2_analyze.py`, was committed at 9f14a41 at 11:01 UTC on 2026-11-07. The OOD-43 arms had started at 10:55 UTC, but no trace had been read. For the replication the script (724546e967c6) differed from 9f14a41 only in its table of task sets and one docstring sentence, and on the first run's traces it reproduced the first run's report byte for byte. For the thinking-on run the script (20e41b9d4ffb) added only a set name and its arm table, and reproduced the replication's report exactly. The descriptive analyses were written after each primary result had been read: `wp7r_descriptive.py` (0c0df7bcf978), and `wp7t_descriptive.py` (4259db8eab07), a copy with paths and arm table changed.
<!-- src: `TZ=UTC git log -1 9f14a41` (2026-11-07 11:01:23 UTC); log 2026-11-07 11:01 UTC (WP7 stage-2 analysis entry), 2026-11-07 23:16 UTC, 2026-11-08 06:32 UTC "Descriptive", 2026-11-08 22:35 UTC "Code at freeze", 2026-11-09 03:12 UTC; shasum of scripts/weight_edit/wp7t_descriptive.py (4259db8eab07) and wp7r_descriptive.py (0c0df7bcf978); `diff` of the two descriptive scripts -->

**Task sets.** The first run used the 44 tasks of OOD-43 (named OOD-45 in some files) and the 71 tasks of OOD-B in `configs/tasks_oodb71_colab.txt`: 115 tasks from the repositories that also supplied the replay's test calls. Both replications used the same 186 OOD-C tasks in 148 repositories. A script checked that no OOD-C repository appears in training, OOD-43, OOD-B or the replay test set; the only local agent runs it found on OOD-C were a 7-task smoke test against a scripted stub, not a model.
<!-- src: configs/tasks_oodb71_colab.txt (71 task lines); outputs/weight_edit/claude/wp7/stage2_report.json descriptive ood/* tasks 44, oodb/* tasks 71; log 2026-11-08 01:08 UTC (check_disjoint.py), 03:05 UTC (186 tasks in 148 repos); kernels/wp7t_think_oodc/wp7t_run.py TASKS (186 ids) -->

### H.3 Results

**Table H.2.** Live results, one run per arm (compact form of the main paper's live-run table). Artefact shares are pooled over counted edit calls; the tests in Table H.3 compare per-task rates. "Fallback" is the share of accepted calls that were accepted only after the harness unescaped the old and new text. Solved tasks on OOD-C count only the 53 gradable tasks (H.4).

| Test | Arm | Edit calls (uncounted) | Artefact, old or new text ↓ | Artefact, old text only ↓ | Old text not found | Fallback | Solved ↑ |
|---|---|---|---|---|---|---|---|
| First run, 115 tasks | Base | 249 (6) | 18.1 % (44 of 243) | 14.4 % | 37.8 % | 19.3 % (26 of 135) | 20 of 115 |
| | Fable's edit | 271 (11) | 3.8 % (10 of 260) | 2.7 % | 35.1 % | 5.9 % (9 of 153) | 22 of 115 |
| | Random adapter | 278 (8) | 20.4 % (55 of 270) | 15.9 % | 33.8 % | 23.9 % (34 of 142) | 20 of 115 |
| OOD-C, thinking off | Base | 353 (6) | 15.0 % (52 of 347) | 13.5 % | 27.2 % | 15.4 % (35 of 228) | 7 of 53 |
| | Fable's edit | 411 (12) | 3.3 % (13 of 399) | 2.5 % | 33.6 % | 5.5 % (13 of 237) | 9 of 53 |
| | Random adapter | 409 (9) | 15.5 % (62 of 400) | 12.8 % | 32.0 % | 13.5 % (31 of 229) | 7 of 53 |
| OOD-C, thinking on | Base | 395 (8) | 7.0 % (27 of 387) | 6.7 % | 25.1 % | 8.7 % (22 of 253) | 8 of 53 |
| | Fable's edit | 433 (14) | 1.2 % (5 of 419) | 0.7 % | 35.1 % | 2.6 % (6 of 233) | 7 of 53 |
| | Random adapter | 460 (20) | 8.2 % (36 of 440) | 6.4 % | 35.7 % | 7.6 % (20 of 263) | 7 of 53 |

<!-- src: outputs/weight_edit/claude/wp7/stage2_report.json descriptive (ood/* + oodb/* summed: edit_calls, counted_calls, artefact_calls, art_old, not_found, strategy_unescaped, accepted, resolved); outputs/weight_edit/claude/wp7r/report.json descriptive.oodc/*; outputs/weight_edit/claude/wp7t/report.json descriptive.oodct/*; solved on 53 from wp7r/descriptive.json and wp7t/descriptive.json gradable_resolved_*; first-run solved from stage2_report.json resolves and descriptive (VM grading) -->

**Table H.3.** Per-task tests. Rates are means of per-task artefact rates on the paired tasks, in points. The interval is a repository-clustered bootstrap (10,000 resamples of repositories, seed 20261108) of the mean difference A − B. The random-against-base rows are descriptive; their p tests "random lower".

| Test | A against B | Paired tasks | A lower / higher / equal | Rates A, B | Mean difference | One-sided p | 95 % interval (repositories) |
|---|---|---|---|---|---|---|---|
| First run | edit, base | 112 | 21 / 5 / 86 | not computed | −12.0 | 0.0012 | not computed |
| | edit, random | 113 | 25 / 3 / 85 | not computed | −14.1 | 1.4 × 10⁻⁵ | not computed |
| | random, base | 112 | 17 / 21 / 74 | not computed | +2.2 | 0.79 | not computed |
| OOD-C, thinking off | edit, base | 184 | 34 / 4 / 146 | 3.1, 14.8 | −11.6 | 3.0 × 10⁻⁷ | −16.3 to −7.3 (147) |
| | edit, random | 183 | 27 / 7 / 149 | 3.2, 12.0 | −8.9 | 4.1 × 10⁻⁴ | −14.1 to −3.8 (147) |
| | random, base | 184 | 28 / 21 / 135 | 12.0, 14.8 | −2.8 | 0.20 | −8.7 to +3.1 (148) |
| OOD-C, thinking on | edit, base | 179 | 16 / 4 / 159 | 0.9, 6.0 | −5.1 | 0.0059 | −8.5 to −2.2 (143) |
| | edit, random | 172 | 23 / 4 / 145 | 0.9, 8.4 | −7.4 | 1.6 × 10⁻⁴ | −11.4 to −3.9 (138) |
| | random, base | 173 | 10 / 19 / 144 | 8.3, 5.7 | +2.7 | 0.97 | −1.0 to +6.5 (138) |

<!-- src: stage2_report.json primary.pooled.{edit_vs_base,edit_vs_rand,rand_vs_base}; wp7r/report.json primary.oodc.*; wp7r/descriptive.json {edit_vs_base,edit_vs_rand,rand_vs_base}.{macro_rate,macro_diff,repo_bootstrap_95,repos}; wp7t/report.json primary.pooled.*; wp7t/descriptive.json; bootstrap settings from scripts/weight_edit/wp7r_descriptive.py docstring and code (random.Random(20261108), 10000 resamples, percentiles 249/9749) -->

Both replications passed their rule: both differences were negative and both p values below 0.05. The thinking-on plan also required at least 150 paired tasks per comparison; there were 179 and 172. Counting the old text only gives the same ordering in every run (Table H.2).
<!-- src: wp7r/report.json, wp7t/report.json as above; docs/plans/wp7t-think-oodc-20261108.md "Design" (at least 150 tasks); log 2026-11-09 03:12 UTC -->

**First run, by set.** The first run pooled two sets. On OOD-43 the edit was lower than the base in 7 tasks and higher in 2 (p = 0.090), and lower than the random adapter in 8 and higher in 2 (p = 0.055). On OOD-B it was 14 against 3 (p = 0.0064) and 17 against 1 (p = 7.2 × 10⁻⁵). Live counts are noisy between runs: in a smoke test of the analysis code, run before any first-run trace was read, three earlier base runs of OOD-43 without any edit had 20, 11 and 19 artefact calls (of 94, 99 and 78 counted). The first run's solved tasks were graded in the run machine's sandbox; no Docker regrade is reported.
<!-- src: stage2_report.json primary.ood.*, primary.oodb.*; log 2026-11-07 11:01 UTC (WP7 stage-2 analysis entry, "Smoke test on three earlier base runs of OOD-43"); log 2026-11-07 13:08 UTC ("VM grading, no Docker regrade yet") -->

### H.4 Solved tasks and the gradable list

Solved tasks on OOD-C count only the 53 tasks in `outputs/weight_edit/claude/wp7r/gradable_cp312.txt` (sha256 74f7a59f8aff). A task is on this list if its reference fix passed the build gate twice, the empty patch failed, and the reference fix passed again under Python 3.12. Both OOD-C runs used this one list. Recounting solved tasks on it from the six arms' `task_results.jsonl` files gives 7, 9 and 7 (thinking off) and 8, 7 and 7 (thinking on) for the base, the edit and the random adapter, as in both descriptive reports. Solved counts over all 186 tasks (14, 19 and 14; 13, 14 and 14) include tasks whose reference fix fails, so we do not read them.
<!-- src: outputs/weight_edit/claude/wp7r/gradable_cp312.txt (53 lines; shasum); log 2026-11-08 03:05 UTC ("gradable 53 on cp312 (58 on cp313, 60 on the shards)"); log 2026-11-09 03:12 UTC ("Resolved on the 53 cp312-gradable tasks (WP7R's list)"); read-only recount with python3 -I over results/wp7r/results/wp7_oodc_*_cap12/task_results.jsonl and results/wp7t/results/wp7_oodct_*_cap12/task_results.jsonl (53 of 186 tasks found in each arm); wp7r/report.json and wp7t/report.json resolves -->

Both OOD-C runs graded under Python 3.12, so the list matches both. The replication ran on Colab, where the harness ran from `/content/venv/lib/python3.12`. The thinking-on run ran on Kaggle, where the harness ran from the notebook image's `/usr/local/lib/python3.12`. In all six arms each task's sandbox environment was a `venv/lib/python3.12` and installed cp312 wheels. The competition's own sandbox image provides Python 3.13, so the 53-task list describes our runs, not the scorer.
<!-- src: results/wp7r/results/wp7_oodc_e001nt_baseline_nothink_t06_e4b_cap12.log (/content/venv/lib/python3.12/...); results/wp7t/wp7_oodct_*_cap12.log (/usr/local/lib/python3.12/dist-packages/..., sandbox "probe:" lines with venv/lib/python3.12, no python3.13 path); results/{wp7r,wp7t}/results/*/ood_env_install.jsonl (only cp312 wheel tags, venv/lib/python3.12 in every arm); docs/competition/page_data-description.md:50 (Dockerfile.sandbox "providing Python 3.13") -->

On the 53 tasks the differences are ties. With thinking off the edit solved 9 and the base 7 (discordant pairs 2 to 0), and against the random adapter 9 to 7 (3 to 1). With thinking on the edit solved 7 and the base 8 (1 to 2), and the random adapter 7 (1 to 1). In the first run the edit solved 22 of 115 and the base 20 (discordant 11 to 9). The plans read a difference under about 7 tasks as a tie. These samples cannot show that the edit leaves solved tasks unchanged.
<!-- src: wp7r/descriptive.json gradable_resolved_edit_vs_base {resolved [9,7], edit_only 2, base_only 0}, gradable_resolved_edit_vs_rand {[9,7], 3, 1}; wp7t/descriptive.json {[7,8], 1, 2} and {[7,7], 1, 1}; stage2_report.json resolves.edit_vs_base {[22,20], edit_only 11, ref_only 9}; wp7 and wp7r plans "Descriptive" (tie under about 7) -->

### H.5 Expectations written in advance

**Table H.4.** What each plan wrote down before its run, against what came out.

| Test | Written in advance | Outcome |
|---|---|---|
| First run | No effect size. About 30 informative tasks per comparison; a solved-task difference under about 7 is a tie. | 26 and 28 informative tasks; solved 22, 20 and 20, a tie. |
| OOD-C, thinking off | Edit/base ratio between 0.2 and 0.6. | 0.21 per task (0.22 per call): met, at the low edge. |
| | Random adapter within noise of the base. | Ratio 0.81; interval of the difference includes 0: met. |
| | Old text not found per call does not move by more than noise. | 27.2, 33.6 and 32.0 % (base, edit, random): not met; both adapters raised it. |
| OOD-C, thinking on | Base rate about 8 to 13 %. | 6.0 % per task: not met, lower. |
| | Edit/base ratio between 0.2 and 0.7. | 0.15 (0.17 per call): not met, lower. |
| | Random adapter within noise of the base. | Ratio 1.47, interval −1.0 to +6.5 points: met loosely. |
| | Primary passes, with about 15–25 tasks lower against 2–6 higher. | 16 / 4 against the base, 23 / 4 against random: met. |
| | No claim on old text not found. | 25.1, 35.1 and 35.7 %. |

<!-- src: docs/plans/wp7-agent-test-20261107.md "Stage 2" (expected information, resolve noise); docs/plans/wp7r-oodc-run-20261108.md "Expectation, written before the run"; docs/plans/wp7t-think-oodc-20261108.md "Expectation, written before the run"; log 2026-11-08 06:32 UTC and 2026-11-09 03:12 UTC "Expectation, written before the run"; wp7r/descriptive.json macro_ratio 0.2130 and 0.8111; wp7t/descriptive.json macro_ratio 0.1504 and 1.4696; stage2_report.json tasks_lower+tasks_higher (21+5, 25+3) -->

The rise in calls whose old text was not found appeared with both adapters in both OOD-C runs, so it is not specific to the edit's direction. Both adapters also raised the number of edit calls per task: 1.90, 2.21 and 2.20 with thinking off, and 2.12, 2.33 and 2.47 with thinking on (base, edit, random). In the first run the not-found share fell slightly with both adapters. We did not test the cause; no zero-weight adapter arm separates serving effects from weight changes.
<!-- src: wp7r/report.json and wp7t/report.json descriptive (not_found/edit_calls; edit_calls/tasks); stage2_report.json (94/249, 95/271, 94/278); log 2026-11-08 06:32 UTC and 2026-11-09 03:12 UTC "Descriptive" -->

### H.6 Notes on the thinking-on run

**The snapshot step failed and was replaced.** The kernel's own step that copies the touched snapshot files left an empty archive (45 bytes), and its log was empty, so the cause is unknown. 173 touched files in 75 tasks were missing locally. A separate CPU kernel (`ozgung/g4-wp7t-files-extract` v1, `extract.py` 4b13dceae87c), with the same matching code, found 138 of them; the other 35 are not in the task archives. They were unpacked without overwriting anything, and the frozen analysis was run again. That run is the result above. A first run of the analysis without those files had already been read; it also passed (15 lower and 4 higher against the base, p = 0.0096; 20 and 4 against random, p = 0.0008). The extra files cut the uncounted calls from 23, 30 and 57 to 8, 14 and 20. Traces were present for 186, 185 and 177 tasks.
<!-- src: log 2026-11-09 03:12 UTC "Deviation in conduct"; results/wp7t/wp7t_files.tgz (45 bytes); outputs/wp7t/analyze.txt (preliminary: counted 372, 403, 403; 15/4/153 p 0.00961; 20/4/138 p 0.000772); outputs/weight_edit/claude/wp7t/report_prelim_nofiles.json; wp7t/report.json descriptive traces_missing 0, 1, 9 -->

**Other conduct.** The arms were e001tt (zip 25709533), e001tte (355b6f98) and e001ttx (dd964ba4), all with a 12-minute cap (e001tt's own value is 8) and 28 tool calls. The kernel ran for 13,261 s in all; vLLM was up after 416 s and each arm took about 12,700 s. The 10.5-hour stop was not reached. The plan's header gives the user's go as "about 22:40 UTC"; the log records it at about 22:28 UTC and kept the header unchanged so that the frozen hash stands.
<!-- src: docs/plans/wp7t-think-oodc-20261108.md "Design"; results/wp7t/results/wp7_oodct_e001tt_think_t06_e4b_cap12/run_meta.json eval_config {max_tool_calls 28, max_time_minutes 12}; results/wp7t/timing.json (vllm_start_s 415.5; wall_s 12735.1, 12720.1, …); log 2026-11-09 03:12 UTC "Run record" (total 13,261 s); log 2026-11-08 22:35 UTC (stamp error note) -->

**An earlier count with thinking on.** Before this test we counted live artefacts in existing paired runs of e001tt and e001nt on OOD-43 and OOD-B (2026-11-02), with the same artefact definition. e001tt had 24 artefact calls of 244 counted (9.8 %) and e001nt 31 of 255 (12.2 %). Per task, thinking on was lower in 20 tasks and higher in 13 of 108 (two-sided p = 0.30). This count was post hoc and descriptive; it informed the thinking-on plan's expected base rate.
<!-- src: log 2026-11-08 21:18 UTC; outputs/weight_edit/claude/tt_artefact/report.txt ("paired tt-nt ood+oodb tasks 108 … tt lower 20 higher 13 p2 0.296"); docs/plans/wp7t-think-oodc-20261108.md "Expectation" (earlier sessions 0.13 → 0.07-0.10) -->

## Appendix I. Retry census and the two go/no-go checks

After round 2 we asked where the base model's remaining misses come from. A census of existing data found that most follow an earlier miss in the same task. Before spending an editor's budget on such retries, we ran two go/no-go checks with the base model only. Both failed their rules, so the third round did not run.

### I.1 Census

**Definitions.** A *retry* is a usable recorded edit call in a task run that already had a recorded not-found call. Its *failed set* is the set of old texts of those earlier not-found calls. A *first attempt* is any other usable call. The census used the round-2 test set and existing traces only, and changed no verdict.
<!-- src: log 2026-11-07 19:50 UTC; log 2026-11-07 20:17 UTC "Terms" -->

Of the 202 recorded not-found calls in the held-out runs, 131 followed an earlier not-found in the same task. The test set holds 249 first attempts and 71 retries.
<!-- src: outputs/weight_edit/claude/retry_census/census_20261107.txt ("held-out recorded not_found: 202 of which after an earlier not_found in the task: 131"; "usable first/retry calls: 249 71") -->

**Table I.1.** Replay of the test set (two draws per call). "Repeats" are retry generations byte-identical to a string in the call's failed set.

| Arm | First attempts accepted | Retries accepted | Retry repeats |
|---|---|---|---|
| Base | 405 of 498 (81 %) | 38 of 142 (27 %) | 78 of 142 (55 %) |
| Opus's edit | 406 (82 %) | 43 (30 %) | 72 (51 %) |
| Fable's edit | 400 (80 %) | 45 (32 %) | 64 (45 %) |
| Informed Opus's edit | 389 (78 %) | 38 (27 %) | 58 (41 %) |
| Two random controls | 405 (81 %) | 38 (27 %) | 78 (55 %) |

<!-- src: retry_census/census_20261107.txt arm table (fable_r2_rand1 and opusinf_r2_rand1 rows) -->

No base repeat was accepted (0 of 78). The other 64 base retry generations were accepted 38 times (59 %); this share is selection-biased, since a stable copy error may simply fail differently. In the recorded traces (230 of e001nt, 115 of e001tt), the next edit after a not-found landed in 19 to 40 % of cases by group and repeated the same old text in 40 to 56 %, including after the agent re-read the file (44 to 50 %). 25 of the 345 traces had a not-found and no edit that ever landed. None of the round-2 edits aimed at retries.
<!-- src: census_20261107.txt ("retry gens identical accepted [0, 78] non-identical accepted [38, 64]"; trace blocks: next_edit_landed 0.19-0.40, same_old_string 0.40-0.56, reread 0.50/0.44; nf_and_no_edit_ever_landed 20 + 5); log 2026-11-07 19:50 UTC -->

### I.2 First check: can a retry land, and is the state readable?

**Rule.** We wrote the rule into the weight-edit log at 20:17 UTC on 2026-11-07, before any new generation. It recorded the sha256 prefixes of five scripts (`retry_gonogo_analyze.py` e38b35353690, `retry_gonogo_keys.py` 152d9dba3f6a, `retry_gonogo_probe.py` a25249ef74e1, `replay_gen.py` e090d2500559, `replay_score.py` bc2a19433c2e). The scripts were not committed then; the hashes were the freeze. They were first committed at 4359669 (16:16 UTC on 2026-11-08). Round 3 would go ahead only if both parts passed.
<!-- src: log 2026-11-07 20:17 UTC "Rule", "Frozen files"; `TZ=UTC git log` for scripts/weight_edit/retry_gonogo_*.py (4359669); current shasum matches all five prefixes except the probe (see amendment) -->

- **G1 (sampling).** Each of the 71 held-out retries was generated 8 times: greedy plus 7 draws at the served settings (temperature 0.6, top-p 0.95, top-k 64). A retry is *stuck* if at least 4 of its 8 draws repeat a string in its failed set, and *escapable* if it is stuck and at least one draw does not repeat. The value is the mean over tasks of the mean, over the task's escapable calls, of acceptance among non-repeating draws. Pass: at least 0.40 over at least 8 tasks.
- **G2 (probe).** A linear probe was fitted on training-repository calls only (120 retries, 120 first attempts) and tested on held-out calls (71 retries, 71 first attempts). Features were the residual stream at the last prefix token, at layers 35, 38, 40 and 41, standardised and reduced to 64 principal components, with ridge regression on ±1 labels (lambda equal to n). Pass: best held-out AUC of the four layers at least 0.80. Choosing the best of four was disclosed in the rule. The rule also noted that a high AUC is necessary, not sufficient: round 1's recipe had a held-out AUC of 0.88 and no effect.

<!-- src: log 2026-11-07 20:17 UTC "G1", "G2"; outputs/weight_edit/claude/retry_gonogo/keys_{train,heldout}_{retry,first}.txt (120, 120, 71, 71 lines); scripts/weight_edit/retry_gonogo_probe.py docstring (positions turn, open, mid) -->

**Amendment, infrastructure only.** At 21:33 UTC the probe's GPU buffer cache reached about 36 GB of the Mac's 48 GB. We stopped it, added a 4 GB cache limit and a cache clear after each forward pass (new sha256 a8bd21616bb9; inputs, positions, layers and analysis unchanged), and resumed from the probe step. No result had been read.
<!-- src: log 2026-11-07 21:33 UTC; shasum scripts/weight_edit/retry_gonogo_probe.py (a8bd21616bb9) -->

**Result (23:00 UTC): no-go.** G1 failed and G2 passed.
<!-- src: log 2026-11-07 23:00 UTC; outputs/weight_edit/claude/retry_gonogo/gonogo_report.json {G1 false, G2 true, GO false} -->

- **G1.** 71 calls and 568 draws. 39 calls were stuck (in 18 of 33 tasks), and only 4 were escapable (in 4 tasks). The value was 0.33 over 4 tasks, against 0.40 over at least 8, so both conditions failed. The repeat counts were bimodal: 35 calls repeated a failed string in all 8 draws, greedy included, 32 never did, and 2 each repeated in 5 and in 7 draws. Acceptance among all non-repeating draws was 0.59. A 100-call replay on training calls, descriptive under the rule, gave the same picture: 46 stuck, 2 escapable, value 0.0.
- **G2.** The best held-out AUC at the rule's position was 0.959, at layer 35; layers 38, 40 and 41 gave 0.945, 0.949 and 0.947. Two positions outside the rule, the start of the model's turn and 8 tokens into the old text, gave 0.990 to 0.994 and 0.55 to 0.63.

<!-- src: gonogo_report.json g1_heldout {calls 71, draws 568, stuck_calls 39, escapable_calls 4, tasks 4, value 0.3333, all_non_identical_acceptance 0.5909}, g1_train_exploratory {stuck 46, escapable 2, value 0.0}, probe.L{35,38,40,41}_{open,turn,mid}.heldout_auc, g2_best_open_auc 0.9589; log 2026-11-07 23:00 UTC (18 of 33 tasks; repeat counts {0:32, 5:2, 7:2, 8:35}) -->

Sampling at the served settings almost never moved a stuck call off its failed string. So G1 had almost no data on whether a stuck call could land at all; it does not show that it could not. That gap motivated the second check.
<!-- src: log 2026-11-07 23:00 UTC "Descriptive, not part of the rule"; log 2026-11-07 23:13 UTC "Why a second check" -->

### I.3 Second check: remove the copy source

**Rule.** We wrote the second rule at 23:13 UTC, after the first check had failed and before any masked generation; it is reported as a second check, not as part of the first. Frozen files: `retry_mask_calls.py` 553c1ed953bb, `retry_mask_analyze.py` 49e1f795cc27, `replay_gen.py` e090d2500559, `replay_score.py` bc2a19433c2e, and the masked call file `retry_gonogo/mask/calls.jsonl` 15a7a88a6c41. The check is an upper bound for an edit that weakens copying: if the model cannot land a stuck retry with the failed string gone from its context, an edit that only weakens the copy should not either. G1b: on the 39 stuck calls fixed by the first check, the mean over tasks of the mean over the task's stuck calls of acceptance over all 8 masked draws, where a draw that still repeats counts as not accepted. Pass: at least 0.40 over at least 8 tasks, the same bar as G1.
<!-- src: log 2026-11-07 23:13 UTC "Why a second check", "G1b", "Frozen files"; current shasum of the four scripts and calls.jsonl match these prefixes -->

**Masking procedure and its check.** For each of the 71 retries, every occurrence of every string in the failed set was deleted from the model-authored parts of the prompt and of the current-turn prefix: the model's own text and its tool-call arguments, longest strings first. Environment text, meaning system and user turns and tool responses (file views, command output and the not-found errors), was copied unchanged, so each call is still a retry. The script asserts, call by call, that every environment span in the prompt is identical before and after masking. On this set the failed strings occur in no tool response. The masked set has 71 rows, 318 deletions in prompts and 55 in prefixes, and no occurrence left; the median call lost 1,503 characters. Generation used the same generator, 8 draws and the same seeds as the first check.
<!-- src: scripts/weight_edit/retry_mask_calls.py (docstring; ENV regex over <|tool_response> and user/system turns; `assert ENV.findall(r["prompt"]) == env_before`); recount of outputs/weight_edit/claude/retry_gonogo/mask/calls.jsonl (71 rows, mask_prompt_deletions 318, mask_prefix_deletions 55); log 2026-11-07 23:13 UTC (no occurrence in tool responses; median 1,503 characters) -->

**Result (23:52 UTC): fail.** On the stuck calls, masked acceptance was 0.166 over 18 tasks, against the bar of 0.40.

**Table I.2.** Base model, context unchanged against masked, 8 draws per call; task-macro shares of draws.

| Calls | Context | Accepted | Accepted and compiling | Repeats a failed string |
|---|---|---|---|---|
| 39 stuck (18 tasks) | unchanged | 0.008 | 0.001 | 0.97 |
| | masked | 0.166 | 0.075 | 0.48 |
| 32 not stuck (24 tasks) | unchanged | 0.594 | 0.552 | 0.000 |
| | masked | 0.602 | 0.505 | 0.005 |

<!-- src: outputs/weight_edit/claude/retry_gonogo/mask/report.json {stuck_base, stuck_masked, not_stuck_base, not_stuck_masked}.{acceptance, accepted_and_compiling, repeat_share, tasks}; G1b_value 0.1661, G1b false, GO_3b false; log 2026-11-07 23:52 UTC -->

With the failed strings deleted, the model still wrote a byte-identical failed string in about half of the draws on stuck calls. Masking barely changed acceptance on the other calls (0.60 against 0.59), but their accepted-and-compiling share fell from 0.55 to 0.51. Under the rule the retry target was closed for this paper.
<!-- src: mask/report.json as above; log 2026-11-07 23:52 UTC "Reading" -->

### I.4 The unused round-3 plan

At 21:36 UTC, after the first rule and before its verdict, we wrote a plan for a third round aimed at retries; it was never run. Two independent Opus runs would each have had 25 forward passes and the round-2 sandbox, with the round-2 test set as development data. The new test set would have been built from base runs on OOD-C. The primary measure was the per-task share of retry generations that repeat a failed string, against the base and two random controls, with the round-2 guards and Holm correction over the runs; retry acceptance was the key secondary measure. Its files were frozen in `round3/FROZEN.sha256` (3e96323d7b65). When the first check failed, the night chain wrote a `NOGO` marker; no smoke test, editor pass or round-3 compute followed.
<!-- src: log 2026-11-07 21:36 UTC; shasum outputs/weight_edit/claude/round3/FROZEN.sha256 (3e96323d7b65); outputs/weight_edit/claude/round3/NOGO exists; log 2026-11-07 23:00 UTC "Consequence" -->

### I.5 What the checks do not show

The probe told retries from first attempts; it was never asked to tell stuck retries from the others. A retry's prompt contains an earlier not-found error, so a surface cue could explain a high AUC, and we report no baseline for that cue. Masking removed exact copies only; near-variants or fragments of a failed string in the model's own text remain. That the model rebuilds the failed string from its reading of the file is a plausible account of the 48 % repeats, not a tested one. Both rules lived in the local log and were committed after the checks ran.
<!-- src: log 2026-11-07 20:17 UTC (probe labels retry vs first); docs/paper/review/draft-v4-simple-claimcheck-20261108.md §2 items 2, 3 and §4 Q4; git 4359669 -->

## Appendix J. Edit against training

Editing by reasoning would be worth more if it gave something training does not. This comparison tested one candidate, using less data, and one question about location: given Fable's module, rank and size, does gradient descent find better or worse values than Fable did?

### J.1 Plan and timing

The plan is `docs/plans/edit-vs-train-20261108.md` (sha256 7e0d96fdd884). It was frozen at 17:08 UTC on 2026-11-08, before the first training step of any arm and before any trained adapter touched the test set. Two things were done before the freeze, both on training-repository data only: the data build (`MANIFEST.json` 8b4056afe239) and a 2-step smoke run of the location-matched form (17:03–17:07 UTC). The learning-rate grid and the 20,000-token length limit were added after the smoke run and before the freeze. The analysis script `gate_g_analyze.py` (76de8740512b) was hashed in the log at 17:12 UTC, before any trained arm was generated on the test set; on the reused base and Fable arms it reproduces the round-2 numbers. The plan, the scripts and the log entry were first committed to git at 93117d8, at 20:55 UTC on 2026-11-09, after all gates had finished (18:27 UTC).
<!-- src: shasum docs/plans/edit-vs-train-20261108.md (7e0d96fdd884…); plan "Freeze" paragraph; log 2026-11-08 17:12 UTC; shasum scripts/weight_edit/gate_g_analyze.py (76de8740512b), data/edit_vs_train/MANIFEST.json (8b4056afe239); `TZ=UTC git log` (93117d8 2026-11-09 20:55:42); `git log -S7e0d96fdd884` on the weight-edit log (93117d8); log 2026-11-09 18:34 UTC ("GATE CHAIN DONE" 18:27) -->

### J.2 Data and labels

The training pool was the 730 round-2 search calls the editors saw, from the four training repositories; the test set was never read for training. Each call's target was the first of two strings that occurs exactly once in the true file: the recorded old text, or its JSON-unescaped form. 411 calls had such a target, 341 with the recorded text and 70 with the unescaped one; the other 319, mostly not-found calls that unescaping does not recover, were dropped. Fable's calls are the unique calls of its read passes 1, 2 and 5, the passes its final answer names as the fit data for its edit: 88 texts over 57 calls. 39 of the 57 have a target, in 32 tasks and 443,690 tokens. The input is the replay's exact prompt and prefix, ending where the old text begins; the loss covers only the target's tokens and the closing quote token.
<!-- src: plan "Data and labels"; data/edit_vs_train/MANIFEST.json (label_tags_730 {none 319, recorded 341, unescaped 70}; all411 {calls 411, tokens 4,555,306}; fable39 {calls 39, tasks 32, unique_read_keys 57, tokens 443,690}); log 2026-11-08 17:12 UTC "Label" -->

### J.3 Arms and training

**Table J.1.** The three trained arms. All are LoRA adapters trained with `scripts/mlx_lora/train.py` on the same 16-bit weights the replay uses, seed 0, cosine learning rate to 0.1 times with 5 % warm-up, then converted to the replay's edit format. Size is the Frobenius norm of the full weight change.

| Arm | Modules | Rank / alpha | Data | Epochs (steps) | Learning rate | Size |
|---|---|---|---|---|---|---|
| Fable's edit (reference) | layer 41 `mlp.down_proj` | 2 | read passes 1, 2 and 5 over 57 calls, plus generation passes | | | 5.06 |
| G-yer-39: Fable's location and size | layer 41 `mlp.down_proj` (25,600 parameters) | 2 / 2 | Fable's 39 | 3 (30) | chosen from 1e-3, 3e-3, 1e-2: **1e-2** | 5.06 (capped) |
| G-tam-39: all layers, 39 calls | all 258 linear modules (17.4 M parameters) | 8 / 16 | Fable's 39 | 3 (30) | chosen from 1e-4, 3e-4, 1e-3: **1e-4** | 9.44 |
| G-tam-411: all layers, 411 calls | all 258 linear modules | 8 / 16 | all 411 | 2 (104) | fixed at 1e-4 | 15.59 |

<!-- src: plan "Arms"; outputs/weight_edit/claude/edit_vs_train/run_train.log (total_fro 5.0600, 9.4367, 15.5871); runs/*.log (n_params 25,600 and 17,440,768; steps 30/30 and 104/104); log 2026-11-08 21:35 UTC (Fable 5.062) -->

The location-matched arm had its size capped: after every optimiser step, if the change's norm exceeded 5.06, Fable's, it was scaled back to 5.06. It ended at the cap, with the same module and shapes as Fable's edit. The two all-layer arms are not size-matched to the edit.
<!-- src: plan "G-yer norm cap"; log 2026-11-08 21:35 UTC ("Frobenius 5.060 (at the cap; Fable 5.062)", "B 2560x2, A 2x10240") -->

### J.4 Learning rates

Only the two 39-call arms tuned their learning rate. Each chose from its grid of three by the lowest final negative log-likelihood (NLL) on 32 validation calls: labelled calls from training-repository tasks outside Fable's 32, drawn with seed 20261108. The test set was never used for a choice. The 411-call arm used a fixed learning rate of 1e-4, the trainer's standard, because one run takes about 7 hours. Its rule: if its training loss (mean of the last 3 steps against step 1) fell by less than 25 %, it would be rerun once at 3e-4. The loss fell by 96.9 % (0.0531 at step 1; 0.0005, 0.0031 and 0.0013 at steps 102 to 104), so 1e-4 is the arm of record and there was no rerun.
<!-- src: plan "No tuning on the gate set", "Validation"; data/edit_vs_train/MANIFEST.json fable39_val (32 calls); outputs/weight_edit/claude/edit_vs_train/runs/tam411_lr1e-4.log (step 1 loss 0.0531; steps 102-104 losses 0.0005, 0.0031, 0.0013); log 2026-11-09 08:23 UTC ("fell 0.969 relative to step 1") -->

**Table J.2.** Every grid run's final validation NLL on the 32 calls (loss tokens). The base model's NLL at step 0 was 0.0089.

| Arm | 1e-4 | 3e-4 | 1e-3 | 3e-3 | 1e-2 | Chosen |
|---|---|---|---|---|---|---|
| G-yer-39 | | | 0.0053 | 0.0082 | 0.0048 | 1e-2 |
| G-tam-39 | 0.0100 | 0.0375 | 7.244 (diverged) | | | 1e-4 |

<!-- src: outputs/weight_edit/claude/edit_vs_train/runs/yer39_pick.txt, tam39_pick.txt; runs/{yer39,tam39}_lr*.log ("val step 0 loss 0.0089", "val step 30 loss …"); log 2026-11-08 21:35 UTC and 2026-11-09 01:06 UTC -->

Every all-layer run on the 39 calls ended above the base's validation NLL, so training all layers on 39 calls overfit at every grid value. The location-matched arm ended below it. This observation is descriptive and was not part of any rule.
<!-- src: log 2026-11-09 01:06 UTC "Descriptive, not part of any rule" -->

### J.5 Results on the replay test

**Rule.** Each trained arm faced the round-2 rule as amended on 2026-11-05: the artefact primary against the base, acceptance non-inferiority (task-macro change at least −1 point), at most 2 working calls lost per draw, and the full-call compile guard (at most 2 points above the base). The reference set was the base only, with no random controls, and Holm correction over the three arms. The base and Fable generations were reused, not regenerated. Gating ran after all training, not after each arm, because training and replay could not share the Mac's memory; the rule was unaffected.
<!-- src: plan "Evaluation"; scripts/weight_edit/gate_g_analyze.py docstring line 12 (non-inferiority ≥ −0.01) and line 65 (limit max(2, floor(0.01 n))); log 2026-11-08 17:12 UTC "Operational change" -->

**Table J.3.** Trained adapters against Fable's edit on the replay test (320 calls, two draws each). Acceptance intervals are repository-clustered (69 repositories). Compile failures are among accepted calls in the full-call replay.

| Arm | Artefacts of 640 ↓ (base 100) | Tasks lower / higher, one-sided p (Holm threshold) | Acceptance change, points [95 % interval] | Accepted of 640 (base 443) | Working calls lost, draw 0 and 1 (limit 2) | Compile failures (base 7.9 %) | Guards failed |
|---|---|---|---|---|---|---|---|
| Fable's edit | 47 | 20 / 0, 9.5 × 10⁻⁷ | −1.0 | 445 | 5, 5 | 9.1 % | acceptance, working calls |
| G-yer-39 | 23 | 27 / 0, 7.5 × 10⁻⁹ (0.05) | −3.6 [−7.1, −0.6] | 433 | 12, 13 | 10.3 % | all three |
| G-tam-39 | 5 | 30 / 0, 9.3 × 10⁻¹⁰ (0.017) | +1.1 [−2.5, +4.7] | 470 | 4, 5 | 10.7 % | working calls, compile |
| G-tam-411 | 4 | 30 / 0, 9.3 × 10⁻¹⁰ (0.025) | +0.3 [−1.6, +2.5] | 449 | 6, 7 | 5.2 % | working calls |

<!-- src: outputs/weight_edit/claude/edit_vs_train/gate_g_report.json: base {artefacts 100, fullcall_compile_fail 0.0788}; fable_r2 {artefacts 47, vs_base_artefact, vs_base_accepted.task_macro_mean_diff −0.0101, served_ok 5/5, fullcall_compile_fail 0.0909}; arms.{G-yer-39,G-tam-39,G-tam-411}.{artefacts, primary_vs_base, holm_threshold, acceptance_vs_base.task_macro_mean_diff and call_micro, bootstrap.accepted_vs_base.repo_clustered_ci95, served_ok, fullcall_compile_fail, pass false} -->

Every trained arm passed the primary and failed the rule. All three lost more than 2 working calls in each draw. The 411-call arm failed nothing else; its compile failure rate was the only one in the test below the base's.
<!-- src: gate_g_report.json pass false for all three; log 2026-11-09 18:34 UTC -->

**Table J.4.** Head-to-head against Fable's edit, per task, two-sided sign tests ("better" at p < 0.05), with repository-clustered 95 % intervals of the artefact difference (arm − Fable, points).

| Arm | Artefacts: lower / higher, p | Artefact interval | Acceptance: higher / lower, p | Verdict (artefacts / acceptance) |
|---|---|---|---|---|
| G-yer-39 | 13 / 1, 0.0018 | −7.8 to −1.5 | 4 / 10, 0.18 | better / tie |
| G-tam-39 | 17 / 0, 1.5 × 10⁻⁵ | −11.5 to −3.4 | 12 / 3, 0.035 | better / better |
| G-tam-411 | 17 / 0, 1.5 × 10⁻⁵ | −11.7 to −3.5 | 8 / 7, 1.0 | better / tie |

<!-- src: gate_g_report.json arms.*.head_to_head_vs_fable.{escape_artefact,accepted}.{tasks_better,tasks_worse,p_sign_two_sided,verdict}; arms.*.bootstrap.artefact_vs_fable.repo_clustered_ci95 ([−0.0781, −0.0153], [−0.1153, −0.0337], [−0.1168, −0.0345]) -->

Against the base, the repository-clustered intervals of the artefact difference were −21.3 to −8.1 points (G-yer-39), −24.7 to −10.0 (G-tam-39) and −24.9 to −10.1 (G-tam-411), over 101 tasks in 69 repositories. Task-level intervals were close to these.
<!-- src: gate_g_report.json arms.*.bootstrap.artefact_vs_base.{repo_clustered_ci95, task_ci95, n_tasks 101, n_repos 69} -->

### J.6 Readings fixed in advance

**Table J.5.** The plan's readings table and how each row came out.

| Outcome written in advance | Reading written in advance | Result |
|---|---|---|
| Both 39-call arms worse than Fable on artefacts, the 411-call arm tie or better | editing has a data-efficiency advantage | not reached: both 39-call arms are better |
| Either 39-call arm tie or better than Fable | no data-efficiency advantage | holds |
| G-yer-39 better, tie or worse than Fable | gradient values at Fable's location and size are better, as good or worse | better on artefacts; acceptance a tie; more working calls lost |
| A trained arm passes the full rule | training reaches what no editor reached | no |
| All trained arms fail the guards too | the loss of working calls is shared by both methods at this target | holds |

<!-- src: docs/plans/edit-vs-train-20261108.md "Readings, fixed now"; log 2026-11-09 18:34 UTC "Readings of the frozen table" -->

These readings rest on one training run per arm, one seed and one target. Only the location-matched arm is matched to the edit in size.
<!-- src: plan "Arms" (seed 0); log 2026-11-09 18:34 UTC -->

### J.7 Compute

All arms trained on the same M4 Max Mac and framework as Fable's forward passes. Training ran at about 330 to 380 tokens per second with a peak of 24 to 26 GB. Each 39-call run took about 1 hour (64 to 73 minutes; one run also contains a 63-minute pause between two steps, cause not logged) and processed 443,690 tokens per epoch over 3 epochs. The grids give each 39-call arm three runs, so the two arms took six runs, about 7 hours without the pause. The 411-call run took 7.3 hours (01:04 to 08:19 UTC on 2026-11-09), over 2 epochs of 4,555,306 tokens. Fable's 25 forward-pass requests (19 generation, 6 read) took 10.6 hours of server time, between 17:26 UTC on 2026-11-04 and 13:42 UTC on 2026-11-05. That figure leaves out the editor's own reasoning and analysis. Gating took 2.6 hours for G-yer-39 (08:20 to 10:57 UTC), 2.4 hours for G-tam-39 (10:57 to 13:22 UTC) and 5.1 hours for G-tam-411 (13:22 to 18:27 UTC, including a Mac sleep), each a plain replay followed by a full-call replay. Times are wall-clock and not corrected for sleep.
<!-- src: outputs/weight_edit/claude/edit_vs_train/run_train.log (start/rc times per run); runs/*.log (tok/s, peak GB; yer39_lr1e-3 steps 25→26 at 18:06:42→19:09:09); data/edit_vs_train/MANIFEST.json (tokens); outputs/weight_edit/proposers/fable_r2/state.json passes[] (25 passes: 19 gen, 6 read; sum of secs 38,124 s = 10.59 h; first start_utc 2026-11-04T17:26:23Z, last utc 2026-11-05T13:42:15Z); log 2026-11-09 11:05, 13:24, 18:34 UTC (gate times) -->

## Appendix K. Code, data and the OOD-C data card

This appendix lists what we are preparing for release, with the version each claim was checked on. Links are added only once each piece is public.

### K.1 `weightprog` and the study kit

`weightprog` is a Python library for gradient-free weight edits of dense Gemma 4 text models, written for a coding agent to use. Version 0.2.0 (commit `810116f`, Apache-2.0) has these parts:
<!-- src: weightprog repository, pyproject.toml (version, licence); docs/paper2/notes-fork.md (fact check); Paper 2 §2. -->

- **Edit constructions**, all closed form: a gate on MLP activations paired with a direction in token-score space (the shape of Fable's edit), patches to little-used MLP units (the shape of Opus's), direction ablation, key-to-value updates weighted by the inputs' second moment as in ROME and MEMIT [6, 10], restriction to directions that preserved inputs barely use as in AlphaEdit [9], and residual-direction writes. The last two restrict or penalise changes on reference activations; they do not guarantee preserved generations, so preserved cases must still be measured.
- **Instruments**: captured activations, native log-probabilities, probes and activation patching, with MLX and PyTorch backends.
- **Evaluation**: good, bad and other outputs recorded separately, in paired draws for the base model, the edit and size-matched random edits; per-case sign tests against the base and every control, a group-clustered interval, losses on preserved cases beside the controls' own losses, and a PASS, FAIL or INCOMPLETE verdict.
- **Export** of an edit to a PEFT LoRA adapter [2] whose change equals BA exactly, the form the competition's vLLM server loads. Export does not certify behaviour.

<!-- src: weightprog README.md; src/weightprog/construct/, evaluate.py, cli.py (census, eval); Paper 2 §2 (merge-checked by Codex). -->
The library ships 94 offline tests on tiny models. On the real E4B, five checks at commit `91edc21` (2026-11-07) confirmed base generation, agreement between the two backends and captured edit effects against this study's outputs. Edited generation through these backends and loading exported adapters into vLLM are not yet tested on the real model. The library supports dense Gemma 4 text models only, not the mixture-of-experts variant, and predicts an edit's effect exactly only for the last layer's `down_proj`.
<!-- src: weightprog docs/real-model-validation.md ("What is untested on the real model"); notes-fork.md. -->

A separate study kit, `weightprog_study`, holds this paper's replay, gate and editor sandbox (forward-pass budget server and audit). It has 25 tests and a gate regression that recomputes the round-2 verdicts from the frozen records and scored outputs (`WEIGHTPROG_GATE_DIR`, `WEIGHTPROG_REPLAY_DIR`, `pytest -q weightprog_study/tests/test_gate_regression.py`).
<!-- src: weightprog_study/tests/test_gate_regression.py; Paper 2 §9. [Snapshot commit to be recorded at freeze.] -->

### K.2 The replay test set

The replay set holds the 320 recorded edit calls of the round-2 test, from 101 tasks in 69 held-out repositories. Each record has the prompt, the model's own prefix, the recorded outcome and the file state. It supports both the old-text replay and the full-call replay (Appendix B); on a fidelity subset the replay reproduced 95 of 103 recorded failures and all 213 successes. Frozen scored outputs for every arm in Tables 2 and 3 come with it, so the verdicts can be recomputed without a GPU.
<!-- src: Paper 2 §6; gate_r2 directory listing (gen_*.scored.jsonl per arm). -->

### K.3 OOD-C data card

| Item | Value |
|---|---|
| Tasks | 186 Python bug-fix tasks, at most two per repository |
| Repositories | 148, none shared with the competition's training repositories, OOD-43, OOD-B or the replay calls |
| Sources | SWE-rebench [4, 5]: 109 from the monthly leaderboard splits, 72 from V2, 5 from V1; issues from January 2025 on |
| Funnel | 258 candidates → 198 built → 189 kept → 186 after a Python 3.12 harness-layout check |
| Gradability flags (nested) | 60 pass the build rule (the reference fix resolves twice, the empty patch does not); 58 of those also pass on the merged Python 3.13 layout; 53 also under Python 3.12 |
| Used in this paper | all 186 for agent behaviour; the 53 for solved tasks |
| Size | 14.4 GB with repository snapshots and offline wheels |
| Licences (from the task rows) | MIT 78, Apache-2.0 50, BSD-3-Clause 39, AGPL-3.0 6, BSD-2-Clause 5, Unlicense 3, MIT-0 2, MPL-2.0 1, UPL-1.0 1, GPL-3.0 1 |
| Metadata licence | CC-BY-4.0 (from SWE-rebench); repository code and wheels keep their own licences |

<!-- src: outputs/weight_edit/claude/wp7r/tasks_oodc_final.jsonl (counts, licences); docs/experiments/oodc-build-20261107.md (funnel, flags); scripts/oodc/DATASET_README.md (licence list, total 186); Paper 2 §5. -->

**Known limits.** "Held-out" means that no repository is shared with our stated sets; it does not mean absent from pretraining. Issue texts are as published and were not audited for hints. In the harness's subprocess sandbox an agent can read the dataset directory, gold patches included, so solution-rate evaluations should keep them out of reach. A task is kept when a scripted agent run completes on it, so the set is built for observing what an agent writes; grading needs the flags above.
