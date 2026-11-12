<!-- Merged Kaggle Writeup, 2026-11-10 (Claude paper session). Sources: docs/paper/writeup-3000-claude.md and docs/paper/writeup-3000-codex.md (sha256 e0630fa0fab7…), both written independently against outline v2 (docs/paper/review/outline-joint-20261110.md) and the joint frame. Numbers: draft-v4-simple.md (claim-checked) and review/figure-data-20261110.md (from the JSON sources). Count: python3 -I scripts/paper/count_words.py docs/paper/writeup-3000.md (cap 2,950). Figures (2026-11-12, after the cut): upload fig1, w2, w3, fig4, w5 and fig7 PNGs from outputs/paper/figures/v4/en/ (palette B), edit_map/edit_arch_p1.png and activations/fig_activations_simple.png, in the order Figure 1 fig1, 2 w2, 3 edit_arch_p1, 4 w3, 5 fig4, 6 fig_activations_simple, 7 w5, 8 fig7. Before submitting: verify release links are live; otherwise use the "being prepared for release" wording in the abstract and §5. -->

# Frontier Models as Weight Editors for Gemma 4 E4B

**A gradient-free weight edit cuts a coding agent's escape-code errors from 15 % to 3 % in live runs, and shows what such fixes must also keep**

## Abstract

<!-- Question; target; access; replay result with the preservation failure; live replications; training; tasks; resources. -->
Can frontier models change a coding agent by writing its weights without gradients? We target Gemma 4 E4B's habit of copying escape codes into the text it wants to replace. In a test fixed in advance on 320 recorded edit calls from 69 held-out repositories, three edits cut this defect by 23–75 % against the unedited model, while size-matched random edits left it unchanged. Each also lost 3–11 of 213 previously working calls in the worse of two generation runs, where our preservation rule allowed 2 per run. One selected edit, served as a LoRA adapter on 4-bit E4B, cut the defect from 15 % to 3 % of live edit calls, and from 7 % to 1 % with thinking (reasoning before each action) on, in replications planned before their runs. Gradient training removed more artefacts but failed preservation too. We detected no solved-task benefit: the harness already repairs many escaped copies. Post-hoc rescoring without that repair shrinks the working-call subset; two edits preserve it and raise the share of old texts found by 4 and 7 points. We release tools, tasks and recorded calls.

## 1. Introduction

<!-- I1: problem and definition. -->
Gemma 4 E4B is a small open model (about 4.5 billion effective parameters, 42 layers). In the Gemma 4 Developer Agent Competition on Kaggle, a *harness* gives it tools to read and edit a Python repository and scores the hidden bug-fix tasks it resolves. To edit a file, the agent supplies *old text* to find and new text to replace it. Tool results show line breaks as the two characters `\n`, and quotes as `\"`; E4B often copies these codes into the old text instead of the real characters (Figure 1). We count a generated text as carrying an *escape artefact* when it contains either code and the file contains no such sequence.

![](../../outputs/paper/figures/v4/en/fig1.png)

**Figure 1.** A synthetic example: the unedited model copies escape codes; the edited model writes real characters.

<!-- I2: fallback and boundary. -->
The tool already repairs many such mistakes. Its *fallback* unescapes both old and new text when matching fails, then retries. It runs in every agent run and the scorer, rescuing 108 of 614 accepted edits on the training repositories. So we did not expect fewer artefacts to solve many more tasks. We chose this frequent, countable defect to ask a narrower question: can a frontier model change one behaviour while preserving calls that already work?

<!-- I3: prior idea and the three questions. -->
Writing weights directly is called *weight-space programming* [1]. We separate three questions (Figure 2): does behaviour change, are working calls preserved, and are more tasks solved?

![](../../outputs/paper/figures/v4/en/w2.png)

**Figure 2.** Evidence map: replay, quantized serving and live runs; arrows mean “tested next”. Fable is Claude Fable 5.1; OOD-C is our new set of 186 tasks.

<!-- I4: contributions and scope. -->
We add controlled editor experiments, live transfer and a gradient baseline. These fixed-budget tests do not establish editing's limits.

## 2. Method

<!-- M1: editor access. -->
The study ran in two rounds; this paper reports round 2, which targeted escape artefacts, and summarises round 1 below. Each editor worked alone in a sandbox (Figure 4a) with weights and recorded calls from four training repositories: fastapi, rich, requests and httpx. It could read any weight but had no test data. One *forward-pass request* either regenerated old text on up to 128 calls, returning the share of old texts found (acceptance) and artefact rates, or read activations and log-probabilities on up to 32 texts. Claude Opus 5.5 and Claude Fable 5.1 had 25 requests each; GPT-6 Astra had 10. A separately registered (planned in advance) *informed* Opus run could read the earlier editors' notebooks, answers and edits and had 10 requests. Each could submit one edit or none. *Gradient-free* means backpropagation and iterative optimisers were forbidden; closed-form fits, including least squares, were allowed. Two later Fable 5.2 runs, 10 requests each, tried an informed edit and repairing a gradient-trained adapter's losses.

Round 1 targeted old-text matching generally. Opus and Fable had 10 requests each; a fixed recipe of ours, a scripted difference-of-means ablation, supplied a third edit. None beat matched random edits on 225 held-back training-repository calls or changed escape artefacts, motivating round 2.

<!-- M2: what an edit writes. -->
Each edit changes one or more weight matrices W to W + BA, with thin matrices A and B: LoRA form [2] (Figure 3). Fable's rank-2 edit acts in the last layer's feed-forward output matrix. A reads two detectors from activations, one before an escaped line break and one before an escaped quote; B raises the scores of raw characters. Both are fitted in closed form. Opus instead repurposed 40 last-layer neurons it judged little used, switched on by a rank-1 "flag" written in layer 40 (second-to-last, numbered 0–41); informed Opus used 80 such neurons, and the withdrawn informed Fable 5.2 candidate 86.

![](../../outputs/paper/figures/edit_map/edit_arch_p1.png)

**Figure 3.** Where the edits sit in E4B: (a) the 42 layers, with each edit on its layer; (b) one layer, with the matrices each edit changes. Layers 24–41 reuse earlier layers' keys and values, so they have no key or value weights to edit.

<!-- M3: replay, controls and guards. -->
We test each frozen edit by replay: the model regenerates old text from the original prompt and fixed preceding context (Figure 4b). *Acceptance* means a successful match under the tool's rules. The *gate*, a test fixed in advance [3], uses 320 calls from 101 tasks in 69 repositories disjoint from editor data. Each call has two *draws*, one greedy and one at temperature 0.6, giving 640 generations. Each edit faces the base and two random controls matched in location, rank and size (Frobenius norm of the weight change), but with random directions. A task's artefact rate averages its calls and both draws. It must fall against all three references in one-sided exact sign tests over tasks, Holm-corrected across Opus and Fable; informed Opus was registered separately. Registration means a timestamped plan in the study log before testing. Three guards limit side effects:

- Task-averaged acceptance may fall at most 1 percentage point.
- *Working calls* are calls that succeeded when recorded and that the base model's replay also matched (213 per draw). At most 2 may be lost per draw (*negative flips* [11]); an accepted match is not a correct bug fix.
- Among accepted calls, *compile failures* in the edited Python file may rise at most 2 points. For this guard only, a second replay regenerates both texts, because the fallback changes both; this form replaced the registered one, which kept the recorded new text, before any edit ran on the test calls. It turns all three edits' compile verdicts from fail to pass but changes no overall verdict (full paper).

![](../../outputs/paper/figures/v4/en/w3.png)

**Figure 4.** Editor access and frozen-context replay. Base replay reproduces 95/103 recorded matching failures and all 213 successes; four other calls failed otherwise, three through multiple matches.

<!-- M4: data and the live metric. -->
OOD means out of distribution: repositories outside training. OOD-43 has 45 tasks from 17 repositories, 43 gradable; OOD-B has 82 from 70 repositories. Five E4B runs supplied 209 usable calls from OOD-43 and 111 from OOD-B: each has verified context, file state and old text. The first live test used 44 OOD-43 and 71 OOD-B tasks; the rest could not run or be graded on our setup. OOD-C, built from SWE-rebench [4, 5], adds 186 tasks from 148 repositories disjoint from both sets and training. Only 53 are *gradable*: the dataset's reference patch passes twice, the empty patch fails, and the reference passes again under Python 3.12. A task is *solved* when the agent's patch passes hidden tests. Live runs count calls with an artefact in either text, leaving out calls whose file state could not be rebuilt.

## 3. Results

### 3.1 Targeted change and preservation

<!-- R11: reduction and failed guard together. Evidence: gate_r2_report.json, gate_r2_opusinf_report.json. -->
All three submitted edits cut artefacts and failed preservation (Figures 5, 7a). Of 640 base generations, 100 carried an artefact; Opus left 77, Fable 47 and informed Opus 25: reductions of 23, 53 and 75 %. Each beat the base and both controls (p ≤ 0.011); controls left 99–102. Yet the edits lost 3, 5 and 11 working calls in their worse draw, exceeding 2. Fable narrowly failed acceptance (−1.01 points), and informed Opus failed it too.

![](../../outputs/paper/figures/v4/en/fig4.png)

**Figure 5.** Artefact reductions against each edit's two matched controls. Edits differ in size; this is not a model ranking.

<!-- Activations (added 2026-11-12; review session benchmark-teslim-plani, weight-edit log 2026-11-12 14:46 UTC; descriptive, post hoc, nothing registered). Evidence: outputs/paper/figures/activations/fig_activations_numbers.json (943c1c441db5) from scripts/paper/make_figure_activations.py (b3d3b3f0155f); capture scripts/paper/activation_capture.py (80e83b478c3d); positions.npz e4ed4a82edb1, tokens.jsonl a120333b83a1, summary.json 27f2caa0d359; fig_activations.png d7c2d9bef9c1. Checked by the paper session 2026-11-12 against the numbers file: 120 calls (all 48 with a base greedy artefact + 72, seed 20261112, at most 2 per task), teacher-forced along the base greedy old text; base top-1 = written token at 0.9994; positions escape 303, other 14,255 (+92 uncounted backslash tokens, reported apart); top-token changes at escape positions fable 177, trained (G-yer-39) 133, rand 0/0; at other positions 2 / 16 / 1 / 5; first escape of each text 30 / 36 of 48; detector values at other positions p10-p90 -0.048..+0.039; by code value 1 median 0.647 before \n (107), value 2 median 0.777 before \" (196); write size relative to MLP output, median at other positions 0.02 (fable) vs 0.306 (trained); panel (d): \" -0.25 (from 0.33), """ +0.17, " +0.08. Gate: G-yer-39 23 artefacts and 12/13 lost, Fable 47 and 5/5. One capture, greedy text, no intervals. -->
A post-hoc check on 120 test calls shows how narrowly Fable's edit acts (Figure 6). Feeding in the unedited model's own old text (teacher forcing), it changes the most likely next token at 177 of the 303 positions where an escape code was written, but at only 2 of 14,255 ordinary positions, as rarely as two random changes of the same size (1 and 5). A gradient-trained change of the same matrix, rank and size (Section 3.2) changes 133 and 16. Along the unedited text, then, the edit acts almost only at escapes; the next paragraph shows how acting there can still lose working calls.

![](../../outputs/paper/figures/activations/fig_activations_simple.png)

**Figure 6.** How often four changes of the last layer's feed-forward output matrix, all of the same rank and size, alter the most likely next token on 120 test calls (post hoc, descriptive; note the two scales). The full paper also shows the edit's two detectors and where it moves probability.

<!-- R12: mechanism; round 1 in one sentence. Evidence: weight-edit log 2026-11-07 01:00 UTC transition table. -->
Removing escapes can still break a copy (Figure 7b). Of 84 base copies rescued by fallback, Opus, Fable and informed Opus moved 24, 45 and 67 away from fallback, mostly to exact copies; 5, 12 and 24 of these were no longer found. These totals span both draws and include calls outside the working subset, unlike the per-draw preservation losses.

<!-- R15: post hoc re-scoring without the fallback (added 2026-11-11, review session). Evidence: outputs/weight_edit/claude/gate_r2/nofallback_report.json from scripts/weight_edit/gate_r2_nofallback.py (rule no_fallback: base accepted 359; opus_r2 +0.041, 10/0, served-ok lost 0,0, full-call compile 0.0675 against limit 0.0754, p against controls 0.019; fable_r2 +0.073, 17/0, lost 0,0, compile 0.0663, p 7.6e-06; opusinf_r2 +0.093, lost 1,0, compile 0.0764). With the fallback on, the script reproduces all 15,360 stored outcomes. Weight-edit log 2026-11-11 14:34 UTC. -->
The same generations show what the repair hides. We rescored the same generations post hoc with the fallback switched off. Base accepted calls fall from 443 to 359 of 640, and working calls from 213 per draw to 177 and 174. Opus and Fable preserve this smaller subset, pass all three guards and raise task-averaged acceptance by 4.1 and 7.3 points. They improve 10 and 17 tasks respectively, worsen none, and beat both controls in sign tests. Informed Opus gains 9.3 points but misses the compile guard by 0.1 point. No agent has run without fallback.

<!-- R13: the editors' decisions. Evidence: opusinf_r2/final_answer.md (opening verdict, §3 table, §4); records.md F.6 (own search data: artefacts 73 to 13 of 768, lost 2 and 3 on set B, 3 and 5 on set D; gate 100 to 25, lost 11 and 11); astra_r2, fable52inf_r2, fable52rep_r2 final answers; log 2026-11-07 01:00, 2026-11-10 04:10 and 12:20 UTC. The off-record v6 gate sentence moved to the full paper only (2026-11-11). -->
Informed Opus named preservation as its main risk: on its own training-repository calls it had lost up to 5 per draw, versus 11 here, while its artefact reduction transferred (82 % there, 75 % here). Astra submitted nothing after finding no effect on its own held-aside tasks; both Fable 5.2 runs withdrew after measuring side effects. These single runs do not rank models.

### 3.2 Comparison with gradient training

<!-- R21: the matched question. Evidence: docs/plans/edit-vs-train-20261108.md (frozen before training). -->
In a comparison fixed in advance, we trained LoRA adapters to write correct old text. A call is *labelled* when its recorded old text or unescaped form occurs exactly once in the file: 411 of the 730 training-repository calls the editors could search. One training run used the 39 labelled calls among those Fable read with its requests, matching its location, rank and size. Two others trained all linear layers, one on those 39 calls and one on all 411, without size matching.

<!-- R22: result. Evidence: edit_vs_train/gate_g_report.json. -->
The three adapters left 23, 5 and 4 artefacts and lost 13, 5 and 7 working calls in the worse draw (Figure 7a). Matched training beat Fable on artefacts in 13 tasks versus 1 (p = 0.002). All three failed preservation; the 411-call run passed the other guards.

![](../../outputs/paper/figures/v4/en/w5.png)

**Figure 7.** Artefacts versus lost working calls. Only the square matches Fable's location, rank and size. “Other guards” means acceptance and compilation. Below: a synthetic raw-copy failure.

<!-- R23: bounded reading. -->
On this target, the edit written by reasoning showed no advantage over training on the same 39 calls. Losing working calls is not specific to gradient-free editing, and all-layer training loses fewer than matched training. Post-hoc rescoring without fallback gains 9–16 acceptance points, but each trained adapter still misses a guard: matched training fails working-call preservation and compilation, all-layer training on 39 calls fails compilation, and training on 411 fails working-call preservation.

### 3.3 Transfer to the running agent

<!-- R31: selection and serving. Evidence: wp7_w4a16 stage-1 report (go_rule.edit_over_base 0.451). -->
We selected Fable after seeing replay results, so this choice was exploratory. Served in the competition's vLLM format, with 4-bit weights and 16-bit activations (W4A16), it kept its effect on the replay: artefacts fell from 102 for the quantized base to 46 for the edit; a random adapter left 100. Working-call losses against that base were 7 and 11 per draw, versus 5 in each draw in 16-bit and none for random. This random adapter, Fable's first gate control, is used in all live runs below. Agents with the edit also ran without errors in the competition's scorer, which checks deployment, not improvement.

<!-- R32: exploratory provenance and the OOD-C replication. Evidence: wp7/stage2_report.json; wp7r/report.json; plan docs/plans/wp7r-oodc-run-20261108.md. -->
The first live run was exploratory: on the 115 OOD-43 and OOD-B tasks, artefact rates were 18 % for base, 4 % for edit and 20 % for random (Figure 8). We then froze the edit, analysis and replication rule: lower task rates than both comparators in one-sided sign tests, each p < 0.05. The expected edit/base ratio was 0.2–0.6. On OOD-C, rates were 15 %, 3 % and 16 %. The edit beat base in 34 tasks versus 4 (p = 3 × 10⁻⁷), and random in 27 versus 7 (p = 4 × 10⁻⁴). Live measurements do not retest working-call preservation.

<!-- R33: thinking on. Evidence: wp7t/report.json; plan docs/plans/wp7t-think-oodc-20261108.md. -->
A second planned OOD-C replication enabled *thinking*, reasoning before each action. Artefact rates were 7 % for base, 1 % for edit and 8 % for random. The edit beat base in 16 tasks versus 4 (p = 0.006) and random too (p = 2 × 10⁻⁴). These were separate experiments, not a planned thinking comparison, so their differing base rates do not measure thinking's effect.

![](../../outputs/paper/figures/v4/en/fig7.png)

**Figure 8.** Live artefact rates pool counted calls; the tests compare paired tasks. Solved counts use 115, 53 and 53 tasks; differences of one or two tasks cannot be told apart at these sizes, which does not show equivalence.

### 3.4 Task outcomes

<!-- R41: counts per set, not pooled; not-found rise. Evidence: wp7 stage2; wp7r and wp7t descriptive.json. -->
No solved-task benefit was detected (Figure 8b); these samples cannot establish equivalence. Old-text not-found rates rose for both adapters on OOD-C: without thinking, 27 % for base, 34 % for edit and 32 % for random; with thinking, 25 %, 35 % and 36 %. The cause is untested.

<!-- R42 (the retry follow-up: 131 of 202 misses follow an earlier miss; two go/no-go checks failed, no third round) is in the full paper only since 2026-11-11. Evidence: retry census (log 2026-11-07 19:50 UTC), gonogo_report.json, mask/report.json. -->

## 4. Related work and scope

<!-- D1: positioning. -->
<!-- Native-rendering arm (tool-side change on the round-2 gate calls): registered before any generation, weight-edit log 2026-11-11 14:34 UTC; result entry 17:30 UTC (review session benchmark-teslim-plani). outputs/weight_edit/claude/gate_r2/nofallback_report_native.json (741ca0ecef07; old-text block identical to 6fd4561a5b21, kept as gate_r2/native/analysis/nofallback_report_native_oldtext_only.json) from scripts/weight_edit/gate_r2_nofallback.py (934955c12cbd) --extra native=gate_r2/native/gen_native.jsonl,gate_r2/native/fullcall/gen_native.jsonl; checked by the paper session 17:4x and 19:2x UTC. Artefacts 78 of 640 (base 100; fewer in 12 tasks, more in 0, sign p 0.0002). Harness rule: accepted 447 vs 443, task macro +0.11 points (repo CI -1.2 to +1.6), better 4, worse 4, p 0.64; working calls lost 1 and 2 of 213 (limit 2, met). No-fallback rule: accepted 382 vs 359, +3.67 points (repo CI +1.3 to +6.9), better 12, worse 1, p 0.0017; lost 0 and 0 of 177 and 174. Opus r2: 77 artefacts; +0.4 / +4.1 points; lost 2,3 / 0,0. Registered reading: descriptive, no claim about which intervention is better. Full-call replay (log 2026-11-11 19:16 UTC): compile failures among accepted, harness 46 of 446 = 0.1031 (base 35 of 444 = 0.0788, limit 0.0988, missed), no_fallback 35 of 390 = 0.0897 (base 20 of 361 = 0.0554, limit 0.0754, missed); opus_r2 0.0835 / 0.0675, fable_r2 0.0909 / 0.0663 (met under both). Accepted and compiling: harness 407 to 398, no_fallback 339 to 353. fullcall_artefacts old/new: base 89/67, native 56/39, opus_r2 65/47, fable_r2 34/17 (registered marks: below 45 and 34; missed). Not in the report (log 19:16 UTC, read after the result): 21 new compile failures, 10 gone; 10 of the 21 in pyjanitor_1741 (6) and pyjanitor_1739 (4); without pyjanitor's 24 generations the guard is met (harness 0.0767 vs limit 0.0893, reproduced by the paper session from the .scored.jsonl files; no_fallback 0.0688 vs 0.0732, from the log). -->
<!-- Barry [12/13]: nathan.rs/posts/weight-space-lp, 2026-11-09, read by the paper session 2026-11-11 19:3x UTC. Verbatim: "With Astra's direction, the LP matches Astra's edit on every benchmark, within noise, while changing 8x fewer weights" (16,291 vs 130,188); LP: minimal-L1 rank-one v with v.h_t <= -delta on looping tokens and |v.h_t| <= eps on productive tokens; "A steering vector does the same job" ("The only difference is delivery."); limits: per-token constraints did not matter much because a constant push fixed loops, "a behavior that changes some tokens and leaves others alone would be a better test" (paraphrased by the fetch tool). Our target: escapes must leave the old text but stay where the file has them (Fable 5.2 v6 corrupted legitimate escapes in new code, log 2026-11-10). -->
Prior weight-space programming used small networks and a 0.8-billion-parameter model [1]. ROME edits factual associations [6]; steering changes activations [7]. No editor used null-space [9] or covariance-weighted [10] restrictions on preserved inputs. Barry's linear program, which bounds an edit's effect on preserved tokens, and a steering vector both matched the edit of [1] that stops repeated output; Barry names behaviours that change some tokens and not others as the better test [12]; escape artefacts are one. We ran no steering comparison. Tool interfaces also matter [8]. Rendering tool results in the chat template's native, unescaped form raised round-1 acceptance 2.0 points (p = 0.031). Registered before generation on round-2 calls, it cut artefacts from 100 to 78. Acceptance moved +0.1 points with fallback, losing 1 and 2 working calls per draw, versus 3.7 points without, losing none (both rules registered). It failed the compile guard under both rules: 10.3 % against base 7.9 % (limit 9.9 %), and 9.0 % against 5.5 % (limit 7.5 %) without fallback. This descriptive comparison has no size-matched random control.

<!-- D2: limits and timing. -->
We tested one model, one main target, one run per editor and one per live configuration, with one training seed. We chose the target after counting base test-call artefacts and the deployed edit after replay; planned replications do not undo either selection. Missing traces and file states reduce samples. No all-zero adapter separates serving effects from changed weights. The round-2 rule was committed locally before the first request, pushed during search; replication and training plans were frozen and hashed before running, committed afterwards. In the companion writeup, three edits cut pydantic 2 names in pydantic 1 projects from 0.29 to at most 0.04 of held-out answers; all failed preservation.

<!-- D3: what carries forward. -->
A tool's tolerance determines what fixing a behaviour gains. Measure preservation separately from target change, and task outcomes separately from both.

## 5. Resources

<!-- S1: library and study kit. Evidence: /Users/ozgun/Dev/weightprog README.md, pyproject.toml (Apache-2.0), docs/real-model-validation.md ("What is untested on the real model"). No usability claim. -->
[`weightprog`](https://github.com/ozgung/weightprog) (alpha, Apache-2.0) provides editing, instruments, matched controls and MLX/PyTorch backends; its helper exported these adapters. The [companion writeup](https://www.kaggle.com/competitions/gemma-4-developer-agent-paper/writeups/tools-held-out-tasks-agent-written-weight-edits) documents validation and data.

<!-- S2: data and reproducibility. Links only once published. -->
[OOD-C](https://www.kaggle.com/datasets/ozgung/g4-oodc) supplies offline tasks; the [replay set](https://www.kaggle.com/datasets/ozgung/g4-weight-edit-replay) supplies calls, prompts and file states. The [study repository](https://github.com/ozgung/gemma4-weight-edit-study) holds protocols, audits, replay/editor tools and the [full paper](https://github.com/ozgung/gemma4-weight-edit-study/blob/main/paper/paper1-full.pdf). A [Kaggle notebook](https://www.kaggle.com/code/ozgung/what-a-weight-edit-does-in-gemma-4-e4b) shows token by token where Fable's edit acts, and fits and exports a rank-1 edit.

## 6. Conclusion

Frontier models' edits changed the target behaviour beyond controls, and one kept the change in live runs, but none met preservation under the competition's harness and no task benefit was detected. Post-hoc rescoring without fallback lets two edits pass: an edit's value depends on the tool's matching rule.

## References

1. Takehi, Teoh, Zhang, Khattab. [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/). Blog post, 2026.
2. Hu et al. [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685). ICLR 2022.
3. van Miltenburg, van der Lee, Krahmer. Preregistering NLP Research. NAACL 2021.
4. Badertdinov et al. [SWE-rebench](https://arxiv.org/abs/2505.20411). NeurIPS 2025.
5. Badertdinov et al. [SWE-rebench V2](https://arxiv.org/abs/2602.23866). 2026.
6. Meng et al. [Locating and Editing Factual Associations in GPT](https://arxiv.org/abs/2202.05262). NeurIPS 2022.
7. Turner et al. [Steering Language Models With Activation Engineering](https://arxiv.org/abs/2308.10248). arXiv 2023.
8. Yang et al. [SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering](https://arxiv.org/abs/2405.15793). NeurIPS 2024.
9. Fang et al. [AlphaEdit: Null-Space Constrained Knowledge Editing for Language Models](https://arxiv.org/abs/2410.02355). ICLR 2025.
10. Meng et al. [Mass-Editing Memory in a Transformer](https://arxiv.org/abs/2210.07229). ICLR 2023.
11. Yan et al. [Positive-Congruent Training: Towards Regression-Free Model Updates](https://arxiv.org/abs/2011.09161). CVPR 2021.
12. Barry. [Can a Linear Program Do Weight-Space Programming?](https://nathan.rs/posts/weight-space-lp/). Blog post, 2026.
