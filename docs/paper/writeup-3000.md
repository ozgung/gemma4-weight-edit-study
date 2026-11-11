<!-- Merged Kaggle Writeup, 2026-11-10 (Claude paper session). Sources: docs/paper/writeup-3000-claude.md and docs/paper/writeup-3000-codex.md (sha256 e0630fa0fab7…), both written independently against outline v2 (docs/paper/review/outline-joint-20261110.md) and the joint frame. Numbers: draft-v4-simple.md (claim-checked) and review/figure-data-20261110.md (from the JSON sources). Count: python3 -I scripts/paper/count_words.py docs/paper/writeup-3000.md (cap 2,950). Figures: upload fig1, w2, fig2, fig3, fig4, w5 and fig7 PNGs from outputs/paper/figures/v4/en/ (palette B). Before submitting: verify release links are live; otherwise use the "being prepared for release" wording in the abstract and §5. -->

# Frontier Models as Weight Editors for Gemma 4 E4B

**A gradient-free weight edit cuts a coding agent's escape-code errors from 15 % to 3 % in live runs, and shows what such fixes must also keep**

## Abstract

<!-- Question; target; access; replay result with the preservation failure; live replications; training; tasks; resources. -->
We study whether frontier models can change a coding agent's behaviour by editing its weights directly. The target is one defect of Gemma 4 E4B: when it edits a file, it often copies escape codes such as `\n` into the text it means to replace. Editors could read the weights and send at most 25 forward-pass requests, but could not compute gradients. In a test fixed in advance on 69 held-out repositories, three submitted edits cut these escape artefacts by 23, 53 and 75 %, beyond size-matched random controls, but each also lost 3 to 11 of 213 previously working edit calls in a draw, where our preservation rule allowed 2. Served as a LoRA adapter on 4-bit E4B in vLLM, one selected edit cut the defect in live agent runs on 148 new repositories from 15 % to 3 % of edit calls, and from 7 % to 1 % with thinking on, in two replications planned before their runs, and it ran in the competition's own scorer. Gradient training on the same calls removed more artefacts and failed the same rule. We detected no difference in solved tasks: the harness already repairs many escaped copies. Re-scored afterwards without that repair, fewer calls work, but two of the three edits lose none of them and raise edit acceptance by 4 and 7 points. We release the library, a new task set and the replay test set.

## 1. Introduction

<!-- I1: problem and definition. -->
Gemma 4 E4B is a small open model that works as a coding agent in the Gemma 4 Developer Agent Competition's harness. To change a file, the agent calls an edit tool with the *old text*, copied exactly from the file, and the new text. The model sees files in an escaped form, where a line break is the two characters `\n` and a quote is `\"`, and it often copies that form into the old text (Figure 1). We call each such leftover an *escape artefact*.

![](../../outputs/paper/figures/v4/en/fig1.png)

**Figure 1.** The defect, in a synthetic example: the unedited model copies escape codes into the old text; the edited model writes the real characters.

<!-- I2: fallback and boundary. -->
The harness already softens this defect. When an old text is not found, a fallback unescapes both the old text and its replacement and tries again; in our training runs it rescued 108 of 614 accepted edits. So we did not expect fewer escape artefacts to solve many more tasks. We chose the defect because it is frequent, countable and specific. That makes it a clean test of a narrower question: can a frontier model deliberately change one behaviour of a small model by writing its weights, without breaking the calls that already work?

<!-- I3: prior idea and the three questions. -->
Frontier models have recently been shown to write changes to neural-network weights directly, an approach called weight-space programming [1]. We test the idea on one behaviour of a software-engineering agent, along three separate questions (Figure 2): does the behaviour change, are working calls preserved, and does the agent solve more tasks?

![](../../outputs/paper/figures/v4/en/w2.png)

**Figure 2.** Evidence map: three groups of tests, each with its own metric and edits; arrows mean "tested next". Below: the three outcomes and a deployment check.

<!-- I4: contributions and scope. -->
We contribute (1) a controlled experiment with frontier-model weight editors, held-out repositories and size-matched random controls; (2) tests of one selected edit in live, quantized serving and against gradient training on the same calls; and (3) a library, a task set and a replay set for repeating such tests, described in a companion writeup. The experiment tests particular edits under fixed budgets; it does not map the limits of weight editing.

## 2. Method

<!-- M1: editor access. -->
Each editor worked alone in a sandbox, with the weights and recorded calls from four training repositories but no access to the test data (Figure 3a). Claude Opus 5.5 and Claude Fable 5.1 each had 25 forward-pass requests and GPT-6 Astra had 10. The edits are *gradient-free*: backpropagation and iterative optimisers were not allowed, but closed-form fits on activations, such as least squares, were. A separately registered informed Opus run could read the earlier editors' work and had 10 requests. Two later Fable 5.2 runs tried an informed edit and the repair of a gradient-trained adapter.

<!-- M2: what an edit writes. -->
Each submission is a set of low-rank changes: a weight matrix W becomes W + BA, where B and A are two thin matrices. This form is a LoRA adapter [2], which the competition's vLLM server can load. Fable's edit, for example, is a rank-2 change in the last layer's feed-forward output. One part detects activations just before an escaped punctuation token, and the other shifts token scores toward the raw form; both are fitted in closed form. Opus instead turned about 40 little-used feed-forward neurons into detectors.

<!-- M3: replay, controls and guards. -->
We judged each frozen edit by replay (Figure 3b). For a recorded edit call, the model regenerates only the old text from the exact original prompt, and the harness's matching code checks it against the file. On a fidelity subset, the replay reproduced 95 of 103 recorded failures and all 213 successes. The test was fixed in advance [3]: 320 calls from 101 tasks in 69 held-out repositories, each generated in two *draws*, one greedy and one at temperature 0.6. Each edit faced the base model and two random controls with the same location, rank and size but a random direction, and its per-task artefact rate had to fall against all three (one-sided sign tests, Holm-corrected across candidates). Three guards limited side effects:

- acceptance, the share of old texts found, averaged over tasks, could fall by at most 1 point;
- among recorded calls that had succeeded and that the base model's replay also matched (213 per draw), which we call *working calls*, at most 2 could be lost in each draw (negative flips [11]); these are accepted old-text matches, not correct bug fixes;
- compile failures among accepted calls could rise by at most 2 points, measured on a replay that regenerates both the old and the new text, because the fallback changes both. This full-call form replaced the registered one before any edit ran on the test calls; the full paper gives both verdicts.

![](../../outputs/paper/figures/v4/en/w3.png)

**Figure 3.** (a) What an editor could and could not do. (b) The replay regenerates only the old text; everything before it stays fixed.

<!-- M4: data and the live metric. -->
The replay calls come from agent runs on two held-out sets, OOD-43 and OOD-B, and the first live run used their 115 tasks. For the planned replications we built OOD-C: 186 tasks from 148 repositories, filtered from SWE-rebench [4, 5], packaged to run offline in the competition harness, and checked by script to share no repository with training or the earlier sets. Only 53 OOD-C tasks meet our full gradability rule: the reference fix passes twice, the empty patch fails, and the reference fix passes again under Python 3.12. In live runs we count edit calls whose old *or* new text carries an artefact, unlike the old-text replay, and leave out calls whose file state could not be rebuilt.

## 3. Results

### 3.1 Targeted change and preservation

<!-- R11: reduction and failed guard together. Evidence: gate_r2_report.json, gate_r2_opusinf_report.json. -->
All three second-round edits reduced the defect, and every one lost more working calls than the rule allowed (Figures 4 and 5a). Of 640 replay generations, 100 of the base model's carried an artefact; Opus's edit left 77, Fable's 47 and informed Opus's 25: cuts of 23, 53 and 75 %, each with p ≤ 0.011 against the base and both of its controls. The six random controls left 99 to 102. But the three edits lost up to 3, 5 and 11 of the 213 working calls in a draw, against a limit of 2, and Fable's (narrowly, −1.01 points) and informed Opus's edits also failed the acceptance guard.

![](../../outputs/paper/figures/v4/en/fig4.png)

**Figure 4.** Every edit cut escape artefacts beyond its own size-matched random controls; the edits differ in size.

<!-- R12: mechanism; round 1 in one sentence. Evidence: weight-edit log 2026-11-07 01:00 UTC transition table. -->
When the base model wrote an escaped old text, the fallback often repaired it and the call succeeded. Writing raw text there gains nothing, and the raw copy is sometimes wrong, for example with a closing brace dropped (Figure 5b). Of the escaped copies that the fallback had rescued, the three edits moved 24, 45 and 67 off the fallback, and 5, 12 and 24 of those failed.

<!-- R15: post hoc re-scoring without the fallback (added 2026-11-11, review session). Evidence: outputs/weight_edit/claude/gate_r2/nofallback_report.json from scripts/weight_edit/gate_r2_nofallback.py (rule no_fallback: base accepted 359; opus_r2 +0.041, 10/0, served-ok lost 0,0, full-call compile 0.0675 against limit 0.0754, p against controls 0.019; fable_r2 +0.073, 17/0, lost 0,0, compile 0.0663, p 7.6e-06; opusinf_r2 +0.093, lost 1,0, compile 0.0764). With the fallback on, the script reproduces all 15,360 stored outcomes. Weight-edit log 2026-11-11 14:34 UTC. -->
The same generations show what that repair hides. Scored again with the harness's own matching code but its unescape fallback switched off (post hoc; nothing is regenerated), the base model's accepted calls fall from 443 to 359 of 640, and its working calls per draw from 213 to 177 and 174. Opus's and Fable's edits then lose none of these, pass all three guards and raise acceptance by 4.1 and 7.3 points (better in 10 and 17 tasks, worse in none; beyond both controls). Informed Opus's edit gains 9.3 points but misses the compile guard by 0.1 point. No agent has run without the fallback.

<!-- R13: the editors' decisions. Evidence: opusinf_r2/final_answer.md (opening verdict, §3 table, §4); records.md F.6 (own search data: artefacts 73 to 13 of 768, lost 2 and 3 on set B, 3 and 5 on set D; gate 100 to 25, lost 11 and 11); astra_r2, fable52inf_r2, fable52rep_r2 final answers; log 2026-11-07 01:00, 2026-11-10 04:10 and 12:20 UTC. The off-record v6 gate sentence moved to the full paper only (2026-11-11). -->
Informed Opus named the lost-call guard as its main risk and submitted anyway: on its own data its edit had lost up to 5 calls per draw, against 11 here, while its artefact cut transferred (82 % there, 75 % here). Astra submitted nothing, and both Fable 5.2 runs withdrew their candidates after measuring side effects themselves. These are single runs, not a ranking of models.

### 3.2 Comparison with gradient training

<!-- R21: the matched question. Evidence: docs/plans/edit-vs-train-20261108.md (frozen before training). -->
Editing by reasoning would be most useful if it needed less data than training. In a comparison fixed in advance, we trained LoRA adapters with gradients to write the correct old text on calls from the training repositories. The matched arm used the 39 labelled calls among those Fable had read, at Fable's location, rank and size. Two larger arms trained all linear layers, on the same 39 calls and on all 411 labelled calls; these two are not size-matched.

<!-- R22: result. Evidence: edit_vs_train/gate_g_report.json. -->
Training removed more artefacts and failed the same rule (Figure 5a). The three arms left 23, 5 and 4 artefacts against Fable's 47; at Fable's location the trained adapter was lower in 13 tasks and higher in 1 (p = 0.002). They lost 13, 5 and 7 working calls in their worse draw. The 411-call arm passed the acceptance and compile guards and still exceeded the limit.

![](../../outputs/paper/figures/v4/en/w5.png)

**Figure 5.** (a) Artefacts left against working calls lost in the worse draw, for three edits and three gradient-trained adapters; all exceed the limit, and only the square is size-matched to Fable's edit. (b) How a raw copy breaks.

<!-- R23: bounded reading. -->
On this target the authored edit showed no data advantage over training, and the loss of working calls was not specific to gradient-free editing. Nor must removing artefacts break calls: the all-layer arms removed more while losing fewer, and without the fallback (Section 3.1) the adapters gain 9 to 16 points of acceptance, though each still misses a guard.

### 3.3 Transfer to the running agent

<!-- R31: selection and serving. Evidence: wp7_w4a16 stage-1 report (go_rule.edit_over_base 0.451). -->
We chose Fable's edit after seeing the replay results, so this step is exploratory. Served as a LoRA adapter on 4-bit (W4A16) E4B in vLLM, it kept its effect: artefacts fell from 102 to 46, close to its 16-bit result, while a random adapter left 100. So the change survives quantized serving, and so does its cost: 7 and 11 working calls lost per draw (5 and 5 in 16-bit; none for the random adapter). Agents with this edit also ran in the competition's scorer without errors, scoring 0.05 with thinking off and on, like our comparable agents without it (0.05 to 0.08).

<!-- R32: exploratory provenance and the OOD-C replication. Evidence: wp7/stage2_report.json; wp7r/report.json; plan docs/plans/wp7r-oodc-run-20261108.md. -->
In live agent runs the edit cut the defect again, on new repositories, under a plan fixed in advance (Figure 6). A first, exploratory run on the 115 OOD-43 and OOD-B tasks gave 4 % of counted edit calls with an artefact, against 18 % for the base and 20 % for a random adapter. Because those repositories had supplied the replay calls, we froze the edit, the analysis, the pass rule and an expected effect, and repeated the three arms on OOD-C. Artefacts fell from 15 % of counted edit calls for the base to 3 % for the edit, with 16 % for the random adapter. Per task, the edit was lower than the base in 34 tasks and higher in 4 (p = 3 × 10⁻⁷), and lower than the random adapter in 27 and higher in 7 (p = 4 × 10⁻⁴). The live gain does not lift the replay's preservation failure.

<!-- R33: thinking on. Evidence: wp7t/report.json; plan docs/plans/wp7t-think-oodc-20261108.md. -->
The agent we submit thinks before it acts; the runs above did not. A second planned replication on the same OOD-C tasks, with thinking on, gave 7 % for the base, 1 % for the edit and 8 % for the random adapter. The edit was lower than the base in 16 tasks and higher in 4 (p = 0.006), and passed against the random adapter too (p = 2 × 10⁻⁴). The two replications ran in separate sessions, so their base rates do not measure what thinking does.

![](../../outputs/paper/figures/v4/en/fig7.png)

**Figure 6.** In live runs the selected edit cut escape artefacts with thinking off and on, and no difference in solved tasks was detected. Shares pool counted calls; the tests compare paired tasks.

### 3.4 Task outcomes

<!-- R41: counts per set, not pooled; not-found rise. Evidence: wp7 stage2; wp7r and wp7t descriptive.json. -->
We detected no difference in solved tasks. For the base, the edit and the random adapter, the first run solved 20, 22 and 20 of 115 tasks; on the 53 gradable OOD-C tasks they solved 7, 9 and 7 with thinking off, and 8, 7 and 7 with thinking on. Comparisons this small cannot show equivalence. On OOD-C, calls whose old text was not found rose with both adapters; its cause is untested.

<!-- R42 (the retry follow-up: 131 of 202 misses follow an earlier miss; two go/no-go checks failed, no third round) is in the full paper only since 2026-11-11. Evidence: retry census (log 2026-11-07 19:50 UTC), gonogo_report.json, mask/report.json. -->

## 4. Related work and scope

<!-- D1: positioning. -->
<!-- Native-rendering arm (tool-side change on the round-2 gate calls): registered before any generation, weight-edit log 2026-11-11 14:34 UTC; result entry 17:30 UTC (review session benchmark-teslim-plani). outputs/weight_edit/claude/gate_r2/nofallback_report_native.json (741ca0ecef07; old-text block identical to 6fd4561a5b21, kept as gate_r2/native/analysis/nofallback_report_native_oldtext_only.json) from scripts/weight_edit/gate_r2_nofallback.py (934955c12cbd) --extra native=gate_r2/native/gen_native.jsonl,gate_r2/native/fullcall/gen_native.jsonl; checked by the paper session 17:4x and 19:2x UTC. Artefacts 78 of 640 (base 100; fewer in 12 tasks, more in 0, sign p 0.0002). Harness rule: accepted 447 vs 443, task macro +0.11 points (repo CI -1.2 to +1.6), better 4, worse 4, p 0.64; working calls lost 1 and 2 of 213 (limit 2, met). No-fallback rule: accepted 382 vs 359, +3.67 points (repo CI +1.3 to +6.9), better 12, worse 1, p 0.0017; lost 0 and 0 of 177 and 174. Opus r2: 77 artefacts; +0.4 / +4.1 points; lost 2,3 / 0,0. Registered reading: descriptive, no claim about which intervention is better. Full-call replay (log 2026-11-11 19:16 UTC): compile failures among accepted, harness 46 of 446 = 0.1031 (base 35 of 444 = 0.0788, limit 0.0988, missed), no_fallback 35 of 390 = 0.0897 (base 20 of 361 = 0.0554, limit 0.0754, missed); opus_r2 0.0835 / 0.0675, fable_r2 0.0909 / 0.0663 (met under both). Accepted and compiling: harness 407 to 398, no_fallback 339 to 353. fullcall_artefacts old/new: base 89/67, native 56/39, opus_r2 65/47, fable_r2 34/17 (registered marks: below 45 and 34; missed). Not in the report (log 19:16 UTC, read after the result): 21 new compile failures, 10 gone; 10 of the 21 in pyjanitor_1741 (6) and pyjanitor_1739 (4); without pyjanitor's 24 generations the guard is met (harness 0.0767 vs limit 0.0893, reproduced by the paper session from the .scored.jsonl files; no_fallback 0.0688 vs 0.0732, from the log). -->
<!-- Barry [12/13]: nathan.rs/posts/weight-space-lp, 2026-11-09, read by the paper session 2026-11-11 19:3x UTC. Verbatim: "With Astra's direction, the LP matches Astra's edit on every benchmark, within noise, while changing 8x fewer weights" (16,291 vs 130,188); LP: minimal-L1 rank-one v with v.h_t <= -delta on looping tokens and |v.h_t| <= eps on productive tokens; "A steering vector does the same job" ("The only difference is delivery."); limits: per-token constraints did not matter much because a constant push fixed loops, "a behavior that changes some tokens and leaves others alone would be a better test" (paraphrased by the fetch tool). Our target: escapes must leave the old text but stay where the file has them (Fable 5.2 v6 corrupted legitimate escapes in new code, log 2026-11-10). -->
Weight-space programming showed that frontier models can write changes to network weights directly, on small networks and a 0.8-billion-parameter model [1]; we add a deployed coding agent, held-out repositories, live runs and a gradient baseline. ROME writes a human-designed rank-one update where a fact is stored [6], and activation steering changes intermediate activations at inference time [7]. No editor tried updates that restrict changes on preserved calls' activations, such as null-space [9] or covariance-weighted [10] edits. A linear program that bounds the edit's effect on preserved tokens matched the loop-removing edit of [1] with 8× fewer changed weights, as did a steering vector; its author names a behaviour that changes some tokens and leaves others alone as the better test [12], and escape artefacts are one. Agent success also depends on its tool interface [8]. In round 1, aimed at whether the old text is found at all, three weight edits gained nothing, while a tool-side change, showing tool results unescaped, raised acceptance by 2.0 points on the same calls (p = 0.031). Registered before generation and run on the round-2 gate calls, the same tool-side change cut artefacts from 100 to 78, like Opus's edit (77), and depended on the matching rule in the same way: with the fallback, acceptance moved by +0.1 points and 1 and 2 working calls were lost per draw; without it, acceptance rose by 3.7 points and none were lost. Unlike Opus's and Fable's edits, it missed the compile guard under both rules: in the replay that regenerates both texts, 10.3 % of accepted calls failed to compile against the base's 7.9 % (limit 9.9 %), and 9.0 % against 5.5 % (limit 7.5 %) without the fallback; the excess sits in two tasks of one repository. A tool-side change has no size-matched random control. We ran no steering comparison.

<!-- D2: limits and timing. -->
The study covers one model, one main target, one run per editor, one live run per arm in each setting, and one training seed. The companion writeup adds a second target: three edits cut E4B's wrong-version pydantic calls on held-out cases from 0.29 to at most 0.04, also failing a preservation check. We chose the target after counting artefacts in the base model's test calls, and the deployed edit after the replay; the replications fix the edit, not those choices. Missing traces and file states reduce the paired samples, and no zero-weight adapter separates serving effects from perturbation effects. The round-2 rule was frozen locally before the first forward pass and published while the search was under way; the live-replication plans were logged before their runs and committed afterwards.

<!-- D3: what carries forward. -->
The lesson that carries forward: a tool's tolerance decides what a behaviour fix is worth. Where the tool already repairs a malformed argument, removing the malformation gains nothing on those calls and can lose some; without the repair, the same edits gain. Preservation must therefore be tested apart from the target change, and live behaviour apart from both.

## 5. Resources

<!-- S1: library and study kit. Evidence: /Users/ozgun/Dev/weightprog README.md, pyproject.toml (Apache-2.0), docs/real-model-validation.md ("What is untested on the real model"). No usability claim. -->
`weightprog` (Apache-2.0, alpha) packages edit constructions, model instruments, size-matched random controls and comparisons, with MLX and PyTorch backends. A [companion writeup](https://www.kaggle.com/competitions/gemma-4-developer-agent-paper/writeups/tools-held-out-tasks-agent-written-weight-edits) describes it and the data below. On the real model its base generation, backend agreement and captured edit effects match the study's outputs; edited generation has run through its MLX backend, and this study's adapters came from its export helper. A separate study kit holds this paper's replay, gate and editor sandbox.

<!-- S2: data and reproducibility. Links only once published. -->
OOD-C packages 186 tasks from 148 repositories for offline agent runs, built from SWE-rebench sources (CC-BY-4.0) with open-source licences only and with gradability flags; with snapshots and wheels it is 14.4 GB. The replay set holds the 320 held-out calls with their prompts and file states. The [full paper](https://github.com/ozgung/gemma4-weight-edit-study/blob/main/paper/paper1-full.pdf) adds round 1 and the retry checks, and the repository holds the protocols, records and audits. Library: [github.com/ozgung/weightprog](https://github.com/ozgung/weightprog); OOD-C: [kaggle.com/datasets/ozgung/g4-oodc](https://www.kaggle.com/datasets/ozgung/g4-oodc); replay set: [kaggle.com/datasets/ozgung/g4-weight-edit-replay](https://www.kaggle.com/datasets/ozgung/g4-weight-edit-replay); repository: [github.com/ozgung/gemma4-weight-edit-study](https://github.com/ozgung/gemma4-weight-edit-study).

## 6. Conclusion

Frontier models wrote gradient-free weight edits that reduced a specific coding-agent behaviour beyond matched controls, and one edit kept that change in the live agent. No edit or gradient-trained adapter met our preservation rule under the competition's harness, and no task benefit was detected. Re-scored post hoc without the harness's repair, two edits met the guards and raised acceptance: the measured cost depends on the matching rule as well as the edit.

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
