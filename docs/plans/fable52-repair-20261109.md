# Repair arm: Fable 5.2 given a gradient-trained adapter (pre-registration)

*2026-11-09, Claude (session deneyler 05c039). Status: **frozen** by the weight-edit log entry that records this file's sha256, written before any pass of this arm and before the informed Fable 5.2 arm (`fable52inf_r2`, "A") has any result. Launch needs the user's go after A finishes. User, about 19:20 UTC: "C ilginç olabilir, bunu A bittikten sonra tekrar değerlendirelim", then "evet" to drafting it now.*

## Question

Can a frontier editor, given a gradient-trained LoRA adapter as its starting point, turn it into an edit that passes the full round-2 rule? Concretely: can it keep most of the adapter's cut in escape artefacts while repairing the side effects that the adapter fails on, namely previously correct copies broken (served-ok) and full-call compile failures?

## What is known, and how it shaped this design (disclosed)

- On the round-2 gate set, every edit and every trained adapter failed at least one guard. Every arm failed served-ok (log 2026-11-09 18:34 UTC).
- The starting adapter is G-tam-39 (rank 8 on all 258 editable matrices, trained by gradient descent on Fable's 39 labelled search calls). On the gate set it had 5 artefacts (base 100, `fable_r2` 47) and was non-inferior on acceptance (+1.1 points). It failed served-ok (4 and 5 lost, limit 2) and the compile guard (0.107 against base 0.079, limit 0.099).
- This arm was chosen knowing those gate-set figures. **The editor sees none of them**: no held-out number or file enters the packet, the brief or `prior/`. Its brief states the same rule every editor saw.
- **Why G-tam-39 and not G-tam-411.** G-tam-411 failed only served-ok, but it was trained on all 411 labelled search calls, so any breakage of previously correct copies would be in-sample on the editor's data. G-tam-39 leaves 691 of the 730 search calls out of its training, so the editor can measure its side effects there.

## Design: the informed Fable 5.2 arm with one factor added

- **Proposer:** `claude-fable-5-2`, effort max, through Claude Code (2.1.321 or later; the runner checks that the CLI recognises the model).
- **Packet:**
  - same as `fable52inf_r2`: dossier rev 3, `search_r2`, `server_spec.md` and the kernel;
  - `prior/` from `round2/prior_rep`. It is `prior_opusinf` with its files unchanged, plus `start_g_tam_39/`: the adapter's 258 `.npz` files copied from `edit_vs_train/arms/G-tam-39/edit`, `train_keys.txt` (39), `val_keys.txt` (32) and a README on how it was made. A short section is added to the prior README. MANIFEST sha256 ca699fa57c1c.
- **Brief:** `round2/brief_fable52rep_r2.md` (f1b01264273e). It is `brief_fable52inf_r2.md` (993e9bca0988) with three changes: the run id, the header line, and one new section, "Repair arm: a gradient-trained starting adapter". That section says what the adapter is and that the editor may submit it unchanged, change it, combine it or ignore it. It also says the no-training rule covers the editor's own work, and which calls are in-sample. The brief contains nothing about the adapter's held-out results.
- **Unchanged:** server v2c (`wsp_server.py` 4ed63ba7) with `--fullcall`; budget 10 passes and the frozen limits; at most 4 counted conversations; the operator infrastructure note; a conversation that ends on a usage limit waits 30 min and does not count.
- **Packet dry run (19:38 UTC):** `diff -rq` against `sandbox/fable52inf_r2` finds that `brief.md`, `prior/README.md` and `prior/MANIFEST.sha256` differ and `prior/start_g_tam_39/` is added. All other files are identical.
- **Runner:** `round2/run_fable52rep.sh` (f0aabdffa363). It is `run_fable52inf.sh` with these changes: the run id, the log folder `round2/fable52rep`, PRIOR `prior_rep`, the gate report `gate_r2_fable52rep_report.json` and SEED_BASE 20261900. Step 0 waits for `fable52inf.done` (A fully gated) and then for a CLI that recognises the model. It does not start the proposer after 2026-11-11 03:00 UTC unless FABLE52REP_LATE_OK=1, which needs the user's word. The run folder `proposers/fable52rep_r2` was initialised at 19:38 UTC, `used` 0.

## Gate (primary, pass/fail)

- The round-2 rule as amended on 2026-11-05, on the same fresh set: 320 calls, 101 tasks, 69 held-out repositories, draws 0 and 1, the same base arm.
- Two matched random controls of the final edit: same modules, rank and per-module norm, seeds from SEED_BASE 20261900.
- The arm is its own family, m = 1: `gate_r2_fable52rep_report.json`, with the full-call compile guard of record in `gate_r2_fable52rep_report_amended.json`.
- **Pass** means all of these hold: the primary (lower per-task artefact rate against base and both controls), acceptance non-inferiority, served-ok preservation in both draws, and the full-call compile guard.
- **Empty `edit/`:** not a candidate; reported as such.
- **G-tam-39 submitted unchanged:** it is gated anyway and reported as "no repair". Its outcome on this set is already known.

## Secondary (descriptive, no pass/fail)

`scripts/weight_edit/rep_vs_start.py` (6d51e774bcfa) compares the final edit with G-tam-39. Its numbers were checked on the existing `fable_r2` and G-tam-39 generations as a stand-in, and the output was discarded. It reports:
- per-task artefacts and acceptance against G-tam-39 (two-sided exact sign test), with bootstrap intervals;
- both arms' guard metrics side by side;
- the label **repaired**: the final edit passes served-ok and the compile guard, and is not worse than G-tam-39 on artefacts.

Also descriptive, from the existing reports: the final edit beside `fable_r2`, G-tam-411 and A.

## Readings, fixed now

| Outcome | Reading |
|---|---|
| The full rule passes | The first edit in this study to pass every guard: an editor turned a trained adapter into a gate-passing edit, so at this target editing adds something to training. Reported with its caveats: a single run, a target chosen after seeing gate results, and a gate set used many times. An agent-level test on OOD-C needs its own registration and the user's go. |
| Fails the rule, but "repaired" holds | The side effects are repaired; the edit fails on the primary or on acceptance. |
| Fails, not "repaired", but fewer served-ok calls lost and a lower compile-fail rate than G-tam-39 | Partial repair: right direction, not enough. |
| Unchanged, or no better than G-tam-39 on the guards | No evidence that an editor can repair a trained adapter's side effects in 10 passes. |
| Empty edit | Not a candidate. |

Comparison with A (`fable52inf_r2`): one factor, the starting adapter, but two single runs, so descriptive only.

## Limits written now

- The gate set has now judged three blind editors, the informed Opus arm, three trained adapters and A. Choosing this arm after seeing G-tam-39's results on that set makes its verdict adaptive. No fresh held-out replay set exists. A pass would be confirmed only at agent level.
- One run, one starting adapter.

## Cost and timing

- Mac only (MLX, the instrument of every gated arm). The torch port is 2.4× slower per call on an L4 and needs a Mac-Colab request bridge that does not exist. The gate must stay on MLX in any case, because generations from different backends are not mixed in one paired analysis.
- Proposer about 11-14 h, then gate plus full call about 5.5 h. If A finishes on 11-10 between 14:00 and 17:00 UTC, the verdict comes about 11-11 10:00-13:00 UTC, before the 11-12 deadline. Mac moves would add to this.
- Claude tokens come from the user's subscription.
