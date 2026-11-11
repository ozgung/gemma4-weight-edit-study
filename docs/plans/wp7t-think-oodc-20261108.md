# WP7T: the WP7R agent test with thinking on (pre-registration)

*2026-11-08, Claude (session deneyler 05c039). Status: **frozen** by the weight-edit log entry that records this file's sha256, before the kernel is pushed and before any WP7T trace exists. User, about 22:40 UTC 11-08: the competition waits until the paper deadline and the Kaggle quota goes to the paper; "evet, ikisine de başla" (this test and the GPT-6.2-sol arm preparation). WP7R, whose design this copies: `docs/plans/wp7r-oodc-run-20261108.md`.*

## Question

Paper 1 §4 lists as a limit that the edit was tested with thinking off only, while the E4B agent we submit (e001tt E4B-Think-T0.6) runs with a thinking budget of 4096. With thinking on, does the frozen edit `fable_r2`, served as a LoRA on the competition's W4A16 model in vLLM, still lower the escape-artefact rate of live `edit_file` calls on OOD-C, against the base and against its size-matched random adapter?

## Design (WP7R with one factor changed: thinking on)

- **Arms:** e001tt E4B-Think-T0.6 (base, zip 25709533…, the submitted e001tt), e001tte E4B-Think-T0.6-FableEdit (e001tt plus `adapter: fableedit`, adapter sha256 0d5688d7…, byte-identical to WP7R's), e001ttx E4B-Think-T0.6-RandEdit (e001tt plus `adapter: randedit`, `fable_r2_rand1`, sha256 bfcdbdb9…). Zips 355b6f98… and dd964ba4…; server manifest `submissions/wp7t_lora_serve_e4b` (c4d5bd33…), both adapters mounted.
- **Task set:** the same 186 OOD-C tasks as WP7R (ids from WP7R's base `task_results.jsonl`), statement g1, read from the output of `ozgung/g4-oodc-assemble`.
- **Conduct:** as WP7R role a: three arms side by side on one LoRA-enabled vLLM, 4 streams per arm, `max_time_minutes` 12 in every arm (e001tt's own value is 8; WP7 and WP7R used 12), 28 calls, other budgets e001tt's.
- **Changed besides thinking (infrastructure, disclosed):** Kaggle L4x4 (`ozgung/g4-wp7t-think-oodc`, scorer settings, tensor parallel over 4 L4s, internet off) instead of one Colab L4; a session hard stop at 10.5 h that keeps finished tasks (WP7R: 7 h per arm). Built by `scripts/make_wp7t_kernel.py`; the kernel embeds `run_ood45.py` and `wp7_extract_files.py` unchanged.
- At least 150 tasks with a counted call in every arm, or the primary is reported as underpowered and descriptive only.

## Analysis

- **Code:** `scripts/weight_edit/wp7_stage2_analyze.py` with one addition: set `oodct` (session `wp7t`, suite `oodc`) and its arm table `ARMS_THINK`; nothing else changed. It reproduces `outputs/weight_edit/claude/wp7r/report.json` byte for byte on `--set oodc`, and the `oodct` path was exercised on the WP7R traces under the new names (output discarded). Run as `--set oodct --out outputs/weight_edit/claude/wp7t/report.json` after the kernel output is placed in `results/wp7t/results/` and `wp7t_files.tgz` is unpacked into `data/ext/ood_files/wp7/`.
- **Primary (same as WP7R):** per-task live escape-artefact rate over counted `edit_file` calls; tasks paired when both arms have a counted call; task-macro difference; one-sided exact sign test, edit lower, against base and against random. **Pass:** both differences negative and both p below 0.05.
- **Descriptive:** artefact rate of base with thinking on against WP7R's thinking-off base (different sessions, descriptive only); not-found per call; acceptance through the unescape fallback; calls per task; no patch; resolved on the same 53 gradable tasks as WP7R (gold passes on Python 3.12), as graded in the kernel, with a difference under about 7 read as a tie.
- **Readings:** a pass means the edit's effect on what the live agent writes holds with thinking on; the §4 limit is then replaced by the result. A fail with the base rate as low as WP7R's random arm or lower means thinking removes most of the target and the edit has little left to act on; a fail with a base rate near WP7R's (about 0.15) means the edit does not carry over to thinking on. Either fail is reported in §4 as it comes out. No resolve claim either way.

## Expectation, written before the run

Thinking on lowers the base rate (OOD-43 plus OOD-B, earlier sessions: task-macro 0.13 → 0.07-0.10), so on OOD-C the base lands at about 0.08-0.13. The edit/base ratio lands between 0.2 and 0.7; the random arm stays within noise of the base; the primary passes, but with fewer discordant tasks than WP7R's 34/4 (about 15-25 lower against 2-6 higher). Not-found per call: no claim (it rose with both adapters in WP7R).

## Cost

Wall time about 4-7 h (WP7R ran in 5.4 h on one L4 without thinking; thinking adds decode tokens; four L4s add throughput). L4x4 counts double: about 9-15 Kaggle GPU quota hours, at most about 22 with the 10.5 h stop.
