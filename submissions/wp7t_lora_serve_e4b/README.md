v001: ref_flexonafft_v2 with Google's recommended Gemma 4 sampling (temperature 1.0, top_p 0.95, top_k 64; model card "Best Practices"). One change only. Submitted 2026-10-25 untested beyond one smoke-15 run, at the user's choice, to use the daily slot.

E4B copy of `submissions/v001_gemma_sampling` (2026-10-29, small-model track): only the model name changed in every agent file; budgets, prompts and sampling are identical.

e001n: e001 with thinking off (thinking_budget 0), the base for a thought-free SFT adapter (2026-10-29).

e001nt: e001n with temperature 0.6 instead of 1.0 (2026-10-29, decided overnight). One change only. Base E4B mostly finds the right file and runs Python after its edit but writes the wrong fix, so a lower temperature may give more careful edits.

e001tt (E4B-Think-T0.6): e001nt with thinking back on (thinking_budget 4096, e001's value), 2026-11-02. One change only. Thinking against no thinking was a tie at T 1.0 (e001 vs e001n: OOD-43 12 vs 13, winnable-91 13 vs 10); T 0.6 was then tested only with thinking off, so this cell was never measured. E4B uses about a third of the runtime limit, so the extra tokens fit.

wp7t_lora_serve_e4b, 2026-11-08: WP7T server manifest only (e001tt plus both adapters mounted, fableedit and randedit); never run as an arm.
