v001: ref_flexonafft_v2 with Google's recommended Gemma 4 sampling (temperature 1.0, top_p 0.95, top_k 64; model card "Best Practices"). One change only. Submitted 2026-10-25 untested beyond one smoke-15 run, at the user's choice, to use the daily slot.

E4B copy of `submissions/v001_gemma_sampling` (2026-10-29, small-model track): only the model name changed in every agent file; budgets, prompts and sampling are identical.

e001n: e001 with thinking off (thinking_budget 0), the base for a thought-free SFT adapter (2026-10-29).

e001nt: e001n with temperature 0.6 instead of 1.0 (2026-10-29, decided overnight). One change only. Base E4B mostly finds the right file and runs Python after its edit but writes the wrong fix, so a lower temperature may give more careful edits.
