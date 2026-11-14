# Flat-Mars boundary-layer run Hb (frictional heating on), Colab L4

**This is a development experiment, not a reproduction of Thomson & Vallis
(2019) M-I.** It ran `--preset dev-mars-gray-flat-bl` from `06f5356`
(configuration `004fd4b8…`): the §2d implicit configuration with the FHZ06
boundary layer in place of the Held–Suarez drag and the dry adjustment. It
ran 60 days on a Colab L4 without resuming. Criteria and predictions were
committed in `04f4547` before the code. See the
[plan and results](../../MARS_MI_PLAN.md) (§2e).

- [summary.json](summary.json): the daily series and the per-state sinks
  over days 40–60 (sponge and hyperdiffusion only; the boundary layer's
  sink is in `series.json` as `bl_kinetic_sink`).
- [tides.json](tides.json): the M5 sample. It is the 2-sol continuation
  past day 60 and also the N1 test, which stayed finite over all 295 steps.
  It was made on Colab with `mars_tides.py` patched in memory to rebuild
  and restore the boundary layer; the same change is now in the tool.
- [verdicts-2e-corrected.json](verdicts-2e-corrected.json): **the judged
  verdict**, `judge_mars_pilot.py --criteria 2e --days 60 --window 41 60`,
  regenerated from this evidence after the M7-window and M4-sink fixes.
  N1 and M2–M4 met; M1, M5, M6 and M7 not met.
- [verdicts-2e-recovery.json](verdicts-2e-recovery.json): the first verdict,
  kept unmodified for provenance (SHA-256 `bea7e011…`). Its M4 left out the
  boundary layer's sink and reports "not met"; it is superseded.
- [climate.png](climate.png).
- The original files are on the user's Google Drive under
  `MyDrive/runs/evidence/`.
