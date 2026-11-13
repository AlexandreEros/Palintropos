# Flat-Mars implicit-exchange run Hi (frictional heating on), Colab T4

**This is a development experiment, not a reproduction of Thomson & Vallis
(2019) M-I.** It ran `--preset dev-mars-gray-flat-sponge-implicit` from
`63d05a2` (configuration `18ee5dc5…`): the sponge rerun with the
backward-Euler surface sensible exchange. It ran 60 days on a Tesla T4
without resuming and took 14.1 minutes. The GPU is recorded in
`events.jsonl`; the cell asked for an L4. Criteria and predictions were
committed in `6701d02` before the code. See the
[plan and results](../../MARS_MI_PLAN.md) (§2d).

- [summary.json](summary.json): the daily series and the per-state sinks
  over days 41–60.
- [tides.json](tides.json): the M5 sample. It is the 2-sol continuation
  past day 60 and also the N1 test, which stayed finite over all 295 steps.
- [verdicts.json](verdicts.json): M1–M6 over the 41–60 window. M1 and M6
  are not met, against the prediction. M5 is not met on the 11–16 h window,
  as predicted.
- [state_diagnostics.json](state_diagnostics.json): from every saved
  state, where max |u| sits, the per-level maxima, the lowest-level winds,
  the exchange stiffness k w / C_a and the ln p_s mass terms. It comes from
  `tools/radiation/mars_state_diagnostics.py`.
- [climate.png](climate.png).
- The [preserved run capsule](../../../runs/dev-mars-gray-flat-sponge-implicit-heated-T21L25-63d05a2/)
  holds the receipt for all 126 original files and the Drive index.
