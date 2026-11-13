# Flat-Mars sponge rerun Hs (frictional heating on), Colab L4

**Development experiment, not a reproduction of Thomson & Vallis (2019) M-I.**
`--preset dev-mars-gray-flat-sponge` from `e38890e` (configuration
`d2e44058…`): the pilot plus M-I's top sponge. It ran 30 days on an NVIDIA
L4 without resuming, in 6.3 minutes. Predictions were committed in `2bdf13f`
before the sponge code; see the [plan and results](../../MARS_MI_PLAN.md).

- [summary.json](summary.json): per-state sinks, sponge included.
- [tides-last2sols.json](tides-last2sols.json) and
  [verdicts-last2sols.json](verdicts-last2sols.json): **the valid M5 sample
  and verdicts.** These are the last two sols of the run, replayed from its
  day-20 checkpoint and verified against its saved daily states (≤ 6 × 10⁻¹⁰).
- [tides.json](tides.json) and [verdicts.json](verdicts.json): the first
  Colab sample, **superseded and kept as the record**. It continued past
  day 30 into a non-finite state, so its amplitudes are NaN and its
  00:00 maximum is an artifact.
- [climate.png](climate.png).
- The [preserved run capsule](../../../runs/dev-mars-gray-flat-sponge-heated-T21L25-e38890e/)
  holds the receipt for all 66 original files and the Drive index.
