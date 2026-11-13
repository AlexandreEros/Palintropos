# Flat-Mars development pilot H (frictional heating on), Colab L4

**Development experiment, not a reproduction of Thomson & Vallis (2019) M-I.**
`--preset dev-mars-gray-flat` from `b3aaef8` (configuration `9f7eaa0a…`),
30 days at T21 with Isca's 25-level grid, on an NVIDIA L4 (CuPy 14.0.1),
run without resuming in 6.5 minutes. Criteria M1–M6 were committed in
`fba92fb`, before any pilot code existed; see the
[plan and results](../../MARS_MI_PLAN.md).

- [summary.json](summary.json): per-state drag and ∇⁸ sinks, days 10–30.
- [tides.json](tides.json): 2-sol sun-synchronous S1/S2 fit after day 30.
- [verdicts.json](verdicts.json): M1–M6.
- [climate.png](climate.png).
- The [preserved run capsule](../../../runs/dev-mars-gray-flat-heated-T21L25-b3aaef8/)
  holds the receipt for all 66 original files and the Drive index.
