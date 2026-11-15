# Flat-Mars boundary-layer run Ub (no frictional heating), Colab L4

**This is a development experiment, not a reproduction of Thomson & Vallis
(2019) M-I.** The unheated partner of Hb: `--preset dev-mars-gray-flat-bl`
from `06f5356` with `frictional_heating` off (configuration `f01c4203…`),
60 days on a Colab L4 without resuming. See the
[plan and results](../../MARS_MI_PLAN.md) (§2e) and the verdicts in
[Hb's evidence](../dev-mars-gray-flat-bl-heated-T21L25-60d-06f5356-l4/).

- [summary.json](summary.json): the daily series and the per-state sinks
  over days 40–60 (sponge and hyperdiffusion only).
- [series.json](series.json): includes `bl_kinetic_sink`, the boundary
  layer's kinetic-energy sink, which M4 adds to the state sinks
  (−0.1037 W m⁻² over days 41–60).
- [climate.png](climate.png).
