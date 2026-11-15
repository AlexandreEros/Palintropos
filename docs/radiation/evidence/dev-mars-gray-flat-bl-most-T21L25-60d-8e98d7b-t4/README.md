# Flat-Mars Monin–Obukhov run Um (frictional heating off), Colab T4

**This is a development experiment, not a reproduction of Thomson & Vallis
(2019) M-I.** It is run Hm with one key changed: frictional heating off.
It ran `--preset dev-mars-gray-flat-bl-most --no-frictional-heating` from
`8e98d7b`, clean tree (configuration `2069b1c3…`). The two configurations
differ in that key and nothing else, so the pair isolates the heating, as
the R1/R2 pair does for Earth.

It ran 60 days on a Tesla T4 and completed, resuming twice through the
day-2 checkpoint; the last leg carried days 2–60 in 29.6 minutes, starting
two seconds after run Hm finished on the same GPU.

**No criterion was registered before this run, and none is judged here**
(`MARS_MI_PLAN.md` §4.1). The comparison is
[`MONIN_OBUKHOV.md` §10](../../MONIN_OBUKHOV.md); its counterpart with
FHZ06 is [run Ub](../dev-mars-gray-flat-bl-T21L25-60d-06f5356-l4/README.md).

- [summary.json](summary.json): the daily series and the days 41–60 window.
- [series.json](series.json): the 61 daily records, including
  `max_wind_bottom`, `bl_max_depth` and `bl_kinetic_sink`.
- [config.json](config.json): the full configuration and its hash.
- [means.npz](means.npz), [climate.png](climate.png): the days 41–60 means.
- [events.jsonl](events.jsonl): creation, checkpoints, resumes, completion.

**Energy**, with F = 99.17 W m⁻² the window-mean throughput (A1 not
applicable, as for Hm):

| | measured | bar | |
| --- | --- | --- | --- |
| A2 \|mean window residual\| / F | 3.04 × 10⁻³ | ≤ 10⁻³ | **outside**, by 3× |
| A3 largest daily \|residual\| / F | 3.47 × 10⁻³ | ≤ 10⁻² | within |
| A4 \|relative drift of mean p_s\| | 2.34 × 10⁻⁵ | ≤ 10⁻⁵ | **outside** |

This is the expected unheated signature and not a defect of the scheme:
run Ub, the same configuration with FHZ06, misses A2 by the same margin
(2.87 × 10⁻³). The dissipation the boundary layer and the sponge remove is
simply not returned. Its size here — about 0.3 W m⁻² against 0.015 in Hm,
a factor of 20 — is the same separation the Earth pair shows, and the
reason frictional heating is the preset default.
