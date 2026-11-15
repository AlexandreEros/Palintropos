# Flat-Mars Monin–Obukhov run Hm (frictional heating on), Colab T4

**This is a development experiment, not a reproduction of Thomson & Vallis
(2019) M-I.** It ran `--preset dev-mars-gray-flat-bl-most` from `8e98d7b`,
clean tree (configuration `8cc1ade2…`): the §2e boundary layer with the
surface exchange and the diffusivity switched from the FHZ06-simplified
forms to the full Monin–Obukhov ones, `monin_obukhov: {}` — every Isca
module default, which is what grey-Mars itself sets. Both substitutes stay
removed (`convection: null`, `drag: null`) and the sponge and implicit
exchange stay on.

It ran 60 days on a Tesla T4 and completed, resuming twice through the
day-2 checkpoint; the last leg carried days 2–60 in 32.5 minutes. The GPU
and the resumes are in `events.jsonl`.

**No criterion was registered before this run, and none is judged here.**
M1–M7 are historical and are not requirements from here on
(`MARS_MI_PLAN.md` §4.1). What this directory holds is a measurement:
what the scheme swap does to a 60-day flat-Mars climate, reported beside
the run it replaces. The comparison is
[`MONIN_OBUKHOV.md` §10](../../MONIN_OBUKHOV.md); its counterpart with
FHZ06 is [run Hb](../dev-mars-gray-flat-bl-heated-T21L25-60d-06f5356-l4/README.md).

- [summary.json](summary.json): the daily series and the days 41–60 window.
- [series.json](series.json): the 61 daily records, including
  `max_wind_bottom`, `bl_max_depth` and `bl_kinetic_sink` — the series §2e
  added so the boundary layer can be read from the series alone.
- [config.json](config.json): the full configuration and its hash.
- [means.npz](means.npz), [climate.png](climate.png): the days 41–60 means.
- [events.jsonl](events.jsonl): creation, checkpoints, resumes, completion.

**Energy, on the criteria of [the radiation report](../../README.md)**
("Acceptance criteria and supported regimes"), with F = 99.18 W m⁻² the
window-mean throughput. A1 is not applicable: it asks for a 100-day
window and this run's is 20 days.

| | measured | bar | |
| --- | --- | --- | --- |
| A2 \|mean window residual\| / F | 1.48 × 10⁻⁴ | ≤ 10⁻³ | within |
| A3 largest daily \|residual\| / F | 1.56 × 10⁻⁴ | ≤ 10⁻² | within |
| A4 \|relative drift of mean p_s\| | 2.72 × 10⁻⁵ | ≤ 10⁻⁵ | **outside** |

A2 and A3 are indistinguishable from run Hb's 1.49 × 10⁻⁴ and
1.59 × 10⁻⁴: the scheme swap does not perturb the energy budget. A4 is
outside the bar in both, and slightly further out here (Hb:
2.57 × 10⁻⁵) — the ln-p_s mass drift of `MARS_MI_PLAN.md` §4.2 is a
property of the dynamical core, and the full Monin–Obukhov forms do not
reduce it.
