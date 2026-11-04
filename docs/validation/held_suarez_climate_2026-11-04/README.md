# Held–Suarez climate figure of `hs-T42L20-prod-002` (2026-11-04)

These are the six latitude–σ climate sections of the production run
`hs-T42L20-prod-002`, drawn on the CPU by `Simulation.plot()`
([CLIMATE_FIGURE.md](../../held_suarez/CLIMATE_FIGURE.md)). The run's archive
is not published (217 MB of states in a git-ignored directory). Only the
figure and its numbers are here.

| file | what |
| --- | --- |
| `hs-T42L20-prod-002_climate.png` | the figure (A4, 200 dpi); its PNG metadata name the PDF, config and states hashes |
| `hs-T42L20-prod-002_climate.pdf` | the same figure as a PDF (filled bands rasterized, lines and text vector) |
| `hs-T42L20-prod-002_climate.json` | sidecar: every number drawn, the 100 state files with their SHA-256, grid, bandlimit, scales |

Command (CPU only, plotting code at the clean commit `8225ba93`):

```python
from tropoi.representation.archive import open_simulation

sim = open_simulation("runs/hs-T42L20-prod-002")
sim.plot("docs/validation/held_suarez_climate_2026-11-04/"
         "hs-T42L20-prod-002_climate.png", sidecar=True)
```

What was averaged:

- the default window `200 < day <= 1200`;
- all 100 saved states, every 10 days from day 210 to day 1200, coverage
  complete;
- equal weights.

Measured on the input: `m_max` = 42 < 64 = nlon/2, and the buffer degree 43 is
exactly zero in every state.

This draw took 61.4 s wall time on the development laptop's CPU, 56.7 s of
it reduction (reading and hashing 100 states and synthesizing u, v, T). Peak
working set was 202 MiB, against 60 MiB after imports. Other draws took
50–65 s. The sidecar values are identical across draws. The PNG bytes are
not guaranteed reproducible across processes.

What this shows: the run's climate as these definitions give it. The
reduction recovers the production accumulators to round-off on a real GPU
fixture (`tests/fixtures/hs-T21L10-dev4`).

What it does not show: physical validation. The run's acceptance verdicts
(tiers A, B, C) are recorded separately in
[STATUS.md](../../held_suarez/STATUS.md) against criteria fixed in advance, and
this figure adds no acceptance claim. Its fixed scales were chosen to cover this
run and are not a reference climatology.
