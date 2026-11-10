# Any pressure in the browser: interactive-pressure mode (2026-11-08)

Measurements for the second slice of
[issue #13](https://github.com/AlexandreEros/Palintropos/issues/13): the
atmosphere page exports the native columns of each saved state and
interpolates them to any pressure chosen offline in the browser
(`Simulation.view(surface=("pressure", Pa), interactive_pressure=True)`).
The method is described in
[ATMOSPHERE_VIEW.md](../../../src/tropoi/representation/visual/ATMOSPHERE_VIEW.md#any-pressure-in-the-browser-interactive-pressure-mode);
the numbers come from the committed script
[examples/isobaric_view.py](../../../examples/isobaric_view.py) on the same
laptop as the earlier records (Intel Core i5-8250U class CPU, Windows 11,
Python 3.12.12, NumPy 2.2.6; headless Chrome with software WebGL through
`tools/browser_cdp.py`, networking blocked; no GPU is used anywhere).

This is a display of saved states. Nothing here validates the runs as
climates: the Mars run is an idealized dry-CO₂ perturbation experiment and
the Held–Suarez run an idealized forcing benchmark.

| file | what it is |
| --- | --- |
| `costs_mars_300Pa_3deg.json` | per-part export costs on three states of the published Mars run at 3° (`measure`): the fixed-surface parts as before, plus `quantize_columns` and the whole column export (`interactive`) |
| `costs_hs-T42L20-prod-002_500hPa_2deg.json` | the same on three states of the Held–Suarez T42L20 run at 2° |
| `mars_isobars_interactive_bench.json` | the browser bench of the committed Mars page: start-up, five pressures × 5 updates with the page's own split (decode, interpolation, mesh, map), 18 saved-time switches with the decoded-state list, heap, a refused typed value, one request, no errors |
| `hs_isobars_2deg_8states_bench.json` | the same on the 29.3 MB eight-state Held–Suarez page (not committed; rebuilt by the command below) |
| `mars_isobars_interactive_p300Pa.png`, `…_p600Pa.png` | captures of the Mars page at 3 hPa and 6 hPa over Tharsis (the bench also captured 150, 1000 and 50 Pa; not kept) |
| `hs_isobars_2deg_8states_p50000Pa.png` | the Held–Suarez page at 500 hPa (the other three captures not kept) |

## Commands

```bash
MARS=docs/runs/20261027T175327Z_orographic-thermal-wave_rot24p62h_r3_l21_dt11080p3s_3a46f966_71c5441a
python examples/isobaric_view.py interactive $MARS --pressure 300 --resolution 3 --out $MARS/assets/mars_isobars_interactive.html
python examples/isobaric_view.py measure $MARS --pressure 300 --times 0 4 -1 --resolution 3 --out docs/validation/interactive_pressure_2026-11-08/costs_mars_300Pa_3deg.json
python examples/isobaric_view.py bench $MARS/assets/mars_isobars_interactive.html docs/validation/interactive_pressure_2026-11-08 --pressures 300 150 600 1000 50 --centre -5 -115
python examples/isobaric_view.py measure runs/hs-T42L20-prod-002 --pressure 50000 --times 0 60 -1 --resolution 2 --out docs/validation/interactive_pressure_2026-11-08/costs_hs-T42L20-prod-002_500hPa_2deg.json
python examples/isobaric_view.py interactive runs/hs-T42L20-prod-002 --pressure 50000 --resolution 2 --times 0 17 34 51 68 85 102 119 --out out/hs_isobars_2deg_8states.html
python examples/isobaric_view.py bench out/hs_isobars_2deg_8states.html docs/validation/interactive_pressure_2026-11-08 --pressures 50000 25000 85000 12345.6 --exaggeration 20
```

The Held–Suarez run (`runs/hs-T42L20-prod-002`, 120 states every 10 days,
[the run record](../../held_suarez)) is local and not committed; the Mars run
is published under `docs/runs/`. `times="all"` on it is refused at the
default budget: 433 MB at 2°, 194 MB at 3° (the message names the
alternatives; 8 states fit at 2°, 19 at 3°).

## The example: the Mars run, any pressure

The committed page
[`assets/mars_isobars_interactive.html`](../../runs/20261027T175327Z_orographic-thermal-wave_rot24p62h_r3_l21_dt11080p3s_3a46f966_71c5441a/assets/mars_isobars_interactive.html)
(3.64 MB, 1.6 s to export, all 9 saved states on a 3° mesh, 7381 nodes and
4 levels each, 0.60 MB decoded per state) opens at 3 hPa and lets any
pressure be chosen. The slider spans 35 Pa to 10.9 hPa (the pressures some
node of some state samples). At the last saved state:

| pressure | valid | under ground (1) | above top level (2) | below bottom level (3) | triangles of 14,400 |
| --- | --- | --- | --- | --- | --- |
| 300 Pa (3 hPa) | 7334 | 11 | 0 | 36 | 14,278 |
| 150 Pa | 7370 | 0 | 11 | 0 | 14,363 |
| 600 Pa | 3252 | 3479 | 0 | 650 | 6,032 |
| 1000 Pa | 19 | 7319 | 0 | 43 | 21 |
| 50 Pa | 143 | 0 | 7238 | 0 | 235 |

At 3 hPa the counts are those of the fixed-surface page of the first slice
(7334 valid, 11 under ground, 36 in the bottom gap, 14,278 triangles): the
page's interpolation reproduces the Python export. The browser test
`tests/test_interactive_pressure_view.py` compares the page's arrays with
`pe_surfaces.interpolate_columns` on the stored columns at arbitrary
pressures (masks identical, values within float32 storage) and measures the
quantization error apart from the interpolation error.

## What the browser pays

| page | interactive after | JS heap after load / after use | pressure update, page median (decode · interpolate · mesh · map) | round trip | saved-time switch, page median (decode) | round trip |
| --- | --- | --- | --- | --- | --- | --- |
| Mars, 3.64 MB, 9 states × 4 levels | 0.88 s | 19.3 / 19.3 MB | 6.3 ms (0 · 0.9 · 2.5 · 2.6) | 39 ms | 12 ms (4.5 ms) | 25 ms |
| Held–Suarez, 29.3 MB, 8 states × 20 levels | 1.05 s | 50.4 / 50.4 MB | 14.7 ms (0 · 3.5 · 6.6 · 3.9) | 55 ms | 140 ms (112 ms) | 146 ms |

The round trip is what the DevTools protocol sees (JSON over a WebSocket on
both sides) and bounds what a user feels from above; the page's own split
is the work. Interpolation is the cheapest part of a pressure update;
decoding a 20-level state (base64 of 2.6 MB, then float32 conversion)
dominates a saved-time switch on the large page. The decoded-state list in
the bench records never exceeds two entries: the eight Held–Suarez states
would decode to 44 MB, the page holds 50 MB of heap in all.

## Stored precision

Each level slice of each field has its own 16-bit range; the surface
pressure is float32. Worst half-quantum over the exported states (the
largest error the storage can add to a stored value):

| run | temperature | wind components | geopotential height |
| --- | --- | --- | --- |
| Mars (9 states) | 0.02 mK | 0.03 mm/s | 0.12 m |
| Held–Suarez (states 0, 60, 119) | 0.4 mK | 1 mm/s | 0.01 m |

Interpolation is linear, so an interpolated value's error is within the
quanta of its two bracket levels (tested). The float32 surface pressure
(6e−8 relative) is finer than any mask boundary a display can show, but
coarser than the 1e−9 end-level tolerance: a node sitting exactly on an end
level can be valid in one representation and in the tolerance band in the
other; no other mask differs between the exact and the stored columns.
