# Browser streamlines and geopotential-height surfaces (2026-11-08)

Measurements for the third slice of
[issue #13](https://github.com/AlexandreEros/Palintropos/issues/13): the
interactive column page traces instantaneous horizontal streamlines in the
browser and offers geopotential-height surfaces next to isobars, switchable
without regenerating the file. The method is described in
[ATMOSPHERE_VIEW.md](../../ATMOSPHERE_VIEW.md#geopotential-height-surfaces).
Same laptop as the earlier records (Intel Core i5-8250U class CPU, Windows
11, Python 3.12, NumPy 2.2; headless Chrome with software WebGL through
`tools/browser_cdp.py`, networking blocked; no GPU used).

These are displays of saved states: the Mars run is an idealized dry-CO₂
perturbation experiment and the Held–Suarez run an idealized forcing
benchmark. Nothing here validates either as a climate.

| file | what it is |
| --- | --- |
| `mars_isobars_interactive_bench.json` | the committed Mars page: start-up, 5 pressures and 5 heights × 5 updates each with the page's own split, 18 saved-time switches, 10 coordinate switches, a slider drag with the deferred trace, rendering, heap, one request, no errors |
| `hs_surfaces_2deg_8states_bench.json` | the same on the 29.3 MB eight-state Held–Suarez page (local run; the page is not committed and is rebuilt by the command below) |
| `mars_isobars_interactive_p300Pa.png`, `…_z0m.png`, `…_z4000m.png` | the Mars page at 3 hPa, at the datum and at 4 km over Tharsis, with streamlines (the other captures the bench writes are not kept) |
| `hs_surfaces_2deg_8states_p50000Pa.png`, `…_z5500m.png` | the Held–Suarez page at 500 hPa and at 5.5 km |

## Commands (the reproducible example)

```bash
MARS=docs/runs/20261027T175327Z_orographic-thermal-wave_rot24p62h_r3_l21_dt11080p3s_3a46f966_71c5441a
python examples/isobaric_view.py interactive $MARS --pressure 300 --resolution 3 --out $MARS/assets/mars_isobars_interactive.html
python examples/isobaric_view.py bench $MARS/assets/mars_isobars_interactive.html docs/validation/derived_surfaces_2026-11-08 --pressures 300 150 600 1000 50 --heights -4000 0 4000 10000 20000 --centre -5 -115
python examples/isobaric_view.py interactive runs/hs-T42L20-prod-002 --pressure 50000 --resolution 2 --times 0 17 34 51 68 85 102 119 --out out/hs_surfaces_2deg_8states.html
python examples/isobaric_view.py bench out/hs_surfaces_2deg_8states.html docs/validation/derived_surfaces_2026-11-08 --pressures 50000 25000 85000 12345.6 --heights 1500 5500 10000 -100 --exaggeration 20
```

Open the committed Mars page from disk: choose any pressure, switch to
**Geopotential height**, choose any height (negative values too), switch
back (each coordinate keeps its value), step or play the saved states. At
heights below about 8 km the surface meets the terrain, and the holes say
why on hover. The streamlines follow every selection.

## Height surfaces cutting the terrain (Mars, last saved state, 3° mesh, 7381 nodes)

The terrain spans −7.8 to +8.4 km on this mesh. The full levels sample
−6.3 to +31.0 km.

| Z₀ | valid | below the terrain (1) | above the top level (2) | between terrain and bottom level (3) | triangles of 14,400 | streamlines |
| --- | --- | --- | --- | --- | --- | --- |
| −4 km | 51 | 6003 | 0 | 1327 | 73 | 1 |
| 0 | 3173 | 3572 | 0 | 636 | 5,881 | 59 |
| 4 km | 6597 | 189 | 0 | 595 | 12,686 | 141 |
| 10 km | 7381 | 0 | 0 | 0 | 14,400 | 158 |
| 20 km | 4704 | 0 | 2677 | 0 | 8,968 | 106 |

On the flat Held–Suarez run every node is valid from 1.5 to 10 km. At
−100 m every node is below the terrain, so the surface is empty, and the
page says so with its controls still live.

## Costs

| page | file | export | interactive after | JS heap | GPU frame |
| --- | --- | --- | --- | --- | --- |
| Mars, 9 states, 3° | 3.71 MB (3.64 before: 39 KB of base64 float32 terrain, the rest a larger page script) | 2.3–4.1 s (two runs) | 1.0 s | 20.5 MB | 0.06 ms |
| Held–Suarez, 8 states, 2° | 29.3 MB (terrain 88 KB) | 6.4 s | 2.8 s | 50.4 MB | 0.11 ms |

Medians of the page's own timer, in ms. The DevTools round trip adds its
own overhead on top.

| page | pressure update | of which interpolation | height update | of which interpolation | coordinate switch | saved-time switch | of which decoding |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Mars | 18.7 | 1.0 | 23.2 | 1.8 | 28.4 | 49 | 5.8 |
| Held–Suarez | 40.1 | 6.0 | 42.8 | 12.2 | 27.8 | 114 | 54 |

**Streamline tracing** is timed apart (`integrate_ms`). A first trace of a
surface, which includes the JIT warm-up of a freshly loaded page, took a
median of 30 ms on Mars and 62 ms on Held–Suarez, at most about 150 ms.
Warm, repeated traces of one Mars surface take 7–18 ms (59–158 lines;
measured through `traceStreamlines`). Pressure updates in the bench
alternate between two values, so their traces come from the cache
(`streams_ms` 0). Saved-time switches trace synchronously, and that cost
is inside their page total.

**A slider drag** (six positions, one per frame): the surface follows each
frame. The previous lines are cleared at once, and the new ones were
published 185 ms (Mars) and 210 ms (Held–Suarez) after the last movement:
the 150 ms settle delay plus one trace. Releasing the slider traces at
once.

Memory: the heap did not grow over the bench (20.5 MB on Mars before and
after; 50.4 MB on Held–Suarez). The decoded-state bound (two states) and the
line cache (8 sets, at most 1.0 MB) hold.

## Numerical agreement (from the tests)

| comparison | result |
| --- | --- |
| browser streamlines vs `integrate_streamlines` on the page's decoded winds and mask | same lines, same points: worst 1e−14 on the unit sphere (fixtures), 1.1e−13° on the Mars run (5 pressures × 2 times, holes included) |
| analytic winds: zonal, over the poles, seam, calm, one-node masked band with 30° steps, all masked | identical to Python; latitude circles to 0.05°; no jump across the band equatorward of 60° |
| browser height surfaces vs `interpolate_columns` on the decoded columns | masks identical at −9, −2.5, 0, 1.2345, 8, 20 km, on three levels' exact heights and beyond both ends; values to the float32 storage of the results (2e−6 relative) |
| isothermal column, exact columns | `p = pₛ exp(−g(Z₀ − hₛ)/(R_d T))` to 1e−12 relative, T to 1e−9 K, every mask reason, signed heights |
| pressure recovered by inverting isobars | at 48 nodes on 4 isobars, the height surface through the isobar's local Z returns p₀ to 1e−11 and the same T, u, v |
| stored (16-bit) vs exact columns on height surfaces | within the first-order bound of the quantum node by node; worst pressure 4.1e−5 relative = ½ quantum of Φ / (R_d T) on the 36 km-relief fixture; temperature 3e−14 K (isothermal); no mask disagreement away from boundaries |

## Limits measured here

- Streamline tracing is the most expensive part of a selection: a first
  trace costs 30–150 ms. Dragging defers it. Typed values, saved-time
  switches and playback trace synchronously, so playback of the 29 MB page
  at 8 frames/s is bound by decoding plus tracing (≈ 110 ms per state).
- Captures use software WebGL. On a real GPU, rendering is negligible
  either way (0.06–0.11 ms per frame here).
