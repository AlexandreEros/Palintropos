# Browser streamlines, geopotential-height and σ surfaces (2026-11-08)

Measurements for the third slice of
[issue #13](https://github.com/AlexandreEros/Palintropos/issues/13): the
interactive column page traces instantaneous horizontal streamlines in the
browser and offers three derived coordinates, switchable without
regenerating the file: isobars, geopotential-height surfaces and
constant-σ surfaces (the last with relative vorticity and divergence). The
method is described in
[ATMOSPHERE_VIEW.md](../../../src/tropoi/representation/visual/ATMOSPHERE_VIEW.md#geopotential-height-surfaces).
Same laptop as the earlier records (Intel Core i5-8250U class CPU, Windows
11, Python 3.12, NumPy 2.2; headless Chrome with software WebGL through
`tools/browser_cdp.py`, networking blocked; no GPU used). All numbers below
are from the final pages, with ζ/δ columns and lazy ζ/δ decoding.

These are displays of saved states: the Mars run is an idealized dry-CO₂
perturbation experiment and the Held–Suarez run an idealized forcing
benchmark. Nothing here validates either as a climate.

| file | what it is |
| --- | --- |
| `mars_isobars_interactive_bench.json` | the committed Mars page: start-up; 5 pressures, 5 heights and 4 σ values × 5 updates each with the page's own split; 18 saved-time switches; 15 coordinate switches; a slider drag with the deferred trace; rendering; heap; one request, no errors |
| `hs_surfaces_2deg_8states_bench.json` | the same on the 43.4 MB eight-state Held–Suarez page (local run, built with `--max-mb 48`; not committed, rebuilt by the command below) |
| `mars_isobars_interactive_p300Pa.png`, `…_z0m.png`, `…_z4000m.png`, `…_s0.5.png` | the Mars page at 3 hPa, at the datum, at 4 km over Tharsis and at σ = 0.5 (relative vorticity), with streamlines |
| `hs_surfaces_2deg_8states_p50000Pa.png`, `…_z5500m.png`, `…_s0.5.png` | the Held–Suarez page at 500 hPa, at 5.5 km and at σ = 0.5 (relative vorticity) |

The bench writes one capture per selected value; only these are kept.

## Commands (the reproducible example)

```bash
MARS=docs/runs/20261027T175327Z_orographic-thermal-wave_rot24p62h_r3_l21_dt11080p3s_3a46f966_71c5441a
python examples/isobaric_view.py interactive $MARS --pressure 300 --resolution 3 --out $MARS/assets/mars_isobars_interactive.html
python examples/isobaric_view.py bench $MARS/assets/mars_isobars_interactive.html docs/validation/derived_surfaces_2026-11-08 --pressures 300 150 600 1000 50 --heights -4000 0 4000 10000 20000 --sigmas 0.2 0.5 0.875 0.95 --centre -5 -115
python examples/isobaric_view.py interactive runs/hs-T42L20-prod-002 --pressure 50000 --resolution 2 --times 0 17 34 51 68 85 102 119 --max-mb 48 --out out/hs_surfaces_2deg_8states.html
python examples/isobaric_view.py bench out/hs_surfaces_2deg_8states.html docs/validation/derived_surfaces_2026-11-08 --pressures 50000 25000 85000 12345.6 --heights 1500 5500 10000 -100 --sigmas 0.3 0.5 0.9 --exaggeration 20
```

Without `--max-mb 48` the Held–Suarez export is refused before anything is
reconstructed: "8 saved states would embed about 43.5 MB, above the 32 MB
bound". The written file is 43.41 MB.

Open the committed Mars page from disk to try it:

- Choose any pressure.
- Switch to **Geopotential height** and choose any height, negative values
  included. Below about 8 km the surface meets the terrain, and hovering
  over a hole says why.
- Switch to **σ** and pick relative vorticity.
- Switch back: each coordinate keeps its value.
- Step or play the saved states.

The streamlines follow every selection.

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

**σ surfaces** are wholly valid or wholly empty:

- Mars: σ = 0.2, 0.5 and 0.875 (the bottom level) are fully drawn, with 157–160
  streamlines. σ = 0.95 lies below the bottom level and is empty, and the page
  says so.
- Held–Suarez: σ = 0.3, 0.5 and 0.9 are fully drawn. The relative-vorticity
  scale is ±20 × 10⁻⁵ s⁻¹, from the column maxima.

Held–Suarez heights: on its flat terrain every node is valid from 1.5 to 10
km. At −100 m every node is below the terrain, so the surface is empty, and
the page says so with its controls still live.

## Costs

| page | file | export | interactive after | JS heap after the bench | GPU frame |
| --- | --- | --- | --- | --- | --- |
| Mars, 9 states, 3° | 5.15 MB (3.64 for slice 2; +ζ/δ columns and 39 KB of terrain) | 1.5 s | 1.2 s | 20.5 MB | 0.06 ms |
| Held–Suarez, 8 states, 2° | 43.4 MB (29.3 without ζ/δ) | 8.0 s | 2.9 s | 72 MB | 0.10 ms |

Medians of the page's own timer, in ms. The DevTools round trip adds its own
overhead on top.

| page | pressure update (interp.) | height update (interp.) | σ update (interp.) | coordinate switch | saved-time switch (decoding) |
| --- | --- | --- | --- | --- | --- |
| Mars | 23.8 (1.4) | 22.2 (1.9) | 19.5 (1.2) | 20.5 | 51 (5.7) |
| Held–Suarez | 44.3 (4.3) | 40.8 (10.4) | 31.0 (3.8) | 29.0 | 149 (90) |

**Lazy ζ/δ decoding.** ζ and δ are decoded only while the σ coordinate is
selected. They are added to a cached state the first time σ is chosen. With
eager decoding, a saved-time switch on the Held–Suarez page took 242 ms (115
ms decoding); lazily it takes 149 ms (90 ms). The slice-2 page without ζ/δ
columns took 114 ms in this session's earlier bench and 140 ms in the slice-2
record. Run-to-run noise on this laptop is a few tens of ms.

**Streamline tracing** is timed apart (`integrate_ms`). A first trace of a
surface, which includes the JIT warm-up of a freshly loaded page, took a
median of 21 ms on Mars and 62 ms on Held–Suarez, at most 141 and 177 ms.
Warm, repeated traces of one Mars surface take 7–18 ms (59–158 lines,
measured through `traceStreamlines`). Pressure updates in the bench
alternate between two values, so their traces come from the cache
(`streams_ms` 0). Saved-time switches trace synchronously, and that cost is
inside their page total.

**A slider drag** (six positions, one per frame): the surface follows each
frame. The previous lines are cleared at once, and the new ones were
published 199 ms (Mars) and 197 ms (Held–Suarez) after the last movement:
the 150 ms settle delay plus one trace. Releasing the slider traces at once.

**Memory.** The heap did not grow over the bench. The decoded-state bound
(two states) and the line cache (8 sets, at most 1.0 MB) hold.

## Numerical agreement (from the tests)

| comparison | result |
| --- | --- |
| browser streamlines vs `integrate_streamlines` on the page's decoded winds and mask | same lines, same points: worst 1e−14 on the unit sphere (fixtures), 1.1e−13° on the Mars run (5 pressures × 2 times, holes included) |
| analytic winds: zonal, over the poles, seam, calm, one-node masked band with 30° steps, all masked | identical to Python; latitude circles to 0.05°; no jump across the band equatorward of 60° |
| browser height surfaces vs `interpolate_columns` on the decoded columns | masks identical at −9, −2.5, 0, 1.2345, 8, 20 km, on three levels' exact heights and beyond both ends; values to the float32 storage of the results (2e−6 relative) |
| isothermal column, exact columns | `p = pₛ exp(−g(Z₀ − hₛ)/(R_d T))` to 1e−12 relative, T to 1e−9 K, every mask reason, signed heights |
| pressure recovered by inverting isobars | at 48 nodes on 4 isobars, the height surface through the isobar's local Z returns p₀ to 1e−11 and the same T, u, v |
| stored (16-bit) vs exact columns on height surfaces | within the first-order bound node by node; worst pressure 4.1e−5 relative = ½ quantum of Φ / (R_d T) on the 36 km-relief fixture |
| σ surface at a level's nominal σₖ | that native level exactly, every field including ζ, δ (1e−12) |
| ζ, δ, u, v on a σ surface vs spectral synthesis of the same combination of level coefficients | ζ, δ to 1e−12; u, v to the wind synthesis's own round-off (2e−10 to 6e−10 relative, ψ = ζR²/l(l+1) ~ 4e8 m²/s differentiated back to ~75 m/s) |
| browser σ surfaces vs Python on the decoded columns | masks identical (all-or-nothing) at every nominal level, between levels and beyond both ends; values, ζ and δ to float32; streamlines identical |

## Limits measured here

- Streamline tracing is the most expensive part of a selection: a first
  trace costs up to ~180 ms. Dragging defers it. Typed values, saved-time
  switches and playback trace synchronously.
- The ζ/δ columns make the files ~50 % larger. The 32 MB default holds 5
  states of the T42L20 run at 2° instead of 8.
- The first switch to σ for a saved state also decodes its ζ/δ, so it costs
  one extra field decode per stack.
- Captures use software WebGL. On a real GPU, rendering is negligible either
  way (0.06–0.10 ms per frame here).
