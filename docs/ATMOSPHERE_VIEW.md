# The 3-D atmosphere view of a saved PE run

For a quick terrain-only Mars globe, an editable low-resolution heightmap,
and examples using your own height function or elevation file, start with
[the terrain quick start](TERRAIN_IMPORT.md#start-with-mars-without-downloading-a-dem).

`Simulation.view()` writes one self-contained HTML page. The page shows the run's
terrain globe and one translucent atmospheric surface: a native sigma level of a
saved primitive-equation state. The surface is placed at that level's hydrostatic
geopotential height and coloured by temperature, pressure, relative vorticity or
divergence. Optional horizontal streamlines can be draped on it. Sliders move
between the exported saved times and native levels, and change the vertical
exaggeration and the opacity.

![Held–Suarez run 002 at day 1200: relative vorticity on native level 6 of 20 (σ = 0.275) as a translucent surface 9.4 km up (×20), with black instantaneous streamlines over a breaking jet; side panel with the time, level, field, opacity and streamline controls and a symmetric colour bar](validation/atmosphere_view_2026-10-23/held_suarez_vorticity_sigma0275.png)

*Held–Suarez run `hs-T42L20-prod-002`, day 1200, relative vorticity on native level 6
of 20 (σ = 0.275, geopotential height 8.8–10.0 km), exaggeration ×20, with
instantaneous horizontal streamlines. This is a headless-Chrome capture of the
exported page, opened from `file://` with networking disabled. Reproduce it with
the command in [the validation record](validation/atmosphere_view_2026-10-23/README.md).*

## Use

```python
from tropoi.representation.archive import open_simulation

sim = open_simulation("runs/hs-T42L20-prod-002")   # PE capsule or Held–Suarez experiment
path = sim.view()                                    # writes the page, opens the browser
path = sim.view("hs.html", times=[0, 60, -1], levels=[0, 10, 19],
                resolution_deg=2.0, field="zeta", open_browser=False)
path = sim[-1].view("last.html")                     # one saved state only
```

From the shell (the same call, as a runnable script):

```bash
python examples/atmosphere_view.py runs/hs-T42L20-prod-002 --times 0 -1 --field zeta --open
```

`view` returns the path of the page. Its options are keyword-only:

| option | meaning | default |
| --- | --- | --- |
| `path` | output file | `<tempdir>/tropoi/<run>-atmosphere-<tag>.html` |
| `times` | saved snapshot indices (negative counts back), or `"all"` | as many evenly spaced states as fit `max_mb` (always the first and the last), with a warning |
| `levels` | native full-level indices, 0 = top | all levels |
| `surface` | `("pressure", Pa)`: one isobaric surface per exported time instead of native levels (see [Fixed-pressure surfaces](#fixed-pressure-surfaces-across-saved-times)); exclusive with `levels` | none (native levels) |
| `level_sigma` | with `surface`: where a native level sits for the interpolation, `"hydrostatic"` or `"nominal"` | `"hydrostatic"` |
| `interactive_pressure` | with `surface` (then the initial isobar): export the native columns and choose any pressure in the page (see [Any pressure in the browser](#any-pressure-in-the-browser-interactive-pressure-mode)); streamlines are traced in the page | `False` |
| `resolution_deg` | display mesh spacing in degrees (a divisor of 180°) | about 4 nodes per shortest resolved wavelength, between 1° and 3°: 2° at T42 |
| `max_mb` | bound on the embedded data, in MB | 32; `None` removes it |
| `streamlines` | horizontal streamlines (exported lines; traced in the page in interactive mode) | `True` |
| `field`, `level`, `time`, `opacity`, `exaggeration`, `centre` | initial page state (`level` is a native index; `time` a position among the exported times; `centre` = (lat, lon)) | temperature; middle exported level; last time; 0.6; the top exported level near 8 % of the radius; the highest terrain point (flat: 35°N, 0°) |
| `gravity` | only for runs that record none (see below) | none |
| `allow_cuda` | rebuild a preset-mountain terrain through the run's CUDA model | `False` |
| `open_browser` | open the written page | `True` |

The page opens from disk (`file://`) with no network, server, Python or CUDA.
All scripts, styles, shaders and data are inside the file: plain WebGL, no
CDN, no remote textures, no fetches.

The page provides:

- **Saved time.** A discrete slider over the exported states, labelled with the
  actual simulation time (`day 1200 (t = 103680000 s)`), with playback
  controls (play/pause, previous/next, speed, loop). See
  [Playback](#playback-saved-states-not-simulated-time).
- **Native level.** A discrete slider over the exported levels, labelled with
  the level number (1 = top) and its σ.
- **Field.** Temperature (K), pressure (hPa), relative vorticity and divergence
  (10⁻⁵ s⁻¹), each with a colour bar and units.
- **Opacity**, plus a switch that hides the surface.
- **Wind streamlines** (on/off).
- **Exaggeration.** One logarithmic slider for terrain and atmosphere together.
- **Colour range.** Fixed by default per field over all exported times and
  levels, so moving a slider never changes the scale. A checkbox switches to
  the range of the current level over the exported times.
- **Flat map** of the same surface with its streamlines, or of the terrain.
  Hover gives values; clicking centres the globe.
- **Physical interpretation and provenance** panels.

Temperature and pressure use sequential colour maps (`inferno`, `viridis`).
Vorticity and divergence use `RdBu_r`, symmetric about zero. These are the
repository's own colour policies (`compose.DISPLAY_DEFAULTS`). The terrain keeps
the terrain viewer's `YlOrBr`.

## Playback: saved states, not simulated time

The play button steps the time slider through the **exported saved states**,
one state per frame, in order. Nothing else happens between frames: there is
no interpolation, no extrapolation and no integration in the page. A played
frame is the same selection as moving the slider to that state: the same
code path rebuilds the surface, colours, streamlines and flat map. A browser
test checks that the two give byte-identical vertex buffers.

Three clocks are easy to confuse:

| clock | what it is | where it shows |
| --- | --- | --- |
| simulated time | the stored time of the saved state | the time label, e.g. `day 0.375 (t = 32400 s)` |
| frame index | position in the exported list | after the label, e.g. `3/4` |
| playback rate | saved states shown per second of **wall-clock** time | the `frames/s` selector (0.5 to 8) |

The solver timestep appears in none of them. Saved states are usually far
apart in model steps: a run with `dt = 60 s` saving every three hours shows
180 steps per frame.

Saved times need not be evenly spaced. The run may save at irregular times,
and the view may export a subset (see [Size and subsampling](#size-and-subsampling)).
Playback still advances one state per frame, so simulated time then passes at
an uneven pace on screen. The note under the controls gives the gap to the
previous exported state and says whether the exported times are evenly
spaced.

Playback stops on the last state. With **Loop** checked it wraps to the first
state. Pressing play on the last state without Loop replays from the first.
Moving the slider or pressing previous/next pauses playback and selects that
state at once. Field, level, opacity, streamlines, exaggeration, camera and
flat-map source stay as they are while time advances. The colour range stays
fixed over all exported times (unless the per-level range is chosen, which is
also fixed over time), so a colour change during playback is a change in the
saved field, not a rescaling. A view of a single saved state has no playback
controls.

### Example: an evolving Mars atmosphere over MOLA terrain

![An evolving Mars atmosphere over MOLA terrain at its last saved state (t = 88642.44 s, 9 of 9): bottom-level divergence as a 55 % translucent surface over the relief (×10) with 157 black instantaneous streamlines; side panel with the playback controls and the note that the 9 states are 11080.305 s apart and are not interpolated](validation/mars_evolving_2026-10-27/mars_evolving_final_delta.png)

*An idealized dry-CO₂ perturbation experiment over MOLA terrain, at the last of
its 9 saved states, one Mars rotation (88642.44 s) after the start. The
balanced resting atmosphere over real relief (`orographic_isothermal_rest`)
received a 1 K coefficient on the (2, 2) temperature mode
(`orographic_thermal_wave`, peak 0.77 K) and was integrated at `l_max` 21 with
4 levels and 60 s RK4 steps. The image shows divergence on the bottom level
(σ = 0.875) at 55 % opacity and ×10 relief, with instantaneous streamlines.
Pressing play steps through the 9 saved states, 3.08 h of simulated time
apart. The first state is at rest and has no streamlines; the winds reach
about 2.5 m/s.*

This is a worked example of imported terrain and saved-state playback.
The measured checks and their limits are in
[the experiment record](validation/mars_evolving_2026-10-27/README.md). There,
the evolution is 10¹⁰ times the drift of a resting control, a 30 s rerun differs
by under 3 × 10⁻⁷ of the evolution, and the relative mass drift is 6 × 10⁻⁷.
These checks apply to this configuration; they do not establish accuracy for
arbitrary terrain or a realistic Mars atmosphere.
The run is published as
[`docs/runs/20261027T175327Z_orographic-thermal-wave_…_71c5441a`](runs/20261027T175327Z_orographic-thermal-wave_rot24p62h_r3_l21_dt11080p3s_3a46f966_71c5441a/),
so the page can be rebuilt without a GPU or the MOLA files:

```bash
python examples/mola_mars_evolving.py view docs/runs/20261027T175327Z_orographic-thermal-wave_rot24p62h_r3_l21_dt11080p3s_3a46f966_71c5441a --out views
```

`views/mars_evolving_all_times.html` (5.2 MB) holds all 9 states at a 3° mesh.
The full experiment (terrain preparation, the four runs, the measurements and
the browser check) is reproduced by the commands in the record.

## Fixed-pressure surfaces across saved times

The first slice of [issue #13](https://github.com/AlexandreEros/Palintropos/issues/13).
Instead of native levels, `surface=("pressure", p₀)` exports **one isobaric
surface per saved time**: the native column of each saved state is
reconstructed on the display mesh and interpolated in the vertical to `p₀`
(pascals, explicit). The page keeps its time slider and playback; the level
slider is replaced by the surface label.

```python
sim = open_simulation("docs/runs/20261027T175327Z_orographic-thermal-wave_…_71c5441a")
sim.view("mars_isobar_300Pa.html", surface=("pressure", 300.0), times="all",
         resolution_deg=3.0)                           # 3 hPa: a Mars pressure
sim.view("hs_500hPa.html", surface=("pressure", 50_000.0), times="all")   # Earth-like
```

```bash
python examples/isobaric_view.py view RUN --pressure 300 --out isobar.html
```

Pressures are planet-specific: 500 hPa is underground everywhere on the Mars
run (its surface pressure is 2.8–12 hPa), and the export then fails with the
pressure range the saved states cover.

![The 3 hPa isobar of the Mars run at its last saved state, coloured by geopotential height (7.65–7.75 km), with a hole over Tharsis where the isobar is under ground and instantaneous streamlines that stop at the hole; the side panel names the surface, the interpolation and the masked node counts](validation/pressure_surface_2026-11-07/mars_isobar_300Pa_t8.png)

*The 3 hPa isobar of the published Mars run at its last saved state (×10
relief, 75 % opacity), coloured by geopotential height. The terrain shows
through the hole over the Tharsis volcanoes, where 3 hPa is under ground
(11 of 7381 nodes) or between the bottom full level and the ground (36
nodes). Streamlines stop at the hole. Headless-Chrome capture of the
committed page [`assets/mars_isobar_300Pa.html`](runs/20261027T175327Z_orographic-thermal-wave_rot24p62h_r3_l21_dt11080p3s_3a46f966_71c5441a/assets/mars_isobar_300Pa.html),
opened from `file://` with networking disabled; see
[the record](validation/pressure_surface_2026-11-07/README.md).*

**What is interpolated, and how.** The numerics live in
`representation.diagnostics.pe_surfaces`, independent of the page, and are
tested on analytic columns.

| item | rule |
| --- | --- |
| vertical coordinate | `ln σ`. The target at each node and saved time is `σ* = p₀ / pₛ`, with `pₛ = exp(ln pₛ)` of the saved state. The node coordinates are the same at every column, so the bracket depends on `pₛ` only through `σ*`. |
| where a level sits | `level_sigma="hydrostatic"` (default): `σʰₖ = σₖ₊½ · e^(−αₖ)`, the pressure at which the Simmons–Burridge `Φₖ` is the hydrostatic geopotential of an isothermal layer (`Φₖ − Φₖ₊½ = R_d Tₖ αₖ` is the step from `ln pₖ₊½` to `ln pₖ₊½ − αₖ`). With it an isothermal column interpolates to the closed form `Φₛ + R_d T ln(pₛ/p₀)` to round-off at every valid pressure, over flat and sloping terrain (tested to 1 µm). `"nominal"`: `σₖ`, the pressure the native-level page labels the level with; on an isothermal column that attachment is off by `R_d T (αₖ − ln(σₖ₊½/σₖ))/g` in the interior (zero for the top layer, `α₁ = ln 2`): 115 m at most on the test grid's four layers, measured. The two positions differ by `σ = 0.3750` versus `0.3679` for the second of four uniform layers. |
| interpolant | linear in `ln σ` between the two bracketing levels, for temperature, geopotential, `u` and `v`; `Z = Φ/g` and the wind speed follow. A target on a level position reproduces that level's values to 1e−12 relative. |
| valid bracket | `σ₁ ≤ σ* ≤ σ_K`, the ends widened by 1e−9 relative so round-off in `p₀/pₛ` cannot push a target placed exactly on an end level out of range. No extrapolation. |
| masks | one byte per node and time: 0 valid; 1 below the surface (`p₀ > pₛ`); 2 above the top level; 3 between the bottom level and the surface (`σ_K < σ* ≤ 1`). Codes 2 and 3 are "outside the sampled full-level range", distinct from 1. |
| failures | a non-finite reconstruction or a column whose `Φₖ` does not decrease downward raises; a saved time with no valid node raises `SurfaceUnavailableError` naming the time and the pressure range the states cover. |

**What the mask does.** A masked node is not a sample, so nothing pretends
it is: the globe draws no triangle that touches one (the terrain shows
through the hole), the flat map greys it, hover says why ("no sample on the
3 hPa surface: below the surface (p₀ > pₛ)"), colour ranges and the 16-bit
quantization use valid nodes only, and streamlines are traced only through
cells whose four nodes are valid (`integrate_streamlines(valid=…)`), so a
line never crosses a gap. A browser test checks each of these on a page
with holes.

**Fields.** Temperature (K), geopotential height (m) and wind speed (m/s).
Pressure is not offered: it is `p₀` everywhere on the surface. **Relative
vorticity and divergence are not exported** on a derived surface: a
horizontal derivative on a σ level is not the derivative on a pressure
level, and interpolating `ζ_σ` would label the wrong quantity. The page says
so. Target-coordinate derivatives from the surface winds are a later slice
of #13. The native-level page is unchanged (its payload stays `version` 1;
the derived page is `version` 2 with `surface` and `mask` blocks).

**Playback and size.** The same contract as native levels: `times="all"`
never subsamples (above `max_mb` it fails naming the alternatives), the
payload is estimated before anything is reconstructed, frames are
processed one saved state at a time, and the time slider and play button
step through the exported saved states. Measured (CPU, the same laptop):

| run | export | mesh | file | export time |
| --- | --- | --- | --- | --- |
| Mars MOLA evolving (9 states, T21L4) | 3 hPa, `times="all"` | 3° | 1.53 MB | 1.8 s |
| `hs-T42L20-prod-002` (120 states, T42L20) | 500 hPa, `times="all"` | 3° | 15.9 MB | 73 s |
| same | 500 hPa, `times="all"` | 2° | 28.9 MB | 99 s |

Where the time goes, per saved state on the T42L20 run at 2° (16,471
nodes): spherical-harmonic synthesis of temperature and geopotential 0.25 s,
winds 0.29 s, hydrostatic coefficients 2 ms, **vertical interpolation 4 ms**
(0.7 % of the reconstruction), streamlines 44 ms. The vorticity and
divergence syntheses the native page pays (0.25 s) are skipped. The cost is
the synthesis, and it scales with the number of saved states times the
display nodes, not with the number of levels embedded: all 120 states fit the
32 MB budget at 2°, where the native page holds 5 states with 20 levels.
Measurements: [`validation/pressure_surface_2026-11-07/`](validation/pressure_surface_2026-11-07/README.md),
reproduced with `python examples/isobaric_view.py measure RUN --pressure P`.

**Not in this slice.** Geopotential-height surfaces (`("geopotential_height",
m)` is refused by name), target-surface vorticity and divergence, lazy or
compressed decoding, any GPU path. The page still decodes all embedded
arrays eagerly; the 29 MB page became interactive in headless Chrome in
1.1 s with 64 MB of JS heap, and switched frames in 25 ms (median), so
lazy decoding is not needed at this size.

## Any pressure in the browser: interactive-pressure mode

The second slice of [issue #13](https://github.com/AlexandreEros/Palintropos/issues/13).
`interactive_pressure=True` exports the **native columns** of each saved
state instead of one surface, and the page interpolates them to **any
pressure chosen in the browser**, offline. Export once; decide the isobars
later.

```python
sim = open_simulation("docs/runs/20261027T175327Z_orographic-thermal-wave_…_71c5441a")
sim.view("mars_isobars.html", surface=("pressure", 300.0),   # initial isobar, Pa
         interactive_pressure=True, times="all", resolution_deg=3.0)
sim.view("hs_isobars.html", surface=("pressure", 50_000.0),
         interactive_pressure=True, times=[0, 60, 119])       # explicit saved indices
```

```bash
python examples/isobaric_view.py interactive RUN --pressure 300 --out isobars.html
```

`surface` is then the initial isobar (required, pascals); `times` and
`max_mb` keep their contract; `level_sigma` chooses the level positions as
before; `levels` and `level` do not apply (they are refused);
`streamlines=False` turns the browser-traced lines off (see below). The
default output name contains `-isobars-`.

**What the file holds.** Per exported saved state: temperature, geopotential,
`u` and `v` on **every** full level (16-bit per level slice, with that
slice's min/max) and the surface pressure (float32); shared: the level
positions in σ, the mask labels, the end tolerance, conservative colour
ranges. Nothing redundant: wind speed and geopotential height are derived
in the page, and the interpolated height serves both the geometry and the
displayed field. Each state is reconstructed once (one spherical-harmonic
synthesis per field and level), quantized, and dropped before the next.
The payload is `version` 3 with a `columns` block; the fixed-surface page
(`version` 2) and the native page (`version` 1) are unchanged.

**What the page computes.** The vertical step of
`representation.diagnostics.pe_surfaces`, line for line: at each node
`σ* = p₀/pₛ`; masks 1 (`σ* > 1`, under ground), 2 (above the top level),
3 (between the bottom level and the ground), the end levels widened by
1e−9; temperature, geopotential, `u` and `v` linear in `ln σ` between the
bracketing levels, the weight clamped (no extrapolation); then `Z = Φ/g` and
the wind speed. One pass produces the fields, the height and the mask
together for one (pressure, saved time), so geometry, colours, masks, flat
map, hover and labels always describe the same selection. Python is the
reference: `pressure_columns` → `interpolate_columns` is what
`pressure_surface` does, and a browser test compares the page's arrays with
it at arbitrary pressures, on both end levels and in holes (masks identical;
values within the float32 storage of the result).

**Controls.** A logarithmic slider over the pressures some node of some
exported state samples (`σ₁·min pₛ` to `σ_K·max pₛ`, 1001 positions) and a
typed value with an explicit unit (hPa or Pa), kept synchronized: a typed
value between slider positions is used exactly and the slider shows the
nearest position. Non-finite, empty, zero or negative input is refused with
the reason, nothing changes. Rapid slider input is coalesced to one
recomputation per animation frame and the newest value wins; the
recomputation itself is synchronous, so no older update can overwrite a
newer selection. Changing the saved time keeps the pressure and recomputes
the surface from that state's columns; nothing is interpolated in time.

**Masks and the empty surface.** As in the fixed-surface page: masked nodes
draw no triangle, are grey on the flat map and say why on hover (every node
the bilinear sample draws on is checked). A pressure no node samples at the
selected time gives an **empty surface** with an explanation (the state's
surface-pressure range and the sampled range) and live controls; the Python
reference raises `SurfaceUnavailableError` for the same case. An initial
pressure no exported state samples anywhere fails at export, naming the
range.

**Colour ranges** are fixed bounds from the exported columns over all times
and levels: temperature and geopotential height cannot leave their column
ranges under linear interpolation, and the speed of an interpolated vector
is at most the larger bracket speed, so the scales stay put while the
pressure moves (the native per-level option does not apply).

**Streamlines are traced in the page.** The wind on an arbitrary surface
exists only in the browser, so the page integrates it there: the
interpolated `u, v` of the surface shown (vorticity and divergence are not
needed) go through a transcription of
`representation.visual.sphere_streamlines.integrate_streamlines`, the
Python reference, with its geometry and rules unchanged: arc length on the
unit sphere, `dx/ds = V_t/|V_t|`, classical RK4 in R³ renormalized onto the
sphere (regular across the poles, periodic across the seam), the wind
interpolated bilinearly in Cartesian components with each pole row carrying
its mean vector, the same 160 Fibonacci seeds, 1.5° steps, at most 24 each
way, stopping below 1 mm/s or 2 % of the surface's maximum speed, and only
through cells whose four nodes are valid, every accepted step checked along
its length (points closer than half a cell, its end included) so a long
step cannot jump a narrow masked band. Nothing extra is exported, so file
size is unchanged. A browser test feeds the page's own decoded winds and
mask to Python and finds the same lines point for point (largest
difference 1e−14 on the unit sphere, libm round-off; the tolerance is 1e−9),
on analytic winds too (solid-body zonal flow, a flow over the poles, the
seam, calm air, a one-node masked band that a 30° step would otherwise
jump, an all-masked surface).

They are instantaneous horizontal streamlines of one saved state on the
selected surface, draped on it: not parcel trajectories, not 3-D flow.

*Scheduling.* Surface fields and geometry follow a dragged slider every
frame; the lines of the previous surface are removed at once and the new
ones are traced 150 ms after the slider stops moving, or immediately on
release. Typed values, saved-time switches, playback and coordinate changes
trace synchronously, so their frame always carries the matching lines. Each
request carries a generation number and only the newest may publish: a
deferred trace superseded by a later selection is dropped. The last 8 line
sets are kept (at most 1 MB), so returning to a surface is instant.

*Fix in the reference.* Writing the browser test exposed a gap in the
Python integrator: the end of an accepted step was never checked against
the mask itself, only the next step's first evaluation was, so the last
point of a line could lie in a masked cell. The whole-step check now
includes the step's end, in Python and in the page alike; fixed-surface
exports can lose that last point of an occasional line.

**Stored precision, measured.** Each level slice of each field is stored
with its own 16-bit range, so the error is at most half a quantum,
`(max − min)/131070`, per value; the surface pressure is float32 (6e−8
relative), which keeps the under-ground comparison `p₀ > pₛ` exact to that
precision and can move a node that sits exactly on an end level across the
1e−9 tolerance band (a test excludes such nodes; it is the only place the
stored form and the exact columns disagree on a mask). Interpolation is
linear, so the interpolated error is within the bracket quanta; the test
measures it apart from the interpolation error (zero on the stored columns
up to float32 storage of the result) and the analytic isothermal column
goes through the stored form to the same bound.

| run | half-quantum of the stored columns (worst level) | Z |
| --- | --- | --- |
| Mars MOLA evolving (T21L4, 3°) | T 0.02 mK, wind 0.03 mm/s | 0.12 m |
| `hs-T42L20-prod-002` (T42L20, 2°) | T 0.4 mK, wind 1 mm/s | 0.01 m |

**Size and speed, measured** (export: the same laptop CPU as above; page:
headless Chrome with software WebGL, `examples/isobaric_view.py bench`,
[the record](validation/interactive_pressure_2026-11-08/README.md)).
Full columns cost more than one surface: per saved state, `8·K` bytes per
node plus 4 for the surface pressure before base64, against 7 (+ streamlines)
for a fixed surface; the estimate is made before anything is reconstructed
and the decoded memory of one state is reported separately (page details
and the export report).

| run | states | mesh | file | export | per decoded state |
| --- | --- | --- | --- | --- | --- |
| Mars (T21L4), `times="all"` | 9 | 3° (7381 nodes) | 3.64 MB | 1.6 s | 0.60 MB |
| `hs-T42L20-prod-002`, `times="all"` | 120 | 2° | **refused**: 433 MB above 32 MB (194 MB at 3°) | – | – |
| same, 8 evenly spaced states | 8 | 2° (16,471 nodes) | 29.3 MB | 6.9 s | 5.6 MB |

| page | interactive after | JS heap | pressure update (page) | of which interpolation | saved-time switch (page) | of which decoding |
| --- | --- | --- | --- | --- | --- | --- |
| Mars, 3.64 MB | 0.9 s | 19 MB | 6 ms | 0.9 ms | 12 ms | 4.5 ms |
| Held–Suarez, 29.3 MB | 1.05 s | 50 MB | 15 ms | 3.5 ms | 140 ms | 112 ms |

The page's own timer splits each update into decoding (only on a saved-time
switch not in the two-state cache), interpolation, mesh (positions,
colours, triangle re-indexing, labels) and flat map; the round trip seen
through the DevTools protocol adds about 30–40 ms of its own. Interpolation
is the cheapest part of a pressure update at both sizes; at 20 levels the
base64 decoding of a state dominates a time switch, so playback of the
29 MB page runs at about 7 states per second, not 8. Only the selected
state is decoded (two held, least recently used dropped); the 29 MB page
holds 50 MB of heap, not the 44 MB its eight states would decode to. Export
is cheaper than the fixed-surface export of the same states (0.69 s against
0.93 s per state on the T42L20 run at 2°) because no streamlines are traced.

**Not in this slice.** Target-surface vorticity
and divergence; geopotential-height surfaces; compression of the columns
(the file is base64 of raw 16-bit slices); any GPU path. The 32 MB default
budget bounds the T42L20 run to 8 states at 2° or 19 at 3°; the budget is
explicit (`max_mb`, `None` removes it) and the refusal says what to change.

## What the surface is

**The level.** Native levels are the core's full levels. With interfaces
σ₁/₂ = 0 < … < σ_K+1/2 = 1 (top to bottom), full level k has
σ_k = (σ_k−1/2 + σ_k+1/2)/2. The stored temperature, vorticity, divergence
and winds are full-level quantities. Geometry, colour, pressure and
streamlines of a slider position all refer to the same level and the same
saved state. Nothing is interpolated in the vertical.

**Geometry.** The surface of level k is drawn at radius `R + E·Z_k`, where

- `Z_k = Φ_k / g` is geopotential height;
- `Φ_k` is the Simmons–Burridge full-level geopotential of the saved
  temperature. It is computed by the core's own
  `spatial.sigma_coordinate.hydrostatic_geopotential`, with the same
  arguments the tendency passes: the stored temperature coefficients, the
  surface geopotential `Φ_s = g·h_s` as the σ = 1 boundary condition, and
  `R_d`;
- `g` is the run's gravity, the same `g` that made `Φ_s = g·h_s`;
- `R` is the run's dynamical radius (for a capsule, the volumetric-mean radius
  `temporal.models.build_pe_model` builds the planet with; for Held–Suarez, the ideal
  sphere's `radius`);
- `E` is the display exaggeration.

`Φ_k` is **absolute**. It includes `Φ_s` exactly once and is zero on the
reference sphere `h = 0`. The terrain is drawn at `R + E·h_s` with the same `R`
and the same `E`. Terrain and atmosphere therefore share one datum and keep
their physical relation at every exaggeration: no level is normalized and no
mean height is removed. `E` changes display geometry only. A test checks
`Z_k(with terrain) − Z_k(flat) = h_s` to 1e−8 m.

**Approximation.** Geopotential height is drawn as radial distance. That is the
model's own convention: the core uses constant gravity and spherical
geopotential surfaces, so geopotential height and geometric height coincide in
it. On a real planet they differ by about `Z²/R`, roughly 0.3 % at 20 km on
Earth. Simmons–Burridge `Φ_k` is the layer's representative geopotential, not a
value interpolated to `p = σ_k p_s`.

**Pressure.** `p_k = σ_k · p_s` with `p_s = exp(ln p_s)` of the saved state, at
every point. The core is pure sigma with the model top at `p = 0` (no
hybrid/top-pressure term). A native level is therefore not an isobaric surface:
its pressure follows the surface pressure. On the Mars run below, the bottom
level (σ = 0.875) reads 1.8 hPa over Tharsis (−10°, −120°, terrain 11.5 km) and
10.9 hPa over Hellas (−38°, 57°, terrain −7.7 km).

![Mars orographic-rest run over MOLA terrain: the bottom native level (4 of 4, σ = 0.875) as a 40 % translucent surface coloured by pressure, following the terrain (×10), low pressure over the Tharsis volcanoes, terrain visible beneath; side panel says there are no streamlines because the state is at rest](validation/atmosphere_view_2026-10-23/mars_orographic_rest_pressure_bottom.png)

*Mars orographic-rest run over real MOLA terrain (`l_max` 42, 4 levels): pressure
on the bottom native level at 40 % opacity, exaggeration ×10. The surface sits
1.5 km above the terrain everywhere, as the isothermal column requires, so it
follows Tharsis and Hellas. The terrain shows through. The run is at rest
(|V| ~ 1e−10 m/s), so no streamlines are drawn. This is a visualization of the
saved orographic-rest state, not evidence of dynamical stability: the view
displays states, it does not test them (the balance check of that run is in
[TERRAIN_IMPORT.md](TERRAIN_IMPORT.md#real-data-mars-from-mola)).*

**Fields** are exact host evaluations of the saved spherical-harmonic
coefficients on the display mesh (`spatial.transforms.host_synthesis`, the
backends' basis). Winds use the cores' Helmholtz reconstruction
(`ψ = ∇⁻²ζ`, `χ = ∇⁻²δ`) with exact spectral derivatives. On a GPU, a test checks
geopotential, winds and surface pressure against the core's own
`geopotential_fields`, `wind_on_state_grid` and `surface_pressure_on_state_grid`
(agreement better than 1e−9 relative). The page stores values as 16-bit integers
per (field, time, level) slice, using that slice's own range. The display
quantization is at most (max − min)/131070, far below one colour step.

## Streamlines

The streamlines are instantaneous horizontal streamlines: curves tangent to the
saved horizontal wind of the selected state on the selected level. They are not
parcel trajectories, which would need the flow to evolve, and not 3-D
streamlines. No vertical velocity is computed or implied, and the changing
height of the surface is not read as vertical motion.

They are integrated in Python on the unit sphere with arc length as the
parameter: `dx/ds = V_t/|V_t|`. This is the Cartesian form of
`dφ/ds = v/|V|`, `dλ/ds = u/(|V| cos φ)`. One RK4 step is 1.5° of arc, followed by
renormalization. The Cartesian form never evaluates the 1/cos φ metric factor,
so lines cross the poles and the 180° meridian without special cases. Tests
check this against rigid rotations about three axes: lines stay on their
circles to 2e−3, with no jumps at the seam or the poles.

The integration is bounded: 160 deterministic seeds (a Fibonacci lattice), at
most 24 steps each way, and it stops where `|V|` falls below 1 mm/s or 2 % of
the slice's maximum. The page drapes each line onto the displayed surface (the
same `R + E·Z`, lifted 0.3 %) whenever time, level or exaggeration changes. A
chevron at each line's middle points downstream. A state at rest has no
streamlines, and the page says so.

## Which saved runs work

| saved run | snapshots | terrain | gravity |
| --- | --- | --- | --- |
| `tropoi run pe` capsule (`pe_coeffs.npy`) | every saved state | flat, or the capsule's `terrain/` (digest-checked) | `gravity` from the configuration |
| Held–Suarez experiment (`config.json` + `states/state-d*.npy`) | every saved state, `t = day × 86400 s` | flat: the protocol has no topography (PROTOCOL.md §1) | `gravity` (9.8) |
| `tropoi run pe --topography mountain` capsule | every saved state | not saved: refused on the CPU; `allow_cuda=True` rebuilds it through the run's model | from the configuration |
| flat `tropoi run pe` preset | every saved state | flat | **none recorded**: pass `gravity=` (e.g. 9.80665 for geopotential metres) |

- **Read-only.** Opening and viewing never write to the run directory. A path
  inside it is refused, except under `assets/`, the repository's place for
  derived products.
- **Held–Suarez experiments** open through `open_simulation` as PE
  simulations (`representation.archive.held_suarez`). Older configurations read
  the way `HeldSuarezConfig.from_dict` reads them: a missing `schema` key means
  version 1, and missing constants take the dataclass defaults. The page lists
  which keys were defaulted.
- **Accumulated statistics are not snapshots.** An experiment with
  `statistics.npz`/`series.json` but no `states/`, or a reference run with only
  `series.npz` (such as the Dinosaur reference), opens with zero snapshots or is
  refused, with a message saying what it holds. No time series is made from
  averages: zonal and time means contain no longitude-resolved state at any
  saved time.
- **CPU only.** Opening, reconstruction and export import NumPy, the archive
  readers, `spatial.sigma_coordinate`, host synthesis and Matplotlib (for
  colour tables). They never import CuPy, the cores, grids, transforms on the
  GPU, or a model. A fresh-interpreter test enforces this. The only exception
  is the explicit `allow_cuda=True` above.

## Size and subsampling

Each exported (time, level) pair stores 5 values per mesh node (4 fields plus
height, 2 bytes each) and its streamlines. The bound is estimated **before**
anything is reconstructed. With `times=None` the export takes as many evenly
spaced saved states as fit `max_mb`. It always includes the first and the last,
and it issues a warning (`AtmosphereViewWarning`) naming the saved indices and
days. The page shows the same note. If not even the first and the last fit
(for a single-snapshot run, its one state), the automatic selection raises
`PayloadTooLargeError` before anything is reconstructed; pass an explicit
`times=[i]` if one state of a longer run is what you want. An explicit
selection that does not fit raises `PayloadTooLargeError` too, instead of
being cut silently.

Measured on this machine (MX110 laptop, CPU path):

| run | exported | mesh | file | export time |
| --- | --- | --- | --- | --- |
| `hs-T42L20-prod-002` (120 states, T42L20), default | 5 of 120 states (days 10, 310, 610, 900, 1200) × 20 levels | 2° | 26.4 MB | 10 s |
| same, `times=[0, -1]`, `levels=[0, 10, 19]` | 2 × 3 | 3° | 1.3 MB | 1.4 s |
| Mars MOLA orographic rest (3 states, T42L4, real terrain) | 3 × 4 | 2° | 3.1 MB | 1 s |
| Mars MOLA evolving (9 states, T21L4, real terrain), `times="all"` | 9 × 4 | 3° | 5.2 MB | not timed |

At the default 2° mesh one Held–Suarez time with all 20 levels costs about
5 MB. Fewer levels, a coarser mesh or `streamlines=False` buy more times. In
headless Chrome with software WebGL these pages became interactive within
1.3–2.3 s.

## Limits

- One surface at a time: a native σ level, one fixed-pressure surface
  ([above](#fixed-pressure-surfaces-across-saved-times)) or any pressure
  interpolated in the page ([interactive mode](#any-pressure-in-the-browser-interactive-pressure-mode),
  streamlines traced in the page). No geopotential-height surfaces yet, no
  vorticity or divergence on a derived surface, no volume rendering, no mass
  streamfunction.
- The streamline mask check samples each step at points closer than half a
  mesh cell, measured in degrees. That stops a line at any masked band one
  node wide or more, except near a pole, where a cell's longitude width in
  arc shrinks below the check spacing: there a step much longer than the
  default 1.5° could cross a one-node band (a test shows it with 30° steps).
  Between two check points a line can also cut the corner of a masked cell,
  by at most a quarter cell.
- Hovering the globe locates points on the undisplaced sphere ("relief ignored
  when picking"), as in the terrain viewer. The flat map is exact.
- Streamline density is uniform in seeds, not adaptive. Lines in weak flow can
  curl into small closed loops around eddies.
- A constant field (an isothermal rest state) has a degenerate colour range and
  draws in one colour.
- WebGL is required for the globe. Without it the page says so, and the flat
  map still works.

## Where it lives

| layer | module |
| --- | --- |
| archive (reading saved runs) | `representation.archive.held_suarez` (Held–Suarez storage), `representation.archive.pe_context` (`PEColumnContext`: σ, constants, radius, gravity, terrain, provenance), `CapsuleStorage.pe_context` |
| physical reconstruction (CPU) | `representation.diagnostics.pe_levels.reconstruct_levels`, `spatial.transforms.host_synthesis.synthesize_wind_latlon`; derived pressure surfaces: `representation.diagnostics.pe_surfaces` (`pressure_columns`, `interpolate_columns`, `pressure_surface`, `interpolate_log_sigma`, masks), `spatial.sigma_coordinate.SigmaGrid.hydrostatic_full_levels` |
| payload | `representation.visual.atmosphere` (selection, bounds, quantization, colour limits; `quantize_columns`/`dequantize_columns` and the column estimates for the interactive page), `representation.visual.sphere_streamlines` |
| page | `representation.visual.terrain_globe.html`: the terrain viewer's template, extended with an optional `atmosphere` block (native levels, one fixed surface, or columns interpolated in the page: `computeSurface`); `terrain.write_globe_page` writes both kinds of page |
| entry points | `Simulation.view`, `Snapshot.view` → `storage.view` |

The terrain-only page is unchanged. A browser capture of the same terrain
payload through the template on `main` and through the extended template
differs in 0 of 2,380,000 pixels.
