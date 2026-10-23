# The 3-D atmosphere view of a saved PE run

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
| `resolution_deg` | display mesh spacing in degrees (a divisor of 180°) | about 4 nodes per shortest resolved wavelength, between 1° and 3°: 2° at T42 |
| `max_mb` | bound on the embedded data, in MB | 32; `None` removes it |
| `streamlines` | export horizontal streamlines | `True` |
| `field`, `level`, `time`, `opacity`, `exaggeration`, `centre` | initial page state (`level` is a native index; `time` a position among the exported times; `centre` = (lat, lon)) | temperature; middle exported level; last time; 0.6; the top exported level near 8 % of the radius; the highest terrain point (flat: 35°N, 0°) |
| `gravity` | only for runs that record none (see below) | none |
| `allow_cuda` | rebuild a preset-mountain terrain through the run's CUDA model | `False` |
| `open_browser` | open the written page | `True` |

The page opens from disk (`file://`) with no network, server, Python or CUDA.
All scripts, styles, shaders and data are inside the file: plain WebGL, no
CDN, no remote textures, no fetches.

The page provides:

- **Saved time.** A discrete slider over the exported states, labelled with the
  actual simulation time (`day 1200 (t = 103680000 s)`).
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
  `cli.pe.build_pe_model` builds the planet with; for Held–Suarez, the ideal
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

At the default 2° mesh one Held–Suarez time with all 20 levels costs about
5 MB. Fewer levels, a coarser mesh or `streamlines=False` buy more times. In
headless Chrome with software WebGL these pages became interactive within
1.3–2.3 s.

## Limits

- One native σ surface at a time. There is no interpolation to pressure or
  height levels, no volume rendering, no mass streamfunction.
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
| physical reconstruction (CPU) | `representation.diagnostics.pe_levels.reconstruct_levels`, `spatial.transforms.host_synthesis.synthesize_wind_latlon` |
| payload | `representation.visual.atmosphere` (selection, bounds, quantization, colour limits), `representation.visual.sphere_streamlines` |
| page | `representation.visual.terrain_globe.html`: the terrain viewer's template, extended with an optional `atmosphere` block; `terrain.write_globe_page` writes both kinds of page |
| entry points | `Simulation.view`, `Snapshot.view` → `storage.view` |

The terrain-only page is unchanged. A browser capture of the same terrain
payload through the template on `main` and through the extended template
differs in 0 of 2,380,000 pixels.
