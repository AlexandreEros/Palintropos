# Visualization

`fields.py`, `specs.py`, `views.py`, `evaluate.py`, `timeline.py` and the
renderers separate field reconstruction, composition, normalization and
drawing. Per-core modules build the scientific figures; `snapshot.py`
adapts the [saved-run API](../../temporal/SAVED_RUNS.md).

| Reference | Implemented view |
| --- | --- |
| [ATMOSPHERE_VIEW.md](ATMOSPHERE_VIEW.md) | Saved PE native levels, pressure/height/sigma surfaces, playback, masks and instantaneous horizontal streamlines |
| [HELD_SUAREZ_CLIMATE.md](HELD_SUAREZ_CLIMATE.md) | CPU six-section climate figure, sampling/eddy definitions, sidecar and PNG/PDF recipe |
| [Terrain import](../../spatial/terrain/IMPORT.md) | Static terrain preview and offline globe |
| [Sky view](../../../../docs/SKY_VIEW.md) | Standalone single-scattering Rayleigh sky page (`sky.py` + `sky_view.html`, with `colour.py` and the [packaged CIE table](data/README.md)); first-order radiative transfer over the shared `tropoi.physics` composition and star data, a visual prototype, not a validated sky colour |

`terrain_globe.html` is package data shared by terrain and atmosphere
writers. These guides stay beside the Python/template owner; validation
assets remain under [docs/validation](../../../../docs/validation/README.md)
and reproducible figure recipes under [docs/figures](../../../../docs/figures/README.md).
Views reconstruct saved fields; interpolation, exaggeration, quantization
and streamlines are display operations, not additional time integration.
There is no radiative-heating solver in this layer.

Spectral coefficient colors use `arg(a_lm)` for hue, without the legacy
`arg(-a_lm)` rotation; amplitude relative to the timeline maximum controls
saturation. Interpreting phase requires the spherical-harmonic convention,
coefficient normalization, longitude origin and any configured palette phase
offset. Changing the longitude origin rotates phases for m>0 even when the
physical field is unchanged. See the [numerical conventions](../../spatial/CONVENTIONS.md)
and [rendering architecture](../../../../docs/ARCHITECTURE.md#output-capsules-and-provenance).

## Interactive capsule viewer

From an environment with Palintropos installed, run the following from the repository root:

```powershell
tropoi view runs/dev-thai-ben2-ck-heated-T21L40-f44c039-layer-linear-dt600-30d
```

Choose **Air temperature** and the final saved time (day 30). The native-level
selector runs from the top of the atmosphere to the bottom. Pressure selections
use pascals, geopotential heights use metres and nominal σ is dimensionless.
You can pass more capsule directories to the same command and switch among
them in the archive selector. A statistics-only core opens its recorded series.

The command opens a browser and serves the selected directories read-only on
an automatically assigned loopback port. Use `--no-browser` to print the
private session URL, and Ctrl+C to close the service. All scripts, styles and
scientific data are local. Each map request reconstructs one saved-time slice
on a 2° display mesh. A shared 32 MiB response cache supports revisiting
recent selections. PNG export captures the labelled 2D map with its selection
and colour scale. The self-contained HTML exporters remain separately available.

For SWE, the lower mesh follows terrain and the translucent upper mesh follows
the fluid surface. Both use the displayed vertical exaggeration. With slab
thickness h = h₀ + φ/g, the upper surface is H = hₛ + h relative to the same
datum as terrain. Cells touching an invalid thickness are masked. A field change
sets the colours on that shell. PE uses its native or vertically interpolated
geopotential height, with masks at the terrain and sampling limits.

Each map has an automatic colour range (all valid samples) or a fixed minimum
and maximum. Geometric day/night shading darkens only the globe; the flat map,
legend and readouts always show model values.

### Explorer: four coordinates, one role each

Every archive is opened through one description of its scalar quantities
(the manifest's `explorer` member). Each quantity records its name, symbol,
units, definition, source fields, how it is evaluated, its horizontal
support (a global field, a diagnostic point collection or none), its
vertical representations and native levels, and its own time axis. The
description also lists the products it supports and the reasons for the ones
it does not. Quantities need not share a mesh or a cadence.

In the **Explorer** panel, pick a quantity and give each coordinate
(longitude, latitude, vertical, time) a role:
- horizontal axis (X) or vertical axis (Y);
- fixed at a value or one saved time;
- reduced by a mean.

A coordinate the quantity lacks has the role *not a coordinate*. Two
plotting axes give a map or a section, one a line, and none a value. Presets
set the roles of the usual products. A longitude × latitude selection is
drawn on selection A's globe and map, which are the same product. Playback
(Play, the time slider) is a sequence of fixed selections at the actual
saved times. Combinations that are not offered are refused with their
reason, never approximated.

One geographic cursor links the views. Clicking the map or globe moves it,
and the map and globe draw the meridian and parallel that the sections
follow. By default the explorer follows selection A's saved time. Clicking a
level in a plot or in the column table selects it on the map. Clicking a
section moves the cursor.

| Product | Roles | Atmospheric archives | Columnar (airless) archives |
| --- | --- | --- | --- |
| Map | lon X, lat Y, vertical and time fixed | Exact band-limited synthesis on a 2° display mesh, then the declared vertical interpolation | The archived value of each native column at one native level; the map and its readout show cells, the globe interpolates between cell centres |
| Meridional / zonal section | lat or lon X, vertical Y, the other fixed | Exact synthesis on native full levels, 2° horizontal spacing, drawn on nominal σ | The native row of archived columns nearest the request; cells between their archived faces |
| Zonal mean | lat X, vertical Y, lon mean, time fixed or mean | Gauss–Legendre latitudes, plain mean over the native longitudes (shared with the Held–Suarez `SectionAccumulator`); also ψ below | Not offered |
| Column profile | vertical Y, lat and lon fixed | Exact synthesis at the point, or at the nearest native Gaussian node (default) | The archived column whose centre is nearest, at its cell centres on the nonuniform mesh |
| Time series | time X, lat, lon and the vertical fixed | The column at each saved state of the range (at most 400) | Every archived sample of the level's own record |
| Time × vertical | time X, vertical Y, lat and lon fixed | Same, drawn on nominal σ | Every archived cell sample; the skin strip uses the surface samples taken at the same steps |
| Recorded series | time X | Series recorded by the run, at their own cadence | |

ψ(φ, σ_{k+½}) = 2πa cos φ / g · Σ_{j≤k} [pₛ v_j] Δσ_j is integrated from
ψ = 0 at σ = 0. [pₛ v] is formed for each state before the longitude mean
and is never replaced by [pₛ][v]. Units are kg s⁻¹, positive for northward
mass transport above the interface. ψ at σ = 1 is the whole-column flux,
reported and not forced to zero.

**Vertical coordinates.** Atmospheric levels can be selected as:
- native full levels;
- nominal σ;
- pressure;
- geopotential height Z = Φ/g above the model datum;
- geopotential height above the local surface, Z − Zₛ.

Z − Zₛ uses the same inversion as constant Z, with the target Φ₀ = g(h + Zₛ)
in each column. It is a geopotential height, not a geometric altitude. Below
the terrain and between the terrain and the lowest full level it is masked,
never extrapolated.

A substrate column is selected by native level, by depth d (positive
downward) or by height relative to the surface, −d. Only native levels are
offered: the diagnosed skin at depth 0 and the cell centres. A value between
native depths is refused, because no reconstruction rule has been adopted.
The skin is its own native level: it is drawn as a separate strip above the
cells, or as a separate marker. It is never joined to the cells at a time it
was not sampled. Mixed atmosphere–substrate surfaces are not offered.

### Columnar (airless) archives

`tropoi view RUN_DIR` opens an archive written by `tropoi run airless` (or
`python -m tropoi.run.airless run`). It is recognized by its file contract
(`config.json`, `grid.npz`, `cycles.json` and the tier-2 directories), never by
the name of the body. The page shows the following:
- the column kind;
- for global grids, the grid shape and area weights;
- every kept tier-2 directory (`cycles/cycle-KKKK`, or `retained` when that
  cycle's own directory is gone);
- the completion status from `retained/status.json`.

The skin temperature and the four surface fluxes (incident shortwave,
absorbed forcing, emission, conduction into the first cell) follow the
surface-sample cadence. Cell temperatures follow the profile cadence, and
point runs also have the every-step record.

Each sample time is the forcing time of the step after which it was taken:
`(first + (k + 1) · every) · dt`. The timing comes from the directory's
`state.npz` or, in older archives, its `cycles.json` record. Every axis names
its source.

A diagnostic point collection has no map, section or area weights. Its
columns are shown one at a time.

Reads are memory-mapped and select the sample, level and columns before
copying. A map or a full time × depth history of a 1.6 GB 4° archive peaks
below 16 MB.

Sections, zonal means, ψ and time sections have their own automatic or fixed colour range,
separate from the map's, because the two often show different quantities. A
fixed minimum and maximum belongs to the plotted quantity (T, u, v, p, Z or ψ):
it stays while you step through saved times or switch between section presets,
and other quantities keep their own setting. The colour bar, the notes and the
exported PNG label a fixed range, and the notes count the values beyond it,
which take the end colours. A maximum that does not exceed the minimum falls
back to the automatic range.

Zonal means and ψ can average every saved state between a first and a last
selection, with equal weights. The plot states the count and the actual
cadence. The lowest full level is an air temperature at σ_{K−1}. The column
always shows it separately from the slab surface temperature. Pressure is
p_k = σ_k pₛ at the nominal level, and Z_k is the absolute Simmons–Burridge
geopotential over g at the hydrostatic level position. Every plot names its
archive directory and config hash, and notes that extrema between saved states
are not represented. Sections and reductions are offered only for pure-σ PE
archives (σ from 0 to 1) that record gravity. Zonal reductions and snapping to
native nodes also need a Gaussian latitude–longitude state grid. On other grids a
column is exact synthesis at the requested point. A flat PE run that records no
gravity (the `tropoi run pe` default) still opens on its native levels. Its T,
winds, ζ, δ and p are defined without g. Geopotential height, σ and pressure
surfaces and the vertical-structure products are withheld, and the manifest
says why. Its levels are drawn flat on the globe. Averaging costs about 0.2 s per T21L40 state and 2 s per T42L20
state read from a Windows drive under WSL. It runs outside the session lock,
so maps stay responsive. A request superseded in the browser still finishes
on the provider, and its result is cached.

CPU is the default. `--backend cuda` enables CuPy scalar synthesis for eligible
horizontal fields with CPU fallback. Wind and PE column evaluation use the
existing host routines. Browser graphics acceleration is independent of the
Python backend. Neutral illumination is available on every globe. Enabled
instantaneous or tidally locked insolation also supplies a geometric
substellar direction at each saved timestamp.
