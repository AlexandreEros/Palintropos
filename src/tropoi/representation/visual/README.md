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
recent selections. The Explorer's *Save PNG* exports its plot (any result,
longitude × latitude included) with its selection and colour scale; the
globe's exports the globe as drawn, with its title and colour scale. The
self-contained HTML exporters remain separately available.

**Page layout.** The Explorer is the primary, analytical workspace and the
page's only flat plotting surface: longitude × latitude maps, sections,
profiles and histories are all its results. The 3D globe below it is a
complementary view of the same selection: it draws the Explorer's quantity at
the Explorer's saved time on the Explorer's fixed vertical surface (when the
Explorer plots or reduces the vertical coordinate, its fixed-surface setting is
kept and clicking a level in a section or profile moves it; when it reduces or
plots time, the globe shows the one saved time the transport selects, and says
so). The globe's Previous/Play/Next move the Explorer's saved time. *Compare
selections* adds a second globe that is independent by design. There is no
second analytical renderer.

For SWE, the lower mesh follows terrain and the translucent upper mesh follows
the fluid surface. Both use the displayed vertical exaggeration. With slab
thickness h = h₀ + φ/g, the upper surface is H = hₛ + h relative to the same
datum as terrain. Cells touching an invalid thickness are masked. A field change
sets the colours on that shell. PE uses its native or vertically interpolated
geopotential height, with masks at the terrain and sampling limits.

Each globe has an automatic colour range (all valid samples) or a fixed minimum
and maximum. Geometric day/night shading darkens only the globe; its legend and
readouts always show model values.

### Explorer: four coordinates, one role each

Every archive is opened through one description of its scalar quantities
(the manifest's `explorer` member). Each quantity records its name, symbol,
units, definition, source fields, how it is evaluated, its horizontal
support (a global field, a diagnostic point collection or none), its
vertical representations and native levels, and its own time axis.
Quantities need not share a mesh or a cadence.

Capability is described **per coordinate**, never per plot type:
`quantities[q].coordinates[c]` lists the roles coordinate `c` accepts, the
reductions it defines and, for the vertical coordinate, the representations
it can be fixed at (`fixed`), plotted along (`axis`), or plotted along only
for a single column at a single time (`per_sample`). A role a coordinate
does not accept carries its reason (`unavailable`).

In the **Explorer**, pick a quantity and give each coordinate
(longitude, latitude, vertical, time) a role:
- horizontal axis (X) or vertical axis (Y);
- fixed at a value or one saved time;
- reduced by a statistic: Mean, Minimum or Maximum (chosen per coordinate).

**Quantity selector.** Quantities are grouped by the registry's catalogue group
(`presentation.group`: thermodynamics; velocity and wind; dynamics;
geopotential and geometry; surface; archived diagnostics and time series),
sorted by name and searchable by name, symbol, unit or group. The card under
the selector gives the name, symbol, unit, definition, whether the quantity is
stored, derived or an archived diagnostic, and its actual coordinates (native
levels, saved samples). The grouping comes from the registry
(`representation.visual.quantities`), not from the page.

**Changing the quantity keeps the analytical question.** Roles, fixed values,
statistics, the time interval, the vertical representation and the axis
display are kept. A surface quantity under a fixed vertical coordinate simply
has none (the setting is kept for 3-D quantities, and the role selector says
so). If the new quantity cannot answer the configuration (a coordinate it
lacks used as an axis or reduced, a statistic it does not define, a
representation it does not offer), the Explorer keeps the configuration, marks
the coordinates concerned, explains why, and offers an explicit *Adapt*
button; nothing is reset, re-preset or broadcast silently. The configuration
also travels with the archive selector.

A coordinate the quantity lacks has the role *not a coordinate*. The result
follows from the roles alone: two plotting axes give a two-dimensional field,
one a line, none a single value. Longitude × latitude is one such
assignment; it is drawn in the explorer like any other field. Presets
(`explorer.presets`) only fill in roles: they enable nothing, and every
assignment they produce can be made by hand. Playback (Play, the time
slider) is a sequence of fixed selections at the actual saved times. Roles a
coordinate does not accept are disabled in the page, with their reason, and
refused by the server, never approximated.

One geographic cursor links the views. Clicking the globe or a field moves
it; the globe draws the meridian and parallel that the sections follow and a
longitude × latitude field draws both as a crosshair. Clicking a level in a
plot or in the column table makes it the fixed surface (the globe draws it).
Clicking a field moves the cursor along the horizontal coordinates it shows.

**Sample sets.** Each role maps a coordinate to the points at which the
quantity is evaluated (`selection.evaluate`; the providers own the sets):

| Role | Atmospheric archives | Columnar (airless) archives |
| --- | --- | --- |
| X or Y (longitude, latitude) | The 2° display mesh (longitudes 0…360 with the closing meridian), exact band-limited synthesis | Every archived row or meridian |
| X or Y when the other horizontal coordinate is reduced | The native Gaussian nodes, where the mean is a quadrature | (as above) |
| Fixed (longitude, latitude) | The point, exact synthesis; or the nearest native Gaussian node when *Snap* is on | The nearest archived row or meridian; the nearest column on the sphere when both are fixed |
| Reduce (longitude, latitude) | The native Gaussian state grid | Every archived row or meridian |
| X or Y (vertical) | Native full levels, drawn on nominal σ; p, Z or Z − Zₛ only for a single column at a single time (their level positions move with the column and the state) | Native cell centres; the skin as its own row |
| Fixed (vertical) | A native level, or a σ, p, Z or Z − Zₛ surface interpolated linearly in ln σ (masked, never extrapolated) | A native level only |
| X, Y or Reduce (time) | Every saved state from first to last (at most 400) | Every archived sample of the level's own record (at most 20 000 for one column, 400 otherwise) |

The numerical definition of a value depends only on these sets: the same
role assignment gives the same numbers whatever preset or path produced it.

**Reductions** (`mean`, `min`, `max`; each payload states its rule in
`coordinates`). Every reduction acts on the native samples the provider
evaluates for it (saved states, native grid nodes, native levels), never on
display-mesh or interpolated values. The rules below were approved by the
owner for this phase (2026-11-27).

| Coordinate | Mean | Minimum / Maximum |
| --- | --- | --- |
| Time (interval first…last, chosen by the user) | Time-weighted with *cell weights*: each saved sample stands for the interval around it, wᵢ = (tᵢ₊₁ − tᵢ₋₁)/2, ends mirrored (w₀ = t₁ − t₀). For a regular cadence all weights are equal and the plain 1/N mean is computed, bit for bit; the payload reports the cadence, the weighting and, when irregular, the weights | Extremum over the saved samples of the interval |
| Longitude | Plain mean over the equally spaced native longitudes (atmosphere: exact for a band-limited field; columns: a row's cells have equal areas) | Extremum over the native longitudes |
| Latitude | Area-weighted: Gauss–Legendre weights (∝ cos φ dφ) on the native Gaussian latitudes; the recorded column area weights of a columnar archive | Extremum over the native latitudes (unweighted) |
| Longitude and latitude | The area mean (the product of the two) | Extremum over the native grid |
| Vertical | Not defined (refused with its reason): a mean over levels needs a mass (pₛΔσ) or thickness weighting rule that has not been approved | Extremum over the native full levels (atmosphere); not offered over substrate cells |

Masks: a masked sample makes every mean that includes it masked (no mean is
taken over a subset of the requested samples). Minimum and maximum use the
valid samples only and are masked only where every sample is masked; the
payload counts the masked samples they excluded. Minimum and maximum are the
extrema of the sampled values, not of the continuous field between samples
(or between saved states). Reductions over different coordinates are applied
in one stated order: longitude, latitude, vertical (within each saved state),
then time; with different statistics the order matters and the notes say so.
On native σ levels each row is the statistic of the level's values; its p and
Z (and every companion that locates rows) are the means of that level's p_k
and Z_k, not values on a surface of constant p or Z. A companion that is a
value of the quantity (the airless skin, the slab Tₛ marker) takes the
quantity's statistic. These *statistics* are unrelated to the colour range's
minimum and maximum.

**Air density** ρ = p/(R T) is an ordinary derived quantity (thermodynamics
group, kg m⁻³). p and T are evaluated at the same samples (place, surface,
saved state) and ρ is formed from them before any reduction (so a time mean
of ρ is the mean of ρ, not p̄/(R T̄)). On a native level p = σ_k pₛ (the
nominal full-level pressure the viewer shows); on a surface of constant p,
the surface's p; elsewhere the interpolated p. R is the gas constant the run
records for its atmosphere (`r_dry` or `gas_constant`; Held–Suarez 286.857,
the Mars CO₂ run 191.8 J kg⁻¹ K⁻¹), the mixture constant of a
fixed-composition gas. ρ is advertised only when such a constant is recorded
and no stored level field other than the dry prognostics (ζ, δ, T) exists: a
moisture or composition tracer would make one R the wrong mixture constant.
No default value is ever assumed.

**Archived statistical series (audit).** The Held–Suarez archive records
daily series in `series.json`. They were compared with the general reductions
(`tests/test_viewer_reductions.py`):

| Series | Definition in the run | Relation to the engine | Kept? |
| --- | --- | --- | --- |
| `t_max`, `t_min` | max/min of T over the native state grid at every level, daily | Equal to the engine's longitude, latitude and vertical maximum/minimum at the saved states (to 1e-9 relative) | Kept: daily cadence, while the engine sees the 10-day saved states only |
| `mean_T` | Gauss-area mean of Σ_k Δσ_k T_k, daily | Equal to Δσ weights applied to the engine's area-mean σ profile; the engine itself defines no vertical mean (refused) | Kept: a distinct (Δσ-weighted, daily) diagnostic |
| `mean_ps` | Gauss-area mean of pₛ, daily | Equal to the engine's longitude and latitude mean of pₛ at the saved states (to 1e-9 relative) | Kept: daily cadence |
| `mean_lnps`, `ke`, `max_abs_u`, `comp_mode` | Daily global diagnostics of the run | Not a single general reduction of a viewer quantity (ln pₛ is shown as pₛ; ke needs the Δσ vertical mean; max |u| combines max u and −min u; comp_mode is not a field statistic) | Kept |

The PE runner's `diagnostics/timeseries.csv` extrema (`t_min`, `t_max`) are
taken over every model sampling, including the product-space transform grid,
so they are not the extrema of the saved states on the state grid; they are
kept. No archived series was removed: none is equivalent in domain, cadence
and definition to a general reduction. A diagnostic point
collection is not a global field: its horizontal coordinates can only be
fixed, and it has no area weights. ψ is defined from the longitude mean
[pₛ v], so its longitude role is always the mean; over time, ψ is formed from
the time mean of [pₛ v].

**Display.** How a vertical axis is drawn is a display choice, separate from
the physical coordinate and from the interpolation space: linear or
logarithmic (σ and p default to logarithmic, heights and depths to linear),
increasing upward or downward. Changing it sends no request. Native levels are
drawn as bands between the model's own σ interfaces (or the archived cell
faces), whatever the scale.

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
selection (cell weights, equal for a regular cadence). The plot states the
count, the actual cadence and the weighting. The lowest full level is an air temperature at σ_{K−1}. The column
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
