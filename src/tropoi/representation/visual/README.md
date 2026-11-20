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

### Linked vertical structure (primitive-equation archives)

One geographic cursor links the views. Clicking the map or globe moves it.
The map draws it as an orange cross along the meridian and parallel that the
sections follow, and the globe draws the same lines. The **Vertical
structure** panel follows selection A's saved time, quantity and native level
unless you choose otherwise. Clicking a level in a plot or in the column
table selects that level on the map. Clicking a section moves the cursor.

| Product | Definition | Sampling |
| --- | --- | --- |
| Column profile | T, u, v, p and Z on every native full level against geopotential height, pressure or nominal σ, with a per-level table | Exact band-limited synthesis at the point. By default the point snaps to the nearest native Gaussian node, so slab Tₛ is the stored node value. Off-node Tₛ is bilinear and labelled so |
| Meridional section | Latitude × σ at the cursor longitude | Exact synthesis on native full levels, 2° horizontal spacing |
| Zonal section | Longitude × σ at the cursor latitude | Same |
| Zonal mean | [u], [v], [T] on native full levels | Run's Gauss–Legendre latitudes and plain mean over its `nlon` longitudes, shared with the Held–Suarez `SectionAccumulator` |
| Mass streamfunction | ψ(φ, σ_{k+½}) = 2πa cos φ / g · Σ_{j≤k} [pₛ v_j] Δσ_j, from ψ = 0 at σ = 0 | Same grid. [pₛ v] is formed for each state before the longitude mean and is never replaced by [pₛ][v]. kg s⁻¹, positive for northward mass transport above the interface. ψ at σ = 1 is the whole-column flux, reported and not forced to zero |

Zonal means and ψ can average every saved state between a first and a last
selection, with equal weights. The plot states the count and the actual
cadence. The lowest full level is an air temperature at σ_{K−1}. The column
always shows it separately from the slab surface temperature. Pressure is
p_k = σ_k pₛ at the nominal level, and Z_k is the absolute Simmons–Burridge
geopotential over g at the hydrostatic level position. Every plot names its
archive directory and config hash, and notes that extrema between saved states
are not represented. Sections and reductions are offered only for pure-σ PE
archives (σ from 0 to 1). Zonal reductions also need a Gaussian latitude–longitude
state grid. Averaging costs about 0.2 s per T21L40 state and 2 s per T42L20
state read from a Windows drive under WSL. It runs outside the session lock,
so maps stay responsive. A request superseded in the browser still finishes
on the provider, and its result is cached.

CPU is the default. `--backend cuda` enables CuPy scalar synthesis for eligible
horizontal fields with CPU fallback. Wind and PE column evaluation use the
existing host routines. Browser graphics acceleration is independent of the
Python backend. Neutral illumination is available on every globe. Enabled
instantaneous or tidally locked insolation also supplies a geometric
substellar direction at each saved timestamp.
