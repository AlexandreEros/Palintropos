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

CPU is the default. `--backend cuda` enables CuPy scalar synthesis for eligible
horizontal fields with CPU fallback. Wind and PE column evaluation use the
existing host routines. Browser graphics acceleration is independent of the
Python backend. Neutral illumination is available on every globe. Enabled
instantaneous or tidally locked insolation also supplies a geometric
substellar direction at each saved timestamp.
