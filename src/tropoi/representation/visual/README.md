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
