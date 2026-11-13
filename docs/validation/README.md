# Validation and measurement records

The [validation overview](../VALIDATION.md) states accepted results. These
records keep configurations, tolerances, provenance, receipts, checksums,
figures and recipes. They are dated evidence; do not treat an old experiment's
implementation paths or fixed timesteps as current API guidance.

| Record | What it supports |
| --- | --- |
| [Historical run preservation](run_archive_2026-11-12/README.md) | Full-cloud checksums, published cores, source matching and rescued interactive evidence |
| [Williamson 5 vs MRI-JMA](williamson5_mri_2026-07-30.md) | Day-zero contract, 15-day model intercomparison, figures and external artifact checksums |
| [Gauss/geodesic comparison](latlon_backend_review.md) | Quadratic quadrature derivation and matched-timestep RH4 measurements |
| [Fine-product characterization](r3_characterization.md) | Factorized experiments, preregistered predictions, corrected attribution and residual band mismatch |
| [Preset support](preset_support_characterization.md) | Storage/nonlinear-cut support gates and measured frozen-mode behavior |
| [MOLA terrain](mola_terrain_2026-10-22/README.md) | Imported real terrain, rest balance and preview provenance |
| [Atmosphere view](atmosphere_view_2026-10-23/README.md) | Saved-state visualization/host reconstruction examples |
| [Mars evolution](mars_evolving_2026-10-27/README.md) | Perturbed dry-CO2 run, resting control, timestep twin and conservation measurements |
| [Held–Suarez climate figure](held_suarez_climate_2026-11-04/README.md) | Rendering sample/scale/sidecar evidence; scientific verdicts remain in [HS status](../held_suarez/STATUS.md) |
| [Pressure slices](pressure_surface_2026-11-07/README.md) | Fixed-pressure interpolation/masks, export cost and browser playback |
| [Interactive pressure](interactive_pressure_2026-11-08/README.md) | Browser interpolation/quantization, latency and payload measurements |
| [Derived surfaces](derived_surfaces_2026-11-08/README.md) | Geopotential-height/sigma surfaces and vorticity/divergence measurements |

The adjacent Williamson artifact directories retain machine-readable
receipts and committed smoke evidence. Most long-run raw data is local or
externally mirrored, as each report states. New claims need their own
configuration and evidence; no general terrain/climate/convergence claim
follows from this collection.
