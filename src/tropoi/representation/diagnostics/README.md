# Scientific diagnostics

This directory turns model states into numbers: conservation and stability
time series during a run, and climate, level and spectral reductions of
saved states afterwards. It never changes a state, and it computes from the
model's own coefficients and quadrature weights, never from interpolated
plotting fields. [Visual](../visual/README.md) draws figures from these
numbers; the figures are not evidence on their own.

There are two kinds of module:

| Kind | Modules | When | Runs on |
| --- | --- | --- | --- |
| Run-time recorders | `bve.py`, `swe.py`, `pe.py` | After every accepted step | GPU, inside the runner |
| Saved-state readers | `spectral.py`, `pe_levels.py`, `pe_surfaces.py`, `held_suarez_sections.py` | After a run, from saved frames | CPU (NumPy only) |

## Run-time recorders

Each recorder appends one row per accepted step to
`RUN/diagnostics/timeseries.csv`; the BVE recorder also writes degree
spectra to `spectra.npz`. A matching `plot_*` function reads those files
later, so recording never depends on plotting.

| Core | Columns | What is conserved, and how it is measured |
| --- | --- | --- |
| BVE | speed, CFL, circulation, energy, relative and absolute enstrophy, `ζ` max/RMS, `energy_l1`, high-degree enstrophy fraction, transform round-trip residual | Computed in spectral space: `E = ½ R⁴ Σ_l P_l(ζ)/(l(l+1))`, `Z = ½ ∫ζ² dA`. On a rotating planet the invariant is **absolute** enstrophy, with `f` entering only the `(1, 0)` coefficient. `grid_diagnostics` gives the same quantities by quadrature for comparison |
| SWE | wind, characteristic speed, CFL, total-geopotential extrema, mass, energy, minimum depth, free-surface range, terrain height, L2 norms of `ζ`, `δ`, `φ` | Mass `R²(4πΦ0 + sqrt(4π) Re φ_00)` is computed spectrally, so it shows the conserved quantity itself rather than geodesic quadrature leakage. Extrema cover every sampling the model uses. `potential_enstrophy` gives `∫(ζ+f)²/(2h) dA` |
| Dry PE | `T` and `p_s` extrema, wind, `|ζ|`, `|δ|`, characteristic speed, diagnostic Courant number, mass and its relative drift | Mass is `∫p_s dA` by state-grid quadrature. There is **no total-energy column**: the PE core has no independently tested global energy diagnostic yet. The Courant number is reported, not used for control, because the PE runner takes a fixed timestep |

The degree power `P_l = Re(a_l0)² + 2 Σ_{m>0} |a_lm|²` uses only `Re(a_l0)`
because synthesis discards `Im(a_l0)`. The derivations are in the `bve.py`
module docstring and [CONVENTIONS.md](../../spatial/CONVENTIONS.md).

## Saved-state readers

| Module | Produces | Used by |
| --- | --- | --- |
| `spectral.py` | Kinetic energy per mode `E(l, m) = (R⁴/2)(P_ζ + P_δ)/(l(l+1))` from the rotational/divergent split, plus mean degree, mean zonal wavenumber, entropy and effective mode count | Visual quantity pages |
| `pe_levels.py` | Temperature, pressure `σ_k p_s`, absolute Simmons–Burridge geopotential and geopotential height on native full levels, on a lat–lon display mesh | Atmosphere view, PE archive context |
| `pe_surfaces.py` | The same fields on a fixed pressure, geopotential height or sigma surface, interpolated linearly in `ln σ`, with per-node masks for nodes where the surface is underground, above the top full level, or between the lowest full level and the ground | Atmosphere view |
| `held_suarez_sections.py` | The six zonal-mean latitude–sigma sections (`[u]`, `[T]`, `[u*v*]`, `[v*T*]`, EKE, `[T*²]`) on the run's Gaussian grid, with sample selection, band-limit checks, mass drift and SHA-256 of the input states | [Held–Suarez climate figure](../visual/HELD_SUAREZ_CLIMATE.md) |

Eddy products in the Held–Suarez sections are formed per state, against
that state's own zonal mean, before time averaging, the same way the run's
accumulator forms them. EKE needs `[v*²]`, which the accumulator does not
keep, so it comes only from saved states.

## Limits

- Snapshot-based numbers (`spectral.py`, the PE readers) describe saved
  times only. They are not per-step measurements, and figures built from
  them must say so.
- Derived PE surfaces are display products. The solver never produces a
  pressure or height surface, and the vertical interpolation is linear in
  `ln σ`, not a model calculation. The rules for where a full level sits
  are in the `pe_surfaces.py` docstring and the
  [atmosphere view guide](../visual/ATMOSPHERE_VIEW.md).
- PE mass uses grid quadrature, so on geodesic grids it includes the
  quadrature's error. A resting uniform `p_s` still integrates exactly.
- Docstrings that say "design doc Section N" mean the
  [PE formulation](../../temporal/tendencies/PRIMITIVE_EQUATIONS.md),
  whose section numbers are unchanged.

[Validation](../../../../docs/VALIDATION.md#conservation-diagnostics-and-analytic-guarantees)
explains which conservation results are analytic and which are measured.
