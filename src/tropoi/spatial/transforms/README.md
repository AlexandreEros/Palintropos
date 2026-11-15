# Spherical-harmonic transforms

A transform turns point values on the sphere into spherical-harmonic
coefficients (analysis) and back (synthesis). Every model state in the
repository is stored in the coefficient layout these classes produce, so
this directory fixes the basis. It does not choose grids, evaluate
products or advance time.

## One basis, several evaluators

All evaluators use the same complex orthonormal harmonics with the
Condon–Shortley phase, stored for `m >= 0` in a dense `(l_max+1, l_max+1)`
array (axis 0 = degree, axis 1 = order). Real fields are implied:

```
f = sum_l a_l0 Y_l0 + 2 Re sum_{m>0} a_lm Y_lm
```

Entries with `m > l` are padding. Point-set analysis is the weighted
projection `a = Y^H (w * f)`. It inverts synthesis only within the
quadrature envelope of the sampling. [CONVENTIONS.md](../CONVENTIONS.md)
defines normalization, signs and layout and records measured envelopes.

| Module | Role | Used by |
| --- | --- | --- |
| `fast_geodesic_sh.py` | `PointSetSphericalHarmonics`: dense basis matrix `Y` (n_points × n_basis) built on GPU by `cuda/sh_matrix.cu`; analysis and synthesis are matrix products. Works on any point set given weights; batched `transform_batch` / `inv_transform_batch` analyze K fields in one GEMM | Both production backends; numerical reference and fallback everywhere |
| `separable_gauss_sh.py` | `SeparableGaussSphericalHarmonics`: the same basis evaluated separably on a Gauss × uniform-longitude grid — a cuFFT `rfft`/`irfft` longitude stage and a `cuda/legendre_stages.cu` latitude stage, so cost is `nlat × n_basis` instead of `n_points × n_basis`. Opt-in; see below | Gauss lat–lon backend when selected |
| `optimized_geodesic_sh.py` | `GeodesicSphericalHarmonics`: wraps the point-set transform with geodesic weights (`"voronoi"` default, `"uniform"`, or least-squares `"optimize"` with an on-disk cache) and warns below 6 points per basis function | Geodesic backend, `Planet` |
| `host_synthesis.py` | NumPy-only exact synthesis on a lat–lon mesh, plus `sin(theta) d/dtheta` and wind from vorticity/divergence. Never imports CuPy | Saved-state previews, PE level/section diagnostics |
| `spherical_harmonics.py` | `LatLonSphericalHarmonics`: legacy Simpson-rule transform with per-(l, m) Legendre kernels from `cuda/legendre.cu` | Comparison code and decorative terrain only |
| [cuda](cuda/README.md) | Kernel sources and their loader | The GPU classes above |

The production Gauss transform, `GaussLatLonSphericalHarmonics`, sits in
[grids/latlon_grid.py](../grids/README.md) with its geometry. It pairs exact
tensor-product weights with **one of two evaluators** of the same basis,
chosen by its `transform_backend` argument (`"separable"`, the default, or
the deprecated `"dense"`); everything else talks to the wrapper and does not
care which.
Backends pair a grid with a transform in
[`spherical_backend.py`](../spherical_backend.py), which also owns the
product sampling and propagates the evaluator choice to it.

## Choosing an evaluator on the Gauss grid

`"separable"` is the default. It exploits the tensor-product structure the
dense path throws away: with uniform longitudes and longitude-independent
weights, `Y_lm(lat_j, lon_k) = N_lm P_l^m(x_j) e^{i m lon_k}`, so the
longitude sum is a DFT and the latitude sum no longer carries the `nlon`
factor.

`"dense"` is the historical basis-matrix path. It is **deprecated** and
raises a `DeprecationWarning` whenever a Gauss grid lands on it — asked for
or fallen back to, so a slow run is never quiet about why. It is kept for
three jobs: the numerical reference the separable path is measured against,
the fallback for grids that cannot factorize, and reproducing runs recorded
before the evaluator became selectable.

- **Selection.** `transform_backend=`, or `TROPOI_SH_BACKEND` to point an
  existing run configuration at either evaluator without editing call sites.
  Either way the choice is recorded in run manifests by
  `SphericalGridBackend.describe` (`transform_backend`,
  `transform_evaluator`) and in every checkpoint by
  `run.held_suarez.experiment.transform_identity`, which refuses a resume
  that would change it.
- **Replaying a saved run** uses the evaluator its manifest records, not the
  current default; a run from before the choice existed records none and is
  replayed on the dense path it ran with
  (`representation.visual.snapshot._recorded_transform_backend`). Changing
  the default therefore does not change how existing runs are re-read.
- **Applicability.** `separable_is_applicable` is the single gate: uniform
  longitudes, weights that do not vary along longitude, and `2*l_max < nlon`
  (every retained zonal wavenumber needs its own real-FFT bin below Nyquist).
  Production grids clear it — the state grid is `nlon >= 2 l_max + 1`, the
  3/2-rule product grid `nlon >= 3 l_max + 1`. A grid that does not falls
  back to the dense evaluator **with a warning**, never silently. The
  unstructured geodesic grid cannot factorize at all and is unaffected.
- **Agreement, not bit-identity.** The Legendre values come from the same
  `cuda/sh_matrix.cu` recurrence (tabulated at longitude zero), so only the
  summation order differs. Measured agreement against the dense path is
  ~1e-14 relative on analysis, synthesis, round trips and a full PE
  tendency; the tolerance in `tests/test_separable_transform.py` is 1e-13.
  It is therefore **not** a drop-in for reproducing a pinned hash.
- **Measured cost** (MX110, float64, 25 levels, `cupyx.profiler.benchmark`):
  at T42 the batched product-grid synthesis is 364 ms → 2.8 ms and a full
  Held–Suarez-style PE step 10.76 s → 0.67 s (16×); at T21 the step is
  0.864 s → 0.372 s (2.3×). The dense basis matrix is also 128 MB per object
  at T42 against ~1 MB of Legendre tables, which is why the dense T42 path
  needs GEMM chunking to survive the Windows TDR and this one does not.
- **Credit.** The factorization and the normalization convention follow
  NVIDIA [torch-harmonics](https://github.com/NVIDIA/torch-harmonics)
  (BSD-3-Clause), which uses the same
  `sqrt((2l+1)/(4π) (l-m)!/(l+m)!)` normalization. No code is vendored: its
  CUDA extensions need compute capability ≥ 7.0 and do not cover this stage,
  and PyTorch is deliberately not a dependency.

## Contracts worth knowing

- **Frozen arithmetic.** The scalar `transform` / `inv_transform` paths are
  kept bit-stable because pinned run hashes depend on them. Use the
  `*_batch` methods for new level-stacked work instead of changing the
  scalar code.
- **Synthesis discards `Im(a_l0)`.** A real field has real `m = 0`
  coefficients. Diagnostics count only `Re(a_l0)` for the same reason.
- **Exactness depends on the grid.** Gauss analysis of a band-limited
  field is exact to rounding when `nlat >= L+1` and `nlon >= 2L+1`.
  Geodesic analysis is approximate and depends on orientation. The
  6-points-per-basis warning guards against gross under-resolution; it is
  not an error bound.
- **Weight cache.** The `"optimize"` mode caches weights only when the
  caller passes `cache_dir`. Its key covers resolution, `l_max`, point
  count and basis identity, not point coordinates, so delete the cached
  file after changing grid orientation or subdivision. Production uses
  Voronoi weights and never reads this cache. (`tropoi clear-cache` clears
  CuPy's compiled-kernel cache, not these weights.) The class docstring's
  claim that weights are optimized by default predates the Voronoi
  default.
- **Legacy Simpson transform.** `LatLonSphericalHarmonics` integrates over
  an open longitude interval, so it misses the closing panel and biases
  analysis low by about `1/nlon` (a 33×65 constant-field probe gave
  `a_00 = 3.4904` instead of `sqrt(4π) = 3.5449`). It is a comparison
  implementation, not an accuracy reference.

## Cost and requirements

The GPU classes need CuPy. The basis matrix holds `n_points × n_basis`
complex values; at geodesic resolution 5 and `l_max = 45` that is about
10 242 × 1 081 entries (about 177 MB). Analysis and synthesis cost one
dense matrix product each, so per-step cost grows with both grid size and
truncation.

[Validation](../../../../docs/VALIDATION.md) records the matched
geodesic/Gauss comparison and orthogonality measurements.
[Operators](../operators/README.md) build derivatives and products on top
of these transforms.
