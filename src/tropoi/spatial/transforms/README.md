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
| `fast_geodesic_sh.py` | `PointSetSphericalHarmonics`: dense basis matrix `Y` (n_points × n_basis) built on GPU by `cuda/sh_matrix.cu`; analysis and synthesis are matrix products. Works on any point set given weights; batched `transform_batch` / `inv_transform_batch` analyze K fields in one GEMM | Both production backends |
| `optimized_geodesic_sh.py` | `GeodesicSphericalHarmonics`: wraps the point-set transform with geodesic weights (`"voronoi"` default, `"uniform"`, or least-squares `"optimize"` with an on-disk cache) and warns below 6 points per basis function | Geodesic backend, `Planet` |
| `host_synthesis.py` | NumPy-only exact synthesis on a lat–lon mesh, plus `sin(theta) d/dtheta` and wind from vorticity/divergence. Never imports CuPy | Saved-state previews, PE level/section diagnostics |
| `spherical_harmonics.py` | `LatLonSphericalHarmonics`: legacy Simpson-rule transform with per-(l, m) Legendre kernels from `cuda/legendre.cu` | Comparison code and decorative terrain only |
| [cuda](cuda/README.md) | Kernel sources and their loader | The GPU classes above |

The production Gauss transform, `GaussLatLonSphericalHarmonics`, sits in
[grids/latlon_grid.py](../grids/README.md) with its geometry. It feeds the
same `PointSetSphericalHarmonics` matrix path with exact tensor-product
weights. Backends pair a grid with a transform in
[`spherical_backend.py`](../spherical_backend.py), which also owns the
product sampling.

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
