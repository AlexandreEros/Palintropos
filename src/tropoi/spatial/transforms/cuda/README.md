# CUDA kernel sources

The GPU transforms compile their basis functions from the `.cu` files in
this directory at run time. The sources ship with the package
(`[tool.setuptools.package-data]` in `pyproject.toml`) and are located
through `importlib.resources`, so an installed distribution needs no
source checkout.

| File | Kernel | Consumer |
| --- | --- | --- |
| `sh_matrix.cu` | `generate_sph_harm_basis`: one thread per point fills that point's row of the complex basis matrix `Y`, all `(l, m)` with `m >= 0`, linear index `l(l+1)/2 + m` | `PointSetSphericalHarmonics` (both production backends) |
| `legendre_stages.cu` | `legendre_analysis` / `legendre_synthesis`: the latitude stage of the separable Gauss transform. One thread per output, contracting over latitude (analysis) or degree (synthesis) against a rectangular `(nlat, l_max+1, l_max+1)` real table, with the complex field as two real accumulators | `SeparableGaussSphericalHarmonics` |
| `legendre.cu` | `compute_legendre`: unnormalized `P_l^m(x)` for one `(l, m)` across all points | Legacy `LatLonSphericalHarmonics` only |
| `cuda_utils.py` | `load_cuda_source(name)` and `raw_module_from_cuda(name)` → `cupy.RawModule` | Both transforms and `tropoi clear-cache` |

## Numerics

Both kernels start from `P_m^m = (-1)^m (2m-1)!! (1-x^2)^(m/2)` (the
Condon–Shortley phase lives here) and climb in degree with the standard
three-term recurrence

```
P_l^m = [(2l-1) x P_{l-1}^m - (l+m-1) P_{l-2}^m] / (l-m)
```

`sh_matrix.cu` then multiplies by `N_lm = sqrt((2l+1)/(4π) (l-m)!/(l+m)!)`,
using `lgamma` for the factorial ratio, and by `e^{i m λ}`. Colatitude is
`π/2 - lat`, so `x = sin(lat)`.

`legendre_stages.cu` evaluates no basis function of its own: its tables are
built by calling `generate_sph_harm_basis` at longitude zero, where the
`e^{i m λ}` factor is 1 and the real part is exactly `N_lm P_l^m(x_j)` (the
same tabulation `representation/visual/evaluate.py` uses host-side). So the
separable transform inherits this recurrence's accuracy and its overflow
limit rather than introducing a second set of numerics.

The recurrence runs on **unnormalized** values. `(2m-1)!!` exceeds the
double-precision range at `m = 151`, so equatorial rows overflow from
there on. That is far above the truncations in use (`l_max` up to about
45; T42 for Held–Suarez), but going past roughly `l_max = 150` needs a
normalized recurrence. Accuracy at high order has not been characterized
separately from the transform-level measurements in
[CONVENTIONS.md](../../CONVENTIONS.md).

## Compilation and caching

CuPy compiles each module the first time it is built and caches the binary
under `~/.cupy/kernel_cache`. `tropoi clear-cache` deletes that cache and
recompiles `sh_matrix` to confirm the toolchain works; use it after a CUDA
or driver change. A compile failure surfaces at transform construction,
not at import time.
