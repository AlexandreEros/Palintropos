# Spectral operators

`spectral_operators.py` holds `SpectralOperators`, the calculus every model
core uses: derivatives, the inverse Laplacian, velocity reconstruction,
vector curl/divergence and pseudospectral products. Every method takes and
returns coefficients in the shared `(l_max+1, l_max+1)` layout of
[transforms](../transforms/README.md). Nothing here knows which grid family
it runs on: products go through the backend's `ProductSpace`
([`spherical_backend.py`](../spherical_backend.py)), so the same code runs
on geodesic and Gauss grids.

`Planet` builds one instance per model as `planet.so`.

## Operators

| Method | Mathematics | Notes |
| --- | --- | --- |
| `d_lambda_coeffs` | `∂/∂λ Y_lm = i m Y_lm`, divided by `R` | Diagonal; exact |
| `sin_theta_d_theta_coeffs` | `sin θ ∂Y_lm/∂θ = C⁺_lm Y_{l+1,m} + C⁻_lm Y_{l-1,m}` (θ = colatitude) | Couples neighbouring degrees; see the truncation note below |
| `adjoint_sin_theta_d_theta_coeffs` | Transpose `Sᵀ` of that coupling | Moves a meridional derivative onto the basis in a quadrature inner product |
| `inv_laplacian` | `∇⁻² Y_lm = -R²/(l(l+1)) Y_lm`, with `l = 0` set to zero | Delegates to [`laplacian.py`](../laplacian.py), the single inversion policy |
| `velocity_from_streamfunction` | `u = -(1/R) ∂ψ/∂φ`, `v = (1/(R cos φ)) ∂ψ/∂λ` | Synthesized on the state grid |
| `jacobian_pseudospectral` | `J(a, b) = (a_snt b_λ - a_λ b_snt) / cos²φ`, i.e. `u_a · ∇b` | Formed pointwise on the product sampling, analyzed once, then cut to the 2/3 band |
| `vector_curl_div_spectral` (and `_batch`) | Weak form: `(div F)_lm = -∫ ∇Y*_lm · F dΩ`, `(curl F)_lm` from the rotated field | **Production** path for PE momentum; one analysis, no derivative round trip |
| `vector_curl_div_roundtrip` | Analyze each component, differentiate, assemble on the grid, analyze again | **Reference** path kept for comparison |

The weak-form derivation, its one-degree-extended analysis and its
geodesic accuracy envelope are in the
[PE formulation](../../temporal/tendencies/PRIMITIVE_EQUATIONS.md). Signs,
normalization and the `C±` recurrence are in
[CONVENTIONS.md](../CONVENTIONS.md#4-spectral-derivatives).

## Products and dealiasing

`product_quadrature` selects where pointwise products are evaluated:

- `"coarse"`: on the state sampling. Its quadrature cannot integrate the
  degree-`2L` content of a quadratic product, so aliasing lands in the
  retained band.
- `"fine"`: on an overresolved sampling chosen by the backend. Gauss uses
  the 3/2 rule (`nlon >= 3L+1`, `nlat >= ceil((3L+1)/2)`), which is exact
  for quadratic products. Geodesic uses a resolution-`r+1` co-grid, which
  reduces but does not remove aliasing (about 6× smaller invariant drift at
  resolution 4, `L = 21`).

Unsupported modes raise `ValueError`; there is no silent fallback. After
analysis, products are truncated once to `floor(2L/3)` (the `dealias` flag
keeps its historical name, but the operation is a truncation). Measurements
are in the [R-3 characterization](../../../../docs/validation/r3_characterization.md)
and the [lat–lon backend review](../../../../docs/validation/latlon_backend_review.md).

## Limits

- **Top-degree derivative.** `sin_theta_d_theta_coeffs` has no slot for
  degree `l_max + 1`, so the meridional derivative of degree-`l_max`
  content is truncated. PE stores `l_max = L + 1` and keeps degree `L + 1`
  as a derivative buffer
  ([retained truncation](../../temporal/tendencies/RETAINED_TRUNCATION.md)).
  BVE and SWE store through `L` and rely on the 2/3 product cut instead.
- **Poles.** Velocity reconstruction sets winds to zero where
  `|cos φ| < 1e-6`, and other metric divisions assume no sample sits on a
  pole. The production grids avoid exact poles; arbitrary point sets that
  include them are not supported.
- **Shared state.** `d_lambda_coeffs` writes to a cached buffer on each
  call. One instance should not be used from several threads at once.
- `vector_curl_div_roundtrip` still cites "handoff Section 1.4b"; that
  handoff was folded into the PE formulation linked above.
