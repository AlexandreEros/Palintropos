# Spherical numerics and coefficient conventions

These conventions are shared by the BVE, SWE and dry hydrostatic PE cores.
The [spatial layer](README.md) owns geometry, transforms, column operators,
states and initialization; [temporal tendencies](../temporal/tendencies/README.md)
assemble the equations. SI units, a spherical dynamical surface, eastward
`u`, northward `v`, latitude `phi`, colatitude `theta = pi/2 - phi`, and
longitude `lambda` are used throughout. Oblateness and procedural elevation
from `Planet.generate` do not enter the dynamics.

## BVE equation and signs

Implemented in [`BarotropicVorticity`](../temporal/tendencies/barotropic.py):

```
dzeta/dt = -J(psi, zeta + f) + nu lap(zeta) + F
lap(psi) = zeta
f = 2 Omega sin(phi)
u = -psi_phi/R,  v = psi_lambda/(R cos(phi))
J(a,b) = (a_lambda b_phi - a_phi b_lambda)/(R^2 cos(phi))
```

The BVE runner supplies `F = 0`; the model accepts an optional spectral
forcing array. `zeta` is the only prognostic field. `psi` is obtained with
[`invert_laplacian`](laplacian.py), setting the undetermined `l=0` mode to
zero. Absolute vorticity is formed directly in spectral space by adding
`f_10 = 2 Omega sqrt(4 pi/3)`; it is not round-tripped through the grid.

With derivatives already divided by `R`, `a_sinth = sin(theta) a_theta/R`
and `a_lam = a_lambda/R`, the implemented Jacobian is
`(a_sinth b_lam - a_lam b_sinth)/cos(phi)^2`.
[`test_spectral_operators.py`](../../../tests/test_spectral_operators.py)
checks its sign and magnitude against solid-body advection and an independent
velocity-form reference; [`test_r5_spectral_eta.py`](../../../tests/test_r5_spectral_eta.py)
checks the spectral absolute-vorticity construction.

## 2. Spherical-harmonic conventions

### 2.1 Basis and normalization

The production transforms use **fully normalized (orthonormal) complex spherical harmonics with the
Condon–Shortley phase**:

```
Y_l^m(θ, λ) = N_l^m · P_l^m(cos θ) · e^{imλ},      N_l^m = sqrt( (2l+1)/(4π) · (l−m)!/(l+m)! )
```

- `θ` = **colatitude** (0 at north pole), `λ` = longitude, m ≥ 0 only.
- The Condon–Shortley factor `(−1)^m` is folded into the associated Legendre functions
  inside the CUDA kernels ([legendre.cu](transforms/cuda/legendre.cu),
  [sh_matrix.cu](transforms/cuda/sh_matrix.cu)) via the seed
  `P_m^m = (−1)^m (2m−1)!! (1−x²)^{m/2}`; the Python-side normalization deliberately does
  *not* apply it a second time (the "FIX" comments in
  [spherical_harmonics.py](transforms/spherical_harmonics.py) record a
  past double-phase bug).
- Orthonormality means `∫ Y_l^m (Y_{l'}^{m'})* dΩ = δ_ll' δ_mm'`, so a constant field maps to
  `a_00 = sqrt(4π) · c`. **[measured]**: the geodesic transform reproduces `a_00 = sqrt(4π)`
  and `a_10 = sqrt(4π/3)` for `f = 1` and `f = sin φ` to 15 digits (resolution 3, l_max 5).
- Degree recursion in both kernels is the standard
  `P_l^m = [(2l−1)x P_{l−1}^m − (l+m−1)P_{l−2}^m]/(l−m)`; normalization uses `lgamma` to
  avoid factorial overflow. Note the *normalization is applied after the unnormalized
  recursion*, so very high degrees (l ≳ 150) would overflow before normalizing; irrelevant
  at the documented L=21–63 examples but a known scaling limit.

### 2.2 Spectral coefficient layout

Two layouts coexist, with fixed mappings between them:

1. **Dense grid layout** (public API): complex array `a[l, m]` of shape
   `(l_max+1, l_max+1)`; only entries with `0 <= m <= l` represent modes. Validated SWE/PE and
   custom states require zero outside that triangle; older BVE producers may
   leave ignored entries there.
2. **Flat basis layout** (internal to `PointSetSphericalHarmonics`): column index
   `k = l(l+1)/2 + m` of the basis matrix `Y ∈ ℂ^{n_points × n_basis}`,
   `n_basis = (l_max+1)(l_max+2)/2`, ordered `(0,0), (1,0), (1,1), (2,0), …`.

Negative-m coefficients are never stored; real fields are assumed, so
`a_{l,−m} = (−1)^m a_{l,m}*` is implicit in the synthesis formula:

```
f(x_i) = Σ_l a_{l,0} Y_{l,0}(x_i) + 2 Re Σ_{l, m>0} a_{l,m} Y_{l,m}(x_i)
```

implemented in `inv_transform` as `2·Re(Y a) − Re(Y a|_{m=0})`. Consequence: **the
imaginary part of m = 0 coefficients is silently discarded**, and complex fields cannot be
represented.

---


## 3. Grids and transforms

[`spherical_backend.py`](spherical_backend.py) pairs geometry and transform
and owns the cached nonlinear `ProductSpace`. Production choices are:

| Backend | Geometry and analysis | Numerical assumption |
| --- | --- | --- |
| Gauss lat–lon | Gauss–Legendre nodes in `cos(theta)`, periodic uniform longitude; dense SH analysis/synthesis | State analysis exact to floating point for band-limited input when `nlat >= L+1`, `nlon >= 2L+1` |
| Geodesic | Subdivided icosahedron, `10*4^resolution+2` points, rotated 0.01 rad about x to avoid exact poles; normalized Voronoi solid-angle weights | Approximate, orientation-dependent quadrature; a fine product co-grid reduces error without making it exact |

For point sets, analysis is `a = Y^H (w*f)` and synthesis follows section 2.
Their composition is the identity only within the quadrature envelope.
Weights sum to `4 pi`; area integrals multiply by `R^2`.

The July 2026 measurements (MX110, CuPy 13.4, Voronoi weights) remain useful
operating examples, rather than accuracy guarantees:

| Geodesic configuration | points/basis | worst diagonal error | leakage RMS |
| --- | --- | --- | --- |
| res 4, L=15 | 18.8 | 2.2e-3 | 6e-3 |
| res 4, L=21 | 11.1 | about 2e-3 | about 1e-2 |
| res 4, L=45 | 2.4 | 7.1e-2 | 3.0e-1 |
| res 5, L=45 | 9.5 | 9.0e-4 | 1.0e-2 |

The geodesic transform warns below six points per basis function. That is
a soft under-resolution guard, not a promised error bound. The matched
random-field comparison at L=21 gives relative L2 residuals 1.0355e-2
(geodesic res 4) and 6.8418e-15 (Gauss 32x64); configuration and measurements
are in [validation](../../../docs/VALIDATION.md).

The optional optimized-weight cache checks resolution, L, point count and
basis identity but does not hash point coordinates. Do not reuse it after
changing grid orientation or subdivision. The legacy
[`LatLonSphericalHarmonics`](transforms/spherical_harmonics.py), using
[`simpson_2d`](grids/integration.py) on an open longitude interval, is a
comparison implementation, **not the production Gauss transform**. Its
missing periodic panel and even-count Simpson fallback remain limitations;
it is not an independent accuracy reference. The July constant-field probe
on 33x65 points recovered `a_00 = 3.4904` instead of `sqrt(4 pi) = 3.5449`.

## 4. Spectral derivatives

[`SpectralOperators`](operators/spectral_operators.py) uses:

```
lap(Y_lm) = -l(l+1)/R^2 Y_lm
d_lambda_coeffs(a)_lm = i m a_lm/R
sin(theta) dY_lm/dtheta = l epsilon_(l+1,m) Y_(l+1,m)
                         - (l+1) epsilon_(l,m) Y_(l-1,m)
epsilon_(l,m) = sqrt((l^2-m^2)/(4l^2-1))
```

[`laplacian.py`](laplacian.py) is the single eigenvalue/inversion authority;
the eigenvalue at zero is exactly zero. Derivative synthesis can need one
degree beyond the retained state band. PE's
[retained-truncation contract](../temporal/tendencies/RETAINED_TRUNCATION.md)
provides that buffer. Weak-form vector curl/divergence, including its
one-degree-extended analysis and geodesic accuracy envelope, is derived in
the [PE formulation](../temporal/tendencies/PRIMITIVE_EQUATIONS.md#8a-vector-curldivergence-analysis-resolved-tendency-milestone).

Supported production grids avoid exact poles. Metric divisions in vector
analysis assume that sampling contract; arbitrary point sets containing
exact poles are not a validated extension. `SpectralOperators` also mutates
its cached zonal-derivative buffer on access; concurrent sharing of a model
instance is not an established thread-safety contract.

## 5. Products, timestepping and limits

The backend synthesizes products on its `ProductSpace`, analyzes once and
returns coefficients. For quadratic degree-L inputs projected through L,
Gauss `fine` requires `nlon >= 3L+1` and `nlat >= ceil((3L+1)/2)`;
geodesic `fine` uses resolution r+1 with approximate quadrature. The
[quadrature derivation and RH4 measurements](../../../docs/validation/latlon_backend_review.md)
and [factorized product measurements](../../../docs/validation/r3_characterization.md)
record what was measured. PE cubic terms and non-polynomial forcing have
separate bounds in [retained truncation](../temporal/tendencies/RETAINED_TRUNCATION.md).

BVE/SWE normally cut analyzed nonlinear tendencies at `floor(2L/3)` while
storing through L. This does not give a general Galerkin conservation
guarantee. Conservation is measured for each configuration. On rotating BVE
the invariant is **absolute** enstrophy `1/2 integral (zeta+f)^2 dA`;
relative enstrophy alone also measures physical exchange with planetary
vorticity.

BVE and SWE use explicit RK4 with a ceiling recomputed from every accepted
state (SWE includes gravity-wave speed). Explicit viscosity has no separate
stability controller. The [PE runner](../run/pe/GUIDE.md) uses fixed steps;
the [semi-implicit stepper](../temporal/SEMI_IMPLICIT.md) is also implemented
and used by [Held–Suarez](../../../docs/held_suarez/README.md). Column
identities do not prove whole-model energy conservation, and passing
Held–Suarez is not general planetary-climate validation.

Solvers use CuPy/CUDA float64/complex128. Host synthesis, saved-run reading,
terrain-file reading and many interface tests run on CPU. Dense transforms
cost O(N L^2) and about `16*N*(L+1)*(L+2)/2` bytes per complex basis matrix;
unnormalized Legendre recurrences also limit very high degree. Windows
display GPUs can hit TDR timeouts. See [current limitations](../../../docs/KNOWN_LIMITATIONS.md)
and [architecture](../../../docs/ARCHITECTURE.md#cpu-and-gpu-requirements).
