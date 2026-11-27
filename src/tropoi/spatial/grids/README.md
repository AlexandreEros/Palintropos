# Grid geometries

A grid geometry says where a model samples the sphere and supplies the
metric data that sampling needs: per-point latitude and longitude,
quadrature weights, and a length scale for the timestep ceiling. It does
not hold state. [Transforms](../transforms/README.md) analyze and
synthesize on these points, and
[`spherical_backend.py`](../spherical_backend.py) pairs each production
geometry with its transform and its product sampling.

## Contract

`GridGeometry` (`grid_base.py`) is the interface the rest of the code
relies on:

| Member | Meaning |
| --- | --- |
| `point_latitudes`, `point_longitudes` | One entry per sample, in the flat order fields are stored |
| `latitudes`, `longitudes`, `colatitudes` | Native axes (1-D for structured grids, per point for geodesic) |
| `n_points`, `grid_shape` | Sample count; `(nlat, nlon)` for structured grids, `None` otherwise |
| `cfl_length_scale` | Advective length for the timestep ceiling. `None` means "none defined", and callers then use an explicit fallback rather than guess |

## Canonical quadrature (CPU-only)

[`quadrature.py`](quadrature.py) is NumPy-only and never imports CuPy. It is
the single construction of the Gauss–Legendre nodes and weights.
`GaussLatLonGridGeometry`, the airless runner's `gauss` columns and the
diagnostics all take theirs from it, so they agree bit for bit in
coordinates, ordering (north to south, longitudes from 0, lat-major) and
physical weights `W_ij = R² (2π/N_lon) w_i`.

`ColumnSet` distinguishes a **global** discretization, whose area weights
tile the sphere and are checked on construction, from a **diagnostic**
point collection, which has no weights. Vertical column meshes live
alongside in [`../vertical.py`](../vertical.py).

## Production geometries

| Geometry | Points | Weights | Exactness | CFL length |
| --- | --- | --- | --- | --- |
| `GeodesicGridGeometry` (`geodesic_grid.py`) | **Deprecated** (see below). Subdivided icosahedron, `10·4^r + 2` vertices, rotated 0.01 rad about x so no vertex sits on a pole | Each spherical triangle (L'Huilier excess) gives a third of its area to each vertex; the code calls these "Voronoi" areas. They sum to exactly `4πR²` | Approximate and orientation-dependent | Shortest edge (chord length) |
| `GaussLatLonGridGeometry` (`latlon_grid.py`) | Gauss–Legendre latitudes in `sin(lat)` × uniform periodic longitudes; no pole rows | Gauss weights × `2π/nlon` | Exact for band-limited analysis when `nlat >= L+1`, `nlon >= 2L+1`; for quadratic products at the 3/2 rule | `R · min Δcolatitude` |

`GaussLatLonGridGeometry` is **the default** for every core (BVE, SWE, PE)
and for `Planet.generate`.

`GeodesicGridGeometry` is **deprecated** and raises a `DeprecationWarning`
when `Planet.generate` builds one. Its sampling is unstructured, so it
admits no separable transform: every transform on it is a dense
basis-matrix product, with no faster path available even in principle. Its
quadrature is approximate and orientation-dependent, and its nonlinear
products use overresolved rather than exact quadrature
(`docs/validation/r3_characterization.md`).

It is **retained, never implicit**. It is reachable only by asking for it
(`--backend geodesic`, `grid_type="geodesic"`), and it is kept for one
concrete purpose: the resolution-3 product-quadrature characterization
(`tests/audit_r3_product.py` with `docs/validation/r3_characterization.md`)
and the exact mass-conservation check on that backend
(`tests/test_swe_runner.py`). No preset selects it any more. Capsules
recorded on it are still readable — a saved configuration predating the
`grid` key is read as geodesic (`temporal.models.LEGACY_GRID_TYPE`) — but a
run that must reproduce bit-for-bit is reproduced from the commit its
manifest records, not by keeping obsolete defaults alive.

## Default sizing

When a caller names no `nlat`/`nlon`, they come from the truncation:
`truncation.gauss_grid_for_truncation(l_max)` applies the **3/2 quadratic
rule** with an even latitude count and `nlon = 2·nlat` —

    nlat = ceil((3·l_max + 1) / 2) rounded up to even,   nlon = 2·nlat

so `l_max = 8 → 14×28`, `10 → 16×32`, `21 → 32×64`, `42 → 64×128`. This
reproduces the resolutions already in production use. By construction it
satisfies analysis exactness (`nlat ≥ l_max+1`, `nlon ≥ 2·l_max+1`), the
3/2 rule (`nlon ≥ 3·l_max+1`) and separable applicability
(`2·l_max < nlon`) at every truncation, so the default grid never forces
the default evaluator onto its dense fallback.

It is a **sizing policy, not a correctness contract**: dimensions given
explicitly always win, and a configuration below it is not thereby wrong —
it forfeits state-grid quadratic exactness and leans on the finer product
grid instead, which is what the Held–Suarez experiment does at
`l_max = 43` on 64×128.

`latlon_grid.py` also defines `GaussLatLonSphericalHarmonics`, the
production Gauss transform: the shared GPU matrix transform fed with these
tensor-product weights. Its module docstring derives the exactness
conditions and explains why it replaces the legacy grid. The lat–lon CFL
scale is meridional only; the docstring on `cfl_length_scale` explains why
the shrinking zonal spacing near the poles does not tighten it for a
spectral method.

## Supporting modules

| Module | Role |
| --- | --- |
| `grid.py` | Legacy `LatLonGridGeometry`: equiangular latitudes **including both poles**, odd counts forced for Simpson's rule, open longitude interval. Used for display meshes and the legacy comparison transform, not for model states |
| `integration.py` | `simpson_2d`, the legacy transform's quadrature |
| `grid_interpolation.py` | `geodesic_to_latlon_grid` / `latlon_to_geodesic_grid` with SciPy `griddata` (nearest, linear or cubic), with longitude wrapping. For display only: interpolation never feeds solver state or diagnostics |
| `cartesian_to_spherical.py` | `(x, y, z)` → `(r, λ, φ)` used to place geodesic vertices |

## Limits

- Geodesic quadrature is not exact at any truncation; the measured
  envelopes and the 6-points-per-basis warning are in
  [CONVENTIONS.md](../CONVENTIONS.md#3-grids-and-transforms). Matched
  comparisons against Gauss are in
  [validation](../../../../docs/VALIDATION.md).
- `GeodesicGridGeometry` builds a sparse inverse-distance adjacency matrix
  on construction; the model only uses it to find the shortest edge.
- The legacy `grid.py` mesh contains exact poles. It is safe for drawing
  but must not be given to code that divides by `cos φ`.
- Display interpolation (`grid_interpolation.py`, `representation.visual`)
  is not model data. Conservation and accuracy are always measured on the
  model's own sampling and weights.
