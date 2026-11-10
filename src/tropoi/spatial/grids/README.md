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

## Production geometries

| Geometry | Points | Weights | Exactness | CFL length |
| --- | --- | --- | --- | --- |
| `GeodesicGridGeometry` (`geodesic_grid.py`) | Subdivided icosahedron, `10·4^r + 2` vertices, rotated 0.01 rad about x so no vertex sits on a pole | Each spherical triangle (L'Huilier excess) gives a third of its area to each vertex; the code calls these "Voronoi" areas. They sum to exactly `4πR²` | Approximate and orientation-dependent | Shortest edge (chord length) |
| `GaussLatLonGridGeometry` (`latlon_grid.py`) | Gauss–Legendre latitudes in `sin(lat)` × uniform periodic longitudes; no pole rows | Gauss weights × `2π/nlon` | Exact for band-limited analysis when `nlat >= L+1`, `nlon >= 2L+1`; for quadratic products at the 3/2 rule | `R · min Δcolatitude` |

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
