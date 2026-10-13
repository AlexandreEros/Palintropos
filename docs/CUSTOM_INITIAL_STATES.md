# Custom planets and initial states (BVE and SWE)

You can define a planet and a barotropic-vorticity (BVE) or shallow-water
(SWE) initial state, save them, reload them in a fresh Python process, and
run them through a documented interface, all without editing Palintropos
source code. The primitive-equation (PE) core is not covered.

```python
from tropoi.run.custom import (DiscretizationSpec, InitialState, PlanetSpec,
                               load_initial_state, run_initial_state)

state = InitialState("bve", zeta_lm, PlanetSpec.ideal_sphere(3.4e6, 88642.0),
                     DiscretizationSpec.gauss_latlon(21, 32, 64), label="my run")
state.save("my_state")                          # versioned JSON + .npy arrays
state = load_initial_state("my_state")          # any process; never imports CuPy
run_dir = run_initial_state(state, duration_days=2.0, n_snapshots=9)
```

`run_dir` is an ordinary run capsule. `open_simulation`, `tropoi inspect` and
`tropoi plot` read it like any preset run
([SAVED_RUNS.md](SAVED_RUNS.md)).

This page covers states whose spectral coefficients are already resolved.
Building a state from physical fields or callables is a separate,
explicit preparation step.

## Three kinds of information, kept apart

| Kind | Object | Saved? |
| --- | --- | --- |
| Prescribed physics | `PlanetSpec`; SWE: `FluidLayerSpec`, `TopographySpec` and the resolved terrain coefficients | yes |
| Discretization | `DiscretizationSpec` (truncation `lmax`, grid family and size, product sampling) | yes, as the grid the state was resolved for |
| Runtime CUDA resources | grids, transforms, product spaces, operators, models | **never**; rebuilt at run time |

Nothing executable is ever stored: no pickles and no Python callables.
Loading a file never runs code from it.

### `PlanetSpec`: radius and geometry convention

The two ways the repository builds a planet stay distinct, and the saved
document records which one was used:

| `construction` | Constructor | `reference_radius_m` | Dynamical radius used by the cores |
| --- | --- | --- | --- |
| `"ideal_sphere"` | `PlanetSpec.ideal_sphere(radius_m, sidereal_day_s)` | the sphere's radius | exactly `reference_radius_m` |
| `"oblate_volumetric_mean"` | `PlanetSpec.oblate(equatorial_radius_m, sidereal_day_s)`, `PlanetSpec.earth_like(day_hours, radius_earth_units)` | the equatorial radius | derived volumetric-mean radius of the first-order rotational oblate spheroid; about 0.06 % below equatorial for Earth-like rotation |

`ideal_sphere` is what benchmarks use: Williamson case 5 keeps
`a = 6.37122e6 m` exactly. `earth_like` reproduces
`PlanetaryParameters.from_earth_like` bit for bit, which is how every
non-benchmark preset builds its planet. Pass `sidereal_day_s=None` (or
`math.inf`) for a non-rotating planet; the document then stores `null`. The
saved block also records the derived dynamical radius, angular velocity and
oblateness. Loading recomputes them and refuses the file if they differ.

### `FluidLayerSpec` (SWE): explicit gravity and mean thickness

`FluidLayerSpec(gravity_m_s2, mean_depth_m)` takes gravity explicitly. It is
never derived from the planet's mass and radius, as in the SWE core itself.

The SWE prognostic `phi` is the **perturbation thickness geopotential**:

```
total fluid thickness   h = H + phi / g          (H = mean_depth_m, g = gravity_m_s2)
thickness geopotential  g h = Phi0 + phi,        Phi0 = g H
free-surface height     h + h_s                  (h_s = bottom elevation)
```

`phi` is neither the full geopotential nor the free-surface elevation. Its
global mean is zero, because the mean thickness lives in `H`.

### `TopographySpec` (SWE): the existing terrain presets

`TopographySpec.flat()`,
`TopographySpec.mountain(height_m, lat_deg, lon_deg, width_deg)` (the
`--topography mountain` Gaussian), or `TopographySpec.williamson5_cone()`
(benchmark terrain, for reproducing or perturbing case 5). Custom terrain
shapes are not supported.

Terrain is projected onto the model grid, so its coefficients depend on the
backend. An SWE initial state therefore stores the **resolved** elevation
coefficients alongside the specification, and a run uses those coefficients
exactly instead of projecting the terrain again. This matters because an SWE
state over terrain contains `-phi_s'`, the resting thickness compensation.
The state is only meaningful with exactly the terrain it was built on.

### `DiscretizationSpec`

`DiscretizationSpec.gauss_latlon(lmax, nlat, nlon)` (the reference grid) or
`DiscretizationSpec.geodesic(lmax, resolution)` (experimental), with
`product_quadrature="fine"` (default; always `"fine"` for the SWE) or
`"coarse"` (BVE only). A run may override the grid. Its `lmax` must equal the
state's: stored coefficients are never truncated or padded implicitly, and
the run keeps the existing 2/3 product truncation on top of the full stored
support.

## `InitialState`

| Solver | `coeffs` (complex128) | Fields and units |
| --- | --- | --- |
| BVE | `(L+1, L+1)` | `zeta`: relative vorticity, s⁻¹ |
| SWE | `(3, L+1, L+1)` | `zeta`, `delta` (s⁻¹); `phi`: perturbation thickness geopotential, m² s⁻² |

The coefficient conventions are the repository's
([MATHEMATICAL_MODEL.md](MATHEMATICAL_MODEL.md)): complex orthonormal
spherical harmonics stored for `m ≥ 0`, axis 0 = degree `l`, axis 1 = order
`m`. Real fields are implied: `Im a_l0 = 0`, and `m > l` entries are exactly
zero. A constant field `c` is the `(0, 0)` coefficient `c·√(4π)`.

Construction and loading always run the **coefficient-level checks** on the
CPU, using the same `FieldSpec` rules as saved runs. A failed check raises
`InitialStateError`; nothing is repaired:

- every value is finite, the padding is zero, and `a_l0` is real;
- SWE `zeta`, `delta` and `phi` have zero global mean, to the cores'
  relative tolerance (1e-10 of the field norm);
- the BVE `zeta` monopole is the conserved circulation and is not
  constrained here, as in the core;
- SWE: a fluid layer is required; non-flat terrain requires its resolved
  coefficients; flat terrain must not have any.

**Physical checks** need the model's transforms and run on the GPU when the
state is executed. For the SWE, `run_swe` checks that `H + phi/g` is strictly
positive on both the state and the nonlinear-product samplings before it
integrates anything. A state that fails this check fails the run, and the
capsule is marked `failed` with the reason.

**Valid is not balanced.** A state that passes every check can still be far
from balance and radiate gravity waves. Nothing balances a state
automatically.

## File format (version 1)

A directory:

| File | Content |
| --- | --- |
| `initial_state.json` | `format: "tropoi.initial_state"`, `version: 1`, `solver`, `label`, `digest`, and the blocks `planet`, `discretization`, `fluid`, `topography`, `state` (file, dtype, shape, SHA-256, layout, normalization and the same `state_schema` block run manifests carry), plus a descriptive `provenance`, `created_utc` and `software` |
| `state_coeffs.npy` | the packed frame, little-endian complex128 |
| `topography_elevation_lm.npy` | SWE over non-flat terrain only: resolved elevation coefficients, metres |

The JSON is strict: non-finite numbers are refused and unknown keys are
rejected. Arrays are read with `allow_pickle=False`. Loading verifies:

- the format and version, and each specification block, including its
  convention statements;
- each array's dtype, shape and SHA-256;
- the `state_schema` block, by re-deriving it;
- the coefficient conventions;
- the identity digest.

A file that fails any of these checks is refused. Coefficients round-trip
bit for bit, signed zeros included.

**Saving is never destructive.** A state is written to a hidden staging
directory beside its destination and then renamed into place. With
`overwrite=True` (which only replaces a directory that is itself a saved
state), the previous state is first renamed aside, the new one is renamed
into place, and only then is the previous one deleted. If installing the new
state fails, the previous one is renamed back. If even that restore fails,
nothing is deleted and the error names where both copies are. If only the
final deletion fails, a warning says where the old copy was left.

## Identity

`InitialState.digest` is `sha256:<hex>`, computed over a canonical
description of the solver, the planet, the fluid layer, the topography
specification, and the exact bytes of the state and terrain coefficients.
It excludes the label, the provenance and the discretization record. Two
states therefore share a digest exactly when their prescribed physics and
coefficients are identical, whatever their names. Changing one coefficient
by one ulp changes the digest.

A custom run records the digest as `initial_state_digest` in its run
configuration, so the digest enters the run id's scientific hash. The label
is recorded as `initial_state_label`, which the hash excludes. Custom run
ids always carry the hash, whatever the snapshot mode. Preset run
configurations contain none of these keys, so their run ids are unchanged.

## Running a state

```python
run_dir = run_initial_state(
    state_or_path, duration_days=1.0,
    n_snapshots=5,                 # or snapshot_interval_s=...
    discretization=None,           # default: the state's own; same lmax required
    viscosity=None,                # BVE only (m^2/s)
    plots=None,                    # "diagnostics" / "snapshots" / "summary" / "all"
    out="runs", experiment=None, overwrite=False)
```

From the shell, the same run is a flag on the ordinary commands:

```bash
tropoi run bve --initial-state states/rh4_small --days 2 --n-snapshots 9
tropoi run swe --initial-state states/mars_jet --backend gauss-latlon --nlat 48 --nlon 96
```

With `--initial-state`:

- **Accepted:** the schedule options (`--days`, `--n-snapshots`,
  `--snapshot-interval-seconds`), `--plot`/`--no-plots`, `--out`,
  `--experiment` and `--overwrite`. The grid options (`--backend`,
  `--resolution`, `--nlat`, `--nlon`, and BVE `--product-quadrature`)
  override the state's recorded grid. `--l-max` is accepted but must equal
  the state's truncation. BVE `--viscosity` is accepted.
- **Refused:** options the state already prescribes. For the BVE these are
  `--preset`, `--scenario`, `--day-hours` and `--radius-earth-units`. The
  SWE refuses the same planet options plus `--gravity`, `--mean-depth`,
  `--topography` and `--mountain-*`.
- **Errors:** a state of the other solver, or a file that fails
  verification, is an argument error. No run directory is created and CUDA
  is not touched.

The defaults are those of the ordinary commands: 1 day and 5 stored states.

Every option is validated on the CPU before CUDA is touched
(`CustomRunConfig.resolve`). The run then goes through the same lifecycle,
runners and cores as `tropoi run bve|swe`. A custom run adds the following:

- **`run_config`**:
  - `solver`, and `scenario: "custom"`;
  - `planet`: the `PlanetSpec` block;
  - `initial_state_format`, `initial_state_digest` and
    `initial_state_label`;
  - SWE only: `gravity`, `mean_depth_m`, the topography keys,
    `topography_parameters` and `topography_digest`.

  `radius_earth_units` is absent, because a custom planet is not measured in
  Earth radii.
- **Manifest blocks**: `state_schema.geometry` records the construction, the
  reference radius and the dynamical radius. `notes.initial_state` names the
  state.
- **`initial_state/`**: a byte-exact copy of the state, written before
  integration and checked against its digest. It makes the capsule
  self-contained: `Snapshot.plot` rebuilds the planet from the `planet`
  block and the terrain from this copy.

### Preset states and equivalence

`initial_state_from_preset(solver, settings, preset=None)` (CUDA) builds a
preset run's state with the run's own builders and initial-condition
factories, and returns it as an `InitialState`:

```python
from tropoi.run.custom import initial_state_from_preset, run_initial_state

w5 = initial_state_from_preset("swe", {"scenario": "williamson5", "grid": "latlon",
                                       "nlat": 64, "nlon": 128, "lmax": 42})
w5.planet.construction        # 'ideal_sphere': a = 6.37122e6 m exactly
w5.save("williamson5_t42")
```

A saved preset state, reloaded and run, reproduces the preset run's
trajectory. `tests/test_custom_runs.py` checks this for two-vortex BVE,
Williamson 5 and Williamson-2 mountain flow. Exporting a state is also how to
pin `random_low_l`, which does not record its RNG seed: the saved file *is*
the realized state.

## CPU and GPU

| Needs only the CPU (no CuPy import, no CUDA) | Needs CUDA |
| --- | --- |
| building and validating specifications and `InitialState`s from coefficients; `save`, `load_initial_state`, `summary`, `digest`; `CustomRunConfig.resolve`; reading custom capsules (`open_simulation`, `tropoi inspect`) | `run_initial_state`, `initial_state_from_preset`, physical-space plotting |

## Limitations

- BVE and SWE only; the PE core is unchanged and not supported.
- Terrain is limited to the existing presets (`flat`, `mountain`,
  `williamson5_cone`). Custom terrain generation is out of scope.
- The SWE runs remain inviscid, and the BVE viscosity remains the optional
  Laplacian term without a stability control of its own. Equations,
  integration, truncation and normalization are the presets'.
- Loading refuses a file whose recorded derived planet values differ from
  what this installation computes. In particular, an `oblate_volumetric_mean`
  planet whose oblateness model gives a last-digit different radius on
  another platform will not load, rather than be silently reinterpreted.
