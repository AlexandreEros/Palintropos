# Prognostic states

Each model core carries its whole prognostic state as **one** complex
CuPy array of spectral coefficients. These classes name the parts of that
array. They hold no grid values and do no arithmetic, so initial
conditions, integrators and archives can share a state without importing
a model's tendency code.

Each module was split verbatim from its tendency module, which still
re-exports every name, so `tropoi.temporal.tendencies.<core>` imports keep
working.

| Core | Class | Array shape | Rows |
| --- | --- | --- | --- |
| BVE | `BarotropicState` (`barotropic.py`) | `(L+1, L+1)` | Relative vorticity `ζ` |
| SWE | `ShallowWaterState` (`shallow_water.py`) | `(3, L+1, L+1)` | `ζ`, divergence `δ`, perturbation thickness geopotential `φ` (indices `ZETA, DELTA, PHI`) |
| Dry PE | `PrimitiveEquationsState` (`primitive_equations.py`) | `(3K+1, L+1, L+1)` | `ζ_1..ζ_K`, `δ_1..δ_K`, `T_1..T_K` top to bottom, then `ln p_s` |

Every 2-D slice uses the shared layout: degree on axis 0, order `m >= 0` on
axis 1, real fields implied, `m > l` entries as padding
([CONVENTIONS.md](../CONVENTIONS.md#2-spherical-harmonic-conventions)). A
constant field `c` is the single coefficient `a_00 = c · sqrt(4π)`.

## Why one packed array

Runge–Kutta stages and semi-implicit solves are then plain array
expressions (`temporal.integration.rk4_step_array`), a saved snapshot is
one `np.save`, and a run's frames stack into `(n_times, rows, L+1, L+1)`.
The property accessors (`zeta`, `delta`, `temperature`, `ln_ps`, `phi`)
return views, so writing through them changes the state in place.
[`modes.py`](../modes.py) gives read-only, named host views of the same
storage for saved runs.

## What each state guarantees

The classes check only array shape (the PE class checks it in
`__post_init__`). Physical rules are checked by each model's
`validate_state`, after every accepted step:

| Core | Hard checks (`validate_state` in [tendencies](../../temporal/tendencies/README.md)) |
| --- | --- |
| BVE | None on the state itself; conservation is monitored by [diagnostics](../../representation/diagnostics/README.md) |
| SWE | Finite coefficients; zero monopoles of `ζ`, `δ` and `φ`; positive total geopotential `Φ0 + φ` on every sampling the model uses |
| Dry PE | Shape against the model's `L` and `K`; finite coefficients; zero per-level `ζ` and `δ` monopoles; positive temperature on every sampling; finite `p_s = exp(ln p_s)` |

Notes on meaning:

- **SWE `φ` is a perturbation.** The mean thickness geopotential `Φ0` is a
  model constant, so `φ_00` stays zero and total mass is conserved exactly
  in spectral space. Over terrain, `φ` is the thickness, not the free
  surface ([SWE formulation](../../temporal/tendencies/SHALLOW_WATER.md)).
- **PE monopoles of `T` and `ln p_s` evolve.** Global-mean temperature
  changes through energy conversion, and the mean of `ln p_s` is not the
  mean of `p_s`, so the PE core does not conserve mass exactly; the drift
  is monitored ([PE formulation](../../temporal/tendencies/PRIMITIVE_EQUATIONS.md)).
- `isothermal_rest_state(l_max, nlev, temperature=, surface_pressure=)`
  builds the PE exact-rest state, whose tendency is exactly zero. The
  [PE presets](../initialization/README.md) start from it.
- `BarotropicState.tendency` is an unused optional field kept for
  compatibility.
