# Named initial states

Each module turns a preset name into the starting state of one model core.
The `--scenario` option of `tropoi run bve|swe|pe` selects from these
lists. User-defined BVE/SWE states are prepared separately by
[run/custom](../../run/custom/README.md), and the Held–Suarez benchmark
builds its own start in `run/held_suarez/model.py`.

| Module | Entry point | Returns | Built |
| --- | --- | --- | --- |
| `bve.py` | `make_ic(name, planet)` | Vorticity **on the grid** | Pointwise, then analyzed by the caller |
| `swe.py` | `make_swe_ic(name, model)` | `ShallowWaterState` | Directly in spectral space |
| `pe.py` | `make_pe_ic(name, model, ...)` | `PrimitiveEquationsState` | Directly in spectral space, from the exact rest state |

## BVE presets

`two_vortices`, `inverted_vortices`, `polar_vortices`,
`inverted_polar_vortices`, `equatorial_vortices` and
`inverted_equatorial_vortices` place two opposite-signed Gaussian vortices
(amplitude 5e-5 s⁻¹, width 10°) at mirrored positions. `rh4` is the
Rossby–Haurwitz wavenumber-4 vorticity
`ζ = 2ν sin φ - 30K sin φ cos⁴φ sin 4λ` with `ν = K = 7.848e-6 s⁻¹`
(Williamson et al. 1992 case 6, the [RH4 benchmark](../../../../docs/VALIDATION.md#rh4-benchmark)).
`random_low_l` draws random coefficients for `l <= 10`.

Because BVE fields are sampled on the grid and then analyzed
(`planet.sh.transform` in `run/bve/execution.py`), the starting
coefficients carry the grid's quadrature error. On geodesic grids a
Gaussian vortex is neither exactly band-limited nor exactly analyzed. `rh4`
is band-limited (degree 5), so on a Gauss grid its analysis is exact.
`random_low_l` uses CuPy's global random generator **without a seed**, so
two runs of it do not start from the same state.

## SWE presets

Every preset fixes a velocity field and a **free-surface** geopotential
anomaly, then subtracts the model's mean-removed surface geopotential, so
the same preset is well defined over terrain and bit-identical to the
historical flat-bottom state when there is none. All are monopole-free,
exactly band-limited, keep the global-mean depth equal to `mean_depth`,
and are validated before they are returned. Terrain that pierces the
fluid layer is rejected here, before integration.

| Preset | State |
| --- | --- |
| `rest` | Zero velocity, constant free surface: the exact lake at rest |
| `gravity_wave` | Fluid at rest plus a `Y_4^2` free-surface bump of `1e-3 · Φ0` |
| `williamson2` | Case 2 zonal flow, `u0 = 2πa / 12 days`; steady only over a flat bottom |
| `williamson5` | Case 5: `u0 = 20 m/s` over the canonical conical mountain; canonical constants are resolved automatically |

Derivations, and why Williamson 2 over a mountain is a mountain-flow
experiment rather than a steady test, are in the
[SWE formulation](../../temporal/tendencies/SHALLOW_WATER.md).

## PE presets

All four start from `isothermal_rest_state`
([states](../states/README.md)), so vorticity and divergence are exactly
zero and every field is exactly band-limited.

| Preset | State | Balanced? |
| --- | --- | --- |
| `isothermal_rest` | Uniform `T0`, uniform `p_s`, no wind. Its tendency is exactly zero | Yes, exactly |
| `thermal_wave` | Rest plus one real coefficient `thermal_amplitude` (K) on `(l, m) = (2, 2)` at every level | No, deliberately: shows a smooth finite response |
| `orographic_isothermal_rest` | Isothermal rest over terrain with `ln p_s = ln p_ref - Φ_s/(R_d T0)`, which cancels the pressure-gradient force analytically | Yes, analytically |
| `orographic_thermal_wave` | The orographic rest state plus the same `(2, 2)` perturbation | No: a perturbation of a balanced base |

`thermal_amplitude` is a coefficient, not a peak. A real coefficient `a` on
`(2, 2)` peaks at `2 sqrt(15/(32π)) a ≈ 0.773 a`; `thermal_wave_peak`
returns that number. The `(2, 2)` mode must sit inside the dealiased product
band, which `spatial.truncation` checks.

The orographic balance holds only for an isothermal column. A resting
non-isothermal atmosphere over terrain is generally not balanced on sigma
surfaces; the `pe.py` module docstring outlines the pressure-coordinate
construction a balanced version would need. Recipes are in the
[PE guide](../../run/pe/GUIDE.md); the
[preset support characterization](../../../../docs/validation/preset_support_characterization.md)
records which presets each truncation can represent.
