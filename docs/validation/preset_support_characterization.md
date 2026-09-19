# Preset support characterization at the product-truncation boundaries

**Sprint 1 record, 2026-09-18.** Branch `feat/support-contracts`, starting
revision `fd0ddff` (main). Measurements taken on the GeForce MX110 with
`venv/Scripts/python.exe` (Python 3.12, CuPy 13.4) on both backends: the
Gauss lat-lon backend (`nlat=32, nlon=64`) and the geodesic backend
(resolution 3). Probe script: session scratchpad `probe_support.py`; the
same observations are pinned as tests in
[`tests/test_support_characterization.py`](../../tests/test_support_characterization.py).

Every section separates **source facts** (read from the code), **predictions**
(written before the probes ran), **measurements**, and **adopted policy**.

## Vocabulary

| Term | Meaning | Where it lives |
| --- | --- | --- |
| stored capacity `l_max` | triangular `0 <= m <= l <= l_max` extent of every prognostic array | model/state shapes |
| product truncation cut | `cut = floor(2 * l_max / 3)`; analyzed nonlinear products keep `l, m <= cut` | `tropoi.support.product_truncation_cut` |
| upper band | degrees `cut < l <= l_max`: stored; receive no analyzed nonlinear-product contribution. Other terms may still act there per core (exact linear operators, hyperdiffusion/viscosity, prescribed topographic terms) | this record |
| retained degree | a degree `l <= cut` that receives the nonlinear-product tendency | this record |

Three distinct notions are kept apart throughout: **stored capacity**
(`l_max`), **nonlinear-product support** (the cut), and **other core-specific
terms** (linear pressure/Coriolis/hydrostatic operators, hyperdiffusion,
topography), which each core applies according to its own formulation. The
probes below used flat terrain and zero hyperdiffusion, so "only the linear
pair acts above the cut" is a statement about those configurations.

Pinned cut values: `1->0, 2->1, 3->2, 4->2, 5->3, 6->4, 10->6, 15->10,
21->14, 42->28, 63->42`.

## Source facts (revision `fd0ddff`)

| Preset | Construction | Literal indices | Minimum storage | Guard before this sprint | Terms that must be active for the advertised behavior |
| --- | --- | --- | --- | --- | --- |
| PE `thermal_wave` | `isothermal_rest_state` + one real coefficient `T[:, 2, 2] = amplitude` at every level | `(l, m) = (2, 2)` | `l_max >= 2` | config `lmax >= 2` (`_SCENARIOS_NEEDING_L2`); factory `l_max >= 2` | "smooth finite response": thermodynamic products at degree 2 (advection, adiabatic heating) plus the linear hydrostatic forcing of divergence |
| SWE `gravity_wave` | `phi[4, 2] += 1e-3 * phi0` on the rest state | `(4, 2)` | `l_max >= 4` | factory `l_max >= 4` only; config accepted `lmax = 3` | linear pressure pair `delta_dot = -lap(phi)`, `phi_dot = -phi0 * delta` (exact spectral operations) |
| SWE `williamson2` | `zeta[1, 0]`, `phi[2, 0]` (flat bottom) | `(1, 0)`, `(2, 0)` | `l_max >= 2` | none (config accepted `lmax = 1`; factory would raise `IndexError`) | steady state needs the degree-2 curl and kinetic-energy products to cancel `-lap(phi)` |
| SWE `williamson5` | same wind/free-surface pair as W2 minus the cone anomaly `phi_s'` | `(1, 0)`, `(2, 0)` + cone | `l_max >= 2` | none (config accepted `lmax = 1`; factory would raise `IndexError`) | benchmark policy unchanged; no integration in this sprint |

The six production cut formulas were: three `SpectralOperators` sites
(`_truncate_product`, `advection_pseudospectral`, `jacobian_pseudospectral`),
the PE `product_truncation_cut` definition (also used by the terrain
band-limit), the SWE cached `_trunc_cut`, and the BVE diagnostic band
`high_l_enstrophy_frac`. `PrimitiveEquationsState.nlev` uses `(rows-1)//3`,
which is a layout formula, not a truncation, and is untouched.

## Predictions (recorded before the probes)

1. PE `thermal_wave`, `lmax = 2` (`cut = 1`): at rest all products vanish, so
   the temperature and `ln p_s` tendencies are exactly zero; the hydrostatic
   term forces degree-2 divergence. Once divergence exists, the degree-2
   temperature products are zeroed by the cut, so `T(2,2)` never changes:
   the "smooth response" is a frozen temperature mode driving divergence.
   At `lmax = 3` (`cut = 2`) `T(2,2)` evolves.
2. SWE `gravity_wave`, `lmax = 4, 5`: the `(4, 2)` mode lies above the cut, so
   its divergence/geopotential pair is *exactly* the linear pair (bitwise
   equal to the diagonal operators applied to the state), and the frequency
   matches the dispersion relation. The lower band `l <= cut` still receives
   nonlinear products of the upper modes, so the system is not linear.
3. Williamson 2, flat, `lmax = 2` (`cut = 1`): the degree-2 curl and
   kinetic-energy products are discarded, leaving the residual
   `delta_dot(2,0) = -lap_2 * phi(2,0)` exactly; `lmax = 3` retains degree 2
   and is steady to round-off.

## Measurements

Exact-zero and bitwise results are structural and held identically on both
backends. Quadrature-limited values are listed per backend.

### PE `thermal_wave` (3 uniform sigma levels, `T0 = 260 K`, `p_s = 101325 Pa`)

| `lmax` (`cut`) | amplitude | tendency at t=0 | after two 1 s stage-validated RK4 steps |
| --- | --- | --- | --- |
| 2 (1) | 1 K | zeta, T, ln p_s rows exactly 0; delta lower band exactly 0; `max|delta_dot(2,2)| = 7.611e-11 s^-2` (both backends) | `T(2,2)` change exactly 0; all upper-band T and ln p_s exactly 0; `delta(2,2)` change 1.522e-10; lower-band T change 5.5e-11 (monopole/degree-1 free) |
| 3 (2) | 1 K | delta forcing 7.611e-11 at (2,2), now in the lower band | `T(2,2)` change 9.218e-9 K (both backends) |
| 2 (1) | 0 K | every row exactly 0 | state bitwise unchanged |

Prediction 1 confirmed: at `lmax = 2` the preset's temperature perturbation
is frozen for all time; only divergence responds. Zero amplitude is exact
rest at every capacity.

### SWE `gravity_wave` (non-rotating, flat, `nu4 = 0`, mean depth 1000 m)

| `lmax` (`cut`) | `delta_dot(4,2) == -lap_4 phi(4,2)` | upper pair after one 300 s step | frequency / exact | lower-band tendency (full) | lower band with upper inputs removed |
| --- | --- | --- | --- | --- | --- |
| 4 (2) Gauss | bitwise | bitwise linear (delta and phi) | 1.000 | 1.894e-9 | 1.229e-13 |
| 5 (3) Gauss | bitwise | bitwise linear | 1.000 | 2.794e-10 | 2.835e-14 |
| 4 (2) geodesic | bitwise | bitwise linear | 1.000 | 1.904e-9 | 1.232e-13 |
| 5 (3) geodesic | bitwise | bitwise linear | 1.000 | 2.709e-10 | 2.862e-14 |

Prediction 2 confirmed for this configuration (non-rotating, flat, `nu4 = 0`;
with rotation the Coriolis product is truncated like any other product, and
with hyperdiffusion the exact diagonal damping also acts above the cut). The
lower-band tendency is four orders of magnitude
larger with the upper inputs present than without them: the `(4, 2)` mode
feeds the retained degrees through the products. Only the *upper* pair is
a linear regime; the preset must not be described as a linear system.

### Williamson 2 (flat bottom, `g h0 = 2.94e4`, `Omega = 7.29212e-5`)

| `lmax` (`cut`) | backend | zeta, phi tendencies | `max|delta_dot|` | `delta_dot(2,0)` | pressure-only term `-lap_2 phi(2,0)` |
| --- | --- | --- | --- | --- | --- |
| 2 (1) | Gauss | exactly 0 | 2.919e-9 | -2.919e-9 (bitwise = pressure term) | -2.919e-9 |
| 2 (1) | geodesic | exactly 0 | 2.919e-9 | -2.919e-9 (bitwise = pressure term) | -2.919e-9 |
| 3 (2) | Gauss | exactly 0 | 2.068e-24 | -2.068e-24 | -2.919e-9 |
| 3 (2) | geodesic | exactly 0 | 5.406e-24 | 0 | -2.919e-9 |
| 4 (2) | both | exactly 0 | as `lmax = 3` | | |
| 2, cut disabled (characterization only) | both | exactly 0 | 2.1e-24 / 5.4e-24 | | |
| 15 (10) | geodesic | exactly 0 | 9.09e-12 (existing envelope) | | |

Prediction 3 confirmed. At `lmax = 2` the entire residual is the degree-2
pressure term whose balancing products were cut; restoring the degree-2
products (either `lmax >= 3` or, for characterization only, disabling the
cut) returns the steady state to round-off on both backends.

### Williamson 5

Source and CPU facts only, no integration: the config layer accepted
`scenario=williamson5, lmax=1` (resolving the cone topography) and the
factory would fail with `IndexError` writing the `(2, 0)` coefficient. The
same capacity fact holds for the low-level `ShallowWaterState` at
`l_max = 1` (pinned test). Separately, `Topography.williamson5_cone` has its
own representability gate (relative projection residual `<= 0.25`), which
fires before any initial-condition guard at tiny capacities (measured
residual 0.868 at `l_max = 3` on the 32x64 Gauss grid); that gate is
benchmark policy and is not changed here, so the factory-boundary test for
`williamson5` uses the documented terrain-less defensive path.

### Mechanical extraction: before/after comparison

All 20 probe records (tendency and RK4-step sha256 hashes plus every
measurement above) are identical before and after replacing the six
formulas with `tropoi.support.product_truncation_cut`.

`tests/test_r3_fine_product.py::test_prediction_p1_5day_energy_drift`
(geodesic res 4, `l_max = 21`, 196 steps): drift
`-4.45545966660124766e-04` before and after (17 digits), final-state
sha256 `ba518a80…b71e0` identical, and identical to the historical R-3
acceptance hash. The historical `[-5.4e-4, -3.6e-4]` assertion is
preserved.

## Adopted policy (phase 3)

| Preset | Contract | Evidence |
| --- | --- | --- |
| PE `thermal_wave` | storage `lmax >= 2` kept; for nonzero amplitude, `product_truncation_cut(lmax) >= 2` (so `lmax >= 3`); zero amplitude keeps the storage boundary | frozen-temperature mechanism measured at `lmax = 2` on both backends |
| SWE `gravity_wave` | `lmax >= 4` at both config and factory; supported at `lmax = 4, 5` | upper pair bitwise linear; frequency exact; lower band documented as nonlinear |
| SWE `williamson2` | capacity `lmax >= 2`; retained degree `product_truncation_cut(lmax) >= 2` (so `lmax >= 3`) | residual at `lmax = 2` is exactly the uncancelled pressure term |
| SWE `williamson5` | initial-state storage/capacity `lmax >= 2` only. This does not supersede the cone/topography representability gate (which rejects the cone at small `lmax`, e.g. residual 0.868 at `lmax = 3` on the 32x64 Gauss grid) and makes no claim that W5 is a useful benchmark at such resolutions; terrain and benchmark policy unchanged | source fact; no integration |

Resolved config dicts and run ids compared between `main` (`fd0ddff`) and
the branch for eight representative configurations (BVE default; SWE
default, gravity_wave, williamson2 `lmax=3`, canonical williamson5 `lmax=42`;
PE default, thermal_wave `lmax=3`, thermal_wave `lmax=2` with zero
amplitude): all identical, e.g. SWE default `0e6cfc38`, gravity_wave
`11055b20`, PE default `fa2db863`.

Direct low-level state construction (`ShallowWaterState`,
`isothermal_rest_state`) remains available for characterization at every
capacity; only the public factories and the CPU configuration layer enforce
the contract. Representative resolved configs and run-id hashes at the
supported resolutions are unchanged (the guards add no config keys).
