# Isca Mars M-I: gap analysis and flat-Mars pilot (2026-11-13)

Target: the intermediate-complexity Mars model **M-I** of Thomson & Vallis
(2019), *Atmosphere* 10, 803 (doi:10.3390/atmos10120803), §4.2 and Table 1.
The pilot defined here is a **development experiment, not a reproduction
of M-I**. A published comparison needs the T42 configuration, a seasonal
integration long enough to average L_s = 225°–315°, and the paper's
Figure 4b (zonal-mean zonal wind) as the reference.

Sources, read 2026-11-13: the paper (Table 1, §4.1–4.3), and Isca's public
source at commit `d1321ac3` (2026-08-13) for what the paper leaves
implicit: `exp/test_cases/grey_mars/grey_mars_test_case.py`,
`two_stream_gray_rad.F90`, `mixed_layer.F90`, `constants.F90`,
`spectral_dynamics.F90` and `spectral_init_cond.F90`. The grey-Mars test
case is not the paper's M-I run. Its own comment ("dry convection, 1.0 m")
contradicts its namelist (`convection_scheme: 'none'`, `depth: 2.0`), so
it is used only where it agrees with the paper or fills a value the paper
does not state, and each such use is marked "Isca default/test case".

## 1. M-I against the current implementation

| item | M-I | Palintropos now | status |
| --- | --- | --- | --- |
| dynamics | primitive equations, spectral, T42, 25 levels | PE spectral core | ✓ (pilot T21) |
| gas | R = 192, κ = 0.22727 (c_p = 844.8), g = 3.71, a = 3396 km, Ω = 2π/88 308 s = 7.115 × 10⁻⁵ (Table 1: 7.12), p_s = 610 Pa | configurable | ✓ |
| vertical grid | 25 levels (Table 1). σ interfaces from Isca's grey-Mars `bk` list (Isca test case) | arbitrary `SigmaGrid`; the config has no kind for an explicit list | **small gap**: add `{"kind": "interfaces"}` |
| longwave | τ = 0.2 e^(−z/H), i.e. 0.2 σ (eq. 14) | `isca_mars_longwave`, `power_law` τ₀ 0.2, exponent 1 | ✓ |
| shortwave | τ = 0.2 e^(−4z/H) = 0.2 σ⁴ (eq. 15), absorbing. Isca: `insolation · exp(−τ)` (vertical path); the reflected `albedo · sw_down_sfc` escapes unabsorbed | `GrayShortwave` takes a slant path `exp(−τ/μ₀)` and re-absorbs the reflected beam | **gap**: atmospheric absorption is 29.6 % of insolation (global and diurnal mean, equinox) against Isca's 18.1 %. Fixed here with a small analytic provider on the existing hook |
| insolation | seasons + diurnal cycle; S₀ = 589 W m⁻², e = 0.0935, obliquity 25.19°, perihelion at L_s = 251° (Isca `per` 70.85° + 180°), year 59 166 360 s, sol 88 440 s | `InstantaneousInsolation` | ✓ (never run in 3-D) |
| surface | "land depth 2 m" = 2 m × ρc_p of seawater (Isca `RHO_CP` = 4.129 × 10⁶ J m⁻³ K⁻¹) = **8.26 × 10⁶ J m⁻² K⁻¹**; albedo 0.3 (Isca test case); no topography | `SlabSurface` | ✓ |
| boundary layer, heat | Monin–Obukhov surface fluxes plus vertical turbulent diffusion; roughness 3.21 × 10⁻⁵ m; no gustiness (Isca test case) | bulk sensible flux into the lowest layer, same roughness, gustiness 0 allowed | **partial**: no vertical diffusion |
| momentum exchange | surface stress plus vertical diffusion of momentum | none; only Rayleigh drag is available | **dependency (new physics)**: bulk surface stress + implicit vertical diffusion. Pilot substitute: Held–Suarez Rayleigh drag, 1 day⁻¹ below σ = 0.7, **labelled** |
| convection | "None" (Table 1): the boundary-layer diffusion removes instability | dry convective adjustment | **dependency (follows from the row above)**. Pilot substitute: instantaneous dry adjustment, **labelled** |
| sponge | Rayleigh 0.125 day above 0.5 Pa, energy-conserving (Isca test case) | none | **gap**: a top-level `DiagonalDamping` using the existing hook. Not in the pilot |
| diffusion and filter | ∇⁴ at 0.1 day (Isca default `damping_order` 2); Robert–Asselin 0.04 | ∇⁸ at 0.1 day; RAW 0.1 / 0.53 | numerics differ. The pilot keeps the project's validated settings |
| temperature floor | none ("energetically closed") | none | ✓ |
| initial state | isothermal 264 K, at rest (Isca default); surface 285 − 40(3 sin²φ − 1)/3 K (Isca test case) | isothermal atmosphere plus a 0.1 m s⁻¹ seed; uniform surface | **small gap**: the surface profile. The pilot uses uniform 285 K |
| start of the year | t = 0 at the northern spring equinox (Isca `equinox_day` 0) | `time_of_perihelion` | ✓ (computed so that L_s(0) = 0) |

Dependencies that are **not** substituted silently: the boundary layer
(vertical diffusion of heat and momentum plus surface stress), and the
sponge. Until the first exists, no run here is M-I, whatever its length.
Out of scope, as requested: topography, dust, CO2 condensation, real-gas
opacity, dynamical-core changes.

## 2. The pilot: `dev-mars-gray-flat` (development experiment)

T21 (`l_max` 22, retained 21, 32 × 64), the Isca 25-level grid, dt =
600 s, 30 days of 86 400 s (29.3 sols), starting at L_s = 0. Everything in
§1 marked ✓ or fixed here. The two labelled substitutes are Rayleigh drag
and dry adjustment. Two runs from one commit:

- **H**: frictional heating on (the preset default);
- **U**: `--no-frictional-heating`.

States are saved daily (for the sink measurement). After H ends, a
two-sol hourly sample is taken from its final checkpoint
(`tools/radiation/mars_tides.py`) for the tide test.

F is the run's mean absorbed stellar flux, about 100 W m⁻². The 100-day
window of A2 does not fit a 30-day run, so the windows below are stated
for this run and were fixed before it started.

| | criterion | applies to |
| --- | --- | --- |
| M1 stability | reaches day 30 with no validation abort; daily max \|u\| < 150 m s⁻¹; 100 K < T < 400 K everywhere, every day | H and U |
| M2 atmospheric energy | \|mean residual over days 11–30\| ≤ 10⁻³ F; every daily \|residual\| ≤ 10⁻² F (days 1–30) | H |
| M3 whole-system energy | d(E_atm + C ⟨T_s⟩)/dt − ⟨TOA net⟩ meets the same two bounds; the slab's daily heat change equals the accumulated surface flux to 10⁻⁹ relative | H |
| M4 frictional heating | U's mean residual over days 11–30 equals its measured drag + ∇⁸ sink within max(2 standard errors, 10⁻³ F); H meets M2 | H and U |
| M5 diurnal response | over the 2-sol sample of H: the equatorial ⟨T_s⟩ diurnal maximum falls between 11 and 16 h local solar time; the sun-synchronous diurnal (S1) surface-pressure amplitude, averaged over ±30°, is ≥ 5 × 10⁻⁴ of p_s and ≥ 3 × the RMS of what remains after removing the mean, S1 and S2 | H |
| M6 mass | \|relative drift of mean p_s over 30 days\| ≤ 10⁻⁵ | H and U |

Reported, not judged: s/step; the S1 and S2 amplitudes and phases; the
diurnal ranges of T_s and of the lowest-level air temperature; the
H − U difference in surface temperature and TOA balance. That difference
is **not** an equilibrium climate sensitivity: 30 days from 264 K is a
spin-up, and M2–M4 concern energy closure, not equilibration.

What a pass would and would not establish. It would show that the coupled
model runs stably with Martian constants and diurnal forcing, closes its
atmosphere and whole-system budgets at the stated tolerance, and responds
to the diurnal cycle with a detectable thermal tide. It would **not** show
an M-I reproduction (no boundary layer, T21, 30 days, no seasonal mean).
A closed global budget also does not show that each operator conserves
energy locally.

## 3. After the pilot, in order

1. The boundary layer: bulk surface stress and implicit vertical diffusion
   of heat and momentum, with their dissipation in the frictional heating.
   Then remove the two substitutes. This is the gating dependency for any
   M-I claim.
2. The energy-conserving top sponge (existing `DiagonalDamping` hook);
   Isca's ∇⁴ / RA numerics as an option, if the comparison needs them.
3. T42, one Mars year plus the L_s 225°–315° average; compare with Fig. 4b.
4. Then MOLA terrain with nothing else changed (the existing terrain
   import), for the flat-to-topography comparison.
