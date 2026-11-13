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

### Results (2026-11-13; criteria above unchanged)

Both runs came from `b3aaef8` with a clean tree, on one Colab NVIDIA L4
(CuPy 14.0.1). They ran without resuming and took 6.5 min (H, `9f7eaa0a…`)
and 5.0 min (U, `28281209…`), against about 44 min each on the MX110
(0.61 s/step, measured first). Evidence is in
`evidence/dev-mars-gray-flat{-heated,}-T21L25-30d-b3aaef8-l4/`, with
capsules in `docs/runs/`. `verdicts.json` comes from
`tools/radiation/judge_mars_pilot.py`, and
`tests/test_radiation_evidence.py` recomputes M2, M3 and M6 from the series.

| | criterion | measured | verdict |
| --- | --- | --- | --- |
| M1 | stability, max \|u\| < 150 m s⁻¹, 100–400 K | completed; T 134–314 K; **max \|u\| 282 (H), 295 (U) m s⁻¹** | **not met** |
| M2 | atmosphere residual, days 11–30 | −0.0114 W m⁻² (1.1 × 10⁻⁴ F); largest daily value 0.012 | met |
| M3 | atmosphere + slab − TOA | equal to M2 to round-off; slab balance exact | met |
| M4 | U residual = U sinks | −0.048 against −0.037 (tolerance 0.10) | met |
| M5 | diurnal response | p_s S1 = 1.29 % of p_s (≥ 5 × 10⁻⁴ ✓, 5.9 × noise ✓); T_s maximum at **17.0 h** | **not met** |
| M6 | p_s drift | −6 × 10⁻⁶ (H and U) | met |

**M1.** The winds above the bound are confined to the top two levels
(p < 0.2 Pa). They grow through the run (98, 187 and 209 m s⁻¹ in the
day-5, day-15 and day-30 states of H). Below 6 Pa (σ > 0.01) the maximum
stays at or below 77 m s⁻¹. That is the layer where M-I applies its
0.125-day Rayleigh sponge (above 0.5 Pa), which this pilot does not have
(§1, "sponge"). The run did not blow up: temperatures stay inside the
bounds, the budget closes, and the top-level Courant number is about 0.5.
The sponge as the cause is consistent with this but untested: no run with
a sponge exists.

**M5.** The tide is there. The migrating diurnal surface-pressure
component is 1.29 % of p_s in the tropics, peaking near 05:30 local time,
and the semidiurnal is 0.61 %. The lowest-level air temperature has a
3.1 K diurnal amplitude at the equator, but the surface only 0.26 K. The
equatorial T_s maximum falls at 17.0 h, outside the 11–16 h window. That
window was written with real, low-thermal-inertia Mars in mind. M-I's
2 m slab (Cω ≈ 590 W m⁻² K⁻¹ against ≈ 2 W m⁻² K⁻¹ of restoring) lags
the insolation by close to a quarter sol, which predicts this. The
shortfall was noticed on a 3-day development run before H and U started,
and the criterion was not changed. So in this configuration the tide is
driven by shortwave absorption in the air, not by the ground.

**Energy, separated from equilibration.** The drag and ∇⁸ sinks are the
same in both runs (−0.036 and −0.037 W m⁻², 70 % drag). The heating
returns them: U's residual minus H's is 0.036 W m⁻². What both runs keep
is about −0.011 W m⁻² (10⁻⁴ F) that the heating does not address. It is
of the same kind as the Earth time-integration residual; it was not split
here. The runs are far from equilibrium: TOA −116 W m⁻², T_s 285 → 240 K.
That spin-up is no evidence about M-I's climate, and the H − U
differences (0.004 K, 0.01 W m⁻²) are too small to read anything from.
The sinks are 4 × 10⁻⁴ F, so on this short, weak-flow run the heating's
effect is below the M4 tolerance. M4 shows consistency, not sensitivity.

**What this establishes.** With Martian constants, diurnal and seasonal
forcing and M-I's radiation, the coupled model integrates 30 days. It
closes the atmosphere and the whole-system (atmosphere + surface) budget
to 10⁻⁴ F, keeps mass to 6 × 10⁻⁶, and develops a diurnal thermal tide.
**What it does not.** Any M-I reproduction (no boundary layer or sponge,
T21, 30 days, no seasonal mean); stability of the top levels; local
energy conservation by each operator.

## 2b. Sponge rerun (criteria fixed 2026-11-13, before the sponge code)

The change: M-I's top sponge, Isca `damping_driver` `rayleigh` at
`d1321ac3`. It acts on momentum only, at rate (1 / 0.125 day) ((p_b −
p) / p_b)² for p < p_b = 0.5 Pa. With frictional heating on, its
kinetic-energy loss returns as heat (Isca `do_conserve_energy`). The
existing `DiagonalDamping` hook needs level-only rates, so p is taken as
σ × 610 Pa, where Isca uses the local full-level pressure. On flat Mars
the two differ by the surface-pressure variation. Nothing else changes.
The preset is `dev-mars-gray-flat-sponge`, with Hs heated and Us
`--no-frictional-heating`, from one commit, on one machine, for 30 days.

Hs and Us are judged by **M1–M6 exactly as written in §2**. Stated now:

- **If the missing sponge caused M1's failure, M1 is met**: daily max
  \|u\| stays below 150 m s⁻¹. If M1 still fails, that explanation is
  refuted or the sponge is insufficient, and it is reported that way.
- **M5 is expected to fail again** on the 11–16 h window (the slab lag of
  §2's results is untouched by a sponge). The window is not changed.
- M2–M4 and M6 are expected to stay met. A failure of M2 would mean the
  heating does not return the sponge's loss.

Reported, not judged: H − Hs differences in max \|u\| below 6 Pa, in the
S1 tide and in the residual; and the sponge's share of the measured sinks.

### Sponge results (2026-11-13; criteria and predictions unchanged)

Hs (`d2e44058…`) and Us (`1f91ab18…`) ran from `e38890e`, clean tree, on
one Colab L4, without resuming (6.3 and 5.0 min).

| | measured | verdict | predicted |
| --- | --- | --- | --- |
| M1 | max \|u\| 118 / 118 m s⁻¹ (H, U: 282 / 295); T 161–306 K | **met** | met |
| M2, M3 | residual −0.0115 W m⁻² (1.1 × 10⁻⁴ F); slab exact | met | met |
| M4 | Us residual −0.047 against sinks −0.037 | met | met |
| M5 | p_s S1 1.29 % (≥ 5 × 10⁻⁴ ✓, 3.7 × noise ✓); T_s maximum 17.0 h ✗ | not met (window only) | not met (window only) |
| M6 | −8 × 10⁻⁶ | met | met |

Every prediction held. The sponge does exactly what it is for: the
top-two-level maximum falls from 288 to 90 m s⁻¹, winds below 6 Pa are
unchanged (77 → 80 m s⁻¹), and the residual is unchanged (−0.0114 →
−0.0115 W m⁻²). Its own sink, 0.0022 W m⁻² or 6 % of the measured sinks,
is returned by the frictional heating like the others. The sponge layer
holds almost no mass.

**The M5 sample had to be redone.** The pre-registered sample continued
2 sols past the end of the run. For Hs that continuation went non-finite
at step 146 (§2c), so the first Colab `tides.json` is all NaN, and its
00:00 "maximum" is `np.argmax` of a NaN curve. Both files are kept
unchanged as the record. The valid sample, `tides-last2sols.json`, is the
last two sols of Hs (days 27.95–30.0). It was replayed from the run's
day-20 checkpoint on the MX110 and checked against all ten saved daily
states in the interval: the departure grows from 1.6 × 10⁻¹³ on day 21 to
5.9 × 10⁻¹⁰ on day 30, inside the tool's 10⁻⁹ tolerance. This changes
the sampling window, not the thresholds. The tool now aborts on any
non-finite state, field or fit coefficient. The same replay for H departs
by 1.4 × 10⁻⁸ (without the sponge, the fast top levels amplify the
cross-platform rounding faster). The tool refuses it, so H's in-run M5
needs the L4. H's recorded M5 came from a continuation that stayed finite.

### 2c. Bottom-layer sensible-heat instability (found 2026-11-13)

Diagnosing the non-finite continuation exposed a defect that the 30-day
criteria did not test. The coupling applies the bulk sensible exchange as
a forward (lagged) step into the lowest layer, and that layer holds
Δp ≈ 0.73 Pa (Isca's grid, σ = 0.9988–1). Its forward-Euler number
2 Δt ρ C |v| g / Δp must stay below 2. On the saved states of **both**
heated runs it crosses 2 near day 20, as the lowest-level winds pass
≈ 25 m s⁻¹. By day 30, H has 141 points above 2 (maximum 6.4) and Hs has
223 (maximum 4.3). In Hs's continuation the maximum reaches 13, and the
lowest-level temperature oscillates with growing amplitude until it is
negative at step 145. The dry adjustment hides warming overshoots by
mixing them upward, but nothing catches a cooling one. So:

- **The sponge did not cause it.** It is present in H too, which survived
  its 2-sol continuation.
- **This configuration has an effective lifetime of about 30 days.** No
  run of it beyond ~30 days, and none at T42 or over a season, is valid
  until the surface exchange is implicit.
- **The fix is part of the boundary-layer dependency:** surface fluxes
  solved implicitly with the vertical diffusion, as Isca does. The 46–51
  m s⁻¹ lowest-level winds at day 30 are a second symptom of the weak
  Rayleigh-drag stand-in for surface stress.

Restoration itself is exact: replaying from the day-20 checkpoint
reproduces the saved day-21 state to 1.6 × 10⁻¹³ across platforms.

## 2d. Implicit surface exchange (criteria fixed 2026-11-13, before the code)

The change addresses §2c. The bulk sensible exchange becomes a
backward-Euler step between the bottom layer (Ca = c_p Δp / g) and the
slab (Cs), with the exchange coefficient k = ρ C |v| c_p frozen over the
step:

  F′ = F / (1 + k w / Ca + k Δt / Cs),

where w is the window over which the coupling applies the heating (2 Δt
in a leapfrog step) and Δt the slab step. The same F′ leaves the slab and
enters the air, so the exchange conserves energy exactly, and the bottom
layer's relaxation per step, x / (1 + x), never exceeds 1. This is a
numerical treatment of the existing flux, not new physics, and it is
opt-in (`"implicit": true` in the sensible spec), so every existing
configuration is bit-identical. There is still no vertical diffusion or
surface stress. The preset is `dev-mars-gray-flat-sponge-implicit`, with
Hi heated and Ui `--no-frictional-heating`, from one commit, on one
machine, **60 days** (twice the former lifetime).

Unit criteria:

- I1: a column at the Mars bottom-layer values measured in §2c
  (k w / Ca ≈ 13) oscillates with growing amplitude under the explicit
  exchange, and relaxes monotonically to equilibrium under the implicit
  one. Ca ΔT_a + Cs ΔT_s = 0 to round-off on every implicit call.
- I2: without the flag, every existing test and hash is unchanged.

Run criteria (M1–M6 as in §2, with the windows moved to the end of the
longer run):

- M1 and M6 over all 60 days; M2–M4 over days 41–60 (same length as
  before).
- **N1**: the 2-sol continuation of Hi past day 60 stays finite (the test
  Hs failed).
- M5 from that continuation (the original protocol).

Stated now: M1–M4, M6 and N1 met; M5 not met on the 11–16 h window alone
(slab lag, untouched). If N1 fails, the explicit exchange was not the only
instability, and it is reported that way. Reported, not judged: the
largest implicit-relaxation number, the lowest-level winds, and the
day-30 differences from Hs.

## 3. After the pilot, in order

1. ~~The energy-conserving top sponge~~: done (§2b). M1 met as predicted.
2. The boundary layer: bulk surface stress and implicit vertical diffusion
   of heat and momentum, **with the surface fluxes solved implicitly**
   (§2c), their dissipation in the frictional heating. Then remove the
   two substitutes. This is the gating dependency for any M-I claim and
   for any run longer than ~30 days. Isca's ∇⁴ / RA numerics as an option,
   if the comparison needs them.
3. T42, one Mars year plus the L_s 225°–315° average; compare with Fig. 4b.
4. Then MOLA terrain with nothing else changed (the existing terrain
   import), for the flat-to-topography comparison.
