# Isca Mars M-I: gap analysis and flat-Mars pilot (2026-11-13)

> **Amended 2026-11-14 (§4).** §1–§3 below are the historical record and
> are unchanged. The criteria M1–M7, N1 and I1–I2 are **historical
> development criteria**. They judged engineering experiments, and their
> verdicts stand as written. They are neither current requirements nor
> published M-I criteria. The current requirements register, the published
> M-I comparison protocol and the remaining configuration differences are
> in §4. The reasoning is in `MARS_MI_REQUIREMENTS_AUDIT.md`.

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
| boundary layer, heat | Monin–Obukhov surface fluxes plus vertical turbulent diffusion; roughness 3.21 × 10⁻⁵ m; no gustiness (Isca test case). **Full** MO drag and similarity (`use_frierson_mo_drag` false by default; read 2026-11-13, after §2d) | bulk sensible flux into the lowest layer, FHZ06-simplified coefficient, same roughness, gustiness 0 allowed; vertical diffusion since §2e | **partial**: FHZ06-simplified vs full MO (stated in §2e) |
| momentum exchange | surface stress plus vertical diffusion of momentum (full MO) | §2e: bulk surface stress + implicit vertical diffusion, FHZ06-simplified coefficient, neutral K profile | **partial since §2e** (was: Held–Suarez Rayleigh drag substitute, **labelled**) |
| convection | "None" (Table 1): the boundary-layer diffusion removes instability | none since §2e (`convection: None` in the preset; was: instantaneous dry adjustment, **labelled**) | ✓ since §2e |
| sponge | Rayleigh 0.125 day above 0.5 Pa, energy-conserving (Isca test case) | none in the pilot | ✓ since §2b (`rayleigh_sponge_damping`, p = σ × 610 Pa) |
| global fixers | Isca `spectral_dynamics_nml` defaults `do_mass_correction` and `do_energy_correction` to `.true.`, and the grey-Mars namelist overrides neither (it sets only `do_water_correction: False`). Every step, p_s is rescaled to the previous global mean, and a uniform temperature increment makes the mass-weighted ⟨½\|v\|² + c_p T⟩ of the new level equal the previous one plus the step's tendencies | none: mass is conserved to the core's truncation, and the energy residual is measured, not removed | **gap, found after §2d** (read 2026-11-13). It bears on M6 and on any energy comparison with Isca (§2d results) |
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

### Implicit-exchange results (2026-11-13; criteria and predictions unchanged)

Hi (`18ee5dc5…`) and Ui (`7d28666a…`) ran from `63d05a2` with a clean
tree, one after the other in one Colab session, without resuming (14.1
and 11.1 min, 0.098 and 0.077 s/step). The GPU was a **Tesla T4**, as
each run's `created` event records, although the cell was written for an
L4. The evidence directories are named `-t4` accordingly. Evidence is in
`evidence/dev-mars-gray-flat-sponge-implicit{-heated,}-T21L25-60d-63d05a2-t4/`.
`verdicts.json` names `fba92fb` (M1–M6) and the 41–60 window of §2d, and
`state_diagnostics.json` holds the reported quantities below
(`tools/radiation/mars_state_diagnostics.py`, from the saved states).

| | measured | verdict | predicted |
| --- | --- | --- | --- |
| I1, I2 | unit tests; every earlier test and hash unchanged | met | met |
| N1 | the continuation past day 60 ran all 295 steps finite | **met** | met |
| M1 | completed; T 136–306 K; **max \|u\| 182 (Hi, day 44) / 178 (Ui, day 43) m s⁻¹**, above 150 on 7 days of each run, none before day 36 | **not met** | met |
| M2, M3 | residual over days 41–60 −0.0120 W m⁻² (1.2 × 10⁻⁴ F); largest daily value 0.0124; slab exact (9 × 10⁻¹⁴) | met | met |
| M4 | Ui residual −0.111 against sinks −0.101 (tolerance 0.099) | met | met |
| M5 | p_s S1 1.33 % (≥ 5 × 10⁻⁴ ✓, 4.1 × noise ✓); T_s maximum 17.0 h ✗ | not met (window only) | not met (window only) |
| M6 | **−1.97 × 10⁻⁵ (Hi), −1.89 × 10⁻⁵ (Ui) over 60 days** | **not met** | met |

Two predictions failed: M1 and M6. Both failures are reported as measured.
Neither comes from the change under test.

**The implicit exchange does what it was for.** Over the 30 days the
explicit runs survived, it changes nothing that matters. At day 30, max
\|u\| is 106.6 against Hs's 104.6 m s⁻¹, with the largest daily difference
5.7. T_s differs by 0.004 K, TOA by 0.03 W m⁻², the residual by less than
0.001 W m⁻², and the mean p_s by less than 10⁻⁷. The one term it alters,
the global-mean sensible flux, differs most: −0.040 against +0.024 W m⁻²
at day 30. That is consistent with the explicit exchange oscillating at
its unstable points. The stiffness that broke the explicit step persists
and grows. k w / C_a exceeds 2 somewhere on 41 of the 60 daily states,
from day 20 on, at up to 403 of the 2048 points, with a maximum of **6.1**
(Hi, day 59; Ui 6.0, day 54). An explicit step is unstable at every one
of those points. The implicit one ran through them and through the
continuation. Lowest-level winds reach 63 m s⁻¹ (Hi) and 61 m s⁻¹ (Ui).

**M1: a strengthening jet after day 35, not an instability.** Through day
30 the largest \|u\| is 112 m s⁻¹ (at 0.48 Pa), and below 6 Pa it is 79,
as in Hs. Then the spin-up continues: kinetic energy goes from 27 to
44 kJ m⁻², T_s from 240 to 222 K, and TOA is still −41 W m⁻² at day 60.
The southern high-latitude westerly (52–69° S) strengthens through 0.5–4
Pa. In Hi, the levels at 0.48, 1.06, 2.2 and 4.2 Pa all pass 150 m s⁻¹,
and below 6 Pa the maximum rises from 79 to 125 m s⁻¹ (Ui: 136). Most
daily maxima sit at 0.48 Pa, just below the sponge's 0.5 Pa base. There
the sponge's rate is (0.04)² of its full value (e-folding ≈ 80 days), as
in Isca, whose profile is the same. Temperatures stay inside the bounds,
the budget closes at 10⁻⁴ F, and N1 holds. So the 150 m s⁻¹ bound,
written for a 30-day spin-up, is crossed by a jet that is still
strengthening. Whether that jet is too strong cannot be decided here.
One measured hint is the drag stand-in, which damps the bottom layer at
1 day⁻¹. Where the exchange is strongest, a bulk surface stress with the
same coefficient as the heat exchange would damp it at x / w ≈ (200
s)⁻¹. Lowest-level winds of 60 m s⁻¹ are what a missing surface stress
would produce. That points to the boundary layer again (§3), and it is
untested.

**M6: the core's mass error, not the change.** The mean-p_s drift is the
same to two digits in all six Mars runs through day 20 (−1.8 × 10⁻⁶ at
day 10, −3.6 × 10⁻⁶ to −3.7 × 10⁻⁶ at day 20), whatever the sponge,
exchange or heating. After
that it depends on the flow: −6.0 × 10⁻⁶ at day 30 without the sponge,
−8.0 × 10⁻⁶ with it, under either exchange. It does not level off: days
31–60 alone give −1.17 × 10⁻⁵. So the bound fails at the 30-day rate as
well as over 60 days. (Extending M6 to 60 days without scaling its bound
was an oversight in §2d. It changes nothing, because days 31–60 alone
exceed the 30-day bound.) In the saved states the drift is the small
remainder of two larger, opposing changes. ⟨ln p_s⟩ falls by
2.3 × 10⁻⁴ while half the variance of ln p_s grows by 2.1 × 10⁻⁴ (Hi,
day 60). The ln p_s form conserves neither mean exactly, and nothing here
corrects it; this is the ln-p_s mass drift already listed in
`docs/KNOWN_LIMITATIONS.md` ("PE accuracy"), now measured on Mars. Isca
does correct it (§1, "global fixers"): every step it rescales p_s
to the previous global mean.

**Energy.** Hi's residual is −0.0120 W m⁻² (1.2 × 10⁻⁴ F), the same size
as in every earlier Mars run. The sinks have grown with the flow to
0.101 W m⁻², about 10⁻³ F. M4 is now a test rather than a consistency
check: the unheated residual is distinguishable from zero at the
tolerance, and it agrees with the measured sinks to 0.010 W m⁻², close to
Hi's own residual. The pair therefore splits the residual as before.
0.10 W m⁻² is the sinks, which the heating returns. About −0.011 W m⁻² is
the time-integration part, which the heating does not address. Isca's
`do_energy_correction` removes exactly that kind of remainder every step,
as a uniform temperature increment. In an energy comparison with Isca,
the residual measured here therefore corresponds to the size of Isca's
correction, not to a closure Isca demonstrates.

**What this establishes.** The implicit exchange removes the §2c
instability without changing the first 30 days. The configuration now
integrates 60 days and continues finite, and its energy closes to
1.2 × 10⁻⁴ F, with the frictional-heating split now resolved by M4. **What
it does not establish:** an M-I reproduction, the wind bound over 60
days, mass conservation over 60 days, or anything about the jet's
realism.

## 2e. The boundary layer (pre-registered 2026-11-13, before any run)

The gating dependency of §1 and §3: bulk surface stress plus implicit
vertical diffusion of heat and momentum, replacing the two labelled
substitutes (Held–Suarez drag and the dry adjustment; M-I Table 1 has
`convection: None`). Preset `dev-mars-gray-flat-bl` = the §2d implicit
configuration with `drag: None`, `convection: None` and the boundary
layer on. Nothing else changes.

**The scheme** (FHZ06 inside Isca's structure; Isca sources as in §1):

- Exchange coefficient: FHZ06 eqs. 12–14 (`tropoi.physics.surface.
  drag_coefficient`, the one the heat exchange already uses), roughness
  3.21 × 10⁻⁵ m, Ri_c = 1, gustiness 0; u\* = √C \|v₁\|.
- Depth h: the lowest height where the bulk Richardson number of the dry
  static energy reaches 1.0 (Isca `diffusivity` `rich_crit_pbl` default,
  `do_simple` branch), interpolated linearly in Ri between the bracketing
  full levels.
- Diffusivity at the interior interfaces, momentum and heat alike
  (neutral Prandtl 1): K = κ u\* z below 0.1 h (`frac_inner`), then
  K = κ u\* z (1 − (z − 0.1h)/(0.9h))² up to h, zero above.
- One backward-Euler tridiagonal solve per column over the stepper
  window for u, v and the dry static energy (geopotential frozen); the
  temperature change is ΔDSE / c_p. Bottom boundary: the stress
  −ρ₁ C \|v₁\| u₁ⁿ⁺¹, implicit in the new lowest-level wind exactly as
  §2d's heat exchange (measured need: at \|v₁\| = 60 m s⁻¹ the stress
  stiffness k w / m₁ ≈ 8 and the first interior diffusion stiffness
  ≈ 10²; both explicit-unstable). The dry static energy has a zero-flux
  bottom: the surface sensible flux stays where §2d put it.
- With frictional heating on, the layer heats by
  −[(u + u′) Δu + (v + v′) Δv] / (2 c_p w), whose mass-weighted column
  integral equals the kinetic energy the solve removed, identically.
- Evaluated on the product grid like every other nonlinear momentum term
  (weak-form vector analysis, product cut); the slab temperature it sees
  is the spectral synthesis of the state-grid field.

**Differences from Isca, stated up front:** Isca's grey Mars runs the
full Monin–Obukhov drag and similarity functions (`use_frierson_mo_drag`
defaults to false; `monin_obukhov_nml` rich_crit 2.0, stable_option 1,
drag_min 10⁻⁵) — found 2026-11-13; §1's "same roughness" row glossed
this. Here both the surface exchange and the K profile use the
FHZ06-simplified forms (neutral φ). Isca also solves the surface heat
flux jointly with the diffusion (one down-up sweep); here they are
split: §2d's implicit exchange first, diffusion with zero surface flux
after, both unconditionally stable alone. Isca's vertical diffusion
returns its dissipation as heat unconditionally (`idealized_moist_phys`
hard-codes it); here it follows `frictional_heating` like every other
sink.

**Run pair and criteria.** Hb (heated) and Ub (unheated), 60 days of
86 400 s, T21L25, dt 600 s, states daily, judged exactly as §2d with
these changes:

| | criterion | prediction |
| --- | --- | --- |
| N1 | day 60 reached, no abort; a restarted continuation stays finite | met |
| M1 | daily max \|u\| < 150 m s⁻¹, 100 K < T < 400 K | **not met** on the wind (the §2d jet lives at 0.5–4 Pa, far above any h the scheme can produce; the stress does not reach it directly), temperatures met |
| M7 (new) | daily max lowest-full-level \|u\| ≤ 40 m s⁻¹ over days 11–60 | met (this is what the stress is for; §2d measured 63 m s⁻¹ without it) |
| M2, M3 | as §2 (residual ≤ 10⁻³ F windowed, ≤ 10⁻² F daily; whole-system same; slab 10⁻⁹) | met |
| M4 | Ub's residual equals its measured sinks (now drag-free but including the boundary layer's kinetic-energy removal) within max(2 se, 10⁻³ F) | met |
| M5 | as §2 over a 2-sol hourly sample of Hb | amplitude met; T_s peak window **not met** (the 17.0 h peak is slab inertia, which the boundary layer does not touch) |
| M6 | \|mean-p_s drift\| ≤ 10⁻⁵ over days 0–30 **and** ≤ 2 × 10⁻⁵ over days 0–60 (the 30-day bound, scaled, fixing §2d's oversight) | 30-day met; 60-day met at low confidence (§2d measured −1.97 × 10⁻⁵, 1.5 % under the bound, and the drift tracks the flow the boundary layer now changes) |

Reported, not judged: s/step; daily maximum and median h; the daily
exchange and diffusion stiffnesses; the boundary layer's kinetic-energy
sink (accumulated in-run like the coupling's budget); the jet's level
and latitude; and the two **fixer-equivalent series** derived from the
existing records with no new run-time code — the per-day mass factor
Isca's `do_mass_correction` would have applied (⟨p_s⟩_{d−1}/⟨p_s⟩_d from
`series.json`) and the uniform temperature increment per day its
`do_energy_correction` would have applied (−residual · g /(c_p ⟨p_s⟩)).
Neither correction is applied; §3 item 4 stays open for the opt-in
versions.

### Boundary-layer results (2026-11-14; criteria, thresholds and predictions unchanged)

Hb (`004fd4b8…`) and Ub (`f01c4203…`) ran 60 days from `06f5356` on a
Colab L4, without resuming. The physics is unchanged at `7dfc39d` and
after, so nothing was rerun. Evidence is in
`evidence/dev-mars-gray-flat-bl{-heated,}-T21L25-60d-06f5356-l4/`. The
judged verdict is **`verdicts-2e-corrected.json`**, written by
`judge_mars_pilot.py --criteria 2e --days 60 --window 41 60` from the saved
evidence alone, after the three postprocessing fixes below. The first
verdict, `verdicts-2e-recovery.json`, is kept unmodified for provenance
(SHA-256 `bea7e011…`); it is superseded on M4 only.

| | measured | verdict | predicted |
| --- | --- | --- | --- |
| N1 | day 60 reached; the 2-sol continuation from the day-60 checkpoint ran all 295 steps finite | met | met |
| M1 | completed; T 140–319 K; **max \|u\| 170 (Hb, day 42) / 186 (Ub, day 56) m s⁻¹**, ≥ 150 on 3 and 5 days, none before day 42 | **not met** | not met |
| M7 | **daily max lowest-level wind speed 46.7 (Hb, day 27) / 47.2 (Ub, day 47) m s⁻¹** over days 11–60; above 40 on 13 (Hb) and 4 (Ub) days | **not met** | met |
| M2, M3 | Hb residual over days 41–60 −0.0148 W m⁻² (1.5 × 10⁻⁴ F); largest daily 0.0158; slab exact | met | met |
| M4 | Ub residual −0.2846 against sinks −0.2700 (sponge + hyperdiffusion −0.1663 on the states, boundary layer −0.1037 from the series); difference −0.0146, tolerance 0.099 | met | met |
| M5 | p_s S1 1.42 % (≥ 5 × 10⁻⁴ ✓) but only 2.5 × noise (0.58 %) ✗; T_s maximum 17.05 h ✗ | **not met** | not met (window only) |
| M6 | **day 30: −1.08 × 10⁻⁵ (Hb), −0.95 × 10⁻⁵ (Ub); day 60: −2.57 × 10⁻⁵ (Hb), −2.21 × 10⁻⁵ (Ub)** | **not met** | met (low confidence) |

The real failures are **M1, M5, M6 and M7**. M1 failed as predicted. M7,
M6 and part of M5 failed against their predictions; all are reported as
measured.

**Postprocessing corrections (no physics, threshold or run changed).**

1. *M7 window.* The judge at `7dfc39d` started M7 at `--window`'s first day,
   but §2e fixes M7 at days 11–60 whatever the M2–M4 window. With the 41–60
   window it would have reported Hb at 45.5 m s⁻¹ (day 44) and missed the
   day-27 peak of 46.7. M7 fails either way. The judge now starts M7 at
   day 11 (`M7_FIRST_DAY`) and records `m7_days`. The recovery file already
   carries the 11–60 maxima.
2. *M4 sink accounting.* `summarize_run.py` measures the sinks on the saved
   states. Those carry no slab temperature, so the boundary layer, which
   needs it, cannot be evaluated there and was left out. The recovery
   verdict compared Ub's residual with the sponge and hyperdiffusion alone
   and reported −0.1183 W m⁻², outside tolerance. The run records the
   boundary layer's own sink (`bl_kinetic_sink`). The judge now adds its
   window mean, −0.1037, once: not when the measured operators already
   include the boundary layer, and not under frictional heating, which
   returns that energy as heat. The difference becomes −0.0146 W m⁻², about
   Hb's own residual, which is the §2d split again.
3. *Tidal restoration.* `mars_tides.py` built its stepper without the
   boundary layer, so the saved hook signatures refused the checkpoint. It
   now builds the boundary layer as `RadiativeRun` does and restores its
   budget state. The Colab sample was made with the same patch applied in
   memory, so `tides.json` stands. A GPU test replays a boundary-layer
   checkpoint onto the run's own saved state with zero departure.

**M7: the stress lowers the lowest-level wind but not to 40 m s⁻¹.** §2d
reached 63 (Hi) and 61 (Ui) m s⁻¹ without a surface stress. The boundary
layer brings that to 47, a quarter less, but the daily maximum exceeds
40 m s⁻¹ in Hb on days 16 and 27 and on 11 days from day 44 on, and in Ub
on days 27, 40, 47 and 52. The
diagnosed depth h reaches 20 km at its deepest, so the scheme is active.
Why the bottom wind stays this strong is not decided here. The documented
departures of §2e (FHZ06-simplified neutral φ against Isca's full
Monin–Obukhov; the split surface-heat and diffusion solves) are untested
candidates. So is the still-strengthening flow aloft (M1).

**M1.** As predicted, the jet aloft still crosses 150 m s⁻¹ after day 40,
at 170 and 186 m s⁻¹ (§2d: 182 and 178). The boundary layer does not reach
it.

**M6.** At day 10 the drift is −1.9 × 10⁻⁶, as in every Mars run. It
departs from §2d by day 20 (−4.1 × 10⁻⁶ to −4.2 × 10⁻⁶ against −3.6 ×
10⁻⁶ to −3.7 × 10⁻⁶) and keeps growing faster: −1.08 × 10⁻⁵ at day 30 in
Hb, just over the bound, and −2.57 × 10⁻⁵ at day 60. Ub meets the 30-day half and fails the
60-day half. The low-confidence prediction failed: the flow the boundary
layer changes moves the ln-p_s mass error the wrong way. The
fixer-equivalent log-sum of the mass factor is 2.57 × 10⁻⁵, the drift
itself, which is what Isca's `do_mass_correction` would have removed.

**M5.** The T_s peak is at 17.05 h, as predicted (slab inertia). Against
the prediction, the tropical p_s S1 (1.42 %) is now only 2.5 times the
residual noise (0.58 %), so the 3 × noise part fails as well. It passed
in §2d (1.33 %, 4.1 × noise). Its amplitude is not smaller; the
non-tidal variability is larger.

**Energy.** Hb's residual, −0.0148 W m⁻² (1.5 × 10⁻⁴ F), stays at the size
of every earlier Mars run. The uniform temperature increment Isca's
`do_energy_correction` would have applied averages 0.0082 K day⁻¹
(largest 0.0098). In Hb the boundary layer removes −0.101 W m⁻² of kinetic
energy over days 41–60 and returns it as heat.

**What this establishes.** The configuration with the boundary layer in
place of both substitutes integrates 60 days and continues finite. Its
energy closes to 1.5 × 10⁻⁴ F and its sinks account for the unheated
residual. **What it does not establish:** the lowest-level wind bound,
the wind bound aloft, mass conservation over 30 or 60 days, the tidal
criterion, or an M-I reproduction.

## 3. After the pilot, in order

1. ~~The energy-conserving top sponge~~: done (§2b). M1 met as predicted.
2. ~~An implicit surface exchange~~: done (§2d, opt-in). The §2c
   instability is gone and runs reach 60 days and beyond. M1 and M6 failed
   on the wind bound and the mass drift, neither caused by the change.
3. The boundary layer: run and judged in §2e. N1 and M2–M4 met. M1, M5,
   M6 and M7 not met: the lowest-level winds fall from 63 to 47 m s⁻¹ but
   stay above the 40 m s⁻¹ bound, and the drift and jet are unchanged in
   kind. Open candidates for M7 are the full Monin–Obukhov forms and
   Isca's joint surface-heat and diffusion sweep. Isca's ∇⁴ / RA numerics
   as an option, if the comparison needs them.
4. Isca's global fixers, as opt-ins that record what they remove: the mass
   correction (it would close M6 by construction, so its per-step factor
   is the diagnostic), and a log of the energy correction Isca would apply,
   without applying it. That log is the like-for-like quantity for an
   energy comparison with M-I.
5. T42, one Mars year plus the L_s 225°–315° average; compare with Fig. 4b.
6. Then MOLA terrain with nothing else changed (the existing terrain
   import), for the flat-to-topography comparison.

## 4. Requirements amendment (2026-11-14)

This amendment corrects how requirements are written and classified. It is
**not** a retrospective attempt to make any experiment pass. Nothing in
§1–§3 changes: no criterion, prediction, window, threshold, measurement or
verdict. No verdict moves from "not met" to "met", and every verdict file
stays byte-identical (§4.9). The analysis behind it, with sources and
methods, is `MARS_MI_REQUIREMENTS_AUDIT.md` (the "audit"). Isca statements
here were checked at the pinned commit `d1321ac3` (audit §4). Those not
checked are listed in §4.10.

### 4.1 Three kinds of statement, kept apart

1. **Historical development criteria.** These are M1–M7, N1 and I1–I2 of
   §2–§2e. They were pre-registered predictions about engineering
   experiments: does a sponge, an implicit exchange or a boundary layer do
   what it was added for? They stay frozen with their verdicts. None of
   them is a published M-I criterion. Their thresholds were chosen inside
   the project or inherited from the dry gray Earth A-series, which was
   itself written after the runs it describes (audit §1–§2). They are not
   requirements from here on.
2. **Current requirements** (§4.3). Each is one of four classes:
   implementation requirement (IR), numerical-integrity test (NI),
   diagnostic measurement (DM) or climate-validation target (CV). Each
   states its origin, justification, observable, evidence source and
   verification method. Only IR and NI are pass/fail. A DM is reported and
   never judged.
3. **The published M-I reproduction target** (§4.4). This is the paper's
   own comparison: Fig. 4b at T42 over L_s 225°–315°. It is the only
   external climate reference the paper gives for M-I. No run so far can
   be compared with it.

Origin codes:

- **P**: the paper (Table 1, the text, the figures).
- **Isca**: Isca's grey-Mars test case or defaults at `d1321ac3`. It is
  used where the paper leaves a value implicit, and is not proof that the
  paper's run used it (§4.10).
- **D**: derived from physical or numerical principle.
- **I**: chosen inside the project.

### 4.2 What each historical failure did and did not establish

| | historical verdict (unchanged) | what the failure establishes | what it does not establish |
| --- | --- | --- | --- |
| M1 (§2) | not met | the top two levels were undamped without the sponge; tested and confirmed by §2b | — |
| M1 (§2d, §2e) | not met | a southern westerly jet at 0.5–4 Pa still strengthening at L_s ≈ 29° (southern autumn). Its zonal mean is about 90–99 m s⁻¹; grid-point maxima are 170–186 m s⁻¹ (audit §3) | an instability, which N1, the temperature bounds and energy closure all rule out. Nor an error: M-I's own Fig. 4b zonal-mean jet exceeds 140 m s⁻¹ |
| M5 window (all) | not met | the T_s peak comes at 17.0–17.05 h, as the linearised slab expectation predicts for M-I's 2 m slab (§4.7) | a defect: a peak inside 11–16 h was not to be expected of M-I's surface |
| M5 3 × noise (§2e) | not met | the tropical p_s varies more about the fitted tide than in §2d | that the tide went undetected: the "noise" is physical variability, not measurement error |
| M6 (§2d, §2e) | not met | the ln-p_s core's mass error grows with the flow: −2.57 × 10⁻⁵ in 60 days (Hb) | an M-I discrepancy (Isca removes the drift every step), or an energy-budget problem (its energy equivalent is about 1.4 × 10⁻⁶ F) |
| M7 (§2e) | not met | the surface stress cut the lowest-level wind from 63 to 47 m s⁻¹, a quarter, not to 40 | that 40 m s⁻¹ is the right value. It was an internal prediction with no published counterpart |

### 4.3 Current requirements register

Status is given at `06f5356`, the §2e configuration
(`dev-mars-gray-flat-bl`).

**Implementation requirements (IR): conformance to M-I's configuration.**
Pass/fail per configuration. A development run may depart from an IR only
when the departure is labelled (§4.5). A run with an IR unmet cannot be
called M-I.

| id | requirement | origin | observable | evidence | verification | status |
| --- | --- | --- | --- | --- | --- | --- |
| IR-1 | a 3396 km, g 3.71, R 192, κ 0.22727, Ω = 2π/88 308 s (7.12 × 10⁻⁵), reference p_s 610 Pa | P (Table 1: a, Ω, 6.1 hPa); Isca `constants_nml` (R, κ, g, rotation period) | config fields | `config.json` | `test_the_flat_mars_development_preset_matches_m_i_where_it_claims_to`; `test_the_bl_preset_removes_both_substitutes_and_moves_no_other_identity` | met |
| IR-2 | 25 levels on Isca's grey-Mars σ interfaces (`bk`, with `pk` = 0) | P (25 levels); Isca test case (the values) | `vertical.sigma` | `config.json` | same test (`ISCA_GREY_MARS_BK`) | met for the interfaces. Full-level placement not compared (§4.10) |
| IR-3 | gray radiation: τ_lw = 0.2 (p/610 Pa), τ_sw = 0.2 (p/610 Pa)⁴, vertical-path shortwave, reflected beam not re-absorbed, albedo 0.3 | P (eqs. 14–15, τ 0.2); Isca `two_stream_gray_rad` (vertical path, reflection, 610 Pa reference) | radiation profiles and provider type | `config.json` | same test (profile terms and reference) | met, except the cosine of the zenith angle (§4.5) |
| IR-4 | diurnal and seasonal insolation: S 589 W m⁻², e 0.0935, obliquity 25.19°, perihelion at L_s 250.85°, year 59 166 360 s, sol 88 440 s, L_s = 0 at t = 0 | P (diurnal cycle; perihelion near L_s 251°); Isca `astronomy_nml`, `constants_nml`, `equinox_day` 0 | insolation config; L_s(0) | `config.json` | same test | met |
| IR-5 | slab surface of 2 m × `RHO_CP` = 8.26 × 10⁶ J m⁻² K⁻¹; flat; roughness 3.21 × 10⁻⁵ m for heat and momentum; no gustiness | P (land depth 2 m, no topography); Isca `mixed_layer`, `constants`, `idealized_moist_phys_nml`, `constant_gust` 0 | surface and BL config | `config.json` | same tests | met |
| IR-6 | Rayleigh sponge, 0.125 day, rate ∝ ((p_b − p)/p_b)² for p < 0.5 Pa at the **local** full-level pressure; its KE loss returned as heat | Isca `damping_driver` | sponge config | `config.json` | `test_rayleigh_sponge_rates_are_iscas_damping_driver` | **partial**: p = σ × 610 Pa (§2b) |
| IR-7 | no convection scheme | P (Table 1 "None") | `convection` | `config.json` | BL preset test | met since §2e |
| IR-8 | boundary layer as Isca: full Monin–Obukhov `mo_drag`; K from `mo_diff` below 0.1 h and tapered above; h from Ri = 1; one implicit down–up sweep with the slab between; vertical-diffusion dissipation always returned as heat | Isca (`surface_flux`, `monin_obukhov`, `diffusivity`, `vert_diff`, `idealized_moist_phys`) | BL config and scheme | source, `config.json` | `test_boundary_layer.py` (the FHZ06 scheme only) | **partial**: FHZ06-simplified, neutral, split solves (§4.5) |
| IR-9 | T42 | P (Table 1) | `l_max`, retained truncation | `config.json` | config check | **not met** (T21) |
| IR-10 | numerics: Δt 110 s; ∇⁴ hyperdiffusion (`damping_order` 2, coefficient (0.1 day)⁻¹, `resolution_dependent`); Robert–Asselin 0.04 with RAW off | Isca (test case and defaults) | `dt`, `hyperdiffusion_*`, `raw_*` | `config.json` | config check | **not met** (§4.5) |
| IR-11 | global mass and energy corrections each step | Isca `spectral_dynamics` defaults (on; not overridden) | — | — | — | **not implemented** (§3 item 4) |
| IR-12 | initial state: isothermal 264 K at rest; T_s = 285 − 40 (3 sin²φ − 1)/3 K | Isca (`initial_temperature`, `tconst`, `delta_T`) | `initial`, `surface` | `config.json` | config check | **partial**: uniform 285 K; 0.1 m s⁻¹ seed |
| IR-13 | no temperature floor, no CO₂ condensation, no dust, no topography | P (§4.2: "no lower bound on the temperature"; no dust or topography in M-I) | config | `config.json` | config check | met |

**Numerical-integrity tests (NI).** Pass/fail on every run.

| id | test | origin and justification | observable | evidence | verification |
| --- | --- | --- | --- | --- | --- |
| NI-1 | the run reaches its configured last day without a validation abort, and every saved state is finite | D: a necessary condition (was M1a) | `completed` event; finiteness of the states | `events.jsonl`, states, `series.json` | the judge's `last_day`; the run's own finiteness checks |
| NI-2 | every daily T inside [0, 700] K | Isca: the grey-Mars `valid_range_T`, the range outside which M-I itself aborts. It replaces the internal 100–400 K, which becomes DM-3. Here it is checked on daily snapshots; Isca checks every step | `t_min`, `t_max` | `series.json` | from the series |
| NI-3 | after any change to the surface coupling, the boundary layer or the time stepping: a run restarted from the final checkpoint stays finite for ≥ 2 sols | D: the regression test for the §2c instability (was N1). It is weak evidence (one state, 2 sols) and only a regression guard | finiteness of every continued step | `tides.json` (`mars_tides.py` aborts on any non-finite value) | the tool |
| NI-4 | the slab's daily heat change equals the accumulated surface flux to 10⁻⁹ relative | D: a bookkeeping identity (observed about 10⁻¹³) (was part of M3) | `C ΔT_s / (surface_net · day) − 1` | `series.json` | judge `slab_max_relative_error` |
| NI-5 | column operators conserve what they should: implicit exchange C_a ΔT_a + C_s ΔT_s = 0 to round-off; tridiagonal solve to 10⁻¹² relative; column dissipative heating equals the kinetic energy removed | D | unit-test quantities | tests | `test_physics_columns.py` (I1); `test_boundary_layer.py` |
| NI-6 | a resumed run is byte-identical on the same platform; a cross-platform replay departs ≤ 10⁻⁹ relative or is refused | D (determinism); I (the 10⁻⁹) | state difference | tests; `mars_tides.py` replay record | `test_a_resumed_boundary_layer_run_is_identical`; the replay check |
| NI-7 | **energy tripwire**: \|window-mean atmosphere and whole-system residual\| ≤ 10⁻³ F and every daily value ≤ 10⁻² F; in a heated/unheated pair, unheated residual − measured sinks within 10⁻³ F | D principle; **I numbers**, inherited from A2/A3 (Earth). Kept because they are a demonstrated detector of one defect class (missing frictional heating: the unheated Earth run fails by a factor of 22). Labelled an engineering tripwire, not a scientific requirement and not an M-I property (Isca enforces closure). A bound derived for Mars and for the CV-1 run length is open (§4.11) | `energy_residual`, the system residual, the sinks | `series.json`, `summary.json` | `judge_mars_pilot.py` (M2–M4 logic, unchanged) |

**Diagnostic measurements (DM).** Reported for every run, with their
evidence file. **Never judged.** A DM becomes a requirement only by a dated
amendment that gives it a P or D basis.

| id | measurement | was | observable and evidence |
| --- | --- | --- | --- |
| DM-1 | winds aloft: daily grid-point max \|u\| with its level and latitude; zonal-mean u maximum and minimum with their level and latitude; the same restricted to p > 6 Pa | M1b | `max_abs_u` (`series.json`); zonal means from the saved states (audit Appendix A) |
| DM-2 | lowest-level winds: the daily max **\|u\|** and the daily max **horizontal speed**, both reported and labelled (§4.6) | M7 | `max_wind_bottom` (speed, `series.json`); \|u\| from the states |
| DM-3 | daily T extremes, against the historical 100–400 K | M1c | `t_min`, `t_max` |
| DM-4 | diurnal response: T_s S1 amplitude and peak hour **with the linearised expectation** arctan(C ω / λ_eff) beside it (§4.7); tropical p_s S1 and S2 amplitudes and phases; the RMS of the fit residual | M5 | `tides.json` |
| DM-5 | mass: relative drift of mean p_s; ⟨ln p_s⟩ and the variance of ln p_s; the fixer-equivalent mass factor | M6 | `series.json`; `state_diagnostics.json`; the judge's `fixer_equivalent` |
| DM-6 | energy: sinks by operator (sponge, hyperdiffusion, boundary layer); the time-integration remainder; the fixer-equivalent uniform temperature increment | M4 numbers, §2e report | `summary.json`, `series.json`, the verdict file's `reported` |
| DM-7 | cost and stiffness: s/step; k w/C_a; BL depth (max and median) | §2c–§2e reports | `events.jsonl`, `state_diagnostics.json`, `series.json` |

**Climate-validation target (CV).**

| id | target | origin | observable | evidence | verification |
| --- | --- | --- | --- | --- | --- |
| CV-1 | M-I's seasonal zonal-mean zonal wind, L_s 225°–315° | P (Fig. 4b and §4.2) | §4.4 | a future T42 multi-year run | §4.4: a qualitative feature comparison; numerical tolerances deliberately not set |

### 4.4 The published M-I comparison protocol

**What the paper specifies.**

- Configuration: Table 1, M-I column, plus §4.2 eqs. 14–15.
- The comparison quantity: Fig. 4 is the "zonal-mean zonal wind for Mars
  averaged over the solar longitude range L_S = 225°–315°", centred on the
  northern winter solstice. Panel 4b is the grey (M-I) model.
- Axes: latitude −90° to 90°; pressure on a log axis from 10⁻² hPa (1 Pa)
  to about 6 hPa.
- Colour scale: ±140 m s⁻¹, labelled every 35 m s⁻¹, with extension arrows.
- The paper's own evaluation is **qualitative** against MACDA (MY24–27).
  The zonal-mean wind is "now a reasonable match". The southern-hemisphere
  eastward jet "is somewhat stronger than in the reanalysis". The much
  stronger westward jet in the upper tropical atmosphere "now compares well
  with observations".

**What the paper does not specify.**

- the integration length or spin-up;
- which year, or how many years, were averaged;
- the output frequency;
- whether the zonal mean was formed on σ or on pressure surfaces;
- any uncertainty or agreement measure.

The Isca test case runs 64 × 30 sols, about 2.87 Mars years, with sol-mean
output. It is not known to be the paper's run (§4.10).

**Our protocol for CV-1** (prerequisites: IR-1 to IR-13 met, or each
departure labelled and argued):

1. T42L25, integrated at least as long as the test case: 1920 sols, from
   L_s = 0.
2. Sol-mean output. Average L_s 225°–315° over the last complete year,
   and report each available year separately to expose year-to-year
   variability.
3. Interpolate the zonal-mean u to pressure. Plot it on Fig. 4b's axes and
   colour scale, beside a crop of Fig. 4b.
4. Compare the features the paper itself describes:
   - a northern winter westerly jet near 60° N whose zonal mean exceeds the
     140 m s⁻¹ cap over roughly 1–100 Pa, as read from the figure;
   - a weaker southern-hemisphere eastward jet;
   - westward flow in the tropical upper atmosphere;
   - the latitude–pressure sign structure (the zero contour).

**No numerical agreement tolerance is set.** The reference is a contour
image with a saturating colour bar and no stated uncertainty, sampling or
averaging. A tolerance would need a defensible basis. Options, for a later
decision:

- the digitisation uncertainty of the figure, combined with our own
  year-to-year spread;
- **running Isca at `d1321ac3` with the grey-Mars namelist as the
  reference**, which gives a full field and Isca's own variability.

The second is the stronger reference, and the only one that would support
quantitative tolerances.

### 4.5 Remaining configuration differences (at `06f5356`)

| item | M-I (Table 1 / Isca `d1321ac3`) | the §2e runs | bearing |
| --- | --- | --- | --- |
| resolution | T42 | T21 (`l_max` 22, retained 21) | IR-9; CV-1 needs T42 |
| time step | `dt_atmos` **110 s** | 600 s | IR-10. Not listed in §1. The stiffness numbers of §2c and §2e (k w/C_a, stress and diffusion stiffness) scale with the step |
| time filter | Robert–Asselin 0.04, RAW off (`raw_filter_coeff` 1.0) | RAW ν 0.1, α 0.53 | IR-10 |
| hyperdiffusion | ∇⁴ (`damping_order` 2), coefficient 1.157 × 10⁻⁴ s⁻¹, `resolution_dependent` (scaling not read, §4.10) | ∇⁸, e-folding 0.1 day at the cut | IR-10; changes the sinks DM-6 reports |
| global fixers | mass and energy corrections every step; the energy correction also returns the ∇⁴ loss as uniform heat | none; residual and drift measured | IR-11. M-I's drift and dynamical residual are zero by construction, so NI-7 and DM-5 have no M-I counterpart |
| surface exchange | full Monin–Obukhov `mo_drag` (`rich_crit` 2.0, `stable_option` 1, `drag_min` 10⁻⁵, `zeta_trans` 0.5); `use_frierson_mo_drag` false | FHZ06-simplified coefficient (Ri_c 1), neutral | IR-8 |
| K profile | inner layer K from `mo_diff` (stability functions); taper above 0.1 h; zero above h | neutral κ u\* z below 0.1 h, the same taper | IR-8 |
| surface–diffusion coupling | one implicit down–up sweep with the slab between (`Tri_surf`) | split: implicit exchange (§2d), then diffusion with zero bottom heat flux (§2e) | IR-8 |
| dissipation of vertical diffusion | always returned as heat (`do_conserve_energy` hard-coded) | follows `frictional_heating`, so the U runs have no Isca counterpart | IR-8; NI-7 pairs |
| stress derivative | `old_dtaudv` True (the same −ρ C\|v\| for u and v) | implicit −ρ C\|v\|u for each component | comparable; not tested |
| `use_actual_surface_temperatures` | False | — | not examined (§4.10) |
| sponge pressure | local full-level p | σ × 610 Pa | IR-6 |
| cosine of the zenith angle | averaged over each 110 s step | instantaneous at the 600 s step | IR-3; small, not measured |
| vertical grid | Isca `bk`, 26 interfaces, `pk` = 0 | the same interfaces | met; full-level placement not compared |
| initial state | quiescent 264 K (any perturbation in `spectral_initialize_fields` not read); T_s 285 − 40(3 sin²φ − 1)/3 | 264 K plus a 0.1 m s⁻¹ seed; uniform 285 K | IR-12; matters only in spin-up |
| time unit | sols (segments of 30 × 88 440 s) | days of 86 400 s | bookkeeping only; L_s is computed exactly |

### 4.6 The M7 observable: a recorded discrepancy

§2e registered M7 as the "daily max lowest-full-level |u|". The judge at
`7dfc39d` and `33e818d` reads `max_wind_bottom`, which `experiment.py`
writes as the lowest-level horizontal **speed** ("an upper bound on |u|").
The discrepancy was not noticed before judging. Both measurements, days
11–60:

| run | registered (\|u\|), *recomputed* from the saved states | judged (speed), `verdicts-2e-corrected.json` |
| --- | --- | --- |
| Hb | 43.1 m s⁻¹ (day 57); above 40 on 6 days | 46.7 m s⁻¹ (day 27); above 40 on 13 days |
| Ub | 46.3 m s⁻¹ (day 47); above 40 on 2 days | 47.2 m s⁻¹ (day 47); above 40 on 4 days |

The historical verdict, **M7 not met**, is unchanged, and it is the same
under either observable. The §2e text and `verdicts-2e-corrected.json` stay
as they are; their numbers are the speed. The \|u\| values were computed
read-only from the saved states, which are local and untracked. The
computation is `tools/radiation/mars_wind_audit.py`; its outputs are
`wind_audit.json` in `evidence/dev-mars-gray-flat-bl-heated-T21L25-60d-06f5356-l4/`
(Hb) and `evidence/dev-mars-gray-flat-bl-T21L25-60d-06f5356-l4/` (Ub), whose
`summary` holds both columns above, and the audit's Appendix A describes the
method. From here on, DM-2 reports both, labelled, and judges neither.

### 4.7 The M5 slab phase: an analytical expectation, not a proof

The paper's §3.2 linearised slab gives a lag of φ ≈ arctan(C ω / λ). It
assumes one harmonic, linear restoring λ ≈ 4σT³ and fixed downwelling
radiation. For M-I's slab, C ω = 587 W m⁻² K⁻¹ while λ is about
2.4–5.3 W m⁻² K⁻¹, plus about 0.3–0.6 W m⁻² K⁻¹ of sensible exchange at
the measured winds. So φ ≈ 89.5–89.8°, and the S1 maximum is expected near
18 h. The measured 17.0–17.05 h is consistent.

A peak at 16 h would need λ ≥ 339 W m⁻² K⁻¹. That makes a pass very
unlikely for any faithful M-I surface, but it is not ruled out
mathematically:

- the model is nonlinear and coupled;
- M5 judges the peak of a fitted S1 + S2 curve;
- the downwelling longwave varies diurnally.

DM-4 reports the measured phase beside the expectation.

### 4.8 How criteria are written from now on

1. **Classify every criterion** as IR, NI, DM or CV. State its origin
   (P, Isca, D or I), its justification, the exact observable (the series
   key or function), the evidence file, the verification method, and what
   a failure would and would not establish.
2. **An I-origin number cannot be a pass/fail scientific requirement.** It
   may be:
   - a labelled engineering tripwire, with a demonstrated detection
     purpose (NI-7);
   - a prediction about an experiment, as in §2–§2e, whose verdict
     informs and does not certify.
3. **Check feasibility before registering.** Check each threshold against
   the configuration's own physics (analytic scalings, the published
   figure). Record the check.
4. **The judge computes exactly the registered observable.** A change of
   observable is a dated amendment made before judging. If one is found
   afterwards, both values are recorded, as in §4.6.
5. **Isca statements name the commit and file they were read at.** Where
   the source was not read, that is said.
6. **Inherited thresholds name their source and how it was set.** That
   includes whether it was written before or after the runs it describes.

### 4.9 Preserved records

No file below is modified by this amendment (SHA-256 prefixes of the
committed files, in `docs/radiation/evidence/`):

| file | SHA-256 |
| --- | --- |
| `dev-mars-gray-flat-heated-T21L25-30d-b3aaef8-l4/verdicts.json` | `1d3368e34c95f351` |
| `dev-mars-gray-flat-heated-T21L25-30d-b3aaef8-l4/tides.json` | `467e008f66ca7eed` |
| `dev-mars-gray-flat-sponge-heated-T21L25-30d-e38890e-l4/verdicts.json` | `bdcef4f5016bc142` |
| `dev-mars-gray-flat-sponge-heated-T21L25-30d-e38890e-l4/verdicts-last2sols.json` | `ce09e6f584970980` |
| `dev-mars-gray-flat-sponge-heated-T21L25-30d-e38890e-l4/tides.json` | `f7189db9f71a1179` |
| `dev-mars-gray-flat-sponge-heated-T21L25-30d-e38890e-l4/tides-last2sols.json` | `9161a34fed35e61e` |
| `dev-mars-gray-flat-sponge-implicit-heated-T21L25-60d-63d05a2-t4/verdicts.json` | `6d995d0685a61e30` |
| `dev-mars-gray-flat-sponge-implicit-heated-T21L25-60d-63d05a2-t4/tides.json` | `3b4f3130058a4d7e` |
| `dev-mars-gray-flat-bl-heated-T21L25-60d-06f5356-l4/verdicts-2e-recovery.json` | `bea7e011067bb6e6` |
| `dev-mars-gray-flat-bl-heated-T21L25-60d-06f5356-l4/verdicts-2e-corrected.json` | `6b5d6d475b7d3eba` |
| `dev-mars-gray-flat-bl-heated-T21L25-60d-06f5356-l4/tides.json` | `36788bf9b39d9c32` |

The judge, the tools and the tests that freeze these verdicts are
unchanged. `judge_mars_pilot.py` still writes the M-labelled verdicts. Its
output is read through §4.2, not as a requirement verdict.

### 4.10 Uncertainties left open

These were not verified at `d1321ac3`:

- the body of `monin_obukhov_drag_1d`, which `mo_drag` calls;
- the effect of `use_actual_surface_temperatures` False and of
  `vert_turb_driver` `do_simple`;
- any perturbation inside `spectral_initialize_fields`;
- the `resolution_dependent` scaling of `damping_coeff`;
- the full-level placement (`vert_difference_option`).

Also open: whether the test case is the run behind Fig. 4b. Its comment
("'dry' convection, 1.0 m") contradicts its namelist, and the paper states
no integration length.

### 4.11 Decisions this amendment leaves open

- **Mass.** Either a mass bound derived from the CV-1 integration (one T42
  run of about three Mars years), or an opt-in mass fixer whose per-step
  factor becomes the DM. §3 item 4 stands.
- **Energy.** A Mars-derived bound for NI-7 in place of the inherited
  10⁻³ F.
- **CV-1's reference and any tolerance.** The figure alone, or an Isca run
  at `d1321ac3` (§4.4).
- **Which IR-8 and IR-10 departures to close before CV-1, and which to
  argue.** The surface exchange, the joint sweep, the time step and the
  ∇⁴/RA numerics.

§3's ordering is unchanged. Its item 3 ("open candidates for M7") now
reads as closing IR-8, not as making M7 pass. Item 5 is CV-1.
