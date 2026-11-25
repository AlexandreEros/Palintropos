# Operation Planet One — frozen acceptance register

**Frozen:** 2026-11-24, before any Planet One numerical result was produced or inspected.
**Baseline:** `substrate-cpu-verified-v1` (`ce4015d`).
**Owner approval:** the plan was approved with two rounds of corrections on 2026-11-24.

The criteria below may be changed only with explicit owner approval, recorded here as a dated amendment that gives its reason. Results that fail a gate are reported as failures. They are never re-labelled after the fact.

## Evidence levels

Every reported result carries exactly one of these levels:

| Level | Meaning |
|---|---|
| **L1** | Numerical verification: analytic, manufactured, conservation, convergence |
| **L2** | Code-to-code agreement with an independent implementation (heat1d, pinned) under matched inputs |
| **L3** | Reproduction of a published Mercury thermophysical model (Vasavada, Paige & Wood 1999), with every coefficient source-verified |
| **L4** | Observational validation |

- Any level whose reference data or source is unavailable is reported as **NOT EVALUATED**.
- A run using the Hayne et al. (2017) lunar-regolith analogue can reach L1 and L2 only. It never counts as L3.
- L4 is not claimed by Planet One.

## Conventions

### Depth, fluxes and geothermal heat

- Depth z points downward.
- The conductive flux is q↓ = −k ∂T/∂z.
- At the bottom, q↓(D) = −G, where G > 0 is the upward geothermal flux.
- At the surface (the diagnosed, massless skin):

  (1−A) F_inc = εσT_s⁴ + q_s

### Orbit and spin frames

**Inertial frame I.** A right-handed frame whose x̂ is the reference direction and ẑ the reference pole.

**Orbit.** With ω = ϖ − Ω,

  r_I = R_z(Ω) R_x(i) R_z(ω) r_pf(ν).

**Spin.** The spin is described by its angular-velocity vector **Ω⃗ = Ω ê** with Ω > 0:

- Body north is ẑ_B = ê.
- Longitudes increase toward the east, and east is the direction of rotation.
- The obliquity ε ∈ [0, π] is the angle between ê and the orbit normal. ε > 90° means the spin is retrograde.
- There is no signed spin rate.

**Body-to-inertial rotation.**

  R_BI(t) = R_z(Ω_node) R_x(i) · R_z(λ_eq) R_x(ε) · R_z(W₀ + Ω (t − t₀))

- λ_eq is the angle, measured in the orbit plane from the orbit's ascending node, to the body equator's ascending node on the orbit plane.
- W₀ is the angle of the prime meridian from that node at the epoch t₀.

**Star direction in the body frame.**

  ŝ_B = R_BIᵀ (r_src − r_p) / |r_src − r_p|

**Time.** All time dependence is evaluated from absolute time; no phase is accumulated step by step.

### Recurrence

- Illumination recurs exactly when Ω/n is rational.
- The quantity 2π/|Ω − n cos ε| is called the **nominal** mean solar day. It is exact only for circular orbits with ε ∈ {0, π}.

## G-A Illumination geometry (L1)

**Kepler propagation and orbit**

1. The Kepler-equation residual is ≤ 1e-14 rad.
2. Each of the following holds to ≤ 1e-14 relative:
   - the periapsis and apoapsis distances;
   - |ŝ| = 1;
   - F·r² is constant.
3. Orbit elements:
   - the ascending node lies at the stated Ω;
   - the periapsis direction is at the stated ϖ;
   - for i < 90° the orbital angular momentum has a positive z-component.

**Spin, frames and orientation**

4. Fixed subsolar point. For a circular orbit, ε = 0 and prograde 1:1 spin, the subsolar point stays fixed to within 1e-15·(1 + |n t|) rad over 10³ orbits, and the irradiance is constant.
5. Oblique synchronous circular orbit:
   - the subsolar latitude equals asin(sin ε sin L_s(t)) to ≤ 1e-13, including its sign (northern summer at L_s = 90°);
   - the subsolar longitude matches an independent analytic vector expression to ≤ 1e-13.
6. Frame invariance. Applying a common arbitrary inertial rotation to the orbit and the spin-axis description changes the body-frame illumination by ≤ 1e-13.
7. Recurrence:
   - commensurate configurations, including oblique eccentric 3:2, recur exactly after their common period, to ≤ 1e-12 rad in subsolar coordinates;
   - for circular orbits with ε ∈ {0, π}, the subsolar longitude has the analytic rate Ω − n cos ε. This covers sign and magnitude, and the retrograde case.

**Multiple sources**

8. The total irradiance equals the sum of the single-source irradiances. A source below the local horizon contributes zero.

**Mercury geometry (3:2, ε = 0, perihelion at t = 0)**

9. The subsolar longitude at successive perihelia alternates between 0° and 180°. At aphelia it is 90° or 270°. Tolerance: 1e-12 rad.
10. The illumination repeats after one solar day (two orbits) to 1e-12.
11. The endpoints of the apparent-retrograde interval satisfy

   (1 + e cos ν)² = (Ω/n)(1 − e²)^{3/2}

   to within 1e-6·P_orb.
12. Horizon-crossing counts at the warm longitudes are reported. They come from the geometry alone.

**Diurnal mean**

13. At a frozen orbital position, the time mean of the instantaneous insolation over one rotation equals `DiurnalMeanInsolation` to ≤ 1e-3·S.

## G-B Nonlinear conduction (L1)

### Residuals

The per-cell residual and the skin residual are both in W m⁻²:

  r_i = ρ_iΔz_i Δh_i/Δt − (F_{i−½} − F_{i+½}) − S_i
  r_s = (1−A)F_inc − εσT_s⁴ − q_s

where

- F is the downward face flux, with F_{½} = q_s and F_{L+½} = −G;
- Δh_i is evaluated by cancellation-free divided differences.

### Newton convergence

Each step requires

- |r_i| ≤ τ_i = 1e-9 + 64 ε_mach · (|ρ_iΔz_iΔh_i/Δt| + |F_{i−½}| + |F_{i+½}| + |S_i|) W m⁻², and
- |r_s| ≤ τ_s = 1e-9 + 64 ε_mach · ((1−A)F_inc + εσT_s⁴ + |q_s|) W m⁻².

The 1e-9 W m⁻² floor is 5e-8 of G = 0.02 W m⁻². These tolerances are **not** guaranteed to be attainable. If they are not met within the iteration cap, or the iteration stagnates, the step raises an error with diagnostics, and that counts as a gate failure.

### Energy identity

  ΔE/Δt − [(1−A)F_inc − εσT_s⁴ + G + ΣS] = Σ r_i − r_s

where ΔE = Σ ρΔz Δh.

- **Per-step bound:** |·| ≤ Σ τ_i + τ_s + δ_sum, where δ_sum is the round-off of the sums.
- **Cumulative bound:** the sum over steps of Δt·(the per-step bound), with the sums evaluated exactly (Shewchuk).
- Both bounds are checked at every step.

### Skin flux

The skin flux is Kirchhoff-exact within the top half-cell:

  q_s = [Φ₁(T_s) − Φ₁(T₁)]/(Δz₁/2),  Φ₁ = ∫k₁ dT

- Jacobian: ∂q_s/∂T_s = 2k₁(T_s)/Δz₁ and ∂q_s/∂T₁ = −2k₁(T₁)/Δz₁.
- The same q_s is used in r_s, in the cell-1 balance and in every diagnostic.

### Interior faces

  G_f = [Δz_i/(2k_i(T_i)) + Δz_{i+1}/(2k_{i+1}(T_{i+1}))]⁻¹

- Material interfaces must coincide with faces.
- The Jacobian is exact and includes dG_f/dT.

### Checks

1. **Jacobian.** The analytic Jacobian matches central finite differences to ≤ 1e-6 relative, on random heterogeneous states, including the stratum faces and the skin.
2. **Manufactured nonlinear transient.** Temperature-dependent k and c_p, two strata, a radiative top and geothermal G. The observed order must be in [0.9, 1.1] in time and [1.8, 2.2] in space on smoothly stretched grids. Both orders include the surface temperature.
3. **Steady Kirchhoff solution.**
   - Within each stratum m: Φ_m(T(z)) = Φ_m(T(z_m)) + G (z − z_m), with T continuous at interfaces.
   - At the surface: εσT_s⁴ = (1−A)F + G.
   - The maximum error converges at second order, and the interface flux is continuous to round-off.
4. **Zero-conductivity limit (G = 0).** As k → 0, T_s tends monotonically to [(1−A)F/(εσ)]^{1/4}. The skin equation with q_s ≡ 0 is solved to ≤ 1e-12 relative.
5. **Constant-property limit, linear boundary conditions only** (prescribed or linearised flux, Robin). The solve takes one Newton iteration and agrees with the closed-form linear step to ≤ 1e-12 relative. With the Stefan–Boltzmann boundary, the constant-property case is covered by checks 2 and 3 instead.
6. **Stress.** A column starting at 40 K is stepped to 14.5 kW m⁻². T stays positive, the solve converges within ≤ 30 iterations, and failures report the indices of the failing columns.
7. **Energy closure.** The per-step and cumulative bounds above hold under random forcing.
8. **Validity ranges.** Each material law declares the valid temperature range stated by its source, and every run records the extent it reached in each stratum. Going outside a law's range is shown as **EXTRAPOLATED** on every output and label. An L3 claim requires that any extrapolation matches the one the source itself used.

## G-R Regression of the original substrate gates (L1, original tolerances)

| Gate | Criterion | Status |
|---|---|---|
| G1M periodic | amplitude ≤ 1 %, phase ≤ 0.01 P | kept |
| G1C | convergence order | kept |
| G2 erfc | ≤ 1 % | kept |
| G3-D Robin decay | ≤ 1 % | kept |
| G4 one layer vs slab | round-off bound | kept |
| G5 energy, linear flux | ≤ 1e-12 cumulative | kept |
| G5b skin study | diagnosed arm | kept |
| G3-M and the G5b layer arm | — | **removed**, together with the `layer` skin mode |

**Why the layer mode was removed.** The owner chose the diagnosed skin (`gates-m1-postdecision/OWNER-DECISION-SKIN.md`). The O(Δz₁) bias of identifying the layer mean with the surface temperature is recorded in the frozen G5b evidence.

Bitwise agreement with the archived gate outputs is reported for information. It is not required.

## G-P Mercury columns: convergence and periodicity (L1)

**Columns.** Equator at 0°, 90°, 180° and 270°E, plus 45°N and 85°N.

| Quantity | Tolerance | Justification |
|---|---|---|
| Surface T_max, T_min | ≤ 0.5 K | far below the ~5 K digitisation uncertainty of published curves and the ~130 K hot/warm contrast |
| Time of T_max | ≤ 2e-3 P_solar | ≈ 0.7° of mean solar motion |
| T at 2, 10, 30 cm (cycle mean and amplitude); bottom T | ≤ 0.2 K | 0.2 K is ≈ 10 % of the geothermal temperature offset G·L/k ≈ 2 K |
| Cycle-mean εσT_s⁴ and cycle-mean \|q_s\| | ≤ 1e-3 relative | 1e-3 in εσT⁴ is ≈ 0.25 K at 700 K, consistent with the T tolerance; \|q_s\| measures subsurface storage |

**Refinement.**

- Δt is halved three times, the number of layers doubled three times, and the depth doubled once.
- The production setting passes when, for every quantity, the Richardson-estimated error is within tolerance and doubling the depth changes the quantity by less than its tolerance.
- Observed convergence orders are reported but not required, because the terminator is not smooth.

**G-P2 Periodicity, at the production Δt.** Between the last two solar days:

- max |T_k − T_{k−1}| over all depths and phases ≤ 0.2 K (surface: 0.5 K);
- bottom drift ≤ 0.01 K per cycle;
- **resolution of the geothermal flux:** |ΔE|/P ≤ 1e-3 W m⁻², i.e. 5 % of G. In a periodic state, cycle-mean(εσT_s⁴ − (1−A)F) − G = −ΔE/P.

**Spin-up.**

- The decay of the cycle-to-cycle difference is fitted, and its e-folding time is reported next to the prediction 4L²/(π²κ_eff).
- The spin-up length is set by this monitoring.
- Coarse-Δt spin-up is allowed, but the periodic claim requires at least 2 consecutive production-Δt cycles that meet the criteria.
- Otherwise the result is labelled **TRANSIENT (cycle k)**.

## G-I Independent comparison (L2)

- **Reference:** heat1d, pinned at `phayne/heat1d@5b73a6c1ba5047f59f6210185b37539e46e68136`, run with the same absorbed-flux series, laws, strata, A, ε and G as our model.
- **Criterion:** the difference shrinks under joint refinement, and at the end

  max |ΔT_s| ≤ e_ours + e_heat1d + 0.2 K,

  where each e is that code's own Richardson error estimate.
- Unremovable formulation differences (for example node-based versus cell-based property placement) are listed explicitly.

## G-M Mercury results

**Structural checks (L1)**

- After a one-orbit shift, the 0° and 180° climatologies match, and so do 90° and 270°, within the G-P tolerances.
- The hot-meridian T_max exceeds the warm-meridian T_max.

**L3** requires source-verified VPW99:

- a table of values with their provenance (text or digitised figure) and uncertainty;
- differences larger than 10 K are investigated and reported;
- **no parameter changes.**

Without the source, L3 is **NOT EVALUATED**.

**L4: NOT EVALUATED.** At most there is a qualitative note on the Mariner 10 nightside, with its caveats.

## G-G Global (L1)

- Global columns agree with standalone runs to ≤ 1e-12 K when both use identical float coordinates, forcing histories, initial conditions and numerical settings.
- Global energy closure is computed with cell weights w = R²Δλ(sin φ_N − sin φ_S), and the per-column bounds are area-weighted.
- One complete solar day is retained after G-P2 passes; otherwise the output is labelled TRANSIENT.

## Output tiers

1. **Every step.** Per-column running cycle integrals and area-weighted global scalars: E, net flux, max |r|, iterations.
2. **Scientific snapshots, final solar day, float64.**
   - surface fields at P/720;
   - full T(z) at P/180;
   - every step for the canonical columns;
   - restart state at each cycle end.
3. **Visualisation.** Downsampled from tier 2 only, with the quantisation stated.

## Source status at freeze

The VPW99 primary text was **not obtained** before the freeze:

- the author-hosted copy has a broken TLS chain;
- other copies were unreachable.

Unverified transcriptions from the feasibility study are not used as source-verified coefficients.

## Amendments

### A1 (2026-11-24, approved by the owner): G-P depth criterion

**Original frozen criterion.** Doubling the depth must change the cycle-mean
bottom temperature by ≤ 0.2 K.

**Result.** FAIL in all six canonical columns. The changes were 1.84–2.15 K
(VPW99 TWO, G = 0.02 W m⁻²; preserved in
`planet-one-evidence/gp-vpw99-two-G0.02-pre-amendment.json`).

**Why the criterion was ill-posed.** The two bottom cells lie at different
physical depths, about 1 m and 2 m. With G > 0 the temperature rises by
about G·1 m/k ≈ 2 K between them, which is physics, not discretization
error.

**Amended criterion.** The cycle-mean temperature at the same physical
depth, 1 m (the shallow column's bottom boundary), must change by ≤ 0.2 K
under depth doubling. The tolerance is unchanged.

- For the shallow column, the temperature at 1 m is its bottom-face value.
  That value is reconstructed Kirchhoff-exactly from the prescribed
  geothermal flux across the bottom half-cell, never clamped to the last
  cell centre.
- For the deep column, the temperature at 1 m comes from its
  finite-volume reconstruction.

All other convergence, periodicity, energy-closure and independent-comparison
criteria are unchanged.

### E1 (2026-11-24, approved by the owner): diagnostic estimator correction

**Problem.** Subsurface temperatures at fixed depths had been interpolated
linearly between cell centres. Across the 2 cm material interface this is a
first-order estimator (observed order 0.9).

**Correction.** Use the finite-volume solution's own reconstruction: cell
centres plus face temperatures. Interior faces use the flux-continuous
value with the solver's half-cell conductances. The surface uses the skin.
The bottom face uses the reconstruction described in A1.

**Tolerances.** Unchanged. Both estimators are kept in the evidence
(`*_centre_interp`).

**Result.** G-P (VPW99 TWO, G = 0.02) passes for every quantity in all six
columns, with every ladder run PERIODIC
(`planet-one-evidence/gp-vpw99-two-G0.02-amended.json`).

### A2 (2026-11-24, owner-directed): added GPU-parity gate G-GPU

This gate was frozen before any GPU result existed. It adds a requirement;
it relaxes nothing.

One mathematical implementation runs on two backends (NumPy and CuPy). It
stays FP64.

Parity is not required to be bitwise. The Newton iteration stops once
|r| ≤ τ (floor 1e-9 W m⁻²), and the smallest Jacobian diagonal is about
C/Δt ≈ 0.2 W m⁻² K⁻¹ (top cell, production Δt). Each step's state is
therefore determined only to about 5e-9 K, and the backends may stop at
different iterations.

1. **Mercury columns.** Run the six canonical columns (VPW99 TWO, G = 0.02,
   production settings) for one full production cycle from an identical
   state on both backends. Require max |T_GPU − T_CPU| ≤ 1e-6 K over every
   cell, the skin and every step. That bound is 2e5 times below the 0.2 K
   G-P tolerance and leaves room for accumulation of the ~5e-9 K per-step
   indeterminacy. Iteration-count differences are reported.
2. **Energy closure on the GPU.** The G-B identity and its per-step and
   cumulative bounds hold under random forcing, with both evaluated on the
   device.
3. **Manufactured solution on the GPU.** Run the two-stratum transient at two
   refinement levels. The errors must equal the CPU errors to ≤ 1e-9
   relative, so the convergence orders are unchanged.
4. **Applicable original regressions.** On the GPU, G4 (one layer vs slab,
   round-off) and G5 (energy, ≤ 1e-12 cumulative) must hold. Ben 2's G7
   bound is not used here, for the reason given above.
5. **G-G for a GPU global run.** Its standalone columns run on the same
   backend. CPU/GPU agreement is covered by item 1.

**Profiling first.** The backend-generic tridiagonal solver is profiled on
the GPU before any specialized CUDA kernel is considered.


### Erratum X1 (2026-11-25): nominal mean solar day

The conventions section defines the "nominal" mean solar day as
2π/|Ω − n cos ε|. The long-term mean solar day is actually
2π/|Ω − n sign(cos ε)|: over one orbit the star's right ascension advances
one full turn, whatever the obliquity. The code
(`StellarIllumination.mean_solar_day`) uses the correct form. The two
agree for ε ∈ {0, π}, which are the only cases the G-A gate 7 tests. No gate
or result changes.

### Record R1 (2026-11-25): G-I evaluation

G-I was first evaluated with all three levels sampled at the coarsest
level's times, a rule chosen after the levels were seen. On that sampling
both columns passed. The frozen wording is "max |ΔT_s|", so the strict
evaluation now takes the finest level's own samples. Its bound uses error
estimates on the densest time set where both codes have two levels (the
level-1 times).

| Column | Finest-level max | Bound | Verdict |
|---|---|---|---|
| 0°N 90°E | 1.36 K | 1.84 K | PASS |
| 0°N 0°E | 2.40 K, at the first sunrise step | 2.26 K | **FAIL** |

At 0°N 0°E the native maxima are 3.75, 4.10 and 2.40 K, which do not shrink
monotonically, while the RMS halves at each level (2.17, 1.12, 0.59 K). The
failure is reported to the owner for a methodological decision. The
criterion is not changed here.
