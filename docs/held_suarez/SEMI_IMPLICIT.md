# Semi-implicit leapfrog for the dry primitive equations (stage S3)

Status: implemented on `feat/held-suarez` (module `tropoi.temporal.semi_implicit`, tests
`tests/test_semi_implicit.py`). This document expands plan §3
(`docs/superpowers/plans/2026-09-27-held-suarez-24h-plan.md`) into the discrete algebra the code
actually executes, and records the S3 verification numbers. Hyperdiffusion, Rayleigh drag and
Newtonian relaxation (stage S4) enter through generic stepper hooks described in §8; with no
hooks the step is exactly the S3 step described in §§1–7.

## 1. State, operators, notation

Spectral state per level and spherical-harmonic mode (l, m):
ζ_k, δ_k, T_k (k = 1..K, top to bottom), q = ln p_s, packed as the PE stack
`(3K+1, l_max+1, l_max+1)` with rows `[ζ_1..ζ_K, δ_1..δ_K, T_1..T_K, q]` and axes (degree l,
order m). The horizontal Laplacian is diagonal: ∇² → −c_l with c_l = l(l+1)/a².

Discrete column operators reused verbatim from `tropoi.spatial.sigma_coordinate`:

| symbol | definition | code |
|---|---|---|
| **G** (K×K) | Simmons–Burridge hydrostatic matrix, Φ_full = Φ_s + G T | `hydrostatic_geopotential(sigma, I_K, 0, R)` column by column |
| **W** (K×K) | (ω/p)^lin: `omega_over_p` with G_k = δ_k and A_k = V·∇q = 0 | `omega_over_p(sigma, I_K, 0)` |
| **τ** (K×K) | τ = −κ T_ref W, so Ṫ\|_L = −τ δ = κ T_ref (ω/p)^lin(δ) | assembled |
| **ν** (K) | layer thickness Δσ_k, q̇\|_L = −νᵀ δ | `SigmaGrid.thickness_array()` |
| **B** (K×K) | B = G τ + R T_ref **1** νᵀ (**1** = all-ones column) | assembled |

No matrix entry is hand-derived: each operator is applied to the K unit vectors and the responses
form the columns. The reference atmosphere is isothermal (T_ref = 300 K, the initial state and
Dinosaur's choice, DEVLOG), resting, with a flat reference surface pressure; with T_ref constant the
vertical advection of the reference vanishes and the linear thermodynamic term is purely the
adiabatic conversion κ T_ref (ω/p).

## 2. The fast operator L

For each (l, m):

    δ̇ |_L = c_l ( G T + R T_ref 1 q )          (= −∇²(G T + R T_ref q))
    Ṫ |_L = −τ δ
    q̇ |_L = −νᵀ δ
    ζ̇ |_L = 0

**Dealiasing mask.** The nonlinear tendency analyzes every product on the dealiasing grid and zeroes
the result above the product cut `c = product_truncation_cut(l_max)` (2/3 rule; c = 14 at T21,
28 at T42) — or above the model's opt-in `retained_truncation` (production: 42 with l_max 43;
`SemiImplicitOperator.from_model` then uses it as `fast_cut`, DEALIASING_AUDIT.md). Three of the four fast terms reach the tendency through that pathway — R T_ref ∇q
through the weak-form pressure-gradient analysis, κ T_ref (ω/p) and −Σ Δσ_k G_k through the
thermodynamic and mass analyses — while −∇²(G T) is an exact diagonal spectral term. L reproduces
this: for l > c only the −∇²(G T) coupling is kept (`mask_l = 0` multiplies τ, ν and the R T_ref q
column). For states band-limited at the cut the mask is invisible; such states are the only
reachable ones (T and q have identically zero tendency above the cut, δ sees only −∇² G T of a zero
T′, ζ only truncated products), and the initial states used here are band-limited.

**Linearization property (tested).** On a non-rotating planet every linear term of the discrete
tendency about isothermal rest is in L, so for perturbations Y band-limited at the cut
‖tendency(X_rest + εY) − tendency(X_rest) − εLY‖ = O(ε²) block by block (ζ, δ, T, q). On the rotating
planet the residual is O(ε) in ζ and δ and equals the Coriolis linearization (the f-part of the
SWE-style pointwise expansions of div(ηV) and k·curl(ηV), analyzed and truncated as the tendency
does) to O(ε²); T and q stay O(ε²). Perturbations **above** the cut are excluded on purpose: the
weak-form curl of R T_ref ∇q′ is analytically zero but the dealiased analysis reproduces that only
inside the cut (measured at T21: |ζ̇| = 1.4e−10 for q′ with content above the cut vs 2.8e−24 for the
same q′ truncated at the cut). This is the same aliasing pathway that forces the terrain field to
be band-limited at the cut in the PE core.

## 3. The leapfrog step and the reduced solve

Three time levels n−1, n, n+1; overbar = ½(·^{n+1} + ·^{n−1}); N(X) = tendency(X) − L X.

    X^{n+1} = X^{n−1} + 2Δt [ N(X^n) + L X̄ ]

Writing T̄ = T^{n−1} + Δt (N_T − τ δ̄) and q̄ = q^{n−1} + Δt (N_q − νᵀ δ̄), substituting into the δ
equation and using δ^{n+1} = 2δ̄ − δ^{n−1} gives, per degree l,

    M_l δ̄ = δ^{n−1} + Δt [ N_δ + c_l G (T^{n−1} + Δt N_T) + c_l R T_ref 1 (q^{n−1} + Δt N_q) ]
    M_l = I + Δt² c_l mask_l B

then

    δ^{n+1} = 2 δ̄ − δ^{n−1}
    T^{n+1} = T^{n−1} + 2Δt ( N_T − mask_l τ δ̄ )
    q^{n+1} = q^{n−1} + 2Δt ( N_q − mask_l νᵀ δ̄ )
    ζ^{n+1} = ζ^{n−1} + 2Δt N_ζ

(the earlier "ν νᵀ" of plan r2 was wrong: q is one scalar per mode entering every level's pressure
gradient identically, hence **1** νᵀ). `SemiImplicitSolver` assembles M_l for all l once per Δt,
inverts each K×K block with LAPACK LU (`numpy.linalg.inv`, K ≤ 20), and applies the inverse as one
batched product over (l, m) on the device that holds the state (no host round trip per step). "Exact
per degree" is verified in two ways: the reduced solution inserted into the unreduced
(2K+1)×(2K+1) system (I − Δt L_l) X^{n+1} = (I + Δt L_l) X^{n−1} + 2Δt N has a residual ≤ 1.3e−15 of
the row scale for Δt ∈ {300, 1200, 3600} s (criterion 1e−12), and it agrees with a diagonally
balanced direct dense solve of that system to ≤ 1e−12 relative in every block. (The unbalanced
dense solve is the worse of the two: its matrix mixes entries of order Δt c_l G ≈ 1e−6 and
Δt τ ≈ 3e4, condition number ≈ 1e9, and at l = 0 it pivots on the τ block against the 1063 K
temperature monopole.)

**Pure linear scheme.** With N ≡ 0 the two-step map is the Cayley transform
C_l = (I − Δt L_l)⁻¹ (I + Δt L_l). Its eigenvalues are e^{±2i·atan(ω Δt)} for the K gravity-wave
frequencies ω_k = √(c_l λ_k(B)) (λ_k(B) real and positive: the squared phase speeds of the discrete
vertical modes) plus one stationary eigenvalue 1 (the (T, q) null space of c_l(G T + R T_ref 1 q)).
All moduli are 1 to 1e−12 (undamped scheme); the phase error 2 atan(ω Δt) − 2ω Δt is O((ω Δt)³),
i.e. the scheme slows fast gravity waves but never amplifies them. Tested on the assembled matrices
and, on the real model, by a 1e−10 s⁻¹ divergence perturbation of isothermal rest on a non-rotating
planet (RAW off): per degree X^{n+1} − C_l X^{n−1} is at the O(ε) nonlinear floor and no vorticity
is generated.

**Isothermal rest.** tendency(X_rest) = 0 bitwise (PE core property) and L X_rest = 0 bitwise
(c_0 = 0 kills the monopole coupling, δ = 0 kills the rest), so N = 0, δ̄ = M⁻¹·0 = 0 and every
update adds an exact +0.0: the state, both time levels, is preserved bit for bit through the RK4
startup, the SI steps and the RAW filter (tested, also with T_ref ≠ T).

## 4. RAW filter

After X^{n+1} is formed (and, since S4, multiplied by the implicit Rayleigh and ∇⁸ factors,
§8), Williams (2009) in the plan's normalisation:

    d = ν_raw ( X^{n−1}_f − 2 X^n + X^{n+1} )
    X^n_f   = X^n + α d
    X^{n+1} −= (1 − α) d

with ν_raw = 0.1, α = 0.53 (PROTOCOL; Williams' own convention writes d with ν/2, so this is his
ν = 0.2). ν_raw = 0 disables the filter exactly. The stepper's two levels are then
`x_prev = X^n_f` and `state = X^{n+1}` (unfiltered until the next step). The filter is tested
against this recurrence directly. The S3 convergence test runs with the filter off; the filtered
scheme is checked in S4b together with the forcing.

## 5. Startup and interface

`initialize(X0, t0)` stores X^0. The **first** `step()` is the RK4 startup X^0 → X^1 over one Δt in
n_sub substeps; every later `step()` is one leapfrog step. n_sub is the smallest integer for which
the RK4 stability polynomial satisfies |R₄(i ω_max Δt / n_sub)| ≤ 1 for the largest discrete
gravity-wave frequency ω_max = max_l √(c_l λ_max(B)) of the assembled operator (R₄ on the imaginary
axis is bounded by 1 up to 2√2 ≈ 2.83; the plan's "2.8"). This is measured from the operator, not
assumed from √(R T_ref): the fastest discrete mode travels at 337.9 m/s (L10) / 340.7 m/s (L20)
against the analytic Lamb speed √(γ R T_ref) = 347.2 m/s of the isothermal column; the slowest
internal mode at 2.6 / 0.6 m/s. Because L has no gravity waves above the dealiasing cut, ω_max is
reached at l = cut: at T42 L20 with Δt = 1200 s, ω_max Δt = 1.83, so n_sub = 1 (n_sub = 2 at
2400 s); at T21 L10, ω_max Δt = 0.92 at 1200 s. The code raises n_sub automatically whenever the
criterion fails.

`TimeStepper` protocol: `dt`, `initialize`, `step`, `state`, `t`, `step_count`, `state_dict`,
`load_state_dict`. `state_dict()` = `{scheme: "si_leapfrog", dt, t, step, raw_nu, raw_alpha,
t_ref, operator (the full signature of L: sigma interfaces, l_max, radius, R, c_p, T_ref, cut),
startup_substeps, x_prev (X^{n−1}_f or None before the first step), x_curr (X^n)}` with array
copies. `load_state_dict` refuses another scheme, a different Δt, filter coefficient, startup
substep count or operator signature (an independent review of the algebra found the original
check covered T_ref only), and a `step > 0` dict without `x_prev` (resuming would silently redo
the RK4 startup). The operator also validates B (real, positive eigenvalues) at construction. Resume is
bit-identical on the same software/hardware configuration (tested: save after 3 steps, continue 4
steps in both, compare bytes of both levels).

## 6. S3 verification list and evidence

Run: `pytest tests/test_semi_implicit.py -q -s` (CPU part in CI; GPU part local, MX110). An
independent read-only review (Sonnet subagent, 2026-09-27) re-derived the reduced system, the
Cayley spectrum, the RAW recurrence and the startup rule from the code and confirmed each; its one
code finding (the resume check above) is fixed and tested.

| plan §3 item | test | result |
|---|---|---|
| (i) L equals the fast-term function from the column operators | `test_operator_matches_fast_terms_from_column_operators`, `..._on_device_...` | rel. diff < 1e−13 (CPU and CuPy) |
| dense L_l equals `apply` on unit vectors; l = 0 has no pressure term | `test_operator_dense_matrix_reproduces_apply_on_unit_vectors` | PASS |
| B eigenvalues real, positive, hydrostatic scale | `test_operator_gravity_wave_speeds_...` | PASS; c_max = 337.9 m/s at L10 (Lamb 347.2), see §5 |
| (ii) Ω = 0 residual O(ε²) | `test_linearization_residual_is_second_order_on_nonrotating_planet` | ratios ε→ε/2 (T21 L6, ε = 1e−2, 5e−3): ζ 4.001, δ 4.007, T 3.999, q 4.000; residual/linear term: δ 8e−4, T 1.0e−2, q 5e−3 |
| (iii) Ω ≠ 0 residual O(ε), equals Coriolis to O(ε²) | `test_linearization_residual_on_rotating_planet_is_the_coriolis_coupling` | residual ratios ζ 1.998, δ 2.003 (O(ε)); residual − Coriolis ratios ζ 4.001, δ 4.007, size 1.4 % / 1.6 % of the Coriolis term; T, q ratios 3.999, 4.000 |
| reduced vs unreduced solve | `test_reduced_solve_matches_unreduced_coupled_system[300/1200/3600]` | residual 1.08e−15 / 9.46e−16 / 1.28e−15 (criterion 1e−12) |
| matrices assembled, not hand-derived | `test_solver_matrices_are_assembled_from_operators_not_hand_derived` | PASS |
| analytic amplification, modulus 1, dispersion | `test_cayley_amplification_has_modulus_one_and_si_dispersion` | PASS (|λ| = 1 ± 1e−12) |
| stepper = Cayley on the pure linear system | `test_stepper_on_pure_linear_tendency_is_exactly_the_cayley_transform` | PASS (1e−13) |
| rest exact | `test_isothermal_rest_is_preserved_bitwise` | bytes identical after 4 SI steps (RAW on, T_ref = T) and 2 steps (T_ref = 250 K ≠ T), both time levels |
| linear gravity waves on the real model | `test_linear_gravity_waves_follow_the_si_amplification_factor` | T21 L6, Ω = 0, δ′ = 1e−10 s⁻¹ at 5 modes, Δt = 1800 s, 8 steps: X^{n+1} − C_l X^{n−1} = 1.6e−7 relative (nonlinear floor); modal amplitude drift 4.6e−11 (modulus 1); vorticity generated 4.5e−16 s⁻¹ (quadratic σ̇∂V/∂σ) |
| 2-day T21 L10 SI vs RK4, Δt, Δt/2, Δt/4 | `test_si_converges_second_order_to_rk4_on_t21_l10` | thermal wave 5 K at 300 K, RAW off, RK4 reference Δt = 300 s: relative error 1.064e−1 (2400 s), 2.446e−2 (1200 s), 5.696e−3 (600 s); ratios **4.349, 4.295** (second order; slightly above 4 because 2 atan(ωΔt) is not yet asymptotic at ωΔt ≈ 0.9 for the fastest excited modes at 2400 s). MX110 wall time: SI 9 / 20 / 37 s, RK4 reference 264 s |
| RAW recurrence | `test_raw_filter_follows_plan_recurrence` | PASS |
| both time levels serialized | `test_state_dict_serializes_both_time_levels_and_resumes_bit_identically`, `test_state_dict_before_first_step_holds_only_x0` | PASS (bit-identical resume) |
| protocol + startup substeps | `test_stepper_satisfies_timestepper_protocol_and_startup_substeps` | PASS |

## 7. Known limits and risks carried into S4

- The damping factors S4 will add are backward-Euler (1/(1+2Δt k)), not exponential; documented in
  the plan, not claimed exact in time.
- The SI treatment assumes T_ref ≥ the actual temperature in the sense of Simmons–Hoskins–Burridge;
  with T_ref = 300 K and the HS T_eq ≤ 315 K the tropical lower troposphere may exceed T_ref by
  ≤ 15 K. The classical result is that the scheme stays stable for modest excess; S4b's 5-day
  stability run at Δt ≥ 900 s is the measurement.
- The explicit part (Coriolis, advection, and in S4 the Newtonian relaxation) is leapfrog-stable
  for f Δt < 1 and U Δt (l_max/a) < 1; at Δt = 1200 s these are 0.17 and ≈ 0.25 for U = 30 m/s at
  T42, so advection, not gravity waves, will set the production Δt.
- Perturbations above the dealiasing cut are outside the verified linearization (see §2); the S5
  experiment must seed its perturbation inside the cut (PROTOCOL: degrees 1–8).

## 8. Held–Suarez forcing and dissipation in the step (stage S4)

Generic hooks (`tropoi.temporal.hooks`; the stepper knows nothing about HS):

- **explicit terms** `term(X) -> dX/dt` are added to N(X^n):
  N = tendency(X^n) + Σ term(X^n) − L X^n (then the reduced solve of §3, unchanged);
- **`DiagonalDamping`** (rate r per row and degree) multiplies X^{n+1} by the backward-Euler
  factor 1/(1 + 2Δt r) **after** `SemiImplicitSolver.advance` and **before** the RAW filter,
  in the order given; its explicit form −r X is used by explicit schemes;
- the RK4 startup integrates the **complete** right-hand side (tendency + terms + −r X). Its
  substep count keeps |R₄(z h)| ≤ 1 for z = −r_l + iω at every degree (ω the gravity-wave
  frequencies of L, r_l the summed damping rates at l plus the terms' `max_rate`); with no hooks
  this is the §5 rule exactly;
- with no hooks the step performs exactly the S3 operations (tested bitwise); hook signatures are
  in `state_dict()` and a mismatch refuses to resume.

Held–Suarez instances (`tropoi.temporal.tendencies.held_suarez`; PROTOCOL §1.1):

| term | where | discretisation |
|---|---|---|
| Newtonian relaxation −k_T(φ,σ_k)(T_k − T_eq(φ, σ_k p_s)) | explicit term (N) | evaluated on the product grid with the instantaneous p_s, analyzed, truncated at the product cut like every analyzed product (so band-limited states stay band-limited and L stays the exact linearization) |
| Rayleigh drag −k_v(σ_k) V | damper on ζ_k, δ_k (all degrees) | ζ, δ ×= 1/(1 + 2Δt k_v,k); exact equivalence because k_v is horizontally uniform |
| ∇⁸ on ζ, δ, T | damper, per degree | ×= 1/(1 + 2Δt K₈ c_l⁴), K₈ = 1/(τ₈ c_ref⁴), τ₈ = 0.1 d, ref = l_max by default (PROTOCOL §2); c_0 = 0 so the global-mean T (T′ only) is untouched; ln p_s not diffused |

No other numerical treatment is applied (no divergence damping, clipping or extra filtering).
The factors are backward Euler, not exponential. More important than the factor's form is its
placement: multiplying X^{n+1} evaluates the damping at n+1 instead of centred at n, a local error
−rΔt Ẋ, so the damped part of the scheme is **first order** in time (an exponential factor has
the same leading term: S4b measured identical 2-day errors to 3 digits). Together with RAW at
α = 0.53 (a small first-order term of its own), the complete scheme converges at first order
asymptotically; without the dampers and RAW it is second order (S4b (b), DEVLOG). A centred
treatment would have to enter the reduced SI system; this is a pre-launch decision (STATUS.md).

S4 evidence (`tests/test_held_suarez_forcing.py`): T_eq, k_T, k_v against Wolfram-computed hand
values at 11 points including σ = σ_b, σ = 1, both poles and both sides of the 200 K floor (worst
relative difference 2.1e−16); forcing exactly 0 at T = T_eq, V = 0; pointwise relaxation follows
exp(−k_T t) to 2e−14 over 10 days (rate/k_T = 1 ± 1e−12); the spectral hook decays at k_a where k_T
is uniform (4e−15) and matches an independent product-grid evaluation with non-uniform p_s
(1.5e−15); drag and ∇⁸ factors equal 1/(1 + 2Δt k) degree by degree (0 ulp for ∇⁸); isothermal rest
is preserved bitwise through startup + SI + drag + ∇⁸ + RAW; a resting, horizontally uniform T_eq
profile stays at rest under the full forcing (|ζ, δ| ≤ 2.6e−21 s⁻¹, |ΔT| ≤ 1.1e−14 K after 1 day).

## 9. Centred damping (opt-in, `damping_scheme="centred"`; 2026-09-28)

Motivation: §8's placement (factor on X^{n+1}) evaluates the damping at n+1, local error −rΔt Ẋ,
and the complete scheme converges at first order (S4b). The option keeps every S3/S4 operation
when off (the default `"lagged"` is untouched, tested bitwise against the sequential-factor path)
and, when on, takes −R X trapezoidally over n−1, n+1, i.e. the implicit operator is L − R with
R = Σ dampers' rates (diagonal per row and degree; the dampers' `apply_implicit` is not called).

**Reduced solve.** With D_T = (I + Δt R_T)⁻¹ (per degree; ∇⁸ rates vary with l) and
d_q = 1/(1 + Δt r_q) (0 for HS: ln p_s is not diffused), T̄ = D_T (T* − Δt m τ δ̄),
q̄ = d_q (q* − Δt m νᵀ δ̄) with T* = T^{n−1} + Δt N_T, q* = q^{n−1} + Δt N_q, and

    M_l δ̄ = δ^{n−1} + Δt [ N_δ + c_l G D_T T* + c_l R T_ref m d_q 1 q* ]
    M_l = I + Δt R_δ + Δt² c_l m_l ( G D_T τ + R T_ref d_q 1 νᵀ )

then δ^{n+1} = 2δ̄ − δ^{n−1} and, for the rows outside the δ block,

    X^{n+1} = X^{n−1} + 2Δt (N_X − [L-coupling] − r X^{n−1}) / (1 + Δt r)

(ζ: N_ζ only; T: −m τ δ̄; q: −m νᵀ δ̄). With R = 0 this is §3 exactly. Nothing is hand-derived:
M_l is `I + Δt diag(r_δ) + Δt² c_l m_l (G @ D_T @ τ + R T_ref d_q 1 νᵀ)` from the same G, τ, ν.
Verified (`tests/test_centred_damping.py`): the reduced solution inserted into the unreduced
(2K+1)×(2K+1) system (I − Δt(L_l − R_l)) X^{n+1} = (I + Δt(L_l − R_l)) X^{n−1} + 2Δt N has residual
1.04e−15 / 1.24e−15 / 1.60e−15 of the row scale at Δt = 300 / 1200 / 3600 s (criterion 1e−12) and
agrees with a diagonally balanced dense solve to ≤ 1e−12 in every block; M_l equals the formula
assembled from the operators and the rate tables; the Cayley matrix (I − Δt A)⁻¹(I + Δt A),
A = L − R, has all moduli ≤ 1 (strictly < 1 for every coupled mode inside the cut; above the cut
the frozen, undiffused ln p_s keeps exactly one eigenvalue 1).

**Scalar recurrence.** For a ζ row the option is exactly the 2×2 map with the damping in the
trapezoidal coefficient: (1 + Δt r) x^{n+1} = (1 − Δt r) x^{n−1} + 2Δt(−k + iω) x^n, then RAW
(tied to the stepper to 1e−13). Pure damping: λ² = (1 − Δt r)/(1 + Δt r), both leapfrog modes
damped, no sign alternation while Δt r < 1 (∇⁸ at l = 42: Δt r = 0.104 at 900 s; k_f: 0.0104).
On x′ = −k_f x + sin(Wt) (2 days, RAW off) the error ratios are 4.0017, 4.0004, 4.0001 (lagged:
1.875, 1.939, 1.970); the 1200 s error is 8× smaller than lagged.

**Unchanged.** RK4 startup (the complete right-hand side and the substep rule read the rates, not
the placement — identical counts), the RAW recurrence, the spectral mask, the two-level restart
state, the operator/hook signature validation. `state_dict()` gains `damping_scheme`; a dict
without the key (written before the option existed) loads as `"lagged"`; a mismatch refuses.
Isothermal rest is preserved bitwise (every factor is exactly 1.0 and every subtracted term
exactly 0.0 on the rest rows). Config: `HeldSuarezConfig.damping_scheme` (default `"lagged"`; the
canonical JSON, hence `config_sha256`, changes for every preset by the added field).

The measured stability bound and the T21 L10 convergence of the option are in DEVLOG (2026-09-28).
