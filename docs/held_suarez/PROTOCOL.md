# Held–Suarez protocol for Palintropos

Status: **development (not frozen)**. This file is frozen at the launch of the first
production experiment (its sha256 is then recorded in the checkpoint and in STATUS.md).
Until then, every change is listed in DEVLOG.md. After the freeze, any change requires a
user decision and applies only to experiments launched afterwards.

## 1. Source: Held & Suarez (1994), BAMS 75, 1825–1830

Text obtained 2026-09-27 from GFDL,
<https://www.gfdl.noaa.gov/bibliography/related_files/ih9401.pdf>
(sha256 `a2a821fb9fb183968203e7ce603d94704d1d41b8dbf81b171bab4d6ee825c7e4`, 6 pages; not
committed — copyright AMS). Transcribed from the boxed specification on p. 1826 and
§3 (pp. 1828–1829):

### 1.1 Forcing (prescribed by the paper; not a modelling choice)

    dv/dt = ... − k_v(σ) v
    dT/dt = ... − k_T(φ, σ) [T − T_eq(φ, p)]

    T_eq = max{ 200 K, [315 K − (ΔT)_y sin²φ − (Δθ)_z log(p/p0) cos²φ] (p/p0)^κ }
    k_T  = k_a + (k_s − k_a) max(0, (σ − σ_b)/(1 − σ_b)) cos⁴φ
    k_v  = k_f max(0, (σ − σ_b)/(1 − σ_b))

| symbol | value |
|---|---|
| σ_b | 0.7 |
| k_f | 1 day⁻¹ |
| k_a | 1/40 day⁻¹ |
| k_s | 1/4 day⁻¹ |
| (ΔT)_y | 60 K |
| (Δθ)_z | 10 K |
| p0 | 1000 mb = 1.0e5 Pa |
| κ = R/c_p | 2/7 |
| c_p | 1004 J kg⁻¹ K⁻¹ (hence R = κ c_p = 286.857… J kg⁻¹ K⁻¹) |
| Ω | 7.292e−5 s⁻¹ |
| g | 9.8 m s⁻² |
| a_e | 6.371e6 m |

Notes from the paper: σ = p/p_s with the *instantaneous* surface pressure; p in T_eq is
the full pressure σ p_s; no topography; hydrostatic or not is a modelling choice; "no
explicit diffusion is included in the specification" — subgrid mixing is part of the
numerical scheme and must be documented; the paper's own cores used only scale-selective
horizontal mixing and no vertical mixing or convective adjustment. Dinosaur's
`held_suarez.py` (2026-09-08, be5409da) implements the identical constants, which is the
second independent confirmation of the transcription.

### 1.2 The paper's spectral core (a modelling choice, documented for comparison)

Hydrostatic σ-coordinate semi-implicit spectral transform model in vorticity–divergence
form (Bourke 1974); 20 levels equally spaced in σ, top at zero pressure; centred
vertical differences with the hydrostatic equation integrated analytically assuming T
constant within each layer (explicitly *not* energy conserving); leapfrog with the Robert
(1966) time filter; horizontal mixing of ζ, δ and T by a Laplacian raised to the fourth
power (∇⁸) with the e-folding time of the smallest resolved wave always 0.1 day;
triangular truncation; results shown at **T63**. Time step and filter coefficient are not
stated.

### 1.3 Protocol

1200-day integration from an isothermal resting atmosphere with small symmetry-breaking
perturbations; climate = average over the last 1000 days (days 200–1200). The paper
states the climates are *not* converged at the resolutions shown and that the difference
between hemispheres estimates sampling error.

### 1.4 Reported climate (T63 spectral and G72 gridpoint, for tier C6)

- single jet of "roughly 30 m s⁻¹ near 45° latitude", closing off above σ = 0.2;
- surface westerlies reaching "almost 8 m s⁻¹ near 45°";
- equatorial easterlies in the model stratosphere;
- eddy temperature variance [T*²] with two mid-latitude maxima, one in the lower
  troposphere (contours to ~40 K² near σ ≈ 0.75–0.85, ~40–45° latitude) and one above the
  tropopause; unrealistic penetration of the surface maximum into the tropics;
- eddy zonal-wind spectrum with two peaks at zonal wavenumber 5 on the storm-track flanks,
  a third at wavenumbers 1–2 at the track centre, and one at wavenumber 1 at the pole.

## 2. Palintropos configuration (project choices)

| item | value | status |
|---|---|---|
| dynamics | dry hydrostatic PE, σ (Lorenz, Simmons–Burridge 1981), ζ–δ–T–ln p_s | fixed by the core |
| horizontal | T42 triangular on the 64×128 Gaussian grid (`grid=latlon`) | choice; the paper's T63 is not affordable in the window. **Open (DEVLOG S4):** the core truncates every analyzed product at the 2/3 cut, so l_max = 42 evolves only l ≤ 28 (effective T28); matching Dinosaur's retained T42 needs l_max = 63 (cut 42) — user decision before launch |
| vertical | 20 equally spaced σ levels, top at σ = 0 | matches paper and Dinosaur |
| constants | R = 286.857 (= 2/7 · 1004), c_p = 1004, Ω = 7.292e−5, a = 6.371e6, g = 9.8, p0 = 1e5 | paper values |
| forcing | §1.1 exactly | fixed |
| ∇⁸ on ζ, δ, T′ | e-folding 0.1 day at l = 42 (paper's rule applied at our truncation) | choice, frozen at launch. **Open:** with the effective truncation 28 this gives 2.45 days at the smallest evolving wave; HS94's rule would put 0.1 day at l = 28 (or keep 42 with l_max = 63) |
| ln p_s | not diffused | matches GFDL practice |
| time scheme | SI leapfrog (T_ref = 300 K isothermal) + RAW filter (ν = 0.1, α = 0.53) | choice, frozen at launch |
| Δt | 900 s (S4b: RAW-filtered explicit advection bound, DEVLOG); the S8 gate projects its cost | frozen at launch |
| initial state | isothermal 300 K (matches the SI T_ref), p_s = p0 flat, seeded random ζ perturbation of amplitude 1e−6 s⁻¹ in degrees 1–8 (seed recorded) | choice |
| length | 1200 days; statistics over days 200–1200 in five 200-day blocks | paper |
| output | zonal/time-mean accumulators online; scalar series daily; full state every 10 days; checkpoints every 10 days (keep 3) | choice |

## 3. Reference run (Dinosaur)

`REFERENCE_CONFIG.json` is the machine-checked list. Dinosaur pinned at commit
`be5409da2636918cd602ee75b201de1617f3b3aa` (2026-09-08), JAX 0.11.2, `jax_enable_x64=True`,
`SigmaCoordinates.equidistant(20, dtype=float64)`, `Grid.T42()` (64×128), HS forcing with
its defaults (identical to §1.1), `horizontal_diffusion_step_filter(order=4, tau=0.1 day)`
on all prognostic fields, `imex_rk_sil3`, Δt = 600 s, initial `isothermal_rest_atmosphere(
tref=300 K, p1=5 kPa surface-pressure perturbation)`, 1200 days.

Matched: truncation, grid, levels, constants (Dinosaur: R = c_p·κ with c_p = 1004, Ω,
a = 6.37122e6 vs our 6.371e6 — 0.003 %, documented), forcing, ∇⁸ order and e-folding,
length, averaging.
Unavoidable differences: integrator (IMEX RK "SIL3" vs SI leapfrog+RAW), Δt, the exact
per-step diffusion factor (exp(−2Δt K c_l⁴) vs 1/(1+2Δt K c_l⁴)), Dinosaur diffuses ln p_s
and the full T (we diffuse T′ only), initial perturbation (surface pressure vs vorticity),
vertical discretisation of the hydrostatic/energy terms, transform/dealiasing details.

## 4. Verdict tiers

See `docs/superpowers/plans/2026-09-27-held-suarez-24h-plan.md` §1–§2 (copied here at
the freeze).
