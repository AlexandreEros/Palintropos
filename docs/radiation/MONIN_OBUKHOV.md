# The Monin–Obukhov surface layer

An opt-in alternative to the FHZ06 surface layer of `MARS_MI_PLAN.md` §2e,
implementing the formulation Isca's grey-Mars experiment actually runs.
FHZ06 remains the default; nothing about an existing run changes.

Status: implemented and tested. This closes one stated difference from
Isca — the §2e note that "the FHZ06-simplified exchange coefficient and
neutral K profile [stand] against Isca's full Monin-Obukhov forms". It
does **not** reproduce the M-I experiment, and it does not address the
other differences listed in §2e or in `MARS_MI_REQUIREMENTS_AUDIT.md`.

## 1. Provenance

Read from `ExeClim/Isca` at commit `d1321ac3` on 2026-11-14:

| File | What was taken from it |
|---|---|
| `src/atmos_param/monin_obukhov/monin_obukhov_kernel.F90` | the stability functions, their integrals, the Newton solve, `drag_1d`, `diff`, `stable_mix`, and the three published checksums |
| `src/atmos_param/monin_obukhov/monin_obukhov.F90` | the namelist defaults and the iteration constants |
| `src/atmos_param/diffusivity/diffusivity.F90` | `pbl_depth`, `diffusivity_pbl` and the `do_simple` branches |
| `src/coupler/surface_flux.F90` | how the coefficients become a sensible heat flux, and the gustiness |
| `exp/test_cases/grey_mars/grey_mars_test_case.py` | the grey-Mars namelist settings |

The four Fortran files were re-fetched from `raw.githubusercontent.com` at
that commit and checked by SHA-256 against the copies used, so the reading
is pinned rather than remembered.

**Licensing.** Isca is GPL-3.0. The Python here is an independent
reimplementation: the formulation, the safeguards and the defaults were
read from those files and re-expressed, and no Fortran was copied. This
repository currently has **no `LICENSE` file**, so the question of whether
a reimplementation of a GPL model's published formulation needs a
compatible licence is open and is the owner's call — it is flagged here
rather than decided.

## 2. What the scheme computes

`tropoi.physics.monin_obukhov` holds the surface layer:

- `stability_derivative_m/_t(zeta)` — φ_m, φ_t. Unstable: the
  Businger–Dyer `(1 − 16ζ)^(−1/4)` and `(1 − 16ζ)^(−1/2)`. Stable:
  `stable_option = 1` (Isca's default), the rational
  `1 + ζ(5 + bζ)/(1 + ζ)` with `b = 1/rich_crit`; `stable_option = 2`, a
  linear branch joined to an asymptotic one at `zeta_trans`.
- `stability_integral_m/_t` — the height-integrated forms f_m, f_t, f_q.
- `solve_zeta` — the Newton iteration on `Ri = ζ f_t / f_m²`.
- `surface_drag` — Isca `monin_obukhov_drag_1d`, returning a
  `SurfaceExchange` with C_m, C_t, C_q, u\*, b\*, Ri, ζ, and the
  convergence and capping masks.
- `diffusivities(z, u*, b*)` — Isca `monin_obukhov_diff`:
  `K = κ max(u*, 1e-10) z / φ(ζ(z))`, `ζ(z) = −κ b* z / u*²`.
- `stable_mix` — the free-atmosphere mixing factor. Not used by this
  configuration (`free_atm_diff` is off in grey-Mars); implemented because
  its checksum is an exact test of the stable branch.

`tropoi.physics.boundary_layer.monin_obukhov_diffusivity_profile` is Isca
`diffusivity_pbl`, which is **not** the FHZ06 taper:

| region | K |
|---|---|
| `z < h_i` | `κ u* z / φ(ζ(z))` |
| `h_i ≤ z < h` | `K(h_i) · (z/h_i) · (1 − (z − h_i)/(h − h_i))²` |
| `z ≥ h` | 0 |

with `h_i = frac_inner · h`. Above `h_i` Isca **freezes φ at its `h_i`
value** instead of continuing to evaluate it, so the outer layer is
neutral-shaped growth times a constant stability factor. In the neutral
limit `K(h_i)·z/h_i = κ u* z`, so this collapses onto the existing FHZ06
profile — which is why the existing taper was already Isca-consistent for
the neutral case. The reduction is checked to ~1e-15 relative.

`tropoi.physics.surface.monin_obukhov_sensible_heat_flux` is the flux
assembly of `surface_flux.F90`:
`H = ρ c_p C_t |v| (T_s − θ_a)`.

## 3. Defaults

Isca's grey-Mars test case sets **no `monin_obukhov_nml` block**, so it
runs the module defaults, and those are what `MoninObukhovOptions` carries:

| parameter | value | note |
|---|---|---|
| `rich_crit` | 2.0 | the **surface-layer** critical Richardson number |
| `drag_min` | 1e-5 | floor on every coefficient |
| `stable_option` | 1 | the rational stable φ |
| `zeta_trans` | 0.5 | `stable_option = 2` only |
| `neutral` | False | |

Not namelist entries but module parameters, also reproduced: 20 iterations,
tolerance 1e-4, `zeta_min` 1e-6, `small` 1e-4 m²/s², `ustar_min` 1e-10 m/s.

Three critical Richardson numbers now coexist and are easy to confuse:

| name | value | where |
|---|---|---|
| FHZ06 `critical_richardson` | 1.0 | the FHZ06 drag coefficient only |
| `monin_obukhov.rich_crit` | 2.0 | the MOST surface layer |
| `rich_crit_pbl` | 1.0 | the boundary-layer depth, both schemes (Isca's own separate default) |

Passing `critical_richardson` to a MOST configuration is **rejected**, not
ignored, because its role passes to `rich_crit` and silently dropping it
would change the physics without saying so.

## 4. Reference agreement

There is no Fortran compiler on the development machine, so agreement is
established from what Isca publishes. Two kinds of evidence are available,
and they are **not** the same claim. The distinction matters, so it is kept
throughout.

### 4a. Aggregate agreement: the published checksums (exact)

Isca's kernel tests store their expected answers as
`transfer(sum(x), integer(8))` — the IEEE-754 bit pattern of a double
reinterpreted as an integer. Adjacent doubles have adjacent bit patterns, so
the *integer difference* between our checksum and Isca's is an error in
units in the last place, and zero means the quantity matched exactly.

All three published checksums (Intel/Lahey values at `d1321ac3`) reproduce
with a difference of **exactly zero**:

| Isca test | checksum | covers | result |
|---|---|---|---|
| `test_drag` | `4466746452959549648` | the full nonlinear solve over five columns: unstable and stable, three roughness regimes | **0 ulp** |
| `test_stable_mix` | `4590035772608644256` | the stable branch, decoding to `0.07454248254045348` | **0 ulp** |
| `test_diff` | `-9222066590093362639` | K_m and K_h at an unstable point | **0 ulp** |

**What that does and does not establish.** Each checksum is
`sum over arrays of bits(sum over elements of array)`. Matching it at zero
constrains those aggregates — not the individual elements. An error in one
element could in principle be cancelled by another element of the same
array, or by another array's bit pattern. It is strong evidence and it is
exact, but it is **aggregate**, not elementwise bit identity, and calling it
"bit for bit" element by element would overstate it.

The two narrower cases are worth stating precisely. `test_stable_mix` is one
array, so its checksum decodes to the exact reference *sum* of five
elements; three of those are beyond `rich_crit` and must be identically
zero, which is asserted separately, leaving the aggregate to constrain two.
`test_diff` uses 1×1×1 arrays, so each `sum` is a single element and there is
no intra-array cancellation — two scalars, one equation, and they sit in
different binades (1.77 and 3.03), so a compensating pair would have to be
contrived rather than unlucky.

Fortran's `integer(8)` addition wraps, so the sums are reduced modulo 2⁶⁴
before comparison; without that the drag checksum appears as
`9224677483616188977`.

### 4b. Elementwise agreement: the published per-column scales

Isca's `test_profile` hardcodes `u_star` and `b_star` **column by column**,
alongside the same five columns of geometry `test_drag` uses. Those are ten
independent published decimal literals, so matching them pins each column
separately and closes the cancellation loophole above.

One subtlety had to be resolved to use them. Isca's test *program* sets
`rich_crit = 10.0` before calling `test_drag`, but `test_profile`'s literals
were evidently generated with the **module default of 2.0**: the four
unstable columns do not depend on `rich_crit` at all (the Businger–Dyer
branch contains no `b_stab`), and only the fifth, stable column does. At
10.0 that one column is off by 4e-4 while the other four match to 1e-14; at
2.0 every column matches. 2.0 is also the value grey-Mars runs, so this
check sits at the production setting.

| quantity | worst elementwise relative difference, all five columns |
|---|---|
| `u_star` | **3.7e-14** |
| `b_star` | **5.6e-13** |

That is a few ulp, and it is limited by the printed precision of the
literals (15–16 significant digits), so it is elementwise agreement **to the
reference's own precision** — not a demonstration of bit identity, which
those literals cannot support.

### 4c. Elementwise bit identity, between our own two implementations

On Isca's `test_drag` inputs the fused CUDA kernel reproduces the NumPy path
**exactly, element for element**, in `drag_m`, `drag_t`, `drag_q`, `u_star`
and `b_star`; only `richardson` differs, and by at most **2 ulp**
(3.4e-16 relative, 1.5 ε).

That difference has a single identified cause rather than a tolerance.
`nvcc` contracts `s*s + small` into `fma(s, s, small)` by default, so the
denominator of `−z·Δb/(s² + small)` is rounded **once** on the device and
twice in NumPy. Rebuilding the NumPy expression with an exactly-rounded fma
in the denominator reproduces the kernel's output **bit for bit on all five
columns**, which is what the test asserts — a stronger statement than any ulp
budget, because it leaves no room for a second source of difference to hide
inside a tolerance. `-fmad` is left at its default on purpose: turning it off
would cost accuracy in the iteration to buy agreement in a diagnostic.

Both sides are ours, so this is a genuine elementwise bit-level claim — about
internal consistency, not about Isca. It does not generalize: see §5.

### 4d. Summary of what is claimed

| claim | scope | strength |
|---|---|---|
| three published checksums reproduce at 0 ulp | aggregate per array, summed | exact |
| ten published per-column scales reproduce | elementwise, every column | a few ulp, at the literals' precision |
| fused kernel equals the NumPy path | elementwise, our own code | exact on the five coefficients; Ri exact once the fma contraction is accounted for (≤2 ulp otherwise) |
| the closure is solved | elementwise, independent of Isca | see below |

### 4e. The closure's own accuracy

Independently of Isca, the converged ζ must satisfy `Ri = ζ f_t / f_m²`.
Over a 40 000-column sweep spanning Ri from −5e6 to +4e6, wind from 1e-4 to
100 m/s and z/z₀ over five decades:

- median relative residual in Ri: **4e-7**;
- maximum: **1.5e-2**, and every column above 1e-3 has `|ζ| < 1`.

That is Isca's stopping rule, not a defect. The test accepts the *smaller*
of the absolute and the relative correction, so for `|ζ| < 1` the bound on
ζ is effectively absolute and its relative accuracy is only `1e-4/|ζ|`.
Checked against the exact root of the closure (found by bisection, not by
Newton) on the loosest 60 columns:

| quantity | worst error against the exact root |
|---|---|
| ζ | 1.5e-2 |
| **C_m** | **5.8e-4** (median 9e-5) |

So the looseness does not reach the quantity the model uses: f_m depends
only weakly on ζ near neutral. Worth knowing if anyone ever wants ζ or the
Obukhov length itself as a diagnostic — `SurfaceExchange.zeta` is exposed
for that, with this caveat.

No column failed to converge anywhere in the sweeps (200 000 columns across
three option sets), and the iteration is nowhere near its 20-step limit:
median 2 iterations, p99 5, maximum 6.

## 5. The CUDA kernel

`tropoi.physics.monin_obukhov_cuda.surface_drag_fused` runs the whole solve
in one launch, one thread per column.

**Why it is correct to fuse.** The columns do not interact. A column leaves
Isca's iteration the moment its own correction passes the tolerance and its
ζ is frozen from then on, so the global loop produces exactly the
trajectory a per-column loop produces; the global exit test only decides
how many times the loop spins.

**Why it is worth it.** The vectorized path cannot avoid ~25 elementwise
kernels per iteration and one device-to-host synchronization per iteration
for the `active.any()` exit test. Measured over a 20 000-column sweep, the
Newton solve needs a **median of 2 iterations and at most 6** — the 20-step
limit is never approached — so a solve is roughly 150 kernel launches and
6 synchronizations. At these grid sizes that is launch-bound, not
arithmetic-bound: 34.5 ms for 2048 columns is about 5.7 ms per iteration
for a few thousand flops.

Measured on the MX110 (2 GB), 20 repetitions, synchronized:

| columns | vectorized CuPy | fused kernel | speed-up |
|---|---|---|---|
| 2 048 (≈T21 product grid) | 34.5 ms | 1.35 ms | **25×** |
| 8 192 | 32.7 ms | 3.83 ms | 8.5× |
| 65 536 | 86.2 ms | 26.2 ms | 3.3× |

**Agreement.** On Isca's own `test_drag` inputs the kernel is elementwise
bit-identical to the NumPy path in all five exchange coefficients and scales,
and differs only in the diagnostic Richardson number by the fma contraction
identified in §4c; it reproduces the published aggregate checksum at 0 ulp.
That does not generalize: over a 60 000-column randomized
sweep the worst relative difference is ~1e-13 of the field maximum, and
~2e-13 in the broadest 200 000-column sweep. It is not guaranteed
bit-identical in general — device and host `log`, `atan`
and `pow` need not round alike, and the 1e-4 stopping rule could in
principle separate two implementations by an iteration — so the path is
**opt-in** (`"fused_kernel": true` in the boundary-layer options) and
records itself in the run signature. `--use_fast_math` is deliberately off.

## 6. Configuration

Preset `dev-mars-gray-flat-bl-most`: the `-bl` preset with the surface
layer swapped. Identical in every other respect (same sponge, same implicit
exchange, `drag` and `convection` still None, same 60 days).

```
"sensible":       {"roughness_length": 3.21e-5, "gustiness": 0.0,
                   "scheme": "monin_obukhov", "monin_obukhov": {},
                   "implicit": true}
"boundary_layer": {"roughness_length": 3.21e-5, "rich_crit_pbl": 1.0,
                   "frac_inner": 0.1, "gustiness": 0.0,
                   "scheme": "monin_obukhov", "monin_obukhov": {}}
```

The scheme must be selected on **both** sides and with the same options;
`RadiativeConfig` rejects a half-selection or a mismatch, because the two
objects are evaluated separately (bulk flux in the column physics, stress
and K profile in the lagged tendency) and nothing structural would
otherwise stop a Monin-Obukhov heat flux running over an FHZ06 K profile.

**Reproducibility.** Every one of the 33 recorded radiative configurations
under `docs/radiation/evidence/` and `docs/runs/` re-hashes identically
after this change. An FHZ06 boundary-layer term's signature is byte for
byte what it was; the MOST keys appear only when the scheme is chosen.

## 7. What actually changes, in numbers

Mars column, z_a = 10 m, z₀ = 3.21e-5 m, T_s = 240 K, Isca defaults.
`C_fhz` is the FHZ06 coefficient at the same bulk Richardson number.

| case | ΔT (K) | \|v\| | Ri | ζ | L (m) | C_m | C_t | C_fhz |
|---|---|---|---|---|---|---|---|---|
| strong daytime heating | +40 | 10 | −6.2e-2 | −0.77 | −13 | 1.18e-3 | 1.25e-3 | 1.00e-3 |
| moderate daytime | +15 | 6 | −6.4e-2 | −0.80 | −12 | 1.18e-3 | 1.26e-3 | 1.00e-3 |
| near neutral | +0.2 | 8 | −4.8e-4 | −6.1e-3 | −1637 | 1.004e-3 | 1.006e-3 | 1.00e-3 |
| light nocturnal inversion | −5 | 6 | +2.1e-2 | +0.30 | 33 | 8.19e-4 | 8.19e-4 | 9.58e-4 |
| strong nocturnal inversion | −25 | 3 | +0.43 | +13.5 | 0.7 | 1.62e-4 | 1.62e-4 | 3.26e-4 |
| strong inversion, calm | −25 | 0.3 | +43 | — | — | 1.0e-5 (capped) | 1.0e-5 | **0** |

Three things to take from it:

1. **At neutral the two schemes agree** (1.004e-3 against 1.00e-3), as they
   must — both reduce to the log law.
2. **Unstable, MOST raises the coefficient ~18 %** and separates heat from
   momentum (C_t/C_m ≈ 1.06 here, and K_h/K_m reaches 3.6 in the K profile
   below). FHZ06 does neither: it uses the neutral coefficient unstable, and
   one K for both.
3. **Stable, MOST suppresses more at moderate stability and never switches
   off.** FHZ06 with Ri_c = 1 goes to *exactly zero* at Ri ≥ 1, so a calm
   Mars night decouples the surface from the air completely; MOST's rational
   stable φ plus the `drag_min` floor always leaves some exchange. That is
   the largest behavioural difference, and nocturnal inversions on Mars are
   where it will show.

K profile at h = 2 km, u\* = 0.4 m/s (m²/s), at 10/50/200/500/1000/1800 m:

| | 10 | 50 | 200 | 500 | 1000 | 1800 |
|---|---|---|---|---|---|---|
| unstable K_m | 2.8 | 20.2 | 114 | 198 | 176 | 12.7 |
| unstable K_h | 4.8 | 51.2 | 406 | 705 | 627 | 45.1 |
| neutral K_m = K_h | 1.6 | 8.0 | 32.0 | 55.6 | 49.4 | 3.6 |
| stable K_m = K_h | 0.6 | 1.5 | 3.2 | 5.5 | 4.9 | 0.35 |
| FHZ06 (neutral) | 1.6 | 8.0 | 32.0 | 55.6 | 49.4 | 3.6 |

The neutral row is the FHZ06 row, which is the identity of §2 made concrete.

None of this says the scheme improves the M-I comparison. It says what the
scheme does differently. Whether that moves a 60-day flat-Mars climate was
an experiment rather than a claim; §10 reports it, and the answer is that it
does not.

## 8. Differences from Isca that remain

Listed because they are differences, not because they are known to matter.

1. **Surface–atmosphere coupling is not jointly implicit.** Isca solves the
   surface fluxes and the vertical diffusion implicitly together
   (`surface_flux` supplies the flux derivatives, `vert_diff` solves the
   coupled column). Here the bulk sensible exchange is backward-Euler
   between bottom layer and slab, and the diffusion is a separate implicit
   column solve. Out of scope by instruction; the largest remaining
   structural difference.
2. **The solve is performed twice per step, on two grids.** The column
   physics computes the heat flux on its grid and the lagged tendency
   computes the stress and the K profile on the product grid. Isca solves
   once and passes u\*, b\* into `diffusivity`. This mirrors the existing
   FHZ06 arrangement exactly and was kept deliberately; it means the u\*
   driving the diffusivity is the product-grid one, which is the right one
   for that term, but the two solves are not the same arithmetic.
3. **The thermodynamic difference driving the heat flux.** Isca uses
   `T_s − θ_a` with `θ_a = T_a (p_s/p_a)^κ`; FHZ06 uses the dry-static-energy
   difference. The MOST path follows Isca, and forms the pressure ratio as
   `exp(g z_a/(c_p T_a))` from the Simmons–Burridge height the Richardson
   number already uses, rather than introducing a second full-level
   pressure. For a hydrostatic layer these are the same number; the two
   *schemes* differ at second order in `x = g z_a/(c_p T_a)`. Measured on a
   Mars column with `z_a = 10` m, where the FHZ06 difference is exactly
   zero, the Monin-Obukhov flux is `−ρ C_t |v| c_p T_a x²/2` — 1.7e-8 of
   the flux a one-kelvin difference would drive, and asserted against that
   closed form rather than merely bounded.
4. **The depth diagnosis.** Kept, and consistent: Isca's `do_simple` always
   takes the neutral/stable branch, and its Richardson number is built from
   `T + (g/c_p) z` where this code uses `c_p T + g z` — the `c_p` cancels,
   and that identity is tested. The one change is the calm-column
   regularization: the MOST path uses Isca's `speed² + 1e-4` offset instead
   of the FHZ06 `max(speed², 1e-12)` clamp, so the depth diagnosis and the
   surface exchange treat a calm column the same way. With a clamp a calm,
   stably stratified column looks critical at the first level; with the
   offset it looks neutral.
5. **Moisture.** `drag_q` is computed and never consumed — the model is dry.
   `roughness_moisture` defaults to `roughness_heat` rather than to
   grey-Mars's 0 m, which would send `ln(z/z_q)` to infinity; nothing else
   changes, because f_q enters nothing else.
6. **Isca's own quirks, reproduced or noted.**
   - A level at or below its roughness height gives `ln(zz/z₀) = 0` and a
     singular coefficient. `max(z, z₀, z_t, z_q)` prevents a *negative*
     logarithm, not a zero one. Isca shares this; neither model guards it.
   - Isca's `solve_zeta` ends on a global `maxval(corr) <= error`, and
     `corr` keeps its last value for columns that dropped out at the
     `zeta_min` test rather than by converging. A stale entry above the
     tolerance runs the loop to `max_iter` and returns `ier = 1` although
     every column has converged. No column's result changes, and Isca's
     callers ignore `ier`. Here convergence is tracked per column instead.
   - Non-convergence is **raised**, not absorbed:
     `MoninObukhovConvergenceError` names the count and the Richardson
     range of the failing columns. Nothing substitutes a neutral or clipped
     coefficient. Capped columns (`Ri ≥ 0.95 rich_crit` → `drag_min`) are
     Isca's design and are not failures.
7. **Not attempted:** entrainment (`do_entrain` is off in grey-Mars too),
   free-atmosphere diffusion (`free_atm_diff` off), `pbl_mcm`, the
   `ind_lcl` depth option, molecular diffusion, `mo_profile` (the 10 m wind
   and 2 m temperature diagnostics), and the unstable branch of
   `pbl_depth`, which `do_simple` makes unreachable.

## 9. Tests

`tests/test_monin_obukhov.py`. CPU: the three Isca checksums; the five
regimes (neutral gives the closed-form log law, unstable enhances exchange
and favours heat over momentum, stable suppresses it monotonically,
near-critical takes `drag_min`, calm is offset-regularized); the closure
residual and the exact-root comparison; the neutral-limit identity with the
FHZ06 K profile; the depth-formulation identity; continuity of
`stable_option = 2` at the transition; option validation; the configuration
contract and the frozen signatures. GPU: NumPy↔CuPy agreement and
NumPy↔fused-kernel agreement over three option sets, the checksum on the
device, and a coupled run — one day closes its budget, two days resume
bit-identically through a checkpoint, and the MOST state differs from the
FHZ06 state it replaces.

## 10. The 60-day experiment (2026-11-15)

The scheme was run on the §2e flat-Mars configuration: preset
`dev-mars-gray-flat-bl-most`, `monin_obukhov: {}` (every Isca module
default), both substitutes removed, 60 days on a Colab T4 from `8e98d7b`
with a clean tree. Two runs, differing only in frictional heating, as the
FHZ06 pair does:

- **Hm**, heating on — `dev-mars-gray-flat-bl-most-heated-T21L25-60d-8e98d7b-t4`
- **Um**, heating off — `dev-mars-gray-flat-bl-most-T21L25-60d-8e98d7b-t4`

They are compared with **Hb** and **Ub**, the §2e runs of the identical
configuration with the FHZ06-simplified forms. Both pairs completed 60
days; all four means are over days 41–60, twenty daily values each.

**This is a measurement, not a judged experiment.** No criterion was
registered before these runs, and none is applied to them: M1–M7 are
historical and are not requirements from here on (`MARS_MI_PLAN.md`
§4.1). What follows is a comparison, in the sense of a DM.

### 10.1 What the swap changes

| days 41–60 mean | Hb (FHZ06) | Hm (MOST) | Ub (FHZ06) | Um (MOST) | within-run sd |
| --- | --- | --- | --- | --- | --- |
| lowest-level max wind (m s⁻¹) | 40.5 | 36.4 | 37.2 | 38.2 | 2.8 / 3.3 |
| max \|u\| anywhere (m s⁻¹) | 138.1 | 145.7 | 145.3 | 146.4 | 12.1 / 14.9 |
| BL depth, daily max (km) | 14.04 | 14.34 | 14.33 | 14.96 | 1.28 / 0.79 |
| BL kinetic sink (W m⁻²) | −0.1007 | −0.0891 | −0.1037 | −0.0880 | 0.010 / 0.014 |
| mean surface temperature (K) | 227.08 | 227.04 | 227.05 | 227.02 | 3.07 / 3.08 |
| \|mean residual\| / F | 1.49 × 10⁻⁴ | 1.48 × 10⁻⁴ | 2.87 × 10⁻³ | 3.04 × 10⁻³ | — |
| relative drift of mean p_s | −2.57 × 10⁻⁵ | −2.72 × 10⁻⁵ | −2.21 × 10⁻⁵ | −2.34 × 10⁻⁵ | — |

The last column is the standard deviation of the twenty daily values
within a run, averaged over the pair — heated / unheated. It is the only
noise estimate available: each configuration was run once, so there is no
ensemble, and the daily values are autocorrelated, which makes the true
uncertainty of a window mean *larger* than twenty independent samples
would give.

**Two differences keep their sign in both pairs**, which is what a scheme
effect looks like with one run per configuration:

- the **boundary-layer kinetic sink falls by 11–15 %** in magnitude
  (−0.101 → −0.089 heated, −0.104 → −0.088 unheated). The full forms
  remove less kinetic energy at the surface than the FHZ06 simplification
  over this climate, despite §7's stable branch never decoupling
  completely;
- the **diagnosed BL depth is slightly greater**, by 0.3 km heated and
  0.6 km unheated — 2–4 %, well inside the day-to-day spread of 0.8–1.3 km.

**Everything else is within the noise.** The mean surface temperature is
unchanged to 0.04 K, a hundredth of its own spread. The wind aloft rises
by 7.6 and 1.1 m s⁻¹ against a spread of 12–15. The energy residual is
indistinguishable between Hb and Hm (1.48 against 1.49 × 10⁻⁴ F): the
swap does not perturb the budget, and the unheated pair misses A2 by the
same factor of three in both schemes, for the reason §6 of
[the radiation report](README.md) gives ("Frictional heating becomes the
preset default").

### 10.2 One reading to avoid

Taken alone, the heated pair says the lowest-level wind fell from 40.5 to
36.4 m s⁻¹ — across the 40 m s⁻¹ line of the historical M7. It should not
be read that way, for two independent reasons. The unheated pair moves the
*other* way (37.2 → 38.2), so the sign is not a property of the scheme;
and both shifts are smaller than the 2.8–3.3 m s⁻¹ spread within either
run. M7 is in any case historical and is not re-judged here.

The same caution applies in the other direction: nothing here shows the
scheme is *worse*. The honest summary is that over 60 days at T21, with
Isca's defaults, the full Monin–Obukhov forms leave this climate where the
FHZ06 simplification left it.

### 10.3 What the comparison does not control

- **Commit and device differ.** Hb/Ub ran at `06f5356` on an L4; Hm/Um at
  `8e98d7b` on a T4. The FHZ06 configuration hashes are unchanged across
  those two commits (§6, and the preset-identity tests), but the FHZ06
  runs were not repeated at `8e98d7b`, so a commit or device contribution
  is bounded only by those identities, not measured.
- **One run per configuration**, so every difference above is a single
  realization of a chaotic flow. Sign consistency across the heated and
  unheated pairs is the substitute for a repeat, not an equal to it.
- **T21, Δt 600 s, 60 days**, none of them M-I's (IR-9, IR-10). A scheme
  difference could be real at T42 and invisible here.
- **The largest structural difference from Isca remains** — the split
  surface and diffusion solves (§8 item 1, IR-8). This experiment changed
  the forms, not the coupling, so it says nothing about that.
