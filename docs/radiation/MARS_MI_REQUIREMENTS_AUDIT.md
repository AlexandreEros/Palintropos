# M-I requirements audit: where M1–M7 and N1 come from (2026-11-14)

This is an independent audit of the criteria used in `MARS_MI_PLAN.md`
§2–§2e. The plan, judge, tests and evidence are treated as a **record of
decisions**, not as proof that those decisions were scientifically
justified. The amendment that acts on it is `MARS_MI_PLAN.md` §4. That
amendment changes how requirements are written and classified. It does not
change any historical criterion, prediction, measurement or verdict.

Nothing in production code or in the evidence was changed for this audit,
and no simulation was run. Numbers marked *recomputed* come from the saved
daily states of the runs, read without integrating anything. Appendix A
gives the method. Every recomputed grid-point max |u| and lowest-level speed
was checked against the run's own `series.json` to a relative 10⁻⁹.

## Sources

- **Thomson & Vallis (2019)**, *Atmosphere* 10, 803. Read: Table 1, §3.2
  (the slab-lag scaling), §4.1–4.4, and Fig. 4. Panel 4b was rendered at
  300 dpi to read its colour bar.
- **Isca at the commit the plan pinned, `d1321ac3`**, read 2026-11-14 on
  GitHub:
  - `exp/test_cases/grey_mars/grey_mars_test_case.py`
  - `src/atmos_spectral/model/spectral_dynamics.F90`
  - `src/atmos_spectral/init/spectral_init_cond.F90`
  - `src/atmos_spectral/driver/solo/idealized_moist_phys.F90`
  - `src/atmos_spectral/driver/solo/mixed_layer.F90`
  - `src/atmos_param/two_stream_gray_rad/two_stream_gray_rad.F90`
  - `src/atmos_param/damping_driver/damping_driver.f90`
  - `src/atmos_param/monin_obukhov/monin_obukhov.F90`
  - `src/atmos_param/diffusivity/diffusivity.F90`
  - `src/atmos_param/vert_diff/vert_diff.F90`
  - `src/coupler/surface_flux.F90`
  - `src/shared/constants/constants.F90`

  The same claims were first read at `master` the same day. Every item
  checked at both commits agreed. Section 5 lists what was not checked.
- **This repository:**
  - `MARS_MI_PLAN.md` and its history, `fba92fb` to `50f5180`;
  - the A1–A4 section of `docs/radiation/README.md` (`1078687`);
  - `tools/radiation/judge_mars_pilot.py` and `mars_tides.py`;
  - `src/tropoi/run/radiative/experiment.py` (how the series is sampled);
  - `tests/test_radiation_evidence.py`, `tests/test_boundary_layer.py` and
    `tests/test_radiative_dynamical.py`;
  - the evidence and capsule directories.

## 1. Findings

1. **None of M1–M7 or N1 is a published M-I criterion.** The paper's
   quantitative M-I material is Table 1, which is configuration, and
   Fig. 4b: the zonal-mean zonal wind averaged over L_s 225°–315°. The paper
   compares Fig. 4b *qualitatively* with MACDA. It reports no wind bound,
   tide amplitude, diurnal phase, energy residual, mass drift or
   near-surface wind. No run so far can be compared with Fig. 4b: all are
   T21, 30–60 days long, and cover L_s 0°–29°.

2. **The numerical thresholds are project choices.** There are two kinds.
   - *Inherited* (M2, M3, M4's floor, M6). These are A2–A4, which
     `README.md` states were "written after the runs above existed" and
     calibrated on dry gray Earth. A2 was sized so that the unheated Earth
     run fails by a factor of 22. A4 sits about ten times above Earth's
     measured drift of 1.2 × 10⁻⁶. Pre-registering M2–M4 and M6 fixed
     the *reuse* of these after-the-fact Earth numbers on Mars. It did not
     derive them for Mars.
   - *New* (M1's 150 m s⁻¹ and 100–400 K, the three parts of M5, M7's
     40 m s⁻¹). No rationale was recorded for any of them. Unlike the
     A-series table, the M table has no "why this size" column.

3. **Two criteria measure quantities that Isca's M-I removes by
   construction.** At `d1321ac3`, `spectral_dynamics_nml` defaults
   `do_mass_correction` and `do_energy_correction` to `.true.`, and the
   grey-Mars namelist sets only `do_water_correction: False`.
   `compute_corrections` does two things every step:
   - It rescales `psg(future)` by `mean_surf_press_previous /
     mean_surf_press_tmp`.
   - It adds the uniform `temperature_correction = grav·(mean_energy_previous
     − mean_energy_tmp)/(cp_air·mean_surf_press_previous)`.

   The reference energy is ⟨½|v|² + c_p T⟩ of the previous level advanced
   by the supplied `dt_ug`, `dt_vg` and `dt_tg`. The spectral damping acts
   before the correction. So the correction also returns the ∇⁴
   kinetic-energy loss as uniform heat. M-I's mean-p_s drift and its
   dynamical energy residual are therefore zero by construction. M2 and M6
   test the Palintropos core, not agreement with M-I.

4. **M5's 11–16 h surface-temperature window conflicts with M-I's own
   surface.** This rests on a linearised analytical expectation, not a
   proof (§3). Its repeated failure is consistent with the slab being the
   specified slab. It is not evidence of a defect.

5. **The M7 observable as judged is not the one registered.** §2e registers
   "lowest-full-level |u|". The judge reads `max_wind_bottom`, the
   lowest-level horizontal wind *speed* (`experiment.py` records it as "an
   upper bound on |u|"). Both measures fail the 40 m s⁻¹ bound, with
   different numbers (§3).

6. **The gap analysis (plan §1) omits configuration differences.** The
   grey-Mars test case uses `dt_atmos` **110 s**; the runs use 600 s. Isca
   averages the cosine of the zenith angle over each step; the runs use the
   instantaneous value. Plan §4.5 records these with the other remaining
   differences.

## 2. Classification of the historical criteria

The columns:

- **Pre?** Was it fixed before the experiment it judges?
- **Basis.** **P** means supported by the published M-I reference. **D**
  means derived from physical or numerical principle. **I** means chosen
  internally.

| | criterion | Pre? | origin of the threshold | basis | what its failure establishes |
| --- | --- | --- | --- | --- | --- |
| N1 | continuation finite past the last day (2 sols, 295 steps) | yes (`6701d02`; restated `04f4547`) | no number; mirrors the §2c failure (Hs went non-finite at step 146) | **D**: a regression test for a diagnosed forward-Euler instability | an instability other than the explicit exchange. Passing is weak evidence: one state, 2 sols |
| M1a | reaches the last day without a validation abort | yes (`fba92fb`) | — | **D**: a necessary condition | numerical failure |
| M1b | daily max \|u\| < 150 m s⁻¹, at any level | yes (`fba92fb`) | none recorded; later described as "written for a 30-day spin-up" | **I**. Not supported by **P**: in Fig. 4b M-I's seasonal *zonal-mean* northern jet exceeds the plot's 140 m s⁻¹ colour cap over roughly 1–100 Pa | pilot (§2): the missing sponge, a diagnosis that §2b then tested and confirmed. §2d and §2e: a southern westerly jet still strengthening in southern autumn. **Not** an instability: N1, the temperature bounds and energy closure all hold |
| M1c | 100 K < T < 400 K | yes (`fba92fb`) | none recorded. Resembles Isca's *default* `valid_range_t` (100, 500 K); the grey-Mars test case widens it to [0, 700] | **I**. The paper: M-I has "no lower bound on the temperature" | never failed |
| M2 | \|window-mean residual\| ≤ 10⁻³ F; daily ≤ 10⁻² F | yes, as a reuse of A2 and A3 | A2 and A3: after the fact, on Earth, sized to separate the heated runs from the unheated one | **D** principle (conservation), **I** number. No **P** counterpart: Isca enforces closure, and the paper says only "energetically closed" | a budget defect larger than 10⁻³ F. How well it can detect one depends on the flow: in the §2 pilot the sinks were 4 × 10⁻⁴ F, below the bound, so it could not have detected missing frictional heating |
| M3 | atmosphere + slab − TOA within the M2 bounds; slab identity to 10⁻⁹ | yes | the M2 numbers; the slab part is a bookkeeping identity (observed to about 10⁻¹³) | **D** (identity) + **I** (bound) | a surface-coupling bookkeeping error |
| M4 | unheated residual = measured sinks within max(2 se, 10⁻³ F) | yes | the 10⁻³ F floor from A2. 2 se never binds (0.006 against 0.099 in §2e), so the test is effectively 10⁻³ F. The se treats 20 autocorrelated daily states as independent and ignores the residual's own uncertainty | **D** structure (residual = sinks + a time-integration part), **I** number. No **P** counterpart: Isca has no unheated boundary-layer case (`do_conserve_energy` is hard-coded on in `vert_diff_init`), and its fixer absorbs the ∇⁴ loss | a sink left out or counted twice. The §2e pass depends on a post-run judge fix (the BL sink taken from the series). The fix is documented and correct, but it reversed the first verdict |
| M5a | equatorial T_s maximum between 11 and 16 h | yes (`fba92fb`) | "real, low-thermal-inertia Mars" (plan §2 results) | **I**. In tension with M-I's own configuration (§3) | under the linearised expectation, the slab is the specified slab |
| M5b | tropical p_s S1 ≥ 5 × 10⁻⁴ p_s | yes | none recorded | **I**: a detectability floor; the paper gives no tide amplitude | never failed (1.29–1.42 %) |
| M5c | S1 ≥ 3 × the RMS of the fit residual | yes | none recorded | **I**. The "noise" is physical variability (non-migrating components, eddies), not measurement error, and an amplitude is compared with an RMS | §2e: tropical p_s variance is less dominated by the migrating tide than in §2d. It is not a detection failure: the S1 amplitude is about 2.5 times the RMS of everything else in a fit over 295 times × 64 longitudes |
| M6 | \|mean-p_s drift\| ≤ 10⁻⁵ (30 d); §2e adds ≤ 2 × 10⁻⁵ (60 d) | yes, as a reuse of A4; the scaling was added in `04f4547` | A4: after the fact, ten times Earth's drift. The 60-day scaling assumes a linear drift, which the evidence contradicts: the drift accelerates with the flow | **D** principle, **I** number. No **P** counterpart: M-I's drift is zero by construction | the known non-conservation of the ln-p_s core grows with the flow on Mars. Its energy equivalent is 1.4 × 10⁻⁴ W m⁻² (about 1.4 × 10⁻⁶ F): negligible for M2–M4, relevant for year-long runs |
| M7 | daily max lowest-full-level \|u\| ≤ 40 m s⁻¹, days 11–60 | yes (`04f4547`, before any BL run) | a prediction of what the stress would do ("§2d measured 63") | **I**. The paper gives no near-surface wind | the stress lowered the bottom wind by about a quarter, not to 40 m s⁻¹. That is a failed prediction, not an established error |

## 3. Checks behind the classification

**M5a: a strong analytical expectation.** The paper's §3.2 gives the slab
lag as φ ≈ arctan(C_g ω / 4σT₀³). It assumes:

- a linearised slab energy balance;
- one harmonic of forcing;
- longwave restoring 4σT³ (emissivity 1);
- no change in the atmosphere's downwelling flux.

For M-I's surface:

- C_g = 2 m × Isca `RHO_CP`, with `RHO_CP = RHO0·CP_OCEAN = 1035 ×
  3989.24…`, giving 8.26 × 10⁶ J m⁻² K⁻¹. `mixed_layer_init` sets
  `land_sea_heat_capacity = depth*RHO_CP`. The test case sets `depth` 2.0
  and no land option.
- ω = 2π / 88 440 s.

That gives C_g ω = 587 W m⁻² K⁻¹, against 4σT³ = 2.4–5.3 W m⁻² K⁻¹ for
220–285 K. So φ ≈ 89.5–89.8°, and the S1 maximum is expected near 18 h.

A peak at 16 h would need φ ≤ 60°, and so an effective restoring of
≥ 339 W m⁻² K⁻¹. The one restoring term the estimate leaves out is the
sensible exchange ρ C |v| c_p. At the lowest level it is:

- 0.26 W m⁻² K⁻¹ at |v| = 20 m s⁻¹;
- 0.61 W m⁻² K⁻¹ at 47 m s⁻¹.

These use z ≈ 6.8 m, a neutral C ≈ 1.06 × 10⁻³ and ρ ≈ 0.014 kg m⁻³.

The estimate is not a proof. M5a judges the peak of the fitted S1 + S2
curve, and the model is nonlinear and fully coupled. The diurnal cycle of
downwelling longwave is forcing, not restoring, and it can shift the phase.
The size of the gap, more than two orders of magnitude in the restoring,
makes a pass very unlikely for any faithful M-I surface. The measured
17.0–17.05 h is consistent with the expectation. The size of the gap was
computable from Table 1 before any run.

**M1 against the published reference.** In Fig. 4b, M-I's northern winter
jet saturates the 140 m s⁻¹ colour cap from about 1 to 100 Pa near 60° N.
That is a seasonal zonal mean; instantaneous grid-point maxima are larger.

*Recomputed* zonal-mean and grid-point winds:

| run | largest zonal-mean u | largest grid-point \|u\| | zonal-mean max, day 30 → day 60 |
| --- | --- | --- | --- |
| Hb | 94.6 m s⁻¹ (day 59, σ 6.9 × 10⁻³ ≈ 4.2 Pa, 58° S) | 170 m s⁻¹ (day 42, σ 7.9 × 10⁻⁴ ≈ 0.48 Pa, 64° S) | 59.5 → 92.2 |
| Ub | 90.4 m s⁻¹ (day 58, ≈ 1.1 Pa, 53° S) | 186 m s⁻¹ (day 56, 64° S) | 60.1 → 89.4 |
| Hi (§2d) | 99.0 m s⁻¹ (day 60, ≈ 4.2 Pa, 53° S) | 182 m s⁻¹ (day 44) | 62.7 → 99.0 |

Day 60 of the runs is L_s ≈ 28.8° (Kepler's equation with the run's own
orbit). That is southern autumn, so a strengthening southern westerly jet is
the expected seasonal response. M1b fails on wave amplitude on top of a
~90 m s⁻¹ zonal-mean jet, mostly just below the sponge base. Whether the jet
is realistic cannot be decided before a run can be compared with Fig. 4b.

**M7, both measures** (days 11–60, *recomputed*; both columns reproduce
over every day):

| run | registered: lowest-level \|u\| max | days > 40 | judged: lowest-level speed max | days > 40 |
| --- | --- | --- | --- | --- |
| Hb | 43.1 m s⁻¹ (day 57) | 6 | 46.7 m s⁻¹ (day 27) | 13 |
| Ub | 46.3 m s⁻¹ (day 47) | 2 | 47.2 m s⁻¹ (day 47) | 4 |

The largest lowest-level |v| over the same days is 38.8 (Hb) and
39.9 m s⁻¹ (Ub). The verdict is "not met" under either measure. The plan's
§2e numbers (46.7, 47.2; 13 and 4 days) are the speed.

Both M1b and M7 are **once-daily instantaneous grid-point extremes**: one
state per day, the largest value anywhere. They depend on the sampling time
and on resolution, so T21 and T42 runs measure different things.

**M6's energy weight.** A drift of 2.57 × 10⁻⁵ of the column enthalpy
(c_p T p_s / g ≈ 2.8 × 10⁷ J m⁻² at 200 K) is 714 J m⁻² over 60 days,
1.4 × 10⁻⁴ W m⁻².

## 4. Isca facts, verified at `d1321ac3`

**grey-Mars namelist**

- `dt_atmos` 110; 64 segments of `30*88440` s (1920 sols, about 2.87 Mars
  years); output `atmos_daily` every 88 440 s.
- `valid_range_T` [0, 700].
- `robert_coeff` 0.04.
- `damping_driver`: `do_rayleigh`, `trayfric` −0.125 (days),
  `sponge_pbottom` 0.5 Pa, `do_conserve_energy`.
- `mixed_layer`: `depth` 2.0, `albedo_value` 0.3, `tconst` 285,
  `prescribe_initial_dist`.
- `two_stream_gray_rad`: `use_time_average_coszen` True, `atm_abs` 0.2,
  `sw_diff` 0, `ir_tau_eq` = `ir_tau_pole` = 0.2, `linear_tau` 1,
  `equinox_day` 0, `solar_constant` 589.
- `surface_flux`: `do_simple`, `old_dtaudv`, `use_virtual_temp` False,
  `use_actual_surface_temperatures` False.
- `vert_turb_driver`: `do_diffusivity`, `do_simple`, `constant_gust` 0,
  `do_mellor_yamada` False.
- `diffusivity`: `do_entrain` False, `do_simple`.
- `idealized_moist_phys`: `convection_scheme` 'none', both roughness
  lengths 3.21 × 10⁻⁵ m.
- `vert_coordinate_nml` `bk`: the 26 values the runs use.
- The test case's comment reads "'dry' convection, 1.0 m mixed-layer
  depth", which contradicts its own namelist.

**Defaults the namelist does not override**

- In `spectral_dynamics`:
  - mass and energy corrections on;
  - `damping_option` 'resolution_dependent', `damping_order` 2,
    `damping_coeff` 1.157 × 10⁻⁴ s⁻¹ (one tenth of a day);
  - `raw_filter_coeff` 1.0 (RAW off);
  - `initial_temperature` 264 (in `spectral_init_cond`).
- In `surface_flux`: `use_frierson_mo_drag` false, so the drag comes from
  `mo_drag`.
- In `monin_obukhov`: `rich_crit` 2.0, `drag_min` 10⁻⁵, `stable_option` 1,
  `zeta_trans` 0.5.
- In `diffusivity`: `frac_inner` 0.1, `rich_crit_pbl` 1.0,
  `free_atm_diff` false.
- In `mixed_layer`: `delta_T` 40.

**Behaviour**

- Shortwave: `sw_down = insolation·exp(−sw_tau)` with
  `sw_tau = sw_tau_0·(p_half/pstd_mks)^4`; `sw_up = albedo·sw_down(surface)`
  at every level, so the reflected beam is not absorbed again. Longwave
  optical depth is linear in `p_half/pstd_mks`.
- With `use_time_average_coszen`, the cosine of the zenith angle is averaged
  over `dt_rad_avg`, which defaults to the 110 s step. The diurnal cycle is
  kept.
- The Rayleigh sponge uses the **local** full-level pressure:
  `fact = rfactr·(sponge_pbottom − p)²/sponge_pbottom²` for
  p < `sponge_pbottom`. With `do_conserve_energy` it adds the heating
  −[(u + ½Δt·udt)·udt + (v + ½Δt·vdt)·vdt]/c_p.
- `vert_diff_init(…, .true., …)` is commented "do_conserve_energy is
  hard-coded in". In `uv_vert_diff` that adds
  −[(u + ½Δt·Δu̇)Δu̇ + (v + ½Δt·Δv̇)Δv̇]/c_p to `dt_t`. Temperature is
  diffused as T + gz/c_p.
- The order is `surface_flux`, then `gcm_vert_diff_down`, then
  `mixed_layer`, then `gcm_vert_diff_up`: one implicit sweep. The mixed
  layer uses `Tri_surf%dflux_t` for the implicit increment of the lowest
  level.
- The PBL depth (`do_simple`) is a bulk Richardson number of T + gz/c_p
  against the lowest level, interpolated at `rich_crit_pbl`. The inner-layer
  K comes from `mo_diff`, with stability functions. The outer-layer taper is
  (z/h_inner)·(1 − (z − h_inner)/(h − h_inner))² times the value at h_inner.
  K is zero at and above h.
- The initial surface is `t_surf = tconst − delta_T·(3 sin²φ − 1)/3`.

## 5. Uncertainties left open

- The body of `monin_obukhov_drag_1d`, which `mo_drag` calls, was not read.
  That it solves for ζ iteratively through `solve_zeta` and the integrated
  ψ functions is plausible, but **unconfirmed**.
- The effect of `use_actual_surface_temperatures` False, and of
  `do_simple` in `vert_turb_driver`, was not examined.
- Whether `spectral_initialize_fields` adds a perturbation to the
  quiescent state was not read.
- How `damping_option` 'resolution_dependent' scales `damping_coeff` was
  not read. The ∇⁴ e-folding time at the truncation is therefore not
  confirmed to be 0.1 day.
- The full-level placement (Isca's `vert_difference_option`) was not
  compared with the runs' full levels.
- Whether the test case is the run behind the paper's Fig. 4b is unknown.
  Its comment and namelist disagree, and the paper states neither the
  integration length nor the number of years averaged.

## Appendix A: how the recomputed numbers were made

The inputs are the saved daily states in `docs/runs/<capsule>/states/`
(`state-dNNNNN.npy`, `surface-dNNNNN.npy`; present locally, not tracked)
for `dev-mars-gray-flat-bl{-heated,}-T21L25-06f5356` and
`runs/radiative/dev-mars-gray-flat-sponge-implicit-heated-T21L25-63d05a2`.
Each run's config was rebuilt from its `config.json`, after checking that
the config reproduces its own hash. For each state:

- u and v come from `model.wind_on_state_grid` on the state grid;
- the lowest-level speed comes from
  `ColumnPhysicsCoupling._lowest_wind_speed`, the function that writes
  `max_wind_bottom`;
- the zonal mean is the longitude mean of u.

The grid-point max |u| and the lowest-level speed were checked against
`series.json` (`max_abs_u`, `max_wind_bottom`) to a relative 10⁻⁹. Both
matched on every day for Hb and Ub. The §2d run has no `max_wind_bottom`
series. L_s was computed from the runs' orbit: e 0.0935, perihelion at
L_s 250.85°, L_s = 0 at t = 0, a year of 59 166 360 s.
