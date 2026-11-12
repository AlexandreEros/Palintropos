# Radiative–dynamical foundation ("Doing RT", issue #26)

Status on 2026-11-10, branch `feat/radiative-dynamical-foundation`. This page
is the handoff: what exists, what was measured, what failed or was cut short,
and what is not validated. Nothing here is a claim about a real planet.

## 1. What exists

| Piece | Where | State |
| --- | --- | --- |
| Gas thermodynamics (R, c_p(T) from molecular constants, H2 ortho/para) | `src/tropoi/physics/thermodynamics.py` | implemented, tested |
| Planet presets with per-planet stretched sigma grids | `physics/planets.py`, `SigmaGrid.stretched` | implemented, grids measured (§5); presets are bulk parameters only |
| Two-stream solver, absorbing + scattering, thermal + stellar, adding method | `physics/radiation/twostream.py` | implemented, tested against independent ODE solutions |
| Column contract and spectral loop | `physics/radiation/scheme.py` | implemented, tested |
| **Gray** optics (FHZ06, SL09, Isca Mars formulas) | `physics/radiation/gray.py` | implemented, tested against analytic solutions |
| **Correlated-k** optics machinery (tables, builder, Planck band weights) | `physics/radiation/kdist.py` | implemented, tested against line-by-line on a **synthetic** spectrum only. **No real-gas opacity data exist in the repository.** |
| Insolation (diurnal mean, diurnal cycle, tidally locked, prescribed) | `physics/insolation.py` | implemented, tested |
| Dry convective adjustment (closed form, enthalpy conserving) | `physics/convection.py` | implemented, tested |
| Slab surface, bulk sensible flux (FHZ06 eqs. 1, 10, 12–14) | `physics/surface.py` | implemented, tested |
| Column physics + single-column radiative-convective model | `physics/column_model.py` | implemented, tested |
| PE coupling as a lagged stepper term, energy accounting | `temporal/column_coupling.py`, `temporal/semi_implicit.py` (`lagged_terms`) | implemented, tested |
| Latitude-dependent Rayleigh drag (SL09) | `temporal/tendencies/drag.py` | implemented; exercised by the coupled test, no dedicated test |
| Experiment driver, checkpoints, resume, CLI | `src/tropoi/run/radiative/` | implemented, tested at T9 |

`tropoi.physics` imports nothing else from `tropoi` (enforced by
`tests/test_layout.py`), so the radiation serves any future dynamical core.
With no `lagged_terms` the stepper is bit-for-bit what it was (tested), so the
dry and Held–Suarez experiments are unchanged.

### Gray versus real-gas: the line that must not blur

- **Gray radiation is a working, tested, coupled capability.** Optical depth
  is a prescribed function of pressure; there is no spectroscopy.
- **Real-gas radiation is an untested-on-real-data code path.** The
  correlated-k provider runs the same solver over (band × g) points and
  reproduces a line-by-line calculation of random Lorentz lines (§4). It has
  never been given a real k-table, has not been compared with any reference
  model (RRTMGP, SOCRATES, petitRADTRANS, exo_k, HARP), has no Rayleigh
  scattering, no continuum or collision-induced absorption, and combines
  gases only by adding k at equal g. It has not been run on the GPU or
  coupled to the dynamics. No planet's radiation is represented.

## 2. Conventions (column contract)

Index 0 is the top. `K` layers, `K + 1` interfaces; `p_{k+1/2} = sigma_{k+1/2}
p_s`, top interface at zero pressure. Fluxes are W m^-2, positive in their own
direction; `net = down − up`; a layer gains `net[k] − net[k+1]`; heating is
`g (net[k] − net[k+1]) / (c_p Δp)`, so the column integral of heating equals
`net_top − net_bottom` to round-off. Lower boundary: a surface (temperature,
emissivity, albedo) or `closed` (no surface: every downwelling watt is
returned upward; interior heat is deposited in the lowest layer, as in SL09).
Optics outside a k-table's (p, T) range raise an error unless clamping is
requested.

## 3. Tests run on this branch (2026-11-10, Windows, MX110, Python 3.12.12, CuPy 13.4.0)

Results are recorded in §9 after the final runs. New test files:

- `tests/test_radiation_twostream.py` (CPU, 6 tests)
- `tests/test_physics_columns.py` (CPU, 17 tests)
- `tests/test_radiation_kdist.py` (CPU, 6 tests)
- `tests/test_radiative_dynamical.py` (2 CPU, 10 GPU tests)

## 4. Radiative-transfer accuracy: what was actually compared

| Check | Reference | Measured |
| --- | --- | --- |
| Layer reflectance/transmittance, direct beam (3 closures) | SciPy collocation solution of the same two-stream ODEs | < 1e-9 (cases where the closure itself goes negative are skipped; the code clips them) |
| 6 scattering layers, linear sources, reflecting surface | same | < 1e-5 W m^-2 on fluxes of O(300) |
| Gray radiative equilibrium | analytic `σT⁴ = OLR(1+τ)/2` | net-flux error 4× smaller per doubling of K (second order) |
| SL09 solar flux | their closed form; general solver with diffuse incidence | 1e-14; Bond albedo 0.3433 (paper: 0.343) |
| Two-stream angular approximation | exact `1 − 2E₃(τ)` | worst-case flux-emissivity error 0.032 (D = 1.66), 0.087 (D = 2). **This is the method's own error and is not removed by anything here.** |
| Correlated-k, thermal, inhomogeneous column, 16 g/band | line-by-line of 300 random Lorentz lines, same solver | outgoing flux +0.17 %, surface down −0.25 % of in-band surface emission; heating within 3.1 % of its peak (0.3 % below the top three levels) |
| SL09 radiative-convective column | paper's statements | OLR 14.0 W m^-2 (paper: uptake 14.0); neutral below ~0.35–0.45 bar (paper: stable above ~0.5); temperature minimum at 0.15 bar (paper: tropopause ~0.15); 169 K at 1 bar |

None of these is a comparison with an independent *radiation model* on a real
atmosphere. Requirement 1 of the brief (accuracy against independent
references across documented conditions) is met only for the solver and the
gray formulas.

## 5. Dynamical-core applicability (no radiation involved)

Evidence: `docs/radiation/evidence/regimes.json`
(`python tools/radiation/regimes.py`, CPU) and
`docs/radiation/evidence/adiabatic_energy.json`
(`python tools/radiation/adiabatic_energy.py`, GPU).

| Planet preset | R, c_p (J/kg/K) | depth / radius | g change over depth | c_p spread over stated T range | top full level |
| --- | --- | --- | --- | --- | --- |
| earth | 287.0, 1003 | 0.009 | 1.8 % | 0.3 % | 50 Pa |
| mars | 191.2, 737 | 0.028 | 5.4 % | 21 % | 0.06 Pa |
| venus | 191.4, 1015 | 0.026 | 5.0 % | **45 %** | 4.6 Pa |
| titan | 303.4, 1069 | **0.096** | **16.7 %** | 0.03 % | 0.7 Pa |
| jupiter (100 bar) | 3593, 10930 | 0.003 | 0.7 % | 20 % | 500 Pa |

What this shows about the existing hydrostatic, shallow, constant-g,
constant-c_p core: constant c_p is a ≥ 20 % approximation on Mars, Jupiter and
Venus (45 %); the shallow, constant-gravity approximation is strained on
Titan. **Neither a variable-c_p thermodynamic equation nor a deep-atmosphere
core was implemented.** Non-ideal-gas effects at 100 bar were not quantified.

Tested (GPU, T9, 12 stretched levels): isothermal rest has exactly zero
tendency on Mars, Venus, Titan and Jupiter constants and grids; one-day
integrations on all five presets stay finite; the semi-implicit operator has
real gravity-wave speeds on every preset grid.

**Vertical grids.** A 100-bar hydrogen column with `τ ∝ p²` in
radiative-convective equilibrium:

| grid | levels | T at 1 bar (K) | levels above 1 bar |
| --- | --- | --- | --- |
| uniform σ | 30 / 60 / 120 | 125.2 / 132.0 / 157.0 | 0 / 1 / 1 |
| stretched (log σ, σ_top 1e-4) | 24 / 48 / 96 | 169.9 / 169.2 / 169.1 | 12 / 24 / 48 |

Uniform σ is not converged at 120 levels; the stretched grid is at 24. Each
preset therefore carries a stretched grid. What was *not* done: a convergence
study of the coupled 3-D model in the vertical, or any check of the stretched
grids against a stratospheric reference.

## 6. Energy: coupling closure and the hyperdiffusion finding

**Coupling closure (requirement 3, partly met).** The daily change of the
model's total energy (`global_energy`: enthalpy + kinetic + surface
potential) is compared with the flux-implied tendency accumulated by the
coupling.

| Run | flux-implied tendency | residual |
| --- | --- | --- |
| Jupiter SL09 physics, T21 L30, days 1–3 | −0.023 W m^-2 (OLR 14.05, absorbed 8.32, interior 5.7) | −2e-4 … −4e-5 W m^-2 |
| Gray dry Earth, T21 L25, days 1–30 | −134 → −9 W m^-2 | +0.07 … −0.03 W m^-2 |

Checkpoint/resume is bit-identical (tested). These are short runs with weak
flow; they show the bookkeeping is consistent, not that long, eddying runs
conserve energy.

With a stronger seed the same budget does **not** close on Jupiter: T9 L12,
1 m/s seed, one day, residual −0.058 W m^-2 with ∇⁸ against a kinetic-energy
loss of −0.059 W m^-2; with ∇⁸ off the residual is −0.003 W m^-2
(`test_with_hyperdiffusion_the_budget_residual_is_the_kinetic_energy_it_deletes`).

Startup step (fixed on 2026-11-11): up to `43cb3aa` the single RK4 startup
step applied the physics heating to the atmosphere but neither advanced the
slab nor entered the accumulated means. The coupling now advances the slab
and accumulates on every model step, so the accumulated time is the whole
run and the slab's heat change equals the accumulated surface flux to 1e-9
over a full day (tested). The long runs recorded below predate the fix; they
are missing one 1200 s slab step at day 0.

**Hyperdiffusion removes kinetic energy and does not return it as heat.**
One day, no physics, 5 m/s seeded rotational flow, T9 L12, total-energy
change as a fraction of the initial kinetic energy:

| preset | without ∇⁸ | with ∇⁸ (0.1 day at the cut) | with ∇⁸, W m^-2 |
| --- | --- | --- | --- |
| earth | −1.1 % | −2.9 % | −0.043 |
| mars | −1.0 % | −2.4 % | −0.0006 |
| titan | −0.0 % | −1.3 % | −0.20 |
| venus | −0.0 % | −1.2 % | −1.85 |
| jupiter (100 bar) | −1.7 % | **−89 %** | **−52** |

What is established: (a) switching ∇⁸ off removes the large Jupiter loss, so
∇⁸ is the sink; (b) the loss is independent of time step (600 vs 300 s), of
the RAW filter, of the vertical grid and of amplitude (5 vs 0.05 m/s give the
same fractions); (c) with Jupiter's rotation reduced 100× the flow keeps
98 % of its kinetic energy and nothing reaches degrees above 6, whereas at
full rotation 10–50 % of the kinetic energy sits above degree 6 within hours.
A deep column makes any such loss large in W m^-2 because the reservoir is
large.

What is **not** established: the mechanism in (c) (unresolved equatorial
deformation scale) is a hypothesis consistent with those three measurements;
it was not tested by a resolution sweep. Without ∇⁸, 18–27 % of the kinetic
energy still disappears from the kinetic term in a day while total energy
changes by ≤ 1.7 %. That transfer and the residual are attributed below.

Consequence: without frictional heating, a coupled energy budget shows a
residual equal to what the momentum sinks remove.

### Attribution by operator (2026-11-11, `tools/radiation/energy_attribution.py`)

The same one-day experiment, with the change of total energy split by
operator: for an operator with tendency D(X), the centred difference of
`global_energy` along D is its instantaneous energy rate, and its integral
over the run is what that operator contributed. All numbers are fractions of
the initial kinetic energy; evidence in `evidence/energy_attribution.json`.

Without ∇⁸:

| preset | kinetic change | integrated conversion by the adiabatic operator | total change by the adiabatic operator | total change, dt = 600 / 300 / 150 s | dt = 600 s, RAW filter off |
| --- | --- | --- | --- | --- | --- |
| earth | −0.267 | −0.263 | 0.0000 | −0.0110 / −0.0038 / −0.0015 | −0.0055 |
| mars | −0.177 | −0.176 | 0.0000 | −0.0096 / −0.0036 / −0.0015 | −0.0039 |
| jupiter | −0.181 | −0.169 | +0.0005 | −0.0172 / −0.0078 / −0.0036 | −0.0029 |

Two things this establishes. (i) The kinetic energy that leaves the kinetic
term is the conversion term of the spatial operator, which changes total
energy by at most 5 × 10^-4: it is transferred to enthalpy by the
discretised equations, not lost. (ii) The −1 to −2 % residual is not in the
spatial operator: it shrinks faster than linearly with the time step and by
a half (Earth) to five sixths (Jupiter) when the Robert–Asselin–Williams
filter is switched off, so it belongs to the time integration. It was not
split further between the filter, the semi-implicit scheme and the startup
step.

With ∇⁸ (0.1 day at the cut), and with `frictional_heating`:

| preset | ∇⁸ on momentum | ∇⁸ on temperature | heating returned | total change without / with heating | left to time stepping |
| --- | --- | --- | --- | --- | --- |
| earth | −0.0148 | −0.0039 | +0.0187 | −0.0293 / −0.0106 | −0.0106 |
| mars | −0.0129 | −0.0022 | +0.0151 | −0.0244 / −0.0093 | −0.0093 |
| jupiter (100 bar) | −0.777 | −0.109 | +0.886 | −0.893 / −0.0067 | −0.0069 |

The ∇⁸ sink has two parts: kinetic energy removed from vorticity and
divergence, and enthalpy removed by the diffusion of temperature along
sigma surfaces, which is not weighted by surface pressure and so changes
the mass-weighted mean (a seventh of the total on Jupiter here).

### Frictional heating (opt-in, 2026-11-11)

`frictional_heating` (configuration field and `--frictional-heating`; off by
default, and left out of the canonical configuration when off so existing
hashes are unchanged) adds `MomentumSinkHeating`
(`temporal/tendencies/drag.py`) as an explicit term: `Q = −V·dV / c_p` for
the wind tendency dV of the drag and the hyperdiffusion, plus a horizontally
uniform correction per level that cancels the enthalpy change of the
temperature diffusion. Instantaneously it returns exactly what the sinks
remove (table above: 0.886 of 0.886); over a run the balance holds to the
time-stepping error.

Two acceptance criteria were written into the tests before their first run
and were not changed:

| test | criterion | measured |
| --- | --- | --- |
| adiabatic Jupiter T9, ∇⁸ on, heating on | total change within 5 % of the ∇⁸ loss of the change without ∇⁸ | −0.0067 against −0.0172 without ∇⁸ and −0.893 with |
| coupled Jupiter T9, ∇⁸ on, heating on | residual below the 5 × 10^-3 W m^-2 bound the coupling meets with ∇⁸ off | 5.7 × 10^-4 W m^-2 (0.058 without heating) |

The first version of the term failed both (coupled residual 0.016 W m^-2):
it was a lagged term evaluated at the old time level and did not return the
temperature-diffusion part. The attribution above located both defects; the
criteria were kept.

Limits: the local heating is not sign-definite for a scale-selective
diffusion; the time-stepping residual is not returned; the option has been
exercised on the two gray presets only.

### Long runs completed on 2026-11-11 (commit `b2b8ea2`, MX110)

Both presets were run to completion by the repository owner from a clean
tree. Records, a summary and a figure for each are in
`evidence/earth-gray-dry-T21L25-300d/` and `evidence/jupiter-sl09-T21L30-200d/`
(`summary.json` is produced by `tools/radiation/summarize_run.py`). These
are gray-radiation runs at T21; neither is a reproduction of a published
benchmark.

**Dry gray Earth, T21 L25, 300 days.** The flow spins up between days 30 and
100 (kinetic energy 0.05 → 1.6 MJ m^-2) and is statistically steady
afterwards. Means over days 201–300 (± standard deviation of daily values):

| quantity | value |
| --- | --- |
| absorbed stellar / OLR / TOA net down | 234.60 / 230.38 ± 0.60 / +4.22 W m^-2 |
| sensible heat flux, surface net | 126.6 ± 5.6, −0.1 ± 5.2 W m^-2 |
| mean surface temperature | 301.39 ± 0.19 K |
| kinetic energy, max \|u\| | 1.38 ± 0.06 MJ m^-2, 64 ± 5 m s^-1 |
| energy-budget residual | **−5.10 ± 0.42 W m^-2** (largest single day: −12.1 on day 62) |
| mean surface pressure drift over 300 days | −1.1 × 10^-6 |

Zonal means over days 151–300 (150 daily samples): westerly jets of 28.2 and
27.4 m s^-1 at 19°N and 25°S, σ = 0.34; lowest-level temperature 310 K at the
equator and 241–242 K at the poles; lowest-level eddy heat flux peaking at
38 K m s^-1 near 42–47° in both hemispheres. The two hemispheres differ by up
to 16 % of the peak wind in this 150-day mean, so it is not a converged
climatology. The model has no boundary-layer diffusion or surface stress, so
these numbers are not comparable with Frierson et al. (2006).

**Where the Earth residual goes (measured).** For each of the 11 saved states
of days 200–300 the tool evaluates the rate at which each momentum sink
changes `global_energy`, as a centred difference along that operator's own
tendency:

| sink | kinetic | enthalpy | total (W m^-2) |
| --- | --- | --- | --- |
| Rayleigh drag (σ > 0.7, 1 day^-1) | −2.42 | 0 | −2.42 |
| ∇⁸ hyperdiffusion (all prognostic rows) | −2.70 | +0.15 | −2.54 |
| both | | | **−4.96 ± 0.12** (standard error of 11 states) |

The residual over the same 100 days is −5.10 W m^-2, leaving −0.14 W m^-2
unaccounted, about one standard error of the sampled sinks. So in this run
the budget residual is the kinetic energy that drag and ∇⁸ remove and the
model does not return as heat, to within the sampling of that measurement.
Two things follow directly from the numbers: the steady state balances at the
top of the atmosphere with +4.2 W m^-2 net absorption (OLR is lower than the
absorbed flux by about the size of the sink), and the sink is 2 % of the
radiative throughput. Limits of this statement: the sinks are sampled every
10 days against a daily-mean residual; the hyperdiffusion rate is the
continuous `-r X` form rather than the stepper's centred factor; no
term-by-term kinetic-energy budget (conversion, advection) was computed, so
the −1 to −2 % adiabatic residual of the table above is still unexplained.

**Jupiter gray (Schneider–Liu parameters), T21 L30 uniform σ to 3 bar, 200
days.** The run is stable and its budget closes: residual −1.3 × 10^-6 W m^-2
against measured sinks of −1.1 × 10^-6 W m^-2, OLR 14.047 W m^-2 against
8.324 absorbed + 5.7 interior (a steady −0.023 W m^-2 imbalance present from
day 1), surface-pressure drift −1.9 × 10^-10. **It has not spun up a
circulation**: after 200 days max \|u\| is 1.0 m s^-1 and kinetic energy is
still growing (16 J m^-2 at day 20, 89 at day 100, 321 at day 200); the
100-day zonal mean shows 0.8 m s^-1 at 64° in the top level and no jets. This
is a long stability and bookkeeping check of the surface-less configuration,
nothing more. The uniform grid puts one level above the radiative-convective
tropopause, so the stratosphere is not resolved. Because the flow is far
weaker than in the T9 experiment above (1 against 5 m s^-1, in 3 against
100 bar), this run says nothing about the ∇⁸ loss at realistic Jovian wind
speeds.

Measured speed over the whole runs: 0.50 s/step (Earth, 21,600 steps) and
0.55 s/step (Jupiter, 14,400 steps).

### Controlled comparison: the same Earth run with frictional heating

Criterion written before the run was started (commit that adds
`frictional_heating`): the 300-day dry gray Earth preset is repeated with
`--frictional-heating` and nothing else changed in the configuration. It
passes if the mean energy-budget residual over days 201–300 is below
0.5 W m^-2 in magnitude (a tenth of the 5.10 W m^-2 of the run without it).
The top-of-atmosphere balance, surface temperature and jets are reported
next to the first run's, not judged. The code also differs from the first
run's by the startup-step fix (the slab is advanced one extra step at the
start), which is not expected to matter after spin-up.

**Result (run completed 2026-11-12, commit `d38d2a3`, MX110, configuration
`748cec4b…`; evidence in `evidence/earth-gray-dry-heated-T21L25-300d/`).**
The criterion is met: the mean residual over days 201–300 is
−0.002 W m^-2.

| days 201–300, mean ± standard deviation of daily values | without heating (`b2b8ea2`) | with heating (`d38d2a3`) |
| --- | --- | --- |
| energy-budget residual, W m^-2 | −5.10 ± 0.42 | −0.002 ± 0.025 |
| largest single-day residual in the whole run, W m^-2 | 12.1 | 0.37 |
| TOA net down, W m^-2 | +4.22 ± 0.60 | −0.59 ± 0.55 |
| OLR, W m^-2 (absorbed 234.60) | 230.38 | 235.19 |
| mean surface temperature, K | 301.39 ± 0.19 | 302.65 ± 0.11 |
| kinetic energy, MJ m^-2 | 1.378 ± 0.058 | 1.376 ± 0.059 |
| max \|u\|, m s^-1 | 64 ± 5 | 62 ± 5 |
| westerly jets, north / south (150-day zonal mean), m s^-1 | 28.2 at 19° / 27.4 at 25° | 26.0 at 25° / 28.0 at 25° |
| surface-pressure drift over 300 days | −1.1 × 10^-6 | −1.2 × 10^-6 |

With the heating the budget closes, the top-of-atmosphere imbalance of
+4.2 W m^-2 is replaced by −0.6 W m^-2 (about one standard deviation of the
daily values from zero), and the surface is 1.3 K warmer. Kinetic energy and
jet strength are the same within their day-to-day spread. Not done for this
run: the per-state measurement of the sinks and of the heating
(`summarize_run.py` was run with `--no-sinks`). The comparison is with one
realisation of each run, and it is not strictly single-variable because of
the startup-step fix noted above.

This run used the per-level (unbatched) heating term. The batched version
committed afterwards (`814bcf8`) is equivalent to round-off but has not been
used for a long run.

## 7. Published comparisons: verified, and where each stands

All five references were read (protocol numbers in the docstrings cite them).

| Reference | Establishes | Needs beyond what exists | Status |
| --- | --- | --- | --- |
| Schneider & Liu 2009 (Jupiter, gray, interior heat, T213) | coupled gray radiation + convection + drag without a surface | T213 and > 28,000 days; drag shoulder shape unspecified in the paper | column structure and energy balance reproduced (§4). A 200-day T21 run completed (§6) with a closed budget but winds below 1 m s^-1: **no jets, not a reproduction**. Their runs are T213 and tens of thousands of days; T21 cannot resolve their jets in any case. |
| Frierson, Held & Zurita-Gotor 2006 (dry limit) | gray radiation + slab + boundary layer with eddies | their K-profile boundary layer, surface stress, T85+ | radiation, slab and bulk flux implemented; **boundary-layer diffusion and surface stress are not**, so no dry-limit comparison is claimed. A 300-day T21 run completed (§6; `evidence/earth-gray-dry-T21L25-300d/`) and shows a steady eddying circulation, but it is a different model from theirs and is not a comparison. The earlier record `evidence/earth-gray-dry-T21L25-partial/` is the first 30 days of an earlier development run made before the first commit of this branch. |
| Isca Mars M-I (gray), M-C (SOCRATES) | seasonal/diurnal Mars circulation, figures only | seasons + diurnal cycle are implemented in `insolation.py` but not run; M-C needs real-gas CO2 with CIA and topography | proposed |
| THAI Ben 1 / Ben 2 | N2 vs CO2 real-gas, tidally locked; inter-model spread 7 K / 6 K in surface temperature, fluxes within 5 % | real-gas RT with CO2 CIA and far wings, boundary layer, soil | proposed; blocked on opacity data |
| HARP Jupiter | real-gas column heating/cooling; their correlated-k vs line-by-line agree within 10 % | HITRAN lines, H2–H2/H2–He CIA, Moses profiles | proposed; blocked on opacity data |

## 8. External software and execution strategy

Investigated, not integrated: pyRTE-RRTMGP (conda for Linux/macOS only;
Earth-only gas optics), SOCRATES (Fortran, Unix build, BSD-3; has Mars
spectral files via ROCKE-3D), petitRADTRANS 3 (needs a Fortran toolchain on
Windows; opacities downloaded on demand), exo_k (pure Python; reads ExoMol and
other k-tables; the most direct source of real tables for `KTable`). None was
installed: no package or opacity file was downloaded in this session. The
architecture does not depend on that choice: a real table enters as a
`KTable` and an external model enters as an independent reference in tests.

Performance (MX110, 2 GB): T21 L30 Jupiter 0.52–0.56 s/step, of which one
dynamics tendency 0.37 s and one gray physics call 0.10 s; T21 L25 Earth
0.50 s/step. No real-gas timing exists, so **the value of neural
acceleration is unmeasured**; with gray radiation the physics is ~18 % of a
step and `radiation_interval` already reduces it.

## 9. Reproduce

```bash
python -m pytest tests/test_radiation_twostream.py tests/test_physics_columns.py tests/test_radiation_kdist.py tests/test_radiative_dynamical.py
python tools/radiation/regimes.py
python tools/radiation/adiabatic_energy.py
python tools/radiation/energy_attribution.py
python -m tropoi.run.radiative column --preset jupiter-schneider-liu
python -m tropoi.run.radiative run --preset jupiter-schneider-liu --days 200 --run-dir runs/radiative/jupiter-sl09-T21L30
python -m tropoi.run.radiative run --preset earth-gray-dry --days 300 --run-dir runs/radiative/earth-gray-dry-T21L25
python tools/radiation/summarize_run.py runs/radiative/earth-gray-dry-T21L25 --name earth-gray-dry-T21L25-300d
python tools/radiation/summarize_run.py runs/radiative/jupiter-sl09-T21L30 --name jupiter-sl09-T21L30-200d
```

The committed evidence was produced from run directories suffixed `-b2b8ea2`
(config hashes `e31ae183…` Earth, `80f280d2…` Jupiter; full values in each
`config.json`). Saved states and checkpoints are not committed.

Dependencies are unchanged (`requirements.txt`; SciPy is used by the tests).
Run configurations are hashed (`config_sha256`) and stored with the commit in
each run's `config.json`.

Test results for this commit are in [TEST_RESULTS.md](TEST_RESULTS.md).

## 10. Next steps, in order

1. Obtain real k-tables (needs a decision on downloads: exo_k + ExoMol/HITRAN
   CIA) and compare `CorrelatedKLongwave` with an independent model for one
   Earth and one CO2 case; add Rayleigh, continua, CIA, gas overlap.
2. Decide whether `frictional_heating` becomes the default of the presets
   (§6). A resolution sweep for the Jupiter ∇⁸ loss; split the time-stepping
   residual between the filter and the semi-implicit scheme.
3. A Jupiter run long and fine enough to develop a circulation (the 200-day
   T21 run does not), on a stretched grid that resolves the stratosphere.
4. Boundary-layer diffusion and surface stress (required by FHZ06 and THAI).
5. Variable c_p(T) (Venus, Mars, giant planets); assess a deep-atmosphere
   core for Titan.
