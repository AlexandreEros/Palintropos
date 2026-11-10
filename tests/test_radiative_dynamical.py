"""The dynamical core in non-Earth regimes, and its coupling to column physics.

Three separate questions, kept separate on purpose:

* does the dry core behave on each planet's constants and stretched grid
  with NO radiation (rest stays at rest, the semi-implicit operator is a
  valid linearization, adiabatic energy is conserved to a measured level)?
* does the lagged physics hook leave an unforced run untouched and is a
  coupled run reproducible across a checkpoint?
* does the coupled model's energy change equal what the fluxes say?

Host-only tests (configuration, single column) run on CPU CI; the rest need
CUDA and skip without it.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from tropoi.physics.planets import PLANETS
from tropoi.run.radiative.config import PRESETS, RadiativeConfig
from tropoi.run.radiative.model import (build_physics, build_sigma, reference_column)


def _has_cuda() -> bool:
    try:
        import cupy
        return bool(cupy.is_available())
    except Exception:
        return False


needs_cuda = pytest.mark.skipif(not _has_cuda(), reason="needs CuPy and a CUDA GPU")


def _tiny(cfg: RadiativeConfig, **changes) -> RadiativeConfig:
    """A configuration small enough for a test (evolves T9 on 16 x 32)."""
    return cfg.with_(l_max=10, retained_truncation=9, nlat=16, nlon=32, **changes)


def _planet_config(name: str, **changes) -> RadiativeConfig:
    """An unforced configuration on a planet's constants and stretched grid."""
    body = PLANETS[name]
    spec = body.vertical_grid
    return _tiny(RadiativeConfig(
        experiment_id=f"{name}-regime", radius=body.radius, omega=body.rotation_rate,
        gravity=body.gravity, gas_constant=body.gas_constant, cp=body.cp,
        reference_pressure=body.reference_pressure, l_max=10, retained_truncation=9,
        nlat=16, nlon=32,
        vertical={"kind": "stretched", "nlev": 12, "sigma_top": spec.sigma_top,
                  "linear_scale": spec.linear_scale},
        dt=600.0, days=1, t_ref=body.temperature_range[1],
        longwave={"kind": "power_law", "tau0": 1.0},
        insolation={"kind": "uniform", "flux": 100.0}, shortwave={"kind": "transparent"},
        surface={"heat_capacity": 1.0e7},
        initial={"kind": "isothermal", "temperature": body.reference_temperature}), **changes)


# ---------------------------------------------------------------- host only
def test_presets_are_valid_hashable_and_round_trip():
    seen = set()
    for name, factory in PRESETS.items():
        cfg = factory()
        again = RadiativeConfig.from_dict(json.loads(json.dumps(cfg.to_dict())))
        assert again == cfg and again.sha256() == cfg.sha256()
        assert cfg.with_(dt=600.0).sha256() != cfg.sha256()
        seen.add(cfg.sha256())
        assert build_sigma(cfg).nlev == cfg.vertical["nlev"]
        assert build_physics(cfg).signature()["kind"] == "column_physics"
    assert len(seen) == len(PRESETS)
    with pytest.raises(ValueError):
        PRESETS["earth-gray-dry"]().with_(dt=7.0)
    misspelt = PRESETS["earth-gray-dry"]().with_(longwave={"kind": "frierson",
                                                             "tau_equatr": 6.0})
    with pytest.raises(ValueError, match="unknown keys"):
        build_physics(misspelt)


def test_reference_column_of_the_jupiter_preset_is_in_energy_balance():
    ref = reference_column(PRESETS["jupiter-schneider-liu"](), tolerance=1e-3)
    assert ref["converged"]
    assert ref["olr"] == pytest.approx(ref["absorbed_stellar"] + 5.7, abs=0.05)
    assert ref["surface_temperature"] is None


# ---------------------------------------------------------------- GPU
@needs_cuda
@pytest.mark.parametrize("planet", ["mars", "venus", "titan", "jupiter"])
def test_rest_stays_at_rest_on_each_planet_and_its_stretched_grid(planet):
    """Isothermal rest must have an exactly zero tendency whatever the gas
    constants, radius, rotation and vertical grid (the discrete hydrostatic
    and pressure-gradient terms cancel identically)."""
    import cupy as cp
    from tropoi.run.radiative.model import build_model
    from tropoi.spatial.states.primitive_equations import isothermal_rest_state
    cfg = _planet_config(planet)
    model = build_model(cfg)
    x = isothermal_rest_state(cfg.l_max, model.nlev,
                              temperature=PLANETS[planet].reference_temperature,
                              surface_pressure=cfg.reference_pressure).coeffs
    tend = model.tendency(x)
    scale = float(cp.abs(x[2 * model.nlev]).max()) / 86400.0          # K per day
    assert float(cp.abs(tend).max()) <= 1e-9 * scale


@needs_cuda
@pytest.mark.parametrize("planet", ["earth", "mars", "jupiter"])
def test_adiabatic_energy_is_conserved_up_to_what_hyperdiffusion_removes(planet):
    """A resting, stratified column with a seeded 5 m/s rotational flow, no
    physics, one day (tools/radiation/adiabatic_energy.py is the same
    experiment for every preset; docs/radiation/README.md, "Energy").

    The bounds below are REGRESSION ENVELOPES around measurements made on
    2026-11-10 (MX110, T9 L12). They are not acceptance criteria and were
    not derived from a requirement; they exist so a change in this
    behaviour is noticed.

    Measured, total energy change over one day as a fraction of the initial
    kinetic energy: without hyperdiffusion Earth -1.1 %, Mars -1.0 %,
    Jupiter -1.7 % (the cause of this residual has not been identified);
    with del^8 hyperdiffusion Earth -2.9 %, Mars -2.4 %, Jupiter -89 %
    (52 W m^-2 for a 5 m/s flow in a 100-bar column). Hyperdiffusion removes
    kinetic energy and does not return it as heat, so it is a pure sink of
    total energy.

    Not established: where the kinetic energy that disappears WITHOUT
    hyperdiffusion (18-27 % in a day) goes. The enthalpy term of the
    diagnostic rises by nearly the same amount, which is consistent with
    conversion to potential/internal energy, but no term-by-term transfer
    (pressure work against the kinetic-energy tendency) was computed."""
    import cupy as cp
    from tropoi.run.radiative.model import build_model, build_stepper, initial_state
    from tropoi.temporal.column_coupling import global_energy
    body = PLANETS[planet]
    change = {}
    for label, efold in (("conservative", 1.0e9), ("hyperdiffusion", 0.1)):
        cfg = _planet_config(planet, perturbation_wind=5.0, perturbation_lmax=4,
                             hyperdiffusion_efold_days=efold)
        model = build_model(cfg)
        x0, _, _ = initial_state(cfg)
        K = model.nlev
        profile = body.reference_temperature * model.sigma.full_levels_array() ** (0.5 * cfg.kappa)
        x0[2 * K:3 * K, 0, 0] = cp.asarray(np.maximum(profile, 0.4 * body.reference_temperature))             * math.sqrt(4.0 * math.pi)
        stepper = build_stepper(cfg, model, None)
        stepper.initialize(x0, 0.0)
        e0 = global_energy(model, x0, cfg.gravity)
        for _ in range(cfg.steps_per_day):
            stepper.step()
        e1 = global_energy(model, stepper.state, cfg.gravity)
        assert bool(cp.isfinite(stepper.state).all())
        assert e1["mean_surface_pressure"] == pytest.approx(e0["mean_surface_pressure"], rel=2e-6)
        change[label] = (e1["total"] - e0["total"]) / e0["kinetic"]
    assert abs(change["conservative"]) < 0.03, change
    assert change["hyperdiffusion"] <= change["conservative"] + 1e-6      # only ever a sink
    if planet == "jupiter":
        assert -0.95 < change["hyperdiffusion"] < -0.8, change
    else:
        assert abs(change["hyperdiffusion"]) < 0.05, change


@needs_cuda
def test_lagged_terms_default_changes_nothing_and_a_zero_term_is_exact():
    import cupy as cp
    from tropoi.run.radiative.model import build_model, build_stepper, initial_state
    cfg = _planet_config("earth", perturbation_wind=3.0)
    model = build_model(cfg)
    x0, _, _ = initial_state(cfg)

    class Zero:
        calls = []

        def signature(self):
            return {"kind": "zero"}

        def __call__(self, x, t, window):
            Zero.calls.append((t, window))
            return cp.zeros_like(x)

    plain = build_stepper(cfg, model, None)
    hooked = build_stepper(cfg, model, Zero())
    plain.initialize(x0, 0.0)
    hooked.initialize(x0, 0.0)
    for _ in range(4):
        plain.step()
        hooked.step()
    assert bool((plain.state == hooked.state).all())
    # startup is called with the old level and a window of dt, leapfrog with 2 dt
    assert Zero.calls[0] == (0.0, cfg.dt)
    assert [w for _, w in Zero.calls[1:]] == [2 * cfg.dt] * 3
    assert [t for t, _ in Zero.calls[1:]] == [0.0, cfg.dt, 2 * cfg.dt]
    assert hooked.state_dict()["hooks"][-1] == {"role": "lagged", "kind": "zero"}
    assert all(h["role"] != "lagged" for h in plain.state_dict()["hooks"])


@needs_cuda
def test_coupled_energy_budget_closes_and_a_resumed_run_is_identical(tmp_path):
    """Jupiter physics at T9 L12 for two days, hyperdiffusion OFF so that the
    residual isolates the coupling: (i) the daily change of the model's
    total energy equals the flux-implied tendency accumulated by the
    coupling (bound 5e-3 W m^-2, set before the first run of this test;
    measured 3e-3); (ii) stopping after day 1 and resuming gives the same
    bits as an uninterrupted run. The hyperdiffusion sink is asserted
    separately below."""
    from tropoi.run.radiative.experiment import RadiativeRun
    cfg = _tiny(PRESETS["jupiter-schneider-liu"](days=2), experiment_id="jup-test",
                vertical={"kind": "uniform", "nlev": 12}, checkpoint_every_days=1,
                state_every_days=1, perturbation_wind=1.0, hyperdiffusion_efold_days=1.0e9)
    full = RadiativeRun(cfg, tmp_path / "full", allow_dirty=True, log=lambda s: None)
    assert full.run()["status"] == "completed"
    series = json.loads((tmp_path / "full" / "series.json").read_text())
    assert max(abs(r) for r in series["energy_residual"]) < 5e-3
    assert series["olr"][-1] == pytest.approx(14.0, abs=0.5)
    assert series["bottom_net_down"][-1] == pytest.approx(0.0, abs=1e-10)
    assert series["interior"][-1] == pytest.approx(5.7)

    part = RadiativeRun(cfg, tmp_path / "split", allow_dirty=True, log=lambda s: None)
    part.run(until_day=1)
    resumed = RadiativeRun(cfg, tmp_path / "split", allow_dirty=True, log=lambda s: None)
    assert resumed.run()["status"] == "completed"
    a = np.load(tmp_path / "full" / "states" / "state-d00002.npy")
    b = np.load(tmp_path / "split" / "states" / "state-d00002.npy")
    assert np.array_equal(a, b)


@needs_cuda
def test_with_hyperdiffusion_the_budget_residual_is_the_kinetic_energy_it_deletes(tmp_path):
    """Same run, one day, del^8 on (0.1 day at the cut). Measured: residual
    -0.0578 W m^-2, kinetic-energy change -0.0591 W m^-2 (89 % of a 1 m/s
    seed). The coupled budget does not close on a fast rotator at this
    resolution, by the amount hyperdiffusion removes; nothing returns it as
    heat. This test records that, it does not excuse it."""
    from tropoi.run.radiative.experiment import RadiativeRun
    cfg = _tiny(PRESETS["jupiter-schneider-liu"](days=1), experiment_id="jup-hyper",
                vertical={"kind": "uniform", "nlev": 12}, checkpoint_every_days=1,
                state_every_days=1, perturbation_wind=1.0)
    RadiativeRun(cfg, tmp_path / "run", allow_dirty=True, log=lambda s: None).run()
    series = json.loads((tmp_path / "run" / "series.json").read_text())
    kinetic_rate = (series["energy_kinetic"][1] - series["energy_kinetic"][0]) / 86400.0
    assert kinetic_rate < -0.04
    assert series["energy_residual"][-1] == pytest.approx(kinetic_rate, rel=0.1)


@needs_cuda
def test_surface_exchange_conserves_energy_between_surface_and_atmosphere(tmp_path):
    """Gray dry Earth for one day: what leaves the slab enters the
    atmosphere. The atmosphere's residual and the slab's heat content change
    are both checked against the accumulated fluxes."""
    from tropoi.run.radiative.experiment import RadiativeRun
    cfg = _tiny(PRESETS["earth-gray-dry"](days=1), experiment_id="earth-test",
                vertical={"kind": "frierson", "nlev": 10}, checkpoint_every_days=1,
                state_every_days=1)
    run = RadiativeRun(cfg, tmp_path / "earth", allow_dirty=True, log=lambda s: None)
    ts0 = float((run.coupling.weights * run.coupling.surface_temperature).sum())
    run.run()
    series = json.loads((tmp_path / "earth" / "series.json").read_text())
    assert abs(series["energy_residual"][-1]) < 0.02 * abs(series["column_energy_tendency"][-1])
    # The accumulator covers the leapfrog steps only: the one RK4 startup
    # step neither advances the slab nor enters the means (its heating of
    # the atmosphere over that single step is therefore unaccounted, once
    # per run).
    accumulated_seconds = 86400.0 - cfg.dt
    slab_heat = cfg.surface["heat_capacity"] * (series["mean_surface_temperature"][-1] - ts0)
    assert slab_heat == pytest.approx(series["surface_net"][-1] * accumulated_seconds, rel=1e-9)
    assert series["sensible"][-1] > 0.0          # the warm slab heats the air
