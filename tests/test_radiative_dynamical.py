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
import pathlib

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


EVIDENCE = pathlib.Path(__file__).resolve().parents[1] / "docs" / "radiation" / "evidence"

# (preset, evidence run with frictional heating, evidence run without, their hashes)
PRESET_IDENTITIES = [
    ("earth-gray-dry", "earth-gray-dry-heated-T21L25-300d", "earth-gray-dry-T21L25-300d",
     "748cec4b", "e31ae183937da3bc9c5a63cbaf5c889bbeefc384a2f70b81e529b7a6965890d5"),
    ("jupiter-schneider-liu", "jupiter-sl09-heated-T21L30-200d-colab",
     "jupiter-sl09-T21L30-200d",
     "0d029212", "80f280d2005fd86a2415f29dabbd90e126e75232b98bbe203b88fd913333d189"),
]


def _recorded(run: str) -> dict:
    return json.loads((EVIDENCE / run / "config.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("preset, heated, unheated, heated_prefix, unheated_digest",
                         PRESET_IDENTITIES)
def test_presets_default_to_frictional_heating_and_name_the_recorded_runs(
        preset, heated, unheated, heated_prefix, unheated_digest):
    """An eddying run without frictional heating fails the energy criteria
    A2/A3 (README §6), so the presets turn it on. Both formulations stay
    reachable by name and both keep the identity of the evidence already
    recorded for them: the heated default is the configuration of the
    committed heated run, and ``frictional_heating=False`` is the
    configuration of the committed unheated run, hash for hash."""
    on = PRESETS[preset]()
    assert on.frictional_heating is True
    assert on.sha256() == _recorded(heated)["config_sha256"]
    assert on.sha256().startswith(heated_prefix)
    assert on.to_dict() == _recorded(heated)["config"]

    off = PRESETS[preset](frictional_heating=False)
    assert off.frictional_heating is False
    assert "frictional_heating" not in off.to_dict()
    assert off.sha256() == _recorded(unheated)["config_sha256"] == unheated_digest
    assert RadiativeConfig.from_dict(json.loads(on.canonical_json())) == on
    assert RadiativeConfig.from_dict(json.loads(off.canonical_json())) == off


def test_a_configuration_written_before_the_default_changed_still_means_unheated():
    """Configurations recorded before the option existed have no
    ``frictional_heating`` key. The dataclass default stays False, so such a
    file still loads as the unheated experiment with its recorded hash (and
    an existing run directory keeps resuming as what it is)."""
    for run in ("earth-gray-dry-T21L25-300d", "jupiter-sl09-T21L30-200d"):
        doc = _recorded(run)
        assert "frictional_heating" not in doc["config"]
        cfg = RadiativeConfig.from_dict(doc["config"])
        assert cfg.frictional_heating is False
        assert cfg.sha256() == doc["config_sha256"]


def test_an_explicit_experiment_id_is_kept_whatever_the_heating():
    for preset in PRESETS:
        for heating in (True, False):
            cfg = PRESETS[preset]("my-run", frictional_heating=heating)
            assert cfg.experiment_id == "my-run"
            assert cfg.frictional_heating is heating


@pytest.mark.parametrize("preset, heated, unheated, heated_prefix, unheated_digest",
                         PRESET_IDENTITIES)
def test_the_command_line_resolves_the_heating_flags(capsys, tmp_path, preset, heated, unheated,
                                                     heated_prefix, unheated_digest):
    from tropoi.run.radiative.__main__ import main

    def digest(*argv) -> str:
        assert main(["show-config", *argv]) == 0
        return json.loads(capsys.readouterr().out)["config_sha256"]

    heated_digest = _recorded(heated)["config_sha256"]
    assert digest("--preset", preset) == heated_digest
    assert digest("--preset", preset, "--frictional-heating") == heated_digest
    assert digest("--preset", preset, "--no-frictional-heating") == unheated_digest
    # a configuration file means what it says unless a flag overrides it
    files = {}
    for run in (heated, unheated):
        files[run] = tmp_path / f"{run}.json"
        files[run].write_text(json.dumps(_recorded(run)["config"]), encoding="utf-8")
    assert digest("--config", str(files[unheated])) == unheated_digest
    assert digest("--config", str(files[heated])) == heated_digest
    assert digest("--config", str(files[heated]), "--no-frictional-heating") != heated_digest


def test_radiation_interval_must_divide_a_day_so_that_resume_is_exact():
    """Held radiative fluxes are not checkpointed; runs stop at whole days.
    72 steps a day with an interval of 5 would resume off the schedule."""
    cfg = PRESETS["earth-gray-dry"]()
    assert cfg.steps_per_day == 72
    assert cfg.with_(radiation_interval=4).radiation_interval == 4
    with pytest.raises(ValueError, match="must divide"):
        cfg.with_(radiation_interval=5)


# ---------------------------------------------------------------- host only
def test_presets_are_valid_hashable_and_round_trip():
    seen = set()
    for name, factory in PRESETS.items():
        cfg = factory()
        again = RadiativeConfig.from_dict(json.loads(json.dumps(cfg.to_dict())))
        assert again == cfg and again.sha256() == cfg.sha256()
        assert cfg.with_(dt=2.0 * cfg.dt).sha256() != cfg.sha256()
        seen.add(cfg.sha256())
        expected = cfg.vertical.get("nlev", len(cfg.vertical.get("sigma", ())) - 1)
        assert build_sigma(cfg).nlev == expected
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
                state_every_days=1, perturbation_wind=1.0, hyperdiffusion_efold_days=1.0e9,
                frictional_heating=False)
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
                state_every_days=1, perturbation_wind=1.0, frictional_heating=False)
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
                state_every_days=1, frictional_heating=False)
    run = RadiativeRun(cfg, tmp_path / "earth", allow_dirty=True, log=lambda s: None)
    ts0 = float((run.coupling.weights * run.coupling.surface_temperature).sum())
    run.run()
    series = json.loads((tmp_path / "earth" / "series.json").read_text())
    assert abs(series["energy_residual"][-1]) < 0.02 * abs(series["column_energy_tendency"][-1])
    # Every model step, the RK4 startup step included, advances the slab and
    # enters the means, so the accumulated time is the whole day.
    accumulated_seconds = 86400.0
    assert run.coupling.budget_time == 0.0       # reset by the daily sample
    slab_heat = cfg.surface["heat_capacity"] * (series["mean_surface_temperature"][-1] - ts0)
    assert slab_heat == pytest.approx(series["surface_net"][-1] * accumulated_seconds, rel=1e-9)
    assert series["sensible"][-1] > 0.0          # the warm slab heats the air



@needs_cuda
def test_frictional_heating_returns_what_hyperdiffusion_removes_adiabatic():
    """Jupiter constants, T9 L12, 5 m/s seed, one day, no physics. Without
    del^8 the total energy changes by dE0; with del^8 by dE8 (89 % of the
    kinetic energy, the documented sink). Criterion, fixed before the first
    run: with frictional heating the change differs from dE0 by less than
    5 % of |dE8|, i.e. at least 95 % of the deleted energy comes back."""
    import cupy as cp
    from tropoi.run.radiative.model import build_model, build_stepper, initial_state
    from tropoi.temporal.column_coupling import global_energy
    body = PLANETS["jupiter"]

    def change(**options):
        cfg = _planet_config("jupiter", perturbation_wind=5.0, perturbation_lmax=4, **options)
        model = build_model(cfg)
        x0, _, _ = initial_state(cfg)
        K = model.nlev
        profile = body.reference_temperature * model.sigma.full_levels_array() ** (0.5 * cfg.kappa)
        x0[2 * K:3 * K, 0, 0] = cp.asarray(np.maximum(profile, 0.4 * body.reference_temperature)) \
            * math.sqrt(4.0 * math.pi)
        stepper = build_stepper(cfg, model, None)
        stepper.initialize(x0, 0.0)
        e0 = global_energy(model, x0, cfg.gravity)
        for _ in range(cfg.steps_per_day):
            stepper.step()
        e1 = global_energy(model, stepper.state, cfg.gravity)
        return e1["total"] - e0["total"], e0["kinetic"]

    d_none, kinetic = change(hyperdiffusion_efold_days=1.0e9)
    d_sink, _ = change()
    d_heat, _ = change(frictional_heating=True)
    assert d_sink < -0.5 * kinetic                       # the sink is there
    assert abs(d_heat - d_none) < 0.05 * abs(d_sink)


@needs_cuda
def test_with_frictional_heating_the_coupled_budget_closes_with_hyperdiffusion_on(tmp_path):
    """The run of the del^8 test above with frictional heating on. Criterion,
    fixed before the first run: the same 5e-3 W m^-2 bound the coupling
    meets with del^8 off (the residual without the heating is 0.058)."""
    from tropoi.run.radiative.experiment import RadiativeRun
    cfg = _tiny(PRESETS["jupiter-schneider-liu"](days=1), experiment_id="jup-heat",
                vertical={"kind": "uniform", "nlev": 12}, checkpoint_every_days=1,
                state_every_days=1, perturbation_wind=1.0, frictional_heating=True)
    RadiativeRun(cfg, tmp_path / "run", allow_dirty=True, log=lambda s: None).run()
    series = json.loads((tmp_path / "run" / "series.json").read_text())
    print("residual", series["energy_residual"])
    assert abs(series["energy_residual"][-1]) < 5e-3


@needs_cuda
def test_a_resumed_run_with_held_radiation_is_identical(tmp_path):
    """radiation_interval = 4 (fluxes held for three of every four steps),
    stopped after day 1 and resumed: same bits as the uninterrupted run."""
    from tropoi.run.radiative.experiment import RadiativeRun
    cfg = _tiny(PRESETS["earth-gray-dry"](days=2), experiment_id="earth-held",
                vertical={"kind": "frierson", "nlev": 10}, checkpoint_every_days=1,
                state_every_days=1, radiation_interval=4, frictional_heating=False)
    RadiativeRun(cfg, tmp_path / "full", allow_dirty=True, log=lambda s: None).run()
    RadiativeRun(cfg, tmp_path / "split", allow_dirty=True, log=lambda s: None).run(until_day=1)
    resumed = RadiativeRun(cfg, tmp_path / "split", allow_dirty=True, log=lambda s: None)
    assert resumed.run()["status"] == "completed"
    for name in ("state", "surface"):
        a = np.load(tmp_path / "full" / "states" / f"{name}-d00002.npy")
        b = np.load(tmp_path / "split" / "states" / f"{name}-d00002.npy")
        assert np.array_equal(a, b)
    series = json.loads((tmp_path / "full" / "series.json").read_text())
    assert abs(series["energy_residual"][-1]) < 0.02 * abs(series["column_energy_tendency"][-1])



ISCA_GREY_MARS_BK = [  # Isca exp/test_cases/grey_mars, commit d1321ac3
    0.0, 1.53008955e-04, 4.63790800e-04, 1.10977640e-03, 2.37044150e-03, 4.74479200e-03,
    9.12245300e-03, 1.70677050e-02, 3.12516100e-02, 5.59939500e-02, 9.76165000e-02,
    1.63754900e-01, 2.60315150e-01, 3.85974250e-01, 5.28084800e-01, 6.65956600e-01,
    7.81088000e-01, 8.65400050e-01, 9.21109250e-01, 9.55343500e-01, 9.75416950e-01,
    9.86856800e-01, 9.93269300e-01, 9.96830200e-01, 9.98799150e-01, 1.0]


def test_the_flat_mars_development_preset_matches_m_i_where_it_claims_to():
    """docs/radiation/MARS_MI_PLAN.md section 1: Table 1 of Thomson & Vallis
    (2019) and the Isca values it leaves implicit. The two substitutes for
    the missing boundary layer are present and are not hidden."""
    from tropoi.physics.radiation.gray import VerticalPathShortwave
    from tropoi.run.radiative.model import build_insolation
    cfg = PRESETS["dev-mars-gray-flat"]()
    assert cfg.experiment_id == "dev-mars-gray-flat-heated-T21L25"
    assert PRESETS["dev-mars-gray-flat"](frictional_heating=False).experiment_id == \
        "dev-mars-gray-flat-T21L25"
    assert cfg.frictional_heating is True
    assert (cfg.radius, cfg.gravity, cfg.gas_constant, cfg.reference_pressure) == \
        (3.396e6, 3.71, 192.0, 610.0)
    assert cfg.kappa == pytest.approx(0.22727, rel=1e-12)
    assert cfg.omega == pytest.approx(2.0 * math.pi / 88308.0, rel=1e-15)
    assert cfg.omega == pytest.approx(7.12e-5, rel=1e-3)                      # Table 1
    assert cfg.surface["heat_capacity"] == pytest.approx(2.0 * 1.035e3 * 3989.24495292815)
    assert cfg.surface["albedo"] == 0.3
    assert cfg.initial == {"kind": "isothermal", "temperature": 264.0}
    assert list(build_sigma(cfg).interfaces) == ISCA_GREY_MARS_BK
    assert build_sigma(cfg).nlev == 25
    # labelled substitutes (no boundary-layer diffusion or surface stress yet)
    assert cfg.drag == {"kind": "held_suarez", "k_f_per_day": 1.0, "sigma_b": 0.7}
    assert cfg.convection == {"timescale": 0.0}
    # radiation: eq. 14 and eq. 15 with Isca's fixed 610 Pa reference
    physics = build_physics(cfg)
    assert isinstance(physics.radiation.shortwave, VerticalPathShortwave)
    assert physics.radiation.shortwave.profile.terms == ((1.0, 4.0),)
    assert physics.radiation.shortwave.profile.reference == 610.0
    assert physics.radiation.longwave.profile.terms == ((1.0, 1.0),)
    assert physics.radiation.longwave.profile.reference == 610.0
    # diurnal + seasonal forcing, year starting at the northern spring equinox
    ins = build_insolation(cfg.insolation)
    assert ins.solar_day == 88440.0
    ill = ins.illumination
    assert ill.solar_longitude(0.0) % (2.0 * math.pi) == pytest.approx(0.0, abs=1e-12)
    assert math.degrees(ill.solar_longitude(ill.time_of_periapsis_s)) % 360.0 == \
        pytest.approx(250.85, abs=1e-9)
    assert ill.as_dict()["eccentricity"] == 0.0935
    assert RadiativeConfig.from_dict(json.loads(cfg.canonical_json())) == cfg


@needs_cuda
def test_flat_mars_one_day_is_stable_and_closes_atmosphere_and_slab_budgets(tmp_path):
    """The development preset at T9 for one day (diurnal and seasonal
    forcing, vertical-path shortwave, 610 Pa). Measured before this test
    was written: residual -0.0056 W m^-2 against 106 W m^-2 absorbed. The
    bound is the pilot's M2 daily bound, 1e-2 F."""
    from tropoi.run.radiative.experiment import RadiativeRun
    cfg = _tiny(PRESETS["dev-mars-gray-flat"](days=1), experiment_id="mars-test")
    run = RadiativeRun(cfg, tmp_path / "mars", allow_dirty=True, log=lambda s: None)
    ts0 = float((run.coupling.weights * run.coupling.surface_temperature).sum())
    assert run.run()["status"] == "completed"
    series = json.loads((tmp_path / "mars" / "series.json").read_text())
    absorbed = series["absorbed_stellar"][-1]
    assert absorbed == pytest.approx(105.0, rel=0.05)
    assert abs(series["energy_residual"][-1]) < 1e-2 * absorbed
    assert series["max_abs_u"][-1] < 150.0
    assert 100.0 < series["t_min"][-1] and series["t_max"][-1] < 400.0
    slab_heat = cfg.surface["heat_capacity"] * (series["mean_surface_temperature"][-1] - ts0)
    assert slab_heat == pytest.approx(series["surface_net"][-1] * 86400.0, rel=1e-9)


def test_rayleigh_sponge_rates_are_iscas_damping_driver():
    """Isca damping_driver `rayleigh` (d1321ac3): momentum only, rate
    (1 / tau) ((p_b - p) / p_b)^2 where p < p_b, here at p = sigma p_ref."""
    from tropoi.temporal.tendencies.drag import rayleigh_sponge_damping
    sigma = build_sigma(PRESETS["dev-mars-gray-flat"]()).full_levels_array()
    K, n = sigma.size, 23
    d = rayleigh_sponge_damping(sigma, n, reference_pressure=610.0, bottom_pressure=0.5,
                                efold_days=0.125)
    p = sigma * 610.0
    expected = np.where(p < 0.5, (1.0 / (0.125 * 86400.0)) * ((0.5 - p) / 0.5) ** 2, 0.0)
    np.testing.assert_allclose(d.rates[0:K], expected[:, None] * np.ones((1, n)), rtol=1e-14)
    np.testing.assert_allclose(d.rates[K:2 * K], expected[:, None] * np.ones((1, n)), rtol=1e-14)
    assert not d.rates[2 * K:].any()                      # temperature and ln p_s untouched
    assert np.count_nonzero(expected) >= 2 and expected[-1] == 0.0
    assert d.name == "rayleigh_sponge"


def test_the_sponge_is_left_out_of_configurations_without_it():
    """Adding the field must not move any recorded identity: the two Mars
    pilot runs (evidence of PR #35) keep their hashes."""
    ev = pathlib.Path(__file__).resolve().parents[1] / "docs" / "radiation" / "evidence"
    for preset_kw, run in (({}, "dev-mars-gray-flat-heated-T21L25-30d-b3aaef8-l4"),
                           ({"frictional_heating": False}, "dev-mars-gray-flat-T21L25-30d-b3aaef8-l4")):
        cfg = PRESETS["dev-mars-gray-flat"](**preset_kw)
        assert cfg.sponge is None and "sponge" not in cfg.to_dict()
        assert cfg.sha256() == json.loads((ev / run / "config.json").read_text())["config_sha256"]


def test_the_sponge_preset_differs_from_the_pilot_by_the_sponge_alone():
    base = PRESETS["dev-mars-gray-flat"]()
    cfg = PRESETS["dev-mars-gray-flat-sponge"]()
    assert cfg.experiment_id == "dev-mars-gray-flat-sponge-heated-T21L25"
    assert PRESETS["dev-mars-gray-flat-sponge"](frictional_heating=False).experiment_id == \
        "dev-mars-gray-flat-sponge-T21L25"
    assert cfg.sponge == {"bottom_pressure": 0.5, "efold_days": 0.125}
    assert cfg.with_(sponge=None, experiment_id=base.experiment_id) == base
    assert RadiativeConfig.from_dict(json.loads(cfg.canonical_json())) == cfg


@needs_cuda
def test_flat_mars_with_the_sponge_returns_its_loss_and_closes_the_budget(tmp_path):
    """One T9 day with the sponge and frictional heating: the sponge is one of
    the stepper's dampers, so the heating returns its kinetic-energy loss; the
    coupled residual meets the pilot's M2 daily bound, 1e-2 F."""
    from tropoi.run.radiative.experiment import RadiativeRun
    from tropoi.run.radiative.model import build_model, build_stepper
    cfg = _tiny(PRESETS["dev-mars-gray-flat-sponge"](days=1), experiment_id="mars-sponge-test")
    names = [d.name for d in build_stepper(cfg, build_model(cfg), None).dampers]
    assert "rayleigh_sponge" in names
    RadiativeRun(cfg, tmp_path / "run", allow_dirty=True, log=lambda s: None).run()
    series = json.loads((tmp_path / "run" / "series.json").read_text())
    assert abs(series["energy_residual"][-1]) < 1e-2 * series["absorbed_stellar"][-1]
    assert series["max_abs_u"][-1] < 150.0
