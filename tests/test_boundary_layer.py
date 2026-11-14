"""The dry boundary layer of MARS_MI_PLAN §2e.

CPU: the column pieces (exchange coefficient, depth, K profile, implicit
solve) against hand-computed and dense-matrix references, and the
configuration contract (the new preset adds the scheme and removes both
substitutes; every earlier configuration keeps its canonical form). GPU:
one short coupled run closes its budget, records the kinetic-energy sink,
and resumes bit-identically through a checkpoint.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

from tropoi.physics.boundary_layer import (boundary_layer_depth, diffusivity_profile,
                                           exchange_coefficient, implicit_column_diffusion)
from tropoi.physics.surface import VON_KARMAN, sensible_heat_flux
from tropoi.run.radiative.config import PRESETS, RadiativeConfig


def _has_cuda() -> bool:
    try:
        import cupy
        return bool(cupy.is_available())
    except Exception:
        return False


needs_cuda = pytest.mark.skipif(not _has_cuda(), reason="needs CuPy and a CUDA GPU")

G, CP, R = 3.71, 844.8, 192.0


# ---------------------------------------------------------------------------
# the column pieces (CPU)
# ---------------------------------------------------------------------------

def test_exchange_coefficient_is_the_bulk_flux_coefficient():
    """The stress uses the same C the sensible heat flux computes (plan §2e:
    one FHZ06 coefficient for momentum and heat)."""
    ts = np.array([285.0, 240.0, 210.0])
    ta = np.array([280.0, 245.0, 212.0])
    z = np.array([7.0, 7.5, 6.5])
    speed = np.array([10.0, 25.0, 0.5])
    rho = np.array([0.014, 0.015, 0.013])
    _, c_flux = sensible_heat_flux(ts, ta, z, speed, rho, CP, G,
                                   roughness_length=3.21e-5, critical_richardson=1.0)
    c = exchange_coefficient(ts, ta, z, speed, CP, G, roughness_length=3.21e-5,
                             critical_richardson=1.0)
    np.testing.assert_array_equal(c, c_flux)


def test_depth_interpolates_linearly_in_the_bulk_richardson_number():
    """Three levels; the bulk Ri crosses 1 between the two upper ones, and
    h is the hand-computed linear interpolation (Isca pbl_depth)."""
    z = np.array([[3000.0], [1000.0], [10.0]])
    # dse chosen so Ri(level 1) = 0.5 and Ri(level 0) = 4.0 at |v|^2 = 1
    s_bot = CP * 240.0
    ds1 = 0.5 * s_bot / (G * z[1, 0])           # Ri_1 = 0.5
    ds0 = 4.0 * s_bot / (G * z[0, 0])           # Ri_0 = 4.0
    dse = np.array([[s_bot + ds0], [s_bot + ds1], [s_bot]])
    v2 = np.ones((3, 1))
    h = boundary_layer_depth(z, dse, v2, G, critical_richardson=1.0)
    expected = 3000.0 + (1000.0 - 3000.0) * (4.0 - 1.0) / (4.0 - 0.5)
    assert h[0] == pytest.approx(expected)


def test_depth_without_a_crossing_stays_at_the_lowest_level():
    z = np.array([[3000.0], [1000.0], [10.0]])
    dse = np.full((3, 1), CP * 240.0)           # neutral column: Ri = 0
    h = boundary_layer_depth(z, dse, np.ones((3, 1)), G)
    assert h[0] == 10.0


def test_the_k_profile_is_fhz06_neutral():
    """kappa u* z below the inner layer, the cubic taper above, zero at h."""
    u_star = np.array([2.0])
    h = np.array([1000.0])
    z = np.array([[50.0], [100.0], [500.0], [1000.0], [1500.0]])
    k = diffusivity_profile(z, u_star, h, frac_inner=0.1)
    assert k[0, 0] == pytest.approx(VON_KARMAN * 2.0 * 50.0)
    assert k[1, 0] == pytest.approx(VON_KARMAN * 2.0 * 100.0)          # continuous at 0.1 h
    taper = (1.0 - (500.0 - 100.0) / 900.0) ** 2
    assert k[2, 0] == pytest.approx(VON_KARMAN * 2.0 * 500.0 * taper)
    assert k[3, 0] == 0.0 and k[4, 0] == 0.0


def test_the_implicit_solve_matches_a_dense_backward_euler_solve():
    """The Thomas sweep against numpy.linalg.solve on the same system."""
    rng = np.random.default_rng(7)
    K, n = 8, 5
    q = rng.normal(250.0, 20.0, (K, n))
    mass = rng.uniform(0.2, 5.0, (K, n))
    cond = rng.uniform(0.0, 0.05, (K - 1, n))
    g_b = rng.uniform(0.0, 0.01, n)
    w = 1800.0
    out = implicit_column_diffusion(q, mass, cond, w, bottom_conductance=g_b)
    for j in range(n):
        A = np.zeros((K, K))
        for k in range(K):
            A[k, k] = mass[k, j]
            if k > 0:
                A[k, k] += w * cond[k - 1, j]
                A[k, k - 1] = -w * cond[k - 1, j]
            if k < K - 1:
                A[k, k] += w * cond[k, j]
                A[k, k + 1] = -w * cond[k, j]
        A[K - 1, K - 1] += w * g_b[j]
        ref = np.linalg.solve(A, mass[:, j] * q[:, j])
        np.testing.assert_allclose(out[:, j], ref, rtol=1e-12)


def test_the_solve_conserves_and_never_overshoots_without_a_bottom_sink():
    rng = np.random.default_rng(11)
    K, n = 25, 40
    q = rng.normal(230.0, 30.0, (K, n))
    mass = rng.uniform(0.1, 4.0, (K, n))
    cond = rng.uniform(0.0, 1.0, (K - 1, n))
    out = implicit_column_diffusion(q, mass, cond, 3600.0)
    np.testing.assert_allclose((mass * out).sum(axis=0), (mass * q).sum(axis=0), rtol=1e-12)
    assert (out <= q.max(axis=0) + 1e-9).all() and (out >= q.min(axis=0) - 1e-9).all()
    # a huge window mixes the column completely where conductances connect it
    mixed = implicit_column_diffusion(q, mass, np.full((K - 1, n), 1.0), 1e7)
    mean = (mass * q).sum(axis=0) / mass.sum(axis=0)
    np.testing.assert_allclose(mixed, np.broadcast_to(mean, (K, n)), rtol=1e-4)


def test_the_bottom_conductance_alone_is_the_implicit_bulk_stress():
    """With no interior mixing, only the lowest layer relaxes, by exactly
    m / (m + w g_b) - the §2d relaxation form, never overshooting."""
    q = np.array([[30.0], [40.0], [60.0]])
    mass = np.array([[1.0], [1.0], [0.2]])
    cond = np.zeros((2, 1))
    g_b = np.array([9.2e-4])
    w = 1800.0
    out = implicit_column_diffusion(q, mass, cond, w, bottom_conductance=g_b)
    assert out[0, 0] == 30.0 and out[1, 0] == 40.0
    assert out[2, 0] == pytest.approx(60.0 * 0.2 / (0.2 + w * 9.2e-4))


# ---------------------------------------------------------------------------
# the configuration contract (CPU)
# ---------------------------------------------------------------------------

def test_the_bl_preset_removes_both_substitutes_and_moves_no_other_identity():
    """Plan §2e: drag and convection become None, the scheme's options are
    the registered ones, and every other preset's canonical form has no
    boundary_layer key (their hashes cannot move)."""
    cfg = PRESETS["dev-mars-gray-flat-bl"]()
    assert cfg.experiment_id == "dev-mars-gray-flat-bl-heated-T21L25"
    assert cfg.days == 60 and cfg.drag is None and cfg.convection is None
    assert cfg.sponge is not None and cfg.sensible["implicit"] is True
    assert cfg.boundary_layer == {"roughness_length": 3.21e-5, "critical_richardson": 1.0,
                                  "rich_crit_pbl": 1.0, "frac_inner": 0.1, "gustiness": 0.0}
    implicit = PRESETS["dev-mars-gray-flat-sponge-implicit"]()
    assert cfg.with_(boundary_layer=None, drag=implicit.drag, convection=implicit.convection,
                     experiment_id=implicit.experiment_id) == implicit
    for name, factory in PRESETS.items():
        if name != "dev-mars-gray-flat-bl":
            assert "boundary_layer" not in factory().to_dict()


def test_the_bl_preset_round_trips_and_requires_a_surface():
    cfg = PRESETS["dev-mars-gray-flat-bl"]()
    again = RadiativeConfig.from_dict(json.loads(json.dumps(cfg.to_dict())))
    assert again == cfg and again.sha256() == cfg.sha256()
    with pytest.raises(ValueError, match="needs a surface"):
        cfg.with_(surface=None, sensible=None, shortwave={"kind": "transparent"})


# ---------------------------------------------------------------------------
# the coupled model (GPU)
# ---------------------------------------------------------------------------

def _tiny_bl(days: int, experiment_id: str) -> RadiativeConfig:
    return PRESETS["dev-mars-gray-flat-bl"](days=days).with_(
        l_max=10, retained_truncation=9, nlat=16, nlon=32, experiment_id=experiment_id,
        checkpoint_every_days=1, state_every_days=1)


@needs_cuda
def test_one_day_with_the_boundary_layer_closes_the_budget_and_records_the_sink(tmp_path):
    from tropoi.run.radiative.experiment import RadiativeRun
    cfg = _tiny_bl(1, "mars-bl-test")
    run = RadiativeRun(cfg, tmp_path / "run", allow_dirty=True, log=lambda s: None)
    assert run.boundary_layer is not None
    run.run()
    series = json.loads((tmp_path / "run" / "series.json").read_text())
    assert abs(series["energy_residual"][-1]) < 1e-2 * series["absorbed_stellar"][-1]
    assert series["bl_kinetic_sink"][-1] <= 0.0          # removal, returned as heat
    assert series["bl_max_depth"][-1] >= 0.0
    slab_heat = cfg.surface["heat_capacity"] * (series["mean_surface_temperature"][-1]
                                                - series["mean_surface_temperature"][0])
    assert slab_heat == pytest.approx(series["surface_net"][-1] * 86400.0, rel=1e-9)


@needs_cuda
def test_a_resumed_boundary_layer_run_is_identical(tmp_path):
    """Two days straight equal one day, a restart and a second day, byte for
    byte in the state and exactly in the recorded series."""
    import cupy as cp
    from tropoi.run.radiative.experiment import RadiativeRun
    cfg = _tiny_bl(2, "mars-bl-resume")
    straight = RadiativeRun(cfg, tmp_path / "a", allow_dirty=True, log=lambda s: None)
    straight.run()
    split = RadiativeRun(cfg, tmp_path / "b", allow_dirty=True, log=lambda s: None)
    split.run(until_day=1)
    resumed = RadiativeRun(cfg, tmp_path / "b", allow_dirty=True, log=lambda s: None)
    resumed.run()
    assert (cp.asnumpy(straight.stepper.state) == cp.asnumpy(resumed.stepper.state)).all()
    sa = json.loads((tmp_path / "a" / "series.json").read_text())
    sb = json.loads((tmp_path / "b" / "series.json").read_text())
    assert sa == sb


# ---------------------------------------------------------------------------
# the 2e judge path (CPU)
# ---------------------------------------------------------------------------

def _fake_evidence(root: pathlib.Path, name: str, *, drift30: float, drift60: float,
                   wind_bottom: float) -> None:
    """A synthetic 60-day evidence directory with everything the judge reads."""
    days = 60
    ps0 = 610.0
    ps = [ps0] + [ps0 * (1.0 + drift30 * min(d, 30) / 30.0
                         + (drift60 - drift30) * max(d - 30, 0) / 30.0)
                  for d in range(1, days + 1)]
    n1 = days + 1
    series = {
        "day": list(range(n1)), "mean_ps": ps,
        "max_abs_u": [100.0] * n1, "t_min": [150.0] * n1, "t_max": [300.0] * n1,
        "max_wind_bottom": [wind_bottom] * n1,
        "bl_kinetic_sink": [-0.05] * n1, "bl_max_depth": [4000.0] * n1,
        "energy_total": [1.0e7] * n1, "mean_surface_temperature": [240.0] * n1,
        "energy_residual": [-0.01] * days, "absorbed_stellar": [100.0] * n1,
        "toa_net_down": [0.0] * n1, "surface_net": [0.0] * n1,
    }
    d = root / name
    d.mkdir(parents=True)
    (d / "series.json").write_text(json.dumps(series))
    (d / "summary.json").write_text(json.dumps(
        {"sinks": {"states": [{"day": day, "all_sinks_total": -0.01}
                              for day in range(11, 31)]}}))
    (d / "config.json").write_text(json.dumps(
        {"config": {"cp": 844.8, "gravity": 3.71, "surface": {"heat_capacity": 8.26e6}}}))
    (d / "tides.json").write_text(json.dumps({"M5": {"met": False}}))


def test_the_2e_judge_applies_the_scaled_m6_and_m7(tmp_path, monkeypatch, capsys):
    """Plan §2e: M6 needs BOTH the 30-day and the scaled 60-day bound; M7
    bounds the lowest-level wind; the fixer-equivalent report follows from
    the series arithmetic. The default (2d) path is untouched by the flag."""
    import importlib
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools" / "radiation"))
    judge = importlib.import_module("judge_mars_pilot")
    monkeypatch.setattr(judge, "EVIDENCE", tmp_path)
    # H passes everything; U fails M6 on the 60-day bound alone and M7
    _fake_evidence(tmp_path, "H", drift30=-8e-6, drift60=-1.5e-5, wind_bottom=30.0)
    _fake_evidence(tmp_path, "U", drift30=-8e-6, drift60=-2.5e-5, wind_bottom=45.0)
    monkeypatch.setattr(sys, "argv", ["judge", "H", "U", "--days", "60",
                                      "--window", "11", "30", "--criteria", "2e"])
    judge.main()
    v = json.loads((tmp_path / "H" / "verdicts.json").read_text())
    assert v["criteria_commit"].startswith("04f4547")
    assert v["H"]["M6"] is True and v["U"]["M6"] is False
    assert v["H"]["M7"] is True and v["U"]["M7"] is False
    assert v["verdict"]["M6"] is False and v["verdict"]["M7"] is False
    assert v["H"]["ps_drift_day30"] == pytest.approx(-8e-6)
    fx = v["reported"]["fixer_equivalent"]
    assert fx["mass_factor_log_sum"] == pytest.approx(-np.log1p(-1.5e-5), rel=1e-6)
    # -res * 86400 * g / (cp * ps): res = -0.01 everywhere
    assert fx["energy_uniform_dT_mean"] == pytest.approx(
        0.01 * 86400.0 * 3.71 / (844.8 * 610.0), rel=1e-4)
    assert v["reported"]["bl_kinetic_sink_window_mean"] == pytest.approx(-0.05)
