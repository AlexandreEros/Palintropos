"""Custom initial states executed on the GPU.

* preset versus reloaded trajectories: a preset's resolved initial state is
  exported, saved, reloaded from disk and run through
  ``run_initial_state``; the stored trajectory must match the preset run's
  (exactly at t = 0; to round-off afterwards);
* custom-state smoke runs on user-defined planets (BVE and SWE);
* SWE positivity: a state with negative reconstructed thickness fails the
  run before integration, and the capsule says so;
* capsule rebuild: ``Snapshot.plot`` reconstructs a custom run's model,
  including persisted terrain.

Every test needs a CUDA device and skips without one (CPU counterparts:
tests/test_custom_initial_states.py).
"""
from __future__ import annotations

import csv
import json
import math

import numpy as np
import pytest


def _has_cuda():
    try:
        import cupy as cp
        return cp.is_available()
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _has_cuda(), reason="needs a CUDA GPU")


def _timeseries(run_dir, column):
    with open(run_dir / "diagnostics" / "timeseries.csv", newline="") as fh:
        return np.array([float(row[column]) for row in csv.DictReader(fh)])


def _assert_same_trajectory(preset, custom, coefficient_file, time_file):
    a = np.load(preset / coefficient_file)
    b = np.load(custom / coefficient_file)
    assert a.shape == b.shape and a.shape[0] >= 2
    # The reloaded state is the preset's resolved state, bit for bit.
    assert a[0].tobytes() == b[0].tobytes()
    # Same planet, grid, core and schedule: the trajectories agree to
    # round-off (identical arithmetic is expected on one device).
    scale = float(np.abs(a).max())
    np.testing.assert_allclose(b, a, rtol=0.0, atol=1e-12 * scale)
    np.testing.assert_array_equal(np.load(preset / time_file),
                                  np.load(custom / time_file))
    np.testing.assert_allclose(_timeseries(custom, "dt_s"),
                               _timeseries(preset, "dt_s"), rtol=1e-12)


# ---------------------------------------------------------------------------
# Preset versus reloaded
# ---------------------------------------------------------------------------

def test_bve_preset_versus_reloaded_trajectory(tmp_path):
    from tropoi.run.bve.execution import execute_run
    from tropoi.representation.archive import resolve_run_directory
    from tropoi.run.bve.config import BVERunConfig
    from tropoi.run.custom import (initial_state_from_config,
                                   load_initial_state, run_initial_state)

    cfg = BVERunConfig.resolve({
        "scenario": "two_vortices", "grid": "latlon", "nlat": 12, "nlon": 24,
        "lmax": 8, "day_hours": 24.0, "duration_days": 0.02,
        "n_snapshots": 3, "out": str(tmp_path / "preset")})
    assert execute_run(cfg) == 0
    preset = resolve_run_directory(tmp_path / "preset")

    saved = initial_state_from_config(cfg).save(tmp_path / "two_vortices")
    state = load_initial_state(saved)
    custom = run_initial_state(state, duration_days=cfg.duration_days,
                               n_snapshots=3, out=str(tmp_path / "custom"))
    _assert_same_trajectory(preset, custom, "vorticity_coeffs.npy",
                            "bve_snapshot_times.npy")
    np.testing.assert_allclose(_timeseries(custom, "energy"),
                               _timeseries(preset, "energy"), rtol=1e-12)


def test_williamson5_preset_versus_reloaded_trajectory(tmp_path):
    from tropoi.run.swe.execution import execute_run
    from tropoi.representation.archive import resolve_run_directory
    from tropoi.run.custom import (initial_state_from_config,
                                   load_initial_state, run_initial_state)
    from tropoi.run.swe.config import SWERunConfig
    from tropoi.spatial.williamson5 import W5_GRAVITY, W5_RADIUS_M

    cfg = SWERunConfig.resolve({
        "scenario": "williamson5", "grid": "latlon", "nlat": 24, "nlon": 48,
        "lmax": 15, "duration_days": 0.05, "n_snapshots": 3,
        "out": str(tmp_path / "preset")})
    assert execute_run(cfg) == 0
    preset = resolve_run_directory(tmp_path / "preset")

    exported = initial_state_from_config(cfg)
    assert exported.planet.construction == "ideal_sphere"
    assert exported.planet.dynamical_radius_m == W5_RADIUS_M
    assert exported.fluid.gravity_m_s2 == W5_GRAVITY
    assert exported.topography.preset == "williamson5_cone"
    state = load_initial_state(exported.save(tmp_path / "w5"))
    custom = run_initial_state(state, duration_days=cfg.duration_days,
                               n_snapshots=3, out=str(tmp_path / "custom"))
    _assert_same_trajectory(preset, custom, "swe_coeffs.npy",
                            "swe_snapshot_times.npy")
    np.testing.assert_allclose(_timeseries(custom, "total_energy"),
                               _timeseries(preset, "total_energy"),
                               rtol=1e-12)
    manifest = json.loads((custom / "manifest.json").read_text())
    geometry = manifest["state_schema"]["geometry"]
    assert geometry["planet"] == "ideal_sphere"
    assert geometry["dynamical_radius_m"] == W5_RADIUS_M


def test_mountain_flow_preset_versus_reloaded_trajectory(tmp_path):
    from tropoi.run.swe.execution import execute_run
    from tropoi.representation.archive import resolve_run_directory
    from tropoi.run.custom import (initial_state_from_config,
                                   load_initial_state, run_initial_state)
    from tropoi.run.swe.config import SWERunConfig

    cfg = SWERunConfig.resolve({
        "scenario": "williamson2", "grid": "latlon", "nlat": 24, "nlon": 48,
        "lmax": 15, "mean_depth_m": 5000.0, "topography": "mountain",
        "mountain_width_deg": 30.0, "duration_days": 0.05, "n_snapshots": 2,
        "out": str(tmp_path / "preset")})
    assert execute_run(cfg) == 0
    preset = resolve_run_directory(tmp_path / "preset")
    state = load_initial_state(
        initial_state_from_config(cfg).save(tmp_path / "w2m"))
    assert state.topography.preset == "mountain"
    custom = run_initial_state(state, duration_days=cfg.duration_days,
                               n_snapshots=2, out=str(tmp_path / "custom"))
    _assert_same_trajectory(preset, custom, "swe_coeffs.npy",
                            "swe_snapshot_times.npy")


# ---------------------------------------------------------------------------
# Custom-state smoke tests
# ---------------------------------------------------------------------------

def _rh_like_vorticity(l_max, radius):
    """Exact spectral zonal flow plus a wave-4 Rossby-Haurwitz-like mode."""
    zeta = np.zeros((l_max + 1, l_max + 1), dtype=np.complex128)
    zeta[1, 0] = 2.0 * 2e-6 * math.sqrt(4.0 * math.pi / 3.0)
    zeta[5, 4] = 3e-6 + 1e-6j
    return zeta


def test_custom_bve_planet_smoke(tmp_path):
    from tropoi.run.bve.model import build_planet
    from tropoi.representation.archive import open_simulation
    from tropoi.run.custom import (DiscretizationSpec, InitialState,
                                   PlanetSpec, load_initial_state,
                                   run_initial_state)

    planet = PlanetSpec.ideal_sphere(radius_m=2.0e6, sidereal_day_s=43200.0)
    state = InitialState(
        "bve", _rh_like_vorticity(10, 2.0e6), planet,
        DiscretizationSpec.gauss_latlon(10, 16, 32), label="rh-like")
    saved = state.save(tmp_path / "rh_like")
    run = run_initial_state(load_initial_state(saved), duration_days=0.05,
                            n_snapshots=3, out=str(tmp_path / "runs"))
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["status"] == "completed"
    sim = open_simulation(run)
    assert sim[0].state["zeta"].coeffs.tobytes() == state.coeffs.tobytes()
    assert np.isfinite(np.asarray(sim[-1].state["zeta"].coeffs)).all()
    energy = _timeseries(run, "energy")
    assert abs(energy[-1] / energy[0] - 1.0) < 1e-5
    rebuilt = build_planet(manifest["run_config"])
    assert float(rebuilt.params.radius) == 2.0e6
    assert float(rebuilt.params.angular_velocity) == 2 * math.pi / 43200.0


def test_custom_swe_planet_smoke(tmp_path):
    from tropoi.representation.archive import open_simulation
    from tropoi.run.custom import (DiscretizationSpec, FluidLayerSpec,
                                   InitialState, PlanetSpec,
                                   run_initial_state)

    l_max = 10
    fluid = FluidLayerSpec(gravity_m_s2=3.71, mean_depth_m=2000.0)
    coeffs = np.zeros((3, l_max + 1, l_max + 1), dtype=np.complex128)
    coeffs[2, 4, 2] = 1e-2 * fluid.phi0
    state = InitialState(
        "swe", coeffs, PlanetSpec.oblate(3.39e6, 88642.0, mass_kg=6.42e23),
        DiscretizationSpec.gauss_latlon(l_max, 16, 32), fluid=fluid,
        label="mars-like gravity wave")
    run = run_initial_state(state, duration_days=0.05, n_snapshots=2,
                            out=str(tmp_path / "runs"))
    sim = open_simulation(run)
    assert sim.metadata["status"] == "completed"
    assert sim[0].state["phi"].coeffs.tobytes() == coeffs[2].tobytes()
    mass = _timeseries(run, "total_mass")
    assert mass[-1] == mass[0]


def test_swe_state_with_negative_thickness_fails_before_integration(tmp_path):
    from tropoi.run.custom import (DiscretizationSpec, FluidLayerSpec,
                                   InitialState, PlanetSpec,
                                   run_initial_state)
    from tropoi.spatial.states.shallow_water import ShallowWaterStateError

    l_max = 8
    fluid = FluidLayerSpec(gravity_m_s2=9.80616, mean_depth_m=100.0)
    coeffs = np.zeros((3, l_max + 1, l_max + 1), dtype=np.complex128)
    # Y_2^0 is -0.315 at the equator: 5 * Phi0 * Y_2^0 < -Phi0 there.
    coeffs[2, 2, 0] = 5.0 * fluid.phi0
    state = InitialState("swe", coeffs, PlanetSpec.earth_like(),
                         DiscretizationSpec.gauss_latlon(l_max, 12, 24),
                         fluid=fluid)
    with pytest.raises(ShallowWaterStateError, match="not strictly positive"):
        run_initial_state(state, duration_days=0.01,
                          out=str(tmp_path / "runs"))
    (run,) = [p for p in (tmp_path / "runs").iterdir() if p.is_dir()]
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["error"]["type"] == "ShallowWaterStateError"


def test_custom_run_with_terrain_rebuilds_for_plotting(tmp_path):
    pytest.importorskip("matplotlib")
    from tropoi.representation.archive import open_simulation
    from tropoi.run.custom import initial_state_from_preset, run_initial_state

    state = initial_state_from_preset("swe", {
        "scenario": "williamson5", "grid": "latlon", "nlat": 24, "nlon": 48,
        "lmax": 15})
    run = run_initial_state(state, duration_days=0.02, n_snapshots=2,
                            out=str(tmp_path / "runs"))
    figure = open_simulation(run)[-1].plot(tmp_path / "w5.png")
    assert figure.exists() and figure.stat().st_size > 0


def test_cli_initial_state_flag_reproduces_the_python_run(tmp_path):
    from tropoi.cli.main import main
    from tropoi.run.bve.config import BVERunConfig
    from tropoi.run.custom import initial_state_from_config, run_initial_state

    cfg = BVERunConfig.resolve({
        "scenario": "two_vortices", "grid": "latlon", "nlat": 12, "nlon": 24,
        "lmax": 8, "day_hours": 24.0})
    saved = initial_state_from_config(cfg).save(tmp_path / "state")
    python_run = run_initial_state(saved, duration_days=0.02, n_snapshots=3,
                                   out=str(tmp_path / "python"))
    assert main(["run", "bve", "--initial-state", str(saved), "--days",
                 "0.02", "--n-snapshots", "3", "--out",
                 str(tmp_path / "cli")]) == 0
    (cli_run,) = [p for p in (tmp_path / "cli").iterdir() if p.is_dir()]
    # Same state, planet, grid and schedule: same scientific run-id hash
    # and the same trajectory.
    assert cli_run.name.split("_")[6] == python_run.name.split("_")[6]
    expected = np.load(python_run / "vorticity_coeffs.npy")
    np.testing.assert_allclose(np.load(cli_run / "vorticity_coeffs.npy"),
                               expected, rtol=0,
                               atol=1e-12 * np.abs(expected).max())
