"""``tropoi run bve|swe --initial-state PATH`` on the CPU.

The CUDA integration is stubbed (``tropoi.run.custom.execute._integrate``),
so these tests exercise the real argument handling, loading, verification,
configuration resolution, lifecycle and capsule contents. The GPU run of
the same path is covered by tests/test_custom_runs.py.
"""
from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pytest

from tropoi.cli.main import main
from tropoi.run.custom import (DiscretizationSpec, FluidLayerSpec,
                               InitialState, PlanetSpec, TopographySpec,
                               load_initial_state)

L = 6


def _bve_state():
    zeta = np.zeros((L + 1, L + 1), dtype=np.complex128)
    zeta[1, 0] = 1e-5
    zeta[4, 2] = 2e-6 - 1e-6j
    return InitialState("bve", zeta, PlanetSpec.ideal_sphere(3.0e6, 36000.0),
                        DiscretizationSpec.gauss_latlon(L, 12, 24),
                        label="cli-bve")


def _swe_state():
    coeffs = np.zeros((3, L + 1, L + 1), dtype=np.complex128)
    coeffs[2, 2, 0] = 50.0
    topo = np.zeros((L + 1, L + 1), dtype=np.complex128)
    topo[0, 0], topo[3, 1] = 30.0, 5.0 + 2.0j
    return InitialState(
        "swe", coeffs, PlanetSpec.oblate(5.0e6, 50000.0),
        DiscretizationSpec.gauss_latlon(L, 12, 24),
        fluid=FluidLayerSpec(3.71, 4000.0),
        topography=TopographySpec.mountain(height_m=2000, lat_deg=30,
                                           lon_deg=90, width_deg=20),
        topography_lm=topo, label="cli-swe")


@pytest.fixture
def stub_integration(monkeypatch):
    import tropoi.run.custom.execute as execute
    from tropoi.representation.archive.schema import (COEFFICIENT_FILES,
                                                      TIME_FILES)

    seen = {}

    def fake(cfg, run_dir, run_config):
        seen["cfg"] = cfg
        times = np.asarray(cfg.snapshot_times_seconds(), dtype=np.float64)
        frames = np.stack([cfg.initial_state.coeffs] * len(times))
        np.save(run_dir.path / COEFFICIENT_FILES[cfg.solver], frames)
        np.save(run_dir.path / TIME_FILES[cfg.solver], times)

    monkeypatch.setattr(execute, "_integrate", fake)
    return seen


def _only_run(base):
    (run,) = [p for p in base.iterdir() if p.is_dir()]
    return run


@pytest.mark.parametrize("make,solver", [(_bve_state, "bve"),
                                         (_swe_state, "swe")])
def test_initial_state_flag_runs_the_saved_state(tmp_path, stub_integration,
                                                 make, solver, capsys):
    state = make()
    path = state.save(tmp_path / "state")
    code = main(["run", solver, "--initial-state", str(path), "--days", "0.5",
                 "--n-snapshots", "3", "--out", str(tmp_path / "runs")])
    assert code == 0
    out = capsys.readouterr().out
    assert state.digest[:19] in out and state.label in out
    run = _only_run(tmp_path / "runs")
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    assert config["initial_state_digest"] == state.digest
    assert config["solver"] == solver and config["scenario"] == "custom"
    assert config["duration_days"] == 0.5 and config["n_snapshots"] == 3
    assert load_initial_state(run / "initial_state").digest == state.digest
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"


def test_grid_options_override_the_recorded_discretization(
        tmp_path, stub_integration):
    path = _bve_state().save(tmp_path / "state")
    assert main(["run", "bve", "--initial-state", str(path),
                 "--backend", "geodesic", "--resolution", "2",
                 "--product-quadrature", "coarse", "--viscosity", "1e4",
                 "--days", "0.1", "--snapshot-interval-seconds", "3600",
                 "--out", str(tmp_path / "runs")]) == 0
    cfg = stub_integration["cfg"]
    assert cfg.discretization == DiscretizationSpec(
        lmax=L, grid="geodesic", resolution=2, nlat=12, nlon=24,
        product_quadrature="coarse")
    assert cfg.snapshot_mode == "interval" and cfg.viscosity == 1e4


def test_defaults_match_the_ordinary_run_commands(tmp_path, stub_integration):
    path = _swe_state().save(tmp_path / "state")
    assert main(["run", "swe", "--initial-state", str(path),
                 "--out", str(tmp_path / "runs"), "--no-plots"]) == 0
    cfg = stub_integration["cfg"]
    assert cfg.duration_days == 1.0
    assert cfg.snapshot_mode == "count" and cfg.n_snapshots == 5
    assert cfg.plots == () and cfg.discretization == _swe_state().discretization


@pytest.mark.parametrize("solver,extra,match", [
    ("bve", ["--scenario", "rh4"], "--scenario cannot be combined"),
    ("bve", ["--preset", "rh4"], "--preset cannot be combined"),
    ("bve", ["--day-hours", "24"], "--day-hours"),
    ("bve", ["--radius-earth-units", "2"], "--radius-earth-units"),
    ("swe", ["--gravity", "9.8"], "--gravity cannot be combined"),
    ("swe", ["--mean-depth", "100"], "--mean-depth"),
    ("swe", ["--topography", "flat"], "--topography"),
    ("swe", ["--mountain-width-deg", "10"], "--mountain-width-deg"),
])
def test_options_the_state_prescribes_are_refused(tmp_path, capsys, solver,
                                                  extra, match):
    path = (_bve_state() if solver == "bve" else _swe_state()).save(
        tmp_path / "state")
    with pytest.raises(SystemExit) as info:
        main(["run", solver, "--initial-state", str(path), *extra,
              "--out", str(tmp_path / "runs")])
    assert info.value.code == 2
    assert match in capsys.readouterr().err
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("args,match", [
    (["run", "swe", "--initial-state", "{bve}"], "holds a BVE state"),
    (["run", "bve", "--initial-state", "{missing}"], "not a saved initial"),
    (["run", "bve", "--initial-state", "{bve}", "--l-max", "8"],
     "never truncated or padded"),
    (["run", "bve", "--initial-state", "{bve}", "--days", "-1"],
     "duration_days"),
    (["run", "swe", "--initial-state", "{swe}", "--backend", "latlon",
      "--nlat", "1"], "nlat"),
])
def test_user_errors_fail_before_any_run(tmp_path, capsys, args, match):
    paths = {"bve": _bve_state().save(tmp_path / "bve"),
             "swe": _swe_state().save(tmp_path / "swe"),
             "missing": tmp_path / "nowhere"}
    argv = [a.format(**{k: str(v) for k, v in paths.items()}) for a in args]
    with pytest.raises(SystemExit) as info:
        main([*argv, "--out", str(tmp_path / "runs")])
    assert info.value.code == 2
    assert match in capsys.readouterr().err
    assert not (tmp_path / "runs").exists()


def test_a_tampered_state_is_refused(tmp_path, capsys):
    path = _bve_state().save(tmp_path / "state")
    arr = np.load(path / "state_coeffs.npy")
    arr[2, 1] = 1.0
    np.save(path / "state_coeffs.npy", arr)
    with pytest.raises(SystemExit):
        main(["run", "bve", "--initial-state", str(path),
              "--out", str(tmp_path / "runs")])
    assert "SHA-256" in capsys.readouterr().err


def test_flag_validation_never_imports_cuda(tmp_path):
    from cli.conftest import HEAVY_MODULES

    path = _bve_state().save(tmp_path / "state")
    code = (
        "import sys\n"
        "from tropoi.cli.main import main\n"
        "try:\n"
        f"    main(['run', 'bve', '--initial-state', {str(path)!r}, "
        "'--l-max', '9'])\n"
        "except SystemExit as exc:\n"
        "    assert exc.code == 2\n"
        f"banned = [m for m in {HEAVY_MODULES!r} if m in sys.modules]\n"
        "assert not banned, banned\n")
    result = subprocess.run([sys.executable, "-c", code], capture_output=True,
                            text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
