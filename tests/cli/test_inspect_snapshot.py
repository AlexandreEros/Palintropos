"""``tropoi inspect --snapshot/--field``: host-only snapshot description.

Each CPU-safety probe writes a real (tiny) capsule to a temporary directory
and runs the command in a FRESH interpreter, asserting that the banned
heavy modules (CuPy, Matplotlib, the runners, tropoi.viz) never load.
Error contracts (invalid index, wrong-solver field, --field without
--snapshot, unknown schema versions, incomplete capsules) run in-process.
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import numpy as np
import pytest

from tropoi.cli.main import main
from tropoi.representation.archive.schema import (COEFFICIENT_FILES,
                                                  STATE_SCHEMA_VERSION,
                                                  TIME_FILES,
                                                  provenance_blocks)

from .conftest import HEAVY_MODULES

L_MAX = 4
N = L_MAX + 1


def _config(solver: str) -> dict:
    base = {"lmax": L_MAX, "grid": "geodesic", "resolution": 2, "nlat": 8,
            "nlon": 16, "day_hours": 24.0, "radius_earth_units": 1.0,
            "duration_days": 0.5, "dt_snapshots": None, "scenario": "x",
            "snapshot_mode": "count", "n_snapshots": 2,
            "snapshot_times": [0.0, 43200.0], "plots": [],
            "product_quadrature": "fine"}
    if solver == "bve":
        base["viscosity"] = 0.0
    elif solver == "swe":
        base.update(solver="swe", gravity=9.80616, mean_depth_m=3000.0)
    else:
        base.update(solver="pe", nlev=3, sigma_interfaces=[0.0, 0.2, 0.6, 1.0],
                    r_dry=287.04, cp_dry=1004.64, dt_seconds=300.0,
                    temperature=260.0, surface_pressure=101325.0,
                    thermal_amplitude=1.0)
    return base


def _coeffs(solver: str, frames: int = 2) -> np.ndarray:
    rows = {"bve": None, "swe": 3, "pe": 3 * 3 + 1}[solver]
    shape = (frames, N, N) if rows is None else (frames, rows, N, N)
    rng = np.random.default_rng(7)
    coeffs = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    l = np.arange(N)[:, None]
    m = np.arange(N)[None, :]
    coeffs[..., m > l] = 0.0
    coeffs[..., :, 0] = coeffs[..., :, 0].real
    coeffs[..., 0, 0] = 0.0
    if solver == "pe":
        coeffs[:, 6:, 0, 0] = 5.0
    return coeffs


def write_capsule(root: pathlib.Path, solver: str, *, with_schema=True,
                  manifest_extra=None, frames=2) -> pathlib.Path:
    root.mkdir(parents=True, exist_ok=True)
    config = _config(solver)
    np.save(root / COEFFICIENT_FILES[solver], _coeffs(solver, frames))
    np.save(root / TIME_FILES[solver],
            np.linspace(0.0, 43200.0, frames) if frames else np.zeros(0))
    manifest = {"run_id": f"{solver}-inspect", "status": "completed",
                "run_config": config,
                "numerics": {"backend": "GeodesicBackend"}}
    if with_schema:
        manifest.update(provenance_blocks(solver, config))
    manifest.update(manifest_extra or {})
    (root / "manifest.json").write_text(json.dumps(manifest),
                                        encoding="utf-8")
    return root


# ---------------------------------------------------------------------------
# Fresh-interpreter CPU safety: a real capsule, a real inspect
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("solver,field", [
    ("bve", "zeta"), ("swe", "phi"), ("pe", "temperature"), ("pe", "ln_ps")])
def test_inspect_snapshot_is_cpu_safe_in_fresh_interpreter(tmp_path, solver,
                                                           field):
    run = write_capsule(tmp_path / solver, solver)
    code = (
        "import sys\n"
        "from tropoi.cli.main import main\n"
        f"code = main(['inspect', {str(run)!r}, '--snapshot', '-1', "
        f"'--field', {field!r}])\n"
        "assert code == 0, code\n"
        f"banned = [m for m in {HEAVY_MODULES!r} if m in sys.modules]\n"
        "assert not banned, banned\n"
        "assert 'tropoi.numerics' not in sys.modules\n"
        "assert 'tropoi.representation.visual.snapshot' not in sys.modules\n"
        # canonical homes of the numerics and cores (legacy names are aliases)
        "assert not [m for m in sys.modules if m.startswith(("
        "'tropoi.spatial.grids', 'tropoi.spatial.transforms', "
        "'tropoi.spatial.operators', 'tropoi.temporal.tendencies'))]\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True,
                            text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"field {field}" in result.stdout
    assert "conventions ok" in result.stdout


def test_inspect_snapshot_output_contract(tmp_path, capsys):
    run = write_capsule(tmp_path / "pe", "pe")
    assert main(["inspect", str(run), "--snapshot", "1"]) == 0
    out = capsys.readouterr().out
    assert "state fields      zeta[3 levels] (s^-1), delta[3 levels] (s^-1), " \
           "temperature[3 levels] (K), ln_ps (ln(Pa)); l_max=4" in out
    assert "Snapshot 1 of 2 (solver pe):" in out
    assert "time              43200 s (12 h)" in out
    assert "schema            manifest" in out
    assert "pe_coeffs.npy memory-mapped read-only" in out
    assert "shape           3x5x5 (level,l,m); l_max=4; 3 level(s), " \
           "sigma=0.1, 0.4, 0.8" in out
    assert "field ln_ps" in out and "shape           5x5 (l,m)" in out
    assert "monopole rule   horizontal-mean" in out
    assert "units           K" in out


def test_inspect_field_selects_only_that_field(tmp_path, capsys):
    run = write_capsule(tmp_path / "swe", "swe")
    assert main(["inspect", str(run), "--snapshot", "0", "--field",
                 "delta"]) == 0
    out = capsys.readouterr().out
    assert "field delta" in out
    assert "field zeta" not in out and "field phi" not in out
    assert "horizontal divergence" in out


def test_inspect_without_snapshot_is_unchanged_and_lists_fields(tmp_path,
                                                                capsys):
    run = write_capsule(tmp_path / "bve", "bve", with_schema=False)
    assert main(["inspect", str(run)]) == 0
    out = capsys.readouterr().out
    assert "Snapshot" not in out
    assert "state fields      zeta (s^-1); l_max=4 (inferred: no " \
           "state_schema block)" in out


# ---------------------------------------------------------------------------
# Error contracts
# ---------------------------------------------------------------------------

def test_field_requires_snapshot(tmp_path, capsys):
    run = write_capsule(tmp_path / "swe", "swe")
    assert main(["inspect", str(run), "--field", "zeta"]) == 2
    captured = capsys.readouterr()
    assert "--field requires --snapshot" in captured.err
    assert captured.out == ""


def test_invalid_index_and_wrong_solver_field(tmp_path, capsys):
    run = write_capsule(tmp_path / "swe", "swe")
    assert main(["inspect", str(run), "--snapshot", "2"]) == 2
    assert "out of range for 2 stored" in capsys.readouterr().err
    assert main(["inspect", str(run), "--snapshot", "0", "--field",
                 "temperature"]) == 2
    err = capsys.readouterr().err
    assert "unknown field 'temperature' for solver 'swe'" in err
    assert "zeta, delta, phi" in err


def test_empty_sequence_inspects_but_cannot_select(tmp_path, capsys):
    run = write_capsule(tmp_path / "empty", "bve", frames=0)
    assert main(["inspect", str(run)]) == 0
    assert main(["inspect", str(run), "--snapshot", "0"]) == 2
    assert "stores no snapshots" in capsys.readouterr().err


def test_unknown_schema_version_keeps_metadata_inspection(tmp_path, capsys):
    run = write_capsule(tmp_path / "future", "swe")
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    manifest["state_schema"]["version"] = STATE_SCHEMA_VERSION + 1
    (run / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert main(["inspect", str(run)]) == 0
    out = capsys.readouterr().out
    assert "swe-inspect" in out and "state fields" not in out
    assert main(["inspect", str(run), "--snapshot", "0"]) == 2
    err = capsys.readouterr().err
    assert "version" in err and "not supported" in err


def test_incomplete_capsule_metadata_inspects_but_snapshot_fails(tmp_path,
                                                                 capsys):
    run = tmp_path / "running"
    run.mkdir()
    (run / "manifest.json").write_text(json.dumps({
        "run_id": "r", "status": "running", "run_config": _config("pe")}),
        encoding="utf-8")
    assert main(["inspect", str(run)]) == 0
    out = capsys.readouterr().out
    assert "running" in out and "state fields      zeta[3 levels]" in out
    assert main(["inspect", str(run), "--snapshot", "0"]) == 2
    assert "missing" in capsys.readouterr().err


def test_invalid_coefficients_are_reported_not_repaired(tmp_path, capsys):
    run = write_capsule(tmp_path / "bad", "bve")
    coeffs = np.load(run / "vorticity_coeffs.npy")
    coeffs[1, 0, 3] = 2.0            # m > l padding
    np.save(run / "vorticity_coeffs.npy", coeffs)
    assert main(["inspect", str(run), "--snapshot", "1"]) == 0
    out = capsys.readouterr().out
    assert "INVALID" in out and "padding" in out
    np.testing.assert_array_equal(np.load(run / "vorticity_coeffs.npy"),
                                  coeffs)
