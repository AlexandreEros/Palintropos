"""The runnable examples in examples/.

CPU: the BVE example's construction, saving and fresh-process reload (with
CuPy blocked), and an independent check of its coefficient convention by
synthesizing the saved coefficients with SciPy's spherical harmonics.
GPU (skipped without CUDA): every step of both examples, including runs.
"""
from __future__ import annotations

import math
import pathlib
import subprocess
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
BVE = ROOT / "examples" / "custom_bve_planet.py"
SWE = ROOT / "examples" / "custom_swe_planet.py"


def _has_cuda():
    try:
        import cupy as cp
        return cp.is_available()
    except Exception:
        return False


def _run(script, *args, block_cupy=False, cwd):
    if block_cupy:
        code = ("import runpy, sys\nsys.modules['cupy'] = None\n"
                f"sys.argv = [{str(script)!r}, *{list(args)!r}]\n"
                f"runpy.run_path({str(script)!r}, run_name='__main__')\n")
        cmd = [sys.executable, "-c", code]
    else:
        cmd = [sys.executable, str(script), *args]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd,
                            timeout=1800)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def _synthesize(coeffs, lat, lon):
    """Real field from m >= 0 orthonormal coefficients (SciPy, CPU)."""
    from scipy.special import sph_harm_y

    colat = np.pi / 2 - lat
    field = np.zeros_like(lat)
    for l in range(coeffs.shape[0]):
        for m in range(l + 1):
            a = coeffs[l, m]
            if a == 0:
                continue
            y = sph_harm_y(l, m, colat, lon)
            field += (a * y).real if m == 0 else 2.0 * (a * y).real
    return field


def test_bve_example_prepares_saves_and_reloads_on_the_cpu(tmp_path):
    out = _run(BVE, "prepare", "state", block_cupy=True, cwd=tmp_path)
    assert "saved" in out
    out = _run(BVE, "inspect", "state", block_cupy=True, cwd=tmp_path)
    assert "cupy imported: False" in out and "ideal_sphere" in out

    from tropoi.run.custom import load_initial_state

    state = load_initial_state(tmp_path / "state")
    rng = np.random.default_rng(0)
    lat = rng.uniform(-1.5, 1.5, 200)
    lon = rng.uniform(0, 2 * np.pi, 200)
    nu = k = 1.5e-5
    analytic = (2 * nu * np.sin(lat)
                - 30 * k * np.sin(lat) * np.cos(lat) ** 4 * np.sin(4 * lon))
    np.testing.assert_allclose(_synthesize(np.asarray(state.coeffs), lat, lon),
                               analytic, rtol=0, atol=1e-12 * 30 * k)
    assert state.planet.dynamical_radius_m == 3.0e6


@pytest.mark.skipif(not _has_cuda(), reason="needs a CUDA GPU")
@pytest.mark.parametrize("script", [BVE, SWE], ids=["bve", "swe"])
def test_examples_run_end_to_end(tmp_path, script):
    _run(script, "prepare", "state", cwd=tmp_path)
    assert "cupy imported: False" in _run(script, "inspect", "state",
                                          block_cupy=True, cwd=tmp_path)
    out = _run(script, "run", "state", "--out", "runs", "--days", "0.1",
               cwd=tmp_path)
    assert "run " in out

    from tropoi.representation.archive import open_simulation
    (run_dir,) = (tmp_path / "runs" / "examples").iterdir()
    sim = open_simulation(run_dir)
    assert sim.metadata["status"] == "completed"
    assert math.isclose(sim.times[-1], 0.1 * 86400.0)
