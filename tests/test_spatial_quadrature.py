"""Canonical horizontal quadrature (``tropoi.spatial.grids.quadrature``) and
its use by the airless runner: Gauss-Legendre exactness, ordering, physical
weights, column-set kinds, CPU-only import, and identity with the spectral
grid's geometry.

Expected values here are closed forms (polynomial moments, sphere areas),
not the module's own output.
"""
import json
import math
import subprocess
import sys

import numpy as np
import pytest

from tropoi.spatial.grids.quadrature import (ColumnSet, gauss_latitude_nodes, gauss_latitudes,
                                             gauss_legendre, gauss_longitudes)

EPS = np.finfo(float).eps


@pytest.mark.parametrize("n", [2, 5, 32, 64])
def test_gauss_legendre_integrates_polynomials_up_to_degree_2n_minus_1(n):
    x, w = gauss_legendre(n)
    for k in range(2 * n):
        exact = 2.0 / (k + 1) if k % 2 == 0 else 0.0
        assert abs(w @ x ** k - exact) <= 64 * n * EPS, k
    # and not degree 2n, where the remainder is large enough to resolve (it
    # falls below round-off for large n)
    if n <= 5:
        assert abs(w @ x ** (2 * n) - 2.0 / (2 * n + 1)) > 1e3 * n * EPS


def test_gauss_nodes_run_north_to_south_and_longitudes_start_at_zero():
    x, w = gauss_latitude_nodes(12)
    assert np.all(np.diff(x) < 0) and np.all(w > 0)
    lat = gauss_latitudes(12)
    np.testing.assert_allclose(np.sin(lat), x, rtol=0, atol=4 * EPS)
    lon = gauss_longitudes(8)
    assert lon[0] == 0.0
    np.testing.assert_allclose(lon, 2 * np.pi * np.arange(8) / 8, rtol=0, atol=4 * EPS)


@pytest.mark.parametrize("nlat,nlon,R", [(32, 64, 6.371e6), (64, 128, 2.4405e6), (48, 96, 1.0)])
def test_gauss_column_weights_are_r2_dlambda_wi_and_tile_the_sphere(nlat, nlon, R):
    c = ColumnSet.gauss(nlat, nlon, R)
    x, w = np.polynomial.legendre.leggauss(nlat)          # independent of the module's ordering
    w_sorted = w[::-1]                                      # leggauss is ascending; rows are descending
    np.testing.assert_allclose(c.weights.reshape(nlat, nlon),
                               np.broadcast_to((R * R * 2 * np.pi / nlon * w_sorted)[:, None],
                                               (nlat, nlon)), rtol=4 * EPS)
    assert abs(c.weights.sum() / (4 * math.pi * R * R) - 1.0) <= c.size * EPS
    # lat-major C order: each row of nlon columns shares a latitude
    la = c.lat.reshape(nlat, nlon)
    assert np.all(la == la[:, :1]) and np.all(np.diff(la[:, 0]) < 0)
    assert c.kind == "global" and c.shape == (nlat, nlon)
    # area-weighted mean of sin^2(lat) is exactly 1/3 (degree 2 in sin(lat))
    assert abs(c.weights @ np.sin(c.lat) ** 2 / c.weights.sum() - 1.0 / 3.0) <= c.size * 4 * EPS


def test_column_set_kinds_are_enforced():
    g = ColumnSet.regular(3, 4, 1.0)
    with pytest.raises(ValueError, match="global column set has weights"):
        ColumnSet(g.lat, g.lon, g.lat_deg, g.lon_deg, "global", None, g.shape, "x", 1.0)
    with pytest.raises(ValueError, match="global column set has weights"):
        ColumnSet(g.lat, g.lon, g.lat_deg, g.lon_deg, "diagnostic", g.weights, None, "x", 1.0)
    with pytest.raises(ValueError, match="sum to the sphere"):
        ColumnSet(g.lat, g.lon, g.lat_deg, g.lon_deg, "global", 2 * g.weights, g.shape, "x", 1.0)
    p = ColumnSet.points([0.0, 10.0], [5.0, 6.0], 1.0)
    assert p.kind == "diagnostic" and p.weights is None and p.shape is None
    with pytest.raises(ValueError):
        ColumnSet.points([], [], 1.0)


def test_cpu_only_modules_do_not_import_cupy():
    code = ("import sys; sys.modules['cupy'] = None\n"
            "import tropoi.spatial.grids.quadrature, tropoi.spatial.vertical\n"
            "import tropoi.run.airless.experiment, tropoi.representation.diagnostics.held_suarez_sections\n"
            "print('ok')")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       env={**__import__('os').environ})
    assert r.returncode == 0 and r.stdout.strip() == "ok", r.stderr


def test_held_suarez_section_latitudes_are_the_grid_construction():
    from tropoi.representation.diagnostics.held_suarez_sections import gaussian_latitudes_deg
    x, _ = np.polynomial.legendre.leggauss(24)
    previous = np.degrees(np.pi / 2.0 - np.arccos(x[np.argsort(-x)]))   # the former inline form
    np.testing.assert_array_equal(gaussian_latitudes_deg(24), previous)


def test_gauss_columns_are_the_spectral_grid_bit_for_bit():
    cp = pytest.importorskip("cupy", reason="the spectral grid geometry needs CuPy")
    try:                       # can raise (e.g. cudaErrorInsufficientDriver), not just be 0
        cp.cuda.runtime.getDeviceCount()
    except Exception:
        pytest.skip("no usable CUDA device")
    from tropoi.spatial.grids.latlon_grid import GaussLatLonGridGeometry
    for nlat, nlon, R in ((32, 64, 6.371e6), (64, 128, 9.0e6)):
        g = GaussLatLonGridGeometry(nlat, nlon, radius=R)
        c = ColumnSet.gauss(nlat, nlon, R)
        np.testing.assert_array_equal(c.lat, g.point_latitudes.get())
        np.testing.assert_array_equal(c.lon, g.point_longitudes.get())
        np.testing.assert_array_equal(c.weights, g.cell_areas.get())


def test_airless_run_on_gauss_columns_and_its_replay(tmp_path):
    from tools.planet_one.global_check import main as gg
    from tropoi.run.airless.config import mercury_vpw99
    from tropoi.run.airless.experiment import AirlessRun, build_columns
    cfg = mercury_vpw99(columns={"kind": "gauss", "nlat": 4, "nlon": 8})
    cfg = cfg.with_(cycle_steps=24, spinup_cycle_steps=None,
                    output={**cfg.output, "surface_samples": 12, "profile_samples": 12})
    cols = build_columns(cfg)
    assert cols.kind == "global"
    np.testing.assert_array_equal(cols.lat, ColumnSet.gauss(4, 8, cfg.radius_m).lat)
    AirlessRun(cfg, tmp_path / "run", backend="numpy", allow_dirty=True, log=lambda s: None,
               workers=2).run_transient(24)
    grid = np.load(tmp_path / "run" / "grid.npz")
    assert str(grid["column_kind"]) == "global"
    np.testing.assert_array_equal(grid["lat_rad"], cols.lat)       # radians stored, not re-derived
    np.testing.assert_array_equal(grid["weights"], cols.weights)
    cyc = json.loads((tmp_path / "run" / "cycles.json").read_text())[0]
    assert "mean_absorbed_W_m2" in cyc
    assert gg([str(tmp_path / "run"), "--columns", "0,9,31"]) == 0
    report = json.loads((tmp_path / "run" / "gg.json").read_text())
    assert report["pass"] and report["max_abs_end_state_diff_K"] == 0.0
