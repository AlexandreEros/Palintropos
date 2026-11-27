"""Global discretizations versus diagnostic point collections in the airless
runner: area weights, quadrature identities and the output schema.

The expected values are computed here from closed-form expressions (band
edges and sin(phi) integrals), independently of the runner's own code.
"""
import json
import math

import numpy as np
import pytest

from tropoi.run.airless.config import mercury_vpw99
from tropoi.run.airless.experiment import AirlessRun, build_columns, load_tier2

EPS = np.finfo(float).eps


def _latlon(nlat, nlon):
    return mercury_vpw99(columns={"kind": "latlon", "nlat": nlat, "nlon": nlon})


def _independent_bands(nlat, R):
    """Band edges, centres and exact band areas per unit longitude, from first
    principles: area of a band = R^2 dlambda (sin phi_N - sin phi_S)."""
    edges = np.radians(np.linspace(-90.0, 90.0, nlat + 1))
    centres = 0.5 * (edges[1:] + edges[:-1])
    return edges, centres, R * R * (np.sin(edges[1:]) - np.sin(edges[:-1]))


@pytest.mark.parametrize("nlat,nlon", [(6, 8), (45, 90), (90, 180)])
def test_regular_grid_weights_are_exact_cell_areas_summing_to_the_sphere(nlat, nlon):
    cfg = _latlon(nlat, nlon)
    c = build_columns(cfg)
    lat, lon, w = c.lat_deg, c.lon_deg, c.weights
    R = cfg.radius_m
    _, _, band = _independent_bands(nlat, R)
    expected = np.repeat(band * (2.0 * math.pi / nlon), nlon)
    # each weight is a few roundings of the same expression
    np.testing.assert_allclose(w, expected, rtol=8 * EPS, atol=0)
    # a sum of n positive terms carries at most about n eps relative error
    assert abs(w.sum() / (4.0 * math.pi * R * R) - 1.0) <= w.size * EPS
    assert build_columns(cfg).kind == "global"


@pytest.mark.parametrize("nlat,nlon", [(6, 8), (90, 180)])
def test_area_weighted_quadrature_identities(nlat, nlon):
    cfg = _latlon(nlat, nlon)
    c = build_columns(cfg)
    lat, w = c.lat_deg, c.weights
    R = cfg.radius_m
    area = 4.0 * math.pi * R * R
    phi = np.radians(lat)
    # a uniform field: weighted mean equals the value
    assert abs(w @ np.full(lat.size, 273.15) / w.sum() - 273.15) <= 273.15 * lat.size * EPS
    # sin(phi) is odd about the equator: its integral vanishes to round-off
    assert abs(w @ np.sin(phi)) <= area * lat.size * EPS
    # sin^2(phi): the midpoint value times the exact band area, summed, has a
    # closed form; it converges to the exact mean 1/3 at second order
    edges, centres, band = _independent_bands(nlat, R)
    discrete = (np.sin(centres) ** 2 * band).sum() * 2.0 * math.pi / area
    assert abs(w @ np.sin(phi) ** 2 / area - discrete) <= lat.size * 4 * EPS


def test_sin_squared_mean_converges_at_second_order():
    errors = []
    for nlat in (20, 40, 80):
        cfg = _latlon(nlat, 4)
        c = build_columns(cfg)
        lat, w = c.lat_deg, c.weights
        errors.append(abs(w @ np.sin(np.radians(lat)) ** 2 / w.sum() - 1.0 / 3.0))
    ratios = [errors[i] / errors[i + 1] for i in range(2)]
    assert all(3.9 < r < 4.1 for r in ratios), ratios


def test_points_are_a_diagnostic_collection_without_weights():
    cfg = mercury_vpw99(columns={"kind": "points", "lat_deg": [0.0, 30.0], "lon_deg": [0.0, 90.0]})
    c = build_columns(cfg)
    lat, lon, w = c.lat_deg, c.lon_deg, c.weights
    assert w is None
    assert build_columns(cfg).kind == "diagnostic"


def _short(cfg):
    return cfg.with_(cycle_steps=24, spinup_cycle_steps=None,
                     output={**cfg.output, "surface_samples": 12, "profile_samples": 12})


def test_point_run_publishes_per_point_diagnostics_and_nothing_global(tmp_path):
    cfg = _short(mercury_vpw99(columns={"kind": "points", "lat_deg": [0.0, 45.0, -60.0],
                                        "lon_deg": [10.0, 200.0, 300.0]}))
    AirlessRun(cfg, tmp_path / "run", allow_dirty=True, log=lambda s: None,
               workers=2).run_transient(30)
    grid = np.load(tmp_path / "run" / "grid.npz")
    assert "weights" not in grid.files and str(grid["column_kind"]) == "diagnostic"
    meta = json.loads((tmp_path / "run" / "config.json").read_text())
    assert meta["column_kind"] == "diagnostic"
    cycles = json.loads((tmp_path / "run" / "cycles.json").read_text())
    for c in cycles:
        assert c["column_kind"] == "diagnostic"
        assert not any(k.startswith(("mean_", "global_")) for k in c)
        assert [p["column"] for p in c["points"]] == [0, 1, 2]
        assert [p["lat_deg"] for p in c["points"]] == [0.0, 45.0, -60.0]
    events = [json.loads(l) for l in (tmp_path / "run" / "events.jsonl").read_text().splitlines()]
    assert not any(k.startswith(("mean_", "global_")) for e in events for k in e)
    series = np.load(tmp_path / "run" / "series" / "cycle-0000.npz")
    assert set(series.files) == {"iterations"} | {f"per_point_{k}_W_m2" for k in
                                                  ("energy_tendency", "net", "residual",
                                                   "tolerance")}
    assert series["per_point_net_W_m2"].shape == (24, 3)
    # the per-point net flux series integrates to the per-point cycle integrals
    ints = np.load(tmp_path / "run" / "cycles" / "cycle-0000" / "integrals.npz")
    dt = cfg.cycle_s / cfg.cycle_steps
    # round-off scale: 24 accumulated steps of terms as large as absorbed and
    # emitted, which nearly cancel in the net
    scale = np.abs(ints["absorbed"]) + np.abs(ints["emitted"]) + np.abs(ints["bottom"])
    np.testing.assert_array_less(np.abs(dt * series["per_point_net_W_m2"].sum(axis=0)
                                        - (ints["absorbed"] - ints["emitted"] + ints["bottom"])),
                                 64 * EPS * scale)
    # with two workers the points are split across blocks and rejoined in order
    seg1 = np.load(tmp_path / "run" / "series" / "cycle-0001.npz")
    assert seg1["per_point_net_W_m2"].shape == (6, 3)


def test_global_series_are_area_integrals_of_the_per_point_physics(tmp_path):
    """Columns are independent, so a global run's integrals equal the area
    weights (computed here) times a point run's per-point series at the same
    coordinates."""
    g = _short(_latlon(4, 6))
    c = build_columns(g)
    lat, lon = c.lat_deg, c.lon_deg
    AirlessRun(g, tmp_path / "g", allow_dirty=True, log=lambda s: None).run_transient(24)
    p = g.with_(columns={"kind": "points", "lat_deg": lat.tolist(), "lon_deg": lon.tolist()},
                output={**g.output, "every_step_profiles": False})
    AirlessRun(p, tmp_path / "p", allow_dirty=True, log=lambda s: None).run_transient(24)
    _, _, band = _independent_bands(4, g.radius_m)
    w = np.repeat(band * (2.0 * math.pi / 6), 6)
    gs = np.load(tmp_path / "g" / "series" / "cycle-0000.npz")
    ps = np.load(tmp_path / "p" / "series" / "cycle-0000.npz")
    for k in ("energy_tendency", "net", "residual", "tolerance"):
        np.testing.assert_allclose(gs[k], ps[f"per_point_{k}_W_m2"] @ w, rtol=1e-12,
                                   atol=1e-12 * np.max(np.abs(gs[k])))
    gc = json.loads((tmp_path / "g" / "cycles.json").read_text())[0]
    assert gc["column_kind"] == "global" and "mean_absorbed_W_m2" in gc
    np.testing.assert_array_equal(load_tier2(tmp_path / "g" / "retained", "profiles"),
                                  load_tier2(tmp_path / "p" / "retained", "profiles"))


def test_view_builder_refuses_a_diagnostic_collection(tmp_path):
    from tools.planet_one.build_view import build
    cfg = _short(mercury_vpw99(columns={"kind": "points", "lat_deg": [0.0], "lon_deg": [0.0]}))
    AirlessRun(cfg, tmp_path / "run", allow_dirty=True, log=lambda s: None).run_transient(24)
    with pytest.raises(SystemExit, match="lat/lon run"):
        build(tmp_path / "run", 2, {})
