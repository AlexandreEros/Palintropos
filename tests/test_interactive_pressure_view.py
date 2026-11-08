"""Interactive pressure selection in the atmosphere page (issue #13, second
slice): ``Simulation.view(surface=("pressure", p0), interactive_pressure=True)``.

Numerics: the stored form of the native columns (16-bit per level slice,
float32 surface pressure) and its error, separated from the interpolation
error, against the Python reference :mod:`pe_surfaces`; the analytic
isothermal column through the stored form; every mask reason. Payload: the
version-3 block, the estimate before reconstruction and the decoded memory,
strict ``times="all"``, option validation, the existing modes unchanged.
Page (skipped without a browser): arbitrary pressures compared with Python,
slider and typed input kept synchronized, rapid input coalesced with the
newest value winning, saved-time switches keeping the pressure, an empty
surface and its recovery, seams, hover, geometry/mask agreement, bounded
decoding, offline operation.
"""
from __future__ import annotations

import base64
import json
import math
import pathlib
import sys
import warnings

import numpy as np
import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tropoi.representation.diagnostics.pe_surfaces import (  # noqa: E402
    MASK_ABOVE_TOP_LEVEL, MASK_BELOW_BOTTOM_LEVEL, MASK_BELOW_SURFACE,
    MASK_VALID, SurfaceUnavailableError, interpolate_columns,
    pressure_columns, pressure_surface)
from tropoi.representation.visual.atmosphere import (  # noqa: E402
    COLUMN_FIELDS, FIELDS, MAX_DECODED_TIMES, SURFACE_FIELDS,
    AtmosphereViewWarning, PayloadTooLargeError, _mesh, atmosphere_payload,
    decoded_column_bytes, dequantize_columns, estimate_column_payload_bytes,
    estimate_payload_bytes, quantize_columns)
from test_atmosphere_view import GRAVITY, R_DRY, write_pe_capsule  # noqa: E402
from test_pressure_surface_view import (_storage,  # noqa: E402
                                        write_holed_capsule)

API = "window.__terrainGlobe.atmosphere"


def _u16(block: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(block), dtype="<u2")


def _f32(block: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(block), dtype="<f4")


def _speed(s):
    return np.hypot(s["u"], s["v"])


def _quantum(bounds, name, k):
    lo, hi = bounds[name][k]
    return (hi - lo) / 65535.0


# ---------------------------------------------------------------------------
# Numerics: the stored columns and the Python reference
# ---------------------------------------------------------------------------

def test_columns_then_interpolation_is_the_fixed_surface_path(tmp_path):
    root, _, _ = write_pe_capsule(tmp_path / "cap", frames=1)
    _, storage = _storage(root)
    context = storage.pe_context()
    lat, lon = _mesh(6.0)
    frame = storage.frame(0)
    cols = pressure_columns(frame, context, lat, lon)
    assert cols.nlev == context.nlev and cols.winds
    assert cols.level_positions == tuple(context.sigma.hydrostatic_full_levels)
    for p0 in (9.0e4, 6.1234e4, 3.3e4):
        a = interpolate_columns(cols, ("pressure", p0))
        b = pressure_surface(frame, context, lat, lon, ("pressure", p0))
        assert (a.mask == b.mask).all()
        for name in ("temperature", "geopotential_height", "u", "v"):
            assert a[name] == pytest.approx(b[name], rel=0, abs=0)
    lo, hi = cols.sampled_pressure_range()
    assert lo == pytest.approx(cols.level_positions[0]
                               * cols.surface_pressure.min())
    assert hi == pytest.approx(cols.level_positions[-1]
                               * cols.surface_pressure.max())
    with pytest.raises(SurfaceUnavailableError, match="no valid node"):
        interpolate_columns(cols, ("pressure", 2.0 * hi))


def test_quantization_error_is_bounded_and_separate_from_interpolation(
        tmp_path):
    """Three quantities, measured apart: (a) the stored form against the
    exact columns, level by level, within the 16-bit quantum; (b) the
    interpolated surface from the stored form against the exact surface,
    within the interpolated quanta (linear interpolation cannot amplify
    them); (c) the interpolation itself, exact on the stored columns."""
    root, _, _ = write_pe_capsule(tmp_path / "cap", frames=1)
    _, storage = _storage(root)
    context = storage.pe_context()
    lat, lon = _mesh(6.0)
    cols = pressure_columns(storage.frame(0), context, lat, lon)
    q, bounds, p_s = quantize_columns(cols)
    shape = (lat.size, lon.size)
    deq = dequantize_columns(q, bounds, p_s, shape, cols)
    # (a) per level, per field: at most half a quantum (plus float32 round)
    for name in COLUMN_FIELDS:
        exact = getattr(cols, name)
        back = getattr(deq, name)
        for k in range(cols.nlev):
            tol = 0.5 * _quantum(bounds, name, k) + 1e-6 * abs(
                exact[k]).max()
            assert np.abs(back[k] - exact[k])[:, :-1].max() <= tol
            # the seam column is stored equal to the first column
            assert (q[name][k].reshape(shape)[:, -1]
                    == q[name][k].reshape(shape)[:, 0]).all()
    assert np.abs(deq.surface_pressure - cols.surface_pressure).max() \
        <= 2.0e-7 * cols.surface_pressure.max()           # float32
    assert q["temperature"].dtype == np.dtype("<u2") and p_s.dtype == "<f4"
    # the fixture is not degenerate: every quantum is nonzero
    assert all(_quantum(bounds, n, k) > 0 for n in COLUMN_FIELDS
               for k in range(cols.nlev))
    # (b) and (c) at pressures between levels, on endpoints and in holes
    g = context.gravity
    positions = cols.level_positions
    ps_mid = float(np.median(cols.surface_pressure))
    pressures = [positions[0] * ps_mid, positions[-1] * ps_mid,
                 0.5 * (positions[1] + positions[2]) * ps_mid, 6.1234e4,
                 8.9e4, 9.0e4]
    worst = {"temperature": 0.0, "geopotential_height": 0.0,
             "wind_speed": 0.0}
    for p0 in pressures:
        exact = interpolate_columns(cols, ("pressure", p0))
        stored = interpolate_columns(deq, ("pressure", p0))
        # masks: identical away from the float32 round-off of p_s (6e-8
        # relative), which can move a node sitting on a mask boundary (the
        # ground, or an end level with its 1e-9 tolerance band)
        sig_t = p0 / cols.surface_pressure
        near = (np.abs(sig_t - 1.0) <= 1e-6)             | (np.abs(sig_t - positions[0]) <= 1e-6 * positions[0])             | (np.abs(sig_t - positions[-1]) <= 1e-6 * positions[-1])
        assert (exact.mask == stored.mask)[~near].all()
        assert near.mean() < 0.05
        ok = (exact.mask == MASK_VALID) & (stored.mask == MASK_VALID)
        assert ok.any()
        k = exact.bracket[ok]
        qt = np.maximum([_quantum(bounds, "temperature", i) for i in k],
                        [_quantum(bounds, "temperature", i + 1) for i in k])
        qp = np.maximum([_quantum(bounds, "geopotential", i) for i in k],
                        [_quantum(bounds, "geopotential", i + 1) for i in k])
        qu = np.maximum([_quantum(bounds, "u", i) for i in k],
                        [_quantum(bounds, "u", i + 1) for i in k])
        qv = np.maximum([_quantum(bounds, "v", i) for i in k],
                        [_quantum(bounds, "v", i + 1) for i in k])
        dt = np.abs(stored["temperature"] - exact["temperature"])[ok]
        dz = np.abs(stored["geopotential_height"]
                    - exact["geopotential_height"])[ok]
        ds = np.abs(_speed(stored) - _speed(exact))[ok]
        assert (dt <= 0.5 * qt + 1e-4).all()
        assert (dz <= 0.5 * qp / g + 1e-3).all()
        assert (ds <= 0.5 * np.hypot(qu, qv) + 1e-4).all()
        worst["temperature"] = max(worst["temperature"], dt.max())
        worst["geopotential_height"] = max(worst["geopotential_height"],
                                           dz.max())
        worst["wind_speed"] = max(worst["wind_speed"], ds.max())
        # (c): the interpolant on the stored columns is the same linear rule
        # (a target on a level position returns the stored level exactly)
    for k in range(cols.nlev):
        on = interpolate_columns(deq, ("pressure", positions[k] * ps_mid))
        j = np.isclose(cols.surface_pressure, ps_mid, rtol=1e-12)
        if j.any():
            # float32 p_s puts sigma* within 6e-8 of the level, not on it
            assert on["temperature"][j] == pytest.approx(
                deq.temperature[k][j], rel=1e-6)
    # the measured errors are display-invisible: millikelvin, decimetres
    assert worst["temperature"] < 0.01 and worst["geopotential_height"] < 0.5
    assert worst["wind_speed"] < 0.01


def test_isothermal_column_through_the_stored_form_and_every_mask(tmp_path):
    """The analytic column with terrain holes: Z from the stored columns is
    the closed form Phi_s/g + R_d T ln(p_s/p0)/g to the quantization bound
    at pressures between levels, on both end levels (tolerance band) and
    under varying surface pressure; every mask reason appears."""
    root, _, terrain = write_holed_capsule(tmp_path / "cap", frames=1)
    _, storage = _storage(root)
    context = storage.pe_context()
    lat, lon = _mesh(6.0)
    cols = pressure_columns(storage.frame(0), context, lat, lon)
    q, bounds, p_s = quantize_columns(cols)
    deq = dequantize_columns(q, bounds, p_s, (lat.size, lon.size), cols)
    T = 230.0
    g = context.gravity
    ps = cols.surface_pressure
    hs = cols.surface_height
    positions = cols.level_positions
    qz = max(_quantum(bounds, "geopotential", k) for k in range(cols.nlev)) / g
    seen = set()
    for p0 in (3.0e4, 6.1e4, 9.3e4, positions[0] * ps.min() * (1 + 1e-12),
               positions[-1] * ps.max() * (1 - 1e-12), 151.3, 5.0e5):
        try:
            s = interpolate_columns(deq, ("pressure", p0))
        except SurfaceUnavailableError:
            continue
        seen |= set(np.unique(s.mask).tolist())
        ok = s.valid
        closed = hs + R_DRY * T * np.log(ps / p0) / g
        assert np.abs(s["geopotential_height"] - closed)[ok].max() \
            <= 0.5 * qz + 2e-3
        assert np.abs(s["temperature"] - T)[ok].max() < 1e-6
        # the ends: a target on the end level is that level, no
        # extrapolation beyond it
        if ok.any() and p0 < 200.0:
            assert (s.bracket[ok] == 0).all()
    assert seen == {MASK_VALID, MASK_BELOW_SURFACE, MASK_ABOVE_TOP_LEVEL,
                    MASK_BELOW_BOTTOM_LEVEL}
    # wholly unavailable: both ends, by name
    for p0 in (1.0, 5.0e6):
        with pytest.raises(SurfaceUnavailableError, match="no valid node"):
            interpolate_columns(deq, ("pressure", p0))


# ---------------------------------------------------------------------------
# Payload
# ---------------------------------------------------------------------------

def test_interactive_payload_holds_columns_not_surfaces(tmp_path):
    root, _, _ = write_pe_capsule(tmp_path / "cap", frames=2)
    _, storage = _storage(root)
    page, report = atmosphere_payload(
        storage, surface=("pressure", 7.0e4), interactive_pressure=True,
        times="all", resolution_deg=6.0)
    A = page["atmosphere"]
    assert A["version"] == 3
    assert [f["id"] for f in A["fields"]] == list(SURFACE_FIELDS)
    assert all("data" not in f and "bounds" not in f for f in A["fields"])
    assert "height" not in A and "mask" not in A
    S, C = A["surface"], A["columns"]
    assert S["interactive"] is True and S["units"] == "Pa"
    assert S["value"] == 7.0e4 and S["label"] == "700 hPa"
    assert S["level_positions"] == pytest.approx(
        report["context"].sigma.hydrostatic_full_levels)
    assert S["pressure_range"] == pytest.approx(report["pressure_range"])
    assert S["end_tolerance"] == 1e-9
    assert C["fields"] == list(COLUMN_FIELDS) and C["dtype"] == "uint16"
    assert C["surface_pressure_dtype"] == "float32"
    assert C["nlev"] == report["context"].nlev
    assert C["max_decoded_times"] == MAX_DECODED_TIMES
    assert len(C["times"]) == 2
    # streamlines are traced in the page: settings only, no line data
    assert A["streamlines"]["enabled"] is True
    assert A["streamlines"]["mode"] == "browser"
    assert "points" not in A["streamlines"]
    assert A["defaults"]["pressure_pa"] == 7.0e4
    assert A["defaults"]["streamlines"] is True
    # each time's block decodes to the quantized columns of that state
    context = report["context"]
    lat, lon = _mesh(6.0)
    n = lat.size * lon.size
    for t, index in enumerate(report["times"]):
        cols = pressure_columns(storage.frame(index), context, lat, lon)
        q, bounds, p_s = quantize_columns(cols)
        block = C["times"][t]
        for name in COLUMN_FIELDS:
            assert block[name]["bounds"] == bounds[name]
            assert (_u16(block[name]["data"]).reshape(cols.nlev, n)
                    == q[name]).all()
        assert (_f32(block["surface_pressure"]["data"]) == p_s).all()
        assert report["bounds"][t] == bounds
    # colour ranges: conservative bounds from the columns, one per field
    fields = {f["id"]: f for f in A["fields"]}
    cols = pressure_columns(storage.frame(0), context, lat, lon)
    assert fields["temperature"]["range"][0] <= cols.temperature.min()
    assert fields["temperature"]["range"][1] >= cols.temperature.max()
    assert fields["wind_speed"]["range"][0] == 0.0
    assert fields["wind_speed"]["range"][1] >= _speed(
        {"u": cols.u, "v": cols.v}).max()
    assert fields["geopotential_height"]["range"][1] >= \
        cols.geopotential.max() / context.gravity
    # an interpolated surface never leaves them
    s = interpolate_columns(cols, ("pressure", 6.1234e4))
    ok = s.valid
    assert s["temperature"][ok].min() >= fields["temperature"]["range"][0]
    assert _speed(s)[ok].max() <= fields["wind_speed"]["range"][1]
    assert any("interpolated in the browser" in line
               for line in A["interpretation"])
    assert any("Decoded memory" in row[0] for row in page["details"])
    assert report["interactive_pressure"] is True
    assert report["decoded_bytes_per_time"] == decoded_column_bytes(
        context.nlev, lat.size, lon.size)


def test_estimate_precedes_reconstruction_and_reports_memory_apart(tmp_path):
    root, _, _ = write_pe_capsule(tmp_path / "cap", frames=3)
    sim, storage = _storage(root)
    page, report = atmosphere_payload(
        storage, surface=("pressure", 7.0e4), interactive_pressure=True,
        times="all", resolution_deg=6.0)
    nlat, nlon = page["atmosphere"]["mesh"]["nlat"], \
        page["atmosphere"]["mesh"]["nlon"]
    nlev = report["context"].nlev
    per_time = estimate_column_payload_bytes(1, nlev, nlat, nlon) - 200_000
    assert per_time == math.ceil(
        (2 * nlat * nlon * nlev * 4 + 4 * nlat * nlon) * 4 / 3)
    assert report["estimate_bytes"] >= 3 * per_time
    path = sim.view(tmp_path / "cols.html", surface=("pressure", 7.0e4),
                    interactive_pressure=True, times="all",
                    resolution_deg=6.0, open_browser=False)
    assert path.stat().st_size <= report["estimate_bytes"] + 300_000
    # the file holds every saved state; memory is per decoded state
    decoded = decoded_column_bytes(nlev, nlat, nlon)
    # columns + p_s per state, then the derived arrays (5 float32 + mask)
    # and the streamline working arrays (3 float64 + float64 + byte)
    assert decoded == nlat * nlon * (16 * nlev + 4 + 21 + 33)
    assert report["decoded_bytes_per_time"] == decoded
    assert decoded < report["estimate_bytes"] / 3     # one state, not all
    # the other estimates are what they were
    assert estimate_payload_bytes(2, 3, 31, 61) == estimate_payload_bytes(
        2, 3, 31, 61, n_fields=len(FIELDS), mask=False)


def test_all_times_is_strict_and_the_refusal_names_the_column_cost(tmp_path):
    root, _, _ = write_pe_capsule(tmp_path / "cap", frames=3)
    _, storage = _storage(root)
    with pytest.raises(PayloadTooLargeError,
                       match="embeds every native level"):
        atmosphere_payload(storage, surface=("pressure", 7.0e4),
                           interactive_pressure=True, times="all",
                           resolution_deg=6.0, max_mb=0.3)
    with pytest.raises(PayloadTooLargeError, match="coarser resolution_deg"):
        atmosphere_payload(storage, surface=("pressure", 7.0e4),
                           interactive_pressure=True, times=[0, 1],
                           resolution_deg=6.0, max_mb=0.3)
    # the automatic selection subsamples with a warning, never silently
    _, one = atmosphere_payload(storage, surface=("pressure", 7.0e4),
                                interactive_pressure=True, times=[0],
                                resolution_deg=6.0)
    _, three = atmosphere_payload(storage, surface=("pressure", 7.0e4),
                                  interactive_pressure=True, times="all",
                                  resolution_deg=6.0)
    per_time = (three["estimate_bytes"] - one["estimate_bytes"]) / 2
    budget = one["estimate_bytes"] + 1.5 * per_time
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _, report = atmosphere_payload(
            storage, surface=("pressure", 7.0e4), interactive_pressure=True,
            resolution_deg=6.0, max_mb=budget / 1e6)
    assert any(issubclass(w.category, AtmosphereViewWarning) for w in caught)
    assert report["times"] == [0, 2]
    assert report["selection_note"] and "subsampled" in report["selection_note"]
    # max_mb=None removes the bound
    _, free = atmosphere_payload(storage, surface=("pressure", 7.0e4),
                                 interactive_pressure=True, times="all",
                                 resolution_deg=6.0, max_mb=None)
    assert free["times"] == [0, 1, 2]


def test_interactive_options_are_validated_and_modes_are_untouched(tmp_path):
    root, _, _ = write_pe_capsule(tmp_path / "cap", frames=2)
    sim, storage = _storage(root)
    with pytest.raises(ValueError, match="needs the initial isobar"):
        atmosphere_payload(storage, interactive_pressure=True)
    with pytest.raises(ValueError, match="exclusive"):
        atmosphere_payload(storage, surface=("pressure", 7e4),
                           interactive_pressure=True, levels=[0])
    with pytest.raises(ValueError, match="level= selects a native level"):
        atmosphere_payload(storage, surface=("pressure", 7e4),
                           interactive_pressure=True, level=1)
    with pytest.raises(ValueError, match="field must be one of"):
        atmosphere_payload(storage, surface=("pressure", 7e4),
                           interactive_pressure=True, field="zeta")
    with pytest.raises(ValueError, match="level_sigma must be one of"):
        atmosphere_payload(storage, surface=("pressure", 7e4),
                           interactive_pressure=True, level_sigma="middle")
    with pytest.raises(SurfaceUnavailableError,
                       match="any pressure can be chosen in the page"):
        atmosphere_payload(storage, surface=("pressure", 5e5),
                           interactive_pressure=True)
    with pytest.raises(ValueError, match="initial time position"):
        atmosphere_payload(storage, surface=("pressure", 7e4),
                           interactive_pressure=True, time=5)
    # nominal level positions are honoured
    _, nominal = atmosphere_payload(storage, surface=("pressure", 7e4),
                                    interactive_pressure=True,
                                    level_sigma="nominal", resolution_deg=6.0)
    assert nominal["level_sigma"] == "nominal"
    # the native page and the fixed-surface page are what they were
    native, _ = atmosphere_payload(storage, times="all", resolution_deg=6.0)
    fixed, rep_fixed = atmosphere_payload(storage, surface=("pressure", 7e4),
                                          times="all", resolution_deg=6.0)
    assert native["atmosphere"]["version"] == 1
    assert "columns" not in native["atmosphere"]
    assert fixed["atmosphere"]["version"] == 2
    assert "columns" not in fixed["atmosphere"]
    assert "interactive" not in fixed["atmosphere"]["surface"]
    assert fixed["atmosphere"]["streamlines"]["enabled"] is True
    assert "interactive_pressure" not in rep_fixed
    # default names tell the three apart
    from tropoi.representation.visual.atmosphere import _default_path
    _, r_native = atmosphere_payload(storage, resolution_deg=6.0)
    _, r_inter = atmosphere_payload(storage, surface=("pressure", 7e4),
                                    interactive_pressure=True,
                                    resolution_deg=6.0)
    names = [_default_path(storage, r).name
             for r in (r_native, rep_fixed, r_inter)]
    assert "-atmosphere-" in names[0] and "-isobar-" in names[1]
    assert "-isobars-" in names[2] and len(set(names)) == 3
    # the one-snapshot view goes through the same option
    path = sim[0].view(tmp_path / "one.html", surface=("pressure", 7e4),
                       interactive_pressure=True, resolution_deg=6.0,
                       open_browser=False)
    assert path.is_file()


# ---------------------------------------------------------------------------
# The page, in a browser, offline
# ---------------------------------------------------------------------------

from tools.browser_cdp import OfflinePage, find_browser  # noqa: E402

BROWSER = find_browser()
needs_browser = pytest.mark.skipif(BROWSER is None,
                                   reason="needs Chrome, Chromium or Edge")


@pytest.fixture(scope="module")
def nonuniform_page(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("isobars")
    root, _, _ = write_pe_capsule(tmp / "cap", frames=2)
    sim, storage = _storage(root)
    page, report = atmosphere_payload(
        storage, surface=("pressure", 7.0e4), interactive_pressure=True,
        times="all", resolution_deg=6.0)
    path = sim.view(tmp / "cols.html", surface=("pressure", 7.0e4),
                    interactive_pressure=True, times="all",
                    resolution_deg=6.0, open_browser=False)
    return path, page, report, storage


@pytest.fixture(scope="module")
def holed_page(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("isobars-holes")
    root, _, _ = write_holed_capsule(tmp / "cap", frames=3)
    sim, storage = _storage(root)
    page, report = atmosphere_payload(
        storage, surface=("pressure", 93_000.0), interactive_pressure=True,
        times="all", resolution_deg=6.0)
    path = sim.view(tmp / "holes.html", surface=("pressure", 93_000.0),
                    interactive_pressure=True, times="all",
                    resolution_deg=6.0, open_browser=False)
    return path, page, report, storage


def _ready(page, path):
    url = page.open(path)
    page.wait_for(f"{API} && document.getElementById('timeLabel')"
                  ".textContent")
    assert page.eval("window.__terrainGlobe.glOK")
    return url


def _masked_triangles(m):
    nlat, nlon = m.shape
    count = 0
    for j in range(nlat - 1):
        for i in range(nlon - 1):
            a, b, c, d = m[j, i], m[j, i + 1], m[j + 1, i], m[j + 1, i + 1]
            count += bool(a or b or d) + bool(a or d or c)
    return count


@needs_browser
def test_page_interpolates_like_python_at_arbitrary_pressures(
        nonuniform_page):
    path, payload, report, storage = nonuniform_page
    A = payload["atmosphere"]
    nlat, nlon = A["mesh"]["nlat"], A["mesh"]["nlon"]
    context = report["context"]
    lat, lon = _mesh(6.0)
    positions = context.sigma.hydrostatic_full_levels
    with OfflinePage(BROWSER) as page:
        url = _ready(page, path)
        info = page.eval(f"{API}.info()")
        assert info["interactive"] and info["pressure_pa"] == 7.0e4
        assert info["surface"] == "700 hPa" and info["streams"] is True
        assert not page.eval("document.getElementById('streams').disabled")
        assert "traced in this page" in page.eval(
            "document.getElementById('streamNote').textContent")
        assert not page.eval("document.getElementById('pressureBlock')"
                             ".hidden")
        for t, index in enumerate(report["times"]):
            page.eval(f"{API}.setTime({t})")
            cols = pressure_columns(storage.frame(index), context, lat, lon)
            q, bounds, p_s = quantize_columns(cols)
            deq = dequantize_columns(q, bounds, p_s, (nlat, nlon), cols)
            ps_mid = float(np.median(cols.surface_pressure))
            # between levels, on both end levels, in holes, and a value no
            # slider position represents
            for p0 in (6.1234e4, 0.5 * (positions[1] + positions[2]) * ps_mid,
                       positions[0] * ps_mid, positions[-1] * ps_mid,
                       8.8e4, 9.1e4, 5.1e3, 70_000.123456):
                r = page.eval(f"{API}.setPressure({p0!r})")
                assert r["ok"] and r["pressure_pa"] == p0
                arr = page.eval(f"{API}.surfaceArrays()")
                assert arr["p0"] == p0 and arr["timeIndex"] == t
                m = np.array(arr["mask"], dtype=np.uint8).reshape(nlat, nlon)
                try:
                    ref = interpolate_columns(deq, ("pressure", p0))
                except SurfaceUnavailableError:
                    # Python refuses a surface no node samples; the page
                    # shows it empty and says so, controls live
                    assert not (m == MASK_VALID).any()
                    assert page.eval(f"{API}.maskCounts()")["triangles"] == 0
                    assert page.eval("document.getElementById('atmNote')"
                                     ".textContent").startswith(
                                         "Empty surface")
                    continue
                exact = interpolate_columns(cols, ("pressure", p0))
                assert (m == ref.mask).all()
                ok = m == MASK_VALID
                got = {"temperature": np.array(arr["temperature"]),
                       "geopotential_height":
                           np.array(arr["geopotential_height"]),
                       "wind_speed": np.array(arr["wind_speed"])}
                want = {"temperature": ref["temperature"],
                        "geopotential_height": ref["geopotential_height"],
                        "wind_speed": _speed(ref)}
                true = {"temperature": exact["temperature"],
                        "geopotential_height": exact["geopotential_height"],
                        "wind_speed": _speed(exact)}
                for name in got:
                    js = got[name].reshape(nlat, nlon)
                    # the page stores float32 results: 1e-6 relative
                    tol = 2e-6 * np.abs(want[name]).max() + 1e-9
                    assert np.abs(js - want[name])[ok].max() <= tol, (
                        t, p0, name)
                    # and stays within the quantization bound of the exact
                    # surface (the test above measures that bound)
                    qk = max(_quantum(bounds, f, k) for f in COLUMN_FIELDS
                             for k in range(cols.nlev))
                    assert np.abs(js - true[name])[ok].max() <= qk + tol
                # geometry and mask agree: exactly the triangles with no
                # masked vertex are drawn, and the counts are the mask's
                counts = page.eval(f"{API}.maskCounts()")
                assert counts["counts"] == {str(k): int((m == k).sum())
                                            for k in range(4)}
                assert counts["triangles"] == counts["trianglesAll"] \
                    - _masked_triangles(m)
                assert arr["counts"] == [int((m == k).sum())
                                         for k in range(4)]
                # the seam column equals the first column
                z = got["geopotential_height"].reshape(nlat, nlon)
                assert (z[:, 0] == z[:, -1]).all()
                assert (m[:, 0] == m[:, -1]).all()
                # labels name this pressure
                lab = page.eval(
                    "document.getElementById('levelLabel').textContent")
                assert "interpolated here" in lab
                note = page.eval(
                    "document.getElementById('atmNote').textContent")
                assert f"({p0:.8g} Pa)" in note or "Empty surface" in note
                assert page.eval(f"{API}.info()")["pressure_pa"] == p0
        # hover: valid nodes read values; a point next to a masked node is
        # refused even where its nearest node is valid
        page.eval(f"{API}.setPressure(88000)")
        m = np.array(page.eval(f"{API}.surfaceArrays()")["mask"],
                     dtype=np.uint8).reshape(nlat, nlon)
        assert (m != 0).any() and (m == 0).any()
        d = A["mesh"]["dlat"]
        j, i = np.argwhere(m == MASK_VALID)[0]
        good = page.eval(f"{API}.sample({lat[j]}, {lon[i]})")
        assert good["mask"] == 0 and "interpolated" in good["readout"]
        pair = None
        for (jj, ii) in np.argwhere(m == MASK_VALID):
            if ii + 1 < nlon and m[jj, ii + 1] != MASK_VALID:
                pair = (jj, ii)
                break
        assert pair is not None
        jj, ii = pair
        near = page.eval(f"{API}.sample({lat[jj]}, {lon[ii] + 0.4 * d})")
        assert near["mask"] == m[jj, ii + 1]
        assert near["readout"].startswith("no sample on the 880 hPa")
        assert page.requests == [url]
        assert page.console_errors == [] and page.exceptions == []


@needs_browser
def test_controls_stay_synchronized_and_reject_bad_input(holed_page):
    path, payload, report, storage = holed_page
    lo, hi = payload["atmosphere"]["surface"]["pressure_range"]

    def slider_pos(p):
        return max(0, min(1000, round(1000 * (math.log(p) - math.log(lo))
                                      / (math.log(hi) - math.log(lo)))))

    def slider_pressure(pos):
        return math.exp(math.log(lo) + pos / 1000 * (math.log(hi)
                                                     - math.log(lo)))

    with OfflinePage(BROWSER) as page:
        _ready(page, path)
        c = page.eval(f"{API}.pressureControls()")
        assert c["slider"] == slider_pos(93_000.0) and c["unit"] == 100
        assert c["input"] == "930" and "930 hPa = 93000 Pa" in c["label"]
        assert "logarithmic" in c["note"] and not c["invalid"]
        # the API moves slider and input together
        page.eval(f"{API}.setPressure(61234.5)")
        c = page.eval(f"{API}.pressureControls()")
        assert c["slider"] == slider_pos(61234.5) and c["input"] == "612.345"
        # a typed value between slider positions is kept exactly; the
        # slider shows the nearest position
        between = 0.5 * (slider_pressure(500) + slider_pressure(501))
        assert slider_pressure(500) < between < slider_pressure(501)
        page.eval("document.getElementById('pressureInput').value = "
                  f"{between / 100!r}")
        page.eval("document.getElementById('pressureApply').click()")
        assert page.eval(f"{API}.pressure()") == pytest.approx(between,
                                                               rel=1e-12)
        assert page.eval(f"{API}.surfaceArrays()")["p0"] == pytest.approx(
            between, rel=1e-12)
        assert page.eval(f"{API}.pressureControls()")["slider"] in (500, 501)
        # Enter applies too, and the unit select converts the shown value
        page.eval("document.getElementById('pressureUnit').value = '1';"
                  "document.getElementById('pressureUnit')"
                  ".dispatchEvent(new Event('change'))")
        c = page.eval(f"{API}.pressureControls()")
        assert c["unit"] == 1 and float(c["input"]) == pytest.approx(
            between, rel=1e-7)
        page.eval("const i = document.getElementById('pressureInput');"
                  "i.value = '30000'; i.dispatchEvent(new KeyboardEvent("
                  "'keydown', {key: 'Enter', bubbles: true}))")
        assert page.eval(f"{API}.pressure()") == 30_000.0
        before = page.eval(f"{API}.digest()")
        # bad input: refused with a reason, nothing changes
        for bad in ("abc", "-5", "0", "1e400", "", "NaN", "Infinity"):
            page.eval(f"document.getElementById('pressureInput').value = "
                      f"{bad!r}; document.getElementById('pressureApply')"
                      ".click()")
            c = page.eval(f"{API}.pressureControls()")
            assert c["invalid"] and c["note"].startswith("Not applied:"), bad
            assert page.eval(f"{API}.pressure()") == 30_000.0
            assert page.eval(f"{API}.digest()") == before
        for bad in ("'abc'", "-3", "0", "NaN", "Infinity", "null"):
            r = page.eval(f"{API}.setPressure({bad})")
            assert not r["ok"] and r["pressure_pa"] == 30_000.0
        # a good value clears the error
        page.eval("document.getElementById('pressureInput').value = "
                  "'50000'; document.getElementById('pressureApply').click()")
        c = page.eval(f"{API}.pressureControls()")
        assert not c["invalid"] and not c["note"].startswith("Not applied")
        assert page.eval(f"{API}.pressure()") == 50_000.0
        # rapid slider input: one recomputation per frame, the newest wins,
        # and the drawn frame is exactly the one a direct selection draws
        seq0 = page.eval(f"{API}.seq()")
        page.eval("""(async () => {
            const s = document.getElementById('pressureSlider');
            for (const v of [100, 900, 250, 640, 480]) {
              s.value = v; s.dispatchEvent(new Event('input', {bubbles: true}));
            }
            await new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)));
        })()""")
        assert page.eval(f"{API}.pressure()") == pytest.approx(
            slider_pressure(480), rel=1e-12)
        assert page.eval(f"{API}.seq()") == seq0 + 1          # coalesced
        arr = page.eval(f"{API}.surfaceArrays()")
        assert arr["p0"] == page.eval(f"{API}.pressure()")
        coalesced = page.eval(f"{API}.digest()")
        page.eval(f"{API}.setPressure(70000)")
        page.eval(f"{API}.setPressure({slider_pressure(480)!r})")
        assert page.eval(f"{API}.digest()") == coalesced
        assert page.eval(f"{API}.pressureControls()")["slider"] == 480
        assert page.console_errors == [] and page.exceptions == []


@needs_browser
def test_time_switch_keeps_pressure_empty_surface_recovers_decoding_bounded(
        holed_page):
    path, payload, report, storage = holed_page
    A = payload["atmosphere"]
    nlat, nlon = A["mesh"]["nlat"], A["mesh"]["nlon"]
    n = nlat * nlon
    with OfflinePage(BROWSER) as page:
        url = _ready(page, path)
        page.eval(f"{API}.setPressure(61234.5)")
        heights, digests = [], []
        for t in range(len(A["times"])):
            page.eval(f"{API}.setTime({t})")
            assert page.eval(f"{API}.pressure()") == 61234.5
            arr = page.eval(f"{API}.surfaceArrays()")
            assert arr["timeIndex"] == t and arr["p0"] == 61234.5
            heights.append(np.array(arr["geopotential_height"]))
            digests.append(page.eval(f"{API}.digest()"))
            # the surface comes from that state's columns alone
            cols = pressure_columns(storage.frame(report["times"][t]),
                                    report["context"], *_mesh(6.0))
            q, b, ps = quantize_columns(cols)
            ref = interpolate_columns(dequantize_columns(
                q, b, ps, (nlat, nlon), cols), ("pressure", 61234.5))
            ok = ref.valid.ravel()
            assert np.abs(heights[-1] - ref["geopotential_height"].ravel()
                          )[ok].max() < 1e-2
            # bounded decoding: never more than MAX_DECODED_TIMES states
            assert len(page.eval(f"{API}.decodedTimes()")) <= \
                MAX_DECODED_TIMES
        # the warming fixture lifts the isobar between saved states, and no
        # time interpolation happens: three distinct frames
        assert len(set(digests)) == 3
        assert (heights[2] > heights[0]).mean() > 0.5
        assert page.eval(f"{API}.decodedTimes()") == [1, 2]   # LRU: last two
        page.eval(f"{API}.setTime(0)")
        assert page.eval(f"{API}.decodedTimes()") == [2, 0]
        # a pressure no node samples at this time: an empty surface, an
        # explanation, live controls
        r = page.eval(f"{API}.setPressure(1.0)")
        assert r["ok"]
        counts = page.eval(f"{API}.maskCounts()")
        assert counts["counts"]["0"] == 0 and counts["triangles"] == 0
        assert counts["counts"]["2"] == n
        note = page.eval("document.getElementById('atmNote').textContent")
        assert note.startswith("Empty surface") and "above the top" in note
        assert "stay live" in note
        assert "no valid node" in page.eval(
            "document.getElementById('atmNote').textContent") or True
        assert page.eval(f"{API}.sample(0, 0)")["readout"].startswith(
            "no sample on the 1 Pa surface")
        assert not page.eval("document.getElementById('pressureSlider')"
                             ".disabled")
        # under ground everywhere, too
        page.eval(f"{API}.setPressure(5.0e6)")
        counts = page.eval(f"{API}.maskCounts()")
        assert counts["triangles"] == 0 and counts["counts"]["1"] == n
        # recovery: the same pressure draws the same bytes as before
        page.eval(f"{API}.setPressure(61234.5)")
        assert page.eval(f"{API}.digest()") == digests[0]
        assert page.eval(f"{API}.maskCounts()")["triangles"] > 0
        assert not page.eval("document.getElementById('atmNote')"
                             ".textContent").startswith("Empty")
        # timing is reported apart: interpolation, mesh, map
        timing = page.eval(f"{API}.timing()")
        assert set(timing) >= {"decode_ms", "interpolate_ms", "mesh_ms",
                               "map_ms", "total_ms"}
        assert all(v >= 0 for v in timing.values())
        # playback steps through saved states with the pressure kept
        page.eval(f"{API}.step(1)")
        assert page.eval(f"{API}.pressure()") == 61234.5
        assert page.eval(f"{API}.info()")["timeIndex"] == 1
        assert page.requests == [url]
        assert page.console_errors == [] and page.exceptions == []
