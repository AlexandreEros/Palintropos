"""Constant-sigma surfaces: ``surface=("sigma", sigma0)``.

The third derived coordinate, interpolated in ``ln sigma`` between the
NOMINAL level positions, so a level's own ``sigma_k`` reproduces that native
level. Because the interpolation weight is the same at every node, the
interpolated relative vorticity and divergence are exactly those of the
interpolated wind: sigma surfaces are the one derived coordinate on which
they are offered. Checked: the specification; native-level coincidence (all
fields, zeta and delta included); the zeta/delta claim against spectral
synthesis of the same combination of level coefficients; all-or-nothing
masks and the end band; ``p = sigma0 p_s``; stored precision of the new
zeta/delta columns; the fixed and column payloads; and, in headless Chrome,
the page against Python on the decoded columns, the three-way coordinate
switch keeping each value, the sigma controls, and streamlines.
"""
from __future__ import annotations

import math
import pathlib
import sys

import numpy as np
import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tropoi.representation.diagnostics.pe_levels import (  # noqa: E402
    reconstruct_levels)
from tropoi.representation.diagnostics.pe_surfaces import (  # noqa: E402
    END_TOLERANCE, MASK_ABOVE_TOP_LEVEL, MASK_BELOW_BOTTOM_LEVEL, MASK_VALID,
    SIGMA_MASK_LABELS, SigmaSurface, SurfaceUnavailableError,
    interpolate_columns, parse_surface, pressure_columns, pressure_surface)
from tropoi.representation.visual.atmosphere import (  # noqa: E402
    COLUMN_FIELDS, SIGMA_SURFACE_FIELDS, _default_path, _mesh,
    atmosphere_payload, dequantize_columns, quantize_columns)
from tropoi.spatial.transforms.host_synthesis import (  # noqa: E402
    synthesize_latlon, synthesize_wind_latlon)
from test_interactive_streamlines import (RES,  # noqa: E402
                                          assert_same_lines, reference,
                                          write_windy_holed_capsule)
from test_pressure_surface_view import _storage  # noqa: E402
from tools.browser_cdp import OfflinePage, find_browser  # noqa: E402

API = "window.__terrainGlobe.atmosphere"
BROWSER = find_browser()
needs_browser = pytest.mark.skipif(BROWSER is None,
                                   reason="needs Chrome, Chromium or Edge")


@pytest.fixture(scope="module")
def capsule(tmp_path_factory):
    root, _, _ = write_windy_holed_capsule(
        tmp_path_factory.mktemp("sigma") / "cap", frames=3)
    sim, storage = _storage(root)
    return root, sim, storage


# ---------------------------------------------------------------------------
# The reference
# ---------------------------------------------------------------------------

def test_specification():
    s = parse_surface(("sigma", 0.5))
    assert isinstance(s, SigmaSurface) and s.kind == "sigma"
    assert s.value == 0.5 and s.units == "" and s.label == "σ = 0.5"
    assert parse_surface(("sigma", 1)).sigma == 1.0      # the ground
    for bad in (0.0, -0.2, 1.0001, math.nan, math.inf):
        with pytest.raises(ValueError, match=r"in \(0, 1\]"):
            parse_surface(("sigma", bad))
    for bad in ("0.5", True, None):
        with pytest.raises(TypeError, match="sigma must be a number"):
            parse_surface(("sigma", bad))
    assert set(SIGMA_MASK_LABELS) == {0, 2, 3}


def test_native_levels_reproduced_exactly_at_their_nominal_sigma(capsule):
    _, _, storage = capsule
    ctx = storage.pe_context()
    lat, lon = _mesh(RES)
    frame = storage.frame(1)
    for k, sk in enumerate(ctx.full_levels):
        s = pressure_surface(frame, ctx, lat, lon, ("sigma", sk))
        assert s.valid.all() and s.level_sigma == "nominal"
        lv = reconstruct_levels(frame, ctx, lat, lon, [k])
        for name in ("temperature", "geopotential_height", "u", "v", "zeta",
                     "delta", "pressure"):
            assert np.allclose(s[name], lv[name][0], rtol=1e-12,
                               atol=1e-12 * np.abs(lv[name][0]).max()), name
    # the hydrostatic level_sigma used by the other coordinates is ignored
    a = pressure_surface(frame, ctx, lat, lon, ("sigma", 0.4),
                         level_sigma="hydrostatic")
    b = pressure_surface(frame, ctx, lat, lon, ("sigma", 0.4),
                         level_sigma="nominal")
    assert (a["zeta"] == b["zeta"]).all()
    assert a.level_positions == tuple(ctx.full_levels)


def test_vorticity_divergence_are_those_of_the_interpolated_wind(capsule):
    """Between two levels the weight w is one number for the whole surface,
    so (1 - w) zeta_k + w zeta_k+1 is the synthesis of the same combination
    of spectral coefficients, and the interpolated (u, v) is the wind of
    that combined (zeta, delta): the four fields are mutually consistent to
    round-off, which a pressure or height surface cannot offer."""
    _, _, storage = capsule
    ctx = storage.pe_context()
    lat, lon = _mesh(RES)
    frame = storage.frame(2)
    x = np.asarray(frame)
    K = ctx.nlev
    full = np.array(ctx.full_levels)
    for s0 in (0.3, 0.61, 0.8):
        s = pressure_surface(frame, ctx, lat, lon, ("sigma", s0))
        k = int(np.unique(s.bracket)[0])
        assert (s.bracket == k).all()              # one bracket everywhere
        w = (math.log(s0) - math.log(full[k])) / (math.log(full[k + 1])
                                                  - math.log(full[k]))
        zc = (1 - w) * x[k] + w * x[k + 1]
        dc = (1 - w) * x[K + k] + w * x[K + k + 1]
        zeta = synthesize_latlon(zc, lat, lon)
        delta = synthesize_latlon(dc, lat, lon)
        u, v = synthesize_wind_latlon(zc, dc, ctx.radius_m, lat, lon)
        scale = np.abs(zeta).max()
        assert np.abs(s["zeta"] - zeta).max() <= 1e-12 * scale
        assert np.abs(s["delta"] - delta).max() <= 1e-12 * np.abs(delta).max()
        # the wind synthesis divides psi = zeta R^2 / l(l+1) (~4e8 m^2/s on
        # an Earth radius) back down to ~75 m/s, so combining coefficients
        # first or winds first differs by its own round-off, about
        # 1e-16 |psi| / R = 4e-8 m/s (measured 2e-10 to 6e-10 relative)
        assert np.abs(s["u"] - u).max() <= 1e-8 * np.abs(u).max()
        assert np.abs(s["v"] - v).max() <= 1e-8 * np.abs(v).max()
        assert (s["pressure"] == s0 * s.surface_pressure).all()


def test_masks_are_all_or_nothing_and_the_end_band(capsule):
    _, _, storage = capsule
    ctx = storage.pe_context()
    lat, lon = _mesh(RES)
    cols = pressure_columns(storage.frame(0), ctx, lat, lon)
    full = ctx.full_levels
    on_top = interpolate_columns(cols, ("sigma", full[0] * (1 + 0.5
                                                            * END_TOLERANCE)))
    assert on_top.valid.all()
    assert np.allclose(on_top["temperature"], cols.temperature[0])
    on_bottom = interpolate_columns(cols, ("sigma", full[-1]))
    assert on_bottom.valid.all()
    assert np.allclose(on_bottom["zeta"], cols.zeta[-1], rtol=1e-12)
    for s0, code in ((full[0] * (1 - 2 * END_TOLERANCE), MASK_ABOVE_TOP_LEVEL),
                     (0.5 * (full[-1] + 1.0), MASK_BELOW_BOTTOM_LEVEL),
                     (1.0, MASK_BELOW_BOTTOM_LEVEL)):
        with pytest.raises(SurfaceUnavailableError, match="nominal"):
            interpolate_columns(cols, ("sigma", s0))
    # the stored form carries zeta and delta within half a quantum
    q, bounds, p_s = quantize_columns(cols)
    deq = dequantize_columns(q, bounds, p_s, (lat.size, lon.size), cols)
    assert {"zeta", "delta"} <= set(COLUMN_FIELDS)
    for name in ("zeta", "delta"):
        exact, back = getattr(cols, name), getattr(deq, name)
        for k in range(ctx.nlev):
            lo, hi = bounds[name][k]
            tol = 0.5 * (hi - lo) / 65535 + 1e-6 * np.abs(exact[k]).max()
            assert np.abs(back[k] - exact[k])[:, :-1].max() <= tol
    assert deq.full_levels == tuple(full)
    # columns without zeta/delta cannot be stored for the page
    bare = pressure_columns(storage.frame(0), ctx, lat, lon,
                            vorticity_divergence=False)
    with pytest.raises(ValueError, match="holds no 'zeta'|hold no 'zeta'"):
        quantize_columns(bare)


# ---------------------------------------------------------------------------
# Payloads
# ---------------------------------------------------------------------------

def test_fixed_sigma_page_and_column_page_payloads(capsule):
    root, sim, storage = capsule
    ctx = storage.pe_context()
    page, report = atmosphere_payload(storage, surface=("sigma", 0.6),
                                      times="all", resolution_deg=RES)
    A = page["atmosphere"]
    assert A["version"] == 2
    assert [f["id"] for f in A["fields"]] == list(SIGMA_SURFACE_FIELDS)
    fields = {f["id"]: f for f in A["fields"]}
    for name in ("zeta", "delta"):
        assert fields[name]["scale"] == "diverging"
        lo, hi = fields[name]["range"]
        assert lo == -hi and hi > 0
    assert A["surface"]["kind"] == "sigma" and A["surface"]["label"] == \
        "σ = 0.6"
    assert A["surface"]["level_sigma"] == "nominal"
    assert A["surface"]["level_positions"] == pytest.approx(ctx.full_levels)
    assert A["surface"]["mask_labels"] == {str(k): v for k, v in
                                           SIGMA_MASK_LABELS.items()}
    assert all(c[MASK_VALID] == A["mesh"]["nlat"] * A["mesh"]["nlon"]
               for c in report["mask_counts"])
    assert A["streamlines"]["enabled"] is True
    assert any("exactly those of the interpolated wind" in line
               for line in A["interpretation"])
    assert "-sigma-" in _default_path(storage, report).name
    with pytest.raises(SurfaceUnavailableError):
        atmosphere_payload(storage, surface=("sigma", 0.99),
                           resolution_deg=RES)
    # the column page: sigma as a third coordinate, nominal positions
    cp, crep = atmosphere_payload(storage, surface=("pressure", 6e4),
                                  interactive_surface=True, times="all",
                                  resolution_deg=RES)
    S = cp["atmosphere"]["surface"]
    assert S["sigma_positions"] == pytest.approx(ctx.full_levels)
    sc = S["coordinates"]["sigma"]
    assert sc["range"] == pytest.approx([ctx.full_levels[0],
                                         ctx.full_levels[-1]])
    assert sc["initial"] == pytest.approx(0.5 * sum(sc["range"]))
    assert cp["atmosphere"]["columns"]["fields"] == list(COLUMN_FIELDS)
    assert cp["atmosphere"]["defaults"]["sigma"] == sc["initial"]
    assert crep["sigma_range"] == tuple(sc["range"])
    sp, srep = atmosphere_payload(storage, surface=("sigma", 0.6),
                                  interactive_surface=True, times="all",
                                  resolution_deg=RES)
    assert sp["atmosphere"]["defaults"]["coordinate"] == "sigma"
    assert sp["atmosphere"]["surface"]["coordinates"]["sigma"]["initial"] \
        == 0.6
    assert "-sigmas-" in _default_path(storage, srep).name
    with pytest.raises(SurfaceUnavailableError, match="any sigma"):
        atmosphere_payload(storage, surface=("sigma", 0.97),
                           interactive_surface=True, resolution_deg=RES)
    path = sim.view(pathlib.Path(root).parent / "zfix.html",
                    surface=("sigma", 0.6), times="all", resolution_deg=RES,
                    open_browser=False)
    assert path.is_file()


# ---------------------------------------------------------------------------
# The page
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sigma_page(capsule):
    root, sim, storage = capsule
    payload, report = atmosphere_payload(
        storage, surface=("pressure", 61_234.5), interactive_surface=True,
        times="all", resolution_deg=RES)
    path = sim.view(pathlib.Path(root).parent / "cols.html",
                    surface=("pressure", 61_234.5), interactive_surface=True,
                    times="all", resolution_deg=RES, open_browser=False)
    return path, payload, report, storage


def _ready(page, path):
    url = page.open(path)
    page.wait_for(f"{API} && document.getElementById('timeLabel')"
                  ".textContent")
    return url


@needs_browser
def test_page_sigma_surfaces_match_python_and_native_levels(sigma_page):
    path, payload, report, storage = sigma_page
    A = payload["atmosphere"]
    nlat, nlon = A["mesh"]["nlat"], A["mesh"]["nlon"]
    full = report["context"].full_levels
    lat, lon = _mesh(RES)
    with OfflinePage(BROWSER) as page:
        url = _ready(page, path)
        for t in range(3):
            page.eval(f"{API}.setTime({t})")
            cols = pressure_columns(storage.frame(report["times"][t]),
                                    report["context"], lat, lon)
            q, b, ps = quantize_columns(cols)
            deq = dequantize_columns(q, b, ps, (nlat, nlon), cols)
            for k, s0 in [(k, s) for k, s in enumerate(full)] + [
                    (None, 0.3), (None, 0.61234), (None, 0.03),
                    (None, 0.95)]:
                r = page.eval(f"{API}.setSigma({s0!r})")
                assert r["ok"] and r["sigma"] == s0
                arr = page.eval(f"{API}.surfaceArrays()")
                assert arr["kind"] == "sigma" and arr["value"] == s0
                m = np.array(arr["mask"])
                try:
                    ref = interpolate_columns(deq, ("sigma", s0))
                except SurfaceUnavailableError:
                    assert (m != 0).all()
                    note = page.eval("document.getElementById('atmNote')"
                                     ".textContent")
                    assert note.startswith("Empty surface") and \
                        "stay live" in note
                    assert page.eval(f"{API}.maskCounts()")["triangles"] == 0
                    continue
                assert (m == ref.mask.ravel()).all()
                for name, want in (("temperature", ref["temperature"]),
                                   ("geopotential_height",
                                    ref["geopotential_height"]),
                                   ("pressure", ref["pressure"]),
                                   ("zeta", ref["zeta"]),
                                   ("delta", ref["delta"]),
                                   ("u", ref["u"]), ("v", ref["v"])):
                    js = np.array(arr[name]).reshape(nlat, nlon)
                    tol = 2e-6 * np.abs(want).max() + 1e-12
                    assert np.abs(js - want).max() <= tol, (t, s0, name)
                if k is not None:
                    # a nominal level: that stored level, exactly as stored
                    zeta = np.array(arr["zeta"]).reshape(nlat, nlon)
                    assert np.allclose(zeta, deq.zeta[k], rtol=0,
                                       atol=2e-6 * np.abs(deq.zeta[k]).max())
                # streamlines of the sigma surface equal Python's
                sl = page.eval(f"{API}.streamlines()")
                valid = m.reshape(nlat, nlon) == 0
                lines = reference(np.array(arr["u"]).reshape(nlat, nlon),
                                  np.array(arr["v"]).reshape(nlat, nlon),
                                  valid)
                assert sl["current"] and sl["key"] == arr["key"]
                assert_same_lines(sl["lines"], lines)
        # vorticity on the globe: offered, labelled, diverging scale
        page.eval(f"{API}.setSigma(0.4)")
        assert page.eval(f"{API}.setField('zeta')") is True
        info = page.eval(f"{API}.info()")
        assert info["field"] == "zeta" and info["coordinate"] == "sigma"
        assert info["range"][0] == -info["range"][1]
        note = page.eval("document.getElementById('atmNote').textContent")
        assert "exactly those of the interpolated wind" in note
        assert page.requests == [url]
        assert page.console_errors == [] and page.exceptions == []


@needs_browser
def test_three_coordinates_keep_their_values_and_fields(sigma_page):
    path, payload, report, storage = sigma_page
    sc = payload["atmosphere"]["surface"]["coordinates"]["sigma"]
    visible = ("[...document.getElementById('fieldButtons').children]"
               ".filter(b => !b.hidden).map(b => b.textContent)")
    with OfflinePage(BROWSER) as page:
        _ready(page, path)
        assert page.eval(visible) == ["Temperature", "Geopotential height",
                                      "Wind speed"]
        z0 = page.eval(f"{API}.height()")
        page.eval("document.getElementById('coordSigma').click()")
        assert page.eval(f"{API}.coordinate()") == "sigma"
        assert page.eval(f"{API}.sigma()") == pytest.approx(sc["initial"])
        assert page.eval("document.getElementById('sigmaBlock').hidden") \
            is False
        assert page.eval("document.getElementById('pressureBlock').hidden")
        assert page.eval(visible) == [
            "Temperature", "Geopotential height", "Wind speed", "Pressure",
            "Relative vorticity", "Divergence"]
        # typed sigma, a native level, refusals
        for text, want in (("0.45", 0.45), ("0,7", 0.7), ("1", 1.0)):
            page.eval("{ const i = document.getElementById('sigmaInput'); "
                      f"i.value = {text!r}; i.dispatchEvent(new "
                      "KeyboardEvent('keydown', {key: 'Enter', "
                      "bubbles: true})) }")
            assert page.eval(f"{API}.sigma()") == want
        page.eval(f"{API}.setSigma(0.45)")
        c = page.eval(f"{API}.sigmaControls()")
        assert c["label"] == "σ = 0.45" and "linear" in c["note"]
        assert "native levels sit at" in c["note"]
        for bad in ("0", "-0.1", "1.5", "abc", "", "NaN"):
            page.eval("document.getElementById('sigmaInput').value = "
                      f"{bad!r}; document.getElementById('sigmaApply')"
                      ".click()")
            c = page.eval(f"{API}.sigmaControls()")
            assert c["invalid"] and c["note"].startswith("Not applied:"), bad
            assert page.eval(f"{API}.sigma()") == 0.45
        # the slider is linear over the native range
        page.eval("const s = document.getElementById('sigmaSlider');"
                  "s.value = 500; s.dispatchEvent(new Event('input'));"
                  "s.dispatchEvent(new Event('change'))")
        lo, hi = sc["range"]
        assert page.eval(f"{API}.sigma()") == pytest.approx(
            0.5 * (lo + hi), rel=1e-12)
        s_kept = page.eval(f"{API}.sigma()")
        page.eval(f"{API}.setField('delta')")
        # the three coordinates keep their values and field choices
        page.eval(f"{API}.setCoordinate('pressure')")
        assert page.eval(f"{API}.pressure()") == 61_234.5
        assert page.eval(f"{API}.setField('zeta')") is False   # not here
        page.eval(f"{API}.setCoordinate('geopotential_height')")
        assert page.eval(f"{API}.height()") == z0
        page.eval(f"{API}.setCoordinate('sigma')")
        assert page.eval(f"{API}.sigma()") == s_kept
        assert page.eval(f"{API}.info()")["field"] == "delta"
        assert page.console_errors == [] and page.exceptions == []
