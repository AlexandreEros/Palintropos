"""Streamlines traced in the browser on interactive derived surfaces.

The interactive page of ``Simulation.view(surface=..., interactive_pressure=
True)`` traces instantaneous horizontal streamlines of the wind it
interpolates, with a transcription of
:func:`tropoi.representation.visual.sphere_streamlines.integrate_streamlines`.
Checked here: the payload settings; the page's lines against the Python
reference on IDENTICAL decoded winds and seeds (the page's own interpolated
``u, v`` and mask); analytic winds (solid-body zonal flow, a flow over the
poles, the longitude seam, calm air, a narrow masked band that a long step
would jump, an all-masked surface); and the scheduling: obsolete lines
cleared at once, deferred tracing while a slider is dragged with only the
newest request publishing, release, typed values, saved-time switches and
playback giving the lines of the surface shown, a bounded line cache.
Browser tests are skipped without Chrome, Chromium or Edge.
"""
from __future__ import annotations

import math
import pathlib
import sys
import time

import numpy as np
import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tropoi.representation.visual.atmosphere import (  # noqa: E402
    STREAMLINE_CACHE, STREAMLINE_MAX_STEPS, STREAMLINE_MIN_SPEED,
    STREAMLINE_RELATIVE_MIN_SPEED, STREAMLINE_SEEDS, STREAMLINE_SETTLE_MS,
    STREAMLINE_STEP_DEG, _mesh, atmosphere_payload, decoded_column_bytes,
    streamline_cache_bytes)
from tropoi.representation.visual.sphere_streamlines import (  # noqa: E402
    _MeshSampler, integrate_streamlines)
from test_atmosphere_view import L_MAX, NLEV, _real_field  # noqa: E402
from test_pressure_surface_view import (_storage,  # noqa: E402
                                        write_holed_capsule)
from tools.browser_cdp import OfflinePage, find_browser  # noqa: E402

API = "window.__terrainGlobe.atmosphere"
BROWSER = find_browser()
needs_browser = pytest.mark.skipif(BROWSER is None,
                                   reason="needs Chrome, Chromium or Edge")
RES = 6.0
#: unit-sphere distance within which a browser point equals the reference
#: point (the two differ only by libm round-off: measured below 1e-13)
POINT_TOL = 1e-9


def write_windy_holed_capsule(root, *, frames=3, seed=5):
    """The holed isothermal capsule (tall terrain, warming between states)
    with random sheared vortices and divergence on every level, so isobars
    and height surfaces carry wind AND masked holes."""
    root, coeffs, terrain = write_holed_capsule(root, frames=frames)
    rng = np.random.default_rng(seed)
    for i in range(frames):
        for k in range(NLEV):
            coeffs[i][k] = _real_field(rng, L_MAX, 3e-5, lmax_used=L_MAX - 1)
            coeffs[i][NLEV + k] = _real_field(rng, L_MAX, 3e-6,
                                              lmax_used=L_MAX - 1)
    np.save(root / "pe_coeffs.npy", coeffs)
    return root, coeffs, terrain


def _xyz(lat, lon):
    la, lo = np.radians(np.asarray(lat)), np.radians(np.asarray(lon))
    return np.stack([np.cos(la) * np.sin(lo), np.sin(la),
                     np.cos(la) * np.cos(lo)], axis=-1)


def reference(u, v, valid, **over):
    lat, lon = _mesh(RES)
    opt = dict(seed_count=STREAMLINE_SEEDS, step_deg=STREAMLINE_STEP_DEG,
               max_steps=STREAMLINE_MAX_STEPS, min_speed=STREAMLINE_MIN_SPEED,
               relative_min_speed=STREAMLINE_RELATIVE_MIN_SPEED)
    opt.update(over)
    return integrate_streamlines(u, v, lat, lon, valid=valid, **opt)


def assert_same_lines(js, ref):
    """Same lines, same order, same number of points, the same points."""
    assert len(js) == len(ref)
    worst = 0.0
    for a, (la, lo) in zip(js, ref):
        assert len(a["lat"]) == la.size
        d = np.abs(_xyz(a["lat"], a["lon"]) - _xyz(la, lo)).max()
        worst = max(worst, float(d))
    assert worst <= POINT_TOL, worst
    return worst


def assert_inside(lines, valid, step_deg=STREAMLINE_STEP_DEG):
    """The reference's clipping guarantee, checked independently: every
    point of a line, and every point the reference checks along each step
    (spaced closer than half a cell), lies in a cell whose four nodes are
    valid, so no masked band is jumped; a dense resampling (a tenth of a
    cell) may cut the corner of a masked cell only within a quarter cell of
    a checked point."""
    lat, lon = _mesh(RES)
    inside = _MeshSampler(valid.astype(float)[..., None], lat, lon)
    count = int(math.ceil(2.0 * step_deg / RES)) + 1
    checked = np.linspace(0.0, 1.0, count + 1)
    dense = np.linspace(0.0, 1.0, int(math.ceil(10 * step_deg / RES)) + 2)

    def ok(a, b, f):
        # (segments, fractions) validity of the points a + f (b - a)
        p = a[:, None, :] + f[None, :, None] * (b - a)[:, None, :]
        p = p / np.linalg.norm(p, axis=-1, keepdims=True)
        la = np.degrees(np.arcsin(np.clip(p[..., 1], -1.0, 1.0)))
        lo = np.degrees(np.arctan2(p[..., 0], p[..., 2]))
        return inside(la, lo)[..., 0] > 1 - 1e-9

    for line in lines:
        x = _xyz(line["lat"], line["lon"])
        a, b = x[:-1], x[1:]
        assert ok(a, b, checked).all()
        bad = ~ok(a, b, dense)
        if bad.any():
            seg = np.degrees(np.linalg.norm(b - a, axis=1))
            gap = np.abs(dense[:, None] - checked[None, :]).min(axis=1)
            assert (gap[None, :] * seg[:, None])[bad].max() <= 0.25 * RES


# ---------------------------------------------------------------------------
# The reference
# ---------------------------------------------------------------------------

def test_reference_never_draws_a_point_in_a_masked_cell(tmp_path):
    """Regression: the step's end is part of the whole-step check, so the
    last point of a line (no step after it to reject it) is never drawn in
    a masked cell. On this fixture 61234.5 Pa put line ends in masked cells
    before the end was checked."""
    from tropoi.representation.diagnostics.pe_surfaces import pressure_surface
    root, _, _ = write_windy_holed_capsule(tmp_path / "cap", frames=1)
    _, storage = _storage(root)
    lat, lon = _mesh(RES)
    lines_total = 0
    for p0 in (61_234.5, 93_000.0, 88_000.0, 30_000.0):
        s = pressure_surface(storage.frame(0), storage.pe_context(), lat, lon,
                             ("pressure", p0))
        lines = reference(s["u"], s["v"], s.valid)
        lines_total += len(lines)
        assert_inside([{"lat": la, "lon": lo} for la, lo in lines], s.valid)
    assert lines_total > 300


# ---------------------------------------------------------------------------
# Payload
# ---------------------------------------------------------------------------

def test_payload_carries_the_integrator_settings_not_lines(tmp_path):
    root, _, _ = write_windy_holed_capsule(tmp_path / "cap", frames=2)
    _, storage = _storage(root)
    page, report = atmosphere_payload(
        storage, surface=("pressure", 7.0e4), interactive_pressure=True,
        times="all", resolution_deg=RES)
    S = page["atmosphere"]["streamlines"]
    assert S["enabled"] is True and S["mode"] == "browser"
    assert "points" not in S and "line_start" not in S     # nothing exported
    assert (S["seed_count"], S["step_deg"], S["max_steps"]) == (
        STREAMLINE_SEEDS, STREAMLINE_STEP_DEG, STREAMLINE_MAX_STEPS)
    assert S["min_speed"] == STREAMLINE_MIN_SPEED
    assert S["relative_min_speed"] == STREAMLINE_RELATIVE_MIN_SPEED
    assert S["settle_ms"] == STREAMLINE_SETTLE_MS
    assert S["cache_size"] == STREAMLINE_CACHE
    assert page["atmosphere"]["defaults"]["streamlines"] is True
    assert any(line.startswith("Streamlines: instantaneous horizontal")
               and "traced in the page" in line
               for line in page["atmosphere"]["interpretation"])
    # the cache bound and the per-node working memory are reported
    assert streamline_cache_bytes() == STREAMLINE_CACHE * STREAMLINE_SEEDS \
        * (2 * STREAMLINE_MAX_STEPS + 1) * 16
    assert any("cached streamlines" in row[1] for row in page["details"])
    lat, lon = _mesh(RES)
    assert report["decoded_bytes_per_time"] == decoded_column_bytes(
        report["context"].nlev, lat.size, lon.size)
    # opting out keeps the controls but disables the lines
    off, _ = atmosphere_payload(
        storage, surface=("pressure", 7.0e4), interactive_pressure=True,
        times="all", resolution_deg=RES, streamlines=False)
    assert off["atmosphere"]["streamlines"]["enabled"] is False
    assert off["atmosphere"]["defaults"]["streamlines"] is False


# ---------------------------------------------------------------------------
# The page, offline
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def windy_page(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("windy")
    root, _, _ = write_windy_holed_capsule(tmp / "cap", frames=3)
    sim, storage = _storage(root)
    path = sim.view(tmp / "windy.html", surface=("pressure", 93_000.0),
                    interactive_pressure=True, times="all",
                    resolution_deg=RES, open_browser=False)
    return path, storage


def _ready(page, path):
    url = page.open(path)
    page.wait_for(f"{API} && document.getElementById('timeLabel')"
                  ".textContent")
    assert page.eval("window.__terrainGlobe.glOK")
    return url


def _page_reference(page):
    """The Python reference on the page's own decoded winds and mask."""
    lat, lon = _mesh(RES)
    shape = (lat.size, lon.size)
    arr = page.eval(f"{API}.surfaceArrays()")
    u = np.array(arr["u"]).reshape(shape)
    v = np.array(arr["v"]).reshape(shape)
    valid = np.array(arr["mask"]).reshape(shape) == 0
    return reference(u, v, valid), valid, arr["key"]


@needs_browser
def test_page_lines_match_python_on_the_decoded_winds(windy_page):
    path, _ = windy_page
    with OfflinePage(BROWSER) as page:
        url = _ready(page, path)
        info = page.eval(f"{API}.info()")
        assert info["streams"] is True
        assert not page.eval("document.getElementById('streams').disabled")
        # the opened page already shows the initial surface's lines
        ref, _, key = _page_reference(page)
        first = page.eval(f"{API}.streamlines()")
        assert first["current"] and first["key"] == key and ref
        assert_same_lines(first["lines"], ref)
        assert first["drawnVerts"] > 0
        drawn, masked_cases, worst = 0, 0, 0.0
        for t in range(3):
            page.eval(f"{API}.setTime({t})")
            for p0 in (93_000.0, 88_000.0, 61_234.5, 30_000.0, 99_500.0,
                       1.0):
                page.eval(f"{API}.setPressure({p0!r})")
                ref, valid, key = _page_reference(page)
                sl = page.eval(f"{API}.streamlines()")
                assert sl["current"] and sl["key"] == key and not sl["pending"]
                assert sl["surfaceKey"] == key
                worst = max(worst, assert_same_lines(sl["lines"], ref))
                assert_inside(sl["lines"], valid)
                # the drawn buffer holds these lines (2 vertices per segment
                # plus a 4-vertex chevron per line), or none when empty
                segs = sum(len(l["lat"]) - 1 for l in sl["lines"])
                assert sl["drawnVerts"] == 2 * segs + 4 * len(sl["lines"])
                drawn += len(ref)
                masked_cases += bool(ref) and not valid.all()
        # the fixture exercises masked, windy surfaces and empty ones
        assert drawn > 1000 and masked_cases >= 6
        note = page.eval("document.getElementById('streamNote').textContent")
        assert "No streamlines on this surface" in note       # p0 = 1 Pa
        page.eval(f"{API}.setPressure(61234.5)")
        note = page.eval("document.getElementById('streamNote').textContent")
        assert "instantaneous horizontal streamlines" in note
        assert "not parcel trajectories" in note.lower()
        assert page.requests == [url]
        assert page.console_errors == [] and page.exceptions == []
    print(f"worst browser-Python point distance: {worst:.3g}")


def _analytic(kind):
    lat, lon = _mesh(RES)
    la = np.radians(lat)[:, None]
    lo = np.radians(lon)[None, :]
    shape = (lat.size, lon.size)
    if kind == "zonal":                      # solid-body, 20 m/s at equator
        u = 20.0 * np.cos(la) + 0.0 * lo
        v = np.zeros(shape)
    elif kind == "polar":                    # rotation about the x axis
        # V = a x r with a = (1, 0, 0) (the point lat 0, lon 90):
        # east and north components of (0, -z, y)
        x = np.cos(la) * np.sin(lo)
        y = np.sin(la) + 0.0 * lo
        z = np.cos(la) * np.cos(lo)
        vx, vy, vz = 0.0 * x, -z, y
        east = (np.cos(lo) * vx - np.sin(lo) * vz)
        north = (-np.sin(la) * np.sin(lo) * vx + np.cos(la) * vy
                 - np.sin(la) * np.cos(lo) * vz)
        u, v = 15.0 * east, 15.0 * north
    elif kind == "calm":
        u = np.zeros(shape)
        v = np.zeros(shape)
    return u, v


@needs_browser
def test_page_integrator_on_analytic_winds(windy_page):
    path, _ = windy_page
    lat, lon = _mesh(RES)
    shape = (lat.size, lon.size)
    allv = np.ones(shape, bool)

    def run(page, u, v, valid, **over):
        js = page.eval(f"{API}.traceStreamlines({u.ravel().tolist()}, "
                       f"{v.ravel().tolist()}, "
                       f"{valid.astype(int).ravel().tolist()}, "
                       f"{__import__('json').dumps(over)})")
        ref = reference(u, v, valid, **over)
        assert_same_lines(js, ref)
        return js

    with OfflinePage(BROWSER) as page:
        _ready(page, path)
        # solid-body zonal flow: every line is a latitude circle (to the
        # bilinear and RK4 errors of a 6-degree mesh and 1.5-degree steps)
        u, v = _analytic("zonal")
        lines = run(page, u, v, allv)
        assert len(lines) == STREAMLINE_SEEDS - 0 or len(lines) > 140
        for line in lines:
            la = np.array(line["lat"])
            assert np.abs(la - la[len(la) // 2]).max() < 0.05
            # eastward flow: listed upstream to downstream
            lo = np.unwrap(np.radians(line["lon"]))
            assert (np.diff(lo) > 0).all()
        # the seam: lines cross lon = +-180 with no jump (unit-sphere chord
        # of one step everywhere)
        h = math.radians(STREAMLINE_STEP_DEG)
        crossing = 0
        for line in lines:
            x = _xyz(line["lat"], line["lon"])
            assert np.linalg.norm(np.diff(x, axis=0), axis=1).max() < 1.01 * h
            crossing += bool((np.abs(np.diff(line["lon"])) > 180).any())
        assert crossing > 10
        # a flow over the poles: rotation about an equatorial axis; each line
        # stays on its circle x . a = const and some pass the poles smoothly
        u, v = _analytic("polar")
        lines = run(page, u, v, allv)
        near_pole = 0
        for line in lines:
            x = _xyz(line["lat"], line["lon"])
            assert np.ptp(x[:, 0]) < 2e-3
            assert np.linalg.norm(np.diff(x, axis=0), axis=1).max() < 1.01 * h
            near_pole += np.abs(line["lat"]).max() > 87.0
        assert near_pole >= 2
        # calm air: no line
        u, v = _analytic("calm")
        assert run(page, u, v, allv) == []
        # an all-masked surface: no line
        u, v = _analytic("zonal")
        assert run(page, u, v, np.zeros(shape, bool)) == []
        # a narrow masked band: one meridian of masked nodes (two cells, 12
        # degrees of longitude). With a 30-degree step the RK4 points alone
        # (15 degrees apart) can straddle it; the whole-step check (points
        # 30/11 degrees apart) stops every line before it, exactly as the
        # reference does. The check spacing is in mesh degrees, so this
        # holds where the band is wider than that spacing in arc, 12 cos(lat)
        # > 30/11 degrees, i.e. equatorward of 76 degrees; the crossing test
        # looks at segments equatorward of 60 degrees (near a pole the
        # meridians converge and a long step can cross a one-node band:
        # a documented limitation of the mesh-degree spacing).
        band = np.ones(shape, bool)
        i_band = int(np.argmin(np.abs(lon - 30.0)))
        band[:, i_band] = False
        unmasked = run(page, u, v, allv, step_deg=30.0, max_steps=6)
        lo_band = lon[i_band]

        def crosses(line):
            lo, la = np.array(line["lon"]), np.array(line["lat"])
            a = np.mod(lo[:-1] - lo_band + 180, 360) - 180
            b = np.mod(lo[1:] - lo_band + 180, 360) - 180
            low = np.maximum(np.abs(la[:-1]), np.abs(la[1:])) < 60.0
            return bool(((a < 0) & (b > 0) & (b - a < 180) & low).any())

        assert sum(crosses(l) for l in unmasked) > 20   # it would cross
        clipped = run(page, u, v, band, step_deg=30.0, max_steps=6)
        assert clipped and not any(crosses(l) for l in clipped)
        assert_inside(clipped, band, step_deg=30.0)
        assert page.console_errors == [] and page.exceptions == []


@needs_browser
def test_dragging_defers_tracing_and_only_the_newest_publishes(windy_page):
    path, _ = windy_page
    with OfflinePage(BROWSER) as page:
        _ready(page, path)
        page.eval(f"{API}.setPressure(61234.5)")
        assert page.eval(f"{API}.streamlines()")["current"]
        published = page.eval(f"{API}.streamlines()")["published"]
        # a burst of slider input (a drag), applied in the next frame: the
        # surface follows, the old lines are gone at once, nothing is traced
        # yet (read in that same frame, well inside the settle delay)
        state = page.eval("""(async () => {
            const s = document.getElementById('pressureSlider'), a = window.__terrainGlobe.atmosphere;
            const seq = a.seq();
            for (const v of [700, 650, 600, 555]) {
              s.value = v; s.dispatchEvent(new Event('input', {bubbles: true}));
            }
            await new Promise(r => requestAnimationFrame(r));
            return Object.assign(a.streamlines(), { applied: a.seq() - seq, slider: +s.value });
        })()""")
        assert state["applied"] == 1 and state["slider"] == 555
        assert state["pending"] and not state["current"]
        assert state["lines"] == [] and state["drawnVerts"] == 0
        assert state["published"] == published
        surface_key = state["surfaceKey"]
        assert surface_key.startswith("pressure:")
        time.sleep(3 * STREAMLINE_SETTLE_MS / 1000)
        after = page.eval(f"{API}.streamlines()")
        assert after["current"] and not after["pending"]
        assert after["key"] == surface_key
        assert after["published"] == published + 1     # one trace, the last
        ref, _, key = _page_reference(page)
        assert key == surface_key
        assert_same_lines(after["lines"], ref)
        digest = page.eval(f"{API}.digest()")
        # the same surface selected directly draws the same bytes
        p_now = page.eval(f"{API}.pressure()")
        page.eval(f"{API}.setPressure(70000)")
        page.eval(f"{API}.setPressure({p_now!r})")
        assert page.eval(f"{API}.digest()") == digest
        # a drag superseded by a saved-time switch: the deferred trace never
        # publishes; the new time's lines are drawn at once
        state = page.eval("""(async () => {
            const s = document.getElementById('pressureSlider');
            s.value = 420; s.dispatchEvent(new Event('input', {bubbles: true}));
            await new Promise(r => requestAnimationFrame(r));
            const a = window.__terrainGlobe.atmosphere;
            const before = a.streamlines();
            a.setTime(0);
            return [before, a.streamlines()];
        })()""")
        before, now = state
        assert before["pending"] and not before["current"]
        assert now["current"] and not now["pending"]
        n_pub = now["published"]
        time.sleep(3 * STREAMLINE_SETTLE_MS / 1000)
        late = page.eval(f"{API}.streamlines()")
        assert late["published"] == n_pub and late["key"] == now["key"]
        assert "pressure:" in late["key"] and late["key"].endswith(":0")
        ref, _, _ = _page_reference(page)
        assert_same_lines(late["lines"], ref)
        # release ('change') traces at once, without the settle delay
        state = page.eval("""(async () => {
            const s = document.getElementById('pressureSlider');
            s.value = 300; s.dispatchEvent(new Event('input', {bubbles: true}));
            s.dispatchEvent(new Event('change', {bubbles: true}));
            return window.__terrainGlobe.atmosphere.streamlines();
        })()""")
        assert state["current"] and not state["pending"]
        assert page.eval(f"{API}.pressureControls()")["slider"] == 300
        ref, _, key = _page_reference(page)
        assert state["key"] == key
        assert_same_lines(state["lines"], ref)
        assert page.console_errors == [] and page.exceptions == []


@needs_browser
def test_typed_time_playback_and_cache_give_the_shown_surfaces_lines(
        windy_page):
    path, _ = windy_page
    with OfflinePage(BROWSER) as page:
        _ready(page, path)

        def check():
            sl = page.eval(f"{API}.streamlines()")
            ref, _, key = _page_reference(page)
            assert sl["current"] and sl["key"] == key
            assert_same_lines(sl["lines"], ref)
            return sl

        # typed value (Enter)
        page.eval("const i = document.getElementById('pressureInput');"
                  "i.value = '612.345'; i.dispatchEvent(new KeyboardEvent("
                  "'keydown', {key: 'Enter', bubbles: true}))")
        check()
        # saved-time switches by slider, steps
        page.eval("const t = document.getElementById('timeSlider');"
                  "t.value = 1; t.dispatchEvent(new Event('input'))")
        assert check()["key"].endswith(":1")
        page.eval(f"{API}.step(1)")
        assert check()["key"].endswith(":2")
        # playback: every played frame has its own lines
        page.eval(f"{API}.setLoop(true); {API}.setSpeed(8); {API}.play()")
        seen = set()
        for _ in range(12):
            time.sleep(0.06)
            # one evaluation, so no frame is played between the two reads
            sl, info = page.eval(f"[{API}.streamlines(), {API}.info()]")
            assert sl["current"]
            assert sl["key"].endswith(f":{info['timeIndex']}")
            seen.add(sl["key"])
        page.eval(f"{API}.pause()")
        assert len(seen) == 3
        check()
        # streamlines off: nothing drawn; on again: the same lines
        digest = page.eval(f"{API}.digest()")
        page.eval(f"{API}.setStreamlines(false)")
        assert page.eval(f"{API}.info()")["streamVerts"] == 0
        page.eval(f"{API}.setStreamlines(true)")
        assert page.eval(f"{API}.digest()") == digest
        # the line cache stays bounded
        for p0 in np.geomspace(20_000.0, 95_000.0, 2 * STREAMLINE_CACHE):
            page.eval(f"{API}.setPressure({float(p0)!r})")
        assert page.eval(f"{API}.streamlines()")["cached"] <= STREAMLINE_CACHE
        timing = page.eval(f"{API}.timing()")
        assert {"streams_ms", "integrate_ms"} <= set(timing)
        assert page.console_errors == [] and page.exceptions == []
