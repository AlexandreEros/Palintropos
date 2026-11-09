"""Geopotential-height surfaces in the atmosphere page.

Fixed: ``Simulation.view(surface=("geopotential_height", Z0))`` exports one
height surface per saved time (temperature, pressure, wind speed; masks;
streamlines). Interactive: every column page (``interactive_surface=True``,
or the older ``interactive_pressure=True``) lets the browser switch between
pressure and geopotential height without regenerating the file, with a
linear height slider and a typed value in m or km. Checked: the payload
(the coordinates block, the float32 terrain, sizes); in headless Chrome, the
page's height arrays against the Python reference on the decoded columns at
signed heights, end levels and beyond (an empty surface); every mask reason;
valid-node geometry independent of the saved time at fixed height while
values and validity change; coordinate switches keeping each coordinate's
own value; controls; streamlines on height surfaces against Python;
deferred tracing on the height slider; a corrupted stored column masked
rather than divided by.
"""
from __future__ import annotations

import base64
import json
import math
import pathlib
import sys
import time

import numpy as np
import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tropoi.representation.diagnostics.pe_surfaces import (  # noqa: E402
    HEIGHT_MASK_LABELS, MASK_ABOVE_TOP_LEVEL, MASK_BELOW_BOTTOM_LEVEL,
    MASK_BELOW_SURFACE, MASK_LABELS, MASK_VALID, SurfaceUnavailableError,
    interpolate_columns, pressure_columns, pressure_surface)
from tropoi.representation.visual.atmosphere import (  # noqa: E402
    HEIGHT_SURFACE_FIELDS, STREAMLINE_SETTLE_MS, SURFACE_FIELDS, _mesh,
    atmosphere_payload, decoded_column_bytes, dequantize_columns,
    quantize_columns)
from test_interactive_streamlines import (RES,  # noqa: E402
                                          assert_inside, assert_same_lines,
                                          reference,
                                          write_windy_holed_capsule)
from test_pressure_surface_view import _storage  # noqa: E402
from tools.browser_cdp import OfflinePage, find_browser  # noqa: E402

API = "window.__terrainGlobe.atmosphere"
BROWSER = find_browser()
needs_browser = pytest.mark.skipif(BROWSER is None,
                                   reason="needs Chrome, Chromium or Edge")


def _f32(block: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(block), dtype="<f4")


def _u8(block: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(block), dtype="u1")


def _decoded(storage, report, t):
    """The columns the page works with at exported time position t."""
    lat, lon = _mesh(RES)
    cols = pressure_columns(storage.frame(report["times"][t]),
                            report["context"], lat, lon)
    q, b, ps = quantize_columns(cols)
    return cols, dequantize_columns(q, b, ps, (lat.size, lon.size), cols)


# ---------------------------------------------------------------------------
# Payload
# ---------------------------------------------------------------------------

def test_column_page_carries_both_coordinates(tmp_path):
    root, _, _ = write_windy_holed_capsule(tmp_path / "cap", frames=2)
    sim, storage = _storage(root)
    page, report = atmosphere_payload(
        storage, surface=("geopotential_height", 1500.0),
        interactive_surface=True, times="all", resolution_deg=RES)
    A = page["atmosphere"]
    assert A["version"] == 4
    S = A["surface"]
    assert S["kind"] == "geopotential_height" and S["value"] == 1500.0
    assert S["units"] == "m" and S["label"] == "1.5 km"
    C = S["coordinates"]
    assert set(C) == {"pressure", "geopotential_height", "sigma"}
    assert C["geopotential_height"]["units"] == "m"
    assert C["pressure"]["units"] == "Pa"
    assert C["geopotential_height"]["initial"] == 1500.0
    # the other coordinate starts in the middle of its own range (no
    # conversion from the chosen height)
    plo, phi = C["pressure"]["range"]
    assert C["pressure"]["initial"] == pytest.approx(math.sqrt(plo * phi))
    zlo, zhi = C["geopotential_height"]["range"]
    assert zlo < 0 < zhi                     # the archive's own datum
    lat, lon = _mesh(RES)
    g = report["context"].gravity
    lows, highs = [], []
    for t in range(2):
        cols, _ = _decoded(storage, report, t)
        lows.append(cols.geopotential[-1].min() / g)
        highs.append(cols.geopotential[0].max() / g)
    assert zlo == pytest.approx(min(lows)) and zhi == pytest.approx(max(highs))
    assert C["pressure"]["mask_labels"] == {str(k): v for k, v in
                                            MASK_LABELS.items()}
    labels = C["geopotential_height"]["mask_labels"]
    assert {k: labels[k] for k in "0123"} == {
        str(k): v for k, v in HEIGHT_MASK_LABELS.items()}
    assert "4" in labels and "invalid stored column" in labels["4"]
    # the terrain once, as float32, for the masks (not per saved state)
    hs = _f32(A["columns"]["surface_height"]["data"])
    cols, deq = _decoded(storage, report, 0)
    assert (hs == cols.surface_height.ravel().astype(np.float32)).all()
    assert (deq.surface_height.ravel() == hs).all()
    # fields name their coordinates
    fields = {f["id"]: f for f in A["fields"]}
    assert set(fields) == {"temperature", "geopotential_height", "pressure",
                           "wind_speed", "zeta", "delta"}
    assert fields["pressure"]["coordinates"] == ["geopotential_height",
                                                 "sigma"]
    assert fields["geopotential_height"]["coordinates"] == ["pressure",
                                                            "sigma"]
    assert fields["zeta"]["coordinates"] == ["sigma"]
    assert set(fields["temperature"]["coordinates"]) == set(C)
    assert fields["pressure"]["units"] == "hPa"
    assert A["defaults"]["coordinate"] == "geopotential_height"
    assert A["defaults"]["height_m"] == 1500.0
    # the old switch gives the same page; the report says what was chosen
    old, rep_old = atmosphere_payload(
        storage, surface=("geopotential_height", 1500.0),
        interactive_pressure=True, times="all", resolution_deg=RES)
    assert old == page
    assert rep_old["interactive_pressure"] is True
    # a pressure-started page offers heights too
    pp, _ = atmosphere_payload(storage, surface=("pressure", 7e4),
                               interactive_pressure=True, times="all",
                               resolution_deg=RES)
    assert pp["atmosphere"]["surface"]["kind"] == "pressure"
    assert pp["atmosphere"]["defaults"]["coordinate"] == "pressure"
    hz = pp["atmosphere"]["surface"]["coordinates"]["geopotential_height"]
    assert hz["initial"] == pytest.approx(0.5 * sum(hz["range"]))
    # sizes: the terrain block is in the estimate, the extra derived arrays
    # in the decoded memory
    nlev = report["context"].nlev
    assert report["decoded_bytes_per_time"] == decoded_column_bytes(
        nlev, lat.size, lon.size)
    path = sim.view(tmp_path / "h.html", surface=("geopotential_height",
                                                  1500.0),
                    interactive_surface=True, times="all",
                    resolution_deg=RES, open_browser=False)
    assert path.stat().st_size <= report["estimate_bytes"] + 300_000
    assert "-heights-" in __import__(
        "tropoi.representation.visual.atmosphere",
        fromlist=["_default_path"])._default_path(storage, report).name
    # an initial height no state samples is refused at export, by name
    with pytest.raises(SurfaceUnavailableError, match="any height can be"):
        atmosphere_payload(storage, surface=("geopotential_height", 9e5),
                           interactive_surface=True, resolution_deg=RES)
    with pytest.raises(ValueError, match="needs the initial"):
        atmosphere_payload(storage, interactive_surface=True)


def test_fixed_height_surface_per_saved_time(tmp_path):
    root, _, _ = write_windy_holed_capsule(tmp_path / "cap", frames=3)
    sim, storage = _storage(root)
    z0 = -2500.0
    page, report = atmosphere_payload(
        storage, surface=("geopotential_height", z0), times="all",
        resolution_deg=RES)
    A = page["atmosphere"]
    assert A["version"] == 2
    assert [f["id"] for f in A["fields"]] == list(HEIGHT_SURFACE_FIELDS)
    assert A["surface"]["kind"] == "geopotential_height"
    assert A["surface"]["units"] == "m" and A["surface"]["label"] == "-2.5 km"
    assert A["surface"]["mask_labels"] == {str(k): v for k, v in
                                           HEIGHT_MASK_LABELS.items()}
    assert A["levels"][0]["label"] == "-2.5 km"
    n = A["mesh"]["nlat"] * A["mesh"]["nlon"]
    mask = _u8(A["mask"]["data"]).reshape(3, n)
    lat, lon = _mesh(RES)
    for t, index in enumerate(report["times"]):
        s = pressure_surface(storage.frame(index), report["context"], lat,
                             lon, ("geopotential_height", z0))
        assert (mask[t] == s.mask.ravel()).all()
        assert report["mask_counts"][t] == s.counts()
    # geometry: the constant height at every saved time
    assert all(b == [z0, z0] for b in A["height"]["bounds"])
    assert A["streamlines"]["enabled"] is True
    pfield = {f["id"]: f for f in A["fields"]}["pressure"]
    assert pfield["units"] == "hPa" and pfield["range"][0] > 0
    path = sim.view(tmp_path / "zfix.html", surface=("geopotential_height",
                                                     z0),
                    times="all", resolution_deg=RES, open_browser=False)
    assert path.is_file()
    with pytest.raises(ValueError, match="exclusive"):
        atmosphere_payload(storage, surface=("geopotential_height", z0),
                           levels=[0])
    with pytest.raises(ValueError, match="field must be one of"):
        atmosphere_payload(storage, surface=("geopotential_height", z0),
                           field="geopotential_height")


# ---------------------------------------------------------------------------
# The page, offline
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def height_page(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("heights")
    root, _, _ = write_windy_holed_capsule(tmp / "cap", frames=3)
    sim, storage = _storage(root)
    page, report = atmosphere_payload(
        storage, surface=("pressure", 61_234.5), interactive_surface=True,
        times="all", resolution_deg=RES)
    path = sim.view(tmp / "heights.html", surface=("pressure", 61_234.5),
                    interactive_surface=True, times="all",
                    resolution_deg=RES, open_browser=False)
    return path, page, report, storage


def _ready(page, path):
    url = page.open(path)
    page.wait_for(f"{API} && document.getElementById('timeLabel')"
                  ".textContent")
    assert page.eval("window.__terrainGlobe.glOK")
    return url


@needs_browser
def test_page_height_surfaces_match_python_signed_ends_and_empty(
        height_page):
    path, payload, report, storage = height_page
    A = payload["atmosphere"]
    nlat, nlon = A["mesh"]["nlat"], A["mesh"]["nlon"]
    g = report["context"].gravity
    with OfflinePage(BROWSER) as page:
        url = _ready(page, path)
        seen = set()
        for t in range(3):
            page.eval(f"{API}.setTime({t})")
            cols, deq = _decoded(storage, report, t)
            j, i = 9, 20
            ends = [float(deq.geopotential[0, j, i] / g),
                    float(deq.geopotential[-1, j, i] / g),
                    float(deq.geopotential[1, j, i] / g)]
            for z0 in [-9000.0, -2500.0, 0.0, 1234.5, 8000.0, 20_000.0,
                       *ends, 1e6, -1e5]:
                r = page.eval(f"{API}.setHeight({z0!r})")
                assert r["ok"] and r["height_m"] == z0
                assert page.eval(f"{API}.coordinate()") == \
                    "geopotential_height"
                arr = page.eval(f"{API}.surfaceArrays()")
                assert arr["kind"] == "geopotential_height"
                assert arr["value"] == z0 and arr["timeIndex"] == t
                m = np.array(arr["mask"], dtype=np.uint8).reshape(nlat, nlon)
                seen |= set(np.unique(m).tolist())
                try:
                    ref = interpolate_columns(deq, ("geopotential_height",
                                                    z0))
                except SurfaceUnavailableError:
                    assert not (m == MASK_VALID).any()
                    assert page.eval(f"{API}.maskCounts()")["triangles"] == 0
                    note = page.eval("document.getElementById('atmNote')"
                                     ".textContent")
                    assert note.startswith("Empty surface")
                    assert "Z₀ =" in note and "stay live" in note
                    continue
                assert (m == ref.mask).all(), (t, z0)
                ok = m == MASK_VALID
                for name, want in (("temperature", ref["temperature"]),
                                   ("pressure", ref["pressure"]),
                                   ("wind_speed", np.hypot(ref["u"],
                                                           ref["v"])),
                                   ("u", ref["u"]), ("v", ref["v"])):
                    js = np.array(arr[name]).reshape(nlat, nlon)
                    tol = 2e-6 * np.abs(want[ok]).max() + 1e-9
                    assert np.abs(js - want)[ok].max() <= tol, (t, z0, name)
                z = np.array(arr["geopotential_height"])
                assert np.allclose(z, z0, rtol=1e-6, atol=1e-3)
                # the exact surface is within the stored precision too: half
                # a geopotential quantum displaces ln p by at most
                # ½ q / (R_d T) (the measured worst is that, 4e-5 here)
                exact = interpolate_columns(cols, ("geopotential_height", z0))
                both = ok & exact.valid
                p = np.array(arr["pressure"]).reshape(nlat, nlon)
                q = max(np.ptp(cols.geopotential[k])
                        for k in range(cols.nlev)) / 65535.0
                bound = 1.05 * 0.5 * q / (287.04 * cols.temperature.min())
                assert np.abs(p / exact["pressure"] - 1)[both].max() \
                    <= bound + 2e-6
                # seam column equals the first
                assert (m[:, 0] == m[:, -1]).all()
                counts = page.eval(f"{API}.maskCounts()")["counts"]
                want = {str(k): int((m == k).sum()) for k in range(4)}
                assert counts == want                 # (no code 4 here)
                # labels name the height
                lab = page.eval("document.getElementById('levelLabel')"
                                ".textContent")
                assert "height surface (interpolated here)" in lab
                note = page.eval("document.getElementById('atmNote')"
                                 ".textContent")
                assert f"({z0:.8g} m)" in note
        assert seen == {0, 1, 2, 3}
        # hover names the height reasons
        page.eval(f"{API}.setTime(0); {API}.setHeight(-2500)")
        m = np.array(page.eval(f"{API}.surfaceArrays()")["mask"]).reshape(
            nlat, nlon)
        lat, lon = _mesh(RES)
        jj, ii = np.argwhere(m == MASK_BELOW_SURFACE)[0]
        rd = page.eval(f"{API}.sample({lat[jj]}, {lon[ii]})")["readout"]
        assert rd.startswith("no sample on the -2.5 km surface: below the "
                             "terrain")
        # a node whose bilinear cell is all valid reads a value (hover
        # refuses a point when any node of its cell is masked)
        cell_ok = ((m[:-1, :-1] == 0) & (m[1:, :-1] == 0) & (m[:-1, 1:] == 0)
                   & (m[1:, 1:] == 0))
        jj, ii = np.argwhere(cell_ok)[0]
        rd = page.eval(f"{API}.sample({lat[jj]}, {lon[ii]})")["readout"]
        assert "interpolated" in rd and "Temperature" in rd
        assert page.requests == [url]
        assert page.console_errors == [] and page.exceptions == []


@needs_browser
def test_fixed_height_geometry_is_time_independent_values_are_not(
        height_page):
    path, payload, report, storage = height_page
    with OfflinePage(BROWSER) as page:
        _ready(page, path)
        page.eval(f"{API}.setHeight(1234.5); {API}.setExaggeration(20)")
        geos, temps, masks = [], [], []
        for t in range(3):
            page.eval(f"{API}.setTime({t})")
            geo = page.eval(f"{API}.geometry()")
            geos.append(np.array(geo["pos"]).reshape(-1, 3))
            masks.append(np.array(geo["mask"]))
            temps.append(np.array(page.eval(
                f"{API}.surfaceArrays()")["temperature"]))
        both = (masks[0] == 0) & (masks[2] == 0)
        assert both.sum() > 100
        # the same radius R + E Z0 at every valid node, at every time
        for t in (1, 2):
            common = (masks[0] == 0) & (masks[t] == 0)
            assert (geos[t][common] == geos[0][common]).all()
        r = np.linalg.norm(geos[0][masks[0] == 0], axis=1)
        R = payload["atmosphere"]["radius_m"]
        assert np.allclose(r, 1 + 20 * 1234.5 / R, rtol=1e-6)
        # fields and validity change with the saved state (warming fixture)
        assert np.abs(temps[2] - temps[0])[both].min() > 1.0
        # warming lifts the full levels: the band between the terrain and
        # the lowest level moves, so validity changes at the same height
        assert (masks[0] != masks[2]).any()
        assert page.console_errors == [] and page.exceptions == []


@needs_browser
def test_coordinate_switch_keeps_each_value_and_controls_work(height_page):
    path, payload, report, storage = height_page
    C = payload["atmosphere"]["surface"]["coordinates"]
    zlo, zhi = C["geopotential_height"]["range"]
    with OfflinePage(BROWSER) as page:
        _ready(page, path)
        assert page.eval(f"{API}.coordinate()") == "pressure"
        assert page.eval(f"{API}.pressure()") == 61_234.5
        assert page.eval("document.getElementById('heightBlock').hidden")
        fields = page.eval("[...document.getElementById('fieldButtons')"
                           ".children].filter(b => !b.hidden)"
                           ".map(b => b.textContent)")
        assert fields == ["Temperature", "Geopotential height", "Wind speed"]
        # to height: its own initial value, not a conversion of 612 hPa
        page.eval("document.getElementById('coordHeight').click()")
        assert page.eval(f"{API}.coordinate()") == "geopotential_height"
        assert page.eval(f"{API}.height()") == pytest.approx(
            0.5 * (zlo + zhi))
        assert not page.eval("document.getElementById('heightBlock').hidden")
        assert page.eval("document.getElementById('pressureBlock').hidden")
        fields = page.eval("[...document.getElementById('fieldButtons')"
                           ".children].filter(b => !b.hidden)"
                           ".map(b => b.textContent)")
        assert fields == ["Temperature", "Wind speed", "Pressure"]
        c = page.eval(f"{API}.heightControls()")
        assert c["unit"] == 1000 and "linear" in c["note"]
        assert "km" in c["label"] and " m" in c["label"]
        # typed heights: signed, zero, km and m, between slider positions
        for text, unit, want in (("-2.5", 1000, -2500.0), ("0", 1, 0.0),
                                 ("1234.5", 1, 1234.5), ("7,25", 1000,
                                                         7250.0)):
            page.eval(f"{{ document.getElementById('heightUnit').value = "
                      f"'{unit}'; const i = document.getElementById("
                      f"'heightInput'); i.value = {text!r}; "
                      "i.dispatchEvent(new KeyboardEvent('keydown', "
                      "{key: 'Enter', bubbles: true})) }")
            assert page.eval(f"{API}.height()") == want
            assert page.eval(f"{API}.surfaceArrays()")["value"] == want
        slider = page.eval(f"{API}.heightControls()")["slider"]
        want_pos = round(1000 * (7250.0 - zlo) / (zhi - zlo))
        assert slider == max(0, min(1000, want_pos))
        before = page.eval(f"{API}.digest()")
        for bad in ("abc", "", "NaN", "Infinity", "1e400", "--3"):
            page.eval("document.getElementById('heightInput').value = "
                      f"{bad!r}; document.getElementById('heightApply')"
                      ".click()")
            c = page.eval(f"{API}.heightControls()")
            assert c["invalid"] and c["note"].startswith("Not applied:"), bad
            assert page.eval(f"{API}.height()") == 7250.0
            assert page.eval(f"{API}.digest()") == before
        for bad in ("'abc'", "NaN", "Infinity", "null"):
            r = page.eval(f"{API}.setHeight({bad})")
            assert not r["ok"] and r["height_m"] == 7250.0
        # the linear slider moves the height
        page.eval("const s = document.getElementById('heightSlider');"
                  "s.value = 250; s.dispatchEvent(new Event('input'));"
                  "s.dispatchEvent(new Event('change'))")
        assert page.eval(f"{API}.height()") == pytest.approx(
            zlo + 0.25 * (zhi - zlo), rel=1e-12)
        z_kept = page.eval(f"{API}.height()")
        # back to pressure: 612.345 hPa kept, then back to height: kept too
        page.eval("document.getElementById('coordPressure').click()")
        assert page.eval(f"{API}.pressure()") == 61_234.5
        assert page.eval(f"{API}.surfaceArrays()")["kind"] == "pressure"
        page.eval(f"{API}.setField(2)")    # wind speed (index as before)
        assert page.eval(f"{API}.setField('pressure')") is False  # not here
        page.eval(f"{API}.setCoordinate('geopotential_height')")
        assert page.eval(f"{API}.height()") == z_kept
        # the field choice is kept per coordinate
        assert page.eval(f"{API}.info()")["field"] == "temperature"
        page.eval(f"{API}.setCoordinate('pressure')")
        assert page.eval(f"{API}.info()")["field"] == "wind_speed"
        # setPressure / setHeight select their coordinate
        page.eval(f"{API}.setHeight(500)")
        assert page.eval(f"{API}.coordinate()") == "geopotential_height"
        page.eval(f"{API}.setPressure(50000)")
        assert page.eval(f"{API}.coordinate()") == "pressure"
        assert page.eval(f"{API}.height()") == 500.0
        assert page.console_errors == [] and page.exceptions == []


@needs_browser
def test_streamlines_on_height_surfaces_and_deferred_height_drag(
        height_page):
    path, payload, report, storage = height_page
    lat, lon = _mesh(RES)
    shape = (lat.size, lon.size)
    with OfflinePage(BROWSER) as page:
        _ready(page, path)
        masked = 0
        for t in range(3):
            page.eval(f"{API}.setTime({t})")
            for z0 in (-2500.0, 1234.5, 8000.0, 20_000.0):
                page.eval(f"{API}.setHeight({z0!r})")
                arr = page.eval(f"{API}.surfaceArrays()")
                valid = np.array(arr["mask"]).reshape(shape) == 0
                ref = reference(np.array(arr["u"]).reshape(shape),
                                np.array(arr["v"]).reshape(shape), valid)
                sl = page.eval(f"{API}.streamlines()")
                assert sl["current"] and sl["key"] == arr["key"]
                assert arr["key"].startswith("geopotential_height:")
                assert_same_lines(sl["lines"], ref)
                assert_inside(sl["lines"], valid)
                masked += bool(ref) and not valid.all()
        assert masked >= 6
        # a coordinate switch draws the other coordinate's lines at once
        page.eval(f"{API}.setCoordinate('pressure')")
        sl = page.eval(f"{API}.streamlines()")
        assert sl["current"] and sl["key"].startswith("pressure:")
        # the height slider defers like the pressure slider
        page.eval(f"{API}.setCoordinate('geopotential_height')")
        state = page.eval("""(async () => {
            const s = document.getElementById('heightSlider'), a = window.__terrainGlobe.atmosphere;
            for (const v of [100, 300, 333]) { s.value = v; s.dispatchEvent(new Event('input', {bubbles: true})); }
            await new Promise(r => requestAnimationFrame(r));
            return a.streamlines();
        })()""")
        assert state["pending"] and not state["current"]
        assert state["drawnVerts"] == 0
        time.sleep(3 * STREAMLINE_SETTLE_MS / 1000)
        sl = page.eval(f"{API}.streamlines()")
        assert sl["current"] and sl["key"] == state["surfaceKey"]
        assert page.console_errors == [] and page.exceptions == []


@needs_browser
def test_a_corrupted_stored_column_is_masked_not_divided(tmp_path):
    """The export refuses inverted stored columns, so a page never holds
    one; if a file were altered, the page masks the column (code 4) and
    says so instead of dividing by a zero or inverted interval."""
    root, _, _ = write_windy_holed_capsule(tmp_path / "cap", frames=1)
    _, storage = _storage(root)
    page_doc, report = atmosphere_payload(
        storage, surface=("geopotential_height", 1234.5),
        interactive_surface=True, times="all", resolution_deg=RES)
    block = page_doc["atmosphere"]["columns"]["times"][0]["geopotential"]
    lat, lon = _mesh(RES)
    n = lat.size * lon.size
    q = np.frombuffer(base64.b64decode(block["data"]), "<u2").copy()
    q = q.reshape(-1, n)
    node = 5 * lon.size + 7
    q[1, node] = 65535                       # level 2 far above level 1
    q[0, node] = 0
    block["data"] = base64.b64encode(q.tobytes()).decode("ascii")
    from tropoi.representation.visual.terrain import write_globe_page
    path = write_globe_page(page_doc, page_doc["title"], tmp_path / "bad.html")
    with OfflinePage(BROWSER) as page:
        _ready(page, path)
        m = page.eval(f"{API}.surfaceArrays()")["mask"]
        assert m[node] == 4 and m.count(4) == 1
        assert np.isfinite(page.eval(f"{API}.surfaceArrays()")
                           ["pressure"]).all()
        note = page.eval("document.getElementById('atmNote').textContent")
        assert "invalid stored column" in note
        assert page.console_errors == [] and page.exceptions == []
