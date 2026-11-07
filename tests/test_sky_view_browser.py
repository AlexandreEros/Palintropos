"""The sky panel of the offline globe page, checked in a headless browser.

Opens exported pages from ``file://`` with networking blocked
(``tools.browser_cdp``) and checks:

* page/NumPy parity: the page's float64 radiance table equals
  ``sky.radiance_table`` evaluated on the page's own column, at sampled
  directions (saved PE state and hypothetical column);
* the page's saved-state column agrees with Python's exact column at the
  initial observer up to display-mesh interpolation (reported, loose);
* altitude <-> pressure placement round-trips in the page;
* refusals (underground, above the top) show a message, raise nothing;
* the star azimuth only rotates the table (no recomputation, same numbers);
* switching back to the globe restores it; pages without a sky block show
  no sky controls; one request (the file itself), no console errors.

Skipped when no Chromium-family browser is installed.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.browser_cdp import OfflinePage, find_browser  # noqa: E402
from test_atmosphere_view import write_pe_capsule  # noqa: E402

from tropoi.representation.visual import sky  # noqa: E402
from tropoi.spatial.optics import COMPOSITION_PRESETS, Composition  # noqa: E402

BROWSER = find_browser()
pytestmark = pytest.mark.skipif(BROWSER is None,
                                reason="needs Chrome, Chromium or Edge")
API = "window.__terrainGlobe.sky"
DIRECTIONS = [(0, 0), (10, 5), (45, 0), (45, 30), (60, 60), (80, 20),
              (88, 0), (89, 45), (90, 10), (91, 0), (95, 30), (120, 60),
              (178, 2)]


def _assert_clean(page, url):
    assert page.requests == [url], page.requests
    assert page.console_errors == [], page.console_errors
    assert page.exceptions == [], page.exceptions


def _parity(page, payload_sky):
    info = page.eval(f"{API}.info()")
    assert info["error"] is None, info["error"]
    col = page.eval(f"{API}.column()")
    o = payload_sky["optics"]
    ref = sky.radiance_table(
        (col["z"], col["lnp"], col["T"]), radius_m=payload_sky["radius_m"],
        observer_z_m=info["place"]["z"],
        star_elevation_deg=info["star"]["el"],
        cross_section_m2=o["cross_section_m2"],
        irradiance=np.array(o["star_radiance"]) * o["star_solid_angle_sr"])
    worst = 0.0
    for iz, ia in DIRECTIONS:
        got = np.array(page.eval(f"{API}.spectral({iz}, {ia})"))
        want = ref["radiance"][iz, ia]
        np.testing.assert_allclose(got, want, rtol=1e-9, atol=0.0,
                                   err_msg=f"direction {iz}, {ia}")
        worst = max(worst, float(np.max(np.abs(got - want) /
                                        np.where(want > 0, want, 1))))
    return worst


def _payload(path):
    import re
    text = pathlib.Path(path).read_text(encoding="utf-8")
    raw = re.search(r'id="terrain-data" type="application/json">(.*?)'
                    r'</script>', text, re.S).group(1)
    return json.loads(raw.replace("\\u003c", "<").replace("\\u003e", ">"))


@pytest.fixture(scope="module")
def saved_page(tmp_path_factory):
    from tropoi.representation.archive import open_simulation
    tmp = tmp_path_factory.mktemp("sky")
    root, _, _ = write_pe_capsule(tmp / "cap", frames=2)
    spec = sky.SkySpec(COMPOSITION_PRESETS["earth_dry_air_illustrative"],
                       observer_lat_deg=20.0, observer_lon_deg=10.0)
    return open_simulation(root).view(tmp / "saved.html", open_browser=False,
                                      resolution_deg=6.0, times="all",
                                      sky=spec)


def test_saved_state_sky_parity_and_controls(saved_page):
    payload = _payload(saved_page)
    with OfflinePage(BROWSER) as page:
        url = page.open(saved_page)
        assert page.eval("document.getElementById('modeSw').hidden") is False
        page.eval(f"{API}.setMode('sky')")
        assert page.eval("document.getElementById('sky').hidden") is False
        _parity(page, payload["sky"])

        # page column vs Python's exact one at the initial observer: same
        # construction, display-mesh interpolation only (loose, reported)
        col = page.eval(f"{API}.column()")
        exact = payload["sky"]["initial_column"]
        assert len(col["z"]) == len(exact["z_m"])
        np.testing.assert_allclose(col["lnp"], exact["ln_p"], atol=0.02)

        # star azimuth only rotates: the same table, no recomputation
        before = page.eval(f"{API}.spectral(45, 30)")
        page.eval(f"{API}.setStar(300, {payload['sky']['star']['elevation_deg']})")
        assert page.eval(f"{API}.spectral(45, 30)") == before

        # altitude <-> pressure in the page
        page.eval(f"{API}.setObserver({{pressure_pa: 70000}})")
        z = page.eval(f"{API}.info()")["place"]["z"]
        page.eval(f"{API}.setObserver({{altitude_m: {z!r}}})")
        assert page.eval(f"{API}.info()")["place"]["p"] == pytest.approx(
            70000.0, rel=1e-12)

        # refusals are messages, not exceptions
        page.eval(f"{API}.setObserver({{altitude_m: -500}})")
        assert "underground" in page.eval(f"{API}.info()")["error"]
        page.eval(f"{API}.setObserver({{altitude_m: 1e6}})")
        assert "above the top saved full level" in \
            page.eval(f"{API}.info()")["error"]
        page.eval(f"{API}.setObserver({{pressure_pa: 2e5}})")
        assert "underground" in page.eval(f"{API}.info()")["error"]
        page.eval(f"{API}.setObserver({{ground: true}})")
        assert page.eval(f"{API}.info()")["error"] is None

        # a new saved time rebuilds the column (still parity)
        page.eval("window.__terrainGlobe.atmosphere.setTime(0)")
        page.eval(f"{API}.update()")
        _parity(page, payload["sky"])

        page.eval(f"{API}.setMode('globe')")
        assert page.eval("document.getElementById('sky').hidden") is True
        _assert_clean(page, url)


def test_hypothetical_sky_parity(tmp_path):
    import math
    from tropoi.representation.visual.terrain import (build_terrain_preview,
                                                      write_globe_html)
    lm = np.zeros((11, 11), complex)
    lm[0, 0] = 1000.0 * math.sqrt(4 * math.pi)
    preview = build_terrain_preview(lm, radius_m=3.3895e6, title="hyp")
    spec = sky.SkySpec(
        Composition.of(CO2=1.0, label="pure CO2 (stated)"),
        observer_lat_deg=-15.0, star_elevation_deg=12.0,
        hypothetical=sky.HypotheticalColumn(
            surface_pressure_pa=610.0, temperature_k=230.0, gravity=3.71,
            gas_constant="from_composition", lapse_rate_k_per_m=0.0025,
            temperature_floor_k=150.0))
    path = write_globe_html(preview, tmp_path / "hyp.html", sky=spec)
    payload = _payload(path)
    with OfflinePage(BROWSER) as page:
        url = page.open(path)
        page.eval(f"{API}.setMode('sky')")
        info = page.eval(f"{API}.info()")
        assert info["columnKind"] == "hypothetical"
        assert "HYPOTHETICAL" in page.eval(
            "document.getElementById('skyLabel').textContent")
        _parity(page, payload["sky"])
        # terrain is 1000 m everywhere: 500 m is underground on this planet
        page.eval(f"{API}.setObserver({{altitude_m: 500}})")
        assert "underground" in page.eval(f"{API}.info()")["error"]
        _assert_clean(page, url)


def test_pages_without_sky_show_no_sky_controls(tmp_path):
    import math
    from tropoi.representation.visual.terrain import (build_terrain_preview,
                                                      write_globe_html)
    lm = np.zeros((11, 11), complex)
    lm[0, 0] = 10.0 * math.sqrt(4 * math.pi)
    path = write_globe_html(build_terrain_preview(lm, radius_m=6.4e6,
                                                  title="plain"),
                            tmp_path / "plain.html")
    with OfflinePage(BROWSER) as page:
        url = page.open(path)
        assert page.eval("window.__terrainGlobe.sky") is None
        assert page.eval("document.getElementById('modeSw').hidden") is True
        assert page.eval("document.getElementById('skySection').hidden") \
            is True
        assert page.eval("document.getElementById('sky').hidden") is True
        _assert_clean(page, url)
