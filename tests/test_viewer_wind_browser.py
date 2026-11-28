"""The Explorer's wind layer in network-isolated Chromium.

Restores the flat-map wind streamlines that the old map drew (map.js flow()):
the same integrator as the globe (overlays.js), now drawn on the Explorer's
longitude × latitude field, which is the page's only flat plotting surface.
"""
import json

import pytest

from tools.browser_cdp import OfflinePage, find_browser
from test_viewer_explorer_browser import A, EXPLORED, explore, serve
from test_viewer_local_browser import until

WIND = "document.querySelector('[data-e=wind]')"


@pytest.fixture
def browser():
    found = find_browser()
    if not found:
        pytest.skip("Needs a Chromium browser")
    return found


# A solid-body eastward wind on the 2° display mesh: u = 20 cos(lat), v = 0.
# Its streamlines are parallels, traced eastward. The southern hemisphere is
# masked, so no drawn point may lie south of the first valid row.
SOLID_BODY = """(async () => {
  const {windLines} = await import('./explorer.js');
  const lon = Array.from({length: 181}, (_, i) => 2 * i), lat = Array.from({length: 91}, (_, j) => -90 + 2 * j);
  const u = lat.map(y => lon.map(() => y < 0 ? null : 20 * Math.cos(y * Math.PI / 180)));
  const v = lat.map(y => lon.map(() => y < 0 ? null : 0));
  const axis = (coordinate, values) => ({coordinate, values});
  const plot = {x: axis('longitude', lon), y: axis('latitude', lat), wind: {u, v}};
  const transposed = {x: axis('latitude', lat), y: axis('longitude', lon),
                      wind: {u: lon.map((_, i) => lat.map((_, j) => u[j][i])), v: lon.map((_, i) => lat.map((_, j) => v[j][i]))}};
  const lines = windLines(plot), other = windLines(transposed);
  let drift = 0, south = 90, westward = 0;
  for (const line of lines) {
    for (const [y] of line) { drift = Math.max(drift, Math.abs(y - line[0][0])); south = Math.min(south, y); }
    for (let k = 1; k < line.length; k++) {
      const d = ((line[k][1] - line[k - 1][1]) % 360 + 540) % 360 - 180;
      if (d <= 0) westward++;
    }
  }
  return {count: lines.length, drift, south, westward, same: JSON.stringify(lines) === JSON.stringify(other)};
})()"""


def test_streamlines_of_a_solid_body_wind_are_eastward_parallels(browser):
    server, thread = serve([A])
    try:
        with OfflinePage(browser, width=1200, height=1400, loopback_port=server.server_port) as page:
            page.call("Page.navigate", url=server.url + "?preset=map")
            until(page, EXPLORED)
            r = page.eval(SOLID_BODY)
            assert r["count"] > 20 and r["westward"] == 0 and r["same"]
            # A parallel is a small circle, not a geodesic: the globe's integrator (triangle-interpolated
            # Cartesian wind, chord steps renormalized to the sphere) drifts 0.033° along the longest
            # line (measured 2026-11-28). Bound: a twentieth of the 2° display cell.
            assert r["drift"] < 0.1, r
            assert r["south"] >= -1e-9, r        # never in a masked cell
            assert not page.exceptions, page.exceptions
    finally:
        server.shutdown()
        thread.join()


def test_wind_layer_on_a_map_and_refused_on_a_section(browser):
    server, thread = serve([A])
    try:
        with OfflinePage(browser, width=1400, height=1800, loopback_port=server.server_port) as page:
            page.call("Page.navigate", url=server.url + "?preset=map")
            until(page, EXPLORED)
            assert page.eval(f"{WIND}.disabled") is False
            page.eval(f"(() => {{const el = {WIND}; el.checked = true; el.dispatchEvent(new Event('change', {{bubbles: true}}));}})()")
            until(page, "viewer.explorer.data?.companions?.wind_u && viewer.explorer.layout?.winds > 20 && " + EXPLORED)
            sel = json.loads(page.eval("viewer.explorer.loaded"))
            assert sel["options"]["wind"] is True
            assert any(n.startswith("Wind layer:") for n in page.eval("viewer.explorer.data.notes"))
            # A section has no wind surface: the control is disabled with its reason.
            explore(page, "preset", "zonal_mean")
            until(page, f"{WIND}.disabled === true && " + EXPLORED)
            reason = page.eval("document.querySelector('[data-e=wind-reason]').textContent")
            assert "longitude × latitude" in reason
            assert "wind" not in json.loads(page.eval("viewer.explorer.loaded"))["options"]
            assert not page.exceptions, page.exceptions
    finally:
        server.shutdown()
        thread.join()
