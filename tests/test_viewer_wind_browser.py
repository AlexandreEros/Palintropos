"""The Explorer's horizontal-wind representations in network-isolated headless Chrome.

Three representations of v = r̂ × ∇ψ + ∇χ on a longitude × latitude result:
ψ contours (streamlines of the rotational wind, spacing Δψ/|v_rot|), streamlines
of the full wind (overlays.js, the globe's integrator) and arrows at native
nodes. Analytic cases run the page's own modules; archive cases go through the
controls.
"""
import json

import pytest

from tools.browser_cdp import OfflinePage, find_browser
from test_viewer_explorer_browser import A, EXPLORED, explore, serve
from test_viewer_local_browser import until
from test_viewer_manifest import SWE as SWE_ID
from test_viewer_wind_layer import HS, ROOT

W5 = ROOT / SWE_ID
LAYOUT = "viewer.explorer.layout?.winds"

# A 2° display mesh (latitudes −90…90, longitudes 0…360 with the closing meridian) and a payload-like plot.
MESH = """
  const lon = Array.from({length: 181}, (_, i) => 2 * i), lat = Array.from({length: 91}, (_, j) => -90 + 2 * j);
  const D = Math.PI / 180, a = 6.371e6;
  const field = f => lat.map(y => lon.map(x => f(y, x)));
  const axis = (coordinate, values) => ({coordinate, values});
  const plot = (wind, transpose = false) => transpose ?
    {x: axis('latitude', lat), y: axis('longitude', lon), wind: {u: T(wind.u), v: T(wind.v), psi: wind.psi && T(wind.psi), info: wind.info,
      settings: wind.settings}} : {x: axis('longitude', lon), y: axis('latitude', lat), wind};
  const T = m => m[0].map((_, i) => m.map(r => r[i]));
  const settings = (o = {}) => ({lines: 'auto', limit: 1, interval: null, density: 1, arrows: false, spacing: 10, length: 'speed',
    colour: false, ...o});
  const info = (share, nondiv = false, max = 20) => ({psi: {available: true, radius_m: a},
    kinetic_energy: {divergent_share: share, non_divergent: nondiv, rotational_max_speed: max}});
"""

SOLID_BODY = """(async () => {
  const {psiContours, windPlan, windLines, quiverDirection} = await import('./explorer.js');
  const {quiverStations} = await import('./wind.js');
  """ + MESH + """
  const U = 20, masked = y => y < 0;
  const wind = {u: field((y) => masked(y) ? null : U * Math.cos(y * D)), v: field((y) => masked(y) ? null : 0),
                psi: field((y) => masked(y) ? null : -U * a * Math.sin(y * D)), info: info(0, true, U), settings: settings()};
  const c = psiContours(plot(wind), 1e7);
  let lines = 0, closed = 0, drift = 0, south = 90, westward = 0;
  for (const set of c.sets) for (const line of set.lines) {
    lines++; if (line.closed) closed++;
    const y0 = Math.asin(-set.level / (U * a)) / D;               // the parallel of this ψ level
    for (const [y] of line.points) { drift = Math.max(drift, Math.abs(y - y0)); south = Math.min(south, y); }
    for (let k = 1; k < line.points.length; k++) {
      const d = ((line.points[k][1] - line.points[k - 1][1]) % 360 + 540) % 360 - 180;
      if (d <= 0) westward++;
    }
  }
  const t = psiContours(plot(wind, true), 1e7);
  const plan = windPlan(plot({...wind, settings: settings({arrows: true, spacing: 10})}));
  const stations = plan.quiver;
  let speedError = 0;
  for (const p of stations) speedError = Math.max(speedError, Math.abs(Math.hypot(p.u, p.v) - U * Math.cos(p.lat * D)));
  // Screen directions: y grows downward on screen (latPx < 0 for an upward latitude axis).
  const east = quiverDirection({lat: 40, u: 5, v: 0}, 2, -2, true), north = quiverDirection({lat: 40, u: 0, v: 5}, 2, -2, true);
  const northT = quiverDirection({lat: 40, u: 0, v: 5}, 2, 3, false);
  // The integrator's defaults (the globe's) are unchanged: 40 lines here before this work.
  const sl = windLines(plot(wind));
  let slDrift = 0, slWest = 0;
  for (const line of sl) for (let k = 1; k < line.length; k++) {
    slDrift = Math.max(slDrift, Math.abs(line[k][0] - line[0][0]));
    if ((((line[k][1] - line[k - 1][1]) % 360 + 540) % 360 - 180) <= 0) slWest++;
  }
  return {streamlines: sl.length, slDrift, slWest, lines, closed, levels: c.sets.length, drift, south, westward, mode: plan.mode, quiver: stations.length,
          stationSouth: Math.min(...stations.map(p => p.lat)), poles: stations.filter(p => Math.abs(p.lat) > 85).length,
          speedError, east, north, northT, sameTransposed: JSON.stringify(t.sets) === JSON.stringify(c.sets)};
})()"""

VORTEX = """(async () => {
  const {psiContours, windPlan} = await import('./explorer.js');
  """ + MESH + """
  // Isolated vortex ψ = A exp(−(d/r0)²) about (30°N, 120°E), plus a second one at (30°N, 150°E) making a saddle.
  const dist = (y, x, y0, x0) => Math.acos(Math.min(1, Math.sin(y*D)*Math.sin(y0*D) + Math.cos(y*D)*Math.cos(y0*D)*Math.cos((x-x0)*D)));
  const psiOf = (A, two) => (y, x) => A * Math.exp(-((dist(y, x, 30, 120) / (12 * D)) ** 2)) +
    (two ? A * Math.exp(-((dist(y, x, 30, 150) / (12 * D)) ** 2)) : 0);
  const velocity = (f, y, x) => {     // u = −(1/a) ∂ψ/∂φ, v = (1/(a cos φ)) ∂ψ/∂λ
    const h = 1e-3;
    return [-(f(y + h, x) - f(y - h, x)) / (2 * h * D * a), (f(y, x + h) - f(y, x - h)) / (2 * h * D * a * Math.cos(y * D))];
  };
  const run = (A, two) => {
    const f = psiOf(A, two), zero = field(() => 0);
    const c = psiContours(plot({u: zero, v: zero, psi: field(f), info: info(0, true), settings: settings()}), 1e6);
    let lines = 0, closed = 0, agree = 0, total = 0;
    for (const set of c.sets) for (const line of set.lines) {
      lines++; if (line.closed) closed++;
      for (let k = 1; k < line.points.length; k++) {
        const [y0, x0] = line.points[k - 1], [y1, x1] = line.points[k];
        const [u, v] = velocity(f, (y0 + y1) / 2, (x0 + x1) / 2);
        const dot = ((x1 - x0) * Math.cos((y0 + y1) / 2 * D)) * u + (y1 - y0) * v;
        total++; if (dot > 0) agree++;
      }
    }
    return {lines, closed, agree, total, levels: c.sets.length};
  };
  return {one: run(1e7, false), stronger: run(2e7, false), pair: run(1e7, true)};
})()"""

DIVERGENT = """(async () => {
  const {windPlan, resolveWind} = await import('./explorer.js');
  """ + MESH + """
  // χ = a·c·sin φ: v = c cos φ northward everywhere, ψ ≡ 0. The rotational contours are empty;
  // the full wind is not, and the automatic choice must draw it, saying so.
  const c0 = 5, zero = field(() => 0), wind = {u: zero, v: field(y => c0 * Math.cos(y * D)), psi: zero,
    info: info(1, false, 0), settings: settings()};
  const auto = windPlan(plot(wind));
  const forced = windPlan(plot({...wind, info: info(0.3, false, 10), settings: settings({lines: 'psi'})}));
  const small = resolveWind({...wind, info: info(0.004, false, 10), settings: settings({limit: 1})});
  const strict = resolveWind({...wind, info: info(0.004, false, 10), settings: settings({limit: 0.1})});
  const northward = auto.lines.every(line => line[line.length - 1][0] > line[0][0]);
  return {auto: {mode: auto.mode, why: auto.why, lines: auto.lines.length, key: auto.key.map(k => k.text)}, northward,
          forced: {mode: forced.mode, key: forced.key.map(k => [k.text, Boolean(k.warn)])}, small, strict};
})()"""


@pytest.fixture
def browser():
    found = find_browser()
    if not found:
        pytest.skip("Needs a Chromium browser")
    return found


def _page_eval(browser, script):
    server, thread = serve([A])
    try:
        with OfflinePage(browser, width=1200, height=1400, loopback_port=server.server_port) as page:
            page.call("Page.navigate", url=server.url + "?preset=map")
            until(page, EXPLORED)
            result = page.eval(script)
            assert not page.exceptions, page.exceptions
            return result
    finally:
        server.shutdown()
        thread.join()


def test_solid_body_rotation(browser):
    """A: ψ = −U a sin φ: contours are parallels, closed across the seam, eastward; arrows exact; masks respected."""
    r = _page_eval(browser, SOLID_BODY)
    assert r["lines"] == r["levels"] > 5 and r["closed"] == r["lines"], r        # every level one closed loop
    assert r["drift"] < 0.02, r              # linear interpolation of sin φ between 2° rows
    assert r["south"] >= 0 and r["westward"] == 0 and r["sameTransposed"], r
    assert r["mode"] == "psi" and r["quiver"] > 50 and r["stationSouth"] >= 0 and r["poles"] == 0, r
    assert r["speedError"] < 1e-5, r         # Float32 samples of U cos φ
    assert r["streamlines"] == 40 and r["slDrift"] < 0.1 and r["slWest"] == 0, r
    assert r["east"] == [1, 0] and r["north"] == [0, -1] and r["northT"] == [1, 0], r


def test_isolated_vortex_and_saddle(browser):
    """B: closed loops, arrows along the velocity, and twice the speed gives twice the lines at one interval."""
    r = _page_eval(browser, VORTEX)
    for case in ("one", "stronger", "pair"):
        c = r[case]
        assert c["lines"] > 3 and c["closed"] == c["lines"], r
        assert c["agree"] == c["total"], r            # every segment oriented along v_rot
    assert abs(r["stronger"]["levels"] - 2 * r["one"]["levels"]) <= 1, r


def test_divergent_flow_is_never_labelled_as_rotational(browser):
    """C: a purely divergent wind has no ψ contours; automatic draws full-wind streamlines and says so."""
    r = _page_eval(browser, DIVERGENT)
    assert r["auto"]["mode"] == "streamlines" and r["auto"]["why"] == "above" and r["auto"]["lines"] > 20, r
    assert "full wind" in r["auto"]["key"][0] and "divergent part 100 %" in r["auto"]["key"][0], r
    assert r["northward"], r
    assert r["forced"]["mode"] == "psi" and "rotational wind only" in r["forced"]["key"][0][0], r
    assert r["forced"]["key"][1][1] and "NOT drawn" in r["forced"]["key"][1][0], r
    assert r["small"]["mode"] == "psi" and r["strict"]["mode"] == "streamlines", r


def _set(page, key, value):
    page.eval(f"""(() => {{const el = document.querySelector('[data-e="{key}"]');
        if (el.type === 'checkbox') el.checked = {json.dumps(value)}; else el.value = {json.dumps(str(value))};
        el.dispatchEvent(new Event('change', {{bubbles: true}}));}})()""")


def test_w5_rotational_contours_match_the_readme_figure(browser):
    """D: day 15 of the README's W5 run: automatic ψ contours every 2 × 10⁷ m² s⁻¹, closed, 4.5° at 40 m s⁻¹,
    0.071 % divergent energy reported; arrows with a speed key."""
    if not W5.is_dir():
        pytest.skip(f"Missing archive: {W5}")
    server, thread = serve([W5])
    try:
        with OfflinePage(browser, width=1400, height=1800, loopback_port=server.server_port) as page:
            page.call("Page.navigate", url=server.url + "?preset=map&time=3")
            until(page, EXPLORED)
            explore(page, "quantity", "free_surface_height")
            until(page, "viewer.explorer.data?.quantity === 'free_surface_height' && " + EXPLORED)
            _set(page, "wind-lines", "auto")
            until(page, f"{LAYOUT}?.mode === 'psi' && " + EXPLORED)
            w = page.eval(LAYOUT)
            assert w["interval"] == 2e7 and w["closed"] == w["lines"] > 8, w
            text = page.eval("viewer.explorer.layout && JSON.stringify(viewer.explorer.data.wind)")
            share = json.loads(text)["kinetic_energy"]["divergent_share"]
            assert abs(share - 7.130896400012388e-4) < 1e-15
            _set(page, "wind-arrows", True)
            until(page, f"{LAYOUT}?.quiver > 50")
            scale = page.eval(f"{LAYOUT}.quiverScale")
            assert scale["reference"] == 20 and 35 < scale["vmax"] < 45, scale
            # Display settings sent no request: the loaded selection is unchanged.
            assert json.loads(page.eval("viewer.explorer.loaded"))["options"]["wind"] is True
            _set(page, "wind-interval", 1e7)
            until(page, f"{LAYOUT}?.interval === 1e7")
            assert not page.exceptions, page.exceptions
    finally:
        server.shutdown()
        thread.join()


def test_held_suarez_instantaneous_and_time_mean(browser):
    """E: an atmospheric level: every representation draws for one state and for a time mean; on a pressure
    surface ψ is unavailable and the automatic choice draws full-wind streamlines, saying why."""
    if not HS.is_dir():
        pytest.skip(f"Missing archive: {HS}")
    server, thread = serve([HS])
    try:
        with OfflinePage(browser, width=1400, height=1800, loopback_port=server.server_port) as page:
            page.call("Page.navigate", url=server.url + "?preset=map&time=119")
            until(page, EXPLORED)
            _set(page, "wind-lines", "psi")
            until(page, f"{LAYOUT}?.mode === 'psi' && " + EXPLORED)
            assert page.eval(f"{LAYOUT}.closed") > 0
            _set(page, "wind-lines", "streamlines")
            until(page, f"{LAYOUT}?.mode === 'streamlines' && {LAYOUT}.lines > 50")
            _set(page, "wind-arrows", True)
            until(page, f"{LAYOUT}?.quiver > 50")
            explore(page, "preset", "time_mean_map")
            until(page, "viewer.explorer.data?.roles?.time === 'reduce' && " + EXPLORED + f" && {LAYOUT}?.quiver > 50")
            assert page.eval("viewer.explorer.data.wind.psi.time_mean") is True
            _set(page, "wind-lines", "auto")
            until(page, f"{LAYOUT}?.mode !== null && " + EXPLORED)
            explore(page, "preset", "map")
            until(page, EXPLORED)
            page.eval("""(() => {const s = document.querySelector('[aria-label="Vertical representation"]')
                || [...document.querySelectorAll('[data-e=coords] select')].find(e => [...e.options].some(o => o.value === 'pressure'));
                s.value = 'pressure'; s.dispatchEvent(new Event('change', {bubbles: true}));})()""")
            until(page, "viewer.explorer.data?.coordinates?.vertical?.representation === 'pressure' && " + EXPLORED)
            assert page.eval(f"{LAYOUT}.mode") == "streamlines"
            assert page.eval("viewer.explorer.data.wind.psi.available") is False
            assert not page.exceptions, page.exceptions
    finally:
        server.shutdown()
        thread.join()


def test_wind_layer_is_refused_on_a_section(browser):
    server, thread = serve([A])
    try:
        with OfflinePage(browser, width=1400, height=1800, loopback_port=server.server_port) as page:
            page.call("Page.navigate", url=server.url + "?preset=map")
            until(page, EXPLORED)
            _set(page, "wind-lines", "auto")
            until(page, "viewer.explorer.data?.companions?.wind_u && " + EXPLORED)
            explore(page, "preset", "zonal_mean")
            until(page, "document.querySelector('[data-e=wind-lines]').disabled === true && " + EXPLORED)
            reason = page.eval("document.querySelector('[data-e=wind-reason]').textContent")
            assert "longitude × latitude" in reason
            assert "wind" not in json.loads(page.eval("viewer.explorer.loaded"))["options"]
            assert not page.exceptions, page.exceptions
    finally:
        server.shutdown()
        thread.join()
