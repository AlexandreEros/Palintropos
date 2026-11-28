"""The explorer through the actual viewer page in network-isolated Chromium.

Ported from the Vertical-structure panel's tests: the same archives and the
same scientific assertions, now made through the four-coordinate explorer.
The Explorer is the page's analytical workspace and owns the selection; the
globe (selection A) draws its quantity at its saved time and fixed surface.
"""
import json
import threading
from pathlib import Path

import numpy as np
import pytest

from tools.browser_cdp import OfflinePage, find_browser
from tropoi.representation.visual.viewer_server import ViewerServer, ViewerSession
from test_viewer_local_browser import set_control, until

ROOT = Path(__file__).resolve().parents[1]
STEM = "runs/dev-thai-ben2-ck-heated-T21L40-f44c039-layer-linear-dt600-"
A, B30, BEXT = (ROOT / (STEM + s) for s in ("30d", "spongeB-120d", "spongeB-120d-from-d030"))
EXPLORED = ('document.querySelector(\'[data-e="png"]\')?.disabled === false && '
            'document.querySelector(\'[data-e="status"]\')?.textContent === ""')
READY = 'document.querySelector(\'[data-id="png"]\')?.disabled === false && ' + EXPLORED
LOADED = "JSON.parse(viewer.explorer.loaded || 'null')"


def explore(page, key, value):
    selector = json.dumps(f'[data-e="{key}"]')
    page.eval(f"(() => {{const el=document.querySelector({selector}); el.value={json.dumps(str(value))};"
              "el.dispatchEvent(new Event('change',{bubbles:true}));})()")


def click_field(page, lat, lon):
    """Dispatch a real click on the Explorer's longitude × latitude field at a geographic position."""
    page.eval(f"""(() => {{const s = viewer.explorer, L = s.layout, r = s.c.plot.getBoundingClientRect();
        const x = L.X({lon}), y = L.Y({lat});
        s.c.plot.dispatchEvent(new MouseEvent('click', {{clientX: r.left + x * r.width / L.width,
            clientY: r.top + y * r.height / L.height, bubbles: true}}));}})()""")


def explorer_level(page, k):
    """Choose a native level as the Explorer's fixed surface (the globe follows)."""
    page.eval(f"""(() => {{const el = document.querySelector('[aria-label="Native level"]'); el.value = "{k}";
        el.dispatchEvent(new Event('change', {{bubbles: true}}));}})()""")


@pytest.fixture
def browser():
    found = find_browser()
    if not found:
        pytest.skip("Needs a Chromium browser")
    return found


def serve(paths):
    for path in paths:
        if not Path(path).is_dir():
            pytest.skip(f"Missing archive: {path}")
    server = ViewerServer(ViewerSession(paths))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def test_day30_nightside_surface_temperature(browser):
    server, thread = serve([A])
    try:
        with OfflinePage(browser, width=1400, height=1800, loopback_port=server.server_port) as page:
            page.call("Page.navigate", url=server.url + "?time=29&preset=map")
            until(page, READY)
            # The slab is a surface quantity: the map roles apply unchanged (no vertical coordinate).
            explore(page, "quantity", "surface_temperature")
            until(page, f"{LOADED}?.quantity === 'surface_temperature' && " + READY)
            assert page.eval("viewer.panels[0].key") == "surface_temperature"      # the globe follows
            # The cursor starts at the substellar point; a click on the field moves it to the antistellar point.
            assert page.eval("viewer.cursor.lon") == 0
            click_field(page, 0, 180)                    # the field cell centred on 180°E
            until(page, "Math.abs(viewer.cursor.lat) < 2 && Math.abs(viewer.cursor.lon - 180) < 2 && " + READY)
            # The automatic range spans the dayside; a fixed range resolves the nightside (the globe's range).
            set_control(page, "range-mode", "fixed")
            low, high = (float(page.eval(f"document.querySelector('[data-id=range-{k}]').value")) for k in ("min", "max"))
            assert high - low > 100 and low > 240
            set_control(page, "range-min", 241)
            set_control(page, "range-max", 264)
            until(page, "viewer.panels[0].view.range[0] === 241 && viewer.panels[0].view.range[1] === 264")
            set_control(page, "light", "star", "input")
            until(page, "document.querySelector('.surface-light').textContent.includes('keep model values')")
            # Back to temperature: the profile at the cursor carries the slab marker.
            explore(page, "quantity", "temperature")
            explore(page, "preset", "profile")
            until(page, f"{LOADED}?.vertical.role === 'y' && " + READY)
            data = page.eval("viewer.explorer.data")
            slab = np.load(A / "states/surface-d00030.npy").reshape(32, 64)
            lat, lon = data["coordinates"]["latitude"]["value"], data["coordinates"]["longitude"]["value"]
            i = int(np.abs(np.degrees(np.arcsin(np.polynomial.legendre.leggauss(32)[0][::-1])) - lat).argmin())
            surface = [m for m in data["markers"] if m["kind"] == "surface"][0]
            assert surface["value"] == slab[i, round(lon / 5.625) % 64]
            notes = page.eval("document.querySelector('[data-e=notes]').innerText")
            assert "not a surface temperature" in notes and "Slab surface temperature" in notes
            assert not page.exceptions, page.exceptions
    finally:
        server.shutdown()
        thread.join()


def test_cold_event_column_section_and_level_link(browser):
    server, thread = serve([BEXT])
    try:
        with OfflinePage(browser, width=1400, height=1800, loopback_port=server.server_port) as page:
            page.call("Page.navigate", url=server.url + "?time=54&lat=-19.4&lon=281&preset=profile")
            until(page, READY)
            assert page.eval("document.querySelector('[data-id=time-label]').textContent").startswith("Day 85 ")
            row = page.eval("document.querySelector('[data-e=table]').tBodies[0].rows[1].innerText").split("\t")
            assert row[0] == "2" and abs(float(row[5]) - 123.28) < 0.01 and 5 < float(row[2]) < 5.3
            # Clicking the level in the column makes it the fixed surface: the globe draws it.
            page.eval("document.querySelector('[data-e=table]').tBodies[0].rows[1].click()")
            until(page, "document.querySelector('[data-id=level]').value === '1' && " + READY)
            assert page.eval("viewer.panels[0].view.range[0]") > 122.0
            for preset in ("meridional_section", "zonal_section", "zonal_mean"):
                explore(page, "preset", preset)
                until(page, f"{LOADED}?.latitude && " + EXPLORED)
            explore(page, "quantity", "mass_streamfunction")
            until(page, "viewer.explorer.data?.quantity === 'mass_streamfunction' && " + EXPLORED)
            # A click in the streamfunction section moves the cursor latitude only.
            page.eval("""(() => {const s = viewer.explorer, L = s.layout, r = s.c.plot.getBoundingClientRect();
                const x = L.X(30), y = L.box.y + L.box.h / 2;
                s.c.plot.dispatchEvent(new MouseEvent('click', {clientX: r.left + x * r.width / L.width,
                    clientY: r.top + y * r.height / L.height, bubbles: true}));})()""")
            until(page, "Math.abs(viewer.cursor.lat - 30) < 6 && Math.abs(viewer.cursor.lon - 281) < 1e-9")
            assert not page.exceptions, page.exceptions
    finally:
        server.shutdown()
        thread.join()


def test_equatorial_jets_days_30_to_120_across_two_directories(browser):
    server, thread = serve([B30, BEXT])
    ext = json.loads((ROOT / "runs/diagnostics/sponge/results-ext/ext.json").read_text())
    try:
        with OfflinePage(browser, width=1400, height=1800, loopback_port=server.server_port) as page:
            page.call("Page.navigate", url=server.url + "?time=29&lat=0&lon=0&preset=zonal_mean")
            until(page, READY)
            peaks = {}
            for run, days in ((0, (30,)), (1, (60, 90, 120))):
                if run:
                    page.eval('(() => {const el=document.getElementById("runs"); el.selectedIndex=1;'
                              'el.dispatchEvent(new Event("change"));})()')
                    until(page, 'location.pathname.endsWith("/1/") && ' + READY)
                for day in days:
                    index = day - (1 if run == 0 else 31)
                    set_control(page, "time", index, "input")
                    until(page, f"{LOADED}?.time.index === {index} && " + EXPLORED)
                    explore(page, "quantity", "u")
                    until(page, f"{LOADED}?.quantity === 'u' && {LOADED}?.longitude.role === 'reduce' && " + EXPLORED)
                    title = page.eval("viewer.explorer.spec.title.join(' | ')")
                    assert f"Day {day} " in title and "config 8ad59ee8124c" in title
                    assert ("from-d030" in title) == bool(run)
                    data = page.eval("viewer.explorer.data")
                    u, lat = np.array(data["values"]), np.array(data["x"]["values"])
                    native = u[:, np.abs(lat) < 3].mean(axis=1)
                    reference = np.array(ext["daily"]["eq_u"][str(day)])
                    peaks[day] = native.max()
                    assert np.argmax(native) == np.argmax(reference)
                    assert abs(native.max() - reference.max()) < 3.0
            assert not page.exceptions, page.exceptions
            assert peaks[120] > peaks[60]
    finally:
        server.shutdown()
        thread.join()


def test_phone_width_has_no_horizontal_scroll(browser):
    server, thread = serve([A])
    try:
        with OfflinePage(browser, width=390, height=900, loopback_port=server.server_port) as page:
            page.call("Page.navigate", url=server.url + "?time=29&preset=profile")
            until(page, READY)
            assert page.eval("document.documentElement.scrollWidth <= window.innerWidth")
    finally:
        server.shutdown()
        thread.join()


def section_state(page):
    return page.eval("""(() => {const s = viewer.explorer, c = s.c;
        return {mode: c["range-mode"].value, min: c["range-min"].value, max: c["range-max"].value,
                hidden: c["range-fieldset"].hidden, inputs: c["range-wrap"].hidden ? null : true,
                drawn: s.layout?.range ?? null, spec: s.spec?.range ?? null,
                time: JSON.parse(s.loaded || "null")?.time.index, title: s.spec?.title.join(" | ")};})()""")


def export_top_colour_rows(page):
    """Export a PNG and count rows of its centre column drawn in the palette's top colour."""
    return page.eval("""(async () => {
        const lut = (await import("./mesh.js")).fieldScale({values: [0]}, {range: [0, 1]}).lut;
        const blob = HTMLCanvasElement.prototype.toBlob;
        let canvas, name;
        HTMLCanvasElement.prototype.toBlob = function (callback, ...rest) { canvas = this; return blob.call(this, callback, ...rest); };
        const click = HTMLAnchorElement.prototype.click;
        HTMLAnchorElement.prototype.click = function () { name = this.download; };
        try {
          viewer.explorer.c.png.click();
          for (let i = 0; i < 100 && !name; i++) await new Promise(r => setTimeout(r, 50));
        } finally {
          HTMLCanvasElement.prototype.toBlob = blob;
          HTMLAnchorElement.prototype.click = click;
        }
        const x = Math.round(canvas.width / 2), pixels = canvas.getContext("2d").getImageData(x, 0, 1, canvas.height).data;
        const [r, g, b] = lut[255];
        let rows = 0;
        for (let y = 0; y < canvas.height; y++) if (pixels[4*y] === r && pixels[4*y+1] === g && pixels[4*y+2] === b) rows++;
        return {rows, name, width: canvas.width};
    })()""")


def test_section_fixed_colour_range_persists_by_quantity_and_exports(browser):
    server, thread = serve([A])
    try:
        with OfflinePage(browser, width=1400, height=1800, loopback_port=server.server_port) as page:
            page.call("Page.navigate", url=server.url + "?time=29&lat=-40&lon=90&preset=meridional_section")
            until(page, READY)
            explore(page, "quantity", "temperature")
            until(page, READY)
            state = section_state(page)
            assert state["mode"] == "auto" and not state["hidden"] and state["spec"] is None
            auto = state["drawn"]
            map_range = page.eval("viewer.panels[0].view.range")
            # Choosing a fixed range starts from the automatic one.
            explore(page, "range-mode", "fixed")
            state = section_state(page)
            assert state["inputs"] and [float(state["min"]), float(state["max"])] == [
                float(f"{auto[0]:.6g}"), float(f"{auto[1]:.6g}")]
            explore(page, "range-min", 150)
            explore(page, "range-max", 300)
            state = section_state(page)
            assert state["drawn"] == [150, 300] and state["spec"] == [150, 300]
            notes = page.eval("document.querySelector('[data-e=notes]').innerText")
            assert "Fixed colour range 150 to 300" in notes
            # The map keeps its own automatic range.
            assert page.eval("document.querySelector('[data-id=range-mode]').value") == "auto"
            assert page.eval("document.querySelector('[data-id=range-wrap]').hidden")
            assert page.eval("viewer.panels[0].view.range") == map_range
            # Another saved time keeps the fixed range for the same quantity.
            set_control(page, "time", 20, "input")
            until(page, f"{LOADED}?.time.index === 20 && " + READY)
            state = section_state(page)
            assert "Day 21 " in state["title"]
            assert (state["mode"], state["min"], state["max"]) == ("fixed", "150", "300")
            assert state["drawn"] == [150, 300]
            # Other quantities have their own setting; returning restores this one.
            explore(page, "quantity", "u")
            until(page, f"{LOADED}?.quantity === 'u' && " + READY)
            state = section_state(page)
            assert state["mode"] == "auto" and state["spec"] is None and state["inputs"] is None
            assert state["drawn"][0] == -state["drawn"][1]
            explore(page, "quantity", "temperature")
            until(page, f"{LOADED}?.quantity === 'temperature' && " + READY)
            assert section_state(page)["drawn"] == [150, 300]
            # The same quantity keeps its range in the zonal section; ψ has its own; columns have none.
            explore(page, "preset", "zonal_section")
            until(page, f"{LOADED}?.longitude.role === 'x' && " + READY)
            assert section_state(page)["drawn"] == [150, 300]
            explore(page, "preset", "zonal_mean")
            until(page, f"{LOADED}?.longitude.role === 'reduce' && " + READY)
            explore(page, "quantity", "mass_streamfunction")
            until(page, f"{LOADED}?.quantity === 'mass_streamfunction' && " + EXPLORED)
            assert section_state(page)["spec"] is None
            # ψ has no longitude × latitude field: the globe says so instead of drawing something else.
            assert "has no longitude × latitude field" in page.eval("viewer.panels[0].controls.status.textContent")
            explore(page, "quantity", "temperature")
            until(page, f"{LOADED}?.quantity === 'temperature' && " + READY)
            explore(page, "preset", "profile")
            until(page, f"{LOADED}?.vertical.role === 'y' && {LOADED}?.latitude.role === 'fixed' && " + READY)
            assert section_state(page)["hidden"]
            explore(page, "preset", "meridional_section")
            until(page, f"{LOADED}?.latitude.role === 'x' && " + READY)
            # An invalid range falls back to automatic and is flagged.
            explore(page, "range-max", 100)
            assert section_state(page)["spec"] is None
            assert not page.eval("document.querySelector('[data-e=range-max]').checkValidity()")
            # A fixed map range leaves the section unchanged.
            set_control(page, "range-mode", "fixed")
            set_control(page, "range-min", 200)
            set_control(page, "range-max", 210)
            until(page, "viewer.panels[0].view.range[0] === 200")
            assert section_state(page)["spec"] is None
            # PNG export uses the section range: [0, 1] K puts every cell at the top colour.
            explore(page, "range-min", 0)
            explore(page, "range-max", 1)
            fixed = export_top_colour_rows(page)
            explore(page, "range-mode", "auto")
            automatic = export_top_colour_rows(page)
            assert fixed["width"] == automatic["width"] == 1650
            assert "Latitude" in fixed["name"] and fixed["name"].endswith("-temperature-t20.png")
            assert fixed["rows"] > 500 > 100 > automatic["rows"]
            assert not page.exceptions, page.exceptions
    finally:
        server.shutdown()
        thread.join()


def test_pending_request_never_draws_stale_data(browser):
    """Changing the quantity while a zonal mean is in flight must not draw an old result as the new one."""
    from test_viewer_fields import MARS
    server, thread = serve([MARS])
    try:
        with OfflinePage(browser, width=1400, height=1800, loopback_port=server.server_port) as page:
            page.call("Page.navigate", url=server.url + "?time=8&preset=profile")
            until(page, READY)
            assert page.eval("viewer.explorer.spec.kind") == "line"
            # Hold zonal-mean responses (the longitude is reduced) until released.
            page.eval("""(() => {const fetch0 = window.fetch; window.held = [];
                window.fetch = (url, options) => decodeURIComponent(String(url)).includes('"reduce"') ?
                  new Promise(resolve => held.push(() => resolve(fetch0(url, options).catch(e => {throw e;})))) :
                  fetch0(url, options);})()""")
            explore(page, "preset", "zonal_mean")
            until(page, "held.length === 1")
            explore(page, "quantity", "u")
            until(page, "held.length === 2")
            explore(page, "range-mode", "fixed")
            assert page.eval("viewer.explorer.spec.kind") == "line"
            assert page.eval("viewer.explorer.c.png.disabled")
            assert not page.exceptions, page.exceptions
            page.eval("held.shift()()")                 # the superseded temperature mean
            page.eval("new Promise(resolve => setTimeout(resolve, 300))")
            assert page.eval("viewer.explorer.spec.kind") == "line"
            page.eval("held.shift()()")
            until(page, "viewer.explorer.spec.kind === 'field' && " + READY)
            spec = page.eval("({quantity: viewer.explorer.data.quantity, label: viewer.explorer.spec.label,"
                             " lat: viewer.explorer.data.x.values.length})")
            assert spec["quantity"] == "u" and spec["label"].startswith("Eastward wind") and spec["lat"] == 34
            # Returning to the loaded column while a new mean is pending shows the column at once.
            explore(page, "preset", "profile")
            until(page, READY + " && viewer.explorer.spec.kind === 'line'")
            # A mean requested and abandoned before it arrives never replaces the column on hand.
            explore(page, "preset", "zonal_mean")
            until(page, "held.length === 1")
            explore(page, "preset", "profile")
            until(page, READY + " && viewer.explorer.spec.kind === 'line'")
            page.eval("held.shift()()")
            page.eval("new Promise(resolve => setTimeout(resolve, 400))")
            assert page.eval("viewer.explorer.spec.kind") == "line"
            assert page.eval(f"{LOADED}.longitude.role") == "fixed"
            set_control(page, "time", 7, "input")
            until(page, f"{LOADED}?.time.index === 7 && " + READY)
            assert page.eval("viewer.explorer.spec.kind") == "line"
            assert not page.exceptions, page.exceptions
    finally:
        server.shutdown()
        thread.join()


def test_flat_gravity_free_and_geodesic_capsules_open(browser, tmp_path):
    from test_viewer_fields import mars_variant
    flat = mars_variant(tmp_path / "flat", flat=True)
    geodesic = mars_variant(tmp_path / "geodesic", geodesic=True)
    server, thread = serve([flat, geodesic])
    try:
        with OfflinePage(browser, width=1400, height=1800, loopback_port=server.server_port) as page:
            page.call("Page.navigate", url=server.url + "?time=8")
            until(page, 'document.querySelector(\'[data-id="png"]\')?.disabled === false')
            assert page.eval("document.querySelector('[data-id=field]').value") == "temperature"
            # Without gravity there is no native-level column evaluator: the explorer is offered,
            # and its vertical coordinate can be fixed but not an axis (disabled, with the reason).
            assert not page.eval("document.getElementById('explorer').hidden")
            vertical = page.eval("""[...document.querySelector('[aria-label="Vertical role"]').options]
                .map(o => [o.value, o.disabled])""")
            assert ["y", True] in vertical and ["fixed", False] in vertical
            coordinates = page.eval("[...document.querySelector('[data-id=coordinate]').options].map(o => o.value)")
            assert coordinates == ["native"]
            assert page.eval("document.querySelector('[data-id=opacity-wrap]').hidden")
            explore(page, "preset", "map")
            explorer_level(page, 0)
            until(page, "document.querySelector('[data-id=level]').value === '0' && " + READY)
            assert not page.exceptions, page.exceptions
            page.eval('(() => {const el=document.getElementById("runs"); el.selectedIndex=1;'
                      'el.dispatchEvent(new Event("change"));})()')
            until(page, 'location.pathname.endsWith("/1/") && ' + READY)
            # The configuration travelled with the archive: the map roles and level are kept.
            assert page.eval(f"{LOADED}.longitude.role") == "x" and page.eval(f"{LOADED}.vertical.value") == 0
            explore(page, "preset", "profile")
            until(page, f"{LOADED}?.latitude.role === 'fixed' && " + READY)
            assert page.eval("document.querySelector('[data-e=snap-wrap]').hidden")
            assert page.eval(f"{LOADED}.options.snap") is False
            assert (page.eval("viewer.explorer.data.coordinates.latitude.sampling") ==
                    "exact band-limited synthesis at the requested point")
            # A geodesic grid has no Gaussian quadrature: every preset with a horizontal mean is
            # unavailable, and only those.
            presets = page.eval("[...viewer.explorer.c.preset.options].filter(o => o.value && !o.disabled).map(o => o.value)")
            assert presets == ["map", "time_mean_map", "meridional_section", "zonal_section", "profile",
                               "time_mean_profile", "time_series", "time_section", "hovmoller"]
            reduce = page.eval("""[...document.querySelector('[aria-label="Longitude role"]').options]
                .find(o => o.value === 'reduce')""" + "?.title")
            assert "quadrature" in reduce
            assert not page.exceptions, page.exceptions
    finally:
        server.shutdown()
        thread.join()


def test_airless_archive_in_the_same_application(browser, tmp_path):
    """A columnar archive opens in the same page: a skin map, a column with the skin apart, time × depth."""
    from test_viewer_airless import _run
    root = _run(tmp_path / "latlon", {"kind": "latlon", "nlat": 6, "nlon": 12}, 30)
    server, thread = serve([root])
    try:
        with OfflinePage(browser, width=1400, height=1800, loopback_port=server.server_port) as page:
            page.call("Page.navigate", url=server.url + "?lat=10&lon=40")
            until(page, READY)
            assert page.eval("viewer.panels[0].key") == "temperature"
            assert "diagnosed skin" in page.eval("document.querySelector('[data-id=level]').selectedOptions[0].text")
            assert "Native cells" in page.eval("document.querySelector('[data-id=status]').textContent")
            data = page.eval("viewer.explorer.data")
            assert data["axes"] == dict(x=None, y="vertical")
            assert data["y"]["representation"] in ("depth", "height_above_surface")
            assert [m["kind"] for m in data["markers"]] == ["surface"]
            explore(page, "preset", "time_section")
            until(page, f"{LOADED}?.time.role === 'x' && " + EXPLORED)
            assert page.eval("viewer.explorer.spec.kind") == "field"
            assert page.eval("viewer.explorer.layout.strip") > 0           # the skin strip, apart from the cells
            # Longitude × latitude is an ordinary two-dimensional result of the explorer; the page
            # settles after its requests (the field, and the globe's frame of the same selection).
            page.eval("""(() => {const f0 = window.fetch; window.selects = 0;
                window.fetch = (u, o) => { if (String(u).includes("select")) selects++; return f0(u, o); };})()""")
            explore(page, "preset", "map")
            until(page, f"{LOADED}?.longitude.role === 'x' && {LOADED}?.latitude.role === 'y' && " + EXPLORED)
            assert page.eval("viewer.explorer.spec.kind") == "field"
            page.eval("new Promise(resolve => setTimeout(resolve, 1500))")
            assert page.eval("selects") <= 3
            # The same roles with a time mean: the mean map, in the same view.
            explore(page, "preset", "time_mean_map")
            until(page, f"{LOADED}?.time.role === 'reduce' && " + EXPLORED)
            assert page.eval("viewer.explorer.data.description").endswith("time mean")
            # A level chosen in the Explorer stays chosen, and the globe draws it.
            explorer_level(page, 3)
            until(page, READY.split(" && ")[0])
            page.eval("new Promise(resolve => setTimeout(resolve, 800))")
            assert page.eval("document.querySelector('[data-id=level]').value") == "3"
            # The archive records column area weights: longitude means are offered as for any field.
            explore(page, "preset", "zonal_mean")
            until(page, f"{LOADED}?.longitude.role === 'reduce' && " + EXPLORED)
            assert page.eval("viewer.explorer.spec.kind") == "field"
            # A vertical mean is not defined: the role is disabled and says why.
            reason = page.eval("""[...document.querySelector('[aria-label="Vertical role"]').options]
                .find(o => o.value === 'reduce').title""")
            assert "thickness weighting" in reason
            assert not page.exceptions, page.exceptions
    finally:
        server.shutdown()
        thread.join()
