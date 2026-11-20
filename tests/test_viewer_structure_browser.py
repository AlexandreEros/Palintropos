"""Ben 2 investigations through the actual Viewer 2 page in network-isolated Chromium."""
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
READY = ('document.querySelector(\'[data-id="png"]\')?.disabled === false && '
         'document.querySelector(\'[data-s="png"]\')?.disabled === false && '
         'document.querySelector(\'[data-s="status"]\')?.textContent === ""')


def structure(page, key, value):
    selector = json.dumps(f'[data-s="{key}"]')
    page.eval(f"(() => {{const el=document.querySelector({selector}); el.value={json.dumps(str(value))};"
              "el.dispatchEvent(new Event('change',{bubbles:true}));})()")


def click_map(page, lat, lon):
    """Dispatch a real click on the 2D map at a geographic position."""
    page.eval(f"""(() => {{const view = viewer.panels[0].view, L = view.layout, r = view.map.getBoundingClientRect();
        const x = L.box.x + L.box.w * ({lon} + 180) / 360, y = L.box.y + L.box.h * (90 - {lat}) / 180;
        view.map.dispatchEvent(new MouseEvent('click', {{clientX: r.left + x * r.width / L.width,
            clientY: r.top + y * r.height / L.height, bubbles: true}}));}})()""")


@pytest.fixture
def browser():
    found = find_browser()
    if not found:
        pytest.skip("Needs a Chromium browser")
    return found


def serve(paths):
    for path in paths:
        if not path.is_dir():
            pytest.skip(f"Missing archive: {path}")
    server = ViewerServer(ViewerSession(paths))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def test_day30_nightside_surface_temperature(browser):
    server, thread = serve([A])
    try:
        with OfflinePage(browser, width=1400, height=1800, loopback_port=server.server_port) as page:
            page.call("Page.navigate", url=server.url + "?time=29&product=column")
            until(page, READY)
            set_control(page, "field", "surface_temperature")
            until(page, READY)
            # The cursor starts at the substellar point; a map click moves it to the antistellar point.
            assert page.eval("viewer.cursor.lon") == 0
            click_map(page, 0, 179)
            until(page, "Math.abs(viewer.cursor.lat) < 2 && Math.abs(viewer.cursor.lon - 180) < 2 && " + READY)
            data = page.eval("viewer.structure.data")
            slab = np.load(A / "states/surface-d00030.npy").reshape(32, 64)
            i = int(np.abs(np.degrees(np.arcsin(np.polynomial.legendre.leggauss(32)[0][::-1])) - data["latitude"]).argmin())
            assert data["surface_temperature"]["value"] == slab[i, round(data["longitude"] / 5.625) % 64]
            notes = page.eval("document.querySelector('[data-s=notes]').innerText")
            assert "is not a surface temperature" in notes and "Slab surface temperature" in notes
            # The automatic range spans the dayside; a fixed range resolves the nightside.
            set_control(page, "range-mode", "fixed")
            low, high = (float(page.eval(f"document.querySelector('[data-id=range-{k}]').value")) for k in ("min", "max"))
            assert high - low > 100 and low > 240
            set_control(page, "range-min", 241)
            set_control(page, "range-max", 264)
            until(page, "viewer.panels[0].view.range[0] === 241 && viewer.panels[0].view.range[1] === 264")
            set_control(page, "light", "star", "input")
            until(page, "document.querySelector('.surface-light').textContent.includes('keep model values')")
            assert not page.exceptions, page.exceptions
    finally:
        server.shutdown()
        thread.join()


def test_cold_event_column_section_and_level_link(browser):
    server, thread = serve([BEXT])
    try:
        with OfflinePage(browser, width=1400, height=1800, loopback_port=server.server_port) as page:
            page.call("Page.navigate", url=server.url + "?time=54&lat=-19.4&lon=281&product=column")
            until(page, READY)
            assert page.eval("document.querySelector('[data-id=time-label]').textContent").startswith("Day 85 ")
            row = page.eval("document.querySelector('[data-s=table]').tBodies[0].rows[1].innerText").split("\t")
            assert row[0] == "2" and abs(float(row[4]) - 123.28) < 0.01 and 5 < float(row[2]) < 5.3
            # Clicking the level in the column selects it on the map.
            page.eval("document.querySelector('[data-s=table]').tBodies[0].rows[1].click()")
            until(page, "document.querySelector('[data-id=level]').value === '1' && " + READY)
            assert page.eval("viewer.panels[0].view.range[0]") > 122.0
            for product in ("meridional", "zonal", "zonal_mean", "streamfunction"):
                structure(page, "product", product)
                until(page, READY)
            # A click in the streamfunction section moves the cursor latitude only.
            page.eval("""(() => {const s = viewer.structure, L = s.layout, r = s.c.plot.getBoundingClientRect();
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
            page.call("Page.navigate", url=server.url + "?time=29&lat=0&lon=0&product=zonal_mean")
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
                    until(page, f"viewer.structure.query?.first === {index} && " + READY)
                    structure(page, "field", "u")
                    until(page, READY)
                    title = page.eval("viewer.structure.spec.title.join(' | ')")
                    assert f"Day {day} " in title and "config 8ad59ee8124c" in title
                    assert ("from-d030" in title) == bool(run)
                    data = page.eval("viewer.structure.data")
                    u, lat = np.array(data["u"]), np.array(data["lat"])
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
            page.call("Page.navigate", url=server.url + "?time=29&product=column")
            until(page, READY)
            assert page.eval("document.documentElement.scrollWidth <= window.innerWidth")
    finally:
        server.shutdown()
        thread.join()


def section_state(page):
    return page.eval("""(() => {const s = viewer.structure, c = s.c;
        return {mode: c["range-mode"].value, min: c["range-min"].value, max: c["range-max"].value,
                hidden: c["range-fieldset"].hidden, inputs: c["range-wrap"].hidden ? null : true,
                drawn: s.layout?.range ?? null, spec: s.spec?.range ?? null, time: s.query?.time,
                title: s.spec?.title.join(" | ")};})()""")


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
          viewer.structure.c.png.click();
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
            page.call("Page.navigate", url=server.url + "?time=29&lat=-40&lon=90&product=meridional")
            until(page, READY)
            structure(page, "field", "temperature")
            until(page, READY)
            state = section_state(page)
            assert state["mode"] == "auto" and not state["hidden"] and state["spec"] is None
            auto = state["drawn"]
            map_range = page.eval("viewer.panels[0].view.range")
            # Choosing a fixed range starts from the automatic one.
            structure(page, "range-mode", "fixed")
            state = section_state(page)
            assert state["inputs"] and [float(state["min"]), float(state["max"])] == [
                float(f"{auto[0]:.6g}"), float(f"{auto[1]:.6g}")]
            structure(page, "range-min", 150)
            structure(page, "range-max", 300)
            state = section_state(page)
            assert state["drawn"] == [150, 300] and state["spec"] == [150, 300]
            notes = page.eval("document.querySelector('[data-s=notes]').innerText")
            assert "Fixed colour range 150 to 300" in notes
            # The map keeps its own automatic range.
            assert page.eval("document.querySelector('[data-id=range-mode]').value") == "auto"
            assert page.eval("document.querySelector('[data-id=range-wrap]').hidden")
            assert page.eval("viewer.panels[0].view.range") == map_range
            # Another saved time keeps the fixed range for the same quantity.
            set_control(page, "time", 20, "input")
            until(page, "viewer.structure.query?.time === 20 && " + READY)
            state = section_state(page)
            assert "Day 21 " in state["title"]
            assert (state["mode"], state["min"], state["max"]) == ("fixed", "150", "300")
            assert state["drawn"] == [150, 300]
            # Other quantities have their own setting; returning restores this one.
            structure(page, "field", "u")
            until(page, READY)
            state = section_state(page)
            assert state["mode"] == "auto" and state["spec"] is None and state["inputs"] is None
            assert state["drawn"][0] == -state["drawn"][1]
            structure(page, "field", "temperature")
            until(page, READY)
            assert section_state(page)["drawn"] == [150, 300]
            # The same quantity keeps its range in the zonal section; ψ has its own; columns have none.
            structure(page, "product", "zonal")
            until(page, READY)
            assert section_state(page)["drawn"] == [150, 300]
            structure(page, "product", "streamfunction")
            until(page, READY)
            assert section_state(page)["spec"] is None
            structure(page, "product", "column")
            until(page, READY)
            assert section_state(page)["hidden"]
            structure(page, "product", "meridional")
            until(page, READY)
            # An invalid range falls back to automatic and is flagged.
            structure(page, "range-max", 100)
            assert section_state(page)["spec"] is None
            assert not page.eval("document.querySelector('[data-s=range-max]').checkValidity()")
            # A fixed map range leaves the section unchanged.
            set_control(page, "range-mode", "fixed")
            set_control(page, "range-min", 200)
            set_control(page, "range-max", 210)
            until(page, "viewer.panels[0].view.range[0] === 200")
            assert section_state(page)["spec"] is None
            # PNG export uses the section range: [0, 1] K puts every cell at the top colour.
            structure(page, "range-min", 0)
            structure(page, "range-max", 1)
            fixed = export_top_colour_rows(page)
            structure(page, "range-mode", "auto")
            automatic = export_top_colour_rows(page)
            assert fixed["width"] == automatic["width"] == 1650
            assert fixed["name"].endswith("-meridional-temperature-t20.png")
            assert fixed["rows"] > 500 > 100 > automatic["rows"]
            assert not page.exceptions, page.exceptions
    finally:
        server.shutdown()
        thread.join()
