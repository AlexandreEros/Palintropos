"""Record the Explorer's horizontal-wind representations as PNG evidence.

Drives the real viewer page in headless Chrome (tools.browser_cdp), sets each
case through the page's own controls, and saves the Explorer's plot canvas
(the same pixels as its Save PNG) with a JSON record of the archive, saved
time, vertical surface, wind settings, what was drawn and the payload's
kinetic-energy split.

    python -m tools.viewer.wind_evidence OUT_DIR
"""
from __future__ import annotations

import base64
import json
import sys
import threading
import time
from pathlib import Path

from tools.browser_cdp import OfflinePage, find_browser
from tropoi.representation.visual.viewer_server import ViewerServer, ViewerSession

ROOT = Path(__file__).resolve().parents[2]
W5 = ROOT / "docs/runs/20260730T011700Z_williamson5_rot23p93h_r4_l63_dt120h_45406d82_668e6c9a"
HS = ROOT / "runs/hs-T42L20-prod-002"
EXPLORED = ('document.querySelector(\'[data-e="png"]\')?.disabled === false && '
            'document.querySelector(\'[data-e="status"]\')?.textContent === "" && !viewer.explorer.pending')

# (name, archive, URL query, steps). A step is ("e", data-e, value), ("level", k) or ("pressure", Pa).
CASES = [
    ("01-w5-day15-auto-psi-contours", W5, "preset=map&time=3",
     [("e", "quantity", "free_surface_height"), ("e", "wind-lines", "auto")]),
    ("02-w5-day15-psi-contours-and-arrows", W5, "preset=map&time=3",
     [("e", "quantity", "free_surface_height"), ("e", "wind-lines", "psi"), ("e", "wind-arrows", True),
      ("e", "wind-spacing", 12), ("e", "wind-colour", True)]),
    ("03-w5-day15-full-wind-streamlines", W5, "preset=map&time=3",
     [("e", "quantity", "free_surface_height"), ("e", "wind-lines", "streamlines")]),
    ("04-w5-day15-arrows-only-equal-length", W5, "preset=map&time=3",
     [("e", "quantity", "free_surface_height"), ("e", "wind-arrows", True), ("e", "wind-length", "unit")]),
    ("05-hs-day1200-level5-auto", HS, "preset=map&time=119", [("level", 4), ("e", "wind-lines", "auto")]),
    ("06-hs-day1200-level5-psi-contours-forced", HS, "preset=map&time=119", [("level", 4), ("e", "wind-lines", "psi")]),
    ("07-hs-days1010-1200-mean-level5-psi-and-arrows", HS, "preset=time_mean_map",
     [("first", 100), ("last", 119), ("level", 4), ("e", "wind-lines", "auto"),
      ("e", "wind-arrows", True)]),
    ("08-hs-day1200-500hPa-auto-no-psi", HS, "preset=map&time=119",
     [("pressure", 50000), ("e", "wind-lines", "auto"), ("e", "wind-arrows", True)]),
]


def until(page, expression, timeout=120):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if page.eval(expression):
            return
        time.sleep(0.15)
    raise TimeoutError(expression)


def step(page, item):
    kind = item[0]
    if kind == "e":
        _, key, value = item
        page.eval(f"""(() => {{const el = document.querySelector('[data-e="{key}"]');
            if (el.type === 'checkbox') el.checked = {json.dumps(value)}; else el.value = {json.dumps(str(value))};
            el.dispatchEvent(new Event('change', {{bubbles: true}}));}})()""")
    elif kind == "level":
        page.eval(f"""(() => {{const el = document.querySelector('[aria-label="Native level"]'); el.value = "{item[1]}";
            el.dispatchEvent(new Event('change', {{bubbles: true}}));}})()""")
    elif kind in ("first", "last"):
        page.eval(f"""(() => {{const el = viewer.explorer.rows.time.{kind}; el.value = "{item[1]}";
            el.dispatchEvent(new Event('change', {{bubbles: true}}));}})()""")
    elif kind == "pressure":
        page.eval("""(() => {const s = [...document.querySelectorAll('[data-e=coords] select')]
            .find(e => [...e.options].some(o => o.value === 'pressure'));
            s.value = 'pressure'; s.dispatchEvent(new Event('change', {bubbles: true}));})()""")
        until(page, "viewer.explorer.data?.coordinates?.vertical?.representation === 'pressure' && " + EXPLORED)
        page.eval(f"""(() => {{const el = [...document.querySelectorAll('[data-e=coords] input[type=number]')]
            .find(e => !e.hidden && e.closest('.coord')?.textContent.includes('Vertical'));
            el.value = "{item[1]}"; el.dispatchEvent(new Event('change', {{bubbles: true}}));}})()""")
    until(page, EXPLORED)


def main(out_dir: str) -> int:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    browser = find_browser()
    if not browser:
        print("No Chromium browser found", file=sys.stderr)
        return 1
    records = []
    for name, archive, query, steps in CASES:
        server = ViewerServer(ViewerSession([archive]))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with OfflinePage(browser, width=1240, height=1600, loopback_port=server.server_port) as page:
                page.call("Page.navigate", url=server.url + "?" + query)
                until(page, EXPLORED)
                for item in steps:
                    step(page, item)
                until(page, "viewer.explorer.layout?.winds && " + EXPLORED)
                png = page.eval("viewer.explorer.c.plot.toDataURL('image/png')")
                (out / f"{name}.png").write_bytes(base64.b64decode(png.split(",", 1)[1]))
                record = page.eval("""(() => { const e = viewer.explorer, d = e.data;
                    return {description: d.description, quantity: d.quantity, coordinates: d.coordinates,
                            settings: e.windSettings(), drawn: e.layout.winds, wind: d.wind,
                            title: e.spec.title}; })()""")
                record.update(name=name, archive=str(archive.relative_to(ROOT)).replace("\\", "/"), query=query,
                              exceptions=page.exceptions)
                records.append(record)
                print(name, record["drawn"].get("mode"), record["drawn"].get("lines"), record["drawn"].get("quiver"))
        finally:
            server.shutdown()
            thread.join()
    (out / "cases.json").write_text(json.dumps(records, indent=1, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "wind-evidence"))
