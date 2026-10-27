"""Saved-state playback in the offline atmosphere page, checked in a browser.

Exports small synthetic PE capsules (four saved states at NONUNIFORM times,
and a one-state capsule), opens each standalone page from ``file://`` in a
headless Chromium with networking blocked (``browser_cdp``), and drives the
page's own controls: play, pause, previous/next, the time slider, the speed
selector and the loop checkbox. Checks:

* every played frame is byte-identical (layer positions, colours, streamline
  vertices) to the same saved state selected with the slider, and carries the
  stored simulation time in its label;
* playback stops on the last state unless looping, and loops when asked;
* manual scrubbing or stepping pauses playback and selects the state at once;
* the wall-clock rate follows the selector (frames per second);
* field, level, opacity, streamlines, exaggeration, camera and map source
  are preserved while time advances; colour limits stay fixed;
* the gap note reports the actual uneven spacing;
* a one-state export hides the controls; the terrain-only page is untouched;
* each page issues exactly one request (its own file) and logs no errors.

Skipped when no Chromium-family browser is installed.
"""
from __future__ import annotations

import json

import pytest

from browser_cdp import OfflinePage, find_browser
from test_atmosphere_view import _band_limited_terrain, write_pe_capsule

BROWSER = find_browser()
pytestmark = pytest.mark.skipif(BROWSER is None,
                                reason="needs Chrome, Chromium or Edge")

TIMES = [0.0, 21600.0, 32400.0, 86400.0]           # gaps 6 h, 3 h, 15 h
API = "window.__terrainGlobe.atmosphere"


def _export(tmp_path, name, frames, times):
    from tropoi.representation.archive import open_simulation

    root, _, _ = write_pe_capsule(tmp_path / name, frames=frames, times=times)
    return open_simulation(root).view(tmp_path / f"{name}.html",
                                      open_browser=False,
                                      resolution_deg=6.0, times="all")


def _text(page, element):
    return page.eval(f"document.getElementById({json.dumps(element)})"
                     ".textContent")


def _click(page, element):
    page.eval(f"document.getElementById({json.dumps(element)}).click()")


def _set_input(page, element, value, event="input"):
    page.eval(f"(() => {{ const e = document.getElementById("
              f"{json.dumps(element)}); e.value = {json.dumps(str(value))};"
              f" e.dispatchEvent(new Event({json.dumps(event)})); }})()")


def _info(page):
    return page.eval(f"{API}.info()")


def _assert_clean(page, url):
    assert page.requests == [url], page.requests
    assert page.console_errors == [], page.console_errors
    assert page.exceptions == [], page.exceptions


@pytest.fixture(scope="module")
def pages(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("playback")
    from tropoi.representation.visual.terrain import (build_terrain_preview,
                                                      write_globe_html)
    terrain = write_globe_html(
        build_terrain_preview(_band_limited_terrain(), radius_m=6.371e6,
                              title="terrain only"), tmp / "terrain.html")
    return {"many": _export(tmp, "many", 4, TIMES),
            "one": _export(tmp, "one", 1, [43200.0]),
            "terrain": terrain}


def test_playback_matches_manual_selection_and_keeps_the_view(pages):
    with OfflinePage(BROWSER) as page:
        url = page.open(pages["many"])
        page.wait_for(f"{API} && document.getElementById('timeLabel')"
                      ".textContent")
        gl_ok = page.eval("window.__terrainGlobe.glOK")
        assert not page.eval("document.getElementById('playRow').hidden")

        # a view that playback must not disturb
        page.eval("document.getElementById('fieldButtons').children[2]"
                  ".click()")
        _set_input(page, "levelSlider", 2)
        _set_input(page, "opacity", 45)
        page.eval("(() => { const b = document.getElementById('streams');"
                  " if (!b.checked) b.click(); })()")
        page.eval(f"{API}.setExaggeration(40); {API}.centreOn(20, 60);"
                  f" {API}.setDistance(3.2)")
        page.eval("document.getElementById('mapAtm').click()")
        view = _info(page)
        assert view["streams"] is True and view["opacity"] == 0.45
        assert view["lines"] > 0

        # reference: each saved state selected with the slider
        manual = []
        for i in range(len(TIMES)):
            _set_input(page, "timeSlider", i)
            manual.append({"digest": page.eval(f"{API}.digest()"),
                           "label": _text(page, "timeLabel"),
                           "note": _text(page, "timeNote"),
                           "time": _info(page)["time"]})
        assert [m["time"]["time_s"] for m in manual] == TIMES
        assert [m["label"] for m in manual] == [
            "day 0 (t = 0 s) · 1/4", "day 0.25 (t = 21600 s) · 2/4",
            "day 0.375 (t = 32400 s) · 3/4", "day 1 (t = 86400 s) · 4/4"]
        assert "First exported state" in manual[0]["note"]
        assert "(10800 s)" in manual[2]["note"]
        assert "unevenly spaced" in manual[2]["note"]
        assert "does not interpolate" in manual[2]["note"]
        if gl_ok:
            assert len({m["digest"] for m in manual}) == len(TIMES)

        # record every frame playback shows, after it is fully drawn
        page.eval("""(() => {
          window.__frames = [];
          const lab = document.getElementById('timeLabel');
          new MutationObserver(() => {
            const a = window.__terrainGlobe.atmosphere, i = a.info();
            window.__frames.push({index: i.timeIndex, label: lab.textContent,
              digest: a.digest(), t: performance.now(), info: i});
          }).observe(lab, {childList: true, characterData: true,
                           subtree: true});
        })()""")
        _set_input(page, "timeSlider", 0)
        _set_input(page, "tSpeed", "2", event="change")
        assert _info(page)["fps"] == 2
        page.eval("window.__frames = []")
        _click(page, "tPlay")
        assert _info(page)["playing"] is True
        assert _text(page, "tPlay") == "❚❚ Pause"
        page.wait_for(f"!{API}.info().playing", timeout=30)
        frames = page.eval("window.__frames")
        assert [f["index"] for f in frames] == [1, 2, 3]   # stopped on last
        assert _text(page, "tPlay") == "▶ Play"
        assert page.eval("+document.getElementById('timeSlider').value") == 3
        for f in frames:
            assert f["label"] == manual[f["index"]]["label"]
            if gl_ok:
                assert f["digest"] == manual[f["index"]]["digest"]
            for key in ("field", "level", "opacity", "exag", "streams",
                        "range", "mapLabel"):
                if key == "mapLabel":
                    assert f["info"][key].startswith("Atmosphere")
                else:
                    assert f["info"][key] == view[key], key
        gaps = [b["t"] - a["t"] for a, b in zip(frames, frames[1:])]
        assert all(450.0 <= g <= 3000.0 for g in gaps), gaps
        cam = page.eval("(() => { const s = window.__terrainGlobe.state;"
                        " return [s.yaw, s.pitch, s.dist]; })()")
        assert cam == pytest.approx([-60 * 3.141592653589793 / 180,
                                     20 * 3.141592653589793 / 180, 3.2])

        # play from the last state replays from the first
        _set_input(page, "tSpeed", "8", event="change")
        page.eval("window.__frames = []")
        _click(page, "tPlay")
        page.wait_for(f"!{API}.info().playing", timeout=30)
        assert [f["index"] for f in page.eval("window.__frames")] == [
            0, 1, 2, 3]

        # loop: wraps from the last state to the first, until paused
        _click(page, "tLoop")
        assert _info(page)["loop"] is True
        page.eval("window.__frames = []")
        _click(page, "tPlay")
        page.wait_for("window.__frames.length >= 6", timeout=30)
        _click(page, "tPlay")                               # pause
        seq = [f["index"] for f in page.eval("window.__frames")]
        assert seq[:6] == [0, 1, 2, 3, 0, 1], seq       # from the last state
        stopped = _info(page)["timeIndex"]
        page.eval("new Promise(r => setTimeout(r, 400))")
        assert _info(page)["timeIndex"] == stopped
        assert _info(page)["playing"] is False

        # scrubbing during playback pauses and selects at once
        _click(page, "tPlay")
        _set_input(page, "timeSlider", 2)
        assert _info(page)["playing"] is False
        assert _info(page)["timeIndex"] == 2
        if gl_ok:
            assert page.eval(f"{API}.digest()") == manual[2]["digest"]
        page.eval("new Promise(r => setTimeout(r, 400))")
        assert _info(page)["timeIndex"] == 2

        # previous / next step (pausing), clamped at the ends
        _click(page, "tPlay")
        _click(page, "tNext")
        assert _info(page)["playing"] is False
        assert _info(page)["timeIndex"] == 3
        _click(page, "tNext")
        assert _info(page)["timeIndex"] == 3
        for expected in (2, 1, 0, 0):
            _click(page, "tPrev")
            assert _info(page)["timeIndex"] == expected
        if gl_ok:
            assert page.eval(f"{API}.digest()") == manual[0]["digest"]
        _assert_clean(page, url)


def test_one_state_export_has_no_playback_and_terrain_page_is_untouched(pages):
    with OfflinePage(BROWSER) as page:
        url = page.open(pages["one"])
        page.wait_for(f"{API} && document.getElementById('timeLabel')"
                      ".textContent")
        assert page.eval("document.getElementById('playRow').hidden")
        assert page.eval("document.getElementById('timeSlider').disabled")
        assert _text(page, "timeLabel") == "day 0.5 (t = 43200 s) · 1/1"
        assert "nothing to play" in _text(page, "timeNote")
        page.eval(f"{API}.play()")
        assert _info(page)["playing"] is False
        _assert_clean(page, url)

    with OfflinePage(BROWSER) as page:
        url = page.open(pages["terrain"])
        page.wait_for("window.__terrainGlobe")
        assert page.eval("window.__terrainGlobe.atmosphere") is None
        assert page.eval("document.getElementById('atmSection').hidden")
        _assert_clean(page, url)
