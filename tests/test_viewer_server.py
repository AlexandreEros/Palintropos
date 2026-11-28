"""Exercise the same loopback HTTP entry points as the browser on actual archives."""
import json
import struct
import threading
from urllib.error import HTTPError
from urllib.request import Request, ProxyHandler, build_opener

import numpy as np
import pytest

from tropoi.representation.visual.viewer_server import ViewerServer, ViewerSession
from test_viewer_manifest import BEN, SWE, BVE, MARS, CORE, FULL, GRAY, fixture


@pytest.fixture(scope="module")
def live():
    names = [BEN, SWE, BVE, MARS, CORE, FULL, GRAY]
    session = ViewerSession([fixture(n) for n in names], cache_bytes=2 << 20)
    with ViewerServer(session) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield server
        finally:
            server.shutdown()
            thread.join()


def fetch(url, **headers):
    opener = build_opener(ProxyHandler({}))
    with opener.open(Request(url, headers=headers), timeout=90) as response:
        return response.read()


def select(base, **selection):
    from urllib.parse import urlencode
    return base + "select?" + urlencode({"q": json.dumps(selection)})


def frame(url):
    data = fetch(url)
    size, = struct.unpack_from("<I", data)
    directory = json.loads(data[4:4+size])
    return {key: np.frombuffer(data, dtype="u1" if key in ("mask", "cells") else "<f4",
                               count=count, offset=4+size+offset)
            for key, (offset, count) in directory.items()}


def test_local_session_routes(live):
    assert live.server_address[0] == "127.0.0.1"
    assert len(json.loads(fetch(live.url + "runs"))["runs"]) == 7
    for headers, code in [({"Host": "unexpected.invalid"}, 403),
                          ({"Origin": "https://example.org"}, 403)]:
        with pytest.raises(HTTPError) as exc:
            fetch(live.url + "manifest", **headers)
        assert exc.value.code == code
    for path in ["config.json", "../../config.json", "frame", "structure", "diagnostic"]:
        with pytest.raises(HTTPError) as exc:
            fetch(live.url + path)
        assert exc.value.code == 404


def surface(coord, value):
    return dict(role="fixed", representation=coord, value=value)


@pytest.mark.parametrize("index,field,coord,value", [
    (0,"temperature","native",39), (1,"phi",None,0), (2,"zeta",None,0),
    (3,"temperature","pressure",300), (5,"ln_ps",None,0), (6,"temperature","sigma",0.5),
    (0,"temperature","height_above_surface",2000)])
def test_final_saved_frame_on_wire(live, index, field, coord, value):
    base = live.url.removesuffix("0/") + f"{index}/"
    manifest = json.loads(fetch(base+"manifest"))
    times = manifest["time_axes"]["snapshots"]["values"]
    vertical = surface(coord, value) if coord else dict(role="none")
    url = select(base, quantity=field, longitude=dict(role="x"), latitude=dict(role="y"),
                 vertical=vertical, time=dict(role="fixed", index=len(times)-1), options=dict(format="frame"))
    arrays = frame(url)
    n = arrays["lat"].size * arrays["lon"].size
    assert arrays["values"].size == arrays["height"].size == arrays["terrain"].size == n
    valid = arrays["mask"] == 0
    assert np.any(valid) and np.isfinite(arrays["values"][valid]).all()
    if index == 0:
        assert times[-1] == 30*86400 and 100 < np.median(arrays["values"][valid]) < 400
        assert manifest["star"]["axes"]["snapshots"][-1] == dict(latitude=0, longitude=0)
    if coord == "height_above_surface":
        # Z - Zs = 2 km everywhere it is sampled: the layer is lifted by the terrain.
        np.testing.assert_allclose(arrays["height"][valid] - arrays["terrain"][valid], 2000, atol=0.01)
    if index == 1:
        assert arrays["terrain"].max() > 1000
        assert (arrays["height"][valid] > arrays["terrain"][valid]).all()
    assert live.session.bytes_used <= live.session.cache_bytes


def test_recorded_series_and_invalid_requests(live):
    base = live.url.removesuffix("0/") + "4/"
    series = json.loads(fetch(select(base, quantity="series:mean_T", time=dict(role="x"))))
    assert series["presentation"]["units"] == "K" and len(series["x"]["values"]) == len(series["values"]) > 1
    sel = dict(quantity="temperature", longitude=dict(role="x"), latitude=dict(role="y"),
               vertical=surface("native", 0), time=dict(role="fixed", index=4320))
    bad = [select(live.url, **sel),                                          # outside the saved axis
           select(live.url, **dict(sel, time=dict(role="fixed", index=0), vertical=surface("native", 0),
                                    latitude=dict(role="x"))),               # no plot for these roles
           select(live.url, **dict(sel, quantity="pressure", time=dict(role="fixed", index=0),
                                    vertical=surface("pressure", 50000))),   # p on an isobar: not offered
           select(live.url, **dict(sel, quantity="wind")),                   # a vector
           live.url + "select?q=%7B%7D&q=%7B%7D",                            # repeated parameter
           live.url + "select?field=temperature",                            # only q is accepted
           live.url + "select?q=" + "x" * 5000]                              # too long
    for url in bad:
        with pytest.raises(HTTPError) as exc:
            fetch(url)
        assert exc.value.code == 400


def test_structure_selections_on_ben2_and_statistics_core(live):
    manifest = json.loads(fetch(live.url + "manifest"))
    quantities = manifest["explorer"]["quantities"]
    assert quantities["mass_streamfunction"]["coordinates"]["longitude"]["roles"] == ["reduce"]
    last = len(manifest["time_axes"]["snapshots"]["values"]) - 1
    fixed = lambda v: dict(role="fixed", value=v)
    column = json.loads(fetch(select(live.url, quantity="temperature", longitude=fixed(-79), latitude=fixed(-19.4),
                                     vertical=dict(role="y", representation="geopotential_height"),
                                     time=dict(role="fixed", index=last))))
    where = column["coordinates"]["longitude"]
    assert where["sampling"] == "native Gaussian grid node" and where["value"] == 281.25
    assert len(column["values"]) == len(column["y"]["values"]) == 40
    # Slab Tₛ and the lowest-level air temperature are separate quantities.
    surface_t = [m for m in column["markers"] if m["kind"] == "surface"][0]
    lowest = [m for m in column["markers"] if m["kind"] == "level"][0]
    assert surface_t["value"] != lowest["value"] and lowest["value"] == column["values"][39]
    section = json.loads(fetch(select(live.url, quantity="u", longitude=dict(role="x"), latitude=fixed(0),
                                      vertical=dict(role="y", representation="sigma"),
                                      time=dict(role="fixed", index=last))))
    assert len(section["x"]["values"]) == 181 and len(section["values"]) == 40 and len(section["values"][0]) == 181
    np.testing.assert_allclose(np.array(section["values"])[:, 0], np.array(section["values"])[:, -1], atol=1e-9)
    mean = json.loads(fetch(select(live.url, quantity="mass_streamfunction", latitude=dict(role="x"),
                                   longitude=dict(role="reduce", op="mean"),
                                   vertical=dict(role="y", representation="sigma"),
                                   time=dict(role="reduce", op="mean", first=last-1, last=last))))
    assert mean["coordinates"]["time"]["count"] == 2 and len(mean["values"]) == 41
    assert np.all(np.array(mean["values"][0]) == 0)
    assert live.session.bytes_used <= live.session.cache_bytes
    core = live.url.removesuffix("0/") + "4/"
    for base, sel in [(core, dict(quantity="temperature", longitude=fixed(0), latitude=fixed(0),
                                  vertical=dict(role="y", representation="sigma"), time=dict(role="fixed", index=0))),
                      (live.url, dict(quantity="temperature", latitude=dict(role="x"), longitude=dict(role="reduce"),
                                      vertical=dict(role="y", representation="sigma"),
                                      time=dict(role="reduce", first=3, last=2))),
                      (live.url, dict(quantity="temperature", longitude=fixed(0), latitude=fixed(float("nan")),
                                      vertical=dict(role="y", representation="sigma"), time=dict(role="fixed", index=0)))]:
        with pytest.raises(HTTPError) as exc:
            fetch(select(base, **sel))
        assert exc.value.code == 400
