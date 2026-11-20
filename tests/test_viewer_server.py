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


def frame(url):
    data = fetch(url)
    size, = struct.unpack_from("<I", data)
    directory = json.loads(data[4:4+size])
    return {key: np.frombuffer(data, dtype="u1" if key == "mask" else "<f4",
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
    for path in ["config.json", "../../config.json"]:
        with pytest.raises(HTTPError) as exc:
            fetch(live.url + path)
        assert exc.value.code == 404


@pytest.mark.parametrize("index,field,coord,value", [
    (0,"temperature","native",39), (1,"phi","native",0), (2,"zeta","native",0),
    (3,"temperature","pressure",300), (5,"ln_ps","native",0), (6,"temperature","sigma",0.5)])
def test_final_saved_frame_on_wire(live, index, field, coord, value):
    base = live.url.removesuffix("0/") + f"{index}/"
    manifest = json.loads(fetch(base+"manifest"))
    times = manifest["time_axes"]["snapshots"]["values"]
    url = base+f"frame?time={len(times)-1}&field={field}&coordinate={coord}&value={value}"
    arrays = frame(url)
    n = arrays["lat"].size * arrays["lon"].size
    assert arrays["values"].size == arrays["height"].size == arrays["terrain"].size == n
    valid = arrays["mask"] == 0
    assert np.any(valid) and np.isfinite(arrays["values"][valid]).all()
    if index == 0:
        assert times[-1] == 30*86400 and 100 < np.median(arrays["values"][valid]) < 400
        assert manifest["star"]["positions"][-1] == dict(latitude=0, longitude=0)
    if index == 1:
        assert arrays["terrain"].max() > 1000
        assert (arrays["height"][valid] > arrays["terrain"][valid]).all()
    assert live.session.bytes_used <= live.session.cache_bytes


def test_core_diagnostic_and_invalid_request(live):
    base = live.url.removesuffix("0/") + "4/"
    series = json.loads(fetch(base + "diagnostic?field=mean_T"))
    assert series["units"] == "K" and len(series["x"]) == len(series["y"]) > 1
    with pytest.raises(HTTPError) as exc:
        fetch(live.url + "frame?time=4320&field=temperature")
    assert exc.value.code == 400
