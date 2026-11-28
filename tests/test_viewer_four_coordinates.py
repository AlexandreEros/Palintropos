"""The four-coordinate selection model: capability per coordinate, one evaluation path.

Each test names the acceptance criterion it demonstrates. Atmospheric tests
use the Held–Suarez T42L20 archive; columnar ones a small global airless run
made by the airless tests' own fixture.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from tropoi.representation.diagnostics.pe_levels import reconstruct_levels
from tropoi.representation.visual import selection as model
from tropoi.representation.visual.selection import PRESETS, SelectionError, parse_selection
from tropoi.representation.visual.viewer_server import ViewerSession, provider_for

ROOT = Path(__file__).resolve().parents[1]
HS = ROOT / "runs/hs-T42L20-prod-002"
fixed = lambda v: dict(role="fixed", value=v)           # noqa: E731
axis = lambda r: dict(role=r)                            # noqa: E731


@pytest.fixture(scope="module")
def hs():
    if not HS.is_dir():
        pytest.skip(f"Missing archive: {HS}")
    session = ViewerSession([HS])
    return session, provider_for(session.roots[0], session.manifests[0])


def evaluate(session, **sel):
    data, mime = session.select(0, json.dumps(sel))
    assert mime == "application/json"
    return json.loads(data)


def array(values):
    return np.array(values, dtype=float)


def test_the_product_table_is_gone():
    """Criterion 1 and 6: no plot type exists in the selection model; presets are plain role data."""
    assert not hasattr(model, "PRODUCTS") and not hasattr(model, "product_for")
    for preset in PRESETS.values():
        assert set(preset) == {"label", "roles"}
        assert set(preset["roles"]) == set(model.COORDINATES)


def test_longitude_latitude_is_an_ordinary_assignment(hs):
    """Criterion 1: lon × lat at fixed vertical and time goes through the general engine, as JSON,
    with the values of the native-level kernel on the display mesh."""
    session, provider = hs
    sel = dict(quantity="temperature", longitude=axis("x"), latitude=axis("y"),
               vertical=dict(role="fixed", representation="native", value=12), time=dict(role="fixed", index=7))
    parsed = parse_selection(session.manifests[0], json.dumps(sel))
    assert "product" not in parsed and parsed["axes"] == dict(x="longitude", y="latitude")
    d = evaluate(session, **sel)
    assert d["axes"] == dict(x="longitude", y="latitude")
    assert d["description"] == "Longitude × latitude · vertical fixed · time fixed"
    lat, lon = provider.lat, np.append(provider.lon, 360.0)
    expected = reconstruct_levels(provider._read("state", 7), provider._context, lat, lon % 360.0, [12],
                                  vorticity_divergence=False)["temperature"][0]
    np.testing.assert_array_equal(array(d["values"]), expected)
    assert d["x"]["values"] == list(lon) and d["y"]["values"] == list(lat)


def test_time_mean_longitude_latitude_field(hs):
    """Criterion 2: the same roles with time = reduce give the equal-weight mean of the fixed-time fields."""
    session, _ = hs
    base = dict(quantity="temperature", longitude=axis("x"), latitude=axis("y"),
                vertical=dict(role="fixed", representation="pressure", value=50000.0))
    mean = evaluate(session, **base, time=dict(role="reduce", first=10, last=12))
    fields = [array(evaluate(session, **base, time=dict(role="fixed", index=t))["values"]) for t in (10, 11, 12)]
    total = np.zeros_like(fields[0])
    for f in fields:
        total += f
    np.testing.assert_array_equal(array(mean["values"]), total / 3)
    time = mean["coordinates"]["time"]
    # Regular cadence: the approved cell weights are all equal, so the mean is the plain 1/N mean, bit for bit.
    assert time["count"] == 3 and time["op"] == "mean" and time["weighting"] == "equal (regular cadence)"
    assert "interval it stands for" in time["rule"] and time["cadence"] == "regular"


def test_vertical_profile_at_a_point_and_a_saved_time(hs):
    """Criterion 3: the profile equals the column evaluator's, including the snapped point."""
    session, provider = hs
    d = evaluate(session, quantity="temperature", longitude=fixed(301.25), latitude=fixed(23.5),
                 vertical=dict(role="y", representation="pressure"), time=dict(role="fixed", index=5))
    column = provider.column(5, 23.5, 301.25, snap=True)
    assert d["values"] == column["temperature"] and d["y"]["values"] == column["pressure"]
    assert d["coordinates"]["latitude"]["value"] == column["latitude"]
    assert [m["value"] for m in d["markers"]] == [column["temperature"][-1]]


def test_time_mean_vertical_profile(hs):
    """Criterion 4: the time-mean profile is the equal-weight mean of the single-time profiles, on σ."""
    session, _ = hs
    base = dict(quantity="u", longitude=fixed(301.25), latitude=fixed(23.5),
                vertical=dict(role="y", representation="sigma"))
    mean = evaluate(session, **base, time=dict(role="reduce", first=20, last=23))
    each = [array(evaluate(session, **base, time=dict(role="fixed", index=t))["values"]) for t in range(20, 24)]
    total = np.zeros_like(each[0])
    for e in each:
        total += e
    np.testing.assert_array_equal(array(mean["values"]), total / 4)
    assert mean["table"]["columns"][-1].endswith("(time mean)") and mean["table"]["columns"][2] == "p (Pa) (mean)"
    # The level positions of p and Z differ between samples: refused locally, with the reason.
    with pytest.raises(SelectionError, match="differ from column to column"):
        evaluate(session, **dict(base, vertical=dict(role="y", representation="pressure")),
                 time=dict(role="reduce", first=20, last=23))


def test_reductions_compose_with_any_coordinate(hs):
    """Longitude, latitude and time means compose; the latitude mean is Gauss (area) weighted."""
    session, _ = hs
    sigma = dict(role="y", representation="sigma")
    t = dict(role="fixed", index=30)
    zonal = evaluate(session, quantity="temperature", longitude=dict(role="reduce"), latitude=axis("x"),
                     vertical=sigma, time=t)
    field = array(zonal["values"])                     # (level, latitude), Gauss nodes south to north
    lat = np.radians(zonal["x"]["values"])
    domain = evaluate(session, quantity="temperature", longitude=dict(role="reduce"),
                      latitude=dict(role="reduce"), vertical=sigma, time=t)
    from tropoi.spatial.grids.quadrature import gauss_latitude_nodes
    w = gauss_latitude_nodes(lat.size)[1][::-1]
    np.testing.assert_allclose(domain["values"], field @ w / w.sum(), rtol=1e-13)
    # Area weighting: the Gauss weights are the cos φ quadrature, not a plain mean over rows.
    assert not np.allclose(domain["values"], field.mean(axis=1), rtol=1e-6)
    assert "cos φ" in domain["coordinates"]["latitude"]["rule"]
    # A meridional-mean profile at one longitude and a zonal-mean profile at one latitude.
    for roles in (dict(longitude=fixed(90.0), latitude=dict(role="reduce")),
                  dict(longitude=dict(role="reduce"), latitude=fixed(-30.0))):
        d = evaluate(session, quantity="temperature", vertical=sigma, time=t, **roles)
        assert len(d["values"]) == 20 and all(np.isfinite(d["values"]))
    # A zonal-mean profile, time mean: three coordinates reduced or fixed, one axis.
    d = evaluate(session, quantity="v", longitude=dict(role="reduce"), latitude=fixed(-30.0), vertical=sigma,
                 time=dict(role="reduce", first=0, last=2))
    assert d["description"] == "Vertical · longitude mean · latitude fixed · time mean"


def test_a_vertical_mean_is_refused_with_its_reason(hs):
    session, _ = hs
    with pytest.raises(SelectionError, match="mass"):
        evaluate(session, quantity="temperature", longitude=fixed(0), latitude=fixed(0),
                 vertical=dict(role="reduce"), time=axis("x"))


def test_linear_sigma_is_a_display_choice(hs):
    """Criterion 5: the σ axis carries physical positions and the model's interfaces as band edges;
    log is only its default display, and no display parameter reaches the evaluation."""
    session, provider = hs
    d = evaluate(session, quantity="temperature", longitude=fixed(0), latitude=axis("x"),
                 vertical=dict(role="y", representation="sigma"), time=dict(role="fixed", index=0))
    assert d["y"]["values"] == list(provider._context.full_levels)
    assert d["y"]["edges"] == list(provider._context.sigma_interfaces)
    assert d["y"]["scale"] == "log" and d["y"]["representation"] == "sigma"
    with pytest.raises(SelectionError, match="Options"):
        evaluate(session, quantity="temperature", longitude=fixed(0), latitude=axis("x"),
                 vertical=dict(role="y", representation="sigma"), time=dict(role="fixed", index=0),
                 options=dict(scale="linear"))
    js = (ROOT / "src/tropoi/representation/visual/viewer_assets/explorer.js").read_text(encoding="utf-8")
    assert 'option("linear", "Linear")' in js and 'option("log", "Logarithmic")' in js


def test_presets_configure_roles_and_define_no_capability(hs):
    """Criterion 6: every preset is accepted or refused by the same per-coordinate checks as any
    hand-made assignment, and assignments that no preset names evaluate as well."""
    session, _ = hs
    manifest = session.manifests[0]
    quantity = manifest["explorer"]["quantities"]["temperature"]
    for preset in PRESETS.values():
        roles = preset["roles"]
        accepted = all(roles[c] in quantity["coordinates"][c]["roles"] for c in model.COORDINATES)
        sel = {c: dict(role=r) for c, r in roles.items()}
        for c in ("longitude", "latitude"):
            if roles[c] == "fixed":
                sel[c]["value"] = 10.0
        if roles["vertical"] != "none":
            sel["vertical"].update(representation="native" if roles["vertical"] == "fixed" else "sigma", value=3)
        sel["time"].update(index=0, first=0, last=1)
        try:
            parse_selection(manifest, json.dumps(dict(quantity="temperature", **sel)))
            parsed = True
        except SelectionError:
            parsed = False
        assert parsed == accepted, preset["label"]
    # Removing the presets from the manifest changes nothing the parser accepts.
    stripped = dict(manifest, explorer=dict(manifest["explorer"], presets={}))
    unnamed = dict(quantity="temperature", longitude=dict(role="reduce"), latitude=axis("y"),
                   vertical=dict(role="fixed", representation="native", value=19), time=axis("x"))
    text = json.dumps(dict(unnamed, time=dict(role="x", first=0, last=2)))
    assert parse_selection(stripped, text) == parse_selection(manifest, text)
    d = evaluate(session, **json.loads(text))            # latitude × time of the zonal mean: no preset names it
    assert d["axes"] == dict(x="time", y="latitude") and array(d["values"]).shape == (64, 3)


def test_globe_frames_are_an_output_format_not_a_product(hs):
    session, _ = hs
    sel = dict(quantity="temperature", longitude=axis("x"), latitude=axis("y"),
               vertical=dict(role="fixed", representation="native", value=19), time=dict(role="reduce", first=0, last=1),
               options=dict(format="frame"))
    with pytest.raises(SelectionError, match="globe draws one saved state"):
        session.select(0, json.dumps(sel))


@pytest.fixture(scope="module")
def airless(tmp_path_factory):
    from test_viewer_airless import _run
    root = _run(tmp_path_factory.mktemp("four") / "latlon", {"kind": "latlon", "nlat": 6, "nlon": 12}, 30)
    return ViewerSession([root])


def test_columnar_time_mean_map_and_area_weighted_profile(airless):
    """The same roles on a columnar archive: archived values, recorded area weights."""
    from tropoi.representation.archive.airless import AirlessArchive
    session = airless
    a = AirlessArchive(session.roots[0])
    base = dict(quantity="temperature", longitude=axis("x"), latitude=axis("y"),
                vertical=dict(role="fixed", representation="native", value=2))
    n = len(session.manifests[0]["time_axes"]["profile_samples"]["values"])
    mean = evaluate(session, **base, time=dict(role="reduce", first=0, last=n - 1))
    fields = [array(evaluate(session, **base, time=dict(role="fixed", index=t))["values"]) for t in range(n)]
    total = np.zeros_like(fields[0])
    for f in fields:
        total += f
    np.testing.assert_array_equal(array(mean["values"]), total / n)
    profile = evaluate(session, quantity="temperature", longitude=dict(role="reduce"), latitude=dict(role="reduce"),
                       vertical=dict(role="y", representation="depth"), time=dict(role="fixed", index=n - 1))
    lats, lons, index = a.layout
    cells = a.profile(n - 1)[:, index]                  # (layers, nlat, nlon)
    w = a.weights[index[:, 0]]
    np.testing.assert_allclose(profile["values"], cells.mean(axis=2) @ w / w.sum(), rtol=1e-14)
    assert "area weights" in profile["coordinates"]["latitude"]["rule"]
    assert "surface" not in profile                      # a skin strip belongs to 2-D results only
    skin = [m for m in profile["markers"] if m["kind"] == "surface"]
    assert len(skin) == 1 and skin[0]["position"] == 0.0  # the skin, as its own level at depth 0
    assert profile["table"]["rows"][0][0] == "skin"
