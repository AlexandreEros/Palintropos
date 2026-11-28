"""The wind layer of a longitude × latitude result (restores the flat-map streamlines).

The layer carries u and v as companions of the field: the same roles, the same
sample set and the same pass over the saved states, so the field itself is
unchanged and the winds are exactly the kernels' own on that surface.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from tropoi.representation.diagnostics.pe_levels import reconstruct_levels
from tropoi.representation.visual.selection import SelectionError, parse_selection
from tropoi.representation.visual.viewer_server import ViewerSession, provider_for
from tropoi.spatial.transforms.host_synthesis import synthesize_wind_latlon

from test_viewer_manifest import BVE as BVE_ID, MARS as MARS_ID, SWE as SWE_ID

ROOT = Path(__file__).resolve().parents[1]
HS = ROOT / "runs/hs-T42L20-prod-002"
axis = lambda r: dict(role=r)                            # noqa: E731
LONLAT = dict(longitude=axis("x"), latitude=axis("y"))


def _session(path):
    if not path.is_dir():
        pytest.skip(f"Missing archive: {path}")
    session = ViewerSession([path])
    return session, provider_for(session.roots[0], session.manifests[0])


@pytest.fixture(scope="module")
def hs():
    return _session(HS)


def evaluate(session, **sel):
    data, mime = session.select(0, json.dumps(sel))
    assert mime == "application/json"
    return json.loads(data)


def array(values):
    return np.array([[np.nan if v is None else v for v in row] for row in values], dtype=float)


def test_capability_is_described_per_quantity(hs):
    session, _ = hs
    quantities = session.manifests[0]["explorer"]["quantities"]
    assert quantities["temperature"]["layers"]["wind"]["available"] is True
    assert quantities["temperature"]["layers"]["wind"]["styles"] == ["psi_contours", "streamlines", "quiver"]
    assert quantities["temperature"]["layers"]["wind"]["psi_surfaces"] == ["native", "sigma"]
    assert "not trajectories" in quantities["temperature"]["layers"]["wind"]["definition"]
    psi = quantities["mass_streamfunction"]["layers"]["wind"]
    assert psi["available"] is False and "latitude × σ" in psi["reason"]
    surface = quantities["ln_ps"]["layers"]["wind"]
    assert surface["available"] is False and "no vertical level" in surface["reason"]


def test_the_field_is_unchanged_and_the_wind_is_the_kernels_own(hs):
    session, provider = hs
    sel = dict(quantity="temperature", **LONLAT, vertical=dict(role="fixed", representation="native", value=12),
               time=dict(role="fixed", index=7))
    plain = evaluate(session, **sel)
    layered = evaluate(session, **sel, options=dict(wind=True))
    assert layered["values"] == plain["values"]
    assert plain["companions"] is None
    lat, lon = provider.lat, np.append(provider.lon, 360.0)
    levels = reconstruct_levels(provider._read("state", 7), provider._context, lat, lon % 360.0, [12],
                                vorticity_divergence=False)
    np.testing.assert_array_equal(array(layered["companions"]["wind_u"]["values"]), levels["u"][0])
    np.testing.assert_array_equal(array(layered["companions"]["wind_v"]["values"]), levels["v"][0])
    assert layered["companions"]["wind_u"]["layer"] == "wind"
    assert any(note.startswith("Wind layer:") for note in layered["notes"])


def test_time_mean_wind_is_the_mean_of_the_saved_winds(hs):
    session, _ = hs
    base = dict(quantity="temperature", **LONLAT, vertical=dict(role="fixed", representation="pressure",
                                                                value=50000.0))
    mean = evaluate(session, **base, time=dict(role="reduce", first=10, last=12), options=dict(wind=True))
    for name in ("wind_u", "wind_v"):
        winds = [array(evaluate(session, **base, time=dict(role="fixed", index=t),
                                options=dict(wind=True))["companions"][name]["values"]) for t in (10, 11, 12)]
        total = np.zeros_like(winds[0])
        for w in winds:
            total += w
        np.testing.assert_array_equal(array(mean["companions"][name]["values"]), total / 3)


@pytest.mark.parametrize("change, reason", [
    (dict(quantity="mass_streamfunction", longitude=dict(role="reduce"), latitude=axis("x"),
          vertical=dict(role="y", representation="sigma"), time=dict(role="fixed", index=0)), "latitude × σ"),
    (dict(longitude=dict(role="fixed", value=0.0), latitude=axis("x"),
          vertical=dict(role="y", representation="sigma")), "longitude × latitude"),
    (dict(time=dict(role="reduce", first=0, last=2, op="max")), "maximum has no wind"),
    (dict(vertical=dict(role="reduce", op="max")), "fix the vertical"),
    (dict(quantity="ln_ps", vertical=dict(role="none")), "no vertical level"),
])
def test_unsupported_wind_layers_are_refused_with_their_reason(hs, change, reason):
    session, _ = hs
    sel = dict(quantity="temperature", **LONLAT, vertical=dict(role="fixed", representation="native", value=5),
               time=dict(role="fixed", index=0), options=dict(wind=True))
    sel.update(change)
    with pytest.raises(SelectionError, match=reason):
        parse_selection(session.manifests[0], json.dumps(sel))


def test_a_masked_surface_masks_its_wind():
    session, _ = _session(ROOT / MARS_ID)
    ps = array(evaluate(session, quantity="ln_ps", **LONLAT, vertical=dict(role="none"),
                        time=dict(role="fixed", index=0))["values"])
    target = float(np.median(ps))
    d = evaluate(session, quantity="temperature", **LONLAT,
                 vertical=dict(role="fixed", representation="pressure", value=target),
                 time=dict(role="fixed", index=0), options=dict(wind=True))
    masked = np.isnan(array(d["values"]))
    assert masked.any() and not masked.all()
    for name in ("wind_u", "wind_v"):
        np.testing.assert_array_equal(np.isnan(array(d["companions"][name]["values"])), masked)


@pytest.mark.parametrize("archive, quantity", [(BVE_ID, "zeta"), (SWE_ID, "phi")])
def test_single_layer_archives_carry_their_wind(archive, quantity):
    session, provider = _session(ROOT / archive)
    d = evaluate(session, quantity=quantity, **LONLAT, vertical=dict(role="none"),
                 time=dict(role="fixed", index=1), options=dict(wind=True))
    frame = provider._read("state", 1)
    swe = provider.solver == "swe"
    zeta = frame[0] if swe else frame
    delta = frame[1] if swe else np.zeros_like(zeta)
    lat, lon = provider.lat, np.append(provider.lon, 360.0)
    u, v = synthesize_wind_latlon(zeta, delta, provider.radius, lat, lon % 360.0)
    np.testing.assert_array_equal(array(d["companions"]["wind_u"]["values"]), u)
    np.testing.assert_array_equal(array(d["companions"]["wind_v"]["values"]), v)


def test_a_columnar_archive_has_no_wind_layer():
    session, _ = _session(ROOT / "runs/airless-ceres-transient-clean-0409b46")
    point = session.manifests[0]["explorer"]["horizontal"]["points"][0]
    sel = dict(quantity="incident_shortwave", longitude=dict(role="fixed", value=point["longitude"]),
               latitude=dict(role="fixed", value=point["latitude"]), vertical=dict(role="none"),
               time=dict(role="x", first=0, last=1), options=dict(wind=True))
    with pytest.raises(SelectionError, match="no horizontal wind"):
        parse_selection(session.manifests[0], json.dumps(sel))
