"""Air density ρ = p/(R T): an ordinary derived quantity of the four-coordinate engine."""
import json
from pathlib import Path

import numpy as np
import pytest

from tropoi.representation.archive.viewer_manifest import _derived_fields
from tropoi.representation.visual.quantities import viewer_presentation
from tropoi.representation.visual.viewer_server import ViewerSession, provider_for

ROOT = Path(__file__).resolve().parents[1]
HS = ROOT / "runs/hs-T42L20-prod-002"
MARS = ROOT / ("docs/runs/20261027T175327Z_orographic-thermal-wave_rot24p62h_r3_l21_dt11080p3s_"
               "3a46f966_71c5441a")


def _levels(**extra):
    fields = dict(zeta=dict(levels=True), delta=dict(levels=True), temperature=dict(levels=True),
                  ln_ps=dict(levels=False), **extra)
    return fields


def test_density_is_advertised_only_with_a_recorded_gas_constant_and_no_tracer():
    geometry = dict(reference_radius_m=6.371e6)
    fields = _levels()
    _derived_fields(fields, dict(gravity=9.8, r_dry=287.0, topography="flat"), geometry, {})
    assert fields["density"]["inputs"] == ["temperature", "pressure", "physics.r_dry"]
    assert fields["density"]["units"] == "kg m^-3"
    fields = _levels()
    _derived_fields(fields, dict(gravity=9.8, topography="flat"), geometry, {})
    assert "density" not in fields                       # never a default 287
    fields = _levels(specific_humidity=dict(levels=True))
    _derived_fields(fields, dict(gravity=9.8, r_dry=287.0, topography="flat"), geometry, {})
    assert "density" not in fields                       # variable composition: one R is not the mixture's


def test_presentation():
    p = viewer_presentation("density", {})
    assert (p["name"], p["symbol"], p["units"], p["origin"], p["group"]) == (
        "Air density", "ρ", "kg m⁻³", "derived", "Thermodynamics")
    assert "records" in p["definition"] and "p/(R T)" in p["definition"]


@pytest.fixture(scope="module", params=[HS, MARS], ids=["held-suarez", "mars-mola"])
def archive(request):
    if not request.param.is_dir():
        pytest.skip(f"Missing archive: {request.param}")
    session = ViewerSession([request.param])
    return session, provider_for(session.roots[0], session.manifests[0])


def select(session, **sel):
    data, _ = session.select(0, json.dumps(sel))
    return json.loads(data)


def _r(session):
    physics = session.manifests[0]["physics"]
    return physics.get("r_dry") or physics["gas_constant"]


@pytest.mark.parametrize("vertical", [
    dict(role="fixed", representation="native", value=1),
    dict(role="fixed", representation="sigma", value=0.7),
    dict(role="fixed", representation="height_above_surface", value=2000.0),
], ids=["native", "sigma", "z-zs"])
def test_density_from_p_and_t_at_the_same_samples(archive, vertical):
    session, _ = archive
    base = dict(longitude=dict(role="x"), latitude=dict(role="y"), vertical=vertical, time=dict(role="fixed", index=3))
    rho = np.array(select(session, quantity="density", **base)["values"], dtype=float)
    p = np.array(select(session, quantity="pressure", **base)["values"], dtype=float)
    t = np.array(select(session, quantity="temperature", **base)["values"], dtype=float)
    np.testing.assert_allclose(rho, p / (_r(session) * t), rtol=1e-15, equal_nan=True)


def test_density_on_a_pressure_surface(archive):
    session, provider = archive
    target = 0.6 * provider.manifest["physics"].get("surface_pressure", 1e5)
    base = dict(longitude=dict(role="x"), latitude=dict(role="y"),
                vertical=dict(role="fixed", representation="pressure", value=target), time=dict(role="fixed", index=2))
    rho = np.array(select(session, quantity="density", **base)["values"], dtype=float)
    t = np.array(select(session, quantity="temperature", **base)["values"], dtype=float)
    valid = np.isfinite(rho)                     # below the terrain the surface is masked
    assert valid.any() and np.array_equal(valid, np.isfinite(t))
    np.testing.assert_allclose((rho * _r(session) * t)[valid], target, rtol=1e-14)


def test_density_profile_and_time_mean_use_the_same_engine(archive):
    session, provider = archive
    point = dict(longitude=dict(role="fixed", value=30.0), latitude=dict(role="fixed", value=10.0),
                 vertical=dict(role="y", representation="sigma"))
    n = len(provider.manifest["time_axes"]["snapshots"]["values"])
    mean = select(session, quantity="density", **point, time=dict(role="reduce", op="mean", first=n - 3, last=n - 1))
    profiles = [select(session, quantity="density", **point, time=dict(role="fixed", index=i))["values"]
                for i in range(n - 3, n)]
    np.testing.assert_allclose(mean["values"], np.mean(profiles, axis=0), rtol=1e-15)
    # Mean of ρ, not p̄/(R T̄): the quantity is evaluated per sample, then reduced.
    assert mean["presentation"]["units"] == "kg m⁻³"


def test_density_magnitudes():
    """Order-of-magnitude sanity: an Earth-like surface ~1 kg m⁻³, the Mars run ~10⁻² kg m⁻³."""
    for root, lo, hi in ((HS, 0.9, 1.6), (MARS, 0.005, 0.03)):
        if not root.is_dir():
            pytest.skip(f"Missing archive: {root}")
        session = ViewerSession([root])
        d = select(session, quantity="density", longitude=dict(role="reduce", op="mean"),
                   latitude=dict(role="reduce", op="mean"), vertical=dict(role="y", representation="sigma"),
                   time=dict(role="fixed", index=0))
        assert lo < d["values"][-1] < hi


def test_globe_frame_density(archive):
    session, provider = archive
    frame = provider.slice(1, "density", "native", 1)
    p = provider.slice(1, "pressure", "native", 1)["values"]
    t = provider.slice(1, "temperature", "native", 1)["values"]
    np.testing.assert_allclose(frame["values"], p / (_r(session) * t), rtol=1e-15)
