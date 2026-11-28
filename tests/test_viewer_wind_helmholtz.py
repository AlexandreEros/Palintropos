"""ψ and the kinetic-energy split of the Explorer's wind layer (v = r̂ × ∇ψ + ∇χ).

ψ = ∇⁻²ζ from the surface's own ζ coefficients (the existing inversion), and
the energy split from the ζ and δ coefficients (the existing modal energy,
the one behind the README's W5 figure). Analytic states are written into a
temporary copy of an archive.
"""
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from tropoi.representation.visual.viewer_server import ViewerSession, provider_for
from tropoi.spatial.laplacian import invert_laplacian, laplacian_eigenvalues
from tropoi.spatial.transforms.host_synthesis import synthesize_latlon

from test_viewer_manifest import BVE as BVE_ID, SWE as SWE_ID

ROOT = Path(__file__).resolve().parents[1]
HS = ROOT / "runs/hs-T42L20-prod-002"
W5 = ROOT / SWE_ID
LONLAT = dict(longitude=dict(role="x"), latitude=dict(role="y"))


def _session(path):
    if not path.is_dir():
        pytest.skip(f"Missing archive: {path}")
    session = ViewerSession([path])
    return session, provider_for(session.roots[0], session.manifests[0])


def evaluate(session, **sel):
    data, mime = session.select(0, json.dumps(dict(sel, options=dict(wind=True))))
    assert mime == "application/json"
    return json.loads(data)


def array(values):
    return np.array([[np.nan if v is None else v for v in row] for row in values], dtype=float)


def _analytic_copy(tmp_path, archive, frame):
    """A copy of ``archive`` whose every saved state is ``frame(shape, radius)``."""
    if not archive.is_dir():
        pytest.skip(f"Missing archive: {archive}")
    target = tmp_path / archive.name
    shutil.copytree(archive, target)
    session, provider = _session(target)
    path = target / session.manifests[0]["resources"]["state"]["path"]
    states = np.load(path)
    states[...] = frame(states.shape[1:], provider.radius)
    np.save(path, states)
    return _session(target)


@pytest.fixture(scope="module")
def hs():
    return _session(HS)


def test_psi_on_a_native_level_is_the_inverse_laplacian_of_its_vorticity(hs):
    session, provider = hs
    d = evaluate(session, quantity="temperature", **LONLAT,
                 vertical=dict(role="fixed", representation="native", value=12), time=dict(role="fixed", index=7))
    context = provider._context
    zeta = provider._read("state", 7)[12]
    psi = synthesize_latlon(invert_laplacian(zeta, laplacian_eigenvalues(zeta.shape[0] - 1, context.radius_m)),
                            provider.lat, np.append(provider.lon, 360.0) % 360.0)
    np.testing.assert_array_equal(array(d["companions"]["psi"]["values"]), psi)
    assert d["wind"]["psi"]["available"] and "l = 0" in d["wind"]["psi"]["convention"]
    assert "area-weighted" in d["wind"]["kinetic_energy"]["rule"]


def test_psi_on_a_sigma_surface_through_a_native_level_is_that_levels_psi(hs):
    session, provider = hs
    base = dict(quantity="temperature", **LONLAT, time=dict(role="fixed", index=7))
    sigma = provider._context.full_levels[12]
    native = evaluate(session, **base, vertical=dict(role="fixed", representation="native", value=12))
    surface = evaluate(session, **base, vertical=dict(role="fixed", representation="sigma", value=sigma))
    a, b = array(native["companions"]["psi"]["values"]), array(surface["companions"]["psi"]["values"])
    np.testing.assert_allclose(b, a, rtol=0, atol=1e-12 * np.abs(a).max())


def test_no_psi_on_a_pressure_surface_and_the_reason(hs):
    session, _ = hs
    d = evaluate(session, quantity="temperature", **LONLAT,
                 vertical=dict(role="fixed", representation="pressure", value=50000.0), time=dict(role="fixed", index=7))
    assert "psi" not in d["companions"] and d["wind"]["psi"]["available"] is False
    assert "interpolation weight varies" in d["wind"]["psi"]["reason"]
    assert {"wind_u", "wind_v"} <= set(d["companions"])          # streamlines and arrows stay available


def test_time_mean_psi_is_the_psi_of_the_time_mean_vorticity(hs):
    session, provider = hs
    base = dict(quantity="temperature", **LONLAT, vertical=dict(role="fixed", representation="native", value=4))
    mean = evaluate(session, **base, time=dict(role="reduce", first=10, last=12))
    single = [array(evaluate(session, **base, time=dict(role="fixed", index=t))["companions"]["psi"]["values"])
              for t in (10, 11, 12)]
    psi = array(mean["companions"]["psi"]["values"])
    # ψ is linear in ζ: the ψ of the mean equals the mean of the ψs up to roundoff.
    np.testing.assert_allclose(psi, sum(single) / 3, rtol=0, atol=1e-12 * np.abs(psi).max())
    assert mean["wind"]["psi"]["time_mean"] is True


def test_w5_divergent_share_matches_the_readme_figure():
    """The README's W5 figure records the largest share over its four maps and each map's rms speeds."""
    session, _ = _session(W5)
    record = json.loads((W5 / "assets" / "overview.json").read_text(encoding="utf-8"))
    shares = []
    for index, figure in enumerate(record["maps"]):
        ke = evaluate(session, quantity="phi", **LONLAT, vertical=dict(role="none"),
                      time=dict(role="fixed", index=index))["wind"]["kinetic_energy"]
        assert ke["rotational_rms_speed"] == pytest.approx(figure["rms_speed"]["rotational"], rel=1e-12)
        assert ke["divergent_rms_speed"] == pytest.approx(figure["rms_speed"]["divergent"], rel=1e-12, abs=1e-12)
        shares.append(ke["divergent_share"])
    assert max(s for s in shares if s is not None) == record["vector_choice"]["max_divergent_kinetic_energy_fraction"]


def test_solid_body_rotation_psi_is_minus_u_a_sin_latitude(tmp_path):
    speed = 13.0

    def frame(shape, radius):
        coeff = np.zeros(shape, dtype=complex)
        coeff[1, 0] = 2 * speed / radius * np.sqrt(4 * np.pi / 3)        # ζ = 2U sin φ / a
        return coeff
    session, provider = _analytic_copy(tmp_path, ROOT / BVE_ID, frame)
    d = evaluate(session, quantity="zeta", **LONLAT, vertical=dict(role="none"), time=dict(role="fixed", index=0))
    psi = array(d["companions"]["psi"]["values"])
    expected = -speed * provider.radius * np.sin(np.radians(provider.lat))[:, None] * np.ones((1, psi.shape[1]))
    np.testing.assert_allclose(psi, expected, rtol=0, atol=1e-9 * speed * provider.radius)
    ke = d["wind"]["kinetic_energy"]
    assert ke["non_divergent"] is True and ke["divergent_share"] == 0.0
    assert ke["rotational_max_speed"] == pytest.approx(speed, rel=1e-9)


def test_purely_divergent_flow_has_no_rotational_part(tmp_path):
    """ζ = 0, δ ∝ Y₁⁰: ψ ≡ 0 and all of the kinetic energy is divergent, yet the wind is not zero."""

    def frame(shape, radius):
        coeff = np.zeros(shape, dtype=complex)
        coeff[0, 0, 0] = 1.0                                 # φ: any layer thickness
        coeff[1, 1, 0] = 1e-6                                # δ
        return coeff
    session, _ = _analytic_copy(tmp_path, W5, frame)
    d = evaluate(session, quantity="phi", **LONLAT, vertical=dict(role="none"), time=dict(role="fixed", index=0))
    assert np.abs(array(d["companions"]["psi"]["values"])).max() == 0.0
    ke = d["wind"]["kinetic_energy"]
    assert ke["divergent_share"] == 1.0 and ke["rotational_max_speed"] == 0.0 and ke["non_divergent"] is False
    assert np.nanmax(np.hypot(array(d["companions"]["wind_u"]["values"]), array(d["companions"]["wind_v"]["values"]))) > 0


def test_psi_follows_the_result_axes(hs):
    session, _ = hs
    base = dict(vertical=dict(role="fixed", representation="native", value=12), time=dict(role="fixed", index=7))
    a = evaluate(session, quantity="temperature", **LONLAT, **base)
    b = evaluate(session, quantity="temperature", longitude=dict(role="y"), latitude=dict(role="x"), **base)
    np.testing.assert_array_equal(array(b["companions"]["psi"]["values"]), array(a["companions"]["psi"]["values"]).T)
