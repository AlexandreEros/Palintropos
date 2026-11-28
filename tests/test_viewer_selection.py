"""The selection model: surface-relative heights and refusals, on actual archives."""
import json
from pathlib import Path

import numpy as np
import pytest

from tropoi.representation.diagnostics.pe_surfaces import pressure_surface
from tropoi.representation.visual.selection import SelectionError
from tropoi.representation.visual.viewer_server import ViewerSession, encode_frame, provider_for

ROOT = Path(__file__).resolve().parents[1]
MOLA = ROOT / "docs/runs/20261027T175327Z_orographic-thermal-wave_rot24p62h_r3_l21_dt11080p3s_3a46f966_71c5441a"
FLAT = ROOT / "runs/hs-T42L20-prod-002"


def _provider(root):
    if not root.is_dir():
        pytest.skip(f"Missing archive: {root}")
    session = ViewerSession([root])
    return session, provider_for(session.roots[0], session.manifests[0])


def test_height_above_a_flat_surface_is_the_constant_height_surface():
    """Over flat terrain Z - Zs = h is the Z = h surface, byte for byte."""
    _, provider = _provider(FLAT)
    for field in ("temperature", "u"):
        relative = encode_frame(provider.slice(3, field, "height_above_surface", 1500.0))
        absolute = encode_frame(provider.slice(3, field, "geopotential_height", 1500.0))
        assert relative == absolute


def test_height_above_terrain_is_the_constant_height_surface_at_h_plus_zs():
    """Over MOLA terrain each column of Z - Zs = h equals the Z = h + Zs(column) surface there."""
    _, provider = _provider(MOLA)
    frame = provider._read("state", 4)
    lat = np.array([-40.0, -10.0, 20.0, 45.0])
    lon = np.array([30.0, 140.0, 250.0])
    h = 3000.0
    relative = pressure_surface(frame, provider._context, lat, lon, ("height_above_surface", h))
    for i, a in enumerate(lat):
        for j, b in enumerate(lon):
            zs = relative.surface_height[i, j]
            point = pressure_surface(frame, provider._context, np.array([a]), np.array([b]),
                                     ("geopotential_height", h + zs))
            assert relative.mask[i, j] == point.mask[0, 0]
            if relative.mask[i, j] == 0:
                for name in ("temperature", "u", "v", "pressure"):
                    assert relative.values[name][i, j] == pytest.approx(point.values[name][0, 0], rel=1e-9)
    assert np.ptp(relative.surface_height) > 1000          # the terrain really varies


def test_fractional_native_level_and_unknown_members_are_refused():
    session, _ = _provider(MOLA)
    fixed = lambda v: dict(role="fixed", value=v)
    base = dict(quantity="temperature", longitude=fixed(10), latitude=fixed(10), time=dict(role="x", first=0, last=2))
    with pytest.raises(SelectionError, match="Native level must be a zero-based full-level index"):
        session.select(0, json.dumps(dict(base, vertical=dict(role="fixed", representation="native", value=1.5))))
    with pytest.raises(SelectionError, match="Unknown selection member"):
        session.select(0, json.dumps(dict(base, vertical=dict(role="fixed", representation="native", value=1),
                                          extra=1)))
    with pytest.raises(SelectionError, match="Only one coordinate"):
        session.select(0, json.dumps(dict(base, latitude=dict(role="x"),
                                          vertical=dict(role="fixed", representation="native", value=1))))
