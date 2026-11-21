"""THAI Ben 2 investigations through the Viewer 2 provider, against independent routines.

Archives (verified by sha256 against the Colab VM lists before use):
A, no sponge, days 1-30; B, moderate sponge, days 1-30, and its extension,
days 31-120, with the same config_sha256 8ad59ee8. Day 30 of B is only in the
first directory, so the two are opened separately and never merged.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from tropoi.representation.archive.viewer_manifest import viewer_manifest
from tropoi.representation.visual.viewer_fields import FrameProvider
from tropoi.spatial.transforms.host_synthesis import synthesize_latlon

ROOT = Path(__file__).resolve().parents[1]
STEM = "runs/dev-thai-ben2-ck-heated-T21L40-f44c039-layer-linear-dt600-"
A, B30, BEXT = (ROOT / (STEM + s) for s in ("30d", "spongeB-120d", "spongeB-120d-from-d030"))
EXT_JSON = ROOT / "runs/diagnostics/sponge/results-ext/ext.json"
DAY = 86400.0


def provider(root):
    if not root.is_dir():
        pytest.skip(f"Missing archive: {root}")
    return FrameProvider(root, viewer_manifest(root))


def index(p, day):
    return p.manifest["time_axes"]["snapshots"]["values"].index(day * DAY)


def test_day30_nightside_surface_temperature_is_resolved_not_uniform():
    p = provider(A)
    t = index(p, 30)
    slab = np.load(A / "states/surface-d00030.npy").reshape(32, 64)
    lon = 360 * np.arange(64) / 64
    night = np.abs((lon + 180) % 360 - 180) > 90           # substellar point at 0°E
    # The viewer's map is the slab bilinearly sampled on the 2° mesh.
    frame = p.slice(t, "surface_temperature")
    mesh_night = np.abs((frame["lon"] + 180) % 360 - 180) > 90
    lo, hi = frame["values"].min(), frame["values"].max()
    night_values = frame["values"][:, mesh_night]
    # Bilinear display sampling stays inside the native nodes' range.
    assert slab.min() - 1e-9 <= night_values.min() < slab[:, night].min() + 0.5
    assert slab[:, night].max() - 0.5 < night_values.max() <= slab.max() + 1e-9
    # Model values span about 22 K on the nightside; the auto range spans about 110 K.
    assert night_values.max() - night_values.min() > 20
    assert (night_values.max() - night_values.min()) / (hi - lo) < 0.25
    # Point readouts at native nodes return the stored slab values, distinct from air.
    for lat, lon_deg in [(2.79, 180.0), (-47.4, 120.9), (64.2, 236.25)]:
        c = p.column(t, lat, lon_deg)
        i = int(np.abs(np.degrees(np.arcsin(np.polynomial.legendre.leggauss(32)[0][::-1])) - c["latitude"]).argmin())
        j = int(round(c["longitude"] / 5.625)) % 64
        assert c["surface_temperature"]["value"] == slab[i, j]
        assert c["surface_temperature"]["value"] != c["lowest_air"]["value"]


def test_upper_level_cold_event_days_85_86():
    p = provider(BEXT)
    series = json.loads((BEXT / "series.json").read_text())
    for day, expected_min in [(85, 123.3), (86, 143.5)]:
        t = index(p, day)
        c = p.column(t, -19.4, 281.0)
        assert (c["latitude"], c["longitude"]) == pytest.approx((-19.3822313, 281.25))
        state = np.load(BEXT / f"states/state-d{day:05d}.npy")
        reference = synthesize_latlon(state[2 * 40 + 1], [c["latitude"]], [c["longitude"]])[0, 0]
        assert c["temperature"][1] == pytest.approx(reference, abs=1e-9)
        assert 4.5 < c["pressure"][1] < 5.5                  # level 1 is about 5 Pa
        assert series["t_min"][series["day"].index(day)] == pytest.approx(expected_min, abs=0.05)
    # The daily snapshot minimum stays above the 120.2 K per-step probe minimum,
    # which falls between saved states and is not in the archive.
    native = p.column(index(p, 85), -19.4, 281.0)["temperature"][1]
    assert 123.0 < native < 123.6
    mesh = p.slice(index(p, 85), "temperature", "native", 1)["values"].min()
    assert mesh >= native - 1.5


@pytest.mark.parametrize("root,day", [(B30, 30), (BEXT, 60), (BEXT, 90), (BEXT, 120)])
def test_equatorial_jet_matches_independent_profiles(root, day):
    if not EXT_JSON.is_file():
        pytest.skip(f"Missing independent analysis: {EXT_JSON}")
    p = provider(root)
    t = index(p, day)
    # The independent definition: cos-weighted mean of [u] over the regular
    # 3.75° latitudes with |lat| < 10°. [u](lat) is exact at any latitude.
    lat = np.linspace(-88.125, 88.125, 48)
    band = lat[np.abs(lat) < 10]
    weights = np.cos(np.radians(band))
    profiles = [np.array(p.section(t, "zonal", float(phi))["u"])[:, :-1].mean(axis=1) for phi in band]
    eq_u = (np.array(profiles) * weights[:, None]).sum(axis=0) / weights.sum()
    reference = np.array(json.loads(EXT_JSON.read_text())["daily"]["eq_u"][str(day)])
    np.testing.assert_allclose(eq_u, reference, rtol=0, atol=1e-9)
    # The viewer's own native-grid zonal mean at the two near-equatorial nodes.
    mean = p.zonal_mean(t, t)
    rows = np.abs(np.array(mean["lat"])) < 3
    native = np.array(mean["u"])[:, rows].mean(axis=1)
    assert np.argmax(native) == np.argmax(reference)
    assert abs(native.max() - reference.max()) < 3.0
