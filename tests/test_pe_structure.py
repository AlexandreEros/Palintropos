"""Columns, sections, zonal means and mass streamfunction of saved PE states."""
import json
import math
from pathlib import Path

import numpy as np
import pytest

from tropoi.representation.archive.pe_context import PEColumnContext
from tropoi.representation.diagnostics.held_suarez_sections import (
    reduce_sections, select_samples, state_grid)
from tropoi.representation.diagnostics.pe_levels import reconstruct_levels
from tropoi.representation.diagnostics.pe_structure import (
    ZonalMeanAccumulator, column, mass_streamfunction, vertical_section)
from tropoi.spatial.sigma_coordinate import SigmaGrid
from tropoi.spatial.transforms.host_synthesis import synthesize_latlon, synthesize_wind_latlon

ROOT = Path(__file__).resolve().parents[1]
BEN_A = ROOT / "runs/dev-thai-ben2-ck-heated-T21L40-f44c039-layer-linear-dt600-30d"
BEN_EXT = ROOT / "runs/dev-thai-ben2-ck-heated-T21L40-f44c039-layer-linear-dt600-spongeB-120d-from-d030"
HS_FULL = ROOT / "runs/hs-T42L20-prod-002"
Y00 = math.sqrt(4 * math.pi)
N10 = math.sqrt(3 / (4 * math.pi))
RADIUS, GRAVITY, P0 = 6.371e6, 9.81, 1.0e5


def _context(nlev=4, lmax=6):
    return PEColumnContext("synthetic", "test", lmax, SigmaGrid.uniform(nlev).interfaces,
                           287.0, 1004.0, RADIUS, GRAVITY, None, "flat")


def _resting_frame(context, temperature=250.0):
    nlev, n = context.nlev, context.l_max + 1
    frame = np.zeros((3 * nlev + 1, n, n), dtype=complex)
    frame[2 * nlev:3 * nlev, 0, 0] = temperature * Y00
    frame[3 * nlev, 0, 0] = math.log(P0) * Y00
    return frame


def _archive(root):
    if not root.is_dir():
        pytest.skip(f"Missing archive: {root}")
    document = json.loads((root / "config.json").read_text(encoding="utf-8"))
    from tropoi.representation.archive.radiative import RadiativeStorage
    return RadiativeStorage(root, document)


def test_analytic_streamfunction_constant_surface_pressure():
    """Divergence c Y_1^0 in layer 2 gives v = -a c N cos(lat) / 2 exactly."""
    ctx = _context()
    frame = _resting_frame(ctx)
    c = 2.0e-6
    frame[ctx.nlev + 2, 1, 0] = c
    acc = ZonalMeanAccumulator(ctx, 12, 24)
    acc.add(frame)
    means = acc.means()
    lat = np.radians(acc.lat_deg)
    v = -RADIUS * c * N10 * np.cos(lat) / 2
    np.testing.assert_allclose(means["v"][2], v, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(means["ps_v"][2], P0 * v, rtol=1e-12, atol=1e-9)
    psi = mass_streamfunction(means["ps_v"], ctx.sigma_interfaces, acc.lat_deg, RADIUS, GRAVITY)
    expected = 2 * math.pi * RADIUS * np.cos(lat) / GRAVITY * P0 * v * 0.25
    np.testing.assert_allclose(psi[:3], 0, atol=1e-6)
    for row in psi[3:]:
        np.testing.assert_allclose(row, expected, rtol=1e-12)
    # Southward flow below the top layers: psi < 0, i.e. net southward above.
    assert (psi[-1][1:-1] < 0).all()


def test_streamfunction_uses_product_mean_not_product_of_means():
    """p_s and v share a wavenumber-1 anomaly, so [p_s v] != [p_s][v]."""
    ctx = _context(nlev=2, lmax=6)
    frame = _resting_frame(ctx)
    # ln p_s anomaly 0.05 cos(lat) cos(lon): coefficient of Y_1^1 (real part).
    n11 = -math.sqrt(3 / (8 * math.pi))
    frame[3 * ctx.nlev, 1, 1] = 0.05 / (2 * n11)
    frame[0, 2, 1] = 3.0e-5j         # psi ~ -sin(lon), so v ~ cos(lon), in phase with p_s
    acc = ZonalMeanAccumulator(ctx, 16, 32)
    acc.add(frame)
    means = acc.means()
    fine = np.arange(720) * 0.5
    p_s = np.exp(synthesize_latlon(frame[-1], acc.lat_deg, fine))
    _, v = synthesize_wind_latlon(frame[0], frame[ctx.nlev], RADIUS, acc.lat_deg, fine)
    np.testing.assert_allclose(means["ps_v"][0], (p_s * v).mean(axis=1), rtol=1e-7, atol=1e-6)
    product_of_means = p_s.mean(axis=1) * v.mean(axis=1)
    assert np.abs(means["ps_v"][0] - product_of_means).max() > 100 * np.abs(product_of_means).max() + 1
    np.testing.assert_allclose(means["v"][0], 0, atol=1e-12)


def test_streamfunction_rejects_non_sigma_columns():
    with pytest.raises(ValueError):
        mass_streamfunction(np.zeros((2, 3)), [0.1, 0.5, 1.0], [0, 1, 2], RADIUS, GRAVITY)
    with pytest.raises(ValueError):
        mass_streamfunction(np.zeros((3, 3)), [0, 0.5, 1.0], [0, 1, 2], RADIUS, GRAVITY)


def test_column_and_sections_are_exact_point_synthesis():
    ctx = _context(nlev=3, lmax=8)
    rng = np.random.default_rng(4)
    frame = _resting_frame(ctx)
    frame[:, 1:, :] += 1e-7 * (rng.normal(size=frame[:, 1:, :].shape) + 1j * rng.normal(size=frame[:, 1:, :].shape))
    frame[2 * ctx.nlev:3 * ctx.nlev, 2:, :2] *= 1e7
    lat, lon = -19.4, 281.0
    reference = reconstruct_levels(frame, ctx, [lat], [lon], None)
    profile = column(frame, ctx, lat, lon - 360)
    for name in ("temperature", "u", "v", "pressure", "geopotential_height"):
        np.testing.assert_array_equal(profile[name], reference[name][:, 0, 0])
    assert profile["longitude"] == lon and profile["surface_height"] == 0.0
    lats = np.linspace(-90, 90, 7)
    meridional = vertical_section(frame, ctx, longitude=lon, points=lats)
    grid = reconstruct_levels(frame, ctx, lats, [lon], None)
    np.testing.assert_array_equal(meridional["temperature"], grid["temperature"][:, :, 0])
    zonal = vertical_section(frame, ctx, latitude=lat, points=[0, 90, 180, 270, 360])
    np.testing.assert_allclose(zonal["u"][:, 0], zonal["u"][:, -1], rtol=0, atol=1e-12)
    np.testing.assert_allclose(zonal["u"][:, 0], reconstruct_levels(
        frame, ctx, [lat], [0.0], None)["u"][:, 0, 0], rtol=0, atol=1e-14)
    with pytest.raises(ValueError):
        vertical_section(frame, ctx, latitude=0, longitude=0, points=lats)


@pytest.mark.parametrize("root,window", [(BEN_A, (15, 30)), (BEN_EXT, (60, 120))])
def test_zonal_means_match_run_accumulator(root, window):
    """means.npz was accumulated on the GPU from the same daily states."""
    storage = _archive(root)
    cfg = storage.run_config
    selection = select_samples(storage.days, {}, interval=window, min_states=1)
    with np.load(root / "means.npz") as saved:
        count = int(saved["count"])
        reference = {k: saved[k].T for k in ("u", "v", "T")}
        np.testing.assert_allclose(saved["lat"], state_grid(cfg["nlat"], cfg["nlon"])[0], atol=1e-9)
    assert selection.count == count
    acc = ZonalMeanAccumulator(storage.context.with_gravity(cfg["gravity"]), cfg["nlat"], cfg["nlon"])
    for index in selection.indices:
        acc.add(storage.frame(index))
    means = acc.means()
    np.testing.assert_allclose(means["u"], reference["u"], rtol=0, atol=2e-9)
    np.testing.assert_allclose(means["v"], reference["v"], rtol=0, atol=2e-9)
    np.testing.assert_allclose(means["temperature"], reference["T"], rtol=0, atol=2e-9)


def test_zonal_means_match_held_suarez_reduction():
    if not HS_FULL.is_dir():
        pytest.skip(f"Missing archive: {HS_FULL}")
    from tropoi.representation.archive.capsule import open_storage
    storage = open_storage(HS_FULL)
    ctx = storage.pe_context()
    cfg = storage.run_config
    days = [int(t // 86400) for t in storage.times]
    selection = select_samples(days, cfg, interval=(days[-3], days[-1]), min_states=2)
    lat, lon, _ = state_grid(cfg["nlat"], cfg["nlon"])
    frames = [np.asarray(storage.frame(i)) for i in selection.indices]
    reference = reduce_sections(frames, selection, nlev=ctx.nlev, l_max=ctx.l_max, nominal_truncation=None,
                                lat_deg=lat, lon_deg=lon, sigma=ctx.full_levels,
                                sigma_interfaces=ctx.sigma_interfaces, radius_m=ctx.radius_m)
    acc = ZonalMeanAccumulator(ctx, cfg["nlat"], cfg["nlon"])
    for frame in frames:
        acc.add(frame)
    means = acc.means()
    for mine, theirs in (("u", "u"), ("v", "v"), ("temperature", "T")):
        np.testing.assert_allclose(means[mine], reference[theirs].T, rtol=0, atol=1e-10)
