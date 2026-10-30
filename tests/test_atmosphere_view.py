"""The 3-D atmosphere view of saved primitive-equation runs.

CPU (small synthetic fixtures; nothing here needs a GPU or a long run):

* hydrostatic geometry: an isothermal column on a NONUNIFORM sigma grid
  against the closed-form Simmons–Burridge heights, and the spectral path
  against the grid-space evaluation of the same diagnostic;
* terrain: Phi_s included exactly once (Z with terrain - Z without = h_s),
  terrain and atmosphere on the same reference sphere, one exaggeration;
* pressure p_k = sigma_k p_s and native level indexing;
* the CPU wind reconstruction and the spherical streamline integrator
  (metric factors, periodic longitude, poles);
* Held–Suarez experiment directories: read-only, legacy configuration
  defaults, statistics-only refusal, the real archive when present;
* payload bounds and default subsampling; CPU-only export in a fresh
  interpreter (no CuPy, no model); the terrain-only page unchanged.

GPU (skipped without CUDA): the host reconstruction agrees with the core's
own ``geopotential_fields``, ``wind_on_state_grid`` and surface pressure.
"""
from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import pathlib
import subprocess
import sys
import warnings

import numpy as np
import pytest

L_MAX = 10
SIGMA = (0.0, 0.1, 0.35, 0.7, 1.0)          # nonuniform, 4 full levels
NLEV = len(SIGMA) - 1
R_DRY, CP_DRY, GRAVITY = 287.04, 1004.64, 9.80616
SQ4PI = math.sqrt(4.0 * math.pi)
REPO = pathlib.Path(__file__).resolve().parents[1]
HS_RUN = REPO / "runs" / "hs-T42L20-prod-002"


def _has_cuda():
    try:
        import cupy as cp
        return cp.is_available()
    except Exception:
        return False


requires_cuda = pytest.mark.skipif(not _has_cuda(), reason="needs a CUDA GPU")


# ---------------------------------------------------------------------------
# Fixtures: nonuniform states, a PE capsule over prepared terrain, a
# Held–Suarez experiment directory
# ---------------------------------------------------------------------------

def _real_field(rng, l_max, amplitude, *, lmax_used=None, monopole=0.0):
    n = l_max + 1
    top = l_max if lmax_used is None else lmax_used
    c = np.zeros((n, n), dtype=np.complex128)
    for l in range(1, top + 1):
        c[l, :l + 1] = rng.normal(size=l + 1) + 1j * rng.normal(size=l + 1)
    c[:, 0] = c[:, 0].real
    c *= amplitude / max(np.abs(c).max(), 1e-300)
    c[0, 0] = monopole * SQ4PI
    return c


def make_frame(rng, *, l_max=L_MAX, nlev=NLEV, t_offset=0.0, top=None):
    """A nonuniform PE frame: warm anomalies, sheared vortices, divergence,
    a p_s field ranging over a few percent."""
    top = l_max - 1 if top is None else top
    n = l_max + 1
    frame = np.zeros((3 * nlev + 1, n, n), dtype=np.complex128)
    for k in range(nlev):
        frame[k] = _real_field(rng, l_max, 3e-5, lmax_used=top)
        frame[nlev + k] = _real_field(rng, l_max, 4e-6, lmax_used=top)
        frame[2 * nlev + k] = _real_field(rng, l_max, 6.0, lmax_used=top,
                                          monopole=220.0 + 15.0 * k
                                          + t_offset)
    frame[3 * nlev] = _real_field(rng, l_max, 0.02, lmax_used=top,
                                  monopole=math.log(1.0e5))
    return frame


def _band_limited_terrain(l_max=L_MAX, cut=6, scale=1500.0):
    rng = np.random.default_rng(7)
    lm = np.zeros((l_max + 1, l_max + 1), dtype=np.complex128)
    for l in range(1, cut + 1):
        lm[l, :l + 1] = rng.normal(size=l + 1) + 1j * rng.normal(size=l + 1)
    lm[:, 0] = lm[:, 0].real
    lm *= scale / np.abs(lm).max()
    lm[0, 0] = 800.0 * SQ4PI
    return lm


def write_pe_capsule(root: pathlib.Path, *, terrain=True, frames=3,
                     gravity=GRAVITY, topography=None, seed=11,
                     times=None):
    """A ``tropoi run pe`` capsule (manifest + coefficients + times), with a
    prepared terrain under terrain/ when ``terrain``."""
    from tropoi.representation.archive.schema import provenance_blocks
    from tropoi.spatial.specification import (DiscretizationSpec, PlanetSpec,
                                              TopographySpec)

    rng = np.random.default_rng(seed)
    root.mkdir(parents=True, exist_ok=True)
    coeffs = np.stack([make_frame(rng, t_offset=4.0 * i)
                       for i in range(frames)])
    times = (np.arange(frames, dtype=np.float64) * 21600.0 if times is None
             else np.asarray(times, dtype=np.float64))
    assert times.shape == (frames,)
    np.save(root / "pe_coeffs.npy", coeffs)
    np.save(root / "pe_snapshot_times.npy", times)
    config = {"solver": "pe", "lmax": L_MAX, "grid": "latlon",
              "resolution": 3, "nlat": 16, "nlon": 32, "day_hours": 24.0,
              "radius_earth_units": 1.0, "nlev": NLEV,
              "sigma_interfaces": list(SIGMA), "r_dry": R_DRY,
              "cp_dry": CP_DRY, "duration_days": 0.5, "dt_seconds": 300.0,
              "scenario": "thermal_wave", "temperature": 260.0,
              "surface_pressure": 1.0e5, "thermal_amplitude": 1.0,
              "product_quadrature": "fine", "snapshot_mode": "count",
              "n_snapshots": frames, "snapshot_times": list(times),
              "plots": [], "run_id": "pe-view-fixture"}
    lm = None
    if terrain:
        from tropoi.representation.archive.prepared_terrain import (
            SavedTerrain, save_prepared_terrain)
        lm = _band_limited_terrain()
        saved = SavedTerrain(
            PlanetSpec.earth_like(),
            DiscretizationSpec.gauss_latlon(L_MAX, 16, 32),
            TopographySpec.imported({"source_name": "fixture.nc",
                                     "source_sha256": "b" * 64,
                                     "format": "netcdf"}),
            lm, l_cut=6, label="fixture terrain")
        save_prepared_terrain(saved, root / "terrain")
        config.update(topography="imported", terrain_digest=saved.digest,
                      terrain_l_cut=6, gravity=gravity,
                      topography_parameters={"source_name": "fixture.nc"})
    elif topography is not None:
        config.update(topography)
    manifest = {"run_id": "pe-view-fixture", "status": "completed",
                "run_config": config, **provenance_blocks("pe", config)}
    (root / "manifest.json").write_text(json.dumps(manifest),
                                        encoding="utf-8")
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return root, coeffs, lm


def write_hs_experiment(root: pathlib.Path, *, days=(10, 20, 30),
                        legacy=False, states=True, l_max=L_MAX, nlev=5):
    """A Held–Suarez experiment directory in the run/held_suarez layout."""
    rng = np.random.default_rng(5)
    root.mkdir(parents=True, exist_ok=True)
    config = {"experiment_id": "hs-fixture", "l_max": l_max, "nlat": 16,
              "nlon": 32, "nlev": nlev, "dt": 1200.0, "days": max(days),
              "forcing": {"sigma_b": 0.7}, "state_every_days": 10,
              "schema": "palintropos.held_suarez.config/1",
              "radius": 6.371e6, "gravity": 9.8, "cp_dry": 1004.0,
              "kappa": 2.0 / 7.0, "grid": "latlon"}
    if legacy:                       # pre-schema-key config, defaults implied
        for key in ("schema", "radius", "gravity", "cp_dry", "kappa",
                    "grid", "state_every_days"):
            config.pop(key)
    doc = {"config": config, "config_sha256": "0" * 64,
           "created": "2026-10-01T00:00:00Z"}
    (root / "config.json").write_text(json.dumps(doc), encoding="utf-8")
    (root / "series.json").write_text("{}", encoding="utf-8")
    np.savez(root / "statistics.npz", counts=np.zeros(1))
    frames = []
    if states:
        (root / "states").mkdir()
        for i, day in enumerate(days):
            frame = make_frame(rng, l_max=l_max, nlev=nlev, t_offset=2.0 * i)
            np.save(root / "states" / f"state-d{day:05d}.npy", frame)
            frames.append(frame)
    return root, frames


def _fingerprint(root: pathlib.Path) -> str:
    h = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        st = path.stat()
        h.update(f"{path.relative_to(root)}|{st.st_size}|{st.st_mtime_ns}|"
                 .encode())
        h.update(hashlib.sha256(path.read_bytes()).digest())
    return h.hexdigest()


def _context(*, terrain=None, gravity=GRAVITY, sigma=SIGMA):
    from tropoi.representation.archive.pe_context import PEColumnContext
    return PEColumnContext(
        label="test", kind="test", l_max=L_MAX, sigma_interfaces=sigma,
        r_dry=R_DRY, cp_dry=CP_DRY, radius_m=6.371e6, gravity=gravity,
        surface_elevation_lm=terrain, terrain_label="test")


LAT = np.linspace(-90.0, 90.0, 37)
LON = np.linspace(-180.0, 180.0, 73)


# ---------------------------------------------------------------------------
# Hydrostatic geometry
# ---------------------------------------------------------------------------

def test_isothermal_column_matches_closed_form_simmons_burridge():
    from tropoi.representation.diagnostics.pe_levels import reconstruct_levels

    t0 = 250.0
    frame = np.zeros((3 * NLEV + 1, L_MAX + 1, L_MAX + 1), complex)
    frame[2 * NLEV:3 * NLEV, 0, 0] = t0 * SQ4PI
    frame[3 * NLEV, 0, 0] = math.log(1.0e5) * SQ4PI
    f = reconstruct_levels(frame, _context(), LAT, LON)
    s = SIGMA
    for k in range(NLEV):
        # alpha_1 = ln 2; alpha_k = 1 - s_{k-1/2}/ds ln(s_{k+1/2}/s_{k-1/2})
        alpha = (math.log(2.0) if k == 0 else
                 1.0 - s[k] / (s[k + 1] - s[k]) * math.log(s[k + 1] / s[k]))
        expected = R_DRY * t0 / GRAVITY * (alpha - math.log(s[k + 1]))
        z = f["geopotential_height"][k]
        assert np.allclose(z, expected, rtol=1e-12, atol=1e-9), (k, z.mean(),
                                                                 expected)
    # heights increase upward (index 0 = top) and stay above the surface
    means = f["geopotential_height"].mean(axis=(1, 2))
    assert np.all(np.diff(means) < 0) and means[-1] > 0


def test_spectral_geopotential_matches_the_grid_space_diagnostic():
    """The view's spectral path equals hydrostatic_geopotential applied to
    the synthesized temperature (the core's geopotential_fields recipe)."""
    from tropoi.representation.diagnostics.pe_levels import reconstruct_levels
    from tropoi.spatial.sigma_coordinate import (SigmaGrid,
                                                 hydrostatic_geopotential)
    from tropoi.spatial.transforms.host_synthesis import synthesize_latlon

    rng = np.random.default_rng(3)
    terrain = _band_limited_terrain()
    ctx = _context(terrain=terrain)
    frame = make_frame(rng)
    f = reconstruct_levels(frame, ctx, LAT, LON)
    T = np.stack([synthesize_latlon(frame[2 * NLEV + k], LAT, LON)
                  for k in range(NLEV)])
    phi_s = GRAVITY * synthesize_latlon(terrain, LAT, LON)
    phi, _ = hydrostatic_geopotential(SigmaGrid(SIGMA), T, phi_s, R_DRY)
    scale = np.abs(phi).max()
    assert np.abs(f["geopotential"] - phi).max() < 1e-12 * scale
    assert np.allclose(f["temperature"], T, rtol=0, atol=1e-10)


def test_terrain_geopotential_is_included_exactly_once():
    from tropoi.representation.diagnostics.pe_levels import reconstruct_levels
    from tropoi.spatial.transforms.host_synthesis import synthesize_latlon

    rng = np.random.default_rng(4)
    frame = make_frame(rng)
    terrain = _band_limited_terrain()
    flat = reconstruct_levels(frame, _context(), LAT, LON, winds=False)
    hilly = reconstruct_levels(frame, _context(terrain=terrain), LAT, LON,
                               winds=False)
    h = synthesize_latlon(terrain, LAT, LON)
    np.testing.assert_allclose(hilly.surface_height, h, atol=1e-9)
    for k in range(NLEV):
        diff = hilly["geopotential_height"][k] - flat["geopotential_height"][k]
        np.testing.assert_allclose(diff, h, atol=1e-8)   # once, not twice
    # absolute: Phi_K - Phi_s is the bottom half layer, alpha_K R T_K
    from tropoi.spatial.sigma_coordinate import SigmaGrid
    alpha = SigmaGrid(SIGMA).alpha[-1]
    np.testing.assert_allclose(
        hilly["geopotential"][-1] - GRAVITY * h,
        alpha * R_DRY * hilly["temperature"][-1], rtol=1e-12, atol=1e-8)


def test_pressure_is_sigma_times_local_surface_pressure_on_each_level():
    from tropoi.representation.diagnostics.pe_levels import reconstruct_levels

    frame = np.zeros((3 * NLEV + 1, L_MAX + 1, L_MAX + 1), complex)
    for k in range(NLEV):                       # level-identifying T_k
        frame[2 * NLEV + k, 0, 0] = (200.0 + 10.0 * k) * SQ4PI
    # ln p_s = ln 1e5 + 0.03 sin(lat): Y_10 = sqrt(3/4pi) sin(lat)
    frame[3 * NLEV, 0, 0] = math.log(1.0e5) * SQ4PI
    frame[3 * NLEV, 1, 0] = 0.03 / math.sqrt(3.0 / (4.0 * math.pi))
    f = reconstruct_levels(frame, _context(), LAT, LON, levels=[3, 0, 2])
    assert f.levels == (3, 0, 2)
    full = [0.5 * (SIGMA[k] + SIGMA[k + 1]) for k in range(NLEV)]
    assert f.sigma == (full[3], full[0], full[2])
    ps = np.broadcast_to(1.0e5 * np.exp(0.03 * np.sin(np.radians(LAT)))[:, None],
                         (LAT.size, LON.size))
    np.testing.assert_allclose(f.surface_pressure, ps, rtol=1e-12)
    for i, k in enumerate((3, 0, 2)):
        np.testing.assert_allclose(f["pressure"][i], full[k] * ps, rtol=1e-12)
        np.testing.assert_allclose(f["temperature"][i], 200.0 + 10.0 * k,
                                   atol=1e-9)
    # not one global pressure per level: p varies with p_s along the surface
    assert np.ptp(f["pressure"][0]) > 0.05 * full[3] * 1.0e5
    with pytest.raises(IndexError):
        reconstruct_levels(frame, _context(), LAT, LON, levels=[NLEV])


def test_flat_run_without_gravity_needs_one_and_recorded_gravity_is_kept():
    from tropoi.representation.archive.pe_context import PEContextError
    from tropoi.representation.diagnostics.pe_levels import reconstruct_levels

    frame = make_frame(np.random.default_rng(1))
    bare = _context(gravity=None)
    with pytest.raises(ValueError, match="gravity"):
        reconstruct_levels(frame, bare, LAT, LON)
    with_g = bare.with_gravity(9.80665)
    assert "caller" in with_g.provenance["gravity"]
    reconstruct_levels(frame, with_g, LAT, LON, winds=False)
    with pytest.raises(PEContextError, match="contradict"):
        _context().with_gravity(9.8)
    with pytest.raises(PEContextError, match="gravity"):
        _context(terrain=_band_limited_terrain(), gravity=None)


# ---------------------------------------------------------------------------
# Winds and streamlines
# ---------------------------------------------------------------------------

def test_host_wind_matches_solid_body_rotation_and_finite_differences():
    from tropoi.spatial.transforms.host_synthesis import (
        synthesize_latlon, synthesize_wind_latlon)

    R, U0 = 6.371e6, 20.0
    z = np.zeros((L_MAX + 1, L_MAX + 1), complex)
    d = np.zeros_like(z)
    z[1, 0] = 2.0 * U0 / (R * math.sqrt(3.0 / (4.0 * math.pi)))
    u, v = synthesize_wind_latlon(z, d, R, LAT, LON)
    np.testing.assert_allclose(
        u, np.broadcast_to(U0 * np.cos(np.radians(LAT))[:, None], u.shape),
        atol=1e-5)                            # pole limit: O(1e-7) relative
    assert np.abs(v).max() == 0.0
    rng = np.random.default_rng(9)
    z, d = _real_field(rng, L_MAX, 1e-5), _real_field(rng, L_MAX, 1e-6)
    lat, lon = np.array([-61.0, -5.0, 37.0, 74.0]), np.array([-179.0, 3.0, 91.0])
    l = np.arange(L_MAX + 1)[:, None]
    inv = np.zeros((L_MAX + 1, 1))
    inv[1:] = -R ** 2 / (l[1:] * (l[1:] + 1.0))
    psi, chi = z * inv, d * inv
    h = 1e-4

    def ddlat(c):
        return (synthesize_latlon(c, lat + h, lon)
                - synthesize_latlon(c, lat - h, lon)) / math.radians(2 * h)

    def ddlon(c):
        return (synthesize_latlon(c, lat, lon + h)
                - synthesize_latlon(c, lat, lon - h)) / math.radians(2 * h)

    c = np.cos(np.radians(lat))[:, None]
    u_fd = -ddlat(psi) / R + ddlon(chi) / (R * c)
    v_fd = ddlon(psi) / (R * c) + ddlat(chi) / R
    u, v = synthesize_wind_latlon(z, d, R, lat, lon)
    assert np.abs(u - u_fd).max() < 1e-8 * np.abs(u).max()
    assert np.abs(v - v_fd).max() < 1e-8 * np.abs(v).max()


def _rotation_wind(axis, lat, lon, speed=20.0):
    la, lo = np.meshgrid(np.radians(lat), np.radians(lon), indexing="ij")
    x = np.stack([np.cos(la) * np.sin(lo), np.sin(la),
                  np.cos(la) * np.cos(lo)], -1)
    w = speed * np.cross(np.asarray(axis, float), x)
    east = np.stack([np.cos(lo), 0 * lo, -np.sin(lo)], -1)
    north = np.stack([-np.sin(la) * np.sin(lo), np.cos(la),
                      -np.sin(la) * np.cos(lo)], -1)
    return (w * east).sum(-1), (w * north).sum(-1)


@pytest.mark.parametrize("axis", [(0, 1, 0), (1, 0, 0), (0.3, 0.2, 0.93)])
def test_streamlines_of_rigid_rotations_are_circles_about_the_axis(axis):
    """Rotation about any axis: streamlines keep x . axis constant, cross the
    poles and the seam without breaks."""
    from tropoi.representation.visual.sphere_streamlines import (
        integrate_streamlines)

    lat = np.linspace(-90.0, 90.0, 91)
    lon = np.linspace(-180.0, 180.0, 181)
    u, v = _rotation_wind(axis, lat, lon)
    lines = integrate_streamlines(u, v, lat, lon, step_deg=1.5, max_steps=60)
    assert len(lines) > 120
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    seam = pole = 0
    for la, lo in lines:
        x = np.stack([np.cos(np.radians(la)) * np.sin(np.radians(lo)),
                      np.sin(np.radians(la)),
                      np.cos(np.radians(la)) * np.cos(np.radians(lo))], -1)
        assert np.ptp(x @ a) < 2e-3                       # stays on its circle
        step = np.degrees(np.arccos(np.clip((x[1:] * x[:-1]).sum(-1), -1, 1)))
        assert np.all(step < 1.5 * 1.01)                  # no jumps anywhere
        seam += int(np.any(np.abs(np.diff(lo)) > 180.0))
        pole += int(np.abs(la).max() > 85.0)
    if abs(a[0]) < 0.9:      # circles about the x axis never meet lon = 180
        assert seam > 0                                   # crossed lon = 180
    if abs(a[1]) < 0.5:
        assert pole > 0                                   # crossed a pole


def test_streamlines_follow_the_flow_direction_and_skip_calm_air():
    from tropoi.representation.visual.sphere_streamlines import (
        integrate_streamlines)

    lat = np.linspace(-90.0, 90.0, 61)
    lon = np.linspace(-180.0, 180.0, 121)
    u = 10.0 * np.cos(np.radians(lat))[:, None] * np.ones(lon.size)
    v = np.zeros_like(u)
    lines = integrate_streamlines(u, v, lat, lon, seeds=([20.0], [0.0]))
    (la, lo), = lines
    assert np.all(np.diff(np.unwrap(np.radians(lo))) > 0)    # eastward order
    assert integrate_streamlines(1e-6 * u, v, lat, lon) == []


# ---------------------------------------------------------------------------
# Archives: PE capsules and Held–Suarez experiments, read-only
# ---------------------------------------------------------------------------

def test_pe_capsule_context_reads_prepared_terrain_and_constants(tmp_path):
    from tropoi.representation.archive import open_simulation
    from tropoi.spatial.environment import PlanetaryParameters

    root, _, lm = write_pe_capsule(tmp_path / "pe")
    ctx = open_simulation(root).storage.pe_context()
    assert ctx.sigma_interfaces == SIGMA and ctx.gravity == GRAVITY
    assert ctx.radius_m == PlanetaryParameters.from_earth_like(
        day_hours=24.0, radius_earth_units=1.0).radius
    np.testing.assert_array_equal(ctx.surface_elevation_lm, lm)
    np.testing.assert_array_equal(ctx.surface_geopotential_lm(),
                                  GRAVITY * lm)


def test_mountain_preset_terrain_is_not_invented_on_the_cpu(tmp_path):
    from tropoi.representation.archive import open_simulation
    from tropoi.representation.archive.pe_context import PEContextError

    root, _, _ = write_pe_capsule(
        tmp_path / "m", terrain=False,
        topography={"topography": "mountain", "mountain_height_m": 1000.0,
                    "mountain_lat_deg": 30.0, "mountain_lon_deg": 0.0,
                    "mountain_width_deg": 20.0, "gravity": GRAVITY})
    with pytest.raises(PEContextError, match="never saved"):
        open_simulation(root).storage.pe_context()


def test_held_suarez_experiment_opens_read_only_with_day_times(tmp_path):
    from tropoi.representation.archive import open_simulation

    root, frames = write_hs_experiment(tmp_path / "hs")
    before = _fingerprint(root)
    sim = open_simulation(root)
    assert sim.solver == "pe" and len(sim) == 3
    np.testing.assert_array_equal(sim.times, [864000.0, 1728000.0, 2592000.0])
    np.testing.assert_array_equal(sim[1].packed, frames[1])
    assert sim[0].state["temperature"].coeffs.shape == (5, L_MAX + 1,
                                                        L_MAX + 1)
    ctx = sim.storage.pe_context()
    assert ctx.is_flat and ctx.gravity == 9.8
    assert ctx.r_dry == pytest.approx(2.0 / 7.0 * 1004.0)
    assert ctx.sigma_interfaces == tuple(k / 5 for k in range(6))
    with pytest.raises(Exception, match="no snapshot figure"):
        sim[0].plot(tmp_path / "x.png")
    out = tmp_path / "views" / "hs.html"
    path = sim.view(out, open_browser=False, resolution_deg=10.0)
    assert path == out and out.stat().st_size > 0
    snap = sim[2].view(tmp_path / "views" / "one.html", open_browser=False,
                       resolution_deg=10.0)
    doc = _page_payload(snap)
    assert [t["index"] for t in doc["atmosphere"]["times"]] == [2]
    with pytest.raises(ValueError, match="inside the run directory"):
        sim.view(root / "view.html", open_browser=False)
    assert _fingerprint(root) == before
    assert sorted(p.name for p in root.iterdir()) == [
        "config.json", "series.json", "states", "statistics.npz"]


def test_legacy_held_suarez_config_takes_the_documented_defaults(tmp_path):
    from tropoi.representation.archive import open_simulation

    root, _ = write_hs_experiment(tmp_path / "old", legacy=True)
    storage = open_simulation(root).storage
    assert set(storage.defaulted) >= {"radius", "gravity", "cp_dry", "kappa"}
    ctx = storage.pe_context()
    assert (ctx.radius_m, ctx.gravity, ctx.cp_dry) == (6.371e6, 9.8, 1004.0)
    assert "default" in ctx.provenance["gravity"]


def test_reader_defaults_match_the_experiment_configuration():
    pytest.importorskip("cupy", reason="run.held_suarez imports the forcing "
                                       "hooks, which need CuPy")
    import dataclasses

    from tropoi.representation.archive.held_suarez import CONFIG_DEFAULTS
    from tropoi.run.held_suarez.config import HeldSuarezConfig

    defaults = {f.name: f.default for f in dataclasses.fields(HeldSuarezConfig)}
    for key, value in CONFIG_DEFAULTS.items():
        assert defaults[key] == value, key


def test_statistics_only_experiments_are_explained_not_invented(tmp_path):
    from tropoi.representation.archive import CapsuleError, open_simulation
    from tropoi.representation.archive.held_suarez import NoSnapshotsError

    root, _ = write_hs_experiment(tmp_path / "stats", states=False)
    sim = open_simulation(root)
    assert len(sim) == 0
    assert "accumulated statistics only" in sim.metadata["statistics_only"]
    with pytest.raises(NoSnapshotsError, match="no time series is made"):
        sim.view(tmp_path / "v.html", open_browser=False)
    ref = tmp_path / "reference"
    ref.mkdir()
    np.savez(ref / "series.npz", day=np.arange(3))
    (ref / "meta.json").write_text("{}", encoding="utf-8")
    with pytest.raises(CapsuleError, match="accumulated statistics only"):
        open_simulation(ref)


@pytest.mark.skipif(not (HS_RUN / "states").is_dir(),
                    reason="the real Held–Suarez run is not on this machine")
def test_real_held_suarez_archive_views_without_change(tmp_path):
    from tropoi.representation.archive import open_simulation

    before = _fingerprint_meta(HS_RUN)
    sim = open_simulation(HS_RUN)
    assert len(sim) == 120 and sim.times[-1] == 1200 * 86400.0
    assert sim.storage.pe_context().nlev == 20
    path = sim.view(tmp_path / "hs.html", open_browser=False, times=[0, -1],
                    levels=[0, 10, 19], resolution_deg=3.0)
    doc = _page_payload(path)["atmosphere"]
    assert [t["label"] for t in doc["times"]] == [
        "day 10 (t = 864000 s)", "day 1200 (t = 103680000 s)"]
    assert [lv["sigma"] for lv in doc["levels"]] == [0.025, 0.525, 0.975]
    assert _fingerprint_meta(HS_RUN) == before


def _fingerprint_meta(root):
    h = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        st = path.stat()
        h.update(f"{path.relative_to(root)}|{st.st_size}|{st.st_mtime_ns}"
                 .encode())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Payload: geometry, ranges, budget
# ---------------------------------------------------------------------------

def _page_payload(path) -> dict:
    text = pathlib.Path(path).read_text(encoding="utf-8")
    start = text.index('<script id="terrain-data" type="application/json">')
    start = text.index(">", start) + 1
    end = text.index("</script>", start)
    return json.loads(text[start:end])


def _u16(b64):
    return np.frombuffer(base64.b64decode(b64), dtype="<u2")


def test_payload_geometry_shares_radius_datum_and_exaggeration(tmp_path):
    from tropoi.representation.archive import open_simulation
    from tropoi.representation.diagnostics.pe_levels import reconstruct_levels
    from tropoi.representation.visual.atmosphere import atmosphere_payload

    root, coeffs, lm = write_pe_capsule(tmp_path / "pe")
    storage = open_simulation(root).storage
    page, report = atmosphere_payload(storage, resolution_deg=5.0,
                                      exaggeration=30.0)
    atm = page["atmosphere"]
    assert page["radius_m"] == atm["radius_m"] == report["context"].radius_m
    assert page["default_exaggeration"] == 30.0
    mesh = atm["mesh"]
    lat = mesh["lat0"] + mesh["dlat"] * np.arange(mesh["nlat"])
    lon = mesh["lon0"] + mesh["dlon"] * np.arange(mesh["nlon"])
    n = lat.size * lon.size
    heights = _u16(atm["height"]["data"]).reshape(-1, n)
    nl = len(atm["levels"])
    assert heights.shape[0] == len(atm["times"]) * nl == 3 * NLEV
    for it, index in enumerate(report["times"]):
        exact = reconstruct_levels(coeffs[index], report["context"], lat, lon,
                                   winds=False)
        for il in range(nl):
            s = it * nl + il
            lo, hi = atm["height"]["bounds"][s]
            z = lo + heights[s] * (hi - lo) / 65535.0
            np.testing.assert_allclose(
                z, exact["geopotential_height"][il].ravel(),
                atol=(hi - lo) / 65535.0)
            # the layer sits above the terrain it is computed from
            assert np.all(exact["geopotential_height"][il]
                          > exact.surface_height)
    # terrain layer of the same page: h_s in metres on its own mesh
    from tropoi.spatial.transforms.host_synthesis import synthesize_latlon
    t_lat = page["lat0"] + page["dlat"] * np.arange(page["nlat"])
    t_lon = page["lon0"] + page["dlon"] * np.arange(page["nlon"])
    terrain = np.frombuffer(base64.b64decode(page["layers"][0]["data"]),
                            dtype="<f4").reshape(page["nlat"], page["nlon"])
    np.testing.assert_allclose(terrain, synthesize_latlon(lm, t_lat, t_lon),
                               atol=1e-3)
    # geometry responds to state and level: slices differ
    assert np.ptp([b[0] for b in atm["height"]["bounds"]]) > 1000.0


def test_colour_limits_are_fixed_and_diverging_fields_are_symmetric(tmp_path):
    from tropoi.representation.archive import open_simulation
    from tropoi.representation.visual.atmosphere import atmosphere_payload

    root, _, _ = write_pe_capsule(tmp_path / "pe")
    page, _ = atmosphere_payload(open_simulation(root).storage,
                                 resolution_deg=5.0)
    fields = {f["id"]: f for f in page["atmosphere"]["fields"]}
    assert list(fields) == ["temperature", "pressure", "zeta", "delta"]
    for name in ("zeta", "delta"):
        lo, hi = fields[name]["range"]
        assert lo == -hi > -np.inf and fields[name]["scale"] == "diverging"
        for a, b in fields[name]["level_ranges"]:
            assert a == -b
    for name in ("temperature", "pressure"):
        f = fields[name]
        assert f["scale"] == "sequential"
        lo, hi = f["range"]
        assert all(lo <= a and b <= hi for a, b in f["bounds"])
    assert fields["pressure"]["units"] == "hPa"


def test_default_subsampling_is_announced_and_explicit_overflow_refused(
        tmp_path):
    from tropoi.representation.archive import open_simulation
    from tropoi.representation.visual.atmosphere import (
        AtmosphereViewWarning, PayloadTooLargeError, _terrain_bytes,
        atmosphere_payload, estimate_payload_bytes)

    root, _ = write_hs_experiment(tmp_path / "hs",
                                  days=tuple(range(10, 130, 10)))
    storage = open_simulation(root).storage
    per_time = estimate_payload_bytes(1, 5, 37, 73) - 200_000
    fixed = 200_000 + _terrain_bytes(L_MAX)
    budget_mb = (fixed + 2.5 * per_time) / 1e6               # two times fit
    with pytest.warns(AtmosphereViewWarning, match="exported 2 of 12"):
        page, report = atmosphere_payload(storage, resolution_deg=5.0,
                                          max_mb=budget_mb)
    assert report["times"] == [0, 11]                   # first and last
    assert "evenly subsampled" in page["atmosphere"]["selection_note"]
    encoded = len(json.dumps(page["atmosphere"]).encode())
    assert encoded <= report["estimate_bytes"]
    with pytest.raises(PayloadTooLargeError, match="max_mb"):
        atmosphere_payload(storage, resolution_deg=5.0, max_mb=budget_mb,
                           times="all")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        page, report = atmosphere_payload(storage, resolution_deg=5.0,
                                          times=[3, -1], levels=[4, 0],
                                          max_mb=None)
    assert report["times"] == [3, 11] and report["levels"] == [0, 4]
    assert [lv["number"] for lv in page["atmosphere"]["levels"]] == [1, 5]


SEL_FIXED, SEL_PER_TIME = 200_000, 1_000_000      # simple byte costs
SEL_TIMES3 = np.array([0.0, 86400.0, 172800.0])


@pytest.mark.parametrize("count, budget", [
    (3, 600_000),             # no state fits at all
    (3, 1_500_000),           # one fits, but the endpoints need two
    (1, 600_000),             # single-snapshot run, its one state too big
])
def test_automatic_selection_refuses_budgets_below_its_minimum(count, budget):
    from tropoi.representation.visual.atmosphere import (
        PayloadTooLargeError, _select_times)

    with pytest.raises(PayloadTooLargeError, match="max_mb"):
        _select_times(SEL_TIMES3[:count], count, None, budget, SEL_PER_TIME,
                      SEL_FIXED)


@pytest.mark.parametrize("count, budget, expected", [
    (1, SEL_FIXED + SEL_PER_TIME, [0]),              # exact fit, one state
    (3, SEL_FIXED + 2 * SEL_PER_TIME, [0, 2]),       # exact fit, endpoints
    (3, SEL_FIXED + 3 * SEL_PER_TIME, [0, 1, 2]),    # exact fit, all
])
def test_automatic_selection_accepts_exact_fits(count, budget, expected):
    from tropoi.representation.visual.atmosphere import _select_times

    chosen, note = _select_times(SEL_TIMES3[:count], count, None, budget,
                                 SEL_PER_TIME, SEL_FIXED)
    assert chosen == expected
    assert SEL_FIXED + SEL_PER_TIME * len(chosen) <= budget
    assert (note is None) == (len(chosen) == count)


def test_automatic_selection_keeps_unbounded_and_explicit_single_state():
    from tropoi.representation.visual.atmosphere import _select_times

    assert _select_times(SEL_TIMES3, 3, None, None, SEL_PER_TIME,
                         SEL_FIXED) == ([0, 1, 2], None)
    # one state of a longer run stays available, explicitly
    assert _select_times(SEL_TIMES3, 3, [-1], 1_500_000, SEL_PER_TIME,
                         SEL_FIXED) == ([2], None)


def test_automatic_export_refuses_a_budget_without_room_for_endpoints(
        tmp_path):
    from tropoi.representation.archive import open_simulation
    from tropoi.representation.visual.atmosphere import (
        PayloadTooLargeError, _terrain_bytes, atmosphere_payload,
        estimate_payload_bytes)

    per_time = estimate_payload_bytes(1, 5, 37, 73) - 200_000
    fixed = 200_000 + _terrain_bytes(L_MAX)
    budget_mb = (fixed + 1.5 * per_time) / 1e6       # one state fits
    many, _ = write_hs_experiment(tmp_path / "many", days=(10, 20, 30))
    with pytest.raises(PayloadTooLargeError, match="first and last of 3"):
        atmosphere_payload(open_simulation(many).storage,
                           resolution_deg=5.0, max_mb=budget_mb)
    one, _ = write_hs_experiment(tmp_path / "one", days=(10,))
    with pytest.raises(PayloadTooLargeError, match="single saved state"):
        atmosphere_payload(open_simulation(one).storage, resolution_deg=5.0,
                           max_mb=(fixed + 0.5 * per_time) / 1e6)
    # the single state fits once the bound allows it
    page, report = atmosphere_payload(open_simulation(one).storage,
                                      resolution_deg=5.0, max_mb=budget_mb)
    assert report["times"] == [0]
    assert report["estimate_bytes"] <= budget_mb * 1e6


def test_view_refuses_other_solvers_and_flat_runs_without_gravity(tmp_path):
    from tropoi.representation.archive import open_simulation

    root, _, _ = write_pe_capsule(tmp_path / "flat", terrain=False)
    sim = open_simulation(root)
    with pytest.raises(ValueError, match="records no gravity"):
        sim.view(tmp_path / "a.html", open_browser=False)
    path = sim.view(tmp_path / "b.html", open_browser=False, gravity=9.80665,
                    resolution_deg=10.0)
    assert "supplied by the caller" in json.dumps(_page_payload(path))


def test_bve_runs_have_no_atmosphere_view(tmp_path):
    from tropoi.representation.archive import open_simulation
    from tropoi.representation.archive.schema import provenance_blocks

    root = tmp_path / "bve"
    root.mkdir()
    config = {"lmax": 3, "grid": "geodesic", "resolution": 2, "nlat": 8,
              "nlon": 16, "day_hours": 24.0, "radius_earth_units": 1.0,
              "duration_days": 1.0, "dt_snapshots": 3600.0, "scenario": "t",
              "viscosity": 0.0, "product_quadrature": "fine",
              "snapshot_mode": "count", "n_snapshots": 1,
              "snapshot_times": [0.0], "plots": []}
    np.save(root / "vorticity_coeffs.npy", np.zeros((1, 4, 4), complex))
    np.save(root / "bve_snapshot_times.npy", np.zeros(1))
    (root / "manifest.json").write_text(json.dumps(
        {"run_id": "b", "run_config": config,
         **provenance_blocks("bve", config)}), encoding="utf-8")
    with pytest.raises(ValueError, match="primitive-equation"):
        open_simulation(root).view(tmp_path / "x.html", open_browser=False)


# ---------------------------------------------------------------------------
# CPU only, and the terrain-only page unchanged
# ---------------------------------------------------------------------------

def test_export_is_cpu_only_in_a_fresh_interpreter(tmp_path):
    root, _, _ = write_pe_capsule(tmp_path / "pe")
    hs, _ = write_hs_experiment(tmp_path / "hs")
    code = (
        "import sys, warnings\n"
        "from tropoi.representation.archive import open_simulation\n"
        f"for run in ({str(root)!r}, {str(hs)!r}):\n"
        "    sim = open_simulation(run)\n"
        f"    sim.view({str(tmp_path / 'out')!r} + '/' + run[-2:] + "
        "'.html', open_browser=False, resolution_deg=10.0)\n"
        "banned = [m for m in sys.modules if m.split('.')[0] in "
        "('cupy', 'cupyx') or m.startswith(('tropoi.temporal.tendencies', "
        "'tropoi.spatial.planet', 'tropoi.spatial.grids', "
        "'tropoi.spatial.operators', 'tropoi.spatial.spherical_backend', "
        "'tropoi.temporal.semi_implicit', 'tropoi.run', 'tropoi.cli'))]\n"
        "assert not banned, banned\n"
        "print('ok')\n")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    result = subprocess.run([sys.executable, "-c", code], capture_output=True,
                            text=True, timeout=300, env=env)
    assert result.returncode == 0 and "ok" in result.stdout, (
        result.stdout + result.stderr)
    assert len(list((tmp_path / "out").glob("*.html"))) == 2


def test_open_stays_light_for_held_suarez_experiments(tmp_path):
    hs, _ = write_hs_experiment(tmp_path / "hs")
    code = (
        "import sys\n"
        "from tropoi.representation.archive import open_simulation\n"
        f"sim = open_simulation({str(hs)!r}); sim[-1].state['ln_ps']\n"
        "bad = [m for m in sys.modules if m.split('.')[0] in ('cupy', "
        "'matplotlib') or m.startswith(('tropoi.representation.visual', "
        "'tropoi.spatial.transforms', 'tropoi.temporal.tendencies'))]\n"
        "assert not bad, bad\n")
    result = subprocess.run([sys.executable, "-c", code], capture_output=True,
                            text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr


def test_terrain_only_page_is_unchanged_by_the_atmosphere_option(tmp_path):
    from tropoi.representation.visual.terrain import (build_terrain_preview,
                                                      globe_payload,
                                                      write_globe_html,
                                                      write_globe_page)

    preview = build_terrain_preview(_band_limited_terrain(), radius_m=6.371e6,
                                    title="t <b>", details=[("k", "v")])
    payload = globe_payload(preview)
    assert "atmosphere" not in payload
    a = write_globe_html(preview, tmp_path / "a.html")
    b = write_globe_page(payload, preview.title, tmp_path / "b.html")
    assert a.read_bytes() == b.read_bytes()
    html = a.read_text(encoding="utf-8")
    for element in ("atmSection", "atmCbarWrap", "mapSource", "interpWrap"):
        tag = html[html.index(f'id="{element}"') - 120:
                   html.index(f'id="{element}"') + 80]
        assert "hidden" in tag, element
    assert "<title>t &lt;b&gt;</title>" in html
    assert _page_payload(a) == json.loads(json.dumps(payload))


# ---------------------------------------------------------------------------
# GPU parity with the core's own diagnostics
# ---------------------------------------------------------------------------

@requires_cuda
def test_host_reconstruction_matches_the_core_on_its_state_grid():
    import cupy as cp

    from tropoi.representation.diagnostics.pe_levels import reconstruct_levels
    from tropoi.spatial.environment import PlanetaryParameters
    from tropoi.spatial.planet import Planet
    from tropoi.spatial.sigma_coordinate import SigmaGrid
    from tropoi.temporal.tendencies.primitive_equations import (
        PrimitiveEquationsModel, PrimitiveEquationsState)

    params = PlanetaryParameters.ideal_sphere(6.371e6, 86400.0)
    planet = Planet.generate(params=params, grid_type="latlon", nlat=16,
                             nlon=32, l_max=L_MAX, product_quadrature="fine")
    terrain = _band_limited_terrain()
    model = PrimitiveEquationsModel(
        planet, SigmaGrid(SIGMA), r_dry=R_DRY, cp_dry=CP_DRY,
        surface_geopotential_lm=cp.asarray(GRAVITY * terrain))
    frame = make_frame(np.random.default_rng(21))    # degree l_max row zero
    state = PrimitiveEquationsState(cp.asarray(frame))
    grid = planet.grid
    lat = np.degrees(grid._latitudes_np)
    lon = np.degrees(grid._longitudes_np)
    from tropoi.representation.archive.pe_context import PEColumnContext
    ctx = PEColumnContext(label="gpu", kind="test", l_max=L_MAX,
                          sigma_interfaces=SIGMA, r_dry=R_DRY, cp_dry=CP_DRY,
                          radius_m=float(params.radius), gravity=GRAVITY,
                          surface_elevation_lm=terrain, terrain_label="t")
    host = reconstruct_levels(frame, ctx, lat, lon)
    shape = (NLEV, lat.size, lon.size)
    phi = cp.asnumpy(model.geopotential_fields(state)["phi_full"]).reshape(shape)
    u, v = (cp.asnumpy(x).reshape(shape)
            for x in model.wind_on_state_grid(state))
    ps = cp.asnumpy(model.surface_pressure_on_state_grid(state)).reshape(
        shape[1:])
    assert np.abs(host["geopotential"] - phi).max() < 1e-9 * np.abs(phi).max()
    assert np.abs(host["u"] - u).max() < 1e-9 * np.abs(u).max()
    assert np.abs(host["v"] - v).max() < 1e-9 * np.abs(v).max()
    np.testing.assert_allclose(host.surface_pressure, ps, rtol=1e-12)


def test_documented_example_runs_on_a_held_suarez_directory(tmp_path):
    hs, _ = write_hs_experiment(tmp_path / "hs")
    out = tmp_path / "page.html"
    result = subprocess.run(
        [sys.executable, str(REPO / "examples" / "atmosphere_view.py"),
         str(hs), "--out", str(out), "--times", "0", "-1", "--levels", "0",
         "4", "--resolution", "10", "--field", "zeta"],
        capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "wrote" in result.stdout and out.is_file()
    doc = _page_payload(out)["atmosphere"]
    assert [t["index"] for t in doc["times"]] == [0, 2]
    assert doc["defaults"]["field"] == "zeta"
