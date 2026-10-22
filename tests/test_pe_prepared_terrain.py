"""Primitive-equation runs over a saved prepared terrain (``--terrain``).

CPU: the prepared-terrain format (round trip, tamper and band-limit
checks), ``PreparedPlanet.save``/``load``, and ``PERunConfig`` resolution
(adopted discretization, rejected mismatches, run-configuration keys that
exclude the machine-specific path).

GPU: ``prepare_planet(solver="pe")`` band-limits at the dealiased cut; a PE
run over a prepared Gaussian mountain is bit-identical to the same run
with ``--topography mountain`` (same coefficients reach the core by the
new path); a run over imported terrain keeps exact orographic rest on the
Gauss grid, stores ``terrain/`` in its capsule, and plots from it.
"""
from __future__ import annotations

import math
import pathlib

import numpy as np
import pytest

from tropoi.spatial.specification import (DiscretizationSpec, PlanetSpec,
                                          TopographySpec)
from tropoi.spatial.truncation import product_truncation_cut


def _has_cuda():
    try:
        import cupy as cp
        return cp.is_available()
    except Exception:
        return False


requires_cuda = pytest.mark.skipif(not _has_cuda(), reason="needs a CUDA GPU")

L = 15
DISC = DiscretizationSpec.gauss_latlon(L, 24, 48)
SPEC = TopographySpec.imported({"source_name": "test.nc",
                                "source_sha256": "a" * 64,
                                "format": "netcdf"})


def _band_limited(lmax=L, cut=None):
    cut = product_truncation_cut(lmax) if cut is None else cut
    rng = np.random.default_rng(3)
    lm = np.zeros((lmax + 1, lmax + 1), dtype=np.complex128)
    for l in range(cut + 1):
        lm[l, :l + 1] = rng.normal(size=l + 1) + 1j * rng.normal(size=l + 1)
    lm[:, 0] = lm[:, 0].real
    lm[0, 0] = 500.0
    return lm * 100.0


def _save(tmp_path, *, cut="pe", lm=None, name="terrain"):
    from tropoi.run.custom import PreparedPlanet
    l_cut = product_truncation_cut(L) if cut == "pe" else cut
    lm = _band_limited() if lm is None else lm
    world = PreparedPlanet(PlanetSpec.earth_like(), DISC, SPEC, lm,
                           l_cut=l_cut, label="test terrain")
    return world, world.save(tmp_path / name)


# ---------------------------------------------------------------------------
# Format and PreparedPlanet (CPU)
# ---------------------------------------------------------------------------

def test_prepared_terrain_round_trip_and_verification(tmp_path):
    from tropoi.representation.archive.prepared_terrain import (
        PreparedTerrainError, load_prepared_terrain)
    from tropoi.run.custom import PreparedPlanet

    world, path = _save(tmp_path)
    saved = load_prepared_terrain(path)
    assert saved.l_cut == product_truncation_cut(L) == 10
    np.testing.assert_array_equal(saved.topography_lm, world.topography_lm)
    again = PreparedPlanet.load(path)
    assert again.l_cut == 10 and again.topography == SPEC
    assert again.label == "test terrain"
    with pytest.raises(FileExistsError):
        world.save(path)
    world.save(path, overwrite=True)
    # Tampering with the coefficients is detected.
    lm = np.load(path / "topography_elevation_lm.npy")
    lm[1, 0] += 1.0
    np.save(path / "topography_elevation_lm.npy", lm)
    with pytest.raises(PreparedTerrainError, match="SHA-256"):
        load_prepared_terrain(path)


def test_band_limit_and_flat_terrain_are_enforced(tmp_path):
    from tropoi.representation.archive.prepared_terrain import (
        PreparedTerrainError, SavedTerrain)
    from tropoi.run.custom import PreparedPlanet

    full = _band_limited(cut=L)
    with pytest.raises(PreparedTerrainError, match="above its band limit"):
        SavedTerrain(PlanetSpec.earth_like(), DISC, SPEC, full, l_cut=10)
    flat = PreparedPlanet(PlanetSpec.earth_like(), DISC,
                          TopographySpec.flat(), None)
    with pytest.raises(ValueError, match="flat bottom"):
        flat.save(tmp_path / "flat")


# ---------------------------------------------------------------------------
# PERunConfig (CPU)
# ---------------------------------------------------------------------------

def _resolve(**explicit):
    from tropoi.run.pe.config import PERunConfig
    base = {"scenario": "orographic_isothermal_rest", "nlev": 4,
            "dt_seconds": 300.0, "duration_days": 300.0 * 4 / 86400.0}
    return PERunConfig.resolve({**base, **explicit})


def test_terrain_resolution_adopts_the_discretization(tmp_path):
    world, path = _save(tmp_path)
    cfg = _resolve(terrain=str(path), gravity=3.71)
    assert (cfg.lmax, cfg.grid, cfg.nlat, cfg.nlon) == (L, "latlon", 24, 48)
    assert cfg.topography == "imported" and cfg.terrain_l_cut == 10
    run = cfg.to_run_config_dict()
    assert run["terrain_digest"].startswith("sha256:")
    assert run["topography_parameters"]["source_name"] == "test.nc"
    assert run["gravity"] == 3.71 and run["terrain_l_cut"] == 10
    assert str(path) not in repr(run)            # path is not identity
    assert "mountain_height_m" not in run
    assert any("prepared terrain (imported from test.nc"
               in line for line in cfg.summary_lines())
    # identical terrain in another directory: identical configuration
    _, other = _save(tmp_path, name="copy")
    run2 = _resolve(terrain=str(other), gravity=3.71).to_run_config_dict()
    assert run2 == run
    assert _resolve(terrain=str(path)).gravity == pytest.approx(9.80616)


def test_terrain_resolution_rejects_conflicts(tmp_path):
    _, path = _save(tmp_path)
    with pytest.raises(ValueError, match="resolved for lmax=15"):
        _resolve(terrain=str(path), lmax=21)
    with pytest.raises(ValueError, match="resolved for nlat=24"):
        _resolve(terrain=str(path), nlat=32)
    assert _resolve(terrain=str(path), grid="gauss-latlon").grid == "latlon"
    with pytest.raises(ValueError, match="replaces --topography"):
        _resolve(terrain=str(path), topography="mountain")
    with pytest.raises(ValueError, match="do not apply"):
        _resolve(terrain=str(path), mountain_height_m=1000.0)
    _, swe = _save(tmp_path, cut=None, lm=_band_limited(cut=L), name="swe")
    with pytest.raises(ValueError, match="solver='pe'"):
        _resolve(terrain=str(swe))
    with pytest.raises(ValueError, match="not a saved prepared terrain"):
        _resolve(terrain=str(tmp_path / "missing"))


def test_cli_accepts_terrain(tmp_path):
    import argparse
    from tropoi.cli.main import add_pe_arguments

    parser = argparse.ArgumentParser()
    add_pe_arguments(parser)
    args = parser.parse_args(["--terrain", str(tmp_path)])
    assert args.terrain == str(tmp_path)


# ---------------------------------------------------------------------------
# GPU
# ---------------------------------------------------------------------------

def _run(tmp_path, name, **explicit):
    from tropoi.cli.pe import execute_run
    out = tmp_path / name
    cfg = _resolve(out=str(out), n_snapshots=3, plots=["diagnostics"],
                   **explicit)
    assert execute_run(cfg) == 0
    (run_dir,) = [p for p in out.iterdir() if p.is_dir()]
    return cfg, run_dir


@requires_cuda
def test_prepared_mountain_matches_the_preset_path_bitwise(tmp_path):
    from tropoi.run.custom import prepare_planet

    mountain = dict(height_m=1500.0, lat_deg=25.0, lon_deg=60.0,
                    width_deg=25.0)
    world = prepare_planet(PlanetSpec.earth_like(), DISC,
                           TopographySpec.mountain(**mountain), solver="pe")
    assert world.l_cut == 10
    assert not np.any(world.topography_lm[11:]) and np.any(
        world.topography_lm[10])
    path = world.save(tmp_path / "mountain_pe")
    _, via_terrain = _run(tmp_path, "terrain", terrain=str(path))
    _, via_preset = _run(
        tmp_path, "preset", topography="mountain", grid="latlon", lmax=L,
        nlat=24, nlon=48, mountain_height_m=1500.0, mountain_lat_deg=25.0,
        mountain_lon_deg=60.0, mountain_width_deg=25.0)
    a = np.load(via_terrain / "pe_coeffs.npy")
    b = np.load(via_preset / "pe_coeffs.npy")
    assert np.array_equal(a, b)                  # same terrain, same run
    assert (via_terrain / "terrain" / "terrain.json").is_file()
    assert not (via_preset / "terrain").exists()


@requires_cuda
def test_imported_terrain_pe_run_keeps_rest_and_its_capsule(tmp_path):
    from test_terrain_import import cell_grid, cone, write_netcdf

    from tropoi.representation.archive import open_simulation
    from tropoi.representation.visual.snapshot import build_resources
    from tropoi.run.custom import (PreparedPlanet, prepare_planet,
                                   read_elevation)

    lat, lon = cell_grid(1.0)
    la, lo = np.meshgrid(lat, lon, indexing="ij")
    nc = write_netcdf(tmp_path / "cone.nc", lat, lon, cone(la, lo))
    world = prepare_planet(PlanetSpec.earth_like(), DISC, read_elevation(nc),
                           solver="pe", smoothing=0.3)
    params = world.topography.parameters
    assert params["l_cut"] == 10 and world.l_cut == 10
    assert not np.any(world.topography_lm[11:])
    path = world.save(tmp_path / "cone_pe")

    cfg, run_dir = _run(tmp_path, "runs", terrain=str(path))
    data = np.atleast_1d(np.genfromtxt(
        run_dir / "diagnostics" / "timeseries.csv", delimiter=",",
        names=True))
    assert np.all(data["max_wind_ms"] <= 1e-12)          # exact rest (GL)
    assert np.all(np.abs(data["mass_rel_drift"]) <= 1e-14)
    assert data["ps_max"][0] > data["ps_min"][0]          # terrain felt
    copy = PreparedPlanet.load(run_dir)                   # via terrain/
    np.testing.assert_array_equal(copy.topography_lm, world.topography_lm)
    sim = open_simulation(run_dir)
    assert sim.storage.run_config["terrain_digest"] == cfg.terrain_digest
    resources = build_resources(sim.storage)
    g = cfg.gravity
    import cupy as cp
    np.testing.assert_allclose(
        cp.asnumpy(resources.model.phi_surface_lm),
        g * world.topography_lm, rtol=0, atol=1e-9)
    # A capsule whose terrain/ no longer matches is refused for plotting.
    lm_path = run_dir / "terrain" / "topography_elevation_lm.npy"
    lm = np.load(lm_path)
    lm[0, 0] += 1.0
    np.save(lm_path, lm)
    with pytest.raises(Exception, match="SHA-256|terrain"):
        build_resources(open_simulation(run_dir).storage)


@requires_cuda
def test_pe_cone_measurements_describe_the_cut_coefficients():
    """The cone preset measures its full projection; after the PE cut the
    published measurements must be those of the coefficients kept."""
    from test_terrain_import import cone

    from tropoi.run.custom import prepare_planet, state_sampling
    from tropoi.spatial.transforms.host_synthesis import synthesize_latlon

    spec = TopographySpec.williamson5_cone()
    pe = prepare_planet(PlanetSpec.earth_like(), DISC, spec, solver="pe")
    swe = prepare_planet(PlanetSpec.earth_like(), DISC, spec)
    assert pe.l_cut == 10 and not np.any(pe.topography_lm[11:])
    s = state_sampling(PlanetSpec.earth_like(), DISC)
    lat = np.degrees(s.latitudes[:, 0])
    lon = np.degrees(s.longitudes[0, :])
    field = synthesize_latlon(pe.topography_lm, lat, lon)
    analytic = cone(*np.meshgrid(lat, lon, indexing="ij"))
    w = s.weights
    residual = math.sqrt(np.sum(w * (field - analytic) ** 2)
                         / np.sum(w * analytic ** 2))
    m = pe.measured
    assert m["elevation_min_m"] == pytest.approx(field.min(), abs=1e-8)
    assert m["elevation_max_m"] == pytest.approx(field.max(), abs=1e-8)
    assert m["projection_residual"] == pytest.approx(residual, rel=1e-10)
    assert m["peak_error_m"] == pytest.approx(2000.0 - field.max(), abs=1e-8)
    # ... and they are not the full-truncation values any more.
    assert m["projection_residual"] > 2 * swe.measured["projection_residual"]
    assert m["elevation_max_m"] < swe.measured["elevation_max_m"] - 100.0
    details = dict(pe.details())
    assert details["Truncation residual (rel. L2)"] == \
        f"{m['projection_residual']:.6g}"
