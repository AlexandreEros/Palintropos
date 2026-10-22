"""Terrain import on the model grid: the Williamson-5 cone round trip,
host/GPU synthesis parity, smoothing, and an SWE run over imported terrain.
Needs CUDA.

Reference: ``Topography.williamson5_cone`` projects the analytic cone at the
state-grid points. The imported path writes the same cone to a file, reads
it back, area-averages to the working resolution and samples bilinearly
before the same projection, so the two differ by raster sampling and
interpolation only. Measured at l_max 42 on Gauss 64 x 128 (0.25 deg
NetCDF): relative coefficient difference 0.014, peak 1792 vs 1843 m.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from test_terrain_import import (cell_grid, cone, geotiff_tags, write_image,
                                 write_netcdf)


def _has_cuda():
    try:
        import cupy as cp
        return cp.is_available()
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _has_cuda(), reason="needs a CUDA GPU")

LMAX, NLAT, NLON = 42, 64, 128
#: Relative L2 difference allowed between imported and analytic terrain
#: (measured 0.014-0.016 for the three formats; see the module docstring).
ROUND_TRIP_TOL = 0.03


def _specs():
    from tropoi.run.custom import DiscretizationSpec, PlanetSpec
    from tropoi.spatial.williamson5 import W5_RADIUS_M
    return (PlanetSpec.ideal_sphere(W5_RADIUS_M, 86164.0),
            DiscretizationSpec.gauss_latlon(LMAX, NLAT, NLON))


@pytest.fixture(scope="module")
def reference():
    from tropoi.run.custom import TopographySpec, prepare_planet
    planet, disc = _specs()
    return prepare_planet(planet, disc, TopographySpec.williamson5_cone())


def _cone_files(tmp_path):
    """The analytic cone written in three formats and conventions."""
    from tropoi.spatial.williamson5 import W5_CONE_HEIGHT_M

    lat, lon = cell_grid(0.25)                       # north first, 0..360
    la, lo = np.meshgrid(lat, lon, indexing="ij")
    nc = write_netcdf(tmp_path / "cone.nc", lat, lon, cone(la, lo) / 1000.0,
                      units="km")
    lat_i = 90 - (np.arange(720) + 0.5) * 0.25      # row 0 = north
    lon_i = -180 + (np.arange(1440) + 0.5) * 0.25
    li, oi = np.meshgrid(lat_i, lon_i, indexing="ij")
    png = write_image(tmp_path / "cone.png", np.rint(
        cone(li, oi) / W5_CONE_HEIGHT_M * 65535).astype(np.uint16))
    lon_t = (np.arange(1440) + 0.5) * 0.25           # GeoTIFF from 0 E
    lt, ot = np.meshgrid(lat_i, lon_t, indexing="ij")
    tif = write_image(tmp_path / "cone.tif",
                      cone(lt, ot).astype(np.float32),
                      tiffinfo=geotiff_tags(0.0, 90.0, 0.25, 0.25))
    return {
        "netcdf": (nc, {}),
        "png16": (png, {"height_range": (0.0, W5_CONE_HEIGHT_M),
                        "lon_west_deg": -180.0}),
        "geotiff": (tif, {"units": "m"}),
    }


def _model_field(prepared):
    from tropoi.spatial.transforms.host_synthesis import synthesize_latlon
    lat = np.linspace(-90, 90, 181)
    lon = np.linspace(-180, 180, 361)
    return lat, lon, synthesize_latlon(prepared.topography_lm, lat, lon)


@pytest.mark.parametrize("fmt", ["netcdf", "png16", "geotiff"])
def test_w5_cone_round_trip(tmp_path, reference, fmt):
    from tropoi.run.custom import prepare_planet, read_elevation

    path, kwargs = _cone_files(tmp_path)[fmt]
    planet, disc = _specs()
    imported = prepare_planet(planet, disc, read_elevation(path, **kwargs))
    a, b = imported.topography_lm, reference.topography_lm
    rel = np.linalg.norm(a - b) / np.linalg.norm(b)
    assert rel < ROUND_TRIP_TOL, rel
    # Orientation and placement: the model terrain peaks at 30 N, 90 W.
    lat, lon, field = _model_field(imported)
    j, i = np.unravel_index(np.argmax(field), field.shape)
    assert abs(lat[j] - 30.0) <= 2.0 and abs(lon[i] + 90.0) <= 2.0
    # Units: same peak height as the analytic projection, within sampling.
    peak_ref = _model_field(reference)[2].max()
    assert field.max() == pytest.approx(peak_ref, rel=0.05)
    params = imported.topography.parameters
    assert imported.topography.preset == "imported"
    assert params["source_name"] == path.name
    assert params["projection_residual"] < 0.05


def test_misdeclared_longitude_is_caught_by_the_round_trip(tmp_path,
                                                           reference):
    from tropoi.run.custom import prepare_planet, read_elevation

    path, kwargs = _cone_files(tmp_path)["png16"]
    planet, disc = _specs()
    wrong = prepare_planet(planet, disc, read_elevation(
        path, **{**kwargs, "lon_west_deg": 0.0}))
    rel = (np.linalg.norm(wrong.topography_lm - reference.topography_lm)
           / np.linalg.norm(reference.topography_lm))
    assert rel > 1.0                 # a 180-degree shift is not subtle
    lat, lon, field = _model_field(wrong)
    j, i = np.unravel_index(np.argmax(field), field.shape)
    assert abs(lon[i] - 90.0) <= 2.0


def test_host_synthesis_matches_the_model_transform(reference):
    import cupy as cp

    from tropoi.cli.bve import build_planet
    from tropoi.run.custom.prepare import _base_config
    from tropoi.spatial.transforms.host_synthesis import synthesize_latlon

    planet, disc = _specs()
    model = build_planet(_base_config(planet, disc))
    lm = reference.topography_lm
    on_grid = cp.asnumpy(model.sh.inv_transform(cp.asarray(lm)).real)
    lat = np.degrees(cp.asnumpy(model.grid.latitudes))
    lon = np.degrees(cp.asnumpy(model.grid.longitudes))
    host = synthesize_latlon(lm, lat, lon)
    np.testing.assert_allclose(host, on_grid.reshape(host.shape),
                               atol=1e-9 * np.abs(on_grid).max())


def test_smoothing_and_clipping(tmp_path):
    from tropoi.run.custom import (PreparationError, TopographySpec,
                                   prepare_planet, read_elevation)
    from tropoi.spatial.terrain.topography import spectral_smoothing_factors

    path, kwargs = _cone_files(tmp_path)["netcdf"]
    raster = read_elevation(path, **kwargs)
    planet, disc = _specs()
    raw = prepare_planet(planet, disc, raster)
    smooth = prepare_planet(planet, disc, raster, smoothing=0.1)
    factors = spectral_smoothing_factors(LMAX, 0.1)
    assert factors[0] == 1.0 and factors[-1] == pytest.approx(0.1)
    np.testing.assert_allclose(smooth.topography_lm,
                               raw.topography_lm * factors[:, None],
                               atol=1e-9)
    p = smooth.topography.parameters
    assert p["smoothing_retained_at_lmax"] == 0.1
    assert p["smoothed_residual"] > p["projection_residual"]
    # Smoothing damps the far-field Gibbs ringing (measured RMS beyond 45
    # degrees from the cone: 0.37 m raw, 0.03 m at 0.1); it does NOT remove
    # the undershoot at the cone's base (-12.5 m raw, -15.1 m smoothed).
    def far_rms(prepared):
        lat, lon, field = _model_field(prepared)
        la, lo = np.radians(np.meshgrid(lat, lon, indexing="ij"))
        c30 = math.radians(30.0)
        cosd = (math.sin(c30) * np.sin(la)
                + math.cos(c30) * np.cos(la) * np.cos(lo + math.pi / 2))
        far = cosd < math.cos(math.radians(45.0))
        w = np.cos(la)[far]
        return math.sqrt(float(np.sum(w * field[far] ** 2) / np.sum(w)))
    assert far_rms(smooth) < 0.2 * far_rms(raw)
    clipped = prepare_planet(planet, disc, raster, clip_below_m=500.0)
    assert clipped.topography.parameters["sampled_min_m"] == 500.0
    with pytest.raises(PreparationError, match="imported rasters only"):
        prepare_planet(planet, disc, TopographySpec.flat(), smoothing=0.5)
    with pytest.raises(PreparationError, match="read the file first"):
        prepare_planet(planet, disc, str(path))


def test_swe_state_and_run_keep_the_imported_terrain(tmp_path):
    from tropoi.representation.archive import open_simulation
    from tropoi.run.custom import (FluidLayerSpec, PreparationError,
                                   PreparedPlanet, load_initial_state,
                                   prepare_planet, prepare_swe_state,
                                   read_elevation, run_initial_state)
    from tropoi.run.custom import DiscretizationSpec, PlanetSpec

    path, kwargs = _cone_files(tmp_path)["netcdf"]
    planet = PlanetSpec.ideal_sphere(6.37122e6, 86164.0)
    disc = DiscretizationSpec.gauss_latlon(21, 32, 64)
    world = prepare_planet(planet, disc, read_elevation(path, **kwargs),
                           label="cone from NetCDF")
    fluid = FluidLayerSpec(gravity_m_s2=9.80616, mean_depth_m=5960.0)
    prepared = prepare_swe_state(planet, disc, fluid, world,
                                 free_surface_anomaly=lambda la, lo: 0 * la)
    state = prepared.state
    assert state.topography.preset == "imported"
    np.testing.assert_array_equal(state.topography_lm, world.topography_lm)
    assert state.topography.parameters["source_sha256"] == \
        world.topography.parameters["source_sha256"]
    state.save(tmp_path / "state")
    loaded = load_initial_state(tmp_path / "state")
    assert loaded.digest == state.digest

    run_dir = run_initial_state(loaded, duration_days=0.02, n_snapshots=2,
                                out=str(tmp_path / "runs"),
                                experiment="terrain-import-test")
    sim = open_simulation(run_dir)
    config = sim.storage.run_config
    assert config["topography"] == "imported"
    assert config["topography_parameters"]["source_name"] == "cone.nc"
    rows = {row["id"]: row for row in sim.quantities()}
    assert rows["terrain"]["available"]
    assert (run_dir / "initial_state" / "topography_elevation_lm.npy"
            ).is_file()
    again = PreparedPlanet.from_initial_state(run_dir)
    np.testing.assert_array_equal(again.topography_lm, world.topography_lm)

    other = DiscretizationSpec.gauss_latlon(21, 48, 96)
    with pytest.raises(PreparationError, match="different planet"):
        prepare_swe_state(planet, other, fluid, world)
