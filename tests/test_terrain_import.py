"""Terrain import: readers, normalization, resampling, host synthesis and
previews. CPU only (no CuPy): these run in CI.

The Williamson-5 cone (30 N, 90 W, 2000 m) is the reference relief: it is
off-centre in both coordinates, so a north-south flip, a 180-degree
longitude shift or a unit mistake moves or rescales its peak.
"""
from __future__ import annotations

import base64
import json
import math
import subprocess
import sys

import numpy as np
import pytest

from tropoi.spatial.terrain.raster import (ElevationImportError,
                                           ElevationRaster, read_elevation)
from tropoi.spatial.williamson5 import (W5_CONE_HEIGHT_M, W5_CONE_LAT_DEG,
                                        W5_CONE_RADIUS_RAD)

CONE_LON_DEG = -90.0


def cone(lat_deg, lon_deg):
    """Analytic Williamson-5 cone (m), NumPy (same formula as the model's)."""
    lat = np.radians(np.asarray(lat_deg, dtype=float))
    lon = np.radians(np.asarray(lon_deg, dtype=float))
    dlam = (lon - math.radians(CONE_LON_DEG) + math.pi) % (2 * math.pi) \
        - math.pi
    r = np.minimum(W5_CONE_RADIUS_RAD,
                   np.sqrt(dlam ** 2 + (lat - math.radians(W5_CONE_LAT_DEG))
                           ** 2))
    return W5_CONE_HEIGHT_M * (1.0 - r / W5_CONE_RADIUS_RAD)


def cell_grid(d, *, north_first=True, lon_start=0.0):
    lat = np.arange(-90 + d / 2, 90, d)
    if north_first:
        lat = lat[::-1]
    lon = lon_start + d / 2 + np.arange(int(round(360 / d))) * d
    return lat, lon


def peak(raster: ElevationRaster):
    j, i = np.unravel_index(np.argmax(raster.elevation_m),
                            raster.elevation_m.shape)
    return raster.lat_deg[j], raster.lon_deg[i], raster.elevation_m[j, i]


def write_netcdf(path, lat, lon, values, *, units="m", name="elevation",
                 order=("lat", "lon"), extra=None, pack=False, fill=None):
    netCDF4 = pytest.importorskip("netCDF4")
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("lat", lat.size)
        ds.createDimension("lon", lon.size)
        v = ds.createVariable("lat", "f8", ("lat",))
        v[:] = lat
        v.units = "degrees_north"
        v = ds.createVariable("lon", "f8", ("lon",))
        v[:] = lon
        v.units = "degrees_east"
        data = values if order == ("lat", "lon") else values.T
        if pack:
            var = ds.createVariable(name, "i2", order, fill_value=-32768)
            var.scale_factor = 0.5
            var.add_offset = 1000.0
        else:
            var = ds.createVariable(name, "f4", order,
                                    fill_value=fill if fill is not None
                                    else False)
        if units is not None:
            var.units = units
        var[:] = data
        for key, val in (extra or {}).items():
            e = ds.createVariable(key, "f4", ("lat", "lon"))
            e[:] = val
    return path


def write_image(path, array, **kwargs):
    from PIL import Image
    Image.fromarray(array).save(path, **kwargs)
    return path


def geotiff_tags(west, north, dlon, dlat, *, projected=None,
                 radius_keys=None):
    from PIL import TiffImagePlugin, TiffTags
    ifd = TiffImagePlugin.ImageFileDirectory_v2()
    ifd[33550] = (float(dlon), float(dlat), 0.0)
    ifd.tagtype[33550] = TiffTags.DOUBLE
    ifd[33922] = (0.0, 0.0, 0.0, float(west), float(north), 0.0)
    ifd.tagtype[33922] = TiffTags.DOUBLE
    if projected is None:
        keys = (1, 1, 0, 3, 1024, 0, 1, 2, 1025, 0, 1, 1, 2054, 0, 1, 9102)
    else:  # equirectangular metres on a sphere of radius `projected`
        # radius_keys replaces the 2057 entry, e.g. ((2056, 0, 1, 7030),)
        # for an ellipsoid code alone, or adds GeogLinearUnits (2052).
        radius = ((2057, 34736, 1, 0),) if radius_keys is None             else tuple(radius_keys)
        entries = ((1024, 0, 1, 1), (1025, 0, 1, 1), *radius,
                   (3075, 0, 1, 17), (3076, 0, 1, 9001))
        entries = tuple(sorted(entries))
        keys = (1, 1, 0, len(entries)) + tuple(v for e in entries for v in e)
        ifd[34736] = (float(projected),)
        ifd.tagtype[34736] = TiffTags.DOUBLE
    ifd[34735] = keys
    ifd.tagtype[34735] = TiffTags.SHORT
    return ifd


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def test_from_array_normalizes_orientation_wrap_and_seam():
    d = 2.0
    lat, lon = cell_grid(d)                        # north first, 0..360
    lon = np.r_[lon, lon[0] + 360.0]               # duplicated seam column
    la, lo = np.meshgrid(lat, lon, indexing="ij")
    r = ElevationRaster.from_array(cone(la, lo), lat, lon)
    assert r.shape == (90, 180) and r.registration == "cell"
    assert np.all(np.diff(r.lat_deg) > 0) and np.all(np.diff(r.lon_deg) > 0)
    assert r.lon_deg[0] == pytest.approx(-179.0)
    assert r.provenance["rows_flipped_to_south_first"]
    assert r.provenance["duplicate_seam_column_dropped"]
    la2, lo2 = np.meshgrid(r.lat_deg, r.lon_deg, indexing="ij")
    np.testing.assert_allclose(r.elevation_m, cone(la2, lo2), atol=1e-9)
    plat, plon, ph = peak(r)
    assert abs(plat - 30) <= d and abs(plon + 90) <= d and ph > 1800


def test_node_registered_grid_is_accepted():
    lat = np.linspace(-90, 90, 91)
    lon = np.linspace(-180, 180, 181)              # seam repeated
    la, lo = np.meshgrid(lat, lon, indexing="ij")
    r = ElevationRaster.from_array(np.sin(np.radians(la)) * 100 + 0 * lo,
                                   lat, lon)
    assert r.registration == "node" and r.shape == (91, 180)


@pytest.mark.parametrize("lat,lon,match", [
    (np.arange(-45 + 1, 45, 2.0), np.arange(1, 360, 2.0), "not a global"),
    (np.arange(-89, 90, 2.0), np.arange(1, 180, 2.0), "cover 360"),
    (np.r_[np.arange(-89, 0, 2.0), np.arange(1.5, 90, 2.0)],
     np.arange(1, 360, 2.0), "uniformly spaced"),
])
def test_non_global_or_irregular_grids_are_rejected(lat, lon, match):
    with pytest.raises(ElevationImportError, match=match):
        ElevationRaster.from_array(np.zeros((lat.size, lon.size)), lat, lon)


def test_seam_column_with_different_values_is_rejected():
    lat, lon = cell_grid(10.0)
    lon = np.r_[lon, lon[0] + 360]
    values = np.zeros((lat.size, lon.size))
    values[:, -1] = 5.0
    with pytest.raises(ElevationImportError, match="same meridian"):
        ElevationRaster.from_array(values, lat, lon)


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------

def test_coarsening_preserves_the_area_mean():
    lat, lon = cell_grid(0.5)
    la, lo = np.meshgrid(lat, lon, indexing="ij")
    r = ElevationRaster.from_array(cone(la, lo) + 300 * np.sin(np.radians(la)),
                                   lat, lon)
    c = r.coarsened(3.0)
    assert c.shape == (60, 120) and c.provenance["coarsened_to_deg"] == 3.0
    assert c.area_mean() == pytest.approx(r.area_mean(), rel=1e-12)
    assert r.coarsened(0.25) is r                  # never refines


def test_bilinear_sampling_is_exact_for_bilinear_fields_and_wraps():
    lat, lon = cell_grid(1.0)
    la, lo = np.meshgrid(lat, lon, indexing="ij")
    r = ElevationRaster.from_array(10.0 * la + 0.0 * lo, lat, lon)
    q_lat = np.array([-80.3, 0.0, 12.25, 89.0])
    np.testing.assert_allclose(r.sample(q_lat, np.array([0, 359.7, -540, 5])),
                               10.0 * q_lat, atol=1e-9)
    # longitudes wrap: 190 E == 170 W
    r2 = ElevationRaster.from_array(np.cos(np.radians(lo)) + 0 * la, lat, lon)
    assert r2.sample(10.0, 190.0) == pytest.approx(r2.sample(10.0, -170.0))
    # each pole is single-valued
    poles = r2.sample(np.full(5, 90.0), np.linspace(-180, 180, 5))
    assert np.ptp(poles) < 1e-12


def test_resample_to_points_area_averages_a_finer_source():
    lat, lon = cell_grid(0.25)
    la, lo = np.meshgrid(lat, lon, indexing="ij")
    r = ElevationRaster.from_array(cone(la, lo), lat, lon)
    vals, info = r.resample_to_points(np.radians([30.0, -30.0]),
                                      np.radians([-90.0, 90.0]),
                                      working_resolution=2.0)
    assert info["resampling"].startswith("area-weighted")
    assert info["working_resolution_deg"] == 2.0
    assert 1700 < vals[0] < W5_CONE_HEIGHT_M and vals[1] == 0.0


# ---------------------------------------------------------------------------
# NetCDF
# ---------------------------------------------------------------------------

def test_netcdf_cone_round_trip_with_units_and_orientation(tmp_path):
    lat, lon = cell_grid(1.0)                      # north first, 0..360
    la, lo = np.meshgrid(lat, lon, indexing="ij")
    path = write_netcdf(tmp_path / "cone.nc", lat, lon, cone(la, lo) / 1000,
                        units="km")
    r = read_elevation(path)
    p = r.provenance
    assert p["format"] == "netcdf" and p["variable"] == "elevation"
    assert p["units"] == "km" and p["unit_factor_to_m"] == 1000.0
    assert len(p["source_sha256"]) == 64 and p["source_name"] == "cone.nc"
    plat, plon, ph = peak(r)
    assert abs(plat - 30) <= 1 and abs(plon + 90) <= 1
    assert ph == pytest.approx(cone(plat, plon), abs=0.05)
    # (lon, lat) storage order is transposed back
    path2 = write_netcdf(tmp_path / "t.nc", lat, lon, cone(la, lo),
                         order=("lon", "lat"))
    np.testing.assert_allclose(read_elevation(path2).elevation_m,
                               r.elevation_m, atol=0.05)


def test_netcdf_cf_packing_and_missing_values(tmp_path):
    lat, lon = cell_grid(10.0)
    values = np.ma.masked_array(np.full((lat.size, lon.size), 1500.0))
    values[0, 0] = np.ma.masked
    path = write_netcdf(tmp_path / "packed.nc", lat, lon, values, pack=True)
    with pytest.raises(ElevationImportError, match="1 missing values"):
        read_elevation(path)
    r = read_elevation(path, fill_value=-5.0)
    assert r.provenance["cf_scale_factor"] == 0.5
    assert r.provenance["missing_count"] == 1
    assert r.provenance["fill_value_m"] == -5.0
    assert sorted(np.unique(r.elevation_m)) == [-5.0, 1500.0]


def test_netcdf_ambiguity_is_rejected(tmp_path):
    lat, lon = cell_grid(10.0)
    z = np.zeros((lat.size, lon.size))
    two = write_netcdf(tmp_path / "two.nc", lat, lon, z, extra={"bathy": z})
    with pytest.raises(ElevationImportError, match="pass variable="):
        read_elevation(two)
    assert read_elevation(two, variable="elevation").shape == (18, 36)
    no_units = write_netcdf(tmp_path / "nu.nc", lat, lon, z, units=None)
    with pytest.raises(ElevationImportError, match="no units attribute"):
        read_elevation(no_units)
    assert read_elevation(no_units, units="m").provenance["units"] == "m"
    with pytest.raises(ElevationImportError, match="contradicts"):
        read_elevation(two, variable="elevation", units="km")
    feet = write_netcdf(tmp_path / "ft.nc", lat, lon, z, units="ft")
    with pytest.raises(ElevationImportError, match="unsupported elevation"):
        read_elevation(feet)


def test_unit_mistakes_beyond_planetary_relief_are_rejected(tmp_path):
    lat, lon = cell_grid(2.0)
    la, lo = np.meshgrid(lat, lon, indexing="ij")
    path = write_netcdf(tmp_path / "cone_m.nc", lat, lon, cone(la, lo),
                        units=None)
    with pytest.raises(ElevationImportError, match="beyond any planetary"):
        read_elevation(path, units="km")           # metres declared as km


# ---------------------------------------------------------------------------
# Bitmaps
# ---------------------------------------------------------------------------

def _cone_image(d=1.0, lon_west=-180.0):
    lat = 90 - (np.arange(int(180 / d)) + 0.5) * d          # row 0 = north
    lon = lon_west + (np.arange(int(360 / d)) + 0.5) * d
    la, lo = np.meshgrid(lat, lon, indexing="ij")
    return np.rint(cone(la, lo) / W5_CONE_HEIGHT_M * 65535).astype(np.uint16)


def test_png16_cone_with_height_range(tmp_path):
    path = write_image(tmp_path / "cone.png", _cone_image())
    r = read_elevation(path, height_range=(0.0, W5_CONE_HEIGHT_M),
                       lon_west_deg=-180)
    assert r.provenance["image_mode"].startswith("I;16")
    plat, plon, ph = peak(r)
    assert abs(plat - 30) <= 1 and abs(plon + 90) <= 1
    assert ph == pytest.approx(cone(plat, plon), abs=0.05)
    # the same pixels declared as starting at 0 E put the peak at 90 E
    shifted = read_elevation(path, height_range=(0.0, W5_CONE_HEIGHT_M),
                             lon_west_deg=0)
    assert abs(peak(shifted)[1] - 90) <= 1


def test_bitmap_inputs_must_be_explicit(tmp_path):
    path = write_image(tmp_path / "cone.png", _cone_image(5.0))
    with pytest.raises(ElevationImportError, match="lon_west_deg"):
        read_elevation(path, height_range=(0, 1))
    with pytest.raises(ElevationImportError, match="carry no units"):
        read_elevation(path, lon_west_deg=-180)
    with pytest.raises(ElevationImportError, match="not both"):
        read_elevation(path, lon_west_deg=-180, scale=1, height_range=(0, 1))
    rgb = np.zeros((36, 72, 3), dtype=np.uint8)
    with pytest.raises(ElevationImportError, match="single-channel"):
        read_elevation(write_image(tmp_path / "rgb.png", rgb),
                       lon_west_deg=0, scale=1)
    grey = write_image(tmp_path / "g8.png", np.full((36, 72), 200, np.uint8))
    r = read_elevation(grey, lon_west_deg=0, scale=10.0, offset=-500.0,
                       nodata=255)
    assert np.all(r.elevation_m == 1500.0) and r.provenance["units"] == "m"
    with pytest.raises(ElevationImportError, match="cannot tell the format"):
        read_elevation(write_image(tmp_path / "x.dat.png", rgb[..., 0]).
                       rename(tmp_path / "x.dat"), lon_west_deg=0, scale=1)


# ---------------------------------------------------------------------------
# TIFF
# ---------------------------------------------------------------------------

def test_plain_tiff_needs_units_and_georeferencing(tmp_path):
    arr = (_cone_image(2.0).astype(np.float32) / 65535 * W5_CONE_HEIGHT_M)
    path = write_image(tmp_path / "cone.tif", arr)
    with pytest.raises(ElevationImportError, match="units are not declared"):
        read_elevation(path, lon_west_deg=-180)
    with pytest.raises(ElevationImportError, match="no georeferencing"):
        read_elevation(path, units="m")
    r = read_elevation(path, units="m", lon_west_deg=-180)
    plat, plon, _ = peak(r)
    assert abs(plat - 30) <= 2 and abs(plon + 90) <= 2


def test_geotiff_geographic_and_equirectangular(tmp_path):
    d = 2.0
    arr = _cone_image(d, lon_west=0.0).astype(np.float32)
    arr = arr / 65535 * W5_CONE_HEIGHT_M
    geo = write_image(tmp_path / "geo.tif", arr,
                      tiffinfo=geotiff_tags(0.0, 90.0, d, d))
    r = read_elevation(geo, units="m")
    assert r.provenance["georeferencing"].startswith("GeoTIFF geographic")
    plat, plon, _ = peak(r)
    assert abs(plat - 30) <= d and abs(plon + 90) <= d
    with pytest.raises(ElevationImportError, match="contradicts"):
        read_elevation(geo, units="m", lon_west_deg=-180)
    radius = 3.3962e6                       # Mars-like sphere, metres
    m_per_deg = math.pi * radius / 180
    proj = write_image(tmp_path / "eqc.tif", arr, tiffinfo=geotiff_tags(
        0.0, 90 * m_per_deg, d * m_per_deg, d * m_per_deg,
        projected=radius))
    rp = read_elevation(proj, units="m")
    assert "equirectangular" in rp.provenance["georeferencing"]
    np.testing.assert_allclose(rp.elevation_m, r.elevation_m)
    np.testing.assert_allclose(rp.lon_deg, r.lon_deg, atol=1e-9)


def test_equirectangular_geotiff_needs_an_explicit_semi_major_axis(tmp_path):
    """2056 is an ellipsoid code, never a radius; 2057 must be present and
    plausible (in metres, or kilometres via GeogLinearUnits 2052)."""
    d = 10.0
    arr = np.zeros((18, 36), dtype=np.float32)
    arr[3, 4] = 100.0
    radius = 3.3962e6
    m = math.pi * radius / 180

    def write(name, radius_keys, axis=radius):
        return write_image(tmp_path / name, arr, tiffinfo=geotiff_tags(
            0.0, 90 * m, d * m, d * m, projected=axis,
            radius_keys=radius_keys))

    only_code = write("ellipsoid.tif", ((2056, 0, 1, 7030),))
    with pytest.raises(ElevationImportError,
                       match="2056 names an ellipsoid code, not a radius"):
        read_elevation(only_code, units="m")
    missing = write("none.tif", ())
    with pytest.raises(ElevationImportError,
                       match="without an explicit sphere radius"):
        read_elevation(missing, units="m")
    # both present: 2057 is used and 2056 ignored
    both = write("both.tif", ((2056, 0, 1, 7030), (2057, 34736, 1, 0)))
    assert "3.3962e+06" in read_elevation(both, units="m").provenance[
        "georeferencing"]
    km = write("km.tif", ((2052, 0, 1, 9036), (2057, 34736, 1, 0)),
               axis=radius / 1000)
    np.testing.assert_allclose(read_elevation(km, units="m").lon_deg,
                               read_elevation(both, units="m").lon_deg)
    with pytest.raises(ElevationImportError, match="outside"):
        read_elevation(write("small.tif", None, axis=3396.2), units="m")
    with pytest.raises(ElevationImportError, match="linear unit 9002"):
        read_elevation(write("feet.tif", ((2052, 0, 1, 9002),
                                          (2057, 34736, 1, 0))), units="m")


def test_netcdf_rejects_image_only_arguments(tmp_path):
    lat, lon = cell_grid(10.0)
    path = write_netcdf(tmp_path / "z.nc", lat, lon,
                        np.zeros((lat.size, lon.size)))
    with pytest.raises(ElevationImportError,
                       match=r"height_range do\(es\) not apply to NetCDF"):
        read_elevation(path, height_range=(0.0, 1.0))
    with pytest.raises(ElevationImportError,
                       match=r"lon_west_deg do\(es\) not apply"):
        read_elevation(path, lon_west_deg=-180)
    with pytest.raises(ElevationImportError,
                       match="height_range, lon_west_deg"):
        read_elevation(path, height_range=(0, 1), lon_west_deg=0)
    assert read_elevation(path).shape == (18, 36)


# ---------------------------------------------------------------------------
# Host synthesis, specification, previews
# ---------------------------------------------------------------------------

def test_host_synthesis_matches_analytic_harmonics():
    from tropoi.spatial.transforms.host_synthesis import synthesize_latlon

    lat = np.linspace(-90, 90, 19)
    lon = np.linspace(-180, 180, 37)
    la, lo = np.meshgrid(np.radians(lat), np.radians(lon), indexing="ij")
    a = np.zeros((4, 4), dtype=complex)
    a[0, 0] = 7.0 * math.sqrt(4 * math.pi)            # constant 7
    a[1, 0] = 2.0                                     # 2 Y_10
    a[1, 1] = 1.0 - 0.5j                              # 2 Re(a Y_11)
    expected = (7.0 + 2.0 * math.sqrt(3 / (4 * math.pi)) * np.sin(la)
                + 2 * (-math.sqrt(3 / (8 * math.pi))) * np.cos(la)
                * (1.0 * np.cos(lo) + 0.5 * np.sin(lo)))
    np.testing.assert_allclose(synthesize_latlon(a, lat, lon), expected,
                               atol=1e-13)


def test_imported_topography_spec_is_a_validated_provenance_record():
    from tropoi.spatial.specification import SpecificationError, TopographySpec

    params = {"source_name": "mars.nc", "source_sha256": "ab" * 32,
              "format": "netcdf", "smoothing_retained_at_lmax": 0.3,
              "clip_below_m": None, "missing_count": 0}
    spec = TopographySpec.imported(params)
    assert TopographySpec.from_dict(spec.to_dict()) == spec
    entries = spec.run_config_entries()
    assert entries["topography"] == "imported"
    assert entries["topography_parameters"]["source_name"] == "mars.nc"
    with pytest.raises(SpecificationError, match="provenance keys"):
        TopographySpec.imported({"format": "netcdf"})
    with pytest.raises(SpecificationError, match="JSON scalar"):
        TopographySpec.imported({**params, "bad": [1, 2]})
    with pytest.raises(SpecificationError, match="not finite"):
        TopographySpec.imported({**params, "bad": float("nan")})


def _l1_preview(raster=None):
    from tropoi.representation.visual.terrain import build_terrain_preview

    a = np.zeros((4, 4), dtype=complex)
    a[1, 0] = 3000.0 / math.sqrt(3 / (4 * math.pi))  # 3 km sin(lat)
    return build_terrain_preview(
        a, radius_m=3.39e6, title="test <planet>", subtitle="sub",
        details=[("Source file", "x.nc")], raster=raster)


def test_preview_and_globe_page(tmp_path):
    from tropoi.representation.visual.terrain import (globe_payload,
                                                      write_globe_html)

    lat, lon = cell_grid(5.0)
    la, lo = np.meshgrid(lat, lon, indexing="ij")
    raster = ElevationRaster.from_array(3000 * np.sin(np.radians(la)) + 0 * lo,
                                        lat, lon)
    preview = _l1_preview(raster)
    assert preview.model_m.shape == (181, 361)
    assert preview.model_m.max() == pytest.approx(3000.0)
    assert preview.highest_point()[0] == 90.0
    assert "before truncation" in preview.source_label
    payload = globe_payload(preview)
    assert payload["units"] == "km" and payload["unit_factor"] == 1000.0
    assert [l["id"] for l in payload["layers"]] == ["model", "source"]
    assert payload["default_exaggeration"] == pytest.approx(50.0)
    model = np.frombuffer(base64.b64decode(payload["layers"][0]["data"]),
                          dtype="<f4").reshape(181, 361)
    np.testing.assert_allclose(model, preview.model_m, rtol=1e-6, atol=1e-3)
    path = write_globe_html(preview, tmp_path / "g" / "globe.html")
    page = path.read_text(encoding="utf-8")
    assert "__TERRAIN_DATA__" not in page and "__PAGE_TITLE__" not in page
    assert "<title>test &lt;planet&gt;</title>" in page
    blob = page.split('type="application/json">', 1)[1].split("</script>")[0]
    assert "<" not in blob                        # inert inside <script>
    assert json.loads(blob)["title"] == "test <planet>"
    assert "http://" not in page.replace("https://maurer", "")  # offline


def test_static_figure(tmp_path):
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    from tropoi.representation.visual.terrain import plot_terrain

    fig = plot_terrain(_l1_preview())
    assert len(fig.axes) == 3                     # map, colour bar, details
    fig.savefig(tmp_path / "t.png")
    lat, lon = cell_grid(5.0)
    la, lo = np.meshgrid(lat, lon, indexing="ij")
    raster = ElevationRaster.from_array(3000 * np.sin(np.radians(la)) + 0 * lo,
                                        lat, lon)
    fig = plot_terrain(_l1_preview(raster))
    titles = [ax.get_title(loc="left") for ax in fig.axes]
    assert any("Source raster" in t for t in titles)
    assert any("Model terrain" in t for t in titles)
    assert any("Model − source" in t for t in titles)


def test_saved_state_previews_without_cupy(tmp_path):
    """A saved SWE state over imported terrain previews on a CPU machine."""
    from tropoi.run.custom import (DiscretizationSpec, FluidLayerSpec,
                                   InitialState, PlanetSpec, TopographySpec)

    L = 7
    lm = np.zeros((L + 1, L + 1), dtype=complex)
    lm[0, 0] = 100.0 * math.sqrt(4 * math.pi)
    lm[2, 1] = 400.0 + 50j
    spec = TopographySpec.imported({
        "source_name": "fake.nc", "source_sha256": "0" * 64,
        "format": "netcdf", "projection_residual": 0.01})
    state = InitialState(
        "swe", np.zeros((3, L + 1, L + 1), dtype=complex),
        PlanetSpec.ideal_sphere(3.3962e6, 88642.66),
        DiscretizationSpec.gauss_latlon(L, 12, 24),
        fluid=FluidLayerSpec(gravity_m_s2=3.71, mean_depth_m=8000.0),
        topography=spec, topography_lm=lm, label="imported terrain")
    state.save(tmp_path / "state")
    script = (
        "import sys, matplotlib; matplotlib.use('Agg');"
        "from tropoi.run.custom import PreparedPlanet;"
        f"p = PreparedPlanet.from_initial_state(r'{tmp_path / 'state'}');"
        "print(p.topography.preset, p.topography.parameters['source_name']);"
        f"p.view(r'{tmp_path / 'g.html'}', open_browser=False);"
        f"p.plot().savefig(r'{tmp_path / 'p.png'}');"
        "print('cupy' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", script], capture_output=True,
                         text=True, check=True).stdout.split()
    assert out == ["imported", "fake.nc", "False"]
    assert (tmp_path / "g.html").stat().st_size > 10_000


# ---------------------------------------------------------------------------
# Example scripts (CPU steps)
# ---------------------------------------------------------------------------

EXAMPLES = __import__("pathlib").Path(__file__).resolve().parents[1] / \
    "examples"


def test_synthetic_example_generates_readable_files(tmp_path):
    pytest.importorskip("netCDF4")
    subprocess.run([sys.executable, str(EXAMPLES / "terrain_planet.py"),
                    "generate", str(tmp_path)], check=True,
                   capture_output=True)
    nc = read_elevation(tmp_path / "mars_like.nc")
    png = read_elevation(tmp_path / "mars_like.png",
                         height_range=(-8000.0, 22000.0), lon_west_deg=-180)
    assert nc.shape == png.shape == (720, 1440)
    step = 30000.0 / 65535
    np.testing.assert_allclose(png.elevation_m, nc.elevation_m,
                               atol=step / 2 + 0.01)


def _fake_megdr(tmp_path, ppd=1):
    """A tiny PDS3 MEGDR-like product with one peak at (20 N, 230 E)."""
    lines, samples = 180 * ppd, 360 * ppd
    lat = 90 - (np.arange(lines) + 0.5) / ppd
    lon = (np.arange(samples) + 0.5) / ppd
    la, lo = np.meshgrid(lat, lon, indexing="ij")
    heights = np.rint(-500 + 9000 * np.exp(
        -((la - 20) ** 2 + (lo - 230) ** 2) / 50.0)).astype(">i2")
    heights.tofile(tmp_path / "FAKE.IMG")
    label = f"""PDS_VERSION_ID = PDS3
^IMAGE = "FAKE.IMG"
DATA_SET_ID = "TEST"
DESCRIPTION = "The minimum and
  maximum topography observations for the entire data set are -500
  and 8500 meters."
OBJECT = IMAGE
 LINES = {lines}
 LINE_SAMPLES = {samples}
 SAMPLE_TYPE = MSB_INTEGER
 SAMPLE_BITS = 16
 UNIT = METER
END_OBJECT = IMAGE
OBJECT = IMAGE_MAP_PROJECTION
 MAP_PROJECTION_TYPE = "SIMPLE CYLINDRICAL"
 POSITIVE_LONGITUDE_DIRECTION = "EAST"
 MAP_RESOLUTION = {ppd}.0 <PIXEL/DEGREE>
 MAXIMUM_LATITUDE = 90.0 <DEGREE>
 MINIMUM_LATITUDE = -90.0 <DEGREE>
 WESTERNMOST_LONGITUDE = 0.0 <DEGREE>
 EASTERNMOST_LONGITUDE = 360.0 <DEGREE>
 COORDINATE_SYSTEM_NAME = "PLANETOCENTRIC"
END_OBJECT = IMAGE_MAP_PROJECTION
END
"""
    (tmp_path / "fake.lbl").write_text(label, encoding="ascii")
    return tmp_path / "fake.lbl", float(heights.max())


def test_mola_converter_reads_pds3_orientation_and_units(tmp_path):
    pytest.importorskip("netCDF4")
    label, highest = _fake_megdr(tmp_path)
    out = tmp_path / "fake.nc"
    done = subprocess.run([sys.executable, str(EXAMPLES / "mola_mars.py"),
                           "convert", str(label), str(out)], check=True,
                          capture_output=True, text=True)
    assert "(label: -500 .. 8500 m)" in done.stdout
    r = read_elevation(out)
    assert r.provenance["units"] == "m" and r.shape == (180, 360)
    lat, lon, h = peak(r)                  # 230 E == 130 W, north stays north
    assert abs(lat - 20) <= 1 and abs(lon + 130) <= 1
    assert h == highest
    # A label that contradicts what the converter relies on is refused.
    text = label.read_text().replace("MSB_INTEGER", "LSB_INTEGER")
    label.write_text(text)
    bad = subprocess.run([sys.executable, str(EXAMPLES / "mola_mars.py"),
                          "convert", str(label), str(out)],
                         capture_output=True, text=True)
    assert bad.returncode != 0 and "SAMPLE_TYPE" in bad.stderr
