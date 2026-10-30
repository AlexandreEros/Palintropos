"""Terrain example input semantics and the offline, CPU-only Mars quick start."""
from __future__ import annotations

import json
import pathlib
import runpy
import subprocess
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_mars_quickstart_without_cuda_or_raw_data(tmp_path):
    script = ROOT / "examples/mola_mars.py"
    # A fresh interpreter blocks CUDA imports; cwd has no raw MOLA files.
    code = ("import runpy, sys\nsys.modules['cupy'] = None\n"
            "sys.modules['netCDF4'] = None\n"
            f"sys.argv = [{str(script)!r}, 'quickstart', '--out', 'mars']\n"
            f"runpy.run_path({str(script)!r}, run_name='__main__')\n")
    result = subprocess.run([sys.executable, "-c", code], cwd=tmp_path,
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    from tropoi.run.custom import PreparedPlanet, read_elevation
    from tropoi.spatial.transforms.host_synthesis import synthesize_latlon

    out = tmp_path / "mars"
    world = PreparedPlanet.load(out / "terrain")
    assert world.l_cut == 14 and world.discretization.lmax == 21
    settings = json.loads((out / "mars_heightmap.json").read_text())
    raster = read_elevation(out / "mars_heightmap.png",
                             **settings["read_elevation"])
    assert raster.shape == (90, 180)
    # Checks sign, units, encoding, longitude origin and north/south order
    # against the original coefficients, not the exported colour preview.
    expected = synthesize_latlon(world.topography_lm,
                                 raster.lat_deg, raster.lon_deg)
    np.testing.assert_allclose(raster.elevation_m, expected, rtol=0,
                               atol=10000 / 65535 + 1e-8)
    assert (out / "mars_globe.html").stat().st_size > 10000
    assert (out / "mars_terrain.png").stat().st_size > 1000


def test_function_image_and_netcdf_describe_the_same_terrain(tmp_path):
    import pytest
    pytest.importorskip("netCDF4")
    example = runpy.run_path(str(ROOT / "examples/custom_topographies.py"))
    read = example["source_raster"]
    function = read("function", tmp_path)
    image = read("image", tmp_path)
    netcdf = read("netcdf", tmp_path)
    for raster in (image, netcdf):
        np.testing.assert_array_equal(raster.lat_deg, function.lat_deg)
        np.testing.assert_array_equal(raster.lon_deg, function.lon_deg)
    np.testing.assert_allclose(image.elevation_m, function.elevation_m,
                               rtol=0, atol=2000 / 65535 + 1e-8)
    np.testing.assert_array_equal(netcdf.elevation_m, function.elevation_m)
