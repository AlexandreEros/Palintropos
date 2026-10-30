"""Three ways to supply terrain for your own planet.

    python examples/custom_topographies.py function --out out/function
    python examples/custom_topographies.py function --expression "where(lat > 0, 800, -400)" --out out/steps
    python examples/custom_topographies.py function --function ./my_terrain.py:height --out out/mine
    python examples/custom_topographies.py image --out out/image
    python examples/custom_topographies.py netcdf --out out/netcdf

Each command creates the same broad, synthetic relief by a different route
(unless you supply your own), prepares it for PE at l_max 21 (CUDA), and
writes terrain/, terrain.png and globe.html. Use --inputs-only to create or
read the input on CPU, or --file to read your own global grayscale image or
NetCDF instead of generating one. For the function route, --expression and
--function supply your own heights without editing this file (the same
inputs as ``tropoi terrain prepare``; --function executes that file).
"""
from __future__ import annotations

import argparse
import pathlib

import numpy as np


def elevation(lat_deg, lon_deg):
    """Metres on a global grid; broad features survive coarse truncation.

    cos(latitude)**2 makes longitude-dependent relief vanish at the poles.
    These low-degree modes avoid hiding tiny mountains in a huge image.
    """
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    return 1200 * np.cos(lat)**2 * np.cos(2 * lon) + 600 * np.sin(lat)


def source_raster(kind, out, *, file=None, height_range=(-2000, 2000),
                  lon_west=-180, variable=None, expression=None,
                  function=None, resolution_deg=2.0):
    """Create an example input or read a user's file; no CUDA needed."""
    from tropoi.run.custom import (ElevationRaster, encode_grey16,
                                   load_height_function, read_elevation)

    if kind == "function" and expression is not None:
        return ElevationRaster.from_expression(
            expression, resolution_deg=resolution_deg)
    if kind == "function" and function is not None:
        height, record = load_height_function(function)   # runs that file
        return ElevationRaster.from_function(
            height, resolution_deg=resolution_deg,
            name=record["function_name"], provenance=record)
    if kind == "function":
        # The built-in elevation() above, sampled on 2-degree cell centres.
        return ElevationRaster.from_function(
            elevation, resolution_deg=resolution_deg, provenance={
                "source_name": "broad synthetic hills and basins",
                "definition": "1200*cos(lat)^2*cos(2*lon)+600*sin(lat) m"})
    if file is None:
        # A 180 x 90 global plate carree grid, row 0 = north. At l_max 21
        # there is no benefit in generating a many-megapixel source here.
        lat = np.arange(89.0, -90.0, -2.0)
        lon = np.arange(-179.0, 180.0, 2.0)
        h = elevation(*np.meshgrid(lat, lon, indexing="ij"))
        if kind == "image":
            from PIL import Image
            file = out / "heightmap.png"
            # Explicit encoding, independent of the image's measured min/max:
            # black = -2000 m, white = +2000 m, with 16-bit grey values;
            # heights outside that range are refused, not wrapped.
            grey = encode_grey16(h, (-2000, 2000))
            Image.fromarray(grey).save(file)
            height_range, lon_west = (-2000, 2000), -180
        else:
            import netCDF4
            file = out / "elevation.nc"
            with netCDF4.Dataset(file, "w") as ds:
                ds.title = "Synthetic broad hills and basins"
                ds.createDimension("lat", lat.size)
                ds.createDimension("lon", lon.size)
                for name, values, units in (
                        ("lat", lat, "degrees_north"),
                        ("lon", lon, "degrees_east")):
                    v = ds.createVariable(name, "f8", (name,))
                    v[:], v.units = values, units
                v = ds.createVariable("elevation", "f8", ("lat", "lon"))
                v[:], v.units = h, "m"
        print("wrote input:", file)
    if kind == "image":
        return read_elevation(file, height_range=height_range,
                              lon_west_deg=lon_west)
    return read_elevation(file, variable=variable)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("kind", choices=("function", "image", "netcdf"))
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--file", type=pathlib.Path,
                        help="your image or NetCDF; otherwise generate one")
    parser.add_argument("--height-range", type=float, nargs=2,
                        default=(-2000, 2000), metavar=("BLACK_M", "WHITE_M"))
    parser.add_argument("--lon-west", type=float, default=-180)
    parser.add_argument("--variable", help="NetCDF elevation variable")
    own = parser.add_mutually_exclusive_group()
    own.add_argument("--expression", help="function route: metres as an "
                     "expression in lat and lon (degrees)")
    own.add_argument("--function", metavar="FILE.py:NAME",
                     help="function route: NAME(lat_deg, lon_deg) -> metres "
                          "in your own file (executes it)")
    parser.add_argument("--resolution-deg", type=float, default=2.0,
                        help="function route: sampling spacing (divides 180)")
    parser.add_argument("--lmax", type=int, default=21)
    parser.add_argument("--inputs-only", action="store_true",
                        help="create/read source on CPU; skip preparation")
    parser.add_argument("--open", action="store_true")
    args = parser.parse_args(argv)
    if args.kind == "function" and args.file is not None:
        parser.error("--file applies to image or netcdf; use --expression "
                     "or --function for the function route")
    if args.kind != "function" and (args.expression or args.function):
        parser.error("--expression and --function apply to the function "
                     "route")
    args.out.mkdir(parents=True, exist_ok=True)
    raster = source_raster(args.kind, args.out, file=args.file,
                           height_range=args.height_range,
                           lon_west=args.lon_west, variable=args.variable,
                           expression=args.expression, function=args.function,
                           resolution_deg=args.resolution_deg)
    print(raster.describe())
    if args.inputs_only:
        return 0

    import matplotlib
    matplotlib.use("Agg")
    from tropoi.run.custom import (DiscretizationSpec, PlanetSpec,
                                   prepare_planet)

    # These are example choices for a custom planet; edit them together with
    # the corresponding PE run arguments printed below.
    planet = PlanetSpec.earth_like(day_hours=30, radius_earth_units=0.8)
    nlat = 3 * (args.lmax + 1) // 2 + 1
    nlat += nlat % 2
    disc = DiscretizationSpec.gauss_latlon(args.lmax, nlat, 2 * nlat)
    world = prepare_planet(planet, disc, raster, solver="pe",
                           label=f"Custom terrain from {args.kind}")
    saved = world.save(args.out / "terrain")
    fig = world.plot()  # includes source, resolved field and their difference
    fig.savefig(args.out / "terrain.png", dpi=110)
    import matplotlib.pyplot as plt
    plt.close(fig)
    page = world.view(args.out / "globe.html", open_browser=args.open)
    print(world.describe())
    print("saved terrain:", saved)
    print("globe:", page)
    print("A short resting-atmosphere example (CUDA):")
    print(f'tropoi run pe --terrain "{saved}" '
          '--scenario orographic_isothermal_rest --day-hours 30 '
          '--radius-earth-units 0.8 --gravity 9.81 --levels 4 '
          '--dt-seconds 60 --days 0.01 --n-snapshots 2 --no-plots')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
