"""Load a planet's terrain from an elevation file and look at it.

    python examples/terrain_planet.py generate terrain              # CPU: synthetic Mars-like files
    python examples/terrain_planet.py show terrain/mars_like.nc     # CUDA: resolve, figure + globe
    python examples/terrain_planet.py show terrain/mars_like.png --height-range -8000 22000 --lon-west -180
    python examples/terrain_planet.py state terrain/mars_like.nc states/mars_like   # CUDA: SWE state
    python examples/terrain_planet.py preview states/mars_like      # CPU only: figure + globe

``generate`` writes a SYNTHETIC Mars-like relief (a north-south dichotomy,
a Tharsis-like rise with four volcanoes, a Hellas-like basin; clearly not
MOLA data) as NetCDF (metres) and as a 16-bit PNG, so the whole file path
can be exercised without downloading anything. ``show`` reads any
supported file, resolves it on a Mars-sized planet at the chosen truncation
and writes a static figure and an interactive globe (one self-contained
HTML file, opened in the browser). ``state`` stores the resolved terrain in
a shallow-water initial state (a resting layer), the form a run keeps it
in; ``preview`` draws a saved state's terrain on a machine without CUDA.

Real Mars: see docs/TERRAIN_IMPORT.md ("Mars from MOLA") for downloading
the USGS MOLA DEM and turning it into a global latitude-longitude file, then
for example ``show mola_4ppd.tif --units m --smoothing 0.3``.
"""
from __future__ import annotations

import argparse
import math
import pathlib

import numpy as np

MARS_RADIUS_M = 3.3962e6
MARS_DAY_S = 88642.66
MARS_MASS_KG = 6.417e23
MARS_GRAVITY = 3.71


def mars_like_relief(lat_deg, lon_deg) -> np.ndarray:
    """Synthetic Mars-like elevation (m). Shapes only resemble Mars."""
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)

    def dist(lat0, lon0):  # great-circle distance (deg) from a centre
        a, b = math.radians(lat0), math.radians(lon0)
        c = np.sin(a) * np.sin(lat) + np.cos(a) * np.cos(lat) * np.cos(lon - b)
        return np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))

    # Dichotomy: low northern plains, high southern highlands, a boundary
    # tilted so it crosses the equator.
    tilt = 10.0 * np.sin(lon - math.radians(-30.0))
    h = 2500.0 * np.tanh((5.0 - lat_deg - tilt) / 12.0) - 500.0
    h += 4500.0 * np.exp(-(dist(0.0, -105.0) / 28.0) ** 2)      # rise
    for lat0, lon0, top, width in ((18.4, -134.0, 19000.0, 3.2),
                                   (-8.4, -120.5, 12000.0, 2.0),
                                   (-0.8, -112.6, 11000.0, 1.8),
                                   (11.8, -104.5, 12000.0, 1.8)):
        h += top * np.exp(-(dist(lat0, lon0) / width) ** 2)     # volcanoes
    h -= 7000.0 * np.exp(-(dist(-42.4, 70.5) / 13.0) ** 4)     # basin
    return h


def generate(out_dir: str) -> None:
    from PIL import Image

    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    d = 0.25
    lat = 90.0 - (np.arange(int(180 / d)) + 0.5) * d      # row 0 = north
    lon = -180.0 + (np.arange(int(360 / d)) + 0.5) * d
    h = mars_like_relief(*np.meshgrid(lat, lon, indexing="ij"))
    lo, hi = -8000.0, 22000.0
    png = out / "mars_like.png"
    Image.fromarray(np.rint((h - lo) / (hi - lo) * 65535).astype(np.uint16)
                    ).save(png)
    print(f"wrote {png} (16-bit grey; --height-range {lo:g} {hi:g} "
          "--lon-west -180)")
    try:
        import netCDF4
    except ImportError:
        print("netCDF4 not installed: skipped the NetCDF file "
              "(pip install netCDF4)")
        return
    nc = out / "mars_like.nc"
    with netCDF4.Dataset(nc, "w") as ds:
        ds.title = "SYNTHETIC Mars-like relief (not MOLA data)"
        ds.createDimension("lat", lat.size)
        ds.createDimension("lon", lon.size)
        v = ds.createVariable("lat", "f8", ("lat",))
        v[:], v.units = lat, "degrees_north"
        v = ds.createVariable("lon", "f8", ("lon",))
        v[:], v.units = lon, "degrees_east"
        v = ds.createVariable("elevation", "f4", ("lat", "lon"), zlib=True)
        v[:], v.units = h, "m"
    print(f"wrote {nc} ({lat.size} x {lon.size}, metres)")


def _specs(lmax: int):
    from tropoi.run.custom import DiscretizationSpec, PlanetSpec

    planet = PlanetSpec.oblate(equatorial_radius_m=MARS_RADIUS_M,
                               sidereal_day_s=MARS_DAY_S,
                               mass_kg=MARS_MASS_KG)
    nlat = 3 * (lmax + 1) // 2 + 1
    nlat += nlat % 2
    return planet, DiscretizationSpec.gauss_latlon(lmax, nlat, 2 * nlat)


def _read(args):
    from tropoi.run.custom import read_elevation

    return read_elevation(
        args.file, variable=args.variable, units=args.units,
        scale=args.scale, offset=args.offset,
        height_range=tuple(args.height_range) if args.height_range else None,
        lon_west_deg=args.lon_west, fill_value=args.fill_value)


def _render(world, stem: pathlib.Path, open_browser: bool) -> None:
    import matplotlib
    if not open_browser:
        matplotlib.use("Agg")
    fig = world.plot()                                   # static figure
    figure = stem.with_name(stem.name + "_terrain.png")
    fig.savefig(figure, dpi=110)
    print("figure:", figure)
    page = world.view(stem.with_name(stem.name + "_globe.html"),
                      open_browser=open_browser)         # interactive globe
    print("globe: ", page)


def show(args) -> None:
    from tropoi.run.custom import prepare_planet

    raster = _read(args)
    print(raster.describe())
    planet, disc = _specs(args.lmax)
    world = prepare_planet(planet, disc, raster, smoothing=args.smoothing,
                           label=f"{pathlib.Path(args.file).name} on a "
                                 f"Mars-sized planet")
    for key, value in world.details():
        print(f"  {key:<32}{value}")
    out = pathlib.Path(args.out or pathlib.Path(args.file).parent)
    out.mkdir(parents=True, exist_ok=True)
    _render(world, out / pathlib.Path(args.file).stem, not args.no_browser)


def state(args) -> None:
    from tropoi.run.custom import (FluidLayerSpec, prepare_planet,
                                   prepare_swe_state)

    planet, disc = _specs(args.lmax)
    world = prepare_planet(planet, disc, _read(args),
                           smoothing=args.smoothing)
    # A resting layer deep enough to cover the highest resolved peak.
    peak = float(world.topography.parameters["elevation_max_m"])
    fluid = FluidLayerSpec(gravity_m_s2=MARS_GRAVITY,
                           mean_depth_m=math.ceil(peak / 1000 + 5) * 1000)
    prepared = prepare_swe_state(planet, disc, fluid, world,
                                 free_surface_anomaly=lambda la, lo: 0 * la,
                                 label="resting layer over imported terrain")
    print("\n".join(prepared.report.lines()))
    print("saved", prepared.save(args.state, overwrite=True))


def preview(args) -> None:
    from tropoi.run.custom import PreparedPlanet

    world = PreparedPlanet.from_initial_state(args.state)
    print(world.describe())
    path = pathlib.Path(args.state)   # outputs go beside, not into, the state
    _render(world, path.parent / path.name, not args.no_browser)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="step", required=True)
    gen = sub.add_parser("generate", help="write synthetic Mars-like files")
    gen.add_argument("out_dir")
    for name in ("show", "state"):
        p = sub.add_parser(name)
        p.add_argument("file", help="NetCDF, PNG/bitmap or TIFF elevation")
        if name == "state":
            p.add_argument("state", help="initial-state directory to write")
        p.add_argument("--lmax", type=int, default=42)
        p.add_argument("--smoothing", type=float, default=None,
                       help="amplitude kept at l_max (0 < s <= 1)")
        p.add_argument("--variable")
        p.add_argument("--units")
        p.add_argument("--scale", type=float)
        p.add_argument("--offset", type=float)
        p.add_argument("--height-range", type=float, nargs=2)
        p.add_argument("--lon-west", type=float)
        p.add_argument("--fill-value", type=float)
        if name == "show":
            p.add_argument("--out", help="output directory")
            p.add_argument("--no-browser", action="store_true")
    pre = sub.add_parser("preview", help="draw a saved state's terrain (CPU)")
    pre.add_argument("state")
    pre.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv)
    {"generate": lambda a: generate(a.out_dir), "show": show,
     "state": state, "preview": preview}[args.step](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
