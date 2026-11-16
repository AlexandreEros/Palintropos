"""Generate a demo planet and save a summary plot.

Invoked by ``tropoi gen``. Heavy imports (CuPy, matplotlib) happen after
argument parsing.

Grid: Gauss lat-lon by default, sized from ``--l-max``, like every other
entry point. ``--grid-resolution`` is the geodesic subdivision level, so
passing it selects the deprecated geodesic grid explicitly (PlanetViewer
renders either -- it builds a triangulation for geodesic points and uses
the structured path for lat-lon).
"""
from __future__ import annotations

import argparse
import pathlib


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--day-hours", type=float, default=24.0,
        help="Length of the sidereal day in hours (default: 24.0).")
    parser.add_argument(
        "--mass-earth-masses", type=float, default=1.0,
        help="Planet mass in Earth masses (default: 1.0).")
    parser.add_argument(
        "--radius-earth-units", "--eq_radius-earth_units",
        dest="radius_earth_units", type=float, default=1.0,
        help="Equatorial radius in Earth radii (default: 1.0). "
             "--eq_radius-earth_units is a compatibility alias.")
    parser.add_argument(
        "--l-max", type=int, default=15,
        help="Maximum spherical harmonic degree (default: 15).")
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for terrain generation (default: None).")
    parser.add_argument(
        "--output", type=str, default="planet_summary.png",
        help="Output PNG file; relative paths are written under out/ "
             "(default: planet_summary.png).")
    parser.add_argument(
        "--nlat", type=int, default=None,
        help="Gauss lat-lon latitudes [default: sized from --l-max by the "
             "3/2 quadratic rule].")
    parser.add_argument(
        "--nlon", type=int, default=None,
        help="Gauss lat-lon longitudes [default: 2 x nlat, sized from "
             "--l-max].")
    parser.add_argument(
        "--grid-resolution", type=int, default=None,
        help="Geodesic grid subdivision level. Giving this option selects "
             "the deprecated geodesic grid (which warns); the default grid "
             "is Gauss lat-lon, sized from --l-max.")


def resolve_output_path(output: str) -> pathlib.Path:
    """Relative outputs go under out/ (historical layout); absolute pass through.

    The parent directory is created, fixing the historical crash when out/
    did not exist in a fresh clone.
    """
    out_path = pathlib.Path(output)
    if not out_path.is_absolute():
        out_path = pathlib.Path("out") / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return out_path


def run(args: argparse.Namespace) -> int:
    # Heavy imports (CuPy via Planet, matplotlib via viz) after parsing.
    from tropoi.spatial.environment import PlanetaryParameters
    from tropoi.spatial.planet import Planet
    from tropoi.spatial.terrain.terrain_spectral import SpectralTerrainParams
    from tropoi.spatial.terrain.tectonics import TectonicParams
    from tropoi.representation.visual.planet_viewer import PlanetViewer

    params = PlanetaryParameters.from_earth_like(
        day_hours=args.day_hours,
        mass_earth_units=args.mass_earth_masses,
        radius_earth_units=args.radius_earth_units,
    )

    # The grid family follows the options actually given. --grid-resolution
    # is a geodesic concept, so asking for it asks for the geodesic grid;
    # with no grid option at all the default is Gauss lat-lon, sized from
    # l_max. Silently ignoring --grid-resolution would be the one
    # unacceptable outcome.
    grid_kwargs: dict = {}
    if args.grid_resolution is not None:
        if args.nlat is not None or args.nlon is not None:
            raise SystemExit(
                "--grid-resolution selects the geodesic grid and --nlat/"
                "--nlon the Gauss lat-lon grid; give one grid's options, "
                "not both")
        grid_kwargs = {"grid_type": "geodesic",
                       "grid_resolution": args.grid_resolution}
    else:
        grid_kwargs = {"grid_type": "latlon",
                       "nlat": args.nlat, "nlon": args.nlon}

    planet = Planet.generate(
        params=params,
        terrain_params=SpectralTerrainParams(seed=args.seed),
        tectonic_params=TectonicParams(),
        l_max=args.l_max,
        **grid_kwargs,
    )

    viewer = PlanetViewer(planet)
    fig = viewer.plot_summary()

    out_path = resolve_output_path(args.output)
    fig.savefig(out_path, dpi=200)
    print(f"Saved planet summary to {out_path}")
    return 0
