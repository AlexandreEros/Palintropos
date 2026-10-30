"""``tropoi terrain prepare``: terrain from a height expression or function.

One command, two sources, one pipeline::

    tropoi terrain prepare --expression "1200*cos(radians(lat))**2*cos(2*radians(lon))" \\
        --resolution-deg 2 --lmax 21 --solver pe --out terrain/my_planet
    tropoi terrain prepare --function ./my_terrain.py:height \\
        --resolution-deg 2 --lmax 21 --solver pe --out terrain/my_planet

Both build an :class:`~tropoi.spatial.terrain.raster.ElevationRaster` with
``ElevationRaster.from_function`` (``--expression`` through the allowlisted
:class:`~tropoi.spatial.terrain.height_function.HeightExpression`;
``--function`` through
:func:`~tropoi.spatial.terrain.height_function.load_height_function`, which
**executes the named file**), then call ``prepare_planet`` and save the
resolved terrain with ``PreparedPlanet.save``.

Sampling and validation run on the CPU (``--sample-only`` stops there and
writes nothing). The spectral preparation uses the model's own transform
and needs CUDA, like every ``prepare_planet`` call. The saved directory
keeps the resolved coefficients and the provenance (the expression text,
or the file's name and SHA-256 and the callable's name, plus the sampling
settings); it never stores or re-runs the function.

Import-light: parsing and ``--help`` import neither CuPy nor matplotlib.
"""
from __future__ import annotations

import argparse
import pathlib
import shlex
import sys

#: The planet the terrain is prepared for unless given (an Earth-sized,
#: 24 h planet). The resolved coefficients do not depend on it; previews
#: and the saved record do, and a PE run takes its planet from its own
#: options.
DEFAULT_RADIUS_EARTH_UNITS = 1.0
DEFAULT_DAY_HOURS = 24.0
DEFAULT_RESOLUTION_DEG = 1.0
DEFAULT_LMAX = 21

EPILOG = """\
examples:
  tropoi terrain prepare --expression "1000" --lmax 21 --solver pe --out terrain/plateau --sample-only
  tropoi terrain prepare --expression "where(lat > 0, 800, -400)" --resolution-deg 1 --lmax 21 --solver swe --out terrain/steps
  tropoi terrain prepare --function "./my_terrain.py:height" --resolution-deg 2 --lmax 21 --solver pe --out terrain/my_planet --figure terrain.png --globe globe.html

expressions: lat and lon in degrees (east longitude), the constant pi,
+ - * / % **, comparisons, & | ~, and the functions radians degrees sin cos
tan arcsin arccos arctan arctan2 sinh cosh tanh exp log log10 sqrt abs
minimum maximum where clip floor ceil sign hypot. Result: metres.

--function FILE.py:NAME executes FILE (trusted code only). NAME is called
once as NAME(lat_deg, lon_deg) with NumPy arrays and returns metres.
"""


def default_gauss_grid(lmax: int) -> tuple[int, int]:
    """The Gauss grid used unless --nlat/--nlon are given.

    ``nlat = 3 (l_max + 1) / 2 + 1`` rounded up to even, ``nlon = 2 nlat``
    (34 x 68 at l_max 21): the grid of the repository's terrain examples.
    """
    nlat = 3 * (int(lmax) + 1) // 2 + 1
    nlat += nlat % 2
    return nlat, 2 * nlat


def add_prepare_arguments(parser: argparse.ArgumentParser) -> None:
    source = parser.add_argument_group(
        "terrain source (exactly one)").add_mutually_exclusive_group(
            required=True)
    source.add_argument(
        "--expression", metavar="TEXT",
        help="Elevation in metres as an expression in lat and lon (degrees), "
             "e.g. \"1200*cos(radians(lat))**2\". Evaluated by an allowlisted "
             "interpreter, never as Python.")
    source.add_argument(
        "--function", metavar="FILE.py:NAME",
        help="A callable NAME(lat_deg, lon_deg) -> metres in your own Python "
             "file. Relative paths are resolved from the current directory; "
             "the text is split at its final ':'. EXECUTES THE FILE.")

    sampling = parser.add_argument_group("sampling (CPU)")
    sampling.add_argument(
        "--resolution-deg", type=float, default=DEFAULT_RESOLUTION_DEG,
        metavar="DEG",
        help="Spacing of the global cell-centre grid the source is sampled "
             "on; must divide 180 (default: %(default)s). Independent of "
             "--lmax: preparation averages a finer sample to the "
             "truncation's working resolution.")
    sampling.add_argument(
        "--lon-start-deg", type=float, default=-180.0, metavar="DEG",
        help="Western edge of the sampled longitudes: -180 passes longitudes "
             "in [-180, 180), 0 passes [0, 360) (default: %(default)s).")
    sampling.add_argument(
        "--pointwise", action="store_true",
        help="--function only: call NAME once per point with floats (for "
             "functions written with math rather than NumPy).")

    model = parser.add_argument_group("model terrain (spectral preparation, "
                                      "CUDA)")
    model.add_argument(
        "--lmax", type=int, default=DEFAULT_LMAX,
        help="Spectral truncation (default: %(default)s).")
    model.add_argument(
        "--solver", choices=("pe", "swe"), required=True,
        help="Core the terrain is for: pe band-limits it at the dealiased "
             "cut floor(2 lmax / 3) that tropoi run pe --terrain requires; "
             "swe keeps the full truncation.")
    model.add_argument(
        "--nlat", type=int, default=None,
        help="Gauss latitudes (default: 3(lmax+1)/2+1 rounded up to even).")
    model.add_argument(
        "--nlon", type=int, default=None,
        help="Longitudes (default: 2 nlat).")
    model.add_argument(
        "--radius-earth-units", type=float,
        default=DEFAULT_RADIUS_EARTH_UNITS,
        help="Planet radius recorded with the terrain and used by previews, "
             "in Earth radii (default: %(default)s). The coefficients do not "
             "depend on it.")
    model.add_argument(
        "--day-hours", type=float, default=DEFAULT_DAY_HOURS,
        help="Rotation period recorded with the terrain, in hours (default: "
             "%(default)s).")
    model.add_argument(
        "--smoothing", type=float, default=None, metavar="S",
        help="Optional spectral smoothing: amplitude kept at the band limit, "
             "0 < S <= 1 (default: none).")
    model.add_argument(
        "--clip-below-m", type=float, default=None, metavar="M",
        help="Optional floor applied to the sampled heights (metres).")

    output = parser.add_argument_group("output")
    output.add_argument(
        "--out", required=True, metavar="DIR",
        help="Directory for the prepared terrain (must not exist unless "
             "--overwrite).")
    output.add_argument(
        "--overwrite", action="store_true",
        help="Replace --out if it is an existing prepared terrain (any other "
             "directory or file is never replaced).")
    output.add_argument("--label", default=None,
                        help="Label stored with the terrain.")
    output.add_argument("--figure", metavar="PNG", default=None,
                        help="Also write the source/resolved/difference "
                             "figure.")
    output.add_argument("--globe", metavar="HTML", default=None,
                        help="Also write the offline 3-D globe page.")
    output.add_argument(
        "--sample-only", action="store_true",
        help="Sample and validate the source on the CPU, print a summary and "
             "stop: no CUDA, nothing written.")


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """``tropoi terrain <action>``."""
    actions = parser.add_subparsers(dest="terrain_action", metavar="ACTION",
                                    required=True)
    prepare = actions.add_parser(
        "prepare", help="Prepare terrain from an expression or a Python "
                        "function and save it.",
        description="Sample a height expression or a Python height function "
                    "on a global grid (CPU), resolve it on a model "
                    "discretization (CUDA) and save the prepared terrain for "
                    "tropoi run pe --terrain or a shallow-water state.",
        epilog=EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_prepare_arguments(prepare)
    prepare.set_defaults(_handler=run_prepare)


def _fail(message: str) -> int:
    print(f"tropoi terrain prepare: error: {message}", file=sys.stderr)
    return 2


def build_raster(args: argparse.Namespace):
    """The sampled source of ``args`` (CPU; raises ElevationImportError)."""
    from tropoi.spatial.terrain.height_function import (HeightFunctionError,
                                                        load_height_function)
    from tropoi.spatial.terrain.raster import ElevationRaster

    if args.expression is not None:
        if args.pointwise:
            raise HeightFunctionError("--pointwise applies to --function only")
        return ElevationRaster.from_expression(
            args.expression, resolution_deg=args.resolution_deg,
            lon_start_deg=args.lon_start_deg)
    function, record = load_height_function(args.function)
    return ElevationRaster.from_function(
        function, resolution_deg=args.resolution_deg,
        lon_start_deg=args.lon_start_deg, pointwise=args.pointwise,
        name=record["function_name"], provenance=record)


def _source_label(args) -> str:
    if args.expression is not None:
        return "expression"
    return args.function.rpartition(":")[0].replace("\\", "/") \
        .rsplit("/", 1)[-1] + ":" + args.function.rpartition(":")[2]


def run_prepare(args: argparse.Namespace) -> int:
    from tropoi.representation.archive.prepared_terrain import (
        is_prepared_terrain)
    from tropoi.spatial.terrain.raster import (ElevationImportError,
                                               working_resolution_deg)

    out = pathlib.Path(args.out)
    if args.lmax < 1:
        return _fail(f"--lmax must be >= 1, got {args.lmax}")
    if not args.sample_only and out.exists():
        if not args.overwrite:
            return _fail(f"{out} already exists; choose a new --out or pass "
                         "--overwrite to replace a prepared terrain")
        if not is_prepared_terrain(out):
            return _fail(f"refusing to overwrite {out}: it is not a prepared "
                         "terrain directory")
    try:
        raster = build_raster(args)
    except ElevationImportError as err:
        return _fail(str(err))
    print(f"sampled {raster.describe()}")
    working = working_resolution_deg(args.lmax)
    if raster.resolution_deg > working * (1 + 1e-9):
        print(f"note: the sample ({raster.resolution_deg:g} deg) is coarser "
              f"than the l_max {args.lmax} working resolution ({working:.3g} "
              "deg); features between samples are interpolated, not "
              "resolved. Use a finer --resolution-deg for small features.")
    if args.sample_only:
        print("sample-only: input valid; nothing prepared or written "
              "(preparation needs CUDA).")
        return 0

    nlat, nlon = default_gauss_grid(args.lmax)
    nlat = args.nlat if args.nlat is not None else nlat
    nlon = args.nlon if args.nlon is not None else 2 * nlat
    from tropoi.run.custom import (DiscretizationSpec, PlanetSpec,
                                   PreparationError, SpecificationError,
                                   prepare_planet)
    try:
        planet = PlanetSpec.earth_like(
            day_hours=args.day_hours,
            radius_earth_units=args.radius_earth_units)
        disc = DiscretizationSpec.gauss_latlon(args.lmax, nlat, nlon)
    except (SpecificationError, ValueError) as err:
        return _fail(str(err))
    label = args.label or f"Terrain from {_source_label(args)}"
    try:
        world = prepare_planet(planet, disc, raster, solver=args.solver,
                               smoothing=args.smoothing,
                               clip_below_m=args.clip_below_m, label=label)
    except ImportError as err:
        return _fail(f"spectral preparation needs CUDA (CuPy): {err}. "
                     "--sample-only checks the input without it.")
    except (PreparationError, SpecificationError, ValueError) as err:
        return _fail(str(err))
    try:
        saved = world.save(out, overwrite=args.overwrite)
    except (FileExistsError, OSError, ValueError) as err:
        return _fail(str(err))
    print(world.describe())
    print(f"saved terrain: {saved}")
    if args.figure:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        figure = world.plot()
        figure.savefig(args.figure, dpi=110)
        plt.close(figure)
        print(f"figure: {args.figure}")
    if args.globe:
        print(f"globe: {world.view(args.globe, open_browser=False)}")
    quoted = shlex.quote(str(saved))
    if args.solver == "pe":
        print("run over it (CUDA), with the same planet:")
        print(f"  tropoi run pe --terrain {quoted} --scenario "
              "orographic_isothermal_rest "
              f"--radius-earth-units {args.radius_earth_units:g} "
              f"--day-hours {args.day_hours:g} --days 0.1")
    else:
        print("use it in a shallow-water state (Python):")
        print(f"  PreparedPlanet.load({str(saved)!r}) as prepare_swe_state("
              "..., topography=...)")
    print("preview later without CUDA: PreparedPlanet.load(DIR).view() / "
          ".plot()")
    return 0
