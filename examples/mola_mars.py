"""Real Mars terrain: MOLA MEGDR topography through the terrain-import path.

    python examples/mola_mars.py convert megt90n000cb.lbl mola/megt_4ppd.nc    # CPU
    python examples/mola_mars.py validate mola/megt_4ppd.nc [--fine mola/megt_16ppd.nc]
    python examples/mola_mars.py show mola/megt_4ppd.nc --lmax 42               # CUDA
    python examples/mola_mars.py prepare-pe mola/megt_4ppd.nc terrain/mola_pe   # CUDA
    tropoi run pe --terrain terrain/mola_pe --scenario orographic_isothermal_rest \
        --radius-earth-units 0.53307 --day-hours 24.6229 --gravity 3.71 \
        --r-dry 191.8 --cp-dry 735 --temperature 210 --surface-pressure 610 \
        --levels 4 --dt-seconds 300 --days 0.25

This is REAL data, kept apart from examples/terrain_planet.py (synthetic
Mars-like relief). Inputs are the Mars Global Surveyor MOLA Mission
Experiment Gridded Data Records (MEGDR) from the NASA PDS Geosciences Node,
dataset MGS-M-MOLA-5-MEGDR-L3-V1.0:

    https://pds-geosciences.wustl.edu/mgs/mgs-m-mola-5-megdr-l3-v1/mgsl_300x/
      meg004/megt90n000cb.img + .lbl   (4 px/deg, 1440 x 720, 2.1 MB)
      meg016/megt90n000eb.img + .lbl   (16 px/deg, 5760 x 2880, 33 MB)

Each sample is the median MOLA topography (planetary radius minus areoid
radius, metres) in its bin. The grids are raw PDS3 images (big-endian
int16), which tropoi does not read directly. ``convert`` parses the PDS3
label, checks every property it relies on, and writes a CF NetCDF file that
``read_elevation`` reads like any other.

``validate`` measures what the import path does with it (orientation against
named landmarks, the hemispheric offset, independence from source
resolution, resolved peaks per truncation); docs/TERRAIN_IMPORT.md records
the numbers. ``show`` writes the static figure and the interactive globe.
``prepare-pe`` resolves the terrain for the primitive-equation core
(band-limited at the dealiased cut) on a Mars-sized planet, saves it, and
draws it; ``tropoi run pe --terrain`` then runs over exactly that terrain.
The PE run above is an orographic-rest balance check with Mars-like
constants (CO2 gas constants, 210 K, 610 Pa), not a Mars climate model.
"""
from __future__ import annotations

import argparse
import hashlib
import math
import pathlib
import re

import numpy as np

#: Landmarks (planetocentric degrees, east longitude), for orientation
#: checks only: (name, lat, lon_east, sign) with sign +1 for a summit and -1
#: for a basin floor.
LANDMARKS = (
    ("Olympus Mons", 18.65, 226.2, +1),
    ("Ascraeus Mons", 11.8, 255.5, +1),
    ("Arsia Mons", -8.3, 239.9, +1),
    ("Elysium Mons", 24.8, 146.9, +1),
    ("Hellas Planitia", -42.4, 70.5, -1),
)
MARS_RADIUS_M = 3.3962e6
MARS_DAY_S = 88642.66
MARS_MASS_KG = 6.417e23


# ---------------------------------------------------------------------------
# PDS3 MEGDR -> NetCDF
# ---------------------------------------------------------------------------

def _label_value(text: str, key: str) -> str:
    match = re.search(rf"^\s*{key}\s*=\s*(.+?)\s*$", text, re.MULTILINE)
    if not match:
        raise ValueError(f"PDS3 label has no {key}")
    return match.group(1).strip().strip('"')


def _number(text: str, key: str) -> float:
    return float(_label_value(text, key).split()[0])


def convert(label_path: str, out_path: str) -> None:
    import netCDF4

    label = pathlib.Path(label_path)
    text = label.read_text(encoding="ascii", errors="replace")
    image = label.with_name(_label_value(text, r"\^IMAGE"))
    if not image.exists():
        image = image.with_name(image.name.lower())
    lines = int(_number(text, "LINES"))
    samples = int(_number(text, "LINE_SAMPLES"))
    checks = {
        "SAMPLE_TYPE": "MSB_INTEGER", "SAMPLE_BITS": "16", "UNIT": "METER",
        "MAP_PROJECTION_TYPE": "SIMPLE CYLINDRICAL",
        "POSITIVE_LONGITUDE_DIRECTION": "EAST",
        "COORDINATE_SYSTEM_NAME": "PLANETOCENTRIC",
    }
    for key, expected in checks.items():
        if _label_value(text, key) != expected:
            raise ValueError(f"{label.name}: {key} is "
                             f"{_label_value(text, key)!r}, expected "
                             f"{expected!r}")
    ppd = _number(text, "MAP_RESOLUTION")
    west, east = _number(text, "WESTERNMOST_LONGITUDE"), _number(
        text, "EASTERNMOST_LONGITUDE")
    north, south = _number(text, "MAXIMUM_LATITUDE"), _number(
        text, "MINIMUM_LATITUDE")
    if (east - west, north - south) != (360.0, 180.0) \
            or (samples, lines) != (round(360 * ppd), round(180 * ppd)):
        raise ValueError(f"{label.name}: not a global {ppd} px/deg grid")
    raw = np.fromfile(image, dtype=">i2")
    if raw.size != lines * samples:
        raise ValueError(f"{image.name}: {raw.size} samples, expected "
                         f"{lines * samples}")
    heights = raw.reshape(lines, samples).astype(np.float32)   # row 0 = north
    lat = north - (np.arange(lines) + 0.5) / ppd
    lon = west + (np.arange(samples) + 0.5) / ppd
    out = pathlib.Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with netCDF4.Dataset(out, "w") as ds:
        ds.title = f"MOLA MEGDR median topography, {ppd:g} px/deg"
        ds.source = (f"{image.name} ({_label_value(text, 'DATA_SET_ID')}, "
                     "NASA PDS Geosciences Node), sha256 "
                     + hashlib.sha256(image.read_bytes()).hexdigest())
        ds.reference = ("topography = planetary radius minus areoid radius "
                        "(GMM3 areoid), planetocentric, east longitude")
        ds.createDimension("lat", lines)
        ds.createDimension("lon", samples)
        v = ds.createVariable("lat", "f8", ("lat",))
        v[:], v.units = lat, "degrees_north"
        v = ds.createVariable("lon", "f8", ("lon",))
        v[:], v.units = lon, "degrees_east"
        v = ds.createVariable("topography", "f4", ("lat", "lon"), zlib=True)
        v[:], v.units = heights, "m"
    stated = re.search(r"observations\s+for\s+the\s+entire\s+data\s+set\s+"
                       r"are\s+(-?\d+)\s+and\s+(-?\d+)", text)
    note = ("" if not stated else f" (label: {stated.group(1)} .. "
            f"{stated.group(2)} m)")
    print(f"wrote {out}: {lines} x {samples}, {heights.min():.0f} .. "
          f"{heights.max():.0f} m{note}")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _gc_deg(lat1, lon1, lat2, lon2):
    a1, a2 = np.radians(lat1), np.radians(lat2)
    c = (np.sin(a1) * np.sin(a2)
         + np.cos(a1) * np.cos(a2) * np.cos(np.radians(lon1 - lon2)))
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def _extremum_near(lat, lon, values, lat0, lon0, sign, radius=6.0):
    la, lo = np.meshgrid(lat, lon, indexing="ij")
    near = _gc_deg(la, lo, lat0, lon0) <= radius
    masked = np.where(near, sign * values, -np.inf)
    j, i = np.unravel_index(np.argmax(masked), values.shape)
    return float(lat[j]), float(lon[i]), float(values[j, i])


def _specs(lmax: int):
    from tropoi.run.custom import DiscretizationSpec, PlanetSpec
    nlat = 3 * (lmax + 1) // 2 + 1
    nlat += nlat % 2
    return (PlanetSpec.oblate(MARS_RADIUS_M, MARS_DAY_S, MARS_MASS_KG),
            DiscretizationSpec.gauss_latlon(lmax, nlat, 2 * nlat))


def _free_gpu() -> None:
    """Release cached device memory between resolutions (2 GB GPUs)."""
    import gc

    import cupy as cp
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()


def validate(path: str, fine: str | None, lmaxes) -> None:
    from tropoi.run.custom import prepare_planet, read_elevation
    from tropoi.spatial.transforms.host_synthesis import synthesize_latlon

    raster = read_elevation(path)
    print(raster.describe())
    print(f"  area mean {raster.area_mean():.1f} m; provenance: "
          f"{raster.provenance.get('variable')} in "
          f"{raster.provenance.get('units')}; first column centred at "
          f"{float(raster.lon_deg[0]):.3f} deg E after normalization")
    for label, pick in (("highest", np.argmax), ("lowest", np.argmin)):
        j, i = np.unravel_index(pick(raster.elevation_m), raster.shape)
        print(f"  {label} sample {raster.elevation_m[j, i]:.0f} m at "
              f"({raster.lat_deg[j]:.2f}, {raster.lon_deg[i]:.2f})")
    print("Landmarks in the imported raster (extremum within 6 deg):")
    for name, lat0, lon0, sign in LANDMARKS:
        lat, lon, h = _extremum_near(raster.lat_deg, raster.lon_deg,
                                     raster.elevation_m, lat0, lon0 - 360
                                     if lon0 > 180 else lon0, sign)
        offset = _gc_deg(lat, lon, lat0, lon0)
        print(f"  {name:<16} found at ({lat:6.2f}, {lon:7.2f}) "
              f"{h:8.0f} m, {offset:4.2f} deg from the reference position")
    north = raster.lat_deg > 0
    w = np.cos(np.radians(raster.lat_deg))
    mean = lambda rows: float(np.sum(w[rows, None] * raster.elevation_m[rows])
                              / (np.sum(w[rows]) * raster.shape[1]))
    print(f"  hemispheric means: north {mean(north):.0f} m, south "
          f"{mean(~north):.0f} m")
    fine_raster = read_elevation(fine) if fine else None
    lat = np.linspace(-90, 90, 361)
    lon = np.linspace(-180, 180, 721)
    print("Resolved model terrain (Gauss lat-lon, Mars-sized planet):")
    for lmax in lmaxes:
        planet, disc = _specs(lmax)
        world = prepare_planet(planet, disc, raster)
        lm = world.topography_lm
        p = world.topography.parameters
        field = synthesize_latlon(lm, lat, lon)
        olympus = _extremum_near(lat, lon, field, 18.65, -133.8, +1)[2]
        hellas = _extremum_near(lat, lon, field, -42.4, 70.5, -1)[2]
        offset = lm[1, 0].real * math.sqrt(3 / (4 * math.pi))
        line = (f"  l_max {lmax:3d}: residual {p['projection_residual']:.3f}, "
                f"range {field.min():7.0f} .. {field.max():6.0f} m, Olympus "
                f"{olympus:6.0f} m, Hellas {hellas:6.0f} m, degree-1 "
                f"north-south term {offset:6.0f} m")
        if fine_raster is not None:
            _free_gpu()
            other = prepare_planet(planet, disc, fine_raster).topography_lm
            rel = np.linalg.norm(other - lm) / np.linalg.norm(lm)
            line += f"; 16 vs 4 px/deg rel. diff {rel:.1e}"
        print(line)
        _free_gpu()


def show(path: str, lmax: int, smoothing, out: str | None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    from tropoi.run.custom import prepare_planet, read_elevation

    planet, disc = _specs(lmax)
    world = prepare_planet(planet, disc, read_elevation(path),
                           smoothing=smoothing,
                           label=f"Mars (MOLA MEGDR), l_max {lmax}")
    stem = pathlib.Path(out or pathlib.Path(path).parent) / \
        f"mola_l{lmax}"
    stem.parent.mkdir(parents=True, exist_ok=True)
    world.plot().savefig(stem.with_name(stem.name + "_terrain.png"), dpi=110)
    page = world.view(stem.with_name(stem.name + "_globe.html"),
                      open_browser=False)
    print("figure:", stem.with_name(stem.name + "_terrain.png"))
    print("globe: ", page)


def prepare_pe(path: str, out: str, lmax: int, smoothing) -> None:
    import matplotlib
    matplotlib.use("Agg")
    from tropoi.run.custom import (DiscretizationSpec, PlanetSpec,
                                   prepare_planet, read_elevation)

    nlat = 3 * (lmax + 1) // 2 + 1
    nlat += nlat % 2
    # The PE runner builds Earth-unit planets; the terrain coefficients do
    # not depend on the planet's size, which the run sets.
    planet = PlanetSpec.earth_like(day_hours=MARS_DAY_S / 3600.0,
                                   radius_earth_units=MARS_RADIUS_M / 6.371e6)
    world = prepare_planet(planet, DiscretizationSpec.gauss_latlon(
        lmax, nlat, 2 * nlat), read_elevation(path), smoothing=smoothing,
        solver="pe", label=f"Mars (MOLA MEGDR) for the PE core, l_max {lmax}")
    saved = world.save(out, overwrite=True)
    world.plot().savefig(pathlib.Path(str(saved) + "_terrain.png"), dpi=110)
    world.view(pathlib.Path(str(saved) + "_globe.html"), open_browser=False)
    print(world.describe())
    print("saved", saved, "(band limit", world.l_cut, ")")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="step", required=True)
    c = sub.add_parser("convert", help="PDS3 MEGDR label -> NetCDF (CPU)")
    c.add_argument("label")
    c.add_argument("out")
    v = sub.add_parser("validate", help="measure the import of MOLA (CUDA)")
    v.add_argument("file")
    v.add_argument("--fine", help="a finer grid of the same data")
    v.add_argument("--lmax", type=int, nargs="+", default=[21, 31, 42])
    s = sub.add_parser("show", help="figure + globe (CUDA)")
    s.add_argument("file")
    s.add_argument("--lmax", type=int, default=42)
    s.add_argument("--smoothing", type=float)
    s.add_argument("--out")
    e = sub.add_parser("prepare-pe", help="resolve + save for PE (CUDA)")
    e.add_argument("file")
    e.add_argument("out")
    e.add_argument("--lmax", type=int, default=42)
    e.add_argument("--smoothing", type=float)
    args = parser.parse_args(argv)
    if args.step == "convert":
        convert(args.label, args.out)
    elif args.step == "prepare-pe":
        prepare_pe(args.file, args.out, args.lmax, args.smoothing)
    elif args.step == "validate":
        validate(args.file, args.fine, args.lmax)
    else:
        show(args.file, args.lmax, args.smoothing, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
