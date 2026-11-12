r"""Sky of a hypothetical atmosphere, from explicit parameters (CPU only).

    python examples/sky_hypothetical.py [--out FILE] [--no-open]
        [--composition N2=1] [--surface-pressure-pa 1e4] [--temperature-k 250]
        [--lapse-rate-k-per-km 0 --floor-k T] [--gravity 9.80665]
        [--gas-constant from_composition|R] [--radius-m 6.371e6]
        [--star-temperature-k 3200] [--star-angular-radius-deg A]
        [--star-azimuth-deg 135] [--star-elevation-deg 30]
        [--altitude-m Z | --pressure-pa P] [--exposure-reference-cd-m2 L]

The defaults are the acceptance case: 0.1 bar of pure N2 under a 3200 K
star. Every other input is stated here, none is inferred:

* isothermal at 250 K; standard gravity 9.80665 m s^-2; the gas constant of
  the stated composition (R*/M of N2, about 296.8 J kg^-1 K^-1);
* planetary radius 6 371 000 m (Earth's mean radius, used as a round
  number), with the surface pressure at that sphere;
* a 3200 K blackbody whose angular radius is chosen so that its bolometric
  irradiance equals that of the IAU nominal Sun at 1 au
  (sin a = sin a_sun * (5772 / 3200)^2, about 0.867 deg): the same total
  power as Earth's sunlight, redder;
* the star 30 deg high at azimuth 135 deg; the observer 2 m above the
  ground; automatic exposure (pass ``--exposure-reference-cd-m2`` for a
  fixed one, so brightness comparisons between runs are meaningful).

It needs no saved simulation, prepared planet, terrain data or CUDA: it
imports only ``tropoi.physics`` and
``tropoi.representation.visual.sky``. The page is ONE offline HTML file in
which atmosphere, composition, star, observer, view and exposure can all be
changed. It is a single-scattering Rayleigh visual prototype, not a
validated sky colour. See docs/SKY_VIEW.md.
"""
from __future__ import annotations

import argparse
import math
import pathlib

from tropoi.representation.visual.sky import (HypotheticalColumn, SkySpec,
                                              write_sky_page)
from tropoi.physics.composition import Composition
from tropoi.physics.star import ApparentStar

SUN_T_K = 5772.0                                      # IAU 2015 B3 nominal
SUN_SIN_ALPHA_1AU = 695700.0 / 149597870.7            # R_sun / au (IAU)


def equal_irradiance_radius_deg(temperature_k: float) -> float:
    """Angular radius of a blackbody of ``temperature_k`` with the IAU Sun's
    bolometric irradiance at 1 au (irradiance ~ T^4 sin^2 a)."""
    s = SUN_SIN_ALPHA_1AU * (SUN_T_K / float(temperature_k)) ** 2
    if not 0.0 < s < math.sin(math.radians(30.0)):
        raise ValueError(f"no such star at {temperature_k} K (sin a = {s})")
    return math.degrees(math.asin(s))


def _fractions(items) -> dict[str, float]:
    out = {}
    for item in items:
        name, _, value = item.partition("=")
        if not value:
            raise SystemExit(f"--composition entries are NAME=FRACTION, got "
                             f"{item!r}")
        out[name.strip()] = float(value)
    return out


def build_spec(args) -> SkySpec:
    fractions = _fractions(args.composition)
    composition = Composition.of(
        label=" + ".join(f"{k} {v:g}" for k, v in fractions.items()),
        source="stated on the command line", **fractions)
    alpha = (args.star_angular_radius_deg if args.star_angular_radius_deg
             is not None else equal_irradiance_radius_deg(
                 args.star_temperature_k))
    star = ApparentStar(args.star_temperature_k, alpha,
                label=f"{args.star_temperature_k:g} K blackbody star",
                source=("stated on the command line" if
                        args.star_angular_radius_deg is not None else
                        "angular radius chosen for the IAU nominal Sun's "
                        "bolometric irradiance at 1 au"))
    gas = (args.gas_constant if args.gas_constant == "from_composition"
           else float(args.gas_constant))
    column = HypotheticalColumn(
        surface_pressure_pa=args.surface_pressure_pa,
        temperature_k=args.temperature_k, gravity=args.gravity,
        gas_constant=gas, radius_m=args.radius_m,
        lapse_rate_k_per_m=args.lapse_rate_k_per_km / 1000.0,
        temperature_floor_k=args.floor_k)
    fixed = args.exposure_reference_cd_m2 is not None
    return SkySpec(
        composition, star=star, altitude_m=args.altitude_m,
        pressure_pa=args.pressure_pa, look_azimuth_deg=args.star_azimuth_deg,
        look_elevation_deg=25.0, fov_deg=100.0,
        star_azimuth_deg=args.star_azimuth_deg,
        star_elevation_deg=args.star_elevation_deg,
        exposure_mode="fixed" if fixed else "auto",
        exposure_reference_cd_m2=args.exposure_reference_cd_m2,
        hypothetical=column)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[1],
        formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--out", help="output HTML (default: temp directory)")
    p.add_argument("--no-open", action="store_true",
                   help="write the page without opening a browser")
    p.add_argument("--composition", nargs="+", default=["N2=1"],
                   help="NAME=MOLE_FRACTION ... (N2, O2, Ar, CO2); must sum "
                        "to 1, never renormalized (default N2=1)")
    p.add_argument("--surface-pressure-pa", type=float, default=1.0e4)
    p.add_argument("--temperature-k", type=float, default=250.0)
    p.add_argument("--lapse-rate-k-per-km", type=float, default=0.0)
    p.add_argument("--floor-k", type=float,
                   help="isothermal temperature above the lapse-rate layer")
    p.add_argument("--gravity", type=float, default=9.80665)
    p.add_argument("--gas-constant", default="from_composition",
                   help="J kg^-1 K^-1, or from_composition (default)")
    p.add_argument("--radius-m", type=float, default=6.371e6)
    p.add_argument("--star-temperature-k", type=float, default=3200.0)
    p.add_argument("--star-angular-radius-deg", type=float,
                   help="default: the Sun's bolometric irradiance at 1 au")
    p.add_argument("--star-azimuth-deg", type=float, default=135.0)
    p.add_argument("--star-elevation-deg", type=float, default=30.0)
    height = p.add_mutually_exclusive_group()
    height.add_argument("--altitude-m", type=float)
    height.add_argument("--pressure-pa", type=float)
    p.add_argument("--exposure-reference-cd-m2", type=float,
                   help="fixed exposure: luminance drawn white at 0 EV")
    return p


def main(argv=None) -> pathlib.Path:
    args = parser().parse_args(argv)
    spec = build_spec(args)
    path = write_sky_page(spec, args.out, open_browser=not args.no_open)
    print(f"wrote {path}")
    return path


if __name__ == "__main__":
    main()
