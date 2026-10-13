"""A zonal jet meeting a mountain on a Mars-sized planet (SWE).

    python examples/custom_swe_planet.py prepare  states/mars_jet   # CUDA (projection)
    python examples/custom_swe_planet.py inspect  states/mars_jet   # CPU, no CuPy
    python examples/custom_swe_planet.py run      states/mars_jet --out runs  # CUDA
    tropoi run swe --initial-state states/mars_jet --days 2       # same run, from the shell

The planet, the layer and the terrain are prescribed separately: an oblate
(ordinary-construction) planet with Mars' size and rotation, a 4 km layer
under Mars' surface gravity given EXPLICITLY (never derived from mass and
radius), and the existing Gaussian-mountain terrain preset.

The fields are given as Python callables of (lat, lon) in radians on host
NumPy arrays, in SI units, and projected on the model grid:

* streamfunction psi = -a u0 sin(lat)        -> zeta = 2 u0 sin(lat) / a
* free-surface anomaly eta' = (C/g)(1/3 - sin^2 lat),  C = a Omega u0 + u0^2/2

(the Williamson-2 balance on this planet). Preparation turns eta' into the
prognostic phi = g eta' - g h_s' (perturbation thickness geopotential; the
fluid thickness is h = H + phi/g), reports projection residuals, and checks
that h > 0 on the state and product samplings. The state is valid; over the
mountain it is not balanced, which is the point of the experiment.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from tropoi.run.custom import (DiscretizationSpec, FluidLayerSpec, PlanetSpec,
                               TopographySpec, load_initial_state)

U0 = 15.0                                   # m/s


def prepare(path: str) -> None:
    from tropoi.run.custom import prepare_swe_state

    planet = PlanetSpec.oblate(equatorial_radius_m=3.3962e6,
                               sidereal_day_s=88642.66, mass_kg=6.417e23)
    fluid = FluidLayerSpec(gravity_m_s2=3.71, mean_depth_m=4000.0)
    terrain = TopographySpec.mountain(height_m=1500.0, lat_deg=30.0,
                                      lon_deg=90.0, width_deg=20.0)
    discretization = DiscretizationSpec.gauss_latlon(lmax=31, nlat=48,
                                                     nlon=96)
    a = planet.dynamical_radius_m
    C = a * planet.angular_velocity_rad_s * U0 + 0.5 * U0 * U0
    g = fluid.gravity_m_s2

    def streamfunction(lat, lon):           # m^2/s
        return -a * U0 * np.sin(lat)

    def free_surface_anomaly(lat, lon):     # m, zero global mean
        return (C / g) * (1.0 / 3.0 - np.sin(lat) ** 2)

    prepared = prepare_swe_state(
        planet, discretization, fluid, terrain,
        streamfunction=streamfunction,
        free_surface_anomaly=free_surface_anomaly,
        label="mars-sized jet over a mountain")
    print("\n".join(prepared.report.lines()))
    print("saved", prepared.save(path, overwrite=True))


def inspect(path: str) -> None:
    state = load_initial_state(path)
    s = state.summary()
    print(f"{s['label']}: {s['planet']}")
    print(f"  fluid: g = {s['fluid']['gravity_m_s2']} m/s^2, "
          f"H = {s['fluid']['mean_depth_m']} m; terrain {s['topography']}")
    for name in state.state:
        view = state.state[name].validate()
        print(f"  {name:<6}{view.units:<10} max |a_lm| "
              f"{view.summary()['max_abs']:.3e}")
    print("  digest", s["digest"])
    print("  cupy imported:", sys.modules.get("cupy") is not None)


def run(path: str, out: str, days: float) -> None:
    from tropoi.representation.archive import open_simulation
    from tropoi.run.custom import run_initial_state

    run_dir = run_initial_state(path, duration_days=days, n_snapshots=5,
                                out=out, experiment="examples")
    sim = open_simulation(run_dir)
    print("run", run_dir)
    print("stored times (h):", [t / 3600.0 for t in sim.times])
    print("draw it with: tropoi plot", run_dir)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=("prepare", "inspect", "run"))
    parser.add_argument("path", help="initial-state directory")
    parser.add_argument("--out", default="runs")
    parser.add_argument("--days", type=float, default=2.0)
    args = parser.parse_args(argv)
    if args.step == "prepare":
        prepare(args.path)
    elif args.step == "inspect":
        inspect(args.path)
    else:
        run(args.path, args.out, args.days)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
