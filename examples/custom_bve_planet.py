"""A Rossby-Haurwitz wave on a small, fast-rotating planet (BVE).

Three steps, each its own process, so the reload genuinely happens in a
fresh interpreter:

    python examples/custom_bve_planet.py prepare  states/rh4_small   # CPU
    python examples/custom_bve_planet.py inspect  states/rh4_small   # CPU, no CuPy
    python examples/custom_bve_planet.py run      states/rh4_small --out runs  # CUDA
    tropoi run bve --initial-state states/rh4_small --days 1      # same run, from the shell

The state is built from exact spherical-harmonic coefficients, so preparing
it needs no GPU. Relative vorticity of the wave (Williamson et al. 1992,
case 6 form):

    zeta = 2 nu sin(lat) - 30 K sin(lat) cos^4(lat) sin(4 lon)

In the repository convention (complex orthonormal Y_l^m, m >= 0 stored, a
real field f = sum a_l0 Y_l0 + 2 Re sum_{m>0} a_lm Y_lm):

    sin(lat)                   = sqrt(4 pi / 3) Y_1^0
    sin(lat) cos^4(lat) e^{4i lon} = Y_5^4 / (945 N_5^4),
    N_5^4 = sqrt(11/(4 pi) * 1!/9!)

so a_10 = 2 nu sqrt(4 pi/3) and, since sin(4 lon) = Re(-i e^{4i lon}),
a_54 = -i (-30 K) / (2 * 945 N_5^4).
"""
from __future__ import annotations

import argparse
import math
import sys

import numpy as np

from tropoi.run.custom import (DiscretizationSpec, PlanetSpec,
                               SpectralCoefficients, load_initial_state,
                               prepare_bve_state)

NU = K = 1.5e-5            # s^-1: wave amplitude, scaled for the faster spin


def rh4_vorticity_coefficients(l_max: int) -> np.ndarray:
    zeta = np.zeros((l_max + 1, l_max + 1), dtype=np.complex128)
    zeta[1, 0] = 2.0 * NU * math.sqrt(4.0 * math.pi / 3.0)
    n54 = math.sqrt(11.0 / (4.0 * math.pi) / math.factorial(9))
    zeta[5, 4] = -1j * (-30.0 * K) / (2.0 * 945.0 * n54)
    return zeta


def prepare(path: str) -> None:
    planet = PlanetSpec.ideal_sphere(radius_m=3.0e6,
                                     sidereal_day_s=10.0 * 3600.0)
    discretization = DiscretizationSpec.gauss_latlon(lmax=21, nlat=32,
                                                     nlon=64)
    prepared = prepare_bve_state(
        planet, discretization,
        zeta=SpectralCoefficients(rh4_vorticity_coefficients(21)),
        label="rh4 on a 3000 km, 10 h planet")
    print("\n".join(prepared.report.lines()))
    print("saved", prepared.save(path, overwrite=True))
    print("digest", prepared.state.digest)


def inspect(path: str) -> None:
    state = load_initial_state(path)            # verified; CuPy not needed
    summary = state.summary()
    print(f"{summary['label']}: {summary['planet']}")
    print(f"  {summary['discretization']}, digest {summary['digest']}")
    zeta = state.state["zeta"].validate()
    print(f"  zeta: {zeta.units}, {zeta.summary()['nonzero']} nonzero "
          f"coefficients, max |a_lm| {zeta.summary()['max_abs']:.3e}")
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
    parser.add_argument("--days", type=float, default=1.0)
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
