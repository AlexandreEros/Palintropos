r"""3-D atmosphere view of a saved primitive-equation run (CPU only).

    python examples/atmosphere_view.py RUN [--out FILE] [--times 0 -1]
        [--levels 0 10 19] [--resolution 2] [--max-mb 32] [--no-streamlines]
        [--field temperature|pressure|zeta|delta] [--level K] [--opacity 0.6]
        [--exaggeration 20] [--centre LAT LON] [--gravity G] [--open]

RUN is a ``tropoi run pe`` capsule or a Held–Suarez experiment directory
(its ``states/state-d*.npy``), for example ``runs/hs-T42L20-prod-002``.
The script is the documented API, nothing more:

    from tropoi.representation.archive import open_simulation

    sim = open_simulation(RUN)
    path = sim.view(out, times=..., levels=..., open_browser=False)

It writes ONE self-contained HTML file (default: the system temporary
directory, never inside RUN) that opens with no network, server, Python or
CUDA: the run's terrain plus one translucent native sigma surface placed at
its hydrostatic geopotential height, with sliders over the exported saved
times, native levels, exaggeration and opacity, a field selector and
optional instantaneous horizontal streamlines. ``times`` are saved snapshot
indices (negative counts back); without ``--times`` as many evenly spaced
states as fit ``--max-mb`` are exported and a warning says which.
See docs/ATMOSPHERE_VIEW.md.
"""
from __future__ import annotations

import argparse
import pathlib


def main(argv=None) -> pathlib.Path:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[1],
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("run", help="PE run capsule or Held–Suarez experiment")
    parser.add_argument("--out", help="output HTML (default: temp directory)")
    parser.add_argument("--times", type=int, nargs="+",
                        help="saved snapshot indices (default: fit --max-mb)")
    parser.add_argument("--all-times", action="store_true",
                        help="every saved state (refused above --max-mb)")
    parser.add_argument("--levels", type=int, nargs="+",
                        help="native level indices, 0 = top (default: all)")
    parser.add_argument("--resolution", type=float,
                        help="display mesh spacing in degrees")
    parser.add_argument("--max-mb", type=float, default=32.0,
                        help="payload bound in MB (default 32)")
    parser.add_argument("--no-streamlines", action="store_true")
    parser.add_argument("--field", default="temperature",
                        choices=("temperature", "pressure", "zeta", "delta"))
    parser.add_argument("--level", type=int,
                        help="initial native level index (default: middle)")
    parser.add_argument("--opacity", type=float, default=0.6)
    parser.add_argument("--centre", type=float, nargs=2, metavar=("LAT", "LON"),
                        help="initial view centre in degrees")
    parser.add_argument("--exaggeration", type=float)
    parser.add_argument("--gravity", type=float,
                        help="only for runs that record none (flat presets)")
    parser.add_argument("--open", action="store_true",
                        help="open the page in the default browser")
    args = parser.parse_args(argv)

    from tropoi.representation.archive import open_simulation

    sim = open_simulation(args.run)
    print(f"{sim!r}: saved times {sim.times[0]:g} .. {sim.times[-1]:g} s"
          if len(sim) else f"{sim!r}")
    path = sim.view(
        args.out, open_browser=args.open,
        times="all" if args.all_times else args.times, levels=args.levels,
        resolution_deg=args.resolution, max_mb=args.max_mb,
        streamlines=not args.no_streamlines, field=args.field,
        level=args.level, opacity=args.opacity,
        exaggeration=args.exaggeration,
        centre=None if args.centre is None else tuple(args.centre),
        gravity=args.gravity)
    print(f"wrote {path} ({path.stat().st_size / 1e6:.1f} MB)")
    return path


if __name__ == "__main__":
    main()
