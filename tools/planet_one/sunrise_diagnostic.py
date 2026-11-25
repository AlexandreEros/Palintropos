"""R2 sunrise attribution experiment (diagnostic; docs/validation/PLANET-ONE-GATES.md, R2).

    python tools/planet_one/sunrise_diagnostic.py GI_DIR   (heat1d venv, PYTHONPATH=src:.)

Starts from the level-3 periodic state of the 0N 0E column at the last
level-3 sample before the first sunrise, and integrates 2 Earth days with:

* O1: ours at the level-3 grid and dt;
* O2: ours at the level-3 grid with dt / 16 (temporal discretization);
* O3: ours on a 4x finer grid with dt / 16 (spatial, including the skin);
* H1: heat1d at level-3 nodes with Fourier number F = 0.5 (its default CFL
  substeps);
* H2: as H1 with F = 0.05 (time-lagged properties and operator splitting);
* H3: heat1d on 4x finer nodes with F = 0.5 (its 3-point surface gradient
  and spatial error).

Every variant has the same absorbed-flux function and laws. T_s is
compared at the level-3 output times. Writes GI_DIR/sunrise.json.
"""
from __future__ import annotations

import json
import math
import pathlib
import sys

import numpy as np

import tools.planet_one.heat1d_compare as hc
from tools.planet_one.convergence import _reconstruction
from tropoi.physics.insolation import insolation_from_spec
from tropoi.physics.substrate import SubstrateState
from tropoi.run.airless.config import mercury_vpw99
from tropoi.run.airless.experiment import build_substrate, load_tier2

WINDOW = 2 * 86400.0


def main(argv=None) -> int:
    gi = pathlib.Path((argv or sys.argv[1:])[0])
    hc.h1p.heatCapacity = lambda planet, T, model="polynomial": hc.LAW.specific_heat(np.asarray(T))
    (grid3, steps3, mn3) = hc.LEVEL3
    base = dict(model="TWO", geothermal_flux=0.02,
                columns={"kind": "points", "lat_deg": [0.0], "lon_deg": [0.0]})
    cfg3 = mercury_vpw99(top_thickness=grid3[0], max_stretch=grid3[1], cycle_steps=steps3,
                         spinup_cycle_steps=None, **base)
    run3 = gi / "ours-L3"
    meta3 = json.loads((run3 / "config.json").read_text())["config"]
    ev = load_tier2(run3 / "retained", "every_step")[:, :, 0]          # column 0N 0E
    ins = insolation_from_spec(cfg3.illumination)
    n = (np.array([1.0]), np.array([0.0]), np.array([0.0]))
    P = cfg3.cycle_s
    dt3 = P / steps3
    absorbed = lambda t: (1 - cfg3.albedo) * float(ins.incident(*n, t)[0][0])
    # the last level-3 sample before the first sunrise after cycle fraction 0.7
    k0 = next(k for k in range(int(0.7 * steps3), steps3) if absorbed((k + 2) * dt3) > 0.0)
    t0 = (k0 + 1) * dt3                                                   # time of sample k0
    z, T = _reconstruction(meta3, np.load(run3 / "grid.npz"), ev[k0:k0 + 1, 0], ev[k0:k0 + 1, 1:])
    zs, Ts0 = z, T[0]
    times = t0 + dt3 * np.arange(1, int(WINDOW / dt3) + 1)

    def ours(top, stretch, sub_steps):
        cfg = mercury_vpw99(top_thickness=top, max_stretch=stretch, cycle_steps=steps3,
                            spinup_cycle_steps=None, **base)
        sub = build_substrate(cfg)
        st = SubstrateState(np.interp(sub.centres, zs, Ts0)[:, None] - 300.0,
                            np.array([Ts0[0] - 300.0]))
        out, h = [], dt3 / sub_steps
        t = t0
        for _ in times:
            for _ in range(sub_steps):
                t += h
                st, _r = sub.step(st, h, shortwave_down=np.array([absorbed(t) / (1 - cfg.albedo)]))
            out.append(float(sub.surface_temperature(st)[0]))
        return np.array(out)

    def heat1d(mn, F):
        p = hc.heat1d_profile("TWO", 1.0, mn, cfg3.emissivity)
        p.config.F = F
        p.T = np.interp(p.z, zs, Ts0)
        p.update_properties()
        out, t = [], t0
        for tk in times:
            nsub = max(1, math.ceil((tk - t) / hc.computeCFL(p, p.config)))
            h = (tk - t) / nsub
            for j in range(nsub):
                p.update_T(h, Qs=absorbed(t + (j + 1) * h), Qb=0.02)
                p.update_properties()
            t = tk
            out.append(float(p.T[0]))
        return np.array(out)

    v = {"O1": ours(grid3[0], grid3[1], 1),
         "O2": ours(grid3[0], grid3[1], 16),
         "O3": ours(grid3[0] / 4, grid3[1] ** 0.25, 16),
         "H1": heat1d(mn3, 0.5),
         "H2": heat1d(mn3, 0.05),
         "H3": heat1d((mn3[0] * 4, mn3[1] * 4), 0.5)}
    l3_ours = ev[k0 + 1:k0 + 1 + times.size, 0]
    pairs = {f"{a}-{b}": float(np.max(np.abs(v[a] - v[b])))
             for a, b in (("O1", "O2"), ("O2", "O3"), ("O1", "O3"), ("H1", "H2"), ("H1", "H3"),
                          ("O1", "H1"), ("O2", "H2"), ("O3", "H3"), ("O3", "H2"))}
    report = {"start_cycle_fraction": (k0 + 1) / steps3, "window_s": WINDOW, "dt3_s": dt3,
              "O1_reproduces_level3_run_K": float(np.max(np.abs(v["O1"] - l3_ours))),
              "max_abs_differences_K": pairs,
              "first_10_samples_K": {k: [round(x, 3) for x in a[:10]] for k, a in v.items()},
              "reading": ("O1~O2~O3: ours converged at sunrise; H2/H3 moving toward O: heat1d's "
                          "offset from lag/splitting (H2) or surface gradient/space (H3); "
                          "the remaining O3-to-(best H) gap: formulation difference")}
    (gi / "sunrise.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
