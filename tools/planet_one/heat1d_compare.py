"""G-I: matched comparison with heat1d (L2 evidence).

heat1d (Hayne et al., MIT licence) is pinned at
phayne/heat1d@5b73a6c1ba5047f59f6210185b37539e46e68136. Its discretization
is independent of ours:

* node-based finite differences;
* the surface temperature on a node, from a 3-point gradient;
* an operator-split Newton surface solve followed by an implicit interior
  solve;
* properties lagged at the previous temperatures.

Run it from the heat1d venv with ``PYTHONPATH=src``::

    python tools/planet_one/heat1d_compare.py OUTDIR [--model TWO] [--geothermal-flux 0.02]

Matching:

* Same absorbed-flux series. The absorbed flux (1 - A) F(t) at every step
  comes from our geometry and is passed to heat1d as ``Qs``.
* Same laws. Ledlow c_p is injected into heat1d. VPW99 strata are custom
  layers, with a node inserted at the 2 cm boundary.
* Same emissivity, G and depth.
* heat1d's surface tolerance ``DTSURF`` is tightened to 1e-9 K, and its
  1 K property cache is disabled (threshold 0).
* heat1d takes CFL-limited substeps (its own ``computeCFL``) within each of
  our steps, which is its documented practice for implicit runs. Its
  surface/interior splitting is not accurate for dt >> dz^2/kappa: a first
  attempt with a shared fixed dt diverged from us at sunrise under grid
  refinement, while our solution was grid-converged there.

Formulation differences that remain, listed in the report:

* cell-mean versus node temperatures;
* properties lagged in time versus solved simultaneously by Newton;
* the boundary treatment;
* interface handling.

Levels i = 0, 1, 2 refine both codes jointly: our cell grid (top
thickness, stretch) and dt, and heat1d's ``m`` and the same dt. Each code
reaches its own periodic state. Starting from our periodic profile is
allowed, since the start does not change the periodic solution; heat1d's
periodicity is checked.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import types

import numpy as np

import heat1d.profile as h1p
from heat1d.config import Configurator
from heat1d.layers import DepthLayer
from heat1d.solvers import computeCFL

from tropoi.physics.insolation import insolation_from_spec
from tropoi.physics.materials import ledlow_1992
from tropoi.run.airless.config import VPW99, mercury_vpw99
from tropoi.run.airless.experiment import AirlessRun, load_tier2

# (our top thickness, stretch), steps per cycle, heat1d (m, n): m nodes per skin
# depth, node spacing growth 1 + 1/n
LEVELS = [((2e-3, 1.1), 2160, (40, 4)), ((1e-3, 1.0488), 4320, (80, 8)),
          ((5e-4, 1.0241), 8640, (160, 16))]
COLUMNS = {"0N 0E": (0.0, 0.0), "0N 90E": (0.0, 90.0)}
LAW = ledlow_1992()


def heat1d_profile(model, depth, mn, emissivity):
    m, n = mn
    t, b = VPW99["top"], VPW99["bottom"]
    planet = types.SimpleNamespace(
        name="Mercury (VPW99 layers)", emissivity=emissivity, albedo=VPW99["albedo"],
        ks=t["kc"], kd=b["kc"], rhos=t["rho"], rhod=b["rho"], H=0.0, cp0=600.0,
        cpCoeff=[0.0], day=175.94 * 86400.0, S=9083.0, Qb=0.0, cp_fixed=None)
    cfg = Configurator(solver="implicit", m=m, n=n, b=40, DTSURF=1e-9,
                       chi=b["chi"])
    if model == "TWO":
        layers = [DepthLayer(0.0, t["bottom"], t["rho"], t["kc"], t["chi"], "top"),
                  DepthLayer(t["bottom"], depth * 2.0, b["rho"], b["kc"], b["chi"], "bottom")]
    else:
        props = t if model == "TOP" else b
        layers = [DepthLayer(0.0, depth * 2.0, props["rho"], props["kc"], props["chi"], model)]
    p = h1p.Profile(planet=planet, lat=0.0, config=cfg, custom_layers=layers)
    # truncate heat1d's node grid at our column depth (a node exactly at depth)
    keep = np.flatnonzero(p.z < depth * (1 - 1e-12))
    z = np.append(p.z[keep], depth)
    for name in ("kc", "rho", "_chi_array"):
        arr = getattr(p, name)
        setattr(p, name, np.append(arr[keep], arr[keep[-1]]))
    p.z, p.nlayers = z, z.size
    p.dz = np.diff(z)
    p.d3z = p.dz[1:] * p.dz[:-1] * (p.dz[1:] + p.dz[:-1])
    p.g1 = 2 * p.dz[1:] / p.d3z
    p.g2 = 2 * p.dz[:-1] / p.d3z
    p._R350_array = p._chi_array / 350.0 ** 3
    p.T = np.full(p.nlayers, p.T[0])
    p._T_at_last_update = None
    p._prop_cache_threshold = 0.0
    return p


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("--model", default="TWO")
    ap.add_argument("--geothermal-flux", type=float, default=0.02)
    ap.add_argument("--max-cycles", type=int, default=40)
    args = ap.parse_args(argv)
    out = pathlib.Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    # matched heat capacity: Ledlow eqs. (6)-(7) for heat1d as well
    h1p.heatCapacity = lambda planet, T, model="polynomial": LAW.specific_heat(np.asarray(T))
    lats = [v[0] for v in COLUMNS.values()]
    lons = [v[1] for v in COLUMNS.values()]
    report = {"heat1d": "phayne/heat1d@5b73a6c1ba5047f59f6210185b37539e46e68136",
              "model": args.model, "geothermal_flux": args.geothermal_flux,
              "differences": ["cell means (ours) vs node values (heat1d)",
                              "Newton-coupled properties and skin (ours) vs operator-split "
                              "surface Newton with properties lagged one step (heat1d)",
                              "Kirchhoff half-cell skin flux (ours) vs 3-point surface "
                              "gradient with k at the surface node (heat1d)",
                              "series half-cell interface conductance (ours) vs node-based "
                              "property placement with a node at 2 cm (heat1d)"],
              "levels": []}
    for i, (grid, steps, m) in enumerate(LEVELS):
        cfg = mercury_vpw99(model=args.model, geothermal_flux=args.geothermal_flux,
                            columns={"kind": "points", "lat_deg": lats, "lon_deg": lons},
                            top_thickness=grid[0], max_stretch=grid[1], cycle_steps=steps,
                            spinup_cycle_steps=360)
        rd = out / f"ours-L{i}"
        if not (rd / "retained" / "status.json").exists():
            AirlessRun(cfg, rd, workers=1, log=lambda s: None).run()
        status = json.loads((rd / "retained" / "status.json").read_text())
        ev = load_tier2(rd / "retained", "every_step")            # (steps, 1+L, ncol)
        centres = np.load(rd / "grid.npz")["centres"]
        ins = insolation_from_spec(cfg.illumination)
        dt = cfg.cycle_s / steps
        level = {"level": i, "ours": {"grid": grid, "steps": steps, "status": status},
                 "heat1d": {"m": m, "steps": steps}, "columns": {}}
        for c, (name, (lat, lon)) in enumerate(COLUMNS.items()):
            la, lo = math.radians(lat), math.radians(lon)
            n = (np.array([math.cos(la) * math.cos(lo)]), np.array([math.cos(la) * math.sin(lo)]),
                 np.array([math.sin(la)]))
            p = heat1d_profile(args.model, cfg.strata[-1]["bottom"], m, cfg.emissivity)
            # start from our periodic end-of-cycle profile
            p.T = np.interp(p.z, np.r_[0.0, centres], np.r_[ev[-1, 0, c], ev[-1, 1:, c]])
            p.update_properties()
            prev = None
            absorbed = lambda t: (1 - cfg.albedo) * float(ins.incident(*n, t)[0][0])
            for cyc in range(args.max_cycles):
                ts = np.empty(steps)
                for k in range(steps):
                    # heat1d's own practice for implicit runs: CFL-limited
                    # substeps (its surface/interior operator splitting is
                    # accurate only for dt ~ dz^2 / kappa), with the absorbed
                    # flux of the same geometry at every substep
                    nsub = max(1, math.ceil(dt / computeCFL(p, p.config)))
                    h = dt / nsub
                    for j in range(nsub):
                        p.update_T(h, Qs=absorbed(k * dt + (j + 1) * h), Qb=args.geothermal_flux)
                        p.update_properties()
                    ts[k] = p.T[0]
                if prev is not None and np.max(np.abs(ts - prev)) < 0.01:
                    break
                prev = ts
            ours = ev[:, 0, c]
            d = ts - ours
            level["columns"][name] = {
                "heat1d_cycles": cyc + 1, "heat1d_last_cycle_change_K": float(np.max(np.abs(ts - prev))) if prev is not None else None,
                "max_abs_dTs_K": float(np.max(np.abs(d))), "rms_dTs_K": float(np.sqrt(np.mean(d * d))),
                "Tmax_ours": float(ours.max()), "Tmax_heat1d": float(ts.max()),
                "Tmin_ours": float(ours.min()), "Tmin_heat1d": float(ts.min()),
                "nodes": int(p.z.size), "cells_ours": int(centres.size)}
            np.save(out / f"heat1d-L{i}-{name.replace(' ', '_')}.npy", ts)
        report["levels"].append(level)
        (out / "gi.json").write_text(json.dumps(report, indent=1))
        print(json.dumps(level["columns"], indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())


def verdict(outdir) -> dict:
    """The frozen G-I criterion on a finished comparison: the difference
    shrinks under joint refinement, and at the finest level
    max|dTs| <= e_ours + e_heat1d + 0.2 K, where each e is that code's own
    Richardson estimate from its three levels sampled at the coarsest
    level's times: e = d2 / (2^p - 1), with p = log2(d1 / d2) and
    d_i = max_t |Q_i - Q_(i-1)|. If p <= 0.5 the conservative 2 d2 is used."""
    out = pathlib.Path(outdir)
    rep = json.loads((out / "gi.json").read_text())
    res = {}
    for name in COLUMNS:
        tag = name.replace(" ", "_")
        ours, h1d = [], []
        for i in range(len(LEVELS)):
            ev = load_tier2(out / f"ours-L{i}" / "retained", "every_step")
            c = list(COLUMNS).index(name)
            stride = 2 ** i
            ours.append(ev[stride - 1::stride, 0, c])
            h1d.append(np.load(out / f"heat1d-L{i}-{tag}.npy")[stride - 1::stride])

        def est(seq):
            d1 = float(np.max(np.abs(seq[1] - seq[0])))
            d2 = float(np.max(np.abs(seq[2] - seq[1])))
            p = math.log2(d1 / d2) if d1 > 0 and d2 > 0 else 0.0
            return (d2 / (2 ** p - 1) if p > 0.5 else 2 * d2), p, d1, d2
        eo, po, *_ = est(ours)
        eh, ph, *_ = est(h1d)
        diffs = [float(np.max(np.abs(o - h))) for o, h in zip(ours, h1d)]
        rms = [float(np.sqrt(np.mean((o - h) ** 2))) for o, h in zip(ours, h1d)]
        bound = eo + eh + 0.2
        res[name] = {"max_abs_dTs_by_level_K": diffs, "rms_dTs_by_level_K": rms,
                     "e_ours_K": eo, "order_ours": po, "e_heat1d_K": eh, "order_heat1d": ph,
                     "bound_K": bound, "final_max_abs_dTs_K": diffs[-1],
                     "shrinks_rms": rms[0] > rms[1] > rms[2],
                     "shrinks_max_last_step": diffs[-1] < diffs[-2],
                     "pass": diffs[-1] <= bound and rms[0] > rms[1] > rms[2]
                     and diffs[-1] < diffs[-2]}
    rep["verdict"] = res
    rep["pass"] = all(r["pass"] for r in res.values())
    (out / "gi.json").write_text(json.dumps(rep, indent=1))
    return rep
