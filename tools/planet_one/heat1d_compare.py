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

from tropoi.physics.insolation import insolation_from_spec
from tropoi.physics.materials import ledlow_1992
from tropoi.run.airless.config import VPW99, mercury_vpw99
from tropoi.run.airless.experiment import AirlessRun, load_tier2

# (our top thickness, stretch), steps per cycle, heat1d (m, n): m nodes per skin
# depth, node spacing growth 1 + 1/n
LEVELS = [((2e-3, 1.1), 2160, (40, 4)), ((1e-3, 1.0488), 4320, (80, 8)),
          ((5e-4, 1.0241), 8640, (160, 16))]
#: R2 (register): the fourth joint level, a joint halving of level 2
LEVEL3 = ((2.5e-4, 1.012), 17280, (320, 32))
COLUMNS = {"0N 0E": (0.0, 0.0), "0N 90E": (0.0, 90.0)}
LAW = ledlow_1992()
PERIODICITY_K = 0.01


def heat1d_profile(model, depth, mn, emissivity):
    import heat1d.profile as h1p
    from heat1d.config import Configurator
    from heat1d.layers import DepthLayer

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


def _run_heat1d_cycles(profile, dt, steps, max_cycles, absorbed, geothermal_flux, compute_cfl):
    prev = None
    change = None
    for cycle in range(1, max_cycles + 1):
        ts = np.empty(steps)
        for k in range(steps):
            # Resolve heat1d's operator splitting with its own CFL-limited substeps.
            nsub = max(1, math.ceil(dt / compute_cfl(profile, profile.config)))
            h = dt / nsub
            for j in range(nsub):
                profile.update_T(h, Qs=absorbed(k * dt + (j + 1) * h), Qb=geothermal_flux)
                profile.update_properties()
            ts[k] = profile.T[0]
        if prev is not None:
            change = float(np.max(np.abs(ts - prev)))
            if change < PERIODICITY_K:
                break
        prev = ts
    return ts, cycle, change


def _heat1d_is_periodic(column):
    change = column.get("heat1d_last_cycle_change_K")
    if change is None or not 0.0 <= change < PERIODICITY_K or column.get("heat1d_cycles", 0) < 2:
        return False
    # Legacy zeroes cannot distinguish a true zero from a self-comparison
    # after exhausting the cycle limit. New reports attest to the comparison.
    return column.get("heat1d_converged", change > 0.0) is True


def _require_heat1d_periodicity(report, nlevels):
    levels = {row["level"]: row["columns"] for row in report["levels"]}
    for i in range(nlevels):
        for name in COLUMNS:
            if not _heat1d_is_periodic(levels.get(i, {}).get(name, {})):
                raise ValueError(f"level {i}, {name}: heat1d periodicity is unverified")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("--model", default="TWO")
    ap.add_argument("--geothermal-flux", type=float, default=0.02)
    ap.add_argument("--max-cycles", type=int, default=40)
    ap.add_argument("--with-level3", action="store_true",
                    help="R2: add the fourth joint level, warm-started from level 2")
    args = ap.parse_args(argv)
    if args.max_cycles < 1:
        ap.error("--max-cycles must be positive")
    import heat1d.profile as h1p
    from heat1d.solvers import computeCFL

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
    levels = LEVELS + ([LEVEL3] if args.with_level3 else [])
    if (out / "gi.json").exists():          # keep every completed level and verdict
        report = json.loads((out / "gi.json").read_text())
    done = {lv["level"] for lv in report["levels"]
            if all(_heat1d_is_periodic(lv["columns"].get(name, {})) for name in COLUMNS)}
    for i, (grid, steps, m) in enumerate(levels):
        warm = i == 3
        cfg = mercury_vpw99(model=args.model, geothermal_flux=args.geothermal_flux,
                            columns={"kind": "points", "lat_deg": lats, "lon_deg": lons},
                            top_thickness=grid[0], max_stretch=grid[1], cycle_steps=steps,
                            spinup_cycle_steps=None if warm else 360)
        rd = out / f"ours-L{i}"
        if not (rd / "retained" / "status.json").exists():
            init, src = (warm_start(out / "ours-L2", cfg), "verified level-2 periodic "
                         "solution (ours-L2 retained cycle end), interpolated on its FV "
                         "reconstruction") if warm else (None, None)
            AirlessRun(cfg, rd, workers=1, log=lambda s: None, initial_state=init,
                       initial_state_source=src).run()
        if i in done:
            continue
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
            # start from our periodic end-of-cycle profile at this level (a
            # verified periodic solution); heat1d's own periodicity is checked
            p.T = np.interp(p.z, np.r_[0.0, centres], np.r_[ev[-1, 0, c], ev[-1, 1:, c]])
            p.update_properties()
            absorbed = lambda t: (1 - cfg.albedo) * float(ins.incident(*n, t)[0][0])
            ts, cycles, change = _run_heat1d_cycles(
                p, dt, steps, args.max_cycles, absorbed, args.geothermal_flux, computeCFL)
            ours = ev[:, 0, c]
            d = ts - ours
            level["columns"][name] = {
                "heat1d_cycles": cycles, "heat1d_last_cycle_change_K": change,
                "heat1d_converged": change is not None and change < PERIODICITY_K,
                "max_abs_dTs_K": float(np.max(np.abs(d))), "rms_dTs_K": float(np.sqrt(np.mean(d * d))),
                "Tmax_ours": float(ours.max()), "Tmax_heat1d": float(ts.max()),
                "Tmin_ours": float(ours.min()), "Tmin_heat1d": float(ts.min()),
                "nodes": int(p.z.size), "cells_ours": int(centres.size)}
            np.save(out / f"heat1d-L{i}-{name.replace(' ', '_')}.npy", ts)
        report["levels"] = sorted(
            [lv for lv in report["levels"] if lv["level"] != i] + [level],
            key=lambda lv: lv["level"])
        # A retried level supersedes any verdict that used its old series.
        report.pop("verdict_R2", None)
        if i < len(LEVELS):
            for key in ("verdict", "pass", "verdict_note"):
                report.pop(key, None)
        (out / "gi.json").write_text(json.dumps(report, indent=1))
        print(json.dumps(level["columns"], indent=1), flush=True)
    try:
        _require_heat1d_periodicity(report, len(levels))
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1
    return 0


def warm_start(prev_run, cfg):
    """The end-of-cycle state of a verified periodic point run, interpolated
    onto ``cfg``'s grid via its FV reconstruction (anomalies, K - 300)."""
    from tropoi.run.airless.experiment import build_substrate
    from tools.planet_one.convergence import _reconstruction
    meta = json.loads((prev_run / "config.json").read_text())["config"]
    grid = np.load(prev_run / "grid.npz")
    ev = load_tier2(prev_run / "retained", "every_step")
    sub = build_substrate(cfg)
    cols = ev.shape[2]
    anom = np.empty((sub.layers, cols))
    skin = np.empty(cols)
    for c in range(cols):
        z, T = _reconstruction(meta, grid, ev[-1:, 0, c], ev[-1:, 1:, c])
        anom[:, c] = np.interp(sub.centres, z, T[0]) - 300.0
        skin[c] = ev[-1, 0, c] - 300.0
    return anom, skin


def verdict4(outdir) -> dict:
    """R2 (register): the four-level G-I evaluation, exactly as frozen.

    * Finest maximum: every native level-3 sample.
    * d = max|Q3 - Q2| on the level-2 times; p from levels 1, 2 and 3 on the
      level-1 times; e = d / (2^p - 1), or 2 d if p <= 0.5.
    * Satisfied for a column iff L3 max <= e_ours + e_heat1d + 0.2 K, the RMS
      decreases at every level, and the L3 native max is below L2's.
    * Also reported: the maximum within the 0E sunrise window (cycle fraction
      0.749-0.752) at native resolution for every level.

    R1's three-level verdict is left in gi.json untouched."""
    out = pathlib.Path(outdir)
    rep = json.loads((out / "gi.json").read_text())
    _require_heat1d_periodicity(rep, 4)
    res = {}
    for name in COLUMNS:
        tag = name.replace(" ", "_")
        c = list(COLUMNS).index(name)
        o = [load_tier2(out / f"ours-L{i}" / "retained", "every_step")[:, 0, c] for i in range(4)]
        h = [np.load(out / f"heat1d-L{i}-{tag}.npy") for i in range(4)]
        at = lambda a, i, base: a[i][2 ** (i - base) - 1::2 ** (i - base)]   # level i at level-base times
        def order(seq):
            q1 = [at(seq, i, 1) for i in (1, 2, 3)]
            d1 = float(np.max(np.abs(q1[1] - q1[0])))
            d2 = float(np.max(np.abs(q1[2] - q1[1])))
            return math.log2(d1 / d2) if d1 > 0 and d2 > 0 else 0.0
        po, ph = order(o), order(h)
        do = float(np.max(np.abs(at(o, 3, 2) - o[2])))
        dh = float(np.max(np.abs(at(h, 3, 2) - h[2])))
        eo = do / (2 ** po - 1) if po > 0.5 else 2 * do
        eh = dh / (2 ** ph - 1) if ph > 0.5 else 2 * dh
        native = [float(np.max(np.abs(o[i] - h[i]))) for i in range(4)]
        rms = [float(np.sqrt(np.mean((o[i] - h[i]) ** 2))) for i in range(4)]
        sunrise = []
        for i in range(4):
            f = (np.arange(o[i].size) + 1) / o[i].size
            w = (f >= 0.749) & (f <= 0.752)
            sunrise.append(float(np.max(np.abs(o[i][w] - h[i][w]))))
        bound = eo + eh + 0.2
        k = int(np.argmax(np.abs(o[3] - h[3])))
        ok = native[3] <= bound and all(rms[i] > rms[i + 1] for i in range(3)) and native[3] < native[2]
        res[name] = {"finest_native_max_K": native[3], "native_max_by_level_K": native,
                     "rms_by_level_K": rms, "sunrise_window_max_by_level_K": sunrise,
                     "at_cycle_fraction": (k + 1) / o[3].size, "e_ours_K": eo, "e_heat1d_K": eh,
                     "order_ours": po, "order_heat1d": ph, "bound_K": bound, "satisfied": bool(ok)}
    rep["verdict_R2"] = {"columns": res, "satisfied": all(r["satisfied"] for r in res.values()),
                         "note": "R2 four-level evidence; R1's three-level FAIL stays recorded"}
    (out / "gi.json").write_text(json.dumps(rep, indent=1))
    return rep


def verdict(outdir) -> dict:
    """The frozen G-I criterion: the difference shrinks under joint
    refinement, and at the finest level max|dTs| <= e_ours + e_heat1d + 0.2 K,
    where each e is that code's own Richardson estimate.

    Strict evaluation (decides pass/fail): max|dTs| over the finest level's
    own samples, against a bound whose error estimates come from the
    densest time set where both codes have two levels. That set is the
    level-1 times: d = max|Q2 - Q1|, and e = d / (2^p - 1), with the order p
    taken from the three-level estimate on the level-0 times.

    Diagnostic only: the same statistics sampled at the coarsest level's
    times. That sampling was chosen after the levels were seen, so it never
    decides the verdict."""
    out = pathlib.Path(outdir)
    rep = json.loads((out / "gi.json").read_text())
    _require_heat1d_periodicity(rep, len(LEVELS))
    res = {}
    for name in COLUMNS:
        tag = name.replace(" ", "_")
        c = list(COLUMNS).index(name)
        o = [load_tier2(out / f"ours-L{i}" / "retained", "every_step")[:, 0, c]
             for i in range(len(LEVELS))]
        h = [np.load(out / f"heat1d-L{i}-{tag}.npy") for i in range(len(LEVELS))]
        oc = [o[i][2 ** i - 1::2 ** i] for i in range(3)]
        hc = [h[i][2 ** i - 1::2 ** i] for i in range(3)]

        def order(seq):
            d1 = float(np.max(np.abs(seq[1] - seq[0])))
            d2 = float(np.max(np.abs(seq[2] - seq[1])))
            return (math.log2(d1 / d2) if d1 > 0 and d2 > 0 else 0.0), d2
        po, d2o_c = order(oc)
        ph, d2h_c = order(hc)
        d2o = float(np.max(np.abs(o[2][1::2] - o[1])))
        d2h = float(np.max(np.abs(h[2][1::2] - h[1])))
        eo = d2o / (2 ** po - 1) if po > 0.5 else 2 * d2o
        eh = d2h / (2 ** ph - 1) if ph > 0.5 else 2 * d2h
        native = [float(np.max(np.abs(o[i] - h[i]))) for i in range(3)]
        rms = [float(np.sqrt(np.mean((o[i] - h[i]) ** 2))) for i in range(3)]
        k = int(np.argmax(np.abs(o[2] - h[2])))
        bound = eo + eh + 0.2
        strict = native[2] <= bound and rms[0] > rms[1] > rms[2] and native[2] < native[1]
        coarse = [float(np.max(np.abs(oc[i] - hc[i]))) for i in range(3)]
        eo_c = d2o_c / (2 ** po - 1) if po > 0.5 else 2 * d2o_c
        eh_c = d2h_c / (2 ** ph - 1) if ph > 0.5 else 2 * d2h_c
        res[name] = {
            "strict": {"max_abs_dTs_native_by_level_K": native, "rms_by_level_K": rms,
                       "finest_max_K": native[2], "at_cycle_fraction": (k + 1) / len(o[2]),
                       "ours_K": float(o[2][k]), "heat1d_K": float(h[2][k]),
                       "e_ours_K": eo, "e_heat1d_K": eh, "order_ours": po, "order_heat1d": ph,
                       "bound_K": bound, "pass": bool(strict)},
            "diagnostic_coarse_times": {"max_abs_dTs_by_level_K": coarse,
                                        "bound_K": eo_c + eh_c + 0.2,
                                        "would_pass": coarse[2] <= eo_c + eh_c + 0.2}}
    rep["verdict"] = res
    rep["pass"] = all(r["strict"]["pass"] for r in res.values())
    rep["verdict_note"] = ("Strict evaluation decides. At 0N 0E the finest-level maximum "
                           "occurs at the first sunrise step. A failure here needs an owner "
                           "decision; the diagnostic sampling must not be used to pass.")
    (out / "gi.json").write_text(json.dumps(rep, indent=1))
    return rep


if __name__ == "__main__":
    sys.exit(main())
