"""Diurnal response and thermal tides of a radiative run (GPU), criterion M5.

Rebuilds the model of a finished run from its ``config.json``, loads its last
checkpoint read-only (the run directory is not modified: nothing is written
to it and no event is appended), and integrates ``--sols`` more sols in
memory, recording surface pressure, surface temperature and lowest-level air
temperature at every step.

Each field is fitted per latitude in the sun-synchronous basis of the
insolation, ``h = longitude + 2 pi t / solar_day`` (``h = 0`` is local noon):

    x(lon, t) = a(lon) + b(lon) t + sum_{n=1,2} [A_n cos(n h) + B_n sin(n h)],

so S1 and S2 are the migrating diurnal and semidiurnal components, the
per-longitude mean and linear trend absorb the spin-up drift, and what is
left (``noise``) is everything else. Writes ``tides.json`` into the named
evidence directory.

    python tools/radiation/mars_tides.py RUN_DIR --name EVIDENCE_NAME [--sols 2]
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib

import numpy as np

EVIDENCE = pathlib.Path(__file__).resolve().parents[2] / "docs" / "radiation" / "evidence"


def _restore(run_dir: pathlib.Path):
    import cupy as cp
    from tropoi.run.held_suarez.checkpoint import list_checkpoints, read_checkpoint
    from tropoi.run.radiative.config import RadiativeConfig
    from tropoi.run.radiative.model import build_model, build_physics, build_stepper
    from tropoi.temporal.column_coupling import ColumnPhysicsCoupling
    doc = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    cfg = RadiativeConfig.from_dict(doc["config"])
    if cfg.sha256() != doc["config_sha256"]:
        raise SystemExit(f"{run_dir}: config.json does not reproduce its own hash")
    arrays, meta = read_checkpoint(list_checkpoints(run_dir / "checkpoints")[-1][1])
    model = build_model(cfg)
    physics = build_physics(cfg)
    coupling = ColumnPhysicsCoupling(model, physics, cfg.dt,
                                     surface_temperature=cp.asarray(arrays["surface_temperature"]))
    stepper = build_stepper(cfg, model, coupling)
    sd = dict(meta["stepper"])
    sd["x_curr"] = cp.asarray(arrays["x_curr"])
    sd["x_prev"] = cp.asarray(arrays["x_prev"])
    stepper.load_state_dict(sd)
    c = dict(meta["coupling"] or {})
    c["surface_temperature"] = arrays["surface_temperature"]
    coupling.load_state_dict(c)
    return cfg, model, coupling, stepper, physics


def _fit(lat, lon, t, field, solar_day):
    """Per-latitude least squares; returns {lat: (S1 amp, S1 h_max, S2 amp, noise rms, mean)}."""
    out = {}
    for la in np.unique(lat):
        sel = lat == la
        lons = lon[sel]
        x = field[:, sel]                                       # (nt, nlon)
        nt, nl = x.shape
        h = lons[None, :] + 2.0 * math.pi * t[:, None] / solar_day
        cols = []
        for j in range(nl):                                     # per-longitude mean and trend
            e = np.zeros((nt, nl)); e[:, j] = 1.0
            cols += [e, e * (t[:, None] - t.mean())]
        for n in (1, 2):
            cols += [np.cos(n * h), np.sin(n * h)]
        design = np.stack([c.ravel() for c in cols], axis=1)
        coef, *_ = np.linalg.lstsq(design, x.ravel(), rcond=None)
        resid = x.ravel() - design @ coef
        a1, b1, a2, b2 = coef[-4:]
        out[float(la)] = {"s1_amplitude": float(math.hypot(a1, b1)),
                          "s1_h_of_max": float(math.atan2(b1, a1)),
                          "s2_amplitude": float(math.hypot(a2, b2)),
                          "noise_rms": float(np.sqrt(np.mean(resid ** 2))),
                          "mean": float(x.mean()), "coef_s1_s2": [float(v) for v in coef[-4:]]}
    return out


def _local_hour(h):
    return (12.0 + h * 12.0 / math.pi) % 24.0


def main() -> None:
    import cupy as cp
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=pathlib.Path)
    ap.add_argument("--name", required=True)
    ap.add_argument("--sols", type=float, default=2.0)
    args = ap.parse_args()
    cfg, model, coupling, stepper, physics = _restore(args.run)
    solar_day = float(physics.insolation.solar_day)
    K = model.nlev
    lat = np.asarray(cp.asnumpy(coupling.latitude))
    lon = np.asarray(cp.asnumpy(coupling.longitude))
    steps = int(round(args.sols * solar_day / cfg.dt))
    t0 = stepper.t
    times, ps, ts, ta = [], [], [], []
    for _ in range(steps):
        stepper.step()
        x = stepper.state
        times.append(stepper.t - t0)
        ps.append(cp.asnumpy(cp.exp(model.sh.inv_transform(x[3 * K]).real).ravel()))
        ts.append(cp.asnumpy(coupling.surface_temperature).ravel())
        ta.append(cp.asnumpy(model.sh.inv_transform(x[3 * K - 1]).real.ravel()))
    t = np.asarray(times) + t0
    fits = {name: _fit(lat, lon, t, np.asarray(series), solar_day)
            for name, series in (("surface_pressure", ps), ("surface_temperature", ts),
                                 ("lowest_air_temperature", ta))}
    # M5: equatorial T_s diurnal maximum (S1 + S2 curve) and the tropical S1 of p_s
    eq = min(fits["surface_temperature"], key=abs)
    a1, b1, a2, b2 = fits["surface_temperature"][eq]["coef_s1_s2"]
    hh = np.linspace(-math.pi, math.pi, 4321)
    curve = a1 * np.cos(hh) + b1 * np.sin(hh) + a2 * np.cos(2 * hh) + b2 * np.sin(2 * hh)
    ts_max_hour = _local_hour(float(hh[np.argmax(curve)]))
    tropics = [la for la in fits["surface_pressure"] if abs(la) <= math.radians(30.0)]
    w = np.cos(tropics)
    rel = lambda key: float(np.sum(w * [fits["surface_pressure"][la][key]
                                        / fits["surface_pressure"][la]["mean"] for la in tropics])
                            / w.sum())
    s1, s2, noise = rel("s1_amplitude"), rel("s2_amplitude"), rel("noise_rms")
    summary = {
        "run": args.run.name, "config_sha256": cfg.sha256(), "sols": args.sols, "steps": steps,
        "start_seconds": t0, "solar_day": solar_day,
        "equatorial_latitude_deg": math.degrees(eq),
        "equatorial_surface_temperature_max_local_hour": ts_max_hour,
        "equatorial_surface_temperature_s1_amplitude_k":
            fits["surface_temperature"][eq]["s1_amplitude"],
        "equatorial_lowest_air_temperature_s1_amplitude_k":
            fits["lowest_air_temperature"][eq]["s1_amplitude"],
        "tropical_surface_pressure_s1_over_ps": s1,
        "tropical_surface_pressure_s2_over_ps": s2,
        "tropical_surface_pressure_noise_rms_over_ps": noise,
        "tropical_surface_pressure_s1_max_local_hour":
            _local_hour(fits["surface_pressure"][min(tropics, key=abs)]["s1_h_of_max"]),
        "M5": {"surface_temperature_max_between_11_and_16_h": 11.0 <= ts_max_hour <= 16.0,
               "s1_at_least_5e-4": s1 >= 5e-4, "s1_at_least_3x_noise": s1 >= 3.0 * noise},
        "by_latitude": {k: {f"{math.degrees(la):.2f}": {kk: vv for kk, vv in v.items()
                                                       if kk != "coef_s1_s2"}
                            for la, v in f.items()} for k, f in fits.items()},
    }
    summary["M5"]["met"] = all(summary["M5"].values())
    out = EVIDENCE / args.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "tides.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n",
                                    encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "by_latitude"}, indent=2))


if __name__ == "__main__":
    main()
