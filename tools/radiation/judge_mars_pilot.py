"""Judge the flat-Mars pilot against criteria M1-M6 (docs/radiation/MARS_MI_PLAN.md).

CPU only. Reads the two evidence summaries written by ``summarize_run.py``
(``series.json``, ``summary.json`` with the per-state sinks) and the H
run's ``tides.json``; writes ``verdicts.json`` next to H's summary. The
criteria are applied exactly as fixed in commit fba92fb.

    python tools/radiation/judge_mars_pilot.py H_EVIDENCE U_EVIDENCE [--tides FILE --out-file FILE]

``--tides`` names the M5 sample inside H's evidence directory (default
``tides.json``); ``--out-file`` the verdict file (default
``verdicts.json``). An existing verdict file is never overwritten.

``--criteria 2e`` judges a boundary-layer pair as pre-registered in plan
§2e (commit 04f4547): M6 becomes the scaled pair (|drift| <= 1e-5 to day
30 AND <= 2e-5 over the run), M7 bounds the daily max lowest-level wind
speed (``max_wind_bottom``) by 40 m/s over days WINDOW[0]..DAYS, and the
report adds the boundary layer's kinetic-energy sink and the two
fixer-equivalent series (the per-day mass factor Isca's mass correction
would have applied, and the uniform temperature increment per day its
energy correction would have applied). The default leaves every earlier
verdict path untouched.
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

EVIDENCE = pathlib.Path(__file__).resolve().parents[2] / "docs" / "radiation" / "evidence"
WINDOW = (11, 30)          # default: days whose daily records enter the window means
DAYS = 30                  # default: the configured last day (M1)


def _load(name):
    d = EVIDENCE / name
    series = json.loads((d / "series.json").read_text(encoding="utf-8"))
    summary = json.loads((d / "summary.json").read_text(encoding="utf-8"))
    config = json.loads((d / "config.json").read_text(encoding="utf-8"))
    return series, summary, config


def _assess(series, config):
    day = np.asarray(series["day"], dtype=float)
    res = np.asarray(series["energy_residual"], dtype=float)          # days 1..N
    rday = day[1:]
    win = (rday >= WINDOW[0]) & (rday <= WINDOW[1])
    absorbed = np.asarray(series["absorbed_stellar"], dtype=float)[1:]
    F = float(absorbed[win].mean())
    cap = float(config["config"]["surface"]["heat_capacity"])
    e_atm = np.asarray(series["energy_total"], dtype=float)
    ts = np.asarray(series["mean_surface_temperature"], dtype=float)
    toa = np.asarray(series["toa_net_down"], dtype=float)[1:]
    system = (np.diff(e_atm) + cap * np.diff(ts)) / 86400.0 - toa
    surface_net = np.asarray(series["surface_net"], dtype=float)[1:]
    slab = cap * np.diff(ts) / (surface_net * 86400.0) - 1.0
    ps = np.asarray(series["mean_ps"], dtype=float)
    out = {
        "F": F, "last_day": float(day[-1]),
        "max_abs_u": float(np.max(series["max_abs_u"])),
        "t_min": float(np.min(series["t_min"])), "t_max": float(np.max(series["t_max"])),
        "residual_window_mean": float(res[win].mean()),
        "residual_window_std": float(res[win].std(ddof=1)),
        "residual_max_abs": float(np.abs(res).max()),
        "system_residual_window_mean": float(system[win].mean()),
        "system_residual_max_abs": float(np.abs(system).max()),
        "slab_max_relative_error": float(np.abs(slab).max()),
        "ps_drift": float(ps[-1] / ps[0] - 1.0),
        "toa_window_mean": float(toa[win].mean()),
        "surface_temperature_last": float(ts[-1]),
    }
    out["M1"] = (out["last_day"] == float(DAYS) and out["max_abs_u"] < 150.0
                 and 100.0 < out["t_min"] and out["t_max"] < 400.0)
    out["M2"] = (abs(out["residual_window_mean"]) <= 1e-3 * F
                 and out["residual_max_abs"] <= 1e-2 * F)
    out["M3"] = (abs(out["system_residual_window_mean"]) <= 1e-3 * F
                 and out["system_residual_max_abs"] <= 1e-2 * F
                 and out["slab_max_relative_error"] <= 1e-9)
    out["M6"] = abs(out["ps_drift"]) <= 1e-5
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("h_evidence")
    ap.add_argument("u_evidence")
    ap.add_argument("--tides", default="tides.json")
    ap.add_argument("--out-file", default="verdicts.json")
    ap.add_argument("--window", type=int, nargs=2, default=list(WINDOW),
                    metavar=("FIRST", "LAST"), help="days of the M2-M4 window means")
    ap.add_argument("--days", type=int, default=DAYS, help="configured last day (M1)")
    ap.add_argument("--criteria", choices=["2d", "2e"], default="2d",
                    help="2e: the boundary-layer pair of plan 2e (scaled M6, M7, "
                         "fixer-equivalent report)")
    args = ap.parse_args()
    globals()["WINDOW"], globals()["DAYS"] = tuple(args.window), args.days
    h_name, u_name = args.h_evidence, args.u_evidence
    (hs, hsum, hcfg), (us, usum, ucfg) = _load(h_name), _load(u_name)
    h, u = _assess(hs, hcfg), _assess(us, ucfg)
    sinks = usum["sinks"]
    states = [s for s in sinks["states"] if WINDOW[0] <= s["day"] <= WINDOW[1]]
    totals = np.array([s["all_sinks_total"] for s in states])
    sink_mean = float(totals.mean())
    sink_se = float(totals.std(ddof=1) / np.sqrt(totals.size))
    tolerance = max(2.0 * sink_se, 1e-3 * u["F"])
    m4 = {"unheated_residual": u["residual_window_mean"], "unheated_sinks": sink_mean,
          "sinks_standard_error": sink_se, "states": int(totals.size), "tolerance": tolerance,
          "difference": u["residual_window_mean"] - sink_mean,
          "met": bool(abs(u["residual_window_mean"] - sink_mean) <= tolerance and h["M2"])}
    tides = json.loads((EVIDENCE / h_name / args.tides).read_text(encoding="utf-8"))
    target = EVIDENCE / h_name / args.out_file
    if target.exists():
        raise SystemExit(f"{target} exists; choose another --out-file (evidence is not overwritten)")
    verdicts = {
        "criteria_commit": "fba92fb", "window_days": list(WINDOW), "tides_file": args.tides,
        "H": h, "U": u, "M4": m4, "M5": tides["M5"],
        "verdict": {"M1": bool(h["M1"] and u["M1"]), "M2": bool(h["M2"]), "M3": bool(h["M3"]),
                    "M4": m4["met"], "M5": bool(tides["M5"]["met"]),
                    "M6": bool(h["M6"] and u["M6"])},
        "reported": {"H_minus_U_surface_temperature_last": h["surface_temperature_last"]
                     - u["surface_temperature_last"],
                     "H_minus_U_toa_window_mean": h["toa_window_mean"] - u["toa_window_mean"],
                     "tides": {k: v for k, v in tides.items() if k not in ("by_latitude", "M5")}},
    }
    if args.criteria == "2e":
        verdicts["criteria_commit"] = "04f4547 (plan 2e)"
        for name, series, assessed in (("H", hs, h), ("U", us, u)):
            day = np.asarray(series["day"], dtype=float)
            ps = np.asarray(series["mean_ps"], dtype=float)
            d30 = np.nonzero(day == 30.0)[0]
            assessed["ps_drift_day30"] = (float(ps[d30[0]] / ps[0] - 1.0) if d30.size
                                          else None)
            assessed["M6"] = bool(assessed["ps_drift_day30"] is not None
                                  and abs(assessed["ps_drift_day30"]) <= 1e-5
                                  and abs(assessed["ps_drift"]) <= 2e-5)
            m7_win = (day >= WINDOW[0]) & (day <= DAYS)
            bottom = np.asarray(series["max_wind_bottom"], dtype=float)
            assessed["max_wind_bottom"] = float(bottom[m7_win].max())
            assessed["M7"] = bool(assessed["max_wind_bottom"] <= 40.0
                                  and assessed["last_day"] == float(DAYS))
        verdicts["verdict"]["M6"] = bool(h["M6"] and u["M6"])
        verdicts["verdict"]["M7"] = bool(h["M7"] and u["M7"])
        # Fixer-equivalent series (plan 2e, reported, never applied): the
        # per-day mass factor Isca's do_mass_correction would have applied,
        # and the uniform temperature increment per day (K/day) that its
        # do_energy_correction would have removed, from H's own records.
        day = np.asarray(hs["day"], dtype=float)
        ps = np.asarray(hs["mean_ps"], dtype=float)
        res = np.asarray(hs["energy_residual"], dtype=float)          # days 1..N, W/m^2
        cp_heat = float(hcfg["config"]["cp"])
        gravity = float(hcfg["config"]["gravity"])
        mass_factor = ps[:-1] / ps[1:]
        energy_dt = -res * 86400.0 * gravity / (cp_heat * ps[1:])
        verdicts["reported"]["fixer_equivalent"] = {
            "mass_factor_day_by_day": [float(x) for x in mass_factor],
            "mass_factor_log_sum": float(np.log(mass_factor).sum()),
            "energy_uniform_dT_per_day": [float(x) for x in energy_dt],
            "energy_uniform_dT_mean": float(energy_dt.mean()),
        }
        if "bl_kinetic_sink" in hs:
            sink = np.asarray(hs["bl_kinetic_sink"], dtype=float)[1:]
            win = (day[1:] >= WINDOW[0]) & (day[1:] <= WINDOW[1])
            verdicts["reported"]["bl_kinetic_sink_window_mean"] = float(sink[win].mean())
            verdicts["reported"]["bl_max_depth_max"] = float(
                np.max(hs["bl_max_depth"]))
    target.write_text(
        json.dumps(verdicts, indent=2, sort_keys=True, default=float) + "\n", encoding="utf-8")
    print(json.dumps(verdicts["verdict"], indent=2))
    print(json.dumps({"H": h, "U": u, "M4": m4}, indent=2, default=float))


if __name__ == "__main__":
    main()
