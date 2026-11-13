"""Judge the flat-Mars pilot against criteria M1-M6 (docs/radiation/MARS_MI_PLAN.md).

CPU only. Reads the two evidence summaries written by ``summarize_run.py``
(``series.json``, ``summary.json`` with the per-state sinks) and the H
run's ``tides.json``; writes ``verdicts.json`` next to H's summary. The
criteria are applied exactly as fixed in commit fba92fb.

    python tools/radiation/judge_mars_pilot.py H_EVIDENCE U_EVIDENCE
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

EVIDENCE = pathlib.Path(__file__).resolve().parents[2] / "docs" / "radiation" / "evidence"
WINDOW = (11, 30)          # days whose daily records enter the window means


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
    out["M1"] = (out["last_day"] == 30.0 and out["max_abs_u"] < 150.0
                 and 100.0 < out["t_min"] and out["t_max"] < 400.0)
    out["M2"] = (abs(out["residual_window_mean"]) <= 1e-3 * F
                 and out["residual_max_abs"] <= 1e-2 * F)
    out["M3"] = (abs(out["system_residual_window_mean"]) <= 1e-3 * F
                 and out["system_residual_max_abs"] <= 1e-2 * F
                 and out["slab_max_relative_error"] <= 1e-9)
    out["M6"] = abs(out["ps_drift"]) <= 1e-5
    return out


def main() -> None:
    h_name, u_name = sys.argv[1], sys.argv[2]
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
    tides = json.loads((EVIDENCE / h_name / "tides.json").read_text(encoding="utf-8"))
    verdicts = {
        "criteria_commit": "fba92fb", "window_days": list(WINDOW),
        "H": h, "U": u, "M4": m4, "M5": tides["M5"],
        "verdict": {"M1": bool(h["M1"] and u["M1"]), "M2": bool(h["M2"]), "M3": bool(h["M3"]),
                    "M4": m4["met"], "M5": bool(tides["M5"]["met"]),
                    "M6": bool(h["M6"] and u["M6"])},
        "reported": {"H_minus_U_surface_temperature_last": h["surface_temperature_last"]
                     - u["surface_temperature_last"],
                     "H_minus_U_toa_window_mean": h["toa_window_mean"] - u["toa_window_mean"],
                     "tides": {k: v for k, v in tides.items() if k not in ("by_latitude", "M5")}},
    }
    (EVIDENCE / h_name / "verdicts.json").write_text(
        json.dumps(verdicts, indent=2, sort_keys=True, default=float) + "\n", encoding="utf-8")
    print(json.dumps(verdicts["verdict"], indent=2))
    print(json.dumps({"H": h, "U": u, "M4": m4}, indent=2, default=float))


if __name__ == "__main__":
    main()
