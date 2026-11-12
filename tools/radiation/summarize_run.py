"""Summarise a finished radiative run and measure its momentum sinks (GPU).

Copies the small run records (``config.json``, ``events.jsonl``,
``series.json``, ``means.npz``) into ``docs/radiation/evidence/<name>/`` and
writes ``summary.json`` and ``climate.png`` next to them:

* the daily series at a few days and averaged over the last ``--window`` days;
* the zonal-mean climate of the run's own averaging window (jets, surface
  temperature contrast, eddy fluxes);
* for every saved state in the window, the rate at which the Rayleigh drag
  and the hyperdiffusion change the kinetic energy and the enthalpy,
  evaluated as a centred difference of ``global_energy`` along each
  operator's own tendency. Neither operator returns the kinetic energy it
  removes as heat, so this is the part of the energy-budget residual the
  model's formulation accounts for; what is left is reported as unexplained.

The sinks are sampled once per saved state (every ``state_every_days``) while
the residual is a daily mean, so the comparison carries sampling noise; the
standard error over the sampled states is reported with it.

    python tools/radiation/summarize_run.py runs/radiative/<run> --name <name>
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import shutil

import numpy as np

EVIDENCE = pathlib.Path(__file__).resolve().parents[2] / "docs" / "radiation" / "evidence"
RECORDS = ("config.json", "events.jsonl", "series.json", "means.npz")
BUDGET = ("toa_net_down", "olr", "absorbed_stellar", "bottom_net_down", "sensible", "interior",
          "surface_net", "column_energy_tendency", "energy_rate", "energy_residual",
          "energy_kinetic", "max_abs_u", "mean_surface_temperature", "t_min", "t_max")


def series_summary(series: dict, window: int) -> dict:
    days = series["day"]
    last = days[-1]

    def at(key, day):
        values = series[key]
        # energy_rate / energy_residual have no day-0 entry
        return values[day - (len(days) - len(values))]

    picks = [d for d in (1, 10, 30, 100, 200, 300, last) if d <= last]
    rows = {str(d): {k: at(k, d) for k in BUDGET if k in series} for d in dict.fromkeys(picks)}
    tail = {}
    for k in BUDGET:
        if k in series:
            v = np.asarray(series[k][-window:], dtype=float)
            tail[k] = {"mean": float(v.mean()), "std": float(v.std(ddof=1)),
                       "min": float(v.min()), "max": float(v.max())}
    ps = series["mean_ps"]
    return {"last_day": last, "window_days": window, "at_day": rows, "window": tail,
            "mean_surface_pressure_drift": ps[-1] / ps[0] - 1.0,
            "max_abs_energy_residual": float(np.abs(series["energy_residual"]).max())}


def climate_summary(means) -> dict:
    lat = means["lat"]                                   # degrees, north first
    sigma = means["sigma"]
    count = int(means["count"])
    f = {k: means[k] for k in ("u", "v", "T", "usvs", "vsTs")}   # already time means
    out = {"samples": count, "latitudes": int(lat.size), "levels": int(sigma.size)}

    def peak(field, mask, sign=1.0):
        a = np.where(mask[:, None], sign * field, -np.inf)
        i, k = np.unravel_index(int(np.argmax(a)), a.shape)
        return {"value": float(field[i, k]), "latitude_deg": float(lat[i]),
                "sigma": float(sigma[k])}

    north, south = lat > 0, lat < 0
    out["westerly_jet_north"] = peak(f["u"], north)
    out["westerly_jet_south"] = peak(f["u"], south)
    out["strongest_easterly"] = peak(f["u"], np.ones_like(north), -1.0)
    out["lowest_level_wind_range"] = [float(f["u"][:, -1].min()), float(f["u"][:, -1].max())]
    low = f["T"][:, -1]
    eq = int(np.argmin(np.abs(lat)))
    out["lowest_level_temperature"] = {"near_equator": float(low[eq]),
                                       "southmost": float(low[np.argmin(lat)]),
                                       "northmost": float(low[np.argmax(lat)])}
    out["temperature_range"] = [float(f["T"].min()), float(f["T"].max())]
    out["eddy_heat_flux_poleward_north"] = peak(f["vsTs"], north)
    out["eddy_heat_flux_poleward_south"] = peak(f["vsTs"], south, -1.0)
    out["eddy_momentum_flux_north"] = peak(f["usvs"], north)
    out["eddy_momentum_flux_south"] = peak(f["usvs"], south, -1.0)
    hemi = np.abs(f["u"] - f["u"][::-1]).max() / max(np.abs(f["u"]).max(), 1e-30)
    out["hemispheric_asymmetry_of_u"] = float(hemi)
    return out, lat, sigma, f


def figure(path: pathlib.Path, lat, sigma, f, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    umax = float(np.abs(f["u"]).max()) or 1.0
    c0 = axes[0].contourf(lat, sigma, f["u"].T, levels=np.linspace(-umax, umax, 21), cmap="RdBu_r")
    fig.colorbar(c0, ax=axes[0], label="zonal-mean u (m/s)")
    c1 = axes[1].contourf(lat, sigma, f["T"].T, levels=20, cmap="viridis")
    fig.colorbar(c1, ax=axes[1], label="zonal-mean T (K)")
    for ax in axes:
        ax.set_xlabel("latitude (deg)")
        ax.set_ylabel("sigma")
        ax.invert_yaxis()
    fig.suptitle(title)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def sinks(run: pathlib.Path, cfg_doc: dict, first_day: int, eps: float = 50.0) -> dict:
    """Energy tendency of each momentum sink on every saved state from
    ``first_day`` on, W m^-2 (negative removes energy)."""
    import cupy as cp
    from tropoi.run.radiative.config import RadiativeConfig
    from tropoi.run.radiative.model import build_model, build_stepper
    from tropoi.temporal.column_coupling import global_energy
    cfg = RadiativeConfig.from_dict(cfg_doc["config"])
    model = build_model(cfg)
    stepper = build_stepper(cfg, model, None)

    operators = {}
    for damper in stepper.dampers:
        rates = cp.asarray(damper.rates)[:, :, None]
        operators[damper.name] = lambda x, rates=rates: -rates * x
    for term in stepper.lagged_terms:
        operators[term.signature()["kind"]] = lambda x, term=term: term(x, 0.0, 0.0)

    def rate(x, tendency):
        up = global_energy(model, x + eps * tendency, cfg.gravity)
        down = global_energy(model, x - eps * tendency, cfg.gravity)
        return {k: (up[k] - down[k]) / (2.0 * eps) for k in ("kinetic", "enthalpy", "total")}

    rows = []
    for path in sorted((run / "states").glob("state-d*.npy")):
        day = int(re.search(r"d(\d+)", path.name).group(1))
        if day < first_day:
            continue
        x = cp.asarray(np.load(path))
        row = {"day": day}
        for name, op in operators.items():
            row[name] = rate(x, op(x))
        row["all_sinks_total"] = sum(row[name]["total"] for name in operators)
        rows.append(row)
    names = list(operators)
    mean = {n: {k: float(np.mean([r[n][k] for r in rows])) for k in ("kinetic", "enthalpy", "total")}
            for n in names}
    totals = np.array([r["all_sinks_total"] for r in rows])
    return {"states": rows, "operators": names, "mean": mean,
            "all_sinks_total_mean": float(totals.mean()),
            "all_sinks_total_standard_error": float(totals.std(ddof=1) / np.sqrt(totals.size))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run", type=pathlib.Path)
    ap.add_argument("--name", required=True, help="evidence directory name")
    ap.add_argument("--window", type=int, default=100, help="trailing days to average")
    ap.add_argument("--no-sinks", action="store_true", help="skip the GPU sink measurement")
    args = ap.parse_args()

    out = EVIDENCE / args.name
    out.mkdir(parents=True, exist_ok=True)
    for name in RECORDS:
        shutil.copyfile(args.run / name, out / name)
    cfg_doc = json.loads((args.run / "config.json").read_text(encoding="utf-8"))
    series = json.loads((args.run / "series.json").read_text(encoding="utf-8"))
    events = [json.loads(line) for line in
              (args.run / "events.jsonl").read_text(encoding="utf-8").splitlines() if line]
    summary = {"experiment_id": cfg_doc["config"]["experiment_id"],
               "config_sha256": cfg_doc["config_sha256"], "code": cfg_doc["code"],
               "environment": events[0].get("environment"),
               "events": {"first": events[0]["time"], "last": events[-1]["time"],
                          "last_event": events[-1]["event"],
                          "resumes": sum(e["event"] == "resumed" for e in events)},
               "series": series_summary(series, args.window)}
    climate, lat, sigma, fields = climate_summary(np.load(args.run / "means.npz"))
    summary["climate"] = climate
    figure(out / "climate.png", lat, sigma, fields,
           f"{summary['experiment_id']}: mean of {climate['samples']} daily samples")
    if not args.no_sinks:
        first = summary["series"]["last_day"] - args.window
        measured = sinks(args.run, cfg_doc, first)
        residual = summary["series"]["window"]["energy_residual"]["mean"]
        measured["energy_residual_window_mean"] = residual
        measured["unexplained"] = residual - measured["all_sinks_total_mean"]
        summary["sinks"] = measured
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n",
                                      encoding="utf-8")
    print(json.dumps({k: summary[k] for k in summary if k != "environment"}, indent=1))


if __name__ == "__main__":
    main()
