"""The winds behind the M-I audit's M1/M7 numbers, read from a finished run's
saved daily states (GPU, read-only).

``MARS_MI_PLAN.md`` §4.6 records that M7 was registered as the daily max of
the lowest-full-level ``|u|`` while the judge read ``max_wind_bottom``, the
lowest-level horizontal *speed*. This tool recomputes both, and the
zonal-mean winds of ``MARS_MI_REQUIREMENTS_AUDIT.md`` §3, from the saved
states ``states/state-dNNNNN.npy`` (and ``surface-dNNNNN.npy``) of a
radiative run. For every saved day it records:

* the grid-point max ``|u|`` with its sigma and latitude (checked against the
  run's ``series.json`` ``max_abs_u`` to a relative 1e-9);
* the lowest-level max ``|u|`` (the registered M7 observable);
* the lowest-level max horizontal speed, from
  ``ColumnPhysicsCoupling._lowest_wind_speed``, the function that writes
  ``max_wind_bottom`` (checked against ``series.json`` to a relative 1e-9
  when the run has that series; the judged M7 observable);
* the lowest-level max ``|v|``;
* the zonal-mean u max and min, with the max's sigma and latitude, and the
  zonal-mean max restricted to sigma > 0.01 (below about 6 Pa).

A mismatch with ``series.json`` aborts: the states are then not the run's own.
The summary covers days ``--first-day``..last (11, the M7 window): the
largest value and its day for each observable, and the number of days on
which the lowest-level ``|u|`` and speed exceed 40 m/s.

Nothing in the run directory is modified. Writes ``wind_audit.json`` into the
named evidence directory (an existing file is refused).

    python tools/radiation/mars_wind_audit.py RUN_DIR --name EVIDENCE_NAME

The per-day and summary arithmetic (``day_record``, ``summarize``) is plain
NumPy so it can be tested without a GPU; only ``main`` imports CuPy.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib

import numpy as np

EVIDENCE = pathlib.Path(__file__).resolve().parents[2] / "docs" / "radiation" / "evidence"
THRESHOLD = 40.0                     # m/s, the M7 bound
SIGMA_FLOOR = 0.01                   # zonal-mean max restricted to sigma > this
SUMMARIZED = ("max_abs_u", "lowest_level_abs_u_max", "lowest_level_speed_max",
              "lowest_level_abs_v_max", "zonal_mean_u_max", "zonal_mean_u_max_sigma_above_0.01")
COUNTED = ("lowest_level_abs_u_max", "lowest_level_speed_max")


def day_record(u, v, speed, sigma, latitude) -> dict:
    """One day's winds. ``u``, ``v``: (K, nlat, nlon) on the state grid, level
    K-1 lowest; ``speed``: the lowest-level horizontal speed (any shape);
    ``sigma``: (K,) full levels; ``latitude``: (nlat,) degrees."""
    u, v = np.asarray(u, dtype=np.float64), np.asarray(v, dtype=np.float64)
    sigma, latitude = np.asarray(sigma, dtype=np.float64), np.asarray(latitude, dtype=np.float64)
    au = np.abs(u)
    kg, jg, _ = np.unravel_index(int(np.argmax(au)), au.shape)
    zm = u.mean(axis=2)                                               # (K, nlat)
    kz, jz = np.unravel_index(int(np.argmax(zm)), zm.shape)
    above = sigma > SIGMA_FLOOR
    if not above.any():
        raise ValueError(f"no level has sigma > {SIGMA_FLOOR}")
    zb = np.where(above[:, None], zm, -np.inf)
    kb, jb = np.unravel_index(int(np.argmax(zb)), zb.shape)
    return {
        "max_abs_u": float(au.max()), "max_abs_u_sigma": float(sigma[kg]),
        "max_abs_u_latitude_deg": float(latitude[jg]),
        "lowest_level_abs_u_max": float(au[-1].max()),
        "lowest_level_speed_max": float(np.max(speed)),
        "lowest_level_abs_v_max": float(np.abs(v[-1]).max()),
        "zonal_mean_u_max": float(zm.max()), "zonal_mean_u_max_sigma": float(sigma[kz]),
        "zonal_mean_u_max_latitude_deg": float(latitude[jz]),
        "zonal_mean_u_min": float(zm.min()),
        "zonal_mean_u_max_sigma_above_0.01": float(zb.max()),
        "zonal_mean_u_max_sigma_above_0.01_sigma": float(sigma[kb]),
        "zonal_mean_u_max_sigma_above_0.01_latitude_deg": float(latitude[jb]),
    }


def summarize(days: dict, first_day: int = 11, threshold: float = THRESHOLD) -> dict:
    """Over days ``first_day``..last: the largest value and its day (the first
    such day on a tie) for each observable, and the days above ``threshold``
    for the lowest-level |u| and speed. ``days`` maps day -> ``day_record``."""
    window = sorted(d for d in map(int, days) if d >= first_day)
    if not window:
        raise ValueError(f"no saved day >= {first_day}")
    rec = {int(d): r for d, r in days.items()}
    out = {"first_day": first_day, "last_day": window[-1], "days": len(window),
           "threshold": threshold}
    for key in SUMMARIZED:
        best = max(window, key=lambda d: (rec[d][key], -d))
        out[key] = {"max": rec[best][key], "day": best}
    for key in COUNTED:
        out[key]["days_above_threshold"] = sum(rec[d][key] > threshold for d in window)
    return out


def check_against_series(day: int, record: dict, series: dict) -> None:
    """Abort unless the recomputed values are the ones the run recorded."""
    for key, column in (("max_abs_u", "max_abs_u"), ("lowest_level_speed_max", "max_wind_bottom")):
        recorded = dict(zip((int(d) for d in series["day"]), series.get(column, [])))
        if day in recorded and not math.isclose(record[key], recorded[day], rel_tol=1e-9):
            raise SystemExit(f"day {day}: {key} {record[key]} differs from the run's series "
                             f"{column} {recorded[day]}; not the run's own state")


def main() -> None:
    import cupy as cp
    from tropoi.run.radiative.config import RadiativeConfig
    from tropoi.run.radiative.model import build_model, build_physics
    from tropoi.spatial.states.primitive_equations import PrimitiveEquationsState
    from tropoi.temporal.column_coupling import ColumnPhysicsCoupling

    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=pathlib.Path)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out-file", default="wind_audit.json")
    ap.add_argument("--first-day", type=int, default=11)
    args = ap.parse_args()
    target = EVIDENCE / args.name / args.out_file
    if target.exists():
        raise SystemExit(f"{target} exists; choose another --out-file (evidence is not overwritten)")
    doc = json.loads((args.run / "config.json").read_text(encoding="utf-8"))
    cfg = RadiativeConfig.from_dict(doc["config"])
    if cfg.sha256() != doc["config_sha256"]:
        raise SystemExit(f"{args.run}: config.json does not reproduce its own hash")
    series = json.loads((args.run / "series.json").read_text(encoding="utf-8"))
    model, physics = build_model(cfg), build_physics(cfg)
    K, grid = model.nlev, model.grid
    sigma = np.asarray(model.sigma.full_levels_array(), dtype=np.float64)
    latitude = np.degrees(np.asarray(cp.asnumpy(grid.point_latitudes)).reshape(grid.nlat, grid.nlon)[:, 0])
    days = {}
    for path in sorted((args.run / "states").glob("state-d*.npy")):
        day = int(path.stem.split("-d")[1])
        x = cp.asarray(np.load(path))
        ts = cp.asarray(np.load(args.run / "states" / f"surface-d{day:05d}.npy")).ravel()
        if not (bool(cp.all(cp.isfinite(x))) and bool(cp.all(cp.isfinite(ts)))):
            raise SystemExit(f"non-finite saved state on day {day}")
        u, v = model.wind_on_state_grid(PrimitiveEquationsState(x))
        coupling = ColumnPhysicsCoupling(model, physics, cfg.dt, surface_temperature=ts)
        record = day_record(cp.asnumpy(u).reshape(K, grid.nlat, grid.nlon),
                            cp.asnumpy(v).reshape(K, grid.nlat, grid.nlon),
                            cp.asnumpy(coupling._lowest_wind_speed(x)), sigma, latitude)
        check_against_series(day, record, series)
        days[day] = record
    if not days:
        raise SystemExit(f"{args.run}: no saved states")
    out = {"run": args.run.name, "config_sha256": cfg.sha256(),
           "sigma_full": [float(s) for s in sigma],
           "reference_pressure": float(cfg.reference_pressure),
           "checked_against_series": sorted(c for c in ("max_abs_u", "max_wind_bottom") if c in series),
           "summary": summarize(days, args.first_day),
           "days": {str(d): r for d, r in sorted(days.items())}}
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(out["summary"], indent=2))


if __name__ == "__main__":
    main()
