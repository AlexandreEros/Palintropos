"""Where the winds peak, how stiff the surface exchange is, and where the mass
goes, read from a finished run's saved daily states (GPU, read-only).

For every saved state ``states/state-dNNNNN.npy`` (and its slab
temperature ``surface-dNNNNN.npy``) of a radiative run this records:

* **winds.** The largest ``|u|`` on the state grid (the run's own
  ``max_abs_u``, which is checked against its ``series.json``), the level,
  sigma and latitude where it occurs, the per-level maxima, and the largest
  lowest-level wind speed (the speed the sensible exchange sees).
* **exchange stiffness.** With ``k = rho C |v| c_p`` the bulk exchange
  coefficient of the lowest layer (``physics.surface.sensible_heat_flux``
  at that state), ``k w / C_a`` with ``w = 2 dt`` the leapfrog window and
  ``C_a = c_p dp / g`` the bottom layer's heat capacity, and ``k dt / C_s``
  for the slab. A forward (explicit) exchange is unstable where ``k w / C_a
  > 2``; the backward-Euler exchange divides the flux by ``1 + k w / C_a +
  k dt / C_s`` and is stable for any value.
* **mass.** The area means of ``p_s`` and of ``ln p_s`` and the variance of
  ``ln p_s`` (Gauss-Legendre weights), which separate a change of the
  ``ln p_s`` monopole from the growth of surface-pressure variance.

Nothing in the run directory is modified. Writes ``state_diagnostics.json``
into the named evidence directory (an existing file is refused).

    python tools/radiation/mars_state_diagnostics.py RUN_DIR --name EVIDENCE_NAME
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib

import numpy as np

EVIDENCE = pathlib.Path(__file__).resolve().parents[2] / "docs" / "radiation" / "evidence"


def main() -> None:
    import cupy as cp
    from tropoi.physics.surface import sensible_heat_flux
    from tropoi.run.radiative.config import RadiativeConfig
    from tropoi.run.radiative.model import build_model, build_physics
    from tropoi.spatial.states.primitive_equations import PrimitiveEquationsState
    from tropoi.temporal.column_coupling import ColumnPhysicsCoupling

    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=pathlib.Path)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out-file", default="state_diagnostics.json")
    args = ap.parse_args()
    target = EVIDENCE / args.name / args.out_file
    if target.exists():
        raise SystemExit(f"{target} exists; choose another --out-file (evidence is not overwritten)")
    doc = json.loads((args.run / "config.json").read_text(encoding="utf-8"))
    cfg = RadiativeConfig.from_dict(doc["config"])
    if cfg.sha256() != doc["config_sha256"]:
        raise SystemExit(f"{args.run}: config.json does not reproduce its own hash")
    series = json.loads((args.run / "series.json").read_text(encoding="utf-8"))
    recorded = dict(zip((int(d) for d in series["day"]), series["max_abs_u"]))
    model, physics = build_model(cfg), build_physics(cfg)
    if physics.sensible is None or physics.surface is None:
        raise SystemExit("the run has no sensible exchange with a slab")
    K, grid = model.nlev, model.grid
    sigma = model.sigma.full_levels_array()
    latitude = np.degrees(np.asarray(cp.asnumpy(grid.point_latitudes)).reshape(grid.nlat, grid.nlon)[:, 0])
    weights = np.asarray(grid._gl_weights, dtype=np.float64) / 2.0          # sums to 1
    options = {k: v for k, v in physics.sensible.items() if k != "implicit"}
    window = 2.0 * cfg.dt
    days = {}
    for path in sorted((args.run / "states").glob("state-d*.npy")):
        day = int(path.stem.split("-d")[1])
        x = cp.asarray(np.load(path))
        ts = cp.asarray(np.load(args.run / "states" / f"surface-d{day:05d}.npy")).ravel()
        if not (bool(cp.all(cp.isfinite(x))) and bool(cp.all(cp.isfinite(ts)))):
            raise SystemExit(f"non-finite saved state on day {day}")
        u, _ = model.wind_on_state_grid(PrimitiveEquationsState(x))
        au = np.abs(cp.asnumpy(u).reshape(K, grid.nlat, grid.nlon))
        k, j, _ = np.unravel_index(int(np.argmax(au)), au.shape)
        if day in recorded and not math.isclose(float(au.max()), recorded[day], rel_tol=1e-9):
            raise SystemExit(f"day {day}: max|u| {au.max()} differs from the run's series "
                             f"{recorded[day]}; not the run's own state")
        coupling = ColumnPhysicsCoupling(model, physics, cfg.dt, surface_temperature=ts)
        p_int, temperature = coupling.columns(x)
        speed = coupling._lowest_wind_speed(x)
        p_top, p_bot = p_int[-2], p_int[-1]
        dp = p_bot - p_top
        t_air = temperature[-1]
        alpha = 1.0 - (p_top / dp) * cp.log(p_bot / p_top)
        height = alpha * physics.gas_constant * t_air / physics.gravity
        density = 0.5 * (p_top + p_bot) / (physics.gas_constant * t_air)
        _, coefficient = sensible_heat_flux(ts, t_air, height, speed, density, physics.cp,
                                            physics.gravity, **options)
        gust = float(options.get("gustiness", 0.0))
        exchange = density * coefficient * cp.sqrt(speed * speed + gust * gust) * physics.cp
        air = cp.asnumpy(exchange * window / (physics.cp * dp / physics.gravity))
        slab = cp.asnumpy(exchange * cfg.dt / physics.surface.heat_capacity)
        lnps = cp.asnumpy(model.sh.inv_transform(x[3 * K]).real).reshape(grid.nlat, grid.nlon)

        def area(f):
            return float((f.mean(axis=1) * weights).sum())

        mean_ln = area(lnps)
        days[str(day)] = {
            "max_abs_u": float(au.max()), "max_level": int(k), "max_sigma": float(sigma[k]),
            "max_latitude_deg": float(latitude[j]),
            "per_level_max_abs_u": [float(a) for a in au.reshape(K, -1).max(axis=1)],
            "lowest_level_speed_max": float(cp.asnumpy(speed).max()),
            "exchange_air_number_max": float(air.max()),
            "exchange_air_points_above_2": int((air > 2.0).sum()),
            "exchange_slab_number_max": float(slab.max()),
            "mean_ps": area(np.exp(lnps)), "mean_ln_ps": mean_ln,
            "variance_ln_ps": area((lnps - mean_ln) ** 2),
        }
    if not days:
        raise SystemExit(f"{args.run}: no saved states")
    worst = max(days, key=lambda d: days[d]["exchange_air_number_max"])
    out = {"run": args.run.name, "config_sha256": cfg.sha256(), "sigma_full": [float(s) for s in sigma],
           "reference_pressure": float(cfg.reference_pressure), "points": int(grid.nlat * grid.nlon),
           "exchange_window_seconds": window,
           "largest_exchange_air_number": {"day": int(worst),
                                           "value": days[worst]["exchange_air_number_max"]},
           "days": days}
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k not in ("days", "sigma_full")}, indent=2))


if __name__ == "__main__":
    main()
