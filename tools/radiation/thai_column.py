"""Column radiation of a radiative preset on prescribed states (CPU).

    python tools/radiation/thai_column.py --preset dev-thai-ben2
        [--reference-table '{"file": ..., "sha256": ...}']
        [--out-dir docs/radiation/evidence/thai-dev-columns]

Builds the radiation of the preset exactly as a run does
(``build_radiation(cfg)``: the configured tables, hashes and bounds), on the
preset's own vertical grid, and solves it for prescribed columns:

* ``isothermal-300``: the protocol's initial state, 300 K air over a 300 K
  surface (what the first radiation call of a run sees);
* ``adiabat-250-160``: a 250 K surface under a dry adiabat of the
  configured kappa, capped by a 160 K isothermal layer (a troposphere and a
  stratosphere, so the heating changes sign).

For each it saves the interface fluxes and the layer heating, and checks
that the pressure-weighted heating equals the convergence of the net flux:

    sum_k c_p dp_k / g  dT/dt_k  =  N_top - N_bottom

With ``--reference-table`` the same columns are solved again with another
table of the same gas (the documented high-resolution reference of the
spectral file) and the differences are recorded.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

from tropoi.physics.radiation.scheme import Column, LowerBoundary, heating_rate
from tropoi.run.radiative.config import DAY_SECONDS, PRESETS, RadiativeConfig
from tropoi.run.radiative.model import build_radiation, build_sigma

#: Stellar inputs of the column (global-mean-like): flux through a
#: horizontal surface at the top and cosine of the zenith angle.
TOA_FLUX, MU0 = 225.0, 0.5


def states(cfg: RadiativeConfig) -> dict:
    sig = build_sigma(cfg).interfaces_array()
    p = (sig * cfg.reference_pressure)[:, None]
    mid = 0.5 * (p[1:] + p[:-1])
    adiabat = np.maximum(250.0 * (mid / cfg.reference_pressure) ** cfg.kappa, 160.0)
    return {"isothermal-300": (p, np.full_like(mid, 300.0), 300.0),
            "adiabat-250-160": (p, adiabat, 250.0)}


def solve(cfg: RadiativeConfig, radiation, p, t, ts) -> dict:
    albedo = float((cfg.surface or {}).get("albedo", 0.0))
    column = Column(p, t, cfg.gravity, cfg.cp, latitude=np.zeros(1))
    boundary = LowerBoundary("surface", temperature=np.full(1, ts), albedo=albedo,
                             emissivity=float((cfg.surface or {}).get("emissivity", 1.0)))
    fl = radiation.fluxes(column, boundary, toa_flux=np.full(1, TOA_FLUX),
                          mu0=np.full(1, MU0))
    dp = (p[1:] - p[:-1])[:, 0]
    net = fl.net_down[:, 0]
    heating = fl.heating[:, 0]
    column_heating = float((cfg.cp * dp / cfg.gravity * heating).sum())
    convergence = float(net[0] - net[-1])
    lw_net = (fl.lw_down - fl.lw_up)
    lw_heating = heating_rate(lw_net, p, cfg.gravity, cfg.cp)[:, 0]
    return {
        "pressure_interfaces_pa": p[:, 0].tolist(),
        "layer_pressure_pa": column.layer_pressure[:, 0].tolist(),
        "temperature_k": t[:, 0].tolist(), "surface_temperature_k": ts,
        "lw_up": fl.lw_up[:, 0].tolist(), "lw_down": fl.lw_down[:, 0].tolist(),
        "sw_up": fl.sw_up[:, 0].tolist(), "sw_down": fl.sw_down[:, 0].tolist(),
        "net_down": net.tolist(),
        "heating_k_per_day": (heating * DAY_SECONDS).tolist(),
        "lw_heating_k_per_day": (lw_heating * DAY_SECONDS).tolist(),
        "olr": float(fl.olr[0]), "lw_down_surface": float(fl.lw_down[-1, 0]),
        "lw_net_up_surface": float(fl.lw_up[-1, 0] - fl.lw_down[-1, 0]),
        "absorbed_stellar_toa": float(fl.absorbed_stellar[0]),
        "sw_absorbed_surface": float(fl.sw_down[-1, 0] - fl.sw_up[-1, 0]),
        "closure": {"column_heating_w_m2": column_heating,
                    "net_flux_convergence_w_m2": convergence,
                    "residual_w_m2": column_heating - convergence,
                    "relative": abs(column_heating - convergence)
                    / max(abs(convergence), 1e-300)},
    }


def compare(a: dict, b: dict) -> dict:
    """Differences ``a - b`` (b is the reference)."""
    p = np.asarray(a["layer_pressure_pa"])
    ha, hb = np.asarray(a["lw_heating_k_per_day"]), np.asarray(b["lw_heating_k_per_day"])
    below = p >= 500.0                                   # below 5 mbar
    out = {k: a[k] - b[k] for k in ("olr", "lw_down_surface", "lw_net_up_surface")}
    out.update({f"{k}_relative": (a[k] - b[k]) / b[k] for k in ("olr", "lw_down_surface")})
    up = np.asarray(a["lw_up"]) - np.asarray(b["lw_up"])
    down = np.asarray(a["lw_down"]) - np.asarray(b["lw_down"])
    out["max_abs_interface_flux_difference"] = float(max(np.abs(up).max(),
                                                          np.abs(down).max()))
    out["lw_heating_max_abs_difference_below_5mbar_k_per_day"] = float(
        np.abs(ha - hb)[below].max())
    out["lw_heating_max_abs_reference_below_5mbar_k_per_day"] = float(np.abs(hb)[below].max())
    out["lw_heating_max_abs_difference_above_5mbar_k_per_day"] = float(
        np.abs(ha - hb)[~below].max())
    out["lw_heating_max_abs_reference_above_5mbar_k_per_day"] = float(
        np.abs(hb)[~below].max())
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--preset", choices=sorted(PRESETS), required=True)
    ap.add_argument("--reference-table", help="JSON {'file': ..., 'sha256': ...}")
    ap.add_argument("--out-dir", default="docs/radiation/evidence/thai-dev-columns")
    args = ap.parse_args(argv)
    cfg = PRESETS[args.preset]()
    radiation = build_radiation(cfg)
    reference = None
    if args.reference_table:
        spec = json.loads(args.reference_table)
        lw = {**cfg.longwave, "tables": [spec]}
        reference = build_radiation(cfg.with_(longwave=lw))
    lw = radiation.longwave
    doc = {"schema": "palintropos.radiation.column/1", "preset": args.preset,
           "experiment_id": cfg.experiment_id, "config_sha256": cfg.sha256(),
           "longwave": cfg.longwave, "shortwave": cfg.shortwave,
           "gravity": cfg.gravity, "cp": cfg.cp, "gas_constant": cfg.gas_constant,
           "thermal_source": cfg.thermal_source,
           "stellar": {"toa_flux_w_m2": TOA_FLUX, "mu0": MU0},
           "longwave_points": int(lw.n_points),
           "planck_coverage": {str(t): lw.spectral_coverage(t) for t in (160.0, 250.0, 300.0)},
           "units": {"fluxes": "W m^-2 at layer interfaces, top first",
                     "heating": "K per 86400 s, layers top first"},
           "columns": {}}
    if reference is not None:
        doc["reference"] = {"table": json.loads(args.reference_table),
                            "longwave_points": int(reference.longwave.n_points)}
    for name, (p, t, ts) in states(cfg).items():
        res = solve(cfg, radiation, p, t, ts)
        if reference is not None:
            ref = solve(cfg, reference, p, t, ts)
            res["reference"] = {k: ref[k] for k in ("lw_up", "lw_down", "olr",
                                                    "lw_down_surface",
                                                    "lw_heating_k_per_day")}
            res["minus_reference"] = compare(res, ref)
        doc["columns"][name] = res
        c = res["closure"]
        print(f"{args.preset} {name}: OLR {res['olr']:.3f}  LW down sfc "
              f"{res['lw_down_surface']:.3f}  closure residual {c['residual_w_m2']:.3e} "
              f"W m^-2 (relative {c['relative']:.1e})")
        if reference is not None:
            m = res["minus_reference"]
            print(f"    minus reference: OLR {m['olr']:+.3f} ({100 * m['olr_relative']:+.2f} %)"
                  f"  LW down sfc {m['lw_down_surface']:+.3f} "
                  f"({100 * m['lw_down_surface_relative']:+.2f} %)  heating below 5 mbar "
                  f"{m['lw_heating_max_abs_difference_below_5mbar_k_per_day']:.3f} of "
                  f"{m['lw_heating_max_abs_reference_below_5mbar_k_per_day']:.3f} K/day")
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{args.preset}.json"
    out.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
