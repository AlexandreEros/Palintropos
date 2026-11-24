"""Build the self-contained interactive Mercury view from an airless run.

    python tools/planet_one/build_view.py RUN_DIR OUT.html [--frames 180]
        [--gp gp.json] [--gi gi.json] [--l3 l3.json]

The page shows one full solar day of the run's retained cycle: the surface
temperature map and a globe with the subsolar point and terminator, the
orbit and spin phase, temperature series at the four equatorial
longitudes, the depth profile of a chosen column, an equatorial
depth-longitude section and the diagnostics. Numbers are quantized to
1/80 K (uint16), gzip-compressed and base64-embedded. The page states the
quantization. Full-precision results stay in the run directory.
"""
from __future__ import annotations

import argparse
import base64
import gzip
import json
import math
import pathlib
import sys

import numpy as np

from tropoi.physics.insolation import insolation_from_spec

Q = 80.0          # quantization: 1/80 K per count


def _pack(a: np.ndarray) -> str:
    q = np.clip(np.round(np.asarray(a, dtype=float) * Q), 0, 65535).astype("<u2")
    return base64.b64encode(gzip.compress(q.tobytes(), 9)).decode()


def build(run: pathlib.Path, frames: int, extras: dict) -> str:
    meta = json.loads((run / "config.json").read_text())
    cfg = meta["config"]
    if cfg["columns"]["kind"] != "latlon":
        raise SystemExit("the view needs a lat/lon run")
    nlat, nlon = int(cfg["columns"]["nlat"]), int(cfg["columns"]["nlon"])
    grid = np.load(run / "grid.npz")
    status = json.loads((run / "retained" / "status.json").read_text())
    cycles = json.loads((run / "cycles.json").read_text())
    surf = np.load(run / "retained" / "surface.npy")       # (S, 5, ncol)
    prof = np.load(run / "retained" / "profiles.npy")      # (Pn, L, ncol)
    ints = np.load(run / "retained" / "integrals.npz")
    S, Pn, L = surf.shape[0], prof.shape[0], prof.shape[1]
    stride = max(1, S // frames)
    sel = np.arange(stride - 1, S, stride)                  # sample indices shown
    P = cfg["cycle_s"]
    cyc = status["cycle"]
    t_sel = cyc * P + (sel + 1) * P / S
    ts = surf[:, 0, :].reshape(S, nlat, nlon)
    lat = grid["lat_deg"].reshape(nlat, nlon)[:, 0]
    lon = grid["lon_deg"].reshape(nlat, nlon)[0]
    ieq = int(np.argmin(np.abs(lat)))
    # four equatorial longitudes, every sample
    series = {}
    for target in (0, 90, 180, 270):
        j = int(np.argmin(np.abs(((lon - target + 180) % 360) - 180)))
        series[f"{lon[j]:.0f}"] = np.round(ts[:, ieq, j], 2).tolist()
    # geometry per shown frame
    ins = insolation_from_spec(cfg["illumination"])
    ill = ins.illumination
    a = ill.orbit.semimajor_axis_m
    geo = []
    for t in t_sel:
        la, lo = ill.subsolar_point(float(t))
        pos = ill.planet_position(float(t))
        w = ill.rotation.prime_meridian_angle(float(t))
        geo.append([round(math.degrees(la), 4), round(math.degrees(lo) % 360, 4),
                    round(pos[0] / a, 6), round(pos[1] / a, 6),
                    round(math.degrees(ill.true_anomaly(float(t))) % 360, 3),
                    round(ill.flux_at(float(t)), 1), round(math.degrees(w) % 360, 3)])
    # equatorial section and a subset of columns for depth profiles
    sec = prof.reshape(Pn, L, nlat, nlon)[:, :, ieq, :]           # (Pn, L, nlon)
    sub_lat = [i for i in range(nlat) if abs(lat[i]) <= 89 and i % max(1, nlat // 12) == 0]
    sub_lon = list(range(0, nlon, max(1, nlon // 24)))
    cube = prof.reshape(Pn, L, nlat, nlon)
    subset = cube[:, :, sub_lat, :][:, :, :, sub_lon]             # (Pn, L, a, b)
    env_min, env_max = cube.min(axis=0), cube.max(axis=0)         # (L, nlat, nlon)
    w = grid["weights"] / grid["weights"].sum()
    last = cycles[-1]
    data = {
        "nlat": nlat, "nlon": nlon, "lat": lat.round(4).tolist(), "lon": lon.round(4).tolist(),
        "frames": len(sel), "samples": S, "profile_frames": Pn, "L": L,
        "centres": grid["centres"].round(6).tolist(), "interfaces": grid["interfaces"].round(6).tolist(),
        "period_days": P / 86400.0, "cycle": cyc, "orbit_days": P / 2 / 86400.0,
        "t_days": ((t_sel - cyc * P) / 86400.0).round(4).tolist(),
        "geo": geo, "e": ill.eccentricity, "series": series, "eq_lat": float(lat[ieq]),
        "surface": _pack(ts[sel]), "section": _pack(sec),
        "subset": _pack(subset), "sub_lat": [float(lat[i]) for i in sub_lat],
        "sub_lon": [float(lon[j]) for j in sub_lon],
        "env_min": _pack(env_min[:, sub_lat][:, :, sub_lon]),
        "env_max": _pack(env_max[:, sub_lat][:, :, sub_lon]),
        "status": status, "experiment": cfg["experiment_id"],
        "config_sha256": meta["config_sha256"], "code": meta["code"],
        "provenance": cfg["provenance"], "strata": cfg["strata"],
        "albedo": cfg["albedo"], "emissivity": cfg["emissivity"],
        "geothermal_flux": cfg["geothermal_flux"],
        "diag": {"Ts_min": float(ts.min()), "Ts_max": float(ts.max()),
                 "mean_absorbed": float(w @ (ints["absorbed"] / P)),
                 "mean_emitted": float(w @ (ints["emitted"] / P)),
                 "max_energy_imbalance": last["max_energy_imbalance_W_m2"],
                 "max_surface_change": last.get("max_surface_change_K"),
                 "max_subsurface_change": last.get("max_subsurface_change_K"),
                 "bottom_drift": last.get("bottom_drift_K"),
                 "residual_ratio": last["max_cell_residual_ratio"],
                 "newton_max": last["max_newton_iterations"],
                 "extent": last["extent_K"], "dt_s": last["dt_s"],
                 "cycles_run": len(cycles)},
        "extras": extras, "quantum_K": 1.0 / Q,
    }
    page = (pathlib.Path(__file__).with_name("view_template.html")).read_text()
    return page.replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("out")
    ap.add_argument("--frames", type=int, default=180)
    ap.add_argument("--gp")
    ap.add_argument("--gi")
    ap.add_argument("--l3")
    args = ap.parse_args(argv)
    extras = {}
    for key in ("gp", "gi", "l3"):
        path = getattr(args, key)
        if path:
            extras[key] = json.loads(pathlib.Path(path).read_text())
    html = build(pathlib.Path(args.run), args.frames, _summaries(extras))
    pathlib.Path(args.out).write_text(html)
    print(f"{args.out}: {len(html) / 1e6:.2f} MB")
    return 0


def _summaries(extras: dict) -> dict:
    """Keep only what the page shows from the evidence files."""
    out = {}
    gp = extras.get("gp")
    if gp:
        fails = [f"{part}: {col} {key}" for part in ("timestep", "grid", "depth")
                 for col, rows in gp[part].items() for key, r in rows.items() if not r["pass"]]
        out["gp"] = {"pass": gp["pass"], "all_periodic": gp["all_periodic"],
                     "production": gp["production"], "failures": fails}
    gi = extras.get("gi")
    if gi:
        out["gi"] = [{"level": lv["level"], **{k: v["max_abs_dTs_K"] for k, v in lv["columns"].items()}}
                     for lv in gi["levels"]]
    if extras.get("l3"):
        out["l3"] = extras["l3"]
    return out


if __name__ == "__main__":
    sys.exit(main())
