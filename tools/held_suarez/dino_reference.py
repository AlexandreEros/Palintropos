"""Resumable Dinosaur Held–Suarez reference run (T42 L20, float64, CPU or GPU).

Configuration is docs/held_suarez/REFERENCE_CONFIG.json (matched to the
Palintropos protocol). Runs in ``--chunk-days`` chunks; after each chunk it
writes an atomic checkpoint (state + accumulators + metadata) so the run can
be killed and resumed with the same command. Daily samples feed online
zonal-mean accumulators; nothing else is stored, so a 1200-day run is a few
MB.

Usage (from the repo root, inside a venv with dinosaur + jax):
    python tools/held_suarez/dino_reference.py --out runs/<dir> --days 1200
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import pickle
import subprocess
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve()
REPO = HERE.parents[2]
CONFIG = REPO / "docs" / "held_suarez" / "REFERENCE_CONFIG.json"

ZONAL_KEYS = ("u", "v", "T", "TsTs", "usus", "vsTs", "usvs")
SCALAR_KEYS = ("day", "mean_lnps", "mean_T", "ke_mass_weighted", "max_abs_u")


def _atomic_write(path: pathlib.Path, data: bytes) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def build(cfg: dict, dt_s: float):
    import jax
    jax.config.update("jax_enable_x64", True)
    import dinosaur
    units = dinosaur.scales.units
    layers = int(cfg["levels"])
    coords = dinosaur.coordinate_systems.CoordinateSystem(
        horizontal=dinosaur.spherical_harmonic.Grid.T42(),
        vertical=dinosaur.sigma_coordinates.SigmaCoordinates.equidistant(
            layers, dtype=np.float64))
    ps = dinosaur.primitive_equations.PrimitiveEquationsSpecs.from_si()
    p0 = cfg["p0_pa"] * units.pascal
    init_fn, aux = dinosaur.primitive_equations_states.isothermal_rest_atmosphere(
        coords=coords, physics_specs=ps, tref=300.0 * units.degK, p0=p0,
        p1=5e3 * units.pascal)
    ref_temps = aux[dinosaur.xarray_utils.REF_TEMP_KEY]
    orog = dinosaur.primitive_equations.truncated_modal_orography(
        aux[dinosaur.xarray_utils.OROGRAPHY], coords)
    prim = dinosaur.primitive_equations.PrimitiveEquations(ref_temps, orog, coords, ps)
    hs = cfg["held_suarez"]
    forcing = dinosaur.held_suarez.HeldSuarezForcing(
        coords=coords, physics_specs=ps, reference_temperature=ref_temps, p0=p0,
        sigma_b=hs["sigma_b"], kf=hs["kf_per_day"] / units.day,
        ka=hs["ka_per_day"] / units.day, ks=hs["ks_per_day"] / units.day,
        minT=hs["Tmin_K"] * units.degK, maxT=hs["Tmax_K"] * units.degK,
        dTy=hs["dTy_K"] * units.degK, dThz=hs["dThz_K"] * units.degK)
    eq = dinosaur.time_integration.compose_equations([prim, forcing])
    dt = ps.nondimensionalize(dt_s * units.second)
    step = dinosaur.time_integration.imex_rk_sil3(eq, dt)
    hd = cfg["hyperdiffusion"]
    tau = ps.nondimensionalize(hd["efold_days_at_truncation"] * units.day)
    step = dinosaur.time_integration.step_with_filters(step, [
        dinosaur.time_integration.horizontal_diffusion_step_filter(
            coords.horizontal, dt, tau=tau, order=int(hd["order"]))])
    return dinosaur, coords, ps, ref_temps, init_fn, step


def diagnostics(dinosaur, coords, ps, ref_temps, state) -> tuple[dict, dict]:
    """Zonal means (lat, level) and global scalars from one nodal snapshot."""
    units = dinosaur.scales.units
    u, v = dinosaur.spherical_harmonic.vor_div_to_uv_nodal(
        coords.horizontal, state.vorticity, state.divergence)
    u = np.asarray(ps.dimensionalize(u, units.meter / units.second))
    v = np.asarray(ps.dimensionalize(v, units.meter / units.second))
    tv = np.asarray(coords.horizontal.to_nodal(state.temperature_variation))
    T = np.asarray(ps.dimensionalize(tv + np.asarray(ref_temps)[:, None, None], units.degK))
    lnps = np.asarray(coords.horizontal.to_nodal(state.log_surface_pressure))[0]
    # shapes: (level, lon, lat); zonal mean over lon (axis 1)
    zm = {k: a.mean(axis=1) for k, a in (("u", u), ("v", v), ("T", T))}
    us, vs, Ts = u - zm["u"][:, None, :], v - zm["v"][:, None, :], T - zm["T"][:, None, :]
    zm["TsTs"] = (Ts * Ts).mean(axis=1)
    zm["usus"] = (us * us).mean(axis=1)
    zm["vsTs"] = (vs * Ts).mean(axis=1)
    zm["usvs"] = (us * vs).mean(axis=1)
    w = np.asarray(coords.horizontal.spherical_harmonics.basis.w)  # per latitude
    w = w / w.sum()
    dsig = np.diff(np.asarray(coords.vertical.boundaries))
    area = lambda f: float((f.mean(axis=0) * w).sum())  # f: (lon, lat)
    ke_col = 0.5 * ((u * u + v * v) * dsig[:, None, None]).sum(axis=0)
    scalars = {"mean_lnps": area(lnps), "mean_T": area((T * dsig[:, None, None]).sum(axis=0)),
               "ke_mass_weighted": area(ke_col), "max_abs_u": float(np.abs(u).max())}
    # store level-major -> (lat, level) for the report
    zm = {k: np.ascontiguousarray(a.T) for k, a in zm.items()}
    return zm, scalars


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--days", type=int, default=1200)
    ap.add_argument("--chunk-days", type=int, default=10)
    ap.add_argument("--dt", type=float, default=None, help="override reference_dt_s")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    cfg = json.loads(CONFIG.read_text())
    dt_s = float(args.dt or cfg["reference_dt_s"])
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ckpt = out / "checkpoint.pkl"

    import jax
    dinosaur, coords, ps, ref_temps, init_fn, step = build(cfg, dt_s)
    steps_per_day = int(round(86400.0 / dt_s))
    assert abs(steps_per_day * dt_s - 86400.0) < 1e-9, "dt must divide a day"

    if ckpt.exists():
        with open(ckpt, "rb") as f:
            saved = pickle.load(f)
        state = saved["state"]
        day = int(saved["day"])
        zonal = saved["zonal"]
        scalars = saved["scalars"]
        print(f"resumed at day {day} from {ckpt}")
    else:
        state = init_fn(jax.random.PRNGKey(args.seed))
        day = 0
        zonal = {k: [] for k in ZONAL_KEYS}
        scalars = {k: [] for k in SCALAR_KEYS}
        meta = {
            "config": cfg, "dt_s": dt_s, "seed": args.seed,
            "script_sha256": hashlib.sha256(HERE.read_bytes()).hexdigest(),
            "dinosaur_version": getattr(dinosaur, "__version__", "?"),
            "jax_version": jax.__version__, "devices": [str(d) for d in jax.devices()],
            "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        try:
            meta["palintropos_commit"] = subprocess.check_output(
                ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
        except Exception:
            meta["palintropos_commit"] = None
        _atomic_write(out / "meta.json", json.dumps(meta, indent=2).encode())

    integrate = jax.jit(dinosaur.time_integration.trajectory_from_step(
        step, outer_steps=args.chunk_days, inner_steps=steps_per_day))

    while day < args.days:
        n = min(args.chunk_days, args.days - day)
        if n != args.chunk_days:
            integrate = jax.jit(dinosaur.time_integration.trajectory_from_step(
                step, outer_steps=n, inner_steps=steps_per_day))
        t0 = time.perf_counter()
        state, traj = jax.block_until_ready(integrate(state))
        traj = jax.device_get(traj)
        for i in range(n):
            snap = jax.tree_util.tree_map(lambda a: a[i], traj)
            zm, sc = diagnostics(dinosaur, coords, ps, ref_temps, snap)
            for k in ZONAL_KEYS:
                zonal[k].append(zm[k])
            sc["day"] = day + i + 1
            for k in SCALAR_KEYS:
                scalars[k].append(sc[k])
            if not np.isfinite(sc["ke_mass_weighted"]):
                raise RuntimeError(f"non-finite state at day {sc['day']}")
        day += n
        _atomic_write(ckpt, pickle.dumps({"state": jax.device_get(state), "day": day,
                                          "zonal": zonal, "scalars": scalars}))
        tmp = out / "series.tmp.npz"
        with open(tmp, "wb") as f:
            np.savez(f, lat=np.degrees(np.arcsin(coords.horizontal.nodal_axes[1])),
                     sigma=np.asarray(coords.vertical.centers),
                     **{k: np.asarray(v) for k, v in zonal.items()},
                     **{k: np.asarray(v) for k, v in scalars.items()})
        os.replace(tmp, out / "series.npz")
        el = time.perf_counter() - t0
        s = scalars
        print(f"day {day:5d}  {el/n:6.1f} s/day  KE={s['ke_mass_weighted'][-1]:.4g} "
              f"meanT={s['mean_T'][-1]:.3f} lnps={s['mean_lnps'][-1]:.6f} "
              f"max|u|={s['max_abs_u'][-1]:.2f}", flush=True)
    print("done", day, "days")
    return 0


if __name__ == "__main__":
    sys.exit(main())
