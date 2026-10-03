"""Mass budget of a Held–Suarez run: where does global-mean p_s go?

Two diagnostics (GPU, CuPy), both on a finished run directory:

``states``  For saved states (``<run>/states/state-dNNNNN.npy``):

  * the semi-discrete mass tendency  d<p_s>/dt / <p_s> = <e^q q_t> / <e^q>,
    q_t the model's spectral ln p_s tendency on the state grid (ln p_s is
    neither forced nor damped, so this is the whole spatial mass budget);
  * the predicted leak of the lagged damping placement. The SI solve
    advances ln p_s with the undamped D*^{n+1}; the dampers then divide D by
    (1 + 2 dt r). The ln p_s update thus carries an extra
    -dt sum_j nu_j (D* - D)_j per step, and each leapfrog chain advances by
    2 dt, so the rate per day is (steps_per_day / 2) <e^q dq> / <e^q>,
    split into Rayleigh drag and del^8.

``restart``  Restarts the stepper from the final checkpoint's X^n (RK4
  startup, then leapfrog) with a chosen damping scheme and dt, runs a few
  days, and records <p_s> after every step plus, for the lagged scheme, the
  accumulated predicted leak along the same trajectory.

Both use per-level transforms by default (``--batched`` to follow the run's
configuration; the MX110 must not run the T42 batched GEMMs). Run from the
repository root, e.g.:

    python tools/held_suarez/mass_budget.py states  runs/hs-T42L20-prod-001 --days 300 1200
    python tools/held_suarez/mass_budget.py restart runs/hs-T42L20-prod-001 \
        --scheme centred --dt 720 --days 2 --out runs/hs-T42L20-prod-001/assets/mass_budget
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import time

import cupy as cp
import numpy as np

from tropoi.run.held_suarez.checkpoint import list_checkpoints, read_checkpoint
from tropoi.run.held_suarez.config import HeldSuarezConfig
from tropoi.run.held_suarez.model import build_model, build_physics, build_stepper
from tropoi.spatial.states.primitive_equations import PrimitiveEquationsState
from tropoi.temporal.semi_implicit import SemiImplicitOperator


class Budget:
    """Model, Gaussian area means and the lagged-leak operator of one config."""

    def __init__(self, cfg: HeldSuarezConfig):
        self.cfg = cfg
        self.model = build_model(cfg)
        self.grid = self.model.grid
        self.w = np.asarray(self.grid._gl_weights, dtype=np.float64) / 2.0
        self.K = self.model.nlev
        op = SemiImplicitOperator.from_model(self.model, t_ref=cfg.t_ref)
        dev = op._dev(cp)
        self.nu, self.mask2 = dev["nu"], dev["mask2"]
        _, (drag, hyper) = build_physics(cfg, self.model)
        h = 2.0 * cfg.dt
        fd = cp.asarray(1.0 + h * drag.rates)
        fh = cp.asarray(1.0 + h * hyper.rates)
        self.excess = {"both": fd * fh - 1.0, "drag": fd - 1.0, "del8": fh - 1.0}

    def area(self, f):
        return float((f.mean(axis=1) * self.w).sum())

    def grid_of(self, c):
        g = self.grid
        return cp.asnumpy(self.model.sh.inv_transform(c).real).reshape(g.nlat, g.nlon)

    def ps(self, x):
        return np.exp(self.grid_of(PrimitiveEquationsState(x).ln_ps))

    def mean_ps(self, x) -> float:
        return self.area(self.ps(x))

    def semi_discrete_rate(self, x) -> float:
        """d<p_s>/dt / <p_s> per day from the spatial discretisation alone."""
        qt = self.grid_of(PrimitiveEquationsState(self.model.tendency(x)).ln_ps)
        ps = self.ps(x)
        return self.area(ps * qt) / self.area(ps) * 86400.0

    def lagged_step_leak(self, x, part: str = "both") -> float:
        """<e^q dq> / <e^q> of one lagged leapfrog step, D the damped D^{n+1}."""
        K = self.K
        d_excess = self.excess[part][K:2 * K, :, None] * x[K:2 * K]      # D* - D
        dq = -self.cfg.dt * self.mask2 * cp.einsum("j,jlm->lm", self.nu, d_excess)
        ps = self.ps(x)
        return self.area(ps * self.grid_of(dq)) / self.area(ps)


def load_config(run: pathlib.Path, batched: bool, **changes) -> HeldSuarezConfig:
    cfg = HeldSuarezConfig.from_dict(json.loads((run / "config.json").read_text())["config"])
    return cfg.with_(batched_transforms=bool(batched) and cfg.batched_transforms, **changes)


def cmd_states(args) -> dict:
    run = pathlib.Path(args.run)
    b = Budget(load_config(run, args.batched))
    half_day_steps = b.cfg.steps_per_day / 2.0
    rows = []
    for d in args.days:
        path = run / "states" / f"state-d{d:05d}.npy"
        x = cp.asarray(np.load(path))
        row = {"day": d, "state_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
               "semi_discrete_per_day": b.semi_discrete_rate(x)}
        for part in ("both", "drag", "del8"):
            row[f"lagged_predicted_per_day_{part}"] = half_day_steps * b.lagged_step_leak(x, part)
        rows.append(row)
        print(json.dumps(row), flush=True)
    return {"diagnostic": "states", "run": str(run), "config_sha256": b.cfg.sha256(),
            "rows": rows}


def cmd_restart(args) -> dict:
    run = pathlib.Path(args.run)
    steps_found = list_checkpoints(run / "checkpoints")
    ck_step, ck_path = steps_found[-1]
    arrays, meta = read_checkpoint(ck_path)
    x0_host = arrays["x_curr"]
    x0_sha = hashlib.sha256(np.ascontiguousarray(x0_host).tobytes()).hexdigest()
    cfg = load_config(run, args.batched, damping_scheme=args.scheme, dt=float(args.dt))
    b = Budget(cfg)
    stepper = build_stepper(cfg, b.model)
    x0 = cp.asarray(x0_host)
    stepper.initialize(x0)
    n_steps = int(round(args.days * cfg.steps_per_day))
    m0 = b.mean_ps(x0)
    series = [{"step": 0, "day": 0.0, "rel_ps": 0.0, "predicted_rel_ps": 0.0}]
    predicted = 0.0
    t0 = time.time()
    for n in range(1, n_steps + 1):
        stepper.step()
        x = stepper.state
        if n > 1 and args.scheme == "lagged":          # step 1 is the RK4 startup
            predicted += 0.5 * b.lagged_step_leak(x)
        rel = b.mean_ps(x) / m0 - 1.0
        series.append({"step": n, "day": n / cfg.steps_per_day, "rel_ps": rel,
                       "predicted_rel_ps": predicted})
        if n % cfg.steps_per_day == 0 or n == n_steps:
            print(f"{args.scheme} dt={cfg.dt:g}: day {n / cfg.steps_per_day:.2f} "
                  f"rel <p_s> {rel: .4e}  predicted lag leak {predicted: .4e}  "
                  f"({time.time() - t0:.0f} s)", flush=True)
    x_end = cp.asnumpy(stepper.state)
    if not np.all(np.isfinite(x_end)):
        raise SystemExit("non-finite state")
    days = np.array([s["day"] for s in series])
    rel = np.array([s["rel_ps"] for s in series])
    after = days >= 0.5                                  # past the startup transient
    slope = float(np.polyfit(days[after], rel[after], 1)[0]) if after.sum() >= 2 else None
    return {"diagnostic": "restart", "run": str(run), "checkpoint": ck_path.name,
            "checkpoint_step": ck_step, "x0_sha256": x0_sha,
            "scheme": args.scheme, "dt": cfg.dt, "days": args.days,
            "config_sha256": cfg.sha256(), "startup_substeps": stepper.startup_substeps,
            "batched_transforms": cfg.batched_transforms,
            "rel_ps_end": float(rel[-1]), "fit_per_day_from_day_0p5": slope,
            "predicted_rel_ps_end": predicted, "series": series}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("states")
    p.add_argument("run")
    p.add_argument("--days", type=int, nargs="+", default=[300, 600, 900, 1200])
    p.add_argument("--batched", action="store_true")
    p.add_argument("--out")
    p = sub.add_parser("restart")
    p.add_argument("run")
    p.add_argument("--scheme", choices=("lagged", "centred"), required=True)
    p.add_argument("--dt", type=float, required=True)
    p.add_argument("--days", type=float, default=2.0)
    p.add_argument("--batched", action="store_true")
    p.add_argument("--out")
    args = ap.parse_args(argv)
    result = cmd_states(args) if args.cmd == "states" else cmd_restart(args)
    if args.out:
        out = pathlib.Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        name = (f"states.json" if args.cmd == "states"
                else f"restart-{args.scheme}-dt{args.dt:g}.json")
        path = out / name
        path.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n",
                        encoding="utf-8", newline="\n")
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
