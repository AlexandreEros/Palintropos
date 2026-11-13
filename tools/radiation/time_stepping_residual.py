"""Split the adiabatic time-stepping energy residual (GPU).

The experiment of ``energy_attribution.py`` (Earth constants, stretched
12-level grid, T9, 5 m/s rotational seed, no physics, del^8 off), run for
three days with the total energy recorded at every step, once per variant:

* ``base``: dt = 600 s, RAW filter nu = 0.1, alpha = 0.53, centred damping;
* ``raw_off``: the filter switched off;
* ``alpha_0.5``: the filter with alpha = 0.5 (Williams' neutral choice);
* ``dt_300``: half the step;
* ``lagged_damping``: the other damping placement (no damping is active
  here, so it must change nothing).

It separates a loss that persists from day to day from one that happens
once, in the first hours, while the unbalanced seed adjusts. Writes
``docs/radiation/evidence/time_stepping_residual.json``.

    python tools/radiation/time_stepping_residual.py
"""
from __future__ import annotations

import json
import math
import time

import numpy as np

from adiabatic_energy import OUT, planet_config

DAYS = 3
VARIANTS = {"base": {}, "raw_off": {"raw_nu": 0.0}, "alpha_0.5": {"raw_alpha": 0.5},
            "dt_300": {"dt": 300.0}, "lagged_damping": {"damping_scheme": "lagged"}}


def run(name: str = "earth", **changes) -> dict:
    import cupy as cp
    from tropoi.physics.planets import PLANETS
    from tropoi.run.radiative.model import build_model, build_stepper, initial_state
    from tropoi.temporal.column_coupling import global_energy
    body = PLANETS[name]
    cfg = planet_config(name, perturbation_wind=5.0, perturbation_lmax=4, days=DAYS,
                        hyperdiffusion_efold_days=1.0e9, **changes)
    model = build_model(cfg)
    x0, _, _ = initial_state(cfg)
    K = model.nlev
    profile = body.reference_temperature * model.sigma.full_levels_array() ** (0.5 * cfg.kappa)
    x0[2 * K:3 * K, 0, 0] = cp.asarray(np.maximum(profile, 0.4 * body.reference_temperature)) \
        * math.sqrt(4.0 * math.pi)
    stepper = build_stepper(cfg, model, None)
    stepper.initialize(x0, 0.0)
    total = [global_energy(model, x0, cfg.gravity)["total"]]
    ke0 = global_energy(model, x0, cfg.gravity)["kinetic"]
    t0 = time.perf_counter()
    for _ in range(cfg.steps_per_day * DAYS):
        stepper.step()
        total.append(global_energy(model, stepper.state, cfg.gravity)["total"])
    wall = time.perf_counter() - t0
    e = (np.array(total) - total[0]) / ke0
    spd = cfg.steps_per_day
    hour = int(round(3600.0 / cfg.dt))
    return {"dt": cfg.dt, "raw_nu": cfg.raw_nu, "raw_alpha": cfg.raw_alpha,
            "damping_scheme": cfg.damping_scheme, "initial_kinetic_j_m2": ke0,
            "daily_total_change_over_initial_kinetic":
                [float(e[(d + 1) * spd] - e[d * spd]) for d in range(DAYS)],
            "startup_step_over_initial_kinetic": float(e[1]),
            "after_hours_over_initial_kinetic": {str(h): float(e[h * hour])
                                                 for h in (1, 2, 3, 4, 6, 12, 24)},
            "seconds_per_step_with_energy_every_step": wall / (spd * DAYS)}


def main() -> None:
    doc = {"experiment": "earth T9 L12 stretched, 5 m/s seed, adiabatic, del^8 off", "days": DAYS,
           "variants": {}}
    for label, changes in VARIANTS.items():
        doc["variants"][label] = row = run(**changes)
        print(label, json.dumps(row), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "time_stepping_residual.json").write_text(json.dumps(doc, indent=2, sort_keys=True)
                                                     + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
