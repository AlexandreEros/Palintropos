"""Adiabatic energy conservation of the dry core on each planet preset (GPU).

No physics: a resting, stably stratified column on the planet's constants
and stretched grid (12 levels, evolved T9) with a seeded 5 m/s rotational
flow, integrated for one day, once without and once with the del^8
hyperdiffusion. Writes ``docs/radiation/evidence/adiabatic_energy.json``:
the rate of change of total energy (W m^-2), the same as a fraction of the
initial kinetic energy, the kinetic energy that survives, and the drift of
mean surface pressure.

The point: what radiation exchanges must be large against what the dynamics
loses on its own, and the only energy the dry core removes is the kinetic
energy its hyperdiffusion deletes (it is not returned as heat).

    python tools/radiation/adiabatic_energy.py
"""
from __future__ import annotations

import json
import math
import pathlib

import numpy as np

OUT = pathlib.Path(__file__).resolve().parents[2] / "docs" / "radiation" / "evidence"


def planet_config(name: str, **changes):
    from tropoi.physics.planets import PLANETS
    from tropoi.run.radiative.config import RadiativeConfig
    body = PLANETS[name]
    spec = body.vertical_grid
    return RadiativeConfig(
        experiment_id=f"{name}-regime", radius=body.radius, omega=body.rotation_rate,
        gravity=body.gravity, gas_constant=body.gas_constant, cp=body.cp,
        reference_pressure=body.reference_pressure, l_max=10, retained_truncation=9,
        nlat=16, nlon=32,
        vertical={"kind": "stretched", "nlev": 12, "sigma_top": spec.sigma_top,
                  "linear_scale": spec.linear_scale},
        dt=600.0, days=1, t_ref=body.temperature_range[1],
        longwave={"kind": "power_law", "tau0": 1.0},
        insolation={"kind": "uniform", "flux": 100.0}, shortwave={"kind": "transparent"},
        surface={"heat_capacity": 1.0e7},
        initial={"kind": "isothermal", "temperature": body.reference_temperature}
    ).with_(**changes)


def one_day(name: str, hyperdiffusion: bool) -> dict:
    import cupy as cp
    from tropoi.physics.planets import PLANETS
    from tropoi.run.radiative.model import build_model, build_stepper, initial_state
    from tropoi.temporal.column_coupling import global_energy
    body = PLANETS[name]
    cfg = planet_config(name, perturbation_wind=5.0, perturbation_lmax=4,
                        hyperdiffusion_efold_days=0.1 if hyperdiffusion else 1.0e9)
    model = build_model(cfg)
    x0, _, _ = initial_state(cfg)
    K = model.nlev
    profile = body.reference_temperature * model.sigma.full_levels_array() ** (0.5 * cfg.kappa)
    x0[2 * K:3 * K, 0, 0] = cp.asarray(np.maximum(profile, 0.4 * body.reference_temperature)) \
        * math.sqrt(4.0 * math.pi)
    stepper = build_stepper(cfg, model, None)
    stepper.initialize(x0, 0.0)
    e0 = global_energy(model, x0, cfg.gravity)
    for _ in range(cfg.steps_per_day):
        stepper.step()
    e1 = global_energy(model, stepper.state, cfg.gravity)
    return {"energy_rate_w_m2": (e1["total"] - e0["total"]) / 86400.0,
            "energy_change_over_initial_kinetic": (e1["total"] - e0["total"]) / e0["kinetic"],
            "kinetic_remaining": e1["kinetic"] / e0["kinetic"],
            "initial_kinetic_j_m2": e0["kinetic"],
            "surface_pressure_drift": e1["mean_surface_pressure"] / e0["mean_surface_pressure"] - 1.0,
            "finite": bool(cp.isfinite(stepper.state).all())}


def main() -> None:
    doc = {}
    for name in ("earth", "mars", "titan", "venus", "jupiter"):
        doc[name] = {"without_hyperdiffusion": one_day(name, False),
                     "with_hyperdiffusion": one_day(name, True)}
        print(name, json.dumps(doc[name]), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "adiabatic_energy.json").write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n",
                                               encoding="utf-8")


if __name__ == "__main__":
    main()
