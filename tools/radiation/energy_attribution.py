"""Where the dry core's energy goes in one adiabatic day (GPU).

The same experiment as ``adiabatic_energy.py`` (planet constants, stretched
12-level grid, T9, 5 m/s rotational seed, no physics), with the change of
total energy over the day split by operator. For an operator with tendency
``D(X)``, its instantaneous energy rate is the centred difference of
``global_energy`` along ``D``; integrating that over the run gives what the
operator contributed. Operators:

* ``adiabatic``: the spatial discretisation of the primitive equations
  (``model.tendency``). Zero if the discretisation conserves energy; its
  kinetic part alone is the conversion between kinetic energy and enthalpy.
* ``diffusion_momentum`` / ``diffusion_temperature``: the del^8 rows.
* ``heating``: the frictional-heating term, when configured.

``time_stepping`` is what is left: actual change minus the sum of the
integrated rates. It contains the semi-implicit leapfrog and the
Robert-Asselin-Williams filter, and is checked by halving the step and by
switching the filter off. Writes
``docs/radiation/evidence/energy_attribution.json``.

    python tools/radiation/energy_attribution.py
"""
from __future__ import annotations

import json
import math
import pathlib

import numpy as np

from adiabatic_energy import OUT, planet_config

NO_DIFFUSION = 1.0e9          # e-folding days: switches del^8 off


def attribute(name: str, *, sample_every: int = 4, eps: float = 20.0, **changes) -> dict:
    import cupy as cp
    from tropoi.physics.planets import PLANETS
    from tropoi.run.radiative.model import build_model, build_stepper, initial_state
    from tropoi.temporal.column_coupling import global_energy
    body = PLANETS[name]
    cfg = planet_config(name, perturbation_wind=5.0, perturbation_lmax=4, **changes)
    model = build_model(cfg)
    x0, _, _ = initial_state(cfg)
    K = model.nlev
    profile = body.reference_temperature * model.sigma.full_levels_array() ** (0.5 * cfg.kappa)
    x0[2 * K:3 * K, 0, 0] = cp.asarray(np.maximum(profile, 0.4 * body.reference_temperature)) \
        * math.sqrt(4.0 * math.pi)
    stepper = build_stepper(cfg, model, None)
    stepper.initialize(x0, 0.0)
    diffusion = stepper.dampers[-1]
    diffusing = cfg.hyperdiffusion_efold_days < NO_DIFFUSION

    def rate(x, tendency):
        up = global_energy(model, x + eps * tendency, cfg.gravity)
        down = global_energy(model, x - eps * tendency, cfg.gravity)
        return {k: (up[k] - down[k]) / (2.0 * eps) for k in ("total", "kinetic")}

    def rates(x):
        out = {"adiabatic": rate(x, model.tendency(x))}
        if diffusing:
            d = diffusion.tendency(x)
            momentum, temperature = d.copy(), d.copy()
            momentum[2 * K:] = 0.0
            temperature[:2 * K] = 0.0
            out["diffusion_momentum"] = rate(x, momentum)
            out["diffusion_temperature"] = rate(x, temperature)
        for term in stepper.explicit_terms:
            out["heating"] = rate(x, term(x))
        return out

    e0 = global_energy(model, x0, cfg.gravity)
    steps = cfg.steps_per_day
    integral: dict = {}
    for n in range(steps + 1):
        if n % sample_every == 0:
            weight = (0.5 if n in (0, steps) else 1.0) * sample_every * cfg.dt
            for op, r in rates(stepper.state).items():
                acc = integral.setdefault(op, {"total": 0.0, "kinetic": 0.0})
                for k in acc:
                    acc[k] += weight * r[k]
        if n < steps:
            stepper.step()
    e1 = global_energy(model, stepper.state, cfg.gravity)
    ke0 = e0["kinetic"]
    actual = e1["total"] - e0["total"]
    by_operator = {op: v["total"] / ke0 for op, v in integral.items()}
    return {"dt": cfg.dt, "raw_nu": cfg.raw_nu, "diffusion": diffusing,
            "frictional_heating": cfg.frictional_heating,
            "initial_kinetic_j_m2": ke0,
            "total_change_over_initial_kinetic": actual / ke0,
            "kinetic_change_over_initial_kinetic": (e1["kinetic"] - e0["kinetic"]) / ke0,
            "integrated_rate_over_initial_kinetic": by_operator,
            "adiabatic_conversion_of_kinetic_over_initial_kinetic":
                integral["adiabatic"]["kinetic"] / ke0,
            "time_stepping_over_initial_kinetic": actual / ke0 - sum(by_operator.values())}


def main() -> None:
    off = {"hyperdiffusion_efold_days": NO_DIFFUSION}
    doc = {}
    for name in ("earth", "mars", "jupiter"):
        doc[name] = {
            "no_diffusion_dt600": attribute(name, **off),
            "no_diffusion_dt300": attribute(name, dt=300.0, sample_every=8, **off),
            "no_diffusion_dt150": attribute(name, dt=150.0, sample_every=16, **off),
            "no_diffusion_dt600_no_filter": attribute(name, raw_nu=0.0, **off),
            "diffusion": attribute(name),
            "diffusion_with_heating": attribute(name, frictional_heating=True),
        }
        for case, row in doc[name].items():
            print(name, case, json.dumps(row), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "energy_attribution.json").write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n",
                                                 encoding="utf-8")


if __name__ == "__main__":
    main()
