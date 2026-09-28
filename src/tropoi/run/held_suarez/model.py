"""Build the Held–Suarez model, physics hooks, stepper and initial state.

Everything is derived from a :class:`~tropoi.run.held_suarez.config.
HeldSuarezConfig`; nothing here keeps state between calls. GPU (CuPy).
"""
from __future__ import annotations

import math
from typing import Any, Callable, Optional

import numpy as np

from tropoi.run.held_suarez.config import HeldSuarezConfig
from tropoi.temporal.tendencies.held_suarez import (DAY_SECONDS, NewtonianRelaxation,
                                                    hyperdiffusion_damping,
                                                    rayleigh_drag_damping)

__all__ = ["build_model", "build_physics", "build_stepper", "perturbation_coefficients",
           "initial_state"]


def build_model(cfg: HeldSuarezConfig):
    """PrimitiveEquationsModel on an ideal sphere (radius exactly ``cfg.radius``,
    rotation exactly ``cfg.omega``), Gauss lat-lon grid, flat surface."""
    from tropoi.spatial.environment import PlanetaryParameters
    from tropoi.spatial.planet import Planet
    from tropoi.spatial.sigma_coordinate import SigmaGrid
    from tropoi.temporal.tendencies.primitive_equations import PrimitiveEquationsModel

    params = PlanetaryParameters.ideal_sphere(cfg.radius, 2.0 * math.pi / cfg.omega)
    params.angular_velocity = float(cfg.omega)
    planet = Planet.generate(params=params, grid_type="latlon", nlat=cfg.nlat,
                             nlon=cfg.nlon, l_max=cfg.l_max,
                             product_quadrature=cfg.product_quadrature)
    return PrimitiveEquationsModel(planet, SigmaGrid.uniform(cfg.nlev),
                                   r_dry=cfg.r_dry, cp_dry=cfg.cp_dry,
                                   batched_transforms=cfg.batched_transforms,
                                   retained_truncation=cfg.retained_truncation)


def build_physics(cfg: HeldSuarezConfig, model) -> tuple[tuple, tuple]:
    """(explicit_terms, dampers): Newtonian relaxation; Rayleigh drag, del^8."""
    newton = NewtonianRelaxation(model, cfg.forcing)
    drag = rayleigh_drag_damping(model.sigma.full_levels_array(), cfg.l_max + 1,
                                 cfg.forcing)
    hyper = hyperdiffusion_damping(cfg.nlev, cfg.l_max, cfg.radius,
                                   cfg.hyperdiffusion_efold_days * DAY_SECONDS,
                                   cfg.hyperdiffusion_degree,
                                   order=cfg.hyperdiffusion_order)
    return (newton,), (drag, hyper)


def build_stepper(cfg: HeldSuarezConfig, model, *,
                  stage_validator: Optional[Callable] = None):
    """The SI leapfrog + RAW stepper with the Held–Suarez hooks."""
    from tropoi.temporal.semi_implicit import (SemiImplicitLeapfrogStepper,
                                               SemiImplicitOperator)
    op = SemiImplicitOperator.from_model(model, t_ref=cfg.t_ref)
    terms, dampers = build_physics(cfg, model)
    return SemiImplicitLeapfrogStepper(
        model.tendency, op, cfg.dt, raw_nu=cfg.raw_nu, raw_alpha=cfg.raw_alpha,
        startup_substeps=cfg.startup_substeps, stage_validator=stage_validator,
        explicit_terms=terms, dampers=dampers, damping_scheme=cfg.damping_scheme)


def perturbation_coefficients(cfg: HeldSuarezConfig) -> tuple[np.ndarray, dict[str, Any]]:
    """Deterministic vorticity perturbation, (nlev, l_max+1, l_max+1) complex.

    Generator ``numpy.random.Generator(PCG64(cfg.seed))``; for each level
    k = 0..K-1 (top to bottom), degree l = lmin..lmax, order m = 0..l, in that
    order, draw two standard normals (re, im); m = 0 keeps only re (a real
    field). Each level is then scaled so the grid RMS of zeta' over the
    sphere equals ``cfg.perturbation_amplitude`` exactly, using the
    repository's real-field convention (orthonormal Y, m > 0 counted twice):
    RMS^2 = (sum_l |a_l0|^2 + 2 sum_{m>0} |a_lm|^2) / (4 pi).
    Degrees are confined to [lmin, lmax] inside the product cut; the global
    mean (l = 0) is untouched; divergence is not seeded.
    Returns the coefficients and the generator's final bit-generator state.
    """
    rng = np.random.Generator(np.random.PCG64(cfg.seed))
    n = cfg.l_max + 1
    out = np.zeros((cfg.nlev, n, n), dtype=np.complex128)
    for k in range(cfg.nlev):
        for l in range(cfg.perturbation_lmin, cfg.perturbation_lmax + 1):
            for m in range(l + 1):
                re, im = rng.standard_normal(2)
                out[k, l, m] = complex(re, im if m > 0 else 0.0)
        power = (np.abs(out[k, :, 0]) ** 2).sum() + 2.0 * (np.abs(out[k, :, 1:]) ** 2).sum()
        out[k] *= cfg.perturbation_amplitude / math.sqrt(power / (4.0 * math.pi))
    return out, rng.bit_generator.state


def initial_state(cfg: HeldSuarezConfig):
    """Isothermal (t_init) resting atmosphere, uniform p_s = ps_init, plus the
    seeded vorticity perturbation. Returns (coeffs (CuPy), rng_state)."""
    import cupy as cp
    from tropoi.spatial.states.primitive_equations import isothermal_rest_state
    x0 = isothermal_rest_state(cfg.l_max, cfg.nlev, temperature=cfg.t_init,
                               surface_pressure=cfg.ps_init).coeffs
    pert, rng_state = perturbation_coefficients(cfg)
    x0[0:cfg.nlev] += cp.asarray(pert)
    return x0, rng_state
