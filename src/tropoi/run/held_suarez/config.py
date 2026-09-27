"""Held–Suarez experiment configuration (CPU-only, hashable).

Every quantity that determines the numerical result of a run is a field of
:class:`HeldSuarezConfig`; its canonical JSON (sorted keys, repr floats) is
hashed into ``config_sha256``, which every checkpoint carries and every
resume compares (PROTOCOL.md §2, plan §2 B5). Nothing here imports CuPy.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Optional

from tropoi.spatial.truncation import product_truncation_cut
from tropoi.temporal.tendencies.held_suarez import (DAY_SECONDS,
                                                    HeldSuarezParameters)

__all__ = ["HeldSuarezConfig", "production_config", "development_config",
           "smoke_config", "PRESETS", "CONFIG_SCHEMA"]

CONFIG_SCHEMA = "palintropos.held_suarez.config/1"


@dataclass(frozen=True)
class HeldSuarezConfig:
    """One Held–Suarez experiment. Times in seconds unless named *_days."""

    experiment_id: str
    l_max: int
    nlat: int
    nlon: int
    nlev: int
    dt: float
    days: int
    spinup_days: int = 200
    block_days: int = 200
    # planet and gas (PROTOCOL §1.1 / §2)
    radius: float = 6.371e6
    omega: float = 7.292e-5
    gravity: float = 9.8
    cp_dry: float = 1004.0
    kappa: float = 2.0 / 7.0
    # initial state
    t_init: float = 300.0
    ps_init: float = 1.0e5
    seed: int = 20260927
    perturbation_amplitude: float = 1.0e-6     # grid RMS of zeta' per level, s^-1
    perturbation_lmin: int = 1
    perturbation_lmax: int = 8
    # time scheme (frozen at launch)
    t_ref: float = 300.0
    raw_nu: float = 0.1
    raw_alpha: float = 0.53
    startup_substeps: Optional[int] = None     # None = the stepper's rule
    # dissipation (frozen at launch)
    hyperdiffusion_order: int = 4
    hyperdiffusion_efold_days: float = 0.1
    hyperdiffusion_reference_degree: Optional[int] = None   # None = l_max (PROTOCOL §2)
    forcing: HeldSuarezParameters = field(default_factory=HeldSuarezParameters)
    # numerics of the transforms
    grid: str = "latlon"
    product_quadrature: str = "fine"
    batched_transforms: bool = True
    # output cadence
    checkpoint_every_days: int = 10
    keep_checkpoints: int = 3
    state_every_days: int = 10

    def __post_init__(self):
        if not self.experiment_id or any(c in self.experiment_id for c in "/\\ "):
            raise ValueError(f"bad experiment_id {self.experiment_id!r}")
        if self.grid != "latlon":
            raise ValueError("the Held–Suarez experiment runs on the Gauss lat-lon grid")
        for name in ("l_max", "nlat", "nlon", "nlev", "days", "block_days",
                     "checkpoint_every_days", "keep_checkpoints", "state_every_days"):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be >= 1")
        if self.nlat < self.l_max + 1 or self.nlon < 2 * self.l_max + 1:
            raise ValueError("state grid too coarse for exact analysis at l_max")
        if not (math.isfinite(self.dt) and self.dt > 0):
            raise ValueError("dt must be finite and > 0")
        spd = DAY_SECONDS / self.dt
        if abs(spd - round(spd)) > 1e-9:
            raise ValueError(f"dt = {self.dt} s must divide one day")
        if self.spinup_days < 0:
            raise ValueError("spinup_days must be >= 0")
        cut = product_truncation_cut(self.l_max)
        if not (1 <= self.perturbation_lmin <= self.perturbation_lmax <= cut):
            raise ValueError(
                f"perturbation degrees [{self.perturbation_lmin}, {self.perturbation_lmax}] "
                f"must lie in [1, product cut {cut}] (SEMI_IMPLICIT.md §2)")
        ref = self.hyperdiffusion_reference_degree
        if ref is not None and not (1 <= ref <= self.l_max):
            raise ValueError("hyperdiffusion_reference_degree must be in [1, l_max]")
        if self.startup_substeps is not None and int(self.startup_substeps) < 1:
            raise ValueError("startup_substeps must be >= 1")
        if isinstance(self.forcing, dict):
            object.__setattr__(self, "forcing", HeldSuarezParameters(**self.forcing))

    # ------------------------------------------------------------------
    @property
    def r_dry(self) -> float:
        return self.kappa * self.cp_dry

    @property
    def steps_per_day(self) -> int:
        return int(round(DAY_SECONDS / self.dt))

    @property
    def total_steps(self) -> int:
        return self.days * self.steps_per_day

    @property
    def product_cut(self) -> int:
        """Highest degree the core evolves (analyzed products are truncated
        here): the effective spectral truncation of the run."""
        return product_truncation_cut(self.l_max)

    @property
    def hyperdiffusion_degree(self) -> int:
        return (self.l_max if self.hyperdiffusion_reference_degree is None
                else int(self.hyperdiffusion_reference_degree))

    @property
    def n_blocks(self) -> int:
        return max(0, (self.days - self.spinup_days) // self.block_days)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["schema"] = CONFIG_SCHEMA
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "HeldSuarezConfig":
        d = dict(d)
        schema = d.pop("schema", CONFIG_SCHEMA)
        if schema != CONFIG_SCHEMA:
            raise ValueError(f"unknown config schema {schema!r}")
        d["forcing"] = HeldSuarezParameters(**d["forcing"])
        return cls(**d)

    def canonical_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    def with_(self, **changes) -> "HeldSuarezConfig":
        return replace(self, **changes)


def production_config(experiment_id: str = "hs-T42L20-prod-001", *,
                      dt: float = 1200.0) -> HeldSuarezConfig:
    """The protocol configuration (PROTOCOL.md §2): T42 L20 on 64x128, 1200 days."""
    return HeldSuarezConfig(experiment_id=experiment_id, l_max=42, nlat=64, nlon=128,
                            nlev=20, dt=dt, days=1200)


def development_config(experiment_id: str = "hs-T21L10-dev", *, days: int = 5,
                       dt: float = 1200.0, nlev: int = 10) -> HeldSuarezConfig:
    """Local development resolution (T21 on 32x64); statistics blocks scaled
    to the run so the pipeline is exercised end to end."""
    return HeldSuarezConfig(experiment_id=experiment_id, l_max=21, nlat=32, nlon=64,
                            nlev=nlev, dt=dt, days=days, spinup_days=0,
                            block_days=max(1, days // 2), checkpoint_every_days=1,
                            state_every_days=1)


def smoke_config(experiment_id: str = "hs-T21L10-smoke") -> HeldSuarezConfig:
    """Plan tier A3: T21 L10, 2 days, SI, checkpoint every day; one 2-day
    statistics block (so the accumulators sum more than one sample)."""
    return development_config(experiment_id, days=2).with_(block_days=2)


PRESETS = {"production": production_config, "development": development_config,
           "smoke": smoke_config}
