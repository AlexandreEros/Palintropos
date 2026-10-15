"""Resolved configuration of a run started from a saved initial state.

Import-light (NumPy + stdlib; never CuPy): every user error -- an invalid
schedule, a discretization that does not match the state's truncation, a
BVE-only option on a shallow-water state -- fails here, before any CUDA
work, exactly like the preset configuration layers.

The run configuration a custom run records (``config.json`` /
``manifest.json["run_config"]``) uses the preset key set where the meaning
is shared (``lmax``, ``grid``, ``resolution``, ``nlat``, ``nlon``,
``product_quadrature``, ``day_hours``, ``duration_days``, the snapshot keys,
``gravity``, ``mean_depth_m``, the topography keys, ``viscosity``) and adds:

``solver``                ``"bve"`` or ``"swe"`` (explicit for both)
``scenario``              ``"custom"``
``planet``                the versioned :class:`PlanetSpec` block
``initial_state_format``  the saved-state format version
``initial_state_digest``  the state's scientific identity (``sha256:...``)
``initial_state_label``   descriptive only; EXCLUDED from the run-id hash
``topography_parameters`` / ``topography_digest``  SWE over terrain

so the run id's scientific hash covers the planet, the discretization, the
schedule and the initial state's content, but not its label or file name.
``radius_earth_units`` is deliberately absent: a custom planet is not
measured in Earth radii.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

from tropoi.representation.archive.initial_state import InitialState
from tropoi.run.bve.config import (DEFAULT_N_SNAPSHOTS, BVERunConfig,
                                   scientific_config_subset)
from tropoi.spatial.specification import DiscretizationSpec
from tropoi.temporal.integration import (SECONDS_PER_DAY,
                                         _require_finite_nonneg,
                                         _require_finite_positive,
                                         count_snapshot_times,
                                         interval_snapshot_times)

#: Cap on t_end (seconds), as in the preset configuration layers.
_MAX_T_END_SECONDS = 1e12


@dataclass(frozen=True, eq=False)
class CustomRunConfig:
    """Fully resolved configuration for one run of a saved initial state.

    Build with :meth:`resolve`. Exposes the attributes the shared run
    lifecycle needs (``out``, ``experiment``, ``overwrite``,
    ``to_run_config_dict()``).
    """

    initial_state: InitialState
    discretization: DiscretizationSpec
    duration_days: float
    snapshot_mode: str
    n_snapshots: Optional[int]
    dt_snapshots: Optional[float]
    viscosity: float = 0.0
    plots: tuple[str, ...] = ()
    out: str = "runs"
    experiment: Optional[str] = None
    overwrite: bool = False

    @property
    def solver(self) -> str:
        return self.initial_state.solver

    @property
    def scenario(self) -> str:
        return "custom"

    @classmethod
    def resolve(cls, initial_state: InitialState, *,
                duration_days: float,
                n_snapshots: Optional[int] = None,
                snapshot_interval_s: Optional[float] = None,
                discretization: Optional[DiscretizationSpec] = None,
                viscosity: Optional[float] = None,
                plots: Optional[Sequence[str]] = None,
                out: str = "runs",
                experiment: Optional[str] = None,
                overwrite: bool = False) -> "CustomRunConfig":
        """Validate and resolve the execution settings for ``initial_state``.

        ``discretization`` defaults to the one the state was resolved for;
        another grid is allowed, but its ``lmax`` must equal the state's
        (no silent truncation or padding). ``n_snapshots`` (count mode,
        default 5) and ``snapshot_interval_s`` (interval mode) are mutually
        exclusive, with the preset semantics. ``viscosity`` (m^2/s) is the
        BVE's optional Laplacian viscosity and is rejected for the SWE.
        ``plots`` selects image products (``"diagnostics"``,
        ``"snapshots"``, ``"summary"`` or ``"all"``); none by default.
        """
        if not isinstance(initial_state, InitialState):
            raise TypeError(
                f"expected an InitialState, got {type(initial_state).__name__}")
        if discretization is None:
            discretization = initial_state.discretization
        if not isinstance(discretization, DiscretizationSpec):
            raise TypeError(
                "discretization must be a DiscretizationSpec, got "
                f"{type(discretization).__name__}")
        if discretization.lmax != initial_state.l_max:
            raise ValueError(
                f"discretization lmax={discretization.lmax} does not match the "
                f"initial state's l_max={initial_state.l_max}; the stored "
                "coefficients are never truncated or padded implicitly")
        if initial_state.solver == "swe" \
                and discretization.product_quadrature != "fine":
            raise ValueError(
                "the shallow-water core always uses product_quadrature='fine'")

        if viscosity is None:
            viscosity = 0.0
        if initial_state.solver == "swe" and viscosity != 0.0:
            raise ValueError(
                "viscosity applies to the BVE only; the shallow-water runs "
                "are inviscid")
        viscosity = float(_require_finite_nonneg("viscosity", viscosity))

        duration = _require_finite_positive("duration_days", duration_days)
        t_end = duration * SECONDS_PER_DAY
        if not math.isfinite(t_end) or t_end > _MAX_T_END_SECONDS:
            raise ValueError(
                f"duration_days = {duration_days} overflows t_end "
                f"(got {t_end} s, cap {_MAX_T_END_SECONDS} s)")

        if n_snapshots is not None and snapshot_interval_s is not None:
            raise ValueError(
                "n_snapshots and snapshot_interval_s are mutually exclusive")
        if snapshot_interval_s is None:
            n = DEFAULT_N_SNAPSHOTS if n_snapshots is None else n_snapshots
            if isinstance(n, bool) or not isinstance(n, int) or n < 0:
                raise ValueError(
                    f"n_snapshots must be an integer >= 0, got {n!r}")
            mode, dt = "count", (t_end / (n - 1) if n >= 2 else None)
            times = count_snapshot_times(n, t_end)
        else:
            n = None
            mode = "interval"
            dt = _require_finite_positive("snapshot_interval_s",
                                          snapshot_interval_s)
            times = interval_snapshot_times(dt, t_end)

        resolved_plots = BVERunConfig._resolve_plots(
            None if plots is None else list(plots), None,
            has_snapshots=bool(times))
        if resolved_plots and not times and any(
                p in ("snapshots", "summary") for p in resolved_plots):
            raise ValueError(
                "the snapshots/summary plots need at least one stored state")

        return cls(initial_state=initial_state, discretization=discretization,
                   duration_days=duration, snapshot_mode=mode, n_snapshots=n,
                   dt_snapshots=dt, viscosity=viscosity,
                   plots=resolved_plots, out=str(out), experiment=experiment,
                   overwrite=bool(overwrite))

    # ------------------------------------------------------------------

    def snapshot_times_seconds(self) -> list[float]:
        t_end = self.duration_days * SECONDS_PER_DAY
        if self.snapshot_mode == "count":
            return count_snapshot_times(self.n_snapshots, t_end)
        return interval_snapshot_times(self.dt_snapshots, t_end)

    def to_run_config_dict(self) -> dict:
        """The run configuration (see the module docstring)."""
        config = self.initial_state.run_config_entries()
        config.update(self.discretization.run_config_entries())
        config.update({
            "duration_days": self.duration_days,
            "dt_snapshots": self.dt_snapshots,
            "out": self.out,
            "experiment": self.experiment,
            "overwrite": self.overwrite,
            "snapshot_mode": self.snapshot_mode,
            "n_snapshots": self.n_snapshots,
            "snapshot_times": self.snapshot_times_seconds(),
            "plots": list(self.plots),
        })
        if self.solver == "bve":
            config["viscosity"] = self.viscosity
        return config

    def scientific_config_dict(self) -> dict:
        return scientific_config_subset(self.to_run_config_dict())

    def summary_lines(self) -> list[str]:
        state = self.initial_state
        lines = ["Resolved run configuration:",
                 f"  solver              {self.solver}",
                 f"  initial state       {state.label or '(unlabelled)'} "
                 f"[{state.digest[:19]}...]",
                 f"  planet              {state.planet.describe()}",
                 f"  discretization      {self.discretization.describe()}",
                 f"  duration            {self.duration_days:g} days"]
        if self.solver == "swe":
            lines.append(
                f"  fluid layer         g = {state.fluid.gravity_m_s2:g} m/s^2,"
                f" H = {state.fluid.mean_depth_m:g} m")
            lines.append(f"  topography          {state.topography.preset}")
        else:
            lines.append(f"  viscosity           {self.viscosity:g} m^2/s")
        return lines


__all__ = ["CustomRunConfig"]
