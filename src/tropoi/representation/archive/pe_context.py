"""What a saved primitive-equation run says about its vertical column (CPU).

A :class:`PEColumnContext` collects, from a saved run's own metadata, every
constant and boundary condition a host reconstruction of physical fields
needs: the native sigma interfaces, the gas constants, the dynamical
radius, the model's gravity (when the run defines one) and the resolved
surface-elevation coefficients (``None`` = a flat lower boundary). Each value
carries a provenance line saying where it came from; nothing is guessed.

Builders live with the archive formats: :func:`pe_capsule_context` for
``tropoi run pe`` capsules (this module) and
``HeldSuarezStorage.pe_context`` for Held–Suarez experiment directories
(``representation.archive.held_suarez``). The consumer is
``representation.diagnostics.pe_levels``.

Host only (NumPy, stdlib, ``spatial.environment`` for the planet radius);
never imports CuPy, except :func:`pe_capsule_context` with
``allow_cuda=True`` for a preset terrain whose coefficients were never saved.
"""
from __future__ import annotations

import math
import pathlib
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

from tropoi.spatial.sigma_coordinate import SigmaGrid

__all__ = [
    "PEColumnContext",
    "PEContextError",
    "PRESSURE_CONVENTION",
    "pe_capsule_context",
]

#: The PE core's vertical coordinate: pure sigma with the model top at p = 0.
PRESSURE_CONVENTION = ("p = sigma * p_s (pure sigma coordinate; model top "
                       "interface sigma = 0, p = 0)")


class PEContextError(ValueError):
    """The saved run does not define what a physical reconstruction needs."""


@dataclass(frozen=True, eq=False)
class PEColumnContext:
    """Constants and boundary conditions of one saved PE run (host values).

    ``surface_elevation_lm`` are surface-elevation coefficients in metres
    (repository layout, ``(l_max+1, l_max+1)``) or ``None`` for a flat lower
    boundary. ``gravity`` is the model's gravity (m s^-2): the factor of
    ``Phi_s = g h_s`` for terrain runs and the Held–Suarez constant; ``None``
    when the run never defined one (a flat ``tropoi run pe`` preset).
    """

    label: str
    kind: str
    l_max: int
    sigma_interfaces: tuple[float, ...]
    r_dry: float
    cp_dry: float
    radius_m: float
    gravity: float | None
    surface_elevation_lm: np.ndarray | None
    terrain_label: str
    provenance: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        SigmaGrid(tuple(self.sigma_interfaces))      # validates
        for name in ("r_dry", "cp_dry", "radius_m"):
            value = float(getattr(self, name))
            if not (math.isfinite(value) and value > 0.0):
                raise PEContextError(f"{name} must be finite and > 0, got "
                                     f"{value}")
        if self.gravity is not None and not (
                math.isfinite(self.gravity) and self.gravity > 0.0):
            raise PEContextError(f"gravity must be finite and > 0, got "
                                 f"{self.gravity}")
        lm = self.surface_elevation_lm
        if lm is not None:
            lm = np.array(lm, dtype=np.complex128)
            if lm.shape != (self.l_max + 1,) * 2:
                raise PEContextError(
                    f"terrain coefficients {lm.shape} do not match l_max "
                    f"{self.l_max}")
            if not np.isfinite(lm).all():
                raise PEContextError("terrain coefficients are not finite")
            lm.flags.writeable = False
            object.__setattr__(self, "surface_elevation_lm", lm)
            if self.gravity is None:
                raise PEContextError(
                    "a run with terrain must record the gravity of Phi_s = "
                    "g h_s")

    @property
    def sigma(self) -> SigmaGrid:
        return SigmaGrid(tuple(self.sigma_interfaces))

    @property
    def nlev(self) -> int:
        return len(self.sigma_interfaces) - 1

    @property
    def full_levels(self) -> tuple[float, ...]:
        return self.sigma.full_levels

    @property
    def is_flat(self) -> bool:
        return self.surface_elevation_lm is None

    @property
    def pressure_convention(self) -> str:
        return PRESSURE_CONVENTION

    def surface_geopotential_lm(self) -> np.ndarray:
        """``Phi_s = g h_s`` coefficients, exactly as the cores build them
        (zeros for a flat boundary)."""
        n = self.l_max + 1
        if self.surface_elevation_lm is None:
            return np.zeros((n, n), dtype=np.complex128)
        return float(self.gravity) * self.surface_elevation_lm

    def with_gravity(self, gravity: float | None) -> "PEColumnContext":
        """This context, with a caller-supplied gravity where none is saved.

        A recorded gravity is never replaced: a different value raises.
        """
        if gravity is None:
            return self
        gravity = float(gravity)
        if self.gravity is not None:
            if gravity != self.gravity:
                raise PEContextError(
                    f"this run records gravity = {self.gravity:g} m s^-2 "
                    f"({self.provenance.get('gravity', 'saved')}); a different "
                    f"value ({gravity:g}) would contradict its geopotential")
            return self
        provenance = dict(self.provenance)
        provenance["gravity"] = ("supplied by the caller (the run itself "
                                 "defines no gravity)")
        return PEColumnContext(
            label=self.label, kind=self.kind, l_max=self.l_max,
            sigma_interfaces=self.sigma_interfaces, r_dry=self.r_dry,
            cp_dry=self.cp_dry, radius_m=self.radius_m, gravity=gravity,
            surface_elevation_lm=self.surface_elevation_lm,
            terrain_label=self.terrain_label, provenance=provenance)


# ---------------------------------------------------------------------------
# tropoi run pe capsules
# ---------------------------------------------------------------------------

def _capsule_radius(run_config: Mapping) -> tuple[float, str]:
    """The dynamical radius ``run.pe.model.build_pe_model`` builds the planet with."""
    from tropoi.spatial.environment import PlanetaryParameters

    try:
        day_hours = float(run_config["day_hours"])
        units = float(run_config["radius_earth_units"])
    except (KeyError, TypeError, ValueError) as err:
        raise PEContextError(
            f"PE run configuration lacks day_hours/radius_earth_units: "
            f"{err}") from err
    radius = float(PlanetaryParameters.from_earth_like(
        day_hours=day_hours, radius_earth_units=units).radius)
    return radius, (f"PlanetaryParameters.from_earth_like(day_hours="
                    f"{day_hours:g}, radius_earth_units={units:g}).radius, "
                    "the volumetric-mean radius run.pe.model.build_pe_model uses")


def _prepared_terrain(storage, run_config: Mapping) -> tuple[np.ndarray, str]:
    from tropoi.representation.archive.prepared_terrain import (
        load_prepared_terrain)

    path = pathlib.Path(storage.run_dir) / "terrain"
    terrain = load_prepared_terrain(path)
    if terrain.digest != run_config["terrain_digest"]:
        raise PEContextError(
            f"the capsule's terrain/ ({terrain.digest}) does not match the "
            f"run's recorded terrain_digest ({run_config['terrain_digest']})")
    name = terrain.label or terrain.topography.preset
    return (np.asarray(terrain.topography_lm),
            f"{name}: prepared terrain persisted in the capsule's terrain/ "
            f"(digest {terrain.digest[:19]}…, band limit "
            f"{run_config.get('terrain_l_cut')})")


def _model_terrain(storage) -> np.ndarray:
    """Preset terrain through the run's own CUDA model (opt-in)."""
    model = storage.resources().model
    gravity = float(storage.run_config["gravity"])
    return np.asarray(model.phi_surface_lm.get()) / gravity


def pe_capsule_context(storage, *, allow_cuda: bool = False
                       ) -> PEColumnContext:
    """The column context of a ``tropoi run pe`` capsule.

    Terrain: a prepared terrain (``terrain_digest``) is read from the
    capsule's ``terrain/`` and digest-checked; a flat run has none. A preset
    mountain was re-projected from its configuration and its coefficients
    were never saved: rebuilding them needs the run's CUDA transform, done
    only with ``allow_cuda=True`` (through the same model resources
    ``Snapshot.plot`` uses); otherwise this raises rather than inventing a
    terrain.
    """
    if storage.solver != "pe":
        raise PEContextError(
            f"the atmosphere view needs a primitive-equation run; this is a "
            f"{storage.solver!r} run")
    run_config = storage.run_config
    schema = storage.schema
    interfaces = tuple(float(s) for s in schema.vertical["interfaces"])
    provenance = {
        "sigma_interfaces": (
            "state_schema vertical block (source: "
            f"{schema.provenance.get('source', 'manifest')})"
            if "sigma_interfaces" in run_config else
            "no sigma_interfaces key: SigmaGrid.uniform(nlev), the "
            "runner's default"),
        "r_dry": "run configuration r_dry",
        "cp_dry": "run configuration cp_dry",
    }
    radius, provenance["radius_m"] = _capsule_radius(run_config)
    gravity = run_config.get("gravity")
    if gravity is not None:
        gravity = float(gravity)
        provenance["gravity"] = "run configuration gravity (Phi_s = g h_s)"
    topography = run_config.get("topography", "flat")
    elevation = None
    if "terrain_digest" in run_config:
        elevation, terrain_label = _prepared_terrain(storage, run_config)
    elif topography == "flat":
        terrain_label = ("flat lower boundary (no topography in the run "
                         "configuration)")
    elif topography == "mountain":
        if not allow_cuda:
            raise PEContextError(
                "this run's terrain is the 'mountain' preset, re-projected "
                "from its configuration by the run's CUDA transform; its "
                "coefficients were never saved, so a CPU view cannot "
                "reproduce it exactly and will not invent one. Pass "
                "allow_cuda=True to rebuild it through the run's model "
                "(needs CUDA)")
        elevation = _model_terrain(storage)
        terrain_label = ("mountain preset rebuilt through the run's CUDA "
                         "model from its configuration")
    else:
        raise PEContextError(
            f"unknown topography {topography!r} in the run configuration")
    provenance["terrain"] = terrain_label
    return PEColumnContext(
        label=str(run_config.get("run_id") or storage.run_dir.name),
        kind="pe_capsule", l_max=int(schema.l_max),
        sigma_interfaces=interfaces, r_dry=float(run_config["r_dry"]),
        cp_dry=float(run_config["cp_dry"]), radius_m=radius, gravity=gravity,
        surface_elevation_lm=elevation, terrain_label=terrain_label,
        provenance=provenance)
