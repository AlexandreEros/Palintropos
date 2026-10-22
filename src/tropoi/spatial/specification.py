"""Versioned, persistable specifications of a custom planet and its discretization.

These objects describe *what* a model run is about, separately from *how*
it is discretized and from the CUDA resources a run builds at execution
time (grids, transforms, product spaces, operators), which are never part
of a specification and never serialized:

* :class:`PlanetSpec` -- the prescribed planet: radius and its geometry
  convention, rotation, mass;
* :class:`FluidLayerSpec` -- the prescribed shallow-water layer: explicit
  gravity and mean fluid thickness;
* :class:`TopographySpec` -- the prescribed SWE bottom topography: the
  terrain presets (``flat``, ``mountain``, ``williamson5_cone``) or the
  provenance record of terrain ``imported`` from an elevation file;
* :class:`DiscretizationSpec` -- spectral truncation, grid family and
  dimensions, nonlinear-product sampling.

Each has ``to_dict()`` / ``from_dict()`` with a ``version`` field and only
JSON-native values (non-finite floats are never written; a non-rotating
planet records ``sidereal_day_s: null``). ``from_dict`` validates and never
evaluates anything taken from the document.

Radius and geometry convention
------------------------------
:class:`PlanetSpec` keeps the existing distinction between the two ways the
repository builds a planet (``spatial/environment.py``):

``"oblate_volumetric_mean"``
    The ordinary construction ``PlanetaryParameters(mass, equatorial_radius,
    sidereal_day)`` (what ``from_earth_like`` and every non-benchmark preset
    use). ``reference_radius_m`` is the *equatorial* radius; the dynamical
    radius the cores use is the volumetric-mean radius of the first-order
    rotational oblate spheroid, which is *derived* (about 0.06 % smaller
    than the equatorial radius for Earth-like rotation; equal to it for a
    non-rotating planet).
``"ideal_sphere"``
    ``PlanetaryParameters.ideal_sphere(radius_m, sidereal_day_s, mass_kg)``:
    a perfect sphere whose dynamical radius is ``reference_radius_m``
    EXACTLY. Benchmarks (Williamson et al. 1992) use this.

``to_dict()`` also records the derived values (dynamical radius, angular
velocity, oblateness); ``from_dict()`` recomputes them and refuses a
document whose recorded values differ, so a reloaded planet is the same
planet bit for bit, or loading fails loudly.

Shallow-water gravity is explicit (:class:`FluidLayerSpec`) and is never
derived from mass and radius, exactly as in the shallow-water core.

Import-light: stdlib plus :mod:`tropoi.spatial.environment` (NumPy/SciPy);
never CuPy.
"""
from __future__ import annotations

import math
import numbers
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from tropoi.spatial.williamson5 import W5_CONE_HEIGHT_M

__all__ = [
    "DiscretizationSpec",
    "FluidLayerSpec",
    "IMPORTED_REQUIRED_KEYS",
    "PLANET_CONSTRUCTIONS",
    "PlanetSpec",
    "SPEC_VERSION",
    "SpecificationError",
    "TOPOGRAPHY_SPEC_PRESETS",
    "TopographySpec",
]

#: Version of every ``to_dict()`` document in this module.
SPEC_VERSION = 1

#: Planet geometry conventions (see the module docstring).
PLANET_CONSTRUCTIONS = ("oblate_volumetric_mean", "ideal_sphere")

_RADIUS_CONVENTIONS = {
    "oblate_volumetric_mean": (
        "reference_radius_m is the equatorial radius; the dynamical radius "
        "is the derived volumetric-mean radius of the first-order "
        "rotational oblate spheroid (PlanetaryParameters(mass, "
        "equatorial_radius, sidereal_day))"),
    "ideal_sphere": (
        "perfect sphere: the dynamical radius is reference_radius_m exactly "
        "(PlanetaryParameters.ideal_sphere)"),
}

#: Default planetary mass (kg); the value ``from_earth_like`` and
#: ``ideal_sphere`` use. The shallow-water and barotropic cores never use
#: the mass dynamically (gravity is explicit).
EARTH_MASS_KG = 5.972e24
#: Earth equatorial radius (m) used by ``PlanetaryParameters.from_earth_like``.
EARTH_RADIUS_M = 6.371e6

#: Grid families and product samplings (kept in sync with
#: ``run.bve.config.GRID_TYPES`` / ``PRODUCT_QUADRATURES`` by a test; this
#: module may not import the run layer).
GRID_TYPES = ("geodesic", "latlon")
PRODUCT_QUADRATURES = ("fine", "coarse")
MIN_NLAT = 2
MIN_NLON = 4

#: Terrain presets a specification may name. The same vocabulary as
#: ``spatial.terrain.topography.TOPOGRAPHY_PRESETS`` (synchronized by a
#: test). ``williamson5_cone`` is the benchmark terrain, for reproducing or
#: perturbing Williamson case 5. ``imported`` records the source and the
#: preprocessing of terrain read from a file; it is never re-resolved from
#: the specification (the resolved coefficients travel with it).
TOPOGRAPHY_SPEC_PRESETS = ("flat", "mountain", "williamson5_cone", "imported")
#: Provenance keys an ``imported`` specification must carry.
IMPORTED_REQUIRED_KEYS = ("source_name", "source_sha256", "format")
#: Physical sanity cap on mountain heights (m); synchronized by a test with
#: ``spatial.terrain.topography.MAX_MOUNTAIN_HEIGHT_M``.
MAX_MOUNTAIN_HEIGHT_M = 1.0e5


class SpecificationError(ValueError):
    """A specification is invalid, or a saved document does not describe one."""


# ---------------------------------------------------------------------------
# Small validators
# ---------------------------------------------------------------------------

def _number(name: str, value: Any) -> float:
    if isinstance(value, bool):
        raise SpecificationError(f"{name} must be a real number, got {value!r}")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise SpecificationError(
            f"{name} must be a real number, got {value!r}") from None
    if not math.isfinite(number):
        raise SpecificationError(f"{name} must be finite, got {value!r}")
    return number


def _positive(name: str, value: Any) -> float:
    number = _number(name, value)
    if not number > 0.0:
        raise SpecificationError(f"{name} must be > 0, got {value!r}")
    return number


def _integer(name: str, value: Any, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise SpecificationError(f"{name} must be an integer, got {value!r}")
    if value < minimum:
        raise SpecificationError(f"{name} must be >= {minimum}, got {value}")
    return int(value)


def _check_statement(data: Mapping, key: str, expected: str) -> None:
    """A recorded convention statement must be the one this version means."""
    if key in data and data[key] != expected:
        raise SpecificationError(
            f"{key} {data[key]!r} is not the convention this reader "
            f"implements ({expected!r}); refusing to reinterpret it")


def _check_document(data: Any, kind: str, keys: set[str]) -> Mapping:
    if not isinstance(data, Mapping):
        raise SpecificationError(
            f"{kind} must be an object, got {type(data).__name__}")
    version = data.get("version")
    if version != SPEC_VERSION:
        raise SpecificationError(
            f"{kind} version {version!r} is not supported (known: "
            f"{SPEC_VERSION})")
    unknown = set(data) - keys - {"version"}
    if unknown:
        raise SpecificationError(
            f"{kind} has unknown key(s) {sorted(unknown)}")
    return data


# ---------------------------------------------------------------------------
# Planet
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PlanetSpec:
    """The prescribed planet: radius and geometry convention, rotation, mass.

    Build instances with :meth:`ideal_sphere`, :meth:`oblate` or
    :meth:`earth_like`. ``sidereal_day_s`` is ``None`` for a non-rotating
    planet (``Omega = 0``). :meth:`to_parameters` returns the
    :class:`~tropoi.spatial.environment.PlanetaryParameters` the cores use,
    constructed exactly as the corresponding preset path constructs them.
    """

    construction: str
    reference_radius_m: float
    sidereal_day_s: float | None
    mass_kg: float = EARTH_MASS_KG

    def __post_init__(self) -> None:
        if self.construction not in PLANET_CONSTRUCTIONS:
            raise SpecificationError(
                f"planet construction must be one of {PLANET_CONSTRUCTIONS}, "
                f"got {self.construction!r}")
        object.__setattr__(self, "reference_radius_m",
                           _positive("reference_radius_m",
                                     self.reference_radius_m))
        if self.sidereal_day_s is not None:
            object.__setattr__(self, "sidereal_day_s",
                               _positive("sidereal_day_s",
                                         self.sidereal_day_s))
        object.__setattr__(self, "mass_kg", _positive("mass_kg", self.mass_kg))

    # -- constructors --------------------------------------------------------

    @classmethod
    def ideal_sphere(cls, radius_m: float, sidereal_day_s: float | None,
                     mass_kg: float = EARTH_MASS_KG) -> "PlanetSpec":
        """A perfect sphere whose dynamical radius is exactly ``radius_m``."""
        return cls("ideal_sphere", radius_m, _day_or_none(sidereal_day_s),
                   mass_kg)

    @classmethod
    def oblate(cls, equatorial_radius_m: float, sidereal_day_s: float | None,
               mass_kg: float = EARTH_MASS_KG) -> "PlanetSpec":
        """The ordinary construction: dynamical radius derived from rotation."""
        return cls("oblate_volumetric_mean", equatorial_radius_m,
                   _day_or_none(sidereal_day_s), mass_kg)

    @classmethod
    def earth_like(cls, day_hours: float = 24.0,
                   radius_earth_units: float = 1.0,
                   mass_earth_units: float = 1.0) -> "PlanetSpec":
        """Mirror of ``PlanetaryParameters.from_earth_like`` (same arithmetic).

        ``day_hours = math.inf`` is the non-rotating planet the BVE presets
        use. The resulting parameters are bitwise identical to
        ``from_earth_like(mass_earth_units, radius_earth_units, day_hours)``.
        """
        return cls.oblate(radius_earth_units * EARTH_RADIUS_M,
                          day_hours * 3600.0,
                          mass_earth_units * EARTH_MASS_KG)

    # -- derived quantities --------------------------------------------------

    def to_parameters(self):
        """The :class:`PlanetaryParameters` the cores consume (CPU only)."""
        from tropoi.spatial.environment import PlanetaryParameters

        day = math.inf if self.sidereal_day_s is None else self.sidereal_day_s
        if self.construction == "ideal_sphere":
            return PlanetaryParameters.ideal_sphere(
                radius_m=self.reference_radius_m, sidereal_day_s=day,
                mass_kg=self.mass_kg)
        return PlanetaryParameters(mass=self.mass_kg,
                                   equatorial_radius=self.reference_radius_m,
                                   sidereal_day=day)

    @property
    def rotating(self) -> bool:
        return self.sidereal_day_s is not None

    @property
    def day_hours(self) -> float:
        """Sidereal day in hours (``math.inf`` when non-rotating).

        The run configuration's ``day_hours`` key (run-id rotation tag,
        inspection) is this value.
        """
        if self.sidereal_day_s is None:
            return math.inf
        return self.sidereal_day_s / 3600.0

    def derived(self) -> dict:
        """Dynamical radius, angular velocity and oblateness, as plain floats."""
        params = self.to_parameters()
        return {
            "dynamical_radius_m": float(params.radius),
            "angular_velocity_rad_s": float(params.angular_velocity),
            "oblateness": float(params.oblateness),
        }

    @property
    def dynamical_radius_m(self) -> float:
        return self.derived()["dynamical_radius_m"]

    @property
    def angular_velocity_rad_s(self) -> float:
        return self.derived()["angular_velocity_rad_s"]

    # -- persistence ---------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "version": SPEC_VERSION,
            "construction": self.construction,
            "radius_convention": _RADIUS_CONVENTIONS[self.construction],
            "reference_radius_m": self.reference_radius_m,
            "sidereal_day_s": self.sidereal_day_s,
            "mass_kg": self.mass_kg,
            "derived": self.derived(),
        }

    @classmethod
    def from_dict(cls, data: Mapping) -> "PlanetSpec":
        data = _check_document(
            data, "planet specification",
            {"construction", "radius_convention", "reference_radius_m",
             "sidereal_day_s", "mass_kg", "derived"})
        try:
            spec = cls(construction=data["construction"],
                       reference_radius_m=data["reference_radius_m"],
                       sidereal_day_s=data["sidereal_day_s"],
                       mass_kg=data["mass_kg"])
        except KeyError as err:
            raise SpecificationError(
                f"planet specification is missing {err}") from None
        _check_statement(data, "radius_convention",
                         _RADIUS_CONVENTIONS[spec.construction])
        recorded = data.get("derived")
        if recorded is not None:
            expected = spec.derived()
            if not isinstance(recorded, Mapping) or dict(recorded) != expected:
                raise SpecificationError(
                    "planet specification records derived values "
                    f"{dict(recorded) if isinstance(recorded, Mapping) else recorded!r} "
                    f"but this installation derives {expected} from the same "
                    "inputs; refusing to reinterpret the planet")
        return spec

    def describe(self) -> str:
        rotation = ("non-rotating" if self.sidereal_day_s is None
                    else f"sidereal day {self.sidereal_day_s:g} s")
        derived = self.derived()
        return (f"{self.construction} planet: dynamical radius "
                f"{derived['dynamical_radius_m']:.10g} m (reference "
                f"{self.reference_radius_m:.10g} m), {rotation}")


def _day_or_none(day: float | None) -> float | None:
    if day is None:
        return None
    if isinstance(day, (int, float)) and not isinstance(day, bool) \
            and math.isinf(day) and day > 0:
        return None
    return day


# ---------------------------------------------------------------------------
# Shallow-water layer
# ---------------------------------------------------------------------------

_PHI_DEFINITION = ("phi is the perturbation thickness geopotential: total "
                   "fluid thickness h = mean_depth_m + phi / gravity_m_s2")

@dataclass(frozen=True)
class FluidLayerSpec:
    """The prescribed shallow-water layer: explicit gravity and mean thickness.

    ``Phi0 = gravity_m_s2 * mean_depth_m`` is the mean thickness geopotential.
    The SWE prognostic ``phi`` is the PERTURBATION thickness geopotential:
    total fluid thickness is ``h = mean_depth_m + phi / gravity_m_s2``. It is
    neither the full geopotential nor the free-surface elevation (over
    terrain the free surface is ``h + h_s``).
    """

    gravity_m_s2: float
    mean_depth_m: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "gravity_m_s2",
                           _positive("gravity_m_s2", self.gravity_m_s2))
        object.__setattr__(self, "mean_depth_m",
                           _positive("mean_depth_m", self.mean_depth_m))

    @property
    def phi0(self) -> float:
        """Mean thickness geopotential ``g * H`` (m^2 s^-2)."""
        return self.gravity_m_s2 * self.mean_depth_m

    def to_dict(self) -> dict:
        return {
            "version": SPEC_VERSION,
            "gravity_m_s2": self.gravity_m_s2,
            "mean_depth_m": self.mean_depth_m,
            "phi_definition": _PHI_DEFINITION,
        }

    @classmethod
    def from_dict(cls, data: Mapping) -> "FluidLayerSpec":
        data = _check_document(data, "fluid-layer specification",
                               {"gravity_m_s2", "mean_depth_m",
                                "phi_definition"})
        _check_statement(data, "phi_definition", _PHI_DEFINITION)
        try:
            return cls(gravity_m_s2=data["gravity_m_s2"],
                       mean_depth_m=data["mean_depth_m"])
        except KeyError as err:
            raise SpecificationError(
                f"fluid-layer specification is missing {err}") from None


# ---------------------------------------------------------------------------
# Topography (existing presets only)
# ---------------------------------------------------------------------------

_MOUNTAIN_KEYS = ("height_m", "lat_deg", "lon_deg", "width_deg")


@dataclass(frozen=True)
class TopographySpec:
    """The prescribed SWE bottom topography, by existing terrain preset.

    ``flat`` takes no parameters; ``mountain`` takes ``height_m``,
    ``lat_deg``, ``lon_deg``, ``width_deg`` (the ``tropoi run swe
    --topography mountain`` Gaussian); ``williamson5_cone`` takes an
    optional ``height_m`` (default: the canonical 2000 m). Resolving a
    specification into spectral coefficients is a projection on the model
    grid (CUDA); a saved initial state stores the resolved coefficients so
    reloading never re-projects.

    ``imported`` (built by ``tropoi.run.custom.prepare_planet`` from an
    elevation file) is a provenance record, not a recipe: its parameters
    (JSON scalars: the file name and SHA-256, the reader's interpretation,
    the preprocessing settings and measured residuals) describe how the
    resolved coefficients were made, and those coefficients must always
    accompany it.
    """

    preset: str = "flat"
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.preset not in TOPOGRAPHY_SPEC_PRESETS:
            raise SpecificationError(
                f"topography preset must be one of {TOPOGRAPHY_SPEC_PRESETS}, "
                f"got {self.preset!r}")
        params = dict(self.parameters or {})
        if self.preset == "flat":
            if params:
                raise SpecificationError(
                    f"flat topography takes no parameters, got {sorted(params)}")
            resolved: dict[str, float] = {}
        elif self.preset == "mountain":
            missing = [k for k in _MOUNTAIN_KEYS if k not in params]
            unknown = sorted(set(params) - set(_MOUNTAIN_KEYS))
            if missing or unknown:
                raise SpecificationError(
                    f"mountain topography needs exactly {list(_MOUNTAIN_KEYS)}"
                    f" (missing {missing}, unknown {unknown})")
            height = _positive("mountain height_m", params["height_m"])
            if height > MAX_MOUNTAIN_HEIGHT_M:
                raise SpecificationError(
                    f"mountain height_m must be <= {MAX_MOUNTAIN_HEIGHT_M:g}, "
                    f"got {height}")
            lat = _number("mountain lat_deg", params["lat_deg"])
            if not -90.0 <= lat <= 90.0:
                raise SpecificationError(
                    f"mountain lat_deg must be in [-90, 90], got {lat}")
            lon = _number("mountain lon_deg", params["lon_deg"])
            if not -360.0 <= lon <= 360.0:
                raise SpecificationError(
                    f"mountain lon_deg must be in [-360, 360], got {lon}")
            width = _positive("mountain width_deg", params["width_deg"])
            if width > 90.0:
                raise SpecificationError(
                    f"mountain width_deg must be <= 90, got {width}")
            resolved = {"height_m": height, "lat_deg": lat, "lon_deg": lon,
                        "width_deg": width}
        elif self.preset == "imported":
            missing = [k for k in IMPORTED_REQUIRED_KEYS if k not in params]
            if missing:
                raise SpecificationError(
                    f"imported topography needs provenance keys {missing}")
            resolved = {}
            for key, value in params.items():
                if isinstance(value, float) and not math.isfinite(value):
                    raise SpecificationError(
                        f"imported topography parameter {key!r} is not "
                        "finite")
                if value is not None and not isinstance(
                        value, (str, int, float, bool)):
                    raise SpecificationError(
                        f"imported topography parameter {key!r} must be a "
                        f"JSON scalar, got {type(value).__name__}")
                resolved[str(key)] = value
        else:  # williamson5_cone
            unknown = sorted(set(params) - {"height_m"})
            if unknown:
                raise SpecificationError(
                    f"williamson5_cone takes only 'height_m', got {unknown}")
            height = _positive("cone height_m",
                               params.get("height_m", W5_CONE_HEIGHT_M))
            if height > MAX_MOUNTAIN_HEIGHT_M:
                raise SpecificationError(
                    f"cone height_m must be <= {MAX_MOUNTAIN_HEIGHT_M:g}, "
                    f"got {height}")
            resolved = {"height_m": height}
        object.__setattr__(self, "parameters", resolved)

    @classmethod
    def flat(cls) -> "TopographySpec":
        return cls("flat")

    @classmethod
    def mountain(cls, *, height_m: float, lat_deg: float, lon_deg: float,
                 width_deg: float) -> "TopographySpec":
        return cls("mountain", {"height_m": height_m, "lat_deg": lat_deg,
                                "lon_deg": lon_deg, "width_deg": width_deg})

    @classmethod
    def williamson5_cone(cls, *, height_m: float = W5_CONE_HEIGHT_M
                         ) -> "TopographySpec":
        return cls("williamson5_cone", {"height_m": height_m})

    @classmethod
    def imported(cls, parameters: Mapping[str, Any]) -> "TopographySpec":
        """Provenance record of terrain imported from a file."""
        return cls("imported", dict(parameters))

    @property
    def is_flat(self) -> bool:
        return self.preset == "flat"

    def to_dict(self) -> dict:
        return {"version": SPEC_VERSION, "preset": self.preset,
                "parameters": dict(self.parameters)}

    @classmethod
    def from_dict(cls, data: Mapping) -> "TopographySpec":
        data = _check_document(data, "topography specification",
                               {"preset", "parameters"})
        params = data.get("parameters") or {}
        if not isinstance(params, Mapping):
            raise SpecificationError(
                "topography parameters must be an object")
        return cls(preset=data.get("preset"), parameters=dict(params))

    def run_config_entries(self) -> dict:
        """The SWE run-configuration keys this terrain contributes.

        Matches the preset run configuration (``SWERunConfig``): nothing for
        a flat bottom; ``topography`` plus the four ``mountain_*`` keys for
        a mountain; ``topography`` for the cone. ``topography_parameters``
        carries the full parameter set.
        """
        if self.preset == "flat":
            return {}
        entries: dict = {"topography": self.preset,
                         "topography_parameters": dict(self.parameters)}
        if self.preset == "mountain":
            for key in _MOUNTAIN_KEYS:
                entries[f"mountain_{key}"] = self.parameters[key]
        return entries


# ---------------------------------------------------------------------------
# Discretization
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DiscretizationSpec:
    """Spectral truncation, grid family and dimensions, product sampling.

    ``grid`` is ``"latlon"`` (the Gauss-Legendre reference grid; the alias
    ``"gauss-latlon"`` is accepted) with ``nlat`` x ``nlon``, or
    ``"geodesic"`` (experimental) with subdivision ``resolution``. The
    unused dimensions keep their defaults and are recorded, as in every
    run configuration. ``product_quadrature`` is ``"fine"`` (default) or
    ``"coarse"`` (BVE only; the shallow-water core always uses ``"fine"``).
    """

    lmax: int
    grid: str = "latlon"
    resolution: int = 4
    nlat: int = 128
    nlon: int = 256
    product_quadrature: str = "fine"

    def __post_init__(self) -> None:
        grid = "latlon" if self.grid == "gauss-latlon" else self.grid
        if grid not in GRID_TYPES:
            raise SpecificationError(
                f"grid must be one of {GRID_TYPES} (or 'gauss-latlon'), got "
                f"{self.grid!r}")
        object.__setattr__(self, "grid", grid)
        object.__setattr__(self, "lmax", _integer("lmax", self.lmax,
                                                  minimum=1))
        object.__setattr__(self, "resolution",
                           _integer("resolution", self.resolution, minimum=0))
        nlat = _integer("nlat", self.nlat, minimum=1)
        nlon = _integer("nlon", self.nlon, minimum=1)
        if grid == "latlon" and (nlat < MIN_NLAT or nlon < MIN_NLON):
            raise SpecificationError(
                f"the lat-lon grid requires nlat >= {MIN_NLAT} and nlon >= "
                f"{MIN_NLON}, got {nlat} x {nlon}")
        object.__setattr__(self, "nlat", nlat)
        object.__setattr__(self, "nlon", nlon)
        if self.product_quadrature not in PRODUCT_QUADRATURES:
            raise SpecificationError(
                f"product_quadrature must be one of {PRODUCT_QUADRATURES}, "
                f"got {self.product_quadrature!r}")

    @classmethod
    def gauss_latlon(cls, lmax: int, nlat: int, nlon: int, *,
                     product_quadrature: str = "fine") -> "DiscretizationSpec":
        return cls(lmax=lmax, grid="latlon", nlat=nlat, nlon=nlon,
                   product_quadrature=product_quadrature)

    @classmethod
    def geodesic(cls, lmax: int, resolution: int, *,
                 product_quadrature: str = "fine") -> "DiscretizationSpec":
        return cls(lmax=lmax, grid="geodesic", resolution=resolution,
                   product_quadrature=product_quadrature)

    def run_config_entries(self) -> dict:
        """The run-configuration keys this discretization fixes."""
        return {"lmax": self.lmax, "grid": self.grid,
                "resolution": self.resolution, "nlat": self.nlat,
                "nlon": self.nlon,
                "product_quadrature": self.product_quadrature}

    def to_dict(self) -> dict:
        return {"version": SPEC_VERSION, **self.run_config_entries()}

    @classmethod
    def from_dict(cls, data: Mapping) -> "DiscretizationSpec":
        data = _check_document(data, "discretization specification",
                               {"lmax", "grid", "resolution", "nlat", "nlon",
                                "product_quadrature"})
        try:
            return cls(lmax=data["lmax"], grid=data["grid"],
                       resolution=data["resolution"], nlat=data["nlat"],
                       nlon=data["nlon"],
                       product_quadrature=data["product_quadrature"])
        except KeyError as err:
            raise SpecificationError(
                f"discretization specification is missing {err}") from None

    def describe(self) -> str:
        if self.grid == "latlon":
            where = f"Gauss lat-lon {self.nlat} x {self.nlon}"
        else:
            where = f"geodesic resolution {self.resolution}"
        return (f"l_max {self.lmax}, {where}, {self.product_quadrature} "
                "product sampling")
