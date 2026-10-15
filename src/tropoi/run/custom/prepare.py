"""Prepare BVE and SWE initial states from user inputs, explicitly.

Physical fields and spectral coefficients are different things and enter
through different, explicit source types; nothing is guessed from an
array's shape:

``SpectralCoefficients(coeffs)``
    Coefficients already in the repository layout: complex, dense
    ``(L+1, L+1)``, axis 0 = degree ``l``, axis 1 = order ``m >= 0``,
    complex orthonormal spherical harmonics of a real field (``Im a_l0 =
    0``, ``m > l`` entries exactly zero). ``L`` must equal the
    discretization's ``lmax``; :func:`pad_coefficients` zero-pads a
    lower-degree array explicitly (lossless). Nothing is ever truncated.
    No transform is involved, so coefficient-only BVE preparation runs on
    the CPU.
``SampledField(values)``
    Real values (SI units) at the discretization's STATE-GRID points, in
    the grid's own order and shape: ``(nlat, nlon)`` for the Gauss lat-lon
    grid (rows north to south, longitudes ``2*pi*j/nlon``), ``(n_points,)``
    for the geodesic grid. :func:`state_sampling` returns those coordinates.
``SampledFunction(function)`` (or a bare callable)
    ``function(lat, lon) -> values``: ``lat`` and ``lon`` are host NumPy
    float64 arrays in RADIANS (latitude in ``[-pi/2, pi/2]``; longitude as
    the grid stores it, so the function must be ``2*pi``-periodic), shaped
    like the state grid; the function must return a real, finite NumPy
    array of exactly that shape. It is evaluated once, on the CPU, never
    on CuPy arrays, and is never stored (only its name, descriptively).

Physical inputs are projected with the backend's own analysis transform
(``sh.transform``, CUDA), exactly as the preset initial conditions are.

Variables (SI units)
--------------------
BVE: ``zeta`` (relative vorticity, s^-1) or ``streamfunction`` (psi,
m^2 s^-1; ``zeta_lm = -l(l+1)/a^2 psi_lm``, exact).

SWE: ``zeta`` (s^-1) or ``streamfunction`` (m^2 s^-1); ``delta``
(divergence, s^-1) or ``velocity_potential`` (chi, m^2 s^-1; ``delta_lm =
-l(l+1)/a^2 chi_lm``); ``phi`` (PERTURBATION thickness geopotential,
m^2 s^-2: total thickness ``h = H + phi/g``) or ``free_surface_anomaly``
(eta', m: the free-surface height minus its mean; ``phi_lm = g eta'_lm -
g h_s'_lm`` with ``h_s'`` the resolved terrain minus its mean, the
construction every SWE preset uses, so ``free_surface_anomaly=0`` is the
exact lake at rest). An omitted SWE variable is exactly zero. ``a`` is the
planet's dynamical radius.

Acceptance policy (:class:`ProjectionPolicy`)
---------------------------------------------
* Projection loses whatever lies outside the truncation (and, on the
  geodesic grid, part of the quadrature error). Each projected input
  reports the quadrature-weighted relative L2 residual between the samples
  and their band-limited synthesis on the same points, plus the maximum
  pointwise residual; preparation FAILS if the relative residual exceeds
  ``max_relative_residual`` (default 1e-2). The field is never smoothed,
  clipped or rescaled to pass.
* Mean modes: ``zeta``, ``delta``, ``phi`` (and ``free_surface_anomaly``)
  must have zero global mean to the cores' relative tolerance (1e-10 of the
  field norm); the BVE ``zeta`` too, because no wind on a sphere has a
  nonzero vorticity integral. A nonzero mean FAILS preparation and reports
  the mean in field units. Means are removed only for the variables the
  caller names in ``subtract_mean``; the removed value is recorded in the
  report. (``streamfunction`` and ``velocity_potential`` means are
  irrelevant: their Laplacian is exactly mean-free.)
* Coefficient conventions (finite, real ``a_l0``, zero padding) are checked
  on every result and never repaired.
* SWE: the reconstructed thickness ``H + phi/g`` must be strictly positive
  on the state AND the nonlinear-product samplings, checked by the model's
  own ``validate_state``; the minimum thickness is reported.
* Spectral support is reported, not changed: the share of each field's
  power above the 2/3 product cut (``product_truncation_cut(lmax)``).
  Stored states keep their full support up to ``lmax``; only analyzed
  nonlinear products are truncated during integration, as always.

A prepared state is VALID, not BALANCED: nothing here adjusts winds and
mass toward balance. Unbalanced states radiate gravity waves; that is the
user's physics.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional

import numpy as np

from tropoi.representation.archive.initial_state import (InitialState,
                                                         InitialStateError)
from tropoi.spatial.modes import (ZERO_MEAN_MONOPOLE_RTOL, FieldSpec,
                                  SpectralConventionError, SpectralModes)
from tropoi.spatial.specification import (DiscretizationSpec, FluidLayerSpec,
                                          PlanetSpec, TopographySpec)
from tropoi.spatial.truncation import product_truncation_cut

__all__ = [
    "FieldReport",
    "PreparationError",
    "PreparationReport",
    "PreparedState",
    "ProjectionPolicy",
    "SampledField",
    "SampledFunction",
    "SpectralCoefficients",
    "StateSampling",
    "pad_coefficients",
    "prepare_bve_state",
    "prepare_swe_state",
    "state_sampling",
]

_SQRT_4PI = math.sqrt(4.0 * math.pi)

#: Units of every accepted input variable.
VARIABLE_UNITS = {
    "zeta": "s^-1",
    "delta": "s^-1",
    "phi": "m^2 s^-2",
    "streamfunction": "m^2 s^-1",
    "velocity_potential": "m^2 s^-1",
    "free_surface_anomaly": "m",
}
#: Inputs whose global mean must vanish (the potentials' means are inert).
_ZERO_MEAN_INPUTS = ("zeta", "delta", "phi", "free_surface_anomaly")


class PreparationError(ValueError):
    """An input cannot become a valid initial state under the policy.

    ``report`` holds whatever was measured before the failure.
    """

    def __init__(self, message: str, report: "PreparationReport | None" = None):
        super().__init__(message)
        self.report = report


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------

def _reject_device_array(name: str, value: Any) -> None:
    if (type(value).__module__ or "").startswith("cupy"):
        raise PreparationError(
            f"{name}: expected a host NumPy array, got a CuPy array; convert "
            "it explicitly with cupy.asnumpy(...)")


@dataclass(frozen=True, eq=False)
class SpectralCoefficients:
    """Coefficients in the repository layout (see the module docstring)."""

    coeffs: np.ndarray

    def __post_init__(self) -> None:
        _reject_device_array("SpectralCoefficients", self.coeffs)
        arr = np.asarray(self.coeffs)
        if not np.iscomplexobj(arr):
            raise PreparationError(
                "SpectralCoefficients must be complex (the m >= 0 complex "
                f"layout), got dtype {arr.dtype}; a physical field must be "
                "passed as SampledField or SampledFunction")
        if arr.ndim != 2 or arr.shape[0] != arr.shape[1] or arr.shape[0] < 1:
            raise PreparationError(
                "SpectralCoefficients must be a square (L+1, L+1) array, got "
                f"shape {arr.shape}")
        if arr.dtype not in (np.complex64, np.complex128):
            raise PreparationError(
                f"unsupported coefficient dtype {arr.dtype}")
        frozen = arr.astype(np.complex128, copy=True)   # exact widening
        frozen.flags.writeable = False
        object.__setattr__(self, "coeffs", frozen)

    @property
    def l_max(self) -> int:
        return self.coeffs.shape[0] - 1


@dataclass(frozen=True, eq=False)
class SampledField:
    """Real values (SI) at the state-grid points, in the grid's shape."""

    values: np.ndarray

    def __post_init__(self) -> None:
        _reject_device_array("SampledField", self.values)
        arr = np.asarray(self.values)
        if np.iscomplexobj(arr) or not np.issubdtype(arr.dtype, np.number) \
                or arr.dtype == bool:
            raise PreparationError(
                "SampledField values must be real numbers, got dtype "
                f"{arr.dtype}; spectral coefficients go in "
                "SpectralCoefficients")
        frozen = arr.astype(np.float64, copy=True)
        frozen.flags.writeable = False
        object.__setattr__(self, "values", frozen)


@dataclass(frozen=True, eq=False)
class SampledFunction:
    """``function(lat, lon) -> values`` on host NumPy arrays (radians)."""

    function: Callable[[np.ndarray, np.ndarray], Any]
    name: str = ""

    def __post_init__(self) -> None:
        if not callable(self.function):
            raise PreparationError("SampledFunction needs a callable")
        if not self.name:
            object.__setattr__(self, "name", getattr(
                self.function, "__qualname__", type(self.function).__name__))


def _as_source(name: str, value: Any):
    if value is None or isinstance(value, (SpectralCoefficients, SampledField,
                                           SampledFunction)):
        return value
    if callable(value):
        return SampledFunction(value)
    raise PreparationError(
        f"{name}: wrap the input explicitly as SpectralCoefficients(...), "
        "SampledField(...) or SampledFunction(...) (or pass a callable); "
        f"got {type(value).__name__}")


def pad_coefficients(coeffs: np.ndarray, l_max: int) -> np.ndarray:
    """Zero-pad an ``(L0+1, L0+1)`` coefficient array to ``(l_max+1, ...)``.

    Lossless and explicit; ``L0 > l_max`` is refused (that would be a
    truncation, which only the caller may decide to make).
    """
    _reject_device_array("pad_coefficients", coeffs)
    arr = np.asarray(coeffs)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise PreparationError(f"expected a square array, got {arr.shape}")
    n0, n = arr.shape[0], int(l_max) + 1
    if n0 > n:
        raise PreparationError(
            f"cannot pad degree {n0 - 1} coefficients to l_max={l_max}: that "
            "is a truncation")
    out = np.zeros((n, n), dtype=np.complex128)
    out[:n0, :n0] = arr
    return out


# ---------------------------------------------------------------------------
# Policy and report
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProjectionPolicy:
    """Explicit acceptance policy (see the module docstring)."""

    max_relative_residual: float = 1e-2
    subtract_mean: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        r = float(self.max_relative_residual)
        if not (math.isfinite(r) and r >= 0.0):
            raise PreparationError(
                f"max_relative_residual must be finite and >= 0, got {r}")
        object.__setattr__(self, "max_relative_residual", r)
        names = tuple(self.subtract_mean)
        unknown = sorted(set(names) - set(_ZERO_MEAN_INPUTS))
        if unknown:
            raise PreparationError(
                f"subtract_mean accepts {list(_ZERO_MEAN_INPUTS)}, got "
                f"{unknown}")
        object.__setattr__(self, "subtract_mean", names)

    def to_dict(self) -> dict:
        return {"max_relative_residual": self.max_relative_residual,
                "subtract_mean": list(self.subtract_mean),
                "monopole_rtol": ZERO_MEAN_MONOPOLE_RTOL}


@dataclass
class FieldReport:
    """What preparation measured for one input variable."""

    name: str
    units: str
    source: str                       # coefficients | samples | function | zero
    function: Optional[str] = None
    relative_residual: Optional[float] = None
    max_abs_residual: Optional[float] = None
    max_abs_value: Optional[float] = None
    mean: Optional[float] = None       # global mean, field units
    subtracted_mean: Optional[float] = None
    power_above_product_cut: Optional[float] = None

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class PreparationReport:
    """Per-field measurements, the policy, and (SWE) the positivity margin."""

    solver: str
    policy: ProjectionPolicy
    l_max: int
    fields: list[FieldReport] = field(default_factory=list)
    min_thickness_m: Optional[float] = None
    max_thickness_m: Optional[float] = None
    topography: Optional[dict] = None

    def to_dict(self) -> dict:
        return {
            "solver": self.solver,
            "l_max": self.l_max,
            "product_truncation_cut": product_truncation_cut(self.l_max),
            "policy": self.policy.to_dict(),
            "fields": [f.to_dict() for f in self.fields],
            "min_thickness_m": self.min_thickness_m,
            "max_thickness_m": self.max_thickness_m,
            "topography": self.topography,
            "balanced": "not assessed: a valid state is not necessarily "
                        "balanced; no balancing was applied",
        }

    def lines(self) -> list[str]:
        out = [f"{self.solver.upper()} preparation, l_max={self.l_max}, "
               f"product cut {product_truncation_cut(self.l_max)}:"]
        for f in self.fields:
            parts = [f"  {f.name:<20} {f.source}"]
            if f.relative_residual is not None:
                parts.append(f"residual {f.relative_residual:.3e}")
            if f.mean is not None:
                parts.append(f"mean {f.mean:.3e} {f.units}")
            if f.subtracted_mean is not None:
                parts.append(f"(subtracted {f.subtracted_mean:.3e})")
            if f.power_above_product_cut is not None:
                parts.append(
                    f"power above cut {f.power_above_product_cut:.2e}")
            out.append(", ".join(parts))
        if self.min_thickness_m is not None:
            out.append(f"  thickness h in [{self.min_thickness_m:.6g}, "
                       f"{self.max_thickness_m:.6g}] m over the state and "
                       "product samplings")
        return out


@dataclass(frozen=True, eq=False)
class PreparedState:
    """The validated :class:`InitialState` and the report that justifies it."""

    state: InitialState
    report: PreparationReport

    def save(self, path, *, overwrite: bool = False):
        return self.state.save(path, overwrite=overwrite)


# ---------------------------------------------------------------------------
# Sampling and model resources (CUDA)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class StateSampling:
    """Host coordinates (radians) and weights of a discretization's state grid."""

    latitudes: np.ndarray
    longitudes: np.ndarray
    weights: np.ndarray       # quadrature solid-angle weights (sum ~ 4 pi)
    shape: tuple[int, ...]
    grid: str


def _host(array) -> np.ndarray:
    if hasattr(array, "get"):
        return np.asarray(array.get())
    return np.asarray(array)


def _base_config(planet: PlanetSpec, discretization: DiscretizationSpec
                 ) -> dict:
    return {**discretization.run_config_entries(),
            "planet": planet.to_dict(), "day_hours": planet.day_hours}


def _build_planet(planet: PlanetSpec, discretization: DiscretizationSpec):
    from tropoi.cli.bve import build_planet

    return build_planet(_base_config(planet, discretization))


def _sampling_of(planet_obj, grid: str) -> StateSampling:
    g = planet_obj.grid
    shape = getattr(g, "grid_shape", None)
    shape = tuple(int(s) for s in shape) if shape else (int(g.n_points),)
    lat = _host(g.point_latitudes).astype(np.float64).reshape(shape)
    lon = _host(g.point_longitudes).astype(np.float64).reshape(shape)
    weights = _host(planet_obj.sh.weights).astype(np.float64).reshape(shape)
    for arr in (lat, lon, weights):
        arr.flags.writeable = False
    return StateSampling(latitudes=lat, longitudes=lon, weights=weights,
                         shape=shape, grid=grid)


def state_sampling(planet: PlanetSpec,
                   discretization: DiscretizationSpec) -> StateSampling:
    """Coordinates of the state grid the inputs are sampled on (CUDA).

    Built from the model's own grid object, so the coordinates are exactly
    the points the projection uses.
    """
    return _sampling_of(_build_planet(planet, discretization),
                        discretization.grid)


class _Resources:
    """Lazily built planet (and sampling) shared by every input."""

    def __init__(self, planet: PlanetSpec, discretization: DiscretizationSpec,
                 planet_obj=None):
        self._planet = planet
        self._discretization = discretization
        self._obj = planet_obj
        self._sampling = None

    @property
    def planet(self):
        if self._obj is None:
            self._obj = _build_planet(self._planet, self._discretization)
        return self._obj

    @property
    def sampling(self) -> StateSampling:
        if self._sampling is None:
            self._sampling = _sampling_of(self.planet,
                                          self._discretization.grid)
        return self._sampling


# ---------------------------------------------------------------------------
# Projection and checks
# ---------------------------------------------------------------------------

def _degree_power_above_cut(coeffs: np.ndarray) -> float:
    n = coeffs.shape[0]
    weight = np.where(np.arange(n)[None, :] == 0, 1.0, 2.0)
    power = (np.abs(coeffs) ** 2 * weight).sum(axis=1)
    total = float(power.sum())
    if total == 0.0:
        return 0.0
    return float(power[product_truncation_cut(n - 1) + 1:].sum() / total)


def _check_conventions(name: str, coeffs: np.ndarray) -> None:
    spec = FieldSpec(name, VARIABLE_UNITS.get(name, ""), "prepared input",
                     monopole="horizontal-mean")
    try:
        SpectralModes(coeffs, spec).validate()
    except SpectralConventionError as err:
        raise PreparationError(str(err)) from err


def _resolve_source(name: str, source, l_max: int, resources: _Resources,
                    policy: ProjectionPolicy, report: PreparationReport
                    ) -> np.ndarray:
    """One input variable -> repository-layout coefficients, with a report."""
    units = VARIABLE_UNITS[name]
    n = l_max + 1
    if source is None:
        report.fields.append(FieldReport(name, units, "zero (omitted)"))
        return np.zeros((n, n), dtype=np.complex128)

    entry = FieldReport(name, units, "")
    report.fields.append(entry)
    if isinstance(source, SpectralCoefficients):
        entry.source = "coefficients"
        if source.l_max != l_max:
            raise PreparationError(
                f"{name}: coefficients have l_max={source.l_max} but the "
                f"discretization has lmax={l_max}; pad explicitly with "
                "pad_coefficients(...) (truncation is never implicit)",
                report)
        coeffs = np.array(source.coeffs)
    else:
        sampling = resources.sampling
        if isinstance(source, SampledFunction):
            entry.source = "function"
            entry.function = source.name
            lat = np.array(sampling.latitudes)
            lon = np.array(sampling.longitudes)
            try:
                raw = source.function(lat, lon)
            except Exception as err:  # noqa: BLE001 - report and re-raise
                raise PreparationError(
                    f"{name}: the function {source.name!r} raised "
                    f"{type(err).__name__}: {err}", report) from err
            _reject_device_array(f"{name} (function result)", raw)
            if not isinstance(raw, np.ndarray):
                raise PreparationError(
                    f"{name}: the function must return a numpy.ndarray, got "
                    f"{type(raw).__name__}", report)
            values = SampledField(raw).values
        else:
            entry.source = "samples"
            values = source.values
        if values.shape != sampling.shape:
            raise PreparationError(
                f"{name}: sampled values have shape {values.shape}, the "
                f"{sampling.grid} state grid has shape {sampling.shape} "
                "(see state_sampling)", report)
        if not np.isfinite(values).all():
            raise PreparationError(f"{name}: sampled values are not all "
                                   "finite", report)
        coeffs, residual, max_abs_res = _project(resources.planet, values)
        entry.relative_residual = residual
        entry.max_abs_residual = max_abs_res
        entry.max_abs_value = float(np.abs(values).max())
        if not residual <= policy.max_relative_residual:
            raise PreparationError(
                f"{name}: relative projection residual {residual:.3e} "
                f"exceeds the policy's max_relative_residual "
                f"{policy.max_relative_residual:g}; the field is not "
                f"representable at lmax={l_max} on this grid (raise lmax, "
                "smooth the input yourself, or relax the policy explicitly)",
                report)

    _check_conventions(name, coeffs)
    entry.power_above_product_cut = _degree_power_above_cut(coeffs)
    if name in _ZERO_MEAN_INPUTS:
        a00 = float(coeffs[0, 0].real)
        entry.mean = a00 / _SQRT_4PI
        scale = float(np.linalg.norm(coeffs))
        if abs(a00) > ZERO_MEAN_MONOPOLE_RTOL * max(scale, 1e-30):
            if name in policy.subtract_mean:
                entry.subtracted_mean = entry.mean
                coeffs[0, 0] = 0.0
            else:
                hint = (" (the mean thickness belongs in "
                        "FluidLayerSpec.mean_depth_m)" if name == "phi" else
                        " (no wind on a sphere has a nonzero vorticity or "
                        "divergence integral)" if name in ("zeta", "delta")
                        else "")
                raise PreparationError(
                    f"{name}: global mean {entry.mean:.6e} {units} is not "
                    f"zero (|a00| = {abs(a00):.3e} > {ZERO_MEAN_MONOPOLE_RTOL:g}"
                    f" x field norm {scale:.3e}){hint}; remove it from the "
                    f"input, or name {name!r} in "
                    "ProjectionPolicy.subtract_mean to subtract it explicitly",
                    report)
    return coeffs


def _project(planet_obj, values: np.ndarray
             ) -> tuple[np.ndarray, float, float]:
    """Analyze on the state grid; measure the synthesis residual there."""
    import cupy as cp

    sh = planet_obj.sh
    flat = cp.asarray(values.reshape(-1), dtype=cp.float64)
    coeffs = cp.asarray(sh.transform(flat), dtype=cp.complex128)
    synth = sh.inv_transform(coeffs).real.reshape(-1)
    w = cp.asarray(sh.weights, dtype=cp.float64).reshape(-1)
    norm = float(cp.sqrt(cp.sum(w * flat * flat)))
    err = float(cp.sqrt(cp.sum(w * (synth - flat) ** 2)))
    max_abs = float(cp.abs(synth - flat).max())
    if norm == 0.0:
        residual = 0.0 if err == 0.0 else math.inf
    else:
        residual = err / norm
    return cp.asnumpy(coeffs), residual, max_abs


def _laplacian(potential_lm: np.ndarray, radius: float) -> np.ndarray:
    """``lap(potential)`` in spectral space: ``-l(l+1)/a^2`` per degree."""
    l = np.arange(potential_lm.shape[0], dtype=np.float64)
    out = (-l * (l + 1.0) / radius ** 2)[:, None] * potential_lm
    out[0, :] = 0.0          # exactly mean-free (avoids -0.0 * a00 noise)
    return out


def _either(name_a: str, a, name_b: str, b) -> tuple[str, Any]:
    if a is not None and b is not None:
        raise PreparationError(
            f"give either {name_a} or {name_b}, not both")
    return (name_b, b) if b is not None else (name_a, a)


def _check_specs(planet, discretization) -> None:
    if not isinstance(planet, PlanetSpec):
        raise PreparationError(
            f"planet must be a PlanetSpec, got {type(planet).__name__}")
    if not isinstance(discretization, DiscretizationSpec):
        raise PreparationError(
            "discretization must be a DiscretizationSpec, got "
            f"{type(discretization).__name__}")


def _finalize(report: PreparationReport, make: Callable[[dict], InitialState]
              ) -> InitialState:
    try:
        return make({"kind": "prepared", "preparation": report.to_dict()})
    except InitialStateError as err:
        raise PreparationError(str(err), report) from err


# ---------------------------------------------------------------------------
# BVE
# ---------------------------------------------------------------------------

def prepare_bve_state(planet: PlanetSpec, discretization: DiscretizationSpec,
                      *, zeta=None, streamfunction=None, label: str = "",
                      policy: Optional[ProjectionPolicy] = None
                      ) -> PreparedState:
    """Build a validated BVE :class:`InitialState` from one explicit input.

    Exactly one of ``zeta`` (s^-1) or ``streamfunction`` (m^2 s^-1). Only
    physical (sampled or function) inputs need CUDA.
    """
    _check_specs(planet, discretization)
    policy = policy or ProjectionPolicy()
    name, source = _either("zeta", zeta, "streamfunction", streamfunction)
    source = _as_source(name, source)
    if source is None:
        raise PreparationError("give zeta or streamfunction")
    l_max = discretization.lmax
    report = PreparationReport("bve", policy, l_max)
    resources = _Resources(planet, discretization)
    coeffs = _resolve_source(name, source, l_max, resources, policy, report)
    if name == "streamfunction":
        coeffs = _laplacian(coeffs, planet.dynamical_radius_m)
        _check_conventions("zeta", coeffs)
    state = _finalize(report, lambda provenance: InitialState(
        "bve", coeffs, planet, discretization, label=label,
        provenance=provenance))
    return PreparedState(state, report)


# ---------------------------------------------------------------------------
# SWE
# ---------------------------------------------------------------------------

def _resolve_topography(planet_obj, spec: TopographySpec):
    """The model's Topography for a specification (CUDA), or None if flat."""
    from tropoi.spatial.terrain.topography import Topography

    if spec.is_flat:
        return None
    p = spec.parameters
    if spec.preset == "mountain":
        return Topography.mountain(planet_obj, height_m=p["height_m"],
                                   lat_deg=p["lat_deg"], lon_deg=p["lon_deg"],
                                   width_deg=p["width_deg"])
    return Topography.williamson5_cone(planet_obj, height_m=p["height_m"])


def prepare_swe_state(planet: PlanetSpec, discretization: DiscretizationSpec,
                      fluid: FluidLayerSpec,
                      topography: Optional[TopographySpec] = None, *,
                      zeta=None, streamfunction=None,
                      delta=None, velocity_potential=None,
                      phi=None, free_surface_anomaly=None,
                      label: str = "",
                      policy: Optional[ProjectionPolicy] = None
                      ) -> PreparedState:
    """Build a validated SWE :class:`InitialState` (requires CUDA).

    At most one input per prognostic: ``zeta`` | ``streamfunction``,
    ``delta`` | ``velocity_potential``, ``phi`` | ``free_surface_anomaly``
    (see the module docstring); omitted ones are exactly zero. The terrain
    specification is resolved on this discretization and stored with the
    state; positivity is checked with the model's own ``validate_state``.
    """
    _check_specs(planet, discretization)
    if not isinstance(fluid, FluidLayerSpec):
        raise PreparationError(
            "fluid must be a FluidLayerSpec (explicit gravity and mean "
            "thickness)")
    if discretization.product_quadrature != "fine":
        raise PreparationError(
            "the shallow-water core always uses product_quadrature='fine'")
    topography = topography or TopographySpec.flat()
    if not isinstance(topography, TopographySpec):
        raise PreparationError(
            f"topography must be a TopographySpec, got "
            f"{type(topography).__name__}")
    policy = policy or ProjectionPolicy()
    rot = _either("zeta", zeta, "streamfunction", streamfunction)
    div = _either("delta", delta, "velocity_potential", velocity_potential)
    mass = _either("phi", phi, "free_surface_anomaly", free_surface_anomaly)
    sources = [(n, _as_source(n, s)) for n, s in (rot, div, mass)]
    for name, source in sources:
        if isinstance(source, SpectralCoefficients) \
                and source.l_max != discretization.lmax:
            raise PreparationError(
                f"{name}: coefficients have l_max={source.l_max} but the "
                f"discretization has lmax={discretization.lmax}; pad "
                "explicitly with pad_coefficients(...)")

    # Everything below needs the model's transforms (CUDA).
    import cupy as cp

    from tropoi.spatial.states.shallow_water import (ShallowWaterState,
                                                     ShallowWaterStateError)
    from tropoi.temporal.tendencies.shallow_water import ShallowWaterModel

    l_max = discretization.lmax
    report = PreparationReport("swe", policy, l_max)
    resources = _Resources(planet, discretization)
    planet_obj = resources.planet
    terrain = _resolve_topography(planet_obj, topography)
    topography_lm = None
    if terrain is not None:
        topography_lm = cp.asnumpy(terrain.elevation_lm)
        report.topography = {
            "preset": topography.preset,
            "measured": {k: float(v) for k, v in terrain.parameters.items()
                         if isinstance(v, (int, float))
                         and math.isfinite(float(v))}}
    radius = planet.dynamical_radius_m
    g = fluid.gravity_m_s2

    stack = []
    for name, source in sources:
        coeffs = _resolve_source(name, source, l_max, resources, policy,
                                 report)
        if name in ("streamfunction", "velocity_potential"):
            coeffs = _laplacian(coeffs, radius)
        elif name == "free_surface_anomaly":
            # phi = g * eta' - g * h_s' : the presets' terrain-aware form.
            coeffs = g * coeffs
            if topography_lm is not None:
                hs_anom = np.array(topography_lm)
                hs_anom[0, 0] = 0.0
                coeffs = coeffs - g * hs_anom
        stack.append(coeffs)
    packed = np.stack(stack).astype(np.complex128)

    model = ShallowWaterModel(planet_obj, gravity=g,
                              mean_depth=fluid.mean_depth_m,
                              topography=terrain)
    sw_state = ShallowWaterState(cp.asarray(packed))
    lo, hi = model.total_geopotential_extrema(sw_state)
    report.min_thickness_m = lo / g
    report.max_thickness_m = hi / g
    try:
        model.validate_state(sw_state, context="in the prepared state")
    except ShallowWaterStateError as err:
        raise PreparationError(str(err), report) from err

    state = _finalize(report, lambda provenance: InitialState(
        "swe", packed, planet, discretization, fluid=fluid,
        topography=topography, topography_lm=topography_lm, label=label,
        provenance=provenance))
    return PreparedState(state, report)
