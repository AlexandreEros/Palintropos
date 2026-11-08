"""Fields of a saved PE state on a fixed-pressure (isobaric) or fixed
geopotential-height surface (CPU).

A derived display surface: the native full-level fields of
:func:`~tropoi.representation.diagnostics.pe_levels.reconstruct_levels`
interpolated in the vertical to one pressure ``p0`` or one geopotential
height ``Z0``. The solver never produces such a surface; this module only
reads saved states. Rules for pressure surfaces:

* **Vertical coordinate.** ``x = ln sigma``. The target at each node is
  ``sigma* = p0 / p_s`` with ``p_s = exp(ln p_s)`` of the saved state. The
  node coordinates ``ln sigma_k`` are the same at every column, so the
  bracket depends on ``p_s`` only through ``sigma*``.
* **Where a full level sits** (``level_sigma``). ``"hydrostatic"`` (default)
  places level k at ``sigma^h_k = sigma_{k+1/2} exp(-alpha_k)``
  (:attr:`~tropoi.spatial.sigma_coordinate.SigmaGrid.hydrostatic_full_levels`):
  the pressure at which the Simmons–Burridge ``Phi_k`` IS the hydrostatic
  geopotential of an isothermal layer, so an isothermal column interpolates
  to the closed form ``Phi_s + R_d T ln(p_s / p0)`` to round-off at every
  valid pressure (tested). ``"nominal"`` places it at ``sigma_k``, the
  pressure the native-level page labels the level with (``p_k = sigma_k
  p_s``); on an isothermal column that attachment carries a height error of
  ``R_d T (alpha_k - ln(sigma_{k+1/2}/sigma_k)) / g`` in the interior
  (zero for the top layer), measured by the tests and reported in the
  documentation. Both are offered so the difference can be seen; the
  default is the one consistent with the solver's hydrostatic scheme.
* **Interpolant.** Linear in ``ln sigma`` between the two bracketing
  levels, applied to temperature, geopotential and the wind components.
  Geopotential height is ``Phi / g`` of the interpolated geopotential.
* **Valid bracket.** ``sigma_1 <= sigma* <= sigma_K`` (first to last level
  position), the ends widened by :data:`END_TOLERANCE` (1e-9, relative) so
  that round-off in ``p0 / p_s`` cannot push a target placed exactly on an
  end level out of range. A target equal to a level position reproduces
  that level's values to round-off. There is no extrapolation: a target in
  the tolerance band takes the end level's values.
* **Masks** (``uint8`` per node): :data:`MASK_VALID` 0; :data:`MASK_BELOW_SURFACE`
  1 (``p0 > p_s``, the surface is underground); :data:`MASK_ABOVE_TOP_LEVEL`
  2 (``sigma* < sigma_1``); :data:`MASK_BELOW_BOTTOM_LEVEL` 3 (``sigma_K <
  sigma* <= 1``: above ground but below the last full level). Codes 2 and 3
  are "outside the sampled full-level range", distinct from 1.
* **Invalid geometry.** A non-finite reconstruction, or a column whose
  ``Phi_k`` does not decrease downward, raises ``ValueError``. A saved
  state with no valid node raises :class:`SurfaceUnavailableError`.

Rules for geopotential-height surfaces (``("geopotential_height", Z0)``,
metres, any sign: the datum is that of the terrain and of the native
reconstruction, ``Phi = g Z`` with the model's own constant ``g``; no other
gravity law, no rotational potential, no new coordinate):

* **Target.** ``Phi0 = g Z0``, the same at every column.
* **Inversion.** Each column's ``Phi_k`` is piecewise linear in
  ``ln sigma`` between the level positions (the pressure rule above, read
  the other way). With ``k`` the upper bracket (``Phi_k >= Phi0 >=
  Phi_{k+1}``), ``w = (Phi0 - Phi_k) / (Phi_{k+1} - Phi_k)`` and
  ``ln sigma* = ln sigma_k + w (ln sigma_{k+1} - ln sigma_k)`` (that is
  ``(1 - w) ln sigma_k + w ln sigma_{k+1}``); temperature and both wind
  components take the same bracket and weight; ``p = p_s exp(ln sigma*)``
  and the wind speed follows from the interpolated components. Where
  ``Phi0`` equals a level's ``Phi_k`` the level is reproduced (``w = 0``).
* **Valid bracket.** ``Phi_K <= Phi0 <= Phi_1`` (bottom to top full
  level), each end widened by :data:`END_TOLERANCE` relative to that end's
  ``|Phi|`` (the round-off of ``g Z0`` is relative too, so the band holds
  at any height, near zero included); the weight is clamped there, no
  extrapolation.
* **Masks** (:data:`HEIGHT_MASK_LABELS`, the same codes): 1 below the
  terrain (``Phi0 < Phi_s = g h_s``, the physical surface geopotential,
  never the exaggerated display), 2 above the highest sampled full level,
  3 between the terrain and the lowest full level (``Phi_s <= Phi0 <
  Phi_K``).
* **Column geometry.** The columns must be finite, ``Phi_k`` strictly
  decreasing downward and ``Phi_K > Phi_s``; otherwise ``ValueError``, so a
  zero or inverted interval is never divided by. The check runs on every
  call, so it also covers stored (quantized) columns.

Vorticity and divergence are deliberately absent on both coordinates: a
horizontal derivative on a sigma level is not the derivative on a pressure
or height surface, and the honest target-coordinate derivative is a later
slice of issue #13.

Two steps, usable apart: :func:`pressure_columns` reconstructs and checks
the native columns of one state ONCE (:class:`PressureColumns`), and
:func:`interpolate_columns` is the pure vertical step for any pressure or
height. :func:`pressure_surface` is the two in sequence (for either
coordinate; the name predates heights). The interactive page of
:mod:`~tropoi.representation.visual.atmosphere` exports the columns and
performs the vertical step itself, in the browser, by these rules; this
module is its reference and the tests compare the two.

NumPy only; never imports CuPy, never touches the page.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from tropoi.representation.diagnostics.pe_levels import (LevelFields,
                                                         reconstruct_levels)

__all__ = [
    "GeopotentialHeightSurface",
    "HEIGHT_MASK_LABELS",
    "HEIGHT_SURFACE_FIELDS",
    "MASK_ABOVE_TOP_LEVEL",
    "MASK_BELOW_BOTTOM_LEVEL",
    "MASK_BELOW_SURFACE",
    "MASK_LABELS",
    "MASK_VALID",
    "PressureColumns",
    "PressureSurface",
    "SURFACE_FIELDS",
    "SurfaceFields",
    "SurfaceUnavailableError",
    "LEVEL_SIGMA_CHOICES",
    "interpolate_columns",
    "interpolate_geopotential",
    "interpolate_log_sigma",
    "level_sigma_positions",
    "parse_surface",
    "pressure_columns",
    "pressure_surface",
]

MASK_VALID = 0
MASK_BELOW_SURFACE = 1
MASK_ABOVE_TOP_LEVEL = 2
MASK_BELOW_BOTTOM_LEVEL = 3

#: Mask code -> what it means (the page shows these words).
MASK_LABELS = {
    MASK_VALID: "valid",
    MASK_BELOW_SURFACE: "below the surface (p0 > ps)",
    MASK_ABOVE_TOP_LEVEL: "above the top full level",
    MASK_BELOW_BOTTOM_LEVEL: "below the bottom full level, above the surface",
}

#: The same codes on a geopotential-height surface (module doc).
HEIGHT_MASK_LABELS = {
    MASK_VALID: "valid",
    MASK_BELOW_SURFACE: "below the terrain (Z0 < hs)",
    MASK_ABOVE_TOP_LEVEL: "above the highest sampled full level",
    MASK_BELOW_BOTTOM_LEVEL: "between the terrain and the lowest full level",
}

#: Interpolated outputs of :func:`pressure_surface` (``u``/``v`` need winds).
SURFACE_FIELDS = ("temperature", "geopotential", "geopotential_height",
                  "u", "v")

#: Outputs on a geopotential-height surface: ``log_sigma`` is ``ln
#: sigma*``, ``pressure`` ``p_s exp(ln sigma*)``, ``geopotential_height``
#: the constant ``Z0``.
HEIGHT_SURFACE_FIELDS = ("temperature", "pressure", "log_sigma",
                         "geopotential_height", "u", "v")

#: Where a full level sits in ``ln sigma`` (module doc).
LEVEL_SIGMA_CHOICES = ("hydrostatic", "nominal")

#: Relative width by which a target just outside the first or last level
#: still counts as on it (round-off of ``p0 / p_s``); no extrapolation
#: happens there, the weight is clamped to the end level.
END_TOLERANCE = 1.0e-9


def level_sigma_positions(context, level_sigma: str = "hydrostatic"
                          ) -> tuple[float, ...]:
    """The K level positions in sigma for ``level_sigma`` (module doc)."""
    if level_sigma == "hydrostatic":
        return tuple(context.sigma.hydrostatic_full_levels)
    if level_sigma == "nominal":
        return tuple(context.full_levels)
    raise ValueError(f"level_sigma must be one of {LEVEL_SIGMA_CHOICES}, "
                     f"got {level_sigma!r}")


class SurfaceUnavailableError(ValueError):
    """No node of a saved state samples the requested surface."""


@dataclass(frozen=True)
class PressureSurface:
    """A fixed-pressure surface, in pascals."""

    pressure_pa: float

    def __post_init__(self) -> None:
        p = self.pressure_pa
        if not _is_number(p):
            raise TypeError(f"pressure must be a number in Pa, got {p!r}")
        p = float(p)
        if not (math.isfinite(p) and p > 0.0):
            raise ValueError(f"pressure must be finite and > 0 Pa, got {p}")
        object.__setattr__(self, "pressure_pa", p)

    kind = "pressure"
    units = "Pa"

    @property
    def value(self) -> float:
        return self.pressure_pa

    @property
    def label(self) -> str:
        """``"500 hPa"`` or, under 100 Pa, ``"61 Pa"``."""
        p = self.pressure_pa
        if p >= 100.0:
            return f"{p / 100.0:.6g} hPa"
        return f"{p:.6g} Pa"


def _is_number(x) -> bool:
    return not isinstance(x, bool) and isinstance(
        x, (int, float, np.integer, np.floating))


@dataclass(frozen=True)
class GeopotentialHeightSurface:
    """A fixed geopotential-height surface ``Z0 = Phi0 / g``, in metres.

    Any finite value: zero is the datum of the terrain and of the native
    reconstruction, and negative heights are valid wherever the terrain
    lies below it.
    """

    height_m: float

    kind = "geopotential_height"
    units = "m"

    def __post_init__(self) -> None:
        z = self.height_m
        if not _is_number(z):
            raise TypeError(f"geopotential height must be a number in m, "
                            f"got {z!r}")
        z = float(z)
        if not math.isfinite(z):
            raise ValueError(f"geopotential height must be finite, got {z}")
        object.__setattr__(self, "height_m", z)

    @property
    def value(self) -> float:
        return self.height_m

    @property
    def label(self) -> str:
        """``"5.5 km"``, or under 1 km in magnitude ``"-250 m"``."""
        z = self.height_m
        if abs(z) >= 1000.0:
            return f"{z / 1000.0:.6g} km"
        return f"{z:.6g} m"


_SPEC_HELP = ("('pressure', value_in_Pa), e.g. ('pressure', 50000.0) for 500 "
              "hPa, or ('geopotential_height', value_in_m), e.g. "
              "('geopotential_height', 5500.0)")


def parse_surface(spec) -> PressureSurface | GeopotentialHeightSurface:
    """``("pressure", Pa)``, ``("geopotential_height", m)`` or one of the
    two surface classes."""
    if isinstance(spec, (PressureSurface, GeopotentialHeightSurface)):
        return spec
    if isinstance(spec, (str, bytes)) or not hasattr(spec, "__len__") \
            or len(spec) != 2:
        raise TypeError(f"surface must be {_SPEC_HELP}; got {spec!r}")
    kind, value = spec
    if kind == "pressure":
        return PressureSurface(value)
    if kind == "geopotential_height":
        return GeopotentialHeightSurface(value)
    raise ValueError(f"unknown surface coordinate {kind!r}; use {_SPEC_HELP}")


@dataclass(frozen=True, eq=False)
class SurfaceFields:
    """One saved state on one derived surface; arrays are ``(nlat, nlon)``.

    ``mask`` holds the codes above; ``bracket`` is the index of the full
    level just above the target (``k`` with ``sigma_k <= sigma* <=
    sigma_{k+1}``), ``-1`` where masked; ``sigma_target`` is ``p0 / p_s``
    on an isobar and ``exp(ln sigma*)`` on a height surface.
    ``values[name]`` is finite
    everywhere; where masked it holds the value of the nearest valid
    bracket end for the top/bottom codes and is meaningless (0.0) under
    ground. Read it only through ``valid``.
    """

    surface: PressureSurface | GeopotentialHeightSurface
    level_sigma: str
    level_positions: tuple[float, ...]
    lat_deg: np.ndarray
    lon_deg: np.ndarray
    values: dict
    mask: np.ndarray
    bracket: np.ndarray
    sigma_target: np.ndarray
    surface_pressure: np.ndarray
    surface_height: np.ndarray | None

    def __getitem__(self, name: str) -> np.ndarray:
        return self.values[name]

    @property
    def valid(self) -> np.ndarray:
        return self.mask == MASK_VALID

    def counts(self) -> dict:
        """Node count per mask code (every code, zeros included)."""
        return {code: int(np.count_nonzero(self.mask == code))
                for code in MASK_LABELS}


def interpolate_log_sigma(sigma_full, columns: dict, sigma_target):
    """Linear interpolation in ``ln sigma`` of level stacks to a target.

    ``sigma_full`` are the K full-level sigmas (strictly increasing, in
    ``(0, 1]``); each ``columns[name]`` is ``(K, *shape)``; ``sigma_target``
    has ``shape`` (``p0 / p_s``). Returns ``(values, mask, bracket)``: a dict
    of ``shape`` arrays, the ``uint8`` mask codes of this module and the
    upper bracket index (``-1`` where masked). Pure NumPy; no physics.
    """
    sig = np.asarray(sigma_full, dtype=np.float64)
    if sig.ndim != 1 or sig.size < 2:
        raise ValueError("need at least two full levels to bracket a target")
    if not (np.all(np.isfinite(sig)) and np.all(sig > 0.0)
            and np.all(sig <= 1.0) and np.all(np.diff(sig) > 0.0)):
        raise ValueError(f"full-level sigmas must be strictly increasing in "
                         f"(0, 1], got {sig}")
    target = np.asarray(sigma_target, dtype=np.float64)
    if not np.all(np.isfinite(target)):
        raise ValueError("sigma* = p0 / p_s is not finite everywhere")
    k_count = sig.size
    for name, stack in columns.items():
        stack = np.asarray(stack)
        if stack.shape != (k_count, *target.shape):
            raise ValueError(f"column {name!r} has shape {stack.shape}; "
                             f"expected {(k_count, *target.shape)}")
        if not np.all(np.isfinite(stack)):
            raise ValueError(f"column {name!r} is not finite")

    # A target within END_TOLERANCE (relative) of the first or last level
    # counts as on it: p0 / p_s for p0 = sigma_K p_s is not exactly sigma_K
    # in floating point, and the native-level coincidence must hold there.
    top = sig[0] * (1.0 - END_TOLERANCE)
    bottom = sig[-1] * (1.0 + END_TOLERANCE)
    mask = np.full(target.shape, MASK_VALID, dtype=np.uint8)
    mask[target > 1.0] = MASK_BELOW_SURFACE
    mask[target < top] = MASK_ABOVE_TOP_LEVEL
    mask[(target > bottom) & (target <= 1.0)] = MASK_BELOW_BOTTOM_LEVEL
    valid = mask == MASK_VALID

    # Upper bracket index: sigma_k <= sigma* <= sigma_{k+1}, k in [0, K-2].
    # searchsorted(side="right") - 1 gives k with sigma_k <= target; clamp so
    # a target exactly on the last level uses the last bracket with weight 1.
    log_sig = np.log(sig)
    k = np.searchsorted(sig, target, side="right") - 1
    k = np.clip(k, 0, k_count - 2)
    bracket = np.where(valid, k, -1).astype(np.int64)
    safe_target = np.where(valid, target, sig[0])
    k_safe = np.where(valid, k, 0)
    x0 = log_sig[k_safe]
    x1 = log_sig[k_safe + 1]
    w = (np.log(safe_target) - x0) / (x1 - x0)
    w = np.clip(w, 0.0, 1.0)

    values = {}
    for name, stack in columns.items():
        stack = np.asarray(stack, dtype=np.float64)
        lower = np.take_along_axis(stack, k_safe[None], axis=0)[0]
        upper = np.take_along_axis(stack, (k_safe + 1)[None], axis=0)[0]
        out = lower + w * (upper - lower)
        # Masked nodes: a finite placeholder, never a sample. Above the top
        # level: the top value; below the bottom level: the bottom value;
        # under ground: 0.
        out = np.where(mask == MASK_ABOVE_TOP_LEVEL, stack[0], out)
        out = np.where(mask == MASK_BELOW_BOTTOM_LEVEL, stack[-1], out)
        out = np.where(mask == MASK_BELOW_SURFACE, 0.0, out)
        values[name] = out
    return values, mask, bracket


def interpolate_geopotential(sigma_full, geopotential, columns: dict,
                             phi_target: float, phi_surface):
    """Invert each column's piecewise-linear ``Phi(ln sigma)`` at ``Phi0``.

    ``sigma_full`` are the K level positions (strictly increasing in (0,
    1]); ``geopotential`` is ``(K, *shape)`` (``Phi_k``, top first);
    ``columns[name]`` are ``(K, *shape)`` stacks interpolated with the same
    bracket and weight; ``phi_target`` is ``Phi0 = g Z0``; ``phi_surface``
    has ``shape`` (``Phi_s = g h_s``). Returns ``(values, mask, bracket)``
    as :func:`interpolate_log_sigma`, ``values`` also holding
    ``log_sigma`` (``ln sigma*``). Raises ``ValueError`` on invalid column
    geometry (module doc). Pure NumPy; no physics.
    """
    sig = np.asarray(sigma_full, dtype=np.float64)
    if sig.ndim != 1 or sig.size < 2:
        raise ValueError("need at least two full levels to bracket a target")
    if not (np.all(np.isfinite(sig)) and np.all(sig > 0.0)
            and np.all(sig <= 1.0) and np.all(np.diff(sig) > 0.0)):
        raise ValueError(f"full-level sigmas must be strictly increasing in "
                         f"(0, 1], got {sig}")
    phi0 = float(phi_target)
    if not math.isfinite(phi0):
        raise ValueError(f"target geopotential must be finite, got {phi0}")
    phi = np.asarray(geopotential, dtype=np.float64)
    k_count = sig.size
    if phi.ndim < 1 or phi.shape[0] != k_count:
        raise ValueError(f"geopotential has shape {phi.shape}; expected "
                         f"({k_count}, ...)")
    shape = phi.shape[1:]
    phi_s = np.broadcast_to(np.asarray(phi_surface, dtype=np.float64), shape)
    if not (np.all(np.isfinite(phi)) and np.all(np.isfinite(phi_s))):
        raise ValueError("column geopotential is not finite")
    inverted = ~np.all(phi[:-1] > phi[1:], axis=0)
    grounded = ~(phi[-1] > phi_s)
    if inverted.any() or grounded.any():
        raise ValueError(
            f"invalid column geometry at {int((inverted | grounded).sum())} "
            f"of {inverted.size} nodes: the full-level geopotential must "
            "decrease strictly downward and the lowest full level must lie "
            "above the terrain (a zero or inverted interval cannot be "
            "inverted); nothing is interpolated")
    for name, stack in columns.items():
        stack = np.asarray(stack)
        if stack.shape != phi.shape:
            raise ValueError(f"column {name!r} has shape {stack.shape}; "
                             f"expected {phi.shape}")
        if not np.all(np.isfinite(stack)):
            raise ValueError(f"column {name!r} is not finite")

    top, bottom = phi[0], phi[-1]
    mask = np.full(shape, MASK_VALID, dtype=np.uint8)
    mask[phi0 > top + END_TOLERANCE * np.abs(top)] = MASK_ABOVE_TOP_LEVEL
    mask[phi0 < bottom - END_TOLERANCE * np.abs(bottom)] = \
        MASK_BELOW_BOTTOM_LEVEL
    mask[phi0 < phi_s] = MASK_BELOW_SURFACE
    valid = mask == MASK_VALID

    # Upper bracket: Phi_k >= Phi0 >= Phi_{k+1}, k in [0, K-2]; the count of
    # levels at or above the target, less one, clamped (a target on the
    # last level uses the last bracket with weight 1).
    k = np.count_nonzero(phi >= phi0, axis=0) - 1
    k = np.clip(k, 0, k_count - 2)
    k_safe = np.where(valid, k, 0)
    bracket = np.where(valid, k, -1).astype(np.int64)
    p0 = np.take_along_axis(phi, k_safe[None], axis=0)[0]
    p1 = np.take_along_axis(phi, (k_safe + 1)[None], axis=0)[0]
    w = (phi0 - p0) / (p1 - p0)          # p1 < p0 everywhere (checked)
    w = np.clip(w, 0.0, 1.0)

    log_sig = np.log(sig)
    stacks = dict(columns)
    stacks["log_sigma"] = np.broadcast_to(
        log_sig.reshape((k_count,) + (1,) * len(shape)), phi.shape)
    values = {}
    for name, stack in stacks.items():
        stack = np.asarray(stack, dtype=np.float64)
        lower = np.take_along_axis(stack, k_safe[None], axis=0)[0]
        upper = np.take_along_axis(stack, (k_safe + 1)[None], axis=0)[0]
        out = lower + w * (upper - lower)
        # masked placeholders as interpolate_log_sigma: the end level's
        # value beyond an end level, 0 under ground; never a sample
        out = np.where(mask == MASK_ABOVE_TOP_LEVEL, stack[0], out)
        out = np.where(mask == MASK_BELOW_BOTTOM_LEVEL, stack[-1], out)
        out = np.where(mask == MASK_BELOW_SURFACE, 0.0, out)
        values[name] = out
    return values, mask, bracket


@dataclass(frozen=True, eq=False)
class PressureColumns:
    """The native columns of one saved state on the display mesh: what the
    vertical interpolation reads and what the interactive page exports.

    Stacks are ``(K, nlat, nlon)``: ``temperature`` (K), ``geopotential``
    (m² s⁻², absolute), ``u`` and ``v`` (m/s; ``None`` without winds).
    ``surface_pressure`` (Pa) is ``exp(ln p_s)`` of the state. The column
    geometry has been checked (finite, ``Phi`` decreasing downward).
    """

    level_sigma: str
    level_positions: tuple[float, ...]
    lat_deg: np.ndarray
    lon_deg: np.ndarray
    temperature: np.ndarray
    geopotential: np.ndarray
    u: np.ndarray | None
    v: np.ndarray | None
    surface_pressure: np.ndarray
    surface_height: np.ndarray | None
    gravity: float

    @property
    def nlev(self) -> int:
        return int(self.temperature.shape[0])

    @property
    def winds(self) -> bool:
        return self.u is not None

    def stacks(self) -> dict:
        """The interpolated fields by name (winds when present)."""
        out = {"temperature": self.temperature,
               "geopotential": self.geopotential}
        if self.winds:
            out["u"] = self.u
            out["v"] = self.v
        return out

    def sampled_pressure_range(self) -> tuple[float, float]:
        """Pressures (Pa) the full levels span over the mesh: ``sigma_1 min
        p_s`` to ``sigma_K max p_s``. A target outside it has no valid node."""
        p_s = self.surface_pressure
        return (float(self.level_positions[0] * p_s.min()),
                float(self.level_positions[-1] * p_s.max()))

    def surface_geopotential(self) -> np.ndarray:
        """``Phi_s = g h_s`` on the mesh (zero for a flat lower boundary)."""
        if self.surface_height is None:
            return np.zeros(self.surface_pressure.shape)
        return self.gravity * np.asarray(self.surface_height,
                                         dtype=np.float64)

    def sampled_height_range(self) -> tuple[float, float]:
        """Geopotential heights (m) the full levels span over the mesh: the
        lowest bottom full level to the highest top one. A target outside
        it has no valid node."""
        phi = self.geopotential
        return (float(phi[-1].min() / self.gravity),
                float(phi[0].max() / self.gravity))


def _check_column_geometry(levels: LevelFields) -> None:
    phi = levels["geopotential"]
    if phi.shape[0] < 2:
        raise ValueError("a pressure surface needs at least two full levels")
    if not np.all(np.isfinite(phi)):
        raise ValueError("reconstructed geopotential is not finite")
    bad = ~np.all(phi[:-1] > phi[1:], axis=0)
    if bad.any():
        raise ValueError(
            f"non-monotonic column geometry: the full-level geopotential "
            f"does not decrease downward at {int(bad.sum())} of {bad.size} "
            "display nodes (a negative or non-finite temperature); nothing "
            "is interpolated")


def pressure_columns(frame, context, lat_deg, lon_deg, *, winds: bool = True,
                     levels: LevelFields | None = None,
                     level_sigma: str = "hydrostatic") -> PressureColumns:
    """The native columns of one saved state, ready to interpolate.

    Reconstructs every full level on the display mesh with
    :func:`reconstruct_levels` (winds when ``winds``; never vorticity or
    divergence), or takes ``levels`` when the caller already has them (they
    must hold every full level of ``context`` on this mesh). Checks the
    column geometry and the surface pressure. ``level_sigma`` chooses where
    a level sits (module doc). This is the one reconstruction a saved state
    needs, whatever pressures are later asked of it.
    """
    positions = level_sigma_positions(context, level_sigma)
    if context.gravity is None:
        raise ValueError("the column context records no gravity, so "
                         "geopotential height Phi/g is undefined")
    if levels is None:
        levels = reconstruct_levels(frame, context, lat_deg, lon_deg, None,
                                    winds=winds, vorticity_divergence=False)
    if levels.levels != tuple(range(context.nlev)):
        raise ValueError("pressure_columns needs every full level of the "
                         f"run, got levels {levels.levels}")
    if winds and "u" not in levels.values:
        raise ValueError("winds were requested but the given levels hold "
                         "no u/v")
    _check_column_geometry(levels)
    p_s = levels.surface_pressure
    if not (np.all(np.isfinite(p_s)) and np.all(p_s > 0.0)):
        raise ValueError("surface pressure is not finite and positive")
    return PressureColumns(
        level_sigma=level_sigma, level_positions=positions,
        lat_deg=levels.lat_deg, lon_deg=levels.lon_deg,
        temperature=np.asarray(levels["temperature"], dtype=np.float64),
        geopotential=np.asarray(levels["geopotential"], dtype=np.float64),
        u=np.asarray(levels["u"], dtype=np.float64) if winds else None,
        v=np.asarray(levels["v"], dtype=np.float64) if winds else None,
        surface_pressure=np.asarray(p_s, dtype=np.float64),
        surface_height=levels.surface_height, gravity=float(context.gravity))


def interpolate_columns(columns: PressureColumns, surface, *,
                        time_label: str = "this saved state"
                        ) -> SurfaceFields:
    """Interpolate prepared columns to the derived surface ``surface``.

    The pure vertical step (module doc). Pressure: ``sigma* = p0 / p_s``,
    linear in ``ln sigma`` for every stack of ``columns``, ``Z = Phi / g``
    after. Geopotential height: the inversion of :func:`interpolate_
    geopotential` at ``Phi0 = g Z0``, giving temperature, ``u``, ``v``,
    ``ln sigma*`` and ``pressure``; the column geometry is re-checked, so
    stored columns are covered. Raises :class:`SurfaceUnavailableError`
    when no node is valid. The browser's interactive page implements
    exactly this step on the exported columns; this function is its
    reference.
    """
    target = parse_surface(surface)
    if target.kind == "geopotential_height":
        return _height_fields(columns, target, time_label)
    positions = columns.level_positions
    p_s = columns.surface_pressure
    sigma_target = target.pressure_pa / p_s
    values, mask, bracket = interpolate_log_sigma(positions, columns.stacks(),
                                                  sigma_target)
    if not np.any(mask == MASK_VALID):
        lo, hi = columns.sampled_pressure_range()
        raise SurfaceUnavailableError(
            f"the {target.label} surface has no valid node at {time_label}: "
            f"the saved surface pressure spans {p_s.min():.6g} to "
            f"{p_s.max():.6g} Pa and the full levels sample {lo:.6g} to "
            f"{hi:.6g} Pa; choose a pressure inside that range")
    values["geopotential_height"] = values["geopotential"] / columns.gravity
    return SurfaceFields(surface=target, level_sigma=columns.level_sigma,
                         level_positions=positions, lat_deg=columns.lat_deg,
                         lon_deg=columns.lon_deg, values=values, mask=mask,
                         bracket=bracket, sigma_target=sigma_target,
                         surface_pressure=p_s,
                         surface_height=columns.surface_height)


def _height_fields(columns: PressureColumns, target, time_label
                   ) -> SurfaceFields:
    g = columns.gravity
    phi_s = columns.surface_geopotential()
    stacks = {"temperature": columns.temperature}
    if columns.winds:
        stacks["u"], stacks["v"] = columns.u, columns.v
    values, mask, bracket = interpolate_geopotential(
        columns.level_positions, columns.geopotential, stacks,
        g * target.height_m, phi_s)
    if not np.any(mask == MASK_VALID):
        lo, hi = columns.sampled_height_range()
        h = phi_s / g
        raise SurfaceUnavailableError(
            f"the {target.label} geopotential-height surface has no valid "
            f"node at {time_label}: the terrain spans {h.min():.6g} to "
            f"{h.max():.6g} m and the full levels sample {lo:.6g} to "
            f"{hi:.6g} m; choose a height inside that range")
    p_s = columns.surface_pressure
    sigma_target = np.exp(values["log_sigma"])
    values["pressure"] = p_s * sigma_target
    values["pressure"][mask == MASK_BELOW_SURFACE] = 0.0
    values["geopotential_height"] = np.full(mask.shape, target.height_m)
    return SurfaceFields(surface=target, level_sigma=columns.level_sigma,
                         level_positions=columns.level_positions,
                         lat_deg=columns.lat_deg, lon_deg=columns.lon_deg,
                         values=values, mask=mask, bracket=bracket,
                         sigma_target=sigma_target, surface_pressure=p_s,
                         surface_height=columns.surface_height)


def pressure_surface(frame, context, lat_deg, lon_deg, surface, *,
                     winds: bool = True, levels: LevelFields | None = None,
                     level_sigma: str = "hydrostatic",
                     time_label: str = "this saved state") -> SurfaceFields:
    """Interpolate one saved state to the derived surface ``surface``.

    ``surface`` is ``("pressure", Pa)``, ``("geopotential_height", m)`` or
    one of the surface classes (the name predates height surfaces).
    :func:`pressure_columns` followed by :func:`interpolate_columns`: the
    native levels are reconstructed on the display mesh (all full levels,
    winds when ``winds``), or taken from ``levels`` when the caller already
    has them. ``level_sigma`` chooses where a level sits (module doc).
    Raises on invalid column geometry and when no node of the state samples
    the surface.
    """
    target = parse_surface(surface)          # refuse a bad spec first
    columns = pressure_columns(frame, context, lat_deg, lon_deg, winds=winds,
                               levels=levels, level_sigma=level_sigma)
    return interpolate_columns(columns, target, time_label=time_label)
