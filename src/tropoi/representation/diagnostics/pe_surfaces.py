"""Fields of a saved PE state on a fixed-pressure (isobaric) surface (CPU).

A derived display surface: the native full-level fields of
:func:`~tropoi.representation.diagnostics.pe_levels.reconstruct_levels`
interpolated in the vertical to one pressure ``p0``. The solver never
produces such a surface; this module only reads saved states. Rules:

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

Vorticity and divergence are deliberately absent: a horizontal derivative on
a sigma level is not the derivative on a pressure level, and the honest
target-coordinate derivative is a later slice of issue #13.

NumPy only; never imports CuPy, never touches the page.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from tropoi.representation.diagnostics.pe_levels import (LevelFields,
                                                         reconstruct_levels)

__all__ = [
    "MASK_ABOVE_TOP_LEVEL",
    "MASK_BELOW_BOTTOM_LEVEL",
    "MASK_BELOW_SURFACE",
    "MASK_LABELS",
    "MASK_VALID",
    "PressureSurface",
    "SURFACE_FIELDS",
    "SurfaceFields",
    "SurfaceUnavailableError",
    "LEVEL_SIGMA_CHOICES",
    "interpolate_log_sigma",
    "level_sigma_positions",
    "parse_surface",
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

#: Interpolated outputs of :func:`pressure_surface` (``u``/``v`` need winds).
SURFACE_FIELDS = ("temperature", "geopotential", "geopotential_height",
                  "u", "v")

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
        if isinstance(p, bool) or not isinstance(p, (int, float, np.integer,
                                                      np.floating)):
            raise TypeError(f"pressure must be a number in Pa, got {p!r}")
        p = float(p)
        if not (math.isfinite(p) and p > 0.0):
            raise ValueError(f"pressure must be finite and > 0 Pa, got {p}")
        object.__setattr__(self, "pressure_pa", p)

    @property
    def label(self) -> str:
        """``"500 hPa"`` or, under 100 Pa, ``"61 Pa"``."""
        p = self.pressure_pa
        if p >= 100.0:
            return f"{p / 100.0:.6g} hPa"
        return f"{p:.6g} Pa"


def parse_surface(spec) -> PressureSurface:
    """``("pressure", value_pa)`` or a :class:`PressureSurface`.

    ``("geopotential_height", ...)`` is refused by name: that coordinate is
    a later slice of issue #13, not a wrong request.
    """
    if isinstance(spec, PressureSurface):
        return spec
    if isinstance(spec, (str, bytes)) or not hasattr(spec, "__len__") \
            or len(spec) != 2:
        raise TypeError(
            "surface must be ('pressure', value_in_Pa), e.g. "
            f"('pressure', 50000.0) for 500 hPa; got {spec!r}")
    kind, value = spec
    if kind == "pressure":
        return PressureSurface(value)
    if kind == "geopotential_height":
        raise NotImplementedError(
            "fixed geopotential-height surfaces are not implemented in this "
            "slice of issue #13; only ('pressure', Pa) is available")
    raise ValueError(
        f"unknown surface coordinate {kind!r}; use ('pressure', Pa)")


@dataclass(frozen=True, eq=False)
class SurfaceFields:
    """One saved state on one pressure surface; arrays are ``(nlat, nlon)``.

    ``mask`` holds the codes above; ``bracket`` is the index of the full
    level just above the target (``k`` with ``sigma_k <= sigma* <=
    sigma_{k+1}``), ``-1`` where masked. ``values[name]`` is finite
    everywhere; where masked it holds the value of the nearest valid
    bracket end for the top/bottom codes and is meaningless (0.0) under
    ground. Read it only through ``valid``.
    """

    surface: PressureSurface
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


def pressure_surface(frame, context, lat_deg, lon_deg, surface, *,
                     winds: bool = True, levels: LevelFields | None = None,
                     level_sigma: str = "hydrostatic",
                     time_label: str = "this saved state") -> SurfaceFields:
    """Interpolate one saved state to the pressure surface ``surface``.

    ``surface`` is a :class:`PressureSurface` or ``("pressure", Pa)``. The
    native levels are reconstructed with :func:`reconstruct_levels` on the
    display mesh (all full levels, winds when ``winds``), or taken from
    ``levels`` when the caller already has them (it must hold every full
    level of ``context`` on this mesh). ``level_sigma`` chooses where a
    level sits (module doc). Raises on invalid column geometry and when no
    node of the state samples the surface.
    """
    target = parse_surface(surface)
    positions = level_sigma_positions(context, level_sigma)
    if levels is None:
        levels = reconstruct_levels(frame, context, lat_deg, lon_deg, None,
                                    winds=winds, vorticity_divergence=False)
    if levels.levels != tuple(range(context.nlev)):
        raise ValueError("pressure_surface needs every full level of the "
                         f"run, got levels {levels.levels}")
    if winds and "u" not in levels.values:
        raise ValueError("winds were requested but the given levels hold "
                         "no u/v")
    _check_column_geometry(levels)
    p_s = levels.surface_pressure
    if not (np.all(np.isfinite(p_s)) and np.all(p_s > 0.0)):
        raise ValueError("surface pressure is not finite and positive")
    sigma_target = target.pressure_pa / p_s
    columns = {"temperature": levels["temperature"],
               "geopotential": levels["geopotential"]}
    if winds:
        columns["u"] = levels["u"]
        columns["v"] = levels["v"]
    values, mask, bracket = interpolate_log_sigma(positions, columns,
                                                  sigma_target)
    if not np.any(mask == MASK_VALID):
        sig = positions
        raise SurfaceUnavailableError(
            f"the {target.label} surface has no valid node at {time_label}: "
            f"the saved surface pressure spans {p_s.min():.6g} to "
            f"{p_s.max():.6g} Pa and the full levels sample "
            f"{sig[0] * p_s.min():.6g} to {sig[-1] * p_s.max():.6g} Pa; "
            "choose a pressure inside that range")
    g = float(context.gravity)
    values["geopotential_height"] = values["geopotential"] / g
    return SurfaceFields(surface=target, level_sigma=level_sigma,
                         level_positions=positions, lat_deg=levels.lat_deg,
                         lon_deg=levels.lon_deg, values=values, mask=mask,
                         bracket=bracket, sigma_target=sigma_target,
                         surface_pressure=p_s,
                         surface_height=levels.surface_height)
