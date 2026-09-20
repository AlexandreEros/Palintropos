"""Immutable field specifications and named host views over packed storage.

Every solver persists ONE packed complex coefficient array per stored time
(``(l_max+1, l_max+1)`` for the BVE, ``(rows, l_max+1, l_max+1)`` for the
shallow-water and primitive-equation stacks). This module names the rows of
that storage without copying or re-arranging it:

* :class:`FieldSpec` says which rows of a packed frame hold one physical
  field, what it means (name, units, description), whether it carries a
  vertical level axis, and which coefficient conventions apply;
* :class:`SpectralModes` is a read-only host view of one field: the
  coefficients, the stored capacity ``l_max``, the level coordinates where
  applicable, and explicit convention validation on request;
* :class:`SpectralState` is the immutable ``name -> SpectralModes`` mapping
  over one packed frame.

Views never own or copy the packed storage: ``SpectralModes.coeffs`` shares
memory with the frame it was sliced from (a tested contract), and the frame
is exposed read-only so a view can never mutate an archive through this
interface. Nothing here performs solver arithmetic; state and rate semantics
are the solver's, and these views only describe the stored prognostic state.

Coefficient conventions (docs/MATHEMATICAL_MODEL.md): complex orthonormal
spherical harmonics stored for ``m >= 0`` in a dense ``(l_max+1, l_max+1)``
array with axis 0 = degree ``l`` and axis 1 = order ``m``; entries with
``m > l`` are storage padding and must be exactly zero; real fields are
implied, so ``Im a_l0`` must be exactly zero.
"""
from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass

import numpy as np

#: The one coefficient layout every core persists.
COEFFICIENT_LAYOUT = "unpacked-l-m-nonnegative"
#: The one normalization every core persists.
COEFFICIENT_NORMALIZATION = "orthonormal-complex-m>=0-real-field"

#: Monopole (``a_00``) rules a field may declare. ``zero-mean`` mirrors the
#: solvers' own ``validate_state`` check (|a_00| <= rtol * ||field||);
#: ``conserved`` means the monopole is an invariant that need not vanish
#: (BVE circulation); ``horizontal-mean`` means the monopole IS the field's
#: area mean and carries physical content (temperature, ln p_s).
MONOPOLE_RULES = ("zero-mean", "conserved", "horizontal-mean")

#: Relative tolerance of the ``zero-mean`` rule; identical to the
#: ``_MONOPOLE_RTOL`` the shallow-water and primitive-equation cores enforce.
ZERO_MEAN_MONOPOLE_RTOL = 1e-10


class SpectralConventionError(ValueError):
    """Stored coefficients violate a declared convention (not repaired)."""


@dataclass(frozen=True)
class FieldSpec:
    """Where one physical field lives inside a packed frame, and what it means.

    ``rows`` is ``None`` when the packed frame IS the field (BVE: a single
    ``(l_max+1, l_max+1)`` block per time), otherwise the half-open
    ``(start, stop)`` row range on the frame's leading axis. ``levels`` marks
    a field that keeps a leading vertical-level axis (``stop - start`` model
    levels, top to bottom); a level-free field selects exactly one row and
    is exposed as a plain ``(l_max+1, l_max+1)`` block with NO padded level
    axis.
    """

    name: str
    units: str
    description: str
    rows: tuple[int, int] | None = None
    levels: bool = False
    monopole: str = "zero-mean"
    normalization: str = COEFFICIENT_NORMALIZATION
    layout: str = COEFFICIENT_LAYOUT

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("field name must be a nonempty string")
        if not isinstance(self.units, str):
            raise TypeError("field units must be a string")
        if self.monopole not in MONOPOLE_RULES:
            raise ValueError(
                f"unknown monopole rule {self.monopole!r}; choose from "
                f"{MONOPOLE_RULES}")
        if self.rows is not None:
            start, stop = (int(self.rows[0]), int(self.rows[1]))
            if start < 0 or stop <= start:
                raise ValueError(
                    f"field {self.name!r} rows must satisfy 0 <= start < "
                    f"stop, got {self.rows}")
            if not self.levels and stop - start != 1:
                raise ValueError(
                    f"level-free field {self.name!r} must select exactly one "
                    f"row, got {self.rows}")
            object.__setattr__(self, "rows", (start, stop))
        elif self.levels:
            raise ValueError(
                f"field {self.name!r} cannot carry a level axis without rows")

    @property
    def dimensions(self) -> tuple[str, ...]:
        """Axis names of the host view this specification produces."""
        return ("level", "l", "m") if self.levels else ("l", "m")

    @property
    def nlev(self) -> int | None:
        if not self.levels:
            return None
        return self.rows[1] - self.rows[0]

    def select(self, frame: np.ndarray) -> np.ndarray:
        """Slice this field out of one packed frame without copying."""
        if self.rows is None:
            if frame.ndim != 2:
                raise SpectralConventionError(
                    f"field {self.name!r} expects a single (l, m) block per "
                    f"frame, got frame shape {frame.shape}")
            return frame
        if frame.ndim != 3:
            raise SpectralConventionError(
                f"field {self.name!r} expects a (rows, l, m) frame, got "
                f"frame shape {frame.shape}")
        start, stop = self.rows
        if stop > frame.shape[0]:
            raise SpectralConventionError(
                f"field {self.name!r} needs rows [{start}, {stop}) but the "
                f"frame has only {frame.shape[0]} row(s)")
        return frame[start:stop] if self.levels else frame[start]

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "units": self.units,
            "description": self.description,
            "rows": None if self.rows is None else list(self.rows),
            "levels": self.levels,
            "dimensions": list(self.dimensions),
            "monopole": self.monopole,
            "normalization": self.normalization,
            "layout": self.layout,
        }

    @classmethod
    def from_dict(cls, data: Mapping) -> "FieldSpec":
        rows = data.get("rows")
        return cls(
            name=str(data["name"]), units=str(data["units"]),
            description=str(data.get("description", "")),
            rows=None if rows is None else (int(rows[0]), int(rows[1])),
            levels=bool(data.get("levels", False)),
            monopole=str(data.get("monopole", "zero-mean")),
            normalization=str(data.get("normalization",
                                       COEFFICIENT_NORMALIZATION)),
            layout=str(data.get("layout", COEFFICIENT_LAYOUT)))


def _read_only(array: np.ndarray) -> np.ndarray:
    """A non-writeable view sharing memory with ``array``.

    NumPy lets a view regain write access with ``setflags(write=True)``
    when its owner is writeable, so callers that must guarantee immutability
    (the capsule storage) freeze the owning array itself; this helper only
    removes the flag on the view it hands out.
    """
    view = array.view()
    view.flags.writeable = False
    return view


@dataclass(frozen=True)
class SpectralModes:
    """Read-only host view of one field's spherical-harmonic coefficients.

    ``coeffs`` has shape ``(l_max+1, l_max+1)`` or, for a field with a
    vertical axis, ``(nlev, l_max+1, l_max+1)``; it shares memory with the
    packed frame it was selected from and rejects writes. ``level_values``
    are the model full-level coordinates (dimensionless sigma) for a
    levelled field, ``None`` otherwise.
    """

    coeffs: np.ndarray
    spec: FieldSpec
    level_values: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        coeffs = np.asarray(self.coeffs)
        expected_ndim = 3 if self.spec.levels else 2
        if coeffs.ndim != expected_ndim:
            raise SpectralConventionError(
                f"field {self.spec.name!r} expects {expected_ndim}-D "
                f"coefficients, got shape {coeffs.shape}")
        if coeffs.shape[-1] != coeffs.shape[-2] or coeffs.shape[-1] == 0:
            raise SpectralConventionError(
                f"field {self.spec.name!r} needs square nonempty (l, m) "
                f"axes, got {coeffs.shape[-2:]}")
        if not np.iscomplexobj(coeffs):
            raise SpectralConventionError(
                f"field {self.spec.name!r} coefficients must be complex, got "
                f"{coeffs.dtype}")
        if self.spec.levels:
            if coeffs.shape[0] != self.spec.nlev:
                raise SpectralConventionError(
                    f"field {self.spec.name!r} declares {self.spec.nlev} "
                    f"level(s) but the view has {coeffs.shape[0]}")
            if self.level_values is not None:
                values = tuple(float(v) for v in self.level_values)
                if len(values) != coeffs.shape[0]:
                    raise SpectralConventionError(
                        f"field {self.spec.name!r} has {coeffs.shape[0]} "
                        f"level(s) but {len(values)} level coordinate(s)")
                object.__setattr__(self, "level_values", values)
        elif self.level_values is not None:
            raise SpectralConventionError(
                f"level-free field {self.spec.name!r} cannot carry level "
                "coordinates")
        object.__setattr__(self, "coeffs", _read_only(coeffs))

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def units(self) -> str:
        return self.spec.units

    @property
    def description(self) -> str:
        return self.spec.description

    @property
    def dimensions(self) -> tuple[str, ...]:
        return self.spec.dimensions

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.coeffs.shape)

    @property
    def l_max(self) -> int:
        return int(self.coeffs.shape[-1]) - 1

    @property
    def nlev(self) -> int | None:
        return int(self.coeffs.shape[0]) if self.spec.levels else None

    @property
    def valid_mask(self) -> np.ndarray:
        """Boolean ``(l_max+1, l_max+1)`` mask of the stored triangle m <= l."""
        n = self.l_max + 1
        return np.arange(n)[None, :] <= np.arange(n)[:, None]

    def level(self, index: int) -> "SpectralModes":
        """One model level of a levelled field as a level-free view."""
        if not self.spec.levels:
            raise SpectralConventionError(
                f"field {self.spec.name!r} has no level axis")
        index = int(index)
        nlev = self.coeffs.shape[0]
        if not -nlev <= index < nlev:
            raise IndexError(
                f"level index {index} out of range for {nlev} level(s)")
        start = self.spec.rows[0] + (index % nlev)
        spec = FieldSpec(
            name=self.spec.name, units=self.spec.units,
            description=self.spec.description, rows=(start, start + 1),
            levels=False, monopole=self.spec.monopole,
            normalization=self.spec.normalization, layout=self.spec.layout)
        return SpectralModes(self.coeffs[index], spec)

    def validate(self) -> "SpectralModes":
        """Check the stored conventions; raise instead of repairing.

        Verifies finiteness of every coefficient, exact zeros on the
        ``m > l`` padding triangle, exactly real ``a_l0`` (the reality
        convention), and the field's monopole rule (``zero-mean`` uses the
        solvers' own relative tolerance; ``conserved`` and
        ``horizontal-mean`` impose nothing). Returns ``self`` for chaining.
        """
        name = self.spec.name
        coeffs = self.coeffs
        if not np.isfinite(coeffs).all():
            raise SpectralConventionError(
                f"field {name!r} contains non-finite coefficients")
        padding = coeffs[..., ~self.valid_mask]
        if padding.size and np.any(padding != 0.0):
            raise SpectralConventionError(
                f"field {name!r} has nonzero coefficients on the m > l "
                "padding triangle; the stored layout is not the dense "
                f"{self.spec.layout} triangle")
        zonal = coeffs[..., :, 0]
        if np.any(zonal.imag != 0.0):
            raise SpectralConventionError(
                f"field {name!r} has nonzero Im(a_l0); the real-field "
                "reality convention is violated")
        if self.spec.monopole == "zero-mean":
            blocks = coeffs.reshape(-1, coeffs.shape[-2], coeffs.shape[-1])
            for k, block in enumerate(blocks):
                monopole = float(abs(block[0, 0]))
                scale = float(np.linalg.norm(block))
                if monopole > ZERO_MEAN_MONOPOLE_RTOL * max(scale, 1e-30):
                    where = f" at level {k + 1}" if self.spec.levels else ""
                    raise SpectralConventionError(
                        f"field {name!r} monopole is nonzero{where}: |a00| = "
                        f"{monopole:g} (field norm {scale:g}); the declared "
                        "rule is zero-mean")
        return self

    def summary(self) -> dict:
        """Small host-only numeric summary (no synthesis, no transforms)."""
        coeffs = self.coeffs
        magnitude = np.abs(coeffs[..., self.valid_mask])
        monopole = coeffs[..., 0, 0]
        return {
            "shape": self.shape,
            "l_max": self.l_max,
            "nlev": self.nlev,
            "max_abs": float(magnitude.max()) if magnitude.size else 0.0,
            "nonzero": int(np.count_nonzero(magnitude)),
            "monopole_real": (
                [float(v) for v in np.atleast_1d(monopole.real)]),
            "finite": bool(np.isfinite(coeffs).all()),
        }


class SpectralState(Mapping):
    """Immutable ``name -> SpectralModes`` mapping over one packed frame.

    Views are created lazily on first access and cached; each one shares
    memory with ``packed`` (never a copy). ``packed`` itself is exposed
    read-only so the mapping can never be used to alter stored data.
    """

    def __init__(self, packed: np.ndarray, fields: tuple[FieldSpec, ...],
                 *, level_values: tuple[float, ...] | None = None) -> None:
        packed = np.asarray(packed)
        if packed.ndim not in (2, 3):
            raise SpectralConventionError(
                "a packed frame must have shape (l, m) or (rows, l, m), got "
                f"{packed.shape}")
        names = [spec.name for spec in fields]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate field names: {names}")
        self._packed = _read_only(packed)
        self._fields = {spec.name: spec for spec in fields}
        self._level_values = (
            None if level_values is None
            else tuple(float(v) for v in level_values))
        self._views: dict[str, SpectralModes] = {}

    @property
    def packed(self) -> np.ndarray:
        """The complete read-only packed frame the views are sliced from."""
        return self._packed

    @property
    def specs(self) -> tuple[FieldSpec, ...]:
        return tuple(self._fields.values())

    def __getitem__(self, name: str) -> SpectralModes:
        try:
            view = self._views[name]
        except KeyError:
            try:
                spec = self._fields[name]
            except KeyError:
                raise KeyError(
                    f"unknown field {name!r}; this state stores "
                    f"{', '.join(self._fields)}") from None
            view = SpectralModes(
                spec.select(self._packed), spec,
                level_values=self._level_values if spec.levels else None)
            self._views[name] = view
        return view

    def __iter__(self) -> Iterator[str]:
        return iter(self._fields)

    def __len__(self) -> int:
        return len(self._fields)

    def __repr__(self) -> str:
        return f"SpectralState({', '.join(self._fields)})"


__all__ = [
    "COEFFICIENT_LAYOUT",
    "COEFFICIENT_NORMALIZATION",
    "FieldSpec",
    "MONOPOLE_RULES",
    "SpectralConventionError",
    "SpectralModes",
    "SpectralState",
    "ZERO_MEAN_MONOPOLE_RTOL",
]
