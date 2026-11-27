"""Material strata for vertically resolved conduction.

A column is a stack of physical :class:`Stratum` objects. The number of
strata has nothing to do with the number of numerical cells: each stratum
is resolved by as many cells as the grid gives it, and every stratum
boundary must coincide with a cell face
(:func:`tropoi.spatial.vertical.stratified_interfaces` builds such grids).

**Property families.** These are the ones every built-in material uses.
Depth z is in m, positive downward; temperature T is in K.

* density ``rho(z)`` (kg m^-3): a constant or a :class:`DepthProfile`;
* conductivity ``k(T, z) = k_c(z) [1 + chi(z) (T / T_r)^3]`` (W m^-1 K^-1).
  With chi = 0 the conductivity is constant in T. The cubic term is the
  usual radiative conduction across pores;
* specific heat c_p(T) (J kg^-1 K^-1), from a heat-capacity law. A law is
  a polynomial in a shifted and scaled temperature, a saturating
  exponential, or a piecewise combination of the two. Each comes with an
  analytic enthalpy integral and a declared valid temperature range.

Each stratum carries the source of its numbers. Mercury, the Moon and THAI
are configurations of these families. None of them is built into the
solver.

**Numerics.** Enthalpy and Kirchhoff-potential differences are evaluated
as divided differences, (T1 - T0) times a symmetric sum of powers. There is
no subtraction of nearly equal large numbers, so small temperature
increments keep full relative precision.

Import-light: NumPy only; arrays may be NumPy or CuPy.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence, Union

import numpy as np

__all__ = ["CAL_G_TO_J_KG", "CellMaterials", "DepthProfile", "PiecewiseHeatCapacity",
           "PolynomialHeatCapacity", "SaturatingExponentialHeatCapacity", "Stratum",
           "heat_capacity_from_dict", "ledlow_1992"]


@dataclass(frozen=True)
class DepthProfile:
    """``deep - (deep - surface) exp(-z / scale)``: equals ``surface`` at
    z = 0 and tends to ``deep`` at depth (the Hayne et al. 2017 form)."""

    surface: float
    deep: float
    scale: float

    def __post_init__(self):
        if not (self.scale > 0.0 and math.isfinite(self.surface)
                and math.isfinite(self.deep)):
            raise ValueError("DepthProfile needs finite values and scale > 0")

    def __call__(self, z):
        z = np.asarray(z, dtype=np.float64)
        return self.deep - (self.deep - self.surface) * np.exp(-z / self.scale)

    def to_dict(self) -> dict:
        return {"profile": "exponential", "surface": self.surface,
                "deep": self.deep, "scale": self.scale}


Value = Union[float, DepthProfile]


def _evaluate(value: Value, z: np.ndarray) -> np.ndarray:
    if isinstance(value, DepthProfile):
        return value(z)
    return np.full(z.shape, float(value))


def _describe(value: Value):
    return value.to_dict() if isinstance(value, DepthProfile) else float(value)


# -- heat-capacity laws --------------------------------------------------------------
#
# Each law gives c_p(T) (J kg^-1 K^-1) and the enthalpy difference
# h(t1) - h(t0) = integral of c_p from t0 to t1 (J kg^-1), computed in closed form and
# without cancellation when t1 is close to t0. ``delta`` (= t1 - t0), when the
# caller has it exactly (from anomalies), replaces the rounded t1 - t0.
# ``factor`` multiplies the law as printed by its source, which keeps the
# source's coefficients and units verbatim (for example cal g^-1 K^-1 ->
# J kg^-1 K^-1 is factor 4184).


def _like(t):
    return 0.0 * t


@dataclass(frozen=True)
class PolynomialHeatCapacity:
    """c_p = factor * sum_j a_j x^j, with x = (T - center) / scale."""

    coefficients: tuple
    center: float = 0.0
    scale: float = 1.0
    factor: float = 1.0

    def __post_init__(self):
        c = tuple(float(a) for a in self.coefficients)
        if not c or not self.scale > 0.0:
            raise ValueError("need coefficients and scale > 0")
        object.__setattr__(self, "coefficients", c)

    def specific_heat(self, T):
        x = (T - self.center) / self.scale
        out = self.coefficients[-1] + _like(x)
        for a in self.coefficients[-2::-1]:
            out = out * x + a
        return self.factor * out

    def enthalpy_difference(self, t1, t0, delta=None):
        """scale * sum_j a_j/(j+1) (x1^(j+1) - x0^(j+1)), written as
        (x1 - x0) sum_j a_j/(j+1) S_j with S_j = sum_m x1^m x0^(j-m)."""
        x1 = (t1 - self.center) / self.scale
        x0 = (t0 - self.center) / self.scale
        s = 1.0 + _like(x1)
        p0 = 1.0 + _like(x0)
        total = self.coefficients[0] * s
        for j in range(1, len(self.coefficients)):
            p0 = p0 * x0
            s = x1 * s + p0
            total = total + self.coefficients[j] * (s / (j + 1))
        d = (t1 - t0) if delta is None else delta
        return self.factor * d * total

    def to_dict(self) -> dict:
        return {"law": "polynomial", "coefficients": list(self.coefficients),
                "center": self.center, "scale": self.scale, "factor": self.factor}


@dataclass(frozen=True)
class SaturatingExponentialHeatCapacity:
    """c_p = factor * (a + b [1 - exp(-(T - t_ref) / scale)])."""

    a: float
    b: float
    t_ref: float
    scale: float
    factor: float = 1.0

    def specific_heat(self, T):
        e = _xpmod(T).exp(-(T - self.t_ref) / self.scale)
        return self.factor * (self.a + self.b * (1.0 - e))

    def enthalpy_difference(self, t1, t0, delta=None):
        """h = (a + b) T + b s exp(-u), u = (T - t_ref)/s, so
        h(t1) - h(t0) = (a + b)(t1 - t0) + b s [exp(-u1) - exp(-u0)], with
        exp(-u1) - exp(-u0) = exp(-u0) expm1(-(t1 - t0)/s)."""
        xp = _xpmod(t1)
        d = (t1 - t0) if delta is None else delta
        u0 = (t0 - self.t_ref) / self.scale
        tail = xp.exp(-u0) * xp.expm1(-d / self.scale)
        return self.factor * ((self.a + self.b) * d + self.b * self.scale * tail)

    def to_dict(self) -> dict:
        return {"law": "saturating_exponential", "a": self.a, "b": self.b,
                "t_ref": self.t_ref, "scale": self.scale, "factor": self.factor}


@dataclass(frozen=True)
class PiecewiseHeatCapacity:
    """``low`` for T <= ``breakpoint``, ``high`` above it. Enthalpy
    differences that straddle the breakpoint are split there, and each part
    uses its own branch's stable formula."""

    low: object
    high: object
    breakpoint: float

    def specific_heat(self, T):
        xp = _xpmod(T)
        return xp.where(T <= self.breakpoint, self.low.specific_heat(T),
                        self.high.specific_heat(T))

    def enthalpy_difference(self, t1, t0, delta=None):
        xp = _xpmod(t1)
        b = self.breakpoint
        d = (t1 - t0) if delta is None else delta
        lo1, lo0 = xp.minimum(t1, b), xp.minimum(t0, b)
        hi1, hi0 = xp.maximum(t1, b), xp.maximum(t0, b)
        below = (t1 <= b) & (t0 <= b)
        above = (t1 >= b) & (t0 >= b)
        d_lo = xp.where(below, d, lo1 - lo0)
        d_hi = xp.where(above, d, hi1 - hi0)
        return (self.low.enthalpy_difference(lo1, lo0, d_lo)
                + self.high.enthalpy_difference(hi1, hi0, d_hi))

    def to_dict(self) -> dict:
        return {"law": "piecewise", "breakpoint": self.breakpoint,
                "low": self.low.to_dict(), "high": self.high.to_dict()}


def _xpmod(a):
    if type(a).__module__.split(".")[0] == "cupy":
        import cupy  # noqa: WPS433
        return cupy
    return np


def heat_capacity_from_dict(d):
    """A heat-capacity law from its ``to_dict`` form, or from a plain list
    (a polynomial in T)."""
    if isinstance(d, (list, tuple)):
        return PolynomialHeatCapacity(tuple(d))
    law = d["law"]
    if law == "polynomial":
        return PolynomialHeatCapacity(tuple(d["coefficients"]), d.get("center", 0.0),
                                      d.get("scale", 1.0), d.get("factor", 1.0))
    if law == "saturating_exponential":
        return SaturatingExponentialHeatCapacity(d["a"], d["b"], d["t_ref"], d["scale"],
                                                 d.get("factor", 1.0))
    if law == "piecewise":
        return PiecewiseHeatCapacity(heat_capacity_from_dict(d["low"]),
                                     heat_capacity_from_dict(d["high"]), d["breakpoint"])
    raise ValueError(f"unknown heat-capacity law {law!r}")


#: Thermochemical calorie (J): cal g^-1 K^-1 -> J kg^-1 K^-1 is 4184.
CAL_G_TO_J_KG = 4184.0


def ledlow_1992() -> PiecewiseHeatCapacity:
    """Ledlow et al. (1992, ApJ 384, 640), section 5.2, p. 650, eqs. (6) and (7):
    the c_p(T) law that Vasavada, Paige & Wood (1999) adopt.

    * eq. (6), T <= 350 K ("from an analysis of soil samples from the Apollo
      missions"): C = 0.1812 + 0.1191 x + 0.0176 x^2 + 0.2721 x^3 +
      0.1869 x^4, with x = (T - T300) / T300 and T300 = 300 K;
    * eq. (7), T > 350 K (Wechsler, Glaser & Fountain 1972, "valid to such
      extreme temperatures"): C = 0.2029 + 0.0383 [1 - exp(-(T - T350)/T100)].

    Both are in cal g^-1 K^-1, converted here with the thermochemical calorie
    (4.184 J); the paper does not say which calorie (the IT calorie, 4.1868 J,
    would differ by 6.9e-4 relative). The paper states that eq. (7) "reduces to
    the previous expression at T = 350 K". As printed, the two branches differ
    by 4.3e-5 cal g^-1 K^-1 (2e-4 relative) there, through the rounding of
    0.2029. That step is kept, not smoothed. Eq. (6) becomes negative below
    about 10.3 K (no lower limit is stated); temperatures there are outside
    the law and are flagged."""
    low = PolynomialHeatCapacity((0.1812, 0.1191, 0.0176, 0.2721, 0.1869),
                                 center=300.0, scale=300.0, factor=CAL_G_TO_J_KG)
    high = SaturatingExponentialHeatCapacity(0.2029, 0.0383, 350.0, 100.0,
                                             factor=CAL_G_TO_J_KG)
    return PiecewiseHeatCapacity(low, high, 350.0)


@dataclass(frozen=True)
class Stratum:
    """One physical material layer, from the bottom of the stratum above it
    (or the surface) down to ``bottom`` (m).

    ``heat_capacity`` is a heat-capacity law (above), or a sequence of
    polynomial coefficients a_0, a_1, ... of c_p(T) in J kg^-1 K^-(1+j).
    ``heat_capacity_range`` is the (low, high) temperature range the source
    states the law is valid for (``high`` may be None when no upper limit is
    stated), or None when the law is exact, as for a prescribed constant.
    ``source`` says where the numbers come from.
    ``radiative_reference_temperature`` T_r of the cubic radiative term has
    no default: it is required when ``radiative_coefficient`` is non-zero
    (VPW99 and Hayne et al. 2017 both use 350 K) and is None otherwise."""

    bottom: float
    density: Value
    contact_conductivity: Value
    heat_capacity: object
    radiative_coefficient: Value = 0.0
    radiative_reference_temperature: Optional[float] = None
    heat_capacity_range: Optional[tuple] = None
    name: str = "stratum"
    source: str = ""

    def __post_init__(self):
        if not (math.isfinite(self.bottom) and self.bottom > 0.0):
            raise ValueError("stratum bottom must be finite and > 0")
        law = self.heat_capacity
        if not hasattr(law, "enthalpy_difference"):
            law = heat_capacity_from_dict(law)
        object.__setattr__(self, "heat_capacity", law)
        radiative = (isinstance(self.radiative_coefficient, DepthProfile)
                     or float(self.radiative_coefficient) != 0.0)
        t_r = self.radiative_reference_temperature
        if t_r is None:
            if radiative:
                raise ValueError("a radiative conductivity term needs its reference "
                                 "temperature T_r (k = k_c [1 + chi (T/T_r)^3]); it has no default")
        elif not (math.isfinite(float(t_r)) and float(t_r) > 0.0):
            raise ValueError("radiative_reference_temperature must be finite and > 0")
        if self.heat_capacity_range is not None:
            lo, hi = self.heat_capacity_range
            lo = float(lo)
            hi = None if hi is None else float(hi)
            if not (0.0 <= lo and (hi is None or lo < hi)):
                raise ValueError("heat_capacity_range must be (low, high) with low < high")
            object.__setattr__(self, "heat_capacity_range", (lo, hi))

    @classmethod
    def volumetric(cls, bottom: float, volumetric_heat_capacity: float,
                   conductivity: float, *, name: str = "constant",
                   source: str = "") -> "Stratum":
        """A constant-property stratum stated by its volumetric heat
        capacity C (J m^-3 K^-1) and conductivity, with no density given.
        Only rho c_p = C enters conduction, so it is stored as density 1
        and c_p = C. The signature records the basis."""
        return cls(bottom, 1.0, float(conductivity),
                   (float(volumetric_heat_capacity),), name=name,
                   source=source or "volumetric heat capacity prescribed; "
                                    "density not specified")

    def to_dict(self) -> dict:
        return {"name": self.name, "bottom": self.bottom,
                "density": _describe(self.density),
                "contact_conductivity": _describe(self.contact_conductivity),
                "radiative_coefficient": _describe(self.radiative_coefficient),
                "radiative_reference_temperature": self.radiative_reference_temperature,
                "heat_capacity": self.heat_capacity.to_dict(),
                "heat_capacity_range": (None if self.heat_capacity_range is None
                                        else list(self.heat_capacity_range)),
                "source": self.source}


@dataclass
class CellMaterials:
    """Per-cell material parameters on a grid, shaped (L, 1) so that they
    broadcast against (L, npts) temperature arrays."""

    density: np.ndarray
    kc: np.ndarray
    chi_over_tr3: np.ndarray
    segments: tuple                   # ((first cell, end cell, heat-capacity law), ...)
    stratum_of_cell: np.ndarray       # (L,) int
    strata: tuple = field(default=())
    _device: dict = field(default_factory=dict, repr=False)

    @classmethod
    def on_grid(cls, strata: Sequence[Stratum], interfaces: np.ndarray) -> "CellMaterials":
        strata = tuple(strata)
        if not strata:
            raise ValueError("at least one stratum is required")
        bottoms = [s.bottom for s in strata]
        if any(b2 <= b1 for b1, b2 in zip(bottoms, bottoms[1:])):
            raise ValueError("strata must be listed top to bottom with increasing bottoms")
        e = np.asarray(interfaces, dtype=np.float64)
        depth = float(e[-1])
        if not math.isclose(bottoms[-1], depth, rel_tol=1e-12):
            raise ValueError(f"the last stratum ends at {bottoms[-1]} m but the grid "
                             f"ends at {depth} m")
        for b in bottoms[:-1]:
            if not np.any(np.isclose(e, b, rtol=1e-12, atol=0.0)):
                raise ValueError(f"stratum boundary at {b} m is not a cell face")
        centres = 0.5 * (e[1:] + e[:-1])
        which = np.searchsorted(np.asarray(bottoms[:-1]), centres)
        L = centres.size
        rho, kc, chi = np.empty(L), np.empty(L), np.empty(L)
        segments = []
        for j, s in enumerate(strata):
            cells = which == j
            if not np.any(cells):
                raise ValueError(f"stratum {s.name!r} contains no cell")
            z = centres[cells]
            rho[cells] = _evaluate(s.density, z)
            kc[cells] = _evaluate(s.contact_conductivity, z)
            # no radiative term: no reference temperature, and chi / T_r^3 = 0
            chi[cells] = (0.0 if s.radiative_reference_temperature is None else
                          _evaluate(s.radiative_coefficient, z)
                          / s.radiative_reference_temperature ** 3)
            rows = np.flatnonzero(cells)
            segments.append((int(rows[0]), int(rows[-1]) + 1, s.heat_capacity))
        if not (np.all(rho > 0) and np.all(kc > 0) and np.all(chi >= 0)):
            raise ValueError("density and contact conductivity must be > 0 and the "
                             "radiative coefficient >= 0 in every cell")
        return cls(rho[:, None], kc[:, None], chi[:, None], tuple(segments), which, strata)

    def on(self, xp):
        """The parameter arrays on backend ``xp`` (cached)."""
        key = xp.__name__
        if key not in self._device:
            self._device[key] = (xp.asarray(self.density), xp.asarray(self.kc),
                                 xp.asarray(self.chi_over_tr3))
        return self._device[key]

    # -- property laws (all evaluated per cell, T shaped (L, npts) or (L', npts)) --

    @staticmethod
    def conductivity(kc, chi, T):
        return kc * (1.0 + chi * T * T * T)

    @staticmethod
    def conductivity_derivative(kc, chi, T):
        return 3.0 * kc * chi * T * T

    @staticmethod
    def kirchhoff_difference(kc, chi, a, b, delta=None):
        """Phi(a) - Phi(b), where Phi = integral of k dT, written as
        (a - b) times a symmetric sum so that no cancellation occurs.
        ``delta`` (= a - b), when given, is used for the leading factor;
        callers that hold temperatures as anomalies can then pass the
        exactly computed difference."""
        d = (a - b) if delta is None else delta
        return kc * d * (1.0 + 0.25 * chi * (a + b) * (a * a + b * b))

    def specific_heat(self, T):
        """c_p per cell (J kg^-1 K^-1), each stratum by its own law."""
        out = _xpmod(T).empty_like(T)
        for r0, r1, law in self.segments:
            out[r0:r1] = law.specific_heat(T[r0:r1])
        return out

    def enthalpy_difference(self, t1, t0, delta=None):
        """h(t1) - h(t0) per cell (J kg^-1), each stratum by its own law."""
        out = _xpmod(t1).empty_like(t1 + 0.0 * t0)
        for r0, r1, law in self.segments:
            d = None if delta is None else delta[r0:r1]
            out[r0:r1] = law.enthalpy_difference(t1[r0:r1], _rows(t0, r0, r1), d)
        return out


def _rows(a, r0, r1):
    return a[r0:r1] if getattr(a, "ndim", 0) >= 1 else a
