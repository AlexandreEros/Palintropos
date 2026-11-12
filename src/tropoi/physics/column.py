"""A horizontally uniform atmospheric column for optics (CPU, NumPy).

The column is a table of nodes ``(z, ln p, T)`` with ``z`` the altitude above
the planet's reference sphere (metres, increasing upward). Between nodes
``ln p`` and ``T`` vary linearly in ``z`` (exact for an isothermal layer;
an approximation otherwise); number density is the ideal-gas
``n = p / (k_B T)``, which needs no composition. Altitude <-> pressure
conversions use the same table, so they invert each other exactly.

Two constructions, both labelled:

* :meth:`AtmosphericColumn.from_levels` -- one column of a saved
  primitive-equation state: full-level geopotential heights ``Z_k = Phi_k/g``
  (used as altitude: the model's constant-gravity convention), pressures
  ``sigma_k p_s`` and temperatures, plus the surface ``(h_s, p_s)``. Below
  the lowest full level the layer is isothermal at its temperature; above
  the top full level an isothermal extension (``T`` of the top level, scale
  height ``r_dry T / g``) carries the column up to ``p = top_ratio * p_s``.
  The saved state says nothing about that extension; it is stated, and an
  observer is only placed between the surface and the top full level.
* :meth:`AtmosphericColumn.hypothetical` -- a profile the user prescribes
  (isothermal, or a constant lapse rate down to a floor temperature, then
  isothermal), with ``p_s`` at the reference sphere (altitude 0). It is not
  from any simulation and says so.

Above ``p = top_ratio * p_s`` the column is empty (stated).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from tropoi.physics.constants import K_B

__all__ = ["AtmosphericColumn", "ColumnError", "DEFAULT_TOP_RATIO",
           "thin_atmosphere_mass_kg", "thin_atmosphere_surface_pressure_pa"]

DEFAULT_TOP_RATIO = 1e-5


def thin_atmosphere_mass_kg(surface_pressure_pa: float, gravity: float,
                            radius_m: float) -> float:
    """Total mass of an atmosphere, M = 4 pi R^2 p_s / g.

    The standard thin-atmosphere relation: the weight of the whole column
    over the sphere's surface. It ignores that the shell widens with height
    (an undercount of about 2H/R, 0.3 % for Earth) and that g falls with
    height, as the hypothetical column itself does.
    """
    return 4.0 * math.pi * radius_m * radius_m * surface_pressure_pa / gravity


def thin_atmosphere_surface_pressure_pa(mass_kg: float, gravity: float,
                                        radius_m: float) -> float:
    """The inverse of :func:`thin_atmosphere_mass_kg`: p_s = g M / (4 pi R^2)."""
    return gravity * mass_kg / (4.0 * math.pi * radius_m * radius_m)


class ColumnError(ValueError):
    """An invalid column or an observer outside it."""


def _finite_positive(name, value) -> float:
    value = float(value)
    if not (math.isfinite(value) and value > 0.0):
        raise ColumnError(f"{name} must be finite and > 0, got {value}")
    return value


@dataclass(frozen=True, eq=False)
class AtmosphericColumn:
    """Nodes ``z_m`` (ascending), ``ln_p`` (Pa) and ``temperature_k``."""

    z_m: np.ndarray
    ln_p: np.ndarray
    temperature_k: np.ndarray
    kind: str                       # "saved_state" | "hypothetical"
    label: str
    ceiling_m: float                # highest allowed observer altitude
    gas_constant: float
    gravity: float
    notes: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        z = np.asarray(self.z_m, dtype=np.float64)
        lp = np.asarray(self.ln_p, dtype=np.float64)
        t = np.asarray(self.temperature_k, dtype=np.float64)
        if not (z.ndim == 1 and z.shape == lp.shape == t.shape
                and z.size >= 2):
            raise ColumnError("a column needs >= 2 nodes of z, ln p and T")
        if not np.all(np.isfinite(z)) or not np.all(np.isfinite(lp)) \
                or not np.all(np.isfinite(t)):
            raise ColumnError("column nodes must be finite")
        if not np.all(np.diff(z) > 0.0):
            raise ColumnError("column altitudes must increase strictly "
                              "upward")
        if not np.all(np.diff(lp) < 0.0):
            raise ColumnError("column pressure must decrease strictly "
                              "upward")
        if not np.all(t > 0.0):
            raise ColumnError("column temperatures must be > 0 K")
        for name, a in (("z_m", z), ("ln_p", lp), ("temperature_k", t)):
            a = a.copy()
            a.flags.writeable = False
            object.__setattr__(self, name, a)

    # -- construction --------------------------------------------------------

    @classmethod
    def from_levels(cls, z_levels_m, p_levels_pa, t_levels_k, *,
                    surface_height_m: float, surface_pressure_pa: float,
                    gas_constant: float, gravity: float,
                    top_ratio: float = DEFAULT_TOP_RATIO,
                    label: str = "saved state") -> "AtmosphericColumn":
        """Column of saved native full levels (any order) plus the surface."""
        z = np.asarray(z_levels_m, dtype=np.float64).ravel()
        p = np.asarray(p_levels_pa, dtype=np.float64).ravel()
        t = np.asarray(t_levels_k, dtype=np.float64).ravel()
        if not (z.shape == p.shape == t.shape and z.size >= 1):
            raise ColumnError("levels need matching Z, p and T")
        order = np.argsort(z)
        z, p, t = z[order], p[order], t[order]
        hs = float(surface_height_m)
        ps = _finite_positive("surface_pressure_pa", surface_pressure_pa)
        r = _finite_positive("gas_constant", gas_constant)
        g = _finite_positive("gravity", gravity)
        if not z[0] > hs:
            raise ColumnError(f"the lowest full level ({z[0]:.6g} m) is not "
                              f"above the surface ({hs:.6g} m)")
        if not np.all(p > 0.0) or not p[-1] < ps:
            raise ColumnError("level pressures must be > 0 and below p_s")
        zs = [hs, *z]
        lps = [math.log(ps), *np.log(p)]
        ts = [t[0], *t]
        lp_end = math.log(ps * float(top_ratio))
        if lps[-1] > lp_end:
            zs.append(z[-1] + (lps[-1] - lp_end) * r * t[-1] / g)
            lps.append(lp_end)
            ts.append(t[-1])
        notes = (
            "Altitudes are geopotential heights Z = Φ/g of the saved native "
            "full levels (the model's constant-gravity convention).",
            "Between nodes ln p and T are linear in altitude; below the "
            "lowest full level the layer is isothermal at its temperature.",
            f"Above the top full level: an isothermal extension (T of the top "
            f"level, scale height r_dry·T/g) up to p = {top_ratio:g}·p_s; the "
            "saved state does not describe it. Nothing above that.",
        )
        return cls(np.array(zs), np.array(lps), np.array(ts), "saved_state",
                   label, float(z[-1]), r, g, notes)

    @classmethod
    def hypothetical(cls, *, surface_pressure_pa: float, temperature_k: float,
                     gravity: float, gas_constant: float,
                     lapse_rate_k_per_m: float = 0.0,
                     temperature_floor_k: float | None = None,
                     top_ratio: float = DEFAULT_TOP_RATIO,
                     nodes: int = 64,
                     label: str = "hypothetical column"
                     ) -> "AtmosphericColumn":
        """A prescribed column with ``p_s`` at altitude 0 (labelled)."""
        ps = _finite_positive("surface_pressure_pa", surface_pressure_pa)
        t0 = _finite_positive("temperature_k", temperature_k)
        g = _finite_positive("gravity", gravity)
        r = _finite_positive("gas_constant", gas_constant)
        gamma = float(lapse_rate_k_per_m)
        if not (math.isfinite(gamma) and gamma >= 0.0):
            raise ColumnError(f"lapse_rate_k_per_m must be >= 0, got {gamma}")
        lp0, lp_end = math.log(ps), math.log(ps * float(top_ratio))
        if gamma == 0.0:
            z_end = (lp0 - lp_end) * r * t0 / g
            zs, lps, ts = [0.0, z_end], [lp0, lp_end], [t0, t0]
            shape = f"isothermal at {t0:g} K"
        else:
            if temperature_floor_k is None:
                raise ColumnError("a lapse rate needs temperature_floor_k "
                                  "(the isothermal temperature above it)")
            tf = _finite_positive("temperature_floor_k", temperature_floor_k)
            if not tf < t0:
                raise ColumnError("temperature_floor_k must be below "
                                  "temperature_k")
            z_f = (t0 - tf) / gamma
            expo = g / (r * gamma)
            zs = list(np.linspace(0.0, z_f, int(nodes)))
            ts = [t0 - gamma * z for z in zs]
            lps = [lp0 + expo * math.log(t / t0) for t in ts]
            if lps[-1] <= lp_end:
                raise ColumnError("the lapse-rate layer reaches the column "
                                  "top; raise temperature_floor_k")
            zs.append(z_f + (lps[-1] - lp_end) * r * tf / g)
            lps.append(lp_end)
            ts.append(tf)
            shape = (f"{t0:g} K at the surface, lapse rate {gamma * 1000:g} "
                     f"K/km to {tf:g} K at {z_f / 1000:.4g} km, isothermal "
                     "above")
        notes = (
            "HYPOTHETICAL column: prescribed by the user, not from a "
            "simulation.",
            f"p_s = {ps:g} Pa at the reference sphere (altitude 0), "
            f"g = {g:g} m s⁻², R = {r:.6g} J kg⁻¹ K⁻¹; {shape}; hydrostatic.",
            "Terrain does not change the column: an observer on high terrain "
            "samples it higher up.",
            f"Nothing above p = {top_ratio:g}·p_s.",
        )
        return cls(np.array(zs), np.array(lps), np.array(ts), "hypothetical",
                   label, float(zs[-1]), r, g, notes)

    # -- evaluation -------------------------------------------------------------

    @property
    def base_m(self) -> float:
        return float(self.z_m[0])

    @property
    def top_m(self) -> float:
        return float(self.z_m[-1])

    @property
    def surface_pressure_pa(self) -> float:
        return float(math.exp(self.ln_p[0]))

    def _check_z(self, z):
        z = np.asarray(z, dtype=np.float64)
        if np.any(z < self.base_m - 1e-9) or np.any(z > self.top_m + 1e-9):
            raise ColumnError(f"altitude outside the column "
                              f"[{self.base_m:.6g}, {self.top_m:.6g}] m")
        return z

    def pressure_at(self, z_m) -> np.ndarray:
        z = self._check_z(z_m)
        return np.exp(np.interp(z, self.z_m, self.ln_p))

    def temperature_at(self, z_m) -> np.ndarray:
        z = self._check_z(z_m)
        return np.interp(z, self.z_m, self.temperature_k)

    def altitude_at(self, pressure_pa) -> np.ndarray:
        p = np.asarray(pressure_pa, dtype=np.float64)
        lp = np.log(p)
        if np.any(lp > self.ln_p[0] + 1e-12) or np.any(lp < self.ln_p[-1]):
            raise ColumnError("pressure outside the column")
        return np.interp(lp, self.ln_p[::-1], self.z_m[::-1])

    def number_density(self, z_m) -> np.ndarray:
        """Ideal-gas number density (m^-3) at altitude; 0 outside."""
        z = np.asarray(z_m, dtype=np.float64)
        lp = np.interp(z, self.z_m, self.ln_p)
        t = np.interp(z, self.z_m, self.temperature_k)
        n = np.exp(lp) / (K_B * t)
        return np.where((z >= self.base_m) & (z <= self.top_m), n, 0.0)

    def observer_altitude(self, *, altitude_m: float | None = None,
                          pressure_pa: float | None = None,
                          terrain_m: float | None = None) -> dict:
        """Place an observer by altitude above the reference sphere OR by
        pressure; refuse underground and above-ceiling positions."""
        if (altitude_m is None) == (pressure_pa is None):
            raise ColumnError("give exactly one of altitude_m or pressure_pa")
        floor = self.base_m if terrain_m is None else max(self.base_m,
                                                          float(terrain_m))
        if pressure_pa is not None:
            p = _finite_positive("pressure_pa", pressure_pa)
            if p > self.surface_pressure_pa:
                raise ColumnError(
                    f"{p:g} Pa exceeds the surface pressure "
                    f"{self.surface_pressure_pa:.6g} Pa here: the observer "
                    "would be underground")
            z = float(self.altitude_at(p))
            mode = "pressure"
        else:
            z = float(altitude_m)
            if not math.isfinite(z):
                raise ColumnError("altitude_m must be finite")
            mode = "altitude"
        if z < floor - 1e-6:
            raise ColumnError(f"altitude {z:.6g} m is below the ground "
                              f"({floor:.6g} m) here: the observer would be "
                              "underground")
        if z > self.ceiling_m + 1e-6:
            raise ColumnError(
                f"altitude {z:.6g} m is above "
                + ("the top saved full level" if self.kind == "saved_state"
                   else "the top of the column")
                + f" ({self.ceiling_m:.6g} m)")
        return {"mode": mode, "altitude_m": z,
                "pressure_pa": float(self.pressure_at(z))}

    def to_dict(self) -> dict:
        return {"kind": self.kind, "label": self.label,
                "z_m": self.z_m.tolist(), "ln_p": self.ln_p.tolist(),
                "temperature_k": self.temperature_k.tolist(),
                "ceiling_m": self.ceiling_m, "gas_constant": self.gas_constant,
                "gravity": self.gravity, "notes": list(self.notes)}
