"""Planetary rotation: the spin vector, the axis orientation and the phase.

:class:`Rotation` owns the planet's spin state, and nothing orbital or
stellar.

**Spin.** The spin is the angular-velocity vector ``Omega_vec = Omega e``.
``Omega = angular_speed >= 0`` is the sidereal rate in rad/s. Body north is
``e`` (right-hand rule), and east is the direction of rotation. There is no
signed rotation rate. A planet that spins against its orbital motion
(Venus, Uranus) has an obliquity above 90 degrees.

**Axis orientation.** The axis is oriented relative to the planet's orbit
frame O: x along the orbit's ascending node on the inertial reference
plane, z along the orbit normal (see :mod:`tropoi.physics.orbit`).

* ``obliquity_rad`` is epsilon in [0, pi], the angle between ``e`` and the
  orbit normal.
* ``equinox_longitude_rad`` is lambda_eq, the angle in the orbit plane from
  O's x-axis, measured in the direction of orbital motion, to the ascending
  node of the body's equator on the orbit plane. With this definition the
  planet's argument of latitude at its northern spring equinox
  (``L_s = 0``) is exactly lambda_eq, so ``L_s = u - lambda_eq``.

**Phase.** ``prime_meridian_epoch_rad`` is W_0, the angle measured eastward
about ``e`` from that equatorial node to the prime meridian at the epoch
``t = 0``. The epoch is shared with the orbit. At time t the angle is
``W(t) = W_0 + Omega t``, evaluated from absolute time.

The body-to-orbit-frame rotation is::

    R_BO(t) = R_z(lambda_eq) R_x(epsilon) R_z(W(t))

so body x points at latitude 0, longitude 0, and body z points north.
Illumination combines this with the orbit and the light sources in
:mod:`tropoi.physics.insolation`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from fractions import Fraction

import numpy as np

__all__ = ["Rotation", "rotation_x", "rotation_z"]

_TWO_PI = 2.0 * math.pi


def rotation_z(angle: float) -> np.ndarray:
    """Active right-handed rotation by ``angle`` about +z."""
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rotation_x(angle: float) -> np.ndarray:
    """Active right-handed rotation by ``angle`` about +x."""
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


@dataclass(frozen=True)
class Rotation:
    """Spin state of a planet in SI units (rad/s and radians). See the
    module docstring for the conventions."""

    angular_speed: float
    obliquity_rad: float = 0.0
    equinox_longitude_rad: float = 0.0
    prime_meridian_epoch_rad: float = 0.0

    def __post_init__(self) -> None:
        for name in ("angular_speed", "obliquity_rad",
                     "equinox_longitude_rad", "prime_meridian_epoch_rad"):
            v = float(getattr(self, name))
            if not math.isfinite(v):
                raise ValueError(f"{name} must be finite, got {v}")
        if not self.angular_speed >= 0.0:
            raise ValueError(
                f"angular_speed must be >= 0 (a retrograde spin is an "
                f"obliquity above 90 deg), got {self.angular_speed}")
        if not 0.0 <= self.obliquity_rad <= math.pi:
            raise ValueError(f"obliquity_rad must be in [0, pi], got "
                             f"{self.obliquity_rad}")

    # -- constructors ------------------------------------------------------------

    @classmethod
    def of(cls, angular_speed: float, *, obliquity_deg: float = 0.0,
           equinox_longitude_deg: float = 0.0,
           prime_meridian_epoch_deg: float = 0.0) -> "Rotation":
        """The same state, with the angles given in degrees."""
        return cls(float(angular_speed), math.radians(float(obliquity_deg)),
                   math.radians(float(equinox_longitude_deg)),
                   math.radians(float(prime_meridian_epoch_deg)))

    @classmethod
    def from_period(cls, sidereal_period_s: float, **angles_deg) -> "Rotation":
        """From the sidereal period in seconds, which must be positive."""
        t = float(sidereal_period_s)
        if not (math.isfinite(t) and t > 0.0):
            raise ValueError(f"sidereal_period_s must be > 0, got {t}")
        return cls.of(_TWO_PI / t, **angles_deg)

    @classmethod
    def resonant(cls, orbital_period_s: float, spins_per_orbit,
                 **angles_deg) -> "Rotation":
        """Spin-orbit resonance: exactly ``spins_per_orbit`` sidereal
        rotations per orbital period, given as a rational (Fraction(3, 2)
        for Mercury, 1 for synchronous rotation). The rate is derived from
        the orbit's own period, so the ratio holds by construction."""
        ratio = Fraction(spins_per_orbit)
        p = float(orbital_period_s)
        if not (math.isfinite(p) and p > 0.0 and ratio > 0):
            raise ValueError("need orbital_period_s > 0 and a positive ratio")
        return cls.of(float(ratio) * _TWO_PI / p, **angles_deg)

    @classmethod
    def synchronous(cls, orbital_period_s: float, **angles_deg) -> "Rotation":
        """1:1 spin-orbit lock."""
        return cls.resonant(orbital_period_s, 1, **angles_deg)

    # -- derived -----------------------------------------------------------------

    @property
    def obliquity_deg(self) -> float:
        return math.degrees(self.obliquity_rad)

    @property
    def sidereal_period_s(self) -> float:
        """2 pi / Omega; infinite for a non-rotating planet."""
        w = self.angular_speed
        return math.inf if w == 0.0 else _TWO_PI / w

    @property
    def is_retrograde(self) -> bool:
        """True when the spin opposes the orbital motion (obliquity > 90 deg)."""
        return self.obliquity_rad > 0.5 * math.pi

    def prime_meridian_angle(self, t: float) -> float:
        """W(t) = W_0 + Omega t, from absolute time."""
        return self.prime_meridian_epoch_rad + self.angular_speed * float(t)

    def body_to_orbit(self, t: float) -> np.ndarray:
        """R_BO(t): columns are the body axes expressed in the orbit frame."""
        return (rotation_z(self.equinox_longitude_rad)
                @ rotation_x(self.obliquity_rad)
                @ rotation_z(self.prime_meridian_angle(t)))

    def with_(self, **changes) -> "Rotation":
        return replace(self, **changes)

    def to_dict(self) -> dict:
        return {"angular_speed": self.angular_speed,
                "obliquity_rad": self.obliquity_rad,
                "equinox_longitude_rad": self.equinox_longitude_rad,
                "prime_meridian_epoch_rad": self.prime_meridian_epoch_rad}
