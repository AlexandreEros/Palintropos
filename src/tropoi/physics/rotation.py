"""Planetary rotation: spin-axis orientation, angular velocity and phase.

:class:`Rotation` owns the planet's spin state — nothing orbital and
nothing stellar:

* ``angular_velocity`` (rad/s, signed; negative = retrograde): the sidereal
  rotation rate about the spin axis;
* the spin-axis orientation, as ``obliquity_rad`` (the tilt of the axis
  from the orbit normal) and ``equinox_longitude_rad`` (where along the
  orbit the axis orientation puts the northern spring equinox: the
  longitude, measured in the orbit's reference frame from its reference
  direction, at which the star stands over the equator moving north). With
  the frame convention of :mod:`tropoi.physics.planets` — reference
  direction at the northern spring equinox — this is 0;
* ``phase_epoch_rad``: the rotation angle of the prime meridian at the
  epoch ``t = 0`` (the same epoch as the orbit's).

Illumination at a place and time — declination, hour angle, zenith angle —
is a *combination* of star, orbit, rotation, location and time; it lives in
:mod:`tropoi.physics.insolation`, not here.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

__all__ = ["Rotation"]


@dataclass(frozen=True)
class Rotation:
    """Spin state of a planet (SI: rad/s and radians)."""

    angular_velocity: float
    obliquity_rad: float = 0.0
    equinox_longitude_rad: float = 0.0
    phase_epoch_rad: float = 0.0

    def __post_init__(self) -> None:
        for name in ("angular_velocity", "obliquity_rad",
                     "equinox_longitude_rad", "phase_epoch_rad"):
            v = float(getattr(self, name))
            if not math.isfinite(v):
                raise ValueError(f"{name} must be finite, got {v}")
        if not 0.0 <= self.obliquity_rad <= math.pi:
            raise ValueError(f"obliquity_rad must be in [0, pi], got "
                             f"{self.obliquity_rad}")

    @classmethod
    def of(cls, angular_velocity: float, *, obliquity_deg: float = 0.0,
           equinox_longitude_deg: float = 0.0, phase_epoch_deg: float = 0.0
           ) -> "Rotation":
        """The same state with the angles in degrees."""
        return cls(float(angular_velocity),
                   math.radians(float(obliquity_deg)),
                   math.radians(float(equinox_longitude_deg)),
                   math.radians(float(phase_epoch_deg)))

    @classmethod
    def from_period(cls, sidereal_period_s: float, **angles_deg) -> "Rotation":
        """From the sidereal rotation period (s; negative = retrograde)."""
        t = float(sidereal_period_s)
        if not (math.isfinite(t) and t != 0.0):
            raise ValueError(f"sidereal_period_s must be nonzero, got {t}")
        return cls.of(2.0 * math.pi / t, **angles_deg)

    @property
    def obliquity_deg(self) -> float:
        return math.degrees(self.obliquity_rad)

    @property
    def sidereal_period_s(self) -> float:
        """2 pi / |angular velocity| (inf for a non-rotating planet)."""
        w = abs(self.angular_velocity)
        return math.inf if w == 0.0 else 2.0 * math.pi / w

    def with_(self, **changes) -> "Rotation":
        return replace(self, **changes)

    def to_dict(self) -> dict:
        return {"angular_velocity": self.angular_velocity,
                "obliquity_rad": self.obliquity_rad,
                "equinox_longitude_rad": self.equinox_longitude_rad,
                "phase_epoch_rad": self.phase_epoch_rad}
