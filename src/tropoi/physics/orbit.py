"""A bound two-body Kepler orbit: canonical elements and propagation.

Canonical state: six Keplerian elements of a bound ellipse around a
primary —

* ``semimajor_axis_m`` (a), ``eccentricity`` (e, dimensionless, 0 <= e < 1);
* ``inclination_rad`` (i), ``ascending_node_rad`` (Omega),
  ``periapsis_longitude_rad`` (the longitude of periapsis
  varpi = Omega + omega);
* ``true_anomaly_epoch_rad`` (nu_0): the true anomaly at the reference
  epoch.

Units are SI throughout: metres, seconds, radians.

**Frame and epoch conventions.** The frame is a right-handed inertial frame
centred on the primary, chosen by the caller: its x-y plane is the
reference plane, its +x axis the reference direction from which all
longitudes are measured, and +z completes the right-handed set.
Inclination is measured from the reference plane (0 <= i <= pi; i > pi/2
is retrograde). The reference epoch is ``t = 0`` of whatever clock the
caller propagates with. For a planet, a convenient choice is the orbital
plane itself as the reference plane with +x toward the northern spring
equinox; :mod:`tropoi.physics.planets` uses exactly that.

**Degenerate cases.** For a circular orbit (e = 0) the periapsis is
undefined; by convention the stated ``periapsis_longitude_rad`` still fixes
the direction from which the true anomaly is counted, and every formula is
regular there. For an equatorial orbit (i = 0 or pi) the node is undefined;
by convention ``ascending_node_rad = 0`` and the longitude of periapsis is
then measured entirely in the reference plane.

**Propagation.** Kepler's problem needs the gravitational parameter
``mu = G (M + m)`` of the pair, which an orbit does not own: pass it
explicitly to :meth:`period`, :meth:`true_anomaly_at`, or let a containing
system (such as :class:`tropoi.physics.insolation.StellarIllumination`,
which can take it from its star) supply it. Mean anomaly advances linearly
from the epoch value implied by ``true_anomaly_epoch_rad``; Kepler's
equation is solved by Newton iteration.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from tropoi.physics.constants import AU_M

__all__ = ["KeplerOrbit", "eccentric_from_true", "mean_from_true",
           "true_from_eccentric", "true_from_mean"]

_TWO_PI = 2.0 * math.pi


def eccentric_from_true(nu: float, e: float) -> float:
    """Eccentric anomaly E from true anomaly (radians)."""
    return 2.0 * math.atan2(math.sqrt(1.0 - e) * math.sin(0.5 * nu),
                            math.sqrt(1.0 + e) * math.cos(0.5 * nu))


def true_from_eccentric(ecc: float, e: float) -> float:
    """True anomaly nu from eccentric anomaly (radians)."""
    return 2.0 * math.atan2(math.sqrt(1.0 + e) * math.sin(0.5 * ecc),
                            math.sqrt(1.0 - e) * math.cos(0.5 * ecc))


def mean_from_true(nu: float, e: float) -> float:
    """Mean anomaly M from true anomaly (Kepler's equation, radians)."""
    ecc = eccentric_from_true(nu, e)
    return ecc - e * math.sin(ecc)


def true_from_mean(mean: float, e: float, *, iterations: int = 50) -> float:
    """True anomaly nu from mean anomaly: Newton on Kepler's equation."""
    mean = mean % _TWO_PI
    ecc = mean
    for _ in range(iterations):
        ecc -= (ecc - e * math.sin(ecc) - mean) / (1.0 - e * math.cos(ecc))
    return true_from_eccentric(ecc, e)


def _finite(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}")
    return value


@dataclass(frozen=True)
class KeplerOrbit:
    """Six Keplerian elements of a bound two-body ellipse (SI; see the
    module docstring for frame, epoch and degenerate-case conventions)."""

    semimajor_axis_m: float
    eccentricity: float = 0.0
    inclination_rad: float = 0.0
    ascending_node_rad: float = 0.0
    periapsis_longitude_rad: float = 0.0
    true_anomaly_epoch_rad: float = 0.0

    def __post_init__(self) -> None:
        a = _finite("semimajor_axis_m", self.semimajor_axis_m)
        if not a > 0.0:
            raise ValueError(f"semimajor_axis_m must be > 0, got {a}")
        e = _finite("eccentricity", self.eccentricity)
        if not 0.0 <= e < 1.0:
            raise ValueError(f"eccentricity must be in [0, 1) for a bound "
                             f"ellipse, got {e}")
        i = _finite("inclination_rad", self.inclination_rad)
        if not 0.0 <= i <= math.pi:
            raise ValueError(f"inclination_rad must be in [0, pi], got {i}")
        for name in ("ascending_node_rad", "periapsis_longitude_rad",
                     "true_anomaly_epoch_rad"):
            _finite(name, getattr(self, name))

    # -- factories --------------------------------------------------------------

    @classmethod
    def from_au(cls, semimajor_axis_au: float, **elements) -> "KeplerOrbit":
        """The same elements with the semimajor axis in astronomical units
        (IAU 2012 au)."""
        return cls(float(semimajor_axis_au) * AU_M, **elements)

    @classmethod
    def from_apsides(cls, periapsis_m: float, apoapsis_m: float, **elements
                     ) -> "KeplerOrbit":
        """From periapsis and apoapsis distances: a = (r_p + r_a) / 2,
        e = (r_a - r_p) / (r_a + r_p)."""
        rp, ra = float(periapsis_m), float(apoapsis_m)
        if not (math.isfinite(rp) and math.isfinite(ra) and 0.0 < rp <= ra):
            raise ValueError(f"need 0 < periapsis <= apoapsis, got "
                             f"{rp}, {ra}")
        if "eccentricity" in elements:
            raise ValueError("the apsides already fix the eccentricity")
        return cls(0.5 * (rp + ra), (ra - rp) / (ra + rp), **elements)

    @classmethod
    def from_period(cls, period_s: float, gravitational_parameter: float,
                    **elements) -> "KeplerOrbit":
        """From the orbital period and mu = G (M + m):
        a = (mu (T / 2 pi)^2)^(1/3) (Kepler's third law)."""
        t, mu = float(period_s), float(gravitational_parameter)
        if not (math.isfinite(t) and t > 0.0 and math.isfinite(mu)
                and mu > 0.0):
            raise ValueError("period_s and gravitational_parameter must be "
                             f"> 0, got {t}, {mu}")
        return cls((mu * (t / _TWO_PI) ** 2) ** (1.0 / 3.0), **elements)

    @classmethod
    def from_irradiance(cls, irradiance_w_m2: float, luminosity_w: float, *,
                        reference: str, **elements) -> "KeplerOrbit":
        """From a stated stellar irradiance S and the star's luminosity L.

        ``reference`` MUST say which distance the irradiance refers to —
        there is no default:

        * ``"semimajor_axis"``: S is the flux at r = a;
        * ``"periapsis"`` / ``"apoapsis"``: S is the flux at closest /
          farthest approach;
        * ``"time_mean"``: S is the time-averaged flux over one orbit,
          <r^-2> = 1 / (a^2 sqrt(1 - e^2)).

        The distance is d = sqrt(L / (4 pi S)); a follows from d, e and the
        stated reference.
        """
        s, lum = float(irradiance_w_m2), float(luminosity_w)
        if not (math.isfinite(s) and s > 0.0 and math.isfinite(lum)
                and lum > 0.0):
            raise ValueError("irradiance_w_m2 and luminosity_w must be > 0")
        e = float(elements.get("eccentricity", 0.0))
        d = math.sqrt(lum / (4.0 * math.pi * s))
        if reference == "semimajor_axis":
            a = d
        elif reference == "periapsis":
            a = d / (1.0 - e)
        elif reference == "apoapsis":
            a = d / (1.0 + e)
        elif reference == "time_mean":
            a = d / (1.0 - e * e) ** 0.25
        else:
            raise ValueError(
                f"unknown irradiance reference {reference!r}; say explicitly "
                "which distance the irradiance refers to: 'semimajor_axis', "
                "'periapsis', 'apoapsis' or 'time_mean'")
        return cls(a, **elements)

    # -- geometry ---------------------------------------------------------------

    @property
    def periapsis_m(self) -> float:
        return self.semimajor_axis_m * (1.0 - self.eccentricity)

    @property
    def apoapsis_m(self) -> float:
        return self.semimajor_axis_m * (1.0 + self.eccentricity)

    @property
    def argument_of_periapsis_rad(self) -> float:
        """omega = varpi - Omega."""
        return self.periapsis_longitude_rad - self.ascending_node_rad

    def radius_at(self, true_anomaly_rad: float) -> float:
        """r(nu) = a (1 - e^2) / (1 + e cos nu)."""
        e = self.eccentricity
        return (self.semimajor_axis_m * (1.0 - e * e)
                / (1.0 + e * math.cos(true_anomaly_rad)))

    def position_at(self, true_anomaly_rad: float) -> np.ndarray:
        """Position vector (m) in the reference frame at true anomaly nu:
        the standard rotation by (Omega, i, omega) of the in-plane ellipse."""
        nu = float(true_anomaly_rad)
        r = self.radius_at(nu)
        u = self.argument_of_periapsis_rad + nu      # argument of latitude
        co, so = math.cos(self.ascending_node_rad), math.sin(self.ascending_node_rad)
        cu, su = math.cos(u), math.sin(u)
        ci, si = math.cos(self.inclination_rad), math.sin(self.inclination_rad)
        return r * np.array([co * cu - so * su * ci,
                             so * cu + co * su * ci,
                             su * si])

    # -- propagation -------------------------------------------------------------

    def period(self, gravitational_parameter: float) -> float:
        """T = 2 pi sqrt(a^3 / mu)."""
        mu = float(gravitational_parameter)
        if not (math.isfinite(mu) and mu > 0.0):
            raise ValueError(f"gravitational_parameter must be > 0, got {mu}")
        return _TWO_PI * math.sqrt(self.semimajor_axis_m ** 3 / mu)

    def mean_motion(self, gravitational_parameter: float) -> float:
        return _TWO_PI / self.period(gravitational_parameter)

    @property
    def mean_anomaly_epoch_rad(self) -> float:
        """The mean anomaly at the epoch, from ``true_anomaly_epoch_rad``."""
        return mean_from_true(self.true_anomaly_epoch_rad, self.eccentricity)

    def true_anomaly_at(self, t: float, *,
                        gravitational_parameter: float | None = None,
                        period_s: float | None = None) -> float:
        """True anomaly at time ``t`` (s) past the epoch. Timing needs one of
        ``gravitational_parameter`` or ``period_s`` (both given must agree
        is not checked; give one)."""
        if (gravitational_parameter is None) == (period_s is None):
            raise ValueError("give exactly one of gravitational_parameter "
                             "or period_s to propagate")
        period = (float(period_s) if period_s is not None
                  else self.period(gravitational_parameter))
        mean = self.mean_anomaly_epoch_rad + _TWO_PI * (float(t) / period)
        return true_from_mean(mean, self.eccentricity)

    def radius_at_time(self, t: float, **timing) -> float:
        return self.radius_at(self.true_anomaly_at(t, **timing))

    # -- misc ---------------------------------------------------------------------

    def with_(self, **changes) -> "KeplerOrbit":
        return replace(self, **changes)

    def to_dict(self) -> dict:
        return {"semimajor_axis_m": self.semimajor_axis_m,
                "eccentricity": self.eccentricity,
                "inclination_rad": self.inclination_rad,
                "ascending_node_rad": self.ascending_node_rad,
                "periapsis_longitude_rad": self.periapsis_longitude_rad,
                "true_anomaly_epoch_rad": self.true_anomaly_epoch_rad}
