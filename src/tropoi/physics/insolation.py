"""Stellar illumination derived from orbit and rotation geometry.

Illumination at a place and time comes from transforming vectors between
frames. Nothing about it is prescribed directly.

* The planet moves on a :class:`~tropoi.physics.orbit.KeplerOrbit` about
  the inertial origin, propagated from absolute time.
* Each :class:`LightSource` (a star, or more than one) has a luminosity and
  an inertial position. A source either sits at the origin or follows its
  own Kepler orbit about it.
* The planet's :class:`~tropoi.physics.rotation.Rotation` gives the
  body-to-orbit-frame matrix. The orbit gives the orbit-frame-to-inertial
  matrix. Together they make ``R_BI(t)``.

For each source j the direction in the body frame is
``s_j = R_BI(t)^T (r_j - r_p) / |r_j - r_p|`` and the irradiance is
``F_j = L_j / (4 pi |r_j - r_p|^2)``. A surface with unit normal n
(body frame) receives ``sum_j F_j max(0, n . s_j)``.

Tidal locking, eccentric orbits, spin-orbit resonances, retrograde spin
(obliquity above 90 degrees), obliquity and inclination are all
*configurations* of this one calculation, not separate algorithms. The
conventions are in :mod:`tropoi.physics.orbit` and
:mod:`tropoi.physics.rotation`, and are verified in
``tests/test_illumination_geometry.py``.

Insolation geometries return the pair the radiation scheme needs per
column: ``toa_flux`` (W m^-2 through a horizontal surface) and ``mu0``.

* :class:`OrbitalInsolation` gives instantaneous illumination from the
  geometry above.
* :class:`DiurnalMeanInsolation` is a deliberate approximation. It
  averages over one rotation at the instantaneous declination and distance,
  both taken from the same geometry.
* :class:`PrescribedInsolation` is any stated function of latitude, for
  example the net surface flux of Frierson et al. (2006).

:func:`illumination_from_spec` builds a planet-centric
:class:`StellarIllumination` from the configuration form used by
experiments.

Import-light: NumPy only. The latitude and longitude arrays may be NumPy or
CuPy.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Callable, Optional

import numpy as np

from tropoi.physics.constants import AU_M
from tropoi.physics.orbit import KeplerOrbit, mean_from_true, true_from_mean
from tropoi.physics.rotation import Rotation
from tropoi.physics.star import Star

__all__ = ["DiurnalMeanInsolation", "LightSource", "OrbitalInsolation",
           "PrescribedInsolation", "StellarIllumination",
           "frierson_net_surface_flux", "illumination_from_spec",
           "insolation_from_spec"]

_TWO_PI = 2.0 * math.pi


def _xp(*arrays):
    for a in arrays:
        if type(a).__module__.split(".")[0] == "cupy":
            import cupy  # noqa: WPS433
            return cupy
    return np


def _unit_normals(latitude, longitude):
    """Unit surface normals in the body frame, east-positive longitude."""
    xp = _xp(latitude, longitude)
    cl = xp.cos(latitude)
    return cl * xp.cos(longitude), cl * xp.sin(longitude), xp.sin(latitude)


@dataclass(frozen=True)
class LightSource:
    """A luminous point source.

    The source sits at the inertial origin when ``orbit`` is None.
    Otherwise it follows ``orbit`` about the origin with period
    ``period_s``. The epoch t = 0 is shared with the planet."""

    luminosity_w: float
    name: str = "star"
    orbit: Optional[KeplerOrbit] = None
    period_s: Optional[float] = None

    def __post_init__(self):
        if not (math.isfinite(self.luminosity_w) and self.luminosity_w >= 0.0):
            raise ValueError("luminosity_w must be finite and >= 0")
        if (self.orbit is None) != (self.period_s is None):
            raise ValueError("an orbiting source needs both orbit and period_s")

    def position(self, t: float) -> np.ndarray:
        if self.orbit is None:
            return np.zeros(3)
        return self.orbit.position_at(self.orbit.true_anomaly_at(
            t, period_s=self.period_s))

    def to_dict(self) -> dict:
        return {"name": self.name, "luminosity_w": self.luminosity_w,
                "orbit": None if self.orbit is None else self.orbit.to_dict(),
                "period_s": self.period_s}


@dataclass(frozen=True)
class StellarIllumination:
    """Geometry of a planet's illumination: the planet's orbit (period
    ``period_s``) about the inertial origin, its rotation, and one or more
    light sources. ``sources[0]`` is the primary; :meth:`flux_at`,
    :meth:`declination` and :meth:`subsolar_point` refer to it unless told
    otherwise.

    ``star`` is provenance only: the Star object the primary came from, if
    it came from one."""

    orbit: KeplerOrbit
    period_s: float
    rotation: Rotation
    sources: tuple = ()
    star: Optional[Star] = field(default=None, compare=False)

    def __post_init__(self):
        if not (math.isfinite(self.period_s) and self.period_s > 0.0):
            raise ValueError("period_s must be > 0")
        if not self.sources:
            raise ValueError("at least one light source is required")
        object.__setattr__(self, "sources", tuple(self.sources))

    # -- factories --------------------------------------------------------------

    @classmethod
    def from_components(cls, star: Star, orbit: KeplerOrbit,
                        rotation: Rotation, *,
                        gravitational_parameter: float | None = None,
                        irradiance_at_semimajor: float | None = None
                        ) -> "StellarIllumination":
        """A planet orbiting ``star``, which sits at the origin. The period
        follows from Kepler's third law, with mu taken either from
        ``gravitational_parameter`` or from the star's mass. If
        ``irradiance_at_semimajor`` is given, it sets the source luminosity
        (L = 4 pi a^2 S); the star's own luminosity is then provenance
        only."""
        mu = gravitational_parameter
        if mu is None:
            mu = star.gravitational_parameter
        if mu is None:
            raise ValueError(
                f"propagating the orbit needs the gravitational parameter: "
                f"the star {star.name!r} has no mass; pass "
                "gravitational_parameter explicitly")
        if irradiance_at_semimajor is None:
            lum = star.luminosity_w
        else:
            a = orbit.semimajor_axis_m
            lum = 4.0 * math.pi * a * a * float(irradiance_at_semimajor)
        return cls(orbit, orbit.period(mu), rotation,
                   (LightSource(lum, star.name),), star=star)

    @classmethod
    def planet_centric(cls, *, irradiance_at_semimajor: float, period_s: float,
                       eccentricity: float = 0.0, obliquity_deg: float = 0.0,
                       perihelion_ls_deg: float = 0.0,
                       time_of_perihelion_s: float = 0.0,
                       angular_speed: float = 0.0,
                       prime_meridian_epoch_deg: float = 0.0,
                       semimajor_axis_m: float = AU_M,
                       star: Optional[Star] = None) -> "StellarIllumination":
        """The planet-centric form that experiment configurations use.

        The single star sits at the origin. The orbit plane is the
        reference plane, with the equatorial node along +x, so
        ``L_s = u``. Periapsis lies at L_s = ``perihelion_ls_deg``, and
        periapsis passage is at ``time_of_perihelion_s``. The irradiance is
        stated at the semimajor-axis distance; the absolute value of
        ``semimajor_axis_m`` only scales the luminosity."""
        e = float(eccentricity)
        p = float(period_s)
        nu0 = true_from_mean(-_TWO_PI * float(time_of_perihelion_s) / p, e)
        orbit = KeplerOrbit(float(semimajor_axis_m), e,
                            periapsis_longitude_rad=math.radians(
                                float(perihelion_ls_deg)),
                            true_anomaly_epoch_rad=nu0)
        rotation = Rotation.of(angular_speed, obliquity_deg=obliquity_deg,
                               prime_meridian_epoch_deg=prime_meridian_epoch_deg)
        a = orbit.semimajor_axis_m
        lum = 4.0 * math.pi * a * a * float(irradiance_at_semimajor)
        return cls(orbit, p, rotation,
                   (LightSource(lum, "star" if star is None else star.name),),
                   star=star)

    def with_rotation(self, rotation: Rotation) -> "StellarIllumination":
        return StellarIllumination(self.orbit, self.period_s, rotation,
                                   self.sources, self.star)

    def with_subsolar_longitude_at(self, t: float, longitude_rad: float
                                   ) -> "StellarIllumination":
        """The same geometry, with the prime meridian rephased so that the
        primary's subsolar longitude at time ``t`` is ``longitude_rad``."""
        _, lon = self.subsolar_point(t)
        r = self.rotation
        w0 = r.prime_meridian_epoch_rad + (lon - float(longitude_rad))
        return self.with_rotation(r.with_(prime_meridian_epoch_rad=w0))

    # -- orbit --------------------------------------------------------------------

    @property
    def mean_motion(self) -> float:
        return _TWO_PI / self.period_s

    @property
    def eccentricity(self) -> float:
        return self.orbit.eccentricity

    def true_anomaly(self, t: float) -> float:
        return self.orbit.true_anomaly_at(t, period_s=self.period_s)

    def planet_position(self, t: float) -> np.ndarray:
        return self.orbit.position_at(self.true_anomaly(t))

    def solar_longitude(self, t: float) -> float:
        """L_s = u - lambda_eq in [0, 2 pi): the season, relative to the
        primary at the origin."""
        u = self.orbit.argument_of_periapsis_rad + self.true_anomaly(t)
        return (u - self.rotation.equinox_longitude_rad) % _TWO_PI

    # -- frames and beams ---------------------------------------------------------

    def body_to_inertial(self, t: float) -> np.ndarray:
        return self.orbit.orbit_to_inertial() @ self.rotation.body_to_orbit(t)

    def beams(self, t: float):
        """``(directions, fluxes)``. Each direction is the unit vector, in
        the body frame, from the planet to a source (shape ``(n, 3)``);
        each flux is the irradiance facing that source (shape ``(n,)``)."""
        rp = self.planet_position(t)
        rbi = self.body_to_inertial(t)
        dirs = np.empty((len(self.sources), 3))
        flux = np.empty(len(self.sources))
        for j, src in enumerate(self.sources):
            d = src.position(t) - rp
            dist2 = float(d @ d)
            dist = math.sqrt(dist2)
            dirs[j] = rbi.T @ (d / dist)
            flux[j] = src.luminosity_w / (4.0 * math.pi * dist2)
        return dirs, flux

    def star_direction(self, t: float, source: int = 0) -> np.ndarray:
        return self.beams(t)[0][source]

    def flux_at(self, t: float, source: int = 0) -> float:
        """Irradiance facing the source at the planet's distance."""
        return float(self.beams(t)[1][source])

    def declination(self, t: float, source: int = 0) -> float:
        s = self.star_direction(t, source)
        return math.asin(max(-1.0, min(1.0, float(s[2]))))

    def subsolar_point(self, t: float, source: int = 0):
        """(latitude, east longitude in [-pi, pi)) of the sub-source point."""
        s = self.star_direction(t, source)
        lat = math.asin(max(-1.0, min(1.0, float(s[2]))))
        lon = math.atan2(float(s[1]), float(s[0]))
        return lat, (lon + math.pi) % _TWO_PI - math.pi

    @property
    def mean_solar_day(self) -> float:
        """Long-term mean solar day relative to the central primary:
        2 pi / |Omega - n sign(cos epsilon)|. Over one orbit the primary's
        right ascension advances by one full turn (prograde for
        obliquity below 90 degrees, retrograde above), whatever the
        eccentricity. This is a mean only: in an eccentric or oblique
        configuration the instantaneous solar day varies. Returns
        infinity for a synchronous lock, and NaN at an obliquity of
        exactly 90 degrees. An orbiting primary is unsupported."""
        if self.sources[0].orbit is not None:
            raise ValueError("mean solar day requires a central primary")
        eps = self.rotation.obliquity_rad
        c = math.cos(eps)
        if abs(c) < 1e-15:
            return math.nan
        rel = self.rotation.angular_speed - self.mean_motion * math.copysign(1.0, c)
        return math.inf if rel == 0.0 else _TWO_PI / abs(rel)

    # -- description ----------------------------------------------------------------

    @property
    def obliquity_deg(self) -> float:
        return self.rotation.obliquity_deg

    def as_dict(self) -> dict:
        return {"orbit": self.orbit.to_dict(), "period_s": self.period_s,
                "rotation": self.rotation.to_dict(),
                "sources": [s.to_dict() for s in self.sources],
                "star": None if self.star is None else self.star.to_dict()}


class OrbitalInsolation:
    """Instantaneous illumination from :class:`StellarIllumination`.

    ``toa_flux = sum_j F_j max(0, n . s_j)``. For a single source ``mu0``
    is ``max(0, n . s)``. With several sources ``mu0`` is the
    irradiance-weighted mean cosine over the sources above the horizon
    (``toa / sum F_j`` over those sources, 0 at night). That is a gray
    single-beam reduction: correct for the bolometric energy budget, but
    not a substitute for per-beam radiative transfer."""

    def __init__(self, illumination: StellarIllumination):
        self.illumination = illumination

    def signature(self) -> dict:
        return {"kind": "orbital", "illumination": self.illumination.as_dict()}

    def incident(self, nx, ny, nz, t: float):
        """(toa_flux, mu0) for body-frame unit normals ``(nx, ny, nz)``."""
        xp = _xp(nx)
        dirs, flux = self.illumination.beams(t)
        total = 0.0 * nx
        lit = 0.0 * nx
        for (sx, sy, sz), f in zip(dirs.tolist(), flux.tolist()):
            mu = xp.maximum(nx * sx + ny * sy + nz * sz, 0.0)
            total = total + f * mu
            lit = lit + xp.where(mu > 0.0, f, 0.0)
        if len(flux) == 1:
            return total, total / float(flux[0]) if flux[0] > 0 else lit
        mu0 = xp.where(lit > 0.0, total / xp.where(lit > 0.0, lit, 1.0), 0.0)
        return total, mu0

    def __call__(self, latitude, longitude, t: float):
        return self.incident(*_unit_normals(latitude, longitude), t)


class DiurnalMeanInsolation:
    """Mean insolation over one rotation, at the instantaneous declination
    and distance of the primary:
    ``(F / pi) (h0 sin(lat) sin(dec) + cos(lat) cos(dec) sin(h0))``.
    ``mu0`` is the daylight mean of the zenith cosine. This is a deliberate
    approximation for problems whose dynamics do not resolve the diurnal
    cycle."""

    def __init__(self, illumination: StellarIllumination):
        self.illumination = illumination

    def signature(self) -> dict:
        return {"kind": "diurnal_mean",
                "illumination": self.illumination.as_dict()}

    def __call__(self, latitude, longitude, t: float):
        xp = _xp(latitude)
        dirs, flux = self.illumination.beams(t)
        dec = math.asin(max(-1.0, min(1.0, float(dirs[0, 2]))))
        s = float(flux[0])
        cos_h0 = xp.clip(-xp.tan(latitude) * math.tan(dec), -1.0, 1.0)
        h0 = xp.arccos(cos_h0)
        integral = (h0 * xp.sin(latitude) * math.sin(dec)
                    + xp.cos(latitude) * math.cos(dec) * xp.sin(h0))
        integral = xp.maximum(integral, 0.0)
        mu0 = xp.where(h0 > 0.0, integral / xp.maximum(h0, 1e-300), 0.0)
        return s * integral / math.pi, mu0


class PrescribedInsolation:
    """``toa_flux = function(latitude)``, with a fixed ``mu0`` (which does
    not matter for a transparent atmosphere)."""

    def __init__(self, function: Callable, name: str, params: dict, mu0: float = 0.5):
        self.function = function
        self.name = name
        self.params = dict(params)
        self.mu0 = float(mu0)

    def signature(self) -> dict:
        return {"kind": "prescribed", "name": self.name, "params": self.params,
                "mu0": self.mu0}

    def __call__(self, latitude, longitude, t: float):
        flux = self.function(latitude)
        return flux, self.mu0 + 0.0 * flux


def frierson_net_surface_flux(net_solar_constant: float = 938.4,
                              delta_s: float = 1.4) -> PrescribedInsolation:
    """FHZ06 eqs. 2-3: ``R_S = R_S0 [1 + Delta_s p2(lat)]``, with
    ``p2 = (1 - 3 sin^2 lat) / 4`` and all of it absorbed at the surface.

    Table 1 of the paper lists 938.4 W m^-2 as the "net solar constant
    (includes albedo)", while the text calls ``R_S0`` the global-mean net
    solar flux. Because ``p2`` has zero global mean, ``R_S0`` is the global
    mean, and 938.4 is taken as four times it (938.4 / 4 = 234.6 W m^-2,
    i.e. 1360 W m^-2 with albedo 0.31). That is the only reading that gives
    an Earth-like energy balance.
    """
    r_s0 = float(net_solar_constant) / 4.0
    ds = float(delta_s)

    def flux(latitude):
        xp = _xp(latitude)
        s = xp.sin(latitude)
        return r_s0 * (1.0 + ds * 0.25 * (1.0 - 3.0 * s * s))

    return PrescribedInsolation(flux, "frierson_net_surface_flux",
                                {"net_solar_constant": float(net_solar_constant),
                                 "delta_s": ds})


# -- configuration form -------------------------------------------------------------

_SPIN_KEYS = ("sidereal_period", "mean_solar_day", "spins_per_orbit")


def _spin_rate(spin: dict, period: float, obliquity_deg: float) -> float:
    given = [k for k in _SPIN_KEYS if k in spin]
    if len(given) != 1:
        raise ValueError(f"spin must give exactly one of {_SPIN_KEYS}, got "
                         f"{sorted(spin)}")
    n = _TWO_PI / period
    if "sidereal_period" in spin:
        return _TWO_PI / float(spin["sidereal_period"])
    if "spins_per_orbit" in spin:
        ratio = spin["spins_per_orbit"]
        ratio = (Fraction(*ratio) if isinstance(ratio, (list, tuple))
                 else Fraction(ratio))
        return float(ratio) * n
    c = math.cos(math.radians(obliquity_deg))
    if abs(c) < 1e-15:
        raise ValueError("a mean solar day is undefined at 90 deg obliquity")
    return _TWO_PI / float(spin["mean_solar_day"]) + n * math.copysign(1.0, c)


def illumination_from_spec(spec: dict) -> StellarIllumination:
    """Planet-centric illumination from its configuration form::

        {"stellar_constant": S at r = a (W m^-2), "period": s,
         "eccentricity": e, "obliquity_deg": deg (0-180),
         "perihelion_longitude_deg": L_s at perihelion,
         "time_of_perihelion": s,
         "spin": {exactly one of "sidereal_period": s,
                  "mean_solar_day": s, "spins_per_orbit": p or [p, q]},
                 optional "subsolar_longitude_at_epoch_deg"}

    Without a ``spin`` the planet does not rotate. That is adequate only
    for :class:`DiurnalMeanInsolation`, which needs the axis but not the
    spin.
    """
    allowed = {"kind", "stellar_constant", "period", "eccentricity",
               "obliquity_deg", "perihelion_longitude_deg",
               "time_of_perihelion", "spin"}
    unknown = set(spec) - allowed
    if unknown:
        raise ValueError(f"unknown illumination options {sorted(unknown)}")
    period = float(spec["period"])
    obliquity = float(spec.get("obliquity_deg", 0.0))
    spin = dict(spec["spin"]) if "spin" in spec else None
    target = None
    omega = 0.0
    if spin is not None:
        target = spin.pop("subsolar_longitude_at_epoch_deg", None)
        unknown = set(spin) - set(_SPIN_KEYS)
        if unknown:
            raise ValueError(f"unknown spin options {sorted(unknown)}")
        omega = _spin_rate(spin, period, obliquity)
    ill = StellarIllumination.planet_centric(
        irradiance_at_semimajor=float(spec["stellar_constant"]),
        period_s=period, eccentricity=float(spec.get("eccentricity", 0.0)),
        obliquity_deg=obliquity,
        perihelion_ls_deg=float(spec.get("perihelion_longitude_deg", 0.0)),
        time_of_perihelion_s=float(spec.get("time_of_perihelion", 0.0)),
        angular_speed=omega)
    if target is not None:
        ill = ill.with_subsolar_longitude_at(0.0, math.radians(float(target)))
    return ill


def insolation_from_spec(spec: Optional[dict]):
    """Build the insolation of an experiment configuration (None stays
    None).

    Kinds:

    * ``orbital``: instantaneous illumination (:class:`OrbitalInsolation`).
    * ``diurnal_mean``: rotation-averaged illumination.
    * ``frierson``: the FHZ06 prescribed net flux.
    * ``uniform``: a constant flux.
    """
    if spec is None:
        return None
    kind = spec["kind"]
    if kind == "frierson":
        opts = {k: v for k, v in spec.items() if k != "kind"}
        return frierson_net_surface_flux(**opts)
    if kind == "uniform":
        flux = float(spec["flux"])
        return PrescribedInsolation(lambda lat: flux + 0.0 * lat, "uniform",
                                    {"flux": flux})
    if kind == "orbital":
        if not spec.get("spin"):
            raise ValueError("orbital insolation needs a spin")
        return OrbitalInsolation(illumination_from_spec(spec))
    if kind == "diurnal_mean":
        return DiurnalMeanInsolation(illumination_from_spec(spec))
    raise ValueError(
        f"unknown insolation kind {kind!r}. The 'instantaneous' and "
        "'tidally_locked' kinds were replaced by 'orbital' (see "
        "docs/radiation/SHARED_PHYSICS.md); archived configurations that use "
        "them are reproduced at tag substrate-cpu-verified-v1")
