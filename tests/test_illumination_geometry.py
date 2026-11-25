"""G-A: illumination derived from orbit and rotation geometry.

The gate numbers refer to docs/validation/PLANET-ONE-GATES.md (frozen
2026-11-24). The references are closed-form expressions written out
independently of tropoi.physics.insolation.
"""
from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest

from tropoi.physics.insolation import (DiurnalMeanInsolation, LightSource,
                                       OrbitalInsolation, StellarIllumination,
                                       illumination_from_spec,
                                       insolation_from_spec)
from tropoi.physics.orbit import KeplerOrbit, eccentric_from_mean
from tropoi.physics.rotation import Rotation

DAY = 86400.0
TWO_PI = 2.0 * math.pi
L_SUN = 3.828e26


def _wrap(x):
    return (x + math.pi) % TWO_PI - math.pi


def _geometry(*, e=0.0, period=10 * DAY, spins=None, obliquity_deg=0.0,
              inclination=0.0, node=0.0, varpi=0.0, nu0=0.0, equinox=0.0,
              w0=0.0, omega=None, sources=None, a=1.0e11):
    orbit = KeplerOrbit(a, e, inclination, node, varpi, nu0)
    if omega is None:
        omega = float(Fraction(spins)) * TWO_PI / period
    rot = Rotation(omega, math.radians(obliquity_deg), equinox, w0)
    return StellarIllumination(orbit, period, rot,
                               sources or (LightSource(L_SUN),))


# -- 1-3: Kepler propagation and orbit ------------------------------------------------

def test_kepler_residual():
    worst = 0.0
    for e in (0.0, 0.0167, 0.2056, 0.5, 0.9, 0.99):
        for m in np.linspace(0.0, TWO_PI, 997, endpoint=False):
            ecc = eccentric_from_mean(m, e)
            worst = max(worst, abs(_wrap(ecc - e * math.sin(ecc) - m)))
    assert worst <= 1e-14


@pytest.mark.parametrize("e", [0.0, 0.2056, 0.7])
def test_apsides_unit_directions_and_inverse_square(e):
    g = _geometry(e=e, spins=Fraction(3, 2), obliquity_deg=17.0,
                  inclination=0.4, node=1.1, varpi=2.0)
    a = g.orbit.semimajor_axis_m
    rp = np.linalg.norm(g.planet_position(0.0))
    ra = np.linalg.norm(g.planet_position(0.5 * g.period_s))
    assert abs(rp / (a * (1 - e)) - 1) <= 1e-14
    assert abs(ra / (a * (1 + e)) - 1) <= 1e-14
    for t in np.linspace(0, 3 * g.period_s, 101):
        dirs, flux = g.beams(t)
        r2 = float(g.planet_position(t) @ g.planet_position(t))
        assert abs(np.linalg.norm(dirs[0]) - 1.0) <= 1e-14
        assert abs(flux[0] * r2 * 4 * math.pi / L_SUN - 1.0) <= 1e-14


def test_orbit_element_conventions():
    node, inc, varpi = 0.7, 0.3, 1.9
    orbit = KeplerOrbit(1.0e11, 0.3, inc, node, varpi)
    omega = varpi - node
    # Ascending node: argument of latitude u = 0, i.e. nu = -omega.
    p = orbit.position_at(-omega)
    assert abs(p[2]) <= 1e-14 * 1e11
    assert abs(math.atan2(p[1], p[0]) - node) <= 1e-14
    # The planet climbs through the reference plane there.
    assert orbit.position_at(-omega + 1e-3)[2] > 0.0
    # Periapsis direction, measured from the node in the orbit plane, is omega.
    q = orbit.orbit_to_inertial().T @ orbit.position_at(0.0)
    assert abs(math.atan2(q[1], q[0]) - omega) <= 1e-14
    # Angular momentum has +z for i < 90 deg.
    h = np.cross(orbit.position_at(0.0), orbit.position_at(1e-3))
    assert h[2] > 0.0


# -- 4-7: spin, frames, orientation --------------------------------------------------

def test_fixed_subsolar_point_for_circular_prograde_synchronous_spin():
    g = _geometry(spins=1, w0=0.3)
    assert g.rotation == Rotation.synchronous(g.period_s, prime_meridian_epoch_deg=
                                              math.degrees(0.3))
    lat0, lon0 = g.subsolar_point(0.0)
    n = TWO_PI / g.period_s
    f0 = g.flux_at(0.0)
    for t in np.linspace(0.0, 1000 * g.period_s, 4001):
        lat, lon = g.subsolar_point(t)
        bound = 1e-15 * (1.0 + abs(n * t))
        assert abs(lat - lat0) <= bound
        assert abs(_wrap(lon - lon0)) <= bound
        assert abs(g.flux_at(t) / f0 - 1.0) <= 1e-14


@pytest.mark.parametrize("eps_deg", [23.44, 60.0, 98.0, 150.0])
def test_oblique_synchronous_seasons(eps_deg):
    eps = math.radians(eps_deg)
    g = _geometry(spins=1, obliquity_deg=eps_deg, w0=0.2)
    n = TWO_PI / g.period_s
    for t in np.linspace(0.0, 2 * g.period_s, 401):
        u = n * t                      # circular orbit, nu0 = varpi = 0
        ls = g.solar_longitude(t)
        assert abs(_wrap(ls - u)) <= 1e-12
        lat, lon = g.subsolar_point(t)
        # Closed form: s_O = -(cos u, sin u, 0); body = Rz(-W) Rx(-eps) s_O.
        lat_ref = math.asin(math.sin(eps) * math.sin(u))
        lon_ref = math.atan2(-math.sin(u) * math.cos(eps), -math.cos(u)) - (0.2 + n * t)
        assert abs(lat - lat_ref) <= 1e-13
        assert abs(_wrap(lon - lon_ref)) <= 1e-13
    # Northern summer at L_s = 90 deg (for eps < 90 deg; north is the spin pole).
    t90 = 0.25 * g.period_s
    assert g.declination(t90) == pytest.approx(eps if eps_deg <= 90 else math.pi - eps,
                                               abs=1e-12)


def test_common_inertial_rotation_leaves_body_illumination_unchanged():
    # The argument of periapsis omega = varpi - Omega is what the orbit
    # plane carries along, so a rotated description keeps omega fixed.
    base = dict(e=0.3, spins=Fraction(5, 3), obliquity_deg=33.0,
                nu0=0.9, equinox=0.5, w0=1.2)
    g1 = _geometry(**base, inclination=0.0, node=0.0, varpi=0.4)
    g2 = _geometry(**base, inclination=1.1, node=2.3, varpi=2.3 + 0.4)
    for t in np.linspace(0, 5 * g1.period_s, 211):
        d1, f1 = g1.beams(t)
        d2, f2 = g2.beams(t)
        assert np.max(np.abs(d1 - d2)) <= 1e-13
        assert abs(f1[0] / f2[0] - 1) <= 1e-13

    # Two sources: rotate the whole system (both orbits) about z by delta.
    def two(delta):
        sec = LightSource(0.3 * L_SUN, "companion",
                          KeplerOrbit(4.0e11, 0.1, 0.2, 0.5 + delta, 1.0 + delta, 0.3),
                          77 * DAY)
        return _geometry(**base, inclination=0.4, node=0.8 + delta, varpi=1.2 + delta,
                         sources=(LightSource(L_SUN), sec))
    a, b = two(0.0), two(1.7)
    for t in np.linspace(0, 3 * a.period_s, 97):
        da, fa = a.beams(t)
        db, fb = b.beams(t)
        assert np.max(np.abs(da - db)) <= 1e-13
        assert np.max(np.abs(fa / fb - 1)) <= 1e-13


def test_commensurate_configuration_recurs_exactly():
    g = _geometry(e=0.2, spins=Fraction(3, 2), obliquity_deg=20.0, varpi=0.7,
                  equinox=0.3, w0=0.1)
    period = 2 * g.period_s          # lcm of the orbit and of 2/3 of an orbit
    for t in np.linspace(0, period, 157):
        la, lo = g.subsolar_point(t)
        lb, lob = g.subsolar_point(t + period)
        assert abs(la - lb) <= 1e-12 and abs(_wrap(lo - lob)) <= 1e-12


@pytest.mark.parametrize("eps_deg", [0.0, 180.0])
@pytest.mark.parametrize("ratio", [Fraction(7, 2), Fraction(1, 3), Fraction(5)])
def test_subsolar_drift_rate_for_circular_aligned_axes(eps_deg, ratio):
    g = _geometry(spins=ratio, obliquity_deg=eps_deg)
    n = TWO_PI / g.period_s
    omega = g.rotation.angular_speed
    rate = -(omega - n * math.cos(math.radians(eps_deg)))
    t1, dt = 0.37 * g.period_s, 0.01 * g.period_s
    _, l1 = g.subsolar_point(t1)
    _, l2 = g.subsolar_point(t1 + dt)
    assert abs(_wrap(l2 - l1 - rate * dt)) <= 1e-12
    assert g.mean_solar_day == pytest.approx(TWO_PI / abs(rate), rel=1e-14)


# -- 8: multiple sources ----------------------------------------------------------------

def test_multiple_sources_superpose_and_respect_horizons():
    sec = LightSource(0.05 * L_SUN, "companion",
                      KeplerOrbit(3.0e12, 0.2, 0.1, 0.0, 0.0, 1.0), 400 * DAY)
    both = _geometry(spins=Fraction(13, 1), sources=(LightSource(L_SUN), sec))
    only1 = _geometry(spins=Fraction(13, 1), sources=(LightSource(L_SUN),))
    only2 = _geometry(spins=Fraction(13, 1), sources=(sec,))
    lat = np.radians(np.linspace(-89, 89, 37))
    lon = np.radians(np.linspace(-180, 175, 72))
    la, lo = (a.ravel() for a in np.meshgrid(lat, lon, indexing="ij"))
    for t in np.linspace(0, 9 * DAY, 23):
        fb, _ = OrbitalInsolation(both)(la, lo, t)
        f1, _ = OrbitalInsolation(only1)(la, lo, t)
        f2, _ = OrbitalInsolation(only2)(la, lo, t)
        assert np.max(np.abs(fb - (f1 + f2))) <= 1e-12 * np.max(fb)
        s2 = only2.star_direction(t)
        normals = np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])
        assert np.all(f2[(s2 @ normals) <= 0.0] == 0.0)


# -- 9-12: Mercury ----------------------------------------------------------------------

def _mercury():
    return illumination_from_spec({
        "stellar_constant": 9083.0, "period": 87.969 * DAY, "eccentricity": 0.2056,
        "spin": {"spins_per_orbit": [3, 2], "subsolar_longitude_at_epoch_deg": 0.0}})


def test_mercury_hot_and_warm_meridians():
    g = _mercury()
    p = g.period_s
    for k in range(6):
        _, lon = g.subsolar_point(k * p)
        assert abs(_wrap(lon - (0.0 if k % 2 == 0 else math.pi))) <= 1e-12
        _, lon = g.subsolar_point((k + 0.5) * p)
        assert abs(abs(_wrap(lon)) - 0.5 * math.pi) <= 1e-12


def test_mercury_repeats_after_one_solar_day():
    g = _mercury()
    solar_day = 2 * g.period_s
    assert g.mean_solar_day == pytest.approx(solar_day, rel=1e-14)
    for t in np.linspace(0, solar_day, 301):
        a, b = g.subsolar_point(t), g.subsolar_point(t + solar_day)
        assert abs(a[0] - b[0]) <= 1e-12 and abs(_wrap(a[1] - b[1])) <= 1e-12


def test_mercury_apparent_reversal_interval():
    g = _mercury()
    e, p = g.eccentricity, g.period_s
    ratio = g.rotation.angular_speed / g.mean_motion
    rhs = ratio * (1 - e * e) ** 1.5
    assert (1 - e) ** 2 < rhs < (1 + e) ** 2        # a reversal exists
    nu_star = math.acos((math.sqrt(rhs) - 1) / e)
    for sign in (+1, -1):
        nu = sign * nu_star
        ecc = 2 * math.atan2(math.sqrt(1 - e) * math.sin(nu / 2),
                             math.sqrt(1 + e) * math.cos(nu / 2))
        t_star = (ecc - e * math.sin(ecc)) / TWO_PI * p
        delta = 1e-6 * p
        _, l0 = g.subsolar_point(t_star)
        _, lm = g.subsolar_point(t_star - delta)
        _, lp = g.subsolar_point(t_star + delta)
        dm, dp = _wrap(lm - l0), _wrap(lp - l0)
        # A stationary point of the apparent motion lies within delta/2 of t_star.
        assert dm * dp > 0.0, (sign, dm, dp)


def test_mercury_horizon_crossings_are_reported():
    g = _mercury()
    ins = OrbitalInsolation(g)
    # Start a quarter orbit after perihelion: at perihelion the Sun is on the
    # horizon of the warm longitudes, which would make the endpoints ambiguous.
    times = 0.25 * g.period_s + np.linspace(0, 2 * g.period_s, 200001)
    counts = {}
    for lon_deg in (0, 90, 180, 270):
        mu = np.array([ins(np.array([0.0]), np.array([math.radians(lon_deg)]), t)[1][0]
                       for t in times[::10]])
        lit = mu > 0
        counts[lon_deg] = int(np.count_nonzero(lit[1:] != lit[:-1]))
    print("equatorial horizon crossings per solar day:", counts)
    assert counts[0] == 2 and counts[180] == 2
    # Warm longitudes see the Sun on the horizon at both perihelia; the
    # apparent reversal there turns each crossing into three.
    assert counts[90] == counts[270] and counts[90] > 2


# -- 13: diurnal mean ---------------------------------------------------------------------

def test_rotation_average_matches_diurnal_mean_at_a_frozen_orbital_position():
    g = _geometry(period=1.0e13, omega=TWO_PI / DAY, obliquity_deg=25.0,
                  nu0=math.radians(60.0))
    lat = np.radians(np.linspace(-85, 85, 35))
    times = np.linspace(0.0, DAY, 4000, endpoint=False)
    inst = OrbitalInsolation(g)
    mean = np.mean([inst(lat, 0.0 * lat, t)[0] for t in times], axis=0)
    ref, _ = DiurnalMeanInsolation(g)(lat, 0.0 * lat, 0.0)
    assert np.max(np.abs(mean - ref)) <= 1e-3 * g.flux_at(0.0)


# -- configuration form ----------------------------------------------------------------

def test_legacy_kinds_are_rejected_with_a_pointer():
    for kind in ("instantaneous", "tidally_locked"):
        with pytest.raises(ValueError, match="substrate-cpu-verified-v1"):
            insolation_from_spec({"kind": kind, "stellar_constant": 900.0})


def test_spin_specifications():
    base = {"stellar_constant": 589.0, "period": 59166360.0, "obliquity_deg": 25.19,
            "eccentricity": 0.0935}
    sol = illumination_from_spec({**base, "spin": {"mean_solar_day": 88440.0}})
    assert sol.mean_solar_day == pytest.approx(88440.0, rel=1e-13)
    retro = illumination_from_spec({**base, "obliquity_deg": 177.36,
                                    "spin": {"mean_solar_day": 1.0e7}})
    assert retro.rotation.is_retrograde
    assert retro.mean_solar_day == pytest.approx(1.0e7, rel=1e-13)
    with pytest.raises(ValueError):
        illumination_from_spec({**base, "spin": {"mean_solar_day": 1.0, "sidereal_period": 1.0}})
    with pytest.raises(ValueError):
        Rotation(-1.0)


def test_near_parabolic_periapsis_and_flux():
    e = 1.0 - 1.0e-10
    g = _geometry(e=e, spins=1)
    periapsis = g.orbit.semimajor_axis_m * (1.0 - e)
    assert g.true_anomaly(0.0) == 0.0
    assert np.linalg.norm(g.planet_position(0.0)) == pytest.approx(periapsis, rel=1e-13)
    assert g.flux_at(0.0) == pytest.approx(L_SUN / (4.0 * math.pi * periapsis**2),
                                           rel=1e-13)


@pytest.mark.parametrize("ecc", [-1.0e-6, 1.0e-6])
def test_near_parabolic_nonzero_mean_anomaly(ecc):
    e = math.nextafter(1.0, 0.0)
    # Kepler's equation from the sine series, without cancellation.
    mean = (1.0 - e) * ecc + e * (ecc**3 / 6.0 - ecc**5 / 120.0)
    assert eccentric_from_mean(mean, e) == pytest.approx(ecc, rel=1e-13, abs=0.0)


def test_mean_solar_day_requires_central_primary():
    companion = LightSource(L_SUN, orbit=KeplerOrbit(3.0e11), period_s=40 * DAY)
    with pytest.raises(ValueError, match="central primary"):
        _geometry(omega=0.0, sources=(companion,)).mean_solar_day
    assert _geometry(omega=0.0, sources=(LightSource(L_SUN), companion)
                     ).mean_solar_day == pytest.approx(10 * DAY)


@pytest.mark.parametrize("spin, error", [
    ({"subsolar_longitude_at_epoch_deg": 10.0}, "exactly one"),
    ({"sidereal_period": DAY, "subsolar_longitude_at_epoc_deg": 10.0},
     "subsolar_longitude_at_epoc_deg"),
])
def test_orbital_spin_options_are_validated(spin, error):
    with pytest.raises(ValueError, match=error):
        insolation_from_spec({"kind": "orbital", "stellar_constant": 589.0,
                              "period": 10 * DAY, "spin": spin})


def test_diurnal_mean_can_omit_spin():
    forcing = insolation_from_spec({"kind": "diurnal_mean", "stellar_constant": 589.0,
                                   "period": 10 * DAY})
    assert isinstance(forcing, DiurnalMeanInsolation)
