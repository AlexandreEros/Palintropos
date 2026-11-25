"""Heat-capacity laws, in particular Ledlow et al. (1992) eqs. (6)-(7), the
law Vasavada, Paige & Wood (1999) adopt.

The references are the printed equations (read from the rendered p. 650 of
the paper), evaluated independently here in cal g^-1 K^-1 and converted at
4184 J kg^-1 K^-1 per cal g^-1 K^-1.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.optimize import brentq

from tropoi.physics.materials import (CAL_G_TO_J_KG, PiecewiseHeatCapacity,
                                      PolynomialHeatCapacity,
                                      SaturatingExponentialHeatCapacity,
                                      heat_capacity_from_dict, ledlow_1992)


def eq6(T):
    x = (T - 300.0) / 300.0
    return 0.1812 + 0.1191 * x + 0.0176 * x ** 2 + 0.2721 * x ** 3 + 0.1869 * x ** 4


def eq7(T):
    return 0.2029 + 0.0383 * (1.0 - math.exp(-(T - 350.0) / 100.0))


def printed(T):
    return CAL_G_TO_J_KG * (eq6(T) if T <= 350.0 else eq7(T))


LAW = ledlow_1992()


def test_branches_reproduce_the_printed_equations():
    for T in (15.0, 40.0, 90.0, 100.0, 200.0, 299.0, 350.0, 350.0001, 400.0, 600.0, 750.0):
        assert float(LAW.specific_heat(np.array(T))) == pytest.approx(printed(T), rel=1e-14)


def test_branch_step_at_350_K_is_the_printed_rounding():
    lo, hi = CAL_G_TO_J_KG * eq6(350.0), CAL_G_TO_J_KG * eq7(350.0)
    assert hi == pytest.approx(0.2029 * CAL_G_TO_J_KG, rel=1e-15)
    step = (hi - lo) / lo
    assert abs(step) < 3e-4                         # "reduces to the previous expression"
    assert float(LAW.specific_heat(np.array(350.0))) == pytest.approx(lo, rel=1e-15)  # eq. (6)
    assert float(LAW.specific_heat(np.nextafter(350.0, 400.0))) == pytest.approx(hi, rel=1e-12)


def test_positivity_and_applicability():
    root = brentq(eq6, 1.0, 50.0)
    assert 10.0 < root < 10.3                        # eq. (6) is negative below ~10.1 K
    T = np.linspace(root + 0.05, 2000.0, 20001)
    assert np.all(LAW.specific_heat(T) > 0.0)
    # eq. (7) saturates: monotone, bounded by 0.2412 cal/g/K
    hi = LAW.high.specific_heat(np.linspace(350.0, 5000.0, 500))
    assert np.all(np.diff(hi) >= 0) and hi.max() <= 0.2412 * CAL_G_TO_J_KG


@pytest.mark.parametrize("t0, t1", [(40.0, 120.0), (90.0, 349.0), (300.0, 350.0),
                                    (349.5, 350.5), (120.0, 700.0), (351.0, 720.0),
                                    (700.0, 95.0), (350.0, 350.0 + 1e-3)])
def test_analytic_enthalpy_matches_quadrature(t0, t1):
    ref, err = quad(printed, t0, t1, points=[350.0] if min(t0, t1) < 350 < max(t0, t1) else None,
                    epsabs=0.0, epsrel=1e-13, limit=200)
    got = float(LAW.enthalpy_difference(np.array(t1), np.array(t0)))
    assert got == pytest.approx(ref, rel=1e-12, abs=1e-12 * abs(printed(t0)) * abs(t1 - t0))


@pytest.mark.parametrize("centre", [120.0, 349.9999, 350.0, 350.0001, 650.0])
def test_tiny_increments_keep_full_precision_including_across_350_K(centre):
    for d in (1e-4, 1e-8, 1e-11):
        t0 = np.array(centre - 0.5 * d)
        t1 = t0 + d
        got = float(LAW.enthalpy_difference(t1, t0, delta=np.array(d)))
        # reference: integral split at 350 K, each piece by its own branch's mean
        if t1 <= 350.0 or t0 >= 350.0:
            branch = eq6 if t1 <= 350.0 else eq7
            ref = CAL_G_TO_J_KG * d * branch(float(centre))
        else:
            ref = CAL_G_TO_J_KG * ((350.0 - float(t0)) * eq6(350.0) + (float(t1) - 350.0) * eq7(350.0))
        assert got == pytest.approx(ref, rel=1e-6)            # O(d) mean-value effect only
        assert got > 0.0


def test_enthalpy_derivative_is_the_specific_heat():
    T = np.array([60.0, 200.0, 349.0, 351.0, 500.0, 700.0])
    h = 1e-4
    fd = (LAW.enthalpy_difference(T + h, T) - LAW.enthalpy_difference(T - h, T)) / (2 * h)
    assert np.allclose(fd, LAW.specific_heat(T), rtol=1e-8)


def test_laws_round_trip_through_their_configuration_form():
    for law in (LAW, PolynomialHeatCapacity((1.0, 2.0), 3.0, 4.0, 5.0),
                SaturatingExponentialHeatCapacity(1.0, 0.5, 300.0, 50.0)):
        assert heat_capacity_from_dict(law.to_dict()) == law
    assert heat_capacity_from_dict([700.0]) == PolynomialHeatCapacity((700.0,))
    assert isinstance(LAW, PiecewiseHeatCapacity) and LAW.breakpoint == 350.0
