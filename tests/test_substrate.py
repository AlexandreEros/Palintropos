"""Fast CPU regression tests for the standalone conductive substrate.

The acceptance gates (G1-G5b) live in ``audit_substrate_gates.py``. These
tests are quick checks of the same kernel: layering, sign convention, energy
identity, the one-layer slab limit and the skin linearization.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tropoi.physics.substrate import (ConductiveSubstrate, ENERGY_REFERENCE_TEMPERATURE,
                                      geometric_interfaces)
from tropoi.physics.surface import SlabSurface


def test_default_layering_is_thai_column():
    sub = ConductiveSubstrate()
    assert sub.layers == 30
    assert sub.interfaces[0] == 0.0 and sub.interfaces[-1] == 3.5
    assert math.isclose(sub.thickness[0], 0.02, rel_tol=1e-12)
    ratios = sub.thickness[1:] / sub.thickness[:-1]
    assert np.all(ratios <= 1.5) and np.all(ratios >= 1.0)
    assert math.isclose(sub.conductivity, 2.0, rel_tol=1e-15)
    assert sub.signature()["kind"] == "conductive"


def test_stretch_layering_merges_thin_remainder():
    e = geometric_interfaces(0.1, 1.0, stretch=1.2)
    assert e[0] == 0.0 and e[-1] == 1.0
    dz = np.diff(e)
    assert np.all(dz > 0)
    with pytest.raises(ValueError):
        geometric_interfaces(0.02, 3.5, stretch=1.6)
    with pytest.raises(ValueError):
        geometric_interfaces(0.02, 3.5, layers=5)          # would need ratio > 1.5


def test_material_arguments_are_exclusive():
    with pytest.raises(ValueError):
        ConductiveSubstrate(thermal_inertia=2000.0, conductivity=2.0)


def test_bottom_flux_enters_bottom_row():
    sub = ConductiveSubstrate()
    st = sub.initial_state(1)
    st = sub.step(st, np.array([0.0]), 600.0, bottom_heat_flux=5.0)
    theta = st.anomaly[:, 0]
    assert theta[-1] > 0.0
    assert np.all(np.diff(theta) >= 0.0)                # warms from below
    assert theta[0] < 1e-6 * theta[-1]


def test_energy_identity_with_signs():
    sub = ConductiveSubstrate()
    rng = np.random.default_rng(1)
    fb = np.array([0.0, 0.1, -0.1])
    st = sub.initial_state(3)
    inflow = np.zeros(3)
    denom = np.zeros(3)
    for _ in range(2000):
        f = rng.uniform(-200.0, 200.0, size=3)
        st = sub.step(st, f, 600.0, bottom_heat_flux=fb)
        inflow += 600.0 * (st.surface_flux + fb)
        denom += 600.0 * np.abs(st.surface_flux + fb)
        resid = np.abs(np.asarray(sub.energy(st)) - inflow)
        assert np.all(resid <= 1e-12 * denom + 1e-30)


def test_one_layer_matches_slab():
    sub = ConductiveSubstrate(interfaces=(0.0, 1.0))
    slab = SlabSurface(2.0e6)
    rng = np.random.default_rng(2)
    st = sub.initial_state(1)
    T = np.array([ENERGY_REFERENCE_TEMPERATURE])
    for _ in range(1000):
        f = rng.uniform(-200.0, 200.0, size=1)
        st = sub.step(st, f, 600.0)
        T = slab.advance(T, f, 600.0)
    assert abs(float(sub.temperature(st)[0, 0]) - float(T[0])) < 1000 * np.spacing(300.0)


@pytest.mark.parametrize("skin", ["layer", "diagnosed"])
def test_robin_linearization_is_exact(skin):
    sub = ConductiveSubstrate(skin=skin)
    h = 8.14
    st = sub.initial_state(1)
    st = st._replace(anomaly=st.anomaly + 5.0)
    ts_old = float(sub.surface_temperature(st)[0]) - 300.0
    new = sub.step(st, np.array([-h * ts_old]), 600.0, flux_derivative=-h)
    t1 = float(new.anomaly[0, 0])
    if skin == "layer":
        expect = -h * t1
    else:
        expect = -h * t1 / (1.0 + h * sub.skin_resistance)
    assert math.isclose(float(new.surface_flux[0]), expect, rel_tol=1e-12)
    if skin == "diagnosed":
        ts_new = float(sub.surface_temperature(new)[0]) - 300.0
        assert math.isclose(float(new.surface_flux[0]), -h * ts_new, rel_tol=1e-12)


def test_step_flux_surface_response():
    """G2 at the default layering (the gate's own criterion, as a regression)."""
    sub = ConductiveSubstrate()
    st = sub.initial_state(1)
    F, dt, k, kappa = 100.0, 600.0, 2.0, 1e-6
    z = 0.5 * sub.thickness[0]
    for n in range(1, 400):
        st = sub.step(st, np.array([F]), dt)
    t = 399 * dt
    s = math.sqrt(kappa * t)
    x = z / (2 * s)
    ana = 2 * F / k * s * (math.exp(-x * x) / math.sqrt(math.pi) - x * math.erfc(x))
    # layer mean against point value at the centre differs at O(dz^2); 1 % covers it
    assert abs(float(st.anomaly[0, 0]) / ana - 1.0) < 0.01
