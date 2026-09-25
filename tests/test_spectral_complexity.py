"""Host spectral kinetic-energy modes and complexity measures. CPU only.

The canonical checks use the committed Williamson-5 T63 capsule and the
tracked grid package ``aeolus_w5_t63.npz`` written by the same run, so the
modal sum is compared with an independent grid quadrature of the saved winds.
"""
from __future__ import annotations

import pathlib

import numpy as np
import pytest

from tropoi.representation.diagnostics.spectral import (
    ZeroKineticEnergyError, kinetic_energy_modes, mode_power,
    spectral_complexity)

ROOT = pathlib.Path(__file__).resolve().parents[1]
W5 = ROOT / "docs" / "validation" / "williamson_5"
CAPSULE = (W5 / "capsules" / "t63" /
           "20260730T011700Z_williamson5_rot23p93h_r4_l63_dt120h_45406d82_"
           "668e6c9a")
W5_RADIUS_M = 6.37122e6


def test_solid_body_rotation_is_one_mode_with_the_analytic_energy():
    u0, radius, l_max = 20.0, 6.0e6, 5
    zeta = np.zeros((l_max + 1, l_max + 1), dtype=np.complex128)
    # zeta = 2 u0 sin(lat) / R and Y_10 = sqrt(3 / (4 pi)) sin(lat).
    zeta[1, 0] = 2.0 * u0 / radius * np.sqrt(4.0 * np.pi / 3.0)
    energy = kinetic_energy_modes(zeta, radius=radius)
    # 1/2 integral (u0 cos lat)^2 dA = (4 pi / 3) u0^2 R^2.
    np.testing.assert_allclose(energy.sum(),
                               4.0 * np.pi / 3.0 * u0 ** 2 * radius ** 2,
                               rtol=1e-14)
    measures = spectral_complexity(energy)
    assert measures.mean_degree == 1.0
    assert measures.mean_order == 0.0
    assert measures.entropy == 0.0
    assert measures.effective_modes == 1.0


def test_positive_orders_carry_their_conjugate_pair_and_padding_is_ignored():
    coeffs = np.zeros((3, 3), dtype=np.complex128)
    coeffs[2, 0] = 3.0 + 0.0j
    coeffs[2, 1] = 1.0 + 1.0j
    coeffs[1, 2] = 100.0      # m > l: storage padding, never a mode
    power = mode_power(coeffs)
    assert power[2, 0] == 9.0
    assert power[2, 1] == pytest.approx(4.0, rel=1e-15)
    assert power[1, 2] == 0.0


def test_divergent_energy_adds_and_monopole_carries_none():
    zeta = np.zeros((4, 4), dtype=np.complex128)
    delta = np.zeros((4, 4), dtype=np.complex128)
    zeta[0, 0] = 5.0
    delta[0, 0] = 7.0
    delta[2, 1] = 1.0j
    energy = kinetic_energy_modes(zeta, delta, radius=1.0)
    assert energy[0, 0] == 0.0
    assert energy[2, 1] == pytest.approx(0.5 * 2.0 / 6.0)


def test_zero_kinetic_energy_is_undefined_not_zero():
    energy = kinetic_energy_modes(np.zeros((4, 4), dtype=np.complex128),
                                  radius=1.0)
    with pytest.raises(ZeroKineticEnergyError):
        spectral_complexity(energy)


def test_canonical_t63_modal_sum_equals_grid_quadrature_of_saved_winds():
    coefficients = np.load(CAPSULE / "swe_coeffs.npy")
    with np.load(W5 / "aeolus_w5_t63.npz") as package:
        u, v = package["u"], package["v"]
        weights = package["gl_weights"]
        nlon = package["longitude"].size
    area = weights[:, None] / (weights.sum() * nlon)
    for index in range(4):
        energy = kinetic_energy_modes(
            coefficients[index, 0], coefficients[index, 1],
            radius=W5_RADIUS_M)
        grid = (0.5 * W5_RADIUS_M ** 2 * 4.0 * np.pi *
                np.sum(area * (u[index] ** 2 + v[index] ** 2)))
        np.testing.assert_allclose(energy.sum(), grid, rtol=1e-11, atol=0)


def test_canonical_t63_complexity_reproduces_the_published_values():
    coefficients = np.load(CAPSULE / "swe_coeffs.npy")
    measured = [spectral_complexity(kinetic_energy_modes(
        coefficients[i, 0], coefficients[i, 1], radius=W5_RADIUS_M))
        for i in range(4)]
    # Printed on the 2026-07-30 figure (and quoted by the README caption).
    published = [(1.000, 0.000, 1.00), (1.563, 0.467, 1.89),
                 (2.149, 0.884, 3.04), (2.663, 1.208, 4.42)]
    for values, (degree, order, modes) in zip(measured, published):
        assert round(values.mean_degree, 3) == degree
        assert round(values.mean_order, 3) == order
        assert round(values.effective_modes, 2) == modes
    # Unrounded values measured 2026-09-25 from the committed capsule.
    np.testing.assert_allclose(
        [m.mean_degree for m in measured],
        [1.0, 1.5626321130924579, 2.1486875097154763, 2.6627801773402773],
        rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        [m.effective_modes for m in measured],
        [1.0, 1.8866285054484508, 3.0417925171371603, 4.418275370120114],
        rtol=0, atol=1e-12)
