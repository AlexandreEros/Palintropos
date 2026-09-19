"""Preset support characterization at the product-truncation boundaries.

Measured record: docs/validation/preset_support_characterization.md
(Sprint 1, 2026-09-18). Each preset stores coefficients through ``l_max``
but analyzed nonlinear products retain only degrees ``<= cut = 2*l_max//3``.
Degrees above the cut are advanced by the exact *linear* spectral
operators only. This file pins that behavior on both backends at the
transition degrees, deliberately through LOW-LEVEL state construction
(``ShallowWaterState`` / ``isothermal_rest_state``) so the observations
stay reproducible after the public factories reject the unsupported
boundaries (tests/test_swe_runner.py, tests/test_pe_run_config.py).

Assertion policy: exact-zero / bitwise assertions are structural
(products of exactly-zero fields, rows zeroed by the cut, exact diagonal
operators) and hold on BOTH backends; quadrature-limited residuals carry
explicit tolerances taken from the measured values with headroom.
"""
from __future__ import annotations

import math

import pytest


def _has_cuda():
    try:
        import cupy as cp
        return cp.is_available()
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _has_cuda(),
                                reason="CUDA/CuPy not available")

GRIDS = ("latlon", "geodesic")

# Williamson (1992) constants, as in tests/test_williamson2.py.
GRAVITY = 9.80616
GH0 = 2.94e4
OMEGA = 7.29212e-5
DAY_HOURS_W2 = 2.0 * math.pi / OMEGA / 3600.0

T0 = 260.0
PS0 = 101325.0


def _cut(l_max):
    # Historical production formula (pinned in tests/test_truncation_contract).
    return (2 * l_max) // 3


def _planet(grid, l_max, day_hours=24.0):
    from tropoi.planet import Planet, PlanetaryParameters
    return Planet.generate(
        params=PlanetaryParameters.from_earth_like(day_hours=day_hours),
        grid_type=grid, nlat=32, nlon=64, l_max=l_max, grid_resolution=3)


def _band_max(a, lo, hi):
    import cupy as cp
    return float(cp.abs(a[..., lo:hi + 1, :]).max()) if hi >= lo else 0.0


# ---------------------------------------------------------------------------
# PE thermal_wave: lmax = 2 stores the degree-2 perturbation but cut = 1
# ---------------------------------------------------------------------------

def _thermal_wave_state(model, amp):
    from tropoi.physics.primitive_equations import isothermal_rest_state
    state = isothermal_rest_state(model.l_max, model.nlev,
                                  temperature=T0, surface_pressure=PS0)
    state.temperature[:, 2, 2] = float(amp)   # low-level, no factory guard
    return state


def _pe_model(grid, l_max, nlev=3):
    from tropoi.physics.primitive_equations import PrimitiveEquationsModel
    from tropoi.physics.sigma_coordinate import SigmaGrid
    return PrimitiveEquationsModel(_planet(grid, l_max), SigmaGrid.uniform(nlev))


def _two_validated_seconds(model, state):
    from tropoi.physics.primitive_equations import PrimitiveEquationsState
    from tropoi.run.engine import rk4_step_array

    def validator(y):
        model.validate_state(PrimitiveEquationsState(y), context="stage")
    y = state.coeffs.copy()
    for _ in range(2):
        y = rk4_step_array(model.tendency, y, 0.0, 1.0,
                           stage_validator=validator)
    return y


@pytest.mark.parametrize("grid", GRIDS)
def test_pe_thermal_wave_lmax2_freezes_the_degree2_temperature(grid):
    """cut(2) = 1: T(2,2) and ln p_s above the cut never change; the
    degree-2 divergence receives the exact hydrostatic forcing."""
    import cupy as cp
    l_max, cut, K = 2, 1, 3
    assert _cut(l_max) == cut
    model = _pe_model(grid, l_max, nlev=K)
    state = _thermal_wave_state(model, 1.0)
    model.validate_state(state)
    ten = model.tendency(state.coeffs)
    zeta, delta = ten[0:K], ten[K:2 * K]
    temp, lnps = ten[2 * K:3 * K], ten[3 * K]
    # Rest winds: every product vanishes exactly.
    assert float(cp.abs(zeta).max()) == 0.0
    assert float(cp.abs(temp).max()) == 0.0
    assert float(cp.abs(lnps).max()) == 0.0
    assert _band_max(delta, 0, cut) == 0.0
    # The hydrostatic term is exact and diagonal: degree-2 delta is forced.
    forced = float(cp.abs(delta[:, 2, 2]).max())
    assert forced > 0.0
    assert forced == pytest.approx(7.611e-11, rel=0.05)   # measured

    y = _two_validated_seconds(model, state)
    d = y - state.coeffs
    assert float(cp.abs(d[2 * K:3 * K, 2, 2]).max()) == 0.0   # T(2,2) frozen
    assert _band_max(d[2 * K:3 * K], cut + 1, l_max) == 0.0  # all upper T
    assert _band_max(d[3 * K], cut + 1, l_max) == 0.0        # upper ln p_s
    assert float(cp.abs(d[K:2 * K, 2, 2]).max()) > 0.0       # delta moved


@pytest.mark.parametrize("grid", GRIDS)
def test_pe_thermal_wave_lmax3_lets_the_degree2_temperature_evolve(grid):
    """cut(3) = 2 retains the degree-2 products: T(2,2) responds."""
    import cupy as cp
    K = 3
    model = _pe_model(grid, 3, nlev=K)
    state = _thermal_wave_state(model, 1.0)
    y = _two_validated_seconds(model, state)
    d = y - state.coeffs
    moved = float(cp.abs(d[2 * K:3 * K, 2, 2]).max())
    assert moved > 0.0
    assert moved == pytest.approx(9.218e-9, rel=0.05)      # measured


@pytest.mark.parametrize("grid", GRIDS)
def test_pe_thermal_wave_zero_amplitude_is_exact_rest_at_lmax2(grid):
    import cupy as cp
    model = _pe_model(grid, 2, nlev=3)
    state = _thermal_wave_state(model, 0.0)
    assert float(cp.abs(model.tendency(state.coeffs)).max()) == 0.0
    y = _two_validated_seconds(model, state)
    assert bool(cp.all(y == state.coeffs))


# ---------------------------------------------------------------------------
# SWE gravity_wave: the Y_4^2 mode lies above the cut at lmax = 4, 5
# ---------------------------------------------------------------------------

def _gravity_wave(grid, l_max):
    from tropoi.physics.shallow_water import (ShallowWaterModel,
                                              ShallowWaterState)
    planet = _planet(grid, l_max, day_hours=math.inf)
    model = ShallowWaterModel(planet, mean_depth=1000.0)   # flat, nu4 = 0
    state = ShallowWaterState.zeros(l_max)
    state.coeffs[2, 4, 2] += 1e-3 * model.phi0            # as the preset
    model.validate_state(state)
    return planet, model, state


@pytest.mark.parametrize("grid", GRIDS)
@pytest.mark.parametrize("l_max", [4, 5])
def test_swe_gravity_wave_upper_pair_is_exactly_linear(grid, l_max):
    """Above the cut, delta/phi see only the exact linear pressure pair."""
    import cupy as cp
    from tropoi.run.engine import rk4_step_array
    cut = _cut(l_max)
    _, model, state = _gravity_wave(grid, l_max)
    ten = model.tendency(state.coeffs)
    assert bool(ten[1, 4, 2] == -model.lap_eigs[4] * state.coeffs[2, 4, 2])
    assert _band_max(ten[2], cut + 1, l_max) == 0.0
    assert float(cp.abs(ten[0]).max()) == 0.0
    y = rk4_step_array(model.tendency, state.coeffs, 0.0, 300.0)
    ten2 = model.tendency(y)
    lin_delta = -model.lap_eigs[:, None] * y[2]
    lin_phi = -model.phi0 * y[1]
    assert bool(cp.all(ten2[1, cut + 1:, :] == lin_delta[cut + 1:, :]))
    assert bool(cp.all(ten2[2, cut + 1:, :] == lin_phi[cut + 1:, :]))


@pytest.mark.parametrize("grid", GRIDS)
@pytest.mark.parametrize("l_max", [4, 5])
def test_swe_gravity_wave_lower_band_is_fed_nonlinearly(grid, l_max):
    """The lower band (l <= cut) is NOT linear: removing the upper delta/phi
    inputs removes its tendency, so the upper modes feed it through the
    retained products. Only the upper pair is a linear regime."""
    from tropoi.run.engine import rk4_step_array
    cut = _cut(l_max)
    _, model, state = _gravity_wave(grid, l_max)
    y = rk4_step_array(model.tendency, state.coeffs, 0.0, 300.0)
    full = model.tendency(y)
    y_lower = y.copy()
    y_lower[:, cut + 1:, :] = 0.0
    without_upper = model.tendency(y_lower)
    lower_full = _band_max(full, 0, cut)
    lower_without = _band_max(without_upper, 0, cut)
    assert lower_full > 0.0
    assert lower_full > 1e3 * lower_without         # measured ~1e4 ratio
    lin_delta = -model.lap_eigs[:, None] * y[2]
    lin_phi = -model.phi0 * y[1]
    lower_linear_residual = max(_band_max(full[1] - lin_delta, 0, cut),
                                _band_max(full[2] - lin_phi, 0, cut))
    assert lower_linear_residual > 0.0


@pytest.mark.parametrize("grid", GRIDS)
@pytest.mark.parametrize("l_max", [4, 5])
def test_swe_gravity_wave_frequency_at_minimum_capacity(grid, l_max):
    from tropoi.run.engine import rk4_step_array
    planet, model, state = _gravity_wave(grid, l_max)
    l, m = 4, 2
    omega_exact = math.sqrt(model.phi0 * l * (l + 1)) / planet.params.radius
    dt = 300.0
    y, t = state.coeffs, 0.0
    prev_val, prev_t = float(y[2, l, m].real), 0.0
    t_zero = None
    for _ in range(int(2.0 * math.pi / omega_exact / dt) + 10):
        y = rk4_step_array(model.tendency, y, t, dt)
        t += dt
        val = float(y[2, l, m].real)
        if prev_val > 0.0 >= val:
            t_zero = prev_t + dt * prev_val / (prev_val - val)
            break
        prev_val, prev_t = val, t
    assert t_zero is not None
    assert math.pi / (2.0 * t_zero) == pytest.approx(omega_exact, rel=2e-3)


# ---------------------------------------------------------------------------
# Williamson 2, flat bottom: the degree-2 product/pressure cancellation
# ---------------------------------------------------------------------------

def _williamson2_lowlevel(grid, l_max, *, retain_all=False):
    """Flat-bottom W2 with canonical constants, built without the factory."""
    from tropoi.physics.shallow_water import (ShallowWaterModel,
                                              ShallowWaterState)
    planet = _planet(grid, l_max, day_hours=DAY_HOURS_W2)
    a = planet.params.radius
    u0 = 2.0 * math.pi * a / (12.0 * 86400.0)
    C = a * OMEGA * u0 + 0.5 * u0 * u0
    model = ShallowWaterModel(planet, gravity=GRAVITY,
                              mean_depth=(GH0 - C / 3.0) / GRAVITY)
    if retain_all:
        model._trunc_cut = l_max      # characterization only: disable cut
    state = ShallowWaterState.zeros(l_max)
    state.coeffs[0, 1, 0] = (2.0 * u0 / a) * math.sqrt(4.0 * math.pi / 3.0)
    state.coeffs[2, 2, 0] = -(4.0 * C / 3.0) * math.sqrt(math.pi / 5.0)
    model.validate_state(state)
    return model, state


@pytest.mark.parametrize("grid", GRIDS)
def test_w2_lmax2_residual_is_exactly_the_uncancelled_pressure_term(grid):
    """cut(2) = 1 discards the degree-2 curl and kinetic-energy products
    that balance -lap(phi) in the steady solution; the residual divergence
    tendency is then exactly the diagonal pressure term."""
    import cupy as cp
    model, state = _williamson2_lowlevel(grid, 2)
    ten = model.tendency(state.coeffs)
    assert float(cp.abs(ten[0]).max()) == 0.0        # zeta steady
    assert float(cp.abs(ten[2]).max()) == 0.0        # phi steady
    pressure_only = -model.lap_eigs[2] * state.coeffs[2, 2, 0]
    assert bool(ten[1, 2, 0] == pressure_only)
    assert abs(float(pressure_only.real)) == pytest.approx(2.919e-9, rel=0.05)
    assert _band_max(ten[1], 0, 1) < 1e-20           # round-off only


@pytest.mark.parametrize("grid", GRIDS)
@pytest.mark.parametrize("l_max", [3, 4])
def test_w2_is_steady_once_the_cut_retains_degree2(grid, l_max):
    import cupy as cp
    assert _cut(l_max) >= 2
    model, state = _williamson2_lowlevel(grid, l_max)
    ten = model.tendency(state.coeffs)
    assert float(cp.abs(ten[0]).max()) == 0.0
    assert float(cp.abs(ten[2]).max()) == 0.0
    # Measured 2.1e-24 (Gauss) / 5.4e-24 (geodesic) vs the 2.9e-9 pressure
    # term: the degree-2 products cancel it to round-off.
    assert float(cp.abs(ten[1]).max()) < 1e-18


@pytest.mark.parametrize("grid", GRIDS)
def test_w2_lmax2_cancellation_returns_when_degree2_products_are_retained(
        grid):
    """Disabling the cut at lmax = 2 (characterization only) restores the
    steady state: the residual is the lost products, not the storage."""
    import cupy as cp
    model, state = _williamson2_lowlevel(grid, 2, retain_all=True)
    ten = model.tendency(state.coeffs)
    assert float(cp.abs(ten).max()) < 1e-18


# ---------------------------------------------------------------------------
# Williamson 5: capacity is a source fact (the (2,0) mode must exist)
# ---------------------------------------------------------------------------

def test_w5_construction_needs_degree_two_storage():
    """The W5 wind/free-surface pair writes the (2,0) coefficient; an
    lmax = 1 state cannot hold it (IndexError at the low level). No W5
    integration is run here."""
    from tropoi.physics.shallow_water import ShallowWaterState
    state = ShallowWaterState.zeros(1)
    with pytest.raises(IndexError):
        state.coeffs[2, 2, 0] += 1.0
