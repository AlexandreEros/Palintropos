"""The one definition of the spherical Laplacian's eigenvalues.

``tropoi.spatial.laplacian.laplacian_eigenvalues`` replaced nine separate
spellings of ``-l(l+1)/R**2`` (the BVE, SWE and PE cores, ``SpectralOperators``,
the semi-implicit solver, the Held-Suarez hyperdiffusion coefficient and rates, a BVE
diagnostic and the custom-run preparation). Before that, a mutation of the
``SpectralOperators`` copy (``l(l+1)`` -> ``l**2``) survived the entire test
suite because only the visualisers read that copy.

Everything here except the last test runs on a CPU-only machine, so CI covers
it. The reference is independent of the code under test: the Legendre
differential equation, evaluated with exact polynomial arithmetic.
"""
from __future__ import annotations

import pathlib
import re

import numpy as np
import pytest
from numpy.polynomial.legendre import Legendre

from tropoi.spatial.laplacian import invert_laplacian, laplacian_eigenvalues

R_EARTH = 6.371e6


@pytest.mark.parametrize("l", range(0, 31))
def test_eigenvalue_solves_the_legendre_equation(l):
    """lap P_l(cos t) = -l(l+1) P_l / R^2 for the m = 0 harmonics.

    With x = cos(theta) the sphere's Laplacian on an axisymmetric function is
    d/dx[(1 - x^2) df/dx] / R^2, so the eigenvalue is read off the polynomial
    identity d/dx[(1 - x^2) P_l'] = -l(l+1) P_l.
    """
    # Stay in the Legendre basis: converting to monomials is ill-conditioned
    # above l ~ 22 and would make the reference, not the code, fail.
    x_series = Legendre.basis(1)
    p = Legendre.basis(l)
    lap = ((Legendre.basis(0) - x_series * x_series) * p.deriv()).deriv()
    x = np.linspace(-0.98, 0.98, 41)
    expected = laplacian_eigenvalues(l, R_EARTH)[l] * R_EARTH**2
    np.testing.assert_allclose(lap(x), expected * p(x), rtol=1e-9, atol=1e-9 * (l + 1) ** 2)


def test_values_for_the_first_degrees():
    lam = laplacian_eigenvalues(3, 2.0)
    np.testing.assert_array_equal(lam, [0.0, -2.0 / 4.0, -6.0 / 4.0, -12.0 / 4.0])
    assert lam.dtype == np.float64 and lam.shape == (4,)


def test_degree_zero_is_exactly_zero_for_every_radius():
    for radius in (1.0, 3.3895e6, R_EARTH, 6.0e7):
        assert laplacian_eigenvalues(10, radius)[0] == 0.0


def test_eigenvalues_scale_as_inverse_radius_squared():
    np.testing.assert_allclose(laplacian_eigenvalues(42, 2.0 * R_EARTH),
                               laplacian_eigenvalues(42, R_EARTH) / 4.0, rtol=1e-15)


def test_negation_is_the_positive_definite_operator():
    pos = -laplacian_eigenvalues(21, R_EARTH)
    assert pos[0] == 0.0 and np.all(pos[1:] > 0.0)


def test_no_second_definition_of_the_eigenvalue_in_src():
    """Nothing under src/ spells x*(x+1) except the one module and two cases
    that are not the Laplacian's eigenvalue (a kinetic-energy weight and a
    spectral-slope ratio in topography). A textual guard: it catches the
    copy-pasted spelling, which is how the nine copies arose, not an
    arbitrary rewrite of the same formula."""
    root = pathlib.Path(__file__).resolve().parents[1] / "src" / "tropoi"
    # "x * (x + 1)" for any name x: the way l(l+1) is spelled when copied.
    pattern = re.compile(r"\b(\w+)\s*\*\s*\(\s*\1\s*\+\s*1(\.0*)?\s*\)")
    allowed = {("spatial", "laplacian.py"), ("run", "held_suarez", "experiment.py"),
               ("spatial", "terrain", "topography.py")}
    found = set()
    for path in root.rglob("*.py"):
        lines = path.read_text(encoding="utf-8").splitlines()
        if any(pattern.search(line) for line in lines
               if not line.lstrip().startswith("#")):
            found.add(path.relative_to(root).parts)
    assert found == allowed, (
        f"spell the Laplacian eigenvalue through laplacian_eigenvalues(): {sorted(found - allowed)}"
        if found - allowed else f"allow-list is stale: {sorted(allowed - found)}")


def _random_coeffs(shape, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)


def test_inversion_is_bit_identical_to_the_old_division_for_l_ge_1():
    """For l >= 1 it is the same single division the five inline copies did."""
    lam = laplacian_eigenvalues(21, R_EARTH)
    zeta = _random_coeffs((22, 22))
    psi = invert_laplacian(zeta, lam)
    np.testing.assert_array_equal(psi[1:], zeta[1:] / lam[1:, None])


def test_inversion_zeroes_l0_and_ignores_its_input():
    """R-10 remainder: SpectralOperators used to return R^2 * zeta_00 here."""
    lam = laplacian_eigenvalues(10, R_EARTH)
    zeta = _random_coeffs((11, 11))
    zeta[0, 0] = 7.0
    psi = invert_laplacian(zeta, lam)
    assert np.all(psi[0] == 0.0)
    assert zeta[0, 0] == 7.0, "the input must not be modified"


def test_inversion_batches_leading_axes_and_round_trips():
    """Levels on a leading axis (the PE layout); lap(inv(z)) = z for l >= 1."""
    lam = laplacian_eigenvalues(15, R_EARTH)
    zeta = _random_coeffs((4, 16, 16), seed=1)
    psi = invert_laplacian(zeta, lam)
    for k in range(4):
        np.testing.assert_array_equal(psi[k], invert_laplacian(zeta[k], lam))
    np.testing.assert_allclose((lam[:, None] * psi)[:, 1:], zeta[:, 1:], rtol=1e-14)


def _has_cuda() -> bool:
    try:
        import cupy as cp

        return bool(cp.is_available())
    except Exception:  # pragma: no cover - import guard
        return False


@pytest.mark.skipif(not _has_cuda(), reason="CUDA/CuPy not available")
def test_cores_read_the_shared_eigenvalues_and_bve_keeps_l0_physical():
    """R-10: the BVE core used to overwrite laplacian_eig[0] with 1/R^2."""
    import cupy as cp

    from tropoi.spatial.environment import PlanetaryParameters
    from tropoi.spatial.planet import Planet
    from tropoi.temporal.tendencies.barotropic import BarotropicVorticity

    params = PlanetaryParameters.from_earth_like(day_hours=24.0)
    planet = Planet.generate(params=params, grid_resolution=4, l_max=21)
    model = BarotropicVorticity(planet, viscosity=0.0)
    expected = laplacian_eigenvalues(21, model.R)
    np.testing.assert_array_equal(cp.asnumpy(model.laplacian_eig), expected)
    np.testing.assert_array_equal(cp.asnumpy(planet.so.lap_eigs), expected)
    assert float(model.laplacian_eig[0]) == 0.0


@pytest.mark.skipif(not _has_cuda(), reason="CUDA/CuPy not available")
def test_every_inversion_route_agrees_including_l0():
    """SpectralOperators.inv_laplacian, the BVE/SWE/PE inversions and the BVE
    diagnostic used to be five implementations; one disagreed at l = 0."""
    import cupy as cp

    from tropoi.spatial.environment import PlanetaryParameters
    from tropoi.spatial.planet import Planet
    from tropoi.spatial.states.barotropic import BarotropicState
    from tropoi.temporal.tendencies.barotropic import BarotropicVorticity

    params = PlanetaryParameters.from_earth_like(day_hours=24.0)
    planet = Planet.generate(params=params, grid_resolution=4, l_max=21)
    model = BarotropicVorticity(planet, viscosity=0.0)
    zeta = cp.asarray(_random_coeffs((22, 22), seed=2))
    zeta[0, 0] = 1.0e-5
    via_operators = planet.so.inv_laplacian(zeta)
    via_core = model.vorticity_to_streamfunction(BarotropicState(zeta))
    np.testing.assert_array_equal(cp.asnumpy(via_operators), cp.asnumpy(via_core))
    assert float(cp.max(cp.abs(via_operators[0]))) == 0.0
