"""Two-stream solver: independent ODE solutions, closed forms, exact angular
integration. CPU only (NumPy/SciPy)."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import solve_bvp
from scipy.special import expn

from tropoi.physics.radiation import twostream as ts


def _bvp(rhs, bc, edges, per_layer=300, tol=1e-10):
    """Collocation solution of the two-stream ODEs (the independent
    reference). The mesh carries every layer edge as a node, so coefficient
    jumps fall between collocation intervals."""
    x = np.unique(np.concatenate([np.linspace(edges[i], edges[i + 1], per_layer)
                                  for i in range(len(edges) - 1)]))
    sol = solve_bvp(rhs, bc, x, np.zeros((2, x.size)), tol=tol, max_nodes=400000)
    # status 1 = the requested tolerance was not certified on a mesh with
    # coefficient jumps; the comparison against it is the test.
    assert sol.status in (0, 1), sol.message
    return sol


@pytest.mark.parametrize("closure", ["eddington", "quadrature", "hemispheric"])
def test_direct_beam_layer_matches_numerical_solution_of_the_two_stream_equations(closure):
    rng = np.random.default_rng(7)
    checked = 0
    for _ in range(25):
        tau = 10 ** rng.uniform(-2, 1)
        w, g, mu = rng.uniform(0.05, 0.999), rng.uniform(-0.2, 0.8), rng.uniform(0.1, 1.0)
        g1, g2, g3 = ts.stellar_coefficients(w, g, mu, closure)

        def rhs(t, y):
            s = w * np.exp(-t / mu) / mu
            return np.vstack([g1 * y[0] - g2 * y[1] - g3 * s,
                              g2 * y[0] - g1 * y[1] + (1 - g3) * s])

        sol = _bvp(rhs, lambda ya, yb: np.array([ya[1], yb[0]]), [0.0, tau], 400)
        if sol.y[0, 0] < 0 or sol.y[1, -1] < 0:
            continue        # the closure itself is unphysical here; the code clips it
        r, t, beam = ts.direct_layer(np.array(tau), np.array(w), g1, g2, g3, np.array(mu))
        assert abs(r - sol.y[0, 0]) < 1e-9 and abs(t - sol.y[1, -1]) < 1e-9
        assert beam == pytest.approx(np.exp(-tau / mu))
        checked += 1
    assert checked >= 15


def test_scattering_thermal_column_matches_numerical_solution():
    """Six inhomogeneous scattering layers, linear-in-tau sources, a
    reflecting emitting surface and diffuse incidence at the top."""
    rng = np.random.default_rng(3)
    K, D = 6, 1.66
    tau = 10 ** rng.uniform(-1, 0.5, K)
    w, g = rng.uniform(0, 0.9, K), rng.uniform(0, 0.7, K)
    planck = rng.uniform(100, 400, K + 1)
    edges = np.concatenate([[0.0], np.cumsum(tau)])
    albedo, source, incident = 0.3, 250.0, 17.0

    def rhs(t, y):
        k = np.clip(np.searchsorted(edges, t, side="right") - 1, 0, K - 1)
        g1, g2 = ts.thermal_coefficients(w[k], g[k], D)
        s = D * (1 - w[k]) * (planck[k] + (planck[k + 1] - planck[k]) * (t - edges[k]) / tau[k])
        return np.vstack([g1 * y[0] - g2 * y[1] - s, g2 * y[0] - g1 * y[1] + s])

    sol = _bvp(rhs, lambda ya, yb: np.array([ya[1] - incident,
                                            yb[0] - albedo * yb[1] - source]), edges,
               tol=1e-7)
    up, down = ts.thermal_fluxes(tau, planck[:-1], planck[1:], surface_source=source,
                                 surface_albedo=albedo, ssa=w, asym=g, diffusivity=D,
                                 top_down=incident)
    ref = sol.sol(edges)
    assert np.abs(up - ref[0]).max() < 1e-5        # W m^-2 on fluxes of O(300)
    assert np.abs(down - ref[1]).max() < 1e-5


def test_conservative_scattering_conserves_energy_and_has_the_closed_form():
    tau = np.array([1e-6, 0.03, 1.0, 40.0])
    for closure in ("eddington", "quadrature", "hemispheric"):
        g1, g2, _ = ts.stellar_coefficients(1.0, 0.3, 0.5, closure)
        r, t = ts.diffuse_layer(tau, g1 + 0 * tau, g2 + 0 * tau)
        assert np.abs(r + t - 1.0).max() < 1e-12
        assert np.abs(r - g1 * tau / (1 + g1 * tau)).max() < 1e-9
        up, down, direct = ts.stellar_fluxes(tau[:, None], np.ones((4, 1)), np.full((4, 1), 0.3),
                                             np.array([0.6]), np.array([100.0]),
                                             surface_albedo=1.0, closure=closure)
        # nothing absorbs: the net flux is zero at every interface
        assert np.abs(down - up).max() < 1e-7        # of an incident 100


def test_absorbing_isothermal_column_is_exact_and_thin_layers_are_stable():
    tau = np.array([1e-12, 1e-8, 1e-4, 0.5, 30.0])[:, None]
    p = np.full_like(tau, 300.0)
    up, down = ts.thermal_fluxes(tau, p, p, surface_source=300.0, diffusivity=1.66)
    assert np.abs(up - 300.0).max() < 1e-10
    cumulative = np.concatenate([[0.0], np.cumsum(tau[:, 0])])
    assert np.abs(down[:, 0] - 300.0 * (1 - np.exp(-1.66 * cumulative))).max() < 1e-10
    # a linear source across a vanishing layer must not blow up
    up2, down2 = ts.thermal_fluxes(tau, p, 2 * p, surface_source=0.0, diffusivity=1.66)
    assert np.isfinite(up2).all() and np.isfinite(down2).all()


def test_two_stream_angular_error_against_exact_exponential_integrals():
    """The method's own limit: flux emissivity of an isothermal absorbing
    slab is exactly 1 - 2 E3(tau); the diffusivity approximation gives
    1 - exp(-D tau). The measured worst case documents the approximation."""
    tau = np.logspace(-3, 1.5, 400)
    exact = 1.0 - 2.0 * expn(3, tau)
    err = {d: np.abs((1.0 - np.exp(-d * tau)) - exact).max() for d in (1.5, 1.66, 2.0)}
    # Measured worst-case error in flux emissivity (fraction of pi B):
    # 0.038 (D = 1.5), 0.032 (D = 1.66), 0.087 (D = 2, hemispheric mean).
    assert err[1.66] == pytest.approx(0.0316, abs=1e-3)
    assert err[2.0] == pytest.approx(0.0868, abs=1e-3)
    assert err[1.66] < err[1.5] < err[2.0]


def test_adding_is_independent_of_how_a_homogeneous_slab_is_split():
    w, g, D = 0.6, 0.4, 1.66
    for n in (1, 7, 40):
        tau = np.full((n, 1), 3.0 / n)
        edges = 200.0 + 100.0 * np.linspace(0, 1, n + 1)[:, None]
        up, down = ts.thermal_fluxes(tau, edges[:-1], edges[1:], surface_source=120.0,
                                     surface_albedo=0.2, ssa=w + 0 * tau, asym=g + 0 * tau,
                                     diffusivity=D)
        if n == 1:
            ref = (up[0, 0], down[-1, 0])
        assert up[0, 0] == pytest.approx(ref[0], rel=1e-12)
        assert down[-1, 0] == pytest.approx(ref[1], rel=1e-12)


@pytest.mark.parametrize("closure,ssa,asym", [("hemispheric", 0.5, 0.0), ("quadrature", 0.3, 0.0),
                                              ("eddington", 0.2, 0.0)])
def test_direct_beam_at_the_removable_singularity_k_mu0_equal_one(closure, ssa, asym):
    """At ``k mu0 = 1`` the closed form is 0/0. A transparent layer must
    scatter nothing, and a finite one must agree with the incidence angles on
    either side of the singular one (the response is smooth through it)."""
    mu = np.sqrt(0.5)
    for _ in range(50):                      # solve k(mu0) mu0 = 1 (g3 may depend on mu0)
        g1, g2, g3 = ts.stellar_coefficients(ssa, asym, mu, closure)
        mu = 1.0 / np.sqrt((g1 - g2) * (g1 + g2))
    assert 0.0 < mu <= 1.0

    def layer(tau, mu0):
        g1, g2, g3 = ts.stellar_coefficients(ssa, asym, mu0, closure)
        return np.array(ts.direct_layer(np.float64(tau), ssa, g1, g2, g3, mu0))

    for tau in (0.0, 1e-8):
        r_dir, t_dir, t_beam = layer(tau, mu)
        assert r_dir <= 1e-7 and t_dir <= 1e-7
        assert t_beam == pytest.approx(1.0, abs=1e-7)
    for tau in (0.1, 1.0, 5.0):
        at = layer(tau, mu)
        around = 0.5 * (layer(tau, mu * (1 - 1e-3)) + layer(tau, mu * (1 + 1e-3)))
        np.testing.assert_allclose(at, around, rtol=0, atol=2e-6)
        assert at[0] > 0.0 and at[1] > 0.0

