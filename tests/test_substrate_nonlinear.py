"""G-B: nonlinear, stratified conduction with a radiative skin.

The checks are numbered after docs/validation/PLANET-ONE-GATES.md, G-B,
which was frozen on 2026-11-24 before any result existed. The references are
written out here, independently of the kernel: finite differences,
manufactured solutions, analytic Kirchhoff profiles and a dense linear
solve.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tropoi.physics.constants import SIGMA_SB
from tropoi.physics.materials import (CellMaterials, DepthProfile, PolynomialHeatCapacity,
                                      Stratum, stratified_interfaces)
from tropoi.physics.substrate import (ConductiveSubstrate, SubstrateSolveError,
                                      ENERGY_REFERENCE_TEMPERATURE as REF)

TR = 350.0
CHI = 2.0          # one chi in both strata keeps the manufactured interface exact


def _two_strata(zi=0.03, depth=0.25, kc=(0.004, 0.02), chi=(CHI, CHI)):
    """Heterogeneous test material. Every property is generic (not any real
    body's): density, contact conductivity and heat-capacity polynomial
    jump at zi."""
    return [
        Stratum(zi, 1300.0, kc[0], (150.0, 2.2, -1.5e-3), radiative_coefficient=chi[0],
                name="upper", source="test material"),
        Stratum(depth, DepthProfile(1500.0, 1900.0, 0.05), kc[1], (250.0, 1.8, -1.0e-3),
                radiative_coefficient=chi[1], name="lower", source="test material"),
    ]


def _faces(zi, depth, n_upper, n_lower, beta=1.5):
    """Smoothly stretched faces: in each stratum, z = top + h (e^(beta s) - 1)/(e^beta - 1)
    for uniform s. Refining doubles n and keeps the same mapping."""
    def seg(a, b, n):
        s = np.linspace(0.0, 1.0, n + 1)
        return a + (b - a) * np.expm1(beta * s) / math.expm1(beta)
    return np.concatenate([seg(0.0, zi, n_upper), seg(zi, depth, n_lower)[1:]])


# -- materials ----------------------------------------------------------------------

def test_material_differences_match_exact_integrals_without_cancellation():
    law = PolynomialHeatCapacity((150.0, 2.2, -1.5e-3))
    t1, t0 = np.array([[612.5]]), np.array([[97.25]])
    exact = sum(c / (j + 1) * (612.5 ** (j + 1) - 97.25 ** (j + 1))
                for j, c in enumerate((150.0, 2.2, -1.5e-3)))
    assert law.enthalpy_difference(t1, t0)[0, 0] == pytest.approx(exact, rel=1e-14)
    # tiny increments keep full relative precision
    t0 = np.array([[300.0]])
    for d in (1e-3, 1e-7, 1e-11):
        got = law.enthalpy_difference(t0 + d, t0, delta=np.array([[d]]))[0, 0]
        a0, a1, a2 = 150.0, 2.2, -1.5e-3
        # exact mean of c_p over [300, 300 + d]: cp + cp' d / 2 + cp'' d^2 / 6
        mean = (a0 + a1 * 300.0 + a2 * 300.0 ** 2) + (a1 + 2 * a2 * 300.0) * d / 2 + a2 * d * d / 3
        assert got / d == pytest.approx(mean, rel=1e-12)
    kc, chi = 0.01, 1.48 / TR ** 3
    a, b = 640.0, 95.0
    phi = lambda T: kc * (T + chi * T ** 4 / 4.0)
    assert CellMaterials.kirchhoff_difference(kc, chi, a, b) == pytest.approx(
        phi(a) - phi(b), rel=1e-14)


def test_strata_must_lie_on_faces_and_grids_honour_boundaries():
    strata = _two_strata()
    with pytest.raises(ValueError, match="not a cell face"):
        ConductiveSubstrate(strata, interfaces=np.linspace(0.0, 0.25, 8))
    edges = stratified_interfaces([0.03, 0.25], 1e-3, 1.2)
    assert any(math.isclose(e, 0.03, rel_tol=1e-12) for e in edges)
    dz = np.diff(edges)
    assert dz[0] == pytest.approx(1e-3) and edges[-1] == 0.25
    sub = ConductiveSubstrate(strata, top_thickness=1e-3, max_stretch=1.2)
    assert sub.materials.stratum_of_cell[0] == 0 and sub.materials.stratum_of_cell[-1] == 1


# -- 1: Jacobian ----------------------------------------------------------------------

def test_analytic_jacobian_matches_finite_differences():
    rng = np.random.default_rng(7)
    edges = _faces(0.03, 0.25, 5, 7)
    sub = ConductiveSubstrate(_two_strata(), interfaces=edges, albedo=0.1, emissivity=0.95)
    n, L = 4, sub.layers
    th_old = rng.uniform(-200.0, 300.0, size=(L, n))
    th = th_old + rng.uniform(-30.0, 30.0, size=(L, n))
    ths = th[0] + rng.uniform(-40.0, 40.0, size=n)
    forcing = (rng.uniform(0.0, 9000.0, n), -rng.uniform(0.0, 5.0, n),
               rng.uniform(-50.0, 50.0, n), np.full(n, 0.02), np.zeros((1, 1)))
    dt = 1800.0
    out = sub._system(np, th, ths, th_old, dt, forcing, jacobian=True)
    lower, diag, upper = out["jacobian"]
    x0 = np.concatenate([ths[None], th], axis=0)

    def resid(x):
        o = sub._system(np, x[1:], x[0], th_old, dt, forcing, jacobian=False)
        return np.concatenate([o["r_s"][None], o["r"]], axis=0)

    for c in range(n):
        dense = np.diag(diag[:, c]) + np.diag(lower[:, c], -1) + np.diag(upper[:, c], 1)
        fd = np.empty_like(dense)
        for j in range(L + 1):
            h = math.sqrt(np.finfo(float).eps) * (REF + abs(x0[j, c]))
            xp_, xm = x0.copy(), x0.copy()
            xp_[j, c] += h
            xm[j, c] -= h
            fd[:, j] = (resid(xp_)[:, c] - resid(xm)[:, c]) / (2 * h)
        scale = np.max(np.abs(dense), axis=1, keepdims=True)
        assert np.max(np.abs(fd - dense) / scale) <= 1e-6


# -- 2: manufactured nonlinear transient ------------------------------------------------

ZI, DEPTH, A_DEC, B_DEC = 0.03, 0.25, 0.04, 0.08
T0, AMP, OMEGA = 240.0, 120.0, 2.0 * math.pi / 4.0e4
ALBEDO, EMISS = 0.1, 0.95


def _psi(z, kc):
    """Shape function with k-weighted slope continuity at ZI (same chi above and below)."""
    z = np.asarray(z, dtype=float)
    up = np.exp(-z / A_DEC)
    ei = math.exp(-ZI / A_DEC)
    slope = -(kc[0] / kc[1]) * ei / A_DEC
    low = ei + slope * B_DEC * (1.0 - np.exp(-(z - ZI) / B_DEC))
    d_up, d_low = -up / A_DEC, slope * np.exp(-(z - ZI) / B_DEC)
    return np.where(z <= ZI, up, low), np.where(z <= ZI, d_up, d_low)


def _exact(z, t, kc):
    psi, dpsi = _psi(z, kc)
    a = AMP * (1.0 + 0.5 * math.sin(OMEGA * t))
    da = AMP * 0.5 * OMEGA * math.cos(OMEGA * t)
    return T0 + a * psi, a * dpsi, da * psi


def _manufactured_run(edges, dt, t_end, strata):
    kc = (strata[0].contact_conductivity, strata[1].contact_conductivity)
    sub = ConductiveSubstrate(strata, interfaces=edges, albedo=ALBEDO, emissivity=EMISS)
    xg, wg = np.polynomial.legendre.leggauss(8)
    e = sub.interfaces
    L = sub.layers
    # cell averages of the exact solution as the initial state
    def cell_avg(fun):
        out = np.empty(L)
        for i in range(L):
            z = 0.5 * (e[i] + e[i + 1]) + 0.5 * (e[i + 1] - e[i]) * xg
            out[i] = 0.5 * np.dot(wg, fun(z))
        return out
    init = cell_avg(lambda z: _exact(z, 0.0, kc)[0])
    st = sub.state_at(init[:, None], np.array([_exact(0.0, 0.0, kc)[0]]))
    which = sub.materials.stratum_of_cell
    steps = int(round(t_end / dt))
    for n in range(1, steps + 1):
        t = n * dt
        src = np.empty((L, 1))
        for i in range(L):
            s = strata[which[i]]
            z = 0.5 * (e[i] + e[i + 1]) + 0.5 * (e[i + 1] - e[i]) * xg
            T, dT, Tt = _exact(z, t, kc)
            rho = s.density(z) if isinstance(s.density, DepthProfile) else s.density
            cp = s.heat_capacity.specific_heat(T)
            storage = 0.5 * (e[i + 1] - e[i]) * np.dot(wg, rho * cp * Tt)

            def down_flux(zz):
                # the exact flux is continuous; evaluate it with the material
                # on the same side of ZI as the shape function's branch
                TT, dd, _ = _exact(np.array([zz]), t, kc)
                kcz = kc[0] if zz <= ZI else kc[1]
                k = kcz * (1 + CHI * (TT[0] / TR) ** 3)
                return -k * dd[0]
            src[i, 0] = storage - (down_flux(e[i]) - down_flux(e[i + 1]))
        Ts, dT0, _ = _exact(np.array([0.0]), t, kc)
        k0 = kc[0] * (1 + CHI * (Ts / TR) ** 3)
        q0 = -k0 * dT0                                     # exact downward surface flux
        sw = (EMISS * SIGMA_SB * Ts ** 4 + q0) / (1.0 - ALBEDO)
        assert np.all(sw > 0.0)
        Tb, dTb, _ = _exact(np.array([DEPTH]), t, kc)
        kb = kc[1] * (1 + CHI * (Tb / TR) ** 3)
        g_up = float((kb * dTb)[0])                         # -q_down(D)
        st, _ = sub.step(st, dt, shortwave_down=sw, bottom_heat_flux=g_up, source=src)
    T_num = sub.temperature(st)[:, 0]
    Ts_num = float(sub.surface_temperature(st)[0])
    T_ex = cell_avg(lambda z: _exact(z, t_end, kc)[0])
    return T_num, T_ex, Ts_num, float(_exact(np.array([0.0]), t_end, kc)[0][0])


def _order(errors):
    return [math.log(errors[i] / errors[i + 1], 2) for i in range(len(errors) - 1)]


def test_manufactured_solution_space_order():
    strata = _two_strata(kc=(0.004, 0.02))
    t_end = 4.0e4
    errs, serrs = [], []
    for k, (nu, nl) in enumerate(((4, 6), (8, 12), (16, 24), (32, 48))):
        dt = 2000.0 / 4 ** k                       # dt ~ dz^2: time error shrinks with space error
        T_num, T_ex, Ts_num, Ts_ex = _manufactured_run(_faces(ZI, DEPTH, nu, nl), dt, t_end, strata)
        errs.append(np.max(np.abs(T_num - T_ex)))
        serrs.append(abs(Ts_num - Ts_ex))
    print("cell errors", errs, "orders", _order(errs))
    print("skin errors", serrs, "orders", _order(serrs))
    assert 1.8 <= _order(errs)[-1] <= 2.2
    assert 1.8 <= _order(serrs)[-1] <= 2.2


def test_manufactured_solution_time_order():
    strata = _two_strata(kc=(0.004, 0.02))
    edges = _faces(ZI, DEPTH, 16, 24)
    t_end = 4.0e4
    runs = [_manufactured_run(edges, dt, t_end, strata) for dt in (4000.0, 2000.0, 1000.0, 500.0, 250.0)]
    # self-convergence on one grid isolates the time error
    errs = [np.max(np.abs(runs[i][0] - runs[i + 1][0])) for i in range(len(runs) - 1)]
    serrs = [abs(runs[i][2] - runs[i + 1][2]) for i in range(len(runs) - 1)]
    print("cell differences", errs, "orders", _order(errs))
    print("skin differences", serrs, "orders", _order(serrs))
    assert 0.9 <= _order(errs)[-1] <= 1.1
    assert 0.9 <= _order(serrs)[-1] <= 1.1


# -- 3: steady Kirchhoff analytic --------------------------------------------------------

def _steady_analytic(z, strata, ts, g):
    """Phi_m(T(z)) = Phi_m(T(z_m)) + G (z - z_m), per stratum, T continuous."""
    def invert(kc, chi, target_phi, guess):
        T = guess
        for _ in range(100):
            f = kc * (T + chi / TR ** 3 * T ** 4 / 4) - target_phi
            T -= f / (kc * (1 + chi * (T / TR) ** 3))
        return T
    out = np.empty_like(z)
    top, t_top = 0.0, ts
    for s in strata:
        kc, chi = s.contact_conductivity, s.radiative_coefficient
        phi_top = kc * (t_top + chi / TR ** 3 * t_top ** 4 / 4)
        sel = (z >= top) & (z <= s.bottom)
        for i in np.flatnonzero(sel):
            out[i] = invert(kc, chi, phi_top + g * (z[i] - top), t_top)
        t_top = invert(kc, chi, phi_top + g * (s.bottom - top), t_top)
        top = s.bottom
    return out


def test_steady_kirchhoff_profile_and_interface_flux():
    strata = [Stratum(0.03, 1300.0, 9e-4, (600.0,), radiative_coefficient=1.5, name="a"),
              Stratum(0.6, 1800.0, 9e-3, (700.0,), radiative_coefficient=0.07, name="b")]
    F, G = 3000.0, 0.5
    ts = ((ALBEDO * 0 + (1 - ALBEDO) * F + G) / (EMISS * SIGMA_SB)) ** 0.25
    errs = []
    for k, (nu, nl) in enumerate(((5, 10), (10, 20), (20, 40), (40, 80))):
        e = _faces(0.03, 0.6, nu, nl)
        sub = ConductiveSubstrate(strata, interfaces=e, albedo=ALBEDO, emissivity=EMISS,
                                  bottom_heat_flux=G)
        centres = 0.5 * (e[1:] + e[:-1])
        exact = _steady_analytic(centres, strata, ts, G)
        st = sub.state_at(exact[:, None] + 5.0, np.array([ts + 5.0]))
        for _ in range(3):                                      # steady: dt -> infinity
            st, rep = sub.step(st, 1e25, shortwave_down=F)
        T = sub.temperature(st)[:, 0]
        errs.append(np.max(np.abs(T - exact)))
        out = sub._system(np, st.anomaly, st.surface_anomaly, st.anomaly, 1e25,
                          ((1 - ALBEDO) * np.array([F]), np.zeros(1), np.zeros(1),
                           np.array([G]), np.zeros((1, 1))), jacobian=False)
        assert np.max(np.abs(out["faces"][:, 0] + G)) <= 1e-9 + 1e-12 * G   # q = -G everywhere
        Ts = float(sub.surface_temperature(st)[0])
        assert EMISS * SIGMA_SB * Ts ** 4 == pytest.approx((1 - ALBEDO) * F + G, rel=1e-10)
    print("steady errors", errs, "orders", _order(errs))
    assert 1.8 <= _order(errs)[-1] <= 2.2


# -- 4: zero-conductivity limit ----------------------------------------------------------

def test_vanishing_conductivity_tends_to_radiative_equilibrium():
    F = 10000.0
    t_eq = ((1 - ALBEDO) * F / (EMISS * SIGMA_SB)) ** 0.25
    gaps = []
    for scale in (1e-2, 1e-4, 1e-6, 1e-8):
        strata = [Stratum(0.5, 1500.0, 0.01 * scale, (700.0,), radiative_coefficient=1.0)]
        sub = ConductiveSubstrate(strata, top_thickness=1e-3, max_stretch=1.2,
                                  albedo=ALBEDO, emissivity=EMISS, bottom_heat_flux=0.0,
                                  initial_temperature=100.0)
        st = sub.initial_state(1)
        for _ in range(60):
            st, _ = sub.step(st, 60.0, shortwave_down=F)
        gaps.append(t_eq - float(sub.surface_temperature(st)[0]))
    assert all(g > 0 for g in gaps) and all(b < a for a, b in zip(gaps, gaps[1:]))
    tiny = [Stratum(0.5, 1500.0, 1e-30, (700.0,))]
    sub = ConductiveSubstrate(tiny, top_thickness=1e-3, max_stretch=1.2, albedo=ALBEDO,
                              emissivity=EMISS, initial_temperature=100.0)
    st, _ = sub.step(sub.initial_state(1), 60.0, shortwave_down=F)
    assert float(sub.surface_temperature(st)[0]) == pytest.approx(t_eq, rel=1e-12)


# -- 5: constant-property limit with linear boundary conditions ---------------------------

@pytest.mark.parametrize("h", [0.0, 6.0])
def test_constant_property_linear_problem_is_one_newton_step(h):
    rng = np.random.default_rng(11)
    C, k = 1.2e6, 0.8
    e = _faces(0.1, 1.0, 6, 10)
    sub = ConductiveSubstrate([Stratum.volumetric(0.1, C, k), Stratum.volumetric(1.0, 2 * C, 3 * k)],
                              interfaces=e, emissivity=0.0, bottom_heat_flux=0.3)
    L = sub.layers
    st = sub.state_at(300.0 + rng.uniform(-20, 20, size=(L, 1)), np.array([310.0]))
    F0, dt = 150.0, 3600.0
    new, rep = sub.step(st, dt, flux=F0, flux_derivative=-h)
    assert rep.iterations == 1
    # independent dense solve in anomalies: unknowns (theta_s, theta_1..L)
    dz = np.diff(e)
    cap = np.where(np.arange(L) < np.searchsorted(e, 0.1) , C, 2 * C) * dz
    kk = np.where(np.arange(L) < np.searchsorted(e, 0.1), k, 3 * k)
    A = np.zeros((L + 1, L + 1))
    b = np.zeros(L + 1)
    th0 = st.anomaly[:, 0]
    ts_ref = float(st.surface_anomaly[0])
    gs = 2 * kk[0] / dz[0]
    A[0, 0], A[0, 1] = -h - gs, gs                  # f(theta_s) - q_s = 0
    b[0] = -(F0 + h * ts_ref)
    for i in range(L):
        A[i + 1, i + 1] += cap[i] / dt
        b[i + 1] += cap[i] / dt * th0[i]
        if i == 0:
            A[1, 0] -= gs
            A[1, 1] += gs
        if i < L - 1:
            g = 1 / (dz[i] / (2 * kk[i]) + dz[i + 1] / (2 * kk[i + 1]))
            A[i + 1, i + 1] += g
            A[i + 1, i + 2] -= g
            A[i + 2, i + 2] += g
            A[i + 2, i + 1] -= g
    b[L] += 0.3
    x = np.linalg.solve(A, b)
    got = np.concatenate([new.surface_anomaly, new.anomaly[:, 0]])
    assert np.max(np.abs(got - x)) <= 1e-12 * np.max(np.abs(x))


# -- 6: stress and failure reporting ------------------------------------------------------

def test_cold_start_into_perihelion_noon():
    sub = ConductiveSubstrate(_two_strata(), top_thickness=5e-4, max_stretch=1.2,
                              albedo=ALBEDO, emissivity=EMISS, initial_temperature=40.0,
                              bottom_heat_flux=0.02)
    st = sub.initial_state(3)
    iters = []
    for _ in range(20):
        st, rep = sub.step(st, 3600.0, shortwave_down=np.array([14500.0, 7000.0, 0.0]))
        iters.append(rep.iterations)
        assert np.all(sub.temperature(st) > 0) and np.all(sub.surface_temperature(st) > 0)
    print("iterations per step", iters)
    assert max(iters) <= 30
    with pytest.raises(SubstrateSolveError, match=r"columns \(first \[0, 1"):
        weak = ConductiveSubstrate(_two_strata(), top_thickness=5e-4, max_stretch=1.2,
                                   albedo=ALBEDO, emissivity=EMISS, initial_temperature=40.0,
                                   max_iterations=1)
        weak.step(weak.initial_state(3), 3600.0, shortwave_down=np.array([14500.0, 7000.0, 0.0]))


# -- 7: energy closure ------------------------------------------------------------------------

def test_energy_identity_and_cumulative_closure_under_random_forcing():
    rng = np.random.default_rng(5)
    sub = ConductiveSubstrate(_two_strata(), top_thickness=1e-3, max_stretch=1.25,
                              albedo=ALBEDO, emissivity=EMISS, initial_temperature=250.0)
    n = 3
    st = sub.initial_state(n)
    E0 = np.asarray(sub.energy(st)).copy()
    net_sum = [[] for _ in range(n)]
    bound_sum = np.zeros(n)
    dt = 900.0
    for _ in range(400):
        sw = rng.uniform(0.0, 12000.0, n) * (rng.uniform(size=n) > 0.4)
        gb = rng.uniform(-0.05, 0.05, n)
        st, rep = sub.step(st, dt, shortwave_down=sw, bottom_heat_flux=gb)
        net = rep.absorbed - rep.emitted + rep.bottom + rep.source
        # the identity, term by term (sign of r_s included)
        assert np.all(np.abs(rep.energy_tendency - net - rep.residual_sum)
                      <= 1e-12 * (np.abs(rep.energy_tendency) + np.abs(net)) + 1e-12)
        assert np.all(np.abs(rep.residual_sum) <= rep.tolerance_sum)
        for c in range(n):
            net_sum[c].append(dt * float(net[c]))
        bound_sum += dt * rep.tolerance_sum
    E = np.asarray(sub.energy(st))
    for c in range(n):
        inflow = math.fsum(net_sum[c])
        # the energy diagnostic itself carries the cells' round-off, C ulp(theta)
        diag = 4 * np.finfo(float).eps * float(np.sum(np.abs(
            sub.materials.density[:, 0] * sub.thickness * 1000.0 * np.abs(st.anomaly[:, c]))))
        assert abs(E[c] - E0[c] - inflow) <= bound_sum[c] + diag


# -- 8: validity ranges ---------------------------------------------------------------------

def test_stratum_extents_are_recorded_for_validity_checks():
    sub = ConductiveSubstrate(_two_strata(), top_thickness=1e-3, max_stretch=1.25,
                              albedo=ALBEDO, emissivity=EMISS, initial_temperature=150.0)
    st, _ = sub.step(sub.initial_state(2), 3600.0, shortwave_down=np.array([9000.0, 0.0]))
    (lo0, hi0), (lo1, hi1) = sub.extent(st)
    assert hi0 == pytest.approx(float(np.max(sub.surface_temperature(st))))
    assert lo0 <= hi0 and lo1 <= hi1 and hi1 < hi0
