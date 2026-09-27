"""Held–Suarez forcing, Rayleigh drag, del^8 hyperdiffusion and the stepper
hooks that carry them (plan S4; docs/held_suarez/SEMI_IMPLICIT.md §8).

CPU part (NumPy, runs in CI): T_eq, k_T, k_v against independently
computed values (Wolfram Language, exact rational constants) at
representative points including every regime boundary; exact zero forcing at
T = T_eq and V = 0; pointwise relaxation at rate k_T; the backward-Euler
Rayleigh and del^8 factors degree by degree; hook placement inside the SI
leapfrog step (explicit term in N, damping after the solve, before RAW),
the startup on the complete right-hand side, and hook signatures in the
state dict.

GPU part (CuPy): the spectral Newtonian term decays at k_a where k_T is
uniform; isothermal rest is preserved bitwise with drag + del^8 + SI + RAW;
a resting, horizontally uniform T_eq profile stays at rest with the full
forcing.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tropoi.temporal.tendencies.held_suarez import (
    DAY_SECONDS, HeldSuarezParameters, equilibrium_temperature,
    hyperdiffusion_coefficient, hyperdiffusion_damping, newtonian_rate,
    newtonian_tendency_grid, rayleigh_drag_damping, rayleigh_rate)

HS = HeldSuarezParameters()
DEG = math.pi / 180.0


def _has_cuda():
    try:
        import cupy as cp
        return cp.is_available()
    except Exception:
        return False


gpu = pytest.mark.skipif(not _has_cuda(), reason="CUDA/CuPy not available")


# ---------------------------------------------------------------------------
# 1. formulas against independent hand values
# ---------------------------------------------------------------------------

# (lat deg, sigma, p_s Pa) -> (T_eq K, k_T s^-1, k_v s^-1), computed in the
# Wolfram Language from the PROTOCOL §1.1 formulas with exact constants
# (k_a = 1/(40 d), k_s = 1/(4 d), k_f = 1/d, sigma_b = 7/10, kappa = 2/7).
HAND = [
    # surface, equator: T_eq = 315 K, k_T = k_s, k_v = k_f (upper ends)
    ((0.0, 1.0, 1.0e5), (315.0, 2.8935185185185185e-06, 1.1574074074074074e-05)),
    # surface, pole: T_eq = 315 - 60, k_T = k_a (cos^4 = 0)
    ((90.0, 1.0, 1.0e5), (255.0, 2.8935185185185185e-07, 1.1574074074074074e-05)),
    # sigma = sigma_b exactly: k_T = k_a, k_v = 0 (ramp boundary)
    ((45.0, 0.7, 1.0e5), (258.99791513716965, 2.8935185185185185e-07, 0.0)),
    # boundary layer, mid-latitude: ramp 1/2, cos^4 = 1/4
    ((45.0, 0.85, 1.0e5), (272.84458645918735, 6.1487268518518519e-07, 5.7870370370370370e-06)),
    # upper troposphere: raw 197.55 K -> floor 200 K
    ((30.0, 0.2, 1.01e5), (200.0, 2.8935185185185185e-07, 0.0)),
    # stratosphere, equator: raw 146.57 K -> floor
    ((0.0, 0.05, 1.0e5), (200.0, 2.8935185185185185e-07, 0.0)),
    # southern hemisphere, free troposphere (symmetry in phi)
    ((-60.0, 0.55, 9.8e4), (227.58993144543814, 2.8935185185185185e-07, 0.0)),
    ((-60.0, 0.25, 9.8e4), (200.0, 2.8935185185185185e-07, 0.0)),        # raw 183.00
    # pole, just above and just below the 200 K floor
    ((90.0, 0.43, 1.0e5), (200.36270304906126, 2.8935185185185185e-07, 0.0)),
    ((90.0, 0.42, 1.0e5), (200.0, 2.8935185185185185e-07, 0.0)),         # raw 199.02
    # near-surface, low latitude, high p_s
    ((15.0, 0.975, 1.025e5), (310.93104954512741, 2.3673986662647018e-06, 1.0609567901234568e-05)),
]


def test_equilibrium_temperature_and_rates_match_hand_values():
    worst = 0.0
    print("\n   lat  sigma     p_s  |   T_eq (code)    hand   |  k_T*day  |  k_v*day")
    for (lat, sig, ps), (t_hand, kt_hand, kv_hand) in HAND:
        t = float(equilibrium_temperature(lat * DEG, sig * ps, HS))
        kt = float(newtonian_rate(lat * DEG, sig, HS))
        kv = float(rayleigh_rate(sig, HS))
        print(f"{lat:6.1f} {sig:6.3f} {ps:8.0f} | {t:10.6f} {t_hand:10.6f} | "
              f"{kt * DAY_SECONDS:.6f} | {kv * DAY_SECONDS:.6f}")
        for got, want in ((t, t_hand), (kt, kt_hand), (kv, kv_hand)):
            if want == 0.0:
                assert got == 0.0
            else:
                worst = max(worst, abs(got - want) / abs(want))
    print(f"worst relative difference from the hand values: {worst:.2e}")
    assert worst < 5e-15


def test_forcing_regime_boundaries_are_exact():
    # ramp: exactly zero at and above sigma_b, exactly k_f at sigma = 1
    assert rayleigh_rate(0.7, HS) == 0.0 and rayleigh_rate(0.3, HS) == 0.0
    assert rayleigh_rate(1.0, HS) == HS.k_f
    assert newtonian_rate(0.3, 0.7, HS) == HS.k_a
    # the floor is a max(): continuous, never below T_min
    lat = np.linspace(-math.pi / 2, math.pi / 2, 181)
    for p in (1e3, 5e3, 2e4, 5e4, 1e5):
        assert np.all(equilibrium_temperature(lat, p, HS) >= HS.t_min)
    # symmetric in latitude
    assert np.array_equal(equilibrium_temperature(lat, 6e4, HS),
                          equilibrium_temperature(-lat, 6e4, HS))


# ---------------------------------------------------------------------------
# 2. forcing is exactly zero at T = T_eq, V = 0
# ---------------------------------------------------------------------------

def test_newtonian_forcing_is_exactly_zero_at_equilibrium_temperature():
    rng = np.random.default_rng(0)
    K, npts = 10, 400
    lat = rng.uniform(-math.pi / 2, math.pi / 2, npts)
    sigma = (np.arange(K) + 0.5) / K
    ps = 1e5 + 2e3 * rng.standard_normal(npts)
    t_eq = equilibrium_temperature(lat[None, :], sigma[:, None] * ps[None, :], HS)
    f = newtonian_tendency_grid(t_eq, lat, sigma, ps, HS)
    assert f.shape == (K, npts)
    assert np.all(f == 0.0)
    # and Rayleigh drag of a zero wind is exactly zero
    drag = rayleigh_drag_damping(sigma, 8, HS)
    x = np.zeros((3 * K + 1, 8, 8), dtype=np.complex128)
    assert np.all(drag.tendency(x) == 0.0)
    assert np.array_equal(drag.apply_implicit(x, 2400.0), x)


# ---------------------------------------------------------------------------
# 3. pure relaxation decays at k_T
# ---------------------------------------------------------------------------

def test_pure_newtonian_relaxation_decays_at_k_T_pointwise():
    """dT/dt = -k_T (T - T_eq) integrated alone (RK4, 10 days, dt = 600 s):
    T - T_eq = (T0 - T_eq) exp(-k_T t) at every point and level."""
    from tropoi.temporal.integration import rk4_step_array
    lat = np.linspace(-89.0, 89.0, 37) * DEG
    sigma = np.array([0.05, 0.3, 0.7, 0.75, 0.85, 0.95, 0.99])
    ps = np.full(lat.size, 1.0e5)
    t_eq = equilibrium_temperature(lat[None, :], sigma[:, None] * ps[None, :], HS)
    t0 = np.full_like(t_eq, 300.0)
    dt, t_end = 600.0, 10 * DAY_SECONDS
    T = t0.copy()
    for _ in range(int(t_end / dt)):
        T = rk4_step_array(lambda y: newtonian_tendency_grid(y, lat, sigma, ps, HS), T, 0.0, dt)
    k = newtonian_rate(lat[None, :], sigma[:, None], HS)
    exact = t_eq + (t0 - t_eq) * np.exp(-k * t_end)
    err = np.abs(T - exact).max() / np.abs(t0 - t_eq).max()
    measured = -np.log((T - t_eq) / (t0 - t_eq)) / t_end
    print(f"\nrelaxation after 10 d: max rel. deviation from exp(-k_T t) {err:.2e}; "
          f"measured rate / k_T in [{(measured / k).min():.12f}, {(measured / k).max():.12f}]")
    assert err < 1e-12
    assert np.allclose(measured, k, rtol=1e-9, atol=0)


# ---------------------------------------------------------------------------
# 4. Rayleigh and del^8 factors are the analytic backward-Euler factors
# ---------------------------------------------------------------------------

def test_rayleigh_and_hyperdiffusion_factors_are_backward_euler_per_degree():
    K, l_max, a, dt = 10, 21, 6.371e6, 1200.0
    sigma = (np.arange(K) + 0.5) / K
    tau = 0.1 * DAY_SECONDS
    hyper = hyperdiffusion_damping(K, l_max, a, tau, reference_degree=l_max)
    drag = rayleigh_drag_damping(sigma, l_max + 1, HS)
    K8 = hyperdiffusion_coefficient(a, tau, l_max)
    c_L = l_max * (l_max + 1.0) / a ** 2
    assert K8 * c_L ** 4 * tau == pytest.approx(1.0, rel=1e-15)   # e-folding tau at l = L
    fh = hyper.factor(2.0 * dt)
    fd = drag.factor(2.0 * dt)
    worst = 0.0
    for l in range(l_max + 1):
        c_l = l * (l + 1.0) / a ** 2
        want = 1.0 / (1.0 + 2 * dt * K8 * c_l ** 4)
        for row in range(3 * K):                               # zeta, delta, T
            worst = max(worst, abs(fh[row, l] - want) / want)
        assert fh[3 * K, l] == 1.0                             # ln p_s not diffused
        for k in range(K):
            want_d = 1.0 / (1.0 + 2 * dt * rayleigh_rate(sigma[k], HS))
            assert fd[k, l] == pytest.approx(want_d, rel=1e-15)      # zeta_k
            assert fd[K + k, l] == pytest.approx(want_d, rel=1e-15)  # delta_k
        assert np.all(fd[2 * K:, l] == 1.0)                    # T, ln p_s undragged
    assert np.all(fh[:, 0] == 1.0)                             # T' only: mean untouched
    print(f"\ndel^8 factor at l = 1, 14, 21 (dt = {dt:g} s): "
          f"{fh[0, 1]:.15f}, {fh[0, 14]:.12f}, {fh[0, 21]:.12f}; "
          f"max rel. deviation from 1/(1+2dt K8 c_l^4) {worst:.1e}")
    print("Rayleigh factor per level: "
          + ", ".join(f"{fd[k, 0]:.9f}" for k in range(K)))
    assert worst <= 2.3e-16
    # applying the damper is exactly one multiplication by the factor
    x = (np.random.default_rng(1).standard_normal((3 * K + 1, l_max + 1, l_max + 1))
         + 0j)
    y = hyper.apply_implicit(x, 2.0 * dt)
    assert np.array_equal(y, x * fh[:, :, None])


# ---------------------------------------------------------------------------
# hook placement inside the SI step (CPU, pure linear tendency)
# ---------------------------------------------------------------------------

def _operator(K=4, l_max=9):
    from tropoi.spatial.sigma_coordinate import SigmaGrid
    from tropoi.temporal.semi_implicit import SemiImplicitOperator
    return SemiImplicitOperator(SigmaGrid.uniform(K), l_max=l_max, radius=6.371e6,
                                r_dry=287.0, cp_dry=1004.0, t_ref=300.0)


def _state(op, seed):
    rng = np.random.default_rng(seed)
    K, n = op.K, op.l_max + 1
    y = np.tril(rng.standard_normal((3 * K + 1, n, n))
                + 1j * rng.standard_normal((3 * K + 1, n, n)))
    y[:, :, 0] = y[:, :, 0].real
    y[:, op.fast_cut + 1:, :] = 0.0
    y[0:2 * K] *= 1e-6
    y[2 * K:3 * K] *= 0.5
    y[3 * K] *= 1e-3
    y[2 * K:3 * K, 0, 0] = 300.0 * math.sqrt(4 * math.pi)
    y[3 * K, 0, 0] = math.log(1e5) * math.sqrt(4 * math.pi)
    return y


class _LinearTerm:
    """A toy explicit term (-k T in the T rows) with a signature."""

    def __init__(self, K, k):
        self.K, self.k = K, k
        self.max_rate = k

    def __call__(self, x):
        out = np.zeros_like(x)
        out[2 * self.K:3 * self.K] = -self.k * x[2 * self.K:3 * self.K]
        return out

    def signature(self):
        return {"kind": "toy", "k": self.k}


def test_hooks_enter_the_si_step_in_the_documented_places():
    from tropoi.temporal.hooks import complete_tendency
    from tropoi.temporal.integration import rk4_step_array
    from tropoi.temporal.semi_implicit import SemiImplicitLeapfrogStepper, SemiImplicitSolver
    op = _operator()
    K, dt, nu, alpha = op.K, 900.0, 0.1, 0.53
    term = _LinearTerm(K, 3e-6)
    sigma = op.sigma.full_levels_array()
    drag = rayleigh_drag_damping(sigma, op.n, HS)
    hyper = hyperdiffusion_damping(K, op.l_max, op.radius, 0.1 * DAY_SECONDS, op.l_max)
    st = SemiImplicitLeapfrogStepper(op.apply, op, dt, raw_nu=nu, raw_alpha=alpha,
                                     explicit_terms=(term,), dampers=(drag, hyper))
    y0 = _state(op, 2)
    st.initialize(y0)
    st.step()
    # startup: RK4 of the COMPLETE right-hand side over dt in n_sub substeps
    f = complete_tendency(op.apply, (term,), (drag, hyper))
    y1 = y0
    for _ in range(st.startup_substeps):
        y1 = rk4_step_array(f, y1, 0.0, dt / st.startup_substeps)
    assert np.array_equal(st.state, y1) and np.array_equal(st.x_prev, y0)
    # one leapfrog step by hand: N = tendency + term - L X^n; advance;
    # drag then del^8 factors on X^{n+1}; then RAW
    x_prev, x_n = st.x_prev.copy(), st.state.copy()
    solver = SemiImplicitSolver(op, dt)
    N = op.apply(x_n) + term(x_n) - op.apply(x_n)
    x_np1 = solver.advance(x_prev, N)
    x_np1 = x_np1 * drag.factor(2 * dt)[:, :, None]
    x_np1 = x_np1 * hyper.factor(2 * dt)[:, :, None]
    d = nu * (x_prev - 2.0 * x_n + x_np1)
    st.step()
    assert np.array_equal(st.x_prev, x_n + alpha * d)
    assert np.array_equal(st.state, x_np1 - (1.0 - alpha) * d)


def test_no_hooks_is_the_s3_step_bitwise_and_hooks_are_in_the_state_dict():
    from tropoi.temporal.semi_implicit import SemiImplicitLeapfrogStepper
    op = _operator()
    dt = 1200.0
    y0 = _state(op, 3)
    plain = SemiImplicitLeapfrogStepper(op.apply, op, dt)
    empty = SemiImplicitLeapfrogStepper(op.apply, op, dt, explicit_terms=(), dampers=())
    assert empty._startup_tendency == op.apply          # no wrapper at all
    for st in (plain, empty):
        st.initialize(y0)
        for _ in range(4):
            st.step()
    assert plain.state.tobytes() == empty.state.tobytes()
    assert plain.state_dict()["hooks"] == []
    drag = rayleigh_drag_damping(op.sigma.full_levels_array(), op.n, HS)
    hooked = SemiImplicitLeapfrogStepper(op.apply, op, dt, dampers=(drag,))
    hooked.initialize(y0)
    hooked.step(); hooked.step()
    saved = hooked.state_dict()
    assert saved["hooks"][0]["name"] == "held_suarez_rayleigh"
    with pytest.raises(ValueError, match="hooks"):
        SemiImplicitLeapfrogStepper(op.apply, op, dt).load_state_dict(saved)
    other = rayleigh_drag_damping(op.sigma.full_levels_array(), op.n,
                                  HeldSuarezParameters(k_f=2.0 / DAY_SECONDS))
    with pytest.raises(ValueError, match="hooks"):
        SemiImplicitLeapfrogStepper(op.apply, op, dt, dampers=(other,)).load_state_dict(saved)
    same = SemiImplicitLeapfrogStepper(op.apply, op, dt, dampers=(drag,))
    same.load_state_dict(saved)
    hooked.step(); same.step()
    assert hooked.state.tobytes() == same.state.tobytes()


def test_startup_substeps_account_for_damping_rates():
    """With damping the startup must keep |R_4((-r + i w) h)| <= 1; a
    damper far stiffer than the gravity waves forces more substeps."""
    from tropoi.temporal.hooks import DiagonalDamping, rk4_real_axis_limit
    from tropoi.temporal.semi_implicit import SemiImplicitLeapfrogStepper
    op = _operator()
    dt = 1200.0
    base = SemiImplicitLeapfrogStepper(op.apply, op, dt).startup_substeps
    hyper = hyperdiffusion_damping(op.K, op.l_max, op.radius, 0.1 * DAY_SECONDS, op.l_max)
    assert SemiImplicitLeapfrogStepper(op.apply, op, dt, dampers=(hyper,)).startup_substeps == base
    stiff = DiagonalDamping("stiff", np.full((3 * op.K + 1, op.n), 10.0 / dt))
    n_sub = SemiImplicitLeapfrogStepper(op.apply, op, dt, dampers=(stiff,)).startup_substeps
    assert 10.0 / n_sub <= rk4_real_axis_limit() + 1e-12
    assert 10.0 / (n_sub - 1) > rk4_real_axis_limit()
    print(f"\nstartup substeps: {base} (L only, and with del^8); {n_sub} with rate 10/dt; "
          f"RK4 real-axis limit {rk4_real_axis_limit():.6f}")


# ---------------------------------------------------------------------------
# GPU: the spectral Newtonian term, and rest configurations
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def hs_model():
    from tropoi.run.held_suarez.config import development_config
    from tropoi.run.held_suarez.model import build_model
    return build_model(development_config())


@gpu
def test_spectral_newtonian_term_decays_at_k_a_where_k_T_is_uniform(hs_model):
    """Above sigma_b, k_T = k_a everywhere and, with uniform p_s, T_eq is a
    fixed field: the spectral term is -k_a (T_lm - P_cut T_eq) and the
    departure decays as exp(-k_a t) mode by mode (RK4, dt = 1 h, 5 days)."""
    import cupy as cp
    from tropoi.temporal.hooks import complete_tendency
    from tropoi.temporal.integration import rk4_step_array
    from tropoi.temporal.tendencies.held_suarez import NewtonianRelaxation
    from tropoi.temporal.tendencies.primitive_equations import isothermal_rest_state
    model = hs_model
    K = model.nlev
    newton = NewtonianRelaxation(model, HS)
    sigma = model.sigma.full_levels_array()
    upper = np.where(sigma <= HS.sigma_b)[0]
    x0 = isothermal_rest_state(model.l_max, K, temperature=300.0,
                               surface_pressure=1e5).coeffs
    # fixed point of the upper-level relaxation: T_lm = P_cut T_eq (from the hook itself)
    f0 = newton(x0)
    t_star = x0[2 * K:3 * K] + f0[2 * K:3 * K] / HS.k_a
    zero = lambda x: cp.zeros_like(x)
    rhs = complete_tendency(zero, (newton,))
    dt, n = 3600.0, 120
    x = x0
    for _ in range(n):
        x = rk4_step_array(rhs, x, 0.0, dt)
    dev0 = (x0[2 * K:3 * K] - t_star)[upper]
    dev = (x[2 * K:3 * K] - t_star)[upper]
    decay = math.exp(-HS.k_a * n * dt)
    err = float(cp.abs(dev - dev0 * decay).max() / cp.abs(dev0).max())
    print(f"\nupper levels {[int(k) for k in upper]}: |dev(5 d) - dev(0) exp(-k_a t)| / |dev(0)| = {err:.2e} "
          f"(exp(-k_a t) = {decay:.9f})")
    assert err < 1e-9
    # rows other than T are untouched, and nothing appears above the cut
    assert cp.all(f0[0:2 * K] == 0) and cp.all(f0[3 * K] == 0)
    cut = model._trunc_cut
    assert cp.all(f0[:, cut + 1:, :] == 0) and cp.all(f0[:, :, cut + 1:] == 0)


@gpu
def test_isothermal_rest_is_preserved_bitwise_with_drag_hyperdiffusion_si_and_raw(hs_model):
    import cupy as cp
    from tropoi.temporal.semi_implicit import (SemiImplicitLeapfrogStepper,
                                               SemiImplicitOperator)
    from tropoi.temporal.tendencies.held_suarez import NewtonianRelaxation
    from tropoi.temporal.tendencies.primitive_equations import isothermal_rest_state
    model = hs_model
    K = model.nlev
    op = SemiImplicitOperator.from_model(model, t_ref=300.0)
    no_relax = HeldSuarezParameters(k_a=0.0, k_s=0.0)
    terms = (NewtonianRelaxation(model, no_relax),)
    dampers = (rayleigh_drag_damping(model.sigma.full_levels_array(), model.l_max + 1, HS),
               hyperdiffusion_damping(K, model.l_max, model.R, 0.1 * DAY_SECONDS, model.l_max))
    x0 = isothermal_rest_state(model.l_max, K, temperature=300.0, surface_pressure=1e5).coeffs
    st = SemiImplicitLeapfrogStepper(model.tendency, op, 1200.0, raw_nu=0.1, raw_alpha=0.53,
                                     explicit_terms=terms, dampers=dampers)
    st.initialize(x0)
    for _ in range(6):
        st.step()
    assert cp.asnumpy(st.state).tobytes() == cp.asnumpy(x0).tobytes()
    assert cp.asnumpy(st.x_prev).tobytes() == cp.asnumpy(x0).tobytes()


@gpu
def test_resting_uniform_equilibrium_profile_stays_at_rest_under_full_forcing(hs_model):
    """With dT_y = dtheta_z = 0, T_eq(p) = max(200, 315 (p/p0)^kappa) is
    horizontally uniform; T_k = T_eq(sigma_k p0) at rest with p_s = p0 is an
    equilibrium of the forced system (no pressure gradients, drag of zero
    wind, T = T_eq). It stays at rest to round-off under the full scheme."""
    import cupy as cp
    from tropoi.temporal.semi_implicit import (SemiImplicitLeapfrogStepper,
                                               SemiImplicitOperator)
    from tropoi.temporal.tendencies.held_suarez import NewtonianRelaxation
    from tropoi.temporal.tendencies.primitive_equations import isothermal_rest_state
    model = hs_model
    K = model.nlev
    flat = HeldSuarezParameters(delta_t_y=0.0, delta_theta_z=0.0)
    sigma = model.sigma.full_levels_array()
    t_prof = equilibrium_temperature(0.0, sigma * 1e5, flat)
    x0 = isothermal_rest_state(model.l_max, K, temperature=300.0, surface_pressure=1e5).coeffs
    x0[2 * K:3 * K, 0, 0] = cp.asarray(t_prof) * math.sqrt(4 * math.pi)
    op = SemiImplicitOperator.from_model(model, t_ref=300.0)
    st = SemiImplicitLeapfrogStepper(
        model.tendency, op, 1200.0, raw_nu=0.1, raw_alpha=0.53,
        explicit_terms=(NewtonianRelaxation(model, flat),),
        dampers=(rayleigh_drag_damping(sigma, model.l_max + 1, flat),
                 hyperdiffusion_damping(K, model.l_max, model.R, 0.1 * DAY_SECONDS,
                                        model.l_max)))
    st.initialize(x0)
    for _ in range(72):                                   # one day
        st.step()
    x = st.state
    wind = float(cp.abs(x[0:2 * K]).max())
    dT = float(cp.abs(x[2 * K:3 * K] - x0[2 * K:3 * K]).max()) / math.sqrt(4 * math.pi)
    dq = float(cp.abs(x[3 * K] - x0[3 * K]).max())
    print(f"\nafter 1 day: max|zeta, delta| = {wind:.2e} s^-1, max|dT| = {dT:.2e} K, "
          f"max|d ln p_s| = {dq:.2e}")
    assert wind < 1e-18
    assert dT < 1e-9
    assert dq < 1e-12


@gpu
def test_newtonian_hook_matches_an_independent_product_grid_calculation(hs_model):
    """The hook against a from-scratch evaluation: latitude structure in T,
    non-uniform ln p_s (degrees 2 and 4), the product grid's own Gauss
    latitudes, HS94 formulas written out here, one analysis, 2/3 cut."""
    import cupy as cp
    from tropoi.spatial.grids.latlon_grid import GaussLatLonGridGeometry
    from tropoi.temporal.tendencies.held_suarez import NewtonianRelaxation
    from tropoi.temporal.tendencies.primitive_equations import isothermal_rest_state
    model = hs_model
    K, n = model.nlev, model.l_max + 1
    x = isothermal_rest_state(model.l_max, K, temperature=280.0, surface_pressure=1e5).coeffs
    rng = np.random.default_rng(11)
    for k in range(K):                                     # T structure in l <= 6
        for l in range(1, 7):
            for m in range(l + 1):
                x[2 * K + k, l, m] = 3.0 * complex(rng.standard_normal(),
                                                   rng.standard_normal() if m else 0.0)
    x[3 * K, 2, 0] = 0.03
    x[3 * K, 4, 3] = 0.01 - 0.02j
    got = NewtonianRelaxation(model, HS)(x)
    # independent evaluation
    ps_space = model._ps
    geom = ps_space.geometry
    assert isinstance(geom, GaussLatLonGridGeometry)
    lat = np.repeat(np.pi / 2 - np.arccos(np.sort(np.polynomial.legendre.leggauss(geom.nlat)[0])[::-1]),
                    geom.nlon)
    assert np.allclose(lat, cp.asnumpy(geom.point_latitudes), atol=1e-14)
    sh = ps_space.sh
    T = np.stack([cp.asnumpy(sh.inv_transform(x[2 * K + k]).real) for k in range(K)])
    ps = np.exp(cp.asnumpy(sh.inv_transform(x[3 * K]).real))
    sig = (np.arange(K) + 0.5) / K
    teq = np.empty_like(T)
    kT = np.empty_like(T)
    for k in range(K):
        p = sig[k] * ps
        raw = (315.0 - 60.0 * np.sin(lat) ** 2 - 10.0 * np.log(p / 1e5) * np.cos(lat) ** 2) \
            * (p / 1e5) ** (2.0 / 7.0)
        teq[k] = np.maximum(200.0, raw)
        ramp = max(0.0, (sig[k] - 0.7) / 0.3)
        kT[k] = 1 / (40 * 86400.0) + (1 / (4 * 86400.0) - 1 / (40 * 86400.0)) * ramp * np.cos(lat) ** 4
    f = -kT * (T - teq)
    cut = model._trunc_cut
    ref = np.zeros((3 * K + 1, n, n), dtype=np.complex128)
    for k in range(K):
        c = cp.asnumpy(sh.transform(cp.asarray(f[k])))
        c[cut + 1:, :] = 0.0
        c[:, cut + 1:] = 0.0
        ref[2 * K + k] = c
    rel = float(np.abs(cp.asnumpy(got) - ref).max() / np.abs(ref).max())
    print(f"\nNewtonian hook vs independent product-grid evaluation: {rel:.2e}")
    assert rel < 1e-12


def test_build_physics_wiring_matches_the_protocol():
    """Default del^8: order 4, tau = 0.1 d at l = l_max; drag then del^8; the
    hyperdiffusion reference degree is validated."""
    from tropoi.run.held_suarez.config import development_config, production_config
    cfg = production_config()
    assert cfg.hyperdiffusion_degree == cfg.l_max == 42
    assert cfg.hyperdiffusion_order == 4 and cfg.hyperdiffusion_efold_days == 0.1
    assert cfg.r_dry == pytest.approx(286.857142857142857, rel=1e-15)
    hyper = hyperdiffusion_damping(cfg.nlev, cfg.l_max, cfg.radius, 0.1 * DAY_SECONDS,
                                   cfg.hyperdiffusion_degree)
    assert hyper.rates[0, 42] * 0.1 * DAY_SECONDS == pytest.approx(1.0, rel=1e-14)
    with pytest.raises(ValueError):
        hyperdiffusion_damping(10, 21, 6.371e6, 8640.0, reference_degree=22)
    with pytest.raises(ValueError):
        development_config().with_(hyperdiffusion_reference_degree=22)
