"""Semi-implicit leapfrog stepper for the dry primitive equations (S3).

Plan: docs/superpowers/plans/2026-09-27-held-suarez-24h-plan.md §3;
formulation: docs/held_suarez/SEMI_IMPLICIT.md.

CPU part (NumPy, runs in CI): the fast operator L assembled from the
discrete column operators, the reduced K×K solve against the unreduced
(2K+1)×(2K+1) coupled system, the analytic SI-leapfrog amplification
(Cayley transform, modulus 1, dispersion), RAW-off exactness of the pure
linear scheme, and serialization of both leapfrog time levels.

GPU part (CuPy): the same operator against the nonlinear PE tendency
(linearization residuals on a non-rotating and a rotating planet), exact
preservation of isothermal rest, the linear gravity-wave test on the real
model, and second-order convergence of SI against RK4 on T21 L10.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tropoi.spatial.sigma_coordinate import (
    SigmaGrid, column_mass_tendency, hydrostatic_geopotential, omega_over_p)
from tropoi.spatial.truncation import product_truncation_cut

R_DRY = 287.04
CP_DRY = 1004.64
RADIUS = 6.371e6
T_REF = 300.0
P0 = 1.0e5
MONOPOLE = math.sqrt(4.0 * math.pi)


def _has_cuda():
    try:
        import cupy as cp
        return cp.is_available()
    except Exception:
        return False


gpu = pytest.mark.skipif(not _has_cuda(), reason="CUDA/CuPy not available")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _random_state(K, l_max, seed, *, xp=np, scale=(1e-5, 1e-5, 5.0, 0.05),
                  cut=None):
    """Random (3K+1, n, n) stack in the (l, m) layout, m <= l, m = 0 real.

    ``cut`` band-limits every field at that degree. The nonlinear tendency
    is dealiased at the product cut: content above it is not reachable from
    band-limited initial data (T and q have no tendency there, delta only
    -lap(G T) of a zero T', zeta only truncated products) and the weak-form
    curl/div analyses are exact only inside it, so the linearization tests
    perturb inside the cut (measured: ln p_s content above the cut alone
    produces a spurious vorticity tendency of 1e-10 vs 3e-24 inside).
    """
    rng = np.random.default_rng(seed)
    n = l_max + 1
    y = rng.standard_normal((3 * K + 1, n, n)) + 1j * rng.standard_normal((3 * K + 1, n, n))
    y = np.tril(y)
    y[:, :, 0] = y[:, :, 0].real
    if cut is not None:
        y[:, cut + 1:, :] = 0.0
        y[:, :, cut + 1:] = 0.0
    y[0:K] *= scale[0]
    y[K:2 * K] *= scale[1]
    y[2 * K:3 * K] *= scale[2]
    y[3 * K] *= scale[3]
    return xp.asarray(y)


def _rest(K, l_max, *, xp=np, temperature=T_REF, ps=P0):
    n = l_max + 1
    y = np.zeros((3 * K + 1, n, n), dtype=np.complex128)
    y[2 * K:3 * K, 0, 0] = temperature * MONOPOLE
    y[3 * K, 0, 0] = math.log(ps) * MONOPOLE
    return xp.asarray(y)


def _fast_terms(op, y):
    """The fast linear terms of plan §3 built directly from the column
    operators (NOT from the assembled matrices): the reference for L."""
    xp = np if isinstance(y, np.ndarray) else __import__("cupy")
    K = op.K
    delta = y[K:2 * K]
    T = y[2 * K:3 * K]
    q = y[3 * K]
    cl = xp.asarray(op.c_l)[None, :, None]
    mask = xp.asarray(op.mask, dtype=float)
    phi, _ = hydrostatic_geopotential(op.sigma, T, 0.0, op.r_dry)
    zeros = delta * 0.0
    wp = omega_over_p(op.sigma, delta, zeros)
    out = xp.zeros_like(y)
    # delta_dot = -lap(G T + R T_ref q) = c_l (G T + R T_ref q)
    out[K:2 * K] = cl * phi + (cl * mask[None, :, None]) * (op.r_dry * op.t_ref) * q[None]
    # T_dot = -tau delta = kappa T_ref (omega/p)^lin(delta)
    out[2 * K:3 * K] = mask[None, :, None] * (op.kappa * op.t_ref) * wp
    # q_dot = -nu . delta
    out[3 * K] = mask[:, None] * column_mass_tendency(op.sigma, delta)
    return out


def _rel(a, b):
    xp = np if isinstance(a, np.ndarray) else __import__("cupy")
    return float(xp.abs(a - b).max() / xp.abs(b).max())


def _make_operator(K=6, l_max=21):
    from tropoi.temporal.semi_implicit import SemiImplicitOperator
    return SemiImplicitOperator(SigmaGrid.uniform(K), l_max=l_max, radius=RADIUS,
                                r_dry=R_DRY, cp_dry=CP_DRY, t_ref=T_REF)


def _make_model(l_max=21, nlev=6, *, rotating=True, batched=True):
    from tropoi.spatial.environment import PlanetaryParameters
    from tropoi.spatial.planet import Planet
    from tropoi.temporal.tendencies.primitive_equations import PrimitiveEquationsModel
    day_hours = 24.0 if rotating else math.inf
    planet = Planet.generate(
        params=PlanetaryParameters.from_earth_like(day_hours=day_hours),
        grid_type="latlon", nlat=32, nlon=64, l_max=l_max, grid_resolution=3,
        product_quadrature="fine")
    return PrimitiveEquationsModel(planet, SigmaGrid.uniform(nlev),
                                   r_dry=R_DRY, cp_dry=CP_DRY,
                                   batched_transforms=batched)


# ---------------------------------------------------------------------------
# 1. fast operator (CPU)
# ---------------------------------------------------------------------------

def test_operator_matches_fast_terms_from_column_operators():
    op = _make_operator()
    y = _random_state(op.K, op.l_max, seed=0)
    got = op.apply(y)
    ref = _fast_terms(op, y)
    assert got.shape == y.shape
    assert np.all(got[0:op.K] == 0.0)            # zeta is not in L
    assert _rel(got, ref) < 1e-13


def test_operator_mask_is_the_product_truncation_cut():
    op = _make_operator(K=4, l_max=21)
    cut = product_truncation_cut(21)
    assert cut == 14
    assert op.fast_cut == cut
    assert list(op.mask) == [1] * (cut + 1) + [0] * (21 - cut)
    # Above the cut only the exact -lap(G T) coupling survives (as in the
    # discrete tendency): T and q have no fast tendency there.
    y = _random_state(op.K, op.l_max, seed=1)
    Ly = op.apply(y)
    assert np.all(Ly[2 * op.K:, cut + 1:, :] == 0.0)
    assert np.any(Ly[op.K:2 * op.K, cut + 1:, :] != 0.0)


def test_operator_dense_matrix_reproduces_apply_on_unit_vectors():
    op = _make_operator(K=5, l_max=10)
    K, n = op.K, op.l_max + 1
    for l in (0, 1, 3, 6, 7, 10):
        A = op.dense_matrix(l)
        assert A.shape == (2 * K + 1, 2 * K + 1)
        for j in range(2 * K + 1):
            y = np.zeros((3 * K + 1, n, n), dtype=np.complex128)
            y[K + j, l, 0] = 1.0
            Ly = op.apply(y)[K:, l, 0]
            assert np.allclose(Ly, A[:, j], rtol=0, atol=1e-14 * max(1.0, abs(A).max()))
    # l = 0 has no fast pressure-gradient term; q and T still respond to delta.
    A0 = op.dense_matrix(0)
    assert np.all(A0[0:K, :] == 0.0)


def test_operator_gravity_wave_speeds_are_real_positive_and_hydrostatic_scale():
    op = _make_operator(K=10, l_max=21)
    c = op.gravity_wave_speeds()
    assert c.shape == (op.K,)
    assert np.all(np.isfinite(c)) and np.all(c > 0)
    # The external (Lamb) mode of an isothermal hydrostatic column travels
    # at sqrt(gamma R T_ref); the discrete column must not exceed it by more
    # than a few percent and the slowest internal mode is far below it.
    lamb = math.sqrt(CP_DRY / (CP_DRY - R_DRY) * R_DRY * T_REF)
    assert 0.9 * math.sqrt(R_DRY * T_REF) < c.max() < 1.05 * lamb
    assert c.min() < 0.3 * c.max()


def test_operator_from_model_matches_direct_construction_cpu_arrays():
    """from_model only reads sigma/constants; check on a CPU stub model."""
    from tropoi.temporal.semi_implicit import SemiImplicitOperator

    class _Stub:
        sigma = SigmaGrid.uniform(7)
        l_max = 15
        R = RADIUS
        r_dry = R_DRY
        cp_dry = CP_DRY

    op = SemiImplicitOperator.from_model(_Stub(), t_ref=T_REF)
    ref = SemiImplicitOperator(SigmaGrid.uniform(7), l_max=15, radius=RADIUS,
                               r_dry=R_DRY, cp_dry=CP_DRY, t_ref=T_REF)
    assert np.array_equal(op.B, ref.B) and np.array_equal(op.c_l, ref.c_l)
    assert op.fast_cut == ref.fast_cut == product_truncation_cut(15)


# ---------------------------------------------------------------------------
# 2. reduced K×K solve against the unreduced (2K+1)×(2K+1) coupled system
# ---------------------------------------------------------------------------

def _unreduced_step(op, dt, x_prev, explicit):
    """Direct dense solve of (I - dt L_l) X^{n+1} = (I + dt L_l) X^{n-1} + 2 dt N
    per degree, in the (delta, T, q) block; zeta explicit.

    The raw block matrix mixes entries of order dt c_l G ~ 1e-6 and dt tau ~
    1e4, so it is diagonally balanced first (x = D x~, D chosen so the
    delta<->T and delta<->q couplings have equal magnitude); otherwise the
    dense reference itself is only good to ~1e-11.
    """
    K, n = op.K, op.l_max + 1
    out = np.empty_like(x_prev)
    out[0:K] = x_prev[0:K] + 2.0 * dt * explicit[0:K]
    I = np.eye(2 * K + 1)
    for l in range(n):
        A = op.dense_matrix(l)
        D = np.ones(2 * K + 1)
        a_t = np.abs(A[0:K, K:2 * K]).max()
        a_q = np.abs(A[0:K, 2 * K]).max()
        b_t = np.abs(A[K:2 * K, 0:K]).max()
        b_q = np.abs(A[2 * K, 0:K]).max()
        if a_t > 0 and b_t > 0:
            D[K:2 * K] = math.sqrt(b_t / a_t)
        elif b_t > 0:                       # l = 0: c_0 = 0, keep pivots on I
            D[K:2 * K] = dt * b_t
        if a_q > 0 and b_q > 0:
            D[2 * K] = math.sqrt(b_q / a_q)
        elif b_q > 0:
            D[2 * K] = dt * b_q
        Dinv = 1.0 / D
        As = (Dinv[:, None] * A) * D[None, :]
        lhs = I - dt * As
        rhs = Dinv[:, None] * ((I + dt * A) @ x_prev[K:, l, :] + 2.0 * dt * explicit[K:, l, :])
        out[K:, l, :] = D[:, None] * np.linalg.solve(lhs, rhs)
    return out


@pytest.mark.parametrize("dt", [300.0, 1200.0, 3600.0])
def test_reduced_solve_matches_unreduced_coupled_system(dt):
    from tropoi.temporal.semi_implicit import SemiImplicitSolver
    op = _make_operator(K=8, l_max=21)
    solver = SemiImplicitSolver(op, dt)
    x_prev = _random_state(op.K, op.l_max, seed=2) + _rest(op.K, op.l_max)
    explicit = _random_state(op.K, op.l_max, seed=3,
                             scale=(1e-10, 1e-10, 1e-5, 1e-8))
    got = solver.advance(x_prev, explicit)
    ref = _unreduced_step(op, dt, x_prev, explicit)
    K = op.K
    blocks = ((slice(0, K), "zeta"), (slice(K, 2 * K), "delta"),
              (slice(2 * K, 3 * K), "T"), (slice(3 * K, 3 * K + 1), "q"))
    # (a) the reduced solution satisfies the unreduced equations per degree:
    #     (I - dt L_l) X^{n+1} - (I + dt L_l) X^{n-1} - 2 dt N = 0, to 1e-12
    #     of the row scale (the plan's "exact per degree" criterion)
    I = np.eye(2 * K + 1)
    worst = 0.0
    for l in range(op.l_max + 1):
        A = op.dense_matrix(l)
        rhs = (I + dt * A) @ x_prev[K:, l, :] + 2.0 * dt * explicit[K:, l, :]
        r = (I - dt * A) @ got[K:, l, :] - rhs
        scale = np.maximum(np.abs(got[K:, l, :]), np.abs(rhs)).max(axis=1)
        worst = max(worst, float((np.abs(r).max(axis=1) / scale).max()))
    print(f"dt={dt:g}: reduced solution residual in the unreduced system {worst:.2e}")
    assert worst <= 1e-12
    # (b) and agrees with the direct dense solve of that system
    for rows, name in blocks:
        num = np.abs(got[rows] - ref[rows]).max()
        assert num <= 1e-12 * np.abs(ref[rows]).max(), (name, num)
    assert np.array_equal(got[0:K], x_prev[0:K] + 2.0 * dt * explicit[0:K])


def test_solver_matrices_are_assembled_from_operators_not_hand_derived():
    from tropoi.temporal.semi_implicit import SemiImplicitSolver
    op = _make_operator(K=6, l_max=12)
    dt = 900.0
    solver = SemiImplicitSolver(op, dt)
    ones = np.ones((op.K, 1))
    B = op.G @ op.tau + op.r_dry * op.t_ref * (ones @ op.nu[None, :])
    assert np.allclose(B, op.B, rtol=0, atol=1e-12 * abs(B).max())
    for l in (0, 5, 8, 12):
        M = np.eye(op.K) + dt * dt * op.c_l[l] * op.mask[l] * B
        assert np.allclose(solver.matrix(l), M, rtol=0, atol=1e-12 * abs(M).max())
        assert np.allclose(solver.matrix(l) @ solver.inverse(l), np.eye(op.K),
                           rtol=0, atol=1e-12)


# ---------------------------------------------------------------------------
# 3. analytic SI-leapfrog amplification: Cayley transform (CPU)
# ---------------------------------------------------------------------------

def test_cayley_amplification_has_modulus_one_and_si_dispersion():
    from tropoi.temporal.semi_implicit import SemiImplicitSolver
    op = _make_operator(K=6, l_max=21)
    dt = 1800.0
    solver = SemiImplicitSolver(op, dt)
    for l in (1, 4, 10, 14):
        C = solver.cayley_matrix(l)
        ev = np.linalg.eigvals(C)
        assert np.allclose(np.abs(ev), 1.0, rtol=0, atol=1e-12)
        # dispersion: e^{±2i atan(ω dt)} for ω = sqrt(c_l λ_k(B)), plus one
        # stationary mode (eigenvalue exactly 1)
        omega = op.frequencies(l)
        expected = np.concatenate([np.exp(2j * np.arctan(omega * dt)),
                                   np.exp(-2j * np.arctan(omega * dt)), [1.0]])
        assert np.allclose(np.sort_complex(ev), np.sort_complex(expected),
                           rtol=0, atol=1e-10)
        # relative to the exact e^{±2iω dt}: the scheme is second order in ω dt
        assert np.all(np.abs(2 * np.arctan(omega * dt) - 2 * omega * dt)
                      <= (2 * omega * dt) ** 3 / 3 + 1e-15)
    # above the cut nothing oscillates: T, q frozen, delta only sees G T
    C = solver.cayley_matrix(op.fast_cut + 1)
    assert np.allclose(np.abs(np.linalg.eigvals(C)), 1.0, atol=1e-12)


def test_stepper_on_pure_linear_tendency_is_exactly_the_cayley_transform():
    from tropoi.temporal.semi_implicit import (SemiImplicitLeapfrogStepper,
                                               SemiImplicitSolver)
    op = _make_operator(K=5, l_max=12)
    dt = 1200.0
    st = SemiImplicitLeapfrogStepper(op.apply, op, dt, raw_nu=0.0)
    y0 = _rest(op.K, op.l_max) + _random_state(op.K, op.l_max, seed=4,
                                               scale=(1e-6, 1e-6, 0.5, 1e-3))
    st.initialize(y0, t0=0.0)
    st.step()                        # RK4 startup -> X^1
    xs = [y0.copy(), st.state.copy()]
    for _ in range(4):
        st.step()
        xs.append(st.state.copy())
    solver = SemiImplicitSolver(op, dt)
    K = op.K
    for n in range(1, len(xs) - 1):
        for l in range(op.l_max + 1):
            C = solver.cayley_matrix(l)
            pred = C @ xs[n - 1][K:, l, :]
            assert np.allclose(xs[n + 1][K:, l, :], pred, rtol=0,
                               atol=1e-13 * max(1.0, np.abs(pred).max()))
        assert np.array_equal(xs[n + 1][0:K], xs[n - 1][0:K])  # zeta untouched
    assert st.t == pytest.approx(5 * dt) and st.step_count == 5


# ---------------------------------------------------------------------------
# 4. protocol, startup, RAW filter, serialization (CPU)
# ---------------------------------------------------------------------------

def test_stepper_satisfies_timestepper_protocol_and_startup_substeps():
    from tropoi.temporal.steppers import TimeStepper
    from tropoi.temporal.semi_implicit import SemiImplicitLeapfrogStepper
    op = _make_operator(K=6, l_max=21)
    st = SemiImplicitLeapfrogStepper(op.apply, op, 1200.0)
    assert isinstance(st, TimeStepper)
    with pytest.raises(RuntimeError):
        st.step()
    # RK4 on the imaginary axis is stable for ω dt_sub <= 2 sqrt 2: the
    # substep count is the smallest integer satisfying the stability
    # polynomial for every discrete gravity-wave frequency.
    omega_max = max(op.frequencies(l).max() for l in range(op.l_max + 1))
    for dt in (600.0, 1200.0, 3600.0, 7200.0):
        n_sub = SemiImplicitLeapfrogStepper(op.apply, op, dt).startup_substeps
        assert n_sub == math.ceil(omega_max * dt / (2.0 * math.sqrt(2.0)) - 1e-12) or \
            n_sub == max(1, math.ceil(omega_max * dt / (2.0 * math.sqrt(2.0))))
        assert omega_max * dt / n_sub <= 2.0 * math.sqrt(2.0) + 1e-12
    assert SemiImplicitLeapfrogStepper(op.apply, op, 600.0, startup_substeps=7).startup_substeps == 7


def test_raw_filter_follows_plan_recurrence():
    """d = ν (X^{n-1}_f - 2 X^n + X^{n+1}); X^n_f = X^n + α d; X^{n+1} -= (1-α) d."""
    from tropoi.temporal.semi_implicit import SemiImplicitLeapfrogStepper
    op = _make_operator(K=3, l_max=6)
    dt = 900.0
    nu, alpha = 0.1, 0.53
    tendency = op.apply
    a = SemiImplicitLeapfrogStepper(tendency, op, dt, raw_nu=nu, raw_alpha=alpha)
    b = SemiImplicitLeapfrogStepper(tendency, op, dt, raw_nu=0.0)
    y0 = _rest(op.K, op.l_max) + _random_state(op.K, op.l_max, seed=5,
                                               scale=(1e-6, 1e-6, 0.5, 1e-3))
    a.initialize(y0); b.initialize(y0)
    a.step(); b.step()                          # identical startup
    assert np.array_equal(a.state, b.state) and np.array_equal(a.x_prev, b.x_prev)
    x_prev_f, x_n = a.x_prev.copy(), a.state.copy()
    b.step()                                    # unfiltered X^{n+1}
    x_np1 = b.state.copy()
    a.step()
    d = nu * (x_prev_f - 2.0 * x_n + x_np1)
    assert np.allclose(a.x_prev, x_n + alpha * d, rtol=0, atol=1e-15 * np.abs(x_n).max())
    assert np.allclose(a.state, x_np1 - (1.0 - alpha) * d, rtol=0,
                       atol=1e-15 * np.abs(x_np1).max())


def test_state_dict_serializes_both_time_levels_and_resumes_bit_identically():
    from tropoi.temporal.semi_implicit import SemiImplicitLeapfrogStepper
    op = _make_operator(K=4, l_max=9)
    dt = 1500.0
    y0 = _rest(op.K, op.l_max) + _random_state(op.K, op.l_max, seed=6,
                                               scale=(1e-6, 1e-6, 0.5, 1e-3))
    a = SemiImplicitLeapfrogStepper(op.apply, op, dt, raw_nu=0.1, raw_alpha=0.53)
    a.initialize(y0, t0=10.0)
    for _ in range(3):
        a.step()
    saved = a.state_dict()
    assert saved["scheme"] == "si_leapfrog"
    assert saved["x_prev"] is not None and saved["x_curr"] is not None
    assert saved["x_prev"].shape == saved["x_curr"].shape == y0.shape
    assert saved["x_prev"] is not a.x_prev and saved["x_curr"] is not a.state  # copies
    assert saved["step"] == 3 and saved["t"] == 10.0 + 3 * dt and saved["dt"] == dt
    b = SemiImplicitLeapfrogStepper(op.apply, op, dt, raw_nu=0.1, raw_alpha=0.53)
    b.load_state_dict(saved)
    for _ in range(4):
        a.step(); b.step()
    assert a.state.tobytes() == b.state.tobytes()
    assert a.x_prev.tobytes() == b.x_prev.tobytes()
    assert a.t == b.t and a.step_count == b.step_count
    # a resumed stepper must never redo the RK4 startup: dropping x_prev is refused
    broken = dict(saved); broken["x_prev"] = None
    with pytest.raises(ValueError):
        b.load_state_dict(broken)
    # and the scheme / dt / filter parameters must match
    with pytest.raises(ValueError):
        SemiImplicitLeapfrogStepper(op.apply, op, dt / 2, raw_nu=0.1).load_state_dict(saved)
    with pytest.raises(ValueError):
        SemiImplicitLeapfrogStepper(op.apply, op, dt, raw_nu=0.2).load_state_dict(saved)
    with pytest.raises(ValueError):
        from tropoi.temporal.steppers import RK4Stepper
        RK4Stepper(op.apply, dt).load_state_dict(saved)
    # ... and so must every parameter that determines L (reviewer finding):
    # a different radius, gas constant, cut, vertical grid or explicit
    # startup substep count is refused even when t_ref agrees
    from tropoi.temporal.semi_implicit import SemiImplicitOperator
    assert saved["operator"] == op.signature()
    for other in (
        SemiImplicitOperator(SigmaGrid.uniform(4), l_max=9, radius=RADIUS * 1.01,
                             r_dry=R_DRY, cp_dry=CP_DRY, t_ref=T_REF),
        SemiImplicitOperator(SigmaGrid.uniform(4), l_max=9, radius=RADIUS,
                             r_dry=R_DRY, cp_dry=CP_DRY, t_ref=T_REF, fast_cut=5),
        SemiImplicitOperator(SigmaGrid.uniform(4), l_max=9, radius=RADIUS,
                             r_dry=R_DRY - 1.0, cp_dry=CP_DRY, t_ref=T_REF),
        SemiImplicitOperator(SigmaGrid((0.0, 0.1, 0.3, 0.6, 1.0)), l_max=9,
                             radius=RADIUS, r_dry=R_DRY, cp_dry=CP_DRY, t_ref=T_REF),
    ):
        with pytest.raises(ValueError):
            SemiImplicitLeapfrogStepper(other.apply, other, dt, raw_nu=0.1,
                                        raw_alpha=0.53).load_state_dict(saved)
    with pytest.raises(ValueError):
        SemiImplicitLeapfrogStepper(op.apply, op, dt, raw_nu=0.1, raw_alpha=0.53,
                                    startup_substeps=3).load_state_dict(saved)


def test_state_dict_before_first_step_holds_only_x0():
    from tropoi.temporal.semi_implicit import SemiImplicitLeapfrogStepper
    op = _make_operator(K=3, l_max=6)
    st = SemiImplicitLeapfrogStepper(op.apply, op, 600.0)
    y0 = _rest(op.K, op.l_max)
    st.initialize(y0, t0=0.0)
    d = st.state_dict()
    assert d["x_prev"] is None and d["step"] == 0
    other = SemiImplicitLeapfrogStepper(op.apply, op, 600.0)
    other.load_state_dict(d)
    st.step(); other.step()
    assert st.state.tobytes() == other.state.tobytes()


# ---------------------------------------------------------------------------
# 5. GPU: the operator against the nonlinear tendency
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def nonrotating_model():
    return _make_model(l_max=21, nlev=6, rotating=False)


@pytest.fixture(scope="module")
def rotating_model():
    return _make_model(l_max=21, nlev=6, rotating=True)


def _linearization_residual(model, op, eps, seed=7):
    import cupy as cp
    K = model.nlev
    x0 = _rest(K, model.l_max, xp=cp)
    y = _random_state(K, model.l_max, seed=seed, xp=cp,
                      scale=(1e-5, 1e-5, 5.0, 0.05),
                      cut=product_truncation_cut(model.l_max))
    f0 = model.tendency(x0)
    f1 = model.tendency(x0 + eps * y)
    return f1 - f0 - eps * op.apply(y), y


BLOCKS = ("zeta", "delta", "T", "q")


def _block_norms(a, K):
    """max-abs per prognostic block (the blocks carry different units)."""
    import cupy as cp
    rows = (slice(0, K), slice(K, 2 * K), slice(2 * K, 3 * K), slice(3 * K, 3 * K + 1))
    return {name: float(cp.abs(a[r]).max()) for name, r in zip(BLOCKS, rows)}


@gpu
def test_operator_apply_on_device_matches_fast_terms(nonrotating_model):
    import cupy as cp
    from tropoi.temporal.semi_implicit import SemiImplicitOperator
    op = SemiImplicitOperator.from_model(nonrotating_model, t_ref=T_REF)
    y = _random_state(op.K, op.l_max, seed=8, xp=cp)
    got = op.apply(y)
    assert isinstance(got, cp.ndarray)
    assert _rel(got, _fast_terms(op, y)) < 1e-13
    assert _rel(cp.asnumpy(got), op.apply(cp.asnumpy(y))) < 1e-14


@gpu
def test_linearization_residual_is_second_order_on_nonrotating_planet(nonrotating_model):
    """(ii) Ω = 0: every linear term about isothermal rest is in L, so
    tendency(X_rest + εY) - tendency(X_rest) - εLY = O(ε²) in every block."""
    import cupy as cp
    from tropoi.temporal.semi_implicit import SemiImplicitOperator
    model = nonrotating_model
    assert model.Omega == 0.0
    op = SemiImplicitOperator.from_model(model, t_ref=T_REF)
    K = model.nlev
    res, lin = {}, {}
    for eps in (1e-2, 5e-3):
        r, y = _linearization_residual(model, op, eps)
        res[eps] = _block_norms(r, K)
        lin[eps] = _block_norms(eps * op.apply(y), K)
    for b in BLOCKS:
        ratio = res[1e-2][b] / res[5e-3][b]
        print(f"nonrotating {b:5s}: residual {res[1e-2][b]:.3e} / {res[5e-3][b]:.3e} "
              f"(ratio {ratio:.3f}); linear term {lin[1e-2][b]:.3e}")
        assert 3.6 < ratio < 4.4, b
        if b != "zeta":                       # zeta has no fast linear term
            assert res[1e-2][b] < 5e-2 * lin[1e-2][b], b   # measured ~1e-2


def _coriolis_linearization(model, y):
    """Linear Coriolis coupling of the discrete tendency: the f part of the
    SWE-style pointwise expansions of div(eta V) and k.curl(eta V), analyzed
    and truncated exactly as the tendency does."""
    import cupy as cp
    K = model.nlev
    ps = model._ps
    sh_p = ps.sh
    coslat = ps.coslat
    f_g = sh_p.inv_transform(model.f_lm).real
    f_lam, f_snt = model._deriv_fields(sh_p, model.f_lm)
    out = cp.zeros_like(y)
    for k in range(K):
        psi = model._inv_laplacian(y[k])
        chi = model._inv_laplacian(y[K + k])
        u, v = model._wind_from(sh_p, coslat, psi, chi)
        zeta_g = sh_p.inv_transform(y[k]).real
        delta_g = sh_p.inv_transform(y[K + k]).real
        div_fv = (u * f_lam - v * f_snt) / coslat + f_g * delta_g
        curl_fv = f_g * zeta_g + (v * f_lam + u * f_snt) / coslat
        out[k] = model._truncate(sh_p.transform(-div_fv))
        out[K + k] = model._truncate(sh_p.transform(curl_fv))
    out[0:2 * K, 0, :] = 0.0
    return out


@gpu
def test_linearization_residual_on_rotating_planet_is_the_coriolis_coupling(rotating_model):
    """(iii) Ω ≠ 0: in zeta/delta the residual is O(ε) and equals the
    Coriolis linearization to O(ε²); T and q stay O(ε²)."""
    import cupy as cp
    from tropoi.temporal.semi_implicit import SemiImplicitOperator
    model = rotating_model
    op = SemiImplicitOperator.from_model(model, t_ref=T_REF)
    K = model.nlev
    res, cor, corn = {}, {}, {}
    for eps in (1e-2, 5e-3):
        r, y = _linearization_residual(model, op, eps)
        c = eps * _coriolis_linearization(model, y)
        res[eps] = _block_norms(r, K)
        cor[eps] = _block_norms(r - c, K)
        corn[eps] = _block_norms(c, K)
    for b in ("zeta", "delta"):
        r_ratio = res[1e-2][b] / res[5e-3][b]
        c_ratio = cor[1e-2][b] / cor[5e-3][b]
        print(f"rotating {b:5s}: residual {res[1e-2][b]:.3e} (ratio {r_ratio:.3f}); "
              f"minus Coriolis {cor[1e-2][b]:.3e} (ratio {c_ratio:.3f}); "
              f"Coriolis term {corn[1e-2][b]:.3e}")
        assert 1.9 < r_ratio < 2.1, b
        assert 3.6 < c_ratio < 4.4, b
        assert cor[1e-2][b] < 5e-2 * corn[1e-2][b], b     # measured ~1.4e-2
    for b in ("T", "q"):
        ratio = res[1e-2][b] / res[5e-3][b]
        print(f"rotating {b:5s}: residual {res[1e-2][b]:.3e} (ratio {ratio:.3f})")
        assert 3.6 < ratio < 4.4, b
        assert corn[1e-2][b] == 0.0


# ---------------------------------------------------------------------------
# 6. GPU: rest preservation, gravity waves, convergence
# ---------------------------------------------------------------------------

@gpu
def test_isothermal_rest_is_preserved_bitwise(rotating_model):
    import cupy as cp
    from tropoi.temporal.semi_implicit import (SemiImplicitLeapfrogStepper,
                                               SemiImplicitOperator)
    from tropoi.temporal.tendencies.primitive_equations import isothermal_rest_state
    model = rotating_model
    op = SemiImplicitOperator.from_model(model, t_ref=T_REF)
    x0 = isothermal_rest_state(model.l_max, model.nlev, temperature=T_REF,
                               surface_pressure=P0).coeffs
    st = SemiImplicitLeapfrogStepper(model.tendency, op, 1200.0,
                                     raw_nu=0.1, raw_alpha=0.53)
    st.initialize(x0)
    for _ in range(4):
        st.step()
    assert cp.asnumpy(st.state).tobytes() == cp.asnumpy(x0).tobytes()
    assert cp.asnumpy(st.x_prev).tobytes() == cp.asnumpy(x0).tobytes()
    # a different T_ref does not break the exact rest either (the operator
    # sees delta = 0 and c_0 = 0 only)
    op2 = SemiImplicitOperator.from_model(model, t_ref=250.0)
    st2 = SemiImplicitLeapfrogStepper(model.tendency, op2, 1200.0)
    st2.initialize(x0)
    st2.step(); st2.step()
    assert cp.asnumpy(st2.state).tobytes() == cp.asnumpy(x0).tobytes()


@gpu
def test_linear_gravity_waves_follow_the_si_amplification_factor(nonrotating_model):
    """Tiny delta perturbation of isothermal rest, Ω = 0, RAW off: per degree
    X^{n+1} = C_l X^{n-1} with the analytic Cayley factor (modulus 1)."""
    import cupy as cp
    from tropoi.temporal.semi_implicit import (SemiImplicitLeapfrogStepper,
                                               SemiImplicitOperator, SemiImplicitSolver)
    model = nonrotating_model
    K = model.nlev
    op = SemiImplicitOperator.from_model(model, t_ref=T_REF)
    dt = 1800.0
    x0 = _rest(K, model.l_max, xp=cp)
    rng = np.random.default_rng(9)
    pert = np.zeros_like(cp.asnumpy(x0))
    for l, m in ((1, 0), (2, 1), (5, 3), (9, 0), (13, 7)):
        pert[K:2 * K, l, m] = 1e-10 * (rng.standard_normal(K) + (1j * rng.standard_normal(K) if m else 0))
    x0 = x0 + cp.asarray(pert)
    st = SemiImplicitLeapfrogStepper(model.tendency, op, dt, raw_nu=0.0)
    st.initialize(x0)
    xs = [cp.asnumpy(x0)]
    for _ in range(8):
        st.step()
        xs.append(cp.asnumpy(st.state))
    solver = SemiImplicitSolver(op, dt)
    base = cp.asnumpy(_rest(K, model.l_max, xp=cp))
    worst = 0.0
    for n in range(1, len(xs) - 1):
        for l in (1, 2, 5, 9, 13):
            C = solver.cayley_matrix(l)
            prev = (xs[n - 1] - base)[K:, l, :]
            pred = C @ prev
            got = (xs[n + 1] - base)[K:, l, :]
            worst = max(worst, np.abs(got - pred).max() / np.abs(pred).max())
    print(f"gravity-wave Cayley residual (relative, nonlinear O(eps)): {worst:.3e}")
    assert worst < 1e-4
    # modulus 1 on the real model: project each degree onto the eigenvectors
    # of C_l; every modal amplitude must be constant along the even and the
    # odd leapfrog chains (X^{n+1} = C_l X^{n-1}) up to the nonlinear floor.
    # (A Euclidean norm over the (delta, T, q) rows is meaningless here: the
    # rows carry different units and energy converts between them.)
    drift = 0.0
    for l in (1, 2, 5, 9, 13):
        C = solver.cayley_matrix(l)
        ev, V = np.linalg.eig(C)
        Vinv = np.linalg.inv(V)
        amps = [np.abs(Vinv @ (xs[n] - base)[K:, l, :]) for n in range(len(xs))]
        for chain in (range(0, len(xs), 2), range(1, len(xs), 2)):
            chain = list(chain)
            a0 = amps[chain[0]]
            floor = 1e-3 * np.abs(a0).max()          # ignore unexcited modes
            for n in chain[1:]:
                sel = a0 > floor
                drift = max(drift, float(np.abs(amps[n][sel] / a0[sel] - 1.0).max()))
    print(f"gravity-wave modal amplitude drift over 8 steps: {drift:.3e}")
    assert drift < 1e-4
    # no linear vorticity source with Ω = 0: what appears is the quadratic
    # sigma_dot dV/dsigma curl (measured 4.5e-16 s^-1 from a 1e-10 s^-1
    # divergence perturbation over 8 steps, i.e. 5e-6 relative)
    zeta = [(xs[n] - base)[0:K] for n in range(len(xs))]
    zeta_max = max(float(np.abs(z).max()) for z in zeta)
    print(f"vorticity generated (Omega = 0): {zeta_max:.3e} s^-1")
    assert zeta_max < 1e-5 * 1e-10


@gpu
def test_si_converges_second_order_to_rk4_on_t21_l10():
    """2-day T21 L10 thermal wave: SI at Δt, Δt/2, Δt/4 against an RK4
    reference; consecutive error ratios ≈ 4."""
    import time
    import cupy as cp
    from tropoi.spatial.initialization.pe import make_pe_ic
    from tropoi.temporal.steppers import RK4Stepper
    from tropoi.temporal.semi_implicit import (SemiImplicitLeapfrogStepper,
                                               SemiImplicitOperator)
    model = _make_model(l_max=21, nlev=10, rotating=True)
    op = SemiImplicitOperator.from_model(model, t_ref=T_REF)
    x0 = make_pe_ic("thermal_wave", model, temperature=T_REF,
                    surface_pressure=P0, thermal_amplitude=5.0).coeffs
    t_end = 2 * 86400.0

    def run_si(dt):
        st = SemiImplicitLeapfrogStepper(model.tendency, op, dt, raw_nu=0.0)
        st.initialize(x0)
        n = int(round(t_end / dt))
        for _ in range(n):
            st.step()
        assert st.t == pytest.approx(t_end)
        return cp.asnumpy(st.state)

    t0 = time.perf_counter()
    dt_ref = 300.0
    rk = RK4Stepper(model.tendency, dt_ref)
    rk.initialize(x0)
    for _ in range(int(round(t_end / dt_ref))):
        rk.step()
    ref = cp.asnumpy(rk.state)
    t_ref_s = time.perf_counter() - t0

    errs = {}
    for dt in (2400.0, 1200.0, 600.0):
        t1 = time.perf_counter()
        sol = run_si(dt)
        errs[dt] = float(np.linalg.norm(sol - ref) / np.linalg.norm(ref - cp.asnumpy(x0)))
        print(f"SI dt={dt:6.0f} s: relative error {errs[dt]:.3e} ({time.perf_counter() - t1:.1f} s)")
    r1 = errs[2400.0] / errs[1200.0]
    r2 = errs[1200.0] / errs[600.0]
    print(f"RK4 reference dt={dt_ref} s took {t_ref_s:.1f} s; error ratios {r1:.3f}, {r2:.3f}")
    # ratios sit slightly above 4 because the SI phase error 2 atan(w dt) is
    # not yet asymptotic for the fastest excited modes at dt = 2400 s
    # (measured 4.35, 4.30); the error at dt = 600 s was 5.7e-3 of the 2-day change
    assert 3.4 < r1 < 4.6 and 3.6 < r2 < 4.4
    assert errs[600.0] < 1e-2
