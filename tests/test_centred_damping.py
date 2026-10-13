"""Opt-in centred (Crank–Nicolson) treatment of the diagonal damping inside
the semi-implicit solve (``damping_scheme="centred"``; DEVLOG 2026-09-28).

The default (``"lagged"``: the backward-Euler factor 1/(1 + 2 dt r) on
X^{n+1} after the solve) is unchanged bit for bit. With the option the
damping -r X enters the implicit part trapezoidally, i.e. L is replaced by
L - R in the reduced per-degree system:

    [ I + dt R_delta + dt^2 c_l m_l ( G D_T tau + R T_ref d_q 1 nu^T ) ] delta_bar = rhs_l,
    D_T = (I + dt R_T)^{-1},  d_q = 1 / (1 + dt r_q),

and for rows outside L (zeta) it is the per-row factor
X^{n+1} = X^{n-1} + 2 dt (N - r X^{n-1}) / (1 + dt r).

CPU: assembly from the operators, the reduced solve against a direct dense
solve of the unreduced (2K+1)x(2K+1) system, the scalar recurrence tied to
the stepper, second order restored on the scalar harness, |lambda| <= 1 at
900 s, state-dict signature, config field. GPU: bitwise rest, T21 L10
convergence (second order with RAW off; RAW alpha = 0.53 quantified).
"""
from __future__ import annotations

import functools
import math

import numpy as np
import pytest

from tropoi.spatial.sigma_coordinate import SigmaGrid
from tropoi.temporal.hooks import DiagonalDamping
from tropoi.temporal.semi_implicit import (SemiImplicitLeapfrogStepper,
                                           SemiImplicitOperator, SemiImplicitSolver)
from tropoi.temporal.tendencies.held_suarez import (DAY_SECONDS, HeldSuarezParameters,
                                                    hyperdiffusion_coefficient,
                                                    hyperdiffusion_damping,
                                                    rayleigh_drag_damping)

from test_held_suarez_scheme import (BLOCKS, CONV_DTS, CONV_REF_DT, NU, ALPHA, _blocks,  # noqa: F401
                                     hs, jet_wind_bound)  # noqa: F401

HS = HeldSuarezParameters()
RADIUS, R_DRY, CP_DRY, T_REF = 6.371e6, 287.0, 1004.0, 300.0
DT_PROD = 900.0


def _has_cuda():
    try:
        import cupy as cp
        return cp.is_available()
    except Exception:
        return False


gpu = pytest.mark.skipif(not _has_cuda(), reason="CUDA/CuPy not available")


def _operator(K=8, l_max=21):
    return SemiImplicitOperator(SigmaGrid.uniform(K), l_max=l_max, radius=RADIUS,
                                r_dry=R_DRY, cp_dry=CP_DRY, t_ref=T_REF)


def _hs_dampers(op, efold_degree=None):
    sigma = op.sigma.full_levels_array()
    return (rayleigh_drag_damping(sigma, op.n, HS),
            hyperdiffusion_damping(op.K, op.l_max, op.radius, 0.1 * DAY_SECONDS,
                                   efold_degree or op.l_max))


def _random_state(K, l_max, seed, scale=(1e-5, 1e-5, 1.0, 1e-2)):
    rng = np.random.default_rng(seed)
    n = l_max + 1
    x = np.zeros((3 * K + 1, n, n), dtype=complex)
    for rows, s in zip((slice(0, K), slice(K, 2 * K), slice(2 * K, 3 * K), slice(3 * K, 3 * K + 1)),
                       scale):
        x[rows] = s * (rng.standard_normal((x[rows].shape)) + 1j * rng.standard_normal(x[rows].shape))
    x[..., 0] = x[..., 0].real
    return np.triu(x.transpose(0, 2, 1)).transpose(0, 2, 1)     # m <= l only


def _rest(K, l_max):
    n = l_max + 1
    x = np.zeros((3 * K + 1, n, n), dtype=complex)
    x[2 * K:3 * K, 0, 0] = T_REF * math.sqrt(4 * math.pi)
    x[3 * K, 0, 0] = math.log(1e5) * math.sqrt(4 * math.pi)
    return x


# ---------------------------------------------------------------------------
# 1. assembly and the reduced solve against the unreduced damped system
# ---------------------------------------------------------------------------

def test_centred_solver_matrix_is_assembled_from_operators_and_rates():
    op = _operator(K=6, l_max=12)
    dt = DT_PROD
    dampers = _hs_dampers(op)
    solver = SemiImplicitSolver(op, dt, dampers=dampers)
    rates = sum(d.rates for d in dampers)
    K = op.K
    ones = np.ones((K, 1))
    for l in (0, 5, 8, 12):
        r_d, r_t, r_q = rates[K:2 * K, l], rates[2 * K:3 * K, l], rates[3 * K, l]
        D_T = np.diag(1.0 / (1.0 + dt * r_t))
        d_q = 1.0 / (1.0 + dt * r_q)
        B_l = op.G @ D_T @ op.tau + op.r_dry * op.t_ref * d_q * (ones @ op.nu[None, :])
        M = np.eye(K) + dt * np.diag(r_d) + dt * dt * op.c_l[l] * op.mask[l] * B_l
        assert np.allclose(solver.matrix(l), M, rtol=0, atol=1e-12 * abs(M).max())
        assert np.allclose(solver.matrix(l) @ solver.inverse(l), np.eye(K), rtol=0, atol=1e-12)
    # no dampers: exactly the S3 matrices
    plain = SemiImplicitSolver(op, dt)
    zero = SemiImplicitSolver(op, dt, dampers=())
    for l in range(op.n):
        assert np.array_equal(plain.matrix(l), zero.matrix(l))


def _unreduced_damped_step(op, rates, dt, x_prev, explicit):
    """Direct, diagonally balanced dense solve of
    (I - dt A_l) X^{n+1} = (I + dt A_l) X^{n-1} + 2 dt N, A_l = L_l - R_l,
    per degree in the (delta, T, q) block; zeta by the per-row factor."""
    K, n = op.K, op.n
    out = np.empty_like(x_prev)
    r_z = rates[0:K][:, :, None]
    out[0:K] = x_prev[0:K] + 2.0 * dt * (explicit[0:K] - r_z * x_prev[0:K]) / (1.0 + dt * r_z)
    I = np.eye(2 * K + 1)
    for l in range(n):
        A = op.dense_matrix(l) - np.diag(rates[K:, l])
        D = np.ones(2 * K + 1)
        a_t = np.abs(A[0:K, K:2 * K]).max()
        a_q = np.abs(A[0:K, 2 * K]).max()
        b_t = np.abs(A[K:2 * K, 0:K]).max()
        b_q = np.abs(A[2 * K, 0:K]).max()
        if a_t > 0 and b_t > 0:
            D[K:2 * K] = math.sqrt(b_t / a_t)
        elif b_t > 0:
            D[K:2 * K] = dt * b_t
        if a_q > 0 and b_q > 0:
            D[2 * K] = math.sqrt(b_q / a_q)
        elif b_q > 0:
            D[2 * K] = dt * b_q
        Dinv = 1.0 / D
        As = (Dinv[:, None] * A) * D[None, :]
        rhs = Dinv[:, None] * ((I + dt * A) @ x_prev[K:, l, :] + 2.0 * dt * explicit[K:, l, :])
        out[K:, l, :] = D[:, None] * np.linalg.solve(I - dt * As, rhs)
    return out


@pytest.mark.parametrize("dt", [300.0, 1200.0, 3600.0])
def test_centred_reduced_solve_matches_unreduced_damped_system(dt):
    op = _operator(K=8, l_max=21)
    dampers = _hs_dampers(op)
    rates = sum(d.rates for d in dampers)
    solver = SemiImplicitSolver(op, dt, dampers=dampers)
    x_prev = _random_state(op.K, op.l_max, seed=2) + _rest(op.K, op.l_max)
    explicit = _random_state(op.K, op.l_max, seed=3, scale=(1e-10, 1e-10, 1e-5, 1e-8))
    got = solver.advance(x_prev, explicit)
    ref = _unreduced_damped_step(op, rates, dt, x_prev, explicit)
    K = op.K
    I = np.eye(2 * K + 1)
    worst = 0.0
    for l in range(op.n):
        A = op.dense_matrix(l) - np.diag(rates[K:, l])
        rhs = (I + dt * A) @ x_prev[K:, l, :] + 2.0 * dt * explicit[K:, l, :]
        r = (I - dt * A) @ got[K:, l, :] - rhs
        scale = np.maximum(np.abs(got[K:, l, :]), np.abs(rhs)).max(axis=1)
        worst = max(worst, float((np.abs(r).max(axis=1) / scale).max()))
    print(f"dt={dt:g}: centred reduced solution residual in the unreduced damped system {worst:.2e}")
    assert worst <= 1e-12
    for rows, name in ((slice(0, K), "zeta"), (slice(K, 2 * K), "delta"),
                       (slice(2 * K, 3 * K), "T"), (slice(3 * K, 3 * K + 1), "q")):
        num = np.abs(got[rows] - ref[rows]).max()
        assert num <= 1e-12 * np.abs(ref[rows]).max(), (name, num)


def test_centred_cayley_matrix_is_contractive_and_reduces_to_s3_without_damping():
    op = _operator(K=6, l_max=21)
    dt = DT_PROD
    solver = SemiImplicitSolver(op, dt, dampers=_hs_dampers(op))
    plain = SemiImplicitSolver(op, dt)
    for l in (1, 4, 10, 14, 21):
        ev = np.sort(np.abs(np.linalg.eigvals(solver.cayley_matrix(l))))
        assert ev.max() <= 1.0 + 1e-12
        if l <= op.fast_cut:
            # every coupled mode damped; at low l the (T, q) balance mode is
            # damped only through T's tiny del^8 rate (1 - 7e-11 at l = 1)
            assert ev.max() < 1.0
            if l >= 10:
                assert ev.max() < 1.0 - 1e-6
        else:
            # above the cut ln p_s is frozen and never diffused: exactly one
            # eigenvalue 1; the damped zeta-free modes follow (1-a)/(1+a)
            assert ev[-1] == pytest.approx(1.0, abs=1e-12) and ev[-2] < 1.0 - 1e-6
        assert np.allclose(np.abs(np.linalg.eigvals(plain.cayley_matrix(l))), 1.0, atol=1e-12)


# ---------------------------------------------------------------------------
# 2. the stepper: scalar recurrence, second order, stability, signature
# ---------------------------------------------------------------------------

def amplification(dt, k=0.0, w_explicit=0.0, w_si=0.0, r=0.0, nu=NU, alpha=ALPHA,
                  centred=False):
    """2x2 map of test_held_suarez_scheme.amplification with, for
    ``centred``, the damping -r x taken trapezoidally over n-1, n+1."""
    F = 1.0 / (1.0 + 2.0 * dt * r)

    def step(p, c):
        if centred:
            xt = ((p * (1 + 1j * w_si * dt - dt * r) + 2 * dt * (-k + 1j * w_explicit) * c)
                  / (1 - 1j * w_si * dt + dt * r))
            n1 = xt
        else:
            xt = (p * (1 + 1j * w_si * dt) + 2 * dt * (-k + 1j * w_explicit) * c) / (1 - 1j * w_si * dt)
            n1 = F * xt
        d = nu * (p - 2 * c + n1)
        return c + alpha * d, n1 - (1 - alpha) * d

    a, b = step(1.0, 0.0), step(0.0, 1.0)
    return np.array([[a[0], b[0]], [a[1], b[1]]], dtype=complex)


def max_gain(*args, **kw) -> float:
    return float(np.abs(np.linalg.eigvals(amplification(*args, **kw))).max())


def _scalar_stepper(op, dt, term, r, **kw):
    rates = np.zeros((3 * op.K + 1, op.n))
    rates[0] = r
    return SemiImplicitLeapfrogStepper(lambda x: np.zeros_like(x), op, dt, explicit_terms=(term,),
                                       dampers=(DiagonalDamping("r", rates),), **kw)


def test_centred_option_is_the_centred_recurrence_on_a_scalar_mode():
    op = _operator(K=2, l_max=3)
    dt, k, w, r = 1200.0, HS.k_s, 2e-4, HS.k_f

    class Term:
        max_rate = k

        def __call__(self, x):
            out = np.zeros_like(x)
            out[0] = (-k + 1j * w) * x[0]
            return out

        def signature(self):
            return {"kind": "scalar"}

    st = _scalar_stepper(op, dt, Term(), r, raw_nu=NU, raw_alpha=ALPHA, damping_scheme="centred")
    y0 = np.zeros((3 * op.K + 1, op.n, op.n), dtype=complex)
    y0[0, 2, 1] = 1.0 + 0.5j
    st.initialize(y0)
    st.step()
    v = np.array([st.x_prev[0, 2, 1], st.state[0, 2, 1]])
    M = amplification(dt, k=k, w_explicit=w, r=r, centred=True)
    M_lag = amplification(dt, k=k, w_explicit=w, r=r, centred=False)
    assert not np.allclose(M, M_lag)
    for _ in range(20):
        st.step()
        v = M @ v
        assert np.allclose([st.x_prev[0, 2, 1], st.state[0, 2, 1]], v, rtol=1e-13, atol=0)


def test_centred_damping_restores_second_order_on_the_scalar_harness():
    """dx/dt = -r x + sin(W t) on the real stepper (RAW off, 64 startup
    substeps, 2 days): lagged r = k_f -> ratios ~2; centred r = k_f -> 4."""
    op = _operator(K=2, l_max=3)
    W, T = 2 * math.pi / (1.3 * DAY_SECONDS), 2 * DAY_SECONDS

    class Forcing:
        max_rate = 0.0
        t = 0.0

        def __call__(self, x):
            out = np.zeros_like(x)
            out[0, 2, 1] = math.sin(W * self.t)
            return out

        def signature(self):
            return {"kind": "sin"}

    def exact(r):
        return (r * math.sin(W * T) - W * math.cos(W * T) + W * math.exp(-r * T)) / (r * r + W * W)

    out = {}
    for scheme in ("lagged", "centred"):
        errs = []
        for dt in (1200.0, 600.0, 300.0, 150.0):
            f = Forcing()
            st = _scalar_stepper(op, dt, f, HS.k_f, raw_nu=0.0, startup_substeps=64,
                                 damping_scheme=scheme)
            st.initialize(np.zeros((3 * op.K + 1, op.n, op.n), dtype=complex))
            for _ in range(int(round(T / dt))):
                f.t = st.t
                st.step()
            errs.append(abs(st.state[0, 2, 1].real - exact(HS.k_f)) / abs(exact(HS.k_f)))
        out[scheme] = (errs, [errs[i] / errs[i + 1] for i in range(3)])
        print(f"\n{scheme}: errors " + " ".join(f"{e:.3e}" for e in errs) + "; ratios "
              + " ".join(f"{x:.4f}" for x in out[scheme][1]))
    assert all(1.8 < x < 2.1 for x in out["lagged"][1])
    assert all(3.9 < x < 4.1 for x in out["centred"][1])
    assert out["centred"][0][0] < 0.2 * out["lagged"][0][0]       # at 1200 s already 8x smaller


def test_centred_recurrence_is_stable_at_the_production_dt():
    """|lambda| <= 1 over k in [0, k_s], r in {0, k_f, del^8 at l = 1..42},
    omega_SI, with RAW(0.1, 0.53) and with RAW off (the centred damping alone
    damps both leapfrog modes: lambda^2 = (1 - dt r)/(1 + dt r))."""
    a = RADIUS
    k8 = hyperdiffusion_coefficient(a, 0.1 * DAY_SECONDS, 42)
    rs = [0.0, HS.k_f] + [k8 * (l * (l + 1) / a ** 2) ** 4 for l in (1, 14, 28, 42)]
    rs += [HS.k_f + rs[-1]]
    for dt in (DT_PROD, 1200.0):
        worst = 0.0
        for k in np.linspace(0.0, HS.k_s, 11):
            for r in rs:
                for w_si in (0.0, 1e-4, 1e-3, 1e-2):
                    worst = max(worst, max_gain(dt, k=k, r=r, w_si=w_si, centred=True))
        pure = [max_gain(dt, r=r, nu=0.0, centred=True) for r in rs]
        expect = [math.sqrt(abs(1 - dt * r) / (1 + dt * r)) for r in rs]
        print(f"\ncentred, dt={dt:.0f}: max |lambda| over the grid {worst:.12f}; pure damping "
              + " ".join(f"{g:.6f}" for g in pure))
        assert worst <= 1.0 + 1e-12
        assert np.allclose(pure, expect, rtol=1e-12)
        assert dt * max(rs) < 1.0                                  # no sign alternation


def test_centred_raw_explicit_oscillation_bound():
    """The jet bound of test_held_suarez_scheme (jet_wind_bound) with centred
    damping, retained T42: equal to the lagged bound within 0.1 % at 600-1200
    s (125.0 vs 125.1 m/s at 900 s), because the first degree to amplify
    (19-23) has dt r(del^8) < 1e-3, where the placement does not matter.
    Growth stays slow (RAW-limited) until the leapfrog limit at l ~ 42; the
    wind at which it becomes fast (gain > 1.01 per step, e-folding < 1 d)
    is where the placement matters: lagged 147.5 vs centred 132.8 m/s at
    900 s, 108.6 vs 95.4 m/s at 1200 s. Printed for the DEVLOG and STATUS.

    Corrected 2026-09-28: with the former frequency U l/(a cos 45 deg) and
    k = k_s the binding modes sat at l ~ 40-42 and w dt ~ 1, where the lagged
    factor extends the leapfrog limit to sqrt(1 + 2 dt r) and the centred one
    shortens it to sqrt(1 - (dt r)^2); that gave centred 94.9 vs lagged
    105.3 m/s at 900 s, an artefact of the overstated frequency."""
    k8 = hyperdiffusion_coefficient(RADIUS, 0.1 * DAY_SECONDS, 42)
    res = {(dt, c): jet_wind_bound(dt, cut=42, gain=functools.partial(max_gain, centred=c),
                                   a=RADIUS)
           for dt in (600.0, 720.0, DT_PROD, 1200.0) for c in (False, True)}
    print("\nretained T42 (cut 42) non-amplifying peak wind: "
          + ", ".join(f"dt={dt:.0f} {'centred' if c else 'lagged'} {u:.1f} m/s (l = {l})"
                      for (dt, c), (u, l) in res.items()))
    for dt in (600.0, 720.0, DT_PROD, 1200.0):
        (lag, _), (cen, l) = res[(dt, False)], res[(dt, True)]
        assert abs(cen - lag) <= 1e-3 * lag
        assert dt * k8 * (l * (l + 1) / RADIUS ** 2) ** 4 < 1e-3
    assert res[(DT_PROD, True)][0] > 120.0 > 94.5
    fast = {(dt, c): jet_wind_bound(dt, cut=42, gain=functools.partial(max_gain, centred=c),
                                    a=RADIUS, gain_limit=1.01)[0]
            for dt in (DT_PROD, 1200.0) for c in (False, True)}
    print("fast growth (gain > 1.01/step) above: "
          + ", ".join(f"dt={dt:.0f} {'centred' if c else 'lagged'} {u:.1f} m/s"
                      for (dt, c), u in fast.items()))
    assert fast[(DT_PROD, False)] > fast[(DT_PROD, True)] > 1.3 * 94.5
    assert fast[(1200.0, False)] > 1.1 * 94.5 > fast[(1200.0, True)]


def test_damping_scheme_enters_the_state_dict_and_default_is_lagged():
    op = _operator(K=2, l_max=3)
    dampers = _hs_dampers(op)
    zero = lambda x: np.zeros_like(x)  # noqa: E731
    st = SemiImplicitLeapfrogStepper(zero, op, DT_PROD, dampers=dampers)
    assert st.damping_scheme == "lagged"
    assert st.state_dict()["damping_scheme"] == "lagged"
    ct = SemiImplicitLeapfrogStepper(zero, op, DT_PROD, dampers=dampers, damping_scheme="centred")
    assert ct.state_dict()["damping_scheme"] == "centred"
    assert ct.startup_substeps == st.startup_substeps          # startup rule unchanged
    x0 = _random_state(op.K, op.l_max, seed=5) + _rest(op.K, op.l_max)
    for s in (st, ct):
        s.initialize(x0)
        s.step()
        s.step()
    assert not np.array_equal(st.state, ct.state)
    with pytest.raises(ValueError, match="damping_scheme"):
        ct.load_state_dict(st.state_dict())
    with pytest.raises(ValueError, match="damping_scheme"):
        st.load_state_dict(ct.state_dict())
    d = ct.state_dict()
    ct2 = SemiImplicitLeapfrogStepper(zero, op, DT_PROD, dampers=dampers, damping_scheme="centred")
    ct2.load_state_dict(d)
    ct.step()
    ct2.step()
    assert ct.state.tobytes() == ct2.state.tobytes()
    # a dict without the key comes from the lagged code and loads as lagged
    d0 = st.state_dict()
    del d0["damping_scheme"]
    st.load_state_dict(d0)
    with pytest.raises(ValueError, match="damping_scheme"):
        SemiImplicitLeapfrogStepper(zero, op, DT_PROD, damping_scheme="exponential")


def test_lagged_default_is_bitwise_the_sequential_factor_path():
    """The default applies each damper's 1/(1 + 2 dt r) to X^{n+1} after the
    S3 solve, in order: reproduced here from the plain solver."""
    op = _operator(K=3, l_max=5)
    dampers = _hs_dampers(op)
    dt = DT_PROD
    x_prev = _random_state(op.K, op.l_max, seed=7) + _rest(op.K, op.l_max)
    x_curr = _random_state(op.K, op.l_max, seed=8) + _rest(op.K, op.l_max)
    tend = lambda x: 1e-3 * x  # noqa: E731
    st = SemiImplicitLeapfrogStepper(tend, op, dt, dampers=dampers, startup_substeps=1)
    st.initialize(x_curr)
    st.load_state_dict({**st.state_dict(), "step": 1, "x_prev": x_prev, "x_curr": x_curr,
                        "t": dt})
    st.step()
    plain = SemiImplicitSolver(op, dt)
    x_next = plain.advance(x_prev, tend(x_curr) - op.apply(x_curr))
    for d in dampers:
        x_next = d.apply_implicit(x_next, 2.0 * dt)
    dd = NU * (x_prev - 2.0 * x_curr + x_next)
    assert st.state.tobytes() == (x_next - (1.0 - ALPHA) * dd).tobytes()


def test_config_has_damping_scheme_field_defaulting_to_lagged():
    from tropoi.run.held_suarez.config import HeldSuarezConfig, production_config
    a = production_config()
    assert a.damping_scheme == "lagged"
    assert a.to_dict()["damping_scheme"] == "lagged"
    c = a.with_(damping_scheme="centred")
    assert c.sha256() != a.sha256()
    assert HeldSuarezConfig.from_dict(c.to_dict()).damping_scheme == "centred"
    with pytest.raises(ValueError, match="damping_scheme"):
        a.with_(damping_scheme="exp")
    # a config document written before the field existed still loads
    d = a.to_dict()
    del d["damping_scheme"]
    assert HeldSuarezConfig.from_dict(d).damping_scheme == "lagged"
    print(f"\nproduction config sha256 (lagged) {a.sha256()}\n  centred {c.sha256()}")


# ---------------------------------------------------------------------------
# 3. GPU: rest, T21 L10 convergence
# ---------------------------------------------------------------------------

@gpu
def test_centred_isothermal_rest_is_preserved_bitwise(hs):
    import cupy as cp
    from tropoi.spatial.states.primitive_equations import isothermal_rest_state
    from tropoi.temporal.tendencies.held_suarez import NewtonianRelaxation
    cfg, model, terms, dampers, op, _ = hs
    no_relax = (NewtonianRelaxation(model, HeldSuarezParameters(k_a=0.0, k_s=0.0)),)
    x0 = isothermal_rest_state(model.l_max, cfg.nlev, temperature=300.0,
                               surface_pressure=1e5).coeffs
    st = SemiImplicitLeapfrogStepper(model.tendency, op, DT_PROD, explicit_terms=no_relax,
                                     dampers=dampers, damping_scheme="centred")
    st.initialize(x0)
    for _ in range(6):
        st.step()
    assert cp.asnumpy(st.state).tobytes() == cp.asnumpy(x0).tobytes()
    assert cp.asnumpy(st.x_prev).tobytes() == cp.asnumpy(x0).tobytes()


def _convergence_centred(hs, raw_nu, raw_alpha, ref):
    import time
    cfg, model, terms, dampers, op, x0 = hs
    B = _blocks(cfg.nlev)
    import cupy as cp
    errs = {}
    for dt in CONV_DTS:
        t1 = time.perf_counter()
        st = SemiImplicitLeapfrogStepper(model.tendency, op, dt, raw_nu=raw_nu, raw_alpha=raw_alpha,
                                         explicit_terms=terms, dampers=dampers,
                                         damping_scheme="centred")
        st.initialize(x0)
        for _ in range(int(round(2 * DAY_SECONDS / dt))):
            st.step()
        errs[dt] = {b: float(cp.linalg.norm(st.state[s] - ref[s]) / cp.linalg.norm(ref[s] - x0[s]))
                    for b, s in B.items()}
        print(f"  centred, RAW nu={raw_nu} alpha={raw_alpha}, SI dt = {dt:5.1f} s "
              f"({time.perf_counter() - t1:.0f} s): "
              + " ".join(f"{b}={v:.3e}" for b, v in errs[dt].items()))
    ratios = {}
    for b in BLOCKS:
        e = [errs[d][b] for d in CONV_DTS]
        ratios[b] = [e[i] / e[i + 1] for i in range(len(e) - 1)]
        print(f"  {b:5s} ratios " + " ".join(f"{x:.3f}" for x in ratios[b]))
    return errs, ratios


@gpu
def test_centred_complete_scheme_converges_second_order_on_t21_l10(hs):
    """2 days, T21 L10, forcing + drag + del^8 + SI with centred damping,
    against RK4 of the same right-hand side (150 s). RAW off: second order in
    every block (S3's criterion). RAW(0.1, 0.53): the remaining first-order
    term is RAW's; recorded (ratios decrease with dt) and its size at 300 s
    printed against the RAW-off error."""
    import time
    from tropoi.temporal.hooks import complete_tendency
    from tropoi.temporal.steppers import RK4Stepper
    cfg, model, terms, dampers, op, x0 = hs
    t0 = time.perf_counter()
    rk = RK4Stepper(complete_tendency(model.tendency, terms, dampers), CONV_REF_DT)
    rk.initialize(x0)
    for _ in range(int(round(2 * DAY_SECONDS / CONV_REF_DT))):
        rk.step()
    ref = rk.state
    print(f"\nRK4 reference dt = {CONV_REF_DT} s ({time.perf_counter() - t0:.0f} s)")
    _, r_off = _convergence_centred(hs, 0.0, ALPHA, ref)
    e_on, r_on = _convergence_centred(hs, NU, ALPHA, ref)
    for b in BLOCKS:
        r1, r2 = r_off[b]
        assert 3.4 < r1 < 4.6 and 3.6 < r2 < 4.4, (b, r_off[b])
        assert all(x > 1.8 for x in r_on[b]) and all(x < 4.6 for x in r_on[b]), (b, r_on[b])
