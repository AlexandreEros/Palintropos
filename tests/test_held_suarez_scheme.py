"""Complete-scheme verification (plan S4b): forcing, Rayleigh drag, del^8,
semi-implicit leapfrog and the RAW filter together.

(a) CPU: the scalar recurrence of leapfrog + explicit relaxation + implicit
    backward-Euler damping + RAW, built here independently and tied to the
    stepper; |lambda| <= 1 at the selected dt, k_s, nu, alpha, and the limit
    the RAW filter puts on explicit oscillation frequencies.
(b) GPU: 2-day T21 L10 convergence of SI against RK4 with identical forcing
    and diffusion.
(c) GPU: 5-day T21 L10 stability at dt >= 900 s with the computational mode
    monitored.
(d) GPU: the RK4 startup of the complete right-hand side at the SI dt.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tropoi.temporal.tendencies.held_suarez import (DAY_SECONDS, HeldSuarezParameters,
                                                    hyperdiffusion_coefficient)

HS = HeldSuarezParameters()
NU, ALPHA = 0.1, 0.53
DT_SELECTED = 900.0            # DEVLOG 2026-09-27 S4b
DT_CANDIDATE = 1200.0


def _has_cuda():
    try:
        import cupy as cp
        return cp.is_available()
    except Exception:
        return False


gpu = pytest.mark.skipif(not _has_cuda(), reason="CUDA/CuPy not available")


# ---------------------------------------------------------------------------
# (a) scalar recurrence
# ---------------------------------------------------------------------------

def amplification(dt, k=0.0, w_explicit=0.0, w_si=0.0, r=0.0, nu=NU, alpha=ALPHA):
    """2x2 map (X^{n-1}_f, X^n) -> (X^n_f, X^{n+1}) for dx/dt = (-k + i w_e) x
    [explicit, at n] + i w_si x [trapezoidal over n-1, n+1] - r x [backward
    Euler factor 1/(1 + 2 dt r) on X^{n+1}], then RAW:
    d = nu (X^{n-1}_f - 2 X^n + X^{n+1}), X^n_f = X^n + alpha d,
    X^{n+1} -= (1 - alpha) d."""
    F = 1.0 / (1.0 + 2.0 * dt * r)

    def step(p, c):
        xt = (p * (1 + 1j * w_si * dt) + 2 * dt * (-k + 1j * w_explicit) * c) / (1 - 1j * w_si * dt)
        n1 = F * xt
        d = nu * (p - 2 * c + n1)
        return c + alpha * d, n1 - (1 - alpha) * d

    a, b = step(1.0, 0.0), step(0.0, 1.0)
    return np.array([[a[0], b[0]], [a[1], b[1]]], dtype=complex)


def max_gain(*args, **kw) -> float:
    return float(np.abs(np.linalg.eigvals(amplification(*args, **kw))).max())


def test_recurrence_is_the_stepper_on_a_scalar_mode():
    """Tie the analysis to the code: a vorticity row (not in L) under an
    explicit term (-k + i w) and a damper evolves exactly by the 2x2 map."""
    from tropoi.spatial.sigma_coordinate import SigmaGrid
    from tropoi.temporal.hooks import DiagonalDamping
    from tropoi.temporal.semi_implicit import SemiImplicitLeapfrogStepper, SemiImplicitOperator
    op = SemiImplicitOperator(SigmaGrid.uniform(2), l_max=3, radius=6.371e6, r_dry=287.0,
                              cp_dry=1004.0, t_ref=300.0)
    dt, k, w, r = 1200.0, HS.k_s, 2e-4, HS.k_f

    class Term:
        max_rate = k

        def __call__(self, x):
            out = np.zeros_like(x)
            out[0] = (-k + 1j * w) * x[0]
            return out

        def signature(self):
            return {"kind": "scalar"}

    rates = np.zeros((3 * op.K + 1, op.n))
    rates[0] = r
    st = SemiImplicitLeapfrogStepper(lambda x: np.zeros_like(x), op, dt, raw_nu=NU,
                                     raw_alpha=ALPHA, explicit_terms=(Term(),),
                                     dampers=(DiagonalDamping("r", rates),))
    y0 = np.zeros((3 * op.K + 1, op.n, op.n), dtype=complex)
    y0[0, 2, 1] = 1.0 + 0.5j
    st.initialize(y0)
    st.step()
    v = np.array([st.x_prev[0, 2, 1], st.state[0, 2, 1]])
    M = amplification(dt, k=k, w_explicit=w, r=r)
    for _ in range(20):
        st.step()
        v = M @ v
        assert np.allclose([st.x_prev[0, 2, 1], st.state[0, 2, 1]], v, rtol=1e-13, atol=0)


def test_leapfrog_relaxation_raw_recurrence_is_stable_at_selected_dt():
    """(a) |lambda| <= 1 for dt k in [0, dt k_s] with nu = 0.1, alpha = 0.53,
    with and without the drag and del^8 factors; without RAW the leapfrog
    computational mode of explicit relaxation grows (|lambda| = dt k +
    sqrt(1 + (dt k)^2) > 1), which the filter removes."""
    a = 6.371e6
    for dt in (DT_SELECTED, DT_CANDIDATE):
        k8 = hyperdiffusion_coefficient(a, 0.1 * DAY_SECONDS, 42)
        rs = [0.0, HS.k_f] + [k8 * (l * (l + 1) / a ** 2) ** 4 for l in (1, 14, 28, 42)]
        worst = 0.0
        for k in np.linspace(0.0, HS.k_s, 11):
            for r in rs:
                for w_si in (0.0, 1e-4, 1e-3, 1e-2):                  # SI gravity waves
                    worst = max(worst, max_gain(dt, k=k, r=r, w_si=w_si))
        ev = np.abs(np.linalg.eigvals(amplification(dt, k=HS.k_s)))
        ev0 = np.abs(np.linalg.eigvals(amplification(dt, k=HS.k_s, nu=0.0)))
        print(f"\ndt={dt:.0f} s, dt k_s={dt * HS.k_s:.5f}: |lambda| with RAW = "
              f"{ev.max():.8f}, {ev.min():.8f} (physical, computational); "
              f"without RAW = {ev0.max():.8f}; exp(-k_s dt) = {math.exp(-HS.k_s * dt):.8f}; "
              f"max |lambda| over k, r, w_si grid = {worst:.12f}")
        assert worst <= 1.0 + 1e-12
        assert ev0.max() > 1.0 + dt * HS.k_s * 0.9                   # unfiltered: grows
        assert ev.min() < 0.85                                       # RAW damps it
        assert ev.max() == pytest.approx(math.exp(-HS.k_s * dt), rel=1e-5)


def _threshold(dt, nu=NU, alpha=ALPHA):
    lo, hi = 0.05, 0.99
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if max_gain(dt, w_explicit=mid / dt, nu=nu, alpha=alpha) <= 1.0 + 1e-14:
            lo = mid
        else:
            hi = mid
    return lo


def test_raw_limits_explicit_oscillations_and_the_selected_dt_covers_the_jet():
    """Explicit oscillations (advection, Coriolis) under RAW(0.1, 0.53): the
    physical mode is non-amplifying only for w dt <= 0.437; beyond, it grows
    slowly. With the del^8 damping per degree (tau = 0.1 d at l = 42) the
    largest zonal wind at 45 deg for which every degree l <= 28 (the T42
    cut) is non-amplifying is ~80 m/s at dt = 1200 s and ~110 m/s at
    dt = 900 s. The Dinosaur T42 reference reaches 94.5 m/s (max|u| days
    200-1200; median 74.5 m/s), so dt = 900 s is selected."""
    thr = _threshold(1200.0)
    print(f"\nRAW(0.1, 0.53) explicit-oscillation limit: w dt <= {thr:.4f}; gain at w dt = 0.5: "
          f"{max_gain(1200.0, w_explicit=0.5 / 1200.0):.6f}")
    assert 0.43 < thr < 0.445
    a = 6.371e6
    k8 = hyperdiffusion_coefficient(a, 0.1 * DAY_SECONDS, 42)
    f0 = 2 * 7.292e-5

    def u_safe(dt, cut=28, coslat=math.cos(math.radians(45.0))):
        def ok(U):
            return all(max_gain(dt, k=HS.k_s, w_explicit=U * l / (a * coslat) + f0,
                                r=k8 * (l * (l + 1) / a ** 2) ** 4) <= 1 + 1e-13
                       for l in range(1, cut + 1))
        lo, hi = 1.0, 400.0
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if ok(mid) else (lo, mid)
        return lo

    u9, u12 = u_safe(900.0), u_safe(1200.0)
    print(f"largest non-amplifying jet wind at 45 deg, T42 (cut 28): dt=900 s {u9:.1f} m/s, "
          f"dt=1200 s {u12:.1f} m/s (Dinosaur reference max|u| 94.5 m/s)")
    assert u9 > 100.0 > 94.5 > u12


def test_damping_applied_to_the_new_level_is_first_order_in_time():
    """Why the complete scheme converges at first order (S4b (b)): the plan's
    damping multiplies X^{n+1} by 1/(1 + 2 dt r) after the SI solve, i.e. the
    damping term is evaluated at n+1 instead of centred at n, a local error
    -r dt dX/dt. The real stepper on dx/dt = -r x + sin(W t) (a zeta row: no L
    coupling, RAW off, 64 startup substeps, 2 days, W = 2 pi / 1.3 d) against
    the exact solution: with r = k_f the error ratios tend to 2; with r = 0
    the same code is exactly second order (ratios 4.000). An exponential
    factor exp(-2 dt r) has the same leading lag term (measured on the full
    model: identical errors to 3 digits)."""
    from tropoi.spatial.sigma_coordinate import SigmaGrid
    from tropoi.temporal.hooks import DiagonalDamping
    from tropoi.temporal.semi_implicit import SemiImplicitLeapfrogStepper, SemiImplicitOperator
    op = SemiImplicitOperator(SigmaGrid.uniform(2), l_max=3, radius=6.371e6, r_dry=287.0,
                              cp_dry=1004.0, t_ref=300.0)
    W = 2 * math.pi / (1.3 * DAY_SECONDS)
    T = 2 * DAY_SECONDS

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
        if r == 0.0:
            return (1.0 - math.cos(W * T)) / W
        return (r * math.sin(W * T) - W * math.cos(W * T) + W * math.exp(-r * T)) / (r * r + W * W)

    ratios = {}
    for r in (HS.k_f, 0.0):
        errs = []
        for dt in (1200.0, 600.0, 300.0, 150.0):
            f = Forcing()
            rates = np.zeros((3 * op.K + 1, op.n))
            rates[0] = r
            st = SemiImplicitLeapfrogStepper(lambda x: np.zeros_like(x), op, dt, raw_nu=0.0,
                                             explicit_terms=(f,), startup_substeps=64,
                                             dampers=(DiagonalDamping("r", rates),))
            st.initialize(np.zeros((3 * op.K + 1, op.n, op.n), dtype=complex))
            for _ in range(int(round(T / dt))):
                f.t = st.t                           # explicit term evaluated at t_n
                st.step()
            errs.append(abs(st.state[0, 2, 1].real - exact(r)) / abs(exact(r)))
        ratios[r] = [errs[i] / errs[i + 1] for i in range(3)]
        print(f"\nr = {r * DAY_SECONDS:.3f}/day: relative errors "
              + " ".join(f"{e:.3e}" for e in errs) + "; ratios "
              + " ".join(f"{x:.4f}" for x in ratios[r]))
    assert all(1.8 < x < 2.1 for x in ratios[HS.k_f])      # first order: the damping lag
    assert all(3.99 < x < 4.01 for x in ratios[0.0])        # second order without damping


def test_raw_filter_with_alpha_above_one_half_is_first_order():
    """The RAW filter's alpha sets its order (Williams 2009, 2011): on the real
    stepper, x' = i w x + sin(W t) (w = 1.5e-4 s^-1, 2 days): nu = 0 and
    alpha = 1/2 are second order; alpha = 1 (Robert–Asselin) first order;
    the protocol's alpha = 0.53 has a small first-order component that takes
    over below ~300 s (ratios 4.01, 3.85, 3.44, 2.76)."""
    from scipy.integrate import solve_ivp
    from tropoi.spatial.sigma_coordinate import SigmaGrid
    from tropoi.temporal.semi_implicit import SemiImplicitLeapfrogStepper, SemiImplicitOperator
    op = SemiImplicitOperator(SigmaGrid.uniform(2), l_max=3, radius=6.371e6, r_dry=287.0,
                              cp_dry=1004.0, t_ref=300.0)
    W, T, w = 2 * math.pi / (1.3 * DAY_SECONDS), 2 * DAY_SECONDS, 1.5e-4

    class Osc:
        max_rate = 0.0
        t = 0.0

        def __call__(self, x):
            out = np.zeros_like(x)
            out[0, 2, 1] = 1j * w * x[0, 2, 1] + math.sin(W * self.t)
            return out

        def signature(self):
            return {"kind": "osc"}

    sol = solve_ivp(lambda t, y: [-w * y[1] + math.sin(W * t), w * y[0]], (0.0, T), [0.0, 0.0],
                    rtol=1e-13, atol=1e-10)
    exact = sol.y[0, -1] + 1j * sol.y[1, -1]
    ratios = {}
    for nu, alpha in ((0.0, ALPHA), (NU, 0.5), (NU, ALPHA), (NU, 1.0)):
        errs = []
        for dt in (1200.0, 600.0, 300.0, 150.0, 75.0):
            f = Osc()
            st = SemiImplicitLeapfrogStepper(lambda x: np.zeros_like(x), op, dt, raw_nu=nu,
                                             raw_alpha=alpha, explicit_terms=(f,),
                                             startup_substeps=64)
            st.initialize(np.zeros((3 * op.K + 1, op.n, op.n), dtype=complex))
            for _ in range(int(round(T / dt))):
                f.t = st.t
                st.step()
            errs.append(abs(st.state[0, 2, 1] - exact) / abs(exact))
        ratios[(nu, alpha)] = [errs[i] / errs[i + 1] for i in range(4)]
        print(f"\nRAW nu={nu} alpha={alpha}: ratios "
              + " ".join(f"{x:.3f}" for x in ratios[(nu, alpha)]))
    assert all(3.95 < x < 4.1 for x in ratios[(0.0, ALPHA)])
    assert all(3.95 < x < 4.1 for x in ratios[(NU, 0.5)])
    assert all(1.9 < x < 2.25 for x in ratios[(NU, 1.0)])
    r = ratios[(NU, ALPHA)]
    assert r[0] > 3.9 and r[-1] < 3.0 and all(r[i] > r[i + 1] for i in range(3))


# ---------------------------------------------------------------------------
# GPU checks (b)-(d)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def hs():
    if not _has_cuda():
        pytest.skip("CUDA/CuPy not available")
    from tropoi.run.held_suarez.config import development_config
    from tropoi.run.held_suarez.model import build_model, build_physics, initial_state
    from tropoi.temporal.semi_implicit import SemiImplicitOperator
    cfg = development_config(days=5)
    model = build_model(cfg)
    terms, dampers = build_physics(cfg, model)
    op = SemiImplicitOperator.from_model(model, t_ref=cfg.t_ref)
    x0, _ = initial_state(cfg)
    return cfg, model, terms, dampers, op, x0


BLOCKS = ("zeta", "delta", "T", "q")


def _blocks(K):
    return {"zeta": slice(0, K), "delta": slice(K, 2 * K), "T": slice(2 * K, 3 * K),
            "q": slice(3 * K, 3 * K + 1)}


def _norm(a):
    import cupy as cp
    return float(cp.linalg.norm(a))


@gpu
@pytest.mark.parametrize("dt", [DT_SELECTED, DT_CANDIDATE])
def test_five_day_stability_with_the_computational_mode_monitored(hs, dt):
    """(c) 5 days, T21 L10, complete scheme, dt >= 900 s. Per step:
    D^n = X^n - X^{n-1}_f per block (the user-specified monitor) and the
    computational-mode indicator |D^n - D^{n-1}| / |D^n + D^{n-1}| (the
    leapfrog computational mode (-1)^n c doubles the numerator and cancels in
    the denominator; smooth physical evolution makes it O(omega dt)). Neither
    may grow after the day-1 spin-up transient; the state stays valid."""
    from tropoi.spatial.states.primitive_equations import PrimitiveEquationsState
    from tropoi.temporal.semi_implicit import SemiImplicitLeapfrogStepper
    cfg, model, terms, dampers, op, x0 = hs
    K = cfg.nlev
    B = _blocks(K)
    st = SemiImplicitLeapfrogStepper(model.tendency, op, dt, explicit_terms=terms, dampers=dampers)
    st.initialize(x0)
    spd = int(round(DAY_SECONDS / dt))
    daily, cur, prev = [], None, None
    for n in range(1, 5 * spd + 1):
        st.step()
        D = st.state - st.x_prev
        if cur is None:
            cur = {"D": dict.fromkeys(B, 0.0), "ind": dict.fromkeys(B, 0.0)}
        for b, s in B.items():
            cur["D"][b] = max(cur["D"][b], _norm(D[s]))
            if prev is not None and n > 2:
                cur["ind"][b] = max(cur["ind"][b],
                                    _norm(D[s] - prev[s]) / max(_norm(D[s] + prev[s]), 1e-300))
        prev = D
        if n % spd == 0:
            model.validate_state(PrimitiveEquationsState(st.state), context=f"(day {n // spd})")
            daily.append(cur)
            cur = None
    print(f"\ndt = {dt:.0f} s, startup substeps {st.startup_substeps}")
    for d, row in enumerate(daily, 1):
        print(f"  day {d}: max|X^n - X^n-1| " + " ".join(f"{b}={row['D'][b]:.3e}" for b in BLOCKS)
              + " | indicator " + " ".join(f"{b}={row['ind'][b]:.3e}" for b in BLOCKS))
    for b in BLOCKS:
        ind = [row["ind"][b] for row in daily]
        dd = [row["D"][b] for row in daily]
        assert max(ind[2:]) <= 1.25 * ind[1], (b, ind)          # no growth after spin-up
        assert max(ind[1:]) < 0.3, (b, ind)                      # physical mode dominates
        assert max(dd[2:]) <= 1.25 * dd[1], (b, dd)


@gpu
def test_rk4_startup_of_the_complete_tendency_at_the_si_dt(hs):
    """(d) The startup substep count comes from L (and the damping rates) but
    integrates the complete nonlinear right-hand side. Verified after the HS
    forcing is installed: X^1 from 1, 2, 4, 8 substeps converges to a
    32-substep reference at fourth order (error ratios -> 16, i.e. inside
    RK4's stable, asymptotic range), the chosen count is within 1e-4 of the
    reference relative to X^1 - X^0, and the first leapfrog steps after it
    show no growth of the computational mode."""
    from tropoi.temporal.hooks import complete_tendency
    from tropoi.temporal.integration import rk4_step_array
    from tropoi.temporal.semi_implicit import SemiImplicitLeapfrogStepper
    cfg, model, terms, dampers, op, x0 = hs
    K = cfg.nlev
    B = _blocks(K)
    dt = DT_SELECTED
    st = SemiImplicitLeapfrogStepper(model.tendency, op, dt, explicit_terms=terms, dampers=dampers)
    bare = SemiImplicitLeapfrogStepper(model.tendency, op, dt)
    print(f"\nstartup substeps at dt = {dt:.0f} s: {bare.startup_substeps} (L only), "
          f"{st.startup_substeps} (L + damping rates + Newtonian k_T)")
    f = complete_tendency(model.tendency, terms, dampers)

    def startup(nsub):
        y = x0
        for _ in range(nsub):
            y = rk4_step_array(f, y, 0.0, dt / nsub)
        return y

    ref = startup(32)
    errs = {}
    for nsub in (1, 2, 4, 8):
        y = startup(nsub)
        errs[nsub] = {b: _norm(y[s] - ref[s]) / _norm(ref[s] - x0[s]) for b, s in B.items()}
        print(f"  n_sub={nsub}: " + " ".join(f"{b}={errs[nsub][b]:.3e}" for b in BLOCKS))
    for b in BLOCKS:
        r = [errs[a][b] / errs[2 * a][b] for a in (1, 2, 4)]
        print(f"  {b:5s} ratios {r[0]:.2f} {r[1]:.2f} {r[2]:.2f}")
        assert r[1] > 10.0 and r[2] > 12.0, (b, r)          # fourth order, asymptotic
        assert errs[st.startup_substeps][b] < 1e-4, b
    st.initialize(x0)
    st.step()
    assert st.state.tobytes() == startup(st.startup_substeps).tobytes()   # the stepper's X^1
    ind, prev = [], None
    for n in range(2, 23):
        st.step()
        D = st.state - st.x_prev
        if prev is not None:
            ind.append(max(_norm(D[s] - prev[s]) / max(_norm(D[s] + prev[s]), 1e-300)
                           for s in B.values()))
        prev = D
    print("  first leapfrog steps, computational-mode indicator: "
          + " ".join(f"{v:.3f}" for v in ind))
    assert max(ind[10:]) <= max(ind[:10]), ind                 # no early growth


# (b) settings, from the measured asymptotic range (DEVLOG 2026-09-27 S4b (b)): the HS forcing
# switches on at t = 0 and excites gravity waves up to the fastest discrete mode at the T21 cut
# (omega = 7.7e-4 s^-1). Its SI phase error accumulates as omega^3 dt^2 t / 3, O(1) over 2 days
# for dt >~ 200 s, so errors at 2400/1200/600 s saturate and their ratios are irregular
# (identical with RAW off and with exact exponential damping factors). The step sequence is
# therefore halved to 300/150/75 s. RK4 at 150 s differs from RK4 at 37.5 s by <= 5.4e-7 of the
# 2-day change (RK4 300 s: 8.5e-6), far below the SI errors (>= 5e-5).
CONV_DTS = (300.0, 150.0, 75.0)
CONV_REF_DT = 150.0


def _convergence(hs, with_dampers: bool, raw_nu: float = NU):
    import time
    from tropoi.temporal.hooks import complete_tendency
    from tropoi.temporal.semi_implicit import SemiImplicitLeapfrogStepper
    from tropoi.temporal.steppers import RK4Stepper
    cfg, model, terms, dampers, op, x0 = hs
    ds = dampers if with_dampers else ()
    B = _blocks(cfg.nlev)
    t_end = 2 * DAY_SECONDS
    t0 = time.perf_counter()
    rk = RK4Stepper(complete_tendency(model.tendency, terms, ds), CONV_REF_DT)
    rk.initialize(x0)
    for _ in range(int(round(t_end / CONV_REF_DT))):
        rk.step()
    ref = rk.state
    label = "complete scheme" if with_dampers else "forcing + SI, no dampers"
    print(f"\n{label} (RAW nu = {raw_nu}): RK4 reference dt = {CONV_REF_DT} s "
          f"({time.perf_counter() - t0:.0f} s)")
    errs = {}
    for dt in CONV_DTS:
        t1 = time.perf_counter()
        st = SemiImplicitLeapfrogStepper(model.tendency, op, dt, raw_nu=raw_nu, raw_alpha=ALPHA,
                                         explicit_terms=terms, dampers=ds)
        st.initialize(x0)
        for _ in range(int(round(t_end / dt))):
            st.step()
        errs[dt] = {b: _norm(st.state[s] - ref[s]) / _norm(ref[s] - x0[s]) for b, s in B.items()}
        print(f"  SI dt = {dt:5.1f} s ({time.perf_counter() - t1:.0f} s): "
              + " ".join(f"{b}={v:.3e}" for b, v in errs[dt].items()))
    ratios = {}
    for b in BLOCKS:
        e = [errs[d][b] for d in CONV_DTS]
        ratios[b] = [e[i] / e[i + 1] for i in range(len(e) - 1)]
        print(f"  {b:5s} ratios " + " ".join(f"{x:.3f}" for x in ratios[b]))
    return errs, ratios


@gpu
def test_forcing_and_si_converge_second_order_without_the_first_order_terms(hs):
    """(b) 2 days, T21 L10, HS initial state, Newtonian forcing + SI; no
    drag/del^8 in either path and RAW off — the two measured first-order terms
    (damping lag; RAW with alpha != 1/2, see the scalar tests) removed:
    second order in the asymptotic range (S3's criterion: first ratio in
    (3.4, 4.6), second in (3.6, 4.4))."""
    _, ratios = _convergence(hs, with_dampers=False, raw_nu=0.0)
    for b in BLOCKS:
        r1, r2 = ratios[b]
        assert 3.4 < r1 < 4.6 and 3.6 < r2 < 4.4, (b, ratios[b])


@gpu
def test_complete_scheme_convergence_is_limited_by_the_damping_lag(hs):
    """(b) the complete scheme (forcing + drag + del^8 + SI + RAW(0.1, 0.53))
    against RK4 of the same right-hand side. Measured: asymptotically FIRST
    order. Two frozen choices of plan §3 each carry a first-order term,
    isolated on the real stepper by the scalar tests: the damping evaluated
    at n+1 (user-mandated backward-Euler factor on X^{n+1}; dominant, via the
    Rayleigh drag on zeta) and RAW with alpha = 0.53 (small). This test records
    that behaviour; it is a DECISION NEEDED item (STATUS.md), not a tolerance:
    every error must still decrease, zeta converges at first order, and the
    order is between 1 and 2 elsewhere."""
    errs, ratios = _convergence(hs, with_dampers=True)
    for b in BLOCKS:
        assert all(x > 1.8 for x in ratios[b]), (b, ratios[b])       # converging, >= 1st order
        assert all(x < 4.4 for x in ratios[b]), (b, ratios[b])
    assert 1.8 < ratios["zeta"][-1] < 2.2, ratios["zeta"]           # first order: the lag term
