"""Opt-in retained truncation of the PE core (docs/held_suarez/DEALIASING_AUDIT.md).

``PrimitiveEquationsModel(retained_truncation=L)`` with stored ``l_max = L + 1``
evolves every degree <= L (Dinosaur's spectral layout) instead of the 2/3
product cut. The default (None) is the historical cut, bitwise.

CPU: the SI operator follows the model's retained cut; configuration checks.
GPU (T21-sized): default path unchanged; degree l_max refused; the top
retained degree keeps the exact rest cancellation that degree l_max breaks;
every quadratic term is analyzed exactly (vs a 3x overresolved Gauss grid);
cubic terms alias only at the 1e-6 level for an HS-like spectrum; the
storage-only degree stays exactly zero; rest linearization is second order
with the SI operator's default mask; scalar and batched paths agree;
terrain is accepted up to the retained degree.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tropoi.spatial.sigma_coordinate import SigmaGrid
from tropoi.spatial.truncation import product_truncation_cut

R_D, CP_D, A = 2.0 / 7.0 * 1004.0, 1004.0, 6.371e6
MONO = math.sqrt(4 * math.pi)
L = 21          # retained
S = L + 1       # stored
K = 6
BLOCKS = {"zeta": slice(0, K), "delta": slice(K, 2 * K), "T": slice(2 * K, 3 * K),
          "q": slice(3 * K, 3 * K + 1)}


def _has_cuda():
    try:
        import cupy as cp
        return cp.is_available()
    except Exception:
        return False


gpu = pytest.mark.skipif(not _has_cuda(), reason="CUDA/CuPy not available")


# ---------------------------------------------------------------------------
# CPU
# ---------------------------------------------------------------------------

def test_semi_implicit_operator_follows_the_model_retained_cut():
    from tropoi.temporal.semi_implicit import SemiImplicitOperator

    class Stub:
        sigma = SigmaGrid.uniform(4)
        l_max = S
        R = A
        r_dry = R_D
        cp_dry = CP_D
        retained_truncation = L

    op = SemiImplicitOperator.from_model(Stub(), t_ref=300.0)
    assert op.fast_cut == L and list(op.mask) == [1.0] * (L + 1) + [0.0]
    del Stub.retained_truncation                       # models without the attribute
    assert SemiImplicitOperator.from_model(Stub(), t_ref=300.0).fast_cut == \
        product_truncation_cut(S)


def test_held_suarez_layouts():
    from tropoi.run.held_suarez.config import (development_config, legacy_production_config,
                                               production_config, smoke_config)
    p = production_config()
    assert (p.l_max, p.retained_cut, p.hyperdiffusion_degree, p.nlat, p.nlon) == (43, 42, 42, 64, 128)
    lg = legacy_production_config()
    assert (lg.l_max, lg.retained_cut, lg.hyperdiffusion_degree) == (42, 28, 42)
    assert (smoke_config().l_max, smoke_config().retained_cut) == (22, 21)
    assert (development_config().l_max, development_config().retained_cut) == (21, 14)
    for bad in (43, 44, 0):
        with pytest.raises(ValueError):
            p.with_(retained_truncation=bad)


# ---------------------------------------------------------------------------
# GPU
# ---------------------------------------------------------------------------

def _model(l_max, nlat, nlon, *, rotating=True, batched=False, retained=None,
           surface_geopotential_lm=None):
    from tropoi.spatial.environment import PlanetaryParameters
    from tropoi.spatial.planet import Planet
    from tropoi.temporal.tendencies.primitive_equations import PrimitiveEquationsModel
    p = PlanetaryParameters.ideal_sphere(A, 86400.0 if rotating else 1e30)
    if not rotating:
        p.angular_velocity = 0.0
    pl = Planet.generate(params=p, grid_type="latlon", nlat=nlat, nlon=nlon, l_max=l_max,
                         product_quadrature="fine")
    return PrimitiveEquationsModel(pl, SigmaGrid.uniform(K), r_dry=R_D, cp_dry=CP_D,
                                   batched_transforms=batched, retained_truncation=retained,
                                   surface_geopotential_lm=surface_geopotential_lm)


def _rest(n):
    import cupy as cp
    x = cp.zeros((3 * K + 1, n, n), dtype=cp.complex128)
    x[2 * K:3 * K, 0, 0] = 300.0 * MONO
    x[3 * K, 0, 0] = math.log(1e5) * MONO
    return x


def _state(n, lb, seed, *, q=True, red=False):
    """Random state band-limited at lb. ``red``: HS-like magnitudes with an
    l^-1.5 coefficient spectrum; otherwise flat."""
    import cupy as cp
    rng = np.random.default_rng(seed)
    y = rng.standard_normal((3 * K + 1, n, n)) + 1j * rng.standard_normal((3 * K + 1, n, n))
    y = np.tril(y)
    y[:, :, 0] = y[:, :, 0].real
    y[:, lb + 1:, :] = 0.0
    y[:, 0, :] = 0.0
    l = np.arange(n, dtype=float)
    if red:
        y = y * np.where(l > 0, np.maximum(l, 1.0) ** -1.5, 0.0)[None, :, None]
        amps = (8e-6, 8e-7, 3.0, 2e-3)
    else:
        amps = tuple(a / math.sqrt(lb) for a in (4e-5, 1e-5, 8.0, 0.02))
    for i, a in enumerate(amps):
        rows = slice(i * K, (i + 1) * K) if i < 3 else slice(3 * K, 3 * K + 1)
        y[rows] *= a
    if not q:
        y[3 * K] = 0.0
    return _rest(n) + cp.asarray(y)


GRID = ((3 * S) // 2 + 1, 3 * S + 1)        # what the "fine" rule picks anyway


@pytest.fixture(scope="module")
def retained():
    if not _has_cuda():
        pytest.skip("CUDA/CuPy not available")
    return _model(S, 32, 64, retained=L)


@gpu
def test_default_is_the_historical_cut_bitwise():
    import cupy as cp
    a = _model(L, 32, 64)
    b = _model(L, 32, 64, retained=product_truncation_cut(L))
    assert a.retained_truncation == a._trunc_cut == product_truncation_cut(L)
    x = _state(L + 1, product_truncation_cut(L), 1)
    assert cp.asnumpy(a.tendency(x)).tobytes() == cp.asnumpy(b.tendency(x)).tobytes()


@gpu
def test_degree_l_max_is_refused_and_why():
    """Content AT l_max breaks the rest cancellation on every degree (the
    dropped degree-(l_max+1) part of sin(theta) d/dtheta), so l_max cannot be
    retained; the top retained degree of the store-(L+1) layout is exact."""
    import cupy as cp
    with pytest.raises(ValueError, match="derivative buffer"):
        _model(L, 32, 64, retained=L)
    legacy = _model(L, 32, 64, rotating=False)
    new = _model(S, 32, 64, rotating=False, retained=L)
    out = {}
    for name, m, n in (("legacy, q' at l_max", legacy, L + 1), ("retained, q' at L", new, S + 1)):
        x = _rest(n)
        x[3 * K, L, 1] += 1e-3
        out[name] = float(cp.abs(m.tendency(x)[0:K]).max())
    print("\n|zeta_dot| from rest + 1e-3 ln p_s at degree 21:", out)
    assert out["legacy, q' at l_max"] > 1e-12
    assert out["retained, q' at L"] < 1e-20


@gpu
def test_quadrature_is_exact_for_quadratic_terms_and_small_for_cubic(retained):
    """Store 22 / retain 21 on its production (3/2-rule) grid vs a 3x
    overresolved Gauss grid, same coefficients. Uniform ln p_s makes every
    term quadratic: exact. With ln p_s structure the V.grad(ln p_s) terms are
    cubic: aliasing at the 1e-6 level for an HS-like spectrum."""
    import cupy as cp
    over = _model(S, 3 * S + 3, 6 * S + 4, retained=L)
    worst = {}
    for label, kw in (("quadratic", {"q": False}), ("cubic, HS-like", {"red": True})):
        x = _state(S + 1, L, 2, **kw)
        a, b = retained.tendency(x), over.tendency(x)
        worst[label] = {n: float(cp.linalg.norm(a[s] - b[s]) / cp.linalg.norm(b[s]))
                        for n, s in BLOCKS.items() if float(cp.abs(b[s]).max()) > 0}
        print(f"\n{label}: relative error vs 3x overresolved " +
              " ".join(f"{n}={v:.1e}" for n, v in worst[label].items()))
    assert max(worst["quadratic"].values()) < 1e-12
    assert max(worst["cubic, HS-like"].values()) < 1e-4


@gpu
def test_storage_only_degree_stays_zero_and_paths_agree(retained):
    import cupy as cp
    batched = _model(S, 32, 64, batched=True, retained=L)
    x = _state(S + 1, L, 3)
    a, b = retained.tendency(x), batched.tendency(x)
    assert float(cp.abs(a[:, S, :]).max()) == 0.0 and float(cp.abs(a[:, :, S]).max()) == 0.0
    assert float(cp.abs(b[:, S, :]).max()) == 0.0
    d = float(cp.abs(a - b).max() / cp.abs(a).max())
    print(f"\nscalar vs batched, retained L = {L}: {d:.1e}")
    assert d < 1e-12


@gpu
def test_rest_linearization_is_second_order_with_the_default_si_mask():
    import cupy as cp
    from tropoi.temporal.semi_implicit import SemiImplicitOperator
    m = _model(S, 32, 64, rotating=False, retained=L)
    op = SemiImplicitOperator.from_model(m, t_ref=300.0)
    assert op.fast_cut == L
    x0 = _rest(S + 1)
    y = _state(S + 1, L, 4) - x0
    f0 = m.tendency(x0)
    res = {}
    for eps in (1e-2, 5e-3):
        r = m.tendency(x0 + eps * y) - f0 - eps * op.apply(y)
        res[eps] = {n: float(cp.abs(r[s]).max()) for n, s in BLOCKS.items()}
    ratios = {n: res[1e-2][n] / res[5e-3][n] for n in BLOCKS}
    print("\nlinearization residual ratios eps -> eps/2:", {n: round(v, 3) for n, v in ratios.items()})
    for n, v in ratios.items():
        assert 3.6 < v < 4.4, (n, v)


@gpu
def test_terrain_is_accepted_up_to_the_retained_degree():
    import cupy as cp
    phi = cp.zeros((S + 1, S + 1), dtype=cp.complex128)
    phi[L, 3] = 50.0
    _model(S, 32, 64, retained=L, surface_geopotential_lm=phi)
    phi[S, 3] = 50.0
    with pytest.raises(ValueError, match="dealiased product truncation"):
        _model(S, 32, 64, retained=L, surface_geopotential_lm=phi)
