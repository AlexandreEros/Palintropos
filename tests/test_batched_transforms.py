"""Level-batched spherical-harmonic transforms (CUDA-gated).

`PointSetSphericalHarmonics.transform_batch` / `inv_transform_batch` analyze
and synthesize a stack of K fields with one GEMM each. They are opt-in: the
per-level `transform` / `inv_transform` stay byte-for-byte unchanged (pinned
run hashes depend on them), and the batched results must agree with the
stacked per-level results to round-off.
"""
from __future__ import annotations

import numpy as np
import pytest


def _has_cuda():
    try:
        import cupy as cp
        return cp.is_available()
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _has_cuda(), reason="CUDA/CuPy not available")


def _rel(a, b):
    import cupy as cp
    return float(cp.abs(a - b).max() / cp.abs(b).max())


@pytest.fixture(scope="module")
def sh():
    from tropoi.spatial.planet import Planet
    from tropoi.spatial.environment import PlanetaryParameters
    planet = Planet.generate(params=PlanetaryParameters.from_earth_like(day_hours=24.0),
                             grid_type="latlon", nlat=32, nlon=64, l_max=21,
                             grid_resolution=3)
    return planet.sh


def _random_coeffs(sh, K, seed=0):
    import cupy as cp
    rng = np.random.default_rng(seed)
    n = sh.l_max + 1
    c = rng.standard_normal((K, n, n)) + 1j * rng.standard_normal((K, n, n))
    c = np.tril(c)               # (l, m) layout: m <= l
    c[:, :, 0] = c[:, :, 0].real  # m = 0 rows real
    return cp.asarray(c)


def test_inv_transform_batch_matches_per_level(sh):
    import cupy as cp
    K = 5
    c = _random_coeffs(sh, K)
    ref = cp.stack([sh.inv_transform(c[k]) for k in range(K)])
    got = sh.inv_transform_batch(c)
    assert got.shape == ref.shape
    assert _rel(got, ref) < 1e-13


def test_transform_batch_matches_per_level(sh):
    import cupy as cp
    K = 5
    fields = sh.inv_transform_batch(_random_coeffs(sh, K, seed=1))
    ref = cp.stack([sh.transform(fields[k]) for k in range(K)])
    got = sh.transform_batch(fields)
    assert got.shape == ref.shape
    assert _rel(got, ref) < 1e-13


def test_per_level_transform_is_unchanged_by_batching_support(sh):
    """The scalar path must keep its exact arithmetic (pinned hashes)."""
    import cupy as cp
    c = _random_coeffs(sh, 1, seed=2)[0]
    f = sh.inv_transform(c)
    w = f * sh.weights
    expected = cp.dot(sh.Y_matrix.conj().T, w)
    got = sh.transform(f)[sh.l_indices, sh.m_indices]
    assert got.tobytes() == expected.tobytes()


def test_pe_tendency_batched_matches_per_level():
    import cupy as cp
    from tropoi.cli.pe import build_pe_model
    cfg = dict(day_hours=24.0, radius_earth_units=1.0, resolution=3, lmax=21,
               grid="latlon", nlat=32, nlon=64, topography="flat", nlev=6,
               r_dry=287.04, cp_dry=1004.64)
    m_ref = build_pe_model(cfg)
    m_bat = build_pe_model(dict(cfg, batched_transforms=True))
    K, n = 6, 22
    rng = np.random.default_rng(3)
    y = np.zeros((3 * K + 1, n, n), dtype=np.complex128)
    y[2 * K:3 * K, 0, 0] = 260.0 * np.sqrt(4 * np.pi)
    y[3 * K, 0, 0] = np.log(1e5) * np.sqrt(4 * np.pi)
    pert = 1e-5 * (rng.standard_normal((3 * K + 1, 8, 8)) + 1j * rng.standard_normal((3 * K + 1, 8, 8)))
    pert = np.tril(pert); pert[:, :, 0] = pert[:, :, 0].real
    y[:, :8, :8] += pert
    y[2 * K:3 * K, :8, :8] += 1e3 * pert[2 * K:3 * K]
    y = cp.asarray(y)
    ref = m_ref.tendency(y)
    got = m_bat.tendency(y)
    assert _rel(got, ref) < 1e-12


def test_pe_batched_rest_state_is_exactly_zero():
    import cupy as cp
    from tropoi.cli.pe import build_pe_model
    cfg = dict(day_hours=24.0, radius_earth_units=1.0, resolution=3, lmax=21,
               grid="latlon", nlat=32, nlon=64, topography="flat", nlev=6,
               r_dry=287.04, cp_dry=1004.64, batched_transforms=True)
    m = build_pe_model(cfg)
    K, n = 6, 22
    y = cp.zeros((3 * K + 1, n, n), dtype=cp.complex128)
    y[2 * K:3 * K, 0, 0] = 260.0 * np.sqrt(4 * np.pi)
    y[3 * K, 0, 0] = np.log(1e5) * np.sqrt(4 * np.pi)
    assert float(cp.abs(m.tendency(y)).max()) == 0.0
