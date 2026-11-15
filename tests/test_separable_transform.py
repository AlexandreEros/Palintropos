"""Separable FFT+Legendre transform against the dense basis-matrix path.

`SeparableGaussSphericalHarmonics` replaces one n_points x n_basis complex
matrix product with a cuFFT longitude stage plus a custom-kernel Legendre
stage. It is a faster evaluator of the *same* basis, not a new basis, so the
whole correctness question is: does it agree with the dense evaluator, which
stays the numerical reference and the default?

Only the summation order differs (Fourier sum along longitude first, then the
latitude quadrature, instead of one sum over all points), so agreement is to
round-off and not bit-for-bit. Tolerances here are the same 1e-13 relative
figure `test_batched_transforms.py` uses for the batched-vs-per-level
invariant; measured agreement is ~1e-14.

Grid shapes are the ones production uses: a state grid (nlon = 2 l_max + 2)
and a 3/2-rule product grid, which has an ODD nlon -- the case where a
half-spectrum FFT has no Nyquist bin at all.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest


def _has_cuda():
    try:
        import cupy as cp
        return cp.is_available()
    except Exception:  # pragma: no cover - import guard
        return False


pytestmark = pytest.mark.skipif(not _has_cuda(), reason="CUDA/CuPy not available")

# (nlat, nlon, l_max): a state grid and a 3/2-rule product grid (odd nlon).
SHAPES = [(32, 64, 22), (34, 67, 22)]
RADIUS = 6.371e6
TOL = 1e-13


def _rel(a, b):
    import cupy as cp
    return float(cp.abs(a - b).max() / cp.abs(b).max())


def _pair(nlat, nlon, l_max):
    """(dense, separable) evaluators on the same grid, points and weights."""
    from tropoi.spatial.grids.latlon_grid import GaussLatLonGridGeometry
    from tropoi.spatial.transforms.fast_geodesic_sh import PointSetSphericalHarmonics
    from tropoi.spatial.transforms.separable_gauss_sh import (
        SeparableGaussSphericalHarmonics)
    g = GaussLatLonGridGeometry(nlat, nlon, radius=RADIUS)
    dense = PointSetSphericalHarmonics(g.point_latitudes, g.point_longitudes,
                                       l_max, weights=g.solid_angle_weights)
    sep = SeparableGaussSphericalHarmonics(g.latitudes, nlon, l_max,
                                           weights=g.solid_angle_weights)
    return g, dense, sep


def _coeffs(l_max, k, seed=0):
    """Coefficients of k real fields: m <= l, and m = 0 rows real."""
    import cupy as cp
    rng = np.random.default_rng(seed)
    n = l_max + 1
    c = np.tril(rng.standard_normal((k, n, n))
                + 1j * rng.standard_normal((k, n, n)))
    c[:, :, 0] = c[:, :, 0].real
    return cp.asarray(c)


# ---------------------------------------------------------------------------
# Agreement with the dense reference
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("nlat,nlon,l_max", SHAPES)
def test_synthesis_matches_dense(nlat, nlon, l_max):
    _, dense, sep = _pair(nlat, nlon, l_max)
    c = _coeffs(l_max, 5, seed=1)
    got, ref = sep.inv_transform_batch(c), dense.inv_transform_batch(c)
    assert got.shape == ref.shape
    assert got.dtype == ref.dtype
    assert _rel(got, ref) < TOL


@pytest.mark.parametrize("nlat,nlon,l_max", SHAPES)
def test_analysis_matches_dense(nlat, nlon, l_max):
    _, dense, sep = _pair(nlat, nlon, l_max)
    fields = dense.inv_transform_batch(_coeffs(l_max, 5, seed=2))
    got, ref = sep.transform_batch(fields), dense.transform_batch(fields)
    assert got.shape == ref.shape
    assert _rel(got, ref) < TOL


@pytest.mark.parametrize("nlat,nlon,l_max", SHAPES)
def test_scalar_paths_match_dense(nlat, nlon, l_max):
    """The scalar paths must land on the same flat (n_points,) layout."""
    _, dense, sep = _pair(nlat, nlon, l_max)
    c = _coeffs(l_max, 1, seed=3)[0]
    f_got, f_ref = sep.inv_transform(c), dense.inv_transform(c)
    assert f_got.shape == f_ref.shape == (nlat * nlon,)
    assert _rel(f_got, f_ref) < TOL
    assert _rel(sep.transform(f_ref), dense.transform(f_ref)) < TOL


@pytest.mark.parametrize("nlat,nlon,l_max", SHAPES)
def test_batch_matches_per_level(nlat, nlon, l_max):
    """Same invariant the dense path is held to: batching changes nothing."""
    import cupy as cp
    _, _, sep = _pair(nlat, nlon, l_max)
    c = _coeffs(l_max, 4, seed=4)
    assert _rel(sep.inv_transform_batch(c),
                cp.stack([sep.inv_transform(c[k]) for k in range(4)])) < TOL
    f = sep.inv_transform_batch(c)
    assert _rel(sep.transform_batch(f),
                cp.stack([sep.transform(f[k]) for k in range(4)])) < TOL


@pytest.mark.parametrize("nlat,nlon,l_max", SHAPES)
def test_roundtrip_and_constant_field_exact(nlat, nlon, l_max):
    """Exact quadrature survives the factorization: a band-limited field
    round-trips, and f == 1 gives a_00 = sqrt(4 pi)."""
    import cupy as cp
    g, _, sep = _pair(nlat, nlon, l_max)
    c = _coeffs(l_max, 3, seed=5)
    assert _rel(sep.transform_batch(sep.inv_transform_batch(c)), c) < 1e-12
    a00 = complex(sep.transform(cp.ones(g.n_points, dtype=cp.float64))[0, 0])
    assert abs(a00 - np.sqrt(4.0 * np.pi)) / np.sqrt(4.0 * np.pi) < 1e-13


@pytest.mark.parametrize("nlat,nlon,l_max,k", [
    (2, 4, 0, 1),      # smallest legal grid, l_max = 0
    (2, 5, 1, 1),      # odd nlon (a half-spectrum with no Nyquist bin), K = 1
    (3, 17, 8, 2),     # under-resolved LATITUDE: nlat = 3 < l_max + 1 = 9
    (17, 36, 17, 7),   # prime nlat, nlon not a power of two
])
def test_degenerate_shapes_match_dense(nlat, nlon, l_max, k):
    """Shapes that stress the indexing, and one where the latitude
    quadrature is inexact.

    Under-resolved latitude matters most: this evaluator must reproduce the
    dense path's inexact quadrature, not quietly improve on it. Both sum the
    same Legendre values over the same nlat nodes, so the leakage is
    identical and a backend swap never moves a result.
    """
    import cupy as cp
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")      # the under-resolution warning
        _, dense, sep = _pair(nlat, nlon, l_max)
    c = _coeffs(l_max, k, seed=7)
    ref = dense.inv_transform_batch(c)
    assert _rel(sep.inv_transform_batch(c), ref) < TOL
    got = sep.transform_batch(ref)
    assert _rel(got, dense.transform_batch(ref)) < TOL
    # A real field has real m = 0 coefficients, exactly.
    assert float(cp.abs(got[:, :, 0].imag).max()) == 0.0


def test_extended_matches_dense_extended():
    """`extended()` is what the vector weak form analyzes against; it must
    equal a dense basis matrix of the extended degree on the same points."""
    from tropoi.spatial.transforms.fast_geodesic_sh import PointSetSphericalHarmonics
    nlat, nlon, l_max = 34, 67, 22
    g, _, sep = _pair(nlat, nlon, l_max)
    ext = sep.extended(l_max + 1)
    assert ext.l_max == l_max + 1
    dense_ext = PointSetSphericalHarmonics(g.point_latitudes,
                                           g.point_longitudes, l_max + 1,
                                           weights=g.solid_angle_weights)
    fields = sep.inv_transform_batch(_coeffs(l_max, 4, seed=6))
    assert _rel(ext.transform_batch(fields),
                dense_ext.transform_batch(fields)) < TOL


def test_rejects_complex_input():
    """The longitude stage is a real-to-complex FFT; say so rather than
    silently analyzing the real part."""
    import cupy as cp
    _, _, sep = _pair(32, 64, 22)
    with pytest.raises(TypeError, match="real fields"):
        sep.transform_batch(cp.zeros((2, 2048), dtype=cp.complex128))


# ---------------------------------------------------------------------------
# Applicability gate and backend selection
# ---------------------------------------------------------------------------

def test_applicability_gate():
    import cupy as cp
    from tropoi.spatial.transforms.separable_gauss_sh import (
        SeparableGaussSphericalHarmonics, SeparableStructureError,
        separable_is_applicable)
    assert separable_is_applicable(32, 64, 22)[0]
    # A longitude axis too coarse for the truncation: wavenumbers would have
    # to share real-FFT bins at or above Nyquist.
    ok, reason = separable_is_applicable(32, 40, 22)
    assert not ok and "2*l_max < nlon" in reason
    # The factorization needs a longitude-independent quadrature weight.
    w = cp.ones(32 * 64)
    w[5] = 2.0
    ok, reason = separable_is_applicable(32, 64, 22, w)
    assert not ok and "longitude" in reason
    # The constructor does not paper over either case.
    with pytest.raises(SeparableStructureError):
        SeparableGaussSphericalHarmonics(cp.zeros(32), 40, 22)


def test_wrapper_selects_backend_and_default_is_separable(monkeypatch):
    from tropoi.spatial.grids.latlon_grid import (
        GaussLatLonGridGeometry, GaussLatLonSphericalHarmonics,
        default_transform_backend)
    from tropoi.spatial.transforms.fast_geodesic_sh import PointSetSphericalHarmonics
    from tropoi.spatial.transforms.separable_gauss_sh import (
        SeparableGaussSphericalHarmonics)
    monkeypatch.delenv("TROPOI_SH_BACKEND", raising=False)
    assert default_transform_backend() == "separable"
    g = GaussLatLonGridGeometry(32, 64, radius=RADIUS)
    assert isinstance(GaussLatLonSphericalHarmonics(g, 22).sh,
                      SeparableGaussSphericalHarmonics)
    # The deprecated reference path stays reachable, and says so.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        explicit_dense = GaussLatLonSphericalHarmonics(
            g, 22, transform_backend="dense")
    assert isinstance(explicit_dense.sh, PointSetSphericalHarmonics)
    assert any(issubclass(w.category, DeprecationWarning)
               and "DEPRECATED" in str(w.message) for w in caught)
    monkeypatch.setenv("TROPOI_SH_BACKEND", "dense")
    assert default_transform_backend() == "dense"
    assert isinstance(GaussLatLonSphericalHarmonics(g, 22).sh,
                      PointSetSphericalHarmonics)
    monkeypatch.setenv("TROPOI_SH_BACKEND", "nonsense")
    with pytest.raises(ValueError):
        default_transform_backend()


def test_inapplicable_grid_warns_and_falls_back():
    """A fallback is never silent (docs/validation/r3 provenance rule)."""
    from tropoi.spatial.grids.latlon_grid import (
        GaussLatLonGridGeometry, GaussLatLonSphericalHarmonics)
    from tropoi.spatial.transforms.fast_geodesic_sh import PointSetSphericalHarmonics
    g = GaussLatLonGridGeometry(32, 40, radius=RADIUS)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sh = GaussLatLonSphericalHarmonics(g, 22, transform_backend="separable")
    assert isinstance(sh.sh, PointSetSphericalHarmonics)
    assert sh.transform_backend == "dense"
    assert any("not applicable" in str(w.message) for w in caught)


# ---------------------------------------------------------------------------
# End to end: the dynamics the transform serves
# ---------------------------------------------------------------------------

def test_pe_tendency_matches_dense(monkeypatch):
    """The whole point: a primitive-equations tendency -- synthesis, the
    pointwise nonlinear products, analysis and the extended-degree vector
    weak form -- comes out the same whichever evaluator computed it."""
    import cupy as cp
    from tropoi.spatial.environment import PlanetaryParameters
    from tropoi.spatial.planet import Planet
    from tropoi.spatial.sigma_coordinate import SigmaGrid
    from tropoi.temporal.tendencies.primitive_equations import PrimitiveEquationsModel

    l_max, nlat, nlon, k = 22, 32, 64, 4

    def build():
        p = PlanetaryParameters.ideal_sphere(RADIUS, 86400.0)
        pl = Planet.generate(params=p, grid_type="latlon", nlat=nlat,
                             nlon=nlon, l_max=l_max, product_quadrature="fine")
        return PrimitiveEquationsModel(pl, SigmaGrid.uniform(k), r_dry=287.0,
                                       cp_dry=1004.0, batched_transforms=True)

    # Name both evaluators: neither end may ride on the ambient default.
    monkeypatch.setenv("TROPOI_SH_BACKEND", "dense")
    dense_model = build()
    monkeypatch.setenv("TROPOI_SH_BACKEND", "separable")
    sep_model = build()
    assert sep_model.so.backend.describe("fine")["transform_backend"] == "separable"
    assert dense_model.so.backend.describe("fine")["transform_backend"] == "dense"

    rng = np.random.default_rng(17)
    n = l_max + 1
    x = np.tril(rng.standard_normal((3 * k + 1, n, n))
                + 1j * rng.standard_normal((3 * k + 1, n, n)))
    x[:, :, 0] = x[:, :, 0].real
    x *= 1e-6                                    # vorticity-scale amplitudes
    x[2 * k:3 * k, 0, 0] = 300.0 * np.sqrt(4.0 * np.pi)   # ~300 K
    x[3 * k, 0, 0] = np.log(1e5) * np.sqrt(4.0 * np.pi)   # ~1000 hPa
    x = cp.asarray(x)

    a, b = dense_model.tendency(x), sep_model.tendency(x)
    assert a.shape == b.shape
    assert _rel(b, a) < 1e-11
