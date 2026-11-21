"""tools/radiation/thai_vertical_coherence.py: exact area statistics and the
alternating-component fit (issue #47). CPU only."""
from __future__ import annotations

import importlib
import json
import math
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools" / "radiation"))
vc = importlib.import_module("thai_vertical_coherence")

L = 22          # dev-thai storage truncation
NLAT, NLON = 32, 64


def _random_coeffs(rng, size=L + 1, mean=0.0):
    a = rng.normal(size=(size, size)) + 1j * rng.normal(size=(size, size))
    a = np.tril(a)                                   # m <= l only
    a[:, 0] = a[:, 0].real                           # synthesis discards Im(a_l0)
    a[0, 0] = mean * math.sqrt(4.0 * math.pi)
    return a


def _spectral_inner(a, b):
    """(1/4 pi) * integral of f g over the sphere, by Parseval."""
    s = (a[:, 0].real * b[:, 0].real).sum() + 2.0 * (a[:, 1:] * b[:, 1:].conj()).real.sum()
    return s / (4.0 * math.pi)


def test_gauss_quadrature_statistics_equal_the_spectral_ones():
    """Grid means, variances and the correlation are the exact sphere integrals."""
    rng = np.random.default_rng(47)
    a, b = _random_coeffs(rng, mean=210.0), _random_coeffs(rng)
    b += 0.6 * a
    b[0, 0] = 0.0
    lat, lon, w = vc.gauss_mesh(NLAT, NLON)
    assert math.isclose(w.sum(), 1.0, rel_tol=1e-14)
    fa, fb = vc.synthesize_levels(np.stack([a, b]), lat, lon)
    assert math.isclose(vc.area_mean(fa, w), 210.0, rel_tol=1e-13)
    a0 = a.copy()
    a0[0, 0] = 0.0
    expected = _spectral_inner(a0, b) / math.sqrt(_spectral_inner(a0, a0) * _spectral_inner(b, b))
    assert abs(vc.weighted_correlation(fa, fb, w) - expected) < 1e-12
    assert abs(vc.area_mean(fb * fb, w) - _spectral_inner(b, b)) < 1e-10 * _spectral_inner(b, b)


def _constant(value):
    a = np.zeros((L + 1, L + 1), dtype=np.complex128)
    a[0, 0] = value * math.sqrt(4.0 * math.pi)
    return a


@pytest.mark.parametrize("value", [0.0, 1.0e-3, 210.0, 1.0e6])
def test_correlation_with_a_spatially_constant_field_is_undefined(value):
    """Zero variance: no Pearson correlation exists, whatever the other field.
    A synthesized constant carries only round-off spread (~1e-16 relative),
    which must not be turned into a correlation."""
    rng = np.random.default_rng(11)
    lat, lon, w = vc.gauss_mesh(NLAT, NLON)
    const, field = vc.synthesize_levels(np.stack([_constant(value), _random_coeffs(rng)]),
                                        lat, lon)
    assert vc.weighted_correlation(const, field, w) is None
    assert vc.weighted_correlation(field, const, w) is None
    assert vc.weighted_correlation(const, const, w) is None
    assert vc.weighted_correlation(np.zeros_like(field), field, w) is None


@pytest.mark.parametrize("scale", [1.0e-30, 1.0e-9, 1.0e12])
def test_correlation_does_not_depend_on_amplitude(scale):
    """The constant-field test is relative to each field's own size: a
    tiny but resolved pattern keeps its correlation."""
    rng = np.random.default_rng(5)
    a, b = _random_coeffs(rng), _random_coeffs(rng)
    b += 0.4 * a
    lat, lon, w = vc.gauss_mesh(NLAT, NLON)
    fa, fb = vc.synthesize_levels(np.stack([a, b]), lat, lon)
    reference = vc.weighted_correlation(fa, fb, w)
    assert abs(vc.weighted_correlation(scale * fa, fb, w) - reference) < 1e-12
    assert abs(vc.weighted_correlation(scale * fa, scale * fb, w) - reference) < 1e-12


@pytest.mark.parametrize("relative", [1.0e-6, 1.0e-10])
def test_small_perturbation_on_a_large_mean_is_still_a_correlation(relative):
    """Near-zero relative variance (a 200 K layer with a 2e-4 K or 2e-8 K
    pattern) is well-defined and equals the spectral (Parseval) value."""
    rng = np.random.default_rng(23)
    pa, pb = _random_coeffs(rng), _random_coeffs(rng)
    pa[0, 0] = pb[0, 0] = 0.0
    pb += 0.7 * pa
    norm = math.sqrt(_spectral_inner(pa, pa))
    pa, pb = pa * (200.0 * relative / norm), pb * (200.0 * relative / norm)
    lat, lon, w = vc.gauss_mesh(NLAT, NLON)
    fa, fb = vc.synthesize_levels(np.stack([_constant(200.0) + pa, _constant(180.0) + pb]),
                                  lat, lon)
    expected = _spectral_inner(pa, pb) / math.sqrt(_spectral_inner(pa, pa)
                                                  * _spectral_inner(pb, pb))
    got = vc.weighted_correlation(fa, fb, w)
    assert got is not None
    # round-off of the 200 K mean (~1e-13 K) against the pattern sets the accuracy
    assert abs(got - expected) < 1e-12 / relative


def test_state_with_a_motionless_layer_records_null_correlations():
    """A layer at rest (zero vorticity and divergence) gives undefined
    correlations with its neighbours, written as strict-JSON null."""
    rng = np.random.default_rng(8)
    nlev, sigma = 10, np.geomspace(2e-5, 0.9, 10)
    state = np.zeros((3 * nlev + 1, L + 1, L + 1), dtype=np.complex128)
    for k in range(nlev):
        if k != 0:
            state[k] = _random_coeffs(rng)
            state[nlev + k] = _random_coeffs(rng)
        state[2 * nlev + k, 0, 0] = 200.0 * math.sqrt(4.0 * math.pi)
    lat, lon, w = vc.gauss_mesh(NLAT, NLON)
    row = vc.state_metrics(state, nlev, sigma, lat, lon, w, top=8)
    for key in ("zeta_adjacent_correlation", "delta_adjacent_correlation"):
        assert row[key][0] is None
        assert all(v is not None and -1.0 <= v <= 1.0 for v in row[key][1:])
    assert row["zeta_rms"][0] == 0.0
    assert "null" in json.dumps(row, allow_nan=False)


def test_alternating_fit_recovers_a_planted_amplitude_and_ignores_smooth_trends():
    sigma = np.geomspace(2e-5, 3e-3, 8)
    s = np.log(sigma)
    smooth = 200.0 - 4.0 * s + 0.3 * s * s
    amp = np.array([[0.0, 1.5], [-11.7, 0.23]])
    t = smooth[:, None, None] + amp[None] * ((-1.0) ** np.arange(8))[:, None, None]
    np.testing.assert_allclose(vc.alternating_amplitude(t, sigma), amp, atol=1e-9)
    np.testing.assert_allclose(vc.alternating_amplitude(smooth, sigma), 0.0, atol=1e-9)
    with pytest.raises(ValueError):
        vc.alternating_amplitude(t[:4], sigma[:4])


def test_state_metrics_on_a_synthetic_state(tmp_path):
    """Correlations of identical layers are 1, of sign-flipped layers -1, and
    the alternating RMS is the planted uniform amplitude."""
    rng = np.random.default_rng(3)
    nlev, sigma = 10, np.geomspace(2e-5, 0.9, 10)
    zeta, delta = _random_coeffs(rng), _random_coeffs(rng)
    state = np.zeros((3 * nlev + 1, L + 1, L + 1), dtype=np.complex128)
    state[0:nlev] = zeta
    state[nlev:2 * nlev] = delta * ((-1.0) ** np.arange(nlev))[:, None, None]
    temp = 190.0 + 3.0 * ((-1.0) ** np.arange(nlev))
    state[2 * nlev:3 * nlev, 0, 0] = temp * math.sqrt(4.0 * math.pi)
    lat, lon, w = vc.gauss_mesh(NLAT, NLON)
    row = vc.state_metrics(state, nlev, sigma, lat, lon, w, top=8)
    np.testing.assert_allclose(row["zeta_adjacent_correlation"], 1.0, atol=1e-12)
    np.testing.assert_allclose(row["delta_adjacent_correlation"], -1.0, atol=1e-12)
    np.testing.assert_allclose(row["global_mean_temperature"], temp, atol=1e-10)
    assert math.isclose(row["alternating_rms_top"], 3.0, rel_tol=1e-10)
    assert math.isclose(row["alternating_mean_top"], 3.0, rel_tol=1e-10)
    json.dumps(row, allow_nan=False)                 # strict JSON, as evidence
    with pytest.raises(ValueError):
        vc.state_metrics(state[:-1], nlev, sigma, lat, lon, w)
