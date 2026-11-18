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
    json.dumps(row)                                  # serializable as evidence
    with pytest.raises(ValueError):
        vc.state_metrics(state[:-1], nlev, sigma, lat, lon, w)
