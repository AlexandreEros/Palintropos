"""Correlated-k optics against a line-by-line calculation of a synthetic
spectrum. CPU only.

The reference solves the same two-stream equations monochromatically on the
full wavenumber grid, so the differences measured here are the error of the
k-distribution method (sorting, the correlation assumption across a column
whose line widths change by three orders of magnitude, table interpolation
and band-mean Planck weighting) and of its implementation. They are not a
statement about any real gas: the lines are random Lorentzians.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tropoi.physics.radiation import twostream
from tropoi.physics.radiation.kdist import (C2_CM_K, CorrelatedKLongwave,
                                            CorrelatedKShortwave, KTable,
                                            gauss_legendre_g, ktable_from_spectrum,
                                            planck_band_fraction)
from tropoi.physics.radiation.scheme import (STEFAN_BOLTZMANN, Column, LowerBoundary,
                                             Radiation, heating_rate)
from tropoi.spatial.sigma_coordinate import SigmaGrid

G, CP = 9.8, 1004.0
NU = np.arange(400.0, 1000.0, 0.02)
BANDS = np.linspace(400.0, 1000.0, 7)
MASS_FRACTION = 2.0e-2


def _lines():
    rng = np.random.default_rng(11)
    n = 300
    centre = rng.uniform(410.0, 990.0, n)
    # strongest near 667 cm^-1, four decades of strength
    strength = 10 ** rng.uniform(-3.0, 1.0, n) * np.exp(-((centre - 667.0) / 90.0) ** 2)
    lower_energy = rng.uniform(0.0, 1500.0, n)
    return centre, strength, lower_energy


CENTRE, STRENGTH, LOWER = _lines()


def absorption(p: float, t: float) -> np.ndarray:
    """Synthetic k_nu (m^2/kg): Lorentz lines, pressure broadening with a
    Doppler-like floor, Boltzmann temperature dependence of the strengths."""
    width = math.hypot(0.08 * (p / 1.0e5) * math.sqrt(296.0 / t), 0.01)
    s = STRENGTH * np.exp(-C2_CM_K * LOWER * (1.0 / t - 1.0 / 296.0))
    k = np.zeros_like(NU)
    for c, si in zip(CENTRE, s):
        k += si * (width / math.pi) / ((NU - c) ** 2 + width * width)
    return k


@pytest.fixture(scope="module")
def table() -> KTable:
    return ktable_from_spectrum("synthetic", NU, absorption, np.logspace(0.5, 5.1, 14),
                                np.array([180.0, 220.0, 260.0, 300.0, 340.0]), BANDS, 16,
                                provenance="random Lorentz lines, seed 11")


@pytest.fixture(scope="module")
def atmosphere():
    sig = np.array(SigmaGrid.stretched(30, 1.0e-4, 0.2).interfaces)
    p = (sig * 1.0e5)[:, None]
    mid = 0.5 * (p[1:] + p[:-1])
    t = np.maximum(290.0 * (mid / 1.0e5) ** 0.19, 215.0) + 12.0 * np.exp(-mid / 300.0)
    return Column(p, t, G, CP), 292.0


def _planck_nu(t):
    """pi B_nu d nu on the NU grid (W m^-2 per grid cell), for t (...,)."""
    c1 = 2.0 * 6.62607015e-34 * 2.99792458e8 ** 2          # 2 h c^2
    nu_m = NU * 100.0
    b = c1 * nu_m ** 3 / np.expm1(C2_CM_K * NU / np.asarray(t)[..., None])
    return math.pi * b * 100.0 * (NU[1] - NU[0])


def _line_by_line(column, surface_t):
    """Monochromatic two-stream reference: (up, down, heating)."""
    p = column.pressure_interfaces
    mid, mass = column.layer_pressure[:, 0], column.layer_pressure_thickness[:, 0] / G
    tau = np.stack([absorption(float(pk), float(tk)) for pk, tk in
                    zip(mid, column.temperature[:, 0])]) * (MASS_FRACTION * mass)[:, None]
    planck = _planck_nu(column.temperature[:, 0])
    up, down = twostream.thermal_fluxes(tau, planck, planck,
                                        surface_source=_planck_nu(surface_t),
                                        diffusivity=1.66)
    up, down = up.sum(axis=-1)[:, None], down.sum(axis=-1)[:, None]
    return up, down, heating_rate(down - up, p, G, CP)


def test_planck_band_fractions():
    assert float(planck_band_fraction(1e-6, 1e6, 300.0)) == pytest.approx(1.0, abs=1e-9)
    # a quarter of the flux lies at wavelengths shorter than the Wien peak
    x_peak = C2_CM_K * 1.0e4 / 2897.77
    assert float(planck_band_fraction(1e-6, x_peak * 250.0 / C2_CM_K, 250.0)) == pytest.approx(0.75, abs=2e-3)
    numeric = _planck_nu(np.array(260.0)).sum() / (STEFAN_BOLTZMANN * 260.0 ** 4)
    assert float(planck_band_fraction(400.0, 1000.0, 260.0)) == pytest.approx(numeric, rel=1e-4)
    g, w = gauss_legendre_g(16)
    assert w.sum() == pytest.approx(1.0) and (np.diff(g) > 0).all()


def test_table_reproduces_band_mean_transmission_of_a_homogeneous_path(table):
    """The defining property of a k-distribution: for one (p, T) it gives the
    band-mean transmission of any absorber amount."""
    p, t = table.pressure[9], table.temperature[2]
    k_nu = absorption(p, t)
    for amount in (0.1, 3.0, 100.0):                       # kg m^-2 of absorber
        for ib in range(table.n_band):
            sel = (NU >= BANDS[ib]) & (NU < BANDS[ib + 1])
            exact = np.exp(-k_nu[sel] * amount).mean()
            approx = (table.g_weights * np.exp(-table.k[9, 2, ib] * amount)).sum()
            assert approx == pytest.approx(exact, abs=4e-3)


def test_thermal_fluxes_and_heating_match_line_by_line(table, atmosphere):
    column, surface_t = atmosphere
    lw = CorrelatedKLongwave([table], {"synthetic": MASS_FRACTION}, gravity=G)
    ck = Radiation(lw, source="isothermal").fluxes(column, LowerBoundary(
        temperature=np.full(1, surface_t)))
    up, down, heating = _line_by_line(column, surface_t)
    scale = up[-1, 0]                                      # surface emission in the bands
    # Measured with 16 g-points per band: outgoing flux +0.17 % and surface
    # downward flux -0.25 % of the in-band surface emission; heating within
    # 3.1 % of its peak (16.6 K/day) at the top level and within 0.3 % of
    # the peak below the top three levels. With 32 g-points: 1.5 % / 0.4 %.
    assert abs(ck.lw_up[0, 0] - up[0, 0]) < 4e-3 * scale
    assert abs(ck.lw_down[-1, 0] - down[-1, 0]) < 6e-3 * scale
    error = np.abs(ck.heating - heating) / np.abs(heating).max()
    assert error.max() < 0.05 and error[3:].max() < 0.01
    # the absorber matters: the band is neither transparent nor black
    assert 0.15 < 1.0 - up[0, 0] / scale < 0.9
    assert lw.spectral_coverage(260.0) == pytest.approx(
        float(planck_band_fraction(400.0, 1000.0, 260.0)))


def test_direct_beam_absorption_matches_line_by_line(table, atmosphere):
    column, _ = atmosphere
    sw = CorrelatedKShortwave([table], {"synthetic": MASS_FRACTION}, gravity=G,
                              stellar_temperature=5772.0)
    mu0, flux = 0.6, 100.0
    ck = Radiation(None, sw).fluxes(column, LowerBoundary(temperature=np.full(1, 280.0)),
                                    toa_flux=np.full(1, flux), mu0=np.full(1, mu0))
    mid, mass = column.layer_pressure[:, 0], column.layer_pressure_thickness[:, 0] / G
    tau = np.stack([absorption(float(pk), float(tk)) for pk, tk in
                    zip(mid, column.temperature[:, 0])]) * (MASS_FRACTION * mass)[:, None]
    weight = _planck_nu(np.array(5772.0))
    weight = flux * weight / weight.sum()
    direct = weight * np.exp(-np.concatenate([np.zeros((1, NU.size)),
                                              np.cumsum(tau, axis=0)]) / mu0)
    assert np.abs(ck.sw_down[:, 0] - direct.sum(axis=1)).max() < 1.5e-2 * flux
    assert ck.sw_down[-1, 0] < 0.9 * flux


def test_states_outside_the_table_are_refused_unless_clamping_is_requested(table, atmosphere):
    column, surface_t = atmosphere
    hot = Column(column.pressure_interfaces, column.temperature + 200.0, G, CP)
    lw = CorrelatedKLongwave([table], {"synthetic": MASS_FRACTION}, gravity=G)
    with pytest.raises(ValueError, match="outside the k-table"):
        Radiation(lw).fluxes(hot, LowerBoundary(temperature=np.full(1, surface_t)))
    clamp = CorrelatedKLongwave([table], {"synthetic": MASS_FRACTION}, gravity=G,
                                bounds="clamp")
    Radiation(clamp).fluxes(hot, LowerBoundary(temperature=np.full(1, surface_t)))
    assert clamp.clamped_layers > 0
    with pytest.raises(ValueError):
        CorrelatedKLongwave([table], {}, gravity=G)


def test_table_file_round_trip(table, tmp_path):
    path = tmp_path / "synthetic.npz"
    table.save(path)
    again = KTable.load(path)
    assert again.species == "synthetic" and again.provenance == table.provenance
    assert np.array_equal(again.k, table.k) and np.array_equal(again.band_edges, BANDS)
