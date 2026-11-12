"""Correlated-k gas optics: the spectral optics provider for real gases.

A gray scheme has one spectral point. A real gas has absorption that varies
by orders of magnitude within any band, and the standard way to integrate
over it at the cost of a few points per band is the k-distribution: within a
band, sort the absorption coefficient ``k`` by value, describe it by its
cumulative distribution ``g`` in [0, 1], and integrate over ``g`` with a
quadrature rule. "Correlated" is the assumption that the ordering in ``g``
is the same at every level of an inhomogeneous column.

This module is everything except the spectroscopy:

* :class:`KTable` — ``k(p, T, band, g)`` for one absorber, with its band
  edges and quadrature weights, and a self-describing ``.npz`` format;
* :func:`ktable_from_spectrum` — builds a table from monochromatic
  absorption coefficients on a wavenumber grid (line-by-line output, a
  cross-section database, or the synthetic spectra of the tests);
* :func:`planck_band_fraction` — the fraction of ``sigma T^4`` in a
  wavenumber interval;
* :class:`CorrelatedKLongwave` / :class:`CorrelatedKShortwave` — optics
  providers for ``scheme.Radiation``.

Tables are interpolated linearly in ``ln p`` and ``T`` on ``ln k``. A state
outside a table's pressure-temperature range is an ERROR by default: an
opacity table is valid only where it was computed, and silently
extrapolating one is how a model ends up "supporting" a regime it has no
data for. ``bounds="clamp"`` holds the edge value instead and counts how
often that happened.

Several absorbers in one band are combined by adding their ``k`` at equal
``g`` (perfectly correlated spectra). That is exact for one absorber and
for pre-mixed tables (the recommended route when the composition is fixed);
for independently varying gases it is an approximation whose error is not
measured here. Random-overlap resorting is not implemented.

No opacity data ship with the package. The method is verified against a
line-by-line calculation of synthetic Lorentz-line spectra
(``tests/test_radiation_kdist.py``); that measures the error of the
k-distribution method and of this implementation, and says nothing about
any real gas.

Import-light: NumPy only; arrays may be NumPy or CuPy.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Mapping, Optional, Union

import numpy as np

from tropoi.physics.radiation.scheme import STEFAN_BOLTZMANN

__all__ = ["KTable", "ktable_from_spectrum", "planck_band_fraction",
           "CorrelatedKLongwave", "CorrelatedKShortwave", "gauss_legendre_g"]

#: hc / k_B in cm K.
C2_CM_K = 1.438776877
KTABLE_SCHEMA = "palintropos.ktable/1"


def _xp(*arrays):
    for a in arrays:
        if type(a).__module__.split(".")[0] == "cupy":
            import cupy  # noqa: WPS433
            return cupy
    return np


def gauss_legendre_g(n: int, split: float = 0.9) -> tuple[np.ndarray, np.ndarray]:
    """Abscissas and weights on g in [0, 1]: half the points on ``[0,
    split]`` and half on ``[split, 1]``, so the steep top of the distribution
    (line centres) gets as many points as the rest."""
    if n < 2 or n % 2:
        raise ValueError("n must be an even number >= 2")
    x, w = np.polynomial.legendre.leggauss(n // 2)
    lo_g, lo_w = 0.5 * split * (x + 1.0), 0.5 * split * w
    hi_g, hi_w = split + 0.5 * (1.0 - split) * (x + 1.0), 0.5 * (1.0 - split) * w
    return np.concatenate([lo_g, hi_g]), np.concatenate([lo_w, hi_w])


def _planck_cumulative(x):
    """Fraction of ``sigma T^4`` at wavenumbers below ``x = c2 nu / T``:
    ``1 - (15 / pi^4) sum_n e^(-n x) (x^3/n + 3 x^2/n^2 + 6 x/n^3 + 6/n^4)``,
    with the small-x power series below 0.5 where the sum converges slowly."""
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    small = x < 0.5
    xs = x[small]
    out[small] = (15.0 / math.pi ** 4) * xs ** 3 * (
        1.0 / 3.0 - xs / 8.0 + xs ** 2 / 60.0 - xs ** 4 / 5040.0 + xs ** 6 / 272160.0)
    xl = x[~small]
    total = np.zeros_like(xl)
    for n in range(1, 60):
        total += np.exp(-n * xl) * (xl ** 3 / n + 3 * xl ** 2 / n ** 2 + 6 * xl / n ** 3
                                    + 6 / n ** 4)
    out[~small] = 1.0 - (15.0 / math.pi ** 4) * total
    return out


def planck_band_fraction(nu_low, nu_high, temperature):
    """Fraction of blackbody flux at ``temperature`` (K) between two
    wavenumbers (cm^-1). Arguments broadcast (NumPy, host side)."""
    t = np.asarray(temperature, dtype=np.float64)
    return (_planck_cumulative(C2_CM_K * np.asarray(nu_high) / t)
            - _planck_cumulative(C2_CM_K * np.asarray(nu_low) / t))


@dataclass
class KTable:
    """k-coefficients of one absorber.

    ``k`` has shape (n_p, n_T, n_band, n_g), in m^2 per kg of the absorber;
    ``pressure`` (Pa) and ``temperature`` (K) are increasing; ``band_edges``
    (n_band + 1,) in cm^-1, increasing; ``g_weights`` (n_g,) sum to 1.
    """

    species: str
    pressure: np.ndarray
    temperature: np.ndarray
    band_edges: np.ndarray
    g_weights: np.ndarray
    k: np.ndarray
    provenance: str = ""

    def __post_init__(self):
        self.pressure = np.asarray(self.pressure, dtype=np.float64)
        self.temperature = np.asarray(self.temperature, dtype=np.float64)
        self.band_edges = np.asarray(self.band_edges, dtype=np.float64)
        self.g_weights = np.asarray(self.g_weights, dtype=np.float64)
        self.k = np.asarray(self.k, dtype=np.float64)
        shape = (self.pressure.size, self.temperature.size, self.band_edges.size - 1,
                 self.g_weights.size)
        if self.k.shape != shape:
            raise ValueError(f"k has shape {self.k.shape}, expected {shape}")
        for name in ("pressure", "temperature", "band_edges"):
            a = getattr(self, name)
            if a.size < 2 or np.any(np.diff(a) <= 0) or a[0] <= 0:
                raise ValueError(f"{name} must be positive and strictly increasing")
        if abs(self.g_weights.sum() - 1.0) > 1e-10 or np.any(self.g_weights <= 0):
            raise ValueError("g_weights must be positive and sum to 1")
        if not np.all(np.isfinite(self.k)) or np.any(self.k < 0):
            raise ValueError("k must be finite and >= 0")

    @property
    def n_band(self) -> int:
        return self.band_edges.size - 1

    @property
    def n_g(self) -> int:
        return self.g_weights.size

    def save(self, path) -> None:
        np.savez_compressed(path, schema=KTABLE_SCHEMA, species=self.species,
                            pressure=self.pressure, temperature=self.temperature,
                            band_edges=self.band_edges, g_weights=self.g_weights, k=self.k,
                            provenance=self.provenance)

    @classmethod
    def load(cls, path) -> "KTable":
        with np.load(path, allow_pickle=False) as d:
            if str(d["schema"]) != KTABLE_SCHEMA:
                raise ValueError(f"{path}: not a {KTABLE_SCHEMA} file")
            return cls(str(d["species"]), d["pressure"], d["temperature"], d["band_edges"],
                       d["g_weights"], d["k"], str(d["provenance"]))


def ktable_from_spectrum(species: str, wavenumber, absorption: Callable, pressure,
                         temperature, band_edges, n_g: int = 16, *,
                         provenance: str = "") -> KTable:
    """Build a :class:`KTable` from monochromatic absorption coefficients.

    ``wavenumber`` is a uniform grid (cm^-1); ``absorption(p, T)`` returns
    ``k_nu`` (m^2/kg) on it. For every (p, T, band) the coefficients are
    sorted and the sorted curve — the inverse cumulative distribution
    ``k(g)`` — is sampled at the quadrature abscissas.
    """
    nu = np.asarray(wavenumber, dtype=np.float64)
    g, w = gauss_legendre_g(n_g)
    edges = np.asarray(band_edges, dtype=np.float64)
    pressure = np.asarray(pressure, dtype=np.float64)
    temperature = np.asarray(temperature, dtype=np.float64)
    k = np.empty((pressure.size, temperature.size, edges.size - 1, n_g))
    for ip, p in enumerate(pressure):
        for it, t in enumerate(temperature):
            k_nu = np.asarray(absorption(float(p), float(t)))
            for ib in range(edges.size - 1):
                sel = np.sort(k_nu[(nu >= edges[ib]) & (nu < edges[ib + 1])])
                if sel.size < 4 * n_g:
                    raise ValueError(f"band {ib} holds only {sel.size} spectral samples")
                centres = (np.arange(sel.size) + 0.5) / sel.size
                k[ip, it, ib] = np.interp(g, centres, sel)
    return KTable(species, pressure, temperature, edges, w, k, provenance)


class _CorrelatedK:
    """Shared table handling for the thermal and stellar providers."""

    def __init__(self, tables, mass_fractions: Mapping[str, Union[float, Callable]], *,
                 gravity: float, bounds: str = "error"):
        tables = list(tables)
        if not tables:
            raise ValueError("at least one KTable is needed")
        first = tables[0]
        for t in tables[1:]:
            if not (np.array_equal(t.band_edges, first.band_edges)
                    and np.array_equal(t.g_weights, first.g_weights)):
                raise ValueError("all tables must share band edges and g weights")
        missing = [t.species for t in tables if t.species not in mass_fractions]
        if missing:
            raise ValueError(f"no mass fraction given for {missing}")
        if bounds not in ("error", "clamp"):
            raise ValueError("bounds must be 'error' or 'clamp'")
        self.tables = tables
        self.mass_fractions = dict(mass_fractions)
        self.gravity = float(gravity)
        self.bounds = bounds
        self.band_edges = first.band_edges
        self.n_band, self.n_g = first.n_band, first.n_g
        self.n_points = self.n_band * self.n_g
        self._g_weights = np.tile(first.g_weights, self.n_band)        # per point
        self._band_of_point = np.repeat(np.arange(self.n_band), self.n_g)
        self.clamped_layers = 0
        self._dev: dict = {}

    def _signature(self, kind: str) -> dict:
        return {"kind": kind, "species": [t.species for t in self.tables],
                "provenance": [t.provenance for t in self.tables],
                "band_edges": [float(x) for x in self.band_edges], "n_g": self.n_g,
                "mass_fractions": {k: (v if isinstance(v, float) else "profile")
                                   for k, v in self.mass_fractions.items()},
                "gravity": self.gravity, "bounds": self.bounds}

    def _table_dev(self, xp, i: int):
        key = (xp.__name__, i)
        d = self._dev.get(key)
        if d is None:
            t = self.tables[i]
            lnk = np.log(np.maximum(t.k.reshape(t.pressure.size, t.temperature.size, -1),
                                    1e-300))
            d = (xp.asarray(np.log(t.pressure)), xp.asarray(t.temperature), xp.asarray(lnk))
            self._dev[key] = d
        return d

    def _layer_k(self, xp, i: int, ln_p, temperature, points):
        """Interpolated k (K, ncol, n) of table ``i`` at the layer states."""
        ln_pg, tg, lnk = self._table_dev(xp, i)
        table = self.tables[i]
        outside = ((ln_p < ln_pg[0]) | (ln_p > ln_pg[-1])
                   | (temperature < tg[0]) | (temperature > tg[-1]))
        n_out = int(outside.sum())
        if n_out:
            if self.bounds == "error":
                raise ValueError(
                    f"{n_out} layer states lie outside the k-table of {table.species!r} "
                    f"(valid: p in [{table.pressure[0]:.3g}, {table.pressure[-1]:.3g}] Pa, "
                    f"T in [{table.temperature[0]:.4g}, {table.temperature[-1]:.4g}] K; "
                    f"asked: p in [{float(xp.exp(ln_p.min())):.3g}, "
                    f"{float(xp.exp(ln_p.max())):.3g}] Pa, T in "
                    f"[{float(temperature.min()):.4g}, {float(temperature.max()):.4g}] K). "
                    "Extend the table or pass bounds='clamp' to hold the edge value.")
            self.clamped_layers += n_out
        lp = xp.clip(ln_p, ln_pg[0], ln_pg[-1])
        tt = xp.clip(temperature, tg[0], tg[-1])
        ip = xp.clip(xp.searchsorted(ln_pg, lp.ravel()).reshape(lp.shape) - 1, 0,
                     ln_pg.size - 2)
        it = xp.clip(xp.searchsorted(tg, tt.ravel()).reshape(tt.shape) - 1, 0, tg.size - 2)
        wp = ((lp - ln_pg[ip]) / (ln_pg[ip + 1] - ln_pg[ip]))[..., None]
        wt = ((tt - tg[it]) / (tg[it + 1] - tg[it]))[..., None]
        sel = lnk[:, :, points]
        out = ((1 - wp) * (1 - wt) * sel[ip, it] + wp * (1 - wt) * sel[ip + 1, it]
               + (1 - wp) * wt * sel[ip, it + 1] + wp * wt * sel[ip + 1, it + 1])
        return xp.exp(out)

    def _tau(self, column, points):
        xp = _xp(column.temperature)
        p_mid = column.layer_pressure
        ln_p = xp.log(p_mid)
        mass = column.layer_pressure_thickness / self.gravity          # kg m^-2
        tau = 0.0
        for i, table in enumerate(self.tables):
            q = self.mass_fractions[table.species]
            q = q(p_mid) if callable(q) else q
            tau = tau + self._layer_k(xp, i, ln_p, column.temperature, points) \
                * (q * mass)[..., None]
        return tau


class CorrelatedKLongwave(_CorrelatedK):
    """Thermal correlated-k optics (absorption and emission; add scattering
    by subclassing ``optical_properties``)."""

    def __init__(self, tables, mass_fractions, *, gravity: float, bounds: str = "error",
                 diffusivity: float = 1.66,
                 temperature_range: tuple[float, float] = (20.0, 3000.0)):
        super().__init__(tables, mass_fractions, gravity=gravity, bounds=bounds)
        self.diffusivity = float(diffusivity)
        # Band Planck fractions on a 1 K grid, interpolated linearly at run time.
        self._t_grid = np.arange(temperature_range[0], temperature_range[1] + 1.0, 1.0)
        frac = planck_band_fraction(self.band_edges[:-1][None, :], self.band_edges[1:][None, :],
                                    self._t_grid[:, None])
        self._fraction = frac[:, self._band_of_point] * self._g_weights[None, :]

    def signature(self) -> dict:
        return {**self._signature("correlated_k_longwave"), "diffusivity": self.diffusivity}

    def optical_properties(self, column, points):
        return self._tau(column, points), None, None

    def planck(self, temperature, points):
        xp = _xp(temperature)
        key = (xp.__name__, "planck")
        table = self._dev.get(key)
        if table is None:
            table = xp.asarray(self._fraction)
            self._dev[key] = table
        t0 = float(self._t_grid[0])
        x = xp.clip(temperature - t0, 0.0, self._t_grid.size - 1.000001)
        i = x.astype(np.int64)
        w = (x - i)[..., None]
        table = table[:, points]           # spectral slice first: bounded by the chunk
        frac = (1.0 - w) * table[i] + w * table[i + 1]
        return frac * (STEFAN_BOLTZMANN * temperature ** 4)[..., None]

    def spectral_coverage(self, temperature: float) -> float:
        """Fraction of ``sigma T^4`` inside the tabulated bands."""
        return float(planck_band_fraction(self.band_edges[0], self.band_edges[-1], temperature))


class CorrelatedKShortwave(_CorrelatedK):
    """Stellar correlated-k optics. The stellar spectrum is a blackbody at
    ``stellar_temperature`` unless ``band_fractions`` (n_band,) is given;
    fractions are renormalized over the tabulated bands. Absorption only
    (``ssa = 0``): Rayleigh scattering is not included yet."""

    closure = "quadrature"
    delta_scale = False
    diffuse_incidence = False

    def __init__(self, tables, mass_fractions, *, gravity: float, bounds: str = "error",
                 stellar_temperature: float = 5772.0,
                 band_fractions: Optional[np.ndarray] = None):
        super().__init__(tables, mass_fractions, gravity=gravity, bounds=bounds)
        if band_fractions is None:
            band_fractions = planck_band_fraction(self.band_edges[:-1], self.band_edges[1:],
                                                  stellar_temperature)
        band_fractions = np.asarray(band_fractions, dtype=np.float64)
        if band_fractions.shape != (self.n_band,) or band_fractions.sum() <= 0:
            raise ValueError("band_fractions must be (n_band,) with a positive sum")
        self.stellar_temperature = float(stellar_temperature)
        self._fractions = (band_fractions / band_fractions.sum())[self._band_of_point] \
            * self._g_weights

    def signature(self) -> dict:
        return {**self._signature("correlated_k_shortwave"),
                "stellar_temperature": self.stellar_temperature}

    def stellar_fractions(self):
        return self._fractions

    def optical_properties(self, column, points):
        tau = self._tau(column, points)
        return tau, 0.0 * tau, 0.0 * tau
