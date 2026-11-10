"""Dry convective adjustment (no moisture, no clouds).

Radiation alone drives the lower part of most atmospheres to a
super-adiabatic lapse rate. Without a model of the convection that removes
it, a hydrostatic core either runs with an unstable column or produces
grid-scale overturning. The adjustment here restores neutrality while
conserving the column enthalpy ``sum c_p T dp / g`` exactly.

The neutral profile is not found by repeated pairwise mixing (which
converges slowly through deep unstable layers) but in closed form. With
potential temperature ``theta_k = T_k / Pi_k``, ``Pi_k = (p_k / p_ref)^kappa``,
mixing a set of layers to a common theta while conserving enthalpy gives
their mean of theta weighted by ``w_k = Pi_k dp_k``. Requiring theta to be
non-decreasing with height then makes the adjusted profile the weighted
*isotonic regression* of theta, whose solution is the min-max formula

    theta_adj[k] = max_{i <= k} min_{j >= k} mean_w(theta[i..j])

(indices running upward from the bottom). It is evaluated for all columns at
once from cumulative sums, costs O(K^2) memory per column and no iteration.

Constant ``kappa`` is assumed (as in the dynamical core).

Import-light: NumPy only; arrays may be NumPy or CuPy.
"""
from __future__ import annotations

import math

import numpy as np

__all__ = ["neutral_profile", "DryConvectiveAdjustment"]


def _xp(*arrays):
    for a in arrays:
        if type(a).__module__.split(".")[0] == "cupy":
            import cupy  # noqa: WPS433
            return cupy
    return np


def neutral_profile(temperature, layer_pressure, layer_thickness, kappa: float,
                    *, lapse_fraction: float = 1.0):
    """The enthalpy-conserving adjusted temperature, (K, ncol), top first.

    ``lapse_fraction`` scales the target lapse rate: columns are adjusted to
    ``d ln T / d ln p = lapse_fraction * kappa`` wherever they are steeper
    (1 = dry adiabat).
    """
    xp = _xp(temperature)
    k_eff = float(kappa) * float(lapse_fraction)
    # Work bottom-up so the isotonic fit is non-decreasing along axis 0.
    exner = (layer_pressure[::-1] / layer_pressure[-1:]) ** k_eff
    w = exner * layer_thickness[::-1]
    theta = temperature[::-1] / exner
    K = theta.shape[0]
    zero = xp.zeros_like(theta[:1])
    cw = xp.concatenate([zero, xp.cumsum(w, axis=0)])              # (K+1, ncol)
    cs = xp.concatenate([zero, xp.cumsum(w * theta, axis=0)])
    # mean[i, j] = weighted mean of theta[i..j] (valid for j >= i)
    num = cs[None, 1:] - cs[:-1, None]
    den = cw[None, 1:] - cw[:-1, None]
    i_idx = xp.arange(K)[:, None, None]
    j_idx = xp.arange(K)[None, :, None]
    valid = j_idx >= i_idx
    mean = xp.where(valid, num / xp.where(valid, den, 1.0), math.inf)
    # m[i, k] = min over j >= k of mean[i, j]: reverse running minimum along j.
    # Explicit loops over the K levels (CuPy has no ufunc.accumulate for
    # minimum/maximum); each pass is one vectorized operation over columns.
    m = mean
    for j in range(K - 2, -1, -1):
        m[:, j] = xp.minimum(m[:, j], m[:, j + 1])
    m = xp.where(valid, m, -math.inf)          # only i <= k may contribute
    theta_adj = xp.empty_like(theta)
    running = m[0]
    theta_adj[0] = running[0]
    for i in range(1, K):
        running = xp.maximum(running, m[i])
        theta_adj[i] = running[i]
    return (theta_adj * exner)[::-1]


class DryConvectiveAdjustment:
    """Relax every column toward its neutral profile.

    ``timescale = 0`` is the instantaneous (hard) adjustment; a positive
    timescale relaxes with ``1 - exp(-dt / timescale)`` per call, the
    quasi-equilibrium form used by Schneider & Liu (2009) with 6 h. Enthalpy
    is conserved for any timescale because the target conserves it.
    """

    def __init__(self, kappa: float, *, timescale: float = 0.0,
                 lapse_fraction: float = 1.0):
        if not (0.0 < kappa < 1.0):
            raise ValueError(f"kappa must be in (0, 1), got {kappa}")
        if timescale < 0.0:
            raise ValueError("timescale must be >= 0")
        self.kappa = float(kappa)
        self.timescale = float(timescale)
        self.lapse_fraction = float(lapse_fraction)

    def signature(self) -> dict:
        return {"kind": "dry_convective_adjustment", "kappa": self.kappa,
                "timescale": self.timescale, "lapse_fraction": self.lapse_fraction}

    def adjust(self, temperature, layer_pressure, layer_thickness, dt: float):
        """Return the temperature after ``dt`` seconds of adjustment."""
        target = neutral_profile(temperature, layer_pressure, layer_thickness,
                                 self.kappa, lapse_fraction=self.lapse_fraction)
        if self.timescale == 0.0:
            return target
        frac = -math.expm1(-float(dt) / self.timescale)
        return temperature + frac * (target - temperature)
