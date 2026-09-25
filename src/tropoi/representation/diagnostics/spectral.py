"""Host-side spectral kinetic-energy diagnostics of saved states (NumPy only).

These numbers are computed from stored coefficients at saved snapshots. They
are never per-step measurements, and callers that plot them must say so.

Kinetic-energy modes
--------------------
The horizontal velocity is split the way the solvers carry it: a rotational
part (streamfunction) plus a divergent part (velocity potential), i.e. the
vector-spherical-harmonic split. Raw lat-lon ``u`` and ``v`` are never
treated as two independent scalar fields. With
``psi_lm = -R^2 zeta_lm / (l(l+1))`` and ``chi_lm = -R^2 delta_lm / (l(l+1))``,
and the two parts orthogonal in the energy integral over a closed surface,

    E(l, m) = (R^4 / 2) * [P_zeta(l, m) + P_delta(l, m)] / (l (l + 1)),  l >= 1
    E(0, 0) = 0                  (the l = 0 mode carries no velocity)

``P`` is the repository's mode power for the ``m >= 0`` storage layout:
``Re(c)^2`` for ``m = 0`` and ``2 |c|^2`` for ``m > 0``, so each stored mode
carries the power of its full ``+/-m`` pair. Positions ``m > l`` are padding
and contribute nothing. ``sum E`` equals ``1/2 * integral |u|^2 dA``: the
area-integrated *specific* kinetic energy in m^4 s^-2, NOT weighted by layer
depth or air mass. For the barotropic vorticity equation ``delta`` is absent.

Spectral complexity
-------------------
With ``p(l, m) = E(l, m) / sum E`` (nonnegative, sums to 1):

    <l>   = sum l p          power-weighted mean degree
    <|m|> = sum m p          power-weighted mean zonal wavenumber
    S     = -sum p ln p      Shannon entropy of p; zero-power modes contribute 0
    N_eff = exp(S)           effective number of occupied modes

``S`` is the Shannon entropy of a modal energy distribution, not a
thermodynamic entropy. ``R`` cancels in ``p``, so the measures do not depend
on the planetary radius. They are undefined for a state with no kinetic
energy (:class:`ZeroKineticEnergyError`).

These are the definitions of the Williamson-5 T63 report
(docs/validation/williamson5_mri_2026-07-30.md, section 4.1), moved verbatim
from ``docs/validation/williamson_5/plot_swe_holistic.py``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "SpectralComplexity",
    "ZeroKineticEnergyError",
    "kinetic_energy_modes",
    "mode_power",
    "spectral_complexity",
]


class ZeroKineticEnergyError(ValueError):
    """The modal distribution is undefined because the state has no motion."""


@dataclass(frozen=True)
class SpectralComplexity:
    """Complexity measures of one saved state's kinetic-energy distribution."""

    mean_degree: float
    mean_order: float
    entropy: float
    effective_modes: float

    def as_dict(self) -> dict[str, float]:
        return {"mean_degree": self.mean_degree,
                "mean_order": self.mean_order,
                "entropy": self.entropy,
                "effective_modes": self.effective_modes}


def _coefficients(values, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError(
            f"{name} must be one (l, m) coefficient array with equal axes, "
            f"got shape {array.shape}")
    if not np.iscomplexobj(array):
        raise TypeError(f"{name} coefficients must be complex-valued")
    return array


def mode_power(coefficients) -> np.ndarray:
    """Per-(l, m) contribution to ``integral |f|^2 dOmega`` (m >= 0 layout)."""
    coeffs = _coefficients(coefficients, "coefficients")
    degree, order = np.indices(coeffs.shape)
    power = np.where(order == 0, coeffs.real ** 2, 2.0 * np.abs(coeffs) ** 2)
    return np.where(order <= degree, power, 0.0)


def kinetic_energy_modes(zeta_lm, delta_lm=None, *,
                         radius: float) -> np.ndarray:
    """Modal kinetic energy ``E(l, m)`` [m^4 s^-2] of one saved state.

    ``zeta_lm`` / ``delta_lm`` are the relative-vorticity and divergence
    coefficients [s^-1] of one horizontal level; ``delta_lm=None`` means a
    non-divergent flow (BVE). ``radius`` is the planetary radius in metres.
    """
    radius = float(radius)
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("radius must be finite and positive")
    power = mode_power(zeta_lm)
    if delta_lm is not None:
        delta_power = mode_power(delta_lm)
        if delta_power.shape != power.shape:
            raise ValueError("zeta and delta coefficients must share a shape")
        power = power + delta_power
    degree = np.arange(power.shape[0], dtype=np.float64)
    # 1/(l(l+1)) with the l = 0 row left at zero: that mode carries no
    # velocity, so it contributes exactly zero energy instead of 0/0.
    inverse_laplacian = np.zeros_like(degree)
    inverse_laplacian[1:] = 1.0 / (degree[1:] * (degree[1:] + 1.0))
    energy = 0.5 * radius ** 4 * inverse_laplacian[:, None] * power
    if not np.isfinite(energy).all():
        raise ValueError("modal kinetic energy is not finite")
    return energy


def spectral_complexity(modal_energy) -> SpectralComplexity:
    """Mean degree, mean zonal wavenumber, entropy and ``N_eff`` of ``E``."""
    energy = np.asarray(modal_energy, dtype=np.float64)
    if energy.ndim != 2 or energy.shape[0] != energy.shape[1]:
        raise ValueError("modal energy must be an (l, m) array")
    if np.any(energy < 0.0) or not np.isfinite(energy).all():
        raise ValueError("modal energy must be finite and nonnegative")
    total = float(energy.sum())
    if total == 0.0:
        raise ZeroKineticEnergyError(
            "spectral complexity is undefined for a state with zero kinetic "
            "energy")
    probability = energy / total
    degree, order = np.indices(energy.shape)
    # Zero-power modes contribute exactly zero to -sum p ln p; excluding them
    # avoids 0*log(0). The +0.0 turns the -0.0 of a single occupied mode
    # into a plain 0.0.
    occupied = probability[probability > 0.0]
    entropy = float(-(occupied * np.log(occupied)).sum()) + 0.0
    return SpectralComplexity(
        mean_degree=float((degree * probability).sum()),
        mean_order=float((order * probability).sum()),
        entropy=entropy,
        effective_modes=float(np.exp(entropy)))
