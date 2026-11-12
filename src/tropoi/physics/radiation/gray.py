"""Gray and semi-gray optics: prescribed optical depth as a function of pressure.

These are the one-spectral-point optics providers of ``scheme.py``. They are
the radiation of several published idealized models, reproduced here from
the papers' stated formulas:

* Frierson, Held & Zurita-Gotor (2006), eqs. 4-5 (:func:`frierson_longwave`);
* Schneider & Liu (2009), appendix B (:func:`schneider_liu_longwave`,
  :class:`SemiInfiniteScatteringShortwave`);
* Thomson & Vallis (2019), eqs. 14-15, the intermediate Mars model
  (:func:`isca_mars_longwave`, :func:`isca_mars_shortwave`).

A gray atmosphere is an idealization with no spectroscopic content: the
optical depths are tuned by their authors to a planet's bulk energy balance.
Agreement of this code with those models establishes that the same equations
are solved, not that the radiation of the planet is represented.

Import-light: NumPy only; arrays may be NumPy or CuPy.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np

from tropoi.physics.radiation.scheme import STEFAN_BOLTZMANN

__all__ = ["OpticalDepthProfile", "GrayLongwave", "GrayShortwave",
           "SemiInfiniteScatteringShortwave", "frierson_longwave",
           "schneider_liu_longwave", "isca_mars_longwave", "isca_mars_shortwave"]


def _xp(*arrays):
    for a in arrays:
        if type(a).__module__.split(".")[0] == "cupy":
            import cupy  # noqa: WPS433
            return cupy
    return np


class OpticalDepthProfile:
    """Cumulative optical depth from the top:

        tau(p) = tau_ref(lat) * sum_i a_i (p / p_ref)^(n_i),

    with ``tau_ref(lat) = equator + (pole - equator) sin^2(lat)`` and
    ``p_ref`` either a fixed pressure (Pa) or, with ``reference="surface"``,
    the local pressure of the lowest interface. ``terms`` is a sequence of
    ``(a_i, n_i)``. Layer optical depths are exact differences of this
    function at the layer edges, so they sum to the column value on any
    vertical grid.
    """

    def __init__(self, terms: Sequence[tuple[float, float]], *, equator: float,
                 pole: Optional[float] = None, reference="surface"):
        self.terms = tuple((float(a), float(n)) for a, n in terms)
        if not self.terms or any(n <= 0 for _, n in self.terms):
            raise ValueError("terms must be a non-empty sequence of (a, n) with n > 0")
        self.equator = float(equator)
        self.pole = self.equator if pole is None else float(pole)
        if reference != "surface":
            reference = float(reference)
            if not (math.isfinite(reference) and reference > 0):
                raise ValueError("a fixed reference pressure must be finite and > 0")
        self.reference = reference

    def signature(self) -> dict:
        return {"terms": [list(t) for t in self.terms], "equator": self.equator,
                "pole": self.pole, "reference": self.reference}

    def cumulative(self, column):
        """tau at the K+1 interfaces, (K+1, ncol)."""
        xp = _xp(column.pressure_interfaces)
        p = column.pressure_interfaces
        p_ref = p[-1] if self.reference == "surface" else self.reference
        x = p / p_ref
        shape = sum(a * x ** n for a, n in self.terms)
        if self.pole == self.equator:
            return self.equator * shape
        if column.latitude is None:
            raise ValueError("a latitude-dependent optical depth needs column.latitude")
        s = xp.sin(column.latitude)
        return (self.equator + (self.pole - self.equator) * s * s) * shape

    def layers(self, column):
        tau = self.cumulative(column)
        return tau[1:] - tau[:-1]


class GrayLongwave:
    """Gray thermal absorption/emission: one spectral point carrying
    ``sigma T^4``. ``diffusivity`` multiplies the optical depth in the
    two-stream closure; papers that fold it into their stated optical depth
    (Frierson et al. 2006) correspond to ``diffusivity=1``."""

    n_points = 1

    def __init__(self, profile: OpticalDepthProfile, *, diffusivity: float = 1.0):
        self.profile = profile
        self.diffusivity = float(diffusivity)

    def signature(self) -> dict:
        return {"kind": "gray_longwave", "profile": self.profile.signature(),
                "diffusivity": self.diffusivity}

    def optical_properties(self, column, points):
        return self.profile.layers(column)[..., None], None, None

    def planck(self, temperature, points):
        return (STEFAN_BOLTZMANN * temperature ** 4)[..., None]


class GrayShortwave:
    """Gray stellar extinction with optional scattering (two-stream solve)."""

    n_points = 1

    def __init__(self, profile: OpticalDepthProfile, *, ssa: float = 0.0,
                 asym: float = 0.0, closure: str = "quadrature",
                 delta_scale: bool = False, diffuse_incidence: bool = False):
        self.profile = profile
        self.ssa = float(ssa)
        self.asym = float(asym)
        self.closure = closure
        self.delta_scale = bool(delta_scale)
        self.diffuse_incidence = bool(diffuse_incidence)

    def signature(self) -> dict:
        return {"kind": "gray_shortwave", "profile": self.profile.signature(),
                "ssa": self.ssa, "asym": self.asym, "closure": self.closure,
                "delta_scale": self.delta_scale,
                "diffuse_incidence": self.diffuse_incidence}

    def stellar_fractions(self):
        return np.ones(1)

    def optical_properties(self, column, points):
        tau = self.profile.layers(column)[..., None]
        return tau, self.ssa + 0.0 * tau, self.asym + 0.0 * tau


class SemiInfiniteScatteringShortwave:
    """Schneider & Liu (2009) stellar flux: diffuse incidence on a
    semi-infinite gray absorbing and scattering atmosphere,

        F(tau) = F_toa (1 - r_inf) exp(-Gamma tau),
        Gamma = 2 sqrt(1 - w) sqrt(1 - w g),
        r_inf = [sqrt(1 - w g) - sqrt(1 - w)] / [sqrt(1 - w g) + sqrt(1 - w)],

    the closed form of the hemispheric-mean two-stream equations (verified
    against the general solver in the tests). ``F`` is the net downward
    flux; it is split here as ``down = F / (1 - r_inf)``, ``up = r_inf
    down``, which is the two-stream solution it comes from.
    """

    def __init__(self, profile: OpticalDepthProfile, *, ssa: float, asym: float):
        self.profile = profile
        self.ssa = float(ssa)
        self.asym = float(asym)
        a = math.sqrt(1.0 - self.ssa)
        b = math.sqrt(1.0 - self.ssa * self.asym)
        self.attenuation = 2.0 * a * b
        self.bond_albedo = (b - a) / (b + a)

    def signature(self) -> dict:
        return {"kind": "semi_infinite_scattering_shortwave",
                "profile": self.profile.signature(), "ssa": self.ssa, "asym": self.asym}

    def analytic_fluxes(self, column, toa_flux, mu0, albedo):
        xp = _xp(column.pressure_interfaces)
        down = toa_flux[None, :] * xp.exp(-self.attenuation * self.profile.cumulative(column))
        return self.bond_albedo * down, down, 0.0 * down


def frierson_longwave(tau_equator: float = 6.0, tau_pole: float = 1.5,
                      linear_fraction: float = 0.1) -> GrayLongwave:
    """FHZ06 eqs. 4-5: ``tau = tau_0(lat) [f_l s + (1 - f_l) s^4]``, ``s = p /
    p_s``, diffusivity folded into tau (their eqs. 6-7)."""
    f = float(linear_fraction)
    return GrayLongwave(OpticalDepthProfile([(f, 1.0), (1.0 - f, 4.0)], equator=tau_equator,
                                            pole=tau_pole, reference="surface"),
                        diffusivity=1.0)


def schneider_liu_longwave(tau0: float = 80.0, p0: float = 3.0e5) -> GrayLongwave:
    """SL09 appendix B: ``tau_l = tau_l0 (p / p0)^2`` (collision-induced
    absorption scales with density squared)."""
    return GrayLongwave(OpticalDepthProfile([(1.0, 2.0)], equator=tau0, reference=p0),
                        diffusivity=1.0)


def isca_mars_longwave(tau0: float = 0.2) -> GrayLongwave:
    """Thomson & Vallis (2019) eq. 14: ``tau = tau0 exp(-z / H)``, i.e. linear
    in ``p / p_s`` for their constant scale height."""
    return GrayLongwave(OpticalDepthProfile([(1.0, 1.0)], equator=tau0, reference="surface"),
                        diffusivity=1.0)


def isca_mars_shortwave(tau0: float = 0.2) -> GrayShortwave:
    """Thomson & Vallis (2019) eq. 15: ``tau = tau0 exp(-4 z / H)``, absorbing."""
    return GrayShortwave(OpticalDepthProfile([(1.0, 4.0)], equator=tau0, reference="surface"))
