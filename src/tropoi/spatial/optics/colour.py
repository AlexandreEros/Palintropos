"""CIE 1931 2-degree colour-matching functions and sRGB conversion.

``data/CIE_xyz_1931_2deg.csv`` is the CIE dataset "CIE 1931 colour-matching
functions, 2 degree observer" (CIE 2019, DOI 10.25039/CIE.DS.xvudnb9b),
copied byte for byte from
https://files.cie.co.at/Publications-datasets/CIE_xyz_1931_2deg.csv
(360-830 nm, 1 nm; SHA-256 ``CIE_SHA256``, checked on load).

:func:`bin_weights` integrates the functions over spectral bins (1 nm
rectangle rule over the bin) so a binned spectral radiance turns into XYZ.
The XYZ -> linear sRGB matrix and the transfer function are IEC 61966-2-1
(sRGB, D65 white).
"""
from __future__ import annotations

import hashlib
import importlib.resources

import numpy as np

__all__ = ["CIE_SHA256", "XYZ_TO_LINEAR_SRGB", "bin_weights", "cie_1931",
           "srgb_encode"]

CIE_SHA256 = "fa663e3535a7e0763a745993a1f0a192eb0275ac46ad2d1befd7626841e713c1"
CIE_SOURCE = ("CIE 1931 2° colour-matching functions, CIE dataset DOI "
              "10.25039/CIE.DS.xvudnb9b (1 nm, verbatim copy)")

XYZ_TO_LINEAR_SRGB = np.array([[3.2406, -1.5372, -0.4986],
                               [-0.9689, 1.8758, 0.0415],
                               [0.0557, -0.2040, 1.0570]])

_CACHE: dict = {}


def cie_1931() -> tuple[np.ndarray, np.ndarray]:
    """(wavelength_nm, cmf[:, xyz]) from the packaged CIE table."""
    if "cmf" not in _CACHE:
        raw = importlib.resources.files("tropoi.spatial.optics").joinpath(
            "data/CIE_xyz_1931_2deg.csv").read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != CIE_SHA256:
            raise RuntimeError(f"packaged CIE table changed: {digest}")
        rows = np.array([[float(v) for v in line.split(",")]
                         for line in raw.decode("ascii").split("\n")
                         if line.strip()])
        _CACHE["cmf"] = (rows[:, 0], rows[:, 1:4])
    return _CACHE["cmf"]


def bin_weights(lo_nm, hi_nm) -> np.ndarray:
    """XYZ weights (nm) of bins [lo, hi): sum of 1-nm CMF rows in the bin."""
    lam, cmf = cie_1931()
    out = []
    for lo, hi in zip(np.atleast_1d(lo_nm), np.atleast_1d(hi_nm)):
        mask = (lam >= lo) & (lam < hi)
        out.append(cmf[mask].sum(axis=0))
    return np.array(out)


def omitted_fraction(lo_nm: float, hi_nm: float) -> np.ndarray:
    """Fraction of each CMF's integral (360-830 nm) outside [lo, hi)."""
    lam, cmf = cie_1931()
    inside = cmf[(lam >= lo_nm) & (lam < hi_nm)].sum(axis=0)
    return 1.0 - inside / cmf.sum(axis=0)


def srgb_encode(linear) -> np.ndarray:
    """IEC 61966-2-1 transfer function on values clipped to [0, 1]."""
    x = np.clip(np.asarray(linear, dtype=np.float64), 0.0, 1.0)
    return np.where(x <= 0.0031308, 12.92 * x,
                    1.055 * np.power(x, 1.0 / 2.4) - 0.055)
