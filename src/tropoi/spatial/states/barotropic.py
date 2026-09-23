"""Barotropic (BVE) state description: vorticity coefficients.

Split verbatim from :mod:`tropoi.temporal.tendencies.barotropic`, which
re-exports it.
"""
from __future__ import annotations
from dataclasses import dataclass

import cupy as cp


@dataclass
class BarotropicState:
    coeffs: cp.ndarray
    tendency: cp.ndarray = None
