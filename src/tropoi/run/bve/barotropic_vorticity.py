"""Compatibility shim: the BVE physics moved to ``physics/barotropic.py``.

The class and dataclass are re-exported unchanged so every historical import
(``tropoi.run.bve.barotropic_vorticity``) keeps working. New code
should import from :mod:`tropoi.physics.barotropic`.
"""
from tropoi.temporal.tendencies.barotropic import (  # noqa: F401
    BarotropicState, BarotropicVorticity)
