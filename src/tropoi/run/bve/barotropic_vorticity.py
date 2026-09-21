"""Compatibility shim: the BVE core lives in ``tropoi.temporal.tendencies``.

The class and dataclass are re-exported unchanged so every historical import
(``tropoi.run.bve.barotropic_vorticity``) keeps working. New code should
import :class:`BarotropicVorticity` from
:mod:`tropoi.temporal.tendencies.barotropic` and :class:`BarotropicState`
from :mod:`tropoi.spatial.states.barotropic`.
"""
from tropoi.spatial.states.barotropic import BarotropicState  # noqa: F401
from tropoi.temporal.tendencies.barotropic import (  # noqa: F401
    BarotropicVorticity)
