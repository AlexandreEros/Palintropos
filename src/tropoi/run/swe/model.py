"""Construct SWE resources from resolved or saved run configuration.

The builder itself lives in :mod:`tropoi.temporal.models`, below both of its
callers (the runners here and the saved-run interface), so that the archive
can rebuild a finished run's model without the representation layer importing
orchestration. This module keeps its historical import path.
"""
from __future__ import annotations

from tropoi.temporal.models import (  # noqa: F401  (re-exports)
    _w5_planet_params, build_swe_model)

__all__ = ["build_swe_model"]
