"""Spatial representations: immutable field specifications and host views.

Import-light by design (NumPy only): the CLI, archive inspection, and host
coefficient access depend on this package, so it must never import CuPy,
Matplotlib, or the numerical cores.
"""
