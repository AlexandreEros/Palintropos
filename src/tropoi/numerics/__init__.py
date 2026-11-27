"""Numerical algorithms shared by the physical and discretization layers.

The bottom layer of ``tropoi``: it imports no other ``tropoi`` package and
knows nothing about planetary physics. Physical models supply their own
equations and admissibility constraints; the algorithms here solve the
resulting algebraic systems. This initializer imports nothing.
"""
