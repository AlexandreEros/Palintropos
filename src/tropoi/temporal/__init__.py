"""Temporal layer: how states evolve and how a saved evolution is read.

``integration`` holds the scheduler, the advective-CFL arithmetic and the
RK4/driver loop; ``tendencies`` holds the BVE, SWE and PE tendency
implementations (with their tightly coupled dissipation terms);
``simulation`` is the read-only ``Simulation``/``Snapshot`` facade over a
small storage protocol, independent of any archive format or renderer.

This initializer imports nothing. ``integration`` and ``simulation`` are
CPU-only (stdlib / NumPy); ``tendencies`` imports CuPy. The temporal layer
depends on ``tropoi.spatial`` only, never on ``tropoi.representation``,
``tropoi.run`` or the CLI.
"""
