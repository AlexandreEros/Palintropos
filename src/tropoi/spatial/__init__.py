"""Spatial layer: what a state is and how it is discretized on the sphere.

Prescribed world descriptions (``environment``, ``williamson5``,
``terrain``, the ``planet`` facade), the discretization (``grids``,
``transforms`` with their CUDA kernels, ``spherical_backend``,
``operators``, ``sigma_coordinate``), state descriptions (``states``,
``modes``), initialization, and the truncation policy (``truncation``).

This initializer imports nothing: ``modes``, ``truncation``,
``environment`` and ``williamson5`` are CPU-only and are consumed by the
CLI and by archive inspection; the discretization modules import CuPy and
are loaded only by the code that uses them. The spatial layer never imports
``tropoi.temporal``, ``tropoi.representation``, ``tropoi.run`` or the CLI.
"""
