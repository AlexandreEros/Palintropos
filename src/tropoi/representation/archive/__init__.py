"""Saved-run archive access (import-light: NumPy + stdlib only).

Public entry point::

    from tropoi.representation.archive import open_simulation

    sim = open_simulation(run_path)
    sim.times                              # saved times in seconds
    snapshot = sim[137]                    # saved output index
    snapshot.time, snapshot.metadata
    snapshot.state["temperature"].coeffs   # read-only host view
    snapshot.plot(output_path=path)        # explicit, lazy rendering

See docs/SAVED_RUNS.md.
"""
from .capsule import (CapsuleError, CapsuleLayoutError, CapsuleStorage,
                      open_simulation, resolve_run_directory)
from .schema import (STATE_SCHEMA_VERSION, SchemaError, StateSchema,
                     UnknownSchemaVersionError)

__all__ = [
    "CapsuleError",
    "CapsuleLayoutError",
    "CapsuleStorage",
    "STATE_SCHEMA_VERSION",
    "SchemaError",
    "StateSchema",
    "UnknownSchemaVersionError",
    "open_simulation",
    "resolve_run_directory",
]
