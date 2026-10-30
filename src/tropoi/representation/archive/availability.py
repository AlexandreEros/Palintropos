"""Saved-state availability errors and statistics-only directory descriptions."""
from __future__ import annotations
import pathlib

class NoSnapshotsError(ValueError):
    """The run stores no reconstructible instantaneous states."""


def statistics_only_description(path) -> str | None:
    """What a directory without saved states holds, or ``None``.

    Recognizes accumulated-statistics products: a Held–Suarez
    ``statistics.npz`` / ``series.json`` and reference-model series
    (``series.npz`` with ``meta.json``, e.g. the Dinosaur reference).
    """
    run_dir = pathlib.Path(path)
    found = [name for name in ("statistics.npz", "series.json", "series.npz")
             if (run_dir / name).is_file()]
    if not found:
        return None
    states = run_dir / "states"
    if states.is_dir() and any(states.glob("state-d*.npy")):
        return None
    what = {
        "statistics.npz": "block means of zonal means on sigma surfaces",
        "series.json": "daily global scalar series",
        "series.npz": "daily zonal-mean series",
    }
    held = "; ".join(f"{name} ({what[name]})" for name in found)
    return (f"{run_dir} stores accumulated statistics only: {held}. These "
            "are zonal/time averages and scalar series, not instantaneous "
            "states, and they contain no longitude-resolved fields at any "
            "saved time, so no snapshot can be reconstructed from them and "
            "no time series is made from the averages. The 3-D view needs "
            "saved states (a Held–Suarez experiment's states/state-d*.npy, "
            "written every state_every_days)")

