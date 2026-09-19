"""Read-only ``Simulation`` and contextual ``Snapshot`` over stored frames.

Both classes are independent of the capsule format and of any plotting
implementation: they consume the small :class:`SnapshotStorage` protocol
and expose saved times, per-index snapshots, named host coefficient views,
and an explicit, lazy ``plot`` operation delegated back to the storage.

Lifetime contract: a :class:`Snapshot` keeps its own reference to the
packed frame it was created from, so it stays valid after another snapshot
is selected or the owning :class:`Simulation` is released; views obtained
through ``snapshot.state`` share memory with that frame and are read-only.
Indices are saved output indices (``sim[137]`` is the 138th stored state),
never interpolated times: no interpolation exists in this interface.
"""
from __future__ import annotations

from collections.abc import Iterator, Mapping
import pathlib
from typing import Any, Protocol, runtime_checkable

import numpy as np

from tropoi.spatial.modes import FieldSpec, SpectralState


@runtime_checkable
class SnapshotStorage(Protocol):
    """What a ``Simulation`` needs from an archive (or any other source)."""

    @property
    def solver(self) -> str: ...

    @property
    def times(self) -> np.ndarray:
        """Saved times in seconds; finite, strictly increasing, read-only."""
        ...

    @property
    def field_specs(self) -> tuple[FieldSpec, ...]: ...

    @property
    def level_values(self) -> tuple[float, ...] | None: ...

    @property
    def metadata(self) -> Mapping[str, Any]: ...

    def frame(self, index: int) -> np.ndarray:
        """The packed read-only frame stored at saved index ``index``."""
        ...

    def plot_snapshot(self, index: int, output_path, **options
                      ) -> pathlib.Path:
        """Render the stored frame explicitly; may require CUDA."""
        ...


def _normalize_index(index, size: int) -> int:
    if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
        raise TypeError(
            f"snapshot index must be an integer saved-output index, got "
            f"{index!r}; time-based or sliced access is not provided")
    normalized = int(index)
    if size == 0:
        raise IndexError(
            "this simulation stores no snapshots (an empty saved sequence); "
            "there is nothing to index")
    if normalized < 0:
        normalized += size
    if not 0 <= normalized < size:
        raise IndexError(
            f"snapshot index {index} is out of range for {size} stored "
            f"snapshot(s)")
    return normalized


class Snapshot:
    """One stored state at one saved time, with named read-only views."""

    __slots__ = ("_storage", "_index", "_time", "_frame", "_state",
                 "_metadata")

    def __init__(self, storage: SnapshotStorage, index: int) -> None:
        self._storage = storage
        self._index = int(index)
        self._time = float(storage.times[self._index])
        # Own reference: the frame (and whatever backs it) outlives the
        # Simulation this snapshot came from.
        self._frame = storage.frame(self._index)
        self._state: SpectralState | None = None
        self._metadata: dict | None = None

    @property
    def index(self) -> int:
        """Saved output index of this snapshot."""
        return self._index

    @property
    def time(self) -> float:
        """Saved time in seconds."""
        return self._time

    @property
    def solver(self) -> str:
        return self._storage.solver

    @property
    def state(self) -> SpectralState:
        """Named read-only host views over this snapshot's packed frame."""
        if self._state is None:
            self._state = SpectralState(
                self._frame, self._storage.field_specs,
                level_values=self._storage.level_values)
        return self._state

    @property
    def packed(self) -> np.ndarray:
        """The complete read-only packed frame (shares memory with views)."""
        return self.state.packed

    @property
    def metadata(self) -> dict:
        """Simulation metadata plus this snapshot's index and time."""
        if self._metadata is None:
            base = dict(self._storage.metadata)
            base["snapshot"] = {"index": self._index, "time_s": self._time,
                                "time_units": "s"}
            self._metadata = base
        return self._metadata

    def plot(self, output_path, **options) -> pathlib.Path:
        """Render this snapshot with the existing scientific composition.

        Explicit and lazy: nothing is synthesized or imported until called.
        Options are forwarded to the storage's renderer (for capsules:
        ``representation`` and ``normalization``; see
        ``tropoi.representation.visual.snapshot``). Returns the written
        image path.
        """
        return self._storage.plot_snapshot(self._index, output_path,
                                           **options)

    def __repr__(self) -> str:
        return (f"Snapshot(solver={self.solver!r}, index={self._index}, "
                f"time={self._time:g} s)")


class Simulation:
    """Read-only sequence of saved snapshots over a storage protocol."""

    def __init__(self, storage: SnapshotStorage) -> None:
        times = np.asarray(storage.times, dtype=np.float64)
        if times.ndim != 1:
            raise ValueError("storage times must be one-dimensional")
        self._storage = storage
        self._times = times.view()
        self._times.flags.writeable = False

    @property
    def storage(self) -> SnapshotStorage:
        return self._storage

    @property
    def solver(self) -> str:
        return self._storage.solver

    @property
    def times(self) -> np.ndarray:
        """Saved times in seconds (read-only host array)."""
        return self._times

    @property
    def metadata(self) -> Mapping[str, Any]:
        return self._storage.metadata

    @property
    def field_names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self._storage.field_specs)

    def __len__(self) -> int:
        return int(self._times.shape[0])

    def __getitem__(self, index) -> Snapshot:
        return Snapshot(self._storage, _normalize_index(index, len(self)))

    def snapshot(self, index) -> Snapshot:
        return self[index]

    def __iter__(self) -> Iterator[Snapshot]:
        for index in range(len(self)):
            yield Snapshot(self._storage, index)

    def __repr__(self) -> str:
        return (f"Simulation(solver={self.solver!r}, snapshots={len(self)}, "
                f"fields={list(self.field_names)})")


__all__ = ["Simulation", "Snapshot", "SnapshotStorage"]
