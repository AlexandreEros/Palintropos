"""Open a saved run capsule as a read-only :class:`Simulation`.

``open_simulation(run_path)`` reads ``manifest.json`` / ``config.json``,
determines the solver and the stored-state schema (from the manifest's
``state_schema`` block, or by inference for the known historical formats),
memory-maps the coefficient array read-only, loads the authoritative time
axis, and validates headers, layout, dtype, and the time axis at open.
Coefficient VALUES are never scanned at open: convention checks (finite,
triangular, real ``a_l0``, monopole rule) run when a field is accessed
through ``Snapshot.state[...]`` (they are also available explicitly through
``SpectralModes.validate``), and nothing is ever repaired.

Everything here is import-light (NumPy + stdlib): CUDA, Matplotlib, the
numerical cores, and the visualization adapters are never imported by
opening or inspecting a capsule. ``Snapshot.plot`` imports
``tropoi.representation.visual.snapshot`` lazily on the explicit call.
"""
from __future__ import annotations

from collections.abc import Mapping
import json
import pathlib
from typing import Any

import numpy as np

from tropoi.spatial.modes import FieldSpec, SpectralConventionError
from tropoi.temporal.simulation import Simulation, SnapshotStorage
from .schema import (SchemaError, StateSchema, UnknownSchemaVersionError,
                     COEFFICIENT_FILES, TIME_FILES, infer_solver,
                     infer_state_schema, diagnostic_definitions_for)

__all__ = [
    "CapsuleError",
    "CapsuleLayoutError",
    "CapsuleStorage",
    "SchemaError",
    "UnknownSchemaVersionError",
    "open_simulation",
    "resolve_run_directory",
]


class CapsuleError(ValueError):
    """The capsule cannot be opened as a saved run."""


class CapsuleLayoutError(CapsuleError):
    """A stored array's header, layout, dtype, or time axis is invalid."""


# ---------------------------------------------------------------------------
# Location and metadata
# ---------------------------------------------------------------------------

def resolve_run_directory(target) -> pathlib.Path:
    """Accept a run directory or a base directory holding ``latest_run.txt``."""
    path = pathlib.Path(target)
    if not path.is_dir():
        raise CapsuleError(f"run path is not a directory: {path}")
    if (path / "manifest.json").exists() or (path / "config.json").exists():
        return path
    pointer = path / "latest_run.txt"
    if pointer.exists():
        content = pointer.read_text(encoding="utf-8").strip()
        if not content:
            raise CapsuleError(f"latest_run.txt at {pointer} is empty")
        pointed = pathlib.Path(content)
        resolved = pointed if pointed.is_absolute() else path / pointed
        if not resolved.is_dir():
            raise CapsuleError(
                f"latest_run.txt at {pointer} points at a missing run "
                f"directory: {resolved}")
        return resolved
    raise CapsuleError(
        f"no manifest.json, config.json, or latest_run.txt under {path}")


def _read_json_object(path: pathlib.Path, label: str) -> dict:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as err:
        raise CapsuleError(f"could not read {label} at {path}: {err}") from err
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as err:
        raise CapsuleError(f"malformed {label} at {path}: {err}") from err
    if not isinstance(data, dict):
        raise CapsuleError(
            f"malformed {label} at {path}: expected an object, got "
            f"{type(data).__name__}")
    return data


def read_capsule_metadata(run_dir: pathlib.Path) -> tuple[dict, dict]:
    """``(manifest, run_config)`` from manifest.json and/or config.json."""
    manifest_path = run_dir / "manifest.json"
    config_path = run_dir / "config.json"
    manifest: dict = {}
    run_config: dict | None = None
    if manifest_path.exists():
        manifest = _read_json_object(manifest_path, "manifest.json")
        raw = manifest.get("run_config")
        if raw is not None:
            if not isinstance(raw, dict):
                raise CapsuleError(
                    f"malformed manifest.json under {run_dir}: 'run_config' "
                    f"is {type(raw).__name__}, expected an object")
            run_config = raw
    if run_config is None:
        if not config_path.exists():
            raise CapsuleError(
                f"no run configuration under {run_dir} (manifest.json has "
                "no run_config and config.json is missing)")
        run_config = _read_json_object(config_path, "config.json")
    return manifest, run_config


# ---------------------------------------------------------------------------
# Array access
# ---------------------------------------------------------------------------

def _open_array(path: pathlib.Path, *, mmap: bool) -> np.ndarray:
    """Memory-map (default) or load one ``.npy`` file, validating its header."""
    if not path.is_file():
        raise CapsuleLayoutError(f"stored array is missing: {path}")
    try:
        array = np.load(path, mmap_mode="r" if mmap else None,
                        allow_pickle=False)
    except (OSError, ValueError) as err:
        raise CapsuleLayoutError(
            f"stored array has an invalid .npy header or is unreadable: "
            f"{path}: {err}") from err
    if not isinstance(array, np.ndarray):
        raise CapsuleLayoutError(
            f"stored file did not decode to an array: {path}")
    if not mmap:
        array = array.view()
    array.flags.writeable = False
    return array


def _validate_coefficients_layout(array: np.ndarray, schema: StateSchema,
                                  path: pathlib.Path) -> None:
    expected = schema.frame_shape
    if array.ndim != len(expected) + 1:
        raise CapsuleLayoutError(
            f"{path.name} must have shape (time, {', '.join(str(d) for d in expected)})"
            f" for solver {schema.solver!r}, got {array.shape}")
    if tuple(array.shape[1:]) != expected:
        raise CapsuleLayoutError(
            f"{path.name} frames have shape {tuple(array.shape[1:])} but the "
            f"{'manifest' if not schema.inferred else 'inferred'} schema "
            f"expects {expected} (l_max={schema.l_max}"
            + (f", nlev={schema.nlev}" if schema.nlev is not None else "")
            + ")")
    if not np.issubdtype(array.dtype, np.complexfloating):
        raise CapsuleLayoutError(
            f"{path.name} must hold complex coefficients, got dtype "
            f"{array.dtype}")


def _validate_times(times: np.ndarray, frame_count: int,
                    label: str) -> np.ndarray:
    times = np.asarray(times)
    if times.ndim != 1 or times.shape[0] != frame_count:
        raise CapsuleLayoutError(
            f"{label} must be a one-dimensional time axis with one entry per "
            f"stored frame ({frame_count}), got shape {times.shape}")
    if not np.issubdtype(times.dtype, np.number) or np.iscomplexobj(times):
        raise CapsuleLayoutError(
            f"{label} must hold real numbers, got dtype {times.dtype}")
    times = times.astype(np.float64)
    if times.size and not np.isfinite(times).all():
        raise CapsuleLayoutError(f"{label} contains non-finite times")
    if times.size > 1 and not np.all(np.diff(times) > 0.0):
        raise CapsuleLayoutError(f"{label} must be strictly increasing")
    times.flags.writeable = False
    return times


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

class CapsuleStorage:
    """:class:`SnapshotStorage` over one run capsule directory."""

    def __init__(self, run_dir: pathlib.Path, *, mmap: bool = True) -> None:
        self.run_dir = pathlib.Path(run_dir)
        self._mmap = bool(mmap)
        self.manifest, self.run_config = read_capsule_metadata(self.run_dir)
        present = {p.name for p in self.run_dir.iterdir() if p.is_file()}

        self.solver, solver_provenance = infer_solver(
            self.run_config, self.manifest, present)

        # Schema: manifest block if present (typed access refused for an
        # unknown version, but metadata stays readable), else inference.
        self._schema_error: SchemaError | None = None
        self._schema: StateSchema | None = None
        block = self.manifest.get("state_schema")
        if block is not None:
            try:
                self._schema = StateSchema.from_manifest_dict(block)
            except SchemaError as err:
                self._schema_error = err
            else:
                if self._schema.solver != self.solver:
                    self._schema_error = SchemaError(
                        f"manifest state_schema is for solver "
                        f"{self._schema.solver!r} but the capsule is a "
                        f"{self.solver!r} run")
                    self._schema = None
        else:
            self._schema = infer_state_schema(self.solver, self.run_config,
                                              solver_provenance)
        self._time_provenance: dict = {}

        coefficient_file = (self._schema.coefficient_file if self._schema
                            else COEFFICIENT_FILES[self.solver])
        self.coefficient_path = self.run_dir / coefficient_file
        self._coefficients = _open_array(self.coefficient_path,
                                         mmap=self._mmap)
        if self._schema is not None:
            _validate_coefficients_layout(self._coefficients, self._schema,
                                          self.coefficient_path)
        elif self._coefficients.ndim < 3:
            raise CapsuleLayoutError(
                f"{self.coefficient_path.name} must carry a leading time "
                f"axis and (l, m) coefficient axes, got shape "
                f"{self._coefficients.shape}")
        self._times = self._load_times()
        self._metadata: dict | None = None
        self._resources: Any = None
        self._resource_builds = 0

    # -- time axis -------------------------------------------------------

    def _load_times(self) -> np.ndarray:
        frames = int(self._coefficients.shape[0])
        time_file = (self._schema.time_file if self._schema
                     else TIME_FILES[self.solver])
        path = self.run_dir / time_file if time_file else None
        if path is not None and path.is_file():
            times = _open_array(path, mmap=False)
            self._time_provenance = {"source": "stored", "file": time_file}
            return _validate_times(np.array(times), frames, path.name)
        if self.solver == "bve" and self.run_config.get("dt_snapshots"):
            # The only known capsule without a stored time axis: the
            # historical psx-bve interval mode stored t = 0 and every
            # interval boundary in order.
            dt = float(self.run_config["dt_snapshots"])
            times = dt * np.arange(frames, dtype=np.float64)
            self._time_provenance = {
                "source": "inferred",
                "rule": "legacy interval mode: t_k = k * dt_snapshots",
                "dt_snapshots": dt,
                "reason": f"{time_file} is absent",
            }
            return _validate_times(times, frames, "inferred time axis")
        raise CapsuleLayoutError(
            f"stored time axis {time_file} is missing under {self.run_dir} "
            "and the capsule is not a known legacy layout with an inferable "
            "schedule")

    # -- SnapshotStorage protocol ---------------------------------------

    @property
    def times(self) -> np.ndarray:
        return self._times

    @property
    def frame_count(self) -> int:
        return int(self._coefficients.shape[0])

    @property
    def coefficients(self) -> np.ndarray:
        """The complete read-only stored coefficient array (time first)."""
        return self._coefficients

    @property
    def schema(self) -> StateSchema:
        if self._schema is None:
            raise self._schema_error
        return self._schema

    @property
    def schema_error(self) -> SchemaError | None:
        return self._schema_error

    @property
    def field_specs(self) -> tuple[FieldSpec, ...]:
        return self.schema.fields

    @property
    def level_values(self) -> tuple[float, ...] | None:
        return self.schema.level_values

    @property
    def time_provenance(self) -> dict:
        return dict(self._time_provenance)

    @property
    def metadata(self) -> Mapping[str, Any]:
        if self._metadata is None:
            manifest = self.manifest
            schema_dict: dict | str
            if self._schema is not None:
                schema_dict = self._schema.to_manifest_dict()
            else:
                schema_dict = f"unavailable: {self._schema_error}"
            diagnostics = manifest.get("diagnostic_definitions")
            diagnostics_source = "manifest"
            if diagnostics is None:
                try:
                    diagnostics = diagnostic_definitions_for(
                        self.solver, self.run_config)
                    diagnostics_source = "inferred"
                except (SchemaError, KeyError):
                    diagnostics = None
                    diagnostics_source = "unavailable"
            self._metadata = {
                "run_path": str(self.run_dir),
                "run_id": manifest.get("run_id") or self.run_config.get("run_id"),
                "solver": self.solver,
                "status": manifest.get("status"),
                "created_utc": manifest.get("created_utc"),
                "run_config": dict(self.run_config),
                "numerics": manifest.get("numerics"),
                "git": manifest.get("git"),
                "versions": manifest.get("versions"),
                "gpu": manifest.get("gpu"),
                "notes": manifest.get("notes"),
                "argv": manifest.get("argv"),
                "schema": schema_dict,
                "diagnostic_definitions": diagnostics,
                "provenance": {
                    "schema": (self._schema.provenance if self._schema
                               else {"source": "unavailable",
                                     "error": str(self._schema_error)}),
                    "diagnostic_definitions": diagnostics_source,
                    "time_axis": dict(self._time_provenance),
                    "coefficient_file": self.coefficient_path.name,
                    "memory_mapped": bool(self._mmap),
                },
                "snapshot_count": self.frame_count,
                "time_units": "s",
            }
        return self._metadata

    def stored_array(self, filename: str) -> np.ndarray:
        """Another persisted array of this capsule, read-only (memory-mapped).

        Used by the rendering adapter for companion arrays such as the
        BVE plotting grid (``vorticity_grid.npy``); the coefficient array
        itself is :attr:`coefficients`.
        """
        if pathlib.Path(filename).name != filename:
            raise CapsuleError(f"stored array name must be a plain filename, "
                               f"got {filename!r}")
        return _open_array(self.run_dir / filename, mmap=self._mmap)

    def frame(self, index: int) -> np.ndarray:
        index = int(index)
        if not 0 <= index < self.frame_count:
            raise IndexError(
                f"snapshot index {index} is out of range for "
                f"{self.frame_count} stored snapshot(s)")
        self.schema  # typed access requires an interpretable schema
        frame = self._coefficients[index]
        frame.flags.writeable = False
        return frame

    # -- rendering (lazy, explicit) -------------------------------------

    def resources(self):
        """The shared model/backend resources for this capsule (lazy).

        Built at most once per storage and shared by every snapshot,
        representation, and field; requires the numerical cores (CUDA).
        """
        if self._resources is None:
            from tropoi.representation.visual.snapshot import build_resources
            self._resources = build_resources(self)
            self._resource_builds += 1
        return self._resources

    @property
    def resource_builds(self) -> int:
        return self._resource_builds

    def plot_snapshot(self, index: int, output_path, **options
                      ) -> pathlib.Path:
        from tropoi.representation.visual.snapshot import render_snapshot
        return render_snapshot(self, int(index), output_path, **options)

    def __repr__(self) -> str:
        return (f"CapsuleStorage({str(self.run_dir)!r}, solver={self.solver!r}, "
                f"frames={self.frame_count})")


def open_simulation(run_path, *, mmap: bool = True) -> Simulation:
    """Open a saved BVE, SWE, or PE run capsule as a read-only Simulation.

    ``run_path`` is a run directory (or a base directory with
    ``latest_run.txt``). Coefficients are memory-mapped read-only
    (``mmap=False`` loads them into host memory instead, still read-only).
    Opening validates metadata, array headers/layout/dtype, and the time
    axis; it never reads coefficient values or initializes CUDA.
    """
    run_dir = resolve_run_directory(run_path)
    return Simulation(CapsuleStorage(run_dir, mmap=mmap))
