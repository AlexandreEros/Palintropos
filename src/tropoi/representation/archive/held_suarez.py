"""Read a Held–Suarez experiment directory as a read-only Simulation (CPU).

``tropoi.run.held_suarez`` writes experiments, not run capsules
(``run/held_suarez/experiment.py``)::

    config.json      {"config": HeldSuarezConfig.to_dict(), "config_sha256",
                      "code", "created", "protocol_sha256"}
    states/          state-d<day>.npy: the full PE state X^n at the end of day
                     <day> (every state_every_days), complex (3K+1, L+1, L+1)
    statistics.npz   block means of zonal means and daily scalar series
    series.json, events.jsonl, progress.json, checkpoints/, report/

:class:`HeldSuarezStorage` exposes the instantaneous states under
``states/`` through the :class:`~tropoi.temporal.simulation.SnapshotStorage`
protocol with the PE packed layout ``[zeta_1..K, delta_1..K, T_1..K,
ln p_s]``. A state of day ``d`` is at ``t = d * 86400 s`` exactly (the
experiment's dt divides one day and states are written after step
``d * steps_per_day``). Accumulated statistics are NOT snapshots: an
experiment without ``states/`` opens as a simulation with no snapshots and
says what it holds instead; no time series is ever made from averages.

Compatibility: the configuration is read as ``HeldSuarezConfig.from_dict``
reads it, a missing ``schema`` key means version 1 and missing constants
take the dataclass defaults (duplicated in :data:`CONFIG_DEFAULTS`, kept in
sync with ``run/held_suarez/config.py`` by a test, so this reader never
imports the run layer). The lower boundary is flat: the protocol has no
topography and the experiment builds its model without terrain.

Nothing here writes to the experiment. NumPy + stdlib only.
"""
from __future__ import annotations
from tropoi.representation.archive.availability import NoSnapshotsError, statistics_only_description

import json
import pathlib
import re
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import numpy as np

from tropoi.representation.archive.pe_context import PEColumnContext
from tropoi.representation.archive.schema import SchemaError, state_schema_for
from tropoi.spatial.modes import FieldSpec

__all__ = [
    "CONFIG_DEFAULTS",
    "HELD_SUAREZ_CONFIG_SCHEMA",
    "HeldSuarezStorage",
    "NoSnapshotsError",
    "is_held_suarez_experiment",
    "statistics_only_description",
]

HELD_SUAREZ_CONFIG_SCHEMA = "palintropos.held_suarez.config/1"
DAY_SECONDS = 86400.0
STATE_PATTERN = re.compile(r"^state-d(\d+)\.npy$")

#: Defaults of the HeldSuarezConfig fields this reader uses (PROTOCOL §2);
#: a key missing from an older config.json takes this value, exactly as
#: ``HeldSuarezConfig.from_dict`` would give it.
CONFIG_DEFAULTS = {
    "radius": 6.371e6,
    "omega": 7.292e-5,
    "gravity": 9.8,
    "cp_dry": 1004.0,
    "kappa": 2.0 / 7.0,
    "state_every_days": 10,
    "retained_truncation": None,
    "grid": "latlon",
}
_REQUIRED = ("experiment_id", "l_max", "nlat", "nlon", "nlev")




class HeldSuarezPlotError(RuntimeError):
    """Held–Suarez experiments have no snapshot figure product."""


def _read_config(run_dir: pathlib.Path) -> dict | None:
    path = run_dir / "config.json"
    if not path.is_file():
        return None
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return doc if isinstance(doc, dict) else None


def _experiment_config(doc: Mapping) -> dict | None:
    config = doc.get("config")
    if not isinstance(config, dict):
        return None
    schema = config.get("schema")
    if schema == HELD_SUAREZ_CONFIG_SCHEMA:
        return config
    if schema is None and "forcing" in config and all(
            k in config for k in _REQUIRED):
        return config            # pre-schema-key configuration (version 1)
    return None


def is_held_suarez_experiment(path) -> bool:
    """True for a directory holding a Held–Suarez experiment config.json."""
    doc = _read_config(pathlib.Path(path))
    return doc is not None and _experiment_config(doc) is not None




class HeldSuarezStorage:
    """:class:`SnapshotStorage` over one Held–Suarez experiment directory."""

    format = "held_suarez_experiment"

    def __init__(self, run_dir, *, mmap: bool = True) -> None:
        self.run_dir = pathlib.Path(run_dir)
        self._mmap = bool(mmap)
        doc = _read_config(self.run_dir)
        config = None if doc is None else _experiment_config(doc)
        if config is None:
            raise SchemaError(
                f"{self.run_dir} has no Held–Suarez config.json "
                f"(schema {HELD_SUAREZ_CONFIG_SCHEMA})")
        missing = [k for k in _REQUIRED if k not in config]
        if missing:
            raise SchemaError(
                f"Held–Suarez config.json under {self.run_dir} lacks "
                f"{missing}")
        self.document = dict(doc)
        self.experiment_config = dict(config)
        self.defaulted = sorted(k for k in CONFIG_DEFAULTS if k not in config)
        cfg = {**CONFIG_DEFAULTS, **config}
        self._cfg = cfg
        if cfg["grid"] != "latlon":
            raise SchemaError(f"unexpected Held–Suarez grid {cfg['grid']!r}")
        self.l_max = int(cfg["l_max"])
        self.nlev = int(cfg["nlev"])
        pe_config = {"lmax": self.l_max, "nlev": self.nlev, "grid": "latlon",
                     "nlat": int(cfg["nlat"]), "nlon": int(cfg["nlon"])}
        schema = state_schema_for("pe", pe_config, provenance={
            "source": "held_suarez_experiment",
            "rule": "PE packed layout of run/held_suarez/experiment.py "
                    "states (X^n rows [zeta_1..K, delta_1..K, T_1..K, "
                    "ln p_s]); SigmaGrid.uniform(nlev) as build_model uses"})
        geometry = {"grid": "latlon", "nlat": int(cfg["nlat"]),
                    "nlon": int(cfg["nlon"]), "planet": "ideal_sphere",
                    "reference_radius_m": float(cfg["radius"])}
        environment = {"scenario": "held_suarez",
                       "r_dry": float(cfg["kappa"]) * float(cfg["cp_dry"]),
                       "cp_dry": float(cfg["cp_dry"]),
                       "gravity": float(cfg["gravity"]),
                       "topography": "flat"}
        self._schema = replace(schema, coefficient_file="states/state-d*.npy",
                               time_file=None, geometry=geometry,
                               environment=environment)
        self._days, self._paths = self._scan_states()
        self._times = np.asarray(self._days, dtype=np.float64) * DAY_SECONDS
        self._times.flags.writeable = False
        self._metadata: dict | None = None

    # -- discovery ---------------------------------------------------------

    def _scan_states(self) -> tuple[list[int], list[pathlib.Path]]:
        states = self.run_dir / "states"
        if not states.is_dir():
            return [], []
        found = []
        for path in states.iterdir():
            match = STATE_PATTERN.match(path.name)
            if match and path.is_file():
                found.append((int(match.group(1)), path))
        found.sort()
        days = [d for d, _ in found]
        if len(set(days)) != len(days):
            raise SchemaError(f"duplicate state days under {states}")
        shape = self._schema.frame_shape
        for day, path in found:
            try:
                array = np.load(path, mmap_mode="r", allow_pickle=False)
            except (OSError, ValueError) as err:
                raise SchemaError(f"unreadable state {path}: {err}") from err
            if tuple(array.shape) != shape or not np.issubdtype(
                    array.dtype, np.complexfloating):
                raise SchemaError(
                    f"{path.name} has shape {array.shape} / {array.dtype}; a "
                    f"Held–Suarez state of this experiment is complex {shape}")
            del array
        return days, [p for _, p in found]

    # -- SnapshotStorage protocol -------------------------------------------

    solver = "pe"

    @property
    def times(self) -> np.ndarray:
        return self._times

    @property
    def days(self) -> tuple[int, ...]:
        return tuple(self._days)

    @property
    def frame_count(self) -> int:
        return len(self._paths)

    @property
    def schema(self):
        return self._schema

    @property
    def field_specs(self) -> tuple[FieldSpec, ...]:
        return self._schema.fields

    @property
    def level_values(self) -> tuple[float, ...] | None:
        return self._schema.level_values

    @property
    def run_config(self) -> dict:
        return dict(self.experiment_config)

    @property
    def metadata(self) -> Mapping[str, Any]:
        if self._metadata is None:
            statistics = statistics_only_description(self.run_dir)
            self._metadata = {
                "run_path": str(self.run_dir),
                "run_id": self.experiment_config["experiment_id"],
                "solver": "pe",
                "format": self.format,
                "experiment": "held_suarez",
                "created_utc": self.document.get("created"),
                "run_config": dict(self.experiment_config),
                "config_sha256": self.document.get("config_sha256"),
                "protocol_sha256": self.document.get("protocol_sha256"),
                "git": self.document.get("code"),
                "schema": self._schema.to_manifest_dict(),
                "provenance": {
                    "schema": dict(self._schema.provenance),
                    "time_axis": {
                        "source": "state file names",
                        "rule": "t = day * 86400 s for states/state-d<day>.npy"},
                    "defaulted_config_keys": list(self.defaulted),
                    "memory_mapped": self._mmap,
                },
                "snapshot_count": self.frame_count,
                "statistics_only": statistics,
                "time_units": "s",
            }
        return self._metadata

    def frame(self, index: int) -> np.ndarray:
        index = int(index)
        if not 0 <= index < self.frame_count:
            raise IndexError(
                f"snapshot index {index} is out of range for "
                f"{self.frame_count} stored snapshot(s)")
        array = np.load(self._paths[index],
                        mmap_mode="r" if self._mmap else None,
                        allow_pickle=False)
        array.flags.writeable = False
        view = array.view()
        return view

    def state_path(self, index: int) -> pathlib.Path:
        return self._paths[int(index)]

    def pe_context(self, *, allow_cuda: bool = False) -> PEColumnContext:
        """Constants and (flat) lower boundary of this experiment.

        ``allow_cuda`` is accepted for interface symmetry with capsules and
        unused: nothing here needs a GPU.
        """
        del allow_cuda
        cfg = self._cfg
        nlev = self.nlev
        provenance = {}
        for key in ("radius", "gravity", "cp_dry", "kappa"):
            provenance[key] = ("HeldSuarezConfig default (absent from "
                               "config.json)" if key in self.defaulted
                               else "config.json")
        return PEColumnContext(
            label=str(cfg["experiment_id"]), kind=self.format,
            l_max=self.l_max,
            sigma_interfaces=tuple(k / nlev for k in range(nlev + 1)),
            r_dry=float(cfg["kappa"]) * float(cfg["cp_dry"]),
            cp_dry=float(cfg["cp_dry"]), radius_m=float(cfg["radius"]),
            gravity=float(cfg["gravity"]), surface_elevation_lm=None,
            terrain_label=("flat lower boundary: the Held–Suarez protocol "
                           "has no topography (PROTOCOL.md §1) and "
                           "run/held_suarez/model.py builds a flat surface"),
            provenance={
                "sigma_interfaces": "SigmaGrid.uniform(nlev), as "
                                    "run/held_suarez/model.py builds it",
                "r_dry": f"kappa * cp_dry ({provenance['kappa']}, "
                         f"{provenance['cp_dry']})",
                "cp_dry": provenance["cp_dry"],
                "radius_m": f"radius, an ideal sphere "
                            f"({provenance['radius']})",
                "gravity": provenance["gravity"],
                "terrain": "flat (protocol)",
            })

    # -- rendering (explicit) ----------------------------------------------

    def view(self, path=None, **options):
        from tropoi.representation.visual.atmosphere import view_atmosphere
        return view_atmosphere(self, path, **options)

    def plot_snapshot(self, index: int, output_path, **options):
        raise HeldSuarezPlotError(
            "Held–Suarez experiments have no snapshot figure product; use "
            "Simulation.view() for the 3-D atmosphere or the experiment's "
            "report (python -m tropoi.run.held_suarez report)")

    def plot_simulation(self, output_path=None, view=None, **options):
        raise HeldSuarezPlotError(
            "Held–Suarez experiments are not run capsules and have no "
            "overview figure here; use Simulation.view() for the 3-D "
            "atmosphere or the experiment's report")

    def quantities(self) -> list[dict]:
        return []

    def __repr__(self) -> str:
        return (f"HeldSuarezStorage({str(self.run_dir)!r}, "
                f"states={self.frame_count})")
