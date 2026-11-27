"""Read-only access to airless conductive-column archives.

The contract is the one the airless runner writes (``tropoi.run.airless``),
read here without importing it:

* ``config.json``: ``{"config": {...}, "config_sha256", "code", ...}``, the
  configuration having ``columns.kind`` (``latlon``, ``gauss`` or
  ``points``), ``radius_m``, ``illumination``, ``strata``, ``output``,
  ``cycle_s`` and ``cycle_steps``;
* ``grid.npz``: ``lat_deg``, ``lon_deg`` and ``weights`` per column,
  ``interfaces`` and ``centres`` (depth, m, positive downward) and, in
  archives written after the distinction was made, ``column_kind``;
* ``cycles.json``: one record per cycle or transient segment (``cycle``,
  ``steps``, ``dt_s`` and, for transients, ``first_step``);
* tier-2 directories ``cycles/cycle-KKKK`` and ``retained``, each with
  ``blocks.json`` (column ranges of the per-block files) and per block
  ``surface-bNNN.npy`` ``(samples, 5, ncol)`` (skin temperature, incident
  shortwave, absorbed forcing, emission, conduction into the first cell),
  ``profiles-bNNN.npy`` ``(samples, L, ncol)`` (cell-centre temperature)
  and, for point runs, ``every_step-bNNN.npy`` ``(steps, 1 + L, ncol)``
  (skin, then the cells); newer directories add ``state.npz`` with their
  own timing; ``retained/status.json`` says which cycle the result quotes.

**Sample times** are the forcing time of the backward-Euler step that
produced the sample, ``t = (first + (k + 1) * every) * dt``, the runner's own
arithmetic (step ``n`` is forced at ``(first + n + 1) * dt``). ``first`` and
``dt`` come from the directory's ``state.npz`` where it has them, otherwise
from the cycle's record in ``cycles.json`` (``first = cycle * steps`` for a
cycle of a periodic run). The cadence ``every`` is the recorded one; where
none was recorded it is ``steps / samples``, accepted only when exact.
Every axis records which of these it used.

Arrays are memory-mapped, and every read selects samples, levels and
columns before anything is copied, so a map, a section or a column never
loads a whole cycle. Every path goes through :func:`resource`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np

from tropoi.representation.archive.local_resource import resource

__all__ = ["AirlessArchive", "is_airless_archive", "SURFACE_PLANES", "READER"]

READER = "airless-columns/1"

#: The planes of ``surface-bNNN.npy`` (``StepReport`` of the substrate step).
SURFACE_PLANES = ("surface_temperature", "incident_shortwave", "absorbed_forcing",
                  "emitted_longwave", "conducted_flux")
GLOBAL_KINDS = ("latlon", "gauss")


def _json(root, name):
    return json.loads(resource(root, name).read_text(encoding="utf-8"))


def is_airless_archive(root) -> bool:
    """True when ``root`` has the files of the contract above."""
    root = Path(root)
    config, grid = resource(root, "config.json"), resource(root, "grid.npz")
    if not (config.is_file() and grid.is_file() and resource(root, "cycles.json").is_file()):
        return False
    document = json.loads(config.read_text(encoding="utf-8"))
    cfg = document.get("config") if isinstance(document, dict) else None
    return isinstance(cfg, dict) and {"columns", "strata", "radius_m"} <= cfg.keys()


@dataclass(frozen=True)
class Segment:
    """One tier-2 directory's samples of one record family."""
    directory: str
    cycle: int
    stage: str
    steps: int
    dt: float
    first: int
    every: int
    count: int
    cadence_source: str
    timing_source: str

    def times(self):
        k = np.arange(1, self.count + 1, dtype=np.int64)
        return (self.first + k * self.every) * self.dt

    def step_index(self, k):
        """Global index of the step after which sample ``k`` was taken."""
        return self.first + (k + 1) * self.every


class AirlessArchive:
    """One archive directory, opened read-only."""

    FAMILIES = ("surface", "profiles", "every_step")

    def __init__(self, root):
        self.root = Path(root).resolve(strict=True)
        document = _json(self.root, "config.json")
        self.document = document
        self.config = document["config"]
        with np.load(resource(self.root, "grid.npz"), allow_pickle=False) as grid:
            self.lat = np.asarray(grid["lat_deg"], dtype=float)
            self.lon = np.asarray(grid["lon_deg"], dtype=float)
            self.weights = np.asarray(grid["weights"], dtype=float) if "weights" in grid else None
            self.interfaces = np.asarray(grid["interfaces"], dtype=float)
            self.centres = np.asarray(grid["centres"], dtype=float)
            stored_kind = str(grid["column_kind"]) if "column_kind" in grid else None
        self.kind = self.config["columns"]["kind"]
        if stored_kind is not None and stored_kind != self.kind:
            raise ValueError(f"grid.npz column_kind {stored_kind!r} disagrees with the "
                             f"configured {self.kind!r}")
        self.kind_source = "grid.npz:column_kind" if stored_kind else "config.json:columns.kind"
        if self.kind not in (*GLOBAL_KINDS, "points"):
            raise ValueError(f"Unknown column kind {self.kind!r}")
        n = self.lat.size
        if self.lat.shape != (n,) or self.lon.shape != (n,) or n == 0:
            raise ValueError("grid.npz coordinates must be one value per column")
        L = self.centres.size
        if (self.interfaces.shape != (L + 1,) or self.interfaces[0] != 0
                or np.any(np.diff(self.interfaces) <= 0)
                or np.any((self.centres <= self.interfaces[:-1]) | (self.centres >= self.interfaces[1:]))):
            raise ValueError("grid.npz depth faces and centres are inconsistent")
        self.columns = n
        self.layers = L
        self.records = {int(r["cycle"]): r for r in _json(self.root, "cycles.json")}
        self.status = (_json(self.root, "retained/status.json")
                       if resource(self.root, "retained/status.json").is_file() else None)
        self.transient_event = None
        events = resource(self.root, "events.jsonl")
        if events.is_file():
            for line in events.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    event = json.loads(line)
                    if event.get("event") == "transient":
                        self.transient_event = event

    # ------------------------------------------------------------------
    # Tier-2 directories and their time axes
    # ------------------------------------------------------------------

    @cached_property
    def directories(self):
        """``(relative path, cycle)`` of every tier-2 directory, in time order.

        ``retained`` is a copy of one cycle; it is used only when that
        cycle's own directory is no longer kept."""
        found = {}
        cycles = resource(self.root, "cycles")
        if cycles.is_dir():
            for path in sorted(cycles.iterdir()):
                name = path.name
                if name.startswith("cycle-") and name[6:].isdigit():
                    resource(self.root, "cycles/" + name)
                    found[int(name[6:])] = "cycles/" + name
        if self.status is not None and int(self.status["cycle"]) not in found:
            found[int(self.status["cycle"])] = "retained"
        return [(found[c], c) for c in sorted(found)]

    def _blocks(self, directory):
        path = resource(self.root, directory + "/blocks.json")
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))["ranges"]

    def _files(self, directory, family):
        """``[(path, lo, hi)]`` of the family's per-block files, or None."""
        ranges = self._blocks(directory)
        if ranges is None:                     # written before per-block streaming
            path = resource(self.root, f"{directory}/{family}.npy")
            return [(path, 0, self.columns)] if path.is_file() else None
        files = [(resource(self.root, f"{directory}/{family}-b{b:03d}.npy"), lo, hi)
                 for b, (lo, hi) in enumerate(ranges)]
        if not all(path.is_file() for path, _, _ in files):
            return None
        if ranges[0][0] != 0 or ranges[-1][1] != self.columns or any(
                a[1] != b[0] for a, b in zip(ranges, ranges[1:])):
            raise ValueError(f"{directory}/blocks.json does not cover the columns")
        return files

    def _mapped(self, path):
        return np.load(path, mmap_mode="r", allow_pickle=False)

    def _timing(self, directory, cycle):
        """``(steps, dt, first, recorded cadence or None, source)``."""
        state = resource(self.root, directory + "/state.npz")
        record = self.records.get(cycle)
        if state.is_file():
            with np.load(state, allow_pickle=False) as s:
                if "dt" in s and "first_step" in s:
                    every = tuple(int(v) for v in s["sample_every"]) if "sample_every" in s else None
                    return (int(s["steps"]), float(s["dt"]), int(s["first_step"]), every,
                            directory + "/state.npz")
        if record is None:
            raise ValueError(f"No cycles.json record for {directory}")
        steps, dt = int(record["steps"]), float(record["dt_s"])
        first = int(record["first_step"]) if "first_step" in record else cycle * steps
        every = None
        if record.get("stage") == "transient" and self.transient_event:
            cadence = self.transient_event.get("sample_every_steps")
            if cadence:
                every = (int(cadence["surface"]), int(cadence["profile"]))
        return steps, dt, first, every, "cycles.json"

    @cached_property
    def segments(self):
        """Per family, the segments of every directory that holds it."""
        out = {family: [] for family in self.FAMILIES}
        for directory, cycle in self.directories:
            steps, dt, first, recorded, source = self._timing(directory, cycle)
            stage = (self.records.get(cycle) or {}).get("stage", "")
            for f, family in enumerate(self.FAMILIES):
                files = self._files(directory, family)
                if files is None:
                    continue
                count = int(self._mapped(files[0][0]).shape[0])
                if family == "every_step":
                    every, how = 1, "every step"
                elif recorded is not None:
                    every, how = recorded[f], "recorded sampling cadence"
                elif count and steps % count == 0:
                    every, how = steps // count, "steps / samples (exact)"
                else:
                    raise ValueError(f"{directory}: {count} {family} samples do not divide "
                                     f"{steps} steps and no cadence was recorded")
                if count * every > steps:
                    raise ValueError(f"{directory}: {count} {family} samples every {every} steps "
                                     f"exceed the {steps} steps of the cycle")
                out[family].append(Segment(directory, cycle, stage, steps, dt, first, every, count,
                                           how, source))
        for family, segments in out.items():
            times = np.concatenate([s.times() for s in segments]) if segments else np.zeros(0)
            if np.any(np.diff(times) <= 0):
                raise ValueError(f"The {family} samples of the tier-2 directories overlap in time")
        return out

    def times(self, family):
        segments = self.segments[family]
        return np.concatenate([s.times() for s in segments]) if segments else np.zeros(0)

    def steps(self, family):
        """Global step index after which each sample was taken."""
        return np.concatenate([s.step_index(np.arange(s.count)) for s in self.segments[family]]
                              ) if self.segments[family] else np.zeros(0, dtype=np.int64)

    def _locate(self, family, index):
        if isinstance(index, bool) or int(index) != index:
            raise ValueError("Saved sample index must be an integer")
        index = int(index)
        for segment in self.segments[family]:
            if 0 <= index < segment.count:
                return segment, index
            index -= segment.count
        raise ValueError("Saved sample index is outside the archived samples")

    # ------------------------------------------------------------------
    # Bounded reads
    # ------------------------------------------------------------------

    def _gather(self, family, directory, select, columns=None):
        """``select(array)`` applied to each block's memmap, joined on columns.

        ``select`` indexes the leading axes; the column axis stays last."""
        parts = []
        for path, lo, hi in self._files(directory, family):
            mapped = self._mapped(path)
            if mapped.shape[-1] != hi - lo:
                raise ValueError(f"{path.name} has {mapped.shape[-1]} columns, not {hi - lo}")
            if columns is None:
                parts.append(np.array(select(mapped)))
            else:
                local = columns[(columns >= lo) & (columns < hi)] - lo
                if local.size:
                    parts.append(np.array(select(mapped)[..., local]))
        return np.concatenate(parts, axis=-1)

    def _columns(self, columns):
        cols = np.asarray(columns, dtype=np.int64)
        if cols.ndim != 1 or np.any(cols < 0) or np.any(cols >= self.columns):
            raise ValueError("Column index outside the archive")
        if np.any(np.diff(cols) <= 0):
            raise ValueError("Columns must be strictly increasing")
        return cols

    def surface(self, index, plane, columns=None):
        """One surface plane at sample ``index``: ``(ncol,)`` or the columns given."""
        segment, k = self._locate("surface", index)
        cols = None if columns is None else self._columns(columns)
        return self._gather("surface", segment.directory, lambda a: a[k, plane], cols)

    def profile(self, index, levels=None, columns=None):
        """Cell-centre temperature at sample ``index``: ``(levels, ncol)``."""
        segment, k = self._locate("profiles", index)
        cols = None if columns is None else self._columns(columns)
        sel = slice(None) if levels is None else np.asarray(levels, dtype=np.int64)
        return self._gather("profiles", segment.directory, lambda a: a[k, sel], cols)

    def every_step(self, index, rows=None, columns=None):
        segment, k = self._locate("every_step", index)
        cols = None if columns is None else self._columns(columns)
        sel = slice(None) if rows is None else np.asarray(rows, dtype=np.int64)
        return self._gather("every_step", segment.directory, lambda a: a[k, sel], cols)

    def history(self, family, column, first, last, rows=None):
        """Samples ``first..last`` (inclusive) of one column: ``(count, rows)``.

        ``rows`` are surface planes, profile levels or every-step rows."""
        column = int(self._columns([column])[0])
        out, start = [], 0
        for segment in self.segments[family]:
            lo, hi = max(first - start, 0), min(last - start, segment.count - 1)
            if lo <= hi:
                for path, c0, c1 in self._files(family=family, directory=segment.directory):
                    if c0 <= column < c1:
                        mapped = self._mapped(path)
                        block = mapped[lo:hi + 1, :, column - c0]
                        out.append(np.array(block if rows is None else block[:, rows]))
                        break
            start += segment.count
        if not out:
            raise ValueError("No saved samples in the selected range")
        return np.concatenate(out, axis=0)

    # ------------------------------------------------------------------
    # Horizontal layout
    # ------------------------------------------------------------------

    @cached_property
    def layout(self):
        """For a global kind, ``(lat (nlat,), lon (nlon,), index (nlat, nlon))``
        on a south-first, east-ascending mesh, from the archived coordinates
        (no ordering rule is assumed); None for a point collection."""
        if self.kind not in GLOBAL_KINDS:
            return None
        lon = self.lon % 360.0
        lats = np.unique(self.lat)
        lons = np.unique(lon)
        if lats.size * lons.size != self.columns:
            raise ValueError("The global columns do not form a latitude-longitude product grid")
        i = np.searchsorted(lats, self.lat)
        j = np.searchsorted(lons, lon)
        index = np.full((lats.size, lons.size), -1, dtype=np.int64)
        index[i, j] = np.arange(self.columns)
        if np.any(index < 0):
            raise ValueError("The global columns do not form a latitude-longitude product grid")
        return lats, lons, index

    @cached_property
    def cell_edges(self):
        """Latitude and longitude cell edges when the grid is the regular
        cell-centred one (centres midway between equally spaced edges)."""
        if self.layout is None or self.kind != "latlon":
            return None
        lats, lons, _ = self.layout
        lat_edges = np.linspace(-90.0, 90.0, lats.size + 1)
        width = 360.0 / lons.size
        lon_edges = lons[0] - width / 2 + width * np.arange(lons.size + 1)
        if not (np.allclose((lat_edges[:-1] + lat_edges[1:]) / 2, lats, atol=1e-9)
                and np.allclose(np.diff(lons), width, atol=1e-9)):
            return None
        return lat_edges, lon_edges

    def nearest_column(self, lat, lon):
        """The column whose centre is nearest on the sphere (great circle)."""
        phi, lam = np.radians(self.lat), np.radians(self.lon)
        p0, l0 = np.radians(lat), np.radians(lon)
        cos = np.sin(phi) * np.sin(p0) + np.cos(phi) * np.cos(p0) * np.cos(lam - l0)
        return int(np.argmax(cos))
