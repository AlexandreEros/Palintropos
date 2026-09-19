"""Lazy adaptation of saved snapshots to the existing figure builders.

``Snapshot.plot`` ends here. Nothing is imported or synthesized until the
explicit call: this module is NOT imported by archive inspection or host
coefficient access, and its own heavy imports (CuPy through the model
builders, Matplotlib through the default renderer) happen inside the
functions.

What is reused, not rebuilt:

* the per-core compositions — ``run.bve.visualization`` (physical and
  spectral BVE frames), ``run.swe.visualization`` (physical and spectral
  SWE frames, topography panels included) and
  ``run.pe.snapshot_visualization`` (upper/lower-level PE frames) — fed
  with the capsule's stored arrays through their ``*_from_data`` entry
  points, so layouts, palettes, view interpolation and the timeline's
  cross-frame normalization are exactly those of the in-run
  ``snapshots/`` product;
* the existing path-returning renderer (``viz.renderers``), including its
  atomic image write;
* the run's own model/backend resources, reconstructed ONCE per capsule
  from the persisted configuration by the solver modules'
  ``build_planet`` / ``build_swe_model`` / ``build_pe_model`` and shared
  by every snapshot, representation and field.

CPU/GPU boundary: the ``"spectral"`` representation (BVE, SWE) needs no
synthesis and renders on the host; every ``"physical"`` representation
synthesizes fields through the model's spherical-harmonic transform and is
therefore CUDA-gated (the BVE physical frame derives streamfunction and
winds spectrally even though a grid field was saved). Without CUDA the
physical request fails with :class:`PlotUnavailableError`; metadata and
host coefficient access are unaffected.

Normalization: ``normalization="timeline"`` (default) composes EVERY
stored frame so the selected image carries the run-wide limits of the
persisted ``snapshots/`` product (identical composition and normalization
to the in-run frames); ``normalization="frame"`` composes only the
selected frame, whose limits then come from that single state — the image
metadata records which was used.
"""
from __future__ import annotations

from dataclasses import dataclass
import pathlib
from typing import Any

import numpy as np

__all__ = [
    "PlotUnavailableError",
    "REPRESENTATIONS",
    "NORMALIZATIONS",
    "SnapshotResources",
    "build_resources",
    "build_snapshot_timeline",
    "render_snapshot",
]

#: Snapshot representations each solver's existing product provides.
REPRESENTATIONS = {
    "bve": ("physical", "spectral"),
    "swe": ("physical", "spectral"),
    "pe": ("physical",),
}
#: Representations that render on the host (no spectral synthesis).
_HOST_ONLY = {("bve", "spectral"), ("swe", "spectral")}
NORMALIZATIONS = ("timeline", "frame")


class PlotUnavailableError(RuntimeError):
    """The requested rendering cannot run here (typically: no CUDA)."""


@dataclass(frozen=True)
class SnapshotResources:
    """The run's reconstructed model/backend resources (shared per capsule)."""

    solver: str
    planet: Any
    model: Any = None

    @property
    def grid(self):
        return self.planet.grid

    @property
    def transform(self):
        return self.planet.sh


def _cuda_failure(err: BaseException) -> bool:
    name = type(err).__name__
    module = type(err).__module__ or ""
    return (isinstance(err, ImportError) or module.startswith("cupy")
            or "CUDA" in name or "cuda" in str(err).lower())


def build_resources(storage) -> SnapshotResources:
    """Reconstruct the run's Planet/model from its persisted configuration.

    Uses the solver modules' own builders (the code path the run itself
    used), never a preset validator, so legacy capsules are not
    re-validated by newer guards. Requires CUDA; a missing CuPy/CUDA
    surfaces as :class:`PlotUnavailableError` with the CPU alternatives.
    """
    solver = storage.solver
    run_config = storage.run_config
    try:
        if solver == "bve":
            from tropoi.cli.bve import build_planet
            planet = build_planet(run_config)
            return SnapshotResources(solver, planet)
        if solver == "swe":
            from tropoi.cli.swe import build_swe_model
            model = build_swe_model(run_config)
            return SnapshotResources(solver, model.planet, model)
        if solver == "pe":
            from tropoi.cli.pe import build_pe_model
            model = build_pe_model(run_config)
            return SnapshotResources(solver, model.planet, model)
    except Exception as err:  # noqa: BLE001 - classified below
        if _cuda_failure(err):
            raise PlotUnavailableError(
                f"reconstructing the {solver.upper()} model of this run needs "
                f"CUDA (CuPy) for spherical-harmonic synthesis, which is not "
                f"available here: {type(err).__name__}: {err}. Metadata and "
                "host coefficient access remain available"
                + (", and representation='spectral' renders on the CPU"
                   if any((solver, r) in _HOST_ONLY
                          for r in REPRESENTATIONS[solver]) else "")
                + ".") from err
        raise
    raise PlotUnavailableError(f"no rendering is defined for solver {solver!r}")


def _figure_metadata(storage, index: int, representation: str,
                     normalization: str) -> dict:
    meta = dict(storage.metadata)
    return {
        "Software": "palintropos",
        "RunId": meta.get("run_id") or "",
        "RunPath": str(storage.run_dir),
        "Source": storage.coefficient_path.name,
        "Snapshot": str(index),
        "Time": f"{float(storage.times[index]):.9g} s",
        "Representation": representation,
        "Normalization": ("timeline (run-wide limits over every stored "
                          "frame)" if normalization == "timeline"
                          else "frame (limits from this single state)"),
    }


def build_snapshot_timeline(storage, representation: str, *,
                            frame_indices: slice | None = None):
    """Compose the requested representation for the selected stored frames.

    Returns a ``FigureTimeline`` built by the solver's existing builder
    from the capsule's stored arrays (host views, sliced by
    ``frame_indices``). Host-only representations never touch the model.
    """
    solver = storage.solver
    if representation not in REPRESENTATIONS[solver]:
        raise ValueError(
            f"solver {solver!r} has no {representation!r} snapshot "
            f"representation; available: "
            f"{', '.join(REPRESENTATIONS[solver])}")
    window = slice(None) if frame_indices is None else frame_indices
    coefficients = storage.coefficients[window]
    times = storage.times[window]
    if coefficients.shape[0] == 0:
        raise ValueError("this simulation stores no snapshots to render")
    scenario = str(storage.run_config.get("scenario") or solver)
    host_only = (solver, representation) in _HOST_ONLY

    if solver == "bve":
        from tropoi.run.bve.visualization import (
            build_bve_snapshot_timeline_from_data,
            build_bve_spectral_snapshot_timeline_from_data)
        if host_only:
            return build_bve_spectral_snapshot_timeline_from_data(
                coefficients, times, scenario=scenario)
        grids = storage.stored_array("vorticity_grid.npy")[window]
        resources = storage.resources()
        return build_bve_snapshot_timeline_from_data(
            resources.planet, grids, times, scenario=scenario,
            coefficients=coefficients)
    if solver == "swe":
        from tropoi.run.swe.visualization import (
            build_swe_snapshot_timelines_from_data)
        model = None if host_only else storage.resources().model
        return build_swe_snapshot_timelines_from_data(
            model, coefficients, times, scenario=scenario,
            representations=(representation,))[representation]
    if solver == "pe":
        from tropoi.run.pe.snapshot_visualization import (
            build_pe_snapshot_timeline_from_data)
        run_id = storage.metadata.get("run_id")
        return build_pe_snapshot_timeline_from_data(
            storage.resources().model, coefficients, times,
            scenario=scenario, run_id=run_id)
    raise PlotUnavailableError(f"no rendering is defined for solver {solver!r}")


def render_snapshot(storage, index: int, output_path, *,
                    representation: str = "physical",
                    normalization: str = "timeline",
                    renderer=None, metadata: dict | None = None
                    ) -> pathlib.Path:
    """Render one stored snapshot to ``output_path`` and return that path.

    ``representation`` selects the existing product view (``"physical"``,
    or ``"spectral"`` for BVE/SWE); ``normalization`` selects run-wide
    (``"timeline"``, the persisted product's limits) or single-frame
    (``"frame"``) color limits. The image is written atomically by the
    existing renderer; ``metadata`` extends the embedded PNG metadata.
    """
    if normalization not in NORMALIZATIONS:
        raise ValueError(
            f"unknown normalization {normalization!r}; choose from "
            f"{', '.join(NORMALIZATIONS)}")
    count = int(storage.times.shape[0])
    if not 0 <= int(index) < count:
        raise IndexError(
            f"snapshot index {index} is out of range for {count} stored "
            "snapshot(s)")
    index = int(index)
    if normalization == "timeline":
        timeline = build_snapshot_timeline(storage, representation)
        selected = index
    else:
        timeline = build_snapshot_timeline(
            storage, representation, frame_indices=slice(index, index + 1))
        selected = 0
    resolved = timeline.resolve_normalizations()
    frame = resolved.frames[selected]
    if not np.isclose(frame.time_seconds, float(storage.times[index])):
        raise RuntimeError(
            "internal error: the composed frame does not carry the "
            "selected snapshot's time")

    from tropoi.viz.renderers import get_default_renderer
    backend = renderer or get_default_renderer()
    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    embedded = _figure_metadata(storage, index, representation,
                                normalization)
    if metadata:
        embedded.update(metadata)
    written = backend.render_figure(frame.specification, output_path,
                                    metadata=embedded)
    return pathlib.Path(written)
