"""``tropoi run swe`` — the shallow-water run executor.

Heavy imports (CuPy, matplotlib) happen inside :func:`_execute_solver`, so
parsing and ``--help`` never initialize CUDA. The provenance lifecycle is
shared with the BVE command (``run/lifecycle.py``).
"""
from __future__ import annotations
from tropoi.run.swe.model import _w5_planet_params, build_swe_model

import pathlib
import re
import shutil
from typing import TYPE_CHECKING, Mapping

from tropoi.run.lifecycle import (execute_with_provenance,
                                                 resolve_writable_base_dir)

if TYPE_CHECKING:
    from tropoi.run.swe.config import SWERunConfig


#: Generated *result* files removed when --overwrite reuses a directory
#: (config.json / manifest.json are lifecycle-managed, never swept).
_GENERATED_RESULT_FILES: tuple[str, ...] = (
    "swe_coeffs.npy",
    "swe_snapshot_times.npy",
    "swe_summary.png",
)

_GENERATED_RESULT_DIRS: tuple[str, ...] = (
    "diagnostics",
    "figures",
    "snapshots",
    "assets",       # post-run figures (tropoi plot) of the replaced run
)

_SNAPSHOT_FRAME_RE = re.compile(r".+_t\d{13}\.\d{9}s\.png\Z")





def _manifest_notes(cfg: "SWERunConfig", topography=None) -> dict:
    """Solver notes, extended with the W5 benchmark record when relevant.

    Before the model exists (lifecycle pre-write) the note carries the
    configured identity; once the terrain is constructed the note also
    records the MEASURED cone projection residual.
    """
    notes = dict(SWE_MANIFEST_NOTES)
    if cfg.scenario == "williamson5":
        tag = ("canonical constants" if cfg.w5_canonical()
               else "NONCANONICAL (W5-derived: physical constants "
                    "overridden)")
        note = ("Williamson et al. (1992) test case 5: zonal flow (u0=20 "
                f"m/s) over the isolated conical mountain; {tag}")
        if topography is not None:
            note += f"; terrain {topography.describe()}"
        notes["benchmark"] = note
    return notes


#: Manifest notes describing the shallow-water solver.
SWE_MANIFEST_NOTES = {
    "equations": "rotating shallow-water equations (vorticity-divergence "
                 "form, perturbation thickness geopotential, optional fixed "
                 "bottom topography entering the divergence tendency as "
                 "-laplacian(phi_s); see physics/shallow_water.py and "
                 "docs/SHALLOW_WATER.md)",
    "timestep_policy": "state-adaptive CFL ceiling "
                       "0.5*cfl_length_scale/max(|u|+sqrt(Phi0+phi)) "
                       "recomputed from every accepted state; steps shorten "
                       "to land exactly on output times and t_end",
    "diagnostics": "see run/swe/diagnostics.py module docstring for definitions",
}


def _resolve_writable_base_dir(requested_out: str) -> tuple[pathlib.Path, bool]:
    # The fallback prefix is an on-disk output-path convention, not a command
    # name: it is deliberately left at its historical spelling so existing
    # SWE fallback directories keep matching. See the same note on the
    # "psx-bve-" prefix in run/bve/execution.py.
    return resolve_writable_base_dir(requested_out,
                                     fallback_prefix="aeolus-swe-")


def _clean_overwrite_artifacts(out_dir: pathlib.Path) -> None:
    """Remove known SWE-generated outputs before a reused run overwrites them.

    Strictly scoped to Palintropos-generated result artifacts, like the BVE
    variant; raises OverwriteCleanupError (via the shared type in run/bve/execution.py)
    if any known stale artifact cannot be removed.
    """
    from tropoi.run.bve.execution import OverwriteCleanupError

    failures: list[str] = []
    targets: list[pathlib.Path] = []
    for name in _GENERATED_RESULT_DIRS:
        target = out_dir / name
        if target.is_dir():
            targets.append(target)
    for name in _GENERATED_RESULT_FILES:
        target = out_dir / name
        if target.exists():
            targets.append(target)
    targets.extend(
        child for child in out_dir.iterdir()
        if child.is_file() and _SNAPSHOT_FRAME_RE.match(child.name))
    for target in targets:
        try:
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
        except OSError as err:
            failures.append(f"{target.name}: {err}")
    if failures:
        raise OverwriteCleanupError(
            "could not remove stale generated artifact(s) during --overwrite "
            f"of {out_dir}: " + "; ".join(failures))


def _execute_solver(cfg: "SWERunConfig", run_dir, run_config: dict) -> None:
    """Heavy numerical portion of a run: build planet + model, drive the solver."""
    from tropoi.representation.archive.writer import (RUN_STATUS_RUNNING,
                                              write_run_manifest)
    from tropoi.spatial.initialization.swe import make_swe_ic
    from tropoi.run.swe.runner import run_swe
    from tropoi.representation.archive.schema import provenance_blocks

    out_dir = run_dir.path
    model = build_swe_model(run_config)
    planet = model.planet
    topography = model.topography if model.has_topography else None
    state0 = make_swe_ic(cfg.scenario, model)

    # Rewrite manifest now that we know the backend/product-sampling
    # provenance (and, for W5, the measured terrain projection). Status
    # stays 'running' until the runner returns.
    write_run_manifest(out_dir, run_config,
                       run_id=run_dir.run_id, experiment=cfg.experiment,
                       numerics=planet.so.backend.describe("fine"),
                       status=RUN_STATUS_RUNNING,
                       notes=_manifest_notes(cfg, topography),
                       **provenance_blocks("swe", run_config))

    run_swe(model=model,
            state0=state0,
            dt_snapshots=cfg.dt_snapshots,
            t_end_days=cfg.duration_days,
            out_dir=out_dir,
            figure_metadata=run_dir.figure_metadata(),
            snapshot_times=cfg.snapshot_times_seconds(),
            plots=cfg.plots,
            snapshot_mode=cfg.snapshot_mode,
            scenario=cfg.scenario)


def execute_run(cfg: "SWERunConfig") -> int:
    """Execute one resolved shallow-water run inside the shared lifecycle."""
    return execute_with_provenance(
        cfg,
        solver=lambda c, run_dir, run_config: _execute_solver(
            c, run_dir, run_config),
        clean_artifacts=lambda out_dir: _clean_overwrite_artifacts(out_dir),
        resolve_base_dir=lambda out: _resolve_writable_base_dir(out),
        notes=_manifest_notes(cfg),
        solver_name="swe")
