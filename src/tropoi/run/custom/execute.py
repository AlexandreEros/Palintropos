"""Execute a saved initial state inside the standard run lifecycle.

:func:`run_initial_state` is the public execution path. It resolves the
configuration on the CPU (:class:`CustomRunConfig`), then hands the run to
the same provenance lifecycle every ``tropoi run`` command uses
(``cli.run_lifecycle.execute_with_provenance``): a unique run directory,
``config.json`` and ``manifest.json`` (status ``running`` -> ``completed``
or ``failed``), the additive ``state_schema`` / ``diagnostic_definitions``
blocks, and ``latest_run.txt`` published only on success. The cores and
runners are the preset ones (``run_bve`` / ``run_swe``); nothing about the
equations, the time stepping or the truncation changes.

The capsule also receives a byte-exact copy of the saved state under
``initial_state/`` (written before any numerical work), so the run records
exactly what it started from and can rebuild its model later (for example
for ``Snapshot.plot``) without the original file.
"""
from __future__ import annotations

import pathlib
import shutil
from typing import Optional, Sequence

from tropoi.representation.archive.initial_state import (InitialState,
                                                         load_initial_state)
from tropoi.run.custom.config import CustomRunConfig
from tropoi.spatial.specification import DiscretizationSpec

__all__ = ["INITIAL_STATE_DIR", "run_initial_state"]

#: Capsule subdirectory holding the copy of the run's initial state.
INITIAL_STATE_DIR = "initial_state"


def _notes(cfg: CustomRunConfig) -> dict:
    from tropoi.cli.swe import SWE_MANIFEST_NOTES
    from tropoi.representation.archive.writer import _DEFAULT_MANIFEST_NOTES

    notes = dict(SWE_MANIFEST_NOTES if cfg.solver == "swe"
                 else _DEFAULT_MANIFEST_NOTES)
    state = cfg.initial_state
    notes["initial_state"] = (
        f"custom resolved initial state {state.label!r} "
        f"({state.digest}), copied byte-exact to {INITIAL_STATE_DIR}/; "
        "a valid initial state is not necessarily a balanced one")
    return notes


def _clean_artifacts(cfg: CustomRunConfig, out_dir: pathlib.Path) -> None:
    from tropoi.cli.bve import OverwriteCleanupError

    target = out_dir / INITIAL_STATE_DIR
    if target.exists():
        try:
            shutil.rmtree(target)
        except OSError as err:
            raise OverwriteCleanupError(
                f"could not remove stale {INITIAL_STATE_DIR}/ during "
                f"--overwrite of {out_dir}: {err}") from err
    if cfg.solver == "swe":
        from tropoi.cli.swe import _clean_overwrite_artifacts
    else:
        from tropoi.cli.bve import _clean_overwrite_artifacts
    _clean_overwrite_artifacts(out_dir)


def _resolve_base_dir(cfg: CustomRunConfig, out: str):
    if cfg.solver == "swe":
        from tropoi.cli.swe import _resolve_writable_base_dir
    else:
        from tropoi.cli.bve import _resolve_writable_base_dir
    return _resolve_writable_base_dir(out)


def _persist_initial_state(cfg: CustomRunConfig, out_dir: pathlib.Path) -> None:
    """Copy the state into the capsule and check that the copy round-trips.

    Done first, so even a run that fails records what it was asked to
    integrate.
    """
    state = cfg.initial_state
    state.save(out_dir / INITIAL_STATE_DIR)
    if load_initial_state(out_dir / INITIAL_STATE_DIR).digest != state.digest:
        raise RuntimeError("the capsule copy of the initial state does not "
                           "reproduce its digest")


def _execute_solver(cfg: CustomRunConfig, run_dir, run_config: dict) -> None:
    """Solver callback of the lifecycle: persist the state, then integrate."""
    _persist_initial_state(cfg, run_dir.path)
    _integrate(cfg, run_dir, run_config)


def _integrate(cfg: CustomRunConfig, run_dir, run_config: dict) -> None:
    """Heavy numerical portion (CUDA): build the model and run the core.

    Isolated so the lifecycle and the capsule contents are testable on the
    CPU with this function stubbed.
    """
    out_dir = run_dir.path
    state = cfg.initial_state

    import cupy as cp

    from tropoi.representation.archive.schema import provenance_blocks
    from tropoi.representation.archive.writer import (RUN_STATUS_RUNNING,
                                                      write_run_manifest)

    if cfg.solver == "bve":
        from tropoi.cli.bve import build_planet
        from tropoi.run.bve.runner import run_bve

        planet = build_planet(run_config)
        write_run_manifest(
            out_dir, run_config, run_id=run_dir.run_id,
            experiment=cfg.experiment,
            numerics=planet.so.backend.describe(
                cfg.discretization.product_quadrature),
            status=RUN_STATUS_RUNNING, notes=_notes(cfg),
            **provenance_blocks("bve", run_config))
        run_bve(planet=planet,
                zeta0_lm=cp.asarray(state.coeffs),
                dt_snapshots=cfg.dt_snapshots,
                t_end_days=cfg.duration_days,
                out_dir=out_dir,
                viscosity=cfg.viscosity,
                scenario=cfg.scenario,
                figure_metadata=run_dir.figure_metadata(),
                snapshot_times=cfg.snapshot_times_seconds(),
                plots=cfg.plots,
                snapshot_mode=cfg.snapshot_mode)
        return

    from tropoi.cli.swe import build_swe_model
    from tropoi.run.swe.runner import run_swe
    from tropoi.spatial.states.shallow_water import ShallowWaterState

    model = build_swe_model(run_config, topography_lm=state.topography_lm)
    write_run_manifest(
        out_dir, run_config, run_id=run_dir.run_id, experiment=cfg.experiment,
        numerics=model.planet.so.backend.describe("fine"),
        status=RUN_STATUS_RUNNING, notes=_notes(cfg),
        **provenance_blocks("swe", run_config))
    # run_swe validates the initial state (finite, zero-mean monopoles,
    # strictly positive thickness on the state AND product samplings)
    # before integrating; a violation fails the run, it is never repaired.
    run_swe(model=model,
            state0=ShallowWaterState(cp.asarray(state.coeffs)),
            dt_snapshots=cfg.dt_snapshots,
            t_end_days=cfg.duration_days,
            out_dir=out_dir,
            figure_metadata=run_dir.figure_metadata(),
            snapshot_times=cfg.snapshot_times_seconds(),
            plots=cfg.plots,
            snapshot_mode=cfg.snapshot_mode,
            scenario=cfg.scenario)


def execute_custom_run(cfg: CustomRunConfig) -> pathlib.Path:
    """Run a resolved :class:`CustomRunConfig`; return the run directory."""
    from tropoi.cli.run_lifecycle import execute_with_provenance

    if not isinstance(cfg, CustomRunConfig):
        raise TypeError(
            f"expected a CustomRunConfig, got {type(cfg).__name__}")
    holder: dict = {}

    def solver(c, run_dir, run_config):
        holder["path"] = run_dir.path
        _execute_solver(c, run_dir, run_config)

    execute_with_provenance(
        cfg, solver=solver,
        clean_artifacts=lambda out_dir: _clean_artifacts(cfg, out_dir),
        resolve_base_dir=lambda out: _resolve_base_dir(cfg, out),
        notes=_notes(cfg), solver_name=cfg.solver)
    return holder["path"]


def run_initial_state(initial_state, *,
                      duration_days: float,
                      n_snapshots: Optional[int] = None,
                      snapshot_interval_s: Optional[float] = None,
                      discretization: Optional[DiscretizationSpec] = None,
                      viscosity: Optional[float] = None,
                      plots: Optional[Sequence[str]] = None,
                      out: str = "runs",
                      experiment: Optional[str] = None,
                      overwrite: bool = False) -> pathlib.Path:
    """Integrate a saved (or in-memory) initial state; return the run directory.

    ``initial_state`` is an :class:`InitialState` or the path of a saved
    one. Every option is validated on the CPU before CUDA is initialized
    (see :meth:`CustomRunConfig.resolve`). The run is written under ``out``
    like any ``tropoi run`` capsule and can be opened with
    ``tropoi.representation.archive.open_simulation``.
    """
    if not isinstance(initial_state, InitialState):
        initial_state = load_initial_state(initial_state)
    cfg = CustomRunConfig.resolve(
        initial_state, duration_days=duration_days, n_snapshots=n_snapshots,
        snapshot_interval_s=snapshot_interval_s,
        discretization=discretization, viscosity=viscosity, plots=plots,
        out=out, experiment=experiment, overwrite=overwrite)
    return execute_custom_run(cfg)
