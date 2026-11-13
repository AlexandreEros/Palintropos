"""Registered runtime-capture scenarios: small, inexpensive, lifecycle-labelled.

Each scenario is a driver that :func:`tools.architecture.runtime_capture.capture`
runs under the recorder. A driver uses only public or production entry points
of the package, labels lifecycle phases with :func:`runtime_capture.phase`,
and records what it can verify (configuration identity, exact resume, which
heavy modules were imported) in the trace's ``verification`` block.

Drivers never change numerics. The only instrumentation they add is a thin
Python wrapper around a few *instance* methods (``stepper.step``,
``_sample``, ``_write_checkpoint``) that switches the phase label and then
calls the original method; such calls therefore appear as ``indirect`` caller
lines in the trace (a non-package frame sat between caller and callee).

List them with ``python -m tools.architecture.runtime_capture --list``.
"""
from __future__ import annotations

import dataclasses
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Callable

from . import runtime_capture as rt

REPO_ROOT = Path(__file__).resolve().parents[2]
#: The smallest published BVE capsule (docs/runs/README.md) and the committed
#: Held-Suarez run fixture (tests/fixtures/README.md); both open on a CPU.
BVE_CAPSULE = ("docs/runs/20260713T024849Z_two-vortices-north-pole_norot_r4_l21_dt12h_"
               "4a840226")
HS_FIXTURE = "tests/fixtures/hs-T21L10-dev4"
#: The published primitive-equation capsule with prepared terrain.
PE_CAPSULE = ("docs/runs/20261027T175327Z_orographic-thermal-wave_rot24p62h_r3_l21_"
              "dt11080p3s_3a46f966_71c5441a")
#: Steps after resume compared bitwise with the uninterrupted run.
RESUME_STEPS = 3


def default_scratch() -> str:
    return str(Path(tempfile.gettempdir()) / "palintropos-atlas-scenarios")


@dataclasses.dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    needs_gpu: bool
    boundaries: tuple[str, ...]
    factory: Callable[[Path], Callable[[rt.Notes], None]]

    def prepare(self, scratch: Path) -> Callable[[rt.Notes], None]:
        directory = Path(scratch) / self.name
        shutil.rmtree(directory, ignore_errors=True)
        directory.mkdir(parents=True, exist_ok=True)
        return self.factory(directory)


def _heavy_modules() -> dict:
    return {m: m in sys.modules for m in ("cupy", "matplotlib", "scipy")}


def _import_claim(checks: dict, claim: str, holds: bool) -> dict:
    """A verification about module imports, valid only in a fresh process.

    ``--measure-overhead`` runs the scenario twice in this process before the
    traced run, and an import made then persists; such a capture cannot
    support an import claim, and says so instead of passing."""
    if checks["heavy_modules_at_start"]["cupy"]:
        return {"passed": None, "claim": claim, "checks": checks,
                "not_determinable": "CuPy was already imported when the traced run started "
                                    "(earlier runs in this process); capture without "
                                    "--measure-overhead to test this claim"}
    return {"passed": holds, "claim": claim, "checks": checks}


def _pilot_config():
    """The pilot's own ``tiny`` resolution of the heated Earth case (reused, not copied)."""
    from tools.colab import radiative_pilot as pilot
    return pilot.config("earth", "tiny"), {
        "derived_from": "tools/colab/radiative_pilot.py config('earth', 'tiny')",
        "pilot_resolution": dict(pilot.RESOLUTIONS["tiny"]),
        "pilot_case": list(pilot.CASES["earth"])}


def _config_identity(cfg, extra: dict) -> dict:
    return {"experiment_id": cfg.experiment_id, "sha256": cfg.sha256(),
            "hash_method": "RadiativeConfig.sha256() (the run's own configuration identity, "
                           "also stored in config.json and every checkpoint)",
            "config": cfg.to_dict(), **extra}


# ---------------------------------------------------------------------------
# setup-cpu
# ---------------------------------------------------------------------------

def _setup_cpu(scratch: Path):
    def drive(notes: rt.Notes) -> None:
        import math

        import numpy as np
        checks: dict = {}
        with rt.phase("preset-resolution"):
            cfg, origin = _pilot_config()
            sha = cfg.sha256()
            again = type(cfg).from_dict(cfg.to_dict())
            checks["config_round_trip_same_sha256"] = again.sha256() == sha
        with rt.phase("validation"):
            try:
                cfg.with_(radiation_interval=0)
                checks["invalid_radiation_interval_rejected"] = False
            except ValueError as exc:
                checks["invalid_radiation_interval_rejected"] = True
                checks["rejection_message"] = str(exc)[:160]
        with rt.phase("component-construction-cpu"):
            from tropoi.run.radiative.model import build_physics, build_sigma
            sigma = build_sigma(cfg)
            build_physics(cfg)
            checks["sigma_levels"] = int(sigma.nlev)
        with rt.phase("planet-specification"):
            from tropoi.spatial.specification import PlanetSpec
            from tropoi.spatial.williamson5 import W5_RADIUS_M
            bench = PlanetSpec.ideal_sphere(W5_RADIUS_M, 86164.0)
            back = PlanetSpec.from_dict(bench.to_dict())
            params = back.to_parameters()
            checks["ideal_sphere_radius_exact"] = float(params.radius) == float(W5_RADIUS_M)
            checks["ideal_sphere_oblateness_zero"] = float(params.oblateness) == 0.0
            earth = PlanetSpec.earth_like(24.0).to_parameters()
            checks["earth_like_dynamical_radius_m"] = float(earth.radius)
            checks["earth_like_is_oblate_volumetric_mean"] = float(earth.radius) < 6.371e6
        with rt.phase("terrain-preparation-cpu"):
            from tropoi.spatial.terrain.raster import ElevationRaster

            def height(lat, lon):
                return 1000.0 * np.cos(np.radians(lat)) ** 2
            raster = ElevationRaster.from_function(height, resolution_deg=10.0,
                                                   name="atlas-scenario")
            checks["raster_shape"] = list(np.shape(raster.elevation_m))
        with rt.phase("physical-descriptor-inspection"):
            # Not a production path: PLANETS has no consumer in src/ (see the
            # migration proposal). Recorded so its data shape is inspectable.
            from tropoi.physics.planets import PLANETS
            body = PLANETS["earth"]
            regime = body.regime()
            checks["planets_earth_radius_m"] = body.radius
            checks["radiative_earth_radius_m"] = cfg.radius
            checks["planets_vs_radiative_radius_equal"] = math.isclose(body.radius, cfg.radius)
            checks["regime_keys"] = sorted(regime)[:8]
        checks["heavy_modules_imported"] = _heavy_modules()
        notes.configuration = _config_identity(cfg, origin)
        notes.verification = {"passed": bool(checks["config_round_trip_same_sha256"]
                                              and checks["invalid_radiation_interval_rejected"]
                                              and checks["ideal_sphere_radius_exact"]
                                              and checks["ideal_sphere_oblateness_zero"]),
                              "checks": checks}
        notes.synchronization_note = "CPU-only scenario; no device is used."
    return drive


# ---------------------------------------------------------------------------
# radiative-tiny-lifecycle
# ---------------------------------------------------------------------------

def _phased(run, prefix: str = "") -> Callable[[], None]:
    """Phase labels for the run's own startup/stepping/output/checkpoint calls.

    Returns a function that removes the wrappers again."""
    stepper = run.stepper
    step, sample, checkpoint = stepper.step, run._sample, run._write_checkpoint

    def phased_step():
        label = "startup-step" if stepper.x_prev is None else "ordinary-step"
        with rt.phase(prefix + label):
            return step()

    def phased_sample(day):
        with rt.phase(prefix + "daily-output"):
            return sample(day)

    def phased_checkpoint():
        with rt.phase(prefix + "checkpoint"):
            return checkpoint()
    stepper.step = phased_step
    run._sample = phased_sample
    run._write_checkpoint = phased_checkpoint

    def restore():
        del stepper.step, run._sample, run._write_checkpoint
    return restore


def _stepper_snapshot(run) -> dict:
    """Host copies of everything exact resume must reproduce (verification only)."""
    import cupy as cp
    import numpy as np
    sd = dict(run.stepper.state_dict())
    out = {"x_curr": cp.asnumpy(sd.pop("x_curr")),
           "x_prev": None if sd.get("x_prev") is None else cp.asnumpy(sd.pop("x_prev"))}
    sd.pop("x_prev", None)
    out["stepper_scalars"] = {k: v for k, v in sd.items()
                              if isinstance(v, (int, float, str, bool, type(None)))}
    if run.coupling is not None:
        c = run.coupling.state_dict()
        ts = c.get("surface_temperature")
        out["surface_temperature"] = None if ts is None else np.asarray(
            cp.asnumpy(ts) if hasattr(ts, "device") else ts)
        out["budget_time"] = c.get("budget_time")
        out["budget"] = {k: float(np.asarray(cp.asnumpy(v) if hasattr(v, "device") else v)
                                  .sum()) for k, v in (c.get("budget") or {}).items()}
    return out


def _compare(a: dict, b: dict) -> dict:
    import numpy as np
    out = {}
    for key in ("x_curr", "x_prev", "surface_temperature"):
        x, y = a.get(key), b.get(key)
        if x is None or y is None:
            out[key] = {"both_absent": x is None and y is None}
            continue
        out[key] = {"bitwise_equal": bool(np.array_equal(x, y)),
                    "max_abs_difference": float(np.max(np.abs(x - y))) if x.shape == y.shape
                    else None, "shape": list(x.shape), "dtype": str(x.dtype)}
    out["stepper_scalars_equal"] = a["stepper_scalars"] == b["stepper_scalars"]
    out["stepper_scalars"] = a["stepper_scalars"]
    out["budget_time_equal"] = a.get("budget_time") == b.get("budget_time")
    out["budget_sums_equal"] = a.get("budget") == b.get("budget")
    return out


def _radiative_lifecycle(scratch: Path):
    def drive(notes: rt.Notes) -> None:
        from tropoi.run.radiative.experiment import RadiativeRun
        cfg, origin = _pilot_config()
        run_dir = scratch / cfg.experiment_id
        with rt.phase("construction"):
            run = RadiativeRun(cfg, run_dir, allow_dirty=True, log=lambda s: None)
        unwrap = _phased(run)
        with rt.phase("production-loop"):
            status = run.run(until_day=1)
        unwrap()
        with rt.phase("resume-construction"):
            resumed = RadiativeRun(cfg, run_dir, allow_dirty=True, log=lambda s: None)
        _phased(resumed, prefix="resumed-")  # every resumed step is an ordinary step
        with rt.phase("resumed-stepping"):
            for _ in range(RESUME_STEPS):
                resumed.stepper.step()
        with rt.phase("reference-continuation"):
            for _ in range(RESUME_STEPS):
                run.stepper.step()
        with rt.phase("verification"):
            comparison = _compare(_stepper_snapshot(run), _stepper_snapshot(resumed))
        events = [line for line in (run_dir / "events.jsonl").read_text("utf-8").splitlines()]
        exact = all(v.get("bitwise_equal", v.get("both_absent", False))
                    for k, v in comparison.items() if isinstance(v, dict)
                    and k in ("x_curr", "x_prev", "surface_temperature"))
        notes.configuration = _config_identity(cfg, origin)
        notes.verification = {
            "passed": bool(exact and comparison["stepper_scalars_equal"]
                           and comparison["budget_time_equal"]
                           and comparison["budget_sums_equal"]),
            "claim": f"a run resumed from its day-1 checkpoint and advanced {RESUME_STEPS} "
                     f"steps equals the uninterrupted run advanced {RESUME_STEPS} steps "
                     "(time levels, step count, surface temperature, coupling budget)",
            "run_status": {k: v for k, v in status.items() if k != "wall_seconds"
                           and k != "seconds_per_step"},
            "steps_per_day": cfg.steps_per_day, "radiation_interval": cfg.radiation_interval,
            "comparison": comparison,
            "events": [e[:200] for e in events][:12]}
        notes.synchronization_note = (
            "The production loop synchronizes where production code does: day-boundary "
            "finiteness/validation checks (bool of a device reduction, invisible to the "
            "collector as a call), the daily sample and checkpoint (host transfers). The "
            "verification phase deliberately copies both runs' states to the host. The "
            "collector itself adds none.")
    return drive


# ---------------------------------------------------------------------------
# saved-run analysis
# ---------------------------------------------------------------------------

def _saved_run_cpu(scratch: Path):
    def drive(notes: rt.Notes) -> None:
        checks: dict = {"heavy_modules_at_start": _heavy_modules()}
        from tropoi.representation.archive import open_simulation
        with rt.phase("open-capsule"):
            sim = open_simulation(REPO_ROOT / BVE_CAPSULE)
        with rt.phase("inspect-snapshot"):
            n = len(sim)
            times = sim.times
            snap = sim[min(2, n - 1)]
            state = snap.state
            checks.update({"frames": n, "solver": sim.solver, "last_time_s": float(times[-1]),
                           "state_type": type(state).__qualname__,
                           "field_names": list(sim.field_names)})
        checks["cupy_after_inspection"] = "cupy" in sys.modules
        with rt.phase("spectral-figure"):
            png = snap.plot(scratch / "frame-spectral.png", representation="spectral")
            checks["spectral_png_bytes"] = Path(png).stat().st_size
        checks["cupy_after_spectral_figure"] = "cupy" in sys.modules
        with rt.phase("open-hs-fixture"):
            hs = open_simulation(REPO_ROOT / HS_FIXTURE)
        with rt.phase("hs-climate-reduction"):
            from tropoi.representation.visual.held_suarez_climate import reduce_experiment
            result = reduce_experiment(hs.storage)
            checks["hs_reduction_type"] = type(result).__qualname__
        with rt.phase("hs-climate-figure"):
            out = hs.plot(scratch / "hs-climate.png", dpi=60)
            checks["hs_png_bytes"] = Path(out).stat().st_size
        checks["heavy_modules_at_end"] = _heavy_modules()
        notes.configuration = {"inputs": [BVE_CAPSULE, HS_FIXTURE],
                               "hash_method": "the capsules' own receipts (SHA256SUMS / "
                                              "manifest) are verified by the package on open"}
        notes.verification = _import_claim(
            checks, "opening, inspecting, reducing and plotting saved runs never imports CuPy",
            not checks["heavy_modules_at_end"]["cupy"])
        notes.synchronization_note = "CPU-only read path; any CuPy import would be recorded."
    return drive


def _saved_run_view(scratch: Path):
    def drive(notes: rt.Notes) -> None:
        checks: dict = {"heavy_modules_at_start": _heavy_modules()}
        from tropoi.representation.archive import open_simulation
        with rt.phase("open-capsule"):
            sim = open_simulation(REPO_ROOT / PE_CAPSULE)
            checks["solver"] = sim.solver
            checks["frames"] = len(sim)
        with rt.phase("html-view"):
            page = sim.view(scratch / "view.html", times=[0], resolution_deg=10.0,
                            open_browser=False)
            checks["html_bytes"] = Path(page).stat().st_size
        checks["cupy_after_html_view"] = "cupy" in sys.modules
        with rt.phase("snapshot-figure"):
            png = sim[0].plot(scratch / "frame.png")
            checks["figure_png_bytes"] = Path(png).stat().st_size
        checks["heavy_modules_at_end"] = _heavy_modules()
        notes.configuration = {"inputs": [PE_CAPSULE]}
        notes.verification = _import_claim(
            checks, "the 3-D page of a saved PE run is built without CuPy "
                    "(cupy_after_html_view); the snapshot figure is recorded as observed",
            not checks["cupy_after_html_view"])
        notes.synchronization_note = ("Synthesis for figures may run on the device; any "
                                      "visible transfer or synchronization is counted.")
    return drive


def _custom_swe_lifecycle(scratch: Path):
    def drive(notes: rt.Notes) -> None:
        import numpy as np
        checks: dict = {}
        with rt.phase("specification"):
            from tropoi.run.custom import (DiscretizationSpec, FluidLayerSpec, PlanetSpec,
                                           load_initial_state, prepare_planet,
                                           prepare_swe_state, run_initial_state)
            from tropoi.spatial.terrain.raster import ElevationRaster
            from tropoi.spatial.williamson5 import W5_RADIUS_M
            planet = PlanetSpec.ideal_sphere(W5_RADIUS_M, 86164.0)
            disc = DiscretizationSpec.gauss_latlon(10, 16, 32)
            fluid = FluidLayerSpec(gravity_m_s2=9.80616, mean_depth_m=5960.0)
        with rt.phase("terrain-raster-cpu"):
            raster = ElevationRaster.from_function(
                lambda lat, lon: 600.0 * np.exp(-((lat - 30.0) ** 2 + (lon - 90.0) ** 2)
                                                / 400.0),
                resolution_deg=5.0, name="atlas-gaussian-hill")
        with rt.phase("terrain-projection"):
            prepared = prepare_planet(planet, disc, raster, smoothing=0.5,
                                      label="atlas-hill")
            checks["terrain_lm_shape"] = list(np.shape(prepared.topography_lm))
        with rt.phase("initial-condition"):
            omega = 2.0 * np.pi / 86164.0
            state = prepare_swe_state(
                planet, disc, fluid, prepared,
                streamfunction=lambda lat, lon: -W5_RADIUS_M * 20.0 * np.sin(lat),
                label="atlas-solid-body").state
            checks["solid_body_omega_rad_s"] = omega
        with rt.phase("persist-initial-state"):
            path = state.save(scratch / "initial-state")
        checks["cupy_already_imported_at_load"] = "cupy" in sys.modules  # same process
        with rt.phase("load-initial-state"):
            loaded = load_initial_state(path)
            checks["loaded_solver"] = loaded.solver
        with rt.phase("custom-run"):
            run_dir = run_initial_state(loaded, duration_days=0.25, n_snapshots=2,
                                        plots=[], out=str(scratch / "runs"),
                                        experiment="atlas-custom-swe")
        with rt.phase("open-run"):
            from tropoi.representation.archive import open_simulation
            sim = open_simulation(run_dir)
            checks["frames"] = len(sim)
            checks["solver"] = sim.solver
        notes.configuration = {"planet": planet.to_dict(), "discretization": disc.to_dict(),
                               "fluid": fluid.to_dict(),
                               "hash_method": "the capsule's own run id and manifest "
                                              "(config hash of the scientific subset)",
                               "run_directory": run_dir.name}
        notes.verification = {"passed": checks["frames"] >= 2 and checks["solver"] == "swe",
                              "claim": "a function-defined SWE state over projected terrain is "
                                       "saved, reloaded and integrated into an openable capsule",
                              "checks": checks}
        notes.synchronization_note = ("Terrain projection, state preparation and the run use "
                                      "the device; snapshots are copied to the host by the "
                                      "production code.")
    return drive


SCENARIOS: dict[str, Scenario] = {s.name: s for s in (
    Scenario("custom-swe-lifecycle",
             "Custom initial condition and terrain on a GPU: PlanetSpec/DiscretizationSpec/"
             "FluidLayerSpec, a sampled terrain raster projected with the model's transform, "
             "an SWE state from a streamfunction over that terrain, save, reload, a 6-hour "
             "custom run at l_max 10 and opening its capsule.",
             True,
             ("tropoi.run.custom.planet.prepare_planet", "tropoi.run.custom.prepare.prepare_*",
              "tropoi.run.custom.execute.*", "tropoi.representation.archive.initial_state.*",
              "tropoi.temporal.models.*", "tropoi.spatial.specification.*.to_parameters"),
             _custom_swe_lifecycle),
    Scenario("setup-cpu",
             "Configuration and setup on a CPU: radiative preset resolution and validation, "
             "CPU column-physics construction, PlanetSpec persistence, terrain sampling, and "
             "a direct look at the PLANETS descriptors.",
             False,
             ("tropoi.run.radiative.config.*", "tropoi.run.radiative.model.build_*",
              "tropoi.spatial.specification.PlanetSpec.*",
              "tropoi.spatial.environment.PlanetaryParameters.*",
              "tropoi.spatial.terrain.raster.ElevationRaster.from_function",
              "tropoi.physics.planets.PlanetaryBody.regime"),
             _setup_cpu),
    Scenario("radiative-tiny-lifecycle",
             "Heated dry Earth at the pilot's tiny resolution (l_max 10, 16x32, 25 levels): "
             "construction, RK4 startup vs leapfrog stepping, one simulated day through the "
             "production loop with its daily sample and checkpoint, then exact resume from "
             "that checkpoint compared bitwise with the uninterrupted run.",
             True,
             ("tropoi.run.radiative.experiment.RadiativeRun.*",
              "tropoi.run.radiative.model.*",
              "tropoi.temporal.column_coupling.ColumnPhysicsCoupling.*",
              "tropoi.temporal.semi_implicit.SemiImplicitLeapfrogStepper.step",
              "tropoi.temporal.semi_implicit.SemiImplicitLeapfrogStepper._startup",
              "tropoi.temporal.semi_implicit.SemiImplicitLeapfrogStepper.initialize",
              "tropoi.temporal.semi_implicit.SemiImplicitLeapfrogStepper.load_state_dict",
              "tropoi.temporal.semi_implicit.SemiImplicitLeapfrogStepper.state_dict",
              "tropoi.physics.column_model.ColumnPhysics.compute",
              "tropoi.spatial.planet.Planet.generate",
              "tropoi.run.held_suarez.checkpoint.*"),
             _radiative_lifecycle),
    Scenario("saved-run-cpu",
             "Saved-run analysis on a CPU: open a published BVE capsule, inspect a snapshot, "
             "draw its spectral figure; open the committed Held-Suarez fixture, reduce its "
             "climate statistics and draw the climate figure.",
             False,
             ("tropoi.representation.archive.capsule.*", "tropoi.temporal.simulation.*",
              "tropoi.representation.archive.held_suarez.*",
              "tropoi.representation.visual.held_suarez_climate.reduce_experiment"),
             _saved_run_cpu),
    Scenario("saved-run-view",
             "Visualization of a published primitive-equation capsule with terrain: the "
             "offline 3-D atmosphere page and a snapshot figure; records whether CUDA is "
             "initialized for them.",
             False,
             ("tropoi.representation.archive.capsule.*", "tropoi.temporal.simulation.*",
              "tropoi.representation.visual.*"),
             _saved_run_view),
)}
