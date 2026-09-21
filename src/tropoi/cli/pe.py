"""``tropoi run pe`` — the dry primitive-equation run executor.

Heavy imports (CuPy, matplotlib) happen inside :func:`_execute_solver`, so
parsing and ``--help`` never initialize CUDA. The provenance lifecycle is the
same one the BVE and SWE commands share (``cli/run_lifecycle.py``): the PE run
capsule uses the identical run-id, atomic-write, status, latest-pointer, and
manifest machinery — there is no weaker PE-specific persistence path.
"""
from __future__ import annotations

import pathlib
import shutil
from typing import TYPE_CHECKING, Mapping

from tropoi.cli.run_lifecycle import (execute_with_provenance,
                                                 resolve_writable_base_dir)

if TYPE_CHECKING:
    from tropoi.run.pe.config import PERunConfig


#: Generated *result* files removed when --overwrite reuses a directory
#: (config.json / manifest.json are lifecycle-managed, never swept).
_GENERATED_RESULT_FILES: tuple[str, ...] = (
    "pe_coeffs.npy",
    "pe_snapshot_times.npy",
    "pe_summary.png",
)

_GENERATED_RESULT_DIRS: tuple[str, ...] = (
    "diagnostics",
    "figures",
    "snapshots",
)

#: Manifest notes describing the dry primitive-equation solver and the exact
#: stored-array contract. The scientific run_config (nlev, sigma_interfaces,
#: dt_seconds, r_dry/cp_dry, IC parameters, snapshot_times, ...) is written
#: separately by the shared lifecycle; these notes document the invariants a
#: reader needs to interpret the capsule.
PE_MANIFEST_NOTES = {
    "equations": "dry hydrostatic primitive equations in vorticity-divergence "
                 "form on a sigma-coordinate vertical grid (see "
                 "physics/primitive_equations.py and "
                 "docs/PRIMITIVE_EQUATIONS_DESIGN.md)",
    "coefficient_ordering": "pe_coeffs.npy has shape "
                            "(n_snapshots, 3*nlev+1, l_max+1, l_max+1); axis 1 "
                            "is [zeta_1..zeta_K, delta_1..delta_K, T_1..T_K, "
                            "ln_ps] top-to-bottom (K = nlev), and the trailing "
                            "two axes are the (degree, order) spherical-harmonic "
                            "coefficient block",
    "timestep_policy": "user-supplied FIXED timestep dt_seconds (no adaptive "
                       "CFL controller); individual steps are shortened only to "
                       "land exactly on requested output times and t_end. No "
                       "forcing, no hyperdiffusion/filters, no semi-implicit "
                       "terms, no adaptive timestepping",
    "diagnostics": "see run/pe/diagnostics.py module docstring; total_mass is "
                   "the integral of p_s dA (mass proxy), and no total-energy-"
                   "conservation claim is made",
    "topography": "optional fixed surface topography: a surface elevation "
                  "h_s (m) band-limited at the DEALIASED product-truncation "
                  "cut floor(2*l_max/3) (physics/topography.py with l_cut; "
                  "content above the cut cannot be balanced by the dealiased "
                  "pressure-gradient pathway and is rejected), consumed as "
                  "the surface geopotential Phi_s = gravity * h_s (m^2/s^2) "
                  "that anchors the hydrostatic geopotential of every "
                  "column; sigma remains p/p_s. Runs without a 'topography' "
                  "key in run_config have exactly zero Phi_s. Terrain is "
                  "reconstructed deterministically from the resolved "
                  "run_config (which participates in the scientific hash); "
                  "no terrain arrays are persisted",
}


def _resolve_writable_base_dir(requested_out: str) -> tuple[pathlib.Path, bool]:
    # The fallback prefix is an on-disk output-path convention, not a command
    # name: it is deliberately left at its historical spelling so existing
    # PE fallback directories keep matching. See the same note on the
    # "psx-bve-" prefix in cli/bve.py.
    return resolve_writable_base_dir(requested_out,
                                     fallback_prefix="aeolus-pe-")


def _clean_overwrite_artifacts(out_dir: pathlib.Path) -> None:
    """Remove known PE-generated outputs before a reused run overwrites them.

    Strictly scoped to Palintropos-generated result artifacts, like the BVE/SWE
    variants; raises OverwriteCleanupError if any known stale artifact cannot
    be removed.
    """
    from tropoi.cli.bve import OverwriteCleanupError

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


def build_pe_model(run_config: Mapping):
    """The primitive-equation model (planet + sigma grid + terrain) of a run.

    Built from the resolved run configuration dict — exactly the values
    ``config.json`` persists — so a saved capsule's model can be
    reconstructed later without re-running preset validation. Topography
    is scenario-independent fixed model data, reconstructed
    deterministically from the resolved configuration (which participates
    in the scientific hash) exactly like the SWE path; the PE model
    consumes the derived surface geopotential Phi_s = g * h_s. PE terrain
    is band-limited at the model's dealiased product-truncation cut
    (l_cut), the resolved band of the dealiased pressure-gradient
    pathway; the projection-residual gate validates the terrain actually
    used and fails loudly when the mountain is too narrow for the cut.
    Imports CuPy; call only after validation.
    """
    from tropoi.spatial.environment import PlanetaryParameters
    from tropoi.spatial.planet import Planet
    from tropoi.temporal.tendencies.primitive_equations import (
        PrimitiveEquationsModel)
    from tropoi.spatial.sigma_coordinate import SigmaGrid
    from tropoi.spatial.terrain.topography import Topography

    planet = Planet.generate(
        params=PlanetaryParameters.from_earth_like(
            day_hours=float(run_config["day_hours"]),
            radius_earth_units=float(run_config["radius_earth_units"])),
        grid_resolution=int(run_config["resolution"]),
        l_max=int(run_config["lmax"]),
        product_quadrature="fine",
        grid_type=run_config.get("grid", "geodesic"),
        nlat=int(run_config["nlat"]),
        nlon=int(run_config["nlon"]),
    )
    topography_kind = run_config.get("topography", "flat")
    if topography_kind == "mountain":
        from tropoi.temporal.tendencies.primitive_equations import (
            product_truncation_cut)
        topography = Topography.mountain(
            planet,
            height_m=float(run_config["mountain_height_m"]),
            lat_deg=float(run_config["mountain_lat_deg"]),
            lon_deg=float(run_config["mountain_lon_deg"]),
            width_deg=float(run_config["mountain_width_deg"]),
            l_cut=product_truncation_cut(int(run_config["lmax"])))
        surface_geopotential_lm = topography.surface_geopotential_lm(
            float(run_config["gravity"]))
    elif topography_kind == "flat":
        surface_geopotential_lm = None
    else:
        raise ValueError(
            f"unknown topography {topography_kind!r} in the run configuration")
    interfaces = run_config.get("sigma_interfaces")
    if interfaces is None:
        sigma = SigmaGrid.uniform(int(run_config["nlev"]))
    else:
        sigma = SigmaGrid(tuple(float(s) for s in interfaces))
    return PrimitiveEquationsModel(
        planet, sigma, r_dry=float(run_config["r_dry"]),
        cp_dry=float(run_config["cp_dry"]),
        surface_geopotential_lm=surface_geopotential_lm)


def _execute_solver(cfg: "PERunConfig", run_dir, run_config: dict) -> None:
    """Heavy numerical portion: build planet + PE model, drive the solver."""
    from tropoi.run.bve.io import (RUN_STATUS_RUNNING,
                                              write_run_manifest)
    from tropoi.spatial.initialization.pe import make_pe_ic
    from tropoi.run.pe.runner import run_pe
    from tropoi.representation.archive.schema import provenance_blocks

    out_dir = run_dir.path
    model = build_pe_model(run_config)
    planet = model.planet
    state0 = make_pe_ic(cfg.scenario, model, temperature=cfg.temperature,
                        surface_pressure=cfg.surface_pressure,
                        thermal_amplitude=cfg.thermal_amplitude)

    # Rewrite manifest now that we know the backend/product-sampling
    # provenance. Status stays 'running' until the runner returns.
    write_run_manifest(out_dir, run_config,
                       run_id=run_dir.run_id, experiment=cfg.experiment,
                       numerics=planet.so.backend.describe("fine"),
                       status=RUN_STATUS_RUNNING, notes=PE_MANIFEST_NOTES,
                       **provenance_blocks("pe", run_config))

    run_pe(model=model,
           state0=state0,
           dt_seconds=cfg.dt_seconds,
           t_end_days=cfg.duration_days,
           out_dir=out_dir,
           snapshot_times=cfg.snapshot_times_seconds(),
           snapshot_mode=cfg.snapshot_mode,
           dt_snapshots=cfg.dt_snapshots,
           figure_metadata=run_dir.figure_metadata(),
           plots=cfg.plots,
           scenario=cfg.scenario)


def execute_run(cfg: "PERunConfig") -> int:
    """Execute one resolved primitive-equation run inside the shared lifecycle."""
    return execute_with_provenance(
        cfg,
        solver=lambda c, run_dir, run_config: _execute_solver(
            c, run_dir, run_config),
        clean_artifacts=lambda out_dir: _clean_overwrite_artifacts(out_dir),
        resolve_base_dir=lambda out: _resolve_writable_base_dir(out),
        notes=PE_MANIFEST_NOTES,
        solver_name="pe")
