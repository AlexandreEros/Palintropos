"""Host tests for the saved-run interface (no CUDA, no Matplotlib).

Synthetic tiny capsules exercise every stored layout (BVE ``zeta``; SWE
``zeta, delta, phi``; PE ``zeta, delta, temperature, ln_ps`` with a
NONUNIFORM sigma grid), named read-only views, storage sharing, snapshot
ownership/lifetime, the legacy missing-solver BVE format, empty saved
sequences, malformed headers/time axes, invalid selections, unknown schema
versions, and invalid accessed coefficients. Real saved capsules establish
compatibility separately (test_saved_run_compat.py).
"""
from __future__ import annotations

import gc
import json
import pathlib
import sys

import numpy as np
import pytest

from tropoi.representation.archive import (CapsuleError, CapsuleLayoutError,
                                           SchemaError, StateSchema,
                                           UnknownSchemaVersionError,
                                           open_simulation)
from tropoi.representation.archive.schema import (STATE_SCHEMA_VERSION,
                                                  provenance_blocks,
                                                  state_schema_for)
from tropoi.spatial.modes import (FieldSpec, SpectralConventionError,
                                  SpectralModes, SpectralState)
from tropoi.temporal.simulation import Simulation

L_MAX = 3
N = L_MAX + 1
SIGMA = (0.0, 0.1, 0.35, 0.7, 1.0)   # nonuniform, 4 levels
NLEV = len(SIGMA) - 1


# ---------------------------------------------------------------------------
# Synthetic capsules
# ---------------------------------------------------------------------------

def _triangular(rng, rows_shape, *, monopole=0.0):
    """Random complex coefficients honoring the stored conventions."""
    coeffs = rng.normal(size=rows_shape + (N, N)) + \
        1j * rng.normal(size=rows_shape + (N, N))
    l = np.arange(N)[:, None]
    m = np.arange(N)[None, :]
    coeffs[..., m > l] = 0.0
    coeffs[..., :, 0] = coeffs[..., :, 0].real
    coeffs[..., 0, 0] = monopole
    return coeffs


def _run_config(solver: str, **extra) -> dict:
    base = {"lmax": L_MAX, "grid": "geodesic", "resolution": 2, "nlat": 8,
            "nlon": 16, "day_hours": 24.0, "radius_earth_units": 1.0,
            "duration_days": 1.0, "dt_snapshots": 3600.0,
            "scenario": "test", "snapshot_mode": "count", "n_snapshots": 3,
            "snapshot_times": [0.0, 3600.0, 7200.0], "plots": []}
    if solver == "bve":
        base.update({"viscosity": 0.0, "product_quadrature": "fine"})
    elif solver == "swe":
        base.update({"solver": "swe", "gravity": 9.80616,
                     "mean_depth_m": 3000.0, "product_quadrature": "fine"})
    else:
        base.update({"solver": "pe", "nlev": NLEV,
                     "sigma_interfaces": list(SIGMA), "r_dry": 287.04,
                     "cp_dry": 1004.64, "dt_seconds": 300.0,
                     "temperature": 260.0, "surface_pressure": 101325.0,
                     "thermal_amplitude": 1.0, "product_quadrature": "fine"})
    base.update(extra)
    return base


def _write_capsule(root: pathlib.Path, solver: str, coeffs, times, *,
                   with_schema: bool = True, manifest_extra: dict | None = None,
                   config_only: bool = False, run_config: dict | None = None,
                   time_file: bool = True) -> pathlib.Path:
    from tropoi.representation.archive.schema import (COEFFICIENT_FILES,
                                                      TIME_FILES)
    root.mkdir(parents=True, exist_ok=True)
    run_config = run_config or _run_config(solver)
    np.save(root / COEFFICIENT_FILES[solver], coeffs)
    if time_file:
        np.save(root / TIME_FILES[solver], np.asarray(times, dtype=np.float64))
    if config_only:
        (root / "config.json").write_text(json.dumps(run_config),
                                          encoding="utf-8")
        return root
    manifest = {"run_id": f"{solver}-test", "status": "completed",
                "run_config": run_config, "notes": {"equations": solver}}
    if with_schema:
        manifest.update(provenance_blocks(solver, run_config))
    if manifest_extra:
        manifest.update(manifest_extra)
    (root / "manifest.json").write_text(json.dumps(manifest),
                                        encoding="utf-8")
    return root


@pytest.fixture
def rng():
    return np.random.default_rng(1234)


@pytest.fixture
def bve_capsule(tmp_path, rng):
    coeffs = _triangular(rng, (3,), monopole=4e-22)
    return _write_capsule(tmp_path / "bve", "bve", coeffs,
                          [0.0, 3600.0, 7200.0]), coeffs


@pytest.fixture
def swe_capsule(tmp_path, rng):
    coeffs = _triangular(rng, (3, 3))
    return _write_capsule(tmp_path / "swe", "swe", coeffs,
                          [0.0, 3600.0, 7200.0]), coeffs


@pytest.fixture
def pe_capsule(tmp_path, rng):
    coeffs = _triangular(rng, (2, 3 * NLEV + 1))
    # temperature and ln_ps carry horizontal means (nonzero monopoles).
    coeffs[:, 2 * NLEV:3 * NLEV, 0, 0] = 260.0 * np.sqrt(4 * np.pi)
    coeffs[:, 3 * NLEV, 0, 0] = np.log(101325.0) * np.sqrt(4 * np.pi)
    return _write_capsule(tmp_path / "pe", "pe", coeffs, [0.0, 900.0]), coeffs


# ---------------------------------------------------------------------------
# Views: shapes, units, dimensions, storage sharing, read-only
# ---------------------------------------------------------------------------

def test_bve_named_view_is_the_whole_frame(bve_capsule):
    path, coeffs = bve_capsule
    sim = open_simulation(path)
    assert sim.solver == "bve" and len(sim) == 3
    assert sim.field_names == ("zeta",)
    snap = sim[1]
    zeta = snap.state["zeta"]
    assert zeta.shape == (N, N) and zeta.dimensions == ("l", "m")
    assert zeta.units == "s^-1" and zeta.spec.monopole == "conserved"
    np.testing.assert_array_equal(zeta.coeffs, coeffs[1])
    assert np.shares_memory(zeta.coeffs, sim.storage.coefficients)
    assert not zeta.coeffs.flags.writeable
    assert zeta.validate() is zeta


def test_swe_named_views_select_rows_without_copy(swe_capsule):
    path, coeffs = swe_capsule
    sim = open_simulation(path)
    snap = sim[2]
    assert list(snap.state) == ["zeta", "delta", "phi"]
    for row, name in enumerate(("zeta", "delta", "phi")):
        view = snap.state[name]
        assert view.shape == (N, N) and view.nlev is None
        np.testing.assert_array_equal(view.coeffs, coeffs[2, row])
        assert np.shares_memory(view.coeffs, snap.packed)
        assert np.shares_memory(view.coeffs, sim.storage.coefficients)
        view.validate()
    assert snap.state["phi"].units == "m^2 s^-2"
    assert "Phi0" in snap.state["phi"].description
    assert snap.state["zeta"] is snap.state["zeta"]  # cached view


def test_pe_named_views_keep_levels_and_do_not_pad_surface(pe_capsule):
    path, coeffs = pe_capsule
    sim = open_simulation(path)
    snap = sim[0]
    K = NLEV
    assert snap.state["zeta"].shape == (K, N, N)
    assert snap.state["delta"].shape == (K, N, N)
    temperature = snap.state["temperature"]
    assert temperature.shape == (K, N, N)
    assert temperature.dimensions == ("level", "l", "m")
    assert temperature.units == "K"
    assert temperature.spec.monopole == "horizontal-mean"
    np.testing.assert_array_equal(temperature.coeffs, coeffs[0, 2 * K:3 * K])
    # Nonuniform full levels from the actual sigma interfaces.
    expected_levels = tuple(0.5 * (SIGMA[k] + SIGMA[k + 1]) for k in range(K))
    assert temperature.level_values == expected_levels
    ln_ps = snap.state["ln_ps"]
    assert ln_ps.shape == (N, N) and ln_ps.dimensions == ("l", "m")
    assert ln_ps.level_values is None and ln_ps.nlev is None
    np.testing.assert_array_equal(ln_ps.coeffs, coeffs[0, 3 * K])
    for name in snap.state:
        assert np.shares_memory(snap.state[name].coeffs, snap.packed)
        snap.state[name].validate()
    # One level of a levelled field is itself a level-free view.
    level = temperature.level(-1)
    assert level.shape == (N, N) and level.spec.rows == (3 * K - 1, 3 * K)
    np.testing.assert_array_equal(level.coeffs, coeffs[0, 3 * K - 1])
    assert np.shares_memory(level.coeffs, snap.packed)
    schema = sim.storage.schema
    assert schema.nlev == K and schema.rows == 3 * K + 1
    assert schema.vertical["interfaces"] == list(SIGMA)


def test_public_views_reject_writes(swe_capsule):
    path, _ = swe_capsule
    snap = open_simulation(path)[0]
    with pytest.raises(ValueError, match="read-only"):
        snap.state["zeta"].coeffs[1, 0] = 5.0
    with pytest.raises(ValueError, match="read-only"):
        snap.packed[0, 1, 0] = 5.0
    with pytest.raises(ValueError, match="read-only"):
        open_simulation(path).times[0] = 1.0
    with pytest.raises(ValueError, match="read-only"):
        open_simulation(path, mmap=False)[0].state["phi"].coeffs[0, 0] = 1.0


def test_memory_map_and_in_memory_open_agree(swe_capsule):
    path, coeffs = swe_capsule
    mapped = open_simulation(path, mmap=True)
    loaded = open_simulation(path, mmap=False)
    assert isinstance(mapped.storage.coefficients, np.memmap)
    assert not isinstance(loaded.storage.coefficients, np.memmap)
    assert mapped.metadata["provenance"]["memory_mapped"] is True
    assert loaded.metadata["provenance"]["memory_mapped"] is False
    np.testing.assert_array_equal(mapped[1].state["delta"].coeffs,
                                  loaded[1].state["delta"].coeffs)
    np.testing.assert_array_equal(mapped.times, [0.0, 3600.0, 7200.0])


# ---------------------------------------------------------------------------
# Snapshot independence and ownership lifetime
# ---------------------------------------------------------------------------

def test_snapshot_survives_other_selection_and_simulation_release(
        pe_capsule):
    path, coeffs = pe_capsule
    sim = open_simulation(path)
    first = sim[0]
    view = first.state["temperature"]
    second = sim[1]
    assert second.time == 900.0 and first.time == 0.0
    np.testing.assert_array_equal(view.coeffs, coeffs[0, 2 * NLEV:3 * NLEV])
    storage = sim.storage
    del sim
    gc.collect()
    # The frame is owned by the snapshot (through its memory map), not by
    # the released Simulation.
    np.testing.assert_array_equal(first.state["ln_ps"].coeffs,
                                  coeffs[0, 3 * NLEV])
    np.testing.assert_array_equal(view.coeffs, coeffs[0, 2 * NLEV:3 * NLEV])
    assert first.metadata["snapshot"] == {"index": 0, "time_s": 0.0,
                                          "time_units": "s"}
    del storage, second
    gc.collect()
    assert float(np.abs(view.coeffs).max()) > 0.0


def test_snapshot_indexing_semantics(bve_capsule):
    path, _ = bve_capsule
    sim = open_simulation(path)
    assert sim[-1].index == 2 and sim[-1].time == 7200.0
    assert [s.index for s in sim] == [0, 1, 2]
    with pytest.raises(IndexError, match="out of range"):
        sim[3]
    with pytest.raises(IndexError, match="out of range"):
        sim[-4]
    with pytest.raises(TypeError, match="integer saved-output index"):
        sim[1.5]
    with pytest.raises(TypeError, match="integer saved-output index"):
        sim[0:2]
    with pytest.raises(TypeError):
        sim[True]


def test_unknown_field_name_lists_available_fields(swe_capsule):
    path, _ = swe_capsule
    snap = open_simulation(path)[0]
    with pytest.raises(KeyError, match="zeta, delta, phi"):
        snap.state["temperature"]


# ---------------------------------------------------------------------------
# Empty sequences, malformed headers and time axes, layout mismatches
# ---------------------------------------------------------------------------

def test_empty_saved_sequence_opens_but_cannot_be_indexed(tmp_path):
    coeffs = np.empty((0, 3, N, N), dtype=np.complex128)
    path = _write_capsule(tmp_path / "empty", "swe", coeffs, [])
    sim = open_simulation(path)
    assert len(sim) == 0 and sim.times.shape == (0,)
    assert sim.metadata["snapshot_count"] == 0
    with pytest.raises(IndexError, match="stores no snapshots"):
        sim[0]
    with pytest.raises(IndexError, match="stores no snapshots"):
        sim[-1]


def test_malformed_npy_header_is_rejected_at_open(tmp_path, rng):
    path = _write_capsule(tmp_path / "bad", "bve", _triangular(rng, (2,)),
                          [0.0, 1.0])
    (path / "vorticity_coeffs.npy").write_bytes(b"\x93NUMPY\x01\x00garbage")
    with pytest.raises(CapsuleLayoutError, match="invalid .npy header"):
        open_simulation(path)


@pytest.mark.parametrize("times,message", [
    ([0.0, 1.0], "one entry per stored frame"),
    ([0.0, np.nan, 2.0], "non-finite"),
    ([0.0, 2.0, 1.0], "strictly increasing"),
    ([0.0, 1.0, 1.0], "strictly increasing"),
])
def test_invalid_time_axes_are_rejected_at_open(tmp_path, rng, times,
                                                message):
    path = _write_capsule(tmp_path / "times", "swe", _triangular(rng, (3, 3)),
                          times)
    with pytest.raises(CapsuleLayoutError, match=message):
        open_simulation(path)


def test_two_dimensional_time_axis_is_rejected(tmp_path, rng):
    path = _write_capsule(tmp_path / "t2", "bve", _triangular(rng, (2,)),
                          [0.0, 1.0])
    np.save(path / "bve_snapshot_times.npy", np.zeros((2, 1)))
    with pytest.raises(CapsuleLayoutError, match="one-dimensional"):
        open_simulation(path)


def test_layout_mismatch_against_schema_is_rejected(tmp_path, rng):
    # PE stack with the wrong number of rows for the declared nlev.
    coeffs = _triangular(rng, (2, 3 * NLEV))
    path = _write_capsule(tmp_path / "rows", "pe", coeffs, [0.0, 1.0])
    with pytest.raises(CapsuleLayoutError, match="expects"):
        open_simulation(path)
    # Non-complex storage.
    path2 = _write_capsule(tmp_path / "real", "bve",
                           np.zeros((2, N, N)), [0.0, 1.0])
    with pytest.raises(CapsuleLayoutError, match="complex"):
        open_simulation(path2)
    # l_max disagreeing with the array extent.
    path3 = _write_capsule(tmp_path / "lmax", "swe", _triangular(rng, (2, 3)),
                           [0.0, 1.0], run_config=_run_config("swe", lmax=5))
    with pytest.raises(CapsuleLayoutError, match="l_max=5"):
        open_simulation(path3)


def test_missing_coefficient_file_and_missing_run_dir(tmp_path):
    (tmp_path / "manifest.json").write_text(
        json.dumps({"run_config": _run_config("swe")}), encoding="utf-8")
    with pytest.raises(CapsuleLayoutError, match="missing"):
        open_simulation(tmp_path)
    empty_dir = tmp_path / "empty_dir"
    empty_dir.mkdir()
    with pytest.raises(CapsuleError, match="no manifest.json"):
        open_simulation(empty_dir)
    with pytest.raises(CapsuleError, match="not a directory"):
        open_simulation(tmp_path / "does_not_exist")


def test_latest_run_pointer_is_followed(tmp_path, rng):
    _write_capsule(tmp_path / "exp" / "run1", "bve", _triangular(rng, (1,)),
                   [0.0])
    (tmp_path / "latest_run.txt").write_text("exp/run1\n", encoding="utf-8")
    assert open_simulation(tmp_path).metadata["run_id"] == "bve-test"


# ---------------------------------------------------------------------------
# Legacy and schema handling
# ---------------------------------------------------------------------------

def test_legacy_missing_solver_bve_is_inferred_and_marked(tmp_path, rng):
    # The historical psx-bve capsule: no solver, no grid, no snapshot_mode,
    # no product_quadrature, no stored time axis, config.json only.
    legacy = {"lmax": L_MAX, "resolution": 2, "nlat": 8, "nlon": 16,
              "day_hours": 24.0, "radius_earth_units": 1.0,
              "duration_days": 1.0, "dt_snapshots": 3600.0,
              "scenario": "two_vortices", "viscosity": 0.0, "out": "runs",
              "experiment": None, "overwrite": False, "run_id": "legacy"}
    coeffs = _triangular(rng, (4,))
    path = _write_capsule(tmp_path / "legacy", "bve", coeffs, None,
                          run_config=legacy, config_only=True,
                          time_file=False)
    sim = open_simulation(path)
    assert sim.solver == "bve"
    np.testing.assert_array_equal(sim.times, [0.0, 3600.0, 7200.0, 10800.0])
    prov = sim.metadata["provenance"]
    assert prov["schema"]["source"] == "inferred"
    assert "psx-bve" in prov["schema"]["solver"]
    assert prov["time_axis"]["source"] == "inferred"
    assert prov["time_axis"]["dt_snapshots"] == 3600.0
    schema = sim.storage.schema
    assert schema.inferred
    assert schema.geometry["grid"] == "geodesic"
    assert schema.provenance["inferred_defaults"]["grid"]
    assert schema.environment["product_quadrature"] == "fine"
    assert "product_quadrature" in schema.provenance["inferred_defaults"]
    np.testing.assert_array_equal(sim[3].state["zeta"].coeffs, coeffs[3])
    assert sim.metadata["run_id"] == "legacy"


def test_unknown_layout_without_solver_is_refused(tmp_path, rng):
    root = tmp_path / "unknown"
    root.mkdir()
    np.save(root / "mystery_coeffs.npy", _triangular(rng, (2, 3)))
    (root / "config.json").write_text(json.dumps({"lmax": L_MAX}),
                                      encoding="utf-8")
    with pytest.raises(SchemaError, match="cannot determine the solver"):
        open_simulation(root)


def test_missing_time_axis_without_inferable_schedule_is_refused(tmp_path,
                                                                 rng):
    path = _write_capsule(tmp_path / "notimes", "swe", _triangular(rng, (2, 3)),
                          [0.0, 1.0], time_file=False)
    with pytest.raises(CapsuleLayoutError, match="time axis .* is missing"):
        open_simulation(path)


def test_manifest_schema_without_block_is_inferred_from_notes_convention(
        tmp_path, rng):
    coeffs = _triangular(rng, (1, 3 * NLEV + 1))
    coeffs[:, 2 * NLEV:, 0, 0] = 1.0
    path = _write_capsule(tmp_path / "noschema", "pe", coeffs, [0.0],
                          with_schema=False)
    sim = open_simulation(path)
    schema = sim.storage.schema
    assert schema.inferred and "coefficient_ordering" in \
        schema.provenance["convention_source"]
    assert sim[0].state["temperature"].shape == (NLEV, N, N)


def test_unknown_schema_version_keeps_metadata_but_refuses_typed_access(
        tmp_path, rng):
    coeffs = _triangular(rng, (2, 3))
    run_config = _run_config("swe")
    block = state_schema_for("swe", run_config).to_manifest_dict()
    block["version"] = STATE_SCHEMA_VERSION + 1
    path = _write_capsule(tmp_path / "future", "swe", coeffs, [0.0, 1.0],
                          with_schema=False,
                          manifest_extra={"state_schema": block})
    sim = open_simulation(path)
    meta = sim.metadata
    assert meta["solver"] == "swe" and meta["snapshot_count"] == 2
    assert meta["provenance"]["schema"]["source"] == "unavailable"
    assert "version" in meta["provenance"]["schema"]["error"]
    assert sim.times.tolist() == [0.0, 1.0]
    with pytest.raises(UnknownSchemaVersionError):
        sim[0].state
    with pytest.raises(UnknownSchemaVersionError):
        sim.field_names


def test_conflicting_schema_solver_is_refused_for_typed_access(tmp_path, rng):
    block = state_schema_for("bve", _run_config("bve")).to_manifest_dict()
    path = _write_capsule(tmp_path / "conflict", "swe", _triangular(rng, (1, 3)),
                          [0.0], with_schema=False,
                          manifest_extra={"state_schema": block})
    sim = open_simulation(path)
    assert sim.metadata["solver"] == "swe"
    with pytest.raises(SchemaError, match="for solver 'bve'"):
        sim[0].state


def test_schema_round_trips_through_manifest_dict():
    for solver in ("bve", "swe", "pe"):
        schema = state_schema_for(solver, _run_config(solver))
        again = StateSchema.from_manifest_dict(
            json.loads(json.dumps(schema.to_manifest_dict())))
        assert again.fields == schema.fields
        assert again.l_max == schema.l_max and again.rows == schema.rows
        assert again.vertical == schema.vertical
        assert again.support["product_truncation_cut"] == 2
        assert not again.inferred


def test_schema_malformed_blocks_raise_schema_error():
    with pytest.raises(SchemaError, match="must be an object"):
        StateSchema.from_manifest_dict([1])
    with pytest.raises(UnknownSchemaVersionError):
        StateSchema.from_manifest_dict({"version": 0, "solver": "swe"})
    with pytest.raises(SchemaError, match="unknown solver"):
        StateSchema.from_manifest_dict({"version": 1, "solver": "mhd"})
    with pytest.raises(SchemaError, match="malformed"):
        StateSchema.from_manifest_dict({"version": 1, "solver": "swe"})


# ---------------------------------------------------------------------------
# Invalid accessed coefficients (never repaired)
# ---------------------------------------------------------------------------

def test_invalid_accessed_coefficients_fail_validation_only(tmp_path, rng):
    coeffs = _triangular(rng, (4, 3))
    coeffs[0, 0, 1, 1] = np.nan                  # non-finite
    coeffs[1, 1, 0, 2] = 1.0                     # m > l padding
    coeffs[2, 2, 2, 0] = 1.0 + 1.0j              # Im a_l0
    coeffs[3, 0, 0, 0] = 1.0                     # zero-mean monopole
    path = _write_capsule(tmp_path / "invalid", "swe", coeffs,
                          [0.0, 1.0, 2.0, 3.0])
    sim = open_simulation(path)                  # open never scans values
    checks = [(0, "zeta", "non-finite"), (1, "delta", "padding"),
              (2, "phi", "Im"), (3, "zeta", "monopole")]
    for index, name, message in checks:
        view = sim[index].state[name]            # access is allowed
        with pytest.raises(SpectralConventionError, match=message):
            view.validate()
        # Nothing was repaired.
        np.testing.assert_array_equal(view.coeffs, coeffs[index][
            ("zeta", "delta", "phi").index(name)])
    # Untouched fields of the same frames still validate.
    sim[0].state["delta"].validate()
    sim[3].state["phi"].validate()


def test_pe_zero_mean_rule_reports_the_level(tmp_path, rng):
    coeffs = _triangular(rng, (1, 3 * NLEV + 1))
    coeffs[0, 2 * NLEV:, 0, 0] = 1.0
    coeffs[0, NLEV + 2, 0, 0] = 1e-3   # delta level 3 monopole
    path = _write_capsule(tmp_path / "pelevel", "pe", coeffs, [0.0])
    with pytest.raises(SpectralConventionError, match="level 3"):
        open_simulation(path)[0].state["delta"].validate()


# ---------------------------------------------------------------------------
# Field specification and view invariants
# ---------------------------------------------------------------------------

def test_field_spec_validation():
    with pytest.raises(ValueError, match="exactly one row"):
        FieldSpec("x", "u", "d", rows=(0, 2), levels=False)
    with pytest.raises(ValueError, match="0 <= start < stop"):
        FieldSpec("x", "u", "d", rows=(2, 2), levels=True)
    with pytest.raises(ValueError, match="without rows"):
        FieldSpec("x", "u", "d", rows=None, levels=True)
    with pytest.raises(ValueError, match="monopole rule"):
        FieldSpec("x", "u", "d", monopole="whatever")
    spec = FieldSpec("t", "K", "d", rows=(2, 5), levels=True)
    assert spec.nlev == 3 and spec.dimensions == ("level", "l", "m")
    assert FieldSpec.from_dict(spec.to_dict()) == spec


def test_spectral_modes_and_state_invariants(rng):
    frame = _triangular(rng, (5,))
    spec = FieldSpec("t", "K", "d", rows=(1, 4), levels=True,
                     monopole="horizontal-mean")
    view = SpectralModes(spec.select(frame), spec, level_values=(0.1, 0.5, 0.9))
    assert view.shape == (3, N, N) and view.l_max == L_MAX
    assert np.shares_memory(view.coeffs, frame)
    with pytest.raises(SpectralConventionError, match="level coordinate"):
        SpectralModes(frame[1:4], spec, level_values=(0.1,))
    with pytest.raises(SpectralConventionError, match="declares 3 level"):
        SpectralModes(frame[1:3], spec)
    with pytest.raises(SpectralConventionError, match="complex"):
        SpectralModes(np.zeros((N, N)), FieldSpec("z", "", "d"))
    with pytest.raises(IndexError):
        view.level(3)
    state = SpectralState(frame, (spec, FieldSpec("s", "", "d", rows=(4, 5))))
    assert len(state) == 2 and set(state) == {"t", "s"}
    assert not state.packed.flags.writeable
    with pytest.raises(ValueError, match="duplicate"):
        SpectralState(frame, (spec, spec))
    with pytest.raises(SpectralConventionError, match="needs rows"):
        SpectralState(frame[:3], (spec,))["t"]
    summary = view.summary()
    assert summary["shape"] == (3, N, N) and summary["finite"]
    assert len(summary["monopole_real"]) == 3


def test_simulation_over_a_custom_storage_is_archive_independent():
    class MemoryStorage:
        solver = "swe"
        times = np.array([0.0, 10.0])
        field_specs = (FieldSpec("zeta", "s^-1", "d", rows=(0, 1)),
                       FieldSpec("delta", "s^-1", "d", rows=(1, 2)),
                       FieldSpec("phi", "m^2 s^-2", "d", rows=(2, 3)))
        level_values = None
        metadata = {"origin": "memory"}
        data = np.zeros((2, 3, N, N), dtype=np.complex128)

        def frame(self, index):
            return self.data[index]

        def plot_snapshot(self, index, output_path, **options):
            return pathlib.Path(output_path)

    sim = Simulation(MemoryStorage())
    assert len(sim) == 2 and sim.field_names == ("zeta", "delta", "phi")
    snap = sim[1]
    assert snap.metadata["origin"] == "memory"
    assert snap.state["phi"].shape == (N, N)
    assert snap.plot("x.png") == pathlib.Path("x.png")


# ---------------------------------------------------------------------------
# CPU-safety of a real open in a fresh interpreter
# ---------------------------------------------------------------------------

def test_open_and_inspect_in_fresh_interpreter_is_cpu_safe(pe_capsule):
    import subprocess
    from cli.conftest import HEAVY_MODULES

    path, _ = pe_capsule
    code = (
        "import sys\n"
        "from tropoi.representation.archive import open_simulation\n"
        f"sim = open_simulation({str(path)!r})\n"
        "snap = sim[-1]\n"
        "t = snap.state['temperature']; t.validate()\n"
        "assert t.shape == (4, 4, 4), t.shape\n"
        "assert snap.metadata['schema']['fields'][2]['name'] == 'temperature'\n"
        f"banned = [m for m in {HEAVY_MODULES!r} if m in sys.modules]\n"
        "assert not banned, banned\n"
        "assert 'tropoi.numerics' not in sys.modules\n"
        "assert 'tropoi.physics.primitive_equations' not in sys.modules\n"
        # canonical homes of the numerics and cores (legacy names are aliases)
        "assert not [m for m in sys.modules if m.startswith(("
        "'tropoi.spatial.grids', 'tropoi.spatial.transforms', "
        "'tropoi.spatial.operators', 'tropoi.spatial.states', "
        "'tropoi.temporal.tendencies'))]\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True,
                            text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
