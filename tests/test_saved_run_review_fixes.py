"""Regression tests for the three review findings on feat/saved-run-api.

1. Schema conventions and consistency are validated before typed access
   (unsupported normalization/layout/time units, conflicting or
   non-covering field rows, inconsistent PE vertical coordinates), and the
   CLI and the reader interpret the same stored data identically.
2. BVE time inference is restricted to the recognised legacy interval
   capsule; modern or explicit-count capsules with a missing time file
   fail clearly instead of getting fabricated timestamps.
3. With ``mmap=False`` the owning array is frozen, so a public view cannot
   regain write access through ``setflags(write=True)``.

Also: field access is a plain view; ``SpectralModes.validate()`` is the
explicit convention check (an intentional API choice, not a defect).
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

from tropoi.cli.main import main
from tropoi.representation.archive import (CapsuleLayoutError, SchemaError,
                                           StateSchema, open_simulation)
from tropoi.representation.archive.schema import (COEFFICIENT_FILES,
                                                  TIME_FILES,
                                                  UnsupportedConventionError,
                                                  provenance_blocks,
                                                  state_schema_for)
from tropoi.spatial.modes import FieldSpec

L_MAX = 3
N = L_MAX + 1
SIGMA = [0.0, 0.2, 0.6, 1.0]
NLEV = 3


def _config(solver: str, **extra) -> dict:
    base = {"lmax": L_MAX, "grid": "geodesic", "resolution": 2, "nlat": 8,
            "nlon": 16, "day_hours": 24.0, "radius_earth_units": 1.0,
            "duration_days": 1.0, "dt_snapshots": 3600.0, "scenario": "x",
            "snapshot_mode": "count", "n_snapshots": 2,
            "snapshot_times": [0.0, 3600.0], "plots": [],
            "product_quadrature": "fine"}
    if solver == "bve":
        base["viscosity"] = 0.0
    elif solver == "swe":
        base.update(solver="swe", gravity=9.80616, mean_depth_m=3000.0)
    else:
        base.update(solver="pe", nlev=NLEV, sigma_interfaces=SIGMA,
                    r_dry=287.04, cp_dry=1004.64, dt_seconds=300.0,
                    temperature=260.0, surface_pressure=101325.0,
                    thermal_amplitude=1.0)
    base.update(extra)
    return base


def _coeffs(solver: str, frames: int = 2) -> np.ndarray:
    rows = {"bve": None, "swe": 3, "pe": 3 * NLEV + 1}[solver]
    shape = (frames, N, N) if rows is None else (frames, rows, N, N)
    rng = np.random.default_rng(11)
    coeffs = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    l = np.arange(N)[:, None]
    m = np.arange(N)[None, :]
    coeffs[..., m > l] = 0.0
    coeffs[..., :, 0] = coeffs[..., :, 0].real
    coeffs[..., 0, 0] = 0.0
    if solver == "pe":
        coeffs[:, 2 * NLEV:, 0, 0] = 5.0
    return coeffs


def _capsule(root: pathlib.Path, solver: str, *, config=None,
             schema_block=None, time_file=True, config_only=False):
    root.mkdir(parents=True)
    config = config or _config(solver)
    np.save(root / COEFFICIENT_FILES[solver], _coeffs(solver))
    if time_file:
        np.save(root / TIME_FILES[solver], np.array([0.0, 3600.0]))
    if config_only:
        (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
        return root
    manifest = {"run_id": f"{solver}-review", "run_config": config,
                **provenance_blocks(solver, config)}
    if schema_block is not None:
        manifest["state_schema"] = schema_block
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def _block(solver: str, **overrides) -> dict:
    block = state_schema_for(solver, _config(solver)).to_manifest_dict()
    block.update(overrides)
    return block


# ---------------------------------------------------------------------------
# 1. Schema conventions and consistency
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("override,message", [
    ({"normalization": "schmidt-semi-normalized"}, "normalization"),
    ({"layout": "packed-triangular"}, "layout"),
    ({"time_units": "hours"}, "time_units"),
])
def test_unsupported_block_conventions_are_refused(tmp_path, override,
                                                   message):
    with pytest.raises(UnsupportedConventionError, match=message):
        StateSchema.from_manifest_dict(_block("swe", **override))
    root = _capsule(tmp_path / "c", "swe", schema_block=_block("swe", **override))
    sim = open_simulation(root)                     # metadata still opens
    assert sim.metadata["provenance"]["schema"]["source"] == "unavailable"
    with pytest.raises(UnsupportedConventionError, match=message):
        sim[0].state


def test_unsupported_field_conventions_are_refused():
    block = _block("swe")
    block["fields"][1]["normalization"] = "4pi"
    with pytest.raises(SchemaError, match="declares normalization '4pi'"):
        StateSchema.from_manifest_dict(block)
    block = _block("swe")
    block["fields"][0]["layout"] = "packed-triangular"
    with pytest.raises(SchemaError, match="declares layout"):
        StateSchema.from_manifest_dict(block)


@pytest.mark.parametrize("mutate,message", [
    (lambda f: f[1].update(rows=[0, 1]), "overlapping rows"),
    (lambda f: f[2].update(rows=[2, 4], levels=True), "exceed the 3 stored"),
    (lambda f: f.pop(), "cover 2 of the 3"),
    (lambda f: f[0].update(rows=None), "has no row range"),
])
def test_conflicting_field_rows_are_refused(mutate, message):
    block = _block("swe")
    mutate(block["fields"])
    with pytest.raises(SchemaError, match=message):
        StateSchema.from_manifest_dict(block)


def test_bve_schema_requires_one_whole_frame_field():
    block = _block("bve")
    block["fields"].append(FieldSpec("extra", "", "d").to_dict())
    with pytest.raises(SchemaError, match="exactly one field"):
        StateSchema.from_manifest_dict(block)


@pytest.mark.parametrize("vertical,message", [
    ({"interfaces": [0.0, 0.2, 0.6, 0.9], "full_levels": [0.1, 0.4, 0.75]},
     "exactly 1.0"),
    ({"interfaces": [0.0, 0.6, 0.2, 1.0], "full_levels": [0.3, 0.4, 0.6]},
     "strictly increasing"),
    ({"interfaces": [0.0, 0.5, 1.0], "full_levels": [0.25, 0.75]},
     "for nlev=3"),
    ({"interfaces": [0.0, 0.2, 0.6, 1.0], "full_levels": [0.1, 0.4, 0.9]},
     "not the interface midpoints"),
])
def test_inconsistent_pe_vertical_coordinates_are_refused(vertical, message):
    with pytest.raises(SchemaError, match=message):
        StateSchema.from_manifest_dict(_block("pe", vertical=vertical))


def test_pe_nlev_must_match_levelled_fields_and_vertical_block():
    with pytest.raises(SchemaError, match="declares nlev=4|must map to rows"):
        StateSchema.from_manifest_dict(_block("pe", nlev=4))
    with pytest.raises(SchemaError, match="require both nlev"):
        StateSchema.from_manifest_dict(_block("pe", vertical=None))


def test_inspection_and_reader_agree_on_a_bad_schema(tmp_path, capsys):
    block = _block("pe")
    block["fields"][2]["rows"] = [NLEV, 2 * NLEV]      # temperature overlaps delta
    root = _capsule(tmp_path / "pe", "pe", schema_block=block)
    # Reader: metadata opens, typed access refused with the same reason.
    sim = open_simulation(root)
    with pytest.raises(SchemaError, match="overlapping rows"):
        sim[0].state
    # CLI: ordinary inspection works and does not print a field summary
    # derived from the conflicting block; --snapshot refuses identically.
    assert main(["inspect", str(root)]) == 0
    assert "state fields" not in capsys.readouterr().out
    assert main(["inspect", str(root), "--snapshot", "0"]) == 2
    assert "overlapping rows" in capsys.readouterr().err
    # A plot request goes through the same schema and is refused too.
    with pytest.raises(SchemaError, match="overlapping rows"):
        sim[0].plot(tmp_path / "x.png")


def test_inferred_schemas_pass_the_same_consistency_checks():
    for solver in ("bve", "swe", "pe"):
        schema = state_schema_for(solver, _config(solver))
        schema.check_consistency()
        StateSchema.from_manifest_dict(schema.to_manifest_dict())


# ---------------------------------------------------------------------------
# 2. Time inference restricted to the legacy interval capsule
# ---------------------------------------------------------------------------

def _legacy_config() -> dict:
    return {"lmax": L_MAX, "resolution": 2, "nlat": 8, "nlon": 16,
            "day_hours": 24.0, "radius_earth_units": 1.0,
            "duration_days": 1.0, "dt_snapshots": 3600.0,
            "scenario": "two_vortices", "viscosity": 0.0, "out": "runs",
            "experiment": None, "overwrite": False, "run_id": "legacy"}


def test_legacy_interval_capsule_still_infers_its_schedule(tmp_path):
    root = _capsule(tmp_path / "legacy", "bve", config=_legacy_config(),
                    time_file=False, config_only=True)
    sim = open_simulation(root)
    np.testing.assert_array_equal(sim.times, [0.0, 3600.0])
    assert sim.metadata["provenance"]["time_axis"]["source"] == "inferred"


@pytest.mark.parametrize("extra", [
    {"snapshot_mode": "count", "n_snapshots": 2,
     "snapshot_times": [0.0, 43200.0]},
    {"snapshot_mode": "interval"},
    {"snapshot_times": [0.0, 43200.0]},
    {"n_snapshots": 2},
    {"dt_snapshots": None},
    {"dt_snapshots": 0.0},
])
def test_missing_time_file_in_non_legacy_capsule_is_an_error(tmp_path, extra):
    config = {**_legacy_config(), **extra}
    root = _capsule(tmp_path / "modern", "bve", config=config,
                    time_file=False, config_only=True)
    with pytest.raises(CapsuleLayoutError, match="never fabricated"):
        open_simulation(root)


def test_modern_bve_manifest_without_time_file_is_an_error(tmp_path):
    root = _capsule(tmp_path / "modern", "bve", time_file=False)
    with pytest.raises(CapsuleLayoutError, match="bve_snapshot_times.npy"):
        open_simulation(root)


def test_explicit_solver_capsules_never_infer_times(tmp_path):
    for solver in ("swe", "pe"):
        root = _capsule(tmp_path / solver, solver, time_file=False)
        with pytest.raises(CapsuleLayoutError, match="never fabricated"):
            open_simulation(root)


# ---------------------------------------------------------------------------
# 3. Ownership with mmap=False
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mmap", [False, True])
def test_public_views_cannot_regain_write_access(tmp_path, mmap):
    root = _capsule(tmp_path / "swe", "swe")
    sim = open_simulation(root, mmap=mmap)
    first, second = sim[0], sim[1]
    view = first.state["phi"].coeffs
    original = view.copy()
    for target in (view, first.packed, sim.storage.coefficients,
                   sim.storage.frame(0), sim.times):
        with pytest.raises(ValueError, match="WRITEABLE"):
            target.setflags(write=True)
        assert not target.flags.writeable
    # The storage the snapshots share is untouched.
    np.testing.assert_array_equal(first.state["phi"].coeffs, original)
    np.testing.assert_array_equal(second.state["phi"].coeffs,
                                  sim.storage.coefficients[1, 2])
    assert not sim.storage.stored_array(TIME_FILES["swe"]).flags.writeable


def test_field_access_is_a_plain_view_and_validate_is_explicit(tmp_path):
    root = _capsule(tmp_path / "bad", "swe")
    coeffs = np.load(root / "swe_coeffs.npy")
    coeffs[0, 0, 0, 2] = 1.0                       # m > l padding
    np.save(root / "swe_coeffs.npy", coeffs)
    view = open_simulation(root)[0].state["zeta"]  # access does not validate
    assert view.coeffs[0, 2] == 1.0
    with pytest.raises(ValueError, match="padding"):
        view.validate()


# ---------------------------------------------------------------------------
# 4. Canonical solver layout: permutations and axis/convention contradictions
# ---------------------------------------------------------------------------

def _permuted_swe_block() -> dict:
    block = _block("swe")
    for spec in block["fields"]:
        spec["rows"] = {"zeta": [1, 2], "delta": [0, 1], "phi": [2, 3]}[spec["name"]]
    return block


def _permuted_pe_block() -> dict:
    block = _block("pe")
    K = NLEV
    for spec in block["fields"]:
        spec["rows"] = {"zeta": [K, 2 * K], "delta": [2 * K, 3 * K],
                        "temperature": [0, K],
                        "ln_ps": [3 * K, 3 * K + 1]}[spec["name"]]
    return block


@pytest.mark.parametrize("solver,block,message", [
    ("swe", _permuted_swe_block, "field 'zeta' must map to rows"),
    ("pe", _permuted_pe_block, "field 'zeta' must map to rows"),
    ("swe", lambda: _block("swe", storage_axes=["time", "l", "m", "field"]),
     "storage_axes"),
    ("pe", lambda: _block("pe", storage_axes=["time", "l", "m", "row"]),
     "storage_axes"),
    ("bve", lambda: _block("bve", storage_axes=["time", "m", "l"]),
     "storage_axes"),
    ("swe", lambda: _block("swe", reality="complex field, m in [-l, l]"),
     "reality"),
    ("swe", lambda: _block("swe", support={"l_max": L_MAX,
                                           "product_truncation_cut": 3,
                                           "triangle": "0 <= m <= l <= l_max"}),
     "product_truncation_cut"),
    ("swe", lambda: _block("swe", support={"l_max": L_MAX,
                                           "product_truncation_cut": 2,
                                           "triangle": "full square"}),
     "triangle"),
])
def test_noncanonical_layouts_are_refused_everywhere(tmp_path, capsys, solver,
                                                     block, message):
    block = block()
    with pytest.raises(SchemaError, match=message):
        StateSchema.from_manifest_dict(block)
    root = _capsule(tmp_path / solver, solver, schema_block=block)
    # Ordinary metadata inspection still works.
    sim = open_simulation(root)
    assert sim.metadata["solver"] == solver and len(sim) == 2
    assert sim.metadata["provenance"]["schema"]["source"] == "unavailable"
    assert main(["inspect", str(root)]) == 0
    assert "state fields" not in capsys.readouterr().out
    # Typed access, the CLI snapshot view, and plotting refuse identically.
    with pytest.raises(SchemaError, match=message):
        sim[0].state
    assert main(["inspect", str(root), "--snapshot", "0"]) == 2
    assert message.split("'")[0].strip() in capsys.readouterr().err
    with pytest.raises(SchemaError, match=message):
        sim[0].plot(tmp_path / "x.png")
    with pytest.raises(SchemaError, match=message):
        sim[0].plot(tmp_path / "x.png", representation="spectral")
    assert not (tmp_path / "x.png").exists()


def test_field_list_order_is_free_but_mapping_is_fixed():
    block = _block("pe")
    block["fields"].reverse()                     # cosmetic reordering
    schema = StateSchema.from_manifest_dict(block)
    assert schema.field_names == ("ln_ps", "temperature", "delta", "zeta")
    assert dict((s.name, s.rows) for s in schema.fields)["temperature"] == \
        (2 * NLEV, 3 * NLEV)
    # Renaming a field breaks the canonical name set.
    block = _block("swe")
    block["fields"][2]["name"] = "geopotential"
    with pytest.raises(SchemaError, match="stores fields"):
        StateSchema.from_manifest_dict(block)


def test_canonical_layout_matches_the_written_schemas():
    from tropoi.representation.archive.schema import canonical_field_layout
    for solver in ("bve", "swe", "pe"):
        schema = state_schema_for(solver, _config(solver))
        assert {s.name: (s.rows, s.levels) for s in schema.fields} == \
            canonical_field_layout(solver, schema.nlev)
