"""New-manifest provenance through the REAL writer/lifecycle integration.

The additive ``state_schema`` / ``diagnostic_definitions`` blocks are
written by ``execute_with_provenance`` (pre-write) and by each solver's
manifest rewrite, and consumed by the saved-run reader. These tests drive
the actual lifecycle with a stubbed numerical solver that persists tiny
arrays, then assert: blocks present and versioned, run ids / scientific
hashes byte-identical to their pinned pre-feature values, diagnostic column
definitions identical to the diagnostics modules, and the reader
interpreting the new manifest WITHOUT inference. CPU only.
"""
from __future__ import annotations

import json
import pathlib
from datetime import datetime, timezone

import numpy as np
import pytest

from tropoi.representation.archive import open_simulation
from tropoi.representation.archive.schema import (
    BVE_DIAGNOSTIC_COLUMNS, DIAGNOSTIC_DEFINITIONS_VERSION,
    EARTH_RADIUS_M, PE_DIAGNOSTIC_COLUMNS, STATE_SCHEMA_VERSION,
    SWE_DIAGNOSTIC_COLUMNS, W5_RADIUS_M, diagnostic_definitions_for,
    high_l_enstrophy_fraction_definition, provenance_blocks,
    state_schema_for)
from tropoi.run.bve.config import BVERunConfig
from tropoi.run.bve.io import _config_hash, make_run_id
from tropoi.run.pe.config import PERunConfig
from tropoi.run.swe.config import SWERunConfig
from tropoi.spatial.modes import ZERO_MEAN_MONOPOLE_RTOL
from tropoi.support import product_truncation_cut

NOW = datetime(2026, 7, 19, 12, 0, 0, tzinfo=timezone.utc)
COMMIT = "abcdef12"

#: Scientific hashes / run ids measured at 0ce4d27 (before this feature)
#: for representative resolved configurations. They must not move.
PINNED = {
    "bve": ("2df940f0",
            "20260719T120000Z_two-vortices_norot_r4_l21_dt12h_2df940f0_abcdef12"),
    "swe": ("0e6cfc38",
            "20260719T120000Z_williamson2_rot23p93h_r4_l21_dt6h_0e6cfc38_abcdef12"),
    "w5": ("cf9bd4e7",
           "20260719T120000Z_williamson5_rot23p93h_r4_l21_dt6h_cf9bd4e7_abcdef12"),
    "pe": ("fa2db863",
           "20260719T120000Z_thermal-wave_rot24h_r3_l10_dt15m_fa2db863_abcdef12"),
}


def _resolved(name: str):
    return {
        "bve": lambda: BVERunConfig.resolve({"n_snapshots": 3}),
        "swe": lambda: SWERunConfig.resolve({"scenario": "williamson2"}),
        "w5": lambda: SWERunConfig.resolve({"scenario": "williamson5"}),
        "pe": lambda: PERunConfig.resolve({"scenario": "thermal_wave"}),
    }[name]()


@pytest.mark.parametrize("name", sorted(PINNED))
def test_scientific_hash_and_run_id_are_unchanged(name):
    run_config = _resolved(name).to_run_config_dict()
    digest, run_id = PINNED[name]
    assert _config_hash(run_config) == digest
    assert make_run_id(run_config, now=NOW, commit=COMMIT) == run_id
    # The provenance blocks are derived FROM run_config and add nothing to it.
    before = json.dumps(run_config, sort_keys=True)
    solver = "swe" if name == "w5" else name
    provenance_blocks(solver, run_config)
    assert json.dumps(run_config, sort_keys=True) == before


def test_bve_run_config_key_set_is_frozen():
    # No 'solver' key is ever added to the historical psx-bve config.
    keys = set(BVERunConfig.resolve({}).to_run_config_dict())
    assert "solver" not in keys and "state_schema" not in keys


# ---------------------------------------------------------------------------
# Real lifecycle: manifest blocks written and read back
# ---------------------------------------------------------------------------

def _stub_solver(coeffs_name, times_name, frames_shape):
    def solver(cfg, run_dir, run_config):
        rng = np.random.default_rng(0)
        coeffs = np.zeros(frames_shape, dtype=np.complex128)
        coeffs[..., 1, 0] = rng.normal(size=frames_shape[:-2])
        np.save(run_dir.path / coeffs_name, coeffs)
        np.save(run_dir.path / times_name,
                np.arange(frames_shape[0], dtype=np.float64) * 60.0)
    return solver


@pytest.mark.parametrize("solver,module_name,cfg,coeffs_name,times_name,rows", [
    ("bve", "tropoi.cli.bve", {"n_snapshots": 2, "lmax": 5},
     "vorticity_coeffs.npy", "bve_snapshot_times.npy", None),
    ("swe", "tropoi.cli.swe", {"scenario": "williamson2", "n_snapshots": 2,
                               "lmax": 5}, "swe_coeffs.npy",
     "swe_snapshot_times.npy", 3),
    ("pe", "tropoi.cli.pe", {"scenario": "thermal_wave", "n_snapshots": 2,
                             "lmax": 5, "sigma_interfaces": [0.0, 0.3, 1.0]},
     "pe_coeffs.npy", "pe_snapshot_times.npy", 7),
])
def test_lifecycle_writes_versioned_blocks_the_reader_consumes(
        tmp_path, monkeypatch, solver, module_name, cfg, coeffs_name,
        times_name, rows):
    import importlib
    module = importlib.import_module(module_name)
    resolver = {"bve": BVERunConfig, "swe": SWERunConfig,
                "pe": PERunConfig}[solver]
    resolved = resolver.resolve({**cfg, "out": str(tmp_path / "runs")})
    n = resolved.lmax + 1
    shape = (2, n, n) if rows is None else (2, rows, n, n)
    monkeypatch.setattr(module, "_execute_solver",
                        _stub_solver(coeffs_name, times_name, shape))
    assert module.execute_run(resolved) == 0

    base = pathlib.Path(resolved.out).resolve()
    run_dir = base / (base / "latest_run.txt").read_text(
        encoding="utf-8").strip()
    manifest = json.loads((run_dir / "manifest.json").read_text(
        encoding="utf-8"))
    assert manifest["status"] == "completed"
    block = manifest["state_schema"]
    assert block["version"] == STATE_SCHEMA_VERSION
    assert block["solver"] == solver
    assert block["provenance"] == {"source": "manifest"}
    assert block["support"] == {
        "l_max": 5, "product_truncation_cut": product_truncation_cut(5),
        "triangle": "0 <= m <= l <= l_max"}
    assert block["rows"] == rows
    diagnostics = manifest["diagnostic_definitions"]
    assert diagnostics["version"] == DIAGNOSTIC_DEFINITIONS_VERSION
    assert diagnostics["solver"] == solver
    # config.json is untouched by the descriptive blocks.
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    assert "state_schema" not in config and "diagnostic_definitions" not in config
    assert manifest["run_config"] == config
    # The reader takes the manifest block as authoritative (no inference).
    sim = open_simulation(run_dir)
    assert sim.metadata["provenance"]["schema"] == {"source": "manifest"}
    assert sim.metadata["provenance"]["diagnostic_definitions"] == "manifest"
    assert not sim.storage.schema.inferred
    assert len(sim) == 2 and sim.times.tolist() == [0.0, 60.0]
    if solver == "pe":
        assert sim[1].state["temperature"].shape == (2, n, n)
        assert sim[1].state["temperature"].level_values == (0.15, 0.65)
        assert sim[1].state["ln_ps"].shape == (n, n)
    for name in sim.field_names:
        sim[0].state[name]
    # Status updates preserve the blocks (they go through update_manifest_status).
    assert manifest.get("updated_utc")


def test_status_rewrite_preserves_blocks(tmp_path):
    from tropoi.run.bve.io import (RUN_STATUS_COMPLETED, update_manifest_status,
                                   write_run_manifest)
    run_config = SWERunConfig.resolve({"scenario": "williamson2"}).to_run_config_dict()
    write_run_manifest(tmp_path, run_config, run_id="x",
                       **provenance_blocks("swe", run_config))
    update_manifest_status(tmp_path, RUN_STATUS_COMPLETED)
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["state_schema"]["fields"][2]["name"] == "phi"
    assert manifest["diagnostic_definitions"]["columns"][0]["name"] == "time_s"


def test_manifest_without_blocks_is_still_written_for_legacy_callers(tmp_path):
    from tropoi.run.bve.io import write_run_manifest
    write_run_manifest(tmp_path, {"scenario": "rh4"}, run_id="x")
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert "state_schema" not in manifest
    assert "diagnostic_definitions" not in manifest


# ---------------------------------------------------------------------------
# Definitions stay synchronized with their sources
# ---------------------------------------------------------------------------

def _cupy_available():
    try:
        import cupy
        cupy.cuda.runtime.getDeviceCount()
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _cupy_available(),
                    reason="diagnostics modules import CuPy at import time")
def test_diagnostic_columns_match_the_recorders():
    from tropoi.run.bve.diagnostics import CSV_COLUMNS
    from tropoi.run.pe.diagnostics import PE_CSV_COLUMNS
    from tropoi.run.swe.diagnostics import SWE_CSV_COLUMNS
    assert tuple(CSV_COLUMNS) == BVE_DIAGNOSTIC_COLUMNS
    assert tuple(SWE_CSV_COLUMNS) == SWE_DIAGNOSTIC_COLUMNS
    assert tuple(PE_CSV_COLUMNS) == PE_DIAGNOSTIC_COLUMNS


@pytest.mark.skipif(not _cupy_available(),
                    reason="physics cores import CuPy at import time")
def test_monopole_tolerance_matches_the_cores():
    from tropoi.physics.primitive_equations import PrimitiveEquationsModel
    from tropoi.physics.shallow_water import ShallowWaterModel
    assert ShallowWaterModel._MONOPOLE_RTOL == ZERO_MEAN_MONOPOLE_RTOL
    assert PrimitiveEquationsModel._MONOPOLE_RTOL == ZERO_MEAN_MONOPOLE_RTOL


@pytest.mark.skipif(not _cupy_available(),
                    reason="physics cores import CuPy at import time")
def test_swe_stack_indices_match_the_core_everywhere():
    from tropoi.physics import shallow_water
    from tropoi.run.swe import visualization
    assert (visualization.ZETA, visualization.DELTA, visualization.PHI) ==         (shallow_water.ZETA, shallow_water.DELTA, shallow_water.PHI)
    schema = state_schema_for("swe", SWERunConfig.resolve(
        {"scenario": "williamson2"}).to_run_config_dict())
    rows = {spec.name: spec.rows[0] for spec in schema.fields}
    assert rows == {"zeta": shallow_water.ZETA, "delta": shallow_water.DELTA,
                    "phi": shallow_water.PHI}


def test_reference_radii_match_their_sources():
    from tropoi.run.swe.config import W5_RADIUS_M as source_w5
    from tropoi.spatial.environment import PlanetaryParameters
    assert W5_RADIUS_M == source_w5
    assert PlanetaryParameters.from_earth_like().equatorial_radius == \
        EARTH_RADIUS_M


def test_high_l_enstrophy_definition_records_the_band_and_conventions():
    definition = high_l_enstrophy_fraction_definition(21)
    assert definition["band"]["cut"] == 14
    assert definition["band"]["degrees"] == "cut < l <= l_max"
    assert "0 <= m <= l" in definition["domain"]
    assert definition["mode_power"]["m > 0"].startswith("2 * |a_lm|^2")
    assert definition["zero_power"] == "0.0 when the denominator is zero"
    assert definition["version"] == 1
    block = diagnostic_definitions_for("bve", {"lmax": 21})
    assert block["high_l_enstrophy_frac"] == definition
    assert [c["name"] for c in block["columns"]] == list(BVE_DIAGNOSTIC_COLUMNS)


def test_diagnostic_column_values_in_real_capsules_are_untouched():
    """Existing CSVs keep their meaning: the definitions describe, never rewrite."""
    runs = pathlib.Path(__file__).resolve().parents[1] / "runs"
    csv_path = (runs / "20260718T224559Z_two-vortices_rot24h_r4_l21_dt1h_"
                "d9d06333_929a4b90" / "diagnostics" / "timeseries.csv")
    if not csv_path.is_file():
        pytest.skip("real BVE capsule not present")
    header = csv_path.read_text(encoding="utf-8").splitlines()[0].split(",")
    assert tuple(header) == BVE_DIAGNOSTIC_COLUMNS
    first = csv_path.read_text(encoding="utf-8").splitlines()[1].split(",")
    # Values measured when the capsule was written (spot check of the
    # high-l band fraction and circulation, unchanged by this feature).
    assert first[header.index("high_l_enstrophy_frac")] == \
        "0.0017432476464780014"
    assert first[header.index("circulation")] == "6.024446079377876e-08"


def test_state_schema_for_rejects_unknown_solver_and_missing_lmax():
    from tropoi.representation.archive.schema import SchemaError
    with pytest.raises(SchemaError, match="unknown solver"):
        state_schema_for("mhd", {"lmax": 3})
    with pytest.raises(SchemaError, match="lmax"):
        state_schema_for("bve", {})
    with pytest.raises(SchemaError, match="sigma_interfaces"):
        state_schema_for("pe", {"lmax": 3})
    with pytest.raises(SchemaError, match="disagrees"):
        state_schema_for("pe", {"lmax": 3, "nlev": 2,
                                "sigma_interfaces": [0.0, 0.5, 0.7, 1.0]})
