"""Descriptive manifest text is not part of a run's scientific identity.

The PE/SWE manifest notes and the ``state_schema`` / ``diagnostic_definitions``
blocks cite source files by path. Those citations were corrected when the
legacy ``physics/`` and ``run/*/diagnostics.py`` import shims were retired.
These tests pin that such descriptive edits cannot move a run's scientific
configuration hash or run ID, and that every cited path exists.

Published capsules keep their historical text; only newly written manifests
carry the corrected paths. CPU only.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pytest

from tropoi.representation.archive.schema import provenance_blocks
from tropoi.representation.archive.writer import (_config_hash, make_run_id,
                                                  write_run_manifest)
from tropoi.run.bve.config import BVERunConfig
from tropoi.run.pe.config import PERunConfig
from tropoi.run.pe.execution import PE_MANIFEST_NOTES
from tropoi.run.swe.config import SWERunConfig
from tropoi.run.swe.execution import SWE_MANIFEST_NOTES

PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "src" / "tropoi"
NOW = datetime(2026, 10, 31, tzinfo=timezone.utc)

#: Notes text as written before the path correction (2026-10-31).
HISTORICAL_PE_NOTES = {
    **PE_MANIFEST_NOTES,
    "equations": "dry hydrostatic primitive equations in vorticity-divergence "
                 "form on a sigma-coordinate vertical grid (see "
                 "physics/primitive_equations.py and "
                 "docs/PRIMITIVE_EQUATIONS_DESIGN.md)",
    "diagnostics": "see run/pe/diagnostics.py module docstring; total_mass is "
                   "the integral of p_s dA (mass proxy), and no total-energy-"
                   "conservation claim is made",
}
HISTORICAL_SWE_NOTES = {
    **SWE_MANIFEST_NOTES,
    "equations": "rotating shallow-water equations (vorticity-divergence "
                 "form, perturbation thickness geopotential, optional fixed "
                 "bottom topography entering the divergence tendency as "
                 "-laplacian(phi_s); see physics/shallow_water.py and "
                 "docs/SHALLOW_WATER.md)",
    "diagnostics": "see run/swe/diagnostics.py module docstring for "
                   "definitions",
}

#: Hashes and run IDs measured before the path correction. Any change to
#: these means descriptive text leaked into scientific identity.
PINNED = {
    "bve-rh4": (lambda: BVERunConfig.resolve({"scenario": "rh4"}),
                "4382a391",
                "20261031T000000Z_rh4_norot_r4_l21_dt6h_4382a391"),
    "swe-default": (lambda: SWERunConfig.resolve({}),
                    "0e6cfc38",
                    "20261031T000000Z_williamson2_rot23p93h_r4_l21_dt6h_"
                    "0e6cfc38"),
    "swe-williamson5": (
        lambda: SWERunConfig.resolve({"scenario": "williamson5"}),
        "cf9bd4e7",
        "20261031T000000Z_williamson5_rot23p93h_r4_l21_dt6h_cf9bd4e7"),
    "pe-thermal-wave": (
        lambda: PERunConfig.resolve({"scenario": "thermal_wave"}),
        "fa2db863",
        "20261031T000000Z_thermal-wave_rot24h_r3_l10_dt15m_fa2db863"),
    "pe-orographic-rest": (
        lambda: PERunConfig.resolve(
            {"scenario": "orographic_isothermal_rest"}),
        "2ea7567a",
        "20261031T000000Z_orographic-isothermal-rest_rot24h_r3_l10_dt15m_"
        "2ea7567a"),
}

_OBSOLETE = re.compile(r"\bphysics/|\brun/(bve|swe|pe)/diagnostics\.py")
_CITED_PATH = re.compile(r"(?<![\w./-])((?:[a-z_]+/)+[a-z_]+\.py)\b")


def _strings(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for value in obj.values():
            yield from _strings(value)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            yield from _strings(value)


def _descriptive_text(solver: str, run_config: dict) -> list[str]:
    notes = {"pe": PE_MANIFEST_NOTES, "swe": SWE_MANIFEST_NOTES}.get(solver, {})
    return [*_strings(notes),
            *_strings(provenance_blocks(solver, run_config))]


@pytest.mark.parametrize("name", sorted(PINNED))
def test_scientific_hash_and_run_id_are_pinned(name):
    make_cfg, digest, run_id = PINNED[name]
    run_config = make_cfg().to_run_config_dict()
    assert _config_hash(run_config) == digest
    assert make_run_id(run_config, now=NOW, commit=None) == run_id


@pytest.mark.parametrize("name", sorted(PINNED))
def test_descriptive_text_is_not_in_the_run_configuration(name):
    make_cfg, _, _ = PINNED[name]
    solver = name.split("-")[0]
    run_config = make_cfg().to_run_config_dict()
    config_text = json.dumps(run_config, sort_keys=True, default=str)
    for text in _descriptive_text(solver, run_config):
        if len(text) > 40:  # skip short tokens such as units or labels
            assert text not in config_text


@pytest.mark.parametrize("solver, make_cfg, historical, current", [
    ("pe", lambda: PERunConfig.resolve({"scenario": "thermal_wave"}),
     HISTORICAL_PE_NOTES, PE_MANIFEST_NOTES),
    ("swe", lambda: SWERunConfig.resolve({}),
     HISTORICAL_SWE_NOTES, SWE_MANIFEST_NOTES),
])
def test_manifests_with_old_and_new_notes_share_identity(
        tmp_path, solver, make_cfg, historical, current):
    run_config = make_cfg().to_run_config_dict()
    run_id = make_run_id(run_config, now=NOW, commit=None)
    manifests = []
    for label, notes in (("historical", historical), ("current", current)):
        out = tmp_path / label
        out.mkdir()
        write_run_manifest(out, run_config, run_id=run_id, notes=notes,
                           **provenance_blocks(solver, run_config))
        manifests.append(json.loads((out / "manifest.json").read_text(
            encoding="utf-8")))
    old, new = manifests
    assert old["notes"] != new["notes"]
    assert old["run_config"] == new["run_config"] == json.loads(
        json.dumps(run_config, default=str))
    for manifest in manifests:
        assert manifest["run_id"] == run_id
        assert make_run_id(manifest["run_config"], now=NOW,
                           commit=None) == run_id
        assert _config_hash(manifest["run_config"]) == _config_hash(
            run_config)


@pytest.mark.parametrize("solver, make_cfg", [
    ("bve", lambda: BVERunConfig.resolve({"scenario": "rh4"})),
    ("swe", lambda: SWERunConfig.resolve({})),
    ("pe", lambda: PERunConfig.resolve({"scenario": "thermal_wave"})),
])
def test_cited_source_paths_exist_and_are_current(solver, make_cfg):
    texts = _descriptive_text(solver, make_cfg().to_run_config_dict())
    cited = set()
    for text in texts:
        assert not _OBSOLETE.search(text), text
        cited.update(m.group(1) for m in _CITED_PATH.finditer(text))
    cited = {path for path in cited if not path.startswith("docs/")}
    assert cited, "expected the notes to cite at least one source file"
    missing = sorted(p for p in cited if not (PACKAGE_ROOT / p).is_file())
    assert not missing, missing
