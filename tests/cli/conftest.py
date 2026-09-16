"""Shared fixtures/helpers for the CPU-safe tropoi CLI test modules.

Anything imported from here must remain import-light (stdlib + first-party
only) so the CLI CPU-safety subprocess tests stay meaningful.
"""
from __future__ import annotations

import subprocess
import sys

import pytest

# The historical psx-bve `vars(args)` key set: make_run_id and the on-disk
# config.json schema depend on it. Frozen for this CLI change.
LEGACY_CONFIG_KEYS = {
    "lmax", "grid", "resolution", "nlat", "nlon", "day_hours",
    "radius_earth_units", "duration_days", "dt_snapshots", "scenario",
    "viscosity", "product_quadrature", "out", "experiment", "overwrite",
}
#: Documented additive provenance keys (see config.py module docstring).
ADDITIVE_CONFIG_KEYS = {"snapshot_mode", "n_snapshots", "snapshot_times", "plots"}

#: Modules whose presence in a fresh interpreter proves CLI help/list/inspect
#: touched CUDA or matplotlib. All of these must stay unimported for the
#: CPU-safety subprocess tests to pass.
HEAVY_MODULES = ("cupy", "cupyx", "matplotlib",
                 "tropoi.planet.planet",
                 "tropoi.run.bve.runner",
                 "tropoi.run.pe.runner",
                 "tropoi.viz")


@pytest.fixture
def stub_execute_run(monkeypatch):
    """Replace cli.bve.execute_run with a capturing stub; return the captured cfg."""
    import tropoi.cli.bve as bve_module

    captured = {}

    def fake_execute_run(cfg):
        captured["cfg"] = cfg
        return 0

    monkeypatch.setattr(bve_module, "execute_run", fake_execute_run)
    return captured


def run_tropoi_stubbed(argv, captured):
    from tropoi.cli.main import main
    assert main(argv) == 0
    return captured["cfg"]


def run_psx_bve_stubbed(argv, captured, monkeypatch):
    import tropoi.cli.bve as bve_module
    monkeypatch.setattr(sys, "argv", ["psx-bve", *argv])
    assert bve_module.main() == 0
    return captured["cfg"]


def cpu_safety_probe(import_stmt: str, call_stmt: str) -> str:
    return (
        "import sys\n"
        f"{import_stmt}\n"
        "try:\n"
        f"    code = {call_stmt}\n"
        "except SystemExit as exc:\n"
        "    code = 0 if exc.code in (0, None) else exc.code\n"
        "assert code == 0, f'exit code {code}'\n"
        f"banned = [m for m in {HEAVY_MODULES!r} if m in sys.modules]\n"
        "assert not banned, f'heavy modules imported: {banned}'\n"
    )


def assert_probe_passes(import_stmt: str, call_stmt: str) -> None:
    result = subprocess.run(
        [sys.executable, "-c", cpu_safety_probe(import_stmt, call_stmt)],
        capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
