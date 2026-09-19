"""CPU-safety of the shared support contract (fresh-interpreter probes).

``tropoi.support`` is consumed by configuration, diagnostics, the CLI, and
the numerical cores alike, so importing it (and the CLI paths that consult
it) must never load CuPy, matplotlib, or the heavy runner/visualization
modules. Each probe runs in a fresh subprocess and asserts the banned
modules are absent from ``sys.modules`` (see conftest.HEAVY_MODULES).
"""
from __future__ import annotations

import subprocess
import sys

from .conftest import HEAVY_MODULES, assert_probe_passes


def test_importing_support_is_cpu_safe():
    assert_probe_passes(
        "from tropoi.support import product_truncation_cut",
        "0 if product_truncation_cut(21) == 14 else 1")


def test_support_import_loads_only_the_support_module():
    # Stronger than the banned-module check: importing tropoi.support must
    # not drag in ANY other tropoi subpackage (it is the neutral module).
    code = (
        "import sys\n"
        "import tropoi.support\n"
        "loaded = sorted(m for m in sys.modules if m.startswith('tropoi'))\n"
        "assert loaded == ['tropoi', 'tropoi.support'], loaded\n"
        f"banned = [m for m in {HEAVY_MODULES!r} if m in sys.modules]\n"
        "assert not banned, banned\n"
    )
    result = subprocess.run([sys.executable, "-c", code],
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr


def test_config_resolution_using_the_cut_is_cpu_safe():
    # SWE/PE configuration resolution consults the cut for preset
    # validation; resolving at a supported boundary must stay CPU-only.
    assert_probe_passes(
        "from tropoi.run.swe.config import SWERunConfig\n"
        "from tropoi.run.pe.config import PERunConfig",
        "0 if (SWERunConfig.resolve({'scenario': 'williamson2', 'lmax': 3})"
        " and PERunConfig.resolve({'scenario': 'thermal_wave', 'lmax': 3}))"
        " else 1")
