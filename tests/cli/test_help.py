"""CPU-safety of top-level and subcommand help, list, and inspect --help.

Every help/list path runs in a fresh interpreter and asserts CuPy,
matplotlib, and the numerical modules never landed in ``sys.modules``.
"""
from __future__ import annotations

import pytest

from tropoi.cli.main import main

from .conftest import assert_probe_passes

TROPOI_HELP_INVOCATIONS = [
    ["--help"],
    ["run", "--help"],
    ["run", "bve", "--help"],
    ["list", "--help"],
    ["list", "presets"],
    ["list", "scenarios"],
    ["inspect", "--help"],
    ["gen", "--help"],
    ["recompile", "--help"],
]


@pytest.mark.parametrize("argv", TROPOI_HELP_INVOCATIONS,
                         ids=lambda a: " ".join(a))
def test_tropoi_help_and_list_are_cpu_safe(argv):
    assert_probe_passes("from tropoi.cli.main import main",
                        f"main({argv!r})")


def test_bare_tropoi_prints_help(capsys):
    assert main([]) == 0
    assert "tropoi" in capsys.readouterr().out
