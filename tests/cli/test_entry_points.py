"""Console-script wiring of the ``palintropos`` distribution.

``tropoi`` is the canonical command; ``aeolus`` and the ``psx-*`` commands are
compatibility entry points that must resolve to the *same* callables rather
than to a parallel implementation. Skipped when the distribution is not
installed (e.g. a bare ``PYTHONPATH=src`` checkout).
"""
from __future__ import annotations

from importlib.metadata import PackageNotFoundError, distribution

import pytest

DISTRIBUTION = "palintropos"

EXPECTED = {
    "tropoi": "tropoi.cli.main:main",
    "aeolus": "tropoi.cli.main:main",
    "psx-bve": "tropoi.cli.bve:main",
    "psx-gen": "tropoi.cli.generate_planet:main",
    "psx-recompile": "tropoi.cli.clear_cache:main",
}


@pytest.fixture(scope="module")
def console_scripts():
    try:
        dist = distribution(DISTRIBUTION)
    except PackageNotFoundError:
        pytest.skip(f"{DISTRIBUTION} is not installed in this environment")
    return {ep.name: ep for ep in dist.entry_points
            if ep.group == "console_scripts"}


def test_all_console_scripts_are_declared(console_scripts):
    assert set(console_scripts) == set(EXPECTED)


@pytest.mark.parametrize("name,target", sorted(EXPECTED.items()))
def test_console_script_targets(console_scripts, name, target):
    assert console_scripts[name].value == target


def test_aeolus_alias_is_the_same_callable(console_scripts):
    """The retained alias must delegate, not duplicate."""
    from tropoi.cli.main import main

    assert console_scripts["aeolus"].load() is main
    assert console_scripts["tropoi"].load() is main
