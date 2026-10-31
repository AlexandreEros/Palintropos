"""The installed distribution exposes only the canonical tropoi command."""
from __future__ import annotations

from importlib.metadata import PackageNotFoundError, distribution

import pytest

DISTRIBUTION = "palintropos"

EXPECTED = {
    "tropoi": "tropoi.cli.main:main",
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


def test_tropoi_entry_point_loads_main(console_scripts):
    from tropoi.cli.main import main

    assert console_scripts["tropoi"].load() is main
