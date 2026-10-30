"""Canonical defaults, option aliases, and persisted configuration keys."""
from __future__ import annotations

import pytest

from tropoi.cli.main import build_parser
from tropoi.run.bve.config import (
    DEFAULT_N_SNAPSHOTS,
)

from .conftest import (
    ADDITIVE_CONFIG_KEYS,
    LEGACY_CONFIG_KEYS,
    run_tropoi_stubbed,
)


def test_canonical_default_is_count_mode_n5(stub_execute_run):
    cfg = run_tropoi_stubbed(["run", "bve"], stub_execute_run)
    assert cfg.snapshot_mode == "count"
    assert cfg.n_snapshots == DEFAULT_N_SNAPSHOTS == 5
    assert cfg.dt_snapshots == 21600.0
    assert cfg.snapshot_times_seconds() == [
        0.0, 21600.0, 43200.0, 64800.0, 86400.0]


def test_defaults_match_legacy_psx_bve_values(stub_execute_run):
    cfg = run_tropoi_stubbed(["run", "bve"], stub_execute_run)
    d = cfg.to_run_config_dict()
    assert set(d) == LEGACY_CONFIG_KEYS | ADDITIVE_CONFIG_KEYS
    assert d["lmax"] == 21
    assert d["grid"] == "geodesic"
    assert d["resolution"] == 4
    assert d["nlat"] == 128 and d["nlon"] == 256
    assert d["day_hours"] == float("inf")
    assert d["duration_days"] == 1.0
    assert d["scenario"] == "two_vortices"
    assert d["viscosity"] == 0.0
    assert d["product_quadrature"] == "fine"
    assert d["out"] == "runs"
    assert d["experiment"] is None
    assert d["overwrite"] is False


@pytest.mark.parametrize("argv,attr,value", [
    (["--backend", "latlon"], "grid", "latlon"),
    (["--grid", "latlon"], "grid", "latlon"),
    (["--backend", "gauss-latlon"], "grid", "latlon"),
    (["--l-max", "8"], "lmax", 8),
    (["--lmax", "8"], "lmax", 8),
    (["--days", "10"], "duration_days", 10.0),
    (["--duration-days", "10"], "duration_days", 10.0),
    (["--snapshot-interval-seconds", "864"], "dt_snapshots", 864.0),
    (["--dt-snapshots", "864"], "dt_snapshots", 864.0),
    (["--radius-earth-units", "2"], "radius_earth_units", 2.0),
], ids=lambda x: " ".join(x) if isinstance(x, list) else str(x))
def test_canonical_and_legacy_spellings(
        stub_execute_run, argv, attr, value):
    cfg = run_tropoi_stubbed(["run", "bve", *argv], stub_execute_run)
    assert getattr(cfg, attr) == value


def test_frozen_argparse_dest_names():
    args = build_parser().parse_args(["run", "bve"])
    for dest in LEGACY_CONFIG_KEYS:
        assert hasattr(args, dest), f"missing frozen dest {dest}"


def test_cli_exposes_and_defaults_product_quadrature(stub_execute_run):
    cfg = run_tropoi_stubbed(["run", "bve"], stub_execute_run)
    assert cfg.to_run_config_dict()["product_quadrature"] == "fine"
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["run", "bve", "--product-quadrature", "exact"])
    args = parser.parse_args(["run", "bve"])
    assert "--product-quadrature" in args._parser.format_help()


def test_canonical_bve_parser_leaves_dt_snapshots_none():
    """Canonical tropoi parser must NOT apply the legacy interval default."""
    args = build_parser().parse_args(["run", "bve"])
    assert args.dt_snapshots is None
    assert args.n_snapshots is None
