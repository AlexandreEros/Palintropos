"""The published canonical Williamson-5 T63 run is intact and readable.

The run is byte-exact evidence: every primary file must match its
``SHA256SUMS`` receipt, the receipt must cover exactly the primary files
(derived ``assets/`` never), and the saved-run API must open the run on the
host (no CUDA) as the inferred-schema legacy capsule it is. CPU only.
"""
from __future__ import annotations

import hashlib
import pathlib

import numpy as np

from tropoi.representation.archive import open_simulation

ROOT = pathlib.Path(__file__).resolve().parents[1]
RUN_ID = ("20260730T011700Z_williamson5_rot23p93h_r4_l63_dt120h_45406d82_"
          "668e6c9a")
CAPSULE = ROOT / "docs" / "runs" / RUN_ID
PRIMARY = {"config.json", "manifest.json", "swe_coeffs.npy",
           "swe_snapshot_times.npy", "diagnostics/timeseries.csv"}


def _recorded_sums() -> dict[str, str]:
    sums = {}
    for line in (CAPSULE / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        sums[name.strip()] = digest
    return sums


def test_the_receipt_covers_exactly_the_primary_run_files():
    sums = _recorded_sums()
    assert set(sums) == PRIMARY
    assert not any(name.startswith("assets/") for name in sums)
    on_disk = {path.relative_to(CAPSULE).as_posix()
               for path in CAPSULE.rglob("*") if path.is_file()}
    derived = {name for name in on_disk if name.startswith("assets/")}
    assert on_disk - derived - {"SHA256SUMS"} == PRIMARY


def test_every_primary_file_matches_its_recorded_sha256():
    for name, digest in _recorded_sums().items():
        data = (CAPSULE / name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == digest, name


def test_coefficients_are_those_the_historical_figure_recorded():
    # overview.png (2026-07-30) embeds the SHA-256 of the coefficients it read.
    assert _recorded_sums()["swe_coeffs.npy"] == (
        "dd28eff96795c93d8b15b199be8d3494838ecd763ffa2808b13bb951b60fb1a7")


def test_capsule_opens_through_the_saved_run_api_on_the_host():
    sim = open_simulation(CAPSULE)
    assert sim.solver == "swe"
    np.testing.assert_array_equal(sim.times / 86400.0, [0.0, 5.0, 10.0, 15.0])
    meta = sim.metadata
    assert meta["run_id"] == RUN_ID
    assert meta["git"]["commit"].startswith("668e6c9a")
    assert meta["git"]["dirty"] is False
    assert meta["provenance"]["schema"]["source"] == "inferred"
    assert sim[0].state["phi"].coeffs.shape == (64, 64)
