"""The committed canonical Williamson-5 T63 capsule is intact and readable.

The capsule is byte-exact evidence: every file must match the SHA-256 sums
recorded beside it, and the saved-run API must open it on the host (no CUDA)
as the inferred-schema legacy capsule it is. CPU only.
"""
from __future__ import annotations

import hashlib
import pathlib

import numpy as np

from tropoi.representation.archive import open_simulation

ROOT = pathlib.Path(__file__).resolve().parents[1]
CAPSULES = ROOT / "docs" / "validation" / "williamson_5" / "capsules"
RUN_ID = ("20260730T011700Z_williamson5_rot23p93h_r4_l63_dt120h_45406d82_"
          "668e6c9a")
CAPSULE = CAPSULES / "t63" / RUN_ID


def _recorded_sums() -> dict[str, str]:
    sums = {}
    for line in (CAPSULES / "SHA256SUMS").read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        digest, name = line.split(maxsplit=1)
        sums[name.strip()] = digest
    return sums


def test_every_capsule_file_matches_its_recorded_sha256():
    sums = _recorded_sums()
    on_disk = {path.relative_to(CAPSULES).as_posix()
               for path in CAPSULE.rglob("*") if path.is_file()}
    assert on_disk == set(sums)
    for name, digest in sums.items():
        data = (CAPSULES / name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == digest, name


def test_coefficients_are_those_the_historical_figure_recorded():
    # overview.png (2026-07-30) embeds the SHA-256 of the coefficients it read.
    assert _recorded_sums()[f"t63/{RUN_ID}/swe_coeffs.npy"] == (
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
