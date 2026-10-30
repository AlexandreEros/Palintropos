"""The published evolving-Mars run is intact, self-contained and as recorded.

``docs/runs/<RUN_ID>`` is the main run of
docs/validation/mars_evolving_2026-10-27. Its receipt and index entry are
checked for every published run by ``test_published_layout.py``; here, on the
host (no CUDA): the saved-run API opens it as an ``orographic_thermal_wave``
PE run with nine saved times over one Mars rotation, its configuration carries
the perturbation record and the MOLA terrain digest, the capsule's own
``terrain/`` is what makes it self-contained, and the validation record's
numbers and figure name this run's coefficients. CPU only.
"""
from __future__ import annotations

import hashlib
import json
import pathlib

import numpy as np
import pytest

from tropoi.representation.archive import open_simulation

ROOT = pathlib.Path(__file__).resolve().parents[1]
RUN_ID = ("20261027T175327Z_orographic-thermal-wave_rot24p62h_r3_l21_"
          "dt11080p3s_3a46f966_71c5441a")
CAPSULE = ROOT / "docs" / "runs" / RUN_ID
RECORD = ROOT / "docs" / "validation" / "mars_evolving_2026-10-27"
ROTATION_S = 24.6229 * 3600.0
TERRAIN = ("sha256:afb016d805517202bfc4374ed24f91d72b71aa6376b789184d4cd010"
           "b35c31b8")


def _receipt() -> dict[str, str]:
    sums = {}
    for line in (CAPSULE / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        sums[name.strip()] = digest
    return sums


def test_capsule_opens_on_the_host_as_recorded():
    sim = open_simulation(CAPSULE)
    assert sim.solver == "pe"
    np.testing.assert_allclose(sim.times, np.linspace(0.0, ROTATION_S, 9),
                               rtol=0, atol=1e-6)
    config = sim.metadata["run_config"]
    assert config["scenario"] == "orographic_thermal_wave"
    assert (config["lmax"], config["nlat"], config["nlon"],
            config["nlev"]) == (21, 34, 68, 4)
    assert config["dt_seconds"] == 60.0
    assert config["terrain_digest"] == TERRAIN
    assert config["terrain_l_cut"] == 14
    rec = config["initial_perturbation"]
    assert rec["base_state"] == "orographic_isothermal_rest"
    assert (rec["degree"], rec["order"], rec["coefficient_real_K"]) == (
        2, 2, 1.0)
    state = sim[-1].state
    assert set(state) == {"zeta", "delta", "temperature", "ln_ps"}


def test_terrain_travels_with_the_capsule():
    terrain = json.loads((CAPSULE / "terrain" / "terrain.json").read_text(
        encoding="utf-8"))
    assert TERRAIN in json.dumps(terrain)
    assert {"terrain/terrain.json",
            "terrain/topography_elevation_lm.npy"} <= set(_receipt())


def test_the_validation_record_is_about_these_bytes():
    from PIL import Image

    results = json.loads((RECORD / "results.json").read_text(
        encoding="utf-8"))
    assert results["runs"]["main"]["run_id"] == RUN_ID
    assert all(results["screens"].values()), results["screens"]
    coeffs = hashlib.sha256((CAPSULE / "pe_coeffs.npy").read_bytes())
    assert _receipt()["pe_coeffs.npy"] == coeffs.hexdigest()
    with Image.open(RECORD / "initial_final_difference.png") as image:
        assert image.info["CoefficientsSHA256"] == coeffs.hexdigest()
        assert image.info["Source"] == RUN_ID


@pytest.mark.parametrize("name", ["browser_check.json", "results.json"])
def test_the_record_holds_no_local_paths(name):
    text = (RECORD / name).read_text(encoding="utf-8")
    assert "Users" not in text and ":\\\\" not in text and "file://" not in text
