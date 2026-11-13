"""Energy-budget acceptance criteria, applied to the committed long-run evidence.

CPU-only. The criteria are the ones stated in docs/radiation/README.md
("Acceptance criteria and supported regimes"). They were written on
2026-11-12, after the runs below existed, so for these runs they are a
record, not a prediction; they are the bar for runs made from now on.

With F the radiative throughput of a run (absorbed stellar flux plus
interior heat flux, W m^-2):

    A1  the run reached its configured last day;
    A2  |mean energy-budget residual over the last 100 days| <= 1e-3 F;
    A3  largest single-day |residual| of the whole run       <= 1e-2 F;
    A4  |relative drift of mean surface pressure|            <= 1e-5.

A run "meets the energy criteria" when all four hold.
"""
from __future__ import annotations

import json
import pathlib

import pytest

EVIDENCE = pathlib.Path(__file__).resolve().parents[1] / "docs" / "radiation" / "evidence"
HEATED = ("earth-gray-dry-heated-T21L25-300d", "earth-gray-dry-heated-T21L25-300d-colab",
          "earth-gray-dry-heated-T21L25-300d-d376c67-l4",
          "jupiter-sl09-heated-T21L30-200d-colab")
MEAN_RESIDUAL, DAILY_RESIDUAL, PRESSURE_DRIFT = 1e-3, 1e-2, 1e-5


def assess(name: str) -> dict:
    summary = json.loads((EVIDENCE / name / "summary.json").read_text(encoding="utf-8"))
    series = summary["series"]
    window = series["window"]
    throughput = window["absorbed_stellar"]["mean"] + window["interior"]["mean"]
    return {
        "A1": summary["events"]["last_event"] == "completed" and series["window_days"] == 100,
        "A2": abs(window["energy_residual"]["mean"]) <= MEAN_RESIDUAL * throughput,
        "A3": series["max_abs_energy_residual"] <= DAILY_RESIDUAL * throughput,
        "A4": abs(series["mean_surface_pressure_drift"]) <= PRESSURE_DRIFT,
        "frictional_heating": json.loads((EVIDENCE / name / "config.json").read_text(
            encoding="utf-8"))["config"].get("frictional_heating", False),
    }


@pytest.mark.parametrize("name", HEATED)
def test_heated_runs_meet_the_energy_criteria(name):
    result = assess(name)
    assert result.pop("frictional_heating") is True
    assert all(result.values()), result


def test_the_unheated_earth_run_fails_the_residual_criteria_and_nothing_else():
    """Without frictional heating the eddying Earth run loses 5.1 W m^-2 of
    234.6 (A2, by a factor of 22) and peaks at 12.1 on one day (A3): the
    criteria separate the two formulations."""
    result = assess("earth-gray-dry-T21L25-300d")
    assert result == {"A1": True, "A2": False, "A3": False, "A4": True,
                      "frictional_heating": False}


def test_the_unheated_jupiter_run_meets_them_only_because_its_flow_is_weak():
    """Winds below 1 m/s remove almost nothing, so this run passes without
    heating; it is not evidence that heating is unnecessary on a fast
    rotator (see the T9 100-bar measurement in the README)."""
    result = assess("jupiter-sl09-T21L30-200d")
    assert result == {"A1": True, "A2": True, "A3": True, "A4": True,
                      "frictional_heating": False}


# The controlled comparison of 2026-11-13 (docs/radiation/README.md, "Frictional
# heating becomes the preset default"). C1-C4 were committed (d376c67) before
# either run started; R1 and R2 then ran from that commit on one Colab L4.
R1 = "earth-gray-dry-heated-T21L25-300d-d376c67-l4"
R2 = "earth-gray-dry-T21L25-300d-d376c67-l4"


def _summary(name: str) -> dict:
    return json.loads((EVIDENCE / name / "summary.json").read_text(encoding="utf-8"))


def test_the_controlled_pair_ran_from_the_preregistering_commit_on_one_machine():
    s1, s2 = _summary(R1), _summary(R2)
    for s in (s1, s2):
        assert s["code"]["git_commit"].startswith("d376c67")
        assert s["code"]["git_dirty"] is False
    assert s1["environment"]["device"] == s2["environment"]["device"]
    assert s1["config_sha256"].startswith("748cec4b")       # the preset as shipped
    assert s2["config_sha256"].startswith("e31ae183")       # --no-frictional-heating


def test_c1_the_heated_preset_meets_the_energy_criteria():
    result = assess(R1)
    assert result.pop("frictional_heating") is True
    assert all(result.values()), result


def test_c2_the_unheated_preset_reproduces_the_defect_at_the_current_code():
    assert assess(R2) == {"A1": True, "A2": False, "A3": False, "A4": True,
                          "frictional_heating": False}


def test_c3_the_startup_step_fix_does_not_matter_after_spin_up():
    window = _summary(R2)["series"]["window"]
    assert abs(window["energy_residual"]["mean"] - (-5.10)) <= 0.5


def test_c4_the_batched_heating_behaves_like_the_unbatched_term_over_a_long_run():
    window = _summary(R1)["series"]["window"]
    assert abs(window["energy_residual"]["mean"]) <= 0.5
    assert abs(window["mean_surface_temperature"]["mean"] - 302.65) <= 0.3


CAPSULES = pathlib.Path(__file__).resolve().parents[1] / "docs" / "runs"


@pytest.mark.parametrize("capsule, evidence", [("earth-gray-dry-heated-T21L25-d376c67", R1),
                                               ("earth-gray-dry-T21L25-d376c67", R2)])
def test_the_controlled_pair_is_preserved_and_is_the_source_of_its_evidence(capsule, evidence):
    """The capsule's receipt covers every original file (66: core, 60 states,
    2 checkpoints) with a Drive record for each, its Git core verifies, and
    the published summary inputs are the capsule's bytes."""
    from tropoi.cli.capsule import read_archive, read_receipt, verify_capsule
    root = CAPSULES / capsule
    sums = read_receipt(root / "SHA256SUMS")
    archive = read_archive(root, sums)
    assert len(sums) == 66
    assert sorted(archive["included_files"]) == ["config.json", "events.jsonl", "means.npz",
                                                 "series.json"]
    assert set(archive["omitted_files"]) == set(sums) - set(archive["included_files"])
    assert verify_capsule(root, root, core=True)["checked"] == 4
    published = EVIDENCE / evidence
    for name in ("config.json", "events.jsonl", "series.json"):
        assert (root / name).read_bytes().replace(b"\r\n", b"\n") == \
            (published / name).read_bytes().replace(b"\r\n", b"\n"), name
    assert (root / "means.npz").read_bytes() == (published / "means.npz").read_bytes()
    assert _summary(evidence)["config_sha256"] == json.loads(
        (root / "config.json").read_text(encoding="utf-8"))["config_sha256"]
