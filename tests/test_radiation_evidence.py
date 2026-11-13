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


# The flat-Mars development pilot (docs/radiation/MARS_MI_PLAN.md). Criteria M1-M6
# were committed in fba92fb before any pilot code; H and U ran from b3aaef8 on
# one Colab L4. A development experiment, not a reproduction of Thomson & Vallis M-I.
MARS_H = "dev-mars-gray-flat-heated-T21L25-30d-b3aaef8-l4"
MARS_U = "dev-mars-gray-flat-T21L25-30d-b3aaef8-l4"
MARS_CAPSULES = {"dev-mars-gray-flat-heated-T21L25-b3aaef8": MARS_H,
                 "dev-mars-gray-flat-T21L25-b3aaef8": MARS_U}


def _mars_series(name):
    import numpy as np
    s = json.loads((EVIDENCE / name / "series.json").read_text(encoding="utf-8"))
    return {k: np.asarray(v, dtype=float) for k, v in s.items() if isinstance(v, list)}


def test_the_mars_pair_ran_from_one_commit_on_one_machine_after_its_criteria():
    s1, s2 = _summary(MARS_H), _summary(MARS_U)
    for s in (s1, s2):
        assert s["code"]["git_commit"].startswith("b3aaef8") and s["code"]["git_dirty"] is False
        assert s["environment"]["device"] == "NVIDIA L4"
    assert s1["config_sha256"].startswith("9f7eaa0a")       # dev-mars-gray-flat
    assert s2["config_sha256"].startswith("28281209")       # --no-frictional-heating
    assert s1["experiment_id"].startswith("dev-") and s2["experiment_id"].startswith("dev-")


def test_the_mars_energy_and_mass_verdicts_follow_from_the_committed_series():
    """M2, M3 and M6 recomputed here from series.json, independently of
    tools/radiation/judge_mars_pilot.py: days 11-30, F = mean absorbed flux."""
    import numpy as np
    s = _mars_series(MARS_H)
    res, day = s["energy_residual"], s["day"][1:]
    win = (day >= 11) & (day <= 30)
    F = s["absorbed_stellar"][1:][win].mean()
    assert abs(res[win].mean()) <= 1e-3 * F and np.abs(res).max() <= 1e-2 * F       # M2
    cap = json.loads((EVIDENCE / MARS_H / "config.json").read_text(
        encoding="utf-8"))["config"]["surface"]["heat_capacity"]
    system = (np.diff(s["energy_total"]) + cap * np.diff(s["mean_surface_temperature"])) \
        / 86400.0 - s["toa_net_down"][1:]
    assert abs(system[win].mean()) <= 1e-3 * F and np.abs(system).max() <= 1e-2 * F  # M3
    for name in (MARS_H, MARS_U):
        ps = _mars_series(name)["mean_ps"]
        assert abs(ps[-1] / ps[0] - 1.0) <= 1e-5                                       # M6


def test_the_mars_verdict_record_is_the_one_reported_including_its_failures():
    """M1 fails on the wind bound alone (282 / 295 m/s, confined to the two
    levels above 0.2 Pa, where M-I has a sponge this pilot lacks); M5 fails
    on the 11-16 h window alone (17.0 h, the lag of M-I's 8.26e6 J m^-2 K^-1
    slab). Neither criterion is relaxed; this test records the outcome."""
    v = json.loads((EVIDENCE / MARS_H / "verdicts.json").read_text(encoding="utf-8"))
    assert v["criteria_commit"] == "fba92fb"
    assert v["verdict"] == {"M1": False, "M2": True, "M3": True, "M4": True, "M5": False,
                            "M6": True}
    for run in ("H", "U"):
        assert 100.0 < v[run]["t_min"] and v[run]["t_max"] < 400.0 and v[run]["last_day"] == 30
        assert v[run]["max_abs_u"] > 150.0
    assert v["M5"] == {"met": False, "s1_at_least_3x_noise": True, "s1_at_least_5e-4": True,
                       "surface_temperature_max_between_11_and_16_h": False}


@pytest.mark.parametrize("capsule, evidence", sorted(MARS_CAPSULES.items()))
def test_the_mars_pair_is_preserved_and_is_the_source_of_its_evidence(capsule, evidence):
    from tropoi.cli.capsule import read_archive, read_receipt, verify_capsule
    root = CAPSULES / capsule
    sums = read_receipt(root / "SHA256SUMS")
    archive = read_archive(root, sums)
    assert len(sums) == 66
    assert set(archive["omitted_files"]) == set(sums) - set(archive["included_files"])
    assert verify_capsule(root, root, core=True)["checked"] == 4
    published = EVIDENCE / evidence
    for name in ("config.json", "events.jsonl", "series.json"):
        assert (root / name).read_bytes().replace(b"\r\n", b"\n") == \
            (published / name).read_bytes().replace(b"\r\n", b"\n"), name
    assert (root / "means.npz").read_bytes() == (published / "means.npz").read_bytes()
