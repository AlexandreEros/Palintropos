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


def test_the_tide_diagnostic_refuses_non_finite_data_instead_of_reporting_it():
    """The sponge run's first Colab tide sample came from a continuation that
    went non-finite; the fit then reported NaN amplitudes and np.argmax picked
    index 0 (a spurious 00:00 maximum). The tool must abort instead."""
    import importlib.util
    import numpy as np
    path = pathlib.Path(__file__).resolve().parents[1] / "tools" / "radiation" / "mars_tides.py"
    spec = importlib.util.spec_from_file_location("mars_tides", path)
    tides = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tides)
    lat = np.repeat([0.1, 0.2], 4); lon = np.tile(np.linspace(0, 6, 4), 2)
    t = np.arange(10) * 600.0
    field = np.ones((10, 8))
    assert tides._fit(lat, lon, t, field, 88440.0)            # finite data is fitted
    field[3, 2] = np.nan
    with pytest.raises(SystemExit, match="non-finite field"):
        tides._fit(lat, lon, t, field, 88440.0)


# The sponge rerun (MARS_MI_PLAN.md section 2b; predictions committed in 2bdf13f
# before the sponge code). Hs/Us ran from e38890e on one Colab L4.
SPONGE_H = "dev-mars-gray-flat-sponge-heated-T21L25-30d-e38890e-l4"
SPONGE_U = "dev-mars-gray-flat-sponge-T21L25-30d-e38890e-l4"
SPONGE_CAPSULES = {"dev-mars-gray-flat-sponge-heated-T21L25-e38890e": SPONGE_H,
                   "dev-mars-gray-flat-sponge-T21L25-e38890e": SPONGE_U}


def _evidence_json(name, file):
    return json.loads((EVIDENCE / name / file).read_text(encoding="utf-8"))


def test_the_sponge_pair_ran_from_one_commit_on_one_machine_after_its_predictions():
    s1, s2 = _summary(SPONGE_H), _summary(SPONGE_U)
    for s in (s1, s2):
        assert s["code"]["git_commit"].startswith("e38890e") and s["code"]["git_dirty"] is False
        assert s["environment"]["device"] == "NVIDIA L4"
        assert s["experiment_id"].startswith("dev-mars-gray-flat-sponge")
    assert s1["config_sha256"].startswith("d2e44058") and s2["config_sha256"].startswith("1f91ab18")


def test_the_sponge_energy_and_mass_verdicts_follow_from_the_committed_series():
    import numpy as np
    s = _mars_series(SPONGE_H)
    res, day = s["energy_residual"], s["day"][1:]
    win = (day >= 11) & (day <= 30)
    F = s["absorbed_stellar"][1:][win].mean()
    assert abs(res[win].mean()) <= 1e-3 * F and np.abs(res).max() <= 1e-2 * F
    for name in (SPONGE_H, SPONGE_U):
        ps = _mars_series(name)["mean_ps"]
        assert abs(ps[-1] / ps[0] - 1.0) <= 1e-5


def test_the_first_sponge_tide_sample_is_kept_as_the_record_of_a_non_finite_continuation():
    """Colab's tides.json continued past day 30; that continuation goes
    non-finite (the explicit bottom-layer sensible exchange, see the plan),
    so every amplitude is NaN and its 00:00 maximum is np.argmax of a NaN
    curve. It is kept, unchanged, and superseded by tides-last2sols.json."""
    import math
    t = _evidence_json(SPONGE_H, "tides.json")
    assert math.isnan(t["tropical_surface_pressure_s1_over_ps"])
    assert t["equatorial_surface_temperature_max_local_hour"] == 0.0


def test_the_sponge_rerun_met_exactly_its_predictions_on_valid_data():
    """M1 met (the missing sponge was the cause), M5 not met on the
    11-16 h window alone (the slab lag), everything else met. The M5 sample
    is the last two sols of Hs, replayed from its day-20 checkpoint and
    verified against its saved daily states."""
    t = _evidence_json(SPONGE_H, "tides-last2sols.json")
    assert t["window"] == "last" and t["restored_checkpoint"] == "checkpoint-s000002880.npz"
    replay = t["replay_max_relative_departure_by_day"]
    assert sorted(map(int, replay)) == list(range(21, 31)) and max(replay.values()) <= 1e-9
    assert not 11.0 <= t["equatorial_surface_temperature_max_local_hour"] <= 16.0
    assert t["M5"] == {"met": False, "s1_at_least_3x_noise": True, "s1_at_least_5e-4": True,
                       "surface_temperature_max_between_11_and_16_h": False}
    v = _evidence_json(SPONGE_H, "verdicts-last2sols.json")
    assert v["tides_file"] == "tides-last2sols.json"
    assert v["verdict"] == {"M1": True, "M2": True, "M3": True, "M4": True, "M5": False,
                            "M6": True}
    assert v["H"]["max_abs_u"] < 150.0 and v["U"]["max_abs_u"] < 150.0


@pytest.mark.parametrize("capsule, evidence", sorted(SPONGE_CAPSULES.items()))
def test_the_sponge_pair_is_preserved_and_is_the_source_of_its_evidence(capsule, evidence):
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


# The implicit surface exchange (MARS_MI_PLAN.md section 2d; criteria and
# predictions committed in 6701d02 before the code). Hi/Ui ran 60 days from
# 63d05a2, consecutively in one Colab session on a Tesla T4.
IMPLICIT_H = "dev-mars-gray-flat-sponge-implicit-heated-T21L25-60d-63d05a2-t4"
IMPLICIT_U = "dev-mars-gray-flat-sponge-implicit-T21L25-60d-63d05a2-t4"
IMPLICIT_CAPSULES = {"dev-mars-gray-flat-sponge-implicit-heated-T21L25-63d05a2": IMPLICIT_H,
                     "dev-mars-gray-flat-sponge-implicit-T21L25-63d05a2": IMPLICIT_U}


def test_the_implicit_pair_ran_from_one_commit_on_the_machine_its_names_record():
    s1, s2 = _summary(IMPLICIT_H), _summary(IMPLICIT_U)
    for s in (s1, s2):
        assert s["code"]["git_commit"].startswith("63d05a2") and s["code"]["git_dirty"] is False
        assert s["environment"]["device"] == "Tesla T4"          # the "-t4" in the names
        assert s["events"]["resumes"] == 0 and s["events"]["last_event"] == "completed"
    assert s1["config_sha256"].startswith("18ee5dc5") and s2["config_sha256"].startswith("7d28666a")
    assert s1["events"]["last"] <= s2["events"]["first"]          # one after the other


def test_the_implicit_presets_reproduce_the_run_identities():
    from tropoi.run.radiative.config import PRESETS
    preset = PRESETS["dev-mars-gray-flat-sponge-implicit"]
    assert preset().sha256() == _summary(IMPLICIT_H)["config_sha256"]
    assert preset(frictional_heating=False).sha256() == _summary(IMPLICIT_U)["config_sha256"]


def test_the_implicit_energy_verdicts_follow_from_the_committed_series():
    """M2 and M3 recomputed from series.json over days 41-60 (section 2d)."""
    import numpy as np
    s = _mars_series(IMPLICIT_H)
    res, day = s["energy_residual"], s["day"][1:]
    win = (day >= 41) & (day <= 60)
    F = s["absorbed_stellar"][1:][win].mean()
    assert abs(res[win].mean()) <= 1e-3 * F and np.abs(res).max() <= 1e-2 * F       # M2
    cap = json.loads((EVIDENCE / IMPLICIT_H / "config.json").read_text(
        encoding="utf-8"))["config"]["surface"]["heat_capacity"]
    system = (np.diff(s["energy_total"]) + cap * np.diff(s["mean_surface_temperature"])) \
        / 86400.0 - s["toa_net_down"][1:]
    assert abs(system[win].mean()) <= 1e-3 * F and np.abs(system).max() <= 1e-2 * F  # M3


def test_the_implicit_wind_and_mass_failures_follow_from_the_committed_series():
    """M1 fails on the wind bound alone, and only after day 35; M6 fails over
    60 days and also over days 31-60 alone, so the 30-day bound fails at the
    30-day rate as well. Neither bound is relaxed."""
    import numpy as np
    for name in (IMPLICIT_H, IMPLICIT_U):
        s = _mars_series(name)
        u, day = s["max_abs_u"], s["day"]
        assert u.max() > 150.0 and day[u > 150.0].min() > 35 and u[day <= 30].max() < 150.0
        assert s["t_min"].min() > 100.0 and s["t_max"].max() < 400.0 and day[-1] == 60
        ps = s["mean_ps"]
        assert abs(ps[60] / ps[0] - 1.0) > 1e-5 and abs(ps[60] / ps[30] - 1.0) > 1e-5
        assert abs(ps[30] / ps[0] - 1.0) <= 1e-5


def test_the_implicit_verdict_record_is_the_one_reported_including_its_failures():
    """Predicted M1-M4, M6 and N1 met, M5 not met on the window alone. Measured:
    M1 and M6 not met as well. N1 is met: the continuation past day 60 ran all
    its steps (the tool aborts on any non-finite state)."""
    import math
    t = _evidence_json(IMPLICIT_H, "tides.json")
    assert t["window"] == "after" and t["restored_checkpoint"] == "checkpoint-s000008640.npz"
    assert t["steps"] == 295 and t["start_seconds"] == 60 * 86400.0
    assert all(math.isfinite(t[k]) for k in t if k.startswith(("tropical_", "equatorial_")))
    assert t["M5"] == {"met": False, "s1_at_least_3x_noise": True, "s1_at_least_5e-4": True,
                       "surface_temperature_max_between_11_and_16_h": False}
    v = _evidence_json(IMPLICIT_H, "verdicts.json")
    assert v["window_days"] == [41, 60] and v["tides_file"] == "tides.json"
    assert v["verdict"] == {"M1": False, "M2": True, "M3": True, "M4": True, "M5": False,
                            "M6": False}
    assert v["M4"]["unheated_sinks"] < -v["M4"]["tolerance"]       # M4 now discriminates


def test_the_implicit_state_diagnostics_are_the_runs_own_states():
    """state_diagnostics.json (tools/radiation/mars_state_diagnostics.py) agrees
    with each run's series, and its numbers are the ones the plan reports."""
    import math
    for name in (IMPLICIT_H, IMPLICIT_U):
        d = _evidence_json(name, "state_diagnostics.json")
        s = _mars_series(name)
        assert d["config_sha256"] == _summary(name)["config_sha256"]
        assert sorted(map(int, d["days"])) == list(range(1, 61))
        for day, o in d["days"].items():
            assert math.isclose(o["max_abs_u"], s["max_abs_u"][int(day)], rel_tol=1e-9)
            assert math.isclose(o["mean_ps"], s["mean_ps"][int(day)], rel_tol=1e-12)
        # the explicit exchange would be unstable (> 2) on many states; the runs went on
        assert d["largest_exchange_air_number"]["value"] > 6.0
        assert sum(o["exchange_air_points_above_2"] > 0 for o in d["days"].values()) == 41
        # the mass drift is the small remainder of two larger, opposing changes
        last, ln0 = d["days"]["60"], math.log(s["mean_ps"][0])
        drift = last["mean_ps"] / s["mean_ps"][0] - 1.0
        assert drift < 0 and last["mean_ln_ps"] - ln0 < 10 * drift            # <ln p_s> falls
        assert last["variance_ln_ps"] / 2 > 10 * abs(drift)                   # variance grows
        # the excess winds are at 0.5-4 Pa over the southern high latitudes
        over = [o for o in d["days"].values() if o["max_abs_u"] > 150.0]
        assert all(-70 < o["max_latitude_deg"] < -50 and o["max_sigma"] * 610 < 5 for o in over)


def test_the_implicit_exchange_left_the_first_30_days_as_the_sponge_run_had_them():
    """Reported, not judged (section 2d): the day-30 differences from Hs."""
    import numpy as np
    hi, hs = _mars_series(IMPLICIT_H), _mars_series(SPONGE_H)
    assert np.abs(hi["max_abs_u"][:31] - hs["max_abs_u"]).max() < 6.0
    assert abs(hi["mean_surface_temperature"][30] - hs["mean_surface_temperature"][30]) < 0.01
    assert np.abs(hi["mean_ps"][:31] / hs["mean_ps"] - 1.0).max() < 1e-7
    assert np.abs(hi["energy_residual"][:30] - hs["energy_residual"]).max() < 1e-3


@pytest.mark.parametrize("capsule, evidence", sorted(IMPLICIT_CAPSULES.items()))
def test_the_implicit_pair_is_preserved_and_is_the_source_of_its_evidence(capsule, evidence):
    from tropoi.cli.capsule import read_archive, read_receipt, verify_capsule
    root = CAPSULES / capsule
    sums = read_receipt(root / "SHA256SUMS")
    archive = read_archive(root, sums)
    assert len(sums) == 126
    assert set(archive["omitted_files"]) == set(sums) - set(archive["included_files"])
    assert verify_capsule(root, root, core=True)["checked"] == 4
    published = EVIDENCE / evidence
    for name in ("config.json", "events.jsonl", "series.json"):
        assert (root / name).read_bytes().replace(b"\r\n", b"\n") == \
            (published / name).read_bytes().replace(b"\r\n", b"\n"), name
    assert (root / "means.npz").read_bytes() == (published / "means.npz").read_bytes()
