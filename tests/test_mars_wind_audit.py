"""tools/radiation/mars_wind_audit.py: the arithmetic and its committed outputs.

CPU-only (CI has no GPU). The tool's per-day and summary functions are plain
NumPy and are tested on synthetic arrays; the committed ``wind_audit.json``
files of the Hb/Ub pair are checked to be the runs' own (same config hash,
same recorded winds) and to carry the numbers MARS_MI_PLAN.md section 4.6 and
MARS_MI_REQUIREMENTS_AUDIT.md section 3 report.
"""
from __future__ import annotations

import importlib.util
import json
import math
import pathlib

import pytest

np = pytest.importorskip("numpy")

ROOT = pathlib.Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs" / "radiation" / "evidence"
HB = "dev-mars-gray-flat-bl-heated-T21L25-60d-06f5356-l4"
UB = "dev-mars-gray-flat-bl-T21L25-60d-06f5356-l4"


@pytest.fixture(scope="module")
def audit():
    path = ROOT / "tools" / "radiation" / "mars_wind_audit.py"
    spec = importlib.util.spec_from_file_location("mars_wind_audit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_importing_the_tool_does_not_import_cupy(audit):
    source = (ROOT / "tools" / "radiation" / "mars_wind_audit.py").read_text(encoding="utf-8")
    top = [line for line in source.splitlines() if line.startswith(("import ", "from "))]
    assert not any("cupy" in line for line in top)


def _synthetic():
    K, nlat, nlon = 3, 4, 5
    sigma = np.array([0.005, 0.3, 0.95])                  # level 0 above sigma 0.01
    latitude = np.array([-60.0, -20.0, 20.0, 60.0])
    u = np.zeros((K, nlat, nlon))
    v = np.zeros((K, nlat, nlon))
    u[0, 0, :] = 80.0                                     # strongest zonal mean, sigma 0.005
    u[1, 3, :] = 50.0                                     # strongest below sigma 0.01
    u[1, 2, 1] = -120.0                                   # strongest grid point, westward
    u[2, 1, 3] = -30.0                                    # strongest lowest-level |u|
    u[2, 1, 4] = 10.0
    u[1, 0, :] = -30.0                                    # most negative zonal mean
    v[2, 2, 2] = -25.0
    v[0, 0, 0] = 99.0                                     # not the lowest level
    speed = np.hypot(u[-1], v[-1]) + 1.0
    return u, v, speed, sigma, latitude


def test_a_day_record_finds_each_extreme_where_it_was_put(audit):
    u, v, speed, sigma, latitude = _synthetic()
    r = audit.day_record(u, v, speed, sigma, latitude)
    assert (r["max_abs_u"], r["max_abs_u_sigma"], r["max_abs_u_latitude_deg"]) == (120.0, 0.3, 20.0)
    assert r["lowest_level_abs_u_max"] == 30.0
    assert r["lowest_level_speed_max"] == 31.0
    assert r["lowest_level_abs_v_max"] == 25.0
    assert (r["zonal_mean_u_max"], r["zonal_mean_u_max_sigma"],
            r["zonal_mean_u_max_latitude_deg"]) == (80.0, 0.005, -60.0)
    assert r["zonal_mean_u_min"] == -30.0
    assert (r["zonal_mean_u_max_sigma_above_0.01"], r["zonal_mean_u_max_sigma_above_0.01_sigma"],
            r["zonal_mean_u_max_sigma_above_0.01_latitude_deg"]) == (50.0, 0.3, 60.0)


def _days(audit, bottom_u, speed):
    """Synthetic per-day records with the given lowest-level |u| and speed."""
    keys = audit.SUMMARIZED
    return {d: {**{k: 0.0 for k in keys}, "lowest_level_abs_u_max": bu,
                "lowest_level_speed_max": sp}
            for d, (bu, sp) in enumerate(zip(bottom_u, speed), start=1)}


def test_the_summary_window_starts_on_the_first_day_and_counts_strictly(audit):
    #            day  1     2     3     4     5     6     7
    bottom_u = [90.0, 39.0, 40.0, 41.0, 45.0, 45.0, 10.0]
    speed = [95.0, 41.0, 40.0, 50.0, 46.0, 48.0, 40.000001]
    s = audit.summarize(_days(audit, bottom_u, speed), first_day=2)
    assert (s["first_day"], s["last_day"], s["days"]) == (2, 7, 6)
    # day 1 (90, 95) is outside the window; the tie at 45 goes to the earlier day
    assert s["lowest_level_abs_u_max"] == {"max": 45.0, "day": 5, "days_above_threshold": 3}
    # exactly 40 is not above 40
    assert s["lowest_level_speed_max"] == {"max": 50.0, "day": 4, "days_above_threshold": 5}
    assert set(s) >= set(audit.SUMMARIZED)


def test_the_summary_accepts_string_days_and_refuses_an_empty_window(audit):
    days = {str(d): r for d, r in _days(audit, [41.0, 42.0], [43.0, 44.0]).items()}
    assert audit.summarize(days, first_day=1)["lowest_level_speed_max"]["day"] == 2
    with pytest.raises(ValueError, match="no saved day"):
        audit.summarize(days, first_day=11)


def test_a_state_that_is_not_the_runs_own_is_refused(audit):
    series = {"day": [3], "max_abs_u": [100.0], "max_wind_bottom": [40.0]}
    audit.check_against_series(3, {"max_abs_u": 100.0, "lowest_level_speed_max": 40.0}, series)
    audit.check_against_series(4, {"max_abs_u": 1.0, "lowest_level_speed_max": 1.0}, series)
    with pytest.raises(SystemExit, match="max_wind_bottom"):
        audit.check_against_series(3, {"max_abs_u": 100.0, "lowest_level_speed_max": 40.1}, series)
    with pytest.raises(SystemExit, match="max_abs_u"):
        audit.check_against_series(3, {"max_abs_u": 100.1, "lowest_level_speed_max": 40.0}, series)
    # a run without the speed series is checked on max_abs_u alone
    audit.check_against_series(3, {"max_abs_u": 100.0, "lowest_level_speed_max": 7.0},
                               {"day": [3], "max_abs_u": [100.0]})


@pytest.mark.parametrize("name", (HB, UB))
def test_the_committed_wind_audit_is_the_runs_own(audit, name):
    d = json.loads((EVIDENCE / name / "wind_audit.json").read_text(encoding="utf-8"))
    summary = json.loads((EVIDENCE / name / "summary.json").read_text(encoding="utf-8"))
    series = json.loads((EVIDENCE / name / "series.json").read_text(encoding="utf-8"))
    assert d["config_sha256"] == summary["config_sha256"]
    assert d["checked_against_series"] == ["max_abs_u", "max_wind_bottom"]
    assert sorted(map(int, d["days"])) == list(range(1, 61))
    for day, o in d["days"].items():
        audit.check_against_series(int(day), o, series)
        assert o["lowest_level_abs_u_max"] <= o["lowest_level_speed_max"] * (1 + 1e-12)
    assert d["summary"] == json.loads(json.dumps(audit.summarize(d["days"], 11)))


# (registered |u| max, day, days > 40), (judged speed max, day, days > 40),
# largest lowest-level |v|, (largest zonal-mean u, day), (largest grid-point |u|, day)
REPORTED = {HB: ((43.1, 57, 6), (46.7, 27, 13), 38.8, (94.6, 59), (170.0, 42)),
            UB: ((46.3, 47, 2), (47.2, 47, 4), 39.9, (90.4, 58), (186.0, 56))}


@pytest.mark.parametrize("name", (HB, UB))
def test_the_committed_wind_audit_gives_the_numbers_the_plan_and_audit_report(name):
    s = json.loads((EVIDENCE / name / "wind_audit.json").read_text(encoding="utf-8"))["summary"]
    (bu, bu_day, bu_n), (sp, sp_day, sp_n), v, (zm, zm_day), (gp, gp_day) = REPORTED[name]
    assert (s["first_day"], s["last_day"]) == (11, 60)
    b = s["lowest_level_abs_u_max"]
    assert (round(b["max"], 1), b["day"], b["days_above_threshold"]) == (bu, bu_day, bu_n)
    b = s["lowest_level_speed_max"]
    assert (round(b["max"], 1), b["day"], b["days_above_threshold"]) == (sp, sp_day, sp_n)
    assert round(s["lowest_level_abs_v_max"]["max"], 1) == v
    assert (round(s["zonal_mean_u_max"]["max"], 1), s["zonal_mean_u_max"]["day"]) == (zm, zm_day)
    assert math.isclose(s["max_abs_u"]["max"], gp, abs_tol=0.5) and s["max_abs_u"]["day"] == gp_day
