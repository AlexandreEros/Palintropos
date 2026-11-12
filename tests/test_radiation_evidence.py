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
