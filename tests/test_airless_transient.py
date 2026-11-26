"""Finite transients and saved-configuration documents for the airless runner."""
import json
import pathlib

import numpy as np
import pytest

from tropoi.run.airless.__main__ import main
from tropoi.run.airless.config import (AirlessConfig, config_from_document,
                                       load_config_document, mercury_vpw99)
from tropoi.run.airless.experiment import (AirlessRun, build_columns, build_substrate,
                                           run_block, transient_steps)

CERES = pathlib.Path(__file__).parent / "fixtures" / "airless" / "ceres-equatorial-column.airless.json"
CERES_SHA = "13dfd66a305cadb5932c8c7e001ef9a3e2f90c6b8eb304ce0788969812750c8a"


def _short_ceres(cycle_steps=24):
    """The Ceres fixture with a short segment, for fast runs (test only)."""
    cfg, _ = load_config_document(CERES)
    return cfg.with_(cycle_steps=cycle_steps, spinup_cycle_steps=None,
                     output={**cfg.output, "surface_samples": 12, "profile_samples": 12})


def test_ceres_document_loads_from_its_envelope():
    cfg, source = load_config_document(CERES)
    assert cfg.sha256() == CERES_SHA
    assert source["envelope"] is True
    doc = json.loads(CERES.read_text())
    assert config_from_document(doc["config"]).sha256() == CERES_SHA        # bare form
    assert config_from_document({"config": doc["config"],
                                 "config_sha256": CERES_SHA}).sha256() == CERES_SHA


def test_config_document_refuses_mismatch_and_malformed():
    doc = json.loads(CERES.read_text())
    with pytest.raises(ValueError, match="hashes to"):
        config_from_document({"config": doc["config"], "config_sha256": "0" * 64})
    body = dict(doc["config"])
    del body["strata"]
    with pytest.raises(ValueError, match="missing \\['strata'\\]"):
        config_from_document(body)
    with pytest.raises(ValueError, match="unknown \\['colour'\\]"):
        config_from_document({**doc["config"], "colour": "red"})
    with pytest.raises(ValueError, match="JSON object"):
        config_from_document([doc])


def test_explicit_time_arguments_are_the_cycle_form(tmp_path):
    """A transient segment's (dt, first_step) reproduces cycle c bit for bit."""
    cfg = mercury_vpw99(columns={"kind": "points", "lat_deg": [0.0], "lon_deg": [90.0]})
    cfg = cfg.with_(output={**cfg.output, "surface_samples": 1, "profile_samples": 1})
    st = build_substrate(cfg).initial_state(1)
    steps, cycle = 8, 3
    runs = []
    for i, extra in enumerate(({}, {"dt": cfg.cycle_s / steps, "first_step": cycle * steps})):
        d = tmp_path / str(i)
        d.mkdir()
        runs.append(run_block(cfg.to_dict(), *build_columns(cfg), st.anomaly, st.surface_anomaly,
                              cycle, steps, True, str(d), None, 0, **extra))
    a, b = runs
    for key in ("anomaly", "surface_anomaly", "energy_end", "iterations"):
        np.testing.assert_array_equal(a[key], b[key])
    for key in a["sums"]:
        np.testing.assert_array_equal(a["sums"][key], b["sums"][key])
    assert a["config_sha256"] == b["config_sha256"] == cfg.sha256()
    np.testing.assert_array_equal(np.load(tmp_path / "0" / "every_step-b000.npy"),
                                  np.load(tmp_path / "1" / "every_step-b000.npy"))


def test_transient_steps_from_count_or_duration():
    cfg = _short_ceres()
    dt = cfg.cycle_s / cfg.cycle_steps
    assert transient_steps(cfg, steps=30)[0] == 30
    assert transient_steps(cfg, duration_s=30 * dt)[0] == 30
    assert transient_steps(cfg, duration_s=30.5 * dt)[0] == 31
    n, req = transient_steps(cfg, duration_s=0.25 * dt)
    assert n == 1 and req["final_time_s"] == dt and req["duration_s"] == 0.25 * dt
    for bad in ({"steps": 0}, {"duration_s": -1.0}, {}, {"steps": 1, "duration_s": 1.0}):
        with pytest.raises(ValueError):
            transient_steps(cfg, **bad)


def test_transient_equals_consecutive_worker_cycles(tmp_path):
    """run_transient over whole segments is the same arithmetic as calling
    the worker cycle by cycle (the scratch reference of the Ceres transient)."""
    cfg = _short_ceres()
    run = AirlessRun(cfg, tmp_path / "run", allow_dirty=True, log=lambda s: None)
    result = run.run_transient(48)
    assert result["status"] == "COMPLETED_TRANSIENT"
    assert result["final_time_s"] == 48 * (cfg.cycle_s / 24)
    claims = result["claims"]
    assert set(claims) == {"solver_diagnostics", "numerical_reproduction",
                           "periodic_equilibrium", "physical_validation"}
    assert claims["numerical_reproduction"].startswith("not evaluated")
    assert claims["periodic_equilibrium"].startswith("not evaluated")
    assert claims["physical_validation"] == "none"
    lat, lon, w = build_columns(cfg)
    st = build_substrate(cfg).initial_state(1)
    state = (st.anomaly, st.surface_anomaly)
    for c in range(2):
        d = tmp_path / f"ref{c}"
        d.mkdir()
        res = run_block(cfg.to_dict(), lat, lon, w, *state, c, 24, True, str(d), None, 0)
        state = (res["anomaly"], res["surface_anomaly"])
    end = np.load(tmp_path / "run" / "retained" / "state.npz")
    np.testing.assert_array_equal(end["end_anomaly"], state[0])
    np.testing.assert_array_equal(end["end_surface_anomaly"], state[1])
    cycles = json.loads((tmp_path / "run" / "cycles.json").read_text())
    assert [c["periodic"] for c in cycles] == [None, None]
    assert all(c["worker_config_sha256"] == cfg.sha256() for c in cycles)


def test_cli_runs_a_saved_envelope_as_a_transient(tmp_path):
    cfg = _short_ceres()
    doc = tmp_path / "ceres-short.json"
    doc.write_text(json.dumps({"note": "envelope", "config": cfg.to_dict()}))
    out = tmp_path / "run"
    assert main(["run", "--config", str(doc), "--steps", "30", "--out", str(out),
                 "--allow-dirty"]) == 0
    meta = json.loads((out / "config.json").read_text())
    assert meta["config_sha256"] == cfg.sha256()
    assert meta["config_source"]["envelope"] is True
    status = json.loads((out / "retained" / "status.json").read_text())
    assert status["status"] == "COMPLETED_TRANSIENT"
    assert status["steps"] == 30 and status["segments"] == 2
    cycles = json.loads((out / "cycles.json").read_text())
    assert [(c["first_step"], c["steps"]) for c in cycles] == [(0, 24), (24, 6)]
    assert cycles[-1]["t_end_s"] == status["final_time_s"] == 30 * (cfg.cycle_s / 24)
    assert (out / "SHA256SUMS").exists()


@pytest.mark.parametrize("extra", [["--points", "0,0"], ["--no-spinup"],
                                   ["--steps", "2", "--max-wall-s", "1"],
                                   ["--steps", "2", "--duration", "10"]])
def test_cli_refuses_conflicting_options(tmp_path, extra):
    with pytest.raises(SystemExit):
        main(["show-config", "--config", str(CERES), *extra])


def test_cli_preset_periodic_run_with_one_worker(tmp_path):
    """The periodic preset path runs with one worker on every platform (no
    fork context is requested), and keeps its TRANSIENT label when stopped
    by the wall-clock limit."""
    out = tmp_path / "run"
    assert main(["run", "--preset", "mercury-vpw99", "--points", "0,90", "--cycle-steps", "720",
                 "--spinup-cycle-steps", "72", "--max-wall-s", "0", "--out", str(out),
                 "--allow-dirty"]) == 0
    status = json.loads((out / "retained" / "status.json").read_text())
    assert status["status"] == "TRANSIENT"
    assert status["label"].startswith("TRANSIENT (cycle 1; G-P2 not met)")
    meta = json.loads((out / "config.json").read_text())
    assert meta["config_source"] is None
    cycles = json.loads((out / "cycles.json").read_text())
    assert [c["stage"] for c in cycles] == ["spinup", "production"]
    assert all(c["periodic"] is False for c in cycles)
