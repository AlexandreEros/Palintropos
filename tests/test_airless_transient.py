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


def _columns(cfg):
    """(lat, lon in radians, weights) of the configured columns, as run_block takes them."""
    c = build_columns(cfg)
    return c.lat, c.lon, c.weights


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
        runs.append(run_block(cfg.to_dict(), *_columns(cfg), st.anomaly, st.surface_anomaly,
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
    run = AirlessRun(cfg, tmp_path / "run", backend="numpy", allow_dirty=True, log=lambda s: None)
    result = run.run_transient(48)
    assert result["status"] == "COMPLETED_TRANSIENT"
    assert result["final_time_s"] == 48 * (cfg.cycle_s / 24)
    claims = result["claims"]
    assert set(claims) == {"solver_diagnostics", "numerical_reproduction",
                           "periodic_equilibrium", "physical_validation"}
    assert claims["numerical_reproduction"].startswith("not evaluated")
    assert claims["periodic_equilibrium"].startswith("not evaluated")
    assert claims["physical_validation"] == "none"
    c = build_columns(cfg)
    lat, lon, w = c.lat, c.lon, c.weights
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
                 "--allow-dirty", "--device", "cpu"]) == 0
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
                 "--allow-dirty", "--device", "cpu"]) == 0
    status = json.loads((out / "retained" / "status.json").read_text())
    assert status["status"] == "TRANSIENT"
    assert status["label"].startswith("TRANSIENT (cycle 1; G-P2 not met)")
    meta = json.loads((out / "config.json").read_text())
    assert meta["config_source"] is None
    cycles = json.loads((out / "cycles.json").read_text())
    assert [c["stage"] for c in cycles] == ["spinup", "production"]
    assert all(c["periodic"] is False for c in cycles)


def test_short_final_segment_keeps_the_production_cadence(tmp_path):
    """A 7-step final segment samples every 2 steps (24 steps / 12 samples),
    not gcd(12, 7) = 1 sample; a 1-step segment has none and no range."""
    cfg = _short_ceres()
    run = AirlessRun(cfg, tmp_path / "a", backend="numpy", allow_dirty=True, log=lambda s: None)
    run.run_transient(31)
    ret = tmp_path / "a" / "retained"
    assert np.load(ret / "surface-b000.npy").shape[0] == 3
    assert np.load(ret / "profiles-b000.npy").shape[0] == 3
    assert tuple(np.load(ret / "state.npz")["sample_every"]) == (2, 2)
    run = AirlessRun(cfg, tmp_path / "b", backend="numpy", allow_dirty=True, log=lambda s: None)
    run.run_transient(25)
    cycles = json.loads((tmp_path / "b" / "cycles.json").read_text())
    assert cycles[-1]["steps"] == 1 and cycles[-1]["surface_T_range_K"] is None
    assert cycles[0]["surface_T_range_K"] is not None


def test_standalone_replay_of_a_transient_tail(tmp_path):
    """G-G replays a retained short transient segment with its own timing."""
    from tools.planet_one.global_check import main as gg
    cfg = _short_ceres()
    run = AirlessRun(cfg, tmp_path / "run", backend="numpy", allow_dirty=True, log=lambda s: None)
    run.run_transient(31)
    assert gg([str(tmp_path / "run"), "--columns", "0"]) == 0
    report = json.loads((tmp_path / "run" / "gg.json").read_text())
    assert report["timing"]["first_step"] == 24 and report["timing"]["sample_every"] == [2, 2]
    assert report["pass"] and report["max_abs_end_state_diff_K"] == 0.0


def test_transient_refuses_fractional_steps(tmp_path):
    run = AirlessRun(_short_ceres(), tmp_path / "run", backend="numpy", allow_dirty=True,
                     log=lambda s: None)
    for bad in (2.5, 0, True):
        with pytest.raises(ValueError, match="whole number"):
            run.run_transient(bad)


def test_config_document_reports_wrong_types_as_value_errors():
    doc = json.loads(CERES.read_text())
    with pytest.raises(ValueError, match="not an airless configuration"):
        config_from_document({**doc["config"], "cycle_steps": "4320"})


def test_standalone_replay_of_a_segment_without_samples(tmp_path):
    """A 1-step tail holds no tier-2 samples: the replay reports them as not
    compared (None, never 0) and judges the end state alone."""
    from tools.planet_one.global_check import main as gg
    run = AirlessRun(_short_ceres(), tmp_path / "run", backend="numpy", allow_dirty=True,
                     log=lambda s: None)
    run.run_transient(25)
    assert gg([str(tmp_path / "run"), "--columns", "0"]) == 0
    report = json.loads((tmp_path / "run" / "gg.json").read_text())
    assert report["samples"] == {"surface": 0, "profiles": 0}
    assert report["max_abs_surface_fields_diff"] is None
    assert report["max_abs_profile_diff_K"] is None
    assert report["compared"] == ["end_state"]
    assert set(report["not_compared"]) == {"surface_fields", "profiles"}
    assert report["pass"] and report["max_abs_end_state_diff_K"] == 0.0


def test_standalone_replay_needs_columns_for_point_runs(tmp_path):
    from tools.planet_one.global_check import main as gg
    run = AirlessRun(_short_ceres(), tmp_path / "run", backend="numpy", allow_dirty=True,
                     log=lambda s: None)
    run.run_transient(24)
    with pytest.raises(SystemExit):
        gg([str(tmp_path / "run")])


@pytest.mark.parametrize("key", ["surface_samples", "profile_samples"])
@pytest.mark.parametrize("bad", [0, -4, 2.0, "12", True, None])
def test_config_refuses_bad_sample_counts(key, bad):
    """A zero sample count used to raise ZeroDivisionError in validation."""
    doc = json.loads(CERES.read_text())
    body = {**doc["config"], "output": {**doc["config"]["output"], key: bad}}
    with pytest.raises(ValueError, match=f"output {key} must be a positive integer"):
        AirlessConfig.from_dict(body)
    with pytest.raises(ValueError, match="not an airless configuration"):
        config_from_document(body)


@pytest.mark.parametrize("key,bad", [("albedo", "0.1"), ("emissivity", None),
                                     ("geothermal_flux", float("nan")), ("cycle_s", True),
                                     ("radius_m", -1.0), ("initial_temperature", 0.0),
                                     ("cycle_steps", 0), ("cycle_steps", 24.0),
                                     ("spinup_cycle_steps", 0)])
def test_config_refuses_wrongly_typed_or_out_of_range_scalars(key, bad):
    """Checked, not coerced: a string albedo used to load and fail later."""
    doc = json.loads(CERES.read_text())
    with pytest.raises(ValueError, match="not an airless configuration"):
        config_from_document({**doc["config"], key: bad})


def test_cli_refuses_an_empty_config_path(tmp_path, capsys):
    """An empty --config used to fall through to PRESETS[None] (KeyError)."""
    with pytest.raises(SystemExit) as exc:
        main(["show-config", "--config", ""])
    assert exc.value.code == 2
    assert "--config needs a file path" in capsys.readouterr().err


def test_standalone_replay_end_state_includes_the_skin(tmp_path):
    """The end-state verdict covers the skin as well as the cells: a corrupted
    retained skin fails the replay of a sample-free segment."""
    from tools.planet_one.global_check import main as gg
    run = AirlessRun(_short_ceres(), tmp_path / "run", backend="numpy", allow_dirty=True,
                     log=lambda s: None)
    run.run_transient(25)
    path = tmp_path / "run" / "retained" / "state.npz"
    with np.load(path) as f:
        z = dict(f)
    z["end_surface_anomaly"] = z["end_surface_anomaly"] + 1e-6
    np.savez(path, **z)
    assert gg([str(tmp_path / "run"), "--columns", "0"]) == 1
    report = json.loads((tmp_path / "run" / "gg.json").read_text())
    assert report["compared"] == ["end_state"] and not report["pass"]


@pytest.mark.parametrize("key", ["albedo", "cycle_s"])
def test_config_refuses_integers_too_large_for_a_float(key):
    """math.isfinite(10**400) raises OverflowError; it must surface as a
    configuration error."""
    doc = json.loads(CERES.read_text())
    with pytest.raises(ValueError, match="not an airless configuration"):
        config_from_document({**doc["config"], key: 10 ** 400})


def test_standalone_replay_with_a_different_sample_count_fails_with_a_report(tmp_path):
    """A replay that cannot reproduce the run's sample layout is a failed
    replay with a written report, not a traceback."""
    from tools.planet_one.global_check import main as gg
    run = AirlessRun(_short_ceres(), tmp_path / "run", backend="numpy", allow_dirty=True,
                     log=lambda s: None)
    run.run_transient(24)
    path = tmp_path / "run" / "retained" / "state.npz"
    with np.load(path) as f:
        z = dict(f)
    z["sample_every"] = np.asarray([4, 4], dtype=np.int64)       # the run sampled every 2
    np.savez(path, **z)
    assert gg([str(tmp_path / "run"), "--columns", "0"]) == 1
    report = json.loads((tmp_path / "run" / "gg.json").read_text())
    assert not report["pass"]
    assert "replay produced" in report["not_compared"]["surface_fields"]
