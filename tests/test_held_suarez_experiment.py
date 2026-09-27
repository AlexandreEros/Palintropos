"""Held–Suarez experiment driver (plan S5).

CPU (CI): configuration hashing, the deterministic degree-1..8 vorticity
perturbation, atomic hash-verified checkpoints (tamper / schema / partial
file detection, pruning, backups).

GPU: the tier-A3 smoke — T21 L10, 2 days, SI, forced stop at day 1 and
resume — is bit-identical to the uninterrupted run (both leapfrog levels,
every accumulator); the online block statistics equal an offline
recomputation from the daily samples to 1e-12; a resume refuses a changed
configuration, a different commit and a corrupt checkpoint; the RK4 startup
runs exactly once per experiment.
"""
from __future__ import annotations

import json
import math
import pathlib

import numpy as np
import pytest

from tropoi.run.held_suarez.checkpoint import (CHECKPOINT_SCHEMA, CheckpointError,
                                               backup_files, list_checkpoints,
                                               prune_checkpoints, read_checkpoint,
                                               write_checkpoint)
from tropoi.run.held_suarez.config import (HeldSuarezConfig, development_config,
                                           production_config, smoke_config)
from tropoi.run.held_suarez.model import perturbation_coefficients


def _has_cuda():
    try:
        import cupy as cp
        return cp.is_available()
    except Exception:
        return False


gpu = pytest.mark.skipif(not _has_cuda(), reason="CUDA/CuPy not available")
CODE = {"git_commit": "test-commit", "git_dirty": False}


# ---------------------------------------------------------------------------
# configuration (CPU)
# ---------------------------------------------------------------------------

def test_config_hash_is_stable_and_sensitive():
    a, b = production_config(), production_config()
    assert a.sha256() == b.sha256()
    assert HeldSuarezConfig.from_dict(json.loads(a.canonical_json())).sha256() == a.sha256()
    for change in ({"dt": 900.0}, {"raw_alpha": 0.5}, {"seed": 1}, {"l_max": 41},
                   {"hyperdiffusion_efold_days": 0.2}, {"t_ref": 290.0}):
        assert a.with_(**change).sha256() != a.sha256(), change
    assert a.l_max == 42 and (a.nlat, a.nlon, a.nlev) == (64, 128, 20)
    assert a.days == 1200 and a.spinup_days == 200 and a.block_days == 200 and a.n_blocks == 5
    assert a.product_cut == 28
    with pytest.raises(ValueError):
        a.with_(dt=1000.0)                               # does not divide a day
    with pytest.raises(ValueError):
        development_config().with_(perturbation_lmax=15)  # above the T21 cut (14)


def test_perturbation_is_deterministic_confined_and_normalized():
    cfg = production_config()
    p1, s1 = perturbation_coefficients(cfg)
    p2, s2 = perturbation_coefficients(cfg)
    assert p1.tobytes() == p2.tobytes() and s1 == s2
    assert p1.shape == (cfg.nlev, cfg.l_max + 1, cfg.l_max + 1)
    nz = np.argwhere(p1 != 0)
    assert nz[:, 1].min() == 1 and nz[:, 1].max() == 8              # degrees 1..8
    assert np.all(nz[:, 2] <= nz[:, 1])                             # m <= l
    assert np.all(p1[:, :, 0].imag == 0)                            # m = 0 real
    for k in range(cfg.nlev):
        power = (np.abs(p1[k, :, 0]) ** 2).sum() + 2 * (np.abs(p1[k, :, 1:]) ** 2).sum()
        assert math.sqrt(power / (4 * math.pi)) == pytest.approx(1e-6, rel=1e-14)
    assert not np.array_equal(p1[0], p1[1])                         # levels independent
    other, _ = perturbation_coefficients(cfg.with_(seed=cfg.seed + 1))
    assert not np.array_equal(other, p1)
    print(f"\nperturbation seed {cfg.seed}: first coefficients "
          f"{p1[0, 1, 0].real:+.6e}, {p1[0, 1, 1]:+.6e}; RNG state {s1['state']['state']}")


# ---------------------------------------------------------------------------
# checkpoints (CPU)
# ---------------------------------------------------------------------------

def _arrays(seed=0):
    rng = np.random.default_rng(seed)
    return {"x_curr": rng.standard_normal((5, 4, 4)) + 1j * rng.standard_normal((5, 4, 4)),
            "stats_counts": np.arange(3, dtype=np.int64), "series_day": np.arange(4.0)}


def test_checkpoint_round_trip_verifies_every_array(tmp_path):
    arrays = _arrays()
    p = write_checkpoint(tmp_path, 72, arrays, {"day": 1.0, "note": "x"})
    assert p.name == "checkpoint-s000000072.npz"
    got, meta = read_checkpoint(p)
    assert meta["schema"] == CHECKPOINT_SCHEMA and meta["day"] == 1.0
    assert set(meta["arrays"]) == set(arrays)
    for k, a in arrays.items():
        assert got[k].tobytes() == a.tobytes() and got[k].dtype == a.dtype
    assert not list(tmp_path.glob("*.tmp"))                         # temp file renamed


def test_checkpoint_tampering_and_truncation_are_detected(tmp_path):
    p = write_checkpoint(tmp_path, 1, _arrays(), {"day": 0.0})
    raw = bytearray(p.read_bytes())
    # flip one byte inside the x_curr payload
    arrays, meta = read_checkpoint(p)
    needle = arrays["x_curr"].tobytes()[:16]
    i = bytes(raw).find(needle)
    assert i > 0
    raw[i + 3] ^= 0xFF
    bad = tmp_path / "checkpoint-s000000002.npz"
    bad.write_bytes(bytes(raw))
    with pytest.raises(CheckpointError, match="SHA-256|unreadable"):
        read_checkpoint(bad)
    trunc = tmp_path / "checkpoint-s000000003.npz"
    trunc.write_bytes(p.read_bytes()[: len(raw) // 2])
    with pytest.raises(CheckpointError):
        read_checkpoint(trunc)


def test_checkpoint_schema_is_enforced(tmp_path, monkeypatch):
    import tropoi.run.held_suarez.checkpoint as ck
    monkeypatch.setattr(ck, "CHECKPOINT_SCHEMA", "palintropos.held_suarez.checkpoint/0")
    p = ck.write_checkpoint(tmp_path, 5, _arrays(), {"day": 0.0})
    monkeypatch.undo()
    with pytest.raises(CheckpointError, match="schema"):
        read_checkpoint(p)


def test_listing_ignores_temporaries_pruning_and_backup(tmp_path):
    d = tmp_path / "ck"
    for s in (72, 144, 216, 288):
        write_checkpoint(d, s, _arrays(s), {"day": s / 72})
    (d / "checkpoint-s000000360.npz.tmp").write_bytes(b"partial")
    assert [s for s, _ in list_checkpoints(d)] == [72, 144, 216, 288]
    deleted = prune_checkpoints(d, 3)
    assert [p.name for p in deleted] == ["checkpoint-s000000072.npz"]
    bdir = tmp_path / "drive"
    extra = tmp_path / "series.json"
    extra.write_text("{}")
    for s, p in list_checkpoints(d):
        backup_files(p, [extra], bdir, keep=2)
    assert [s for s, _ in list_checkpoints(bdir / "checkpoints")] == [216, 288]
    assert (bdir / "series.json").read_text() == "{}"


# ---------------------------------------------------------------------------
# GPU: the driver
# ---------------------------------------------------------------------------

def _final(run_dir: pathlib.Path):
    _, path = list_checkpoints(run_dir / "checkpoints")[-1]
    return read_checkpoint(path)


@pytest.fixture(scope="module")
def uninterrupted(tmp_path_factory):
    if not _has_cuda():
        pytest.skip("CUDA/CuPy not available")
    from tropoi.run.held_suarez.experiment import HeldSuarezRun
    rd = tmp_path_factory.mktemp("hs") / "full"
    samples = {}
    run = HeldSuarezRun(smoke_config(), rd, code=CODE, log=lambda s: None,
                        sample_hook=lambda d, s: samples.__setitem__(d, s))
    res = run.run()
    return rd, res, samples, run


@gpu
def test_smoke_forced_stop_and_resume_is_bit_identical(uninterrupted, tmp_path):
    """Plan A3: T21 L10, 2 days, SI; stop at day 1, resume in a fresh driver
    (new model, new stepper), finish: final state bit-identical."""
    import hashlib
    from tropoi.run.held_suarez.experiment import HeldSuarezRun
    full_dir, res, _, _ = uninterrupted
    assert res["status"] == "completed" and res["day"] == 2.0
    rd = tmp_path / "split"
    first = HeldSuarezRun(smoke_config(), rd, code=CODE, log=lambda s: None)
    assert first.run(until_day=1)["status"] == "reached"
    # a crash after the day-1 checkpoint: 10 more steps are lost
    for _ in range(10):
        first.stepper.step()
    del first
    second = HeldSuarezRun(smoke_config(), rd, code=CODE, log=lambda s: None)
    assert second.step == 72 and second.stepper.x_prev is not None     # no RK4 redo
    assert second.run()["status"] == "completed"
    a_arr, a_meta = _final(full_dir)
    b_arr, b_meta = _final(rd)
    assert set(a_arr) == set(b_arr)
    for k in sorted(a_arr):
        assert a_arr[k].tobytes() == b_arr[k].tobytes(), k
    h = lambda arr: hashlib.sha256(arr["x_curr"].tobytes() + arr["x_prev"].tobytes()).hexdigest()
    print(f"\nuninterrupted final sha256 {h(a_arr)}\nresumed       final sha256 {h(b_arr)}\n"
          "forced stop/resume bit-identical: PASS")
    assert a_meta["stepper"] == b_meta["stepper"]
    ev = [e["event"] for e in second.events()]
    assert ev.count("created") == 1 and ev.count("resumed") == 1 and ev[-1] == "completed"
    resumed = next(e for e in second.events() if e["event"] == "resumed")
    assert set(resumed["verified_arrays"]) == set(b_arr)            # every array hash-checked
    assert resumed["checkpoint"] == "checkpoint-s000000072.npz"


@gpu
def test_online_statistics_equal_offline_recomputation(uninterrupted):
    from tropoi.run.held_suarez.statistics import SQUARE_KEYS, ZONAL_KEYS
    rd, _, samples, run = uninterrupted
    cfg = smoke_config()
    assert sorted(samples) == [0, 1, 2]
    bm = run.stats.block_means()
    worst = 0.0
    for b in range(cfg.n_blocks):
        days = [d for d in samples if run.stats.block_of(d) == b]
        assert days
        for k in ZONAL_KEYS:
            off = np.mean([samples[d][k] for d in days], axis=0)
            worst = max(worst, float(np.abs(bm[k][b] - off).max() / np.abs(off).max()))
        for k in SQUARE_KEYS:
            off = np.mean([samples[d][k] ** 2 for d in days], axis=0)
            worst = max(worst, float(np.abs(bm[f"{k}_sq"][b] - off).max() / np.abs(off).max()))
        off = np.mean([samples[d]["ke_spectrum"] for d in days], axis=0)
        worst = max(worst, float(np.abs(bm["ke_spectrum"][b] - off).max() / np.abs(off).max()))
    ser = run.stats.series_arrays()
    for k in ("mean_ps", "ke", "mean_T"):
        assert np.array_equal(ser[k], [samples[d][k] for d in sorted(samples)])
    print(f"\nonline vs offline block means: max relative difference {worst:.2e}")
    assert worst <= 1e-12
    # the export used by the report carries the same numbers
    with np.load(rd / "statistics.npz") as z:
        assert np.array_equal(z["block_u"], bm["u"]) and np.array_equal(z["counts"], run.stats.counts)
    # B2 quantity: the Gaussian-weighted mean of exp(ln p_s), day 0 = p0 exactly
    assert ser["mean_ps"][0] == pytest.approx(1e5, rel=1e-13)
    print(f"mean p_s: {ser['mean_ps']}; relative change {abs(ser['mean_ps'][-1] / ser['mean_ps'][0] - 1):.2e}")


@gpu
def test_resume_refuses_incompatible_configuration_commit_and_corruption(uninterrupted, tmp_path):
    import shutil
    from tropoi.run.held_suarez.experiment import HeldSuarezRun, RunError
    full_dir = uninterrupted[0]
    rd = tmp_path / "copy"
    shutil.copytree(full_dir, rd)
    with pytest.raises(RunError, match="config_sha256"):
        HeldSuarezRun(smoke_config().with_(dt=900.0), rd, code=CODE, log=lambda s: None)
    with pytest.raises(RunError, match="commit"):
        HeldSuarezRun(smoke_config(), rd, code={"git_commit": "other", "git_dirty": False},
                      log=lambda s: None)
    with pytest.raises(RunError, match="uncommitted"):
        HeldSuarezRun(smoke_config(), rd, code={"git_commit": "x", "git_dirty": True},
                      log=lambda s: None)
    _, newest = list_checkpoints(rd / "checkpoints")[-1]
    raw = bytearray(newest.read_bytes())
    raw[len(raw) // 2] ^= 0xFF
    newest.write_bytes(bytes(raw))
    with pytest.raises(CheckpointError):
        HeldSuarezRun(smoke_config(), rd, code=CODE, log=lambda s: None)


@gpu
def test_initial_state_seeds_vorticity_only_inside_the_cut(uninterrupted):
    import cupy as cp
    from tropoi.run.held_suarez.model import initial_state
    cfg = smoke_config()
    x0, rng_state = initial_state(cfg)
    K = cfg.nlev
    assert float(cp.abs(x0[K:2 * K]).max()) == 0.0                   # divergence not seeded
    assert float(cp.abs(x0[0:K, 9:, :]).max()) == 0.0                 # zeta only l <= 8
    assert float(cp.abs(x0[0:K, 0, :]).max()) == 0.0                  # zero global circulation
    run = uninterrupted[3]
    zeta = cp.stack([run.model.sh.inv_transform(x0[k]).real for k in range(K)])
    w = run.model.grid.solid_angle_weights
    rms = cp.sqrt((zeta ** 2 * w[None]).sum(axis=1) / (4 * math.pi))
    assert float(cp.abs(rms - 1e-6).max()) < 1e-6 * 1e-12


@gpu
def test_mid_day_stop_resume_is_bit_identical_and_startup_runs_once(uninterrupted, tmp_path,
                                                                       monkeypatch):
    """A segment stopped mid-day (step 100 of 144, statistics state between two
    daily samples) and resumed in a fresh driver finishes bit-identically; the
    RK4 startup is executed exactly once over both segments."""
    from tropoi.run.held_suarez.experiment import HeldSuarezRun
    from tropoi.temporal.semi_implicit import SemiImplicitLeapfrogStepper
    calls = []
    orig = SemiImplicitLeapfrogStepper._startup

    def counting(self):
        calls.append(self.step_count)
        return orig(self)

    monkeypatch.setattr(SemiImplicitLeapfrogStepper, "_startup", counting)
    rd = tmp_path / "midday"
    a = HeldSuarezRun(smoke_config(), rd, code=CODE, log=lambda s: None)
    assert a.run(max_steps=100)["status"] == "step_limit" and a.step == 100
    del a
    b = HeldSuarezRun(smoke_config(), rd, code=CODE, log=lambda s: None)
    assert b.step == 100
    assert b.run()["status"] == "completed"
    assert calls == [0]                                          # one startup, at step 0
    full, _ = _final(uninterrupted[0])
    got, _ = _final(rd)
    for k in sorted(full):
        assert full[k].tobytes() == got[k].tobytes(), k
    print("\nmid-day (step 100) stop/resume bit-identical: PASS; RK4 startups executed:", len(calls))


@gpu
def test_refuses_to_start_over_orphaned_checkpoints_or_foreign_backup(uninterrupted, tmp_path):
    import shutil
    from tropoi.run.held_suarez.experiment import HeldSuarezRun, RunError
    rd = tmp_path / "orphan"
    shutil.copytree(uninterrupted[0], rd)
    (rd / "config.json").unlink()
    with pytest.raises(RunError, match="config.json is missing"):
        HeldSuarezRun(smoke_config(), rd, code=CODE, log=lambda s: None)
    foreign = tmp_path / "drive"
    foreign.mkdir()
    (foreign / "config.json").write_text(json.dumps({"config_sha256": "0" * 64}))
    with pytest.raises(RunError, match="another configuration"):
        HeldSuarezRun(smoke_config(), tmp_path / "new", code=CODE, backup_dir=foreign,
                      log=lambda s: None)


@gpu
def test_backup_failure_is_logged_and_the_run_continues(tmp_path, monkeypatch):
    """A dropped Drive mount (OSError while backing up) must not stop the run
    or count as an abort (tier B1); state snapshots reach the backup when it
    works."""
    import tropoi.run.held_suarez.experiment as ex
    good = tmp_path / "drive_ok"
    run = ex.HeldSuarezRun(smoke_config(), tmp_path / "ok", code=CODE, backup_dir=good,
                           log=lambda s: None)
    run.run(until_day=1, backup_every_days=1)
    assert (good / "states" / "state-d00001.npy").exists()
    assert list((good / "checkpoints").glob("checkpoint-s*.npz"))

    def broken(*a, **k):
        raise OSError(107, "Transport endpoint is not connected")

    monkeypatch.setattr(ex, "backup_files", broken)
    run2 = ex.HeldSuarezRun(smoke_config(), tmp_path / "bad", code=CODE,
                            backup_dir=tmp_path / "drive_bad", log=lambda s: None)
    assert run2.run(backup_every_days=1)["status"] == "completed"
    events = [e["event"] for e in run2.events()]
    assert "backup_failed" in events and "interrupted" not in events and "aborted" not in events
