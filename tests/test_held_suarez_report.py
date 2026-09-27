"""Deterministic Held–Suarez report on synthetic fixtures (plan S5; CPU, CI).

The fixtures are tiny (8 Gaussian latitudes, 4 sigma levels) but have the
real file layout: a run directory (config.json, events.jsonl,
statistics.npz) and a Dinosaur-style reference series.npz with daily zonal
means. Every verdict of the vocabulary — PASS, FAIL, INCONCLUSIVE,
INCOMPLETE — is produced by a fixture built to produce it, and missing or
incomplete data never yields PASS.
"""
from __future__ import annotations

import json
import math
import pathlib

import numpy as np
import pytest

from tropoi.run.held_suarez.config import HeldSuarezConfig
from tropoi.run.held_suarez.report import (FAIL, INCOMPLETE, INCONCLUSIVE, PASS,
                                           build_report, render_markdown, worst,
                                           write_report)

NLAT, K, LMAX = 8, 4, 7
DAYS, SPIN, BLOCK = 1200, 200, 200
X, W = np.polynomial.legendre.leggauss(NLAT)
LAT_ASC = np.degrees(np.arcsin(X))
SIGMA = (np.arange(K) + 0.5) / K


def _cfg(days=DAYS) -> HeldSuarezConfig:
    return HeldSuarezConfig(experiment_id="hs-synthetic", l_max=LMAX, nlat=NLAT, nlon=16,
                            nlev=K, dt=1200.0, days=days, spinup_days=SPIN,
                            block_days=BLOCK, perturbation_lmax=4,
                            hyperdiffusion_reference_degree=(2 * LMAX) // 3)


def _ref_config(**over) -> dict:
    d = {"truncation": (2 * LMAX) // 3, "grid": f"gaussian {NLAT}x16", "levels": K, "p0_pa": 1e5, "kappa": 2 / 7, "cp": 1004.0,
         "omega": 7.292e-5, "radius_m_dinosaur": 6371220.0, "g": 9.8, "float64": True,
         "held_suarez": {"sigma_b": 0.7, "kf_per_day": 1.0, "ka_per_day": 0.025,
                         "ks_per_day": 0.25, "dTy_K": 60.0, "dThz_K": 10.0,
                         "Tmin_K": 200.0, "Tmax_K": 315.0},
         "hyperdiffusion": {"order": 4, "efold_days_at_truncation": 0.1},
         "days_total": DAYS, "days_spinup": SPIN, "block_days": BLOCK}
    d.update(over)
    return d


def _fields(lat, jet=30.0, jet_lat=45.0):
    """Synthetic climate on (lat, sigma): jets at +-jet_lat, eddy maxima."""
    L = lat[:, None]
    s = SIGMA[None, :]
    u = jet * np.exp(-((np.abs(L) - jet_lat) / 12.0) ** 2) * np.exp(-((s - 0.3) / 0.25) ** 2) - 2.0
    T = 300.0 - 40.0 * np.sin(np.radians(L)) ** 2 - 60.0 * (1.0 - s)
    TsTs = 30.0 * np.exp(-((np.abs(L) - 45.0) / 15.0) ** 2) * np.exp(-((s - 0.8) / 0.2) ** 2)
    vsTs = np.sign(L) * 15.0 * np.exp(-((np.abs(L) - 45.0) / 15.0) ** 2) * np.exp(-((s - 0.8) / 0.2) ** 2)
    return {"u": u, "v": 0.1 * u, "T": T, "TsTs": TsTs, "usus": 2 * TsTs, "vsTs": vsTs,
            "usvs": 0.5 * vsTs}


def _block_noise(b, amp):
    """Deterministic, sign-alternating block-to-block perturbation."""
    rng = np.random.default_rng(100 + b)
    return amp * rng.standard_normal((NLAT, K))


def make_run(tmp: pathlib.Path, *, days_done=DAYS, jet=30.0, jet_lat=45.0, noise=0.05,
             ps_drift=0.0, spectrum_tail=-3.0, aborted=False, resume_commit=None,
             name="run", ke_trend=0.0, t_offset=0.0, sh_jet_factor=1.0, flip_sh_vt=False,
             events_file=True, unverified_resume=False) -> pathlib.Path:
    rd = tmp / name
    rd.mkdir()
    cfg = _cfg()
    commit = "a" * 40
    (rd / "config.json").write_text(json.dumps(
        {"config": cfg.to_dict(), "config_sha256": cfg.sha256(),
         "code": {"git_commit": commit, "git_dirty": False}}))
    events = [{"event": "created", "day": 0}]
    events.append({"event": "resumed", "day": min(400, days_done), "checkpoint": "x.npz",
                   "verified_arrays": {} if unverified_resume else {"x_curr": "0" * 64},
                   "verified_meta_sha256": "1" * 64, "config_sha256": cfg.sha256(),
                   "git_commit": resume_commit or commit, "git_dirty": False})
    if aborted:
        events.append({"event": "aborted", "day": days_done, "error": "NaN"})
    if events_file:
        (rd / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    day = np.arange(days_done + 1, dtype=float)
    ps = 1e5 * (1.0 + ps_drift * day / DAYS)
    ke = 100.0 + 0.5 * np.sin(day / 7.0) + ke_trend * day / DAYS
    mT = 250.0 + 0.1 * np.cos(day / 11.0)
    lat_desc = LAT_ASC[::-1]
    base = _fields(lat_desc, jet, jet_lat)
    base["T"] = base["T"] + t_offset
    south = lat_desc[:, None] < 0
    base["u"] = np.where(south, (base["u"] + 2.0) * sh_jet_factor - 2.0, base["u"])
    if flip_sh_vt:
        base["vsTs"] = np.where(south, -base["vsTs"], base["vsTs"])
    nb = (DAYS - SPIN) // BLOCK
    counts = np.array([max(0, min(BLOCK, days_done - SPIN - b * BLOCK)) for b in range(nb)])
    blocks = {}
    for k, f in base.items():
        arr = np.stack([f + _block_noise(b, noise * (np.abs(f).max() + 1e-12))[::-1] for b in range(nb)])
        arr[counts == 0] = np.nan
        blocks[f"block_{k}"] = arr
    l = np.arange(LMAX + 1, dtype=float)
    spec = np.where(l > 0, np.maximum(l, 1.0) ** -3.0, 0.0)
    cut = (2 * LMAX) // 3
    spec[cut - 1:cut + 1] *= np.array([1.0, 10.0]) if spectrum_tail > 0 else 1.0
    blocks["block_ke_spectrum"] = np.stack([spec] * nb)
    np.savez(rd / "statistics.npz", counts=counts, lat=lat_desc, sigma=SIGMA,
             dsigma=np.full(K, 1.0 / K), gauss_weights=W[::-1] / 2,
             series_day=day, series_mean_ps=ps, series_ke=ke, series_mean_T=mT,
             series_mean_lnps=np.log(ps), series_max_abs_u=np.full_like(day, 40.0),
             series_t_min=np.full_like(day, 190.0), series_t_max=np.full_like(day, 310.0),
             series_comp_mode=np.zeros_like(day), **blocks)
    return rd


def make_reference(tmp: pathlib.Path, *, days=DAYS, noise=0.05) -> pathlib.Path:
    base = _fields(LAT_ASC)
    day = np.arange(1, days + 1)
    out = {}
    for k, f in base.items():
        a = np.empty((days, NLAT, K))
        for d in range(days):
            b = (d - SPIN) // BLOCK if d >= SPIN else 0
            wiggle = (1 if d % 2 else -1) * 0.01 * np.abs(f).max()     # zero mean per block
            a[d] = f + _block_noise(b, noise * (np.abs(f).max() + 1e-12)) + wiggle
        out[k] = a
    p = tmp / "reference_series.npz"
    np.savez(p, lat=LAT_ASC, sigma=SIGMA, day=day, mean_lnps=np.full(days, 11.5),
             mean_T=250.0 + 0.1 * np.cos(day / 11.0), ke_mass_weighted=100.0 + 0.5 * np.sin(day / 7.0),
             max_abs_u=np.full(days, 40.0), **out)
    return p


def _files(tmp, ref_cfg=None, b2=0.0):
    rc = tmp / "ref_config.json"
    rc.write_text(json.dumps(ref_cfg or _ref_config()))
    b2p = tmp / "ref_b2.json"
    b2p.write_text(json.dumps({"rel_change": b2}))
    return rc, b2p


def _evidence(tmp, **over):
    e = {"pytest": {"passed": 1100, "failed": 0, "skipped": 5, "errors": 0},
         "tests_diff_additions_only": True, "new_skips": [],
         "new_tests": {"t1": "PASS", "t2": "PASS"}, "smoke_resume_bit_identical": True,
         "solver_commit_pushed_ancestor": True, "solver_commit": "b" * 40}
    e.update(over)
    p = tmp / "evidence.json"
    p.write_text(json.dumps(e))
    return p


def _verdicts(rep):
    return {c["id"]: c["verdict"] for t in rep["tiers"] for c in t["criteria"]}


def _tiers(rep):
    return {t["tier"]: t["verdict"] for t in rep["tiers"]}


# ---------------------------------------------------------------------------

def test_worst_verdict_ordering():
    assert worst([PASS, INCOMPLETE]) == INCOMPLETE
    assert worst([PASS, INCOMPLETE, INCONCLUSIVE]) == INCONCLUSIVE
    assert worst([INCONCLUSIVE, FAIL, PASS]) == FAIL
    assert worst([PASS, PASS]) == PASS
    assert worst([]) == INCOMPLETE                      # nothing evaluated is never PASS


def test_all_pass_fixture(tmp_path):
    rd = make_run(tmp_path)
    ref = make_reference(tmp_path)
    rc, b2 = _files(tmp_path)
    rep = build_report(rd, reference=ref, reference_config=rc, reference_b2=b2,
                       tier_a_evidence=_evidence(tmp_path))
    v = _verdicts(rep)
    print("\n" + render_markdown(rep))
    assert _tiers(rep) == {"A": PASS, "B": PASS, "C": PASS}, v
    assert v["C6"] == "INFO"


def test_fail_fixture(tmp_path):
    rd = make_run(tmp_path, jet=45.0, ps_drift=2e-3, noise=0.01)
    ref = make_reference(tmp_path, noise=0.01)
    rc, b2 = _files(tmp_path)
    rep = build_report(rd, reference=ref, reference_config=rc, reference_b2=b2,
                       tier_a_evidence=_evidence(tmp_path, pytest={"passed": 10, "failed": 1,
                                                                   "skipped": 0}))
    v = _verdicts(rep)
    assert v["A1"] == FAIL and v["B2"] == FAIL and v["C1"] == FAIL and v["C2"] == FAIL
    assert _tiers(rep) == {"A": FAIL, "B": FAIL, "C": FAIL}


def test_inconclusive_when_difference_is_within_block_noise(tmp_path):
    # jet 15 % stronger (violates +-10 %) but the blocks are very noisy
    rd = make_run(tmp_path, jet=34.5, noise=0.6)
    ref = make_reference(tmp_path, noise=0.6)
    rc, b2 = _files(tmp_path)
    rep = build_report(rd, reference=ref, reference_config=rc, reference_b2=b2)
    c2 = next(c for c in rep["tiers"][2]["criteria"] if c["id"] == "C2")
    print(json.dumps(c2["values"]["NH"], indent=1))
    assert c2["values"]["NH"]["verdicts"][0] == INCONCLUSIVE
    assert _tiers(rep)["C"] in (INCONCLUSIVE,)


@pytest.mark.parametrize("case", ["missing", "mismatch", "reference_fails_b"])
def test_reference_problems_make_tier_c_inconclusive(tmp_path, case):
    rd = make_run(tmp_path)
    ref = make_reference(tmp_path)
    if case == "missing":
        rc, b2 = _files(tmp_path)
        rep = build_report(rd, reference=tmp_path / "nope.npz", reference_config=rc, reference_b2=b2)
        expect = "reference unavailable"
    elif case == "mismatch":
        rc, b2 = _files(tmp_path, ref_cfg=_ref_config(truncation=42, levels=20))
        rep = build_report(rd, reference=ref, reference_config=rc, reference_b2=b2)
        expect = "configuration mismatch: truncation (effective"
    else:
        rc, b2 = _files(tmp_path, b2=5e-3)                        # reference fails its B2
        rep = build_report(rd, reference=ref, reference_config=rc, reference_b2=b2)
        expect = "reference fails its own tier B"
    v = _verdicts(rep)
    for c in ("C1", "C2", "C3", "C4", "C5"):
        assert v[c] == INCONCLUSIVE
    detail = rep["tiers"][2]["criteria"][0]["detail"]
    assert expect in detail, detail
    assert _tiers(rep)["C"] == INCONCLUSIVE


def test_incomplete_run_is_never_pass(tmp_path):
    rd = make_run(tmp_path, days_done=600)
    ref = make_reference(tmp_path)
    rc, b2 = _files(tmp_path)
    rep = build_report(rd, reference=ref, reference_config=rc, reference_b2=b2)
    v = _verdicts(rep)
    for c in ("B1", "B2", "B3", "B4", "B5", "C1", "C2", "C3", "C4", "C5"):
        assert v[c] == INCOMPLETE, (c, v[c])
    for c in ("A1", "A2", "A3", "A4"):                            # no evidence given
        assert v[c] == INCOMPLETE
    assert _tiers(rep) == {"A": INCOMPLETE, "B": INCOMPLETE, "C": INCOMPLETE}


def test_incomplete_run_can_still_fail_b2_and_b1(tmp_path):
    rd = make_run(tmp_path, days_done=300, ps_drift=0.01, aborted=True)
    rep = build_report(rd)
    v = _verdicts(rep)
    assert v["B1"] == FAIL and v["B2"] == FAIL
    assert _tiers(rep)["C"] == INCONCLUSIVE                        # no reference given


def test_b4_pile_up_and_b5_commit_change_fail(tmp_path):
    rep = build_report(make_run(tmp_path, spectrum_tail=+1.0, resume_commit="c" * 40))
    v = _verdicts(rep)
    assert v["B4"] == FAIL and v["B5"] == FAIL


def test_reference_with_missing_days_is_inconclusive(tmp_path):
    rd = make_run(tmp_path)
    ref = make_reference(tmp_path, days=900)
    rc, b2 = _files(tmp_path)
    rep = build_report(rd, reference=ref, reference_config=rc, reference_b2=b2)
    assert _tiers(rep)["C"] == INCONCLUSIVE
    assert rep["reference_tier_b"]["verdict"] == FAIL


def test_report_is_byte_deterministic(tmp_path):
    rd = make_run(tmp_path)
    ref = make_reference(tmp_path)
    rc, b2 = _files(tmp_path)
    outs = []
    for i in range(2):
        rep = build_report(rd, reference=ref, reference_config=rc, reference_b2=b2,
                           tier_a_evidence=_evidence(tmp_path))
        j, m = write_report(rep, tmp_path / f"out{i}")
        outs.append((j.read_bytes(), m.read_bytes()))
    assert outs[0] == outs[1]
    assert b"PASS" in outs[0][1]


def test_production_configuration_against_the_real_reference_config():
    """The frozen reference list vs the production preset: exactly the
    effective-truncation findings (DEVLOG S4), nothing else."""
    from tropoi.run.held_suarez.config import production_config
    from tropoi.run.held_suarez.report import config_mismatches
    ref = json.loads((pathlib.Path(__file__).resolve().parents[1] / "docs" / "held_suarez"
                      / "REFERENCE_CONFIG.json").read_text())
    mm = config_mismatches(production_config(dt=900.0).to_dict(), ref)
    print("\n" + "\n".join(mm))
    assert mm == ["truncation (effective, retained degrees): ours 28 vs reference 42",
                  "hyperdiffusion reference degree = our effective truncation (smallest evolving "
                  "wave): ours 42 vs reference 28"]
    # the Dinosaur-matching alternative (l_max = 63, cut 42, del^8 at l = 42) matches
    alt = production_config(dt=900.0).with_(l_max=63, hyperdiffusion_reference_degree=42)
    assert config_mismatches(alt.to_dict(), ref) == []


@pytest.mark.parametrize("case,crit", [("ke_trend", "B3"), ("t_offset", "C3"),
                                       ("sh_jet", "C5"), ("flip_sh_vt", "C4")])
def test_each_remaining_criterion_can_fail(tmp_path, case, crit):
    kw = {"ke_trend": {"ke_trend": 40.0}, "t_offset": {"t_offset": 10.0},
          "sh_jet": {"sh_jet_factor": 0.6}, "flip_sh_vt": {"flip_sh_vt": True}}[case]
    rd = make_run(tmp_path, noise=0.01, **kw)
    ref = make_reference(tmp_path, noise=0.01)
    rc, b2 = _files(tmp_path)
    rep = build_report(rd, reference=ref, reference_config=rc, reference_b2=b2)
    v = _verdicts(rep)
    assert v[crit] == FAIL, (crit, v)


def test_missing_event_log_or_unverified_resume_is_never_pass(tmp_path):
    v = _verdicts(build_report(make_run(tmp_path, events_file=False)))
    assert v["B1"] == INCOMPLETE and v["B5"] == INCOMPLETE
    v = _verdicts(build_report(make_run(tmp_path, unverified_resume=True, name="r2")))
    assert v["B1"] == FAIL


def test_reference_without_b2_evidence_is_inconclusive(tmp_path):
    rd = make_run(tmp_path)
    ref = make_reference(tmp_path)
    rc, _ = _files(tmp_path)
    rep = build_report(rd, reference=ref, reference_config=rc, reference_b2=tmp_path / "none.json")
    assert rep["reference_tier_b"]["verdict"] == INCOMPLETE
    assert _tiers(rep)["C"] == INCONCLUSIVE
    assert "not established" in rep["tiers"][2]["criteria"][0]["detail"]
