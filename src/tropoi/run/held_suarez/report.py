"""Deterministic Held–Suarez tier A/B/C report (CPU-only).

Implements plan §1–§2 (docs/superpowers/plans/2026-09-27-held-suarez-24h-
plan.md, revision 3) with the verdict vocabulary

    PASS          every criterion holds on the frozen experiment
    FAIL          a criterion is violated beyond sampling uncertainty
    INCONCLUSIVE  violated but within 2 sigma of block sampling noise; or the
                  reference is unavailable, unsuitable or configuration-mismatched
    INCOMPLETE    run shorter than the protocol / stage not reached; never PASS

A tier takes the worst per-criterion verdict, FAIL > INCONCLUSIVE >
INCOMPLETE > PASS. The same inputs always give byte-identical REPORT.md and
report.json (no clock, no environment, fixed number formats, sorted keys).

Interpretations fixed here (the plan's wording, made computable):

* B2  max over every daily sample of |<p_s>(d) - <p_s>(0)| / <p_s>(0), where
      <p_s> is the Gaussian-weighted global mean of p_s = exp(ln p_s). A
      partial run can FAIL (a violation cannot be undone) but not PASS.
* B3  least-squares line through the five block means against block-centre
      day; "trend" = |slope| x (n_blocks x block_days), the fitted change
      across the averaging window. KE <= 10 % of the mean of the block means;
      mass-weighted T <= 0.5 K.
* B4  L = the run's effective truncation (the product cut, the highest
      degree the core evolves); fit log KE vs log l over floor(2L/3) < l <= L
      (negative slope) and KE(L) < KE(floor(2L/3)).
* C   both cores' daily zonal means are averaged per 200-day block over days
      spinup+1..days. Scalar metrics (jet magnitude/latitude/level, peak
      magnitudes/latitudes) are computed per block for both cores;
      sigma_block = sqrt((s_ours^2 + s_ref^2) / 2) (ddof = 1); a violated
      criterion with |difference| <= 2 sigma_block sqrt(2/5) is INCONCLUSIVE.
      Field metrics (pattern correlation, mass-weighted RMSE) use the per-point
      pooled block sigma: the sampling RMS of a difference of two 5-block
      means is e = sqrt(<2 sigma^2 / 5>_w); a violated field criterion with
      RMSE(difference) <= 2 e is INCONCLUSIVE. Hemispheric quantities (C2, C4
      peaks) are checked per hemisphere. Latitudes are compared on the
      reference's Gaussian latitudes (ours interpolated linearly in latitude
      if the grids differ). Mass weight = Gaussian weight x dsigma.
* C5  per block, NH - SH jet magnitude and |latitude| of OUR core; mean
      difference d, sigma = std of the five block differences; FAIL only if
      |d| > 10 % of the mean AND |d| > 3 sigma.
* C6  informational only (no verdict weight).
"""
from __future__ import annotations

import json
import math
import re
import pathlib
from typing import Any, Optional

import numpy as np

__all__ = ["PASS", "FAIL", "INCONCLUSIVE", "INCOMPLETE", "worst", "build_report",
           "write_report", "reference_summary"]

PASS, FAIL, INCONCLUSIVE, INCOMPLETE = "PASS", "FAIL", "INCONCLUSIVE", "INCOMPLETE"
INFO = "INFO"
_RANK = {PASS: 0, INCOMPLETE: 1, INCONCLUSIVE: 2, FAIL: 3}

THRESHOLDS = {
    "B2_rel_ps": 1e-3, "B3_ke_frac": 0.10, "B3_T_K": 0.5,
    "C1_corr": 0.98, "C1_rmse_ms": 2.0,
    "C2_mag_frac": 0.10, "C2_lat_deg": 4.0, "C2_level": 1,
    "C3_rmse_K": 1.5,
    "C4_mag_frac": 0.20, "C4_lat_deg": 5.0, "C4_corr": 0.9,
    "C5_frac": 0.10, "C5_sigmas": 3.0,
    "noise_sigmas": 2.0,
}
REF_ZONAL = {"u": "u", "T": "T", "TsTs": "TsTs", "vsTs": "vsTs"}


def worst(verdicts) -> str:
    vs = [v for v in verdicts if v in _RANK]
    return max(vs, key=_RANK.__getitem__) if vs else INCOMPLETE


def _crit(cid: str, verdict: str, text: str, **values) -> dict:
    return {"id": cid, "verdict": verdict, "detail": text,
            "values": {k: _clean(v) for k, v in values.items()}}


def _clean(v):
    if isinstance(v, (np.floating, float)):
        v = float(v)
        return None if not math.isfinite(v) else v
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, np.ndarray):
        return [_clean(x) for x in v.tolist()]
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    if isinstance(v, dict):
        return {k: _clean(x) for k, x in v.items()}
    return v


def _tier(name: str, criteria: list[dict]) -> dict:
    return {"tier": name, "verdict": worst(c["verdict"] for c in criteria
                                           if c["verdict"] != INFO),
            "criteria": criteria}


# ---------------------------------------------------------------------------
# tier A (implementation evidence, supplied as JSON)
# ---------------------------------------------------------------------------

def tier_a(evidence: Optional[dict]) -> dict:
    e = evidence or {}
    out = []
    py = e.get("pytest")
    if py is None or "tests_diff_additions_only" not in e:
        out.append(_crit("A1", INCOMPLETE, "no full-suite result / tests diff evidence"))
    else:
        bad = int(py.get("failed", 0)) + int(py.get("errors", 0))
        skips_ok = all(s.get("reason") for s in e.get("new_skips", []))
        ok = bad == 0 and bool(e["tests_diff_additions_only"]) and skips_ok
        out.append(_crit("A1", PASS if ok else FAIL,
                         "pre-existing tests pass, tests/ diff is additions only, new skips have reasons",
                         passed=py.get("passed"), failed=py.get("failed"),
                         skipped=py.get("skipped"), errors=py.get("errors", 0),
                         additions_only=e["tests_diff_additions_only"],
                         new_skips=e.get("new_skips", [])))
    nt = e.get("new_tests")
    if not nt:
        out.append(_crit("A2", INCOMPLETE, "no new-test results"))
    else:
        failed = sorted(k for k, v in nt.items() if v != PASS)
        out.append(_crit("A2", FAIL if failed else PASS, "every new S1-S6 test passes",
                         n_tests=len(nt), failed=failed))
    a3 = e.get("smoke_resume_bit_identical")
    out.append(_crit("A3", INCOMPLETE if a3 is None else (PASS if a3 else FAIL),
                     "T21 L10 2-day SI smoke, forced stop + resume at day 1, bit-identical",
                     bit_identical=a3))
    pushed = e.get("solver_commit_pushed_ancestor")
    out.append(_crit("A4", INCOMPLETE if pushed is None else (PASS if pushed else FAIL),
                     "branch pushed; notebook SOLVER_COMMIT is a pushed ancestor of HEAD",
                     solver_commit=e.get("solver_commit"), pushed_ancestor=pushed))
    return _tier("A", out)


# ---------------------------------------------------------------------------
# run loading
# ---------------------------------------------------------------------------

def load_run(run_dir: pathlib.Path) -> dict:
    run_dir = pathlib.Path(run_dir)
    doc = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    events = []
    ev = run_dir / "events.jsonl"
    if ev.exists():
        events = [json.loads(x) for x in ev.read_text(encoding="utf-8").splitlines() if x]
    with np.load(run_dir / "statistics.npz", allow_pickle=False) as z:
        stats = {k: np.array(z[k]) for k in z.files}
    return {"config": doc["config"], "config_sha256": doc["config_sha256"],
            "code": doc.get("code", {}), "events": events, "stats": stats}


def _effective_truncation(cfg: dict) -> int:
    return (2 * int(cfg["l_max"])) // 3


# ---------------------------------------------------------------------------
# tier B
# ---------------------------------------------------------------------------

def _lsq_slope(x, y) -> float:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    xm = x - x.mean()
    return float((xm * (y - y.mean())).sum() / (xm * xm).sum())


def _series_blocks(day, values, spinup, block_days, n_blocks) -> Optional[np.ndarray]:
    out = []
    for b in range(n_blocks):
        lo, hi = spinup + b * block_days + 1, spinup + (b + 1) * block_days
        sel = (day >= lo) & (day <= hi)
        if int(sel.sum()) != block_days:
            return None
        out.append(float(np.mean(values[sel])))
    return np.asarray(out)


def tier_b(run: dict) -> dict:
    cfg, st, events = run["config"], run["stats"], run["events"]
    days_total = int(cfg["days"])
    day = st["series_day"]
    done = int(day.max()) if day.size else 0
    complete = done >= days_total
    has_log = bool(events)
    out = []
    aborted = [e for e in events if e["event"] == "aborted"]
    resumes = [e for e in events if e["event"] == "resumed"]
    unverified = [e for e in resumes if not e.get("verified_arrays")
                  or not e.get("verified_meta_sha256", True)]
    finite = bool(np.all(np.isfinite(st["series_ke"]))) and bool(np.all(np.isfinite(st["series_mean_ps"])))
    if aborted or not finite or unverified:
        v = FAIL
    else:
        v = PASS if (complete and has_log) else INCOMPLETE
    out.append(_crit("B1", v, f"reaches day {days_total} without abort/NaN; every resume hash-verified",
                     days_completed=done, days_total=days_total, aborted=len(aborted),
                     resumes=len(resumes), unverified_resumes=len(unverified)))
    ps = st["series_mean_ps"]
    rel = float(np.max(np.abs(ps - ps[0])) / ps[0]) if ps.size else float("nan")
    rel_end = float(abs(ps[-1] - ps[0]) / ps[0]) if ps.size else float("nan")
    if not math.isfinite(rel):
        v = INCOMPLETE if not ps.size else FAIL
    elif rel > THRESHOLDS["B2_rel_ps"]:
        v = FAIL
    else:
        v = PASS if complete else INCOMPLETE
    out.append(_crit("B2", v, "max_d |<p_s>(d) - <p_s>(0)| / <p_s>(0) <= 1e-3 (Gaussian-weighted mean of exp(ln p_s))",
                     max_rel_change=rel, final_rel_change=rel_end, threshold=THRESHOLDS["B2_rel_ps"]))
    spin, bd = int(cfg["spinup_days"]), int(cfg["block_days"])
    nb = max(0, (days_total - spin) // bd)
    ke_b = _series_blocks(day, st["series_ke"], spin, bd, nb) if nb else None
    t_b = _series_blocks(day, st["series_mean_T"], spin, bd, nb) if nb else None
    if ke_b is None or t_b is None or nb < 2:
        out.append(_crit("B3", INCOMPLETE, "stationarity needs every block complete",
                         n_blocks=nb))
    else:
        centres = spin + bd * (np.arange(nb) + 0.5)
        span = nb * bd
        ke_trend = abs(_lsq_slope(centres, ke_b)) * span
        t_trend = abs(_lsq_slope(centres, t_b)) * span
        ok = ke_trend <= THRESHOLDS["B3_ke_frac"] * float(ke_b.mean()) and \
            t_trend <= THRESHOLDS["B3_T_K"]
        out.append(_crit("B3", PASS if ok else FAIL,
                         "linear trend across block means: KE <= 10 % of mean, <T> <= 0.5 K",
                         ke_block_means=ke_b, ke_trend=ke_trend,
                         ke_trend_frac=ke_trend / float(ke_b.mean()),
                         T_block_means=t_b, T_trend_K=t_trend))
    counts = st.get("counts", np.zeros(0))
    L = _effective_truncation(cfg)
    l0 = (2 * L) // 3
    if not counts.size or np.any(counts != bd) or not complete:
        out.append(_crit("B4", INCOMPLETE, "time-mean spectrum needs every block complete",
                         L=L))
    else:
        spec = (st["block_ke_spectrum"] * counts[:, None]).sum(axis=0) / counts.sum()
        ls = np.arange(l0 + 1, L + 1)
        e = spec[ls]
        if np.any(e <= 0) or spec[l0] <= 0:
            out.append(_crit("B4", INCONCLUSIVE, "non-positive KE in the fit range", L=L))
        else:
            slope = _lsq_slope(np.log(ls), np.log(e))
            ok = slope < 0 and spec[L] < spec[l0]
            out.append(_crit("B4", PASS if ok else FAIL,
                             f"log-log KE slope over ({l0}, {L}] < 0 and KE({L}) < KE({l0}); "
                             "L = effective (product-cut) truncation",
                             L=L, l0=l0, slope=slope, ke_L=spec[L], ke_l0=spec[l0]))
    shas = {e.get("config_sha256") for e in resumes} | {run["config_sha256"]}
    commits = {e.get("git_commit") for e in resumes} | {run["code"].get("git_commit")}
    dirty = bool(run["code"].get("git_dirty")) or any(e.get("git_dirty") for e in resumes)
    if len(shas) > 1 or len(commits) > 1 or None in commits or dirty:
        v = FAIL
    else:
        v = PASS if (complete and has_log) else INCOMPLETE
    out.append(_crit("B5", v, "one configuration hash and one clean commit across every resume",
                     config_sha256=sorted(s for s in shas if s),
                     git_commits=sorted(c for c in commits if c), git_dirty=dirty,
                     event_log=has_log))
    return _tier("B", out)


# ---------------------------------------------------------------------------
# tier C
# ---------------------------------------------------------------------------

def reference_summary(series: dict, spinup: int, block_days: int, n_blocks: int
                      ) -> Optional[dict]:
    """Block means of the reference's daily zonal means, latitude ascending."""
    day = np.asarray(series["day"])
    lat = np.asarray(series["lat"], float)
    order = np.argsort(lat)
    out = {"lat": lat[order], "sigma": np.asarray(series["sigma"], float)}
    for key in REF_ZONAL.values():
        a = np.asarray(series[key], float)[:, order, :]
        blocks = []
        for b in range(n_blocks):
            lo, hi = spinup + b * block_days + 1, spinup + (b + 1) * block_days
            sel = (day >= lo) & (day <= hi)
            if int(sel.sum()) != block_days:
                return None
            blocks.append(a[sel].mean(axis=0))
        out[key] = np.stack(blocks)
    return out


def _ours_summary(run: dict, lat_ref: np.ndarray) -> dict:
    st = run["stats"]
    lat = np.asarray(st["lat"], float)
    order = np.argsort(lat)
    lat = lat[order]
    out = {"lat": lat_ref, "sigma": np.asarray(st["sigma"], float)}
    for key in REF_ZONAL:
        a = np.asarray(st[f"block_{key}"], float)[:, order, :]
        if lat.shape != lat_ref.shape or not np.allclose(lat, lat_ref, atol=1e-6):
            a = np.stack([np.stack([np.interp(lat_ref, lat, a[b, :, k])
                                    for k in range(a.shape[2])], axis=1)
                          for b in range(a.shape[0])])
        out[key] = a
    return out


def _gauss_weights(lat_deg: np.ndarray) -> np.ndarray:
    x, w = np.polynomial.legendre.leggauss(lat_deg.size)
    lat_g = np.degrees(np.arcsin(x))
    if np.allclose(np.sort(lat_g), np.sort(lat_deg), atol=1e-6):
        return w[np.argsort(lat_g)] / 2.0 if np.all(np.diff(lat_deg) > 0) else w / 2.0
    c = np.cos(np.radians(lat_deg))
    return c / c.sum()


def _wstats(a, b, w):
    wm = lambda f: float((w * f).sum() / w.sum())
    rmse = math.sqrt(wm((a - b) ** 2))
    am, bm = a - wm(a), b - wm(b)
    corr = wm(am * bm) / math.sqrt(wm(am * am) * wm(bm * bm))
    return corr, rmse


def _jet(u, lat, hemi):
    """(magnitude, |lat|, level index) of the max of u over one hemisphere."""
    sel = lat > 0 if hemi == "NH" else lat < 0
    sub = u[sel]
    i, k = np.unravel_index(np.argmax(sub), sub.shape)
    return float(sub[i, k]), float(abs(lat[sel][i])), int(k)


def _peak(f, lat, hemi, sign=1.0):
    sel = lat > 0 if hemi == "NH" else lat < 0
    sub = sign * f[sel]
    i, k = np.unravel_index(np.argmax(sub), sub.shape)
    return float(sub[i, k]), float(abs(lat[sel][i]))


def _scalar_verdict(ok: bool, diff: float, per_block_ours, per_block_ref) -> tuple[str, float]:
    s1 = float(np.std(per_block_ours, ddof=1))
    s2 = float(np.std(per_block_ref, ddof=1))
    sigma = math.sqrt((s1 * s1 + s2 * s2) / 2.0)
    noise = THRESHOLDS["noise_sigmas"] * sigma * math.sqrt(2.0 / len(per_block_ours))
    if ok:
        return PASS, noise
    return (INCONCLUSIVE if abs(diff) <= noise else FAIL), noise


def _field_verdict(ok: bool, ours_b, ref_b, w) -> tuple[str, float]:
    n = ours_b.shape[0]
    var = (np.var(ours_b, axis=0, ddof=1) + np.var(ref_b, axis=0, ddof=1)) / 2.0
    e = math.sqrt(float((w * (2.0 * var / n)).sum() / w.sum()))
    if ok:
        return PASS, e
    d = ours_b.mean(0) - ref_b.mean(0)
    rmse = math.sqrt(float((w * d * d).sum() / w.sum()))
    return (INCONCLUSIVE if rmse <= THRESHOLDS["noise_sigmas"] * e else FAIL), e


def config_mismatches(cfg: dict, ref_cfg: dict) -> list[str]:
    """The plan's 'what must match Dinosaur' list, checked field by field."""
    mm = []
    hs = cfg["forcing"]
    rhs = ref_cfg["held_suarez"]
    day = 86400.0

    def chk(name, ours, ref, rtol=1e-12):
        if ours is None or ref is None or not math.isclose(float(ours), float(ref), rel_tol=rtol, abs_tol=0.0):
            mm.append(f"{name}: ours {ours} vs reference {ref}")

    chk("truncation (effective, retained degrees)", _effective_truncation(cfg), ref_cfg["truncation"])
    grid = str(ref_cfg.get("grid", ""))
    m = re.search(r"(\d+)x(\d+)", grid)
    if m:
        chk("diagnostic grid nlat", cfg["nlat"], int(m.group(1)))
        chk("diagnostic grid nlon", cfg["nlon"], int(m.group(2)))
    else:
        mm.append(f"reference grid not stated ({grid!r})")
    chk("levels", cfg["nlev"], ref_cfg["levels"])
    chk("p0", cfg["forcing"]["p0"], ref_cfg["p0_pa"])
    chk("kappa", cfg["kappa"], ref_cfg["kappa"])
    chk("cp", cfg["cp_dry"], ref_cfg["cp"])
    chk("omega", cfg["omega"], ref_cfg["omega"])
    chk("radius (0.01 % tolerance)", cfg["radius"], ref_cfg.get("radius_m_dinosaur", ref_cfg.get("radius_m_palintropos")), rtol=1e-4)
    chk("g", cfg["gravity"], ref_cfg["g"])
    chk("sigma_b", hs["sigma_b"], rhs["sigma_b"])
    chk("k_f", hs["k_f"] * day, rhs["kf_per_day"])
    chk("k_a", hs["k_a"] * day, rhs["ka_per_day"])
    chk("k_s", hs["k_s"] * day, rhs["ks_per_day"])
    chk("dT_y", hs["delta_t_y"], rhs["dTy_K"])
    chk("dtheta_z", hs["delta_theta_z"], rhs["dThz_K"])
    chk("T_min", hs["t_min"], rhs["Tmin_K"])
    chk("T_max", hs["t_max"], rhs["Tmax_K"])
    chk("hyperdiffusion order", cfg["hyperdiffusion_order"], ref_cfg["hyperdiffusion"]["order"])
    chk("hyperdiffusion e-folding days", cfg["hyperdiffusion_efold_days"],
        ref_cfg["hyperdiffusion"]["efold_days_at_truncation"])
    ref_deg = cfg["hyperdiffusion_reference_degree"] or cfg["l_max"]
    chk("hyperdiffusion reference degree = reference truncation", ref_deg, ref_cfg["truncation"])
    chk("hyperdiffusion reference degree = our effective truncation (smallest evolving wave)",
        ref_deg, _effective_truncation(cfg))
    chk("days", cfg["days"], ref_cfg["days_total"])
    chk("spinup days", cfg["spinup_days"], ref_cfg["days_spinup"])
    chk("block days", cfg["block_days"], ref_cfg["block_days"])
    if not ref_cfg.get("float64", False):
        mm.append("reference not float64")
    return mm


def tier_c(run: dict, ref_series: Optional[dict], ref_cfg: Optional[dict],
           ref_b: Optional[dict] = None) -> dict:
    cfg = run["config"]
    ids = ("C1", "C2", "C3", "C4", "C5")
    days_total = int(cfg["days"])
    done = int(run["stats"]["series_day"].max()) if run["stats"]["series_day"].size else 0
    blockers = []
    if ref_series is None or ref_cfg is None:
        blockers.append("reference unavailable")
    else:
        blockers += [f"configuration mismatch: {m}" for m in config_mismatches(cfg, ref_cfg)]
        if ref_b is not None and ref_b.get("verdict") == FAIL:
            blockers.append(f"reference fails its own tier B: {ref_b.get('failed')}")
        elif ref_b is not None and ref_b.get("verdict") != PASS:
            blockers.append(f"reference tier B not established: {ref_b.get('notes')}")
    if blockers:
        crit = [_crit(i, INCONCLUSIVE, "; ".join(blockers)) for i in ids]
        crit.append(_crit("C6", INFO, "not evaluated"))
        return _tier("C", crit)
    if done < days_total:
        crit = [_crit(i, INCOMPLETE, f"run at day {done} of {days_total}") for i in ids]
        crit.append(_crit("C6", INFO, "not evaluated"))
        return _tier("C", crit)
    spin, bd = int(cfg["spinup_days"]), int(cfg["block_days"])
    nb = (days_total - spin) // bd
    R = reference_summary(ref_series, spin, bd, nb)
    if R is None:
        crit = [_crit(i, INCONCLUSIVE, "reference series does not cover every block") for i in ids]
        crit.append(_crit("C6", INFO, "not evaluated"))
        return _tier("C", crit)
    if R["sigma"].shape != run["stats"]["sigma"].shape or \
            not np.allclose(R["sigma"], run["stats"]["sigma"], atol=1e-9):
        crit = [_crit(i, INCONCLUSIVE, "sigma levels differ from the reference") for i in ids]
        crit.append(_crit("C6", INFO, "not evaluated"))
        return _tier("C", crit)
    O = _ours_summary(run, R["lat"])
    lat = R["lat"]
    w = _gauss_weights(lat)[:, None] * np.asarray(run["stats"]["dsigma"], float)[None, :]
    out = []
    # C1
    corr, rmse = _wstats(O["u"].mean(0), R["u"].mean(0), w)
    v_c, e = _field_verdict(corr >= THRESHOLDS["C1_corr"], O["u"], R["u"], w)
    v_r, _ = _field_verdict(rmse <= THRESHOLDS["C1_rmse_ms"], O["u"], R["u"], w)
    out.append(_crit("C1", worst([v_c, v_r]), "zonal-mean u: pattern corr >= 0.98, mass-weighted RMSE <= 2.0 m/s",
                     corr=corr, rmse=rmse, sampling_rms=e, corr_verdict=v_c, rmse_verdict=v_r))
    # C2
    subs, vals = [], {}
    for hemi in ("NH", "SH"):
        jo = [_jet(O["u"][b], lat, hemi) for b in range(nb)]
        jr = [_jet(R["u"][b], lat, hemi) for b in range(nb)]
        mo, lo_, ko = _jet(O["u"].mean(0), lat, hemi)
        mr, lr, kr = _jet(R["u"].mean(0), lat, hemi)
        v1, n1 = _scalar_verdict(abs(mo - mr) <= THRESHOLDS["C2_mag_frac"] * abs(mr), mo - mr,
                                 [j[0] for j in jo], [j[0] for j in jr])
        v2, n2 = _scalar_verdict(abs(lo_ - lr) <= THRESHOLDS["C2_lat_deg"], lo_ - lr,
                                 [j[1] for j in jo], [j[1] for j in jr])
        v3, n3 = _scalar_verdict(abs(ko - kr) <= THRESHOLDS["C2_level"], ko - kr,
                                 [j[2] for j in jo], [j[2] for j in jr])
        subs += [v1, v2, v3]
        vals[hemi] = {"ours": [mo, lo_, ko], "reference": [mr, lr, kr],
                      "verdicts": [v1, v2, v3], "noise": [n1, n2, n3]}
    out.append(_crit("C2", worst(subs), "jet max: magnitude +-10 %, latitude +-4 deg, level +-1 (per hemisphere)",
                     **vals))
    # C3
    _, rmse_t = _wstats(O["T"].mean(0), R["T"].mean(0), w)
    v, e = _field_verdict(rmse_t <= THRESHOLDS["C3_rmse_K"], O["T"], R["T"], w)
    out.append(_crit("C3", v, "zonal-mean T mass-weighted RMSE <= 1.5 K", rmse=rmse_t, sampling_rms=e))
    # C4
    subs, vals = [], {}
    for key in ("TsTs", "vsTs"):
        corr_k, _ = _wstats(O[key].mean(0), R[key].mean(0), w)
        vc, _ = _field_verdict(corr_k >= THRESHOLDS["C4_corr"], O[key], R[key], w)
        subs.append(vc)
        vals[key] = {"corr": corr_k, "corr_verdict": vc}
        for hemi in ("NH", "SH"):
            sign = -1.0 if (key == "vsTs" and hemi == "SH") else 1.0
            po = [_peak(O[key][b], lat, hemi, sign) for b in range(nb)]
            pr = [_peak(R[key][b], lat, hemi, sign) for b in range(nb)]
            mo, lo_ = _peak(O[key].mean(0), lat, hemi, sign)
            mr, lr = _peak(R[key].mean(0), lat, hemi, sign)
            v1, _ = _scalar_verdict(abs(mo - mr) <= THRESHOLDS["C4_mag_frac"] * abs(mr), mo - mr,
                                    [p[0] for p in po], [p[0] for p in pr])
            v2, _ = _scalar_verdict(abs(lo_ - lr) <= THRESHOLDS["C4_lat_deg"], lo_ - lr,
                                    [p[1] for p in po], [p[1] for p in pr])
            subs += [v1, v2]
            vals[key][hemi] = {"ours": [mo, lo_], "reference": [mr, lr], "verdicts": [v1, v2]}
    out.append(_crit("C4", worst(subs), "[T'^2], [v'T']: peak +-20 %, peak lat +-5 deg, pattern corr >= 0.9",
                     **vals))
    # C5 (our core's hemispheric symmetry)
    dm, dl, mags, lats = [], [], [], []
    for b in range(nb):
        mn, ln_, _ = _jet(O["u"][b], lat, "NH")
        ms, ls_, _ = _jet(O["u"][b], lat, "SH")
        dm.append(mn - ms)
        dl.append(ln_ - ls_)
        mags += [mn, ms]
        lats += [ln_, ls_]
    res = {}
    subs = []
    for name, d, ref in (("magnitude", dm, np.mean(mags)), ("latitude", dl, np.mean(lats))):
        dmean = float(np.mean(d))
        sig = float(np.std(d, ddof=1))
        bad = abs(dmean) > THRESHOLDS["C5_frac"] * abs(ref) and abs(dmean) > THRESHOLDS["C5_sigmas"] * sig
        subs.append(FAIL if bad else PASS)
        res[name] = {"per_block": d, "mean": dmean, "sigma_block": sig, "verdict": subs[-1]}
    out.append(_crit("C5", worst(subs), "|NH - SH| jet: FAIL only if > 10 % and > 3 sigma_block", **res))
    # C6 (informational)
    mo, lo_, ko = _jet(O["u"].mean(0), lat, "NH")
    sigma = R["sigma"]
    trop = np.abs(lat) < 15.0
    u_sfc = O["u"].mean(0)[trop, -1]
    vt = O["vsTs"].mean(0)
    i, k = np.unravel_index(np.argmax(vt[lat > 0]), vt[lat > 0].shape)
    out.append(_crit("C6", INFO, "paper figures 1-3 (no verdict weight)",
                     jet_ms=mo, jet_lat=lo_, jet_sigma=float(sigma[ko]),
                     jet_consistent=bool(abs(mo - 30.0) <= 10.0 and abs(lo_ - 45.0) <= 10.0),
                     tropical_surface_u_mean=float(u_sfc.mean()),
                     surface_easterlies=bool(u_sfc.mean() < 0),
                     vT_peak_lat=float(abs(lat[lat > 0][i])), vT_peak_sigma=float(sigma[k])))
    return _tier("C", out)


def reference_tier_b(ref_series: dict, ref_b2: Optional[dict], spinup=200, block_days=200,
                     n_blocks=5, days=1200) -> dict:
    """The reference's own B1/B2/B3 from what it stores (B2 from a separate
    exp(ln p_s) computation, since its series has only mean ln p_s)."""
    day = np.asarray(ref_series["day"])
    failed, notes = [], ["B4 not checkable (no KE spectrum stored)",
                         "B5 by construction (single unmodified run, meta.json script_sha256)"]
    missing = False
    if day.max() < days or not np.all(np.isfinite(ref_series["ke_mass_weighted"])):
        failed.append("B1")
    if ref_b2 is None:
        missing = True
        notes.append("B2 unverified (no exp(ln p_s) evidence, b2.json)")
    elif ref_b2.get("rel_change") is None or ref_b2["rel_change"] > THRESHOLDS["B2_rel_ps"]:
        failed.append("B2")
    else:
        notes.append(f"B2 at endpoints only: rel. change {ref_b2['rel_change']:.3e}")
    ke = _series_blocks(day, np.asarray(ref_series["ke_mass_weighted"]), spinup, block_days, n_blocks)
    tt = _series_blocks(day, np.asarray(ref_series["mean_T"]), spinup, block_days, n_blocks)
    if ke is None or tt is None:
        failed.append("B3")
    else:
        c = spinup + block_days * (np.arange(n_blocks) + 0.5)
        span = n_blocks * block_days
        if abs(_lsq_slope(c, ke)) * span > THRESHOLDS["B3_ke_frac"] * ke.mean() or \
                abs(_lsq_slope(c, tt)) * span > THRESHOLDS["B3_T_K"]:
            failed.append("B3")
    verdict = FAIL if failed else (INCOMPLETE if missing else PASS)
    return {"verdict": verdict, "failed": failed, "notes": notes}


# ---------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------

def build_report(run_dir: pathlib.Path, *, reference: Optional[pathlib.Path] = None,
                 reference_config: Optional[pathlib.Path] = None,
                 reference_b2: Optional[pathlib.Path] = None,
                 tier_a_evidence: Optional[pathlib.Path] = None) -> dict:
    run = load_run(run_dir)
    ev = json.loads(pathlib.Path(tier_a_evidence).read_text()) if tier_a_evidence else None
    ref_series = ref_cfg = ref_b2 = None
    if reference is not None and pathlib.Path(reference).exists():
        with np.load(reference, allow_pickle=False) as z:
            ref_series = {k: np.array(z[k]) for k in z.files}
    if reference_config is not None and pathlib.Path(reference_config).exists():
        ref_cfg = json.loads(pathlib.Path(reference_config).read_text())
    if reference_b2 is not None and pathlib.Path(reference_b2).exists():
        ref_b2 = json.loads(pathlib.Path(reference_b2).read_text())
    ref_b = None
    if ref_series is not None and ref_cfg is not None:
        ref_b = reference_tier_b(ref_series, ref_b2, int(ref_cfg["days_spinup"]),
                                 int(ref_cfg["block_days"]),
                                 (int(ref_cfg["days_total"]) - int(ref_cfg["days_spinup"]))
                                 // int(ref_cfg["block_days"]), int(ref_cfg["days_total"]))
    tiers = [tier_a(ev), tier_b(run), tier_c(run, ref_series, ref_cfg, ref_b)]
    return _clean({"experiment_id": run["config"]["experiment_id"],
                   "config_sha256": run["config_sha256"],
                   "git_commit": run["code"].get("git_commit"),
                   "effective_truncation": _effective_truncation(run["config"]),
                   "thresholds": THRESHOLDS,
                   "reference_tier_b": ref_b,
                   "tiers": tiers})


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.6g}"
    if isinstance(v, list):
        return "[" + ", ".join(_fmt(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{" + ", ".join(f"{k}: {_fmt(x)}" for k, x in sorted(v.items())) + "}"
    return str(v)


def render_markdown(rep: dict) -> str:
    lines = [f"# Held–Suarez report: {rep['experiment_id']}", "",
             f"- config_sha256: `{rep['config_sha256']}`",
             f"- git_commit: `{rep['git_commit']}`",
             f"- effective truncation (product cut): T{rep['effective_truncation']}", ""]
    if rep.get("reference_tier_b") is not None:
        rb = rep["reference_tier_b"]
        lines += [f"- reference own tier B: {rb['verdict']} (failed {rb['failed']}; {rb['notes']})", ""]
    lines += ["| tier | verdict |", "|---|---|"]
    lines += [f"| {t['tier']} | **{t['verdict']}** |" for t in rep["tiers"]]
    for t in rep["tiers"]:
        lines += ["", f"## Tier {t['tier']}: {t['verdict']}", "", "| criterion | verdict | detail | values |",
                  "|---|---|---|---|"]
        for c in t["criteria"]:
            vals = "; ".join(f"{k}={_fmt(v)}" for k, v in sorted(c["values"].items()))
            cell = lambda s: str(s).replace("|", "\\|")
            lines.append(f"| {c['id']} | {c['verdict']} | {cell(c['detail'])} | {cell(vals)} |")
    return "\n".join(lines) + "\n"


def write_report(rep: dict, out_dir: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    j = out_dir / "report.json"
    m = out_dir / "REPORT.md"
    j.write_text(json.dumps(rep, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    m.write_text(render_markdown(rep), encoding="utf-8")
    return j, m
