"""CPU validation gates G1-G5b for the standalone conductive substrate.

The definitions are fixed in the Ben 2 package addendum
``regolith/gates-m1/ADDENDUM-G1-G5b.md`` (rev. 1), written before this
script ran. Mandatory gates: G1-M, G2, G3-M, G4, G5. Reported evidence: G1-C,
G1 at P = 1 and 2 d, G3-D, G5b. The run stops at the first failed mandatory
gate.

Usage::

    python tests/audit_substrate_gates.py OUTDIR [--only G1M,G2,...]
        [--grids production|historical] [--fixture PATH]

Writes one JSON per gate to OUTDIR. NumPy on CPU (decision D3). Gate names
for ``--only`` are those in ``GATES`` (G1M, G1C, G2, G3, G4, G5, G5b); an
unknown name is an error.

Grids:
* ``production`` (default) builds every grid with the current public layering
  API. Columns deeper than 3.5 m (G1-M at P > 12.4 d) and G5b top thicknesses
  other than 2 cm use ``geometric_interfaces(top, D, stretch=<default
  stretch>, depth_is_minimum=True)``: the default top and stretch are kept
  exactly, and the column ends at the first geometric interface at or below
  the requested depth D. That keeps the depth >= 6 delta requirement and the
  [1, 1.5] ratio guarantee.
* ``historical`` reads the exact interface grids of the original milestone-1
  run from the fixture ``tests/fixtures/substrate_m1_historical_grids.json``,
  so the frozen results can be reproduced without re-implementing the
  original grid-extension algorithm.
"""
from __future__ import annotations

import argparse
import cmath
import hashlib
import json
import math
import pathlib
import platform
import sys
import time

import numpy as np

from tropoi.physics.substrate import ConductiveSubstrate, geometric_interfaces
from tropoi.physics.surface import SlabSurface

DAY = 86400.0
C_V = 2.0e6
INERTIA = 2000.0
K = INERTIA * INERTIA / C_V          # 2 W m^-1 K^-1
KAPPA = K / C_V                      # 1e-6 m^2 s^-1
DT = 600.0
DEPTH = 3.5
TOP = 0.02
LAYERS = 30
SEED = 20261124
H_SET = (0.93, 2.93, 8.14, 30.0)
TOL_AMP = 0.01
TOL_PHASE = 0.01
U = 2.0 ** -53


def gamma(n):
    return n * U / (1.0 - n * U)


def default_interfaces():
    return geometric_interfaces(TOP, DEPTH, layers=LAYERS)


def default_stretch():
    e = np.asarray(default_interfaces())
    dz = np.diff(e)
    return float(dz[1] / dz[0])


FIXTURE = pathlib.Path(__file__).resolve().parent / "fixtures" / "substrate_m1_historical_grids.json"
GATE_NAMES = ("G1M", "G1C", "G2", "G3", "G4", "G5", "G5b")
HISTORICAL = None          # {key: interfaces} when --grids historical


def grid(key, build):
    """Interfaces for one gate configuration: the historical fixture's exact
    grid in historical mode, otherwise ``build()``."""
    if HISTORICAL is not None:
        if key not in HISTORICAL:
            raise KeyError(f"historical fixture has no grid {key!r}")
        return tuple(HISTORICAL[key])
    return tuple(build())


def extended_interfaces(depth, top=TOP):
    """Production grid for a column of at least ``depth``. Uses the default
    3.5 m grid when it suffices; otherwise top ``top`` with the default
    stretch, ending at the first geometric interface at or below ``depth``."""
    if top == TOP and depth <= DEPTH * (1 + 1e-12):
        return default_interfaces()
    return geometric_interfaces(top, depth, stretch=default_stretch(), depth_is_minimum=True)


def substrate(interfaces, skin="layer"):
    return ConductiveSubstrate(volumetric_heat_capacity=C_V, thermal_inertia=INERTIA,
                               interfaces=interfaces, skin=skin)


# ------------------------------------------------------------------ reconstruction
def reconstruct(edges, values, z):
    """Mean-preserving quadratic through the layer containing z and its two
    neighbours (one-sided at the ends); evaluated at z. Linear in values."""
    e = np.asarray(edges, dtype=np.float64)
    dz = np.diff(e)
    n = dz.size
    if n < 3:
        raise ValueError("reconstruction needs >= 3 layers")
    j = int(np.searchsorted(e, z, side="right") - 1)
    j = min(max(j, 0), n - 1)
    s = min(max(j - 1, 0), n - 3)
    idx = range(s, s + 3)
    A = np.array([[((e[i + 1] - z) ** (p + 1) - (e[i] - z) ** (p + 1)) / ((p + 1) * dz[i])
                   for p in range(3)] for i in idx])
    w = np.linalg.solve(A.T, np.array([1.0, 0.0, 0.0]))   # value at z = w . means
    return sum(w[t] * values[i] for t, i in enumerate(idx))


# ------------------------------------------------------------------ analytic forms
def theta_inf(z, F0, q):
    return F0 / (K * q) * cmath.exp(-q * z)


def theta_fin(z, F0, q, D):
    return F0 * (cmath.exp(-q * z) + cmath.exp(-q * (2 * D - z))) / (K * q * (1 - cmath.exp(-2 * q * D)))


def theta_fin_mean(a, b, F0, q, D):
    num = (cmath.exp(-q * a) - cmath.exp(-q * b)) + (cmath.exp(-q * (2 * D - b)) - cmath.exp(-q * (2 * D - a)))
    return F0 * num / (K * q * q * (b - a) * (1 - cmath.exp(-2 * q * D)))


def theta_inf_mean(a, b, F0, q):
    return F0 * (cmath.exp(-q * a) - cmath.exp(-q * b)) / (K * q * q * (b - a))


def ierfc(x):
    return math.exp(-x * x) / math.sqrt(math.pi) - x * math.erfc(x)


def i2erfc(x):
    return 0.25 * ((1 + 2 * x * x) * math.erfc(x) - 2 / math.sqrt(math.pi) * x * math.exp(-x * x))


def mu1(bi):
    lo, hi = 1e-14, math.pi / 2 - 1e-15
    for _ in range(300):
        m = 0.5 * (lo + hi)
        if m * math.tan(m) < bi:
            lo = m
        else:
            hi = m
    return 0.5 * (lo + hi)


# ------------------------------------------------------------------ periodic runner
def run_periodic(edges, P, dt, skin="layer", F0=100.0):
    """Six periods from the exact finite-column periodic state; returns per-layer
    complex first-harmonic amplitudes over period 6 and surface series."""
    sub = substrate(edges, skin)
    e = sub.interfaces
    D = float(e[-1])
    w = 2 * math.pi / P
    delta = math.sqrt(KAPPA * P / math.pi)
    q = (1 + 1j) / delta
    init = np.array([theta_fin_mean(e[i], e[i + 1], F0, q, D).real for i in range(sub.layers)])
    st = sub.initial_state(1)
    st = st._replace(anomaly=init.reshape(-1, 1).copy())
    nsteps = int(math.ceil(6 * P / dt - 1e-9))
    t_s, th_s, fl_s = [], [], []
    for n in range(nsteps):
        t0, t1 = n * dt, (n + 1) * dt
        fbar = F0 / (w * dt) * (math.sin(w * t1) - math.sin(w * t0))
        st = sub.step(st, np.array([fbar]), dt)
        if t1 > 5 * P:
            t_s.append(t1)
            th_s.append(st.anomaly[:, 0].copy())
            fl_s.append(float(st.surface_flux[0]))
    t = np.array(t_s)
    X = np.column_stack([np.ones_like(t), np.cos(w * t), np.sin(w * t)])
    Y = np.array(th_s)
    coef, *_ = np.linalg.lstsq(X, Y, rcond=None)
    amp = coef[1] - 1j * coef[2]          # T = Re[amp e^{i w t}]
    return dict(sub=sub, edges=e, D=D, q=q, delta=delta, w=w, amp=amp, t=t, Y=Y,
                flux=np.array(fl_s), F0=F0)


def compare_point(run, z):
    num = reconstruct(run["edges"], run["amp"], z)
    ana = theta_inf(z, run["F0"], run["q"])
    fin = theta_fin(z, run["F0"], run["q"], run["D"])
    e = run["edges"]
    exact_means = [theta_fin_mean(e[i], e[i + 1], run["F0"], run["q"], run["D"]) for i in range(len(e) - 1)]
    rec = reconstruct(e, exact_means, z)
    j = int(min(max(np.searchsorted(e, z, side="right") - 1, 0), len(e) - 2))
    lm_num = run["amp"][j]
    lm_ana = theta_inf_mean(e[j], e[j + 1], run["F0"], run["q"])
    r = num / ana
    return {
        "z": z, "amp_err": abs(r) - 1.0, "phase_err_P": cmath.phase(r) / (2 * math.pi),
        "complex_rel_err": abs(r - 1.0),
        "vs_finite_column": {"amp_err": abs(num / fin) - 1.0, "phase_err_P": cmath.phase(num / fin) / (2 * math.pi)},
        "finite_vs_semi_infinite": {"amp": abs(fin / ana) - 1.0, "phase_P": cmath.phase(fin / ana) / (2 * math.pi)},
        "reconstruction_only": {"amp": abs(rec / fin) - 1.0, "phase_P": cmath.phase(rec / fin) / (2 * math.pi)},
        "layer_mean": {"layer": j, "amp_err": abs(lm_num / lm_ana) - 1.0,
                       "phase_err_P": cmath.phase(lm_num / lm_ana) / (2 * math.pi)},
    }


# ------------------------------------------------------------------ gates
def gate_g1m():
    periods = (1.0, 2.0, 6.1, 20.0, 61.0, 200.0)
    rows, ok = [], True
    for Pd in periods:
        P = Pd * DAY
        delta = math.sqrt(KAPPA * P / math.pi)
        D = max(6 * delta, DEPTH)
        edges = grid(f"G1M:P={Pd}", lambda: extended_interfaces(D))
        D_req, D = D, float(edges[-1])
        run = run_periodic(edges, P, DT)
        pts = {"top_centre": 0.5 * (edges[1] - edges[0]), "delta": delta, "two_delta": 2 * delta}
        res = {k: compare_point(run, z) for k, z in pts.items()}
        mandatory = Pd >= 6.1
        passed = all(abs(v["amp_err"]) <= TOL_AMP and abs(v["phase_err_P"]) <= TOL_PHASE for v in res.values())
        if mandatory:
            ok = ok and passed
        rows.append({"P_days": Pd, "D_m": D, "D_requested_m": D_req, "D_over_delta": D / delta,
                     "layers": len(edges) - 1,
                     "extended": D > DEPTH * (1 + 1e-12), "mandatory": mandatory,
                     "pass": passed, "points": res})
    return {"gate": "G1-M", "mandatory": True, "pass": ok, "dt": DT,
            "tolerances": {"amp": TOL_AMP, "phase_P": TOL_PHASE}, "periods": rows}


def gate_g1c():
    P = 6.1 * DAY
    delta = math.sqrt(KAPPA * P / math.pi)
    pts = {"z_1cm": 0.01, "delta": delta, "two_delta": 2 * delta}

    def measure(edges, dt):
        run = run_periodic(edges, P, dt)
        return {k: reconstruct(run["edges"], run["amp"], z) for k, z in pts.items()}

    def summarise(series, key_name, ratio_name):
        out = []
        for i, (key, vals) in enumerate(series):
            row = {key_name: key}
            for p, z in pts.items():
                ana = theta_inf(z, 100.0, (1 + 1j) / delta)
                row[p] = {"err_vs_analytic": abs(vals[p] / ana - 1.0)}
                if i > 0:
                    prev = series[i - 1][1]
                    row[p]["diff_from_previous"] = abs(vals[p] - prev[p]) / abs(ana)
                if i > 1:
                    d1 = abs(series[i - 1][1][p] - series[i - 2][1][p])
                    d2 = abs(vals[p] - series[i - 1][1][p])
                    row[p]["observed_order_self"] = math.log(d1 / d2) / math.log(ratio_name)
                if i > 0:
                    e1 = abs(series[i - 1][1][p] / ana - 1.0)
                    e2 = row[p]["err_vs_analytic"]
                    row[p]["observed_order_vs_analytic"] = math.log(e1 / e2) / math.log(ratio_name)
            out.append(row)
        return out

    space = []
    for N in (25, 50, 100, 200):
        edges = grid(f"uniform:N={N}", lambda: np.linspace(0.0, DEPTH, N + 1))
        space.append((N, measure(edges, DT)))
    time_ = []
    edges200 = grid("uniform:N=200", lambda: np.linspace(0.0, DEPTH, 201))
    for dt in (4800.0, 2400.0, 1200.0, 600.0, 300.0):
        time_.append((dt, measure(edges200, dt)))
    return {"gate": "G1-C", "mandatory": False, "expected_orders": {"space": 2, "time": 1},
            "P_days": 6.1, "D_m": DEPTH, "space_dt": DT,
            "space": summarise(space, "N", 2.0), "time_N": 200,
            "time": summarise(time_, "dt", 2.0)}


def gate_g2():
    edges = grid("default", default_interfaces)
    sub = substrate(edges)
    st = sub.initial_state(1)
    F = 100.0
    tmax = DEPTH ** 2 / (16 * KAPPA)
    z = 0.5 * (edges[1] - edges[0])
    worst, worst_t, worst_lm = 0.0, None, 0.0
    n = 0
    samples = []
    while True:
        n += 1
        st = sub.step(st, np.array([F]), DT)
        t = n * DT
        if t > tmax:
            break
        if n >= 20:
            s = math.sqrt(KAPPA * t)
            ana = 2 * F / K * s * ierfc(z / (2 * s))
            num = reconstruct(edges, st.anomaly[:, 0], z)
            err = num / ana - 1.0
            a, b = edges[0], edges[1]
            lm_ana = 2 * F / K * s * 2 * s * (i2erfc(a / (2 * s)) - i2erfc(b / (2 * s))) / (b - a)
            lm_err = st.anomaly[0, 0] / lm_ana - 1.0
            worst_lm = max(worst_lm, abs(lm_err))
            if abs(err) > worst:
                worst, worst_t = abs(err), t
            if n in (20, 40, 100, 200, 500, 1000) or t + DT > tmax:
                samples.append({"t_s": t, "steps": n, "rel_err": err, "layer_mean_rel_err": lm_err})
    return {"gate": "G2", "mandatory": True, "pass": worst <= 0.01, "tolerance": 0.01,
            "window_s": [20 * DT, tmax], "z_m": z, "max_abs_rel_err": worst,
            "at_t_s": worst_t, "layer_mean_max_abs_rel_err": worst_lm, "samples": samples}


def run_robin(skin):
    edges = grid("default", default_interfaces)
    sub = substrate(edges, skin)
    L = sub.depth
    h = np.array(H_SET)
    mus = [mu1(hh * L / K) for hh in H_SET]
    lam = np.array([KAPPA * m * m / L ** 2 for m in mus])
    e = sub.interfaces
    init = np.empty((sub.layers, len(H_SET)))
    for c, m in enumerate(mus):
        for i in range(sub.layers):
            a, b = e[i], e[i + 1]
            init[i, c] = 10.0 * L / (m * (b - a)) * (math.sin(m * (L - a) / L) - math.sin(m * (L - b) / L))
    st = sub.initial_state(len(H_SET))._replace(anomaly=init)
    tend = 3.0 / lam.min()
    nsteps = int(math.ceil(tend / DT))
    ts, Es = [], []
    for n in range(nsteps):
        ts_old = sub.surface_temperature(st) - 300.0
        st = sub.step(st, -h * ts_old, DT, flux_derivative=-h)
        ts.append((n + 1) * DT)
        Es.append(np.asarray(sub.energy(st)).copy())
    ts = np.array(ts)
    Es = np.array(Es)
    rows = []
    for c, hh in enumerate(H_SET):
        sel = (ts >= 0.5 / lam[c]) & (ts <= 3.0 / lam[c])
        slope = np.polyfit(ts[sel], np.log(Es[sel, c]), 1)[0]
        lam_num = -slope
        rows.append({"h": hh, "Bi": hh * L / K, "mu1": mus[c], "efold_days_analytic": 1 / lam[c] / DAY,
                     "lambda_analytic": lam[c], "lambda_num": lam_num, "rel_err": lam_num / lam[c] - 1.0,
                     "fit_points": int(sel.sum())})
    return rows


def gate_g3():
    prod = run_robin("layer")
    diag = run_robin("diagnosed")
    ok = all(abs(r["rel_err"]) <= 0.01 for r in prod)
    return {"gate": "G3-M", "mandatory": True, "pass": ok, "tolerance": 0.01,
            "production_formulation": "layer (identification, provisional)",
            "G3-M_identification": prod, "G3-D_diagnosed_reported_only": diag}


def gate_g4():
    sub = substrate(grid("G4", lambda: (0.0, 1.0)))
    slab = SlabSurface(2.0e6)
    rng = np.random.default_rng(SEED)
    n = 100000
    F = rng.uniform(-200.0, 200.0, size=n)
    st = sub.initial_state(1)
    T_slab = np.array([300.0])
    m = float(sub.layer_capacity[0])
    acc = 0.0
    worst_ratio, worst_diff = 0.0, 0.0
    diffs = []
    g2, g3 = gamma(2), gamma(3)
    for j in range(n):
        theta_j = float(st.anomaly[0, 0])
        Fj = np.array([F[j]])
        st = sub.step(st, Fj, DT)
        T_slab = slab.advance(T_slab, Fj, DT)
        delta_j = abs(DT * F[j] / m)
        acc += g2 * delta_j + 0.5 * float(np.spacing(abs(T_slab[0]))) + g3 * (abs(theta_j) + delta_j)
        T_sub = float(sub.temperature(st)[0, 0])
        bound = acc + 0.5 * float(np.spacing(abs(T_sub)))
        diff = abs(T_sub - float(T_slab[0]))
        if diff > bound:
            return {"gate": "G4", "mandatory": True, "pass": False, "failed_at_step": j + 1,
                    "diff": diff, "bound": bound}
        worst_ratio = max(worst_ratio, diff / bound)
        worst_diff = max(worst_diff, diff)
        if (j + 1) in (1, 10, 100, 1000, 10000, 100000):
            diffs.append({"step": j + 1, "diff_K": diff, "bound_K": bound})
    rms_walk = math.sqrt(n) * 0.5 * float(np.spacing(300.0)) / math.sqrt(3.0)
    return {"gate": "G4", "mandatory": True, "pass": True, "steps": n, "max_diff_K": worst_diff,
            "max_diff_over_bound": worst_ratio, "final_bound_K": bound,
            "rms_random_walk_expectation_K": rms_walk, "checkpoints": diffs,
            "bitwise_equal_steps_fraction": None}


class ExactSum:
    """Shewchuk partials: exact running sum of floats."""

    def __init__(self):
        self.p = []

    def add(self, x):
        i = 0
        for y in self.p:
            if abs(x) < abs(y):
                x, y = y, x
            hi = x + y
            lo = y - (hi - x)
            if lo:
                self.p[i] = lo
                i += 1
            x = hi
        self.p[i:] = [x]

    def value(self):
        return math.fsum(self.p)


def gate_g5():
    edges = grid("default", default_interfaces)
    sub = substrate(edges)
    fbs = np.array([0.0, 0.1, -0.1])
    ncol = fbs.size
    rng = np.random.default_rng(SEED)
    n = 100000
    F = rng.uniform(-200.0, 200.0, size=(n, ncol))
    st = sub.initial_state(ncol)
    m = sub.layer_capacity[:, None]
    fs = [ExactSum() for _ in range(ncol)]
    fbsum = [ExactSum() for _ in range(ncol)]
    den = [ExactSum() for _ in range(ncol)]
    worst = np.zeros(ncol)
    worst_step = np.zeros(ncol, dtype=int)
    for k in range(n):
        st = sub.step(st, F[k], DT, bottom_heat_flux=fbs)
        applied = st.surface_flux
        prod = m * st.anomaly
        for c in range(ncol):
            fs[c].add(float(applied[c]))
            fbsum[c].add(float(fbs[c]))
            den[c].add(abs(float(applied[c]) + float(fbs[c])))
            E = math.fsum(prod[:, c])
            inflow = DT * math.fsum([fs[c].value(), fbsum[c].value()])
            resid = abs(E - inflow)
            denom = DT * den[c].value()
            ratio = resid / denom if denom > 0 else (0.0 if resid == 0 else math.inf)
            if ratio > 1e-12:
                return {"gate": "G5", "mandatory": True, "pass": False, "column_F_b": float(fbs[c]),
                        "failed_at_step": k + 1, "residual_J_m2": resid, "denominator_J_m2": denom,
                        "ratio": ratio}
            if ratio > worst[c]:
                worst[c], worst_step[c] = ratio, k + 1
    applied_equals_prescribed = bool(np.all(np.asarray(st.surface_flux) == F[-1]))
    return {"gate": "G5", "mandatory": True, "pass": True, "steps": n, "tolerance_ratio": 1e-12,
            "columns": [{"F_b": float(fbs[c]), "max_ratio": float(worst[c]),
                         "at_step": int(worst_step[c])} for c in range(ncol)],
            "applied_flux_equals_prescribed_last_step": applied_equals_prescribed}


def gate_g5b():
    tops = (0.005, 0.01, 0.02, 0.04)
    out = {"gate": "G5b", "mandatory": False, "tops_m": tops, "h_set": H_SET,
           "periodic": [], "step": []}
    for Pd, dts in ((1.0, (600.0, 60.0)), (6.1, (600.0, 60.0)), (61.0, (600.0,))):
        P = Pd * DAY
        delta = math.sqrt(KAPPA * P / math.pi)
        D = max(6 * delta, DEPTH)
        for dt in dts:
            rows = []
            for top in tops:
                edges = grid(f"G5b:periodic:P={Pd}:top={top}", lambda: extended_interfaces(D, top))
                run = run_periodic(edges, P, dt)
                r = run["sub"].skin_resistance
                T1 = run["Y"][:, 0]
                ana_s = theta_inf(0.0, run["F0"], run["q"])
                ana_series = np.real(ana_s * np.exp(1j * run["w"] * run["t"]))
                row = {"top_m": top, "layers": len(edges) - 1, "depth_m": float(edges[-1])}
                for name, series in (("identification", T1), ("diagnosed", T1 + r * run["flux"])):
                    X = np.column_stack([np.ones_like(run["t"]), np.cos(run["w"] * run["t"]),
                                         np.sin(run["w"] * run["t"])])
                    cf, *_ = np.linalg.lstsq(X, series, rcond=None)
                    amp = cf[1] - 1j * cf[2]
                    ratio = amp / ana_s
                    maxdT = float(np.max(np.abs(series - ana_series)))
                    row[name] = {"amp_err": abs(ratio) - 1.0, "phase_err_P": cmath.phase(ratio) / (2 * math.pi),
                                 "complex_rel_err": abs(ratio - 1.0), "max_abs_dTs_K": maxdT,
                                 "flux_err_W_m2": {str(h): h * maxdT for h in H_SET}}
                row["skin_minus_T1_amplitude_K"] = float(abs(r) * np.max(np.abs(run["flux"])))
                rows.append(row)
            _orders(rows, tops, ("identification", "diagnosed"), "complex_rel_err")
            out["periodic"].append({"P_days": Pd, "dt": dt, "D_m": D, "rows": rows})
    for dt in (600.0, 60.0):
        rows = []
        F = 100.0
        tmax = DEPTH ** 2 / (16 * KAPPA)
        for top in tops:
            edges = grid(f"G5b:step:top={top}", lambda: extended_interfaces(DEPTH, top))
            sub = substrate(edges)
            r = sub.skin_resistance
            st = sub.initial_state(1)
            n = 0
            worst = {"identification": 0.0, "diagnosed": 0.0}
            last = {}
            while True:
                n += 1
                st = sub.step(st, np.array([F]), dt)
                t = n * dt
                if t > tmax:
                    break
                if n >= 20:
                    ana = 2 * F * math.sqrt(t / math.pi) / INERTIA
                    T1 = float(st.anomaly[0, 0])
                    for name, val in (("identification", T1), ("diagnosed", T1 + r * float(st.surface_flux[0]))):
                        err = val / ana - 1.0
                        worst[name] = max(worst[name], abs(err))
                        last[name] = {"t_s": t, "rel_err": err, "dTs_K": val - ana}
            row = {"top_m": top, "layers": len(edges) - 1, "depth_m": float(edges[-1])}
            for name in worst:
                row[name] = {"max_abs_rel_err": worst[name], "end_of_window": last[name],
                             "flux_err_end_W_m2": {str(h): h * abs(last[name]["dTs_K"]) for h in H_SET}}
            rows.append(row)
        _orders(rows, tops, ("identification", "diagnosed"), "max_abs_rel_err")
        out["step"].append({"dt": dt, "window_s": [20 * dt, tmax], "rows": rows})
    return out


def _orders(rows, tops, names, key):
    for i in range(1, len(rows)):
        for name in names:
            e1, e0 = rows[i][name][key], rows[i - 1][name][key]
            rows[i][name]["observed_order_vs_previous_top"] = (
                math.log(e1 / e0) / math.log(tops[i] / tops[i - 1]) if e0 > 0 and e1 > 0 else None)


GATES = [("G1M", gate_g1m), ("G1C", gate_g1c), ("G2", gate_g2), ("G3", gate_g3),
         ("G4", gate_g4), ("G5", gate_g5), ("G5b", gate_g5b)]
assert tuple(n for n, _ in GATES) == GATE_NAMES


def _default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, complex):
        return [o.real, o.imag]
    raise TypeError(type(o))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("--only", default="")
    ap.add_argument("--grids", choices=("production", "historical"), default="production")
    ap.add_argument("--fixture", default=str(FIXTURE))
    args = ap.parse_args(argv)
    only = {s for s in args.only.split(",") if s}
    unknown = sorted(only - set(GATE_NAMES))
    if unknown:
        print(f"unknown gate name(s) {unknown}; valid: {', '.join(GATE_NAMES)}", file=sys.stderr)
        return 2
    global HISTORICAL
    HISTORICAL = None
    fixture_meta = None
    if args.grids == "historical":
        raw = pathlib.Path(args.fixture).read_bytes()
        HISTORICAL = json.loads(raw)["grids"]
        fixture_meta = {"path": args.fixture, "sha256": hashlib.sha256(raw).hexdigest()}
    out = pathlib.Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    meta = {"python": sys.version, "numpy": np.__version__, "platform": platform.platform(),
            "default_interfaces": list(default_interfaces()), "default_stretch": default_stretch(),
            "dt": DT, "k": K, "kappa": KAPPA, "seed": SEED,
            "grids": args.grids, "fixture": fixture_meta}
    (out / "meta.json").write_text(json.dumps(meta, indent=1, default=_default))
    for name, fn in GATES:
        if only and name not in only:
            continue
        t0 = time.time()
        res = fn()
        res["wall_s"] = time.time() - t0
        (out / f"{name}.json").write_text(json.dumps(res, indent=1, default=_default))
        status = "PASS" if res.get("pass") else ("FAIL" if res.get("mandatory") else "REPORTED")
        print(f"{name}: {status} ({res['wall_s']:.1f} s)", flush=True)
        if res.get("mandatory") and not res.get("pass"):
            print("Stopping at the first failed mandatory gate.", flush=True)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
