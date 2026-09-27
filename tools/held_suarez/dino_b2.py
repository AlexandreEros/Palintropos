"""Tier-B2 evidence for the Dinosaur reference: global-mean surface pressure.

B2 asks for |Δ<p_s>| / <p_s> with <p_s> the Gaussian-weighted global mean of
p_s = exp(ln p_s). The reference driver (dino_reference.py) stored only the
daily mean of ln p_s, which does not measure mass: <ln p_s> <= ln <p_s>
(Jensen), and the gap grows with the p_s variance, so a drift of <ln p_s>
can come entirely from developing eddies at constant mass.

What the reference run retained: ``checkpoint.pkl`` holds the day-1200
spectral state only (overwritten every chunk); ``series.npz`` holds daily
zonal means of u, v, T and eddy products plus four global scalars, none of
which is p_s. So:

* the day-1200 <p_s> is computed from the stored final state;
* the day-0 <p_s> is computed from the initial state REBUILT with the same
  code, configuration and seed (``init_fn(PRNGKey(seed))`` of
  dino_reference.build — deterministic; not a stored array);
* daily <p_s> cannot be established for this run (no daily p_s or full
  state was kept). The driver now records ``mean_ps`` daily for future runs.

Run inside the Dinosaur venv, from the repository root:
    python tools/held_suarez/dino_b2.py --run runs/hs-reference-dinosaur-T42L20-001
Writes ``<run>/b2.json``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import pickle
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))

import dino_reference as dr  # noqa: E402


def global_means(dinosaur, coords, ps, state) -> dict:
    units = dinosaur.scales.units
    lnps = np.asarray(coords.horizontal.to_nodal(state.log_surface_pressure))[0]  # (lon, lat)
    w = np.asarray(coords.horizontal.spherical_harmonics.basis.w)
    w = w / w.sum()
    area = lambda f: float((f.mean(axis=0) * w).sum())
    p = np.asarray(ps.dimensionalize(np.exp(lnps), units.pascal))
    return {"mean_ps_pa": area(p), "mean_lnps_nondim": area(lnps),
            "ln_mean_ps_pa": float(np.log(area(p))), "min_ps_pa": float(p.min()),
            "max_ps_pa": float(p.max())}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    args = ap.parse_args(argv)
    run = pathlib.Path(args.run)
    meta = json.loads((run / "meta.json").read_text())
    cfg = meta["config"]
    import jax
    dinosaur, coords, ps, ref_temps, init_fn, _ = dr.build(cfg, float(meta["dt_s"]))
    init = init_fn(jax.random.PRNGKey(int(meta["seed"])))
    with open(run / "checkpoint.pkl", "rb") as fh:
        saved = pickle.load(fh)
    m0 = global_means(dinosaur, coords, ps, init)
    m1 = global_means(dinosaur, coords, ps, saved["state"])
    rel = abs(m1["mean_ps_pa"] - m0["mean_ps_pa"]) / m0["mean_ps_pa"]
    lnps_series = np.load(run / "series.npz")["mean_lnps"]
    out = {
        "quantity": "Gaussian-weighted global mean of p_s = exp(ln p_s)",
        "day0": m0, "day_final": m1, "day_final_index": int(saved["day"]),
        "rel_change": rel,
        "b2_threshold": 1e-3, "b2_endpoints_verdict": "PASS" if rel <= 1e-3 else "FAIL",
        "initial_state_source": "rebuilt: dino_reference.build(config, dt) init_fn(PRNGKey(seed))",
        "final_state_source": "checkpoint.pkl (stored day-%d state)" % int(saved["day"]),
        "checkpoint_sha256": hashlib.sha256((run / "checkpoint.pkl").read_bytes()).hexdigest(),
        "series_sha256": hashlib.sha256((run / "series.npz").read_bytes()).hexdigest(),
        "mean_lnps_series_drift_nondim": float(lnps_series[-1] - lnps_series[0]),
        "cannot_establish": "daily <p_s> (only daily <ln p_s> was stored); B2 is established "
                            "at the endpoints day 0 and day %d only" % int(saved["day"]),
    }
    (run / "b2.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
