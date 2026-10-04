"""Climate difference between two Held–Suarez runs, in tier-C terms.

Uses the report's own machinery (``load_run``, ``_ours_summary``,
``_wstats``, ``_field_verdict``'s sampling error, ``_jet``) on both runs'
``statistics.npz``: days 200–1200 in five 200-day blocks, Gaussian x
dsigma weights. ``sampling_rms`` is the RMS standard error of the
difference of two 5-block means, so an RMSE near it means the runs are
indistinguishable at this sampling.

    python tools/held_suarez/compare_runs.py runs/hs-T42L20-prod-001 runs/hs-T42L20-prod-002
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib

import numpy as np

from tropoi.run.held_suarez.report import (_gauss_weights, _jet, _ours_summary, _wstats,
                                           load_run)

FIELDS = ("u", "T", "TsTs", "vsTs")


def compare(a_dir: pathlib.Path, b_dir: pathlib.Path) -> dict:
    a, b = load_run(a_dir), load_run(b_dir)
    lat = np.sort(np.asarray(a["stats"]["lat"], float))
    A, B = _ours_summary(a, lat), _ours_summary(b, lat)
    w = _gauss_weights(lat)[:, None] * np.asarray(a["stats"]["dsigma"], float)[None, :]
    out = {"a": str(a_dir), "b": str(b_dir),
           "a_config_sha256": a["config_sha256"], "b_config_sha256": b["config_sha256"],
           "fields": {}, "jets": {}}
    for key in FIELDS:
        n = A[key].shape[0]
        corr, rmse = _wstats(A[key].mean(0), B[key].mean(0), w)
        var = (np.var(A[key], axis=0, ddof=1) + np.var(B[key], axis=0, ddof=1)) / 2.0
        e = math.sqrt(float((w * (2.0 * var / n)).sum() / w.sum()))
        out["fields"][key] = {"corr": corr, "rmse": rmse, "sampling_rms": e,
                              "rmse_over_sampling": rmse / e}
    for hemi in ("NH", "SH"):
        ja, jb = _jet(A["u"].mean(0), lat, hemi), _jet(B["u"].mean(0), lat, hemi)
        out["jets"][hemi] = {"a": ja, "b": jb, "diff_ms": jb[0] - ja[0]}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run_a")
    ap.add_argument("run_b")
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    res = compare(pathlib.Path(args.run_a), pathlib.Path(args.run_b))
    text = json.dumps(res, indent=1, sort_keys=True) + "\n"
    print(text)
    if args.out:
        pathlib.Path(args.out).write_text(text, encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
