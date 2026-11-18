"""Layer-scale vertical structure of a radiative run, from its native saved
states (CPU, NumPy only; read-only).

    python tools/radiation/thai_vertical_coherence.py RUN_DIR [RUN_DIR ...]
        [--top 8] [--evidence-dir docs/radiation/evidence] [--write]

For every saved state ``states/state-dNNNNN.npy`` of each run this records,
from the full-precision spectral coefficients (issue #47):

* **temperature.** The area-weighted global mean of every layer.
* **alternating component.** In each column, the top ``--top`` layer
  temperatures are fitted by least squares with ``c0 + c1 s + c2 s^2 +
  a (-1)^k``, ``s = ln sigma_k`` (layer index ``k`` from the top, 0-based).
  Since ``ln p_k = ln sigma_k + ln p_s`` and ``ln p_s`` is constant in a
  column, this is the same as a quadratic trend in log layer pressure. The
  record is ``sqrt(<a^2>)``, the area-weighted RMS of the fitted amplitude
  (K): the RMS over columns of the alternating part ``a (-1)^k``.
* **coherence.** The area-weighted Pearson correlation of relative
  vorticity and of horizontal divergence between every pair of adjacent
  layers, and each layer's area-weighted RMS.

Sampling and weighting: each field is synthesized exactly from its
coefficients (:func:`tropoi.spatial.transforms.host_synthesis.synthesize_latlon`)
on the ``nlat`` Gauss-Legendre latitudes and ``nlon`` uniform longitudes of
the run's own configuration. Gauss weights times uniform longitude weights
integrate products of two degree-``l_max`` fields exactly, so every mean,
variance and covariance here is the exact sphere integral of the model's
band-limited fields, not a sample estimate. Native layer ``k`` (0-based,
top first) is the issue's "level ``k + 1``".

The run's ``config.json`` must reproduce its own ``config_sha256``.
Without ``--write`` the results are only printed. With ``--write`` they are
saved as ``<evidence-dir>/<run name>/vertical_coherence.json``; an existing
file is refused (evidence is not overwritten).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

SCHEMA = "palintropos.radiation.vertical_coherence/1"
OUT_FILE = "vertical_coherence.json"


def gauss_mesh(nlat: int, nlon: int):
    """Latitudes and longitudes (degrees) and area weights ``(nlat, nlon)``
    that sum to one and integrate degree <= 2 nlat - 1 exactly."""
    x, w = np.polynomial.legendre.leggauss(nlat)
    lat = np.degrees(np.arcsin(x))
    lon = np.arange(nlon) * (360.0 / nlon)
    weights = np.repeat((w / 2.0)[:, None] / nlon, nlon, axis=1)
    return lat, lon, weights


def synthesize_levels(coeffs, lat, lon) -> np.ndarray:
    """``(K, nlat, nlon)`` grid values of ``(K, L+1, L+1)`` coefficients."""
    from tropoi.spatial.transforms.host_synthesis import synthesize_latlon
    return np.stack([synthesize_latlon(c, lat, lon) for c in coeffs])


def area_mean(field, weights) -> np.ndarray:
    """Area mean over the last two axes."""
    return np.tensordot(field, weights, axes=([-2, -1], [0, 1]))


def weighted_correlation(a, b, weights) -> float:
    """Area-weighted Pearson correlation of two fields (means removed)."""
    a = a - area_mean(a, weights)
    b = b - area_mean(b, weights)
    return float(area_mean(a * b, weights)
                 / np.sqrt(area_mean(a * a, weights) * area_mean(b * b, weights)))


def alternating_design(sigma) -> np.ndarray:
    """Columns ``1, s, s^2, (-1)^k`` with ``s = ln sigma``."""
    s = np.log(np.asarray(sigma, dtype=np.float64))
    k = np.arange(s.size)
    return np.stack([np.ones_like(s), s, s * s, (-1.0) ** k], axis=1)


def alternating_amplitude(temperature, sigma) -> np.ndarray:
    """Least-squares amplitude ``a`` of ``(-1)^k`` in every column.

    ``temperature`` is ``(K, ...)`` with ``K == len(sigma)`` (top first);
    returns an array of the trailing shape.
    """
    t = np.asarray(temperature, dtype=np.float64)
    design = alternating_design(sigma)
    if t.shape[0] != design.shape[0]:
        raise ValueError(f"{t.shape[0]} layers against {design.shape[0]} sigma values")
    if design.shape[0] <= design.shape[1]:
        raise ValueError("the fit needs more layers than its four parameters")
    coef, *_ = np.linalg.lstsq(design, t.reshape(t.shape[0], -1), rcond=None)
    return coef[3].reshape(t.shape[1:])


def state_metrics(state, nlev: int, sigma, lat, lon, weights, *, top: int = 8) -> dict:
    """The day's record for one ``(3 nlev + 1, L+1, L+1)`` PE state."""
    x = np.asarray(state)
    if x.shape[0] != 3 * nlev + 1:
        raise ValueError(f"state has {x.shape[0]} rows, expected {3 * nlev + 1}")
    if not np.all(np.isfinite(x)):
        raise ValueError("non-finite saved state")
    zeta = synthesize_levels(x[0:nlev], lat, lon)
    delta = synthesize_levels(x[nlev:2 * nlev], lat, lon)
    temperature = synthesize_levels(x[2 * nlev:3 * nlev], lat, lon)
    amplitude = alternating_amplitude(temperature[:top], sigma[:top])

    def pairs(f):
        return [weighted_correlation(f[k], f[k + 1], weights) for k in range(nlev - 1)]

    def rms(f):
        return [float(np.sqrt(area_mean(f[k] * f[k], weights))) for k in range(nlev)]

    return {
        "global_mean_temperature": [float(v) for v in area_mean(temperature, weights)],
        "alternating_rms_top": float(np.sqrt(area_mean(amplitude * amplitude, weights))),
        "alternating_mean_top": float(area_mean(amplitude, weights)),
        "zeta_adjacent_correlation": pairs(zeta),
        "delta_adjacent_correlation": pairs(delta),
        "zeta_rms": rms(zeta),
        "delta_rms": rms(delta),
    }


def run_metrics(run: pathlib.Path, *, top: int = 8) -> dict:
    from tropoi.run.radiative.config import RadiativeConfig
    from tropoi.run.radiative.model import build_sigma

    doc = json.loads((run / "config.json").read_text(encoding="utf-8"))
    cfg = RadiativeConfig.from_dict(doc["config"])
    if cfg.sha256() != doc["config_sha256"]:
        raise SystemExit(f"{run}: config.json does not reproduce its own hash")
    sigma = build_sigma(cfg).full_levels_array()
    nlev = sigma.size
    lat, lon, weights = gauss_mesh(cfg.nlat, cfg.nlon)
    days = {}
    for path in sorted((run / "states").glob("state-d*.npy")):
        day = int(path.stem.split("-d")[1])
        days[str(day)] = state_metrics(np.load(path), nlev, sigma, lat, lon, weights, top=top)
    if not days:
        raise SystemExit(f"{run}: no saved states")
    return {
        "schema": SCHEMA, "run": run.name, "config_sha256": doc["config_sha256"],
        "git_commit": doc["code"].get("git_commit"), "git_dirty": doc["code"].get("git_dirty"),
        "thermal_source": cfg.thermal_source, "nlev": nlev, "nlat": cfg.nlat, "nlon": cfg.nlon,
        "l_max": cfg.l_max, "sigma_full": [float(s) for s in sigma], "top_layers_fitted": top,
        "fit": "T_k = c0 + c1 ln(sigma_k) + c2 ln(sigma_k)^2 + a (-1)^k, k = 0..top-1",
        "weighting": "Gauss-Legendre x uniform longitude; exact for products of the "
                     "band-limited fields",
        "days": days,
    }


def _print(result: dict) -> None:
    print(f"{result['run']}  [{result['thermal_source']}]  {result['config_sha256'][:12]}")
    print("  day  RMS alt(K)  mean T levels 1-4 (K)              "
          "zeta r 1-2 2-3 3-4       delta r 1-2 2-3 3-4")
    for day, row in result["days"].items():
        t = row["global_mean_temperature"][:4]
        z = row["zeta_adjacent_correlation"][:3]
        d = row["delta_adjacent_correlation"][:3]
        print(f"  {day:>3}  {row['alternating_rms_top']:9.3f}  "
              + " ".join(f"{v:7.2f}" for v in t) + "   "
              + " ".join(f"{v:+.2f}" for v in z) + "   "
              + " ".join(f"{v:+.2f}" for v in d))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+", type=pathlib.Path)
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--evidence-dir", type=pathlib.Path,
                    default=pathlib.Path(__file__).resolve().parents[2] / "docs" / "radiation"
                    / "evidence")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args(argv)
    for run in args.runs:
        result = run_metrics(run, top=args.top)
        _print(result)
        if args.write:
            target = args.evidence_dir / run.name / OUT_FILE
            if target.exists():
                raise SystemExit(f"{target} exists (evidence is not overwritten)")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n",
                              encoding="utf-8")
            print(f"  wrote {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
