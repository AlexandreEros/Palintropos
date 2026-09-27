"""Online Held–Suarez statistics (CPU-only, NumPy float64).

Once per simulated day the driver hands :meth:`OnlineStatistics.add` one
*daily sample* (host arrays, see :func:`tropoi.run.held_suarez.experiment.
daily_sample`):

* zonal means on the state grid, shape (nlat, nlev), latitude north->south:
  ``u, v, T`` and the eddy (deviation from the instantaneous zonal mean)
  products ``TsTs = [T'^2]``, ``usus = [u'^2]``, ``vsTs = [v'T']``,
  ``usvs = [u'v']`` — the same definitions as the Dinosaur reference driver
  (tools/held_suarez/dino_reference.py), so the two cores are compared like
  for like;
* ``ke_spectrum`` (l_max+1,): the global-mean kinetic energy per degree,
  sigma-weighted like the scalar KE;
* scalars: ``mean_ps`` (Gaussian-weighted global mean of p_s = exp(ln p_s),
  Pa — the B2 quantity), ``mean_lnps``, ``mean_T`` (sum_k dsigma_k global
  mean of T_k), ``ke`` (global mean of sum_k dsigma_k |V_k|^2 / 2, the
  reference's ``ke_mass_weighted``), ``max_abs_u``, ``t_min``, ``t_max``,
  ``comp_mode`` (|X^n - X^{n-1}_f| / |X^n| over the zeta and delta rows,
  the leapfrog computational-mode monitor).

The plan's checkpoint schema lists sums, sums of squares and cross sums:
the eddy products ``[T'^2], [u'^2], [v'T'], [u'v']`` are the per-day
(cross) second moments about the instantaneous zonal mean, their block
sums are the cross sums; ``u^2`` and ``T^2`` of the zonal means are the
sums of squares (temporal variance maps, reported, not scored).

Every sample is appended to the daily scalar series (day 0 = the initial
state). Samples on days ``spinup < day <= spinup + n_blocks * block_days``
are also added, in day order, to the sums of their 200-day block:
per-block sums of every zonal field and of the KE spectrum, sums of squares
of ``u`` and ``T`` (temporal variance maps), and the block counts. The
summation order is fixed (one addition per day), so the accumulators of an
interrupted and resumed run are bit-identical to an uninterrupted one.
"""
from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["ZONAL_KEYS", "SQUARE_KEYS", "SCALAR_KEYS", "OnlineStatistics"]

ZONAL_KEYS = ("u", "v", "T", "TsTs", "usus", "vsTs", "usvs")
SQUARE_KEYS = ("u", "T")
SCALAR_KEYS = ("day", "mean_ps", "mean_lnps", "mean_T", "ke", "max_abs_u",
               "t_min", "t_max", "comp_mode")


class OnlineStatistics:
    """Block accumulators + daily scalar series; serializable to arrays."""

    def __init__(self, *, nlat: int, nlev: int, n_degrees: int, spinup_days: int,
                 block_days: int, n_blocks: int):
        self.nlat, self.nlev, self.n_degrees = int(nlat), int(nlev), int(n_degrees)
        self.spinup_days = int(spinup_days)
        self.block_days = int(block_days)
        self.n_blocks = int(n_blocks)
        nb = max(self.n_blocks, 0)
        self.sums = {k: np.zeros((nb, self.nlat, self.nlev)) for k in ZONAL_KEYS}
        self.squares = {k: np.zeros((nb, self.nlat, self.nlev)) for k in SQUARE_KEYS}
        self.spectrum = np.zeros((nb, self.n_degrees))
        self.counts = np.zeros(nb, dtype=np.int64)
        self.series = {k: [] for k in SCALAR_KEYS}

    @classmethod
    def for_config(cls, cfg) -> "OnlineStatistics":
        return cls(nlat=cfg.nlat, nlev=cfg.nlev, n_degrees=cfg.l_max + 1,
                   spinup_days=cfg.spinup_days, block_days=cfg.block_days,
                   n_blocks=cfg.n_blocks)

    # ------------------------------------------------------------------
    def block_of(self, day: int) -> int | None:
        """Block index of a sample taken at the end of ``day`` (None outside)."""
        if day <= self.spinup_days:
            return None
        b = (day - self.spinup_days - 1) // self.block_days
        return b if b < self.n_blocks else None

    @property
    def last_day(self) -> int | None:
        return int(self.series["day"][-1]) if self.series["day"] else None

    def add(self, day: int, sample: dict[str, Any]) -> None:
        day = int(day)
        last = self.last_day
        if last is not None and day != last + 1:
            raise ValueError(f"daily samples must be consecutive: got day {day} after {last}")
        if last is None and day != 0:
            raise ValueError("the first sample must be day 0 (the initial state)")
        for k in SCALAR_KEYS:
            self.series[k].append(float(day) if k == "day" else float(sample[k]))
        b = self.block_of(day)
        if b is None:
            return
        for k in ZONAL_KEYS:
            a = np.asarray(sample[k], dtype=np.float64)
            if a.shape != (self.nlat, self.nlev):
                raise ValueError(f"{k} has shape {a.shape}, expected {(self.nlat, self.nlev)}")
            self.sums[k][b] += a
        for k in SQUARE_KEYS:
            a = np.asarray(sample[k], dtype=np.float64)
            self.squares[k][b] += a * a
        self.spectrum[b] += np.asarray(sample["ke_spectrum"], dtype=np.float64)
        self.counts[b] += 1

    # ------------------------------------------------------------------
    def block_means(self) -> dict[str, np.ndarray]:
        """Per-block time means (NaN for empty blocks)."""
        with np.errstate(invalid="ignore", divide="ignore"):
            c = self.counts.astype(np.float64)
            out = {k: v / c[:, None, None] for k, v in self.sums.items()}
            out.update({f"{k}_sq": v / c[:, None, None] for k, v in self.squares.items()})
            out["ke_spectrum"] = self.spectrum / c[:, None]
        return out

    def series_arrays(self) -> dict[str, np.ndarray]:
        return {k: np.asarray(v, dtype=np.float64) for k, v in self.series.items()}

    def to_arrays(self) -> dict[str, np.ndarray]:
        """Everything as named arrays (for the checkpoint)."""
        out = {f"stats_sum_{k}": v for k, v in self.sums.items()}
        out.update({f"stats_sq_{k}": v for k, v in self.squares.items()})
        out["stats_spectrum"] = self.spectrum
        out["stats_counts"] = self.counts
        out.update({f"series_{k}": v for k, v in self.series_arrays().items()})
        return out

    def meta(self) -> dict[str, Any]:
        return {"nlat": self.nlat, "nlev": self.nlev, "n_degrees": self.n_degrees,
                "spinup_days": self.spinup_days, "block_days": self.block_days,
                "n_blocks": self.n_blocks}

    @classmethod
    def from_arrays(cls, meta: dict[str, Any], arrays: dict[str, np.ndarray]
                    ) -> "OnlineStatistics":
        st = cls(**meta)
        for k in ZONAL_KEYS:
            st.sums[k] = np.array(arrays[f"stats_sum_{k}"], dtype=np.float64)
        for k in SQUARE_KEYS:
            st.squares[k] = np.array(arrays[f"stats_sq_{k}"], dtype=np.float64)
        st.spectrum = np.array(arrays["stats_spectrum"], dtype=np.float64)
        st.counts = np.array(arrays["stats_counts"], dtype=np.int64)
        st.series = {k: [float(x) for x in arrays[f"series_{k}"]] for k in SCALAR_KEYS}
        return st
