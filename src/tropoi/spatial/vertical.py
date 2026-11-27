"""Vertical meshes of conductive columns: cell faces (m, positive down from
0) built geometrically from a top cell.

Spatial discretization, not physics. The conductive substrate receives its
faces explicitly (``ConductiveSubstrate(strata, interfaces=...)``); the
experiment that configures it builds them here. Every generated grid is
checked on return: adjacent thickness ratios lie in [1, 1.5] to 1e-9.

Import-light: standard library and NumPy only.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np

__all__ = ["geometric_interfaces", "stratified_interfaces"]

_MAX_STRETCH = 1.5
_RATIO_TOL = 1e-9          # round-off allowance on the [1, 1.5] ratio guarantee


def geometric_interfaces(top_thickness: float, depth: float, *,
                         layers: Optional[int] = None,
                         stretch: Optional[float] = None,
                         depth_is_minimum: bool = False) -> tuple:
    """Layer interfaces (m, from 0) with thicknesses growing geometrically
    from ``top_thickness``.

    Give exactly one of ``layers`` and ``stretch``. Both keep
    ``top_thickness`` exact.

    * ``layers``: the ratio is solved so that ``layers`` layers end exactly
      at ``depth``. The solved ratio must lie in [1, 1.5].
    * ``stretch``: layers of thickness ``top * stretch**i`` are added while
      they fit within ``depth``. A gap g left at the bottom is handled
      deterministically:

      - if g is zero to within 1e-12 relative, the grid ends exactly at
        ``depth``;
      - otherwise two terminations are considered: merging g into the last
        layer, or keeping g as an extra layer. Each is acceptable only if its
        final thickness ratio lies in [1, 1.5]. Of the acceptable ones, the
        one whose final ratio is closest to ``stretch`` is used (merging on a
        tie). The requested ratio then holds for every layer except the last;
      - if neither is acceptable, ``ValueError`` is raised. In that case the
        requested top thickness, depth, stretch and ratio bound are
        incompatible. Use ``layers`` for an exact geometric fit, choose
        another depth, or pass ``depth_is_minimum=True``;
      - with ``depth_is_minimum=True``, ``depth`` is a lower bound: the grid
        ends at the first exact geometric interface at or below ``depth``.
        Every ratio is then exactly ``stretch``, and the column is at least
        as deep as requested.

    **Ratio bounds.** The regolith specification requires only an upper
    bound, stretch <= 1.5. This function guarantees more: every
    adjacent-layer ratio of a grid it returns lies in [1, 1.5] (to a
    round-off allowance of 1e-9, checked on return), so thickness never
    decreases with depth. ``layers`` must be an integer. Grids passed explicitly as ``interfaces`` to
    :class:`~tropoi.physics.substrate.ConductiveSubstrate`, for example
    uniform or historical fixture grids, are not subject to that guarantee."""
    top = float(top_thickness)
    total = float(depth)
    if not (math.isfinite(top) and math.isfinite(total)):
        raise ValueError("top_thickness and depth must be finite")
    if not (top > 0.0 and total > 0.0 and top <= total):
        raise ValueError("need 0 < top_thickness <= depth")
    if (layers is None) == (stretch is None):
        raise ValueError("give exactly one of layers and stretch")
    if layers is not None and depth_is_minimum:
        raise ValueError("depth_is_minimum applies to stretch, not layers")
    if layers is not None:
        if isinstance(layers, bool) or int(layers) != layers:
            raise ValueError(f"layers must be an integer, got {layers!r}")
        n = int(layers)
        if n < 1:
            raise ValueError("layers must be >= 1")
        if n == 1:
            if not math.isclose(top, total, rel_tol=1e-12):
                raise ValueError("one layer requires top_thickness == depth")
            return (0.0, total)
        if top * n > total * (1.0 + 1e-12):
            raise ValueError("top_thickness * layers exceeds depth (ratio < 1)")
        if _geometric_sum(top, _MAX_STRETCH, n) < total * (1.0 - 1e-12):
            raise ValueError(f"{n} layers from top_thickness={top} cannot reach depth={total} "
                             f"with a ratio <= {_MAX_STRETCH}")
        if math.isclose(top * n, total, rel_tol=1e-12):
            ratio = 1.0
        else:
            lo, hi = 1.0 + 1e-15, 2.0
            for _ in range(200):
                mid = 0.5 * (lo + hi)
                if _geometric_sum(top, mid, n) < total:
                    lo = mid
                else:
                    hi = mid
            ratio = 0.5 * (lo + hi)
        if ratio > _MAX_STRETCH:
            raise ValueError(f"solved stretch {ratio:.6f} exceeds {_MAX_STRETCH}")
        edges = [0.0]
        t = top
        for _ in range(n - 1):
            edges.append(edges[-1] + t)
            t *= ratio
        edges.append(total)
        return _checked(edges)
    ratio = float(stretch)
    if not (math.isfinite(ratio) and 1.0 <= ratio <= _MAX_STRETCH):
        raise ValueError(f"stretch must lie in [1, {_MAX_STRETCH}]")
    edges = [0.0]
    t = top
    while edges[-1] + t <= total * (1.0 + 1e-12):
        edges.append(edges[-1] + t)
        t *= ratio
    gap = total - edges[-1]
    if gap <= 1e-12 * total:
        edges[-1] = total
        return _checked(edges)
    if depth_is_minimum:
        edges.append(edges[-1] + t)
        return _checked(edges)
    dz = [edges[i + 1] - edges[i] for i in range(len(edges) - 1)]
    candidates = []
    if len(dz) >= 2:                      # merge into the last layer (never the top one)
        merged = (dz[-1] + gap) / dz[-2]
        if 1.0 <= merged <= _MAX_STRETCH:
            candidates.append((abs(merged - ratio), 0, "merge"))
    else:
        merged = math.nan
    appended = gap / dz[-1]
    if 1.0 <= appended <= _MAX_STRETCH:
        candidates.append((abs(appended - ratio), 1, "append"))
    if not candidates:
        raise ValueError(
            f"top_thickness={top}, depth={total}, stretch={ratio} are incompatible with "
            f"final-layer ratios in [1, {_MAX_STRETCH}]: merging the {gap:.6g} m remainder "
            f"gives {merged:.4f}, keeping it as a layer gives {appended:.4f}. Use layers= "
            "for an exact geometric fit, another depth, or depth_is_minimum=True")
    if min(candidates)[2] == "merge":
        edges[-1] = total
    else:
        edges.append(total)
    return _checked(edges)


def _geometric_sum(top: float, ratio: float, n: int) -> float:
    """top * (ratio**n - 1) / (ratio - 1). Near ratio 1 it uses expm1/log1p to
    avoid cancellation; the plain form is kept elsewhere, so the default grid's
    arithmetic is unchanged."""
    x = ratio - 1.0
    if x < 1e-3:
        return top * math.expm1(n * math.log1p(x)) / x
    return top * (ratio ** n - 1.0) / (ratio - 1.0)


def _checked(edges) -> tuple:
    """Assert the documented ratio guarantee (to 1e-9 relative) on a generated grid."""
    dz = [edges[i + 1] - edges[i] for i in range(len(edges) - 1)]
    for i in range(1, len(dz)):
        q = dz[i] / dz[i - 1]
        if not (1.0 - _RATIO_TOL <= q <= _MAX_STRETCH + _RATIO_TOL):
            raise ValueError(f"generated grid ratio {q!r} at layer {i} is outside "
                             f"[1, {_MAX_STRETCH}]; this is a bug or a degenerate request")
    return tuple(edges)


def stratified_interfaces(boundaries: Sequence[float], top_thickness: float,
                          max_stretch: float) -> tuple:
    """Cell faces from 0 to ``boundaries[-1]`` with a face at every stratum
    boundary.

    The first cell is ``top_thickness`` thick. Within each stratum the cells
    grow geometrically, using the fewest cells whose exact geometric fit has
    a ratio no larger than ``max_stretch``. Each stratum after the first
    starts with a cell as thick as the last cell above it, so thickness
    never drops across a boundary unless the stratum is thinner than that
    cell (then it is one cell)."""
    b = [float(x) for x in boundaries]
    if not b or any(x <= 0 for x in b) or any(y <= x for x, y in zip(b, b[1:])):
        raise ValueError("boundaries must be positive and increasing")
    if not 1.0 <= max_stretch <= 1.5:
        raise ValueError("max_stretch must lie in [1, 1.5]")
    edges = [0.0]
    first = float(top_thickness)
    if not math.isfinite(first) or first <= 0 or first > b[0]:
        raise ValueError("need 0 < top_thickness <= first stratum depth")
    top = 0.0
    for bottom in b:
        length = bottom - top
        if first >= length * (1.0 - 1e-12):
            seg = (0.0, length)
        else:
            n = int((length / first) * (1.0 + 1e-12))
            seg = None
            for n_try in range(1, n + 1):
                try:
                    cand = geometric_interfaces(first, length, layers=n_try)
                except ValueError:
                    continue
                dz = np.diff(cand)
                if len(dz) < 2 or np.max(dz[1:] / dz[:-1]) <= max_stretch * (1 + 1e-12):
                    seg = cand
                    break
            if seg is None:
                raise ValueError(f"cannot fit stratum ending at {bottom} m with "
                                 f"first cell {first} m and max_stretch={max_stretch}")
        edges.extend(top + x for x in seg[1:])
        edges[-1] = bottom
        first = edges[-1] - edges[-2]
        top = bottom
    return tuple(edges)
