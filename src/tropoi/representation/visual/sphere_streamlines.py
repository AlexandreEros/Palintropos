"""Instantaneous horizontal streamlines of a wind field on the sphere (CPU).

A streamline is a curve everywhere tangent to the horizontal wind
``V = u e_east + v e_north`` of ONE saved state on ONE surface. It is not a
trajectory (the field is frozen in time) and not a 3-D streamline (there is
no vertical component). Integration uses the arc length s on the unit sphere
as parameter,

    dx/ds = V_t(x) / |V_t(x)|,    |x| = 1,

with ``x`` the Cartesian position and ``V_t`` the wind's tangent vector. In
latitude-longitude form this is ``dlat/ds = v/|V|``,
``dlon/ds = u / (|V| cos lat)``: the 1/cos(lat) metric factor is exact
and the Cartesian form never evaluates it, so the integration is regular
across the poles and periodic in longitude by construction. Each step is a
classical RK4 step in R^3 followed by renormalization onto the sphere.

The wind is interpolated bilinearly (in latitude and periodic longitude) in
its Cartesian components from a regular node mesh that includes both poles;
on a pole row every node carries the same Cartesian vector (the mean of the
row), so the interpolant is continuous there.

Bounded by construction: a fixed number of deterministic seeds (a Fibonacci
lattice), at most ``max_steps`` steps of ``step_deg`` each way. Integration
stops where the speed falls below ``min_speed`` (absolute, m/s) or below
``relative_min_speed`` times the field's maximum.

NumPy only.
"""
from __future__ import annotations

import math

import numpy as np

__all__ = ["cartesian_wind", "fibonacci_seeds", "integrate_streamlines"]


def fibonacci_seeds(count: int) -> tuple[np.ndarray, np.ndarray]:
    """``count`` quasi-uniform points on the sphere: (lat_deg, lon_deg)."""
    count = int(count)
    if count < 1:
        raise ValueError("count must be >= 1")
    i = np.arange(count, dtype=np.float64) + 0.5
    lat = np.degrees(np.arcsin(1.0 - 2.0 * i / count))
    golden = math.pi * (3.0 - math.sqrt(5.0))
    lon = np.degrees((golden * i) % (2.0 * math.pi)) - 180.0
    return lat, lon


def _to_xyz(lat_deg, lon_deg) -> np.ndarray:
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    c = np.cos(lat)
    return np.stack([c * np.sin(lon), np.sin(lat), c * np.cos(lon)], axis=-1)


def _to_latlon(xyz) -> tuple[np.ndarray, np.ndarray]:
    lat = np.degrees(np.arcsin(np.clip(xyz[..., 1], -1.0, 1.0)))
    lon = np.degrees(np.arctan2(xyz[..., 0], xyz[..., 2]))
    return lat, lon


def cartesian_wind(u, v, lat_deg, lon_deg) -> np.ndarray:
    """``(nlat, nlon, 3)`` Cartesian wind on the mesh (x = cos lat sin lon,
    y = sin lat, z = cos lat cos lon); pole rows averaged to one vector."""
    lat = np.radians(np.asarray(lat_deg, dtype=np.float64))[:, None]
    lon = np.radians(np.asarray(lon_deg, dtype=np.float64))[None, :]
    east = np.stack(np.broadcast_arrays(np.cos(lon), 0.0 * lat,
                                        -np.sin(lon)), axis=-1)
    north = np.stack(np.broadcast_arrays(-np.sin(lat) * np.sin(lon),
                                         np.cos(lat) + 0.0 * lon,
                                         -np.sin(lat) * np.cos(lon)), axis=-1)
    w = np.asarray(u)[..., None] * east + np.asarray(v)[..., None] * north
    lat_d = np.asarray(lat_deg, dtype=np.float64)
    seam = np.isclose(np.asarray(lon_deg)[-1] - np.asarray(lon_deg)[0], 360.0)
    cols = slice(0, w.shape[1] - 1) if seam else slice(None)
    for row in np.flatnonzero(np.isclose(np.abs(lat_d), 90.0)):
        w[row] = w[row, cols].mean(axis=0)
    return w


class _MeshSampler:
    def __init__(self, field, lat_deg, lon_deg):
        lat = np.asarray(lat_deg, dtype=np.float64)
        lon = np.asarray(lon_deg, dtype=np.float64)
        if not (np.isclose(lat[0], -90.0) and np.isclose(lat[-1], 90.0)):
            raise ValueError("the mesh must include both poles")
        if not np.isclose(lon[-1] - lon[0], 360.0):
            raise ValueError("the mesh must repeat the seam column")
        self.field = field
        self.lat0, self.dlat = lat[0], lat[1] - lat[0]
        self.lon0, self.dlon = lon[0], lon[1] - lon[0]
        self.nlat, self.nlon = lat.size, lon.size

    def __call__(self, lat_deg, lon_deg) -> np.ndarray:
        y = np.clip((lat_deg - self.lat0) / self.dlat, 0.0,
                    self.nlat - 1.000001)
        x = np.mod(lon_deg - self.lon0, 360.0) / self.dlon
        x = np.clip(x, 0.0, self.nlon - 1.000001)
        j = np.floor(y).astype(int)
        i = np.floor(x).astype(int)
        fy = (y - j)[..., None]
        fx = (x - i)[..., None]
        f = self.field
        return ((f[j, i] * (1 - fx) + f[j, i + 1] * fx) * (1 - fy)
                + (f[j + 1, i] * (1 - fx) + f[j + 1, i + 1] * fx) * fy)


def integrate_streamlines(u, v, lat_deg, lon_deg, *, seeds=None,
                          seed_count: int = 160, step_deg: float = 1.5,
                          max_steps: int = 24, min_speed: float = 1.0e-3,
                          relative_min_speed: float = 0.02
                          ) -> list[tuple[np.ndarray, np.ndarray]]:
    """Streamlines through each seed, ordered downstream.

    ``u``, ``v`` are ``(nlat, nlon)`` on the mesh ``lat_deg`` (``-90 .. 90``)
    x ``lon_deg`` (seam repeated). ``seeds`` = (lat_deg, lon_deg) arrays
    (default: ``seed_count`` Fibonacci points). Returns one ``(lat, lon)``
    pair of arrays per drawn line (at least 3 points), each listed from its
    upstream end to its downstream end; seeds in calm air give no line.
    """
    wind = cartesian_wind(u, v, lat_deg, lon_deg)
    speed = np.linalg.norm(wind, axis=-1)
    vmax = float(speed.max()) if speed.size else 0.0
    threshold = max(float(min_speed), float(relative_min_speed) * vmax)
    if not vmax > threshold:
        return []
    sample = _MeshSampler(wind, lat_deg, lon_deg)
    if seeds is None:
        seeds = fibonacci_seeds(seed_count)
    start = _to_xyz(np.asarray(seeds[0], float), np.asarray(seeds[1], float))
    h = math.radians(float(step_deg))

    def direction(x):
        lat, lon = _to_latlon(x)
        w = sample(lat, lon)
        w = w - np.sum(w * x, axis=-1, keepdims=True) * x
        s = np.linalg.norm(w, axis=-1, keepdims=True)
        ok = s[..., 0] > threshold
        return np.where(ok[..., None], w / np.where(s > 0, s, 1.0), 0.0), ok

    def trace(sign: float) -> tuple[np.ndarray, np.ndarray]:
        points = [start]
        alive = direction(start)[1]
        valid = [alive.copy()]
        x = start
        for _ in range(int(max_steps)):
            k1, a1 = direction(x)
            k2, a2 = direction(_unit(x + 0.5 * sign * h * k1))
            k3, a3 = direction(_unit(x + 0.5 * sign * h * k2))
            k4, a4 = direction(_unit(x + sign * h * k3))
            alive = alive & a1 & a2 & a3 & a4
            x = np.where(alive[:, None],
                         _unit(x + sign * h / 6.0 * (k1 + 2 * k2 + 2 * k3
                                                     + k4)), x)
            points.append(x)
            valid.append(alive.copy())
            if not alive.any():
                break
        return np.stack(points, axis=1), np.stack(valid, axis=1)

    fwd, fwd_ok = trace(1.0)
    bwd, bwd_ok = trace(-1.0)
    lines = []
    for n in range(start.shape[0]):
        if not fwd_ok[n, 0]:
            continue
        f = fwd[n, fwd_ok[n]]
        b = bwd[n, bwd_ok[n]][1:][::-1]
        pts = np.concatenate([b, f], axis=0)
        if pts.shape[0] < 3:
            continue
        lat, lon = _to_latlon(pts)
        lines.append((lat, lon))
    return lines


def _unit(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True)
