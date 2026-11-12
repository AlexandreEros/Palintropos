"""The radiation column contract and the solver that every optics provider shares.

What the dynamical core hands over is a :class:`Column` (pressures at layer
interfaces, layer temperatures, gravity, heat capacity, latitude) and a
:class:`LowerBoundary`; what it gets back is a :class:`RadiativeFluxes`
(upward and downward fluxes at every interface for the thermal and stellar
streams, and the heating rate that is exactly their divergence). Nothing in
this package knows about sigma coordinates, spherical harmonics or CuPy
devices, so the same code serves any dynamical core, a single-column model
and the CPU tests.

Spectral structure lives in *optics providers*:

* a thermal provider returns, for a slice of its spectral points, layer
  optical depths ``tau`` (K, ncol, n) (optionally ``ssa``/``asym``) and a
  ``planck(temperature, points)`` function giving the flux-Planck source
  ``pi B`` integrated over each point (so the sum over all points of a
  provider that spans the spectrum is ``sigma T^4``);
* a stellar provider returns ``tau``, ``ssa``, ``asym`` and the fraction of
  the stellar flux carried by each point.

A gray scheme is the one-point provider; a correlated-k scheme is a provider
with (bands x g-points) points. :class:`Radiation` loops over the points in
chunks (bounded memory on a small GPU) and sums fluxes.

Sign and index conventions are those of ``twostream.py``: index 0 is the
top, fluxes are positive in their own direction, ``net = down - up``.

Import-light: NumPy only; arrays may be NumPy or CuPy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from tropoi.physics.radiation import twostream

__all__ = ["STEFAN_BOLTZMANN", "Column", "LowerBoundary", "RadiativeFluxes",
           "Radiation", "interface_temperature", "heating_rate"]

#: Stefan–Boltzmann constant (CODATA 2018, exact from the SI redefinition).
STEFAN_BOLTZMANN = 5.670374419e-8


def _xp(*arrays):
    for a in arrays:
        if type(a).__module__.split(".")[0] == "cupy":
            import cupy  # noqa: WPS433
            return cupy
    return np


@dataclass
class Column:
    """A batch of atmospheric columns, top to bottom.

    ``pressure_interfaces`` (K+1, ncol) Pa, strictly increasing downward; the
    first row may be 0 (a model top at zero pressure). ``temperature``
    (K, ncol) K, layer means. ``gravity`` m s^-2 and ``cp`` J kg^-1 K^-1 are
    scalars or arrays broadcastable to (K, ncol). ``latitude`` (ncol,)
    radians is used only by optics that prescribe a latitude dependence.
    """

    pressure_interfaces: Any
    temperature: Any
    gravity: Any
    cp: Any
    latitude: Any = None

    def __post_init__(self):
        p, t = self.pressure_interfaces, self.temperature
        if p.ndim != 2 or t.ndim != 2 or p.shape[0] != t.shape[0] + 1 or p.shape[1] != t.shape[1]:
            raise ValueError(
                "pressure_interfaces must be (K+1, ncol) and temperature (K, ncol); got "
                f"{p.shape} and {t.shape}")

    @property
    def nlay(self) -> int:
        return self.temperature.shape[0]

    @property
    def ncol(self) -> int:
        return self.temperature.shape[1]

    @property
    def layer_pressure_thickness(self):
        p = self.pressure_interfaces
        return p[1:] - p[:-1]

    @property
    def layer_pressure(self):
        """Mass-weighted mean layer pressure (arithmetic mean of the edges)."""
        p = self.pressure_interfaces
        return 0.5 * (p[1:] + p[:-1])

    @property
    def surface_pressure(self):
        return self.pressure_interfaces[-1]


@dataclass
class LowerBoundary:
    """What lies under the lowest layer.

    ``kind="surface"``: a surface at ``temperature`` (ncol,) with thermal
    ``emissivity`` and stellar ``albedo``. ``kind="closed"``: no surface (a
    giant planet's interior): stellar flux reaching the bottom is absorbed
    there and every downwelling watt, stellar and thermal, is returned upward
    as thermal radiation, so the radiation scheme neither gains nor loses
    energy through the bottom. Interior heat is not radiation; the column
    physics deposits it separately.
    """

    kind: str = "surface"
    temperature: Any = None
    emissivity: float = 1.0
    albedo: Any = 0.0

    def __post_init__(self):
        if self.kind not in ("surface", "closed"):
            raise ValueError(f"kind must be 'surface' or 'closed', got {self.kind!r}")
        if self.kind == "surface" and self.temperature is None:
            raise ValueError("a surface lower boundary needs a temperature")


@dataclass
class RadiativeFluxes:
    """Interface fluxes (K+1, ncol) in W m^-2 and layer heating (K, ncol) in K/s."""

    lw_up: Any
    lw_down: Any
    sw_up: Any
    sw_down: Any
    heating: Any
    sw_down_direct: Any = None
    extras: dict = field(default_factory=dict)

    @property
    def net_down(self):
        return self.lw_down - self.lw_up + self.sw_down - self.sw_up

    @property
    def olr(self):
        """Outgoing thermal flux at the top."""
        return self.lw_up[0]

    @property
    def absorbed_stellar(self):
        """Net stellar flux entering at the top."""
        return self.sw_down[0] - self.sw_up[0]

    @property
    def toa_net_down(self):
        return self.net_down[0]

    @property
    def bottom_net_down(self):
        return self.net_down[-1]


def heating_rate(net_down, pressure_interfaces, gravity, cp):
    """dT/dt (K/s) of each layer from the divergence of the net downward flux:
    ``g (N_k - N_{k+1}) / (c_p dp_k)``. Summed over a column with weights
    ``c_p dp / g`` it telescopes to ``N_top - N_bottom`` exactly."""
    dp = pressure_interfaces[1:] - pressure_interfaces[:-1]
    return gravity * (net_down[:-1] - net_down[1:]) / (cp * dp)


def interface_temperature(column: Column):
    """Temperatures at the K+1 interfaces for the linear-in-tau source.

    Interior interfaces: linear interpolation in pressure between the two
    adjacent layer means. Top and bottom: the adjacent layer's temperature
    (no extrapolation; a surface at a different temperature is a separate
    source, so a surface discontinuity is represented and not smeared).
    """
    xp = _xp(column.temperature)
    t = column.temperature
    p_mid = column.layer_pressure
    p_int = column.pressure_interfaces
    out = xp.empty_like(p_int)
    w = (p_int[1:-1] - p_mid[:-1]) / (p_mid[1:] - p_mid[:-1])
    out[1:-1] = t[:-1] + w * (t[1:] - t[:-1])
    out[0] = t[0]
    out[-1] = t[-1]
    return out


class Radiation:
    """Thermal + stellar radiative transfer for a batch of columns.

    ``longwave`` / ``shortwave`` are optics providers (either may be None).
    ``source`` selects the thermal source inside a layer: ``"linear"`` in
    optical depth between interpolated interface temperatures, or
    ``"isothermal"`` layers. ``chunk`` bounds how many spectral points are
    solved at once.
    """

    def __init__(self, longwave=None, shortwave=None, *, source: str = "linear",
                 chunk: int = 16):
        if source not in ("linear", "isothermal"):
            raise ValueError(f"source must be 'linear' or 'isothermal', got {source!r}")
        self.longwave = longwave
        self.shortwave = shortwave
        self.source = source
        self.chunk = int(chunk)

    def signature(self) -> dict:
        return {"kind": "radiation", "source": self.source,
                "longwave": None if self.longwave is None else self.longwave.signature(),
                "shortwave": None if self.shortwave is None else self.shortwave.signature()}

    # ------------------------------------------------------------------
    def _stellar(self, column, boundary, toa_flux, mu0):
        xp = _xp(column.temperature)
        zeros = xp.zeros_like(column.pressure_interfaces)
        sw = self.shortwave
        if sw is None or toa_flux is None:
            return zeros, zeros.copy(), zeros.copy()
        albedo = 0.0 if boundary.kind == "closed" else boundary.albedo
        analytic = getattr(sw, "analytic_fluxes", None)
        if analytic is not None:
            return analytic(column, toa_flux, mu0, albedo)
        up, down, direct = zeros.copy(), zeros.copy(), zeros.copy()
        weights = sw.stellar_fractions()
        for start in range(0, sw.n_points, self.chunk):
            pts = slice(start, min(start + self.chunk, sw.n_points))
            tau, ssa, asym = sw.optical_properties(column, pts)
            w = xp.asarray(weights[pts])
            flux = toa_flux[:, None] * w[None, :]
            mu = None if mu0 is None else mu0[:, None]
            u, d, dd = twostream.stellar_fluxes(
                tau, ssa, asym, mu, flux, surface_albedo=_points(albedo, xp),
                closure=sw.closure, scale=sw.delta_scale,
                diffuse_incidence=sw.diffuse_incidence)
            up += u.sum(axis=-1)
            down += d.sum(axis=-1)
            direct += dd.sum(axis=-1)
        return up, down, direct

    def _thermal(self, column, boundary, bottom_absorbed_stellar):
        xp = _xp(column.temperature)
        zeros = xp.zeros_like(column.pressure_interfaces)
        lw = self.longwave
        if lw is None:
            return zeros, zeros.copy()
        up, down = zeros.copy(), zeros.copy()
        t_lay = column.temperature
        t_int = interface_temperature(column) if self.source == "linear" else None
        closed = boundary.kind == "closed"
        if closed:
            # The absorbed stellar flux is re-emitted with the spectral
            # shape of the lowest layer's Planck function.
            ref = lw.planck(t_lay[-1], slice(0, lw.n_points))
            share_total = ref.sum(axis=-1)
        for start in range(0, lw.n_points, self.chunk):
            pts = slice(start, min(start + self.chunk, lw.n_points))
            tau, ssa, asym = lw.optical_properties(column, pts)
            if t_int is None:
                p_top = p_bot = lw.planck(t_lay, pts)
            else:
                p_all = lw.planck(t_int, pts)
                p_top, p_bot = p_all[:-1], p_all[1:]
            if closed:
                share = lw.planck(t_lay[-1], pts) / share_total[:, None]
                source = bottom_absorbed_stellar[:, None] * share
                albedo = 1.0
            else:
                source = boundary.emissivity * lw.planck(boundary.temperature, pts)
                albedo = 1.0 - boundary.emissivity
            u, d = twostream.thermal_fluxes(
                tau, p_top, p_bot, surface_source=source, surface_albedo=albedo,
                ssa=ssa, asym=asym, diffusivity=lw.diffusivity)
            up += u.sum(axis=-1)
            down += d.sum(axis=-1)
        return up, down

    def fluxes(self, column: Column, boundary: LowerBoundary, *,
               toa_flux=None, mu0=None) -> RadiativeFluxes:
        """Solve both streams. ``toa_flux`` (ncol,) is the stellar flux
        through a horizontal surface at the top (W m^-2), ``mu0`` (ncol,) the
        cosine of the stellar zenith angle (None for diffuse or analytic
        stellar schemes)."""
        sw_up, sw_down, sw_direct = self._stellar(column, boundary, toa_flux, mu0)
        absorbed_bottom = sw_down[-1] - sw_up[-1]
        lw_up, lw_down = self._thermal(column, boundary, absorbed_bottom)
        net = lw_down - lw_up + sw_down - sw_up
        heating = heating_rate(net, column.pressure_interfaces, column.gravity, column.cp)
        return RadiativeFluxes(lw_up=lw_up, lw_down=lw_down, sw_up=sw_up, sw_down=sw_down,
                               heating=heating, sw_down_direct=sw_direct)


def _points(value, xp):
    """Give a per-column quantity a trailing spectral axis."""
    if isinstance(value, (int, float)):
        return value
    return xp.asarray(value)[:, None]
