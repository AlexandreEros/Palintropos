"""Regime and vertical-grid evidence for the planetary presets (CPU only).

Writes ``docs/radiation/evidence/regimes.json`` with three measurements:

``regimes``     per planet: gas constants, scale height, model depth over
                radius, gravity change across the model depth, spread of
                c_p(T) over the stated temperature range, Lamb-wave speed.

``grids``       per planet: the preset's stretched grid against a uniform
                grid of the same size — pressure of the top full level,
                layers per pressure scale height, the error of the
                Simmons-Burridge hydrostatic integration for a dry adiabat,
                and the gravity-wave speeds of the semi-implicit reference
                operator on that grid (it refuses a grid on which they are
                not real).

``resolution``  a 100-bar hydrogen column with a gray optical depth
                proportional to p^2 (the Schneider & Liu law continued
                downward) in radiative-convective equilibrium on uniform and
                stretched grids of increasing size: outgoing flux,
                temperature at 1 bar and tropopause pressure. This is the
                measurement behind "choose the vertical grid per regime".

Run from the repository root:

    python tools/radiation/regimes.py
"""
from __future__ import annotations

import json
import math
import pathlib

import numpy as np

from tropoi.physics.column_model import ColumnPhysics, RadiativeConvectiveColumn
from tropoi.physics.convection import DryConvectiveAdjustment
from tropoi.physics.insolation import PrescribedInsolation
from tropoi.physics.planets import PLANETS
from tropoi.physics.radiation.gray import (OpticalDepthProfile,
                                           SemiInfiniteScatteringShortwave,
                                           schneider_liu_longwave)
from tropoi.physics.radiation.scheme import Radiation
from tropoi.spatial.sigma_coordinate import SigmaGrid, hydrostatic_geopotential
from tropoi.temporal.semi_implicit import SemiImplicitOperator

OUT = pathlib.Path(__file__).resolve().parents[2] / "docs" / "radiation" / "evidence"


def hydrostatic_error(grid: SigmaGrid, kappa: float) -> float:
    """Largest relative error of the interface geopotential for the dry
    adiabat ``T = T_s sigma^kappa`` (analytic: ``Phi = c_p T_s (1 -
    sigma^kappa)``), with layer temperatures taken at the full levels."""
    sig_full = grid.full_levels_array()
    t = sig_full ** kappa                         # T_s = 1, R = 1, c_p = 1 / kappa
    _, below = hydrostatic_geopotential(grid, t, 0.0, 1.0)
    inter = grid.interfaces_array()[1:-1]         # interfaces between layers
    exact = (1.0 - inter ** kappa) / kappa
    return float(np.max(np.abs(np.asarray(below)[:-1] - exact) / exact))


def grid_report(body) -> dict:
    spec = body.vertical_grid
    out = {}
    for name, grid in (("stretched", SigmaGrid.stretched(spec.nlev, spec.sigma_top,
                                                         spec.linear_scale)),
                       ("uniform", SigmaGrid.uniform(spec.nlev))):
        s = grid.interfaces_array()
        dln = np.diff(np.log(s[1:]))
        op = SemiImplicitOperator(grid, l_max=21, radius=body.radius, r_dry=body.gas_constant,
                                  cp_dry=body.cp, t_ref=body.temperature_range[1])
        speeds = op.gravity_wave_speeds()
        out[name] = {
            "nlev": grid.nlev,
            "top_full_level_pressure_pa": float(grid.full_levels[0] * body.reference_pressure),
            "largest_layer_in_scale_heights": float(dln.max()),
            "layers_above_one_percent_of_bottom_pressure": int((s[1:] <= 0.01).sum()),
            "thickest_layer_sigma": float(np.diff(s).max()),
            "hydrostatic_error_dry_adiabat": hydrostatic_error(grid, body.kappa),
            "fastest_gravity_wave_m_s": float(speeds[0]),
            "slowest_gravity_wave_m_s": float(speeds[-1]),
        }
    return out


def deep_column(grid: SigmaGrid) -> dict:
    """Radiative-convective equilibrium of a 100-bar hydrogen column."""
    g, r, cp, p_bottom = 26.0, 3605.38, 12619.0, 1.0e7
    rad = Radiation(schneider_liu_longwave(), SemiInfiniteScatteringShortwave(
        OpticalDepthProfile([(1.0, 1.0)], equator=3.0, reference=3.0e5), ssa=0.8, asym=0.204))
    physics = ColumnPhysics(rad, gravity=g, cp=cp, gas_constant=r, internal_heat_flux=5.7,
                            convection=DryConvectiveAdjustment(r / cp),
                            insolation=PrescribedInsolation(lambda lat: 50.7 / 4 + 0 * lat,
                                                            "global_mean", {}))
    p = (grid.interfaces_array() * p_bottom)[:, None]
    mid = 0.5 * (p[1:, 0] + p[:-1, 0])
    start = np.maximum(165.0 * (mid / 1.0e5) ** (r / cp), 110.0)[:, None]
    col = RadiativeConvectiveColumn(physics, p, start, latitude=np.zeros(1),
                                    longitude=np.zeros(1))
    res = col.equilibrate(tolerance=2e-3, max_iterations=150000)
    t = col.temperature[:, 0]
    out = col.evaluate()
    return {"nlev": grid.nlev, "converged": bool(res["converged"]),
            "iterations": int(res["iterations"]), "olr": float(out.fluxes.olr[0]),
            "absorbed_plus_interior": float(out.fluxes.absorbed_stellar[0]) + 5.7,
            "temperature_1bar": float(np.interp(math.log(1.0e5), np.log(mid), t)),
            "tropopause_pressure_pa": float(mid[int(np.argmin(t))]),
            "top_full_level_pressure_pa": float(mid[0]),
            "levels_above_1bar": int((mid < 1.0e5).sum())}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    doc = {"regimes": {n: b.regime() for n, b in PLANETS.items()},
           "grids": {n: grid_report(b) for n, b in PLANETS.items()},
           "resolution": {}}
    for label, grids in (("uniform", [SigmaGrid.uniform(n) for n in (30, 60, 120)]),
                         ("stretched", [SigmaGrid.stretched(n, 1.0e-4) for n in (24, 48, 96)])):
        doc["resolution"][label] = []
        for grid in grids:
            row = deep_column(grid)
            doc["resolution"][label].append(row)
            print(label, row, flush=True)
    (OUT / "regimes.json").write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n",
                                      encoding="utf-8")
    for name, reg in doc["regimes"].items():
        print(name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in reg.items()})
    for name, g in doc["grids"].items():
        print(name, json.dumps(g))


if __name__ == "__main__":
    main()
