"""Evaluate catalogue quantities from one saved run, once each.

:class:`RunFields` is the single place where stored coefficients become
physical numbers for plotting. It works on two samplings:

* the run's **state grid**, with its quadrature weights: every statistic
  (area means, maxima, potential enstrophy) is computed here, never on an
  interpolated view;
* a **display grid** for drawing: the state grid itself for Gauss
  latitude-longitude runs (exact samples, no interpolation), or the shared
  91 x 181 view grid for geodesic runs (``grid_adapter.map_to_uniform_latlon``).

Every synthesized field is cached per ``(quantity, snapshot, level)``, so a
background, a contour layer and a statistic that need the same field share
one synthesis, and the model is reconstructed at most once per capsule
(``storage.resources()``). Spectral-complexity measures and diagnostic
series are host-only and never touch the model.

Future vertical slices (profiles, sections, zonal means) are further
samplings of the same per-level state-grid fields; they belong here as new
methods, not in the renderer.
"""
from __future__ import annotations

from dataclasses import dataclass
import pathlib

import numpy as np

from tropoi.representation.visual.fields import ScalarGridField
from tropoi.representation.visual.quantities import (
    QuantityUnavailableError, quantity)

__all__ = ["DisplayVectors", "REST_SPEED_MS", "RunFields", "SeriesData"]

#: Winds whose largest (or root-mean-square) speed is below this are treated
#: as rest: roundoff in a state at rest reaches ~1e-15 m/s, and every flow
#: these cores are used for is many orders of magnitude faster. Such winds
#: are not drawn and have no kinetic-energy spectrum worth describing.
REST_SPEED_MS = 1.0e-9


def _host(values) -> np.ndarray:
    if hasattr(values, "get"):
        values = values.get()
    return np.asarray(values)


@dataclass(frozen=True)
class DisplayVectors:
    latitudes: np.ndarray
    longitudes: np.ndarray
    zonal: np.ndarray
    meridional: np.ndarray
    radius: float


@dataclass(frozen=True)
class SeriesData:
    """A diagnostic series with its true sampling cadence."""

    times_seconds: np.ndarray
    values: np.ndarray
    cadence: str          # "every step" or "saved snapshots"
    label: str


class RunFields:
    """Cached physical fields of one saved run (see module docstring)."""

    def __init__(self, storage) -> None:
        self._storage = storage
        self._cache: dict[tuple, object] = {}
        self._csv = None
        #: Number of spherical-harmonic syntheses performed (tests use it to
        #: prove that shared fields are synthesized once).
        self.synthesis_count = 0

    # -- run facts (host) -------------------------------------------------

    @property
    def storage(self):
        return self._storage

    @property
    def solver(self) -> str:
        return self._storage.solver

    @property
    def times(self) -> np.ndarray:
        return np.asarray(self._storage.times, dtype=np.float64)

    @property
    def run_config(self) -> dict:
        return dict(self._storage.run_config)

    @property
    def nlev(self) -> int | None:
        levels = self._storage.level_values
        return None if levels is None else len(levels)

    @property
    def sigma_levels(self) -> tuple[float, ...] | None:
        return self._storage.level_values

    @property
    def has_terrain(self) -> bool:
        return self.solver in ("swe", "pe") and self.run_config.get(
            "topography", "flat") not in (None, "flat")

    def _frame(self, index: int) -> np.ndarray:
        return np.asarray(self._storage.frame(int(index)))

    def _check(self, identifier: str, level):
        entry = quantity(identifier)
        entry.check_solver(self.solver)
        if self.solver == "pe" and entry.per_level_in_pe:
            if level is None:
                raise QuantityUnavailableError(
                    f"{identifier!r} varies with height in PE runs; choose a "
                    f"level (0..{self.nlev - 1}, top to bottom)")
            if not 0 <= int(level) < self.nlev:
                raise QuantityUnavailableError(
                    f"level {level} is outside 0..{self.nlev - 1}")
        elif level is not None:
            raise QuantityUnavailableError(
                f"{identifier!r} has no vertical level in "
                f"{self.solver.upper()} runs")
        return entry

    # -- model resources (CUDA) ---------------------------------------------

    def _resources(self):
        return self._storage.resources()

    def _model(self):
        resources = self._resources()
        return resources.model if resources.model is not None else None

    def _transform(self):
        return self._resources().planet.sh

    @property
    def radius(self) -> float:
        """The model's dynamical radius (needs the model)."""
        model = self._model()
        if model is not None:
            return float(model.R)
        return float(self._resources().planet.params.radius)

    def _synthesize(self, coefficients) -> np.ndarray:
        import cupy as cp
        self.synthesis_count += 1
        return _host(self._transform().inv_transform(
            cp.asarray(np.ascontiguousarray(coefficients)))).real

    def _grid(self):
        return self._resources().planet.grid

    def state_weights(self) -> np.ndarray:
        """Normalized state-grid area weights (sum 1)."""
        key = ("weights",)
        if key not in self._cache:
            weights = _host(self._transform().weights).astype(np.float64)
            self._cache[key] = weights / weights.sum()
        return self._cache[key]

    # -- state-grid fields --------------------------------------------------

    def _inverse_laplacian(self, coefficients) -> np.ndarray:
        coefficients = np.asarray(coefficients)
        degree = np.arange(coefficients.shape[0], dtype=np.float64)
        factor = np.zeros_like(degree)
        factor[1:] = -self.radius ** 2 / (degree[1:] * (degree[1:] + 1.0))
        return coefficients * factor[:, None]

    def state_values(self, identifier: str, index: int,
                     level: int | None = None) -> np.ndarray:
        """A scalar quantity on the state grid (flat host array)."""
        self._check(identifier, level)
        key = ("state", identifier,
               None if index is None else int(index), level)
        if key in self._cache:
            return self._cache[key]
        solver, frame = self.solver, None
        k = None if level is None else int(level)
        if identifier == "terrain":
            values = self._terrain_state()
        elif identifier == "wind_speed":
            u, v = self.state_wind(index, level)
            values = np.hypot(u, v)
        else:
            frame = self._frame(index)
            K = self.nlev
            rows = {
                ("bve", "vorticity"): lambda: frame,
                ("swe", "vorticity"): lambda: frame[0],
                ("swe", "divergence"): lambda: frame[1],
                ("pe", "vorticity"): lambda: frame[k],
                ("pe", "divergence"): lambda: frame[K + k],
                ("pe", "temperature"): lambda: frame[2 * K + k],
            }
            if (solver, identifier) in rows:
                values = self._synthesize(rows[(solver, identifier)]())
            elif identifier == "streamfunction":
                zeta = frame if solver == "bve" else (
                    frame[0] if solver == "swe" else frame[k])
                values = self._synthesize(self._inverse_laplacian(zeta))
            elif identifier == "velocity_potential":
                delta = frame[1] if solver == "swe" else frame[K + k]
                values = self._synthesize(self._inverse_laplacian(delta))
            elif identifier == "layer_depth":
                model = self._model()
                values = (model.phi0 + self._synthesize(frame[2])) / (
                    model.gravity)
            elif identifier == "free_surface_height":
                depth = self.state_values("layer_depth", index)
                values = depth + (self._terrain_state() if self.has_terrain
                                  else 0.0)
            elif identifier == "temperature_anomaly":
                row = np.array(frame[2 * K + k], copy=True)
                row[0, 0] = 0.0     # the (0,0) mode is the exact area mean
                values = self._synthesize(row)
            elif identifier == "surface_pressure_anomaly":
                pressure = np.exp(self._synthesize(frame[3 * K]))
                weights = self.state_weights()
                values = (pressure - float(np.sum(weights * pressure))) / 100.0
            else:  # pragma: no cover - catalogue and evaluator disagree
                raise QuantityUnavailableError(
                    f"no evaluation rule for {identifier!r}")
        self._cache[key] = values
        return values

    def _terrain_state(self) -> np.ndarray:
        key = ("terrain",)
        if key in self._cache:
            return self._cache[key]
        if not self.has_terrain:
            raise QuantityUnavailableError(
                "terrain is not defined: this run has a flat bottom")
        model = self._model()
        if self.solver == "swe":
            values = _host(model.surface_geopotential_on_state_grid()) / (
                model.gravity)
        else:
            values = self._synthesize(_host(model.phi_surface_lm)) / float(
                self.run_config["gravity"])
        self._cache[key] = values
        return values

    def state_wind(self, index: int, level: int | None = None
                   ) -> tuple[np.ndarray, np.ndarray]:
        """(u, v) in m/s on the state grid."""
        self._check("wind", level)
        key = ("wind", int(index), level)
        if key in self._cache:
            return self._cache[key]
        import cupy as cp
        frame = self._frame(index)
        if self.solver == "bve":
            operators = self._resources().planet.so
            psi = operators.inv_laplacian(cp.asarray(frame))
            u, v = operators.velocity_from_streamfunction(psi)
            self.synthesis_count += 2
            wind = (_host(u).real, _host(v).real)
        elif self.solver == "swe":
            from tropoi.temporal.tendencies.shallow_water import (
                ShallowWaterState)
            u, v = self._model().wind_on_state_grid(
                ShallowWaterState(cp.asarray(frame)))
            self.synthesis_count += 2
            wind = (_host(u), _host(v))
        else:
            from tropoi.spatial.states.primitive_equations import (
                PrimitiveEquationsState)
            u, v = self._model().wind_on_state_grid(
                PrimitiveEquationsState(cp.asarray(frame)))
            self.synthesis_count += 2 * self.nlev
            u, v = _host(u), _host(v)
            for k in range(self.nlev):
                self._cache[("wind", int(index), k)] = (u[k], v[k])
            wind = (u[int(level)], v[int(level)])
        self._cache[key] = wind
        return wind

    # -- display sampling ---------------------------------------------------

    def _display(self, flat_values) -> tuple[np.ndarray, np.ndarray,
                                             np.ndarray]:
        from tropoi.spatial.grids.latlon_grid import GaussLatLonGridGeometry
        grid = self._grid()
        if isinstance(grid, GaussLatLonGridGeometry):
            return (_host(grid.latitudes), _host(grid.longitudes),
                    np.asarray(flat_values).reshape(grid.nlat, grid.nlon))
        from tropoi.representation.visual.grid_adapter import (
            map_to_uniform_latlon)
        target = self._cache.get(("view-grid",))
        target, mapped = map_to_uniform_latlon(flat_values, grid,
                                               target_grid=target)
        self._cache[("view-grid",)] = target
        return _host(target.latitudes), _host(target.longitudes), mapped

    @property
    def display_is_native(self) -> bool:
        from tropoi.spatial.grids.latlon_grid import GaussLatLonGridGeometry
        return isinstance(self._grid(), GaussLatLonGridGeometry)

    def scalar_field(self, identifier: str, index: int | None,
                     level: int | None = None, *,
                     name: str | None = None) -> ScalarGridField:
        """A scalar on the display grid (``index=None`` for static fields)."""
        entry = quantity(identifier)
        values = (self._terrain_state() if identifier == "terrain"
                  else self.state_values(identifier, index, level))
        lat, lon, grid_values = self._display(values)
        times = None if index is None else self.times[int(index):int(index) + 1]
        return ScalarGridField(grid_values, lat, lon,
                               name=name or entry.long_name,
                               units=entry.units, times=times)

    def display_wind(self, index: int, level: int | None = None
                     ) -> DisplayVectors:
        u, v = self.state_wind(index, level)
        lat, lon, u_view = self._display(u)
        _, _, v_view = self._display(v)
        return DisplayVectors(lat, lon, u_view, v_view, self.radius)

    # -- numbers at saved snapshots -----------------------------------------

    def area_mean(self, identifier: str, index: int,
                  level: int | None = None) -> float:
        values = self.state_values(identifier, index, level)
        return float(np.sum(self.state_weights() * values))

    def potential_enstrophy(self, index: int) -> float:
        self._check("potential_enstrophy", None)
        key = ("potential_enstrophy", int(index))
        if key not in self._cache:
            import cupy as cp
            from tropoi.representation.diagnostics.swe import (
                potential_enstrophy)
            from tropoi.temporal.tendencies.shallow_water import (
                ShallowWaterState)
            self._cache[key] = float(potential_enstrophy(
                self._model(), ShallowWaterState(
                    cp.asarray(self._frame(index)))))
        return self._cache[key]

    @property
    def host_radius_estimate(self) -> float:
        """Planet radius from the saved schema (host only; used only for
        the rest threshold, where 0.1 % does not matter)."""
        schema = self._storage.metadata.get("schema")
        geometry = schema.get("geometry", {}) if isinstance(schema, dict) else {}
        reference = float(geometry.get("reference_radius_m") or 6.371e6)
        return reference * float(geometry.get("radius_earth_units") or 1.0)

    def _energy_modes(self, index: int, level):
        from tropoi.representation.diagnostics.spectral import (
            kinetic_energy_modes)
        frame = self._frame(index)
        if self.solver == "bve":
            zeta, delta = frame, None
        elif self.solver == "swe":
            zeta, delta = frame[0], frame[1]
        else:
            zeta, delta = frame[level], frame[self.nlev + level]
        return kinetic_energy_modes(zeta, delta,
                                    radius=self.host_radius_estimate)

    def rms_speed(self, index: int, level: int | None = None) -> float:
        """Area-rms wind speed from the kinetic-energy spectrum (host)."""
        self._check("spectral_complexity", level)
        radius = self.host_radius_estimate
        energy = float(self._energy_modes(index, level).sum())
        return float(np.sqrt(2.0 * energy / (4.0 * np.pi * radius ** 2)))

    def spectral_complexity(self, index: int, level: int | None = None):
        """Host-only KE complexity measures (radius-independent).

        Raises ``ZeroKineticEnergyError`` for a state at rest, including
        roundoff-level motion (rms speed below :data:`REST_SPEED_MS`).
        """
        from tropoi.representation.diagnostics.spectral import (
            ZeroKineticEnergyError, spectral_complexity)
        self._check("spectral_complexity", level)
        if self.rms_speed(index, level) < REST_SPEED_MS:
            raise ZeroKineticEnergyError(
                f"no resolved motion (rms speed below {REST_SPEED_MS:g} m/s)")
        return spectral_complexity(self._energy_modes(index, level))

    # -- recorded diagnostics -----------------------------------------------

    @property
    def diagnostics_path(self) -> pathlib.Path:
        return pathlib.Path(self._storage.run_dir) / "diagnostics" / (
            "timeseries.csv")

    def _table(self):
        if self._csv is None:
            path = self.diagnostics_path
            if not path.is_file():
                raise QuantityUnavailableError(
                    f"this run has no per-step diagnostics ({path} not found)")
            table = np.genfromtxt(path, delimiter=",", names=True)
            self._csv = np.atleast_1d(table)
        return self._csv

    def recorded(self, column: str) -> SeriesData:
        table = self._table()
        if column not in table.dtype.names:
            raise QuantityUnavailableError(
                f"diagnostics/timeseries.csv has no column {column!r}; "
                f"columns: {', '.join(table.dtype.names)}")
        values = np.asarray(table[column], dtype=np.float64)
        times = np.asarray(table["time_s"], dtype=np.float64)
        finite = np.isfinite(values)
        return SeriesData(times[finite], values[finite], "every step", column)

    def recorded_at_snapshots(self, column: str) -> tuple[np.ndarray, list]:
        """Values of a recorded column at the rows whose time equals each
        saved snapshot time exactly; ``None`` where no row matches."""
        series = self.recorded(column)
        out, missing = [], []
        for index, target in enumerate(self.times):
            rows = np.flatnonzero(series.times_seconds == target)
            if rows.size:
                out.append(float(series.values[rows[0]]))
            else:
                out.append(None)
                missing.append(index)
        return np.array([np.nan if v is None else v for v in out]), missing
