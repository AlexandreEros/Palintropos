"""Prepared planets: a planet, its discretization and its resolved terrain.

The step before a simulation: load terrain (from an elevation file or a
preset), resolve it on the model's discretization, and look at exactly what
the solver will use::

    from tropoi.run.custom import (DiscretizationSpec, PlanetSpec,
                                   prepare_planet, read_elevation)

    raster = read_elevation("mars_topography.nc")          # CPU
    planet = prepare_planet(PlanetSpec.oblate(3.3962e6, 88642.66),
                            DiscretizationSpec.gauss_latlon(42, 64, 128),
                            raster, smoothing=0.3)          # CUDA (projection)
    planet.plot()                                           # static figure
    planet.view()                                           # interactive globe

Resolving needs CUDA (the projection uses the model's own transform).
Previewing does not: :meth:`PreparedPlanet.load` reads a saved prepared
terrain, and :meth:`PreparedPlanet.from_initial_state` a saved shallow-water
initial state or a custom run's ``initial_state/``, on any machine.

Where the terrain goes:

* shallow water: pass the :class:`PreparedPlanet` as ``topography`` to
  :func:`~tropoi.run.custom.prepare.prepare_swe_state`; the state (and every
  run from it) keeps these exact coefficients and the source's provenance;
* primitive equations: resolve with ``solver="pe"`` (band-limited at the
  dealiased product cut the PE core requires), :meth:`PreparedPlanet.save`
  it, and run ``tropoi run pe --terrain DIR``; the capsule keeps a copy
  under ``terrain/``.

Plotting and HTML rendering live in
:mod:`tropoi.representation.visual.terrain`, imported on use.
"""
from __future__ import annotations

import hashlib
import math
import pathlib
import re
import tempfile
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from tropoi.spatial.specification import (DiscretizationSpec, PlanetSpec,
                                          TopographySpec)

__all__ = ["PreparedPlanet", "prepare_planet"]

#: Labels of the provenance / preprocessing keys shown in previews, in order.
_DETAIL_LABELS = (
    ("source_name", "Source"),
    ("source_sha256", "SHA-256"),
    ("format", "Format"),
    ("expression", "Expression"),
    ("function_name", "Height function"),
    ("function_file", "Function file"),
    ("function_file_sha256", "Function file SHA-256"),
    ("sampling_resolution_deg", "Sampling spacing (deg)"),
    ("sampling_lon_range_deg", "Sampled longitudes (deg)"),
    ("sampling_mode", "Sampling"),
    ("variable", "Variable"),
    ("image_mode", "Pixel mode"),
    ("georeferencing", "Georeferencing"),
    ("units", "Units"),
    ("scale", "Scale"),
    ("offset", "Offset"),
    ("nodata", "No-data value"),
    ("missing_count", "Missing values"),
    ("fill_value_m", "Filled with (m)"),
    ("raster_shape", "Raster (rows x cols)"),
    ("raster_resolution_deg", "Raster spacing (deg)"),
    ("resampling", "Resampling"),
    ("working_resolution_deg", "Working resolution (deg)"),
    ("clip_below_m", "Clipped below (m)"),
    ("smoothing_retained_at_band_limit",
     "Smoothing (amplitude kept at band limit)"),
    ("smoothing_band_limit", "Smoothing band limit (degree)"),
    ("smoothing_retained_at_lmax",
     "Smoothing (amplitude kept at band limit)"),       # legacy key
    ("projection_residual", "Truncation residual (rel. L2)"),
    ("smoothed_residual", "Residual after smoothing"),
)


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


@dataclass(frozen=True, eq=False)
class PreparedPlanet:
    """A planet with terrain resolved on a discretization (host arrays).

    ``topography_lm`` holds the resolved surface-elevation coefficients
    (metres, ``(L+1, L+1)``), or ``None`` for a flat bottom. ``l_cut`` is
    the band limit (the PE dealiased product cut) or ``None`` for the full
    truncation. ``source`` is the elevation raster the terrain came from,
    when it was imported in this session (it is not persisted; saved
    terrain keeps its provenance).
    """

    planet: PlanetSpec
    discretization: DiscretizationSpec
    topography: TopographySpec
    topography_lm: np.ndarray | None
    measured: Mapping[str, Any] = field(default_factory=dict)
    source: Any = None
    label: str = ""
    l_cut: int | None = None

    def __post_init__(self) -> None:
        lm = self.topography_lm
        if lm is not None:
            lm = np.array(lm, dtype=np.complex128)
            lm.flags.writeable = False
            if lm.shape != (self.discretization.lmax + 1,) * 2:
                raise ValueError(
                    f"terrain coefficients {lm.shape} do not match lmax="
                    f"{self.discretization.lmax}")
            object.__setattr__(self, "topography_lm", lm)

    # -- construction from saved data (CPU) ---------------------------------

    @classmethod
    def from_initial_state(cls, state) -> "PreparedPlanet":
        """The prepared planet of a saved SWE initial state (CPU only).

        ``state`` is an :class:`~tropoi.representation.archive.initial_state.InitialState`,
        a saved state directory, or a custom run directory (its
        ``initial_state/`` is used).
        """
        from tropoi.representation.archive.initial_state import (
            InitialState, load_initial_state)

        if not isinstance(state, InitialState):
            path = pathlib.Path(state)
            if (path / "initial_state").is_dir():
                path = path / "initial_state"
            state = load_initial_state(path)
        if state.solver != "swe":
            raise ValueError(
                f"a {state.solver.upper()} initial state has no terrain; "
                "only shallow-water states carry topography")
        return cls(state.planet, state.discretization, state.topography,
                   state.topography_lm, label=state.label)

    @classmethod
    def load(cls, path) -> "PreparedPlanet":
        """Read a saved prepared terrain (CPU only).

        ``path`` is a directory written by :meth:`save`, or a PE run
        directory whose capsule keeps one under ``terrain/``.
        """
        from tropoi.representation.archive.prepared_terrain import (
            is_prepared_terrain, load_prepared_terrain)

        path = pathlib.Path(path)
        if not is_prepared_terrain(path) and is_prepared_terrain(
                path / "terrain"):
            path = path / "terrain"
        saved = load_prepared_terrain(path)
        return cls(saved.planet, saved.discretization, saved.topography,
                   saved.topography_lm, label=saved.label, l_cut=saved.l_cut)

    def save(self, path, *, overwrite: bool = False) -> pathlib.Path:
        """Write the resolved terrain and its record to ``path`` (CPU).

        The source raster is not stored; its provenance is. A PE run reads
        the directory with ``tropoi run pe --terrain PATH``.
        """
        from tropoi.representation.archive.prepared_terrain import (
            SavedTerrain, save_prepared_terrain)

        if self.topography_lm is None:
            raise ValueError("a flat bottom has no terrain to save")
        return save_prepared_terrain(
            SavedTerrain(self.planet, self.discretization, self.topography,
                         self.topography_lm, l_cut=self.l_cut,
                         label=self.label), path, overwrite=overwrite)

    # -- description ----------------------------------------------------------

    @property
    def is_flat(self) -> bool:
        return self.topography_lm is None

    def describe(self) -> str:
        spec = self.topography
        if spec.preset == "imported":
            what = f"imported from {spec.parameters.get('source_name')}"
        elif spec.preset == "mountain":
            p = spec.parameters
            what = (f"mountain {p['height_m']:g} m at ({p['lat_deg']:g}, "
                    f"{p['lon_deg']:g}), width {p['width_deg']:g} deg")
        else:
            what = spec.preset
        band = "" if self.l_cut is None else f", band limit {self.l_cut} (PE)"
        return f"{what}; {self.discretization.describe()}{band}"

    def details(self) -> list[tuple[str, str]]:
        """(label, value) lines: planet, grid, terrain provenance, ranges."""
        lines = [("Planet", self.planet.describe()),
                 ("Discretization", self.discretization.describe()),
                 ("Terrain", self.topography.preset)]
        if self.l_cut is not None:
            lines.append(("Band limit", f"l_cut = {self.l_cut}: the PE "
                          "dealiased product cut; nothing above it"))
        params = dict(self.topography.parameters)
        params.update(self.measured)
        for key, label in _DETAIL_LABELS:
            value = params.get(key)
            if value is not None:
                lines.append((label, _fmt(value)))
        if self.topography.preset == "mountain":
            for key in ("height_m", "lat_deg", "lon_deg", "width_deg"):
                lines.append((key, _fmt(params[key])))
        lo, hi = params.get("elevation_min_m"), params.get("elevation_max_m")
        if lo is not None and hi is not None:
            lines.append(("Model range on state grid (m)",
                          f"{lo:.6g} .. {hi:.6g}"))
        lo, hi = params.get("source_min_m"), params.get("source_max_m")
        if lo is not None and hi is not None:
            lines.append(("Source range (m)", f"{lo:.6g} .. {hi:.6g}"))
        if self.topography_lm is not None:
            from tropoi.representation.archive.initial_state import (
                array_digest)
            lines.append(("Coefficient digest",
                          array_digest(self.topography_lm)))
        return lines

    # -- previews ---------------------------------------------------------------

    def preview(self):
        """The :class:`~tropoi.representation.visual.terrain.TerrainPreview`."""
        from tropoi.representation.visual.terrain import build_terrain_preview

        lm = self.topography_lm
        if lm is None:
            n = self.discretization.lmax + 1
            lm = np.zeros((n, n), dtype=np.complex128)
        title = self.label or f"Terrain: {self.describe()}"
        return build_terrain_preview(
            lm, radius_m=self.planet.dynamical_radius_m, title=title,
            subtitle=self.describe() if self.label else "",
            details=self.details(), raster=self.source)

    def plot(self, *, figsize=None):
        """Static terrain figure (Matplotlib); returns the figure."""
        from tropoi.representation.visual.terrain import plot_terrain

        return plot_terrain(self.preview(), figsize=figsize)

    def view(self, path=None, *, open_browser: bool = True,
             units: str | None = None,
             centre: tuple[float, float] | None = None,
             exaggeration: float | None = None) -> pathlib.Path:
        """Write the interactive globe (one HTML file) and open it.

        ``path`` defaults to a file named after this terrain in the system
        temporary directory. The page is self-contained: it opens from disk
        with no server, network, Python or CUDA. Returns the path.
        ``units`` ("m" or "km"), ``centre`` = (lat, lon) and
        ``exaggeration`` set the page's initial colour-bar unit, view and
        display-only relief factor; all remain adjustable in the page.
        """
        from tropoi.representation.visual.terrain import write_globe_html

        if path is None:
            path = pathlib.Path(tempfile.gettempdir()) / "tropoi" / (
                self._slug() + "-globe.html")
        path = write_globe_html(self.preview(), path, units=units,
                                centre=centre, exaggeration=exaggeration)
        if open_browser:
            import webbrowser
            webbrowser.open(path.resolve().as_uri())
        return path

    def _slug(self) -> str:
        base = self.label or str(self.topography.parameters.get(
            "source_name") or self.topography.preset)
        base = re.sub(r"[^A-Za-z0-9]+", "-", base).strip("-").lower()[:40]
        lm = (b"" if self.topography_lm is None
              else self.topography_lm.tobytes())
        tag = hashlib.sha256(lm + self.discretization.describe().encode()
                             ).hexdigest()[:8]
        return f"{base or 'terrain'}-l{self.discretization.lmax}-{tag}"


#: Measurements a preset reports about its resolved coefficients.
_COEFFICIENT_MEASURES = ("projection_residual", "elevation_min_m",
                         "elevation_max_m", "peak_error_m")


def _measure_after_cut(planet_obj, spec: TopographySpec, lm: np.ndarray,
                       measured: Mapping[str, float]) -> dict:
    """Re-measure a preset from the coefficients kept after a band cut.

    Extrema come from synthesizing ``lm`` on the state grid. The residual
    and peak error need the analytic field, which is known here for the
    Williamson-5 cone only; for any other preset they are omitted rather
    than reported from the uncut projection.
    """
    import cupy as cp

    out = {k: v for k, v in measured.items()
           if k not in _COEFFICIENT_MEASURES}
    sh = planet_obj.sh
    synth = sh.inv_transform(cp.asarray(lm)).real
    out["elevation_min_m"] = float(synth.min())
    out["elevation_max_m"] = float(synth.max())
    if spec.preset == "williamson5_cone":
        from tropoi.spatial.terrain.topography import (
            williamson5_cone_elevation)
        grid = planet_obj.grid
        analytic = williamson5_cone_elevation(
            cp.asarray(grid.point_latitudes, dtype=cp.float64),
            cp.asarray(grid.point_longitudes, dtype=cp.float64),
            height_m=spec.parameters["height_m"]).reshape(synth.shape)
        w = cp.asarray(sh.weights, dtype=cp.float64).reshape(synth.shape)
        out["projection_residual"] = float(
            cp.sqrt(cp.sum(w * (synth - analytic) ** 2))
            / cp.sqrt(cp.sum(w * analytic ** 2)))
        out["peak_error_m"] = float(spec.parameters["height_m"]) - float(
            synth.max())
    return out


def prepare_planet(planet: PlanetSpec, discretization: DiscretizationSpec,
                   terrain=None, *, smoothing: float | None = None,
                   clip_below_m: float | None = None,
                   label: str = "", solver: str = "swe") -> PreparedPlanet:
    """Resolve terrain on a discretization (requires CUDA).

    ``terrain`` is an :class:`~tropoi.spatial.terrain.raster.ElevationRaster`
    (from :func:`~tropoi.spatial.terrain.raster.read_elevation`), a
    :class:`TopographySpec` preset, or ``None`` (flat). For a raster,
    ``smoothing`` (amplitude retained at the band limit: ``l_cut`` with
    ``solver="pe"``, ``l_max`` otherwise; 0 < s <= 1) and
    ``clip_below_m`` are applied as described in
    :meth:`tropoi.spatial.terrain.topography.Topography.from_raster`, and
    the resulting :class:`TopographySpec` is an ``imported`` provenance
    record.

    ``solver`` says which core the terrain is for. ``"swe"`` (default)
    resolves at the full truncation. ``"pe"`` band-limits it at the
    dealiased product cut ``floor(2 l_max / 3)``, which the primitive-
    equation core requires (content above it cannot be balanced by the
    dealiased pressure-gradient pathway; see ``Topography.mountain``).
    """
    from tropoi.run.custom.resources import (PreparationError,
                                             build_spec_planet, check_specs,
                                             resolve_topography)
    from tropoi.spatial.terrain.raster import ElevationRaster

    check_specs(planet, discretization)
    if solver not in ("swe", "pe"):
        raise PreparationError(
            f"solver must be 'swe' or 'pe', got {solver!r}")
    l_cut = None
    if solver == "pe":
        from tropoi.spatial.truncation import product_truncation_cut
        l_cut = product_truncation_cut(discretization.lmax)
    if isinstance(terrain, (str, pathlib.Path)):
        raise PreparationError(
            "read the file first with read_elevation(path, ...) so its "
            "interpretation (units, orientation, georeferencing) is explicit")
    if terrain is None:
        terrain = TopographySpec.flat()
    if isinstance(terrain, TopographySpec):
        if smoothing is not None or clip_below_m is not None:
            raise PreparationError(
                "smoothing and clip_below_m apply to imported rasters only")
        if terrain.preset == "imported":
            raise PreparationError(
                "an 'imported' TopographySpec is a provenance record; pass "
                "the ElevationRaster instead")
        if terrain.is_flat:
            return PreparedPlanet(planet, discretization, terrain, None,
                                  label=label, l_cut=l_cut)
        import cupy as cp
        planet_obj = build_spec_planet(planet, discretization)
        if l_cut is not None and terrain.preset == "mountain":
            from tropoi.spatial.terrain.topography import Topography
            p = terrain.parameters
            resolved = Topography.mountain(
                planet_obj, height_m=p["height_m"], lat_deg=p["lat_deg"],
                lon_deg=p["lon_deg"], width_deg=p["width_deg"], l_cut=l_cut)
        else:
            resolved = resolve_topography(planet_obj, terrain)
        full = cp.asnumpy(resolved.elevation_lm)
        lm = full.copy()
        if l_cut is not None:
            lm[l_cut + 1:, :] = 0.0
            lm[:, l_cut + 1:] = 0.0
        measured = {k: float(v) for k, v in resolved.parameters.items()
                    if isinstance(v, (int, float)) and not isinstance(v, bool)
                    and math.isfinite(float(v))}
        if not np.array_equal(lm, full):
            # The preset measured its full-truncation projection; the cut
            # changed the coefficients, so measure what is actually kept.
            measured = _measure_after_cut(planet_obj, terrain, lm, measured)
        return PreparedPlanet(planet, discretization, terrain, lm,
                              measured=measured, label=label, l_cut=l_cut)
    if not isinstance(terrain, ElevationRaster):
        raise PreparationError(
            "terrain must be an ElevationRaster, a TopographySpec or None, "
            f"got {type(terrain).__name__}")

    import cupy as cp

    from tropoi.spatial.terrain.topography import Topography, TopographyError

    planet_obj = build_spec_planet(planet, discretization)
    try:
        resolved = Topography.from_raster(planet_obj, terrain,
                                          smoothing=smoothing,
                                          clip_below_m=clip_below_m,
                                          l_cut=l_cut)
    except TopographyError as err:
        raise PreparationError(str(err)) from err
    spec = TopographySpec.imported(resolved.parameters)
    return PreparedPlanet(planet, discretization, spec,
                          cp.asnumpy(resolved.elevation_lm), source=terrain,
                          label=label, l_cut=l_cut)
