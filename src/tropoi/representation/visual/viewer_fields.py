"""Read-only Viewer 2 selections on a common south-first, east-positive mesh."""
from __future__ import annotations

from functools import cached_property
from pathlib import Path

import numpy as np

from tropoi.representation.archive.local_resource import resource
from tropoi.representation.archive.pe_context import PEColumnContext
from tropoi.representation.archive.prepared_terrain import load_prepared_terrain
from tropoi.representation.diagnostics.held_suarez_sections import state_grid
from tropoi.representation.diagnostics.pe_levels import reconstruct_levels
from tropoi.representation.diagnostics.pe_structure import (
    ZonalMeanAccumulator, column, mass_streamfunction, vertical_section)
from tropoi.representation.diagnostics.pe_surfaces import pressure_surface
from tropoi.spatial.transforms.host_synthesis import synthesize_latlon, synthesize_wind_latlon
from tropoi.spatial.williamson5 import W5_PROJECTION_POLICY


def _cone_coefficients(physics, geometry, lmax):
    """Analyze the analytic cone on the solver's own Gaussian state quadrature."""
    from scipy.special import sph_harm_y

    if geometry["grid"] != "latlon" or physics.get("w5_projection") != W5_PROJECTION_POLICY:
        raise ValueError("Williamson-5 terrain needs the Gaussian state-grid projection")
    nlat, nlon = geometry["nlat"], geometry["nlon"]
    from tropoi.spatial.grids.quadrature import gauss_legendre
    nodes, weights = gauss_legendre(nlat)
    lat = np.arcsin(nodes)
    lon = 2 * np.pi * np.arange(nlon) / nlon
    dlon = (lon - np.radians(physics["w5_cone_lon_deg"]) + np.pi) % (2*np.pi) - np.pi
    radius = np.hypot(lat[:, None] - np.radians(physics["w5_cone_lat_deg"]), dlon)
    cone = physics["w5_cone_height_m"] * np.maximum(0, 1 - radius / physics["w5_cone_radius_rad"])
    orders = np.fft.fft(cone, axis=1) * (2 * np.pi / nlon) * weights[:, None]
    coeffs = np.zeros((lmax + 1, lmax + 1), dtype=np.complex128)
    theta = np.pi/2 - lat
    for m in range(lmax + 1):
        degrees = np.arange(m, lmax + 1)
        legendre = sph_harm_y(degrees[:, None], m, theta, 0).real
        coeffs[m:, m] = legendre @ orders[:, m % nlon]
    return coeffs


def _slab_map(values, geometry, lat, lon):
    """Bilinear in geographic angles, periodic in longitude, nearest polar row."""
    nlat, nlon = geometry["nlat"], geometry["nlon"]
    if (geometry.get("slab_order") != "latitude, longitude"
            or geometry.get("latitude") != "Gaussian north to south"
            or geometry.get("longitude") != "360*j/nlon degrees east"):
        raise ValueError("Unrecognized slab grid ordering")
    source_lat, source_lon, _ = state_grid(nlat, nlon)
    rows = np.asarray(values).reshape(nlat, nlon)[::-1]
    by_lon = np.stack([np.interp(lon, source_lon, row, period=360) for row in rows])
    return np.stack([np.interp(lat, source_lat[::-1], column) for column in by_lon.T], axis=1)


class FrameProvider:
    """Evaluate exact saved-time indices. The caller owns the shared result cache.

    PE native values are zero-based full-level indices (0 = top). Pressure
    (Pa), geopotential height (m) and geopotential height above the local
    surface (m, ``Z - Z_s``) use hydrostatic full-level positions; sigma
    uses nominal positions. Surface-only fields use the native coordinate.
    """

    def __init__(self, root, manifest, *, backend="cpu"):
        if backend not in ("cpu", "cuda"):
            raise ValueError("backend must be cpu or cuda")
        self.root = Path(root).resolve(strict=True)
        self.manifest = manifest
        self.solver = manifest["source"].get("solver")
        self.geometry = manifest["geometry"]
        self.physics = manifest["physics"]
        self.lat = np.linspace(-90, 90, 91)
        self.lon = np.arange(0, 360, 2, dtype=float)
        self.shape = (self.lat.size, self.lon.size)
        self.backend = "cpu"
        self.backend_reason = ""
        self._gpu = None
        if backend == "cuda" and self.solver in ("bve", "swe"):
            try:
                from tropoi.spatial.transforms.separable_gauss_sh import SeparableGaussSphericalHarmonics

                # Synthesis only: the separated evaluation accepts arbitrary
                # latitude rows and uniformly spaced longitudes.
                self._gpu = SeparableGaussSphericalHarmonics(
                    np.radians(self.lat), self.lon.size, manifest["axes"]["support"]["l_max"])
                self.backend = "cuda"
            except (ImportError, OSError, RuntimeError, ValueError) as err:
                self.backend_reason = str(err)
        elif backend == "cuda":
            self.backend_reason = "The atmospheric column evaluator uses NumPy."

    # The viewer's selection protocol (``selection.evaluate``): sample sets per
    # role and per-state evaluation live in ``viewer_atmosphere``.
    def plan(self, selection):
        from tropoi.representation.visual import viewer_atmosphere
        return viewer_atmosphere.plan(self, selection)

    def samples(self, selection, plan, indices):
        from tropoi.representation.visual import viewer_atmosphere
        return viewer_atmosphere.samples(self, selection, plan, indices)

    def post(self, selection, plan, cube):
        from tropoi.representation.visual import viewer_atmosphere
        return viewer_atmosphere.post(self, selection, plan, cube)

    def finish(self, selection, plan, payload):
        from tropoi.representation.visual import viewer_atmosphere
        viewer_atmosphere.finish(self, selection, plan, payload)

    def frame(self, selection):
        """One saved state on one surface for the globe: :meth:`slice`."""
        vertical = selection["vertical"]
        return self.slice(selection["time"]["index"], selection["quantity"],
                          vertical.get("representation", "native"), vertical.get("value", 0))

    def _scalar(self, coeffs):
        if self._gpu is not None:
            try:
                return self._gpu.inv_transform(coeffs).get().reshape(self.shape)
            except (OSError, RuntimeError) as err:
                self.backend_reason = str(err)
                self.backend = "cpu"
                self._gpu = None
        return synthesize_latlon(coeffs, self.lat, self.lon)

    def _read(self, kind, index):
        spec = self.manifest["resources"][kind]
        name = spec["paths"][index] if "paths" in spec else spec["path"]
        data = np.load(resource(self.root, name), mmap_mode="r", allow_pickle=False)
        frame = data if "paths" in spec else data[index]
        expected_kind = "f" if kind == "slab" else "c"
        if tuple(spec["frame_shape"]) != frame.shape or frame.dtype.kind != expected_kind:
            raise ValueError(f"Invalid {kind} snapshot shape or dtype in {name}")
        return frame

    @cached_property
    def _terrain_lm(self):
        saved = self.manifest["resources"].get("terrain")
        if saved:
            # The digest-checking reader uses two fixed resource names.
            resource(self.root, "terrain/terrain.json")
            path = resource(self.root, "terrain/topography_elevation_lm.npy")
            terrain = load_prepared_terrain(path.parent)
            if terrain.digest != saved["digest"]:
                raise ValueError("Prepared terrain disagrees with the archived digest")
            return np.asarray(terrain.topography_lm)
        if self.physics.get("topography") == "williamson5_cone":
            return _cone_coefficients(self.physics, self.geometry,
                                      self.manifest["axes"]["support"]["l_max"])
        if self.physics.get("topography", "flat") != "flat" or self.physics.get("terrain_digest"):
            raise ValueError("The archived terrain cannot be reconstructed")
        return None

    @cached_property
    def terrain(self):
        if self._terrain_lm is None:
            return np.zeros(self.shape)
        return self._scalar(self._terrain_lm)

    @property
    def radius(self):
        return self.geometry.get("dynamical_radius_m") or self.geometry["reference_radius_m"]

    @cached_property
    def _context(self):
        p = self.physics
        return PEColumnContext(
            self.root.name, self.manifest["source"]["format"], self.manifest["axes"]["support"]["l_max"],
            tuple(self.manifest["axes"]["vertical"]["interfaces"]),
            p.get("r_dry", p.get("gas_constant")), p.get("cp_dry", p.get("cp")),
            self.radius, p.get("gravity"), self._terrain_lm, p.get("topography", "flat"))

    def _result(self, values, height, mask=None, wind=None):
        if mask is None:
            mask = np.zeros(self.shape, dtype=np.uint8)
        result = dict(lat=self.lat, lon=self.lon, values=values, terrain=self.terrain,
                      height=height, mask=mask)
        if wind is not None:
            result.update(u=wind[0], v=wind[1])
        if any(not np.isfinite(a).all() for a in result.values()):
            raise ValueError("Selected field or geometry contains non-finite samples")
        return result

    def slice(self, time: int, field: str, coordinate: str = "native", value: float = 0):
        times = self.manifest["time_axes"].get("snapshots", {}).get("values", [])
        if isinstance(time, bool) or not isinstance(time, (int, np.integer)) or not 0 <= time < len(times):
            raise ValueError("Saved time index is out of range")
        if field not in self.manifest["fields"]:
            raise ValueError(f"Unavailable field {field!r}")
        if not np.isfinite(value):
            raise ValueError("Coordinate value must be finite")
        if coordinate != "native" and field not in self.manifest["operations"].get(coordinate, []):
            raise ValueError(f"Unavailable coordinate {coordinate!r} for {field}")
        if field == "surface_temperature":
            slab_times = self.manifest["time_axes"]["slab"]["values"]
            if times[time] not in slab_times:
                raise ValueError("No slab snapshot at the selected atmospheric time")
            slab = self._read("slab", slab_times.index(times[time]))
            return self._result(_slab_map(slab, self.geometry, self.lat, self.lon), self.terrain)
        if self.solver == "pe" and field == "terrain":
            return self._result(self.terrain, self.terrain)
        frame = self._read("state", time)
        if self.solver == "pe":
            if field == "ln_ps":
                return self._result(np.exp(self._scalar(frame[-1])), self.terrain)
            return self._pe_slice(frame, field, coordinate, value)
        if coordinate != "native" or self.solver not in ("bve", "swe"):
            raise ValueError("Run has no evaluator for this selection")
        return self._horizontal_slice(frame, field)

    def _horizontal_slice(self, frame, field):
        swe = self.solver == "swe"
        zeta = frame[0] if swe else frame
        delta = frame[1] if swe else np.zeros_like(zeta)
        wind = synthesize_wind_latlon(zeta, delta, self.radius, self.lat, self.lon)
        fields = {"u": wind[0], "v": wind[1]}
        height = self.terrain
        mask = None
        if swe:
            gravity = self.physics["gravity"]
            phi = self._scalar(frame[2])
            thickness = self.physics["mean_depth_m"] + phi / gravity
            height = self.terrain + thickness
            mask = (thickness <= 0).astype(np.uint8)
            fields.update(phi=phi, terrain=self.terrain, layer_depth=thickness,
                          free_surface_height=height, free_surface_geopotential=gravity * height)
        if field in ("wind", "wind_speed"):
            values = np.hypot(*wind)
        elif field in ("zeta", "delta"):
            values = self._scalar(zeta if field == "zeta" else delta)
        else:
            values = fields[field]
        return self._result(values, height, mask, wind)

    def _pe_slice(self, frame, field, coordinate, value):
        if coordinate == "native":
            if int(value) != value or not 0 <= value < self._context.nlev:
                raise ValueError("Native level must be a zero-based full-level index")
            # A flat run may record no gravity: its native fields are still
            # defined, but Z = Φ/g is not, so the layer is drawn on the datum.
            heights = self._context.gravity is not None
            native = reconstruct_levels(frame, self._context, self.lat, self.lon, [int(value)],
                                        vorticity_divergence=field in ("zeta", "delta"), heights=heights)
            fields = {name: arr[0] for name, arr in native.values.items()}
            if not heights:
                fields["geopotential_height"] = self.terrain
            mask = None
        else:
            if coordinate not in ("pressure", "sigma", "geopotential_height", "height_above_surface"):
                raise ValueError(f"Unsupported coordinate {coordinate!r}")
            if coordinate != "sigma" and field in ("zeta", "delta"):
                raise ValueError("Vorticity and divergence require a native or sigma surface")
            surface = pressure_surface(frame, self._context, self.lat, self.lon, (coordinate, value))
            fields, mask = surface.values, surface.mask
            if coordinate == "pressure" and field == "pressure":
                fields["pressure"] = np.full(self.shape, value, dtype=float)
            if coordinate in ("geopotential_height", "height_above_surface") and field == "geopotential":
                fields["geopotential"] = fields["geopotential_height"] * self._context.gravity
        u, v = fields["u"], fields["v"]
        if field == "density":
            from tropoi.representation.visual.viewer_atmosphere import density
            pressure = fields.get("pressure")
            if pressure is None:                         # a surface of constant p carries its own p
                pressure = np.full(self.shape, value, dtype=float)
            # Masked samples (p = 0 below the terrain) stay masked; avoid 0/0 there.
            temperature = np.where(mask != 0, 1.0, fields["temperature"]) if mask is not None else fields["temperature"]
            fields = dict(fields, density=density(pressure, temperature, self._context.r_dry))
        values = np.hypot(u, v) if field in ("wind", "wind_speed") else fields[field]
        return self._result(values, fields["geopotential_height"], mask, (u, v))

    # ------------------------------------------------------------------
    # Linked vertical structure (PE only). Scientific reductions use the
    # run's own Gaussian state grid; columns and sections are exact
    # band-limited point synthesis, never values read off the 2° mesh.
    # The viewer evaluates selections through ``selection.evaluate``; the
    # methods below remain the library entry points of the Ben-2
    # acceptance tests, on the same kernels.
    # ------------------------------------------------------------------

    def _structure_context(self):
        if self.solver != "pe" or "structure" not in self.manifest["operations"]:
            raise ValueError("This archive has no atmospheric column evaluator")
        return self._context

    def _saved_index(self, time):
        times = self.manifest["time_axes"].get("snapshots", {}).get("values", [])
        if isinstance(time, bool) or not isinstance(time, (int, np.integer)) or not 0 <= time < len(times):
            raise ValueError("Saved time index is out of range")
        return int(time), times[time]

    def _native_grid(self):
        return state_grid(self.geometry["nlat"], self.geometry["nlon"])[:2]

    def _slab_point(self, seconds, lat, lon, node):
        """Slab temperature at a native node, or bilinear between nodes."""
        axis = self.manifest["time_axes"].get("slab")
        if not axis or seconds not in axis["values"]:
            return None
        values = self._read("slab", axis["values"].index(seconds))
        nlat, nlon = self.geometry["nlat"], self.geometry["nlon"]
        if node is not None:
            value = float(np.asarray(values).reshape(nlat, nlon)[node])
            sampling = "slab value at the native grid node"
        else:
            value = float(_slab_map(values, self.geometry, np.array([lat]), np.array([lon]))[0, 0])
            sampling = "bilinear between native slab nodes (periodic in longitude)"
        return dict(value=value, units="K", sampling=sampling,
                    definition="Slab surface temperature Tₛ, a stored grid-point prognostic")

    def column(self, time, lat, lon, snap=True):
        """Native-level profile at one geographic point and saved time."""
        context = self._structure_context()
        index, seconds = self._saved_index(time)
        lat, lon = float(lat), float(lon) % 360.0
        node = None
        # Only Gauss–Legendre latitude–longitude grids have nodes to snap to;
        # elsewhere the column is exact point synthesis, and says so.
        if snap and self.geometry.get("grid") == "latlon":
            grid_lat, grid_lon = self._native_grid()
            i = int(np.abs(grid_lat - lat).argmin())
            j = int(np.abs((grid_lon - lon + 180) % 360 - 180).argmin())
            lat, lon, node = float(grid_lat[i]), float(grid_lon[j]), (i, j)
        profile = column(self._read("state", index), context, lat, lon)
        k = context.nlev - 1
        result = {key: (value.tolist() if isinstance(value, np.ndarray) else value)
                  for key, value in profile.items()}
        result.update(
            time=seconds, sampling=("native Gaussian grid node" if node is not None else
                                    "exact band-limited synthesis at the requested point"),
            lowest_air=dict(value=float(profile["temperature"][k]), level=k, sigma=float(profile["sigma"][k]),
                            pressure=float(profile["pressure"][k]),
                            height=float(profile["geopotential_height"][k]),
                            definition="Air temperature at the lowest full level, not a surface temperature"),
            surface_temperature=(self._slab_point(seconds, lat, lon, node)
                                 if "surface_temperature" in self.manifest["fields"] else None),
            conventions=dict(
                pressure="p_k = σ_k pₛ at the nominal full level (Pa)",
                geopotential_height=("Z_k = Φ_k/g, Simmons–Burridge full-level geopotential, absolute, at the "
                                     "hydrostatic level position σʰ_k (m above the model datum)"),
                wind="u eastward, v northward (m s⁻¹)"))
        return result

    def section(self, time, kind, at):
        """Latitude × level at a longitude, or longitude × level at a latitude."""
        context = self._structure_context()
        index, seconds = self._saved_index(time)
        if kind == "meridional":
            points = self.lat
            data = vertical_section(self._read("state", index), context, longitude=at, points=points)
        elif kind == "zonal":
            points = np.append(self.lon, 360.0)
            data = vertical_section(self._read("state", index), context, latitude=at, points=points)
        else:
            raise ValueError("Section kind must be meridional or zonal")
        result = {key: value.tolist() for key, value in data.items()}
        result.update(kind=kind, at=float(at) % 360.0 if kind == "meridional" else float(at),
                      points=points.tolist(), time=seconds,
                      sigma=list(context.full_levels), sigma_interfaces=list(context.sigma_interfaces),
                      sampling="exact band-limited synthesis on native full levels, 2° horizontal sample spacing")
        return result

    def zonal_mean(self, first, last):
        """Equal-weight zonal means of saved states first..last on the native grid."""
        context = self._structure_context()
        if "zonal_mean" not in self.manifest["operations"]["structure"]:
            raise ValueError(self.manifest["unavailable"].get("zonal_mean", "No zonal reduction"))
        first, start = self._saved_index(first)
        last, end = self._saved_index(last)
        if last < first:
            raise ValueError("The last saved time precedes the first")
        if last - first + 1 > 400:
            raise ValueError("Select at most 400 saved states for one mean")
        accumulator = ZonalMeanAccumulator(context, self.geometry["nlat"], self.geometry["nlon"])
        for index in range(first, last + 1):
            accumulator.add(self._read("state", index))
        means = accumulator.means()
        interfaces = context.sigma_interfaces
        psi = mass_streamfunction(means["ps_v"], interfaces, accumulator.lat_deg,
                                  context.radius_m, context.gravity)
        times = self.manifest["time_axes"]["snapshots"]["values"][first:last + 1]
        steps = np.diff(times)
        intervals = {str(float(step)): int(count) for step, count in zip(*np.unique(steps, return_counts=True))}
        order = slice(None, None, -1)                       # south to north, as the display mesh
        result = {key: means[key][:, order].tolist() for key in ("u", "v", "temperature")}
        result.update(
            lat=accumulator.lat_deg[order].tolist(), sigma=list(context.full_levels),
            sigma_interfaces=list(interfaces), streamfunction=psi[:, order].tolist(),
            surface_pressure=means["surface_pressure"][order].tolist(),
            column_flux_residual=dict(
                max_abs=float(np.abs(psi[-1]).max()), max_abs_interior=float(np.abs(psi).max()),
                note="ψ at σ = 1 is the whole-column northward mass flux; not forced to zero"),
            sample=dict(rule="equal-weight mean of every saved state from the first to the last selection",
                        times=times, count=len(times), intervals_seconds=intervals,
                        cadence=("single state" if len(times) == 1 else
                                 "regular" if len(intervals) == 1 else "irregular")),
            grid=dict(latitudes=f"{len(accumulator.lat_deg)} Gauss–Legendre nodes, south to north",
                      longitudes=f"{len(accumulator.lon_deg)} equally spaced, plain mean",
                      source=accumulator.latitude_source),
            definitions=dict(
                zonal_mean="[x] is the plain mean over the native longitudes at each latitude and full level",
                streamfunction=("ψ(φ, σ_{k+½}) = 2πa cos φ / g · Σ_{j≤k} [pₛ v_j] Δσ_j, integrated downward from "
                                "ψ = 0 at σ = 0; kg s⁻¹, positive for northward mass transport above the interface. "
                                "[pₛ v] is formed per state before the longitude mean")))
        return result
