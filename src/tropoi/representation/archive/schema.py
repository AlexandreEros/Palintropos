"""One versioned field/support/diagnostic definition per solver.

The same definitions serve two consumers:

* the run lifecycle writes them into new manifests as the additive
  ``state_schema`` and ``diagnostic_definitions`` blocks (descriptive
  provenance only: nothing here enters ``run_config`` or the scientific
  configuration hash);
* the capsule reader consumes them to interpret stored coefficient arrays.
  Manifests written before these blocks existed are interpreted by
  :func:`infer_state_schema` for the KNOWN historical formats only, and the
  result says so (``provenance.source == "inferred"``).

Everything is derived from the resolved ``run_config`` and the shared
definitions in the physics/diagnostics modules' docstrings, never from
array contents, so the writer needs no model object and the reader needs
no CUDA. Import-light (stdlib + NumPy-free): consumed by the CLI.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import math

from tropoi.spatial.modes import (COEFFICIENT_LAYOUT,
                                  COEFFICIENT_NORMALIZATION, FieldSpec)
from tropoi.spatial.truncation import product_truncation_cut

#: Version of the ``state_schema`` manifest block this module writes/reads.
STATE_SCHEMA_VERSION = 1
#: Version of the ``diagnostic_definitions`` manifest block.
DIAGNOSTIC_DEFINITIONS_VERSION = 1

SOLVERS = ("bve", "swe", "pe")

#: Coefficient file and stored-time file per solver (capsule root).
COEFFICIENT_FILES = {
    "bve": "vorticity_coeffs.npy",
    "swe": "swe_coeffs.npy",
    "pe": "pe_coeffs.npy",
}
TIME_FILES = {
    "bve": "bve_snapshot_times.npy",
    "swe": "swe_snapshot_times.npy",
    "pe": "pe_snapshot_times.npy",
}
#: Axis names of the persisted coefficient arrays.
STORAGE_AXES = {
    "bve": ("time", "l", "m"),
    "swe": ("time", "field", "l", "m"),
    "pe": ("time", "row", "l", "m"),
}

REALITY_CONVENTION = ("real field implied: coefficients stored for m >= 0 "
                      "only, Im(a_l0) == 0, m > l entries are zero padding")
TIME_UNITS = "s"

#: Earth radius and rotation used by ``PlanetaryParameters.from_earth_like``
#: (planet/planetary_parameters.py) and the Williamson (1992) perfect sphere
#: (run/swe/config.py). Duplicated here so the schema stays import-light; a
#: test keeps them synchronized with their sources.
EARTH_RADIUS_M = 6.371e6
W5_RADIUS_M = 6.37122e6


class SchemaError(ValueError):
    """A manifest schema block cannot be interpreted."""


class UnknownSchemaVersionError(SchemaError):
    """The manifest carries a schema version this reader does not know."""


class UnsupportedConventionError(SchemaError):
    """The manifest declares a coefficient/time convention not supported."""


@dataclass(frozen=True)
class StateSchema:
    """Structured interpretation of one capsule's stored prognostic state."""

    solver: str
    fields: tuple[FieldSpec, ...]
    l_max: int
    coefficient_file: str
    time_file: str | None
    storage_axes: tuple[str, ...]
    rows: int | None
    nlev: int | None = None
    vertical: dict | None = None
    geometry: dict = field(default_factory=dict)
    environment: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)
    version: int = STATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        self.check_consistency()

    def check_consistency(self) -> None:
        """Reject schemas the reader could not interpret unambiguously.

        Both the manifest block and the inferred schema pass through here,
        so inspection and plotting interpret the same stored data
        identically or fail identically: unsupported coefficient
        conventions, conflicting/overlapping field rows, rows outside the
        packed frame, and inconsistent PE vertical coordinates are
        SchemaErrors at open (typed access is refused; metadata stays
        readable).
        """
        if self.solver not in SOLVERS:
            raise SchemaError(f"unknown solver {self.solver!r}")
        if not self.fields:
            raise SchemaError("state_schema declares no fields")
        if int(self.l_max) < 0:
            raise SchemaError(f"l_max must be >= 0, got {self.l_max}")
        names = [spec.name for spec in self.fields]
        if len(set(names)) != len(names):
            raise SchemaError(f"state_schema repeats field names: {names}")
        for spec in self.fields:
            if spec.normalization != COEFFICIENT_NORMALIZATION:
                raise SchemaError(
                    f"field {spec.name!r} declares normalization "
                    f"{spec.normalization!r}; this reader supports only "
                    f"{COEFFICIENT_NORMALIZATION!r}")
            if spec.layout != COEFFICIENT_LAYOUT:
                raise SchemaError(
                    f"field {spec.name!r} declares layout {spec.layout!r}; "
                    f"this reader supports only {COEFFICIENT_LAYOUT!r}")
        if self.rows is None:
            if len(self.fields) != 1 or self.fields[0].rows is not None:
                raise SchemaError(
                    "a schema without a row axis must declare exactly one "
                    "field spanning the whole (l, m) frame")
        else:
            rows = int(self.rows)
            if rows < 1:
                raise SchemaError(f"rows must be >= 1, got {rows}")
            occupied: list[tuple[int, int, str]] = []
            for spec in self.fields:
                if spec.rows is None:
                    raise SchemaError(
                        f"field {spec.name!r} has no row range but the "
                        f"frame has {rows} row(s)")
                start, stop = spec.rows
                if stop > rows:
                    raise SchemaError(
                        f"field {spec.name!r} rows [{start}, {stop}) exceed "
                        f"the {rows} stored row(s)")
                for o_start, o_stop, o_name in occupied:
                    if start < o_stop and o_start < stop:
                        raise SchemaError(
                            f"fields {o_name!r} and {spec.name!r} claim "
                            "overlapping rows "
                            f"[{o_start}, {o_stop}) and [{start}, {stop})")
                occupied.append((start, stop, spec.name))
            covered = sum(stop - start for start, stop, _ in occupied)
            if covered != rows:
                raise SchemaError(
                    f"fields cover {covered} of the {rows} stored row(s); "
                    "the row layout is ambiguous")
        self._check_canonical_layout()
        levelled = [spec for spec in self.fields if spec.levels]
        if levelled or self.nlev is not None or self.vertical is not None:
            if self.nlev is None or self.vertical is None:
                raise SchemaError(
                    "levelled fields require both nlev and the vertical "
                    "coordinate block")
            nlev = int(self.nlev)
            for spec in levelled:
                if spec.nlev != nlev:
                    raise SchemaError(
                        f"field {spec.name!r} spans {spec.nlev} level(s) "
                        f"but the schema declares nlev={nlev}")
            try:
                interfaces = [float(v) for v in self.vertical["interfaces"]]
                full = [float(v) for v in self.vertical["full_levels"]]
            except (KeyError, TypeError, ValueError) as err:
                raise SchemaError(
                    f"malformed vertical coordinate block: {err}") from err
            if len(interfaces) != nlev + 1 or len(full) != nlev:
                raise SchemaError(
                    f"vertical block has {len(interfaces)} interfaces and "
                    f"{len(full)} full levels for nlev={nlev}")
            if (not all(math.isfinite(v) for v in interfaces)
                    or interfaces[0] != 0.0 or interfaces[-1] != 1.0
                    or any(interfaces[k] >= interfaces[k + 1]
                           for k in range(nlev))):
                raise SchemaError(
                    "sigma interfaces must be finite, strictly increasing "
                    f"from exactly 0.0 to exactly 1.0, got {interfaces}")
            expected = [0.5 * (interfaces[k] + interfaces[k + 1])
                        for k in range(nlev)]
            if any(abs(a - b) > 1e-12 for a, b in zip(full, expected)):
                raise SchemaError(
                    "full levels are not the interface midpoints: "
                    f"{full} vs {expected}")

    def _check_canonical_layout(self) -> None:
        """Each named field must map to the rows the solver/renderer use.

        The renderers and cores interpret the packed array with ONE fixed
        layout per solver (run/*/runner.py); a manifest describing a
        structurally valid but permuted mapping would make typed access
        disagree with plotting, so it is refused. Field-list ordering is
        free; names, row ranges and level axes are not.
        """
        if self.storage_axes != STORAGE_AXES[self.solver]:
            raise UnsupportedConventionError(
                f"storage_axes {list(self.storage_axes)} differ from the "
                f"persisted {list(STORAGE_AXES[self.solver])} for solver "
                f"{self.solver!r}; axis permutations are not supported")
        canonical = canonical_field_layout(self.solver, self.nlev)
        declared = {spec.name: (spec.rows, spec.levels) for spec in self.fields}
        if set(declared) != set(canonical):
            raise SchemaError(
                f"solver {self.solver!r} stores fields "
                f"{sorted(canonical)}, but the schema declares "
                f"{sorted(declared)}")
        for name, (rows, levels) in canonical.items():
            if declared[name] != (rows, levels):
                raise SchemaError(
                    f"field {name!r} must map to rows {rows} "
                    f"({'levelled' if levels else 'level-free'}) for solver "
                    f"{self.solver!r}, but the schema declares rows "
                    f"{declared[name][0]} "
                    f"({'levelled' if declared[name][1] else 'level-free'})")

    @property
    def field_names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.fields)

    @property
    def support(self) -> dict:
        return {
            "l_max": self.l_max,
            "product_truncation_cut": product_truncation_cut(self.l_max),
            "triangle": "0 <= m <= l <= l_max",
        }

    @property
    def level_values(self) -> tuple[float, ...] | None:
        if self.vertical is None:
            return None
        return tuple(float(v) for v in self.vertical["full_levels"])

    @property
    def frame_shape(self) -> tuple[int, ...]:
        """Expected shape of one packed frame (without the time axis)."""
        n = self.l_max + 1
        return (n, n) if self.rows is None else (self.rows, n, n)

    @property
    def inferred(self) -> bool:
        return self.provenance.get("source") == "inferred"

    def to_manifest_dict(self) -> dict:
        return {
            "version": self.version,
            "solver": self.solver,
            "coefficient_file": self.coefficient_file,
            "time_file": self.time_file,
            "time_units": TIME_UNITS,
            "storage_axes": list(self.storage_axes),
            "rows": self.rows,
            "fields": [spec.to_dict() for spec in self.fields],
            "support": self.support,
            "normalization": COEFFICIENT_NORMALIZATION,
            "layout": COEFFICIENT_LAYOUT,
            "reality": REALITY_CONVENTION,
            "nlev": self.nlev,
            "vertical": self.vertical,
            "geometry": dict(self.geometry),
            "environment": dict(self.environment),
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_manifest_dict(cls, data: Mapping) -> "StateSchema":
        if not isinstance(data, Mapping):
            raise SchemaError(
                f"state_schema must be an object, got {type(data).__name__}")
        version = data.get("version")
        if version != STATE_SCHEMA_VERSION:
            raise UnknownSchemaVersionError(
                f"state_schema version {version!r} is not supported by this "
                f"reader (known: {STATE_SCHEMA_VERSION}); metadata remains "
                "readable but typed field access is refused")
        solver = data.get("solver")
        if solver not in SOLVERS:
            raise SchemaError(f"state_schema names unknown solver {solver!r}")
        try:
            fields = tuple(FieldSpec.from_dict(f) for f in data["fields"])
            l_max = int(data["support"]["l_max"])
        except (KeyError, TypeError, ValueError) as err:
            raise SchemaError(f"malformed state_schema: {err}") from err
        if not fields:
            raise SchemaError("state_schema declares no fields")
        time_units = data.get("time_units", TIME_UNITS)
        if time_units != TIME_UNITS:
            raise UnsupportedConventionError(
                f"state_schema time_units {time_units!r} is not supported; "
                f"this reader interprets {TIME_UNITS!r} only")
        for key, supported in (("normalization", COEFFICIENT_NORMALIZATION),
                               ("layout", COEFFICIENT_LAYOUT)):
            declared = data.get(key, supported)
            if declared != supported:
                raise UnsupportedConventionError(
                    f"state_schema {key} {declared!r} is not supported; this "
                    f"reader interprets {supported!r} only")
        reality = data.get("reality", REALITY_CONVENTION)
        if reality != REALITY_CONVENTION:
            raise UnsupportedConventionError(
                f"state_schema reality convention {reality!r} is not "
                f"supported; this reader assumes {REALITY_CONVENTION!r}")
        support = data.get("support") or {}
        expected_cut = product_truncation_cut(l_max)
        if "product_truncation_cut" in support and                 int(support["product_truncation_cut"]) != expected_cut:
            raise SchemaError(
                f"state_schema support declares product_truncation_cut="
                f"{support['product_truncation_cut']} but l_max={l_max} "
                f"implies {expected_cut}")
        if "triangle" in support and                 support["triangle"] != "0 <= m <= l <= l_max":
            raise UnsupportedConventionError(
                f"state_schema support triangle {support['triangle']!r} is "
                "not the stored '0 <= m <= l <= l_max' extent")
        rows = data.get("rows")
        return cls(
            solver=solver, fields=fields, l_max=l_max,
            coefficient_file=str(data.get("coefficient_file",
                                          COEFFICIENT_FILES[solver])),
            time_file=data.get("time_file"),
            storage_axes=tuple(data.get("storage_axes",
                                        STORAGE_AXES[solver])),
            rows=None if rows is None else int(rows),
            nlev=None if data.get("nlev") is None else int(data["nlev"]),
            vertical=data.get("vertical"),
            geometry=dict(data.get("geometry") or {}),
            environment=dict(data.get("environment") or {}),
            provenance=dict(data.get("provenance") or
                            {"source": "manifest"}),
            version=int(version))


# ---------------------------------------------------------------------------
# Field definitions (scientific meanings; see the physics module docstrings)
# ---------------------------------------------------------------------------

def canonical_field_layout(solver: str, nlev: int | None
                           ) -> dict[str, tuple[tuple[int, int] | None, bool]]:
    """``name -> (rows, levelled)`` as persisted by each runner.

    BVE: one whole-frame ``zeta``; SWE: ``[zeta, delta, phi]`` rows 0..2;
    PE: ``[zeta_1..K, delta_1..K, T_1..K, ln_ps]`` for ``K = nlev``.
    """
    if solver == "bve":
        return {"zeta": (None, False)}
    if solver == "swe":
        return {"zeta": ((0, 1), False), "delta": ((1, 2), False),
                "phi": ((2, 3), False)}
    if solver == "pe":
        if nlev is None:
            raise SchemaError("a PE schema needs nlev for its row layout")
        K = int(nlev)
        return {"zeta": ((0, K), True), "delta": ((K, 2 * K), True),
                "temperature": ((2 * K, 3 * K), True),
                "ln_ps": ((3 * K, 3 * K + 1), False)}
    raise SchemaError(f"unknown solver {solver!r}")


def _bve_fields() -> tuple[FieldSpec, ...]:
    return (FieldSpec(
        "zeta", "s^-1",
        "relative vorticity; the BVE prognostic (docs/MATHEMATICAL_MODEL.md)",
        rows=None, levels=False, monopole="conserved"),)


def _swe_fields(gravity, mean_depth_m) -> tuple[FieldSpec, ...]:
    return (
        FieldSpec("zeta", "s^-1", "relative vorticity", rows=(0, 1)),
        FieldSpec("delta", "s^-1", "horizontal divergence", rows=(1, 2)),
        FieldSpec(
            "phi", "m^2 s^-2",
            "perturbation thickness geopotential phi: the layer-thickness "
            "geopotential is Phi = g*h = Phi0 + phi with Phi0 = gravity * "
            f"mean_depth_m = {gravity} * {mean_depth_m}; the free-surface "
            "geopotential adds the fixed surface geopotential phi_s over "
            "topography (physics/shallow_water.py)",
            rows=(2, 3)),
    )


def _pe_fields(nlev: int) -> tuple[FieldSpec, ...]:
    K = int(nlev)
    return (
        FieldSpec("zeta", "s^-1", "relative vorticity per full level, top "
                  "to bottom", rows=(0, K), levels=True),
        FieldSpec("delta", "s^-1", "horizontal divergence per full level, "
                  "top to bottom", rows=(K, 2 * K), levels=True),
        FieldSpec("temperature", "K", "full temperature per full level, top "
                  "to bottom (not a perturbation; the (0,0) monopole is the "
                  "horizontal mean times sqrt(4*pi))",
                  rows=(2 * K, 3 * K), levels=True,
                  monopole="horizontal-mean"),
        FieldSpec("ln_ps", "ln(Pa)", "natural logarithm of surface pressure "
                  "(single surface row, no level axis)",
                  rows=(3 * K, 3 * K + 1), levels=False,
                  monopole="horizontal-mean"),
    )


def _geometry(run_config: Mapping, *, inferred: dict) -> dict:
    grid = run_config.get("grid")
    if grid is None:
        grid = "geodesic"
        inferred["grid"] = "no 'grid' key; the only backend at the time"
    geometry = {"grid": grid}
    if grid == "latlon":
        geometry["nlat"] = run_config.get("nlat")
        geometry["nlon"] = run_config.get("nlon")
    else:
        geometry["resolution"] = run_config.get("resolution")
    geometry["radius_earth_units"] = run_config.get("radius_earth_units")
    geometry["day_hours"] = run_config.get("day_hours")
    if run_config.get("scenario") == "williamson5":
        geometry["planet"] = "ideal_sphere"
        geometry["reference_radius_m"] = W5_RADIUS_M
    else:
        geometry["planet"] = "from_earth_like"
        geometry["reference_radius_m"] = EARTH_RADIUS_M
    return geometry


def _environment(solver: str, run_config: Mapping) -> dict:
    keys = {
        "bve": ("scenario", "viscosity", "product_quadrature"),
        "swe": ("scenario", "gravity", "mean_depth_m", "topography",
                "mountain_height_m", "mountain_lat_deg", "mountain_lon_deg",
                "mountain_width_deg", "w5_u0_ms", "w5_cone_height_m",
                "w5_cone_radius_rad", "w5_cone_lat_deg", "w5_cone_lon_deg",
                "w5_projection", "w5_canonical"),
        "pe": ("scenario", "r_dry", "cp_dry", "temperature",
               "surface_pressure", "thermal_amplitude", "dt_seconds",
               "topography", "mountain_height_m", "mountain_lat_deg",
               "mountain_lon_deg", "mountain_width_deg", "gravity"),
    }[solver]
    env = {k: run_config[k] for k in keys if k in run_config}
    if solver in ("swe", "pe") and "topography" not in env:
        env["topography"] = "flat"
    return env


def state_schema_for(solver: str, run_config: Mapping, *,
                     provenance: Mapping | None = None) -> StateSchema:
    """Build the schema of a capsule from its resolved run configuration.

    Used by the writer (``provenance`` omitted: ``{"source": "manifest"}``)
    and by legacy inference (``provenance`` states what was inferred).
    """
    if solver not in SOLVERS:
        raise SchemaError(f"unknown solver {solver!r}")
    inferred: dict = {}
    l_max = run_config.get("lmax")
    if l_max is None:
        raise SchemaError("run configuration has no 'lmax'")
    l_max = int(l_max)
    geometry = _geometry(run_config, inferred=inferred)
    environment = _environment(solver, run_config)
    nlev = None
    vertical = None
    rows: int | None
    if solver == "bve":
        fields = _bve_fields()
        rows = None
        if "product_quadrature" not in run_config:
            environment["product_quadrature"] = "fine"
            inferred["product_quadrature"] = (
                "no 'product_quadrature' key; the Planet.generate default")
    elif solver == "swe":
        fields = _swe_fields(run_config.get("gravity"),
                             run_config.get("mean_depth_m"))
        rows = 3
    else:
        interfaces = run_config.get("sigma_interfaces")
        nlev = run_config.get("nlev")
        if interfaces is None and nlev is None:
            raise SchemaError(
                "PE run configuration has neither 'sigma_interfaces' nor "
                "'nlev'")
        if interfaces is None:
            nlev = int(nlev)
            interfaces = [k / nlev for k in range(nlev + 1)]
            inferred["sigma_interfaces"] = (
                "no 'sigma_interfaces' key; uniform SigmaGrid.uniform(nlev)")
        interfaces = [float(s) for s in interfaces]
        if nlev is None:
            nlev = len(interfaces) - 1
        nlev = int(nlev)
        if len(interfaces) != nlev + 1:
            raise SchemaError(
                f"PE nlev={nlev} disagrees with {len(interfaces)} sigma "
                "interfaces")
        full_levels = [0.5 * (interfaces[k] + interfaces[k + 1])
                       for k in range(nlev)]
        vertical = {
            "coordinate": "sigma = p / p_s (Lorenz staggering, top to bottom)",
            "interfaces": interfaces,
            "full_levels": full_levels,
        }
        fields = _pe_fields(nlev)
        rows = 3 * nlev + 1
    prov = {"source": "manifest"}
    if provenance is not None:
        prov = dict(provenance)
    if inferred:
        prov.setdefault("inferred_defaults", {}).update(inferred)
    return StateSchema(
        solver=solver, fields=fields, l_max=l_max,
        coefficient_file=COEFFICIENT_FILES[solver],
        time_file=TIME_FILES[solver], storage_axes=STORAGE_AXES[solver],
        rows=rows, nlev=nlev, vertical=vertical, geometry=geometry,
        environment=environment, provenance=prov)


# ---------------------------------------------------------------------------
# Legacy inference (known historical formats only)
# ---------------------------------------------------------------------------

def infer_solver(run_config: Mapping, manifest: Mapping | None,
                 present_files: set[str]) -> tuple[str, dict]:
    """Determine the solver of a capsule; say when it had to be inferred.

    Returns ``(solver, provenance)``. Explicit ``run_config['solver']`` is
    authoritative. The one known missing-solver format is the historical
    BVE capsule (``psx-bve`` before ``solver`` existed): recognised by its
    ``vorticity_coeffs.npy`` plus ``viscosity`` key. Anything else is an
    unknown layout and is refused rather than guessed.
    """
    explicit = run_config.get("solver")
    if explicit is not None:
        if explicit not in SOLVERS:
            raise SchemaError(f"run_config names unknown solver {explicit!r}")
        return explicit, {"source": "manifest"}
    if (COEFFICIENT_FILES["bve"] in present_files
            and "viscosity" in run_config
            and not any(COEFFICIENT_FILES[s] in present_files
                        for s in ("swe", "pe"))):
        return "bve", {
            "source": "inferred",
            "solver": "no 'solver' key: recognised as the historical "
                      "psx-bve capsule (vorticity_coeffs.npy + viscosity)",
        }
    raise SchemaError(
        "cannot determine the solver: run_config has no 'solver' key and the "
        "capsule does not match the known legacy BVE layout "
        f"(files present: {sorted(present_files) or 'none'})")


def infer_state_schema(solver: str, run_config: Mapping,
                       solver_provenance: Mapping) -> StateSchema:
    """Schema for a manifest that predates the ``state_schema`` block."""
    provenance = {
        "source": "inferred",
        "reason": "manifest has no state_schema block",
        "convention_source": {
            "bve": "run/bve/runner.py (vorticity_coeffs.npy: (time, l, m))",
            "swe": "run/swe/runner.py (swe_coeffs.npy: (time, [zeta, delta, "
                   "phi], l, m))",
            "pe": "run/pe/runner.py + manifest notes coefficient_ordering "
                  "(pe_coeffs.npy: (time, [zeta_1..K, delta_1..K, T_1..K, "
                  "ln_ps], l, m))",
        }[solver],
    }
    for key, value in solver_provenance.items():
        if key != "source":
            provenance[key] = value
    return state_schema_for(solver, run_config, provenance=provenance)


# ---------------------------------------------------------------------------
# Diagnostic definitions (docstrings of run/*/diagnostics.py, structured)
# ---------------------------------------------------------------------------

BVE_DIAGNOSTIC_COLUMNS = (
    "time_s", "dt_s", "step", "max_speed_ms", "cfl", "circulation", "energy",
    "enstrophy_rel", "enstrophy_abs", "zeta_max", "zeta_rms", "energy_l1",
    "high_l_enstrophy_frac", "roundtrip_residual")
SWE_DIAGNOSTIC_COLUMNS = (
    "time_s", "dt_s", "step", "max_wind_ms", "max_char_speed_ms", "cfl",
    "phi_total_min", "phi_total_max", "total_mass", "total_energy",
    "h_min_m", "eta_min_m", "eta_max_m", "terrain_max_m", "zeta_l2",
    "delta_l2", "phi_l2")
PE_DIAGNOSTIC_COLUMNS = (
    "time_s", "dt_s", "step", "t_min", "t_max", "ps_min", "ps_max",
    "max_wind_ms", "max_abs_zeta", "max_abs_delta", "max_char_speed_ms",
    "courant", "total_mass", "mass_rel_drift")

_BVE_COLUMN_MEANINGS = {
    "time_s": "model time (s)",
    "dt_s": "accepted step length (s)",
    "step": "accepted step index",
    "max_speed_ms": "max |u| over the state grid (m/s)",
    "cfl": "max_speed * dt / cfl_length_scale (advective Courant number)",
    "circulation": "R^2 sqrt(4 pi) Re a_00(zeta)",
    "energy": "1/2 R^4 sum_{l>=1} P_l(zeta) / (l(l+1))",
    "enstrophy_rel": "1/2 R^2 sum_l P_l(zeta)",
    "enstrophy_abs": "1/2 R^2 sum_l P_l(zeta + f), f a pure (1,0) mode",
    "zeta_max": "max |zeta| on the state grid (s^-1)",
    "zeta_rms": "RMS zeta on the state grid (s^-1)",
    "energy_l1": "energy in degree l = 1",
    "high_l_enstrophy_frac": "see the structured definition",
    "roundtrip_residual": "periodic transform round-trip residual",
}
_SWE_COLUMN_MEANINGS = {
    "time_s": "model time (s)", "dt_s": "accepted step length (s)",
    "step": "accepted step index",
    "max_wind_ms": "max |u| over the state grid (m/s)",
    "max_char_speed_ms": "max|u| + sqrt(max(Phi0 + phi)) over every model "
                         "sampling",
    "cfl": "max_char_speed * dt / cfl_length_scale",
    "phi_total_min": "min of Phi0 + phi over every model sampling",
    "phi_total_max": "max of Phi0 + phi over every model sampling",
    "total_mass": "R^2 (4 pi Phi0 + sqrt(4 pi) Re phi_00): layer mass proxy",
    "total_energy": "integral [Phi |u|^2/2 + Phi^2/2 + Phi phi_s] dA, "
                    "Phi = Phi0 + phi (g omitted)",
    "h_min_m": "phi_total_min / gravity (positivity margin, m)",
    "eta_min_m": "min free-surface anomaly (phi + phi_s')/g on the state grid",
    "eta_max_m": "max free-surface anomaly (phi + phi_s')/g on the state grid",
    "terrain_max_m": "max surface elevation over every model sampling (m)",
    "zeta_l2": "sqrt(integral zeta^2 dA)",
    "delta_l2": "sqrt(integral delta^2 dA)",
    "phi_l2": "sqrt(integral phi^2 dA)",
}
_PE_COLUMN_MEANINGS = {
    "time_s": "model time (s)", "dt_s": "accepted step length (s)",
    "step": "accepted step index",
    "t_min": "min temperature over every model sampling (K)",
    "t_max": "max temperature over every model sampling (K)",
    "ps_min": "min surface pressure on the state grid (Pa)",
    "ps_max": "max surface pressure on the state grid (Pa)",
    "max_wind_ms": "max |V| over the state grid, all levels (m/s)",
    "max_abs_zeta": "max |zeta| over the state grid, all levels (s^-1)",
    "max_abs_delta": "max |delta| over the state grid, all levels (s^-1)",
    "max_char_speed_ms": "max|V| + sqrt(gamma R_d T_max)",
    "courant": "max_char_speed * dt / cfl_length_scale (diagnostic only; "
               "the PE runner uses a fixed dt)",
    "total_mass": "integral p_s dA by state-grid quadrature (rho/g omitted)",
    "mass_rel_drift": "(total_mass - total_mass_0) / total_mass_0",
}


def high_l_enstrophy_fraction_definition(l_max: int) -> dict:
    """Structured definition of the BVE ``high_l_enstrophy_frac`` column.

    Mirrors ``run/bve/diagnostics.spectral_diagnostics`` exactly: the
    per-degree enstrophy ``Z_l = 1/2 R^2 sum_m P_lm`` is summed over the
    triangular domain ``0 <= m <= l <= l_max`` with positive-order
    multiplicity (``P_l0 = Re(a_l0)^2``, ``P_lm = 2 |a_lm|^2`` for m > 0);
    the fraction is ``sum_{l > cut} Z_l / sum_l Z_l`` with
    ``cut = product_truncation_cut(l_max)``, and exactly ``0.0`` when the
    total enstrophy is zero.
    """
    cut = product_truncation_cut(int(l_max))
    return {
        "version": 1,
        "definition": "sum_{l > cut} Z_l / sum_{l} Z_l",
        "band": {"degrees": f"cut < l <= l_max", "cut": cut,
                 "cut_rule": "product_truncation_cut(l_max) = "
                             "floor(2 * l_max / 3)", "l_max": int(l_max)},
        "domain": "triangular 0 <= m <= l <= l_max (m > l padding excluded)",
        "mode_power": {"m == 0": "Re(a_l0)^2",
                       "m > 0": "2 * |a_lm|^2 (positive-order multiplicity)"},
        "per_degree": "Z_l = 1/2 R^2 sum_m P_lm",
        "denominator": "sum over ALL degrees 0..l_max of Z_l",
        "zero_power": "0.0 when the denominator is zero",
        "source": "run/bve/diagnostics.py::spectral_diagnostics",
    }


def diagnostic_definitions_for(solver: str, run_config: Mapping) -> dict:
    """The ``diagnostic_definitions`` manifest block for one solver."""
    if solver not in SOLVERS:
        raise SchemaError(f"unknown solver {solver!r}")
    columns, meanings, source = {
        "bve": (BVE_DIAGNOSTIC_COLUMNS, _BVE_COLUMN_MEANINGS,
                "run/bve/diagnostics.py"),
        "swe": (SWE_DIAGNOSTIC_COLUMNS, _SWE_COLUMN_MEANINGS,
                "run/swe/diagnostics.py"),
        "pe": (PE_DIAGNOSTIC_COLUMNS, _PE_COLUMN_MEANINGS,
               "run/pe/diagnostics.py"),
    }[solver]
    block = {
        "version": DIAGNOSTIC_DEFINITIONS_VERSION,
        "solver": solver,
        "timeseries": "diagnostics/timeseries.csv (one row per accepted "
                      "step, columns in file order)",
        "columns": [{"name": name, "meaning": meanings[name]}
                    for name in columns],
        "source": source,
    }
    if solver == "bve":
        block["spectra"] = ("diagnostics/spectra.npz: times, energy_l, "
                            "enstrophy_l per recorded step")
        block["high_l_enstrophy_frac"] = (
            high_l_enstrophy_fraction_definition(int(run_config["lmax"])))
    return block


def provenance_blocks(solver: str, run_config: Mapping) -> dict:
    """Both additive manifest blocks, ready for ``write_run_manifest``."""
    return {
        "state_schema": state_schema_for(solver, run_config).to_manifest_dict(),
        "diagnostic_definitions": diagnostic_definitions_for(solver,
                                                             run_config),
    }


__all__ = [
    "BVE_DIAGNOSTIC_COLUMNS",
    "COEFFICIENT_FILES",
    "DIAGNOSTIC_DEFINITIONS_VERSION",
    "PE_DIAGNOSTIC_COLUMNS",
    "SOLVERS",
    "STATE_SCHEMA_VERSION",
    "STORAGE_AXES",
    "SWE_DIAGNOSTIC_COLUMNS",
    "SchemaError",
    "StateSchema",
    "TIME_FILES",
    "UnknownSchemaVersionError",
    "UnsupportedConventionError",
    "canonical_field_layout",
    "diagnostic_definitions_for",
    "high_l_enstrophy_fraction_definition",
    "infer_solver",
    "infer_state_schema",
    "provenance_blocks",
    "state_schema_for",
]
