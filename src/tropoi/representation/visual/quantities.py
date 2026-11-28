"""The plottable quantities of saved runs: meaning, units, cadence and needs.

A small, explicit catalogue. Every entry states whether it is a stored
prognostic, a quantity derived from the stored state, a static field
reconstructed from the run's configuration, or a column recorded per step
by the run. Derived entries name their recipe; ``needs`` says whether
evaluating it synthesizes fields through the model (CUDA) or runs on the
host.

This module is pure data: importing it touches neither CuPy nor
Matplotlib, so discovery (``Simulation.quantities``) works on any machine.
"""
from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "Quantity",
    "QuantityUnavailableError",
    "QUANTITIES",
    "available_quantities",
    "quantity",
    "viewer_presentation",
]

KINDS = ("prognostic", "derived", "static", "recorded")
SHAPES = ("scalar", "vector", "number", "series")


#: Catalogue groups of the viewer's quantity selector, in display order.
GROUPS = ("Thermodynamics", "Velocity and wind", "Dynamics", "Geopotential and geometry", "Surface",
          "Archived diagnostics and time series")
ARCHIVED = GROUPS[-1]


@dataclass(frozen=True)
class _Presentation:
    name: str
    symbol: str
    units: str
    role: str
    origin: str
    definition: str
    group: str = ""


# Public archive IDs remain usable for requests. These describe the values a
# person sees after host synthesis and any declared pointwise conversion.
_VIEWER_FIELDS = {
    "zeta": _Presentation("Relative vorticity", "ζ", "s⁻¹", "prognostic", "stored",
        "Vertical component of the curl of horizontal velocity, positive for local counterclockwise rotation viewed from above.", "Dynamics"),
    "delta": _Presentation("Horizontal divergence", "δ", "s⁻¹", "prognostic", "stored",
        "Horizontal divergence of velocity. Positive values indicate spreading flow.", "Dynamics"),
    "temperature": _Presentation("Air temperature", "T", "K", "prognostic", "stored",
        "Absolute air temperature on the selected atmospheric level or surface.", "Thermodynamics"),
    "pressure": _Presentation("Air pressure", "p", "Pa", "diagnostic", "derived",
        "Local air pressure at the selected atmospheric surface. On a native level p = σ pₛ.", "Thermodynamics"),
    "density": _Presentation("Air density", "ρ", "kg m⁻³", "diagnostic", "derived",
        "Mass density of the air from the ideal-gas law, ρ = p/(R T), with p and T evaluated at the same place, "
        "surface and saved time, and R the gas constant the run records for its atmosphere (the mixture constant "
        "of a fixed-composition gas; no moisture or variable composition). On a native level p = σ pₛ.",
        "Thermodynamics"),
    "geopotential": _Presentation("Atmospheric geopotential", "Φ", "m² s⁻²", "diagnostic", "derived",
        "Hydrostatic geopotential relative to the model datum, including the lower-boundary geopotential.", "Geopotential and geometry"),
    "geopotential_height": _Presentation("Geopotential height", "Z", "m", "diagnostic", "derived",
        "Z = Φ/g, the hydrostatic geopotential divided by model gravity, relative to the model height datum.", "Geopotential and geometry"),
    "ln_ps": _Presentation("Surface pressure", "pₛ", "Pa", "prognostic", "stored",
        "Pressure at the lower atmospheric boundary. The stored prognostic is ln(pₛ/Pa). The map shows its pointwise exponential in pascals.", "Surface"),
    "surface_temperature": _Presentation("Slab surface temperature", "Tₛ", "K", "prognostic", "stored",
        "Temperature of the surface slab at its saved snapshot time.", "Surface"),
    "phi": _Presentation("Layer-thickness geopotential anomaly", "φ", "m² s⁻²", "prognostic", "stored",
        "φ = g(h − h₀), with h the fluid thickness and h₀ the reference depth.", "Geopotential and geometry"),
    "layer_depth": _Presentation("Layer thickness", "h", "m", "diagnostic", "derived",
        "h = h₀ + φ/g, the depth of fluid above the terrain.", "Geopotential and geometry"),
    "terrain": _Presentation("Terrain height", "hₛ", "m", "static", "derived",
        "Band-limited lower-boundary elevation relative to the model height datum.", "Geopotential and geometry"),
    "free_surface_height": _Presentation("Free-surface geopotential height", "H", "m", "diagnostic", "derived",
        "H = h + hₛ, the geopotential height of the upper fluid surface. Under constant gravity H also gives its geometric elevation above the reference equipotential.", "Geopotential and geometry"),
    "free_surface_geopotential": _Presentation("Free-surface geopotential", "Φ_free", "m² s⁻²", "diagnostic", "derived",
        "Φ_free = gH = g(h + hₛ), the total geopotential at the free surface.", "Geopotential and geometry"),
    "wind": _Presentation("Horizontal wind", "(u, v)", "m s⁻¹", "diagnostic", "derived",
        "Horizontal velocity in the local eastward and northward basis.", "Velocity and wind"),
    "wind_speed": _Presentation("Wind speed", "|V|", "m s⁻¹", "diagnostic", "derived",
        "Magnitude of horizontal velocity, √(u² + v²).", "Velocity and wind"),
    "u": _Presentation("Eastward wind", "u", "m s⁻¹", "diagnostic", "derived",
        "Eastward component of horizontal velocity. Westward flow has negative values.", "Velocity and wind"),
    "v": _Presentation("Northward wind", "v", "m s⁻¹", "diagnostic", "derived",
        "Northward component of horizontal velocity. Southward flow has negative values.", "Velocity and wind"),
}

_SERIES_NAMES = {
    "mean_T": "Mean air temperature", "t_min": "Minimum air temperature",
    "t_max": "Maximum air temperature", "mean_ps": "Mean surface pressure",
    "mean_lnps": "Mean log surface pressure", "mean_surface_temperature": "Mean slab surface temperature",
    "max_abs_u": "Maximum absolute eastward wind", "max_wind_bottom": "Maximum bottom-level wind speed",
    "ke": "Kinetic energy per unit mass", "comp_mode": "Computational mode",
    "toa_net_down": "Net downward radiation at the top of atmosphere",
    "bottom_net_down": "Net downward radiation at the bottom of atmosphere",
    "olr": "Outgoing longwave radiation", "absorbed_stellar": "Absorbed stellar radiation",
    "bl_kinetic_sink": "Boundary-layer kinetic energy sink", "bl_max_depth": "Maximum boundary-layer depth",
}


def viewer_presentation(identifier: str, metadata: dict, *, recorded=False) -> dict:
    """Describe displayed values separately from source schema and evaluator IDs."""
    from dataclasses import asdict

    if recorded:
        member = metadata.get("member", identifier)
        name = _SERIES_NAMES.get(member, member.replace("_", " ").capitalize())
        return dict(name=name, symbol="", units=metadata.get("units"), role="diagnostic",
                    origin="recorded", definition=f"{name} recorded in {metadata['path']} ({member}).",
                    group=ARCHIVED)
    entry = _VIEWER_FIELDS.get(identifier)
    if entry is None:
        return dict(name=identifier, symbol="", units=metadata.get("units"), role="unspecified",
                    origin="stored", definition=metadata.get("meaning", ""), group="")
    result = asdict(entry)
    if identifier == "terrain" and "resources.terrain" in metadata.get("inputs", []):
        result["origin"] = "stored"
    if identifier == "ln_ps":
        result["transform"] = "exp"
    return result


class QuantityUnavailableError(ValueError):
    """The quantity is not defined for, or not stored by, this run."""


@dataclass(frozen=True)
class Quantity:
    id: str
    long_name: str
    units: str
    kind: str
    shape: str
    solvers: tuple[str, ...]
    definition: str
    #: ``"snapshot"`` (at saved states only), ``"static"`` or ``"every step"``.
    cadence: str = "snapshot"
    #: ``"cuda"`` when evaluation synthesizes through the model, else ``"host"``.
    needs: str = "cuda"
    #: True when the PE value depends on a vertical level (``level=``).
    per_level_in_pe: bool = False
    #: Why the quantity is not defined for a solver not listed in ``solvers``.
    unsupported_reason: dict | None = None

    def __post_init__(self) -> None:
        if self.kind not in KINDS or self.shape not in SHAPES:
            raise ValueError(f"invalid catalogue entry {self.id!r}")

    def check_solver(self, solver: str) -> None:
        if solver in self.solvers:
            return
        reason = (self.unsupported_reason or {}).get(solver)
        raise QuantityUnavailableError(
            f"{self.id!r} is not available for {solver.upper()} runs"
            + (f": {reason}" if reason else
               f"; it is defined for {', '.join(self.solvers)}"))


_NON_DIVERGENT = ("the barotropic vorticity equation is non-divergent, so "
                  "this is identically zero by construction")

QUANTITIES: dict[str, Quantity] = {q.id: q for q in (
    Quantity("vorticity", "relative vorticity", "s^-1", "prognostic",
             "scalar", ("bve", "swe", "pe"), "zeta (stored)",
             per_level_in_pe=True),
    Quantity("divergence", "horizontal divergence", "s^-1", "prognostic",
             "scalar", ("swe", "pe"), "delta (stored)", per_level_in_pe=True,
             unsupported_reason={"bve": _NON_DIVERGENT}),
    Quantity("streamfunction", "streamfunction", "m^2 s^-1", "derived",
             "scalar", ("bve", "swe", "pe"),
             "psi_lm = -R^2 zeta_lm / (l(l+1)), l = 0 set to 0; in SWE and "
             "PE this is the rotational part of the flow only",
             per_level_in_pe=True),
    Quantity("velocity_potential", "velocity potential", "m^2 s^-1",
             "derived", "scalar", ("swe", "pe"),
             "chi_lm = -R^2 delta_lm / (l(l+1)), l = 0 set to 0 (divergent "
             "part of the flow)", per_level_in_pe=True,
             unsupported_reason={"bve": _NON_DIVERGENT}),
    Quantity("wind", "horizontal wind", "m s^-1", "derived", "vector",
             ("bve", "swe", "pe"),
             "u = k x grad(psi) + grad(chi) (full flow; chi = 0 in BVE)",
             per_level_in_pe=True),
    Quantity("wind_speed", "wind speed", "m s^-1", "derived", "scalar",
             ("bve", "swe", "pe"), "|u| of the full horizontal wind",
             per_level_in_pe=True),
    Quantity("layer_depth", "layer depth", "m", "derived", "scalar",
             ("swe",), "h = (Phi0 + phi) / g, the fluid thickness"),
    Quantity("free_surface_height", "free-surface height", "m", "derived",
             "scalar", ("swe",),
             "H = h + h_s: layer depth plus band-limited terrain height (equal "
             "to the layer depth over a flat bottom)"),
    Quantity("free_surface_perturbation", "free-surface perturbation", "m",
             "derived", "scalar", ("swe",),
             "eta' = H - H_ref, H_ref = area mean of H at the same saved time: "
             "the level surface the same fluid volume has at rest over this "
             "terrain (lake at rest: eta' = 0)"),
    Quantity("terrain", "terrain height", "m", "static", "scalar",
             ("swe", "pe"),
             "h_s = Phi_s / g, the band-limited surface elevation "
             "reconstructed from the run configuration", cadence="static",
             unsupported_reason={"bve": "the BVE core has no bottom "
                                        "topography"}),
    Quantity("temperature", "temperature", "K", "prognostic", "scalar",
             ("pe",), "T at a full sigma level (stored)",
             per_level_in_pe=True),
    Quantity("temperature_anomaly", "temperature anomaly", "K", "derived",
             "scalar", ("pe",),
             "T minus its area mean at the same sigma level (the (0,0) mode "
             "removed)", per_level_in_pe=True),
    Quantity("surface_pressure_anomaly", "surface-pressure anomaly", "hPa",
             "derived", "scalar", ("pe",),
             "p_s = exp(ln p_s) minus its area-weighted mean"),
    Quantity("potential_enstrophy", "potential enstrophy", "m s^-2",
             "derived", "number", ("swe",),
             "Z = integral (zeta + f)^2 / (2 h) dA by state-grid quadrature "
             "(per unit density; evaluated at saved snapshots only)"),
    Quantity("spectral_complexity", "kinetic-energy spectral complexity",
             "1", "derived", "number", ("bve", "swe", "pe"),
             "mean degree, mean zonal wavenumber and effective mode count "
             "exp(S) of the modal distribution of 1/2 integral |u|^2 dA "
             "(representation.diagnostics.spectral)", needs="host",
             per_level_in_pe=True),
)}


def quantity(identifier: str) -> Quantity:
    try:
        return QUANTITIES[identifier]
    except KeyError:
        raise QuantityUnavailableError(
            f"unknown quantity {identifier!r}; known quantities: "
            f"{', '.join(sorted(QUANTITIES))}") from None


def available_quantities(solver: str, *, run_config=None,
                         diagnostic_columns=(), diagnostics_present=False
                         ) -> list[dict]:
    """Rows describing what one run can provide (host only, no evaluation)."""
    run_config = dict(run_config or {})
    flat = run_config.get("topography", "flat") in (None, "flat")
    rows = []
    for entry in QUANTITIES.values():
        available, note = solver in entry.solvers, ""
        if not available:
            note = (entry.unsupported_reason or {}).get(
                solver, f"defined for {', '.join(entry.solvers)}")
        elif entry.id == "terrain" and flat:
            available, note = False, "flat bottom in this run"
        rows.append({"id": entry.id, "long_name": entry.long_name,
                     "units": entry.units, "kind": entry.kind,
                     "shape": entry.shape, "cadence": entry.cadence,
                     "needs": entry.needs,
                     "levels": solver == "pe" and entry.per_level_in_pe,
                     "available": available, "note": note,
                     "definition": entry.definition})
    for column in diagnostic_columns:
        rows.append({"id": column["name"], "long_name": column["meaning"],
                     "units": "see meaning", "kind": "recorded",
                     "shape": "series", "cadence": "every step",
                     "needs": "host", "levels": False,
                     "available": diagnostics_present,
                     "note": "" if diagnostics_present else
                     "diagnostics/timeseries.csv not found",
                     "definition": "diagnostics/timeseries.csv column"})
    return rows
