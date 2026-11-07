"""Sky seen by an observer: a single-scattering optics laboratory (CPU).

The standalone sky page (``sky_view.html``) and the NumPy reference of what
it computes. The page repeats the same arithmetic in float64 JavaScript, so
the two agree to rounding; ``tests/test_sky_view_browser.py`` checks that.

Two entry points, neither needs CUDA:

* :func:`write_sky_page` -- a labelled HYPOTHETICAL atmosphere from explicit
  composition, surface pressure, temperature (profile), gravity, gas
  constant, planetary radius and star. It needs no saved run, no prepared
  planet and no terrain, and imports nothing from those parts of tropoi.
  In the page every one of those inputs can be changed.
* :func:`saved_sky_page` (``Simulation.sky``) -- the column of one saved
  primitive-equation state above one location, reconstructed exactly on the
  CPU. The column is fixed in the page; composition, star, observer height,
  view and exposure stay adjustable.

Physics (stated in the page; this is NOT a validated sky colour):

* a horizontally uniform, spherically symmetric column above the observer
  (:class:`~tropoi.spatial.optics.AtmosphericColumn`), of radius R + z;
* Rayleigh scattering only, by an explicit
  :class:`~tropoi.spatial.optics.Composition` (no mixture is ever
  inferred), with the phase function 3/(16 pi) (1 + cos^2 Theta)
  (depolarization neglected in the phase function, kept in the King
  factor of the cross section);
* single scattering of a blackbody star (:class:`~tropoi.spatial.optics.Star`)
  with attenuation along the view and star paths; the solid sphere at the
  column base shadows star paths and ends view paths;
* spectral bins of 20 nm from 400 to 760 nm (inside every refractive-index
  record), converted to CIE XYZ with the packaged CIE 1931 table, then
  exposure and the sRGB transfer function; out-of-gamut and over-exposed
  values are clipped and counted.

Exposure. ``auto`` divides by the brightest sky luminance of the current
table (so a change of pressure or star brightness is rescaled away);
``fixed`` divides by a reference luminance in cd m^-2 that stays put, so
brightness differences between settings remain visible. The page shows
which is active.

Optical depth. The page reports the Rayleigh optical depth above the
observer (zenith) and along the path to the star, and warns when single
scattering becomes unreliable (:data:`TAU_CAUTION`, :data:`TAU_UNRELIABLE`:
rules of thumb, not derived error bounds).

Numerics. Directions: a table over view zenith angle (1 degree) and
azimuth relative to the star (3 degrees; single scattering in a spherically
symmetric column is symmetric about the star's vertical plane). Each view
ray is cut at its lowest point and sampled with quadratically graded
midpoint steps that are densest there (``VIEW_STEPS``); each star path from
a sample likewise (``SUN_STEPS``). Number density is read from a uniform
table of ln n in altitude (``DENSITY_NODES``).
"""
from __future__ import annotations

import importlib.resources
import json
import math
import pathlib
import re
import tempfile
from dataclasses import dataclass, field

import numpy as np

from tropoi.spatial.optics import (SPECIES, AtmosphericColumn, Composition,
                                   Star, gas_constant_report)
from tropoi.spatial.optics.colour import (CIE_SOURCE, XYZ_TO_LINEAR_SRGB,
                                          bin_weights, omitted_fraction)
from tropoi.spatial.optics.column import DEFAULT_TOP_RATIO
from tropoi.spatial.optics.composition import R_DRY_TOLERANCE, SUM_TOLERANCE
from tropoi.spatial.optics.constants import (ATOMIC_WEIGHT_SOURCE, C, H,
                                             K_B, LUMINOUS_EFFICACY, R_MOLAR)
from tropoi.spatial.optics.rayleigh import KING_CHECK

__all__ = ["BIN_EDGES_NM", "EXPOSURE_MODES", "HypotheticalColumn", "SkySpec",
           "TAU_CAUTION", "TAU_UNRELIABLE", "UNSUPPORTED", "density_grid",
           "optical_depth_diagnostic", "radiance_table", "saved_sky_page",
           "sky_block", "star_path_column", "write_sky_page"]

BIN_EDGES_NM = np.arange(400.0, 761.0, 20.0)
WAVELENGTHS_NM = 0.5 * (BIN_EDGES_NM[:-1] + BIN_EDGES_NM[1:])
N_ZENITH = 181                  # 0..180 deg, 1 deg
N_AZIMUTH = 61                  # 0..180 deg from the star, 3 deg
VIEW_STEPS = 40
SUN_STEPS = 20
DENSITY_NODES = 2048
HYPOTHETICAL_NODES = 64         # AtmosphericColumn.hypothetical's default
GROUND_OFFSET_M = 2.0           # "on the ground" = this far above it
EXPOSURE_MODES = ("auto", "fixed")
#: Zenith Rayleigh optical depth (largest over the bins) from which the page
#: cautions that multiple scattering is significant, and from which it calls
#: single scattering unreliable. Rules of thumb, not derived error bounds.
TAU_CAUTION = 0.1
TAU_UNRELIABLE = 1.0
PROTOTYPE_LABEL = ("Visual prototype — single-scattering Rayleigh sky; not a "
                   "validated sky colour")

UNSUPPORTED = (
    "aerosols, dust and haze (on Mars and Titan they dominate the real sky "
    "colour)",
    "clouds", "gas absorption (ozone, CO₂, CH₄, H₂O and others)",
    "multiple scattering", "refraction", "polarization",
    "surface reflection and a lit ground (the ground is drawn unlit)",
    "terrain shadows and a terrain horizon (only the base sphere occludes)",
    "limb darkening and non-blackbody stellar spectra",
    "horizontal variation of the atmosphere (one column at the observer)",
    "night-sky sources", "a star position from time and an orbit",
)


@dataclass(frozen=True)
class HypotheticalColumn:
    """A user-prescribed, labelled column (no simulation behind it).

    ``surface_pressure_pa`` is at the reference sphere of radius
    ``radius_m`` (altitude 0). ``gas_constant`` is a number
    (J kg^-1 K^-1) or the explicit string ``"from_composition"``
    (R*/M of the sky's composition). Isothermal at ``temperature_k``, or
    a constant ``lapse_rate_k_per_m`` down to ``temperature_floor_k`` and
    isothermal above.
    """

    surface_pressure_pa: float
    temperature_k: float
    gravity: float
    gas_constant: float | str
    radius_m: float
    lapse_rate_k_per_m: float = 0.0
    temperature_floor_k: float | None = None

    def __post_init__(self) -> None:
        r = float(self.radius_m)
        if not (math.isfinite(r) and r > 0.0):
            raise ValueError(f"radius_m must be finite and > 0, got {r}")
        if isinstance(self.gas_constant, str) and \
                self.gas_constant != "from_composition":
            raise ValueError("gas_constant must be a number or "
                             "'from_composition'")

    def gas_constant_for(self, composition: Composition) -> float:
        if isinstance(self.gas_constant, str):
            return composition.gas_constant
        return float(self.gas_constant)

    def build(self, composition: Composition) -> AtmosphericColumn:
        return AtmosphericColumn.hypothetical(
            surface_pressure_pa=self.surface_pressure_pa,
            temperature_k=self.temperature_k, gravity=self.gravity,
            gas_constant=self.gas_constant_for(composition),
            lapse_rate_k_per_m=self.lapse_rate_k_per_m,
            temperature_floor_k=self.temperature_floor_k,
            nodes=HYPOTHETICAL_NODES)

    def to_dict(self) -> dict:
        return {"surface_pressure_pa": float(self.surface_pressure_pa),
                "temperature_k": float(self.temperature_k),
                "gravity": float(self.gravity),
                "gas_constant": (self.gas_constant
                                 if isinstance(self.gas_constant, str)
                                 else float(self.gas_constant)),
                "radius_m": float(self.radius_m),
                "lapse_rate_k_per_m": float(self.lapse_rate_k_per_m),
                "temperature_floor_k": (None if self.temperature_floor_k
                                        is None else
                                        float(self.temperature_floor_k)),
                "top_ratio": DEFAULT_TOP_RATIO,
                "nodes": HYPOTHETICAL_NODES}


@dataclass(frozen=True)
class SkySpec:
    """What the sky page starts with. ``composition`` is required.

    The observer is EITHER ``altitude_m`` above the reference sphere OR at
    ``pressure_pa``; with neither, 2 m above the ground. A saved run's sky
    also needs ``observer_lat_deg``/``observer_lon_deg`` (which column); a
    hypothetical column is horizontally uniform, so it refuses them. Angles
    in degrees; azimuth clockwise from north. ``exposure_mode`` is
    ``"auto"`` (white = brightest sky luminance) or ``"fixed"`` (white =
    ``exposure_reference_cd_m2``).
    """

    composition: Composition
    star: Star = field(default_factory=Star.sun_at_1au)
    observer_lat_deg: float | None = None
    observer_lon_deg: float | None = None
    altitude_m: float | None = None
    pressure_pa: float | None = None
    look_azimuth_deg: float = 180.0
    look_elevation_deg: float = 20.0
    fov_deg: float = 90.0
    star_azimuth_deg: float = 135.0
    star_elevation_deg: float = 30.0
    exposure_ev: float = 0.0
    exposure_mode: str = "auto"
    exposure_reference_cd_m2: float | None = None
    hypothetical: HypotheticalColumn | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.composition, Composition):
            raise TypeError("SkySpec needs an explicit Composition; no gas "
                            "mixture is ever inferred")
        if not isinstance(self.star, Star):
            raise TypeError("star must be a Star")
        if self.altitude_m is not None and self.pressure_pa is not None:
            raise ValueError("give altitude_m or pressure_pa, not both")
        if (self.observer_lat_deg is None) != (self.observer_lon_deg is None):
            raise ValueError("give both observer_lat_deg and "
                             "observer_lon_deg, or neither")
        if self.hypothetical is not None:
            if not isinstance(self.hypothetical, HypotheticalColumn):
                raise TypeError("hypothetical must be a HypotheticalColumn")
            if self.observer_lat_deg is not None:
                raise ValueError(
                    "a hypothetical column is horizontally uniform: the "
                    "observer's latitude and longitude would have no effect, "
                    "so they are not accepted")
        if self.observer_lat_deg is not None and \
                not -90.0 <= float(self.observer_lat_deg) <= 90.0:
            raise ValueError("observer_lat_deg must be in [-90, 90]")
        for name in ("look_elevation_deg", "star_elevation_deg"):
            if not -90.0 <= float(getattr(self, name)) <= 90.0:
                raise ValueError(f"{name} must be in [-90, 90]")
        if not 10.0 <= float(self.fov_deg) <= 150.0:
            raise ValueError("fov_deg must be in [10, 150]")
        if self.exposure_mode not in EXPOSURE_MODES:
            raise ValueError(f"exposure_mode must be one of {EXPOSURE_MODES}")
        ref = self.exposure_reference_cd_m2
        if self.exposure_mode == "fixed":
            if ref is None or not (math.isfinite(float(ref))
                                   and float(ref) > 0.0):
                raise ValueError("a fixed exposure needs "
                                 "exposure_reference_cd_m2 > 0 (the "
                                 "luminance drawn as white at 0 EV)")
        elif ref is not None:
            raise ValueError("exposure_reference_cd_m2 is for "
                             "exposure_mode='fixed'; 'auto' uses the "
                             "brightest sky luminance")


# ---------------------------------------------------------------------------
# Reference computation (the page mirrors it)
# ---------------------------------------------------------------------------

def density_grid(z_m, ln_p, temperature_k, nodes: int = DENSITY_NODES):
    """(base, dz, ln n[nodes]) on a uniform altitude grid over the column."""
    z = np.asarray(z_m, dtype=np.float64)
    base, top = float(z[0]), float(z[-1])
    dz = (top - base) / (nodes - 1)
    zg = base + dz * np.arange(nodes)
    lp = np.interp(zg, z, ln_p)
    t = np.interp(zg, z, temperature_k)
    return base, dz, lp - np.log(K_B * t)


def _density(z, base, dz, lnn):
    x = (z - base) / dz
    i = np.clip(np.floor(x), 0, lnn.size - 2).astype(np.int64)
    f = x - i
    out = np.exp(lnn[i] + f * (lnn[i + 1] - lnn[i]))
    return np.where((x >= 0.0) & (x <= lnn.size - 1), out, 0.0)


def _segments(r0, mu, length, n):
    """Midpoints and widths of n graded steps on [0, length] (vectorized
    over rays), densest at the ray's lowest point s_low = clamp(-r0 mu)."""
    r0, mu, length = np.broadcast_arrays(r0, mu, length)
    s_low = np.clip(-r0 * mu, 0.0, length)
    frac = np.where(length > 0, s_low / np.where(length > 0, length, 1), 0)
    na = np.where(s_low <= 0.0, 0,
                  np.where(s_low >= length, n,
                           np.clip(np.floor(n * frac + 0.5), 1, n - 1)))
    j = np.arange(n + 1)
    shape = r0.shape + (n + 1,)
    na_ = na[..., None]
    nb_ = (n - na)[..., None]
    sl = s_low[..., None]
    ln = length[..., None]
    jj = np.broadcast_to(j, shape)
    ta = np.where(na_ > 0, jj / np.where(na_ > 0, na_, 1), 0.0)
    ea = sl * (1.0 - (1.0 - ta) ** 2)
    tb = np.where(nb_ > 0, (jj - na_) / np.where(nb_ > 0, nb_, 1), 0.0)
    eb = sl + (ln - sl) * tb ** 2
    edges = np.where(jj <= na_, ea, eb)
    mid = 0.5 * (edges[..., 1:] + edges[..., :-1])
    width = edges[..., 1:] - edges[..., :-1]
    return mid, width


def radiance_table(column_nodes, *, radius_m: float, observer_z_m: float,
                   star_elevation_deg: float, cross_section_m2, irradiance,
                   n_zenith: int = N_ZENITH, n_azimuth: int = N_AZIMUTH,
                   view_steps: int = VIEW_STEPS, sun_steps: int = SUN_STEPS,
                   density_nodes: int = DENSITY_NODES) -> dict:
    """Single-scattering spectral radiance over the direction table.

    ``column_nodes`` = (z_m, ln_p, temperature_k). Returns ``radiance``
    (n_zenith, n_azimuth, bins) in W m^-2 sr^-1 nm^-1, ``column`` (total
    molecules m^-2 along each view ray; ``column[0, 0]`` is the zenith
    column above the observer), ``ground`` (ray ends on the base sphere),
    and the grids.
    """
    z_m, ln_p, temp = (np.asarray(a, dtype=np.float64) for a in column_nodes)
    base, dz, lnn = density_grid(z_m, ln_p, temp, density_nodes)
    top = base + dz * (density_nodes - 1)
    rb, rt = radius_m + base, radius_m + top
    ro = radius_m + float(observer_z_m)
    sig = np.asarray(cross_section_m2, dtype=np.float64)
    irr = np.asarray(irradiance, dtype=np.float64)
    zen = np.radians(np.linspace(0.0, 180.0, n_zenith))
    azi = np.radians(np.linspace(0.0, 180.0, n_azimuth))
    th, ph = np.meshgrid(zen, azi, indexing="ij")
    mu, st = np.cos(th), np.sin(th)
    e = math.radians(star_elevation_deg)
    ce, se = math.cos(e), math.sin(e)
    cos_big = st * np.cos(ph) * ce + mu * se
    disc_g = rb * rb - ro * ro * (1.0 - mu * mu)
    ground = (mu < 0.0) & (disc_g >= 0.0)
    length = np.where(
        ground, -ro * mu - np.sqrt(np.maximum(disc_g, 0.0)),
        -ro * mu + np.sqrt(np.maximum(rt * rt - ro * ro * (1.0 - mu * mu),
                                      0.0)))
    length = np.maximum(length, 0.0)
    s, ds = _segments(np.full(th.shape, ro), mu, length, view_steps)
    rp = np.sqrt(np.maximum(ro * ro + s * s + 2.0 * ro * s * mu[..., None],
                            0.0))
    n = _density(rp - radius_m, base, dz, lnn)
    mus = (s * (st * np.cos(ph) * ce + mu * se)[..., None] + ro * se) / rp
    disc_s = rb * rb - rp * rp * (1.0 - mus * mus)
    blocked = (mus < 0.0) & (disc_s >= 0.0)
    ls = np.maximum(-rp * mus + np.sqrt(np.maximum(
        rt * rt - rp * rp * (1.0 - mus * mus), 0.0)), 0.0)
    q, dq = _segments(rp, mus, ls, sun_steps)
    rq = np.sqrt(np.maximum(rp[..., None] ** 2 + q * q
                            + 2.0 * rp[..., None] * q * mus[..., None], 0.0))
    nsun = (_density(rq - radius_m, base, dz, lnn) * dq).sum(axis=-1)
    step = n * ds
    nv = np.cumsum(step, axis=-1) - 0.5 * step
    tau = sig[None, None, None, :] * (nv + nsun)[..., None]
    contrib = np.where(blocked[..., None], 0.0,
                       step[..., None] * np.exp(-tau))
    phase = 3.0 / (16.0 * math.pi) * (1.0 + cos_big ** 2)
    rad = contrib.sum(axis=2) * sig * irr * phase[..., None]
    return {"radiance": rad, "column": step.sum(axis=-1), "ground": ground,
            "zenith_deg": np.degrees(zen), "azimuth_deg": np.degrees(azi)}


def star_path_column(column_nodes, *, radius_m: float, observer_z_m: float,
                     star_elevation_deg: float, view_steps: int = VIEW_STEPS,
                     density_nodes: int = DENSITY_NODES) -> float | None:
    """Molecules m^-2 from the observer to the top along the star's
    direction (the star disk's attenuation); ``None`` when the solid sphere
    hides the star."""
    z_m, ln_p, temp = (np.asarray(a, dtype=np.float64) for a in column_nodes)
    base, dz, lnn = density_grid(z_m, ln_p, temp, density_nodes)
    rb, rt = radius_m + base, radius_m + base + dz * (density_nodes - 1)
    ro = radius_m + float(observer_z_m)
    se = math.sin(math.radians(star_elevation_deg))
    if se < 0.0 and rb * rb - ro * ro * (1.0 - se * se) >= 0.0:
        return None
    length = max(-ro * se + math.sqrt(max(rt * rt - ro * ro * (1 - se * se),
                                          0.0)), 0.0)
    s, ds = _segments(np.array(ro), np.array(se), np.array(length),
                      view_steps)
    r = np.sqrt(np.maximum(ro * ro + s * s + 2.0 * ro * s * se, 0.0))
    return float((_density(r - radius_m, base, dz, lnn) * ds).sum())


def optical_depth_diagnostic(cross_section_m2, zenith_column: float,
                             star_column: float | None) -> dict:
    """Rayleigh optical depths per bin and the single-scattering verdict.

    ``level`` is ``"ok"``, ``"caution"`` (largest zenith optical depth >=
    :data:`TAU_CAUTION`) or ``"unreliable"`` (>= :data:`TAU_UNRELIABLE`).
    """
    sig = np.asarray(cross_section_m2, dtype=np.float64)
    tz = sig * float(zenith_column)
    ts = None if star_column is None else sig * float(star_column)
    peak = float(tz.max())
    level = ("unreliable" if peak >= TAU_UNRELIABLE else
             "caution" if peak >= TAU_CAUTION else "ok")
    return {"tau_zenith": tz.tolist(),
            "tau_star": None if ts is None else ts.tolist(),
            "peak_zenith": peak, "level": level}


def xyz(radiance, weights) -> np.ndarray:
    """CIE XYZ (Y in cd m^-2) of binned spectral radiance (..., bins)."""
    return LUMINOUS_EFFICACY * np.asarray(radiance) @ np.asarray(weights)


# ---------------------------------------------------------------------------
# Page block
# ---------------------------------------------------------------------------

def _optics_block() -> dict:
    lo, hi = BIN_EDGES_NM[:-1], BIN_EDGES_NM[1:]
    return {
        "wavelengths_nm": WAVELENGTHS_NM.tolist(),
        "bin_edges_nm": BIN_EDGES_NM.tolist(),
        "cmf_weights": bin_weights(lo, hi).tolist(),
        "cmf_omitted_fraction": omitted_fraction(400.0, 760.0).tolist(),
        "luminous_efficacy": LUMINOUS_EFFICACY,
        "xyz_to_linear_srgb": XYZ_TO_LINEAR_SRGB.tolist(),
        "planck": {"h": H, "c": C, "k_b": K_B},
    }


def _species_block() -> dict:
    """Every gas with optical data: its cross sections at the bin centres
    (computed here from the sourced records) and molar mass; the page mixes
    them as sum x_i sigma_i, exactly as ``Composition.cross_section``."""
    return {name: {"cross_section_m2": sp.cross_section(WAVELENGTHS_NM)
                   .tolist(),
                   "molar_mass_g_per_mol": sp.molar_mass_g_per_mol,
                   "source": sp.describe()}
            for name, sp in SPECIES.items()}


def sky_block(spec: SkySpec, *, column: AtmosphericColumn,
              column_kind: str, radius_m: float, r_dry: float | None,
              terrain_m: float = 0.0, context: dict | None = None) -> dict:
    """The page's ``sky`` block (validated: the initial observer must fit).

    ``column`` is the initial column (hypothetical: built from
    ``spec.hypothetical``; saved state: the reconstructed one, fixed in the
    page). ``context`` describes a saved column (location, time, run).
    """
    if column_kind not in ("hypothetical", "saved_state"):
        raise ValueError(f"unknown column kind {column_kind!r}")
    if spec.altitude_m is None and spec.pressure_pa is None:
        obs = column.observer_altitude(
            altitude_m=max(terrain_m, column.base_m) + GROUND_OFFSET_M,
            terrain_m=terrain_m)
        obs["mode"] = "ground"
    else:
        obs = column.observer_altitude(altitude_m=spec.altitude_m,
                                       pressure_pa=spec.pressure_pa,
                                       terrain_m=terrain_m)
    hyp = spec.hypothetical
    report = gas_constant_report(
        spec.composition, r_dry,
        against=("the run uses r_dry" if column_kind == "saved_state"
                 else "the hypothetical column uses R"))
    om = omitted_fraction(400.0, 760.0)
    notes = [
        PROTOTYPE_LABEL + ".",
        ("Column: " + ("one column of the saved state above the observer"
                       if column_kind == "saved_state" else
                       "HYPOTHETICAL, prescribed by the user; horizontally "
                       "uniform, so the observer's location has no effect")
         + ". " + " ".join(column.notes)),
        ("The composition supplies only the scattering cross section per "
         "molecule (σ = Σ xᵢσᵢ); the number density is p/(k_B T). "
         + ("The column's gas constant follows the composition "
            "(\"from_composition\")."
            if hyp is not None and isinstance(hyp.gas_constant, str) else
            "The column's gas constant is fixed; the page reports whether "
            "the composition agrees with it and never changes it.")),
        "Star: a uniform blackbody disk, no limb darkening.",
        ("Spectrum: 18 bins of 20 nm, 400–760 nm; the CIE functions outside "
         f"that range (x̄ {om[0] * 100:.2g} %, ȳ {om[1] * 100:.2g} %, z̄ "
         f"{om[2] * 100:.2g} % of their integrals) are omitted."),
        ("Exposure: displayed linear sRGB = M·XYZ / L_white · 2^EV, then the "
         "sRGB transfer function. AUTO: L_white is the brightest sky "
         "luminance in the direction table (star disk excluded), so changes "
         "of pressure or illumination are rescaled away. FIXED: L_white is a "
         "reference luminance you set, so brightness differences stay "
         "visible. Clipped pixels are counted."),
        (f"Optical depth: the page warns from a zenith Rayleigh optical "
         f"depth of {TAU_CAUTION:g} (multiple scattering, not modelled, "
         f"becomes significant) and calls single scattering unreliable from "
         f"{TAU_UNRELIABLE:g}. These are rules of thumb, not error bounds."),
        ("The star stays a uniform disk of its true angular size, drawn at "
         "least one pixel in radius (then marked enlarged)."),
    ]
    sources = ([sp["source"] for sp in _species_block().values()]
               + [KING_CHECK, f"Molar masses: {ATOMIC_WEIGHT_SOURCE}",
                  CIE_SOURCE, "XYZ → sRGB: IEC 61966-2-1", spec.star.source])
    return {
        "version": 2,
        "label": PROTOTYPE_LABEL,
        "column_kind": column_kind,
        "radius_m": float(radius_m),
        "hypothetical": None if hyp is None else hyp.to_dict(),
        "column": column.to_dict(),
        "terrain_m": float(terrain_m),
        "r_dry": None if r_dry is None else float(r_dry),
        "context": context,
        "composition": spec.composition.to_dict(),
        "composition_rules": {"sum_tolerance": SUM_TOLERANCE,
                              "r_tolerance": R_DRY_TOLERANCE,
                              "r_molar": R_MOLAR},
        "consistency": report,
        "species": _species_block(),
        "star": dict(spec.star.to_dict(), azimuth_deg=spec.star_azimuth_deg,
                     elevation_deg=spec.star_elevation_deg),
        "optics": _optics_block(),
        "observer": {"ground_offset_m": GROUND_OFFSET_M, **obs},
        "look": {"azimuth_deg": float(spec.look_azimuth_deg),
                 "elevation_deg": float(spec.look_elevation_deg),
                 "fov_deg": float(spec.fov_deg)},
        "exposure": {"mode": spec.exposure_mode,
                     "ev": float(spec.exposure_ev),
                     "reference_cd_m2": (None if spec.exposure_reference_cd_m2
                                         is None else
                                         float(spec.exposure_reference_cd_m2))},
        "diagnostic": {"tau_caution": TAU_CAUTION,
                       "tau_unreliable": TAU_UNRELIABLE},
        "numerics": {"n_zenith": N_ZENITH, "n_azimuth": N_AZIMUTH,
                     "view_steps": VIEW_STEPS, "sun_steps": SUN_STEPS,
                     "density_nodes": DENSITY_NODES},
        "notes": notes,
        "unsupported": list(UNSUPPORTED),
        "sources": sources,
    }


def _write_page(block: dict, title: str, path) -> pathlib.Path:
    template = importlib.resources.files(
        "tropoi.representation.visual").joinpath(
        "sky_view.html").read_text(encoding="utf-8")
    data = json.dumps(block, allow_nan=False, separators=(",", ":"))
    data = data.replace("<", "\\u003c").replace(">", "\\u003e")
    title = (str(title).replace("&", "&amp;").replace("<", "&lt;")
             .replace(">", "&gt;"))
    page = (template.replace("__PAGE_TITLE__", title)
            .replace("__SKY_DATA__", data))
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(page, encoding="utf-8")
    return path


def _default_path(name: str) -> pathlib.Path:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", name).strip("-").lower()[:48]
    return pathlib.Path(tempfile.gettempdir()) / "tropoi" / (
        f"sky-{slug or 'page'}.html")


def _open(path: pathlib.Path) -> None:
    import webbrowser
    webbrowser.open(path.resolve().as_uri())


def write_sky_page(spec: SkySpec, path=None, *, title: str | None = None,
                   open_browser: bool = False) -> pathlib.Path:
    """Write the standalone sky page of a HYPOTHETICAL atmosphere.

    ``spec.hypothetical`` gives the column (pressure, temperature, gravity,
    gas constant, radius); the composition and star come from ``spec``.
    No saved run, prepared planet, terrain or CUDA is involved. One offline
    HTML file; ``path`` defaults to the system temporary directory.
    """
    if not isinstance(spec, SkySpec):
        raise TypeError("spec must be a SkySpec")
    hyp = spec.hypothetical
    if hyp is None:
        raise ValueError(
            "write_sky_page needs spec.hypothetical (a HypotheticalColumn); "
            "for the column of a saved run use Simulation.sky(...)")
    column = hyp.build(spec.composition)
    block = sky_block(spec, column=column, column_kind="hypothetical",
                      radius_m=hyp.radius_m, r_dry=column.gas_constant)
    title = title or f"Sky — {spec.composition.label} (hypothetical)"
    path = _write_page(block, title,
                       path if path is not None else _default_path(title))
    if open_browser:
        _open(path)
    return path


def saved_sky_page(storage, spec: SkySpec, path=None, *, time: int = -1,
                   gravity: float | None = None, allow_cuda: bool = False,
                   open_browser: bool = True) -> pathlib.Path:
    """Write the sky page above (lat, lon) of one saved PE state.

    The column is reconstructed exactly on the CPU at the observer's point
    from saved state ``time`` (a saved index; negative counts back), with
    the surface height from the exact terrain synthesis there. It needs
    the run's gravity (or ``gravity=``) like ``Simulation.view``. Nothing
    is written into the run.
    """
    from tropoi.representation.diagnostics.pe_levels import reconstruct_levels
    from tropoi.representation.visual.atmosphere import (_check_outside_run,
                                                         _time_label)
    from tropoi.spatial.transforms.host_synthesis import synthesize_latlon

    if not isinstance(spec, SkySpec):
        raise TypeError("spec must be a SkySpec")
    if spec.hypothetical is not None:
        raise ValueError("a saved run's sky uses the saved state's own "
                         "column; use write_sky_page for a hypothetical one")
    if spec.observer_lat_deg is None:
        raise ValueError("a saved run's sky needs observer_lat_deg and "
                         "observer_lon_deg (which column to use)")
    context = storage.pe_context(allow_cuda=allow_cuda).with_gravity(gravity)
    if context.gravity is None:
        raise ValueError(
            "this run records no gravity; pass gravity= to read its "
            "geopotential heights (as for Simulation.view)")
    count = int(np.asarray(storage.times).shape[0])
    index = int(time)
    if not -count <= index < count:
        raise IndexError(f"saved index {time} outside 0..{count - 1}")
    index %= count
    lat = np.array([float(spec.observer_lat_deg)])
    lon = np.array([float(spec.observer_lon_deg)])
    f = reconstruct_levels(storage.frame(index), context, lat, lon, None,
                           winds=False)
    full = np.asarray(context.full_levels, dtype=np.float64)
    p_s = float(f["pressure"][-1, 0, 0] / full[-1])
    h_s = (0.0 if context.surface_elevation_lm is None else float(
        synthesize_latlon(context.surface_elevation_lm, lat, lon)[0, 0]))
    column = AtmosphericColumn.from_levels(
        f["geopotential_height"][:, 0, 0], f["pressure"][:, 0, 0],
        f["temperature"][:, 0, 0], surface_height_m=h_s,
        surface_pressure_pa=p_s, gas_constant=context.r_dry,
        gravity=float(context.gravity), label=context.label)
    t = float(np.asarray(storage.times)[index])
    where = {"run": context.label, "lat_deg": float(lat[0]),
             "lon_deg": float(lon[0]), "time_index": index,
             "time_label": _time_label(t),
             "gravity": float(context.gravity),
             "gravity_source": str(context.provenance.get("gravity"))}
    block = sky_block(spec, column=column, column_kind="saved_state",
                      radius_m=context.radius_m, r_dry=context.r_dry,
                      terrain_m=h_s, context=where)
    title = f"Sky — {context.label}, saved state {index}"
    if path is None:
        path = _default_path(f"{context.label}-{index}")
    path = pathlib.Path(path)
    _check_outside_run(path, storage.run_dir)
    path = _write_page(block, title, path)
    if open_browser:
        _open(path)
    return path
