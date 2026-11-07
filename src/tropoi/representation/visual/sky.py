"""Sky from a point on the planet: single-scattering visual prototype (CPU).

The optional ``sky`` block of the globe page (``terrain_globe.html``) and
the NumPy reference of what the page computes. The page repeats exactly
the same arithmetic in JavaScript (float64), so the two agree to rounding;
``tests/test_sky_view_browser.py`` checks that.

Physics (stated in the page; this is NOT a validated sky colour):

* a horizontally uniform, spherically symmetric column above the observer
  (:class:`~tropoi.spatial.optics.AtmosphericColumn`), of radius R + z;
* Rayleigh scattering only, by an explicit
  :class:`~tropoi.spatial.optics.Composition` (no mixture is ever
  inferred), with the phase function 3/(16 pi) (1 + cos^2 Theta)
  (depolarization neglected in the phase function, kept in the King
  factor of the cross section);
* single scattering of a blackbody star (:class:`~tropoi.spatial.optics.Star`)
  with attenuation along the view and star paths; the planet's solid
  sphere at the column base shadows star paths and ends view paths
  (terrain relief itself casts no shadow and draws no horizon);
* spectral bins of 20 nm from 400 to 760 nm (inside every refractive-index
  record), converted to CIE XYZ with the packaged CIE 1931 table, then
  exposure and the sRGB transfer function; out-of-gamut and over-exposed
  values are clipped and counted.

Unsupported: aerosols and dust, clouds, absorption (ozone, CO2, H2O, ...),
multiple scattering, refraction, polarization, surface reflection and a lit
ground, limb darkening and stellar spectra other than a blackbody, horizontal
variation of the atmosphere, night-sky sources.

Numerics. Directions: a table over view zenith angle (1 degree) and
azimuth relative to the star (3 degrees; single scattering in a spherically
symmetric column is symmetric about the star's vertical plane). Each view
ray is cut at its lowest point and sampled with quadratically graded
midpoint steps that are densest there (``VIEW_STEPS``); each star path from
a sample likewise (``SUN_STEPS``). Number density is read from a uniform
table of ln n in altitude (``DENSITY_NODES``).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from tropoi.spatial.optics import (AtmosphericColumn, Composition, Star,
                                   gas_constant_report)
from tropoi.spatial.optics.colour import (CIE_SOURCE, XYZ_TO_LINEAR_SRGB,
                                          bin_weights, omitted_fraction)
from tropoi.spatial.optics.constants import K_B, LUMINOUS_EFFICACY

__all__ = ["BIN_EDGES_NM", "HypotheticalColumn", "SkySpec", "UNSUPPORTED",
           "density_grid", "radiance_table", "sky_block"]

BIN_EDGES_NM = np.arange(400.0, 761.0, 20.0)
WAVELENGTHS_NM = 0.5 * (BIN_EDGES_NM[:-1] + BIN_EDGES_NM[1:])
N_ZENITH = 181                  # 0..180 deg, 1 deg
N_AZIMUTH = 61                  # 0..180 deg from the star, 3 deg
VIEW_STEPS = 40
SUN_STEPS = 20
DENSITY_NODES = 2048
PROTOTYPE_LABEL = ("Visual prototype — single-scattering Rayleigh sky; not a "
                   "validated sky colour")

UNSUPPORTED = (
    "aerosols and dust (on Mars they dominate the real sky colour)",
    "clouds", "gas absorption (ozone, CO₂, H₂O and others)",
    "multiple scattering", "refraction", "polarization",
    "surface reflection and a lit ground (the ground is drawn unlit)",
    "terrain shadows and a terrain horizon (only the base sphere occludes)",
    "limb darkening and non-blackbody stellar spectra",
    "horizontal variation of the atmosphere (one column at the observer)",
    "night-sky sources",
)


@dataclass(frozen=True)
class HypotheticalColumn:
    """A user-prescribed column for pages without a saved run (labelled).

    ``gas_constant`` is a number (J kg^-1 K^-1) or the explicit string
    ``"from_composition"`` (R*/M of the sky's composition).
    """

    surface_pressure_pa: float
    temperature_k: float
    gravity: float
    gas_constant: float | str
    lapse_rate_k_per_m: float = 0.0
    temperature_floor_k: float | None = None

    def build(self, composition: Composition) -> AtmosphericColumn:
        r = self.gas_constant
        if isinstance(r, str):
            if r != "from_composition":
                raise ValueError("gas_constant must be a number or "
                                 "'from_composition'")
            r = composition.gas_constant
        return AtmosphericColumn.hypothetical(
            surface_pressure_pa=self.surface_pressure_pa,
            temperature_k=self.temperature_k, gravity=self.gravity,
            gas_constant=float(r),
            lapse_rate_k_per_m=self.lapse_rate_k_per_m,
            temperature_floor_k=self.temperature_floor_k)


@dataclass(frozen=True)
class SkySpec:
    """What the sky panel starts with. ``composition`` is required.

    The observer is at (``observer_lat_deg``, ``observer_lon_deg``) and
    EITHER ``altitude_m`` above the reference sphere OR ``pressure_pa``;
    with neither, 2 m above the ground there. Angles in degrees; azimuth
    clockwise from north.
    """

    composition: Composition
    star: Star = field(default_factory=Star.sun_at_1au)
    observer_lat_deg: float = 0.0
    observer_lon_deg: float = 0.0
    altitude_m: float | None = None
    pressure_pa: float | None = None
    look_azimuth_deg: float = 180.0
    look_elevation_deg: float = 20.0
    fov_deg: float = 90.0
    star_azimuth_deg: float = 135.0
    star_elevation_deg: float = 30.0
    exposure_ev: float = 0.0
    hypothetical: HypotheticalColumn | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.composition, Composition):
            raise TypeError("SkySpec needs an explicit Composition; no gas "
                            "mixture is ever inferred")
        if self.altitude_m is not None and self.pressure_pa is not None:
            raise ValueError("give altitude_m or pressure_pa, not both")
        if not -90.0 <= float(self.observer_lat_deg) <= 90.0:
            raise ValueError("observer_lat_deg must be in [-90, 90]")
        if not -90.0 <= float(self.look_elevation_deg) <= 90.0:
            raise ValueError("look_elevation_deg must be in [-90, 90]")
        if not -90.0 <= float(self.star_elevation_deg) <= 90.0:
            raise ValueError("star_elevation_deg must be in [-90, 90]")
        if not 10.0 <= float(self.fov_deg) <= 150.0:
            raise ValueError("fov_deg must be in [10, 150]")


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
    molecules m^-2 along each view ray), ``ground`` (ray ends on the base
    sphere), and the grids.
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


def xyz(radiance, weights) -> np.ndarray:
    """CIE XYZ (Y in cd m^-2) of binned spectral radiance (..., bins)."""
    return LUMINOUS_EFFICACY * np.asarray(radiance) @ np.asarray(weights)


# ---------------------------------------------------------------------------
# Page block
# ---------------------------------------------------------------------------

def optics_block(composition: Composition, star: Star) -> dict:
    lo, hi = BIN_EDGES_NM[:-1], BIN_EDGES_NM[1:]
    sig = composition.cross_section(WAVELENGTHS_NM)
    w = bin_weights(lo, hi)
    return {
        "wavelengths_nm": WAVELENGTHS_NM.tolist(),
        "bin_edges_nm": BIN_EDGES_NM.tolist(),
        "cross_section_m2": sig.tolist(),
        "cmf_weights": w.tolist(),
        "cmf_omitted_fraction": omitted_fraction(400.0, 760.0).tolist(),
        "luminous_efficacy": LUMINOUS_EFFICACY,
        "xyz_to_linear_srgb": XYZ_TO_LINEAR_SRGB.tolist(),
        "star_radiance": star.radiance(WAVELENGTHS_NM).tolist(),
        "star_solid_angle_sr": star.solid_angle_sr,
    }


def sky_block(spec: SkySpec, *, column: AtmosphericColumn | None,
              column_kind: str, radius_m: float, r_dry: float | None,
              gravity: float | None, terrain_m: float = 0.0,
              top_ratio: float | None = None) -> dict:
    """The page's ``sky`` block (validated: the initial observer must fit).

    ``column`` is the initial observer's column (saved state: built by the
    caller at the initial time; hypothetical: the prescribed one).
    """
    if column is None:
        raise ValueError("the sky needs a column")
    if spec.altitude_m is None and spec.pressure_pa is None:
        obs = column.observer_altitude(
            altitude_m=max(terrain_m, column.base_m) + 2.0,
            terrain_m=terrain_m)
        obs["mode"] = "ground"
    else:
        obs = column.observer_altitude(altitude_m=spec.altitude_m,
                                       pressure_pa=spec.pressure_pa,
                                       terrain_m=terrain_m)
    report = gas_constant_report(
        spec.composition, r_dry,
        against=("the run uses r_dry" if column_kind == "saved_state"
                 else "the hypothetical column uses R"))
    om = omitted_fraction(400.0, 760.0)
    notes = [
        PROTOTYPE_LABEL + ".",
        ("Column: " + ("one column of the saved state at the observer"
                       if column_kind == "saved_state" else
                       "HYPOTHETICAL, prescribed by the user")
         + "; " + " ".join(column.notes)),
        f"Composition: {spec.composition.describe()}.",
        report["text"],
        f"Star: {spec.star.describe()}; a uniform disk, no limb darkening.",
        ("Spectrum: 18 bins of 20 nm, 400–760 nm; the CIE functions outside "
         f"that range (x̄ {om[0] * 100:.2g} %, ȳ {om[1] * 100:.2g} %, z̄ "
         f"{om[2] * 100:.2g} % of their integrals) are omitted."),
        ("Exposure: displayed linear sRGB = M·XYZ / L_white · 2^EV, then the "
         "sRGB transfer function; L_white is the brightest sky luminance in "
         "the direction table (star disk excluded). Clipped pixels are "
         "counted."),
        ("Star stays a uniform disk of its true angular size, drawn at least "
         "2 screen pixels across (then marked enlarged)."),
    ]
    sources = (spec.composition.sources()
               + [CIE_SOURCE, "XYZ → sRGB: IEC 61966-2-1",
                  spec.star.source])
    return {
        "version": 1,
        "label": PROTOTYPE_LABEL,
        "column_kind": column_kind,
        "column": column.to_dict() if column_kind == "hypothetical" else None,
        "initial_column": column.to_dict(),
        "radius_m": float(radius_m),
        "r_dry": None if r_dry is None else float(r_dry),
        "gravity": None if gravity is None else float(gravity),
        "top_ratio": top_ratio,
        "composition": spec.composition.to_dict(),
        "consistency": report,
        "star": dict(spec.star.to_dict(), azimuth_deg=spec.star_azimuth_deg,
                     elevation_deg=spec.star_elevation_deg),
        "optics": optics_block(spec.composition, spec.star),
        "observer": {"lat_deg": float(spec.observer_lat_deg),
                     "lon_deg": float(spec.observer_lon_deg), **obs},
        "look": {"azimuth_deg": float(spec.look_azimuth_deg),
                 "elevation_deg": float(spec.look_elevation_deg),
                 "fov_deg": float(spec.fov_deg)},
        "exposure_ev": float(spec.exposure_ev),
        "numerics": {"n_zenith": N_ZENITH, "n_azimuth": N_AZIMUTH,
                     "view_steps": VIEW_STEPS, "sun_steps": SUN_STEPS,
                     "density_nodes": DENSITY_NODES},
        "notes": notes,
        "unsupported": list(UNSUPPORTED),
        "sources": sources,
    }
