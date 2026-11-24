"""Configuration of an airless-body thermal experiment: independent
conductive columns under geometric stellar illumination.

An experiment is fully described by a frozen :class:`AirlessConfig`. Its
canonical JSON is hashed (``sha256``). Every material coefficient and
orbital constant carries a source in ``provenance``. Presets are
configurations of the general machinery; nothing body-specific lives in the
solver.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field, replace
from typing import Optional

from tropoi.physics.constants import AU_M, L_SUN_W
from tropoi.physics.materials import DepthProfile, Stratum

__all__ = ["AirlessConfig", "PRESETS", "mercury_hayne_analogue", "mercury_vpw99",
           "strata_from_dicts", "vpw99_strata"]

DAY = 86400.0
#: IAU 2015 Resolution B3 nominal solar mass parameter (m^3 s^-2).
GM_SUN = 1.3271244e20


def _value(v):
    if isinstance(v, dict):
        if v.get("profile") != "exponential":
            raise ValueError(f"unknown profile {v!r}")
        return DepthProfile(float(v["surface"]), float(v["deep"]), float(v["scale"]))
    return float(v)


def strata_from_dicts(items) -> list:
    """Strata from their configuration (``Stratum.to_dict``) form."""
    out = []
    for d in items:
        rng = d.get("heat_capacity_range")
        out.append(Stratum(float(d["bottom"]), _value(d["density"]),
                           _value(d["contact_conductivity"]),
                           d["heat_capacity"],
                           radiative_coefficient=_value(d.get("radiative_coefficient", 0.0)),
                           radiative_reference_temperature=float(
                               d.get("radiative_reference_temperature", 350.0)),
                           heat_capacity_range=None if rng is None else tuple(rng),
                           name=d.get("name", "stratum"), source=d.get("source", "")))
    return out


@dataclass(frozen=True)
class AirlessConfig:
    """One experiment.

    * ``illumination``: an ``orbital`` insolation spec
      (:func:`tropoi.physics.insolation.insolation_from_spec`).
    * ``strata``: material strata in :meth:`Stratum.to_dict` form, top to
      bottom. The last ``bottom`` is the column depth.
    * ``columns``: ``{"kind": "latlon", "nlat", "nlon"}`` (regular,
      cell-centred, latitudes south to north, east longitudes from 0) or
      ``{"kind": "points", "lat_deg": [...], "lon_deg": [...]}``.
    * ``vertical``: ``{"top_thickness", "max_stretch"}``.
    * ``cycle_steps``: steps per illumination cycle at the production
      timestep. ``spinup_cycle_steps`` is the coarse spin-up count, or None
      for no coarse stage.
    * ``cycle_s``: length of the illumination cycle, an exact recurrence
      period of the geometry (Mercury: two orbits).
    * ``periodicity``: the frozen G-P2 thresholds and cycle limits.
    * ``output``: tier-2 sampling per cycle (``surface_samples``,
      ``profile_samples``) and ``every_step_profiles`` for point runs.
    """

    experiment_id: str
    illumination: dict
    strata: tuple
    albedo: float
    emissivity: float
    geothermal_flux: float
    columns: dict
    vertical: dict
    cycle_s: float
    cycle_steps: int
    spinup_cycle_steps: Optional[int]
    initial_temperature: float
    periodicity: dict
    output: dict
    radius_m: float
    provenance: dict = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "strata", tuple(dict(s) for s in self.strata))
        if self.cycle_steps <= 0 or (self.spinup_cycle_steps is not None
                                     and self.spinup_cycle_steps <= 0):
            raise ValueError("cycle step counts must be positive")
        for key in ("surface_samples", "profile_samples"):
            n = int(self.output[key])
            if self.cycle_steps % n:
                raise ValueError(f"{key}={n} must divide the production cycle steps "
                                 f"{self.cycle_steps}")

    @property
    def dt(self) -> float:
        return self.cycle_s / self.cycle_steps

    def to_dict(self) -> dict:
        return asdict(self)

    def canonical_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()

    @classmethod
    def from_dict(cls, d: dict) -> "AirlessConfig":
        return cls(**{**d, "strata": tuple(d["strata"])})

    def with_(self, **changes) -> "AirlessConfig":
        return replace(self, **changes)


# -- Mercury -----------------------------------------------------------------------------

#: Hayne et al. (2017, JGR Planets 122, doi:10.1002/2017JE005387; arXiv:1711.00977),
#: Appendix A, Table A1, as read from the arXiv text on 2026-11-24.
HAYNE_2017 = {
    "rho_s": 1100.0, "rho_d": 1800.0, "K_s": 7.4e-4, "K_d": 3.4e-3, "H": 0.06,
    "chi": 2.7, "cp": (-3.6125, 2.7431, 2.3616e-3, -1.2340e-5, 8.9093e-9),
    "emissivity": 0.95,
    # "The polynomial fit cp(T) is valid for temperatures from < 90 K to > 400 K"
    "cp_valid_range_K": (90.0, 400.0),
}

#: JPL SSD "Approximate Positions of the Planets", Table 1 (valid 1800-2050 AD),
#: Mercury: a = 0.38709927 au, e = 0.20563593 (fixed; rates ignored).
MERCURY_A_AU = 0.38709927
MERCURY_E = 0.20563593
#: NASA NSSDC Mercury fact sheet (updated 2024-01-11): Bond albedo 0.068.
MERCURY_BOND_ALBEDO = 0.068


def _mercury(experiment_id, strata, *, albedo, emissivity, geothermal_flux, columns,
             top_thickness, max_stretch, cycle_steps, spinup_cycle_steps,
             initial_temperature, provenance) -> AirlessConfig:
    """Idealized airless Mercury: an eccentric Kepler orbit, an exact 3:2
    spin-orbit resonance, zero obliquity and a flat spherical surface. The
    illumination is a point-source Sun (no finite disk, no limb darkening).
    The material is given by ``strata``."""
    a_m = MERCURY_A_AU * AU_M
    period = 2.0 * math.pi * math.sqrt(a_m ** 3 / GM_SUN)
    s_a = L_SUN_W / (4.0 * math.pi * a_m * a_m)
    if columns is None:
        columns = {"kind": "latlon", "nlat": 90, "nlon": 180}
    illumination = {"kind": "orbital", "stellar_constant": s_a, "period": period,
                    "eccentricity": MERCURY_E, "obliquity_deg": 0.0,
                    "perihelion_longitude_deg": 0.0, "time_of_perihelion": 0.0,
                    "spin": {"spins_per_orbit": [3, 2],
                             "subsolar_longitude_at_epoch_deg": 0.0}}
    base = {
        "orbit": "a, e: JPL SSD approximate elements Table 1 (fixed, rates ignored); "
                 "period from Kepler III with the IAU 2015 B3 nominal GM_sun "
                 f"({GM_SUN} m^3 s^-2); epoch t = 0 at perihelion",
        "irradiance": "IAU 2015 B3 nominal solar luminosity 3.828e26 W; point source "
                      "(no finite solar disk or limb darkening)",
        "spin": "exact 3:2 resonance (spin rate 3n/2 from the orbit's own period); "
                "obliquity 0 (idealized; NSSDC lists 0.034 deg); longitude 0 subsolar "
                "at perihelion",
        "geography": "smooth sphere, uniform material, no topography or shadowing",
        "longitude": "east-positive; VPW99 quote west longitudes (0 W = 0 E, "
                     "90 W = 270 E)",
    }
    return AirlessConfig(
        experiment_id=experiment_id, illumination=illumination,
        strata=tuple(st.to_dict() for st in strata), albedo=float(albedo),
        emissivity=float(emissivity), geothermal_flux=float(geothermal_flux),
        columns=columns, vertical={"top_thickness": top_thickness, "max_stretch": max_stretch},
        cycle_s=2.0 * period, cycle_steps=int(cycle_steps),
        spinup_cycle_steps=None if spinup_cycle_steps is None else int(spinup_cycle_steps),
        initial_temperature=float(initial_temperature),
        periodicity={"surface_K": 0.5, "subsurface_K": 0.2, "bottom_drift_K": 0.01,
                     "energy_W_m2": 1e-3, "production_cycles_required": 2,
                     "max_spinup_cycles": 200, "max_production_cycles": 6,
                     "spinup_acceleration": True},
        output={"surface_samples": 720, "profile_samples": 180,
                "every_step_profiles": columns.get("kind") == "points"},
        radius_m=2.4405e6, provenance={**base, **provenance})


#: Vasavada, Paige & Wood (1999, Icarus 141, 179), section II.A, p. 181, read
#: from the rendered page on 2026-11-24.
VPW99 = {
    "top": {"bottom": 0.02, "rho": 1300.0, "kc": 9.22e-4, "chi": 1.48},
    "bottom": {"rho": 1800.0, "kc": 9.3e-3, "chi": 0.073},
    "albedo": 0.10, "emissivity": 0.95, "internal_heat_flux": 0.020,
}
LEDLOW_POSITIVE_ABOVE_K = 10.13     # eq. (6) of Ledlow et al. 1992 is negative below


def vpw99_strata(model: str, depth: float) -> list:
    """The regolith of VPW99. ``model`` is ``TWO`` (2 cm top layer over the
    bottom material), ``TOP`` (top material throughout) or ``BOT`` (bottom
    material throughout). The heat capacity is Ledlow et al. (1992) eqs. (6)-(7)
    in every case."""
    from tropoi.physics.materials import ledlow_1992
    law = ledlow_1992()
    rng = (LEDLOW_POSITIVE_ABOVE_K, None)
    src = "Vasavada, Paige & Wood 1999, sec. II.A p. 181; c_p: Ledlow et al. 1992 eqs. (6)-(7)"
    t, b = VPW99["top"], VPW99["bottom"]

    def layer(props, bottom, name):
        return Stratum(bottom, props["rho"], props["kc"], law,
                       radiative_coefficient=props["chi"], heat_capacity_range=rng,
                       name=name, source=src)
    if model == "TWO":
        return [layer(t, t["bottom"], "VPW99 top layer"), layer(b, depth, "VPW99 bottom layer")]
    if model == "TOP":
        return [layer(t, depth, "VPW99 top-layer material (Model TOP)")]
    if model == "BOT":
        return [layer(b, depth, "VPW99 bottom-layer material (Model BOT)")]
    raise ValueError(f"model must be TWO, TOP or BOT, not {model!r}")


def mercury_vpw99(experiment_id: Optional[str] = None, *, model: str = "TWO",
                  geothermal_flux: float = VPW99["internal_heat_flux"],
                  columns: Optional[dict] = None, depth: float = 1.0,
                  top_thickness: float = 1.0e-3, max_stretch: float = 1.1,
                  cycle_steps: int = 4320, spinup_cycle_steps: Optional[int] = 720,
                  initial_temperature: float = 350.0) -> AirlessConfig:
    """Mercury with the Vasavada, Paige & Wood (1999) regolith (primary
    Planet One experiment).

    VPW99's flat-surface results (their sec. II.B: Models BOT, TOP and TWO)
    neglect the internal heat flux. Reproducing them needs
    ``geothermal_flux=0``. The general configuration uses their stated
    0.020 W m^-2."""
    if experiment_id is None:
        experiment_id = f"mercury-vpw99-{model.lower()}-G{geothermal_flux:g}"
    return _mercury(
        experiment_id, vpw99_strata(model, depth), albedo=VPW99["albedo"],
        emissivity=VPW99["emissivity"], geothermal_flux=geothermal_flux, columns=columns,
        top_thickness=top_thickness, max_stretch=max_stretch, cycle_steps=cycle_steps,
        spinup_cycle_steps=spinup_cycle_steps, initial_temperature=initial_temperature,
        provenance={
            "material": f"VPW99 Model {model} (sec. II.A-B, p. 181-182): two-layer "
                        "regolith, k = kc [1 + chi (T/350)^3]; c_p Ledlow et al. 1992 "
                        "eqs. (6)-(7), cal g^-1 K^-1 x 4184 (thermochemical calorie)",
            "albedo": "0.10 (VPW99 p. 181), constant",
            "emissivity": "0.95 (VPW99 p. 181)",
            "geothermal_flux": (f"{geothermal_flux} W m^-2; VPW99 assume 0.020 for Mercury "
                                "(Schubert et al. 1988) but neglect it in their flat-surface "
                                "results"),
            "differences_from_VPW99": "point-source Sun (VPW99: finite disk with limb "
                                      "darkening, Allen 1973); diagnosed Kirchhoff skin "
                                      "(VPW99: second-order extrapolation from the top three "
                                      "layers); idealized fixed-element orbit and exact 3:2 "
                                      "rotation (VPW99: orbit updated each step); our grid, "
                                      "timestep and depth chosen by convergence tests",
        })


def mercury_hayne_analogue(experiment_id: str = "mercury-hayne2017-analogue", *,
                           columns: Optional[dict] = None, depth: float = 2.0,
                           top_thickness: float = 1.0e-3, max_stretch: float = 1.1,
                           cycle_steps: int = 4320, spinup_cycle_steps: Optional[int] = 720,
                           geothermal_flux: float = 0.02,
                           initial_temperature: float = 300.0) -> AirlessConfig:
    """Mercury with the Hayne et al. (2017) **lunar** regolith as an analogue,
    kept for numerical testing (L1/L2 evidence only, never L3)."""
    h = HAYNE_2017
    stratum = Stratum(depth, DepthProfile(h["rho_s"], h["rho_d"], h["H"]),
                      DepthProfile(h["K_s"], h["K_d"], h["H"]), h["cp"],
                      radiative_coefficient=h["chi"], heat_capacity_range=h["cp_valid_range_K"],
                      name="lunar regolith (Hayne et al. 2017)",
                      source="Hayne et al. 2017 Table A1 (arXiv:1711.00977, Appendix A)")
    return _mercury(
        experiment_id, [stratum], albedo=MERCURY_BOND_ALBEDO, emissivity=h["emissivity"],
        geothermal_flux=geothermal_flux, columns=columns, top_thickness=top_thickness,
        max_stretch=max_stretch, cycle_steps=cycle_steps,
        spinup_cycle_steps=spinup_cycle_steps, initial_temperature=initial_temperature,
        provenance={
            "albedo": "Bond albedo 0.068, NASA NSSDC Mercury fact sheet (2024-01-11); constant",
            "emissivity": "0.95, Hayne et al. 2017 Table A1 (lunar regolith)",
            "material": "Hayne et al. 2017 Table A1 lunar regolith: LUNAR ANALOGUE, not a "
                        "Mercury-specific material; c_p polynomial stated valid from <90 K "
                        "to >400 K, so Mercury daytime values are EXTRAPOLATED",
            "geothermal_flux": f"{geothermal_flux} W m^-2 (VPW99's Mercury value)",
        })


PRESETS = {"mercury-vpw99": mercury_vpw99,
           "mercury-hayne2017-analogue": mercury_hayne_analogue}
