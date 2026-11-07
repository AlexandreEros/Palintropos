# Sky from the surface (visual prototype)

The offline globe page can show the sky as seen by an observer on the planet.
The observer has a latitude, a longitude, and either an altitude above the
reference sphere or a pressure. They look in a chosen direction, and a
configurable star lights the sky. The sky is a **visual prototype**: Rayleigh
single scattering by an explicit gas mixture. **It is not a validated sky
colour**, and the page says so on screen.

It is an optional `sky` block of the page that `Simulation.view()` and
`PreparedPlanet.view()` already write. Without it, both pages are unchanged:
browser captures of a terrain page and of a PE atmosphere page through the
template on `main` and through this one differ in 0 of 1,070,176 pixels.

## Use

A saved primitive-equation run (the column is the saved state's):

```python
from tropoi.representation.archive import open_simulation
from tropoi.representation.visual.sky import SkySpec
from tropoi.spatial.optics import COMPOSITION_PRESETS, Composition

air = COMPOSITION_PRESETS["earth_dry_air_illustrative"]   # or Composition.of(N2=..., O2=..., ...)
sim = open_simulation("runs/hs-T42L20-prod-002")
sim.view("sky.html", sky=SkySpec(air, observer_lat_deg=40, observer_lon_deg=-105,
                                 star_azimuth_deg=200, star_elevation_deg=25))
```

A terrain-only planet. There is no saved atmosphere, so you prescribe a
**hypothetical** column; it is labelled as such everywhere:

```python
from tropoi.representation.visual.sky import HypotheticalColumn, SkySpec
from tropoi.spatial.optics import Composition

co2 = Composition.of(CO2=1.0, label="pure CO2", source="stated for this example")
planet.view(sky=SkySpec(co2, observer_lat_deg=-4.6, observer_lon_deg=137.4,
                        hypothetical=HypotheticalColumn(
                            surface_pressure_pa=610.0, temperature_k=210.0,
                            gravity=3.71, gas_constant="from_composition")))
```

In the page, **Sky** (top left of the stage) switches the stage to the sky.
Drag to look around. The Sky panel sets:
- the observer's latitude and longitude (or "Globe centre", or a click on the flat map in Sky mode);
- how the observer's height is given: 2 m above the ground, altitude above `R`, or pressure;
- the star's azimuth (clockwise from north) and elevation;
- the field of view;
- the exposure, in stops.

Changing the star's azimuth only rotates the sky. Changing its elevation, the
observer, or the saved time recomputes the sky (about 0.8 s in headless
Chrome).

## Inputs and refusals

| input | rule |
| --- | --- |
| composition | **required and explicit.** Species with sourced optical data: N₂, O₂, Ar, CO₂. Mole fractions must sum to 1 (±1e-6); they are never renormalized. Unknown species are refused. A mixture is never inferred from `r_dry`, `kappa` or a molar mass. |
| observer height | altitude above the reference sphere **or** pressure (both is refused). With neither: 2 m above the ground. |
| below ground | refused: an altitude below the terrain, or a pressure above the local surface pressure. |
| too high | refused: above the top saved full level (saved runs) or above the column (hypothetical). |
| saved runs | every native level must be exported (`levels=None`); the run must record gravity (as for `view`). |
| hypothetical | `surface_pressure_pa` (at altitude 0), `temperature_k`, `gravity`, and `gas_constant` (a number, or the explicit string `"from_composition"`); optionally `lapse_rate_k_per_m` with `temperature_floor_k`. |

In the page a refusal is a message in the panel, not an error.

## How composition relates to the dry thermodynamics

A composition implies a mean molar mass `M = Σ xᵢ Mᵢ`, using IUPAC/CIAAW
standard atomic weights, and so a gas constant `R_mix = R*/M`. The page
reports whether `R_mix` agrees with the run's `r_dry` (0.1 % reporting
threshold), disagrees with it, or has nothing to compare against. It never
changes `r_dry`, and no solver reads a composition.

The column's number density is the ideal-gas `n = p/(k_B T)` from the saved
`p` and `T`, which needs no composition. The composition only supplies the
cross section per molecule. `c_p` is not compared yet: that needs sourced,
temperature-dependent heat capacities.

Nothing is written to a run: `config.json`, scientific hashes and run ids are
unchanged (a test hashes the capsule before and after).

When radiative transfer arrives:
- composition becomes a solver input the way topography did: emitted into the
  config only when present, so existing runs keep byte-identical hashes;
- a composition that disagrees with `r_dry`/`cp_dry` will be an error to
  resolve, not a silent override.

The optics already live in `spatial.optics`, so a `temporal` tendency can
import them.

## Physics and its limits

- **Column.** One horizontally uniform, spherically symmetric column at the observer:
  - nodes `(z, ln p, T)`, with `ln p` and `T` linear in altitude between nodes;
  - saved runs: the native full levels at the observer, interpolated from the exported display mesh. `Z = Φ/g` is used as altitude, the model's convention. The layer below the lowest full level is isothermal at that level's temperature. Above the top full level, an isothermal extension (`r_dry T / g` scale height) up to `p = 10⁻⁵ p_s`, which the saved state does not describe;
  - empty above that.
- **Scattering.** Rayleigh, single scattering, phase function `3/(16π)(1+cos²Θ)`. Depolarization is kept in the King factor but not in the phase function. Attenuation applies along both the view path and the star path.
- **Star.** A uniform blackbody disk (Planck radiance × solid angle). `Star.sun_at_1au()` uses the IAU 2015 B3 nominal Sun (5772 K, 695 700 km) and the IAU 2012 au.
- **Ground.** The solid sphere at the column base shadows star paths and ends view paths. The ground itself is drawn unlit; only the air in front of it scatters.
- **Colour.** 18 bins of 20 nm from 400 to 760 nm, then CIE 1931 XYZ, then IEC 61966-2-1 sRGB.
  - The CIE functions outside 400–760 nm are omitted and reported: x̄ 0.11 %, ȳ 0.004 %, z̄ 0.49 % of their integrals.
  - Exposure: `M·XYZ / L_white · 2^EV`, where `L_white` is the brightest sky luminance in the direction table, star excluded.
  - Over-exposed and out-of-gamut channels are clipped and counted in the panel.
  - The star disk is drawn at least one pixel in radius, and the panel says when it is enlarged.

**Not modelled:**
- aerosols and dust (on Mars these dominate the real sky colour, so a Rayleigh-only CO₂ sky is *not* what Mars looks like);
- clouds;
- gas absorption (ozone, CO₂, H₂O and others);
- multiple scattering;
- refraction;
- polarization;
- surface reflection and a lit ground;
- terrain shadows and a terrain horizon;
- limb darkening and non-blackbody spectra;
- horizontal variation of the atmosphere;
- night-sky sources.

## Optical data and their status

| quantity | source | status |
| --- | --- | --- |
| N₂, Ar refractive index | Börzsönyi et al. 2008, Appl. Opt. 47, 4856 (0.4–1.0 µm, 273 K, 1000 hPa) | formulas from the refractiveindex.info database (Polyanskiy 2024), fetched 2026-11-07 |
| O₂ refractive index | Zhang, Lu & Wang 2008, Appl. Opt. 47, 3143 (0.4–1.8 µm, 293 K, 101 325 Pa) | same |
| CO₂ refractive index | Bideau-Mehu et al. 1973, Opt. Commun. 9, 432 (0.18–1.69 µm, 273.15 K, 101 325 Pa) | same |
| King factors | Bates 1984, as tabulated by Bodhaine et al. 1999 | **transcribed without re-reading the paper** (`king_verified = False`); check before any quantitative use |
| colour matching | CIE 1931 2° dataset, DOI 10.25039/CIE.DS.xvudnb9b | verbatim copy, SHA-256 checked on load |
| sRGB | IEC 61966-2-1 matrix and transfer function | |

Wavelengths outside a record's measured range are refused, not extrapolated.

## What the tests establish (and what they do not)

**Established:**
- The page's float64 radiance table equals the NumPy reference (`sky.radiance_table`) to a relative 1e-9 in every sampled direction, for a saved state and for a hypothetical column (`tests/test_sky_view_browser.py`).
- Hydrostatics are exact for isothermal columns.
- Altitude ↔ pressure round-trips.
- The vertical column count equals `p_s R/(k_B g)` within 2e-3.
- A sky with no scatterers is black.
- Radiance is linear in irradiance.
- The column is fully shadowed when the star is far below the horizon.
- Quadrupling every step count changes the spectral radiance above the horizon by at most 2.8e-3 (measured; tested against 5e-3).
- The refusals work.
- The no-inference rule holds.
- Runs are left untouched.
- Pages without a sky are unchanged.

**Not established:** that any colour or brightness matches a real sky. That
would need a comparison of radiances, not colours, against a reference
radiative-transfer code (for example libRadtran or DISORT) for a standard
atmosphere, plus verified King factors. None of the colours or optical depths
shown is an acceptance target.

## Deriving the star from time and an orbit (not implemented)

The star's azimuth and elevation are manual. Deriving them needs data a saved
run does not carry:
- the rotation period (in `PlanetSpec`/the run config, not passed to the viewer);
- obliquity, the sense of rotation, and the prime-meridian angle at an epoch;
- the orbital elements (a, e, i, Ω, ω, mean anomaly at the epoch) and the mapping from model `t = 0` to that epoch;
- the star's luminosity and radius, for irradiance and angular size.

None of the current PE scenarios has insolation or a diurnal cycle, so a
time-derived star would not be related to the simulated temperatures. The
page should say so when this is added.

## Where it lives

| layer | module |
| --- | --- |
| optics (CPU, spatial layer) | `spatial.optics`: `rayleigh`, `composition`, `column`, `star`, `colour`, `constants`, `data/CIE_xyz_1931_2deg.csv` |
| reference + page block | `representation.visual.sky` (`SkySpec`, `HypotheticalColumn`, `radiance_table`, `sky_block`) |
| entry points | `Simulation.view(sky=...)` → `atmosphere.atmosphere_payload`; `PreparedPlanet.view(sky=...)` → `terrain.write_globe_html` |
| page | `representation.visual.terrain_globe.html`, sky section (`window.__terrainGlobe.sky` test API) |
