# Atmosphere view: browser captures and offline checks (2026-10-23)

These are browser captures of pages written by `Simulation.view()`
([docs/ATMOSPHERE_VIEW.md](../../ATMOSPHERE_VIEW.md)) from two saved runs on this
machine. Neither run is published here: both live in git-ignored directories.
The Mars capture shows the saved orographic-rest state as a visualization only.
These checks concern the page (geometry, controls, offline packaging), not the
dynamics: they are no evidence of the run's dynamical stability.

| file | run | page settings |
| --- | --- | --- |
| `held_suarez_vorticity_sigma0275.png` | `runs/hs-T42L20-prod-002` (Held–Suarez, T42L20, 120 saved states, flat) | default export (5 of 120 states, 20 levels, 2° mesh), field `zeta`, level index 5 (σ = 0.275), centre (50, −30), exaggeration ×20 (default), opacity 0.6 |
| `mars_orographic_rest_pressure_bottom.png` | `out/mola/runs/20261022T001540Z_orographic-isothermal-rest_…_45e8decc` (MOLA terrain, `l_max` 42, 4 levels, 3 states at rest) | field `pressure`, level index 3 (σ = 0.875), opacity 0.4, centre (5, −100), exaggeration ×10 (default) |

Commands (CPU only):

```bash
python examples/atmosphere_view.py runs/hs-T42L20-prod-002 --field zeta --level 5 --centre 50 -30 --out hs.html
python examples/atmosphere_view.py out/mola/runs/20261022T001540Z_orographic-isothermal-rest_rot24p62h_r3_l42_dt3h_8940ef51_45e8decc --field pressure --level 3 --opacity 0.4 --centre 5 -100 --out mola.html
```

The first command warns that it exported 5 of 120 saved states (indices 0, 30,
60, 89, 119; days 10, 310, 610, 900, 1200) to stay under the 32 MB bound. It
writes 26.4 MB; the second writes 3.1 MB.

## How the pages were checked

The pages were opened in headless Chrome 155.0.8059.80 (`--headless=new`, software WebGL
through SwiftShader) from `file://` URLs, driven over the DevTools protocol.
Networking was disabled three ways: a dead proxy, a host resolver that maps every
name to NOTFOUND, and DevTools offline emulation. Every request a page issued
was recorded.

Completed in the browser:

- **Offline.** Every page (Held–Suarez, Mars, a nonuniform synthetic capsule,
  terrain-only) issued exactly one request, its own `file://` URL. There were
  no console errors and no exceptions.
- **Controls.** These were driven through the page's own controls (slider
  `input` events, button and checkbox clicks), and the page's state was read
  back.
  - **Time slider.** Labels show the actual times (`day 0 (t = 0 s)`,
    `day 0.25 (t = 21600 s)`, `day 0.5 (t = 43200 s)`). The surface's mean
    radius changed at every step on the nonuniform fixture and on
    Held–Suarez (mean radius at days 310, 610, 900, 1200 all distinct).
  - **Level slider.** Labels read `1 / 4 · σ = 0.0500` … `4 / 4 · σ = 0.8500`.
    Mean radius decreased strictly from top to bottom. On Held–Suarez the
    page reported Z = 23.7 km (level 1), 5.05 km (level 11) and 0.21 km
    (level 20) node means, with p = 23.8–25.5, 498.8–536.3 and 926.3–996.1 hPa.
  - **Field buttons.** All four fields worked. Vorticity and divergence
    ranges were exactly symmetric, and the colour-bar titles and units
    updated (`Pressure (hPa)`). The per-level range checkbox switched the
    colour-bar note.
  - **Exaggeration.** The surface lift scaled linearly (lift at ×100 / lift
    at ×1 = 100.0). No terrain node rose above the atmospheric layer at ×1
    or ×100, on the fixture or on any Mars level.
  - **Streamlines.** The checkbox removed and restored the lines (0 ↔ 15,902
    vertices). The line set changed with time and level. The Mars rest
    state showed none, and the page said why.
  - **Opacity.** 0, 35, 60 and 100 % all worked; the surface on/off switch
    and the flat-map source switch (atmosphere/terrain) worked too. The hover
    readout gave the field, Z and terrain height.
- **Seams, poles, transparency.** Views centred on the north pole, the south
  pole and the 180° meridian showed no crack, pinch or gap in the surface.
  Streamlines ran continuously across the pole and the seam. At 35–40 %
  opacity the terrain is visible beneath the layer; the far hemisphere is
  hidden by the opaque terrain.
- **Terrain viewer unchanged.** The same terrain payload, rendered through
  the template on `main` and through the extended template, gave 0 differing
  pixels out of 2,380,000.

Not checked here: other browsers (Firefox, Safari), hardware-GPU WebGL, touch
input, and high-DPI displays. Everything above used one Chromium build with
software rendering.
