# Mars MOLA terrain in the interactive globe (2026-10-22)

`mola_globe_l42.png` is a browser capture of the actual terrain viewer
(`PreparedPlanet.view()`), not an illustration. It shows real Mars topography
resolved by the import path. The measured validation of that import is in
[docs/TERRAIN_IMPORT.md](../../TERRAIN_IMPORT.md#real-data-mars-from-mola).

| | |
| --- | --- |
| Data | MOLA MEGDR median topography, 4 px/deg (`meg004/megt90n000cb.img`, NASA PDS Geosciences Node, dataset `MGS-M-MOLA-5-MEGDR-L3-V1.0`), SHA-256 `25f16fb7aaf857898dcf98bc4f841341a24f8b9f7e98453ca083bc45d897ca2c`; converted to NetCDF by `examples/mola_mars.py convert` (`megt_4ppd.nc`, SHA-256 `aea919137cbab87df2ffe5c2efb115a0741afda2bae7634aace5ce89c2ef24f2`) |
| Terrain | Mars-sized planet, Gauss lat-lon 66 × 132, `l_max` 42, no smoothing, full truncation (SWE use) |
| Layer shown | **Model** (the resolved coefficients evaluated exactly, as the solver uses them): −7325 .. 16383 m |
| Colour bar | metres; one scale for both layers (model and the 1°-averaged source), so it spans −7391 .. 19820 m |
| Relief | radial exaggeration ×20, display only |
| View | centred on (0°, 95° W): Olympus Mons near the upper-left limb, the Tharsis volcanoes, Valles Marineris to the right |
| Capture | Google Chrome 155.0.8059.80, headless, 1280 × 860 window at device scale 1, then cropped to 1280 × 828 (the cut-off flat-map heading removed); dark colour scheme |
| Code | branch `feat/terrain-import` (PR #7) |
| Image SHA-256 | `4534bb5f2e8d8db418bbcb710499243ca26c7bbd79c3cb7f7e3065889f7829aa` |

Checked in the captured page's DOM (`--dump-dom` of the same page and
settings): the pressed layer button is "Model", the colour-bar title is
"Elevation (m)", and the exaggeration label reads "×20".

## Reproduce

Download `megt90n000cb.img` and `.lbl` from
<https://pds-geosciences.wustl.edu/mgs/mgs-m-mola-5-megdr-l3-v1/mgsl_300x/meg004/>, then:

```bash
python examples/mola_mars.py convert megt90n000cb.lbl mola/megt_4ppd.nc
```

```bash
python examples/mola_mars.py show mola/megt_4ppd.nc --lmax 42 --units m --centre 0 -95 --exaggeration 20 --out mola --open
```

`show` needs CUDA to resolve the terrain. It writes `mola/mola_l42_globe.html`
(a self-contained page) and opens it in the default browser. Opening that page
later needs nothing but a browser. To capture it the same way:

```powershell
& "C:\Program Files\Google\Chrome\Application\chrome.exe" --headless=new --hide-scrollbars --window-size=1280,860 --force-device-scale-factor=1 --virtual-time-budget=8000 --screenshot=mola_globe.png "file:///<absolute path>/mola/mola_l42_globe.html"
```

A PNG capture is not bit-reproducible across browsers, GPUs or drivers.
