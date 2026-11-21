# Real-gas absorption: tables, configuration, THAI dry development runs

Status on 2026-11-16, branch `feat/thai-dry-real-gas-longwave`. Multi-band
CO2 absorption from documented k-tables now drives the column physics and
the primitive-equation model through the ordinary radiative configuration.
It was exercised on the two dry THAI compositions:

* **Ben 2**, 1 bar CO2: column verified, 5-day coupled run with a stop and
  an exact resume.
* **Ben 1**, 1 bar N2 with 400 ppmv CO2: column verified, 5-day coupled
  run.
* Both again with real-gas **stellar** absorption added (presets `…-ck`).
  Those two runs complete and close their budgets, but **their top layers
  are not usable**: Ben 2 reaches 293 m s^-1 at 2 Pa (§6).

These are **development runs with the THAI compositions, not THAI
reproductions**: five days from the isothermal start, and the differences
listed in §7. Equilibration and the
comparison with Turbet et al. (2022) are left for later
(the plan in `THAI_BEN_PLAN.md`, a draft that is not yet tracked).

## 1. What changed

| piece | where |
| --- | --- |
| `correlated_k` kind for the `longwave` / `shortwave` entries of `RadiativeConfig` | [model.py](../../src/tropoi/run/radiative/model.py) `_correlated_k` |
| finding a table by name, checking its content hash, holding a premixed table to its gas | [tables.py](../../src/tropoi/physics/radiation/tables.py) |
| `KTable`: per-band quadrature weights, `metadata`, `content_sha256()` | [kdist.py](../../src/tropoi/physics/radiation/kdist.py) |
| presets `dev-thai-ben1`, `dev-thai-ben2` | [config.py](../../src/tropoi/run/radiative/config.py) `thai_dry_dev` |
| reader and importer for SOCRATES look-up spectral files, fetch with pinned hashes | [tools/radiation/opacity](../../tools/radiation/opacity) |
| column fluxes and heating of a preset, with a reference comparison | [thai_column.py](../../tools/radiation/thai_column.py) |
| run records and the resume comparison | [thai_dev_evidence.py](../../tools/radiation/thai_dev_evidence.py) |

The existing `KTable`, `CorrelatedKLongwave` / `CorrelatedKShortwave`,
two-stream solver, `ColumnPhysics` and `ColumnPhysicsCoupling` are reused
unchanged in what they compute. The stellar presets needed no solver or
provider change: a table and the star's band fractions are inputs. The gray presets are untouched and keep
their configuration hashes.

## 2. Configuration

```json
"longwave": {"kind": "correlated_k",
             "tables": [{"file": "thai-ben2-sp-lw-17-dsa-mars.npz",
                         "sha256": "9d4c18a6…"}],
             "mass_fractions": {"thai_ben2": 1.0},
             "bounds": "error"}
```

* `tables` name `KTable` archives. A bare file name is looked up in the
  directories of `TROPOI_OPACITY_PATH`, then in `opacity_cache/tables`
  under the working directory. Tables are data and are not in the
  repository (§4 gives the two commands that produce them).
* `sha256` is `KTable.content_sha256()`: the hash of the table's numbers
  (species label, pressure, temperature, band edges, weights, k), not of
  the `.npz` container, whose bytes depend on the compressor and on
  timestamps. It is required, it is part of `config_sha256`, and a table
  whose numbers differ refuses to load. A re-imported table reproduces the
  hash exactly when the source files are the pinned ones.
* `mass_fractions` maps each table's `species` to the mass fraction that
  multiplies its k. A premixed table is per kg of the mixture: its fraction
  must be 1.0, and the model's `gas_constant` must equal the one the table
  was mixed for (relative 1e-6), otherwise the build fails. This is what
  keeps the radiative and the bulk composition from drifting apart.
* `bounds` is `"error"` by default: a layer outside the table's pressure
  or temperature range stops the run. `"clamp"` holds the edge value and
  counts the layers; nothing uses it.
* Thermal option `diffusivity` (1.66). Stellar options
  `stellar_temperature` (blackbody star) or `band_fractions`.

Any table in the `KTable` format takes this path, whatever produced it: a
future gas needs a table and its hash, not code.

The two THAI presets differ in four configuration keys and nothing else
(asserted in `tests/test_radiation_ktable_config.py`):

| | `dev-thai-ben1` | `dev-thai-ben2` |
| --- | --- | --- |
| composition preset (`tropoi.physics.composition`) | `thai_ben1`: N2 0.9996, CO2 0.0004 | `thai_ben2`: CO2 1.0 |
| `gas_constant` (from the molar mass) | 296.7289 J kg^-1 K^-1 | 188.9264 J kg^-1 K^-1 |
| `cp` (rigid-rotor/harmonic-oscillator value at 250 K, held constant) | 1038.713 J kg^-1 K^-1 | 791.485 J kg^-1 K^-1 |
| `longwave` table | `thai-ben1-sp-lw-15-dsa-ar10bar.npz` | `thai-ben2-sp-lw-17-dsa-mars.npz` |
| `experiment_id` | `dev-thai-ben1-lwck-heated-T21L40` | `dev-thai-ben2-lwck-heated-T21L40` |

`dev-thai-ben1-ck` and `dev-thai-ben2-ck` use a `correlated_k` stellar table
with the star's `band_fractions` (ids `dev-thai-ben*-ck-heated-T21L40`).
Both use `thermal_source="linear"`, as every recorded coupled run below
did, and keep their recorded hashes (`dev-thai-ben2-ck`: `fdbcf1cf…`).
The corrected Ben 2 source for the upper-atmosphere issue
[#47](https://github.com/AlexandreEros/Palintropos/issues/47) is a separate
opt-in preset, `dev-thai-ben2-ck-layer-linear` (id
`dev-thai-ben2-ck-layer-linear-heated-T21L40`, `08bc8714…`): the same
configuration with `thermal_source="layer-linear"` and its own experiment
id. No run has been recorded under that id yet. The archived replay of §6
(`…-9b2060c-layer-linear`, `3f2de9d8…`) was made before the preset existed,
as `dev-thai-ben2-ck` with the source overridden; it is identical to the
new preset apart from the experiment id, which a test asserts. Ben 1 keeps
`linear`.

## 3. The tables

Both are imported from SOCRATES spectral files prepared for ROCKE-3D by
D. S. Amundsen and published on the NASA NCCS data share. The statements
about spectroscopy below are those of the ROCKE-3D documentation
(<https://simplex.giss.nasa.gov/gcm/ROCKE-3D/Spectralfiles.html>, read
2026-11-16, SHA-256 of the page `ff20b814…a983a6`); where it is silent the
table says so. The complete record of each table (every source file with
its URL, size and SHA-256, the per-band composition of the points, all
assumptions) is in [opacity/](opacity/), written by the importer and also
stored inside the `.npz` as `metadata`.

| | Ben 2 | Ben 1 |
| --- | --- | --- |
| table | `thai-ben2-sp-lw-17-dsa-mars.npz` | `thai-ben1-sp-lw-15-dsa-ar10bar.npz` |
| content SHA-256 | `9d4c18a6822690aa12051598695ca4d6946324bb36f8752a35d57b9963087e18` | `699c00492b5e5d5b9e6642da054a8daeee3f3269670a351ef689356c350e9038` |
| record | [thai-ben2-sp-lw-17-dsa-mars.json](opacity/thai-ben2-sp-lw-17-dsa-mars.json) | [thai-ben1-sp-lw-15-dsa-ar10bar.json](opacity/thai-ben1-sp-lw-15-dsa-ar10bar.json) |
| source files (`…/spectral_files/<dir>/`) | `sp_lw_dsa_mars/sp_lw_17_dsa_mars` (39 293 bytes, `e1197e49…bc75f5`) and `…_k` (5 222 825 bytes, `57e679a7…7d2785`) | `sp_lw_dsa_ar10bar/sp_lw_15_dsa_ar10bar` (49 157 bytes, `7bfb63eb…540bb7`) and `…_k` (4 414 929 bytes, `da6393d5…13af88`) |
| units, mass basis | k in m^2 per kg **of the mixture** (premixed, mass fraction 1); pressure Pa; temperature K; band edges cm^-1 | same |
| mixed for | CO2 mole fraction 1; R = 188.9264 | N2 0.9996, CO2 0.0004 (mass fractions 0.999372, 0.000628); R = 296.7289 |
| spectral coverage | 1–3000 cm^-1, 17 bands, 200 spectral points | 1–3000 cm^-1, 15 bands, 240 spectral points |
| pressure, temperature | 1 Pa – 10^6 Pa, 61 points (10 per decade); 100–400 K, 13 points (25 K) | same |
| documented validity | CO2-dominated, > 90 % CO2, surface pressure up to 10 bar | N2/H2-dominated with up to 1 % CO2 and CH4, up to 10 bar |
| documented accuracy | about 1 W m^-2 below about 310 K, against its 350-band reference | same |
| line data | HITRAN 2012 | HITRAN 2012 |
| broadening | CO2 self-broadened (look-up axis of CO2 fraction 0.9 and 1.0; 1.0 used) | CO2 broadened by **air** (the file's only self-broadening entry is CO2 fraction 0), used for an N2 background |
| far wings | CO2 sub-Lorentzian line shape (Perrin & Hartmann 1989; Wordsworth et al. 2010). Truncation distance not stated | not stated for this file |
| continua | CO2–CO2 collision-induced absorption (Gruszka & Borysow 1998; Baranov et al. 2004; Wordsworth et al. 2010), in bands 1 and 11–14 | N2–N2 collision-induced absorption from HITRAN (Richard et al. 2012), in bands 1–3 and 12–15. **N2–CO2 CIA is not in the file** (documentation: data not found in the literature) |
| dropped (gas absent) | H2O lines and continuum | H2O, CH4, H2 lines; H2O continuum; N2–CH4, N2–H2, H2–CH4, H2–H2, H2–CO2 CIA |

### Stellar tables (presets `…-ck`)

| | Ben 2 | Ben 1 |
| --- | --- | --- |
| table | `thai-ben2-sp-sw-42-dsa-mars-trappist1.npz` | `thai-ben1-sp-sw-43-dsa-ar10bar-trappist1.npz` |
| content SHA-256 | `c2e873d40e3da41dd52a9b3187fd47cf65631481b03a152363f6bfc6300e8386` | `c473fca1eceb69cbb69016d856eff67093d0183e3c5542301cf28429ae35c09d` |
| record | [thai-ben2-sp-sw-42-dsa-mars-trappist1.json](opacity/thai-ben2-sp-sw-42-dsa-mars-trappist1.json) | [thai-ben1-sp-sw-43-dsa-ar10bar-trappist1.json](opacity/thai-ben1-sp-sw-43-dsa-ar10bar-trappist1.json) |
| source files | `sp_sw_dsa_mars/sp_sw_42_dsa_mars_trappist1` (88 479 bytes, `f2730774…7099fe`) and `…_k` (10 410 696 bytes, `222801f3…f6bef4ed`) | `sp_sw_dsa_ar10bar/sp_sw_43_dsa_ar10bar_trappist1` (116 554 bytes, `704f9776…3c54b4`) and `…_k` (11 400 655 bytes, `977f2cdd…4a65c1`) |
| units, mass basis | as the longwave tables: m^2 per kg of the mixture, premixed | same |
| spectral coverage | 500–50000 cm^-1 (0.2–20 μm), 42 bands, 392 spectral points; 10 bands hold no absorber for this gas | 500–50000 cm^-1, 43 bands, 337 points; 11 bands without an absorber |
| pressure, temperature | 1 Pa – 10^6 Pa (61), 100–400 K (13) | same |
| stellar spectrum | the TRAPPIST-1 band fluxes of the file (block 2), copied into the configuration as `band_fractions`. **The documentation does not say which stellar spectrum they were integrated from**, so they are not known to be the protocol's BT-Settl 2600 K spectrum | same star, the 43 bands of this file |
| documented accuracy | below about 5 W m^-2 for all stars, against the 280-band reference | same |
| lines, broadening, far wings | as the longwave file of the same family | as the longwave file of the same family |
| continua kept | CO2–CO2 CIA in the three bands between 1111 and 1818 cm^-1 | N2–N2 CIA in the four bands between 1818 and 3175 cm^-1 |
| Rayleigh scattering | **not applied.** The per-gas coefficients of the file are recorded in the metadata; the flux-weighted Rayleigh optical depth of the column for this star is 0.0076 | **not applied**; 0.0048 |

How the importer combines terms, and what it assumes:

* `k_mix(p, T) = Σ_gas q_gas k_gas(p, T) + Σ_cont q_1 q_2 ρ(p, T) k_cont(T)`
  with mass fractions q, `ρ = p / (R_mix T)`, `k_gas` in m^2 per kg of the
  gas and `k_cont` in m^5 kg^-2. The continuum look-up (10 K steps) is
  interpolated linearly in T to the gas table's temperatures. Below 200 K
  the file itself holds the CO2–CO2 coefficients at their 200 K values.
* **This is not a premix on a common wavenumber grid.** A k-term file no
  longer contains the spectrum, so lines and continuum cannot be summed
  before the k-distribution is formed. The file prescribes random overlap
  between a gas and a continuum, and the importer forms the products of
  their k-terms (weights multiplied, k added). That is exact wherever
  either has a single term. Random overlap actually acts in bands 12 and
  14 (Ben 2) and 3, 13, 14, 15 (Ben 1).
* The band Planck fractions are computed by the provider. SOCRATES' own
  Planck table is not read. The bands stop at 3000 cm^-1: 0.03 % of
  σT^4 at 300 K lies outside and is not emitted.
* Two facts about the file layout are not in its headers. The nesting of
  pressure and self-broadening records is measured from the data on every
  import (the wrong nesting makes ln k zig-zag along pressure: roughness
  0.27 against 0.005). The continuum's density factor follows from its
  units; its magnitude was checked against the HITRAN CO2–CO2 peak
  (2·10^-43 cm^5 molecule^-2 ≈ 4·10^-3 m^5 kg^-2; the largest band-1 term
  is 1.2·10^-3). **No SOCRATES code was run**, so the reading of the format
  itself has no independent confirmation (§7).

## 4. Acquisition and conversion

```bash
# fetch the two long-wave spectral files (10 MB); pinned SHA-256, manifest.json
python -I tools/radiation/opacity/fetch_socrates.py --set core
# import them: writes opacity_cache/tables/*.npz and the records in docs/radiation/opacity
python -I tools/radiation/opacity/import_socrates.py --case thai_ben2 --record-dir docs/radiation/opacity
python -I tools/radiation/opacity/import_socrates.py --case thai_ben1 --record-dir docs/radiation/opacity

# only for the column comparison: the 350-band references (450 MB of text)
python -I tools/radiation/opacity/fetch_socrates.py --set reference
python -I tools/radiation/opacity/import_socrates.py --case thai_ben2_reference --record-dir docs/radiation/opacity
python -I tools/radiation/opacity/import_socrates.py --case thai_ben1_reference --record-dir docs/radiation/opacity
```

```bash
# stellar absorption (22 MB), for the -ck presets
python -I tools/radiation/opacity/fetch_socrates.py --set shortwave
python -I tools/radiation/opacity/import_socrates.py --case thai_ben2_shortwave --record-dir docs/radiation/opacity
python -I tools/radiation/opacity/import_socrates.py --case thai_ben1_shortwave --record-dir docs/radiation/opacity
```

`opacity_cache/` is ignored by git. Each import prints the table's content
hash; the presets pin the four above. Re-importing the two longwave tables
with the later version of the importer reproduced their hashes exactly. Another composition or spectral file:
`import_socrates.py --spectral-file <dir/file> --composition <preset>`,
after adding the file's documented source statements to `SOURCES`.

## 5. Columns (CPU, measured)

```bash
python tools/radiation/thai_column.py --preset dev-thai-ben2 --reference-table '{"file": "thai-ben2-sp-lw-350-dsa-mars.npz", "sha256": "5b2304e0917fdaecc149e2b2453a4a19e18a7cad0fac7a016ae5673a244e8a24"}'
python tools/radiation/thai_column.py --preset dev-thai-ben1 --reference-table '{"file": "thai-ben1-sp-lw-350-dsa-ar10bar.npz", "sha256": "e3ff146d05db4acc21fddf9934c10b7c20a36653b317ea81169e9b0623b153bf"}'
```

The radiation is built from the preset as a run builds it, on the preset's
40-level grid (top full level 2 Pa). Interface fluxes (thermal and stellar,
up and down), net flux and layer heating are saved in
[evidence/thai-dev-columns/](evidence/thai-dev-columns/). Two prescribed
columns: the protocol's initial state (300 K air over a 300 K surface) and
a 250 K surface under a dry adiabat capped at 160 K. Stellar input: 225
W m^-2 at μ0 = 0.5 through a transparent atmosphere over albedo 0.3.

| W m^-2 | Ben 2 isothermal 300 | Ben 2 adiabat 250/160 | Ben 1 isothermal 300 | Ben 1 adiabat 250/160 |
| --- | --- | --- | --- | --- |
| outgoing longwave at the top | 459.15 | 125.08 | 459.15 | 190.60 |
| downward longwave at the surface | 332.84 | 127.79 | 116.55 | 46.79 |
| net upward longwave at the surface | 126.31 | 93.70 | 342.60 | 174.70 |
| stellar absorbed (all at the surface) | 157.50 | 157.50 | 157.50 | 157.50 |
| Σ c_p Δp/g · heating | −332.8426 | −31.3806 | −116.5538 | −15.8985 |
| N_top − N_bottom | −332.8426 | −31.3806 | −116.5538 | −15.8985 |
| closure residual | 0.0 | 0.0 | 0.0 | 0.0 |
| heating, K/day: top layer (2 Pa) | −1557 | −32.8 | −64.5 | −1.33 |
| heating, K/day: lowest layer | −0.45 | −0.39 | −0.33 | −0.22 |

* Closure is exact because the heating is defined as the flux divergence
  over the layer masses (`heating_rate`). The check therefore covers the
  wiring - the configured c_p, g and layer masses reach the heating - and
  not the opacities.
* The isothermal column is an analytic check that needs no opacity: the
  outgoing flux must be σT^4 times the Planck fraction inside the bands,
  459.30 × 0.99968 = 459.15 W m^-2, for any gas. Both tables give it
  (asserted to 1e-6 in the tests).
* Under the same adiabat, 1 bar of CO2 emits 125 W m^-2 to space and N2
  with 400 ppmv CO2 emits 191 W m^-2 (a transparent atmosphere would emit
  221.5 W m^-2).
* Pure CO2 at 300 K cools the 2 Pa layer at 1557 K/day. A run started from
  the 300 K isothermal state loses most of its upper-level temperature in
  the first day (§6).

### Comparison with the high-resolution reference tables

The documentation names a 350-band reference for each compact file. They
were imported by the same importer (5768 and 4091 spectral points) and the
same columns solved with them:

| compact − reference | Ben 2 isothermal | Ben 2 adiabat | Ben 1 isothermal | Ben 1 adiabat |
| --- | --- | --- | --- | --- |
| outgoing longwave, W m^-2 | −0.13 (−0.03 %) | −0.59 (−0.47 %) | −0.13 (−0.03 %) | −0.36 (−0.19 %) |
| downward longwave at the surface, W m^-2 | +2.16 (+0.65 %) | +0.51 (+0.40 %) | −0.95 (−0.81 %) | +0.52 (+1.12 %) |
| largest interface-flux difference, W m^-2 | 3.21 | 0.60 | 1.11 | 0.52 |
| heating below 5 mbar: largest difference / largest reference value, K/day | 2.24 / 46.1 | 0.48 / 1.52 | 0.11 / 6.91 | 0.013 / 0.28 |
| heating above 5 mbar: same, K/day | 72 / 1630 | 1.6 / 33.1 | 0.46 / 65.0 | 0.009 / 1.34 |

Against the plan's column targets (OLR and surface fluxes within 1 % or
1 W m^-2, heating within 5 % below 5 mbar): the adiabat columns meet the
flux target in both cases; Ben 1's surface flux is 1.12 % but 0.52 W m^-2.
The isothermal 300 K columns exceed 1 W m^-2 at the surface for Ben 2
(2.16 W m^-2, 0.65 %). The Ben 2 adiabat heating differs by 0.48 K/day in
the layer at 5.6 mbar, where the reference is −1.52 K/day: **32 %, outside
the 5 % target** at that level. The 350-band files extend to 3500 cm^-1,
which accounts for the −0.13 W m^-2 of the isothermal columns.

### Stellar absorption (presets `…-ck`)

```bash
python tools/radiation/thai_column.py --preset dev-thai-ben2-ck
python tools/radiation/thai_column.py --preset dev-thai-ben1-ck
```

Same columns, same stellar input (225 W m^-2 at μ0 = 0.5, surface albedo
0.3). The thermal fluxes are those of the table above, unchanged.

| W m^-2 | Ben 2 isothermal 300 | Ben 2 adiabat 250/160 | Ben 1 isothermal 300 | Ben 1 adiabat 250/160 |
| --- | --- | --- | --- | --- |
| stellar absorbed, total | 183.86 | 181.14 | 162.12 | 161.76 |
| of which at the surface | 102.69 | 108.52 | 148.54 | 149.18 |
| of which in the atmosphere | 81.16 | 72.62 | 13.58 | 12.58 |
| planetary albedo of the column | 0.183 | 0.195 | 0.280 | 0.281 |
| Σ c_p Δp/g · heating = N_top − N_bottom | −251.678 | +41.238 | −102.972 | −3.317 |
| closure residual | −3·10^-14 | 0.0 | 0.0 | 0.0 |
| stellar heating, K/day: top layer (2 Pa) | +260 | +113 | +10.1 | +5.0 |
| stellar heating, K/day: lowest layer | +0.29 | +0.32 | +0.06 | +0.06 |

A CO2 atmosphere absorbs a third of this star's light before it reaches
the ground; N2 with 400 ppmv CO2 absorbs 6 %. The column albedos, 0.19 and
0.28, fall inside the Bond albedos THAI reports (0.18–0.22 and 0.27–0.30).
That is a plausibility check on one prescribed column at one zenith angle,
not a comparison: the inputs are not matched and nothing is in equilibrium.
The stellar tables were **not** compared with their 280-band references
(not fetched).

What this comparison is and is not. It is independent of the 17- and
15-band structure and of their k-term fits, and it exercises random overlap
in different bands. It shares the spectroscopy, this importer's reading of
the format, and the two-stream solver. **No comparison with an external
radiation code or a line-by-line calculation was made.** That reference is
still missing (§7).

## 6. Coupled runs (MX110, commit `9f8c776`)

```bash
python -m tropoi.run.radiative show-config --preset dev-thai-ben2      # config_sha256 80e5a182…
python -m tropoi.run.radiative show-config --preset dev-thai-ben1      # config_sha256 72494a31…

R=runs/radiative
# Ben 2: stop at day 2, then the same command without --until-day resumes to day 5
python -m tropoi.run.radiative run --preset dev-thai-ben2 --run-dir $R/dev-thai-ben2-lwck-heated-T21L40-9f8c776 --until-day 2
python -m tropoi.run.radiative run --preset dev-thai-ben2 --run-dir $R/dev-thai-ben2-lwck-heated-T21L40-9f8c776
# control: the same configuration and commit, never stopped, to day 3
python -m tropoi.run.radiative run --preset dev-thai-ben2 --run-dir $R/dev-thai-ben2-lwck-heated-T21L40-9f8c776-uninterrupted --until-day 3
# Ben 1
python -m tropoi.run.radiative run --preset dev-thai-ben1 --run-dir $R/dev-thai-ben1-lwck-heated-T21L40-9f8c776

python tools/radiation/thai_dev_evidence.py $R/dev-thai-ben2-lwck-heated-T21L40-9f8c776 --uninterrupted $R/dev-thai-ben2-lwck-heated-T21L40-9f8c776-uninterrupted
python tools/radiation/thai_dev_evidence.py $R/dev-thai-ben1-lwck-heated-T21L40-9f8c776
```

All four invocations ran from a clean tree at `9f8c776` (no `--allow-dirty`;
`git_dirty` is false in every `config.json`). T21 L40, dt = 1200 s, the
radiation recomputed every third step (hourly) and its interface fluxes
held in between, frictional heating on. Records and `summary.json` are in
[evidence/dev-thai-ben2-lwck-heated-T21L40-9f8c776](evidence/dev-thai-ben2-lwck-heated-T21L40-9f8c776/)
and
[evidence/dev-thai-ben1-lwck-heated-T21L40-9f8c776](evidence/dev-thai-ben1-lwck-heated-T21L40-9f8c776/).
The run directories themselves stay under the ignored `runs/`.

**Ben 2** (`config_sha256` `80e5a1824e5b378f4d97491f13e6d0aa0479fb2d41b5ad5647cdc40195dd1d28`), 1.21 s/step
(1.24 before the stop):

| day | T min (K) | T max (K) | max \|u\| (m/s) | OLR (W m^-2) | stellar absorbed (W m^-2) | mean surface T (K) | energy residual (W m^-2) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 159.2 | 304.8 | 0.34 | 414.2 | 157.4 | 301.11 | +0.552 |
| 2 | 143.3 | 315.6 | 1.52 | 378.4 | 157.4 | 301.77 | +0.231 |
| 3 | 134.7 | 325.3 | 3.32 | 361.0 | 157.4 | 301.95 | +0.168 |
| 4 | 129.1 | 330.9 | 4.81 | 348.7 | 157.4 | 301.73 | +0.134 |
| 5 | 124.8 | 339.3 | 7.47 | 338.7 | 157.4 | 301.14 | +0.116 |

**Ben 1** (`config_sha256` `72494a3143b330c9221721257d5aa4e9e3ef2379ff99723bda8520b5b0dad0c1`), 1.46 s/step:

| day | T min (K) | T max (K) | max \|u\| (m/s) | OLR (W m^-2) | stellar absorbed (W m^-2) | mean surface T (K) | energy residual (W m^-2) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 259.3 | 300.3 | 0.38 | 441.4 | 157.4 | 292.87 | -0.031 |
| 2 | 239.7 | 301.8 | 0.63 | 412.1 | 157.4 | 287.14 | -0.012 |
| 3 | 227.2 | 304.3 | 1.46 | 390.8 | 157.4 | 282.36 | +0.002 |
| 4 | 218.3 | 307.2 | 2.85 | 374.3 | 157.4 | 278.26 | +0.004 |
| 5 | 211.4 | 309.2 | 4.61 | 360.9 | 157.4 | 274.71 | +0.005 |

Both ran to day 5 without a validation failure or a table-bounds error.
"Energy residual" is the daily change of the model's total energy minus the
flux-implied tendency the coupling accumulated. The stellar absorption is
900/4 × 0.7 = 157.5 W m^-2 by construction (transparent atmosphere).

**Radiation refresh.** `radiation_interval = 3`: the column physics solves
the radiation on every third call and reuses the held interface fluxes in
between, re-deriving the heating from them over the current layer masses.
The number of solves per run was not counted;
`test_radiation_refresh_holds_and_renews_fluxes` checks the hold and the
renewal on this provider path.

**Checkpoint and resume (Ben 2).** Events: created, checkpoints at days 0,
1, 2, **resumed at day 2** from `checkpoint-s000000144.npz`, checkpoints at
days 3, 4, 5, completed. Against the uninterrupted control
(`same_configuration_and_commit: true`):
the state and surface arrays of days 1, 2 and 3 are **bit-identical**
(6 arrays, largest difference 0), and so is every
daily series over the 3 common days. The day of the resume is a
radiation-refresh step, as the configuration requires (`radiation_interval`
divides the steps of a day), so the held fluxes need not be checkpointed.

What these five days show, and what they do not:

* They are a spin-down from the protocol's 300 K isothermal start, nowhere
  near equilibrium: the global-mean OLR is still 339 W m^-2 (Ben 2) and
  361 W m^-2 (Ben 1) against 157.5 W m^-2 absorbed.
* The two gases behave differently through the same code path. Ben 2's
  upper levels cool to 125 K in five days and its mean surface stays near
  301 K; Ben 1's mean surface cools by 25 K and its coldest level is
  211 K. The initial columns of §5 give the reason: 333 W m^-2 of
  back-radiation under CO2 against 117 W m^-2 under N2 with 400 ppmv.
* **Ben 2 is heading for the table's edges.** The minimum temperature fell
  159, 143, 135, 129, 125 K over the five days, still dropping about 4 K
  per day, towards the 100 K lower bound; the maximum rose to 339 K (the
  bound is 400 K). With no stellar absorption aloft and strict bounds, a
  longer Ben 2 run in this configuration would be expected to stop at one
  of them. That is the strict default doing its job, not a reason to
  clamp. Adding the stellar absorption did not remove it (below: 116 K on
  day 4).
* Nothing here is compared with Turbet et al. (2022).

### With stellar absorption (MX110, commit `809ffdd`)

```bash
python -m tropoi.run.radiative run --preset dev-thai-ben2-ck --run-dir $R/dev-thai-ben2-ck-heated-T21L40-809ffdd
python -m tropoi.run.radiative run --preset dev-thai-ben1-ck --run-dir $R/dev-thai-ben1-ck-heated-T21L40-809ffdd
python tools/radiation/thai_dev_evidence.py $R/dev-thai-ben2-ck-heated-T21L40-809ffdd
python tools/radiation/thai_dev_evidence.py $R/dev-thai-ben1-ck-heated-T21L40-809ffdd
python tools/radiation/mars_state_diagnostics.py $R/<run> --name <run>    # where the wind maximum is
```

Clean tree, the same numerics, 5 days, 3.53 and 3.49 s/step (592 and 577
spectral points against 200 and 240). No stop and resume was exercised on
these two. Records:
[Ben 2](evidence/dev-thai-ben2-ck-heated-T21L40-809ffdd/),
[Ben 1](evidence/dev-thai-ben1-ck-heated-T21L40-809ffdd/).

**Ben 2 with stellar absorption** (`config_sha256` `fdbcf1cfd7ebcfaaf3ca887896c3cc6a049a371abdadc5985c78131304c209f3`):

| day | T min (K) | T max (K) | max \|u\| (m/s) | level of that maximum | OLR (W m^-2) | stellar absorbed (W m^-2) | mean surface T (K) | energy residual (W m^-2) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 170.8 | 303.8 | 174.2 | top layer | 417.9 | 183.4 | 299.12 | +0.429 |
| 2 | 145.9 | 311.0 | 274.0 | top layer | 384.0 | 183.3 | 298.27 | +0.147 |
| 3 | 126.2 | 317.3 | 293.0 | top layer | 367.0 | 183.3 | 297.38 | +0.105 |
| 4 | 116.2 | 323.3 | 248.6 | second layer | 354.8 | 183.3 | 296.44 | +0.083 |
| 5 | 124.1 | 328.0 | 239.7 | second layer | 345.0 | 183.3 | 295.44 | +0.066 |

**Ben 1 with stellar absorption** (`config_sha256` `ba8254e236f2f6b1bef33d469dd15fa0ca3372d93596c3cfa3bac3061253e4df`):

| day | T min (K) | T max (K) | max \|u\| (m/s) | level of that maximum | OLR (W m^-2) | stellar absorbed (W m^-2) | mean surface T (K) | energy residual (W m^-2) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 261.5 | 300.4 | 11.2 | top layer | 440.7 | 161.9 | 292.55 | -0.036 |
| 2 | 244.7 | 301.9 | 20.3 | top layer | 410.2 | 161.9 | 286.57 | -0.016 |
| 3 | 233.4 | 304.7 | 32.2 | top layer | 388.1 | 161.9 | 281.61 | -0.003 |
| 4 | 224.7 | 307.9 | 41.2 | top layer | 371.0 | 161.9 | 277.39 | +0.000 |
| 5 | 218.1 | 309.9 | 51.5 | top layer | 357.3 | 161.9 | 273.76 | +0.003 |

* Both completed without a validation failure or a bounds error, and the
  daily energy residuals are as small as in the first pair. The global-mean
  stellar absorption is 183.3 W m^-2 (Ben 2) and 161.9 W m^-2 (Ben 1)
  against 157.5 W m^-2 for the transparent atmosphere: Bond albedos of
  0.185 and 0.280 for these five days.
* **The top of the model is not credible in these runs.** In the first
  pair the largest wind was at the surface (5–7 m s^-1). Here it is in the
  top layer (2 Pa), near the equator: 174, 274, 293 m s^-1 on days 1–3 of
  Ben 2, then 249 and 240 m s^-1 in the second layer. The speed of sound
  in CO2 at those temperatures is about 190 m s^-1. Ben 1 reaches 51
  m s^-1 in the top layer and is still rising. The stellar heating enters
  the 2 Pa layer on the dayside only (+260 K/day at μ0 = 0.5 in §5) against
  about −1500 K/day of thermal cooling everywhere, and nothing damps the
  layer: the configuration has no sponge. Whether a sponge, a lower top or
  a shorter radiation interval is the remedy was not tested.
* What the two runs establish is that the configured stellar path runs
  coupled on the GPU and closes its budget. They say nothing reliable about
  the circulation aloft. Where the minimum temperatures of the tables
  occur was not looked up.
* The minimum temperature of Ben 2 reached 116 K on day 4 (bound: 100 K).

### Layer-centered thermal emission in Ben 2

The original linear longwave source builds Planck fluxes from temperatures
interpolated to the two faces. A layer-scale warm/cold alternation can be
nearly absent at the faces. In an optically thin layer, the two-stream
emission depends to first order on the mean Planck flux through that
layer. Using only the two faces therefore weakens the radiative response to
the prognosed layer temperature. This combines with the weak hydrostatic
response to the vertical computational mode of the Lorenz grid.

The `layer-linear` source retains the Planck slope inferred from the faces
and centers it on the Planck flux evaluated at the layer temperature. The
average of the two reconstructed endpoints then equals that layer flux.
The slope is bounded by the center flux so both endpoints are nonnegative.
The same optical depths, two-stream solution, and flux-divergence heating
apply. The layer Planck flux approximates the optical-depth mean for the
layer's homogeneous optics; a coarse layer crossing a sharp inversion
still needs a resolution check. The independent gray-column tests cover
the optically thin limit, the isothermal limit, and convergence to the
formal solution on a smooth column, from optically thin layers to
`Δτ = 31` per layer; at every tested layer depth the `layer-linear` net
fluxes are closer to the formal solution than the `linear` ones.

### Vertical structure of the stellar-absorbing runs (issue #47, MX110)

Measured from the native saved states (full-precision spectral
coefficients) by
[thai_vertical_coherence.py](../../tools/radiation/thai_vertical_coherence.py):
each field is synthesized exactly on the run's 32 Gauss latitudes × 64
longitudes, and the Gauss × uniform-longitude weights integrate the
products of two band-limited fields exactly, so the means, RMS values and
area-weighted Pearson correlations below are exact sphere integrals. Native
layer `k` (0-based, top first) is the issue's "level `k + 1`". The
alternating component is the least-squares amplitude `a` of `(-1)^k` in
`T_k = c0 + c1 ln σ_k + c2 (ln σ_k)^2 + a (-1)^k` over the top eight
layers of each column (equivalently a quadratic in log layer pressure);
the record is the area-weighted RMS of `a`. A correlation is undefined
when either layer has no spatial variance (identically zero, or a standard
deviation within 64 machine epsilons of the layer's own RMS, i.e. the
round-off of a constant); it is then recorded as `null`. None of the
recorded runs has such a layer.

```bash
R=runs/radiative
# dev-thai-ben2-ck with each thermal source, as the two archived runs were made:
# "linear" is the preset itself, the recorded 809ffdd configuration (fdbcf1cf…);
# "layer-linear" is the archived replay (3f2de9d8…). For a new run of the
# corrected source use --preset dev-thai-ben2-ck-layer-linear (08bc8714…).
python -c "import json; from tropoi.run.radiative.config import PRESETS; c = PRESETS['dev-thai-ben2-ck'](); \
[json.dump(c.with_(thermal_source=s).to_dict(), open(f'ben2-ck-{s}.json', 'w')) for s in ('linear', 'layer-linear')]"
python -m tropoi.run.radiative run --config ben2-ck-linear.json --run-dir $R/dev-thai-ben2-ck-heated-T21L40-9b2060c-linear
python -m tropoi.run.radiative run --config ben2-ck-layer-linear.json --run-dir $R/dev-thai-ben2-ck-heated-T21L40-9b2060c-layer-linear
python tools/radiation/thai_vertical_coherence.py $R/<run> [...] --write
```

**Reproduction.** The issue's preliminary numbers, read from the
quantized globe viewer, agree with the native states to every quoted
digit: day-5 layer means 197.30, 174.96, 197.13, 177.10 K for layers 1–4;
vorticity correlations of the pairs 1–2, 2–3, 3–4 +0.89/+0.91/+0.93 on day
1 and +0.30/+0.28/+0.36 on day 5; divergence +0.64/+0.50/+0.82 and
−0.20/−0.23/−0.06. The PR's 2.30 and 11.71 K are reproduced by the tool
(2.303, 11.706 K). The Colab T4 copy of the same run agrees with the MX110
run to ~1e-11 in `max|u|` at day 5.

**Control.** The configuration with `thermal_source="linear"`, rerun at
`9b2060c`, is bit-identical to the `809ffdd` run in all ten saved arrays
(states and slab temperatures, days 1–5): the branch leaves the historical
source path unchanged.

**One factor at a time.** Day 1 → day 5, means over the eight top layer
pairs; "δ < 0.5 to" is the deepest layer whose divergence correlation with
the layer above it is below 0.5 on day 5:

| run | source | alternating RMS (K) | ζ r, top 8 | δ r, top 8 | δ < 0.5 to | max\|u\| day 5 |
|---|---|---:|---:|---:|---:|---:|
| Ben 1, transparent SW (`9f8c776`) | linear | 0.56 → 1.02 | +1.00 → +0.98 | +0.96 → +0.87 | — | 5 m/s at 972 hPa |
| Ben 2, transparent SW (`9f8c776`) | linear | 1.49 → 1.79 | +1.00 → +0.84 | +0.98 → +0.46 | 8.4 hPa | 7 m/s at 646 hPa |
| Ben 1, stellar absorption (`809ffdd`) | linear | 0.52 → 0.84 | +0.74 → +0.53 | +0.67 → −0.20 | 0.67 hPa | 51 m/s at 2 Pa |
| Ben 2, stellar absorption (`809ffdd`) | linear | 2.30 → 11.71 | +0.94 → +0.50 | +0.83 → −0.16 | 12.6 hPa | 240 m/s at 5 Pa |
| Ben 2, stellar absorption (`9b2060c`) | layer-linear | 0.23 → 1.06 | +0.96 → +0.53 | +0.87 → −0.05 | 5.6 hPa | 241 m/s at 5 Pa |

What this establishes:

* **The temperature alternation is a radiative-source defect.** With only
  the thermal source changed (and the unchanged path shown bit-identical),
  the alternating component falls tenfold and the day-5 layer means of
  layers 1–4 become 192.3, 187.2, 187.7, 187.1 K. The new source closes
  the energy budget as the old one did (largest |residual| 0.424 against
  0.429 W m^-2).
* **The loss of wind coherence is not caused by the temperature
  alternation.** It is unchanged when the alternation is removed, and it
  occurs in Ben 1 with stellar absorption, where the alternation never
  exceeds 0.84 K and the divergence of layers 1 and 2 is already
  anticorrelated on day 1 (−0.83).
* **It appears with stellar absorption.** The `-ck` presets differ from
  the transparent-shortwave ones only in the `shortwave` entry (asserted
  by `test_thai_stellar_absorption_is_one_more_input`). Without it, Ben 1
  stays coherent and Ben 2 loses divergence coherence only slowly, with
  winds below 7 m/s whose upper-level divergence is small (a correlation
  is blind to amplitude, so this row is weak evidence either way). With
  it, the shortwave heats the 2 Pa layer of a 300 K Ben 2 column by about
  260 K/day (10 K/day in Ben 1) and the wind maximum moves to the
  unsponged top layers.

Not yet discriminated, and the remaining part of #47: whether the
incoherent upper-level flow forced by stellar heating at the unsponged top
is a numerical artifact of the upper boundary or of vertical resolution,
or a physically consistent response. The next controlled experiments are
one-factor changes of the upper-boundary treatment and of the vertical
resolution of the `-ck` Ben 2 configuration, measured with the same tool.
Radiation cadence was checked and is not the cause: the stiffest
radiative relaxation (300 K isothermal top layer, about 21 day^-1) times
the one-hour radiation interval is 0.87, below the oscillation threshold
of 1 for a held tendency.

## 7. Approximations, gaps and what was not done

* **Shortwave is an approximation in both variants, by configuration.**
  The default presets use `{"kind": "transparent"}`: no CO2 near-infrared
  absorption, the surface absorbs 70 % of the incident light. §5 measures
  what that leaves out: 73–81 W m^-2 of 225 for Ben 2, 13 W m^-2 for
  Ben 1. The `-ck` presets add the absorption but **no Rayleigh
  scattering** (`CorrelatedKShortwave` has no scattering term; the
  optical depth left out is 0.008 and 0.005), take the star's band fluxes
  from a spectral file whose stellar spectrum is not documented, and were
  not compared with a reference.
* **The model top cannot carry the stellar heating as configured.** With
  the `-ck` presets the wind maximum moves to the top layer (2 Pa) and
  reaches 293 m s^-1 in Ben 2 and 51 m s^-1, still rising, in Ben 1 (§6).
  There is no sponge and the top layer has no layer above it. A sponge or
  a lower top is needed before any longer `-ck` run; neither was tried.
* **No external reference.** §5 compares the longwave tables against the
  documented high-resolution files only. A line-by-line or external-code calculation
  on the same columns is the missing check of the spectroscopy and of the
  format reading.
* **N2–CO2 collision-induced absorption is absent** from the Ben 1 table
  (not in the source file). The CO2 lines of Ben 1 are air-broadened, not
  N2-broadened. Far-wing treatment of the Ben 1 file is undocumented.
* **Random overlap** between lines and continuum in the bands listed in
  §3, in place of a premix on a common wavenumber grid.
* **Strict bounds meet a cold model top.** Tables cover 100–400 K. A layer
  below 100 K stops the run by design. See §6 for how close the runs came.
* **Constant c_p** at its 250 K value (CO2's changes by about 15 % between
  200 and 300 K).
* **The custom spectroscopy generator is paused.** A line-by-line route
  (HITRAN lines and CIA summed on one wavenumber grid before forming the
  k-distribution) would remove the random-overlap and N2-broadening
  approximations and give the independent reference. Only its fetch step
  exists, uncommitted (`tools/radiation/opacity/fetch_hitran.py`); a
  partial download of the CO2 line list is in `opacity_cache/hitran/`.
* Not a THAI reproduction: T21 L40 to 4 Pa without a sponge, a one-layer
  2·10^6 J m^-2 K^-1 slab, the FHZ06 dry boundary layer of the Mars
  development runs, five days from the isothermal start.

## 8. Tests

```bash
python -m pytest tests/test_radiation_ktable_config.py tests/test_radiation_kdist.py tests/test_radiation_twostream.py tests/test_physics_columns.py tests/test_physics_shared.py tests/test_radiation_evidence.py tests/test_radiative_dynamical.py tests/test_layout.py tests/test_thai_vertical_coherence.py -q
```

Run on 2026-11-18 on the issue-#47 work over `9b2060c` (same machine and
versions, tables present): **169 passed, 0 failed, 0 skipped** in 256 s;
`python -m tools.architecture.generate --check` up to date. Run on
2026-11-16 at `809ffdd` (Windows, MX110, Python 3.12.12, CuPy
13.4.0), with the imported tables present, without the last file: **150 passed, 0 failed,
0 skipped** in 305 s. The same command gave 147 passed at
`9f8c776`, before the stellar presets and their three tests were added. It covers the new configuration path, the
k-distribution method against line-by-line on the synthetic spectrum, the
two-stream solver, the column physics, the frozen gray configuration
hashes, the energy criteria A1–A4 on the recorded gray runs, the coupled
gray runs with their checkpoint/resume tests, and the import-layer rules.
The full suite was not run.

`tests/test_radiation_ktable_config.py` (12 tests):

* a `correlated_k` configuration reaches layer heating through
  `build_physics`, the heating closes against the flux convergence to
  1e-12, and the fluxes are bit-identical to providers built by hand;
* a radiation interval of 3 holds the fluxes for two calls and renews them
  on the fourth;
* the table hash is part of `config_sha256`; a wrong hash and a missing
  file are refused; metadata and the container do not move the hash;
* a state outside the table raises by default, and `bounds="clamp"` counts
  the clamped layers;
* unknown options, mass fractions without a table, and a stellar provider
  without a spectrum are rejected;
* per-band zero-padded weights give the fluxes of the shared quadrature
  (1e-13) and padded points are not solved;
* a premixed table refuses another gas constant and a mass fraction other
  than 1;
* the two THAI presets differ only in `experiment_id`, `gas_constant`,
  `cp` and `longwave`, and take R and c_p from the composition;
* with the imported tables present (skipped otherwise): the pinned hash
  loads, the table was mixed for the preset's gas, an isothermal column
  emits σT^4 over the bands to 1e-6, and a 90 K column is refused;
* the default presets keep the configuration hashes of the recorded runs;
* each `-ck` preset differs from its default only in `experiment_id` and
  `shortwave`, its band fractions sum to 1 and match the table's bands,
  and (tables present) the gas darkens the column below its surface
  albedo while the heating still closes.

The first seven use the synthetic Lorentz-line fixture of
`test_radiation_kdist.py` and need no data. The importer and the SOCRATES
reader have no unit test of their own: they are exercised by the imports,
by the reproduced content hashes on re-import, and by the last test.
