# The shared physical world (`tropoi.physics`)

Reconciliation of the sky page (PR #22) and the radiative–dynamical
foundation (PR #29): both now consume one description of composition,
molecular properties and stellar illumination, while keeping the distinct
radiation methods each needs (single-scattering Rayleigh optics for the sky;
two-stream gray/correlated-k schemes for the climate).

## What is shared, and where it lives

| Responsibility | Module | Packaged data |
| --- | --- | --- |
| Fundamental and nominal constants | `physics.constants` | — (exact SI values and IAU conventions are definitions, not data) |
| Species identity and molar masses | `physics.species` | `data/species.toml` (CIAAW 2021 atomic weights; molar masses computed per molecule) |
| Explicit compositions | `physics.composition` | `data/mixtures.toml` (climate and illustrative preset families, kept distinct) |
| Heat capacities (RRHO) | `physics.thermodynamics` | `data/thermodynamics.toml` |
| Rayleigh optical records | `physics.rayleigh` | `data/rayleigh.toml` |
| 1-D hydrostatic optics column | `physics.column` | — |
| Stars (intrinsic) and apparent stars | `physics.star` | `data/stars.toml` |
| Kepler orbits | `physics.orbit` | — |
| Planetary rotation | `physics.rotation` | — |
| Illumination (star + orbit + rotation) | `physics.insolation` | — |
| CIE colour matching (sky display only) | `representation.visual.colour` | `representation/visual/data/CIE_xyz_1931_2deg.csv` |

Design rules, enforced by construction and tests:

- **Identity is separate from property coverage.** A `Composition` may name
  any known species; each consumer checks the coverage it needs
  (`require_coverage`) and refuses missing species by name. Nothing is
  silently ignored, and no composition is ever inferred from `r_dry`.
- **Strict fractions.** Stated mole or mass fractions must sum to 1 within
  `SUM_TOLERANCE = 1e-6` (float round-off headroom). Rounded statements go
  through the explicit `Composition.normalized` factory, which preserves the
  stated fractions and their total as provenance and refuses totals further
  than 2 % from 1.
- **Data as data.** Measured/fitted records, catalogues and presets are TOML
  files under `tropoi/physics/data/`, loaded once at import through
  `importlib.resources` (outside every hot loop) and SHA-256-identified.
  Records select dispersion models by identifier (the refractiveindex.info
  formula number); equations stay in Python. A species may carry several
  optical records with different validity ranges; out-of-range wavelengths
  are refused, never extrapolated. New radiative-experiment `config.json`
  files carry a descriptive `physics_datasets` block
  (`tropoi.physics.species.dataset_identities`) naming each dataset's
  SHA-256, so a result can be traced to the data that produced it. The block
  is provenance only — never part of `config_sha256`.
- **Orbit / star / rotation separation.** `KeplerOrbit` holds the six
  elements of a bound ellipse in SI with explicit frame/epoch conventions
  and takes the gravitational parameter as an argument (or from a containing
  system). `Star` owns mass, radius, luminosity and T_eff with an explicit
  spectral model (blackbody constructors derive a missing quantity from the
  other two and reject contradictions; a supplied non-blackbody spectrum
  object is accommodated). Apparent angular size and irradiance are viewing
  geometry (`Star.seen_from` → `ApparentStar`); the sky's
  temperature-plus-angular-radius workflow needs no invented mass or
  distance. `Rotation` owns the spin axis, rate and phase.
  `StellarIllumination` combines the three (or states the reduced numbers
  explicitly) and drives the retained forcing modes: prescribed,
  diurnal-mean, instantaneous, tidally locked.
- **Irradiance-based factories state their reference.**
  `KeplerOrbit.from_irradiance(..., reference=...)` requires
  `"semimajor_axis"`, `"periapsis"`, `"apoapsis"` or `"time_mean"`; nothing
  is chosen silently. `StellarIllumination.prescribed` documents that its
  `stellar_constant` is the flux at the semimajor-axis distance.

## Removed duplication and the canonical choices

Before this reconciliation the repository carried two of each:

| Quantity | Climate side (`physics.thermodynamics`) | Sky side (`spatial.optics`) | Canonical now |
| --- | --- | --- | --- |
| Molar gas constant | literal `8.31446261815324` | `K_B · N_A` | `K_B · N_A` (identical value; exact by definition) |
| Molar masses | per-molecule literals from pre-2021 standard atomic weights | per-molecule from CIAAW 2021 abridged/conventional atomic weights | computed per molecule from the one CIAAW 2021 table in `species.toml` |
| Mixture presets | `MIXTURES` (round numbers, silently normalized after a 2 % check) | `COMPOSITION_PRESETS` (strict, sourced) | one `COMPOSITIONS` registry holding **both families unchanged**, the round-numbered climate mixtures now explicitly normalized with their stated totals preserved |
| Second radiation constant | literal `1.438776877` cm K | — | `h c / k_B` exactly (`1.43877687750…`) |
| Stefan–Boltzmann, IAU solar values | — | literals in `star.py` | derived/nominal values in `physics.constants` |

The sky side's CIAAW 2021 values are canonical because they carry the
documented, verifiable provenance (Prohaska et al., Pure Appl. Chem. 94,
573, 2022). Measured effect of the change on the climate presets
(old `GasMixture` arithmetic reproduced exactly, 2026-11-12):

| mixture | R old (J kg⁻¹ K⁻¹) | R new | relative change |
| --- | --- | --- | --- |
| earth_dry_air | 287.0457 | 287.0426 | −1.1e−5 |
| mars | 191.2431 | 191.2450 | +9.8e−6 |
| venus | 191.3586 | 191.3607 | +1.1e−5 |
| titan | 303.4384 | 303.4318 | −2.2e−5 |
| jupiter | 3592.9149 | 3592.7530 | −4.5e−5 |
| saturn | 3592.3582 | 3592.1905 | −4.7e−5 |
| thai_ben1 | 296.7353 | 296.7289 | −2.1e−5 |
| thai_ben2 | 188.9243 | 188.9264 | +1.1e−5 |

`c_p` of earth_dry_air at 260 K shifts by the same −1.1e−5 relative. No
experiment configuration is affected: `RadiativeConfig` stores plain
`gas_constant`/`cp` numbers (287.04, 3605.38, …), and the frozen
`config_sha256` of the committed evidence run
`earth-gray-dry-T21L25-300d` is reproduced bit-for-bit by today's code
(asserted in `tests/test_physics_shared.py`). The sky page's numbers are
unchanged (its masses were already the CIAAW values, and the angular-size
arithmetic keeps the historical km-constant expressions bit-for-bit).

## Compatibility contracts kept

- **Config hashes and resume.** Insolation stays a `{"kind": ...}` spec in
  `RadiativeConfig`, resolved by `run.radiative.model.build_insolation` into
  the shared objects. The canonical JSON and `config_sha256` are
  byte-identical, so existing checkpoints resume unchanged.
- **Signatures.** `DiurnalMeanInsolation`/`InstantaneousInsolation`
  signatures keep the legacy `orbit` keys (`stellar_constant`, `period`,
  `eccentricity`, `obliquity_deg`, `perihelion_longitude_deg`,
  `time_of_perihelion`); component-built illuminations add a `components`
  block.
- **Forcing numerics.** The Kepler solve and the daily-mean/instantaneous
  formulas are the pre-reconciliation expressions verbatim
  (`StellarIllumination` reuses them; `tests/test_physics_columns.py` and
  `tests/test_physics_shared.py` assert the closed forms).
- **Planet presets.** `physics.planets.PLANETS` keeps every stated
  fact-sheet scalar exactly and attaches the shared `Star`, `KeplerOrbit`
  and `Rotation` objects; the stated stellar constants agree with the IAU
  nominal Sun at the fact-sheet distances to ~0.1 % (the rounding of those
  references).

## Packaging

The wheel built at integration time (`pip wheel --no-deps`, 2026-11-12,
palintropos 0.3.0) contains:

```text
tropoi/physics/data/{species,thermodynamics,rayleigh,mixtures,stars}.toml
tropoi/representation/visual/data/CIE_xyz_1931_2deg.csv
tropoi/representation/visual/{sky_view,terrain_globe}.html
tropoi/spatial/transforms/cuda/{legendre,sh_matrix}.cu
```

All loading goes through `importlib.resources`;
`tests/test_physics_shared.py` checks the pyproject package-data
declarations against the files on disk, and the CIE table's SHA-256 is
checked on every load.

## Test evidence at the integration head

At `fa73a41` (branch `reconcile/sky-radiative-physics`, 2026-11-12):

| Environment | Selection | Result |
| --- | --- | --- |
| Colab, NVIDIA L4, CUDA 13.0, CuPy 14.0.1, Python 3.13.16, Linux | full suite (`all`, routine + slow) via `notebooks/Palintropos-Colab-CUDA-Regression.ipynb` | **1696 passed, 50 skipped** (all skips environmental: 28 need a browser, the rest local-only capsules and the opt-in W5 acceptance run); evidence ZIP `palintropos-cuda-fa73a41026a6-all-20261112T143540570728Z` |
| Local Windows, GeForce MX110, CUDA 11.8, CuPy 13.4.0, Python 3.12 | full suite minus the browser file (collected at `caa048f^`; the three test files edited afterwards re-ran separately and pass) | **1705 passed, 5 skipped** in 1 h 44 min |
| Local Windows (same) | `tests/test_sky_view_browser.py` (headless Chrome parity) | **11 passed** |
| GitHub Actions | CPU-only CI and the architecture check run per push | per push |

`python -m tools.architecture.generate --check` passes on the committed
snapshot.

## Deliberately out of scope

No spectroscopic opacity collection (the correlated-k module still carries
**no real opacity data**), no Rayleigh-to-climate coupling (the gray schemes
take prescribed optical-depth profiles, not compositions), no heavier
storage dependency (HDF5 waits for a dataset that needs it), and no
general-purpose data platform. The boundaries above are where those would
attach: absorption records would become a third property dataset beside
thermodynamics and Rayleigh optics, selected by the same coverage checks.
