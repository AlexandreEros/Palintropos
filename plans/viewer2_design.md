# Palintropos Viewer 2

Agreed technical design · 2026-11-20

## Purpose and architecture

One installed, offline-capable viewer opens BVE, SWE, PE and radiatively coupled PE archives. The proposed command is `tropoi view <capsule>`. A read-only Python provider serves packaged browser assets and selected scientific data on loopback for the duration of the session. Portable selections and optional self-contained HTML use the same manifest, selection semantics and rendering modules. Existing Simulation.view, Simulation.sky, plotting APIs and standalone exporters retain their contracts.

Archive adapters interpret storage formats. Scientific evaluators implement versioned quantity definitions. The interface discovers operations by their input requirements and available evaluators. Experiment names and solver flags do not select a frontend. Future experiments using an understood format inherit its operations automatically.

## Repository baseline

- [open_simulation](../src/tropoi/representation/archive/capsule.py) and the Held–Suarez adapter expose read-only frames, named state and times through Simulation. Canonical coefficient arrays support memory mapping. Historical time inference is limited to documented formats.
- [The quantity catalogue](../src/tropoi/representation/visual/quantities.py) distinguishes prognostic, derived, static and recorded quantities. Some evaluators use CUDA. [Figure composition](../src/tropoi/representation/visual/compose.py) already supports normalization, vector layers, diagnostic cadence and PNG rendering.
- [The atmosphere exporter](../src/tropoi/representation/visual/ATMOSPHERE_VIEW.md) shares a WebGL terrain globe and flat map. Its interactive surface mode embeds quantized temperature, geopotential and wind columns plus surface pressure for every selected frame. Browser interpolation already synchronizes geometry, hover, colours, masks and streamlines. Extract rendering and data access incrementally, using existing exporter and browser parity tests as references.
- [PE reconstruction](../src/tropoi/representation/diagnostics/pe_surfaces.py), host spectral synthesis, [Held–Suarez reductions](../src/tropoi/representation/diagnostics/held_suarez_sections.py) and [saved-column sky optics](../src/tropoi/representation/visual/sky.py) supply scientific reference implementations.
- The existing reader opened the canonical and Held–Suarez fixtures below. It rejected both radiative directories during format detection. Existing Williamson-5 and evolving-Mars capsule tests passed (nine tests). Viewer 2 acceptance criteria below describe future verification.

## Viewer manifest v1

Readers expose a lightweight manifest in memory. Portable exports serialize it. Existing evidence files retain their original schemas, hashes and paths.

| Block | Contract |
|---|---|
| Schema and source | Major/minor version, archive format, run identity, reader version, source commit and checksums where recorded. |
| Axes and geometry | Named dimensions, coordinate resources, units, order, staggering, vector basis, longitude convention, planet radius and height datum. Distinguish spectral layout and normalization, retained truncation, native grid and display mesh. Declare full levels and interfaces independently. |
| Time axes | Per-resource exact timestamps or documented indexing rule, units and stored/inferred origin. Record gaps, sampling, averaging windows and counts where known. Snapshot, slab, series and aggregate axes can differ. |
| Resources | Stable local references to arrays, members, indexed frames or portable chunks with dtype, layout, shape and known integrity evidence. Dense values live outside the manifest. |
| Fields and diagnostics | Physical definition, quantity ID, units, scalar/vector components, dimensions, axes, masks, support and cadence. Identify stored resources or a versioned recipe with dependencies. Aggregates record weights, coverage and source window. |
| Physics and provenance | Documented gravity, composition, insolation geometry, stellar parameters, radiative scheme, bands, conventions and input origins. Distinguish recorded values, inferred metadata, derived products and explicit user assumptions. |

Operations declare required quantities, coordinates, geometry and evaluator support. A horizontal scalar enables a map. Geographically oriented wind enables vectors and instantaneous streamlines. A vertical axis and physical coordinate mapping enable applicable sections and surfaces. A stored latitude–level aggregate can enable a zonal section without enabling a point profile. Omit physically inapplicable controls. Explain missing resources when that helps a user open a fuller archive.

Prefer validated stored metadata, followed by documented legacy rules with provenance. Accept additive optional fields in compatible minor versions. Expose readable metadata and explain affected operations for unsupported versions or conventions. One radiative adapter handles both fixtures, checking packed shapes, saved days, slab grid order and vertical coordinates against saved interfaces. No experiment-specific opener or archive migration is required.

## Selection, evaluation and delivery

Maintain a collection of view selections. Each identifies run, actual saved time, quantity, level or surface with units and convention, geographic cursor or transect, and display settings. Link groups explicitly synchronize chosen dimensions while allowing others to differ. Support same-run comparison of times, fields and levels first. Extend to cross-run comparison using explicit time and coordinate matching, compatible units and documented interpolation. Show the actual selections and unmatched samples. Share colour scales when quantities and units are comparable. Discard stale responses per view.

Request maps, surfaces, profiles, sections or reductions. The provider chooses the source reads. A point in a spectral field can require all harmonic orders of that field. Synthesize only the requested geometry and necessary dependencies. For rapid surface scrubbing, optionally stream one frame’s required columns and reuse browser interpolation. Return binary arrays and masks. Bound provider and browser caches in bytes across views, keying entries by source, selection, recipe, grid and precision. Keep caches and exports outside authoritative archives by default. Preserve random access to every saved timestamp. No full dense conversion, cloud dependency or simulation rerun is required.

Local Python introduces a process dependency in exchange for direct archive access. Portable selections exchange full archive scope for autonomous playback. The browser has no CUDA dependency. The provider can use existing CuPy/CUDA routines where beneficial or host evaluators on machines without NVIDIA hardware. Reuse working algorithms. Benchmark cold and warm map, slice and reduction requests, transfer cost, first-frame latency and peak memory before choosing defaults, cache sizes or chunk layouts. Verify numerical agreement using documented operation-specific absolute and relative tolerances and mask comparisons. Performance parity is unnecessary.

## Six-stage delivery and acceptance

1. **Discovery and compatibility.** Implement inspectable manifests, adapters and operation predicates. Exercise every fixture below, including both Held–Suarez archive forms. Trace discovery I/O to verify it reads metadata, indices, coordinates and required headers without fetching scientific frame payloads. Check unknown conventions, missing files and partial products.
2. **Offline globe, map and PNG.** Extract camera, terrain, lighting, graticule, normalization, wind, field selection and saved-time playback modules. Deliver stepping, looping, playback speed, exact time readout, first same-run comparisons and 2D PNG summaries with reproducible selection metadata. Preserve neutral display illumination. Where directional and time/frame metadata suffice, add the star direction, background position and geometric day/night lighting. Apparent disk size needs angular-size information. Label geometric shading separately from archived irradiance. With external networking disabled and no NVIDIA hardware, exercise applicable views on all six fixtures. Jump directly to the final frame without reading preceding frames. Verify interactions, masks, PNG values, scales, units and timestamps. On GPU-equipped systems, compare accelerated results to CPU references and benchmark before selecting a default.
3. **Linked atmospheric structure.** Start with meridional and zonal sections at the cursor, a column profile, and corresponding planes on the globe. Support native levels, pressure, nominal σ and geopotential height where defined. Preserve hydrostatic versus nominal level placement, log-σ interpolation, height datum and existing underground/out-of-bracket mask codes. Test agreement at shared points, endpoints, longitude wrap and rapid selection changes. **Subsequent 3D cutaway milestone:** allow interactive globe clipping and wedge removal, expose atmospheric layers on the cut faces and intersect terrain with those faces using the same datum, masks and display exaggeration. Rotate and move a wedge through the Mars mountain. Compare the exposed profiles and terrain intersection against the linked sections and verify that below-ground atmosphere stays masked.
4. **Zonal, circulation and radiative diagnostics.** Offer zonal wind for jets, zonal temperature and defined eddy products. Prefer stored aggregates when their definition and sample window match. For native pure-σ meridional overturning, integrate downward from zero at the top, adding 2πa cos(φ) ⟨v Δp⟩ / g per layer, with local Δp = p_s Δσ and a longitude mean over complete coverage. Units are kg/s, positive for northward transport above the interface. Mean v alone is insufficient. Pressure-coordinate transport requires explicit partial layers and terrain coverage. Verify an analytic transport case, longitude-independent means and parity with existing Held–Suarez reductions on identical samples. Extend linked comparison to compatible runs. Register RT fluxes, heating rates, spectral-band contributions and optical diagnostics as discoverable quantities, profiles or sections according to their actual dimensions. Record flux direction, band bounds and weights, units, vertical placement, sampling and whether stored or derived. Test band totals against a compatible broadband quantity where available and suppress unresolved band/column views for scalar-only records. A configured opacity table alone cannot establish a saved band flux.
5. **Sky and physical illumination.** Feed a selected column, documented composition and star into the existing single-scattering Rayleigh model. Distinguish display lighting, geometric incidence, recorded irradiance (including location, level and averaging) and modeled directional sky radiance. Use surface incident radiation for terrain when supported. Distinguish direct and diffuse components where available and avoid applying a second cosine factor to horizontal irradiance. Identify blackbody and scattering approximations, user assumptions, fixed/auto exposure, clipping and false-colour scales. Verify fixed-input sky agreement with the existing explorer, stellar alignment across globe and sky, the terminator for known geometry and limited capabilities when only global flux exists.
6. **Portable distribution.** Add a portable offline selection and size-budgeted single-file HTML backed by the same renderer and selection contract. Estimate size before export. Record included times, fields, precision, masks, recipes and provenance. Disconnect both network and provider and reproduce the selected maps, values, masks, units and timestamps. Preserve existing standalone export behavior.

## Delivery status · 2026-11-20

Stages 1 and 2 are delivered. PR #58 also promotes a bounded part of Stages 3 and 4:

- **Delivered from Stage 3:**
  - A linked geographic cursor, drawn on the map and as lines on the globe.
  - A native-level column profile at the cursor against geopotential height, pressure or nominal σ.
  - Meridional and zonal sections on native levels with log-σ axes.
  - Level selection linked back to the map.
- **Delivered from Stage 4:**
  - Native-grid zonal-mean [u], [v] and [T] for one saved state, or an equal-weight mean of a consecutive range.
  - The pure-σ meridional mass streamfunction, as specified above.
- **Verification:**
  - An analytic transport case, and a wavenumber-1 case where [pₛ v] ≠ [pₛ][v].
  - Parity with `reduce_sections` on full Held–Suarez states.
  - Parity with the Ben 2 run accumulator (`means.npz`, days 16–30 and 61–120), within 2 × 10⁻⁹.
  - The equatorial jet against the independent sponge-study profiles at days 30, 60, 90 and 120, within 10⁻⁹ m s⁻¹.
- **Not in scope:**
  - Sections on pressure or height surfaces.
  - Eddy products, cross-run linking, the 3D cutaway, radiative-transfer diagnostics, sky optics and portable HTML.

## Fixture acceptance matrix

| Archive | Verified local inventory | Required result |
|---|---|---|
| docs/runs/20260712T023305Z_two-vortices_rot24h_r4_l21_dt10h_501bce43 | BVE, 25 frames over days 0–10, zeta | Vorticity and nondivergent wind, legacy interval time provenance, horizontal controls only. |
| docs/runs/20260730T011700Z_williamson5_rot23p93h_r4_l63_dt120h_45406d82_668e6c9a | SWE, days 0, 5, 10, 15, zeta/delta/phi | Wind, layer depth and free surface. Check band-limited mountain and free-surface height equals depth plus terrain. |
| docs/runs/20261027T175327Z_orographic-thermal-wave_rot24p62h_r3_l21_dt11080p3s_3a46f966_71c5441a | PE, nine frames and prepared terrain | Compare globe and flat map to existing Mars export for pressure, σ and height. Match masks exactly and values within documented display precision. Exercise clipping and terrain intersections. |
| docs/runs/hs-T42L20-prod-002 and full local runs/hs-T42L20-prod-002 | Published statistics/series with zero frames, full local archive with 120 | Core opens to supported diagnostics. Full archive adds playback, sections and columns. Verify sampling provenance for both. |
| docs/runs/MOST_H | Gray RT, 60 atmosphere/slab pairs, 25 explicit levels, recorded mean count 30 | Resolve actual interfaces, daily series, slab snapshots and independently sampled means. |
| runs/dev-thai-ben2-ck-heated-T21L40-f44c039-layer-linear-dt600-30d | Multiband RT, 30 daily pairs, 40 stretched levels, recorded mean count 15 | Same radiative adapter. Direct day-30 access, separate slab and lowest-air temperatures, exact vertical geometry. Offer RT diagnostics at available resolution and cadence. All 30 saved states are selectable across the 4,320-step integration. |

Across fixtures, verify archive identities and available receipts before and after viewing/export. Exercise multiple linked selections under the shared cache limit. Record bytes read and transferred, backend, approximation, precision and tolerances so regressions can be distinguished from valid differences in sampling or display.
