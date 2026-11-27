# Viewer 3: one selection model for atmospheric and airless archives

Viewer 3 replaces Viewer 2's three request paths (`frame`, `structure` and
`diagnostic`) with one selection endpoint, and its Vertical-structure panel and
series chart with the four-coordinate explorer. These files are the evidence
that the replacement preserves Viewer 2's results and reads airless archives
faithfully.

## Equivalence with Viewer 2 (`selection-equivalence.json`)

`tools/viewer/selection_equivalence.py` evaluates matched selections in two
ways:
- the old way: `ViewerSession.frame` and `ViewerSession.structure` of `main`
  before the change, run in a subprocess;
- the new way: `ViewerSession.select`.

It compares every result. Nothing is tolerance-based.

| Archive | What it covers |
| --- | --- |
| `hs-T42L20-prod-002` | Held–Suarez T42L20, flat |
| `20261027T175327Z_orographic-thermal-wave_…` | Mars with MOLA terrain (prepared terrain, digest-checked), PE T21 |
| `dev-thai-ben2-ck-heated-T21L40-f44c039-layer-linear-dt600-30d` | THAI Ben 2, radiative schema, slab surface |
| `MOST_H` | Mars with the Monin–Obukhov boundary layer, radiative schema, slab surface |

The checks:
- **Maps:** every map quantity on its top, middle and bottom native levels and,
  where offered, on σ = 0.5, half the reference pressure and Z = 5 km, at the
  first and last saved times. The binary frames are byte-identical: values,
  terrain, layer heights, winds and masks.
- **Columns:** snapped and unsnapped, against Z, p and σ. Values, the level
  table, the location, the time, the plotted coordinate and the slab Tₛ marker
  are exactly equal.
- **Sections:** meridional and zonal. Values, points, σ, location, p and Z are
  exactly equal.
- **Zonal means:** of one state and of four states, for [T], [u], [v] and ψ.
  Values, latitudes, σ and the sample record are exactly equal.

Pressure on an isobar and Z on a constant-Z surface are left out because
they are constants that neither page offers.

Result: **742 comparisons, 0 differences** (old `b595e01`, new: see the file).

## Columnar (airless) archives

Read through the same endpoint and checked against the tier-2 arrays read
directly. Tests: `tests/test_viewer_airless.py` and
`tests/test_viewer_explorer_browser.py`.

- **Timestamps.** Every surface sample's incident shortwave (plane 1) is
  recomputed from the archived illumination at the reconstructed time and
  must be equal:
  - in the tests, exactly, on runner-written latlon, Gaussian and point
    archives, including a short final transient segment that keeps the
    cadence;
  - on the real archives: Mercury 4° agrees to 2.4 × 10⁻¹⁰ W m⁻² (written on a
    T4, recomputed on the CPU), where a sample one step early would differ by
    2.39 W m⁻²; the Ceres transient agrees exactly, where one step early would
    differ by 0.005 W m⁻².
- **Cadences.** The surface record (every 2 steps in the tests) and the cell
  record (every 6) keep their own axes. A cell map at a surface-only index is
  refused.
- **Maps.** The skin map is surface plane 0 and each cell map is the profile
  record at that cell, through all three vertical representations. Both are
  equal to the archived values cast to float32. A depth between two cell
  centres is refused, not interpolated.
- **Profiles, sections, histories.**
  - The profile is the archived cell-centre temperature of the nearest
    column, on the archived faces (signed −d for height).
  - The skin marker is the surface sample of the same step.
  - Sections use the native row of columns and the regular cell edges.
  - Histories are the concatenated kept cycles at their own cadence.
- **Point collections.** No map, section, zonal mean or area weights; the
  every-step record is used.
- **Bounded memory.** On the real 1.6 GB 4° Mercury archive, a map plus a full
  540-sample time × depth history peak below 16 MB (tracemalloc).
- **Security.** Every file goes through `resource()`. A symlink inside an
  archive is refused, and the selection endpoint accepts exactly one
  parameter, `q`, of at most 4 KiB.
- **Genericity.** No viewer source file names a planet or a test case
  (tested).

## Not offered in this release

- Zonal reductions of columnar archives.
- Vertical reconstruction between native substrate depths.
- Mixed atmosphere–substrate surfaces.
- Integrals and other statistics.
- Atmospheric surfaces of constant σ, p or Z as time histories: histories use
  native levels.
- Coordinates other than geocentric longitude, latitude and elapsed time.
