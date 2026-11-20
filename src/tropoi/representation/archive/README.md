# Saved data and provenance

`open_simulation` opens capsule or Held–Suarez storage behind the temporal
Simulation protocol. `schema.py` defines versioned fields/units/conventions;
`capsule.py` reads them, `writer.py` manages capsule output, and
`held_suarez.py` adapts experiment directories. `initial_state.py` and
`prepared_terrain.py` persist validated CPU data; `pe_context.py` supplies
column constants/terrain for saved-state reconstruction.

Read the [saved-run contract](../../temporal/SAVED_RUNS.md) and
[architecture provenance section](../../../../docs/ARCHITECTURE.md#output-capsules-and-provenance).
[Custom preparation](../../run/custom/README.md) writes initial-state and
terrain bundles; [visual](../visual/README.md) renders explicit requests.
Legacy manifests keep their original descriptive strings and checksums.
Readers do not rewrite evidence; generated assets are separate from it.

Viewer 2 Stage 1 discovery is available with `tropoi inspect RUN --viewer-manifest`
or `viewer_manifest(RUN)` from `tropoi.representation.archive.viewer_manifest`.
Pass the archive directory itself. Discovery uses local files, opens no network
listener, and returns the versioned manifest on stdout. Indexed operations
describe scientific selections for the viewer. `operations.structure` lists the
column, section and zonal-reduction products of a pure-σ PE archive. Zonal
mean and mass streamfunction are omitted, with a reason under `unavailable`,
when the state grid has no Gaussian latitude–longitude quadrature. Daily
radiative states also open through `open_simulation`. Portable bundles belong
to a later stage.
