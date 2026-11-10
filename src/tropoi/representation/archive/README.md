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
