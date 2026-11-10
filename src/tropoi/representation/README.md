# Recorded and displayed representations

This layer turns resolved/saved states into archives, diagnostics and views.
It may consume temporal and spatial resources; they do not depend on it.

- [archive](archive/README.md): schema, capsule readers/writers, prepared
  terrain/initial-state persistence and Held–Suarez storage adapters.
- [diagnostics](diagnostics/README.md): per-core conservation/time series, CPU native-level and
  derived-surface PE reconstruction, and Held–Suarez climate reductions.
- [visual](visual/README.md): figure fields/specifications, normalization,
  timelines, renderers and self-contained terrain/atmosphere pages.
- `identity.py`: scientific configuration hash exclusions shared by readers
  and run configurations.

The public read-only [Simulation API](../temporal/SAVED_RUNS.md) belongs to
temporal; [run](../run/README.md) owns execution. Derived figures go under
`RUN/assets/`; they do not change run identity or scientific evidence.
Display interpolation and quantization must stay distinct from solver state.
