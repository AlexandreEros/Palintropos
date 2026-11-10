# Runnable examples

These scripts exercise public preparation, persistence and visualization
interfaces. [Custom initial states](../src/tropoi/run/custom/INITIAL_STATES.md)
and [terrain import](../src/tropoi/spatial/terrain/IMPORT.md) explain their
contracts and CPU/CUDA requirements.

- `terrain_planet.py`: synthetic terrain preparation and preview.
- `mola_mars.py`: real Mars MOLA terrain and isothermal-rest checks.
- `mola_mars_evolving.py`: idealized dry perturbation, resting control and
  timestep comparison, with [dated evidence](../docs/validation/mars_evolving_2026-10-27/README.md).

Use each script's `--help` where provided and inspect its explicit physical
parameters. Worked examples demonstrate the documented path; they do not
establish real planetary climates or arbitrary-terrain accuracy.
