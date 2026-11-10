# Command-line interface

`main.py` is the single `tropoi` entry point declared in `pyproject.toml`.
It parses arguments and dispatches to [run orchestration](../run/README.md);
`terrain.py` prepares terrain, `generate_planet.py` builds the procedural
planet view and `clear_cache.py` manages compiled CUDA cache files.

Run `tropoi --help`, `tropoi run --help`, `tropoi list presets` and
`tropoi inspect RUN_PATH` for discovery. Help, configuration validation and
inspection work without CUDA. Numerical runs and terrain projection require
CUDA; the [architecture](../../../docs/ARCHITECTURE.md#cpu-and-gpu-requirements)
defines the boundary. Detailed examples live with their
[owning subsystems](../README.md), rather than in duplicate CLI guides.
