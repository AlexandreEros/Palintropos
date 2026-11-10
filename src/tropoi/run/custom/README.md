# Custom preparation and execution

`prepare.py` resolves BVE/SWE initial states; `planet.py` builds a
PreparedPlanet with terrain preview/save/load; `resources.py` owns shared
resource checks. `config.py` validates custom execution, `execute.py` runs
resolved states, and `presets.py` exports named preset states.

[INITIAL_STATES.md](INITIAL_STATES.md) describes specifications, projection
residuals, state validation, scientific identity and runnable examples.
[Terrain import](../../spatial/terrain/IMPORT.md) covers elevation data and
PE prepared terrain. [Archive](../../representation/archive/README.md)
persists both; [visual](../../representation/visual/README.md) inspects them.
Building resources, projection and solver execution require CUDA; loading
verified bundles and inspecting host coefficients can run on CPU. General
custom PE initial states are not part of the BVE/SWE initial-state API.
