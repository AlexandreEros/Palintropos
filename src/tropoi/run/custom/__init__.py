"""User-defined planets and initial states for the BVE and SWE cores.

Public interface (CPU-importable; CUDA is needed only to execute a run or
to resolve a preset's state)::

    from tropoi.run.custom import (
        PlanetSpec, DiscretizationSpec, FluidLayerSpec, TopographySpec,
        InitialState, prepare_bve_state, prepare_swe_state,
        load_initial_state, run_initial_state, prepare_planet,
        read_elevation)

    world = prepare_planet(planet, discretization,
                           read_elevation("terrain.nc"))   # terrain (CUDA)
    world.plot(); world.view()                   # static figure, 3D globe

    prepared = prepare_bve_state(planet, discretization,
                                 zeta=lambda lat, lon: ...)   # or coefficients
    prepared.report.lines()                      # residuals, means, support
    prepared.state.save("my_state")              # versioned JSON + .npy
    state = load_initial_state("my_state")       # fresh process, no CuPy
    run_dir = run_initial_state("my_state", duration_days=1.0)

See docs/CUSTOM_INITIAL_STATES.md and docs/TERRAIN_IMPORT.md.
"""
from tropoi.representation.archive.initial_state import (
    INITIAL_STATE_FORMAT, INITIAL_STATE_FORMAT_VERSION, InitialState,
    InitialStateError, load_initial_state, save_initial_state)
from tropoi.run.custom.config import CustomRunConfig
from tropoi.run.custom.execute import (INITIAL_STATE_DIR, execute_custom_run,
                                       run_initial_state)
from tropoi.run.custom.planet import PreparedPlanet, prepare_planet
from tropoi.run.custom.prepare import (PreparationError, PreparationReport,
                                       PreparedState, ProjectionPolicy,
                                       SampledField, SampledFunction,
                                       SpectralCoefficients, StateSampling,
                                       pad_coefficients, prepare_bve_state,
                                       prepare_swe_state, state_sampling)
from tropoi.run.custom.presets import (initial_state_from_config,
                                       initial_state_from_preset,
                                       planet_spec_for_run_config,
                                       topography_spec_for_run_config)
from tropoi.spatial.specification import (DiscretizationSpec, FluidLayerSpec,
                                          PlanetSpec, SpecificationError,
                                          TopographySpec)
from tropoi.spatial.terrain.raster import (ElevationImportError,
                                           ElevationRaster, read_elevation)

__all__ = [
    "CustomRunConfig",
    "DiscretizationSpec",
    "ElevationImportError",
    "ElevationRaster",
    "FluidLayerSpec",
    "INITIAL_STATE_DIR",
    "INITIAL_STATE_FORMAT",
    "INITIAL_STATE_FORMAT_VERSION",
    "InitialState",
    "InitialStateError",
    "PlanetSpec",
    "PreparationError",
    "PreparationReport",
    "PreparedPlanet",
    "PreparedState",
    "ProjectionPolicy",
    "SampledField",
    "SampledFunction",
    "SpectralCoefficients",
    "StateSampling",
    "SpecificationError",
    "TopographySpec",
    "execute_custom_run",
    "initial_state_from_config",
    "initial_state_from_preset",
    "load_initial_state",
    "pad_coefficients",
    "planet_spec_for_run_config",
    "prepare_bve_state",
    "prepare_planet",
    "prepare_swe_state",
    "read_elevation",
    "run_initial_state",
    "save_initial_state",
    "state_sampling",
    "topography_spec_for_run_config",
]
