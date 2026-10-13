"""User-defined planets and initial states for the BVE and SWE cores.

Public interface (CPU-importable; CUDA is needed only to execute a run or
to resolve a preset's state)::

    from tropoi.run.custom import (
        PlanetSpec, DiscretizationSpec, FluidLayerSpec, TopographySpec,
        InitialState, load_initial_state, run_initial_state)

    state = InitialState("bve", zeta_lm, planet, discretization)
    state.save("my_state")                       # versioned JSON + .npy
    state = load_initial_state("my_state")       # fresh process, no CuPy
    run_dir = run_initial_state("my_state", duration_days=1.0)

See docs/CUSTOM_INITIAL_STATES.md.
"""
from tropoi.representation.archive.initial_state import (
    INITIAL_STATE_FORMAT, INITIAL_STATE_FORMAT_VERSION, InitialState,
    InitialStateError, load_initial_state, save_initial_state)
from tropoi.run.custom.config import CustomRunConfig
from tropoi.run.custom.execute import (INITIAL_STATE_DIR, execute_custom_run,
                                       run_initial_state)
from tropoi.run.custom.presets import (initial_state_from_config,
                                       initial_state_from_preset,
                                       planet_spec_for_run_config,
                                       topography_spec_for_run_config)
from tropoi.spatial.specification import (DiscretizationSpec, FluidLayerSpec,
                                          PlanetSpec, SpecificationError,
                                          TopographySpec)

__all__ = [
    "CustomRunConfig",
    "DiscretizationSpec",
    "FluidLayerSpec",
    "INITIAL_STATE_DIR",
    "INITIAL_STATE_FORMAT",
    "INITIAL_STATE_FORMAT_VERSION",
    "InitialState",
    "InitialStateError",
    "PlanetSpec",
    "SpecificationError",
    "TopographySpec",
    "execute_custom_run",
    "initial_state_from_config",
    "initial_state_from_preset",
    "load_initial_state",
    "planet_spec_for_run_config",
    "run_initial_state",
    "save_initial_state",
    "topography_spec_for_run_config",
]
