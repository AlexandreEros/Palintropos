from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TectonicParams:
    """
    Controls how strongly tectonics modify the terrain.
    """
    dt: float = 1.0                 # "tectonic time step"
    kappa_height: float = 1e-6      # spectral diffusion (erosion) for height
    kappa_strain: float = 5e-6      # diffusion/smoothing for strain field
    noise_strength: float = 0.3     # how much new mountain noise to inject
    l_cut_noise: int = 8            # only add noise for l >= l_cut_noise
    gamma_activity: float = 2.0     # nonlinearity on activity (S_norm ** gamma)
    renormalize_height: bool = True # optionally keep RMS of h roughly fixed


