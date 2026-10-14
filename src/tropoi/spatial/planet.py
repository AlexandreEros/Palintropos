import numpy as np
import cupy as cp

from typing import Optional, Tuple
from tropoi.spatial.operators.spectral_operators import SpectralOperators
from tropoi.spatial.transforms.optimized_geodesic_sh import GeodesicSphericalHarmonics
from tropoi.spatial.grids.grid_base import GridGeometry as GridGeometryBase
from tropoi.spatial.grids.geodesic_grid import GeodesicGridGeometry
from tropoi.spatial.environment import PlanetaryParameters
from tropoi.spatial.terrain.elevation_data import ElevationData
from tropoi.spatial.terrain.terrain_spectral import generate_spectral_terrain_gpu, SpectralTerrainParams
from tropoi.spatial.terrain.tectonics import TectonicParams, tectonic_update_step

class Planet:
    """
    A procedurally generated planet with realistic physical properties.

    This class separates concerns:
    - Physical parameters (mass, rotation, etc.)
    - Grid geometry
    - Elevation data
    - Noise generation
    """

    def __init__(self,
                 params: PlanetaryParameters,
                 grid: GeodesicGridGeometry,
                 elevation: ElevationData,
                 sh: GeodesicSphericalHarmonics,
                 so: SpectralOperators):

        self.params = params
        self.grid = grid
        self.elevation = elevation
        self.sh = sh
        self.so = so

    def __repr__(self):
        return (f"Planet(M={self.params.mass/5.972e24:.2f}M⊕, "
                f"R={self.params.equatorial_radius/1e3:.0f}km, "
                f"T={self.params.sidereal_day/3600:.1f}h, "
                f"f={self.params.oblateness:.5f}, "
                f"grid=GridGeometry(n_points={self.grid.n_points})")

    
    @classmethod
    def generate(cls,
                 params: PlanetaryParameters,
                 grid_resolution: int = 3,
                 terrain_params: Optional[SpectralTerrainParams] = None,
                 tectonic_params: Optional[TectonicParams] = None,
                 l_max: int = 15,
                 product_quadrature: str = "fine",
                 grid_type: str = "geodesic",
                 nlat: int = 128,
                 nlon: int = 256) -> 'Planet':
        """
        product_quadrature : {"fine", "coarse"}
            Passed to SpectralOperators. "fine" (default) evaluates nonlinear
            products on a backend-chosen overresolved product sampling:
            geodesic uses a resolution-(grid_resolution+1) co-grid
            ("overresolved product quadrature", docs/KNOWN_RISKS.md R-3), lat-lon
            a 3/2-rule Gauss-Legendre grid (exact product quadrature).
            "coarse" retains the historical state-grid product path for
            A/B comparisons.
        grid_type : {"geodesic", "latlon"}
            Grid family. "geodesic" (default) uses `grid_resolution`;
            "latlon" uses `nlat`/`nlon` (Gauss-Legendre latitudes, uniform
            longitudes). Unknown values raise (no silent fallback).
        """
        if terrain_params is None:
            terrain_params = SpectralTerrainParams(
                rms_elevation=params.radius * 0.001,  # 0.1% of radius
            )
        if tectonic_params is None:
            tectonic_params = TectonicParams()

        # Grid geometry, SH transform, and their pairing (the backend, which
        # owns product-quadrature policy) are three separate objects.
        from tropoi.spatial.spherical_backend import GeodesicBackend, LatLonBackend
        from tropoi.spatial.grids.latlon_grid import (
            GaussLatLonGridGeometry, GaussLatLonSphericalHarmonics)

        if grid_type == "geodesic":
            grid = GeodesicGridGeometry(grid_resolution, params.radius)
            sh = GeodesicSphericalHarmonics(grid, l_max)
            backend = GeodesicBackend(grid, sh)
        elif grid_type == "latlon":
            grid = GaussLatLonGridGeometry(nlat, nlon, params.radius)
            sh = GaussLatLonSphericalHarmonics(grid, l_max)
            backend = LatLonBackend(grid, sh)
        else:
            raise ValueError(
                f"grid_type must be 'geodesic' or 'latlon', got {grid_type!r} "
                "(no silent fallback)")
        so = SpectralOperators(sh, params.radius, grid,
                               product_quadrature=product_quadrature,
                               backend=backend)

        height_coeffs = generate_spectral_terrain_gpu(
            sph=sh,
            params=terrain_params
        )

        tect_params = TectonicParams(
            dt=2.0,
            kappa_height=1e-5,
            kappa_strain=5e-6,
            noise_strength=5.0,
            l_cut_noise=8,
            gamma_activity=4.0,
        )

        # Final terrain back to CPU grid:
        height = sh.inv_transform(height_coeffs)

        # Calculate radial distances (includes oblateness)
        radial_distance = cls._calculate_radial_distances(
            params, grid, height
        )

        # Create elevation data structure
        elevation = ElevationData(
            surface_height=height,
            radial_distance=radial_distance,
            sh_coeffs=height_coeffs,
            power_spectrum = cls._compute_power_spectrum(
                height_coeffs,
                sh.l_max
            )
        )

        # Calculate radial distances
        radial_distance = cls._calculate_radial_distances(params, grid, height)

        pln = cls(params, grid, elevation, sh, so)

        return pln
    

    @staticmethod
    def _calculate_radial_distances(params: PlanetaryParameters,
                                    grid: GridGeometryBase,
                                    surface_height: np.ndarray) -> np.ndarray:
        """
        Calculate distance from planet center including oblateness and terrain.
        """
        R_eq = params.equatorial_radius
        f = params.oblateness
        cos_lat = np.cos(grid.point_latitudes)
        sin_lat = np.sin(grid.point_latitudes)

        reference_radius = R_eq * np.sqrt(
            cos_lat**2 + (1 - f)**2 * sin_lat**2
            )

        # Total radial distance
        radial_distance = reference_radius + surface_height

        return radial_distance

    @staticmethod
    def _compute_power_spectrum(coeffs: cp.ndarray, l_max: int) -> np.ndarray:
        """Compute power spectrum from spherical harmonic coefficients
        stored for m >= 0 only."""
        if coeffs is None:
            return np.array([])

        n_l, n_m = coeffs.shape
        l_max = min(l_max, n_l - 1)

        # m = 0 term
        power = cp.real(coeffs[:, 0] * cp.conj(coeffs[:, 0]))

        # m > 0 terms: 2 * sum |a_l^m|^2
        if n_m > 1:
            pos_m = coeffs[:, 1:]
            power += 2.0 * cp.real((pos_m * cp.conj(pos_m)).sum(axis=1))

        # Bring back to CPU and truncate to requested l_max
        return cp.asnumpy(power[:l_max + 1])


    def reconstruct_surface(self,
                            l_max_filt: Optional[int] = None,
                            m_max_filt: Optional[int] = None
                            ) -> np.ndarray:
        """
        Reconstruct function from spherical harmonic coefficients.
        """
        if self.sh is None or self.elevation.sh_coeffs is None:
            raise ValueError("Spherical harmonics not computed")

        coeffs = self.elevation.sh_coeffs

        coeffs = coeffs.copy()
        if l_max_filt is not None:
            # Keep only degrees 0..l_max_filt
            l_max_filt = min(l_max_filt, coeffs.shape[0] - 1)
            coeffs[l_max_filt + 1:, :] = 0.0

        if m_max_filt is not None:
            # Keep only orders 0...m_max_filt
            m_max_filt = min(m_max_filt, coeffs.shape[1] - 1)
            coeffs[:, m_max_filt + 1:] = 0.0

        reconstructed = self.sh.inv_transform(coeffs)

        return reconstructed
