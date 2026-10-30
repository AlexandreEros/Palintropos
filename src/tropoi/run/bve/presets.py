"""Named BVE configuration bundles shared by Python and CLI callers."""

PRESETS = {
    "rh4": {
        "description": "One-day Rossby-Haurwitz wavenumber-4 validation run "
                       "(the docs/VALIDATION.md configuration).",
        "settings": {
            "scenario": "rh4",
            "lmax": 21,
            "resolution": 4,
            "day_hours": 24.0,
            "duration_days": 1.0,
            "dt_snapshots": 21600.0,
            "product_quadrature": "fine",
            "viscosity": 0.0,
            "experiment": "validation-rh4",
        },
    },
    "two-vortices-quick": {
        "description": "Small, fast two-vortex smoke run "
                       "(the README quickstart configuration).",
        "settings": {
            "scenario": "two_vortices",
            "lmax": 8,
            "resolution": 3,
            "nlat": 12,
            "nlon": 24,
            "duration_days": 0.02,
            "dt_snapshots": 864.0,
            "experiment": "quickstart",
        },
    },
}

