"""Recipe for the README figure: Williamson test case 5 at T63.

Run from the repository root (needs CUDA; about 10 s on an MX110)::

    python docs/figures/williamson5_t63_overview.py

It draws the committed canonical run capsule through the public plotting
interface (``Simulation.plot``) and writes

* ``docs/assets/williamson5_t63_overview.png``: the figure, whose PNG
  metadata records the run id, the SHA-256 of the coefficients and the
  per-step diagnostics it read, and this recipe;
* ``docs/assets/williamson5_t63_overview.json``: every number the figure
  shows (drifts, spectral-complexity measures, per-map ranges and speeds).

Every field of every view object is written out below, never left to a
default, so later changes to the defaults of ``Simulation.plot`` or
``tropoi plot`` cannot change this figure silently.
``tests/test_readme_figure_recipe.py`` enforces that, and checks that the
committed PNG was produced by exactly this recipe.
"""
from __future__ import annotations

import pathlib

from tropoi.representation.archive import open_simulation
from tropoi.representation.visual.views import (
    Complexity, Contours, Drift, Map, Overview, Streamlines, Style)

ROOT = pathlib.Path(__file__).resolve().parents[2]
CAPSULE = (ROOT / "docs" / "validation" / "williamson_5" / "capsules" /
           "t63" / "20260730T011700Z_williamson5_rot23p93h_r4_l63_dt120h_"
           "45406d82_668e6c9a")
OUTPUT = ROOT / "docs" / "assets" / "williamson5_t63_overview.png"

RECIPE = Overview(
    map=Map(
        background="free_surface_height",
        level=None,
        contours=(Contours(quantity="terrain",
                           levels=(500.0, 1000.0, 1500.0),
                           color="#3b3b3b", line_width=0.6,
                           line_style="solid"),),
        vectors=Streamlines(vector="wind", color_by=None, color="black",
                            width_by="speed", line_width_range=(0.25, 1.6),
                            density=1.0, arrow_size=0.7, max_length=0.35,
                            seed_count=220),
        color_policy="cividis:0.35:1.0",
        symmetric=False,
        limits=None),
    snapshots=(0, 1, 2, 3),
    max_maps=4,
    static=Map(background="terrain", level=None, contours=(), vectors=None,
               color_policy="YlOrBr:0.0:0.85", symmetric=False,
               limits=None),
    diagnostics=(
        Drift(recorded=("total_mass", "total_energy"),
              at_snapshots=("potential_enstrophy",),
              linear_threshold=1e-9),
        Complexity(measures=("mean_degree", "mean_order", "effective_modes"),
                   level=None),
    ),
    title=("Williamson test case 5: flow over an isolated mountain · "
           "shallow-water core"),
    style=Style(width_inches=7.2, base_font_size=8.0, dpi=250,
                map_columns=2, map_row_height_inches=1.85,
                static_height_inches=1.6, diagnostics_height_inches=2.45),
)


def main() -> None:
    simulation = open_simulation(CAPSULE)
    written = simulation.plot(OUTPUT, RECIPE, sidecar=True)
    print(f"Wrote {written}")
    print(f"Wrote {written.with_suffix('.json')}")


if __name__ == "__main__":
    main()
