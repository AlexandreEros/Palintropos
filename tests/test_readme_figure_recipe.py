"""The README figure is pinned to an explicit recipe and to its inputs.

CPU only. The recipe must spell out every field of every view object, so no
change of a plotting default can alter the figure silently; the committed
PNG must embed exactly this recipe and the hashes of the committed capsule
it read; and the committed sidecar's numbers must agree with values
recomputed here from the capsule and from the run's independent grid package.
"""
from __future__ import annotations

import ast
import dataclasses
import hashlib
import importlib.util
import json
import pathlib

import numpy as np
import pytest

from tropoi.representation.diagnostics.spectral import (
    kinetic_energy_modes, spectral_complexity)
from tropoi.representation.visual import views
from tropoi.representation.visual.views import describe

ROOT = pathlib.Path(__file__).resolve().parents[1]
RECIPE_PATH = ROOT / "docs" / "figures" / "williamson5_t63_overview.py"
FIGURE = ROOT / "docs" / "assets" / "williamson5_t63_overview.png"
SIDECAR = FIGURE.with_suffix(".json")
W5 = ROOT / "docs" / "validation" / "williamson_5"


def _recipe_module():
    spec = importlib.util.spec_from_file_location("w5_recipe", RECIPE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_recipe_spells_out_every_field_of_every_view_object():
    tree = ast.parse(RECIPE_PATH.read_text(encoding="utf-8"))
    checked = 0
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and hasattr(views, node.func.id)):
            continue
        cls = getattr(views, node.func.id)
        if not dataclasses.is_dataclass(cls):
            continue
        given = {keyword.arg for keyword in node.keywords}
        required = {field.name for field in dataclasses.fields(cls)}
        assert not node.args, f"{cls.__name__}: use keywords only"
        assert given == required, (
            f"{cls.__name__} leaves {sorted(required - given)} to defaults")
        checked += 1
    assert checked >= 8


def test_committed_figure_was_made_by_this_recipe_from_the_committed_capsule():
    from PIL import Image
    recipe = _recipe_module()
    with Image.open(FIGURE) as image:
        info = dict(image.info)
        assert image.size[0] == 1800
    assert json.loads(info["View"]) == describe(recipe.RECIPE)
    sums = {}
    for line in (W5 / "capsules" / "SHA256SUMS").read_text().splitlines():
        if line and not line.startswith("#"):
            digest, name = line.split(maxsplit=1)
            sums[pathlib.PurePosixPath(name).name] = digest
    assert info["CoefficientsSHA256"] == sums["swe_coeffs.npy"]
    assert info["DiagnosticsSHA256"] == sums["timeseries.csv"]
    assert info["RunId"] == recipe.CAPSULE.name


def test_committed_sidecar_agrees_with_numbers_recomputed_here():
    record = json.loads(SIDECAR.read_text(encoding="utf-8"))
    capsule = _recipe_module().CAPSULE
    coefficients = np.load(capsule / "swe_coeffs.npy")
    times = np.load(capsule / "swe_snapshot_times.npy")
    for index in range(4):
        expected = spectral_complexity(kinetic_energy_modes(
            coefficients[index, 0], coefficients[index, 1], radius=1.0))
        complexity = record["spectral_complexity"]
        assert complexity["mean_degree"][index] == pytest.approx(
            expected.mean_degree, abs=1e-12)
        assert complexity["effective_modes"][index] == pytest.approx(
            expected.effective_modes, abs=1e-12)
    table = np.genfromtxt(capsule / "diagnostics" / "timeseries.csv",
                          delimiter=",", names=True)
    rows = [np.flatnonzero(table["time_s"] == t)[0] for t in times]
    energy = table["total_energy"][rows]
    np.testing.assert_allclose(
        record["drift"]["total_energy"]["relative_drift_at_saved_states"],
        energy / energy[0] - 1.0, rtol=0, atol=1e-15)
    assert record["drift"]["total_mass"][
        "relative_drift_at_saved_states"] == [0.0, 0.0, 0.0, 0.0]
    with np.load(W5 / "aeolus_w5_t63.npz") as package:
        independent = package["potential_enstrophy"]
    np.testing.assert_allclose(record["drift"]["potential_enstrophy"]["values"],
                               independent, rtol=1e-13)
