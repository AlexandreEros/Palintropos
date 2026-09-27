"""Who owns which committed artifact: the layout rules, checked for every file.

CPU only. The rules (docs/runs/README.md, docs/assets/README.md):

* ``docs/runs/<run-id>/`` is a published run: its primary files are
  evidence, covered by a ``SHA256SUMS`` receipt that lists every primary file
  and nothing under ``assets/``; line endings are never converted;
* ``<run>/assets/`` holds derived products of that run only, and every
  derived figure records the SHA-256 of the run files it read, which must
  match the receipt;
* ``docs/assets/`` is for material that belongs to no single run; every file
  there must be listed in its README inventory, so nothing lands there
  unclassified.

``tests/test_w5_canonical_capsule.py`` additionally pins the exact file set
and the saved-run API reading of the canonical Williamson-5 run.
"""
from __future__ import annotations

import hashlib
import pathlib
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
PUBLISHED = ROOT / "docs" / "runs"
REFERENCE = ROOT / "docs" / "assets"
RUNS = sorted(path for path in PUBLISHED.iterdir() if path.is_dir())


def _receipt(run: pathlib.Path) -> dict[str, str]:
    sums = {}
    for line in (run / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        sums[name.strip()] = digest
    return sums


def _files(run: pathlib.Path) -> set[str]:
    return {path.relative_to(run).as_posix() for path in run.rglob("*")
            if path.is_file()}


def test_there_is_at_least_one_published_run():
    assert RUNS


@pytest.mark.parametrize("run", RUNS, ids=lambda path: path.name[:40])
def test_a_published_runs_receipt_covers_its_evidence_and_nothing_derived(
        run):
    sums = _receipt(run)
    assert not any(name.startswith("assets/") for name in sums)
    primary = {name for name in _files(run)
               if not name.startswith("assets/")} - {"SHA256SUMS"}
    assert set(sums) == primary
    for name, digest in sums.items():
        assert hashlib.sha256((run / name).read_bytes()).hexdigest() == (
            digest), name


@pytest.mark.parametrize("run", RUNS, ids=lambda path: path.name[:40])
def test_a_published_runs_figures_name_the_evidence_they_read(run):
    from PIL import Image
    sums = _receipt(run)
    coefficients = [name for name in sums if name.endswith("_coeffs.npy")]
    assert len(coefficients) == 1
    for figure in sorted((run / "assets").glob("*.png")):
        with Image.open(figure) as image:
            info = dict(image.info)
        assert info.get("CoefficientsSHA256") == sums[coefficients[0]], (
            figure.name)
        if "diagnostics/timeseries.csv" in sums:
            assert info.get("DiagnosticsSHA256") == sums[
                "diagnostics/timeseries.csv"], figure.name


def test_every_published_run_is_indexed():
    index = (PUBLISHED / "README.md").read_text(encoding="utf-8")
    for run in RUNS:
        assert f"`{run.name}`" in index, run.name


def test_published_runs_keep_their_bytes_through_git():
    result = subprocess.run(
        ["git", "check-attr", "text", "--",
         *(f"docs/runs/{run.name}/manifest.json" for run in RUNS)],
        cwd=ROOT, capture_output=True, text=True, check=True)
    for line in result.stdout.splitlines():
        assert line.endswith(": text: unset"), line


def test_every_reference_asset_is_inventoried():
    inventory = (REFERENCE / "README.md").read_text(encoding="utf-8")
    for path in sorted(REFERENCE.rglob("*")):
        if path.is_file() and path.name != "README.md":
            name = path.relative_to(REFERENCE).as_posix()
            assert f"`{name}`" in inventory, (
                f"docs/assets/{name} is not in docs/assets/README.md: a "
                "figure of one run belongs in that run's assets/, one that "
                "combines runs beside its study in docs/validation/")
