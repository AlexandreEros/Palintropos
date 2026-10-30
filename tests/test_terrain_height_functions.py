"""Terrain from height functions, expressions and user files.

CPU: ``ElevationRaster.from_function`` / ``from_expression`` (contract,
orientation, units, broadcasting, constants, invalid results and sampling
settings), the allowlisted expression interpreter (accepted grammar and
clear rejections), ``load_height_function`` (files outside the repository,
relative/absolute/spaced/drive-letter paths, final-colon split, failures),
agreement of equivalent expression, callable, file and array inputs,
``encode_grey16`` range checks, and ``tropoi terrain prepare`` argument
errors and CUDA-free sampling (in a fresh interpreter with CuPy blocked).

GPU: the CLI prepares PE and SWE terrain from an expression and from a
file; the saved terrain keeps its provenance, reloads, previews and feeds a
``tropoi run pe --terrain`` run and a shallow-water state after the
function file is deleted; equivalent inputs give identical coefficients.
"""
from __future__ import annotations

import hashlib
import json
import math
import pathlib
import subprocess
import sys

import numpy as np
import pytest

from tropoi.run.custom import (ElevationImportError, ElevationRaster,
                               HeightExpression, HeightExpressionError,
                               HeightFunctionError, encode_grey16,
                               load_height_function)

EXPR = ("1200*cos(radians(lat))**2*cos(2*radians(lon)) "
        "+ 600*sin(radians(lat))")


def hills(lat_deg, lon_deg):
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    return 1200 * np.cos(lat) ** 2 * np.cos(2 * lon) + 600 * np.sin(lat)


HILLS_FILE = '''\
import numpy as np

def height(lat_deg, lon_deg):
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    return 1200 * np.cos(lat)**2 * np.cos(2 * lon) + 600 * np.sin(lat)

class Shapes:
    @staticmethod
    def plateau(lat_deg, lon_deg):
        return 1000.0

NOT_CALLABLE = 3.0
'''


def _has_cuda():
    try:
        import cupy as cp
        return cp.is_available()
    except Exception:
        return False


requires_cuda = pytest.mark.skipif(not _has_cuda(),
                                   reason="needs a CUDA GPU")


@pytest.fixture
def terrain_file(tmp_path):
    """A user's height function outside the repository, path with spaces."""
    folder = tmp_path / "my terrain dir"
    folder.mkdir()
    path = folder / "my terrain.py"
    path.write_text(HILLS_FILE, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# The callable contract (CPU)
# ---------------------------------------------------------------------------

def test_samples_the_function_at_cell_centres_in_degrees_south_to_north():
    seen = {}

    def height(lat_deg, lon_deg):
        seen["lat"], seen["lon"] = lat_deg.copy(), lon_deg.copy()
        return 1000.0 * np.sin(np.radians(lat_deg)) + lon_deg

    r = ElevationRaster.from_function(height, resolution_deg=2)
    assert r.shape == (90, 180) and r.registration == "cell"
    assert seen["lat"].shape == seen["lon"].shape == (90, 180)
    np.testing.assert_array_equal(r.lat_deg, np.arange(-89.0, 90.0, 2.0))
    np.testing.assert_array_equal(r.lon_deg, np.arange(-179.0, 180.0, 2.0))
    np.testing.assert_array_equal(seen["lat"][:, 0], r.lat_deg)  # degrees
    np.testing.assert_array_equal(seen["lon"][0], r.lon_deg)
    # metres, north up, east longitude: the value at each node is f(node)
    la, lo = np.meshgrid(r.lat_deg, r.lon_deg, indexing="ij")
    np.testing.assert_array_equal(r.elevation_m,
                                  1000.0 * np.sin(np.radians(la)) + lo)
    assert r.elevation_m[-1].mean() > 900 and r.elevation_m[0].mean() < -900
    j = np.searchsorted(r.lat_deg, 1.0)
    assert r.sample(1.0, 1.0) == pytest.approx(1000 * math.sin(
        math.radians(1.0)) + 1.0)
    assert j == 45
    p = r.provenance
    assert p["format"] == "function" and p["function_name"].endswith(
        "height")
    assert p["units"] == "m" and p["sampling_resolution_deg"] == 2.0
    assert p["sampling_lon_range_deg"] == "[-180, 180)"
    assert "degrees" in p["coordinate_convention"]
    assert len(p["source_sha256"]) == 64


def test_resolution_is_the_sampling_spacing_only():
    for d, shape in ((4, (45, 90)), (1, (180, 360)), (0.5, (360, 720))):
        assert ElevationRaster.from_function(
            hills, resolution_deg=d).shape == shape


def test_constant_heights_in_every_form():
    rasters = [ElevationRaster.from_function(1000, resolution_deg=3),
               ElevationRaster.from_function(1000.0, resolution_deg=3),
               ElevationRaster.from_function(lambda la, lo: 1000,
                                             resolution_deg=3),
               ElevationRaster.from_function(lambda la, lo: np.float32(1000),
                                             resolution_deg=3),
               ElevationRaster.from_expression("1000", resolution_deg=3)]
    for r in rasters:
        assert r.shape == (60, 120)
        assert np.all(r.elevation_m == 1000.0)
    assert rasters[0].provenance["sampling_mode"] == "constant"
    assert rasters[-1].provenance["expression"] == "1000"


def test_results_broadcast_to_the_grid():
    column = ElevationRaster.from_function(
        lambda la, lo: 100 * np.cos(np.radians(la[:, :1])), resolution_deg=3)
    row = ElevationRaster.from_function(
        lambda la, lo: 100 * np.cos(np.radians(lo[0])), resolution_deg=3)
    full = ElevationRaster.from_function(
        lambda la, lo: 100 * np.cos(np.radians(la)), resolution_deg=3)
    np.testing.assert_array_equal(column.elevation_m, full.elevation_m)
    assert np.ptp(row.elevation_m[:, 7]) == 0          # constant in lat
    integer = ElevationRaster.from_function(
        lambda la, lo: np.full(la.shape, 7, dtype=np.int16), resolution_deg=3)
    assert integer.elevation_m.dtype == np.float64
    assert np.all(integer.elevation_m == 7)


def test_lon_start_moves_the_longitudes_the_function_sees():
    seen = {}

    def height(la, lo):
        seen["lon"] = lo
        return 500 * np.cos(2 * np.radians(lo)) * np.cos(np.radians(la))

    east = ElevationRaster.from_function(height, resolution_deg=2,
                                         lon_start_deg=0)
    assert seen["lon"].min() == 1.0 and seen["lon"].max() == 359.0
    assert east.provenance["sampling_lon_range_deg"] == "[0, 360)"
    signed = ElevationRaster.from_function(height, resolution_deg=2)
    np.testing.assert_array_equal(east.lon_deg, signed.lon_deg)  # normalized
    np.testing.assert_allclose(east.elevation_m, signed.elevation_m,
                               rtol=0, atol=1e-9)


def test_pointwise_calls_scalar_functions():
    def scalar(lat, lon):
        return 100.0 * math.cos(math.radians(lat))

    with pytest.raises(HeightFunctionError, match="pointwise=True"):
        ElevationRaster.from_function(scalar, resolution_deg=10)
    r = ElevationRaster.from_function(scalar, resolution_deg=10,
                                      pointwise=True)
    np.testing.assert_allclose(
        r.elevation_m, ElevationRaster.from_function(
            lambda la, lo: 100 * np.cos(np.radians(la)),
            resolution_deg=10).elevation_m, rtol=0, atol=1e-12)
    assert r.provenance["sampling_mode"] == "pointwise"


@pytest.mark.parametrize("result, match", [
    (lambda la, lo: la + 1j, "complex"),
    (lambda la, lo: la > 0, "booleans"),
    (lambda la, lo: np.where(la > 80, np.nan, 0.0), "non-finite"),
    (lambda la, lo: np.where(la > 80, np.inf, 0.0), "non-finite"),
    (lambda la, lo: np.full(la.shape, "x"), "real numbers"),
    (lambda la, lo: [[1, 2], [3]], "not a numeric array|real numbers"),
    (lambda la, lo: np.ones((3, 3)), "does not broadcast"),
    (lambda la, lo: la[:, 0], "does not broadcast"),
    (lambda la, lo: np.ones((2,) + la.shape), "does not broadcast"),
    (lambda la, lo: np.ma.masked_less(la, 0), "masked"),
    (lambda la, lo: None, "real numbers"),
])
def test_invalid_results_are_refused_with_the_reason(result, match):
    with pytest.raises(HeightFunctionError, match=match):
        ElevationRaster.from_function(result, resolution_deg=10)


def test_function_failures_name_the_function():
    def broken(lat, lon):
        raise RuntimeError("no terrain today")

    with pytest.raises(HeightFunctionError,
                       match="'.*broken' failed: RuntimeError: no terrain"
                       ) as info:
        ElevationRaster.from_function(broken, resolution_deg=10)
    assert isinstance(info.value.__cause__, RuntimeError)
    assert isinstance(info.value, ElevationImportError)    # one error family
    with pytest.raises(HeightFunctionError, match="callable"):
        ElevationRaster.from_function("1000", resolution_deg=10)
    with pytest.raises(HeightFunctionError, match="boolean"):
        ElevationRaster.from_function(True, resolution_deg=10)


@pytest.mark.parametrize("resolution, match", [
    (0, "positive"), (-2, "positive"), (float("nan"), "positive"),
    (float("inf"), "positive"), (7, "does not divide 180"),
    (0.7, "does not divide 180"), (120, "coarser than 90"),
    ("two", "must be a number"), (None, "must be a number"),
    (0.01, "above the .* limit"),
])
def test_invalid_sampling_settings(resolution, match):
    with pytest.raises(HeightFunctionError, match=match):
        ElevationRaster.from_function(hills, resolution_deg=resolution)


def test_invalid_longitude_origin():
    with pytest.raises(HeightFunctionError, match="lon_start_deg"):
        ElevationRaster.from_function(hills, resolution_deg=2,
                                      lon_start_deg=float("nan"))


def test_resolution_is_required():
    with pytest.raises(TypeError):
        ElevationRaster.from_function(hills)        # no silent default


# ---------------------------------------------------------------------------
# Expressions (CPU)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text, expected", [
    ("1000", lambda la, lo: np.full(la.shape, 1000.0)),
    (EXPR, hills),
    ("-500*sin(radians(lat))", lambda la, lo: -500 * np.sin(np.radians(la))),
    ("where(lat > 0, 800, -400)", lambda la, lo: np.where(la > 0, 800., -400.)),
    ("where((lat > 0) & (lon < 0) | (lat < -60), 1, 0)",
     lambda la, lo: np.where((la > 0) & (lo < 0) | (la < -60), 1., 0.)),
    ("where(-30 < lat <= 30, 100, 0)",
     lambda la, lo: np.where((-30 < la) & (la <= 30), 100., 0.)),
    ("where(~(lat > 0), 5, 0)", lambda la, lo: np.where(la <= 0, 5., 0.)),
    ("maximum(minimum(lat*10, 300), -200) + abs(lon) % 7",
     lambda la, lo: np.maximum(np.minimum(la * 10, 300), -200)
     + np.abs(lo) % 7),
    ("2000*exp(-(hypot(lat - 20, lon + 40)/15)**2)",
     lambda la, lo: 2000 * np.exp(-(np.hypot(la - 20, lo + 40) / 15) ** 2)),
    ("clip(sqrt(abs(lat))*100, 0, 500) * sign(lon) + floor(lat/10)*ceil(1.5)",
     lambda la, lo: np.clip(np.sqrt(np.abs(la)) * 100, 0, 500) * np.sign(lo)
     + np.floor(la / 10) * 2.0),
    ("pi*degrees(arctan2(lat, lon + 1000))",
     lambda la, lo: np.pi * np.degrees(np.arctan2(la, lo + 1000))),
])
def test_expressions_evaluate_like_numpy(text, expected):
    r = ElevationRaster.from_expression(text, resolution_deg=3)
    la, lo = np.meshgrid(r.lat_deg, r.lon_deg, indexing="ij")
    np.testing.assert_allclose(r.elevation_m, expected(la, lo), rtol=1e-14,
                               atol=1e-9)
    assert r.provenance["format"] == "expression"
    assert r.provenance["expression"] == text
    assert r.provenance["expression_sha256"] == hashlib.sha256(
        text.encode()).hexdigest()
    assert "function_name" not in r.provenance


@pytest.mark.parametrize("text, match", [
    ("__import__('os')", "unknown function '__import__'"),
    ("open('x')", "unknown function 'open'"),
    ("np.sin(lat)", "attribute access"),
    ("lat.real", "attribute access"),
    ("().__class__", "attribute access"),
    ("lat[0]", "subscripts"),
    ("[x for x in lat]", "comprehensions"),
    ("(x for x in lat)", "comprehensions"),
    ("lambda: 1", "lambda"),
    ("lat if lat else lon", "where"),
    ("lat > 0 and lon > 0", "'and' / 'or'"),
    ("not lat", "'not'"),
    ("x + 1", "unknown name 'x'"),
    ("height", "unknown name 'height'"),
    ("sin", "is a function; call it"),
    ("foo(lat)", "unknown function 'foo'"),
    ("sin(lat, lon)", "takes 1 argument, got 2"),
    ("where(lat > 0, 1)", "takes 3 arguments, got 2"),
    ("sin(x=lat)", "keyword arguments"),
    ("sin(*lat)", r"\*arguments"),
    ("'abc'", "str literal"),
    ("1j * lat", "complex number"),
    ("True", "bool literal"),
    ("lat // 2", "'//'"),
    ("lat @ lon", "'@'"),
    ("lat ^ lon", "'\\^'"),
    ("lat in lon", "'in'"),
    ("(lat, lon)", "tuples"),
    ("(y := 3)", "assignment"),
    ("f'{lat}'", "f-strings"),
    ("import os", "invalid height expression"),
    ("x = 1", "invalid height expression"),
    ("1000; 2", "invalid height expression"),
    ("", "empty"),
    ("   ", "empty"),
    ("1" * 10_001, "characters long"),
])
def test_unsupported_syntax_and_names_are_rejected_before_evaluation(
        text, match):
    with pytest.raises(HeightExpressionError, match=match):
        HeightExpression(text)


def test_expressions_never_reach_python_eval(monkeypatch):
    import builtins
    # (compile stays: ast.parse uses it to build a syntax tree only, never
    # a code object; the tree is then walked by the interpreter)
    for name in ("eval", "exec"):
        monkeypatch.setattr(builtins, name, _forbidden)
    r = ElevationRaster.from_expression(EXPR, resolution_deg=6)
    assert np.isfinite(r.elevation_m).all()


def _forbidden(*args, **kwargs):  # pragma: no cover - must never run
    raise AssertionError("expression evaluation used eval/exec")


def test_huge_powers_overflow_to_a_clear_error_quickly():
    # literals are floats, so this cannot become an endless integer power
    with pytest.raises(HeightFunctionError, match="non-finite"):
        ElevationRaster.from_expression("10**10**10", resolution_deg=10)
    with pytest.raises(HeightFunctionError, match="non-finite"):
        ElevationRaster.from_expression("log(lat - lat)", resolution_deg=10)


def test_expression_is_a_height_function_too():
    expr = HeightExpression("100*cos(radians(lat))")
    assert expr(0.0, 10.0) == pytest.approx(100.0)
    with pytest.raises(HeightExpressionError):
        HeightExpression(3)


# ---------------------------------------------------------------------------
# User files (CPU)
# ---------------------------------------------------------------------------

def test_loads_a_function_from_outside_the_repository(terrain_file):
    function, record = load_height_function(f"{terrain_file}:height")
    assert record == {
        "format": "function", "function_file": "my terrain.py",
        "function_file_sha256": hashlib.sha256(
            terrain_file.read_bytes()).hexdigest(),
        "function_name": "height"}
    assert str(terrain_file.parent) not in json.dumps(record)  # no abs path
    assert function(0.0, 0.0) == pytest.approx(1200.0)


def test_paths_split_at_the_final_colon_and_resolve_from_cwd(
        terrain_file, monkeypatch):
    absolute = str(terrain_file)
    if sys.platform == "win32":
        assert absolute[1] == ":"               # a drive-letter path
    load_height_function(absolute + ":height")
    load_height_function(absolute.replace("\\", "/") + ":height")
    load_height_function(f"  {absolute}:height  ")
    monkeypatch.chdir(terrain_file.parent.parent)
    _, record = load_height_function("my terrain dir/my terrain.py:height")
    assert record["function_file"] == "my terrain.py"
    monkeypatch.chdir(terrain_file.parent)
    load_height_function("./my terrain.py:height")
    load_height_function("my terrain.py:height")
    # cwd= overrides the process working directory
    load_height_function("my terrain.py:height", cwd=terrain_file.parent)
    plateau, record = load_height_function("my terrain.py:Shapes.plateau")
    assert plateau(0, 0) == 1000.0 and record["function_name"] == \
        "Shapes.plateau"


@pytest.mark.parametrize("spec, match", [
    ("my terrain.py", "expects FILE.py:NAME"),
    ("my terrain.py:", "expects FILE.py:NAME"),
    (":height", "expects FILE.py:NAME"),
    ("my terrain.py:1height", "not a Python name"),
    ("my terrain.py:height()", "not a Python name"),
    ("missing.py:height", "file not found"),
    (".:height", "not a file"),
    ("my terrain.py:heigth", "defines no 'heigth'; callables defined "
                             "there: Shapes, height"),
    ("my terrain.py:NOT_CALLABLE", "is float, not a callable"),
])
def test_bad_function_specifications(spec, match, terrain_file,
                                     monkeypatch):
    monkeypatch.chdir(terrain_file.parent)
    with pytest.raises(HeightFunctionError, match=match):
        load_height_function(spec)


def test_an_edited_file_runs_its_new_source_not_cached_bytecode(
        tmp_path, monkeypatch):
    # Regression: an import loader may reuse __pycache__ bytecode whose
    # recorded mtime (whole seconds) and size match, executing the OLD code
    # while the provenance hashes the NEW source. Same size, same mtime,
    # with bytecode writing on (as in a default interpreter; CI or a shell
    # may set PYTHONDONTWRITEBYTECODE, which would hide the bug):
    import os
    monkeypatch.setattr(sys, "dont_write_bytecode", False)
    monkeypatch.setattr(sys, "pycache_prefix", None)
    path = tmp_path / "edited.py"
    path.write_text("def height(lat, lon):\n    return 100\n",
                    encoding="utf-8")
    stat = path.stat()
    first, record = load_height_function(f"{path}:height")
    assert first(0, 0) == 100
    path.write_text("def height(lat, lon):\n    return 900\n",
                    encoding="utf-8")
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert path.stat().st_size == stat.st_size
    again, new_record = load_height_function(f"{path}:height")
    assert again(0, 0) == 900
    assert new_record["function_file_sha256"] == hashlib.sha256(
        path.read_bytes()).hexdigest() != record["function_file_sha256"]
    raster = ElevationRaster.from_function(again, resolution_deg=10,
                                           provenance=new_record)
    assert np.all(raster.elevation_m == 900)
    # nothing is cached next to the user's file
    assert not (tmp_path / "__pycache__").exists()


def test_a_loaded_file_behaves_like_a_module(tmp_path):
    path = tmp_path / "mod.py"
    path.write_text(
        "import dataclasses, numpy as np\n"
        "@dataclasses.dataclass\nclass Hill:\n    h: float = 300.0\n"
        "    def __call__(self, lat, lon):\n"
        "        return self.h * np.cos(np.radians(lat))\n"
        "hill = Hill()\n"
        "assert __file__.endswith('mod.py'), __file__\n"
        "assert __name__ != '__main__'\n", encoding="utf-8")
    function, _ = load_height_function(f"{path}:hill")
    assert function(0.0, 0.0) == pytest.approx(300.0)


def test_errors_while_executing_or_calling_a_file_are_reported(tmp_path):
    bad = tmp_path / "bad.py"
    bad.write_text("import numpy as np\nraise ValueError('broken import')\n",
                   encoding="utf-8")
    with pytest.raises(HeightFunctionError,
                       match="error while executing bad.py: ValueError: "
                             "broken import"):
        load_height_function(f"{bad}:height")
    fails = tmp_path / "fails.py"
    fails.write_text("def height(lat, lon):\n    return 1 / 0\n",
                     encoding="utf-8")
    function, record = load_height_function(f"{fails}:height")
    with pytest.raises(HeightFunctionError,
                       match="'height' failed: ZeroDivisionError"):
        ElevationRaster.from_function(function, resolution_deg=10,
                                      name=record["function_name"])


def test_equivalent_inputs_give_the_same_raster(terrain_file):
    function, record = load_height_function(f"{terrain_file}:height")
    from_file = ElevationRaster.from_function(
        function, resolution_deg=2, name="height", provenance=record)
    from_callable = ElevationRaster.from_function(hills, resolution_deg=2)
    from_expression = ElevationRaster.from_expression(EXPR, resolution_deg=2)
    lat = np.arange(-89.0, 90.0, 2.0)
    lon = np.arange(-179.0, 180.0, 2.0)
    from_array = ElevationRaster.from_array(
        hills(*np.meshgrid(lat, lon, indexing="ij")), lat, lon)
    for other in (from_callable, from_expression, from_array):
        np.testing.assert_array_equal(other.lat_deg, from_file.lat_deg)
        np.testing.assert_array_equal(other.lon_deg, from_file.lon_deg)
        np.testing.assert_allclose(other.elevation_m, from_file.elevation_m,
                                   rtol=0, atol=1e-9)
    # the sampled-values hash identifies the same samples across routes
    assert from_callable.provenance["source_sha256"] == \
        from_array.provenance["source_sha256"]
    assert from_file.provenance["function_file"] == "my terrain.py"


# ---------------------------------------------------------------------------
# Grey heightmap encoding (CPU)
# ---------------------------------------------------------------------------

def test_grey16_encoding_round_trips_and_refuses_out_of_range(tmp_path):
    from PIL import Image

    from tropoi.run.custom import read_elevation

    lat = np.arange(89.0, -90.0, -2.0)
    lon = np.arange(-179.0, 180.0, 2.0)
    h = hills(*np.meshgrid(lat, lon, indexing="ij"))
    grey = encode_grey16(h, (-2000, 2000))
    assert grey.dtype == np.uint16
    Image.fromarray(grey).save(tmp_path / "h.png")
    back = read_elevation(tmp_path / "h.png", height_range=(-2000, 2000),
                          lon_west_deg=-180)
    np.testing.assert_allclose(back.elevation_m, h[::-1], rtol=0,
                               atol=4000 / 65535)
    np.testing.assert_array_equal(encode_grey16([-2000, 2000], (-2000, 2000)),
                                  [0, 65535])
    # 1300 m above the top would wrap to a low grey value with astype
    with pytest.raises(ElevationImportError, match="outside the encoding "
                                                   "range"):
        encode_grey16(h, (-1000, 1000))
    with pytest.raises(ElevationImportError, match="outside"):
        encode_grey16([2000.01], (-2000, 2000))
    with pytest.raises(ElevationImportError, match="NaN/Inf"):
        encode_grey16([np.nan], (-2000, 2000))
    with pytest.raises(ElevationImportError, match="black < white"):
        encode_grey16([0.0], (2000, -2000))
    with pytest.raises(ElevationImportError, match="black_m, white_m"):
        encode_grey16([0.0], 2000)


def test_example_heightmaps_use_the_checked_encoding():
    root = pathlib.Path(__file__).resolve().parents[1]
    for name in ("custom_topographies.py", "mola_mars.py",
                 "terrain_planet.py"):
        text = (root / "examples" / name).read_text(encoding="utf-8")
        assert "astype(np.uint16)" not in text, name
        assert "encode_grey16" in text, name


# ---------------------------------------------------------------------------
# tropoi terrain prepare: parsing, errors and CUDA-free sampling (CPU)
# ---------------------------------------------------------------------------

def _cli(argv, capsys):
    from tropoi.cli.main import main
    code = main(argv)
    out = capsys.readouterr()
    return code, out.out, out.err


@pytest.mark.parametrize("argv", [
    ["terrain", "prepare", "--solver", "pe", "--out", "x"],
    ["terrain", "prepare", "--expression", "1", "--function", "f.py:h",
     "--solver", "pe", "--out", "x"],
    ["terrain", "prepare", "--expression", "1", "--out", "x"],
    ["terrain", "prepare", "--expression", "1", "--solver", "pe"],
    ["terrain", "prepare", "--expression", "1", "--solver", "bve",
     "--out", "x"],
    ["terrain"],
])
def test_cli_argument_errors(argv, capsys):
    from tropoi.cli.main import main
    with pytest.raises(SystemExit) as info:
        main(argv)
    assert info.value.code == 2


def test_cli_sample_only_writes_nothing(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    code, out, err = _cli(
        ["terrain", "prepare", "--expression", EXPR, "--resolution-deg", "2",
         "--lmax", "21", "--solver", "pe", "--out", "terrain/x",
         "--sample-only"], capsys)
    assert code == 0, err
    assert "expression: 90 x 180 (2 deg" in out and "-1274 .. 1274 m" in out
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("argv, match", [
    (["--expression", "np.cos(lat)"], "attribute access"),
    (["--expression", "x"], "unknown name 'x'"),
    (["--expression", "1", "--resolution-deg", "7"], "does not divide 180"),
    (["--expression", "1", "--pointwise"], "--pointwise applies to "
                                           "--function"),
    (["--function", "nowhere.py:height"], "file not found"),
    (["--function", "nowhere.py"], "expects FILE.py:NAME"),
    (["--expression", "1", "--lmax", "0"], "--lmax must be >= 1"),
])
def test_cli_reports_input_errors(argv, match, tmp_path, capsys,
                                  monkeypatch):
    monkeypatch.chdir(tmp_path)
    code, out, err = _cli(["terrain", "prepare", *argv, "--solver", "pe",
                           "--out", "t", "--sample-only"], capsys)
    assert code == 2
    assert err.startswith("tropoi terrain prepare: error:")
    import re
    assert re.search(match, err), err


def test_cli_never_overwrites_silently(tmp_path, capsys):
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "notes.txt").write_text("mine", encoding="utf-8")
    base = ["terrain", "prepare", "--expression", "1000", "--solver", "pe",
            "--out", str(occupied)]
    code, _, err = _cli(base, capsys)
    assert code == 2 and "already exists" in err
    code, _, err = _cli(base + ["--overwrite"], capsys)
    assert code == 2 and "not a prepared terrain" in err
    assert (occupied / "notes.txt").read_text(encoding="utf-8") == "mine"


def test_cli_samples_without_cuda_and_explains_the_cuda_step(
        terrain_file, tmp_path):
    # A fresh interpreter in which importing CuPy fails.
    probe = (
        "import sys\nsys.modules['cupy'] = None\n"
        "from tropoi.cli.main import main\n"
        f"spec = {str(terrain_file) + ':height'!r}\n"
        "rc = main(['terrain', 'prepare', '--function', spec, "
        "'--resolution-deg', '2', '--solver', 'pe', '--out', 'terrain', "
        "'--sample-only'])\n"
        "assert rc == 0, rc\n"
        "rc = main(['terrain', 'prepare', '--expression', '1000*sin(radians("
        "lat))', '--solver', 'pe', '--out', 'terrain'])\n"
        "assert rc == 2, rc\n"
        "banned = [m for m in ('cupy', 'cupyx', 'matplotlib') "
        "if sys.modules.get(m) is not None]\n"
        "assert not banned, banned\n")
    result = subprocess.run([sys.executable, "-c", probe], cwd=tmp_path,
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "my terrain.py" not in result.stderr
    assert "sampled height: 90 x 180" in result.stdout
    assert "spectral preparation needs CUDA" in result.stderr
    assert "--sample-only" in result.stderr
    assert not (tmp_path / "terrain").exists()


# ---------------------------------------------------------------------------
# Preparation, persistence and consumption (GPU)
# ---------------------------------------------------------------------------

LMAX = 10                                   # default grid 18 x 36


@requires_cuda
def test_cli_function_terrain_survives_its_file_and_feeds_a_pe_run(
        terrain_file, tmp_path, capsys, monkeypatch):
    import matplotlib
    matplotlib.use("Agg")
    from tropoi.representation.archive import open_simulation
    from tropoi.run.custom import PreparedPlanet

    monkeypatch.chdir(terrain_file.parent)
    out = tmp_path / "saved terrain"
    code, stdout, err = _cli(
        ["terrain", "prepare", "--function", "./my terrain.py:height",
         "--resolution-deg", "2", "--lmax", str(LMAX), "--solver", "pe",
         "--out", str(out), "--figure", str(tmp_path / "t.png"),
         "--globe", str(tmp_path / "g.html")], capsys)
    assert code == 0, err
    assert "tropoi run pe --terrain" in stdout
    assert (tmp_path / "t.png").stat().st_size > 1000
    assert (tmp_path / "g.html").stat().st_size > 1000
    digest = hashlib.sha256(terrain_file.read_bytes()).hexdigest()
    monkeypatch.chdir(tmp_path)
    terrain_file.unlink()                            # the source is gone
    terrain_file.parent.rmdir()

    world = PreparedPlanet.load(out)
    p = world.topography.parameters
    assert world.topography.preset == "imported" and world.l_cut == 6
    assert p["format"] == "function" and p["function_name"] == "height"
    assert p["function_file"] == "my terrain.py"
    assert p["function_file_sha256"] == digest
    assert p["sampling_resolution_deg"] == 2.0
    assert p["sampling_lon_range_deg"] == "[-180, 180)"
    assert "degrees" in p["coordinate_convention"]
    assert not np.any(world.topography_lm[7:])
    saved = json.loads((out / "terrain.json").read_text(encoding="utf-8"))
    assert "def height" not in json.dumps(saved)       # no source, no pickle
    assert sorted(q.name for q in out.iterdir()) == [
        "terrain.json", "topography_elevation_lm.npy"]
    labels = dict(world.details())
    assert labels["Function file"] == "my terrain.py"
    assert labels["Height function"] == "height"
    figure = world.plot()                              # preview, CPU
    import matplotlib.pyplot as plt
    plt.close(figure)
    assert world.view(tmp_path / "again.html", open_browser=False).is_file()

    # existing output: refused, then replaced only with --overwrite
    again = ["terrain", "prepare", "--expression", EXPR, "--resolution-deg",
             "2", "--lmax", str(LMAX), "--solver", "pe", "--out", str(out)]
    code, _, err = _cli(again, capsys)
    assert code == 2 and "already exists" in err
    assert PreparedPlanet.load(out).topography.parameters[
        "function_file_sha256"] == digest
    code, _, err = _cli(again + ["--overwrite"], capsys)
    assert code == 0, err
    by_expression = PreparedPlanet.load(out)
    assert by_expression.topography.parameters["expression"] == EXPR
    # same heights by both routes: identical coefficients
    np.testing.assert_array_equal(by_expression.topography_lm,
                                  world.topography_lm)

    code, _, err = _cli(
        ["run", "pe", "--terrain", str(out), "--scenario",
         "orographic_isothermal_rest", "--levels", "3", "--dt-seconds",
         "600", "--days", str(1800 / 86400), "--n-snapshots", "2",
         "--no-plots", "--out", str(tmp_path / "runs")], capsys)
    assert code == 0, err
    (run_dir,) = [d for d in (tmp_path / "runs").iterdir() if d.is_dir()]
    config = open_simulation(run_dir).storage.run_config
    assert config["topography"] == "imported"
    assert dict(config["topography_parameters"])["expression"] == EXPR
    copy = PreparedPlanet.load(run_dir)               # the capsule's terrain/
    np.testing.assert_array_equal(copy.topography_lm, world.topography_lm)


@requires_cuda
def test_cli_swe_terrain_feeds_a_shallow_water_state(tmp_path, capsys):
    from tropoi.run.custom import (FluidLayerSpec, PreparedPlanet,
                                   prepare_planet, prepare_swe_state)

    out = tmp_path / "swe terrain"
    code, stdout, err = _cli(
        ["terrain", "prepare", "--expression", "where(lat > 0, 800, -400)",
         "--resolution-deg", "1", "--lmax", str(LMAX), "--solver", "swe",
         "--radius-earth-units", "0.5", "--day-hours", "30",
         "--out", str(out)], capsys)
    assert code == 0, err
    assert "prepare_swe_state" in stdout
    world = PreparedPlanet.load(out)
    assert world.l_cut is None and np.any(world.topography_lm[LMAX])
    assert world.planet.sidereal_day_s == 30 * 3600.0
    # the CLI is the Python API: same raster, same preparation
    direct = prepare_planet(
        world.planet, world.discretization,
        ElevationRaster.from_function(
            lambda la, lo: np.where(la > 0, 800.0, -400.0),
            resolution_deg=1), solver="swe")
    np.testing.assert_array_equal(direct.topography_lm, world.topography_lm)
    fluid = FluidLayerSpec(gravity_m_s2=9.81, mean_depth_m=5000.0)
    state = prepare_swe_state(world.planet, world.discretization, fluid,
                              world, free_surface_anomaly=lambda la, lo:
                              0 * la).state
    np.testing.assert_array_equal(state.topography_lm, world.topography_lm)
    assert state.topography.parameters["expression"] == \
        "where(lat > 0, 800, -400)"
