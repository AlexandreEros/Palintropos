"""Terrain from a height function: callables, expressions, user files (CPU).

Three ways to describe elevation as a function of position, all sampled by
one routine (:func:`sample_height_function`) onto a regular global grid and
handed to :meth:`ElevationRaster.from_array
<tropoi.spatial.terrain.raster.ElevationRaster.from_array>`, so they share
the raster's normalization, validation and provenance, and then the usual
preparation (``prepare_planet``):

* a Python callable, :meth:`ElevationRaster.from_function
  <tropoi.spatial.terrain.raster.ElevationRaster.from_function>`;
* a mathematical expression in ``lat`` and ``lon``, :class:`HeightExpression`
  (evaluated by a small allowlisted interpreter, never ``eval``);
* a callable in the user's own Python file, :func:`load_height_function`
  (``"path/to/file.py:name"``; this *executes* that file: trusted code only).

The height-function contract
----------------------------
``height(lat_deg, lon_deg) -> elevation in metres``

* **Inputs.** Two float64 NumPy arrays of the same shape ``(nlat, nlon)``
  (``numpy.meshgrid(..., indexing="ij")``): latitude in degrees, north
  positive, at cell centres ``-90 + d/2 .. 90 - d/2`` (south to north), and
  east longitude in degrees at cell centres ``lon_start + d/2 ..
  lon_start + 360 - d/2``, where ``d`` is the sampling resolution and
  ``lon_start`` is -180 (default; longitudes in [-180, 180)) or any other
  origin, such as 0 for [0, 360). The function is called **once** with the
  whole grid, so it must be vectorized (NumPy functions, not ``math``);
  ``pointwise=True`` calls it once per point with Python floats instead.
  These are degrees: :class:`tropoi.run.custom.SampledFunction` initial
  fields take radians.
* **Output.** Real elevation in metres: an array of shape ``(nlat, nlon)``,
  or anything NumPy broadcasts to it (a scalar for a constant height, an
  ``(nlat, 1)`` column for a function of latitude only, ...). Integer and
  floating dtypes are accepted; complex, boolean, object and string results,
  NaN and +-Inf, masked values and shapes that do not broadcast to
  ``(nlat, nlon)`` are rejected, naming the problem.
* **Sampling.** ``resolution_deg`` is the spacing of the sampled grid; it
  must divide 180 (2, 1, 0.5, 0.25, ... degrees) and is independent of the
  spectral truncation ``l_max`` chosen later. Preparation area-averages a
  sample finer than the truncation's working resolution and interpolates a
  coarser one (``ElevationRaster.resample_to_points``); sampling finer than
  about half the shortest resolved wavelength adds nothing the model keeps.

Expressions
-----------
:class:`HeightExpression` accepts one Python *expression* built from

* the variables ``lat`` and ``lon`` (degrees, as above) and the constant
  ``pi``; numeric literals (integers are evaluated as floats);
* ``+ - * / % **``, unary ``-``/``+``, comparisons ``< <= > >= == !=``
  (chained comparisons too) and ``& | ~`` to combine conditions;
* calls to the functions in :data:`EXPRESSION_FUNCTIONS`, all NumPy:
  ``radians degrees sin cos tan arcsin arccos arctan arctan2 sinh cosh tanh
  exp log log10 sqrt abs minimum maximum where clip floor ceil sign hypot``.

For example ``"1200*cos(radians(lat))**2*cos(2*radians(lon))"``,
``"where(lat > 0, 800, -400)"`` or simply ``"1000"``. Everything else
(statements, imports, attribute access such as ``np.sin``, subscripts,
comprehensions, lambdas, keyword arguments, ``and``/``or``/``not``,
``if``-expressions, strings, complex literals, unknown names) is rejected
before anything is evaluated.

NumPy only; never imports CuPy.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import math
import numbers
import operator
import pathlib
import re
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from tropoi.spatial.terrain.raster import (MAX_RASTER_PIXELS,
                                           ElevationImportError)

__all__ = [
    "EXPRESSION_FUNCTIONS",
    "EXPRESSION_VARIABLES",
    "HeightExpression",
    "HeightExpressionError",
    "HeightFunctionError",
    "load_height_function",
    "sample_height_function",
    "sampling_grid",
]


class HeightFunctionError(ElevationImportError):
    """A height function, its sampling settings or its result is invalid."""


class HeightExpressionError(HeightFunctionError):
    """A height expression uses syntax or names outside the allowlist."""


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

#: Coordinate convention recorded with every sampled raster.
COORDINATE_CONVENTION = ("height(lat_deg, lon_deg) in metres; latitude in "
                         "degrees north at cell centres, south to north; "
                         "east longitude in degrees at cell centres")


def sampling_grid(resolution_deg: float, lon_start_deg: float = -180.0
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Cell-centre latitudes (ascending) and longitudes of a sampling grid.

    ``resolution_deg`` must be finite, positive, at most 90 and divide 180;
    the grid must not exceed ``MAX_RASTER_PIXELS``.
    """
    try:
        d = float(resolution_deg)
    except (TypeError, ValueError):
        raise HeightFunctionError(
            f"resolution_deg must be a number, got {resolution_deg!r}"
        ) from None
    if not math.isfinite(d) or d <= 0.0:
        raise HeightFunctionError(
            f"resolution_deg must be a positive number of degrees, got {d:g}")
    if d > 90.0:
        raise HeightFunctionError(
            f"resolution_deg={d:g} is coarser than 90 degrees; the sampled "
            "grid needs at least two latitude rows")
    nlat = round(180.0 / d)
    if abs(nlat * d - 180.0) > 1e-9 * 180.0:
        raise HeightFunctionError(
            f"resolution_deg={d:g} does not divide 180 degrees; use a "
            "spacing such as 2, 1, 0.5 or 0.25 so the cells tile the globe")
    nlon = 2 * nlat
    if nlat * nlon > MAX_RASTER_PIXELS:
        raise HeightFunctionError(
            f"resolution_deg={d:g} gives {nlat} x {nlon} = {nlat * nlon} "
            f"samples, above the {MAX_RASTER_PIXELS} limit; sample more "
            "coarsely (the spectral truncation, not the sampling, decides "
            "what the model resolves)")
    try:
        start = float(lon_start_deg)
    except (TypeError, ValueError):
        raise HeightFunctionError(
            f"lon_start_deg must be a number, got {lon_start_deg!r}") from None
    if not math.isfinite(start):
        raise HeightFunctionError(
            f"lon_start_deg must be finite, got {lon_start_deg!r}")
    step = 180.0 / nlat
    lat = -90.0 + (np.arange(nlat) + 0.5) * step
    lon = start + (np.arange(nlon) + 0.5) * step
    return lat, lon


def _describe_shape(shape) -> str:
    return "scalar" if shape == () else f"shape {tuple(shape)}"


def _check_result(result: Any, shape: tuple[int, int], name: str
                  ) -> np.ndarray:
    """Validate a height function's result and broadcast it to ``shape``."""
    if isinstance(result, np.ma.MaskedArray) and np.ma.is_masked(result):
        raise HeightFunctionError(
            f"{name} returned a masked array with "
            f"{int(np.ma.count_masked(result))} masked value(s); return "
            "elevation everywhere")
    try:
        values = np.asarray(result)
    except Exception as err:  # e.g. ragged sequences
        raise HeightFunctionError(
            f"{name} returned {type(result).__name__}, which is not a "
            f"numeric array: {err}") from err
    kind = values.dtype.kind
    if kind == "c":
        raise HeightFunctionError(
            f"{name} returned complex values ({values.dtype}); elevation "
            "must be real (take .real explicitly if that is intended)")
    if kind == "b":
        raise HeightFunctionError(
            f"{name} returned booleans; elevation must be numeric metres "
            "(for example 1000 * (lat > 0) or where(lat > 0, 1000, 0))")
    if kind not in "iuf":
        raise HeightFunctionError(
            f"{name} returned {values.dtype} data ({type(result).__name__}); "
            "elevation must be real numbers in metres")
    try:
        if np.broadcast_shapes(values.shape, shape) != shape:
            raise ValueError
    except ValueError:
        raise HeightFunctionError(
            f"{name} returned {_describe_shape(values.shape)}, which does "
            f"not broadcast to the sampled grid {shape} (nlat, nlon); return "
            "an array shaped like its inputs, or a scalar for a constant "
            "height") from None
    out = np.broadcast_to(values.astype(np.float64, copy=False), shape).copy()
    bad = ~np.isfinite(out)
    if bad.any():
        j, i = np.argwhere(bad)[0]
        raise HeightFunctionError(
            f"{name} returned {int(bad.sum())} non-finite value(s) (NaN or "
            f"Inf), the first at sample index (row {j}, column {i}); "
            "elevation must be finite everywhere")
    return out


def sample_height_function(height, *, resolution_deg: float,
                           lon_start_deg: float = -180.0,
                           pointwise: bool = False, name: str | None = None
                           ) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                      dict]:
    """Sample ``height`` on the global cell-centre grid (module docstring).

    ``height`` is a callable ``height(lat_deg, lon_deg)`` or a real number
    (a constant height). Returns ``(elevation_m, lat_deg, lon_deg,
    provenance)`` with rows south to north; ``provenance`` records the
    function's name, the coordinate convention and the sampling settings.
    Exceptions raised by the function are re-raised as
    :class:`HeightFunctionError` naming it (the original is chained).
    """
    lat, lon = sampling_grid(resolution_deg, lon_start_deg)
    shape = (lat.size, lon.size)
    if isinstance(height, (bool, np.bool_)):
        raise HeightFunctionError(
            "height must be a callable or a number of metres, not a boolean")
    if isinstance(height, numbers.Real) or (
            isinstance(height, np.ndarray) and height.ndim == 0):
        label = name or "constant height"
        values = _check_result(height, shape, label)
        mode = "constant"
    elif callable(height):
        label = name or getattr(height, "__qualname__", None) \
            or type(height).__name__
        where = f"height function {label!r}"
        la, lo = np.meshgrid(lat, lon, indexing="ij")
        try:
            if pointwise:
                result = np.empty(shape, dtype=np.float64)
                for j in range(shape[0]):
                    for i in range(shape[1]):
                        point = _check_result(
                            height(float(la[j, i]), float(lo[j, i])), (),
                            f"{where} at (lat {la[j, i]:g}, lon "
                            f"{lo[j, i]:g})")
                        result[j, i] = point
            else:
                with np.errstate(all="ignore"):
                    result = height(la, lo)
        except HeightFunctionError:
            raise
        except Exception as err:
            hint = ""
            if not pointwise and isinstance(err, TypeError):
                hint = (" (the function receives whole NumPy arrays: use "
                        "numpy functions such as np.cos rather than math.cos, "
                        "or pass pointwise=True)")
            raise HeightFunctionError(
                f"{where} failed: {type(err).__name__}: {err}{hint}") from err
        values = _check_result(result, shape, where)
        mode = "pointwise" if pointwise else "vectorized"
    else:
        raise HeightFunctionError(
            "height must be a callable height(lat_deg, lon_deg) or a number "
            f"of metres, got {type(height).__name__}")
    step = 180.0 / lat.size
    provenance = {
        "source_name": label,
        "coordinate_convention": COORDINATE_CONVENTION,
        "units": "m",
        "sampling_resolution_deg": step,
        "sampling_points": "cell centres",
        "sampling_lon_range_deg": (f"[{lon[0] - step / 2:g}, "
                                   f"{lon[-1] + step / 2:g})"),
        "sampling_mode": mode,
    }
    if callable(height) and mode != "constant":
        provenance["function_name"] = label
    return values, lat, lon, provenance


# ---------------------------------------------------------------------------
# Expressions
# ---------------------------------------------------------------------------

#: Names an expression may call: NumPy functions, nothing else.
EXPRESSION_FUNCTIONS: Mapping[str, tuple[Callable, int]] = {
    "radians": (np.radians, 1), "degrees": (np.degrees, 1),
    "sin": (np.sin, 1), "cos": (np.cos, 1), "tan": (np.tan, 1),
    "arcsin": (np.arcsin, 1), "arccos": (np.arccos, 1),
    "arctan": (np.arctan, 1), "arctan2": (np.arctan2, 2),
    "sinh": (np.sinh, 1), "cosh": (np.cosh, 1), "tanh": (np.tanh, 1),
    "exp": (np.exp, 1), "log": (np.log, 1), "log10": (np.log10, 1),
    "sqrt": (np.sqrt, 1), "abs": (np.abs, 1),
    "minimum": (np.minimum, 2), "maximum": (np.maximum, 2),
    "where": (np.where, 3), "clip": (np.clip, 3),
    "floor": (np.floor, 1), "ceil": (np.ceil, 1), "sign": (np.sign, 1),
    "hypot": (np.hypot, 2),
}

#: Variables (degrees) and constants an expression may name.
EXPRESSION_VARIABLES = ("lat", "lon")
_CONSTANTS = {"pi": math.pi}

_BINARY = {ast.Add: operator.add, ast.Sub: operator.sub,
           ast.Mult: operator.mul, ast.Div: operator.truediv,
           ast.Mod: operator.mod, ast.Pow: operator.pow,
           ast.BitAnd: operator.and_, ast.BitOr: operator.or_}
_UNARY = {ast.USub: operator.neg, ast.UAdd: operator.pos,
          ast.Invert: operator.invert}
_COMPARE = {ast.Lt: operator.lt, ast.LtE: operator.le, ast.Gt: operator.gt,
            ast.GtE: operator.ge, ast.Eq: operator.eq, ast.NotEq: operator.ne}
_SYMBOL = {ast.FloorDiv: "//", ast.MatMult: "@", ast.LShift: "<<",
           ast.RShift: ">>", ast.BitXor: "^", ast.Not: "not",
           ast.Is: "is", ast.IsNot: "is not", ast.In: "in",
           ast.NotIn: "not in"}

#: Longest expression accepted (characters).
MAX_EXPRESSION_LENGTH = 10_000


def _vocabulary() -> str:
    return (f"variables {', '.join(EXPRESSION_VARIABLES)} (degrees); "
            f"constant {', '.join(_CONSTANTS)}; functions "
            f"{', '.join(EXPRESSION_FUNCTIONS)}")


def _where(node: ast.AST) -> str:
    col = getattr(node, "col_offset", None)
    return "" if col is None else f" at column {col + 1}"


def _reject(node: ast.AST, what: str) -> HeightExpressionError:
    return HeightExpressionError(f"{what}{_where(node)} is not supported in "
                                 "a height expression")


def _check_node(node: ast.AST) -> None:
    """Raise unless ``node`` (recursively) is inside the allowlist."""
    if isinstance(node, ast.Expression):
        _check_node(node.body)
    elif isinstance(node, ast.Constant):
        value = node.value
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            kind = ("complex number" if isinstance(value, complex)
                    else f"{type(value).__name__} literal {value!r}")
            raise _reject(node, f"the {kind}")
    elif isinstance(node, ast.Name):
        if node.id not in EXPRESSION_VARIABLES and node.id not in _CONSTANTS:
            if node.id in EXPRESSION_FUNCTIONS:
                raise HeightExpressionError(
                    f"{node.id!r}{_where(node)} is a function; call it, "
                    f"e.g. {node.id}(lat)")
            raise HeightExpressionError(
                f"unknown name {node.id!r}{_where(node)}; available: "
                f"{_vocabulary()}")
    elif isinstance(node, ast.BinOp):
        if type(node.op) not in _BINARY:
            raise _reject(node, "the operator "
                          f"{_SYMBOL.get(type(node.op), type(node.op).__name__)!r}")
        _check_node(node.left)
        _check_node(node.right)
    elif isinstance(node, ast.UnaryOp):
        if type(node.op) not in _UNARY:
            raise _reject(node, f"{_SYMBOL.get(type(node.op), 'the operator')!r}"
                          " (use ~ to negate a condition)")
        _check_node(node.operand)
    elif isinstance(node, ast.Compare):
        for op in node.ops:
            if type(op) not in _COMPARE:
                raise _reject(node, "the comparison "
                              f"{_SYMBOL.get(type(op), type(op).__name__)!r}")
        _check_node(node.left)
        for comparator in node.comparators:
            _check_node(comparator)
    elif isinstance(node, ast.BoolOp):
        raise _reject(node, "'and' / 'or'" + (
            "; combine conditions with & and | inside parentheses, e.g. "
            "where((lat > 0) & (lon < 90), 1000, 0)"))
    elif isinstance(node, ast.Call):
        func = node.func
        if isinstance(func, ast.Attribute):
            raise HeightExpressionError(
                f"attribute access{_where(func)} is not supported; call the "
                f"function by its bare name, e.g. {func.attr}(...) instead of "
                f"{ast.unparse(func)}(...)")
        if not isinstance(func, ast.Name):
            raise _reject(func, f"calling {type(func).__name__}")
        if func.id not in EXPRESSION_FUNCTIONS:
            raise HeightExpressionError(
                f"unknown function {func.id!r}{_where(func)}; available: "
                f"{', '.join(EXPRESSION_FUNCTIONS)}")
        if node.keywords:
            raise _reject(node, "keyword arguments")
        if any(isinstance(a, ast.Starred) for a in node.args):
            raise _reject(node, "*arguments")
        arity = EXPRESSION_FUNCTIONS[func.id][1]
        if len(node.args) != arity:
            raise HeightExpressionError(
                f"{func.id}(){_where(node)} takes {arity} argument"
                f"{'s' if arity != 1 else ''}, got {len(node.args)}")
        for arg in node.args:
            _check_node(arg)
    elif isinstance(node, ast.Attribute):
        raise HeightExpressionError(
            f"attribute access {ast.unparse(node)!r}{_where(node)} is not "
            "supported in a height expression")
    else:
        names = {ast.Subscript: "subscripts", ast.Lambda: "lambda",
                 ast.IfExp: "'x if c else y' (use where(c, x, y))",
                 ast.ListComp: "comprehensions", ast.SetComp: "comprehensions",
                 ast.DictComp: "comprehensions",
                 ast.GeneratorExp: "comprehensions",
                 ast.NamedExpr: "assignment (:=)",
                 ast.JoinedStr: "f-strings", ast.Await: "await",
                 ast.Tuple: "tuples", ast.List: "lists", ast.Dict: "dicts",
                 ast.Set: "sets", ast.Starred: "*arguments",
                 ast.Yield: "yield", ast.YieldFrom: "yield"}
        raise _reject(node, names.get(type(node), type(node).__name__))


def _evaluate(node: ast.AST, scope: Mapping[str, Any]):
    if isinstance(node, ast.Constant):
        return np.float64(node.value)    # floats: no unbounded integer powers
    if isinstance(node, ast.Name):
        return scope[node.id]
    if isinstance(node, ast.BinOp):
        return _BINARY[type(node.op)](_evaluate(node.left, scope),
                                      _evaluate(node.right, scope))
    if isinstance(node, ast.UnaryOp):
        return _UNARY[type(node.op)](_evaluate(node.operand, scope))
    if isinstance(node, ast.Compare):
        left = _evaluate(node.left, scope)
        result = None
        for op, comparator in zip(node.ops, node.comparators):
            right = _evaluate(comparator, scope)
            part = _COMPARE[type(op)](left, right)
            result = part if result is None else result & part
            left = right
        return result
    if isinstance(node, ast.Call):
        function = EXPRESSION_FUNCTIONS[node.func.id][0]
        return function(*(_evaluate(a, scope) for a in node.args))
    raise AssertionError(f"unchecked node {type(node).__name__}")  # pragma: no cover


@dataclass(frozen=True)
class HeightExpression:
    """A checked height expression in ``lat`` and ``lon`` (degrees).

    Construction parses and checks the text against the allowlist (module
    docstring) and raises :class:`HeightExpressionError` for anything
    outside it, before any evaluation. Calling it evaluates the expression
    on NumPy arrays (or floats) with the height-function signature, so it
    can be passed wherever a height function is accepted.
    """

    text: str
    _tree: ast.Expression = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise HeightExpressionError(
                f"a height expression is text, got {type(self.text).__name__}")
        text = self.text.strip()
        if not text:
            raise HeightExpressionError("the height expression is empty")
        if len(text) > MAX_EXPRESSION_LENGTH:
            raise HeightExpressionError(
                f"the height expression is {len(text)} characters long; the "
                f"limit is {MAX_EXPRESSION_LENGTH} (use --function FILE:NAME "
                "for long definitions)")
        try:
            tree = ast.parse(text, mode="eval")
        except SyntaxError as err:
            where = f" at column {err.offset}" if err.offset else ""
            raise HeightExpressionError(
                f"invalid height expression{where}: {err.msg}. Only a single "
                f"expression is accepted, e.g. "
                "1000*cos(radians(lat))**2") from None
        except (RecursionError, MemoryError, ValueError) as err:
            raise HeightExpressionError(
                f"invalid height expression: {type(err).__name__}") from None
        try:
            _check_node(tree)
        except RecursionError:
            raise HeightExpressionError(
                "the height expression is nested too deeply") from None
        object.__setattr__(self, "_tree", tree)

    def __call__(self, lat_deg, lon_deg):
        scope = {"lat": np.asarray(lat_deg, dtype=np.float64),
                 "lon": np.asarray(lon_deg, dtype=np.float64), **_CONSTANTS}
        try:
            with np.errstate(all="ignore"):
                return _evaluate(self._tree.body, scope)
        except RecursionError:
            raise HeightExpressionError(
                "the height expression is nested too deeply") from None
        except (TypeError, ValueError) as err:
            raise HeightExpressionError(
                f"the height expression could not be evaluated: {err}"
            ) from err

    def provenance(self) -> dict:
        """The expression's record: its text and the text's SHA-256."""
        return {"format": "expression", "expression": self.text.strip(),
                "expression_sha256": hashlib.sha256(
                    self.text.strip().encode("utf-8")).hexdigest()}


# ---------------------------------------------------------------------------
# A callable in a user's Python file
# ---------------------------------------------------------------------------

_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*")


def load_height_function(spec, *, cwd=None) -> tuple[Callable, dict]:
    """Load ``"path/to/file.py:name"``: the callable and its provenance.

    **This executes the file** (as ``python file.py`` would, but as a
    module, without ``__main__``): use it only with code you trust. It is
    not sandboxed.

    The specification is split at its *final* colon, so Windows paths with
    a drive letter (``C:\\terrain\\my hills.py:height``) and paths with
    spaces work; ``name`` may be dotted (``module_object.method``). A
    relative path is resolved against ``cwd`` (default: the current working
    directory). Returns ``(function, provenance)``; the provenance records
    the file's name and SHA-256 (of the bytes executed) and the callable's
    name, not its absolute path. Raises :class:`HeightFunctionError` for a
    malformed specification, a missing file, an error while executing it,
    or a missing or non-callable name.
    """
    text = str(spec)
    path_text, colon, name = text.rpartition(":")
    if not colon or not path_text.strip() or not name.strip():
        raise HeightFunctionError(
            f"--function expects FILE.py:NAME (for example "
            f"./my_terrain.py:height), got {text!r}")
    name = name.strip()
    if not _IDENTIFIER.fullmatch(name):
        raise HeightFunctionError(
            f"{name!r} after the final ':' in {text!r} is not a Python name "
            "(expected FILE.py:NAME)")
    path = pathlib.Path(path_text.strip()).expanduser()
    if not path.is_absolute():
        path = pathlib.Path.cwd() / path if cwd is None \
            else pathlib.Path(cwd) / path
    path = path.resolve()
    if not path.exists():
        raise HeightFunctionError(f"height-function file not found: {path}")
    if not path.is_file():
        raise HeightFunctionError(
            f"height-function path is not a file: {path}")
    source = path.read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    module_name = f"_tropoi_height_function_{digest[:16]}"
    module_spec = importlib.util.spec_from_file_location(module_name, path)
    if module_spec is None or module_spec.loader is None:
        raise HeightFunctionError(
            f"cannot load {path} as Python source (use a .py file)")
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_name] = module      # dataclasses etc. look it up
    try:
        module_spec.loader.exec_module(module)
    except BaseException as err:
        sys.modules.pop(module_name, None)
        if isinstance(err, (KeyboardInterrupt, SystemExit)):
            raise
        raise HeightFunctionError(
            f"error while executing {path.name}: {type(err).__name__}: "
            f"{err}") from err
    target: Any = module
    for part in name.split("."):
        try:
            target = getattr(target, part)
        except AttributeError:
            available = sorted(n for n, v in vars(module).items()
                               if callable(v) and not n.startswith("_")
                               and getattr(v, "__module__", None)
                               == module_name)
            raise HeightFunctionError(
                f"{path.name} defines no {name!r}"
                + (f"; callables defined there: {', '.join(available)}"
                   if available else "")) from None
    if not callable(target):
        raise HeightFunctionError(
            f"{name!r} in {path.name} is {type(target).__name__}, not a "
            "callable height(lat_deg, lon_deg)")
    return target, {"format": "function", "function_file": path.name,
                    "function_file_sha256": digest, "function_name": name}
