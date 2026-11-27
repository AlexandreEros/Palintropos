"""Feasibility prototype (NOT production code) for decision D9: one
definition of a constitutive law, executed by NumPy/CuPy and compiled into
CUDA device code.

The law methods in ``tropoi.physics.materials`` are left exactly as they are.
They are called once with :class:`Sym` arguments. ``Sym`` is a stand-in that
records each arithmetic operation as one SSA statement of C (``double _v7 =
_v3 * _v5;``) in the order Python performs it. The recorded statements become
the body of a CuPy ``ElementwiseKernel`` (or a ``__device__`` function)
compiled with ``--fmad=false``.

Without contraction and with the same operation order, the compiled law is
expected to agree with the CuPy evaluation of the same methods bit for bit,
since each CuPy ufunc is one rounded IEEE operation. ``exp``/``expm1`` are
the same libdevice functions in both. ``minimum``/``maximum``/``where`` use
CuPy's own definitions, NaN propagation included.

This is deliberately not an expression language. Nothing is parsed, there
is no symbolic algebra or simplification, and nothing is written by hand.
It supports only the operations the laws already use. Anything else raises
``TypeError``, so an unsupported construct fails loudly instead of being
mistranslated. The NumPy evaluation of the same methods stays the
independent CPU reference.
"""
from __future__ import annotations

import math

import numpy as np


class _Trace:
    def __init__(self):
        self.lines: list[str] = []
        self.n = 0

    def emit(self, ctype: str, expr: str) -> "Sym":
        name = f"_v{self.n}"
        self.n += 1
        self.lines.append(f"{ctype} {name} = {expr};")
        return Sym(name, self, ctype)


def _lit(v) -> str:
    if isinstance(v, (bool, np.bool_)):
        return "true" if v else "false"
    f = float(v)
    if math.isnan(f):
        return "NAN"
    if math.isinf(f):
        return "INFINITY" if f > 0 else "(-INFINITY)"
    r = repr(f)                       # shortest round-tripping decimal
    return r if any(c in r for c in ".en") else r + ".0"


class Sym:
    """A traced double (or bool) value."""

    __array_priority__ = 1000

    def __init__(self, name, trace, ctype="double"):
        self.name, self.trace, self.ctype = name, trace, ctype

    # -- operands ------------------------------------------------------------
    def _o(self, other) -> str:
        if isinstance(other, Sym):
            if other.trace is not self.trace:
                raise TypeError("mixing traces")
            return other.name
        if isinstance(other, (int, float, np.floating, np.integer, bool, np.bool_)):
            return _lit(other)
        raise TypeError(f"unsupported operand {type(other).__name__} in a traced law")

    def _bin(self, op, other, reverse=False, ctype="double"):
        a, b = (self._o(other), self.name) if reverse else (self.name, self._o(other))
        return self.trace.emit(ctype, f"{a} {op} {b}")

    __add__ = lambda s, o: s._bin("+", o)
    __radd__ = lambda s, o: s._bin("+", o, True)
    __sub__ = lambda s, o: s._bin("-", o)
    __rsub__ = lambda s, o: s._bin("-", o, True)
    __mul__ = lambda s, o: s._bin("*", o)
    __rmul__ = lambda s, o: s._bin("*", o, True)
    __truediv__ = lambda s, o: s._bin("/", o)
    __rtruediv__ = lambda s, o: s._bin("/", o, True)
    __lt__ = lambda s, o: s._bin("<", o, ctype="bool")
    __le__ = lambda s, o: s._bin("<=", o, ctype="bool")
    __gt__ = lambda s, o: s._bin(">", o, ctype="bool")
    __ge__ = lambda s, o: s._bin(">=", o, ctype="bool")
    __and__ = lambda s, o: s._bin("&", o, ctype="bool")

    def __neg__(self):
        return self.trace.emit("double", f"-{self.name}")

    def __bool__(self):
        raise TypeError("a traced value has no truth value (data-dependent Python branch)")

    def __pow__(self, other):
        raise TypeError("** is not traced: CuPy and CUDA pow may round differently; "
                        "write the product explicitly")

    # -- NumPy dispatch: np.exp(sym), np.where(...), np.minimum(...) ----------
    def __array_ufunc__(self, ufunc, method, *inputs, **kw):
        if method != "__call__" or kw:
            raise TypeError(f"unsupported ufunc use {ufunc.__name__}.{method}")
        name = ufunc.__name__
        if name in ("exp", "expm1") and len(inputs) == 1:
            return self.trace.emit("double", f"{name}({self._o(inputs[0])})")
        if name in ("minimum", "maximum") and len(inputs) == 2:
            a, b = (self._o(x) for x in inputs)
            f = "min" if name == "minimum" else "max"
            return self.trace.emit("double", f"(isnan({a}) | isnan({b})) ? (double)NAN "
                                             f": {f}({a}, {b})")
        if name in ("add", "subtract", "multiply", "true_divide") and len(inputs) == 2:
            op = {"add": "+", "subtract": "-", "multiply": "*", "true_divide": "/"}[name]
            a, b = (self._o(x) for x in inputs)
            return self.trace.emit("double", f"{a} {op} {b}")
        raise TypeError(f"ufunc {name} is not supported in a traced law")

    def __array_function__(self, func, types, args, kwargs):
        if func is np.where and len(args) == 3 and not kwargs:
            c, a, b = (self._o(x) for x in args)
            return self.trace.emit("double", f"{c} ? {a} : {b}")
        raise TypeError(f"{func.__name__} is not supported in a traced law")


def trace(fn, names):
    """Trace ``fn(*syms)``; returns (statements, result variable)."""
    tr = _Trace()
    args = [Sym(n, tr) for n in names]
    out = fn(*args)
    if not isinstance(out, Sym):
        raise TypeError("the traced function must return a traced value")
    return tr.lines, out.name


#: CuPy's own float preamble (cupy/_math/misc.py), so NaN is the same NaN
_PREAMBLE = """
#ifndef NAN
#define NAN __int_as_float(0x7fffffff)
#endif
#ifndef INFINITY
#define INFINITY __longlong_as_double(0x7ff0000000000000LL)
#endif
"""


def elementwise(fn, names, kernel_name, fmad=False):
    """A CuPy ElementwiseKernel computing ``y = fn(*inputs)`` from one trace."""
    import cupy as cp
    lines, result = trace(fn, names)
    body = "\n".join(lines) + f"\ny = {result};"
    params = ", ".join(f"float64 {n}" for n in names)
    return cp.ElementwiseKernel(params, "float64 y", body, kernel_name, preamble=_PREAMBLE,
                                options=("--fmad=true",) if fmad else ("--fmad=false",)), body


def device_function(fn, names, fname):
    """The same trace as a ``__device__`` function, for a fused column kernel."""
    lines, result = trace(fn, names)
    params = ", ".join(f"const double {n}" for n in names)
    body = "\n    ".join(lines)
    return f"__device__ __forceinline__ double {fname}({params}) {{\n    {body}\n    return {result};\n}}\n"
