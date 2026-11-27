"""Execution of the native (C++) substrate step on the GPU and the CPU.

The canonical implementation is the C++ in ``tropoi/{numerics,physics,
temporal}/native/pt``. This module only maps a configured
:class:`~tropoi.physics.substrate.ConductiveSubstrate` to that code and
launches it:

* **GPU**: ``temporal/native/drivers/cuda_driver.cu`` is compiled at run time
  by NVRTC through CuPy, specialized exactly for the configuration (layer
  count and heat-capacity law types as template arguments), one thread per
  column. CuPy caches the binary on disk; the cache key includes a digest of
  every header (CuPy itself does not track included headers).
* **CPU**: the library ``tropoi.temporal._substrate_native`` built at
  installation (``setup.py``) from the same headers, with run-time law
  selection and a fixed layer capacity. It is called through a plain C ABI.
  No CUDA is needed, and nothing is compiled at run time. A library whose
  build digest does not match the installed sources is refused.

Nothing here defines physics or numerics; see the headers.
"""
from __future__ import annotations

import ctypes
import importlib.util
from typing import Optional

import numpy as np

from tropoi.temporal.native import build_identity

__all__ = ["NativeLibraryError", "describe", "host_library", "law_parameters", "specialization"]

ABI_VERSION = 2
REBUILD = ("build it with a C++ compiler (GCC or Clang): `pip install -e .` from the "
           "repository, or `python setup.py build_ext --inplace` for a source tree used "
           "through PYTHONPATH")


class NativeLibraryError(RuntimeError):
    """The native CPU library is missing, stale or incompatible."""


# -- configuration -> specialization and flat data ---------------------------------------

_KIND = {"poly": 0, "satexp": 1, "piecewise": 2}


def _law(law) -> tuple:
    """(C++ type, flat kind-prefixed parameters, nesting depth) of a law."""
    from tropoi.physics.materials import (PiecewiseHeatCapacity, PolynomialHeatCapacity,
                                          SaturatingExponentialHeatCapacity)
    if isinstance(law, PolynomialHeatCapacity):
        c = [float(a) for a in law.coefficients]
        return (f"pt::Poly<{len(c)}>", [_KIND["poly"], len(c), *c, law.center, law.scale,
                                        law.factor], 0, len(c))
    if isinstance(law, SaturatingExponentialHeatCapacity):
        return ("pt::SatExp", [_KIND["satexp"], law.a, law.b, law.t_ref, law.scale, law.factor],
                0, 0)
    if isinstance(law, PiecewiseHeatCapacity):
        lt, lp, ld, ln = _law(law.low)
        ht, hp, hd, hn = _law(law.high)
        return (f"pt::Piecewise<{lt}, {ht}>", [_KIND["piecewise"], law.breakpoint, *lp, *hp],
                1 + max(ld, hd), max(ln, hn))
    raise TypeError(f"no native form of the heat-capacity law {type(law).__name__}")


def specialization(sub) -> tuple:
    """(C++ law-set type, per-cell stratum index (L,), strata count, deepest
    law nesting, largest polynomial length) of a substrate."""
    types, strat = [], np.empty(sub.layers, dtype=np.int32)
    depth = poly = 0
    for s, (r0, r1, law) in enumerate(sub.materials.segments):
        t, _, d, p = _law(law)
        types.append(t)
        strat[r0:r1] = s
        depth, poly = max(depth, d), max(poly, p)
    return f"pt::StaticLaws<{', '.join(types)}>", strat, len(types), depth, poly


def law_parameters(sub) -> np.ndarray:
    """The flat law data of every stratum, in stratum order."""
    out = []
    for _, _, law in sub.materials.segments:
        out += _law(law)[1]
    return np.asarray(out, dtype=np.float64)


def scalars(sub, dt: float) -> np.ndarray:
    """The host-computed scalars of a step (pt::StepScalars), in the
    operation order of the reference: (eps sigma), ((4 eps) sigma) and
    (1 - A)."""
    from tropoi.physics.constants import SIGMA_SB
    from tropoi.physics.substrate import TOLERANCE_FLOOR, TOLERANCE_ROUNDOFF
    dz1 = float(sub.thickness[0])
    return np.array([float(dt), 2.0 / dz1, -(2.0 / dz1), sub.emissivity * SIGMA_SB,
                     4.0 * sub.emissivity * SIGMA_SB, TOLERANCE_FLOOR, TOLERANCE_ROUNDOFF,
                     1.0 - sub.albedo, sub.emissivity])


# -- CPU library ---------------------------------------------------------------------------

class _StepScalars(ctypes.Structure):
    _fields_ = [(k, ctypes.c_double) for k in ("dt", "two_over_dz1", "neg_two_over_dz1",
                                               "eps_sigma", "four_eps_sigma", "tol_floor",
                                               "tol_roundoff", "one_minus_albedo",
                                               "emissivity")]


class _ColumnInput(ctypes.Structure):
    _fields_ = [("p", ctypes.c_void_p), ("stride", ctypes.c_longlong), ("value", ctypes.c_double)]


_P = ctypes.c_void_p
_INPUTS = ("shortwave", "longwave", "flux", "dflux", "fref_temperature", "bottom")
_OUTPUTS = ("th", "ths", "iterations", "failed", "invalid", "ratio", "storage", "r", "tau", "r_s",
            "tau_s", "q_s", "emitted", "absorbed", "extent")


class _ColumnArrays(ctypes.Structure):
    _fields_ = ([("L", ctypes.c_int), ("n", ctypes.c_int), ("strata", ctypes.c_int),
                 ("th_old", _P), ("ths_old", _P)]
                + [(k, _ColumnInput) for k in _INPUTS]
                + [("has_fref", ctypes.c_int), ("source", _P), ("source_row", ctypes.c_longlong),
                   ("source_col", ctypes.c_longlong)]
                + [(k, _P) for k in ("rho", "kc", "chi", "dz", "half", "stratum")]
                + [(k, _P) for k in _OUTPUTS])


_HOST = None


def host_library():
    """The verified CPU library (loaded once); NativeLibraryError otherwise."""
    global _HOST
    if _HOST is not None:
        return _HOST
    spec = importlib.util.find_spec("tropoi.temporal._substrate_native")
    if spec is None or not spec.origin:
        raise NativeLibraryError("the native CPU substrate library tropoi.temporal."
                                 f"_substrate_native is not built; {REBUILD}")
    lib = ctypes.CDLL(spec.origin)
    for name, res in (("pt_substrate_abi_version", ctypes.c_int),
                      ("pt_substrate_build_digest", ctypes.c_char_p),
                      ("pt_substrate_build_flags", ctypes.c_char_p),
                      ("pt_substrate_layer_capacity", ctypes.c_int),
                      ("pt_substrate_max_strata", ctypes.c_int),
                      ("pt_substrate_step", ctypes.c_int),
                      ("pt_substrate_evaluate", ctypes.c_int)):
        getattr(lib, name).restype = res
    abi = lib.pt_substrate_abi_version()
    built = lib.pt_substrate_build_digest().decode()
    expected = _digest(build_identity.HOST_FLAGS)
    if abi != ABI_VERSION or built != expected:
        raise NativeLibraryError(
            f"the native CPU substrate library at {spec.origin} was built from other sources "
            f"(ABI {abi}, digest {built[:16]}...; these sources: ABI {ABI_VERSION}, "
            f"{expected[:16]}...); rebuild it: {REBUILD}")
    lib.origin = spec.origin
    lib.digest = built
    lib.flags = lib.pt_substrate_build_flags().decode()
    lib.layer_capacity = lib.pt_substrate_layer_capacity()
    lib.max_strata = lib.pt_substrate_max_strata()
    _HOST = lib
    return lib


def _check_host_capacity(sub, lib, strata, depth, poly):
    if sub.layers > lib.layer_capacity:
        raise ValueError(f"the CPU substrate library supports at most {lib.layer_capacity} "
                         f"layers; this column has {sub.layers} (it is never truncated)")
    if strata > lib.max_strata:
        raise ValueError(f"the CPU substrate library supports at most {lib.max_strata} strata; "
                         f"this column has {strata}")
    if depth > 1:
        raise ValueError("the CPU substrate library supports a piecewise law of plain laws, "
                         "not nested piecewise laws")
    if poly > 8:
        raise ValueError(f"the CPU substrate library supports polynomial laws of at most 8 "
                         f"coefficients; this one has {poly}")


# -- GPU module ---------------------------------------------------------------------------

_KERNELS: dict = {}
#: the kernel's entry point
ENTRY = "pt_substrate_step"
_DIGESTS: dict = {}


def _digest(flags) -> str:
    """The source digest, computed once per process (sources do not change
    under a running process)."""
    key = tuple(flags)
    if key not in _DIGESTS:
        _DIGESTS[key] = build_identity.digest(flags)
    return _DIGESTS[key]


def _device_kernel(sub):
    import cupy as cp
    laws, _, _, _, _ = specialization(sub)
    name = f"substrate_step<{sub.layers}, {laws}>"
    key = (name, _digest(build_identity.DEVICE_OPTIONS))
    if key not in _KERNELS:
        # A plain extern "C" entry for the specialization, not a name
        # expression: CuPy does not cache modules compiled with name
        # expressions on disk. The cache key is this source (with the digest
        # of every header) and the options.
        code = (f"// pt sources {key[1]}\n"
                + (build_identity.DRIVERS / "cuda_driver.cu").read_text()
                + f'\nextern "C" __global__ void {ENTRY}(PT_STEP_PARAMS) {{\n'
                  f"    {name}(PT_STEP_ARGS);\n}}\n")
        options = tuple(f"-I{d}" for d in (*build_identity.INCLUDE_DIRS, build_identity.DRIVERS))
        module = cp.RawModule(code=code, options=options + build_identity.DEVICE_OPTIONS)
        _KERNELS[key] = module.get_function(ENTRY)
    return _KERNELS[key], name


def _prepared(sub, xp) -> dict:
    """Per-substrate data of the native step on backend ``xp``, prepared once
    (specialization, law and stratum data, materials, the kernel) and cached
    on the substrate, so a step does no host work for them."""
    cache = sub.__dict__.setdefault("_native", {})
    key = xp.__name__
    if key not in cache:
        laws, strat, strata, depth, poly = specialization(sub)
        rho, kc, chi = (xp.ascontiguousarray(v.ravel()) for v in sub.materials.on(xp))
        dz, half = (xp.ascontiguousarray(v.ravel()) for v in sub._consts(xp))
        prep = {"strata": strata, "depth": depth, "poly": poly, "rho": rho, "kc": kc,
                "chi": chi, "dz": dz, "half": half,
                "strat": xp.asarray(np.ascontiguousarray(strat)),
                "params": xp.asarray(law_parameters(sub)), "scalars": {}}
        if xp is np:
            prep["lib"] = host_library()
            _check_host_capacity(sub, prep["lib"], strata, depth, poly)
        else:
            prep["kernel"], _ = _device_kernel(sub)
        cache[key] = prep
    return cache[key]


def _step_scalars(prep, sub, xp, dt):
    """The step scalars for ``dt`` (cached: production steps share one dt)."""
    if dt not in prep["scalars"]:
        prep["scalars"] = {dt: (scalars(sub, dt) if xp is np else xp.asarray(scalars(sub, dt)))}
    return prep["scalars"][dt]


# -- one step -------------------------------------------------------------------------------

def column_input(v, xp, n):
    """A per-column forcing input as (array or None, stride in elements,
    scalar value): a host scalar is passed by value; an array is broadcast to
    (n,) and passed as a strided view, so nothing is copied or materialized."""
    if not hasattr(v, "shape") or isinstance(v, np.generic) or (isinstance(v, np.ndarray)
                                                                and v.ndim == 0):
        return None, -1, float(np.asarray(v, dtype=np.float64))
    arr = xp.broadcast_to(xp.asarray(v, dtype=xp.float64), (n,))
    stride = arr.strides[0] // arr.itemsize
    if stride < 0 or arr.strides[0] % arr.itemsize:
        arr, stride = xp.ascontiguousarray(arr), 1
    return arr, stride, 0.0


def source_view(source, xp, L, n):
    """The source (L, n) as a strided view of the given array (or None)."""
    if source is None:
        return None, 0, 0
    view = xp.broadcast_to(source, (L, n))
    row, col = (s // view.itemsize for s in view.strides)
    if row < 0 or col < 0 or any(s % view.itemsize for s in view.strides):
        view = xp.ascontiguousarray(view)
        row, col = n, 1
    return view, row, col


def run_step(sub, xp, th_old, ths_old, dt, *, shortwave, longwave, flux, dflux,
             fref_temperature, bottom, source, max_iterations: int, block: int = 64) -> dict:
    """Advance the columns one step with the native implementation.

    The forcing inputs are host scalars or arrays broadcastable to (n,);
    ``fref_temperature`` is None for the skin's own anomaly, and ``source``
    a float64 array broadcastable to (L, n) or None. The native step
    validates the forcing and forms the absorbed constant itself. Returns
    the new anomalies, per-column iterations, failure and invalid-forcing
    flags and residual ratios, the converged evaluation's report arrays, the
    per-column extents (2 strata, n) and ``summary`` (largest iteration
    count, any failed, any invalid), on the host for NumPy and on the device
    for CuPy."""
    L, n = th_old.shape
    f64 = lambda a: xp.ascontiguousarray(a, dtype=xp.float64)       # noqa: E731
    th_old, ths_old = f64(th_old), f64(ths_old)
    has_fref = fref_temperature is not None
    inputs = [column_input(v, xp, n) for v in (shortwave, longwave, flux, dflux,
                                               fref_temperature if has_fref else 0.0, bottom)]
    src, s_row, s_col = source_view(source, xp, L, n)
    prep = _prepared(sub, xp)
    S = prep["strata"]
    out = {"th": xp.empty((L, n)), "ths": xp.empty(n),
           "iterations": xp.empty(n, dtype=xp.int32), "failed": xp.empty(n, dtype=xp.int32),
           "invalid": xp.empty(n, dtype=xp.int32),
           "ratio": xp.empty(n), "storage": xp.empty((L, n)), "r": xp.empty((L, n)),
           "tau": xp.empty((L, n)), "r_s": xp.empty(n), "tau_s": xp.empty(n),
           "q_s": xp.empty(n), "emitted": xp.empty(n), "absorbed": xp.empty(n),
           "extent": xp.empty((2 * S, n))}
    sd = _step_scalars(prep, sub, xp, float(dt))
    if xp is np:
        lib = prep["lib"]
        ptr = lambda a: None if a is None else a.ctypes.data                  # noqa: E731
        arrays = _ColumnArrays(
            L, n, S, ptr(th_old), ptr(ths_old),
            *[_ColumnInput(ptr(arr), stride, value) for arr, stride, value in inputs],
            int(has_fref), ptr(src), s_row, s_col,
            *[ptr(prep[k]) for k in ("rho", "kc", "chi", "dz", "half", "strat")],
            *[ptr(out[k]) for k in _OUTPUTS])
        summary = np.zeros(3, dtype=np.int32)
        status = lib.pt_substrate_step(ctypes.byref(arrays), ctypes.byref(_StepScalars(*sd)),
                                       prep["params"].ctypes.data_as(ctypes.c_void_p),
                                       ctypes.c_int(max_iterations),
                                       summary.ctypes.data_as(ctypes.c_void_p))
        if status:
            raise NativeLibraryError(f"the CPU substrate library rejected the column (status "
                                     f"{status})")
        out["summary"] = summary
        return out
    dummy = prep.setdefault("dummy", xp.zeros(1))
    args = []
    for arr, stride, value in inputs:
        args += [dummy if arr is None else arr, np.int64(stride), np.float64(value)]
    out["summary"] = summary = xp.zeros(3, dtype=xp.int32)
    prep["kernel"](((n + block - 1) // block,), (block,),
                   (np.int32(L), np.int32(n), np.int32(S), th_old, ths_old, *args[:15],
                    np.int32(has_fref), *args[15:], dummy if src is None else src,
                    np.int32(src is not None), np.int64(s_row), np.int64(s_col), prep["rho"],
                    prep["kc"], prep["chi"], prep["dz"], prep["half"], prep["strat"],
                    prep["params"], sd, np.int32(max_iterations),
                    *[out[k] for k in _OUTPUTS], summary))
    return out


def evaluate_host(sub, th_old, x, dt, absorbed, dflux, fref, bottom, source=None) -> tuple:
    """Residuals and Jacobian (lower, diag, upper) of one column at the
    unknowns ``x`` = [skin, cells] (anomalies), with the skin balance's
    absorbed constant, a, reference anomaly and G given directly, from the
    CPU library: for verifying the analytic Jacobian against finite
    differences."""
    lib = host_library()
    L = sub.layers
    laws, strat, strata, depth, poly = specialization(sub)
    _check_host_capacity(sub, lib, strata, depth, poly)
    th_old = np.ascontiguousarray(np.reshape(th_old, (L, 1)), dtype=np.float64)
    src = None if source is None else np.ascontiguousarray(np.reshape(source, (L, 1)),
                                                           dtype=np.float64)
    rho, kc, chi = (np.ascontiguousarray(v.ravel()) for v in sub.materials.on(np))
    dz, half = (np.ascontiguousarray(v.ravel()) for v in sub._consts(np))
    strat = np.ascontiguousarray(strat)
    ptr = lambda a: None if a is None else a.ctypes.data                      # noqa: E731
    none = _ColumnInput(None, -1, 0.0)
    arrays = _ColumnArrays(L, 1, strata, ptr(th_old), None, *([none] * len(_INPUTS)), 0,
                           ptr(src), 1, 0, *[ptr(a) for a in (rho, kc, chi, dz, half, strat)],
                           *([None] * len(_OUTPUTS)))
    xin = np.ascontiguousarray(x, dtype=np.float64)
    R, lo, di, up = np.empty(L + 1), np.empty(L), np.empty(L + 1), np.empty(L)
    params = law_parameters(sub)
    first = lambda v: float(np.reshape(v, (-1,))[0])                          # noqa: E731
    status = lib.pt_substrate_evaluate(
        ctypes.byref(arrays), ctypes.byref(_StepScalars(*scalars(sub, dt))),
        params.ctypes.data_as(ctypes.c_void_p), ctypes.c_double(first(absorbed)),
        ctypes.c_double(first(dflux)), ctypes.c_double(first(fref)),
        ctypes.c_double(first(bottom)), xin.ctypes.data_as(ctypes.c_void_p),
        R.ctypes.data_as(ctypes.c_void_p), lo.ctypes.data_as(ctypes.c_void_p),
        di.ctypes.data_as(ctypes.c_void_p), up.ctypes.data_as(ctypes.c_void_p))
    if status:
        raise NativeLibraryError(f"the CPU substrate library rejected the column ({status})")
    return R, lo, di, up


def describe(sub, backend: str) -> dict:
    """Provenance of the implementation a run uses, without compiling."""
    laws, _, _, _, _ = specialization(sub)
    out = {"implementation": "native C++ (tropoi pt headers), backward Euler + safeguarded "
                             "Newton + Thomas, FP64, no FMA contraction",
           "source_digest_host": build_identity.digest(build_identity.HOST_FLAGS),
           "source_digest_device": build_identity.digest(build_identity.DEVICE_OPTIONS),
           "backend": backend}
    if backend == "cupy":
        try:
            import cupy as cp
            out["nvrtc_version"] = list(cp.cuda.nvrtc.getVersion())
        except Exception as exc:                                    # noqa: BLE001
            out["nvrtc_version"] = f"unavailable: {exc!r}"
        out["device_specialization"] = f"substrate_step<{sub.layers}, {laws}>"
        out["device_options"] = list(build_identity.DEVICE_OPTIONS)
    else:
        try:
            lib = host_library()
            out.update(host_library=lib.origin, host_build=lib.flags,
                       host_layer_capacity=lib.layer_capacity)
        except NativeLibraryError as exc:
            out["host_library"] = f"unavailable: {exc}"
    return out
