"""Python glue for the portable C++ substrate prototype (NOT production code).

Python keeps composition: it maps a configured ``ConductiveSubstrate`` to a
template specialization (law *types*) plus flat runtime data (coefficients,
per-cell materials). The arithmetic lives once, in ``include/pt``. No source
code is generated, only a type name such as
``substrate_step<51, pt::Piecewise<pt::Poly<5>, pt::SatExp>, ...>``.

* Device: ``cupy.RawModule`` compiles ``drivers/cuda_driver.cu`` with NVRTC
  (CuPy's default flags, so libdevice matches CuPy's ufuncs). The header
  include path is passed with ``-I``. CuPy caches the result on disk.
* Host: ``drivers/host_driver.cpp.in`` is instantiated and compiled with
  ``$PT_CXX`` (default ``g++``, or ``python -m ziglang c++`` where no system
  compiler exists) using ``-O2 -std=c++14 -ffp-contract=off``, cached by the
  SHA-256 of sources, flags and compiler identity, and loaded with ctypes.
"""
from __future__ import annotations

import ctypes
import hashlib
import os
import pathlib
import shlex
import subprocess
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
INCLUDE = HERE / "include"
DRIVERS = HERE / "drivers"
HOST_FLAGS = ["-O2", "-std=c++14", "-ffp-contract=off", "-fno-fast-math", "-shared", "-fPIC"]


def law_spec(law) -> tuple:
    """(C++ type, flat parameters) of a heat-capacity law object."""
    from tropoi.physics.materials import (PiecewiseHeatCapacity, PolynomialHeatCapacity,
                                          SaturatingExponentialHeatCapacity)
    if isinstance(law, PolynomialHeatCapacity):
        c = list(law.coefficients)
        return f"pt::Poly<{len(c)}>", c + [law.center, law.scale, law.factor]
    if isinstance(law, SaturatingExponentialHeatCapacity):
        return "pt::SatExp", [law.a, law.b, law.t_ref, law.scale, law.factor]
    if isinstance(law, PiecewiseHeatCapacity):
        lt, lp = law_spec(law.low)
        ht, hp = law_spec(law.high)
        return f"pt::Piecewise<{lt}, {ht}>", [law.breakpoint] + lp + hp
    raise TypeError(f"no C++ form for {type(law).__name__}")


def specialization(sub) -> tuple:
    """(L, law0 type, law1 type, segment_end, flat law parameters)."""
    segs = sub.materials.segments
    if len(segs) > 2:
        raise NotImplementedError("the prototype supports one or two strata")
    t0, p0 = law_spec(segs[0][2])
    if len(segs) == 2:
        t1, p1 = law_spec(segs[1][2])
        end = segs[0][1]
    else:
        t1, p1, end = t0, p0, sub.layers
    return sub.layers, t0, t1, int(end), np.asarray(p0 + p1, dtype=np.float64)


def _scalars(sub, dt):
    from tropoi.physics.constants import SIGMA_SB
    from tropoi.physics.substrate import TOLERANCE_FLOOR, TOLERANCE_ROUNDOFF
    dz1 = float(sub.thickness[0])
    return np.array([float(dt), 2.0 / dz1, -(2.0 / dz1), sub.emissivity * SIGMA_SB,
                     4.0 * sub.emissivity * SIGMA_SB, TOLERANCE_FLOOR, TOLERANCE_ROUNDOFF])


def _sources_digest() -> str:
    h = hashlib.sha256()
    for p in sorted(INCLUDE.rglob("*.h")) + sorted(DRIVERS.glob("*")):
        h.update(p.name.encode()); h.update(p.read_bytes())
    return h.hexdigest()


class CudaStep:
    def __init__(self, sub, block: int = 64):
        import cupy as cp
        self.sub, self.block = sub, block
        L, t0, t1, self.segment_end, params = specialization(sub)
        self.name = f"substrate_step<{L}, {t0}, {t1}>"
        code = (DRIVERS / "cuda_driver.cu").read_text()
        # the digest keeps CuPy's cache honest when a header (not the .cu) changes
        code = f"// sources {_sources_digest()}\n" + code
        self.module = cp.RawModule(code=code, options=(f"-I{INCLUDE}", f"-I{DRIVERS}",
                                                       "--std=c++14"),
                                   name_expressions=[self.name])
        self.params = cp.asarray(params)
        rho, kc, chi = sub.materials.on(cp)
        self.rho, self.kc, self.chi = (cp.ascontiguousarray(v.ravel()) for v in (rho, kc, chi))
        dz, half = sub._consts(cp)
        self.dz, self.half = cp.ascontiguousarray(dz.ravel()), cp.ascontiguousarray(half.ravel())
        self.kernel = None

    def compile(self):
        self.kernel = self.module.get_function(self.name)
        return self.kernel

    def step(self, state, dt, shortwave_down):
        import cupy as cp
        from tropoi.physics.substrate import SubstrateState
        sub = self.sub
        if self.kernel is None:
            self.compile()
        th, ths = state.anomaly, state.surface_anomaly
        n = th.shape[1]
        sw = cp.broadcast_to(cp.asarray(shortwave_down, dtype=cp.float64), (n,))
        zero = cp.zeros((n,))
        const = (1.0 - sub.albedo) * sw + sub.emissivity * zero + zero
        a = cp.zeros((n,))
        gb = cp.full((n,), float(sub.bottom_heat_flux))
        th_out, ths_out = cp.empty_like(th), cp.empty_like(ths)
        iters, fail = cp.empty(n, dtype=cp.int32), cp.empty(n, dtype=cp.int32)
        ratio = cp.empty(n)
        sd = cp.asarray(_scalars(sub, dt))
        self.kernel(((n + self.block - 1) // self.block,), (self.block,),
                    (cp.ascontiguousarray(th), cp.ascontiguousarray(ths),
                     cp.ascontiguousarray(const), a, cp.ascontiguousarray(ths), gb, self.rho,
                     self.kc, self.chi, self.dz, self.half, self.params, sd,
                     cp.int32(self.segment_end), cp.int32(sub.max_iterations), cp.int32(n),
                     th_out, ths_out, iters, fail, ratio))
        return SubstrateState(th_out, ths_out), iters, fail, ratio


def _compiler() -> list:
    spec = os.environ.get("PT_CXX")
    if spec:
        return shlex.split(spec)
    for c in ("g++", "clang++", "c++"):
        if subprocess.run(["which", c], capture_output=True).returncode == 0:
            return [c]
    return [sys.executable, "-m", "ziglang", "c++"]


class HostStep:
    """The same column step, compiled natively for the CPU (no CUDA)."""

    def __init__(self, sub, cache_dir=None):
        self.sub = sub
        L, t0, t1, self.segment_end, self.params = specialization(sub)
        src = ((DRIVERS / "host_driver.cpp.in").read_text()
               .replace("@L@", str(L)).replace("@LAW0@", t0).replace("@LAW1@", t1))
        cxx = _compiler()
        ident = subprocess.run(cxx + ["--version"], capture_output=True, text=True).stdout
        key = hashlib.sha256("\n".join([src, _sources_digest(), " ".join(cxx), ident,
                                        " ".join(HOST_FLAGS)]).encode()).hexdigest()[:20]
        cache = pathlib.Path(cache_dir or os.environ.get(
            "PT_HOST_CACHE", pathlib.Path.home() / ".cache" / "pt-host"))
        cache.mkdir(parents=True, exist_ok=True)
        lib = cache / f"substrate_{key}.so"
        self.compiled = False
        if not lib.exists():
            cpp = cache / f"substrate_{key}.cpp"
            cpp.write_text(src)
            subprocess.check_call(cxx + HOST_FLAGS + [f"-I{INCLUDE}", f"-I{DRIVERS}",
                                                       "-o", str(lib), str(cpp)])
            self.compiled = True
        self.lib = ctypes.CDLL(str(lib))
        self.compiler = " ".join(cxx)
        self.compiler_identity = ident.splitlines()[0] if ident else "?"
        self.fn = self.lib.substrate_step_host
        self.fn.restype = None
        rho, kc, chi = sub.materials.on(np)
        self.rho, self.kc, self.chi = (np.ascontiguousarray(v.ravel()) for v in (rho, kc, chi))
        dz, half = sub._consts(np)
        self.dz, self.half = np.ascontiguousarray(dz.ravel()), np.ascontiguousarray(half.ravel())

    def step(self, state, dt, shortwave_down):
        from tropoi.physics.substrate import SubstrateState
        sub = self.sub
        th = np.ascontiguousarray(state.anomaly, dtype=np.float64)
        ths = np.ascontiguousarray(state.surface_anomaly, dtype=np.float64)
        n = th.shape[1]
        sw = np.broadcast_to(np.asarray(shortwave_down, dtype=np.float64), (n,))
        zero = np.zeros((n,))
        const = np.ascontiguousarray((1.0 - sub.albedo) * sw + sub.emissivity * zero + zero)
        a = np.zeros((n,))
        gb = np.full((n,), float(sub.bottom_heat_flux))
        sc = _scalars(sub, dt)

        class S(ctypes.Structure):
            _fields_ = [(k, ctypes.c_double) for k in ("dt", "a", "b", "c", "d", "e", "f")] + [
                ("segment_end", ctypes.c_int), ("max_iterations", ctypes.c_int)]
        s = S(*sc.tolist(), self.segment_end, sub.max_iterations)
        th_out, ths_out = np.empty_like(th), np.empty_like(ths)
        iters, fail = np.empty(n, dtype=np.int32), np.empty(n, dtype=np.int32)
        ratio = np.empty(n)
        p = lambda a: a.ctypes.data_as(ctypes.c_void_p)              # noqa: E731
        self.fn(p(th), p(ths), p(const), p(a), p(ths), p(gb), p(self.rho), p(self.kc), p(self.chi),
                p(self.dz), p(self.half), p(self.params), ctypes.byref(s), ctypes.c_int(n),
                p(th_out), p(ths_out), p(iters), p(fail), p(ratio))
        return SubstrateState(th_out, ths_out), iters, fail, ratio
