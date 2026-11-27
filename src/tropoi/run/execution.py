"""Execution backend of a run, and its record (decisions D6/U6).

Production execution is CUDA (CuPy). CPU execution (NumPy) exists for
verification and development. It is chosen only explicitly, and never as a
fallback:

1. an explicit device: the ``--device {cuda,cpu}`` option of a run command
   (hidden from ``--help``), or the ``backend`` argument of a runner's
   Python API (``"cupy"`` or ``"numpy"``);
2. otherwise the environment variable ``TROPOI_EXECUTION=cpu-verification``,
   for subprocess tests and verification scripts;
3. otherwise CUDA.

An explicit CUDA request while ``TROPOI_EXECUTION=cpu-verification`` is
set is a conflict and an error. When CUDA is selected but unusable, the run
fails at once and explains how CPU verification is requested; it never
continues on the CPU.

:func:`execution_identity` records what actually runs: the backend, the
precision, and the GPU for a CUDA run or the CPU model for a CPU run. It
does not import CuPy or initialize CUDA for a CPU run.

Standard library and NumPy only at import.
"""
from __future__ import annotations

import os
import platform
from typing import Any, Mapping, Optional

import numpy as np

__all__ = ["CPU_VERIFICATION", "ENVIRONMENT_VARIABLE", "ExecutionError", "execution_identity",
           "require_cuda", "resolve_backend"]

ENVIRONMENT_VARIABLE = "TROPOI_EXECUTION"
CPU_VERIFICATION = "cpu-verification"
#: device names of the run commands -> array backends
DEVICES = {"cuda": "cupy", "cpu": "numpy"}
BACKENDS = ("cupy", "numpy")
HOW_TO_VERIFY_ON_CPU = (f"CPU execution is for verification only: pass --device cpu to the run "
                        f"command, backend=\"numpy\" to the Python API, or set "
                        f"{ENVIRONMENT_VARIABLE}={CPU_VERIFICATION}")


class ExecutionError(RuntimeError):
    """The requested execution is unavailable or contradictory."""


def _environment(environ: Optional[Mapping[str, str]]) -> Optional[str]:
    value = (os.environ if environ is None else environ).get(ENVIRONMENT_VARIABLE, "")
    if value in ("", None):
        return None
    if value != CPU_VERIFICATION:
        raise ExecutionError(f"{ENVIRONMENT_VARIABLE}={value!r} is not recognized; its only "
                             f"value is {CPU_VERIFICATION!r}")
    return value


def resolve_backend(*, device: Optional[str] = None, backend: Optional[str] = None,
                    environ: Optional[Mapping[str, str]] = None) -> str:
    """The array backend of a run, ``"cupy"`` or ``"numpy"``, by the
    precedence of the module docstring. ``device`` is a run command's
    ``--device`` value and ``backend`` a Python API's argument; at most one is
    given. A CUDA result is checked with :func:`require_cuda`."""
    if device is not None and backend is not None:
        raise ExecutionError("give a device or a backend, not both")
    env = _environment(environ)
    if device is not None:
        if device not in DEVICES:
            raise ExecutionError(f"device must be one of {sorted(DEVICES)}, not {device!r}")
        backend = DEVICES[device]
    if backend is not None:
        if backend not in BACKENDS:
            raise ExecutionError(f"backend must be one of {list(BACKENDS)}, not {backend!r}")
        if backend == "cupy" and env == CPU_VERIFICATION:
            raise ExecutionError(f"CUDA execution was requested explicitly while "
                                 f"{ENVIRONMENT_VARIABLE}={CPU_VERIFICATION} is set; unset one")
    else:
        backend = "numpy" if env == CPU_VERIFICATION else "cupy"
    if backend == "cupy":
        require_cuda()
    return backend


def require_cuda() -> None:
    """Raise :class:`ExecutionError` unless CuPy imports and sees a CUDA
    device."""
    try:
        import cupy  # noqa: WPS433
        count = int(cupy.cuda.runtime.getDeviceCount())
    except Exception as exc:                                  # noqa: BLE001
        raise ExecutionError(f"CUDA execution is required, but CuPy/CUDA is unusable here "
                             f"({type(exc).__name__}: {exc}). {HOW_TO_VERIFY_ON_CPU}") from exc
    if count < 1:
        raise ExecutionError(f"CUDA execution is required, but no CUDA device is visible. "
                             f"{HOW_TO_VERIFY_ON_CPU}")


def _cpu_model() -> str:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine() or "unknown"


def execution_identity(backend: str) -> dict[str, Any]:
    """What a run on ``backend`` executes on: software versions, the
    backend, the precision, and the GPU (CUDA) or the CPU model (NumPy).
    A NumPy run does not import CuPy."""
    if backend not in BACKENDS:
        raise ExecutionError(f"backend must be one of {list(BACKENDS)}, not {backend!r}")
    env: dict[str, Any] = {"python": platform.python_version(), "numpy": np.__version__,
                           "platform": platform.platform(), "backend": backend,
                           "precision": "float64"}
    if backend == "numpy":
        env.update(device="cpu", cpu=_cpu_model(), cpu_count=os.cpu_count())
        return env
    import cupy as cp  # noqa: WPS433
    env["cupy"] = cp.__version__
    env["cuda_runtime"] = int(cp.cuda.runtime.runtimeGetVersion())
    env["cuda_driver"] = int(cp.cuda.runtime.driverGetVersion())
    props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    name = props["name"]
    env["device"] = name.decode() if isinstance(name, bytes) else str(name)
    return env
