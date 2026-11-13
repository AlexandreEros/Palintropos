"""Bounded, opt-in runtime evidence for the architecture atlas.

Records which analysed-package functions actually called which, in a named
scenario, plus selected *boundary* observations: the metadata of arguments,
return values and receiver state at functions chosen with ``--boundary``.
The output is a trace JSON (``schema`` 2) that the generator joins into
``interactive.html`` when it is committed under
``docs/architecture/evidence/runtime/``.

Two ways to name the code that runs, both IN THIS PROCESS::

    # a registered scenario (tools/architecture/scenarios.py): phases,
    # configuration identity and verification are recorded by the driver
    python -m tools.architecture.runtime_capture --scenario radiative-tiny-lifecycle \\
        --out docs/architecture/evidence/runtime/radiative-tiny-lifecycle.json

    # any module or script, as on the command line
    python -m tools.architecture.runtime_capture --scenario help --out /tmp/t.json \\
        --boundary 'tropoi.cli.*' --module tropoi.cli.main -- run pe --help

What is and is not measured
---------------------------
* Calls are Python ``call``/``return`` events (``sys.setprofile``) in the
  thread that runs the scenario. Work in other threads or processes, native
  code and GPU kernels are not seen as calls. A generator resume counts as
  a call.
* ``wall_ns_inclusive`` is host wall time between call and return, including
  callees, tracing overhead and *asynchronous* GPU launches (a CuPy call
  usually returns before its kernel finishes). It is never GPU execution
  time. GPU execution time must come from a separate measurement
  (``tools/colab/radiative_pilot.py profile --mode sync`` or Nsight).
* The collector inserts no GPU synchronization and never copies device
  memory: arrays are described by type, shape, dtype, size, device pointer
  and location, read from host-side metadata. Small CPU arrays (up to
  ``--fingerprint-bytes``) get a SHA-256 content fingerprint.
* Device events: calls that *do* synchronize or move data and are visible
  as Python or C calls (``Stream.synchronize``, ``deviceSynchronize``,
  ``ndarray.get``/``item``/``tolist``, ``cupy.asnumpy``, ``cupy.asarray``
  of a NumPy array) are counted per phase and caller. This is a lower
  bound: implicit conversions such as ``bool(device_array)`` go through C
  slots and are invisible.
* Mutation: at a boundary the receiver's attributes and the arguments are
  described before and after the call and compared: rebinding (identity
  change), value change (scalars), metadata change (shape/dtype) and
  fingerprint change (small CPU arrays). An array that is the same object
  before and after without a fingerprint (device or large) is listed as
  ``unverified``: in-place updates there are invisible.
* Strings are truncated; values under keys that look like secrets are
  redacted; the repository root and home directory are replaced by
  ``<repo>``/``<home>``. Nothing infers units or meanings from names.
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import fnmatch
import hashlib
import json
import math
import os
import platform
import re
import runpy
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

SCHEMA = 2
PACKAGE = "tropoi"
REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclasses.dataclass
class Limits:
    """Every bound on what one capture records (all stated in the output)."""

    max_edges: int = 20000
    max_boundary_examples: int = 1        # per (function, phase)
    max_boundaries: int = 300             # total
    stored_depth: int = 1                 # nesting kept in the output (diffs use max_depth)
    max_attrs: int = 24                   # attributes/fields per object
    max_items: int = 6                    # items per container
    max_str: int = 160
    max_depth: int = 2
    fingerprint_bytes: int = 4096
    max_lines_per_edge: int = 4
    max_json_bytes: int = 3_000_000


_SENSITIVE = re.compile(r"(secret|token|passw|api[_-]?key|credential|auth|cookie|private)", re.I)
_DEVICE_C = {"synchronize": "synchronize", "deviceSynchronize": "synchronize",
             "get": "device_to_host", "item": "device_to_host", "tolist": "device_to_host"}

_ACTIVE: "Recorder | None" = None


# ---------------------------------------------------------------------------
# Value description (never calls user code, never touches device memory)
# ---------------------------------------------------------------------------

def _module_root(obj) -> str:
    return (type(obj).__module__ or "").split(".", 1)[0]


def _is_array(value) -> bool:
    typ = type(value)
    root = (typ.__module__ or "").split(".", 1)[0]
    if root in ("numpy", "cupy") and hasattr(typ, "shape") and hasattr(typ, "dtype"):
        return not _is_numpy_scalar(value)
    return hasattr(typ, "__array_interface__") or hasattr(typ, "__cuda_array_interface__")


def _is_numpy_scalar(value) -> bool:
    np = sys.modules.get("numpy")
    return np is not None and isinstance(value, np.generic)


def _on_device(value) -> bool:
    typ = type(value)
    # Class-level checks only: reading __cuda_array_interface__ on an
    # instance may synchronize the stream in CuPy.
    return (typ.__module__ or "").startswith("cupy") or hasattr(typ, "__cuda_array_interface__")


class Describer:
    def __init__(self, limits: Limits, scrub: list[tuple[str, str]]):
        self.limits = limits
        self.scrub = scrub

    def text(self, value: str, key: str | None = None) -> dict:
        if key is not None and _SENSITIVE.search(key):
            return {"type": "str", "value": "<redacted>", "redacted": True}
        for raw, label in self.scrub:
            if raw:
                value = value.replace(raw, label)
        cap = self.limits.max_str
        out = {"type": "str", "value": value[:cap]}
        if len(value) > cap:
            out["truncated"] = True
            out["length"] = len(value)
        return out

    def __call__(self, value, depth: int = 0, key: str | None = None) -> dict:
        L = self.limits
        if key is not None and _SENSITIVE.search(key):
            # Before any type dispatch: a secret may be a number, a list or an
            # object, and none of its contents may reach the trace.
            return {"type": type(value).__name__, "value": "<redacted>", "redacted": True}
        if value is None or isinstance(value, bool):
            return {"type": type(value).__name__, "value": value}
        if isinstance(value, int):
            return {"type": "int", "value": value if abs(value) < 2 ** 53 else str(value)}
        if isinstance(value, float):
            if not math.isfinite(value):
                return {"type": "float", "value": str(value), "nonfinite": True}
            return {"type": "float", "value": value}
        if isinstance(value, complex):
            return {"type": "complex", "value": str(value)}
        if isinstance(value, str):
            return self.text(value, key)
        if isinstance(value, (bytes, bytearray)):
            return {"type": type(value).__name__, "length": len(value)}
        if isinstance(value, Path):
            return {"type": "Path", **{k: v for k, v in self.text(str(value), key).items()
                                       if k != "type"}}
        typ = type(value)
        name = f"{typ.__module__}.{typ.__qualname__}"
        if _is_numpy_scalar(value):
            item = value.item()        # host scalar, no device involved
            return {"type": name, "value": item if isinstance(item, (int, float, bool))
                    and (not isinstance(item, float) or math.isfinite(item)) else str(item)}
        if _is_array(value):
            return self.array(value, name)
        if isinstance(value, type):
            return {"type": "class", "value": f"{value.__module__}.{value.__qualname__}"}
        if callable(value) and hasattr(value, "__qualname__") and not hasattr(value, "__dict__"):
            return {"type": "callable", "value": f"{getattr(value, '__module__', '')}."
                                                 f"{value.__qualname__}"}
        out: dict[str, Any] = {"type": name, "id": id(value)}
        if isinstance(value, (list, tuple, set, frozenset)):
            out["length"] = len(value)
            if depth < L.max_depth and not isinstance(value, (set, frozenset)):
                out["fields"] = {f"[{i}]": self(v, depth + 1) for i, v in
                                 enumerate(list(value)[:L.max_items])}
                if len(value) > L.max_items:
                    out["omitted"] = len(value) - L.max_items
            return out
        if isinstance(value, dict):
            out["length"] = len(value)
            if depth < L.max_depth:
                keys = list(value)[:L.max_items]
                out["fields"] = {str(k)[:60]: self(value[k], depth + 1, str(k)) for k in keys}
                if len(value) > L.max_items:
                    out["omitted"] = len(value) - L.max_items
            return out
        if hasattr(value, "__module__") and type(value).__name__ == "module":
            return {"type": "module", "value": value.__name__}
        if depth < L.max_depth:
            fields = self.fields(value)
            if fields is not None:
                names = sorted(fields)
                out["fields"] = {k: self(fields[k], depth + 1, k) for k in names[:L.max_attrs]}
                if len(names) > L.max_attrs:
                    out["omitted"] = len(names) - L.max_attrs
        return out

    @staticmethod
    def fields(value) -> dict | None:
        """An object's own attributes without running properties or methods."""
        own = getattr(value, "__dict__", None)
        if isinstance(own, dict):
            return dict(own)
        slots = [s for cls in type(value).__mro__ for s in getattr(cls, "__slots__", ())]
        if slots:
            out = {}
            for s in slots:
                try:
                    out[s] = object.__getattribute__(value, s)
                except AttributeError:
                    pass
            return out
        return None

    def array(self, value, name: str) -> dict:
        gpu = _on_device(value)
        out: dict[str, Any] = {"type": name, "id": id(value),
                               "location": "GPU" if gpu else "CPU"}
        try:
            out["shape"] = [int(n) for n in value.shape]
            out["dtype"] = str(value.dtype)
            nbytes = getattr(value, "nbytes", None)
            if nbytes is not None:
                out["bytes"] = int(nbytes)
        except Exception:                                       # noqa: BLE001
            return out
        if gpu:
            try:
                out["ptr"] = int(value.data.ptr)                 # host-side metadata
                out["device"] = int(value.device.id)
            except Exception:                                   # noqa: BLE001
                pass
            return out
        try:
            out["ptr"] = int(value.__array_interface__["data"][0])
        except Exception:                                       # noqa: BLE001
            pass
        if out.get("bytes", 1 << 62) <= self.limits.fingerprint_bytes:
            try:
                np = sys.modules["numpy"]
                out["fingerprint"] = hashlib.sha256(np.ascontiguousarray(value).tobytes()
                                                    ).hexdigest()
            except Exception:                                   # noqa: BLE001
                pass
        return out


def prune(desc, depth: int):
    """A description cut to ``depth`` levels of nested fields (for storage)."""
    if not isinstance(desc, dict):
        return desc
    out = {k: v for k, v in desc.items() if k != "fields"}
    if "fields" in desc:
        if depth > 0:
            out["fields"] = {k: prune(v, depth - 1) for k, v in desc["fields"].items()}
        else:
            out["fields_omitted"] = len(desc["fields"])
    return out


def changes(before: dict | None, after: dict | None, prefix: str = "") -> tuple[dict, list]:
    """``({path: change}, [unverified paths])`` between two descriptions."""
    out: dict[str, dict] = {}
    unverified: list[str] = []
    b = (before or {}).get("fields") or {}
    a = (after or {}).get("fields") or {}
    for key in sorted(set(b) | set(a)):
        path = f"{prefix}{key}"
        x, y = b.get(key), a.get(key)
        if x is None:
            out[path] = {"kind": "added", "after": y}
            continue
        if y is None:
            out[path] = {"kind": "removed"}
            continue
        if "id" in x and "id" in y and x["id"] != y["id"]:
            out[path] = {"kind": "rebound", "after": y}
            continue
        if "shape" in x or "shape" in y:
            if (x.get("shape"), x.get("dtype")) != (y.get("shape"), y.get("dtype")):
                out[path] = {"kind": "metadata_changed", "after": y}
            elif x.get("ptr") != y.get("ptr"):
                out[path] = {"kind": "buffer_replaced", "after": y}
            elif "fingerprint" in x and "fingerprint" in y:
                if x["fingerprint"] != y["fingerprint"]:
                    out[path] = {"kind": "content_changed", "after": y}
            else:
                unverified.append(path)
            continue
        if "value" in x or "value" in y:
            if x.get("value") != y.get("value"):
                out[path] = {"kind": "value_changed", "after": y}
            continue
        if x.get("length") != y.get("length"):
            out[path] = {"kind": "length_changed", "after": {"type": y.get("type"),
                                                             "length": y.get("length")}}
        if "fields" in x and "fields" in y:
            sub, unv = changes(x, y, prefix=f"{path}.")
            out.update(sub)
            unverified.extend(unv)
    return out, unverified


# ---------------------------------------------------------------------------
# The recorder
# ---------------------------------------------------------------------------

def _module_name(frame) -> str:
    module = frame.f_globals.get("__name__", "")
    if module == "__main__":
        spec = frame.f_globals.get("__spec__")
        module = getattr(spec, "name", None) or module
    return module


def function_name(frame) -> str:
    return _module_name(frame) + "." + frame.f_code.co_qualname


class Recorder:
    def __init__(self, patterns=(), limits: Limits | None = None,
                 describer: Describer | None = None, package: str = PACKAGE):
        self.patterns = tuple(patterns)
        self.limits = limits or Limits()
        self.describe = describer or Describer(self.limits, [])
        self.package, self.prefix = package, package + "."
        self.phase = "default"
        self.phases_seen: list[str] = ["default"]
        self.stack: dict[int, tuple] = {}
        self.edges: dict[tuple[str, str], dict] = {}
        self.functions: dict[str, Counter] = defaultdict(Counter)
        self.boundaries: list[dict] = []
        self.boundary_counts: Counter = Counter()
        self.device_events: Counter = Counter()
        self.selected_cache: dict[str, bool] = {}
        self.calls_seen = 0
        self.dropped_edges = 0
        self.dropped_boundaries = 0
        self._busy = False

    def set_phase(self, name: str) -> None:
        self.phase = name
        if name not in self.phases_seen:
            self.phases_seen.append(name)

    def _ours(self, module: str) -> bool:
        return module == self.package or module.startswith(self.prefix)

    def _caller(self, frame):
        parent = frame.f_back
        how = "direct"
        while parent is not None:
            if self._ours(_module_name(parent)):
                return function_name(parent), parent.f_lineno, how
            how = "indirect"
            parent = parent.f_back
        return None, None, None

    def _selected(self, name: str) -> bool:
        hit = self.selected_cache.get(name)
        if hit is None:
            hit = any(fnmatch.fnmatchcase(name, p) for p in self.patterns)
            self.selected_cache[name] = hit
        return hit

    def _device(self, frame, kind: str) -> None:
        caller = function_name(frame) if self._ours(_module_name(frame)) else (
            self._caller(frame)[0] or "<outside package>")
        self.device_events[(self.phase, caller, kind)] += 1

    def __call__(self, frame, event, arg):
        if self._busy:
            return
        if event == "c_call":
            kind = _DEVICE_C.get(getattr(arg, "__name__", ""))
            if kind:
                owner = getattr(arg, "__self__", None)
                if isinstance(owner, type(sys)):
                    mod = owner.__name__
                elif owner is not None:
                    mod = type(owner).__module__ or ""
                else:
                    mod = getattr(arg, "__module__", "") or ""
                if mod.startswith("cupy"):
                    self._device(frame, kind)
            return
        if event not in ("call", "return"):
            return
        module = _module_name(frame)
        if not self._ours(module):
            if event == "call" and module.startswith("cupy"):
                q = frame.f_code.co_qualname
                if q == "asnumpy":
                    self._device(frame.f_back or frame, "device_to_host")
                elif q in ("asarray", "array"):
                    first = frame.f_locals.get("a", frame.f_locals.get("obj"))
                    if first is not None and _module_root(first) == "numpy":
                        self._device(frame.f_back or frame, "host_to_device")
            return
        key = id(frame)
        if event == "call":
            self.calls_seen += 1
            name = module + "." + frame.f_code.co_qualname
            caller, line, how = self._caller(frame)
            before = None
            if self._selected(name):
                slot = (name, self.phase)
                self.boundary_counts[slot] += 1
                if (self.boundary_counts[slot] <= self.limits.max_boundary_examples):
                    if len(self.boundaries) < self.limits.max_boundaries:
                        before = self._snapshot(frame)
                    else:
                        self.dropped_boundaries += 1
            self.stack[key] = (caller, name, line, how, self.phase, time.perf_counter_ns(),
                               before)
            return
        start = self.stack.pop(key, None)
        if start is None:
            return
        caller, name, line, how, phase, since, before = start
        elapsed = time.perf_counter_ns() - since
        self.functions[name][phase] += 1
        if before is not None:
            self._finish_boundary(frame, name, phase, before, arg)
        if caller is None:
            return
        edge = self.edges.get((caller, name))
        if edge is None:
            if len(self.edges) >= self.limits.max_edges:
                self.dropped_edges += 1
                return
            edge = self.edges[(caller, name)] = {"count": 0, "wall_ns_inclusive": 0,
                                                 "phases": Counter(), "lines": []}
        edge["count"] += 1
        edge["wall_ns_inclusive"] += elapsed
        edge["phases"][phase] += 1
        site = [line, how]
        if site not in edge["lines"] and len(edge["lines"]) < self.limits.max_lines_per_edge:
            edge["lines"].append(site)

    def _snapshot(self, frame) -> dict:
        self._busy = True
        try:
            code = frame.f_code
            n = code.co_argcount + code.co_kwonlyargcount
            n += bool(code.co_flags & 0x04) + bool(code.co_flags & 0x08)
            names = code.co_varnames[:n]
            local = frame.f_locals
            args = {k: self.describe(local[k], 0, k) for k in names[:16]
                    if k in local and k not in ("self", "cls")}
            receiver = local.get("self")
            return {"args": args,
                    "self": None if receiver is None else self.describe(receiver, 0),
                    "_objects": {k: local[k] for k in names[:16] if k in local}}
        finally:
            self._busy = False

    def _finish_boundary(self, frame, name, phase, before, value) -> None:
        self._busy = True
        try:
            objects = before.pop("_objects")
            keep = self.limits.stored_depth
            record = {"function": name, "phase": phase,
                      "call_index": self.boundary_counts[(name, phase)],
                      "args": {k: prune(v, keep) for k, v in before["args"].items()},
                      "return": prune(self.describe(value, 0), keep)}
            mutations, unverified = {}, []
            if before["self"] is not None and "self" in objects:
                after = self.describe(objects["self"], 0)
                record["self_before"] = prune(before["self"], keep)
                m, u = changes(before["self"], after, "self.")
                mutations.update(m)
                unverified += u
            for k, desc in before["args"].items():
                if k in objects and ("fields" in desc or "shape" in desc):
                    after = self.describe(objects[k], 0, k)
                    if "shape" in desc:
                        m, u = changes({"fields": {k: desc}}, {"fields": {k: after}})
                    else:
                        m, u = changes(desc, after, f"{k}.")
                    mutations.update(m)
                    unverified += u
            record["mutations"] = {k: {**v, **({"after": prune(v["after"], 0)}
                                                if "after" in v else {})}
                                   for k, v in mutations.items()}
            record["unverified"] = sorted(set(unverified))
            self.boundaries.append(record)
        finally:
            self._busy = False

    # -- output ----------------------------------------------------------
    def to_edges(self) -> list[dict]:
        return [{"source": s, "target": t, "count": e["count"],
                 "wall_ns_inclusive": e["wall_ns_inclusive"],
                 "phases": dict(e["phases"]), "lines": e["lines"]}
                for (s, t), e in sorted(self.edges.items())]

    def device_summary(self) -> list[dict]:
        return [{"phase": p, "caller": c, "kind": k, "count": n}
                for (p, c, k), n in sorted(self.device_events.items())]


@contextlib.contextmanager
def phase(name: str):
    """Label the calls made inside the block (no-op when nothing is recording)."""
    rec = _ACTIVE
    if rec is None:
        yield
        return
    previous = rec.phase
    rec.set_phase(name)
    try:
        yield
    finally:
        rec.set_phase(previous)


def set_phase(name: str) -> None:
    if _ACTIVE is not None:
        _ACTIVE.set_phase(name)


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

def _git(root: Path, *args) -> str | None:
    try:
        return subprocess.check_output(["git", *args], cwd=root, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def code_identity(root: Path, src: str = "src", package: str = PACKAGE) -> dict:
    """Revision, dirty status and content hash of the code a capture ran."""
    status_src = (_git(root, "status", "--porcelain", "--untracked-files=all", "--",
                       f"{src}/{package}") or "").splitlines()
    status_tools = (_git(root, "status", "--porcelain", "--untracked-files=all", "--",
                         "tools") or "").splitlines()
    out: dict[str, Any] = {"commit": _git(root, "rev-parse", "HEAD"),
                           "branch": _git(root, "rev-parse", "--abbrev-ref", "HEAD"),
                           "source_dirty": bool(status_src), "source_changes": status_src[:40],
                           "tools_dirty": bool(status_tools),
                           "tools_changes": status_tools[:40]}
    try:
        from .provenance import working_tree
        tree = working_tree(root, src, package)
        out["source_sha256"] = tree["source"]["sha256"]
        out["source_hash_method"] = ("tools.architecture.provenance.working_tree: SHA-256 over "
                                     "LF-normalized .py/.cu/.html files (comparable with a "
                                     "snapshot's metrics.json)")
    except Exception as exc:                                    # noqa: BLE001
        out["source_sha256"] = None
        out["source_hash_error"] = repr(exc)[:200]
    if status_src:
        diff = _git(root, "diff", "HEAD", "--", f"{src}/{package}") or ""
        out["source_diff_sha256"] = hashlib.sha256(diff.encode()).hexdigest()
    return out


def _versions() -> dict:
    out = {}
    for name in ("numpy", "scipy", "cupy", "matplotlib", "grimp", "networkx"):
        mod = sys.modules.get(name)
        if mod is not None:
            out[name] = getattr(mod, "__version__", "?")
    return out


def _nvidia_smi() -> str | None:
    try:
        return subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
             "--format=csv,noheader"], text=True, timeout=20).strip()
    except Exception:                                           # noqa: BLE001
        return None


def environment(device_query: bool = True) -> dict:
    """Interpreter, packages already imported, and devices (host-side queries only)."""
    env: dict[str, Any] = {
        "python": sys.version.split()[0], "implementation": platform.python_implementation(),
        "platform": platform.platform(), "machine": platform.machine(),
        "packages": _versions(),
        "variables": {k: os.environ[k] for k in ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS",
                                                 "OPENBLAS_NUM_THREADS",
                                                 "CUPY_CUDA_ARRAY_INTERFACE_SYNC")
                      if k in os.environ}}
    if device_query:
        env["nvidia_smi"] = _nvidia_smi()
    cp = sys.modules.get("cupy")
    if cp is not None:
        try:
            rt = cp.cuda.runtime
            props = rt.getDeviceProperties(cp.cuda.Device().id)
            name = props["name"]
            env["device"] = {"name": name.decode() if isinstance(name, bytes) else str(name),
                             "compute_capability": f"{props['major']}.{props['minor']}",
                             "total_memory_bytes": int(props["totalGlobalMem"]),
                             "cuda_runtime": rt.runtimeGetVersion(),
                             "cuda_driver": rt.driverGetVersion()}
        except Exception as exc:                                # noqa: BLE001
            env["device"] = {"error": repr(exc)[:200]}
    else:
        env["device"] = None
    return env


def _scrub_pairs(root: Path) -> list[tuple[str, str]]:
    pairs = [(str(root), "<repo>"), (str(root).replace("\\", "/"), "<repo>"),
             (str(Path.home()), "<home>"), (str(Path.home()).replace("\\", "/"), "<home>")]
    return sorted(set(pairs), key=lambda p: -len(p[0]))


def scrub(value, pairs):
    if isinstance(value, str):
        for raw, label in pairs:
            value = value.replace(raw, label)
        return value
    if isinstance(value, list):
        return [scrub(v, pairs) for v in value]
    if isinstance(value, dict):
        return {k: ("<redacted>" if _SENSITIVE.search(str(k)) else scrub(v, pairs))
                for k, v in value.items()}
    return value


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------

class Notes:
    """What a scenario driver adds: configuration identity, verification, notes."""

    def __init__(self):
        self.configuration: dict | None = None
        self.verification: dict | None = None
        self.synchronization_note: str | None = None
        self.attachments: dict = {}


def capture(scenario: str, func: Callable[[Notes], Any], *, command: str,
            boundaries=(), root: Path | None = None, limits: Limits | None = None,
            configuration: dict | None = None, description: str = "",
            measure_overhead: Callable[["Notes"], Any] | None = None,
            device_query: bool = True, package: str = PACKAGE) -> dict:
    """Run ``func(notes)`` once under the recorder and return the trace document.

    ``measure_overhead``, when given, is an equivalent uninstrumented run
    executed first; its wall time is recorded next to the traced one.
    """
    global _ACTIVE
    root = (root or REPO_ROOT).resolve()
    limits = limits or Limits()
    pairs = _scrub_pairs(root)
    code = code_identity(root)
    env_before = environment(device_query)
    baseline_ns = cold_ns = None
    if measure_overhead is not None:
        # Cold run absorbs one-time costs (imports, CUDA context, kernel
        # compilation); the warm run is the baseline the traced run compares to.
        t0 = time.perf_counter_ns()
        measure_overhead(Notes())
        cold_ns = time.perf_counter_ns() - t0
        t0 = time.perf_counter_ns()
        measure_overhead(Notes())
        baseline_ns = time.perf_counter_ns() - t0
    rec = Recorder(boundaries, limits, Describer(limits, pairs), package=package)
    notes = Notes()
    error = None
    begin = time.perf_counter_ns()
    _ACTIVE = rec
    sys.setprofile(rec)
    try:
        func(notes)
    except SystemExit as exc:
        if exc.code not in (None, 0):
            error = {"type": "SystemExit", "message": str(exc.code)}
    except BaseException as exc:                                # noqa: BLE001
        error = {"type": type(exc).__name__, "message": scrub(str(exc)[:500], pairs)}
    finally:
        sys.setprofile(None)
        _ACTIVE = None
    elapsed = time.perf_counter_ns() - begin
    env_after = environment(device_query=False)
    env = {**env_before, "packages": env_after["packages"], "device": env_after["device"]}
    device = rec.device_summary()
    observed = Counter()
    for d in device:
        observed[d["kind"]] += d["count"]
    doc = {
        "schema": SCHEMA, "id": scenario, "scenario": scenario, "description": description,
        "provenance": {
            "code": code,
            "invocation": {"command": scrub(command, pairs),
                           "argv": scrub(list(sys.argv), pairs),
                           "cwd": scrub(os.getcwd(), pairs)},
            "configuration": scrub(notes.configuration or configuration, pairs),
            "environment": scrub(env, pairs),
            "capture": {"method": "sys.setprofile: Python call/return events in the scenario "
                                  "thread; selected boundary metadata; device events from "
                                  "visible CuPy calls",
                        "boundaries": list(boundaries), "limits": dataclasses.asdict(limits),
                        "phases": rec.phases_seen, "threads": "scenario thread only"},
            "synchronization": {
                "inserted_by_collector": False,
                "device_memory_read_by_collector": False,
                "observed": dict(sorted(observed.items())),
                "observed_is_lower_bound": True,
                "note": notes.synchronization_note or ""},
            "timing": {"instrumented_wall_ns": elapsed,
                       "uninstrumented_wall_ns": baseline_ns,
                       "uninstrumented_cold_wall_ns": cold_ns,
                       "overhead_protocol": ("cold uninstrumented run, warm uninstrumented "
                                             "run, traced run, in one process; ratio = "
                                             "traced / warm" if baseline_ns else None),
                       "overhead_ratio": (round(elapsed / baseline_ns, 3)
                                          if baseline_ns else None),
                       "overhead_measured": baseline_ns is not None,
                       "gpu_execution_time": "not measured by this capture"},
        },
        "semantics": {
            "wall_ns_inclusive": "host wall time from call to return, including callees, "
                                 "tracing overhead and asynchronous GPU launches; recursive "
                                 "calls count their time at every level",
            "gpu_execution_time": "NOT MEASURED; use a synchronized profile or Nsight in a "
                                  "separate run",
            "unobserved": "a function absent from this capture was not called in THIS "
                          "scenario; it is not evidence of unused code",
            "units": "none inferred; only explicit source contracts are shown by the atlas",
        },
        "summary": {"calls_seen": rec.calls_seen, "observed_edges": len(rec.edges),
                    "functions": len(rec.functions), "boundaries": len(rec.boundaries),
                    "dropped_edges": rec.dropped_edges,
                    "dropped_boundaries": rec.dropped_boundaries},
        "edges": rec.to_edges(),
        "functions": {k: dict(v) for k, v in sorted(rec.functions.items())},
        "boundaries": rec.boundaries,
        "device_events": device,
        "verification": scrub(notes.verification, pairs),
        "attachments": scrub(notes.attachments, pairs),
        "error": error,
    }
    return _bound(doc, limits.max_json_bytes)


def _bound(doc: dict, cap: int) -> dict:
    """Shrink a document past ``cap`` bytes: boundaries first, then caller lines."""
    def size():
        return len(json.dumps(doc, default=str))
    removed = 0
    while size() > cap and doc["boundaries"]:
        doc["boundaries"].pop()
        removed += 1
    if size() > cap:
        for e in doc["edges"]:
            e["lines"] = []
        doc["summary"]["lines_dropped_for_size"] = True
    doc["summary"]["boundaries_dropped_for_size"] = removed
    return doc


def write(doc: dict, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, sort_keys=True, indent=1, default=str) + "\n",
                   encoding="utf-8", newline="\n")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m tools.architecture.runtime_capture",
                                description=__doc__.split("\n\n")[0])
    p.add_argument("--scenario",
                   help="identifier; a registered name (see --list) runs its driver")
    p.add_argument("--list", action="store_true", help="list registered scenarios and exit")
    p.add_argument("--out", type=Path)
    p.add_argument("--boundary", action="append", default=[],
                   help="fnmatch pattern of functions whose data is described "
                        "(repeatable; added to a registered scenario's own)")
    p.add_argument("--config", type=Path, help="input file whose SHA-256 is recorded")
    p.add_argument("--scratch", type=Path, help="scratch directory for scenario outputs")
    p.add_argument("--measure-overhead", action="store_true",
                   help="run the scenario once uninstrumented first and record both times")
    p.add_argument("--max-boundaries", type=int, default=Limits.max_boundaries)
    p.add_argument("--max-boundary-examples", type=int, default=Limits.max_boundary_examples,
                   help="boundary records kept per (function, phase)")
    p.add_argument("--fingerprint-bytes", type=int, default=Limits.fingerprint_bytes)
    p.add_argument("--max-edges", type=int, default=Limits.max_edges)
    group = p.add_mutually_exclusive_group()
    group.add_argument("--module")
    group.add_argument("--script")
    p.add_argument("arguments", nargs=argparse.REMAINDER)
    args = p.parse_args(argv)
    from . import scenarios as registry
    if args.list:
        for name, s in sorted(registry.SCENARIOS.items()):
            print(f"{name:32s} {'GPU' if s.needs_gpu else 'CPU'}  {s.description}")
        return 0
    if args.scenario is None or args.out is None:
        p.error("--scenario and --out are required")
    if min(args.max_boundaries, args.fingerprint_bytes, args.max_edges) < 0:
        p.error("limits must not be negative")
    limits = Limits(max_boundaries=args.max_boundaries,
                    max_boundary_examples=args.max_boundary_examples,
                    fingerprint_bytes=args.fingerprint_bytes, max_edges=args.max_edges)
    passed = list(args.arguments)
    if passed and passed[0] == "--":
        passed.pop(0)
    configuration = None
    if args.config:
        configuration = {"file": str(args.config), "hash_method": "SHA-256 of the file bytes",
                         "sha256": hashlib.sha256(args.config.read_bytes()).hexdigest()}
    command = "python -m tools.architecture.runtime_capture " + " ".join(
        a if " " not in a else f'"{a}"' for a in (argv if argv is not None else sys.argv[1:]))
    if args.module or args.script:
        target = args.module or args.script

        def invoke(_notes):
            original = sys.argv
            sys.argv = [target, *passed]
            try:
                if args.module:
                    runpy.run_module(args.module, run_name="__main__", alter_sys=True)
                else:
                    runpy.run_path(args.script, run_name="__main__")
            finally:
                sys.argv = original
        doc = capture(args.scenario, invoke, command=command, boundaries=args.boundary,
                      limits=limits, configuration=configuration,
                      description=f"command-line target {target} {' '.join(passed)}",
                      measure_overhead=None)
    else:
        if args.scenario not in registry.SCENARIOS:
            p.error(f"unknown scenario {args.scenario!r}; give --module/--script or use "
                    f"one of: {', '.join(sorted(registry.SCENARIOS))}")
        s = registry.SCENARIOS[args.scenario]
        scratch = args.scratch or Path(registry.default_scratch())
        drive = s.prepare(scratch)
        doc = capture(args.scenario, drive, command=command,
                      boundaries=[*s.boundaries, *args.boundary], limits=limits,
                      configuration=configuration, description=s.description,
                      measure_overhead=(lambda n: s.prepare(scratch / "uninstrumented")(n))
                      if args.measure_overhead else None)
    write(doc, args.out)
    print(f"Recorded {len(doc['edges'])} observed call relations, "
          f"{len(doc['boundaries'])} boundary records to {args.out}; "
          "GPU execution time was not measured.")
    if doc["error"]:
        print(f"Scenario failed: {doc['error']}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    # ``python -m`` runs this file as ``__main__``; scenario drivers import
    # ``tools.architecture.runtime_capture``. Run the canonical module so that
    # both see the same active recorder (phase labels would otherwise be lost).
    from tools.architecture import runtime_capture as _canonical
    sys.exit(_canonical.main())
