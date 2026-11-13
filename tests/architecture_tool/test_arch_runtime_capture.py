"""runtime_capture: provenance, phases, boundary data, mutations, safety bounds.

CPU only. A synthetic package (``archrt``) is imported from a temporary
directory and traced; ``tropoi`` is never imported. Device arrays are
represented by a fake class whose data accessors all raise, so describing
one proves the collector reads host-side metadata only.
"""
from __future__ import annotations

import importlib
import json
import sys

import numpy as np
import pytest

from tools.architecture import runtime_capture as rt

PKG = {
    "archrt/__init__.py": "",
    "archrt/model.py": """
        import numpy as np


        class Stepper:
            def __init__(self, n):
                self.step_count = 0
                self.small = np.zeros(4)
                self.large = np.zeros(10000)
                self.buffer = np.zeros(4)
                self.config = {"api_key": "s3cr3t", "nlat": n}

            def step(self):
                self.step_count += 1
                self.small += 1.0            # in place, fingerprinted
                self.large += 1.0            # in place, too large to fingerprint
                self.buffer = self.buffer + 1.0   # rebound
                return self.step_count


        def advance(stepper, n):
            for _ in range(n):
                helper(stepper)
            return stepper.step_count


        def helper(stepper):
            return stepper.step()


        def fail():
            raise RuntimeError("boom")
    """,
    "cupyfake/__init__.py": """
        def asnumpy(x):
            return x
    """,
}


@pytest.fixture(scope="module")
def archrt(tmp_path_factory):
    from conftest import write_tree
    root = write_tree(tmp_path_factory.mktemp("rt"), PKG)
    sys.path.insert(0, str(root))
    try:
        yield importlib.import_module("archrt.model")
    finally:
        sys.path.remove(str(root))
        for name in [m for m in sys.modules if m.startswith(("archrt", "cupyfake"))]:
            del sys.modules[name]


def _capture(fn, **kw):
    return rt.capture("t", fn, command="test", package="archrt", device_query=False, **kw)


def test_edges_phases_counts_and_caller_lines(archrt):
    def drive(notes):
        s = archrt.Stepper(8)
        with rt.phase("startup"):
            archrt.advance(s, 1)
        with rt.phase("stepping"):
            archrt.advance(s, 3)
    doc = _capture(drive)
    assert doc["schema"] == rt.SCHEMA and doc["error"] is None
    edges = {(e["source"], e["target"]): e for e in doc["edges"]}
    e = edges[("archrt.model.advance", "archrt.model.helper")]
    assert e["count"] == 4 and e["phases"] == {"startup": 1, "stepping": 3}
    assert e["lines"] and e["lines"][0][1] == "direct"
    assert ("archrt.model.helper", "archrt.model.Stepper.step") in edges
    assert doc["functions"]["archrt.model.Stepper.step"] == {"startup": 1, "stepping": 3}
    assert doc["provenance"]["capture"]["phases"] == ["default", "startup", "stepping"]
    sync = doc["provenance"]["synchronization"]
    assert sync["inserted_by_collector"] is False
    assert sync["device_memory_read_by_collector"] is False
    assert doc["semantics"]["gpu_execution_time"].startswith("NOT MEASURED")


def test_boundary_records_arguments_return_and_mutations(archrt):
    def drive(notes):
        s = archrt.Stepper(8)
        with rt.phase("stepping"):
            archrt.advance(s, 2)
    doc = _capture(drive, boundaries=["archrt.model.Stepper.step"])
    (b,) = doc["boundaries"]                # one example per (function, phase)
    assert b["function"] == "archrt.model.Stepper.step" and b["phase"] == "stepping"
    assert b["return"] == {"type": "int", "value": 1}
    m = b["mutations"]
    assert m["self.step_count"]["kind"] == "value_changed"
    assert m["self.small"]["kind"] == "content_changed"
    assert m["self.buffer"]["kind"] == "rebound"
    assert "self.large" in b["unverified"] and "self.large" not in m
    small = b["self_before"]["fields"]["small"]
    assert small["shape"] == [4] and small["location"] == "CPU" and "fingerprint" in small
    assert "fingerprint" not in b["self_before"]["fields"]["large"]


def test_secrets_are_redacted_and_paths_scrubbed(archrt, tmp_path):
    d = rt.Describer(rt.Limits(), rt._scrub_pairs(tmp_path))
    out = d({"api_key": "s3cr3t", "token_file": "x", "path": str(tmp_path / "run"),
             "nlat": 8})
    assert out["fields"]["api_key"]["value"] == "<redacted>"
    assert out["fields"]["token_file"]["redacted"]
    assert out["fields"]["path"]["value"] == "<repo>" + ("\\run" if "\\" in str(tmp_path)
                                                         else "/run")
    long = d("x" * 500)
    assert long["truncated"] and len(long["value"]) == rt.Limits().max_str


class _Ptr:
    ptr = 1234


class _Device:
    id = 0


class FakeDeviceArray:
    """Host metadata only; every data accessor raises."""

    shape = (3, 5)
    dtype = "complex128"
    nbytes = 240
    data = _Ptr()
    device = _Device()

    @property
    def __cuda_array_interface__(self):          # reading it could synchronize
        raise AssertionError("collector read __cuda_array_interface__")

    def get(self):
        raise AssertionError("collector copied device memory")

    def __array__(self, *a, **k):
        raise AssertionError("collector converted to NumPy")

    def tobytes(self):
        raise AssertionError("collector read device bytes")

    def item(self):
        raise AssertionError("collector read a device scalar")


def test_device_arrays_are_described_from_host_metadata_only():
    out = rt.Describer(rt.Limits(), [])(FakeDeviceArray())
    assert out["location"] == "GPU" and out["shape"] == [3, 5]
    assert out["dtype"] == "complex128" and out["bytes"] == 240
    assert out["ptr"] == 1234 and out["device"] == 0 and "fingerprint" not in out


def test_cupy_arrays_never_synchronize_when_described():
    cp = pytest.importorskip("cupy")
    try:
        x = cp.zeros((4, 4))
    except Exception:                                   # noqa: BLE001
        pytest.skip("no usable CUDA device")
    seen = []

    def watch(frame, event, arg):
        if event == "c_call":
            owner = getattr(arg, "__self__", None)
            module = (owner.__name__ if isinstance(owner, type(sys)) else
                      type(owner).__module__ if owner is not None else
                      getattr(arg, "__module__", "")) or ""
            if module.startswith("cupy"):
                seen.append(getattr(arg, "__name__", ""))
        elif event == "call" and frame.f_globals.get("__name__", "").startswith("cupy"):
            seen.append(frame.f_code.co_qualname)
    describe = rt.Describer(rt.Limits(), [])
    sys.setprofile(watch)
    try:
        out = describe(x)
    finally:
        sys.setprofile(None)
    assert out["location"] == "GPU" and out["shape"] == [4, 4] and "fingerprint" not in out
    forbidden = {"synchronize", "deviceSynchronize", "get", "item", "tolist", "asnumpy",
                 "__array__", "tobytes"}
    assert not forbidden & set(seen), sorted(forbidden & set(seen))


def test_visible_device_transfers_are_counted(archrt):
    import cupyfake

    def drive(notes):
        with rt.phase("io"):
            archrt.helper(type("S", (), {"step": staticmethod(lambda: cupyfake.asnumpy(1))})())
    doc = _capture(drive)
    assert doc["device_events"] == [{"phase": "io", "caller": "archrt.model.helper",
                                     "kind": "device_to_host", "count": 1}]
    assert doc["provenance"]["synchronization"]["observed"] == {"device_to_host": 1}


def test_failures_are_recorded_not_raised(archrt):
    doc = _capture(lambda notes: archrt.fail())
    assert doc["error"]["type"] == "RuntimeError"
    ok = _capture(lambda notes: (_ for _ in ()).throw(SystemExit(0)))
    assert ok["error"] is None


def test_provenance_identifies_revision_and_overhead(archrt):
    runs = []
    doc = _capture(lambda notes: runs.append(1),
                   measure_overhead=lambda notes: runs.append(0))
    code = doc["provenance"]["code"]
    assert {"commit", "source_dirty", "tools_dirty", "source_sha256"} <= set(code)
    timing = doc["provenance"]["timing"]
    assert runs == [0, 0, 1]                    # cold, warm baseline, traced
    assert timing["overhead_measured"] and timing["uninstrumented_cold_wall_ns"] is not None
    assert doc["provenance"]["capture"]["limits"]["max_boundary_examples"] == 1


def test_documents_are_bounded(archrt):
    def drive(notes):
        s = archrt.Stepper(4)
        for i in range(30):
            with rt.phase(f"p{i}"):
                s.step()
    doc = _capture(drive, boundaries=["archrt.*"],
                   limits=rt.Limits(max_json_bytes=12000))
    assert len(json.dumps(doc)) <= 12000 or not doc["boundaries"]
    assert doc["summary"]["boundaries_dropped_for_size"] > 0
    capped = _capture(drive, boundaries=["archrt.*"], limits=rt.Limits(max_boundaries=3))
    assert len(capped["boundaries"]) == 3 and capped["summary"]["dropped_boundaries"] > 0


def test_registered_scenarios_are_listed_without_importing_tropoi(capsys):
    def ours():
        return {m for m in sys.modules if m == "tropoi" or m.startswith("tropoi.")}
    before = ours()
    assert rt.main(["--list"]) == 0
    out = capsys.readouterr().out
    for name in ("setup-cpu", "radiative-tiny-lifecycle", "saved-run-cpu", "saved-run-view"):
        assert name in out
    # The generator refuses to run where tropoi is imported; listing must not import it.
    assert ours() == before
