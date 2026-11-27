"""GPU-first execution policy (decisions D6/U6): CUDA by default, CPU only
when requested explicitly, conflicts refused, no fallback, and provenance
that records what actually ran without initializing CUDA for a CPU run."""
import json
import os
import pathlib
import subprocess
import sys

import pytest

from tropoi.run import execution
from tropoi.run.execution import (CPU_VERIFICATION, ENVIRONMENT_VARIABLE, ExecutionError,
                                  execution_identity, resolve_backend)

ROOT = pathlib.Path(__file__).resolve().parents[1]
CERES = ROOT / "tests" / "fixtures" / "airless" / "ceres-equatorial-column.airless.json"
CPU = {ENVIRONMENT_VARIABLE: CPU_VERIFICATION}


@pytest.fixture
def no_cuda(monkeypatch):
    """CuPy unusable, as on a machine without a GPU."""
    def unusable():
        raise ExecutionError("CUDA execution is required, but CuPy/CUDA is unusable here (test). "
                             + execution.HOW_TO_VERIFY_ON_CPU)
    monkeypatch.setattr(execution, "require_cuda", unusable)


def test_cuda_is_the_default_and_its_absence_is_an_error_not_a_fallback(no_cuda):
    with pytest.raises(ExecutionError, match="TROPOI_EXECUTION=cpu-verification"):
        resolve_backend(environ={})
    with pytest.raises(ExecutionError, match="--device cpu"):
        resolve_backend(device="cuda", environ={})


def test_cpu_is_selected_only_explicitly(no_cuda):
    assert resolve_backend(device="cpu", environ={}) == "numpy"
    assert resolve_backend(backend="numpy", environ={}) == "numpy"
    assert resolve_backend(environ=CPU) == "numpy"
    assert resolve_backend(device="cpu", environ=CPU) == "numpy"


@pytest.mark.parametrize("kw", [{"device": "cuda"}, {"backend": "cupy"}])
def test_an_explicit_cuda_request_conflicts_with_cpu_verification(kw):
    with pytest.raises(ExecutionError, match="unset one"):
        resolve_backend(environ=CPU, **kw)


@pytest.mark.parametrize("kw, message", [
    ({"device": "gpu"}, "device must be one of"),
    ({"backend": "cpu"}, "backend must be one of"),
    ({"device": "cpu", "backend": "numpy"}, "not both"),
])
def test_unknown_or_double_requests_are_refused(kw, message):
    with pytest.raises(ExecutionError, match=message):
        resolve_backend(environ={}, **kw)


def test_an_unrecognized_environment_value_is_refused():
    with pytest.raises(ExecutionError, match="not recognized"):
        resolve_backend(environ={ENVIRONMENT_VARIABLE: "cpu"})


def test_cpu_identity_records_the_cpu_and_precision_without_cupy():
    env = execution_identity("numpy")
    assert env["backend"] == "numpy" and env["device"] == "cpu"
    assert env["precision"] == "float64" and env["cpu"]
    assert "cupy" not in env and "cuda_runtime" not in env


def _python(code: str, env: dict) -> subprocess.CompletedProcess:
    path = os.pathsep.join([str(ROOT / "src"), str(ROOT), os.environ.get("PYTHONPATH", "")])
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          cwd=ROOT, env={**os.environ, "PYTHONPATH": path, **env})


def test_a_cpu_verification_run_never_imports_cupy_and_records_the_cpu(tmp_path):
    """The airless command under TROPOI_EXECUTION=cpu-verification: the run
    completes, CuPy is never imported (so CUDA is never initialized), and
    config.json records the CPU execution."""
    out = tmp_path / "run"
    code = (
        "import sys, json\n"
        "from tropoi.run.airless.config import load_config_document\n"
        "cfg, _ = load_config_document(sys.argv[1] if len(sys.argv) > 1 else %r)\n"
        "cfg = cfg.with_(cycle_steps=24, spinup_cycle_steps=None,\n"
        "                output={**cfg.output, 'surface_samples': 12, 'profile_samples': 12})\n"
        "from tropoi.run.airless.experiment import AirlessRun\n"
        "AirlessRun(cfg, %r, allow_dirty=True, log=lambda s: None).run_transient(4)\n"
        "assert 'cupy' not in sys.modules, 'CuPy was imported by a CPU run'\n"
        % (str(CERES), str(out)))
    r = _python(code, CPU)
    assert r.returncode == 0, r.stdout + r.stderr
    meta = json.loads((out / "config.json").read_text())
    assert meta["backend"] == "numpy"
    env = meta["environment"]
    assert env["backend"] == "numpy" and env["device"] == "cpu" and env["precision"] == "float64"
    assert "cupy" not in env


def test_the_command_hides_the_cpu_option_and_refuses_without_cuda(tmp_path):
    help_text = _python("from tropoi.run.airless.__main__ import main; main(['--help'])", {})
    assert help_text.returncode == 0
    assert "--device" not in help_text.stdout and "--backend" not in help_text.stdout
    # no CUDA (CuPy blocked), no CPU request: a prompt error, no run directory
    out = tmp_path / "run"
    r = _python("import sys; sys.modules['cupy'] = None\n"
                "from tropoi.run.airless.__main__ import main\n"
                f"sys.exit(main(['run', '--config', {str(CERES)!r}, '--steps', '2', "
                f"'--out', {str(out)!r}, '--allow-dirty']))", {ENVIRONMENT_VARIABLE: ""})
    assert r.returncode == 2, r.stdout + r.stderr
    assert "CUDA execution is required" in r.stderr and "--device cpu" in r.stderr
    assert not out.exists()
