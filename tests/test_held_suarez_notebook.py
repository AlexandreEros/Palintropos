"""The Held–Suarez Colab launcher (plan S6).

CPU (CI): structure — the pinned SOLVER_COMMIT is a full commit hash that
exists in this repository and contains the solver; no stored outputs, no
credentials or tokens; every numerical action goes through
``python -m tropoi.run.held_suarez``; no automatic resolution fallback.

GPU: the notebook's smoke path — its OWN code cells, executed in order in
one namespace with ``HS_NOTEBOOK_LOCAL_SMOKE=1`` (no Drive, no clone) — runs
the GPU/float64 check, the 200-step benchmark gate, a forced stop at day 1
and a resume in a fresh process (no second RK4 startup), Drive-style
backups, and the report, all through the same entrypoint the Colab run uses.
"""
from __future__ import annotations

import json
import os
import pathlib
import re
import subprocess

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
NB = REPO / "notebooks" / "held_suarez_colab.ipynb"


def _cells():
    nb = json.loads(NB.read_text(encoding="utf-8"))
    return nb, [("".join(c["source"]), c) for c in nb["cells"]]


def _has_cuda():
    try:
        import cupy as cp
        return cp.is_available()
    except Exception:
        return False


def test_notebook_structure_pin_and_hygiene():
    nb, cells = _cells()
    code = [s for s, c in cells if c["cell_type"] == "code"]
    text = "\n".join(s for s, _ in cells)
    assert all(not c.get("outputs") and c.get("execution_count") is None
               for _, c in cells if c["cell_type"] == "code"), "notebook must be saved without outputs"
    m = re.search(r'SOLVER_COMMIT = "([0-9a-f]{40})"', text)
    assert m, "SOLVER_COMMIT must be pinned to a full 40-hex commit"
    pin = m.group(1)
    subprocess.run(["git", "-C", str(REPO), "cat-file", "-e", f"{pin}^{{commit}}"], check=True)
    tree = subprocess.check_output(["git", "-C", str(REPO), "ls-tree", "-r", "--name-only", pin],
                                   text=True).split()
    for f in ("src/tropoi/run/held_suarez/__main__.py", "src/tropoi/run/held_suarez/experiment.py",
              "src/tropoi/temporal/semi_implicit.py", "src/tropoi/temporal/tendencies/held_suarez.py"):
        assert f in tree, f"pinned commit {pin} lacks {f}"
    anc = subprocess.run(["git", "-C", str(REPO), "merge-base", "--is-ancestor", pin, "HEAD"])
    assert anc.returncode == 0, "SOLVER_COMMIT must be an ancestor of HEAD"
    # no credentials, tokens, or personal paths
    for pat in (r"ghp_[A-Za-z0-9]{20,}", r"github_pat_", r"(?i)api[_-]?key\s*=", r"(?i)password",
                r"(?i)token\s*=\s*['\"]", r"@gmail\.com", r"C:\\\\Users"):
        assert not re.search(pat, text), pat
    # one entrypoint; the essential stages are present
    joined = "\n".join(code)
    assert '"-m", "tropoi.run.held_suarez"' in joined
    for word in ('["benchmark"]', '["run"]', '["report"', '["verify"', "drive.mount",
                 "float64", "BENCHMARK_STEPS = 200", '"--backup-every-days", "10"', "DEADLINE",
                 "SystemExit(\"DECISION NEEDED"):
        assert word in joined, word
    # production preset is T42 L20 and nothing lowers it
    assert 'PRESET = "production"' in joined
    for bad in ("l_max=21", "--l-max", "T21\"", "fallback"):
        assert bad not in joined.replace('"smoke"', ""), bad


@pytest.mark.skipif(not _has_cuda(), reason="CUDA/CuPy not available")
def test_notebook_smoke_path_executes_through_the_real_entrypoint(tmp_path, monkeypatch):
    _, cells = _cells()
    monkeypatch.setenv("HS_NOTEBOOK_LOCAL_SMOKE", "1")
    monkeypatch.setenv("HS_NOTEBOOK_WORK", str(tmp_path))
    monkeypatch.setenv("HS_NOTEBOOK_SRC", str(REPO))
    ns: dict = {"__name__": "__main__"}
    for i, (src, c) in enumerate(cells):
        if c["cell_type"] != "code":
            continue
        exec(compile(src, f"<notebook cell {i}>", "exec"), ns)
    run_dir = ns["RUN_DIR"]
    events = [json.loads(x)["event"] for x in (run_dir / "events.jsonl").read_text().splitlines()]
    print("\nsmoke events:", events)
    assert events.count("created") == 1 and events.count("resumed") == 1
    assert events[-1] == "completed"
    bench = json.loads((ns["BENCH_DIR"] / "benchmark-hs-T21L10-smoke.json").read_text())
    print(f"benchmark gate: {bench['t_step_mean_s']:.4f} s/step over {bench['steps_timed']} steps, "
          f"{bench['n_samples']} statistics samples, checkpoint {bench['t_checkpoint_s']:.3f} s, "
          f"projected {bench['projected_hours'] * 3600:.1f} s -> {bench['verdict']}")
    assert bench["steps_timed"] == 200 and bench["verdict"] == "within_budget"
    backups = sorted((ns["BACKUP_DIR"] / "checkpoints").glob("checkpoint-s*.npz"))
    assert backups and (ns["BACKUP_DIR"] / "config.json").exists()
    rep = json.loads((ns["REPORT_DIR"] / "report.json").read_text())
    verdicts = {t["tier"]: t["verdict"] for t in rep["tiers"]}
    print("smoke report tiers:", verdicts)
    assert verdicts["B"] == "INCOMPLETE" or verdicts["B"] == "PASS"
    assert verdicts["A"] == "INCOMPLETE"
    # Fresh-VM resume: the local run directory is gone; the restore cell must
    # bring back the newest Drive backup that verifies, skipping a corrupt one.
    import shutil
    backup_ckpts = sorted((ns["BACKUP_DIR"] / "checkpoints").glob("checkpoint-s*.npz"))
    assert len(backup_ckpts) >= 2
    assert (ns["BACKUP_DIR"] / "states" / "state-d00001.npy").exists()
    raw = bytearray(backup_ckpts[-1].read_bytes())
    raw[len(raw) // 2] ^= 0xFF
    backup_ckpts[-1].write_bytes(bytes(raw))
    shutil.rmtree(run_dir)
    restore = next(src for src, c in cells if src.startswith("# ---- 6. restore"))
    exec(compile(restore, "<notebook restore cell>", "exec"), ns)
    restored = sorted((run_dir / "checkpoints").glob("checkpoint-s*.npz"))
    print("restored after corrupting the newest backup:", [p.name for p in restored])
    assert [p.name for p in restored] == [backup_ckpts[-2].name]
