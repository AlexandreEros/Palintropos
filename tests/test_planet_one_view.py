import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tools.browser_cdp import OfflinePage, find_browser
from tools.planet_one.build_view import build
from tropoi.run.airless.config import mercury_vpw99
from tropoi.run.airless.experiment import build_columns, build_substrate


def _viewer(tmp_path, mark, *, cycle_extent):
    cfg = mercury_vpw99(columns={"kind": "latlon", "nlat": 2, "nlon": 4})
    data = cfg.to_dict()
    data["provenance"] = {mark: mark}
    retained = tmp_path / "retained"
    retained.mkdir()
    (tmp_path / "config.json").write_text(json.dumps({
        "config": data, "config_sha256": "abc", "code": {"git_commit": "def"},
    }), encoding="utf-8")
    (retained / "status.json").write_text(json.dumps({
        "cycle": 0, "status": "PERIODIC", "label": "PERIODIC",
    }), encoding="utf-8")
    cycles = {"extent_K": [[60.0, 700.0], [80.0, 600.0]], "dt_s": cfg.cycle_s / 2,
              "max_energy_imbalance_W_m2": 0.0, "max_cell_residual_ratio": 0.0,
              "max_newton_iterations": 1}
    if cycle_extent:
        cycles["extent_scope"] = "cycle"
    (tmp_path / "cycles.json").write_text(json.dumps([cycles]), encoding="utf-8")
    lat, lon, weights = build_columns(cfg)
    sub = build_substrate(cfg)
    np.savez(tmp_path / "grid.npz", lat_deg=lat, lon_deg=lon, weights=weights,
             centres=sub.centres, interfaces=sub.interfaces)
    (retained / "blocks.json").write_text('{"ranges":[[0,8]]}', encoding="utf-8")
    np.save(retained / "surface-b000.npy", np.full((2, 5, 8), 300.0))
    np.save(retained / "profiles-b000.npy", np.full((2, sub.layers, 8), 300.0))
    np.savez(retained / "integrals.npz", absorbed=np.zeros(8), emitted=np.zeros(8))
    extras = {"l3": {"summary": mark}, "gp": {"pass": False, "failures": [mark]}}
    return build(tmp_path, 2, extras)


def test_viewer_embeds_json_without_literal_script_end_tag(tmp_path):
    marker = "</ScRiPt> & <b>metadata</b>"
    html = _viewer(tmp_path, marker, cycle_extent=True)
    payload = html.split("const D = ", 1)[1].split(";\n", 1)[0]
    assert "<" not in payload
    assert json.loads(payload)["extras"]["l3"]["summary"] == marker


@pytest.mark.parametrize("cycle_extent", [True, False])
def test_viewer_renders_evidence_as_text_and_labels_legacy_extent(tmp_path, cycle_extent):
    browser = find_browser()
    if browser is None:
        pytest.skip("needs Chrome, Chromium or Edge")
    marker = '<b data-run-mark="1">metadata</b>'
    html = _viewer(tmp_path, marker, cycle_extent=cycle_extent)
    path = tmp_path / "view.html"
    path.write_text(html, encoding="utf-8")
    with OfflinePage(browser) as page:
        url = page.open(path)
        page.wait_for("document.querySelectorAll('#evidence td').length > 0")
        assert page.eval("document.querySelector('#evidence').textContent").count(marker) == 4
        assert page.eval("document.querySelectorAll('[data-run-mark]').length") == 0
        extent_note = page.eval("document.querySelector('#chips').textContent")
        assert ("CYCLE EXTENT UNVERIFIED" in extent_note) is not cycle_extent
        assert page.requests == [url]
        assert page.console_errors == [] and page.exceptions == []


def test_viewer_omits_old_verdict_when_heat1d_periodicity_is_unverified():
    from tools.planet_one.build_view import _summaries

    bad = {"heat1d_cycles": 40, "heat1d_last_cycle_change_K": 0.0, "max_abs_dTs_K": 0.2}
    gi = {"levels": [{"level": 0, "columns": {"0N 0E": bad}}],
          "verdict": {"0N 0E": {"strict": {"pass": True, "finest_max_K": 0.2, "bound_K": 1}}}}
    summary = _summaries({"gi": gi})
    assert "gi_verdict" not in summary
    assert summary["gi_unverified"]


def test_viewer_script_starts_with_only_src_on_pythonpath():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "tools/planet_one/build_view.py", "--help"], cwd=root,
        env={**os.environ, "PYTHONPATH": str(root / "src")},
        capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert "--l3" in result.stdout
