"""``tropoi plot``: argument handling and the import-light listing.

CPU only; rendering itself is covered in tests/test_saved_run_overview.py.
"""
from __future__ import annotations

import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
CANONICAL = (ROOT / "docs" / "validation" / "williamson_5" / "capsules" /
             "t63" / "20260730T011700Z_williamson5_rot23p93h_r4_l63_dt120h_"
             "45406d82_668e6c9a")


def test_cli_lists_quantities_without_cuda():
    code = ("import sys; from tropoi.cli.main import main; "
            "rc = main(['plot', sys.argv[1], '--list-quantities']); "
            "print('CUPY', int('cupy' in sys.modules)); sys.exit(rc)")
    output = subprocess.run([sys.executable, "-c", code, str(CANONICAL)],
                            capture_output=True, text=True, check=True)
    assert "free_surface_height" in output.stdout
    assert "per-step columns" in output.stdout
    assert "CUPY 0" in output.stdout


def test_cli_rejects_contradictory_selections():
    from tropoi.cli.main import main
    with pytest.raises(SystemExit):
        main(["plot", str(CANONICAL), "--at", "-1", "--snapshots", "0"])
    with pytest.raises(SystemExit):
        main(["plot", str(CANONICAL), "--level", "1", "--sigma", "0.5"])


def test_plot_without_a_run_is_an_error(tmp_path, capsys):
    from tropoi.cli.main import main
    assert main(["plot", str(tmp_path)]) == 2
    assert "no run found" in capsys.readouterr().err


def test_cli_options_build_the_described_views():
    from tropoi.cli.main import build_parser, _plot_view
    from tropoi.representation.archive import open_simulation
    from tropoi.representation.visual.views import (Arrows, Map, Overview,
                                                    Streamlines)
    storage = open_simulation(CANONICAL).storage
    parser = build_parser()

    view, at = _plot_view(parser.parse_args(["plot", str(CANONICAL)]),
                          storage)
    assert isinstance(view, Overview) and at is None
    assert view.map.background == "free_surface_height"

    view, at = _plot_view(parser.parse_args(
        ["plot", str(CANONICAL), "--at", "-1", "--map", "none"]), storage)
    assert at == -1
    assert view == Map(None, vectors=Streamlines(),
                       contours=view.contours)

    view, at = _plot_view(parser.parse_args(
        ["plot", str(CANONICAL), "--map", "wind_speed", "--vectors",
         "arrows", "--snapshots", "0,5d", "--no-static",
         "--contours", "terrain:1000"]), storage)
    assert view.snapshots == (0, "5d") and view.static is None
    assert view.map.background == "wind_speed"
    assert isinstance(view.map.vectors, Arrows)
    assert view.map.contours[0].levels == (1000.0,)
