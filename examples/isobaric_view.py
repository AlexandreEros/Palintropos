r"""Isobaric surfaces of a saved PE run across its saved times (CPU only).

    python examples/isobaric_view.py view RUN --pressure 300 [--out FILE]
        [--resolution 3] [--max-mb 32] [--no-streamlines] [--field F] [--open]
    python examples/isobaric_view.py interactive RUN --pressure 300 [--out FILE]
        [--times all | 0 4 -1] [--resolution 3] [--max-mb 32] [--open]
    python examples/isobaric_view.py measure RUN --pressure 300 [--times 0 -1]
        [--resolution 2] [--out results.json]
    python examples/isobaric_view.py capture PAGE OUT_DIR [--centre LAT LON]
    python examples/isobaric_view.py bench PAGE OUT_DIR [--pressures 300 150 ...]

RUN is a ``tropoi run pe`` capsule or a Held–Suarez experiment directory.
Issue #13, first and second slices: ``view`` and ``interactive`` are the
documented API, nothing more:

    from tropoi.representation.archive import open_simulation

    sim = open_simulation(RUN)
    sim.view(out, surface=("pressure", 300.0), times="all", open_browser=False)
    sim.view(out2, surface=("pressure", 300.0), interactive_pressure=True,
             times="all", open_browser=False)

``view`` writes ONE self-contained HTML file with the run's terrain plus the
requested isobar of EVERY saved state (``times="all"`` never subsamples: it
fails above ``--max-mb`` and says what to change), coloured by temperature,
geopotential height or wind speed, with masks where the isobar is under
ground or outside the sampled full levels. ``interactive`` writes the native
columns of every exported saved state instead, and the page interpolates
them to ANY pressure chosen in the browser (slider or typed value), offline.
Pressure is in pascals: 300 Pa (3 hPa) suits the published Mars run; 50000
Pa (500 hPa) an Earth-like one.

``measure`` times the parts of one export separately (spherical-harmonic
synthesis of the column, hydrostatic coefficients, winds, the vorticity and
divergence syntheses the native page pays and this one skips, vertical
interpolation, streamlines, quantization and serialization, and the column
export of the interactive page) and writes them as JSON. ``capture`` opens a
written page in headless Chrome with networking blocked, checks it and saves
screenshots and a JSON report (tools/browser_cdp). ``bench`` does the same
for an interactive page and measures what the browser pays: start-up,
pressure updates (interpolation apart from mesh and map work), saved-time
switches and JS heap. See docs/ATMOSPHERE_VIEW.md.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import platform
import sys
import time

REPO = pathlib.Path(__file__).resolve().parents[1]


def _storage(sim):
    return getattr(sim, "_storage", None) or sim.storage


def cmd_view(args) -> pathlib.Path:
    from tropoi.representation.archive import open_simulation

    sim = open_simulation(args.run)
    print(f"{sim!r}: saved times {sim.times[0]:g} .. {sim.times[-1]:g} s")
    path = sim.view(
        args.out, open_browser=args.open,
        surface=("pressure", float(args.pressure)), times="all",
        resolution_deg=args.resolution, max_mb=args.max_mb,
        streamlines=not args.no_streamlines, field=args.field,
        opacity=args.opacity, exaggeration=args.exaggeration,
        centre=None if args.centre is None else tuple(args.centre),
        gravity=args.gravity)
    print(f"wrote {path} ({path.stat().st_size / 1e6:.2f} MB)")
    return path


def _times_option(values, count):
    if values is None or (len(values) == 1 and values[0] == "all"):
        return "all"
    return [int(t) for t in values]


def cmd_interactive(args) -> pathlib.Path:
    from tropoi.representation.archive import open_simulation

    sim = open_simulation(args.run)
    print(f"{sim!r}: saved times {sim.times[0]:g} .. {sim.times[-1]:g} s")
    t0 = time.perf_counter()
    path = sim.view(
        args.out, open_browser=args.open,
        surface=("pressure", float(args.pressure)), interactive_pressure=True,
        times=_times_option(args.times, len(sim)),
        resolution_deg=args.resolution, max_mb=args.max_mb, field=args.field,
        opacity=args.opacity, exaggeration=args.exaggeration,
        centre=None if args.centre is None else tuple(args.centre),
        gravity=args.gravity)
    print(f"wrote {path} ({path.stat().st_size / 1e6:.2f} MB in "
          f"{time.perf_counter() - t0:.1f} s)")
    return path


def cmd_measure(args) -> dict:
    """Per-part wall-clock costs of one derived-surface export."""
    import numpy as np

    from tropoi.representation.archive import open_simulation
    from tropoi.representation.diagnostics.pe_levels import (
        LEVEL_FIELDS, level_geopotential_coefficients, reconstruct_levels)
    from tropoi.representation.diagnostics.pe_surfaces import (
        pressure_columns, pressure_surface)
    from tropoi.representation.visual.atmosphere import (
        STREAMLINE_MAX_STEPS, STREAMLINE_MIN_SPEED, STREAMLINE_SEEDS,
        STREAMLINE_STEP_DEG, _mesh, atmosphere_payload, default_resolution_deg,
        quantize_columns)
    from tropoi.representation.visual.sphere_streamlines import (
        integrate_streamlines)
    from tropoi.representation.visual.terrain import write_globe_page
    from tropoi.spatial.transforms.host_synthesis import (
        synthesize_latlon, synthesize_wind_latlon)

    sim = open_simulation(args.run)
    storage = _storage(sim)
    context = storage.pe_context().with_gravity(args.gravity)
    res = float(args.resolution or default_resolution_deg(context.l_max))
    lat, lon = _mesh(res)
    count = len(sim)
    times = [int(t) + (count if int(t) < 0 else 0) for t in args.times]
    nlev, l_max = context.nlev, context.l_max
    p0 = float(args.pressure)
    parts = {k: [] for k in (
        "synthesis_T_phi", "hydrostatic_coefficients", "winds",
        "zeta_delta_syntheses_skipped", "reconstruct_levels_total",
        "vertical_interpolation", "streamlines", "quantize_columns")}
    valid_fraction = []
    for index in times:
        frame = storage.frame(index)
        x = np.asarray(frame)
        temperature = x[2 * nlev:3 * nlev]
        t0 = time.perf_counter()
        phi_lm = level_geopotential_coefficients(frame, context)
        t1 = time.perf_counter()
        for k in range(nlev):
            synthesize_latlon(temperature[k], lat, lon)
            synthesize_latlon(phi_lm[k], lat, lon)
        t2 = time.perf_counter()
        for k in range(nlev):
            synthesize_wind_latlon(x[k], x[nlev + k], context.radius_m, lat,
                                   lon)
        t3 = time.perf_counter()
        for k in range(nlev):
            synthesize_latlon(x[k], lat, lon)
            synthesize_latlon(x[nlev + k], lat, lon)
        t4 = time.perf_counter()
        levels = reconstruct_levels(frame, context, lat, lon, None, winds=True,
                                    vorticity_divergence=False)
        t5 = time.perf_counter()
        s = pressure_surface(frame, context, lat, lon, ("pressure", p0),
                             levels=levels)
        t6 = time.perf_counter()
        integrate_streamlines(s["u"], s["v"], lat, lon,
                              seed_count=STREAMLINE_SEEDS,
                              step_deg=STREAMLINE_STEP_DEG,
                              max_steps=STREAMLINE_MAX_STEPS,
                              min_speed=STREAMLINE_MIN_SPEED, valid=s.valid)
        t7 = time.perf_counter()
        quantize_columns(pressure_columns(frame, context, lat, lon,
                                          levels=levels))
        t8 = time.perf_counter()
        parts["quantize_columns"].append(t8 - t7)
        parts["hydrostatic_coefficients"].append(t1 - t0)
        parts["synthesis_T_phi"].append(t2 - t1)
        parts["winds"].append(t3 - t2)
        parts["zeta_delta_syntheses_skipped"].append(t4 - t3)
        parts["reconstruct_levels_total"].append(t5 - t4)
        parts["vertical_interpolation"].append(t6 - t5)
        parts["streamlines"].append(t7 - t6)
        valid_fraction.append(float(s.valid.mean()))
    # the whole export of the same times, and its serialization alone
    t0 = time.perf_counter()
    page, report = atmosphere_payload(storage, surface=("pressure", p0),
                                      times=times, resolution_deg=res,
                                      gravity=args.gravity)
    t1 = time.perf_counter()
    import tempfile
    scratch = (args.scratch if args.scratch else
               pathlib.Path(args.out).parent if args.out else
               pathlib.Path(tempfile.gettempdir()) / "tropoi")
    out_html = pathlib.Path(scratch) / "measure.html"
    out_html.parent.mkdir(parents=True, exist_ok=True)
    write_globe_page(page, page["title"], out_html)
    t2 = time.perf_counter()
    size = out_html.stat().st_size
    out_html.unlink()
    # the interactive export of the same times: columns instead of a surface
    t3 = time.perf_counter()
    page_i, report_i = atmosphere_payload(
        storage, surface=("pressure", p0), interactive_pressure=True,
        times=times, resolution_deg=res, gravity=args.gravity, max_mb=None)
    t4 = time.perf_counter()
    write_globe_page(page_i, page_i["title"], out_html)
    t5 = time.perf_counter()
    size_i = out_html.stat().st_size
    out_html.unlink()
    nodes = lat.size * lon.size
    per_time = {k: float(np.mean(v)) for k, v in parts.items()}
    result = {
        "run": str(pathlib.Path(args.run)), "label": context.label,
        "l_max": l_max, "nlev": nlev, "resolution_deg": res,
        "mesh_nodes": nodes, "pressure_pa": p0, "times": times,
        "valid_fraction": valid_fraction,
        "seconds_per_time": per_time,
        "seconds_total_payload": (t1 - t0),
        "seconds_total_payload_per_time": (t1 - t0) / len(times),
        "seconds_write_page": (t2 - t1),
        "estimate_bytes": report["estimate_bytes"],
        "file_bytes": size,
        "interactive": {
            "seconds_total_payload": (t4 - t3),
            "seconds_total_payload_per_time": (t4 - t3) / len(times),
            "seconds_write_page": (t5 - t4),
            "estimate_bytes": report_i["estimate_bytes"],
            "file_bytes": size_i,
            "decoded_bytes_per_time": report_i["decoded_bytes_per_time"],
            "pressure_range_pa": list(report_i["pressure_range"]),
        },
        "ratio_interpolation_to_reconstruction": (
            per_time["vertical_interpolation"]
            / per_time["reconstruct_levels_total"]),
        "level_fields_native_page": list(LEVEL_FIELDS),
        "machine": {"python": platform.python_version(),
                    "processor": platform.processor(),
                    "system": platform.platform()},
    }
    text = json.dumps(result, indent=2)
    print(text)
    if args.out:
        pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.out).write_text(text + "\n", encoding="utf-8",
                                          newline="\n")
    return result


def cmd_capture(args) -> dict:
    """Open PAGE offline in headless Chrome, check it, save captures."""
    sys.path.insert(0, str(REPO))
    from tools.browser_cdp import OfflinePage, find_browser

    browser = find_browser()
    if browser is None:
        raise SystemExit("no Chrome, Chromium or Edge found (TROPOI_BROWSER)")
    api = "window.__terrainGlobe.atmosphere"
    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    page_path = pathlib.Path(args.page)
    report = {"page": page_path.name, "frames": []}
    with OfflinePage(browser, width=1400, height=1000) as page:
        t0 = time.perf_counter()
        url = page.open(page_path)
        page.wait_for(f"{api} && document.getElementById('timeLabel')"
                      ".textContent")
        report["seconds_to_interactive"] = time.perf_counter() - t0
        report["glOK"] = page.eval("window.__terrainGlobe.glOK")
        nt = page.eval(f"{api}.info().time ? {api}.info() && "
                       "document.getElementById('timeSlider').max : 0")
        nt = int(nt) + 1
        page.eval(f"{api}.setField({args.field}); {api}.setExaggeration("
                  f"{args.exaggeration}); {api}.setOpacity(0.75)")
        if args.centre:
            page.eval(f"{api}.centreOn({args.centre[0]}, {args.centre[1]}); "
                      f"{api}.setDistance(2.6)")
        digests = []
        for i in range(nt):
            t1 = time.perf_counter()
            page.eval(f"{api}.setTime({i})")
            switch = time.perf_counter() - t1
            info = page.eval(f"{api}.info()")
            counts = page.eval(f"{api}.maskCounts()")
            digests.append(page.eval(f"{api}.digest()"))
            report["frames"].append({
                "time": info["time"], "triangles": counts["triangles"],
                "triangles_all": counts["trianglesAll"],
                "mask_counts": counts["counts"], "lines": info["lines"],
                "range": info["range"], "seconds_switch": switch})
            if i in (0, nt - 1):
                shot = out / f"{page_path.stem}_t{i}.png"
                page.screenshot(shot)
                report.setdefault("captures", []).append(shot.name)
        if args.hover:
            report["hover"] = [dict(lat=la, lon=lo,
                                    **page.eval(f"{api}.sample({la}, {lo})"))
                               for la, lo in args.hover]
        report["distinct_frames"] = len(set(digests))
        # the page's own file is the only request; record it by name, not by
        # the machine's path
        report["requests"] = [r.replace(url, page_path.name)
                              for r in page.requests]
        report["requests_ok"] = page.requests == [url]
        report["console_errors"] = page.console_errors
        report["exceptions"] = page.exceptions
        report["memory_mb"] = page.eval(
            "performance.memory ? performance.memory.usedJSHeapSize / 1e6 "
            ": null")
    text = json.dumps(report, indent=2)
    print(text)
    (out / f"{page_path.stem}_browser_check.json").write_text(
        text + "\n", encoding="utf-8", newline="\n")
    return report


def cmd_bench(args) -> dict:
    """Open an interactive-pressure PAGE offline and measure what the
    browser pays: start-up, each pressure update (the page's own split
    into decode, interpolation, mesh and map), saved-time switches, heap."""
    sys.path.insert(0, str(REPO))
    import statistics

    from tools.browser_cdp import OfflinePage, find_browser

    browser = find_browser()
    if browser is None:
        raise SystemExit("no Chrome, Chromium or Edge found (TROPOI_BROWSER)")
    api = "window.__terrainGlobe.atmosphere"
    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    page_path = pathlib.Path(args.page)
    report = {"page": page_path.name, "file_mb": page_path.stat().st_size
              / 1e6, "pressures": [], "time_switches": []}
    with OfflinePage(browser, width=1400, height=1000) as page:
        t0 = time.perf_counter()
        url = page.open(page_path)
        page.wait_for(f"{api} && document.getElementById('timeLabel')"
                      ".textContent")
        report["seconds_to_interactive"] = time.perf_counter() - t0
        report["glOK"] = page.eval("window.__terrainGlobe.glOK")
        info = page.eval(f"{api}.info()")
        if not info.get("interactive"):
            raise SystemExit(f"{page_path} is not an interactive-pressure "
                             "page")
        report["initial"] = {"pressure_pa": info["pressure_pa"],
                             "time": info["time"], "triangles":
                             info["triangles"]}
        report["memory_mb_after_load"] = page.eval(f"{api}.memory()")
        rng = page.eval("JSON.parse(document.getElementById('terrain-data')"
                        ".textContent).atmosphere.surface.pressure_range")
        report["pressure_range_pa"] = rng
        page.eval(f"{api}.setField({args.field}); {api}.setExaggeration("
                  f"{args.exaggeration}); {api}.setOpacity(0.75)")
        if args.centre:
            page.eval(f"{api}.centreOn({args.centre[0]}, {args.centre[1]}); "
                      f"{api}.setDistance(2.6)")
        pressures = args.pressures or [info["pressure_pa"]]
        nt = int(page.eval("document.getElementById('timeSlider').max")) + 1
        for p0 in pressures:
            for rep_i in range(args.repeat):
                # alternate with another pressure so every update recomputes
                page.eval(f"{api}.setPressure({rng[0] * 1.5!r})")
                t1 = time.perf_counter()
                r = page.eval(f"{api}.setPressure({float(p0)!r})")
                round_trip = time.perf_counter() - t1
                timing = page.eval(f"{api}.timing()")
                counts = page.eval(f"{api}.maskCounts()")
                report["pressures"].append({
                    "pressure_pa": float(p0), "ok": r["ok"],
                    "seconds_round_trip": round_trip,
                    "page_ms": timing, "triangles": counts["triangles"],
                    "mask_counts": counts["counts"]})
            shot = out / f"{page_path.stem}_p{float(p0):g}Pa.png"
            page.screenshot(shot)
            report.setdefault("captures", []).append(shot.name)
        page.eval(f"{api}.setPressure({float(pressures[0])!r})")
        for i in list(range(nt)) + list(range(nt - 1, -1, -1)):
            t1 = time.perf_counter()
            page.eval(f"{api}.setTime({i})")
            switch = time.perf_counter() - t1
            timing = page.eval(f"{api}.timing()")
            report["time_switches"].append({
                "time_index": i, "seconds_round_trip": switch,
                "page_ms": timing,
                "decoded_times": page.eval(f"{api}.decodedTimes()")})
        report["memory_mb_after_use"] = page.eval(f"{api}.memory()")
        # typed-input path once, for the record
        page.eval("document.getElementById('pressureInput').value = "
                  "'abc'; document.getElementById('pressureApply').click()")
        report["bad_input"] = page.eval(f"{api}.pressureControls()")
        report["requests"] = [r.replace(url, page_path.name)
                              for r in page.requests]
        report["requests_ok"] = page.requests == [url]
        report["console_errors"] = page.console_errors
        report["exceptions"] = page.exceptions

    def med(key, rows):
        vals = [row[key] for row in rows]
        return statistics.median(vals) if vals else None

    rows = report["pressures"]
    report["summary"] = {
        "pressure_update_ms_median_page_total": med(
            "t", [{"t": r["page_ms"]["total_ms"]} for r in rows]),
        "pressure_update_ms_median_interpolate": med(
            "t", [{"t": r["page_ms"]["interpolate_ms"]} for r in rows]),
        "pressure_update_ms_median_mesh": med(
            "t", [{"t": r["page_ms"]["mesh_ms"]} for r in rows]),
        "pressure_update_ms_median_map": med(
            "t", [{"t": r["page_ms"]["map_ms"]} for r in rows]),
        "pressure_update_ms_median_round_trip": 1000 * med(
            "seconds_round_trip", rows),
        "time_switch_ms_median_round_trip": 1000 * med(
            "seconds_round_trip", report["time_switches"]),
        "time_switch_ms_median_decode": med(
            "t", [{"t": r["page_ms"]["decode_ms"]}
                  for r in report["time_switches"]]),
        "time_switch_ms_median_page_total": med(
            "t", [{"t": r["page_ms"]["total_ms"]}
                  for r in report["time_switches"]]),
    }
    text = json.dumps(report, indent=2)
    print(text)
    (out / f"{page_path.stem}_bench.json").write_text(
        text + "\n", encoding="utf-8", newline="\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[1],
        formatter_class=argparse.RawTextHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    v = sub.add_parser("view", help="write the isobar page for every saved "
                                    "state")
    v.add_argument("run")
    v.add_argument("--pressure", type=float, required=True,
                   help="surface pressure level in Pa (300 = 3 hPa)")
    v.add_argument("--out", help="output HTML (default: temp directory)")
    v.add_argument("--resolution", type=float,
                   help="display mesh spacing in degrees")
    v.add_argument("--max-mb", type=float, default=32.0)
    v.add_argument("--no-streamlines", action="store_true")
    v.add_argument("--field", default="temperature",
                   choices=("temperature", "geopotential_height",
                            "wind_speed"))
    v.add_argument("--opacity", type=float, default=0.6)
    v.add_argument("--exaggeration", type=float)
    v.add_argument("--centre", type=float, nargs=2, metavar=("LAT", "LON"))
    v.add_argument("--gravity", type=float,
                   help="only for runs that record none (flat presets)")
    v.add_argument("--open", action="store_true")
    v.set_defaults(func=cmd_view)

    i = sub.add_parser("interactive", help="write the interactive-pressure "
                                           "page (native columns)")
    i.add_argument("run")
    i.add_argument("--pressure", type=float, required=True,
                   help="initial isobar in Pa (any pressure in the page)")
    i.add_argument("--times", nargs="+", default=["all"],
                   help="'all' (default) or saved indices")
    i.add_argument("--out", help="output HTML (default: temp directory)")
    i.add_argument("--resolution", type=float)
    i.add_argument("--max-mb", type=float, default=32.0)
    i.add_argument("--field", default="temperature",
                   choices=("temperature", "geopotential_height",
                            "wind_speed"))
    i.add_argument("--opacity", type=float, default=0.6)
    i.add_argument("--exaggeration", type=float)
    i.add_argument("--centre", type=float, nargs=2, metavar=("LAT", "LON"))
    i.add_argument("--gravity", type=float)
    i.add_argument("--open", action="store_true")
    i.set_defaults(func=cmd_interactive)

    m = sub.add_parser("measure", help="time the parts of one export")
    m.add_argument("run")
    m.add_argument("--pressure", type=float, required=True)
    m.add_argument("--times", type=int, nargs="+", default=[0, -1],
                   help="saved indices to time (default: first and last)")
    m.add_argument("--resolution", type=float)
    m.add_argument("--gravity", type=float)
    m.add_argument("--out", help="JSON file for the measurements")
    m.add_argument("--scratch", help="directory for the temporary page")
    m.set_defaults(func=cmd_measure)

    c = sub.add_parser("capture", help="offline browser check and captures")
    c.add_argument("page")
    c.add_argument("out_dir")
    c.add_argument("--field", type=int, default=1,
                   help="field button index (0 T, 1 Z, 2 speed)")
    c.add_argument("--exaggeration", type=float, default=10.0)
    c.add_argument("--centre", type=float, nargs=2, metavar=("LAT", "LON"))
    c.add_argument("--hover", type=float, nargs=2, action="append",
                   metavar=("LAT", "LON"), help="points to read out")
    c.set_defaults(func=cmd_capture)

    b = sub.add_parser("bench", help="browser costs of an interactive page")
    b.add_argument("page")
    b.add_argument("out_dir")
    b.add_argument("--pressures", type=float, nargs="+",
                   help="pressures (Pa) to select in turn")
    b.add_argument("--repeat", type=int, default=5)
    b.add_argument("--field", type=int, default=1,
                   help="field button index (0 T, 1 Z, 2 speed)")
    b.add_argument("--exaggeration", type=float, default=10.0)
    b.add_argument("--centre", type=float, nargs=2, metavar=("LAT", "LON"))
    b.set_defaults(func=cmd_bench)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    main()
