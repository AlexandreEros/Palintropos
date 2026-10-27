r"""An evolving dry PE atmosphere over real Mars terrain (MOLA), end to end.

An idealized dry-CO2 perturbation experiment: the balanced isothermal
atmosphere at rest over MOLA terrain (resolved through degree 14), plus a
small (2, 2) temperature perturbation, integrated for one rotation period
and shown in the offline 3-D atmosphere view. Not a Mars climate model: no
radiation, no diurnal heating, no surface physics.

    # terrain (CUDA; examples/mola_mars.py converts the PDS3 grid first)
    python examples/mola_mars.py prepare-pe mola/megt_4ppd.nc out/mola/mola_pe_l21 --lmax 21
    # runs (CUDA): the experiment, its halved-timestep twin, the rest control
    python examples/mola_mars_evolving.py run out/mola/mola_pe_l21 pilot
    python examples/mola_mars_evolving.py run out/mola/mola_pe_l21 main
    python examples/mola_mars_evolving.py run out/mola/mola_pe_l21 halved
    python examples/mola_mars_evolving.py run out/mola/mola_pe_l21 control
    # measurements (CPU): writes results.json
    python examples/mola_mars_evolving.py measure out/mola/evolving --out results.json
    # offline pages and the browser check (CPU + headless Chromium)
    python examples/mola_mars_evolving.py view out/mola/evolving --out out/mola/evolving/views
    python examples/mola_mars_evolving.py browser-check out/mola/evolving/views --out DIR

``run`` prints and executes the exact ``tropoi run pe`` command of each
variant (also recorded in ``<base>/<variant>/command.json``), timing
initialization, integration and persistence and sampling device memory.
``PROTOCOL`` below is the versioned definition of every measured quantity;
``measure`` applies it to the saved runs and writes the numbers, separately.
See docs/validation/mars_evolving_2026-10-27/README.md for the results.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import pathlib
import subprocess
import sys
import threading
import time

import numpy as np

#: Mars-like constants (the terrain-import example's set), one rotation
#: period, nine saved states, four uniform sigma layers.
MARS = {"radius_earth_units": 0.53307, "day_hours": 24.6229, "gravity": 3.71,
        "r_dry": 191.8, "cp_dry": 735.0, "temperature": 210.0,
        "surface_pressure": 610.0}
LEVELS = 4
AMPLITUDE_K = 1.0
ROTATION_S = MARS["day_hours"] * 3600.0              # 88642.44 s
N_SAVED = 9

#: variant -> (scenario, dt_seconds, duration_s, saved states)
VARIANTS = {
    "pilot": ("orographic_thermal_wave", 60.0, 1800.0, 2),
    "main": ("orographic_thermal_wave", 60.0, ROTATION_S, N_SAVED),
    "halved": ("orographic_thermal_wave", 30.0, ROTATION_S, N_SAVED),
    "control": ("orographic_isothermal_rest", 60.0, ROTATION_S, N_SAVED),
}

#: The measurement protocol (versioned; results are stored separately).
PROTOCOL = {
    "version": 1,
    "fields": {
        "temperature": "T_k on the full levels (K)",
        "surface_pressure": "p_s = exp(ln p_s) (Pa)",
        "wind": "horizontal wind (u, v) on the full levels (m/s), "
                "Helmholtz reconstruction from zeta and delta; vector norm",
        "vorticity": "relative vorticity zeta_k (1/s)",
        "divergence": "divergence delta_k (1/s)",
    },
    "grid": "the run's own Gauss-Legendre grid (34 x 68): exact host "
            "synthesis of the saved coefficients at the Gauss latitudes "
            "and uniform longitudes",
    "horizontal_weights": "Gauss weights w_j / (2 nlon) per node (they sum "
                          "to 1: an area mean over the sphere)",
    "vertical_weights": "sigma-layer thickness d_sigma_k (sum to 1) for "
                        "level fields; none for p_s",
    "rms": "sqrt(sum_k d_sigma_k sum_nodes w |X_a - X_b|^2), |.| the "
           "vector norm for the wind",
    "max_abs": "maximum over levels and grid nodes of |X_a - X_b|",
    "evolution": "max over saved times t_j of rms(X(t_j) - X(t_0)) within "
                 "one run",
    "control_drift": "the same evolution measure on the "
                     "orographic_isothermal_rest control (same terrain, "
                     "duration, resolution and 60 s step)",
    "floating_point_floor": "64 * 2.22e-16 * max|X(t_0)| for each field, "
                            "with max|X(t_0)| replaced by the field's "
                            "maximum over the whole run when X(t_0) = 0 "
                            "(winds, vorticity, divergence)",
    "timestep_sensitivity": "rms(X_60(t_j) - X_30(t_j)) at every saved time "
                            "(byte-identical initial states), normalized by "
                            "the maximum evolution of the 30 s run",
    "screening_targets": {
        "evolution_over_control_and_floor": 100.0,
        "timestep_sensitivity_max": 0.05,
        "mass_rel_drift_max": 1e-6,
        "later_wind_min_ms": 0.01,
        "states_with_streamlines_min": 3,
    },
    "notes": "Screening choices for this experiment, not solver guarantees. "
             "Vorticity/divergence sensitivity is screened only where "
             "their evolution exceeds 100 times the floor.",
}


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

def command(terrain: str, variant: str, out: pathlib.Path) -> list[str]:
    scenario, dt, duration, saved = VARIANTS[variant]
    argv = ["run", "pe", "--terrain", str(terrain), "--scenario", scenario,
            "--radius-earth-units", repr(MARS["radius_earth_units"]),
            "--day-hours", repr(MARS["day_hours"]),
            "--gravity", repr(MARS["gravity"]),
            "--r-dry", repr(MARS["r_dry"]), "--cp-dry", repr(MARS["cp_dry"]),
            "--temperature", repr(MARS["temperature"]),
            "--surface-pressure", repr(MARS["surface_pressure"]),
            "--levels", str(LEVELS), "--dt-seconds", repr(dt),
            # --days counts 86400 s days, independently of --day-hours
            "--days", repr(duration / 86400.0),
            "--n-snapshots", str(saved), "--out", str(out)]
    if scenario == "orographic_thermal_wave":
        argv += ["--thermal-amplitude", repr(AMPLITUDE_K)]
    return argv


class _Stamped(io.TextIOBase):
    """Tee stdout, keeping the wall time of every line."""

    def __init__(self, sink):
        self.sink, self.lines, self._part = sink, [], ""

    def write(self, text):
        self.sink.write(text)
        self._part += text
        while "\n" in self._part:
            line, self._part = self._part.split("\n", 1)
            self.lines.append((time.perf_counter(), line))
        return len(text)

    def flush(self):
        self.sink.flush()


def _device_memory_mib() -> float | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"], capture_output=True,
            text=True, timeout=10).stdout
        return float(out.split()[0])
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


def run(terrain: str, variant: str, base: str) -> pathlib.Path:
    import cupy as cp
    from tropoi.cli.main import main

    out = pathlib.Path(base) / variant
    argv = command(terrain, variant, out)
    print("tropoi " + " ".join(argv))
    idle = _device_memory_mib()
    samples, stop = [], threading.Event()

    def sample():
        while not stop.is_set():
            value = _device_memory_mib()
            if value is not None:
                samples.append(value)
            stop.wait(0.25)

    sampler = threading.Thread(target=sample, daemon=True)
    stamped = _Stamped(sys.stdout)
    sampler.start()
    t0 = time.perf_counter()
    try:
        with contextlib.redirect_stdout(stamped):
            rc = main(argv)
    finally:
        stop.set()
        sampler.join()
    t_end = time.perf_counter()
    if rc != 0:
        raise SystemExit(f"tropoi run pe failed ({rc})")
    run_dir = out / (out / "latest_run.txt").read_text(
        encoding="utf-8").strip()
    stores = [t for t, line in stamped.lines if line.startswith("Time:")]
    steps = sum(1 for _ in open(run_dir / "diagnostics" / "timeseries.csv",
                                encoding="utf-8")) - 2
    integration = stores[-1] - stores[0]
    record = {
        "variant": variant, "command": ["tropoi", *argv],
        "run_dir": str(run_dir), "steps": steps,
        "wall_s": {"initialization": stores[0] - t0,
                   "integration": integration,
                   "persistence_and_exit": t_end - stores[-1],
                   "total": t_end - t0},
        "integration_s_per_step": integration / max(steps, 1),
        "device_memory_mib": {
            "idle_before": idle, "peak_sampled": max(samples, default=None),
            "sampling": "nvidia-smi memory.used every 0.25 s (whole device, "
                        "including the CUDA context)"},
        "cupy_pool_bytes_after": int(cp.get_default_memory_pool()
                                     .total_bytes()),
        # the manifest's git "dirty" counts untracked files too: keep the
        # exact status so a clean tracked tree can be told apart
        "git_status_porcelain": subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True,
            text=True).stdout.splitlines(),
    }
    (out / "command.json").write_text(json.dumps(record, indent=2) + "\n",
                                      encoding="utf-8")
    print(json.dumps(record["wall_s"]), record["integration_s_per_step"],
          "s/step; peak", record["device_memory_mib"]["peak_sampled"], "MiB")
    return run_dir


# ---------------------------------------------------------------------------
# measure
# ---------------------------------------------------------------------------

def _latest(base: pathlib.Path, variant: str) -> pathlib.Path:
    out = base / variant
    return out / (out / "latest_run.txt").read_text(encoding="utf-8").strip()


class _Fields:
    """Exact host synthesis of saved PE states on the run's Gauss grid."""

    def __init__(self, run_dir: pathlib.Path):
        from tropoi.representation.archive import open_simulation

        self.run_dir = run_dir
        self.config = json.loads((run_dir / "config.json").read_text(
            encoding="utf-8"))
        self.coeffs = np.load(run_dir / "pe_coeffs.npy")
        self.times = np.load(run_dir / "pe_snapshot_times.npy")
        self.K = int(self.config["nlev"])
        storage = open_simulation(run_dir).storage
        self.radius = storage.pe_context().radius_m
        nlat, nlon = int(self.config["nlat"]), int(self.config["nlon"])
        x, w = np.polynomial.legendre.leggauss(nlat)
        self.lat = np.degrees(np.arcsin(x))
        self.lon = np.arange(nlon) * 360.0 / nlon
        self.w = (w / (2.0 * nlon))[:, None] * np.ones((1, nlon))
        s = np.asarray(self.config["sigma_interfaces"], dtype=float)
        self.dsigma = np.diff(s)
        self._cache = {}

    def get(self, j: int) -> dict:
        if j in self._cache:
            return self._cache[j]
        from tropoi.spatial.transforms.host_synthesis import (
            synthesize_latlon, synthesize_wind_latlon)
        c, K = self.coeffs[j], self.K
        syn = lambda lm: synthesize_latlon(lm, self.lat, self.lon)
        wind = [synthesize_wind_latlon(c[k], c[K + k], self.radius,
                                       self.lat, self.lon) for k in range(K)]
        out = {"temperature": np.stack([syn(c[2 * K + k]) for k in range(K)]),
               "surface_pressure": np.exp(syn(c[3 * K]))[None],
               "wind": np.stack([np.stack(uv) for uv in wind]),
               "vorticity": np.stack([syn(c[k]) for k in range(K)]),
               "divergence": np.stack([syn(c[K + k]) for k in range(K)])}
        self._cache[j] = out
        return out

    def diff(self, a: dict, b: dict, name: str) -> tuple[float, float]:
        d = a[name] - b[name]
        if name == "wind":
            d2 = (d ** 2).sum(axis=1)                  # |du|^2 + |dv|^2
            mag = np.sqrt(d2)
        else:
            d2, mag = d ** 2, np.abs(d)
        vw = self.dsigma if d2.shape[0] == self.K else np.ones(1)
        rms = math.sqrt(float(np.sum(vw[:, None, None] * self.w[None] * d2)))
        return rms, float(mag.max())

    def magnitude(self, j: int, name: str) -> float:
        x = self.get(j)[name]
        return float(np.sqrt((x ** 2).sum(axis=1)).max() if name == "wind"
                     else np.abs(x).max())


def _diagnostics(run_dir: pathlib.Path) -> dict:
    import csv
    with open(run_dir / "diagnostics" / "timeseries.csv",
              encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    col = lambda k: np.array([float(r[k]) for r in rows])
    courant = col("courant")
    return {"steps": int(col("step")[-1]),
            "final_time_s": float(col("time_s")[-1]),
            "dt_values_s": sorted({float(v) for v in col("dt_s")[1:]}),
            "t_min_K": float(col("t_min").min()),
            "t_max_K": float(col("t_max").max()),
            "ps_min_Pa": float(col("ps_min").min()),
            "ps_max_Pa": float(col("ps_max").max()),
            "max_wind_ms": float(col("max_wind_ms").max()),
            "final_max_wind_ms": float(col("max_wind_ms")[-1]),
            "max_courant": float(np.nanmax(courant)),
            "max_abs_mass_rel_drift": float(np.abs(col("mass_rel_drift"))
                                            .max()),
            "all_finite": bool(all(np.all(np.isfinite(col(k)))
                                   for k in rows[0] if k != "courant"))}


def measure(base: str, out: str) -> dict:
    base = pathlib.Path(base)
    runs = {v: _Fields(_latest(base, v)) for v in ("main", "halved",
                                                   "control")}
    main_, half, ctrl = runs["main"], runs["halved"], runs["control"]
    # the 60/30 pair: same initial bytes, same terrain, same saved times
    same_start = main_.coeffs[0].tobytes() == half.coeffs[0].tobytes()
    differ = {k for k in set(main_.config) | set(half.config)
              if main_.config.get(k) != half.config.get(k)}
    assert np.array_equal(main_.times, half.times)
    assert np.array_equal(main_.times, ctrl.times)
    names = list(PROTOCOL["fields"])
    targets = PROTOCOL["screening_targets"]
    result = {"protocol": PROTOCOL, "runs": {}, "fields": {},
              "pair": {"initial_states_byte_identical": same_start,
                       "config_keys_that_differ": sorted(differ)},
              "saved_times_s": [float(t) for t in main_.times]}
    for v, f in runs.items():
        manifest = json.loads((f.run_dir / "manifest.json").read_text(
            encoding="utf-8"))
        timing = json.loads((base / v / "command.json").read_text(
            encoding="utf-8"))
        result["runs"][v] = {
            "run_id": manifest["run_id"], "run_dir": f.run_dir.name,
            "terrain_digest": f.config.get("terrain_digest"),
            "dt_seconds": f.config["dt_seconds"],
            "git": manifest.get("git"), "diagnostics": _diagnostics(f.run_dir),
            "timing": {k: timing[k] for k in (
                "wall_s", "integration_s_per_step", "device_memory_mib",
                "cupy_pool_bytes_after")}}
    n = len(main_.times)
    for name in names:
        evo = lambda f: [f.diff(f.get(j), f.get(0), name) for j in range(n)]
        e_main, e_half, e_ctrl = evo(main_), evo(half), evo(ctrl)
        scale = main_.magnitude(0, name) or max(
            main_.magnitude(j, name) for j in range(n))
        floor = 64 * 2.22e-16 * scale
        sens = [main_.diff(main_.get(j), half.get(j), name)
                for j in range(n)]
        evolution = max(r for r, _ in e_main)
        evolution_half = max(r for r, _ in e_half)
        control = max(r for r, _ in e_ctrl)
        resolved = evolution > 100 * floor
        result["fields"][name] = {
            "evolution_rms_by_time": [r for r, _ in e_main],
            "evolution_max_abs_by_time": [m for _, m in e_main],
            "evolution_rms_max": evolution,
            "evolution_rms_max_halved": evolution_half,
            "control_drift_rms_max": control,
            "control_drift_max_abs": max(m for _, m in e_ctrl),
            "floating_point_floor": floor,
            "evolution_over_control_and_floor":
                evolution / max(control, floor),
            "timestep_rms_by_time": [r for r, _ in sens],
            "timestep_max_abs_by_time": [m for _, m in sens],
            "timestep_sensitivity_by_time":
                [r / evolution_half for r, _ in sens],
            "timestep_sensitivity_max":
                max(r for r, _ in sens) / evolution_half,
            "resolved_above_floor": bool(resolved),
        }
    d_main = result["runs"]["main"]["diagnostics"]
    fields = result["fields"]
    screens = {
        "evolution_over_control_and_floor": all(
            fields[k]["evolution_over_control_and_floor"]
            >= targets["evolution_over_control_and_floor"] for k in names),
        "timestep_sensitivity": all(
            fields[k]["timestep_sensitivity_max"]
            <= targets["timestep_sensitivity_max"]
            for k in names if fields[k]["resolved_above_floor"]),
        "mass": max(result["runs"][v]["diagnostics"]["max_abs_mass_rel_drift"]
                    for v in ("main", "halved"))
        <= targets["mass_rel_drift_max"],
        "positive_and_finite": all(
            result["runs"][v]["diagnostics"]["t_min_K"] > 0
            and result["runs"][v]["diagnostics"]["ps_min_Pa"] > 0
            and result["runs"][v]["diagnostics"]["all_finite"]
            and bool(np.all(np.isfinite(runs[v].coeffs)))
            for v in runs),
        "later_wind": d_main["final_max_wind_ms"] >= targets[
            "later_wind_min_ms"],
    }
    result["screens"] = screens
    pathlib.Path(out).write_text(json.dumps(result, indent=2) + "\n",
                                 encoding="utf-8", newline="\n")
    for name in names:
        f = fields[name]
        print(f"{name:17s} evolution {f['evolution_rms_max']:.3e}  control "
              f"{f['control_drift_rms_max']:.2e}  floor "
              f"{f['floating_point_floor']:.1e}  ratio "
              f"{f['evolution_over_control_and_floor']:.2e}  60/30 "
              f"{f['timestep_sensitivity_max']:.2e}")
    print("screens:", screens)
    return result


def figure(base: str, out: str, level: int = LEVELS - 1) -> None:
    """Initial/final/difference maps with common colour limits."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    f = _Fields(_latest(pathlib.Path(base), "main"))
    a, b = f.get(0), f.get(len(f.times) - 1)
    lon = np.concatenate([f.lon, [360.0]])
    wrap = lambda x: np.concatenate([x, x[:, :1]], axis=1)
    speed = lambda s: np.hypot(*s["wind"][level])
    rows = [("T (K)", a["temperature"][level], b["temperature"][level],
             "inferno"),
            ("p_s (Pa)", a["surface_pressure"][0], b["surface_pressure"][0],
             "viridis"),
            ("|V| (m/s)", speed(a), speed(b), "magma")]
    fig, axes = plt.subplots(3, 3, figsize=(15, 9.5), constrained_layout=True)
    days = f.times[-1] / 86400.0
    for r, (label, x0, x1, cmap) in enumerate(rows):
        lo, hi = min(x0.min(), x1.min()), max(x0.max(), x1.max())
        d = x1 - x0
        m = float(np.abs(d).max()) or 1.0
        for c, (x, title, kw) in enumerate((
                (x0, "t = 0", dict(cmap=cmap, vmin=lo, vmax=hi)),
                (x1, f"t = {f.times[-1]:.2f} s ({days:.4g} d)",
                 dict(cmap=cmap, vmin=lo, vmax=hi)),
                (d, "final - initial", dict(cmap="RdBu_r", vmin=-m, vmax=m)))):
            ax = axes[r, c]
            im = ax.pcolormesh(lon, f.lat, wrap(x), shading="nearest", **kw)
            ax.set_title(f"{label}: {title}", fontsize=10)
            ax.set_xlim(0, 360)
            ax.set_ylim(-90, 90)
            ax.set_xticks(range(0, 361, 60))
            ax.set_yticks(range(-90, 91, 30))
            fig.colorbar(im, ax=ax, shrink=0.85)
    sigma = 0.5 * (np.asarray(f.config["sigma_interfaces"][:-1])
                   + np.asarray(f.config["sigma_interfaces"][1:]))[level]
    fig.suptitle(f"Mars orographic_thermal_wave over MOLA (degree <= 14), "
                 f"{f.config['run_id']}: level {level + 1} of {f.K} "
                 f"(sigma = {sigma:g}) and surface pressure; Gauss grid "
                 f"{len(f.lat)} x {len(f.lon)}, east longitude", fontsize=11)
    fig.savefig(out, dpi=90)
    print("wrote", out)


# ---------------------------------------------------------------------------
# view and browser check
# ---------------------------------------------------------------------------

VIEW = {"resolution_deg": 3.0, "field": "delta", "level": LEVELS - 1,
        "opacity": 0.55, "centre": (0.0, -100.0), "exaggeration": 10.0}


def view(base: str, out: str) -> dict:
    from tropoi.representation.archive import open_simulation

    sim = open_simulation(_latest(pathlib.Path(base), "main"))
    out = pathlib.Path(out)
    out.mkdir(parents=True, exist_ok=True)
    pages = {"all": sim.view(out / "mars_evolving_all_times.html",
                             open_browser=False, times="all", **VIEW),
             "initial": sim.view(out / "mars_evolving_initial.html",
                                 open_browser=False, times=[0], **VIEW),
             "final": sim.view(out / "mars_evolving_final.html",
                               open_browser=False, times=[-1], **VIEW)}
    for name, path in pages.items():
        print(f"{name:8s} {path} ({path.stat().st_size / 1e6:.2f} MB)")
    return pages


def browser_check(views: str, out: str) -> dict:
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    from tools.browser_cdp import OfflinePage, find_browser

    browser = find_browser()
    if browser is None:
        raise SystemExit("no Chromium-family browser found")
    views, out = pathlib.Path(views), pathlib.Path(out)
    out.mkdir(parents=True, exist_ok=True)
    api = "window.__terrainGlobe.atmosphere"
    record = {"pages": {}}
    with OfflinePage(browser, width=1400, height=1000) as page:
        record["browser"] = page.version
        url = page.open(views / "mars_evolving_all_times.html")
        page.wait_for(f"{api} && document.getElementById('timeLabel')"
                      ".textContent")
        nt = page.eval(f"{api}.info() && "
                       "+document.getElementById('timeSlider').max + 1")
        nl = page.eval("+document.getElementById('levelSlider').max + 1")
        nf = page.eval("document.getElementById('fieldButtons')"
                       ".children.length")
        states = []
        for it in range(nt):
            page.eval("(() => { const s = document.getElementById("
                      f"'timeSlider'); s.value = {it}; s.dispatchEvent("
                      "new Event('input')); })()")
            info = page.eval(f"{api}.info()")
            states.append({
                "label": page.eval("document.getElementById('timeLabel')"
                                   ".textContent"),
                "digest": page.eval(f"{api}.digest()"),
                "streamlines": info["lines"],
                "radius_mean": info["radius"]["mean"],
                "terrain_nodes_above_layer": info["terrainNodesAboveLayer"]})
        levels_fields = []
        for il in range(nl):
            for k in range(nf):
                page.eval(f"{api}.setLevel({il}); {api}.setField({k})")
                info = page.eval(f"{api}.info()")
                levels_fields.append({"level": il, "field": info["field"],
                                      "lines": info["lines"],
                                      "range": info["range"]})
        page.eval(f"{api}.setLevel({VIEW['level']}); {api}.setField(3);"
                  f" {api}.setTime(0)")
        page.eval("document.getElementById('tSpeed').value = '4';"
                  "document.getElementById('tSpeed').dispatchEvent("
                  "new Event('change'))")
        page.eval("document.getElementById('tPlay').click()")
        page.wait_for(f"!{api}.info().playing", timeout=60)
        played_to = page.eval(f"{api}.info().timeIndex")
        page.eval(f"{api}.centreOn{VIEW['centre']}; {api}.setTime(0)")
        page.eval("new Promise(r => setTimeout(r, 500))")
        page.screenshot(out / "mars_evolving_initial_delta.png")
        page.eval(f"{api}.setTime({nt - 1})")
        page.eval("new Promise(r => setTimeout(r, 500))")
        page.screenshot(out / "mars_evolving_final_delta.png")
        record["pages"]["all"] = {
            "url_requests": page.requests, "expected": [url],
            "console_errors": page.console_errors,
            "exceptions": page.exceptions, "glOK": page.eval(
                "window.__terrainGlobe.glOK"),
            "states": states, "levels_fields": levels_fields,
            "distinct_digests": len({s["digest"] for s in states}),
            "playback_stopped_at": played_to}
    for name in ("initial", "final"):
        with OfflinePage(browser, width=1400, height=1000) as page:
            url = page.open(views / f"mars_evolving_{name}.html")
            page.wait_for(f"{api}")
            record["pages"][name] = {
                "url_requests": page.requests, "expected": [url],
                "console_errors": page.console_errors,
                "exceptions": page.exceptions,
                "play_controls_hidden": page.eval(
                    "document.getElementById('playRow').hidden")}
    ok = all(p["url_requests"] == p["expected"] and not p["console_errors"]
             and not p["exceptions"] for p in record["pages"].values())
    record["offline_and_clean"] = ok
    (out / "browser_check.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8", newline="\n")
    s = record["pages"]["all"]
    print(f"{record['browser']}: offline/clean {ok}; {s['distinct_digests']}"
          f" distinct states of {nt}; streamlines per state "
          f"{[x['streamlines'] for x in s['states']]}; playback stopped at "
          f"{s['playback_stopped_at']}")
    return record


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="step", required=True)
    r = sub.add_parser("run", help="one variant (CUDA)")
    r.add_argument("terrain")
    r.add_argument("variant", choices=list(VARIANTS))
    r.add_argument("--base", default="out/mola/evolving")
    m = sub.add_parser("measure", help="apply PROTOCOL (CPU)")
    m.add_argument("base")
    m.add_argument("--out", required=True)
    f = sub.add_parser("figure", help="initial/final/difference maps (CPU)")
    f.add_argument("base")
    f.add_argument("--out", required=True)
    v = sub.add_parser("view", help="offline pages (CPU)")
    v.add_argument("base")
    v.add_argument("--out", required=True)
    b = sub.add_parser("browser-check", help="drive the pages offline")
    b.add_argument("views")
    b.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if args.step == "run":
        run(args.terrain, args.variant, args.base)
    elif args.step == "measure":
        measure(args.base, args.out)
    elif args.step == "figure":
        figure(args.base, args.out)
    elif args.step == "view":
        view(args.base, args.out)
    else:
        browser_check(args.views, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
