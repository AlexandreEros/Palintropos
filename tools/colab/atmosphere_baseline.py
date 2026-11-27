"""Atmospheric baseline on the T4: the complete primitive-equation step of the
heated radiative runs, by component. Runs tools/colab/radiative_pilot.py
(unchanged) from the checkout in the current directory, with two extra
cases/labels added here:

* the THAI Ben 2 correlated-k case (``dev-thai-ben2-ck``, T21L40), which needs
  its k-tables under TROPOI_OPACITY_PATH;
* labels for the boundary-layer pieces (the implicit column diffusion and the
  diffusivity profile) and for the radiative-transfer flux evaluation.

    PYTHONPATH=src:. python tools/colab/atmosphere_baseline.py OUT_DIR
        # throughput, sync and cprofile for every case
    PYTHONPATH=src:. nsys profile -t cuda,nvtx -o REPORT \
        python tools/colab/atmosphere_baseline.py OUT_DIR nvtx CASE RES

The THAI case needs its k-tables on TROPOI_OPACITY_PATH. Results:
docs/validation/atmosphere-baseline/.
"""
import importlib.util
import json
import pathlib
import sys
import time

sys.path[:0] = ["src", "."]
spec = importlib.util.spec_from_file_location("rp", "tools/colab/radiative_pilot.py")
rp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rp)

from tropoi.run.radiative.config import PRESETS  # noqa: E402

_config = rp.config


def config(case, resolution="t21"):
    if case == "thai-ben2-ck":
        cfg = PRESETS["dev-thai-ben2-ck"]()
        if resolution != "t21":
            cfg = cfg.with_(experiment_id=f"{cfg.experiment_id}-{resolution}",
                            **rp.RESOLUTIONS[resolution])
        return cfg
    return _config(case, resolution)


rp.config = config
rp.CASES["thai-ben2-ck"] = ("dev-thai-ben2-ck", None)
_instrument = rp.instrument


def instrument(run, rec):
    done = _instrument(run, rec)
    import tropoi.temporal.tendencies.boundary_layer as tbl
    for name in ("implicit_column_diffusion", "monin_obukhov_diffusivity_profile",
                 "boundary_layer_depth", "diffusivity_profile", "exchange_coefficient"):
        if hasattr(tbl, name):
            setattr(tbl, name, rec.wrap("bl." + name, getattr(tbl, name)))
            done.append("bl." + name)
    if run.coupling is not None and rec.patch(run.coupling.physics.radiation, "fluxes",
                                              "physics.radiation.fluxes"):
        done.append("radiation.fluxes")
    return done


rp.instrument = instrument

CASES = [("earth", "t21"), ("earth", "t42"), ("thai-ben2-ck", "t21")]


def throughput(case, res, out, steps=72, warm=6):
    """Production stepping, uninstrumented: wall time per step, synchronized
    only at the ends."""
    import cupy as cp
    from tropoi.run.radiative.experiment import RadiativeRun
    cfg = config(case, res)
    run = RadiativeRun(cfg, out / f"scratch-{case}-{res}-{int(time.time())}", allow_dirty=True,
                       log=lambda s: None)
    for _ in range(warm):
        run.stepper.step()
    cp.cuda.Device().synchronize()
    t = time.perf_counter()
    for _ in range(steps):
        run.stepper.step()
    cp.cuda.Device().synchronize()
    pool = cp.get_default_memory_pool()
    return {"case": case, "resolution": res, "steps": steps,
            "seconds_per_step": (time.perf_counter() - t) / steps,
            "nlat": cfg.nlat, "nlon": cfg.nlon, "nlev": run.model.nlev, "dt": cfg.dt,
            "memory_pool_total_bytes": int(pool.total_bytes()), "code": run.code,
            "config_sha256": cfg.sha256()}


def main():
    out = pathlib.Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    if len(sys.argv) > 2 and sys.argv[2] == "nvtx":
        case, res = sys.argv[3], sys.argv[4]
        rp.main(["profile", "--case", case, "--resolution", res, "--mode", "nvtx",
                 "--steps", "24", "--out", str(out), "--scratch", str(out / "scratch")])
        return
    results = {"device": rp.device(), "throughput": [], "errors": []}
    for case, res in CASES:
        for step in ("throughput", "sync", "cprofile"):
            try:
                if step == "throughput":
                    results["throughput"].append(throughput(case, res, out))
                else:
                    rp.main(["profile", "--case", case, "--resolution", res, "--mode", step,
                             "--steps", "36", "--out", str(out),
                             "--scratch", str(out / "scratch")])
            except Exception as exc:                                  # noqa: BLE001
                results["errors"].append({"case": case, "resolution": res, "step": step,
                                          "error": repr(exc)[:2000]})
            (out / "baseline-summary.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
