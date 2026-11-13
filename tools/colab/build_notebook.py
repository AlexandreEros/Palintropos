"""Write ``notebooks/radiative_colab_pilot.ipynb`` with ``radiative_pilot.py``
embedded, so the notebook needs nothing from the repository but the pinned
commit it clones.

    python tools/colab/build_notebook.py
"""
from __future__ import annotations

import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
PINNED = "d38d2a3"          # resolved to the full hash in the notebook and asserted


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(True)}


def main() -> None:
    import subprocess
    full = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", PINNED],
                                   text=True).strip()
    pilot = (HERE / "radiative_pilot.py").read_text(encoding="utf-8")
    drag = (REPO / "src" / "tropoi" / "temporal" / "tendencies" / "drag.py").read_text(
        encoding="utf-8")
    cells = [
        md(f"""
# Heated radiative runs: timing pilot (Colab)

Times a short, warmed-up stretch of the **production** heated Earth or heated
Jupiter run (gray radiation, frictional heating on, T21) with profiling off,
and leaves a resumable checkpoint on Google Drive.

**Run cells 1 to 4 in order.** Cells 5 and 6 are optional and are not needed
for the timing. Nothing here installs Nsight.

- Code is pinned to `{full}` of `AlexandreEros/Palintropos` (public clone).
- Set `CASE` in cell 1 to `"earth"` or `"jupiter"`. To run both at once, open
  a second copy of this notebook (File > Save a copy in Drive), set the other
  case there, and run it in its own runtime.
- Runtime: Runtime > Change runtime type > T4 GPU (the notebook asks for a
  GPU; cell 1 prints the one you got).
"""),
        code(f"""
# 1. Choose the case, clone the pinned commit, show the GPU
CASE = "earth"            # "earth" or "jupiter"
UNTIL_DAY = 10            # the pilot stops at this day's checkpoint
PINNED = "{full}"

import os, subprocess
assert CASE in ("earth", "jupiter")
if not os.path.isdir("/content/Palintropos/.git"):
    subprocess.run(["git", "clone", "-q", "https://github.com/AlexandreEros/Palintropos.git",
                    "/content/Palintropos"], check=True)
subprocess.run(["git", "-C", "/content/Palintropos", "fetch", "-q", "origin", PINNED], check=False)
subprocess.run(["git", "-C", "/content/Palintropos", "checkout", "-q", "--detach", PINNED], check=True)
head = subprocess.check_output(["git", "-C", "/content/Palintropos", "rev-parse", "HEAD"], text=True).strip()
assert head == PINNED, head
print("code:", head)
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv"],
                     capture_output=True, text=True).stdout)
import cupy, numpy, scipy
print("cupy", cupy.__version__, "numpy", numpy.__version__, "scipy", scipy.__version__)
"""),
        code("""
# 2. Where the run is kept. Drive keeps checkpoints and states outside this VM
#    (a pop-up asks you to allow access). Set USE_DRIVE = False to keep the run
#    on the VM only: it is then lost when the runtime is deleted.
USE_DRIVE = True

EXPERIMENT = {"earth": "earth-gray-dry-heated-T21L25", "jupiter": "jupiter-sl09-heated-T21L30"}[CASE]
if USE_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    ROOT = "/content/drive/MyDrive/palintropos-runs/radiative"
else:
    ROOT = "/content/palintropos-runs/radiative"
RUN_DIR = f"{ROOT}/{EXPERIMENT}-{PINNED[:7]}-colab"
os.makedirs(RUN_DIR, exist_ok=True)
print("run directory:", RUN_DIR)
"""),
        code("%%writefile /content/radiative_pilot.py\n# 3. The pilot script (written to the VM; changes nothing in the package)\n"
             + pilot),
        code("""
# 4. TIMING PILOT: one warm-up day, then a timed stretch to UNTIL_DAY with the
#    production loop. Prints and saves pilot-timing-d<day>.json in RUN_DIR.
#    Re-running with a larger UNTIL_DAY resumes from the last checkpoint.
env = dict(os.environ, PYTHONPATH="/content/Palintropos/src")
log = open(f"{RUN_DIR}/pilot-timing.log", "a")
proc = subprocess.Popen(["python", "/content/radiative_pilot.py", "time", "--case", CASE,
                         "--run-dir", RUN_DIR, "--until-day", str(UNTIL_DAY)],
                        cwd="/content/Palintropos", env=env, text=True,
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
for line in proc.stdout:
    print(line, end="")
    log.write(line)
log.close()
assert proc.wait() == 0, "pilot failed: see the output above"

import glob, json
doc = json.load(open(sorted(glob.glob(f"{RUN_DIR}/pilot-timing-d*.json"))[-1]))
print()
print("GPU                         :", doc["device"]["name"])
print("seconds per step (all in)   : %.4f" % doc["seconds_per_step_all_inclusive"])
print("seconds per step (stepping) : %.4f" % doc["seconds_per_step_stepping_only"])
print("daily sample / checkpoint   : %.3f s / %.3f s" % (doc["seconds_per_daily_sample"], doc["seconds_per_checkpoint"]))
print("simulated days per hour     : %.1f" % doc["simulated_days_per_wall_hour"])
print("estimated full run          : %.2f h for %d days" % (doc["estimated_full_run_hours"], doc["target_days"]))
print("GPU utilisation (nvidia-smi): ", doc["gpu_utilisation"])
"""),
        md("""
## Optional (not needed for the timing)

Cell 5 is a profile with a device synchronisation around every instrumented
call, on a scratch directory; its numbers are **not** throughput. Cell 6
continues the production run to its full length from the pilot's checkpoint.
"""),
        code("""
# 5. OPTIONAL: synchronised per-component timings (scratch run, T21), with the
#    transforms split by caller and by scalar/batched entry point. Not a
#    throughput number. Needs cells 1 to 3; the JSON and this table are saved
#    in RUN_DIR/profile.
env = dict(os.environ, PYTHONPATH="/content/Palintropos/src")
r = subprocess.run(["python", "/content/radiative_pilot.py", "profile", "--case", CASE,
                    "--mode", "sync", "--steps", "36", "--out", f"{RUN_DIR}/profile",
                    "--scratch", "/content/scratch-profile"],
                   cwd="/content/Palintropos", env=env, text=True, capture_output=True)
print(r.stderr[-3000:])
print(open(f"{RUN_DIR}/profile/profile-{CASE}-t21-sync.txt").read() if r.returncode == 0 else r.stdout[-4000:])
"""),
        md("""
### Profile of the batched heating term (uncommitted change)

Cells 5b and 5c profile the same configuration with a newer
`temporal/tendencies/drag.py`, in which the frictional-heating term uses one
batched synthesis and one batched analysis. The file is laid over a **copy**
of the pinned checkout; the production checkout and run directory are not
touched, and the profile records `git_dirty: true`. Needs cells 1 to 3.
"""),
        code("%%writefile /content/drag_batched.py\n" + drag),
        code("""
# 5c. OPTIONAL: synchronised profile with the batched heating term (scratch run).
import shutil
if not os.path.isdir("/content/Palintropos-batched"):
    shutil.copytree("/content/Palintropos", "/content/Palintropos-batched", symlinks=True)
shutil.copyfile("/content/drag_batched.py",
                "/content/Palintropos-batched/src/tropoi/temporal/tendencies/drag.py")
env = dict(os.environ, PYTHONPATH="/content/Palintropos-batched/src")
r = subprocess.run(["python", "/content/radiative_pilot.py", "profile", "--case", CASE,
                    "--mode", "sync", "--steps", "36", "--out", f"{RUN_DIR}/profile-batched",
                    "--scratch", "/content/scratch-profile-batched"],
                   cwd="/content/Palintropos-batched", env=env, text=True, capture_output=True)
print(r.stderr[-3000:])
print(open(f"{RUN_DIR}/profile-batched/profile-{CASE}-t21-sync.txt").read() if r.returncode == 0 else r.stdout[-4000:])
"""),
        code("""
# 6. OPTIONAL: continue the full run from the pilot's checkpoint (hours).
PRESET = {"earth": "earth-gray-dry", "jupiter": "jupiter-schneider-liu"}[CASE]
env = dict(os.environ, PYTHONPATH="/content/Palintropos/src")
subprocess.run(["python", "-m", "tropoi.run.radiative", "run", "--preset", PRESET,
                "--frictional-heating", "--experiment-id", EXPERIMENT, "--run-dir", RUN_DIR],
               cwd="/content/Palintropos", env=env, check=True)
"""),
    ]
    nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
          "metadata": {"accelerator": "GPU",
                       "colab": {"provenance": [], "gpuType": "T4"},
                       "kernelspec": {"display_name": "Python 3", "name": "python3"},
                       "language_info": {"name": "python"}}}
    out = REPO / "notebooks" / "radiative_colab_pilot.ipynb"
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8",
                   newline="\n")
    print(out)


if __name__ == "__main__":
    main()
