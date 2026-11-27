"""Launch ``substrate_benchmark.py`` on a Colab GPU at a pinned commit.

Executed in the Colab kernel (``colab exec -f``), not imported:

    colab new --gpu T4 -s bench
    colab exec -s bench -f tools/colab/substrate_benchmark_colab.py --timeout 600 \\
        --env BENCH_SHA=<full commit> --env "BENCH_ARGS=all"
    colab exec -s bench -f tools/colab/substrate_benchmark_colab.py --env BENCH_POLL=1 ...
    colab download -s bench /content/bench/results-<sha7>.json results.json
    colab stop -s bench

It clones the public repository into ``/content/bench``, checks out
``BENCH_SHA`` and asserts that HEAD is exactly that commit, and uses Colab's
own CuPy (nothing is pip-installed, so the environment is the image's and
is recorded by the benchmark). The benchmark runs detached, so a dropped
``colab exec`` does not stop it; ``BENCH_POLL=1`` prints its progress.
In a notebook, set the two variables in ``os.environ`` and run this file's
text as a cell.
"""
import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path("/content/bench")
REPO = ROOT / "Palintropos"
SHA = os.environ["BENCH_SHA"]
TAG = SHA[:7]
LOG = ROOT / f"bench-{TAG}.log"
OUT = ROOT / f"results-{TAG}.json"

if os.environ.get("BENCH_POLL"):
    print(LOG.read_text()[-3000:] if LOG.exists() else "no log yet")
    print("DONE" if (ROOT / f"done-{TAG}").exists() else "RUNNING")
    raise SystemExit(0)

ROOT.mkdir(parents=True, exist_ok=True)
if not REPO.exists():
    subprocess.check_call(["git", "clone", "-q", "https://github.com/AlexandreEros/Palintropos.git",
                           str(REPO)])
subprocess.check_call(["git", "-C", str(REPO), "fetch", "-q", "origin", SHA])
subprocess.check_call(["git", "-C", str(REPO), "checkout", "-q", "--detach", SHA])
head = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
assert head == SHA, (head, SHA)
args = os.environ.get("BENCH_ARGS", "all").split()
cmd = (f"{sys.executable} tools/colab/substrate_benchmark.py {' '.join(args)} --out {OUT} "
       f"--state-file {ROOT / 'state.npz'} > {LOG} 2>&1; touch {ROOT / f'done-{TAG}'}")
env = {**os.environ, "PYTHONPATH": "src:."}
subprocess.Popen(["bash", "-c", cmd], cwd=REPO, env=env, start_new_session=True)
print(f"started at {head}: {cmd}")
