# Colab radiative pilot tooling

[radiative_pilot.py](radiative_pilot.py) times warmed-up production steps
and leaves a checkpoint. Its profiling mode uses a scratch run and supports
synchronised timings, NVTX ranges, and Python profiling.

[build_notebook.py](build_notebook.py) builds a timing notebook with the
script embedded. The preserved [Colab notebook](../../notebooks/radiative_colab_pilot.ipynb)
also carries the historical batched-heating profiling cells. Run it as
supplied to use that workflow. The [notebook index](../../notebooks/README.md)
links the resulting evidence.

## Substrate and atmosphere measurements

* [substrate_gate.py](substrate_gate.py): the substrate's regression gate on
  the T4 (representative state, timings, P5, CPU against CUDA).
* [substrate_nsys.py](substrate_nsys.py): the per-step kernel and
  synchronization census (P3 protocol).
* [atmosphere_baseline.py](atmosphere_baseline.py): the atmospheric step by
  component (throughput, synchronized components, cProfile, NVTX for Nsight).
* [bl_thomas_equivalence.py](bl_thomas_equivalence.py): the D3 study of the
  boundary layer's tridiagonal solve against the canonical `pt::thomas`.
