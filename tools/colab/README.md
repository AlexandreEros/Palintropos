# Colab radiative pilot tooling

[radiative_pilot.py](radiative_pilot.py) times warmed-up production steps
and leaves a checkpoint. Its profiling mode uses a scratch run and supports
synchronised timings, NVTX ranges, and Python profiling.

[build_notebook.py](build_notebook.py) builds a timing notebook with the
script embedded. The preserved [Colab notebook](../../notebooks/radiative_colab_pilot.ipynb)
also carries the historical batched-heating profiling cells. Run it as
supplied to use that workflow. The [notebook index](../../notebooks/README.md)
links the resulting evidence.
