# Integration tests

The barotropic-vorticity tests here exercise the controller and runner across
their model, time integration, and output boundaries. Run
`python -m pytest tests/integration -q`. The [test guide](../README.md)
describes how CUDA-dependent coverage is handled.
