# GitHub Actions workflows

- [tests.yml](tests.yml) runs the CPU pytest suite. Tests requiring CUDA can
  skip in this environment.
- [architecture.yml](architecture.yml) checks the current generated
  architecture snapshot against the source and runs the architecture-tool tests.
  See the [generation instructions](../../docs/architecture/README.md).
