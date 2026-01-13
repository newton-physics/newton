# Newton Guidelines

## Public API and `_src` boundary

- **`newton/_src/` is internal library implementation only.**
  - User code **must not** import from `newton._src`.
  - Internal refactors can freely reorganize code under `_src` as long as the public API stays stable.
- **Any user-facing class/function/object added under `_src` must be exposed via the public Newton API.**
  - Add re-exports in the appropriate public module (e.g. `newton/geometry.py`, `newton/solvers.py`, `newton/sensors.py`, etc.).
  - Prefer a single, discoverable public import path. Example: `from newton.geometry import BroadPhaseAllPairs` (not `from newton._src.geometry.broad_phase_all_pairs import BroadPhaseAllPairs`).

## API design rules (naming + structure)

- **Prefix-first naming for discoverability (autocomplete).**
  - **Classes**: `ActuatorPD`, `ActuatorPID` (not `PDActuator`, `PIDActuator`).
  - **Methods**: `add_shape_sphere()` (not `add_sphere_shape()`).
- **Method names are `snake_case`.**
- **CLI arguments are `kebab-case`.**
  - Example: `--use-cuda-graph` (not `--use_cuda_graph`).
- **Prefer nested classes when self-contained.**
  - If a helper type is only meaningful inside one parent class and doesn’t need a public identity, define it as a nested class instead of creating a new top-level class/module.

## Tooling: use `uv` for running, testing, and benchmarking

We standardize on `uv` for local workflows. Example commands (from `docs/guide/development.rst`):

### Run examples

Newton examples live under `newton/examples/` and its subfolders. See `README.md` for uv commands.

```bash
# set up the uv environment for running Newton examples
uv sync --extra examples

# run an example
uv run -m newton.examples basic_pendulum
```

### Run tests

```bash
# install development extras and run tests
uv run --extra dev -m newton.tests

# include tests that require PyTorch
uv run --extra dev --extra torch-cu12 -m newton.tests

# run a specific example test
uv run --extra dev -m newton.tests.test_examples -k test_basic.example_basic_shapes
```

### Pre-commit (lint/format hooks)

```bash
uvx pre-commit run -a
uvx pre-commit install
```

### Benchmarks (ASV) via `uv`

> Note: ASV needs to be available in the environment; prefer running it through `uv` so the tool and deps are consistent.

```bash
# Unix shells
uv run --group dev asv run --launch-method spawn main^!

# Windows CMD (escape ^ as ^^)
uv run --group dev asv run --launch-method spawn main^^!
```

