# Newton Guidelines

## Public API and `_src` boundary

- **`newton/_src/` is internal library implementation only.**
  - User code, that means Newton examples (under `newton/examples/`) and documentation, **must not** import from `newton._src`.
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
  - If a helper type or an enum is only meaningful inside one parent class and doesn't need a public identity, define it as a nested class instead of creating a new top-level class/module.
- **Follow PEP 8 for Python code.**
- **Use Google-style docstrings.**
  - Write clear, concise docstrings that explain what the function does, its parameters, and its return value.
- **Keep the documentation up-to-date.**
  - When adding new files or symbols that are part of the public-facing API, make sure to keep the auto-generated documentation updated by running `docs/generate_api.py`.
- **Add examples to README.md**
  - When contributing a new Newton example you must follow the format of the existing examples, where we have an `Example` class. Then register the example in the appropriate table in `README.md` with the corresponding uv run command and a screenshot.
  - Ensure your example implements a meaningful `test_final()` method that is executed after the example has been run to verify the state of the simulation is valid.
  - Optionally you may implement a `test_post_step()` method that is evaluated after every `step()` of the example.

## Dependencies

- **Avoid adding new required dependencies.** Newton's core should remain lightweight and minimize external requirements.
- **Strongly prefer not adding new optional dependencies.** If additional functionality requires a new package, carefully consider whether the benefit justifies the added complexity and maintenance burden. When possible, implement functionality using existing dependencies, including Warp functions and kernels, NumPy, or the standard library.

## Tooling: prefer `uv` for running, testing, and benchmarking

We standardize on `uv` for local workflows when available. If `uv` is not installed, fall back to a virtual environment created with `venv` or `conda`.

Example commands using `uv` (from `docs/guide/development.rst`):

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

### Benchmarks (ASV)

```bash
# Unix shells
uvx --with virtualenv asv run --launch-method spawn main^!

# Windows CMD (escape ^ as ^^)
uvx --with virtualenv asv run --launch-method spawn main^^!
```

## Commit and Pull Request Guidelines

Follow conventional commit message practices.
- Use clear, descriptive commit messages that explain what changed and why.
- Keep commits focused and atomic—one logical change per commit.
- Reference related issues in commit messages when applicable.

For detailed guidance on writing good commit messages and structuring pull requests, see [Apache Airflow's Pull Request Guidelines](https://github.com/apache/airflow/blob/main/AGENTS.md#pull-request-guidelines).
