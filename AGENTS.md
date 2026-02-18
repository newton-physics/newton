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
- **State SI units for all physical quantities in docstrings.**
  - Use inline `[unit]` notation, e.g. `"""Particle positions [m], shape [particle_count, 3], float."""`.
  - For joint-type-dependent quantities use `[m or rad, depending on joint type]`.
  - For spatial vectors annotate both components, e.g. `[N, N·m]`.
  - For compound arrays list per-component units, e.g. `[0] k_mu [Pa], [1] k_lambda [Pa], ...`.
  - When a parameter's interpretation varies across solvers, document each solver's convention instead of a single unit.
  - Skip non-physical fields (indices, keys, counts, flags).
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

- **Use `uv run python -c` for inline Python**: When running one-off Python commands, use `uv run python -c "..."` instead of `python3 -c "..."`.
- **Use `uv run --no-project`** to run standalone Python scripts without a `pyproject.toml` (e.g., in CI after switching to a branch with no project files). Combine with `--with` for one-off tool usage: `uv run --no-project --with yamllint yamllint <file>`.

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

# run a specific test file by name (-k filters by unittest-parallel pattern)
uv run --extra dev -m newton.tests -k test_viewer_log_shapes

# run a specific example test
uv run --extra dev -m newton.tests -k test_basic.example_basic_shapes
```

**Warp kernel cache:**

Use a session-specific cache directory to avoid interference with parallel sessions:
```bash
export WARP_CACHE_ROOT=/tmp/claude/warp-cache-$$
```

Use `--no-cache-clear` to skip clearing the kernel cache for faster turnaround:
```bash
uv run --extra dev -m newton.tests --no-cache-clear -k test_model
```

### Pre-commit (lint/format hooks)

**CRITICAL: Always run pre-commit hooks BEFORE committing, not after.**

Proper workflow:
1. Make your code changes
2. Run `uvx pre-commit run -a` to check ALL files
3. If pre-commit modifies any files (e.g., formatting), review the changes
4. Stage the modified files with `git add`
5. Run `uvx pre-commit run -a` again to ensure all checks pass
6. Only then create your commit with `git commit`

```bash
# Run pre-commit checks on all files
uvx pre-commit run -a

# Install hooks to run automatically on every commit (recommended)
uvx pre-commit install
```

**Common mistake to avoid:**
- ❌ Don't commit first and then run pre-commit (requires amending commits)
- ✅ Do run pre-commit before committing (clean workflow)

### Benchmarks (ASV)

```bash
# Unix shells
uvx --with virtualenv asv run --launch-method spawn main^!

# Windows CMD (escape ^ as ^^)
uvx --with virtualenv asv run --launch-method spawn main^^!
```

## Commit and Pull Request Guidelines

Follow conventional commit message practices.

- **Use feature branches**: All development work should be on branches named `<username>/feature-desc` (e.g., `jdoe/docs-versioning`). Do not commit directly to `main`.
- **CRITICAL: Run pre-commit hooks BEFORE committing**:
  1. Make your changes
  2. Run `uvx pre-commit run -a`
  3. If files are modified by pre-commit (formatting, etc.), stage them with `git add`
  4. Run `uvx pre-commit run -a` again to verify all checks pass
  5. Then commit with `git commit`
  - Consider installing hooks with `uvx pre-commit install` to automate this
  - **Never commit first and run pre-commit after** - this requires amending commits and creates messy history
- Keep commits focused and atomic—one logical change per commit.
- Reference related issues in commit messages when applicable.
- **When iterating on PR feedback**, prefer adding new commits over amending existing ones. This avoids force-pushing and lets the reviewer easily verify each change request was addressed.
- **Do not include AI attribution or co-authorship lines** (e.g., "Co-Authored-By: Claude...") in commit messages. Commits should represent human contributions without explicit AI attribution.
- **Commit message format**:
  - Separate subject from body with a blank line
  - Subject: imperative mood, capitalized, ~50 chars, no trailing period
    - Write as a command: "Fix bug" not "Fixed bug" or "Fixes bug"
    - Test: "If applied, this commit will _[your subject]_"
  - Body: wrap at 72 chars, explain _what_ and _why_ (not _how_—the diff shows that)

## File headers and copyright

- New files must use the current year (2026) in the SPDX copyright header:
  ```
  # SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
  # SPDX-License-Identifier: Apache-2.0
  ```
- Do not change the year in existing file headers.

## Sandbox & Networking

- Network access (e.g., `git push`) is blocked by the sandbox. Use `dangerouslyDisableSandbox: true` so the user gets an approval prompt — don't ask them to run it manually.

## GitHub Actions and CI/CD

- IMPORTANT: Pin actions by SHA hash. Use `action@<sha>  # vX.Y.Z` format for supply-chain security. Check existing workflows in `.github/workflows/` for the allowlisted hashes. New actions or versions require repo admin approval to be added to the allowlist.

## Testing Guidelines

- **Always verify regression tests fail without the fix.** When writing a regression test for a bug fix, temporarily revert the fix and run the test to confirm it fails. Then reapply the fix and verify the test passes. This ensures the test actually covers the bug.

### Debugging Warp kernels

**Do not add `wp.printf` to kernels and run via the test runner.** Newton's test infrastructure captures stdout at the file-descriptor level (`os.dup2`) via `CheckOutput`/`StdOutCapture` in `newton/tests/unittest_utils.py`. By default (`check_output=True`), any unexpected stdout — including `wp.printf` — **causes the test to fail** with `"Unexpected output"`. Tests that opt out with `check_output=False` avoid that failure, but their stdout is still lost because `unittest-parallel` runs tests in spawned child processes.

To debug Warp kernel behavior:

1. **Write a standalone reproduction script** and run it directly with `uv run python -c "..."` or `uv run python script.py`. This keeps stdout visible and avoids the test framework entirely.
2. **Use high-precision format strings** for floating-point debugging (e.g., `wp.printf("val=%.15e\n", x)`) — the default `%f` format hides values smaller than ~1e-6 that can still affect control flow.
3. **Remove debug prints before committing.** `wp.printf` in kernels affects performance and will cause `check_output=True` tests to fail.
