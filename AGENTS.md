# Newton Guidelines

## Public API and `_src` boundary

- `newton/_src/` is internal. Examples and docs must not import from `newton._src`.
- Expose user-facing symbols via public modules (`newton/geometry.py`, `newton/solvers.py`, etc.).
- Prefer a single public import path: `from newton.geometry import BroadPhaseAllPairs`.

## Breaking API changes

- **Breaking changes require a deprecation first.** Do not remove or rename public API symbols without deprecating them in a prior release.

## Code style

- **Prefix-first naming** for autocomplete: `ActuatorPD` (not `PDActuator`), `add_shape_sphere()` (not `add_sphere_shape()`).
- Methods: `snake_case`. CLI args: `kebab-case` (`--use-cuda-graph`).
- Prefer nested classes for self-contained helper types/enums.
- PEP 8. PEP 604 unions (`x | None`, not `Optional[x]`).
- Annotate Warp arrays with concrete dtypes (`wp.array(dtype=wp.vec3)`).
- Consistent parameter names across base/override APIs (`xforms`, `scales`, `colors`, `materials`).
- **Docstrings**: Google-style. Types in annotations, not docstrings. `Args:` use `name: description`.
  - Sphinx cross-refs (`:class:`, `:meth:`) with shortest possible targets. Prefer public API paths; never use `newton._src`.
  - SI units for physical quantities: `"""Particle positions [m], shape [particle_count, 3]."""`
  - Joint-dependent: `[m or rad]`. Spatial vectors: `[N, N·m]`. Compound arrays: per-component. Skip non-physical fields.
  - Public API docstrings only.
- Run `docs/generate_api.py` when adding public API symbols.

## Examples

- Follow the `Example` class format. Implement `test_final()`; optionally `test_post_step()`.
- Register in `README.md` with uv command and screenshot.

## Dependencies

Avoid new required dependencies. Strongly prefer not adding optional ones — use Warp, NumPy, or stdlib.

## Commands

All commands use `uv`. Fall back to `venv`/`conda` if unavailable. Use `uv run --no-project` for standalone scripts without a `pyproject.toml`.

```bash
# Examples
uv sync --extra examples
uv run -m newton.examples basic_pendulum

# Tests
uv run --extra dev -m newton.tests
uv run --extra dev -m newton.tests -k test_viewer_log_shapes           # specific test
uv run --extra dev -m newton.tests -k test_basic.example_basic_shapes  # example test
uv run --extra dev --extra torch-cu12 -m newton.tests                  # with PyTorch

# Lint (run BEFORE committing)
uvx pre-commit run -a

# Benchmarks
uvx --with virtualenv asv run --launch-method spawn main^!
```

## Changelog

- **Update `CHANGELOG.md` for every user-facing change** targeting the `[Unreleased]` section.
- Use **imperative present tense**: "Add X", not "Added X" or "This adds X".
- Place entries under the correct category: `Added`, `Changed`, `Deprecated`, `Removed`, or `Fixed`.
- Avoid internal implementation details users wouldn't understand.
- **For `Deprecated`, `Changed`, and `Removed` entries, include migration guidance.** Tell users what replaces the old behavior so they can update their code.
  - Example: "Deprecate `Model.geo_meshes` in favor of `Model.shapes`" or "Remove `build_model()` — use `ModelBuilder.finalize()` instead".

## Commits and PRs

- **Branches**: `<username>/feature-desc`. Never commit to `main`.
- **Commits**: imperative subject (~50 chars), blank line, body wraps at 72 chars explaining _what_ and _why_.
- **No AI attribution** in commits or PR descriptions.
- **Iterate with new commits**, not amends, so reviewers see incremental changes.
- Verify regression tests fail without the fix before committing.

## File headers

```
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
```

Do not change years in existing headers.

## CI/CD

Pin GitHub Actions by SHA: `action@<sha>  # vX.Y.Z`. Check `.github/workflows/` for allowlisted hashes.
