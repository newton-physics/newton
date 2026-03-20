# Newton Guidelines

- `newton/_src/` is internal. Examples and docs must not import from `newton._src`. Expose user-facing symbols via public modules (`newton/geometry.py`, `newton/solvers.py`, etc.).
- Breaking changes require a deprecation first. Do not remove or rename public API symbols without deprecating them in a prior release.
- Prefix-first naming for autocomplete: `ActuatorPD` (not `PDActuator`), `add_shape_sphere()` (not `add_sphere_shape()`).
- Prefer nested classes for self-contained helper types/enums.
- PEP 604 unions (`x | None`, not `Optional[x]`). Annotate Warp arrays with concrete dtypes (`wp.array(dtype=wp.vec3)`).
- Google-style docstrings. Types in annotations, not docstrings. `Args:` use `name: description`.
  - Sphinx cross-refs (`:class:`, `:meth:`) with shortest possible targets. Prefer public API paths; never use `newton._src`.
  - SI units for physical quantities in public API docstrings: `"""Particle positions [m], shape [particle_count, 3]."""`. Joint-dependent: `[m or rad]`. Spatial vectors: `[N, N·m]`. Compound arrays: per-component. Skip non-physical fields.
- Run `docs/generate_api.py` when adding public API symbols.
- Avoid new required dependencies. Strongly prefer not adding optional ones — use Warp, NumPy, or stdlib.
- Create a feature branch before committing — never commit directly to `main`. Use `<username>/feature-desc`.
- Imperative mood in commit messages ("Fix X", not "Fixed X"), ~50 char subject, body wraps at 72 chars explaining _what_ and _why_.
- Verify regression tests fail without the fix before committing.
- Pin GitHub Actions by SHA: `action@<sha>  # vX.Y.Z`. Check `.github/workflows/` for allowlisted hashes.

Run `uvx pre-commit run -a` to lint/format before committing. Use `uv` for all commands; fall back to `venv`/`conda` if unavailable.

```bash
# Examples
uv sync --extra examples
uv run -m newton.examples basic_pendulum

# Tests
uv run --extra dev -m newton.tests
uv run --extra dev -m newton.tests -k test_viewer_log_shapes           # specific test
uv run --extra dev -m newton.tests -k test_basic.example_basic_shapes  # example test
uv run --extra dev --extra torch-cu12 -m newton.tests                  # with PyTorch

# Benchmarks
uvx --with virtualenv asv run --launch-method spawn main^!
```

## PR Instructions

- If opening a pull request on GitHub, use the template in `.github/PULL_REQUEST_TEMPLATE.md`.
- If a change modifies user-facing behavior, append an entry at the end of the correct category (`Added`, `Changed`, `Deprecated`, `Removed`, `Fixed`) in `CHANGELOG.md`'s `[Unreleased]` section. Use imperative present tense ("Add X") and avoid internal implementation details.
- For `Deprecated`, `Changed`, and `Removed` entries, include migration guidance: "Deprecate `Model.geo_meshes` in favor of `Model.shapes`".

## Examples

- Follow the `Example` class format.
  - Implement `test_final()` — runs after the example completes to verify simulation state is valid.
  - Optionally implement `test_post_step()` — runs after every `step()` for per-step validation.
- Register in `README.md` with uv command and a 320x320 jpg screenshot.
