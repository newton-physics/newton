# Newton Development Guidelines

- `newton/_src/` is internal. Examples and docs must not import from `newton._src`. Expose user-facing symbols via public modules (`newton/geometry.py`, `newton/solvers.py`, etc.).
- Breaking changes require a deprecation first. Do not remove or rename public API symbols without deprecating them in a prior release.
- Prefix-first naming for autocomplete: `ActuatorPD` (not `PDActuator`), `add_shape_sphere()` (not `add_sphere_shape()`).
- Prefer nested classes for self-contained helper types/enums.
- PEP 604 unions (`x | None`, not `Optional[x]`).
- Annotate Warp arrays with bracket syntax (`wp.array[wp.vec3]`, `wp.array2d[float]`, `wp.array[Any]`), not the parenthesized form (`wp.array(dtype=...)`). Use `wp.array[X]` for 1-D arrays, not `wp.array1d[X]`.
- Follow Google-style docstrings. Types in annotations, not docstrings. `Args:` use `name: description`.
  - Sphinx cross-refs (`:class:`, `:meth:`) with shortest possible targets. Prefer public API paths; never use `newton._src`.
  - SI units for physical quantities in public API docstrings: `"""Particle positions [m], shape [particle_count, 3]."""`. Joint-dependent: `[m or rad]`. Spatial vectors: `[N, N·m]`. Compound arrays: per-component. Skip non-physical fields.
- Code comments: brief, and only for non-obvious code. Explain *why* (intent, constraints, edge cases), not *what* the code already shows. Prefer a cross-reference (doc, `:class:`/`:meth:`) over re-explaining context.
- Run `docs/generate_api.py` when adding public API symbols.
- Before relying on or changing a documented claim, open the relevant internal cross-references and external primary-source links. Verify Newton-specific behavior against the current code; if a linked source is unavailable, state that limitation instead of assuming it supports the claim.
- Avoid new required dependencies. Strongly prefer not adding optional ones — use Warp, NumPy, or stdlib.
- Create a feature branch before committing — never commit directly to `main`. Use `<username>/feature-desc`.
- Imperative mood in commit messages ("Fix X", not "Fixed X"), ~50 char subject, body wraps at 72 chars explaining _what_ and _why_.
- Verify regression tests fail without the fix before committing.
- Pin GitHub Actions by SHA: `action@<sha>  # vX.Y.Z`. Check `.github/workflows/` for allowlisted hashes.
- In SPDX copyright lines, use the year the file was first created. Do not create date ranges or update the year when modifying a file.

Run `uvx pre-commit run -a` to lint/format before committing. Use `uv` for all commands; fall back to `venv`/`conda` if unavailable.

```bash
# Examples
uv sync --extra examples
uv run -m newton.examples basic_pendulum
```

## Tests

Always use `unittest`, not pytest.

```bash
uv run --extra dev -m newton.tests
uv run --extra dev -m newton.tests -k test_viewer_log_shapes           # specific test
uv run --extra dev -m newton.tests -k test_basic.example_basic_shapes  # example test
uv run --extra dev --extra torch-cu12 -m newton.tests                  # with PyTorch
```

### Testing guidelines

- Give every test function or method a docstring using triple double quotes (`"""..."""`). Start with a concise one-line summary in imperative mood that states what the test verifies. For a particularly complex test, add a body that elaborates on the tested behavior, separated from the summary by a blank line following Google-style docstring conventions.
- Never call `wp.synchronize()` or `wp.synchronize_device()` right before `.numpy()` on a Warp array. This is redundant as `.numpy()` performs a synchronous device-to-host copy that completes all outstanding work.

```bash
# Benchmarks
uvx --with virtualenv asv run --launch-method spawn main^!
```

## PR Instructions

- If opening a pull request on GitHub, use the template in `.github/PULL_REQUEST_TEMPLATE.md`.
- Every normal pull request must add exactly one uniquely named file under `changelog.d/`. Use a `.md` fragment for
  user-facing changes or a `.skip` fragment containing a one-line reason when there is no user-facing change. Generate
  it before opening the pull request with `scripts/changelog.py create`; never wait for a pull request number or rename
  the file to include one. The command derives a readable slug from the supplied name (or current branch) and appends a
  random identifier. See `changelog.d/README.md` for commands and the exact format.
- A `.md` fragment may contain multiple `### Added`, `### Changed`, `### Deprecated`, `### Removed`, and `### Fixed`
  sections in that exact order. Start every entry with `- `, end every entry with a period `.`, use imperative present tense ("Add X"), and avoid internal
  implementation details. For `Deprecated`, `Changed`, and `Removed` entries, include migration guidance: "Deprecate
  `Model.geo_meshes` in favor of `Model.shapes`".
- Do not edit `CHANGELOG.md` or its generated `changelog-fragment` comments directly in a normal pull request. Maintainers
  may consolidate pending fragments into `[Unreleased]` at any time with `scripts/changelog.py build`. Final releases use
  `scripts/changelog.py release`, which strips provenance comments from the dated public section; post-release
  synchronization to `main` uses `scripts/changelog.py reconcile`. Add `--dry-run` to any of these commands to preview
  without editing files or consuming fragments. Commit each mutating operation in a changelog-only pull request labeled
  `changelog-maintenance`.
- Run `uv run --no-project python scripts/changelog.py validate` after editing fragments.
- If you want to annotate a changelog entry with a GitHub link following the #xxxx format, you must use the number of the pull request, not the issue number.

## Examples

- Follow the `Example` class format.
  - Implement `test_final()` — runs after the example completes to verify simulation state is valid.
  - Optionally implement `test_post_step()` — runs after every `step()` for per-step validation.
- Register in `README.md` with `python -m newton.examples <name>` command and a 320x320 jpg screenshot.
