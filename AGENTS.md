# Newton Agent Notes

## Project Shape

- Newton is a Python 3.10+ GPU physics engine built on `warp-lang`; the local default Python is `.python-version` 3.12.
- Public API is re-exported from `newton/__init__.py` and public modules such as `newton/geometry.py`, `newton/solvers.py`, `newton/viewer.py`; `newton/_src/` is internal and must not be imported from examples or docs.
- Solver implementation packages live under `newton/_src/solvers/`; public solver access goes through `newton.solvers` and generated API docs include only exported/public symbols.
- `CLAUDE.md` only includes this file, so keep all shared agent guidance here.

## Commands

- Install/update dev env with uv: `uv sync --extra dev`; examples need `uv sync --extra examples` or `uv run --extra examples ...`.
- Full unit suite: `uv run --extra dev -m newton.tests`.
- Focused test: `uv run --extra dev -m newton.tests -k test_viewer_log_shapes`.
- Focused example test: `uv run --extra dev -m newton.tests -k test_basic.example_basic_shapes`.
- Torch/CUDA tests use explicit extras, e.g. `uv run --extra dev --extra torch-cu12 -m newton.tests`.
- CI test command adds cache/report flags: `uv run --extra dev -m newton.tests --no-cache-clear --junit-report-xml rspec.xml --coverage --coverage-xml coverage.xml`.
- Validate lockfile freshness with `uv lock --check`; pre-commit runs `uv-lock` locally but pre-commit.ci skips it because it needs network access.
- Lint/format before commits with `uvx pre-commit run -a`; hooks run Ruff fix/format, `uv-lock`, typos, and `scripts/check_warp_array_syntax.py`.
- Build docs like CI: `uv run --extra docs --extra sim sphinx-build -j auto -W -b html docs docs/_build/html` and doctests with `uv run --extra docs --extra sim sphinx-build -j auto -W -b doctest docs docs/_build/doctest`.
- Run `uv run --extra docs --extra sim python docs/generate_api.py` after adding public API symbols, then ensure `git diff --exit-code docs/api/` is clean.
- Run examples through the package entrypoint, e.g. `uv run --extra examples -m newton.examples basic_pendulum`; example test mode is `--test` and a non-rendering viewer is `--viewer null`.
- Benchmarks use ASV with virtualenv: `uvx --with virtualenv asv run --launch-method spawn main^!`.

## API And Style Rules

- Breaking public API changes require a deprecation first; do not remove or rename public symbols without a prior deprecation release.
- Prefix-first naming is preferred for autocomplete, e.g. `ActuatorPD` and `add_shape_sphere()`.
- Prefer nested classes for self-contained helper types/enums.
- PEP 604 unions (`x | None`, not `Optional[x]`).
- Annotate Warp arrays with bracket syntax (`wp.array[wp.vec3]`, `wp.array2d[float]`, `wp.array[Any]`), not the parenthesized form (`wp.array(dtype=...)`). Use `wp.array[X]` for 1-D arrays, not `wp.array1d[X]`.
- Ruff bans heavy optional dependencies as module-level imports in `newton/_src` and `newton/tests`; import them lazily where needed. Examples and docs are exempt from this rule.
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

## Tests And Examples

- Tests are `unittest`-based via `newton.tests`; do not add pytest tests.
- Do not call `wp.synchronize()` or `wp.synchronize_device()` immediately before `.numpy()`; `.numpy()` already performs the synchronous device-to-host copy.
- Example files are discovered from `newton/examples/*/example_*.py`; the short CLI name strips `example_` and `.py`.
- New examples should follow the `Example` class pattern and implement `test_final()`; use `test_post_step()` only when per-step validation is needed.
- If adding a README-listed example, register it in `README.md` with a `python -m newton.examples <name>` command and a 320x320 JPG screenshot.

## Changes, PRs, And Releases

- Avoid new required dependencies; strongly prefer Warp, NumPy, or stdlib before adding optional dependencies.
- If a user-facing behavior changes, add a random-position entry under the correct `[Unreleased]` `CHANGELOG.md` category using imperative present tense; `Deprecated`, `Changed`, and `Removed` entries need migration guidance.
- Use `.github/PULL_REQUEST_TEMPLATE.md` when opening a PR.
- Commit only on a feature branch, never directly on `main`; branch names should look like `<username>/feature-desc`.
- Commit messages use imperative mood, roughly 50-character subjects, and bodies wrapped near 72 columns explaining what and why.
- Before committing a bug fix, verify the regression test fails without the fix when feasible.

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

When the user types `/graphify`, invoke the `skill` tool with `skill: "graphify"` before doing anything else.

- Give every test function or method a docstring using triple double quotes (`"""..."""`). Start with a concise one-line summary in imperative mood that states what the test verifies. For a particularly complex test, add a body that elaborates on the tested behavior, separated from the summary by a blank line following Google-style docstring conventions.
- Never call `wp.synchronize()` or `wp.synchronize_device()` right before `.numpy()` on a Warp array. This is redundant as `.numpy()` performs a synchronous device-to-host copy that completes all outstanding work.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- Dirty graphify-out/ files are expected after hooks or incremental updates; dirty graph files are not a reason to skip graphify. Only skip graphify if the task is about stale or incorrect graph output, or the user explicitly says not to use it.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
