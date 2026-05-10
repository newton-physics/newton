# Classification Helpers

This reference is loaded during Phases 3, 4, and 5 of the skill. It defines concrete path and naming rules Claude relies on when analyzing commits and CHANGELOG entries.

## Public API surface (Phase 4a, 4e)

Used to decide whether a symbol is "genuinely new" vs. "pre-existed and got extended":

Newton does NOT expose its public surface through a single `__init__.py` (unlike Warp). It uses a set of per-topic public re-export modules. The authoritative list lives in `docs/generate_api.py` under `MODULES`:

- `newton/__init__.py` — top-level (`newton.Model`, `newton.State`, `newton.Axis`, etc.)
- `newton/geometry.py`
- `newton/ik.py`
- `newton/math.py`
- `newton/selection.py`
- `newton/sensors.py`
- `newton/solvers.py`
- `newton/usd.py`
- `newton/utils.py`
- `newton/viewer.py`

Each public module re-exports from `newton/_src/<topic>/...`. `newton._src` is internal (AGENTS.md: "Examples and docs must not import from `newton._src`").

**To determine if `newton.X` existed at base**:
- For top-level symbols: `git show <base>:newton/__init__.py` and check `from ._src.<submodule> import X` / `__all__`.
- For submodule public attributes (e.g., `newton.geometry.Mesh`): `git show <base>:newton/geometry.py` and check imports / `__all__`.
- For method additions on an existing class (e.g., `SolverXPBD.update_contacts`): resolve the class's real source file (e.g., `newton/_src/solvers/xpbd.py`) and `ast`-walk it at base.

**Public-API exposure check (Phase 4a addition)**: for every symbol that is genuinely new, verify at HEAD that it is reachable via at least one public module. If the symbol lives only in `newton._src.<path>` and is not re-exported, raise a 🕵️ Private-only flag. Reason: AGENTS.md forbids examples/docs from importing `newton._src`, so a user-facing symbol that is not re-exported is unusable by Newton's own examples and will churn.

## Newton has no kernel-scope builtin registry

Warp's release-audit walks `warp/_src/builtins.py` for `add_builtin("<name>", ...)` calls. Newton has no such registry. All user-facing Newton symbols are ordinary Python (classes, functions, enums, constants) defined in `newton/_src/**` and re-exported through the public modules above. Skip every "kernel-scope" codepath from the Warp version of this skill.

## Paths that trigger Phase 4f semantic-change review

Commits touching these paths get per-commit judgment (Phase 4f). Not every change in them is a semantic shift — Claude reads the diff and decides:

- `newton/_src/solvers/**` — solver implementations (XPBD, MuJoCo, Featherstone, VBD, implicit MPM). Changes can alter convergence, contact handling, step semantics.
- `newton/_src/sim/**` — integrators, collision pipeline, model building, state transfer. Changes can alter per-step dynamics or contact ordering.
- `newton/_src/math/**` — math helpers, raycast, quaternion utilities. Changes can shift numerical output.
- `newton/_src/geometry/**` — geometry primitives, SDF / mesh representations, support functions. Changes can shift contact normals, distances, or inside/outside tests.

Typical NOT-SEMANTIC-SHIFTING signals in these paths:
- Pure internal refactors, renames of internal identifiers.
- Comment or formatting changes.
- Performance optimizations that preserve observable output (e.g., vectorization, precomputed AABBs).
- Caching / memoization where the cached value is equivalent.
- Test-only changes.
- Build-system touches.
- Bug fixes where the previous behavior was demonstrably wrong.

Typical SEMANTIC-SHIFTING signals:
- Algorithm swaps that produce different numerical results (e.g., switching SDF construction from winding-number to parity).
- Default parameter value changes that affect physics (e.g., changing MPM quadrature, contact stiffness, solver iteration defaults).
- Ordering / determinism changes that affect contact reduction or constraint solving.
- Convergence criteria changes (different tolerance, different iteration bound behavior).
- Unit / frame-of-reference changes for authored values (e.g., "stop multiplying joint damping by 180/π").
- Control-flow changes that alter when callbacks / registered hooks / user kernels are invoked.

**Never attempt to build and run Newton at base vs. HEAD for verification.** Newton's output depends on Warp codegen, MuJoCo, and GPU state; reliable head-to-head numerical verification in a one-shot audit is out of scope. When in doubt, route the commit to a "Review candidates" section in the report's review notes rather than speculating in Breaking Changes.

## Heuristic paths commonly relevant (reference only)

These are noted here so Claude can pattern-match when reading commits, but the skill does NOT use them to produce an appendix-style commit audit. Purpose is recognition, not bucketing:

- `.github/**`, `.pre-commit-config.yaml`, `uv.lock`, `.python-version` — infrastructure, not user-facing.
- `asv.conf.json`, `asv/**`, root-level `_bench_*.py` — benchmark harness.
- `docs/**`, root-level `*.md`, `CHANGELOG.md` — documentation.
- `newton/examples/**` — example scripts. New files here are user-facing (a new example is a release-notable addition). Changes to existing examples are typically not release-notable unless they change the example's registered name or behavior.
- `newton/_src/**` other than the solver / sim / math / geometry paths above — internal Python implementation.
- `pyproject.toml` — dependency / version changes. Version bumps are not release-notable on their own; dependency changes may belong in "Behavioral & Support Changes" if a user-visible pin moves (e.g., `mujoco-warp ~=3.7.0`).

## Newton-specific rename and parameter-reorder recognition (Phase 4d, 5a)

Newton's CHANGELOG does NOT use the `**Breaking:**` literal marker. Instead, migration-required changes appear in `### Changed` with prose like:

- "Rename `X.old_name` to `X.new_name`. Old name still accepted as keyword argument but emits a `DeprecationWarning`."
- "Reorder `X()` parameters so `a` precedes `b`."
- "Migrate all Y logic to Z, all Y functions now return ..."

When Phase 4d / 5a encounter these patterns, treat them as migration-required changes (Kind `rename` or `parameter reorder` in the Changes-to-Existing-API table). Check that the entry includes migration guidance (`Use ...`, `in favor of ...`, `prefer ...`). If guidance is missing, raise a 📐 flag in the language review.

## Deprecation policy (Phase 4d)

AGENTS.md: "Breaking changes require a deprecation first." This means every `### Removed` entry should correspond to a prior `### Deprecated` entry in an earlier release.

When Phase 4d cannot find the prior Deprecated entry:
- Surface as `🚨 Policy: removed without prior deprecation` in the Breaking Changes section.
- Cite the Removed entry in full.
- Note whether the current CHANGELOG's own `### Deprecated` section also names the same symbol. Deprecating and removing in the same release is a policy violation: the feature needs to ship in at least one release with the Deprecated entry before removal.

A release manager reading the report should be able to resolve the flag in one of three ways:
1. Point to the prior Deprecated entry that Claude missed (Claude re-scans).
2. Confirm the symbol was deprecated via runtime `DeprecationWarning` but the CHANGELOG entry was forgotten.
3. Block the release / add a deprecation shim for one more release cycle.

Claude does NOT decide which path is correct; it surfaces the gap with enough evidence for the release manager to decide.
