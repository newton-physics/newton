# CHANGELOG Language Review Examples

Loaded during Phase 5 of the skill. Use these examples to calibrate the language-review pass: flagging entries whose language would confuse end users, whose refs look wrong, or whose content violates Newton's conventions.

## What to flag

### 🗣️ Internal / implementation language

Flag entries that reference:

- **Internal module paths**: `newton._src.foo`, `newton._src.solvers.xpbd.internal_helper`.
- **Warp-internal helpers** that leaked into user-facing prose: e.g. `warp.fem.geometry.closest_point` (a Newton internal refactor may cite this; it is internal to Warp).
- **Private identifiers** with leading underscores: `_foo`, `Model._finalize`.
- **Implementation-detail verbs** without a user-visible outcome: "Refactor internal dispatch path", "Reorganize private helpers", "Rewrite foo bar".

Good examples (user-facing):

> Add `newton.geometry.compute_offset_mesh()` for extracting offset surface meshes from any collision shape, and a viewer toggle to visualize gap + margin wireframes in the GL viewer.
>
> Use pre-computed local AABB for `CONVEX_MESH` shapes in `compute_shape_aabbs`, avoiding a per-frame support-function AABB computation.

Flag these:

> Inline a `wp.vec3`-specialized point-to-triangle squared-distance helper in the implicit-MPM rasterized collider, removing the dependency on Warp's internal `warp.fem.geometry.closest_point`.
> *Reason:* references `warp.fem.geometry.closest_point`, which is Warp-internal. User-facing rewrite: "Inline point-to-triangle distance in the implicit-MPM collider so it no longer depends on Warp's `warp.fem` module."
>
> Refactor `newton._src.solvers.xpbd._update_constraints` to unify storage path.
> *Reason:* references internal module + private method. Implementation detail with no user-observable effect. Candidate for deletion from CHANGELOG, not rewording.

### 📝 Too terse

Flag entries that are too short to convey meaning or that omit enough context for a user to act on:

- Under ~10 words AND no linked issue for context.
- Entries like "Fix bug in XPBD solver" without specifying which bug or what the fix does.

Flag:

> Fix bug.
> *Reason:* insufficient — user can't tell what was fixed.
>
> Improve solver performance.
> *Reason:* no specifics; which solver, how much, under what conditions?

Don't flag (long but load-bearing):

> Fix MJCF importer in `compiler.angle="degree"` mode: (1) stop multiplying joint `damping`/`stiffness` by `180/π` (MuJoCo stores these in `N·m·s/rad` and `N·m/rad` regardless of `angle`); (2) stop `deg2rad`-scaling the default `±MAXVAL` sentinel for joints without an explicit `range=`.

### 🔗 Suspected wrong GH reference

Flag when the CHANGELOG entry's topic doesn't match the commits that cite that GH number.

**Tier-1 heuristic (always on, fully local):**

For entry "Add support for Gaussian splats (GH-NNNN)", fetch commits tagged `GH-NNNN` and look at their subjects and file paths. If every commit only touches `.github/**` or `docs/**`, the GH ref is likely wrong — the entry describes a model-builder API change, but nothing in those commits modifies the builder or geometry. Flag.

Don't flag if:
- Commits touch `newton/_src/geometry/**` or `newton/_src/sim/builder.py` for a geometry-addition entry → topic matches.
- Commits touch `docs/**` and the entry is in the Documentation section → topic matches.
- Commits touch `newton/_src/solvers/**` for a solver-capability entry → topic matches.

**Tier-2 heuristic (only if `gh` CLI is installed + authenticated):**

For each GH ref, run `gh issue view <num> --json title,body`. Compare the issue's title/topic to the entry's description. If clearly unrelated (e.g., issue is "Improve MuJoCo solver performance", entry is "Add USD tetmesh import"), flag.

Skip tier-2 silently if `gh` is absent or auth fails.

### 🕵️ Private-only symbol

Flag `### Added` entries whose named public symbol exists only under `newton._src.*` at HEAD and is not re-exported through a public module (see `classification-rules.md` for the module list).

Example flag:

> Add `newton.geometry.compute_offset_mesh()` for extracting offset surface meshes.
> *Check:* grep `newton/geometry.py` at HEAD. If `compute_offset_mesh` is not re-exported, raise 🕵️. If it is (the expected case), no flag.

Rationale: AGENTS.md forbids examples and docs from importing `newton._src`. A user-facing symbol that lives only in `_src` cannot be used by Newton's own examples and is a maintenance liability.

### 📐 Missing migration guidance (Newton-specific)

AGENTS.md: "For `Deprecated`, `Changed`, and `Removed` entries, include migration guidance: 'Deprecate `Model.geo_meshes` in favor of `Model.shapes`'."

Flag `### Deprecated`, `### Removed`, and `### Changed` entries that name a rename / removal / reorder but do NOT express migration direction. Direction can be expressed with one of these phrases (case-insensitive): `use`, `in favor of`, `renamed to`, `replaced by`, `switch to`, `migrate to`, `prefer`. Direction can also be expressed structurally — an imperative `Rename X to Y`, an arrow `X → Y`, or a parameter-rename table — which counts as migration guidance even when none of the listed phrases appear verbatim (see the `ModelBuilder.add_shape_ellipsoid` example below).

Flag:

> Remove `SolverXPBD.legacy_substep()`.
> *Reason:* no migration guidance. User does not know which method replaces it.
>
> Deprecate `Model.geo_meshes`.
> *Reason:* no replacement named. Should read "Deprecate `Model.geo_meshes` in favor of `Model.shapes`."

Don't flag:

> Deprecate `SensorContact.net_force` in favor of `SensorContact.total_force` and `SensorContact.force_matrix`
> *Reason:* migration guidance present ("in favor of ...").
>
> Rename `ModelBuilder.add_shape_ellipsoid()` parameters `a`, `b`, `c` to `rx`, `ry`, `rz`. Old names are still accepted as keyword arguments but emit a `DeprecationWarning`
> *Reason:* rename is explicit and the old-name behavior is documented.

### 🏷️ Naming-convention drift (Newton-specific)

AGENTS.md: "Prefix-first naming for autocomplete: `ActuatorPD` (not `PDActuator`), `add_shape_sphere()` (not `add_sphere_shape()`)."

Flag `### Added` entries whose newly-named public symbol puts the discriminator before the prefix. Examples of names to flag:

- `PDActuator`, `VelocityActuator` → should be `ActuatorPD`, `ActuatorVelocity`
- `SphereShape`, `CapsuleShape` → should be `ShapeSphere`, `ShapeCapsule`
- `add_sphere_shape()`, `add_mesh_shape()` → should be `add_shape_sphere()`, `add_shape_mesh()`

Before flagging, cross-check against existing sibling symbols in the same module. If the rest of the module uses `Foo<Kind>` rather than `<Kind>Foo`, the new symbol should match the established pattern whichever direction it goes. Prefer-consistency beats prefer-the-rule-in-AGENTS.md when the module has an entrenched local convention.

## Judgment philosophy

**Err on "mention, don't block"**: flagging should raise a question for human review, not gate the report. The audit appendix shows flagged entries and a one-line reason; a human decides.

**Don't auto-rewrite**: Claude flags the entry, never modifies it. The release manager updates CHANGELOG.md manually.

**Prefer false positives over false negatives**: a flag that turns out to be fine costs a 5-second eyeball. A missed wrong-ref or jargon-leak ships to users.

**Exception for 🚨 policy flags.** The `🚨 Policy: removed without prior deprecation` flag (Phase 4d) is NOT a false-positive-tolerant flag. If Claude raises it, it is because an exhaustive search of prior released sections failed to find a Deprecated entry for the symbol. The release manager needs a clear, non-hedged signal, not a "maybe verify" line. Only raise this flag when the search truly came back empty.

## Row format (for the report)

| Entry (excerpt) | Flag | Why |
|---|---|---|
| "Inline a `wp.vec3`-specialized ..." | 🗣️ Internal language | References `warp.fem.geometry.closest_point` in user-facing prose |
| "Fix crash" | 📝 Too terse | 2 words, no context link |
| "Add `newton.foo.bar` (GH-NNNN)" | 🔗 Wrong ref? | Commits tagged GH-NNNN touch only CI files |
| "Deprecate `Model.foo`" | 📐 Missing migration guidance | No "in favor of" replacement named |
| "Add `PDActuator`" | 🏷️ Naming-convention drift | Should be `ActuatorPD` per prefix-first rule |
| "Add `newton.utils._x`" | 🕵️ Private-only | Named symbol lives only in `newton._src`, not re-exported |

Entry excerpt should be ~60-80 chars so the table stays scannable. Keep the full entry text in the detail sections (never truncate there).
