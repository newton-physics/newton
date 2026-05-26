# Newton {{VERSION_STRING}} {{REPORT_KIND}} Report
Generated: {{REPORT_DATE}}

<!-- {{REPORT_KIND}} is "Pre-Release", "Release Candidate", or "Retrospective",
     chosen in Phase 1 from the argument, version string, and head ref.
     {{VERSION_STRING}} is the raw version (e.g. "1.2.0.dev0" for pre-release,
     "1.2.0rc1" for RC, "1.1.0" for retrospective). -->

- Mode: {{MODE_DESCRIPTION}}
- Head: {{HEAD_REF}} @ `{{HEAD_SHA_SHORT}}`
- Base: {{BASE_REF}} @ `{{BASE_SHA_SHORT}}`
- Commits in range: {{N_COMMITS}}

<!-- {{MODE_DESCRIPTION}} is a one-liner:
     Pre-release: "Pre-release audit of unreleased work on main"
     RC: "Release candidate readiness review (release branch cut)"
     Retrospective: "Retrospective audit of shipped release v<version>; calibration
                     pass cross-references Claude's flags against post-target history" -->


**Headline counts**

- {{N_NEW_API}} new public APIs
- {{N_BREAKING}} breaking changes
- {{N_CHANGED}} changes to existing API
- {{N_BEHAVIORAL}} behavioral / support changes
- {{N_FIXED}} fixes
- {{N_EXAMPLES}} new examples

**Bake distribution**

| Bucket | Commits |
|---|---:|
| 🟢 > 14 days in main | {{N_BAKE_GREEN}} |
| 🟡 7 to 14 days | {{N_BAKE_YELLOW}} |
| 🟠 < 7 days | {{N_BAKE_ORANGE}} |

{{ANOMALY_BANNER_IF_ANY}}

<!-- Anomaly banner appears ONLY when any commit has main_equivalent_sha: null. Example:
     > ⚠️ **N commits in the release have no equivalent on main. Investigate: these
     > shipped without nightly/main-branch bake.**
-->

---

## Release highlights

{{HEADLINE_SUMMARY}}

<!-- Claude's qualitative synthesis of what would land in the official release
     notes. Drafted in Phase 7a. NOT release notes: a reviewer's preview so the
     release manager can see at a glance whether the real release notes will
     match expectations and spot items that need a keep/defer decision.

     Shape: one short intro paragraph (2-3 sentences) followed by 4-8 bulleted
     highlight items. Each bullet starts with a bold 2-6 word headline, then a
     colon and a one-sentence rationale (what it is and why it matters).

     Include risk flags inline when they apply:
       - 🟠 `N days` bake (if the headline item's minimum bake is < 7 days)
       - ⚠️ Experimental (if the CHANGELOG entry carries an "Experimental" marker)
       - ⚠️ Breaking (if the headline item is a breaking change)

     Do NOT include counts ("4 new APIs were added"): that's already in the
     headline counts block above. Highlights are qualitative.

     Example shape (do NOT copy verbatim; pick items that are actually headline
     material in THIS release):

     Newton 1.2 lands Gaussian splat geometry, deterministic contact ordering, and
     a new texture-based SDF pipeline that replaces NanoVDB for mesh-mesh collision.
     Two migration-required changes (add_shape_ellipsoid parameter rename and
     add_shape_gaussian parameter reorder) need migration notes in the release post.

     - **Gaussian splat geometry** ([GH-NNN](...)): `Gaussian`, `ModelBuilder.add_shape_gaussian()`,
       and USD import unlock radiance-field-style rendering as a first-class shape.
       🟢 30 days bake.
     - **Deterministic contact ordering** ([GH-NNN](...)): `deterministic` flag on
       `CollisionPipeline` and `NarrowPhase` makes contact output GPU-thread-schedule
       independent via radix sort and fingerprint tiebreaking. Required for reproducible
       RL training. 🟢 22 days bake.
     - **Texture-based mesh SDF** ([GH-NNN](...), [GH-NNN](...)): replaces NanoVDB
       volumes in the mesh-mesh collision pipeline; faster and CPU-compatible. Every
       GH ref is an individual hyperlink; never collapse to "(multiple GHs)" or a
       plain-text comma list.
     - **⚠️ Breaking: `ModelBuilder.add_shape_ellipsoid` parameters renamed** ([GH-NNN](...)):
       `a`, `b`, `c` → `rx`, `ry`, `rz`. Old names accepted as keyword arguments with
       `DeprecationWarning`. Needs a migration snippet.
     - **⚠️ Breaking: `add_shape_gaussian()` parameter order** ([GH-NNN](...)): `xform`
       now precedes `gaussian`, matching other `add_shape_*` methods. Positional callers
       must switch to keyword form.
     - **⚠️ Experimental: differentiable rigid contacts** ([GH-NNN](...)): gradients
       with respect to body poses via `CollisionPipeline` when `requires_grad=True`;
       contact-pipeline API may change in 1.3.
     - **RJ45 plug-socket insertion example**: demonstrates SDF contacts, latch joint,
       and interactive gizmo; release notes should lead with the showcase.
-->

---

## Contents

{{CONTENTS_BULLETS}}

<!--
Expand the TOC to include every `###` heading rendered in the body, not
just the `##` top-level sections. List each per-symbol / per-topic heading
as a sub-bullet under its parent section. Example shape (replace with the
real symbols / topics present in this specific report):

- [New API](#new-api)
  - [`newton.<symbol>`](#newtonsymbol)
  - [`newton.<submodule>.<symbol>`](#newtonsubmodulesymbol)
  - ...
- [Breaking Changes](#breaking-changes)
  - per-entry heading as sub-bullet
- [Changes to Existing API](#changes-to-existing-api)
  - per-entry heading as sub-bullet
- [Behavioral & Support Changes](#behavioral--support-changes)
  - per-topic heading as sub-bullet
- [Fixed](#fixed)
- [Calibration Notes](#calibration-notes) (retrospective mode only)
  - per-flag-class sub-bullet
- [CHANGELOG Review Notes](#changelog-review-notes) (only if the conditional
  appendix renders content)

The sample symbol names above are illustrative placeholders; do NOT ship them
as-is. GitHub auto-renders a floating outline panel, but an explicit TOC
still helps raw-text readers.
-->

---

## New API

{{NEW_API_TABLES_BY_KIND}}

<!-- Render one summary table per Kind. Columns: Symbol | Description | GH | Bake.
     Kind groupings for Newton: "Functions", "Classes", "Methods on existing classes",
     "Enums / flags", "Constants", "Examples". If only one Kind has entries, one
     table is fine.

     The Symbol cell uses a short-form call shape (no type annotations; defaults
     included). Examples:
     - `newton.geometry.compute_offset_mesh(shape, offset)`
     - `newton.TetMesh(vertices, tets)`
     - `SolverXPBD.update_contacts(contacts)`  (method on existing class)
     - `newton.ShapeFlags`  (no parens for enums, decorators)
     - `newton.examples.basic_pendulum`  (example module; the Symbol cell links to
       the runnable name `python -m newton.examples basic_pendulum`). -->

{{NEW_API_DETAIL_BLOCKS}}

<!-- Per-symbol block template. Headings use the symbol name alone (no colons/em dashes).
     The single fenced code block underneath holds signature-shaped content and
     the docstring.

Simple data class (constructor-only rendering):

### `newton.TetMesh`

Links: [GH-NNN](...), commits: [abc1234](...), [def5678](...)
Source: `newton/_src/geometry/tetmesh.py`
Bake: 🟢 47 days in main

```python
class TetMesh:
    """Tetrahedral mesh geometry for soft-body simulation.

    Holds per-vertex positions and per-tet index quads; builders accept this
    shape via ModelBuilder.add_shape_tetmesh.
    """

    def __init__(self, vertices: wp.array[wp.vec3], tets: wp.array[wp.vec4i])
```

Class with multiple public methods (interface-style): list every public method
with its signature and docstring, not just __init__.

### `newton.solvers.SolverXPBD`

Links: [GH-NNN](...), commit: [sha](...)
Source: `newton/_src/solvers/xpbd.py`
Bake: 🟢 30 days in main

```python
class SolverXPBD:
    """Position-based-dynamics solver with XPBD constraints.

    Supports rigid bodies, particles, cloth, and soft-body elements with a
    common substepping loop.
    """

    def __init__(self, model: Model, iterations: int = 10)

    def step(self, state_in: State, state_out: State, control: Control, dt: float) -> None
    """Advance the simulation by dt seconds."""

    def update_contacts(self, contacts: Contacts) -> None
    """Populate contacts.force with per-contact spatial forces from XPBD impulses."""
```

Enum / IntFlag: list every member with its value and per-member doc.

### `newton.ShapeFlags`

Links: [GH-NNN](...), commit: [sha](...)
Source: `newton/_src/geometry/flags.py`
Bake: 🟡 12 days in main

```python
class ShapeFlags(IntFlag):
    """Flags controlling collision and visibility of a shape."""

    VISIBLE = 1 << 0
    """Shape is rendered by viewers."""

    COLLIDE_SHAPES = 1 << 1
    """Shape participates in shape-shape collision."""

    COLLIDE_GROUND = 1 << 2
    """Shape participates in ground-plane collision."""
```

Method added to an existing class: the heading names the fully-qualified
method path. The fenced block shows only that method, not the whole class.

### `newton.solvers.SolverXPBD.update_contacts`

Links: [GH-NNN](...), commit: [sha](...)
Source: `newton/_src/solvers/xpbd.py`
Bake: 🟢 21 days in main

```python
def update_contacts(self, contacts: Contacts) -> None
"""Populate contacts.force with per-contact spatial forces.

Derives linear force and torque [N, N·m] from XPBD constraint impulses
accumulated during the most recent step. Call after step().
"""
```

Example (runnable under `python -m newton.examples <name>`): the Source cell
points to the example module; list the one-liner user invocation and the
headline simulation topic.

### `newton.examples.robot.example_robot_humanoid`

Links: [GH-NNN](...), commit: [sha](...)
Source: `newton/examples/robot/example_robot_humanoid.py`
Bake: 🟢 19 days in main

Run:

```
python -m newton.examples example_robot_humanoid
```

Short paragraph explaining what the example demonstrates (the solver used,
what the user will see, any dependencies like MuJoCo assets).
-->

---

## ⚠️ Breaking Changes

{{BREAKING_ENTRIES}}

<!-- Flat list. Do NOT group under "Author-labeled" / "Unlabeled" / "Semantic"
     subheadings. Every entry is simply a confirmed breaking change, regardless
     of how it was identified. Claude collects entries from four sources:
     (1) CHANGELOG Changed entries whose prose describes a rename / parameter
         reorder / signature shift (Newton does not use a **Breaking:** marker;
         recognition is prose-based),
     (2) CHANGELOG Removed entries (implicitly breaking; flag as 🚨 Policy if
         no prior Deprecated entry is found),
     (3) public-surface AST diffs between base and HEAD that aren't covered by
         a CHANGELOG Changed / Removed entry,
     (4) solver / integrator / math / geometry commits whose diff Claude reads
         as clearly semantic-shifting (a numerical-output shift that a user
         would observe). Ambiguous candidates go to "Semantic-Change Review
         Candidates" in the review notes, NOT here.

     Per-entry render format:

     ### <heading: symbol name or short descriptive title>

     Links: [GH-NNN](...), commit(s): [sha](...). 🟢 N days baked in main.

     [If signature diff applies, a fenced diff block.]
     [If behavior shift, a before/after code snippet synthesized from the diff
      and the commit message.]

     [1-3 sentences of explanatory prose in plain user-facing language.]

     [Full CHANGELOG text blockquoted, if the entry came from CHANGELOG.]

     [For 🚨 Policy: removed without prior deprecation entries: a one-line
      callout naming the missing deprecation window and what the release
      manager should do about it.]

     Illustrative example (do not copy verbatim):

     ### `ModelBuilder.add_shape_gaussian` parameter reorder

     Links: [GH-NNN](https://github.com/newton-physics/newton/issues/NNN),
     commit: [abc1234](https://github.com/newton-physics/newton/commit/abc1234). 🟢 22 days baked in main.

     The `xform` argument now precedes `gaussian` to match every other
     `add_shape_*` method on `ModelBuilder`. Callers that relied on positional
     arguments must switch to keyword form. Passing a `Gaussian` as the second
     positional argument still works but emits a `DeprecationWarning`.

     Before:
     ```python
     builder.add_shape_gaussian(body_idx, gaussian, xform=my_xform)
     ```

     After:
     ```python
     builder.add_shape_gaussian(body_idx, xform=my_xform, gaussian=gaussian)
     ```
-->

---

## Changes to Existing API

<!-- Covers CHANGELOG Changed, Removed, Deprecated, plus capability extensions
     routed here from the new-API classification pass (e.g., "Add support for X
     in existing Y"). -->

{{CHANGED_SUMMARY_TABLE}}

<!-- Columns: API | Kind | Breaking | Description | GH | Commits | Bake
     Kind values: signature change, new parameter, capability extension, rename,
     parameter reorder, removed, deprecated, semantic change. Description is a
     short phrase (≤ 10 words). Breaking cell values: Yes / No / Experimental. -->

{{CHANGED_DETAIL_BLOCKS}}

<!-- Per-entry template (signature change). Use a colon in the heading, not an em dash.

### `SolverMuJoCo`: new `enable_multiccd` parameter

Breaking: **No** (additive keyword-only default)
Links: [GH-NNN](...), commit: [5c5f67e9](...)
Bake: 🟢 38 days in main

```diff
- def __init__(self, model: Model, ..., use_mujoco_contacts: bool = True)
+ def __init__(self, model: Model, ..., use_mujoco_contacts: bool = True, enable_multiccd: bool = False)
```

**From CHANGELOG**
> Add `enable_multiccd` parameter to `SolverMuJoCo` for multi-CCD contact generation
> (up to 4 contact points per geom pair).

Capability extension (existing API gains a new option/behavior):

### `ModelBuilder.add_shape_*`: per-shape color

Links: [GH-NNN](...), commit: [abcdef12](...)
Bake: 🟢 35 days in main

Every `add_shape_*` method gains a `color=` keyword argument; `Model.shape_color`
holds the authored per-shape display color for runtime edits; mesh shapes fall
back to `Mesh.color` when available.

```diff
- def add_shape_sphere(self, body: int, radius: float, ..., is_visible: bool = True)
+ def add_shape_sphere(self, body: int, radius: float, ..., is_visible: bool = True, color: Vec3 | None = None)
```

**From CHANGELOG**
> <full CHANGELOG text blockquoted>

Rename (Newton-specific kind, soft deprecation via DeprecationWarning):

### `ModelBuilder.add_shape_ellipsoid`: parameter rename

Breaking: **Yes** (source break if old names used positionally)
Links: [GH-NNN](...), commit: [sha](...)
Bake: 🟢 28 days in main

```diff
- def add_shape_ellipsoid(self, body: int, a: float, b: float, c: float, ...)
+ def add_shape_ellipsoid(self, body: int, rx: float, ry: float, rz: float, ...)
```

Old names `a`, `b`, `c` are still accepted as keyword arguments but emit a
`DeprecationWarning`. Positional callers must switch to keyword form.

**From CHANGELOG**
> Rename `ModelBuilder.add_shape_ellipsoid()` parameters `a`, `b`, `c` to `rx`, `ry`, `rz`.
> Old names are still accepted as keyword arguments but emit a `DeprecationWarning`.

Removed symbol:

### `Model.geo_meshes`: removed

Links: [GH-NNN](...), commit: [sha](...)
Deprecation window: Deprecated in 1.0.0; removed here.

```diff
- geo_meshes: list[Mesh]
```

**From CHANGELOG (prior deprecation in 1.0.0)**
> Deprecate `Model.geo_meshes` in favor of `Model.shapes`. Will be removed in a
> future release.

**From CHANGELOG (removal)**
> <full CHANGELOG text blockquoted>

Experimental-softened entry (symbol was shipped as experimental in a prior release; change here is technically source-breaking but the stability bar was advertised up front):

### `SensorTiledCamera`: RenderContext reorganization

Breaking: **Experimental** (since 1.0.0)
Links: [GH-NNN](...), commit: [sha](...)
Bake: 🟢 40 days in main

`SensorTiledCamera` landed as an experimental sensor in 1.0.0; this release
splits `RenderContext` into `RenderConfig` (config types) and `utils` (runtime
access), deprecates the old names, and adjusts default Gaussian sorting modes.

```diff
- class SensorTiledCamera:
-     render_context: RenderContext
+ class SensorTiledCamera:
+     render_config: RenderConfig
+     utils: SensorTiledCameraUtils
```

**From CHANGELOG (1.0.0 introduction, still experimental)**
> Add `SensorTiledCamera` (experimental): GPU-tiled camera sensor for multi-agent RL.

**From CHANGELOG (this release)**
> <full CHANGELOG text blockquoted>
-->

---

## Behavioral & Support Changes

<!-- Group by topic with short descriptive section headings synthesized from
     the entry content. Use colons in headings if separation is needed.
     Related topics should live together.

     Example headings (illustrative, choose based on actual content):
     - "Python minimum version bumped"
     - "MuJoCo / mujoco-warp dependency pins"
     - "Deterministic contact ordering"
     - "USD import: visibility honored"
     - "Mesh SDF construction: parity path"
     - "Asset pinning for reproducible builds"

     Each topic: a short paragraph summary, links, commits, bake. -->

{{BEHAVIORAL_SECTIONS}}

---

## Fixed

{{FIXED_TABLE}}

<!-- Columns: Fix | GH | Commits | Bake
     Keep the full CHANGELOG text in the Fix column; no truncation.
     Do NOT mention fixes from previously-shipped patch releases. The commit-list
     tool scopes to <base>..<head> so those are already excluded. -->

{{CALIBRATION_NOTES}}

<!-- Retrospective mode only. Pre-release / RC mode must leave
     {{CALIBRATION_NOTES}} empty (template consumers should strip the whole
     placeholder line, not leave a stray section heading).

     Retrospective mode renders a ## Calibration Notes section here, immediately
     after Fixed and before the optional audit appendix. Shape:

     ## Calibration Notes

     One-paragraph headline: overall validated / invalidated / unresolved counts
     across all flag classes, plus a one-sentence takeaway on what the calibration
     suggests about skill precision for this release.

     ### 🚨 Missing-deprecation

     | Flag subject | Outcome | Evidence |
     |---|---|---|
     | `Model.foo` | Validated | No prior Deprecated entry in 1.0.0 or earlier; symbol stayed removed through 1.1.2. |
     | `Model.bar` | Invalidated | Deprecated in 1.0.0 under `### Deprecated` — Claude's search missed it (lowercase `model.bar` in the entry). |

     ### 🕵️ Private-only

     | Flag subject | Outcome | Evidence |
     |---|---|---|
     | `newton._src.utils._helper` | Validated | Re-exported as `newton.utils.helper` in [1.1.1](commit-link). |

     ### 📐 Missing-migration-guidance

     | Flag subject | Outcome | Evidence |
     |---|---|---|

     ### 🏷️ Naming-convention drift

     | Flag subject | Outcome | Evidence |
     |---|---|---|

     ### Semantic-change review candidates

     | File / feature | Outcome | Evidence |
     |---|---|---|
     | `newton/_src/solvers/xpbd.py` contact-ordering diff | Validated | [1.1.1](commit-link) fixes "convergence regression on stacked bodies" — consistent with the flagged change's hypothesis. |

     ### Deprecation follow-through

     | Deprecated in this release | Followed through | Evidence |
     |---|---|---|
     | `Model.geo_meshes` | Followed through in 1.2.0 | Removed entry cites `Use Model.shapes`; matches the promised migration path. |
     | `SensorContact.net_force` | Still deprecated | Symbol remains in the public surface through 1.2.1 with DeprecationWarning. |

     Closing note (one or two sentences): what the calibration tells the reader
     about Newton's release hygiene for this specific release.
-->

---

{{OPTIONAL_APPENDIX}}

<!-- Render conditionally based on content and wrap each non-empty section in
     a GFM <details> block so it collapses by default (the umbrella content is
     reference material, not headline reading).

     Three cases:

     1. Both CHANGELOG-orphan list AND language-review flags are empty:
        Render nothing. No appendix heading, no trailing section.

     2. Exactly ONE is non-empty: Render that one as a top-level section
        (no "Audit Appendix" umbrella) with the table inside <details>.

        ## CHANGELOG Entries Without Matching Commits
        <details>
        <summary>N entries (click to expand)</summary>

        | Entry | GH refs | Suspected reason |
        |---|---|---|
        | full entry text | ... | ... |

        </details>

     3. Both are non-empty: Render an umbrella section; each subsection gets
        its own <details>.

        ## Audit Appendix

        <details>
        <summary>N CHANGELOG entries without matching commits (click to expand)</summary>

        | Entry | GH refs | Suspected reason |
        |---|---|---|
        ...

        </details>

        <details>
        <summary>N CHANGELOG entries flagged for review (click to expand)</summary>

        | Entry | Flag | Why |
        |---|---|---|
        ...

        </details>

     Column rules for BOTH tables: full entry text (no truncation).
     Flag glyphs: 🔗 (suspected wrong GH ref), 🗣️ (internal language),
                  📝 (too terse or missing context),
                  🕵️ (private-only symbol not re-exported through a public module),
                  📐 (Deprecated / Removed / rename entry missing migration guidance),
                  🏷️ (new public symbol violates Newton's prefix-first naming),
                  🚨 (removed without prior deprecation — policy violation, surfaced
                      in Breaking Changes but also listed here for auditability).
     An entry with multiple flags appears once per flag. -->

<!-- Report ends here. Do NOT append "end of report", a closing quote, a thanks
     note, or any terminal marker. -->
