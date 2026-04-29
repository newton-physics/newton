# Rendering Rules

This reference is loaded during Phase 7b of the release-audit skill. It defines the hard constraints and conventions for how the report is formatted when the template is filled in. Phase-flow logic lives in `SKILL.md`; this file is pure reference material.

## Key rendering rules

- Full commit URLs: `https://github.com/newton-physics/newton/commit/<full-sha>`.
- Full issue URLs: `https://github.com/newton-physics/newton/issues/<num>`.
- Preserve `diff` fence blocks for signature diffs.
- Anomaly banner appears ONLY if any commit has `main_equivalent_sha: null`.
- **Table of contents** sits immediately after the Release Highlights section (so the front matter reads: counts → bake → highlights → TOC → body). Link every top-level `##` section and every per-symbol / per-topic `###` heading under them.
- **Behavioral & Support Changes** section: group by topic with short descriptive titles (e.g., "Deterministic contact ordering", "Dependency pins", "Build requirements"). Claude synthesizes the titles from the entry content.

## Output style — hard constraints on the generated report

1. **No em dashes (`—`) anywhere in the report output.** Use colons, parentheses, or rewrite the sentence. This includes headings, bullet points, table cells, prose. Check every line before writing.

2. **No internal skill terminology in the output.** The reader does not know what "Phase 4f", "Phase 5a", "tier-1 heuristic", or similar skill-internal names refer to. If a section needs explanation of how flags were produced, write it in plain user terms (e.g., "Flagged because commits tagged with this GH ref don't touch any solver code" instead of "Tier-1 topic mismatch from Phase 5a").

3. **No mention of previously-shipped patch-release fixes.** The commit-list tool already scopes to `<base>..<head>`, so patch-release content is excluded automatically; do not manufacture a comparison to it.

4. **No "end of report" or similar terminal markers.** The last section is the last section. No "— end —", no "Thanks for reading", no concluding paragraph, no closing quote.

5. **Every GH ref is a markdown hyperlink** to `https://github.com/newton-physics/newton/issues/NNNN`. Plain-text `GH-NNNN` tokens, paren-grouped lists like `(GH-1287, GH-1298, ...)`, and shorthand like `(multiple GHs)` / `(see CHANGELOG)` are NOT acceptable, even when many refs bunch into one bullet or cell.

6. **Signature and docstring render as a single fenced code block**, shaped like Python source so users see them in one glance. The exact shape depends on the kind of symbol.

   **Functions and methods:**

   ```python
   newton.geometry.compute_offset_mesh(shape: ShapeFlags, offset: float) -> Mesh
   """Extract the offset surface mesh of a collision shape.

   Longer description if present in the real docstring.
   """
   ```

   **Classes with only a constructor** (simple data holders or context managers):

   ```python
   class TetMesh:
       """Tetrahedral mesh geometry for soft-body simulation.

       Longer description if present.
       """

       def __init__(self, vertices: wp.array[wp.vec3], tets: wp.array[wp.vec4i])
   ```

   **Classes with additional public methods**: list EVERY public (non-dunder, non-leading-underscore) method with its signature, plus the class docstring and each method's docstring if present. Do not just show `__init__`.

   ```python
   class SolverXPBD:
       """Position-based-dynamics solver with XPBD constraints.

       Short description of the solver's scope and typical use.
       """

       def __init__(self, model: Model, iterations: int = 10)

       def step(self, state_in: State, state_out: State, control: Control, dt: float) -> None
       """Advance the simulation by dt seconds."""

       def update_contacts(self, contacts: Contacts) -> None
       """Populate contacts.force with per-contact spatial forces from XPBD impulses."""
   ```

   **Enum / IntEnum / IntFlag classes:** list every member with its integer value and its attribute docstring or comment, plus the class docstring. Do not show a constructor for enums.

   ```python
   class ShapeFlags(IntFlag):
       """Flags controlling collision and visibility of a shape."""

       VISIBLE = 1 << 0
       """Shape is rendered by viewers."""

       COLLIDE_SHAPES = 1 << 1
       """Shape participates in shape-shape collision."""
   ```

   Extract member docstrings from the source using `ast` attribute-docstring form (`"""..."""` immediately following the assignment). If the member uses a `#:` comment or a trailing `#` comment, preserve that instead. If there is no per-member doc, show the member without one.

   Do NOT separate "Signature" and "Docstring" into two headed subsections. Do NOT blockquote the docstring line by line; it lives inside the code block as Python source.

7. **API summary tables** include a Description column AND a short-form signature in the Symbol cell. The Symbol cell shows the call shape WITHOUT type annotations so readers can skim the args at a glance. Defaults ARE included. The column order for New API tables is: `Symbol | Description | GH | Bake`. Examples of Symbol cells:

   - Function: `newton.geometry.compute_offset_mesh(shape, offset)`
   - Method: `SolverXPBD.update_contacts(contacts)`
   - Class with constructor: `newton.TetMesh(vertices, tets)`
   - Enum / flag (no call form): `newton.ShapeFlags`
   - Decorator: `@newton.experimental`

   Description is a short (≤ 10 word) phrase summarizing what the symbol does, pulled from the first sentence of its docstring or the CHANGELOG entry.

8. **New API tables are grouped by Kind.** Render one table per Kind ("Functions", "Classes", "Methods on existing classes", "Enums / flags", "Examples"). Newton's "Examples" kind specifically covers additions under `newton/examples/**` that ship a user-runnable `python -m newton.examples <name>` entry.

9. **Changes-to-Existing-API table columns**: `API | Kind | Breaking | Description | GH | Commits | Bake`. The API cell uses the same short-form call-shape convention as rule 7 (parameter names + defaults, no annotations). Description is a short phrase. Kind values include: `signature change`, `new parameter`, `capability extension`, `rename` (Newton-specific), `parameter reorder` (Newton-specific), `removed`, `deprecated`, `semantic change`. For any entry tagged experimental in Phase 4g, the Breaking cell reads `Experimental`. For any Removed entry, the Description also includes the deprecation-window fact from Phase 4d (e.g., "Deprecated in 1.0.0; removed here.").

   **Bake-cell format (tables only).** Render the Bake column as `🟢 47d`, `🟡 12d`, `🟠 4d` — emoji, space, number-and-`d` joined with no intervening space. The joined `<N>d` form keeps the bucket/duration pair on one line when a narrow Markdown table wraps. For per-symbol detail blocks and prose, continue to spell out `🟢 47 days in main`; the compact form is for table cells only.

10. **Audit appendix rendering is conditional.** If only one of the audit sections (CHANGELOG-orphan entries / language-review flags / Phase 4f semantic-review candidates) has any content, do not render an "Audit Appendix" umbrella heading. Just render the non-empty sections with their own top-level headings (e.g., `## CHANGELOG Review Notes`, `## Semantic-Change Review Candidates`). Only use an umbrella when two or more subsections are non-empty.

11. **No Phase names anywhere.** If the report needs to explain a flag, write it in user-facing terms. Never write "Phase 4e", "Phase 5", etc.
