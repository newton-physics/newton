---
name: release-audit
description: "Use when auditing a Newton release for keep/defer decisions before a cut, reviewing an RC for readiness, or calibrating the skill against an already-shipped release."
disable-model-invocation: true
argument-hint: "[target-version]"
allowed-tools: Bash(git log *) Bash(git show *) Bash(git tag *) Bash(git rev-parse *) Bash(git cherry *) Bash(git diff *) Bash(git diff-tree *) Bash(git worktree *) Bash(git stash *) Bash(git checkout *) Bash(python3 *list_commits.py*) Bash(python3 /tmp/*) Bash(rm /tmp/newton-*) Bash(uv run python3 *) Bash(uv run -m *) Bash(gh --version) Bash(gh auth status) Bash(gh gist create *) Bash(gh gist list *) Bash(gh gist edit *) Bash(gh issue view *) Bash(gh issue list *) Read Write Grep Glob
---

# Release Audit

Generates a markdown audit of a Newton release for keep/defer decisions (or, in retrospective mode, for skill calibration). Three modes, auto-detected in Phase 1:

- **Pre-release**: spot-check while work is still landing on main. No release branch cut. Version string is `X.Y.Z.devN`.
- **Release-candidate**: readiness review after the release branch is cut. Version string is `X.Y.ZrcN` or head is `release-X.Y`.
- **Retrospective**: audit an already-shipped release (e.g., `v1.1.0`) against its predecessor, with a Calibration Notes section (Phase 7) that checks Claude's flags against what subsequent patch/minor releases actually did. Triggered by passing a bare released-version argument that matches an existing git tag.

**Output:** a single markdown report, filed according to the destination chosen in Phase 1:
- **Secret gist** (default when `gh` is available and authenticated): stable filename `newton-<version-string>-<prerelease|rc|retrospective>-report.md`, stable description `Newton <version-string> <Pre-Release|Release Candidate|Retrospective> Report`. Later runs against the same version revise the same gist in place; prior versions are preserved in the gist's git history.
- **Local markdown file** (fallback when `gh` unavailable, or opt-in when `gh` available): dated path at `$(git rev-parse --show-toplevel)/newton-<version-string>-<prerelease|rc|retrospective>-report-<YYYY-MM-DD>.md`. Not auto-committed; user moves, shares, or deletes as desired.

**Inputs inferred from repo state:**
- Pre-release / RC: target version from `pyproject.toml` (`version = "..."`). `newton/_version.py` reads from installed metadata at runtime and is not a static source.
- Retrospective: target version from the user's argument. `pyproject.toml` is ignored.
- Base = latest tag matching previous minor's line (`vX.Y-prev.*`, latest patch).
- Head (pre-release / RC) = `upstream/release-<target>` if it exists, else `upstream/main`. Head (retrospective) = the `vX.Y.Z` tag itself.

**Reference documents to load on demand** (via `Read`):
- `references/report-template.md` — fill this in during Phase 7b (the `{{HEADLINE_SUMMARY}}` placeholder is drafted in Phase 7a).
- `references/render-rules.md` — rendering conventions and output style hard constraints for Phase 7b.
- `references/classification-rules.md` — path/symbol rules used in Phases 3-5.
- `references/language-review-examples.md` — Phase 5 language-review calibration.

## Phase 1 — Align on scope

1. **Determine the mode.** Look at the user's argument (`$1`), if any:

   - **Argument present AND matches an existing tag** (`git rev-parse --verify v<arg>` or `git rev-parse --verify <arg>` succeeds, AND the resolved name looks like `vX.Y.Z` / `vX.Y.ZrcN`): **Retrospective mode**. The argument is the already-shipped target version. Skip `pyproject.toml` entirely; the tag is authoritative. Record the raw version string (e.g., `1.1.0`) for the report header and filenames. Do not run the pre-release / RC reconciliation in step 4.
   - **Argument present but does NOT match any tag**: treat as a version override for the upcoming release. Use it as if it came from `pyproject.toml`, then fall through to the pre-release / RC detection below.
   - **No argument**: read the version string from `pyproject.toml` — the top-level `[project]` table's `version = "..."` line.

   For the non-retrospective path, parse the version string to extract the target minor (e.g., `1.2.0.dev0` → target `1.2`) and pre-classify mode:
   - If the version string contains `"rc"` (e.g., `1.2.0rc1`) → **RC mode candidate**: this is a release-candidate readiness report.
   - If the version string contains `"dev"` (e.g., `1.2.0.dev0`) → **Pre-release mode candidate**: this is an early-stage audit of unreleased work.
   - Otherwise → pre-release mode (default), but record the raw version string so the header can show it as-is.

2. **Enumerate previous-minor tags** (applies to all modes):
   ```bash
   git tag --list 'v<prev-major>.<prev-minor>.*' --sort=-v:refname
   ```
   where `<prev-major>.<prev-minor>` is `target - 0.1` (e.g., for target `1.2`, previous minor is `1.1`). Take the first result as the base candidate. Ignore pre-1.0 `beta-*` tags when a stable `vX.Y.Z` line exists.

   **Major-boundary fallback.** When `target.minor == 0` (e.g., `2.0.0`), the `target - 0.1` computation yields a minor line that never existed (`1.9`), and the tag list comes back empty. In that case, enumerate the highest minor line of the previous major instead:
   ```bash
   git tag --list 'v<target-major - 1>.*' --sort=-v:refname
   ```
   Take the first result (the last-patch of the last-minor of the previous major) as the base candidate. If both the primary and fallback searches return empty (which should only happen on a never-released line), surface that to the user in step 6 rather than silently proceeding.

   For **retrospective mode** with target `X.Y.Z`: the base is the last `vX.Y-prev.*` tag strictly before `vX.Y.Z`. Also check whether `X.Y.Z` is itself a patch release (`.Z > 0`): if so, the "base" could be either the previous patch on the same minor (`vX.Y.<Z-1>`) OR the previous minor's latest. Present both in step 6 and let the user pick — a patch retrospective usually wants patch-on-patch; a minor-release retrospective wants previous-minor's last patch.

3. **Probe for the head**:
   - **Retrospective mode**: head is the `vX.Y.Z` tag directly. Resolve with `git rev-parse --verify v<target>`.
   - **Pre-release / RC**:
     ```bash
     git rev-parse --verify upstream/release-<target>
     ```
     If this succeeds, head = `upstream/release-<target>` and this is also a strong signal for **RC mode** (the branch-cut has happened). Otherwise try `origin/release-<target>`; otherwise head = `upstream/main` (falling back to `origin/main`, then `main`) which is **pre-release mode**. Record whichever fallback was used for the report header.

4. **Reconcile mode** from version string and head (skipped for retrospective mode; its mode is already fixed):
   - Version says RC AND head is a release branch → **RC report** (strong match).
   - Version says RC but head is main (branch not cut yet) → **RC report** (version is authoritative; note the mismatch in the header).
   - Version says dev AND head is a release branch → **RC report** (branch cut implies we're past the dev window).
   - Version says dev AND head is main → **Pre-release report**.

5. **Probe `gh` availability and look up any existing matching gist.**

   First run:
   ```bash
   gh --version && gh auth status
   ```
   If either fails → `gh` unavailable; destination will be a local markdown file only; skip the rest of this step.

   Both succeeded → compute the stable gist title for this report:
   ```text
   Newton <version-string> <Pre-Release|Release Candidate|Retrospective> Report
   ```
   (e.g., `Newton 1.2.0rc1 Release Candidate Report`, `Newton 1.2.0.dev0 Pre-Release Report`, `Newton 1.1.0 Retrospective Report`). No date. The gist filename and description are stable so later runs can find the same gist and revise it; gist git history preserves prior versions automatically.

   Run `gh gist list --limit 1000` and filter rows whose description exactly matches that stable title. Capture the matching gist IDs; display URLs are `https://gist.github.com/<id>`. Record the match count (0, 1, or N ≥ 2) for step 6.

6. Present proposal in chat and **wait for explicit user confirmation** of refs AND output destination. Mandatory pause.

   Lead with the mode-specific intro line:
   - Pre-release: `Generating **pre-release report** for Newton **<version>**. Base **<base-ref>** → Head **<head-ref>**. **<N>** commits in range.`
   - RC: `Generating **release-candidate report** for Newton **<version>**. Base **<base-ref>** → Head **<head-ref>** (release branch cut). **<N>** commits in range.`
   - Retrospective: `Generating **retrospective report** for Newton **<version>** (already shipped). Base **<base-ref>** → Head **v<version>**. **<N>** commits in range. Calibration pass will cross-reference Claude's flags against subsequent patch / minor releases.`

   For retrospective mode with a patch-target (when `.Z > 0`), also present the base choice explicitly:
   > Base candidates for retrospective of `vX.Y.Z`:
   > 1. `vX.Y.<Z-1>` (patch-on-patch: what changed since the previous patch on the same minor)
   > 2. `v<prev-major>.<prev-minor>.<latest-patch>` (minor-boundary: what changed since the previous minor)
   >
   > Pick (1) or (2), or specify a different base.

   Append the output block for the current `gh` / match state:

   **`gh` unavailable:**
   > Output: markdown file at repo root (`gh` not available). Confirm, or specify different refs?

   **`gh` available, 0 matches:**
   > Output: (a) new secret gist [default], (b) local markdown file at repo root. Confirm refs + pick.

   **`gh` available, 1 match:**
   > Output: (a) revise existing gist `<url>` [default], (b) new secret gist, (c) local markdown file at repo root. Confirm refs + pick.

   **`gh` available, N matches (N ≥ 2):**
   > Multiple existing gists share the stable title:
   > 1. `<url-1>` — updated `<time-1>`
   > 2. `<url-2>` — updated `<time-2>`
   > ...
   >
   > Output: (a) revise gist by number, (b) new secret gist, (c) local markdown file at repo root. Confirm refs + pick.

   Do not run any further phase until the user confirms refs and (when `gh` is available) chooses destination. Translate the user's reply into exactly one of the destination tokens `local`, `new-gist`, `revise-gist:<id>` and record it for Phase 7c. Letters `(a)`, `(b)`, `(c)` are positional within the current branch's prompt, not global: resolve them against the option list you just showed the user. For the N-match branch, the user picks a gist by the number you listed (e.g., "revise 2" → `revise-gist:<id-of-listed-row-2>`), or says new / local.

## Phase 2 — Gather ground truth

1. Run the commit-list tool:
   ```bash
   python3 ${CLAUDE_SKILL_DIR}/scripts/list_commits.py \
     --base <base-ref> \
     --head <head-ref> \
     --report-date "$(date +%F)" \
     --main-ref <resolved-main-ref>
   ```
   from the repo root (`$(git rev-parse --show-toplevel)`). Capture stdout as the `commit_list_json`.

2. Read `CHANGELOG.md` using the Read tool. Choose the section by mode:
   - **Pre-release / RC**: read `CHANGELOG.md` at HEAD. Locate the `## [Unreleased]` header. Collect all content from that header up to (but not including) the next `## [X.Y.Z]` header.
   - **Retrospective**: read `CHANGELOG.md` at HEAD (the current working tree — CHANGELOG is append-only, so the section for a prior release is still present). Locate the `## [<target-version>]` header (e.g., `## [1.1.0] - 2026-04-13`; match the prefix `## [<target-version>]` and tolerate optional date text after it). Collect content from that header up to (but not including) the next `## [X.Y.Z]` header (the previous release's section). If the header cannot be found at HEAD, fall back to `git show v<target>:CHANGELOG.md` and parse that file the same way (in case the section was renamed or removed in a later refactor).

3. Parse subsections. Each starts with `### Added`, `### Removed`, `### Deprecated`, `### Changed`, `### Fixed`, or `### Documentation`. For every bullet under each subsection, extract:
   - **Raw text (FULL — never truncate)**: the full bullet content (may span multiple lines).
   - **Section**: one of the six names above.
   - **GH refs**: regex `GH-(\d+)` over the bullet text. Dedup. (Newton commits and CHANGELOG entries sometimes also reference `#NNNN` as a bare PR number; capture these too.)
   - **Migration hint**: does the prose contain a migration phrase (`use <new>`, `in favor of`, `renamed to`, `replaced by`, `switch to`, `migrate to`)? Record a boolean.

   **Newton does NOT use a `**Breaking:**` literal marker in CHANGELOG** (unlike Warp). Do not rely on its presence. Breaking-change detection in Newton comes from: (a) the `### Removed` section (implicitly breaking), (b) signature-diff AST analysis (Phase 4e), and (c) reading `### Changed` prose for rename / parameter-reorder / signature-shift language (Phase 4d).

## Phase 3 — Cross-reference

1. **Build the commit ↔ CHANGELOG join** on GH-ref overlap:
   - For each CHANGELOG entry with at least one GH ref, find commits (from `commit_list_json`) whose `gh_refs` intersect.
   - For each CHANGELOG entry with ZERO matching commits on GH ref, attempt a secondary lookup — find the commit(s) on HEAD that introduced or modified this exact entry text in `CHANGELOG.md`:
     ```bash
     git log --reverse -S'<distinctive substring from the entry>' --format='%H|%s|%cs' -- CHANGELOG.md
     ```
     The first commit whose subject isn't a CHANGELOG-only edit (e.g., not "Clean up changelog", not a version bump) usually IS the code change associated with the entry. Record that as the backing commit.
   - After these two passes, any CHANGELOG entry still with no matching commit is a genuine orphan case (deferred/dropped). Keep full entry text; do not truncate in the report.

2. **Do NOT build an audit trace of unmatched commits.** The old "commits without CHANGELOG entries" appendix adds noise without value. Commits that don't map to a CHANGELOG entry are not surfaced in the report.

## Phase 4 — Analyze API surface

### 4a — Determine what is genuinely NEW API

For each CHANGELOG `Added` entry, extract the named symbol(s) — text in backticks matching `newton.*`, bare `ClassName`, bare `snake_case_name()`, or `ClassName.method()` patterns.

For EACH named symbol, check if it existed at base. Newton exposes user-facing symbols through multiple public re-export modules (not a single `__init__.py`). See `references/classification-rules.md` → "Public API surface" for the authoritative list. The source of truth is `MODULES` in `docs/generate_api.py`, which may have grown or shrunk between releases — load it at HEAD for pre-release / RC mode, or at `v<target>` (via `git show v<target>:docs/generate_api.py`) for retrospective mode.

To check presence at base, run `git show <base>:<path>` for each relevant public module and grep its imported names. For method additions on an existing class (e.g., `SolverXPBD.update_contacts()`), inspect the class body at base in its real source (resolved via `_src/`). For retrospective mode, "HEAD" in the symbol-resolution text below means the `v<target>` tag, not the working tree. Use `git show v<target>:<path>` everywhere the pre-release / RC flow uses the working tree.

Classification:
- **Genuinely new** (symbol was not present at base) → **New API** section.
- **Existed at base** (entry is adding a new parameter, option, or capability to something that already existed) → **Changes to Existing API** section with a "capability extension" or "new parameter" kind. Cite the backing commit's signature diff.
- **Does not name a single symbol** (e.g., "Interactive example browser in the GL viewer with tree-view navigation") and describes a cross-cutting capability → **Behavioral & Support Changes** section unless one of the mentioned symbols is genuinely new (in which case split: put the new symbol in New API, the capability description in Behavioral).

**If an entry mentions multiple symbols where some are new and some pre-existed** (e.g., "Add `newton.geometry.compute_offset_mesh()` and a viewer toggle"), split: the genuinely new symbols each get a New API entry; the extensions to existing symbols each get a Changes entry.

**Public-API exposure check.** For every symbol that passes the "genuinely new" test, also verify at HEAD that it is reachable via one of the public re-export modules listed above. If the symbol only exists under `newton._src.*` and is not re-exported through a public module, flag it in the report (Section "CHANGELOG Review Notes" → 🕵️ Private-only) — AGENTS.md requires user-facing symbols to be re-exported and forbids examples/docs from importing `newton._src`. Do not treat this as a hard block on the entry; surface it so the release manager can confirm the symbol was intended to be public.

### 4b — Resolve New API signatures + docstrings

For each symbol confirmed as new:

1. Find its real source module by following the re-export in the public module (e.g., `newton/solvers.py` → `newton/_src/solvers/xpbd.py`).
2. `ast.parse` the source module; find the `FunctionDef` / `ClassDef` / `AsyncFunctionDef`.
3. Extract the signature by re-stringifying the args (preserve type annotations).
4. Extract the docstring verbatim via `ast.get_docstring(node)`.
5. Render as shown in the template.

Newton has no kernel-scope builtins layer to extract (no `add_builtin()` registry like Warp). All user-facing symbols live in Python modules and `ast` resolves them directly.

### 4c — Symbol resolution fallbacks

- **Symbol in backticks doesn't resolve at HEAD**: render the entry with a ⚠️ note: "Couldn't resolve `<symbol>` in source — verify entry names a real public symbol." Do not fabricate a header like `newton.*(no symbol)*`. Use the entry's natural subject as the section title.
- **Entry describes a topic, not a symbol** (e.g., "Interactive example browser in the GL viewer"): use a short descriptive title summarized from the entry (e.g., "Interactive example browser"), not a synthetic `newton.*` name.
- **Header naming rule**: section headers in the report should be either real fully-qualified symbols (`newton.solvers.SolverXPBD.update_contacts`) OR short descriptive titles extracted from the entry. Never `newton.*`, `newton.(something)`, or similar stub patterns.

### 4d — Changed / Removed / Deprecated signature diffs

For each CHANGELOG entry in Changed / Removed / Deprecated (plus any "capability extension" entries routed here from 4a):

- Compute signatures at base and HEAD (same resolution as 4b).
- For signature-shape changes: render a fenced `diff` block showing `-` and `+` lines.
- For semantic-only changes (no signature shift) where the prose describes a rename / reorder / behavioral flip: skip the diff block; include the backing commit's URL and the full CHANGELOG text.
- For Removed entries: show the old signature on a `-` line; omit `+`.

**Deprecation-window lookup for Removed entries.** Newton's policy (per AGENTS.md) is: *breaking changes require a deprecation first*. Every Removed entry should correspond to a prior Deprecated entry. For every Removed entry (and every Changed entry whose prose describes a removal), search CHANGELOG.md for the matching prior Deprecated entry:

1. Extract distinctive tokens from the Removed entry: the named symbol(s) in backticks and, if the entry carries a GH ref, that ref number.
2. Scan the appropriate released-version sections of CHANGELOG.md for a `### Deprecated` bullet that names the same symbol(s) OR the same GH ref. The search scope depends on mode:
   - **Pre-release / RC** (current target is `Unreleased`): scan everything below `## [Unreleased]` (i.e., all historical released sections), top-down.
   - **Retrospective** (current target is `X.Y.Z`): scan ONLY the sections strictly below `## [X.Y.Z]` in CHANGELOG.md — the prior-release sections. Do NOT consider `## [Unreleased]` or later-version sections; they were written after `X.Y.Z` shipped and cannot have preceded it. Top-down within the allowed range.
   The FIRST such entry (highest version, since CHANGELOG is reverse-chronological) is the deprecation introduction.
3. Record: (a) the release version heading that contains the Deprecated entry (e.g., `1.0.0`), (b) the full Deprecated entry text.
4. In the rendered Removed entry, include a one-line deprecation window: `Deprecated in X.Y.Z; removed here.` If the deprecation entry cannot be found, write `No prior **Deprecated** entry found in CHANGELOG — Newton's policy requires deprecation before removal; please verify this was announced earlier.` Do NOT fabricate a version.

The deprecation window belongs in BOTH the Breaking Changes entry for the removal AND the Changes-to-Existing-API row (in the Description cell or as an appended sentence in the detail block). A reader should never have to ask "was this deprecated first, and for how long?"

**Missing-deprecation flag.** If a Removed entry has no prior Deprecated entry in a previously-released section of CHANGELOG.md, surface this prominently in the Breaking Changes section as `🚨 Policy: removed without prior deprecation` rather than a quiet "verify please" line. This fires whether or not the current release's own `### Deprecated` section ALSO names the symbol — Newton's policy is *prior* deprecation, and a deprecate-and-remove in the same release is itself a violation. The release manager needs this to block the release or add migration tooling.

**Exception: `1.0.0` pre-stable cleanup.** Removed entries in the `1.0.0` release (and `1.0.0rcN`) are exempt from the deprecation-first policy — PRs labeled `1.0-release` are the pre-stable API cleanup and were not required to go through a prior deprecation window. When the target version is `1.0.0` or `1.0.0rcN`, do not emit the `🚨 Policy` flag or the "No prior **Deprecated** entry found" line for its Removed entries. Still render the deprecation-window line if a matching Deprecated entry happens to exist; otherwise note `1.0 pre-stable cleanup; no prior deprecation required.`

### 4e — Signature-AST diff for unlabeled migration-required changes

Independently of CHANGELOG content, compute the public API surface at base vs. HEAD by walking every symbol re-exported from the public modules in 4a:

- Base: `git show <base-ref>:<module.py>` for each public module and for every real-source module it re-exports from → parse with `ast`. Resolve each re-export's real signature at base.
- Target: same as Base, but at the target ref — the working tree in pre-release / RC mode, or `git show v<target>:<module.py>` in retrospective mode. Do NOT read the working tree in retrospective mode; later commits on `main` would otherwise be falsely attributed to `vX.Y.Z`.
- For each symbol whose signature shape changed AND whose matching CHANGELOG entry (if any) doesn't indicate a rename / parameter shift / removal → add to Breaking Changes section as "unlabeled signature change — please verify".

**Exception: Removed symbols are breaking by definition.** A symbol that appears in CHANGELOG's `Removed` section does NOT need prose hedging to be valid. Do NOT flag Removed entries as "unlabeled breaking" — the section name itself communicates the breakage. Removed entries surface in the Breaking Changes callout and in the Changes-to-Existing-API section (as "removed" kind), but the report must not whinge about missing Breaking labels on them.

**Exception: additive keyword-only parameters with defaults are not migration-required.** A signature change that only adds new kwargs with defaults after `*` is additive; do not surface in Breaking Changes. It belongs in Changes-to-Existing-API with Kind `new parameter` and Breaking `No`.

### 4f — Semantic-change candidates from solver / integrator commits

Newton has no native C++/CUDA code (unlike Warp); there is no build-and-run verification step. However, changes to solver internals, integrators, collision pipelines, and math helpers can still change observable numerical behavior without altering any signature.

For each commit in `commit_list_json` that touches paths under `newton/_src/solvers/**`, `newton/_src/sim/**` (integrator / collision code), or `newton/_src/math/**`:

1. Skip if the commit is already mapped to a CHANGELOG entry explicitly describing the semantic shift.
2. Read the commit's diff: `git show --stat <sha>` then `git show <sha>` for small diffs, or read specific hunks for large ones.
3. **Triage** into one of three buckets:
   - **Clearly not semantic-breaking** → drop. Examples: renaming internal symbols, comment / format changes, pure internal refactors, performance optimizations that preserve output, test-only changes, bug fixes where the pre-fix behavior was itself a bug.
   - **Clearly semantic-shifting** with an obvious user-observable numerical / behavioral change visible from the diff alone → include in Breaking Changes with a short "Before / After" explanation derived from the diff and commit message.
   - **Ambiguous** — the diff suggests the change could affect numerical output but Claude cannot tell from reading alone whether users would notice → add a concise entry in a "Review candidates" subsection of CHANGELOG Review Notes (NOT Breaking Changes). Include the commit, the touched file(s), and a one-sentence hypothesis. Do not speculate in Breaking Changes.

**Do NOT attempt to build and run Newton at base vs. HEAD.** Newton's solver outputs depend on Warp, MuJoCo, and GPU state; reproducing a numerical diff in a one-shot audit is not reliable and not worth the setup cost. When in doubt, route to the review-candidates list and leave verification to the release manager.

### 4g — Experimental-marker cross-reference

Some Newton features ship with an explicit `experimental` note in the CHANGELOG entry that introduced them (e.g., "Add differentiable rigid contacts (experimental)") or in docstrings. Changes to those features do NOT carry the same stability contract as changes to stable APIs. The report must reflect that so the release manager does not over-weight the concern.

For each entry in Breaking Changes, Changes to Existing API, and Removed (as collected through 4a–4f), determine whether the affected symbol or feature area is currently experimental:

1. Collect candidate symbols / feature-area phrases from the entry: backticked identifiers, class names, and (for topic-style entries) the most distinctive descriptive noun phrase.
2. Search CHANGELOG.md in released-version sections for bullets that name one of the candidates AND contain the literal substring `experimental` (case-insensitive). Also match via GH ref if the current entry and a prior experimental entry share a GH number. Scope:
   - **Pre-release / RC**: everything below `## [Unreleased]`.
   - **Retrospective** (current target is `X.Y.Z`): everything below `## [X.Y.Z]`. The stability-promotion check (step 3 below) likewise only considers versions strictly before `X.Y.Z`.
3. If a match exists AND there is no subsequent CHANGELOG bullet in a later released version explicitly promoting the symbol to stable (e.g., "Promote X out of experimental", "Stabilize Y"), the symbol is still experimental. Record: (a) the release version that introduced the symbol as experimental, (b) the full text of that introduction bullet.
4. Also check the module source at HEAD for an in-code `.. experimental` / `Experimental:` / `experimental_api` / `@experimental` annotation on the symbol's declaration. If present, treat as experimental regardless of CHANGELOG signal.

Tag every matched entry internally as `experimental=True`. Do not alter the CHANGELOG text itself.

**How the tag changes rendering:**
- Breaking Changes heading for the entry: prefix with `⚠️ Experimental:` rather than the implicit "Breaking" framing. Include a short sentence reminding readers this symbol is opt-in and documented as experimental, with the release where the experimental marker was introduced.
- Changes-to-Existing-API table: the Breaking column shows `Experimental` rather than `Yes`.
- Release Highlights bullet (Phase 7a): use `⚠️ Experimental:` as the risk prefix, not `⚠️ Breaking:`.

**Never drop the entry.** Experimental softening is about alarm level, not suppression. A removed or signature-changed experimental symbol still appears in Changes to Existing API.

## Phase 5 — Review CHANGELOG language and bake

### 5a — Language review

Read `references/language-review-examples.md`. For EACH CHANGELOG entry, apply LLM judgment:

- **🔗 Wrong ref (tier-1)**: for every GH ref in the entry, fetch the mapped commits' subjects and paths. If the entry topic doesn't match the commits' actual scope, flag.
- **🔗 Wrong ref (tier-2)**: if `gh --version` and `gh auth status` both succeed, run `gh issue view <num> --json title,body` per ref and compare issue title to entry topic. Skip silently if `gh` unavailable.
- **🗣️ Internal language**: internal module paths (`newton._src.*`), private identifiers with a leading underscore, Warp-internal types (`wp.types.*` that are not documented user types), implementation-detail verbs ("refactor", "reorganize", "rewrite") without a user-visible outcome.
- **📝 Too terse**: under ~10 words with no context, or missing migration guidance in a Deprecated / Changed entry that names a rename or removal.
- **🕵️ Private-only symbol**: the CHANGELOG `### Added` entry names a symbol that exists only in `newton._src.*` at HEAD and is not re-exported through a public module. See Phase 4a.
- **📐 Missing migration guidance** (Newton-specific): entries in `### Deprecated`, `### Removed`, or `### Changed` (where the prose indicates rename / reorder / removal) MUST include migration guidance per AGENTS.md ("Use `Y` instead", "in favor of `Y`", "switch to `Y`"). Flag entries that rename or remove symbols without pointing to the replacement.
- **🏷️ Naming-convention drift** (Newton-specific): new public symbols in `### Added` whose names violate Newton's prefix-first convention (e.g., `PDActuator` should be `ActuatorPD`; `add_sphere_shape()` should be `add_shape_sphere()`). See AGENTS.md.

Record flagged entries. Keep the FULL entry text in the audit table — do not truncate.

### 5b — Bake aggregation

From `commit_list_json`:
- Bucket each commit's `days_in_main` into **🟢 (>14 days)**, **🟡 (7–14 days)**, **🟠 (<7 days)**. Commits whose `days_in_main` is `null` (no main equivalent) skip bucketing — do NOT coerce `null` to `0` or compare it to a numeric threshold; those commits are accounted for in the next bullet instead.
- Count anomalies: commits with `main_equivalent_sha == null`. If non-zero, prepare the ⚠️ banner for the report header.

## Phase 6 — Calibration Notes (retrospective mode only)

**Skip this phase entirely in pre-release / RC mode.** There is no shipped history to calibrate against.

In retrospective mode, Claude's Phase 3–5 output is a set of flags and classifications that would have surfaced had the skill been run *before* `X.Y.Z` shipped. The point of calibration is to check those flags against what *actually* happened after the release — did the concerns hold up? The output of this phase is a "Calibration Notes" section in the report (Phase 7b renders it).

### 6a — Enumerate post-target history

Identify releases that shipped strictly after `X.Y.Z`:

```bash
git tag --list 'v*' --sort=v:refname
```

Take the sorted list, drop everything up to and including `vX.Y.Z`, keep the rest. Typical post-target history for a minor release `vX.Y.0` is:
- `vX.Y.1`, `vX.Y.2`, ... (patch releases on the same minor)
- `vX.Y+1.0`, `vX.Y+1.1`, ... (the next minor and its patches)

For each post-target tag, read its CHANGELOG section (the `## [<version>]` block in `CHANGELOG.md` at HEAD, or via `git show <tag>:CHANGELOG.md` if the section has since been edited). Parse the six subsections (`Added`, `Changed`, `Deprecated`, `Removed`, `Fixed`, `Documentation`) the same way Phase 2 does.

Also collect the commits in each post-target range (`<prior-tag>..<tag>`) — one full `list_commits.py` invocation per range, with the same required args as Phase 2. `--main-ref` reuses the main ref resolved in Phase 1 (don't rely on the script's `upstream/main` default; Phase 1 may have fallen back to a different remote):

```bash
python3 ${CLAUDE_SKILL_DIR}/scripts/list_commits.py \
  --base <prior-tag> \
  --head <tag> \
  --report-date "$(date +%F)" \
  --main-ref <resolved-main-ref>
```

These commits become the evidence pool for validating Claude's Phase 4f semantic-change candidates.

### 6b — Cross-reference Claude's flags against post-target history

For each flag class, run the check below and record one of three outcomes per flag:
- **Validated** — subsequent history confirms Claude's concern was real.
- **Invalidated** — subsequent history contradicts Claude's concern (false positive).
- **Unresolved** — no post-target evidence either way (flag stands as-is, but the reader can treat it as lower-confidence).

The calibration section groups results by flag class and, within each, by outcome. Never drop a flag from the original report body because of calibration — the calibration is a separate layer of commentary.

**🚨 Missing-deprecation (Removed without prior Deprecated):**
- Validated if: (a) no post-target release adds a retroactive Deprecated entry for the symbol, (b) the symbol stays removed. Users actually lost the API without a deprecation window.
- Invalidated if: a prior Deprecated entry existed that Claude missed. Re-scan CHANGELOG scope with fuzzier matching (different casing, plural / singular, alternate backtick placement). Report the missed entry with its version.
- Invalidated if: a later patch (`vX.Y.<Z+1>`) added the symbol *back* (revert). Name the revert commit and note "removal reverted in vX.Y.<Z+1>".

**🕵️ Private-only (new public API with `_src`-only reachability):**
- Validated if: a later release adds a re-export for the symbol through a public module (`newton.<module>.py`). Cite the release and the re-export commit.
- Invalidated if: the symbol stays reachable only via `_src` in every post-target release AND no issues reference it — it was probably intentional internal-only despite its position in the Added section. Note this as "intended internal; CHANGELOG language could have been clearer".
- Unresolved if: the symbol was removed / renamed before re-export resolution.

**📐 Missing-migration-guidance:**
- Validated if: a later release's CHANGELOG bullet (in `### Changed` or `### Fixed`) retroactively points to the replacement, OR a subsequent issue references users stuck on the migration. The second signal is only available if `gh issue list` returns matches for the symbol name; probe with `gh issue list --search "<symbol>" --state all --json number,title`.
- Invalidated if: the migration was handled by a runtime `DeprecationWarning` with a clear message (grep the code at `v<target>` for the warning text).
- Unresolved: default.

**🏷️ Naming-convention drift:**
- Validated if: a later release renames the symbol to match Newton's prefix-first rule, typically with a DeprecationWarning on the old name. Cite the rename commit and the release.
- Invalidated if: the symbol stays and matches the *local* module convention rather than the global prefix-first rule (e.g., a module of siblings that all use `<Kind>Foo`). Note which convention it actually follows.
- Unresolved: default.

**Phase 4f semantic-change candidates** (the ones Claude routed to "Review Candidates", not Breaking Changes):
- Validated if: a later patch release's `### Fixed` entry touches the same file(s) and describes a symptom consistent with the candidate's hypothesis (e.g., "Fix convergence regression in SolverXPBD step ordering" after a candidate about solver step ordering). Cite the fix commit and the release.
- Invalidated if: the file stayed stable for the rest of the minor line with no related bug reports. Probe `gh issue list --search "<file-or-feature>" --state all --json number,title,createdAt` if `gh` is authenticated — only count issues created after `v<target>`'s release date. Skip the probe silently if `gh` is unavailable.
- Unresolved: default.

**Deprecation-follow-through (new check, retrospective-only):** for every entry in `X.Y.Z`'s own `### Deprecated` section, check subsequent minor releases for the corresponding `### Removed` entry.
- **Followed through** — a later minor removed the symbol as promised. Cite the release.
- **Still deprecated** — symbol remains in the public surface with the deprecation warning in all later minors. No action required; just a fact.
- **Silently dropped** — symbol was removed but no `### Removed` entry cites it (CHANGELOG hygiene bug in the later release). Flag this as a separate concern.

### 6c — Compose the calibration narrative

Aggregate the per-flag outcomes into a short section with this shape:
- One-paragraph headline: how many flags were validated / invalidated / unresolved overall, and what that suggests about the skill's precision on this release.
- Per-flag-class subsection with a compact table: `Flag | Subject | Outcome | Evidence`.
- A closing note on anything the calibration taught the reader about Newton's release hygiene for `X.Y.Z` (e.g., "All Removed entries had prior Deprecated entries — policy held"; or "Two `_src`-only additions were re-exported in 1.1.1, confirming the private-only flag would have caught the release").

Keep this section qualitative. The report body already has the exhaustive per-flag detail — this is a scoreboard.

## Phase 7 — Write report to the chosen destination

### 7a — Draft the release highlights

Before filling the template, synthesize the `{{HEADLINE_SUMMARY}}` section. This is the only part of the report that requires qualitative judgment rather than mechanical rendering. Everything else flows from the cross-reference and classification work in Phases 3-5; this step picks what a reader should know *first*.

**What the summary is (and isn't):**
- IS: a reviewer's preview of what the official release notes will likely call out, written so the release manager can sanity-check the upcoming release post at a glance.
- IS NOT: the actual release notes. Do not write copy the marketing team would ship.
- IS NOT: a restatement of the headline counts. The counts block right above it already carries the quantitative summary; the highlights carry the qualitative one.

**How to pick items.** Select 4 to 8 bullets from the material already analyzed (New API, Breaking Changes, Changes to Existing API, Behavioral & Support, Removed). Use LLM judgment. An item belongs in the highlights if at least one of these is true:
- It changes a user's mental model of Newton (a new solver, a new simulation concept, a platform dropped, a new geometry type).
- It is a migration-required change that needs a note in the release post.
- It is user-facing and is explicitly experimental (readers need to know the stability bar).
- It unlocks a workflow that was previously impossible or awkward (e.g., Gaussian splat support, deterministic contact ordering, differentiable contacts).
- Multiple smaller entries form a coherent theme worth a single combined bullet (e.g., "new MPM examples: beam twist, snow ball, viscous coiling").

An item does NOT belong in the highlights if any of these is true (drop even if the CHANGELOG entry is present):
- It is a pure bug fix whose symptom description fits in one line and has no surprising semantics (goes under Fixed, not highlights).
- It is a build-system, CI, or infrastructure change with no runtime user effect.
- It is an internal refactor already scoped away from user-visible surface.
- It is a capability extension to an existing parameter that a typical user would not notice (e.g., a defaults tidy-up).

Aim for 4-8 bullets total. Fewer than 4 almost always means you missed a theme; more than 8 means you listed changes instead of highlights.

**How to write each bullet.** Each bullet leads with a bold 2-6 word headline, then a colon, then one sentence of rationale that explains what it is and why it matters. Attach risk flags inline when they apply: prepend `⚠️ Breaking:` or `⚠️ Experimental:` to the headline where relevant, and append a bake hint (`🟠 N days bake.`) when the headline item's minimum bake is under 7 days. Example:

> - **⚠️ Breaking: `ModelBuilder.add_shape_ellipsoid` parameters renamed** ([GH-NNN](...)): `a`, `b`, `c` → `rx`, `ry`, `rz`; release post needs a migration snippet. Old names still accept keyword form with a DeprecationWarning.

**Lead with the unlock, not the mechanism.** The rationale sentence should answer "what is newly possible, and why would a user care?" — not "what API names were added." API names are a detail; novel capability is the story. If a feature introduces a new artifact or format (a new solver, a new geometry type, a new import path), NAME that artifact and state what it unlocks.

**GH refs MUST be hyperlinks, always.** Every `GH-NNNN` in a highlight bullet is a markdown link to `https://github.com/newton-physics/newton/issues/NNNN`. This applies even when a single bullet combines multiple GH refs. Do NOT use shortcuts like `(multiple GHs)`, `(GH-1287, GH-1298, ...)` in plain text, or `(see CHANGELOG)`. If the bullet covers six issues, render all six as individual links, either inline (`([GH-1287](...), [GH-1298](...), [GH-1335](...))`) or in a trailing parenthesis at the end of the headline. There is no upper limit on link count; a reader can scan links but cannot resolve plain numbers.

**Experimental softening.** If Phase 4g tagged an entry as experimental, the highlight bullet uses `⚠️ Experimental:` as its risk prefix rather than `⚠️ Breaking:`, even if the change is technically source-breaking. The rationale sentence should lead with what changed and its stability bar, not with migration urgency.

Open the summary with a 2-3 sentence intro paragraph that names the shape of the release in plain language. This sets the tone for everything below it. Do not stuff the intro with numbers or repeat the bake distribution.

**Output style rules apply here too.** No em dashes. No skill-internal terminology ("Phase 4f"). No "end of summary" markers. The summary reads as release-note input, not as an audit artifact.

### 7b — Fill template

Read `references/report-template.md`. Fill in every `{{PLACEHOLDER}}` marker, including the `{{HEADLINE_SUMMARY}}` produced in 7a. In retrospective mode, also fill the `{{CALIBRATION_NOTES}}` placeholder using the narrative composed in Phase 6c; leave it empty (and the template will elide the section) in pre-release / RC mode.

**Rendering conventions live in `references/render-rules.md`.** Read that file when starting 7b. It covers:

- URL shapes for commits and issues, `diff`-fence preservation, and the anomaly-banner condition.
- Table of contents placement and Behavioral & Support Changes grouping.
- Signature + docstring fenced-code forms for functions, methods, constructor-only classes, classes with multiple methods, and enums / flags.
- New API table columns (Symbol / Description / GH / Bake) with the short-form call-shape Symbol cell, and the Kind groupings.
- Changes-to-Existing-API table columns and Kind values, plus the compact `🟢 47d` Bake-cell format for tables vs. the spelled-out form for prose.
- Audit-appendix conditional rendering.
- Output style hard constraints — no em dashes, no skill-internal terminology or Phase names, no end-of-report markers, every GH ref rendered as a full markdown hyperlink, and no manufactured comparison to previously-shipped patch-release fixes.

### 7c — Write output to chosen destination

The destination was decided in Phase 1: one of `local`, `new-gist`, or `revise-gist:<id>`. Act on that choice.

**Filename conventions:**
- Local file (at repo root): `newton-<version-string>-<prerelease|rc|retrospective>-report-<today>.md` — dated, user-facing.
- Gist file (inside the gist): `newton-<version-string>-<prerelease|rc|retrospective>-report.md` — no date. Stable name so later runs can revise the same gist in place.

**Stable gist description** (used when creating a new gist; also the matching key for Phase 1):
```text
Newton <version-string> <Pre-Release|Release Candidate|Retrospective> Report
```

**If destination is `local`:**
1. Write to `$(git rev-parse --show-toplevel)/<local-filename>` using the Write tool.
2. Print a one-line chat summary:
   - Local path.
   - Headline counts (N new APIs, K breaking, M changed, L behavioral, F fixes).

**If destination is `new-gist`:**
1. Write to `/tmp/<gist-filename>` (stable name, no date) using the Write tool.
2. Create the gist:
   ```bash
   gh gist create --desc "<stable-desc>" /tmp/<gist-filename>
   ```
   Capture the gist URL from stdout.
3. Delete `/tmp/<gist-filename>` so no local artifact remains.
4. Print a one-line chat summary:
   - Gist URL.
   - Headline counts.

**If destination is `revise-gist:<id>`:**
1. Write to `/tmp/<gist-filename>` (same stable name the existing gist already uses) using the Write tool.
2. Revise the gist:
   ```bash
   gh gist edit <id> /tmp/<gist-filename>
   ```
   `gh` matches the basename to the existing file in the gist and replaces its contents; the prior version is preserved in the gist's git history. Do NOT pass `--desc` — keeping the description stable is what lets the next run match this gist again.
3. Delete `/tmp/<gist-filename>`.
4. Print a one-line chat summary:
   - Gist URL (`https://gist.github.com/<id>`).
   - Note: "revised in place; prior versions kept in gist git history".
   - Headline counts.

Never pass `--public`. Never file a destination the user did not choose.

## Regexes and parsing rules (inline reference)

- GH ref: `\bGH-(\d+)` — word boundary prevents matching inside other identifiers.
- Bare PR ref: `(?<![\w/])#(\d+)\b` — Newton entries occasionally reference PR numbers as `#NNNN`. Treat as a GH ref candidate.
- CHANGELOG Unreleased section header: `## [Unreleased]` — may have trailing date text; match prefix only.
- CHANGELOG subsection headers: `### Added`, `### Removed`, `### Deprecated`, `### Changed`, `### Fixed`, `### Documentation`.
- Symbol extraction from entry text: backtick-quoted `newton.X`, `newton.X.Y`, `ClassName.method`, bare `ClassName` (capitalized identifier), bare `snake_case_name()`. The FIRST backtick-quoted symbol in the bullet is usually the primary subject.
- Migration-guidance regex (Phase 5a 📐): case-insensitive search for `use\s+\x60`, `in favor of`, `renamed? to`, `replaced? by`, `switch to`, `migrate to`, `prefer\s+\x60` (where `\x60` matches a backtick).

## Failure modes

- **CHANGELOG entry with zero GH refs AND no backing commit found via `git log -S`**: render in the review notes with "no associated commit found — verify".
- **`Added` entry names a symbol not resolvable at HEAD**: render with a ⚠️ note; do NOT emit synthetic `newton.*` stub names.
- **`upstream/` remote missing**: substitute `origin/`. Note the substitution in the report header.
- **Release branch exists but contains no new commits past main**: treat as head==main effectively; skip cherry-pick detection.
- **CHANGELOG `[Unreleased]` missing or empty**: header warns: "No `[Unreleased]` entries found in CHANGELOG.md."
- **`gh` installed but not authenticated**: treat as `gh` unavailable; skip gist matching and gist prompt; add one-line chat note.
- **`pyproject.toml` version is non-standard** (not matching `X.Y.ZdevN`, `X.Y.ZrcN`, or `X.Y.Z`): treat as pre-release mode, record the raw string in the header, and continue.
