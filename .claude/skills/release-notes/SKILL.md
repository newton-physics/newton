---
name: release-notes
description: Use when drafting or reviewing Newton GitHub Release notes, release announcements, or high-level release summaries for patch, minor, RC, or final releases.
---

# Newton Release Notes

Draft concise GitHub Release notes that explain why a Newton release exists and
what users should know. Do not reproduce the full changelog; link to it.

## Workflow

1. Identify the artifact, target version, and release type:
   - **GitHub release description**: public release page. Follow Newton/Warp
     release-page style and include user-actionable migration sections.
   - **Internal release announcement**: short internal email/doc. Use release
     metadata first, then grouped bullets such as `New Features`,
     `Performance Improvements`, and `Developer Experience`.
   - If the user wants both and the content should converge, keep one document
     only when explicitly requested; otherwise maintain separate drafts.
   - Patch / bugfix releases: emphasize fixes, compatibility, and stability.
   - Minor / feature releases: emphasize major user-facing capabilities.
   - RCs: state that this is a release candidate and what needs validation.
     If drafting final-release text from an RC tag or release branch, do not
     mention the RC; use the RC only as the temporary source of truth.
2. Read the matching `CHANGELOG.md` section from the release tag or release
   branch. Do not rely on `main` unless the release is actually cut from `main`.
3. Determine the previous release tag:
   - Patch release `X.Y.Z`, `Z > 0`: use the highest earlier `vX.Y.<Z'>` tag.
   - Feature release `X.Y.0`: use the highest `vX.<Y-1>.*` tag. If `Y == 0`,
     use the highest tag from the previous major.
4. Compare the release range against the previous release:
   ```bash
   git log --no-merges --oneline --cherry-pick --right-only v<previous>...v<target>
   git diff --name-status v<previous>..v<target>
   ```
   Use the symmetric-difference `...` log form for counting and scanning commits
   when a release branch contains cherry-picks; it avoids overstating scope with
   equivalent patches already present in the previous release. Use the normal
   `..` diff for changed files. If the final tag does not exist yet, use the
   latest RC tag or `upstream/release-X.Y` as the temporary target.
5. Inspect the PRs behind candidate highlights. Commit subjects usually carry
   `(#NNNN)`; use PR bodies when needed to understand impact and any linked
   issues.
6. Check prior release-note and announcement formats when available:
   - Newton release pages, e.g. `v1.2.0`, for the local public-release shape.
   - Warp release pages for `Announcements`, `Upcoming removals`, and
     acknowledgement patterns.
   - Internal announcements for short internal release-email structure.
7. Draft a high-level overview and a short highlights list. Keep only
   consumer-relevant items. Omit CI, workflow, README layout, release-link
   pinning, and other internal/docs polish unless the user explicitly asks or it
   affects library users.
8. Keep a review-ready release-branch draft. After tagging, refresh it against
   the final tag and inspect the workflow-created draft GitHub Release. Publish
   only after verifying the final artifact and links per `docs/guide/release.rst`.

## Linking Rules

- Link the changelog to the final release tag:
  `https://github.com/newton-physics/newton/blob/vX.Y.Z/CHANGELOG.md#anchor`
- If drafting before the final tag exists, use `release-X.Y` or the RC tag
  temporarily, but note that the published release should use the final tag.
- Never link the changelog to `main` for a release branch unless `main` is the
  authoritative release ref; `main` may have newer unreleased changes.
- Add one GitHub reference to each highlight, usually the PR. Use an issue only
  when there is no useful PR or the issue is the canonical context.
- Avoid linking both an issue and a PR for the same highlight unless that extra
  context is necessary.
- In GitHub Release notes, render same-repository references as plain `#NNNN`
  instead of full markdown links. GitHub auto-links them on the release page, and
  the compact form matches Warp's release-note style.
- Link in-tree examples or docs on the release tag URL, never `main`, so links
  do not drift after publication.
- For dependency summaries, prefer a link to the relevant `pyproject.toml`
  compare over listing every changed constraint:
  `https://github.com/newton-physics/newton/compare/v<previous>...v<target>?diff=split`
  Mention only high-signal dependency changes in bullets.

## Style

- Start with a short paragraph naming the version, release type, and purpose.
- Use a `## Highlights` section with 3-6 grouped bullets.
- For bugfix releases, keep notes slim: no `## New features` section unless a
  genuinely notable capability shipped in the patch. The highlights are a
  categorized digest of fixes.
- For feature releases, group related features by user workflow and order by
  impact. Lead with the most important user-facing capability, not API names.
- Prefer user impact over implementation detail.
- Keep bullets compact: one bold label, one or two explanatory sentences, then
  the `#NNNN` reference.
- Treat the changelog as source material, not prose to copy. Translate internal
  implementation terms into user-facing impact.
- Include `## Announcements` only for changes users must act on, such as
  removals, deprecations, platform-support changes, or dependency constraints.
- For public feature releases, include focused sections for:
  - breaking changes and removals,
  - new deprecations,
  - upcoming removals,
  - dependency updates,
  - acknowledgements.
  Keep these concise; the changelog carries the exhaustive detail.
- Include `## Notes` only for compatibility, install, or migration information
  users need. Drop the section when there is nothing useful to say.
- Include `## Acknowledgments` when there are meaningful contributions from
  outside Newton maintainers and project-member groups. Omit trivial typo-only
  or formatting-only changes.
- For patch releases, say whether the release is intended to be API-compatible
  with the previous patch/minor when that is true.
- Avoid em dashes in rendered prose.
- Keep raw Markdown soft-wrapped: use one physical line per paragraph and list item; let viewers wrap it to their window width.

## Dependency Updates

Do not default to a large table. Keep dependency updates high signal:

- Call out major/minor runtime baseline bumps, new extras, dependency caps that
  affect users, and removed caps when they unblock compatibility.
- Link to the `pyproject.toml` compare for the complete detail.
- If the prior patch release changed dependencies, compare against the closest
  prior release tag, not the earlier minor tag.

## Acknowledgments

Before listing outside contributors, verify they are not in Newton's maintainer
or project-member groups.

1. Read `newton-governance/CONTRIBUTORS.md` or the current governance source
   when reachable. Treat it as the authority for maintainer and project-member
   membership.
2. If GitHub access is available and governance points to org teams, query the
   current team slugs from GitHub rather than relying on a hardcoded list:
   ```bash
   gh api orgs/newton-physics/teams --paginate --jq '.[] | [.slug, .name] | @tsv'
   gh api orgs/newton-physics/teams/<team-slug>/members --paginate --jq '.[] | .login'
   ```
   Prioritize teams named by governance, or teams whose current names clearly
   identify maintainers, TSC, project members, or project-member organizations.
3. Cross-check candidate PR authors with `gh pr view <number> --json author`
   and commit emails. Commit email/company is a hint, not a substitute for the
   team/governance check.
4. Phrase the section accurately, for example "outside the Newton maintainer
   and project-member groups" when that is the filter used.

## Template

```markdown
# Newton vX.Y.Z

Newton vX.Y.Z is a patch release following vX.Y.W. It focuses on bug fixes and
compatibility updates for the X.Y release line, especially around <areas>.

For the complete list of changes, see the [changelog](https://github.com/newton-physics/newton/blob/vX.Y.Z/CHANGELOG.md#anchor).

## Highlights

- **<User-facing fix or improvement>.** <Brief impact statement.> (#NNNN)

- **<Another user-facing fix or improvement>.** <Brief impact statement.> (#NNNN)

## Announcements

<Only include this section when there is a removal, deprecation,
platform-support change, dependency constraint, or other user-actionable
announcement. Drop it otherwise.>

## Notes

This release is intended to be API-compatible with Newton vX.Y.W. No breaking
changes are expected.
```
