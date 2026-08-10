# Changelog fragments

Pull requests with user-facing changes add changelog fragments instead of
editing `CHANGELOG.md`. Towncrier combines the fragments on a release branch.

## Choose an identifier

Use a GitHub issue number when the work has an issue:

```text
3607.added.md
3607.fixed.md
```

The number identifies an issue, not the pull request. This lets a developer or
agent create the fragment before opening the pull request.

When there is no issue, use a readable orphan identifier with a short random
suffix:

```text
+camera-rays-a1b2c3d4.added.md
```

The leading `+` tells Towncrier not to create a GitHub link. Use the same
logical identifier for related entries when practical.

## Choose categories and entries

The supported categories, in rendered order, are:

1. `added`
2. `changed`
3. `deprecated`
4. `removed`
5. `fixed`

Towncrier treats each file as one entry and adds the Markdown bullet itself.
Write imperative present tense, end the entry with a period, and do not start
it with a hyphen followed by a space. Multiline entries and nested lists are
supported:

```markdown
Add camera ray helpers with:

  - Pinhole support.
  - Fisheye support.
```

Use one file per entry. Add a numeric counter when one pull request has several
entries in the same category:

```text
3607.added.md
3607.added.1.md
3607.deprecated.md
3607.deprecated.1.md
```

For `changed`, `deprecated`, and `removed`, include migration guidance in the
entry, such as: “Deprecate `Model.geo_meshes` in favor of `Model.shapes`.”

If a pull request has no user-facing change, it may add a `.skip` file to record
a one-line reason:

```text
+camera-tests-a1b2c3d4.skip
```

Towncrier ignores `.skip` files. They are optional and do not affect the
rendered changelog.

## Create and preview

Towncrier can create an issue-linked fragment:

```console
uvx --from towncrier==25.8.0 towncrier create 3607.added.md \
  --content "Add camera ray helpers."
```

For an orphan, supply the readable random identifier directly. Towncrier's
shorter `+.added.md` form also works, but generates an opaque random name.

Preview the rendered output without modifying `CHANGELOG.md` or deleting
fragments:

```console
uvx --from towncrier==25.8.0 towncrier build --draft \
  --version X.Y.Z --date YYYY-MM-DD
```

`--draft` is the safe command to run during development. CI runs the same
preview to confirm that Towncrier can render the current fragment set.

## Assemble and maintain a release

Build only on `release-X.Y`. After the initial release scope audit, assemble the
current fragments early during RC stabilization. First render and approve the
non-mutating preview, then run the mutating build:

```console
uvx --from towncrier==25.8.0 towncrier build --draft \
  --version X.Y.Z --date YYYY-MM-DD
uvx --from towncrier==25.8.0 towncrier build --yes \
  --version X.Y.Z --date YYYY-MM-DD
git rm --ignore-unmatch "changelog/*.skip"
git add -A CHANGELOG.md changelog
```

Towncrier inserts the dated section below `[Unreleased]` and deletes the
rendered `.md` fragments. It ignores `.skip` files, so the release commit
deletes those explicitly. Review the staged diff before committing.

The first Towncrier release is a one-time transition. The insertion marker is
already above the legacy `[Unreleased]` categories, so the old entries remain
under the first generated release title. During that release audit, merge any
duplicate category headings and verify that every legacy entry is retained.

Open the build as a changelog-only pull request with the existing
`release-management` label. Run the `release-changelog` cleanup against the
assembled section for completeness, grouping, deduplication, wording,
categories, and migration guidance.

If more changes are cherry-picked before tagging, render only their fragments
with `--draft`, fold the previewed entries into the existing dated section,
delete exactly those `.md` and `.skip` fragments, and rerun the changelog and
release audits. Final GA preparation verifies the completed section and that no
release fragments remain.

After the release, cherry-pick every changelog-management commit in order onto
a changelog-only branch from `main`: the initial build, editorial cleanup, and
later fragment additions. Open another `release-management` pull request. Git
removes only fragments that shipped; fragments added to `main` after the
release branch was cut remain pending for the next release.
