# Changelog fragments

Every normal pull request adds exactly one uniquely named fragment to this directory. The filename is generated before
the pull request exists, so it does not depend on a pull request number.

For a user-facing change, create a Markdown fragment:

```bash
uv run --no-project python scripts/changelog.py create camera-ray-helpers \
    --category added \
    --content "Add scalar camera ray helpers to \`SensorTiledCamera.utils\`."
```

For a change with no user-facing effect, create a skip fragment with a reviewable reason:

```bash
uv run --no-project python scripts/changelog.py create internal-test-cleanup \
    --skip "No user-facing change: reorganize tests only."
```

The generated filename has the form `<descriptive-slug>-<8 lowercase hex>.(md|skip)`. Keep that name even if the branch
or pull request title changes. The descriptive name is optional and defaults to the current branch name; the random
suffix makes the filename unique without waiting for a pull request number.

## Markdown format

The `create` command writes a one-entry starter fragment. You may edit that generated file to add more entries or
categories. A `.md` fragment may contain any number of entries under each heading and one or more of Newton's changelog
headings in this exact order. An individual entry may span multiple lines; indent every continuation line by two spaces
so it remains part of the preceding `- ` bullet:

```markdown
### Added

- Add a first capability.
- Add a second capability with additional context that is easier to read
  on an indented continuation line.

### Deprecated

- Deprecate `old_a()` in favor of `new_a()`.
- Deprecate `old_b()` in favor of `new_b()`.
  Use `new_b(compatibility=True)` to preserve the previous behavior.
- Deprecate `old_c()` in favor of `new_c()`.
```

Allowed headings are `Added`, `Changed`, `Deprecated`, `Removed`, and `Fixed`. Every entry starts with `- `, uses
imperative present tense, ends with a period, and describes user-visible behavior rather than implementation details.
`Changed`, `Deprecated`, and `Removed` entries include migration guidance. Do not repeat a heading within one fragment;
put all entries for that category together under its single heading.

Validate pending fragments with:

```bash
uv run --no-project python scripts/changelog.py validate
```

## Dry runs

The mutating `build`, `release`, and `reconcile` commands all accept `--dry-run`. A dry run performs the same validation
and prints the proposed Markdown to standard output, but it does not edit `CHANGELOG.md` or delete any fragment files:

```bash
uv run --no-project python scripts/changelog.py build --dry-run
uv run --no-project python scripts/changelog.py release --version X.Y.Z --date YYYY-MM-DD --dry-run
uv run --no-project python scripts/changelog.py reconcile --source-ref vX.Y.Z --version X.Y.Z --dry-run
```

Use the preview for review or release auditing, then rerun the same command without `--dry-run` to apply it.

## Consolidation and releases

`CHANGELOG.md` remains the durable accumulator. It always keeps the complete `[Unreleased]` section and every dated
release. Maintainers may consolidate pending fragments into `[Unreleased]` at any time:

```bash
uv run --no-project python scripts/changelog.py build --dry-run
uv run --no-project python scripts/changelog.py build
```

`build` merges entries by category, consumes the fragment files, and preserves all existing Unreleased and released
content. Repeated builds are safe. Generated invisible provenance comments connect accumulated `[Unreleased]` entries
to their original fragment IDs; do not add, edit, or remove those comments manually. The final `release` command strips
this internal metadata from the dated public release section.

For the final release, promote the complete accumulator, including any fragments not yet consolidated:

```bash
uv run --no-project python scripts/changelog.py release --version X.Y.Z --date YYYY-MM-DD --dry-run
uv run --no-project python scripts/changelog.py release --version X.Y.Z --date YYYY-MM-DD
```

Commit any editorial changes to `[Unreleased]` before promotion so later branch reconciliation can recover the exact
fragment identities from the pre-promotion commit. The dated release itself contains only public changelog prose.

After the release, reconcile the tagged changelog onto `main` while retaining main-only post-cut entries:

```bash
uv run --no-project python scripts/changelog.py reconcile --source-ref vX.Y.Z --version X.Y.Z --dry-run
uv run --no-project python scripts/changelog.py reconcile --source-ref vX.Y.Z --version X.Y.Z
```

Commit each mutating operation in a changelog-only pull request labeled `changelog-maintenance`.
