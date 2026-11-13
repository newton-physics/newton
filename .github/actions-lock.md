# Workflow dependency lockfile pilot

Only `.github/workflows/pr_license_check.yml` is onboarded to
`.github/workflows/actions.lock`. The lockfile pins its direct and transitive
action dependencies, including the Go setup action used by the license checker.
The workflow retains Newton's full SHA references and version comments.

Install the pinned CLI and regenerate the pilot after changing its action
dependencies:

```bash
gh extension install github/gh-actions-lock --pin v0.4.1
gh actions-lock --no-narrow .github/workflows/pr_license_check.yml
gh actions-lock --verify .github/workflows/pr_license_check.yml
```

If the extension is already installed, check `gh actions-lock --version` before
regenerating. Use version 0.4.1 for this pilot.

Use `--no-narrow` to retain Newton's full SHA references and version comments.
The CLI's suggestion to replace bare SHA references with tags is advisory;
retain the SHA references required by Newton's coding guidelines.
Always pass the workflow path during the pilot. An unscoped generation command
would attempt to onboard the remaining workflows. The license workflow runs the
scoped verification on pull requests and merge groups using a read-only token.

Commit the generated lockfile with dependency changes; do not edit it by hand.
Review direct and nested commit changes, and investigate changes to `owner_id`
or `repo_id`. To advance a deliberately floating upstream reference, review its
history before using `--relock --no-narrow` with the same workflow path. Do not
automatically accept moved references or replace a provenance failure with a
new pin.

For an offline coverage check, use
`gh actions-lock --verify-local .github/workflows/pr_license_check.yml`.
This does not verify upstream provenance; CI uses `--verify` for that.

Repository-wide onboarding, dependency update automation, and Require lockfile
enforcement are tracked in [#5342](https://github.com/newton-physics/newton/issues/5342).
Keep enforcement disabled for workflows outside this pilot. Lockfiles cover
action code, not tools downloaded by actions, Go modules, or runner images.

See GitHub's [lockfile documentation](https://docs.github.com/en/actions/how-tos/secure-your-work/lock-workflow-dependencies).
