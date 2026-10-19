# Maintainer delegated approval

This opt-in workflow lets a maintainer take responsibility for small fixes they
pushed to somebody else's PR. A dedicated GitHub App submits one attributable
approval for the exact requested head. The maintainer then uses the normal merge
queue. The App never merges, enqueues, dismisses human reviews, resolves threads,
or bypasses branch rules.

This is an exception to independent review: one human explicitly takes
responsibility for both the review and their fixes. It does not represent a
second human review. Normal contributors cannot invoke the exception.

## How to use it

1. Review the PR, push any small fixes, and inspect the final diff.
2. Obtain the current full head SHA, for example with
   `gh pr view NUMBER --repo newton-physics/newton --json headRefOid --jq .headRefOid`.
3. Post a new PR comment containing one line:

   ```text
   /maintainer-approve FULL_40_CHARACTER_HEAD_SHA Fixed the missing assertion after reviewing the final diff.
   ```

   Replace the placeholder with the actual lowercase SHA and give a reason of
   1–500 characters. The comment is an explicit attestation that you reviewed and
   accept responsibility for that head. Edits and workflow reruns are not commands;
   post a new comment to retry.
4. Inspect the **Maintainer delegated approval** workflow result and the App's
   review. Rejections appear as failed workflow runs; the bot does not post
   failure comments. If the workflow is disabled, the job is skipped.
5. Add the PR to the normal merge queue once GitHub considers it eligible.

Only active members of `newton-physics/maintainers` may invoke this command.
The requester must be a human other than the PR author. The initial scope is
open, non-draft PRs targeting `newton-physics/newton:main`, including fork PRs.
Repository write access, organization membership alone, and labels confer no
authorization. The App checks current team membership through GitHub's API.

Changes under `.github/`, `.agents/`, `.claude/`, `.codex/`, and `scripts/`, plus
`AGENTS.md`, `CODING_GUIDELINES.rst`, `CONTRIBUTING.md`, and
`REVIEW_GUIDELINES.rst`, require independent review. Both sides of a rename are
checked. "Small" is the maintainer's judgment; the workflow does not infer risk
from line count. Required CODEOWNER reviews and other review requirements remain
GitHub's responsibility and can still block queue entry.

## Manual setup and rollout

Keep repository variable `MAINTAINER_APPROVAL_ENABLED` unset or `false` until the
integration tests below pass. Merging the files alone does not enable approvals.

1. **Prove compatibility in a sandbox repository first.** Adapt the repository
   constants, team, and workflow guard in the sandbox copy. Reproduce Newton's
   review rules and merge queue configuration. Confirm that this App's approval
   counts toward the required approval and latest-push requirements, while the
   human maintainer's approval satisfies any applicable CODEOWNER requirement.
   An API response with `APPROVED` is insufficient evidence. If GitHub does not
   count the App review, stop rollout and revisit the design; do not weaken the
   review rules or grant bypass permissions to make the test pass.
2. **Register an organization-owned GitHub App**, with webhooks disabled; Actions
   handles comment events. Install it only on the selected repository. Grant:
   - Repository **Pull requests: Read and write**.
   - Organization **Members: Read-only**, for private team-membership checks.
   - Repository **Metadata: Read-only** (automatic).

   Grant no contents write, administration, or ruleset bypass. The API permission
   necessarily allows more PR operations than this script uses; protect the key
   accordingly. No human personal access token or OAuth callback is needed.
3. **Create the `maintainer-approval` Actions environment.** Restrict its deployment
   branches to the selected branch `main` only, with no tag access. Required
   environment reviewers would add another approval step, so leave them off if
   preserving the one-maintainer workflow. Store the App private key as environment
   secret `MAINTAINER_APPROVAL_APP_PRIVATE_KEY`. Store the App client ID as
   environment variable `MAINTAINER_APPROVAL_APP_CLIENT_ID`. Do not store the key
   as a general repository secret. Limit environment and App administration to
   trusted maintainers/administrators; rotate keys under that ownership.
4. **Review the credential boundary.** Every trusted workflow that can access this
   environment is part of the security boundary. Protect default-branch changes,
   workflow files, supporting scripts, and CODEOWNERS through maintainer review.
   The environment's branch restriction blocks PR-branch workflows from accessing
   the key; it does not isolate one default-branch workflow from other trusted
   default-branch workflows. If that trust boundary is too broad, host the App in
   a separate restricted automation repository or service instead.
5. **Retain the review and queue protections.** Require approval of the latest
   reviewable push and dismiss stale approvals when new changes are pushed.
   Retain required reviews, CODEOWNER approvals, thread resolution, required CI,
   and merge queue requirements. Ensure required CI runs on `merge_group`.
   If review dismissal is restricted, allow this App to dismiss its own approval
   for rollback, without granting merge/ruleset bypass. Confirm that this permission
   works in the sandbox before enabling the feature.
6. **Allow the pinned token action** in organization/repository Actions policy if
   necessary. The ordinary workflow `GITHUB_TOKEN` has only contents read; the
   dedicated App installation token performs the approval. Broadly enabling
   approvals by the ordinary Actions token is not part of this setup.
7. **Merge through ordinary review**, then set repository Actions variable
   `MAINTAINER_APPROVAL_ENABLED` to the literal value `true` after the sandbox
   validation passes. The `issue_comment` workflow must be on the default branch
   before it can receive events. Conduct one supervised production smoke test.

## Required integration evidence

Unit tests exercise authorization and API orchestration, but do not establish
GitHub's ruleset enforcement. Before rollout, record sandbox evidence for:

- A maintainer pushing a small fix to another author's PR, invoking the command,
  and successfully adding it to the normal queue without bypass.
- Non-maintainers (including ordinary write collaborators), the PR author, pending
  team members, edited/deleted comments, and stale SHAs being rejected.
- A new contributor push invalidating the approval and requiring a fresh command.
- A push racing review submission: an approval for the old SHA must not make the
  new head eligible. GitHub's review API has no atomic "head must still equal SHA"
  precondition. The script checks before and after submission and attempts to
  dismiss its review on a detected change; this cannot eliminate the race window.
  Native latest-push/stale-review enforcement must close it. If it does not, stop
  rollout rather than relying on best-effort dismissal.
- Replay/duplicate commands producing no extra approval, and a dismissed review
  not being restored by replaying the same command.
- Existing requested changes, unresolved threads, missing CODEOWNER approval, and
  failed required checks continuing to block the PR as configured.
- Required CI running on the merge group and failed queue CI preventing a merge.
- A workflow on a PR branch failing to obtain the environment key.

Requests for the same PR are serialized by workflow concurrency. Pending runs
may be replaced by newer pending runs under GitHub concurrency semantics; post a
fresh command if yours was cancelled. The script does not retry failed approval
API calls: if a response is lost, inspect reviews before posting another command.
Membership changes after a completed approval do not automatically revoke it.

## Disable and recover

Set `MAINTAINER_APPROVAL_ENABLED=false` to stop new runs, cancel any in-flight run,
and inspect existing App approvals. The variable does not cancel a running job or
revoke an existing approval. Dismiss any approval that should no longer stand;
remove the PR from the queue if necessary. If credentials may be compromised,
suspend the App installation or revoke its key as well. Do not bypass the queue.

## Implementation and verification

The workflow checks out only its trusted default-branch event SHA, disables
persisted checkout credentials, and reads command data from `GITHUB_EVENT_PATH`.
It never checks out or executes PR code and never interpolates comment text into
a shell command. The official token action creates a short-lived installation
token scoped to `newton` and revokes it at job completion.

The script re-fetches the comment, team membership, and PR head; checks all changed
file paths; deduplicates App reviews; then submits an `APPROVE` review with an
explicit `commit_id`, requester, reason, and command link. It revalidates after
submission and attempts rollback if that validation fails. API/dismissal failures
fail the run and require inspection; successful rollback is not guaranteed during
an outage or when permissions are misconfigured.

Run the focused tests without installing Newton or initializing a GPU:

```console
uv run --no-project --python 3.12 -m unittest discover -s scripts/ci/tests -p test_maintainer_approve.py -v
```

Existing CI discovers these tests under `scripts/ci/tests`.

References: [App tokens in Actions](https://docs.github.com/en/apps/creating-github-apps/authenticating-with-a-github-app/making-authenticated-api-requests-with-a-github-app-in-a-github-actions-workflow),
[comment events](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#issue_comment),
[review API](https://docs.github.com/en/rest/pulls/reviews), and
[merge queue](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/configuring-pull-request-merges/managing-a-merge-queue).
