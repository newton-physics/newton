# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import copy
import importlib.util
import json
import os
import subprocess
import unittest
from pathlib import Path
from unittest import mock

SCRIPT = Path(__file__).resolve().parents[1] / "maintainer_approve.py"
spec = importlib.util.spec_from_file_location("maintainer_approve", SCRIPT)
approval = importlib.util.module_from_spec(spec)
spec.loader.exec_module(approval)

SHA = "a" * 40
BASE_SHA = "c" * 40
BOT = "newton-maintainer-approval[bot]"
PREFIX = "repos/newton-physics/newton"
PULL = f"{PREFIX}/pulls/42"
COMMENT = f"{PREFIX}/issues/comments/123"
MEMBER = "orgs/newton-physics/teams/maintainers/memberships/reviewer"


class FakeGitHub:
    def __init__(self):
        self.event = {
            "action": "created",
            "repository": {"full_name": "newton-physics/newton"},
            "issue": {"number": 42, "pull_request": {}},
            "sender": {"id": 10},
            "comment": {
                "id": 123,
                "body": f"/maintainer-approve {SHA} Fix the missing assertion.",
                "user": {"id": 10, "login": "reviewer", "type": "User"},
                "created_at": "2026-10-19T10:00:00Z",
                "updated_at": "2026-10-19T10:00:00Z",
            },
        }
        self.comment = copy.deepcopy(self.event["comment"])
        self.pull = {
            "state": "open",
            "draft": False,
            "user": {"id": 20},
            "head": {"sha": SHA},
            "base": {"repo": {"full_name": "newton-physics/newton"}, "ref": "main", "sha": BASE_SHA},
            "changed_files": 1,
        }
        self.membership = {"state": "active"}
        self.files = [{"filename": "newton/_src/example.py"}]
        self.reviews = []
        self.calls = []
        self.fail_endpoint = None
        self.after_post = lambda: None
        self.pull_reads = 0
        self.before_second_pull = lambda: None

    def request(self, method, endpoint, body=None):
        self.calls.append((method, endpoint, copy.deepcopy(body)))
        if endpoint == self.fail_endpoint:
            raise approval.Rejected("GitHub request failed")
        if method == "POST":
            self.after_post()
            return {"id": 500}
        if method == "PUT":
            return {"state": "DISMISSED"}
        if endpoint == COMMENT:
            return copy.deepcopy(self.comment)
        if endpoint == MEMBER:
            return copy.deepcopy(self.membership)
        if endpoint == f"{PREFIX}/compare/{BASE_SHA}...{SHA}?per_page=1&page=1":
            return {"base_commit": {"sha": BASE_SHA}, "files": copy.deepcopy(self.files)}
        if endpoint == PULL:
            self.pull_reads += 1
            if self.pull_reads == 2:
                self.before_second_pull()
            return copy.deepcopy(self.pull)
        raise AssertionError((method, endpoint))

    def list(self, endpoint):
        self.calls.append(("GET", endpoint, None))
        if endpoint.endswith("/files"):
            return copy.deepcopy(self.files)
        if endpoint.endswith("/reviews"):
            return copy.deepcopy(self.reviews)
        raise AssertionError(endpoint)


class TestMaintainerApprove(unittest.TestCase):
    def setUp(self):
        self.api = FakeGitHub()

    def run_approval(self):
        return approval.approve(self.api.event, self.api, bot_login=BOT)

    def assert_rejected_without_write(self):
        with self.assertRaises(approval.Rejected):
            self.run_approval()
        self.assertFalse(any(method != "GET" for method, _, _ in self.api.calls))

    def test_authorized_command_approves_exact_head_without_merging(self):
        """Approve the requested head and retain an attributable audit record."""
        self.run_approval()
        writes = [(method, path, body) for method, path, body in self.api.calls if method != "GET"]
        self.assertEqual(len(writes), 1)
        method, path, body = writes[0]
        self.assertEqual((method, path), ("POST", f"{PULL}/reviews"))
        self.assertEqual(body["event"], "APPROVE")
        self.assertEqual(body["commit_id"], SHA)
        self.assertIn("@reviewer", body["body"])
        self.assertIn("#issuecomment-123", body["body"])
        self.assertIn("not an independent review", body["body"])
        self.assertEqual(self.api.pull_reads, 3)

    def test_rejects_non_members_and_unavailable_authorization(self):
        """Fail closed on membership denial and API failures."""
        for state in ("pending", "inactive", "api-error"):
            with self.subTest(state=state):
                self.setUp()
                self.api.membership["state"] = state
                if state == "api-error":
                    self.api.fail_endpoint = MEMBER
                self.assert_rejected_without_write()

    def test_rejects_pr_author_even_when_maintainer(self):
        """Reject a maintainer requesting approval for their own PR."""
        self.api.pull["user"]["id"] = 10
        self.assert_rejected_without_write()

    def test_rejects_stale_sha(self):
        """Reject commands for an earlier head."""
        self.api.pull["head"]["sha"] = "b" * 40
        self.assert_rejected_without_write()

    def test_rejects_malformed_or_multiline_commands(self):
        """Require an exact command, full SHA, and bounded single-line reason."""
        for body in (
            f"quoted /maintainer-approve {SHA} reason",
            "/maintainer-approve abc123 reason",
            f"/maintainer-approve {SHA}",
            f"/maintainer-approve {SHA} reason\nsecond line",
            f"/maintainer-approve {SHA} {'x' * 501}",
        ):
            with self.subTest(body=body):
                self.setUp()
                self.api.event["comment"]["body"] = body
                self.assert_rejected_without_write()

    def test_rejects_edited_and_deleted_comments(self):
        """Revalidate comment existence and immutable command content."""
        for change in ("body", "updated_at", "deleted"):
            with self.subTest(change=change):
                self.setUp()
                if change == "deleted":
                    self.api.fail_endpoint = COMMENT
                else:
                    self.api.comment[change] += "changed"
                self.assert_rejected_without_write()

    def test_rejects_non_created_non_pr_and_wrong_repository_events(self):
        """Reject events outside the authorized command surface."""
        for change in ("edited", "issue", "fork", "sender", "bot"):
            with self.subTest(change=change):
                self.setUp()
                if change == "edited":
                    self.api.event["action"] = "edited"
                elif change == "issue":
                    del self.api.event["issue"]["pull_request"]
                elif change == "fork":
                    self.api.event["repository"]["full_name"] = "someone/newton"
                elif change == "sender":
                    self.api.event["sender"]["id"] = 99
                else:
                    self.api.event["comment"]["user"]["type"] = "Bot"
                self.assert_rejected_without_write()

    def test_rejects_closed_draft_and_other_target_prs(self):
        """Restrict the exception to open non-draft PRs targeting main."""
        for change in ("closed", "draft", "release", "repository"):
            with self.subTest(change=change):
                self.setUp()
                if change == "closed":
                    self.api.pull["state"] = "closed"
                elif change == "draft":
                    self.api.pull["draft"] = True
                elif change == "release":
                    self.api.pull["base"]["ref"] = "release-1.6"
                else:
                    self.api.pull["base"]["repo"]["full_name"] = "someone/newton"
                self.assert_rejected_without_write()

    def test_rejects_sensitive_paths_including_renames(self):
        """Require independent review for automation and policy changes."""
        for path in (".github/CODEOWNERS", "scripts/ci/maintainer_approve.py", "AGENTS.md", ".agents/policy.md"):
            for field in ("filename", "previous_filename"):
                with self.subTest(path=path, field=field):
                    self.setUp()
                    self.api.files[0][field] = path
                    self.assert_rejected_without_write()

    def test_rejects_incomplete_changed_file_list(self):
        """Fail closed when GitHub truncates or changes the file list."""
        self.api.pull["changed_files"] = 3001
        self.assert_rejected_without_write()

    def test_temporary_head_cannot_hide_protected_changes(self):
        """Inspect the requested commit even if a temporary head has allowed files."""
        self.api.files = [{"filename": ".github/workflows/untrusted.yml"}]
        original_list = self.api.list

        def list_with_head_flip(endpoint):
            if endpoint.endswith("/files"):
                # The mutable PR-files endpoint can observe head B while every
                # metadata check sees requested head A, which has protected files.
                self.api.pull["head"]["sha"] = "b" * 40
                result = [{"filename": "newton/_src/example.py"}]
                self.api.pull["head"]["sha"] = SHA
                return result
            return original_list(endpoint)

        with mock.patch.object(self.api, "list", side_effect=list_with_head_flip):
            self.assert_rejected_without_write()

    def test_comparison_file_limit_fails_closed(self):
        """Reject the comparison cap even when the PR file count matches it."""
        self.api.files = [{"filename": f"newton/file{index}.py"} for index in range(300)]
        self.api.pull["changed_files"] = 300
        self.assert_rejected_without_write()

    def test_complete_comparison_below_limit_can_be_approved(self):
        """Inspect all files in a complete comparison below the API limit."""
        self.api.files = [{"filename": f"newton/file{index}.py"} for index in range(299)]
        self.api.pull["changed_files"] = 299
        self.run_approval()
        self.assertTrue(any(method == "POST" for method, _, _ in self.api.calls))

    def test_base_change_before_submission_prevents_approval(self):
        """Reject a changed base after inspecting the pinned comparison."""
        self.api.before_second_pull = lambda: self.api.pull["base"].update(sha="d" * 40)
        self.assert_rejected_without_write()

    def test_comparison_failure_does_not_fall_back_to_mutable_files(self):
        """Reject unavailable immutable comparisons without a mutable fallback."""
        self.api.fail_endpoint = f"{PREFIX}/compare/{BASE_SHA}...{SHA}?per_page=1&page=1"
        self.assert_rejected_without_write()

    def test_rejects_wrong_comparison_base(self):
        """Reject a comparison response that identifies another base commit."""
        request = self.api.request

        def wrong_base(method, endpoint, body=None):
            result = request(method, endpoint, body)
            if "/compare/" in endpoint:
                result["base_commit"]["sha"] = "d" * 40
            return result

        with mock.patch.object(self.api, "request", side_effect=wrong_base):
            self.assert_rejected_without_write()

    def test_replay_does_not_resurrect_dismissed_approval(self):
        """Deduplicate consumed commands even after review dismissal."""
        self.api.reviews = [
            {
                "user": {"login": BOT, "type": "Bot"},
                "body": f"<!-- maintainer-approval:123:{SHA} -->",
                "state": "DISMISSED",
                "commit_id": SHA,
            }
        ]
        self.assertIn("already processed", self.run_approval())
        self.assertFalse(any(method != "GET" for method, _, _ in self.api.calls))

    def test_spoofed_marker_cannot_suppress_app_approval(self):
        """Accept deduplication markers only from the dedicated bot."""
        self.api.reviews = [
            {
                "user": {"login": "someone", "type": "User"},
                "body": f"<!-- maintainer-approval:123:{SHA} -->",
            }
        ]
        self.run_approval()
        self.assertTrue(any(method == "POST" for method, _, _ in self.api.calls))

    def test_another_command_cannot_add_a_second_app_approval(self):
        """Avoid submitting another approval for an already approved head."""
        self.api.reviews = [
            {
                "user": {"login": BOT, "type": "Bot"},
                "body": f"<!-- maintainer-approval:122:{SHA} -->",
                "state": "APPROVED",
                "commit_id": SHA,
            }
        ]
        self.assertIn("already approved", self.run_approval())
        self.assertFalse(any(method != "GET" for method, _, _ in self.api.calls))

    def test_delegated_approval_does_not_modify_human_reviews(self):
        """Leave another reviewer's requested changes for GitHub to enforce."""
        self.api.reviews = [
            {
                "user": {"login": "other-maintainer", "type": "User"},
                "body": "Please address the regression.",
                "state": "CHANGES_REQUESTED",
                "commit_id": SHA,
            }
        ]
        self.run_approval()
        self.assertEqual(
            [(method, path) for method, path, _ in self.api.calls if method != "GET"],
            [("POST", f"{PULL}/reviews")],
        )

    def test_push_before_submission_prevents_approval(self):
        """Recheck the head immediately before submitting the review."""
        self.api.before_second_pull = lambda: self.api.pull["head"].update(sha="b" * 40)
        self.assert_rejected_without_write()

    def test_change_during_submission_dismisses_created_review(self):
        """Roll back approval when head or authorization changes during submission."""
        for change in ("push", "base", "membership", "comment", "api-error"):
            with self.subTest(change=change):
                self.setUp()

                def change_after_post(change=change):
                    if change == "push":
                        self.api.pull["head"]["sha"] = "b" * 40
                    elif change == "base":
                        self.api.pull["base"]["sha"] = "d" * 40
                    elif change == "membership":
                        self.api.membership["state"] = "pending"
                    elif change == "comment":
                        self.api.comment["body"] += " edited"
                    else:
                        self.api.fail_endpoint = COMMENT

                self.api.after_post = change_after_post
                with self.assertRaises(approval.Rejected):
                    self.run_approval()
                writes = [(method, path) for method, path, _ in self.api.calls if method != "GET"]
                self.assertEqual(writes, [("POST", f"{PULL}/reviews"), ("PUT", f"{PULL}/reviews/500/dismissals")])

    def test_api_uses_json_stdin_without_shell_interpolation(self):
        """Keep comment text as JSON data and omit credentials from arguments."""
        reason = "literal `command` $(command) $SECRET"
        with mock.patch.object(subprocess, "run", return_value=mock.Mock(returncode=0, stdout="{}")) as run:
            approval.GitHub().request("POST", f"{PULL}/reviews", {"body": reason})
        args, kwargs = run.call_args
        self.assertNotIn("shell", kwargs)
        self.assertNotIn(reason, args[0])
        self.assertEqual(json.loads(kwargs["input"]), {"body": reason})

    def test_api_failure_is_not_retried_or_logged_with_response(self):
        """Avoid duplicate approvals and response leakage after API errors."""
        with mock.patch.object(subprocess, "run", return_value=mock.Mock(returncode=1, stderr="sensitive")) as run:
            with self.assertRaisesRegex(approval.Rejected, "no approval retry") as error:
                approval.GitHub().request("POST", f"{PULL}/reviews", {})
        self.assertEqual(run.call_count, 1)
        self.assertNotIn("sensitive", str(error.exception))

    def test_review_listing_paginates(self):
        """Find reviews beyond the first API page before deduplication."""
        api = approval.GitHub()
        with mock.patch.object(api, "request", side_effect=[[{}] * 100, [{}]]) as request:
            self.assertEqual(len(api.list(f"{PULL}/reviews")), 101)
        self.assertEqual(request.call_args.args, ("GET", f"{PULL}/reviews?per_page=100&page=2"))

    def test_reruns_cannot_reuse_old_event_authorization(self):
        """Require a fresh command instead of rerunning an old workflow."""
        with mock.patch.dict(os.environ, {"GITHUB_EVENT_NAME": "issue_comment", "GITHUB_RUN_ATTEMPT": "2"}):
            with self.assertRaisesRegex(approval.Rejected, "reruns require a new command"):
                approval.main()


if __name__ == "__main__":
    unittest.main(verbosity=2)
