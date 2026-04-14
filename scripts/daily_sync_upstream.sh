#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Daily Newton sync helper.

Usage:
  bash scripts/daily_sync_upstream.sh [--branch <name>] [--push]

Options:
  -b, --branch <name>  Research branch to rebase onto main.
                       Default: current branch if not main.
  --push               Push rebased branch with --force-with-lease.
  -h, --help           Show this help.

Behavior:
  1) Fetches upstream tags and refs
  2) Fast-forwards local main to upstream/main
  3) Rebases your research branch onto main (if provided or inferred)

Safety:
  - Refuses to run with a dirty working tree
  - Refuses to rewrite local main if main has local-only commits
EOF
}

die() {
  echo "Error: $*" >&2
  exit 1
}

ensure_clean_worktree() {
  if [[ -n "$(git status --porcelain)" ]]; then
    die "Working tree is dirty. Commit or stash changes, then rerun."
  fi
}

require_remote() {
  local remote="$1"
  git remote get-url "$remote" >/dev/null 2>&1 || die "Missing remote '$remote'."
}

SYNC_BRANCH=""
PUSH_REBASED="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -b|--branch)
      [[ $# -ge 2 ]] || die "--branch requires a value"
      SYNC_BRANCH="$2"
      shift 2
      ;;
    --push)
      PUSH_REBASED="true"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "Unknown argument: $1"
      ;;
  esac
done

git rev-parse --is-inside-work-tree >/dev/null 2>&1 || die "Not inside a git repository."
require_remote upstream

CURRENT_BRANCH="$(git symbolic-ref --quiet --short HEAD || true)"
[[ -n "$CURRENT_BRANCH" ]] || die "Detached HEAD is not supported by this script."

if [[ -z "$SYNC_BRANCH" && "$CURRENT_BRANCH" != "main" ]]; then
  SYNC_BRANCH="$CURRENT_BRANCH"
fi

if [[ -n "$SYNC_BRANCH" ]]; then
  git show-ref --verify --quiet "refs/heads/$SYNC_BRANCH" || die "Branch '$SYNC_BRANCH' does not exist locally."
  [[ "$SYNC_BRANCH" != "main" ]] || die "Sync branch cannot be 'main'. Use a research branch."
fi

ensure_clean_worktree

echo "==> Fetching upstream"
git fetch upstream --tags

git show-ref --verify --quiet refs/remotes/upstream/main || die "Remote branch upstream/main not found."

echo "==> Checking local main status"
read -r MAIN_ONLY UPSTREAM_ONLY < <(git rev-list --left-right --count main...upstream/main)

if [[ "$MAIN_ONLY" -gt 0 ]]; then
  cat >&2 <<EOF
Error: local main has $MAIN_ONLY local-only commit(s).

To preserve local work safely:
  git switch main
  git switch -c research/saved-main-work
  git switch main
  git reset --hard upstream/main

Then rerun this script.
EOF
  exit 1
fi

if [[ "$UPSTREAM_ONLY" -eq 0 ]]; then
  echo "==> main already up to date with upstream/main"
else
  echo "==> Fast-forwarding main by $UPSTREAM_ONLY commit(s)"
  git switch main
  git merge --ff-only upstream/main
fi

if [[ -n "$SYNC_BRANCH" ]]; then
  echo "==> Rebasing $SYNC_BRANCH onto main"
  git switch "$SYNC_BRANCH"
  git rebase main

  if [[ "$PUSH_REBASED" == "true" ]]; then
    require_remote origin
    echo "==> Pushing $SYNC_BRANCH with --force-with-lease"
    git push --force-with-lease origin "$SYNC_BRANCH"
  fi
fi

echo "==> Done"
if [[ -n "$SYNC_BRANCH" ]]; then
  echo "main is current, and $SYNC_BRANCH is rebased on top of main."
else
  echo "main is current. Provide --branch <research-branch> to also rebase your work branch."
fi
