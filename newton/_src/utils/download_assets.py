# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import hashlib
import os
import shutil
import stat
import time
from pathlib import Path

try:
    from warp.thirdparty.appdirs import user_cache_dir
except (ImportError, ModuleNotFoundError):
    from warp._src.thirdparty.appdirs import user_cache_dir


def _get_newton_cache_dir() -> str:
    """Gets the persistent Newton cache directory."""
    if "NEWTON_CACHE_PATH" in os.environ:
        return os.environ["NEWTON_CACHE_PATH"]
    return user_cache_dir("newton", "newton-physics")


def _handle_remove_readonly(func, path, exc):
    """Error handler for Windows readonly files during shutil.rmtree()."""
    if os.path.exists(path):
        # Make the file writable and try again
        os.chmod(path, stat.S_IWRITE)
        func(path)


def _safe_rmtree(path):
    """Safely remove directory tree, handling Windows readonly files."""
    if os.path.exists(path):
        shutil.rmtree(path, onerror=_handle_remove_readonly)


def _get_latest_commit_via_git(git_url: str, branch: str) -> str | None:
    """Resolve latest commit SHA for a branch via 'git ls-remote'."""
    try:
        import git  # noqa: PLC0415

        out = git.cmd.Git().ls_remote("--heads", git_url, branch)
        # Output format: "<sha>\trefs/heads/<branch>\n"
        return out.split()[0] if out else None
    except Exception:
        # Fail silently on any error (offline, auth issue, etc.)
        return None


def _read_cached_commit(cache_folder: Path) -> str | None:
    """Return HEAD commit of cached repo, or None on failure."""
    try:
        import git  # noqa: PLC0415

        repo = git.Repo(cache_folder)
        try:
            return repo.head.commit.hexsha
        finally:
            repo.close()
    except Exception:
        return None


def _stamp_fresh(stamp_file: Path, ttl_seconds: int) -> bool:
    """True if stamp file exists and is younger than TTL."""
    try:
        return stamp_file.exists() and (time.time() - stamp_file.stat().st_mtime) < ttl_seconds
    except OSError:
        return False


def _touch(path: Path) -> None:
    """Create/refresh a file's mtime; ignore filesystem errors."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch(exist_ok=True)
    except OSError:
        pass


def download_git_folder(
    git_url: str, folder_path: str, cache_dir: str | None = None, branch: str = "main", force_refresh: bool = False
) -> Path:
    """
    Downloads a specific folder from a git repository into a local cache.

    Uses the cached version when up-to-date; otherwise refreshes by comparing the
    cached repo's HEAD with the remote's latest commit (via 'git ls-remote').

    Args:
        git_url: The git repository URL (HTTPS or SSH)
        folder_path: The path to the folder within the repository (e.g., "assets/models")
        cache_dir: Directory to cache downloads.
            If ``None``, the path is determined in the following order:
            1. ``NEWTON_CACHE_PATH`` environment variable.
            2. System's user cache directory (via ``appdirs.user_cache_dir``).
        branch: Git branch/tag/commit to checkout (default: "main")
        force_refresh: If True, re-downloads even if cached version exists

    Returns:
        Path to the downloaded folder in the local cache

    Raises:
        ImportError: If git package is not available
        RuntimeError: If git operations fail

    Example:
        >>> folder_path = download_git_folder("https://github.com/user/repo.git", "assets/models", cache_dir="./cache")
        >>> print(f"Downloaded to: {folder_path}")
    """
    try:
        import git  # noqa: PLC0415
        from git.exc import GitCommandError  # noqa: PLC0415
    except ImportError as e:
        raise ImportError(
            "GitPython package is required for downloading git folders. Install it with: pip install GitPython"
        ) from e

    # Set up cache directory
    if cache_dir is None:
        cache_dir = _get_newton_cache_dir()
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    # Create a unique folder name based on git URL, folder path, and branch
    url_hash = hashlib.md5(f"{git_url}#{folder_path}#{branch}".encode()).hexdigest()[:8]
    repo_name = Path(git_url.rstrip("/")).stem.replace(".git", "")
    folder_name = folder_path.replace("/", "_").replace("\\", "_")
    cache_folder = cache_path / f"{repo_name}_{folder_name}_{url_hash}"

    target_folder = cache_folder / folder_path

    # 1. Handle force_refresh
    if force_refresh and cache_folder.exists():
        _safe_rmtree(cache_folder)

    # 2. Check cache validity using Git
    # TTL to avoid repeated network checks
    stamp_file = cache_folder / ".newton_last_check"
    ttl_seconds = 3600

    is_cached = target_folder.exists() and (cache_folder / ".git").exists()
    if is_cached and not force_refresh:
        if _stamp_fresh(stamp_file, ttl_seconds):
            return target_folder

        current_commit = _read_cached_commit(cache_folder)
        latest_commit = _get_latest_commit_via_git(git_url, branch)

        # If we cannot determine latest (offline, etc.) or they match, use cache
        if latest_commit is None or (current_commit is not None and latest_commit == current_commit):
            _touch(stamp_file)
            return target_folder

        # Different commit detected: clear cache to refresh
        print(
            f"New version of {folder_path} found (cached: {str(current_commit)[:7] if current_commit else 'unknown'}, "
            f"latest: {latest_commit[:7]}). Refreshing..."
        )
        _safe_rmtree(cache_folder)

    # 3. Download if not cached (or if cache was just cleared)
    try:
        # Clone the repository with sparse checkout
        print(f"Cloning {git_url} (branch: {branch})...")
        repo = git.Repo.clone_from(
            git_url,
            cache_folder,
            branch=branch,
            depth=1,  # Shallow clone for efficiency
        )

        # Configure sparse checkout to only include the target folder
        sparse_checkout_file = cache_folder / ".git" / "info" / "sparse-checkout"
        sparse_checkout_file.parent.mkdir(parents=True, exist_ok=True)

        with open(sparse_checkout_file, "w") as f:
            f.write(f"{folder_path}\n")

        # Apply sparse checkout configuration
        with repo.config_writer() as config:
            config.set_value("core", "sparseCheckout", "true")

        # Re-read the index to apply sparse checkout
        repo.git.read_tree("-m", "-u", "HEAD")

        # Verify the folder exists
        if not target_folder.exists():
            raise RuntimeError(f"Folder '{folder_path}' not found in repository {git_url}")

        _touch(stamp_file)

        repo.close()

        print(f"Successfully downloaded folder to: {target_folder}")
        return target_folder

    except GitCommandError as e:
        # Clean up on failure
        if cache_folder.exists():
            _safe_rmtree(cache_folder)
        raise RuntimeError(f"Git operation failed: {e}") from e
    except Exception as e:
        # Clean up on failure
        if cache_folder.exists():
            _safe_rmtree(cache_folder)
        raise RuntimeError(f"Failed to download git folder: {e}") from e


def clear_git_cache(cache_dir: str | None = None) -> None:
    """
    Clears the git download cache directory.

    Args:
        cache_dir: Cache directory to clear.
            If ``None``, the path is determined in the following order:
            1. ``NEWTON_CACHE_PATH`` environment variable.
            2. System's user cache directory (via ``appdirs.user_cache_dir``).
    """
    if cache_dir is None:
        cache_dir = _get_newton_cache_dir()

    cache_path = Path(cache_dir)
    if cache_path.exists():
        _safe_rmtree(cache_path)
        print(f"Cleared git cache: {cache_path}")
    else:
        print("Git cache directory does not exist")


def download_asset(asset_folder: str, cache_dir: str | None = None, force_refresh: bool = False) -> Path:
    """
    Downloads a specific folder from the newton-assets GitHub repository into a local cache.

    Args:
        asset_folder: The folder within the repository to download (e.g., "assets/models")
        cache_dir: Directory to cache downloads.
            If ``None``, the path is determined in the following order:
            1. ``NEWTON_CACHE_PATH`` environment variable.
            2. System's user cache directory (via ``appdirs.user_cache_dir``).
        force_refresh: If True, re-downloads even if cached version exists

    Returns:
        Path to the downloaded folder in the local cache
    """
    return download_git_folder(
        "https://github.com/newton-physics/newton-assets.git",
        asset_folder,
        cache_dir=cache_dir,
        force_refresh=force_refresh,
    )
