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

import errno
import hashlib
import os
import re
import shutil
import stat
import threading
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


def _safe_rename(src, dst, attempts=5, delay=0.1):
    """Rename src to dst, tolerating races where another process wins.

    If *dst* already exists (``FileExistsError`` or ``ENOTEMPTY``), the call
    returns silently — the caller should clean up *src*.  Transient OS errors
    (e.g. Windows file-lock contention) are retried up to *attempts* times.
    """
    for i in range(attempts):
        try:
            os.rename(src, dst)
            return
        except FileExistsError:
            return
        except OSError as e:
            if e.errno == errno.ENOTEMPTY:
                return
            if i < attempts - 1:
                time.sleep(delay)
            else:
                raise


def _temp_cache_path(cache_folder: Path) -> Path:
    """Return a per-process, per-thread temp path next to *cache_folder*."""
    return Path(f"{cache_folder}_p{os.getpid()}_t{threading.get_ident()}")


_TEMP_DIR_RE = re.compile(r"_(?:stale_)?p\d+_t\d+$")


def _cleanup_stale_temp_dirs(cache_folder: Path, max_age: float = 3600.0) -> None:
    """Remove orphaned temp/stale directories left by crashed processes.

    Scans the parent directory for siblings matching the temp dir naming pattern
    (``*_p{pid}_t{tid}`` or ``*_stale_p{pid}_t{tid}``) whose mtime is older
    than *max_age* seconds.  Safe to call concurrently — ``_safe_rmtree``
    tolerates races.
    """
    parent = cache_folder.parent
    prefix = cache_folder.name
    now = time.time()
    try:
        for entry in parent.iterdir():
            name = entry.name
            if not name.startswith(prefix):
                continue
            suffix = name[len(prefix) :]
            if not _TEMP_DIR_RE.match(suffix):
                continue
            try:
                age = now - entry.stat().st_mtime
            except OSError:
                continue
            if age > max_age:
                _safe_rmtree(entry)
    except OSError:
        pass


def _get_latest_commit_via_git(git_url: str, branch: str) -> str | None:
    """Resolve latest commit SHA for a branch via 'git ls-remote'."""
    try:
        import git

        out = git.cmd.Git().ls_remote("--heads", git_url, branch)
        # Output format: "<sha>\trefs/heads/<branch>\n"
        return out.split()[0] if out else None
    except Exception:
        # Fail silently on any error (offline, auth issue, etc.)
        return None


def _read_cached_commit(cache_folder: Path) -> str | None:
    """Return HEAD commit of cached repo, or None on failure."""
    try:
        import git

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


def _find_parent_cache(
    cache_path: Path,
    repo_name: str,
    folder_path: str,
    branch: str,
    git_url: str,
) -> tuple[Path, Path] | None:
    """Check if folder_path exists inside an already-cached parent folder.

    For example, if folder_path is "unitree_g1/usd" and we have
    "newton-assets_unitree_g1_<hash>" cached, return the paths.

    Args:
        cache_path: The base cache directory
        repo_name: Repository name (e.g., "newton-assets")
        folder_path: The requested folder path (e.g., "unitree_g1/usd")
        branch: Git branch name
        git_url: Full git URL for hash computation

    Returns:
        Tuple of (parent_cache_folder, target_subfolder) if found, None otherwise.
    """
    parts = folder_path.split("/")
    if len(parts) <= 1:
        return None  # No parent to check

    # Generate all potential parent paths: "a/b/c" -> ["a", "a/b"]
    parent_paths = ["/".join(parts[:i]) for i in range(1, len(parts))]

    for parent_path in parent_paths:
        # Generate the cache folder name for this parent
        parent_hash = hashlib.md5(f"{git_url}#{parent_path}#{branch}".encode()).hexdigest()[:8]
        parent_folder_name = parent_path.replace("/", "_").replace("\\", "_")
        parent_cache = cache_path / f"{repo_name}_{parent_folder_name}_{parent_hash}"

        # Check if this parent cache exists and contains our target
        target_in_parent = parent_cache / folder_path
        if target_in_parent.exists() and (parent_cache / ".git").exists():
            return (parent_cache, target_in_parent)

    return None


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
        import git
        from git.exc import GitCommandError
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

    # TTL to avoid repeated network checks
    ttl_seconds = 3600

    # Check if the requested folder exists in an already-cached parent
    # This avoids redundant downloads when a parent folder already contains the subfolder
    if not force_refresh:
        parent_result = _find_parent_cache(cache_path, repo_name, folder_path, branch, git_url)
        if parent_result is not None:
            parent_cache, target_in_parent = parent_result
            stamp_file = parent_cache / ".newton_last_check"

            if _stamp_fresh(stamp_file, ttl_seconds):
                return target_in_parent

            # Verify parent cache is up-to-date
            current_commit = _read_cached_commit(parent_cache)
            latest_commit = _get_latest_commit_via_git(git_url, branch)
            if latest_commit is None or (current_commit and latest_commit == current_commit):
                _touch(stamp_file)
                return target_in_parent
            # If parent is stale, fall through to download fresh subfolder

    # Clean up orphaned temp directories from crashed processes
    _cleanup_stale_temp_dirs(cache_folder)

    # Check cache validity using Git
    stamp_file = cache_folder / ".newton_last_check"

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

        # Different commit detected: will re-download below
        print(
            f"New version of {folder_path} found (cached: {str(current_commit)[:7] if current_commit else 'unknown'}, "
            f"latest: {latest_commit[:7]}). Refreshing..."
        )

    # 3. Download into a process/thread-unique temp directory, then rename
    temp_dir = _temp_cache_path(cache_folder)
    try:
        # Clean up any stale temp dir from a previous crash
        if temp_dir.exists():
            _safe_rmtree(temp_dir)

        print(f"Cloning {git_url} (branch: {branch})...")
        repo = git.Repo.clone_from(
            git_url,
            temp_dir,
            branch=branch,
            depth=1,
            no_checkout=True,
            multi_options=["--filter=blob:none", "--sparse"],
        )

        try:
            repo.git.sparse_checkout("set", folder_path)
            repo.git.checkout(branch)
        finally:
            repo.close()

        temp_target = temp_dir / folder_path
        if not temp_target.exists():
            raise RuntimeError(f"Folder '{folder_path}' not found in repository {git_url}")

        _touch(temp_dir / ".newton_last_check")

        # Move stale cache out of the way (if any), then rename temp into place
        stale_dir = Path(f"{cache_folder}_stale_p{os.getpid()}_t{threading.get_ident()}")
        if cache_folder.exists():
            try:
                os.rename(cache_folder, stale_dir)
            except FileNotFoundError:
                # Another thread already moved/removed it
                pass
        _safe_rename(temp_dir, cache_folder)
        # Clean up stale dir
        if stale_dir.exists():
            _safe_rmtree(stale_dir)

        if cache_folder.exists():
            print(f"Successfully downloaded folder to: {cache_folder / folder_path}")
            return cache_folder / folder_path

        # Should not happen, but handle gracefully
        raise RuntimeError(f"Failed to place cache folder at {cache_folder}")

    except GitCommandError as e:
        raise RuntimeError(f"Git operation failed: {e}") from e
    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(f"Failed to download git folder: {e}") from e
    finally:
        # Always clean up temp dir
        if temp_dir.exists():
            _safe_rmtree(temp_dir)


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
