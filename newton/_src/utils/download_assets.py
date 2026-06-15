# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

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

from warp._src.thirdparty.appdirs import user_cache_dir

# External asset repositories and their pinned revisions.
# Pinning to commit SHAs ensures reproducible downloads for any given Newton
# commit.  Update these SHAs when assets change upstream and the new versions
# have been validated against Newton's test suite.
NEWTON_ASSETS_URL = "https://github.com/newton-physics/newton-assets.git"
NEWTON_ASSETS_REF = "8e8df07d2e4829442d3d3d3aeecee1857f9951d7"

MENAGERIE_URL = "https://github.com/google-deepmind/mujoco_menagerie.git"
MENAGERIE_REF = "feadf76d42f8a2162426f7d226a3b539556b3bf5"

_SHA_RE = re.compile(r"[0-9a-f]{40}")


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


def _temp_cache_path(final_dir: Path) -> Path:
    """Return a per-process, per-thread temp path next to *final_dir*."""
    return Path(f"{final_dir}_p{os.getpid()}_t{threading.get_ident()}")


_TEMP_DIR_RE = re.compile(r"_p\d+_t\d+$")


def _cleanup_stale_temp_dirs(cache_path: Path, base_prefix: str, max_age: float = 3600.0) -> None:
    """Remove orphaned temp directories left by crashed processes.

    Scans *cache_path* for directories matching ``{base_prefix}_*`` whose names
    contain a temp-dir suffix (``_p{pid}_t{tid}``) and whose mtime is older
    than *max_age* seconds.  Safe to call concurrently.
    """
    now = time.time()
    try:
        for entry in cache_path.iterdir():
            name = entry.name
            if not name.startswith(f"{base_prefix}_") or not entry.is_dir():
                continue
            suffix = name[len(base_prefix) + 1 :]
            if not _TEMP_DIR_RE.search(suffix):
                continue
            try:
                age = now - entry.stat().st_mtime
            except OSError:
                continue
            if age > max_age:
                try:
                    _safe_rmtree(entry)
                except OSError:
                    pass
    except OSError:
        pass


def _find_cached_version(cache_path: Path, base_prefix: str) -> Path | None:
    """Find an existing content-hash cache directory for the given prefix.

    Scans ``{cache_path}`` for directories matching ``{base_prefix}_*/``,
    filters out temp directories (matching ``_p\\d+_t\\d+`` suffix), and
    returns the match with the newest mtime.  Returns ``None`` if no match
    is found.
    """
    candidates = []
    try:
        for entry in cache_path.iterdir():
            name = entry.name
            if not name.startswith(f"{base_prefix}_") or not entry.is_dir():
                continue
            suffix = name[len(base_prefix) + 1 :]
            if _TEMP_DIR_RE.search(suffix):
                continue
            try:
                mtime = entry.stat().st_mtime
            except OSError:
                continue
            candidates.append((mtime, entry))
    except OSError:
        return None
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def _get_latest_commit_via_git(git_url: str, ref: str) -> str | None:
    """Resolve latest commit SHA for a branch or tag via 'git ls-remote'.

    If *ref* is already a 40-character commit SHA it is returned as-is.
    For annotated tags the dereferenced commit SHA is preferred.
    """
    if _SHA_RE.fullmatch(ref):
        return ref
    try:
        import git

        # Request the ref and its dereferenced form (for annotated tags).
        out = git.cmd.Git().ls_remote(git_url, ref, f"{ref}^{{}}")
        if not out:
            return None
        # Parse lines: "<sha>\t<ref>\n"
        # Prefer dereferenced tag (^{}) > branch > lightweight tag
        best = None
        for line in out.strip().splitlines():
            sha, refname = line.split("\t", 1)
            if refname == f"refs/tags/{ref}^{{}}":
                return sha  # annotated tag → underlying commit SHA
            if refname in (f"refs/heads/{ref}", f"refs/tags/{ref}"):
                best = sha
        return best
    except Exception:
        # Fail silently on any error (offline, auth issue, etc.)
        return None


def _find_parent_cache(
    cache_path: Path,
    repo_name: str,
    folder_path: str,
    ref: str,
    git_url: str,
) -> tuple[Path, Path] | None:
    """Check if folder_path exists inside an already-cached parent folder.

    For example, if folder_path is "unitree_g1/usd" and we have
    "newton-assets_unitree_g1_<hash>" cached, return the paths.

    Args:
        cache_path: The base cache directory
        repo_name: Repository name (e.g., "newton-assets")
        folder_path: The requested folder path (e.g., "unitree_g1/usd")
        ref: Git branch, tag, or commit SHA.
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
        parent_hash = hashlib.md5(f"{git_url}#{parent_path}#{ref}".encode()).hexdigest()[:8]
        parent_folder_name = parent_path.replace("/", "_").replace("\\", "_")
        base_prefix = f"{repo_name}_{parent_folder_name}_{parent_hash}"

        cached = _find_cached_version(cache_path, base_prefix)
        if cached is None:
            continue

        # Check if this parent cache contains our target
        target_in_parent = cached / folder_path
        if target_in_parent.exists() and (cached / ".git").exists():
            return (cached, target_in_parent)

    return None


def _cleanup_old_versions(cache_path: Path, base_prefix: str, current_dir: Path) -> None:
    """Best-effort removal of old content-hash directories after a new download.

    Scans for directories matching *base_prefix* (excluding temp dirs and
    *current_dir*) and removes them.  Failures are silently ignored.
    """
    try:
        for entry in cache_path.iterdir():
            if entry == current_dir or not entry.is_dir():
                continue
            name = entry.name
            if not name.startswith(f"{base_prefix}_"):
                continue
            suffix = name[len(base_prefix) + 1 :]
            if _TEMP_DIR_RE.search(suffix):
                continue
            try:
                _safe_rmtree(entry)
            except OSError:
                pass
    except OSError:
        pass


def download_git_folder(
    git_url: str, folder_path: str, cache_dir: str | None = None, ref: str = "main", force_refresh: bool = False
) -> Path:
    """Downloads a specific folder from a git repository into a local cache.

    Uses content-addressed directories: each cached version includes the Git
    commit SHA in its directory name.  When upstream publishes new assets the
    SHA changes, producing a new directory — no in-place eviction is needed.

    Args:
        git_url: The git repository URL (HTTPS or SSH).
        folder_path: Path to the folder within the repository.
        cache_dir: Directory to cache downloads.  If ``None``, determined by
            ``NEWTON_CACHE_PATH`` env-var or the system user cache directory.
        ref: Git branch, tag, or commit SHA to checkout (default: ``"main"``).
        force_refresh: If ``True``, bypass TTL and verify the cached version
            against the remote.  Re-downloads only if the remote SHA differs.

    Returns:
        Path to the downloaded folder in the local cache.
    """
    try:
        import git as gitpython
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

    # Compute identity hash (stable across content changes)
    identity_hash = hashlib.md5(f"{git_url}#{folder_path}#{ref}".encode()).hexdigest()[:8]
    repo_name = Path(git_url.rstrip("/")).stem.replace(".git", "")
    folder_name = folder_path.replace("/", "_").replace("\\", "_")
    base_prefix = f"{repo_name}_{folder_name}_{identity_hash}"

    ttl_seconds = 3600
    latest_commit = None  # reused across parent-cache check and primary resolution to avoid redundant ls-remote

    # --- Parent folder optimization ---
    if not force_refresh:
        parent_result = _find_parent_cache(cache_path, repo_name, folder_path, ref, git_url)
        if parent_result is not None:
            parent_dir, target_in_parent = parent_result
            try:
                age = time.time() - parent_dir.stat().st_mtime
            except OSError:
                age = ttl_seconds + 1
            if age < ttl_seconds:
                return target_in_parent

            # TTL expired — check remote
            parent_sha_suffix = parent_dir.name.rsplit("_", 1)[-1]
            latest_commit = _get_latest_commit_via_git(git_url, ref)
            if latest_commit is None:
                # Offline — touch mtime and return cached
                try:
                    os.utime(parent_dir, None)
                except OSError:
                    pass
                return target_in_parent
            if latest_commit[:8] == parent_sha_suffix:
                try:
                    os.utime(parent_dir, None)
                except OSError:
                    pass
                return target_in_parent
            # Parent is stale — fall through to download

    # --- Resolution flow ---
    cached = _find_cached_version(cache_path, base_prefix)
    if cached is not None and not force_refresh:
        try:
            age = time.time() - cached.stat().st_mtime
        except OSError:
            age = ttl_seconds + 1
        if age < ttl_seconds:
            return cached / folder_path

    # Check remote for current commit (reuse result from parent check if available)
    if latest_commit is None:
        latest_commit = _get_latest_commit_via_git(git_url, ref)

    if latest_commit is None:
        if cached is not None:
            try:
                os.utime(cached, None)
            except OSError:
                pass
            return cached / folder_path
        raise RuntimeError(
            f"Cannot determine remote commit SHA for {git_url} (ref: {ref}) and no cached version exists."
        )

    # Check if we already have this exact version
    if cached is not None and not force_refresh:
        cached_sha_suffix = cached.name.rsplit("_", 1)[-1]
        if latest_commit[:8] == cached_sha_suffix:
            try:
                os.utime(cached, None)
            except OSError:
                pass
            return cached / folder_path

    # --- Download into content-addressed directory ---
    final_dir = cache_path / f"{base_prefix}_{latest_commit[:8]}"
    temp_dir = _temp_cache_path(final_dir)

    # Clean up orphaned temp directories
    _cleanup_stale_temp_dirs(cache_path, base_prefix)

    try:
        if temp_dir.exists():
            _safe_rmtree(temp_dir)

        if cached is not None:
            print(
                f"New version of {folder_path} found "
                f"(cached: {cached.name.rsplit('_', 1)[-1]}, "
                f"latest: {latest_commit[:8]}). Refreshing..."
            )
        print(f"Cloning {git_url} (ref: {ref})...")

        is_sha = bool(_SHA_RE.fullmatch(ref))
        if is_sha:
            # Single fetch — skip the clone, which would download the
            # default-branch tip only to throw it away.
            repo = gitpython.Repo.init(temp_dir)
            try:
                repo.create_remote("origin", git_url)
                repo.git.sparse_checkout("init")
                repo.git.sparse_checkout("set", folder_path)
                repo.git.fetch("origin", ref, "--depth=1", "--filter=blob:none")
                repo.git.checkout("FETCH_HEAD")
            finally:
                repo.close()
        else:
            repo = gitpython.Repo.clone_from(
                git_url,
                temp_dir,
                branch=ref,
                depth=1,
                no_checkout=True,
                multi_options=["--filter=blob:none", "--sparse"],
            )
            try:
                repo.git.sparse_checkout("set", folder_path)
                repo.git.checkout(ref)
            finally:
                repo.close()

        temp_target = temp_dir / folder_path
        if not temp_target.exists():
            raise RuntimeError(f"Folder '{folder_path}' not found in repository {git_url}")

        # Place the finished download into its final location
        _safe_rename(temp_dir, final_dir)

        if temp_dir.exists():
            # Another process already placed this exact version — use theirs
            _safe_rmtree(temp_dir)

        # Set mtime to now for TTL tracking
        os.utime(final_dir, None)

        print(f"Successfully downloaded folder to: {final_dir / folder_path}")

        # Best-effort cleanup of old versions
        _cleanup_old_versions(cache_path, base_prefix, final_dir)

        return final_dir / folder_path

    except GitCommandError as e:
        raise RuntimeError(f"Git operation failed: {e}") from e
    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(f"Failed to download git folder: {e}") from e
    finally:
        try:
            if temp_dir.exists():
                _safe_rmtree(temp_dir)
        except OSError:
            pass


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


def download_asset(
    asset_folder: str,
    cache_dir: str | None = None,
    force_refresh: bool = False,
    ref: str | None = None,
) -> Path:
    """Download a specific folder from the newton-assets GitHub repository into a local cache.

    Args:
        asset_folder: The folder within the repository to download (e.g., "assets/models")
        cache_dir: Directory to cache downloads.
            If ``None``, the path is determined in the following order:
            1. ``NEWTON_CACHE_PATH`` environment variable.
            2. System's user cache directory (via ``appdirs.user_cache_dir``).
        force_refresh: If ``True``, bypass TTL and verify the cached version
            against the remote.  Re-downloads only if the remote SHA differs.
        ref: Git branch, tag, or commit SHA to checkout.  Defaults to the
            revision pinned in :data:`NEWTON_ASSETS_REF`.

    Returns:
        Path to the downloaded folder in the local cache.
    """
    return download_git_folder(
        NEWTON_ASSETS_URL,
        asset_folder,
        cache_dir=cache_dir,
        ref=ref or NEWTON_ASSETS_REF,
        force_refresh=force_refresh,
    )
