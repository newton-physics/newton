# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import atexit
import errno
import hashlib
import logging
import os
import posixpath
import re
import shutil
import stat
import tempfile
import threading
import time
from pathlib import Path

from ... import config

logger = logging.getLogger(__name__)

# External asset repositories and their pinned revisions.
# Pinning to commit SHAs ensures reproducible downloads for any given Newton
# commit.  Update these SHAs when assets change upstream and the new versions
# have been validated against Newton's test suite.
NEWTON_ASSETS_URL = "https://github.com/newton-physics/newton-assets.git"
NEWTON_ASSETS_REF = "ac1f172b6f53915f9b81aa9ccd765ae792cc264c"

MENAGERIE_URL = "https://github.com/google-deepmind/mujoco_menagerie.git"
MENAGERIE_REF = "da76818e269b82289eba39808e2fb91d679d6994"

_SHA_RE = re.compile(r"[0-9a-f]{40}")
_HTTP_SERVER_ERROR_RE = re.compile(r"\bhttp 5\d\d\b", re.IGNORECASE)
_GIT_DOWNLOAD_ATTEMPTS = 3
_GIT_DOWNLOAD_RETRY_DELAY = 1.0
_TRANSIENT_GIT_ERROR_PATTERNS = (
    "could not resolve host",
    "failed to connect to",
    "timed out",
    "rpc failed",
    "early eof",
    "the remote end hung up",
    "error reading section header",
    "expected 'packfile'",
)
_PERMANENT_GIT_ERROR_PATTERNS = (
    "authentication failed",
    "permission denied",
    "repository not found",
    "couldn't find remote ref",
    "http 401",
    "http 403",
    "http 404",
)


def _is_transient_git_error(error: Exception) -> bool:
    """Return whether a Git error indicates a transient network failure."""
    stderr = getattr(error, "stderr", "") or ""
    if isinstance(stderr, bytes):
        stderr = stderr.decode(errors="replace")
    details = stderr.lower()
    if any(pattern in details for pattern in _PERMANENT_GIT_ERROR_PATTERNS):
        return False
    return bool(_HTTP_SERVER_ERROR_RE.search(details)) or any(
        pattern in details for pattern in _TRANSIENT_GIT_ERROR_PATTERNS
    )


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


def _cleanup_temporary_cache(path: str, owner_pid: int) -> None:
    """Remove temporary assets only when the process that downloaded them exits."""
    # Forked children inherit atexit callbacks but must not delete their parent's assets.
    if os.getpid() == owner_pid:
        _safe_rmtree(path)


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


def _cache_prefix(git_url: str, folder_path: str) -> str:
    """Identify a repository and normalized folder independently of the requested ref."""
    # Leave room for Git's internal paths under temporary checkouts on Windows.
    return hashlib.sha256(f"{git_url}\0{folder_path}".encode()).hexdigest()[:16]


def _ref_key(git_url: str, folder_path: str, ref: str) -> str:
    """Identify a request in metadata, independently of the content directory name."""
    return hashlib.sha256(f"{git_url}\0{folder_path}\0{ref}".encode()).hexdigest()


def _ref_path(cache_dir: Path, ref_key: str) -> Path:
    """Return the current-resolution record in the cache root."""
    return cache_dir / ".newton-refs" / ref_key


def _read_ref_record(path: Path) -> tuple[str, str] | None:
    try:
        directory, commit = path.read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeError, ValueError):
        return None
    return (directory, commit) if _SHA_RE.fullmatch(commit) else None


def _read_cached_ref(cache_dir: Path, ref_key: str) -> str | None:
    record = _read_ref_record(_ref_path(cache_dir.parent, ref_key))
    return record[1] if record is not None and record[0] == cache_dir.name else None


def _find_cached_version(cache_path: Path, base_prefix: str, ref_key: str) -> Path | None:
    """Read the authoritative content directory for this exact request."""
    record = _read_ref_record(_ref_path(cache_path, ref_key))
    if record is None:
        return None
    directory, _ = record
    # Accept only an immediate content directory, never a temporary or external path.
    if not directory.startswith(f"{base_prefix}_") or not _SHA_RE.fullmatch(directory[len(base_prefix) + 1 :]):
        return None
    cached = cache_path / directory
    try:
        return cached if cached.is_dir() else None
    except OSError:
        return None


def _write_cached_ref(cache_dir: Path, ref_key: str, commit: str) -> None:
    """Publish the current directory and commit atomically; use mtime only for check TTL."""
    path = _ref_path(cache_dir.parent, ref_key)
    temporary = _temp_cache_path(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(f"{cache_dir.name}\n{commit}\n", encoding="ascii")
        for attempt in range(5):
            try:
                os.replace(temporary, path)
                break
            except PermissionError:
                # Windows readers and concurrent writers can briefly prevent replacement.
                if attempt == 4:
                    raise
                time.sleep(0.1)
    except OSError as e:
        # Metadata controls future cache lookup; it must not hide usable asset contents.
        logger.debug("Could not update cached ref %s: %s", path, e)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError as e:
            logger.debug("Could not remove temporary cached ref %s: %s", temporary, e)


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
    folder_path: str,
    ref: str,
    git_url: str,
) -> tuple[Path, Path, str] | None:
    """Check if folder_path exists inside an already-cached parent folder.

    For example, if folder_path is "unitree_g1/usd" and "unitree_g1" is
    already cached for this repository and ref, return the paths.

    Args:
        cache_path: The base cache directory
        folder_path: The requested folder path (e.g., "unitree_g1/usd")
        ref: Git branch, tag, or commit SHA.
        git_url: Full git URL for hash computation

    Returns:
        Tuple of (parent_cache_folder, target_subfolder, parent_ref_key) if found, None otherwise.
    """
    parts = folder_path.split("/")
    if len(parts) <= 1:
        return None  # No parent to check

    # Generate all potential parent paths: "a/b/c" -> ["a", "a/b"]
    parent_paths = ["/".join(parts[:i]) for i in range(1, len(parts))]

    for parent_path in parent_paths:
        parent_key = _ref_key(git_url, parent_path, ref)
        base_prefix = _cache_prefix(git_url, parent_path)

        cached = _find_cached_version(cache_path, base_prefix, parent_key)
        if cached is None:
            continue

        # Check if this parent cache contains our target
        target_in_parent = cached / folder_path
        if target_in_parent.is_dir():
            return (cached, target_in_parent, parent_key)

    return None


def download_git_folder(
    git_url: str,
    folder_path: str,
    cache_dir: str | os.PathLike[str] | None = None,
    ref: str = "main",
    force_refresh: bool = False,
) -> Path:
    """Downloads a specific folder from a git repository into a local cache.

    Cache directory names combine a hash of the repository URL and normalized
    folder path with the folder's full Git tree hash, which covers file names,
    contents, and modes. Unchanged contents of the same repository folder
    share a directory across refs and commits, including commits that only
    change other assets. One atomic metadata record per request selects the
    current directory and commit; its modification time controls check TTL only.
    Different contents remain available until :func:`clear_git_cache` is called.

    Args:
        git_url: The git repository URL (HTTPS or SSH).
        folder_path: Path to the folder within the repository. Redundant separators
            and dot components are normalized before cache lookup.
        cache_dir: Directory to cache downloads. If ``None``, use
            :attr:`newton.config.cache_dir_assets`. If that setting is ``None``,
            download into a fresh temporary directory kept until process exit.
            An explicit directory overrides the global setting.
        ref: Git branch, tag, or commit SHA to checkout (default: ``"main"``).
        force_refresh: If ``True``, bypass TTL and verify the cached version
            against the remote.  Re-downloads only if the remote SHA differs.

    Returns:
        Path to the downloaded folder, available until process exit when caching is disabled.
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
        cache_dir = config.cache_dir_assets
    if cache_dir is None:
        # Callers receive a Path and may load referenced files later. Keep each
        # uncached download alive until exit, without reusing it on later calls.
        temporary_cache = tempfile.mkdtemp(prefix="newton-assets-")
        try:
            result = download_git_folder(git_url, folder_path, temporary_cache, ref, force_refresh)
        except BaseException:
            _safe_rmtree(temporary_cache)
            raise
        atexit.register(_cleanup_temporary_cache, temporary_cache, os.getpid())
        return result
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    folder_path = posixpath.normpath(folder_path.replace("\\", "/"))
    ref_key = _ref_key(git_url, folder_path, ref)
    base_prefix = _cache_prefix(git_url, folder_path)

    ttl_seconds = 3600
    cached_key = ref_key
    cached = _find_cached_version(cache_path, base_prefix, ref_key)
    cached_target = cached / folder_path if cached is not None else None
    if cached_target is not None and not cached_target.is_dir():
        cached = None
        cached_target = None
    if cached is None and not force_refresh:
        parent_result = _find_parent_cache(cache_path, folder_path, ref, git_url)
        if parent_result is not None:
            cached, cached_target, cached_key = parent_result

    if cached is not None and not force_refresh:
        try:
            age = time.time() - _ref_path(cache_path, cached_key).stat().st_mtime
        except OSError:
            age = ttl_seconds + 1
        if age < ttl_seconds:
            return cached_target

    latest_commit = _get_latest_commit_via_git(git_url, ref)

    if latest_commit is None:
        if cached is not None:
            # Throttle failed remote checks without changing the cached ref resolution.
            try:
                os.utime(_ref_path(cache_path, cached_key), None)
            except OSError as e:
                logger.debug("Could not update cached ref check time for %s: %s", cached, e)
            return cached_target
        raise RuntimeError(
            f"Cannot determine remote commit SHA for {git_url} (ref: {ref}) and no cached version exists."
        )

    if cached is not None and latest_commit == _read_cached_ref(cached, cached_key):
        _write_cached_ref(cached, cached_key, latest_commit)
        return cached_target

    # The content hash is only known after fetching the requested tree.
    temp_dir = _temp_cache_path(cache_path / f"{base_prefix}_{ref_key}")

    # Clean up orphaned temp directories
    _cleanup_stale_temp_dirs(cache_path, base_prefix)

    try:
        if cached is not None:
            logger.info(
                "New version of %s found (cached: %s, latest: %s). Refreshing...",
                folder_path,
                (_read_cached_ref(cached, cached_key) or "unknown")[:8],
                latest_commit[:8],
            )
        logger.info("Cloning %s (ref: %s)...", git_url, ref)

        is_sha = bool(_SHA_RE.fullmatch(ref))
        for attempt in range(_GIT_DOWNLOAD_ATTEMPTS):
            try:
                if temp_dir.exists():
                    _safe_rmtree(temp_dir)

                if is_sha:
                    # Single fetch — skip the clone, which would download the
                    # default-branch tip only to throw it away.
                    repo = gitpython.Repo.init(temp_dir)
                    try:
                        repo.create_remote("origin", git_url)
                        repo.git.sparse_checkout("init")
                        repo.git.sparse_checkout("set", folder_path)
                        repo.git.fetch("origin", ref, "--depth=1", "--filter=blob:none")
                    except BaseException:
                        repo.close()
                        raise
                    checkout_ref = "FETCH_HEAD"
                else:
                    repo = gitpython.Repo.clone_from(
                        git_url,
                        temp_dir,
                        branch=ref,
                        depth=1,
                        no_checkout=True,
                        multi_options=["--filter=blob:none", "--sparse"],
                    )
                    checkout_ref = ref
                try:
                    commit = repo.commit(checkout_ref)
                    try:
                        tree = commit.tree / folder_path
                    except KeyError as e:
                        raise RuntimeError(f"Folder '{folder_path}' not found in repository {git_url}") from e
                    if tree.type != "tree":
                        raise RuntimeError(f"'{folder_path}' is not a folder in repository {git_url}")
                    final_dir = cache_path / f"{base_prefix}_{tree.hexsha}"
                    # Avoid materializing unchanged files when another ref already cached them.
                    if not (final_dir / folder_path).is_dir():
                        repo.git.sparse_checkout("set", folder_path)
                        repo.git.checkout(checkout_ref)
                    latest_commit = commit.hexsha
                finally:
                    repo.close()
                break
            except GitCommandError as e:
                if attempt == _GIT_DOWNLOAD_ATTEMPTS - 1 or not _is_transient_git_error(e):
                    raise
                delay = _GIT_DOWNLOAD_RETRY_DELAY * 2**attempt
                logger.info("Transient Git error. Retrying in %g seconds...", delay)
                time.sleep(delay)

        if not (final_dir / folder_path).is_dir():
            if not (temp_dir / folder_path).is_dir():
                raise RuntimeError(f"Folder '{folder_path}' not found in repository {git_url}")
            _safe_rename(temp_dir, final_dir)

        if temp_dir.exists():
            # Another process already placed this exact version — use theirs
            _safe_rmtree(temp_dir)

        _write_cached_ref(final_dir, ref_key, latest_commit)

        logger.info("Successfully downloaded folder to: %s", final_dir / folder_path)

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


def clear_git_cache(cache_dir: str | os.PathLike[str] | None = None) -> None:
    """
    Clears the git download cache directory.

    Args:
        cache_dir: Cache directory to clear. If ``None``, use
            :attr:`newton.config.cache_dir_assets`. If that setting is ``None``,
            do nothing; temporary downloads remain available until process exit.
    """
    if cache_dir is None:
        cache_dir = config.cache_dir_assets
    if cache_dir is None:
        return

    cache_path = Path(cache_dir)
    if cache_path.exists():
        _safe_rmtree(cache_path)
        logger.info("Cleared git cache: %s", cache_path)
    else:
        logger.info("Git cache directory does not exist")


def download_asset(
    asset_folder: str,
    cache_dir: str | os.PathLike[str] | None = None,
    force_refresh: bool = False,
    ref: str | None = None,
) -> Path:
    """Download a specific folder from the newton-assets GitHub repository into a local cache.

    The cache directory is named using the repository and folder identity hash
    plus the folder's full Git tree hash, so unchanged contents of the same
    asset folder share a directory across revisions. Older cache
    formats are left untouched and are downloaded again on first use. Use the
    returned path instead of constructing a cache directory name.

    Args:
        asset_folder: The folder within the repository to download (e.g., "assets/models")
        cache_dir: Directory to cache downloads. If ``None``, use
            :attr:`newton.config.cache_dir_assets`, which defaults to
            ``NEWTON_CACHE_PATH`` or the platform's Newton cache directory.
            Set that global setting to ``None`` to download each request into
            a fresh temporary directory kept until process exit. An explicit
            directory overrides the global setting.
        force_refresh: If ``True``, bypass TTL and verify the cached version
            against the remote.  Re-downloads only if the remote SHA differs.
        ref: Git branch, tag, or commit SHA to checkout.  Defaults to the
            revision pinned in :data:`NEWTON_ASSETS_REF`.

    Returns:
        Path to the downloaded folder, available until process exit when caching is disabled.
    """
    return download_git_folder(
        NEWTON_ASSETS_URL,
        asset_folder,
        cache_dir=cache_dir,
        ref=ref or NEWTON_ASSETS_REF,
        force_refresh=force_refresh,
    )
