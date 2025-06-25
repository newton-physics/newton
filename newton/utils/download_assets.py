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
import tempfile
from pathlib import Path


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


def download_git_folder(
    git_url: str, folder_path: str, cache_dir: str | None = None, branch: str = "main", force_refresh: bool = False
) -> Path:
    """
    Downloads a specific folder from a git repository into a local cache.

    Args:
        git_url: The git repository URL (HTTPS or SSH)
        folder_path: The path to the folder within the repository (e.g., "assets/models")
        cache_dir: Directory to cache downloads. If None, uses system temp directory
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
        cache_dir = os.path.join(tempfile.gettempdir(), "newton_git_cache")

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    # Create a unique folder name based on git URL, folder path, and branch
    url_hash = hashlib.md5(f"{git_url}#{folder_path}#{branch}".encode()).hexdigest()[:8]
    repo_name = Path(git_url.rstrip("/")).stem.replace(".git", "")
    folder_name = folder_path.replace("/", "_").replace("\\", "_")
    cache_folder = cache_path / f"{repo_name}_{folder_name}_{url_hash}"

    # Check if already cached and not forcing refresh
    if cache_folder.exists() and not force_refresh:
        target_folder = cache_folder / folder_path
        if target_folder.exists():
            return target_folder

    # Clean up existing cache folder if it exists
    if cache_folder.exists():
        _safe_rmtree(cache_folder)

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
        target_folder = cache_folder / folder_path
        if not target_folder.exists():
            raise RuntimeError(f"Folder '{folder_path}' not found in repository {git_url}")

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
        cache_dir: Cache directory to clear. If None, uses default temp directory
    """
    if cache_dir is None:
        cache_dir = os.path.join(tempfile.gettempdir(), "newton_git_cache")

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
        cache_dir: Directory to cache downloads. If None, uses system temp directory
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
