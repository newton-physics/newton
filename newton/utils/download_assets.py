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
import urllib.request
import zipfile
from pathlib import Path
from urllib.parse import urlparse


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


def _git_url_to_archive_url(git_url: str, branch: str = "main") -> str:
    """
    Convert a git URL to an archive download URL.

    Args:
        git_url: Git repository URL (HTTPS or SSH)
        branch: Branch/tag/commit to download

    Returns:
        Archive download URL

    Raises:
        ValueError: If URL format is not supported
    """
    # Remove .git suffix if present
    if git_url.endswith(".git"):
        git_url = git_url[:-4]

    # Handle SSH URLs (git@github.com:user/repo)
    if git_url.startswith("git@"):
        # Convert SSH to HTTPS
        parts = git_url.replace("git@", "").replace(":", "/")
        git_url = f"https://{parts}"

    # Parse URL to identify hosting service
    parsed = urlparse(git_url)

    if "github.com" in parsed.netloc:
        return f"{git_url}/archive/refs/heads/{branch}.zip"
    elif "gitlab.com" in parsed.netloc:
        return f"{git_url}/-/archive/{branch}/{branch}.zip"
    elif "bitbucket.org" in parsed.netloc:
        return f"{git_url}/get/{branch}.zip"
    else:
        # Try GitHub format as fallback
        return f"{git_url}/archive/refs/heads/{branch}.zip"


def download_git_folder(
    git_url: str,
    folder_path: str,
    cache_dir: str | None = None,
    branch: str = "main",
    force_refresh: bool = False,
) -> Path:
    """
    Downloads a specific folder from a git repository into a local cache.

    Args:
        git_url: The git repository URL (HTTPS or SSH)
        folder_path: The path to the folder within the repository
                    (e.g., "assets/models")
        cache_dir: Directory to cache downloads. If None, uses system temp
                  directory
        branch: Git branch/tag/commit to checkout (default: "main")
        force_refresh: If True, re-downloads even if cached version exists

    Returns:
        Path to the downloaded folder in the local cache

    Raises:
        RuntimeError: If download or extraction fails

    Example:
        >>> folder_path = download_git_folder("https://github.com/user/repo.git", "assets/models", cache_dir="./cache")
        >>> print(f"Downloaded to: {folder_path}")
    """
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
        # Convert git URL to archive download URL
        archive_url = _git_url_to_archive_url(git_url, branch)

        print(f"Downloading {git_url} (branch: {branch})...")

        # Download the archive
        temp_dir = cache_folder / "temp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        archive_path = temp_dir / "archive.zip"

        with urllib.request.urlopen(archive_url) as response:
            with open(archive_path, "wb") as f:
                shutil.copyfileobj(response, f)

        # Extract the archive
        with zipfile.ZipFile(archive_path, "r") as zip_ref:
            # Get the root folder name from the archive
            names = zip_ref.namelist()
            if not names:
                raise RuntimeError("Downloaded archive is empty")

            root_folder = names[0].split("/")[0]

            # Extract only files from the target folder
            target_prefix = f"{root_folder}/{folder_path}/"
            extracted_files = []

            for name in names:
                if name.startswith(target_prefix) and not name.endswith("/"):
                    # Extract to cache folder, removing the root folder prefix
                    relative_path = name[len(root_folder) + 1 :]
                    extract_path = cache_folder / relative_path
                    extract_path.parent.mkdir(parents=True, exist_ok=True)

                    with zip_ref.open(name) as source:
                        with open(extract_path, "wb") as target:
                            shutil.copyfileobj(source, target)

                    extracted_files.append(extract_path)

            if not extracted_files:
                raise RuntimeError(f"Folder '{folder_path}' not found in repository {git_url}")

        # Clean up temporary files
        _safe_rmtree(temp_dir)

        # Verify the folder exists
        target_folder = cache_folder / folder_path
        if not target_folder.exists():
            raise RuntimeError(f"Folder '{folder_path}' not found in repository {git_url}")

        print(f"Successfully downloaded folder to: {target_folder}")
        return target_folder

    except Exception as e:
        # Clean up on failure
        if cache_folder.exists():
            _safe_rmtree(cache_folder)
        raise RuntimeError(f"Failed to download git folder: {e}") from e


def clear_git_cache(cache_dir: str | None = None) -> None:
    """
    Clears the git download cache directory.

    Args:
        cache_dir: Cache directory to clear. If None, uses default temp
                  directory
    """
    if cache_dir is None:
        cache_dir = os.path.join(tempfile.gettempdir(), "newton_git_cache")

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
) -> Path:
    """
    Downloads a specific folder from the newton-assets GitHub repository into
    a local cache.

    Args:
        asset_folder: The folder within the repository to download
                     (e.g., "assets/models")
        cache_dir: Directory to cache downloads. If None, uses system temp
                  directory
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
