# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Download USD assets and localize their references."""

from __future__ import annotations

import collections
import datetime
import hashlib
import os
import posixpath
import re
from pathlib import PureWindowsPath
from urllib.parse import urljoin, urlparse


def _validate_https_usd_url(url: str) -> None:
    """Reject non-HTTPS URLs before USD asset downloads."""
    if urlparse(url).scheme != "https":
        raise ValueError(f"USD URL downloads require HTTPS: {url}")


def _cache_path_for_absolute_usd_reference(url: str) -> str:
    """Return a safe cache-relative path for an absolute USD reference URL."""
    parsed = urlparse(url)
    basename = posixpath.basename(parsed.path) or "reference.usd"
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    return posixpath.join("_external_usd", digest, basename)


def _reject_windows_rooted_usd_path(path: str) -> None:
    """Reject paths with Windows drive, root, or UNC semantics."""
    windows_path = PureWindowsPath(path)
    if windows_path.drive or windows_path.root:
        raise ValueError(f"USD reference path must be relative: {path}")


def _normalize_usd_cache_relative_path(path: str) -> str:
    """Normalize a relative cache path while rejecting POSIX and Windows escapes."""
    _reject_windows_rooted_usd_path(path)

    normalized = posixpath.normpath(path.replace("\\", "/"))
    if normalized in {"", ".", ".."} or posixpath.isabs(normalized) or normalized.startswith("../"):
        raise ValueError(f"USD reference path escapes the target folder: {path}")
    return normalized


def _resolve_usd_cache_path(target_folder_name: str, relative_path: str) -> str:
    """Return the canonical cache path when it remains beneath the target folder."""
    normalized = _normalize_usd_cache_relative_path(relative_path)
    target_root = os.path.realpath(target_folder_name)
    candidate = os.path.realpath(os.path.join(target_root, *normalized.split("/")))
    try:
        common_path = os.path.commonpath((target_root, candidate))
    except ValueError as exc:
        raise ValueError(f"USD reference path escapes the target folder: {relative_path}") from exc
    if os.path.normcase(common_path) != os.path.normcase(target_root):
        raise ValueError(f"USD reference path escapes the target folder: {relative_path}")
    return candidate


def resolve_usd_from_url(url: str, target_folder_name: str | None = None, export_usda: bool = False) -> str:
    """Download a USD file from a URL and resolves all references to other USD files to be downloaded to the given target folder.

    Args:
        url: URL to the USD file.
        target_folder_name: Target folder name. If ``None``, a time-stamped
          folder will be created in the current directory.
        export_usda: If ``True``, converts each downloaded USD file to USDA and
          saves the additional USDA file in the target folder with the same
          base name as the original USD file.

    Returns:
        File path to the downloaded USD file.

    Raises:
        ValueError: If a URL is not HTTPS or a referenced asset cannot be
            localized within the download cache.
    """

    import requests

    try:
        from pxr import Usd
    except ImportError as e:
        raise ImportError("Failed to import pxr. Please install USD (e.g. via `pip install usd-core`).") from e

    def _download_https_url(source_url: str):
        """Download a URL while validating every redirect target is HTTPS."""
        current_url = source_url
        request_timeout_s = 30
        for _ in range(10):
            _validate_https_usd_url(current_url)
            response = requests.get(current_url, allow_redirects=False, timeout=request_timeout_s)
            if int(response.status_code) in {301, 302, 303, 307, 308}:
                redirect_url = response.headers.get("Location")
                if not redirect_url:
                    return response, current_url
                current_url = urljoin(current_url, redirect_url)
                continue
            final_url = getattr(response, "url", current_url)
            if not isinstance(final_url, str):
                final_url = current_url
            _validate_https_usd_url(final_url)
            return response, final_url
        raise RuntimeError(f"Too many redirects while downloading USD file: {source_url}")

    response, resolved_url = _download_https_url(url)
    if response.status_code != 200:
        raise RuntimeError(f"Failed to download USD file. Status code: {response.status_code}")
    file = response.content
    dot = os.path.extsep
    base = posixpath.basename(urlparse(resolved_url).path)
    url_folder = posixpath.dirname(resolved_url)
    base_name = dot.join(base.split(dot)[:-1])
    if target_folder_name is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        target_folder_name = os.path.join(".usd_cache", f"{base_name}_{timestamp}")
    os.makedirs(target_folder_name, exist_ok=True)
    target_folder_name = os.path.realpath(target_folder_name)
    target_filename = _resolve_usd_cache_path(target_folder_name, base)
    with open(target_filename, "wb") as f:
        f.write(file)

    stage = Usd.Stage.Open(target_filename, Usd.Stage.LoadNone)
    root_layer = stage.GetRootLayer()
    stage_str = root_layer.ExportToString()
    print(f"Downloaded USD file to {target_filename}.")

    # Recursively resolve referenced USD files like `references = @./franka_collisions.usd@`
    # Each entry in the queue is (resolved_url, cache_relative_path).
    downloaded_urls: set[str] = {url, resolved_url}
    pending: collections.deque[tuple[str, str]] = collections.deque()

    def _write_layer_string(filename: str, layer, layer_str: str) -> None:
        """Persist rewritten USDA text to both the layer and cache file."""
        import_from_string = getattr(layer, "ImportFromString", None)
        if callable(import_from_string):
            import_from_string(layer_str)
            save = getattr(layer, "Save", None)
            if callable(save):
                save()
        with open(filename, "w") as f:
            f.write(layer_str)

    def _extract_references(layer_str, parent_url_folder, parent_local_folder):
        """Extract references, queue downloads, and return rewritten layer text."""
        reference_assignment_pattern = re.compile(
            r"(?P<prefix>references\s*=\s*)"
            r"(?P<value>@[^@]*@(?:<[^>]*>)?(?:\s*\([^)]*\))?|\[[^]]*\])",
            re.DOTALL,
        )
        reference_item_pattern = re.compile(r"@(?P<path>[^@]*)@(?P<suffix>(?:<[^>]*>)?(?:\s*\([^)]*\))?)")

        def _prepare_reference(raw_ref):
            """Return the rewritten path, source URL, and cache-relative path."""
            raw_ref_scheme = urlparse(raw_ref).scheme
            if raw_ref_scheme in {"http", "https"}:
                ref_url = urljoin(parent_url_folder + "/", raw_ref)
                _validate_https_usd_url(ref_url)
                local_path = _cache_path_for_absolute_usd_reference(ref_url)
                rewritten_ref = local_path
            else:
                _reject_windows_rooted_usd_path(raw_ref)
                local_path = _normalize_usd_cache_relative_path(posixpath.join(parent_local_folder, raw_ref))
                ref_url = urljoin(parent_url_folder + "/", raw_ref.replace("\\", "/"))
                rewritten_ref = raw_ref
            return rewritten_ref, ref_url, local_path

        def _rewrite_reference_item(match):
            """Validate one asset reference and rewrite its cache path when needed."""
            raw_ref = match.group("path")
            rewritten_ref, ref_url, local_path = _prepare_reference(raw_ref)
            _resolve_usd_cache_path(target_folder_name, local_path)
            if ref_url not in downloaded_urls:
                pending.append((ref_url, local_path))
            return f"@{rewritten_ref}@{match.group('suffix')}"

        def _rewrite_reference_assignment(match):
            """Rewrite asset references without changing other reference-list entries."""
            rewritten_value = reference_item_pattern.sub(_rewrite_reference_item, match.group("value"))
            return match.group("prefix") + rewritten_value

        return reference_assignment_pattern.sub(_rewrite_reference_assignment, layer_str)

    rewritten_stage_str = _extract_references(stage_str, url_folder, "")
    if rewritten_stage_str != stage_str:
        _write_layer_string(target_filename, root_layer, rewritten_stage_str)
        stage_str = rewritten_stage_str

    if export_usda:
        usda_filename = _resolve_usd_cache_path(target_folder_name, base_name + ".usda")
        with open(usda_filename, "w") as f:
            f.write(stage_str)
            print(f"Exported USDA file to {usda_filename}.")

    while pending:
        ref_url, local_path = pending.popleft()
        if ref_url in downloaded_urls:
            continue
        downloaded_urls.add(ref_url)
        try:
            response, resolved_ref_url = _download_https_url(ref_url)
            if response.status_code != 200:
                print(f"Failed to download reference {local_path}. Status code: {response.status_code}")
                continue
            downloaded_urls.add(resolved_ref_url)
            file = response.content
            local_dir = posixpath.dirname(local_path)
            try:
                ref_filename = _resolve_usd_cache_path(target_folder_name, local_path)
            except ValueError:
                print(f"Skipping reference that escapes target folder: {local_path}")
                continue
            os.makedirs(os.path.dirname(ref_filename), exist_ok=True)
            if not os.path.exists(ref_filename):
                with open(ref_filename, "wb") as f:
                    f.write(file)
            print(f"Downloaded USD reference {local_path} to {ref_filename}.")

            ref_stage = Usd.Stage.Open(ref_filename, Usd.Stage.LoadNone)
            ref_layer = ref_stage.GetRootLayer()
            ref_stage_str = ref_layer.ExportToString()

            rewritten_ref_stage_str = _extract_references(ref_stage_str, posixpath.dirname(resolved_ref_url), local_dir)
            if rewritten_ref_stage_str != ref_stage_str:
                _write_layer_string(ref_filename, ref_layer, rewritten_ref_stage_str)
                ref_stage_str = rewritten_ref_stage_str

            if export_usda:
                ref_base = os.path.basename(ref_filename)
                ref_base_name = dot.join(ref_base.split(dot)[:-1])
                usda_relative_path = (
                    posixpath.join(local_dir, ref_base_name + ".usda") if local_dir else ref_base_name + ".usda"
                )
                usda_filename = _resolve_usd_cache_path(target_folder_name, usda_relative_path)
                with open(usda_filename, "w") as f:
                    f.write(ref_stage_str)
                    print(f"Exported USDA file to {usda_filename}.")
        except ValueError:
            raise
        except Exception:
            print(f"Failed to download {local_path}.")
    return target_filename
