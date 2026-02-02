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

import io
import os
import warnings
from urllib.parse import unquote, urlparse
from urllib.request import urlopen

import numpy as np

from ..core.types import nparray

_texture_url_cache: dict[str, bytes] = {}


def _is_http_url(path: str) -> bool:
    parsed = urlparse(path)
    return parsed.scheme in ("http", "https")


def _resolve_file_url(path: str) -> str:
    parsed = urlparse(path)
    if parsed.scheme != "file":
        return path
    return unquote(parsed.path)


def _download_texture_from_file_bytes(url: str) -> bytes | None:
    if url in _texture_url_cache:
        return _texture_url_cache[url]
    try:
        with urlopen(url, timeout=10) as response:
            data = response.read()
        _texture_url_cache[url] = data
        return data
    except Exception as exc:
        warnings.warn(f"Failed to download texture image: {url} ({exc})", stacklevel=2)
        return None


def load_texture_from_file(texture_path: str | None) -> nparray | None:
    """Load a texture image from disk or URL into a numpy array.

    Args:
        texture_path: Path or URL to the texture image.

    Returns:
        Texture image as uint8 numpy array (H, W, C), or None if load fails.
    """
    if texture_path is None:
        return None
    try:
        from PIL import Image  # noqa: PLC0415

        if _is_http_url(texture_path):
            data = _download_texture_from_file_bytes(texture_path)
            if data is None:
                return None
            with Image.open(io.BytesIO(data)) as source_img:
                img = source_img.convert("RGBA")
                return np.array(img)

        texture_path = _resolve_file_url(texture_path)
        with Image.open(texture_path) as source_img:
            img = source_img.convert("RGBA")
            return np.array(img)
    except Exception as exc:
        warnings.warn(f"Failed to load texture image: {texture_path} ({exc})", stacklevel=2)
        return None


def normalize_texture_input(texture: str | os.PathLike[str] | nparray | None) -> nparray | None:
    """Normalize a texture input into a contiguous image array.

    Args:
        texture: Path/URL to a texture image or an array (H, W, C).

    Returns:
        np.ndarray | None: Contiguous image array, or None if unavailable.
    """
    if texture is None:
        return None

    if isinstance(texture, os.PathLike):
        texture = os.fspath(texture)

    if isinstance(texture, str):
        loaded = load_texture_from_file(texture)
        if loaded is None:
            return None
        return np.ascontiguousarray(loaded)

    return np.ascontiguousarray(np.asarray(texture))


def compute_texture_hash(texture: nparray | None) -> int:
    """Compute a stable hash for a texture array.

    Args:
        texture: Texture image array (H, W, C) or None.

    Returns:
        Hash of the texture metadata and sampled contents, or 0 for None.
    """
    if texture is None:
        return 0

    texture = np.ascontiguousarray(texture)
    flat_size = texture.size
    if flat_size == 0:
        sample_bytes = b""
    else:
        # Only sample a small portion of the texture to avoid hashing large textures in full.
        flat = texture.ravel()
        max_samples = 1024
        step = max(1, flat.size // max_samples)
        sample = flat[::step]
        sample_bytes = sample.tobytes()

    return hash((texture.shape, texture.dtype.str, sample_bytes))
