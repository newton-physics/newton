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
        Texture image as uint8 RGBA numpy array (H, W, 4), or None if load fails.
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


def load_texture(texture: str | os.PathLike[str] | nparray | None) -> nparray | None:
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


def normalize_texture(
    texture_image: nparray | None,
    *,
    flip_vertical: bool = False,
    require_channels: bool = False,
    scale_unit_range: bool = True,
) -> nparray | None:
    """Normalize a texture array for rendering.

    Args:
        texture_image: Texture image array (H, W, C) or None.
        flip_vertical: Whether to flip the image vertically.
        require_channels: Whether to enforce 3/4-channel images and expand grayscale.
        scale_unit_range: Whether to scale unit-range floats to 0-255.

    Returns:
        np.ndarray | None: Normalized uint8 image array or None if unavailable.
    """
    if texture_image is None:
        return None

    image = np.asarray(texture_image)
    if image.dtype != np.uint8:
        image = np.clip(image, 0.0, 255.0)
        if scale_unit_range and image.max() <= 1.0:
            image = image * 255.0
        image = image.astype(np.uint8)

    if require_channels:
        if image.ndim == 2:
            image = np.repeat(image[:, :, None], 3, axis=2)
        if image.ndim < 2 or image.shape[0] == 0 or image.shape[1] == 0:
            raise ValueError("Texture image has invalid dimensions.")
        if image.shape[2] not in (3, 4):
            raise ValueError(f"Unsupported texture channels: {image.shape[2]}")

    if flip_vertical:
        image = np.flipud(image)

    return np.ascontiguousarray(image)


def compute_texture_hash(texture: str | os.PathLike[str] | nparray | None) -> int:
    """Compute a stable hash for a texture (path or array).

    Args:
        texture: Texture path/URL string, PathLike, or image array (H, W, C), or None.

    Returns:
        Hash of the texture path or array contents, or 0 for None.
    """
    if texture is None:
        return 0

    # Handle string paths and PathLike - hash the path string without decoding
    if isinstance(texture, os.PathLike):
        return hash(os.fspath(texture))
    if isinstance(texture, str):
        return hash(texture)

    # Array input - hash based on shape and sampled content
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
