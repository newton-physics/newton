# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import io
import os
import warnings
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING
from urllib.parse import unquote, urlparse
from urllib.request import urlopen

import numpy as np

from ..exceptions import NewtonWarning

if TYPE_CHECKING:
    from concurrent.futures import Future

    from PIL import Image

_texture_url_cache: dict[str, bytes] = {}

# Most decoded pixels [bytes] that the textures decoded ahead may hold while they wait for the
# thread that loads them. A single texture may hold more. A bound on memory, not a setting to
# tune: decoding further ahead than that thread consumes does not load faster.
_MAX_PENDING_TEXTURE_BYTES = 256 * 1024 * 1024


def _linear_to_srgb(linear: np.ndarray) -> np.ndarray:
    return np.where(linear <= 0.0031308, linear * 12.92, 1.055 * np.power(linear, 1.0 / 2.4) - 0.055)


def _make_linear_to_srgb_uint8_lut() -> np.ndarray:
    linear = np.arange(256, dtype=np.float32) / 255.0
    srgb = _linear_to_srgb(linear)
    lut = np.clip(np.round(srgb * 255.0), 0.0, 255.0).astype(np.uint8)
    lut.setflags(write=False)
    return lut


_LINEAR_TO_SRGB_UINT8_LUT = _make_linear_to_srgb_uint8_lut()


def _is_http_url(path: str) -> bool:
    parsed = urlparse(path)
    return parsed.scheme in ("http", "https")


def _resolve_file_url(path: str) -> str:
    parsed = urlparse(path)
    if parsed.scheme != "file":
        return path
    return unquote(parsed.path)


def _is_usd_package_path(path: str) -> bool:
    """Return whether *path* addresses an asset inside a USD package (e.g. ``scene.usdz[tex.png]``)."""
    # Cheap bracket check first so the pxr import is only paid for real packages.
    if "[" not in path:
        return False
    try:
        from pxr import Ar
    except ImportError:
        return False
    return Ar.IsPackageRelativePath(path)


def _read_usd_package_asset_bytes(path: str) -> bytes | None:
    """Read the raw bytes of an asset packaged inside a USD file (e.g. a texture in a ``.usdz``).

    ``.usdz`` archives address their contents with package-relative paths of the
    form ``archive.usdz[inner/path.png]``, which are not valid filesystem paths.
    Resolve them through USD's asset resolver, which also handles nested packages.
    """
    try:
        from pxr import Ar
    except ImportError:
        return None
    try:
        asset = Ar.GetResolver().OpenAsset(Ar.ResolvedPath(path))
        if not asset:
            warnings.warn(
                f"Failed to read packaged texture image: {path} (not found in package)", NewtonWarning, stacklevel=3
            )
            return None
        return bytes(asset.GetBuffer())
    except Exception as exc:
        warnings.warn(f"Failed to read packaged texture image: {path} ({exc})", NewtonWarning, stacklevel=3)
        return None


def _download_texture_from_file_bytes(url: str) -> bytes | None:
    if url in _texture_url_cache:
        return _texture_url_cache[url]
    try:
        with urlopen(url, timeout=10) as response:
            data = response.read()
        _texture_url_cache[url] = data
        return data
    except Exception as exc:
        warnings.warn(f"Failed to download texture image: {url} ({exc})", NewtonWarning, stacklevel=2)
        return None


def load_texture_from_file(texture_path: str | None) -> np.ndarray | None:
    """Load a texture image from disk or URL into a numpy array.

    Args:
        texture_path: Path or URL to the texture image.

    Returns:
        Texture image as uint8 RGBA numpy array (H, W, 4), or None if load fails.
    """
    if texture_path is None:
        return None
    try:
        from PIL import Image

        if _is_http_url(texture_path):
            data = _download_texture_from_file_bytes(texture_path)
            if data is None:
                return None
            with Image.open(io.BytesIO(data)) as source_img:
                img = source_img.convert("RGBA")
                return np.array(img)

        if _is_usd_package_path(texture_path):
            data = _read_usd_package_asset_bytes(texture_path)
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
        warnings.warn(f"Failed to load texture image: {texture_path} ({exc})", NewtonWarning, stacklevel=2)
        return None


def load_texture(texture: str | os.PathLike[str] | np.ndarray | None) -> np.ndarray | None:
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


def _count_usable_cpus() -> int:
    """Count the CPUs this process may run on, which can be fewer than the machine has."""
    if hasattr(os, "process_cpu_count"):
        return os.process_cpu_count() or 1
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def _convert_to_rgba(image: Image.Image) -> Image.Image:
    """Decode an opened image file and convert it to RGBA, then close it."""
    try:
        return image.convert("RGBA")
    finally:
        # Closing the file alone would keep the decoded source pixels until the calling thread
        # drops the image, one texture later.
        image.close()


def _load_textures(
    textures: Sequence[str | os.PathLike[str] | np.ndarray | None],
) -> Iterator[np.ndarray | None]:
    """Load textures in order, decoding the image files ahead on worker threads.

    Decoding an image and converting it to RGBA release the GIL, so workers do both while
    the calling thread copies the previous images out and its caller consumes them. That
    thread is the bottleneck, so the workers only decode as far ahead as it is seen to wait
    for, and only while the pixels waiting fit in :data:`_MAX_PENDING_TEXTURE_BYTES`.

    Close the iterator if it is not exhausted, so that no worker thread outlives it.

    Args:
        textures: Paths or URLs to texture images, or image arrays (H, W, C).

    Yields:
        What :func:`load_texture` returns for each texture, with the warnings it emits.
    """
    # Only files on disk decode on workers: the same ones that load_texture_from_file() opens directly.
    paths: list[str | None] = []
    for texture in textures:
        path = os.fspath(texture) if isinstance(texture, os.PathLike) else texture
        file_path = None
        if isinstance(path, str):
            try:
                if not _is_http_url(path) and not _is_usd_package_path(path):
                    file_path = _resolve_file_url(path)
            except ValueError:
                # Not a path that parses: left to load_texture(), which reports it once the texture is reached.
                pass
        paths.append(file_path)
    file_count = len(paths) - paths.count(None)
    # A single file has nothing to decode ahead of, and the calling thread keeps a CPU for itself.
    worker_count = min(_count_usable_cpus() - 1, file_count) if file_count >= 2 else 0
    if worker_count >= 1:
        try:
            from PIL import Image
        except ImportError:
            worker_count = 0
    if worker_count < 1:
        yield from map(load_texture, textures)
        return

    pool = ThreadPoolExecutor(worker_count, thread_name_prefix="newton-texture")
    # The files that workers are decoding or have decoded, by texture index, each with its opened
    # image and the size of its pixels.
    pending: dict[int, tuple[Future[Image.Image], Image.Image, int]] = {}
    pending_bytes = 0
    next_index = 0
    lookahead = 1
    try:
        for index, texture in enumerate(textures):
            future, _, pixel_bytes = pending.pop(index, (None, None, 0))
            pending_bytes -= pixel_bytes
            # Waiting for a worker means the workers are behind: decode one more texture ahead.
            if future is not None and not future.done() and lookahead < worker_count:
                lookahead += 1
            # Submit the next files before the copy below, so the workers decode during it.
            next_index = max(next_index, index + 1)
            while next_index < len(paths) and len(pending) < lookahead:
                path = paths[next_index]
                if path is not None:
                    try:
                        image = Image.open(path)
                    except Exception:
                        # Left to load_texture() below, which reports the failure once the texture is reached.
                        pass
                    else:
                        pixel_bytes = 4 * image.width * image.height
                        if pending and pending_bytes + pixel_bytes > _MAX_PENDING_TEXTURE_BYTES:
                            # Opened again once fewer pixels wait: at the latest when none do.
                            image.close()
                            break
                        pending[next_index] = (pool.submit(_convert_to_rgba, image), image, pixel_bytes)
                        pending_bytes += pixel_bytes
                next_index += 1

            pixels = None
            if future is not None:
                try:
                    pixels = np.ascontiguousarray(np.array(future.result()))
                except Exception:
                    # Decode again below, to report the failure the way load_texture() does.
                    pass
                # Drop the decoded image now rather than at the next texture.
                future = None
            if pixels is None:
                pixels = load_texture(texture)
            yield pixels
    finally:
        for future, image, _ in pending.values():
            if future.cancel():
                image.close()
        pool.shutdown(wait=True)


def linear_texture_to_srgb(texture_image: np.ndarray | None) -> np.ndarray | None:
    """Convert RGB channels from linear light to sRGB/display encoding."""
    if texture_image is None:
        return None

    image = np.asarray(texture_image)
    if image.ndim < 3 or image.shape[-1] < 3 or image.size == 0:
        return np.ascontiguousarray(image)

    out = image.copy()
    if out.dtype == np.uint8:
        # One channel at a time: a 3-channel fancy index triples the gathered temporary.
        for channel in range(3):
            out[..., channel] = _LINEAR_TO_SRGB_UINT8_LUT[out[..., channel]]
        return np.ascontiguousarray(out)

    if np.issubdtype(out.dtype, np.integer):
        scale = float(np.iinfo(out.dtype).max)
        rgb = np.clip(out[..., :3].astype(np.float32) / scale, 0.0, 1.0)
        srgb = _linear_to_srgb(rgb)
        out[..., :3] = np.clip(np.round(srgb * scale), 0.0, scale).astype(out.dtype)
        return np.ascontiguousarray(out)

    out = out.astype(np.float32, copy=False)
    rgb = np.clip(out[..., :3], 0.0, 1.0)
    out[..., :3] = _linear_to_srgb(rgb)
    return np.ascontiguousarray(out.astype(image.dtype, copy=False))


def normalize_texture(
    texture_image: np.ndarray | None,
    *,
    flip_vertical: bool = False,
    require_channels: bool = False,
    scale_unit_range: bool = True,
) -> np.ndarray | None:
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


def compute_texture_hash(texture: str | os.PathLike[str] | np.ndarray | None) -> int:
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
