# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import io
import os
import warnings
from urllib.parse import unquote, urlparse
from urllib.request import urlopen

import numpy as np
import warp as wp

from ..exceptions import NewtonWarning

_texture_url_cache: dict[str, bytes] = {}


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


def _decode_hdr_rgbe(data: bytes) -> np.ndarray:
    """Decode Radiance RGBE (``.hdr``) bytes into a linear float RGB image.

    Supports Y-major ``32-bit_rle_rgbe`` images with flat or adaptive
    run-length-encoded scanlines. Old-style run-length packets are rejected.

    Args:
        data: Raw file contents of a Radiance ``.hdr`` image.

    Returns:
        Linear float32 RGB image of shape ``(H, W, 3)``, ordered top to
        bottom and left to right.
    """
    scan = np.frombuffer(data, dtype=np.uint8)

    def readline(start: int) -> tuple[bytes, int]:
        end = data.find(b"\n", start)
        if end < 0:
            raise ValueError("Incomplete HDR header")
        return data[start:end].strip(), end + 1

    line, ptr = readline(0)
    if not line.startswith(b"#?"):
        raise ValueError("Not a Radiance HDR file")
    formats = []
    while True:
        line, ptr = readline(ptr)
        if not line:
            break
        if line.startswith(b"FORMAT="):
            formats.append(line[7:])
    if formats != [b"32-bit_rle_rgbe"]:
        raise ValueError("Only Radiance 32-bit_rle_rgbe images are supported")

    resolution, ptr = readline(ptr)
    tokens = resolution.split()
    if len(tokens) != 4 or tokens[0] not in (b"-Y", b"+Y") or tokens[2] not in (b"-X", b"+X"):
        raise ValueError(f"Unsupported HDR resolution line: {resolution!r}")
    height, width = int(tokens[1]), int(tokens[3])
    if height <= 0 or width <= 0:
        raise ValueError("HDR dimensions must be positive")

    rgbe = np.empty((height, width, 4), dtype=np.uint8)
    for y in range(height):
        if ptr + 4 > scan.size:
            raise ValueError("Truncated HDR scanline")
        is_new_rle = 8 <= width <= 0x7FFF and scan[ptr] == 2 and scan[ptr + 1] == 2 and scan[ptr + 2] < 128
        if is_new_rle:
            if ((int(scan[ptr + 2]) << 8) | int(scan[ptr + 3])) != width:
                raise ValueError("HDR scanline width does not match the header")
            ptr += 4
            for channel in range(4):
                x = 0
                while x < width:
                    if ptr >= scan.size:
                        raise ValueError("Truncated HDR RLE packet")
                    count = int(scan[ptr])
                    ptr += 1
                    run = count - 128 if count > 128 else count
                    size = 1 if count > 128 else run
                    if run == 0 or run > width - x or ptr + size > scan.size:
                        raise ValueError("Invalid HDR RLE packet")
                    rgbe[y, x : x + run, channel] = scan[ptr : ptr + size]
                    ptr += size
                    x += run
        else:
            if ptr + width * 4 > scan.size:
                raise ValueError("Truncated HDR scanline")
            row = scan[ptr : ptr + width * 4].reshape(width, 4)
            if np.any(np.all(row[:, :3] == 1, axis=1)):
                raise ValueError("Old-style HDR run-length encoding is unsupported")
            rgbe[y] = row
            ptr += width * 4

    exponent = rgbe[..., 3].astype(np.int32)
    scale = np.where(exponent > 0, np.ldexp(1.0, exponent - (128 + 8)), 0.0).astype(np.float32)
    rgb = rgbe[..., :3].astype(np.float32) * scale[..., None]

    if tokens[0] == b"+Y":
        rgb = rgb[::-1]
    if tokens[2] == b"-X":
        rgb = rgb[:, ::-1]
    return np.ascontiguousarray(rgb, dtype=np.float32)


def load_hdr_image(path: str | os.PathLike[str]) -> np.ndarray:
    """Load a local Radiance RGBE image, raising on missing or malformed input."""
    with open(path, "rb") as handle:
        return _decode_hdr_rgbe(handle.read())


def load_equirect_image(image: str | os.PathLike[str] | np.ndarray | wp.array) -> np.ndarray:
    """Load a nonempty equirectangular image as finite, non-negative linear RGB.

    Floating arrays contain linear radiance. Integer arrays and LDR image files
    contain sRGB values. Channels after RGB are ignored.
    """
    if isinstance(image, wp.array):
        image = image.numpy()
    if isinstance(image, (str, os.PathLike)):
        path = os.fspath(image)
        image = load_hdr_image(path) if os.path.splitext(path)[1].lower() == ".hdr" else load_texture(path)
    if image is None:
        raise ValueError("Failed to load dome light image")
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] < 3 or min(array.shape[:2]) == 0:
        raise ValueError(f"Dome image must have nonempty shape (H, W, C>=3), got {array.shape}")
    if array.dtype.kind not in "uif":
        raise ValueError("Dome image must contain real numeric radiance")
    rgb = array[..., :3].astype(np.float32)
    if not np.isfinite(rgb).all() or np.any(rgb < 0.0):
        raise ValueError("Dome radiance must be finite and non-negative")
    if np.issubdtype(array.dtype, np.integer):
        rgb /= float(np.iinfo(array.dtype).max)
        rgb = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    return np.ascontiguousarray(rgb, dtype=np.float32)


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
