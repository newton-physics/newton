# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Per-name image logging for the OpenGL viewer.

Owns GL textures, PBOs, and Warp GL-interop registrations for images
logged via :meth:`~newton.viewer.ViewerBase.log_image`. Displays the
selected image name as a dockable ImGui window containing a tile grid.
Selection is driven by a sidebar dropdown (one window at a time).
"""

from __future__ import annotations

import ctypes
import math
import sys
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import warp as wp

_ACCEPTED_C = (1, 3, 4)
_TILE_SPACING_PX: float = 2.0
_INITIAL_TILE_PX: int = 192
_INITIAL_WINDOW_MAX_W: int = 1200
_INITIAL_WINDOW_MAX_H: int = 800
_INITIAL_WINDOW_PAD_X: int = 20
_INITIAL_WINDOW_PAD_Y: int = 40


def _atlas_layout(n: int) -> tuple[int, int]:
    """Pick a square-ish ``(cols, rows)`` for an N-tile texture atlas.

    Packing tiles as a 2D atlas instead of a vertical strip keeps both
    texture dimensions close to ``sqrt(N) * tile_dim``, which avoids
    blowing past ``GL_MAX_TEXTURE_SIZE`` for moderate batch counts.
    """
    if n <= 0:
        return 0, 0
    cols = max(1, math.ceil(math.sqrt(n)))
    rows = math.ceil(n / cols)
    return cols, rows


def compute_grid_layout(
    n: int,
    tile_aspect: float,
    window_w: float,
    window_h: float,
    *,
    spacing_x: float = 0.0,
    spacing_y: float = 0.0,
) -> tuple[int, int, float, float]:
    """Lay out *n* tiles of a given aspect ratio in a window.

    Picks the column count that maximizes total rendered tile area,
    breaking ties toward smaller column counts (wider layouts).

    Args:
        n: Number of tiles (>= 0).
        tile_aspect: Source tile height / width.
        window_w: Available window content width [px].
        window_h: Available window content height [px].
        spacing_x: Horizontal gap between cells [px] (e.g. ``style.item_spacing.x``).
        spacing_y: Vertical gap between cells [px] (e.g. ``style.item_spacing.y``).

    Returns:
        ``(rows, cols, cell_w, cell_h)``. For ``n == 0``, returns
        ``(0, 0, 0.0, 0.0)``.
    """
    if n <= 0:
        return 0, 0, 0.0, 0.0

    best = (1, 1, 0.0, 0.0)
    best_area = -1.0
    for cols in range(1, n + 1):
        rows = math.ceil(n / cols)
        usable_w = max(1.0, window_w - (cols - 1) * spacing_x)
        usable_h = max(1.0, window_h - (rows - 1) * spacing_y)
        cell_w = usable_w / cols
        cell_h = cell_w * tile_aspect
        if rows * cell_h > usable_h:
            cell_h = usable_h / rows
            cell_w = cell_h / tile_aspect if tile_aspect > 0 else 0.0
        area = n * cell_w * cell_h
        # Strict `>` keeps earliest (smaller-cols) on tie.
        if area > best_area:
            best_area = area
            best = (rows, cols, cell_w, cell_h)
    return best


def _dtype_ok(arr: Any) -> bool:
    """Return True if *arr*'s dtype is uint8 or float32 (Warp or NumPy)."""
    if isinstance(arr, np.ndarray):
        return arr.dtype in (np.dtype(np.uint8), np.dtype(np.float32))
    if isinstance(arr, wp.array):
        return arr.dtype in (wp.uint8, wp.float32)
    return False


def _shape_of(arr: Any) -> tuple[int, ...]:
    return tuple(arr.shape)


def _validate(name: str, image: Any) -> tuple[int, int, int, int]:
    """Validate a ``log_image`` call and return a canonical ``(N, H, W, C)``.

    Raises ``ValueError`` with an actionable message on failure. On success,
    returns the canonical 4D shape the caller should use for conversion.

    Canonical shape rules:
      * 2D ``(H, W)``                          -> ``(1, H, W, 1)``
      * 3D ``(H, W, C)`` with C in {1,3,4}    -> ``(1, H, W, C)``
      * 3D otherwise ``(N, H, W)``             -> ``(N, H, W, 1)``
      * 4D ``(N, H, W, C)`` with C in {1,3,4} -> ``(N, H, W, C)``
    """
    if not isinstance(name, str) or not name:
        raise ValueError("log_image: name must be a non-empty string")
    if not isinstance(image, (np.ndarray, wp.array)):
        raise ValueError(f"log_image: expected wp.array or np.ndarray, got {type(image).__name__}")
    if not _dtype_ok(image):
        dtype = image.dtype
        raise ValueError(f"log_image('{name}'): expected uint8 or float32, got {dtype}")

    shape = _shape_of(image)
    if len(shape) not in (2, 3, 4):
        raise ValueError(f"log_image('{name}'): expected 2D, 3D, or 4D array, got shape {shape}")
    if any(dim <= 0 for dim in shape):
        raise ValueError(f"log_image('{name}'): all dimensions must be positive, got shape {shape}")

    if len(shape) == 2:
        h, w = shape
        return 1, h, w, 1
    if len(shape) == 3:
        a, b, c = shape
        # Disambiguate: last dim in {1,3,4} means HWC single color image.
        if c in _ACCEPTED_C:
            return 1, a, b, c
        return a, b, c, 1
    # 4D
    n, h, w, c = shape
    if c not in _ACCEPTED_C:
        raise ValueError(f"log_image('{name}'): expected channel count C in (1, 3, 4), got C={c}")
    return n, h, w, c


def _to_canonical_4d_numpy(image: np.ndarray) -> tuple[np.ndarray, int, int, int, int]:
    """Reshape/tag a validated numpy image into canonical ``(N, H, W, C)`` view.

    Returns ``(arr_4d, N, H, W, C)``. The returned array is a view (no copy)
    when possible; it is always 4D with channel-last layout.
    """
    ndim = image.ndim
    if ndim == 2:
        h, w = image.shape
        return image.reshape(1, h, w, 1), 1, h, w, 1
    if ndim == 3:
        a, b, c = image.shape
        if c in _ACCEPTED_C:
            return image.reshape(1, a, b, c), 1, a, b, c
        return image.reshape(a, b, c, 1), a, b, c, 1
    # 4D
    n, h, w, c = image.shape
    return image, n, h, w, c


def _to_canonical_4d_wp(image: wp.array[Any], n: int, h: int, w: int, c: int) -> wp.array[Any]:
    """Reshape a validated Warp image into canonical ``(N, H, W, C)``."""
    ndim = len(image.shape)
    if ndim == 2:
        return image.reshape((1, h, w, 1))
    if ndim == 3:
        if image.shape[-1] in _ACCEPTED_C:
            return image.reshape((1, h, w, c))
        return image.reshape((n, h, w, 1))
    return image  # 4D — already canonical


@wp.kernel
def _pack_rgba_u8_kernel(
    src: wp.array4d[wp.uint8],
    c_count: int,
    atlas_cols: int,
    out: wp.array3d[wp.uint8],
):
    n, h, w = wp.tid()
    H = src.shape[1]
    W = src.shape[2]
    dst_row = (n // atlas_cols) * H + h
    dst_col = (n % atlas_cols) * W + w
    if c_count == 1:
        v = src[n, h, w, 0]
        out[dst_row, dst_col, 0] = v
        out[dst_row, dst_col, 1] = v
        out[dst_row, dst_col, 2] = v
        out[dst_row, dst_col, 3] = wp.uint8(255)
    elif c_count == 3:
        out[dst_row, dst_col, 0] = src[n, h, w, 0]
        out[dst_row, dst_col, 1] = src[n, h, w, 1]
        out[dst_row, dst_col, 2] = src[n, h, w, 2]
        out[dst_row, dst_col, 3] = wp.uint8(255)
    else:  # c_count == 4
        out[dst_row, dst_col, 0] = src[n, h, w, 0]
        out[dst_row, dst_col, 1] = src[n, h, w, 1]
        out[dst_row, dst_col, 2] = src[n, h, w, 2]
        out[dst_row, dst_col, 3] = src[n, h, w, 3]


@wp.kernel
def _pack_rgba_f32_kernel(
    src: wp.array4d[wp.float32],
    c_count: int,
    atlas_cols: int,
    out: wp.array3d[wp.uint8],
):
    # Truncation (matches CPU reference). Do NOT use +0.5 rounding.
    n, h, w = wp.tid()
    H = src.shape[1]
    W = src.shape[2]
    dst_row = (n // atlas_cols) * H + h
    dst_col = (n % atlas_cols) * W + w
    if c_count == 1:
        v = wp.uint8(wp.int32(wp.clamp(src[n, h, w, 0], 0.0, 1.0) * 255.0))
        out[dst_row, dst_col, 0] = v
        out[dst_row, dst_col, 1] = v
        out[dst_row, dst_col, 2] = v
        out[dst_row, dst_col, 3] = wp.uint8(255)
    elif c_count == 3:
        out[dst_row, dst_col, 0] = wp.uint8(wp.int32(wp.clamp(src[n, h, w, 0], 0.0, 1.0) * 255.0))
        out[dst_row, dst_col, 1] = wp.uint8(wp.int32(wp.clamp(src[n, h, w, 1], 0.0, 1.0) * 255.0))
        out[dst_row, dst_col, 2] = wp.uint8(wp.int32(wp.clamp(src[n, h, w, 2], 0.0, 1.0) * 255.0))
        out[dst_row, dst_col, 3] = wp.uint8(255)
    else:  # c_count == 4
        out[dst_row, dst_col, 0] = wp.uint8(wp.int32(wp.clamp(src[n, h, w, 0], 0.0, 1.0) * 255.0))
        out[dst_row, dst_col, 1] = wp.uint8(wp.int32(wp.clamp(src[n, h, w, 1], 0.0, 1.0) * 255.0))
        out[dst_row, dst_col, 2] = wp.uint8(wp.int32(wp.clamp(src[n, h, w, 2], 0.0, 1.0) * 255.0))
        out[dst_row, dst_col, 3] = wp.uint8(wp.int32(wp.clamp(src[n, h, w, 3], 0.0, 1.0) * 255.0))


def _pack_rgba_warp(src: wp.array[Any], c_count: int, atlas_cols: int, out: wp.array[Any]) -> None:
    """Dispatch to the correct pack kernel for *src*'s dtype.

    *src* must be ``wp.array4d`` of ``wp.uint8`` or ``wp.float32``.
    *out* must be ``wp.array3d[wp.uint8]`` with shape
    ``(rows*H, cols*W, 4)`` where ``cols == atlas_cols`` and
    ``rows == ceil(N / cols)``. Tile ``i`` lands at row-major slot
    ``(i // cols, i % cols)``; trailing slots in the last row are
    untouched, so callers should clear ``out`` before launch if those
    pixels matter (the texture realloc path does this implicitly).
    """
    n, h, w, _ = src.shape
    if src.dtype == wp.uint8:
        kernel = _pack_rgba_u8_kernel
    elif src.dtype == wp.float32:
        kernel = _pack_rgba_f32_kernel
    else:
        raise TypeError(f"_pack_rgba_warp: unsupported dtype {src.dtype}")
    wp.launch(
        kernel,
        dim=(n, h, w),
        inputs=[src, int(c_count), int(atlas_cols)],
        outputs=[out],
        device=src.device,
    )


def _convert_to_packed_rgba_numpy(image: np.ndarray, atlas_cols: int) -> np.ndarray:
    """Convert any validated image to a packed atlas of ``(rows*H, cols*W, 4) uint8``.

    Grayscale replicates luma to RGB, alpha defaults to 255, RGB expands to
    RGBA with A=255, float32 is clipped to [0,1] and scaled by 255. Tile
    ``i`` lands at row-major slot ``(i // atlas_cols, i % atlas_cols)``;
    trailing unused slots in the last row are zero-filled.
    """
    arr, n, h, w, c = _to_canonical_4d_numpy(image)
    cols = max(1, int(atlas_cols))
    rows = math.ceil(n / cols)

    if arr.dtype == np.float32:
        arr = np.clip(arr, 0.0, 1.0)
        arr = (arr * 255.0).astype(np.uint8)  # truncate toward zero

    # Per-tile RGBA expansion first, then scatter into the atlas grid.
    tiles = np.empty((n, h, w, 4), dtype=np.uint8)
    if c == 1:
        tiles[..., 0] = arr[..., 0]
        tiles[..., 1] = arr[..., 0]
        tiles[..., 2] = arr[..., 0]
        tiles[..., 3] = 255
    elif c == 3:
        tiles[..., :3] = arr
        tiles[..., 3] = 255
    else:  # c == 4
        tiles[...] = arr

    out = np.zeros((rows * h, cols * w, 4), dtype=np.uint8)
    for i in range(n):
        r, ccol = divmod(i, cols)
        out[r * h : (r + 1) * h, ccol * w : (ccol + 1) * w] = tiles[i]
    return out


@dataclass
class LoggedImage:
    """Per-name state for an image logged to the viewer."""

    name: str
    tex_id: int = 0
    tex_w: int = 0
    tex_h: int = 0
    pbo_id: int | None = None
    wp_pbo: wp.RegisteredGLBuffer | None = None
    n: int = 0
    h: int = 0
    w: int = 0
    c: int = 0
    # Atlas grid the texture is packed as: cols*W = tex_w, rows*H = tex_h.
    # ``atlas_cols * atlas_rows`` may exceed ``n``; trailing slots are zero.
    atlas_cols: int = 0
    atlas_rows: int = 0
    tile_aspect: float = 1.0
    window_initialized: bool = False


class ImageLogger:
    """Owns GL resources for images logged via :meth:`~newton.viewer.ViewerBase.log_image`.

    One instance is constructed by :class:`ViewerGL`. This class has no
    reference to the viewer itself; it takes the viewer's CUDA device at
    construction and owns only its own GL/Warp state.
    """

    def __init__(self, device: wp.Device, sidebar_width_px: float = 300.0):
        """Create an ``ImageLogger``.

        Args:
            device: The CUDA device used by the viewer. Warp arrays on this
                device are uploaded via the GPU path (PBO + kernel); all
                others fall back to a CPU copy.
            sidebar_width_px: Width of the viewer's main sidebar [px]. Used
                to avoid placing newly-opened image windows underneath the
                sidebar on their first appearance.
        """
        self._device = device
        self._sidebar_width_px = sidebar_width_px
        self._images: dict[str, LoggedImage] = {}
        self._warned_device_mismatch: dict[str, wp.Device] = {}
        self._selected: str | None = None

    @property
    def device(self) -> wp.Device:
        """The CUDA device this logger was bound to."""
        return self._device

    def log(self, name: str, image: wp.array[Any] | np.ndarray) -> None:
        """Validate, convert, and upload an image under *name*.

        See :meth:`~newton.viewer.ViewerBase.log_image` for the public contract.
        """
        n, h, w, c = _validate(name, image)
        entry = self._images.get(name)
        if entry is None:
            entry = LoggedImage(name=name)
            self._images[name] = entry
            # Auto-select the first logged image so the user sees something
            # immediately. Don't switch selection for subsequent new names.
            if self._selected is None:
                self._selected = name

        needs_realloc = (entry.n, entry.h, entry.w, entry.c) != (n, h, w, c)

        # Pack as a square-ish atlas instead of a single (1, N) strip so we
        # don't blow past GL_MAX_TEXTURE_SIZE for moderate batches.
        atlas_cols, atlas_rows = _atlas_layout(n)
        tex_w = atlas_cols * w
        tex_h = atlas_rows * h

        if self._can_use_gpu_path(image):
            src_4d = _to_canonical_4d_wp(image, n, h, w, c)
            self._upload_gpu(entry, src_4d, n, h, w, c, atlas_cols, tex_w, tex_h, needs_realloc)
        else:
            self._maybe_warn_cross_device(name, image)
            self._upload_cpu(entry, image, n, h, w, c, atlas_cols, tex_w, tex_h, needs_realloc)

        # Only commit metadata after upload succeeds.
        entry.n, entry.h, entry.w, entry.c = n, h, w, c
        entry.atlas_cols, entry.atlas_rows = atlas_cols, atlas_rows
        entry.tile_aspect = h / w

    def draw(self) -> None:
        """Draw the selected image window (if any).

        Called once per frame inside the viewer's ImGui frame block.
        At most one window is visible at a time; selection is driven by
        :meth:`draw_controls`.
        """
        from imgui_bundle import imgui

        if self._selected is None:
            return
        entry = self._images.get(self._selected)
        if entry is None or entry.n == 0 or entry.tex_id == 0:
            return

        if not entry.window_initialized:
            init_cols = max(1, math.ceil(math.sqrt(entry.n)))
            init_rows = math.ceil(entry.n / init_cols)
            w_px = min(
                int(init_cols * _INITIAL_TILE_PX + (init_cols - 1) * _TILE_SPACING_PX + _INITIAL_WINDOW_PAD_X),
                _INITIAL_WINDOW_MAX_W,
            )
            h_px = min(
                int(
                    init_rows * _INITIAL_TILE_PX * entry.tile_aspect
                    + (init_rows - 1) * _TILE_SPACING_PX
                    + _INITIAL_WINDOW_PAD_Y
                ),
                _INITIAL_WINDOW_MAX_H,
            )
            viewport = imgui.get_main_viewport()
            vp_w = viewport.work_size.x
            vp_h = viewport.work_size.y
            avail_w = max(0.0, vp_w - self._sidebar_width_px)
            pos_x = self._sidebar_width_px + max(0.0, (avail_w - w_px) / 2.0)
            pos_y = max(0.0, (vp_h - h_px) / 2.0)
            imgui.set_next_window_pos(imgui.ImVec2(float(pos_x), float(pos_y)), imgui.Cond_.once)
            imgui.set_next_window_size(imgui.ImVec2(float(w_px), float(h_px)), imgui.Cond_.once)
            entry.window_initialized = True

        expanded, stays_open = imgui.begin(entry.name, True)
        try:
            if expanded:
                imgui.push_style_var(
                    imgui.StyleVar_.item_spacing,
                    imgui.ImVec2(_TILE_SPACING_PX, _TILE_SPACING_PX),
                )
                content_w = imgui.get_content_region_avail().x
                content_h = imgui.get_content_region_avail().y
                _, cols, cell_w, cell_h = compute_grid_layout(
                    entry.n,
                    entry.tile_aspect,
                    content_w,
                    content_h,
                    spacing_x=_TILE_SPACING_PX,
                    spacing_y=_TILE_SPACING_PX,
                )
                # UVs sample tile ``i`` from atlas slot
                # ``(i // atlas_cols, i % atlas_cols)``. The display grid
                # ``cols`` (above) is independent of ``atlas_cols``.
                u_step = entry.w / float(entry.tex_w) if entry.tex_w > 0 else 0.0
                v_step = entry.h / float(entry.tex_h) if entry.tex_h > 0 else 0.0
                for i in range(entry.n):
                    if i > 0 and (i % cols) != 0:
                        imgui.same_line()
                    a_row, a_col = divmod(i, entry.atlas_cols)
                    uv0 = imgui.ImVec2(a_col * u_step, a_row * v_step)
                    uv1 = imgui.ImVec2((a_col + 1) * u_step, (a_row + 1) * v_step)
                    imgui.image(
                        imgui.ImTextureRef(entry.tex_id),
                        imgui.ImVec2(cell_w, cell_h),
                        uv0=uv0,
                        uv1=uv1,
                    )
                imgui.pop_style_var()
        finally:
            imgui.end()

        if not stays_open:
            # User clicked the window's close (X); de-select so the dropdown
            # reflects reality and the window can be re-opened via the dropdown.
            self._selected = None

    def draw_controls(self) -> None:
        """Render the sidebar dropdown selecting which image window is shown.

        Intended to be called from inside the viewer's main sidebar
        ImGui block. No-op when no images have been logged. Renders as a
        collapsing header to match the other sidebar sections.
        """
        if not self._images:
            return

        from imgui_bundle import imgui

        label = f"Logged Images ({len(self._images)})"
        if not imgui.collapsing_header(label, imgui.TreeNodeFlags_.default_open.value):
            return

        names = list(self._images.keys())
        items = ["Hide", *names]
        if self._selected is not None and self._selected in names:
            current = names.index(self._selected) + 1
        else:
            current = 0

        changed, new_idx = imgui.combo("##logged_images", current, items)
        if changed:
            self._selected = None if new_idx == 0 else names[new_idx - 1]

    def clear(self) -> None:
        """Destroy all GL resources. Idempotent."""
        for entry in list(self._images.values()):
            self._free_entry(entry)
        self._images.clear()
        self._selected = None

    # --- Internals ---

    def _can_use_gpu_path(self, image) -> bool:
        """Return True if *image* is on a CUDA device matching the viewer."""
        if not isinstance(image, wp.array):
            return False
        if not self._device.is_cuda:
            return False
        return image.device == self._device

    def _maybe_warn_cross_device(self, name: str, image) -> None:
        """Warn when a CUDA wp.array on a non-viewer device forces a D2H copy.

        Re-warns if the observed device changes for the same name, so a user
        who switches sources mid-run still gets feedback.
        """
        if not isinstance(image, wp.array):
            return
        if not self._device.is_cuda:
            return
        if image.device == self._device:
            return
        if not getattr(image.device, "is_cuda", False):
            return
        if self._warned_device_mismatch.get(name) == image.device:
            return
        warnings.warn(
            f"log_image('{name}'): array is on {image.device} but viewer is on "
            f"{self._device}; falling back to CPU upload (D2H copy per frame).",
            stacklevel=3,
        )
        self._warned_device_mismatch[name] = image.device

    def _ensure_texture(self, entry: LoggedImage, tex_w: int, tex_h: int, realloc: bool) -> None:
        from pyglet import gl

        max_size = (gl.GLint * 1)()
        gl.glGetIntegerv(gl.GL_MAX_TEXTURE_SIZE, max_size)
        if tex_w > max_size[0] or tex_h > max_size[0]:
            raise ValueError(
                f"log_image('{entry.name}'): texture size {tex_w}x{tex_h} exceeds "
                f"GL_MAX_TEXTURE_SIZE={max_size[0]}. Reduce batch count or tile resolution."
            )

        if entry.tex_id == 0:
            tex = (gl.GLuint * 1)()
            gl.glGenTextures(1, tex)
            entry.tex_id = int(tex[0])
            gl.glBindTexture(gl.GL_TEXTURE_2D, entry.tex_id)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
            realloc = True

        if realloc or entry.tex_w != tex_w or entry.tex_h != tex_h:
            gl.glBindTexture(gl.GL_TEXTURE_2D, entry.tex_id)
            gl.glTexImage2D(
                gl.GL_TEXTURE_2D,
                0,
                gl.GL_RGBA8,
                tex_w,
                tex_h,
                0,
                gl.GL_RGBA,
                gl.GL_UNSIGNED_BYTE,
                None,
            )
            entry.tex_w, entry.tex_h = tex_w, tex_h

    def _ensure_pbo(self, entry: LoggedImage, byte_size: int, realloc: bool) -> None:
        from pyglet import gl

        if entry.pbo_id is None:
            pbo = gl.GLuint()
            gl.glGenBuffers(1, pbo)
            entry.pbo_id = int(pbo.value)
            realloc = True

        if realloc:
            if entry.wp_pbo is not None:
                entry.wp_pbo = None
            gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, entry.pbo_id)
            gl.glBufferData(gl.GL_PIXEL_UNPACK_BUFFER, gl.GLsizeiptr(byte_size), None, gl.GL_STREAM_DRAW)
            gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, 0)
            try:
                entry.wp_pbo = wp.RegisteredGLBuffer(
                    gl_buffer_id=entry.pbo_id,
                    device=self._device,
                    flags=wp.RegisteredGLBuffer.WRITE_DISCARD,
                )
            except Exception:
                # Rollback the PBO so the entry isn't left in a
                # half-initialized state; next call will retry fresh.
                buf = (gl.GLuint * 1)(entry.pbo_id)
                gl.glDeleteBuffers(1, buf)
                entry.pbo_id = None
                raise

    def _upload_gpu(
        self,
        entry: LoggedImage,
        image,
        n: int,
        h: int,
        w: int,
        c: int,
        atlas_cols: int,
        tex_w: int,
        tex_h: int,
        realloc: bool,
    ) -> None:
        from pyglet import gl

        byte_size = tex_w * tex_h * 4
        # ``glTexImage2D(..., None)`` on realloc clears the texture, but the
        # PBO (and thus the kernel's output buffer) carries stale bytes
        # across frames. Trailing atlas slots aren't written by the pack
        # kernel, so any tile-shape change must zero the buffer to avoid
        # showing leftover pixels in the unused last-row slots.
        atlas_changed = realloc or entry.tex_w != tex_w or entry.tex_h != tex_h
        self._ensure_texture(entry, tex_w, tex_h, realloc)
        self._ensure_pbo(entry, byte_size, realloc)

        if entry.wp_pbo is None:
            raise RuntimeError(f"log_image('{entry.name}'): PBO-CUDA registration failed to initialize")
        mapped = None
        try:
            mapped = entry.wp_pbo.map(dtype=wp.uint8, shape=(tex_h, tex_w, 4))
            if atlas_changed:
                mapped.zero_()
            _pack_rgba_warp(image, c, atlas_cols, mapped)
        finally:
            if mapped is not None:
                entry.wp_pbo.unmap()

        gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, entry.pbo_id)
        gl.glBindTexture(gl.GL_TEXTURE_2D, entry.tex_id)
        gl.glTexSubImage2D(
            gl.GL_TEXTURE_2D,
            0,
            0,
            0,
            tex_w,
            tex_h,
            gl.GL_RGBA,
            gl.GL_UNSIGNED_BYTE,
            ctypes.c_void_p(0),
        )
        gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, 0)

    def _upload_cpu(
        self,
        entry: LoggedImage,
        image,
        n: int,
        h: int,
        w: int,
        c: int,
        atlas_cols: int,
        tex_w: int,
        tex_h: int,
        realloc: bool,
    ) -> None:
        from pyglet import gl

        self._ensure_texture(entry, tex_w, tex_h, realloc)
        if isinstance(image, wp.array):
            host = image.numpy()
        else:
            host = image
        packed = _convert_to_packed_rgba_numpy(host, atlas_cols)  # (tex_h, tex_w, 4) uint8
        gl.glBindTexture(gl.GL_TEXTURE_2D, entry.tex_id)
        gl.glTexSubImage2D(
            gl.GL_TEXTURE_2D,
            0,
            0,
            0,
            tex_w,
            tex_h,
            gl.GL_RGBA,
            gl.GL_UNSIGNED_BYTE,
            packed.ctypes.data_as(ctypes.c_void_p),
        )

    def _free_entry(self, entry: LoggedImage) -> None:
        from pyglet import gl

        # Drop the Warp registration before deleting the GL buffer: the Warp
        # side holds a handle that becomes invalid once glDeleteBuffers runs.
        if entry.wp_pbo is not None:
            entry.wp_pbo = None

        try:
            if entry.pbo_id is not None:
                buf = (gl.GLuint * 1)(entry.pbo_id)
                gl.glDeleteBuffers(1, buf)
            if entry.tex_id != 0:
                tex = (gl.GLuint * 1)(entry.tex_id)
                gl.glDeleteTextures(1, tex)
        except Exception as exc:
            # GL calls may fail when the context is already torn down during
            # interpreter shutdown; there's no reasonable recovery and the
            # warnings machinery may itself be unavailable at that point.
            if not sys.is_finalizing():
                warnings.warn(
                    f"log_image('{entry.name}'): GL cleanup failed: {exc!r}",
                    stacklevel=2,
                )
        finally:
            entry.pbo_id = None
            entry.tex_id = 0
