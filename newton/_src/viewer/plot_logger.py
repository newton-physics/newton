# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import collections
import ctypes
from collections.abc import Callable
from typing import Any

import numpy as np
import warp as wp


class PlotLogger:
    """Store and draw live scalar plots and array heatmaps for GL and RTX viewers.

    Logging is independent of window creation. OpenGL is loaded only when
    drawing heatmaps; texture operations activate the owning window's context.
    """

    def __init__(self, plot_history_size: int, *, get_window: Callable[[], Any]):
        if not isinstance(plot_history_size, int) or isinstance(plot_history_size, bool):
            raise TypeError("plot_history_size must be an integer")
        if plot_history_size <= 0:
            raise ValueError("plot_history_size must be > 0")
        self._scalar_buffers: dict[str, collections.deque] = {}
        self._scalar_arrays: dict[str, np.ndarray | None] = {}
        self._scalar_accumulators: dict[str, list[float]] = {}
        self._scalar_smoothing: dict[str, int] = {}
        self._array_buffers: dict[str, np.ndarray] = {}
        self._array_dirty: set[str] = set()
        self._array_textures: dict[str, dict[str, Any]] = {}
        self._heatmap_min_cell_pixels = 3.0
        self._heatmap_nan_rgba = np.array([51, 51, 51, 255], dtype=np.uint8)
        self._heatmap_color_lut = self._build_heatmap_color_lut()
        self._plot_history_size = plot_history_size

        self._get_window = get_window

    def _get_gl(self):
        try:
            self._get_window().switch_to()
        except AttributeError:
            # The window or its context may already be destroyed during shutdown.
            pass
        from pyglet import gl

        return gl

    def log_array(self, name: str, array: wp.array[Any] | np.ndarray | None):
        """
        Log a numeric array for visualization.

        Args:
            name: Unique path/name for the array signal.
            array: Array data to visualize, or ``None`` to remove a previously
                logged array.
        """
        if array is None:
            self._array_buffers.pop(name, None)
            self._array_dirty.discard(name)
            self._delete_array_texture(name)
            return

        array_np = array.numpy() if isinstance(array, wp.array) else np.asarray(array)
        array_np = np.asarray(array_np, dtype=np.float32)

        if array_np.ndim == 0:
            array_np = array_np.reshape(1, 1)
        elif array_np.ndim == 1:
            array_np = array_np.reshape(1, -1)
        elif array_np.ndim != 2:
            raise ValueError("log_array only supports scalar, 1-D, or 2-D arrays.")

        self._array_buffers[name] = np.ascontiguousarray(array_np)
        self._array_dirty.add(name)

    def log_scalar(
        self,
        name: str,
        value: int | float | bool | np.number,
        *,
        clear: bool = False,
        smoothing: int = 1,
    ):
        """
        Log a scalar value as a live time-series plot.

        Each unique *name* creates a separate line plot displayed in an
        auto-generated "Plots" window.  Values are stored in a rolling
        buffer of the last ``plot_history_size`` samples.

        Args:
            name: Unique path/name for the scalar signal.
            value: Scalar value to record.
            clear: If ``True``, discard previously recorded samples for
                *name* before logging the new value.
            smoothing: Number of raw samples to average before committing
                a point to the plot history.  Defaults to ``1`` (no smoothing).
        """
        if smoothing < 1:
            raise ValueError("smoothing must be >= 1")
        val = float(value.item() if hasattr(value, "item") else value)
        buf = self._scalar_buffers.get(name)
        if buf is None:
            buf = collections.deque(maxlen=self._plot_history_size)
            self._scalar_buffers[name] = buf
        elif clear:
            buf.clear()
            self._scalar_accumulators.pop(name, None)

        self._scalar_smoothing[name] = smoothing
        if smoothing <= 1:
            buf.append(val)
        else:
            acc = self._scalar_accumulators.get(name)
            if acc is None:
                acc = []
                self._scalar_accumulators[name] = acc
            acc.append(val)
            if len(acc) >= smoothing:
                buf.append(sum(acc) / len(acc))
                acc.clear()

        self._scalar_arrays[name] = None

    def clear_matching(self, owns: Callable[[str], bool]) -> None:
        """Release signals and textures whose qualified names match ``owns``."""
        for buffers in (
            self._scalar_buffers,
            self._scalar_arrays,
            self._scalar_accumulators,
            self._scalar_smoothing,
            self._array_buffers,
        ):
            for name in list(buffers):
                if owns(name):
                    del buffers[name]
        self._array_dirty.difference_update(name for name in list(self._array_dirty) if owns(name))
        for name in list(self._array_textures):
            if owns(name):
                self._delete_array_texture(name)

    def clear(self) -> None:
        """Release all logged data and textures before closing the window."""
        self.clear_matching(lambda _name: True)

    def _delete_array_texture(self, name: str):
        texture_state = self._array_textures.pop(name, None)
        if texture_state is None:
            return
        texture_id = texture_state.get("texture_id")
        if texture_id is None:
            return
        gl = self._get_gl()
        texture_ids = (gl.GLuint * 1)(texture_id)
        gl.glDeleteTextures(1, texture_ids)

    def draw(self, ui) -> None:
        """Render floating time-series plot window for log_scalar() data and array heatmaps."""
        scalar_buffers = self._scalar_buffers
        array_buffers = self._array_buffers
        if not scalar_buffers and not array_buffers:
            return
        imgui = ui.imgui
        io = ui.io
        s = ui.dpi_scale
        scalar_arrays = self._scalar_arrays
        plot_history_size = self._plot_history_size
        window_width = 400 * s
        item_height = len(scalar_buffers) * 140 * s + len(array_buffers) * 260 * s
        window_height = min(io.display_size[1] - 20 * s, item_height + 60 * s)
        # ``first_use_ever`` keeps user-dragged positions stable across
        # collapse/expand cycles and survives ``imgui.ini`` reloads.
        imgui.set_next_window_pos(
            imgui.ImVec2(io.display_size[0] - window_width - 10 * s, io.display_size[1] - window_height - 10 * s),
            imgui.Cond_.first_use_ever,
        )
        imgui.set_next_window_size(imgui.ImVec2(window_width, window_height), imgui.Cond_.first_use_ever)
        n = plot_history_size
        expanded = imgui.begin("Plots")
        if expanded:
            graph_size = imgui.ImVec2(-1, 100 * s)
            for name, buf in scalar_buffers.items():
                arr = scalar_arrays.get(name)
                if arr is None:
                    arr = np.full(n, np.nan, dtype=np.float32)
                    arr[n - len(buf) :] = np.array(buf, dtype=np.float32)
                    scalar_arrays[name] = arr
                overlay = f"{buf[-1]:.4g}" if buf else ""
                if imgui.collapsing_header(name, imgui.TreeNodeFlags_.default_open.value):
                    imgui.plot_lines(f"##{name}", arr, graph_size=graph_size, overlay_text=overlay)
            for name, array in array_buffers.items():
                if imgui.collapsing_header(name, imgui.TreeNodeFlags_.default_open.value):
                    self._render_array_heatmap(imgui, name, array, window_width - 40.0 * s, dpi_scale=s)
        imgui.end()

    @staticmethod
    def _build_heatmap_color_lut() -> np.ndarray:
        inferno_stops = (
            (0.0, (0.001, 0.000, 0.014)),
            (0.2, (0.169, 0.042, 0.341)),
            (0.4, (0.416, 0.090, 0.433)),
            (0.6, (0.698, 0.165, 0.388)),
            (0.8, (0.944, 0.403, 0.121)),
            (1.0, (0.988, 0.998, 0.645)),
        )
        lut = np.empty((256, 4), dtype=np.uint8)
        for index, value in enumerate(np.linspace(0.0, 1.0, 256, dtype=np.float32)):
            for stop_index in range(len(inferno_stops) - 1):
                t0, c0 = inferno_stops[stop_index]
                t1, c1 = inferno_stops[stop_index + 1]
                if value <= t1:
                    alpha = 0.0 if t1 <= t0 else (float(value) - t0) / (t1 - t0)
                    rgb = [round(255.0 * ((1.0 - alpha) * c0[channel] + alpha * c1[channel])) for channel in range(3)]
                    lut[index, :3] = rgb
                    lut[index, 3] = 255
                    break
            else:
                lut[index, :3] = [round(255.0 * channel) for channel in inferno_stops[-1][1]]
                lut[index, 3] = 255
        return lut

    @staticmethod
    def _downsample_heatmap(array: np.ndarray, target_rows: int, target_cols: int) -> np.ndarray:
        rows, cols = array.shape
        if rows <= target_rows and cols <= target_cols:
            return array

        row_factor = max(1, (rows + target_rows - 1) // target_rows)
        col_factor = max(1, (cols + target_cols - 1) // target_cols)
        new_rows = max(1, rows // row_factor)
        new_cols = max(1, cols // col_factor)
        if new_rows == rows and new_cols == cols:
            return array

        trimmed = array[: new_rows * row_factor, : new_cols * col_factor]
        finite_mask = np.isfinite(trimmed)
        safe_values = np.where(finite_mask, trimmed, 0.0)
        reshaped_shape = (new_rows, row_factor, new_cols, col_factor)
        value_sum = safe_values.reshape(reshaped_shape).sum(axis=(1, 3), dtype=np.float64)
        value_count = finite_mask.reshape(reshaped_shape).sum(axis=(1, 3))
        downsampled = np.full((new_rows, new_cols), np.nan, dtype=np.float32)
        np.divide(value_sum, value_count, out=downsampled, where=value_count > 0)
        return downsampled

    def _colorize_heatmap(self, array: np.ndarray) -> tuple[np.ndarray, float, float]:
        finite_mask = np.isfinite(array)
        if not np.any(finite_mask):
            rgba = np.empty((*array.shape, 4), dtype=np.uint8)
            rgba[...] = self._heatmap_nan_rgba
            return np.ascontiguousarray(rgba), float("nan"), float("nan")

        finite_values = array[finite_mask]
        value_min = float(np.min(finite_values))
        value_max = float(np.max(finite_values))
        denom = max(value_max - value_min, 1.0e-8)

        normalized = np.zeros(array.shape, dtype=np.float32)
        np.subtract(array, value_min, out=normalized, where=finite_mask)
        np.divide(normalized, denom, out=normalized, where=finite_mask)
        np.clip(normalized, 0.0, 1.0, out=normalized)

        lut_indices = np.rint(normalized * 255.0).astype(np.uint8)
        rgba = self._heatmap_color_lut[lut_indices].copy()
        rgba[~finite_mask] = self._heatmap_nan_rgba
        return np.ascontiguousarray(rgba), value_min, value_max

    def _ensure_array_texture(self, name: str, width: int, height: int) -> dict[str, Any]:
        texture_state = self._array_textures.get(name)
        if texture_state is not None and texture_state["size"] == (width, height):
            return texture_state

        if texture_state is not None:
            self._delete_array_texture(name)

        gl = self._get_gl()
        texture_id = (gl.GLuint * 1)()
        gl.glGenTextures(1, texture_id)
        gl.glBindTexture(gl.GL_TEXTURE_2D, texture_id[0])
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
        gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D,
            0,
            gl.GL_RGBA8,
            width,
            height,
            0,
            gl.GL_RGBA,
            gl.GL_UNSIGNED_BYTE,
            None,
        )
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

        texture_state = {
            "texture_id": texture_id[0],
            "size": (width, height),
            "source_shape": None,
            "display_shape": None,
            "value_min": 0.0,
            "value_max": 0.0,
        }
        self._array_textures[name] = texture_state
        return texture_state

    def _update_array_texture(self, texture_id: int, rgba: np.ndarray):
        gl = self._get_gl()
        gl.glBindTexture(gl.GL_TEXTURE_2D, texture_id)
        gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
        gl.glTexSubImage2D(
            gl.GL_TEXTURE_2D,
            0,
            0,
            0,
            rgba.shape[1],
            rgba.shape[0],
            gl.GL_RGBA,
            gl.GL_UNSIGNED_BYTE,
            rgba.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
        )
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

    def _render_array_heatmap(self, imgui, name: str, array: np.ndarray, width: float, dpi_scale: float = 1.0):
        s = max(1.0, float(dpi_scale))

        rows, cols = array.shape
        if array.size == 0:
            imgui.text(f"shape {rows}x{cols} (empty)")
            return
        heatmap_width = max(120.0 * s, width)
        heatmap_height = float(np.clip(heatmap_width * rows / max(cols, 1), 80.0 * s, 220.0 * s))
        min_cell_px = max(1.0, self._heatmap_min_cell_pixels * s)
        target_cols = max(1, min(cols, int(heatmap_width / min_cell_px)))
        target_rows = max(1, min(rows, int(heatmap_height / min_cell_px)))
        display_array = self._downsample_heatmap(array, target_rows, target_cols)
        display_rows, display_cols = display_array.shape
        texture_state = self._ensure_array_texture(name, display_cols, display_rows)

        if (
            name in self._array_dirty
            or texture_state["source_shape"] != array.shape
            or texture_state["display_shape"] != display_array.shape
        ):
            rgba, value_min, value_max = self._colorize_heatmap(display_array)
            self._update_array_texture(texture_state["texture_id"], rgba)
            texture_state["source_shape"] = array.shape
            texture_state["display_shape"] = display_array.shape
            texture_state["value_min"] = value_min
            texture_state["value_max"] = value_max
            self._array_dirty.discard(name)

        draw_list = imgui.get_window_draw_list()
        origin = imgui.get_cursor_screen_pos()
        imgui.image(imgui.ImTextureRef(texture_state["texture_id"]), imgui.ImVec2(heatmap_width, heatmap_height))

        border_color = imgui.color_convert_float4_to_u32(imgui.ImVec4(1.0, 1.0, 1.0, 0.25))
        draw_list.add_rect(
            imgui.ImVec2(origin.x, origin.y),
            imgui.ImVec2(origin.x + heatmap_width, origin.y + heatmap_height),
            border_color,
        )
        shape_text = f"shape {rows}x{cols}"
        if (display_rows, display_cols) != (rows, cols):
            shape_text += f"  shown {display_rows}x{display_cols}"
        if np.isfinite(texture_state["value_min"]) and np.isfinite(texture_state["value_max"]):
            range_text = f"min {texture_state['value_min']:.4g}  max {texture_state['value_max']:.4g}"
        else:
            range_text = "min --  max --"
        imgui.text(f"{shape_text}  {range_text}")
