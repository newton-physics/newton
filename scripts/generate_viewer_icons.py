# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Bake the GUI icon sources: uv run --extra dev scripts/generate_viewer_icons.py.

Uses an offscreen ViewerGL context and the existing ImGui rasterizer. No image
editor, extra font, or network dependency is needed. Regenerate after editing
viewer/gui/icons.py; the PNG is shipped with the package for fast startup.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np
import warp as wp

from newton._src.viewer.gl.opengl import RendererGL
from newton._src.viewer.gui.icons import ICON_COLUMNS, ICON_PIXELS, ICON_SPRITES, ICON_TILE, draw_icon
from newton._src.viewer.gui.imgui_backend import ImGuiBackend


def main() -> None:
    """Render grayscale artwork and recover straight alpha from two backgrounds."""
    width = ICON_COLUMNS * ICON_TILE
    height = (len(ICON_SPRITES) + ICON_COLUMNS - 1) // ICON_COLUMNS * ICON_TILE
    with wp.ScopedDevice("cpu"):
        renderer = RendererGL(screen_width=width, screen_height=height, headless=True)
    backend = ImGuiBackend(renderer.window)
    ui, gl = backend.imgui, renderer.gl
    framebuffer, texture = gl.GLuint(), gl.GLuint()
    try:
        backend.io.display_size = (width, height)
        backend.io.display_framebuffer_scale = (1, 1)
        backend.begin_frame()
        draw = ui.get_foreground_draw_list()
        # Geometric AA is independent of the font atlas's baked line widths.
        draw.flags &= ~ui.ImDrawListFlags_.anti_aliased_lines_use_tex
        padding = (ICON_TILE - ICON_PIXELS) // 2
        for index, (name, size, solid) in enumerate(ICON_SPRITES):
            position = index % ICON_COLUMNS * ICON_TILE + padding, index // ICON_COLUMNS * ICON_TILE + padding
            draw_icon(ui, draw, name, position, ICON_PIXELS, scale=ICON_PIXELS / size, rgba=(1, 1, 1, 1), solid=solid)
        backend.end_frame()
        gl.glGenTextures(1, texture)
        gl.glBindTexture(gl.GL_TEXTURE_2D, texture)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA8, width, height, 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, None)
        gl.glGenFramebuffers(1, framebuffer)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, framebuffer)
        gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D, texture, 0)
        if gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER) != gl.GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError("Icon atlas framebuffer is incomplete")
        images = []
        for background in (0, 1):
            gl.glDisable(gl.GL_SCISSOR_TEST)
            gl.glClearColor(background, background, background, 1)
            gl.glClear(gl.GL_COLOR_BUFFER_BIT)
            backend.render()
            pixels = np.empty((height, width, 4), dtype=np.uint8)
            gl.glReadPixels(
                0, 0, width, height, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, pixels.ctypes.data_as(ctypes.c_void_p)
            )
            images.append(pixels[::-1].astype(np.float32))
        # Two opaque backgrounds recover coverage correctly even where shaded
        # faces overlap antialiased strokes, without depending on alpha blending.
        alpha = np.clip(255 + images[0][..., 0] - images[1][..., 0], 0, 255)
        rgb = np.ones((height, width, 3), dtype=np.float32) * 255
        np.divide(images[0][..., :3] * 255, alpha[..., None], out=rgb, where=alpha[..., None] > 0)
        pixels = np.round(np.clip(np.dstack((rgb, alpha)), 0, 255)).astype(np.uint8)
        from pyglet.image import ImageData

        path = Path(__file__).resolve().parents[1] / "newton/_src/viewer/gui/icons.png"
        ImageData(width, height, "RGBA", pixels.tobytes(), pitch=-width * 4).save(str(path))
        print(f"Wrote {len(ICON_SPRITES)} sprites to {path.name} ({width} x {height})")
    finally:
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        gl.glDeleteFramebuffers(1, framebuffer)
        gl.glDeleteTextures(1, texture)
        backend.shutdown()
        renderer.window.close()
        ui.destroy_context()


if __name__ == "__main__":
    main()
