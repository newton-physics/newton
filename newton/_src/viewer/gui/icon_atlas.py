# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Share one mipmapped image for static GUI icons."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .icons import ICON_COLUMNS, ICON_PIXELS, ICON_SPRITES, ICON_TILE

if TYPE_CHECKING:
    from types import ModuleType

    from imgui_bundle import imgui
    from pyglet.image import Texture


class IconAtlas:
    """Draw static icons as tinted quads instead of rebuilding vector geometry.

    The shipped PNG preserves the vector artwork's grayscale shading and alpha.
    Mipmaps filter it at each DPI size. All widgets share one lazily uploaded GL
    texture, released with the ImGui backend. Unusual sizes use the vector source.
    """

    def __init__(self, ui: ModuleType) -> None:
        """Bind the GUI context without loading pixels or allocating a texture."""
        self.ui = ui
        self._texture: Texture | None = None
        self._reference: imgui.ImTextureRef | None = None
        self._uvs: dict[tuple[str, int, bool], tuple[imgui.ImVec2, imgui.ImVec2]] = {}

    def _load(self) -> None:
        """Upload the shared image and mipmaps once, preserving the texture binding."""
        # These optional imports need the window's current GL context. Pyglet's
        # PNG decoder avoids adding an image-processing dependency.
        from pyglet import gl, image

        previous = gl.GLint()
        gl.glGetIntegerv(gl.GL_TEXTURE_BINDING_2D, previous)
        try:
            texture = image.load(str(Path(__file__).with_name("icons.png"))).get_texture()
            self._texture = texture
            gl.glBindTexture(texture.target, texture.id)
            gl.glTexParameteri(texture.target, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR_MIPMAP_LINEAR)
            gl.glTexParameteri(texture.target, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
            gl.glTexParameteri(texture.target, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
            gl.glTexParameteri(texture.target, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
            gl.glGenerateMipmap(texture.target)
            self._reference = self.ui.ImTextureRef(texture.id)
            padding = (ICON_TILE - ICON_PIXELS) // 2
            for index, sprite in enumerate(ICON_SPRITES):
                x, y = index % ICON_COLUMNS * ICON_TILE + padding, index // ICON_COLUMNS * ICON_TILE + padding
                # PNG sprite rows run top-to-bottom; GL image uploads run bottom-to-top.
                self._uvs[sprite] = (
                    self.ui.ImVec2(x / texture.width, 1 - y / texture.height),
                    self.ui.ImVec2((x + ICON_PIXELS) / texture.width, 1 - (y + ICON_PIXELS) / texture.height),
                )
        finally:
            gl.glBindTexture(gl.GL_TEXTURE_2D, previous.value)

    def draw(
        self,
        draw: imgui.ImDrawList,
        name: str,
        position: imgui.ImVec2 | tuple[float, float],
        size: float,
        *,
        scale: float,
        color: int,
        solid: bool = False,
    ) -> bool:
        """Emit one textured quad, or return False for an unsupported size/name."""
        if self._texture is None:
            self._load()
        uvs = self._uvs.get((name, round(size / scale, 3), solid))
        if uvs is None or size < 1 or size > ICON_PIXELS:
            return False
        px, py = position
        draw.add_image(self._reference, self.ui.ImVec2(px, py), self.ui.ImVec2(px + size, py + size), *uvs, color)
        return True

    def close(self) -> None:
        """Release the owned texture while its GL context is still current."""
        if self._texture is not None:
            self._texture.delete()
            self._texture = None
            self._reference = None
            self._uvs.clear()
