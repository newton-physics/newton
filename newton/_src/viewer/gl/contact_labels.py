# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Small contact numbers anchored in scene depth, using the existing font backend."""

from __future__ import annotations

import ctypes
from typing import TYPE_CHECKING

import numpy as np

from .shaders import ShaderGL, arr_pointer

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence
    from types import ModuleType

    from numpy.typing import ArrayLike

    from newton._src.viewer.gl.opengl import RendererGL


def _layout_labels(
    labels: Sequence[tuple[ArrayLike, str, bool]],
    matrix: np.ndarray,
    size: np.ndarray,
    measure: Callable[[str], tuple[float, float]],
    *,
    scale: float = 1.0,
) -> list[tuple[np.ndarray, str, bool, np.ndarray]]:
    """Place all visible numbers, coalescing only overlapping witnesses of one contact.

    Inputs are cached host witnesses. A screen-space grid bounds overlap queries;
    camera motion never reads depth or contact arrays back from the GPU.
    """
    occupied = {}
    result = []

    def cells(rect: tuple[float, float, float, float]) -> Iterator[tuple[int, int]]:
        x0, y0, x1, y1 = (int(np.floor(value / 8)) for value in rect)
        return ((x, y) for x in range(x0, x1 + 1) for y in range(y0, y1 + 1))

    for witnesses, text, selected in sorted(labels, key=lambda item: not item[2]):
        points = np.asarray(witnesses).reshape(-1, 3)
        projected = np.column_stack((points, np.ones(len(points)))) @ matrix
        valid = np.isfinite(projected).all(axis=1) & (projected[:, 3] > 0)
        valid &= np.all(np.abs(projected[:, :3]) <= projected[:, 3:4], axis=1)
        points, projected = points[valid], projected[valid]
        if not len(points):
            continue
        ndc = projected[:, :3] / projected[:, 3:4]
        screen = (ndc[:, :2] * 0.5 + 0.5) * size
        width, height = measure(text)
        if len(points) == 2 and np.all(np.abs(screen[1] - screen[0]) < (width + 4, height + 4)):
            # Keep the nearer witness so coincident numbers still respect depth.
            keep = int(np.argmin(ndc[:, 2]))
            points, screen = points[keep : keep + 1], screen[keep : keep + 1]
        for point, anchor in zip(points, screen, strict=True):
            best, best_cost = None, float("inf")
            # Never drop a different contact's number, even in a dense cluster.
            # Prefer nearby free space, retaining a leader when displaced.
            for row in range(12):
                for left in (False, True):
                    offset = np.array((-width - 8 if left else 8, 8 + row * (height + 4)), dtype=float)
                    position = np.clip(anchor + offset, (2, 2), np.maximum((2, 2), size - (width + 2, height + 2)))
                    position = np.round(position * scale) / scale
                    rect = (position[0], position[1], position[0] + width, position[1] + height)
                    # Tile occupancy bounds the cost even when thousands of
                    # contacts project to the same tiny screen region.
                    cost = sum(occupied.get(cell, 0) for cell in cells(rect))
                    if cost < best_cost:
                        best, best_cost = rect, cost
                    if cost == 0:
                        break
                if best_cost == 0:
                    break
            for cell in cells(best):
                occupied[cell] = occupied.get(cell, 0) + 1
            result.append((point, text, selected, np.asarray(best[:2]) - anchor))
    return result


class ContactLabels(ShaderGL):
    """Batch font-atlas glyphs anchored to cached contact witnesses in scene space.

    The caller controls label eligibility and supplies immutable label batches.
    Camera, viewport, DPI, or batch identity changes rebuild screen placement;
    stationary frames reuse uploaded geometry. Labels retain witness depth so
    opaque geometry occludes them and later transparent passes attenuate them.
    """

    def __init__(self, gl: ModuleType) -> None:
        """Create digit glyphs, a smooth font sampler, and GL buffers for depth-aware contact labels."""
        import pyglet.font
        from pyglet.graphics.shader import Shader, ShaderProgram

        super().__init__()
        self._gl = gl
        # Rasterize above display resolution so DPI scaling does not magnify
        # individual font texels. Only ten glyphs are needed, lazily on first use.
        self._font_scale = 0.5
        self.font = pyglet.font.load(("Consolas", "DejaVu Sans Mono", "Menlo"), 24, dpi=96)
        result = self.font.get_glyphs("0123456789")
        glyphs = result[0] if isinstance(result, tuple) else result
        self.glyphs = dict(zip("0123456789", glyphs, strict=True))
        self.texture = glyphs[0].owner
        # Pyglet's font atlas defaults to nearest filtering. A dedicated sampler
        # smooths this overlay without changing a shared font texture's state.
        self.sampler = gl.GLuint()
        gl.glGenSamplers(1, self.sampler)
        gl.glSamplerParameteri(self.sampler, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glSamplerParameteri(self.sampler, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        self.shader_program = ShaderProgram(
            Shader(
                """
                #version 330 core
                layout(location=0) in vec3 anchor;
                layout(location=1) in vec2 offset;
                layout(location=2) in vec2 uv;
                layout(location=3) in vec3 color;
                layout(location=4) in float leader;
                uniform mat4 view;
                uniform mat4 projection;
                uniform vec2 pixel;
                out vec2 TexCoord;
                out vec3 Color;
                flat out float Leader;
                void main() {
                    vec4 p = projection * view * vec4(anchor, 1.0);
                    p.xy += offset * pixel * p.w;
                    // Only a small depth bias for coincident surface witnesses.
                    p.z -= 1e-6 * p.w;
                    gl_Position = p;
                    TexCoord = uv;
                    Color = color;
                    Leader = leader;
                }
            """,
                "vertex",
            ),
            Shader(
                """
                #version 330 core
                uniform sampler2D atlas;
                in vec2 TexCoord;
                in vec3 Color;
                flat in float Leader;
                out vec4 FragColor;
                void main() {
                    if (Leader > 0.5) {
                        FragColor = vec4(Color, 0.3);
                        return;
                    }
                    vec2 texel = 1.0 / vec2(textureSize(atlas, 0));
                    float ink = texture(atlas, TexCoord).a;
                    // A light one-texel halo at the oversampled font resolution.
                    // Four taps keep the edge soft without a chunky square rim.
                    float outline = max(ink, texture(atlas, TexCoord + vec2(texel.x, 0)).a);
                    outline = max(outline, texture(atlas, TexCoord - vec2(texel.x, 0)).a);
                    outline = max(outline, texture(atlas, TexCoord + vec2(0, texel.y)).a);
                    outline = max(outline, texture(atlas, TexCoord - vec2(0, texel.y)).a);
                    if (outline < 0.02) discard;
                    FragColor = vec4(mix(vec3(0.025), Color, ink / max(outline, 1e-6)), outline);
                }
            """,
                "fragment",
            ),
        )
        self.locations = {name: self._get_uniform_location(name) for name in ("view", "projection", "pixel", "atlas")}
        self.vao, self.vbo = gl.GLuint(), gl.GLuint()
        gl.glGenVertexArrays(1, self.vao)
        gl.glGenBuffers(1, self.vbo)
        gl.glBindVertexArray(self.vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        for index, (size, offset) in enumerate(((3, 0), (2, 3), (2, 5), (3, 7), (1, 10))):
            gl.glVertexAttribPointer(index, size, gl.GL_FLOAT, False, 44, ctypes.c_void_p(offset * 4))
            gl.glEnableVertexAttribArray(index)
        gl.glBindVertexArray(0)
        self.labels = None
        self.layout_key = None
        self.layout = ()
        self.count = 0

    def render(self, renderer: RendererGL, labels: Sequence[tuple[ArrayLike, str, bool]]) -> None:
        """Draw (witness positions, number text, selected) entries in the scene viewport.

        Replace the label sequence when contents change; its identity is part of
        the layout cache key. Each entry can contain both opposing witnesses,
        coalesced only when their projected numbers would overlap each other.
        """
        gl = self._gl
        gl.glBindVertexArray(self.vao)
        _, _, width, height = renderer._get_scene_viewport()
        scale = getattr(renderer, "_contact_label_scale", 1.0)
        view = np.asarray(renderer._view_matrix).reshape(4, 4)
        projection = np.asarray(renderer._projection_matrix).reshape(4, 4)
        layout_key = (view.tobytes(), projection.tobytes(), width, height, scale)
        if labels is not self.labels or layout_key != self.layout_key:
            self.labels = labels
            self.layout_key = layout_key

            def measure(text: str) -> tuple[float, float]:
                return (
                    sum(self.glyphs[char].advance for char in text) * self._font_scale,
                    (self.font.ascent - self.font.descent) * self._font_scale,
                )

            self.layout = _layout_labels(
                labels, view @ projection, np.array((width, height)) / scale, measure, scale=scale
            )
            vertices = []
            for point, text, selected, offset in self.layout:
                x, y = offset
                color = (1.0, 0.62, 0.22) if selected else (0.84, 0.94, 0.88)
                if not np.allclose(offset, (8, 8)):
                    # A thin leader ties stacked labels to their own witness.
                    end = np.clip(np.zeros(2), offset, offset + measure(text))
                    length = np.linalg.norm(end)
                    if length > 6:
                        start = end * (3 / length)
                        normal = np.array((-end[1], end[0])) * (0.5 / length)
                        corners = (start - normal, end - normal, end + normal, start + normal)
                        for i in (0, 1, 2, 0, 2, 3):
                            vertices.append((*point, *corners[i], 0, 0, *color, 1))
                for char in text:
                    glyph = self.glyphs[char]
                    x0, y0, x1, y1 = (value * self._font_scale for value in glyph.vertices)
                    uv = np.asarray(glyph.tex_coords).reshape(4, 3)[:, :2]
                    corners = ((x + x0, y + y0), (x + x1, y + y0), (x + x1, y + y1), (x + x0, y + y1))
                    for i in (0, 1, 2, 0, 2, 3):
                        vertices.append((*point, *corners[i], *uv[i], *color, 0))
                    x += glyph.advance * self._font_scale
            data = np.asarray(vertices, dtype=np.float32)
            self.count = len(data)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
            gl.glBufferData(gl.GL_ARRAY_BUFFER, data.nbytes, data.ctypes.data, gl.GL_DYNAMIC_DRAW)
        with self:
            gl.glUniformMatrix4fv(self.locations["view"], 1, False, arr_pointer(renderer._view_matrix))
            gl.glUniformMatrix4fv(self.locations["projection"], 1, False, arr_pointer(renderer._projection_matrix))
            gl.glUniform2f(self.locations["pixel"], 2 * scale / max(width, 1), 2 * scale / max(height, 1))
            gl.glActiveTexture(gl.GL_TEXTURE0)
            gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture.id)
            gl.glBindSampler(0, self.sampler)
            gl.glUniform1i(self.locations["atlas"], 0)
            gl.glEnable(gl.GL_DEPTH_TEST)
            gl.glDepthMask(True)
            gl.glDisable(gl.GL_CULL_FACE)
            gl.glEnable(gl.GL_BLEND)
            gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
            gl.glDrawArrays(gl.GL_TRIANGLES, 0, self.count)
            gl.glBindSampler(0, 0)
            gl.glDisable(gl.GL_BLEND)
        gl.glBindVertexArray(0)

    def destroy(self) -> None:
        """Release owned sampler, buffers, and shader while leaving the shared font atlas intact."""
        self._gl.glDeleteSamplers(1, self.sampler)
        self._gl.glDeleteBuffers(1, self.vbo)
        self._gl.glDeleteVertexArrays(1, self.vao)
        self.shader_program.delete()
