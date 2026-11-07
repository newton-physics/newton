# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Submit ImGui draw lists using the same lightweight GL bindings as the scene."""

from __future__ import annotations

import ctypes

# This backend is imported lazily by ImGuiBackend after the viewer creates its
# GL context; importing Newton or headless viewers does not load these modules.
from imgui_bundle import imgui  # noqa: TID253
from imgui_bundle.python_backends.pyglet_backend import PygletProgrammablePipelineRenderer  # noqa: TID253
from pyglet import gl  # noqa: TID253


class ImGuiRenderer(PygletProgrammablePipelineRenderer):
    """Keep upstream input, shaders and font textures; specialize draw submission.

    PyOpenGL's conversions and per-call error polling become expensive with
    clipped property fields and tree columns. Pyglet already provides the
    context's typed GL entry points. Use them only for the per-frame draw path,
    without changing global PyOpenGL settings or the input/texture lifecycle.
    """

    def render(self, draw_data: imgui.ImDrawData) -> None:
        """Draw UI without mutating draw data or leaking GL state into the scene."""
        self._update_textures()
        self._handle_mouse_cursor()
        size, scale, origin = draw_data.display_size, draw_data.framebuffer_scale, draw_data.display_pos
        width, height = int(size.x * scale.x), int(size.y * scale.y)
        if width <= 0 or height <= 0:
            return

        def integer(name: int) -> int:
            """Read scalar GL state without PyOpenGL's array conversion wrappers."""
            value = gl.GLint()
            gl.glGetIntegerv(name, ctypes.byref(value))
            return value.value

        program = integer(gl.GL_CURRENT_PROGRAM)
        vao = integer(gl.GL_VERTEX_ARRAY_BINDING)
        buffer = integer(gl.GL_ARRAY_BUFFER_BINDING)
        active_texture = integer(gl.GL_ACTIVE_TEXTURE)
        gl.glActiveTexture(gl.GL_TEXTURE0)
        texture = integer(gl.GL_TEXTURE_BINDING_2D)
        sampler = integer(gl.GL_SAMPLER_BINDING)
        viewport, scissor, polygon = (gl.GLint * 4)(), (gl.GLint * 4)(), (gl.GLint * 2)()
        gl.glGetIntegerv(gl.GL_VIEWPORT, viewport)
        gl.glGetIntegerv(gl.GL_SCISSOR_BOX, scissor)
        gl.glGetIntegerv(gl.GL_POLYGON_MODE, polygon)
        blend = tuple(
            integer(name)
            for name in (
                gl.GL_BLEND_SRC_RGB,
                gl.GL_BLEND_DST_RGB,
                gl.GL_BLEND_SRC_ALPHA,
                gl.GL_BLEND_DST_ALPHA,
                gl.GL_BLEND_EQUATION_RGB,
                gl.GL_BLEND_EQUATION_ALPHA,
            )
        )
        capabilities = (gl.GL_BLEND, gl.GL_CULL_FACE, gl.GL_DEPTH_TEST, gl.GL_SCISSOR_TEST, gl.GL_PRIMITIVE_RESTART)
        enabled = tuple(gl.glIsEnabled(capability) for capability in capabilities)
        try:
            gl.glEnable(gl.GL_BLEND)
            gl.glBlendEquation(gl.GL_FUNC_ADD)
            gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
            gl.glDisable(gl.GL_CULL_FACE)
            gl.glDisable(gl.GL_DEPTH_TEST)
            gl.glDisable(gl.GL_PRIMITIVE_RESTART)
            gl.glEnable(gl.GL_SCISSOR_TEST)
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
            gl.glViewport(0, 0, width, height)
            gl.glBindSampler(0, 0)
            projection = (gl.GLfloat * 16)(
                2.0 / size.x,
                0,
                0,
                0,
                0,
                -2.0 / size.y,
                0,
                0,
                0,
                0,
                -1,
                0,
                -1 - 2 * origin.x / size.x,
                1 + 2 * origin.y / size.y,
                0,
                1,
            )
            gl.glUseProgram(self._shader_handle)
            gl.glUniform1i(self._attrib_location_tex, 0)
            gl.glUniformMatrix4fv(self._attrib_proj_mtx, 1, gl.GL_FALSE, projection)
            gl.glBindVertexArray(self._vao_handle)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._vbo_handle)
            gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self._elements_handle)
            index_type = gl.GL_UNSIGNED_SHORT if imgui.INDEX_SIZE == 2 else gl.GL_UNSIGNED_INT
            bound_texture = None
            for commands in draw_data.cmd_lists:
                # Upload existing contiguous ImGui storage directly; no Python
                # array conversion, GPU readback, or per-command buffer upload.
                gl.glBufferData(
                    gl.GL_ARRAY_BUFFER,
                    commands.vtx_buffer.size() * imgui.VERTEX_SIZE,
                    ctypes.c_void_p(commands.vtx_buffer.data_address()),
                    gl.GL_STREAM_DRAW,
                )
                gl.glBufferData(
                    gl.GL_ELEMENT_ARRAY_BUFFER,
                    commands.idx_buffer.size() * imgui.INDEX_SIZE,
                    ctypes.c_void_p(commands.idx_buffer.data_address()),
                    gl.GL_STREAM_DRAW,
                )
                for command in commands.cmd_buffer:
                    if not command.elem_count:
                        continue
                    rect = command.clip_rect
                    x0, y0 = max(0, (rect.x - origin.x) * scale.x), max(0, (rect.y - origin.y) * scale.y)
                    x1, y1 = min(width, (rect.z - origin.x) * scale.x), min(height, (rect.w - origin.y) * scale.y)
                    if x1 <= x0 or y1 <= y0:
                        continue
                    current_texture = command.tex_ref.get_tex_id()
                    if current_texture != bound_texture:
                        gl.glBindTexture(gl.GL_TEXTURE_2D, current_texture)
                        bound_texture = current_texture
                    gl.glScissor(int(x0), int(height - y1), int(x1 - x0), int(y1 - y0))
                    gl.glDrawElementsBaseVertex(
                        gl.GL_TRIANGLES,
                        command.elem_count,
                        index_type,
                        ctypes.c_void_p(command.idx_offset * imgui.INDEX_SIZE),
                        command.vtx_offset,
                    )
        finally:
            gl.glUseProgram(program)
            gl.glBindVertexArray(vao)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, buffer)
            gl.glBindTexture(gl.GL_TEXTURE_2D, texture)
            gl.glBindSampler(0, sampler)
            gl.glActiveTexture(active_texture)
            gl.glBlendFuncSeparate(*blend[:4])
            gl.glBlendEquationSeparate(*blend[4:])
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, polygon[0])
            for capability, was_enabled in zip(capabilities, enabled, strict=True):
                (gl.glEnable if was_enabled else gl.glDisable)(capability)
            gl.glViewport(*viewport)
            gl.glScissor(*scissor)
