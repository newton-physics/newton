# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Depth-aware joint annotations, batched separately from the 2D interface."""

from __future__ import annotations

import ctypes
from typing import TYPE_CHECKING

from .shaders import ShaderGL

if TYPE_CHECKING:
    from types import ModuleType

    import numpy as np


_VERTEX = """
#version 330 core
layout(location=0) in vec4 position;
layout(location=1) in vec4 color;
out vec4 Color;
out float ViewDepth;
void main() {
    gl_Position = position;
    Color = color;
    ViewDepth = position.w;
}
"""

_FRAGMENT = """
#version 330 core
layout(location=0) out vec4 FragColor;
layout(location=1) out vec4 Revealage;
in vec4 Color;
in float ViewDepth;
uniform bool oit;
uniform float inv_reference;
void main() {
    if (oit) {
        float depth = ViewDepth * inv_reference;
        float weight = Color.a * clamp(10.0 / (1e-5 + pow(2.0 * depth, 2.0) + pow(0.6 * depth, 6.0)), 1e-2, 3e3);
        FragColor = vec4(Color.rgb * Color.a, Color.a) * weight;
        Revealage = vec4(Color.a);
    } else {
        FragColor = Color;
        Revealage = vec4(Color.a);
    }
}
"""


class JointGlyphShader(ShaderGL):
    def __init__(self, gl: ModuleType) -> None:
        from pyglet.graphics.shader import Shader, ShaderProgram

        super().__init__()
        self._gl = gl
        self.shader_program = ShaderProgram(Shader(_VERTEX, "vertex"), Shader(_FRAGMENT, "fragment"))
        self.oit = self._get_uniform_location("oit")
        self.reference = self._get_uniform_location("inv_reference")
        self.vao, self.vbo = gl.GLuint(), gl.GLuint()
        gl.glGenVertexArrays(1, self.vao)
        gl.glGenBuffers(1, self.vbo)
        gl.glBindVertexArray(self.vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        for attribute in range(2):
            gl.glVertexAttribPointer(attribute, 4, gl.GL_FLOAT, False, 32, ctypes.c_void_p(attribute * 16))
            gl.glEnableVertexAttribArray(attribute)
        gl.glBindVertexArray(0)

    def render(
        self, vertices: np.ndarray, *, oit: bool = False, reference: float = 1.0, depth_equal: bool = False
    ) -> None:
        if not len(vertices):
            return
        gl = self._gl
        with self:
            gl.glUniform1i(self.oit, oit)
            gl.glUniform1f(self.reference, 1.0 / max(reference, 1e-6))
            gl.glEnable(gl.GL_DEPTH_TEST)
            gl.glDepthFunc(gl.GL_LEQUAL if depth_equal else gl.GL_LESS)
            gl.glDisable(gl.GL_CULL_FACE)
            gl.glBindVertexArray(self.vao)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
            gl.glBufferData(gl.GL_ARRAY_BUFFER, vertices.nbytes, vertices.ctypes.data, gl.GL_STREAM_DRAW)
            gl.glDrawArrays(gl.GL_TRIANGLES, 0, len(vertices))
            gl.glBindVertexArray(0)
            gl.glDepthFunc(gl.GL_LESS)

    def destroy(self) -> None:
        self._gl.glDeleteBuffers(1, self.vbo)
        self._gl.glDeleteVertexArrays(1, self.vao)
        self.shader_program.delete()
