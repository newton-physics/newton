# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Lazy, depth-tested selection silhouettes using the existing instance buffers."""

import ctypes

from .shaders import ShaderGL, arr_pointer, capsule_vertex_transform, frame_vertex_shader, str_buffer


class SelectionSurfaceShader(ShaderGL):
    """Tint selected faces and draw their individual edges in the transparency pass."""

    def __init__(self, gl):
        from pyglet.graphics.shader import Shader, ShaderProgram

        super().__init__()
        self._gl = gl
        vertex = """
        #version 330 core
        layout(location=0) in vec3 position;
        uniform mat4 view;
        uniform mat4 projection;
        uniform vec3 view_pos;
        out float vertex_depth;
        void main() {
            vec3 p = mat3(view) * (position - view_pos);
            gl_Position = projection * vec4(p, 1.0);
            vertex_depth = max(-p.z, 0.0);
        }
        """
        geometry = """
        #version 330 core
        layout(triangles) in;
        layout(triangle_strip, max_vertices=3) out;
        in float vertex_depth[];
        out float view_depth;
        noperspective out vec3 edge_distance;
        flat out float spacing;
        uniform vec2 viewport_size;
        void main() {
            vec2 p[3];
            for (int i=0; i<3; ++i)
                p[i] = gl_in[i].gl_Position.xy / max(abs(gl_in[i].gl_Position.w), 1e-9) * viewport_size * 0.5;
            vec3 sides = vec3(length(p[1]-p[2]), length(p[2]-p[0]), length(p[0]-p[1]));
            vec2 a = p[1]-p[0], b = p[2]-p[0];
            float area2 = abs(a.x*b.y - a.y*b.x);
            vec3 heights = area2 / max(sides, vec3(1e-9));
            for (int i=0; i<3; ++i) {
                edge_distance = vec3(0.0);
                edge_distance[i] = heights[i];
                spacing = area2 / max(sides.x+sides.y+sides.z, 1e-9);
                view_depth = vertex_depth[i];
                gl_Position = gl_in[i].gl_Position;
                EmitVertex();
            }
            EndPrimitive();
        }
        """
        fragment = """
        #version 330 core
        layout(location=0) out vec4 FragColor;
        layout(location=1) out vec4 Revealage;
        in float view_depth;
        noperspective in vec3 edge_distance;
        flat in float spacing;
        uniform bool oit_enabled;
        uniform float inv_reference;
        uniform float pixel_scale;
        void main() {
            float distance = min(min(edge_distance.x, edge_distance.y), edge_distance.z) / pixel_scale;
            float edge = 1.0 - smoothstep(0.35, 1.25, distance);
            float detail = smoothstep(0.6, 3.0, spacing / pixel_scale);
            float alpha = mix(0.14, 0.80, edge * detail * min(1.0, 0.10 * spacing / pixel_scale));
            vec3 color = vec3(1.0, 0.43, 0.07);
            // Back faces remain recognizable without hiding front faces or
            // contacts. This is local face orientation, not inter-mesh fading.
            alpha *= gl_FrontFacing ? 1.0 : 0.45;
            color *= gl_FrontFacing ? 1.0 : 0.70;
            if (oit_enabled) {
                float depth = view_depth * inv_reference;
                float weight = alpha * clamp(10.0 / (1e-5 + pow(2.0*depth, 2.0) + pow(0.6*depth, 6.0)), 1e-2, 3e3);
                FragColor = vec4(color * alpha, alpha) * weight;
            } else {
                FragColor = vec4(color, alpha);
            }
            Revealage = vec4(alpha);
        }
        """
        self.shader_program = ShaderProgram(
            Shader(vertex, "vertex"), Shader(geometry, "geometry"), Shader(fragment, "fragment")
        )
        self.locations = {
            name: self._get_uniform_location(name)
            for name in (
                "view",
                "projection",
                "view_pos",
                "viewport_size",
                "oit_enabled",
                "inv_reference",
                "pixel_scale",
            )
        }

    def destroy(self):
        self.shader_program.delete()

    def render(self, renderer, meshes, *, oit):
        gl, loc = self._gl, self.locations
        with self:
            gl.glUniformMatrix4fv(loc["view"], 1, gl.GL_FALSE, arr_pointer(renderer._view_matrix))
            gl.glUniformMatrix4fv(loc["projection"], 1, gl.GL_FALSE, arr_pointer(renderer._projection_matrix))
            gl.glUniform3f(loc["view_pos"], *renderer.camera.pos)
            gl.glUniform2f(loc["viewport_size"], *renderer._get_scene_viewport()[2:])
            gl.glUniform1i(loc["oit_enabled"], oit)
            gl.glUniform1f(loc["inv_reference"], 1.0 / max(renderer.camera.pivot_distance, 1e-6))
            gl.glUniform1f(loc["pixel_scale"], getattr(renderer, "_selection_scale", 1.0))
            gl.glEnable(gl.GL_POLYGON_OFFSET_FILL)
            gl.glPolygonOffset(-1.0, -1.0)
            for mesh in meshes:
                mesh.render()
            gl.glDisable(gl.GL_POLYGON_OFFSET_FILL)


_MASK_VERTEX = (
    """
#version 330 core
"""
    + capsule_vertex_transform
    + """
layout (location = 0) in vec3 position;
layout (location = 3) in vec4 column0;
layout (location = 4) in vec4 column1;
layout (location = 5) in vec4 column2;
layout (location = 6) in vec4 column3;
layout (location = 9) in float opacity;
uniform mat4 view;
uniform mat4 projection;
uniform vec3 view_pos;
out float visible;
void main() {
    mat4 transform = mat4(column0, column1, column2, column3);
    vec3 relative_position = mat3(transform) * mesh_position(position, transform) + transform[3].xyz - view_pos;
    gl_Position = projection * vec4(mat3(view) * relative_position, 1.0);
    visible = opacity;
}
"""
)

_MASK_FRAGMENT = """
#version 330 core
in float visible;
uniform sampler2D scene_depth;
out vec2 mask;
void main() {
    if (visible <= 0.0) discard;
    float depth = texelFetch(scene_depth, ivec2(gl_FragCoord.xy), 0).r;
    float tolerance = 1.5 * max(abs(dFdx(gl_FragCoord.z)), abs(dFdy(gl_FragCoord.z))) + 1e-6;
    // Keep coverage separate from visibility. Resolved MSAA depth and contact
    // lines can occlude individual fragments inside an otherwise solid object;
    // those fragments must not turn into false silhouette edges.
    mask = vec2(1.0, gl_FragCoord.z <= depth + tolerance ? 1.0 - gl_FragCoord.z : 0.0);
}
"""

_OUTLINE_FRAGMENT = """
#version 330 core
uniform sampler2D selection_mask;
uniform float radius;
out vec4 color;
const vec2 directions[8] = vec2[](
    vec2(1, 0), vec2(-1, 0), vec2(0, 1), vec2(0, -1),
    vec2(0.7071, 0.7071), vec2(-0.7071, 0.7071),
    vec2(0.7071, -0.7071), vec2(-0.7071, -0.7071)
);
void main() {
    ivec2 pixel = ivec2(gl_FragCoord.xy);
    if (texelFetch(selection_mask, pixel, 0).r > 0.5) discard;
    ivec2 size = textureSize(selection_mask, 0);
    float nearest = 0.0;
    for (int ring = 1; ring <= 2; ++ring) {
        for (int i = 0; i < 8; ++i) {
            vec2 offset = directions[i] * radius * float(ring) * 0.5;
            ivec2 sample_pixel = clamp(pixel + ivec2(round(offset)), ivec2(0), size - 1);
            nearest = max(nearest, texelFetch(selection_mask, sample_pixel, 0).g);
        }
    }
    if (nearest <= 0.0) discard;
    gl_FragDepth = 1.0 - nearest;
    color = vec4(1.0, 0.43, 0.07, 1.0);
}
"""


class SelectionOutline:
    """Render a union mask of selected shapes, then composite its outside edge."""

    def __init__(self, gl):
        from pyglet.graphics.shader import Shader, ShaderProgram

        self.gl = gl
        self.mask_shader = ShaderProgram(Shader(_MASK_VERTEX, "vertex"), Shader(_MASK_FRAGMENT, "fragment"))
        self.outline_shader = ShaderProgram(
            Shader(frame_vertex_shader, "vertex"), Shader(_OUTLINE_FRAGMENT, "fragment")
        )
        self.uniforms = {
            name: gl.glGetUniformLocation(self.mask_shader.id, str_buffer(name))
            for name in ("view", "projection", "view_pos")
        }
        self.radius = gl.glGetUniformLocation(self.outline_shader.id, str_buffer("radius"))
        self.fbo, self.texture, self.vao = gl.GLuint(), gl.GLuint(), gl.GLuint()
        gl.glGenFramebuffers(1, self.fbo)
        gl.glGenTextures(1, self.texture)
        gl.glGenVertexArrays(1, self.vao)
        self.size = None

    def destroy(self):
        gl = self.gl
        gl.glDeleteFramebuffers(1, self.fbo)
        gl.glDeleteTextures(1, self.texture)
        gl.glDeleteVertexArrays(1, self.vao)
        self.mask_shader.delete()
        self.outline_shader.delete()

    def _resize(self, renderer):
        gl = self.gl
        size = (renderer._screen_width, renderer._screen_height)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.fbo)
        if size != self.size:
            gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, 0)
            gl.glActiveTexture(gl.GL_TEXTURE0)
            gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture)
            # Keep the nearest visible surface depth for the outside edge. It
            # must be occluded and tinted by the same foreground as its object.
            gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RG32F, *size, 0, gl.GL_RG, gl.GL_FLOAT, None)
            for parameter in (gl.GL_TEXTURE_MIN_FILTER, gl.GL_TEXTURE_MAG_FILTER):
                gl.glTexParameteri(gl.GL_TEXTURE_2D, parameter, gl.GL_LINEAR)
            for parameter in (gl.GL_TEXTURE_WRAP_S, gl.GL_TEXTURE_WRAP_T):
                gl.glTexParameteri(gl.GL_TEXTURE_2D, parameter, gl.GL_CLAMP_TO_EDGE)
            gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D, self.texture, 0)
            self.size = size
        if gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER) != gl.GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError("Selection framebuffer is incomplete")

    def render(self, renderer, instances):
        gl = self.gl
        self._resize(renderer)
        gl.glDrawBuffer(gl.GL_COLOR_ATTACHMENT0)
        # Clear the full mask so a resized panel cannot leave old silhouettes
        # beside the viewport. Scene depth is sampled without modifying it.
        gl.glDisable(gl.GL_SCISSOR_TEST)
        gl.glClearBufferfv(gl.GL_COLOR, 0, (gl.GLfloat * 4)(0, 0, 0, 0))
        gl.glEnable(gl.GL_SCISSOR_TEST)
        gl.glScissor(*renderer._get_scene_viewport())
        gl.glViewport(*renderer._get_scene_viewport())
        gl.glDisable(gl.GL_DEPTH_TEST)
        gl.glDepthMask(False)
        # Union both coverage and visibility across every selected shape.
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendEquation(gl.GL_MAX)
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, renderer._frame_depth_texture)
        gl.glUseProgram(self.mask_shader.id)
        gl.glUniformMatrix4fv(self.uniforms["view"], 1, gl.GL_FALSE, arr_pointer(renderer._view_matrix))
        gl.glUniformMatrix4fv(self.uniforms["projection"], 1, gl.GL_FALSE, arr_pointer(renderer._projection_matrix))
        gl.glUniform3f(self.uniforms["view_pos"], *renderer.camera.pos)
        gl.glBindVertexArray(self.vao)
        for obj, index in instances:
            if index == -1:
                # Deformable selection vertices already carry the displayed
                # world/layer transform and use their own compact index buffer.
                gl.glDisable(gl.GL_CULL_FACE)
                gl.glVertexAttrib1f(10, 0.0)
                gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, obj.ebo)
                gl.glBindBuffer(gl.GL_ARRAY_BUFFER, obj.vbo)
                gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, obj.vertex_byte_size, ctypes.c_void_p(0))
                gl.glEnableVertexAttribArray(0)
                for column in range(4):
                    gl.glDisableVertexAttribArray(3 + column)
                    gl.glVertexAttrib4f(3 + column, *(float(row == column) for row in range(4)))
                gl.glDisableVertexAttribArray(9)
                gl.glVertexAttrib1f(9, 1.0)
                gl.glDrawElements(gl.GL_TRIANGLES, obj.num_indices, gl.GL_UNSIGNED_INT, None)
                continue
            if obj.hidden or index >= obj.active_instances or obj.instance_transform_buffer is None:
                continue
            mesh = obj.mesh
            gl.glVertexAttrib1f(10, float(mesh._capsule_instancing))
            (gl.glEnable if mesh.backface_culling else gl.glDisable)(gl.GL_CULL_FACE)
            gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, mesh.ebo)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, mesh.vbo)
            gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, mesh.vertex_byte_size, ctypes.c_void_p(0))
            gl.glEnableVertexAttribArray(0)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, obj.instance_transform_buffer)
            for column in range(4):
                gl.glVertexAttribPointer(
                    3 + column, 4, gl.GL_FLOAT, gl.GL_FALSE, 64, ctypes.c_void_p(index * 64 + column * 16)
                )
                gl.glEnableVertexAttribArray(3 + column)
                gl.glVertexAttribDivisor(3 + column, 1)
            if obj._opacity_attribute_enabled:
                gl.glBindBuffer(gl.GL_ARRAY_BUFFER, obj.instance_opacity_buffer)
                gl.glVertexAttribPointer(9, 1, gl.GL_FLOAT, gl.GL_FALSE, 4, ctypes.c_void_p(index * 4))
                gl.glEnableVertexAttribArray(9)
                gl.glVertexAttribDivisor(9, 1)
            else:
                gl.glDisableVertexAttribArray(9)
                gl.glVertexAttrib1f(9, 1.0)
            gl.glDrawElementsInstanced(gl.GL_TRIANGLES, mesh.num_indices, gl.GL_UNSIGNED_INT, None, 1)
        gl.glBindVertexArray(0)
        gl.glDisable(gl.GL_CULL_FACE)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, renderer._frame_fbo)
        gl.glDrawBuffer(gl.GL_COLOR_ATTACHMENT0)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glDepthMask(True)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendEquation(gl.GL_FUNC_ADD)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glUseProgram(self.outline_shader.id)
        gl.glUniform1f(self.radius, 2.0 * getattr(renderer, "_selection_scale", 1.0))
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture)
        gl.glBindVertexArray(renderer._frame_vao)
        gl.glDrawElements(gl.GL_TRIANGLES, len(renderer._frame_indices), gl.GL_UNSIGNED_INT, None)
        gl.glBindVertexArray(0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
        gl.glDisable(gl.GL_BLEND)
        gl.glDepthMask(True)
        gl.glEnable(gl.GL_DEPTH_TEST)
