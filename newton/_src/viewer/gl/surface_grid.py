# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Filtered metric grids on existing plane and box surfaces."""

from .shaders import ShaderGL, _with_shader_define, arr_pointer, camera_relative_fragment, shape_vertex_shader


class SurfaceGridShader(ShaderGL):
    """Blend grid decades using screen derivatives, without subdividing meshes."""

    def __init__(self, gl):
        from pyglet.graphics.shader import Shader, ShaderProgram

        super().__init__()
        self._gl = gl
        vertex = _with_shader_define(shape_vertex_shader, "ENABLE_TRANSPARENCY")
        vertex = vertex.replace(
            "void main()", "out vec3 GridPosition; flat out vec3 GridNormal; flat out vec3 GridExtent; void main()"
        )
        vertex = vertex.replace(
            "LocalPos = aPos;",
            """
            LocalPos = aPos;
            vec3 scale = vec3(length(transform[0].xyz), length(transform[1].xyz), length(transform[2].xyz));
            GridPosition = aPos * scale;
            // Box dimensions are baked into their corner vertices, not the
            // instance transform. Every corner carries the same half-extents.
            GridExtent = abs(GridPosition);
            GridNormal = abs(aNormal);
        """,
        )
        fragment = (
            """
            #version 330 core
            layout(location=0) out vec4 FragColor;
            layout(location=1) out vec4 Revealage;
            in vec3 GridPosition;
            flat in vec3 GridNormal;
            flat in vec3 GridExtent;
            in float ViewDepth;
            in float Opacity;
            uniform bool oit_enabled;
            uniform bool plane;
            uniform float inv_reference;
        """
            + camera_relative_fragment
            + """
            float lines(vec2 uv, vec2 footprint, float spacing, float half_width) {
                vec2 nearest = abs(fract(uv / spacing + 0.5) - 0.5) * spacing;
                vec2 coverage = 1.0 - smoothstep(vec2(half_width - 0.5), vec2(half_width + 0.5), nearest / footprint);
                // Anisotropic filtering prevents moire at grazing angles.
                coverage *= 1.0 - smoothstep(vec2(0.12), vec2(0.40), footprint / spacing);
                return max(coverage.x, coverage.y);
            }
            void main() {
                // Match the surface's corrected depth; rasterized depth and
                // polygon offset diverge from it on large clipped triangles.
                ReconstructCameraRelativePosition();
                float depth = gl_FragDepth;
                // Allow one pixel's depth slope for resolved MSAA samples,
                // plus float rounding, while retaining foreground occlusion.
                float bias = max(abs(dFdx(depth)), abs(dFdy(depth))) + 2.0e-7;
                gl_FragDepth = max(0.0, depth - bias);
                vec2 uv = GridPosition.xy;
                vec2 extent = GridExtent.xy;
                if (GridNormal.x > 0.5) { uv = GridPosition.yz; extent = GridExtent.yz; }
                else if (GridNormal.y > 0.5) { uv = GridPosition.xz; extent = GridExtent.xz; }
                vec2 footprint = max(fwidth(uv), vec2(1e-9));
                float cell = max(footprint.x, footprint.y) * 36.0;
                float level = log(max(cell, 1e-9)) / log(10.0);
                float spacing = pow(10.0, floor(level));
                float blend = smoothstep(0.0, 1.0, fract(level));
                // Keep a major tier throughout the decade transition. The
                // middle tier gradually becomes minor as the finest fades;
                // the next coarser tier remains bright and wide. Both width
                // and opacity match exactly when the spacing advances.
                float fine = lines(uv, footprint, spacing, 0.65) * (0.18 * (1.0 - blend));
                float middle = lines(uv, footprint, spacing * 10.0, mix(1.15, 0.65, blend))
                    * mix(0.50, 0.18, blend);
                float coarse = lines(uv, footprint, spacing * 100.0, 1.15) * 0.50;
                // Nested lines overlap. Taking their maximum avoids bright
                // intersections and keeps minor/major contrast predictable.
                float alpha = max(fine, max(middle, coarse));
                // Only broad faces need extra detail. Thin rods and small
                // boxes retain the ordinary wire overlay until viewed closely.
                if (!plane) alpha *= smoothstep(100.0, 240.0, min(2.0 * extent.x / footprint.x, 2.0 * extent.y / footprint.y));
                alpha *= gl_FrontFacing ? 1.0 : 0.24;
                alpha *= Opacity;
                if (alpha < 0.002) discard;
                vec3 color = vec3(gl_FrontFacing ? 0.76 : 0.43);
                float d = ViewDepth * inv_reference;
                float weight = alpha * clamp(10.0 / (1e-5 + pow(2.0*d, 2.0) + pow(0.6*d, 6.0)), 1e-2, 3e3);
                FragColor = oit_enabled ? vec4(color * alpha, alpha) * weight : vec4(color, alpha);
                Revealage = vec4(alpha);
            }
        """
        )
        self.shader_program = ShaderProgram(Shader(vertex, "vertex"), Shader(fragment, "fragment"))
        self.locations = {
            name: self._get_uniform_location(name)
            for name in (
                "view",
                "projection",
                "inverse_projection",
                "viewport_size",
                "viewport_origin",
                "view_pos",
                "oit_enabled",
                "plane",
                "inv_reference",
            )
        }

    def render(self, renderer, objects, *, oit):
        gl, loc = self._gl, self.locations
        with self:
            gl.glUniformMatrix4fv(loc["view"], 1, gl.GL_FALSE, arr_pointer(renderer._view_matrix))
            gl.glUniformMatrix4fv(loc["projection"], 1, gl.GL_FALSE, arr_pointer(renderer._projection_matrix))
            gl.glUniformMatrix4fv(
                loc["inverse_projection"],
                1,
                gl.GL_FALSE,
                arr_pointer(renderer._shape_shader._cached_inverse_projection),
            )
            x, y, width, height = renderer._get_scene_viewport()
            gl.glUniform2f(loc["viewport_origin"], x, y)
            gl.glUniform2f(loc["viewport_size"], width, height)
            gl.glUniform3f(loc["view_pos"], *renderer.camera.pos)
            gl.glUniform1i(loc["oit_enabled"], oit)
            gl.glUniform1f(loc["inv_reference"], 1.0 / max(renderer.camera.pivot_distance, 1e-6))
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
            for obj in objects:
                gl.glUniform1i(loc["plane"], obj._surface_grid == 2)
                if not getattr(obj, "_opacity_attribute_enabled", False):
                    gl.glVertexAttrib1f(9, 1.0)
                mesh = getattr(obj, "mesh", obj)
                cull = mesh.backface_culling
                try:
                    mesh.backface_culling = False
                    obj.render()
                finally:
                    mesh.backface_culling = cull

    def destroy(self):
        self.shader_program.delete()
