# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Camera-adaptive collision surfaces and visual shells for physics inspection.

RendererGL owns framebuffer, blending, and depth-mask setup. These shaders share
its opaque depth and transparency passes: diagnostics write depth before faded
surfaces composite over them, so X-ray reveals contacts without drawing through
opaque foreground geometry. Surface and wire shading remain separate to avoid
paying for a geometry shader when only filled surfaces are needed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .shaders import (
    ShaderGL,
    ShaderShape,
    _with_shader_define,
    arr_pointer,
    shape_vertex_shader,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from types import ModuleType

    from newton._src.viewer.gl.opengl import MeshGL, MeshInstancerGL, RendererGL


def _proximity_vertex(source: str) -> str:
    """Add instance-box camera clearance to the shared shape vertex shader.

    The resulting proximity is constant across a rigid instance, avoiding a
    patchwork of opaque and transparent triangles. SdfMeshLOD's pass culling
    mirrors this calculation and must stay conservative if the formula changes.
    """
    # Evaluate once per vertex from the existing instance transform. No per-frame
    # host transforms, camera-dependent batches or extra instance buffers.
    source = source.replace(
        "void main()",
        """
        flat out float Proximity;
        uniform vec3 bounds_center;
        uniform vec3 bounds_half;
        uniform float proximity_range;
        uniform float inspection_radius;
        void main()
        """,
    )
    return source.replace(
        "LocalPos = aPos;",
        """
        LocalPos = aPos;
        vec3 scale = max(vec3(length(transform[0].xyz), length(transform[1].xyz),
                             length(transform[2].xyz)), vec3(1e-12));
        vec3 delta = view_pos - transform[3].xyz - mat3(transform) * bounds_center;
        vec3 local_camera = vec3(dot(delta, transform[0].xyz) / scale.x,
                                 dot(delta, transform[1].xyz) / scale.y,
                                 dot(delta, transform[2].xyz) / scale.z);
        vec3 extent = bounds_half * scale;
        if (aCapsule > 0.5) extent.z = scale.z;
        float clearance = length(max(abs(local_camera) - extent, vec3(0.0)));
        float radius = max(max(length(extent), inspection_radius), 1e-9);
        // Scale independent, including nonuniform/mirrored instances. Surface
        // clearance handles broad ramps better than distance to their origin.
        // A shared inspection scale keeps small parts as transparent as their
        // neighbors. Do not use orbit distance: that cancels dollying.
        float reach = radius * projection[1][1] * max(proximity_range, 1e-6);
        Proximity = proximity_range > 0.0 ? 1.0 - smoothstep(1.0, 5.0, clearance / reach) : 0.0;
        """,
    )


def _mesh_metrics(mesh: MeshGL) -> tuple[np.ndarray, np.ndarray]:
    if not hasattr(mesh, "_physics_metrics"):
        points = mesh._points.numpy() if mesh._points is not None else np.empty((0, 3))
        lower, upper = (points.min(axis=0), points.max(axis=0)) if len(points) else (-np.ones(3), np.ones(3))
        # One readback per shared mesh on first X-ray use; never per world/frame.
        mesh._physics_metrics = ((upper + lower) * 0.5, (upper - lower) * 0.5)
    return mesh._physics_metrics


def _render_inspection_object(obj: MeshGL | MeshInstancerGL, pass_draws: str) -> None:
    """Use prepared instance ranges when available, preserving the ordinary draws."""
    draws = getattr(obj, pass_draws, None)
    if draws is None:
        obj.render()
    elif draws:
        original = obj._lod_draws
        try:
            obj._lod_draws = draws
            obj.render()
        finally:
            obj._lod_draws = original


class PhysicsSurfaceShader(ShaderShape):
    """Split colliders between opaque and faded passes while retaining shadow coverage.

    Proximity partitions collision instances between passes; visual shells and
    deformable surfaces use their own opacity rules. Shadow sampling reuses the
    regular shape shader. X-ray opacity is a display aid and does not redefine
    an object's physical material or the shadow caster geometry.
    """

    @staticmethod
    def _shader_sources(enable_transparency: bool) -> tuple[str, str]:
        """Extend the shared surface shaders with proximity opacity, pass filtering, and X-ray shadow coverage."""
        vertex, fragment = ShaderShape._shader_sources(enable_transparency)
        vertex = _proximity_vertex(vertex)
        fragment = (
            fragment.replace(
                "void main()",
                """
            flat in float Proximity;
            uniform bool opaque_pass;
            uniform bool visual_pass;
            uniform bool adaptive;
            uniform float collision_opacity;
            uniform float shell_opacity;
            uniform float shadow_coverage;
            uniform bool deformable;
            uniform float deformable_opacity;
            void main()
            """,
            )
            .replace(
                "    // This reconstruction also corrects depth before the fragment is committed.",
                """
            if (Opacity <= 0.0 || (!visual_pass && (opaque_pass ? Proximity > 0.0 : Proximity <= 0.0))) discard;
            // This reconstruction also corrects depth before the fragment is committed.
            """,
            )
            .replace(
                "float alpha = clamp(Opacity, 0.0, 1.0);",
                """
            float rim = 1.0 - abs(dot(N, V));
            float alpha = mix(1.0, collision_opacity * mix(0.65, 1.5, rim * rim), Proximity);
            if (visual_pass) {
                float depth = ViewDepth * oit_inv_depth_reference;
                float fade = adaptive ? mix(1.0, 0.30, smoothstep(1.5, 7.0, depth)) : 1.0;
                alpha = shell_opacity * Opacity * fade * mix(0.45, 1.25, rim * rim);
                alpha *= mix(1.0, 0.45, Proximity);
                vec3 base = mix(ObjectColor, vec3(0.62, 0.75, 0.83), 0.70);
                color = base * (0.65 + 0.35 * abs(dot(N, normalize(vec3(0.3, -0.4, 0.85)))));
                color *= 1.0 - 0.45 * shadow * NdotL;
            }
            if (deformable) {
                // Cloth and volume boundaries are the physical surface, not a
                // separate visual skin. Retain their material identity while
                // revealing folds, cells and contacts on the other side.
                float depth = ViewDepth * oit_inv_depth_reference;
                float fade = adaptive ? mix(1.0, 0.5, smoothstep(2.0, 8.0, depth)) : 1.0;
                alpha = deformable_opacity * Opacity * fade * mix(0.70, 1.20, rim * rim);
                color = mix(ObjectColor, vec3(0.55, 0.75, 0.78), 0.25);
                color *= 0.72 + 0.28 * abs(dot(N, normalize(vec3(0.3, -0.4, 0.85))));
                color *= 1.0 - 0.35 * shadow * NdotL;
            }
            // X-ray is an inspection aid, not transparent physical material.
            // Preserve the shadow's coverage as the receiver fades: composite
            // a dark shadow under its surface color in this same depth-tested
            // pass. The common receiver-plane PCF keeps the contact attached.
            // Light-facing weighting avoids darkening every back surface.
            // Strong coverage is useful on static support surfaces. On moving
            // assemblies it stacks across the visual skin and its collision
            // proxies, hiding the very mechanism X-ray should reveal.
            float coverage = alpha + (1.0 - alpha) * shadow_coverage * shadow * NdotL;
            color *= alpha / max(coverage, 1e-6);
            alpha = coverage;
            """,
            )
        )
        return vertex, fragment

    def __init__(self, gl: ModuleType) -> None:
        """Compile the transparency-capable surface shader and cache X-ray uniform locations."""
        super().__init__(gl, enable_transparency=True)
        self.locations = {
            name: self._get_uniform_location(name)
            for name in (
                "bounds_center",
                "bounds_half",
                "proximity_range",
                "inspection_radius",
                "opaque_pass",
                "collision_opacity",
                "visual_pass",
                "adaptive",
                "shell_opacity",
                "shadow_coverage",
                "deformable",
                "deformable_opacity",
            )
        }

    def render(
        self,
        renderer: RendererGL,
        objects: Sequence[tuple[str, MeshGL | MeshInstancerGL]],
        *,
        opaque: bool,
        oit: bool = False,
        visual: bool = False,
    ) -> None:
        """Draw visual shells or collision surfaces with X-ray settings and per-object shadow and LOD selection."""
        gl, loc = self._gl, self.locations
        renderer._update_shape_shader(self)
        self.set_oit_enabled(oit)
        with self:
            gl.glUniform1i(loc["opaque_pass"], opaque)
            gl.glUniform1i(loc["visual_pass"], visual)
            gl.glUniform1i(loc["adaptive"], renderer._xray_adaptive)
            gl.glUniform1f(loc["shell_opacity"], renderer._xray_opacity)
            gl.glUniform1f(loc["collision_opacity"], renderer._xray_collision_opacity)
            gl.glUniform1f(loc["proximity_range"], renderer._xray_proximity_range)
            gl.glUniform1f(loc["inspection_radius"], renderer._xray_inspection_radius)
            gl.glUniform1f(loc["deformable_opacity"], getattr(renderer, "_xray_deformable_opacity", 0.28))
            for _, obj in objects:
                deformable = getattr(obj, "_physics_role", None) == "deformable"
                gl.glUniform1i(loc["deformable"], deformable)
                coverage = (
                    0.22
                    if deformable
                    else 0.65
                    if getattr(obj, "_physics_static", False)
                    else (0.0 if visual else 0.08)
                )
                gl.glUniform1f(loc["shadow_coverage"], coverage)
                # Deformed points already live in display space and change
                # every frame. No readback or stale rest-pose bounding box.
                center, half = ((0, 0, 0), (0, 0, 0)) if deformable else _mesh_metrics(getattr(obj, "mesh", obj))
                gl.glUniform3f(loc["bounds_center"], *center)
                gl.glUniform3f(loc["bounds_half"], *half)
                if not getattr(obj, "_opacity_attribute_enabled", False):
                    gl.glVertexAttrib1f(9, 1.0)
                if visual:
                    obj.render()
                else:
                    _render_inspection_object(obj, "_lod_opaque_draws" if opaque else "_lod_transparent_draws")


class PhysicsContactShader(ShaderGL):
    """Filled loaded contacts and hollow candidate rings, bounded in screen size."""

    def __init__(self, gl: ModuleType) -> None:
        """Compile analytic contact discs with bounded pixel size and selected-contact rings."""
        from pyglet.graphics.shader import Shader, ShaderProgram

        super().__init__()
        self._gl = gl
        self._quad = None
        vertex = """
        #version 330 core
        layout(location=0) in vec3 aPos;
        layout(location=3) in vec4 t0;
        layout(location=4) in vec4 t1;
        layout(location=5) in vec4 t2;
        layout(location=6) in vec4 t3;
        layout(location=7) in vec3 aColor;
        uniform mat4 view;
        uniform mat4 projection;
        uniform vec3 view_pos;
        uniform float viewport_height;
        uniform float disk_scale;
        uniform ivec2 selected_instances;
        out vec2 Disk;
        out vec3 Color;
        flat out vec3 Normal;
        flat out float Ring;
        void main() {
            vec3 center = mat3(view) * (t3.xyz - view_pos);
            float pixels = length(t0.xyz) * projection[1][1] * viewport_height * 0.5 / max(-center.z, 1e-9);
            float factor = clamp(pixels * 0.8, 3.0, 14.0) / max(pixels, 1e-9);
            bool selected = gl_InstanceID >= selected_instances.x
                && gl_InstanceID < selected_instances.x + selected_instances.y;
            Ring = selected ? 2.0 / max(pixels * factor * disk_scale, 1.0) : 0.0;
            // Use the viewer-facing cap of the regular thin mode cylinder.
            // Keeping its thickness avoids a coplanar disk flickering against
            // the ground, without bypassing depth or foreground opacity.
            vec3 cap = t2.xyz * (dot(t2.xyz, view_pos - t3.xyz) >= 0.0 ? 1.0 : -1.0);
            vec3 offset = (mat3(t0.xyz, t1.xyz, t2.xyz) * aPos * disk_scale * (1.0 + Ring) + cap) * factor;
            gl_Position = projection * vec4(center + mat3(view) * offset, 1.0);
            Disk = aPos.xy * (1.0 + Ring);
            Color = aColor;
            Normal = normalize(cross(t0.xyz, t1.xyz) + vec3(0.0, 0.0, 1e-20));
        }
        """
        fragment = """
        #version 330 core
        in vec2 Disk;
        in vec3 Color;
        flat in vec3 Normal;
        flat in float Ring;
        uniform vec3 sun_direction;
        out vec4 FragColor;
        void main() {
            float r = length(Disk);
            float aa = max(fwidth(r), 0.01);
            float alpha = 1.0 - smoothstep(1.0+Ring-aa, 1.0+Ring+aa, r);
            if (alpha < 0.01) discard;
            bool open = distance(Color, OPEN_COLOR) < 0.001;
            bool unknown = distance(Color, UNKNOWN_COLOR) < 0.001;
            if (open || unknown) {
                // Discard the centre, including its depth: an unloaded
                // candidate must not hide the surface or another contact.
                alpha *= smoothstep(0.72-aa, 0.72+aa, r);
                if (unknown && r < 1.0) {
                    float dash = sin(atan(Disk.y, Disk.x) * 6.0);
                    alpha *= smoothstep(-0.15, 0.15, dash);
                }
                if (r < 1.0) alpha *= 0.75;
                if (alpha < 0.01) discard;
            }
            vec3 color = Color * (0.78 + 0.22 * abs(dot(Normal, sun_direction)));
            if (!open && !unknown) {
                color = mix(color, vec3(0.04), smoothstep(0.84-aa, 0.90+aa, r));
                color = mix(color, vec3(0.65), smoothstep(0.94-aa, 1.0, r));
            }
            if (Ring > 0.0) color = mix(color, vec3(1.0, 0.45, 0.08), smoothstep(1.0-aa, 1.0+aa, r));
            FragColor = vec4(color, alpha);
        }
        """
        from ..kernels import _CONTACT_MODE_COLORS  # noqa: PLC0415

        for token, mode in (("OPEN_COLOR", "open"), ("UNKNOWN_COLOR", "unknown")):
            fragment = fragment.replace(token, "vec3(" + ", ".join(map(str, _CONTACT_MODE_COLORS[mode])) + ")")
        self.shader_program = ShaderProgram(Shader(vertex, "vertex"), Shader(fragment, "fragment"))
        self.locations = {
            name: self._get_uniform_location(name)
            for name in (
                "view",
                "projection",
                "view_pos",
                "viewport_height",
                "sun_direction",
                "disk_scale",
                "selected_instances",
            )
        }

    def render(self, renderer: RendererGL, objects: Mapping[str, MeshInstancerGL]) -> None:
        """Draw nonempty contact batches using a lazily allocated shared quad.

        Temporarily substitute the quad for each contact mesh and restore its
        original buffers afterward. The shader retains scene depth while
        sizing discs and selection rings in screen pixels.
        """
        gl, loc = self._gl, self.locations
        if self._quad is None:
            import warp as wp  # noqa: PLC0415

            from .opengl import MeshGL  # noqa: PLC0415

            device = next(iter(objects.values())).device
            self._quad = MeshGL(4, 6, device, backface_culling=False)
            self._quad.update(
                wp.array([(-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)], dtype=wp.vec3, device=device),
                wp.array([0, 1, 2, 0, 2, 3], dtype=int, device=device),
                wp.array([(0, 0, 1)] * 4, dtype=wp.vec3, device=device),
                None,
            )
        with self:
            gl.glUniformMatrix4fv(loc["view"], 1, gl.GL_FALSE, arr_pointer(renderer._view_matrix))
            gl.glUniformMatrix4fv(loc["projection"], 1, gl.GL_FALSE, arr_pointer(renderer._projection_matrix))
            gl.glUniform3f(loc["view_pos"], *renderer.camera.pos)
            gl.glUniform1f(loc["viewport_height"], renderer._get_scene_viewport()[3])
            gl.glUniform1f(loc["disk_scale"], renderer.contact_disk_scale)
            gl.glUniform3f(loc["sun_direction"], *renderer._sun_direction)
            gl.glEnable(gl.GL_BLEND)
            gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
            selection = getattr(renderer, "_contact_selection", None)
            for name, obj in objects.items():
                selected = selection[1:] if selection is not None and selection[0] == name else (-1, 0)
                gl.glUniform2i(loc["selected_instances"], *selected)
                # The fragment shader draws the disk analytically; a cylinder
                # wastes 128 triangles for each of its two witness points.
                original = obj.mesh
                try:
                    obj.mesh = self._quad
                    obj._bind_mesh_buffers()
                    obj.render()
                finally:
                    obj.mesh = original
                    obj._bind_mesh_buffers()
            gl.glDisable(gl.GL_BLEND)

    def destroy(self) -> None:
        """Release the shared contact quad, if allocated, and the shader program."""
        if self._quad is not None:
            self._quad.destroy()
        self.shader_program.delete()


_fragment = """
#version 330 core
layout(location=0) out vec4 FragColor;
layout(location=1) out vec4 Revealage;
in vec3 ObjectColor;
in float Opacity;
in float ViewDepth;
flat in float EdgePixels;
flat in float Proximity;
uniform float shell_opacity;
uniform float collision_opacity;
uniform float inv_reference;
uniform bool adaptive;
uniform bool oit_enabled;
uniform bool collision;
uniform bool wire_only;
uniform bool deformable;
uniform float deformable_opacity;
void main() {
    if (Opacity <= 0.0) discard;
    float depth = ViewDepth * inv_reference;
    float fade = adaptive ? mix(1.0, 0.30, smoothstep(1.5, 7.0, depth)) : 1.0;
    // Triangle inradius measures the *narrow* spacing of projected edges,
    // including skinny triangles and converging pole fans. Line coverage
    // grows inversely with this spacing, so bound its optical contribution
    // before gradually revealing details that have enough pixels to resolve.
    float detail = adaptive ? smoothstep(1.0, 4.0, EdgePixels) : 1.0;
    float budget = adaptive ? 0.016 * EdgePixels : 1.0;
    float alpha = min(min(0.22, shell_opacity * 1.2), budget) * Opacity * fade * detail;
    vec3 color = vec3(0.46, 0.65, 0.73);
    alpha *= mix(1.0, 0.45, Proximity);
    if (collision) {
        // Warm, restrained collision edges distinguish them from visual skin
        // and the brighter contact normals. No wire clutter in distant worlds.
        alpha = min(min(0.22, collision_opacity * 1.5), budget) * Opacity * Proximity * detail;
        color = mix(ObjectColor, vec3(0.90, 0.73, 0.43), 0.70);
    }
    if (wire_only) {
        detail = adaptive ? smoothstep(0.15, 1.0, EdgePixels) : 1.0;
        alpha = min(0.65, adaptive ? 0.045 * EdgePixels : 0.65) * Opacity * detail;
        color = mix(ObjectColor, vec3(0.68, 0.80, 0.86), 0.75);
    }
    if (deformable && !wire_only) {
        alpha = min(min(0.35, deformable_opacity * 1.5), budget) * Opacity * fade * detail;
        color = mix(ObjectColor, vec3(0.72, 0.84, 0.88), 0.65);
    }
    // Use the triangle winding, like face culling, rather than authored
    // shading normals or the depth of other meshes in the scene.
    alpha *= gl_FrontFacing ? 1.0 : 0.40;
    color *= gl_FrontFacing ? 1.15 : 0.65;
    if (alpha < 0.003) discard;
    if (oit_enabled) {
        float weight = alpha * clamp(10.0 / (1e-5 + pow(2.0 * depth, 2.0) + pow(0.6 * depth, 6.0)), 1e-2, 3e3);
        FragColor = vec4(color * alpha, alpha) * weight;
        Revealage = vec4(alpha);
    } else {
        FragColor = vec4(color, alpha);
        Revealage = vec4(alpha);
    }
}
"""


_wire_geometry = """
#version 330 core
layout(triangles) in;
layout(triangle_strip, max_vertices=3) out;
in vec3 VertexObjectColor[];
in float VertexOpacity[];
in float VertexViewDepth[];
flat in float VertexProximity[];
out vec3 ObjectColor;
out float Opacity;
out float ViewDepth;
flat out float Proximity;
flat out float EdgePixels;
uniform vec2 viewport_size;
uniform bool adaptive;
uniform bool collision;
uniform bool wire_only;
void main() {
    if (collision && VertexProximity[0] <= 0.0) return;
    vec2 p[3];
    for (int i = 0; i < 3; ++i)
        p[i] = gl_in[i].gl_Position.xy / max(abs(gl_in[i].gl_Position.w), 1e-9) * viewport_size * 0.5;
    vec2 a = p[1] - p[0], b = p[2] - p[0];
    float perimeter = length(a) + length(b) + length(p[2] - p[1]);
    float spacing = abs(a.x*b.y - a.y*b.x) / max(perimeter, 1e-9);
    // Cull subpixel wire triangles before rasterization. No geometry copies,
    // mesh readbacks or CPU work as the camera and individual instances move.
    if (adaptive && spacing <= (wire_only ? 0.15 : 1.0)) return;
    for (int i = 0; i < 3; ++i) {
        ObjectColor = VertexObjectColor[i];
        Opacity = VertexOpacity[i];
        ViewDepth = VertexViewDepth[i];
        Proximity = VertexProximity[i];
        EdgePixels = spacing;
        gl_Position = gl_in[i].gl_Position;
        EmitVertex();
    }
    EndPrimitive();
}
"""


class PhysicsShellShader(ShaderGL):
    """Draw wires with projected-density filtering and face-winding contrast.

    Triangle inradius in pixels limits wire opacity so small or densely meshed
    objects do not become solid-looking. Back-facing wires use lower contrast
    based on winding, not other meshes' depths. Filled shells use the cheaper
    PhysicsSurfaceShader path without the wire geometry stage.
    """

    def __init__(self, gl: ModuleType) -> None:
        """Compile the wire shader with projected triangle spacing and proximity-based opacity."""
        import re  # noqa: PLC0415

        from pyglet.graphics.shader import Shader, ShaderProgram

        super().__init__()
        self._gl = gl
        vertex = _proximity_vertex(_with_shader_define(shape_vertex_shader, "ENABLE_TRANSPARENCY"))
        # Filled surfaces use the ordinary indexed vertex/fragment path. Only
        # wire overlays need the projected triangle spacing geometry stage.
        for name in ("ObjectColor", "Opacity", "ViewDepth", "Proximity"):
            vertex = re.sub(r"\b" + name + r"\b", "Vertex" + name, vertex)
        self.shader_program = ShaderProgram(
            Shader(vertex, "vertex"), Shader(_wire_geometry, "geometry"), Shader(_fragment, "fragment")
        )
        self.locations = {
            name: self._get_uniform_location(name)
            for name in (
                "view",
                "projection",
                "view_pos",
                "viewport_size",
                "shell_opacity",
                "collision_opacity",
                "inv_reference",
                "adaptive",
                "oit_enabled",
                "collision",
                "bounds_center",
                "bounds_half",
                "proximity_range",
                "inspection_radius",
                "wire_only",
                "deformable",
                "deformable_opacity",
            )
        }

    def render(
        self,
        renderer: RendererGL,
        objects: Sequence[tuple[str, MeshGL | MeshInstancerGL]],
        *,
        oit: bool,
        collision: bool = False,
        wire_only: bool = False,
    ) -> None:
        """Submit a wire pass only when the relevant rigid or deformable edge overlay is enabled."""
        edges = getattr(renderer, "_xray_collision_edges" if collision else "_xray_edges", True)
        soft_edges = getattr(renderer, "_xray_deformable_edges", True) and any(
            getattr(obj, "_physics_role", None) == "deformable" for _, obj in objects
        )
        if edges or wire_only or soft_edges:
            self._render_pass(renderer, objects, oit=oit, collision=collision, wire_only=wire_only)

    def _render_pass(
        self,
        renderer: RendererGL,
        objects: Sequence[tuple[str, MeshGL | MeshInstancerGL]],
        *,
        oit: bool,
        collision: bool,
        wire_only: bool = False,
    ) -> None:
        """Draw depth-tested front and back wire edges with per-object opacity and LOD settings.

        Temporarily disable mesh culling so the shader can distinguish front
        and back faces. Restore mesh culling and fill polygon mode afterward.
        """
        gl, loc = self._gl, self.locations
        with self:
            gl.glUniformMatrix4fv(loc["view"], 1, gl.GL_FALSE, arr_pointer(renderer._view_matrix))
            gl.glUniformMatrix4fv(loc["projection"], 1, gl.GL_FALSE, arr_pointer(renderer._projection_matrix))
            gl.glUniform3f(loc["view_pos"], *renderer.camera.pos)
            gl.glUniform2f(loc["viewport_size"], *renderer._get_scene_viewport()[2:])
            gl.glUniform1f(loc["shell_opacity"], getattr(renderer, "_xray_opacity", 0.65))
            gl.glUniform1f(loc["collision_opacity"], getattr(renderer, "_xray_collision_opacity", 0.196))
            gl.glUniform1f(loc["inv_reference"], 1.0 / max(renderer.camera.pivot_distance, 1e-6))
            gl.glUniform1i(loc["adaptive"], getattr(renderer, "_xray_adaptive", True))
            gl.glUniform1i(loc["oit_enabled"], oit)
            gl.glUniform1i(loc["collision"], collision)
            gl.glUniform1f(loc["proximity_range"], getattr(renderer, "_xray_proximity_range", 0.0))
            gl.glUniform1f(loc["inspection_radius"], getattr(renderer, "_xray_inspection_radius", 0.0))
            gl.glUniform1i(loc["wire_only"], wire_only)
            gl.glUniform1f(loc["deformable_opacity"], getattr(renderer, "_xray_deformable_opacity", 0.28))
            gl.glDepthFunc(gl.GL_LESS)
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
            gl.glEnable(gl.GL_POLYGON_OFFSET_LINE)
            gl.glPolygonOffset(-1.0, -1.0)
            for _, obj in objects:
                mesh = getattr(obj, "mesh", obj)
                deformable = getattr(obj, "_physics_role", None) == "deformable"
                edges = getattr(
                    renderer,
                    "_xray_deformable_edges" if deformable else "_xray_collision_edges" if collision else "_xray_edges",
                    True,
                )
                if not wire_only and not edges:
                    continue
                if not getattr(mesh, "draw_edge", True):
                    continue
                gl.glUniform1i(loc["deformable"], deformable)
                center, half = ((0, 0, 0), (0, 0, 0)) if deformable else _mesh_metrics(mesh)
                gl.glUniform3f(loc["bounds_center"], *center)
                gl.glUniform3f(loc["bounds_half"], *half)
                if not getattr(obj, "_opacity_attribute_enabled", False):
                    gl.glVertexAttrib1f(9, 1.0)
                # Keep the back-facing wires visible at reduced contrast in
                # both visual meshes and collision proxies. Fills retain their
                # original culling policy.
                cull = mesh.backface_culling
                try:
                    mesh.backface_culling = False
                    if collision and not wire_only:
                        _render_inspection_object(obj, "_lod_wire_draws")
                    else:
                        obj.render()
                finally:
                    mesh.backface_culling = cull
            gl.glDisable(gl.GL_POLYGON_OFFSET_LINE)
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
            gl.glDepthFunc(gl.GL_LESS)
