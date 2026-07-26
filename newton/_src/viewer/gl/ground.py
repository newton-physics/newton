# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Analytical shadow-catcher ground plane.

The ground is rendered as a single fullscreen triangle that performs an
analytical ray-plane intersection in the fragment shader. The plane is
geometrically infinite — there is no quad, no edge, no falloff hack — so the
floor extends to the horizon at any scene scale. The plane height is auto-
derived from the scene AABB along the active up-axis whenever the model
changes, which keeps the floor sitting just below the lowest body whether
the scene is cm- or km-scale.

The shader uses the same three directional lights and shadow map as the lit
scene pass; the catcher itself receives shadows but does not cast them.
"""

from __future__ import annotations

import numpy as np


def _ground_normal(up_axis: int) -> tuple[float, float, float]:
    """Unit world-space normal of the ground plane for a given up-axis."""
    if up_axis == 0:
        return (1.0, 0.0, 0.0)
    if up_axis == 1:
        return (0.0, 1.0, 0.0)
    return (0.0, 0.0, 1.0)


class GroundRenderer:
    """Manages the analytical ground plane state and its draw call."""

    def __init__(self, gl):
        self._gl = gl

        # Plane parameters — refreshed by ``update`` each frame.
        self.up_axis: int = 2
        self.normal: tuple[float, float, float] = (0.0, 0.0, 1.0)
        self.height: float = 0.0

        # Visual material (set by ``RendererGL.apply_lookdev``).
        self.color: tuple[float, float, float] = (0.6, 0.6, 0.62)
        self.roughness: float = 0.7

        # An empty VAO is required by core profile when calling glDrawArrays,
        # even though the vertex shader synthesises positions from
        # ``gl_VertexID`` alone.
        self.vao = gl.GLuint()
        gl.glGenVertexArrays(1, self.vao)

    def update(
        self,
        up_axis: int,
        scene_min: np.ndarray | None = None,
        scene_max: np.ndarray | None = None,
    ) -> None:
        """Refresh the plane height + normal from the current scene bounds.

        Args:
            up_axis: Scene up-axis index (0=X, 1=Y, 2=Z).
            scene_min: Optional ``(3,)`` world-space minimum bound.
            scene_max: Optional ``(3,)`` world-space maximum bound. When both
                bounds are provided the plane sits at ``scene_min[up_axis]``
                with a small ``0.01 * scene_extent`` margin so geometry isn't
                z-fighting against the floor at any scale.
        """
        self.up_axis = up_axis
        self.normal = _ground_normal(up_axis)

        if scene_min is not None and scene_max is not None:
            extent = np.asarray(scene_max) - np.asarray(scene_min)
            scene_extent = float(np.max(np.abs(extent))) or 1.0
            self.height = float(scene_min[up_axis]) - 0.01 * scene_extent
        else:
            self.height = 0.0

    def draw(
        self,
        shader,
        *,
        view_matrix: np.ndarray,
        projection_matrix: np.ndarray,
        view_pos: tuple[float, float, float],
        light_space_matrix: np.ndarray,
        shadow_texture: int,
        light_dirs: np.ndarray,
        light_colors: np.ndarray,
        sky_upper: tuple[float, float, float] = (0.78, 0.78, 0.80),
        sky_lower: tuple[float, float, float] = (0.92, 0.92, 0.94),
        ambient_sky_color: tuple[float, float, float] = (0.8, 0.8, 0.85),
        ambient_ground_color: tuple[float, float, float] = (0.3, 0.3, 0.35),
        shadow_radius: float = 3.0,
        shadow_extents: float = 10.0,
        shadow_resolution: float = 2048.0,
        exposure: float = 1.2,
        spotlight_enabled: bool = False,
        shadow_catcher: bool = False,
    ) -> None:
        """Issue the ground draw call (one fullscreen triangle, no buffers)."""
        gl = self._gl
        from .shaders import arr_pointer  # noqa: PLC0415  (avoids cycle)

        # World view-projection: forward for projecting the world hit back to
        # clip-space depth, inverse for reconstructing rays in the vertex
        # shader. Compose + invert in float64 because the view-projection
        # matrix is intrinsically ill-conditioned and float32 inverses produce
        # horizon-jitter at grazing angles. A degenerate matrix (zero-size
        # viewport during a resize, uninitialized camera) skips the ground for
        # the frame rather than crashing the render loop.
        view_m = np.asarray(view_matrix, dtype=np.float64).reshape((4, 4), order="F")
        proj_m = np.asarray(projection_matrix, dtype=np.float64).reshape((4, 4), order="F")
        vp = proj_m @ view_m
        if not np.all(np.isfinite(vp)):
            return
        try:
            inv_vp = np.linalg.inv(vp)
        except np.linalg.LinAlgError:
            return
        vp_cm = np.ascontiguousarray(vp.T, dtype=np.float32)
        inv_vp_cm = np.ascontiguousarray(inv_vp.T, dtype=np.float32)

        with shader:
            gl.glUniformMatrix4fv(shader.loc_view_proj, 1, gl.GL_FALSE, arr_pointer(vp_cm))
            gl.glUniformMatrix4fv(shader.loc_inv_view_proj, 1, gl.GL_FALSE, arr_pointer(inv_vp_cm))
            gl.glUniformMatrix4fv(shader.loc_light_space_matrix, 1, gl.GL_FALSE, arr_pointer(light_space_matrix))
            gl.glUniform3f(shader.loc_view_pos, *view_pos)
            gl.glUniform1i(shader.loc_up_axis, self.up_axis)

            # Plane equation: dot(N, X) = plane_offset.
            gl.glUniform3f(shader.loc_plane_normal, *self.normal)
            gl.glUniform1f(shader.loc_plane_offset, float(self.height))

            # Material
            gl.glUniform3f(shader.loc_ground_color, *self.color)
            gl.glUniform1f(shader.loc_ground_roughness, float(self.roughness))

            # Lighting (3 directional lights, shared with shape shader).
            gl.glUniform3f(shader.loc_sun_direction, *light_dirs[0])
            gl.glUniform3fv(shader.loc_light_dirs, 3, arr_pointer(light_dirs))
            gl.glUniform3fv(shader.loc_light_colors, 3, arr_pointer(light_colors))

            # Shadow map
            gl.glActiveTexture(gl.GL_TEXTURE0)
            if shadow_texture is not None:
                gl.glBindTexture(gl.GL_TEXTURE_2D, shadow_texture)
            gl.glUniform1i(shader.loc_shadow_map, 0)
            gl.glUniform1f(shader.loc_shadow_radius, float(shadow_radius))
            gl.glUniform1f(shader.loc_shadow_extents, float(shadow_extents))
            gl.glUniform1f(shader.loc_shadow_resolution, float(shadow_resolution))

            # Hemispherical ambient, shared with the shape shader.
            gl.glUniform3f(shader.loc_ambient_sky_color, *ambient_sky_color)
            gl.glUniform3f(shader.loc_ambient_ground_color, *ambient_ground_color)

            # Sky gradient endpoints — drive the Fresnel horizon blend that
            # fades the floor into the sky at grazing angles, eliminating
            # the visible step where the matte ground meets the gradient.
            gl.glUniform3f(shader.loc_sky_upper, *sky_upper)
            gl.glUniform3f(shader.loc_sky_lower, *sky_lower)

            # Tone-mapping exposure (applied inline, matching the shape shader).
            gl.glUniform1f(shader.loc_exposure, float(exposure))
            gl.glUniform1i(shader.loc_spotlight_enabled, 1 if spotlight_enabled else 0)
            gl.glUniform1i(shader.loc_shadow_catcher, 1 if shadow_catcher else 0)

            gl.glBindVertexArray(self.vao)
            gl.glDrawArrays(gl.GL_TRIANGLES, 0, 3)
            gl.glBindVertexArray(0)

    def destroy(self) -> None:
        gl = self._gl
        if self.vao is not None:
            gl.glDeleteVertexArrays(1, self.vao)
            self.vao = None
