# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import os
import sys
from abc import abstractmethod

import numpy as np
import warp as wp

import newton
from newton.utils import (
    compute_world_offsets,
    create_box_mesh,
    create_capsule_mesh,
    create_cone_mesh,
    create_cylinder_mesh,
    create_ellipsoid_mesh,
    create_plane_mesh,
    create_sphere_mesh,
    solidify_mesh,
)

from ..core.types import MAXVAL, nparray
from .kernels import compute_hydro_contact_surface_lines, estimate_world_extents


class ViewerBase:
    def __init__(self):
        self.time = 0.0

        self.device = wp.get_device()
        self.model = None
        self.model_changed = True

        # map from shape hash -> Instances
        self._shape_instances = {}

        # inertia box instances -- created on-demand
        self._inertia_box_instances: ViewerBase.ShapeInstances | None = None

        # cache for geometry created via log_shapes()
        # maps from geometry hash -> mesh path
        self._geometry_cache: dict[str, str] = {}

        # line vertices for contact visualization
        self._contact_points0 = None
        self._contact_points1 = None

        # line vertices for joint basis vectors (3 lines per joint)
        self._joint_points0 = None
        self._joint_points1 = None
        self._joint_colors = None

        self._com_positions = None
        self._com_colors = None
        self._com_radii = None

        # World offset support
        self.world_offsets = None  # Array of vec3 offsets per world
        self.max_worlds = None  # Limit on worlds to render (None = all)

        # Display options as individual boolean attributes
        self.show_joints = False
        self.show_com = False
        self.show_particles = False
        self.show_contacts = False
        self.show_springs = False
        self.show_triangles = True
        self.show_collision = False  # force show collision shapes
        self.show_visual = True  # show visual shapes (non collider)
        self.show_static = False  # force static shapes to be visible
        self.show_inertia_boxes = False
        self.show_hydro_contact_surface = False  # show hydroelastic contact surface wireframe
        self.picking_enabled = True  # enable interactive picking via mouse

        # cache for hydroelastic contact surface line rendering (lazily allocated)
        self._hydro_surface_line_starts: wp.array | None = None
        self._hydro_surface_line_ends: wp.array | None = None
        self._hydro_surface_line_colors: wp.array | None = None

        self.model_shape_color: wp.array(dtype=wp.vec3) = None
        """Color of shapes created from ``self.model``, shape (model.shape_count,)"""
        # map from shape index to the slot in the contiguous shape color array ``self.model_shape_color``
        self._shape_to_slot: nparray | None = None
        # map from shape index -> Instances
        self._shape_to_batch: list[ViewerBase.ShapeInstances | None] | None = None

        # cache for isomeshes (computed on demand for collision shapes with SDF volumes)
        # keyed by volume.id (uint64) to deduplicate when multiple shapes share the same SDF volume
        self._isomesh_cache: dict[int, object] = {}

        # SDF isomesh instances -- created on-demand for collision visualization
        self._sdf_isomesh_instances: dict[int, ViewerBase.ShapeInstances] = {}
        self._sdf_isomesh_populated: bool = False  # lazy flag for SDF isomesh population

    def is_running(self) -> bool:
        return True

    def is_paused(self) -> bool:
        return False

    def is_key_down(self, key) -> bool:
        """Default key query API. Concrete viewers can override.

        Args:
            key: Key identifier (string or backend-specific code)

        Returns:
            bool: Always False by default.
        """
        return False

    def set_model(self, model, max_worlds: int | None = None):
        """
        Set the model to be visualized.

        Args:
            model: The Newton model to visualize.
            max_worlds: Maximum number of worlds to render (None = all).
                        Useful for performance when training with many environments.
        """
        if self.model is not None:
            raise RuntimeError("Viewer set_model() can be called only once.")

        self.model = model
        self.max_worlds = max_worlds

        if model is not None:
            self.device = model.device
            self._populate_shapes()

            # Auto-compute world offsets if not already set
            if self.world_offsets is None:
                self._auto_compute_world_offsets()

    def _should_render_world(self, world_idx: int) -> bool:
        """Check if a world should be rendered based on max_worlds limit."""
        if world_idx == -1:  # Global entities always rendered
            return True
        if self.max_worlds is None:
            return True
        return world_idx < self.max_worlds

    def _get_render_world_count(self) -> int:
        """Get the number of worlds to render."""
        if self.model is None:
            return 0
        if self.max_worlds is None:
            return self.model.num_worlds
        return min(self.max_worlds, self.model.num_worlds)

    def _get_shape_isomesh(self, shape_idx: int):
        """Get the isomesh for a collision shape with an SDF volume.

        Computes the marching-cubes isosurface from the SDF volume and caches it.
        Uses the volume.id (uint64) as cache key, so shapes sharing the same SDF
        volume will reuse the same isomesh.

        Args:
            shape_idx: Index of the shape.

        Returns:
            Mesh object for the isomesh, or None if shape has no SDF volume.
        """
        if self.model is None:
            return None

        # Check if this shape has an SDF volume
        sdf_volume = self.model.shape_sdf_volume[shape_idx] if self.model.shape_sdf_volume else None
        if sdf_volume is None:
            return None

        # Use volume.id as cache key - this is a unique uint64 pointer
        volume_id = sdf_volume.id

        # Check if already computed (use False as sentinel for "computed but no mesh")
        if volume_id in self._isomesh_cache:
            cached = self._isomesh_cache[volume_id]
            return cached if cached is not False else None

        # Compute isomesh from SDF volume
        from ..geometry.sdf_utils import compute_isomesh  # noqa: PLC0415

        isomesh = compute_isomesh(sdf_volume)
        self._isomesh_cache[volume_id] = isomesh if isomesh is not None else False
        return isomesh

    def set_camera(self, pos: wp.vec3, pitch: float, yaw: float):
        pass

    def set_world_offsets(self, spacing: tuple[float, float, float] | list[float] | wp.vec3):
        """Set world offsets for visual separation of multiple worlds.

        Args:
            spacing: Spacing between worlds along each axis as a tuple, list, or wp.vec3.
                     Example: (5.0, 5.0, 0.0) for 5 units spacing in X and Y.

        Raises:
            RuntimeError: If model has not been set yet
        """
        if self.model is None:
            raise RuntimeError("Model must be set before calling set_world_offsets()")

        num_worlds = self._get_render_world_count()

        # Get up axis from model
        up_axis = self.model.up_axis

        # Convert to tuple if needed
        if isinstance(spacing, (list, wp.vec3)):
            spacing = (float(spacing[0]), float(spacing[1]), float(spacing[2]))

        # Compute offsets using the shared utility function
        world_offsets = compute_world_offsets(num_worlds, spacing, up_axis)

        # Convert to warp array
        self.world_offsets = wp.array(world_offsets, dtype=wp.vec3, device=self.device)

    def _get_world_extents(self) -> tuple[float, float, float] | None:
        """Get the maximum extents of all worlds in the model."""
        if self.model is None:
            return None

        num_worlds = self.model.num_worlds

        # Initialize bounds arrays for all worlds
        world_bounds_min = wp.full((num_worlds, 3), MAXVAL, dtype=wp.float32, device=self.device)
        world_bounds_max = wp.full((num_worlds, 3), -MAXVAL, dtype=wp.float32, device=self.device)

        # Get initial state for body transforms
        state = self.model.state()

        # Launch kernel to compute bounds for all worlds
        wp.launch(
            kernel=estimate_world_extents,
            dim=self.model.shape_count,
            inputs=[
                self.model.shape_transform,
                self.model.shape_body,
                self.model.shape_collision_radius,
                self.model.shape_world,
                state.body_q,
                num_worlds,
            ],
            outputs=[world_bounds_min, world_bounds_max],
            device=self.device,
        )

        # Get bounds back to CPU
        bounds_min_np = world_bounds_min.numpy()
        bounds_max_np = world_bounds_max.numpy()

        # Find maximum extents across all worlds
        # Mask out invalid bounds (inf values)
        valid_mask = ~np.isinf(bounds_min_np[:, 0])

        if not valid_mask.any():
            # No valid worlds found
            return None

        # Compute extents for valid worlds and take maximum
        valid_min = bounds_min_np[valid_mask]
        valid_max = bounds_max_np[valid_mask]
        world_extents = valid_max - valid_min
        max_extents = np.max(world_extents, axis=0)

        return tuple(max_extents)

    def _auto_compute_world_offsets(self):
        """Automatically compute world offsets based on model extents."""
        # If only one world or no worlds, no offsets needed
        if self._get_render_world_count() <= 1:
            return

        max_extents = self._get_world_extents()
        if max_extents is None:
            return

        # Add margin
        margin = 1.5  # 50% margin between worlds

        # Default to 2D square grid arrangement perpendicular to up axis
        spacing = [np.ceil(max(max_extents) * margin)] * 3
        spacing[self.model.up_axis] = 0.0

        # Set world offsets with computed spacing
        self.set_world_offsets(tuple(spacing))

    def begin_frame(self, time):
        self.time = time

    def log_state(self, state):
        """Render the Newton model."""

        if self.model is None:
            return

        # compute shape transforms and render
        for shapes in self._shape_instances.values():
            visible = self._should_show_shape(shapes.flags, shapes.static)

            if visible:
                shapes.update(state, world_offsets=self.world_offsets)

            colors = shapes.colors if self.model_changed or shapes.colors_changed else None
            materials = shapes.materials if self.model_changed else None

            # Capsules may be rendered via a specialized path by the concrete viewer/backend
            # (e.g., instanced cylinder body + instanced sphere end caps for better batching).
            # The base implementation of log_capsules() falls back to log_instances().
            if shapes.geo_type == newton.GeoType.CAPSULE:
                self.log_capsules(
                    shapes.name,
                    shapes.mesh,
                    shapes.world_xforms,
                    shapes.scales,
                    colors,
                    materials,
                    hidden=not visible,
                )
            else:
                self.log_instances(
                    shapes.name,
                    shapes.mesh,
                    shapes.world_xforms,
                    shapes.scales,  # Always pass scales - needed for transform matrix calculation
                    colors,
                    materials,
                    hidden=not visible,
                )

            shapes.colors_changed = False

        # render SDF isomesh instances for collision visualization (lazily populated)
        sdf_isomesh_just_populated = False
        if self.show_collision and not self._sdf_isomesh_populated:
            self._populate_sdf_isomesh_instances()
            self._sdf_isomesh_populated = True
            sdf_isomesh_just_populated = True

        for shapes in self._sdf_isomesh_instances.values():
            visible = self.show_collision

            if visible:
                shapes.update(state, world_offsets=self.world_offsets)

            # Send colors/materials on model change OR when isomeshes were just populated
            send_appearance = self.model_changed or sdf_isomesh_just_populated
            self.log_instances(
                shapes.name,
                shapes.mesh,
                shapes.world_xforms,
                shapes.scales,
                shapes.colors if send_appearance else None,
                shapes.materials if send_appearance else None,
                hidden=not visible,
            )

        # update inertia box transforms if visible
        if self.show_inertia_boxes:
            if self._inertia_box_instances is None:
                # create instance batch on-demand
                self._populate_inertia_boxes()
            self._inertia_box_instances.update(state, world_offsets=self.world_offsets)

        if self._inertia_box_instances is not None:
            self.log_instances(
                self._inertia_box_instances.name,
                self._inertia_box_instances.mesh,
                self._inertia_box_instances.world_xforms,
                self._inertia_box_instances.scales,
                self._inertia_box_instances.colors,
                self._inertia_box_instances.materials,
                hidden=not self.show_inertia_boxes,
            )

        self._log_triangles(state)
        self._log_particles(state)
        self._log_joints(state)
        self._log_com(state)

        self.model_changed = False

    def log_contacts(self, contacts, state):
        """
        Creates line segments along contact normals for rendering.
        Args:
            name: Identifier for the contact lines
            contacts (newton.Contacts): The contacts to render.
            state: Current simulation state
        """

        if not self.show_contacts:
            # Pass None to hide joints - renderer will handle creating empty arrays
            self.log_lines("/contacts", None, None, None)
            return

        # Get contact count (handle case where it might be zero)
        num_contacts = contacts.rigid_contact_count.numpy()[0]
        max_contacts = contacts.rigid_contact_max

        # Ensure we have buffers for line endpoints
        if self._contact_points0 is None or len(self._contact_points0) < max_contacts:
            self._contact_points0 = wp.array(np.zeros((max_contacts, 3)), dtype=wp.vec3, device=self.device)
            self._contact_points1 = wp.array(np.zeros((max_contacts, 3)), dtype=wp.vec3, device=self.device)

        # Always run the kernel to ensure buffers are properly cleared/updated
        if max_contacts > 0:
            from .kernels import compute_contact_lines  # noqa: PLC0415

            wp.launch(
                kernel=compute_contact_lines,
                dim=max_contacts,
                inputs=[
                    state.body_q,
                    self.model.shape_body,
                    self.model.shape_world,
                    self.world_offsets,
                    contacts.rigid_contact_count,
                    contacts.rigid_contact_shape0,
                    contacts.rigid_contact_shape1,
                    contacts.rigid_contact_point0,
                    contacts.rigid_contact_point1,
                    contacts.rigid_contact_normal,
                    0.1,  # line length scale factor
                ],
                outputs=[
                    self._contact_points0,  # line start points
                    self._contact_points1,  # line end points
                ],
                device=self.device,
            )

        # Always call log_lines to update the renderer (handles zero contacts gracefully)
        if num_contacts > 0:
            # Slice arrays to only include active contacts
            starts = self._contact_points0[:num_contacts]
            ends = self._contact_points1[:num_contacts]
        else:
            # Create empty arrays for zero contacts case
            starts = wp.array([], dtype=wp.vec3, device=self.device)
            ends = wp.array([], dtype=wp.vec3, device=self.device)

        # Use orange-red color for contact normals
        colors = (0.0, 1.0, 0.0)

        self.log_lines("/contacts", starts, ends, colors)

    def log_hydro_contact_surface(self, contact_surface_data, penetrating_only: bool = True):
        """
        Render the hydroelastic contact surface triangles as wireframe lines.

        Args:
            contact_surface_data: A HydroelasticContactSurfaceData instance containing vertex arrays
                for visualization, or None if hydroelastic collision is not enabled.
            penetrating_only: If True, only render penetrating contacts (depth < 0).
        """
        if contact_surface_data is None or not self.show_hydro_contact_surface:
            self.log_lines("/hydro_contact_surface", None, None, None)
            return

        # Get the number of face contacts (triangles)
        num_contacts = int(contact_surface_data.face_contact_count.numpy()[0])

        if num_contacts == 0:
            self.log_lines("/hydro_contact_surface", None, None, None)
            return

        # Each triangle has 3 edges -> 3 line segments per contact
        num_lines = 3 * num_contacts
        max_lines = 3 * contact_surface_data.max_num_face_contacts

        # Pre-allocate line buffers (only once, to max capacity)
        if self._hydro_surface_line_starts is None or len(self._hydro_surface_line_starts) < max_lines:
            self._hydro_surface_line_starts = wp.zeros(max_lines, dtype=wp.vec3, device=self.device)
            self._hydro_surface_line_ends = wp.zeros(max_lines, dtype=wp.vec3, device=self.device)
            self._hydro_surface_line_colors = wp.zeros(max_lines, dtype=wp.vec3, device=self.device)

        # Get depth range for colormap
        depths = contact_surface_data.contact_surface_depth[:num_contacts]

        # Convert triangles to line segments with depth-based colors
        vertices = contact_surface_data.contact_surface_point
        shape_pairs = contact_surface_data.contact_surface_shape_pair
        wp.launch(
            compute_hydro_contact_surface_lines,
            dim=num_contacts,
            inputs=[
                vertices,
                depths,
                shape_pairs,
                self.model.shape_world,
                self.world_offsets,
                num_contacts,
                0.0,
                0.0005,
                penetrating_only,
            ],
            outputs=[self._hydro_surface_line_starts, self._hydro_surface_line_ends, self._hydro_surface_line_colors],
            device=self.device,
        )

        # Render as lines
        self.log_lines(
            "/hydro_contact_surface",
            self._hydro_surface_line_starts[:num_lines],
            self._hydro_surface_line_ends[:num_lines],
            self._hydro_surface_line_colors[:num_lines],
        )

    def log_shapes(
        self,
        name: str,
        geo_type: int,
        geo_scale,
        xforms,
        colors=None,
        materials=None,
        geo_thickness: float = 0.0,
        geo_is_solid: bool = True,
        geo_src=None,
        hidden=False,
    ):
        """
        Convenience helper to create/cache a mesh of a given geometry and
        render a batch of instances with the provided transforms/colors/materials.

        Args:
            name: Instance path/name (e.g., "/world/spheres").
            geo_type: newton.GEO_* constant.
            geo_scale: Geometry scale parameters:
                - Sphere: float radius
                - Capsule/Cylinder/Cone: (radius, height)
                - Plane: (width, length) or float for both
                - Box: (x_extent, y_extent, z_extent) or float for all
            xforms: wp.array(dtype=wp.transform) of instance transforms
            colors: wp.array(dtype=wp.vec3) or None (broadcasted if length 1)
            materials: wp.array(dtype=wp.vec4) or None (broadcasted if length 1)
            thickness: Optional thickness (used for hashing consistency)
            is_solid: If False, can be used for wire/solid hashing parity
            geo_src: Source geometry to use only when `geo_type` is `newton.GeoType.MESH`
            hidden: If True, the shape will not be rendered
        """

        # normalize geo_scale to a list for hashing + mesh creation
        def _as_float_list(value):
            if isinstance(value, tuple | list | np.ndarray):
                return [float(v) for v in value]
            else:
                return [float(value)]

        geo_scale = _as_float_list(geo_scale)

        # ensure mesh exists (shared with populate path)
        mesh_path = self._populate_geometry(
            int(geo_type),
            tuple(geo_scale),
            float(geo_thickness),
            bool(geo_is_solid),
            geo_src=geo_src,
        )

        # prepare instance properties
        num_instances = len(xforms)

        # scales default to ones
        scales = wp.array([wp.vec3(1.0, 1.0, 1.0)] * num_instances, dtype=wp.vec3, device=self.device)

        # broadcast helpers
        def _ensure_vec3_array(arr, default):
            if arr is None:
                return wp.array([default] * num_instances, dtype=wp.vec3, device=self.device)
            if len(arr) == 1 and num_instances > 1:
                return wp.array([arr[0]] * num_instances, dtype=wp.vec3, device=self.device)
            return arr

        def _ensure_vec4_array(arr, default):
            if arr is None:
                return wp.array([default] * num_instances, dtype=wp.vec4, device=self.device)
            if len(arr) == 1 and num_instances > 1:
                return wp.array([arr[0]] * num_instances, dtype=wp.vec4, device=self.device)
            return arr

        # defaults
        default_color = wp.vec3(0.3, 0.8, 0.9)
        default_material = wp.vec4(0.0, 0.7, 0.0, 0.0)

        # planes default to checkerboard and mid-gray if not overridden
        if geo_type == newton.GeoType.PLANE:
            default_color = wp.vec3(0.125, 0.125, 0.25)
            default_material = wp.vec4(0.5, 0.7, 1.0, 0.0)

        colors = _ensure_vec3_array(colors, default_color)
        materials = _ensure_vec4_array(materials, default_material)

        # finally, log the instances
        self.log_instances(name, mesh_path, xforms, scales, colors, materials, hidden=hidden)

    def log_geo(
        self,
        name,
        geo_type: int,
        geo_scale: tuple[float, ...],
        geo_thickness: float,
        geo_is_solid: bool,
        geo_src=None,
        hidden=False,
    ):
        """
        Create a primitive mesh and upload it via log_mesh.

        Expects mesh generators to return interleaved vertices [x, y, z, nx, ny, nz, u, v]
        and an index buffer. Slices them into separate arrays and forwards to log_mesh.
        """

        # GEO_MESH handled by provided source geometry
        if geo_type in (newton.GeoType.MESH, newton.GeoType.CONVEX_MESH):
            if geo_src is None:
                raise ValueError(f"log_geo requires geo_src for MESH or CONVEX_MESH (name={name})")

            # resolve points/indices from source, solidify if requested
            if not geo_is_solid:
                indices, points = solidify_mesh(geo_src.indices, geo_src.vertices, geo_thickness)
            else:
                indices, points = geo_src.indices, geo_src.vertices

            # prepare warp arrays; synthesize normals/uvs
            points = wp.array(points, dtype=wp.vec3, device=self.device)
            indices = wp.array(indices, dtype=wp.int32, device=self.device)
            normals = None
            uvs = None

            if geo_src._normals is not None:
                normals = wp.array(geo_src._normals, dtype=wp.vec3, device=self.device)

            if geo_src._uvs is not None:
                uvs = wp.array(geo_src._uvs, dtype=wp.vec2, device=self.device)

            self.log_mesh(name, points, indices, normals, uvs, hidden=hidden)
            return

        # Generate vertices/indices for supported primitive types
        if geo_type == newton.GeoType.PLANE:
            # Handle "infinite" planes encoded with non-positive scales
            width = geo_scale[0] if geo_scale and geo_scale[0] > 0.0 else 1000.0
            length = geo_scale[1] if len(geo_scale) > 1 and geo_scale[1] > 0.0 else 1000.0
            vertices, indices = create_plane_mesh(width, length)

        elif geo_type == newton.GeoType.SPHERE:
            radius = geo_scale[0]
            vertices, indices = create_sphere_mesh(radius)

        elif geo_type == newton.GeoType.CAPSULE:
            radius, half_height = geo_scale[:2]
            vertices, indices = create_capsule_mesh(radius, half_height, up_axis=2)

        elif geo_type == newton.GeoType.CYLINDER:
            radius, half_height = geo_scale[:2]
            vertices, indices = create_cylinder_mesh(radius, half_height, up_axis=2)

        elif geo_type == newton.GeoType.CONE:
            radius, half_height = geo_scale[:2]
            vertices, indices = create_cone_mesh(radius, half_height, up_axis=2)

        elif geo_type == newton.GeoType.BOX:
            if len(geo_scale) == 1:
                ext = (geo_scale[0],) * 3
            else:
                ext = tuple(geo_scale[:3])
            vertices, indices = create_box_mesh(ext)

        elif geo_type == newton.GeoType.ELLIPSOID:
            # geo_scale contains (rx, ry, rz) semi-axes
            rx = geo_scale[0] if len(geo_scale) > 0 else 1.0
            ry = geo_scale[1] if len(geo_scale) > 1 else rx
            rz = geo_scale[2] if len(geo_scale) > 2 else rx
            vertices, indices = create_ellipsoid_mesh(rx, ry, rz)
        else:
            raise ValueError(f"log_geo does not support geo_type={geo_type} (name={name})")

        # Convert to Warp arrays and forward to log_mesh
        points = wp.array(vertices[:, 0:3], dtype=wp.vec3, device=self.device)
        normals = wp.array(vertices[:, 3:6], dtype=wp.vec3, device=self.device)
        uvs = wp.array(vertices[:, 6:8], dtype=wp.vec2, device=self.device)
        indices = wp.array(indices, dtype=wp.int32, device=self.device)

        self.log_mesh(name, points, indices, normals, uvs, hidden=hidden)

    def log_gizmo(
        self,
        name,
        transform,
    ):
        # Optional: for interactive viewers
        pass

    @abstractmethod
    def log_mesh(
        self,
        name,
        points: wp.array,
        indices: wp.array,
        normals: wp.array | None = None,
        uvs: wp.array | None = None,
        hidden=False,
        backface_culling=True,
    ):
        pass

    @abstractmethod
    def log_instances(self, name, mesh, xforms, scales, colors, materials, hidden=False):
        pass

    # Optional specialized capsule path. Backends can override.
    def log_capsules(self, name, mesh, xforms, scales, colors, materials, hidden=False):
        self.log_instances(name, mesh, xforms, scales, colors, materials, hidden=hidden)

    @abstractmethod
    def log_lines(self, name, starts, ends, colors, width: float = 0.01, hidden=False):
        pass

    @abstractmethod
    def log_points(self, name, points, radii, colors, hidden=False):
        pass

    @abstractmethod
    def log_array(self, name, array):
        pass

    @abstractmethod
    def log_scalar(self, name, value):
        pass

    def apply_forces(self, state):
        pass

    @abstractmethod
    def end_frame(self):
        pass

    @abstractmethod
    def close(self):
        pass

    # handles a batch of mesh instances attached to bodies in the Newton Model
    class ShapeInstances:
        def __init__(self, name, static, flags, mesh, device):
            self.name = name
            self.static = static
            self.flags = flags
            self.mesh = mesh
            self.device = device
            # Optional geometry type for specialized rendering paths (e.g., capsules).
            # -1 means "unknown / not set".
            self.geo_type = -1

            self.parents = []
            self.xforms = []
            self.scales = []
            self.colors = []
            """Color (vec3f) per instance."""
            self.materials = []
            self.worlds = []  # World index for each shape

            self.model_shapes = []

            self.world_xforms = None
            self.colors_changed: bool = False
            """Indicates that the (finalized) ``self.colors`` has changed and it should be included
            in ``Viewer.log_instances()``."""

        def add(self, parent, xform, scale, color, material, shape_index, world=-1):
            # add an instance of the geometry to the batch
            self.parents.append(parent)
            self.xforms.append(xform)
            self.scales.append(scale)
            self.colors.append(color)
            self.materials.append(material)
            self.worlds.append(world)
            self.model_shapes.append(shape_index)

        def finalize(self, shape_colors: wp.array(dtype=wp.vec3) | None = None):
            # convert to warp arrays
            self.parents = wp.array(self.parents, dtype=int, device=self.device)
            self.xforms = wp.array(self.xforms, dtype=wp.transform, device=self.device)
            self.scales = wp.array(self.scales, dtype=wp.vec3, device=self.device)
            if shape_colors is not None:
                assert len(shape_colors) == len(self.scales), "shape_colors length mismatch"
                self.colors = shape_colors
            else:
                self.colors = wp.array(self.colors, dtype=wp.vec3, device=self.device)
            self.materials = wp.array(self.materials, dtype=wp.vec4, device=self.device)
            self.worlds = wp.array(self.worlds, dtype=int, device=self.device)

            self.world_xforms = wp.zeros_like(self.xforms)

        def update(self, state, world_offsets=None):
            from .kernels import update_shape_xforms  # noqa: PLC0415

            wp.launch(
                kernel=update_shape_xforms,
                dim=len(self.xforms),
                inputs=[
                    self.xforms,
                    self.parents,
                    state.body_q,
                    self.worlds,
                    world_offsets,
                ],
                outputs=[self.world_xforms],
                device=self.device,
            )

    # returns a unique (non-stable) identifier for a geometry configuration
    def _hash_geometry(self, geo_type: int, geo_scale, thickness: float, is_solid: bool, geo_src=None) -> int:
        return hash((int(geo_type), geo_src, *geo_scale, float(thickness), bool(is_solid)))

    def _hash_shape(self, geo_hash, shape_static, shape_flags) -> int:
        return hash((geo_hash, shape_static, shape_flags))

    def _should_show_shape(self, flags: int, is_static: bool) -> bool:
        """Determine if a shape should be visible based on current settings."""

        has_collide_flag = bool(flags & int(newton.ShapeFlags.COLLIDE_SHAPES))
        has_visible_flag = bool(flags & int(newton.ShapeFlags.VISIBLE))

        # Static shapes override (e.g., for debugging)
        if is_static and self.show_static:
            return True

        # Shapes can be both collision AND visual (e.g., ground plane).
        # Show if either relevant toggle is enabled.
        if has_collide_flag and self.show_collision:
            return True

        if has_visible_flag and self.show_visual:
            return True

        # Hide if shape has no enabled flags
        return False

    def _populate_geometry(
        self,
        geo_type: int,
        geo_scale,
        thickness: float,
        is_solid: bool,
        geo_src=None,
    ) -> str:
        """Ensure a geometry mesh exists and return its mesh path.

        Computes a stable hash from the parameters; creates and caches the mesh path if needed.
        """

        # normalize
        if isinstance(geo_scale, list | tuple | np.ndarray):
            scale_list = [float(v) for v in geo_scale]
        else:
            scale_list = [float(geo_scale)]

        # include geo_src in hash to match model-driven batching
        geo_hash = self._hash_geometry(
            int(geo_type),
            tuple(scale_list),
            float(thickness),
            bool(is_solid),
            geo_src,
        )

        if geo_hash in self._geometry_cache:
            return self._geometry_cache[geo_hash]

        base_name = {
            newton.GeoType.PLANE: "plane",
            newton.GeoType.SPHERE: "sphere",
            newton.GeoType.CAPSULE: "capsule",
            newton.GeoType.CYLINDER: "cylinder",
            newton.GeoType.CONE: "cone",
            newton.GeoType.BOX: "box",
            newton.GeoType.ELLIPSOID: "ellipsoid",
            newton.GeoType.MESH: "mesh",
            newton.GeoType.CONVEX_MESH: "convex_hull",
        }.get(geo_type)

        if base_name is None:
            raise ValueError(f"Unsupported geo_type for ensure_geometry: {geo_type}")

        mesh_path = f"/geometry/{base_name}_{len(self._geometry_cache)}"
        self.log_geo(
            mesh_path,
            int(geo_type),
            tuple(scale_list),
            float(thickness),
            bool(is_solid),
            geo_src=geo_src if geo_type in (newton.GeoType.MESH, newton.GeoType.CONVEX_MESH) else None,
            hidden=True,
        )
        self._geometry_cache[geo_hash] = mesh_path
        return mesh_path

    # creates meshes and instances for each shape in the Model
    def _populate_shapes(self):
        # convert to NumPy
        shape_body = self.model.shape_body.numpy()
        shape_geo_src = self.model.shape_source
        shape_geo_type = self.model.shape_type.numpy()
        shape_geo_scale = self.model.shape_scale.numpy()
        shape_geo_thickness = self.model.shape_thickness.numpy()
        shape_geo_is_solid = self.model.shape_is_solid.numpy()
        shape_transform = self.model.shape_transform.numpy()
        shape_flags = self.model.shape_flags.numpy()
        shape_world = self.model.shape_world.numpy()
        shape_count = len(shape_body)

        # loop over shapes
        for s in range(shape_count):
            # skip shapes from worlds beyond max_worlds limit
            if not self._should_render_world(shape_world[s]):
                continue

            geo_type = shape_geo_type[s]
            geo_scale = [float(v) for v in shape_geo_scale[s]]
            geo_thickness = float(shape_geo_thickness[s])
            geo_is_solid = bool(shape_geo_is_solid[s])
            geo_src = shape_geo_src[s]

            # skip unsupported
            if geo_type == newton.GeoType.SDF:
                continue

            # check whether we can instance an already created shape with the same geometry
            geo_hash = self._hash_geometry(
                int(geo_type),
                tuple(geo_scale),
                float(geo_thickness),
                bool(geo_is_solid),
                geo_src,
            )

            # ensure geometry exists and get mesh path
            if geo_hash not in self._geometry_cache:
                mesh_name = self._populate_geometry(
                    int(geo_type),
                    tuple(geo_scale),
                    float(geo_thickness),
                    bool(geo_is_solid),
                    geo_src=geo_src if geo_type in (newton.GeoType.MESH, newton.GeoType.CONVEX_MESH) else None,
                )
            else:
                mesh_name = self._geometry_cache[geo_hash]

            # shape options
            flags = shape_flags[s]
            parent = shape_body[s]
            static = parent == -1

            # For collision shapes that ALSO have the VISIBLE flag AND have SDF volumes,
            # treat the original mesh as visual geometry (the SDF isomesh will be rendered
            # separately for collision visualization).
            #
            # Shapes that only have COLLIDE_SHAPES (no VISIBLE) should remain as collision
            # shapes - these are typically convex hull approximations where a separate
            # visual-only copy exists.
            is_collision_shape = flags & int(newton.ShapeFlags.COLLIDE_SHAPES)
            is_visible = flags & int(newton.ShapeFlags.VISIBLE)
            # Check for SDF volume existence without computing the isomesh (lazy evaluation)
            has_sdf = self.model.shape_sdf_volume and self.model.shape_sdf_volume[s] is not None
            if is_collision_shape and is_visible and has_sdf:
                # Remove COLLIDE_SHAPES flag so this is treated as a visual shape
                flags = flags & ~int(newton.ShapeFlags.COLLIDE_SHAPES)

            shape_hash = self._hash_shape(geo_hash, static, flags)

            # ensure batch exists
            if shape_hash not in self._shape_instances:
                shape_name = f"/model/shapes/shape_{len(self._shape_instances)}"
                batch = ViewerBase.ShapeInstances(shape_name, static, flags, mesh_name, self.device)
                batch.geo_type = geo_type
                self._shape_instances[shape_hash] = batch
            else:
                batch = self._shape_instances[shape_hash]

            xform = wp.transform_expand(shape_transform[s])
            scale = np.array([1.0, 1.0, 1.0])

            if (shape_flags[s] & int(newton.ShapeFlags.COLLIDE_SHAPES)) == 0:
                color = wp.vec3(0.5, 0.5, 0.5)
            else:
                # Use shape index for color to ensure each collision shape has a different color
                color = wp.vec3(self._shape_color_map(s))

            material = wp.vec4(0.5, 0.0, 0.0, 0.0)  # roughness, metallic, checker, unused

            if geo_type in (newton.GeoType.MESH, newton.GeoType.CONVEX_MESH):
                scale = np.asarray(geo_scale, dtype=np.float32)

                if geo_src._color is not None:
                    color = wp.vec3(geo_src._color[0:3])

            # plane appearance: checkerboard + gray
            if geo_type == newton.GeoType.PLANE:
                color = wp.vec3(0.125, 0.125, 0.15)
                material = wp.vec4(0.5, 0.5, 1.0, 0.0)

            # add render instance
            batch.add(
                parent=parent,
                xform=xform,
                scale=scale,
                color=color,
                material=material,
                shape_index=s,
                world=shape_world[s],
            )

        # each shape instance object (batch) is associated with one slice
        batches = list(self._shape_instances.values())
        offsets = np.cumsum(np.array([0, *[len(b.scales) for b in batches]], dtype=np.int32)).tolist()
        total_instances = int(offsets[-1])

        # Allocate single contiguous color buffer and copy initial per-batch colors
        if total_instances:
            self.model_shape_color = wp.zeros(total_instances, dtype=wp.vec3, device=self.device)

        for b_idx, batch in enumerate(batches):
            if total_instances:
                color_array = self.model_shape_color[offsets[b_idx] : offsets[b_idx + 1]]
                color_array.assign(wp.array(batch.colors, dtype=wp.vec3, device=self.device))
                batch.finalize(shape_colors=color_array)
            else:
                batch.finalize()

        shape_to_slot = np.full(shape_count, -1, dtype=np.int32)
        for b_idx, batch in enumerate(batches):
            start = offsets[b_idx]
            for local_idx, s_idx in enumerate(batch.model_shapes):
                shape_to_slot[s_idx] = start + local_idx
        self._shape_to_slot = shape_to_slot

        # Build shape -> batch reference mapping for change signalling
        shape_to_batch = [None] * shape_count
        for batch in batches:
            for s_idx in batch.model_shapes:
                shape_to_batch[s_idx] = batch
        self._shape_to_batch = shape_to_batch

        # Note: SDF isomesh instances are populated lazily when show_collision is True
        # to avoid GPU memory allocation until actually needed for visualization

    def _populate_sdf_isomesh_instances(self):
        """Create shape instances for SDF isomeshes (marching cubes visualization).

        These are rendered separately based on the show_collision flag to allow
        independent control of visual mesh and SDF collision visualization.
        """
        if self.model is None:
            return

        shape_body = self.model.shape_body.numpy()
        shape_transform = self.model.shape_transform.numpy()
        shape_flags = self.model.shape_flags.numpy()
        shape_world = self.model.shape_world.numpy()
        shape_geo_scale = self.model.shape_scale.numpy()
        shape_sdf_data = self.model.shape_sdf_data.numpy() if self.model.shape_sdf_data is not None else None
        shape_count = len(shape_body)

        for s in range(shape_count):
            # skip shapes from worlds beyond max_worlds limit
            if not self._should_render_world(shape_world[s]):
                continue

            # Only process collision shapes with SDF volumes
            is_collision_shape = shape_flags[s] & int(newton.ShapeFlags.COLLIDE_SHAPES)
            if not is_collision_shape:
                continue

            isomesh = self._get_shape_isomesh(s)
            if isomesh is None:
                continue

            # Check if scale was baked into the SDF
            scale_baked = shape_sdf_data[s]["scale_baked"] if shape_sdf_data is not None else True

            # Create isomesh geometry (always use (1,1,1) for geometry since isomesh is in SDF space)
            geo_type = newton.GeoType.MESH
            geo_scale = (1.0, 1.0, 1.0)
            geo_thickness = 0.0
            geo_is_solid = True

            geo_hash = self._hash_geometry(
                int(geo_type),
                geo_scale,
                geo_thickness,
                geo_is_solid,
                isomesh,
            )

            # Ensure geometry exists and get mesh path
            if geo_hash not in self._geometry_cache:
                mesh_name = self._populate_geometry(
                    int(geo_type),
                    geo_scale,
                    geo_thickness,
                    geo_is_solid,
                    geo_src=isomesh,
                )
            else:
                mesh_name = self._geometry_cache[geo_hash]

            # Shape options
            flags = shape_flags[s]
            parent = shape_body[s]
            static = parent == -1

            # Use the geo_hash as the batch key for SDF isomesh instances
            if geo_hash not in self._sdf_isomesh_instances:
                shape_name = f"/model/sdf_isomesh/isomesh_{len(self._sdf_isomesh_instances)}"
                batch = ViewerBase.ShapeInstances(shape_name, static, flags, mesh_name, self.device)
                batch.geo_type = geo_type
                self._sdf_isomesh_instances[geo_hash] = batch
            else:
                batch = self._sdf_isomesh_instances[geo_hash]

            xform = wp.transform_expand(shape_transform[s])
            # Apply shape scale if not baked into SDF, otherwise use (1,1,1)
            if scale_baked:
                scale = np.array([1.0, 1.0, 1.0])
            else:
                scale = np.asarray(shape_geo_scale[s], dtype=np.float32)

            # Use distinct collision color palette (different from visual shapes)
            color = wp.vec3(self._collision_color_map(s))
            material = wp.vec4(0.3, 0.0, 0.0, 0.0)  # roughness, metallic, checker, unused

            batch.add(
                parent=parent,
                xform=xform,
                scale=scale,
                color=color,
                material=material,
                shape_index=s,
                world=shape_world[s],
            )

        # Finalize all SDF isomesh batches
        for batch in self._sdf_isomesh_instances.values():
            batch.finalize()

    def update_shape_colors(self, shape_colors: dict[int, wp.vec3 | tuple[float, float, float]]):
        """
        Set colors for a set of shapes at runtime.
        Args:
            shape_colors: mapping from shape index -> color
        """
        if self.model_shape_color is None or self._shape_to_slot is None or self._shape_to_batch is None:
            return

        for s_idx, col in shape_colors.items():
            if s_idx < 0 or s_idx >= len(self._shape_to_slot):
                raise ValueError(f"Shape index {s_idx} out of bounds")
            slot = int(self._shape_to_slot[s_idx])
            if slot < 0:
                continue
            self.model_shape_color[slot : slot + 1].fill_(wp.vec3(col))
            batch_ref = self._shape_to_batch[s_idx]
            if batch_ref is not None:
                batch_ref.colors_changed = True

    # creates meshes and instances for each shape in the Model
    def _populate_inertia_boxes(self):
        # convert to NumPy
        body_count = self.model.body_count
        body_inertia = self.model.body_inertia.numpy()
        body_inv_mass = self.model.body_inv_mass.numpy()
        body_com = self.model.body_com.numpy()
        body_world = self.model.body_world.numpy()

        scale = (1.0, 1.0, 1.0)
        thickness = 0.0
        is_solid = True
        geo_src = None
        geo_args = (newton.GeoType.BOX, scale, thickness, is_solid, geo_src)
        geo_hash = self._hash_geometry(*geo_args)
        if geo_hash not in self._geometry_cache:
            mesh_name = self._populate_geometry(*geo_args)
        else:
            mesh_name = self._geometry_cache[geo_hash]

        static = False
        flags = newton.ShapeFlags.VISIBLE

        shape_name = "/model/inertia_boxes"
        batch = ViewerBase.ShapeInstances(shape_name, static, flags, mesh_name, self.device)

        # loop over bodies
        for body in range(body_count):
            # skip bodies from worlds beyond max_worlds limit
            if not self._should_render_world(body_world[body]):
                continue

            rot, principal_inertia = wp.eig3(wp.mat33(body_inertia[body]))
            xform = wp.transform(body_com[body], wp.quat_from_matrix(rot))

            # computes extents of the solid box that would have similar inertia
            # Note: GeoType.BOX exemplar has sides of length 2.0
            box_inertia = principal_inertia * body_inv_mass[body] * (12 / 8.0)
            scale = (
                np.sqrt(box_inertia[2] + box_inertia[1] - box_inertia[0]),
                np.sqrt(box_inertia[0] + box_inertia[2] - box_inertia[1]),
                np.sqrt(box_inertia[1] + box_inertia[0] - box_inertia[2]),
            )

            # shape options
            parent = body

            color = self._shape_color_map(body)
            if color is None:
                color = wp.vec3(0.5, 0.5, 0.5)
            else:
                color = wp.vec3(color)

            material = wp.vec4(0.5, 0.0, 0.0, 0.0)  # roughness, metallic, checker, unused

            # add render instance
            batch.add(
                parent=parent,
                xform=xform,
                scale=scale,
                color=color,
                material=material,
                shape_index=body,
                world=body_world[body],
            )

        # batch to the GPU
        batch.finalize()
        self._inertia_box_instances = batch

    def _log_joints(self, state):
        """
        Creates line segments for joint basis vectors for rendering.
        Args:
            state: Current simulation state
        """
        if not self.show_joints:
            # Pass None to hide joints - renderer will handle creating empty arrays
            self.log_lines("/model/joints", None, None, None)
            return

        # Get the number of joints
        num_joints = len(self.model.joint_type)
        if num_joints == 0:
            return

        # Each joint produces 3 lines (x, y, z axes)
        max_lines = num_joints * 3

        # Ensure we have buffers for joint line endpoints
        if self._joint_points0 is None or len(self._joint_points0) < max_lines:
            self._joint_points0 = wp.zeros(max_lines, dtype=wp.vec3, device=self.device)
            self._joint_points1 = wp.zeros(max_lines, dtype=wp.vec3, device=self.device)
            self._joint_colors = wp.zeros(max_lines, dtype=wp.vec3, device=self.device)

        # Run the kernel to compute joint basis lines
        # Launch with 3 * num_joints threads (3 lines per joint)
        from .kernels import compute_joint_basis_lines  # noqa: PLC0415

        wp.launch(
            kernel=compute_joint_basis_lines,
            dim=max_lines,
            inputs=[
                self.model.joint_type,
                self.model.joint_parent,
                self.model.joint_child,
                self.model.joint_X_p,
                state.body_q,
                self.model.body_world,
                self.world_offsets,
                self.model.shape_collision_radius,
                self.model.shape_body,
                0.1,  # line scale factor
            ],
            outputs=[
                self._joint_points0,
                self._joint_points1,
                self._joint_colors,
            ],
            device=self.device,
        )

        # Log all joint lines in a single call
        self.log_lines("/model/joints", self._joint_points0, self._joint_points1, self._joint_colors)

    def _log_com(self, state):
        num_bodies = self.model.body_count
        if num_bodies == 0:
            return

        if self._com_positions is None or len(self._com_positions) < num_bodies:
            self._com_positions = wp.zeros(num_bodies, dtype=wp.vec3, device=self.device)
            self._com_colors = wp.full(num_bodies, wp.vec3(1.0, 0.8, 0.0), device=self.device)
            self._com_radii = wp.full(num_bodies, 0.05, dtype=float, device=self.device)

        from .kernels import compute_com_positions  # noqa: PLC0415

        wp.launch(
            kernel=compute_com_positions,
            dim=num_bodies,
            inputs=[
                state.body_q,
                self.model.body_com,
                self.model.body_world,
                self.world_offsets,
            ],
            outputs=[self._com_positions],
            device=self.device,
        )

        self.log_points("/model/com", self._com_positions, self._com_radii, self._com_colors, hidden=not self.show_com)

    def _log_triangles(self, state):
        if self.model.tri_count:
            self.log_mesh(
                "/model/triangles",
                state.particle_q,
                self.model.tri_indices.flatten(),
                hidden=not self.show_triangles,
                backface_culling=False,
            )

    def _log_particles(self, state):
        if self.model.particle_count:
            # just set colors on first frame
            if self.model_changed:
                colors = wp.full(shape=self.model.particle_count, value=wp.vec3(0.7, 0.6, 0.4), device=self.device)
            else:
                colors = None

            self.log_points(
                name="/model/particles",
                points=state.particle_q,
                radii=self.model.particle_radius,
                colors=colors,
                hidden=not self.show_particles,
            )

    @staticmethod
    def _shape_color_map(i: int) -> list[float]:
        # Paul Tol - Bright 9
        colors = [
            [68, 119, 170],  # blue
            [102, 204, 238],  # cyan
            [34, 136, 51],  # green
            [204, 187, 68],  # yellow
            [238, 102, 119],  # red
            [170, 51, 119],  # magenta
            [187, 187, 187],  # grey
            [238, 153, 51],  # orange
            [0, 153, 136],  # teal
        ]

        num_colors = len(colors)
        return [c / 255.0 for c in colors[i % num_colors]]

    @staticmethod
    def _collision_color_map(i: int) -> list[float]:
        # Distinct palette for collision shapes (semi-transparent wireframe look)
        # Uses cooler, more desaturated tones to contrast with bright visual colors
        colors = [
            [180, 120, 200],  # lavender
            [120, 180, 160],  # sage
            [200, 160, 120],  # tan
            [140, 160, 200],  # steel blue
            [200, 140, 160],  # dusty rose
            [160, 200, 140],  # moss
            [180, 180, 140],  # khaki
            [140, 180, 180],  # slate
            [200, 180, 200],  # mauve
        ]

        num_colors = len(colors)
        return [c / 255.0 for c in colors[i % num_colors]]


def is_jupyter_notebook():
    try:
        # Check if get_ipython is defined (available in IPython environments)
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            # This indicates a Jupyter Notebook or JupyterLab environment
            return True
        elif shell == "TerminalInteractiveShell":
            # This indicates a standard IPython terminal
            return False
        else:
            # Other IPython-like environments
            return False
    except NameError:
        # get_ipython is not defined, so it's likely a standard Python script
        return False


def is_sphinx_build() -> bool:
    """
    Detect if we're running inside a Sphinx documentation build (via nbsphinx).

    Returns:
        True if running in Sphinx/nbsphinx, False if in regular Jupyter session.
    """

    # Check for Newton's custom env var (set in docs/conf.py, inherited by nbsphinx subprocesses)
    if os.environ.get("NEWTON_SPHINX_BUILD"):
        return True

    # nbsphinx sets SPHINXBUILD or we can check for sphinx in the call stack
    if os.environ.get("SPHINXBUILD"):
        return True

    # Check if sphinx is in the module list (imported during doc build)
    if "sphinx" in sys.modules or "nbsphinx" in sys.modules:
        return True

    # Check call stack for sphinx-related frames
    try:
        import traceback  # noqa: PLC0415

        for frame_info in traceback.extract_stack():
            if "sphinx" in frame_info.filename.lower() or "nbsphinx" in frame_info.filename.lower():
                return True
    except Exception:
        pass

    return False
