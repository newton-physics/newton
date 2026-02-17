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

import ctypes
import time

import numpy as np
import warp as wp

import newton as nt
from newton.selection import ArticulationView
from newton.utils import create_sphere_mesh

from ..core.types import override
from ..utils.render import copy_rgb_frame_uint8
from .camera import Camera
from .gl.gui import UI
from .gl.opengl import LinesGL, MeshGL, MeshInstancerGL, RendererGL
from .picking import Picking
from .viewer import ViewerBase
from .wind import Wind


@wp.kernel
def _capsule_duplicate_vec3(in_values: wp.array(dtype=wp.vec3), out_values: wp.array(dtype=wp.vec3)):
    # Duplicate N values into 2N values (two caps per capsule).
    tid = wp.tid()
    out_values[tid] = in_values[tid // 2]


@wp.kernel
def _capsule_duplicate_vec4(in_values: wp.array(dtype=wp.vec4), out_values: wp.array(dtype=wp.vec4)):
    # Duplicate N values into 2N values (two caps per capsule).
    tid = wp.tid()
    out_values[tid] = in_values[tid // 2]


@wp.kernel
def _capsule_build_body_scales(
    shape_scale: wp.array(dtype=wp.vec3),
    shape_indices: wp.array(dtype=wp.int32),
    out_scales: wp.array(dtype=wp.vec3),
):
    # model.shape_scale stores capsule params as (radius, half_height, _unused).
    # ViewerGL instances scale meshes with a full (x, y, z) vector, so we expand to
    # (radius, radius, half_height) for the cylinder body.
    tid = wp.tid()
    s = shape_indices[tid]
    scale = shape_scale[s]
    r = scale[0]
    half_height = scale[1]
    out_scales[tid] = wp.vec3(r, r, half_height)


@wp.kernel
def _capsule_build_cap_xforms_and_scales(
    capsule_xforms: wp.array(dtype=wp.transform),
    capsule_scales: wp.array(dtype=wp.vec3),
    out_xforms: wp.array(dtype=wp.transform),
    out_scales: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    i = tid // 2
    # Each capsule has two caps; even tid is the +Z end, odd tid is the -Z end.
    is_plus_end = (tid % 2) == 0

    t = capsule_xforms[i]
    p = wp.transform_get_translation(t)
    q = wp.transform_get_rotation(t)

    r = capsule_scales[i][0]
    half_height = capsule_scales[i][2]
    offset_local = wp.vec3(0.0, 0.0, half_height if is_plus_end else -half_height)
    p2 = p + wp.quat_rotate(q, offset_local)

    out_xforms[tid] = wp.transform(p2, q)
    out_scales[tid] = wp.vec3(r, r, r)


@wp.kernel
def _compute_shape_vbo_xforms(
    shape_transform: wp.array(dtype=wp.transformf),
    shape_body: wp.array(dtype=int),
    body_q: wp.array(dtype=wp.transformf),
    shape_scale: wp.array(dtype=wp.vec3),
    shape_type: wp.array(dtype=int),
    shape_world: wp.array(dtype=int),
    world_offsets: wp.array(dtype=wp.vec3),
    write_indices: wp.array(dtype=int),
    out_world_xforms: wp.array(dtype=wp.transformf),
    out_vbo_xforms: wp.array(dtype=wp.mat44),
):
    """Process all model shapes, write mat44 to grouped output positions."""
    tid = wp.tid()
    out_idx = write_indices[tid]
    if out_idx < 0:
        return

    local_xform = shape_transform[tid]
    parent = shape_body[tid]

    if parent >= 0:
        xform = wp.transform_multiply(body_q[parent], local_xform)
    else:
        xform = local_xform

    if world_offsets:
        wi = shape_world[tid]
        if wi >= 0 and wi < world_offsets.shape[0]:
            p = wp.transform_get_translation(xform)
            xform = wp.transform(p + world_offsets[wi], wp.transform_get_rotation(xform))

    out_world_xforms[out_idx] = xform

    p = wp.transform_get_translation(xform)
    q = wp.transform_get_rotation(xform)
    R = wp.quat_to_matrix(q)

    # Only mesh/convex_mesh shapes use model scale; other primitives have
    # their dimensions baked into the geometry mesh, so scale is (1,1,1).
    geo = shape_type[tid]
    if geo == int(nt.GeoType.MESH) or geo == int(nt.GeoType.CONVEX_MESH):
        s = shape_scale[tid]
    else:
        s = wp.vec3(1.0, 1.0, 1.0)

    out_vbo_xforms[out_idx] = wp.mat44(
        R[0, 0] * s[0],
        R[1, 0] * s[0],
        R[2, 0] * s[0],
        0.0,
        R[0, 1] * s[1],
        R[1, 1] * s[1],
        R[2, 1] * s[1],
        0.0,
        R[0, 2] * s[2],
        R[1, 2] * s[2],
        R[2, 2] * s[2],
        0.0,
        p[0],
        p[1],
        p[2],
        1.0,
    )


class ViewerGL(ViewerBase):
    """
    OpenGL-based interactive viewer for Newton physics models.

    This class provides a graphical interface for visualizing and interacting with
    Newton models using OpenGL rendering. It supports real-time simulation control,
    camera navigation, object picking, wind effects, and a rich ImGui-based UI for
    model introspection and visualization options.

    Key Features:
        - Real-time 3D rendering of Newton models and simulation states.
        - Camera navigation with WASD/QE and mouse controls.
        - Object picking and manipulation via mouse.
        - Visualization toggles for joints, contacts, particles, springs, etc.
        - Wind force controls and visualization.
        - Performance statistics overlay (FPS, object counts, etc.).
        - Selection panel for introspecting and filtering model attributes.
        - Extensible logging of meshes, lines, points, and arrays for custom visualization.
    """

    def __init__(self, width=1920, height=1080, vsync=False, headless=False):
        """
        Initialize the OpenGL viewer and UI.

        Args:
            width (int): Window width in pixels.
            height (int): Window height in pixels.
            vsync (bool): Enable vertical sync.
            headless (bool): Run in headless mode (no window).
        """
        super().__init__()

        # map from path to any object type
        self.objects = {}
        self.lines = {}
        self.renderer = RendererGL(vsync=vsync, screen_width=width, screen_height=height, headless=headless)
        self.renderer.set_title("Newton Viewer")

        self._paused = False
        self._packed_vbo_xforms = None

        # State caching for selection panel
        self._last_state = None
        self._last_control = None

        # Selection panel state
        self._selection_ui_state = {
            "selected_articulation_pattern": "*",
            "selected_articulation_view": None,
            "selected_attribute": "joint_q",
            "attribute_options": ["joint_q", "joint_qd", "joint_f", "body_q", "body_qd"],
            "include_joints": "",
            "exclude_joints": "",
            "include_links": "",
            "exclude_links": "",
            "show_values": False,
            "selected_batch_idx": 0,
            "error_message": "",
        }

        self.renderer.register_key_press(self.on_key_press)
        self.renderer.register_key_release(self.on_key_release)
        self.renderer.register_mouse_press(self.on_mouse_press)
        self.renderer.register_mouse_release(self.on_mouse_release)
        self.renderer.register_mouse_drag(self.on_mouse_drag)
        self.renderer.register_mouse_scroll(self.on_mouse_scroll)
        self.renderer.register_resize(self.on_resize)

        # Camera movement settings
        self._camera_speed = 0.04
        self._cam_vel = np.zeros(3, dtype=np.float32)
        self._cam_speed = 4.0  # m/s
        self._cam_damp_tau = 0.083  # s

        # initialize viewer-local timer for per-frame integration
        self._last_time = time.perf_counter()

        # Only create UI in non-headless mode to avoid OpenGL context dependency
        if not headless:
            self.ui = UI(self.renderer.window)
        else:
            self.ui = None
        self._gizmo_log = None

        # Performance tracking
        self._fps_history = []
        self._last_fps_time = time.perf_counter()
        self._frame_count = 0
        self._current_fps = 0.0

        # a low resolution sphere mesh for point rendering
        self._point_mesh = None

        # UI visibility toggle
        self.show_ui = True

        # UI callback system - organized by position
        # positions: "side", "stats", "free"
        self._ui_callbacks = {"side": [], "stats": [], "free": []}

        # Initialize PBO (Pixel Buffer Object) resources used in the `get_frame` method.
        self._pbo = None
        self._wp_pbo = None

        self.set_model(None)

    def _hash_geometry(self, geo_type: int, geo_scale, thickness: float, is_solid: bool, geo_src=None) -> int:
        # For capsules, ignore (radius, half_height) in the geometry hash so varying-length capsules batch together.
        # Capsule dimensions are stored per-shape in model.shape_scale as (radius, half_height, _unused) and
        # are remapped in set_model() to per-instance render scales (radius, radius, half_height).
        if geo_type == nt.GeoType.CAPSULE:
            geo_scale = (1.0, 1.0)
        return super()._hash_geometry(geo_type, geo_scale, thickness, is_solid, geo_src)

    def _invalidate_pbo(self):
        """Invalidate PBO resources, forcing reallocation on next get_frame() call."""
        if self._wp_pbo is not None:
            self._wp_pbo = None  # Let Python garbage collect the RegisteredGLBuffer
        if self._pbo is not None:
            gl = RendererGL.gl
            pbo_id = (gl.GLuint * 1)(self._pbo)
            gl.glDeleteBuffers(1, pbo_id)
            self._pbo = None

    def register_ui_callback(self, callback, position="side"):
        """
        Register a UI callback to be rendered during the UI phase.

        Args:
            callback: Function to be called during UI rendering
            position: Position where the UI should be rendered. One of:
                     "side" - Side callback (default)
                     "stats" - Stats/metrics area
                     "free" - Free-floating UI elements
        """
        if not callable(callback):
            raise TypeError("callback must be callable")

        if position not in self._ui_callbacks:
            valid_positions = list(self._ui_callbacks.keys())
            raise ValueError(f"Invalid position '{position}'. Must be one of: {valid_positions}")

        self._ui_callbacks[position].append(callback)

    # helper function to create a low resolution sphere mesh for point rendering
    def _create_point_mesh(self):
        """
        Create a low-resolution sphere mesh for point rendering.
        """
        vertices, indices = create_sphere_mesh(1.0, 6, 6)
        self._point_mesh = MeshGL(len(vertices), len(indices), self.device)

        points = wp.array(vertices[:, 0:3], dtype=wp.vec3, device=self.device)
        normals = wp.array(vertices[:, 3:6], dtype=wp.vec3, device=self.device)
        uvs = wp.array(vertices[:, 6:8], dtype=wp.vec2, device=self.device)
        indices = wp.array(indices, dtype=wp.int32, device=self.device)

        self._point_mesh.update(points, indices, normals, uvs)

    @override
    def log_gizmo(
        self,
        name,
        transform,
    ):
        # Store for this frame; call this every frame you want it drawn/active
        self._gizmo_log[name] = transform

    @override
    def set_model(self, model, max_worlds: int | None = None):
        """
        Set the Newton model to visualize.

        Args:
            model: The Newton model instance.
            max_worlds: Maximum number of worlds to render (None = all).
        """
        super().set_model(model, max_worlds=max_worlds)

        if self.model is not None:
            # For capsule batches, replace per-instance scales with (radius, radius, half_height)
            # so the capsule instancer path has the needed parameters.
            shape_scale = self.model.shape_scale
            if shape_scale.device != self.device:
                # Defensive: ensure inputs are on the launch device.
                shape_scale = wp.clone(shape_scale, device=self.device)

            def _ensure_indices_wp(model_shapes) -> wp.array:
                # Return shape indices as a Warp array on the viewer device
                if isinstance(model_shapes, wp.array):
                    if model_shapes.device == self.device:
                        return model_shapes
                    return wp.array(model_shapes.numpy().astype(np.int32), dtype=wp.int32, device=self.device)
                return wp.array(model_shapes, dtype=wp.int32, device=self.device)

            for batch in self._shape_instances.values():
                if batch.geo_type != nt.GeoType.CAPSULE:
                    continue

                shape_indices = _ensure_indices_wp(batch.model_shapes)
                num_shapes = len(shape_indices)
                out_scales = wp.empty(num_shapes, dtype=wp.vec3, device=self.device)
                if num_shapes == 0:
                    batch.scales = out_scales
                    continue
                wp.launch(
                    _capsule_build_body_scales,
                    dim=num_shapes,
                    inputs=[shape_scale, shape_indices],
                    outputs=[out_scales],
                    device=self.device,
                    record_tape=False,
                )
                batch.scales = out_scales

        self.picking = Picking(model, pick_stiffness=10000.0, pick_damping=1000.0, world_offsets=self.world_offsets)
        self.wind = Wind(model)

        # Build packed arrays for batched GPU rendering of shape instances
        self._build_packed_vbo_arrays()

        fb_w, fb_h = self.renderer.window.get_framebuffer_size()
        self.camera = Camera(width=fb_w, height=fb_h, up_axis=model.up_axis if model else "Z")

    def _build_packed_vbo_arrays(self):
        """Build write-index + output arrays for batched shape transform computation.

        The kernel processes all model shapes (coalesced reads), uses a write-index
        array to scatter results into contiguous groups in the output buffer.
        """
        from .gl.opengl import MeshGL, MeshInstancerGL  # noqa: PLC0415

        if self.model is None:
            self._packed_groups = []
            return

        shape_count = self.model.shape_count
        device = self.device

        groups = []
        capsule_keys = set()
        total = 0

        for key, shapes in self._shape_instances.items():
            n = shapes.xforms.shape[0] if isinstance(shapes.xforms, wp.array) else len(shapes.xforms)
            if n == 0:
                continue
            if shapes.geo_type == nt.GeoType.CAPSULE:
                capsule_keys.add(key)
            groups.append((key, shapes, total, n))
            total += n

        self._capsule_keys = capsule_keys
        self._packed_groups = groups

        if total == 0:
            return

        # Write-index: maps model shape index → packed output position (-1 = skip)
        write_np = np.full(shape_count, -1, dtype=np.int32)
        # World xforms output (capsules read these for cap sphere computation)
        all_world_xforms = wp.empty(total, dtype=wp.transform, device=device)

        for _key, shapes, offset, n in groups:
            model_shapes = np.asarray(shapes.model_shapes, dtype=np.int32)
            write_np[model_shapes] = np.arange(offset, offset + n, dtype=np.int32)

            if _key in capsule_keys:
                shapes.world_xforms = all_world_xforms[offset : offset + n]

            if _key not in capsule_keys:
                if shapes.name not in self.objects:
                    if shapes.mesh in self.objects and isinstance(self.objects[shapes.mesh], MeshGL):
                        self.objects[shapes.name] = MeshInstancerGL(max(n, 1), self.objects[shapes.mesh])

        self._packed_write_indices = wp.array(write_np, dtype=int, device=device)
        self._packed_world_xforms = all_world_xforms
        self._packed_vbo_xforms = wp.empty(total, dtype=wp.mat44, device=device)
        self._packed_vbo_xforms_host = wp.empty(total, dtype=wp.mat44, device="cpu", pinned=True)

    @override
    def set_world_offsets(self, spacing: tuple[float, float, float] | list[float] | wp.vec3):
        """Set world offsets and update the picking system.

        Args:
            spacing: Spacing between worlds along each axis.
        """
        super().set_world_offsets(spacing)
        # Update picking system with new world offsets
        if hasattr(self, "picking") and self.picking is not None:
            self.picking.world_offsets = self.world_offsets

    @override
    def set_camera(self, pos: wp.vec3, pitch: float, yaw: float):
        """
        Set the camera position, pitch, and yaw.

        Args:
            pos: The camera position.
            pitch: The camera pitch.
            yaw: The camera yaw.
        """
        self.camera.pos = pos
        self.camera.pitch = pitch
        self.camera.yaw = yaw

    @override
    def log_mesh(
        self,
        name,
        points: wp.array,
        indices: wp.array,
        normals: wp.array | None = None,
        uvs: wp.array | None = None,
        texture: np.ndarray | str | None = None,
        hidden=False,
        backface_culling=True,
    ):
        """
        Log a mesh for rendering.

        Args:
            name (str): Unique name for the mesh.
            points (wp.array): Vertex positions.
            indices (wp.array): Triangle indices.
            normals (wp.array, optional): Vertex normals.
            uvs (wp.array, optional): Vertex UVs.
            texture (np.ndarray | str, optional): Texture path/URL or image array (H, W, C).
            hidden (bool): Whether the mesh is hidden.
            backface_culling (bool): Enable backface culling.
        """
        assert isinstance(points, wp.array)
        assert isinstance(indices, wp.array)
        assert normals is None or isinstance(normals, wp.array)
        assert uvs is None or isinstance(uvs, wp.array)

        if name not in self.objects:
            self.objects[name] = MeshGL(
                len(points), len(indices), self.device, hidden=hidden, backface_culling=backface_culling
            )

        self.objects[name].update(points, indices, normals, uvs, texture)
        self.objects[name].hidden = hidden
        self.objects[name].backface_culling = backface_culling

    @override
    def log_instances(self, name, mesh, xforms, scales, colors, materials, hidden=False):
        """
        Log a batch of mesh instances for rendering.

        Args:
            name (str): Unique name for the instancer.
            mesh (str): Name of the base mesh.
            xforms: Array of transforms.
            scales: Array of scales.
            colors: Array of colors.
            materials: Array of materials.
            hidden: Whether the instances are hidden.
        """
        if mesh not in self.objects:
            raise RuntimeError(f"Path {mesh} not found")

        # check it is a mesh object
        if not isinstance(self.objects[mesh], MeshGL):
            raise RuntimeError(f"Path {mesh} is not a Mesh object")

        instancer = self.objects.get(name, None)
        transform_count = len(xforms) if xforms is not None else 0
        resized = False

        if instancer is None:
            capacity = max(transform_count, 1)
            instancer = MeshInstancerGL(capacity, self.objects[mesh])
            self.objects[name] = instancer
            resized = True
        elif transform_count > instancer.num_instances:
            new_capacity = max(transform_count, instancer.num_instances * 2)
            instancer = MeshInstancerGL(new_capacity, self.objects[mesh])
            self.objects[name] = instancer
            resized = True

        needs_update = resized or not hidden
        if needs_update:
            self.objects[name].update_from_transforms(xforms, scales, colors, materials)

        self.objects[name].hidden = hidden

    @override
    def log_capsules(self, name, mesh, xforms, scales, colors, materials, hidden=False):
        """
        Render capsules using instanced cylinder bodies + instanced sphere end caps.

        This specialized path improves batching for varying-length capsules by reusing two
        prototype meshes (unit cylinder + unit sphere) and applying per-instance transforms/scales.

        Args:
            name (str): Unique name for the capsule instancer group.
            mesh: Capsule prototype mesh path from ViewerBase (unused in this backend).
            xforms: Capsule instance transforms (wp.transform), length N.
            scales: Capsule body instance scales, expected (radius, radius, half_height), length N.
            colors: Capsule instance colors (wp.vec3), length N or None (no update).
            materials: Capsule instance materials (wp.vec4), length N or None (no update).
            hidden (bool): Whether the instances are hidden.
        """
        # Render capsules via instanced cylinder body + instanced sphere caps.
        sphere_mesh = "/geometry/_capsule_instancer/sphere"
        cylinder_mesh = "/geometry/_capsule_instancer/cylinder"

        if sphere_mesh not in self.objects:
            self.log_geo(sphere_mesh, nt.GeoType.SPHERE, (1.0,), 0.0, True, hidden=True)
        if cylinder_mesh not in self.objects:
            self.log_geo(cylinder_mesh, nt.GeoType.CYLINDER, (1.0, 1.0), 0.0, True, hidden=True)

        # Cylinder body uses the capsule transforms and (radius, radius, half_height) scaling.
        cyl_name = f"{name}/capsule_cylinder"
        cap_name = f"{name}/capsule_caps"

        # If hidden, just hide the instancers (skip all per-frame cap buffer work).
        if hidden:
            self.log_instances(cyl_name, cylinder_mesh, None, None, None, None, hidden=True)
            self.log_instances(cap_name, sphere_mesh, None, None, None, None, hidden=True)
            return

        self.log_instances(cyl_name, cylinder_mesh, xforms, scales, colors, materials, hidden=hidden)

        # Sphere caps: two spheres per capsule, offset by ±half_height along local +Z.
        n = len(xforms) if xforms is not None else 0
        if n == 0:
            self.log_instances(cap_name, sphere_mesh, None, None, None, None, hidden=True)
            return

        cap_count = n * 2
        cap_xforms = wp.empty(cap_count, dtype=wp.transform, device=self.device)
        cap_scales = wp.empty(cap_count, dtype=wp.vec3, device=self.device)

        wp.launch(
            _capsule_build_cap_xforms_and_scales,
            dim=cap_count,
            inputs=[xforms, scales],
            outputs=[cap_xforms, cap_scales],
            device=self.device,
            record_tape=False,
        )

        cap_colors = None
        if colors is not None:
            cap_colors = wp.empty(cap_count, dtype=wp.vec3, device=self.device)
            wp.launch(
                _capsule_duplicate_vec3,
                dim=cap_count,
                inputs=[colors],
                outputs=[cap_colors],
                device=self.device,
                record_tape=False,
            )

        cap_materials = None
        if materials is not None:
            cap_materials = wp.empty(cap_count, dtype=wp.vec4, device=self.device)
            wp.launch(
                _capsule_duplicate_vec4,
                dim=cap_count,
                inputs=[materials],
                outputs=[cap_materials],
                device=self.device,
                record_tape=False,
            )

        self.log_instances(cap_name, sphere_mesh, cap_xforms, cap_scales, cap_colors, cap_materials, hidden=hidden)

    @override
    def log_lines(
        self,
        name,
        starts: wp.array,
        ends: wp.array,
        colors,
        width: float = 0.01,
        hidden=False,
    ):
        """
        Log line data for rendering.

        Args:
            name (str): Unique identifier for the line batch.
            starts (wp.array): Array of line start positions (shape: [N, 3]) or None for empty.
            ends (wp.array): Array of line end positions (shape: [N, 3]) or None for empty.
            colors: Array of line colors (shape: [N, 3]) or tuple/list of RGB or None for empty.
            width: The width of the lines (float)
            hidden (bool): Whether the lines are initially hidden.
        """
        # Handle empty logs by resetting the LinesGL object
        if starts is None or ends is None or colors is None:
            if name in self.lines:
                self.lines[name].update(None, None, None)
            return

        assert isinstance(starts, wp.array)
        assert isinstance(ends, wp.array)
        num_lines = len(starts)
        assert len(ends) == num_lines, "Number of line ends must match line begins"

        # Handle tuple/list colors by expanding to array (only if not already converted above)
        if isinstance(colors, tuple | list):
            if num_lines > 0:
                color_vec = wp.vec3(*colors)
                colors = wp.zeros(num_lines, dtype=wp.vec3, device=self.device)
                colors.fill_(color_vec)  # Efficiently fill on GPU
            else:
                # Handle zero lines case
                colors = wp.array([], dtype=wp.vec3, device=self.device)

        assert isinstance(colors, wp.array)
        assert len(colors) == num_lines, "Number of line colors must match line begins"

        # Create or resize LinesGL object based on current requirements
        if name not in self.lines:
            # Start with reasonable default size, will expand as needed
            max_lines = max(num_lines, 1000)  # Reasonable default
            self.lines[name] = LinesGL(max_lines, self.device, hidden=hidden)
        elif num_lines > self.lines[name].max_lines:
            # Need to recreate with larger capacity
            self.lines[name].destroy()
            max_lines = max(num_lines, self.lines[name].max_lines * 2)
            self.lines[name] = LinesGL(max_lines, self.device, hidden=hidden)

        self.lines[name].update(starts, ends, colors)

    @override
    def log_points(self, name, points, radii, colors, hidden=False):
        """
        Log a batch of points for rendering as spheres.

        Args:
            name (str): Unique name for the point batch.
            points: Array of point positions.
            radii: Array of point radius values.
            colors: Array of point colors.
            hidden (bool): Whether the points are hidden.
        """
        if self._point_mesh is None:
            self._create_point_mesh()

        num_points = len(points)
        if name not in self.objects:
            # Start with a reasonable default.
            initial_capacity = max(num_points, 256)
            self.objects[name] = MeshInstancerGL(initial_capacity, self._point_mesh)
        elif num_points > self.objects[name].num_instances:
            old = self.objects[name]
            new_capacity = max(num_points, old.num_instances * 2)
            self.objects[name] = MeshInstancerGL(new_capacity, self._point_mesh)

        self.objects[name].update_from_points(points, radii, colors)
        self.objects[name].hidden = hidden

    @override
    def log_array(self, name, array):
        """
        Log a generic array for visualization (not implemented).
        """
        pass

    @override
    def log_scalar(self, name, value):
        """
        Log a scalar value for visualization (not implemented).
        """
        pass

    @override
    def log_state(self, state):
        """
        Log the current simulation state for rendering.

        For shape instances on CUDA, uses a batched path: 2 kernel launches +
        1 D2H copy to a shared pinned buffer, then uploads slices per instancer.
        Everything else (capsules, SDF, particles, joints, …) uses the standard path.
        """
        self._last_state = state

        if self.model is None:
            return

        if self._packed_vbo_xforms is not None and self.device.is_cuda:
            # ---- Single kernel over all model shapes, scatter-write to grouped output ----
            wp.launch(
                _compute_shape_vbo_xforms,
                dim=self.model.shape_count,
                inputs=[
                    self.model.shape_transform,
                    self.model.shape_body,
                    state.body_q,
                    self.model.shape_scale,
                    self.model.shape_type,
                    self.model.shape_world,
                    self.world_offsets,
                    self._packed_write_indices,
                ],
                outputs=[self._packed_world_xforms, self._packed_vbo_xforms],
                device=self.device,
                record_tape=False,
            )
            wp.copy(self._packed_vbo_xforms_host, self._packed_vbo_xforms)
            wp.synchronize()  # copy is async (pinned destination), must sync before CPU read

            # ---- Upload pinned host slices to GL per instancer ----
            host_np = self._packed_vbo_xforms_host.numpy()

            for key, shapes, offset, count in self._packed_groups:
                visible = self._should_show_shape(shapes.flags, shapes.static)
                colors = shapes.colors if self.model_changed or shapes.colors_changed else None
                materials = shapes.materials if self.model_changed else None

                if key in self._capsule_keys:
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
                    instancer = self.objects.get(shapes.name)
                    if instancer is not None:
                        instancer.hidden = not visible
                        instancer.update_from_pinned(
                            host_np[offset : offset + count],
                            count,
                            colors,
                            materials,
                        )

                shapes.colors_changed = False

            # ---- Non-shape rendering uses standard synchronous paths ----
            self._log_non_shape_state(state)
            self.model_changed = False
        else:
            # Fallback for CPU or when no packed data is available
            super().log_state(state)

        self._render_picking_line(state)

    def _render_picking_line(self, state):
        """
        Render a line from the mouse cursor to the actual picked point on the geometry.

        Args:
            state: The current simulation state.
        """
        if not self.picking_enabled or not self.picking.is_picking():
            # Clear the picking line if not picking
            self.log_lines("picking_line", None, None, None)
            return

        # Get the picked body index
        pick_body_idx = self.picking.pick_body.numpy()[0]
        if pick_body_idx < 0:
            self.log_lines("picking_line", None, None, None)
            return

        # Get the pick target and current picked point on geometry (in physics space)
        pick_state = self.picking.pick_state.numpy()
        pick_target = np.array([pick_state[8], pick_state[9], pick_state[10]], dtype=np.float32)
        picked_point = np.array([pick_state[11], pick_state[12], pick_state[13]], dtype=np.float32)

        # Apply world offset to convert from physics space to visual space
        if self.world_offsets is not None and self.world_offsets.shape[0] > 0:
            if self.model.body_world is not None:
                body_world_idx = self.model.body_world.numpy()[pick_body_idx]
                if body_world_idx >= 0 and body_world_idx < self.world_offsets.shape[0]:
                    world_offset = self.world_offsets.numpy()[body_world_idx]
                    pick_target = pick_target + world_offset
                    picked_point = picked_point + world_offset

        # Create line data
        starts = wp.array(
            [wp.vec3(picked_point[0], picked_point[1], picked_point[2])], dtype=wp.vec3, device=self.device
        )
        ends = wp.array([wp.vec3(pick_target[0], pick_target[1], pick_target[2])], dtype=wp.vec3, device=self.device)
        colors = wp.array([wp.vec3(0.0, 1.0, 1.0)], dtype=wp.vec3, device=self.device)

        # Render the line
        self.log_lines("picking_line", starts, ends, colors, hidden=False)

    @override
    def begin_frame(self, time):
        """
        Begin a new frame (calls parent implementation).

        Args:
            time: Current simulation time.
        """
        super().begin_frame(time)
        self._gizmo_log = {}

    @override
    def end_frame(self):
        """
        Finish rendering the current frame and process window events.

        This method first updates the renderer which will poll and process
        window events.  It is possible that the user closes the window during
        this event processing step, which would invalidate the underlying
        OpenGL context.  Trying to issue GL calls after the context has been
        destroyed results in a crash (access violation).  Therefore we check
        whether an exit was requested and early-out before touching GL if so.
        """
        self._update()

    @override
    def apply_forces(self, state):
        """
        Apply viewer-driven forces (picking, wind) to the model.

        Args:
            state: The current simulation state.
        """
        if self.picking_enabled:
            self.picking._apply_picking_force(state)

        # Apply wind forces
        self.wind._apply_wind_force(state)

    def _update(self):
        """
        Internal update: process events, update camera, wind, render scene and UI.
        """
        self.renderer.update()

        # Integrate camera motion with viewer-owned timing
        now = time.perf_counter()
        dt = max(0.0, min(0.1, now - self._last_time))
        self._last_time = now
        self._update_camera(dt)

        self.wind.update(dt)

        # If the window was closed during event processing, skip rendering
        if self.renderer.has_exit():
            return

        # Render the scene and present it
        self.renderer.render(self.camera, self.objects, self.lines)

        # Always update FPS tracking, even if UI is hidden
        self._update_fps()

        if self.ui and self.ui.is_available and self.show_ui:
            self.ui.begin_frame()

            # Render the UI
            self._render_ui()

            self.ui.end_frame()
            self.ui.render()

        self.renderer.present()

    def get_frame(self, target_image: wp.array | None = None, render_ui: bool = False) -> wp.array:
        """
        Retrieve the last rendered frame.

        This method uses OpenGL Pixel Buffer Objects (PBO) and CUDA interoperability
        to transfer pixel data entirely on the GPU, avoiding expensive CPU-GPU transfers.

        Args:
            target_image (wp.array, optional):
                Optional pre-allocated Warp array with shape `(height, width, 3)`
                and dtype `wp.uint8`. If `None`, a new array will be created.
            render_ui (bool): Whether to render the UI.

        Returns:
            wp.array: GPU array containing RGB image data with shape `(height, width, 3)`
                and dtype `wp.uint8`. Origin is top-left (OpenGL's bottom-left is flipped).
        """

        gl = RendererGL.gl
        w, h = self.renderer._screen_width, self.renderer._screen_height

        # Lazy initialization of PBO (Pixel Buffer Object).
        if self._pbo is None:
            pbo_id = (gl.GLuint * 1)()
            gl.glGenBuffers(1, pbo_id)
            self._pbo = pbo_id[0]

            # Allocate PBO storage.
            gl.glBindBuffer(gl.GL_PIXEL_PACK_BUFFER, self._pbo)
            gl.glBufferData(gl.GL_PIXEL_PACK_BUFFER, gl.GLsizeiptr(w * h * 3), None, gl.GL_STREAM_READ)
            gl.glBindBuffer(gl.GL_PIXEL_PACK_BUFFER, 0)

            # Register with CUDA.
            self._wp_pbo = wp.RegisteredGLBuffer(
                gl_buffer_id=int(self._pbo),
                device=self.device,
                flags=wp.RegisteredGLBuffer.READ_ONLY,
            )

            # Set alignment once.
            gl.glPixelStorei(gl.GL_PACK_ALIGNMENT, 1)

        # GPU-to-GPU readback into PBO.
        assert self.renderer._frame_fbo is not None
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.renderer._frame_fbo)
        gl.glBindBuffer(gl.GL_PIXEL_PACK_BUFFER, self._pbo)

        if render_ui and self.ui:
            self.ui.begin_frame()
            self._render_ui()
            self.ui.end_frame()
            self.ui.render()

        gl.glReadPixels(0, 0, w, h, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, ctypes.c_void_p(0))
        gl.glBindBuffer(gl.GL_PIXEL_PACK_BUFFER, 0)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)

        # Map PBO buffer and copy using RGB kernel.
        assert self._wp_pbo is not None
        buf = self._wp_pbo.map(dtype=wp.uint8, shape=(w * h * 3,))

        if target_image is None:
            target_image = wp.empty(
                shape=(h, w, 3),
                dtype=wp.uint8,  # pyright: ignore[reportArgumentType]
                device=self.device,
            )

        if target_image.shape != (h, w, 3):
            raise ValueError(f"Shape of `target_image` must be ({h}, {w}, 3), got {target_image.shape}")

        # Launch the RGB kernel.
        wp.launch(
            copy_rgb_frame_uint8,
            dim=(w, h),
            inputs=[buf, w, h],
            outputs=[target_image],
            device=self.device,
        )

        # Unmap the PBO buffer.
        self._wp_pbo.unmap()

        return target_image

    @override
    def is_running(self) -> bool:
        """
        Check if the viewer is still running.

        Returns:
            bool: True if the window is open, False if closed.
        """
        return not self.renderer.has_exit()

    @override
    def is_paused(self) -> bool:
        """
        Check if the simulation is paused.

        Returns:
            bool: True if paused, False otherwise.
        """
        return self._paused

    @override
    def close(self):
        """
        Close the viewer and clean up resources.
        """
        self.renderer.close()

    @property
    def vsync(self) -> bool:
        """
        Get the current vsync state.

        Returns:
            bool: True if vsync is enabled, False otherwise.
        """
        return self.renderer.get_vsync()

    @vsync.setter
    def vsync(self, enabled: bool):
        """
        Set the vsync state.

        Args:
            enabled (bool): Enable or disable vsync.
        """
        self.renderer.set_vsync(enabled)

    @override
    def is_key_down(self, key):
        """
        Check if a key is currently pressed.

        Args:
            key: Either a string representing a character/key name, or an int
                 representing a pyglet key constant.

                 String examples: 'w', 'a', 's', 'd', 'space', 'escape', 'enter'
                 Int examples: pyglet.window.key.W, pyglet.window.key.SPACE

        Returns:
            bool: True if the key is currently pressed, False otherwise.
        """
        try:
            import pyglet
        except Exception:
            return False

        if isinstance(key, str):
            # Convert string to pyglet key constant
            key = key.lower()

            # Handle single characters
            if len(key) == 1 and key.isalpha():
                key_code = getattr(pyglet.window.key, key.upper(), None)
            elif len(key) == 1 and key.isdigit():
                key_code = getattr(pyglet.window.key, f"_{key}", None)
            else:
                # Handle special key names
                special_keys = {
                    "space": pyglet.window.key.SPACE,
                    "escape": pyglet.window.key.ESCAPE,
                    "esc": pyglet.window.key.ESCAPE,
                    "enter": pyglet.window.key.ENTER,
                    "return": pyglet.window.key.ENTER,
                    "tab": pyglet.window.key.TAB,
                    "shift": pyglet.window.key.LSHIFT,
                    "ctrl": pyglet.window.key.LCTRL,
                    "alt": pyglet.window.key.LALT,
                    "up": pyglet.window.key.UP,
                    "down": pyglet.window.key.DOWN,
                    "left": pyglet.window.key.LEFT,
                    "right": pyglet.window.key.RIGHT,
                    "backspace": pyglet.window.key.BACKSPACE,
                    "delete": pyglet.window.key.DELETE,
                }
                key_code = special_keys.get(key, None)

            if key_code is None:
                return False
        else:
            # Assume it's already a pyglet key constant
            key_code = key

        return self.renderer.is_key_down(key_code)

    # events

    def on_mouse_scroll(self, x, y, scroll_x, scroll_y):
        """
        Handle mouse scroll for zooming (FOV adjustment).

        Args:
            x, y: Mouse position.
            scroll_x, scroll_y: Scroll deltas.
        """
        if self.ui and self.ui.is_capturing():
            return

        fov_delta = scroll_y * 2.0
        self.camera.fov -= fov_delta
        self.camera.fov = max(min(self.camera.fov, 90.0), 15.0)

    def _to_framebuffer_coords(self, x: float, y: float) -> tuple[float, float]:
        """Convert window coordinates to framebuffer coordinates."""
        fb_w, fb_h = self.renderer.window.get_framebuffer_size()
        win_w, win_h = self.renderer.window.get_size()
        if win_w <= 0 or win_h <= 0:
            return float(x), float(y)
        scale_x = fb_w / win_w
        scale_y = fb_h / win_h
        return float(x) * scale_x, float(y) * scale_y

    def on_mouse_press(self, x, y, button, modifiers):
        """
        Handle mouse press events (object picking).

        Args:
            x, y: Mouse position.
            button: Mouse button pressed.
            modifiers: Modifier keys.
        """
        if self.ui and self.ui.is_capturing():
            return

        import pyglet

        # Handle right-click for picking
        if button == pyglet.window.mouse.RIGHT and self.picking_enabled:
            fb_x, fb_y = self._to_framebuffer_coords(x, y)
            ray_start, ray_dir = self.camera.get_world_ray(fb_x, fb_y)
            if self._last_state is not None:
                self.picking.pick(self._last_state, ray_start, ray_dir)

    def on_mouse_release(self, x, y, button, modifiers):
        """
        Handle mouse release events to stop dragging.

        Args:
            x, y: Mouse position.
            button: Mouse button released.
            modifiers: Modifier keys.
        """
        self.picking.release()

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        """
        Handle mouse drag events for camera and picking.

        Args:
            x, y: Mouse position.
            dx, dy: Mouse movement deltas.
            buttons: Mouse buttons pressed.
            modifiers: Modifier keys.
        """
        if self.ui and self.ui.is_capturing():
            return

        import pyglet

        if buttons & pyglet.window.mouse.LEFT:
            sensitivity = 0.1
            dx *= sensitivity
            dy *= sensitivity

            # Map screen-space right drag to a right turn (clockwise),
            # independent of world up-axis convention.
            self.camera.yaw -= dx
            self.camera.pitch += dy

        if buttons & pyglet.window.mouse.RIGHT and self.picking_enabled:
            fb_x, fb_y = self._to_framebuffer_coords(x, y)
            ray_start, ray_dir = self.camera.get_world_ray(fb_x, fb_y)

            if self.picking.is_picking():
                self.picking.update(ray_start, ray_dir)

    def on_mouse_motion(self, x, y, dx, dy):
        """
        Handle mouse motion events (not used).
        """
        pass

    def on_key_press(self, symbol, modifiers):
        """
        Handle key press events for UI and simulation control.

        Args:
            symbol: Key symbol.
            modifiers: Modifier keys.
        """
        if self.ui and self.ui.is_capturing():
            return

        try:
            import pyglet
        except Exception:
            return

        if symbol == pyglet.window.key.H:
            self.show_ui = not self.show_ui
        elif symbol == pyglet.window.key.SPACE:
            # Toggle pause with space key
            self._paused = not self._paused
        elif symbol == pyglet.window.key.F:
            # Frame camera around model bounds
            self._frame_camera_on_model()
        elif symbol == pyglet.window.key.ESCAPE:
            # Exit with Escape key
            self.renderer.close()

    def on_key_release(self, symbol, modifiers):
        """
        Handle key release events (not used).
        """
        pass

    def _frame_camera_on_model(self):
        """
        Frame the camera to show all visible objects in the scene.
        """
        if self.model is None:
            return

        # Compute bounds from all visible objects
        min_bounds = np.array([float("inf")] * 3)
        max_bounds = np.array([float("-inf")] * 3)
        found_objects = False

        # Check body positions if available
        if hasattr(self, "_last_state") and self._last_state is not None:
            if hasattr(self._last_state, "body_q") and self._last_state.body_q is not None:
                body_q = self._last_state.body_q.numpy()
                # body_q is an array of transforms (7 values: 3 pos + 4 quat)
                # Extract positions (first 3 values of each transform)
                for i in range(len(body_q)):
                    pos = body_q[i, :3]
                    min_bounds = np.minimum(min_bounds, pos)
                    max_bounds = np.maximum(max_bounds, pos)
                    found_objects = True

        # If no objects found, use default bounds
        if not found_objects:
            min_bounds = np.array([-5.0, -5.0, -5.0])
            max_bounds = np.array([5.0, 5.0, 5.0])

        # Calculate center and size of bounding box
        center = (min_bounds + max_bounds) * 0.5
        size = max_bounds - min_bounds
        max_extent = np.max(size)

        # Ensure minimum size to avoid camera being too close
        if max_extent < 1.0:
            max_extent = 1.0

        # Calculate camera distance based on field of view
        # Distance = extent / tan(fov/2) with some padding
        fov_rad = np.radians(self.camera.fov)
        padding = 1.5
        distance = max_extent / (2.0 * np.tan(fov_rad / 2.0)) * padding

        # Position camera at distance from current viewing direction, looking at center
        from pyglet.math import Vec3 as PyVec3

        front = self.camera.get_front()
        new_pos = PyVec3(
            center[0] - front.x * distance,
            center[1] - front.y * distance,
            center[2] - front.z * distance,
        )
        self.camera.pos = new_pos

    def _update_camera(self, dt: float):
        """
        Update the camera position and orientation based on user input.

        Args:
            dt (float): Time delta since last update.
        """
        if self.ui and self.ui.is_capturing():
            return

        # camera-relative basis
        forward = np.array(self.camera.get_front(), dtype=np.float32)
        right = np.array(self.camera.get_right(), dtype=np.float32)
        up = np.array(self.camera.get_up(), dtype=np.float32)

        # keep motion in the horizontal plane
        forward -= up * float(np.dot(forward, up))
        right -= up * float(np.dot(right, up))
        # renormalize
        fn = float(np.linalg.norm(forward))
        ln = float(np.linalg.norm(right))
        if fn > 1.0e-6:
            forward /= fn
        if ln > 1.0e-6:
            right /= ln

        import pyglet

        desired = np.zeros(3, dtype=np.float32)
        if self.renderer.is_key_down(pyglet.window.key.W) or self.renderer.is_key_down(pyglet.window.key.UP):
            desired += forward
        if self.renderer.is_key_down(pyglet.window.key.S) or self.renderer.is_key_down(pyglet.window.key.DOWN):
            desired -= forward
        if self.renderer.is_key_down(pyglet.window.key.A) or self.renderer.is_key_down(pyglet.window.key.LEFT):
            desired -= right  # strafe left
        if self.renderer.is_key_down(pyglet.window.key.D) or self.renderer.is_key_down(pyglet.window.key.RIGHT):
            desired += right  # strafe right
        if self.renderer.is_key_down(pyglet.window.key.Q):
            desired -= up  # pan down
        if self.renderer.is_key_down(pyglet.window.key.E):
            desired += up  # pan up

        dn = float(np.linalg.norm(desired))
        if dn > 1.0e-6:
            desired = desired / dn * self._cam_speed
        else:
            desired[:] = 0.0

        tau = max(1.0e-4, float(self._cam_damp_tau))
        self._cam_vel += (desired - self._cam_vel) * (dt / tau)

        # integrate position
        dv = type(self.camera.pos)(*self._cam_vel)
        self.camera.pos += dv * dt

    def on_resize(self, width, height):
        """
        Handle window resize events.

        Args:
            width (int): New window width.
            height (int): New window height.
        """
        fb_w, fb_h = self.renderer.window.get_framebuffer_size()
        self.camera.update_screen_size(fb_w, fb_h)
        self._invalidate_pbo()

        if self.ui:
            self.ui.resize(width, height)

    def _update_fps(self):
        """
        Update FPS calculation and statistics.
        """
        current_time = time.perf_counter()
        self._frame_count += 1

        # Update FPS every second
        if current_time - self._last_fps_time >= 1.0:
            time_delta = current_time - self._last_fps_time
            self._current_fps = self._frame_count / time_delta
            self._fps_history.append(self._current_fps)

            # Keep only last 60 FPS readings
            if len(self._fps_history) > 60:
                self._fps_history.pop(0)

            self._last_fps_time = current_time
            self._frame_count = 0

    def _render_gizmos(self):
        if not self._gizmo_log or not self.ui:
            return

        giz = self.ui.giz
        io = self.ui.io

        # Setup ImGuizmo viewport
        giz.set_orthographic(False)
        giz.set_rect(0.0, 0.0, float(io.display_size[0]), float(io.display_size[1]))
        giz.set_gizmo_size_clip_space(0.07)
        giz.set_axis_limit(0.0)
        giz.set_plane_limit(0.0)

        # Camera matrices
        view = self.camera.get_view_matrix().reshape(4, 4).transpose()
        proj = self.camera.get_projection_matrix().reshape(4, 4).transpose()

        # Draw & mutate each gizmo
        for gid, transform in self._gizmo_log.items():
            giz.push_id(str(gid))

            M = wp.transform_to_matrix(transform)

            def m44_to_mat16(m):
                """Row-major 4x4 -> giz.Matrix16 (column-major, 16 floats)."""
                m = np.asarray(m, dtype=np.float32).reshape(4, 4)
                return giz.Matrix16(m.flatten(order="F").tolist())

            view_ = m44_to_mat16(view)
            proj_ = m44_to_mat16(proj)
            M_ = m44_to_mat16(M)

            giz.manipulate(view_, proj_, giz.OPERATION.rotate, giz.MODE.world, M_, None, None)
            giz.manipulate(view_, proj_, giz.OPERATION.translate, giz.MODE.world, M_, None, None)

            M[:] = M_.values.reshape(4, 4, order="F")
            transform[:] = wp.transform_from_matrix(M)

            giz.pop_id()

    def _render_ui(self):
        """
        Render the complete ImGui interface (left panel, stats overlay, and custom UI).
        """
        if not self.ui or not self.ui.is_available:
            return

        # Render gizmos
        self._render_gizmos()

        # Render left panel
        self._render_left_panel()

        # Render top-right stats overlay
        self._render_stats_overlay()

        # allow users to create custom windows
        for callback in self._ui_callbacks["free"]:
            callback(self.ui.imgui)

    def _render_left_panel(self):
        """
        Render the left panel with model info and visualization controls.
        """
        imgui = self.ui.imgui

        # Use theme colors directly
        nav_highlight_color = self.ui.get_theme_color(imgui.Col_.nav_cursor, (1.0, 1.0, 1.0, 1.0))

        # Position the window on the left side
        io = self.ui.io
        imgui.set_next_window_pos(imgui.ImVec2(10, 10))
        imgui.set_next_window_size(imgui.ImVec2(300, io.display_size[1] - 20))

        # Main control panel window - use safe flag values
        flags = imgui.WindowFlags_.no_resize.value

        if imgui.begin(f"Newton Viewer v{nt.__version__}", flags=flags):
            imgui.separator()

            # Collapsing headers default-open handling (first frame only)
            header_flags = 0

            # Model Information section
            if self.model is not None:
                imgui.set_next_item_open(True, imgui.Cond_.appearing)
                if imgui.collapsing_header("Model Information", flags=header_flags):
                    imgui.separator()
                    imgui.text(f"Worlds: {self.model.world_count}")
                    axis_names = ["X", "Y", "Z"]
                    imgui.text(f"Up Axis: {axis_names[self.model.up_axis]}")
                    gravity = self.model.gravity.numpy()[0]
                    gravity_text = f"Gravity: ({gravity[0]:.2f}, {gravity[1]:.2f}, {gravity[2]:.2f})"
                    imgui.text(gravity_text)

                    # Pause simulation checkbox
                    changed, self._paused = imgui.checkbox("Pause", self._paused)

                # Visualization Controls section
                imgui.set_next_item_open(True, imgui.Cond_.appearing)
                if imgui.collapsing_header("Visualization", flags=header_flags):
                    imgui.separator()

                    # Joint visualization
                    show_joints = self.show_joints
                    changed, self.show_joints = imgui.checkbox("Show Joints", show_joints)

                    # Contact visualization
                    show_contacts = self.show_contacts
                    changed, self.show_contacts = imgui.checkbox("Show Contacts", show_contacts)

                    # Particle visualization
                    show_particles = self.show_particles
                    changed, self.show_particles = imgui.checkbox("Show Particles", show_particles)

                    # Spring visualization
                    show_springs = self.show_springs
                    changed, self.show_springs = imgui.checkbox("Show Springs", show_springs)

                    # Center of mass visualization
                    show_com = self.show_com
                    changed, self.show_com = imgui.checkbox("Show Center of Mass", show_com)

                    # Triangle mesh visualization
                    show_triangles = self.show_triangles
                    changed, self.show_triangles = imgui.checkbox("Show Cloth", show_triangles)

                    # Collision geometry toggle
                    show_collision = self.show_collision
                    changed, self.show_collision = imgui.checkbox("Show Collision", show_collision)

                    # Visual geometry toggle
                    show_visual = self.show_visual
                    changed, self.show_visual = imgui.checkbox("Show Visual", show_visual)

                    # Inertia boxes toggle
                    show_inertia_boxes = self.show_inertia_boxes
                    changed, self.show_inertia_boxes = imgui.checkbox("Show Inertia Boxes", show_inertia_boxes)

            imgui.set_next_item_open(True, imgui.Cond_.appearing)
            if imgui.collapsing_header("Example Options"):
                # Render UI callbacks for side panel
                for callback in self._ui_callbacks["side"]:
                    callback(self.ui.imgui)

            # Rendering Options section
            imgui.set_next_item_open(True, imgui.Cond_.appearing)
            if imgui.collapsing_header("Rendering Options"):
                imgui.separator()

                # VSync
                changed, vsync = imgui.checkbox("VSync", self.vsync)
                if changed:
                    self.vsync = vsync

                # Sky rendering
                changed, self.renderer.draw_sky = imgui.checkbox("Sky", self.renderer.draw_sky)

                # Shadow rendering
                changed, self.renderer.draw_shadows = imgui.checkbox("Shadows", self.renderer.draw_shadows)

                # Wireframe mode
                changed, self.renderer.draw_wireframe = imgui.checkbox("Wireframe", self.renderer.draw_wireframe)

                # Light color
                changed, self.renderer._light_color = imgui.color_edit3("Light Color", self.renderer._light_color)
                # Sky color
                changed, self.renderer.sky_upper = imgui.color_edit3("Sky Color", self.renderer.sky_upper)
                # Ground color
                changed, self.renderer.sky_lower = imgui.color_edit3("Ground Color", self.renderer.sky_lower)

            # Wind Effects section
            imgui.set_next_item_open(False, imgui.Cond_.once)
            if imgui.collapsing_header("Wind"):
                imgui.separator()

                # Wind amplitude slider
                changed, amplitude = imgui.slider_float("Wind Amplitude", self.wind.amplitude, -2.0, 2.0, "%.2f")
                if changed:
                    self.wind.amplitude = amplitude

                # Wind period slider
                changed, period = imgui.slider_float("Wind Period", self.wind.period, 1.0, 30.0, "%.2f")
                if changed:
                    self.wind.period = period

                # Wind frequency slider
                changed, frequency = imgui.slider_float("Wind Frequency", self.wind.frequency, 0.1, 5.0, "%.2f")
                if changed:
                    self.wind.frequency = frequency

                # Wind direction sliders
                direction = [self.wind.direction[0], self.wind.direction[1], self.wind.direction[2]]
                changed, direction = imgui.slider_float3("Wind Direction", direction, -1.0, 1.0, "%.2f")
                if changed:
                    self.wind.direction = direction

            # Camera Information section
            imgui.set_next_item_open(True, imgui.Cond_.appearing)
            if imgui.collapsing_header("Camera"):
                imgui.separator()

                pos = self.camera.pos
                pos_text = f"Position: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})"
                imgui.text(pos_text)
                imgui.text(f"FOV: {self.camera.fov:.1f}°")
                imgui.text(f"Pitch: {self.camera.pitch:.1f}°")
                imgui.text(f"Yaw: {self.camera.yaw:.1f}°")

                # Camera controls hint
                imgui.separator()
                imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(*nav_highlight_color))
                imgui.text("Controls:")
                imgui.pop_style_color()
                imgui.text("WASD - Move camera")
                imgui.text("QE - Pan up/down")
                imgui.text("Left Click - Look around")
                imgui.text("Right Click - Pick objects")
                imgui.text("Scroll - Zoom")
                imgui.text("Space - Pause/Resume")
                imgui.text("H - Toggle UI")
                imgui.text("F - Frame camera around model")

            # Selection API section
            self._render_selection_panel()

        imgui.end()

    def _render_stats_overlay(self):
        """
        Render performance stats overlay in the top-right corner.
        """
        imgui = self.ui.imgui
        io = self.ui.io

        # Use fallback color for FPS display
        fps_color = (1.0, 1.0, 1.0, 1.0)  # Bright white

        # Position in top-right corner
        window_pos = (io.display_size[0] - 10, 10)
        imgui.set_next_window_pos(imgui.ImVec2(window_pos[0], window_pos[1]), pivot=imgui.ImVec2(1.0, 0.0))

        # Transparent background, auto-sized, non-resizable/movable - use safe flags
        #        try:
        flags: imgui.WindowFlags = (
            imgui.WindowFlags_.no_decoration.value
            | imgui.WindowFlags_.always_auto_resize.value
            | imgui.WindowFlags_.no_resize.value
            | imgui.WindowFlags_.no_saved_settings.value
            | imgui.WindowFlags_.no_focus_on_appearing.value
            | imgui.WindowFlags_.no_nav.value
            | imgui.WindowFlags_.no_move.value
        )

        # Set semi-transparent background for the overlay window
        pushed_window_bg = False
        try:
            # Preferred API name in pyimgui
            imgui.set_next_window_bg_alpha(0.7)
        except AttributeError:
            # Fallback: temporarily override window bg color alpha
            try:
                style = imgui.get_style()
                bg = style.color_(imgui.Col_.window_bg)
                r, g, b = bg.x, bg.y, bg.z
            except Exception:
                # Reasonable dark default
                r, g, b = 0.094, 0.094, 0.094
            imgui.push_style_color(imgui.Col_.window_bg, imgui.ImVec4(r, g, b, 0.7))
            pushed_window_bg = True

        if imgui.begin("Performance Stats", flags=flags):
            # FPS display
            fps_text = f"FPS: {self._current_fps:.1f}"
            imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(*fps_color))
            imgui.text(fps_text)
            imgui.pop_style_color()

            # Model stats
            if self.model is not None:
                imgui.separator()
                imgui.text(f"Bodies: {self.model.body_count}")
                imgui.text(f"Shapes: {self.model.shape_count}")
                imgui.text(f"Joints: {self.model.joint_count}")
                imgui.text(f"Particles: {self.model.particle_count}")
                imgui.text(f"Springs: {self.model.spring_count}")
                imgui.text(f"Triangles: {self.model.tri_count}")
                imgui.text(f"Edges: {self.model.edge_count}")
                imgui.text(f"Tetrahedra: {self.model.tet_count}")

            # Rendered objects count
            imgui.separator()
            imgui.text(f"Unique Objects: {len(self.objects)}")

        # Custom stats
        for callback in self._ui_callbacks["stats"]:
            callback(self.ui.imgui)

        imgui.end()

        # Restore bg color if we pushed it
        if pushed_window_bg:
            imgui.pop_style_color()

    def _render_selection_panel(self):
        """
        Render the selection panel for Newton Model introspection.
        """
        imgui = self.ui.imgui

        # Selection Panel section
        header_flags = 0
        imgui.set_next_item_open(False, imgui.Cond_.appearing)  # Default to closed
        if imgui.collapsing_header("Selection API", flags=header_flags):
            imgui.separator()

            # Check if we have state data available
            if self._last_state is None:
                imgui.text("No state data available.")
                imgui.text("Start simulation to enable selection.")
                return

            state = self._selection_ui_state

            # Display error message if any
            if state["error_message"]:
                imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(1.0, 0.3, 0.3, 1.0))
                imgui.text(f"Error: {state['error_message']}")
                imgui.pop_style_color()
                imgui.separator()

            # Articulation Pattern Input
            imgui.text("Articulation Pattern:")
            imgui.push_item_width(200)
            _changed, state["selected_articulation_pattern"] = imgui.input_text(
                "##pattern", state["selected_articulation_pattern"]
            )
            imgui.pop_item_width()
            if imgui.is_item_hovered():
                tooltip = "Pattern to match articulations (e.g., '*', 'robot*', 'cartpole')"
                imgui.set_tooltip(tooltip)

            # Joint filtering
            imgui.spacing()
            imgui.text("Joint Filters (optional):")
            imgui.push_item_width(150)
            imgui.text("Include:")
            imgui.same_line()
            _, state["include_joints"] = imgui.input_text("##inc_joints", state["include_joints"])
            if imgui.is_item_hovered():
                imgui.set_tooltip("Comma-separated joint names/patterns")

            imgui.text("Exclude:")
            imgui.same_line()
            _, state["exclude_joints"] = imgui.input_text("##exc_joints", state["exclude_joints"])
            if imgui.is_item_hovered():
                imgui.set_tooltip("Comma-separated joint names/patterns")
            imgui.pop_item_width()

            # Link filtering
            imgui.spacing()
            imgui.text("Link Filters (optional):")
            imgui.push_item_width(150)
            imgui.text("Include:")
            imgui.same_line()
            _, state["include_links"] = imgui.input_text("##inc_links", state["include_links"])
            if imgui.is_item_hovered():
                imgui.set_tooltip("Comma-separated link names/patterns")

            imgui.text("Exclude:")
            imgui.same_line()
            _, state["exclude_links"] = imgui.input_text("##exc_links", state["exclude_links"])
            if imgui.is_item_hovered():
                imgui.set_tooltip("Comma-separated link names/patterns")
            imgui.pop_item_width()

            # Create View Button
            imgui.spacing()
            if imgui.button("Create Articulation View"):
                self._create_articulation_view()

            # Show view info if created
            if state["selected_articulation_view"] is not None:
                view = state["selected_articulation_view"]
                imgui.separator()
                imgui.text(f"  Count: {view.count}")
                imgui.text(f"  Joints: {view.joint_count}")
                imgui.text(f"  Links: {view.link_count}")
                imgui.text(f"  DOFs: {view.joint_dof_count}")
                imgui.text(f"  Fixed base: {view.is_fixed_base}")
                imgui.text(f"  Floating base: {view.is_floating_base}")

                # Attribute selector
                imgui.spacing()
                imgui.text("Select Attribute:")
                imgui.push_item_width(150)
                if state["selected_attribute"] in state["attribute_options"]:
                    current_attr_idx = state["attribute_options"].index(state["selected_attribute"])
                else:
                    current_attr_idx = 0
                _, new_attr_idx = imgui.combo("##attribute", current_attr_idx, state["attribute_options"])
                state["selected_attribute"] = state["attribute_options"][new_attr_idx]
                imgui.pop_item_width()

                # Toggle values display
                _, state["show_values"] = imgui.checkbox("Show Values", state["show_values"])

                # Display attribute values if requested
                if state["show_values"]:
                    self._render_attribute_values(view, state["selected_attribute"])

    def _create_articulation_view(self):
        """
        Create an ArticulationView based on current UI state.
        """
        state = self._selection_ui_state

        try:
            # Clear any previous error
            state["error_message"] = ""

            # Parse filter strings
            if state["include_joints"]:
                include_joints = [j.strip() for j in state["include_joints"].split(",") if j.strip()]
            else:
                include_joints = None

            if state["exclude_joints"]:
                exclude_joints = [j.strip() for j in state["exclude_joints"].split(",") if j.strip()]
            else:
                exclude_joints = None

            if state["include_links"]:
                include_links = [link.strip() for link in state["include_links"].split(",") if link.strip()]
            else:
                include_links = None

            if state["exclude_links"]:
                exclude_links = [link.strip() for link in state["exclude_links"].split(",") if link.strip()]
            else:
                exclude_links = None

            # Create ArticulationView
            state["selected_articulation_view"] = ArticulationView(
                model=self.model,
                pattern=state["selected_articulation_pattern"],
                include_joints=include_joints,
                exclude_joints=exclude_joints,
                include_links=include_links,
                exclude_links=exclude_links,
                verbose=False,  # Don't print to console in UI
            )

        except Exception as e:
            state["error_message"] = str(e)
            state["selected_articulation_view"] = None

    def _render_attribute_values(self, view: ArticulationView, attribute_name: str):
        """
        Render the values of the selected attribute in the selection panel.

        Args:
            view (ArticulationView): The current articulation view.
            attribute_name (str): The attribute to display.
        """
        imgui = self.ui.imgui
        state = self._selection_ui_state

        try:
            # Determine source based on attribute
            if attribute_name.startswith("joint_f"):
                # Forces come from control
                if hasattr(self, "_last_control") and self._last_control is not None:
                    source = self._last_control
                else:
                    imgui.text("No control data available for forces")
                    return
            else:
                # Other attributes come from state or model
                source = self._last_state

            # Get the attribute values
            values = view.get_attribute(attribute_name, source).numpy()

            imgui.separator()
            imgui.text(f"Attribute: {attribute_name}")
            imgui.text(f"Shape: {values.shape}")
            imgui.text(f"Dtype: {values.dtype}")

            # Handle batch dimension selection for 2D arrays
            if len(values.shape) == 2:
                batch_size = values.shape[0]
                imgui.spacing()
                imgui.text("Batch/World Selection:")
                imgui.push_item_width(100)

                # Ensure selected batch index is valid
                state["selected_batch_idx"] = max(0, min(state["selected_batch_idx"], batch_size - 1))

                _, state["selected_batch_idx"] = imgui.slider_int(
                    "##batch", state["selected_batch_idx"], 0, batch_size - 1
                )
                imgui.pop_item_width()
                imgui.same_line()
                text = f"World {state['selected_batch_idx']} / {batch_size}"
                imgui.text(text)

            # Display values as sliders in a scrollable region
            imgui.spacing()
            imgui.text("Values:")

            # Create a child window for scrollable content
            if imgui.begin_child("values_scroll", 0, 300, border=True):
                if len(values.shape) == 1:
                    # 1D array - show as sliders with names if available
                    names = self._get_attribute_names(view, attribute_name)
                    self._render_value_sliders(values, names, attribute_name, state)

                elif len(values.shape) == 2:
                    # 2D array - show selected batch with joint names
                    batch_idx = state["selected_batch_idx"]
                    selected_batch = values[batch_idx]
                    names = self._get_attribute_names(view, attribute_name)
                    self._render_value_sliders(selected_batch, names, attribute_name, state)

                else:
                    # Higher dimensional - just show summary
                    shape_str = str(values.shape)
                    imgui.text(f"Multi-dimensional array with shape {shape_str}")

            imgui.end_child()

            # Show some statistics for numeric data
            if values.dtype.kind in "biufc":  # numeric types
                imgui.spacing()

                # Calculate stats on the selected batch for 2D arrays
                if len(values.shape) == 2:
                    batch_idx = state["selected_batch_idx"]
                    stats_data = values[batch_idx]
                    imgui.text(f"Statistics for World {batch_idx}:")
                else:
                    stats_data = values
                    imgui.text("Statistics:")

                imgui.text(f"  Min: {np.min(stats_data):.6f}")
                imgui.text(f"  Max: {np.max(stats_data):.6f}")
                imgui.text(f"  Mean: {np.mean(stats_data):.6f}")
                if stats_data.size > 1:
                    imgui.text(f"  Std: {np.std(stats_data):.6f}")

        except Exception as e:
            imgui.text(f"Error getting attribute: {e!s}")

    def _get_attribute_names(self, view: ArticulationView, attribute_name: str):
        """
        Get the names associated with an attribute (joint names, link names, etc.).

        Args:
            view (ArticulationView): The current articulation view.
            attribute_name (str): The attribute to get names for.

        Returns:
            list or None: List of names or None if not available.
        """
        try:
            if attribute_name.startswith("joint_q") or attribute_name.startswith("joint_f"):
                # For joint positions/velocities/forces, return DOF names or coord names
                if attribute_name == "joint_q":
                    return view.joint_coord_names
                else:  # joint_qd, joint_f
                    return view.joint_dof_names
            elif attribute_name.startswith("body_"):
                # For body attributes, return body/link names
                return view.body_names
            else:
                return None
        except Exception:
            return None

    def _render_value_sliders(self, values, names, attribute_name: str, state):
        """
        Render values as individual sliders for each DOF.

        Args:
            values: Array of values to display.
            names: List of names for each value.
            attribute_name (str): The attribute being displayed.
            state (dict): UI state dictionary.
        """
        imgui = self.ui.imgui

        # Determine appropriate slider ranges based on attribute type
        if attribute_name.startswith("joint_q"):
            # Joint positions - use reasonable angle/position ranges
            slider_min, slider_max = -3.14159, 3.14159  # Default to ±π
        elif attribute_name.startswith("joint_qd"):
            # Joint velocities - use reasonable velocity ranges
            slider_min, slider_max = -10.0, 10.0
        elif attribute_name.startswith("joint_f"):
            # Joint forces - use reasonable force ranges
            slider_min, slider_max = -100.0, 100.0
        else:
            # For other attributes, use data-driven ranges
            if len(values) > 0 and values.dtype.kind in "biufc":  # numeric
                val_min, val_max = float(np.min(values)), float(np.max(values))
                val_range = val_max - val_min
                if val_range < 1e-6:  # Nearly constant values
                    slider_min = val_min - 1.0
                    slider_max = val_max + 1.0
                else:
                    # Add 20% padding
                    padding = val_range * 0.2
                    slider_min = val_min - padding
                    slider_max = val_max + padding
            else:
                slider_min, slider_max = -1.0, 1.0

        # Initialize slider state if needed
        if "slider_values" not in state:
            state["slider_values"] = {}

        slider_key = f"{attribute_name}_sliders"
        if slider_key not in state["slider_values"]:
            state["slider_values"][slider_key] = [float(v) for v in values]

        # Ensure slider values array has correct length
        current_sliders = state["slider_values"][slider_key]
        while len(current_sliders) < len(values):
            current_sliders.append(0.0)
        while len(current_sliders) > len(values):
            current_sliders.pop()

        # Update slider values to match current data
        for i, val in enumerate(values):
            if i < len(current_sliders):
                current_sliders[i] = float(val)

        # Render sliders
        for i, val in enumerate(values):
            name = names[i] if names and i < len(names) else f"[{i}]"

            if isinstance(val, int | float) or hasattr(val, "dtype"):
                # shorten floating base key for ui
                # todo: consider doing this in the importers
                if name.startswith("floating_base"):
                    name = "base"

                # Truncate name for display but keep full name for tooltip
                display_name = name[:8] + "..." if len(name) > 8 else name
                # Pad display name to ensure consistent width
                display_name = f"{display_name:<11}"

                # Show truncated name with tooltip
                imgui.text(display_name)
                if imgui.is_item_hovered() and len(name) > 8:
                    imgui.set_tooltip(name)
                imgui.same_line()

                # Use slider for numeric values with fixed width
                imgui.push_item_width(150)
                slider_id = f"##{attribute_name}_{i}"
                _changed, _new_val = imgui.slider_float(slider_id, current_sliders[i], slider_min, slider_max, "%.6f")
                imgui.pop_item_width()
                # if changed:
                #     current_sliders[i] = new_val

            else:
                # For non-numeric values, just show as text
                imgui.text(f"{name}: {val}")
