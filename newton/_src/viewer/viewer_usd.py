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

import numpy as np
import warp as wp

from ..core.types import override

try:
    from pxr import Gf, Sdf, Usd, UsdGeom, Vt
except ImportError:
    Gf = Sdf = Usd = UsdGeom = Vt = None

from .viewer import ViewerBase


# transforms a cylinder such that it connects the two points pos0, pos1
def _compute_segment_xform(pos0, pos1):
    mid = (pos0 + pos1) * 0.5
    height = (pos1 - pos0).GetLength()

    dir = (pos1 - pos0) / height

    rot = Gf.Rotation()
    rot.SetRotateInto((0.0, 0.0, 1.0), Gf.Vec3d(dir))

    scale = Gf.Vec3f(1.0, 1.0, height)

    return (mid, Gf.Quath(rot.GetQuat()), scale)


def _usd_add_xform(prim):
    prim = UsdGeom.Xform(prim)
    prim.ClearXformOpOrder()

    prim.AddTranslateOp()
    prim.AddOrientOp()
    prim.AddScaleOp()


def _usd_set_xform(
    xform,
    pos: tuple | None = None,
    rot: tuple | None = None,
    scale: tuple | None = None,
    time: float = 0.0,
):
    xform = UsdGeom.Xform(xform)

    xform_ops = xform.GetOrderedXformOps()

    if pos is not None:
        xform_ops[0].Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])), time)
    if rot is not None:
        xform_ops[1].Set(Gf.Quatf(float(rot[3]), float(rot[0]), float(rot[1]), float(rot[2])), time)
    if scale is not None:
        xform_ops[2].Set(Gf.Vec3d(float(scale[0]), float(scale[1]), float(scale[2])), time)


class ViewerUSD(ViewerBase):
    """
    USD viewer backend for Newton physics simulations.

    This backend creates a USD stage and manages mesh prototypes and instanced rendering
    using PointInstancers. It supports time-sampled transforms for efficient playback
    and visualization of simulation data.
    """

    def __init__(self, output_path, fps=60, up_axis="Z", num_frames=100, scaling=1.0):
        """
        Initialize the USD viewer backend for Newton physics simulations.

        Args:
            output_path (str): Path to the output USD file.
            fps (int, optional): Frames per second for time sampling. Default is 60.
            up_axis (str, optional): USD up axis, either 'Y' or 'Z'. Default is 'Z'.
            num_frames (int, optional): Maximum number of frames to record. Default is 100. If None, recording is unlimited.
            scaling (float, optional): Uniform scaling applied to the scene root. Default is 1.0.

        Raises:
            ImportError: If the usd-core package is not installed.
        """
        if Usd is None:
            raise ImportError("usd-core package is required for ViewerUSD. Install with: pip install usd-core")

        super().__init__()

        self.output_path = output_path
        self.fps = fps
        self.up_axis = up_axis
        self.num_frames = num_frames

        # Create USD stage
        self.stage = Usd.Stage.CreateNew(output_path)
        self.stage.SetTimeCodesPerSecond(fps)  # number of timeCodes per second for data storage
        self.stage.SetFramesPerSecond(fps)  # display frame rate (timeline FPS in DCC tools)
        self.stage.SetStartTimeCode(0)

        axis_token = {
            "X": UsdGeom.Tokens.x,
            "Y": UsdGeom.Tokens.y,
            "Z": UsdGeom.Tokens.z,
        }.get(self.up_axis.strip().upper())

        UsdGeom.SetStageUpAxis(self.stage, axis_token)
        UsdGeom.SetStageMetersPerUnit(self.stage, 1.0)

        self.root = UsdGeom.Xform.Define(self.stage, "/root")

        # apply root scaling
        self.root.ClearXformOpOrder()
        s = self.root.AddScaleOp()
        s.Set(Gf.Vec3d(float(scaling), float(scaling), float(scaling)), 0.0)

        self.stage.SetDefaultPrim(self.root.GetPrim())

        # Track meshes and instancers
        self._meshes = {}  # mesh_name -> prototype_path
        self._instancers = {}  # instancer_name -> UsdGeomPointInstancer
        self._points = {}  # point_name -> UsdGeomPoints

        # Track current frame
        self._frame_index = 0
        self._frame_count = 0

        self.set_model(None)

    @override
    def begin_frame(self, time):
        """
        Begin a new frame at the given simulation time.

        Parameters:
            time (float): The simulation time for the new frame.
        """
        super().begin_frame(time)
        self._frame_index = int(time * self.fps)
        self._frame_count += 1

        # Update stage end time if needed
        if self._frame_index > self.stage.GetEndTimeCode():
            self.stage.SetEndTimeCode(self._frame_index)

    @override
    def end_frame(self):
        """
        End the current frame.

        This method is a placeholder for any end-of-frame logic required by the backend.
        """
        pass

    @override
    def is_running(self):
        """
        Check if the viewer is still running.

        Returns:
            bool: False if the frame limit is exceeded, True otherwise.
        """
        if self.num_frames is not None:
            return self._frame_count < self.num_frames
        return True

    @override
    def close(self):
        """
        Finalize and save the USD stage.

        This should be called when all logging is complete to ensure the USD file is written.
        """
        self.stage.GetRootLayer().Save()
        self.stage = None

        if self.output_path:
            print(f"USD output saved in: {os.path.abspath(self.output_path)}")

    def _get_path(self, name):
        # Handle both absolute and relative paths correctly
        if name.startswith("/"):
            return "/root" + name
        else:
            return "/root/" + name

    @override
    def log_mesh(
        self,
        name,
        points: wp.array,
        indices: wp.array,
        normals: wp.array | None = None,
        uvs: wp.array | None = None,
        texture=None,
        hidden=False,
        backface_culling=True,
    ):
        """
        Create a USD mesh prototype from vertex and index data.

        Parameters:
            name (str): Mesh name or Sdf.Path string.
            points (wp.array): Vertex positions as a warp array of wp.vec3.
            indices (wp.array): Triangle indices as a warp array of wp.uint32.
            normals (wp.array, optional): Vertex normals as a warp array of wp.vec3.
            uvs (wp.array, optional): UV coordinates as a warp array of wp.vec2.
            hidden (bool, optional): If True, mesh will be hidden. Default is False.
            backface_culling (bool, optional): If True, enable backface culling. Default is True.

        Returns:
            str: The mesh prototype path.
        """

        # Convert warp arrays to numpy
        points_np = points.numpy().astype(np.float32)
        indices_np = indices.numpy().astype(np.uint32)

        if name not in self._meshes:
            self._ensure_scopes_for_path(self.stage, self._get_path(name))

            mesh_prim = UsdGeom.Mesh.Define(self.stage, self._get_path(name))

            # setup topology once (do not set every frame)
            face_vertex_counts = [3] * (len(indices_np) // 3)
            mesh_prim.GetFaceVertexCountsAttr().Set(face_vertex_counts)
            mesh_prim.GetFaceVertexIndicesAttr().Set(indices_np)

            # Store the prototype path
            self._meshes[name] = mesh_prim

        mesh_prim = self._meshes[name]
        mesh_prim.GetPointsAttr().Set(points_np, self._frame_index)

        # Set normals if provided
        if normals is not None:
            normals_np = normals.numpy().astype(np.float32)
            mesh_prim.GetNormalsAttr().Set(normals_np, self._frame_index)
            mesh_prim.SetNormalsInterpolation(UsdGeom.Tokens.vertex)

        # Set UVs if provided (simplified for now)
        if uvs is not None:
            # TODO: Implement UV support for USD meshes
            pass

        # how to hide the prototype mesh but not the instances in USD?
        mesh_prim.GetVisibilityAttr().Set("inherited" if not hidden else "invisible", self._frame_index)

    # log a set of instances as individual mesh prims, slower but makes it easier
    # to do post-editing of instance materials etc. default for Newton shapes
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
        # Get prototype path
        if mesh not in self._meshes:
            msg = f"Mesh prototype '{mesh}' not found for log_instances(). Call log_mesh() first."
            raise RuntimeError(msg)

        self._ensure_scopes_for_path(self.stage, self._get_path(name) + "/scope")

        if xforms:
            xforms = xforms.numpy()

        if scales:
            scales = scales.numpy()
        else:
            scales = np.ones((len(xforms), 3), dtype=np.float32)

        if colors:
            colors = colors.numpy()

        for i in range(len(xforms)):
            instance_path = self._get_path(name) + f"/instance_{i}"
            instance = self.stage.GetPrimAtPath(instance_path)

            if not instance:
                instance = self.stage.DefinePrim(instance_path)
                instance.GetReferences().AddInternalReference(self._get_path(mesh))

                UsdGeom.Imageable(instance).GetVisibilityAttr().Set("inherited" if not hidden else "invisible")
                _usd_add_xform(instance)

            # update transform
            if xforms is not None:
                pos = xforms[i][:3]
                rot = xforms[i][3:7]

                _usd_set_xform(instance, pos, rot, scales[i], self._frame_index)

            # update color
            if colors is not None:
                displayColor = UsdGeom.PrimvarsAPI(instance).GetPrimvar("displayColor")
                displayColor.Set(colors[i], self._frame_index)

    # log a set of instances as a point instancer, faster but less flexible
    def log_instances_point_instancer(self, name, mesh, xforms, scales, colors, materials):
        """
        Create or update a PointInstancer for mesh instances.

        Parameters:
            name (str): Instancer name or Sdf.Path string.
            mesh (str): Mesh prototype name (must be previously logged).
            xforms (wp.array): Instance transforms as a warp array of wp.transform.
            scales (wp.array): Instance scales as a warp array of wp.vec3.
            colors (wp.array): Instance colors as a warp array of wp.vec3.
            materials (wp.array): Instance materials as a warp array of wp.vec4.

        Raises:
            RuntimeError: If the mesh prototype is not found.
        """
        # Get prototype path
        if mesh not in self._meshes:
            msg = f"Mesh prototype '{mesh}' not found for log_instances(). Call log_mesh() first."
            raise RuntimeError(msg)

        num_instances = len(xforms)

        # Create instancer if it doesn't exist
        if name not in self._instancers:
            self._ensure_scopes_for_path(self.stage, self._get_path(name))

            instancer = UsdGeom.PointInstancer.Define(self.stage, self._get_path(name))
            instancer.CreateIdsAttr().Set(list(range(num_instances)))
            instancer.CreateProtoIndicesAttr().Set([0] * num_instances)
            UsdGeom.PrimvarsAPI(instancer).CreatePrimvar(
                "displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.vertex, 1
            )

            # Set the prototype relationship
            instancer.GetPrototypesRel().AddTarget(self._get_path(mesh))

            self._instancers[name] = instancer

        instancer = self._instancers[name]

        # Convert transforms to USD format
        if xforms is not None:
            xforms_np = xforms.numpy()

            # Extract positions from warp transforms using vectorized operations
            # Warp transform format: [x, y, z, qx, qy, qz, qw]
            positions = xforms_np[:, :3].astype(np.float32)

            # Convert quaternion format: Warp (x, y, z, w) → USD (w, (x,y,z))
            # USD expects quaternions as Gf.Quath(real, imag_vec3)
            quat_w = xforms_np[:, 6].astype(np.float32)
            quat_xyz = xforms_np[:, 3:6].astype(np.float32)

            # Create orientations list with proper USD quaternion format
            orientations = []
            for i in range(num_instances):
                quat = Gf.Quath(
                    float(quat_w[i]), Gf.Vec3h(float(quat_xyz[i, 0]), float(quat_xyz[i, 1]), float(quat_xyz[i, 2]))
                )
                orientations.append(quat)

            # Handle scales with numpy operations
            if scales is None:
                scales = np.ones((num_instances, 3), dtype=np.float32)
            elif isinstance(scales, wp.array):
                scales = scales.numpy().astype(np.float32)

            # Set attributes at current time
            instancer.GetPositionsAttr().Set(positions, self._frame_index)
            instancer.GetOrientationsAttr().Set(orientations, self._frame_index)

            if scales is not None:
                instancer.GetScalesAttr().Set(scales, self._frame_index)

            if colors is not None:
                # Promote colors to proper numpy array format
                colors_np = self._promote_colors_to_array(colors, num_instances)

                # Set color per-instance
                displayColor = UsdGeom.PrimvarsAPI(instancer).GetPrimvar("displayColor")
                displayColor.Set(colors_np, self._frame_index)

                # Explicit identity indices [0, 1, 2, ...], otherwise OV won't pick them up
                indices = Vt.IntArray(range(num_instances))
                displayColor.SetIndices(indices, self._frame_index)

    # Abstract methods that need basic implementations
    @override
    def log_lines(self, name, starts, ends, colors, width: float = 0.01, hidden=False):
        """Debug helper to add a line list as a set of capsules

        Args:
            starts: The vertices of the lines (wp.array)
            ends: The vertices of the lines (wp.array)
            colors: The colors of the lines (wp.array)
            width: The width of the lines (float)
            hidden: Whether the lines are hidden (bool)
        """

        if name not in self._instancers:
            self._ensure_scopes_for_path(self.stage, self._get_path(name))

            instancer = UsdGeom.PointInstancer.Define(self.stage, self._get_path(name))

            # define nested capsule prim
            instancer_capsule = UsdGeom.Capsule.Define(self.stage, instancer.GetPath().AppendChild("capsule"))
            instancer_capsule.GetRadiusAttr().Set(width)

            instancer.CreatePrototypesRel().SetTargets([instancer_capsule.GetPath()])
            UsdGeom.PrimvarsAPI(instancer).CreatePrimvar(
                "displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.vertex, 1
            )

            self._instancers[name] = instancer

        instancer = self._instancers[name]

        if starts is not None and ends is not None:
            num_lines = int(len(starts))
            if num_lines > 0:
                # bring to host
                starts = starts.numpy()
                ends = ends.numpy()

                line_positions = []
                line_rotations = []
                line_scales = []

                for i in range(num_lines):
                    pos0 = starts[i]
                    pos1 = ends[i]

                    (pos, rot, scale) = _compute_segment_xform(
                        Gf.Vec3f(float(pos0[0]), float(pos0[1]), float(pos0[2])),
                        Gf.Vec3f(float(pos1[0]), float(pos1[1]), float(pos1[2])),
                    )

                    line_positions.append(pos)
                    line_rotations.append(rot)
                    line_scales.append(scale)

                instancer.GetPositionsAttr().Set(line_positions, self._frame_index)
                instancer.GetOrientationsAttr().Set(line_rotations, self._frame_index)
                instancer.GetScalesAttr().Set(line_scales, self._frame_index)
                instancer.GetProtoIndicesAttr().Set([0] * num_lines, self._frame_index)
                instancer.CreateIdsAttr().Set(list(range(num_lines)))

                if colors is not None:
                    # Promote colors to proper numpy array format
                    colors_np = self._promote_colors_to_array(colors, num_lines)

                    # Set color per-instance
                    displayColor = UsdGeom.PrimvarsAPI(instancer).GetPrimvar("displayColor")
                    displayColor.Set(colors_np, self._frame_index)

                    # Explicit identity indices [0, 1, 2, ...], otherwise OV won't pick them up
                    indices = Vt.IntArray(range(num_lines))
                    displayColor.SetIndices(indices, self._frame_index)

        instancer.GetVisibilityAttr().Set("inherited" if not hidden else "invisible", self._frame_index)

    @override
    def log_points(self, name, points, radii, colors, hidden=False):
        num_points = len(points)

        if np.isscalar(radii):
            radius_interp = "constant"
        else:
            radius_interp = "vertex"

        colors, color_interp = self._normalize_point_colors(colors, num_points)

        path = self._get_path(name)
        instancer = UsdGeom.Points.Get(self.stage, path)
        if not instancer:
            self._ensure_scopes_for_path(self.stage, path)
            instancer = UsdGeom.Points.Define(self.stage, path)

            UsdGeom.Primvar(instancer.GetWidthsAttr()).SetInterpolation(radius_interp)
            UsdGeom.Primvar(instancer.GetDisplayColorAttr()).SetInterpolation(color_interp)

        instancer.GetPointsAttr().Set(points.numpy(), self._frame_index)

        # convert radii to widths for USD
        if np.isscalar(radii):
            widths = (radii * 2.0,)
        elif isinstance(radii, wp.array):
            widths = radii.numpy() * 2.0
        else:
            widths = np.array(radii) * 2.0

        instancer.GetWidthsAttr().Set(widths, self._frame_index)

        if colors is not None:
            instancer.GetDisplayColorAttr().Set(colors, self._frame_index)

        instancer.GetVisibilityAttr().Set("inherited" if not hidden else "invisible", self._frame_index)
        return instancer.GetPath()

    @override
    def log_array(self, name, array):
        """
        Log array data (not implemented for USD backend).

        This method is a placeholder and does not log array data in the USD backend.
        """
        pass

    @override
    def log_scalar(self, name, value):
        """
        Log scalar value (not implemented for USD backend).

        This method is a placeholder and does not log scalar values in the USD backend.
        """
        pass

    def _promote_colors_to_array(self, colors, num_items):
        """
        Helper method to promote colors to a numpy array format.

        Parameters:
            colors: Input colors in various formats (wp.array, list/tuple, np.ndarray)
            num_items (int): Number of items that need colors

        Returns:
            np.ndarray: Colors as numpy array with shape (num_items, 3)
        """
        if colors is None:
            return None

        if isinstance(colors, wp.array):
            # Convert warp array to numpy
            return colors.numpy()
        elif isinstance(colors, list | tuple) and len(colors) == 3 and all(np.isscalar(x) for x in colors):
            # Single color (list/tuple of 3 floats) - promote to array with one value per item
            return np.tile(colors, (num_items, 1))
        elif isinstance(colors, np.ndarray):
            # Already numpy array - pass through
            return colors
        else:
            # Fallback for other formats
            return np.array(colors)

    @staticmethod
    def _is_single_rgb_triplet(colors) -> bool:
        """Returns True when colors represent one RGB triplet."""
        if isinstance(colors, np.ndarray):
            return colors.ndim == 1 and colors.shape[0] == 3

        if isinstance(colors, list | tuple):
            return len(colors) == 3 and all(np.isscalar(x) for x in colors)

        return False

    def _normalize_point_colors(self, colors, num_points):
        """Normalize point colors and return (values, interpolation token)."""
        if colors is None:
            return None, "constant"

        if isinstance(colors, wp.array):
            colors = colors.numpy()

        if self._is_single_rgb_triplet(colors):
            colors_arr = np.asarray(colors, dtype=np.float32)
            return colors_arr.reshape(1, 3), "constant"

        if isinstance(colors, np.ndarray):
            return colors, "vertex"

        if isinstance(colors, list | tuple):
            # Keep list/tuple inputs as-is for existing valid per-point color inputs.
            if len(colors) == num_points:
                return colors, "vertex"
            return np.asarray(colors), "vertex"

        return np.asarray(colors), "vertex"

    @staticmethod
    def _ensure_scopes_for_path(stage: Usd.Stage, prim_path_str: str):
        """
        Ensure that all parent prims in the hierarchy exist as 'Scope' prims.

        If a prim does not exist at the given path, this method creates all
        non-existent parent prims in its hierarchy as 'Scope' prims. This is
        useful for ensuring a valid hierarchy before defining a prim.

        Parameters:
            stage (Usd.Stage): The USD stage to operate on.
            prim_path_str (str): The Sdf.Path string for the target prim.
        """
        # Convert the string to an Sdf.Path object for robust manipulation
        prim_path = Sdf.Path(prim_path_str)

        # First, check if the target prim already exists.
        if stage.GetPrimAtPath(prim_path):
            return

        # We only want to create the parent hierarchy, not the final prim itself.
        parent_path = prim_path.GetParentPath()

        # GetPrefixes() provides a convenient list of all ancestor paths.
        # For "/A/B/C", it returns ["/", "/A", "/A/B"].
        for path in parent_path.GetPrefixes():
            # The absolute root path ('/') always exists, so we can skip it.
            if path == Sdf.Path.absoluteRootPath:
                continue

            # Check if a prim exists at the current ancestor path.
            if not stage.GetPrimAtPath(path):
                stage.DefinePrim(path, "Scope")
