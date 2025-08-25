from __future__ import annotations

import numpy as np
import warp as wp

try:
    from pxr import Gf, Sdf, Usd, UsdGeom, Vt
except ImportError:
    Gf = Sdf = Usd = UsdGeom = Vt = None

from .viewer import ViewerBase


class ViewerUSD(ViewerBase):
    """
    USD viewer backend for Newton physics simulations.

    This backend creates a USD stage and manages mesh prototypes and instanced rendering
    using PointInstancers. It supports time-sampled transforms for efficient playback
    and visualization of simulation data.
    """

    def __init__(self, output_path, fps=60, up_axis="Z", num_frames=None):
        """
        Initialize the USD viewer backend for Newton physics simulations.

        Args:
            output_path (str): Path to the output USD file.
            fps (int, optional): Frames per second for time sampling. Default is 60.
            up_axis (str, optional): USD up axis, either 'Y' or 'Z'. Default is 'Z'.
            num_frames (int, optional): Maximum number of frames to record. If None, recording is unlimited.

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

        # Set stage metadata
        if up_axis == "Y":
            UsdGeom.SetStageUpAxis(self.stage, UsdGeom.Tokens.y)
        else:
            UsdGeom.SetStageUpAxis(self.stage, UsdGeom.Tokens.z)

        self.stage.SetFramesPerSecond(fps)
        self.stage.SetStartTimeCode(0)

        # Track meshes and instancers
        self._meshes = {}  # mesh_name -> prototype_path
        self._instancers = {}  # instancer_name -> UsdGeomPointInstancer

        # Track current frame
        self._current_frame = 0
        self._frame_time = 0.0
        self._frame_count = 0

    def log_mesh(
        self,
        name,
        points: wp.array,
        indices: wp.array,
        normals: wp.array = None,
        uvs: wp.array = None,
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

        self._ensure_scopes_for_path(self.stage, name)

        # Convert warp arrays to numpy
        points_np = points.numpy().astype(np.float32)
        indices_np = indices.numpy().astype(np.uint32)

        mesh_path = name
        mesh_prim = UsdGeom.Mesh.Define(self.stage, mesh_path)

        # Set vertex positions
        vec3_points = [Gf.Vec3f(float(pt[0]), float(pt[1]), float(pt[2])) for pt in points_np]
        mesh_prim.CreatePointsAttr().Set(vec3_points)

        # Set face vertex counts (all triangles)
        face_vertex_counts = [3] * (len(indices_np) // 3)
        mesh_prim.CreateFaceVertexCountsAttr().Set(face_vertex_counts)

        # Set face vertex indices
        mesh_prim.CreateFaceVertexIndicesAttr().Set(indices_np.tolist())

        # Set normals if provided
        if normals is not None:
            normals_np = normals.numpy().astype(np.float32)
            vec3_normals = [Gf.Vec3f(float(n[0]), float(n[1]), float(n[2])) for n in normals_np]
            mesh_prim.CreateNormalsAttr().Set(vec3_normals)
            mesh_prim.SetNormalsInterpolation(UsdGeom.Tokens.vertex)

        # Set UVs if provided (simplified for now)
        if uvs is not None:
            # TODO: Implement UV support for USD meshes
            pass

        # Store the prototype path
        self._meshes[name] = mesh_path

        return mesh_path

    def log_instances(self, name, mesh, xforms, scales, colors, materials):
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
            msg = f"Mesh prototype '{mesh}' not found. Call log_mesh first."
            raise RuntimeError(msg)

        # Create instancer if it doesn't exist
        if name not in self._instancers:
            self._ensure_scopes_for_path(self.stage, name)

            instancer_path = name
            instancer = UsdGeom.PointInstancer.Define(self.stage, instancer_path)

            # Set the prototype relationship
            instancer.GetPrototypesRel().AddTarget(mesh)

            self._instancers[name] = instancer

        instancer = self._instancers[name]

        # Convert transforms to USD format
        if xforms is not None:
            xforms_np = xforms.numpy()
            num_instances = len(xforms_np)

            # Extract positions and orientations from warp transforms
            # Warp transform format: [x, y, z, qx, qy, qz, qw]
            positions = []
            orientations = []

            for i in range(num_instances):
                xform = xforms_np[i]
                pos = Gf.Vec3f(float(xform[0]), float(xform[1]), float(xform[2]))
                # Warp quaternion is (x, y, z, w) → USD expects (w, (x,y,z))
                quat_w = float(xform[6])
                quat_xyz = Gf.Vec3h(float(xform[3]), float(xform[4]), float(xform[5]))
                quat = Gf.Quath(quat_w, quat_xyz)

                positions.append(pos)
                orientations.append(quat)

            # Convert scales
            scales_gf = []
            if scales is not None:
                scales_np = scales.numpy()
                for i in range(num_instances):
                    scale = scales_np[i]
                    scales_gf.append(
                        Gf.Vec3f(
                            float(scale[0]),
                            float(scale[1]),
                            float(scale[2]),
                        )
                    )
            else:
                scales_gf = [Gf.Vec3f(1.0, 1.0, 1.0)] * num_instances

            # Convert colors
            colors_gf = []
            if colors is not None:
                colors_np = colors.numpy()
                for i in range(num_instances):
                    color = colors_np[i]
                    colors_gf.append(
                        Gf.Vec3f(
                            float(color[0]),
                            float(color[1]),
                            float(color[2]),
                        )
                    )
            else:
                colors_gf = [Gf.Vec3f(0.7, 0.7, 0.7)] * num_instances

            # Set prototype indices (all instances use prototype 0)
            proto_indices = [0] * num_instances

            # Set attributes at current time
            time_code = Usd.TimeCode(self._current_frame)

            # Initialize ids and protoIndices once for count stability
            ids_attr = instancer.GetIdsAttr()
            if ids_attr.GetNumTimeSamples() == 0 and not ids_attr.HasAuthoredValueOpinion():
                instancer.CreateIdsAttr().Set(list(range(num_instances)))
                instancer.CreateProtoIndicesAttr().Set([0] * num_instances)

            instancer.CreateProtoIndicesAttr().Set(proto_indices, time_code)
            instancer.CreatePositionsAttr().Set(positions, time_code)
            instancer.CreateOrientationsAttr().Set(orientations, time_code)
            instancer.CreateScalesAttr().Set(scales_gf, time_code)

            # Per-instance colors via primvars:displayColor on the
            # PointInstancer
            if colors is not None:
                try:
                    pv_api = UsdGeom.PrimvarsAPI(instancer)
                    if pv_api.HasPrimvar("displayColor"):
                        col_pv = pv_api.GetPrimvar("displayColor")
                    else:
                        col_pv = pv_api.CreatePrimvar(
                            "displayColor",
                            Sdf.ValueTypeNames.Color3fArray,
                            UsdGeom.Tokens.vertex,
                        )

                    # Set color per-instance
                    col_pv.Set(colors_gf, time_code)

                    # Explicit identity indices [0, 1, 2, ...], otherwise OV won't pick them up
                    num_instances = len(colors_gf)
                    indices = Vt.IntArray(range(num_instances))
                    col_pv.SetIndices(indices, time_code)

                except Exception:
                    # Be robust if PrimvarsAPI or types are unavailable
                    pass

    def begin_frame(self, time):
        """
        Begin a new frame at the given simulation time.

        Parameters:
            time (float): The simulation time for the new frame.
        """
        super().begin_frame(time)
        self._frame_time = time
        self._current_frame = int(time * self.fps)
        self._frame_count += 1

        # Update stage end time if needed
        if self._current_frame > self.stage.GetEndTimeCode():
            self.stage.SetEndTimeCode(self._current_frame)

    def end_frame(self):
        """
        End the current frame.

        This method is a placeholder for any end-of-frame logic required by the backend.
        """
        pass

    def is_running(self):
        """
        Check if the viewer is still running.

        Returns:
            bool: False if the frame limit is exceeded, True otherwise.
        """
        if self.num_frames is not None:
            return self._frame_count < self.num_frames
        return True

    def close(self):
        """
        Finalize and save the USD stage.

        This should be called when all logging is complete to ensure the USD file is written.
        """
        self.stage.GetRootLayer().Save()
        self.stage = None

    # Abstract methods that need basic implementations
    def log_lines(self, name, line_begins, line_ends, line_colors, hidden=False):
        """
        Log lines (not implemented for USD backend).

        This method is a placeholder and does not log lines in the USD backend.
        """
        pass

    def log_points(self, name, points, widths, colors, hidden=False):
        """
        Log points (not implemented for USD backend).

        This method is a placeholder and does not log points in the USD backend.
        """
        pass

    def log_array(self, name, array):
        """
        Log array data (not implemented for USD backend).

        This method is a placeholder and does not log array data in the USD backend.
        """
        pass

    def log_scalar(self, name, value):
        """
        Log scalar value (not implemented for USD backend).

        This method is a placeholder and does not log scalar values in the USD backend.
        """
        pass

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
