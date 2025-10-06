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

import subprocess

import numpy as np
import warp as wp

from ..core.types import override
from .viewer import ViewerBase

try:
    import rerun as rr
except ImportError:
    rr = None


class ViewerRerun(ViewerBase):
    """
    ViewerRerun provides a backend for visualizing Newton simulations using the rerun visualization library.

    This viewer logs mesh and instance data to rerun, enabling real-time or offline visualization of simulation
    geometry and transforms. It supports both server and client modes, and can optionally launch a web viewer.
    The class manages mesh assets, instanced geometry, and frame/timeline synchronization with rerun.
    """

    def __init__(
        self,
        server: bool = True,
        address: str = "127.0.0.1:9876",
        launch_viewer: bool = True,
        app_id: str | None = None,
    ):
        """
        Initialize the ViewerRerun backend for Newton using the rerun visualization library.

        Args:
            server (bool): If True, start rerun in server mode (TCP/gRPC).
            address (str): Address and port for rerun server mode.
            launch_viewer (bool): If True, launch a local rerun viewer client.
            app_id (Optional[str]): Application ID for rerun (defaults to 'newton-viewer').
        """
        if rr is None:
            raise ImportError("rerun package is required for ViewerRerun. Install with: pip install rerun-sdk")

        super().__init__()

        self.server = server
        self.address = address
        self.launch_viewer = launch_viewer
        self.app_id = app_id or "newton-viewer"
        self._running = True
        self._viewer_process = None

        # Initialize rerun
        rr.init(self.app_id)

        # Set up connection based on mode
        if self.server:
            server_uri = rr.serve_grpc()

        # Optionally launch viewer client
        if self.launch_viewer:
            rr.serve_web_viewer(connect_to=server_uri)

        # Store mesh data for instances
        self._meshes = {}
        self._instances = {}

    @override
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
        """
        Log a mesh to rerun for visualization.

        Args:
            name (str): Entity path for the mesh.
            points (wp.array): Vertex positions (wp.vec3).
            indices (wp.array): Triangle indices (wp.uint32).
            normals (wp.array, optional): Vertex normals (wp.vec3).
            uvs (wp.array, optional): UV coordinates (wp.vec2).
            hidden (bool): Whether the mesh is hidden (unused).
            backface_culling (bool): Whether to enable backface culling (unused).
        """
        assert isinstance(points, wp.array)
        assert isinstance(indices, wp.array)
        assert normals is None or isinstance(normals, wp.array)
        assert uvs is None or isinstance(uvs, wp.array)

        # Convert to numpy arrays
        points_np = points.numpy().astype(np.float32)
        indices_np = indices.numpy().astype(np.uint32)

        # Rerun expects indices as (N, 3) for triangles
        if indices_np.ndim == 1:
            indices_np = indices_np.reshape(-1, 3)

        # Store mesh data for instancing
        self._meshes[name] = {
            "points": points_np,
            "indices": indices_np,
            "normals": (normals.numpy().astype(np.float32) if normals is not None else None),
            "uvs": uvs.numpy().astype(np.float32) if uvs is not None else None,
        }

        # Log the mesh as a static asset
        mesh_3d = rr.Mesh3D(
            vertex_positions=points_np,
            triangle_indices=indices_np,
            vertex_normals=self._meshes[name]["normals"],
        )

        rr.log(name, mesh_3d, static=True)

    @override
    def log_instances(self, name, mesh, xforms, scales, colors, materials, hidden=False):
        """
        Log instanced mesh data to rerun using InstancePoses3D.

        Args:
            name (str): Entity path for the instances.
            mesh (str): Name of the mesh asset to instance.
            xforms (wp.array): Instance transforms (wp.transform).
            scales (wp.array): Instance scales (wp.vec3).
            colors (wp.array): Instance colors (wp.vec3).
            materials (wp.array): Instance materials (wp.vec4).
            hidden (bool): Whether the instances are hidden. (unused)
        """
        # Check that mesh exists
        if mesh not in self._meshes:
            raise RuntimeError(f"Mesh {mesh} not found. Call log_mesh first.")

        # re-run needs to generate a new mesh for each instancer
        if name not in self._instances:
            mesh_data = self._meshes[mesh]

            # Handle colors - ReRun doesn't support per-instance colors
            # so we just use the first instance's color for all instances
            if colors is not None:
                colors_np = colors.numpy().astype(np.float32)
                # Take the first instance's color and apply to all vertices
                first_color = colors_np[0]
                color_rgb = np.array(first_color * 255, dtype=np.uint8)
                num_vertices = len(mesh_data["points"])
                vertex_colors = np.tile(color_rgb, (num_vertices, 1))

            # Log the base mesh with optional colors
            mesh_3d = rr.Mesh3D(
                vertex_positions=mesh_data["points"],
                triangle_indices=mesh_data["indices"],
                vertex_normals=mesh_data["normals"],
                vertex_colors=vertex_colors,
            )
            rr.log(name, mesh_3d)

            # save reference
            self._instances[name] = mesh_3d

        # Convert transforms and properties to numpy
        if xforms is not None:
            # Convert warp arrays to numpy first
            xforms_np = xforms.numpy()

            # Extract positions and quaternions using vectorized operations
            # Warp transform format: [x, y, z, qx, qy, qz, qw]
            translations = xforms_np[:, :3].astype(np.float32)

            # Warp quaternion is in (x, y, z, w) order,
            # rerun expects (x, y, z, w) for Quaternion datatype
            quaternions = xforms_np[:, 3:7].astype(np.float32)

            scales_np = None
            if scales is not None:
                scales_np = scales.numpy().astype(np.float32)

            # Colors are already handled in the mesh
            # (first instance color applied to all)

            # Create instance poses
            instance_poses = rr.InstancePoses3D(
                translations=translations,
                quaternions=quaternions,
                scales=scales_np,
            )

            # Log the instance poses
            rr.log(name, instance_poses)

    @override
    def begin_frame(self, time):
        """
        Begin a new frame and set the timeline for rerun.

        Args:
            time (float): The current simulation time.
        """
        self.time = time
        # Set the timeline for this frame
        rr.set_time("time", timestamp=time)

    @override
    def end_frame(self):
        """
        End the current frame.

        Note:
            Rerun handles frame finishing automatically.
        """
        # Rerun handles frame finishing automatically
        pass

    @override
    def is_running(self) -> bool:
        """
        Check if the viewer is still running.

        Returns:
            bool: True if the viewer is running, False otherwise.
        """
        # Check if viewer process is still alive
        if self._viewer_process is not None:
            return self._viewer_process.poll() is None
        return self._running

    @override
    def close(self):
        """
        Close the viewer and clean up resources.

        This will terminate any spawned viewer process and disconnect from rerun.
        """
        self._running = False

        # Close viewer process if we spawned one
        if self._viewer_process is not None:
            try:
                self._viewer_process.terminate()
                self._viewer_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._viewer_process.kill()
            except Exception:
                pass
            self._viewer_process = None

        # Disconnect from rerun
        try:
            rr.disconnect()
        except Exception:
            pass

    # Not implemented yet - placeholder methods from ViewerBase
    @override
    def log_lines(self, name, starts, ends, colors, width: float = 0.01, hidden=False):
        """
        Placeholder for logging lines to rerun.

        Args:
            name (str): Name of the line batch.
            starts: Line start points.
            ends: Line end points.
            colors: Line colors.
            hidden (bool): Whether the lines are hidden.
        """
        pass

    @override
    def log_points(self, name, points, radii, colors, hidden=False):
        """
        Placeholder for logging points to rerun.

        Args:
            name (str): Name of the point batch.
            points: Point positions.
            radius: Point radii.
            colors: Point colors.
            hidden (bool): Whether the points are hidden.
        """
        pass

    @override
    def log_array(self, name, array):
        """
        Placeholder for logging a generic array to rerun.

        Args:
            name (str): Name of the array.
            array: The array data.
        """
        pass

    @override
    def log_scalar(self, name, value):
        """
        Placeholder for logging a scalar value to rerun.

        Args:
            name (str): Name of the scalar.
            value: The scalar value.
        """
        pass
