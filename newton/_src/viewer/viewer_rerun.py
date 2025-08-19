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
from typing import Optional

import numpy as np
import warp as wp

from .viewer import ViewerBase

try:
    import rerun as rr
except ImportError:
    rr = None


class ViewerRerun(ViewerBase):
    def __init__(
        self,
        model,
        server: bool = True,
        address: str = "127.0.0.1:9876",
        launch_viewer: bool = True,
        app_id: Optional[str] = None,
    ):
        """Initialize the Rerun viewer backend.

        Args:
            model: Newton model to visualize
            server: Whether to start in server mode (TCP/gRPC)
            address: Address and port for server mode
            launch_viewer: Whether to spawn a local rerun viewer client
            app_id: Application ID for rerun (defaults to 'newton-viewer')
        """
        if rr is None:
            raise ImportError("rerun package is required for ViewerRerun. Install with: pip install rerun-sdk")

        super().__init__(model)

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

        self._populate(model)

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
        """Log mesh data to rerun.

        Args:
            name: Entity path for the mesh
            points: Vertex positions (wp.array of wp.vec3)
            indices: Triangle indices (wp.array of wp.uint32)
            normals: Vertex normals (optional, wp.array of wp.vec3)
            uvs: UV coordinates (optional, wp.array of wp.vec2)
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

    def log_instances(self, name, mesh, xforms, scales, colors, materials):
        """Log instanced mesh data to rerun using InstancePoses3D.

        Args:
            name: Entity path for instances
            mesh: Path to mesh asset
            xforms: Instance transforms (wp.array of wp.transform)
            scales: Instance scales (wp.array of wp.vec3)
            colors: Instance colors (wp.array of wp.vec3)
            materials: Instance materials (wp.array of wp.vec4)
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

    def begin_frame(self, time):
        """Begin a new frame with given time."""
        self.time = time
        # Set the timeline for this frame
        rr.set_time("time", timestamp=time)

    def end_frame(self):
        """End the current frame."""
        # Rerun handles frame finishing automatically
        pass

    def is_running(self) -> bool:
        """Check if the viewer is still running."""
        # Check if viewer process is still alive
        if self._viewer_process is not None:
            return self._viewer_process.poll() is None
        return self._running

    def close(self):
        """Close the viewer and clean up resources."""
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
    def log_lines(self, name, line_begins, line_ends, line_colors, hidden=False):
        pass

    def log_points(self, name, state):
        pass

    def log_array(self, name, array):
        pass

    def log_scalar(self, name, value):
        pass
