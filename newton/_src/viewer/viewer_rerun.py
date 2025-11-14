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

import newton
from newton.utils import create_plane_mesh

from ..core.types import override
from .viewer import ViewerBase

try:
    import rerun as rr
    import rerun.blueprint as rrb
except ImportError:
    rr = None


class ViewerRerun(ViewerBase):
    """
    ViewerRerun provides a backend for visualizing Newton simulations using the rerun visualization library.

    This viewer logs mesh and instance data to rerun, enabling real-time or offline visualization of simulation
    geometry and transforms. It supports both server and client modes, and can optionally launch a web viewer.
    The class manages mesh assets, instanced geometry, and frame/timeline synchronization with rerun.
    """

    @staticmethod
    def _to_numpy(x) -> np.ndarray | None:
        """Convert warp arrays or other array-like objects to numpy arrays."""
        if x is None:
            return None
        if hasattr(x, "numpy"):
            return x.numpy()
        return np.asarray(x)

    def __init__(
        self,
        server: bool = True,
        address: str = "127.0.0.1:9876",
        launch_viewer: bool = True,
        app_id: str | None = None,
        keep_historical_data: bool = False,
        keep_scalar_history: bool = True,
        record_to_rrd: str | None = None,
    ):
        """
        Initialize the ViewerRerun backend for Newton using the Rerun.io visualization library.

        This viewer detects whether it is created inside a Jupyter notebook environment and automatically generates
        an output widget for the viewer. If it is not created inside a Jupyter notebook environment, it will start a
        local rerun server serving over gRPC that can be connected to from a web browser.

        Args:
            server (bool): If True, start rerun in server mode (gRPC).
            address (str): Address and port for rerun server mode (only used if server is True).
            launch_viewer (bool): If True, launch a local rerun viewer client.
            app_id (Optional[str]): Application ID for rerun (defaults to 'newton-viewer').
            keep_historical_data (bool): If True, keep historical data in the timeline of the web viewer.
                If False, the web viewer will only show the current frame to keep the memory usage constant when sending transform updates via :meth:`ViewerRerun.log_state`.
                This is useful for visualizing long and complex simulations that would quickly fill up the web viewer's memory if the historical data was kept.
                If True, the historical simulation data is kept in the viewer to be able to scrub through the simulation timeline. Defaults to False.
            keep_scalar_history (bool): If True, historical scala data logged via :meth:`ViewerRerun.log_scalar` is kept in the viewer.
            record_to_rrd (str): Path to record the viewer to a ``*.rrd`` recording file (e.g. "my_recording.rrd"). If None, the viewer will not record to a file.
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
        self.keep_historical_data = keep_historical_data
        self.keep_scalar_history = keep_scalar_history

        # Store mesh data for instances
        self._meshes = {}
        self._instances = {}

        # Store scalar data for logging
        self._scalars = {}

        # Initialize rerun using a blueprint that only shows the 3D view and a collapsed time panel
        blueprint = self._get_blueprint()
        rr.init(self.app_id, default_blueprint=blueprint)

        if record_to_rrd is not None:
            rr.save(record_to_rrd, default_blueprint=blueprint)

        # Launch viewer client
        self.is_jupyter_notebook = _is_jupyter_notebook()
        if not self.is_jupyter_notebook:
            # Set up connection based on mode
            server_uri = None
            if self.server:
                server_uri = rr.serve_grpc(default_blueprint=blueprint)

            if self.launch_viewer:
                if server_uri is not None:
                    rr.serve_web_viewer(connect_to=server_uri)
                else:
                    rr.serve_web_viewer()

        # Make sure the timeline is set up
        rr.set_time("time", timestamp=0.0)

    def _get_blueprint(self):
        scalar_panel = None
        if len(self._scalars) > 0:
            scalar_panel = rrb.TimeSeriesView()

        return rrb.Blueprint(
            rrb.Horizontal(
                *[rrb.Spatial3DView(), scalar_panel] if scalar_panel is not None else [rrb.Spatial3DView()],
            ),
            rrb.TimePanel(timeline="time", state="collapsed"),
            collapse_panels=True,
        )

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
        points_np = self._to_numpy(points).astype(np.float32)
        indices_np = self._to_numpy(indices).astype(np.uint32)

        # Rerun expects indices as (N, 3) for triangles
        if indices_np.ndim == 1:
            indices_np = indices_np.reshape(-1, 3)

        if normals is None:
            normals = wp.zeros_like(points)
            wp.launch(_compute_normals, dim=len(indices_np), inputs=[points, indices, normals], device=self.device)
            # normalize the normals
            wp.map(wp.normalize, normals, out=normals)
            normals_np = normals.numpy()
        else:
            normals_np = self._to_numpy(normals)

        # make sure deformable mesh updates are not kept in the viewer if desired
        static = name in self._meshes and not self.keep_historical_data

        # Store mesh data for instancing
        self._meshes[name] = {
            "points": points_np,
            "indices": indices_np,
            "normals": normals_np,
            "uvs": self._to_numpy(uvs).astype(np.float32) if uvs is not None else None,
        }

        # Log the mesh as a static asset
        mesh_3d = rr.Mesh3D(
            vertex_positions=points_np,
            triangle_indices=indices_np,
            vertex_normals=self._meshes[name]["normals"],
        )

        rr.log(name, mesh_3d, static=static)

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
                colors_np = self._to_numpy(colors).astype(np.float32)
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

            # hide the reference mesh
            rr.log(mesh, rr.Clear(recursive=False))

        # Convert transforms and properties to numpy
        if xforms is not None:
            # Convert warp arrays to numpy first
            xforms_np = self._to_numpy(xforms)

            # Extract positions and quaternions using vectorized operations
            # Warp transform format: [x, y, z, qx, qy, qz, qw]
            translations = xforms_np[:, :3].astype(np.float32)

            # Warp quaternion is in (x, y, z, w) order,
            # rerun expects (x, y, z, w) for Quaternion datatype
            quaternions = xforms_np[:, 3:7].astype(np.float32)

            scales_np = None
            if scales is not None:
                scales_np = self._to_numpy(scales).astype(np.float32)

            # Colors are already handled in the mesh
            # (first instance color applied to all)

            # Create instance poses
            instance_poses = rr.InstancePoses3D(
                translations=translations,
                quaternions=quaternions,
                scales=scales_np,
            )

            # Log the instance poses
            rr.log(name, instance_poses, static=not self.keep_historical_data)

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

    @override
    def log_lines(self, name, starts, ends, colors, width: float = 0.01, hidden=False):
        """
        Log lines for visualization.

        Args:
            name (str): Name of the line batch.
            starts: Line start points.
            ends: Line end points.
            colors: Line colors.
            width (float): Line width.
            hidden (bool): Whether the lines are hidden.
        """

        if hidden:
            return  # Do not log hidden lines

        if starts is None or ends is None:
            return  # Nothing to log

        # Convert inputs to numpy for rerun API compatibility
        # Expecting starts/ends as wp arrays or numpy arrays
        starts_np = self._to_numpy(starts)
        ends_np = self._to_numpy(ends)
        colors_np = self._to_numpy(colors) if colors is not None else None

        # Both starts and ends should be (N, 3)
        if starts_np is None or ends_np is None or len(starts_np) == 0:
            return

        # LineStrips3D expects a list of line strips, where each strip is a sequence of points
        # For disconnected line segments, each segment becomes its own strip of 2 points
        line_strips = []
        for start, end in zip(starts_np, ends_np, strict=False):
            line_strips.append([start, end])

        # Prepare line color argument
        rr_kwargs = {}
        if colors_np is not None:
            # If single color for all lines (shape (3,))
            if colors_np.ndim == 1 and colors_np.shape[0] == 3:
                rr_kwargs["colors"] = colors_np
            # If (N,3), per-line colors
            elif colors_np.ndim == 2 and colors_np.shape[1] == 3:
                rr_kwargs["colors"] = colors_np
        if width is not None:
            rr_kwargs["radii"] = width

        # Log to rerun
        rr.log(name, rr.LineStrips3D(line_strips, **rr_kwargs), static=not self.keep_historical_data)

    @override
    def log_array(self, name, array):
        """
        Log a generic array for visualization.

        Args:
            name (str): Name of the array.
            array: The array data (can be a wp.array or a numpy array).
        """
        if array is None:
            return
        array_np = self._to_numpy(array)
        rr.log(name, rr.Scalars(array_np), static=not self.keep_historical_data)

    @override
    def log_scalar(self, name, value):
        """
        Log a scalar value for visualization.

        Args:
            name (str): Name of the scalar.
            value: The scalar value.
        """
        # Basic scalar logging for rerun: log as a 'Scalar' component (if present)
        if name is None:
            return

        # Only support standard Python/numpy scalars, not generic objects for now
        if hasattr(value, "item"):
            val = value.item()
        else:
            val = value
        rr.log(name, rr.Scalars(val), static=not self.keep_scalar_history)

        if len(self._scalars) == 0:
            self._scalars[name] = val
            blueprint = self._get_blueprint()
            rr.send_blueprint(blueprint)
        else:
            self._scalars[name] = val

    @override
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
        # Generate vertices/indices for supported primitive types
        if geo_type == newton.GeoType.PLANE:
            # Handle "infinite" planes encoded with non-positive scales
            if geo_scale[0] == 0.0 or geo_scale[1] == 0.0:
                extents = self._get_world_extents()
                if extents is None:
                    width, length = 10.0, 10.0
                else:
                    max_extent = max(extents) * 1.5
                    width = max_extent
                    length = max_extent
            else:
                width = geo_scale[0]
                length = geo_scale[1] if len(geo_scale) > 1 else 10.0
            vertices, indices = create_plane_mesh(width, length)
            points = wp.array(vertices[:, 0:3], dtype=wp.vec3, device=self.device)
            normals = wp.array(vertices[:, 3:6], dtype=wp.vec3, device=self.device)
            uvs = wp.array(vertices[:, 6:8], dtype=wp.vec2, device=self.device)
            indices = wp.array(indices, dtype=wp.int32, device=self.device)
            self.log_mesh(name, points, indices, normals, uvs)
        else:
            super().log_geo(name, geo_type, geo_scale, geo_thickness, geo_is_solid, geo_src, hidden)

    @override
    def log_points(self, name, points, radii=None, colors=None, hidden=False):
        """
        Log points for visualization.

        Args:
            name (str): Name of the point batch.
            points: Point positions (can be a wp.array or a numpy array).
            radii: Point radii (can be a wp.array or a numpy array).
            colors: Point colors (can be a wp.array or a numpy array).
            hidden (bool): Whether the points are hidden.
        """
        if hidden:
            # Optionally, skip logging hidden points
            return

        if points is None:
            return

        pts = self._to_numpy(points)
        n_points = pts.shape[0]

        # Handle radii (point size)
        if radii is not None:
            size = self._to_numpy(radii)
            if size.ndim == 0 or size.shape == ():
                sizes = np.full((n_points,), float(size))
            elif size.shape == (n_points,):
                sizes = size
            else:
                sizes = np.full((n_points,), 0.1)
        else:
            sizes = np.full((n_points,), 0.1)

        # Handle colors
        if colors is not None:
            cols = self._to_numpy(colors)
            if cols.shape == (n_points, 3):
                colors_val = cols
            elif cols.shape == (3,):
                colors_val = np.tile(cols, (n_points, 1))
            else:
                colors_val = np.full((n_points, 3), 1.0)
        else:
            colors_val = np.full((n_points, 3), 1.0)

        # Log as points to rerun
        rr.log(
            name,
            rr.Points3D(
                positions=pts,
                radii=sizes,
                colors=colors_val,
            ),
            static=not self.keep_historical_data,
        )

    def show_notebook(self, width: int = 1000, height: int = 400, legacy_notebook_show: bool = False):
        """
        Show the viewer in a Jupyter notebook.

        Args:
            width (int): Width of the viewer in pixels.
            height (int): Height of the viewer in pixels.
            legacy_notebook_show (bool): Whether to use ``rr.legacy_notebook_show`` instead of ``rr.notebook_show`` for displaying the viewer as static HTML with embedded recording data.
        """
        if legacy_notebook_show:
            rr.legacy_notebook_show(width=width, height=height, blueprint=self._get_blueprint())
        else:
            rr.notebook_show(width=width, height=height, blueprint=self._get_blueprint())

    def _ipython_display_(self):
        """
        Display the viewer in an IPython notebook when the viewer is at the end of a cell.
        """
        self.show_notebook()


def _is_jupyter_notebook():
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


@wp.kernel
def _compute_normals(
    points: wp.array(dtype=wp.vec3),
    indices: wp.array(dtype=wp.int32),
    # output
    normals: wp.array(dtype=wp.vec3),
):
    face = wp.tid()
    i0 = indices[face * 3]
    i1 = indices[face * 3 + 1]
    i2 = indices[face * 3 + 2]
    v0 = points[i0]
    v1 = points[i1]
    v2 = points[i2]
    normal = wp.normalize(wp.cross(v1 - v0, v2 - v0))
    wp.atomic_add(normals, i0, normal)
    wp.atomic_add(normals, i1, normal)
    wp.atomic_add(normals, i2, normal)
