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

import numpy as np
import warp as wp

import newton
from newton.utils import create_plane_mesh

from ..core.types import override
from .viewer import ViewerBase


class ViewerViser(ViewerBase):
    """
    ViewerViser provides a backend for visualizing Newton simulations using the viser library.

    Viser is a Python library for interactive 3D visualization in the browser. This viewer
    launches a web server and renders simulation geometry via WebGL. It supports both
    standalone browser viewing and Jupyter notebook integration.

    Features:
        - Real-time 3D visualization in any web browser
        - Jupyter notebook integration with inline display
        - Static HTML export for sharing visualizations
        - Interactive camera controls
    """

    _viser_module = None

    @classmethod
    def _get_viser(cls):
        """Lazily import and return the viser module."""
        if cls._viser_module is None:
            try:
                import viser  # noqa: PLC0415

                cls._viser_module = viser
            except ImportError as e:
                raise ImportError(
                    "viser package is required for ViewerViser. Install with: pip install viser"
                ) from e
        return cls._viser_module

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
        *,
        port: int = 8080,
        label: str | None = None,
        verbose: bool = True,
        share: bool = False,
    ):
        """
        Initialize the ViewerViser backend for Newton using the viser visualization library.

        This viewer supports both standalone browser viewing and Jupyter notebook environments.
        It launches a web server that serves an interactive 3D visualization.

        Args:
            port (int): Port number for the web server. Defaults to 8080.
            label (str | None): Optional label for the viser server window title.
            verbose (bool): If True, print the server URL when starting. Defaults to True.
            share (bool): If True, create a publicly accessible URL via viser's share feature.
        """
        viser = self._get_viser()

        super().__init__()

        self._running = True
        self.verbose = verbose

        # Store mesh data for instances
        self._meshes = {}
        self._instances = {}
        self._scene_handles = {}  # Track viser scene node handles

        # Store scalar data for logging
        self._scalars = {}

        # Initialize viser server
        self._server = viser.ViserServer(port=port, label=label or "Newton Viewer")

        if share:
            self._share_url = self._server.request_share_url()
            if verbose:
                print(f"Viser share URL: {self._share_url}")
        else:
            self._share_url = None

        if verbose:
            print(f"Viser server running at: http://localhost:{port}")

        # Store configuration
        self._port = port

        # Track if running in Jupyter
        self.is_jupyter_notebook = _is_jupyter_notebook()

        # Recording state
        self._recording = False
        self._serializer = None
        self._last_frame_time = None
        self._record_fps = 30.0

        # Set up default scene
        self._setup_scene()

    def _setup_scene(self):
        """Set up the default scene configuration."""
        # Set up world axes
        self._server.scene.add_frame(
            "/world",
            wxyz=(1.0, 0.0, 0.0, 0.0),
            position=(0.0, 0.0, 0.0),
            axes_length=0.5,
            axes_radius=0.01,
            show_axes=False,
        )

        # Set background color
        # self._server.scene.set_background_image(
        #     media_type="image/png",
        #     base64_data="",  # Empty for solid color
        # )

    @property
    def url(self) -> str:
        """Get the URL of the viser server."""
        return f"http://localhost:{self._port}"

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
        Log a mesh to viser for visualization.

        Args:
            name (str): Entity path for the mesh.
            points (wp.array): Vertex positions (wp.vec3).
            indices (wp.array): Triangle indices (wp.uint32).
            normals (wp.array, optional): Vertex normals (wp.vec3).
            uvs (wp.array, optional): UV coordinates (wp.vec2).
            hidden (bool): Whether the mesh is hidden.
            backface_culling (bool): Whether to enable backface culling.
        """
        assert isinstance(points, wp.array)
        assert isinstance(indices, wp.array)
        assert normals is None or isinstance(normals, wp.array)
        assert uvs is None or isinstance(uvs, wp.array)

        # Convert to numpy arrays
        points_np = self._to_numpy(points).astype(np.float32)
        indices_np = self._to_numpy(indices).astype(np.uint32)

        # Viser expects indices as (N, 3) for triangles
        if indices_np.ndim == 1:
            indices_np = indices_np.reshape(-1, 3)

        # Compute normals if not provided
        if normals is None:
            normals_wp = wp.zeros_like(points)
            wp.launch(_compute_normals, dim=len(indices_np), inputs=[points, indices, normals_wp], device=self.device)
            wp.map(wp.normalize, normals_wp, out=normals_wp)
            normals_np = normals_wp.numpy()
        else:
            normals_np = self._to_numpy(normals)

        # Store mesh data for instancing
        self._meshes[name] = {
            "points": points_np,
            "indices": indices_np,
            "normals": normals_np,
            "uvs": self._to_numpy(uvs).astype(np.float32) if uvs is not None else None,
        }

        # Remove existing mesh if present
        if name in self._scene_handles:
            try:
                self._scene_handles[name].remove()
            except Exception:
                pass

        if hidden:
            return

        # Add mesh to viser scene
        handle = self._server.scene.add_mesh_simple(
            name=name,
            vertices=points_np,
            faces=indices_np,
            color=(180, 180, 180),  # Default gray color
            wireframe=False,
            side="double" if not backface_culling else "front",
        )
        self._scene_handles[name] = handle

    @override
    def log_instances(self, name, mesh, xforms, scales, colors, materials, hidden=False):
        """
        Log instanced mesh data to viser using efficient batched rendering.

        Uses viser's add_batched_meshes_simple for GPU-accelerated instanced rendering.
        Supports in-place updates of transforms for real-time animation.

        Args:
            name (str): Entity path for the instances.
            mesh (str): Name of the mesh asset to instance.
            xforms (wp.array): Instance transforms (wp.transform).
            scales (wp.array): Instance scales (wp.vec3).
            colors (wp.array): Instance colors (wp.vec3).
            materials (wp.array): Instance materials (wp.vec4).
            hidden (bool): Whether the instances are hidden.
        """
        # Check that mesh exists
        if mesh not in self._meshes:
            raise RuntimeError(f"Mesh {mesh} not found. Call log_mesh first.")

        mesh_data = self._meshes[mesh]
        base_points = mesh_data["points"]
        base_indices = mesh_data["indices"]

        if hidden:
            # Remove existing instances if present
            if name in self._scene_handles:
                try:
                    self._scene_handles[name].remove()
                except Exception:
                    pass
                del self._scene_handles[name]
                if name in self._instances:
                    del self._instances[name]
            return

        # Convert transforms and properties to numpy
        if xforms is None:
            return

        xforms_np = self._to_numpy(xforms)
        scales_np = self._to_numpy(scales) if scales is not None else None
        colors_np = self._to_numpy(colors) if colors is not None else None

        num_instances = len(xforms_np)

        # Extract positions from transforms
        # Warp transform format: [x, y, z, qx, qy, qz, qw]
        positions = xforms_np[:, :3].astype(np.float32)

        # Convert quaternions from Warp format (x, y, z, w) to viser format (w, x, y, z)
        quats_xyzw = xforms_np[:, 3:7]
        quats_wxyz = np.zeros((num_instances, 4), dtype=np.float32)
        quats_wxyz[:, 0] = quats_xyzw[:, 3]  # w
        quats_wxyz[:, 1] = quats_xyzw[:, 0]  # x
        quats_wxyz[:, 2] = quats_xyzw[:, 1]  # y
        quats_wxyz[:, 3] = quats_xyzw[:, 2]  # z

        # Prepare scales
        if scales_np is not None:
            batched_scales = scales_np.astype(np.float32)
        else:
            batched_scales = np.ones((num_instances, 3), dtype=np.float32)

        # Prepare colors (convert from 0-1 float to 0-255 uint8)
        if colors_np is not None:
            batched_colors = (colors_np * 255).astype(np.uint8)
        else:
            # Default gray color
            batched_colors = np.full((num_instances, 3), 180, dtype=np.uint8)

        # Check if we already have a batched mesh handle for this name
        if name in self._instances and name in self._scene_handles:
            # Update existing batched mesh in-place (much faster)
            handle = self._scene_handles[name]
            prev_count = self._instances[name]["count"]

            # If instance count changed, we need to recreate the mesh
            if prev_count != num_instances:
                try:
                    handle.remove()
                except Exception:
                    pass
                del self._scene_handles[name]
                del self._instances[name]
            else:
                # Update transforms in-place
                try:
                    handle.batched_positions = positions
                    handle.batched_wxyzs = quats_wxyz
                    handle.batched_scales = batched_scales
                    handle.batched_colors = batched_colors
                    return
                except Exception:
                    # If update fails, recreate the mesh
                    try:
                        handle.remove()
                    except Exception:
                        pass
                    del self._scene_handles[name]
                    del self._instances[name]

        # Create new batched mesh
        handle = self._server.scene.add_batched_meshes_simple(
            name=name,
            vertices=base_points,
            faces=base_indices,
            batched_positions=positions,
            batched_wxyzs=quats_wxyz,
            batched_scales=batched_scales,
            batched_colors=batched_colors,
            lod="auto",
        )

        self._scene_handles[name] = handle
        self._instances[name] = {
            "mesh": mesh,
            "count": num_instances,
        }

    @override
    def begin_frame(self, time):
        """
        Begin a new frame.

        Args:
            time (float): The current simulation time.
        """
        self.time = time

    @override
    def end_frame(self):
        """
        End the current frame.

        If recording is active, inserts a sleep command for playback timing.
        """
        if self._recording and self._serializer is not None:
            # Insert sleep for frame timing during recording
            frame_dt = 1.0 / self._record_fps
            self._serializer.insert_sleep(frame_dt)

    @override
    def is_running(self) -> bool:
        """
        Check if the viewer is still running.

        Returns:
            bool: True if the viewer is running, False otherwise.
        """
        return self._running

    @override
    def close(self):
        """
        Close the viewer and clean up resources.
        """
        self._running = False
        try:
            self._server.stop()
        except Exception:
            pass

    # =========================================================================
    # Recording functionality for embedding visualizations
    # =========================================================================

    def start_recording(self, fps: float = 30.0):
        """
        Start recording the scene for later playback.

        This captures all scene updates and can be saved to a .viser file
        for embedding in static HTML pages.

        Args:
            fps (float): Target frames per second for playback. Defaults to 30.0.

        Example:
            >>> viewer.start_recording(fps=30)
            >>> for i in range(100):
            ...     viewer.begin_frame(i * dt)
            ...     viewer.log_state(state)
            ...     viewer.end_frame()
            >>> viewer.save_recording("simulation.viser")
        """
        self._recording = True
        self._record_fps = fps
        self._serializer = self._server.get_scene_serializer()
        if self.verbose:
            print(f"Started recording at {fps} FPS")

    def stop_recording(self):
        """
        Stop recording without saving.

        Use save_recording() instead if you want to save the recording.
        """
        self._recording = False
        self._serializer = None
        if self.verbose:
            print("Recording stopped")

    def save_recording(self, filepath: str):
        """
        Save the current recording to a .viser file.

        The recording can be played back in a static HTML viewer.
        See build_static_viewer() for creating the HTML player.

        Args:
            filepath (str): Path to save the recording (should end in .viser).

        Example:
            >>> viewer.start_recording()
            >>> # ... run simulation ...
            >>> viewer.save_recording("my_simulation.viser")
        """
        if self._serializer is None:
            raise RuntimeError("No recording in progress. Call start_recording() first.")

        from pathlib import Path  # noqa: PLC0415

        data = self._serializer.serialize()
        Path(filepath).write_bytes(data)

        self._recording = False
        self._serializer = None

        if self.verbose:
            print(f"Recording saved to: {filepath}")

    def add_save_button(self, label: str = "Save Recording"):
        """
        Add a GUI button that triggers a download of the current scene state.

        When clicked in the browser, downloads a .viser file that can be
        embedded in static HTML pages.

        Args:
            label (str): Label for the button. Defaults to "Save Recording".

        Returns:
            The button handle from viser.
        """
        save_button = self._server.gui.add_button(label)

        @save_button.on_click
        def _(event):
            if event.client is not None:
                serializer = self._server.get_scene_serializer()
                event.client.send_file_download("recording.viser", serializer.serialize())

        return save_button

    @staticmethod
    def build_static_viewer(output_dir: str):
        """
        Build the static viser client for embedding visualizations.

        This creates the HTML/JS/CSS files needed to host recorded .viser files.
        Requires the viser package to be installed.

        Args:
            output_dir (str): Directory to output the client files.

        Example:
            >>> ViewerViser.build_static_viewer("./viser-client")
            >>> # Then host with: python -m http.server 8000
            >>> # Open: http://localhost:8000/viser-client/?playbackPath=http://localhost:8000/recording.viser
        """
        import subprocess  # noqa: PLC0415
        import sys  # noqa: PLC0415

        result = subprocess.run(
            [sys.executable, "-m", "viser.scripts.build_client", "--out-dir", output_dir],
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            # Try alternative command
            result = subprocess.run(
                ["viser-build-client", "--out-dir", output_dir],
                capture_output=True,
                text=True,
                check=False,
            )

        if result.returncode == 0:
            print(f"Static viewer built in: {output_dir}")
            print("\nTo use:")
            print("  1. Place your .viser recording in a 'recordings/' folder")
            print("  2. Start a local server: python -m http.server 8000")
            print(
                f"  3. Open: http://localhost:8000/{output_dir}/?playbackPath=http://localhost:8000/recordings/your_recording.viser"
            )
        else:
            raise RuntimeError(f"Failed to build static viewer: {result.stderr}")

    def get_embed_html(
        self,
        recording_url: str,
        width: int = 800,
        height: int = 600,
        client_url: str = "viser-client",
    ) -> str:
        """
        Get HTML for embedding a recorded visualization.

        Args:
            recording_url (str): URL to the .viser recording file.
            width (int): Width of the embedded viewer in pixels.
            height (int): Height of the embedded viewer in pixels.
            client_url (str): URL/path to the viser client files.

        Returns:
            str: HTML string with embedded iframe.

        Example:
            >>> html = viewer.get_embed_html(
            ...     recording_url="https://example.com/recording.viser", client_url="https://example.com/viser-client"
            ... )
        """
        full_url = f"{client_url}/?playbackPath={recording_url}"
        return f'<iframe src="{full_url}" width="{width}" height="{height}" frameborder="0"></iframe>'

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
        # Remove existing lines if present
        if name in self._scene_handles:
            try:
                self._scene_handles[name].remove()
            except Exception:
                pass

        if hidden:
            return

        if starts is None or ends is None:
            return

        starts_np = self._to_numpy(starts)
        ends_np = self._to_numpy(ends)

        if starts_np is None or ends_np is None or len(starts_np) == 0:
            return

        # Viser expects line segments as (N, 2, 3) or we can use points format
        # Build line points array: interleave starts and ends
        num_lines = len(starts_np)
        line_points = np.zeros((num_lines * 2, 3), dtype=np.float32)
        line_points[0::2] = starts_np
        line_points[1::2] = ends_np

        # Process colors
        if colors is not None:
            colors_np = self._to_numpy(colors)
            if colors_np is not None:
                if colors_np.ndim == 1 and len(colors_np) == 3:
                    # Single color for all lines
                    color_rgb = tuple((colors_np * 255).astype(np.uint8).tolist())
                else:
                    # Per-line colors - expand to per-point
                    color_rgb = (0, 255, 0)  # Default green
            else:
                color_rgb = (0, 255, 0)
        else:
            color_rgb = (0, 255, 0)

        # Add line segments to viser
        handle = self._server.scene.add_line_segments(
            name=name,
            points=line_points,
            colors=color_rgb,
            line_width=width * 100,  # Scale for visibility
        )
        self._scene_handles[name] = handle

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
        # Viser doesn't have direct array visualization, store for potential future use
        self._scalars[name] = self._to_numpy(array)

    @override
    def log_scalar(self, name, value):
        """
        Log a scalar value for visualization.

        Args:
            name (str): Name of the scalar.
            value: The scalar value.
        """
        if name is None:
            return

        if hasattr(value, "item"):
            val = value.item()
        else:
            val = value

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
        """Log geometry primitives."""
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
        # Remove existing points if present
        if name in self._scene_handles:
            try:
                self._scene_handles[name].remove()
            except Exception:
                pass

        if hidden:
            return

        if points is None:
            return

        pts = self._to_numpy(points)
        n_points = pts.shape[0]

        if n_points == 0:
            return

        # Handle radii (point size)
        if radii is not None:
            size = self._to_numpy(radii)
            if size.ndim == 0 or size.shape == ():
                point_size = float(size)
            elif len(size) == n_points:
                point_size = float(np.mean(size))  # Use average for uniform size
            else:
                point_size = 0.1
        else:
            point_size = 0.1

        # Handle colors
        if colors is not None:
            cols = self._to_numpy(colors)
            if cols.shape == (n_points, 3):
                # Convert from 0-1 to 0-255
                colors_val = (cols * 255).astype(np.uint8)
            elif cols.shape == (3,):
                colors_val = np.tile((cols * 255).astype(np.uint8), (n_points, 1))
            else:
                colors_val = np.full((n_points, 3), 255, dtype=np.uint8)
        else:
            colors_val = np.full((n_points, 3), 255, dtype=np.uint8)

        # Add point cloud to viser
        handle = self._server.scene.add_point_cloud(
            name=name,
            points=pts.astype(np.float32),
            colors=colors_val,
            point_size=point_size,
            point_shape="circle",
        )
        self._scene_handles[name] = handle

    def show_notebook(self, width: int = 800, height: int = 600):
        """
        Show the viewer in a Jupyter notebook.

        This displays an iframe containing the viser visualization.

        Args:
            width (int): Width of the viewer in pixels.
            height (int): Height of the viewer in pixels.
        """
        from IPython.display import IFrame, display  # noqa: PLC0415

        # Use share URL if available, otherwise use localhost
        if self._share_url:
            url = self._share_url
        else:
            url = f"http://localhost:{self._port}"

        display(IFrame(src=url, width=width, height=height))

    def save_html(self, filepath: str, width: int = 800, height: int = 600):
        """
        Save the current visualization as a static HTML file.

        This creates an HTML file with an embedded iframe pointing to the viewer.
        Note: The viewer server must be running for the HTML to work.

        For a fully self-contained static export, consider using the viewer's
        screenshot functionality or exporting mesh data separately.

        Args:
            filepath (str): Path to save the HTML file.
            width (int): Width of the embedded viewer in pixels.
            height (int): Height of the embedded viewer in pixels.
        """
        if self._share_url:
            url = self._share_url
        else:
            url = f"http://localhost:{self._port}"

        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Newton Viewer - Viser</title>
    <style>
        body {{
            margin: 0;
            padding: 0;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            background-color: #1a1a2e;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }}
        .container {{
            text-align: center;
        }}
        h1 {{
            color: #eee;
            margin-bottom: 20px;
        }}
        iframe {{
            border: 2px solid #4a4a6a;
            border-radius: 8px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        }}
        .note {{
            color: #888;
            margin-top: 15px;
            font-size: 14px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Newton Physics Viewer</h1>
        <iframe src="{url}" width="{width}" height="{height}" frameborder="0"></iframe>
        <p class="note">Note: The Newton viewer server must be running for this visualization to work.</p>
    </div>
</body>
</html>
"""
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html_content)

        if self.verbose:
            print(f"HTML saved to: {filepath}")

    def get_notebook_html(self, width: int = 800, height: int = 600) -> str:
        """
        Get HTML string for embedding in a Jupyter notebook.

        This can be used with IPython.display.HTML to embed the viewer.

        Args:
            width (int): Width of the viewer in pixels.
            height (int): Height of the viewer in pixels.

        Returns:
            str: HTML string with embedded iframe.
        """
        if self._share_url:
            url = self._share_url
        else:
            url = f"http://localhost:{self._port}"

        return f'<iframe src="{url}" width="{width}" height="{height}" frameborder="0"></iframe>'

    def _ipython_display_(self):
        """
        Display the viewer in an IPython notebook when the viewer is at the end of a cell.
        """
        self.show_notebook()


def _is_jupyter_notebook():
    """Check if running in a Jupyter notebook environment."""
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True
        elif shell == "TerminalInteractiveShell":
            return False
        else:
            return False
    except NameError:
        return False


@wp.kernel
def _compute_normals(
    points: wp.array(dtype=wp.vec3),
    indices: wp.array(dtype=wp.int32),
    # output
    normals: wp.array(dtype=wp.vec3),
):
    """Compute vertex normals from mesh faces."""
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
