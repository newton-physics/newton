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

from pathlib import Path
from typing import ClassVar

import numpy as np
import warp as wp

import newton
from newton.utils import create_plane_mesh

from ..core.types import override
from .viewer import ViewerBase, is_jupyter_notebook


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
                raise ImportError("viser package is required for ViewerViser. Install with: pip install viser") from e
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
        record_to_viser: str | None = None,
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
            record_to_viser (str | None): Path to record the viewer to a ``*.viser`` recording file
                (e.g. "my_recording.viser"). If None, the viewer will not record to a file.
        """
        viser = self._get_viser()

        super().__init__()

        self._running = True
        self.verbose = verbose

        # Store mesh data for instances
        self._meshes = {}
        self._instances = {}
        self._scene_handles = {}  # Track viser scene node handles

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
        self.is_jupyter_notebook = is_jupyter_notebook()

        # Recording state
        self._frame_dt = 0.0
        self._record_to_viser = record_to_viser
        self._serializer = self._server.get_scene_serializer() if record_to_viser else None

        # Set up default scene
        self._setup_scene()

        if self._serializer is not None and verbose:
            print(f"Recording to: {record_to_viser}")

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

        self._server.scene.add_light_ambient("ambient_light")

        # remove HDR map
        self._server.scene.configure_environment_map(hdri=None)

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
            normals (wp.array, optional): Vertex normals, unused in viser (wp.vec3).
            uvs (wp.array, optional): UV coordinates, unused in viser (wp.vec2).
            hidden (bool): Whether the mesh is hidden.
            backface_culling (bool): Whether to enable backface culling.
        """
        assert isinstance(points, wp.array)
        assert isinstance(indices, wp.array)

        # Convert to numpy arrays
        points_np = self._to_numpy(points).astype(np.float32)
        indices_np = self._to_numpy(indices).astype(np.uint32)

        # Viser expects indices as (N, 3) for triangles
        if indices_np.ndim == 1:
            indices_np = indices_np.reshape(-1, 3)

        # Store mesh data for instancing
        self._meshes[name] = {
            "points": points_np,
            "indices": indices_np,
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
            batched_colors = None  # Will use cached colors or default gray

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
                    # Only update colors if they were explicitly provided
                    if batched_colors is not None:
                        handle.batched_colors = batched_colors
                        # Cache the colors for future reference
                        self._instances[name]["colors"] = batched_colors
                    return
                except Exception:
                    # If update fails, recreate the mesh
                    try:
                        handle.remove()
                    except Exception:
                        pass
                    del self._scene_handles[name]
                    del self._instances[name]

        # For new instances, use provided colors or default gray
        if batched_colors is None:
            batched_colors = np.full((num_instances, 3), 180, dtype=np.uint8)

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
            "colors": batched_colors,  # Cache the colors
        }

    @override
    def begin_frame(self, time):
        """
        Begin a new frame.

        Args:
            time (float): The current simulation time.
        """
        self._frame_dt = time - self.time
        self.time = time

    @override
    def end_frame(self):
        """
        End the current frame.

        If recording is active, inserts a sleep command for playback timing.
        """
        if self._serializer is not None:
            # Insert sleep for frame timing during recording
            self._serializer.insert_sleep(self._frame_dt)

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
            if self._serializer is not None:
                self.save_recording()
        except Exception:
            pass

    def save_recording(self):
        """
        Save the current recording to a .viser file.

        The recording can be played back in a static HTML viewer.
        See build_static_viewer() for creating the HTML player.

        Note:
            Recording must be enabled by passing ``record_to_viser`` to the constructor.

        Example:
            >>> viewer = ViewerViser(record_to_viser="my_simulation.viser")
            >>> # ... run simulation ...
            >>> viewer.save_recording()
        """
        if self._serializer is None or self._record_to_viser is None:
            raise RuntimeError("No recording in progress. Pass record_to_viser to the constructor.")

        from pathlib import Path  # noqa: PLC0415

        data = self._serializer.serialize()
        Path(self._record_to_viser).write_bytes(data)

        self._serializer = None

        if self.verbose:
            print(f"Recording saved to: {self._record_to_viser}")

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

    def show_notebook(self, width: int | str = "100%", height: int | str = 400):
        """
        Show the viewer in a Jupyter notebook.

        If recording is active, saves the recording and displays using the static HTML
        viewer with timeline controls. Otherwise, displays the live server in an IFrame.

        Args:
            width: Width of the embedded player in pixels.
            height: Height of the embedded player in pixels.

        Returns:
            The display object.

        Example:
            >>> viewer = newton.viewer.ViewerViser(record_to_viser="my_sim.viser")
            >>> viewer.set_model(model)
            >>> # ... run simulation ...
            >>> viewer.show_notebook()  # Saves recording and displays with timeline
        """

        from IPython.display import HTML, IFrame, display  # noqa: PLC0415

        from .viewer import is_sphinx_build  # noqa: PLC0415

        if self._record_to_viser is None:
            # No recording - display the live server via IFrame
            return display(IFrame(src=self.url, width=width, height=height))

        if self._serializer is not None:
            # Recording is active - save it first
            recording_path = Path(self._record_to_viser)
            recording_path.parent.mkdir(parents=True, exist_ok=True)
            self.save_recording()

        # Check if recording path contains _static - indicates Sphinx docs build
        recording_str = str(self._record_to_viser).replace("\\", "/")

        if is_sphinx_build():
            # Sphinx build - use static HTML with viser player
            # The recording path needs to be relative to the viser index.html location
            # which is at _static/viser/index.html

            # Find the _static portion of the path
            static_idx = recording_str.find("_static/")
            if static_idx == -1:
                raise ValueError(
                    f"Recordings that are supposed to appear in the Sphinx documentation must be stored in docs/_static/, but the path {recording_str} does not contain _static/"
                )
            else:
                # Extract path from _static onwards (e.g., "_static/recordings/foo.viser")
                static_relative = recording_str[static_idx:]
                # The viser index.html is at _static/viser/index.html
                # So from there, we need "../recordings/foo.viser"
                # Remove the "_static/" prefix and prepend "../"
                playback_path = "../" + static_relative[len("_static/") :]

            embed_html = f"""
<div class="viser-player-container" style="margin: 20px 0;">
<iframe
    src="../_static/viser/index.html?playbackPath={playback_path}"
    width="{width}"
    height="{height}"
    frameborder="0"
    style="border: 1px solid #ccc; border-radius: 8px;">
</iframe>
</div>
"""
            return display(HTML(embed_html))
        else:
            # Regular Jupyter - use local HTTP server with viser client
            player_url = self._serve_viser_recording(self._record_to_viser)
            return display(IFrame(src=player_url, width=width, height=height))

    def _ipython_display_(self):
        """
        Display the viewer in an IPython notebook when the viewer is at the end of a cell.
        """
        self.show_notebook()

    @staticmethod
    def _serve_viser_recording(recording_path: str) -> str:
        """
        Hosts a simple HTTP server to serve the viser recording file with the viser client
        and returns the URL of the player.

        Args:
            recording_path: Path to the .viser recording file.

        Returns:
            URL of the player.
        """
        import socket  # noqa: PLC0415
        import threading  # noqa: PLC0415
        from http.server import HTTPServer, SimpleHTTPRequestHandler  # noqa: PLC0415

        # Get viser client directory from docs/_static/viser
        # This is a curated version of the viser client for notebook playback
        import newton  # noqa: PLC0415

        recording_path = Path(recording_path).resolve()
        if not recording_path.exists():
            raise FileNotFoundError(f"Recording file not found: {recording_path}")

        newton_root = Path(newton.__file__).parent.parent
        viser_client_dir = newton_root / "docs" / "_static" / "viser"

        if not viser_client_dir.exists():
            raise FileNotFoundError(
                f"Viser client not found at {viser_client_dir}. "
                "Please ensure docs/_static/viser contains the viser client files."
            )

        # Read the recording file content
        recording_bytes = recording_path.read_bytes()

        # Find an available port
        def find_free_port():
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("", 0))
                return s.getsockname()[1]

        port = find_free_port()

        # Create a custom HTTP handler factory that serves both viser client and the recording
        def make_handler(recording_data: bytes, client_dir: str):
            class RecordingHandler(SimpleHTTPRequestHandler):
                # Fix MIME types for JavaScript and other files (Windows often has wrong mappings)
                extensions_map: ClassVar = {  # pyright: ignore[reportIncompatibleVariableOverride]
                    **SimpleHTTPRequestHandler.extensions_map,
                    ".html": "text/html",
                    ".htm": "text/html",
                    ".css": "text/css",
                    ".js": "application/javascript",
                    ".json": "application/json",
                    ".wasm": "application/wasm",
                    ".svg": "image/svg+xml",
                    ".png": "image/png",
                    ".jpg": "image/jpeg",
                    ".jpeg": "image/jpeg",
                    ".ico": "image/x-icon",
                    ".ttf": "font/ttf",
                    ".hdr": "application/octet-stream",
                    ".viser": "application/octet-stream",
                    "": "application/octet-stream",
                }

                def __init__(self, *args, **kwargs):
                    self.recording_data = recording_data
                    super().__init__(*args, directory=client_dir, **kwargs)

                def do_GET(self):
                    # Parse path without query string
                    path = self.path.split("?")[0]

                    # Serve the recording file at /recording.viser
                    if path == "/recording.viser":
                        self.send_response(200)
                        self.send_header("Content-Type", "application/octet-stream")
                        self.send_header("Content-Length", str(len(self.recording_data)))
                        self.send_header("Access-Control-Allow-Origin", "*")
                        self.end_headers()
                        self.wfile.write(self.recording_data)
                    else:
                        # Serve viser client files
                        super().do_GET()

                def log_message(self, format, *args):
                    pass  # Suppress log messages

            return RecordingHandler

        handler_class = make_handler(recording_bytes, str(viser_client_dir))
        # Bind to all interfaces so IFrame can access it
        server = HTTPServer(("", port), handler_class)

        # Start server in background thread
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()

        base_url = f"http://127.0.0.1:{port}"

        # Create URL with playback path pointing to the served recording
        player_url = f"{base_url}/?playbackPath=/recording.viser"

        return player_url
