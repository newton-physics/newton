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

import inspect
import os
import warnings
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import warp as wp

import newton

from ..core.types import nparray, override
from ..utils.texture import load_texture, normalize_texture
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
                import viser

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

    @staticmethod
    def _prepare_texture(texture: np.ndarray | str | None) -> np.ndarray | None:
        """Load and normalize texture data for viser/glTF usage."""
        return normalize_texture(
            load_texture(texture),
            flip_vertical=False,
            require_channels=True,
            scale_unit_range=True,
        )

    @staticmethod
    def _build_trimesh_mesh(points: np.ndarray, indices: np.ndarray, uvs: np.ndarray, texture: np.ndarray):
        """Create a trimesh object with texture visuals (if trimesh is available)."""
        try:
            import trimesh
        except Exception:
            return None

        if len(uvs) != len(points):
            return None

        faces = indices.astype(np.int64)
        mesh = trimesh.Trimesh(vertices=points, faces=faces, process=False)

        try:
            from PIL import Image
            from trimesh.visual.texture import TextureVisuals

            image = Image.fromarray(texture)
            mesh.visual = TextureVisuals(uv=uvs, image=image)
        except Exception:
            visual_mod = getattr(trimesh, "visual", None)
            TextureVisuals = getattr(visual_mod, "TextureVisuals", None) if visual_mod is not None else None
            if TextureVisuals is not None:
                mesh.visual = TextureVisuals(uv=uvs, image=texture)

        return mesh

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
            port: Port number for the web server. Defaults to 8080.
            label: Optional label for the viser server window title.
            verbose: If True, print the server URL when starting. Defaults to True.
            share: If True, create a publicly accessible URL via viser's share feature.
            record_to_viser: Path to record the viewer to a ``*.viser`` recording file
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
        self._camera_request: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        self._pending_camera_clients: set[int] = set()
        self._server.on_client_connect(self._handle_client_connect)
        self._server.on_client_disconnect(self._handle_client_disconnect)

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

        self._server.scene.add_light_ambient("ambient_light")

        # remove HDR map
        self._server.scene.configure_environment_map(hdri=None)

    @staticmethod
    def _call_scene_method(method, **kwargs):
        """Call a viser scene method with only supported keyword args."""
        try:
            signature = inspect.signature(method)
            allowed = {k: v for k, v in kwargs.items() if k in signature.parameters}
            return method(**allowed)
        except Exception:
            return method(**kwargs)

    @property
    def url(self) -> str:
        """Get the URL of the viser server.

        Returns:
            str: Local HTTP URL for the running viser server.
        """
        return f"http://localhost:{self._port}"

    @staticmethod
    def _is_client_camera_ready(client: Any) -> bool:
        """Return True if the client has reported an initial camera state."""
        try:
            update_timestamp = float(client.camera.update_timestamp)
        except Exception:
            # Older viser versions may not expose update_timestamp.
            try:
                _ = client.camera.position
            except Exception:
                return False
            return True
        return update_timestamp > 0.0

    def _handle_client_connect(self, client: Any):
        """Apply cached camera settings to newly connected clients."""
        self._pending_camera_clients.discard(int(client.client_id))
        self._apply_camera_to_client(client)

    def _handle_client_disconnect(self, client: Any):
        """Clear pending camera state for disconnected clients."""
        self._pending_camera_clients.discard(int(client.client_id))

    def _get_camera_up_axis(self) -> int:
        """Resolve the model up-axis as an integer index (0, 1, 2)."""
        if self.model is None:
            return 2
        up_axis = self.model.up_axis
        if isinstance(up_axis, str):
            return "XYZ".index(up_axis.upper())
        return int(up_axis)

    def _compute_camera_front_up(self, pitch: float, yaw: float) -> tuple[np.ndarray, np.ndarray]:
        """Compute camera front and up vectors from pitch/yaw angles."""
        pitch = float(np.clip(pitch, -89.0, 89.0))
        yaw = float(yaw)
        pitch_rad = np.deg2rad(pitch)
        yaw_rad = np.deg2rad(yaw)
        up_axis = self._get_camera_up_axis()

        if up_axis == 0:  # X up
            front = np.array(
                [
                    np.sin(pitch_rad),
                    np.cos(yaw_rad) * np.cos(pitch_rad),
                    np.sin(yaw_rad) * np.cos(pitch_rad),
                ],
                dtype=np.float64,
            )
            world_up = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        elif up_axis == 2:  # Z up
            front = np.array(
                [
                    np.cos(yaw_rad) * np.cos(pitch_rad),
                    np.sin(yaw_rad) * np.cos(pitch_rad),
                    np.sin(pitch_rad),
                ],
                dtype=np.float64,
            )
            world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        else:  # Y up
            front = np.array(
                [
                    np.cos(yaw_rad) * np.cos(pitch_rad),
                    np.sin(pitch_rad),
                    np.sin(yaw_rad) * np.cos(pitch_rad),
                ],
                dtype=np.float64,
            )
            world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)

        front /= np.linalg.norm(front)
        right = np.cross(front, world_up)
        right_norm = np.linalg.norm(right)
        if right_norm < 1.0e-8:
            fallback_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            if np.linalg.norm(np.cross(front, fallback_up)) < 1.0e-8:
                fallback_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            right = np.cross(front, fallback_up)
            right_norm = np.linalg.norm(right)
        right /= right_norm

        up = np.cross(right, front)
        up /= np.linalg.norm(up)
        return front, up

    def _apply_camera_to_client(self, client: Any):
        """Apply the cached camera request to a connected client if ready."""
        if self._camera_request is None:
            return

        client_id = int(client.client_id)
        if not self._is_client_camera_ready(client):
            if client_id in self._pending_camera_clients:
                return

            self._pending_camera_clients.add(client_id)

            def _on_camera_update(_camera: Any):
                if client_id not in self._pending_camera_clients:
                    return
                self._pending_camera_clients.discard(client_id)
                self._apply_camera_to_client(client)

            client.camera.on_update(_on_camera_update)
            return

        self._pending_camera_clients.discard(client_id)
        position, look_at, up_direction = self._camera_request

        # Keep camera updates synchronized to avoid transient jitter.
        if hasattr(client, "atomic"):
            with client.atomic():
                client.camera.position = tuple(position.tolist())
                client.camera.look_at = tuple(look_at.tolist())
                client.camera.up_direction = tuple(up_direction.tolist())
        else:
            client.camera.position = tuple(position.tolist())
            client.camera.look_at = tuple(look_at.tolist())
            client.camera.up_direction = tuple(up_direction.tolist())

    @override
    def set_camera(self, pos: wp.vec3, pitch: float, yaw: float):
        """Set camera position and orientation for connected Viser clients.

        The requested view is also cached so that newly connected clients receive
        the same camera setup as soon as they report camera state.

        Args:
            pos: Requested camera position.
            pitch: Requested camera pitch angle.
            yaw: Requested camera yaw angle.
        """
        position = np.asarray((float(pos[0]), float(pos[1]), float(pos[2])), dtype=np.float64)
        front, up_direction = self._compute_camera_front_up(pitch, yaw)
        look_at = position + front
        self._camera_request = (position, look_at, up_direction)

        if hasattr(self._server, "initial_camera"):
            self._server.initial_camera.position = tuple(position.tolist())
            self._server.initial_camera.look_at = tuple(look_at.tolist())
            if hasattr(self._server.initial_camera, "up"):
                self._server.initial_camera.up = tuple(up_direction.tolist())
            elif hasattr(self._server.initial_camera, "up_direction"):
                self._server.initial_camera.up_direction = tuple(up_direction.tolist())

        for client in self._server.get_clients().values():
            self._apply_camera_to_client(client)

    @staticmethod
    def _camera_query_from_request(camera_request: tuple[np.ndarray, np.ndarray, np.ndarray] | None) -> str:
        """Build URL query parameters for playback initial camera overrides."""
        if camera_request is None:
            return ""

        position, look_at, up_direction = camera_request

        def _format_vec3(values: np.ndarray) -> str:
            values = np.asarray(values, dtype=np.float64).reshape(3)
            return ",".join(f"{float(v):.9g}" for v in values)

        return (
            f"&initialCameraPosition={_format_vec3(position)}"
            f"&initialCameraLookAt={_format_vec3(look_at)}"
            f"&initialCameraUp={_format_vec3(up_direction)}"
        )

    @override
    def log_mesh(
        self,
        name: str,
        points: wp.array(dtype=wp.vec3),
        indices: wp.array(dtype=wp.int32) | wp.array(dtype=wp.uint32),
        normals: wp.array(dtype=wp.vec3) | None = None,
        uvs: wp.array(dtype=wp.vec2) | None = None,
        texture: np.ndarray | str | None = None,
        hidden: bool = False,
        backface_culling: bool = True,
    ):
        """
        Log a mesh to viser for visualization.

        Args:
            name: Entity path for the mesh.
            points: Vertex positions.
            indices: Triangle indices.
            normals: Vertex normals, unused in viser.
            uvs: UV coordinates, used for textures if supported.
            texture: Texture path/URL or image array (H, W, C).
            hidden: Whether the mesh is hidden.
            backface_culling: Whether to enable backface culling.
        """
        assert isinstance(points, wp.array)
        assert isinstance(indices, wp.array)

        # Convert to numpy arrays
        points_np = self._to_numpy(points).astype(np.float32)
        indices_np = self._to_numpy(indices).astype(np.uint32)
        uvs_np = self._to_numpy(uvs).astype(np.float32) if uvs is not None else None
        texture_image = self._prepare_texture(texture)

        if texture_image is not None and uvs_np is None:
            warnings.warn(f"Mesh {name} has a texture but no UVs; texture will be ignored.", stacklevel=2)
            texture_image = None
        if texture_image is not None and uvs_np is not None and len(uvs_np) != len(points_np):
            warnings.warn(
                f"Mesh {name} has {len(uvs_np)} UVs for {len(points_np)} vertices; texture will be ignored.",
                stacklevel=2,
            )
            texture_image = None

        # Viser expects indices as (N, 3) for triangles
        if indices_np.ndim == 1:
            indices_np = indices_np.reshape(-1, 3)

        trimesh_mesh = None
        if texture_image is not None and uvs_np is not None:
            trimesh_mesh = self._build_trimesh_mesh(points_np, indices_np, uvs_np, texture_image)
            if trimesh_mesh is None:
                warnings.warn(
                    "Viser textured meshes require trimesh; falling back to untextured rendering.",
                    stacklevel=2,
                )

        # Store mesh data for instancing
        self._meshes[name] = {
            "points": points_np,
            "indices": indices_np,
            "uvs": uvs_np,
            "texture": texture_image,
            "trimesh": trimesh_mesh,
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
        if trimesh_mesh is not None:
            handle = self._call_scene_method(
                self._server.scene.add_mesh_trimesh,
                name=name,
                mesh=trimesh_mesh,
            )
        else:
            handle = self._call_scene_method(
                self._server.scene.add_mesh_simple,
                name=name,
                vertices=points_np,
                faces=indices_np,
                color=(180, 180, 180),  # Default gray color
                wireframe=False,
                side="double" if not backface_culling else "front",
            )
        self._scene_handles[name] = handle

    @override
    def log_instances(
        self,
        name: str,
        mesh: str,
        xforms: wp.array(dtype=wp.transform) | None,
        scales: wp.array(dtype=wp.vec3) | None,
        colors: wp.array(dtype=wp.vec3) | None,
        materials: wp.array(dtype=wp.vec4) | None,
        hidden: bool = False,
    ):
        """
        Log instanced mesh data to viser using efficient batched rendering.

        Uses viser's add_batched_meshes_simple for GPU-accelerated instanced rendering.
        Supports in-place updates of transforms for real-time animation.

        Args:
            name: Entity path for the instances.
            mesh: Name of the mesh asset to instance.
            xforms: Instance transforms.
            scales: Instance scales.
            colors: Instance colors.
            materials: Instance materials.
            hidden: Whether the instances are hidden.
        """
        # Check that mesh exists
        if mesh not in self._meshes:
            raise RuntimeError(f"Mesh {mesh} not found. Call log_mesh first.")

        mesh_data = self._meshes[mesh]
        base_points = mesh_data["points"]
        base_indices = mesh_data["indices"]
        base_uvs = mesh_data.get("uvs")
        texture_image = self._prepare_texture(mesh_data.get("texture"))
        trimesh_mesh = mesh_data.get("trimesh")

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
        use_trimesh = trimesh_mesh is not None and texture_image is not None and base_uvs is not None
        if name in self._instances and name in self._scene_handles:
            # Update existing batched mesh in-place (much faster)
            handle = self._scene_handles[name]
            prev_count = self._instances[name]["count"]
            prev_use_trimesh = self._instances[name].get("use_trimesh", False)

            # If instance count changed, we need to recreate the mesh
            if prev_count != num_instances or prev_use_trimesh != use_trimesh:
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
                    if hasattr(handle, "batched_scales"):
                        handle.batched_scales = batched_scales
                    # Only update colors if they were explicitly provided
                    if batched_colors is not None and hasattr(handle, "batched_colors"):
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
        if use_trimesh:
            handle = self._call_scene_method(
                self._server.scene.add_batched_meshes_trimesh,
                name=name,
                mesh=trimesh_mesh,
                batched_positions=positions,
                batched_wxyzs=quats_wxyz,
                batched_scales=batched_scales,
                lod="auto",
            )
        else:
            handle = self._call_scene_method(
                self._server.scene.add_batched_meshes_simple,
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
            "use_trimesh": use_trimesh,
        }

    @override
    def begin_frame(self, time: float):
        """
        Begin a new frame.

        Args:
            time: The current simulation time.
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

    @override
    def apply_forces(self, state: newton.State):
        """Viser backend does not apply interactive forces.

        Args:
            state: Current simulation state.
        """
        pass

    def save_recording(self):
        """
        Save the current recording to a .viser file.

        The recording can be played back in a static HTML viewer.
        See build_static_viewer() for creating the HTML player.

        Note:
            Recording must be enabled by passing ``record_to_viser`` to the constructor.

        Example:

            .. code-block:: python

                viewer = ViewerViser(record_to_viser="my_simulation.viser")
                # ... run simulation ...
                viewer.save_recording()
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
    def log_lines(
        self,
        name: str,
        starts: wp.array(dtype=wp.vec3) | None,
        ends: wp.array(dtype=wp.vec3) | None,
        colors: (
            wp.array(dtype=wp.vec3) | wp.array(dtype=wp.float32) | tuple[float, float, float] | list[float] | None
        ),
        width: float = 0.01,
        hidden: bool = False,
    ):
        """
        Log lines for visualization.

        Args:
            name: Name of the line batch.
            starts: Line start points.
            ends: Line end points.
            colors: Line colors.
            width: Line width.
            hidden: Whether the lines are hidden.
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

        starts_np = np.asarray(starts_np, dtype=np.float32)
        ends_np = np.asarray(ends_np, dtype=np.float32)
        num_lines = len(starts_np)

        # Viser requires points with shape (N, 2, 3): [start, end] per segment.
        line_points = np.stack((starts_np, ends_np), axis=1)

        def _rgb_to_uint8_array(rgb: np.ndarray) -> np.ndarray:
            rgb = np.asarray(rgb, dtype=np.float32)
            max_val = float(np.max(rgb)) if rgb.size > 0 else 0.0
            if max_val <= 1.0:
                rgb = rgb * 255.0
            return np.clip(rgb, 0, 255).astype(np.uint8)

        # Process colors
        color_rgb: tuple[int, int, int] | np.ndarray = (0, 255, 0)
        if colors is not None:
            colors_np = self._to_numpy(colors)
            if colors_np is not None:
                colors_np = np.asarray(colors_np)
                if colors_np.ndim == 1 and colors_np.shape[0] == 3:
                    # Single color for all lines.
                    color_rgb = tuple(_rgb_to_uint8_array(colors_np).tolist())
                elif colors_np.ndim == 2 and colors_np.shape == (num_lines, 3):
                    # Per-line colors: repeat each line color for [start, end].
                    line_colors = _rgb_to_uint8_array(colors_np)
                    color_rgb = np.repeat(line_colors[:, None, :], 2, axis=1)
                elif colors_np.ndim == 3 and colors_np.shape == (num_lines, 2, 3):
                    # Already per-point-per-segment colors.
                    color_rgb = _rgb_to_uint8_array(colors_np)

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
        name: str,
        geo_type: int,
        geo_scale: tuple[float, ...],
        geo_thickness: float,
        geo_is_solid: bool,
        geo_src: newton.Mesh | newton.Heightfield | None = None,
        hidden: bool = False,
    ):
        """Log a geometry primitive, with plane expansion for infinite planes.

        Args:
            name: Unique path/name for the geometry asset.
            geo_type: Geometry type value from `newton.GeoType`.
            geo_scale: Geometry scale tuple interpreted by `geo_type`.
            geo_thickness: Shell thickness for mesh-like geometry.
            geo_is_solid: Whether mesh geometry is treated as solid.
            geo_src: Optional source geometry for mesh-backed types.
            hidden: Whether the resulting geometry is hidden.
        """
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
            mesh = newton.Mesh.create_plane(width, length, compute_inertia=False)
            points = wp.array(mesh.vertices, dtype=wp.vec3, device=self.device)
            normals = wp.array(mesh.normals, dtype=wp.vec3, device=self.device)
            uvs = wp.array(mesh.uvs, dtype=wp.vec2, device=self.device)
            indices = wp.array(mesh.indices, dtype=wp.int32, device=self.device)
            self.log_mesh(name, points, indices, normals, uvs, hidden=hidden)
        else:
            super().log_geo(name, geo_type, geo_scale, geo_thickness, geo_is_solid, geo_src, hidden)

    @override
    def log_points(
        self,
        name: str,
        points: wp.array(dtype=wp.vec3) | None,
        radii: wp.array(dtype=wp.float32) | float | None = None,
        colors: (
            wp.array(dtype=wp.vec3) | wp.array(dtype=wp.float32) | tuple[float, float, float] | list[float] | None
        ) = None,
        hidden: bool = False,
    ):
        """
        Log points for visualization.

        Args:
            name: Name of the point batch.
            points: Point positions (can be a wp.array or a numpy array).
            radii: Point radii (can be a wp.array or a numpy array).
            colors: Point colors (can be a wp.array or a numpy array).
            hidden: Whether the points are hidden.
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

    @override
    def log_array(self, name: str, array: wp.array(dtype=Any) | nparray):
        """Viser viewer does not visualize generic arrays.

        Args:
            name: Unique path/name for the array signal.
            array: Array data to visualize.
        """
        pass

    @override
    def log_scalar(self, name: str, value: int | float | bool | np.number):
        """Viser viewer does not visualize scalar signals.

        Args:
            name: Unique path/name for the scalar signal.
            value: Scalar value to visualize.
        """
        pass

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

            .. code-block:: python

                viewer = newton.viewer.ViewerViser(record_to_viser="my_sim.viser")
                viewer.set_model(model)
                # ... run simulation ...
                viewer.show_notebook()  # Saves recording and displays with timeline
        """

        from IPython.display import HTML, IFrame, display

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

            camera_query = self._camera_query_from_request(self._camera_request)

            embed_html = f"""
<div class="viser-player-container" style="margin: 20px 0;">
<iframe
    src="../_static/viser/index.html?playbackPath={playback_path}{camera_query}"
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
            player_url = self._serve_viser_recording(
                self._record_to_viser,
                camera_request=self._camera_request,
            )
            return display(IFrame(src=player_url, width=width, height=height))

    def _ipython_display_(self):
        """
        Display the viewer in an IPython notebook when the viewer is at the end of a cell.
        """
        self.show_notebook()

    @staticmethod
    def _serve_viser_recording(
        recording_path: str, camera_request: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    ) -> str:
        """
        Hosts a simple HTTP server to serve the viser recording file with the viser client
        and returns the URL of the player.

        Args:
            recording_path: Path to the .viser recording file.
            camera_request: Optional `(position, look_at, up_direction)` triple used
                to append initial camera URL overrides for playback.

        Returns:
            URL of the player.
        """
        import socket  # noqa: PLC0415
        import threading  # noqa: PLC0415
        from http.server import HTTPServer, SimpleHTTPRequestHandler  # noqa: PLC0415

        # Get viser client directory (bundled with package at newton/_src/viewer/static/viser)
        recording_path = Path(recording_path).resolve()
        if not recording_path.exists():
            raise FileNotFoundError(f"Recording file not found: {recording_path}")

        viser_client_dir = Path(__file__).parent / "viser" / "static"

        if not viser_client_dir.exists():
            raise FileNotFoundError(
                f"Viser client files not found at {viser_client_dir}. "
                "The notebook playback feature requires the viser client assets."
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

        # Keep playbackPath relative so notebook proxy prefixes (e.g. /lab/proxy/<port>/)
        # are preserved. Each viewer instance uses a different port, so paths stay distinct.
        playback_path = "recording.viser"
        base_url = f"http://127.0.0.1:{port}"
        player_url = f"{base_url}/?playbackPath={playback_path}"

        # Route through Jupyter's proxy only when jupyter-server-proxy is installed.
        # Without that package, proxy URLs may be unavailable and break playback.
        jupyter_base_url = None
        try:
            from importlib.util import find_spec  # noqa: PLC0415

            has_jupyter_server_proxy = find_spec("jupyter_server_proxy") is not None
        except Exception:
            has_jupyter_server_proxy = False

        if has_jupyter_server_proxy:
            # JUPYTER_BASE_URL is not always exported (e.g. CLI --NotebookApp.base_url).
            # In that case, fall back to common env vars and running server metadata.
            for env_name in ("JUPYTER_BASE_URL", "JUPYTERHUB_SERVICE_PREFIX", "NB_PREFIX"):
                candidate = os.environ.get(env_name)
                if candidate:
                    jupyter_base_url = candidate
                    break

            if not jupyter_base_url:
                try:
                    from jupyter_server.serverapp import list_running_servers  # noqa: PLC0415

                    for server in list_running_servers():
                        candidate = server.get("base_url")
                        if candidate:
                            jupyter_base_url = candidate
                            break
                except Exception:
                    pass

            if jupyter_base_url:
                if not jupyter_base_url.startswith("/"):
                    jupyter_base_url = "/" + jupyter_base_url
                if jupyter_base_url != "/":
                    jupyter_base_url = jupyter_base_url.rstrip("/")
                else:
                    jupyter_base_url = ""
                player_url = f"{jupyter_base_url}/proxy/{port}/?playbackPath={playback_path}"

        return player_url + ViewerViser._camera_query_from_request(camera_request)
