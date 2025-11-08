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

import math

import numpy as np
import warp as wp

from ..geometry.raycast import raycast_sensor_kernel, raycast_sensor_particles_kernel
from ..sim import Model, State


@wp.kernel
def clamp_no_hits_kernel(depth_image: wp.array(dtype=float), max_dist: float):
    """Kernel to replace max_distance values with -1.0 to indicate no intersection."""
    tid = wp.tid()
    if depth_image[tid] >= max_dist:
        depth_image[tid] = -1.0


class RaycastSensor:
    """Raycast-based depth sensor for generating depth images.

    The RaycastSensor simulates a depth camera by casting rays from a virtual camera through each pixel
    in an image. For each pixel, it finds the closest intersection with the scene geometry and records
    the distance as a depth value.

    The sensor supports perspective cameras with configurable field of view, aspect ratio, and resolution.
    The resulting depth image has the same resolution as specified, with depth values representing the
    distance from the camera to the closest surface along each ray.

    .. rubric:: Camera Coordinate System

    The camera uses a right-handed coordinate system where:
    - The forward direction (camera_direction) is the direction the camera is looking
    - The up direction (camera_up) defines the camera's vertical orientation
    - The right direction (camera_right) is computed as the cross product of forward and up

    .. rubric:: Depth Values

    - Positive depth values: Distance to the closest surface
    - Negative depth values (-1.0): No intersection found (ray missed all geometry)

    Attributes:
        device: The device (CPU/GPU) where computations are performed
        camera_position: 3D position of the camera in world space
        camera_direction: Forward direction vector (normalized)
        camera_up: Up direction vector (normalized)
        camera_right: Right direction vector (normalized)
        fov_radians: Vertical field of view in radians
        aspect_ratio: Width/height aspect ratio
        width: Image width in pixels
        height: Image height in pixels
        depth_image: 2D depth image array (height, width)
    """

    def __init__(
        self,
        model: Model,
        camera_position: tuple[float, float, float] | np.ndarray,
        camera_direction: tuple[float, float, float] | np.ndarray,
        camera_up: tuple[float, float, float] | np.ndarray,
        fov_radians: float,
        width: int,
        height: int,
        max_distance: float = 1000.0,
    ):
        """Initialize a RaycastSensor.

        Args:
            model: The Newton model containing the geometry to raycast against
            camera_position: 3D position of the camera in world space
            camera_direction: Forward direction of the camera (will be normalized)
            camera_up: Up direction of the camera (will be normalized)
            fov_radians: Vertical field of view in radians
            width: Image width in pixels
            height: Image height in pixels
            max_distance: Maximum ray distance; rays beyond this return no hit
        """
        self.model = model
        self.device = model.device
        self.width = width
        self.height = height
        self.fov_radians = fov_radians
        self.aspect_ratio = float(width) / float(height)
        self.max_distance = max_distance

        # Set initial camera parameters
        self.camera_position = np.array(camera_position, dtype=np.float32)
        camera_dir = np.array(camera_direction, dtype=np.float32)
        camera_up = np.array(camera_up, dtype=np.float32)

        # Pre-compute field of view scale
        self.fov_scale = math.tan(fov_radians * 0.5)

        # Create depth image buffer
        self._depth_buffer = wp.zeros((height, width), dtype=float, device=self.device)
        self.depth_image = self._depth_buffer
        self._resolution = wp.vec2(float(width), float(height))

        # Compute camera basis vectors and warp vectors
        self._compute_camera_basis(camera_dir, camera_up)

        # Lazily constructed structure for particle queries
        self._particle_grid: wp.HashGrid | None = None

    def _compute_camera_basis(self, direction: np.ndarray, up: np.ndarray):
        """Compute orthonormal camera basis vectors and update warp vectors.

        Args:
            direction: Camera direction vector (will be normalized)
            up: Camera up vector (will be normalized)
        """
        # Normalize direction vectors
        self.camera_direction = direction / np.linalg.norm(direction)
        self.camera_up = up / np.linalg.norm(up)

        # Compute right vector as cross product of forward and up
        self.camera_right = np.cross(self.camera_direction, self.camera_up)
        right_norm = np.linalg.norm(self.camera_right)
        if right_norm < 1e-8:
            # Camera direction and up are parallel, use a different up vector
            if abs(self.camera_direction[2]) < 0.9:
                temp_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            else:
                temp_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            self.camera_right = np.cross(self.camera_direction, temp_up)
            right_norm = np.linalg.norm(self.camera_right)
        self.camera_right = self.camera_right / right_norm

        # Recompute up vector to ensure orthogonality
        self.camera_up = np.cross(self.camera_right, self.camera_direction)
        self.camera_up = self.camera_up / np.linalg.norm(self.camera_up)

    def eval(
        self,
        state: State,
        include_particles: bool = False,
        particle_march_step: float | None = None,
    ):
        """Evaluate the raycast sensor to generate a depth image.

        Casts rays from the camera through each pixel and records the distance to the closest
        intersection with the scene geometry. When ``include_particles`` is enabled (not enabled by default),
        particles stored in the simulation state are also considered.

        Args:
            state: The current state of the simulation containing body poses
            include_particles: Whether to test ray intersections against particles present in ``state``
            particle_march_step: Optional stride used when marching along each ray during particle queries.
                Defaults to half of the maximum particle radius when particles are available.
        """

        if include_particles and particle_march_step is not None and particle_march_step <= 0.0:
            raise ValueError("particle_march_step must be positive when provided.")

        # Reset depth buffer to maximum distance
        self._depth_buffer.fill_(self.max_distance)
        num_shapes = len(self.model.shape_body)

        if (include_particles and self._does_state_have_particles(state)) or num_shapes != 0:
            camera_position = wp.vec3(*self.camera_position)
            camera_direction = wp.vec3(*self.camera_direction)
            camera_up = wp.vec3(*self.camera_up)
            camera_right = wp.vec3(*self.camera_right)

        # Launch raycast kernel for each pixel-shape combination
        # We use 3D launch with dimensions (width, height, num_shapes)
        if num_shapes > 0:
            wp.launch(
                kernel=raycast_sensor_kernel,
                dim=(self.width, self.height, num_shapes),
                inputs=[
                    # Model data
                    state.body_q,
                    self.model.shape_body,
                    self.model.shape_transform,
                    self.model.shape_type,
                    self.model.shape_scale,
                    self.model.shape_source_ptr,
                    # Camera parameters
                    camera_position,
                    camera_direction,
                    camera_up,
                    camera_right,
                    self.fov_scale,
                    self.aspect_ratio,
                    self._resolution,
                ],
                outputs=[self._depth_buffer],
                device=self.device,
            )

        if include_particles and self._does_state_have_particles(state):
            self._raycast_particles(
                state=state,
                camera_position=camera_position,
                camera_direction=camera_direction,
                camera_up=camera_up,
                camera_right=camera_right,
                march_step=particle_march_step,
            )

        # Set pixels that still have max_distance to -1.0 to indicate no hit
        self._clamp_no_hits()

    def _get_particle_grid(self) -> wp.HashGrid:
        """Return a hash grid for particle queries, constructing it lazily."""
        if self._particle_grid is None:
            with wp.ScopedDevice(self.device):
                self._particle_grid = wp.HashGrid(128, 128, 128)
        return self._particle_grid

    def _raycast_particles(
        self,
        state: State,
        camera_position: wp.vec3,
        camera_direction: wp.vec3,
        camera_up: wp.vec3,
        camera_right: wp.vec3,
        march_step: float | None,
    ) -> None:
        """Intersect rays with particles using a spatial hash grid."""

        particle_positions = state.particle_q
        particle_count = state.particle_count
        particle_radius = self.model.particle_radius
        max_radius = float(self.model.particle_max_radius)

        search_radius = max_radius + 1.0e-6
        step = march_step if march_step is not None else 0.5 * search_radius

        grid = self._get_particle_grid()
        with wp.ScopedDevice(self.device):
            grid.reserve(particle_count)
            grid.build(particle_positions, radius=search_radius)

        wp.launch(
            kernel=raycast_sensor_particles_kernel,
            dim=(self.width, self.height),
            inputs=[
                grid.id,
                particle_positions,
                particle_radius,
                float(search_radius),
                float(step),
                camera_position,
                camera_direction,
                camera_up,
                camera_right,
                float(self.fov_scale),
                float(self.aspect_ratio),
                self._resolution,
                float(self.max_distance),
            ],
            outputs=[self._depth_buffer],
            device=self.device,
        )

    def _does_state_have_particles(self, state: State) -> bool:
        """Check if the given state has particles available for raycasting."""
        particle_positions = state.particle_q
        if particle_positions is None or state.particle_count == 0:
            return False

        if self.model.particle_radius is None or self.model.particle_max_radius <= 0.0:
            raise ValueError("Model must have valid particle radius to raycast when particles are present.")

        return True

    def _clamp_no_hits(self):
        """Replace max_distance values with -1.0 to indicate no intersection."""
        # Flatten the depth buffer for linear indexing
        flattened_buffer = self._depth_buffer.flatten()

        wp.launch(
            kernel=clamp_no_hits_kernel,
            dim=self.height * self.width,
            inputs=[flattened_buffer, self.max_distance],
            device=self.device,
        )

    def get_depth_image(self) -> wp.array2d:
        """Get the depth image as a 2D array.

        Returns:
            2D depth image array with shape (height, width). Values are:
            - Positive: Distance to closest surface
            - -1.0: No intersection found
        """
        return self.depth_image

    def get_depth_image_numpy(self) -> np.ndarray:
        """Get the depth image as a numpy array.

        Returns:
            Numpy array with shape (height, width) containing depth values.
            Values are the same as get_depth_image() but as a numpy array.
        """
        return self.depth_image.numpy()

    def update_camera_pose(
        self,
        position: tuple[float, float, float] | np.ndarray | None = None,
        direction: tuple[float, float, float] | np.ndarray | None = None,
        up: tuple[float, float, float] | np.ndarray | None = None,
    ):
        """Update the camera pose parameters.

        Args:
            position: New camera position (if provided)
            direction: New camera direction (if provided, will be normalized)
            up: New camera up vector (if provided, will be normalized)
        """
        if position is not None:
            self.camera_position = np.array(position, dtype=np.float32)

        if direction is not None or up is not None:
            # Use current values if not provided
            camera_dir = np.array(direction, dtype=np.float32) if direction is not None else self.camera_direction
            camera_up = np.array(up, dtype=np.float32) if up is not None else self.camera_up

            # Recompute camera basis using shared method
            self._compute_camera_basis(camera_dir, camera_up)

    def update_camera_parameters(
        self,
        fov_radians: float | None = None,
        width: int | None = None,
        height: int | None = None,
        max_distance: float | None = None,
    ):
        """Update camera intrinsic parameters.

        Args:
            fov_radians: New vertical field of view in radians
            width: New image width in pixels
            height: New image height in pixels
            max_distance: New maximum ray distance
        """
        recreate_buffer = False

        if width is not None and width != self.width:
            self.width = width
            recreate_buffer = True

        if height is not None and height != self.height:
            self.height = height
            recreate_buffer = True

        if fov_radians is not None:
            self.fov_radians = fov_radians
            self.fov_scale = math.tan(fov_radians * 0.5)

        if max_distance is not None:
            self.max_distance = max_distance

        if recreate_buffer:
            self.aspect_ratio = float(self.width) / float(self.height)
            self._resolution = wp.vec2(float(self.width), float(self.height))
            self._depth_buffer = wp.zeros((self.height, self.width), dtype=float, device=self.device)
            self.depth_image = self._depth_buffer

    def point_camera_at(
        self,
        target: tuple[float, float, float] | np.ndarray,
        position: tuple[float, float, float] | np.ndarray | None = None,
        up: tuple[float, float, float] | np.ndarray | None = None,
    ):
        """Point the camera at a specific target location.

        Args:
            target: 3D point to look at
            position: New camera position (if provided)
            up: Up vector for camera orientation (default: [0, 0, 1])
        """
        if position is not None:
            self.camera_position = np.array(position, dtype=np.float32)
        if up is None:
            up = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        target = np.array(target, dtype=np.float32)
        direction = target - self.camera_position

        self.update_camera_pose(
            position=self.camera_position,
            direction=direction,
            up=up,
        )
