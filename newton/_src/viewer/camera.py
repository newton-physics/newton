# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import numpy as np


class Camera:
    """Camera class that encapsulates all camera settings and logic."""

    DEFAULT_PIVOT_DISTANCE = 5.0
    MIN_PIVOT_DISTANCE = 0.05

    def __init__(self, fov=45.0, near=0.01, far=1000.0, width=1280, height=720, pos=None, up_axis="Z"):
        """
        Initialize camera with given parameters.

        Args:
            fov (float): Field of view in degrees
            near (float): Near clipping plane
            far (float): Far clipping plane
            width (int): Screen width
            height (int): Screen height
            pos (tuple): Initial camera position (if None, uses appropriate default for up_axis)
            up_axis (str): Up axis ("X", "Y", or "Z")
        """
        from pyglet.math import Vec3 as PyVec3

        self.fov = fov
        self.near = near
        self.far = far
        self.width = width
        self.height = height

        # Handle up axis properly first
        if isinstance(up_axis, int):
            self.up_axis = up_axis
        else:
            self.up_axis = "XYZ".index(up_axis.upper())

        # Set appropriate defaults based on up_axis
        if pos is None:
            if self.up_axis == 0:  # X up
                pos = (2.0, 0.0, 10.0)  # 2 units up in X, 10 units back in Z
            elif self.up_axis == 2:  # Z up
                pos = (10.0, 0.0, 2.0)  # 2 units up in Z, 10 units back in Y
            else:  # Y up (default)
                pos = (0.0, 2.0, 10.0)  # 2 units up in Y, 10 units back in Z

        # Camera position
        self.pos = PyVec3(*pos)

        # Camera orientation - this is what users can modify
        self.pitch = 0.0
        self.yaw = -180.0

        self.pivot = self.pos + self.get_front() * self.DEFAULT_PIVOT_DISTANCE

    @staticmethod
    def _as_vec3(value):
        """Convert a 3D sequence to pyglet's Vec3."""
        from pyglet.math import Vec3 as PyVec3

        return PyVec3(float(value[0]), float(value[1]), float(value[2]))

    @staticmethod
    def _length(value) -> float:
        """Return the Euclidean length of a 3D vector."""
        return float(np.linalg.norm((value[0], value[1], value[2])))

    @staticmethod
    def _clamp_pitch(pitch: float) -> float:
        return max(min(float(pitch), 89.0), -89.0)

    @staticmethod
    def _wrap_yaw(yaw: float) -> float:
        return (float(yaw) + 180.0) % 360.0 - 180.0

    @property
    def pivot_distance(self) -> float:
        """Distance from the camera position to the orbit pivot [m]."""
        distance = self._length(self.pivot - self.pos)
        return max(distance, self.MIN_PIVOT_DISTANCE)

    def _set_orientation_from_direction(self, direction):
        """Set yaw and pitch from a normalized view direction."""
        direction = self._as_vec3(direction).normalize()

        if self.up_axis == 0:  # X up
            pitch = np.rad2deg(np.arcsin(np.clip(direction.x, -1.0, 1.0)))
            yaw = np.rad2deg(np.arctan2(direction.z, direction.y))
        elif self.up_axis == 2:  # Z up
            pitch = np.rad2deg(np.arcsin(np.clip(direction.z, -1.0, 1.0)))
            yaw = np.rad2deg(np.arctan2(direction.y, direction.x))
        else:  # Y up (default)
            pitch = np.rad2deg(np.arcsin(np.clip(direction.y, -1.0, 1.0)))
            yaw = np.rad2deg(np.arctan2(direction.z, direction.x))

        self.pitch = self._clamp_pitch(pitch)
        self.yaw = self._wrap_yaw(yaw)

    def set_pivot(self, pivot):
        """Set the orbit pivot without changing the current view direction."""
        self.pivot = self._as_vec3(pivot)
        if self._length(self.pivot - self.pos) <= self.MIN_PIVOT_DISTANCE:
            self.sync_pivot_to_view(distance=self.MIN_PIVOT_DISTANCE)

    def sync_pivot_to_view(self, distance: float | None = None):
        """Place the orbit pivot along the current view direction.

        Args:
            distance: Optional distance from the camera to the pivot [m].
        """
        if distance is None:
            distance = self.pivot_distance
        distance = max(float(distance), self.MIN_PIVOT_DISTANCE)
        self.pivot = self.pos + self.get_front() * distance

    def look_at(self, target):
        """Point the camera at a world-space target and set it as the pivot."""
        target = self._as_vec3(target)
        to_target = target - self.pos
        distance = self._length(to_target)
        if distance <= self.MIN_PIVOT_DISTANCE:
            self.set_pivot(target)
            return

        self._set_orientation_from_direction(to_target)
        self.pivot = target

    def translate(self, delta):
        """Translate the camera and pivot by the same world-space offset [m]."""
        delta = self._as_vec3(delta)
        self.pos += delta
        self.pivot += delta

    def orbit(self, delta_yaw: float, delta_pitch: float):
        """Orbit the camera around the pivot.

        Args:
            delta_yaw: Yaw delta in degrees.
            delta_pitch: Pitch delta in degrees.
        """
        distance = self.pivot_distance
        self.yaw = self._wrap_yaw(self.yaw + delta_yaw)
        self.pitch = self._clamp_pitch(self.pitch + delta_pitch)
        self.pos = self.pivot - self.get_front() * distance

    def pan(self, delta_right: float, delta_up: float):
        """Pan the camera and pivot in the camera plane [m]."""
        delta = self.get_right() * float(delta_right) + self.get_up() * float(delta_up)
        self.translate(delta)

    def dolly(self, amount: float):
        """Move the camera toward or away from the pivot.

        Positive values move the camera toward the pivot; negative values move it
        away. The pivot remains fixed.
        """
        distance = self.pivot_distance
        to_pivot = self.pivot - self.pos
        if self._length(to_pivot) <= self.MIN_PIVOT_DISTANCE:
            self.sync_pivot_to_view(distance=distance)
            to_pivot = self.pivot - self.pos

        direction_to_pivot = to_pivot.normalize()
        self._set_orientation_from_direction(direction_to_pivot)

        new_distance = max(distance * float(np.exp(-amount)), self.MIN_PIVOT_DISTANCE)
        self.pos = self.pivot - direction_to_pivot * new_distance

    def get_front(self):
        """Get the camera front direction vector (read-only)."""
        from pyglet.math import Vec3 as PyVec3

        # Clamp pitch to avoid gimbal lock
        pitch = self._clamp_pitch(self.pitch)

        # Calculate front vector directly in the coordinate system based on up_axis
        # This ensures yaw/pitch work correctly for each coordinate system

        if self.up_axis == 0:  # X up
            # Yaw rotates around X (vertical), pitch is elevation
            front_x = np.sin(np.deg2rad(pitch))
            front_y = np.cos(np.deg2rad(self.yaw)) * np.cos(np.deg2rad(pitch))
            front_z = np.sin(np.deg2rad(self.yaw)) * np.cos(np.deg2rad(pitch))
            return PyVec3(front_x, front_y, front_z).normalize()

        elif self.up_axis == 2:  # Z up
            # Yaw rotates around Z (vertical), pitch is elevation
            front_x = np.cos(np.deg2rad(self.yaw)) * np.cos(np.deg2rad(pitch))
            front_y = np.sin(np.deg2rad(self.yaw)) * np.cos(np.deg2rad(pitch))
            front_z = np.sin(np.deg2rad(pitch))
            return PyVec3(front_x, front_y, front_z).normalize()

        else:  # Y up (default)
            # Yaw rotates around Y (vertical), pitch is elevation
            front_x = np.cos(np.deg2rad(self.yaw)) * np.cos(np.deg2rad(pitch))
            front_y = np.sin(np.deg2rad(pitch))
            front_z = np.sin(np.deg2rad(self.yaw)) * np.cos(np.deg2rad(pitch))
            return PyVec3(front_x, front_y, front_z).normalize()

    def get_right(self):
        """Get the camera right direction vector (read-only)."""
        from pyglet.math import Vec3 as PyVec3

        return PyVec3.cross(self.get_front(), self.get_up()).normalize()

    def get_up(self):
        """Get the camera up direction vector (read-only)."""
        from pyglet.math import Vec3 as PyVec3

        # World up vector based on up axis
        if self.up_axis == 0:  # X up
            world_up = PyVec3(1.0, 0.0, 0.0)
        elif self.up_axis == 2:  # Z up
            world_up = PyVec3(0.0, 0.0, 1.0)
        else:  # Y up (default)
            world_up = PyVec3(0.0, 1.0, 0.0)

        # Compute right vector and use it to get proper up vector
        front = self.get_front()
        right = PyVec3.cross(front, world_up).normalize()
        return PyVec3.cross(right, front).normalize()

    def get_view_matrix(self, scaling=1.0):
        """
        Compute view matrix handling up axis properly.

        Args:
            scaling (float): Scene scaling factor

        Returns:
            np.ndarray: 4x4 view matrix
        """
        from pyglet.math import Mat4, Vec3

        # Get camera vectors (already transformed for up axis)
        pos = Vec3(*(self.pos / scaling))
        front = Vec3(*self.get_front())
        up = Vec3(*self.get_up())

        return np.array(Mat4.look_at(pos, pos + front, up), dtype=np.float32)

    def get_projection_matrix(self):
        """
        Compute projection matrix.

        Returns:
            np.ndarray: 4x4 projection matrix
        """
        from pyglet.math import Mat4 as PyMat4

        if self.height == 0:
            return np.eye(4, dtype=np.float32)

        aspect_ratio = self.width / self.height
        return np.array(PyMat4.perspective_projection(aspect_ratio, self.near, self.far, self.fov))

    def get_world_ray(self, x: float, y: float):
        """Get the world ray for a given pixel.

        returns:
            p: wp.vec3, ray origin
            d: wp.vec3, ray direction
        """
        from pyglet.math import Vec3 as PyVec3

        aspect_ratio = self.width / self.height

        # pre-compute factor from vertical FOV
        fov_rad = np.radians(self.fov)
        alpha = float(np.tan(fov_rad * 0.5))  # = tan(fov/2)

        # build an orthonormal basis (front, right, up)
        front = self.get_front()
        right = self.get_right()
        up = self.get_up()

        # normalised pixel coordinates
        u = 2.0 * (x / self.width) - 1.0  # [-1, 1] left → right
        v = 2.0 * (y / self.height) - 1.0  # [-1, 1] bottom → top

        # ray direction in world space (before normalisation)
        direction = front + right * u * alpha * aspect_ratio + up * v * alpha
        direction = direction / float(np.linalg.norm(direction))

        return self.pos, PyVec3(*direction)

    def update_screen_size(self, width, height):
        """Update screen dimensions."""
        self.width = width
        self.height = height
