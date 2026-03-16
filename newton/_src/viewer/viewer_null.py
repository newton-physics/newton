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

import time as _time
from typing import Any

import numpy as np
import warp as wp

import newton

from ..core.types import nparray, override
from .viewer import ViewerBase


class ViewerNull(ViewerBase):
    """
    A no-operation (no-op) viewer implementation for Newton.

    This class provides a minimal, non-interactive viewer that does not perform any rendering
    or visualization. It is intended for use in headless or automated worlds where
    visualization is not required. The viewer runs for a fixed number of frames and provides
    stub implementations for all logging and frame management methods.
    """

    def __init__(
        self,
        num_frames: int = 1000,
        benchmark: bool = False,
        benchmark_timeout: float | None = None,
        benchmark_start_frame: int = 3,
    ):
        """
        Initialize a no-op Viewer that runs for a fixed number of frames.

        Args:
            num_frames: The number of frames to run before stopping.
            benchmark: Enable benchmark timing (FPS measurement after warmup).
            benchmark_timeout: If set, stop after this many seconds of
                steady-state simulation (measured after warmup). Implicitly
                enables *benchmark*.
            benchmark_start_frame: Number of warmup frames before benchmark
                timing starts.
        """
        super().__init__()

        self.num_frames = num_frames
        self.frame_count = 0

        self.benchmark = benchmark or benchmark_timeout is not None
        self.benchmark_timeout = benchmark_timeout
        self.benchmark_start_frame = benchmark_start_frame
        self._bench_start_time: float | None = None
        self._bench_frames = 0
        self._bench_elapsed = 0.0

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
        No-op implementation for logging a mesh.

        Args:
            name: Name of the mesh.
            points: Vertex positions.
            indices: Mesh indices.
            normals: Vertex normals (optional).
            uvs: Texture coordinates (optional).
            texture: Optional texture path/URL or image array.
            hidden: Whether the mesh is hidden.
            backface_culling: Whether to enable backface culling.
        """
        pass

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
        No-op implementation for logging mesh instances.

        Args:
            name: Name of the instance batch.
            mesh: Mesh object.
            xforms: Instance transforms.
            scales: Instance scales.
            colors: Instance colors.
            materials: Instance materials.
            hidden: Whether the instances are hidden.
        """
        pass

    @override
    def begin_frame(self, time: float):
        """
        No-op implementation for beginning a frame.

        Args:
            time: The current simulation time.
        """
        pass

    @override
    def end_frame(self):
        """
        Increment the frame count at the end of each frame.
        """
        self.frame_count += 1

        if self.benchmark:
            if self.frame_count == self.benchmark_start_frame:
                wp.synchronize()
                self._bench_start_time = _time.perf_counter()
            elif self._bench_start_time is not None:
                wp.synchronize()
                self._bench_frames = self.frame_count - self.benchmark_start_frame
                self._bench_elapsed = _time.perf_counter() - self._bench_start_time

    @override
    def is_running(self) -> bool:
        """
        Check if the viewer should continue running.

        Returns:
            bool: True if the frame count is less than the maximum number of frames
            and the benchmark timeout (if any) has not been reached.
        """
        if self.frame_count >= self.num_frames:
            return False
        if (
            self.benchmark_timeout is not None
            and self._bench_start_time is not None
            and self._bench_elapsed >= self.benchmark_timeout
        ):
            return False
        return True

    def benchmark_result(self) -> dict[str, float | int] | None:
        """Return benchmark results, or ``None`` if benchmarking was not enabled.

        Returns:
            Dictionary with ``fps``, ``frames``, and ``elapsed`` keys,
            or ``None`` if benchmarking is not enabled.
        """
        if not self.benchmark:
            return None
        if self._bench_frames == 0 or self._bench_elapsed == 0.0:
            return {"fps": 0.0, "frames": 0, "elapsed": 0.0}
        return {
            "fps": self._bench_frames / self._bench_elapsed,
            "frames": self._bench_frames,
            "elapsed": self._bench_elapsed,
        }

    @override
    def close(self):
        """
        No-op implementation for closing the viewer.
        """
        pass

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
        No-op implementation for logging lines.

        Args:
            name: Name of the line batch.
            starts: Line start points.
            ends: Line end points.
            colors: Line colors.
            width: Line width hint.
            hidden: Whether the lines are hidden.
        """
        pass

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
        No-op implementation for logging points.

        Args:
            name: Name of the point batch.
            points: Point positions.
            radii: Point radii.
            colors: Point colors.
            hidden: Whether the points are hidden.
        """
        pass

    @override
    def log_array(self, name: str, array: wp.array(dtype=Any) | nparray):
        """
        No-op implementation for logging a generic array.

        Args:
            name: Name of the array.
            array: The array data.
        """
        pass

    @override
    def log_scalar(self, name: str, value: int | float | bool | np.number):
        """
        No-op implementation for logging a scalar value.

        Args:
            name: Name of the scalar.
            value: The scalar value.
        """
        pass

    @override
    def apply_forces(self, state: newton.State):
        """Null backend does not apply interactive forces.

        Args:
            state: Current simulation state.
        """
        pass
