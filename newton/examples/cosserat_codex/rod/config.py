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

"""Rod configuration and batch containers for Cosserat rod simulation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .base import RodStateBase


@dataclass
class RodConfig:
    """Configuration parameters for a single Cosserat rod.

    Attributes:
        num_points: Number of particles in the rod.
        segment_length: Rest length of each segment.
        particle_mass: Mass of each particle.
        particle_height: Initial Z height of particles.
        rod_radius: Radius of the rod for collision detection.
        bend_stiffness: Bending stiffness coefficient.
        twist_stiffness: Twist stiffness coefficient.
        rest_bend_d1: Rest curvature in d1 direction.
        rest_bend_d2: Rest curvature in d2 direction.
        rest_twist: Rest twist angle.
        young_modulus: Young's modulus for stretch stiffness.
        torsion_modulus: Torsion modulus for twist stiffness.
        gravity: 3D gravity vector.
        lock_root_rotation: Whether to lock root particle rotation.
    """

    num_points: int
    segment_length: float
    particle_mass: float
    particle_height: float
    rod_radius: float
    bend_stiffness: float
    twist_stiffness: float
    rest_bend_d1: float
    rest_bend_d2: float
    rest_twist: float
    young_modulus: float
    torsion_modulus: float
    gravity: np.ndarray
    lock_root_rotation: bool

    def __post_init__(self):
        """Ensure gravity is a numpy array."""
        self.gravity = np.asarray(self.gravity, dtype=np.float32)


class RodBatch:
    """Metadata and per-rod parameters for a batched rod collection.

    This class manages multiple rods with potentially different configurations
    and provides indexing information for batched operations.

    Attributes:
        configs: List of rod configurations.
        rod_count: Number of rods in the batch.
        rod_offsets: Cumulative point offsets for each rod.
        edge_offsets: Cumulative edge offsets for each rod.
        total_points: Total number of particles across all rods.
        total_edges: Total number of edges across all rods.
        particle_rod_id: Rod index for each particle.
        edge_rod_id: Rod index for each edge.
        gravity: Stacked gravity vectors for all rods.
        bend_stiffness: Bend stiffness for each rod.
        twist_stiffness: Twist stiffness for each rod.
        rest_bend_d1: Rest bend d1 for each rod.
        rest_bend_d2: Rest bend d2 for each rod.
        rest_twist: Rest twist for each rod.
    """

    def __init__(self, configs: Sequence[RodConfig]):
        """Initialize a batch of rod configurations.

        Args:
            configs: Sequence of rod configurations.

        Raises:
            ValueError: If no configurations are provided.
        """
        if not configs:
            raise ValueError("RodBatch requires at least one RodConfig.")
        self.configs = list(configs)
        self.rod_count = len(self.configs)

        # Compute offset arrays for indexing
        self.rod_offsets = np.zeros(self.rod_count + 1, dtype=np.int32)
        self.edge_offsets = np.zeros(self.rod_count + 1, dtype=np.int32)

        point_cursor = 0
        edge_cursor = 0
        for i, config in enumerate(self.configs):
            self.rod_offsets[i] = point_cursor
            self.edge_offsets[i] = edge_cursor
            point_cursor += config.num_points
            edge_cursor += max(0, config.num_points - 1)
        self.rod_offsets[self.rod_count] = point_cursor
        self.edge_offsets[self.rod_count] = edge_cursor

        self.total_points = point_cursor
        self.total_edges = edge_cursor

        # Create rod ID arrays for each particle and edge
        self.particle_rod_id = np.zeros(self.total_points, dtype=np.int32)
        self.edge_rod_id = np.zeros(self.total_edges, dtype=np.int32)

        for i in range(self.rod_count):
            start = self.rod_offsets[i]
            end = self.rod_offsets[i + 1]
            self.particle_rod_id[start:end] = i

            edge_start = self.edge_offsets[i]
            edge_end = self.edge_offsets[i + 1]
            if edge_end > edge_start:
                self.edge_rod_id[edge_start:edge_end] = i

        # Stack per-rod parameters
        self.gravity = np.stack(
            [np.asarray(config.gravity, dtype=np.float32) for config in self.configs],
            axis=0,
        )
        self.bend_stiffness = np.array(
            [config.bend_stiffness for config in self.configs],
            dtype=np.float32,
        )
        self.twist_stiffness = np.array(
            [config.twist_stiffness for config in self.configs],
            dtype=np.float32,
        )
        self.rest_bend_d1 = np.array(
            [config.rest_bend_d1 for config in self.configs],
            dtype=np.float32,
        )
        self.rest_bend_d2 = np.array(
            [config.rest_bend_d2 for config in self.configs],
            dtype=np.float32,
        )
        self.rest_twist = np.array(
            [config.rest_twist for config in self.configs],
            dtype=np.float32,
        )

    def create_state(
        self,
        device=None,
        use_banded: bool = False,
        use_cuda_graph: bool = False,
    ) -> RodState:
        """Create a RodState containing all rods in the batch.

        Args:
            device: Warp device for GPU arrays.
            use_banded: Whether to use banded solver.
            use_cuda_graph: Whether to use CUDA graph capture.

        Returns:
            RodState containing WarpResidentRodState instances for each rod.
        """
        from .warp_rod import WarpResidentRodState

        rods = []
        for config in self.configs:
            rod = WarpResidentRodState(
                num_points=config.num_points,
                segment_length=config.segment_length,
                mass=config.particle_mass,
                particle_height=config.particle_height,
                rod_radius=config.rod_radius,
                bend_stiffness=config.bend_stiffness,
                twist_stiffness=config.twist_stiffness,
                rest_bend_d1=config.rest_bend_d1,
                rest_bend_d2=config.rest_bend_d2,
                rest_twist=config.rest_twist,
                young_modulus=config.young_modulus,
                torsion_modulus=config.torsion_modulus,
                gravity=config.gravity,
                lock_root_rotation=config.lock_root_rotation,
                use_banded=use_banded,
                device=device,
            )
            rod.set_solver_mode(use_banded)
            rod.set_use_cuda_graph(use_cuda_graph)
            rods.append(rod)
        return RodState(self, rods)


class RodState:
    """Holds per-rod device state for a batched simulation.

    This container manages multiple rod state objects and provides
    batch-level operations like reset and parameter updates.

    Attributes:
        batch: The RodBatch that describes the rod configurations.
        rods: List of individual rod state objects.
        total_points: Total number of particles across all rods.
        total_edges: Total number of edges across all rods.
    """

    def __init__(self, batch: RodBatch, rods: list[RodStateBase]):
        """Initialize a RodState container.

        Args:
            batch: The RodBatch configuration.
            rods: List of rod state objects matching the batch configuration.

        Raises:
            ValueError: If rod count doesn't match batch configuration.
        """
        if len(rods) != batch.rod_count:
            raise ValueError("RodState rod count must match RodBatch.")
        self.batch = batch
        self.rods = list(rods)
        self.total_points = batch.total_points
        self.total_edges = batch.total_edges

    def destroy(self) -> None:
        """Destroy all rod state objects and free resources."""
        for rod in self.rods:
            rod.destroy()

    def reset(self) -> None:
        """Reset all rods to their initial state."""
        for rod in self.rods:
            rod.reset()

    def set_use_cuda_graph(self, enabled: bool) -> bool:
        """Enable or disable CUDA graph capture for all rods.

        Args:
            enabled: Whether to enable CUDA graph capture.

        Returns:
            Whether CUDA graph is active (may be False if not supported).
        """
        active = enabled
        for rod in self.rods:
            rod.set_use_cuda_graph(enabled)
            if not rod.use_cuda_graph:
                active = False
        return active

    def set_parallel_kernels(self, enabled: bool) -> None:
        """Toggle between parallel and sequential kernel implementations.

        Args:
            enabled: If True, use parallel GPU kernels. If False, use legacy
                     sequential single-threaded kernels for comparison.
        """
        for rod in self.rods:
            rod.set_parallel_kernels(enabled)

    def set_solver_mode(self, use_banded: bool) -> bool:
        """Set solver mode for all rods.

        Args:
            use_banded: Whether to use banded solver.

        Returns:
            Actual solver mode (may differ if not supported).
        """
        for rod in self.rods:
            rod.set_solver_mode(use_banded)
        return self.rods[0].use_banded if self.rods else use_banded

    def set_gravity(self, gravity: np.ndarray) -> None:
        """Set gravity for all rods.

        Args:
            gravity: 3D gravity vector.
        """
        for rod in self.rods:
            rod.set_gravity(gravity)

    def set_bend_stiffness(self, bend_stiffness: float, twist_stiffness: float) -> None:
        """Set bend and twist stiffness for all rods.

        Args:
            bend_stiffness: Bending stiffness coefficient.
            twist_stiffness: Twist stiffness coefficient.
        """
        for rod in self.rods:
            rod.set_bend_stiffness(bend_stiffness, twist_stiffness)

    def set_rest_darboux(self, rest_bend_d1: float, rest_bend_d2: float, rest_twist: float) -> None:
        """Set rest Darboux vector for all rods.

        Args:
            rest_bend_d1: Rest curvature in d1 direction.
            rest_bend_d2: Rest curvature in d2 direction.
            rest_twist: Rest twist angle.
        """
        for rod in self.rods:
            rod.set_rest_darboux(rest_bend_d1, rest_bend_d2, rest_twist)

    def apply_floor_collisions(self, floor_z: float, restitution: float = 0.0) -> None:
        """Apply floor collision constraints to all rods.

        Args:
            floor_z: Z coordinate of the floor plane.
            restitution: Coefficient of restitution for bouncing.
        """
        for rod in self.rods:
            rod.apply_floor_collisions(floor_z, restitution)


__all__ = ["RodBatch", "RodConfig", "RodState"]
