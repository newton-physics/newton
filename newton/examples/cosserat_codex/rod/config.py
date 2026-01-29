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
import warp as wp

from newton.examples.cosserat_codex.constants import BAND_LDAB

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
        batched_arrays: Optional BatchedGPUArrays for batched simulation.
        use_batched_step: Whether to use batched step for multiple rods.
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

        # Batched execution support
        self.batched_arrays: BatchedGPUArrays | None = None
        self.use_batched_step = False
        self._batched_graph = None
        self._batched_graph_params = None

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

    def set_use_batched_step(self, enabled: bool, device=None) -> bool:
        """Enable or disable batched step execution.

        When enabled, multiple rods are processed with single kernel launches
        instead of separate launches per rod. This reduces kernel launch overhead.

        Args:
            enabled: Whether to enable batched step execution.
            device: Warp device for batched arrays.

        Returns:
            Whether batched step is actually enabled.
        """
        if enabled and self.batch.rod_count < 2:
            # Batched step doesn't help with single rod
            enabled = False

        if enabled and self.batched_arrays is None:
            self.batched_arrays = BatchedGPUArrays(self.batch, device=device)
            self.batched_arrays.sync_from_rods(self.rods)

        self.use_batched_step = enabled
        self._batched_graph = None
        self._batched_graph_params = None
        return enabled

    def step(self, dt: float, linear_damping: float, angular_damping: float) -> None:
        """Advance simulation by one time step for all rods.

        If batched execution is enabled, uses single kernel launches for all rods.
        Otherwise, steps each rod individually.

        Args:
            dt: Time step size.
            linear_damping: Linear velocity damping factor.
            angular_damping: Angular velocity damping factor.
        """
        if self.use_batched_step and self.batched_arrays is not None:
            self._step_batched_impl(dt, linear_damping, angular_damping)
        else:
            for rod in self.rods:
                rod.step(dt, linear_damping, angular_damping)

    def _step_batched_impl(self, dt: float, linear_damping: float, angular_damping: float) -> None:
        """Internal batched step implementation.

        Uses ~12 kernel launches total regardless of rod count.

        Args:
            dt: Time step size.
            linear_damping: Linear velocity damping factor.
            angular_damping: Angular velocity damping factor.
        """
        from newton.examples.cosserat_codex.kernels import (
            _warp_apply_accumulated_corrections,
            _warp_assemble_jmjt_blocks_batched,
            _warp_block_thomas_solve_batched,
            _warp_build_rhs,
            _warp_compute_corrections_batched,
            _warp_compute_inv_inertia_world_batched,
            _warp_compute_jacobians_batched,
            _warp_integrate_positions_batched,
            _warp_integrate_rotations_batched,
            _warp_predict_positions_batched,
            _warp_predict_rotations_batched,
            _warp_prepare_compliance_batched,
            _warp_update_constraints_batched_v2,
            _warp_zero_float,
            _warp_zero_vec3,
        )

        b = self.batched_arrays
        if b is None:
            return

        device = b.device
        total_points = b.total_points
        total_edges = b.total_edges
        n_rods = b.n_rods
        n_dofs = b.n_dofs

        if total_edges == 0:
            return

        # Sync from individual rods to batched arrays (picks up keyboard/parameter changes)
        b.sync_from_rods(self.rods)

        # ====================================================================
        # 1. Predict positions (1 kernel)
        # ====================================================================
        wp.launch(
            _warp_predict_positions_batched,
            dim=total_points,
            inputs=[
                b.positions_wp,
                b.velocities_wp,
                b.forces_wp,
                b.inv_masses_wp,
                b.gravity_wp,
                b.particle_rod_id_wp,
                float(dt),
                float(linear_damping),
                b.predicted_positions_wp,
            ],
            device=device,
        )

        # ====================================================================
        # 2. Predict rotations (1 kernel)
        # ====================================================================
        wp.launch(
            _warp_predict_rotations_batched,
            dim=total_points,
            inputs=[
                b.orientations_wp,
                b.angular_velocities_wp,
                b.torques_wp,
                b.quat_inv_masses_wp,
                float(dt),
                float(angular_damping),
                b.predicted_orientations_wp,
            ],
            device=device,
        )

        # ====================================================================
        # 3. Prepare compliance (1 kernel)
        # ====================================================================
        wp.launch(
            _warp_zero_float,
            dim=n_dofs,
            inputs=[b.lambda_sum_wp],
            device=device,
        )
        wp.launch(
            _warp_prepare_compliance_batched,
            dim=total_edges,
            inputs=[
                b.rest_lengths_wp,
                b.bend_stiffness_wp,
                b.edge_rod_id_wp,
                b.young_modulus_wp,
                b.torsion_modulus_wp,
                float(dt),
                b.compliance_wp,
            ],
            device=device,
        )

        # ====================================================================
        # 4. Update constraints (1 kernel)
        # ====================================================================
        wp.launch(
            _warp_update_constraints_batched_v2,
            dim=total_edges,
            inputs=[
                b.predicted_positions_wp,
                b.predicted_orientations_wp,
                b.rest_lengths_wp,
                b.rest_darboux_wp,
                b.rod_offsets_wp,
                b.edge_offsets_wp,
                b.edge_rod_id_wp,
                b.constraint_values_wp,
            ],
            device=device,
        )

        # ====================================================================
        # 5. Compute Jacobians (1 kernel)
        # ====================================================================
        wp.launch(
            _warp_compute_jacobians_batched,
            dim=total_edges,
            inputs=[
                b.predicted_orientations_wp,
                b.rest_lengths_wp,
                b.rod_offsets_wp,
                b.edge_offsets_wp,
                b.edge_rod_id_wp,
                b.jacobian_pos_wp,
                b.jacobian_rot_wp,
            ],
            device=device,
        )

        # ====================================================================
        # 6. Compute inverse inertia in world frame (1 kernel)
        # ====================================================================
        wp.launch(
            _warp_compute_inv_inertia_world_batched,
            dim=total_points,
            inputs=[
                b.predicted_orientations_wp,
                b.quat_inv_masses_wp,
                b.inv_inertia_local_diag_wp,
                b.particle_rod_id_wp,
                b.inv_inertia_wp,
            ],
            device=device,
        )

        # ====================================================================
        # 7. Assemble JMJT blocks (1 kernel)
        # ====================================================================
        wp.launch(
            _warp_assemble_jmjt_blocks_batched,
            dim=total_edges,
            inputs=[
                b.jacobian_pos_wp,
                b.jacobian_rot_wp,
                b.compliance_wp,
                b.inv_masses_wp,
                b.inv_inertia_wp,
                b.rod_offsets_wp,
                b.edge_offsets_wp,
                b.edge_rod_id_wp,
                b.diag_blocks_wp,
                b.offdiag_blocks_wp,
            ],
            device=device,
        )

        # ====================================================================
        # 8. Build RHS (1 kernel)
        # ====================================================================
        wp.launch(
            _warp_build_rhs,
            dim=n_dofs,
            inputs=[
                b.constraint_values_wp,
                b.compliance_wp,
                b.lambda_sum_wp,
                int(n_dofs),
                b.rhs_wp,
            ],
            device=device,
        )

        # ====================================================================
        # 9. Batched Thomas solve (1 kernel with n_rods threads)
        # ====================================================================
        wp.launch(
            _warp_block_thomas_solve_batched,
            dim=n_rods,
            inputs=[
                b.diag_blocks_wp,
                b.offdiag_blocks_wp,
                b.rhs_wp,
                b.edge_offsets_wp,
                int(n_rods),
                b.c_blocks_wp,
                b.d_prime_wp,
                b.delta_lambda_wp,
            ],
            device=device,
        )

        # ====================================================================
        # 10. Compute corrections (1 kernel)
        # ====================================================================
        wp.launch(
            _warp_zero_vec3,
            dim=total_points,
            inputs=[b.pos_corrections_wp],
            device=device,
        )
        wp.launch(
            _warp_zero_vec3,
            dim=total_points,
            inputs=[b.rot_corrections_wp],
            device=device,
        )
        wp.launch(
            _warp_zero_float,
            dim=1,
            inputs=[b._delta_lambda_max_wp],
            device=device,
        )
        wp.launch(
            _warp_zero_float,
            dim=1,
            inputs=[b._correction_max_wp],
            device=device,
        )
        wp.launch(
            _warp_compute_corrections_batched,
            dim=total_edges,
            inputs=[
                b.predicted_positions_wp,
                b.inv_masses_wp,
                b.quat_inv_masses_wp,
                b.inv_inertia_wp,
                b.jacobian_pos_wp,
                b.jacobian_rot_wp,
                b.delta_lambda_wp,
                b.lambda_sum_wp,
                b.rod_offsets_wp,
                b.edge_offsets_wp,
                b.edge_rod_id_wp,
                b.pos_corrections_wp,
                b.rot_corrections_wp,
                b._delta_lambda_max_wp,
                b._correction_max_wp,
            ],
            device=device,
        )

        # ====================================================================
        # 11. Apply accumulated corrections (1 kernel)
        # ====================================================================
        wp.launch(
            _warp_apply_accumulated_corrections,
            dim=total_points,
            inputs=[
                b.predicted_positions_wp,
                b.predicted_orientations_wp,
                b.pos_corrections_wp,
                b.rot_corrections_wp,
                int(total_points),
            ],
            device=device,
        )

        # ====================================================================
        # 12. Integrate positions (1 kernel)
        # ====================================================================
        wp.launch(
            _warp_integrate_positions_batched,
            dim=total_points,
            inputs=[
                b.positions_wp,
                b.predicted_positions_wp,
                b.velocities_wp,
                b.inv_masses_wp,
                float(dt),
            ],
            device=device,
        )

        # ====================================================================
        # 13. Integrate rotations (1 kernel)
        # ====================================================================
        wp.launch(
            _warp_integrate_rotations_batched,
            dim=total_points,
            inputs=[
                b.orientations_wp,
                b.predicted_orientations_wp,
                b.prev_orientations_wp,
                b.angular_velocities_wp,
                b.quat_inv_masses_wp,
                float(dt),
            ],
            device=device,
        )

        # Sync results back to individual rods (for rendering and external access)
        b.sync_to_rods(self.rods)

    def sync_batched_to_rods(self) -> None:
        """Synchronize batched arrays back to individual rod states.

        Call this after batched step if you need to access individual rod data.
        """
        if self.batched_arrays is not None:
            self.batched_arrays.sync_to_rods(self.rods)

    def sync_rods_to_batched(self) -> None:
        """Synchronize individual rod states to batched arrays.

        Call this if rod states were modified individually and you want to
        continue with batched simulation.
        """
        if self.batched_arrays is not None:
            self.batched_arrays.sync_from_rods(self.rods)


class BatchedGPUArrays:
    """Concatenated GPU arrays for batched rod simulation.

    This class provides unified GPU-resident storage for all rods in a batch,
    enabling single kernel launches that process all rods simultaneously.
    Arrays are concatenated across rods, with offset arrays for indexing.

    Attributes:
        batch: The RodBatch configuration.
        device: Warp device for GPU arrays.
        total_points: Total particles across all rods.
        total_edges: Total edges across all rods.
        n_rods: Number of rods.
        n_dofs: Total DOFs (6 per edge).
    """

    def __init__(self, batch: RodBatch, device=None):
        """Initialize batched GPU arrays.

        Args:
            batch: RodBatch configuration with rod metadata.
            device: Warp device for GPU arrays (defaults to current device).
        """
        self.batch = batch
        self.device = device or wp.get_device()
        self.total_points = batch.total_points
        self.total_edges = batch.total_edges
        self.n_rods = batch.rod_count
        self.n_dofs = batch.total_edges * 6

        alloc_points = max(1, self.total_points)
        alloc_edges = max(1, self.total_edges)
        alloc_dofs = max(1, self.n_dofs)

        # ====================================================================
        # Per-particle arrays (concatenated across all rods)
        # ====================================================================
        self.positions_wp = wp.zeros(alloc_points, dtype=wp.vec3, device=self.device)
        self.predicted_positions_wp = wp.zeros(alloc_points, dtype=wp.vec3, device=self.device)
        self.velocities_wp = wp.zeros(alloc_points, dtype=wp.vec3, device=self.device)
        self.forces_wp = wp.zeros(alloc_points, dtype=wp.vec3, device=self.device)

        self.orientations_wp = wp.zeros(alloc_points, dtype=wp.quat, device=self.device)
        self.predicted_orientations_wp = wp.zeros(alloc_points, dtype=wp.quat, device=self.device)
        self.prev_orientations_wp = wp.zeros(alloc_points, dtype=wp.quat, device=self.device)
        self.angular_velocities_wp = wp.zeros(alloc_points, dtype=wp.vec3, device=self.device)
        self.torques_wp = wp.zeros(alloc_points, dtype=wp.vec3, device=self.device)

        self.inv_masses_wp = wp.zeros(alloc_points, dtype=wp.float32, device=self.device)
        self.quat_inv_masses_wp = wp.zeros(alloc_points, dtype=wp.float32, device=self.device)

        # Inverse inertia tensors (9 floats per particle)
        self.inv_inertia_wp = wp.zeros(alloc_points * 9, dtype=wp.float32, device=self.device)

        # Correction workspace (per-particle)
        self.pos_corrections_wp = wp.zeros(alloc_points, dtype=wp.vec3, device=self.device)
        self.rot_corrections_wp = wp.zeros(alloc_points, dtype=wp.vec3, device=self.device)

        # ====================================================================
        # Per-edge arrays (concatenated across all rods)
        # ====================================================================
        self.rest_lengths_wp = wp.zeros(alloc_edges, dtype=wp.float32, device=self.device)
        self.rest_darboux_wp = wp.zeros(alloc_edges, dtype=wp.vec3, device=self.device)
        self.bend_stiffness_wp = wp.zeros(alloc_edges, dtype=wp.vec3, device=self.device)

        # Constraint values (6 per edge)
        self.constraint_values_wp = wp.zeros(alloc_dofs, dtype=wp.float32, device=self.device)
        self.compliance_wp = wp.zeros(alloc_dofs, dtype=wp.float32, device=self.device)
        self.lambda_sum_wp = wp.zeros(alloc_dofs, dtype=wp.float32, device=self.device)
        self.rhs_wp = wp.zeros(alloc_dofs, dtype=wp.float32, device=self.device)
        self.delta_lambda_wp = wp.zeros(alloc_dofs, dtype=wp.float32, device=self.device)

        # Jacobians (36 floats per edge)
        self.jacobian_pos_wp = wp.zeros(alloc_edges * 36, dtype=wp.float32, device=self.device)
        self.jacobian_rot_wp = wp.zeros(alloc_edges * 36, dtype=wp.float32, device=self.device)

        # Block Thomas solver workspace
        self.diag_blocks_wp = wp.zeros(alloc_edges * 36, dtype=wp.float32, device=self.device)
        self.offdiag_blocks_wp = wp.zeros(alloc_edges * 36, dtype=wp.float32, device=self.device)
        self.c_blocks_wp = wp.zeros(alloc_edges * 36, dtype=wp.float32, device=self.device)
        self.d_prime_wp = wp.zeros(alloc_edges * 6, dtype=wp.float32, device=self.device)

        # Banded solver workspace
        self.ab_wp = wp.zeros((BAND_LDAB, alloc_dofs), dtype=wp.float32, device=self.device)

        # ====================================================================
        # Per-rod parameter arrays (on GPU for kernel lookups)
        # ====================================================================
        self.gravity_wp = wp.zeros(self.n_rods, dtype=wp.vec3, device=self.device)
        self.young_modulus_wp = wp.zeros(self.n_rods, dtype=wp.float32, device=self.device)
        self.torsion_modulus_wp = wp.zeros(self.n_rods, dtype=wp.float32, device=self.device)
        self.inv_inertia_local_diag_wp = wp.zeros(self.n_rods, dtype=wp.vec3, device=self.device)

        # ====================================================================
        # Offset arrays (on GPU for kernel lookups)
        # ====================================================================
        self.rod_offsets_wp = wp.array(batch.rod_offsets, dtype=wp.int32, device=self.device)
        self.edge_offsets_wp = wp.array(batch.edge_offsets, dtype=wp.int32, device=self.device)
        self.particle_rod_id_wp = wp.array(batch.particle_rod_id, dtype=wp.int32, device=self.device)
        self.edge_rod_id_wp = wp.array(batch.edge_rod_id, dtype=wp.int32, device=self.device)

        # ====================================================================
        # Diagnostics
        # ====================================================================
        self._constraint_max_wp = wp.zeros(1, dtype=wp.float32, device=self.device)
        self._delta_lambda_max_wp = wp.zeros(1, dtype=wp.float32, device=self.device)
        self._correction_max_wp = wp.zeros(1, dtype=wp.float32, device=self.device)

    def sync_from_rods(self, rods: list) -> None:
        """Synchronize batched arrays from individual rod states.

        Copies data from each rod's GPU arrays into the concatenated batched arrays.
        Uses numpy as an intermediate format for reliable cross-array copying.

        Args:
            rods: List of WarpResidentRodState objects.
        """
        # Collect all data into numpy arrays first
        all_positions = []
        all_predicted_positions = []
        all_velocities = []
        all_forces = []
        all_orientations = []
        all_predicted_orientations = []
        all_prev_orientations = []
        all_angular_velocities = []
        all_torques = []
        all_inv_masses = []
        all_quat_inv_masses = []

        all_rest_lengths = []
        all_rest_darboux = []
        all_bend_stiffness = []

        for rod in rods:
            all_positions.append(rod.positions_wp.numpy())
            all_predicted_positions.append(rod.predicted_positions_wp.numpy())
            all_velocities.append(rod.velocities_wp.numpy())
            all_forces.append(rod.forces_wp.numpy())
            all_orientations.append(rod.orientations_wp.numpy())
            all_predicted_orientations.append(rod.predicted_orientations_wp.numpy())
            all_prev_orientations.append(rod.prev_orientations_wp.numpy())
            all_angular_velocities.append(rod.angular_velocities_wp.numpy())
            all_torques.append(rod.torques_wp.numpy())
            all_inv_masses.append(rod.inv_masses_wp.numpy())
            all_quat_inv_masses.append(rod.quat_inv_masses_wp.numpy())

            if rod.num_edges > 0:
                all_rest_lengths.append(rod.rest_lengths_wp.numpy())
                all_rest_darboux.append(rod.rest_darboux_wp.numpy())
                all_bend_stiffness.append(rod.bend_stiffness_wp.numpy())

        # Concatenate and assign to batched arrays
        self.positions_wp.assign(
            wp.array(np.concatenate(all_positions), dtype=wp.vec3, device=self.device)
        )
        self.predicted_positions_wp.assign(
            wp.array(np.concatenate(all_predicted_positions), dtype=wp.vec3, device=self.device)
        )
        self.velocities_wp.assign(
            wp.array(np.concatenate(all_velocities), dtype=wp.vec3, device=self.device)
        )
        self.forces_wp.assign(
            wp.array(np.concatenate(all_forces), dtype=wp.vec3, device=self.device)
        )
        self.orientations_wp.assign(
            wp.array(np.concatenate(all_orientations), dtype=wp.quat, device=self.device)
        )
        self.predicted_orientations_wp.assign(
            wp.array(np.concatenate(all_predicted_orientations), dtype=wp.quat, device=self.device)
        )
        self.prev_orientations_wp.assign(
            wp.array(np.concatenate(all_prev_orientations), dtype=wp.quat, device=self.device)
        )
        self.angular_velocities_wp.assign(
            wp.array(np.concatenate(all_angular_velocities), dtype=wp.vec3, device=self.device)
        )
        self.torques_wp.assign(
            wp.array(np.concatenate(all_torques), dtype=wp.vec3, device=self.device)
        )
        self.inv_masses_wp.assign(
            wp.array(np.concatenate(all_inv_masses), dtype=wp.float32, device=self.device)
        )
        self.quat_inv_masses_wp.assign(
            wp.array(np.concatenate(all_quat_inv_masses), dtype=wp.float32, device=self.device)
        )

        if all_rest_lengths:
            self.rest_lengths_wp.assign(
                wp.array(np.concatenate(all_rest_lengths), dtype=wp.float32, device=self.device)
            )
            self.rest_darboux_wp.assign(
                wp.array(np.concatenate(all_rest_darboux), dtype=wp.vec3, device=self.device)
            )
            self.bend_stiffness_wp.assign(
                wp.array(np.concatenate(all_bend_stiffness), dtype=wp.vec3, device=self.device)
            )

        # Copy per-rod parameters
        gravity_np = np.array([
            [rod.gravity[0, 0], rod.gravity[0, 1], rod.gravity[0, 2]]
            for rod in rods
        ], dtype=np.float32)
        self.gravity_wp.assign(wp.array(gravity_np, dtype=wp.vec3, device=self.device))

        young_modulus_np = np.array([rod.young_modulus for rod in rods], dtype=np.float32)
        self.young_modulus_wp.assign(wp.array(young_modulus_np, dtype=wp.float32, device=self.device))

        torsion_modulus_np = np.array([rod.torsion_modulus for rod in rods], dtype=np.float32)
        self.torsion_modulus_wp.assign(wp.array(torsion_modulus_np, dtype=wp.float32, device=self.device))

        inv_inertia_local_np = np.array([
            rod.inv_inertia_local_diag for rod in rods
        ], dtype=np.float32)
        self.inv_inertia_local_diag_wp.assign(
            wp.array(inv_inertia_local_np, dtype=wp.vec3, device=self.device)
        )

    def sync_to_rods(self, rods: list) -> None:
        """Synchronize individual rod states from batched arrays.

        Copies data from concatenated batched arrays back to each rod's GPU arrays.
        Uses numpy as an intermediate format for reliable cross-array copying.

        Args:
            rods: List of WarpResidentRodState objects.
        """
        # Get all data from batched arrays
        all_positions = self.positions_wp.numpy()
        all_predicted_positions = self.predicted_positions_wp.numpy()
        all_velocities = self.velocities_wp.numpy()
        all_orientations = self.orientations_wp.numpy()
        all_predicted_orientations = self.predicted_orientations_wp.numpy()
        all_prev_orientations = self.prev_orientations_wp.numpy()
        all_angular_velocities = self.angular_velocities_wp.numpy()

        for i, rod in enumerate(rods):
            rod_start = self.batch.rod_offsets[i]
            rod_end = self.batch.rod_offsets[i + 1]

            rod.positions_wp.assign(
                wp.array(all_positions[rod_start:rod_end], dtype=wp.vec3, device=rod.device)
            )
            rod.predicted_positions_wp.assign(
                wp.array(all_predicted_positions[rod_start:rod_end], dtype=wp.vec3, device=rod.device)
            )
            rod.velocities_wp.assign(
                wp.array(all_velocities[rod_start:rod_end], dtype=wp.vec3, device=rod.device)
            )
            rod.orientations_wp.assign(
                wp.array(all_orientations[rod_start:rod_end], dtype=wp.quat, device=rod.device)
            )
            rod.predicted_orientations_wp.assign(
                wp.array(all_predicted_orientations[rod_start:rod_end], dtype=wp.quat, device=rod.device)
            )
            rod.prev_orientations_wp.assign(
                wp.array(all_prev_orientations[rod_start:rod_end], dtype=wp.quat, device=rod.device)
            )
            rod.angular_velocities_wp.assign(
                wp.array(all_angular_velocities[rod_start:rod_end], dtype=wp.vec3, device=rod.device)
            )


__all__ = ["BatchedGPUArrays", "RodBatch", "RodConfig", "RodState"]
