# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Warp GPU backend with CUDA graph support.

This backend provides maximum performance through GPU acceleration and
optional CUDA graph capture for reduced kernel launch overhead.
"""

from typing import TYPE_CHECKING, Optional

import numpy as np
import warp as wp

from .base import BackendBase

if TYPE_CHECKING:
    from ..model import CosseratRodModel


class WarpGPUBackend(BackendBase):
    """Warp backend running on GPU with CUDA graph support.

    This backend provides maximum performance through:
    - GPU-parallel kernel execution
    - Optional CUDA graph capture for reduced kernel launch overhead
    - Minimal CPU-GPU synchronization during simulation

    When use_cuda_graph is True, the first call to step() will capture
    the kernel sequence into a CUDA graph. Subsequent calls replay the
    graph with near-zero launch overhead.

    Note: CUDA graphs require that array sizes don't change between steps.
    If you need to resize arrays, disable CUDA graphs or recreate the backend.
    """

    def __init__(
        self,
        model: "CosseratRodModel",
        device: str = "cuda:0",
        use_cuda_graph: bool = False,
    ):
        """Initialize Warp GPU backend.

        Args:
            model: The rod model to operate on.
            device: CUDA device string (e.g., "cuda:0", "cuda:1").
            use_cuda_graph: Whether to capture CUDA graph on first step.
        """
        super().__init__(model)

        self.device = wp.get_device(device)
        self.use_cuda_graph = use_cuda_graph

        # CUDA graph state
        self._graph: Optional[wp.Graph] = None
        self._graph_captured = False

        # Cache frequently used values
        self._dt_cached = 0.0

        # Initialize Warp arrays
        self._init_arrays()

    @property
    def name(self) -> str:
        suffix = " + CUDA Graph" if self.use_cuda_graph else ""
        return f"Warp (GPU){suffix}"

    def _init_arrays(self):
        """Initialize Warp arrays on GPU."""
        from ..kernels import BAND_LDAB

        n = self.model.n_particles
        n_edges = self.model.n_edges
        n_dofs = 6 * n_edges

        # Position state
        self.positions_wp = wp.zeros(n, dtype=wp.vec3, device=self.device)
        self.predicted_positions_wp = wp.zeros(n, dtype=wp.vec3, device=self.device)
        self.velocities_wp = wp.zeros(n, dtype=wp.vec3, device=self.device)
        self.forces_wp = wp.zeros(n, dtype=wp.vec3, device=self.device)
        self.inv_masses_wp = wp.zeros(n, dtype=wp.float32, device=self.device)

        # Orientation state
        self.orientations_wp = wp.zeros(n, dtype=wp.quat, device=self.device)
        self.predicted_orientations_wp = wp.zeros(n, dtype=wp.quat, device=self.device)
        self.prev_orientations_wp = wp.zeros(n, dtype=wp.quat, device=self.device)
        self.angular_velocities_wp = wp.zeros(n, dtype=wp.vec3, device=self.device)
        self.torques_wp = wp.zeros(n, dtype=wp.vec3, device=self.device)
        self.quat_inv_masses_wp = wp.zeros(n, dtype=wp.float32, device=self.device)

        # Rod properties
        alloc_edges = max(1, n_edges)
        self.rest_lengths_wp = wp.zeros(alloc_edges, dtype=wp.float32, device=self.device)
        self.rest_darboux_wp = wp.zeros(alloc_edges, dtype=wp.vec3, device=self.device)
        self.bend_stiffness_wp = wp.zeros(alloc_edges, dtype=wp.vec3, device=self.device)

        # Constraint state
        alloc_dofs = max(1, n_dofs)
        self.n_dofs = n_dofs

        self.constraint_values_wp = wp.zeros(alloc_dofs, dtype=wp.float32, device=self.device)
        self.compliance_wp = wp.zeros(alloc_dofs, dtype=wp.float32, device=self.device)
        self.lambda_sum_wp = wp.zeros(alloc_dofs, dtype=wp.float32, device=self.device)

        # Jacobians
        self.jacobian_pos_wp = wp.zeros(alloc_edges * 36, dtype=wp.float32, device=self.device)
        self.jacobian_rot_wp = wp.zeros(alloc_edges * 36, dtype=wp.float32, device=self.device)

        # Banded matrix and solver arrays
        self.ab_wp = wp.zeros((BAND_LDAB, alloc_dofs), dtype=wp.float32, device=self.device)
        self.rhs_wp = wp.zeros(alloc_dofs, dtype=wp.float32, device=self.device)

        # Sync initial state
        self._sync_from_model()

    def _sync_from_model(self):
        """Copy model state to GPU arrays."""
        m = self.model

        # Convert positions/velocities from (n, 4) to vec3
        pos_3d = m.positions[:, :3].astype(np.float32)
        self.positions_wp.assign(wp.array(pos_3d, dtype=wp.vec3, device=self.device))

        pred_pos_3d = m.predicted_positions[:, :3].astype(np.float32)
        self.predicted_positions_wp.assign(wp.array(pred_pos_3d, dtype=wp.vec3, device=self.device))

        vel_3d = m.velocities[:, :3].astype(np.float32)
        self.velocities_wp.assign(wp.array(vel_3d, dtype=wp.vec3, device=self.device))

        forces_3d = m.forces[:, :3].astype(np.float32)
        self.forces_wp.assign(wp.array(forces_3d, dtype=wp.vec3, device=self.device))

        self.inv_masses_wp.assign(wp.array(m.inv_masses, dtype=wp.float32, device=self.device))

        # Orientations
        self.orientations_wp.assign(wp.array(m.orientations, dtype=wp.quat, device=self.device))
        self.predicted_orientations_wp.assign(wp.array(m.predicted_orientations, dtype=wp.quat, device=self.device))
        self.prev_orientations_wp.assign(wp.array(m.prev_orientations, dtype=wp.quat, device=self.device))

        ang_vel_3d = m.angular_velocities[:, :3].astype(np.float32)
        self.angular_velocities_wp.assign(wp.array(ang_vel_3d, dtype=wp.vec3, device=self.device))

        torques_3d = m.torques[:, :3].astype(np.float32)
        self.torques_wp.assign(wp.array(torques_3d, dtype=wp.vec3, device=self.device))

        self.quat_inv_masses_wp.assign(wp.array(m.quat_inv_masses, dtype=wp.float32, device=self.device))

        # Rod properties
        if m.n_edges > 0:
            self.rest_lengths_wp.assign(wp.array(m.rest_lengths, dtype=wp.float32, device=self.device))
            self.rest_darboux_wp.assign(wp.array(m.rest_darboux, dtype=wp.vec3, device=self.device))
            self.bend_stiffness_wp.assign(wp.array(m.bend_stiffness, dtype=wp.vec3, device=self.device))

    def _sync_to_model(self):
        """Copy GPU arrays back to model."""
        m = self.model

        # Positions
        pos_np = self.positions_wp.numpy()
        m.positions[:, :3] = pos_np

        pred_pos_np = self.predicted_positions_wp.numpy()
        m.predicted_positions[:, :3] = pred_pos_np

        vel_np = self.velocities_wp.numpy()
        m.velocities[:, :3] = vel_np

        # Orientations
        m.orientations[:] = self.orientations_wp.numpy()
        m.predicted_orientations[:] = self.predicted_orientations_wp.numpy()
        m.prev_orientations[:] = self.prev_orientations_wp.numpy()

        ang_vel_np = self.angular_velocities_wp.numpy()
        m.angular_velocities[:, :3] = ang_vel_np

    def step(self, dt: float):
        """Advance simulation by one timestep using Warp kernels on GPU."""
        # If using CUDA graph and already captured, replay it
        if self.use_cuda_graph and self._graph_captured:
            if dt != self._dt_cached:
                # dt changed, need to recapture
                self._graph_captured = False
                self._graph = None
            else:
                # Sync model to GPU (only needed for external changes)
                self._sync_from_model()

                # Replay graph
                wp.capture_launch(self._graph)

                # Sync back
                self._sync_to_model()
                self.model.clear_forces()
                return

        # Run the step (with potential graph capture)
        if self.use_cuda_graph and not self._graph_captured:
            self._dt_cached = dt
            self._sync_from_model()

            # Begin capture
            wp.capture_begin(device=self.device)
            try:
                self._run_step_kernels(dt)
            finally:
                self._graph = wp.capture_end(device=self.device)

            self._graph_captured = True

            # Now replay the captured graph for this step
            wp.capture_launch(self._graph)

            self._sync_to_model()
            self.model.clear_forces()
        else:
            # Regular execution without CUDA graph
            self._sync_from_model()
            self._run_step_kernels(dt)
            self._sync_to_model()
            self.model.clear_forces()

    def _run_step_kernels(self, dt: float):
        """Run the sequence of simulation kernels."""
        from ..kernels import (
            kernel_predict_positions,
            kernel_predict_rotations,
            kernel_integrate_positions,
            kernel_integrate_rotations,
            kernel_prepare_compliance,
            kernel_update_constraints,
            kernel_compute_jacobians,
            kernel_assemble_jmjt_banded,
            kernel_build_rhs,
            kernel_solve_banded_cholesky,
            kernel_apply_corrections,
            kernel_zero_array,
        )

        m = self.model
        n_edges = m.n_edges

        # 1. Predict positions
        gravity = wp.vec3(m.config.gravity[0], m.config.gravity[1], m.config.gravity[2])
        wp.launch(
            kernel_predict_positions,
            dim=m.n_particles,
            inputs=[
                self.positions_wp,
                self.velocities_wp,
                self.forces_wp,
                self.inv_masses_wp,
                gravity,
                float(dt),
                float(m.config.position_damping),
                self.predicted_positions_wp,
            ],
            device=self.device,
        )

        # 2. Predict rotations
        wp.launch(
            kernel_predict_rotations,
            dim=m.n_particles,
            inputs=[
                self.orientations_wp,
                self.angular_velocities_wp,
                self.torques_wp,
                self.quat_inv_masses_wp,
                float(dt),
                float(m.config.rotation_damping),
                self.predicted_orientations_wp,
            ],
            device=self.device,
        )

        if n_edges > 0:
            # 3. Zero lambda accumulator and prepare compliance
            wp.launch(
                kernel_zero_array,
                dim=self.n_dofs,
                inputs=[self.lambda_sum_wp],
                device=self.device,
            )

            wp.launch(
                kernel_prepare_compliance,
                dim=n_edges,
                inputs=[
                    self.rest_lengths_wp,
                    self.bend_stiffness_wp,
                    float(m.material.young_modulus),
                    float(m.material.torsion_modulus),
                    float(dt),
                    self.compliance_wp,
                ],
                device=self.device,
            )

            # 4. Update constraints
            wp.launch(
                kernel_update_constraints,
                dim=n_edges,
                inputs=[
                    self.predicted_positions_wp,
                    self.predicted_orientations_wp,
                    self.rest_lengths_wp,
                    self.rest_darboux_wp,
                    self.constraint_values_wp,
                ],
                device=self.device,
            )

            # 5. Compute Jacobians
            wp.launch(
                kernel_compute_jacobians,
                dim=n_edges,
                inputs=[
                    self.predicted_orientations_wp,
                    self.rest_lengths_wp,
                    self.jacobian_pos_wp,
                    self.jacobian_rot_wp,
                ],
                device=self.device,
            )

            # 6. Assemble banded matrix
            self.ab_wp.zero_()
            wp.launch(
                kernel_assemble_jmjt_banded,
                dim=n_edges,
                inputs=[
                    self.jacobian_pos_wp,
                    self.jacobian_rot_wp,
                    self.compliance_wp,
                    int(self.n_dofs),
                    self.ab_wp,
                ],
                device=self.device,
            )

            # 7. Build RHS and solve
            wp.launch(
                kernel_build_rhs,
                dim=self.n_dofs,
                inputs=[
                    self.constraint_values_wp,
                    self.compliance_wp,
                    self.lambda_sum_wp,
                    int(self.n_dofs),
                    self.rhs_wp,
                ],
                device=self.device,
            )

            wp.launch(
                kernel_solve_banded_cholesky,
                dim=1,
                inputs=[int(self.n_dofs), self.ab_wp, self.rhs_wp],
                device=self.device,
            )

            # 8. Apply corrections
            wp.launch(
                kernel_apply_corrections,
                dim=1,
                inputs=[
                    self.predicted_positions_wp,
                    self.predicted_orientations_wp,
                    self.inv_masses_wp,
                    self.quat_inv_masses_wp,
                    self.jacobian_pos_wp,
                    self.jacobian_rot_wp,
                    self.rhs_wp,
                    self.lambda_sum_wp,
                    int(n_edges),
                ],
                device=self.device,
            )

        # 9. Integrate positions
        wp.launch(
            kernel_integrate_positions,
            dim=m.n_particles,
            inputs=[
                self.positions_wp,
                self.predicted_positions_wp,
                self.velocities_wp,
                self.inv_masses_wp,
                float(dt),
            ],
            device=self.device,
        )

        # 10. Integrate rotations
        wp.launch(
            kernel_integrate_rotations,
            dim=m.n_particles,
            inputs=[
                self.orientations_wp,
                self.predicted_orientations_wp,
                self.prev_orientations_wp,
                self.angular_velocities_wp,
                self.quat_inv_masses_wp,
                float(dt),
            ],
            device=self.device,
        )

        # Clear forces on GPU
        self.forces_wp.zero_()
        self.torques_wp.zero_()

    # =========================================================================
    # State change notifications
    # =========================================================================

    def on_gravity_changed(self):
        # Gravity is passed as a kernel argument, so nothing to cache
        pass

    def on_rest_shape_changed(self):
        if self.model.n_edges > 0:
            self.rest_darboux_wp.assign(
                wp.array(self.model.rest_darboux, dtype=wp.vec3, device=self.device)
            )
        # Invalidate CUDA graph since material changed
        self._graph_captured = False

    def on_stiffness_changed(self):
        if self.model.n_edges > 0:
            self.bend_stiffness_wp.assign(
                wp.array(self.model.bend_stiffness, dtype=wp.vec3, device=self.device)
            )
        # Invalidate CUDA graph since material changed
        self._graph_captured = False

    def on_material_changed(self):
        # Material properties are passed as kernel arguments
        # Invalidate CUDA graph since material changed
        self._graph_captured = False

    def on_reset(self):
        self._sync_from_model()
        # Keep CUDA graph valid since structure didn't change
