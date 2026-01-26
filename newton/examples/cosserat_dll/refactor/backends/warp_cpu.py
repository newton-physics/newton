# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Warp CPU backend for debugging.

This backend runs Warp kernels on CPU, making it easier to debug kernel code
without GPU-specific issues. Uses numpy-backed Warp arrays.
"""

from typing import TYPE_CHECKING

import numpy as np
import warp as wp

from .base import BackendBase

if TYPE_CHECKING:
    from ..model import CosseratRodModel


class WarpCPUBackend(BackendBase):
    """Warp backend running on CPU for debugging.

    This backend uses the same Warp kernels as the GPU backend but runs them
    on CPU. This makes debugging easier since:
    - No GPU memory transfers to debug
    - Better error messages
    - Can use Python debugging tools more easily

    The implementation mirrors WarpGPUBackend but forces CPU device.
    """

    def __init__(self, model: "CosseratRodModel"):
        """Initialize Warp CPU backend.

        Args:
            model: The rod model to operate on.
        """
        super().__init__(model)

        # Force CPU device
        self.device = wp.get_device("cpu")

        # Initialize Warp arrays
        self._init_arrays()

    @property
    def name(self) -> str:
        return "Warp (CPU)"

    def _init_arrays(self):
        """Initialize Warp arrays on CPU."""
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
        """Copy model state to Warp arrays."""
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
        """Copy Warp arrays back to model."""
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
        """Advance simulation by one timestep using Warp kernels on CPU."""
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

        # Sync model to Warp arrays
        self._sync_from_model()

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
                    self.rhs_wp,  # delta_lambda is stored in rhs after solve
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

        # Sync back to model and clear forces
        self._sync_to_model()
        m.clear_forces()

        # Also clear Warp force arrays
        self.forces_wp.zero_()
        self.torques_wp.zero_()

    # =========================================================================
    # State change notifications
    # =========================================================================

    def on_rest_shape_changed(self):
        if self.model.n_edges > 0:
            self.rest_darboux_wp.assign(
                wp.array(self.model.rest_darboux, dtype=wp.vec3, device=self.device)
            )

    def on_stiffness_changed(self):
        if self.model.n_edges > 0:
            self.bend_stiffness_wp.assign(
                wp.array(self.model.bend_stiffness, dtype=wp.vec3, device=self.device)
            )

    def on_reset(self):
        self._sync_from_model()
