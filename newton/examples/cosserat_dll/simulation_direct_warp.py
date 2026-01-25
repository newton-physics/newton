# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""GPU-parallel direct Cosserat rod solver using Warp.

This module provides a Warp-based implementation of the direct Cosserat rod solver
that can run entirely on GPU. It mirrors the structure of simulation_direct_numpy.py
but uses Warp kernels for parallel computation.

For verification, the solver supports a "verification mode" where numpy data is copied
to warp arrays before each step and results are copied back for comparison.
"""

import numpy as np
import warp as wp

from .rod_state import RodState

# Import existing warp kernels from cosserat_codex module
from newton.examples.cosserat_codex import warp_cosserat_codex as warp_base


# =============================================================================
# Warp kernel helper functions (reuse from cosserat_codex)
# =============================================================================

# Quaternion helpers
_quat_mul = warp_base._warp_quat_mul
_quat_conjugate = warp_base._warp_quat_conjugate
_quat_normalize = warp_base._warp_quat_normalize
_quat_rotate_vector = warp_base._warp_quat_rotate_vector

# Jacobian indexing
_jacobian_index = warp_base._warp_jacobian_index

# Prediction and integration kernels
_warp_predict_positions = warp_base._warp_predict_positions
_warp_integrate_positions = warp_base._warp_integrate_positions
_warp_predict_rotations = warp_base._warp_predict_rotations
_warp_integrate_rotations = warp_base._warp_integrate_rotations

# Constraint kernels
_warp_prepare_compliance = warp_base._warp_prepare_compliance
_warp_update_constraints_direct = warp_base._warp_update_constraints_direct
_warp_compute_jacobians_direct = warp_base._warp_compute_jacobians_direct

# Assembly and solver kernels
_warp_assemble_jmjt_blocks = warp_base._warp_assemble_jmjt_blocks
_warp_build_rhs = warp_base._warp_build_rhs
_warp_block_thomas_solve = warp_base._warp_block_thomas_solve

# Banded Cholesky constants and kernel
BAND_KD = warp_base.BAND_KD
BAND_LDAB = warp_base.BAND_LDAB
_warp_spbsv_u11_1rhs = warp_base._warp_spbsv_u11_1rhs
_warp_assemble_jmjt_banded = warp_base._warp_assemble_jmjt_banded


# =============================================================================
# Additional kernels for correction application
# =============================================================================


@wp.func
def _quat_correction_g(q: wp.quat, dtheta: wp.vec3) -> wp.quat:
    """Apply rotation correction using G matrix."""
    norm_sq = dtheta.x * dtheta.x + dtheta.y * dtheta.y + dtheta.z * dtheta.z
    if norm_sq < 1.0e-20:
        return q

    # G matrix maps angular correction to quaternion correction
    corr_x = 0.5 * (q.w * dtheta.x + q.z * dtheta.y - q.y * dtheta.z)
    corr_y = 0.5 * (-q.z * dtheta.x + q.w * dtheta.y + q.x * dtheta.z)
    corr_z = 0.5 * (q.y * dtheta.x - q.x * dtheta.y + q.w * dtheta.z)
    corr_w = 0.5 * (-q.x * dtheta.x - q.y * dtheta.y - q.z * dtheta.z)

    q_new = wp.quat(q.x + corr_x, q.y + corr_y, q.z + corr_z, q.w + corr_w)
    return _quat_normalize(q_new)


@wp.func
def _jacobian_dot(
    jacobian: wp.array(dtype=wp.float32),
    edge: int,
    col: int,
    dl0: float,
    dl1: float,
    dl2: float,
    dl3: float,
    dl4: float,
    dl5: float,
) -> float:
    """Compute J^T * delta_lambda for a single column."""
    return (
        jacobian[_jacobian_index(edge, 0, col)] * dl0
        + jacobian[_jacobian_index(edge, 1, col)] * dl1
        + jacobian[_jacobian_index(edge, 2, col)] * dl2
        + jacobian[_jacobian_index(edge, 3, col)] * dl3
        + jacobian[_jacobian_index(edge, 4, col)] * dl4
        + jacobian[_jacobian_index(edge, 5, col)] * dl5
    )


@wp.kernel
def _warp_zero_lambdas(lambda_sum: wp.array(dtype=wp.float32)):
    """Zero out Lagrange multiplier accumulator."""
    i = wp.tid()
    lambda_sum[i] = 0.0


@wp.kernel
def _warp_apply_corrections(
    predicted_positions: wp.array(dtype=wp.vec3),
    predicted_orientations: wp.array(dtype=wp.quat),
    inv_masses: wp.array(dtype=wp.float32),
    quat_inv_masses: wp.array(dtype=wp.float32),
    jacobian_pos: wp.array(dtype=wp.float32),
    jacobian_rot: wp.array(dtype=wp.float32),
    delta_lambda: wp.array(dtype=wp.float32),
    lambda_sum: wp.array(dtype=wp.float32),
    n_edges: int,
):
    """Apply corrections from constraint solve to positions and orientations."""
    tid = wp.tid()
    if tid != 0:
        return

    for edge in range(n_edges):
        base_idx = edge * 6
        dl0 = delta_lambda[base_idx + 0]
        dl1 = delta_lambda[base_idx + 1]
        dl2 = delta_lambda[base_idx + 2]
        dl3 = delta_lambda[base_idx + 3]
        dl4 = delta_lambda[base_idx + 4]
        dl5 = delta_lambda[base_idx + 5]

        # Accumulate lambda
        lambda_sum[base_idx + 0] = lambda_sum[base_idx + 0] + dl0
        lambda_sum[base_idx + 1] = lambda_sum[base_idx + 1] + dl1
        lambda_sum[base_idx + 2] = lambda_sum[base_idx + 2] + dl2
        lambda_sum[base_idx + 3] = lambda_sum[base_idx + 3] + dl3
        lambda_sum[base_idx + 4] = lambda_sum[base_idx + 4] + dl4
        lambda_sum[base_idx + 5] = lambda_sum[base_idx + 5] + dl5

        # Apply position corrections
        inv_m0 = inv_masses[edge]
        inv_m1 = inv_masses[edge + 1]

        if inv_m0 > 0.0:
            dp0_x = _jacobian_dot(jacobian_pos, edge, 0, dl0, dl1, dl2, dl3, dl4, dl5)
            dp0_y = _jacobian_dot(jacobian_pos, edge, 1, dl0, dl1, dl2, dl3, dl4, dl5)
            dp0_z = _jacobian_dot(jacobian_pos, edge, 2, dl0, dl1, dl2, dl3, dl4, dl5)
            dp0 = wp.vec3(dp0_x * inv_m0, dp0_y * inv_m0, dp0_z * inv_m0)
            predicted_positions[edge] = predicted_positions[edge] + dp0

        if inv_m1 > 0.0:
            dp1_x = _jacobian_dot(jacobian_pos, edge, 3, dl0, dl1, dl2, dl3, dl4, dl5)
            dp1_y = _jacobian_dot(jacobian_pos, edge, 4, dl0, dl1, dl2, dl3, dl4, dl5)
            dp1_z = _jacobian_dot(jacobian_pos, edge, 5, dl0, dl1, dl2, dl3, dl4, dl5)
            dp1 = wp.vec3(dp1_x * inv_m1, dp1_y * inv_m1, dp1_z * inv_m1)
            predicted_positions[edge + 1] = predicted_positions[edge + 1] + dp1

        # Apply rotation corrections
        if quat_inv_masses[edge] > 0.0:
            dtheta0 = wp.vec3(
                _jacobian_dot(jacobian_rot, edge, 0, dl0, dl1, dl2, dl3, dl4, dl5),
                _jacobian_dot(jacobian_rot, edge, 1, dl0, dl1, dl2, dl3, dl4, dl5),
                _jacobian_dot(jacobian_rot, edge, 2, dl0, dl1, dl2, dl3, dl4, dl5),
            )
            predicted_orientations[edge] = _quat_correction_g(predicted_orientations[edge], dtheta0)

        if quat_inv_masses[edge + 1] > 0.0:
            dtheta1 = wp.vec3(
                _jacobian_dot(jacobian_rot, edge, 3, dl0, dl1, dl2, dl3, dl4, dl5),
                _jacobian_dot(jacobian_rot, edge, 4, dl0, dl1, dl2, dl3, dl4, dl5),
                _jacobian_dot(jacobian_rot, edge, 5, dl0, dl1, dl2, dl3, dl4, dl5),
            )
            predicted_orientations[edge + 1] = _quat_correction_g(predicted_orientations[edge + 1], dtheta1)


# =============================================================================
# Main simulation class
# =============================================================================


class DirectCosseratRodSimulationWarp:
    """GPU-parallel direct Cosserat rod solver using Warp.

    This class provides a Warp-based implementation that mirrors the NumPy solver.
    In verification mode, it copies data between numpy and warp arrays to enable
    step-by-step comparison with the numpy reference implementation.
    """

    # Solver backends
    BACKEND_BLOCK_THOMAS = "block_thomas"
    BACKEND_BANDED_CHOLESKY = "banded_cholesky"

    def __init__(self, state: RodState, device: str = "cuda:0"):
        """Initialize Warp solver.

        Args:
            state: Rod state to simulate (numpy arrays).
            device: Warp device string (e.g., "cuda:0", "cpu").
        """
        self.state = state
        self.device = wp.get_device(device)

        # Simulation parameters
        self.position_damping = 0.001
        self.rotation_damping = 0.001
        self.gravity = np.array([0.0, 0.0, -9.81], dtype=np.float32)

        # Material parameters
        self.young_modulus = 1.0e6
        self.torsion_modulus = 1.0e6

        # Bend stiffness coefficients (per edge)
        self.bend_stiffness = np.ones((state.n_edges, 3), dtype=np.float32)
        self.rest_darboux = np.zeros((state.n_edges, 3), dtype=np.float32)

        # Solver configuration
        self.solver_backend = self.BACKEND_BLOCK_THOMAS

        # Verification mode
        self.verification_mode = True

        # Initialize warp arrays
        self._init_warp_arrays()

    def _init_warp_arrays(self):
        """Initialize GPU arrays mirroring numpy state."""
        n = self.state.n_particles
        n_edges = self.state.n_edges

        # Position state (vec3)
        self.positions_wp = wp.zeros(n, dtype=wp.vec3, device=self.device)
        self.predicted_positions_wp = wp.zeros(n, dtype=wp.vec3, device=self.device)
        self.velocities_wp = wp.zeros(n, dtype=wp.vec3, device=self.device)
        self.forces_wp = wp.zeros(n, dtype=wp.vec3, device=self.device)
        self.inv_masses_wp = wp.zeros(n, dtype=wp.float32, device=self.device)

        # Orientation state (quat)
        self.orientations_wp = wp.zeros(n, dtype=wp.quat, device=self.device)
        self.predicted_orientations_wp = wp.zeros(n, dtype=wp.quat, device=self.device)
        self.prev_orientations_wp = wp.zeros(n, dtype=wp.quat, device=self.device)
        self.angular_velocities_wp = wp.zeros(n, dtype=wp.vec3, device=self.device)
        self.torques_wp = wp.zeros(n, dtype=wp.vec3, device=self.device)
        self.quat_inv_masses_wp = wp.zeros(n, dtype=wp.float32, device=self.device)

        # Rod properties
        self.rest_lengths_wp = wp.zeros(max(1, n_edges), dtype=wp.float32, device=self.device)
        self.rest_darboux_wp = wp.zeros(max(1, n_edges), dtype=wp.vec3, device=self.device)
        self.bend_stiffness_wp = wp.zeros(max(1, n_edges), dtype=wp.vec3, device=self.device)

        # Constraint/solver state
        n_dofs = 6 * n_edges
        self.n_dofs = n_dofs
        alloc_dofs = max(1, n_dofs)
        alloc_edges = max(1, n_edges)

        self.constraint_values_wp = wp.zeros(alloc_dofs, dtype=wp.float32, device=self.device)
        self.compliance_wp = wp.zeros(alloc_dofs, dtype=wp.float32, device=self.device)
        self.lambda_sum_wp = wp.zeros(alloc_dofs, dtype=wp.float32, device=self.device)

        # Jacobian storage: [n_edges][6 rows][6 cols]
        self.jacobian_pos_wp = wp.zeros(alloc_edges * 36, dtype=wp.float32, device=self.device)
        self.jacobian_rot_wp = wp.zeros(alloc_edges * 36, dtype=wp.float32, device=self.device)

        # Block Thomas solver storage
        self.diag_blocks_wp = wp.zeros(alloc_edges * 36, dtype=wp.float32, device=self.device)
        self.offdiag_blocks_wp = wp.zeros(alloc_edges * 36, dtype=wp.float32, device=self.device)
        self.c_blocks_wp = wp.zeros(alloc_edges * 36, dtype=wp.float32, device=self.device)
        self.d_prime_wp = wp.zeros(alloc_dofs, dtype=wp.float32, device=self.device)
        self.rhs_wp = wp.zeros(alloc_dofs, dtype=wp.float32, device=self.device)
        self.delta_lambda_wp = wp.zeros(alloc_dofs, dtype=wp.float32, device=self.device)

        # Banded Cholesky storage
        self.ab_wp = wp.zeros((BAND_LDAB, alloc_dofs), dtype=wp.float32, device=self.device)

    def sync_numpy_to_warp(self):
        """Copy numpy state to warp arrays."""
        s = self.state

        # Convert positions from (n, 4) to vec3
        self.positions_wp.assign(wp.array(s.positions[:, :3].astype(np.float32), dtype=wp.vec3, device=self.device))
        self.predicted_positions_wp.assign(
            wp.array(s.predicted_positions[:, :3].astype(np.float32), dtype=wp.vec3, device=self.device)
        )
        self.velocities_wp.assign(wp.array(s.velocities[:, :3].astype(np.float32), dtype=wp.vec3, device=self.device))
        self.forces_wp.assign(wp.array(s.forces[:, :3].astype(np.float32), dtype=wp.vec3, device=self.device))
        self.inv_masses_wp.assign(wp.array(s.inv_masses, dtype=wp.float32, device=self.device))

        # Convert orientations (n, 4) to quat
        self.orientations_wp.assign(wp.array(s.orientations, dtype=wp.quat, device=self.device))
        self.predicted_orientations_wp.assign(wp.array(s.predicted_orientations, dtype=wp.quat, device=self.device))
        self.prev_orientations_wp.assign(wp.array(s.prev_orientations, dtype=wp.quat, device=self.device))
        self.angular_velocities_wp.assign(
            wp.array(s.angular_velocities[:, :3].astype(np.float32), dtype=wp.vec3, device=self.device)
        )
        self.torques_wp.assign(wp.array(s.torques[:, :3].astype(np.float32), dtype=wp.vec3, device=self.device))
        self.quat_inv_masses_wp.assign(wp.array(s.quat_inv_masses, dtype=wp.float32, device=self.device))

        # Rod properties
        if s.n_edges > 0:
            self.rest_lengths_wp.assign(wp.array(s.rest_lengths, dtype=wp.float32, device=self.device))
            self.rest_darboux_wp.assign(wp.array(self.rest_darboux, dtype=wp.vec3, device=self.device))
            self.bend_stiffness_wp.assign(wp.array(self.bend_stiffness, dtype=wp.vec3, device=self.device))

    def sync_warp_to_numpy(self):
        """Copy warp arrays back to numpy state."""
        s = self.state

        # Positions
        pos_np = self.positions_wp.numpy()
        s.positions[:, :3] = pos_np

        pred_pos_np = self.predicted_positions_wp.numpy()
        s.predicted_positions[:, :3] = pred_pos_np

        vel_np = self.velocities_wp.numpy()
        s.velocities[:, :3] = vel_np

        # Orientations
        s.orientations[:] = self.orientations_wp.numpy()
        s.predicted_orientations[:] = self.predicted_orientations_wp.numpy()
        s.prev_orientations[:] = self.prev_orientations_wp.numpy()

        ang_vel_np = self.angular_velocities_wp.numpy()
        s.angular_velocities[:, :3] = ang_vel_np

    def step(self, dt: float):
        """Advance simulation by one timestep.

        Args:
            dt: Time step size in seconds.
        """
        s = self.state
        n_edges = s.n_edges

        if self.verification_mode:
            self.sync_numpy_to_warp()

        # 1. Predict positions
        gravity = wp.vec3(self.gravity[0], self.gravity[1], self.gravity[2])
        wp.launch(
            _warp_predict_positions,
            dim=s.n_particles,
            inputs=[
                self.positions_wp,
                self.velocities_wp,
                self.forces_wp,
                self.inv_masses_wp,
                gravity,
                float(dt),
                float(self.position_damping),
                self.predicted_positions_wp,
            ],
            device=self.device,
        )

        # 2. Predict rotations
        wp.launch(
            _warp_predict_rotations,
            dim=s.n_particles,
            inputs=[
                self.orientations_wp,
                self.angular_velocities_wp,
                self.torques_wp,
                self.quat_inv_masses_wp,
                float(dt),
                float(self.rotation_damping),
                self.predicted_orientations_wp,
            ],
            device=self.device,
        )

        # 3. Prepare constraints (compliance + reset lambdas)
        if n_edges > 0:
            wp.launch(
                _warp_zero_lambdas,
                dim=self.n_dofs,
                inputs=[self.lambda_sum_wp],
                device=self.device,
            )
            wp.launch(
                _warp_prepare_compliance,
                dim=n_edges,
                inputs=[
                    self.rest_lengths_wp,
                    self.bend_stiffness_wp,
                    float(self.young_modulus),
                    float(self.torsion_modulus),
                    float(dt),
                    self.compliance_wp,
                ],
                device=self.device,
            )

            # 4. Update constraints (compute violations)
            wp.launch(
                _warp_update_constraints_direct,
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
                _warp_compute_jacobians_direct,
                dim=n_edges,
                inputs=[
                    self.predicted_orientations_wp,
                    self.rest_lengths_wp,
                    self.jacobian_pos_wp,
                    self.jacobian_rot_wp,
                ],
                device=self.device,
            )

            # 6. Assemble and solve
            if self.solver_backend == self.BACKEND_BANDED_CHOLESKY:
                # Banded Cholesky solver
                self.ab_wp.zero_()
                wp.launch(
                    _warp_assemble_jmjt_banded,
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
                wp.launch(
                    _warp_build_rhs,
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
                    _warp_spbsv_u11_1rhs,
                    dim=1,
                    inputs=[int(self.n_dofs), self.ab_wp, self.rhs_wp],
                    device=self.device,
                )
                delta_lambda = self.rhs_wp
            else:
                # Block Thomas solver (default)
                wp.launch(
                    _warp_assemble_jmjt_blocks,
                    dim=n_edges,
                    inputs=[
                        self.jacobian_pos_wp,
                        self.jacobian_rot_wp,
                        self.compliance_wp,
                        int(n_edges),
                        self.diag_blocks_wp,
                        self.offdiag_blocks_wp,
                    ],
                    device=self.device,
                )
                wp.launch(
                    _warp_build_rhs,
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
                    _warp_block_thomas_solve,
                    dim=1,
                    inputs=[
                        self.diag_blocks_wp,
                        self.offdiag_blocks_wp,
                        self.rhs_wp,
                        int(n_edges),
                        self.c_blocks_wp,
                        self.d_prime_wp,
                        self.delta_lambda_wp,
                    ],
                    device=self.device,
                )
                delta_lambda = self.delta_lambda_wp

            # 7. Apply corrections
            wp.launch(
                _warp_apply_corrections,
                dim=1,
                inputs=[
                    self.predicted_positions_wp,
                    self.predicted_orientations_wp,
                    self.inv_masses_wp,
                    self.quat_inv_masses_wp,
                    self.jacobian_pos_wp,
                    self.jacobian_rot_wp,
                    delta_lambda,
                    self.lambda_sum_wp,
                    int(n_edges),
                ],
                device=self.device,
            )

        # 8. Integrate positions
        wp.launch(
            _warp_integrate_positions,
            dim=s.n_particles,
            inputs=[
                self.positions_wp,
                self.predicted_positions_wp,
                self.velocities_wp,
                self.inv_masses_wp,
                float(dt),
            ],
            device=self.device,
        )

        # 9. Integrate rotations
        wp.launch(
            _warp_integrate_rotations,
            dim=s.n_particles,
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

        # 10. Sync back and clear forces
        if self.verification_mode:
            self.sync_warp_to_numpy()
            s.clear_forces()
            # Also clear warp forces
            self.forces_wp.zero_()
            self.torques_wp.zero_()

    def set_gravity(self, gx: float, gy: float, gz: float):
        """Set gravity vector."""
        self.gravity[0] = gx
        self.gravity[1] = gy
        self.gravity[2] = gz

    def set_rest_curvature(self, kappa1: float, kappa2: float, tau: float):
        """Set uniform rest curvature for the entire rod."""
        self.rest_darboux[:, 0] = kappa1
        self.rest_darboux[:, 1] = kappa2
        self.rest_darboux[:, 2] = tau
        if self.state.n_edges > 0 and self.verification_mode:
            self.rest_darboux_wp.assign(wp.array(self.rest_darboux, dtype=wp.vec3, device=self.device))

    def set_bend_stiffness(self, k1: float, k2: float, k_tau: float):
        """Set uniform bending/twist stiffness for the entire rod."""
        self.bend_stiffness[:, 0] = k1
        self.bend_stiffness[:, 1] = k2
        self.bend_stiffness[:, 2] = k_tau
        if self.state.n_edges > 0 and self.verification_mode:
            self.bend_stiffness_wp.assign(wp.array(self.bend_stiffness, dtype=wp.vec3, device=self.device))

    def get_positions_3d(self) -> np.ndarray:
        """Get positions as (n, 3) array."""
        if self.verification_mode:
            return self.state.get_positions_3d()
        return self.positions_wp.numpy()
