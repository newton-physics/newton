# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Direct Cosserat rod simulation with NumPy implementations.

This is a copy of simulation_direct.py that allows gradual replacement
of DLL calls with pure NumPy implementations for testing and comparison.
"""

import numpy as np

from .defkit_wrapper import DefKitWrapper
from .rod_state import RodState


class DirectCosseratRodSimulationNumPy:
    """Direct Cosserat rod simulation with NumPy implementations.

    This class mirrors DirectCosseratRodSimulation but allows individual
    methods to be switched between DLL and NumPy implementations.

    Flags to control which implementation is used:
    - use_numpy_predict_positions: Use NumPy for position prediction
    - use_numpy_predict_rotations: Use NumPy for rotation prediction
    - use_numpy_integrate_positions: Use NumPy for position integration
    - use_numpy_integrate_rotations: Use NumPy for rotation integration
    - use_numpy_prepare: Use NumPy for constraint preparation
    - (more flags to be added as methods are ported)
    """

    def __init__(self, state: RodState, dll_path: str = "unity_ref"):
        """Initialize direct solver simulation.

        Args:
            state: Rod state to simulate.
            dll_path: Path to directory containing DefKit DLLs.
        """
        self.dll = DefKitWrapper(dll_path)
        self.state = state

        # Simulation parameters
        self.position_damping = 0.001
        self.rotation_damping = 0.001
        self.gravity = np.array([0.0, 0.0, -9.81, 0.0], dtype=np.float32)

        # Material parameters
        self.radius = 0.01  # Rod cross-section radius
        self.young_modulus = 1.0  # Base Young's modulus
        self.torsion_modulus = 1.0  # Base torsion modulus
        self.young_modulus_mult = 1.0e6  # Effective Young's modulus (Pa)
        self.torsion_modulus_mult = 1.0e6  # Effective torsion modulus (Pa)

        # Direct solver uses Vector3 for rest Darboux (not quaternion)
        self.rest_darboux_vec = np.zeros((state.n_edges, 4), dtype=np.float32)

        # Bend stiffness coefficients (kappa1, kappa2, tau stiffness)
        self.bend_stiffness = np.ones((state.n_edges, 4), dtype=np.float32)

        # Stiffness multipliers (for NumPy solver tuning)
        # With proper K matrix coupling (J * J^T includes [r]_× * [r]_×^T), the
        # position-rotation coupling is now handled correctly, so no empirical factor needed.
        self.stretch_stiffness_mult = 1.0
        self.shear_stiffness_mult = 1.0
        self.bend_stiffness_mult = 1.0  # No empirical factor needed with proper K matrix

        # Initialize the direct solver (needed for DLL fallback)
        self._rod_ptr = self.dll.init_direct_elastic_rod(
            state.positions,
            state.orientations,
            self.radius,
            state.rest_lengths,
            self.young_modulus,
            self.torsion_modulus,
        )

        # === NumPy solver internal state ===
        # Each constraint has 6 DOF: 3 stretch-shear + 3 bend-twist
        n_edges = state.n_edges

        # Lagrange multipliers (accumulated constraint forces)
        # Shape: (n_edges, 6) - [stretch, shear1, shear2, bend1, bend2, twist]
        self.lambdas = np.zeros((n_edges, 6), dtype=np.float32)

        # Compliance values (inverse stiffness scaled by dt^2)
        # Shape: (n_edges, 6)
        self.compliance = np.zeros((n_edges, 6), dtype=np.float32)

        # Constraint values/errors
        # Shape: (n_edges, 6)
        self.constraint_values = np.zeros((n_edges, 6), dtype=np.float32)

        # Current rest lengths and Darboux vectors (copied during prepare)
        self.current_rest_lengths = state.rest_lengths.copy()
        self.current_rest_darboux = np.zeros((n_edges, 3), dtype=np.float32)

        # Note: Cross-section properties (E*I) are NOT used by the direct solver.
        # The C++ PrepareDirectElasticRodConstraints uses bendStiff * modulus directly.
        # The _update_cross_section_properties method is kept for potential future use
        # but doesn't affect the current compliance calculation.
        self._update_cross_section_properties()

        # === Implementation flags ===
        # Set to True to use NumPy implementation, False to use DLL
        self.use_numpy_predict_positions = False
        self.use_numpy_predict_rotations = False
        self.use_numpy_integrate_positions = False
        self.use_numpy_integrate_rotations = False
        self.use_numpy_prepare = False
        self.use_numpy_update = False
        self.use_numpy_jacobians = False
        self.use_numpy_assemble = False
        self.use_numpy_solve = False
        # Use C++-equivalent banded Cholesky solve (spbsv_u11_1rhs)
        self.use_numpy_solve_spbsv = False

        # Non-banded solver flag (replaces update+jacobians+assemble+solve)
        self.use_numpy_project_direct = False

    def _update_cross_section_properties(self):
        """Update cross-section properties based on current radius."""
        r = self.radius
        self.cross_section_area = np.pi * r * r
        self.second_moment_area = np.pi * r**4 / 4.0  # I for bending
        self.polar_moment = np.pi * r**4 / 2.0  # J for torsion

    def __del__(self):
        """Clean up native resources."""
        if hasattr(self, "_rod_ptr") and self._rod_ptr is not None:
            self.dll.destroy_direct_elastic_rod(self._rod_ptr)
            self._rod_ptr = None

    # =========================================================================
    # NumPy implementations (to be filled in one by one)
    # =========================================================================

    def _predict_positions_numpy(self, dt: float):
        """NumPy implementation of position prediction."""
        s = self.state
        damping = self.position_damping
        gravity = self.gravity

        for i in range(s.n_particles):
            if s.inv_masses[i] > 0:
                # Apply damping to velocity
                s.velocities[i, :3] *= (1.0 - damping)
                # Add acceleration from gravity and forces
                accel = s.inv_masses[i] * (gravity[:3] + s.forces[i, :3])
                s.velocities[i, :3] += dt * accel
                # Predict position
                s.predicted_positions[i, :3] = s.positions[i, :3] + dt * s.velocities[i, :3]
            else:
                # Fixed particle
                s.predicted_positions[i, :3] = s.positions[i, :3]

    def _predict_rotations_numpy(self, dt: float):
        """NumPy implementation of rotation prediction.

        Following C++ PredictRotationsPBD:
        1. Apply damping to angular velocity
        2. Add angular acceleration from torques
        3. Integrate quaternion using: q' = q + 0.5 * dt * omega_quat * q
           where omega_quat = (omega.x, omega.y, omega.z, 0)
        4. Normalize the resulting quaternion
        """
        s = self.state
        damping = self.rotation_damping

        for i in range(s.n_particles):
            if s.quat_inv_masses[i] > 0:
                # Apply damping to angular velocity
                s.angular_velocities[i, :3] *= (1.0 - damping)

                # Add angular acceleration from torques
                # For unit inertia: alpha = inv_inertia * torque = torque
                accel = s.torques[i, :3]
                s.angular_velocities[i, :3] += dt * accel

                # Get current orientation and angular velocity
                q = s.orientations[i].copy()  # (x, y, z, w)
                omega = s.angular_velocities[i, :3]

                # Integrate quaternion: q' = q + 0.5 * dt * omega_quat * q
                # omega_quat = (omega.x, omega.y, omega.z, 0)
                # Using quaternion multiplication: omega_quat * q
                wx, wy, wz = omega
                qx, qy, qz, qw = q

                # omega_quat * q where omega_quat = (wx, wy, wz, 0)
                dq = np.array([
                    0.0 * qw + wx * qw + wy * qz - wz * qy,  # = wx*qw + wy*qz - wz*qy
                    0.0 * qw - wx * qz + wy * qw + wz * qx,  # = -wx*qz + wy*qw + wz*qx
                    0.0 * qw + wx * qy - wy * qx + wz * qw,  # = wx*qy - wy*qx + wz*qw
                    0.0 * qw - wx * qx - wy * qy - wz * qz,  # = -wx*qx - wy*qy - wz*qz
                ], dtype=np.float32)

                # q_new = q + 0.5 * dt * dq
                q_new = q + 0.5 * dt * dq

                # Normalize
                norm = np.linalg.norm(q_new)
                if norm > 1e-10:
                    q_new /= norm

                s.predicted_orientations[i] = q_new
            else:
                # Fixed orientation
                s.predicted_orientations[i] = s.orientations[i].copy()

    def _integrate_positions_numpy(self, dt: float):
        """NumPy implementation of position integration.

        Following C++ Integrate_native:
        1. For dynamic particles (inv_mass > 0):
           - velocity = (predicted_position - position) / dt
           - position = predicted_position
        2. For fixed particles (inv_mass == 0):
           - No velocity update
           - position = predicted_position (in case it was moved externally)
        """
        s = self.state
        inv_dt = 1.0 / dt if dt > 1e-10 else 0.0

        for i in range(s.n_particles):
            if s.inv_masses[i] > 0:
                # Update velocity from position change
                s.velocities[i, :3] = (s.predicted_positions[i, :3] - s.positions[i, :3]) * inv_dt
                # Update position
                s.positions[i, :3] = s.predicted_positions[i, :3]
            else:
                # Fixed particle: position might have been moved externally
                s.positions[i, :3] = s.predicted_positions[i, :3]
                # Keep velocity at zero
                s.velocities[i, :3] = 0.0

    def _integrate_rotations_numpy(self, dt: float):
        """NumPy implementation of rotation integration.

        Following C++ IntegrateRotationsPBD:
        1. Store current orientation as prev_orientation
        2. Update orientation from predicted_orientation
        3. Derive angular velocity from orientation change:
           omega = 2/dt * (q_new * q_old^{-1}).xyz
           (only the imaginary part of the relative quaternion)
        """
        s = self.state
        inv_dt = 1.0 / dt if dt > 1e-10 else 0.0

        for i in range(s.n_particles):
            if s.quat_inv_masses[i] > 0:
                # Store current as previous
                q_old = s.orientations[i].copy()
                s.prev_orientations[i] = q_old

                # Update orientation from predicted
                q_new = s.predicted_orientations[i].copy()
                s.orientations[i] = q_new

                # Derive angular velocity from quaternion difference
                # omega = 2/dt * (q_new * q_old^{-1}).xyz
                # q_old^{-1} for unit quaternion = conjugate = (-x, -y, -z, w)
                q_old_inv = self._quat_conjugate(q_old)

                # q_rel = q_new * q_old_inv
                q_rel = self._quat_multiply(q_new, q_old_inv)

                # Angular velocity = 2/dt * imaginary part
                # Sign convention: if q_rel.w < 0, flip sign to take shorter path
                sign = 1.0 if q_rel[3] >= 0 else -1.0
                s.angular_velocities[i, 0] = sign * 2.0 * inv_dt * q_rel[0]
                s.angular_velocities[i, 1] = sign * 2.0 * inv_dt * q_rel[1]
                s.angular_velocities[i, 2] = sign * 2.0 * inv_dt * q_rel[2]
            else:
                # Fixed orientation
                s.prev_orientations[i] = s.orientations[i].copy()
                s.orientations[i] = s.predicted_orientations[i].copy()
                s.angular_velocities[i, :3] = 0.0

    def _prepare_numpy(self, dt: float):
        """NumPy implementation of constraint preparation.

        This prepares the direct solver for a new timestep by:
        1. Resetting Lagrange multipliers (lambdas) to zero
        2. Computing compliance values from stiffness parameters
        3. Storing current rest shape parameters

        Following C++ reference (PrepareDirectElasticRodConstraints):

        Stretch compliance:
            - C++ uses near-zero compliance (1e-12 / dt²) for inextensible rods
            - We use the same for consistency

        Bend/twist stiffness:
            - C++ PrepareDirectElasticRodConstraints REPLACES stiffnessK with:
              stiffnessK = [bendStiff.x * youngModulusMult, bendStiff.y * youngModulusMult, bendStiff.z * torsionModulusMult]
            - This does NOT use E*I formula; it uses bendStiff * modulus directly
            - Compliance = (1/dt²) / stiffnessK / L

        Note: The stiffness coefficients are user parameters that directly control
        bending/twist response, NOT derived from material properties and cross-section.
        """
        s = self.state
        n_edges = s.n_edges

        # 1. Reset Lagrange multipliers
        self.lambdas.fill(0.0)

        # 2. Copy current rest shape parameters
        np.copyto(self.current_rest_lengths, s.rest_lengths)
        self.current_rest_darboux[:, 0] = self.rest_darboux_vec[:, 0]  # kappa1
        self.current_rest_darboux[:, 1] = self.rest_darboux_vec[:, 1]  # kappa2
        self.current_rest_darboux[:, 2] = self.rest_darboux_vec[:, 2]  # tau

        # 3. Compute compliance values following C++ reference
        dt2 = dt * dt
        inv_dt2 = 1.0 / dt2

        # C++ stretch regularization parameter (nearly inextensible)
        stretch_regularization = 1e-12

        for i in range(n_edges):
            L = self.current_rest_lengths[i]

            # === Stretch compliance (C++ uses tiny regularization) ===
            # stretchCompliance = stretchRegularizationParameter * (1/dt²)
            stretch_compliance = stretch_regularization * inv_dt2 * self.stretch_stiffness_mult
            self.compliance[i, 0] = stretch_compliance  # stretch
            self.compliance[i, 1] = stretch_compliance  # shear1
            self.compliance[i, 2] = stretch_compliance  # shear2

            # === Bend/twist stiffness coefficients K ===
            # C++ PrepareDirectElasticRodConstraints:
            #   stiffnessK = [bendStiff.x * youngModulusMult, bendStiff.y * youngModulusMult, bendStiff.z * torsionModulusMult]
            # This DIRECTLY uses bendStiff * modulus, NOT E*I formula!
            K_bend1 = self.bend_stiffness[i, 0] * self.young_modulus_mult * self.bend_stiffness_mult
            K_bend2 = self.bend_stiffness[i, 1] * self.young_modulus_mult * self.bend_stiffness_mult
            K_twist = self.bend_stiffness[i, 2] * self.torsion_modulus_mult * self.bend_stiffness_mult

            # === Bend/twist compliance ===
            # C++: bendingAndTorsionCompliance = (1/dt²) / K / L
            eps = 1e-10
            self.compliance[i, 3] = inv_dt2 / (K_bend1 + eps) / L  # bend1
            self.compliance[i, 4] = inv_dt2 / (K_bend2 + eps) / L  # bend2
            self.compliance[i, 5] = inv_dt2 / (K_twist + eps) / L  # twist

    def _update_numpy(self):
        """NumPy implementation of constraint update.

        This evaluates the current constraint violations following the C++ reference.

        1. Stretch-Shear Constraint (3 DOF per edge):
           connector0 = p0 + (L/2) * d3_0  (point on segment 0 at constraint location)
           connector1 = p1 - (L/2) * d3_1  (point on segment 1 at constraint location)
           C_ss = connector0 - connector1 (should be zero at rest)

        2. Bend-Twist Constraint (3 DOF per edge):
           C_bt = omega - omega_rest
           Where omega = im(q0^{-1} * q1) is the Darboux vector (WITHOUT 2/L factor).
        """
        s = self.state
        n_edges = s.n_edges

        for i in range(n_edges):
            # Get predicted positions and orientations
            p0 = s.predicted_positions[i, :3]
            p1 = s.predicted_positions[i + 1, :3]
            q0 = s.predicted_orientations[i]
            q1 = s.predicted_orientations[i + 1]

            L = self.current_rest_lengths[i]

            # === Stretch-Shear Constraint ===
            # Get rod axis directions d3 from quaternions
            # d3 is the third column of the rotation matrix (z-axis in local frame)
            d3_0 = self._quat_rotate_vector(q0, np.array([0, 0, 1], dtype=np.float32))
            d3_1 = self._quat_rotate_vector(q1, np.array([0, 0, 1], dtype=np.float32))

            # Connectors: points on each segment where constraint applies
            # connector0 = p0 + (L/2) * d3_0 (midpoint from segment 0's perspective)
            # connector1 = p1 - (L/2) * d3_1 (midpoint from segment 1's perspective)
            connector0 = p0 + 0.5 * L * d3_0
            connector1 = p1 - 0.5 * L * d3_1

            stretch_violation = connector0 - connector1

            self.constraint_values[i, 0] = stretch_violation[0]
            self.constraint_values[i, 1] = stretch_violation[1]
            self.constraint_values[i, 2] = stretch_violation[2]

            # === Bend-Twist Constraint ===
            # C++ uses: darbouxVector = (q0.conjugate() * q1).vec()
            # WITHOUT the 2/L factor!
            q0_inv = self._quat_conjugate(q0)
            q_rel = self._quat_multiply(q0_inv, q1)

            # Darboux vector is just the imaginary part (x, y, z) of the relative quaternion
            omega = q_rel[:3].copy()

            # Rest Darboux vector
            omega_rest = self.current_rest_darboux[i]

            # Bend-twist error
            darboux_error = omega - omega_rest
            self.constraint_values[i, 3] = darboux_error[0]  # bend1 (kappa1)
            self.constraint_values[i, 4] = darboux_error[1]  # bend2 (kappa2)
            self.constraint_values[i, 5] = darboux_error[2]  # twist (tau)

    def _quat_rotate_vector(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Rotate vector v by quaternion q.

        Uses the formula: v' = q * v * q^{-1}
        Optimized implementation without explicit quaternion multiplication.
        """
        x, y, z, w = q
        vx, vy, vz = v

        # t = 2 * cross(q.xyz, v)
        tx = 2.0 * (y * vz - z * vy)
        ty = 2.0 * (z * vx - x * vz)
        tz = 2.0 * (x * vy - y * vx)

        # v' = v + w * t + cross(q.xyz, t)
        return np.array([
            vx + w * tx + y * tz - z * ty,
            vy + w * ty + z * tx - x * tz,
            vz + w * tz + x * ty - y * tx,
        ], dtype=np.float32)

    def _quat_conjugate(self, q: np.ndarray) -> np.ndarray:
        """Compute quaternion conjugate (inverse for unit quaternions)."""
        return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float32)

    def _quat_multiply(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Multiply two quaternions: q1 * q2."""
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
        return np.array([
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ], dtype=np.float32)

    def _compute_matrix_G(self, q: np.ndarray) -> np.ndarray:
        """Compute the G matrix that converts angular velocity to quaternion derivative.

        From C++: G is a 4x3 matrix where q_dot = G * omega / 2
        """
        # Quaternion order: (x, y, z, w)
        x, y, z, w = q
        G = np.array([
            [0.5 * w,  0.5 * z, -0.5 * y],
            [-0.5 * z, 0.5 * w,  0.5 * x],
            [0.5 * y, -0.5 * x,  0.5 * w],
            [-0.5 * x, -0.5 * y, -0.5 * z]
        ], dtype=np.float32)
        return G

    def _compute_jOmega(self, q0: np.ndarray, q1: np.ndarray) -> tuple:
        """Compute the Jacobians of Darboux vector w.r.t. quaternion components.

        From C++:
        jOmega0 is ∂ω/∂q0 (3x4 matrix)
        jOmega1 is ∂ω/∂q1 (3x4 matrix)

        Where ω = im(q0^{-1} * q1)
        """
        # Quaternion order: (x, y, z, w)
        x0, y0, z0, w0 = q0
        x1, y1, z1, w1 = q1

        # From C++ computeBendingAndTorsionJacobians:
        # jOmega0 <<
        #     -q1.w(), -q1.z(), q1.y(), q1.x(),
        #     q1.z(), -q1.w(), -q1.x(), q1.y(),
        #     -q1.y(), q1.x(), -q1.w(), q1.z();
        jOmega0 = np.array([
            [-w1, -z1,  y1, x1],
            [ z1, -w1, -x1, y1],
            [-y1,  x1, -w1, z1]
        ], dtype=np.float32)

        # jOmega1 <<
        #     q0.w(), q0.z(), -q0.y(), -q0.x(),
        #     -q0.z(), q0.w(), q0.x(), -q0.y(),
        #     q0.y(), -q0.x(), q0.w(), -q0.z();
        jOmega1 = np.array([
            [ w0,  z0, -y0, -x0],
            [-z0,  w0,  x0, -y0],
            [ y0, -x0,  w0, -z0]
        ], dtype=np.float32)

        return jOmega0, jOmega1

    def _jacobians_numpy(self):
        """NumPy implementation of Jacobian computation following C++ reference.

        Following the C++ banded solver, we compute a full 6x6 Jacobian matrix J
        for each segment pair. The J matrix structure is:

        J = [ sign*I    -sign*[r]_×  ]  (stretch-shear: rows 0-2)
            [   0          jOmegaG    ]  (bend-twist: rows 3-5)

        Where columns are [position (3) | rotation (3)], so J maps from
        segment DOFs (pos, rot) to constraint violations.

        The JMJT computation then naturally produces the K matrix coupling
        between position and rotation through the [r]_× term.
        """
        s = self.state
        n_edges = s.n_edges

        # Allocate full 6x6 Jacobian storage per segment
        # J_fwd[i]: Jacobian from segment i to constraint i
        # J_bwd[i]: Jacobian from segment i+1 to constraint i
        if not hasattr(self, 'J_fwd') or self.J_fwd.shape[0] != n_edges:
            self.J_fwd = np.zeros((n_edges, 6, 6), dtype=np.float32)
            self.J_bwd = np.zeros((n_edges, 6, 6), dtype=np.float32)
            # Keep old storage for compatibility
            self.jacobian_pos = np.zeros((n_edges, 6, 6), dtype=np.float32)
            self.jacobian_rot = np.zeros((n_edges, 6, 6), dtype=np.float32)

        for i in range(n_edges):
            q0 = s.predicted_orientations[i]
            q1 = s.predicted_orientations[i + 1]

            L = self.current_rest_lengths[i]
            if L <= 1e-8:
                L = 1e-8

            # Get rod axis directions d3 from quaternions
            d3_0 = self._quat_rotate_vector(q0, np.array([0, 0, 1], dtype=np.float32))
            d3_1 = self._quat_rotate_vector(q1, np.array([0, 0, 1], dtype=np.float32))

            # Connector offsets from positions (in world frame)
            # r0 = connector0 - p0 = (L/2) * d3_0
            # r1 = connector1 - p1 = -(L/2) * d3_1
            r0 = 0.5 * L * d3_0
            r1 = -0.5 * L * d3_1

            # Compute G matrices for quaternion-to-angular mapping
            G0 = self._compute_matrix_G(q0)  # 4x3
            G1 = self._compute_matrix_G(q1)  # 4x3

            # Compute jOmega matrices for Darboux vector Jacobians (3x4)
            jOmega0, jOmega1 = self._compute_jOmega(q0, q1)

            # Bend-twist Jacobians: jOmega * G (3x3)
            jOmegaG0 = jOmega0 @ G0
            jOmegaG1 = jOmega1 @ G1

            # === Build forward Jacobian (segment 0 -> constraint) ===
            # Following C++: sign = 1 for forward
            sign_fwd = 1.0
            r0_cross = self._skew_symmetric(r0)

            # Upper-left: sign * I (position contribution to stretch)
            self.J_fwd[i, 0:3, 0:3] = sign_fwd * np.eye(3, dtype=np.float32)
            # Upper-right: -sign * [r]_× (rotation contribution to stretch via lever arm)
            self.J_fwd[i, 0:3, 3:6] = -sign_fwd * r0_cross
            # Lower-left: zero (bend-twist doesn't depend on position)
            self.J_fwd[i, 3:6, 0:3] = 0.0
            # Lower-right: jOmegaG (rotation contribution to bend-twist)
            self.J_fwd[i, 3:6, 3:6] = jOmegaG0

            # === Build backward Jacobian (segment 1 -> constraint) ===
            # Following C++: sign = -1 for backward
            sign_bwd = -1.0
            r1_cross = self._skew_symmetric(r1)

            # Upper-left: sign * I
            self.J_bwd[i, 0:3, 0:3] = sign_bwd * np.eye(3, dtype=np.float32)
            # Upper-right: -sign * [r]_× = +[r1]_×
            self.J_bwd[i, 0:3, 3:6] = -sign_bwd * r1_cross
            # Lower-left: zero
            self.J_bwd[i, 3:6, 0:3] = 0.0
            # Lower-right: jOmegaG
            self.J_bwd[i, 3:6, 3:6] = jOmegaG1

            # === Also fill old separate storage for backward compatibility ===
            # Position Jacobians (just the position columns)
            self.jacobian_pos[i, 0:3, 0:3] = self.J_fwd[i, 0:3, 0:3]
            self.jacobian_pos[i, 0:3, 3:6] = self.J_bwd[i, 0:3, 0:3]
            self.jacobian_pos[i, 3:6, 0:3] = 0.0
            self.jacobian_pos[i, 3:6, 3:6] = 0.0

            # Rotation Jacobians (the rotation columns)
            self.jacobian_rot[i, 0:3, 0:3] = self.J_fwd[i, 0:3, 3:6]
            self.jacobian_rot[i, 0:3, 3:6] = self.J_bwd[i, 0:3, 3:6]
            self.jacobian_rot[i, 3:6, 0:3] = self.J_fwd[i, 3:6, 3:6]
            self.jacobian_rot[i, 3:6, 3:6] = self.J_bwd[i, 3:6, 3:6]

    def _skew_symmetric(self, v: np.ndarray) -> np.ndarray:
        """Create skew-symmetric matrix from vector (cross product matrix)."""
        return np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0],
        ], dtype=np.float32)

    def _assemble_numpy(self):
        """NumPy implementation of JMJT assembly for banded solver.

        Following the C++ banded solver approach:
        - Uses full 6x6 Jacobians (J_fwd, J_bwd)
        - JMJT = J_fwd * J_fwd^T + J_bwd * J_bwd^T (unit mass/inertia)
        - Compliance added to diagonal
        - Off-diagonal coupling for adjacent constraints

        The matrix has block tri-diagonal structure:
        - Diagonal blocks: 6x6 (constraint with itself)
        - Super/sub-diagonal blocks: 6x6 (coupling between adjacent constraints)
        - Total bandwidth = 11 (block_size + block_size - 1 = 6 + 6 - 1)
        """
        s = self.state
        n_edges = s.n_edges
        n_dofs = 6 * n_edges

        # Bandwidth: block tridiagonal with 6x6 blocks requires bandwidth = 11
        # (diagonal spans -5 to +5, super-diagonal spans +1 to +11)
        self.bandwidth = 11

        # Allocate banded matrix storage
        if not hasattr(self, 'A_banded') or self.A_banded.shape[1] != n_dofs:
            self.A_banded = np.zeros((2 * self.bandwidth + 1, n_dofs), dtype=np.float32)
            self.rhs = np.zeros(n_dofs, dtype=np.float32)

        self.A_banded.fill(0.0)

        for i in range(n_edges):
            # Get full 6x6 Jacobians (computed by _jacobians_numpy)
            J_fwd = self.J_fwd[i]
            J_bwd = self.J_bwd[i]

            # Diagonal block: JMJT = J_fwd * J_fwd^T + J_bwd * J_bwd^T
            # This uses implicit unit mass/inertia (same as C++ banded solver)
            JMJT_block = J_fwd @ J_fwd.T + J_bwd @ J_bwd.T

            # Add compliance to diagonal
            for k in range(6):
                JMJT_block[k, k] += self.compliance[i, k]

            # Store diagonal block in banded format
            # scipy banded format: ab[u + i - j, j] = a[i, j] where u = upper bandwidth
            block_start = 6 * i
            for row in range(6):
                for col in range(6):
                    global_row = block_start + row
                    global_col = block_start + col
                    band_row = self.bandwidth + global_row - global_col
                    self.A_banded[band_row, global_col] = JMJT_block[row, col]

            # Off-diagonal coupling: constraint i shares segment i+1 with constraint i+1
            if i < n_edges - 1:
                J_fwd_next = self.J_fwd[i + 1]
                # Super-diagonal: A[i, i+1] = J_bwd[i] @ J_fwd[i+1]^T
                coupling = J_bwd @ J_fwd_next.T

                # Store super-diagonal block (rows i*6:(i+1)*6, cols (i+1)*6:(i+2)*6)
                for row in range(6):
                    for col in range(6):
                        global_row = block_start + row
                        global_col = block_start + 6 + col
                        band_row = self.bandwidth + global_row - global_col
                        self.A_banded[band_row, global_col] = coupling[row, col]

                # Store sub-diagonal block (transpose of super-diagonal for symmetric matrix)
                # Sub-diagonal: A[i+1, i] = coupling^T
                for row in range(6):
                    for col in range(6):
                        global_row = block_start + 6 + row
                        global_col = block_start + col
                        band_row = self.bandwidth + global_row - global_col
                        self.A_banded[band_row, global_col] = coupling[col, row]

    def _solve_numpy(self):
        """NumPy implementation of constraint solve for banded system.

        Solves the system: A * Δλ = -C
        where A = JMJT + α (assembled matrix), C = constraint values

        Following C++ banded solver:
        - Position corrections use actual inv_mass
        - Rotation corrections use unit inertia (matching assembly)
        """
        s = self.state
        n_edges = s.n_edges

        # Build RHS: -C (negative constraint values)
        for i in range(n_edges):
            for k in range(6):
                self.rhs[6 * i + k] = -self.constraint_values[i, k]

        # Solve banded system
        if self.use_numpy_solve_spbsv:
            delta_lambda = self._solve_banded_spbsv_u11_1rhs()
        else:
            try:
                from scipy.linalg import solve_banded
                delta_lambda = solve_banded(
                    (self.bandwidth, self.bandwidth),
                    self.A_banded,
                    self.rhs,
                    overwrite_ab=False,
                    overwrite_b=False
                )
            except ImportError:
                # Fallback: solve as dense system (slower)
                n_dofs = 6 * n_edges
                A_dense = np.zeros((n_dofs, n_dofs), dtype=np.float32)
                for col in range(n_dofs):
                    for band_row in range(2 * self.bandwidth + 1):
                        row = col + band_row - self.bandwidth
                        if 0 <= row < n_dofs:
                            A_dense[row, col] = self.A_banded[band_row, col]
                delta_lambda = np.linalg.solve(A_dense, self.rhs)

        # Apply corrections using full J matrices (matching non-banded approach)
        inv_masses = s.inv_masses

        for i in range(n_edges):
            dl = delta_lambda[6 * i: 6 * i + 6]

            J_fwd = self.J_fwd[i]
            J_bwd = self.J_bwd[i]

            # Correction for segment i (via J_fwd)
            inv_m0 = inv_masses[i]
            if inv_m0 > 0.0:
                correction0 = J_fwd.T @ dl
                dp0 = inv_m0 * correction0[:3]
                s.predicted_positions[i, :3] += dp0

            # Rotation correction: unit inertia (matching C++ banded solver)
            if s.quat_inv_masses[i] > 0.0:
                correction0 = J_fwd.T @ dl
                dtheta0 = correction0[3:6]
                self._apply_quaternion_correction(s.predicted_orientations, i, dtheta0)

            # Correction for segment i+1 (via J_bwd)
            inv_m1 = inv_masses[i + 1]
            if inv_m1 > 0.0:
                correction1 = J_bwd.T @ dl
                dp1 = inv_m1 * correction1[:3]
                s.predicted_positions[i + 1, :3] += dp1

            if s.quat_inv_masses[i + 1] > 0.0:
                correction1 = J_bwd.T @ dl
                dtheta1 = correction1[3:6]
                self._apply_quaternion_correction(s.predicted_orientations, i + 1, dtheta1)

    def _solve_banded_spbsv_u11_1rhs(self) -> np.ndarray:
        """Solve the banded system with in-place Cholesky (KD=11).

        Mirrors PositionBasedElasticRods.cpp spbsv_u11_1rhs:
        - Upper-band storage (rows 0..KD)
        - In-place Cholesky factorization of the band
        - Forward/back substitution for a single RHS

        Returns:
            Solution vector for the banded system.
        """
        kd = self.bandwidth
        n = self.rhs.shape[0]

        if n == 0:
            return self.rhs.copy()

        # Copy upper band (rows 0..KD) since factorization is in-place
        ab = self.A_banded[:kd + 1, :].copy()
        b = self.rhs.copy()

        # 1) In-place Cholesky on AB -> U
        for j in range(n):
            sum_sq = 0.0
            kmax = j if j < kd else kd
            for k in range(1, kmax + 1):
                u = ab[kd - k, j]
                sum_sq += u * u

            ajj = ab[kd, j] - sum_sq
            if ajj <= 1e-6:
                # Regularize near-singular diagonals (matches C++ path)
                ajj = 1e-6
            ujj = np.sqrt(ajj)
            ab[kd, j] = ujj

            imax = (n - j - 1) if (n - j - 1) < kd else kd
            for i in range(1, imax + 1):
                dot = 0.0
                k2max = j if j < (kd - i) else (kd - i)
                for k in range(1, k2max + 1):
                    dot += ab[kd - k, j] * ab[kd - i - k, j + i]
                aji = ab[kd - i, j + i] - dot
                ab[kd - i, j + i] = aji / ujj

        # 2) Forward solve U^T * y = B
        for i in range(n):
            sum_val = 0.0
            k0 = 0 if i < kd else i - kd
            for k in range(k0, i):
                sum_val += ab[kd + k - i, i] * b[k]
            b[i] = (b[i] - sum_val) / ab[kd, i]

        # 3) Backward solve U * x = y
        for i in range(n - 1, -1, -1):
            sum_val = 0.0
            k1 = (i + kd) if (i + kd) < n else (n - 1)
            for k in range(i + 1, k1 + 1):
                sum_val += ab[kd + i - k, k] * b[k]
            b[i] = (b[i] - sum_val) / ab[kd, i]

        return b

    def _apply_quaternion_correction(self, orientations: np.ndarray, idx: int, dtheta: np.ndarray):
        """Apply a rotation correction to a quaternion using the G matrix.

        Following C++ reference: deltaQ = G * dtheta
        where G is the 4x3 matrix that converts angular velocity to quaternion derivative.

        This properly accounts for the current quaternion orientation.
        """
        if np.linalg.norm(dtheta) < 1e-10:
            return

        q = orientations[idx]

        # Compute G matrix for current quaternion (4x3)
        G = self._compute_matrix_G(q)

        # Convert angular correction to quaternion correction: dq = G * dtheta
        dq = G @ dtheta

        # Apply correction: q_new = q + dq
        q_new = q + dq

        # Normalize
        norm = np.linalg.norm(q_new)
        if norm > 1e-10:
            q_new /= norm

        orientations[idx] = q_new

    # =========================================================================
    # Non-banded direct solver (single method replacing update+jacobians+assemble+solve)
    # =========================================================================

    def _project_direct_numpy(self):
        """NumPy implementation of non-banded direct constraint projection.

        Following the C++ banded solver approach:
        1. Compute constraint violations
        2. Compute full 6x6 Jacobians (J_fwd, J_bwd) for each constraint
        3. Assemble JMJT using J * J^T with unit mass/inertia
        4. Solve linear system
        5. Apply corrections with actual mass/inertia

        The C++ code uses unit mass/inertia in JMJT assembly (implicit in J*J^T),
        then applies actual masses during correction. This matches the paper's
        formulation where K matrix naturally emerges from J*J^T.
        """
        s = self.state
        n_edges = s.n_edges

        if n_edges == 0:
            return

        # Step 1: Compute constraint values
        self._update_numpy()

        # Step 2: Compute full Jacobians (J_fwd, J_bwd)
        self._jacobians_numpy()

        # Step 3: Build dense JMJT matrix following C++ banded solver
        # C++ uses J * J^T directly (unit mass/inertia in assembly)
        n_dofs = 6 * n_edges
        A = np.zeros((n_dofs, n_dofs), dtype=np.float32)
        rhs = -self.constraint_values.reshape(n_dofs)

        for i in range(n_edges):
            # Get full 6x6 Jacobians for this constraint
            J_fwd = self.J_fwd[i]  # (6, 6) - segment i contribution
            J_bwd = self.J_bwd[i]  # (6, 6) - segment i+1 contribution

            # Diagonal block: JMJT = J_fwd * J_fwd^T + J_bwd * J_bwd^T
            # This naturally produces the K matrix with position-rotation coupling!
            # K = I + [r]_× * [r]_×^T comes from J_fwd[:3,:] * J_fwd[:3,:]^T
            JMJT_block = J_fwd @ J_fwd.T + J_bwd @ J_bwd.T

            # Add compliance to diagonal (regularization / inverse stiffness)
            # C++ stores D = -compliance then does JMJT -= D, which equals JMJT += compliance
            for k in range(6):
                JMJT_block[k, k] += self.compliance[i, k]

            # Store in global matrix
            block = slice(6 * i, 6 * i + 6)
            A[block, block] = JMJT_block

            # Off-diagonal coupling: constraint i shares segment i+1 with constraint i+1
            # J_bwd[i] affects segment i+1, J_fwd[i+1] also affects segment i+1
            if i < n_edges - 1:
                J_fwd_next = self.J_fwd[i + 1]  # Next constraint's forward Jacobian
                # Coupling: J_bwd[i] * J_fwd[i+1]^T (shared segment i+1)
                coupling = J_bwd @ J_fwd_next.T
                next_block = slice(6 * (i + 1), 6 * (i + 1) + 6)
                A[block, next_block] = coupling
                A[next_block, block] = coupling.T

        # Step 4: Solve system A * Δλ = -C
        try:
            delta_lambda = np.linalg.solve(A, rhs)
        except np.linalg.LinAlgError:
            # Matrix is singular, use least squares
            delta_lambda = np.linalg.lstsq(A, rhs, rcond=None)[0]

        # Step 5: Apply corrections
        # Following C++ banded solver: corrections use actual masses for position,
        # but unit inertia for rotation (implicit in G matrix mapping)
        inv_masses = s.inv_masses

        for i in range(n_edges):
            dl = delta_lambda[6 * i: 6 * i + 6]

            J_fwd = self.J_fwd[i]
            J_bwd = self.J_bwd[i]

            # Correction for segment i (via J_fwd)
            # dx = J^T * dl, where x = [pos, rot]
            # Position correction: top 3 rows of J^T * dl, scaled by inv_mass
            inv_m0 = inv_masses[i]
            if inv_m0 > 0.0:
                # J_fwd^T @ dl gives [dp; dtheta], take position part
                correction0 = J_fwd.T @ dl
                dp0 = inv_m0 * correction0[:3]
                s.predicted_positions[i, :3] += dp0

            # Rotation correction: bottom 3 rows of J^T * dl
            # C++ uses unit inertia (correction = G * dtheta directly)
            if s.quat_inv_masses[i] > 0.0:
                correction0 = J_fwd.T @ dl
                dtheta0 = correction0[3:6]  # Unit inertia
                self._apply_quaternion_correction(s.predicted_orientations, i, dtheta0)

            # Correction for segment i+1 (via J_bwd)
            inv_m1 = inv_masses[i + 1]
            if inv_m1 > 0.0:
                correction1 = J_bwd.T @ dl
                dp1 = inv_m1 * correction1[:3]
                s.predicted_positions[i + 1, :3] += dp1

            if s.quat_inv_masses[i + 1] > 0.0:
                correction1 = J_bwd.T @ dl
                dtheta1 = correction1[3:6]  # Unit inertia
                self._apply_quaternion_correction(s.predicted_orientations, i + 1, dtheta1)

    # =========================================================================
    # Main step function
    # =========================================================================

    def step(self, dt: float):
        """Advance simulation by one timestep.

        Args:
            dt: Time step size in seconds.
        """
        s = self.state
        n_constraints = s.n_edges

        # 1. Predict positions
        if self.use_numpy_predict_positions:
            self._predict_positions_numpy(dt)
        else:
            self.dll.predict_positions(
                dt,
                self.position_damping,
                s.positions,
                s.predicted_positions,
                s.velocities,
                s.forces,
                s.inv_masses,
                self.gravity,
            )

        # 2. Predict orientations
        if self.use_numpy_predict_rotations:
            self._predict_rotations_numpy(dt)
        else:
            self.dll.predict_rotations(
                dt,
                self.rotation_damping,
                s.orientations,
                s.predicted_orientations,
                s.angular_velocities,
                s.torques,
                s.quat_inv_masses,
            )

        # 3. Prepare direct solver (always needed for compliance computation)
        if self.use_numpy_prepare or self.use_numpy_project_direct:
            self._prepare_numpy(dt)
        else:
            self.dll.prepare_direct_elastic_rod_constraints(
                self._rod_ptr,
                n_constraints,
                dt,
                self.bend_stiffness,
                self.rest_darboux_vec,
                s.rest_lengths,
                self.young_modulus_mult,
                self.torsion_modulus_mult,
            )

        # Steps 4-7: Either use non-banded (single call) or banded (separate steps)
        if self.use_numpy_project_direct:
            # Non-banded: single method does update+jacobians+assemble+solve
            self._project_direct_numpy()
        else:
            # Banded approach: separate steps

            # 4. Update constraint state
            if self.use_numpy_update:
                self._update_numpy()
            else:
                self.dll.update_direct_constraints(
                    self._rod_ptr,
                    s.predicted_positions,
                    s.predicted_orientations,
                    s.inv_masses,
                )

            # 5. Compute Jacobians
            if self.use_numpy_jacobians:
                self._jacobians_numpy()
            else:
                self.dll.compute_jacobians_direct(
                    self._rod_ptr,
                    0,
                    n_constraints,
                    s.predicted_positions,
                    s.predicted_orientations,
                    s.inv_masses,
                )

            # 6. Assemble JMJT banded matrix
            if self.use_numpy_assemble:
                self._assemble_numpy()
            else:
                self.dll.assemble_jmjt_direct(
                    self._rod_ptr,
                    0,
                    n_constraints,
                    s.predicted_positions,
                    s.predicted_orientations,
                    s.inv_masses,
                )

            # 7. Solve and apply corrections
            if self.use_numpy_solve:
                self._solve_numpy()
            else:
                self.dll.solve_direct_constraints(
                    self._rod_ptr,
                    s.predicted_positions,
                    s.predicted_orientations,
                s.inv_masses,
            )

        # 8. Integrate positions
        if self.use_numpy_integrate_positions:
            self._integrate_positions_numpy(dt)
        else:
            self.dll.integrate_positions(
                dt,
                s.positions,
                s.predicted_positions,
                s.velocities,
                s.inv_masses,
            )

        # 9. Integrate orientations
        if self.use_numpy_integrate_rotations:
            self._integrate_rotations_numpy(dt)
        else:
            self.dll.integrate_rotations(
                dt,
                s.orientations,
                s.predicted_orientations,
                s.prev_orientations,
                s.angular_velocities,
                s.quat_inv_masses,
            )

        # 10. Clear forces for next step
        s.clear_forces()

    def set_gravity(self, gx: float, gy: float, gz: float):
        """Set gravity vector."""
        self.gravity[0] = gx
        self.gravity[1] = gy
        self.gravity[2] = gz

    def set_rest_curvature(self, kappa1: float, kappa2: float, tau: float):
        """Set uniform rest curvature for the entire rod."""
        self.rest_darboux_vec[:, 0] = kappa1
        self.rest_darboux_vec[:, 1] = kappa2
        self.rest_darboux_vec[:, 2] = tau

    def set_bend_stiffness(self, k1: float, k2: float, k_tau: float):
        """Set uniform bending/twist stiffness for the entire rod."""
        self.bend_stiffness[:, 0] = k1
        self.bend_stiffness[:, 1] = k2
        self.bend_stiffness[:, 2] = k_tau
