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
        # With accurate Jacobians from C++ reference, these should be close to 1.0
        self.stretch_stiffness_mult = 1.0
        self.shear_stiffness_mult = 1.0
        self.bend_stiffness_mult = 1.0  # Now using accurate Jacobians

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

        # Precomputed cross-section properties
        # For circular cross-section with radius r:
        # A = pi * r^2 (area)
        # I = pi * r^4 / 4 (second moment of area for bending)
        # J = pi * r^4 / 2 (polar moment for torsion)
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
        """NumPy implementation of rotation prediction (placeholder)."""
        # TODO: Implement quaternion integration
        # For now, fall back to DLL
        s = self.state
        self.dll.predict_rotations(
            dt,
            self.rotation_damping,
            s.orientations,
            s.predicted_orientations,
            s.angular_velocities,
            s.torques,
            s.quat_inv_masses,
        )

    def _integrate_positions_numpy(self, dt: float):
        """NumPy implementation of position integration (placeholder)."""
        # TODO: Implement
        s = self.state
        self.dll.integrate_positions(
            dt,
            s.positions,
            s.predicted_positions,
            s.velocities,
            s.inv_masses,
        )

    def _integrate_rotations_numpy(self, dt: float):
        """NumPy implementation of rotation integration (placeholder)."""
        # TODO: Implement
        s = self.state
        self.dll.integrate_rotations(
            dt,
            s.orientations,
            s.predicted_orientations,
            s.prev_orientations,
            s.angular_velocities,
            s.quat_inv_masses,
        )

    def _prepare_numpy(self, dt: float):
        """NumPy implementation of constraint preparation.

        This prepares the direct solver for a new timestep by:
        1. Resetting Lagrange multipliers (lambdas) to zero
        2. Computing compliance values from stiffness parameters
        3. Storing current rest shape parameters

        The compliance α is the inverse stiffness scaled by dt²:
            α = 1 / (k * dt²)

        For elastic rods:
        - Stretch stiffness: k_s = E * A / L  (axial stiffness)
        - Shear stiffness: same as stretch for isotropic
        - Bend stiffness: k_b = E * I / L  (flexural stiffness)
        - Twist stiffness: k_t = G * J / L  (torsional stiffness)

        Where:
        - E = Young's modulus
        - G = Torsion/shear modulus
        - A = cross-section area = π * r²
        - I = second moment of area = π * r⁴ / 4
        - J = polar moment = π * r⁴ / 2
        - L = rest length
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

        # 3. Compute compliance values
        # Effective moduli
        E = self.young_modulus * self.young_modulus_mult
        G = self.torsion_modulus * self.torsion_modulus_mult

        # Cross-section properties
        A = self.cross_section_area
        I = self.second_moment_area
        J = self.polar_moment

        dt2 = dt * dt

        for i in range(n_edges):
            L = self.current_rest_lengths[i]

            # Stiffness values
            # Stretch and shear use axial stiffness
            k_stretch = E * A / L * self.stretch_stiffness_mult
            k_shear = E * A / L * self.shear_stiffness_mult

            # Bending stiffness (scaled by bend_stiffness coefficients and multiplier)
            k_bend1 = E * I / L * self.bend_stiffness[i, 0] * self.bend_stiffness_mult
            k_bend2 = E * I / L * self.bend_stiffness[i, 1] * self.bend_stiffness_mult

            # Twist stiffness (scaled by twist coefficient and multiplier)
            k_twist = G * J / L * self.bend_stiffness[i, 2] * self.bend_stiffness_mult

            # Compliance = 1 / (stiffness * dt²)
            # Add small epsilon to avoid division by zero
            eps = 1e-10

            self.compliance[i, 0] = 1.0 / (k_stretch * dt2 + eps)  # stretch
            self.compliance[i, 1] = 1.0 / (k_shear * dt2 + eps)    # shear1
            self.compliance[i, 2] = 1.0 / (k_shear * dt2 + eps)    # shear2
            self.compliance[i, 3] = 1.0 / (k_bend1 * dt2 + eps)    # bend1
            self.compliance[i, 4] = 1.0 / (k_bend2 * dt2 + eps)    # bend2
            self.compliance[i, 5] = 1.0 / (k_twist * dt2 + eps)    # twist

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

        The Jacobian structure per constraint is a 6x6 matrix for each segment:
        J = [ I or -I     r_cross    ]  (stretch-shear w.r.t. position and rotation)
            [ 0           jOmega*G   ]  (bend-twist w.r.t. rotation only)

        Where:
        - r_cross is the skew-symmetric matrix of (connector - position)
          r0 = (L/2) * d3_0, r1 = -(L/2) * d3_1
        - jOmega*G is the 3x3 bend-twist Jacobian from quaternion formulas
        """
        s = self.state
        n_edges = s.n_edges

        # Allocate Jacobian storage
        if not hasattr(self, 'jacobian_pos') or self.jacobian_pos.shape[0] != n_edges:
            self.jacobian_pos = np.zeros((n_edges, 6, 6), dtype=np.float32)
            self.jacobian_rot = np.zeros((n_edges, 6, 6), dtype=np.float32)

        for i in range(n_edges):
            q0 = s.predicted_orientations[i]
            q1 = s.predicted_orientations[i + 1]

            L = self.current_rest_lengths[i]
            if L <= 1e-8:
                L = 1e-8

            # === Stretch-Shear Jacobians (rows 0-2) ===
            # C = connector0 - connector1
            # connector0 = p0 + (L/2) * d3_0
            # connector1 = p1 - (L/2) * d3_1
            # ∂C/∂p0 = I, ∂C/∂p1 = -I
            self.jacobian_pos[i, 0:3, 0:3] = np.eye(3, dtype=np.float32)   # w.r.t. p_i
            self.jacobian_pos[i, 0:3, 3:6] = -np.eye(3, dtype=np.float32)  # w.r.t. p_{i+1}

            # Get rod axis directions d3 from quaternions
            d3_0 = self._quat_rotate_vector(q0, np.array([0, 0, 1], dtype=np.float32))
            d3_1 = self._quat_rotate_vector(q1, np.array([0, 0, 1], dtype=np.float32))

            # Connector offsets from positions
            # r0 = connector0 - p0 = (L/2) * d3_0
            # r1 = connector1 - p1 = -(L/2) * d3_1
            r0 = 0.5 * L * d3_0
            r1 = -0.5 * L * d3_1

            r0_cross = self._skew_symmetric(r0)
            r1_cross = self._skew_symmetric(r1)

            # Sign convention from C++: sign=1 for segment0, sign=-1 for segment1
            # ∂C/∂θ0 = ∂connector0/∂θ0 = -[r0]_× (derivative of rotating r0)
            # ∂C/∂θ1 = -∂connector1/∂θ1 = +[r1]_× (connector1 has negative contribution)
            self.jacobian_rot[i, 0:3, 0:3] = -r0_cross  # w.r.t. θ_i
            self.jacobian_rot[i, 0:3, 3:6] = r1_cross   # w.r.t. θ_{i+1}

            # === Bend-Twist Jacobians (rows 3-5) ===
            # C_bt = ω - ω_rest, where ω = im(q0^{-1} * q1)
            # The Jacobian is: ∂ω/∂θ = jOmega * G
            # where jOmega is 3x4 and G is 4x3

            # Compute G matrices
            G0 = self._compute_matrix_G(q0)  # 4x3
            G1 = self._compute_matrix_G(q1)  # 4x3

            # Compute jOmega matrices (3x4)
            jOmega0, jOmega1 = self._compute_jOmega(q0, q1)

            # Bend-twist Jacobians: jOmega * G (3x3)
            self.jacobian_rot[i, 3:6, 0:3] = jOmega0 @ G0  # w.r.t. θ_i
            self.jacobian_rot[i, 3:6, 3:6] = jOmega1 @ G1  # w.r.t. θ_{i+1}

            # Bend-twist doesn't depend on positions
            self.jacobian_pos[i, 3:6, 0:3] = 0.0
            self.jacobian_pos[i, 3:6, 3:6] = 0.0

    def _skew_symmetric(self, v: np.ndarray) -> np.ndarray:
        """Create skew-symmetric matrix from vector (cross product matrix)."""
        return np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0],
        ], dtype=np.float32)

    def _assemble_numpy(self):
        """NumPy implementation of JMJT assembly.

        Assembles the system matrix A = J * M^{-1} * J^T + α
        where:
        - J is the constraint Jacobian
        - M^{-1} is the inverse mass matrix
        - α is the compliance (regularization)

        For a rod with n_edges constraints and 6 DOFs per constraint,
        the system has size 6*n_edges x 6*n_edges.

        The matrix has banded structure due to local connectivity:
        - Each constraint only affects adjacent particles
        - Bandwidth = 12 (6 DOFs overlap with previous + 6 with next)

        We store in banded format for scipy.linalg.solve_banded.
        """
        s = self.state
        n_edges = s.n_edges
        n_dofs = 6 * n_edges  # Total constraint DOFs

        # Bandwidth: each constraint block (6x6) overlaps with neighbors
        # Upper bandwidth = lower bandwidth = 6 (one constraint block)
        self.bandwidth = 6

        # Allocate banded matrix storage
        # Format for solve_banded: (l+u+1, n) where l=u=bandwidth
        # Row i contains diagonal offset i-bandwidth
        if not hasattr(self, 'A_banded') or self.A_banded.shape[1] != n_dofs:
            self.A_banded = np.zeros((2 * self.bandwidth + 1, n_dofs), dtype=np.float32)
            self.rhs = np.zeros(n_dofs, dtype=np.float32)

        self.A_banded.fill(0.0)

        for i in range(n_edges):
            # Get inverse masses
            inv_m0 = s.inv_masses[i]
            inv_m1 = s.inv_masses[i + 1]
            inv_I0 = s.quat_inv_masses[i]  # Rotational inverse mass
            inv_I1 = s.quat_inv_masses[i + 1]

            # Get Jacobians for this constraint
            J_pos = self.jacobian_pos[i]  # (6, 6) - [J_p0 | J_p1]
            J_rot = self.jacobian_rot[i]  # (6, 6) - [J_θ0 | J_θ1]

            # Build local JMJT block
            # JMJT = J_p0 * inv_m0 * J_p0^T + J_p1 * inv_m1 * J_p1^T
            #      + J_θ0 * inv_I0 * J_θ0^T + J_θ1 * inv_I1 * J_θ1^T
            J_p0 = J_pos[:, 0:3]
            J_p1 = J_pos[:, 3:6]
            J_t0 = J_rot[:, 0:3]
            J_t1 = J_rot[:, 3:6]

            JMJT = (inv_m0 * J_p0 @ J_p0.T +
                    inv_m1 * J_p1 @ J_p1.T +
                    inv_I0 * J_t0 @ J_t0.T +
                    inv_I1 * J_t1 @ J_t1.T)

            # Add compliance to diagonal
            for k in range(6):
                JMJT[k, k] += self.compliance[i, k]

            # Store in banded format
            # The block for constraint i starts at row/col 6*i
            block_start = 6 * i

            for row in range(6):
                for col in range(6):
                    global_row = block_start + row
                    global_col = block_start + col

                    # Banded storage: A_banded[bandwidth + row - col, col] = A[row, col]
                    band_row = self.bandwidth + global_row - global_col
                    if 0 <= band_row < 2 * self.bandwidth + 1:
                        self.A_banded[band_row, global_col] += JMJT[row, col]

            # Add coupling with previous constraint (if exists)
            if i > 0:
                # The coupling comes from shared particles between constraints
                # Constraint i-1 affects particle i, constraint i also affects particle i
                # This creates off-diagonal blocks
                # For simplicity, we approximate this coupling
                pass  # TODO: Add off-diagonal coupling for better accuracy

    def _solve_numpy(self):
        """NumPy implementation of constraint solve.

        Solves the system: A * Δλ = -C
        where A = JMJT + α (assembled matrix), C = constraint values

        Then applies corrections:
        - Δp = M^{-1} * J_pos^T * Δλ
        - Δθ = I^{-1} * J_rot^T * Δλ (in tangent space)
        """
        s = self.state
        n_edges = s.n_edges

        # Build RHS: -C (negative constraint values)
        for i in range(n_edges):
            for k in range(6):
                self.rhs[6 * i + k] = -self.constraint_values[i, k]

        # Solve banded system
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

        # Apply corrections
        for i in range(n_edges):
            # Get delta lambda for this constraint
            dl = delta_lambda[6 * i: 6 * i + 6]

            # Get inverse masses
            inv_m0 = s.inv_masses[i]
            inv_m1 = s.inv_masses[i + 1]
            inv_I0 = s.quat_inv_masses[i]
            inv_I1 = s.quat_inv_masses[i + 1]

            # Get Jacobians
            J_p0 = self.jacobian_pos[i, :, 0:3]
            J_p1 = self.jacobian_pos[i, :, 3:6]
            J_t0 = self.jacobian_rot[i, :, 0:3]
            J_t1 = self.jacobian_rot[i, :, 3:6]

            # Position corrections: Δp = inv_m * J^T * Δλ
            dp0 = inv_m0 * (J_p0.T @ dl)
            dp1 = inv_m1 * (J_p1.T @ dl)

            # Orientation corrections (in tangent space): Δθ = inv_I * J^T * Δλ
            dtheta0 = inv_I0 * (J_t0.T @ dl)
            dtheta1 = inv_I1 * (J_t1.T @ dl)

            # Apply position corrections
            s.predicted_positions[i, :3] += dp0
            s.predicted_positions[i + 1, :3] += dp1

            # Apply orientation corrections (convert tangent vector to quaternion update)
            self._apply_quaternion_correction(s.predicted_orientations, i, dtheta0)
            self._apply_quaternion_correction(s.predicted_orientations, i + 1, dtheta1)

    def _apply_quaternion_correction(self, orientations: np.ndarray, idx: int, dtheta: np.ndarray):
        """Apply a small rotation correction to a quaternion.

        For small angle δθ, the quaternion update is:
        q' = q + 0.5 * [δθ_x, δθ_y, δθ_z, 0] * q
        """
        if np.linalg.norm(dtheta) < 1e-10:
            return

        q = orientations[idx]

        # Quaternion for small rotation: (δθ/2, 1) ≈ (sin(|δθ|/2) * δθ/|δθ|, cos(|δθ|/2))
        # For small angles: ≈ (δθ/2, 1)
        dq = np.array([0.5 * dtheta[0], 0.5 * dtheta[1], 0.5 * dtheta[2], 0.0], dtype=np.float32)

        # q' = q + dq * q (using quaternion multiplication)
        q_new = q + self._quat_multiply(dq, q)

        # Normalize
        q_new /= np.linalg.norm(q_new)

        orientations[idx] = q_new

    # =========================================================================
    # Non-banded direct solver (single method replacing update+jacobians+assemble+solve)
    # =========================================================================

    def _project_direct_numpy(self):
        """NumPy implementation of non-banded direct constraint projection.

        This matches the working reference implementation in numpy_cosserat_codex.py.
        It reuses the constraint update and Jacobian computation methods, then
        builds and solves a dense system.
        """
        s = self.state
        n_edges = s.n_edges

        if n_edges == 0:
            return

        # Step 1: Compute constraint values using existing method
        self._update_numpy()

        # Step 2: Compute Jacobians using existing method
        self._jacobians_numpy()

        # Step 3: Build dense JMJT matrix
        n_dofs = 6 * n_edges
        A = np.zeros((n_dofs, n_dofs), dtype=np.float32)
        rhs = -self.constraint_values.reshape(n_dofs)

        inv_masses = s.inv_masses
        inv_I = s.quat_inv_masses

        for i in range(n_edges):
            # Get stored Jacobians
            J_pos = self.jacobian_pos[i]  # (6, 6) - [J_p0 | J_p1]
            J_rot = self.jacobian_rot[i]  # (6, 6) - [J_t0 | J_t1]
            J_p0 = J_pos[:, 0:3]  # (6, 3)
            J_p1 = J_pos[:, 3:6]  # (6, 3)
            J_t0 = J_rot[:, 0:3]  # (6, 3)
            J_t1 = J_rot[:, 3:6]  # (6, 3)

            inv_m0 = inv_masses[i]
            inv_m1 = inv_masses[i + 1]
            inv_I0 = inv_I[i]
            inv_I1 = inv_I[i + 1]

            # Build diagonal block: JMJT = J * M^-1 * J^T
            JMJT = (
                inv_m0 * (J_p0 @ J_p0.T) +
                inv_m1 * (J_p1 @ J_p1.T) +
                inv_I0 * (J_t0 @ J_t0.T) +
                inv_I1 * (J_t1 @ J_t1.T)
            )

            # Add compliance to diagonal
            JMJT += np.diag(self.compliance[i])

            # Store in global matrix
            block = slice(6 * i, 6 * i + 6)
            A[block, block] += JMJT

            # Off-diagonal coupling with previous constraint (shared particle)
            if i > 0:
                J_pos_prev = self.jacobian_pos[i - 1]
                J_rot_prev = self.jacobian_rot[i - 1]
                J_p1_prev = J_pos_prev[:, 3:6]  # Constraint i-1's Jacobian w.r.t. particle i
                J_t1_prev = J_rot_prev[:, 3:6]  # Constraint i-1's Jacobian w.r.t. orientation i

                # Coupling through shared particle i (position and orientation)
                coupling = inv_m0 * (J_p1_prev @ J_p0.T) + inv_I0 * (J_t1_prev @ J_t0.T)

                prev_block = slice(6 * (i - 1), 6 * (i - 1) + 6)
                A[prev_block, block] += coupling
                A[block, prev_block] += coupling.T

        # Step 4: Solve system A * Δλ = -C
        try:
            delta_lambda = np.linalg.solve(A, rhs)
        except np.linalg.LinAlgError:
            # Matrix is singular, use least squares
            delta_lambda = np.linalg.lstsq(A, rhs, rcond=None)[0]

        # Step 5: Apply corrections using stored Jacobians
        for i in range(n_edges):
            dl = delta_lambda[6 * i: 6 * i + 6]

            J_pos = self.jacobian_pos[i]
            J_rot = self.jacobian_rot[i]
            J_p0 = J_pos[:, 0:3]
            J_p1 = J_pos[:, 3:6]
            J_t0 = J_rot[:, 0:3]
            J_t1 = J_rot[:, 3:6]

            inv_m0 = inv_masses[i]
            inv_m1 = inv_masses[i + 1]
            inv_I0 = inv_I[i]
            inv_I1 = inv_I[i + 1]

            # Position corrections: Δp = inv_m * J_p^T * Δλ
            if inv_m0 > 0.0:
                dp0 = inv_m0 * (J_p0.T @ dl)
                s.predicted_positions[i, :3] += dp0
            if inv_m1 > 0.0:
                dp1 = inv_m1 * (J_p1.T @ dl)
                s.predicted_positions[i + 1, :3] += dp1

            # Orientation corrections: Δθ = inv_I * J_t^T * Δλ
            if inv_I0 > 0.0:
                dtheta0 = inv_I0 * (J_t0.T @ dl)
                self._apply_quaternion_correction(s.predicted_orientations, i, dtheta0)
            if inv_I1 > 0.0:
                dtheta1 = inv_I1 * (J_t1.T @ dl)
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
