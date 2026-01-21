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

###########################################################################
# Example Block-Tridiagonal Cosserat Rod Solver
#
# Demonstrates a block-tridiagonal solver for Cosserat rod constraints
# where each element has multiple coupled DOFs (stretch-shear + bend-twist).
#
# The system matrix has block-tridiagonal structure:
#
#   [ D₀  U₀   0   0  ... ]   [ x₀ ]   [ b₀ ]
#   [ L₁  D₁  U₁   0  ... ]   [ x₁ ]   [ b₁ ]
#   [  0  L₂  D₂  U₂  ... ] × [ x₂ ] = [ b₂ ]
#   [  ⋮           ⋱      ]   [  ⋮ ]   [  ⋮ ]
#
# Where each block is BLOCK×BLOCK (e.g., 6×6 for full Cosserat).
#
# Block Thomas Algorithm:
#   Forward: D'[0] = D[0], for i>0: D'[i] = D[i] - L[i] @ inv(D'[i-1]) @ U[i-1]
#            b'[0] = b[0], for i>0: b'[i] = b[i] - L[i] @ inv(D'[i-1]) @ b'[i-1]
#   Back:    x[n-1] = inv(D'[n-1]) @ b'[n-1]
#            x[i] = inv(D'[i]) @ (b'[i] - U[i] @ x[i+1])
#
# This example uses a simplified 3-DOF block (stretch only) for demonstration.
# Extend BLOCK to 6 for full stretch-shear + bend-twist coupling.
#
# Command: uv run -m newton.examples cosserat_block_tridiagonal
#
###########################################################################

import warp as wp

import newton
import newton.examples

# Block configuration
BLOCK = 3  # 3×3 blocks (stretch-shear). Use 6 for full Cosserat (+ bend-twist)
BLOCK_DIM = 128

# Rod configuration
NUM_ELEMENTS = 32
NUM_NODES = NUM_ELEMENTS + 1


@wp.func
def mat3_inverse(m: wp.mat33) -> wp.mat33:
    """Compute inverse of 3×3 matrix."""
    det = wp.determinant(m)
    if wp.abs(det) < 1.0e-10:
        return wp.identity(n=3, dtype=float)

    inv_det = 1.0 / det

    # Adjugate matrix
    result = wp.mat33(
        (m[1, 1] * m[2, 2] - m[1, 2] * m[2, 1]) * inv_det,
        (m[0, 2] * m[2, 1] - m[0, 1] * m[2, 2]) * inv_det,
        (m[0, 1] * m[1, 2] - m[0, 2] * m[1, 1]) * inv_det,
        (m[1, 2] * m[2, 0] - m[1, 0] * m[2, 2]) * inv_det,
        (m[0, 0] * m[2, 2] - m[0, 2] * m[2, 0]) * inv_det,
        (m[0, 2] * m[1, 0] - m[0, 0] * m[1, 2]) * inv_det,
        (m[1, 0] * m[2, 1] - m[1, 1] * m[2, 0]) * inv_det,
        (m[0, 1] * m[2, 0] - m[0, 0] * m[2, 1]) * inv_det,
        (m[0, 0] * m[1, 1] - m[0, 1] * m[1, 0]) * inv_det,
    )
    return result


@wp.func
def load_mat3(arr: wp.array3d(dtype=float), block_idx: int) -> wp.mat33:
    """Load a 3×3 block from array."""
    return wp.mat33(
        arr[block_idx, 0, 0], arr[block_idx, 0, 1], arr[block_idx, 0, 2],
        arr[block_idx, 1, 0], arr[block_idx, 1, 1], arr[block_idx, 1, 2],
        arr[block_idx, 2, 0], arr[block_idx, 2, 1], arr[block_idx, 2, 2],
    )


@wp.func
def store_mat3(arr: wp.array3d(dtype=float), block_idx: int, m: wp.mat33):
    """Store a 3×3 block to array."""
    arr[block_idx, 0, 0] = m[0, 0]
    arr[block_idx, 0, 1] = m[0, 1]
    arr[block_idx, 0, 2] = m[0, 2]
    arr[block_idx, 1, 0] = m[1, 0]
    arr[block_idx, 1, 1] = m[1, 1]
    arr[block_idx, 1, 2] = m[1, 2]
    arr[block_idx, 2, 0] = m[2, 0]
    arr[block_idx, 2, 1] = m[2, 1]
    arr[block_idx, 2, 2] = m[2, 2]


@wp.func
def load_vec3(arr: wp.array2d(dtype=float), block_idx: int) -> wp.vec3:
    """Load a 3-vector from array."""
    return wp.vec3(arr[block_idx, 0], arr[block_idx, 1], arr[block_idx, 2])


@wp.func
def store_vec3(arr: wp.array2d(dtype=float), block_idx: int, v: wp.vec3):
    """Store a 3-vector to array."""
    arr[block_idx, 0] = v[0]
    arr[block_idx, 1] = v[1]
    arr[block_idx, 2] = v[2]


@wp.kernel
def integrate_nodes_kernel(
    node_q: wp.array(dtype=wp.vec3),
    node_qd: wp.array(dtype=wp.vec3),
    node_inv_mass: wp.array(dtype=float),
    gravity: wp.vec3,
    dt: float,
    # outputs
    node_q_pred: wp.array(dtype=wp.vec3),
    node_qd_new: wp.array(dtype=wp.vec3),
):
    """Semi-implicit Euler integration for rod nodes."""
    tid = wp.tid()
    inv_mass = node_inv_mass[tid]

    if inv_mass == 0.0:
        node_q_pred[tid] = node_q[tid]
        node_qd_new[tid] = node_qd[tid]
        return

    v_new = node_qd[tid] + gravity * dt
    x_pred = node_q[tid] + v_new * dt

    node_q_pred[tid] = x_pred
    node_qd_new[tid] = v_new


@wp.kernel
def compute_element_frames_kernel(
    node_q: wp.array(dtype=wp.vec3),
    rest_length: wp.array(dtype=float),
    num_elements: int,
    # outputs
    element_strain: wp.array(dtype=wp.vec3),  # stretch-shear strain
    element_director: wp.array(dtype=wp.mat33),  # local frame (d1, d2, d3)
):
    """
    Compute element strains and local frames.

    For each element, we compute:
    - Tangent vector t = (x_{i+1} - x_i) / L
    - Strain ε = [|edge|/L - 1, shear_1, shear_2]  (simplified)
    - Director frame (for full Cosserat, includes material rotation)
    """
    tid = wp.tid()
    if tid >= num_elements:
        return

    x0 = node_q[tid]
    x1 = node_q[tid + 1]
    L = rest_length[tid]

    edge = x1 - x0
    length = wp.length(edge)

    # Tangent (d3 direction)
    if length > 1.0e-8:
        t = edge / length
    else:
        t = wp.vec3(0.0, 0.0, 1.0)

    # Simplified strain: [stretch, shear_y, shear_z]
    # stretch = |edge|/L - 1 (0 at rest)
    stretch = length / L - 1.0

    # For now, assume shear is zero (extensible rod approximation)
    # In full Cosserat, shear comes from d3 · (x1-x0)/|x1-x0| != 1
    element_strain[tid] = wp.vec3(stretch, 0.0, 0.0)

    # Build a local frame (simplified - just from tangent)
    # For full Cosserat, this would involve material frame rotation
    d3 = t

    # Find orthogonal vectors d1, d2
    if wp.abs(d3[0]) < 0.9:
        d1 = wp.normalize(wp.cross(wp.vec3(1.0, 0.0, 0.0), d3))
    else:
        d1 = wp.normalize(wp.cross(wp.vec3(0.0, 1.0, 0.0), d3))
    d2 = wp.cross(d3, d1)

    element_director[tid] = wp.mat33(
        d1[0], d2[0], d3[0],
        d1[1], d2[1], d3[1],
        d1[2], d2[2], d3[2],
    )


@wp.kernel
def assemble_block_system_kernel(
    node_inv_mass: wp.array(dtype=float),
    element_strain: wp.array(dtype=wp.vec3),
    element_director: wp.array(dtype=wp.mat33),
    rest_length: wp.array(dtype=float),
    stiffness: wp.vec3,  # [k_stretch, k_shear_1, k_shear_2]
    compliance_factor: float,
    num_elements: int,
    # outputs - block tridiagonal system
    D_blocks: wp.array3d(dtype=float),  # (n, 3, 3) diagonal blocks
    L_blocks: wp.array3d(dtype=float),  # (n-1, 3, 3) lower diagonal blocks
    U_blocks: wp.array3d(dtype=float),  # (n-1, 3, 3) upper diagonal blocks
    b_blocks: wp.array2d(dtype=float),  # (n, 3) RHS blocks
):
    """
    Assemble block-tridiagonal system for Cosserat rod constraints.

    System matrix A = J M^{-1} J^T + α/dt² I

    For element k connecting nodes k and k+1:
    - Jacobian J_k has blocks at columns k and k+1
    - J_k^T M^{-1} J_k contributes to D[k], D[k+1], L[k+1], U[k]

    Block structure (for stretch-shear only):
    - D[i] = sum of contributions from elements i-1 and i
    - L[i] = coupling from element i-1 (node i is right node)
    - U[i] = coupling from element i (node i is left node)
    """
    # Single thread assembles (could parallelize with atomics)

    # Initialize to zero
    for i in range(num_elements):
        for r in range(BLOCK):
            b_blocks[i, r] = 0.0
            for c in range(BLOCK):
                D_blocks[i, r, c] = 0.0
                if i < num_elements - 1:
                    L_blocks[i, r, c] = 0.0
                    U_blocks[i, r, c] = 0.0

    # For each element, compute constraint contribution
    for k in range(num_elements):
        L_k = rest_length[k]
        R_k = element_director[k]  # local frame
        strain_k = element_strain[k]

        # Inverse masses of connected nodes
        w0 = node_inv_mass[k]
        w1 = node_inv_mass[k + 1]

        # Constraint Jacobian in local frame (simplified for stretch)
        # For stretch constraint: C = |x1-x0|/L - 1
        # dC/dx0 = -t/L, dC/dx1 = +t/L where t = (x1-x0)/|x1-x0|
        # In local frame, this is approximately [-1/L, 0, 0] and [1/L, 0, 0]

        # For the block system, we work in constraint space
        # J_left = -R^T / L (3×3 block for node k)
        # J_right = +R^T / L (3×3 block for node k+1)

        inv_L = 1.0 / L_k

        # Contribution to A = J M^{-1} J^T
        # A[k,k] += J_left @ (w0 * I) @ J_left^T = w0 * R^T R / L² = w0/L² * I
        # A[k+1,k+1] += J_right @ (w1 * I) @ J_right^T = w1/L² * I
        # A[k,k+1] += J_left @ 0 @ J_right^T = 0 (no shared mass)
        # But wait - the constraint couples nodes, so off-diagonal comes from
        # the fact that both nodes affect the same constraint.

        # Actually for JMJT with constraint k:
        # Row k of J has: [..., -n_k/L, +n_k/L, ...] at columns for nodes k, k+1
        # (J M^{-1} J^T)[k,k] = w_k * ||-n/L||² + w_{k+1} * ||+n/L||² = (w_k + w_{k+1})/L²

        # For block case with 3 constraints per element:
        # Each constraint row i couples to nodes k, k+1
        # D[k] gets contribution from element k's left side: w_k * J_left^T J_left
        # D[k] also gets contribution from element k-1's right side (if k > 0)

        # Simplified: assume identity coupling (stretch only affects tangent direction)
        scale = inv_L * inv_L

        # Diagonal contribution to node k (from being left node of element k)
        for r in range(BLOCK):
            for c in range(BLOCK):
                if r == c:
                    D_blocks[k, r, c] = D_blocks[k, r, c] + w0 * scale

        # Diagonal contribution to node k+1 (from being right node of element k)
        # This goes into D[k+1] but we handle elements sequentially
        # We'll add it when processing element k to the NEXT diagonal block
        if k + 1 < num_elements:
            for r in range(BLOCK):
                for c in range(BLOCK):
                    if r == c:
                        D_blocks[k + 1, r, c] = D_blocks[k + 1, r, c] + w1 * scale

        # Off-diagonal coupling between constraints k and k+1
        # This comes from shared node k+1
        if k + 1 < num_elements:
            # Coupling term: -w_{k+1} * (n_k · n_{k+1}) / (L_k * L_{k+1})
            # For blocks, this becomes a 3×3 coupling matrix
            L_k1 = rest_length[k + 1]
            coupling_scale = -w1 * inv_L / L_k1

            # Simplified: assume tangent directions are similar
            # Full version would use: R_k^T @ R_{k+1}
            for r in range(BLOCK):
                for c in range(BLOCK):
                    if r == c:
                        L_blocks[k, r, c] = coupling_scale
                        U_blocks[k, r, c] = coupling_scale

        # RHS: -C_k (constraint violation)
        # For stretch: C = strain[0] = |edge|/L - 1
        b_blocks[k, 0] = -strain_k[0]
        b_blocks[k, 1] = -strain_k[1]
        b_blocks[k, 2] = -strain_k[2]

    # Add compliance to diagonal
    for i in range(num_elements):
        for r in range(BLOCK):
            D_blocks[i, r, r] = D_blocks[i, r, r] + compliance_factor


@wp.kernel
def block_thomas_solve_kernel(
    D_blocks: wp.array3d(dtype=float),  # (n, 3, 3)
    L_blocks: wp.array3d(dtype=float),  # (n-1, 3, 3)
    U_blocks: wp.array3d(dtype=float),  # (n-1, 3, 3)
    b_blocks: wp.array2d(dtype=float),  # (n, 3)
    num_elements: int,
    # workspace
    D_prime: wp.array3d(dtype=float),  # (n, 3, 3) modified diagonal
    b_prime: wp.array2d(dtype=float),  # (n, 3) modified RHS
    # output
    x_blocks: wp.array2d(dtype=float),  # (n, 3) solution
):
    """
    Block Thomas algorithm for block-tridiagonal system.

    Forward sweep:
        D'[0] = D[0]
        b'[0] = b[0]
        for i = 1 to n-1:
            D'[i] = D[i] - L[i-1] @ inv(D'[i-1]) @ U[i-1]
            b'[i] = b[i] - L[i-1] @ inv(D'[i-1]) @ b'[i-1]

    Back substitution:
        x[n-1] = inv(D'[n-1]) @ b'[n-1]
        for i = n-2 down to 0:
            x[i] = inv(D'[i]) @ (b'[i] - U[i] @ x[i+1])
    """
    n = num_elements

    # Forward sweep
    # D'[0] = D[0], b'[0] = b[0]
    D0 = load_mat3(D_blocks, 0)
    store_mat3(D_prime, 0, D0)
    b0 = load_vec3(b_blocks, 0)
    store_vec3(b_prime, 0, b0)

    for i in range(1, n):
        # Load blocks
        D_i = load_mat3(D_blocks, i)
        L_im1 = load_mat3(L_blocks, i - 1)
        U_im1 = load_mat3(U_blocks, i - 1)
        D_prime_im1 = load_mat3(D_prime, i - 1)
        b_i = load_vec3(b_blocks, i)
        b_prime_im1 = load_vec3(b_prime, i - 1)

        # inv(D'[i-1])
        D_prime_im1_inv = mat3_inverse(D_prime_im1)

        # D'[i] = D[i] - L[i-1] @ inv(D'[i-1]) @ U[i-1]
        temp_mat = L_im1 * D_prime_im1_inv * U_im1
        D_prime_i = D_i - temp_mat
        store_mat3(D_prime, i, D_prime_i)

        # b'[i] = b[i] - L[i-1] @ inv(D'[i-1]) @ b'[i-1]
        temp_vec = L_im1 * D_prime_im1_inv * b_prime_im1
        b_prime_i = b_i - temp_vec
        store_vec3(b_prime, i, b_prime_i)

    # Back substitution
    # x[n-1] = inv(D'[n-1]) @ b'[n-1]
    D_prime_nm1 = load_mat3(D_prime, n - 1)
    D_prime_nm1_inv = mat3_inverse(D_prime_nm1)
    b_prime_nm1 = load_vec3(b_prime, n - 1)
    x_nm1 = D_prime_nm1_inv * b_prime_nm1
    store_vec3(x_blocks, n - 1, x_nm1)

    for i in range(n - 2, -1, -1):
        # x[i] = inv(D'[i]) @ (b'[i] - U[i] @ x[i+1])
        D_prime_i = load_mat3(D_prime, i)
        D_prime_i_inv = mat3_inverse(D_prime_i)
        U_i = load_mat3(U_blocks, i)
        b_prime_i = load_vec3(b_prime, i)
        x_ip1 = load_vec3(x_blocks, i + 1)

        rhs = b_prime_i - U_i * x_ip1
        x_i = D_prime_i_inv * rhs
        store_vec3(x_blocks, i, x_i)


@wp.kernel
def apply_block_corrections_kernel(
    node_q: wp.array(dtype=wp.vec3),
    node_inv_mass: wp.array(dtype=float),
    element_director: wp.array(dtype=wp.mat33),
    rest_length: wp.array(dtype=float),
    delta_lambda: wp.array2d(dtype=float),  # (n_elements, 3)
    num_elements: int,
    # output
    node_q_corrected: wp.array(dtype=wp.vec3),
):
    """
    Apply position corrections from constraint multipliers.

    delta_x_i = M^{-1} @ J^T @ delta_lambda

    For node i, contributions come from:
    - Element i-1 (if exists): node i is right node, J^T = +R/L
    - Element i (if exists): node i is left node, J^T = -R/L
    """
    tid = wp.tid()

    inv_mass = node_inv_mass[tid]
    pos = node_q[tid]

    if inv_mass == 0.0:
        node_q_corrected[tid] = pos
        return

    correction = wp.vec3(0.0, 0.0, 0.0)

    # Contribution from element tid-1 (this node is right node)
    if tid > 0 and tid - 1 < num_elements:
        R = element_director[tid - 1]
        L = rest_length[tid - 1]
        dl = wp.vec3(
            delta_lambda[tid - 1, 0],
            delta_lambda[tid - 1, 1],
            delta_lambda[tid - 1, 2],
        )
        # J^T @ dl = +R @ dl / L (right node has positive Jacobian)
        local_correction = R * dl / L
        correction = correction + local_correction * inv_mass

    # Contribution from element tid (this node is left node)
    if tid < num_elements:
        R = element_director[tid]
        L = rest_length[tid]
        dl = wp.vec3(
            delta_lambda[tid, 0],
            delta_lambda[tid, 1],
            delta_lambda[tid, 2],
        )
        # J^T @ dl = -R @ dl / L (left node has negative Jacobian)
        local_correction = -R * dl / L
        correction = correction + local_correction * inv_mass

    node_q_corrected[tid] = pos + correction


@wp.kernel
def update_velocities_kernel(
    node_q_old: wp.array(dtype=wp.vec3),
    node_q_new: wp.array(dtype=wp.vec3),
    node_inv_mass: wp.array(dtype=float),
    dt: float,
    # output
    node_qd: wp.array(dtype=wp.vec3),
):
    """Update velocities from position change."""
    tid = wp.tid()

    if node_inv_mass[tid] == 0.0:
        node_qd[tid] = wp.vec3(0.0, 0.0, 0.0)
        return

    delta_x = node_q_new[tid] - node_q_old[tid]
    node_qd[tid] = delta_x / dt


@wp.kernel
def solve_ground_collision_kernel(
    node_q: wp.array(dtype=wp.vec3),
    node_inv_mass: wp.array(dtype=float),
    ground_level: float,
    radius: float,
    # output
    node_q_out: wp.array(dtype=wp.vec3),
):
    """Ground collision constraint."""
    tid = wp.tid()

    inv_mass = node_inv_mass[tid]
    pos = node_q[tid]

    if inv_mass == 0.0:
        node_q_out[tid] = pos
        return

    min_z = ground_level + radius
    if pos[2] < min_z:
        node_q_out[tid] = wp.vec3(pos[0], pos[1], min_z)
    else:
        node_q_out[tid] = pos


class Example:
    def __init__(self, viewer, args=None):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.constraint_iterations = 3

        self.viewer = viewer
        self.args = args

        # Rod parameters
        self.num_elements = NUM_ELEMENTS
        self.num_nodes = NUM_NODES
        element_length = 0.1
        node_mass = 0.05
        self.node_radius = 0.02
        start_height = 4.0

        # Stiffness [stretch, shear_1, shear_2]
        self.stiffness = wp.vec3(1.0e4, 1.0e3, 1.0e3)

        # Compliance
        self.compliance = 1.0e-6
        self.compliance_factor = self.compliance / (self.sim_dt * self.sim_dt)

        self.gravity = wp.vec3(0.0, 0.0, -9.81)

        # Build particle model for visualization
        builder = newton.ModelBuilder()
        builder.add_ground_plane()

        for i in range(self.num_nodes):
            mass = 0.0 if i == 0 else node_mass
            builder.add_particle(
                pos=(i * element_length, 0.0, start_height),
                vel=(0.0, 0.0, 0.0),
                mass=mass,
                radius=self.node_radius,
            )

        self.model = builder.finalize()
        device = self.model.device

        # State
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        # Collision pipeline
        self.collision_pipeline = newton.examples.create_collision_pipeline(self.model, self.args)
        self.contacts = self.model.collide(self.state_0, collision_pipeline=self.collision_pipeline)

        # Inverse mass array
        inv_mass_np = [0.0] + [1.0 / node_mass] * (self.num_nodes - 1)
        self.node_inv_mass = wp.array(inv_mass_np, dtype=float, device=device)

        # Rest lengths
        rest_length_np = [element_length] * self.num_elements
        self.rest_length = wp.array(rest_length_np, dtype=float, device=device)

        # Element data
        self.element_strain = wp.zeros(self.num_elements, dtype=wp.vec3, device=device)
        self.element_director = wp.zeros(self.num_elements, dtype=wp.mat33, device=device)

        # Block-tridiagonal system storage
        self.D_blocks = wp.zeros((self.num_elements, BLOCK, BLOCK), dtype=float, device=device)
        self.L_blocks = wp.zeros((self.num_elements - 1, BLOCK, BLOCK), dtype=float, device=device)
        self.U_blocks = wp.zeros((self.num_elements - 1, BLOCK, BLOCK), dtype=float, device=device)
        self.b_blocks = wp.zeros((self.num_elements, BLOCK), dtype=float, device=device)

        # Workspace for Thomas algorithm
        self.D_prime = wp.zeros((self.num_elements, BLOCK, BLOCK), dtype=float, device=device)
        self.b_prime = wp.zeros((self.num_elements, BLOCK), dtype=float, device=device)

        # Solution
        self.delta_lambda = wp.zeros((self.num_elements, BLOCK), dtype=float, device=device)

        # Temp buffers
        self.node_q_pred = wp.zeros(self.num_nodes, dtype=wp.vec3, device=device)
        self.node_q_temp = wp.zeros(self.num_nodes, dtype=wp.vec3, device=device)

        self.viewer.set_model(self.model)
        self.viewer.show_particles = True

    def simulate(self):
        for _ in range(self.sim_substeps):
            wp.copy(self.node_q_temp, self.state_0.particle_q)

            # Integrate
            wp.launch(
                kernel=integrate_nodes_kernel,
                dim=self.num_nodes,
                inputs=[
                    self.state_0.particle_q,
                    self.state_0.particle_qd,
                    self.node_inv_mass,
                    self.gravity,
                    self.sim_dt,
                ],
                outputs=[self.node_q_pred, self.state_1.particle_qd],
                device=self.model.device,
            )
            wp.copy(self.state_1.particle_q, self.node_q_pred)

            # Constraint iterations
            for _ in range(self.constraint_iterations):
                # Compute element frames and strains
                wp.launch(
                    kernel=compute_element_frames_kernel,
                    dim=self.num_elements,
                    inputs=[
                        self.state_1.particle_q,
                        self.rest_length,
                        self.num_elements,
                    ],
                    outputs=[self.element_strain, self.element_director],
                    device=self.model.device,
                )

                # Assemble block-tridiagonal system
                wp.launch(
                    kernel=assemble_block_system_kernel,
                    dim=1,
                    inputs=[
                        self.node_inv_mass,
                        self.element_strain,
                        self.element_director,
                        self.rest_length,
                        self.stiffness,
                        self.compliance_factor,
                        self.num_elements,
                    ],
                    outputs=[self.D_blocks, self.L_blocks, self.U_blocks, self.b_blocks],
                    device=self.model.device,
                )

                # Solve using block Thomas
                wp.launch(
                    kernel=block_thomas_solve_kernel,
                    dim=1,
                    inputs=[
                        self.D_blocks,
                        self.L_blocks,
                        self.U_blocks,
                        self.b_blocks,
                        self.num_elements,
                        self.D_prime,
                        self.b_prime,
                    ],
                    outputs=[self.delta_lambda],
                    device=self.model.device,
                )

                # Apply corrections
                wp.launch(
                    kernel=apply_block_corrections_kernel,
                    dim=self.num_nodes,
                    inputs=[
                        self.state_1.particle_q,
                        self.node_inv_mass,
                        self.element_director,
                        self.rest_length,
                        self.delta_lambda,
                        self.num_elements,
                    ],
                    outputs=[self.node_q_pred],
                    device=self.model.device,
                )
                wp.copy(self.state_1.particle_q, self.node_q_pred)

            # Ground collision
            wp.launch(
                kernel=solve_ground_collision_kernel,
                dim=self.num_nodes,
                inputs=[
                    self.state_1.particle_q,
                    self.node_inv_mass,
                    0.0,
                    self.node_radius,
                ],
                outputs=[self.node_q_pred],
                device=self.model.device,
            )
            wp.copy(self.state_1.particle_q, self.node_q_pred)

            # Update velocities
            wp.launch(
                kernel=update_velocities_kernel,
                dim=self.num_nodes,
                inputs=[
                    self.node_q_temp,
                    self.state_1.particle_q,
                    self.node_inv_mass,
                    self.sim_dt,
                ],
                outputs=[self.state_1.particle_qd],
                device=self.model.device,
            )

            self.state_0, self.state_1 = self.state_1, self.state_0
            self.contacts = self.model.collide(self.state_0, collision_pipeline=self.collision_pipeline)

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt

    def test_final(self):
        newton.examples.test_particle_state(
            self.state_0,
            "anchor node is stationary",
            lambda q, qd: wp.length(qd) < 1e-6,
            indices=[0],
        )

        newton.examples.test_particle_state(
            self.state_0,
            "nodes are above ground",
            lambda q, qd: q[2] >= -0.01,
        )

        p_lower = wp.vec3(-2.0, -4.0, -0.1)
        p_upper = wp.vec3(6.0, 4.0, 6.0)
        newton.examples.test_particle_state(
            self.state_0,
            "nodes within bounds",
            lambda q, qd: newton.utils.vec_inside_limits(q, p_lower, p_upper),
        )

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()


if __name__ == "__main__":
    viewer, args = newton.examples.init()

    if isinstance(viewer, newton.viewer.ViewerGL):
        viewer.show_particles = True

    example = Example(viewer, args)

    newton.examples.run(example, args)
