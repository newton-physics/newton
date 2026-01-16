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
# Example Global Cosserat Rod
#
# Demonstrates a unified global matrix-based Position Based Dynamics approach
# for Cosserat rod constraints (stretch/shear + bend/twist) using Warp's tile
# Cholesky decomposition on GPU.
#
# Mathematical formulation:
#   - Stretch constraint k: C_stretch[k] = |p_{k+1} - p_k| - L_k
#   - Bend constraint k: C_bend[k] = angular deviation between d3(q_k) and d3(q_{k+1})
#   - System matrix: A = J M^{-1} J^T + alpha/dt^2 (block-structured SPD)
#   - Solve: A * delta_lambda = -C
#   - Position correction: delta_p = M^{-1}_pos J_stretch^T delta_lambda_stretch
#   - Quaternion correction: delta_q = M^{-1}_quat J_bend^T delta_lambda_bend
#
# The system matrix has block structure:
#   - Upper-left: stretch-stretch coupling (tridiagonal from positions)
#   - Lower-right: bend-bend coupling (tridiagonal from quaternions)
#   - Off-diagonal: shear coupling (stretch constraint depends on quaternion direction)
#
# Command: uv run -m newton.examples basic_global_cosserat_rod
#
###########################################################################

import math

import warp as wp

import newton
import newton.examples

# Warp tile configuration
BLOCK_DIM = 128
TILE = 32  # 32x32 tile for Cholesky

# Rod configuration - sized to fit in single tile
NUM_PARTICLES = 17  # Gives 31 constraints (16 stretch + 15 bend), fits in 32x32
NUM_STRETCH = NUM_PARTICLES - 1  # 16 stretch constraints
NUM_BEND = NUM_PARTICLES - 2  # 15 bend constraints
NUM_CONSTRAINTS = NUM_STRETCH + NUM_BEND  # 31 total constraints


@wp.func
def quat_rotate_e3(q: wp.quat) -> wp.vec3:
    """Compute the third director d3 = q * e3 * conjugate(q) where e3 = (0,0,1).

    This is an optimized computation of rotating the z-axis by quaternion q.
    """
    x, y, z, w = q[0], q[1], q[2], q[3]
    d3_x = 2.0 * (x * z + w * y)
    d3_y = 2.0 * (y * z - w * x)
    d3_z = w * w - x * x - y * y + z * z
    return wp.vec3(d3_x, d3_y, d3_z)


@wp.kernel
def integrate_particles_kernel(
    particle_q: wp.array(dtype=wp.vec3),
    particle_qd: wp.array(dtype=wp.vec3),
    particle_inv_mass: wp.array(dtype=float),
    gravity: wp.vec3,
    dt: float,
    # outputs
    particle_q_predicted: wp.array(dtype=wp.vec3),
    particle_qd_new: wp.array(dtype=wp.vec3),
):
    """Semi-implicit Euler integration step for particles."""
    tid = wp.tid()
    inv_mass = particle_inv_mass[tid]

    if inv_mass == 0.0:
        # Kinematic particle - don't move
        particle_q_predicted[tid] = particle_q[tid]
        particle_qd_new[tid] = particle_qd[tid]
        return

    # v_new = v + g * dt
    v_new = particle_qd[tid] + gravity * dt
    # x_predicted = x + v_new * dt
    x_predicted = particle_q[tid] + v_new * dt

    particle_q_predicted[tid] = x_predicted
    particle_qd_new[tid] = v_new


@wp.kernel
def compute_constraint_data_kernel(
    particle_q: wp.array(dtype=wp.vec3),
    edge_q: wp.array(dtype=wp.quat),
    rest_length: wp.array(dtype=float),
    rest_angle: wp.array(dtype=float),
    num_stretch: int,
    num_bend: int,
    # outputs
    stretch_violation: wp.array(dtype=float),
    stretch_direction: wp.array(dtype=wp.vec3),
    bend_violation: wp.array(dtype=float),
    director_d3: wp.array(dtype=wp.vec3),
):
    """Compute constraint violations and auxiliary data for all constraints."""
    # Single thread computes all constraint data (small problem size)

    # Compute stretch constraints and directors
    for k in range(num_stretch):
        p0 = particle_q[k]
        p1 = particle_q[k + 1]
        q = edge_q[k]

        # Edge direction and length
        diff = p1 - p0
        length = wp.length(diff)
        L = rest_length[k]

        # Stretch violation: C = |x1 - x0| - L
        stretch_violation[k] = length - L

        # Direction vector (normalized)
        if length > 1.0e-8:
            stretch_direction[k] = diff / length
        else:
            stretch_direction[k] = wp.vec3(1.0, 0.0, 0.0)

        # Director d3 for this edge
        director_d3[k] = quat_rotate_e3(q)

    # Compute bend constraints
    for k in range(num_bend):
        # Bend constraint between edges k and k+1
        d3_k = director_d3[k]
        d3_k1 = director_d3[k + 1]

        # Angular deviation: cos(theta) = d3_k . d3_k1
        cos_theta = wp.dot(d3_k, d3_k1)
        cos_theta = wp.clamp(cos_theta, -1.0, 1.0)

        # Bend violation: theta - rest_angle
        theta = wp.acos(cos_theta)
        bend_violation[k] = theta - rest_angle[k]


@wp.kernel
def assemble_global_system_kernel(
    particle_inv_mass: wp.array(dtype=float),
    edge_inv_mass: wp.array(dtype=float),
    stretch_direction: wp.array(dtype=wp.vec3),
    stretch_violation: wp.array(dtype=float),
    director_d3: wp.array(dtype=wp.vec3),
    bend_violation: wp.array(dtype=float),
    compliance_stretch: float,
    compliance_bend: float,
    num_stretch: int,
    num_bend: int,
    # outputs
    system_matrix: wp.array2d(dtype=float),
    system_rhs: wp.array1d(dtype=float),
):
    """
    Assemble the unified global system matrix A and RHS vector b.

    System matrix structure:
    - Upper-left (num_stretch x num_stretch): stretch-stretch coupling (tridiagonal)
    - Lower-right (num_bend x num_bend): bend-bend coupling (tridiagonal)
    - Off-diagonal blocks are zero in this simplified formulation

    For stretch constraints (like distance constraints):
    - A[i,i] = w_i + w_{i+1} + compliance
    - A[i,i+1] = A[i+1,i] = -w_{i+1} * (n_i . n_{i+1})

    For bend constraints (angular between quaternions):
    - A[i,i] = inv_mass_q_i + inv_mass_q_{i+1} + compliance
    - A[i,i+1] = A[i+1,i] = -inv_mass_q_{i+1} * coupling_factor
    """
    # Initialize matrix to zero
    for i in range(TILE):
        system_rhs[i] = 0.0
        for j in range(TILE):
            system_matrix[i, j] = 0.0

    # ========== Stretch-Stretch Block (upper-left) ==========
    for k in range(num_stretch):
        # Particles involved: k and k+1
        w0 = particle_inv_mass[k]
        w1 = particle_inv_mass[k + 1]

        # Diagonal: A[k,k] = w0 + w1 + compliance
        diag = w0 + w1 + compliance_stretch
        system_matrix[k, k] = diag

        # RHS: -C_stretch[k]
        system_rhs[k] = -stretch_violation[k]

        # Off-diagonal coupling with next stretch constraint
        if k + 1 < num_stretch:
            n_k = stretch_direction[k]
            n_k1 = stretch_direction[k + 1]

            # Coupling through shared particle k+1
            coupling = -particle_inv_mass[k + 1] * wp.dot(n_k, n_k1)
            system_matrix[k, k + 1] = coupling
            system_matrix[k + 1, k] = coupling

    # ========== Bend-Bend Block (lower-right) ==========
    bend_offset = num_stretch  # Bend constraints start after stretch

    for k in range(num_bend):
        row = bend_offset + k

        # Quaternions involved: k and k+1
        w_q0 = edge_inv_mass[k]
        w_q1 = edge_inv_mass[k + 1]

        # Diagonal: A[row,row] = w_q0 + w_q1 + compliance
        diag = w_q0 + w_q1 + compliance_bend
        system_matrix[row, row] = diag

        # RHS: -C_bend[k]
        system_rhs[row] = -bend_violation[k]

        # Off-diagonal coupling with next bend constraint
        if k + 1 < num_bend:
            # Coupling through shared quaternion k+1
            d3_k = director_d3[k]
            d3_k1 = director_d3[k + 1]
            d3_k2 = director_d3[k + 2]

            # Simplified coupling factor based on director alignment
            coupling_factor = wp.dot(d3_k1, d3_k2) * 0.5 + wp.dot(d3_k, d3_k1) * 0.5
            coupling = -edge_inv_mass[k + 1] * coupling_factor
            system_matrix[row, row + 1] = coupling
            system_matrix[row + 1, row] = coupling

    # Pad unused rows/columns with identity to keep matrix SPD
    num_total = num_stretch + num_bend
    for i in range(num_total, TILE):
        system_matrix[i, i] = 1.0


@wp.kernel
def cholesky_solve_kernel(
    A: wp.array2d(dtype=float),
    b: wp.array1d(dtype=float),
    # output
    x: wp.array1d(dtype=float),
):
    """Solve Ax = b using tile Cholesky decomposition."""
    a_tile = wp.tile_load(A, shape=(TILE, TILE))
    b_tile = wp.tile_load(b, shape=TILE)

    # Cholesky factorization: A = L L^T
    L = wp.tile_cholesky(a_tile)

    # Solve: L L^T x = b
    x_tile = wp.tile_cholesky_solve(L, b_tile)

    wp.tile_store(x, x_tile)


@wp.kernel
def apply_position_corrections_kernel(
    particle_q: wp.array(dtype=wp.vec3),
    particle_inv_mass: wp.array(dtype=float),
    stretch_direction: wp.array(dtype=wp.vec3),
    delta_lambda: wp.array1d(dtype=float),
    num_stretch: int,
    num_particles: int,
    # output
    particle_q_corrected: wp.array(dtype=wp.vec3),
):
    """
    Apply position corrections from stretch constraints.

    delta_p_i = sum over stretch constraints k involving particle i:
                inv_mass[i] * grad_C_k[i] * delta_lambda[k]

    For stretch constraint k between particles k and k+1:
    - grad_C_k[k] = -n_k
    - grad_C_k[k+1] = +n_k
    """
    tid = wp.tid()

    inv_mass = particle_inv_mass[tid]
    pos = particle_q[tid]

    if inv_mass == 0.0:
        # Kinematic particle - no correction
        particle_q_corrected[tid] = pos
        return

    correction = wp.vec3(0.0, 0.0, 0.0)

    # Contribution from constraint tid-1 (if exists): this particle is the "right" particle
    if tid > 0 and tid - 1 < num_stretch:
        n_prev = stretch_direction[tid - 1]
        dl_prev = delta_lambda[tid - 1]
        # grad_C_{tid-1}[tid] = +n_{tid-1}
        correction = correction + n_prev * dl_prev * inv_mass

    # Contribution from constraint tid (if exists): this particle is the "left" particle
    if tid < num_stretch:
        n_curr = stretch_direction[tid]
        dl_curr = delta_lambda[tid]
        # grad_C_tid[tid] = -n_tid
        correction = correction - n_curr * dl_curr * inv_mass

    particle_q_corrected[tid] = pos + correction


@wp.kernel
def apply_quaternion_corrections_kernel(
    edge_q: wp.array(dtype=wp.quat),
    edge_inv_mass: wp.array(dtype=float),
    director_d3: wp.array(dtype=wp.vec3),
    delta_lambda: wp.array1d(dtype=float),
    num_stretch: int,
    num_bend: int,
    # output
    edge_q_corrected: wp.array(dtype=wp.quat),
):
    """
    Apply quaternion corrections from bend constraints.

    The bend constraint measures angular deviation between adjacent directors.
    Corrections rotate quaternions to reduce this deviation.
    """
    tid = wp.tid()

    inv_mass = edge_inv_mass[tid]
    q = edge_q[tid]

    if inv_mass == 0.0:
        edge_q_corrected[tid] = q
        return

    # Accumulate angular correction as axis-angle
    correction = wp.vec3(0.0, 0.0, 0.0)
    bend_offset = num_stretch

    # Contribution from bend constraint tid-1 (if exists): this is the "right" quaternion
    if tid > 0 and tid - 1 < num_bend:
        dl = delta_lambda[bend_offset + tid - 1]
        d3_prev = director_d3[tid - 1]
        d3_curr = director_d3[tid]

        # Rotation axis is perpendicular to both directors
        axis = wp.cross(d3_prev, d3_curr)
        axis_len = wp.length(axis)
        if axis_len > 1.0e-8:
            axis = axis / axis_len
            # This quaternion needs to rotate toward the previous one
            correction = correction + axis * dl * inv_mass

    # Contribution from bend constraint tid (if exists): this is the "left" quaternion
    if tid < num_bend:
        dl = delta_lambda[bend_offset + tid]
        d3_curr = director_d3[tid]
        d3_next = director_d3[tid + 1]

        # Rotation axis is perpendicular to both directors
        axis = wp.cross(d3_curr, d3_next)
        axis_len = wp.length(axis)
        if axis_len > 1.0e-8:
            axis = axis / axis_len
            # This quaternion needs to rotate toward the next one
            correction = correction - axis * dl * inv_mass

    # Apply correction as small rotation
    corr_len = wp.length(correction)
    if corr_len > 1.0e-8:
        # Convert axis-angle to quaternion delta
        half_angle = corr_len * 0.5
        s = wp.sin(half_angle) / corr_len
        c = wp.cos(half_angle)
        dq = wp.quat(correction[0] * s, correction[1] * s, correction[2] * s, c)

        # Apply rotation: q_new = dq * q
        q_new = wp.mul(dq, q)
        q_new = wp.normalize(q_new)
        edge_q_corrected[tid] = q_new
    else:
        edge_q_corrected[tid] = q


@wp.kernel
def update_velocities_kernel(
    particle_q_old: wp.array(dtype=wp.vec3),
    particle_q_new: wp.array(dtype=wp.vec3),
    particle_inv_mass: wp.array(dtype=float),
    dt: float,
    # output
    particle_qd: wp.array(dtype=wp.vec3),
):
    """Update velocities from position change: v = (x_new - x_old) / dt"""
    tid = wp.tid()

    if particle_inv_mass[tid] == 0.0:
        particle_qd[tid] = wp.vec3(0.0, 0.0, 0.0)
        return

    delta_x = particle_q_new[tid] - particle_q_old[tid]
    particle_qd[tid] = delta_x / dt


@wp.kernel
def align_quaternions_to_edges_kernel(
    particle_q: wp.array(dtype=wp.vec3),
    edge_q: wp.array(dtype=wp.quat),
    edge_inv_mass: wp.array(dtype=float),
    shear_stiffness: float,
    num_edges: int,
    # output
    edge_q_out: wp.array(dtype=wp.quat),
):
    """
    Align quaternion d3 directors with actual edge directions.
    
    This implements the shear constraint: d3(q) should align with (p1-p0)/|p1-p0|.
    We compute the rotation from current d3 to target direction and apply it.
    """
    tid = wp.tid()
    
    if tid >= num_edges:
        return
    
    inv_mass = edge_inv_mass[tid]
    q = edge_q[tid]
    
    if inv_mass == 0.0:
        edge_q_out[tid] = q
        return
    
    # Get actual edge direction from particles
    p0 = particle_q[tid]
    p1 = particle_q[tid + 1]
    edge_dir = p1 - p0
    edge_len = wp.length(edge_dir)
    
    if edge_len < 1.0e-8:
        edge_q_out[tid] = q
        return
    
    target_dir = edge_dir / edge_len
    
    # Current d3 director
    current_d3 = quat_rotate_e3(q)
    
    # Compute rotation from current_d3 to target_dir
    # Using axis-angle: axis = cross(current, target), angle = acos(dot)
    dot_val = wp.dot(current_d3, target_dir)
    dot_val = wp.clamp(dot_val, -1.0, 1.0)
    
    # If already aligned, skip
    if dot_val > 0.9999:
        edge_q_out[tid] = q
        return
    
    # Compute rotation axis
    axis = wp.cross(current_d3, target_dir)
    axis_len = wp.length(axis)
    
    if axis_len < 1.0e-8:
        # Vectors are anti-parallel, pick arbitrary perpendicular axis
        if wp.abs(current_d3[0]) < 0.9:
            axis = wp.cross(current_d3, wp.vec3(1.0, 0.0, 0.0))
        else:
            axis = wp.cross(current_d3, wp.vec3(0.0, 1.0, 0.0))
        axis_len = wp.length(axis)
    
    axis = axis / axis_len
    
    # Rotation angle (scaled by stiffness for compliance)
    angle = wp.acos(dot_val) * shear_stiffness
    
    # Convert to quaternion rotation
    half_angle = angle * 0.5
    s = wp.sin(half_angle)
    c = wp.cos(half_angle)
    dq = wp.quat(axis[0] * s, axis[1] * s, axis[2] * s, c)
    
    # Apply rotation: q_new = dq * q
    q_new = wp.mul(dq, q)
    q_new = wp.normalize(q_new)
    
    edge_q_out[tid] = q_new


@wp.kernel
def compute_director_lines_kernel(
    particle_q: wp.array(dtype=wp.vec3),
    edge_q: wp.array(dtype=wp.quat),
    num_edges: int,
    axis_length: float,
    # outputs: 3 axes per edge (d1=red, d2=green, d3=blue)
    line_starts: wp.array(dtype=wp.vec3),
    line_ends: wp.array(dtype=wp.vec3),
    line_colors: wp.array(dtype=wp.vec3),
):
    """
    Compute line segments for visualizing material frames.
    Each edge has 3 axes (d1, d2, d3) drawn from its midpoint.
    """
    tid = wp.tid()
    edge_idx = tid // 3
    axis_idx = tid % 3  # 0=d1(red), 1=d2(green), 2=d3(blue)

    if edge_idx >= num_edges:
        return

    # Compute edge midpoint
    p0 = particle_q[edge_idx]
    p1 = particle_q[edge_idx + 1]
    midpoint = (p0 + p1) * 0.5

    q = edge_q[edge_idx]

    # Compute the director based on axis_idx
    if axis_idx == 0:
        # d1 = q * e1 * conj(q) where e1 = (1,0,0)
        x, y, z, w = q[0], q[1], q[2], q[3]
        d1_x = w * w + x * x - y * y - z * z
        d1_y = 2.0 * (x * y + w * z)
        d1_z = 2.0 * (x * z - w * y)
        director = wp.vec3(d1_x, d1_y, d1_z)
        color = wp.vec3(1.0, 0.0, 0.0)  # Red
    elif axis_idx == 1:
        # d2 = q * e2 * conj(q) where e2 = (0,1,0)
        x, y, z, w = q[0], q[1], q[2], q[3]
        d2_x = 2.0 * (x * y - w * z)
        d2_y = w * w - x * x + y * y - z * z
        d2_z = 2.0 * (y * z + w * x)
        director = wp.vec3(d2_x, d2_y, d2_z)
        color = wp.vec3(0.0, 1.0, 0.0)  # Green
    else:
        # d3 = q * e3 * conj(q) where e3 = (0,0,1)
        x, y, z, w = q[0], q[1], q[2], q[3]
        d3_x = 2.0 * (x * z + w * y)
        d3_y = 2.0 * (y * z - w * x)
        d3_z = w * w - x * x - y * y + z * z
        director = wp.vec3(d3_x, d3_y, d3_z)
        color = wp.vec3(0.0, 0.0, 1.0)  # Blue

    line_starts[tid] = midpoint
    line_ends[tid] = midpoint + director * axis_length
    line_colors[tid] = color


@wp.kernel
def solve_ground_collision_kernel(
    particle_q: wp.array(dtype=wp.vec3),
    particle_inv_mass: wp.array(dtype=float),
    particle_radius: wp.array(dtype=float),
    ground_level: float,
    # output
    particle_q_out: wp.array(dtype=wp.vec3),
):
    """Simple ground plane collision constraint."""
    tid = wp.tid()

    inv_mass = particle_inv_mass[tid]
    pos = particle_q[tid]

    if inv_mass == 0.0:
        particle_q_out[tid] = pos
        return

    radius = particle_radius[tid]
    min_z = ground_level + radius
    penetration = min_z - pos[2]

    if penetration > 0.0:
        particle_q_out[tid] = wp.vec3(pos[0], pos[1], min_z)
    else:
        particle_q_out[tid] = pos


class Example:
    def __init__(self, viewer, args=None):
        # Simulation parameters
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 8
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.constraint_iterations = 2  # Number of global solve iterations per substep

        self.viewer = viewer
        self.args = args

        # Rod parameters
        self.num_particles = NUM_PARTICLES
        self.num_stretch = NUM_STRETCH
        self.num_bend = NUM_BEND
        self.num_constraints = NUM_CONSTRAINTS

        particle_spacing = 0.1  # rest length between particles
        particle_mass = 0.1
        particle_radius = 0.02
        edge_mass = 0.01  # mass associated with each quaternion
        start_height = 3.0

        # Compliance: alpha/dt^2 (lower = stiffer)
        self.stretch_stiffness = 1.0
        self.bend_stiffness = 1.0
        self.shear_stiffness = 1.0  # How strongly quaternions track edge directions
        self.compliance_stretch = 1.0e-6
        self.compliance_bend = 1.0e-5
        self.compliance_factor_stretch = self.compliance_stretch / (self.sim_dt * self.sim_dt)
        self.compliance_factor_bend = self.compliance_bend / (self.sim_dt * self.sim_dt)

        self.gravity = wp.vec3(0.0, 0.0, -9.81)

        # Build the model
        builder = newton.ModelBuilder()
        builder.add_ground_plane()

        # Create particles: first one is fixed (kinematic)
        for i in range(self.num_particles):
            mass = 0.0 if i == 0 else particle_mass
            builder.add_particle(
                pos=(i * particle_spacing, 0.0, start_height),
                vel=(0.0, 0.0, 0.0),
                mass=mass,
                radius=particle_radius,
            )

        self.model = builder.finalize()

        # Soft contact parameters for particle-ground collision
        self.model.soft_contact_ke = 1.0e3
        self.model.soft_contact_kd = 1.0e1
        self.model.soft_contact_mu = 0.5

        # State buffers
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        # Collision pipeline
        self.collision_pipeline = newton.examples.create_collision_pipeline(self.model, self.args)
        self.contacts = self.model.collide(self.state_0, collision_pipeline=self.collision_pipeline)

        device = self.model.device

        # Particle inverse mass array
        inv_mass_np = [0.0] + [1.0 / particle_mass] * (self.num_particles - 1)
        self.particle_inv_mass = wp.array(inv_mass_np, dtype=float, device=device)

        # Edge quaternions (initialized to rotate z-axis to x-axis for horizontal rod)
        angle = math.pi / 2.0
        q_init = wp.quat(0.0, math.sin(angle / 2.0), 0.0, math.cos(angle / 2.0))
        edge_q_init = [q_init] * self.num_stretch
        self.edge_q = wp.array(edge_q_init, dtype=wp.quat, device=device)
        self.edge_q_new = wp.array(edge_q_init, dtype=wp.quat, device=device)

        # Edge inverse masses
        edge_inv_mass_np = [1.0 / edge_mass] * self.num_stretch
        self.edge_inv_mass = wp.array(edge_inv_mass_np, dtype=float, device=device)

        # Rest lengths for stretch constraints
        rest_length_np = [particle_spacing] * self.num_stretch
        self.rest_length = wp.array(rest_length_np, dtype=float, device=device)

        # Rest angles for bend constraints (straight rod = 0 angle)
        rest_angle_np = [0.0] * self.num_bend
        self.rest_angle = wp.array(rest_angle_np, dtype=float, device=device)

        # Buffers for constraint data
        self.stretch_violation = wp.zeros(self.num_stretch, dtype=float, device=device)
        self.stretch_direction = wp.zeros(self.num_stretch, dtype=wp.vec3, device=device)
        self.bend_violation = wp.zeros(self.num_bend, dtype=float, device=device)
        self.director_d3 = wp.zeros(self.num_stretch, dtype=wp.vec3, device=device)

        # Buffers for global system (32x32 tile)
        self.system_matrix = wp.zeros((TILE, TILE), dtype=float, device=device)
        self.system_rhs = wp.zeros(TILE, dtype=float, device=device)
        self.delta_lambda = wp.zeros(TILE, dtype=float, device=device)

        # Temporary buffer for predicted positions
        self.particle_q_predicted = wp.zeros(self.num_particles, dtype=wp.vec3, device=device)
        self.particle_q_temp = wp.zeros(self.num_particles, dtype=wp.vec3, device=device)

        self.viewer.set_model(self.model)
        self.viewer.show_particles = True

        # Director visualization buffers (3 axes per edge)
        num_director_lines = self.num_stretch * 3
        self.director_line_starts = wp.zeros(num_director_lines, dtype=wp.vec3, device=device)
        self.director_line_ends = wp.zeros(num_director_lines, dtype=wp.vec3, device=device)
        self.director_line_colors = wp.zeros(num_director_lines, dtype=wp.vec3, device=device)
        self.show_directors = True
        self.director_scale = 0.05  # Length of director axes

        self.graph = None

    def simulate(self):
        for _ in range(self.sim_substeps):
            # Store old positions for velocity update
            wp.copy(self.particle_q_temp, self.state_0.particle_q)

            # uddata parameters:
            self.compliance_stretch = (1.0 - self.stretch_stiffness) * 1e-2
            self.compliance_factor_stretch = self.compliance_stretch / (self.sim_dt * self.sim_dt)
            self.compliance_bend = (1.0 - self.bend_stiffness)
            self.compliance_factor_bend = self.compliance_bend / (self.sim_dt * self.sim_dt)



            # Step 1: Integrate (predict positions)
            wp.launch(
                kernel=integrate_particles_kernel,
                dim=self.num_particles,
                inputs=[
                    self.state_0.particle_q,
                    self.state_0.particle_qd,
                    self.particle_inv_mass,
                    self.gravity,
                    self.sim_dt,
                ],
                outputs=[self.particle_q_predicted, self.state_1.particle_qd],
                device=self.model.device,
            )

            # Copy predicted to state_1 for constraint solving
            wp.copy(self.state_1.particle_q, self.particle_q_predicted)

            # Step 2: Iterative constraint solving using global Cholesky
            for _ in range(self.constraint_iterations):
                # Compute constraint violations and auxiliary data
                wp.launch(
                    kernel=compute_constraint_data_kernel,
                    dim=1,
                    inputs=[
                        self.state_1.particle_q,
                        self.edge_q,
                        self.rest_length,
                        self.rest_angle,
                        self.num_stretch,
                        self.num_bend,
                    ],
                    outputs=[
                        self.stretch_violation,
                        self.stretch_direction,
                        self.bend_violation,
                        self.director_d3,
                    ],
                    device=self.model.device,
                )

                # Assemble global system matrix and RHS
                wp.launch(
                    kernel=assemble_global_system_kernel,
                    dim=1,
                    inputs=[
                        self.particle_inv_mass,
                        self.edge_inv_mass,
                        self.stretch_direction,
                        self.stretch_violation,
                        self.director_d3,
                        self.bend_violation,
                        self.compliance_factor_stretch,
                        self.compliance_factor_bend,
                        self.num_stretch,
                        self.num_bend,
                    ],
                    outputs=[self.system_matrix, self.system_rhs],
                    device=self.model.device,
                )

                # Solve using tile Cholesky
                wp.launch_tiled(
                    kernel=cholesky_solve_kernel,
                    dim=[1, 1],
                    inputs=[self.system_matrix, self.system_rhs],
                    outputs=[self.delta_lambda],
                    block_dim=BLOCK_DIM,
                    device=self.model.device,
                )

                # Apply position corrections from stretch constraints
                wp.launch(
                    kernel=apply_position_corrections_kernel,
                    dim=self.num_particles,
                    inputs=[
                        self.state_1.particle_q,
                        self.particle_inv_mass,
                        self.stretch_direction,
                        self.delta_lambda,
                        self.num_stretch,
                        self.num_particles,
                    ],
                    outputs=[self.particle_q_predicted],
                    device=self.model.device,
                )

                # Copy corrected positions back
                wp.copy(self.state_1.particle_q, self.particle_q_predicted)

                # Align quaternions to actual edge directions (shear constraint)
                wp.launch(
                    kernel=align_quaternions_to_edges_kernel,
                    dim=self.num_stretch,
                    inputs=[
                        self.state_1.particle_q,
                        self.edge_q,
                        self.edge_inv_mass,
                        self.shear_stiffness,
                        self.num_stretch,
                    ],
                    outputs=[self.edge_q_new],
                    device=self.model.device,
                )
                
                # Swap quaternion buffers after shear alignment
                self.edge_q, self.edge_q_new = self.edge_q_new, self.edge_q

                # Apply quaternion corrections from bend constraints
                wp.launch(
                    kernel=apply_quaternion_corrections_kernel,
                    dim=self.num_stretch,
                    inputs=[
                        self.edge_q,
                        self.edge_inv_mass,
                        self.director_d3,
                        self.delta_lambda,
                        self.num_stretch,
                        self.num_bend,
                    ],
                    outputs=[self.edge_q_new],
                    device=self.model.device,
                )

                # Swap quaternion buffers
                self.edge_q, self.edge_q_new = self.edge_q_new, self.edge_q

            # Step 3: Ground collision
            wp.launch(
                kernel=solve_ground_collision_kernel,
                dim=self.num_particles,
                inputs=[
                    self.state_1.particle_q,
                    self.particle_inv_mass,
                    self.model.particle_radius,
                    0.0,
                ],
                outputs=[self.particle_q_predicted],
                device=self.model.device,
            )
            wp.copy(self.state_1.particle_q, self.particle_q_predicted)

            # Step 4: Update velocities from position change
            wp.launch(
                kernel=update_velocities_kernel,
                dim=self.num_particles,
                inputs=[
                    self.particle_q_temp,
                    self.state_1.particle_q,
                    self.particle_inv_mass,
                    self.sim_dt,
                ],
                outputs=[self.state_1.particle_qd],
                device=self.model.device,
            )

            # Swap states
            self.state_0, self.state_1 = self.state_1, self.state_0

            # Update contacts for visualization
            self.contacts = self.model.collide(self.state_0, collision_pipeline=self.collision_pipeline)

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt

    def test_final(self):
        # Verify that the anchor particle (first particle) is still at rest
        newton.examples.test_particle_state(
            self.state_0,
            "anchor particle is stationary",
            lambda q, qd: wp.length(qd) < 1e-6,
            indices=[0],
        )

        # Verify all particles are above the ground
        newton.examples.test_particle_state(
            self.state_0,
            "particles are above the ground",
            lambda q, qd: q[2] >= -0.01,
        )

        # Verify particles are within reasonable bounds
        # Rod has 17 particles with 0.1 spacing (~1.6m length)
        p_lower = wp.vec3(-2.0, -3.0, -0.1)
        p_upper = wp.vec3(4.0, 3.0, 5.0)
        newton.examples.test_particle_state(
            self.state_0,
            "particles are within reasonable bounds",
            lambda q, qd: newton.utils.vec_inside_limits(q, p_lower, p_upper),
        )

        # Verify edge quaternions are normalized
        edge_q_np = self.edge_q.numpy()
        for i, q in enumerate(edge_q_np):
            norm = (q[0] ** 2 + q[1] ** 2 + q[2] ** 2 + q[3] ** 2) ** 0.5
            assert abs(norm - 1.0) < 0.1, f"Edge quaternion {i} not normalized: norm={norm}"

    def gui(self, ui):
        ui.text("Cosserat Rod Parameters")
        _changed, self.stretch_stiffness = ui.slider_float("Stretch Stiffness", self.stretch_stiffness, 0.0, 1.0)
        _changed, self.shear_stiffness = ui.slider_float("Shear Stiffness", self.shear_stiffness, 0.0, 1.0)
        _changed, self.bend_stiffness = ui.slider_float("Bend Stiffness", self.bend_stiffness, 0.0, 1.0)
        ui.separator()
        ui.text("Visualization")
        _changed, self.show_directors = ui.checkbox("Show Directors", self.show_directors)
        _changed, self.director_scale = ui.slider_float("Director Scale", self.director_scale, 0.01, 0.2)




    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)

        # Visualize material frames (directors)
        if self.show_directors:
            wp.launch(
                kernel=compute_director_lines_kernel,
                dim=self.num_stretch * 3,
                inputs=[
                    self.state_0.particle_q,
                    self.edge_q,
                    self.num_stretch,
                    self.director_scale,
                ],
                outputs=[
                    self.director_line_starts,
                    self.director_line_ends,
                    self.director_line_colors,
                ],
                device=self.model.device,
            )
            self.viewer.log_lines(
                "/directors",
                self.director_line_starts,
                self.director_line_ends,
                self.director_line_colors,
            )
        else:
            self.viewer.log_lines("/directors", None, None, None)

        self.viewer.end_frame()


if __name__ == "__main__":
    # Parse arguments and initialize viewer
    viewer, args = newton.examples.init()

    # Enable particle visualization if using GL viewer
    if isinstance(viewer, newton.viewer.ViewerGL):
        viewer.show_particles = True

    # Create and run example
    example = Example(viewer, args)

    newton.examples.run(example, args)
