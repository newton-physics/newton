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
# Example Global PBD Chain
#
# Demonstrates a global matrix-based Position Based Dynamics approach for
# distance constraints on a particle chain. Instead of iterative Gauss-Seidel
# constraint projection, this example assembles a global system matrix and
# solves it using Warp's tile Cholesky decomposition on GPU.
#
# Mathematical formulation:
#   - Distance constraint k: C_k = |x_{k+1} - x_k| - L_k = 0
#   - Jacobian J: Each row has [-n_k, n_k] where n_k is unit direction
#   - System matrix: A = J M^{-1} J^T + alpha/dt^2 (tridiagonal SPD)
#   - Solve: A * delta_lambda = -C
#   - Position correction: delta_x = M^{-1} J^T delta_lambda
#
# Command: uv run -m newton.examples basic_global_pbd_chain
#
###########################################################################

import warp as wp

import newton
import newton.examples

# Warp tile configuration
BLOCK_DIM = 128
TILE = 32  # 32x32 tile fits 31 constraints (32 particles)
NUM_PARTICLES = 32
NUM_CONSTRAINTS = NUM_PARTICLES - 1  # 31 distance constraints


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
    """Semi-implicit Euler integration step."""
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
    rest_length: wp.array(dtype=float),
    # outputs
    constraint_violation: wp.array(dtype=float),
    constraint_direction: wp.array(dtype=wp.vec3),
):
    """Compute constraint violations and direction vectors for all distance constraints."""
    tid = wp.tid()

    # Distance constraint between particles tid and tid+1
    p0 = particle_q[tid]
    p1 = particle_q[tid + 1]

    diff = p1 - p0
    length = wp.length(diff)
    L = rest_length[tid]

    # Constraint violation: C = |x1 - x0| - L
    constraint_violation[tid] = length - L

    # Direction vector (normalized)
    if length > 1.0e-8:
        constraint_direction[tid] = diff / length
    else:
        constraint_direction[tid] = wp.vec3(1.0, 0.0, 0.0)


@wp.kernel
def assemble_global_system_kernel(
    particle_inv_mass: wp.array(dtype=float),
    constraint_direction: wp.array(dtype=wp.vec3),
    constraint_violation: wp.array(dtype=float),
    compliance_factor: float,  # alpha / dt^2
    num_constraints: int,
    # outputs
    system_matrix: wp.array2d(dtype=float),
    system_rhs: wp.array1d(dtype=float),
):
    """
    Assemble the global system matrix A and RHS vector b.

    System matrix A = J M^{-1} J^T + alpha/dt^2 * I

    For distance constraints in a chain, A is tridiagonal:
    - A[i,i] = w_i + w_{i+1} + alpha/dt^2
      where w_i = inv_mass[i] + inv_mass[i+1] (sum of inverse masses for constraint i)
    - A[i,i+1] = A[i+1,i] = -inv_mass[i+1] * (n_i . n_{i+1})
      (coupling through shared particle)
    """
    # Single thread assembles the whole matrix (small size)
    # Initialize matrix to zero
    for i in range(TILE):
        system_rhs[i] = 0.0
        for j in range(TILE):
            system_matrix[i, j] = 0.0

    # Fill in the tridiagonal structure
    for k in range(num_constraints):
        # Particles involved in constraint k: k and k+1
        w0 = particle_inv_mass[k]
        w1 = particle_inv_mass[k + 1]

        # Diagonal: contribution from constraint k
        # A[k,k] = (w0 + w1) * (n_k . n_k) + alpha/dt^2 = w0 + w1 + compliance
        diag = w0 + w1 + compliance_factor
        system_matrix[k, k] = diag

        # RHS: -C_k
        system_rhs[k] = -constraint_violation[k]

        # Off-diagonal coupling with next constraint (k+1)
        # Constraints k and k+1 share particle k+1
        if k + 1 < num_constraints:
            n_k = constraint_direction[k]
            n_k1 = constraint_direction[k + 1]

            # Coupling: -w_{k+1} * (n_k . n_{k+1})
            # For constraint k: grad_C_k w.r.t. particle k+1 is +n_k
            # For constraint k+1: grad_C_{k+1} w.r.t. particle k+1 is -n_{k+1}
            # Contribution to A[k, k+1]: w_{k+1} * (+n_k) . (-n_{k+1}) = -w_{k+1} * (n_k . n_{k+1})
            coupling = -particle_inv_mass[k + 1] * wp.dot(n_k, n_k1)
            system_matrix[k, k + 1] = coupling
            system_matrix[k + 1, k] = coupling

    # Pad the unused rows/columns with identity to keep matrix SPD
    for i in range(num_constraints, TILE):
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
def apply_global_corrections_kernel(
    particle_q: wp.array(dtype=wp.vec3),
    particle_inv_mass: wp.array(dtype=float),
    constraint_direction: wp.array(dtype=wp.vec3),
    delta_lambda: wp.array1d(dtype=float),
    num_constraints: int,
    # output
    particle_q_corrected: wp.array(dtype=wp.vec3),
):
    """
    Apply position corrections: delta_x = M^{-1} J^T delta_lambda

    For particle i, the correction is:
    delta_x_i = sum over constraints k involving particle i:
                inv_mass[i] * grad_C_k[i] * delta_lambda[k]

    For distance constraint k between particles k and k+1:
    - grad_C_k[k] = -n_k (pointing from k to k+1)
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
    if tid > 0 and tid - 1 < num_constraints:
        n_prev = constraint_direction[tid - 1]
        dl_prev = delta_lambda[tid - 1]
        # grad_C_{tid-1}[tid] = +n_{tid-1}
        correction = correction + n_prev * dl_prev * inv_mass

    # Contribution from constraint tid (if exists): this particle is the "left" particle
    if tid < num_constraints:
        n_curr = constraint_direction[tid]
        dl_curr = delta_lambda[tid]
        # grad_C_tid[tid] = -n_tid
        correction = correction - n_curr * dl_curr * inv_mass

    particle_q_corrected[tid] = pos + correction


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
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.constraint_iterations = 3  # Number of global solve iterations per substep

        self.viewer = viewer
        self.args = args

        # Particle chain parameters
        self.num_particles = NUM_PARTICLES
        self.num_constraints = NUM_CONSTRAINTS
        particle_spacing = 0.05  # rest length between particles
        particle_mass = 0.1
        particle_radius = 0.02
        start_height = 3.0

        # Compliance: alpha/dt^2 (lower = stiffer, but can cause instability)
        self.compliance = 1.0e-6
        self.compliance_factor = self.compliance / (self.sim_dt * self.sim_dt)

        self.gravity = wp.vec3(0.0, 0.0, -9.81)
        
        self.stiffness = 1.0
        self.spacing = 2.0

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

        # Inverse mass array
        inv_mass_np = [0.0] + [1.0 / particle_mass] * (self.num_particles - 1)
        self.particle_inv_mass = wp.array(inv_mass_np, dtype=float, device=device)

        # Rest lengths for distance constraints
        rest_length_np = [particle_spacing] * self.num_constraints
        self.rest_length = wp.array(rest_length_np, dtype=float, device=device)

        # Buffers for constraint data
        self.constraint_violation = wp.zeros(self.num_constraints, dtype=float, device=device)
        self.constraint_direction = wp.zeros(self.num_constraints, dtype=wp.vec3, device=device)

        # Buffers for global system (32x32 tile)
        self.system_matrix = wp.zeros((TILE, TILE), dtype=float, device=device)
        self.system_rhs = wp.zeros(TILE, dtype=float, device=device)
        self.delta_lambda = wp.zeros(TILE, dtype=float, device=device)

        # Temporary buffer for predicted positions
        self.particle_q_predicted = wp.zeros(self.num_particles, dtype=wp.vec3, device=device)
        self.particle_q_temp = wp.zeros(self.num_particles, dtype=wp.vec3, device=device)

        self.viewer.set_model(self.model)
        self.viewer.show_particles = True

        self.graph = None

    def simulate(self):
        for _ in range(self.sim_substeps):
            # Store old positions for velocity update
            wp.copy(self.particle_q_temp, self.state_0.particle_q)

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
                # Compute constraint violations and directions
                wp.launch(
                    kernel=compute_constraint_data_kernel,
                    dim=self.num_constraints,
                    inputs=[
                        self.state_1.particle_q,
                        self.rest_length,
                    ],
                    outputs=[self.constraint_violation, self.constraint_direction],
                    device=self.model.device,
                )

                # Assemble global system matrix and RHS
                wp.launch(
                    kernel=assemble_global_system_kernel,
                    dim=1,
                    inputs=[
                        self.particle_inv_mass,
                        self.constraint_direction,
                        self.constraint_violation,
                        self.compliance_factor,
                        self.num_constraints,
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
                self.compliance = (1.0 - self.stiffness) * 1e-2
                self.compliance_factor = self.compliance / (self.sim_dt * self.sim_dt)

                # Apply position corrections
                wp.launch(
                    kernel=apply_global_corrections_kernel,
                    dim=self.num_particles,
                    inputs=[
                        self.state_1.particle_q,
                        self.particle_inv_mass,
                        self.constraint_direction,
                        self.delta_lambda,
                        self.num_constraints,
                    ],
                    outputs=[self.particle_q_predicted],
                    device=self.model.device,
                )

                # Copy corrected positions back
                wp.copy(self.state_1.particle_q, self.particle_q_predicted)

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


    def gui(self, ui):
        ui.text("Custom UI text")
        _changed, self.stiffness = ui.slider_float("Stiffness", self.stiffness, 0.0, 1.0)
        _changed, self.spacing = ui.slider_float("Spacing", self.spacing, 0.0, 10.0)

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
        # Chain has 32 particles with 0.05 spacing, but can swing and hang down
        p_lower = wp.vec3(-2.0, -3.0, -0.1)
        p_upper = wp.vec3(4.0, 3.0, 5.0)
        newton.examples.test_particle_state(
            self.state_0,
            "particles are within reasonable bounds",
            lambda q, qd: newton.utils.vec_inside_limits(q, p_lower, p_upper),
        )

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
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
