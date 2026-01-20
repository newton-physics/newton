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
# Example Global PBD Chain - Multi-Tile
#
# Demonstrates a Block-Jacobi approach for global matrix-based Position Based
# Dynamics on longer particle chains. The chain is partitioned into multiple
# 32x32 tiles that are solved in parallel using Warp's tile Cholesky API.
#
# For 128 particles with 127 distance constraints:
#   - Tile 0: constraints 0-31   (particles 0-32)
#   - Tile 1: constraints 32-63  (particles 32-64)
#   - Tile 2: constraints 64-95  (particles 64-96)
#   - Tile 3: constraints 96-126 (particles 96-127)
#
# Each tile is solved independently in parallel. Boundary particles receive
# corrections from both adjacent tiles using atomic operations. Coupling
# between tiles happens through outer constraint iterations.
#
# Command: uv run -m newton.examples basic_global_pbd_chain_multitile
#
###########################################################################

import warp as wp

import newton
import newton.examples

# Warp tile configuration
BLOCK_DIM = 128
TILE = 32  # 32x32 tile size for Cholesky

# Chain configuration
NUM_TILES = 4
CONSTRAINTS_PER_TILE = 32
NUM_PARTICLES = NUM_TILES * TILE  # 128 particles
NUM_CONSTRAINTS = NUM_PARTICLES - 1  # 127 distance constraints


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
    num_constraints: int,
    # outputs
    constraint_violation: wp.array(dtype=float),
    constraint_direction: wp.array(dtype=wp.vec3),
):
    """Compute constraint violations and direction vectors for all distance constraints."""
    tid = wp.tid()

    if tid >= num_constraints:
        return

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
def assemble_block_systems_kernel(
    particle_inv_mass: wp.array(dtype=float),
    constraint_direction: wp.array(dtype=wp.vec3),
    constraint_violation: wp.array(dtype=float),
    compliance_factor: float,
    num_constraints: int,
    constraints_per_tile: int,
    # outputs (per-tile arrays)
    system_matrices: wp.array3d(dtype=float),  # (NUM_TILES, TILE, TILE)
    system_rhs: wp.array2d(dtype=float),  # (NUM_TILES, TILE)
):
    """
    Assemble block system matrices for each tile.
    Each tile handles a contiguous block of constraints.
    """
    tile_idx = wp.tid()

    # Compute constraint range for this tile
    constraint_start = tile_idx * constraints_per_tile
    constraint_end = wp.min(constraint_start + constraints_per_tile, num_constraints)
    num_in_tile = constraint_end - constraint_start

    # Initialize tile matrix to zero
    for i in range(TILE):
        system_rhs[tile_idx, i] = 0.0
        for j in range(TILE):
            system_matrices[tile_idx, i, j] = 0.0

    # Fill in the tridiagonal structure for this tile's constraints
    for local_k in range(num_in_tile):
        global_k = constraint_start + local_k

        # Particles involved in constraint global_k: global_k and global_k+1
        w0 = particle_inv_mass[global_k]
        w1 = particle_inv_mass[global_k + 1]

        # Diagonal element
        diag = w0 + w1 + compliance_factor
        system_matrices[tile_idx, local_k, local_k] = diag

        # RHS: -C_k
        system_rhs[tile_idx, local_k] = -constraint_violation[global_k]

        # Off-diagonal coupling with next constraint in this tile
        if local_k + 1 < num_in_tile:
            n_k = constraint_direction[global_k]
            n_k1 = constraint_direction[global_k + 1]

            # Coupling through shared particle
            coupling = -particle_inv_mass[global_k + 1] * wp.dot(n_k, n_k1)
            system_matrices[tile_idx, local_k, local_k + 1] = coupling
            system_matrices[tile_idx, local_k + 1, local_k] = coupling

    # Pad unused rows/columns with identity to keep matrix SPD
    for i in range(num_in_tile, TILE):
        system_matrices[tile_idx, i, i] = 1.0


@wp.kernel
def cholesky_solve_batched_kernel(
    A: wp.array3d(dtype=float),  # (NUM_TILES, TILE, TILE)
    b: wp.array2d(dtype=float),  # (NUM_TILES, TILE)
    # output
    x: wp.array2d(dtype=float),  # (NUM_TILES, TILE)
):
    """Batched Cholesky solve - one tile per thread block."""
    tile_idx = wp.tid()

    # Load this tile's system matrix and RHS
    a_tile = wp.tile_load(A[tile_idx], shape=(TILE, TILE))
    b_tile = wp.tile_load(b[tile_idx], shape=TILE)

    # Cholesky factorization and solve
    L = wp.tile_cholesky(a_tile)
    x_tile = wp.tile_cholesky_solve(L, b_tile)

    # Store solution
    wp.tile_store(x[tile_idx], x_tile)


@wp.kernel
def zero_corrections_kernel(
    particle_corrections: wp.array(dtype=wp.vec3),
):
    """Zero out particle corrections buffer."""
    tid = wp.tid()
    particle_corrections[tid] = wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def apply_block_corrections_kernel(
    particle_inv_mass: wp.array(dtype=float),
    constraint_direction: wp.array(dtype=wp.vec3),
    delta_lambdas: wp.array2d(dtype=float),  # (NUM_TILES, TILE)
    num_constraints: int,
    constraints_per_tile: int,
    num_tiles: int,
    # output (accumulated via atomics)
    particle_corrections: wp.array(dtype=wp.vec3),
):
    """
    Apply position corrections from all tiles.
    Each tile's solution contributes to the particles it affects.
    Uses atomic operations for boundary particles shared between tiles.
    """
    tile_idx = wp.tid()

    # Compute constraint range for this tile
    constraint_start = tile_idx * constraints_per_tile
    constraint_end = wp.min(constraint_start + constraints_per_tile, num_constraints)
    num_in_tile = constraint_end - constraint_start

    # Apply corrections from this tile's constraints
    for local_k in range(num_in_tile):
        global_k = constraint_start + local_k
        delta_lambda = delta_lambdas[tile_idx, local_k]
        n = constraint_direction[global_k]

        # Particle global_k (left particle of constraint): grad = -n
        inv_m0 = particle_inv_mass[global_k]
        if inv_m0 > 0.0:
            correction0 = -n * delta_lambda * inv_m0
            wp.atomic_add(particle_corrections, global_k, correction0)

        # Particle global_k+1 (right particle of constraint): grad = +n
        inv_m1 = particle_inv_mass[global_k + 1]
        if inv_m1 > 0.0:
            correction1 = n * delta_lambda * inv_m1
            wp.atomic_add(particle_corrections, global_k + 1, correction1)


@wp.kernel
def apply_corrections_to_positions_kernel(
    particle_q: wp.array(dtype=wp.vec3),
    particle_corrections: wp.array(dtype=wp.vec3),
    particle_inv_mass: wp.array(dtype=float),
    # output
    particle_q_out: wp.array(dtype=wp.vec3),
):
    """Apply accumulated corrections to particle positions."""
    tid = wp.tid()

    if particle_inv_mass[tid] == 0.0:
        particle_q_out[tid] = particle_q[tid]
        return

    particle_q_out[tid] = particle_q[tid] + particle_corrections[tid]


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
        self.constraint_iterations = 5  # More iterations for block-Jacobi coupling

        self.viewer = viewer
        self.args = args

        # Particle chain parameters
        self.num_particles = NUM_PARTICLES
        self.num_constraints = NUM_CONSTRAINTS
        self.num_tiles = NUM_TILES
        self.constraints_per_tile = CONSTRAINTS_PER_TILE

        particle_spacing = 0.05  # rest length between particles
        particle_mass = 0.1
        particle_radius = 0.02
        start_height = 4.0  # Higher start for longer chain

        # Compliance: alpha/dt^2 (lower = stiffer)
        self.compliance = 1.0e-6
        self.compliance_factor = self.compliance / (self.sim_dt * self.sim_dt)

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

        # Soft contact parameters
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

        # Per-tile system matrices and vectors
        self.system_matrices = wp.zeros((self.num_tiles, TILE, TILE), dtype=float, device=device)
        self.system_rhs = wp.zeros((self.num_tiles, TILE), dtype=float, device=device)
        self.delta_lambdas = wp.zeros((self.num_tiles, TILE), dtype=float, device=device)

        # Particle correction accumulator (for atomic adds)
        self.particle_corrections = wp.zeros(self.num_particles, dtype=wp.vec3, device=device)

        # Temporary buffers
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

            # Step 2: Block-Jacobi iterative constraint solving
            for _ in range(self.constraint_iterations):
                # Compute constraint violations and directions for ALL constraints
                wp.launch(
                    kernel=compute_constraint_data_kernel,
                    dim=self.num_constraints,
                    inputs=[
                        self.state_1.particle_q,
                        self.rest_length,
                        self.num_constraints,
                    ],
                    outputs=[self.constraint_violation, self.constraint_direction],
                    device=self.model.device,
                )

                # Assemble block system matrices (one per tile, in parallel)
                wp.launch(
                    kernel=assemble_block_systems_kernel,
                    dim=self.num_tiles,
                    inputs=[
                        self.particle_inv_mass,
                        self.constraint_direction,
                        self.constraint_violation,
                        self.compliance_factor,
                        self.num_constraints,
                        self.constraints_per_tile,
                    ],
                    outputs=[self.system_matrices, self.system_rhs],
                    device=self.model.device,
                )

                # Batched Cholesky solve (all tiles in parallel)
                wp.launch_tiled(
                    kernel=cholesky_solve_batched_kernel,
                    dim=[self.num_tiles, 1],
                    inputs=[self.system_matrices, self.system_rhs],
                    outputs=[self.delta_lambdas],
                    block_dim=BLOCK_DIM,
                    device=self.model.device,
                )

                # Zero out correction accumulator
                wp.launch(
                    kernel=zero_corrections_kernel,
                    dim=self.num_particles,
                    inputs=[self.particle_corrections],
                    device=self.model.device,
                )

                # Apply block corrections (with atomic adds for boundaries)
                wp.launch(
                    kernel=apply_block_corrections_kernel,
                    dim=self.num_tiles,
                    inputs=[
                        self.particle_inv_mass,
                        self.constraint_direction,
                        self.delta_lambdas,
                        self.num_constraints,
                        self.constraints_per_tile,
                        self.num_tiles,
                    ],
                    outputs=[self.particle_corrections],
                    device=self.model.device,
                )

                # Apply accumulated corrections to positions
                wp.launch(
                    kernel=apply_corrections_to_positions_kernel,
                    dim=self.num_particles,
                    inputs=[
                        self.state_1.particle_q,
                        self.particle_corrections,
                        self.particle_inv_mass,
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
        # Chain has 128 particles with 0.05 spacing (~6.4m length)
        p_lower = wp.vec3(-3.0, -5.0, -0.1)
        p_upper = wp.vec3(10.0, 5.0, 6.0)
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
