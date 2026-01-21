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
# Example Global PBD Chain - Thomas Algorithm
#
# Demonstrates a global matrix-based Position Based Dynamics approach using
# the Thomas algorithm (aka TDMA - TriDiagonal Matrix Algorithm) for solving
# the tridiagonal system. Unlike tile-based Cholesky which is limited by
# shared memory, Thomas algorithm can handle arbitrarily long chains.
#
# Mathematical formulation:
#   - Distance constraint k: C_k = |x_{k+1} - x_k| - L_k = 0
#   - System matrix A = J M^{-1} J^T + alpha/dt^2 is tridiagonal
#   - Thomas algorithm solves Ax = b in O(n) time
#
# Thomas Algorithm (for symmetric tridiagonal):
#   Forward sweep:  c'[i] = c[i] / (d[i] - a[i]*c'[i-1])
#                   b'[i] = (b[i] - a[i]*b'[i-1]) / (d[i] - a[i]*c'[i-1])
#   Back substitution: x[n-1] = b'[n-1]
#                      x[i] = b'[i] - c'[i]*x[i+1]
#
# Command: uv run -m newton.examples cosserat_global_pbd_chain_thomas
#
###########################################################################

import warp as wp

import newton
import newton.examples

# Chain configuration - can be much larger than tile-based approach!
NUM_PARTICLES = 128
NUM_CONSTRAINTS = NUM_PARTICLES - 1


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
        particle_q_predicted[tid] = particle_q[tid]
        particle_qd_new[tid] = particle_qd[tid]
        return

    v_new = particle_qd[tid] + gravity * dt
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
    """Compute constraint violations and direction vectors."""
    tid = wp.tid()
    if tid >= num_constraints:
        return

    p0 = particle_q[tid]
    p1 = particle_q[tid + 1]

    diff = p1 - p0
    length = wp.length(diff)
    L = rest_length[tid]

    constraint_violation[tid] = length - L

    if length > 1.0e-8:
        constraint_direction[tid] = diff / length
    else:
        constraint_direction[tid] = wp.vec3(1.0, 0.0, 0.0)


@wp.kernel
def assemble_tridiagonal_system_kernel(
    particle_inv_mass: wp.array(dtype=float),
    constraint_direction: wp.array(dtype=wp.vec3),
    constraint_violation: wp.array(dtype=float),
    compliance_factor: float,
    num_constraints: int,
    # outputs - tridiagonal representation
    diag: wp.array(dtype=float),  # main diagonal d[i]
    off_diag: wp.array(dtype=float),  # sub/super diagonal a[i] (symmetric)
    rhs: wp.array(dtype=float),  # right-hand side b[i]
):
    """
    Assemble the tridiagonal system in compact form.

    For a tridiagonal matrix:
        [ d[0]  a[0]    0     0   ...  ]
        [ a[0]  d[1]  a[1]    0   ...  ]
        [   0   a[1]  d[2]  a[2]  ...  ]
        [  ...                         ]

    We store:
        diag[i] = d[i] = w_i + w_{i+1} + compliance
        off_diag[i] = a[i] = -w_{i+1} * (n_i . n_{i+1})
    """
    # Single thread assembles the system (sequential but O(n))
    for k in range(num_constraints):
        w0 = particle_inv_mass[k]
        w1 = particle_inv_mass[k + 1]

        # Diagonal: A[k,k] = w_k + w_{k+1} + compliance
        diag[k] = w0 + w1 + compliance_factor

        # RHS: -C_k
        rhs[k] = -constraint_violation[k]

        # Off-diagonal coupling with next constraint
        if k + 1 < num_constraints:
            n_k = constraint_direction[k]
            n_k1 = constraint_direction[k + 1]
            # A[k, k+1] = -w_{k+1} * (n_k . n_{k+1})
            off_diag[k] = -particle_inv_mass[k + 1] * wp.dot(n_k, n_k1)


@wp.kernel
def thomas_solve_kernel(
    diag: wp.array(dtype=float),
    off_diag: wp.array(dtype=float),
    rhs: wp.array(dtype=float),
    num_constraints: int,
    # workspace
    c_prime: wp.array(dtype=float),
    d_prime: wp.array(dtype=float),
    # output
    x: wp.array(dtype=float),
):
    """
    Thomas algorithm (TDMA) for symmetric tridiagonal systems.

    Solves Ax = b where A is tridiagonal with:
        - diag[i] = main diagonal
        - off_diag[i] = sub/super diagonal (A[i,i+1] = A[i+1,i])

    Algorithm is O(n) - forward elimination then back substitution.
    """
    n = num_constraints

    # Forward elimination
    # c'[0] = c[0] / d[0], where c[0] = off_diag[0]
    c_prime[0] = off_diag[0] / diag[0]
    d_prime[0] = rhs[0] / diag[0]

    for i in range(1, n):
        # a[i] = off_diag[i-1] (sub-diagonal element)
        a_i = off_diag[i - 1]

        # denom = d[i] - a[i] * c'[i-1]
        denom = diag[i] - a_i * c_prime[i - 1]

        # c'[i] = c[i] / denom (only if not last row)
        if i < n - 1:
            c_prime[i] = off_diag[i] / denom

        # d'[i] = (b[i] - a[i] * d'[i-1]) / denom
        d_prime[i] = (rhs[i] - a_i * d_prime[i - 1]) / denom

    # Back substitution
    x[n - 1] = d_prime[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i + 1]


@wp.kernel
def apply_corrections_kernel(
    particle_q: wp.array(dtype=wp.vec3),
    particle_inv_mass: wp.array(dtype=float),
    constraint_direction: wp.array(dtype=wp.vec3),
    delta_lambda: wp.array(dtype=float),
    num_constraints: int,
    # output
    particle_q_corrected: wp.array(dtype=wp.vec3),
):
    """
    Apply position corrections: delta_x = M^{-1} J^T delta_lambda

    For particle i:
    - If involved in constraint i-1 (as right particle): +n_{i-1} * dl_{i-1}
    - If involved in constraint i (as left particle): -n_i * dl_i
    """
    tid = wp.tid()

    inv_mass = particle_inv_mass[tid]
    pos = particle_q[tid]

    if inv_mass == 0.0:
        particle_q_corrected[tid] = pos
        return

    correction = wp.vec3(0.0, 0.0, 0.0)

    # Contribution from constraint tid-1 (this particle is right particle)
    if tid > 0 and tid - 1 < num_constraints:
        n_prev = constraint_direction[tid - 1]
        dl_prev = delta_lambda[tid - 1]
        correction = correction + n_prev * dl_prev * inv_mass

    # Contribution from constraint tid (this particle is left particle)
    if tid < num_constraints:
        n_curr = constraint_direction[tid]
        dl_curr = delta_lambda[tid]
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
    """Update velocities from position change."""
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
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.constraint_iterations = 3

        self.viewer = viewer
        self.args = args

        # Chain parameters
        self.num_particles = NUM_PARTICLES
        self.num_constraints = NUM_CONSTRAINTS
        particle_spacing = 0.05
        particle_mass = 0.1
        particle_radius = 0.02
        start_height = 4.0

        # Compliance
        self.compliance = 1.0e-6
        self.compliance_factor = self.compliance / (self.sim_dt * self.sim_dt)

        self.gravity = wp.vec3(0.0, 0.0, -9.81)

        # Build the model
        builder = newton.ModelBuilder()
        builder.add_ground_plane()

        for i in range(self.num_particles):
            mass = 0.0 if i == 0 else particle_mass
            builder.add_particle(
                pos=(i * particle_spacing, 0.0, start_height),
                vel=(0.0, 0.0, 0.0),
                mass=mass,
                radius=particle_radius,
            )

        self.model = builder.finalize()

        self.model.soft_contact_ke = 1.0e3
        self.model.soft_contact_kd = 1.0e1
        self.model.soft_contact_mu = 0.5

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        self.collision_pipeline = newton.examples.create_collision_pipeline(self.model, self.args)
        self.contacts = self.model.collide(self.state_0, collision_pipeline=self.collision_pipeline)

        device = self.model.device

        # Inverse mass array
        inv_mass_np = [0.0] + [1.0 / particle_mass] * (self.num_particles - 1)
        self.particle_inv_mass = wp.array(inv_mass_np, dtype=float, device=device)

        # Rest lengths
        rest_length_np = [particle_spacing] * self.num_constraints
        self.rest_length = wp.array(rest_length_np, dtype=float, device=device)

        # Constraint data buffers
        self.constraint_violation = wp.zeros(self.num_constraints, dtype=float, device=device)
        self.constraint_direction = wp.zeros(self.num_constraints, dtype=wp.vec3, device=device)

        # Tridiagonal system storage (compact form - no full matrix needed!)
        self.diag = wp.zeros(self.num_constraints, dtype=float, device=device)
        self.off_diag = wp.zeros(self.num_constraints - 1, dtype=float, device=device)
        self.rhs = wp.zeros(self.num_constraints, dtype=float, device=device)
        self.delta_lambda = wp.zeros(self.num_constraints, dtype=float, device=device)

        # Thomas algorithm workspace
        self.c_prime = wp.zeros(self.num_constraints, dtype=float, device=device)
        self.d_prime = wp.zeros(self.num_constraints, dtype=float, device=device)

        # Position buffers
        self.particle_q_predicted = wp.zeros(self.num_particles, dtype=wp.vec3, device=device)
        self.particle_q_temp = wp.zeros(self.num_particles, dtype=wp.vec3, device=device)

        self.viewer.set_model(self.model)
        self.viewer.show_particles = True

    def simulate(self):
        for _ in range(self.sim_substeps):
            wp.copy(self.particle_q_temp, self.state_0.particle_q)

            # Integrate
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

            wp.copy(self.state_1.particle_q, self.particle_q_predicted)

            # Constraint solving iterations
            for _ in range(self.constraint_iterations):
                # Compute constraint data
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

                # Assemble tridiagonal system (compact form)
                wp.launch(
                    kernel=assemble_tridiagonal_system_kernel,
                    dim=1,
                    inputs=[
                        self.particle_inv_mass,
                        self.constraint_direction,
                        self.constraint_violation,
                        self.compliance_factor,
                        self.num_constraints,
                    ],
                    outputs=[self.diag, self.off_diag, self.rhs],
                    device=self.model.device,
                )

                # Solve using Thomas algorithm
                wp.launch(
                    kernel=thomas_solve_kernel,
                    dim=1,
                    inputs=[
                        self.diag,
                        self.off_diag,
                        self.rhs,
                        self.num_constraints,
                        self.c_prime,
                        self.d_prime,
                    ],
                    outputs=[self.delta_lambda],
                    device=self.model.device,
                )

                # Apply corrections
                wp.launch(
                    kernel=apply_corrections_kernel,
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

                wp.copy(self.state_1.particle_q, self.particle_q_predicted)

            # Ground collision
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

            # Update velocities
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

            self.state_0, self.state_1 = self.state_1, self.state_0
            self.contacts = self.model.collide(self.state_0, collision_pipeline=self.collision_pipeline)

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt

    def test_final(self):
        newton.examples.test_particle_state(
            self.state_0,
            "anchor particle is stationary",
            lambda q, qd: wp.length(qd) < 1e-6,
            indices=[0],
        )

        newton.examples.test_particle_state(
            self.state_0,
            "particles are above the ground",
            lambda q, qd: q[2] >= -0.01,
        )

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
    viewer, args = newton.examples.init()

    if isinstance(viewer, newton.viewer.ViewerGL):
        viewer.show_particles = True

    example = Example(viewer, args)

    newton.examples.run(example, args)
