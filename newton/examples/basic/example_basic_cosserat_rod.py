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
# Example Basic Cosserat Rod
#
# Demonstrates Position And Orientation Based Cosserat Rods using GPU Warp
# kernels. Implements the two core constraint solvers from the paper:
# - Stretch/Shear constraint: enforces edge length and alignment
# - Bend/Twist constraint: enforces relative rotation via Darboux vector
#
# Reference: "Position And Orientation Based Cosserat Rods"
# by Tassilo Kugelstadt, RWTH Aachen University
# https://animation.rwth-aachen.de/publication/0550/
#
# Command: python -m newton.examples basic_cosserat_rod
#
###########################################################################

import warp as wp

import newton
import newton.examples


@wp.func
def quat_rotate_vector_by_e3(q: wp.quat) -> wp.vec3:
    """Compute the third director d3 = q * e3 * conjugate(q) where e3 = (0,0,1).

    This is an optimized computation of rotating the z-axis by quaternion q.
    """
    # d3 = q * (0,0,1,0) * conjugate(q)
    # Optimized formula from the reference implementation
    x, y, z, w = q[0], q[1], q[2], q[3]
    d3_x = 2.0 * (x * z + w * y)
    d3_y = 2.0 * (y * z - w * x)
    d3_z = w * w - x * x - y * y + z * z
    return wp.vec3(d3_x, d3_y, d3_z)


@wp.func
def quat_e3_conjugate(q: wp.quat) -> wp.quat:
    """Compute q * e3_bar where e3_bar is the quaternion (0,0,-1,0).

    Correct mapping for Warp (x,y,z,w) to match Eigen's (z, -y, x, -w) in (w,x,y,z):
    x' = -q.y
    y' = q.x
    z' = -q.w
    w' = q.z
    """
    return wp.quat(-q[1], q[0], -q[3], q[2])

# @wp.func
# def quat_e3_conjugate(q: wp.quat) -> wp.quat:
#     """Compute q * e3_bar where e3_bar is the quaternion (0,0,-1,0).
    
#     Correct mapping for Warp (x,y,z,w) to match Eigen's (z, -y, x, -w) in (w,x,y,z):
#     """
#     return wp.quat(-q[1], q[0], -q[3], q[2])


@wp.func
def quat_from_vec3(v: wp.vec3) -> wp.quat:
    """Create a quaternion with w=0 from a vec3 (pure quaternion)."""
    return wp.quat(v[0], v[1], v[2], 0.0)


@wp.kernel
def integrate_particles_kernel(
    particle_q: wp.array(dtype=wp.vec3),
    particle_qd: wp.array(dtype=wp.vec3),
    particle_inv_mass: wp.array(dtype=float),
    gravity: wp.vec3,
    dt: float,
    particle_q_new: wp.array(dtype=wp.vec3),
    particle_qd_new: wp.array(dtype=wp.vec3),
):
    """Semi-implicit Euler integration for particles."""
    tid = wp.tid()
    inv_mass = particle_inv_mass[tid]

    if inv_mass == 0.0:
        # Kinematic particle - no integration
        particle_q_new[tid] = particle_q[tid]
        particle_qd_new[tid] = particle_qd[tid]
        return

    # Update velocity with gravity
    v = particle_qd[tid] + gravity * dt
    # Update position
    x = particle_q[tid] + v * dt

    particle_q_new[tid] = x
    particle_qd_new[tid] = v


@wp.kernel
def solve_stretch_shear_constraint_kernel(
    particle_q: wp.array(dtype=wp.vec3),
    particle_inv_mass: wp.array(dtype=float),
    edge_q: wp.array(dtype=wp.quat),
    edge_inv_mass: wp.array(dtype=float),
    edge_rest_length: wp.array(dtype=float),
    stretch_shear_stiffness: wp.vec3,
    dt: float,
    # outputs
    particle_delta: wp.array(dtype=wp.vec3),
    edge_q_delta: wp.array(dtype=wp.quat),
):
    """Solve stretch and shear constraint for Cosserat rods.

    Based on Eq. 37 from "Position And Orientation Based Cosserat Rods".
    Constrains the edge direction (p1-p0) to align with the quaternion's d3 director
    while maintaining the rest length.
    """
    tid = wp.tid()
    eps = 1.0e-6

    # Edge tid connects particles tid and tid+1
    p0 = particle_q[tid]
    p1 = particle_q[tid + 1]
    q0 = edge_q[tid]

    inv_mass_p0 = particle_inv_mass[tid]
    inv_mass_p1 = particle_inv_mass[tid + 1]
    inv_mass_q0 = edge_inv_mass[tid]
    rest_length = edge_rest_length[tid]

    # Compute third director d3 = q0 * e3 * conjugate(q0)
    d3 = quat_rotate_vector_by_e3(q0)

    # Compute constraint violation: gamma = (p1 - p0) / L - d3
    edge_vec = p1 - p0
    gamma = edge_vec / rest_length - d3

    # Compute denominator for constraint scaling
    denom = (inv_mass_p0 + inv_mass_p1) / rest_length + inv_mass_q0 * 4.0 * rest_length + eps

    # Scale gamma by inverse denominator
    gamma = gamma / denom

    # Apply stretching and shearing stiffness in local space (Anisotropic)
    # Ks_w = R(q0) * diag(Ks) * R^T(q0) * gamma
    gamma_loc = wp.quat_rotate_inv(q0, gamma)
    gamma_loc_x = gamma_loc[0] * stretch_shear_stiffness[0]
    gamma_loc_y = gamma_loc[1] * stretch_shear_stiffness[1]
    gamma_loc_z = gamma_loc[2] * stretch_shear_stiffness[2]
    gamma = wp.quat_rotate(q0, wp.vec3(gamma_loc_x, gamma_loc_y, gamma_loc_z))

    # Compute position corrections
    corr0 = gamma * inv_mass_p0
    corr1 = gamma * (-inv_mass_p1)

    # Compute quaternion correction
    # corrq0 = Quaternion(0, gamma) * q_e3_bar * (2 * invMassq0 * restLength)
    q_e3_bar = quat_e3_conjugate(q0)
    gamma_quat = quat_from_vec3(gamma)
    corrq0 = wp.mul(gamma_quat, q_e3_bar)

    scale = 2.0 * inv_mass_q0 * rest_length
    corrq0 = wp.quat(corrq0[0] * scale, corrq0[1] * scale, corrq0[2] * scale, corrq0[3] * scale)

    # Accumulate corrections using atomic adds
    wp.atomic_add(particle_delta, tid, corr0)
    wp.atomic_add(particle_delta, tid + 1, corr1)

    # For quaternions, we accumulate the correction directly
    wp.atomic_add(edge_q_delta, tid, corrq0)


@wp.kernel
def solve_bend_twist_constraint_kernel(
    edge_q: wp.array(dtype=wp.quat),
    edge_inv_mass: wp.array(dtype=float),
    rest_darboux: wp.array(dtype=wp.quat),
    bend_twist_stiffness: wp.vec3,
    # outputs
    edge_q_delta: wp.array(dtype=wp.quat),
):
    """Solve bend and twist constraint for Cosserat rods.

    Based on Eq. 40 from "Position And Orientation Based Cosserat Rods".
    Constrains the relative rotation between adjacent quaternions to match
    the rest Darboux vector.
    """
    tid = wp.tid()
    eps = 1.0e-6

    # Constraint tid connects edges tid and tid+1
    q0 = edge_q[tid]
    q1 = edge_q[tid + 1]

    inv_mass_q0 = edge_inv_mass[tid]
    inv_mass_q1 = edge_inv_mass[tid + 1]
    rest_darboux_q = rest_darboux[tid]

    # Compute Darboux vector: omega = conjugate(q0) * q1
    q0_conj = wp.quat(-q0[0], -q0[1], -q0[2], q0[3])
    omega = wp.mul(q0_conj, q1)

    # Handle quaternion double-cover: choose the shorter path
    # omega_plus = omega + restDarboux
    # omega_minus = omega - restDarboux
    omega_plus_x = omega[0] + rest_darboux_q[0]
    omega_plus_y = omega[1] + rest_darboux_q[1]
    omega_plus_z = omega[2] + rest_darboux_q[2]
    omega_plus_w = omega[3] + rest_darboux_q[3]

    omega_minus_x = omega[0] - rest_darboux_q[0]
    omega_minus_y = omega[1] - rest_darboux_q[1]
    omega_minus_z = omega[2] - rest_darboux_q[2]
    omega_minus_w = omega[3] - rest_darboux_q[3]

    # Squared norms
    norm_plus_sq = omega_plus_x * omega_plus_x + omega_plus_y * omega_plus_y + omega_plus_z * omega_plus_z + omega_plus_w * omega_plus_w
    norm_minus_sq = omega_minus_x * omega_minus_x + omega_minus_y * omega_minus_y + omega_minus_z * omega_minus_z + omega_minus_w * omega_minus_w

    # Choose the smaller deviation
    if norm_minus_sq > norm_plus_sq:
        omega_x = omega_plus_x
        omega_y = omega_plus_y
        omega_z = omega_plus_z
    else:
        omega_x = omega_minus_x
        omega_y = omega_minus_y
        omega_z = omega_minus_z

    # Apply bending and twisting stiffness
    denom = inv_mass_q0 + inv_mass_q1 + eps
    omega_x = omega_x * bend_twist_stiffness[0] / denom
    omega_y = omega_y * bend_twist_stiffness[1] / denom
    omega_z = omega_z * bend_twist_stiffness[2] / denom

    # Omega with w=0 (discrete Darboux vector has vanishing scalar part)
    omega_corrected = wp.quat(omega_x, omega_y, omega_z, 0.0)

    # Compute quaternion corrections
    # corrq0 = q1 * omega * invMassq0
    # corrq1 = q0 * omega * (-invMassq1)
    corrq0_raw = wp.mul(q1, omega_corrected)
    corrq1_raw = wp.mul(q0, omega_corrected)

    corrq0 = wp.quat(
        corrq0_raw[0] * inv_mass_q0,
        corrq0_raw[1] * inv_mass_q0,
        corrq0_raw[2] * inv_mass_q0,
        corrq0_raw[3] * inv_mass_q0,
    )
    corrq1 = wp.quat(
        corrq1_raw[0] * (-inv_mass_q1),
        corrq1_raw[1] * (-inv_mass_q1),
        corrq1_raw[2] * (-inv_mass_q1),
        corrq1_raw[3] * (-inv_mass_q1),
    )

    # Accumulate corrections
    wp.atomic_add(edge_q_delta, tid, corrq0)
    wp.atomic_add(edge_q_delta, tid + 1, corrq1)


@wp.kernel
def apply_particle_corrections_kernel(
    particle_q: wp.array(dtype=wp.vec3),
    particle_qd: wp.array(dtype=wp.vec3),
    particle_delta: wp.array(dtype=wp.vec3),
    particle_inv_mass: wp.array(dtype=float),
    dt: float,
    particle_q_out: wp.array(dtype=wp.vec3),
    particle_qd_out: wp.array(dtype=wp.vec3),
):
    """Apply accumulated position corrections to particles."""
    tid = wp.tid()

    inv_mass = particle_inv_mass[tid]
    if inv_mass == 0.0:
        particle_q_out[tid] = particle_q[tid]
        particle_qd_out[tid] = particle_qd[tid]
        return

    delta = particle_delta[tid]
    q_new = particle_q[tid] + delta

    # Update velocity based on position change
    qd_new = particle_qd[tid] + delta / dt

    particle_q_out[tid] = q_new
    particle_qd_out[tid] = qd_new


@wp.kernel
def apply_quaternion_corrections_kernel(
    edge_q: wp.array(dtype=wp.quat),
    edge_q_delta: wp.array(dtype=wp.quat),
    edge_inv_mass: wp.array(dtype=float),
    edge_q_out: wp.array(dtype=wp.quat),
):
    """Apply accumulated quaternion corrections and normalize."""
    tid = wp.tid()

    inv_mass = edge_inv_mass[tid]
    if inv_mass == 0.0:
        edge_q_out[tid] = edge_q[tid]
        return

    q = edge_q[tid]
    dq = edge_q_delta[tid]

    # Add correction
    q_new = wp.quat(q[0] + dq[0], q[1] + dq[1], q[2] + dq[2], q[3] + dq[3])

    # Normalize to maintain unit quaternion
    q_new = wp.normalize(q_new)

    edge_q_out[tid] = q_new


@wp.kernel
def zero_vec3_kernel(arr: wp.array(dtype=wp.vec3)):
    """Zero out a vec3 array."""
    tid = wp.tid()
    arr[tid] = wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def zero_quat_kernel(arr: wp.array(dtype=wp.quat)):
    """Zero out a quaternion array."""
    tid = wp.tid()
    arr[tid] = wp.quat(0.0, 0.0, 0.0, 0.0)


@wp.kernel
def solve_ground_collision_kernel(
    particle_q: wp.array(dtype=wp.vec3),
    particle_inv_mass: wp.array(dtype=float),
    particle_radius: wp.array(dtype=float),
    ground_level: float,
    # outputs
    particle_delta: wp.array(dtype=wp.vec3),
):
    """Solve ground plane collision constraint for particles.

    Pushes particles above the ground plane (z >= ground_level + radius).
    """
    tid = wp.tid()

    inv_mass = particle_inv_mass[tid]
    if inv_mass == 0.0:
        return

    pos = particle_q[tid]
    radius = particle_radius[tid]

    # Check penetration with ground plane
    min_z = ground_level + radius
    penetration = min_z - pos[2]

    if penetration > 0.0:
        # Push particle out of ground
        correction = wp.vec3(0.0, 0.0, penetration)
        wp.atomic_add(particle_delta, tid, correction)


class Example:
    def __init__(self, viewer, args=None):
        # Setup simulation parameters
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 8
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.constraint_iterations = 2

        self.viewer = viewer
        self.args = args

        # Rod parameters
        self.num_particles = 100
        self.num_edges = self.num_particles - 1
        self.num_bend_twist = self.num_edges - 1

        particle_spacing = 0.05  # distance between particles (rest length)
        particle_mass = 0.01
        particle_radius = 0.01
        edge_mass = 0.001  # mass associated with each quaternion
        start_height = 3.0

        # Stiffness parameters (isotropic for simplicity)
        self.stretch_shear_stiffness = wp.vec3(1.0, 1.0, 1.0)  # normalized, actual stiffness via XPBD
        #self.stretch_shear_stiffness = wp.vec3(0.1, 0.1, 0.1)  # normalized, actual stiffness via XPBD
        
        #self.bend_twist_stiffness = wp.vec3(0.1, 0.1, 0.1)  # bending and torsion stiffness
        self.bend_twist_stiffness = wp.vec3(0.5, 0.5, 0.5)  # bending and torsion stiffness
        self.gravity = wp.vec3(0.0, 0.0, -9.81)

        # Create model for collision detection and rendering
        builder = newton.ModelBuilder()
        builder.add_ground_plane()

        # Add particles
        for i in range(self.num_particles):
            mass = 0.0 if i == 0 else particle_mass  # First particle is kinematic
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

        # Create collision pipeline
        self.collision_pipeline = newton.examples.create_collision_pipeline(self.model, self.args)
        self.contacts = self.model.collide(self.state_0, collision_pipeline=self.collision_pipeline)

        # Allocate Cosserat rod specific arrays
        device = self.model.device

        # Particle inverse masses
        self.particle_inv_mass = wp.zeros(self.num_particles, dtype=float, device=device)
        inv_mass_np = [0.0] + [1.0 / particle_mass] * (self.num_particles - 1)
        self.particle_inv_mass = wp.array(inv_mass_np, dtype=float, device=device)

        # Edge quaternions (initialized to identity - aligned with x-axis initially)
        # For a horizontal rod along x-axis, the local z-axis (d3) should point along x
        # We need quat that rotates (0,0,1) to (1,0,0): 90 degrees around y-axis
        # q = (sin(45°)*0, sin(45°)*1, sin(45°)*0, cos(45°)) = (0, 0.707, 0, 0.707)
        import math

        angle = math.pi / 2.0
        q_init = wp.quat(0.0, math.sin(angle / 2.0), 0.0, math.cos(angle / 2.0))
        edge_q_init = [q_init] * self.num_edges
        self.edge_q = wp.array(edge_q_init, dtype=wp.quat, device=device)
        self.edge_q_new = wp.array(edge_q_init, dtype=wp.quat, device=device)

        # Edge inverse masses
        edge_inv_mass_np = [1.0 / edge_mass] * self.num_edges
        self.edge_inv_mass = wp.array(edge_inv_mass_np, dtype=float, device=device)

        # Rest lengths
        rest_length_np = [particle_spacing] * self.num_edges
        self.edge_rest_length = wp.array(rest_length_np, dtype=float, device=device)

        # Rest Darboux vectors (identity rotation between adjacent edges at rest)
        # For a straight rod, rest Darboux vector is identity quaternion
        #rest_darboux_np = [wp.quat(0.0, 0.0, 0.0, 1.0)] * self.num_bend_twist
        rest_darboux_np = [wp.quat(0.0, 0.1, 0.1, 1.0)] * self.num_bend_twist
        self.rest_darboux = wp.array(rest_darboux_np, dtype=wp.quat, device=device)

        # Correction accumulators
        self.particle_delta = wp.zeros(self.num_particles, dtype=wp.vec3, device=device)
        self.edge_q_delta = wp.zeros(self.num_edges, dtype=wp.quat, device=device)

        self.viewer.set_model(self.model)
        self.viewer.show_particles = True

        # Note: CUDA graph capture disabled for this example due to custom simulation loop
        self.graph = None

    def simulate(self):
        for _ in range(self.sim_substeps):
            # Integration step
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
                outputs=[self.state_1.particle_q, self.state_1.particle_qd],
                device=self.model.device,
            )

            # Swap states for constraint solving
            self.state_0, self.state_1 = self.state_1, self.state_0

            # Constraint solving iterations
            for _ in range(self.constraint_iterations):
                # Zero out correction accumulators
                wp.launch(
                    kernel=zero_vec3_kernel,
                    dim=self.num_particles,
                    inputs=[self.particle_delta],
                    device=self.model.device,
                )
                wp.launch(
                    kernel=zero_quat_kernel,
                    dim=self.num_edges,
                    inputs=[self.edge_q_delta],
                    device=self.model.device,
                )

                # Solve stretch/shear constraints
                wp.launch(
                    kernel=solve_stretch_shear_constraint_kernel,
                    dim=self.num_edges,
                    inputs=[
                        self.state_0.particle_q,
                        self.particle_inv_mass,
                        self.edge_q,
                        self.edge_inv_mass,
                        self.edge_rest_length,
                        self.stretch_shear_stiffness,
                        self.sim_dt,
                    ],
                    outputs=[self.particle_delta, self.edge_q_delta],
                    device=self.model.device,
                )

                # Solve bend/twist constraints
                if self.num_bend_twist > 0:
                    wp.launch(
                        kernel=solve_bend_twist_constraint_kernel,
                        dim=self.num_bend_twist,
                        inputs=[
                            self.edge_q,
                            self.edge_inv_mass,
                            self.rest_darboux,
                            self.bend_twist_stiffness,
                        ],
                        outputs=[self.edge_q_delta],
                        device=self.model.device,
                    )

                # Solve ground collision constraints
                wp.launch(
                    kernel=solve_ground_collision_kernel,
                    dim=self.num_particles,
                    inputs=[
                        self.state_0.particle_q,
                        self.particle_inv_mass,
                        self.model.particle_radius,
                        0.0,  # ground level at z=0
                    ],
                    outputs=[self.particle_delta],
                    device=self.model.device,
                )

                # Apply particle corrections
                wp.launch(
                    kernel=apply_particle_corrections_kernel,
                    dim=self.num_particles,
                    inputs=[
                        self.state_0.particle_q,
                        self.state_0.particle_qd,
                        self.particle_delta,
                        self.particle_inv_mass,
                        self.sim_dt,
                    ],
                    outputs=[self.state_1.particle_q, self.state_1.particle_qd],
                    device=self.model.device,
                )

                # Apply quaternion corrections
                wp.launch(
                    kernel=apply_quaternion_corrections_kernel,
                    dim=self.num_edges,
                    inputs=[
                        self.edge_q,
                        self.edge_q_delta,
                        self.edge_inv_mass,
                    ],
                    outputs=[self.edge_q_new],
                    device=self.model.device,
                )

                # Swap for next iteration
                self.state_0, self.state_1 = self.state_1, self.state_0
                self.edge_q, self.edge_q_new = self.edge_q_new, self.edge_q

            # Update contacts for ground collision
            self.contacts = self.model.collide(self.state_0, collision_pipeline=self.collision_pipeline)

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt

    def test_final(self):
        # Verify that the anchor particle (first particle) is still at rest
        newton.examples.test_particle_state(
            self.state_0,
            "anchor particle is stationary",
            lambda q, qd: wp.length(qd) < 1.0,  # Allow some small velocity
            indices=[0],
        )

        # Verify all particles are above the ground (with small tolerance)
        newton.examples.test_particle_state(
            self.state_0,
            "particles are above the ground",
            lambda q, qd: q[2] >= -0.1,
        )

        # Verify particles are within reasonable bounds
        p_lower = wp.vec3(-2.0, -2.0, -0.2)
        p_upper = wp.vec3(5.0, 2.0, 5.0)
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

    # Create example and run
    example = Example(viewer, args)

    newton.examples.run(example, args)
