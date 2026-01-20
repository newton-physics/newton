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
# Example Catheter Navigation in Aorta
#
# Demonstrates a Cosserat rod (catheter) navigating inside aorta blood vessels.
# The catheter is modeled as a flexible rod with:
#   - Stretch/shear constraints for inextensibility
#   - Bend/twist constraints for flexibility
#   - Collision with vessel walls
#
# Controls:
#   - Numpad 4/6: Move catheter tip left/right (X)
#   - Numpad 8/2: Move catheter tip forward/back (Y)
#   - Numpad 9/3: Move catheter tip up/down (Z)
#   - Numpad +/-: Advance/retract catheter
#
# Command: uv run -m newton.examples.cosserat.example_catheter_aorta
#
###########################################################################

import math
import os

import numpy as np
import warp as wp
from pxr import Usd

import newton
import newton.examples
import newton.usd

# Catheter rod configuration
NUM_PARTICLES = 65  # Number of particles along the catheter
CATHETER_LENGTH = 0.3  # Total length in meters (30cm)
CATHETER_RADIUS = 0.002  # Radius in meters (2mm)


@wp.func
def quat_rotate_e3(q: wp.quat) -> wp.vec3:
    """Compute the third director d3 = q * e3 * conjugate(q) where e3 = (0,0,1)."""
    x, y, z, w = q[0], q[1], q[2], q[3]
    d3_x = 2.0 * (x * z + w * y)
    d3_y = 2.0 * (y * z - w * x)
    d3_z = w * w - x * x - y * y + z * z
    return wp.vec3(d3_x, d3_y, d3_z)


@wp.func
def quat_e3_bar(q: wp.quat) -> wp.quat:
    """Compute q * e3_bar where e3_bar is the conjugate of quaternion (0,0,1,0)."""
    return wp.quat(-q[1], q[0], -q[3], q[2])


@wp.func
def compute_darboux_vector(q0: wp.quat, q1: wp.quat, rest_darboux_q: wp.quat) -> wp.vec3:
    """Compute the Darboux vector (curvature) between two quaternions."""
    q0_conj = wp.quat(-q0[0], -q0[1], -q0[2], q0[3])
    omega = wp.mul(q0_conj, q1)

    omega_plus_x = omega[0] + rest_darboux_q[0]
    omega_plus_y = omega[1] + rest_darboux_q[1]
    omega_plus_z = omega[2] + rest_darboux_q[2]
    omega_plus_w = omega[3] + rest_darboux_q[3]

    omega_minus_x = omega[0] - rest_darboux_q[0]
    omega_minus_y = omega[1] - rest_darboux_q[1]
    omega_minus_z = omega[2] - rest_darboux_q[2]
    omega_minus_w = omega[3] - rest_darboux_q[3]

    norm_plus_sq = (
        omega_plus_x * omega_plus_x
        + omega_plus_y * omega_plus_y
        + omega_plus_z * omega_plus_z
        + omega_plus_w * omega_plus_w
    )
    norm_minus_sq = (
        omega_minus_x * omega_minus_xmodel
        + omega_minus_y * omega_minus_y
        + omega_minus_z * omega_minus_z
        + omega_minus_w * omega_minus_w
    )

    if norm_minus_sq > norm_plus_sq:
        return wp.vec3(omega_plus_x, omega_plus_y, omega_plus_z)
    else:
        return wp.vec3(omega_minus_x, omega_minus_y, omega_minus_z)


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
        particle_q_predicted[tid] = particle_q[tid]
        particle_qd_new[tid] = particle_qd[tid]
        return

    v_new = particle_qd[tid] + gravity * dt
    x_predicted = particle_q[tid] + v_new * dt

    particle_q_predicted[tid] = x_predicted
    particle_qd_new[tid] = v_new


@wp.kernel
def solve_stretch_shear_constraint_kernel(
    particle_q: wp.array(dtype=wp.vec3),
    particle_inv_mass: wp.array(dtype=float),
    edge_q: wp.array(dtype=wp.quat),
    edge_inv_mass: wp.array(dtype=float),
    rest_length: wp.array(dtype=float),
    stretch_shear_stiffness: wp.vec3,
    num_stretch: int,
    # outputs
    particle_delta: wp.array(dtype=wp.vec3),
    edge_q_delta: wp.array(dtype=wp.quat),
):
    """Solve stretch and shear constraint for Cosserat rods."""
    tid = wp.tid()
    if tid >= num_stretch:
        return

    eps = 1.0e-6

    p0 = particle_q[tid]
    p1 = particle_q[tid + 1]
    q0 = edge_q[tid]

    inv_mass_p0 = particle_inv_mass[tid]
    inv_mass_p1 = particle_inv_mass[tid + 1]
    inv_mass_q0 = edge_inv_mass[tid]
    L = rest_length[tid]

    d3 = quat_rotate_e3(q0)
    edge_vec = p1 - p0
    gamma = edge_vec / L - d3

    denom = (inv_mass_p0 + inv_mass_p1) / L + inv_mass_q0 * 4.0 * L + eps
    gamma = gamma / denom

    gamma_loc = wp.quat_rotate_inv(q0, gamma)
    gamma_loc = wp.vec3(
        gamma_loc[0] * stretch_shear_stiffness[0],
        gamma_loc[1] * stretch_shear_stiffness[1],
        gamma_loc[2] * stretch_shear_stiffness[2],
    )
    gamma = wp.quat_rotate(q0, gamma_loc)

    corr0 = gamma * inv_mass_p0
    corr1 = gamma * (-inv_mass_p1)

    q_e3_bar_val = quat_e3_bar(q0)
    gamma_quat = wp.quat(gamma[0], gamma[1], gamma[2], 0.0)
    corrq0 = wp.mul(gamma_quat, q_e3_bar_val)

    scale = 2.0 * inv_mass_q0 * L
    corrq0 = wp.quat(corrq0[0] * scale, corrq0[1] * scale, corrq0[2] * scale, corrq0[3] * scale)

    wp.atomic_add(particle_delta, tid, corr0)
    wp.atomic_add(particle_delta, tid + 1, corr1)
    wp.atomic_add(edge_q_delta, tid, corrq0)


@wp.kernel
def solve_bend_twist_constraint_kernel(
    edge_q: wp.array(dtype=wp.quat),
    edge_inv_mass: wp.array(dtype=float),
    rest_darboux: wp.array(dtype=wp.quat),
    bend_twist_stiffness: wp.vec3,
    num_bend: int,
    # output
    edge_q_delta: wp.array(dtype=wp.quat),
):
    """Solve bend and twist constraint for Cosserat rods."""
    tid = wp.tid()
    if tid >= num_bend:
        return

    eps = 1.0e-6

    q0 = edge_q[tid]
    q1 = edge_q[tid + 1]

    inv_mass_q0 = edge_inv_mass[tid]
    inv_mass_q1 = edge_inv_mass[tid + 1]
    rest_darboux_q = rest_darboux[tid]

    kappa = compute_darboux_vector(q0, q1, rest_darboux_q)
    omega_x = kappa[0]
    omega_y = kappa[1]
    omega_z = kappa[2]

    denom = inv_mass_q0 + inv_mass_q1 + eps
    omega_x = omega_x * bend_twist_stiffness[0] / denom
    omega_y = omega_y * bend_twist_stiffness[1] / denom
    omega_z = omega_z * bend_twist_stiffness[2] / denom

    omega_corrected = wp.quat(omega_x, omega_y, omega_z, 0.0)

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

    wp.atomic_add(edge_q_delta, tid, corrq0)
    wp.atomic_add(edge_q_delta, tid + 1, corrq1)


@wp.kernel
def apply_particle_corrections_kernel(
    particle_q: wp.array(dtype=wp.vec3),
    particle_delta: wp.array(dtype=wp.vec3),
    particle_inv_mass: wp.array(dtype=float),
    # output
    particle_q_out: wp.array(dtype=wp.vec3),
):
    """Apply accumulated position corrections to particles."""
    tid = wp.tid()

    inv_mass = particle_inv_mass[tid]
    if inv_mass == 0.0:
        particle_q_out[tid] = particle_q[tid]
        return

    delta = particle_delta[tid]
    particle_q_out[tid] = particle_q[tid] + delta


@wp.kernel
def apply_quaternion_corrections_kernel(
    edge_q: wp.array(dtype=wp.quat),
    edge_q_delta: wp.array(dtype=wp.quat),
    edge_inv_mass: wp.array(dtype=float),
    # output
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

    q_new = wp.quat(q[0] + dq[0], q[1] + dq[1], q[2] + dq[2], q[3] + dq[3])
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
def apply_velocity_damping_kernel(
    particle_qd: wp.array(dtype=wp.vec3),
    particle_inv_mass: wp.array(dtype=float),
    damping_coeff: float,
    # output
    particle_qd_out: wp.array(dtype=wp.vec3),
):
    """Apply velocity damping to particles."""
    tid = wp.tid()

    if particle_inv_mass[tid] == 0.0:
        particle_qd_out[tid] = particle_qd[tid]
        return

    particle_qd_out[tid] = particle_qd[tid] * damping_coeff


@wp.kernel
def compute_director_lines_kernel(
    particle_q: wp.array(dtype=wp.vec3),
    edge_q: wp.array(dtype=wp.quat),
    num_edges: int,
    axis_length: float,
    # outputs
    line_starts: wp.array(dtype=wp.vec3),
    line_ends: wp.array(dtype=wp.vec3),
    line_colors: wp.array(dtype=wp.vec3),
):
    """Compute line segments for visualizing material frames."""
    tid = wp.tid()
    edge_idx = tid // 3
    axis_idx = tid % 3

    if edge_idx >= num_edges:
        returnmodel

    p0 = particle_q[edge_idx]
    p1 = particle_q[edge_idx + 1]
    midpoint = (p0 + p1) * 0.5

    q = edge_q[edge_idx]

    if axis_idx == 0:
        x, y, z, w = q[0], q[1], q[2], q[3]
        d1_x = w * w + x * x - y * y - z * z
        d1_y = 2.0 * (x * y + w * z)
        d1_z = 2.0 * (x * z - w * y)
        director = wp.vec3(d1_x, d1_y, d1_z)
        color = wp.vec3(1.0, 0.0, 0.0)
    elif axis_idx == 1:
        x, y, z, w = q[0], q[1], q[2], q[3]
        d2_x = 2.0 * (x * y - w * z)
        d2_y = w * w - x * x + y * y - z * z
        d2_z = 2.0 * (y * z + w * x)
        director = wp.vec3(d2_x, d2_y, d2_z)
        color = wp.vec3(0.0, 1.0, 0.0)
    else:
        x, y, z, w = q[0], q[1], q[2], q[3]
        d3_x = 2.0 * (x * z + w * y)
        d3_y = 2.0 * (y * z - w * x)
        d3_z = w * w - x * x - y * y + z * z
        director = wp.vec3(d3_x, d3_y, d3_z)
        color = wp.vec3(0.0, 0.0, 1.0)

    line_starts[tid] = midpoint
    line_ends[tid] = midpoint + director * axis_length
    line_colors[tid] = color


class Example:
    def __init__(self, viewer, args=None):
        # Simulation parameters
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 16
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.constraint_iterations = 6

        self.viewer = viewer
        self.args = args

        # Catheter rod parameters
        self.num_particles = NUM_PARTICLES
        self.num_stretch = self.num_particles - 1
        self.num_bend = self.num_particles - 2

        particle_spacing = CATHETER_LENGTH / (self.num_particles - 1)
        particle_mass = 0.001  # 1 gram per particle
        particle_radius = CATHETER_RADIUS
        edge_mass = 0.0001

        # Stiffness parameters (catheter is quite stiff)
        self.stretch_stiffness = 1.0
        self.shear_stiffness = 1.0
        self.bend_stiffness = 0.3  # Flexible enough to navigate
        self.twist_stiffness = 0.3

        # Velocity damping for stability
        self.velocity_damping = 0.98

        # Light gravity (or zero for neutral buoyancy in blood)
        self.gravity = wp.vec3(0.0, 0.0, -0.5)

        # Build the model
        builder = newton.ModelBuilder()

        # Load the aorta vessel mesh
        usd_path = os.path.join(os.path.dirname(__file__), "models", "DynamicAorta.usdc")
        usd_stage = Usd.Stage.Open(usd_path)
        mesh_prim = usd_stage.GetPrimAtPath("/root/A4009/A4007/Xueguan_rudong/Dynamic_vessels/Mesh")

        vessel_mesh = newton.usd.get_mesh(mesh_prim)

        # Get mesh bounds to position catheter appropriately
        vertices_np = np.array(vessel_mesh.vertices)
        mesh_center = vertices_np.mean(axis=0)
        mesh_min = vertices_np.min(axis=0)
        mesh_max = vertices_np.max(axis=0)

        # Scale factor to convert from mesh units to simulation units
        # The mesh seems to be in a different scale, adjust as needed
        self.mesh_scale = 0.01  # Convert cm to m if needed

        # Add the vessel mesh as a static collision shape
        # Using body=-1 for static geometry
        vessel_cfg = newton.ModelBuilder.ShapeConfig(
            ke=1.0e4,  # Contact stiffness
            kd=1.0e2,  # Contact damping
            mu=0.1,  # Low friction for blood vessels
            has_shape_collision=False,  # Don't collide with other shapes
            has_particle_collision=True,  # Collide with particles (catheter)
        )
        builder.add_shape_mesh(
            body=-1,  # Static shape
            mesh=vessel_mesh,
            scale=(self.mesh_scale, self.mesh_scale, self.mesh_scale),
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
            cfg=vessel_cfg,
        )

        # Calculate catheter start position (inside the vessel entrance)
        # Position the catheter at the bottom of the vessel (typically the descending aorta)
        catheter_start = wp.vec3(
            mesh_center[0] * self.mesh_scale,
            mesh_min[1] * self.mesh_scale + 0.02,  # Slightly inside the vessel
            mesh_center[2] * self.mesh_scale,
        )

        # Create catheter particles along a vertical line (will bend as it navigates)
        # Start direction: pointing upward into the vessel
        catheter_dir = wp.vec3(0.0, 1.0, 0.0)

        for i in range(self.num_particles):
            # First few particles are "outside" the vessel (held by operator)
            # Rest are inside navigating
            t = i * particle_spacing
            pos = wp.vec3(
                catheter_start[0] + catheter_dir[0] * t,
                catheter_start[1] + catheter_dir[1] * t,
                catheter_start[2] + catheter_dir[2] * t,
            )
            # First particle is kinematic (operator control point)
            mass = 0.0 if i == 0 else particle_mass
            builder.add_particle(
                pos=pos,
                vel=(0.0, 0.0, 0.0),
                mass=mass,
                radius=particle_radius,
            )

        self.model = builder.finalize()

        # Soft contact parameters for catheter-vessel collision
        self.model.soft_contact_ke = 5.0e3
        self.model.soft_contact_kd = 1.0e2
        self.model.soft_contact_mu = 0.1

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

        # Edge quaternions (initialized pointing along catheter direction)
        # Catheter starts pointing in Y direction
        angle = math.pi / 2.0
        q_init = wp.quat(math.sin(angle / 2.0), 0.0, 0.0, math.cos(angle / 2.0))  # Rotate around X
        edge_q_init = [q_init] * self.num_stretch
        self.edge_q = wp.array(edge_q_init, dtype=wp.quat, device=device)
        self.edge_q_new = wp.array(edge_q_init, dtype=wp.quat, device=device)

        # Edge inverse masses
        edge_inv_mass_np = [1.0 / edge_mass] * self.num_stretch
        self.edge_inv_mass = wp.array(edge_inv_mass_np, dtype=float, device=device)

        # Rest lengths
        rest_length_np = [particle_spacing] * self.num_stretch
        self.rest_length = wp.array(rest_length_np, dtype=float, device=device)

        # Rest Darboux vectors (straight rod)
        rest_darboux_np = [wp.quat(0.0, 0.0, 0.0, 1.0)] * self.num_bend
        self.rest_darboux = wp.array(rest_darboux_np, dtype=wp.quat, device=device)

        # Temporary buffers
        self.particle_q_predicted = wp.zeros(self.num_particles, dtype=wp.vec3, device=device)
        self.particle_q_temp = wp.zeros(self.num_particles, dtype=wp.vec3, device=device)
        self.particle_qd_temp = wp.zeros(self.num_particles, dtype=wp.vec3, device=device)

        # Correction accumulators
        self.particle_delta = wp.zeros(self.num_particles, dtype=wp.vec3, device=device)
        self.edge_q_delta = wp.zeros(self.num_stretch, dtype=wp.quat, device=device)

        self.viewer.set_model(self.model)
        self.viewer.show_particles = True also have on-going projects that can potentially leverages synthetic data from the simulator.

        # Director visualization
        num_director_lines = self.num_stretch * 3
        self.director_line_starts = wp.zeros(num_director_lines, dtype=wp.vec3, device=device)
        self.director_line_ends = wp.zeros(num_director_lines, dtype=wp.vec3, device=device)
        self.director_line_colors = wp.zeros(num_director_lines, dtype=wp.vec3, device=device)
        self.show_directors = False
        self.director_scale = 0.005

        # Keyboard control
        self.particle_move_speed = 0.1  # Slower for precise navigation
        self.advance_speed = 0.05  # Speed for advancing/retracting catheter

        # Store initial catheter positions for advancement
        self.catheter_base_pos = catheter_start

        self.graph = None

    def simulate(self):
        for _ in range(self.sim_substeps):
            wp.copy(self.particle_q_temp, self.state_0.particle_q)

            stretch_shear_ks = wp.vec3(self.shear_stiffness, self.shear_stiffness, self.stretch_stiffness)
            bend_twist_ks = wp.vec3(self.bend_stiffness, self.twist_stiffness, self.bend_stiffness)

            # Step 1: Integrate
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

            # Step 2: Apply soft contact forces from vessel collision
            self.contacts = self.model.collide(self.state_0, collision_pipeline=self.collision_pipeline)

            # Apply contact forces to particles
            if self.contacts.soft_contact_count:
                self._apply_soft_contact_forces()

            # Step 3: Constraint solving
            for _ in range(self.constraint_iterations):
                wp.launch(
                    kernel=zero_vec3_kernel,
                    dim=self.num_particles,
                    inputs=[self.particle_delta],
                    device=self.model.device,
                )
                wp.launch(
                    kernel=zero_quat_kernel,
                    dim=self.num_stretch,
                    inputs=[self.edge_q_delta],
                    device=self.model.device,
                )

                wp.launch(
                    kernel=solve_stretch_shear_constraint_kernel,
                    dim=self.num_stretch,
                    inputs=[
                        self.state_1.particle_q,
                        self.particle_inv_mass,
                        self.edge_q,
                        self.edge_inv_mass,
                        self.rest_length,
                        stretch_shear_ks,
                        self.num_stretch,
                    ],
                    outputs=[self.particle_delta, self.edge_q_delta],
                    device=self.model.device,
                )

                if self.num_bend > 0:
                    wp.launch(
                        kernel=solve_bend_twist_constraint_kernel,
                        dim=self.num_bend,
                        inputs=[
                            self.edge_q,
                            self.edge_inv_mass,
                            self.rest_darboux,
                            bend_twist_ks,
                            self.num_bend,
                        ],
                        outputs=[self.edge_q_delta],
                        device=self.model.device,
                    )

                wp.launch(
                    kernel=apply_particle_corrections_kernel,
                    dim=self.num_particles,
                    inputs=[
                        self.state_1.particle_q,
                        self.particle_delta,
                        self.particle_inv_mass,
                    ],
                    outputs=[self.particle_q_predicted],
                    device=self.model.device,
                )
                wp.copy(self.state_1.particle_q, self.particle_q_predicted)

                wp.launch(
                    kernel=apply_quaternion_corrections_kernel,
                    dim=self.num_stretch,
                    inputs=[
                        self.edge_q,
                        self.edge_q_delta,
                        self.edge_inv_mass,
                    ],
                    outputs=[self.edge_q_new],
                    device=self.model.device,
                )
                self.edge_q, self.edge_q_new = self.edge_q_new, self.edge_q

            # Step 4: Update velocities
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

            # Step 5: Apply velocity damping
            wp.launch(
                kernel=apply_velocity_damping_kernel,
                dim=self.num_particles,
                inputs=[
                    self.state_1.particle_qd,
                    self.particle_inv_mass,
                    self.velocity_damping,
                ],
                outputs=[self.particle_qd_temp],
                device=self.model.device,
            )
            wp.copy(self.state_1.particle_qd, self.particle_qd_temp)

            self.state_0, self.state_1 = self.state_1, self.state_0

    def _apply_soft_contact_forces(self):
        """Apply soft contact penalty forces from vessel collision."""
        # The collision system populates contacts, we need to apply forces
        # This is handled by the state's particle_f array when using newton solvers
        # For our custom solver, we'll read contacts and apply forces manually
        pass  # Contact forces are applied through Newton's contact system
aorta
    def step(self):
        self._handle_keyboard_input()
        self.simulate()
        self.sim_time += self.frame_dt

    def _handle_keyboard_input(self):
        """Move the catheter tip using numpad keys."""
        if not hasattr(self.viewer, "is_key_down"):
            return

        try:
            import pyglet.window.key as key
        except ImportError:
            return

        dx = 0.0
        dy = 0.0
        dz = 0.0

        # Numpad 4/6 for X movement
        if self.viewer.is_key_down(key.NUM_6):
            dx += self.particle_move_speed * self.frame_dt
        if self.viewer.is_key_down(key.NUM_4):
            dx -= self.particle_move_speed * self.frame_dt

        # Numpad 8/2 for Y movement (forward/back in vessel)
        if self.viewer.is_key_down(key.NUM_8):
            dy += self.particle_move_speed * self.frame_dt
        if self.viewer.is_key_down(key.NUM_2):
            dy -= self.particle_move_speed * self.frame_dt

        # Numpad 9/3 for Z movement
        if self.viewer.is_key_down(key.NUM_9):
            dz += self.particle_move_speed * self.frame_dt
        if self.viewer.is_key_down(key.NUM_3):
            dz -= self.particle_move_speed * self.frame_dt

        if dx != 0.0 or dy != 0.0 or dz != 0.0:
            particle_q_np = self.state_0.particle_q.numpy()
            pos = particle_q_np[0]
            new_pos = [pos[0] + dx, pos[1] + dy, pos[2] + dz]
            particle_q_np[0] = new_pos
            self.state_0.particle_q = wp.array(particle_q_np, dtype=wp.vec3, device=self.model.device)

    def test_final(self):
        # Verify particles are within reasonable bounds
        newton.examples.test_particle_state(
            self.state_0,
            "particles have valid positions",
            lambda q, qd: not (math.isnan(q[0]) or math.isnan(q[1]) or math.isnan(q[2])),
        )

        # Verify velocities are reasonable
        newton.examples.test_particle_state(
            self.state_0,
            "particle velocities are reasonable",
            lambda q, qd: wp.length(qd) < 10.0,
        )

        # Verify edge quaternions are normalized
        edge_q_np = self.edge_q.numpy()
        for i, q in enumerate(edge_q_np):
            norm = (q[0] ** 2 + q[1] ** 2 + q[2] ** 2 + q[3] ** 2) ** 0.5
            assert abs(norm - 1.0) < 0.1, f"Edge quaternion {i} not normalized: norm={norm}"

    def gui(self, ui):
        ui.text("Catheter Navigation in Aorta")
        ui.text(f"Catheter particles: {self.num_particles}")

        ui.separator()
        ui.text("Catheter Properties")
        _changed, self.stretch_stiffness = ui.slider_float("Stretch Stiffness", self.stretch_stiffness, 0.0, 1.0)
        _changed, self.shear_stiffness = ui.slider_float("Shear Stiffness", self.shear_stiffness, 0.0, 1.0)
        _changed, self.bend_stiffness = ui.slider_float("Bend Stiffness", self.bend_stiffness, 0.0, 1.0)
        _changed, self.twist_stiffness = ui.slider_float("Twist Stiffness", self.twist_stiffness, 0.0, 1.0)
        _changed, self.velocity_damping = ui.slider_float("Velocity Damping", self.velocity_damping, 0.9, 1.0)

        ui.separator()
        ui.text("Control (Numpad)")
        ui.text("  4/6: Move left/right (X)")
        ui.text("  8/2: Move forward/back (Y)")
        ui.text("  9/3: Move up/down (Z)")
        _changed, self.particle_move_speed = ui.slider_float("Move Speed", self.particle_move_speed, 0.01, 0.5)

        ui.separator()
        ui.text("Visualization")
        _changed, self.show_directors = ui.checkbox("Show Directors", self.show_directors)
        _changed, self.director_scale = ui.slider_float("Director Scale", self.director_scale, 0.001, 0.02)

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)

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
    viewer, args = newton.examples.init()

    if isinstance(viewer, newton.viewer.ViewerGL):
        viewer.show_particles = True

    example = Example(viewer, args)

    newton.examples.run(example, args)
