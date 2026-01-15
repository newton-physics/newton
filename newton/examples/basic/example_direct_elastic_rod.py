# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Direct elastic rod example using a Warp tile Cholesky solve to
# enforce a simplified stretch + bend/twist constraint per joint.
#
# Command: uv run -m newton.examples direct_elastic_rod

import math

import warp as wp

import newton
import newton.examples


BLOCK_DIM = 128
TILE = 8  # embed the 6x6 SPD system in an 8x8 tile for Warp's Cholesky


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
    tid = wp.tid()
    inv_mass = particle_inv_mass[tid]

    if inv_mass == 0.0:
        particle_q_new[tid] = particle_q[tid]
        particle_qd_new[tid] = particle_qd[tid]
        return

    v = particle_qd[tid] + gravity * dt
    x = particle_q[tid] + v * dt

    particle_q_new[tid] = x
    particle_qd_new[tid] = v


@wp.kernel
def zero_vec3_kernel(arr: wp.array(dtype=wp.vec3)):
    tid = wp.tid()
    arr[tid] = wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def zero_quat_kernel(arr: wp.array(dtype=wp.quat)):
    tid = wp.tid()
    arr[tid] = wp.quat(0.0, 0.0, 0.0, 0.0)


@wp.kernel
def assemble_direct_system_kernel(
    particle_q: wp.array(dtype=wp.vec3),
    particle_inv_mass: wp.array(dtype=float),
    edge_q: wp.array(dtype=wp.quat),
    edge_inv_mass: wp.array(dtype=float),
    edge_rest_length: wp.array(dtype=float),
    rest_darboux: wp.array(dtype=wp.quat),
    stretch_compliance: float,
    bend_compliance: float,
    system_matrix: wp.array2d(dtype=float),
    rhs: wp.array1d(dtype=float),
):
    """Assemble a per-joint 6x6 SPD system into an 8x8 tile."""
    tid = wp.tid()
    base = tid * TILE

    # Initialize tile to identity to keep SPD in unused rows/cols.
    for r in range(TILE):
        for c in range(TILE):
            system_matrix[base + r, c] = 1.0 if r == c else 0.0
    for k in range(TILE):
        rhs[base + k] = 0.0

    # Segment / joint indices
    p0 = particle_q[tid]
    p1 = particle_q[tid + 1]
    p2 = particle_q[tid + 2]

    q0 = edge_q[tid]
    q1 = edge_q[tid + 1]

    l0 = edge_rest_length[tid]
    l1 = edge_rest_length[tid + 1]
    avg_len = 0.5 * (l0 + l1)

    # Segment centers
    center0 = (p0 + p1) * 0.5
    center1 = (p1 + p2) * 0.5

    # Connectors: right end of segment 0 to left end of segment 1
    local_right0 = wp.vec3(0.5 * l0, 0.0, 0.0)
    local_left1 = wp.vec3(-0.5 * l1, 0.0, 0.0)
    connector0 = wp.quat_rotate(q0, local_right0) + center0
    connector1 = wp.quat_rotate(q1, local_left1) + center1

    stretch_violation = connector0 - connector1

    # Darboux vector approximation: 2/avg_len * (q0_conj * q1).vec
    q0_conj = wp.quat(-q0[0], -q0[1], -q0[2], q0[3])
    omega = wp.mul(q0_conj, q1)
    darboux = wp.vec3(omega[0], omega[1], omega[2]) * (2.0 / (avg_len + 1.0e-6))
    rest = rest_darboux[tid]
    bend_violation = darboux - wp.vec3(rest[0], rest[1], rest[2])

    rhs[base + 0] = -stretch_violation[0]
    rhs[base + 1] = -stretch_violation[1]
    rhs[base + 2] = -stretch_violation[2]
    rhs[base + 3] = -bend_violation[0]
    rhs[base + 4] = -bend_violation[1]
    rhs[base + 5] = -bend_violation[2]

    # Simple diagonal mass approximation for SPD matrix
    inv_m0 = particle_inv_mass[tid]
    inv_m1 = particle_inv_mass[tid + 1]
    inv_m2 = particle_inv_mass[tid + 2]
    inv_center0 = 0.5 * (inv_m0 + inv_m1)
    inv_center1 = 0.5 * (inv_m1 + inv_m2)

    diag_stretch = inv_center0 + inv_center1 + stretch_compliance
    diag_bend = edge_inv_mass[tid] + edge_inv_mass[tid + 1] + bend_compliance

    if diag_stretch < 1.0e-6:
        diag_stretch = 1.0e-6
    if diag_bend < 1.0e-6:
        diag_bend = 1.0e-6

    system_matrix[base + 0, 0] = diag_stretch
    system_matrix[base + 1, 1] = diag_stretch
    system_matrix[base + 2, 2] = diag_stretch
    system_matrix[base + 3, 3] = diag_bend
    system_matrix[base + 4, 4] = diag_bend
    system_matrix[base + 5, 5] = diag_bend


@wp.kernel
def cholesky_solve_kernel(
    A: wp.array2d(dtype=float),
    X: wp.array1d(dtype=float),
    Y: wp.array1d(dtype=float),
):
    """Factor and solve each tile independently."""
    a = wp.tile_load(A, shape=(TILE, TILE))
    l = wp.tile_cholesky(a)
    x = wp.tile_load(X, shape=TILE)
    y = wp.tile_cholesky_solve(l, x)
    wp.tile_store(Y, y)


@wp.kernel
def apply_direct_corrections_kernel(
    particle_inv_mass: wp.array(dtype=float),
    edge_inv_mass: wp.array(dtype=float),
    delta_lambda: wp.array1d(dtype=float),
    particle_delta: wp.array(dtype=wp.vec3),
    edge_q_delta: wp.array(dtype=wp.quat),
):
    """Distribute the solved stretch and bend updates."""
    tid = wp.tid()
    base = tid * TILE

    dl_stretch = wp.vec3(delta_lambda[base + 0], delta_lambda[base + 1], delta_lambda[base + 2])
    dl_bend = wp.vec3(delta_lambda[base + 3], delta_lambda[base + 4], delta_lambda[base + 5])

    inv_m0 = particle_inv_mass[tid]
    inv_m1 = particle_inv_mass[tid + 1]
    inv_m2 = particle_inv_mass[tid + 2]
    inv_center0 = 0.5 * (inv_m0 + inv_m1)
    inv_center1 = 0.5 * (inv_m1 + inv_m2)

    corr_center0 = dl_stretch * inv_center0
    corr_center1 = dl_stretch * (-inv_center1)

    # Distribute center corrections to endpoints (shared middle gets both).
    wp.atomic_add(particle_delta, tid, corr_center0 * 0.5)
    wp.atomic_add(particle_delta, tid + 1, corr_center0 * 0.5 + corr_center1 * 0.5)
    wp.atomic_add(particle_delta, tid + 2, corr_center1 * 0.5)

    inv_q0 = edge_inv_mass[tid]
    inv_q1 = edge_inv_mass[tid + 1]
    corr_q0 = wp.quat(dl_bend[0] * inv_q0, dl_bend[1] * inv_q0, dl_bend[2] * inv_q0, 0.0)
    corr_q1 = wp.quat(-dl_bend[0] * inv_q1, -dl_bend[1] * inv_q1, -dl_bend[2] * inv_q1, 0.0)

    wp.atomic_add(edge_q_delta, tid, corr_q0)
    wp.atomic_add(edge_q_delta, tid + 1, corr_q1)


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
    tid = wp.tid()

    inv_mass = particle_inv_mass[tid]
    if inv_mass == 0.0:
        particle_q_out[tid] = particle_q[tid]
        particle_qd_out[tid] = particle_qd[tid]
        return

    delta = particle_delta[tid]
    q_new = particle_q[tid] + delta
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
def solve_ground_collision_kernel(
    particle_q: wp.array(dtype=wp.vec3),
    particle_inv_mass: wp.array(dtype=float),
    particle_radius: wp.array(dtype=float),
    ground_level: float,
    particle_delta: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()

    inv_mass = particle_inv_mass[tid]
    if inv_mass == 0.0:
        return

    pos = particle_q[tid]
    radius = particle_radius[tid]
    min_z = ground_level + radius
    penetration = min_z - pos[2]

    if penetration > 0.0:
        correction = wp.vec3(0.0, 0.0, penetration)
        wp.atomic_add(particle_delta, tid, correction)


class Example:
    def __init__(self, viewer, args=None):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 8
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.constraint_iterations = 1
        self.sim_time = 0.0
        self.viewer = viewer
        self.args = args

        # Rod parameters
        self.num_particles = 48
        self.num_edges = self.num_particles - 1
        self.num_constraints = self.num_edges - 1  # stretch + bend/twist joints

        particle_spacing = 0.05
        particle_mass = 0.02
        edge_mass = 0.004
        particle_radius = 0.01
        start_height = 2.0

        self.stretch_compliance = 1.0e-6
        self.bend_compliance = 1.0e-5
        self.gravity = wp.vec3(0.0, 0.0, -9.81)

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

        inv_mass_np = [0.0] + [1.0 / particle_mass] * (self.num_particles - 1)
        self.particle_inv_mass = wp.array(inv_mass_np, dtype=float, device=device)

        # Initialize edge orientations along +x axis (rotate z->x)
        angle = math.pi / 2.0
        q_init = wp.quat(0.0, math.sin(angle / 2.0), 0.0, math.cos(angle / 2.0))
        edge_q_init = [q_init] * self.num_edges
        self.edge_q = wp.array(edge_q_init, dtype=wp.quat, device=device)
        self.edge_q_new = wp.array(edge_q_init, dtype=wp.quat, device=device)

        edge_inv_mass_np = [1.0 / edge_mass] * self.num_edges
        self.edge_inv_mass = wp.array(edge_inv_mass_np, dtype=float, device=device)

        rest_length_np = [particle_spacing] * self.num_edges
        self.edge_rest_length = wp.array(rest_length_np, dtype=float, device=device)

        rest_darboux_np = [wp.quat(0.0, 0.0, 0.0, 0.0)] * self.num_constraints
        self.rest_darboux = wp.array(rest_darboux_np, dtype=wp.quat, device=device)

        self.particle_delta = wp.zeros(self.num_particles, dtype=wp.vec3, device=device)
        self.edge_q_delta = wp.zeros(self.num_edges, dtype=wp.quat, device=device)

        # Buffers for tiled Cholesky solves
        self.system_matrix = wp.zeros((self.num_constraints * TILE, TILE), dtype=float, device=device)
        self.system_rhs = wp.zeros(self.num_constraints * TILE, dtype=float, device=device)
        self.system_solution = wp.zeros_like(self.system_rhs)

        self.viewer.set_model(self.model)
        self.viewer.show_particles = True
        self.graph = None

    def simulate(self):
        for _ in range(self.sim_substeps):
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

            self.state_0, self.state_1 = self.state_1, self.state_0

            for _ in range(self.constraint_iterations):
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

                wp.launch(
                    kernel=assemble_direct_system_kernel,
                    dim=self.num_constraints,
                    inputs=[
                        self.state_0.particle_q,
                        self.particle_inv_mass,
                        self.edge_q,
                        self.edge_inv_mass,
                        self.edge_rest_length,
                        self.rest_darboux,
                        self.stretch_compliance,
                        self.bend_compliance,
                        self.system_matrix,
                        self.system_rhs,
                    ],
                    device=self.model.device,
                )

                wp.launch_tiled(
                    kernel=cholesky_solve_kernel,
                    dim=[self.num_constraints, 1],
                    inputs=[self.system_matrix, self.system_rhs],
                    outputs=[self.system_solution],
                    block_dim=BLOCK_DIM,
                    device=self.model.device,
                )

                wp.launch(
                    kernel=apply_direct_corrections_kernel,
                    dim=self.num_constraints,
                    inputs=[
                        self.particle_inv_mass,
                        self.edge_inv_mass,
                        self.system_solution,
                    ],
                    outputs=[self.particle_delta, self.edge_q_delta],
                    device=self.model.device,
                )

                wp.launch(
                    kernel=solve_ground_collision_kernel,
                    dim=self.num_particles,
                    inputs=[
                        self.state_0.particle_q,
                        self.particle_inv_mass,
                        self.model.particle_radius,
                        0.0,
                    ],
                    outputs=[self.particle_delta],
                    device=self.model.device,
                )

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

                self.state_0, self.state_1 = self.state_1, self.state_0
                self.edge_q, self.edge_q_new = self.edge_q_new, self.edge_q

            self.contacts = self.model.collide(self.state_0, collision_pipeline=self.collision_pipeline)

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt

    def test_final(self):
        newton.examples.test_particle_state(
            self.state_0,
            "anchor particle is stationary",
            lambda q, qd: wp.length(qd) < 1.0,
            indices=[0],
        )

        newton.examples.test_particle_state(
            self.state_0,
            "particles are above the ground",
            lambda q, qd: q[2] >= -0.05,
        )

        p_lower = wp.vec3(-2.0, -2.0, -0.1)
        p_upper = wp.vec3(5.0, 2.0, 4.0)
        newton.examples.test_particle_state(
            self.state_0,
            "particles are within reasonable bounds",
            lambda q, qd: newton.utils.vec_inside_limits(q, p_lower, p_upper),
        )

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
    viewer, args = newton.examples.init()

    if isinstance(viewer, newton.viewer.ViewerGL):
        viewer.show_particles = True

    example = Example(viewer, args)
    newton.examples.run(example, args)
