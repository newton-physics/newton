# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Softbody Hanging
#
# This simulation demonstrates volumetric soft bodies (tetrahedral grids) hanging
# from fixed particles on the left side. Four grids with different damping values
# (1e-1 to 1e-4) showcase the effect of damping on Neo-Hookean elastic behavior.
#
# Command: uv run -m newton.examples softbody.example_softbody_hanging
#
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.solver_type = args.solver
        self.sim_time = 0.0
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 10
        self.iterations = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        if self.solver_type != "vbd":
            raise ValueError("The hanging softbody example only supports the VBD solver.")

        builder = newton.ModelBuilder()
        builder.add_ground_plane()

        # Grid dimensions
        dim_x = 12
        dim_y = 4
        dim_z = 4
        cell_size = 0.1

        # Create 4 grids with different damping values
        damping_values = [1e-1, 1e-2, 1e-3, 1e-4]
        spacing = 0.6  # Space between grids along Y-axis

        for i, k_damp in enumerate(damping_values):
            y_offset = i * spacing
            builder.add_soft_grid(
                pos=wp.vec3(0.0, 1.0 + y_offset, 1.0),
                rot=wp.quat_identity(),
                vel=wp.vec3(0.0, 0.0, 0.0),
                dim_x=dim_x,
                dim_y=dim_y,
                dim_z=dim_z,
                cell_x=cell_size,
                cell_y=cell_size,
                cell_z=cell_size,
                density=1.0e3,
                k_mu=1.0e5,
                k_lambda=1.0e5,
                k_damp=k_damp,
                fix_left=True,
            )

        # Color the mesh for VBD solver
        builder.color()

        self.model = builder.finalize()
        self.model.soft_contact_ke = 1.0e2
        self.model.soft_contact_kd = 0
        self.model.soft_contact_mu = 1.0

        self.solver = newton.solvers.SolverVBD(
            model=self.model,
            iterations=self.iterations,
            particle_enable_self_contact=False,
            particle_enable_tile_solve=False,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        self.contacts = self.model.contacts()

        self.viewer.set_model(self.model)

        self.capture()

    def capture(self):
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()

            # apply forces to the model
            self.viewer.apply_forces(self.state_0)

            self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            # swap states
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt

    def test_final(self):
        # Empirical state at 120 frames (CUDA, deterministic):
        #   centroid=[0.57, 2.10, 1.03], bbox_diag=2.78, max_vel=1.26
        #   min_pos=[0.0, 1.0, 0.23], max_pos=[1.23, 3.20, 1.40]
        particle_q = self.state_0.particle_q.numpy()
        particle_qd = self.state_0.particle_qd.numpy()

        min_pos = np.min(particle_q, axis=0)
        max_pos = np.max(particle_q, axis=0)
        centroid = np.mean(particle_q, axis=0)

        # Centroid check: observed [0.57, 2.10, 1.03], tolerance ±max(0.02, |val|*0.05)
        expected_centroid = np.array([0.57, 2.10, 1.03])
        centroid_tol = np.maximum(0.02, np.abs(expected_centroid) * 0.05)
        for i, axis in enumerate("xyz"):
            assert abs(centroid[i] - expected_centroid[i]) < centroid_tol[i], (
                f"Centroid {axis} drifted: {centroid[i]:.3f} vs expected {expected_centroid[i]:.3f}"
            )

        # Bounding box diagonal check: observed 2.78, allow up to 2.92
        bbox_size = np.linalg.norm(max_pos - min_pos)
        assert bbox_size < 2.92, f"Bounding box too large: {bbox_size:.2f}"

        # Particle position bounds (tightened from observed min/max with small margin)
        p_lower = wp.vec3(-0.02, 0.98, 0.21)
        p_upper = wp.vec3(1.25, 3.22, 1.42)
        newton.examples.test_particle_state(
            self.state_0,
            "particles are within the expected volume",
            lambda q, _qd: newton.math.vec_inside_limits(q, p_lower, p_upper),
        )

        # Min Z check: observed 0.23, allow down to 0.21
        assert min_pos[2] > 0.21, f"Excessive ground penetration: z_min={min_pos[2]:.4f}"

        # Velocity check: observed max 1.26, allow up to 1.89
        max_vel = np.max(np.linalg.norm(particle_qd, axis=1))
        assert max_vel < 1.89, f"Particle velocity too high: {max_vel:.2f}"

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument(
            "--solver",
            help="Type of solver (only 'vbd' supports volumetric soft bodies)",
            type=str,
            choices=["vbd"],
            default="vbd",
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
