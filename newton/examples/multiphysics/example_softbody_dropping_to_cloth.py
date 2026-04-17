# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Softbody Dropping to Cloth
#
# This simulation demonstrates a volumetric soft body (tetrahedral grid)
# dropping onto a cloth sheet. The soft body uses Neo-Hookean elasticity
# and deforms on impact with the cloth.
#
# Command: python -m newton.examples.multiphysics.example_softbody_dropping_to_cloth
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
            raise ValueError("The softbody dropping to cloth example only supports the VBD solver.")

        builder = newton.ModelBuilder()
        builder.add_ground_plane()

        # Add soft body (tetrahedral grid) at elevated position
        builder.add_soft_grid(
            pos=wp.vec3(0.0, 0.0, 2.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=6,
            dim_y=6,
            dim_z=3,
            cell_x=0.1,
            cell_y=0.1,
            cell_z=0.1,
            density=1.0e3,
            k_mu=1.0e5,
            k_lambda=1.0e5,
            k_damp=1e-3,
        )

        # Add cloth grid below the soft body
        builder.add_cloth_grid(
            pos=wp.vec3(-1.0, -1.0, 1.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            fix_left=True,
            fix_right=True,
            dim_x=40,
            dim_y=40,
            cell_x=0.05,
            cell_y=0.05,
            mass=0.0005,
            tri_ke=1e5,
            tri_ka=1e5,
            tri_kd=1e-5,
            edge_ke=0.01,
            edge_kd=1e-2,
            particle_radius=0.05,
        )

        # Color the mesh for VBD solver
        builder.color()

        self.model = builder.finalize()

        # Contact parameters
        self.model.soft_contact_ke = 1.0e5
        self.model.soft_contact_kd = 1e-5
        self.model.soft_contact_mu = 1.0

        self.solver = newton.solvers.SolverVBD(
            model=self.model,
            iterations=self.iterations,
            particle_enable_self_contact=True,
            particle_self_contact_radius=0.01,
            particle_self_contact_margin=0.02,
            particle_enable_tile_solve=True,
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
        # Empirical state at 200 frames (CUDA, 6 runs):
        #   centroid x in [-0.02, 0.02], y in [0.04, 0.20], z in [0.92, 0.97]
        #   bbox_diag in [2.85, 3.67], min_z in [0.09, 0.88], max_vel in [1.0, 5.2]
        #   Cloth sag introduces run-to-run variation in y and z.
        particle_q = self.state_0.particle_q.numpy()
        particle_qd = self.state_0.particle_qd.numpy()

        min_pos = np.min(particle_q, axis=0)
        max_pos = np.max(particle_q, axis=0)
        centroid = np.mean(particle_q, axis=0)
        bbox_size = np.linalg.norm(max_pos - min_pos)

        # Centroid check: wide tolerance to cover observed variation
        assert abs(centroid[0]) < 0.10, f"Centroid x drifted: {centroid[0]:.3f}"
        assert -0.05 < centroid[1] < 0.30, f"Centroid y out of range: {centroid[1]:.3f}"
        assert 0.85 < centroid[2] < 1.05, f"Centroid z out of range: {centroid[2]:.3f}"

        # Bounding box diagonal check: observed up to 3.67, allow up to 4.0
        assert bbox_size < 4.0, f"Bounding box too large: {bbox_size:.2f}"

        # Min Z check: observed as low as 0.09 when cloth sags, allow down to -0.05
        assert min_pos[2] > -0.05, f"Excessive ground penetration: z_min={min_pos[2]:.4f}"

        # Velocity check: observed up to 5.2, allow up to 8.0
        max_vel = np.max(np.linalg.norm(particle_qd, axis=1))
        assert max_vel < 8.0, f"Particle velocity too high: {max_vel:.2f}"

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
            help="Type of solver (only 'vbd' supports volumetric soft bodies in this example)",
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
