# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Example Softbody Hanging

This simulation demonstrates volumetric soft bodies (tetrahedral grids) hanging
from fixed particles on the left side. Four grids with different damping values
(1e4 to 1e1) showcase the effect of damping on Neo-Hookean elastic behavior.
With ``--solver style3d``, the grids instead compare Young's moduli of
250000, 375000, 500000, and 1000000 Pa at Poisson ratio 0.48. Style3D uses
co-rotational tetrahedral elasticity and volume projection; material damping
is not supported in this mode.

Success criteria:
- 4 hanging elongated softbodies appear with different damping values.
- The softbodies are oscillating until reaching a steady state.
- With VBD, higher damping on the left produces earlier settling.
- With Style3D, all four grids remain bounded while deforming under gravity.

Runs:
- `python -m newton.examples softbody_hanging`
- `python -m newton.examples softbody_hanging --solver style3d`
"""

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

        builder = newton.ModelBuilder()
        if self.solver_type == "style3d":
            newton.solvers.SolverStyle3D.register_custom_attributes(builder)
        builder.add_ground_plane()

        # Grid dimensions
        dim_x = 12
        dim_y = 4
        dim_z = 4
        cell_size = 0.1

        # Create 4 grids with different damping values
        damping_values = [1e4, 1e3, 1e2, 1e1]
        spacing = 0.6  # Space between grids along Y-axis

        for i, damping in enumerate(damping_values):
            k_damp = damping
            mu = lam = 1.0e5
            if self.solver_type == "style3d":
                youngs = (2.5e5, 3.75e5, 5e5, 1e6)[i]
                nu = 0.48
                mu = youngs / (2.0 * (1.0 + nu))
                lam = youngs * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
                k_damp = 0.0
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
                k_mu=mu,
                k_lambda=lam,
                k_damp=k_damp,
                fix_left=True,
            )

        # Color the mesh for VBD solver
        if self.solver_type == "vbd":
            builder.color()

        self.model = builder.finalize()
        self.model.soft_contact_ke = 1.0e2
        self.model.soft_contact_kd = 0
        self.model.soft_contact_mu = 1.0

        if self.solver_type == "style3d":
            self.solver = newton.solvers.SolverStyle3D(
                self.model,
                config=newton.solvers.SolverStyle3D.Config(iterations=self.iterations, linear_iterations=15),
            )
            self.solver.collision = None
        else:
            self.solver = newton.solvers.SolverVBD(
                model=self.model,
                iterations=self.iterations,
                particle_enable_self_contact=False,
                particle_enable_tile_solve=False,
            )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        self.collision_pipeline = newton.CollisionPipeline(self.model)
        self.contacts = self.collision_pipeline.contacts()

        self.viewer.set_model(self.model)

        self.capture()

    def capture(self):
        self.graph = None
        if not self.model.device.is_cuda:
            return
        with wp.ScopedCapture() as capture:
            self.simulate()
        self.graph = capture.graph

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()

            # apply forces to the model
            self.viewer.apply_forces(self.state_0)

            self.collision_pipeline.collide(self.state_0, self.contacts)
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
        """Verify that the hanging grids remain within the expected volume."""
        # Test that particles are in a reasonable range (soft body may settle or deform)
        # We check that they haven't exploded or collapsed completely
        # 4 grids, each roughly 1.2 x 0.4 x 0.4 in size, positioned along Y-axis
        # Initial positions: Y from 1.0 to ~3.2, X from 0 to 1.2, Z around 1.0 to 1.4
        # With fix_left=True, grids hang and sag significantly towards the ground
        p_lower = wp.vec3(-1.0, -0.5, 0.0)
        p_upper = wp.vec3(3.0, 4.0, 3.0)
        newton.examples.test_particle_state(
            self.state_0,
            "particles are within a reasonable volume",
            lambda q, _qd: newton.math.vec_inside_limits(q, p_lower, p_upper),
        )

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
            help="Solver: VBD damping comparison or Style3D elastic stiffness comparison",
            type=str,
            choices=["vbd", "style3d"],
            default="vbd",
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
