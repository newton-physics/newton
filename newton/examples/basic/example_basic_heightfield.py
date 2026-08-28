# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Basic Heightfield
#
# Demonstrates heightfield terrain with objects dropped onto it.
# Supports XPBD, MuJoCo, and Kamino DVI solvers.
#
# Command: uv run -m newton.examples basic_heightfield
# MuJoCo: uv run -m newton.examples basic_heightfield --solver mujoco
# Kamino DVI: uv run -m newton.examples basic_heightfield --solver kamino
#
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples


class Example:
    def __init__(self, viewer, args):
        newton.use_coord_layout_targets = True
        self.viewer = viewer
        self.solver_type = args.solver if hasattr(args, "solver") and args.solver else "xpbd"

        self.fps = 100
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 3 if self.solver_type == "kamino" else 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        builder = newton.ModelBuilder()
        if self.solver_type == "kamino":
            newton.solvers.SolverKamino.register_custom_attributes(builder)

        # Create a wave-like heightfield terrain
        nrow, ncol = 50, 50
        hx, hy = 5.0, 5.0
        x = np.linspace(-hx, hx, ncol)
        y = np.linspace(-hy, hy, nrow)
        xx, yy = np.meshgrid(x, y)
        elevation = np.sin(xx * 1.0) * np.cos(yy * 1.0) * 0.5

        hfield = newton.Heightfield(
            data=elevation,
            nrow=nrow,
            ncol=ncol,
            hx=hx,
            hy=hy,
        )
        builder.add_shape_heightfield(heightfield=hfield)

        # Drop several spheres onto the terrain
        drop_z = 1.0
        self.sphere_bodies = []
        positions = [
            (-2.0, -2.0),
            (0.0, 0.0),
            (2.0, 2.0),
            (-1.0, 1.5),
            (1.5, -1.0),
        ]
        for x_pos, y_pos in positions:
            body = builder.add_body(
                xform=wp.transform(p=wp.vec3(x_pos, y_pos, drop_z), q=wp.quat_identity()),
            )
            builder.add_shape_sphere(body=body, radius=0.3)
            self.sphere_bodies.append(body)

        self.model = builder.finalize()

        self.use_mujoco_contacts = self.solver_type == "mujoco"
        if self.solver_type == "mujoco":
            self.solver = newton.solvers.SolverMuJoCo(self.model)
            self.contacts = newton.Contacts(self.solver.get_max_contact_count(), 0)
            self.collision_pipeline = None
        elif self.solver_type == "kamino":
            solver_config = newton.solvers.SolverKamino.Config.from_model(
                self.model,
                dynamics_solver="dvi",
                sparse_dynamics=True,
                sparse_jacobian=True,
            )
            solver_config.use_collision_detector = True
            solver_config.integrator = "moreau"
            solver_config.dvi.max_alternating_iterations = 8
            self.solver = newton.solvers.SolverKamino(self.model, config=solver_config)
            self.collision_pipeline = None
            self.contacts = newton.Contacts(
                self.model.rigid_contact_max,
                0,
                device=self.model.device,
                requested_attributes=self.model.get_requested_contact_attributes(),
            )
        else:
            self.solver = newton.solvers.SolverXPBD(self.model, iterations=10)
            self.collision_pipeline = newton.CollisionPipeline(self.model)
            self.contacts = self.collision_pipeline.contacts()

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        self.viewer.set_model(self.model)

        self.capture()

    def capture(self):
        if wp.get_device().is_cuda and not wp.config.verify_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self):
        contacts = None if self.solver_type == "kamino" else self.contacts
        if self.collision_pipeline is not None:
            self.collision_pipeline.collide(self.state_0, self.contacts)
        for substep in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, contacts, self.sim_dt)
            if self.sim_substeps % 2 == 1 and substep == self.sim_substeps - 1:
                self.state_0.assign(self.state_1)
            else:
                self.state_0, self.state_1 = self.state_1, self.state_0
        if self.use_mujoco_contacts:
            self.solver.update_contacts(self.contacts, self.state_0)

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        if self.contacts is not None:
            if self.solver_type == "kamino" and self.viewer.show_contacts:
                self.solver.update_contacts(self.contacts, self.state_0)
            self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        """Verify all spheres are resting on the heightfield (not fallen through)."""
        body_q = self.state_0.body_q.numpy()
        for body_idx in self.sphere_bodies:
            z = float(body_q[body_idx, 2])
            assert z > -1.0, f"Sphere body {body_idx} fell through heightfield: z={z:.4f}"


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument(
        "--solver",
        type=str,
        default="xpbd",
        choices=["xpbd", "mujoco", "kamino"],
        help="Solver type: xpbd (default), mujoco, or kamino",
    )
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
