# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Basic Dzhanibekov
#
# Demonstrates the Dzhanibekov effect on a free T-shaped rigid body.
# The body is composed of a horizontal bar (一) and a vertical stem (|).
# The initial angular velocity is about the stem axis, gravity is
# disabled, and the body is advanced by a rigid-body solver.
#
# Command: python -m newton.examples basic_dzhanibekov
# XPBD: python -m newton.examples basic_dzhanibekov --solver xpbd
# MuJoCo: python -m newton.examples basic_dzhanibekov --solver mujoco
#
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples

_STEM_RADIUS = 0.10
_STEM_HALF_HEIGHT = 0.5
_STEM_DENSITY = 50.0

_BAR_RADIUS = 0.20
_BAR_HALF_HEIGHT = 1.0
_BAR_DENSITY = 100.0


class Example:
    def __init__(self, viewer, args):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 4
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer
        self.solver_type = args.solver if hasattr(args, "solver") and args.solver else "vbd"
        self.min_stem_axis_y = 1.0

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)

        self.body = builder.add_body(
            xform=wp.transform(wp.vec3(0.0, 0.0, 2.0), wp.quat_identity()),
            label="dzhanibekov_t",
        )
        self.free_joint = len(builder.joint_type) - 1

        # Build T-shaped rigid body: bar (一) + stem (|)
        # Cylinders default to Z axis; rotate bar to X and stem to Y.
        q_bar = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), -wp.pi / 2.0)  # Z → X
        q_stem = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), wp.pi / 2.0)  # Z → Y

        bar_center = np.array((0.0, 0.0, 0.0), dtype=np.float32)
        stem_center = np.array((0.0, -(_BAR_RADIUS + _STEM_HALF_HEIGHT), 0.0), dtype=np.float32)
        bar_mass = 2.0 * wp.pi * _BAR_RADIUS**2 * _BAR_HALF_HEIGHT * _BAR_DENSITY
        stem_mass = 2.0 * wp.pi * _STEM_RADIUS**2 * _STEM_HALF_HEIGHT * _STEM_DENSITY
        center_of_mass = (bar_center * bar_mass + stem_center * stem_mass) / (bar_mass + stem_mass)

        cfg_bar = builder.default_shape_cfg.copy()
        cfg_bar.density = _BAR_DENSITY
        cfg_bar.has_shape_collision = False
        builder.add_shape_cylinder(
            self.body,
            xform=wp.transform(wp.vec3(*(bar_center - center_of_mass)), q_bar),
            radius=_BAR_RADIUS,
            half_height=_BAR_HALF_HEIGHT,
            cfg=cfg_bar,
            color=wp.vec3(0.75, 0.18, 0.12),
        )

        cfg_stem = builder.default_shape_cfg.copy()
        cfg_stem.density = _STEM_DENSITY
        cfg_stem.has_shape_collision = False
        builder.add_shape_cylinder(
            self.body,
            xform=wp.transform(wp.vec3(*(stem_center - center_of_mass)), q_stem),
            radius=_STEM_RADIUS,
            half_height=_STEM_HALF_HEIGHT,
            cfg=cfg_stem,
            color=wp.vec3(0.15, 0.42, 0.78),
        )

        builder.body_qd[self.body] = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 20.0, 0.0)
        builder.joint_qd = np.array(builder.body_qd).flatten().tolist()

        if self.solver_type == "vbd":
            builder.color()

        builder.add_ground_plane()

        self.model = builder.finalize()
        self.model.set_gravity((0.0, 0.0, 0.0))

        if self.solver_type == "vbd":
            self.solver = newton.solvers.SolverVBD(self.model, iterations=4)
            self.collision_pipeline = newton.CollisionPipeline(self.model)
            self.contacts = self.collision_pipeline.contacts()
        elif self.solver_type == "xpbd":
            self.solver = newton.solvers.SolverXPBD(self.model, iterations=10)
            self.collision_pipeline = newton.CollisionPipeline(self.model)
            self.contacts = self.collision_pipeline.contacts()
        elif self.solver_type == "mujoco":
            self.solver = newton.solvers.SolverMuJoCo(
                self.model,
                disable_contacts=True,
                solver="newton",
                integrator="implicitfast",
                iterations=10,
            )
            self.collision_pipeline = None
            self.contacts = None
        else:
            raise ValueError(f"Unknown solver type: {self.solver_type}. Choose from 'vbd', 'xpbd', or 'mujoco'.")

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        self.initial_body_q = self.state_0.body_q.numpy().copy()
        self.body_com = self.model.body_com.numpy()[self.body].copy()
        self.initial_com = (
            self.initial_body_q[self.body][:3]
            + np.array(wp.quat_to_matrix(wp.quat(*self.initial_body_q[self.body][3:7])), dtype=np.float32).reshape(3, 3)
            @ self.body_com
        )

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
            self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt
        body_q = self.state_0.body_q.numpy()
        local_y_world = np.array(wp.quat_to_matrix(wp.quat(*body_q[self.body][3:7])), dtype=np.float32).reshape(3, 3)[
            :, 1
        ]
        self.min_stem_axis_y = min(self.min_stem_axis_y, float(local_y_world[1]))

    def test_final(self):
        body_q = self.state_0.body_q.numpy()
        body_qd = self.state_0.body_qd.numpy()
        if not np.isfinite(body_q[self.body]).all() or not np.isfinite(body_qd[self.body]).all():
            raise ValueError("Dzhanibekov rigid body state contains non-finite values")
        final_com = (
            body_q[self.body][:3]
            + np.array(wp.quat_to_matrix(wp.quat(*body_q[self.body][3:7])), dtype=np.float32).reshape(3, 3)
            @ self.body_com
        )
        if np.linalg.norm(final_com - self.initial_com) >= 1.0e-3:
            raise ValueError("Free rigid body developed linear drift")
        if self.min_stem_axis_y >= 0.0:
            raise ValueError("Dzhanibekov flip did not occur")

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument(
        "--solver",
        type=str,
        default="vbd",
        choices=["vbd", "xpbd", "mujoco"],
        help="Solver type: vbd (default), xpbd, or mujoco.",
    )
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
