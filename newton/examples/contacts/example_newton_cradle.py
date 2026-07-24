# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Newton Cradle
#
# Builds a five-ball Newton's cradle with revolute-joint pendulums and
# sphere-sphere contact.
#
# Command: python -m newton.examples newton_cradle
#
###########################################################################

from __future__ import annotations

import math

import warp as wp

import newton
import newton.examples

NUM_BALLS = 5
BALL_MASS = 1.0
BALL_RADIUS = 0.05
STRING_LENGTH = 1.0
INITIAL_ANGLE = math.radians(45.0)
GRAVITY = -9.81

SUBSTEPS = {
    "xpbd": 10,
    "vbd": 10,
    "mujoco": 10,
    "featherstone": 20,
    "kamino": 5,
}
SOLVER_CHOICES = ("xpbd", "vbd", "mujoco", "featherstone", "kamino")
NATIVE_CONTACT_SOLVERS = {"mujoco", "kamino"}


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.solver_name = str(getattr(args, "solver", "xpbd")).lower()

        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = SUBSTEPS[self.solver_name]
        self.sim_dt = self.frame_dt / self.sim_substeps

        builder = newton.ModelBuilder(gravity=(0.0, 0.0, GRAVITY))
        builder.rigid_gap = 0.001
        builder.default_shape_cfg.ke = 1.0e5
        builder.default_shape_cfg.kd = 0.0
        builder.default_shape_cfg.kf = 0.0
        builder.default_shape_cfg.mu = 2.0e-5 if self.solver_name == "mujoco" else 0.0
        builder.default_shape_cfg.restitution = 1.0

        total_width = (NUM_BALLS - 1) * 2.0 * BALL_RADIUS
        x_start = -0.5 * total_width
        joint_indices = []

        for i in range(NUM_BALLS):
            pivot_x = x_start + i * 2.0 * BALL_RADIUS
            angle = INITIAL_ANGLE if i == 0 else 0.0
            ball_pos = wp.vec3(
                pivot_x + STRING_LENGTH * math.sin(angle),
                0.0,
                -STRING_LENGTH * math.cos(angle),
            )

            inertia = 0.4 * BALL_MASS * BALL_RADIUS * BALL_RADIUS
            body = builder.add_link(
                xform=wp.transform(p=ball_pos, q=wp.quat_identity()),
                mass=BALL_MASS,
                inertia=wp.mat33(inertia, 0.0, 0.0, 0.0, inertia, 0.0, 0.0, 0.0, inertia),
                com=wp.vec3(0.0, 0.0, 0.0),
                lock_inertia=True,
                label=f"ball_{i}",
            )

            color_t = i / max(NUM_BALLS - 1, 1)
            builder.add_shape_sphere(
                body,
                radius=BALL_RADIUS,
                color=wp.vec3(0.05 + 0.1 * color_t, 0.2 + 0.6 * color_t, 0.9 - 0.55 * color_t),
            )
            builder.add_shape_capsule(
                body,
                xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.5 * STRING_LENGTH), q=wp.quat_identity()),
                radius=0.004,
                half_height=0.5 * STRING_LENGTH - BALL_RADIUS,
                as_site=True,
                color=wp.vec3(0.55, 0.45, 0.35),
            )

            joint = builder.add_joint_revolute(
                parent=-1,
                child=body,
                axis=wp.vec3(0.0, 1.0, 0.0),
                parent_xform=wp.transform(p=wp.vec3(pivot_x, 0.0, 0.0), q=wp.quat_identity()),
                child_xform=wp.transform(p=wp.vec3(0.0, 0.0, STRING_LENGTH), q=wp.quat_identity()),
                limit_lower=-math.pi,
                limit_upper=math.pi,
                limit_ke=0.0,
                limit_kd=0.0,
                label=f"string_{i}",
            )
            joint_indices.append(joint)

        builder.add_shape_box(
            -1,
            xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity()),
            hx=0.5 * total_width + 0.12,
            hy=0.015,
            hz=0.015,
            color=wp.vec3(0.3, 0.3, 0.3),
        )

        builder.add_articulation(joint_indices, label="cradle")
        builder.color()
        self.model = builder.finalize()

        joint_q = self.model.joint_q.numpy()
        joint_q[0] = INITIAL_ANGLE
        self.model.joint_q.assign(joint_q)

        # Keep model.body_q synchronized before solver construction; VBD reads
        # these transforms during setup.
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.model)

        if self.solver_name in NATIVE_CONTACT_SOLVERS:
            self.collision_pipeline = None
            self.contacts = None
        else:
            self.collision_pipeline = newton.CollisionPipeline(self.model)
            self.contacts = self.collision_pipeline.contacts()

        if self.solver_name == "xpbd":
            self.solver = newton.solvers.SolverXPBD(self.model, iterations=10, enable_restitution=True)
        elif self.solver_name == "vbd":
            self.solver = newton.solvers.SolverVBD(self.model, iterations=10, rigid_contact_hard=False)
        elif self.solver_name == "mujoco":
            self.solver = newton.solvers.SolverMuJoCo(self.model, njmax=2048, nconmax=1024, cone="elliptic")
        elif self.solver_name == "featherstone":
            self.solver = newton.solvers.SolverFeatherstone(self.model, angular_damping=0.0)
        elif self.solver_name == "kamino":
            solver_config = newton.solvers.SolverKamino.Config.from_model(self.model)
            solver_config.use_collision_detector = True
            self.solver = newton.solvers.SolverKamino(self.model, config=solver_config)
        else:
            raise ValueError(f"Unknown solver: {self.solver_name}")
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(-0.25, -2.35, 0.25), pitch=-17.7, yaw=90.0)

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
            if self.solver_name in NATIVE_CONTACT_SOLVERS:
                contacts = None
            else:
                self.collision_pipeline.collide(self.state_0, self.contacts)
                contacts = self.contacts
            self.solver.step(self.state_0, self.state_1, self.control, contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def test_final(self):
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "cradle remains in its planar workspace",
            lambda q, qd: abs(q[0]) < 1.25 and abs(q[1]) < 1.0e-3 and abs(q[2]) < 1.1,
        )

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        if self.solver_name not in NATIVE_CONTACT_SOLVERS:
            self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--solver", default="xpbd", choices=SOLVER_CHOICES, help="Solver used for the cradle.")
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
