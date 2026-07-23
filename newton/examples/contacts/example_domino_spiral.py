# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Domino Spiral
#
# Places dominoes along a spiral and tilts the first one to start a
# contact-driven chain reaction.
#
# Command: python -m newton.examples domino_spiral
#
###########################################################################

from __future__ import annotations

import math

import warp as wp

import newton
import newton.examples

NUM_DOMINOES = 30
DOMINO_SPACING = 0.12
DOMINO_HALF = (0.06, 0.016, 0.18)
DOMINO_DENSITY = 580.0
SPIRAL_INNER_RADIUS = 0.35
SPIRAL_PITCH = 0.32
INITIAL_TILT = math.radians(15.0)
GRAVITY = -9.81

SUBSTEPS = {
    "xpbd": 10,
    "vbd": 10,
    "mujoco": 10,
    "featherstone": 10,
    "kamino": 5,
}
SOLVER_CHOICES = ("xpbd", "vbd", "mujoco", "featherstone", "kamino")
NATIVE_CONTACT_SOLVERS = {"mujoco", "kamino"}


def _rainbow_color(i: int, count: int) -> wp.vec3:
    t = i / max(count - 1, 1)
    return wp.vec3(0.9 * (1.0 - t) + 0.2 * t, 0.25 + 0.55 * t, 0.15 + 0.75 * (1.0 - abs(0.5 - t) * 2.0))


def _domino_spiral_pose(index: int) -> tuple[wp.vec3, wp.quat, float]:
    b = SPIRAL_PITCH / (2.0 * math.pi)
    theta = 0.0
    for _ in range(index):
        r = SPIRAL_INNER_RADIUS + b * theta
        theta += DOMINO_SPACING / math.sqrt(r * r + b * b)

    r = SPIRAL_INNER_RADIUS + b * theta
    x = r * math.cos(theta)
    y = r * math.sin(theta)

    tx = b * math.cos(theta) - r * math.sin(theta)
    ty = b * math.sin(theta) + r * math.cos(theta)
    yaw = math.atan2(-tx, ty)
    q_yaw = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), yaw)
    return wp.vec3(x, y, DOMINO_HALF[2]), q_yaw, r


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
        builder.default_shape_cfg.ke = 1.0e4
        builder.default_shape_cfg.kd = 0.0
        builder.default_shape_cfg.kf = 1.0e3
        builder.default_shape_cfg.mu = 1.0
        builder.default_shape_cfg.restitution = 0.15
        builder.default_shape_cfg.density = DOMINO_DENSITY
        builder.add_ground_plane(cfg=builder.default_shape_cfg.copy())

        hx, hy, hz = DOMINO_HALF
        for i in range(NUM_DOMINOES):
            pos, q_yaw, _ = _domino_spiral_pose(i)
            if i == 0:
                q = q_yaw * wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -INITIAL_TILT)
            else:
                q = q_yaw
            body = builder.add_body(
                xform=wp.transform(p=pos, q=q),
                label=f"domino_{i}",
            )
            builder.add_shape_box(
                body,
                hx=hx,
                hy=hy,
                hz=hz,
                cfg=builder.default_shape_cfg.copy(),
                color=_rainbow_color(i, NUM_DOMINOES),
            )

        builder.color()
        self.model = builder.finalize()

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
        self.viewer.set_camera(pos=wp.vec3(-1.16, -1.26, 1.06), pitch=-27.8, yaw=43.6)

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
            "dominoes remain within scene bounds",
            lambda q, qd: abs(q[0]) < 3.1 and abs(q[1]) < 3.1 and -0.5 < q[2] < 3.1,
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
        parser.add_argument("--solver", default="xpbd", choices=SOLVER_CHOICES, help="Solver used for the dominoes.")
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
