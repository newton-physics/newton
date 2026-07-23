# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Balance Bird
#
# Balances a small procedural toy body on a narrow pedestal. The body is
# built from primitive shapes so the example has no external mesh asset.
#
# Command: python -m newton.examples balance_bird
#
###########################################################################

from __future__ import annotations

import warp as wp

import newton
import newton.examples

PEDESTAL_BOTTOM_RADIUS = 0.08
PEDESTAL_TOP_RADIUS = 0.006
PEDESTAL_HEIGHT = 0.12
PEDESTAL_SEGMENTS = 32
TIP_RADIUS = 0.008
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
        builder.default_shape_cfg.kf = 1.0e3
        builder.default_shape_cfg.mu = 0.8
        builder.default_shape_cfg.restitution = 0.0
        builder.add_ground_plane(cfg=builder.default_shape_cfg.copy())

        pedestal_mesh = newton.Mesh.create_cylinder(
            PEDESTAL_BOTTOM_RADIUS,
            PEDESTAL_HEIGHT / 2.0,
            up_axis=newton.Axis.Z,
            segments=PEDESTAL_SEGMENTS,
            top_radius=PEDESTAL_TOP_RADIUS,
            compute_inertia=False,
        )
        builder.add_shape_mesh(
            -1,
            xform=wp.transform(p=wp.vec3(0.0, 0.0, PEDESTAL_HEIGHT / 2.0), q=wp.quat_identity()),
            mesh=pedestal_mesh,
            cfg=builder.default_shape_cfg.copy(),
            color=wp.vec3(0.55, 0.48, 0.32),
        )

        body = builder.add_body(
            xform=wp.transform(p=wp.vec3(0.0, 0.0, PEDESTAL_HEIGHT + TIP_RADIUS + 0.001), q=wp.quat_identity()),
            label="balance_bird",
        )

        light_cfg = builder.default_shape_cfg.copy()
        light_cfg.density = 350.0
        heavy_cfg = builder.default_shape_cfg.copy()
        heavy_cfg.density = 1800.0

        # The two lower weights place the center of mass below the contact tip.
        builder.add_shape_sphere(body, radius=TIP_RADIUS, cfg=light_cfg, color=wp.vec3(0.9, 0.35, 0.1))
        builder.add_shape_box(
            body,
            xform=wp.transform(p=wp.vec3(0.05, 0.0, 0.025), q=wp.quat_identity()),
            hx=0.065,
            hy=0.022,
            hz=0.018,
            cfg=light_cfg,
            color=wp.vec3(0.25, 0.45, 0.85),
        )
        builder.add_shape_box(
            body,
            xform=wp.transform(p=wp.vec3(0.02, 0.0, 0.02), q=wp.quat_identity()),
            hx=0.018,
            hy=0.19,
            hz=0.01,
            cfg=light_cfg,
            color=wp.vec3(0.15, 0.65, 0.8),
        )
        builder.add_shape_box(
            body,
            xform=wp.transform(p=wp.vec3(0.12, 0.0, 0.035), q=wp.quat_identity()),
            hx=0.04,
            hy=0.055,
            hz=0.012,
            cfg=light_cfg,
            color=wp.vec3(0.8, 0.35, 0.25),
        )
        builder.add_shape_sphere(
            body,
            xform=wp.transform(p=wp.vec3(-0.055, 0.145, -0.075), q=wp.quat_identity()),
            radius=0.025,
            cfg=heavy_cfg,
            color=wp.vec3(0.1, 0.2, 0.7),
        )
        builder.add_shape_sphere(
            body,
            xform=wp.transform(p=wp.vec3(-0.055, -0.145, -0.075), q=wp.quat_identity()),
            radius=0.025,
            cfg=heavy_cfg,
            color=wp.vec3(0.1, 0.2, 0.7),
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
            self.solver = newton.solvers.SolverXPBD(self.model, iterations=10, enable_restitution=False)
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
        self.viewer.set_camera(pos=wp.vec3(-0.42, -0.42, 0.28), pitch=-18.9, yaw=43.7)

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
            "balance bird remains within scene bounds",
            lambda q, qd: abs(q[0]) < 3.0 and abs(q[1]) < 3.0 and -0.5 < q[2] < 3.0,
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
        parser.add_argument("--solver", default="xpbd", choices=SOLVER_CHOICES, help="Solver used for the balance toy.")
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
