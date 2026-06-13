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

import math

import numpy as np
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
    "fs": 20,
    "kamino": 5,
}
SOLVER_CHOICES = ("xpbd", "vbd", "mujoco", "fs", "kamino")
NATIVE_CONTACT_SOLVERS = {"mujoco", "kamino"}


def _validate_body_state(model: newton.Model, state: newton.State, *, max_abs_pos: float, min_z: float):
    if state.body_q is None:
        raise RuntimeError("Body state is not available.")
    body_q = state.body_q.numpy()
    if not np.all(np.isfinite(body_q)):
        raise ValueError("NaN/Inf in body transforms.")
    pos = body_q[:, 0:3]
    if np.max(np.abs(pos)) > max_abs_pos:
        raise ValueError(f"Body moved outside expected bounds for {model.body_count} bodies.")
    if np.min(pos[:, 2]) < min_z:
        raise ValueError("Body fell below the expected lower Z bound.")


def _look_at_z_up(pos: wp.vec3, target: wp.vec3) -> tuple[float, float]:
    dx = float(target[0] - pos[0])
    dy = float(target[1] - pos[1])
    dz = float(target[2] - pos[2])
    length = math.sqrt(dx * dx + dy * dy + dz * dz)
    pitch = math.degrees(math.asin(dz / length))
    yaw = math.degrees(math.atan2(dy, dx))
    return pitch, yaw


def _set_camera_look_at(viewer, pos: wp.vec3, target: wp.vec3):
    pitch, yaw = _look_at_z_up(pos, target)
    viewer.set_camera(pos=pos, pitch=pitch, yaw=yaw)
    camera = getattr(viewer, "camera", None)
    if camera is not None and hasattr(camera, "look_at"):
        camera.look_at(target)


def _make_frustum_mesh(bottom_radius: float, top_radius: float, height: float, segments: int) -> newton.Mesh:
    vertices: list[list[float]] = []
    indices: list[int] = []

    for i in range(segments):
        angle = 2.0 * math.pi * i / segments
        c = math.cos(angle)
        s = math.sin(angle)
        vertices.append([bottom_radius * c, bottom_radius * s, 0.0])
        vertices.append([top_radius * c, top_radius * s, height])

    bottom_center = len(vertices)
    vertices.append([0.0, 0.0, 0.0])
    top_center = len(vertices)
    vertices.append([0.0, 0.0, height])

    for i in range(segments):
        n = (i + 1) % segments
        b0 = 2 * i
        t0 = 2 * i + 1
        b1 = 2 * n
        t1 = 2 * n + 1
        indices.extend([b0, t0, b1, b1, t0, t1])
        indices.extend([bottom_center, b0, b1])
        indices.extend([top_center, t1, t0])

    return newton.Mesh(
        vertices=np.array(vertices, dtype=np.float32),
        indices=np.array(indices, dtype=np.int32),
        compute_inertia=False,
    )


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.solver_name = str(getattr(args, "solver", "xpbd")).lower()

        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = SUBSTEPS[self.solver_name]
        self.sim_dt = self.frame_dt / self.sim_substeps

        builder = newton.ModelBuilder()
        builder.gravity = GRAVITY
        builder.rigid_gap = 0.001
        builder.default_shape_cfg.ke = 1.0e5
        builder.default_shape_cfg.kd = 0.0
        builder.default_shape_cfg.kf = 1.0e3
        builder.default_shape_cfg.mu = 0.8
        builder.default_shape_cfg.restitution = 0.0
        builder.add_ground_plane(cfg=builder.default_shape_cfg.copy())

        pedestal_mesh = _make_frustum_mesh(
            PEDESTAL_BOTTOM_RADIUS, PEDESTAL_TOP_RADIUS, PEDESTAL_HEIGHT, PEDESTAL_SEGMENTS
        )
        builder.add_shape_mesh(
            -1,
            xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity()),
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

        if self.solver_name == "xpbd":
            self.solver = newton.solvers.SolverXPBD(self.model, iterations=10, enable_restitution=False)
        elif self.solver_name == "vbd":
            self.solver = newton.solvers.SolverVBD(self.model, iterations=10, rigid_contact_hard=False)
        elif self.solver_name == "mujoco":
            self.solver = newton.solvers.SolverMuJoCo(self.model, njmax=2048, nconmax=1024, cone="elliptic")
        elif self.solver_name == "fs":
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
        self.contacts = self.model.contacts()

        self.viewer.set_model(self.model)
        _set_camera_look_at(
            self.viewer,
            pos=wp.vec3(-0.42, -0.42, 0.28),
            target=wp.vec3(0.02, 0.0, PEDESTAL_HEIGHT * 0.6),
        )

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
                self.model.collide(self.state_0, self.contacts)
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
        _validate_body_state(self.model, self.state_0, max_abs_pos=3.0, min_z=-0.5)

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
