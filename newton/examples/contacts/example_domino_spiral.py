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

import numpy as np
import warp as wp

import newton
import newton.examples

NUM_DOMINOES = 30
DOMINO_SPACING = 0.12
DOMINO_HALF = (0.06, 0.016, 0.18)
DOMINO_MASS = 0.8
SPIRAL_INNER_RADIUS = 0.35
SPIRAL_PITCH = 0.32
INITIAL_TILT = math.radians(15.0)
GRAVITY = -9.81

SUBSTEPS = {
    "xpbd": 10,
    "vbd": 10,
    "mujoco": 10,
    "fs": 10,
    "kamino": 5,
}
SOLVER_CHOICES = ("xpbd", "vbd", "mujoco", "fs", "kamino")
NATIVE_CONTACT_SOLVERS = {"mujoco", "kamino"}


def _box_inertia(mass: float, hx: float, hy: float, hz: float) -> wp.mat33:
    ixx = mass / 3.0 * (hy * hy + hz * hz)
    iyy = mass / 3.0 * (hx * hx + hz * hz)
    izz = mass / 3.0 * (hx * hx + hy * hy)
    return wp.mat33(ixx, 0.0, 0.0, 0.0, iyy, 0.0, 0.0, 0.0, izz)


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


def _spiral_bounds() -> tuple[float, float, float, float]:
    xs = []
    ys = []
    for i in range(NUM_DOMINOES):
        pos, _, _ = _domino_spiral_pose(i)
        xs.append(float(pos[0]))
        ys.append(float(pos[1]))
    return min(xs), max(xs), min(ys), max(ys)


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
        builder.default_shape_cfg.ke = 1.0e4
        builder.default_shape_cfg.kd = 0.0
        builder.default_shape_cfg.kf = 1.0e3
        builder.default_shape_cfg.mu = 1.0
        builder.default_shape_cfg.restitution = 0.15
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
                mass=DOMINO_MASS,
                inertia=_box_inertia(DOMINO_MASS, hx, hy, hz),
                com=wp.vec3(0.0, 0.0, 0.0),
                lock_inertia=True,
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

        if self.solver_name == "xpbd":
            self.solver = newton.solvers.SolverXPBD(self.model, iterations=10, enable_restitution=True)
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
        x_min, x_max, y_min, y_max = _spiral_bounds()
        center_x = 0.5 * (x_min + x_max)
        center_y = 0.5 * (y_min + y_max)
        span = max(x_max - x_min, y_max - y_min)
        center = wp.vec3(center_x, center_y, 0.16)
        _set_camera_look_at(
            self.viewer,
            pos=wp.vec3(center_x - 1.05 * span, center_y - 1.0 * span, 0.9 * span),
            target=center,
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
        x_min, x_max, y_min, y_max = _spiral_bounds()
        span = max(x_max - x_min, y_max - y_min)
        max_abs_pos = max(abs(x_min), abs(x_max), abs(y_min), abs(y_max)) + 2.0 * span
        _validate_body_state(self.model, self.state_0, max_abs_pos=max_abs_pos, min_z=-0.5)

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
