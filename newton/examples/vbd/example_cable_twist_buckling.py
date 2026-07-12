# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Cable Twist Buckling Link Verification
#
# Fixed-span dynamic VBD cable example:
#
#   1. build a straight cable with both tips fixed from the start
#   2. let it settle briefly
#   3. twist only the tip capsule about its local cable tangent
#   4. hold the final twist as the centerline buckles into a helical crown
#
# There is no shortening ramp and no initial U-shaped/slack geometry. With a
# sufficiently large gradual twist, the straight cable can still develop a
# crown-like centerline wave. The test checks that the open-curve link quantity,
# Lk = Tw + Wr, remains close to the commanded endpoint twist while self-contact
# prevents strand passage.
#
# Run interactively:
#   uv run --extra examples python -m newton.examples.vbd.example_cable_twist_buckling
#
# Run as a test:
#   uv run --extra examples python -m newton.examples.vbd.example_cable_twist_buckling --test --viewer null
#
###########################################################################

import math

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.vbd._viewer import node_xyz, set_viewer_camera


@wp.kernel
def _drive_tip_twist_kernel(
    root_body: int,
    tip_body: int,
    root_pose: wp.transform,
    tip_pos: wp.vec3,
    tip_rest_rot: wp.quat,
    twist_angles: wp.array[wp.float32],
    twist_index: int,
    body_q0: wp.array[wp.transform],
    body_q1: wp.array[wp.transform],
):
    twist_angle = twist_angles[twist_index]
    axis_world = wp.quat_rotate(tip_rest_rot, wp.vec3(0.0, 0.0, 1.0))
    twist_rot = wp.quat_from_axis_angle(axis_world, twist_angle)
    tip_pose = wp.transform(tip_pos, wp.mul(twist_rot, tip_rest_rot))

    body_q0[root_body] = root_pose
    body_q1[root_body] = root_pose
    body_q0[tip_body] = tip_pose
    body_q1[tip_body] = tip_pose


# The public example is fixed. Report scripts can still pass these attributes
# on their argparse.Namespace to reuse the same setup for sensitivity studies.
_INTERNAL_PARAMETER_OVERRIDES = (
    "bend_stiffness",
    "twist_stiffness",
    "twist_turns",
    "iterations",
    "substeps",
    "seed_offset",
    "contact_mode",
    "contact_stiffness",
    "contact_damping",
    "contact_gap",
)


class Example:
    NUM_ELEMENTS = 64
    SEGMENT_LENGTH = 0.075
    CABLE_RADIUS = 0.014

    TWIST_TURNS = 9.0

    SETTLE_TIME = 1.5
    TWIST_TIME = 5.0
    HOLD_TIME = 4.0
    TOTAL_TIME = SETTLE_TIME + TWIST_TIME + HOLD_TIME

    STRETCH_STIFFNESS = 1.0e5
    CONTACT_MODE = "hard-history"
    CONTACT_STIFFNESS = 1.0e5
    CONTACT_DAMPING = 0.0
    CONTACT_GAP = 0.05
    CONTACT_TOPOLOGICAL_FILTER_SPAN = 2

    BEND_STIFFNESS = 40.0
    TWIST_STIFFNESS = 50.0
    BEND_DAMPING = 0.001
    TWIST_DAMPING = 0.001
    GRAVITY = -9.81

    FPS = 60
    SIM_SUBSTEPS = 6
    SIM_ITERATIONS = 10

    LINK_TOLERANCE_TURNS = 0.08
    MIN_SELF_DISTANCE_DIAMETERS = 2.0

    # Symmetry-breaking seed. A taut clamped rod under twist has the Greenhill
    # critical twist ``theta_c = 4*pi*sqrt(EI/GJ)``; below it the straight rod
    # is stable, above it the helical mode is unstable. The unstable mode
    # cannot grow from an exactly planar state, so we add one small lateral
    # offset at the mid body during init. Gravity sag also breaks symmetry
    # but only along -Z; this seed kicks +Y to give the helix a deterministic
    # phase.
    SEED_BODY_OFFSET_Y = 5.0e-4  # 0.5 mm

    def __init__(self, viewer, args=None):
        self.viewer = viewer
        self.args = args

        params = self._resolve_params(args)
        self.bend_stiffness = params["bend_stiffness"]
        self.twist_stiffness = params["twist_stiffness"]
        self.bend_damping = self.BEND_DAMPING * self.bend_stiffness
        self.twist_damping = self.TWIST_DAMPING * self.twist_stiffness
        self.twist_turns = params["twist_turns"]
        self.seed_offset = params["seed_offset"]
        self.contact_mode = str(params["contact_mode"])
        self.contact_stiffness = float(params["contact_stiffness"])
        self.contact_damping = float(params["contact_damping"])
        self.contact_gap = float(params["contact_gap"])
        if self.contact_mode not in ("none", "soft", "hard", "hard-history"):
            raise ValueError(f"unknown contact_mode {self.contact_mode!r}")
        self.contact_enabled = self.contact_mode != "none"

        self.fps = self.FPS
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = int(params["substeps"])
        self.sim_iterations = int(params["iterations"])
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.frame_index = 0

        self.cable_length = self.NUM_ELEMENTS * self.SEGMENT_LENGTH
        self.target_twist = 2.0 * math.pi * self.twist_turns
        self.total_time = self.TOTAL_TIME

        points_np = self._straight_points(
            self.NUM_ELEMENTS,
            self.SEGMENT_LENGTH,
        )
        points = [wp.vec3(*p) for p in points_np]

        builder = newton.ModelBuilder(gravity=self.GRAVITY)
        shape_cfg = None
        if self.contact_enabled:
            shape_cfg = newton.ModelBuilder.ShapeConfig(
                ke=self.contact_stiffness,
                kd=self.contact_damping,
                gap=self.contact_gap,
            )
        bodies, joints = builder.add_rod(
            positions=points,
            quaternions=None,
            radius=self.CABLE_RADIUS,
            cfg=shape_cfg,
            stretch_stiffness=self.STRETCH_STIFFNESS,
            stretch_damping=0.0,
            bend_stiffness=self.bend_stiffness,
            bend_damping=self.bend_damping,
            twist_stiffness=self.twist_stiffness,
            twist_damping=self.twist_damping,
            label="twist_buckling",
            wrap_in_articulation=False,
            body_frame_origin="com",
        )

        self.bodies = list(map(int, bodies))
        self.joints = list(map(int, joints))
        if self.contact_enabled:
            self._filter_near_rod_collision_pairs(builder, self.bodies, self.CONTACT_TOPOLOGICAL_FILTER_SPAN)

        self.root_body = self.bodies[0]
        self.tip_body = self.bodies[-1]
        for body in (self.root_body, self.tip_body):
            builder.body_mass[body] = 0.0
            builder.body_inv_mass[body] = 0.0
            builder.body_inertia[body] = wp.mat33(0.0)
            builder.body_inv_inertia[body] = wp.mat33(0.0)
        builder.add_articulation(self.joints, label="twist_buckling_articulation")

        builder.color()
        self.model = builder.finalize()
        hard_contact = self.contact_mode in ("hard", "hard-history")
        contact_history = self.contact_mode == "hard-history"
        self.solver = newton.solvers.SolverVBD(
            self.model,
            iterations=self.sim_iterations,
            rigid_contact_hard=hard_contact,
            rigid_contact_history=contact_history,
            rigid_contact_stick_motion_eps=0.0,
            rigid_contact_stick_freeze_translation_eps=0.0,
            rigid_contact_stick_freeze_angular_eps=0.0,
            rigid_body_contact_buffer_size=256,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        if contact_history:
            pipeline = newton.CollisionPipeline(self.model, contact_matching="latest")
            self.contacts = self.model.contacts(collision_pipeline=pipeline)
        else:
            self.contacts = self.model.contacts()

        body_q = self.state_0.body_q.numpy()
        self.rest_pos = np.asarray([node_xyz(body_q[b], self.SEGMENT_LENGTH) for b in self.bodies], dtype=np.float64)
        self.rest_q = [np.asarray(body_q[b][3:7], dtype=np.float64) for b in self.bodies]
        self.root_rest_pos = self.rest_pos[0].copy()
        self.tip_rest_pos = np.asarray(body_q[self.tip_body][:3], dtype=np.float64)
        self.root_pose = wp.transform(
            wp.vec3(*body_q[self.root_body][:3]),
            wp.quat(*body_q[self.root_body][3:7]),
        )
        self.tip_rest_rot = wp.quat(*body_q[self.tip_body][3:7])

        # Symmetry-breaking seed at the mid body. Strictly necessary for the
        # twist-only Greenhill mode: the rod is launched exactly planar, so
        # without a seed the unstable mode has nothing to grow from.
        mid = self.NUM_ELEMENTS // 2
        body_q_np = self.state_0.body_q.numpy()
        body_q_np[self.bodies[mid], 1] += self.seed_offset
        self.state_0.body_q.assign(body_q_np)
        self.state_1.body_q.assign(body_q_np)
        self.twist_angles_np = np.zeros(self.sim_substeps, dtype=np.float32)
        self.twist_angles = wp.array(self.twist_angles_np, dtype=wp.float32, device=self.model.device)

        self.viewer.set_model(self.model)
        set_viewer_camera(
            self.viewer,
            pos=wp.vec3(0.50 * self.cable_length, -6.20, 1.05),
            target=wp.vec3(0.50 * self.cable_length, 0.0, 0.60),
            fov=32.0,
            show_joints=True,
            joint_scale=3.0,
        )
        self.graph = None
        self.capture()

    @classmethod
    def _resolve_params(cls, args) -> dict:
        defaults = {
            "bend_stiffness": cls.BEND_STIFFNESS,
            "twist_stiffness": cls.TWIST_STIFFNESS,
            "twist_turns": cls.TWIST_TURNS,
            "iterations": cls.SIM_ITERATIONS,
            "substeps": cls.SIM_SUBSTEPS,
            "seed_offset": cls.SEED_BODY_OFFSET_Y,
            "contact_mode": cls.CONTACT_MODE,
            "contact_stiffness": cls.CONTACT_STIFFNESS,
            "contact_damping": cls.CONTACT_DAMPING,
            "contact_gap": cls.CONTACT_GAP,
        }
        resolved = {}
        for name in _INTERNAL_PARAMETER_OVERRIDES:
            value = getattr(args, name, None) if args is not None else None
            resolved[name] = defaults[name] if value is None else value
        return resolved

    @staticmethod
    def _smoothstep(x: float) -> float:
        x = min(1.0, max(0.0, float(x)))
        return x * x * (3.0 - 2.0 * x)

    @staticmethod
    def _straight_points(num_elements: int, segment_length: float) -> np.ndarray:
        length = num_elements * segment_length
        x = np.linspace(0.0, length, num_elements + 1)
        return np.column_stack([x, np.zeros_like(x), np.full_like(x, 0.75)])

    @staticmethod
    def _filter_near_rod_collision_pairs(builder, bodies: list[int], span: int) -> None:
        """Filter near-topological capsule pairs from self-collision.

        With h=0.075 m and r=0.014 m, second-neighbor capsule surfaces are
        only about 0.047 m apart in the straight rest state. The contact gap
        would otherwise create rest contacts between i and i+2.
        """
        for i, body_i in enumerate(bodies):
            for j in range(i + 1, min(len(bodies), i + span + 1)):
                body_j = bodies[j]
                for shape_i in builder.body_shapes.get(body_i, []):
                    for shape_j in builder.body_shapes.get(body_j, []):
                        builder.add_shape_collision_filter_pair(int(shape_i), int(shape_j))

    def _command(self, t: float) -> tuple[float, str]:
        t = float(t)
        if t <= self.SETTLE_TIME:
            return 0.0, "settle"
        t -= self.SETTLE_TIME
        a = self._smoothstep(t / self.TWIST_TIME)
        if t <= self.TWIST_TIME:
            return self.target_twist * a, "twist"
        return self.target_twist, "twist hold"

    def _update_twist_angles(self) -> None:
        for i in range(self.sim_substeps):
            sub_t = self.sim_time + i * self.sim_dt
            self.twist_angles_np[i] = self._command(sub_t)[0]
        self.twist_angles.assign(self.twist_angles_np)

    def _apply_command(self, twist_angle: float | None = None, substep_index: int = 0) -> None:
        if twist_angle is not None:
            self.twist_angles_np[0] = twist_angle
            self.twist_angles.assign(self.twist_angles_np)
        wp.launch(
            _drive_tip_twist_kernel,
            dim=1,
            inputs=[
                self.root_body,
                self.tip_body,
                self.root_pose,
                wp.vec3(*self.tip_rest_pos),
                self.tip_rest_rot,
                self.twist_angles,
                int(substep_index),
            ],
            outputs=[self.state_0.body_q, self.state_1.body_q],
            device=self.model.device,
        )

    def simulate(self) -> None:
        for i in range(self.sim_substeps):
            self._apply_command(substep_index=i)
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            if self.contact_enabled:
                self.model.collide(self.state_0, self.contacts)
            self.solver.set_rigid_history_update(True)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def capture(self) -> None:
        if self.solver.device.is_cuda and self.sim_substeps % 2 == 0:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def step(self):
        self._update_twist_angles()
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt
        self.frame_index += 1

    def current_points(self) -> np.ndarray:
        body_q = self.state_0.body_q.numpy()
        return np.asarray([node_xyz(body_q[b], self.SEGMENT_LENGTH) for b in self.bodies], dtype=np.float64)

    @staticmethod
    def _quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
        x, y, z, w = q
        u = np.asarray([x, y, z], dtype=np.float64)
        return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)

    @classmethod
    def _local_axes(cls, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        q = np.asarray(q, dtype=np.float64)
        q = q / max(np.linalg.norm(q), 1.0e-12)
        tangent = cls._quat_rotate(q, np.asarray([0.0, 0.0, 1.0], dtype=np.float64))
        material = cls._quat_rotate(q, np.asarray([1.0, 0.0, 0.0], dtype=np.float64))
        return tangent, material

    @classmethod
    def _total_twist_turns(cls, body_q: np.ndarray, bodies: list[int]) -> float:
        total = 0.0
        for i in range(len(bodies) - 1):
            t0, m0 = cls._local_axes(body_q[bodies[i]][3:7])
            t1, m1 = cls._local_axes(body_q[bodies[i + 1]][3:7])
            denom = 1.0 + float(np.dot(t0, t1))
            if denom > 1.0e-9:
                transported = m0 - (float(np.dot(m0, t1)) / denom) * (t0 + t1)
            else:
                transported = m0
            transported /= max(np.linalg.norm(transported), 1.0e-12)
            sin_theta = float(np.dot(t1, np.cross(transported, m1)))
            cos_theta = float(np.dot(transported, m1))
            total += math.atan2(sin_theta, cos_theta)
        return total / (2.0 * math.pi)

    @staticmethod
    def _segment_distance(p0: np.ndarray, p1: np.ndarray, q0: np.ndarray, q1: np.ndarray) -> float:
        u = p1 - p0
        v = q1 - q0
        w = p0 - q0
        a = float(np.dot(u, u))
        b = float(np.dot(u, v))
        c = float(np.dot(v, v))
        d = float(np.dot(u, w))
        e = float(np.dot(v, w))
        denom = a * c - b * b
        eps = 1.0e-12

        if a <= eps and c <= eps:
            return float(np.linalg.norm(p0 - q0))
        if a <= eps:
            s = 0.0
            t = min(1.0, max(0.0, e / c))
        elif c <= eps:
            t = 0.0
            s = min(1.0, max(0.0, -d / a))
        else:
            s = min(1.0, max(0.0, (b * e - c * d) / denom)) if denom > eps else 0.0
            t = (b * s + e) / c
            if t < 0.0:
                t = 0.0
                s = min(1.0, max(0.0, -d / a))
            elif t > 1.0:
                t = 1.0
                s = min(1.0, max(0.0, (b - d) / a))

        closest_p = p0 + s * u
        closest_q = q0 + t * v
        return float(np.linalg.norm(closest_p - closest_q))

    @classmethod
    def _min_nonlocal_segment_distance(cls, points: np.ndarray, skip_neighbors: int) -> float:
        best = float("inf")
        for i in range(len(points) - 1):
            for j in range(i + skip_neighbors + 1, len(points) - 1):
                best = min(best, cls._segment_distance(points[i], points[i + 1], points[j], points[j + 1]))
        return best

    @staticmethod
    def _writhe_klenin_langowski(points: np.ndarray) -> float:
        points = np.asarray(points, dtype=np.float64)

        def normalized(v: np.ndarray) -> np.ndarray:
            return v / max(np.linalg.norm(v), 1.0e-12)

        def safe_asin(x: float) -> float:
            return math.asin(max(-1.0, min(1.0, x)))

        total = 0.0
        for i in range(len(points) - 1):
            r1, r2 = points[i], points[i + 1]
            for j in range(i + 2, len(points) - 1):
                r3, r4 = points[j], points[j + 1]
                r13 = r3 - r1
                r14 = r4 - r1
                r23 = r3 - r2
                r24 = r4 - r2
                n1 = normalized(np.cross(r13, r14))
                n2 = normalized(np.cross(r14, r24))
                n3 = normalized(np.cross(r24, r23))
                n4 = normalized(np.cross(r23, r13))
                omega = (
                    safe_asin(float(np.dot(n1, n2)))
                    + safe_asin(float(np.dot(n2, n3)))
                    + safe_asin(float(np.dot(n3, n4)))
                    + safe_asin(float(np.dot(n4, n1)))
                )
                sign = np.sign(float(np.dot(np.cross(r4 - r3, r2 - r1), r13)))
                total += sign * omega
        return total / (2.0 * math.pi)

    def link_metrics(self) -> dict[str, float]:
        body_q = self.state_0.body_q.numpy()
        points = self.current_points()
        twist = self._total_twist_turns(body_q, self.bodies)
        writhe = self._writhe_klenin_langowski(points)
        commanded = self._command(self.sim_time)[0] / (2.0 * math.pi)
        link = twist + writhe
        min_distance = self._min_nonlocal_segment_distance(points, self.CONTACT_TOPOLOGICAL_FILTER_SPAN)
        return {
            "twist_turns": twist,
            "writhe_turns": writhe,
            "link_turns": link,
            "commanded_twist_turns": commanded,
            "link_error_turns": commanded - link,
            "min_nonlocal_distance": min_distance,
            "min_nonlocal_distance_diameters": min_distance / (2.0 * self.CABLE_RADIUS),
        }

    def metrics(self) -> dict[str, float | str | bool]:
        points = self.current_points()
        twist_angle, stage = self._command(self.sim_time)
        segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
        lateral = points - self.rest_pos
        lateral[:, 0] = 0.0
        span = float(np.linalg.norm(points[-1] - points[0]))
        roll_range, roll_jump, principal_roll_max = self._roll_profile_metrics(self.state_0.body_q.numpy())
        return {
            "stage": stage,
            "commanded_twist_turns": float(twist_angle / (2.0 * math.pi)),
            "tip_command_principal_turns": float(self._principal_turns(twist_angle / (2.0 * math.pi))),
            "roll_profile_range_turns": roll_range,
            "roll_profile_max_jump_turns": roll_jump,
            "principal_roll_max_turns": principal_roll_max,
            "span_fraction": span / self.cable_length,
            "max_segment_stretch": float(np.max(np.abs(segment_lengths - self.SEGMENT_LENGTH)) / self.SEGMENT_LENGTH),
            "max_lateral_motion_pct_l": 100.0
            * float(np.max(np.linalg.norm(lateral[:, 1:3], axis=1)))
            / self.cable_length,
            "contact_mode": self.contact_mode,
            "contact_stiffness": self.contact_stiffness,
            "contact_damping": self.contact_damping,
            "contact_gap": self.contact_gap,
            "finite": bool(np.isfinite(points).all()),
        }

    @staticmethod
    def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        ax, ay, az, aw = a
        bx, by, bz, bw = b
        return np.array(
            [
                aw * bx + ax * bw + ay * bz - az * by,
                aw * by - ax * bz + ay * bw + az * bx,
                aw * bz + ax * by - ay * bx + az * bw,
                aw * bw - ax * bx - ay * by - az * bz,
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _quat_inv(q: np.ndarray) -> np.ndarray:
        return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float64) / float(np.dot(q, q))

    @classmethod
    def _local_z_roll_turns(cls, q: np.ndarray, q_rest: np.ndarray) -> float:
        q = np.asarray(q, dtype=np.float64)
        q = q / max(np.linalg.norm(q), 1.0e-12)
        q_rest = np.asarray(q_rest, dtype=np.float64)
        q_rest = q_rest / max(np.linalg.norm(q_rest), 1.0e-12)
        rel = cls._quat_mul(cls._quat_inv(q_rest), q)
        angle = 2.0 * math.atan2(float(rel[2]), float(rel[3]))
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle / (2.0 * math.pi)

    @staticmethod
    def _principal_turns(turns: np.ndarray | float) -> np.ndarray | float:
        return ((turns + 0.5) % 1.0) - 0.5

    def _roll_profile(self, body_q: np.ndarray) -> np.ndarray:
        return np.asarray(
            [self._local_z_roll_turns(body_q[body][3:7], self.rest_q[i]) for i, body in enumerate(self.bodies)],
            dtype=np.float64,
        )

    def _roll_profile_metrics(self, body_q: np.ndarray) -> tuple[float, float, float]:
        rolls = self._roll_profile(body_q)
        unwrapped = np.unwrap(2.0 * math.pi * rolls) / (2.0 * math.pi)
        roll_range = float(np.max(unwrapped) - np.min(unwrapped))
        roll_jump = float(np.max(np.abs(np.diff(unwrapped)))) if len(unwrapped) > 1 else 0.0
        principal_roll_max = float(np.max(np.abs(rolls))) if len(rolls) else 0.0
        return roll_range, roll_jump, principal_roll_max

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        row = self.metrics()
        link = self.link_metrics()
        assert row["finite"], f"non-finite cable centerline: {row}"
        assert row["commanded_twist_turns"] > 0.95 * self.twist_turns, f"twist command did not complete: {row}"
        assert row["roll_profile_range_turns"] > 0.95 * self.twist_turns, f"roll did not transfer cleanly: {row}"
        assert row["roll_profile_max_jump_turns"] < 0.35, f"roll profile jumps too much: {row}"
        assert row["max_segment_stretch"] < 0.06, f"segments stretched too much: {row}"
        assert row["max_lateral_motion_pct_l"] > 1.5, f"straight twist-only cable did not visibly crown: {row}"
        assert abs(link["link_error_turns"]) < self.LINK_TOLERANCE_TURNS, f"link number drifted too much: {link}"
        assert link["min_nonlocal_distance_diameters"] > self.MIN_SELF_DISTANCE_DIAMETERS, (
            f"cable self-distance is too small for a link-conservation test: {link}"
        )


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=int(Example.FPS * Example.TOTAL_TIME))
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
