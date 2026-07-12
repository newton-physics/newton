# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Cable Twist Transfer Validation
#
# Routed twist-transfer/localization verification for three cables held between
# fixed endpoints:
#
#   1. straight reference
#   2. V-shaped kink
#   3. semicircular arc
#
# The root body is twisted about its local cable tangent while the tip body is
# held at its rest orientation. The scene checks that twist propagates through
# smooth paths, localizes before a sharp kink, and does not create large
# centerline drift.
#
# The acceptance criteria check Newton's VBD bend/twist split directly: twist
# should remain a tangent-axis mode and should transfer across routed cable
# geometry.
#
# Run interactively:
#   uv run --extra examples python -m newton.examples.vbd.example_cable_twist_transfer
#
# Run as a test:
#   uv run --extra examples python -m newton.examples.vbd.example_cable_twist_transfer --test --viewer null
#
###########################################################################

import math

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.vbd._viewer import node_xyz, set_viewer_camera


@wp.kernel
def _spin_roots_kernel(
    body_indices: wp.array[wp.int32],
    twist_rate: wp.array[float],
    dt: float,
    body_q0: wp.array[wp.transform],
    body_q1: wp.array[wp.transform],
):
    tid = wp.tid()
    body_id = body_indices[tid]

    X = body_q0[body_id]
    pos = wp.transform_get_translation(X)
    rot = wp.transform_get_rotation(X)
    axis_world = wp.quat_rotate(rot, wp.vec3(0.0, 0.0, 1.0))
    dq = wp.quat_from_axis_angle(axis_world, twist_rate[0] * dt)
    X_new = wp.transform(pos, wp.mul(dq, rot))
    body_q0[body_id] = X_new
    body_q1[body_id] = X_new


class Example:
    NUM_ELEMENTS = 32
    CABLE_RADIUS = 0.012
    TARGET_ROOT_TWIST = math.radians(90.0)
    RAMP_TIME = 2.0
    HOLD_TIME = 4.0

    STRETCH_STIFFNESS = 1.0e6
    BEND_STIFFNESS = 5.0e3
    TWIST_STIFFNESS = 2.0e2
    BEND_DAMPING = 5.0e3
    TWIST_DAMPING = 2.0e2

    def __init__(self, viewer, args=None):
        self.viewer = viewer
        self.args = args

        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_iterations = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.cases: list[dict] = []

        builder = newton.ModelBuilder(gravity=0.0)

        path_builders = [
            ("straight", self._straight_points(1.3)),
            ("v_kink", self._v_points(0.0)),
            ("semicircle", self._semicircle_points(-1.3)),
        ]

        for label, points in path_builders:
            points_np = self._points_array(points)
            bodies, joints = builder.add_rod(
                positions=points,
                radius=self.CABLE_RADIUS,
                stretch_stiffness=self.STRETCH_STIFFNESS,
                bend_stiffness=self.BEND_STIFFNESS,
                bend_damping=self.BEND_DAMPING,
                twist_stiffness=self.TWIST_STIFFNESS,
                twist_damping=self.TWIST_DAMPING,
                label=f"twist_transfer_{label}",
                wrap_in_articulation=False,
                body_frame_origin="com",
            )
            root_body = int(bodies[0])
            tip_body = int(bodies[-1])
            for body in (root_body, tip_body):
                builder.body_mass[body] = 0.0
                builder.body_inv_mass[body] = 0.0
                builder.body_inertia[body] = wp.mat33(0.0)
                builder.body_inv_inertia[body] = wp.mat33(0.0)

            builder.add_articulation(list(joints), label=f"twist_transfer_{label}_articulation")
            self.cases.append(
                {
                    "label": label,
                    "bodies": list(map(int, bodies)),
                    "segment_length": self._polyline_length(points_np) / self.NUM_ELEMENTS,
                }
            )

        builder.color()
        self.model = builder.finalize()
        self.solver = newton.solvers.SolverVBD(self.model, iterations=self.sim_iterations)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        body_q = self.state_0.body_q.numpy()
        for case in self.cases:
            case["rest_pos"] = np.asarray(
                [node_xyz(body_q[b], case["segment_length"]) for b in case["bodies"]],
                dtype=np.float64,
            )
            case["rest_q"] = [np.asarray(body_q[b][3:7], dtype=np.float64) for b in case["bodies"]]
            case["arc_length"] = self._polyline_length(case["rest_pos"])

        self._root_indices = wp.array([case["bodies"][0] for case in self.cases], dtype=wp.int32)
        self._twist_rate = self.TARGET_ROOT_TWIST / self.RAMP_TIME
        self._twist_rate_np = np.zeros(1, dtype=np.float32)
        self._twist_rate_wp = wp.array(self._twist_rate_np, dtype=float)

        self.viewer.set_model(self.model)
        set_viewer_camera(
            self.viewer,
            pos=wp.vec3(4.4, 0.0, 1.45),
            target=wp.vec3(0.0, 0.0, 0.35),
            fov=34.0,
            show_joints=False,
        )
        self.graph = None
        self.capture()

    @classmethod
    def _straight_points(cls, y_offset: float) -> list[wp.vec3]:
        length = 2.4
        return [
            wp.vec3(length * i / cls.NUM_ELEMENTS - 0.5 * length, y_offset, 0.35) for i in range(cls.NUM_ELEMENTS + 1)
        ]

    @classmethod
    def _v_points(cls, y_offset: float) -> list[wp.vec3]:
        half = cls.NUM_ELEMENTS // 2
        left = np.array([-1.2, y_offset - 0.5, 0.35], dtype=np.float64)
        apex = np.array([0.0, y_offset + 0.45, 0.35], dtype=np.float64)
        right = np.array([1.2, y_offset - 0.5, 0.35], dtype=np.float64)
        points = []
        for i in range(half + 1):
            p = (1.0 - i / half) * left + (i / half) * apex
            points.append(wp.vec3(*p))
        for i in range(1, cls.NUM_ELEMENTS - half + 1):
            denom = cls.NUM_ELEMENTS - half
            p = (1.0 - i / denom) * apex + (i / denom) * right
            points.append(wp.vec3(*p))
        return points

    @classmethod
    def _semicircle_points(cls, y_offset: float) -> list[wp.vec3]:
        radius = 0.82
        center = np.array([0.0, y_offset - 0.25, 0.35], dtype=np.float64)
        points = []
        for i in range(cls.NUM_ELEMENTS + 1):
            theta = math.pi * (1.0 - i / cls.NUM_ELEMENTS)
            x = radius * math.cos(theta)
            y = radius * math.sin(theta)
            p = center + np.array([x, y, 0.0], dtype=np.float64)
            points.append(wp.vec3(*p))
        return points

    @staticmethod
    def _polyline_length(points: np.ndarray) -> float:
        return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))

    @staticmethod
    def _points_array(points: list[wp.vec3]) -> np.ndarray:
        return np.asarray([[float(p[0]), float(p[1]), float(p[2])] for p in points], dtype=np.float64)

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
    def _quat_conj(q: np.ndarray) -> np.ndarray:
        return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float64)

    @staticmethod
    def _quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
        qv = np.array([v[0], v[1], v[2], 0.0], dtype=np.float64)
        return Example._quat_mul(Example._quat_mul(q, qv), Example._quat_conj(q))[:3]

    @staticmethod
    def _quat_axis_angle(q: np.ndarray) -> tuple[np.ndarray, float]:
        x, y, z, w = float(q[0]), float(q[1]), float(q[2]), float(q[3])
        if w < 0.0:
            x, y, z, w = -x, -y, -z, -w
        w = max(-1.0, min(1.0, w))
        angle = 2.0 * math.acos(w)
        s = math.sqrt(max(0.0, 1.0 - w * w))
        if s < 1.0e-9:
            return np.array([1.0, 0.0, 0.0], dtype=np.float64), 0.0
        return np.array([x / s, y / s, z / s], dtype=np.float64), angle

    def _twist_rate_at_time(self, t: float) -> float:
        return self._twist_rate if t < self.RAMP_TIME else 0.0

    def _update_twist_rate(self, twist_rate: float) -> None:
        self._twist_rate_np[0] = twist_rate
        self._twist_rate_wp.assign(self._twist_rate_np)

    def _simulate_substeps(self) -> None:
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            wp.launch(
                _spin_roots_kernel,
                dim=len(self.cases),
                inputs=[self._root_indices, self._twist_rate_wp, self.sim_dt],
                outputs=[self.state_0.body_q, self.state_1.body_q],
            )
            self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def capture(self) -> None:
        if self.solver.device.is_cuda:
            with wp.ScopedCapture() as capture:
                self._simulate_substeps()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self, twist_rate: float) -> None:
        self._update_twist_rate(twist_rate)
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self._simulate_substeps()

    def step(self):
        self.simulate(self._twist_rate_at_time(self.sim_time))
        self.sim_time += self.frame_dt

    def _measure_twist_profile(self, case: dict) -> np.ndarray:
        body_q = self.state_0.body_q.numpy()
        twists = []
        for body, rest_q in zip(case["bodies"], case["rest_q"], strict=True):
            q_now = np.asarray(body_q[body][3:7], dtype=np.float64)
            q_delta = self._quat_mul(q_now, self._quat_conj(rest_q))
            axis, angle = self._quat_axis_angle(q_delta)
            tangent = self._quat_rotate(rest_q, np.array([0.0, 0.0, 1.0], dtype=np.float64))
            signed = float(np.dot(axis, tangent) * angle)
            twists.append(signed)
        twists_np = np.asarray(twists, dtype=np.float64)
        if twists_np[0] < 0.0:
            twists_np = -twists_np
        return twists_np

    def _current_points(self, case: dict) -> np.ndarray:
        body_q = self.state_0.body_q.numpy()
        return np.asarray([node_xyz(body_q[b], case["segment_length"]) for b in case["bodies"]], dtype=np.float64)

    def _log_twist_ticks(self, case: dict, twists: np.ndarray, color: tuple[float, float, float]) -> None:
        body_q = self.state_0.body_q.numpy()
        starts = []
        ends = []
        tick_len = 0.12
        for body, rest_q, twist in zip(case["bodies"], case["rest_q"], twists, strict=True):
            p = node_xyz(body_q[body], case["segment_length"])
            tangent = self._quat_rotate(rest_q, np.array([0.0, 0.0, 1.0], dtype=np.float64))
            normal = self._quat_rotate(rest_q, np.array([1.0, 0.0, 0.0], dtype=np.float64))
            q_twist = np.array(
                [
                    tangent[0] * math.sin(0.5 * twist),
                    tangent[1] * math.sin(0.5 * twist),
                    tangent[2] * math.sin(0.5 * twist),
                    math.cos(0.5 * twist),
                ],
                dtype=np.float64,
            )
            normal_twisted = self._quat_rotate(q_twist, normal)
            starts.append(p - 0.5 * tick_len * normal_twisted)
            ends.append(p + 0.5 * tick_len * normal_twisted)
        self.viewer.log_lines(
            f"/twist_transfer/ticks/{case['label']}",
            wp.array(np.asarray(starts, dtype=np.float32), dtype=wp.vec3),
            wp.array(np.asarray(ends, dtype=np.float32), dtype=wp.vec3),
            color,
            width=0.01,
        )

    @staticmethod
    def _log_polyline(viewer, name: str, points: np.ndarray, color: tuple[float, float, float], width: float) -> None:
        viewer.log_lines(
            name,
            wp.array(points[:-1].astype(np.float32), dtype=wp.vec3),
            wp.array(points[1:].astype(np.float32), dtype=wp.vec3),
            color,
            width=width,
        )

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        colors = [(0.1, 0.85, 1.0), (1.0, 0.55, 0.1), (0.25, 1.0, 0.35)]
        for case, color in zip(self.cases, colors, strict=True):
            self._log_polyline(
                self.viewer,
                f"/twist_transfer/rest/{case['label']}",
                case["rest_pos"] + np.array([0.0, 0.0, -0.055]),
                (0.35, 0.35, 0.35),
                0.006,
            )
            self._log_twist_ticks(case, self._measure_twist_profile(case), color)
        self.viewer.end_frame()

    def test_final(self):
        body_q = self.state_0.body_q.numpy()
        body_qd = self.state_0.body_qd.numpy()
        assert np.isfinite(body_q).all(), "non-finite body transforms"
        assert np.isfinite(body_qd).all(), "non-finite body velocities"

        root_errors = []
        centerline_drifts = []
        twist_metrics = {}
        for case in self.cases:
            twists = self._measure_twist_profile(case)
            points = self._current_points(case)
            drift = float(np.max(np.linalg.norm(points - case["rest_pos"], axis=1)) / case["arc_length"])
            mid = len(twists) // 2
            max_second_half = float(np.max(np.abs(twists[mid:])))
            root_errors.append(abs(twists[0] - self.TARGET_ROOT_TWIST))
            centerline_drifts.append(drift)
            twist_metrics[case["label"]] = {
                "mid": abs(float(twists[mid])),
                "pre_kink": abs(float(twists[mid - 2])),
                "max_second_half": max_second_half,
            }

        assert max(root_errors) < math.radians(4.0), f"root twist target errors too large: {root_errors}"
        assert twist_metrics["straight"]["mid"] > math.radians(25.0), (
            f"straight twist did not distribute: {twist_metrics}"
        )
        assert twist_metrics["v_kink"]["pre_kink"] > math.radians(8.0), (
            f"V-kink twist did not reach the kink: {twist_metrics}"
        )
        assert twist_metrics["v_kink"]["pre_kink"] > 2.0 * twist_metrics["v_kink"]["max_second_half"], (
            f"V-kink twist should remain concentrated before the kink: {twist_metrics}"
        )
        assert twist_metrics["v_kink"]["max_second_half"] < math.radians(8.0), (
            f"V-kink should localize twist before the sharp kink in this model: {twist_metrics}"
        )
        assert twist_metrics["semicircle"]["max_second_half"] > math.radians(8.0), (
            f"semicircle twist did not transfer around the curve: {twist_metrics}"
        )
        assert max(centerline_drifts) < 0.08, f"centerline drift too large: {centerline_drifts}"


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=int(60 * (Example.RAMP_TIME + Example.HOLD_TIME)) + 30)
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
