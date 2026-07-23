# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Cable Bend/Twist Analytic Validation
#
# This is a visual and numerical comparison against two exact discrete
# boundary-value solutions:
#
#   Bend rows:
#       Root and tip are kinematic. The tip is placed on a constant-curvature
#       discrete arc and rotated by the target bend angle. With identical bend
#       springs and no external load, the analytic minimum-energy solution is
#       uniform bend:
#
#           theta_i = i / (N - 1) * theta_tip
#
#       The numerical test compares against that analytic centerline. The
#       viewer overlays simulated and analytic centerlines so bend agreement is
#       visible without enabling joint-frame axes by default.
#
#   Twist rows:
#       Root and tip are kinematic at their straight positions. The tip is
#       twisted about the cable axis. With identical twist springs and no bend
#       load, the analytic solution is uniform twist:
#
#           theta_i = i / (N - 1) * theta_tip
#
#       The numerical test compares against that analytic twist distribution.
#       The viewer overlays simulated and analytic material-frame spokes so
#       twist agreement is visible without enabling joint-frame axes by default.
#
# These rows are direct analytic checks for "bend does not create twist" and
# "twist does not create bend" without relying on a continuum EI calibration.
#
# Run interactively:
#   uv run --extra examples python -m newton.examples.vbd.example_cable_bend_twist_analytic
#
# Run as a test:
#   uv run --extra examples python -m newton.examples.vbd.example_cable_bend_twist_analytic --test --viewer null
#
# Verification/report modes:
#   --cable-analytic-mode {all,bend,twist,twist_max}   (default: all)
#
###########################################################################

import math

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.vbd._viewer import com_from_node, node_xyz, set_viewer_camera


@wp.kernel
def _set_kinematic_targets_kernel(
    body_indices: wp.array[wp.int32],
    positions: wp.array[wp.vec3],
    rotations: wp.array[wp.quat],
    body_q0: wp.array[wp.transform],
    body_q1: wp.array[wp.transform],
):
    tid = wp.tid()
    b = body_indices[tid]
    T = wp.transform(positions[tid], rotations[tid])
    body_q0[b] = T
    body_q1[b] = T


class Example:
    """Bend/twist analytic boundary-value comparison against exact discrete solutions."""

    NUM_ELEMENTS = 14
    SEGMENT_LENGTH = 0.10
    CABLE_RADIUS = 0.010
    STRETCH_STIFFNESS = 1.0e6

    TARGET_ANGLES = (math.radians(30.0), math.radians(60.0), math.radians(90.0))
    VISUAL_TWIST_TARGET_ANGLE = math.radians(150.0)

    BEND_STIFFNESS = 400.0
    TWIST_STIFFNESS = 120.0

    BEND_ROW_Y = 0.72
    TWIST_ROW_Y = -0.72
    ROW_SPACING_Z = 0.42

    RAMP_TIME = 2.0
    HOLD_TIME = 4.0

    def __init__(self, viewer, args=None):
        self.viewer = viewer
        self.args = args

        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_iterations = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.cable_length = self.NUM_ELEMENTS * self.SEGMENT_LENGTH
        self.num_joints = self.NUM_ELEMENTS - 1

        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))

        self.bend_cases = []
        self.twist_cases = []
        mode = getattr(args, "cable_analytic_mode", "all")
        if mode not in ("all", "bend", "twist", "twist_max"):
            raise ValueError(f"Unknown cable_analytic_mode: {mode}")

        if mode in ("all", "bend"):
            bend_targets = self.TARGET_ANGLES
            for i, target in enumerate(bend_targets):
                z = (len(bend_targets) - 1 - i) * self.ROW_SPACING_Z
                bodies, joints = self._add_cable(
                    builder,
                    y=self.BEND_ROW_Y,
                    z=z,
                    bend_stiffness=self.BEND_STIFFNESS,
                    twist_stiffness=1000.0,
                    label=f"analytic_bend_{int(math.degrees(target))}",
                )
                self._make_kinematic(builder, bodies[0])
                self._make_kinematic(builder, bodies[-1])
                builder.add_articulation(joints, label=f"analytic_bend_articulation_{i}")
                self.bend_cases.append({"target": target, "bodies": bodies, "tip": bodies[-1]})

        if mode in ("all", "twist", "twist_max"):
            twist_targets = (self.VISUAL_TWIST_TARGET_ANGLE,) if mode == "twist_max" else self.TARGET_ANGLES
            for i, target in enumerate(twist_targets):
                z = (len(twist_targets) - 1 - i) * self.ROW_SPACING_Z
                bodies, joints = self._add_cable(
                    builder,
                    y=self.TWIST_ROW_Y,
                    z=z,
                    bend_stiffness=2000.0,
                    twist_stiffness=self.TWIST_STIFFNESS,
                    label=f"analytic_twist_{int(math.degrees(target))}",
                )
                self._make_kinematic(builder, bodies[0])
                self._make_kinematic(builder, bodies[-1])
                builder.add_articulation(joints, label=f"analytic_twist_articulation_{i}")
                self.twist_cases.append({"target": target, "bodies": bodies, "tip": bodies[-1]})

        builder.color()
        self.model = builder.finalize()

        self.solver = newton.solvers.SolverVBD(self.model, iterations=self.sim_iterations)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        body_q = self.state_0.body_q.numpy()
        for case in self.bend_cases + self.twist_cases:
            case["rest_points"] = np.asarray(
                [node_xyz(body_q[b], self.SEGMENT_LENGTH) for b in case["bodies"]], dtype=np.float64
            )
            case["rest_q"] = [np.asarray(body_q[b][3:7], dtype=np.float64) for b in case["bodies"]]
            case["tip_rest_q"] = np.asarray(body_q[case["tip"]][3:7], dtype=np.float64)

        self._kinematic_indices = wp.array(
            [case["tip"] for case in self.bend_cases + self.twist_cases],
            dtype=wp.int32,
        )
        self._kinematic_pos_np = np.zeros((len(self.bend_cases) + len(self.twist_cases), 3), dtype=np.float32)
        self._kinematic_rot_np = np.zeros((len(self.bend_cases) + len(self.twist_cases), 4), dtype=np.float32)
        self._kinematic_pos = wp.array(self._kinematic_pos_np, dtype=wp.vec3)
        self._kinematic_rot = wp.array(self._kinematic_rot_np, dtype=wp.quat)

        self.viewer.set_model(self.model)
        set_viewer_camera(
            self.viewer,
            pos=wp.vec3(0.5 * self.cable_length + 4.2, 0.0, 1.35),
            target=wp.vec3(0.5 * self.cable_length, 0.0, 0.40),
            fov=34.0,
            show_joints=False,
        )
        self.graph = None
        self.capture()

    def _add_cable(
        self,
        builder,
        y: float,
        z: float,
        bend_stiffness: float,
        twist_stiffness: float,
        label: str,
    ) -> tuple[list[int], list[int]]:
        points = newton.utils.create_straight_cable_points(
            start=wp.vec3(0.0, y, z),
            direction=wp.vec3(1.0, 0.0, 0.0),
            length=self.cable_length,
            num_segments=self.NUM_ELEMENTS,
        )
        quats = newton.utils.create_parallel_transport_cable_quaternions(points)
        bend_damping = bend_stiffness
        twist_damping = twist_stiffness
        bodies, joints = builder.add_rod(
            positions=points,
            quaternions=quats,
            radius=self.CABLE_RADIUS,
            stretch_stiffness=self.STRETCH_STIFFNESS,
            bend_stiffness=bend_stiffness,
            bend_damping=bend_damping,
            twist_stiffness=twist_stiffness,
            twist_damping=twist_damping,
            label=label,
            wrap_in_articulation=False,
            body_frame_origin="com",
        )
        return list(bodies), list(joints)

    def _make_kinematic(self, builder, body_index: int) -> None:
        builder.body_mass[body_index] = 0.0
        builder.body_inv_mass[body_index] = 0.0
        builder.body_inertia[body_index] = wp.mat33(0.0)
        builder.body_inv_inertia[body_index] = wp.mat33(0.0)

    def _load_scale(self, t: float) -> float:
        if t >= self.RAMP_TIME:
            return 1.0
        return max(0.0, t / self.RAMP_TIME)

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
    def _quat_axis_angle(q: np.ndarray) -> tuple[np.ndarray, float]:
        x, y, z, w = float(q[0]), float(q[1]), float(q[2]), float(q[3])
        if w < 0.0:
            x, y, z, w = -x, -y, -z, -w
        w = max(-1.0, min(1.0, w))
        angle = 2.0 * math.acos(w)
        s = math.sqrt(max(0.0, 1.0 - w * w))
        if s < 1.0e-9:
            return np.array([1.0, 0.0, 0.0], dtype=np.float64), 0.0
        return np.array([x / s, y / s, z / s]), angle

    @staticmethod
    def _axis_quat(axis: np.ndarray, angle: float) -> np.ndarray:
        half = 0.5 * angle
        s = math.sin(half)
        return np.array([axis[0] * s, axis[1] * s, axis[2] * s, math.cos(half)], dtype=np.float64)

    @staticmethod
    def _quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
        qv = np.array([v[0], v[1], v[2], 0.0], dtype=np.float64)
        return Example._quat_mul(Example._quat_mul(q, qv), Example._quat_conj(q))[:3]

    def _analytic_angles(self, target: float, scale: float = 1.0) -> np.ndarray:
        tip = target * scale
        return np.asarray([i / self.num_joints * tip for i in range(self.NUM_ELEMENTS)], dtype=np.float64)

    def _analytic_bend_points(self, rest_points: np.ndarray, target: float, scale: float = 1.0) -> np.ndarray:
        angles = self._analytic_angles(target, scale)
        pts = [rest_points[0].copy()]
        for theta in angles[:-1]:
            direction = np.array([math.cos(theta), 0.0, -math.sin(theta)], dtype=np.float64)
            pts.append(pts[-1] + self.SEGMENT_LENGTH * direction)
        return np.asarray(pts, dtype=np.float64)

    def _bend_visual_offsets(self, target: float, scale: float = 1.0) -> np.ndarray:
        angles = self._analytic_angles(target, scale)
        offsets = []
        offset_distance = 0.065
        for theta in angles:
            # Offset in the camera image plane, normal to the local bend tangent.
            # A fixed +Z offset becomes tangent-like near the 90-degree row root,
            # which makes the reference visually merge with the cable.
            normal = np.array([math.sin(theta), 0.0, math.cos(theta)], dtype=np.float64)
            offsets.append(offset_distance * normal)
        return np.asarray(offsets, dtype=np.float64)

    def _update_kinematic_targets(self, scale: float) -> None:
        row = 0
        for case in self.bend_cases:
            target = case["target"] * scale
            points = self._analytic_bend_points(case["rest_points"], case["target"], scale)
            q_bend = self._axis_quat(np.array([0.0, 1.0, 0.0]), target)
            rot = self._quat_mul(q_bend, case["tip_rest_q"])
            self._kinematic_pos_np[row] = com_from_node(points[-1], rot, self.SEGMENT_LENGTH).astype(np.float32)
            self._kinematic_rot_np[row] = rot.astype(np.float32)
            row += 1

        for case in self.twist_cases:
            target = case["target"] * scale
            q_twist = self._axis_quat(np.array([1.0, 0.0, 0.0]), target)
            rot = self._quat_mul(q_twist, case["tip_rest_q"])
            self._kinematic_pos_np[row] = com_from_node(case["rest_points"][-1], rot, self.SEGMENT_LENGTH).astype(
                np.float32
            )
            self._kinematic_rot_np[row] = rot.astype(np.float32)
            row += 1

        self._kinematic_pos.assign(self._kinematic_pos_np)
        self._kinematic_rot.assign(self._kinematic_rot_np)

    def _simulate_substeps(self) -> None:
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            wp.launch(
                _set_kinematic_targets_kernel,
                dim=len(self._kinematic_pos_np),
                inputs=[self._kinematic_indices, self._kinematic_pos, self._kinematic_rot],
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

    def simulate(self, scale: float) -> None:
        self._update_kinematic_targets(scale)
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self._simulate_substeps()

    def step(self):
        self.simulate(self._load_scale(self.sim_time))
        self.sim_time += self.frame_dt

    def _rod_points(self, bodies: list[int]) -> np.ndarray:
        body_q = self.state_0.body_q.numpy()
        return np.asarray([node_xyz(body_q[b], self.SEGMENT_LENGTH) for b in bodies], dtype=np.float64)

    def _measure_case_angles(self, case: dict, axis_index: int) -> np.ndarray:
        body_q = self.state_0.body_q.numpy()
        angles = []
        for b, rest_q in zip(case["bodies"], case["rest_q"], strict=True):
            q_rel = self._quat_mul(np.asarray(body_q[b][3:7], dtype=np.float64), self._quat_conj(rest_q))
            axis, angle = self._quat_axis_angle(q_rel)
            angles.append(float(axis[axis_index] * angle))
        return np.asarray(angles, dtype=np.float64)

    @staticmethod
    def _log_polyline(viewer, name: str, points: np.ndarray, color: tuple[float, float, float], width: float) -> None:
        viewer.log_lines(
            name,
            wp.array(points[:-1].astype(np.float32), dtype=wp.vec3),
            wp.array(points[1:].astype(np.float32), dtype=wp.vec3),
            color,
            width=width,
        )

    def _log_point_markers(
        self,
        name: str,
        points: np.ndarray,
        color: tuple[float, float, float],
        radius: float,
    ) -> None:
        self.viewer.log_points(
            name,
            wp.array(points.astype(np.float32), dtype=wp.vec3),
            wp.array(np.full(len(points), radius, dtype=np.float32), dtype=wp.float32),
            wp.array(np.tile(np.asarray(color, dtype=np.float32), (len(points), 1)), dtype=wp.vec3),
        )

    def _log_twist_phase_comparison(self, name: str, case: dict, twists: np.ndarray) -> None:
        body_q = self.state_0.body_q.numpy()
        centers = []
        sim_tips = []
        ideal_tips = []
        sim_radius = 0.145
        ideal_radius = 0.200
        tangent = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        rest_normal_world = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        for body, rest_q, twist in zip(case["bodies"], case["rest_q"], twists, strict=True):
            q_now = np.asarray(body_q[body][3:7], dtype=np.float64)
            p = node_xyz(body_q[body], self.SEGMENT_LENGTH)

            rest_local_normal = self._quat_rotate(self._quat_conj(rest_q), rest_normal_world)
            sim_normal = self._quat_rotate(q_now, rest_local_normal)

            q_twist = self._axis_quat(tangent, float(twist))
            ideal_normal = self._quat_rotate(q_twist, rest_normal_world)

            centers.append(p)
            sim_tips.append(p + sim_radius * sim_normal)
            ideal_tips.append(p + ideal_radius * ideal_normal)

        centers = np.asarray(centers, dtype=np.float64)
        sim_tips = np.asarray(sim_tips, dtype=np.float64)
        ideal_tips = np.asarray(ideal_tips, dtype=np.float64)

        cyan = (0.0, 0.95, 1.0)
        orange = (1.0, 0.55, 0.05)
        self.viewer.log_lines(
            f"{name}_ideal_spokes",
            wp.array(centers.astype(np.float32), dtype=wp.vec3),
            wp.array(ideal_tips.astype(np.float32), dtype=wp.vec3),
            cyan,
            width=0.018,
        )
        self.viewer.log_lines(
            f"{name}_sim_spokes",
            wp.array(centers.astype(np.float32), dtype=wp.vec3),
            wp.array(sim_tips.astype(np.float32), dtype=wp.vec3),
            orange,
            width=0.018,
        )
        self._log_polyline(self.viewer, f"{name}_ideal_phase", ideal_tips, cyan, 0.018)
        self._log_polyline(self.viewer, f"{name}_sim_phase", sim_tips, orange, 0.018)
        self._log_point_markers(f"{name}_ideal_points", ideal_tips, cyan, 0.011)
        self._log_point_markers(f"{name}_sim_points", sim_tips, orange, 0.010)

    def _log_analytic_references(self) -> None:
        scale = self._load_scale(max(0.0, self.sim_time - self.frame_dt))
        for i, case in enumerate(self.bend_cases):
            points = self._analytic_bend_points(case["rest_points"], case["target"], scale)
            measured_points = self._rod_points(case["bodies"])
            visual_offsets = self._bend_visual_offsets(case["target"], scale)
            points_visual = points + visual_offsets
            measured_points_visual = measured_points + visual_offsets
            self._log_polyline(
                self.viewer,
                f"/analytic_reference/bend_vbd_centerline_{i}",
                measured_points_visual,
                (0.0, 0.65, 1.0),
                0.018,
            )
            self._log_point_markers(
                f"/analytic_reference/bend_vbd_points_{i}",
                measured_points_visual,
                (0.0, 0.65, 1.0),
                0.010,
            )
            self._log_polyline(
                self.viewer,
                f"/analytic_reference/bend_{i}",
                points_visual,
                (0.15, 1.0, 0.35),
                0.018,
            )
            self._log_point_markers(
                f"/analytic_reference/bend_points_{i}",
                points_visual,
                (0.15, 1.0, 0.35),
                0.014,
            )

        for i, case in enumerate(self.twist_cases):
            twists = self._analytic_angles(case["target"], scale)
            self._log_twist_phase_comparison(f"/analytic_reference/twist_phase_{i}", case, twists)

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self._log_analytic_references()
        self.viewer.end_frame()

    def test_final(self):
        bend_tip_errors = []
        bend_angle_rms_errors = []
        bend_shape_errors = []
        twist_tip_errors = []
        twist_linear_errors = []
        twist_transverse = []

        for case in self.bend_cases:
            measured_angles = self._measure_case_angles(case, axis_index=1)
            expected_angles = self._analytic_angles(case["target"])
            measured_points = self._rod_points(case["bodies"])
            expected_points = self._analytic_bend_points(case["rest_points"], case["target"])
            if abs(measured_angles[-1] + expected_angles[-1]) < abs(measured_angles[-1] - expected_angles[-1]):
                measured_angles = -measured_angles

            tip_err = abs(measured_angles[-1] - expected_angles[-1])
            angle_rms = float(np.sqrt(np.mean((measured_angles - expected_angles) ** 2)))
            shape_rms = float(np.sqrt(np.mean(np.sum((measured_points - expected_points) ** 2, axis=1))))
            bend_tip_errors.append(tip_err)
            bend_angle_rms_errors.append(angle_rms)
            bend_shape_errors.append(shape_rms / self.cable_length)

        for case in self.twist_cases:
            measured_angles = self._measure_case_angles(case, axis_index=0)
            expected_angles = self._analytic_angles(case["target"])
            if abs(measured_angles[-1] + expected_angles[-1]) < abs(measured_angles[-1] - expected_angles[-1]):
                measured_angles = -measured_angles
            points = self._rod_points(case["bodies"])
            rest = case["rest_points"]
            trans = float(np.max(np.linalg.norm((points - rest)[:, 1:3], axis=1)))

            tip_err = abs(measured_angles[-1] - expected_angles[-1])
            linear_rms = float(np.sqrt(np.mean((measured_angles - expected_angles) ** 2)))
            twist_tip_errors.append(tip_err)
            twist_linear_errors.append(linear_rms)
            twist_transverse.append(trans / self.cable_length)

        if bend_tip_errors:
            # Current CPU baseline is about 0.0008 deg angle RMS and 3.4e-6 L
            # shape RMS. These gates keep a small solver-noise margin while
            # catching visible regressions in the exact boundary-value solve.
            assert max(bend_tip_errors) < math.radians(0.005), f"bend tip errors too high: {bend_tip_errors}"
            assert max(bend_angle_rms_errors) < math.radians(0.02), (
                f"bend angle RMS errors too high: {bend_angle_rms_errors}"
            )
            assert max(bend_shape_errors) < 2.0e-5, f"bend shape errors too high: {bend_shape_errors}"
        if twist_tip_errors:
            # Current CPU baseline is about 0.004 deg linear RMS and 1.8e-6 L
            # transverse motion. Keep the example as a regression gate, not just
            # a smoke test.
            assert max(twist_tip_errors) < math.radians(0.02), f"twist tip errors too high: {twist_tip_errors}"
            assert max(twist_linear_errors) < math.radians(0.02), f"twist linear errors too high: {twist_linear_errors}"
            assert max(twist_transverse) < 1.0e-5, f"pure twist created transverse motion: {twist_transverse}"


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument(
        "--cable-analytic-mode",
        metavar="MODE",
        choices=("all", "bend", "twist", "twist_max"),
        default="all",
        help="Verification/report rows to build: all, bend-only, twist-only, or one max-twist row.",
    )
    parser.set_defaults(num_frames=int(60 * (Example.RAMP_TIME + Example.HOLD_TIME)) + 30)
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
