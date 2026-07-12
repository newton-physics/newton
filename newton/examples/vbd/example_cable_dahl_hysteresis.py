# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Cable Dahl Hysteresis Validation
#
# Four cantilever cables are driven through slow one-sided cycles:
#
#   1. bend elastic: cyclic transverse tip force
#   2. bend Dahl:    same force, Dahl history enabled
#   3. twist elastic: cyclic kinematic tip twist
#   4. twist Dahl:    same kinematic twist, Dahl history enabled
#
# The bend pair validates visible force/deflection hysteresis and permanent
# set. The twist pair validates torque/twist hysteresis in the split twist
# subspace without the dynamic artifacts of a free moment-driven tip.
#
# Run interactively:
#   uv run --extra examples python -m newton.examples.vbd.example_cable_dahl_hysteresis
#
# Run as a test:
#   uv run --extra examples python -m newton.examples.vbd.example_cable_dahl_hysteresis --test --viewer null
#
# Verification/report modes:
#   --cable-dahl-mode {all,bend,twist}   (default: all)
#
###########################################################################

import math
from typing import ClassVar

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.vbd._viewer import node_xyz, set_viewer_camera


@wp.kernel
def _set_kinematic_targets_kernel(
    body_indices: wp.array[wp.int32],
    positions: wp.array[wp.vec3],
    rotations: wp.array[wp.quat],
    body_q0: wp.array[wp.transform],
    body_q1: wp.array[wp.transform],
):
    tid = wp.tid()
    body = body_indices[tid]
    target = wp.transform(positions[tid], rotations[tid])
    body_q0[body] = target
    body_q1[body] = target


class Example:
    """Bend and twist Dahl hysteresis containment test."""

    NUM_ELEMENTS = 16
    SEGMENT_LENGTH = 0.10
    CABLE_RADIUS = 0.01
    STRETCH_STIFFNESS = 1.0e6
    BEND_STIFFNESS = 250.0
    TWIST_STIFFNESS = 250.0
    BEND_DAMPING = 75.0
    TWIST_DAMPING = 75.0

    TIP_FORCE_MAX = 0.50  # N
    TWIST_TARGET_MAX = math.radians(45.0)
    CASE_Y: ClassVar[dict[str, float]] = {
        "bend_elastic": -0.72,
        "bend_dahl": -0.42,
        "twist_elastic": 0.18,
        "twist_dahl": 0.48,
    }

    DAHL_EPS_MAX = 0.10
    DAHL_TAU = 0.05

    PHASE_DURATION = 3.0
    NUM_PHASES = 4
    SETTLE_TIME = 2.0

    def __init__(self, viewer, args=None):
        self.viewer = viewer
        self.args = args

        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_iterations = 10
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.video_mode = getattr(args, "cable_dahl_mode", "all") if args is not None else "all"
        if self.video_mode not in {"all", "bend", "twist"}:
            self.video_mode = "all"

        self.cable_length = self.NUM_ELEMENTS * self.SEGMENT_LENGTH
        self.cycle_duration = self.NUM_PHASES * self.PHASE_DURATION

        builder = newton.ModelBuilder(gravity=0.0)
        newton.solvers.SolverVBD.register_custom_attributes(builder, dahl_defaults_enabled=False)

        self.cases: list[dict] = []
        for name, mode, has_dahl in (
            ("bend_elastic", "bend", False),
            ("bend_dahl", "bend", True),
            ("twist_elastic", "twist", False),
            ("twist_dahl", "twist", True),
        ):
            bodies, joint_range = self._add_cantilever(builder, name, mode)
            self.cases.append(
                {
                    "name": name,
                    "label": name.replace("_", " "),
                    "mode": mode,
                    "has_dahl": has_dahl,
                    "bodies": bodies,
                    "tip_body": int(bodies[-1]),
                    "joint_range": joint_range,
                    "color": self._case_color(name),
                }
            )

        builder.color()
        self.model = builder.finalize()
        self._configure_dahl_attributes()
        self.solver = newton.solvers.SolverVBD(self.model, iterations=self.sim_iterations)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        body_q = self.state_0.body_q.numpy()
        for case in self.cases:
            case["rest_pos"] = np.asarray(
                [node_xyz(body_q[b], self.SEGMENT_LENGTH) for b in case["bodies"]], dtype=np.float64
            )
            case["rest_q"] = [np.asarray(body_q[b][3:7], dtype=np.float64) for b in case["bodies"]]
            # COM-origin transform drives the kinematic twist tip directly (twist about the
            # tangent leaves the COM fixed), so this stays in body-frame (COM) coordinates.
            case["tip_rest_pos"] = np.asarray(body_q[case["tip_body"]][:3], dtype=np.float64)
            case["tip_rest_q"] = np.asarray(body_q[case["tip_body"]][3:7], dtype=np.float64)

        self.twist_cases = [case for case in self.cases if case["mode"] == "twist"]
        self._kinematic_indices = wp.array(
            np.asarray([case["tip_body"] for case in self.twist_cases], dtype=np.int32),
            dtype=wp.int32,
        )
        self._kinematic_pos_np = np.asarray(
            [case["tip_rest_pos"] for case in self.twist_cases],
            dtype=np.float32,
        )
        self._kinematic_rot_np = np.asarray(
            [case["tip_rest_q"] for case in self.twist_cases],
            dtype=np.float32,
        )
        self._kinematic_pos = wp.array(self._kinematic_pos_np, dtype=wp.vec3)
        self._kinematic_rot = wp.array(self._kinematic_rot_np, dtype=wp.quat)

        self.viewer.set_model(self.model)
        set_viewer_camera(
            self.viewer,
            pos=wp.vec3(0.5 * self.cable_length, -3.5, 1.05),
            target=wp.vec3(0.5 * self.cable_length, -0.12, 0.0),
            fov=34.0,
            show_joints=True,
            joint_scale=1.0,
        )

        self._wrench_np = np.zeros((self.model.body_count, 6), dtype=np.float32)
        self.tip_wrench = wp.array(self._wrench_np, dtype=wp.spatial_vector)

        self.history_force: list[float] = []
        self.history_twist_command: list[float] = []
        self.history_bend_down: dict[str, list[float]] = {
            case["name"]: [] for case in self.cases if case["mode"] == "bend"
        }
        self.history_twist_angle: dict[str, list[float]] = {case["name"]: [] for case in self.twist_cases}
        self.history_twist_reaction: dict[str, list[float]] = {case["name"]: [] for case in self.twist_cases}
        self.max_tip_x_disp: dict[str, float] = {case["name"]: 0.0 for case in self.cases}
        self.max_tip_y_disp: dict[str, float] = {case["name"]: 0.0 for case in self.cases}
        self.max_twist_centerline_drift: dict[str, float] = {case["name"]: 0.0 for case in self.cases}
        self.max_active_sigma: dict[str, float] = {case["name"]: 0.0 for case in self.cases}
        self.max_leak_sigma: dict[str, float] = {case["name"]: 0.0 for case in self.cases}
        self.max_active_kappa: dict[str, float] = {case["name"]: 0.0 for case in self.cases}
        self.max_leak_kappa: dict[str, float] = {case["name"]: 0.0 for case in self.cases}

        self.graph = None
        self.capture()

    def _add_cantilever(self, builder, name: str, mode: str) -> tuple[list[int], tuple[int, int]]:
        start = wp.vec3(0.0, self.CASE_Y[name], 0.0)
        points = newton.utils.create_straight_cable_points(
            start=start,
            direction=wp.vec3(1.0, 0.0, 0.0),
            length=self.cable_length,
            num_segments=self.NUM_ELEMENTS,
        )
        quats = newton.utils.create_parallel_transport_cable_quaternions(points)

        joint_count_before = builder.joint_count
        rod_bodies, _ = builder.add_rod(
            positions=points,
            quaternions=quats,
            radius=self.CABLE_RADIUS,
            stretch_stiffness=self.STRETCH_STIFFNESS,
            stretch_damping=0.0,
            bend_stiffness=self.BEND_STIFFNESS,
            bend_damping=self.BEND_DAMPING,
            twist_stiffness=self.TWIST_STIFFNESS,
            twist_damping=self.TWIST_DAMPING,
            label=f"dahl_{name}",
            body_frame_origin="com",
        )
        joint_count_after = builder.joint_count

        # Root is fixed for every cantilever. Twist cases also prescribe the
        # tip pose, so the twist validation is a clean quasi-static reaction
        # check rather than a free spinning torque-driven dynamics case.
        kinematic_bodies = [int(rod_bodies[0])]
        if mode == "twist":
            kinematic_bodies.append(int(rod_bodies[-1]))
        for body in kinematic_bodies:
            builder.body_mass[body] = 0.0
            builder.body_inv_mass[body] = 0.0
            builder.body_inertia[body] = wp.mat33(0.0)
            builder.body_inv_inertia[body] = wp.mat33(0.0)

        return list(map(int, rod_bodies)), (joint_count_before, joint_count_after)

    @staticmethod
    def _case_color(name: str) -> tuple[float, float, float]:
        colors = {
            "bend_elastic": (1.0, 0.55, 0.12),
            "bend_dahl": (0.20, 0.48, 1.0),
            "twist_elastic": (1.0, 0.70, 0.20),
            "twist_dahl": (0.25, 0.62, 1.0),
        }
        return colors[name]

    def _configure_dahl_attributes(self) -> None:
        eps_max = np.zeros(self.model.joint_count, dtype=np.float32)
        tau = np.full(self.model.joint_count, 1.0, dtype=np.float32)
        for case in self.cases:
            if not case["has_dahl"]:
                continue
            s, e = case["joint_range"]
            eps_max[s:e] = self.DAHL_EPS_MAX
            tau[s:e] = self.DAHL_TAU
        self.model.vbd.dahl_eps_max.assign(eps_max)
        self.model.vbd.dahl_tau.assign(tau)

    def _drive_at_time(self, t: float) -> float:
        if t >= self.cycle_duration:
            return 0.0
        phase = t / self.PHASE_DURATION
        i = int(phase)
        frac = phase - i
        return frac if i % 2 == 0 else 1.0 - frac

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

    @classmethod
    def _quat_rotate(cls, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        qv = np.array([v[0], v[1], v[2], 0.0], dtype=np.float64)
        return cls._quat_mul(cls._quat_mul(q, qv), cls._quat_conj(q))[:3]

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

    @staticmethod
    def _axis_quat(axis: np.ndarray, angle: float) -> np.ndarray:
        axis = np.asarray(axis, dtype=np.float64)
        axis /= max(np.linalg.norm(axis), 1.0e-12)
        s = math.sin(0.5 * angle)
        return np.array([axis[0] * s, axis[1] * s, axis[2] * s, math.cos(0.5 * angle)], dtype=np.float64)

    def _update_twist_targets(self, twist_target: float) -> None:
        for i, case in enumerate(self.twist_cases):
            q_twist = self._axis_quat(np.array([1.0, 0.0, 0.0], dtype=np.float64), twist_target)
            self._kinematic_pos_np[i] = case["tip_rest_pos"].astype(np.float32)
            self._kinematic_rot_np[i] = self._quat_mul(q_twist, case["tip_rest_q"]).astype(np.float32)
        self._kinematic_pos.assign(self._kinematic_pos_np)
        self._kinematic_rot.assign(self._kinematic_rot_np)

    def _update_drive_targets(self, force_now: float, twist_target: float) -> None:
        self._update_twist_targets(twist_target)
        self._wrench_np.fill(0.0)
        for case in self.cases:
            if case["mode"] == "bend":
                self._wrench_np[case["tip_body"], 2] = -force_now
        self.tip_wrench.assign(self._wrench_np)

    def _simulate_substeps(self) -> None:
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.state_0.body_f.assign(self.tip_wrench)
            wp.launch(
                _set_kinematic_targets_kernel,
                dim=len(self.twist_cases),
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

    def simulate(self, force_now: float, twist_target: float) -> None:
        self._update_drive_targets(force_now, twist_target)
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self._simulate_substeps()

    def step(self) -> None:
        drive = self._drive_at_time(self.sim_time)
        force_now = self.TIP_FORCE_MAX * drive
        twist_target = self.TWIST_TARGET_MAX * drive
        self.simulate(force_now, twist_target)
        self.sim_time += self.frame_dt
        self._record_frame(force_now, twist_target)

    def _record_frame(self, force_now: float, twist_target: float) -> None:
        body_q = self.state_0.body_q.numpy()
        self.history_force.append(force_now)
        self.history_twist_command.append(twist_target)

        self._record_subspace_state()

        for case in self.cases:
            tip_pos = node_xyz(body_q[case["tip_body"]], self.SEGMENT_LENGTH)
            tip_delta = tip_pos - case["rest_pos"][-1]
            self.max_tip_x_disp[case["name"]] = max(self.max_tip_x_disp[case["name"]], abs(float(tip_delta[0])))
            self.max_tip_y_disp[case["name"]] = max(self.max_tip_y_disp[case["name"]], abs(float(tip_delta[1])))
            if case["mode"] == "bend":
                self.history_bend_down[case["name"]].append(float(-tip_delta[2]))
            else:
                self.history_twist_angle[case["name"]].append(float(self._tip_twist(case)))
                self.history_twist_reaction[case["name"]].append(float(self._twist_reaction(case)))
                current = self._current_points(case)
                drift = current - case["rest_pos"]
                transverse = np.sqrt(drift[:, 1] * drift[:, 1] + drift[:, 2] * drift[:, 2])
                self.max_twist_centerline_drift[case["name"]] = max(
                    self.max_twist_centerline_drift[case["name"]],
                    float(np.max(transverse)),
                )

    def _record_subspace_state(self) -> None:
        sigma = np.asarray(self.solver.joint_sigma_prev.numpy(), dtype=np.float64)
        kappa = np.asarray(self.solver.joint_kappa_prev.numpy(), dtype=np.float64)
        for case in self.cases:
            s, e = case["joint_range"]
            if e <= s:
                continue
            sigma_case = sigma[s:e]
            kappa_case = kappa[s:e]
            if case["mode"] == "bend":
                active_sigma = float(np.max(np.linalg.norm(sigma_case[:, :2], axis=1)))
                leak_sigma = float(np.max(np.abs(sigma_case[:, 2])))
                active_kappa = float(np.max(np.linalg.norm(kappa_case[:, :2], axis=1)))
                leak_kappa = float(np.max(np.abs(kappa_case[:, 2])))
            else:
                active_sigma = float(np.max(np.abs(sigma_case[:, 2])))
                leak_sigma = float(np.max(np.linalg.norm(sigma_case[:, :2], axis=1)))
                active_kappa = float(np.max(np.abs(kappa_case[:, 2])))
                leak_kappa = float(np.max(np.linalg.norm(kappa_case[:, :2], axis=1)))
            self.max_active_sigma[case["name"]] = max(self.max_active_sigma[case["name"]], active_sigma)
            self.max_leak_sigma[case["name"]] = max(self.max_leak_sigma[case["name"]], leak_sigma)
            self.max_active_kappa[case["name"]] = max(self.max_active_kappa[case["name"]], active_kappa)
            self.max_leak_kappa[case["name"]] = max(self.max_leak_kappa[case["name"]], leak_kappa)

    def _tip_twist(self, case: dict) -> float:
        body_q = self.state_0.body_q.numpy()
        q_now = np.asarray(body_q[case["tip_body"]][3:7], dtype=np.float64)
        q_delta = self._quat_mul(q_now, self._quat_conj(case["tip_rest_q"]))
        axis, angle = self._quat_axis_angle(q_delta)
        tangent = self._quat_rotate(case["tip_rest_q"], np.array([0.0, 0.0, 1.0], dtype=np.float64))
        return float(np.dot(axis, tangent) * angle)

    def _twist_profile(self, case: dict) -> np.ndarray:
        body_q = self.state_0.body_q.numpy()
        twists = []
        for body, rest_q in zip(case["bodies"], case["rest_q"], strict=True):
            q_now = np.asarray(body_q[body][3:7], dtype=np.float64)
            q_delta = self._quat_mul(q_now, self._quat_conj(rest_q))
            axis, angle = self._quat_axis_angle(q_delta)
            tangent = self._quat_rotate(rest_q, np.array([0.0, 0.0, 1.0], dtype=np.float64))
            twists.append(float(np.dot(axis, tangent) * angle))
        return np.asarray(twists, dtype=np.float64)

    def _twist_reaction(self, case: dict) -> float:
        sigma = np.asarray(self.solver.joint_sigma_prev.numpy(), dtype=np.float64)
        kappa = np.asarray(self.solver.joint_kappa_prev.numpy(), dtype=np.float64)
        s, e = case["joint_range"]
        if e <= s:
            return 0.0
        joint_twist = kappa[s:e, 2]
        joint_sigma = sigma[s:e, 2]
        return float(self.TWIST_STIFFNESS * np.mean(joint_twist) + np.mean(joint_sigma))

    def _current_points(self, case: dict) -> np.ndarray:
        body_q = self.state_0.body_q.numpy()
        return np.asarray([node_xyz(body_q[b], self.SEGMENT_LENGTH) for b in case["bodies"]], dtype=np.float64)

    @staticmethod
    def _loop_area(xs: list[float] | np.ndarray, ys: list[float] | np.ndarray) -> float:
        xs = np.asarray(xs, dtype=np.float64)
        ys = np.asarray(ys, dtype=np.float64)
        if len(xs) < 3:
            return 0.0
        return float(0.5 * abs(np.dot(xs, np.roll(ys, -1)) - np.dot(np.roll(xs, -1), ys)))

    def _cycle_count(self) -> int:
        return min(len(self.history_force), int(self.cycle_duration / self.frame_dt))

    def analysis_metrics(self) -> dict:
        n_cycle = self._cycle_count()
        force = np.asarray(self.history_force, dtype=np.float64)
        twist_command = np.asarray(self.history_twist_command, dtype=np.float64)
        bend_rows = []
        twist_rows = []

        for case in self.cases:
            name = case["name"]
            if case["mode"] == "bend":
                response = np.asarray(self.history_bend_down[name], dtype=np.float64)
                cycle_response = response[:n_cycle]
                max_response = float(np.max(np.abs(cycle_response)))
                residual = float(abs(response[-1]))
                area = self._loop_area(force[:n_cycle], cycle_response)
                fatness = area / max(max_response * max_response, 1.0e-12)
                bend_rows.append(
                    {
                        "name": name,
                        "label": case["label"],
                        "has_dahl": bool(case["has_dahl"]),
                        "max_deflection": max_response,
                        "residual": residual,
                        "loop_area": area,
                        "loop_fatness": fatness,
                        "max_tip_x": self.max_tip_x_disp[name],
                        "max_tip_y": self.max_tip_y_disp[name],
                        "active_sigma": self.max_active_sigma[name],
                        "leak_sigma": self.max_leak_sigma[name],
                        "active_kappa": self.max_active_kappa[name],
                        "leak_kappa": self.max_leak_kappa[name],
                    }
                )
            else:
                angles = np.asarray(self.history_twist_angle[name], dtype=np.float64)
                reaction = np.asarray(self.history_twist_reaction[name], dtype=np.float64)
                max_angle = float(np.max(np.abs(angles[:n_cycle])))
                residual_angle = float(abs(angles[-1]))
                max_reaction = float(np.max(np.abs(reaction[:n_cycle])))
                residual_reaction = float(abs(reaction[-1]))
                area = self._loop_area(twist_command[:n_cycle], reaction[:n_cycle])
                norm = max(np.max(np.abs(twist_command[:n_cycle])) * max_reaction, 1.0e-12)
                twist_rows.append(
                    {
                        "name": name,
                        "label": case["label"],
                        "has_dahl": bool(case["has_dahl"]),
                        "max_twist": max_angle,
                        "max_twist_deg": math.degrees(max_angle),
                        "residual_twist": residual_angle,
                        "residual_twist_deg": math.degrees(residual_angle),
                        "max_reaction": max_reaction,
                        "residual_reaction": residual_reaction,
                        "loop_area": area,
                        "loop_area_norm": area / norm,
                        "centerline_drift": self.max_twist_centerline_drift[name],
                        "active_sigma": self.max_active_sigma[name],
                        "leak_sigma": self.max_leak_sigma[name],
                        "active_kappa": self.max_active_kappa[name],
                        "leak_kappa": self.max_leak_kappa[name],
                    }
                )

        return {
            "force": force,
            "twist_command": twist_command,
            "bend_downward": [np.asarray(self.history_bend_down[row["name"]], dtype=np.float64) for row in bend_rows],
            "twist_reaction": [
                np.asarray(self.history_twist_reaction[row["name"]], dtype=np.float64) for row in twist_rows
            ],
            "twist_angles": [np.asarray(self.history_twist_angle[row["name"]], dtype=np.float64) for row in twist_rows],
            "bend_rows": bend_rows,
            "twist_rows": twist_rows,
            "cable_length": self.cable_length,
            "tip_force_max": self.TIP_FORCE_MAX,
            "twist_target_max": self.TWIST_TARGET_MAX,
            "dahl_eps_max": self.DAHL_EPS_MAX,
            "dahl_tau": self.DAHL_TAU,
        }

    def render(self) -> None:
        self.viewer.begin_frame(self.sim_time)
        if self.video_mode == "all":
            self.viewer.log_state(self.state_0)
        self._log_centerlines()
        self._log_hysteresis_plots()
        self._log_tip_trails()
        self._log_twist_ticks()
        self.viewer.end_frame()

    def _log_polyline(
        self,
        name: str,
        points: list[np.ndarray] | np.ndarray,
        color: tuple[float, float, float],
        width: float,
    ) -> None:
        if len(points) < 2:
            return
        pts = np.asarray(points, dtype=np.float32)
        self.viewer.log_lines(
            name,
            wp.array(pts[:-1], dtype=wp.vec3),
            wp.array(pts[1:], dtype=wp.vec3),
            color,
            width=width,
        )

    def _case_visible(self, case: dict) -> bool:
        return self.video_mode == "all" or case["mode"] == self.video_mode

    def _log_centerlines(self) -> None:
        for case in self.cases:
            if not self._case_visible(case):
                continue
            self._log_polyline(
                f"/dahl/centerline/{case['name']}",
                self._current_points(case),
                case["color"],
                0.018,
            )

    def _log_hysteresis_plots(self) -> None:
        if len(self.history_force) < 2:
            return

        if self.video_mode == "bend":
            bend_origin = np.array([1.62, -0.55, 0.08], dtype=np.float64)
            twist_origin = None
        elif self.video_mode == "twist":
            bend_origin = None
            twist_origin = np.array([1.62, 0.34, 0.22], dtype=np.float64)
        else:
            bend_origin = np.array([1.82, -0.57, 0.10], dtype=np.float64)
            twist_origin = np.array([1.82, 0.32, 0.22], dtype=np.float64)
        plot_width = 0.68
        plot_height = 0.36

        force_scale = plot_width / max(self.TIP_FORCE_MAX, 1.0e-9)
        defl_scale = 1.55
        command_scale = plot_width / max(self.TWIST_TARGET_MAX, 1.0e-9)
        reaction_scale = 0.014

        if bend_origin is not None:
            self._log_axes(bend_origin, x_len=plot_width, z_pos=plot_height, name="/dahl_plot/bend_axes")
            for row_offset, case in zip((-0.035, 0.035), [c for c in self.cases if c["mode"] == "bend"], strict=True):
                pts = [
                    bend_origin + np.array([force * force_scale, row_offset, z * defl_scale], dtype=np.float64)
                    for force, z in zip(self.history_force, self.history_bend_down[case["name"]], strict=True)
                ]
                self._log_polyline(f"/dahl_plot/{case['name']}_bend_loop", pts, case["color"], 0.012)

        if twist_origin is not None:
            self._log_axes(
                twist_origin,
                x_len=plot_width,
                z_pos=plot_height,
                z_neg=0.16,
                name="/dahl_plot/twist_axes",
            )
            for row_offset, case in zip((-0.035, 0.035), self.twist_cases, strict=True):
                pts = [
                    twist_origin + np.array([cmd * command_scale, row_offset, tau * reaction_scale], dtype=np.float64)
                    for cmd, tau in zip(
                        self.history_twist_command, self.history_twist_reaction[case["name"]], strict=True
                    )
                ]
                self._log_polyline(f"/dahl_plot/{case['name']}_twist_loop", pts, case["color"], 0.012)

    def _log_axes(self, origin: np.ndarray, x_len: float, z_pos: float, name: str, z_neg: float = 0.0) -> None:
        self._log_polyline(name + "_x", [origin, origin + np.array([x_len, 0.0, 0.0])], (0.78, 0.78, 0.78), 0.008)
        self._log_polyline(
            name + "_z",
            [
                origin - np.array([0.0, 0.0, z_neg], dtype=np.float64),
                origin + np.array([0.0, 0.0, z_pos], dtype=np.float64),
            ],
            (0.78, 0.78, 0.78),
            0.008,
        )

    def _log_tip_trails(self) -> None:
        if len(self.history_force) < 2:
            return
        for case in self.cases:
            if case["mode"] != "bend":
                continue
            if not self._case_visible(case):
                continue
            rest = case["tip_rest_pos"]
            pts = [
                np.array([rest[0], rest[1] + 0.04, rest[2] - down], dtype=np.float64)
                for down in self.history_bend_down[case["name"]]
            ]
            self._log_polyline(f"/dahl/trails/{case['name']}", pts, case["color"], 0.010)

    def _log_twist_ticks(self) -> None:
        for case in self.twist_cases:
            if not self._case_visible(case):
                continue
            positions = self._current_points(case)
            twists = self._twist_profile(case)
            starts = []
            ends = []
            for p, rest_q, twist in zip(positions, case["rest_q"], twists, strict=True):
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
                starts.append(p - 0.070 * normal_twisted)
                ends.append(p + 0.070 * normal_twisted)
            self.viewer.log_lines(
                f"/dahl/twist_ticks/{case['name']}",
                wp.array(np.asarray(starts, dtype=np.float32), dtype=wp.vec3),
                wp.array(np.asarray(ends, dtype=np.float32), dtype=wp.vec3),
                case["color"],
                width=0.010,
            )

    def test_final(self) -> None:
        metrics = self.analysis_metrics()
        bend_rows = {row["name"]: row for row in metrics["bend_rows"]}
        twist_rows = {row["name"]: row for row in metrics["twist_rows"]}

        be = bend_rows["bend_elastic"]
        bd = bend_rows["bend_dahl"]
        te = twist_rows["twist_elastic"]
        td = twist_rows["twist_dahl"]

        self._assert_bend_metrics(be, bd)
        self._assert_twist_metrics(te, td)
        self._assert_subspace_containment(metrics)

    def _assert_bend_metrics(self, elastic: dict, dahl: dict) -> None:
        for row in (elastic, dahl):
            assert math.isfinite(row["max_deflection"]) and math.isfinite(row["residual"])
            assert row["max_deflection"] > 0.03, f"{row['label']} barely deflected: {row}"
            rel = row["max_deflection"] / self.cable_length
            assert rel < 0.30, f"{row['label']} left the small-deflection bend regime: {row}"

        assert dahl["max_deflection"] < elastic["max_deflection"], (
            f"Dahl bend should reduce peak deflection: elastic={elastic}, dahl={dahl}"
        )
        assert dahl["loop_fatness"] > elastic["loop_fatness"], (
            f"Dahl bend loop should be wider per response range: elastic={elastic}, dahl={dahl}"
        )
        elastic_residual_rel = elastic["residual"] / elastic["max_deflection"]
        dahl_residual_rel = dahl["residual"] / dahl["max_deflection"]
        assert elastic_residual_rel < 0.05, f"elastic bend did not return near zero: {elastic}"
        assert dahl_residual_rel > 3.0 * elastic_residual_rel, (
            f"Dahl bend residual should exceed elastic residual fraction: elastic={elastic}, dahl={dahl}"
        )
        assert dahl_residual_rel > 0.05, f"Dahl bend residual too small: {dahl}"

        for row in (elastic, dahl):
            assert row["max_tip_x"] / self.cable_length < 0.30, f"excessive bend foreshortening: {row}"
            assert row["max_tip_y"] / self.cable_length < 0.005, f"pure bend drifted out of plane: {row}"

    def _assert_twist_metrics(self, elastic: dict, dahl: dict) -> None:
        for row in (elastic, dahl):
            assert math.isfinite(row["max_twist"]) and math.isfinite(row["residual_reaction"])
            assert row["max_twist"] > math.radians(40.0), f"{row['label']} barely twisted: {row}"
            assert row["residual_twist"] < math.radians(0.05), f"kinematic tip did not return to zero: {row}"

        assert dahl["loop_area"] > 2.0 * max(elastic["loop_area"], 1.0e-6), (
            f"Dahl twist should create a larger torque/twist loop: elastic={elastic}, dahl={dahl}"
        )
        assert elastic["residual_reaction"] < 0.25, f"elastic twist retained unexpected reaction: {elastic}"
        assert dahl["residual_reaction"] > 1.0, f"Dahl twist residual reaction too small: {dahl}"
        assert dahl["residual_reaction"] > 5.0 * max(elastic["residual_reaction"], 1.0e-6), (
            f"Dahl twist residual should exceed elastic residual reaction: elastic={elastic}, dahl={dahl}"
        )

        for row in (elastic, dahl):
            assert row["centerline_drift"] / self.cable_length < 0.002, (
                f"pure twist moved the centerline too much: {row}"
            )

    def _assert_subspace_containment(self, metrics: dict) -> None:
        rows = list(metrics["bend_rows"]) + list(metrics["twist_rows"])
        for row in rows:
            # Dahl sigma is accumulated through the full VBD step in float32.
            # Keep this as a containment check, but allow a small solver-noise
            # margin around the active history component.
            sigma_gate = max(1.0e-5, 2.0e-3 * max(row["active_sigma"], 1.0))
            kappa_gate = max(1.0e-6, 1.0e-4 * max(row["active_kappa"], 1.0))
            assert row["leak_sigma"] < sigma_gate, f"Dahl sigma leaked across bend/twist subspaces: {row}"
            assert row["leak_kappa"] < kappa_gate, f"Dahl kappa leaked across bend/twist subspaces: {row}"


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument(
        "--cable-dahl-mode",
        metavar="MODE",
        choices=("all", "bend", "twist"),
        default="all",
        help="Verification/report rows to show: all, bend-only, or twist-only.",
    )
    parser.set_defaults(num_frames=int(60 * (Example.NUM_PHASES * Example.PHASE_DURATION + Example.SETTLE_TIME)) + 30)
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
