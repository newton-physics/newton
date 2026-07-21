# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Benchmark a scripted G1 bimanual pushing control loop.

The measured loop covers deterministic two-hand six-DoF command generation, IK,
joint-target writes, and two completed MuJoCo physics substeps. Rendering,
physical input devices, transport, perception, and display latency are outside
the benchmark scope.
"""

from __future__ import annotations

import copy
import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import warp as wp
from asv_runner.benchmarks.mark import SkipNotImplemented

wp.config.enable_backward = False
wp.config.log_level = wp.LOG_WARNING

import newton
import newton.ik as ik
import newton.utils
from newton import JointTargetMode


@wp.kernel
def _write_robot_targets(
    joint_q_ik: wp.array2d[wp.float32],
    nominal_joint_q: wp.array[wp.float32],
    arm_joint_mask: wp.array[wp.int32],
    previous_joint_target_q: wp.array[wp.float32],
    dt: wp.float32,
    joint_target_q: wp.array[wp.float32],
    joint_target_qd: wp.array[wp.float32],
):
    i = wp.tid()
    q = nominal_joint_q[i]
    if arm_joint_mask[i] != 0:
        q = joint_q_ik[0, i]

    qd = 1.5 * (q - previous_joint_target_q[i]) / dt
    joint_target_q[i] = q
    joint_target_qd[i] = wp.clamp(qd, -20.0, 20.0)
    previous_joint_target_q[i] = q


@dataclass(frozen=True)
class _TeleopMode:
    device: str
    mujoco_backend: str
    use_graph: bool = False
    requires_cuda: bool = False
    requires_cuda_graph: bool = False


_TELEOP_MODES = {
    "mjwarp_cuda_graph": _TeleopMode(
        device="cuda:0",
        mujoco_backend="warp",
        use_graph=True,
        requires_cuda=True,
        requires_cuda_graph=True,
    ),
    "mjwarp_cpu_graph": _TeleopMode(device="cpu", mujoco_backend="warp", use_graph=True),
    "mjwarp_cpu_eager": _TeleopMode(device="cpu", mujoco_backend="warp"),
    "mujoco_cpu_eager": _TeleopMode(device="cpu", mujoco_backend="cpu"),
}


class _WindowStats:
    def __init__(self, maxlen: int):
        self.values: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=maxlen))

    def add(self, name: str, value: float) -> None:
        if math.isfinite(value):
            self.values[name].append(float(value))

    def summary(self, name: str) -> tuple[float, float, float]:
        values = self.values.get(name)
        if not values:
            raise RuntimeError(f"No teleop samples collected for {name!r}")
        ordered = sorted(values)
        p95_index = min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)
        return sum(values) / len(values), ordered[p95_index], ordered[-1]

    def coefficient_of_variation_pct(self, name: str) -> float:
        values = self.values.get(name)
        if not values:
            raise RuntimeError(f"No teleop samples collected for {name!r}")
        mean = sum(values) / len(values)
        if mean == 0.0:
            return 0.0
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        return 100.0 * math.sqrt(variance) / mean

    def clear(self) -> None:
        self.values.clear()


def _quat_to_vec4(q: wp.quat) -> wp.vec4:
    return wp.vec4(float(q[0]), float(q[1]), float(q[2]), float(q[3]))


def _quat_to_np(q: wp.quat) -> np.ndarray:
    return np.array([float(q[0]), float(q[1]), float(q[2]), float(q[3])], dtype=np.float32)


def _vec3_to_np(v: wp.vec3) -> np.ndarray:
    return np.array([float(v[0]), float(v[1]), float(v[2])], dtype=np.float32)


class _TeleopLoop:
    # Keep the control loop below the physics rate so every command advances
    # multiple completed physics steps. Performance targets remain external.
    control_hz = 100
    physics_hz = 200
    sim_substeps = 2
    linear_speed = 0.5
    sweep_half_period = 1.6
    arm_joint_indices = (*range(15, 22), *range(29, 36))

    def __init__(self, mode: _TeleopMode, stats_window: int):
        self.device = wp.get_device(mode.device)
        self.frame_dt = 1.0 / self.control_hz
        self.sim_dt = 1.0 / self.physics_hz
        if not math.isclose(self.frame_dt, self.sim_substeps * self.sim_dt):
            raise ValueError("The control period must contain an integer number of physics steps")
        self.sim_time = 0.0
        self.frame_index = 0
        self.stats = _WindowStats(stats_window)

        robot = self._build_robot()
        self.model_ik = copy.deepcopy(robot).finalize()

        scene = newton.ModelBuilder()
        scene.add_builder(robot)
        self.object_body_index, self.object_shape_index = self._add_scene(scene)
        self.model = scene.finalize()

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        state_ik = self.model_ik.state()
        newton.eval_fk(self.model_ik, self.model_ik.joint_q, self.model_ik.joint_qd, state_ik)
        self.ee_indices = [
            self._find_body(self.model_ik, "left_wrist_yaw_link"),
            self._find_body(self.model_ik, "right_wrist_yaw_link"),
        ]
        body_q = state_ik.body_q.numpy()
        self.target_tfs = [wp.transform(*body_q[index]) for index in self.ee_indices]
        self.target_rotations = [wp.transform_get_rotation(target) for target in self.target_tfs]
        self.initial_object_position = self.state_0.body_q.numpy()[self.object_body_index, :3].copy()
        self.hand_shape_indices = {
            shape_index
            for shape_index, shape_label in enumerate(self.model.shape_label)
            if "/left_hand_" in shape_label or "/right_hand_" in shape_label
        }

        self.nominal_joint_q = wp.clone(self.model.joint_q)
        self.arm_joint_mask = wp.array(
            [int(i in self.arm_joint_indices) for i in range(self.model.joint_coord_count)],
            dtype=wp.int32,
            device=self.device,
        )
        self.previous_joint_target_q = wp.empty(self.model.joint_coord_count, dtype=wp.float32, device=self.device)
        self._setup_ik()
        wp.copy(self.control.joint_target_q, self.model.joint_q)
        wp.copy(self.previous_joint_target_q, self.control.joint_target_q)
        self._write_targets()

        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            use_mujoco_cpu=mode.mujoco_backend == "cpu",
            solver="newton",
            integrator="implicitfast",
            cone="pyramidal",
            iterations=20,
            ls_iterations=8,
            njmax=200,
            nconmax=100,
        )
        self.mjc_geom_to_newton_shape = self.solver.mjc_geom_to_newton_shape.numpy()[0]
        self.contacts = None
        if mode.mujoco_backend == "warp":
            self.contacts = newton.Contacts(self.solver.get_max_contact_count(), 0)

        self.graph_ik = None
        if mode.use_graph:
            with wp.ScopedCapture(device=self.device) as capture:
                self.ik_solver.step(self.joint_q_ik, self.joint_q_ik, iterations=16)
            self.graph_ik = capture.graph

        self.graph_sim = None
        if mode.use_graph and mode.mujoco_backend == "warp":
            with wp.ScopedCapture(device=self.device) as capture:
                self._simulate()
            self.graph_sim = capture.graph

    def _build_robot(self) -> newton.ModelBuilder:
        robot = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(robot)
        robot.default_joint_cfg = newton.ModelBuilder.JointDofConfig(limit_ke=1.0e3, limit_kd=1.0e1, friction=1.0e-5)
        robot.default_shape_cfg.ke = 1.0e3
        robot.default_shape_cfg.kd = 2.0e2
        robot.default_shape_cfg.kf = 1.0e3
        robot.default_shape_cfg.mu = 0.75
        robot.add_usd(
            str(newton.utils.download_asset("unitree_g1") / "usd_structured" / "g1_29dof_with_hand_rev_1_0.usda"),
            floating=False,
            collapse_fixed_joints=True,
            enable_self_collisions=False,
            hide_collision_shapes=True,
            skip_mesh_approximation=True,
        )
        robot.approximate_meshes("bounding_box")

        for i in range(robot.joint_dof_count):
            robot.joint_target_ke[i] = 2500.0 if i in self.arm_joint_indices else 500.0
            robot.joint_target_kd[i] = 120.0 if i in self.arm_joint_indices else 20.0
            robot.joint_target_mode[i] = int(JointTargetMode.POSITION_VELOCITY)
        for i in self.arm_joint_indices:
            robot.joint_effort_limit[i] = 200.0
        return robot

    @staticmethod
    def _add_scene(builder: newton.ModelBuilder) -> tuple[int, int]:
        builder.add_shape_box(
            body=-1,
            xform=wp.transform(wp.vec3(0.48, 0.0, 0.72), wp.quat_identity()),
            hx=0.34,
            hy=0.50,
            hz=0.04,
            cfg=newton.ModelBuilder.ShapeConfig(mu=0.8, kd=50.0),
        )
        box_size = wp.vec3(0.10, 0.10, 0.10)
        box_body = builder.add_body(xform=wp.transform(wp.vec3(0.48, 0.0, 0.81), wp.quat_identity()))
        box_shape = builder.add_shape_box(
            body=box_body,
            hx=0.5 * box_size[0],
            hy=0.5 * box_size[1],
            hz=0.5 * box_size[2],
            cfg=newton.ModelBuilder.ShapeConfig(mu=1.2, kd=80.0, density=12000.0),
        )
        builder.add_ground_plane()
        return box_body, box_shape

    @staticmethod
    def _find_body(model: newton.Model, name: str) -> int:
        for index, label in enumerate(model.body_label):
            if label.endswith(f"/{name}") or label == name:
                return index
        raise RuntimeError(f"Body {name!r} was not found in the teleop model")

    def _setup_ik(self) -> None:
        self.pos_objectives = []
        self.rot_objectives = []
        for link_index, target in zip(self.ee_indices, self.target_tfs, strict=True):
            target_pos = wp.transform_get_translation(target)
            target_rot = wp.transform_get_rotation(target)
            self.pos_objectives.append(
                ik.IKObjectivePosition(
                    link_index=link_index,
                    link_offset=wp.vec3(0.0, 0.0, 0.0),
                    target_positions=wp.array([target_pos], dtype=wp.vec3, device=self.device),
                )
            )
            self.rot_objectives.append(
                ik.IKObjectiveRotation(
                    link_index=link_index,
                    link_offset_rotation=wp.quat_identity(),
                    target_rotations=wp.array([_quat_to_vec4(target_rot)], dtype=wp.vec4, device=self.device),
                )
            )

        nominal_q = self.model_ik.joint_q.numpy()
        limit_lower = self.model_ik.joint_limit_lower.numpy()
        limit_upper = self.model_ik.joint_limit_upper.numpy()
        fixed_joint_indices = set(range(self.model_ik.joint_coord_count)) - set(self.arm_joint_indices)
        for index in fixed_joint_indices:
            limit_lower[index] = nominal_q[index] - 1.0e-4
            limit_upper[index] = nominal_q[index] + 1.0e-4
        joint_limit_objective = ik.IKObjectiveJointLimit(
            joint_limit_lower=wp.array(limit_lower, device=self.device),
            joint_limit_upper=wp.array(limit_upper, device=self.device),
            weight=100.0,
        )
        self.joint_q_ik = wp.array(self.model_ik.joint_q, shape=(1, self.model_ik.joint_coord_count))
        self.ik_solver = ik.IKSolver(
            model=self.model_ik,
            n_problems=1,
            objectives=[*self.pos_objectives, *self.rot_objectives, joint_limit_objective],
            lambda_initial=0.1,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
        )

    def _update_command(self) -> None:
        cycle_position = self.sim_time / self.sweep_half_period
        active_hand = int(cycle_position) % 2
        progress = cycle_position % 1.0
        desired_positions = [
            np.array([0.24, 0.20, 0.95], dtype=np.float32),
            np.array([0.24, -0.20, 0.95], dtype=np.float32),
        ]

        sweep_start = 0.20 if active_hand == 0 else -0.20
        sweep_end = -0.20 if active_hand == 0 else 0.20
        active_position = desired_positions[active_hand]
        # Advance above the box so the approach does not add a forward impulse.
        if progress < 0.2:
            active_position[0] = 0.24 + (0.38 - 0.24) * progress / 0.2
        elif progress < 0.3:
            active_position[0] = 0.38
            active_position[2] = 0.95 + (0.84 - 0.95) * (progress - 0.2) / 0.1
        elif progress < 0.8:
            sweep_progress = (progress - 0.3) / 0.5
            active_position[0] = 0.38
            active_position[1] = sweep_start + (sweep_end - sweep_start) * sweep_progress
            active_position[2] = 0.84
        elif progress < 0.9:
            active_position[0] = 0.38
            active_position[1] = sweep_end
            active_position[2] = 0.84 + (0.95 - 0.84) * (progress - 0.8) / 0.1
        else:
            active_position[0] = 0.38 + (0.24 - 0.38) * (progress - 0.9) / 0.1
            active_position[1] = sweep_end

        max_delta = self.linear_speed * self.frame_dt
        for index, (target, desired_position) in enumerate(zip(self.target_tfs, desired_positions, strict=True)):
            current_position = _vec3_to_np(wp.transform_get_translation(target))
            current_position += np.clip(desired_position - current_position, -max_delta, max_delta)
            self.target_tfs[index] = wp.transform(wp.vec3(*current_position), self.target_rotations[index])

    def _write_targets(self) -> None:
        wp.launch(
            _write_robot_targets,
            dim=self.model_ik.joint_coord_count,
            inputs=[
                self.joint_q_ik,
                self.nominal_joint_q,
                self.arm_joint_mask,
                self.previous_joint_target_q,
                self.frame_dt,
            ],
            outputs=[self.control.joint_target_q, self.control.joint_target_qd],
            device=self.device,
        )

    def _simulate(self) -> None:
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.state_1.clear_forces()
            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def _record_workload_state(self) -> None:
        body_q = self.state_0.body_q.numpy()
        body_qd = self.state_0.body_qd.numpy()
        if not np.isfinite(body_q).all() or not np.isfinite(body_qd).all():
            raise RuntimeError("The G1 pushing workload produced non-finite body state")

        position_errors = []
        rotation_errors = []
        for ee_index, target in zip(self.ee_indices, self.target_tfs, strict=True):
            end_effector = body_q[ee_index]
            target_pos = _vec3_to_np(wp.transform_get_translation(target))
            position_errors.append(float(np.linalg.norm(target_pos - end_effector[:3])))
            target_rot = _quat_to_np(wp.transform_get_rotation(target))
            quat_dot = min(1.0, abs(float(np.dot(end_effector[3:7], target_rot))))
            rotation_errors.append(2.0 * math.acos(quat_dot))
        self.stats.add("target_error_m", float(np.mean(position_errors)))
        self.stats.add("target_rotation_error_rad", float(np.mean(rotation_errors)))

        object_position = body_q[self.object_body_index, :3]
        self.stats.add("object_displacement_m", float(np.linalg.norm(object_position - self.initial_object_position)))
        self.stats.add("hand_object_contact", float(self._has_hand_object_contact()))

    def _has_hand_object_contact(self) -> bool:
        if self.contacts is not None:
            self.solver.update_contacts(self.contacts, self.state_0)
            contact_count = int(self.contacts.rigid_contact_count.numpy()[0])
            shape0 = self.contacts.rigid_contact_shape0.numpy()[:contact_count]
            shape1 = self.contacts.rigid_contact_shape1.numpy()[:contact_count]
        else:
            contact_count = int(self.solver.mj_data.ncon)
            geom_pairs = self.solver.mj_data.contact.geom[:contact_count]
            shape0 = self.mjc_geom_to_newton_shape[geom_pairs[:, 0]]
            shape1 = self.mjc_geom_to_newton_shape[geom_pairs[:, 1]]

        for first_shape, second_shape in zip(shape0, shape1, strict=True):
            if (first_shape == self.object_shape_index and second_shape in self.hand_shape_indices) or (
                second_shape == self.object_shape_index and first_shape in self.hand_shape_indices
            ):
                return True
        return False

    def step(self, *, collect_workload_state: bool = False) -> None:
        frame_start = time.perf_counter()
        self._update_command()
        for pos_objective, rot_objective, target in zip(
            self.pos_objectives, self.rot_objectives, self.target_tfs, strict=True
        ):
            pos_objective.set_target_position(0, wp.transform_get_translation(target))
            rot_objective.set_target_rotation(0, _quat_to_vec4(wp.transform_get_rotation(target)))
        if self.graph_ik is None:
            self.ik_solver.step(self.joint_q_ik, self.joint_q_ik, iterations=16)
        else:
            wp.capture_launch(self.graph_ik)
        self._write_targets()
        if self.graph_sim is None:
            self._simulate()
        else:
            wp.capture_launch(self.graph_sim)

        if collect_workload_state:
            self._record_workload_state()
        else:
            wp.synchronize_device(self.device)
            self.stats.add("local_loop_ms", (time.perf_counter() - frame_start) * 1000.0)

        self.sim_time += self.frame_dt
        self.frame_index += 1

    def clear_metrics(self) -> None:
        self.stats.clear()


def _skip_unavailable_mode(mode: _TeleopMode) -> None:
    if mode.requires_cuda and wp.get_cuda_device_count() == 0:
        raise SkipNotImplemented
    if mode.requires_cuda_graph:
        with wp.ScopedDevice(mode.device):
            if not wp.is_mempool_enabled(wp.get_device()):
                raise SkipNotImplemented


class _TeleopMuJoCoBenchmark:
    """Shared setup for scripted synchronous teleop benchmarks."""

    params: ClassVar[tuple[tuple[str, ...]]] = (tuple(_TELEOP_MODES.keys()),)
    param_names: ClassVar[list[str]] = ["mode"]
    repeat = 3
    number = 1
    rounds = 2
    timeout = 600
    num_frames = 300
    warmup_frames = 60

    def setup(self, mode: str) -> None:
        self.mode = _TELEOP_MODES[mode]
        _skip_unavailable_mode(self.mode)
        previous_target_layout = newton.use_coord_layout_targets
        newton.use_coord_layout_targets = True
        try:
            with wp.ScopedDevice(self.mode.device):
                self.loop = _TeleopLoop(self.mode, self.num_frames)
        finally:
            newton.use_coord_layout_targets = previous_target_layout

        self._step_frames(self.warmup_frames)
        self.loop.clear_metrics()

    def time_teleop_loop(self, mode: str) -> None:
        self._step_frames(self.num_frames)

    def _step_frames(self, frame_count: int, *, collect_workload_state: bool = False) -> None:
        with wp.ScopedDevice(self.mode.device):
            for _ in range(frame_count):
                self.loop.step(collect_workload_state=collect_workload_state)

    def _measure_frames(self, *, collect_workload_state: bool = False) -> None:
        self.loop.clear_metrics()
        self._step_frames(self.num_frames, collect_workload_state=collect_workload_state)

    def _validate_workload(self) -> None:
        if self.loop.stats.summary("hand_object_contact")[2] == 0.0:
            raise RuntimeError("The G1 pushing workload did not produce hand-object contact")
        if self.loop.stats.summary("object_displacement_m")[2] < 0.01:
            raise RuntimeError("The G1 pushing workload did not move the object by at least 0.01 m")


class FastTeleopMuJoCo(_TeleopMuJoCoBenchmark):
    """Pull-request smoke benchmarks across GPU and CPU execution modes."""

    params: ClassVar[tuple[tuple[str, ...]]] = (tuple(_TELEOP_MODES),)
    repeat = 2
    num_frames = 120
    warmup_frames = 30

    def track_mean_loop_ms(self, mode: str) -> float:
        self._measure_frames()
        return self.loop.stats.summary("local_loop_ms")[0]

    track_mean_loop_ms.unit = "ms/frame"

    def track_p95_loop_ms(self, mode: str) -> float:
        self._measure_frames()
        return self.loop.stats.summary("local_loop_ms")[1]

    track_p95_loop_ms.unit = "ms/frame"


class TeleopMuJoCo(_TeleopMuJoCoBenchmark):
    """Nightly teleop benchmark covering GPU and CPU solver backends."""

    def track_mean_loop_ms(self, mode: str) -> float:
        self._measure_frames()
        return self.loop.stats.summary("local_loop_ms")[0]

    track_mean_loop_ms.unit = "ms/frame"

    def track_p95_loop_ms(self, mode: str) -> float:
        self._measure_frames()
        return self.loop.stats.summary("local_loop_ms")[1]

    track_p95_loop_ms.unit = "ms/frame"

    def track_frame_overrun_pct(self, mode: str) -> float:
        self._measure_frames()
        values = self.loop.stats.values["local_loop_ms"]
        overruns = sum(value > self.loop.frame_dt * 1000.0 for value in values)
        return 100.0 * overruns / len(values)

    track_frame_overrun_pct.unit = "%"

    def track_loop_time_cv_pct(self, mode: str) -> float:
        self._measure_frames()
        return self.loop.stats.coefficient_of_variation_pct("local_loop_ms")

    track_loop_time_cv_pct.unit = "%"

    def track_real_time_factor(self, mode: str) -> float:
        self._measure_frames()
        mean_loop_ms = self.loop.stats.summary("local_loop_ms")[0]
        return self.loop.frame_dt * 1000.0 / mean_loop_ms

    track_real_time_factor.unit = "x"

    def track_sustainable_physics_step_hz(self, mode: str) -> float:
        self._measure_frames()
        mean_loop_ms = self.loop.stats.summary("local_loop_ms")[0]
        return self.loop.sim_substeps * 1000.0 / mean_loop_ms

    track_sustainable_physics_step_hz.unit = "Hz"

    def track_mean_target_error_m(self, mode: str) -> float:
        self._measure_frames(collect_workload_state=True)
        return self.loop.stats.summary("target_error_m")[0]

    track_mean_target_error_m.unit = "m"

    def track_mean_target_rotation_error_rad(self, mode: str) -> float:
        self._measure_frames(collect_workload_state=True)
        return self.loop.stats.summary("target_rotation_error_rad")[0]

    track_mean_target_rotation_error_rad.unit = "rad"

    def track_hand_object_contact_frame_pct(self, mode: str) -> float:
        self._measure_frames(collect_workload_state=True)
        self._validate_workload()
        return 100.0 * self.loop.stats.summary("hand_object_contact")[0]

    track_hand_object_contact_frame_pct.unit = "%"

    def track_object_displacement_m(self, mode: str) -> float:
        self._measure_frames(collect_workload_state=True)
        self._validate_workload()
        return self.loop.stats.summary("object_displacement_m")[2]

    track_object_displacement_m.unit = "m"


if __name__ == "__main__":
    import argparse

    from newton.utils import run_benchmark

    benchmark_list = {
        "FastTeleopMuJoCo": FastTeleopMuJoCo,
        "TeleopMuJoCo": TeleopMuJoCo,
    }
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-b", "--bench", action="append", choices=benchmark_list.keys())
    args = parser.parse_known_args()[0]

    benchmarks = args.bench if args.bench is not None else benchmark_list.keys()
    for key in benchmarks:
        run_benchmark(benchmark_list[key])
