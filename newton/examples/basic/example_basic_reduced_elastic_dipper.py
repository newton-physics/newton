# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

###########################################################################
# Example Basic Reduced Elastic Dipper
#
# Demonstrates a flexible dipper arm pinned to a vertical support, driven by a
# lower prismatic actuator, and loaded by a suspended rigid tip weight.
#
# Command: python -m newton.examples basic_reduced_elastic_dipper
#
###########################################################################

import math

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.basic._reduced_elastic import (
    beam_render_sample_points,
    elastic_shape_volume_ratio,
    init_elastic_solver_metric_tracking,
    joint_endpoint_world,
    transform_point,
    update_elastic_solver_metric_tracking,
)


def _axis_xz(theta: float) -> np.ndarray:
    return np.array([math.cos(theta), 0.0, math.sin(theta)], dtype=float)


def _quat_from_xz_angle(theta: float):
    return wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), -theta)


def _angle_from_xz_vector(vec: np.ndarray) -> float:
    return math.atan2(float(vec[2]), float(vec[0]))


def _rotate_local_xz(theta: float, point: np.ndarray) -> np.ndarray:
    return np.array(
        [
            math.cos(theta) * float(point[0]) - math.sin(theta) * float(point[2]),
            float(point[1]),
            math.sin(theta) * float(point[0]) + math.cos(theta) * float(point[2]),
        ],
        dtype=float,
    )


def _set_camera_from_bounds(viewer, bounds_min: np.ndarray, bounds_max: np.ndarray, offset_dir: np.ndarray):
    center = 0.5 * (bounds_min + bounds_max)
    extent = float(np.max(bounds_max - bounds_min))
    distance = max(extent, 1.0) / (2.0 * math.tan(math.radians(45.0) * 0.5)) * 1.35
    offset_dir = offset_dir / np.linalg.norm(offset_dir)
    pos = center + offset_dir * distance
    front = center - pos
    front /= np.linalg.norm(front)
    yaw = math.degrees(math.atan2(front[1], front[0]))
    pitch = math.degrees(math.asin(front[2]))
    viewer.set_camera(wp.vec3(*pos), pitch, yaw)


class Example:
    solver_iterations = 32

    def __init__(self, viewer, args):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 3
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.step_count = 0

        self.viewer = viewer
        self.args = args
        self.show_elastic_strain = True

        self.gravity = 9.81
        self.arm_length = 1.30
        self.arm_hy = 0.055
        self.arm_hz = 0.038
        self.arm_pivot = np.array([0.48, 0.0, 0.82], dtype=float)
        self.arm_theta0 = math.pi
        self.arm_axis0 = _axis_xz(self.arm_theta0)
        self.fulcrum_local = np.array([-0.22 * self.arm_length, 0.0, 0.0], dtype=float)
        self.arm_center = self.arm_pivot - _rotate_local_xz(self.arm_theta0, self.fulcrum_local)
        self.arm_attach_local = np.array([-0.48 * self.arm_length, 0.0, 0.055], dtype=float)
        self.arm_attach_from_pivot_local = self.arm_attach_local - self.fulcrum_local
        self.lower_pivot = np.array([0.62, 0.0, 0.22], dtype=float)
        self.tip_local = np.array([0.5 * self.arm_length, 0.0, 0.0], dtype=float)
        self.tip_world0 = self.arm_center + _rotate_local_xz(self.arm_theta0, self.tip_local)
        self.weight_offset = 0.18
        self.weight_mass = 3.5
        self.drive_amplitude = 0.08
        self.drive_frequency = 0.35
        self.max_joint_residual = 0.0
        self.max_drive_error = 0.0
        self.max_tip_bend = 0.0
        self.max_tip_vertical_motion = 0.0
        self.max_volume_ratio_error = 0.0
        self.tip_z_min = float(self.tip_world0[2])
        self.tip_z_max = float(self.tip_world0[2])
        self.max_mode_step = 0.0
        self.max_mode_accel = 0.0
        self._previous_modes = None
        self._previous_previous_modes = None
        self._last_drive_target = 0.0
        self._last_target_theta = self.arm_theta0

        arm_attach_world = self.arm_center + _rotate_local_xz(self.arm_theta0, self.arm_attach_local)
        cylinder_axis = arm_attach_world - self.lower_pivot
        self.cylinder_length = float(np.linalg.norm(cylinder_axis))
        self.cylinder_dir = cylinder_axis / self.cylinder_length
        self.cylinder_theta0 = _angle_from_xz_vector(cylinder_axis)
        self.cylinder_quat = _quat_from_xz_angle(self.cylinder_theta0)

        sample_points = beam_render_sample_points(
            self.arm_length,
            self.arm_hy,
            self.arm_hz,
            extra_points=(
                tuple(self.fulcrum_local),
                tuple(self.arm_attach_local),
                tuple(self.tip_local),
            ),
        )
        arm_basis = newton.ModalGeneratorBeam(
            length=self.arm_length,
            half_width_y=self.arm_hy,
            half_width_z=self.arm_hz,
            mode_specs=[
                {"type": newton.ModalGeneratorBeam.Mode.AXIAL},
                {
                    "type": newton.ModalGeneratorBeam.Mode.BENDING_Z,
                    "boundary": newton.ModalGeneratorBeam.Boundary.CANTILEVER_TIP,
                    "order": 1,
                },
                {
                    "type": newton.ModalGeneratorBeam.Mode.BENDING_Z,
                    "boundary": newton.ModalGeneratorBeam.Boundary.CANTILEVER_TIP,
                    "order": 2,
                },
                {
                    "type": newton.ModalGeneratorBeam.Mode.BENDING_Z,
                    "boundary": newton.ModalGeneratorBeam.Boundary.CANTILEVER_TIP,
                    "order": 3,
                },
            ],
            density=420.0,
            young_modulus=7.5e5,
            damping_ratio=0.018,
            label="dipper_arm_basis",
        ).build(sample_points=sample_points)

        builder = newton.ModelBuilder(gravity=-self.gravity)
        builder.add_ground_plane()

        shape_cfg = newton.ModelBuilder.ShapeConfig()
        shape_cfg.density = 0.0
        shape_cfg.has_shape_collision = False
        shape_cfg.has_particle_collision = False

        builder.add_shape_box(
            -1,
            xform=wp.transform(wp.vec3(0.48, 0.0, 0.475), wp.quat_identity()),
            hx=0.045,
            hy=0.14,
            hz=0.325,
            cfg=shape_cfg,
            label="rear_support_post",
        )
        builder.add_shape_box(
            -1,
            xform=wp.transform(wp.vec3(0.56, 0.0, 0.075), wp.quat_identity()),
            hx=0.18,
            hy=0.16,
            hz=0.075,
            cfg=shape_cfg,
            label="actuator_base",
        )
        builder.add_shape_box(
            -1,
            xform=wp.transform(wp.vec3(0.62, 0.0, 0.25), wp.quat_identity()),
            hx=0.035,
            hy=0.06,
            hz=0.18,
            cfg=shape_cfg,
            label="actuator_stand",
        )

        arm_inertia = wp.mat33(0.025, 0.0, 0.0, 0.0, 0.11, 0.0, 0.0, 0.0, 0.11)
        rigid_inertia = wp.mat33(0.04, 0.0, 0.0, 0.0, 0.04, 0.0, 0.0, 0.0, 0.04)
        actuator_inertia = wp.mat33(0.012, 0.0, 0.0, 0.0, 0.018, 0.0, 0.0, 0.0, 0.012)
        self.arm = builder.add_body_elastic(
            xform=wp.transform(wp.vec3(*self.arm_center), _quat_from_xz_angle(self.arm_theta0)),
            mass=1.2,
            inertia=arm_inertia,
            mode_q=[0.0, 0.0, 0.0, 0.0],
            mode_mass=[0.16, 0.09, 0.07, 0.055],
            mode_stiffness=[1.8e5, 264.0, 6.4e4, 2.0e5],
            mode_damping=[4.0, 4.65, 0.60, 0.85],
            modal_basis=arm_basis,
            label="flexible_dipper_arm",
        )
        self.barrel = builder.add_body(
            xform=wp.transform(wp.vec3(*self.lower_pivot), self.cylinder_quat),
            mass=0.55,
            inertia=actuator_inertia,
            is_kinematic=True,
            label="actuator_barrel",
        )
        self.rod = builder.add_body(
            xform=wp.transform(wp.vec3(*arm_attach_world), self.cylinder_quat),
            mass=0.35,
            inertia=actuator_inertia,
            is_kinematic=True,
            label="actuator_rod",
        )
        self.weight = builder.add_body(
            xform=wp.transform(
                wp.vec3(*(self.tip_world0 + np.array([0.0, 0.0, -self.weight_offset]))), wp.quat_identity()
            ),
            mass=self.weight_mass,
            inertia=rigid_inertia,
            label="suspended_tip_weight",
        )

        builder.add_shape_box(self.arm, hx=0.5 * self.arm_length, hy=self.arm_hy, hz=self.arm_hz, cfg=shape_cfg)
        builder.add_shape_box(
            self.barrel,
            xform=wp.transform(wp.vec3(0.32 * self.cylinder_length, 0.0, 0.0), wp.quat_identity()),
            hx=0.32 * self.cylinder_length,
            hy=0.028,
            hz=0.028,
            cfg=shape_cfg,
        )
        builder.add_shape_box(
            self.rod,
            xform=wp.transform(wp.vec3(-0.20, 0.0, 0.0), wp.quat_identity()),
            hx=0.20,
            hy=0.018,
            hz=0.018,
            cfg=shape_cfg,
        )
        builder.add_shape_box(
            self.weight,
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.5 * self.weight_offset), wp.quat_identity()),
            hx=0.012,
            hy=0.012,
            hz=0.5 * self.weight_offset,
            cfg=shape_cfg,
        )
        builder.add_shape_box(self.weight, hx=0.09, hy=0.09, hz=0.09, cfg=shape_cfg)

        self.j_arm_root = builder.add_joint_revolute(
            parent=-1,
            child=self.arm,
            axis=(0.0, 1.0, 0.0),
            parent_xform=wp.transform(wp.vec3(*self.arm_pivot), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(*self.fulcrum_local), wp.quat_identity()),
            label="support_to_dipper_arm",
        )
        self.j_cylinder_base = builder.add_joint_revolute(
            parent=-1,
            child=self.barrel,
            axis=(0.0, 1.0, 0.0),
            parent_xform=wp.transform(wp.vec3(*self.lower_pivot), wp.quat_identity()),
            child_xform=wp.transform_identity(),
            label="base_to_actuator_barrel",
        )
        self.j_cylinder_slide = builder.add_joint_prismatic(
            parent=self.barrel,
            child=self.rod,
            axis=(1.0, 0.0, 0.0),
            parent_xform=wp.transform(wp.vec3(self.cylinder_length, 0.0, 0.0), wp.quat_identity()),
            child_xform=wp.transform_identity(),
            target_pos=0.0,
            target_ke=2.4e4,
            target_kd=280.0,
            limit_lower=-0.22,
            limit_upper=0.22,
            limit_ke=2.0e4,
            limit_kd=50.0,
            label="hydraulic_prismatic_drive",
        )
        self.j_rod_arm = builder.add_joint_revolute(
            parent=self.rod,
            child=self.arm,
            axis=(0.0, 1.0, 0.0),
            parent_xform=wp.transform_identity(),
            child_xform=wp.transform(wp.vec3(*self.arm_attach_local), wp.quat_identity()),
            label="actuator_rod_to_dipper_arm",
        )
        self.j_tip_weight = builder.add_joint_revolute(
            parent=self.arm,
            child=self.weight,
            axis=(0.0, 1.0, 0.0),
            parent_xform=wp.transform(wp.vec3(*self.tip_local), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(0.0, 0.0, self.weight_offset), wp.quat_identity()),
            label="dipper_tip_to_weight",
        )
        builder.color()

        self.model = builder.finalize()
        self.device = self.model.device
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = None

        self.elastic_joint = int(self.model.elastic_joint.numpy()[0])
        self.elastic_q_start = int(self.model.joint_q_start.numpy()[self.elastic_joint])
        self.elastic_qd_start = int(self.model.joint_qd_start.numpy()[self.elastic_joint])
        self.drive_q_start = int(self.model.joint_target_q_start.numpy()[self.j_cylinder_slide])
        self._target_pos = self.control.joint_target_q.numpy()
        self._joint_f = self.control.joint_f.numpy()
        self.max_volume_ratio_error = abs(elastic_shape_volume_ratio(self.model, self.state_0) - 1.0)

        self.solver = newton.solvers.SolverVBD(
            self.model,
            iterations=self.solver_iterations,
            rigid_joint_linear_k_start=5.0e5,
            rigid_joint_angular_k_start=5.0e5,
            rigid_joint_linear_ke=5.0e5,
            rigid_joint_angular_ke=5.0e5,
            rigid_joint_linear_kd=3.0e-5,
            rigid_joint_angular_kd=0.0,
            rigid_joint_adaptive_stiffness=False,
        )
        init_elastic_solver_metric_tracking(self)

        self.viewer.set_model(self.model)
        self.viewer.show_elastic_strain = True
        self.viewer.elastic_strain_color_max = 0.04
        bounds_min = np.array([-0.55, -0.20, 0.0], dtype=float)
        bounds_max = np.array([1.05, 0.20, 1.24], dtype=float)
        _set_camera_from_bounds(self.viewer, bounds_min, bounds_max, np.array([-0.45, -1.0, 0.34], dtype=float))

    def _mode_values(self) -> np.ndarray:
        return self.state_0.joint_q.numpy()[self.elastic_q_start + 7 : self.elastic_q_start + 11]

    def _tip_bend(self) -> float:
        body_q = self.state_0.body_q.numpy()
        rigid_tip = transform_point(body_q[self.arm], self.tip_local)
        deformed_tip = joint_endpoint_world(self.model, self.state_0, self.j_tip_weight, "parent")
        return float(np.linalg.norm(deformed_tip - rigid_tip))

    def _drive_stroke(self) -> float:
        lower = joint_endpoint_world(self.model, self.state_0, self.j_cylinder_base, "parent")
        upper = joint_endpoint_world(self.model, self.state_0, self.j_rod_arm, "parent")
        return float(np.linalg.norm(upper - lower) - self.cylinder_length)

    def _joint_residuals(self) -> list[float]:
        joints = [self.j_arm_root, self.j_cylinder_base, self.j_rod_arm, self.j_tip_weight]
        return [
            float(
                np.linalg.norm(
                    joint_endpoint_world(self.model, self.state_0, joint, "parent")
                    - joint_endpoint_world(self.model, self.state_0, joint, "child")
                )
            )
            for joint in joints
        ]

    def _drive_target(self, t: float) -> float:
        hold = 0.18
        if t < hold:
            return 0.0
        elapsed = t - hold
        ramp = min(elapsed / 0.45, 1.0)
        envelope = ramp * ramp * (3.0 - 2.0 * ramp)
        phase = 2.0 * math.pi * self.drive_frequency * elapsed
        return -self.drive_amplitude * envelope * math.sin(phase)

    def _target_attachment_world(self, theta: float) -> np.ndarray:
        return self.arm_pivot + _rotate_local_xz(theta, self.arm_attach_from_pivot_local)

    def _solve_target_theta(self, stroke: float) -> float:
        target_length = max(0.05, self.cylinder_length + stroke)
        theta = self._last_target_theta

        for _ in range(16):
            arm_offset = _rotate_local_xz(theta, self.arm_attach_from_pivot_local)
            endpoint = self.arm_pivot + arm_offset
            delta = endpoint - self.lower_pivot
            length = float(np.linalg.norm(delta))
            if length <= 1.0e-8:
                break

            c = math.cos(theta)
            s = math.sin(theta)
            local = self.arm_attach_from_pivot_local
            endpoint_deriv = np.array(
                [
                    -s * float(local[0]) - c * float(local[2]),
                    0.0,
                    c * float(local[0]) - s * float(local[2]),
                ],
                dtype=float,
            )
            residual = length - target_length
            if abs(residual) <= 1.0e-6:
                return theta

            gradient = float(np.dot(delta, endpoint_deriv) / length)
            if abs(gradient) <= 1.0e-7:
                break
            theta -= max(-0.18, min(0.18, residual / gradient))

        # Keep the actuator on the same linkage branch if Newton iteration gets
        # close to the toggle position.
        best_theta = theta
        best_error = float("inf")
        for candidate in np.linspace(self._last_target_theta - 0.45, self._last_target_theta + 0.45, 181):
            length = float(np.linalg.norm(self._target_attachment_world(float(candidate)) - self.lower_pivot))
            error = abs(length - target_length)
            if error < best_error:
                best_theta = float(candidate)
                best_error = error
        return best_theta

    def _set_controls(self, t: float):
        self._last_drive_target = self._drive_target(t)
        self._target_pos[self.drive_q_start] = self._last_drive_target
        self.control.joint_target_q.assign(self._target_pos)

        self._joint_f[:] = 0.0
        self.control.joint_f.assign(self._joint_f)

        target_theta = self._solve_target_theta(self._last_drive_target)
        target_endpoint = self._target_attachment_world(target_theta)
        cylinder_axis = target_endpoint - self.lower_pivot
        cylinder_length = float(np.linalg.norm(cylinder_axis))
        if cylinder_length <= 1.0e-8:
            cylinder_quat = self.cylinder_quat
        else:
            cylinder_theta = _angle_from_xz_vector(cylinder_axis)
            cylinder_quat = _quat_from_xz_angle(cylinder_theta)
        self._last_target_theta = target_theta

        body_q = self.state_0.body_q.numpy()
        body_q[self.barrel, :3] = self.lower_pivot
        body_q[self.barrel, 3:7] = np.array(cylinder_quat, dtype=np.float32)
        body_q[self.rod, :3] = target_endpoint
        body_q[self.rod, 3:7] = np.array(cylinder_quat, dtype=np.float32)
        self.state_0.body_q.assign(body_q)

    def _update_metrics(self):
        update_elastic_solver_metric_tracking(self)
        modes = self._mode_values().copy()
        if self.step_count > 30 and self._previous_modes is not None:
            step = modes - self._previous_modes
            self.max_mode_step = max(self.max_mode_step, float(np.max(np.abs(step))))
        if self.step_count > 30 and self._previous_modes is not None and self._previous_previous_modes is not None:
            accel = modes - 2.0 * self._previous_modes + self._previous_previous_modes
            self.max_mode_accel = max(self.max_mode_accel, float(np.max(np.abs(accel))))
        self._previous_previous_modes = self._previous_modes
        self._previous_modes = modes

        self.max_joint_residual = max(self.max_joint_residual, *self._joint_residuals())
        self.max_drive_error = max(self.max_drive_error, abs(self._drive_stroke() - self._last_drive_target))
        self.max_tip_bend = max(self.max_tip_bend, self._tip_bend())
        tip = joint_endpoint_world(self.model, self.state_0, self.j_tip_weight, "parent")
        self.tip_z_min = min(self.tip_z_min, float(tip[2]))
        self.tip_z_max = max(self.tip_z_max, float(tip[2]))
        self.max_tip_vertical_motion = max(self.max_tip_vertical_motion, self.tip_z_max - self.tip_z_min)
        self.max_volume_ratio_error = max(
            self.max_volume_ratio_error, abs(elastic_shape_volume_ratio(self.model, self.state_0) - 1.0)
        )

    def simulate(self):
        for substep in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self._set_controls(self.sim_time + substep * self.sim_dt)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt
        self.step_count += 1
        self._update_metrics()

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        if not np.isfinite(self.state_0.body_q.numpy()).all():
            raise AssertionError("body transforms contain non-finite values")
        if not np.isfinite(self.state_0.joint_q.numpy()).all():
            raise AssertionError("joint coordinates contain non-finite values")
        if self.max_joint_residual > 5.0e-3:
            raise AssertionError(f"dipper arm joint residual too large: {self.max_joint_residual}")
        if self.max_drive_error > 1.4e-1:
            raise AssertionError(f"hydraulic drive stroke error too large: {self.max_drive_error}")
        if self.max_tip_bend < 2.5e-2:
            raise AssertionError(f"dipper arm bending was too small: {self.max_tip_bend}")
        if self.max_tip_bend > 1.2e-1:
            raise AssertionError(f"dipper arm bending was too large: {self.max_tip_bend}")
        if self.max_tip_vertical_motion < 2.3e-1:
            raise AssertionError("dipper tip did not move under the driven actuator and payload")
        if self.max_tip_vertical_motion > 6.5e-1:
            raise AssertionError(f"dipper tip vertical motion was too large: {self.max_tip_vertical_motion}")
        if self.max_mode_step > 0.01 or self.max_mode_accel > 0.003:
            raise AssertionError(
                f"dipper arm modal ringing too high: step={self.max_mode_step}, accel={self.max_mode_accel}"
            )
        if np.max(np.abs(self._mode_values())) > 0.45:
            raise AssertionError(f"dipper arm mode amplitudes are out of range: {self._mode_values()}")
        if self.final_modal_solve_residual_ratio > 0.02:
            raise AssertionError(f"dipper modal solve residual ratio too high: {self.final_modal_solve_residual_ratio}")
        if self.final_modal_update_norm > 1.0e-4 or self.max_modal_update_norm > 2.0e-4:
            raise AssertionError(
                f"dipper modal update too large: final={self.final_modal_update_norm}, max={self.max_modal_update_norm}"
            )
        if self.max_volume_ratio_error > 0.5:
            raise AssertionError(f"dipper arm volume changed by {self.max_volume_ratio_error:.3f}")


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
