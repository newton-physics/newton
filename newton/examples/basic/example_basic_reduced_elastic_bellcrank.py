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
# Example Basic Reduced Elastic Bellcrank
#
# Demonstrates a two-stage bell-crank transfer with two reduced elastic
# pushrods connecting rigid rockers.
#
# Command: python -m newton.examples basic_reduced_elastic_bellcrank
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
    joint_endpoint_world,
)


def _quat_from_angle_z(theta: float):
    return wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), theta)


def _angle_from_quat_z(q: np.ndarray) -> float:
    return 2.0 * math.atan2(float(q[2]), float(q[3]))


def _beam_basis(length: float, half_width_y: float, half_width_z: float, label: str):
    sample_points = beam_render_sample_points(
        length,
        half_width_y,
        half_width_z,
        extra_points=((-0.5 * length, 0.0, 0.0), (0.5 * length, 0.0, 0.0)),
    )
    return newton.ModalGeneratorBeam(
        length=length,
        half_width_y=half_width_y,
        half_width_z=half_width_z,
        mode_specs=[
            {"type": newton.ModalGeneratorBeam.Mode.AXIAL},
            {
                "type": newton.ModalGeneratorBeam.Mode.BENDING_Y,
                "boundary": newton.ModalGeneratorBeam.Boundary.PINNED_PINNED,
                "order": 1,
            },
        ],
        density=300.0,
        young_modulus=8.0e5,
        damping_ratio=0.03,
        label=label,
    ).build(sample_points=sample_points)


class Example:
    def __init__(self, viewer, args):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 3
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.step_count = 0

        self.viewer = viewer
        self.args = args

        self.z = 0.18
        self.input_length = 0.32
        self.output_length = 0.34
        self.input_theta0 = 0.70
        self.output_theta0 = 2.35
        self.rod_hy = 0.027
        self.rod_hz = 0.023
        self.rod1_q0 = 0.018
        self.rod2_q0 = 0.026
        self.input_pivot = np.array([0.0, 0.0, self.z], dtype=float)
        self.bell_pivot = np.array([0.68, 0.0, self.z], dtype=float)
        self.output_pivot = np.array([1.26, 0.0, self.z], dtype=float)
        self.bell_input_local = np.array([-0.24, 0.18, 0.0], dtype=float)
        self.bell_output_local = np.array([0.24, -0.14, 0.0], dtype=float)
        self.max_joint_residual = 0.0
        self.max_rod_length_error = 0.0
        self.max_volume_ratio_error = 0.0
        self.output_min = self.output_theta0
        self.output_max = self.output_theta0

        input_axis = np.array([math.cos(self.input_theta0), math.sin(self.input_theta0), 0.0], dtype=float)
        output_axis = np.array([math.cos(self.output_theta0), math.sin(self.output_theta0), 0.0], dtype=float)
        input_tip = self.input_pivot + self.input_length * input_axis
        bell_input_tip = self.bell_pivot + self.bell_input_local
        bell_output_tip = self.bell_pivot + self.bell_output_local
        output_tip = self.output_pivot + self.output_length * output_axis
        self.rod1_rest_length = float(np.linalg.norm(bell_input_tip - input_tip) - self.rod1_q0)
        self.rod2_rest_length = float(np.linalg.norm(output_tip - bell_output_tip) - self.rod2_q0)

        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        builder.add_ground_plane()

        inertia = wp.mat33(0.012, 0.0, 0.0, 0.0, 0.012, 0.0, 0.0, 0.0, 0.012)
        self.input_rocker = builder.add_body(
            xform=wp.transform(
                wp.vec3(*(self.input_pivot + 0.5 * self.input_length * input_axis)),
                _quat_from_angle_z(self.input_theta0),
            ),
            mass=1.0,
            inertia=inertia,
            label="input_rocker",
        )
        self.bellcrank = builder.add_body(
            xform=wp.transform(wp.vec3(*self.bell_pivot), wp.quat_identity()),
            mass=1.2,
            inertia=inertia,
            label="rigid_bellcrank",
        )
        self.output_rocker = builder.add_body(
            xform=wp.transform(
                wp.vec3(*(self.output_pivot + 0.5 * self.output_length * output_axis)),
                _quat_from_angle_z(self.output_theta0),
            ),
            mass=1.0,
            inertia=inertia,
            label="output_rocker",
        )

        rod1_axis = bell_input_tip - input_tip
        rod2_axis = output_tip - bell_output_tip
        rod1_theta = math.atan2(float(rod1_axis[1]), float(rod1_axis[0]))
        rod2_theta = math.atan2(float(rod2_axis[1]), float(rod2_axis[0]))
        rod1_basis = _beam_basis(self.rod1_rest_length, self.rod_hy, self.rod_hz, "bellcrank_pushrod_1")
        rod2_basis = _beam_basis(self.rod2_rest_length, self.rod_hy, self.rod_hz, "bellcrank_pushrod_2")
        self.rod1 = builder.add_body_elastic(
            xform=wp.transform(wp.vec3(*(0.5 * (input_tip + bell_input_tip))), _quat_from_angle_z(rod1_theta)),
            mass=0.35,
            inertia=inertia,
            mode_q=[self.rod1_q0, 0.012],
            mode_mass=[0.03, 0.025],
            mode_stiffness=[18000.0, 60.0],
            mode_damping=[2.0, 0.55],
            modal_basis=rod1_basis,
            label="elastic_input_pushrod",
        )
        self.rod2 = builder.add_body_elastic(
            xform=wp.transform(wp.vec3(*(0.5 * (bell_output_tip + output_tip))), _quat_from_angle_z(rod2_theta)),
            mass=0.45,
            inertia=inertia,
            mode_q=[self.rod2_q0, -0.016],
            mode_mass=[0.035, 0.025],
            mode_stiffness=[18000.0, 60.0],
            mode_damping=[2.0, 0.55],
            modal_basis=rod2_basis,
            label="elastic_output_pushrod",
        )

        shape_cfg = newton.ModelBuilder.ShapeConfig()
        shape_cfg.density = 0.0
        shape_cfg.has_shape_collision = False
        shape_cfg.has_particle_collision = False
        builder.add_shape_box(self.input_rocker, hx=0.5 * self.input_length, hy=0.025, hz=0.024, cfg=shape_cfg)
        builder.add_shape_box(self.output_rocker, hx=0.5 * self.output_length, hy=0.025, hz=0.024, cfg=shape_cfg)
        for local in (self.bell_input_local, self.bell_output_local):
            length = float(np.linalg.norm(local))
            angle = math.atan2(float(local[1]), float(local[0]))
            builder.add_shape_box(
                self.bellcrank,
                xform=wp.transform(wp.vec3(*(0.5 * local)), _quat_from_angle_z(angle)),
                hx=0.5 * length,
                hy=0.027,
                hz=0.025,
                cfg=shape_cfg,
            )
        builder.add_shape_box(self.rod1, hx=0.5 * self.rod1_rest_length, hy=self.rod_hy, hz=self.rod_hz, cfg=shape_cfg)
        builder.add_shape_box(self.rod2, hx=0.5 * self.rod2_rest_length, hy=self.rod_hy, hz=self.rod_hz, cfg=shape_cfg)

        self.j_input_ground = builder.add_joint_revolute(
            parent=-1,
            child=self.input_rocker,
            axis=(0.0, 0.0, 1.0),
            parent_xform=wp.transform(wp.vec3(*self.input_pivot), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(-0.5 * self.input_length, 0.0, 0.0), wp.quat_identity()),
            target_pos=self.input_theta0,
            target_ke=160.0,
            target_kd=0.5,
            label="ground_to_input_rocker",
        )
        self.j_input_rod = builder.add_joint_revolute(
            parent=self.input_rocker,
            child=self.rod1,
            axis=(0.0, 0.0, 1.0),
            parent_xform=wp.transform(wp.vec3(0.5 * self.input_length, 0.0, 0.0), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(-0.5 * self.rod1_rest_length, 0.0, 0.0), wp.quat_identity()),
            label="input_rocker_to_pushrod",
        )
        self.j_rod_bell_in = builder.add_joint_revolute(
            parent=self.rod1,
            child=self.bellcrank,
            axis=(0.0, 0.0, 1.0),
            parent_xform=wp.transform(wp.vec3(0.5 * self.rod1_rest_length, 0.0, 0.0), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(*self.bell_input_local), wp.quat_identity()),
            label="pushrod_to_bellcrank_input",
        )
        self.j_bell_ground = builder.add_joint_revolute(
            parent=-1,
            child=self.bellcrank,
            axis=(0.0, 0.0, 1.0),
            parent_xform=wp.transform(wp.vec3(*self.bell_pivot), wp.quat_identity()),
            child_xform=wp.transform_identity(),
            label="ground_to_bellcrank",
        )
        self.j_bell_rod_out = builder.add_joint_revolute(
            parent=self.bellcrank,
            child=self.rod2,
            axis=(0.0, 0.0, 1.0),
            parent_xform=wp.transform(wp.vec3(*self.bell_output_local), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(-0.5 * self.rod2_rest_length, 0.0, 0.0), wp.quat_identity()),
            label="bellcrank_to_output_pushrod",
        )
        self.j_rod_output = builder.add_joint_revolute(
            parent=self.rod2,
            child=self.output_rocker,
            axis=(0.0, 0.0, 1.0),
            parent_xform=wp.transform(wp.vec3(0.5 * self.rod2_rest_length, 0.0, 0.0), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(0.5 * self.output_length, 0.0, 0.0), wp.quat_identity()),
            label="output_pushrod_to_rocker",
        )
        self.j_output_ground = builder.add_joint_revolute(
            parent=-1,
            child=self.output_rocker,
            axis=(0.0, 0.0, 1.0),
            parent_xform=wp.transform(wp.vec3(*self.output_pivot), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(-0.5 * self.output_length, 0.0, 0.0), wp.quat_identity()),
            label="ground_to_output_rocker",
        )
        builder.color()

        self.model = builder.finalize()
        self.device = self.model.device
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = None

        rod1_elastic = int(self.model.body_elastic_index.numpy()[self.rod1])
        rod2_elastic = int(self.model.body_elastic_index.numpy()[self.rod2])
        self.rod1_joint = int(self.model.elastic_joint.numpy()[rod1_elastic])
        self.rod2_joint = int(self.model.elastic_joint.numpy()[rod2_elastic])
        self.rod1_q_start = int(self.model.joint_q_start.numpy()[self.rod1_joint])
        self.rod2_q_start = int(self.model.joint_q_start.numpy()[self.rod2_joint])
        self.rod1_qd_start = int(self.model.joint_qd_start.numpy()[self.rod1_joint])
        self.rod2_qd_start = int(self.model.joint_qd_start.numpy()[self.rod2_joint])
        self.drive_qd_start = int(self.model.joint_qd_start.numpy()[self.j_input_ground])
        self._target_pos = self.control.joint_target_pos.numpy()
        self._joint_f = self.control.joint_f.numpy()
        self.max_volume_ratio_error = max(
            abs(elastic_shape_volume_ratio(self.model, self.state_0, 0) - 1.0),
            abs(elastic_shape_volume_ratio(self.model, self.state_0, 1) - 1.0),
        )

        self.solver = newton.solvers.SolverVBD(
            self.model,
            iterations=24,
            rigid_joint_linear_k_start=1.0e5,
            rigid_joint_angular_k_start=1.0e4,
            rigid_joint_linear_ke=2.0e6,
            rigid_joint_angular_ke=5.0e5,
        )

        self.viewer.set_model(self.model)
        self.viewer.set_camera(wp.vec3(0.62, -1.45, 0.82), -27.0, 88.0)

    def _rod_modes(self, rod_index: int) -> np.ndarray:
        q_start = self.rod1_q_start if rod_index == 0 else self.rod2_q_start
        return self.state_0.joint_q.numpy()[q_start + 7 : q_start + 9]

    def _joint_residuals(self) -> list[float]:
        joints = [
            self.j_input_ground,
            self.j_input_rod,
            self.j_rod_bell_in,
            self.j_bell_ground,
            self.j_bell_rod_out,
            self.j_rod_output,
            self.j_output_ground,
        ]
        return [
            float(
                np.linalg.norm(
                    joint_endpoint_world(self.model, self.state_0, joint, "parent")
                    - joint_endpoint_world(self.model, self.state_0, joint, "child")
                )
            )
            for joint in joints
        ]

    def _rod_length_errors(self) -> tuple[float, float]:
        rod1_left = joint_endpoint_world(self.model, self.state_0, self.j_input_rod, "child")
        rod1_right = joint_endpoint_world(self.model, self.state_0, self.j_rod_bell_in, "parent")
        rod2_left = joint_endpoint_world(self.model, self.state_0, self.j_bell_rod_out, "child")
        rod2_right = joint_endpoint_world(self.model, self.state_0, self.j_rod_output, "parent")
        rod1_expected = self.rod1_rest_length + float(self._rod_modes(0)[0])
        rod2_expected = self.rod2_rest_length + float(self._rod_modes(1)[0])
        return (
            abs(float(np.linalg.norm(rod1_right - rod1_left)) - rod1_expected),
            abs(float(np.linalg.norm(rod2_right - rod2_left)) - rod2_expected),
        )

    def _set_controls(self, t: float):
        self._target_pos[self.drive_qd_start] = self.input_theta0 + 6.24 * math.sin(1.15 * t)
        self.control.joint_target_pos.assign(self._target_pos)

        self._joint_f[:] = 0.0
        self.control.joint_f.assign(self._joint_f)

    def _update_metrics(self):
        self.max_joint_residual = max(self.max_joint_residual, *self._joint_residuals())
        self.max_rod_length_error = max(self.max_rod_length_error, *self._rod_length_errors())
        output_angle = _angle_from_quat_z(self.state_0.body_q.numpy()[self.output_rocker, 3:7])
        self.output_min = min(self.output_min, output_angle)
        self.output_max = max(self.output_max, output_angle)
        self.max_volume_ratio_error = max(
            self.max_volume_ratio_error,
            abs(elastic_shape_volume_ratio(self.model, self.state_0, 0) - 1.0),
            abs(elastic_shape_volume_ratio(self.model, self.state_0, 1) - 1.0),
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
        if self.max_joint_residual > 3.0e-2:
            raise AssertionError(f"bell-crank joint residual too large: {self.max_joint_residual}")
        if self.max_rod_length_error > 1.0e-4:
            raise AssertionError(f"pushrod modal length error too large: {self.max_rod_length_error}")
        if self.output_max - self.output_min < 0.08:
            raise AssertionError("output rocker did not move through the elastic pushrod chain")
        if max(np.max(np.abs(self._rod_modes(0))), np.max(np.abs(self._rod_modes(1)))) > 0.4:
            raise AssertionError(
                f"pushrod mode amplitudes are out of range: {self._rod_modes(0)}, {self._rod_modes(1)}"
            )
        if self.max_volume_ratio_error > 1.0:
            raise AssertionError(f"pushrod volume changed by {self.max_volume_ratio_error:.3f}")


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
