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
# Example Basic Reduced Elastic Crank Slider
#
# Demonstrates a textbook slider-crank with a reduced elastic connecting rod.
#
# Command: python -m newton.examples basic_reduced_elastic_crank_slider
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


def _solve_slider_x(theta: float, crank_length: float, rod_length: float) -> float:
    y = crank_length * math.sin(theta)
    return crank_length * math.cos(theta) + math.sqrt(max(rod_length * rod_length - y * y, 0.0))


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

        self.crank_length = 0.24
        self.rod_rest_length = 0.64
        self.rod_hy = 0.04
        self.rod_hz = 0.032
        self.z = 0.16
        self.theta0 = 0.72
        self.initial_axial_q = 0.045
        self.max_slider_formula_error = 0.0
        self.max_joint_residual = 0.0
        self.max_volume_ratio_error = 0.0
        self.max_mode_abs = 0.0

        rod_length = self.rod_rest_length + self.initial_axial_q
        slider_x = _solve_slider_x(self.theta0, self.crank_length, rod_length)
        crank_axis = np.array([math.cos(self.theta0), math.sin(self.theta0), 0.0], dtype=float)
        crank_pin = np.array([self.crank_length * crank_axis[0], self.crank_length * crank_axis[1], self.z])
        slider_pin = np.array([slider_x, 0.0, self.z])
        rod_axis = slider_pin - crank_pin
        rod_theta = math.atan2(float(rod_axis[1]), float(rod_axis[0]))
        rod_center = 0.5 * (crank_pin + slider_pin)

        sample_points = beam_render_sample_points(
            self.rod_rest_length,
            self.rod_hy,
            self.rod_hz,
            extra_points=((-0.5 * self.rod_rest_length, 0.0, 0.0), (0.5 * self.rod_rest_length, 0.0, 0.0)),
        )
        rod_basis = newton.ModalGeneratorBeam(
            length=self.rod_rest_length,
            half_width_y=self.rod_hy,
            half_width_z=self.rod_hz,
            mode_specs=[
                {"type": newton.ModalGeneratorBeam.Mode.AXIAL},
                {
                    "type": newton.ModalGeneratorBeam.Mode.BENDING_Y,
                    "boundary": newton.ModalGeneratorBeam.Boundary.PINNED_PINNED,
                    "order": 1,
                },
                {
                    "type": newton.ModalGeneratorBeam.Mode.BENDING_Y,
                    "boundary": newton.ModalGeneratorBeam.Boundary.PINNED_PINNED,
                    "order": 2,
                },
            ],
            density=300.0,
            young_modulus=2.5e6,
            damping_ratio=0.035,
            label="slider_crank_rod_basis",
        ).build(sample_points=sample_points)

        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        builder.add_ground_plane()

        inertia = wp.mat33(0.015, 0.0, 0.0, 0.0, 0.015, 0.0, 0.0, 0.0, 0.015)
        self.crank = builder.add_body(
            xform=wp.transform(
                wp.vec3(*(np.array([0.0, 0.0, self.z]) + 0.5 * self.crank_length * crank_axis)),
                _quat_from_angle_z(self.theta0),
            ),
            mass=1.0,
            inertia=inertia,
            label="rigid_crank",
        )
        self.rod = builder.add_body_elastic(
            xform=wp.transform(wp.vec3(*rod_center), _quat_from_angle_z(rod_theta)),
            mass=0.6,
            inertia=inertia,
            mode_q=[self.initial_axial_q, 0.025, -0.015],
            modal_basis=rod_basis,
            label="elastic_connecting_rod",
        )
        self.slider = builder.add_body(
            xform=wp.transform(wp.vec3(*slider_pin), wp.quat_identity()),
            mass=1.2,
            inertia=inertia,
            label="rigid_slider",
        )

        shape_cfg = newton.ModelBuilder.ShapeConfig()
        shape_cfg.density = 0.0
        shape_cfg.has_shape_collision = False
        shape_cfg.has_particle_collision = False
        builder.add_shape_box(self.crank, hx=0.5 * self.crank_length, hy=0.025, hz=0.025, cfg=shape_cfg)
        builder.add_shape_box(self.rod, hx=0.5 * self.rod_rest_length, hy=self.rod_hy, hz=self.rod_hz, cfg=shape_cfg)
        builder.add_shape_box(self.slider, hx=0.055, hy=0.09, hz=0.055, cfg=shape_cfg)

        self.j_crank_ground = builder.add_joint_revolute(
            parent=-1,
            child=self.crank,
            axis=(0.0, 0.0, 1.0),
            parent_xform=wp.transform(wp.vec3(0.0, 0.0, self.z), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(-0.5 * self.crank_length, 0.0, 0.0), wp.quat_identity()),
            target_pos=self.theta0,
            target_ke=120.0,
            target_kd=0.6,
            label="ground_to_crank",
        )
        self.j_crank_rod = builder.add_joint_revolute(
            parent=self.crank,
            child=self.rod,
            axis=(0.0, 0.0, 1.0),
            parent_xform=wp.transform(wp.vec3(0.5 * self.crank_length, 0.0, 0.0), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(-0.5 * self.rod_rest_length, 0.0, 0.0), wp.quat_identity()),
            label="crank_pin_to_elastic_rod",
        )
        self.j_rod_slider = builder.add_joint_revolute(
            parent=self.rod,
            child=self.slider,
            axis=(0.0, 0.0, 1.0),
            parent_xform=wp.transform(wp.vec3(0.5 * self.rod_rest_length, 0.0, 0.0), wp.quat_identity()),
            child_xform=wp.transform_identity(),
            label="elastic_rod_to_slider_pin",
        )
        self.j_slider_ground = builder.add_joint_prismatic(
            parent=-1,
            child=self.slider,
            axis=(1.0, 0.0, 0.0),
            parent_xform=wp.transform(wp.vec3(0.0, 0.0, self.z), wp.quat_identity()),
            child_xform=wp.transform_identity(),
            label="slider_rail",
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
        self.drive_qd_start = int(self.model.joint_qd_start.numpy()[self.j_crank_ground])
        self._target_pos = self.control.joint_target_pos.numpy()
        self._joint_f = self.control.joint_f.numpy()
        self.max_volume_ratio_error = abs(elastic_shape_volume_ratio(self.model, self.state_0) - 1.0)

        self.solver = newton.solvers.SolverVBD(
            self.model,
            iterations=20,
            rigid_joint_linear_k_start=1.0e5,
            rigid_joint_angular_k_start=1.0e4,
            rigid_joint_linear_ke=2.0e6,
            rigid_joint_angular_ke=5.0e5,
        )

        self.viewer.set_model(self.model)
        self.viewer.set_camera(wp.vec3(0.55, -1.25, 0.72), -24.0, 86.0)

    def _mode_values(self) -> np.ndarray:
        return self.state_0.joint_q.numpy()[self.elastic_q_start + 7 : self.elastic_q_start + 10]

    def _slider_formula_error(self) -> float:
        body_q = self.state_0.body_q.numpy()
        theta = _angle_from_quat_z(body_q[self.crank, 3:7])
        rod_length = self.rod_rest_length + float(self._mode_values()[0])
        expected_x = _solve_slider_x(theta, self.crank_length, rod_length)
        return abs(float(body_q[self.slider, 0]) - expected_x)

    def _joint_residuals(self) -> list[float]:
        joints = [self.j_crank_ground, self.j_crank_rod, self.j_rod_slider]
        return [
            float(
                np.linalg.norm(
                    joint_endpoint_world(self.model, self.state_0, joint, "parent")
                    - joint_endpoint_world(self.model, self.state_0, joint, "child")
                )
            )
            for joint in joints
        ]

    def _set_controls(self, t: float):
        target_theta = self.theta0 + 0.55 * math.sin(1.25 * t)
        self._target_pos[self.drive_qd_start] = target_theta
        self.control.joint_target_pos.assign(self._target_pos)

        joint_q = self.state_0.joint_q.numpy()
        joint_qd = self.state_0.joint_qd.numpy()
        axial_q = float(joint_q[self.elastic_q_start + 7])
        axial_qd = float(joint_qd[self.elastic_qd_start + 6])
        axial_target = 0.04 + 0.035 * math.sin(2.1 * t + 0.4)
        self._joint_f[:] = 0.0
        self._joint_f[self.elastic_qd_start + 6] = 260.0 * (axial_target - axial_q) - 6.0 * axial_qd
        self._joint_f[self.elastic_qd_start + 7] = 0.9 * math.sin(4.4 * t)
        self._joint_f[self.elastic_qd_start + 8] = 2.2 * math.cos(3.0 * t)
        self.control.joint_f.assign(self._joint_f)

    def _update_metrics(self):
        self.max_slider_formula_error = max(self.max_slider_formula_error, self._slider_formula_error())
        self.max_joint_residual = max(self.max_joint_residual, *self._joint_residuals())
        self.max_mode_abs = max(self.max_mode_abs, float(np.max(np.abs(self._mode_values()))))
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
        if self.max_slider_formula_error > 3.5e-2:
            raise AssertionError(f"slider-crank closure error too large: {self.max_slider_formula_error}")
        if self.max_joint_residual > 2.5e-2:
            raise AssertionError(f"joint residual too large: {self.max_joint_residual}")
        if self.max_mode_abs > 0.35:
            raise AssertionError(f"elastic rod mode amplitude too large: {self.max_mode_abs}")
        if self.max_volume_ratio_error > 0.2:
            raise AssertionError(f"elastic rod volume changed by {self.max_volume_ratio_error:.3f}")


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
