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
# Example Basic Reduced Elastic Prismatic
#
# Demonstrates a prismatic actuator compressing and stretching a reduced
# elastic body. The elastic mode is an axial strain field with Poisson lateral
# strain, so the bar bulges under compression and necks under tension.
#
# Command: python -m newton.examples basic_reduced_elastic_prismatic
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
    poisson_axial_mode,
)


def _quat_angle_error(q_a: np.ndarray, q_b: np.ndarray) -> float:
    dot = abs(float(np.dot(q_a, q_b)))
    dot = min(max(dot, -1.0), 1.0)
    return 2.0 * math.acos(dot)


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

        self.length = 0.75
        self.hy = 0.07
        self.hz = 0.05
        self.z = 0.18
        self.poisson_ratio = 0.49
        self.slide_amplitude = 0.452
        self.slide_frequency = 0.45
        self.rotation_amplitude = 0.35
        self.rotation_frequency = 0.30
        self.mode_target_ke = 4.0e4
        self.mode_target_kd = 140.0
        self._last_slide_target = 0.0
        self._last_rotation_target = 0.0
        self.min_mode_q = 0.0
        self.max_mode_q = 0.0

        sample_points = beam_render_sample_points(
            self.length,
            self.hy,
            self.hz,
            extra_points=((-0.5 * self.length, 0.0, 0.0), (0.5 * self.length, 0.0, 0.0)),
        )
        axial_phi = poisson_axial_mode(sample_points, self.length, self.poisson_ratio, clamped_ends=True)
        axial_basis = newton.ModalBasis(
            sample_points=sample_points,
            sample_phi=axial_phi.reshape((-1, 1, 3)),
            mode_mass=[0.12],
            mode_stiffness=[18.0],
            mode_damping=[0.12],
            label="poisson_axial_basis",
        )

        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        builder.add_ground_plane()

        inertia = wp.mat33(0.02, 0.0, 0.0, 0.0, 0.02, 0.0, 0.0, 0.0, 0.02)
        self.base = builder.add_body(
            xform=wp.transform(wp.vec3(0.0, 0.0, self.z), wp.quat_identity()),
            mass=1.0,
            inertia=inertia,
            label="rotating_prismatic_base",
        )
        self.driver = builder.add_body(
            xform=wp.transform(wp.vec3(self.length, 0.0, self.z), wp.quat_identity()),
            mass=0.8,
            inertia=inertia,
            label="prismatic_driver",
        )
        self.elastic = builder.add_body_elastic(
            xform=wp.transform(wp.vec3(0.5 * self.length, 0.0, self.z), wp.quat_identity()),
            mass=1.0,
            inertia=inertia,
            mode_q=[0.0],
            modal_basis=axial_basis,
            label="poisson_elastic_bar",
        )

        shape_cfg = newton.ModelBuilder.ShapeConfig()
        shape_cfg.density = 0.0
        shape_cfg.has_shape_collision = False
        shape_cfg.has_particle_collision = False
        builder.add_shape_box(self.base, hx=0.035, hy=0.11, hz=0.08, cfg=shape_cfg)
        builder.add_shape_box(self.driver, hx=0.035, hy=0.12, hz=0.09, cfg=shape_cfg)
        builder.add_shape_box(self.elastic, hx=0.5 * self.length, hy=self.hy, hz=self.hz, cfg=shape_cfg)

        self.j_base = builder.add_joint_revolute(
            parent=-1,
            child=self.base,
            axis=(0.0, 0.0, 1.0),
            parent_xform=wp.transform(wp.vec3(0.0, 0.0, self.z), wp.quat_identity()),
            child_xform=wp.transform_identity(),
            target_pos=0.0,
            target_ke=100.0,
            target_kd=1.5,
            label="world_to_rotating_base",
        )
        self.j_slide = builder.add_joint_prismatic(
            parent=self.base,
            child=self.driver,
            axis=(1.0, 0.0, 0.0),
            parent_xform=wp.transform(wp.vec3(self.length, 0.0, 0.0), wp.quat_identity()),
            child_xform=wp.transform_identity(),
            target_pos=0.0,
            target_ke=600.0,
            target_kd=4.0,
            label="base_to_driver_prismatic",
        )
        self.j_left = builder.add_joint_fixed(
            parent=self.base,
            child=self.elastic,
            parent_xform=wp.transform_identity(),
            child_xform=wp.transform(wp.vec3(-0.5 * self.length, 0.0, 0.0), wp.quat_identity()),
            label="base_to_elastic_left",
        )
        self.j_right = builder.add_joint_fixed(
            parent=self.driver,
            child=self.elastic,
            parent_xform=wp.transform_identity(),
            child_xform=wp.transform(wp.vec3(0.5 * self.length, 0.0, 0.0), wp.quat_identity()),
            label="driver_to_elastic_right",
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
        self.base_qd_start = int(self.model.joint_qd_start.numpy()[self.j_base])
        self.slide_qd_start = int(self.model.joint_qd_start.numpy()[self.j_slide])
        self._target_pos = self.control.joint_target_pos.numpy()
        self._joint_f = self.control.joint_f.numpy()
        self.initial_volume_ratio = elastic_shape_volume_ratio(self.model, self.state_0)
        self.max_volume_ratio_error = abs(self.initial_volume_ratio - 1.0)

        self.solver = newton.solvers.SolverVBD(
            self.model,
            iterations=18,
            rigid_joint_linear_k_start=1.0e5,
            rigid_joint_angular_k_start=1.0e5,
            rigid_joint_linear_ke=2.0e6,
            rigid_joint_angular_ke=2.0e6,
        )

        self.viewer.set_model(self.model)
        self.viewer.set_camera(wp.vec3(0.34, -1.25, 0.68), -25.0, 78.0)

    def _mode_value(self) -> float:
        return float(self.state_0.joint_q.numpy()[self.elastic_q_start + 7])

    def _rotation_error(self) -> float:
        body_q = self.state_0.body_q.numpy()
        return _quat_angle_error(body_q[self.base, 3:7], body_q[self.elastic, 3:7])

    def _update_volume_metric(self):
        ratio = elastic_shape_volume_ratio(self.model, self.state_0)
        self.max_volume_ratio_error = max(self.max_volume_ratio_error, abs(ratio - 1.0))

    def _set_controls(self, t: float):
        slide_phase = 2.0 * math.pi * self.slide_frequency * t
        rotate_phase = 2.0 * math.pi * self.rotation_frequency * t
        self._last_slide_target = self.slide_amplitude * math.sin(slide_phase)
        self._last_rotation_target = self.rotation_amplitude * math.sin(rotate_phase)
        self._target_pos[self.slide_qd_start] = self._last_slide_target
        self._target_pos[self.base_qd_start] = self._last_rotation_target
        self.control.joint_target_pos.assign(self._target_pos)

        joint_q = self.state_0.joint_q.numpy()
        joint_qd = self.state_0.joint_qd.numpy()
        mode_q = float(joint_q[self.elastic_q_start + 7])
        mode_qd = float(joint_qd[self.elastic_qd_start + 6])
        self._joint_f[:] = 0.0
        self._joint_f[self.elastic_qd_start + 6] = (
            self.mode_target_ke * (self._last_slide_target - mode_q) - self.mode_target_kd * mode_qd
        )
        self.control.joint_f.assign(self._joint_f)

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
        q = self._mode_value()
        self.min_mode_q = min(self.min_mode_q, q)
        self.max_mode_q = max(self.max_mode_q, q)
        self._update_volume_metric()

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        if not np.isfinite(self.state_0.joint_q.numpy()).all():
            raise AssertionError("joint coordinates contain non-finite values")
        if self.max_mode_q - self.min_mode_q < 0.25:
            raise AssertionError("prismatic drive did not stretch/compress the elastic mode")
        if abs(self._mode_value() - self._last_slide_target) > 0.025:
            raise AssertionError(
                f"elastic axial mode {self._mode_value()} differs from prismatic target {self._last_slide_target}"
            )
        if abs(self._last_rotation_target) > 0.1 and self._rotation_error() > 0.04:
            raise AssertionError(f"elastic child did not rotate with parent: angle error {self._rotation_error()}")
        if self.max_volume_ratio_error > 0.25:
            raise AssertionError(f"Poisson axial mode changed volume by {self.max_volume_ratio_error:.3f}")


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
