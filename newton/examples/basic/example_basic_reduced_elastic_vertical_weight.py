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
# Example Basic Reduced Elastic Vertical Weight
#
# Demonstrates a vertical elastic bar stretching under a rigid suspended mass.
#
# Command: python -m newton.examples basic_reduced_elastic_vertical_weight
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


def _top_fixed_poisson_stretch_mode(points: np.ndarray, length: float, poisson_ratio: float) -> np.ndarray:
    """Return axial stretch with zero displacement at the top endpoint."""
    points = np.asarray(points, dtype=np.float32)
    phi = np.zeros_like(points, dtype=np.float32)
    axial = np.clip((points[:, 0] + 0.5 * length) / length, 0.0, 1.0)
    phi[:, 0] = axial
    phi[:, 1] = -float(poisson_ratio) * points[:, 1] / float(length)
    phi[:, 2] = -float(poisson_ratio) * points[:, 2] / float(length)
    return phi


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

        self.length = 0.72
        self.hy = 0.035
        self.hz = 0.035
        self.top_z = 1.16
        self.gravity = 9.81
        self.weight_mass = 45.0
        self.expected_extension = 0.095
        self.load = self.weight_mass * self.gravity
        self.mode_stiffness = self.load / self.expected_extension
        self.mode_mass = 0.004
        self.mode_damping = 0.001
        self.max_joint_residual = 0.0
        self.max_volume_ratio_error = 0.0

        sample_points = beam_render_sample_points(
            self.length,
            self.hy,
            self.hz,
            extra_points=((-0.5 * self.length, 0.0, 0.0), (0.5 * self.length, 0.0, 0.0)),
        )
        phi = _top_fixed_poisson_stretch_mode(sample_points, self.length, 0.45)
        basis = newton.ModalBasis(
            sample_points=sample_points,
            sample_phi=phi.reshape((-1, 1, 3)),
            mode_mass=[self.mode_mass],
            mode_stiffness=[self.mode_stiffness],
            mode_damping=[self.mode_damping],
            label="gravity_vertical_axial_basis",
        )

        q_vertical = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), 0.5 * math.pi)
        center_z = self.top_z - 0.5 * self.length
        bottom_z = self.top_z - self.length

        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -self.gravity))
        builder.add_ground_plane()

        inertia = wp.mat33(0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01)
        self.top_fixture = builder.add_body(
            xform=wp.transform(wp.vec3(0.0, 0.0, self.top_z + 0.045), wp.quat_identity()),
            mass=1.0,
            inertia=inertia,
            label="fixed_top_fixture",
        )
        self.bar = builder.add_body_elastic(
            xform=wp.transform(wp.vec3(0.0, 0.0, center_z), q_vertical),
            mass=0.2,
            inertia=inertia,
            mode_q=[0.0],
            modal_basis=basis,
            label="vertical_elastic_bar",
        )
        self.weight = builder.add_body(
            xform=wp.transform(wp.vec3(0.0, 0.0, bottom_z), wp.quat_identity()),
            mass=self.weight_mass,
            inertia=inertia,
            label="suspended_weight",
        )

        shape_cfg = newton.ModelBuilder.ShapeConfig()
        shape_cfg.density = 0.0
        shape_cfg.has_shape_collision = False
        shape_cfg.has_particle_collision = False
        builder.add_shape_box(self.top_fixture, hx=0.09, hy=0.09, hz=0.045, cfg=shape_cfg)
        builder.add_shape_box(self.bar, hx=0.5 * self.length, hy=self.hy, hz=self.hz, cfg=shape_cfg)
        builder.add_shape_box(self.weight, hx=0.07, hy=0.07, hz=0.07, cfg=shape_cfg)

        self.j_fixture = builder.add_joint_fixed(
            parent=-1,
            child=self.top_fixture,
            parent_xform=wp.transform(wp.vec3(0.0, 0.0, self.top_z + 0.045), wp.quat_identity()),
            child_xform=wp.transform_identity(),
            label="ground_to_top_fixture",
        )
        self.j_top = builder.add_joint_fixed(
            parent=self.top_fixture,
            child=self.bar,
            parent_xform=wp.transform(wp.vec3(0.0, 0.0, -0.045), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(-0.5 * self.length, 0.0, 0.0), wp.quat_identity()),
            label="top_fixture_to_vertical_bar",
        )
        self.j_bottom = builder.add_joint_fixed(
            parent=self.bar,
            child=self.weight,
            parent_xform=wp.transform(wp.vec3(0.5 * self.length, 0.0, 0.0), wp.quat_identity()),
            child_xform=wp.transform_identity(),
            label="vertical_bar_to_weight",
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
        self._joint_f = self.control.joint_f.numpy()
        self.max_volume_ratio_error = abs(elastic_shape_volume_ratio(self.model, self.state_0) - 1.0)

        self.solver = newton.solvers.SolverVBD(
            self.model,
            iterations=24,
            rigid_joint_linear_k_start=1.0e5,
            rigid_joint_angular_k_start=1.0e5,
            rigid_joint_linear_ke=2.0e6,
            rigid_joint_angular_ke=2.0e6,
        )

        self.viewer.set_model(self.model)
        bounds_min = np.array([-0.12, -0.12, bottom_z - 0.08])
        bounds_max = np.array([0.12, 0.12, self.top_z + 0.12])
        _set_camera_from_bounds(self.viewer, bounds_min, bounds_max, np.array([0.28, -1.0, 0.2]))

    def _mode_value(self) -> float:
        return float(self.state_0.joint_q.numpy()[self.elastic_q_start + 7])

    def _stretch_value(self) -> float:
        top = joint_endpoint_world(self.model, self.state_0, self.j_top, "child")
        bottom = joint_endpoint_world(self.model, self.state_0, self.j_bottom, "parent")
        return float(np.linalg.norm(bottom - top) - self.length)

    def _set_controls(self):
        self._joint_f[:] = 0.0
        self.control.joint_f.assign(self._joint_f)

    def _update_metrics(self):
        top_residual = np.linalg.norm(
            joint_endpoint_world(self.model, self.state_0, self.j_top, "parent")
            - joint_endpoint_world(self.model, self.state_0, self.j_top, "child")
        )
        bottom_residual = np.linalg.norm(
            joint_endpoint_world(self.model, self.state_0, self.j_bottom, "parent")
            - joint_endpoint_world(self.model, self.state_0, self.j_bottom, "child")
        )
        self.max_joint_residual = max(self.max_joint_residual, float(max(top_residual, bottom_residual)))
        self.max_volume_ratio_error = max(
            self.max_volume_ratio_error, abs(elastic_shape_volume_ratio(self.model, self.state_0) - 1.0)
        )

    def simulate(self):
        for _substep in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self._set_controls()
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
        if not np.isfinite(self.state_0.joint_q.numpy()).all():
            raise AssertionError("joint coordinates contain non-finite values")
        stretch = self._stretch_value()
        if stretch <= 0.0:
            raise AssertionError(f"vertical bar is compressed by {-stretch}")
        if abs(stretch - self.expected_extension) > 1.0e-2:
            raise AssertionError(f"vertical bar stretch {stretch} differs from {self.expected_extension}")
        if self.max_joint_residual > 2.0e-2:
            raise AssertionError(f"vertical weight joint residual too large: {self.max_joint_residual}")
        if self.max_volume_ratio_error > 0.08:
            raise AssertionError(f"vertical bar volume changed by {self.max_volume_ratio_error:.3f}")


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
