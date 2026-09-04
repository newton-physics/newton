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
# Example Basic Reduced Elastic Cantilever Weight
#
# Demonstrates a cantilever beam carrying a rigid tip weight under gravity.
#
# Command: python -m newton.examples basic_reduced_elastic_cantilever_weight
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


def _cantilever_tip_mode(points: np.ndarray, length: float) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    s = np.clip(points[:, 0] + 0.5 * length, 0.0, length)
    phi = (s * s * (3.0 * length - s)) / (2.0 * length**3)
    slope = (3.0 * s * (2.0 * length - s)) / (2.0 * length**3)
    out = np.zeros_like(points, dtype=np.float32)
    out[:, 0] = -points[:, 2] * slope
    out[:, 2] = phi
    return out


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
    def __init__(self, viewer, args):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 3
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.step_count = 0

        self.viewer = viewer
        self.args = args

        self.length = 0.82
        self.hy = 0.045
        self.hz = 0.035
        self.z = 0.58
        self.gravity = 9.81
        self.weight_mass = 48.6
        self.expected_tip_deflection = 0.115
        self.tip_load = self.weight_mass * self.gravity
        self.mode_stiffness = self.tip_load / self.expected_tip_deflection
        self.expected_q = -self.expected_tip_deflection
        self.release_damping_start = 1.0
        self.release_damping = 0.0
        self.min_mode_q = 0.0
        self.max_joint_residual = 0.0
        self.max_volume_ratio_error = 0.0

        sample_points = beam_render_sample_points(
            self.length,
            self.hy,
            self.hz,
            extra_points=((-0.5 * self.length, 0.0, 0.0), (0.5 * self.length, 0.0, 0.0)),
        )
        phi = _cantilever_tip_mode(sample_points, self.length)
        basis = newton.ModalBasis(
            sample_points=sample_points,
            sample_phi=phi.reshape((-1, 1, 3)),
            mode_mass=[0.0],
            mode_stiffness=[self.mode_stiffness],
            mode_damping=[0.0],
            label="gravity_cantilever_tip_basis",
        )

        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -self.gravity))
        builder.add_ground_plane()

        inertia = wp.mat33(0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01)
        self.beam = builder.add_body_elastic(
            xform=wp.transform(wp.vec3(0.0, 0.0, self.z), wp.quat_identity()),
            mass=0.2,
            inertia=inertia,
            mode_q=[0.0],
            mode_mass=[0.08],
            modal_basis=basis,
            label="gravity_cantilever_beam",
        )
        self.weight = builder.add_body(
            xform=wp.transform(wp.vec3(0.5 * self.length, 0.0, self.z), wp.quat_identity()),
            mass=self.weight_mass,
            inertia=inertia,
            label="rigid_tip_weight",
        )

        shape_cfg = newton.ModelBuilder.ShapeConfig()
        shape_cfg.density = 0.0
        shape_cfg.has_shape_collision = False
        shape_cfg.has_particle_collision = False
        builder.add_shape_box(self.beam, hx=0.5 * self.length, hy=self.hy, hz=self.hz, cfg=shape_cfg)
        builder.add_shape_box(self.weight, hx=0.065, hy=0.065, hz=0.065, cfg=shape_cfg)

        self.j_root = builder.add_joint_fixed(
            parent=-1,
            child=self.beam,
            parent_xform=wp.transform(wp.vec3(-0.5 * self.length, 0.0, self.z), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(-0.5 * self.length, 0.0, 0.0), wp.quat_identity()),
            label="ground_to_cantilever_root",
        )
        self.j_tip = builder.add_joint_fixed(
            parent=self.beam,
            child=self.weight,
            parent_xform=wp.transform(wp.vec3(0.5 * self.length, 0.0, 0.0), wp.quat_identity()),
            child_xform=wp.transform_identity(),
            label="cantilever_tip_to_weight",
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
        bounds_min = np.array([-0.5 * self.length - 0.06, -0.08, self.z + 1.4 * self.expected_q - 0.08])
        bounds_max = np.array([0.5 * self.length + 0.1, 0.08, self.z + 0.1])
        _set_camera_from_bounds(self.viewer, bounds_min, bounds_max, np.array([-0.35, -1.0, 0.35]))

    def _mode_value(self) -> float:
        return float(self.state_0.joint_q.numpy()[self.elastic_q_start + 7])

    def _set_controls(self):
        self._joint_f[:] = 0.0
        if self.sim_time >= self.release_damping_start:
            mode_qd = float(self.state_0.joint_qd.numpy()[self.elastic_qd_start + 6])
            self._joint_f[self.elastic_qd_start + 6] = -self.release_damping * mode_qd
        self.control.joint_f.assign(self._joint_f)

    def _update_metrics(self):
        tip_residual = np.linalg.norm(
            joint_endpoint_world(self.model, self.state_0, self.j_tip, "parent")
            - joint_endpoint_world(self.model, self.state_0, self.j_tip, "child")
        )
        root_residual = np.linalg.norm(
            joint_endpoint_world(self.model, self.state_0, self.j_root, "parent")
            - joint_endpoint_world(self.model, self.state_0, self.j_root, "child")
        )
        self.max_joint_residual = max(self.max_joint_residual, float(max(tip_residual, root_residual)))
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
        self.min_mode_q = min(self.min_mode_q, self._mode_value())
        self._update_metrics()

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        if not np.isfinite(self.state_0.joint_q.numpy()).all():
            raise AssertionError("joint coordinates contain non-finite values")
        if self.min_mode_q > 0.5 * self.expected_q:
            raise AssertionError("cantilever weight did not visibly drop from the horizontal release")
        if abs(self._mode_value() - self.expected_q) > 1.2e-2:
            raise AssertionError(f"cantilever tip deflection {self._mode_value()} differs from {self.expected_q}")
        if self.max_joint_residual > 1.0e-2:
            raise AssertionError(f"cantilever weight joint residual too large: {self.max_joint_residual}")
        if self.max_volume_ratio_error > 0.2:
            raise AssertionError(f"cantilever beam volume changed by {self.max_volume_ratio_error:.3f}")


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
