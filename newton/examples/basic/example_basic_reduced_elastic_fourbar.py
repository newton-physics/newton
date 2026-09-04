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
# Example Basic Reduced Elastic Fourbar
#
# Demonstrates a closed four-bar linkage with a reduced elastic coupler link.
#
# Command: python -m newton.examples basic_reduced_elastic_fourbar
#
###########################################################################

import math

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.basic._reduced_elastic import beam_render_sample_points, elastic_shape_volume_ratio


def _quat_from_angle_z(theta: float):
    return wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), theta)


def _quat_rotate_np(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    qv = np.array(q[:3], dtype=float)
    return v + 2.0 * np.cross(qv, np.cross(qv, v) + q[3] * v)


def _transform_point_np(xform: np.ndarray, point: np.ndarray) -> np.ndarray:
    return xform[:3] + _quat_rotate_np(xform[3:7], point)


def _solve_fourbar(theta2: float, a: float, b: float, c: float, d: float) -> tuple[float, float]:
    k1 = d / a
    k2 = d / c
    k3 = (a * a - b * b + c * c + d * d) / (2.0 * a * c)
    A = k1 - math.cos(theta2)
    B = -math.sin(theta2)
    C = k2 * math.cos(theta2) - k3
    theta4 = math.atan2(B, A) + math.acos(np.clip(C / math.sqrt(A * A + B * B), -1.0, 1.0))
    theta3 = math.atan2(c * math.sin(theta4) - a * math.sin(theta2), d + c * math.cos(theta4) - a * math.cos(theta2))
    return theta3, theta4


class Example:
    def __init__(self, viewer, args):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 2
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        self.viewer = viewer
        self.args = args

        self.a = 0.35
        self.b_rest = 0.65
        self.c = 0.55
        self.d = 0.75
        self.z = 0.12
        self.coupler_hy = 0.05
        self.coupler_hz = 0.03
        self.theta2_0 = 0.75
        self.mode_q0 = np.array([0.08, 0.08, 0.0], dtype=np.float32)

        theta3, theta4 = _solve_fourbar(self.theta2_0, self.a, self.b_rest + float(self.mode_q0[0]), self.c, self.d)
        A = np.array([0.0, 0.0, self.z])
        D = np.array([self.d, 0.0, self.z])
        e2 = np.array([math.cos(self.theta2_0), math.sin(self.theta2_0), 0.0])
        e3 = np.array([math.cos(theta3), math.sin(theta3), 0.0])
        e4 = np.array([math.cos(theta4), math.sin(theta4), 0.0])
        B = A + self.a * e2

        coupler_basis = newton.ModalGeneratorBeam(
            length=self.b_rest,
            half_width_y=self.coupler_hy,
            half_width_z=self.coupler_hz,
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
            sample_count=33,
            label="coupler_basis",
        ).build(
            sample_points=beam_render_sample_points(
                self.b_rest,
                self.coupler_hy,
                self.coupler_hz,
                extra_points=((-0.5 * self.b_rest, 0.0, 0.0), (0.5 * self.b_rest, 0.0, 0.0)),
            )
        )

        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        builder.add_ground_plane()

        inertia = wp.mat33(0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01)
        self.crank = builder.add_link(
            xform=wp.transform(wp.vec3(*(A + 0.5 * self.a * e2)), _quat_from_angle_z(self.theta2_0)),
            mass=1.0,
            inertia=inertia,
            label="crank",
        )
        self.coupler = builder.add_link_elastic(
            xform=wp.transform(
                wp.vec3(*(B + 0.5 * (self.b_rest + float(self.mode_q0[0])) * e3)),
                _quat_from_angle_z(theta3),
            ),
            mass=1.0,
            inertia=inertia,
            mode_count=3,
            mode_mass=[0.04, 0.035, 0.035],
            mode_stiffness=[10.0, 18.0, 288.0],
            mode_damping=[0.3, 0.3, 0.65],
            mode_q=self.mode_q0,
            modal_basis=coupler_basis,
            label="elastic_coupler",
        )
        self.rocker = builder.add_link(
            xform=wp.transform(wp.vec3(*(D + 0.5 * self.c * e4)), _quat_from_angle_z(theta4)),
            mass=1.0,
            inertia=inertia,
            label="rocker",
        )

        shape_cfg = newton.ModelBuilder.ShapeConfig()
        shape_cfg.density = 0.0
        builder.add_shape_box(self.crank, hx=self.a / 2.0, hy=0.035, hz=0.025, cfg=shape_cfg)
        builder.add_shape_box(self.coupler, hx=self.b_rest / 2.0, hy=self.coupler_hy, hz=self.coupler_hz, cfg=shape_cfg)
        builder.add_shape_box(self.rocker, hx=self.c / 2.0, hy=0.035, hz=0.025, cfg=shape_cfg)

        self.j_drive = builder.add_joint_revolute(
            parent=-1,
            child=self.crank,
            axis=(0.0, 0.0, 1.0),
            parent_xform=wp.transform(wp.vec3(*A), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(-self.a / 2.0, 0.0, 0.0), wp.quat_identity()),
            target_pos=self.theta2_0,
            target_ke=40.0,
            target_kd=0.08,
            label="ground_crank",
        )
        self.j_crank_coupler = builder.add_joint_revolute(
            parent=self.crank,
            child=self.coupler,
            axis=(0.0, 0.0, 1.0),
            parent_xform=wp.transform(wp.vec3(self.a / 2.0, 0.0, 0.0), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(-self.b_rest / 2.0, 0.0, 0.0), wp.quat_identity()),
            label="crank_elastic_coupler",
        )
        self.j_coupler_rocker = builder.add_joint_revolute(
            parent=self.coupler,
            child=self.rocker,
            axis=(0.0, 0.0, 1.0),
            parent_xform=wp.transform(wp.vec3(self.b_rest / 2.0, 0.0, 0.0), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(self.c / 2.0, 0.0, 0.0), wp.quat_identity()),
            label="elastic_coupler_rocker",
        )
        self.j_ground_rocker = builder.add_joint_revolute(
            parent=-1,
            child=self.rocker,
            axis=(0.0, 0.0, 1.0),
            parent_xform=wp.transform(wp.vec3(*D), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(-self.c / 2.0, 0.0, 0.0), wp.quat_identity()),
            label="ground_rocker",
        )

        builder.joint_q[builder.joint_q_start[self.j_drive]] = self.theta2_0
        builder.add_articulation([self.j_drive, self.j_crank_coupler, self.j_coupler_rocker], label="fourbar")
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
        self.drive_qd_start = int(self.model.joint_qd_start.numpy()[self.j_drive])
        self._target_pos = self.control.joint_target_pos.numpy()
        self._joint_f = self.control.joint_f.numpy()
        self.initial_volume_ratio = elastic_shape_volume_ratio(self.model, self.state_0)
        self.max_volume_ratio_error = abs(self.initial_volume_ratio - 1.0)

        self.solver = newton.solvers.SolverVBD(
            self.model,
            iterations=20,
            rigid_joint_linear_k_start=1.0e5,
            rigid_joint_angular_k_start=1.0e3,
            rigid_joint_linear_ke=2.0e6,
            rigid_joint_angular_ke=1.0e5,
        )

        self.viewer.set_model(self.model)
        self.viewer.set_camera(wp.vec3(0.38, -1.45, 0.85), -26.0, 90.0)

    def _mode_shape(self, x: np.ndarray) -> np.ndarray:
        s = float(x[0] + 0.5 * self.b_rest)
        xi = s / self.b_rest
        axial = float(x[0] / self.b_rest)

        first = math.sin(math.pi * xi)
        first_slope = (math.pi / self.b_rest) * math.cos(math.pi * xi)
        second = math.sin(2.0 * math.pi * xi)
        second_slope = (2.0 * math.pi / self.b_rest) * math.cos(2.0 * math.pi * xi)
        return np.array(
            [
                [axial, 0.0, 0.0],
                [-float(x[1]) * first_slope, first, 0.0],
                [-float(x[1]) * second_slope, second, 0.0],
            ],
            dtype=np.float32,
        )

    def _mode_values(self) -> np.ndarray:
        return self.state_0.joint_q.numpy()[self.elastic_q_start + 7 : self.elastic_q_start + 10]

    def _update_volume_metric(self):
        ratio = elastic_shape_volume_ratio(self.model, self.state_0)
        self.max_volume_ratio_error = max(self.max_volume_ratio_error, abs(ratio - 1.0))

    def _endpoint_world(self, joint: int, child_side: bool) -> np.ndarray:
        body_ids = self.model.joint_child.numpy() if child_side else self.model.joint_parent.numpy()
        xforms = self.model.joint_X_c.numpy() if child_side else self.model.joint_X_p.numpy()
        endpoint_ids = (
            self.model.joint_child_elastic_endpoint.numpy()
            if child_side
            else self.model.joint_parent_elastic_endpoint.numpy()
        )

        body = int(body_ids[joint])
        local = np.array(xforms[joint, :3], dtype=float)
        if int(endpoint_ids[joint]) >= 0:
            mode_shape = self._mode_shape(local)
            mode_values = self._mode_values()
            for mode in range(len(mode_values)):
                local += mode_shape[mode] * mode_values[mode]

        if body < 0:
            return local
        return _transform_point_np(self.state_0.body_q.numpy()[body], local)

    def _joint_residuals(self) -> list[float]:
        joints = [self.j_drive, self.j_crank_coupler, self.j_coupler_rocker, self.j_ground_rocker]
        return [
            float(np.linalg.norm(self._endpoint_world(joint, False) - self._endpoint_world(joint, True)))
            for joint in joints
        ]

    def _set_controls(self, t: float):
        self._target_pos[self.drive_qd_start] = self.theta2_0 + 0.35 * math.sin(1.5 * t)
        self.control.joint_target_pos.assign(self._target_pos)

        self._joint_f[:] = 0.0
        self._joint_f[self.elastic_qd_start + 6] = 1.1 * math.sin(4.2 * t)
        self._joint_f[self.elastic_qd_start + 7] = 1.4 * math.sin(3.0 * t)
        self._joint_f[self.elastic_qd_start + 8] = 7.0 * math.cos(2.4 * t)
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
        self._update_volume_metric()

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        if not np.isfinite(self.state_0.body_q.numpy()).all():
            raise AssertionError("body transforms contain non-finite values")
        if not np.isfinite(self.state_0.joint_q.numpy()).all():
            raise AssertionError("joint coordinates contain non-finite values")

        residual = max(self._joint_residuals())
        if residual > 2.0e-2:
            raise AssertionError(f"four-bar joint residual too large: {residual}")

        if np.max(np.abs(self._mode_values())) > 0.5:
            raise AssertionError(f"elastic mode amplitude is out of expected range: {self._mode_values()}")
        if self.max_volume_ratio_error > 0.25:
            raise AssertionError(f"elastic coupler volume changed by {self.max_volume_ratio_error:.3f}")


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
