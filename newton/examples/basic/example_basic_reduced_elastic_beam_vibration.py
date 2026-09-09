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
# Example Basic Reduced Elastic Beam Vibration
#
# Demonstrates a reduced elastic cantilever beam with finite modal mass. The
# beam is released from an initial deflection and vibrates under its modal
# stiffness and damping.
#
# Command: python -m newton.examples basic_reduced_elastic_beam_vibration
#
###########################################################################

import math

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.basic._reduced_elastic import beam_render_sample_points, elastic_shape_volume_ratio


class Example:
    def __init__(self, viewer, args):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 2
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.step_count = 0

        self.viewer = viewer
        self.args = args

        self.length = 0.9
        self.sim_substeps = 8
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.hy = 0.045
        self.hz = 0.035
        self.z = 0.24
        self.mode_q0 = 0.09
        self.mode_qd0 = 0.0

        beam_basis = newton.ModalGeneratorBeam(
            length=self.length,
            half_width_y=self.hy,
            half_width_z=self.hz,
            mode_specs=[
                {
                    "type": newton.ModalGeneratorBeam.Mode.BENDING_Z,
                    "boundary": newton.ModalGeneratorBeam.Boundary.CANTILEVER_TIP,
                }
            ],
            sample_count=41,
            density=250.0,
            young_modulus=3.2e7,
            damping_ratio=0.001,
            label="cantilever_vibration_basis",
        ).build(
            sample_points=beam_render_sample_points(
                self.length,
                self.hy,
                self.hz,
                extra_points=((-0.5 * self.length, 0.0, 0.0), (0.5 * self.length, 0.0, 0.0)),
            )
        )
        self.mode_mass = float(beam_basis.mode_mass[0])
        self.mode_stiffness = float(beam_basis.mode_stiffness[0])
        self.mode_damping = float(beam_basis.mode_damping[0])
        self.natural_frequency = math.sqrt(self.mode_stiffness / self.mode_mass)

        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        builder.add_ground_plane()

        inertia = wp.mat33(0.015, 0.0, 0.0, 0.0, 0.015, 0.0, 0.0, 0.0, 0.015)
        self.beam = builder.add_body_elastic(
            xform=wp.transform(wp.vec3(0.5 * self.length, 0.0, self.z), wp.quat_identity()),
            mass=1.0,
            inertia=inertia,
            mode_q=[self.mode_q0],
            mode_qd=[self.mode_qd0],
            modal_basis=beam_basis,
            label="elastic_cantilever_vibration",
        )

        shape_cfg = newton.ModelBuilder.ShapeConfig()
        shape_cfg.density = 0.0
        builder.add_shape_box(self.beam, hx=0.5 * self.length, hy=self.hy, hz=self.hz, cfg=shape_cfg)
        self.fixed_joint = builder.add_joint_fixed(
            parent=-1,
            child=self.beam,
            parent_xform=wp.transform(wp.vec3(0.0, 0.0, self.z), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(-0.5 * self.length, 0.0, 0.0), wp.quat_identity()),
            label="world_cantilever_vibration_root",
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
        self._expected_q = self.mode_q0
        self._expected_qd = self.mode_qd0
        self.initial_volume_ratio = elastic_shape_volume_ratio(self.model, self.state_0)
        self.max_volume_ratio_error = abs(self.initial_volume_ratio - 1.0)

        self.solver = newton.solvers.SolverVBD(
            self.model,
            iterations=8,
            rigid_joint_linear_k_start=1.0e5,
            rigid_joint_angular_k_start=1.0e5,
            rigid_joint_linear_ke=1.0e6,
            rigid_joint_angular_ke=1.0e6,
        )

        self.viewer.set_model(self.model)
        self.viewer.set_camera(wp.vec3(0.45, -1.25, 0.62), -22.0, 90.0)

    def _mode_value(self) -> float:
        return float(self.state_0.joint_q.numpy()[self.elastic_q_start + 7])

    def _mode_velocity(self) -> float:
        return float(self.state_0.joint_qd.numpy()[self.elastic_qd_start + 6])

    def _update_volume_metric(self):
        ratio = elastic_shape_volume_ratio(self.model, self.state_0)
        self.max_volume_ratio_error = max(self.max_volume_ratio_error, abs(ratio - 1.0))

    def _advance_expected(self):
        denom = self.mode_mass + self.sim_dt * self.mode_damping + self.sim_dt * self.sim_dt * self.mode_stiffness
        self._expected_qd = (
            self.mode_mass * self._expected_qd - self.sim_dt * self.mode_stiffness * self._expected_q
        ) / denom
        self._expected_q = self._expected_q + self.sim_dt * self._expected_qd

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self._advance_expected()
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt
        self.step_count += 1
        self._update_volume_metric()

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        if not np.isfinite(self.state_0.joint_q.numpy()).all():
            raise AssertionError("joint coordinates contain non-finite values")
        if abs(self._mode_value() - self._expected_q) > 1.0e-5:
            raise AssertionError(f"modal position {self._mode_value()} differs from expected {self._expected_q}")
        if abs(self._mode_velocity() - self._expected_qd) > 1.0e-5:
            raise AssertionError(f"modal velocity {self._mode_velocity()} differs from expected {self._expected_qd}")
        if self.step_count > 40 and abs(self._mode_value() - self.mode_q0) < 1.0e-3:
            raise AssertionError("cantilever vibration did not evolve from the initial displacement")
        if self.max_volume_ratio_error > 0.03:
            raise AssertionError(f"cantilever vibration volume changed by {self.max_volume_ratio_error:.3f}")


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
