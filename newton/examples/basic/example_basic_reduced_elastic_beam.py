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
# Example Basic Reduced Elastic Beam
#
# Demonstrates a reduced elastic cantilever beam using the Euler-Bernoulli
# tip-load shape function. The modal coordinate has the analytic static
# solution q = P L^3 / (3 E I) for tip load P.
#
# Command: python -m newton.examples basic_reduced_elastic_beam
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

        self.viewer = viewer
        self.args = args

        self.length = 0.9
        self.hy = 0.045
        self.hz = 0.035
        self.z = 0.22
        self.ei = 0.24
        self.tip_load_mean = 0.12
        self.tip_load_amp = 0.05
        self.mode_stiffness = 3.0 * self.ei / (self.length**3)
        self._last_tip_load = self.tip_load_mean

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
            sample_count=33,
            label="cantilever_basis",
        ).build(
            sample_points=beam_render_sample_points(
                self.length,
                self.hy,
                self.hz,
                extra_points=((-0.5 * self.length, 0.0, 0.0), (0.5 * self.length, 0.0, 0.0)),
            )
        )

        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        builder.add_ground_plane()

        inertia = wp.mat33(0.015, 0.0, 0.0, 0.0, 0.015, 0.0, 0.0, 0.0, 0.015)
        self.beam = builder.add_body_elastic(
            xform=wp.transform(wp.vec3(0.5 * self.length, 0.0, self.z), wp.quat_identity()),
            mass=1.0,
            inertia=inertia,
            mode_count=1,
            mode_mass=[0.0],
            mode_stiffness=[self.mode_stiffness],
            mode_damping=[0.0],
            mode_q=[self.tip_load_mean / self.mode_stiffness],
            modal_basis=beam_basis,
            label="elastic_cantilever",
        )

        shape_cfg = newton.ModelBuilder.ShapeConfig()
        shape_cfg.density = 0.0
        builder.add_shape_box(self.beam, hx=0.5 * self.length, hy=self.hy, hz=self.hz, cfg=shape_cfg)
        self.fixed_joint = builder.add_joint_fixed(
            parent=-1,
            child=self.beam,
            parent_xform=wp.transform(wp.vec3(0.0, 0.0, self.z), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(-0.5 * self.length, 0.0, 0.0), wp.quat_identity()),
            label="world_cantilever_root",
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
        self.viewer.set_camera(wp.vec3(0.45, -1.25, 0.58), -22.0, 90.0)

    def _mode_shape(self, x: np.ndarray) -> np.ndarray:
        s = float(x[0] + 0.5 * self.length)
        s = min(max(s, 0.0), self.length)
        phi = (s * s * (3.0 * self.length - s)) / (2.0 * self.length**3)
        slope = (3.0 * s * (2.0 * self.length - s)) / (2.0 * self.length**3)
        return np.array([[-float(x[2]) * slope, 0.0, phi]], dtype=np.float32)

    def _mode_value(self) -> float:
        return float(self.state_0.joint_q.numpy()[self.elastic_q_start + 7])

    def _update_volume_metric(self):
        ratio = elastic_shape_volume_ratio(self.model, self.state_0)
        self.max_volume_ratio_error = max(self.max_volume_ratio_error, abs(ratio - 1.0))

    def _set_controls(self, t: float):
        self._last_tip_load = self.tip_load_mean + self.tip_load_amp * math.sin(2.0 * math.pi * 0.6 * t)
        self._joint_f[:] = 0.0
        self._joint_f[self.elastic_qd_start + 6] = self._last_tip_load
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
        if not np.isfinite(self.state_0.joint_q.numpy()).all():
            raise AssertionError("joint coordinates contain non-finite values")
        expected = self._last_tip_load / self.mode_stiffness
        if abs(self._mode_value() - expected) > 2.0e-4:
            raise AssertionError(f"cantilever tip deflection {self._mode_value()} differs from analytic {expected}")
        if self.max_volume_ratio_error > 0.06:
            raise AssertionError(f"cantilever volume changed by {self.max_volume_ratio_error:.3f}")


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
