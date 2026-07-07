# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
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
# Example Suction Cup
#
# Starting template for a suction-cup surface-gripper demo. For now it just
# drops a box onto a ground plane; the suction-cup model (seal detection and
# the spring-damper hold wrench) will be built on top of this scaffolding.
#
# Command: python -m newton.examples suction_cup
###########################################################################

import math

import warp as wp

import newton
import newton.examples
from newton.examples.suctioncup.surface_gripper import (
    PadShape,
    SurfaceGripper,
    SurfaceGripperBuilder,
    evaluate_gripper_force,
    latch_engagement,
)


class Example:
    def __init__(self, viewer, args):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 4
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer
        self.args = args

        # box half-extents: A is the small gripper body, B is the larger gripped box
        LA = 0.1
        LB = 0.2

        def box_inertia(mass, half):
            s = (1.0 / 6.0) * mass * (2.0 * half) ** 2  # solid cube about its centre
            return wp.diag(wp.vec3(s, s, s))

        m_a, m_b = 0.5, 1.0

        # B centred at z = 0 (top at LB). A's downward pad (at A_center - LA) touches B's
        # top when A_center = LA + LB. NOTE: your 2*LA + LB puts the pad LA above B (a gap),
        # so the seal would latch that gap -- using LA + LB here for a touching seal.
        pose_a = wp.transform(wp.vec3(0.0, 0.0, LA + LB), wp.quat_identity())
        pose_b = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())

        builder = newton.ModelBuilder(gravity=-10.0)

        self.box_a = builder.add_body(xform=pose_a, mass=m_a, inertia=box_inertia(m_a, LA), label="gripper_box_A")
        builder.add_shape_box(self.box_a, hx=LA, hy=LA, hz=LA)

        self.box_b = builder.add_body(xform=pose_b, mass=m_b, inertia=box_inertia(m_b, LB), label="gripped_box_B")
        builder.add_shape_box(self.box_b, hx=LB, hy=LB, hz=LB)

        builder.add_ground_plane()

        self.model = builder.finalize()
        self.solver = newton.solvers.SolverMuJoCo(self.model)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        # surface gripper on box A: one pad covering A's bottom (-z) face
        gripper = SurfaceGripper(
            body_id=self.box_a,
            xform=wp.transform_identity(),  # gripper frame == box A body frame
            k_normal=5000.0,
            d_normal=100.0,
            f_normal_max=200.0,
            f_grip_max=50.0,
            k_shear_x=2000.0,
            k_shear_y=2000.0,
            mu_x=1.0,
            mu_y=1.0,
            d_peel_x=5.0,
            d_peel_y=5.0,
            shape=int(PadShape.RECTANGLE),
            dim_a=LA,
            dim_b=LA,
        )
        # pad at the bottom face: origin (0,0,-LA), +z (suction dir) rotated to point down (-z of A)
        gripper.add_pad(wp.transform(wp.vec3(0.0, 0.0, -LA), wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), math.pi)))

        gbuilder = SurfaceGripperBuilder()
        gbuilder.add_gripper(gripper)
        self.gripper_model = gbuilder.finalize(device=self.model.device)
        self.gripper_state = self.gripper_model.state()
        self.gripper_control = self.gripper_model.control()
        self.gripper_control.pad_grip_control.fill_(1.0)  # full grip command

        # trivial Phase-1 seal decision for now: the single pad is always engaged to box B
        self.seal_engaged = wp.full(1, True, dtype=wp.bool, device=self.model.device)
        self.seal_body_b = wp.full(1, self.box_b, dtype=wp.int32, device=self.model.device)

        self.viewer.set_model(self.model)
        self.capture()

    def capture(self):
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()

            self.viewer.apply_forces(self.state_0)
            self.model.collide(self.state_0, self.contacts)

            # surface gripper -- Phase 1 (trivial: always engaged) then Phase 2 (apply wrench)
            latch_engagement(self.state_0, self.gripper_model, self.gripper_state, self.seal_engaged, self.seal_body_b)
            evaluate_gripper_force(
                self.model, self.state_0, self.gripper_model, self.gripper_state, self.gripper_control
            )

            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt

    def test_final(self):
        # sanity check: the box stayed in a sane range (didn't tunnel through the
        # ground or fly off) -- replace with a suction-specific check later.
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "box stays in a sane range",
            lambda q, qd: 0.0 < q[2] < 0.6 and abs(q[0]) < 0.2 and abs(q[1]) < 0.2,
            [0],
        )

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    newton.examples.run(Example(viewer, args), args)
