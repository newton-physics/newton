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
# Example Basic Joints
#
# Shows how to use the ModelBuilder API to programmatically create different
# joint types: BALL, DISTANCE, PRISMATIC, and REVOLUTE.
#
# Command: python -m newton.examples basic_joints
#
###########################################################################

import warp as wp

import newton
import newton.examples


class Example:
    def __init__(self, viewer, args=None):
        # setup simulation parameters first
        self.fps = 100
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer
        self.args = args

        builder = newton.ModelBuilder()

        # add ground plane
        builder.add_ground_plane()

        # common geometry settings
        cuboid_hx = 0.1
        cuboid_hy = 0.1
        cuboid_hz = 0.75
        upper_hz = 0.25 * cuboid_hz

        # layout positions (y-rows)
        rows = [-3.0, 0.0, 3.0]
        drop_z = 2.0

        # -----------------------------
        # REVOLUTE (hinge) joint demo
        # -----------------------------
        y = rows[0]

        a_rev = builder.add_link(xform=wp.transform(p=wp.vec3(0.0, y, drop_z + upper_hz), q=wp.quat_identity()))
        b_rev = builder.add_link(
            xform=wp.transform(
                p=wp.vec3(0.0, y, drop_z - cuboid_hz), q=wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.15)
            ),
            key="b_rev",
        )
        builder.add_shape_box(a_rev, hx=cuboid_hx, hy=cuboid_hy, hz=upper_hz)
        builder.add_shape_box(b_rev, hx=cuboid_hx, hy=cuboid_hy, hz=cuboid_hz)

        j_fixed_rev = builder.add_joint_fixed(
            parent=-1,
            child=a_rev,
            parent_xform=wp.transform(p=wp.vec3(0.0, y, drop_z + upper_hz), q=wp.quat_identity()),
            child_xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity()),
            key="fixed_revolute_anchor",
        )
        j_revolute = builder.add_joint_revolute(
            parent=a_rev,
            child=b_rev,
            axis=wp.vec3(1.0, 0.0, 0.0),
            parent_xform=wp.transform(p=wp.vec3(0.0, 0.0, -upper_hz), q=wp.quat_identity()),
            child_xform=wp.transform(p=wp.vec3(0.0, 0.0, +cuboid_hz), q=wp.quat_identity()),
            key="revolute_a_b",
        )
        # Create articulation from joints
        builder.add_articulation([j_fixed_rev, j_revolute], key="revolute_articulation")

        # set initial joint angle
        builder.joint_q[-1] = wp.pi * 0.5

        # -----------------------------
        # PRISMATIC (slider) joint demo
        # -----------------------------
        y = rows[1]
        a_pri = builder.add_link(xform=wp.transform(p=wp.vec3(0.0, y, drop_z + upper_hz), q=wp.quat_identity()))
        b_pri = builder.add_link(
            xform=wp.transform(
                p=wp.vec3(0.0, y, drop_z - cuboid_hz), q=wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), 0.12)
            ),
            key="b_prismatic",
        )
        builder.add_shape_box(a_pri, hx=cuboid_hx, hy=cuboid_hy, hz=upper_hz)
        builder.add_shape_box(b_pri, hx=cuboid_hx, hy=cuboid_hy, hz=cuboid_hz)

        j_fixed_pri = builder.add_joint_fixed(
            parent=-1,
            child=a_pri,
            parent_xform=wp.transform(p=wp.vec3(0.0, y, drop_z + upper_hz), q=wp.quat_identity()),
            child_xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity()),
            key="fixed_prismatic_anchor",
        )
        j_prismatic = builder.add_joint_prismatic(
            parent=a_pri,
            child=b_pri,
            axis=wp.vec3(0.0, 0.0, 1.0),  # slide along Z
            parent_xform=wp.transform(p=wp.vec3(0.0, 0.0, -upper_hz), q=wp.quat_identity()),
            child_xform=wp.transform(p=wp.vec3(0.0, 0.0, +cuboid_hz), q=wp.quat_identity()),
            limit_lower=-0.3,
            limit_upper=0.3,
            key="prismatic_a_b",
        )
        # Create articulation from joints
        builder.add_articulation([j_fixed_pri, j_prismatic], key="prismatic_articulation")

        # -----------------------------
        # BALL joint demo (sphere + cuboid)
        # -----------------------------
        y = rows[2]
        radius = 0.3
        z_offset = -1.0  # Shift down by 2 units

        # kinematic (massless) sphere as the parent anchor
        a_ball = builder.add_link(
            xform=wp.transform(p=wp.vec3(0.0, y, drop_z + radius + cuboid_hz + z_offset), q=wp.quat_identity())
        )
        b_ball = builder.add_link(
            xform=wp.transform(
                p=wp.vec3(0.0, y, drop_z + radius + z_offset), q=wp.quat_from_axis_angle(wp.vec3(1.0, 1.0, 0.0), 0.1)
            ),
            key="b_ball",
        )

        rigid_cfg = newton.ModelBuilder.ShapeConfig()
        rigid_cfg.density = 0.0
        builder.add_shape_sphere(a_ball, radius=radius, cfg=rigid_cfg)
        builder.add_shape_box(b_ball, hx=cuboid_hx, hy=cuboid_hy, hz=cuboid_hz)

        # Connect parent to world
        j_fixed_ball = builder.add_joint_fixed(
            parent=-1,
            child=a_ball,
            parent_xform=wp.transform(p=wp.vec3(0.0, y, drop_z + radius + cuboid_hz + z_offset), q=wp.quat_identity()),
            child_xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity()),
            key="fixed_ball_anchor",
        )
        j_ball = builder.add_joint_ball(
            parent=a_ball,
            child=b_ball,
            parent_xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity()),
            child_xform=wp.transform(p=wp.vec3(0.0, 0.0, +cuboid_hz), q=wp.quat_identity()),
            key="ball_a_b",
        )

        # Create articulation from joints
        builder.add_articulation([j_fixed_ball, j_ball], key="ball_articulation")

        # set initial joint angle
        builder.joint_q[-4:] = wp.quat_rpy(0.5, 0.6, 0.7)

        # finalize model
        self.model = builder.finalize()

        self.solver = newton.solvers.SolverXPBD(self.model)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        # not required for MuJoCo, but required for other solvers
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        # Create collision pipeline from command-line args (default: CollisionPipeline with EXPLICIT)
        self.collision_pipeline = newton.examples.create_collision_pipeline(self.model, self.args)
        self.contacts = self.model.collide(self.state_0, collision_pipeline=self.collision_pipeline)

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

            # apply forces to the model
            self.viewer.apply_forces(self.state_0)

            self.contacts = self.model.collide(self.state_0, collision_pipeline=self.collision_pipeline)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            # swap states
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt

    def test_post_step(self):
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "revolute motion in plane",
            lambda q, qd: wp.length(abs(wp.cross(wp.spatial_bottom(qd), wp.vec3(1.0, 0.0, 0.0)))) < 1e-5,
            indices=[self.model.body_key.index("b_rev")],
        )

        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "linear motion on axis",
            lambda q, qd: wp.length(abs(wp.cross(wp.spatial_top(qd), wp.vec3(0.0, 0.0, 1.0)))) < 1e-5
            and wp.length(wp.spatial_bottom(qd)) < 1e-5,
            indices=[self.model.body_key.index("b_prismatic")],
        )

        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "ball motion on sphere",
            lambda q, qd: abs(wp.dot(wp.spatial_bottom(qd), wp.vec3(0.0, 0.0, 1.0))) < 1e-3,
            indices=[self.model.body_key.index("b_ball")],
        )

    def test_final(self):
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "static bodies are not moving",
            lambda q, qd: max(abs(qd)) == 0.0,
            indices=[2, 4],
        )
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "fixed link body has come to a rest",
            lambda q, qd: max(abs(qd)) < 1e-2,
            indices=[0],
        )
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "slider link body has come to a rest",
            lambda q, qd: max(abs(qd)) < 1e-5,
            indices=[3],
        )
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "movable links are not moving too fast",
            lambda q, qd: max(abs(qd)) < 3.0,
            indices=[1, 5],
        )

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()


if __name__ == "__main__":
    # Parse arguments and initialize viewer
    viewer, args = newton.examples.init()

    # Create viewer and run
    example = Example(viewer, args)

    newton.examples.run(example, args)
