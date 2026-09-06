# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Tendon Pinhole
#
# A cable threaded through a pinhole (eyelet) on a fixed guide body.
# Two weights hang on either side — the heavier weight pulls cable
# through the pinhole, descending while the lighter weight ascends.
#
# Unlike a rolling pulley, the pinhole has no radius and no wrap arc.
# The cable simply passes through a single point on the body.
#
# Command: python -m newton.examples tendon_pinhole
#
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples
from newton import Axis, TendonLinkType
from newton.examples.cable.cable import assert_tendon_total_length, get_tendon_cable_lines


class Example:
    def __init__(self, viewer, args):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 16
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer
        self.args = args

        builder = newton.ModelBuilder(up_axis=Axis.Z, gravity=(0.0, 0.0, -9.81))

        Dof = newton.ModelBuilder.JointDofConfig

        guide = builder.add_body(
            xform=wp.transform(p=wp.vec3(0.0, 0.0, 3.5), q=wp.quat_identity()),
            mass=0.0,
            is_kinematic=True,
        )
        builder.add_shape_sphere(guide, radius=0.05)

        planar_lin = [Dof(axis=Axis.X), Dof(axis=Axis.Z)]
        planar_ang = [Dof(axis=Axis.Y)]

        self.left_idx = left = builder.add_link(
            xform=wp.transform(p=wp.vec3(-0.3, 0.0, 2.0), q=wp.quat_identity()),
            mass=1.0,
        )
        builder.add_shape_box(left, hx=0.06, hy=0.06, hz=0.06)
        j1 = builder.add_joint_d6(
            parent=-1,
            child=left,
            linear_axes=planar_lin,
            angular_axes=planar_ang,
            parent_xform=wp.transform(p=wp.vec3(-0.3, 0.0, 2.0), q=wp.quat_identity()),
            child_xform=wp.transform(),
        )
        builder.add_articulation([j1])

        self.right_idx = right = builder.add_link(
            xform=wp.transform(p=wp.vec3(0.3, 0.0, 2.0), q=wp.quat_identity()),
            mass=3.0,
        )
        builder.add_shape_box(right, hx=0.10, hy=0.10, hz=0.10)
        j2 = builder.add_joint_d6(
            parent=-1,
            child=right,
            linear_axes=planar_lin,
            angular_axes=planar_ang,
            parent_xform=wp.transform(p=wp.vec3(0.3, 0.0, 2.0), q=wp.quat_identity()),
            child_xform=wp.transform(),
        )
        builder.add_articulation([j2])

        axis = (0.0, 1.0, 0.0)
        builder.add_tendon()
        builder.add_tendon_link(
            body=left,
            link_type=TendonLinkType.ATTACHMENT,
            offset=(0.0, 0.0, 0.06),
            axis=axis,
        )
        builder.add_tendon_link(
            body=guide,
            link_type=TendonLinkType.PINHOLE,
            offset=(0.0, 0.0, 0.0),
            axis=axis,
            compliance=1.0e-6,
            damping=0.1,
            rest_length=-1.0,
        )
        builder.add_tendon_link(
            body=right,
            link_type=TendonLinkType.ATTACHMENT,
            offset=(0.0, 0.0, 0.10),
            axis=axis,
            compliance=1.0e-6,
            damping=0.1,
            rest_length=-1.0,
        )

        builder.add_ground_plane()

        self.model = builder.finalize()

        self.solver = newton.solvers.SolverXPBD(
            self.model,
            iterations=8,
            joint_linear_relaxation=0.8,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.collision_pipeline = newton.CollisionPipeline(self.model)
        self.contacts = self.collision_pipeline.contacts()

        if self.viewer is not None:
            self.viewer.set_model(self.model)
            self.viewer.set_camera(pos=wp.vec3(0.0, -4.0, 2.0), pitch=5.0, yaw=90.0)
            if hasattr(self.viewer, "renderer"):
                self.viewer.renderer.show_wireframe_overlay = True

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            if self.viewer is not None:
                self.viewer.apply_forces(self.state_0)
            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt

    def test_post_step(self):
        assert_tendon_total_length(self)

    def test_final(self):
        assert_tendon_total_length(self)
        body_q = self.state_0.body_q.numpy()
        assert np.isfinite(body_q).all(), "Non-finite values in body positions"
        assert (np.abs(body_q[:, :3]) < 100.0).all(), "Body positions diverged"

    def render(self):
        if self.viewer is not None:
            self.viewer.begin_frame(self.sim_time)
            self.viewer.log_state(self.state_0)
            starts, ends = get_tendon_cable_lines(self.solver, self.model, self.state_0)
            self.viewer.log_lines("cable", starts, ends, colors=(0.9, 0.2, 0.2), width=0.008)
            self.viewer.end_frame()


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    example = Example(viewer, args)
    newton.examples.run(example, args)
