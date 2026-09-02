# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Tendon Pulley
#
# Simplest tendon demonstration: a fixed anchor point with a cable
# attached to a hanging weight.  The tendon acts as a unilateral
# distance constraint — the weight swings freely under gravity like
# a pendulum on a string.
#
# Command: python -m newton.examples tendon_pulley
#
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples
from newton import Axis, TendonLinkType
from newton.examples.cable.cable import assert_tendon_total_length


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

        anchor = builder.add_body(
            xform=wp.transform(p=wp.vec3(0.0, 0.0, 3.0), q=wp.quat_identity()),
            mass=0.0,
            is_kinematic=True,
        )
        builder.add_shape_sphere(anchor, radius=0.05)

        weight = builder.add_link(
            xform=wp.transform(p=wp.vec3(0.5, 0.0, 1.5), q=wp.quat_identity()),
            mass=2.0,
        )
        builder.add_shape_box(weight, hx=0.10, hy=0.10, hz=0.10)

        Dof = newton.ModelBuilder.JointDofConfig
        free_lin = [Dof(axis=Axis.X), Dof(axis=Axis.Y), Dof(axis=Axis.Z)]
        free_ang = [Dof(axis=Axis.X), Dof(axis=Axis.Y), Dof(axis=Axis.Z)]
        j = builder.add_joint_d6(
            parent=-1,
            child=weight,
            linear_axes=free_lin,
            angular_axes=free_ang,
            parent_xform=wp.transform(p=wp.vec3(0.5, 0.0, 1.5), q=wp.quat_identity()),
            child_xform=wp.transform(),
        )
        builder.add_articulation([j])

        builder.add_tendon()
        builder.add_tendon_link(
            body=anchor,
            link_type=TendonLinkType.ATTACHMENT,
            offset=(0.0, 0.0, 0.0),
            axis=(0.0, 1.0, 0.0),
        )
        builder.add_tendon_link(
            body=weight,
            link_type=TendonLinkType.ATTACHMENT,
            offset=(0.0, 0.0, 0.10),
            axis=(0.0, 1.0, 0.0),
            compliance=1.0e-5,
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

        if viewer is not None:
            self.viewer.set_model(self.model)
            self.viewer.set_camera(pos=wp.vec3(0.0, -4.0, 1.5), pitch=5.0, yaw=90.0)
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
        weight_z = body_q[1][2]
        assert weight_z < 2.5, f"Weight should be below anchor: z={weight_z}"

    def render(self):
        if self.viewer is not None:
            self.viewer.begin_frame(self.sim_time)
            self.viewer.log_state(self.state_0)
            att_l = self.solver.tendon_seg_attachment_l.numpy()
            att_r = self.solver.tendon_seg_attachment_r.numpy()
            starts = wp.array(att_l, dtype=wp.vec3)
            ends = wp.array(att_r, dtype=wp.vec3)
            self.viewer.log_lines("cable", starts, ends, colors=(0.9, 0.2, 0.2), width=0.008)
            self.viewer.end_frame()


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    example = Example(viewer, args)
    newton.examples.run(example, args)
