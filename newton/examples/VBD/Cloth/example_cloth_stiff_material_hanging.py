# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Sim Cloth Stiff Material Hanging
#
# This simulation loads the `square_cloth.usd` asset and hangs it from one
# edge under gravity, using the VBD solver with stiff material parameters.
#
# Command: python -m newton.examples cloth_stiff_material_hanging
#
###########################################################################

import os

import numpy as np
import warp as wp
import warp.examples
from pxr import Usd

import newton
import newton.examples
import newton.usd
from newton import ParticleFlags


class Example:
    def __init__(self, viewer, args=None):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps

        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.iterations = 10

        self.viewer = viewer

        usd_stage = Usd.Stage.Open(os.path.join(warp.examples.get_asset_directory(), "square_cloth.usd"))
        usd_prim = usd_stage.GetPrimAtPath("/root/cloth/cloth")

        cloth_mesh = newton.usd.get_mesh(usd_prim)
        mesh_points = cloth_mesh.vertices
        mesh_indices = cloth_mesh.indices

        vertices = [wp.vec3(v) for v in mesh_points]

        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        builder.add_cloth_mesh(
            pos=wp.vec3(0.0, 0.0, 2.0),
            rot=wp.quat_from_axis_angle(wp.vec3(1, 0, 0), np.pi / 2),
            # rot=wp.quat_identity(),
            scale=0.01,
            vertices=vertices,
            indices=mesh_indices,
            vel=wp.vec3(0.0, 0.0, 0.0),
            density=0.2,
            tri_ke=1.0e8,
            tri_ka=1.0e8,
            tri_kd=0,
            edge_ke=1.0e1,
            edge_kd=0.0,
        )
        builder.color(include_bending=True)

        self.model = builder.finalize()
        self.model.soft_contact_ke = 1.0e2
        self.model.soft_contact_kd = 1.0e0
        self.model.soft_contact_mu = 1.0

        cloth_size = 50
        fixed_side = [cloth_size - 1 + i * cloth_size for i in range(cloth_size)]

        flags = self.model.particle_flags.numpy()
        for fixed_vertex_id in fixed_side:
            flags[fixed_vertex_id] = flags[fixed_vertex_id] & ~ParticleFlags.ACTIVE
        self.model.particle_flags = wp.array(flags)

        self.solver = newton.solvers.SolverVBD(
            model=self.model,
            iterations=self.iterations,
            particle_enable_self_contact=False,
            particle_self_contact_radius=0.002,
            particle_self_contact_margin=0.0035,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        self.contacts = self.model.contacts()

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
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt

    def test_final(self):
        newton.examples.test_particle_state(
            self.state_0,
            "particles are above the ground",
            lambda q, qd: q[2] > 0.0,
        )

        p_lower = wp.vec3(-1.0, -1.0, 0.0)
        p_upper = wp.vec3(1.0, 1.0, 3.0)
        newton.examples.test_particle_state(
            self.state_0,
            "particles are within a reasonable volume",
            lambda q, qd: newton.math.vec_inside_limits(q, p_lower, p_upper),
        )

        # Non-explosion: velocities stay bounded. StVK at this stiffness
        # has a non-PSD membrane Hessian under compression and VBD blows up
        # to thousands of m/s; Neo-Hookean keeps residual swing under 20.
        newton.examples.test_particle_state(
            self.state_0,
            "particle velocities do not explode",
            lambda q, qd: wp.length(qd) < 20.0,
        )

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()


if __name__ == "__main__":
    parser = newton.examples.create_parser()

    viewer, args = newton.examples.init(parser)

    example = Example(viewer=viewer, args=args)

    newton.examples.run(example, args)
