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
# Example Sim Cloth Bending
#
# This simulation demonstrates cloth bending behavior using the Vertex Block
# Descent (VBD) integrator. A cloth mesh, initially curved, is dropped on
# the ground. The cloth maintains its curved shape due to bending stiffness,
# controlled by edge_ke and edge_kd parameters.
#
###########################################################################

import numpy as np
import warp as wp
from pxr import Usd, UsdGeom

import newton
import newton.examples


class Example:
    def __init__(self, viewer):
        # setup simulation parameters first
        self.fps = 60
        self.frame_dt = 1.0 / self.fps

        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.iterations = 10

        self.viewer = viewer

        usd_stage = Usd.Stage.Open(newton.examples.get_asset("curvedSurface.usd"))
        usd_geom = UsdGeom.Mesh(usd_stage.GetPrimAtPath("/root/cloth"))

        mesh_points = np.array(usd_geom.GetPointsAttr().Get())
        mesh_indices = np.array(usd_geom.GetFaceVertexIndicesAttr().Get())

        self.input_scale_factor = 1.0
        vertices = [wp.vec3(v) * self.input_scale_factor for v in mesh_points]
        self.faces = mesh_indices.reshape(-1, 3)

        builder = newton.ModelBuilder()

        contact_ke = 1.0e2
        contact_kd = 1.0e0
        contact_mu = 0.5
        builder.default_shape_cfg.ke = contact_ke
        builder.default_shape_cfg.kd = contact_kd
        builder.default_shape_cfg.mu = contact_mu
        builder.add_cloth_mesh(
            pos=wp.vec3(0.0, 0.0, 10.0),
            rot=wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), wp.pi / 6.0),
            scale=1.0,
            vertices=vertices,
            indices=mesh_indices,
            vel=wp.vec3(0.0, 0.0, 0.0),
            density=0.02,
            tri_ke=5.0e1,
            tri_ka=5.0e1,
            tri_kd=1.0e-1,
            edge_ke=1.0e1,
            edge_kd=1.0e0,
        )

        builder.color(include_bending=True)
        builder.add_ground_plane()

        self.model = builder.finalize()
        self.model.soft_contact_ke = contact_ke
        self.model.soft_contact_kd = contact_kd
        self.model.soft_contact_mu = contact_mu

        self.solver = newton.solvers.SolverVBD(
            self.model,
            self.iterations,
            handle_self_contact=True,
            self_contact_radius=0.2,
            self_contact_margin=0.35,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.collide(self.state_0)

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

            self.contacts = self.model.collide(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            # swap states
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test(self):
        newton.examples.test_particle_state(
            self.state_0,
            "particles have come close to a rest",
            lambda q, qd: max(abs(qd)) < 0.1,
        )

        p_lower = wp.vec3(-3.0, -3.0, 0.0)
        p_upper = wp.vec3(3.0, 3.0, 2.0)
        newton.examples.test_particle_state(
            self.state_0,
            "particles are within a reasonable volume",
            lambda q, qd: newton.utils.vec_inside_limits(q, p_lower, p_upper),
        )

        newton.examples.test_particle_state(
            self.state_0,
            "lower particles touch the ground",
            lambda q, qd: q[2] < 0.15,
            indices=[4, 5, 12, 13],
        )


if __name__ == "__main__":
    # Parse arguments and initialize viewer
    viewer, args = newton.examples.init()

    # Create viewer and run
    example = Example(viewer)

    newton.examples.run(example, args)
