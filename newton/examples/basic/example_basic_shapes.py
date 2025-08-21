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
# Example Basic Shapes
#
# Shows how to programmatically creates a variety of
# collision shapes using the newton.ModelBuilder() API.
#
# Command: python -m newton.examples basic_shapes
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
        self.fps = 100
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer

        builder = newton.ModelBuilder()

        # add ground plane
        builder.add_ground_plane()

        # z height to drop shapes from
        drop_z = 2.0

        # SPHERE
        body_sphere = builder.add_body(xform=wp.transform(p=wp.vec3(0.0, -2.0, drop_z), q=wp.quat_identity()))
        builder.add_shape_sphere(body_sphere, radius=0.5)

        # CAPSULE
        body_capsule = builder.add_body(xform=wp.transform(p=wp.vec3(0.0, 0.0, drop_z), q=wp.quat_identity()))
        builder.add_shape_capsule(body_capsule, radius=0.3, half_height=0.7)

        # CYLINDER (no collision support)
        # body_cylinder = builder.add_body(xform=wp.transform(p=wp.vec3(0.0, -4.0, drop_z), q=wp.quat_identity()))
        # builder.add_shape_cylinder(body_cylinder, radius=0.4, half_height=0.6)

        # BOX
        body_box = builder.add_body(xform=wp.transform(p=wp.vec3(0.0, 2.0, drop_z), q=wp.quat_identity()))
        builder.add_shape_box(body_box, hx=0.5, hy=0.35, hz=0.25)

        # CONE (no collision support)
        # body_cone = builder.add_body(xform=wp.transform(p=wp.vec3(0.0, 6.0, drop_z), q=wp.quat_identity()))
        # builder.add_shape_cone(body_cone, radius=0.45, half_height=0.6)

        # MESH (bunny)
        usd_stage = Usd.Stage.Open(newton.examples.get_asset("bunny.usd"))
        usd_geom = UsdGeom.Mesh(usd_stage.GetPrimAtPath("/root/bunny"))

        mesh_vertices = np.array(usd_geom.GetPointsAttr().Get())
        mesh_indices = np.array(usd_geom.GetFaceVertexIndicesAttr().Get())

        demo_mesh = newton.Mesh(mesh_vertices, mesh_indices)

        body_mesh = builder.add_body(
            xform=wp.transform(p=wp.vec3(0.0, 4.0, drop_z - 0.5), q=wp.quat(0.5, 0.5, 0.5, 0.5))
        )
        builder.add_shape_mesh(body_mesh, mesh=demo_mesh)

        # finalize model
        self.model = builder.finalize()

        self.solver = newton.solvers.SolverXPBD(self.model)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.collide(self.state_0)

        self.viewer.set_model(self.model)

        # not required for MuJoCo, but required for other solvers
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

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

    def test(self):
        pass

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()


if __name__ == "__main__":
    # Parse arguments and initialize viewer
    viewer, args = newton.examples.init()

    # Create viewer and run
    example = Example(viewer)

    newton.examples.run(example)
