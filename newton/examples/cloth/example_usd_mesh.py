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

import numpy as np
import warp as wp
from pxr import Usd, UsdGeom

import newton
import newton.examples
import newton.utils
from newton import Mesh, ParticleFlags
from newton._src.solvers.style3d import CollisionHandler


def parse_xform(prim):
    xform = UsdGeom.Xform(prim)
    mat = np.array(xform.GetLocalTransformation(), dtype=np.float32)
    rot = wp.quat_from_matrix(wp.mat33(mat[:3, :3].T.flatten()))
    pos = mat[3, :3]
    return wp.transform(pos, rot)


def load_mesh_shapes_from_usd_prim(builder, prim, body_id, incoming_xform=None, scale=1.0):
    xform = parse_xform(prim)
    xform.p *= scale
    if incoming_xform is not None:
        xform = incoming_xform * xform
    if prim.GetTypeName().lower() == "mesh":
        mesh = UsdGeom.Mesh(prim)
        points = np.array(mesh.GetPointsAttr().Get(), dtype=np.float32) * scale
        indices = np.array(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)
        counts = mesh.GetFaceVertexCountsAttr().Get()
        faces = []
        face_id = 0
        for count in counts:
            if count == 3:
                faces.append(indices[face_id : face_id + 3])
            elif count == 4:
                faces.append(indices[face_id : face_id + 3])
                faces.append(indices[[face_id, face_id + 2, face_id + 3]])
            else:
                continue
            face_id += count
        builder.add_shape_mesh(
            body_id,
            xform,
            mesh=Mesh(points, np.array(faces, dtype=np.int32).flatten()),
            key=str(prim.GetPath()).split("/")[-1],
        )
    for child in prim.GetChildren():
        load_mesh_shapes_from_usd_prim(builder, child, body_id, incoming_xform=xform, scale=scale)


class Example:
    def __init__(self, viewer):
        # setup simulation parameters first
        self.fps = 60
        self.frame_dt = 1.0 / self.fps

        # must be an even number when using CUDA Graph
        self.sim_substeps = 10
        self.sim_time = 0.0
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.iterations = 4

        self.viewer = viewer
        builder = newton.Style3DModelBuilder(up_axis=newton.Axis.Z)

        # Avatar
        avatar_asset = newton.examples.get_asset("bones_tpose_with_mesh.usd")
        usd_stage = Usd.Stage.Open(avatar_asset)
        usd_geom_avatar = usd_stage.GetPrimAtPath("/OUTPUT/c_geometry_grp")
        load_mesh_shapes_from_usd_prim(
            builder,
            usd_geom_avatar,
            body_id=-1,
            incoming_xform=wp.transform(
                wp.vec3(0.0),
                wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), wp.pi * 0.5),
            ),
            scale=0.01,
        )

        grid_dim = 100
        grid_width = 1.0
        cloth_density = 0.3
        builder.add_aniso_cloth_grid(
            pos=wp.vec3(0.5, 0.0, 2.0),
            rot=wp.quat_from_axis_angle(axis=wp.vec3(0, 0, 1), angle=wp.pi),
            dim_x=grid_dim,
            dim_y=grid_dim,
            cell_x=grid_width / grid_dim,
            cell_y=grid_width / grid_dim,
            vel=wp.vec3(0.0, 0.0, 0.0),
            mass=cloth_density * (grid_width * grid_width) / (grid_dim * grid_dim),
            tri_aniso_ke=wp.vec3(1.0e2, 1.0e2, 1.0e1),
            tri_ka=1.0e2,
            tri_kd=2.0e-6,
            edge_aniso_ke=wp.vec3(2.0e-4, 1.0e-4, 5.0e-5),
        )
        fixed_points = [0, grid_dim]

        # add a table
        builder.add_ground_plane()
        self.model = builder.finalize()

        # set fixed points
        flags = self.model.particle_flags.numpy()
        for fixed_vertex_id in fixed_points:
            flags[fixed_vertex_id] = flags[fixed_vertex_id] & ~ParticleFlags.ACTIVE
        self.model.particle_flags = wp.array(flags)

        # set up contact query and contact detection distances
        self.model.soft_contact_radius = 0.2e-2
        self.model.soft_contact_margin = 0.35e-2
        self.model.soft_contact_ke = 1.0e1
        self.model.soft_contact_kd = 1.0e-6
        self.model.soft_contact_mu = 0.2

        self.solver = newton.solvers.SolverStyle3D(
            model=self.model,
            iterations=self.iterations,
            collision_handler=CollisionHandler(self.model),
        )
        self.solver.precompute(
            builder,
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
        self.contacts = self.model.collide(self.state_0)
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()

            # apply forces to the model
            self.viewer.apply_forces(self.state_0)

            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            (self.state_0, self.state_1) = (self.state_1, self.state_0)

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

    # Create example and run
    example = Example(viewer)

    newton.examples.run(example)
