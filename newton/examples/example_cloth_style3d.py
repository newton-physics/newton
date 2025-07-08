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
import newton.solvers.style3d.kernels
import newton.solvers.style3d.linear_solver
import newton.utils
from newton.geometry import PARTICLE_FLAG_ACTIVE, Mesh


class Example:
    def __init__(self, stage_path="example_cloth_style3d.usd", num_frames=3000):
        fps = 60
        self.frame_dt = 1.0 / fps
        # must be an even number when using CUDA Graph
        self.num_substeps = 2
        self.iterations = 20
        self.dt = self.frame_dt / self.num_substeps
        self.num_frames = num_frames
        self.sim_time = 0.0
        self.profiler = {}
        self.use_cuda_graph = wp.get_device().is_cuda
        builder = newton.sim.Style3DModelBuilder(up_axis=newton.Axis.Y)

        use_cloth_mesh = True
        if use_cloth_mesh:
            asset_path = newton.utils.download_asset("style3d_description")

            # Grament
            # garment_usd_name = "Women_Skirt"
            # garment_usd_name = "Female_T_Shirt"
            garment_usd_name = "Women_Sweatshirt"
            usd_stage = Usd.Stage.Open(str(asset_path / "garments" / (garment_usd_name + ".usd")))
            usd_geom_garment = UsdGeom.Mesh(usd_stage.GetPrimAtPath(str("/Root/" + garment_usd_name + "/Root_Garment")))

            garment_prim = UsdGeom.PrimvarsAPI(usd_geom_garment.GetPrim()).GetPrimvar("st")
            garment_mesh_indices = np.array(usd_geom_garment.GetFaceVertexIndicesAttr().Get())
            garment_mesh_points = np.array(usd_geom_garment.GetPointsAttr().Get())
            garment_mesh_uv_indices = np.array(garment_prim.GetIndices())
            garment_mesh_uv = np.array(garment_prim.Get()) * 1e-3

            # Avatar
            usd_stage = Usd.Stage.Open(str(asset_path / "avatars" / "Female.usd"))
            usd_geom_avatar = UsdGeom.Mesh(usd_stage.GetPrimAtPath("/Root/Female/Root_SkinnedMesh_Avatar_0_Sub_2"))
            avatar_mesh_indices = np.array(usd_geom_avatar.GetFaceVertexIndicesAttr().Get())
            avatar_mesh_points = np.array(usd_geom_avatar.GetPointsAttr().Get())

            builder.add_aniso_cloth_mesh(
                pos=wp.vec3(0, 0, 0),
                rot=wp.quat_identity(),
                vel=wp.vec3(0.0, 0.0, 0.0),
                tri_aniso_ke=wp.vec3(1.0e2, 1.0e2, 1.0e1),
                edge_aniso_ke=wp.vec3(2.0e-5, 1.0e-5, 5.0e-6),
                panel_verts=garment_mesh_uv.tolist(),
                panel_indices=garment_mesh_uv_indices.tolist(),
                vertices=garment_mesh_points.tolist(),
                indices=garment_mesh_indices.tolist(),
                density=0.3,
                scale=1.0,
                particle_radius=5.0e-3,
            )
            builder.add_shape_mesh(
                body=builder.add_body(),
                mesh=Mesh(avatar_mesh_points, avatar_mesh_indices),
            )
            # fixed_points = [0]
            fixed_points = []
        else:
            grid_dim = 100
            grid_width = 1.0
            cloth_density = 0.3
            builder.add_aniso_cloth_grid(
                pos=wp.vec3(-0.5, 2.0, 0.0),
                rot=wp.quat_from_axis_angle(axis=wp.vec3(1, 0, 0), angle=wp.pi / 2.0),
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
        builder.add_shape_box(
            body=builder.add_body(),
            hx=2.5,
            hy=0.1,
            hz=2.5,
        )
        self.model = builder.finalize()

        # set fixed points
        flags = self.model.particle_flags.numpy()
        for fixed_vertex_id in fixed_points:
            flags[fixed_vertex_id] = wp.uint32(int(flags[fixed_vertex_id]) & ~int(PARTICLE_FLAG_ACTIVE))
        self.model.particle_flags = wp.array(flags)

        # set up contact query and contact detection distances
        self.model.soft_contact_radius = 0.2e-2
        self.model.soft_contact_margin = 0.35e-2
        self.model.soft_contact_ke = 1.0e1
        self.model.soft_contact_kd = 1.0e-6
        self.model.soft_contact_mu = 0.2

        self.solver = newton.solvers.Style3DSolver(
            self.model,
            self.iterations,
        )
        self.solver.precompute(
            builder,
        )
        self.state0 = self.model.state()
        self.state1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.collide(self.state0)

        self.renderer = None
        if stage_path:
            self.renderer = newton.utils.SimRendererOpenGL(path=stage_path, model=self.model, camera_fov=30.0)
            self.renderer.enable_backface_culling = False
            self.renderer.render_wireframe = True
            self.renderer.show_particles = False
            self.renderer.draw_grid = True
            self.renderer.paused = True

        self.cuda_graph = None
        if self.use_cuda_graph:
            # Initial graph launch, load modules (necessary for drivers prior to CUDA 12.3)
            wp.load_module(newton.solvers.style3d.kernels, device=wp.get_device())
            wp.load_module(newton.solvers.style3d.linear_solver, device=wp.get_device())

            with wp.ScopedCapture() as capture:
                self.integrate_frame_substeps()
            self.cuda_graph = capture.graph

    def integrate_frame_substeps(self):
        self.contacts = self.model.collide(self.state0)
        for _ in range(self.num_substeps):
            self.solver.step(self.state0, self.state1, self.control, self.contacts, self.dt)
            (self.state0, self.state1) = (self.state1, self.state0)

    def step(self):
        with wp.ScopedTimer("step", print=False, dict=self.profiler):
            if self.use_cuda_graph:
                wp.capture_launch(self.cuda_graph)
            else:
                self.integrate_frame_substeps()
            self.sim_time += self.dt

    def run(self):
        for frame_idx in range(self.num_frames):
            if self.renderer and self.renderer.has_exit:
                break
            self.step()
            self.render()

            if self.renderer is None:
                print(f"[{frame_idx:4d}/{self.num_frames}]")

    def render(self):
        if self.renderer is not None:
            self.renderer.begin_frame(self.sim_time)
            self.renderer.render(self.state0)
            self.renderer.end_frame()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument(
        "--stage-path",
        type=lambda x: None if x == "None" else str(x),
        default="example_cloth_style3d.usd",
        help="Path to the output USD file.",
    )
    parser.add_argument("--num-frames", type=int, default=3000, help="Total number of frames.")

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(stage_path=args.stage_path, num_frames=args.num_frames)
        example.run()

        if example.renderer:
            example.renderer.save()
