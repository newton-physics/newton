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
# Example MPM 2-Way Coupling
#
# A simple scene spawning a dozen rigid shapes above a plane. The shapes
# fall and collide using the XPBD solver. Demonstrates basic builder APIs
# and the standard example structure.
#
# Command: python -m newton.examples mpm_2way_coupling
#
###########################################################################

from __future__ import annotations

import numpy as np
import warp as wp
from pxr import Usd, UsdGeom

import newton
import newton.examples
from newton._src.solvers.solver import integrate_bodies, integrate_rigid_body
from newton.solvers import SolverImplicitMPM

# @wp.kernel
# def compute_body_velocities(
#     dt: float,
#     body_q: wp.array(dtype=wp.transform),
#     body_q_next: wp.array(dtype=wp.transform),
#     body_com: wp.array(dtype=wp.vec3),
#     body_qd: wp.array(dtype=wp.spatial_vector),
# ):
#     i = wp.tid()

#     q_inv = wp.quat_inverse(wp.transform_get_rotation(body_q[i]))
#     delta_q = wp.transform_get_rotation(body_q_next[i]) * q_inv

#     v_com = (
#         wp.transform_point(body_q_next[i], body_com[i])
#         - wp.transform_point(body_q[i], body_com[i])
#     ) / dt

#     axis = wp.vec3()
#     angle = float(0.0)
#     wp.quat_to_axis_angle(delta_q, axis, angle)

#     body_qd[i] = wp.spatial_vector(axis * angle / dt, v_com)


@wp.kernel
def update_collider_meshes(
    collider_id: wp.array(dtype=wp.vec2i),
    collider_meshes: wp.array(dtype=wp.uint64),
    src_points: wp.array(dtype=wp.vec3),
    src_shape: wp.array(dtype=int),
    shape_transforms: wp.array(dtype=wp.transform),
    shape_body_id: wp.array(dtype=int),
    body_q_cur: wp.array(dtype=wp.transform),
    body_q_next: wp.array(dtype=wp.transform),
    dt: float,
    body_f: wp.array(dtype=wp.spatial_vector),
    body_inv_inertia: wp.array(dtype=wp.mat33),
    body_coms: wp.array(dtype=wp.vec3),
    body_inv_mass: wp.array(dtype=float),
):
    v = wp.tid()

    cid = collider_id[v][0]
    cv = collider_id[v][1]

    res_mesh = collider_meshes[cid]
    res = wp.mesh_get(res_mesh)

    shape_id = src_shape[v]
    p = wp.transform_point(shape_transforms[shape_id], src_points[v])

    body_id = shape_body_id[shape_id]

    # Remove previously applied force
    f = body_f[body_id]
    delta_v = dt * body_inv_mass[body_id] * wp.spatial_top(f)
    r = wp.transform_get_rotation(body_q_next[body_id])

    dw = dt * body_inv_inertia[body_id] * wp.quat_rotate_inv(r, wp.spatial_bottom(f))
    delta_v += wp.quat_rotate(r, wp.cross(dw, p - body_coms[body_id]))

    # (body_inv_mass[body_id] > 0.0)

    # q_new, qd_new = integrate_rigid_body(
    #     q,
    #     body_f.dtype(0.0),
    #     -f,
    #     body_coms[body_id],
    #     wp.mat33(0.0),
    #     body_inv_mass[body_id],
    #     body_inv_inertia[body_id],
    #     wp.vec3(0.0),
    #     0.0,
    #     dt,
    # )

    cur_p = wp.transform_point(body_q_cur[body_id], p)  # res.points[cv] + dt * res.velocities[cv]
    next_p = wp.transform_point(body_q_next[body_id], p)
    res.velocities[cv] = (next_p - cur_p) / dt - delta_v
    res.points[cv] = cur_p


@wp.kernel
def compute_body_forces(
    dt: float,
    collider_ids: wp.array(dtype=int),
    collider_impulses: wp.array(dtype=wp.vec3),
    collider_impulse_pos: wp.array(dtype=wp.vec3),
    body_ids: wp.array(dtype=int),
    body_q: wp.array(dtype=wp.transform),
    body_com: wp.array(dtype=wp.vec3),
    body_f: wp.array(dtype=wp.spatial_vector),
):
    i = wp.tid()

    cid = collider_ids[i]
    if cid >= 0 and cid < body_ids.shape[0]:
        body_index = body_ids[cid]
        f_world = collider_impulses[i] / dt

        X_wb = body_q[body_index]
        X_com = body_com[body_index]
        r = collider_impulse_pos[i] - wp.transform_point(X_wb, X_com)
        wp.atomic_add(body_f, body_index, wp.spatial_vector(f_world, wp.cross(r, f_world)))


@wp.kernel
def update_collider_coms(
    body_id: wp.array(dtype=int),
    body_q: wp.array(dtype=wp.transform),
    body_inv_inertia: wp.array(dtype=wp.mat33),
    body_coms: wp.array(dtype=wp.vec3),
    collider_inv_inertia: wp.array(dtype=wp.mat33),
    collider_coms: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    body_index = body_id[i]

    X_wb = body_q[body_index]
    X_com = body_coms[body_index]

    collider_coms[i] = wp.transform_point(X_wb, X_com)
    R = wp.quat_to_matrix(wp.transform_get_rotation(X_wb))
    collider_inv_inertia[i] = R @ body_inv_inertia[body_index] @ wp.transpose(R)


class Example:
    def __init__(self, viewer):
        # setup simulation parameters first
        self.fps = 250
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 1
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer

        collider_meshes = []
        collider_ids = []
        collider_shape_ids = []
        collider_body_id = []

        builder = newton.ModelBuilder()

        # add ground plane
        builder.add_ground_plane()

        # z height to drop shapes from
        drop_z = 2.0

        # layout: spawn shapes near the same XY so they collide/stack
        offsets_xy = [
            (0.00, 0.00),
            (0.10, 0.00),
            (-0.10, 0.00),
            (0.00, 0.10),
            (0.00, -0.10),
            (0.10, 0.10),
            (-0.10, 0.10),
            (0.10, -0.10),
            (-0.10, -0.10),
            (0.15, 0.00),
            (-0.15, 0.00),
            (0.00, 0.15),
        ]
        offset_index = 0
        z_index = 0
        z_separation = 0.6  # vertical spacing to avoid initial overlap

        # generate a few shapes with varying types and sizes
        # boxes
        # boxes = [(0.45, 0.35, 0.25)]  # (hx, hy, hz)
        boxes = [(0.45, 0.35, 0.25), (0.25, 0.25, 0.25), (0.6, 0.2, 0.2)]  # (hx, hy, hz)
        for box in boxes:
            (hx, hy, hz) = box

            ox, oy = offsets_xy[offset_index % len(offsets_xy)]
            offset_index += 1
            pz = drop_z + float(z_index) * z_separation
            z_index += 1
            body = builder.add_body(
                xform=wp.transform(p=wp.vec3(float(ox), float(oy), pz), q=wp.normalize(wp.quatf(0.0, 0.0, 0.0, 1.0))),
                mass=1000.0,
            )
            shape_id = builder.add_shape_box(body, hx=float(hx), hy=float(hy), hz=float(hz))
            # shape_id = builder.add_shape_capsule(body, radius=0.5 * hx, half_height=hz)

            cube_points, cube_indices = newton.utils.create_box_mesh(extents=box)
            # cube_points, cube_indices = newton.utils.create_capsule_mesh(radius=0.5 * hx, half_height=hz, up_axis=2)
            nv = cube_points.shape[0]
            cube_mesh = wp.Mesh(
                wp.array(cube_points[:, 0:3], dtype=wp.vec3),
                wp.array(cube_indices, dtype=int),
                velocities=wp.zeros(nv, dtype=wp.vec3),
            )
            # print(cube_indices)
            # cube_points = cube_points[:, 3]

            # cube_mesh = wp.Mesh(
            #     wp.array(np.ascontiguousarray(cube_points), dtype=wp.vec3),
            #     wp.array(cube_indices, dtype=int),
            # )

            collider_meshes.append(cube_mesh)
            collider_ids.append(
                np.hstack(
                    (
                        np.full(nv, len(collider_ids)).reshape(-1, 1),
                        np.arange(nv).reshape(-1, 1),
                    )
                )
            )
            collider_body_id.append(body)
            collider_shape_ids.append(np.full(nv, shape_id, dtype=int))

        # # a few bunnies as triangle meshes
        # usd_stage = Usd.Stage.Open(newton.examples.get_asset("bunny.usd"))
        # usd_geom = UsdGeom.Mesh(usd_stage.GetPrimAtPath("/root/bunny"))
        # mesh_vertices = np.array(usd_geom.GetPointsAttr().Get())
        # mesh_indices = np.array(usd_geom.GetFaceVertexIndicesAttr().Get())
        # demo_mesh = newton.Mesh(mesh_vertices, mesh_indices)

        # # compute uniform scale so the largest dimension is <= 0.5 m
        # vmin = mesh_vertices.min(axis=0)
        # vmax = mesh_vertices.max(axis=0)
        # vsize = vmax - vmin
        # vmax_dim = float(np.max(vsize)) if vsize.size else 1.0
        # bunny_scale = 0.5 / vmax_dim if vmax_dim > 0.0 else 1.0

        # bunny_offsets = [(-0.7, 0.35, 0.2), (0.7, 0.35, 0.0), (0.0, -0.7, -1.0)]
        # for _, (ox, oy, oz) in enumerate(bunny_offsets):
        #     bx, by = offsets_xy[offset_index % len(offsets_xy)]
        #     offset_index += 1
        #     pz = drop_z + float(z_index) * z_separation
        #     z_index += 1
        #     p = wp.vec3(float(bx + ox), float(by + oy), float(pz - 0.5 + oz))
        #     q = wp.quat(0.5, 0.5, 0.5, 0.5)
        #     body = builder.add_body(xform=wp.transform(p=p, q=q))
        #     builder.add_shape_mesh(body, mesh=demo_mesh, scale=wp.vec3(bunny_scale, bunny_scale, bunny_scale))

        #   collider_meshes.append(demo_mesh)

        # ------------------------------------------
        # Add sand bed (2m x 2m x 0.5m) above ground
        # ------------------------------------------
        voxel_size = 0.05  # 5 cm
        particles_per_cell = 3.0
        density = 2500.0

        bed_lo = np.array([-1.0, -1.0, 0.0])
        bed_hi = np.array([1.0, 1.0, 0.5])
        bed_res = np.array(np.ceil(particles_per_cell * (bed_hi - bed_lo) / voxel_size), dtype=int)

        # spawn particles on a jittered grid
        Nx, Ny, Nz = bed_res
        px = np.linspace(bed_lo[0], bed_hi[0], Nx + 1)
        py = np.linspace(bed_lo[1], bed_hi[1], Ny + 1)
        pz = np.linspace(bed_lo[2], bed_hi[2], Nz + 1)
        points = np.stack(np.meshgrid(px, py, pz)).reshape(3, -1).T

        cell_size = (bed_hi - bed_lo) / bed_res
        cell_volume = np.prod(cell_size)
        radius = float(np.max(cell_size) * 0.5)
        mass = float(np.prod(cell_volume) * density)

        rng = np.random.default_rng()
        points += 2.0 * radius * (rng.random(points.shape) - 0.5)
        vel = np.zeros_like(points)

        sand_builder = newton.ModelBuilder()
        sand_builder.particle_q = points
        sand_builder.particle_qd = vel
        sand_builder.particle_mass = np.full(points.shape[0], mass)
        sand_builder.particle_radius = np.full(points.shape[0], radius)
        sand_builder.particle_flags = np.ones(points.shape[0], dtype=int)

        # finalize models
        # for now keep two separate models, as we do not have enough control
        # over the collision pipeline
        self.model = builder.finalize()
        self.sand_model = sand_builder.finalize()

        # basic particle material params
        self.sand_model.particle_mu = 0.48
        self.sand_model.particle_ke = 1.0e15

        # setup rigid-body solver
        self.solver = newton.solvers.SolverXPBD(self.model)

        self.collider_meshes = collider_meshes
        self.collider_mesh_ids = wp.array([mesh.id for mesh in collider_meshes], dtype=wp.uint64)
        self.collider_ids = wp.array(np.vstack(collider_ids), dtype=wp.vec2i)
        self.collider_shape_ids = wp.array(np.concatenate(collider_shape_ids), dtype=int)
        self.collider_body_id = wp.array(collider_body_id, dtype=int)
        self.collider_rest_points = wp.array(
            np.vstack([mesh.points.numpy() for mesh in collider_meshes]), dtype=wp.vec3
        )

        # setup mpm solver
        mpm_options = SolverImplicitMPM.Options()
        mpm_options.voxel_size = voxel_size
        mpm_options.tolerance = 1.0e-8
        mpm_options.grid_type = "sparse"
        mpm_options.strain_basis = "P0"
        mpm_options.max_iterations = 250

        mpm_model = SolverImplicitMPM.Model(self.sand_model, mpm_options)

        body_masses = self.model.body_mass.numpy()
        mpm_model.setup_collider(
            colliders=self.collider_meshes,
            collider_masses=[body_masses[body_id] for body_id in collider_body_id],
            collider_friction=[0.5 for _ in collider_body_id],
            collider_adhesion=[1.0e5 for _ in collider_body_id],
            # collider_thicknesses=[0.1 for _ in collider_body_id],
        )
        self.mpm_solver = SolverImplicitMPM(mpm_model, mpm_options)

        # simulation state
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.sand_state_0 = self.sand_model.state()
        self.sand_state_1 = self.sand_model.state()

        self.sand_body_forces = wp.zeros_like(self.state_0.body_f)

        # enrich states for MPM particles
        self.mpm_solver.enrich_state(self.sand_state_0)
        self.mpm_solver.enrich_state(self.sand_state_1)

        self.control = self.model.control()
        self.contacts = self.model.collide(self.state_0)

        self.viewer.set_model(self.model)

        # not required for MuJoCo, but required for other solvers
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self.ref_q = wp.clone(self.state_0.body_q)
        self._update_collider_mesh(self.state_0)
        self.collect_collider_impulses()

        self.particle_render_colors = wp.full(
            self.sand_model.particle_count, value=wp.vec3(0.7, 0.6, 0.4), dtype=wp.vec3, device=self.sand_model.device
        )

        self.capture()

    def capture(self):
        if False and wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()

            # self.sand_body_forces.zero_()
            wp.launch(
                compute_body_forces,
                dim=self.collider_impulse_ids.shape[0],
                inputs=[
                    self.frame_dt,
                    self.collider_impulse_ids,
                    self.collider_impulses,
                    self.collider_impulse_pos,
                    self.collider_body_id,
                    self.state_0.body_q,
                    self.model.body_com,
                    self.state_0.body_f,
                ],
            )
            self.sand_body_forces.assign(self.state_0.body_f)
            # wp.assign(self.state_0.body_f, self.sand_body_forces)
            print(self.state_0.body_f)

            # self.state_0.body_f.assign(self.sand_body_forces)

            # apply forces to the model
            self.viewer.apply_forces(self.state_0)

            self.contacts = self.model.collide(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            # swap states
            self.state_0, self.state_1 = self.state_1, self.state_0

    def collect_collider_impulses(self):
        self.collider_impulses, self.collider_impulse_pos, self.collider_impulse_ids = (
            self.mpm_solver.collect_collider_impulses(self.sand_state_0)
        )

    def simulate_sand(self):
        # one MPM step per frame (no collider setup for now)
        self._update_collider_mesh(self.state_0)

        self.mpm_solver.step(self.sand_state_0, self.sand_state_0, contacts=None, control=None, dt=self.frame_dt)

        self.collect_collider_impulses()

        # body_impulses = collider_impulses.numpy()[collider_ids.numpy() == 0]

        # print(self.state_0.body_q)
        # delta_qd = wp.zeros_like(self.state_0.body_qd)
        # wp.launch(
        #     kernel=integrate_bodies,
        #     dim=self.model.body_count,
        #     inputs=[
        #         self.state_0.body_q,
        #         delta_qd,
        #         self.sand_body_forces,
        #         self.model.body_com,
        #         self.model.body_mass,
        #         self.model.body_inertia,
        #         self.model.body_inv_mass,
        #         self.model.body_inv_inertia,
        #         wp.vec3(0.0),
        #         1.0,
        #         self.frame_dt,
        #     ],
        #     outputs=[self.state_0.body_q, delta_qd],
        #     device=self.model.device,
        # )
        # self.state_0.body_qd += delta_qd

        # print(np.sum(body_impulses) / self.frame_dt)
        # print(self.model.gravity * self.model.body_mass.numpy()[0])
        # print(self.sand_body_forces.numpy())

    def step(self):
        self.ref_q.assign(self.state_0.body_q)

        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.simulate_sand()

        self.sim_time += self.frame_dt

    def test(self):
        pass

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.log_points(
            "sand",
            points=self.sand_state_0.particle_q,
            radii=self.sand_model.particle_radius,
            colors=self.particle_render_colors,
            hidden=False,
        )

        impulses, pos, cid = self.mpm_solver.collect_collider_impulses(self.sand_state_0)
        self.viewer.log_lines(
            "impulses",
            starts=pos,
            ends=pos + impulses,
            colors=wp.full(pos.shape[0], value=wp.vec3(1.0, 0.0, 0.0), dtype=wp.vec3),
        )

        self.viewer.end_frame()

    def _update_collider_mesh(self, state_next):
        wp.launch(
            update_collider_coms,
            dim=self.collider_body_id.shape[0],
            inputs=[
                self.collider_body_id,
                state_next.body_q,
                self.model.body_inv_inertia,
                self.model.body_com,
                self.mpm_solver.mpm_model.collider_inv_inertia,
                self.mpm_solver.mpm_model.collider_coms,
            ],
        )
        wp.launch(
            update_collider_meshes,
            dim=self.collider_rest_points.shape[0],
            inputs=[
                self.collider_ids,
                self.collider_mesh_ids,
                self.collider_rest_points,
                self.collider_shape_ids,
                self.model.shape_transform,
                self.model.shape_body,
                self.ref_q,
                state_next.body_q,
                self.frame_dt,
                self.sand_body_forces,
                self.model.body_inv_inertia,
                self.model.body_com,
                self.model.body_inv_mass,
            ],
        )

        for mesh in self.collider_meshes:
            mesh.refit()


if __name__ == "__main__":
    # Parse arguments and initialize viewer
    viewer, args = newton.examples.init()

    # Create example and run
    example = Example(viewer)

    newton.examples.run(example)
