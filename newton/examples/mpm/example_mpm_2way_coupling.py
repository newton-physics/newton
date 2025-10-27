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

import newton
import newton.examples
from newton.solvers import SolverImplicitMPM


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
def substract_body_force(
    dt: float,
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_f: wp.array(dtype=wp.spatial_vector),
    body_inv_inertia: wp.array(dtype=wp.mat33),
    body_inv_mass: wp.array(dtype=float),
    body_q_res: wp.array(dtype=wp.transform),
    body_qd_res: wp.array(dtype=wp.spatial_vector),
):
    body_id = wp.tid()

    # Remove previously applied force
    f = body_f[body_id]
    delta_v = dt * body_inv_mass[body_id] * wp.spatial_top(f)
    r = wp.transform_get_rotation(body_q[body_id])

    delta_w = dt * wp.quat_rotate(r, body_inv_inertia[body_id] * wp.quat_rotate_inv(r, wp.spatial_bottom(f)))

    body_q_res[body_id] = body_q[body_id]
    body_qd_res[body_id] = body_qd[body_id] - wp.spatial_vector(delta_v, delta_w)


class Example:
    def __init__(self, viewer):
        # setup simulation parameters first
        self.fps = 100
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 4
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer

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

        # generate a few boxes with varying sizes
        # boxes = [(0.45, 0.35, 0.25)]  # (hx, hy, hz)
        boxes = [
            (0.25, 0.35, 0.25),
            (0.25, 0.25, 0.25),
            (0.3, 0.2, 0.2),
            (0.25, 0.35, 0.25),
            (0.25, 0.25, 0.25),
            (0.3, 0.2, 0.2),
        ]  # (hx, hy, hz)
        collider_body_id = []
        for box in boxes:
            (hx, hy, hz) = box

            ox, oy = offsets_xy[offset_index % len(offsets_xy)]
            offset_index += 1
            pz = drop_z + float(z_index) * z_separation
            z_index += 1
            body = builder.add_body(
                xform=wp.transform(p=wp.vec3(float(ox), float(oy), pz), q=wp.normalize(wp.quatf(0.0, 0.0, 0.0, 1.0))),
                mass=75.0,
            )
            builder.add_shape_box(body, hx=float(hx), hy=float(hy), hz=float(hz))
            # shape_id = builder.add_shape_capsule(body, radius=0.5 * hx, half_height=hz)
            collider_body_id.append(body)

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

        # setup mpm solver
        mpm_options = SolverImplicitMPM.Options()
        mpm_options.voxel_size = voxel_size
        mpm_options.tolerance = 1.0e-6
        mpm_options.grid_type = "fixed"
        mpm_options.grid_padding = 20
        mpm_options.max_active_cell_count = 1 << 15

        mpm_options.strain_basis = "P0"
        mpm_options.max_iterations = 50
        mpm_options.critical_fraction = 0.0

        mpm_model = SolverImplicitMPM.Model(self.sand_model, mpm_options)
        mpm_model.setup_collider(
            model=self.model,  # read colliders from the RB model
            collider_body_ids=collider_body_id,
            collider_friction=[0.5 for _ in collider_body_id],
            collider_adhesion=[0.0 for _ in collider_body_id],
        )
        self.collider_body_id = wp.array(collider_body_id, dtype=int)

        self.mpm_solver = SolverImplicitMPM(mpm_model, mpm_options)

        # simulation state
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        self.sand_state_0 = self.sand_model.state()

        self.sand_body_forces = wp.zeros_like(self.state_0.body_f)

        # enrich states for MPM particles
        self.mpm_solver.enrich_state(self.sand_state_0)

        self.control = self.model.control()
        self.contacts = self.model.collide(self.state_0)

        self.viewer.set_model(self.model)

        # not required for MuJoCo, but required for other solvers
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self.collider_impulses = None
        self.collider_impulse_pos = None
        self.collider_impulse_ids = None
        self.collect_collider_impulses()

        self.particle_render_colors = wp.full(
            self.sand_model.particle_count, value=wp.vec3(0.7, 0.6, 0.4), dtype=wp.vec3, device=self.sand_model.device
        )

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
            # saved applied force to subtract later on
            self.sand_body_forces.assign(self.state_0.body_f)

            # apply forces to the model
            self.viewer.apply_forces(self.state_0)

            self.contacts = self.model.collide(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            # swap states
            self.state_0, self.state_1 = self.state_1, self.state_0

        self.simulate_sand()

    def collect_collider_impulses(self):
        if self.collider_impulses is None:
            self.collider_impulses, self.collider_impulse_pos, self.collider_impulse_ids = (
                self.mpm_solver.collect_collider_impulses(self.sand_state_0)
            )
        else:
            collider_impulses, collider_impulse_pos, collider_impulse_ids = self.mpm_solver.collect_collider_impulses(
                self.sand_state_0
            )
            self.collider_impulses.assign(collider_impulses)
            self.collider_impulse_pos.assign(collider_impulse_pos)
            self.collider_impulse_ids.assign(collider_impulse_ids)

    def simulate_sand(self):
        # Subtract previously applied impulses from body velocities
        wp.launch(
            substract_body_force,
            dim=self.sand_state_0.body_q.shape,
            inputs=[
                self.frame_dt,
                self.state_0.body_q,
                self.state_0.body_qd,
                self.sand_body_forces,
                self.model.body_inv_inertia,
                self.model.body_inv_mass,
                self.sand_state_0.body_q,
                self.sand_state_0.body_qd,
            ],
        )

        self.mpm_solver.step(self.sand_state_0, self.sand_state_0, contacts=None, control=None, dt=self.frame_dt)

        # Save applied impulses
        self.collect_collider_impulses()

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


if __name__ == "__main__":
    # Parse arguments and initialize viewer
    viewer, args = newton.examples.init()

    # Create example and run
    example = Example(viewer)

    newton.examples.run(example, args)
