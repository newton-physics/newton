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
# Command: python -m newton.examples mpm_twoway_coupling
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
    """Compute forces applied by sand to rigid bodies.

    Sum the impulses applied on each mpm grid node and convert to
    forces and torques at the body's center of mass.
    """

    i = wp.tid()

    cid = collider_ids[i]
    if cid >= 0 and cid < body_ids.shape[0]:
        body_index = body_ids[cid]
        if body_index == -1:
            return

        f_world = collider_impulses[i] / dt

        X_wb = body_q[body_index]
        X_com = body_com[body_index]
        r = collider_impulse_pos[i] - wp.transform_point(X_wb, X_com)
        wp.atomic_add(body_f, body_index, wp.spatial_vector(f_world, wp.cross(r, f_world)))


@wp.kernel
def subtract_body_force(
    dt: float,
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_f: wp.array(dtype=wp.spatial_vector),
    body_inv_inertia: wp.array(dtype=wp.mat33),
    body_inv_mass: wp.array(dtype=float),
    body_q_res: wp.array(dtype=wp.transform),
    body_qd_res: wp.array(dtype=wp.spatial_vector),
):
    """Update the rigid bodies velocity to remove the forces applied by sand at the last step.

    This is necessary to compute the total impulses that are required to enforce the complementarity-based
    frictional contact boundary conditions.
    """

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

        # setup rigid-body model builder
        builder = newton.ModelBuilder()
        builder.default_shape_cfg.mu = 0.5
        self._emit_rigid_bodies(builder)

        # add ground plane
        builder.add_ground_plane()

        # setup sand model builder
        sand_builder = newton.ModelBuilder()
        voxel_size = 0.05  # 5 cm
        self._emit_particles(sand_builder, voxel_size)

        # finalize models
        self.model = builder.finalize()
        self.sand_model = sand_builder.finalize()

        # basic particle material params
        self.sand_model.particle_mu = 0.48
        self.sand_model.particle_ke = 1.0e15

        # setup mpm solver
        mpm_options = SolverImplicitMPM.Options()
        mpm_options.voxel_size = voxel_size
        mpm_options.tolerance = 1.0e-6
        mpm_options.grid_type = "fixed"  # fixed grid so we can graph-capture
        mpm_options.grid_padding = 50
        mpm_options.max_active_cell_count = 1 << 15

        mpm_options.strain_basis = "P0"
        mpm_options.max_iterations = 50
        mpm_options.critical_fraction = 0.0

        mpm_model = SolverImplicitMPM.Model(self.sand_model, mpm_options)
        # read colliders from the RB model rather than the sand model
        mpm_model.setup_collider(model=self.model)

        self.mpm_solver = SolverImplicitMPM(mpm_model, mpm_options)

        # setup rigid-body solver
        self.solver = newton.solvers.SolverXPBD(self.model)

        # simulation state
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        self.sand_state_0 = self.sand_model.state()
        self.mpm_solver.enrich_state(self.sand_state_0)

        self.control = self.model.control()
        self.contacts = self.model.collide(self.state_0)

        # viewer
        self.viewer.set_model(self.model)
        if isinstance(self.viewer, newton.viewer.ViewerGL):
            self.viewer.register_ui_callback(self.render_ui, position="side")
        self.viewer.show_particles = True
        self.show_impulses = False

        # not required for MuJoCo, but required for other solvers
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        # Additional buffers for tracking two-way coupling forces
        max_nodes = 1 << 20
        self.collider_impulses = wp.zeros(max_nodes, dtype=wp.vec3, device=self.model.device)
        self.collider_impulse_pos = wp.zeros(max_nodes, dtype=wp.vec3, device=self.model.device)
        self.collider_impulse_ids = wp.full(max_nodes, value=-1, dtype=int, device=self.model.device)
        self.collect_collider_impulses()

        # map from collider index to body index
        self.collider_body_id = mpm_model.collider.collider_body_index

        # per-body forces and torques applied by sand to rigid bodies
        self.body_sand_forces = wp.zeros_like(self.state_0.body_f)

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
            self.body_sand_forces.assign(self.state_0.body_f)

            # apply forces to the model
            self.viewer.apply_forces(self.state_0)

            self.contacts = self.model.collide(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            # swap states
            self.state_0, self.state_1 = self.state_1, self.state_0

        self.simulate_sand()

    def collect_collider_impulses(self):
        collider_impulses, collider_impulse_pos, collider_impulse_ids = self.mpm_solver.collect_collider_impulses(
            self.sand_state_0
        )
        self.collider_impulse_ids.fill_(-1)
        n_colliders = min(collider_impulses.shape[0], self.collider_impulses.shape[0])
        self.collider_impulses[:n_colliders].assign(collider_impulses[:n_colliders])
        self.collider_impulse_pos[:n_colliders].assign(collider_impulse_pos[:n_colliders])
        self.collider_impulse_ids[:n_colliders].assign(collider_impulse_ids[:n_colliders])

    def simulate_sand(self):
        # Subtract previously applied impulses from body velocities

        if self.sand_state_0.body_q is not None:
            wp.launch(
                subtract_body_force,
                dim=self.sand_state_0.body_q.shape,
                inputs=[
                    self.frame_dt,
                    self.state_0.body_q,
                    self.state_0.body_qd,
                    self.body_sand_forces,
                    self.model.body_inv_inertia,
                    self.model.body_inv_mass,
                    self.sand_state_0.body_q,
                    self.sand_state_0.body_qd,
                ],
            )

        self.mpm_solver.step(self.sand_state_0, self.sand_state_0, contacts=None, control=None, dt=self.frame_dt)

        # Save impulses to apply back to rigid bodies
        self.collect_collider_impulses()

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt

    def test_final(self):
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "all bodies are above the sand",
            lambda q, qd: q[2] > 0.45,
        )
        voxel_size = self.mpm_solver.mpm_model.voxel_size
        newton.examples.test_particle_state(
            self.sand_state_0,
            "all particles are above the ground",
            lambda q, qd: q[2] > -voxel_size,
        )

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)

        self.viewer.log_points(
            "/sand",
            points=self.sand_state_0.particle_q,
            radii=self.sand_model.particle_radius,
            colors=self.particle_render_colors,
            hidden=not self.viewer.show_particles,
        )

        if self.show_impulses:
            impulses, pos, _cid = self.mpm_solver.collect_collider_impulses(self.sand_state_0)
            self.viewer.log_lines(
                "/impulses",
                starts=pos,
                ends=pos + impulses,
                colors=wp.full(pos.shape[0], value=wp.vec3(1.0, 0.0, 0.0), dtype=wp.vec3),
            )
        else:
            self.viewer.log_lines("/impulses", None, None, None)

        self.viewer.end_frame()

    def render_ui(self, imgui):
        _changed, self.show_impulses = imgui.checkbox("Show Impulses", self.show_impulses)

    def _emit_rigid_bodies(self, builder: newton.ModelBuilder):
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

    def _emit_particles(self, sand_builder: newton.ModelBuilder, voxel_size: float):
        # ------------------------------------------
        # Add sand bed (2m x 2m x 0.5m) above ground
        # ------------------------------------------

        particles_per_cell = 3.0
        density = 2500.0

        bed_lo = np.array([-1.0, -1.0, 0.0])
        bed_hi = np.array([1.0, 1.0, 0.5])
        bed_res = np.array(np.ceil(particles_per_cell * (bed_hi - bed_lo) / voxel_size), dtype=int)

        cell_size = (bed_hi - bed_lo) / bed_res
        cell_volume = np.prod(cell_size)
        radius = float(np.max(cell_size) * 0.5)
        mass = float(np.prod(cell_volume) * density)

        sand_builder.add_particle_grid(
            pos=wp.vec3(bed_lo),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=bed_res[0] + 1,
            dim_y=bed_res[1] + 1,
            dim_z=bed_res[2] + 1,
            cell_x=cell_size[0],
            cell_y=cell_size[1],
            cell_z=cell_size[2],
            mass=mass,
            jitter=2.0 * radius,
            radius_mean=radius,
        )


if __name__ == "__main__":
    # Parse arguments and initialize viewer
    viewer, args = newton.examples.init()

    # Create example and run
    example = Example(viewer)

    newton.examples.run(example, args)
