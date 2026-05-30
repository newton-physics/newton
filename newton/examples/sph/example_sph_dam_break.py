# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example SPH Dam Break
#
# Smoothed Particle Hydrodynamics simulation of a dam break scenario.
# A block of fluid particles collapses under gravity and settles.
#
# Command: python -m newton.examples sph_dam_break
#
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples


class Example:
    def __init__(self, viewer, args):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 4
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer

        spacing = 0.15
        dim_x = 6
        dim_y = 8
        dim_z = 4

        builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
        newton.solvers.SolverSPH.register_custom_attributes(builder)

        # Create a block of fluid particles
        builder.add_particle_grid(
            pos=wp.vec3(0.0, 0.0, 0.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=dim_x,
            dim_y=dim_y,
            dim_z=dim_z,
            cell_x=spacing,
            cell_y=spacing,
            cell_z=spacing,
            mass=0.001,
            jitter=0.0,
        )

        # Ground plane
        builder.add_ground_plane()

        # Enclosure walls (boxes)
        wall_thick = 0.1
        half_w = dim_x * spacing * 0.5 + wall_thick
        half_d = dim_z * spacing * 0.5 + wall_thick
        wall_h = dim_y * spacing * 2.0

        ground = builder.add_shape_box(
            -1,
            pos=wp.vec3(half_w + wall_thick, wall_h * 0.5, 0.0),
            hx=wall_thick,
            hy=wall_h,
            hz=half_d + wall_thick,
        )
        builder.add_shape_box(
            -1,
            pos=wp.vec3(-half_w - wall_thick, wall_h * 0.5, 0.0),
            hx=wall_thick,
            hy=wall_h,
            hz=half_d + wall_thick,
        )
        builder.add_shape_box(
            -1,
            pos=wp.vec3(0.0, wall_h * 0.5, half_d + wall_thick),
            hx=half_w + wall_thick,
            hy=wall_h,
            hz=wall_thick,
        )
        builder.add_shape_box(
            -1,
            pos=wp.vec3(0.0, wall_h * 0.5, -half_d - wall_thick),
            hx=half_w + wall_thick,
            hy=wall_h,
            hz=wall_thick,
        )

        self.model = builder.finalize()
        self.model.set_gravity((0.0, -9.81, 0.0))

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.contacts = self.model.contacts()

        self.solver = newton.solvers.SolverSPH(
            self.model,
            smoothing_length=spacing * 2.5,
            rest_density=1000.0,
            pressure_stiffness=500.0,
            dynamic_viscosity=0.1,
        )

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, None, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt

    def test_final(self):
        # Verify particles have settled (no NaN positions)
        pos = self.state_0.particle_q.numpy()
        assert not np.any(np.isnan(pos)), "Particle positions contain NaN"
        # Verify particles are above ground (y > -0.5)
        assert np.all(pos[:, 1] > -0.5), "Particles below ground plane"

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    newton.examples.run(Example(viewer, args), args)
