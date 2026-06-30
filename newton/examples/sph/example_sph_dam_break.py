# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compact free-surface SPH block released inside a Newton collider tank."""

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import SolverWCSPH, sph

from ._config import (
    SPHExampleBase,
    _positive_int,
    add_sph_analytic_tank_shapes,
    add_sph_particle_arguments,
    add_sph_solver_config_arguments,
    add_sph_tank_arguments,
    add_sph_timestep_arguments,
    assert_sph_state_finite,
    sph_options_from_args,
    sph_particle_spacing_from_args,
)


class Example(SPHExampleBase):
    def __init__(self, viewer, args):
        super().__init__(viewer, args)

        builder = newton.ModelBuilder(up_axis=newton.Axis.Y)

        spacing, radius, smoothing_length = sph_particle_spacing_from_args(args)
        material = sph.SPHMaterial(
            rest_density=args.rest_density,
            sound_speed=args.sound_speed,
            viscosity=args.viscosity,
            smoothing_length=smoothing_length,
        )

        fluid_origin = wp.vec3(
            -0.5 * args.tank_length + args.wall_thickness + args.fluid_offset_x,
            args.wall_thickness + args.fluid_offset_y,
            -0.5 * args.fluid_dim_z * spacing,
        )
        self.fluid_indices = np.asarray(
            list(
                sph.add_sph_particle_grid(
                    builder,
                    pos=fluid_origin,
                    rot=wp.quat_identity(),
                    vel=wp.vec3(0.0),
                    dim_x=args.fluid_dim_x,
                    dim_y=args.fluid_dim_y,
                    dim_z=args.fluid_dim_z,
                    cell_x=spacing,
                    cell_y=spacing,
                    cell_z=spacing,
                    material=material,
                    jitter=args.jitter,
                    radius_mean=radius,
                )
            )
        )

        add_sph_analytic_tank_shapes(
            builder,
            tank_length=args.tank_length,
            tank_width=args.tank_width,
            wall_height=args.wall_height,
            wall_thickness=args.wall_thickness,
            boundary_friction=args.boundary_friction,
        )

        self.model = builder.finalize()
        self.model.set_gravity((0.0, args.gravity, 0.0))
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        self.solver = SolverWCSPH(self.model, sph_options_from_args(args))

        self.viewer.set_model(self.model)
        self.viewer.show_particles = True
        self.viewer.set_camera(pos=wp.vec3(1.25, 0.68, 1.10), pitch=-18.0, yaw=-137.0)

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        add_sph_timestep_arguments(parser, substeps=24)
        parser.add_argument(
            "--fluid-dim-x", type=_positive_int, default=5, help="Initial water column particle count along X."
        )
        parser.add_argument(
            "--fluid-dim-y", type=_positive_int, default=7, help="Initial water column particle count along Y."
        )
        parser.add_argument(
            "--fluid-dim-z", type=_positive_int, default=5, help="Initial water column particle count along Z."
        )
        add_sph_particle_arguments(parser, spacing=0.055)
        add_sph_tank_arguments(parser, tank_width=0.45, wall_height=0.45, fluid_offset_y=0.08)
        parser.add_argument(
            "--fluid-offset-x", type=float, default=0.06, help="Initial water offset from the left wall [m]."
        )
        add_sph_solver_config_arguments(
            parser,
            viscosity=0.01,
            boundary_friction=0.05,
        )
        return parser

    def test_final(self):
        assert_sph_state_finite(self.state_0, "density", "pressure")
        q = self.state_0.particle_q.numpy()
        qd = self.state_0.particle_qd.numpy()
        density = self.state_0.sph.density.numpy()
        fluid_q = q[self.fluid_indices]

        assert len(self.fluid_indices) == self.model.particle_count
        assert self.solver.collider_body_index.numpy().tolist() == [-1]
        assert np.all(density[self.fluid_indices] > 0.0)
        assert np.min(fluid_q[:, self.model.up_axis]) >= -1.0e-4
        assert np.max(np.linalg.norm(qd[self.fluid_indices], axis=1)) > 0.0


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
