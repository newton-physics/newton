# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Newton-native SPH fluid block."""

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import SolverWCSPH, sph

from ._config import (
    SPHExampleBase,
    _non_negative_float,
    _positive_float,
    add_sph_block_dimension_arguments,
    add_sph_particle_arguments,
    add_sph_solver_config_arguments,
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
        self.fluid_indices = np.asarray(
            list(
                sph.add_sph_particle_grid(
                    builder,
                    pos=wp.vec3(-0.5 * args.dim_x * spacing, args.height, -0.5 * args.dim_z * spacing),
                    rot=wp.quat_identity(),
                    vel=wp.vec3(0.0),
                    dim_x=args.dim_x,
                    dim_y=args.dim_y,
                    dim_z=args.dim_z,
                    cell_x=spacing,
                    cell_y=spacing,
                    cell_z=spacing,
                    material=material,
                    jitter=args.jitter,
                    radius_mean=radius,
                )
            ),
            dtype=np.int64,
        )
        builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=args.boundary_friction))

        self.model = builder.finalize()
        self.model.set_gravity((0.0, -9.81, 0.0))

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.solver = SolverWCSPH(self.model, sph_options_from_args(args))

        self.viewer.set_model(self.model)
        self.viewer.show_particles = True
        self.viewer.set_camera(pos=wp.vec3(2.0, 1.4, 3.0), pitch=-20.0, yaw=-145.0)

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        add_sph_timestep_arguments(parser, substeps=16)
        add_sph_block_dimension_arguments(parser, dim_x=16, dim_y=16, dim_z=16)
        add_sph_particle_arguments(parser, spacing=0.03, jitter=0.001)
        parser.add_argument("--height", type=_positive_float, default=0.75, help="Initial block height [m].")
        add_sph_solver_config_arguments(
            parser,
            kernel="poly6",
            xsph=0.05,
        )
        parser.add_argument(
            "--boundary-friction", type=_non_negative_float, default=0.05, help="Analytic boundary friction."
        )
        return parser

    def test_final(self):
        assert_sph_state_finite(self.state_0, "density")
        q = self.state_0.particle_q.numpy()

        assert np.min(q[:, self.model.up_axis]) >= -1.0e-4


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
