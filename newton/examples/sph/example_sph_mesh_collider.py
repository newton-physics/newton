# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""WCSPH fluid sliding over a Newton triangle-mesh collider."""

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import SolverWCSPH, sph

from ._config import (
    SPHExampleBase,
    _finite_float,
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


def _make_ramp_mesh(length: float, width: float, height: float) -> newton.Mesh:
    half_length = 0.5 * length
    half_width = 0.5 * width
    thickness = 0.025
    vertices = np.array(
        (
            (-half_length, 0.0, -half_width),
            (half_length, height, -half_width),
            (-half_length, 0.0, half_width),
            (half_length, height, half_width),
            (-half_length, -thickness, -half_width),
            (half_length, height - thickness, -half_width),
            (-half_length, -thickness, half_width),
            (half_length, height - thickness, half_width),
        ),
        dtype=np.float32,
    )
    indices = np.array(
        (
            0,
            2,
            1,
            1,
            2,
            3,
            4,
            5,
            6,
            5,
            7,
            6,
            0,
            1,
            4,
            1,
            5,
            4,
            2,
            6,
            3,
            3,
            6,
            7,
            0,
            4,
            2,
            2,
            4,
            6,
            1,
            3,
            5,
            3,
            7,
            5,
        ),
        dtype=np.int32,
    )
    return newton.Mesh(vertices, indices, compute_inertia=False)


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
                    pos=wp.vec3(args.fluid_x, args.height, -0.5 * args.dim_z * spacing),
                    rot=wp.quat_identity(),
                    vel=wp.vec3(args.fluid_velocity, 0.0, 0.0),
                    dim_x=args.dim_x,
                    dim_y=args.dim_y,
                    dim_z=args.dim_z,
                    cell_x=spacing,
                    cell_y=spacing,
                    cell_z=spacing,
                    material=material,
                    role=sph.SPHRole.FLUID,
                    jitter=args.jitter,
                    radius_mean=radius,
                )
            ),
            dtype=np.int64,
        )

        shape_cfg = newton.ModelBuilder.ShapeConfig(density=0.0, mu=args.boundary_friction)
        shape_cfg.margin = args.collider_margin
        builder.add_shape_mesh(
            body=-1,
            mesh=_make_ramp_mesh(args.ramp_length, args.ramp_width, args.ramp_height),
            cfg=shape_cfg,
            color=(0.82, 0.60, 0.36),
        )

        self.model = builder.finalize()
        self.model.set_gravity((0.0, args.gravity, 0.0))

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.solver = SolverWCSPH(self.model, sph_options_from_args(args))
        self.saw_boundary_impulse = False

        self.viewer.set_model(self.model)
        self.viewer.show_particles = True
        self.viewer.set_camera(pos=wp.vec3(0.75, 0.38, 0.72), pitch=-22.0, yaw=-135.0)

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        add_sph_timestep_arguments(parser, substeps=12)
        add_sph_block_dimension_arguments(parser, dim_x=6, dim_y=3, dim_z=5)
        add_sph_particle_arguments(parser, spacing=0.035, jitter=0.001)
        parser.add_argument("--height", type=_positive_float, default=0.065, help="Initial block height [m].")
        add_sph_solver_config_arguments(
            parser,
            viscosity=0.004,
            xsph=0.04,
            boundary_friction=0.08,
        )
        parser.add_argument(
            "--gravity", type=_finite_float, default=-9.81, help="Vertical gravity acceleration [m/s^2]."
        )
        parser.add_argument("--fluid-x", type=float, default=-0.18, help="Initial fluid block X position [m].")
        parser.add_argument("--fluid-velocity", type=float, default=0.45, help="Initial fluid velocity along +X [m/s].")
        parser.add_argument("--ramp-length", type=_positive_float, default=0.46, help="Ramp mesh length [m].")
        parser.add_argument("--ramp-width", type=_positive_float, default=0.42, help="Ramp mesh width [m].")
        parser.add_argument("--ramp-height", type=_positive_float, default=0.085, help="Ramp mesh height rise [m].")
        parser.add_argument(
            "--collider-margin", type=_non_negative_float, default=0.01, help="Mesh collider margin [m]."
        )
        return parser

    def after_substep(self):
        impulses = self.state_0.sph.boundary_impulse.numpy()[self.fluid_indices]
        self.saw_boundary_impulse |= bool(np.any(np.linalg.norm(impulses, axis=1) > 1.0e-8))

    def test_final(self):
        assert_sph_state_finite(self.state_0, "density", "pressure")
        assert self.saw_boundary_impulse


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
