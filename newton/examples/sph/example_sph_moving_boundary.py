# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Kinematic Newton paddle moving through WCSPH fluid."""

import math

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import SolverWCSPH, sph

from ._config import (
    SPHExampleBase,
    _non_negative_float,
    _positive_float,
    add_sph_analytic_tank_shapes,
    add_sph_block_dimension_arguments,
    add_sph_particle_arguments,
    add_sph_solver_config_arguments,
    add_sph_tank_arguments,
    add_sph_timestep_arguments,
    assert_sph_state_finite,
    sph_options_from_args,
    sph_particle_spacing_from_args,
)


class Example(SPHExampleBase):
    advance_time_per_substep = True

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

        fluid_origin_x = -0.5 * args.dim_x * spacing + 0.03
        fluid_origin_z = -0.5 * args.dim_z * spacing
        self.fluid_indices = np.asarray(
            list(
                sph.add_sph_particle_grid(
                    builder,
                    pos=wp.vec3(fluid_origin_x, args.height, fluid_origin_z),
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

        self.paddle_base = np.array([args.paddle_x, args.paddle_y, 0.0], dtype=np.float32)
        self.paddle_body = builder.add_body(
            xform=self._paddle_transform(0.0, args),
            mass=args.paddle_mass,
            inertia=wp.diag(wp.vec3(args.paddle_inertia)),
            label="sph_moving_paddle",
            lock_inertia=True,
            is_kinematic=True,
        )
        paddle_cfg = newton.ModelBuilder.ShapeConfig(density=0.0, mu=args.boundary_friction)
        paddle_cfg.has_shape_collision = False
        paddle_cfg.has_particle_collision = True
        builder.add_shape_box(
            body=self.paddle_body,
            hx=args.paddle_thickness,
            hy=args.paddle_height,
            hz=args.paddle_width,
            cfg=paddle_cfg,
            color=(0.85, 0.35, 0.25),
        )

        self.model = builder.finalize()
        self.model.set_gravity((0.0, args.gravity, 0.0))
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self._apply_paddle_motion(self.state_0, 0.0, args)
        self._apply_paddle_motion(self.state_1, 0.0, args)
        self.initial_fluid_q = self.state_0.particle_q.numpy()[self.fluid_indices].copy()
        self.max_paddle_impulse_norm = 0.0

        self.args = args
        self.solver = SolverWCSPH(self.model, sph_options_from_args(args))
        self.solver.setup_collider(collider_body_ids=[-1, self.paddle_body])

        self.viewer.set_model(self.model)
        self.viewer.show_particles = True
        self.viewer.set_camera(pos=wp.vec3(1.2, 0.8, 1.4), pitch=-20.0, yaw=-135.0)

    def _paddle_transform(self, time: float, args):
        phase = 2.0 * math.pi * args.frequency * time
        x = float(self.paddle_base[0] + args.amplitude * math.sin(phase))
        y = float(self.paddle_base[1])
        angle = args.tilt_amplitude * math.sin(phase + 0.5 * math.pi)
        return wp.transform(
            p=wp.vec3(x, y, 0.0),
            q=wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), angle),
        )

    def _apply_paddle_motion(self, state, time: float, args):
        if state.body_q is None or state.body_qd is None:
            return

        phase = 2.0 * math.pi * args.frequency * time
        velocity_x = args.amplitude * 2.0 * math.pi * args.frequency * math.cos(phase)
        angular_z = args.tilt_amplitude * 2.0 * math.pi * args.frequency * math.cos(phase + 0.5 * math.pi)

        body_q = state.body_q.numpy()
        transform = self._paddle_transform(time, args)
        body_q[self.paddle_body] = np.array(transform, dtype=np.float32)
        state.body_q.assign(body_q)

        body_qd = state.body_qd.numpy()
        body_qd[self.paddle_body] = np.array([velocity_x, 0.0, 0.0, 0.0, 0.0, angular_z], dtype=np.float32)
        state.body_qd.assign(body_qd)

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        add_sph_timestep_arguments(parser, substeps=24)
        add_sph_block_dimension_arguments(parser, dim_x=14, dim_y=4, dim_z=10, label="Fluid particle")
        add_sph_particle_arguments(parser, spacing=0.035)
        parser.add_argument("--height", type=_positive_float, default=0.08, help="Fluid block base height [m].")
        add_sph_tank_arguments(parser, tank_width=0.52, wall_height=0.34, fluid_offset_y=0.08)
        add_sph_solver_config_arguments(
            parser,
            viscosity=0.006,
            xsph=0.04,
            boundary_friction=0.05,
        )
        parser.add_argument("--paddle-x", type=float, default=-0.27, help="Mean paddle X position [m].")
        parser.add_argument("--paddle-y", type=float, default=0.13, help="Mean paddle Y position [m].")
        parser.add_argument(
            "--paddle-thickness", type=_positive_float, default=0.018, help="Paddle half-thickness [m]."
        )
        parser.add_argument("--paddle-height", type=_positive_float, default=0.075, help="Paddle half-height [m].")
        parser.add_argument("--paddle-width", type=_positive_float, default=0.24, help="Paddle half-width [m].")
        parser.add_argument("--paddle-mass", type=_positive_float, default=1.0, help="Diagnostic paddle mass [kg].")
        parser.add_argument(
            "--paddle-inertia", type=_positive_float, default=0.01, help="Diagonal paddle inertia [kg m^2]."
        )
        parser.add_argument(
            "--amplitude", type=_positive_float, default=0.015, help="Paddle horizontal motion amplitude [m]."
        )
        parser.add_argument("--frequency", type=_positive_float, default=0.9, help="Paddle oscillation frequency [Hz].")
        parser.add_argument(
            "--tilt-amplitude", type=_non_negative_float, default=0.05, help="Paddle tilt amplitude [rad]."
        )
        return parser

    def before_substep(self):
        self._apply_paddle_motion(self.state_0, self.sim_time, self.args)
        self._apply_paddle_motion(self.state_1, self.sim_time, self.args)

    def after_substep(self):
        impulses, _, _ = self.solver.collect_collider_impulses(self.state_0)
        if impulses.shape[0] == 0:
            return
        impulse_norm = float(np.max(np.linalg.norm(impulses.numpy(), axis=1)))
        self.max_paddle_impulse_norm = max(self.max_paddle_impulse_norm, impulse_norm)

    def test_final(self):
        assert_sph_state_finite(self.state_0, "density", "pressure")
        q = self.state_0.particle_q.numpy()

        assert self.solver.collider_body_index.numpy().tolist() == [-1, self.paddle_body]
        assert np.max(np.linalg.norm(q[self.fluid_indices] - self.initial_fluid_q, axis=1)) > 1.0e-5
        assert np.min(q[self.fluid_indices, self.model.up_axis]) >= -1.0e-4


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
