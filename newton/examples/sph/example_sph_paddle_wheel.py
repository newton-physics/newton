# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Analytic rigid paddle wheel stirring WCSPH fluid in a tank."""

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
    _positive_int,
    add_sph_analytic_tank_shapes,
    add_sph_block_dimension_arguments,
    add_sph_particle_arguments,
    add_sph_solver_config_arguments,
    add_sph_tank_arguments,
    add_sph_timestep_arguments,
    assert_sph_state_finite,
    create_sph_visual_box_mesh,
    log_sph_fluid_points,
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

        fluid_origin_x = -0.5 * args.dim_x * spacing + 0.06
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

        self.wheel_center = np.array([args.wheel_x, args.wheel_y, 0.0], dtype=np.float32)
        self.wheel_body = builder.add_body(
            xform=self._wheel_transform(0.0, args),
            mass=args.wheel_mass,
            inertia=wp.diag(wp.vec3(args.wheel_inertia)),
            label="sph_paddle_wheel",
            lock_inertia=True,
            is_kinematic=True,
        )
        paddle_cfg = newton.ModelBuilder.ShapeConfig(density=0.0, mu=args.boundary_friction)
        paddle_cfg.has_shape_collision = False
        paddle_cfg.has_particle_collision = True
        paddle_cfg.margin = args.wheel_margin
        self.blade_local_xforms = []
        blade_center = args.hub_radius + 0.5 * args.blade_length
        for blade in range(args.blade_count):
            angle = 2.0 * math.pi * float(blade) / float(args.blade_count)
            c, s = math.cos(angle), math.sin(angle)
            blade_xform = wp.transform(
                wp.vec3(blade_center * c, blade_center * s, 0.0),
                wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), angle),
            )
            self.blade_local_xforms.append(blade_xform)
            builder.add_shape_box(
                body=self.wheel_body,
                xform=blade_xform,
                hx=0.5 * args.blade_length,
                hy=0.5 * args.blade_thickness,
                hz=0.5 * args.blade_width,
                cfg=paddle_cfg,
                color=(1.0, 0.46, 0.05),
                label=f"sph_paddle_wheel_blade_{blade}",
            )

        if args.hub_radius > 0.0:
            builder.add_shape_cylinder(
                body=self.wheel_body,
                radius=args.hub_radius,
                half_height=0.5 * args.blade_width,
                cfg=paddle_cfg,
                color=(0.65, 0.65, 0.68),
                label="sph_paddle_wheel_hub",
            )
        self.model = builder.finalize()
        self.model.set_gravity((0.0, args.gravity, 0.0))
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self._apply_wheel_motion(self.state_0, 0.0, args)
        self._apply_wheel_motion(self.state_1, 0.0, args)
        self.initial_fluid_q = self.state_0.particle_q.numpy()[self.fluid_indices].copy()
        self.max_wheel_impulse_norm = 0.0
        self.fluid_render_radius_scale = 0.42
        self.blade_visual_scale = (0.5 * args.blade_length, 0.5 * args.blade_thickness, 0.5 * args.blade_width)
        self.blade_visual_mesh = create_sph_visual_box_mesh(self.blade_visual_scale)
        self.blade_visual_color = wp.array([wp.vec3(1.0, 0.46, 0.05)], dtype=wp.vec3)
        self.blade_visual_material = wp.array([wp.vec4(0.65, 0.05, 0.0, 0.0)], dtype=wp.vec4)
        self.hub_visual_color = wp.array([wp.vec3(0.65, 0.65, 0.68)], dtype=wp.vec3)
        self.hub_visual_material = wp.array([wp.vec4(0.55, 0.05, 0.0, 0.0)], dtype=wp.vec4)

        self.args = args
        self.solver = SolverWCSPH(
            self.model,
            sph_options_from_args(args),
        )
        self.solver.setup_collider(collider_body_ids=[-1, self.wheel_body])

        self.viewer.set_model(self.model)
        self.viewer.show_particles = True
        self.viewer.set_camera(pos=wp.vec3(0.34, 0.42, 1.05), pitch=-18.0, yaw=-90.0)

    def _wheel_transform(self, time: float, args):
        angle = args.angular_speed * time + args.initial_angle
        return wp.transform(
            p=wp.vec3(self.wheel_center[0], self.wheel_center[1], self.wheel_center[2]),
            q=wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), angle),
        )

    def _apply_wheel_motion(self, state, time: float, args):
        if state.body_q is None or state.body_qd is None:
            return

        body_q = state.body_q.numpy()
        body_q[self.wheel_body] = np.array(self._wheel_transform(time, args), dtype=np.float32)
        state.body_q.assign(body_q)

        body_qd = state.body_qd.numpy()
        body_qd[self.wheel_body] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, args.angular_speed], dtype=np.float32)
        state.body_qd.assign(body_qd)

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        add_sph_timestep_arguments(parser, substeps=32)
        add_sph_block_dimension_arguments(parser, dim_x=20, dim_y=6, dim_z=12, label="Fluid particle")
        add_sph_particle_arguments(parser, spacing=0.03)
        parser.add_argument("--height", type=_positive_float, default=0.045, help="Fluid block base height [m].")
        add_sph_tank_arguments(parser, tank_width=0.58, wall_height=0.44, fluid_offset_y=0.06)
        add_sph_solver_config_arguments(
            parser,
            viscosity=0.006,
            xsph=0.04,
            boundary_friction=0.05,
        )
        parser.add_argument("--wheel-x", type=float, default=0.00, help="Wheel center X position [m].")
        parser.add_argument("--wheel-y", type=_positive_float, default=0.18, help="Wheel center Y position [m].")
        parser.add_argument("--wheel-margin", type=_non_negative_float, default=0.0, help="Wheel collider margin [m].")
        parser.add_argument("--blade-count", type=_positive_int, default=4, help="Number of paddle blades.")
        parser.add_argument("--blade-length", type=_positive_float, default=0.11, help="Paddle blade length [m].")
        parser.add_argument(
            "--blade-thickness", type=_positive_float, default=0.04, help="Paddle blade collider thickness [m]."
        )
        parser.add_argument("--blade-width", type=_positive_float, default=0.26, help="Paddle blade width [m].")
        parser.add_argument("--hub-radius", type=_non_negative_float, default=0.035, help="Wheel hub radius [m].")
        parser.add_argument("--wheel-mass", type=_positive_float, default=1.0, help="Diagnostic wheel mass [kg].")
        parser.add_argument(
            "--wheel-inertia", type=_positive_float, default=0.01, help="Diagonal wheel inertia [kg m^2]."
        )
        parser.add_argument("--angular-speed", type=float, default=2.4, help="Wheel angular speed [rad/s].")
        parser.add_argument("--initial-angle", type=float, default=0.35, help="Initial wheel angle [rad].")
        return parser

    def before_substep(self):
        self._apply_wheel_motion(self.state_0, self.sim_time, self.args)
        self._apply_wheel_motion(self.state_1, self.sim_time, self.args)

    def after_substep(self):
        impulses, _, _ = self.solver.collect_collider_impulses(self.state_0)
        if impulses.shape[0] == 0:
            return
        impulse_norm = float(np.max(np.linalg.norm(impulses.numpy(), axis=1)))
        self.max_wheel_impulse_norm = max(self.max_wheel_impulse_norm, impulse_norm)

    def _blade_visual_xforms(self):
        wheel_xform = self._wheel_transform(self.sim_time, self.args)
        return wp.array(
            [wp.transform_multiply(wheel_xform, blade_xform) for blade_xform in self.blade_local_xforms],
            dtype=wp.transform,
        )

    def render(self):
        show_particles = self.viewer.show_particles
        self.viewer.begin_frame(self.sim_time)

        log_sph_fluid_points(
            self.viewer,
            self.state_0,
            self.model,
            self.fluid_indices,
            radius_scale=self.fluid_render_radius_scale,
            hidden=not show_particles,
        )
        self.viewer.log_shapes(
            "/sph_paddle_wheel_blades",
            newton.GeoType.MESH,
            (1.0, 1.0, 1.0),
            self._blade_visual_xforms(),
            self.blade_visual_color,
            self.blade_visual_material,
            geo_src=self.blade_visual_mesh,
        )
        if self.args.hub_radius > 0.0:
            self.viewer.log_shapes(
                "/sph_paddle_wheel_hub",
                newton.GeoType.CYLINDER,
                (self.args.hub_radius, 0.5 * self.args.blade_width),
                wp.array([self._wheel_transform(self.sim_time, self.args)], dtype=wp.transform),
                self.hub_visual_color,
                self.hub_visual_material,
            )
        self.viewer.log_points("/sph_tank_boundary", points=None, hidden=True)
        self.viewer.end_frame()

    def test_final(self):
        assert_sph_state_finite(self.state_0, "density", "pressure")
        q = self.state_0.particle_q.numpy()

        assert len(self.fluid_indices) == self.model.particle_count
        assert self.solver.collider_body_index.numpy().tolist() == [-1, self.wheel_body]
        assert self.max_wheel_impulse_norm > 1.0e-5
        assert np.max(np.linalg.norm(q[self.fluid_indices] - self.initial_fluid_q, axis=1)) > 0.03
        assert np.min(q[self.fluid_indices, self.model.up_axis]) >= -1.0e-3


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
