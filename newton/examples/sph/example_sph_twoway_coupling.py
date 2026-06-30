# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example SPH 2-Way Coupling
#
# A compact WCSPH fluid block interacts with a rigid sphere. The rigid body is
# integrated by MuJoCo; SolverWCSPH owns the fluid step and analytic collider
# projection. The coupling bridge converts SPH collider impulses into
# State.body_f forces using the same high-level exchange pattern as the implicit
# MPM two-way example.
#
# Command: python -m newton.examples sph_twoway_coupling
#
###########################################################################

from __future__ import annotations

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import SolverWCSPH, sph

from ._config import (
    _non_negative_float,
    _positive_float,
    _positive_int,
    add_sph_particle_arguments,
    add_sph_particle_grid_filtered,
    add_sph_solver_config_arguments,
    add_sph_tank_arguments,
    add_sph_timestep_arguments,
    assert_sph_state_finite,
    log_sph_fluid_points,
    sph_options_from_args,
    sph_particle_spacing_from_args,
    validate_sph_example_timestep_args,
)
from ._coupling import SPHRigidBodyCoupling


class Example:
    def __init__(self, viewer, args):
        validate_sph_example_timestep_args(args)
        self.fps = args.fps
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = args.substeps
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer

        spacing, radius, smoothing_length = sph_particle_spacing_from_args(args)
        material = sph.SPHMaterial(
            rest_density=args.rest_density,
            sound_speed=args.sound_speed,
            viscosity=args.viscosity,
            smoothing_length=smoothing_length,
        )
        self.wall_height = args.wall_height
        self.float_radius = args.float_radius

        builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=args.gravity)
        self._emit_rigid_body(builder, args)
        self._emit_rigid_tank(builder, args)
        builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=args.collider_friction))

        fluid_builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=args.gravity)
        self._emit_particles(fluid_builder, args, spacing, radius, material)

        self.model = builder.finalize()
        self.fluid_model = fluid_builder.finalize()

        self.sph_solver = SolverWCSPH(
            self.fluid_model,
            sph_options_from_args(args),
        )

        self.solver = newton.solvers.SolverMuJoCo(self.model, use_mujoco_contacts=False, njmax=100)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.fluid_state_0 = self.fluid_model.state()
        self.fluid_state_1 = self.fluid_model.state()

        self.control = self.model.control()
        self.contacts = self.model.contacts()

        # Keep the state valid if this example is switched to another rigid-body solver.
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self.coupling = SPHRigidBodyCoupling(
            self.model,
            self.sph_solver,
            self.state_0,
            (self.fluid_state_0, self.fluid_state_1),
            self.sim_dt,
        )

        self.initial_float_position = self.state_0.body_q.numpy()[self.float_body, 0:3].copy()

        self.viewer.set_model(self.model)
        if isinstance(self.viewer, newton.viewer.ViewerGL):
            self.viewer.register_ui_callback(self.render_ui, position="side")
        self.viewer.show_particles = True
        self.viewer.set_camera(pos=wp.vec3(0.9, 0.50, 0.82), pitch=-24.0, yaw=-135.0)
        self.show_impulses = False
        self.fluid_render_radius_scale = 0.55

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        add_sph_timestep_arguments(parser, fps=60.0, substeps=24)
        add_sph_particle_arguments(parser, spacing=0.028)
        add_sph_tank_arguments(parser, tank_length=0.96, tank_width=0.64, wall_height=0.58, fluid_offset_y=0.07)
        add_sph_solver_config_arguments(
            parser,
            sound_speed=18.0,
            viscosity=0.004,
            xsph=0.03,
            boundary_friction=0.05,
        )
        parser.add_argument("--dim-x", type=_positive_int, default=30, help="Fluid particle count along X.")
        parser.add_argument("--dim-y", type=_positive_int, default=14, help="Fluid particle count along Y.")
        parser.add_argument("--dim-z", type=_positive_int, default=20, help="Fluid particle count along Z.")
        parser.add_argument("--fluid-height", type=_positive_float, default=0.055, help="Fluid block base height [m].")
        parser.add_argument("--fluid-velocity", type=float, default=0.0, help="Initial fluid velocity along +X [m/s].")
        parser.add_argument("--float-x", type=float, default=0.02, help="Initial rigid float X position [m].")
        parser.add_argument("--float-y", type=_positive_float, default=0.50, help="Initial rigid float Y position [m].")
        parser.add_argument("--float-radius", type=_positive_float, default=0.095, help="Rigid float body radius [m].")
        parser.add_argument("--float-mass", type=_positive_float, default=1.80, help="Rigid float mass [kg].")
        parser.add_argument(
            "--float-inertia", type=_positive_float, default=0.0065, help="Rigid float diagonal inertia."
        )
        parser.add_argument(
            "--collider-margin", type=_non_negative_float, default=0.0, help="Rigid collider margin [m]."
        )
        parser.add_argument(
            "--collider-friction", type=_non_negative_float, default=0.05, help="Rigid collider friction."
        )
        return parser

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()

            # apply SPH forces collected from the previous fluid substep
            self.coupling.apply_forces(self.state_0)

            # save applied force to subtract from collider velocities in the SPH step
            self.coupling.save_applied_forces(self.state_0)

            # apply external viewer forces to the model
            self.viewer.apply_forces(self.state_0)

            self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            self.state_0, self.state_1 = self.state_1, self.state_0

            # subtract previously applied rigid-body response from collider velocities
            self.coupling.update_fluid_state(self.state_0, self.fluid_state_0)
            self.sph_solver.step(self.fluid_state_0, self.fluid_state_1, control=None, contacts=None, dt=self.sim_dt)

            self.fluid_state_0, self.fluid_state_1 = self.fluid_state_1, self.fluid_state_0

            # save impulses to apply back to rigid bodies on the next substep
            self.coupling.collect_impulses(self.fluid_state_0)

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt

    def test_final(self):
        assert_sph_state_finite(self.fluid_state_0, "density", "pressure")
        q = self.fluid_state_0.particle_q.numpy()
        body_q = self.state_0.body_q.numpy()
        assert np.isfinite(q).all()
        assert np.isfinite(body_q).all()
        assert self.coupling.max_collider_impulse_norm > 0.0
        assert np.linalg.norm(body_q[self.float_body, 0:3] - self.initial_float_position) < 2.0
        assert np.max(q[self.fluid_indices, 1]) < self.wall_height + self.float_radius

    def render(self):
        show_particles = self.viewer.show_particles
        self.viewer.begin_frame(self.sim_time)
        self.viewer.show_particles = False
        self.viewer.log_state(self.state_0)
        self.viewer.show_particles = show_particles
        self.viewer.log_contacts(self.contacts, self.state_0)

        log_sph_fluid_points(
            self.viewer,
            self.fluid_state_0,
            self.fluid_model,
            self.fluid_indices,
            radius_scale=self.fluid_render_radius_scale,
            hidden=not self.viewer.show_particles,
        )

        if self.show_impulses:
            impulses, pos, _cid = self.sph_solver.collect_collider_impulses(self.fluid_state_0)
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

    def _emit_rigid_body(self, builder: newton.ModelBuilder, args):
        shape_cfg = newton.ModelBuilder.ShapeConfig(density=0.0, mu=args.collider_friction, kd=400.0)
        shape_cfg.margin = args.collider_margin
        shape_cfg.has_particle_collision = True

        self.float_body = builder.add_body(
            xform=wp.transform(
                wp.vec3(args.float_x, args.float_y, 0.0),
                wp.quat_identity(),
            ),
            mass=args.float_mass,
            inertia=wp.diag(wp.vec3(args.float_inertia)),
            label="sph_coupled_float",
        )
        builder.add_shape_sphere(
            self.float_body,
            radius=args.float_radius,
            cfg=shape_cfg,
            color=(0.95, 0.76, 0.18),
        )

    def _emit_rigid_tank(self, builder: newton.ModelBuilder, args):
        shape_cfg = newton.ModelBuilder.ShapeConfig(density=0.0, ke=900.0, kd=500.0, mu=args.collider_friction)
        shape_cfg.has_particle_collision = True
        shape_cfg.is_visible = False

        half_length = 0.5 * args.tank_length
        half_width = 0.5 * args.tank_width
        half_height = 0.5 * args.wall_height
        half_wall = 0.5 * args.wall_thickness
        wall_specs = (
            (wp.vec3(-half_length, half_height, 0.0), half_wall, half_height, half_width),
            (wp.vec3(half_length, half_height, 0.0), half_wall, half_height, half_width),
            (wp.vec3(0.0, half_height, -half_width), half_length, half_height, half_wall),
            (wp.vec3(0.0, half_height, half_width), half_length, half_height, half_wall),
        )
        for center, hx, hy, hz in wall_specs:
            builder.add_shape_box(
                body=-1,
                xform=wp.transform(center, wp.quat_identity()),
                hx=hx,
                hy=hy,
                hz=hz,
                cfg=shape_cfg,
            )

    def _emit_particles(
        self,
        fluid_builder: newton.ModelBuilder,
        args,
        spacing: float,
        radius: float,
        material: sph.SPHMaterial,
    ):
        fluid_origin_x = -0.5 * args.dim_x * spacing
        fluid_origin_z = -0.5 * args.dim_z * spacing
        float_center = np.array((args.float_x, args.float_y, 0.0), dtype=np.float64)
        exclusion_radius = args.float_radius + spacing
        self.fluid_indices = add_sph_particle_grid_filtered(
            fluid_builder,
            pos=(fluid_origin_x, args.fluid_height, fluid_origin_z),
            vel=(args.fluid_velocity, 0.0, 0.0),
            dim_x=args.dim_x,
            dim_y=args.dim_y,
            dim_z=args.dim_z,
            cell_x=spacing,
            cell_y=spacing,
            cell_z=spacing,
            material=material,
            is_excluded=lambda point: np.linalg.norm(point - float_center) < exclusion_radius,
            jitter=args.jitter,
            radius_mean=radius,
        )


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
