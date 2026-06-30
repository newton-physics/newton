# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""WCSPH fluid opening a hinged Newton flap."""

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import SolverWCSPH, sph

from ._config import (
    _non_negative_float,
    _positive_float,
    _positive_int,
    add_sph_analytic_tank_shapes,
    add_sph_particle_arguments,
    add_sph_solver_config_arguments,
    add_sph_tank_arguments,
    add_sph_timestep_arguments,
    assert_sph_state_finite,
    create_sph_visual_box_mesh,
    log_sph_fluid_points,
    sph_options_from_args,
    sph_particle_spacing_from_args,
    validate_sph_example_timestep_args,
)
from ._coupling import SPHRigidBodyCoupling


class Example:
    """Hinged flap driven by SPH collider impulses through ``State.body_f``."""

    def __init__(self, viewer, args):
        validate_sph_example_timestep_args(args)
        if args.initial_angle < 0.0 or args.initial_angle > args.joint_limit:
            raise ValueError("initial-angle must lie between zero and joint-limit")
        self.fps = args.fps
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = args.substeps
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.viewer = viewer

        spacing, radius, smoothing_length = sph_particle_spacing_from_args(args)
        self.particle_spacing = spacing
        self.tank_half_length = 0.5 * args.tank_length
        self.tank_half_width = 0.5 * args.tank_width
        self.wall_height = args.wall_height
        self.hinge_x = args.hinge_x
        material = sph.SPHMaterial(
            rest_density=args.rest_density,
            sound_speed=args.sound_speed,
            viscosity=args.viscosity,
            smoothing_length=smoothing_length,
        )

        fluid_builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=args.gravity)
        fluid_origin_z = -0.5 * args.dim_z * spacing
        self.fluid_indices = np.asarray(
            list(
                sph.add_sph_particle_grid(
                    fluid_builder,
                    pos=wp.vec3(args.fluid_x, args.fluid_height, fluid_origin_z),
                    rot=wp.quat_identity(),
                    vel=wp.vec3(args.fluid_velocity, 0.0, 0.0),
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

        self.fluid_model = fluid_builder.finalize()

        mechanism_builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=0.0)
        add_sph_analytic_tank_shapes(
            mechanism_builder,
            tank_length=args.tank_length,
            tank_width=args.tank_width,
            wall_height=args.wall_height,
            wall_thickness=args.wall_thickness,
            boundary_friction=args.boundary_friction,
        )
        shape_cfg = newton.ModelBuilder.ShapeConfig(density=args.flap_density, mu=args.collider_friction)
        shape_cfg.margin = args.collider_margin
        shape_cfg.is_visible = False
        self.flap_body = mechanism_builder.add_link(label="sph_hinged_flap")
        mechanism_builder.add_shape_box(
            self.flap_body,
            hx=args.flap_half_length,
            hy=args.flap_half_height,
            hz=args.flap_half_width,
            cfg=shape_cfg,
            color=(0.85, 0.18, 0.08),
        )
        self.flap_joint = mechanism_builder.add_joint_revolute(
            parent=-1,
            child=self.flap_body,
            axis=wp.vec3(0.0, 0.0, 1.0),
            parent_xform=wp.transform(wp.vec3(args.hinge_x, args.hinge_y, 0.0), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(-args.flap_half_length, 0.0, 0.0), wp.quat_identity()),
            damping=args.joint_damping,
            limit_lower=0.0,
            limit_upper=args.joint_limit,
            limit_ke=args.joint_limit_stiffness,
            limit_kd=args.joint_limit_damping,
            friction=args.joint_friction,
            label="sph_flap_hinge",
        )
        mechanism_builder.add_articulation([self.flap_joint], label="sph_hinged_flap_articulation")
        self.model = mechanism_builder.finalize()
        joint_q = self.model.joint_q.numpy()
        joint_q[0] = args.initial_angle
        self.model.joint_q.assign(joint_q)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.state_0.joint_q.assign(self.model.joint_q)
        self.state_1.joint_q.assign(self.model.joint_q)
        self.control = self.model.control()
        self.contacts = self.model.contacts()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_1)

        self.fluid_state_0 = self.fluid_model.state()
        self.fluid_state_1 = self.fluid_model.state()

        self.solver = newton.solvers.SolverMuJoCo(self.model, use_mujoco_contacts=False, njmax=100)
        self.sph_solver = SolverWCSPH(self.fluid_model, sph_options_from_args(args))
        self.coupling = SPHRigidBodyCoupling(
            self.model,
            self.sph_solver,
            self.state_0,
            (self.fluid_state_0, self.fluid_state_1),
            self.sim_dt,
        )
        self.initial_joint_q = self.state_0.joint_q.numpy().copy()
        self.fluid_render_radius_scale = 0.55
        self.flap_visual_mesh = create_sph_visual_box_mesh(
            (args.flap_half_length, args.flap_half_height, args.flap_half_width)
        )
        self.flap_visual_color = wp.array([wp.vec3(0.85, 0.18, 0.08)], dtype=wp.vec3)
        self.flap_visual_material = wp.array([wp.vec4(0.65, 0.05, 0.0, 0.0)], dtype=wp.vec4)

        self.viewer.set_model(self.model)
        self.viewer.show_particles = True
        self.viewer.set_camera(pos=wp.vec3(0.40, 0.32, 0.88), pitch=-18.0, yaw=-90.0)

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        add_sph_timestep_arguments(parser, fps=60.0, substeps=48)
        add_sph_particle_arguments(parser, spacing=0.0175)
        add_sph_tank_arguments(parser, tank_length=0.72, tank_width=0.50, wall_height=0.32, fluid_offset_y=0.08)
        add_sph_solver_config_arguments(
            parser,
            viscosity=0.006,
            xsph=0.05,
            boundary_friction=0.05,
        )
        parser.add_argument("--dim-x", type=_positive_int, default=16, help="Fluid particle count along X.")
        parser.add_argument("--dim-y", type=_positive_int, default=12, help="Fluid particle count along Y.")
        parser.add_argument("--dim-z", type=_positive_int, default=20, help="Fluid particle count along Z.")
        parser.add_argument("--fluid-x", type=float, default=-0.32, help="Initial fluid block X position [m].")
        parser.add_argument("--fluid-height", type=_positive_float, default=0.035, help="Fluid block base height [m].")
        parser.add_argument("--fluid-velocity", type=float, default=0.0, help="Initial fluid velocity along +X [m/s].")
        parser.add_argument("--hinge-x", type=float, default=-0.03, help="World-space hinge X position [m].")
        parser.add_argument("--hinge-y", type=_positive_float, default=0.035, help="World-space hinge Y position [m].")
        parser.add_argument("--initial-angle", type=float, default=1.5, help="Initial hinge angle [rad].")
        parser.add_argument("--flap-half-length", type=_positive_float, default=0.12, help="Flap half length [m].")
        parser.add_argument("--flap-half-height", type=_positive_float, default=0.015, help="Flap half height [m].")
        parser.add_argument("--flap-half-width", type=_positive_float, default=0.24, help="Flap half width [m].")
        parser.add_argument("--flap-density", type=_positive_float, default=160.0, help="Flap density [kg/m^3].")
        parser.add_argument("--joint-limit", type=_positive_float, default=1.5708, help="Hinge angular limit [rad].")
        parser.add_argument("--joint-damping", type=_non_negative_float, default=0.3, help="Passive hinge damping.")
        parser.add_argument(
            "--joint-limit-stiffness", type=_positive_float, default=1200.0, help="Joint limit stiffness."
        )
        parser.add_argument("--joint-limit-damping", type=_positive_float, default=60.0, help="Joint limit damping.")
        parser.add_argument("--joint-friction", type=_non_negative_float, default=0.03, help="Hinge friction.")
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
            self.coupling.apply_forces(self.state_0)
            self.coupling.save_applied_forces(self.state_0)

            self.viewer.apply_forces(self.state_0)
            self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

            self.coupling.update_fluid_state(self.state_0, self.fluid_state_0)
            self.sph_solver.step(self.fluid_state_0, self.fluid_state_1, control=None, contacts=None, dt=self.sim_dt)
            self.fluid_state_0, self.fluid_state_1 = self.fluid_state_1, self.fluid_state_0

            self.coupling.collect_impulses(self.fluid_state_0)

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        show_particles = self.viewer.show_particles
        self.viewer.begin_frame(self.sim_time)
        self.viewer.show_particles = False
        self.viewer.log_state(self.state_0)
        self.viewer.show_particles = show_particles
        log_sph_fluid_points(
            self.viewer,
            self.fluid_state_0,
            self.fluid_model,
            self.fluid_indices,
            radius_scale=self.fluid_render_radius_scale,
            hidden=not self.viewer.show_particles,
        )
        self.viewer.log_shapes(
            "/sph_hinged_flap_visual",
            newton.GeoType.MESH,
            (1.0, 1.0, 1.0),
            self.state_0.body_q[self.flap_body : self.flap_body + 1],
            self.flap_visual_color,
            self.flap_visual_material,
            geo_src=self.flap_visual_mesh,
        )
        self.viewer.end_frame()

    def test_final(self):
        assert_sph_state_finite(self.fluid_state_0, "density", "pressure")
        joint_q = self.state_0.joint_q.numpy()
        fluid_q = self.fluid_state_0.particle_q.numpy()[self.fluid_indices]
        assert np.isfinite(fluid_q).all()
        assert np.isfinite(self.state_0.body_q.numpy()).all()
        assert np.isfinite(joint_q).all()
        assert np.max(np.abs(joint_q - self.initial_joint_q)) < 2.0
        assert self.sph_solver.collider_body_index.numpy().tolist() == [-1, self.flap_body]
        assert self.coupling.max_collider_impulse_norm > 0.0
        assert joint_q[0] < self.initial_joint_q[0] - 0.5
        assert joint_q[0] >= -0.1

        tolerance = self.particle_spacing
        assert np.min(fluid_q[:, 0]) >= -self.tank_half_length - tolerance
        assert np.max(fluid_q[:, 0]) <= self.tank_half_length + tolerance
        assert np.min(fluid_q[:, 1]) >= -tolerance
        assert np.max(fluid_q[:, 1]) <= self.wall_height + tolerance
        assert np.min(fluid_q[:, 2]) >= -self.tank_half_width - tolerance
        assert np.max(fluid_q[:, 2]) <= self.tank_half_width + tolerance
        assert np.count_nonzero(fluid_q[:, 0] > self.hinge_x) >= fluid_q.shape[0] // 10


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
