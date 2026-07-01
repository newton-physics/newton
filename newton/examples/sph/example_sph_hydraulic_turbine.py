# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""A WCSPH dam-break wave driving a freely rotating hydraulic turbine."""

import math

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
    add_sph_block_dimension_arguments,
    add_sph_particle_arguments,
    add_sph_solver_config_arguments,
    add_sph_tank_arguments,
    add_sph_timestep_arguments,
    assert_sph_state_finite,
    log_sph_fluid_points,
    sph_options_from_args,
)
from ._coupling import SPHRigidBodyCoupling


@wp.kernel
def _compose_blade_transforms(
    body: int,
    body_q: wp.array[wp.transform],
    blade_local_q: wp.array[wp.transform],
    blade_world_q: wp.array[wp.transform],
):
    blade = wp.tid()
    blade_world_q[blade] = wp.transform_multiply(body_q[body], blade_local_q[blade])


class Example:
    """Dam-break flow transferring angular momentum to a Newton articulation."""

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.fps = args.fps
        self.frame_dt = 1.0 / self.fps
        self.resolution_scale = args.resolution_scale
        self.sim_substeps = max(1, int(math.ceil(args.substeps * self.resolution_scale)))
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        spacing = args.spacing / self.resolution_scale
        radius = args.radius if args.radius > 0.0 else 0.5 * spacing
        smoothing_length = args.smoothing_length if args.smoothing_length > 0.0 else 2.0 * spacing
        acoustic_limit = 0.25 * smoothing_length / args.sound_speed
        if self.sim_dt > acoustic_limit:
            raise ValueError(
                f"SPH timestep {self.sim_dt:.3g} exceeds the acoustic stability limit {acoustic_limit:.3g}"
            )

        dim_x = max(1, int(round(args.dim_x * self.resolution_scale)))
        dim_y = max(1, int(round(args.dim_y * self.resolution_scale)))
        dim_z = max(1, int(round(args.dim_z * self.resolution_scale)))
        jitter = args.jitter / self.resolution_scale
        self.particle_spacing = spacing
        self.tank_half_length = 0.5 * args.tank_length
        self.tank_half_width = 0.5 * args.tank_width
        self.wall_height = args.wall_height
        self.wheel_x = args.wheel_x

        material = sph.SPHMaterial(
            rest_density=args.rest_density,
            sound_speed=args.sound_speed,
            viscosity=args.viscosity,
            smoothing_length=smoothing_length,
        )
        fluid_builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=args.gravity)
        fluid_origin_z = -0.5 * float(dim_z - 1) * spacing
        self.fluid_indices = np.asarray(
            list(
                sph.add_sph_particle_grid(
                    fluid_builder,
                    pos=wp.vec3(args.fluid_x, args.fluid_offset_y, fluid_origin_z),
                    vel=wp.vec3(args.fluid_velocity, 0.0, 0.0),
                    dim_x=dim_x,
                    dim_y=dim_y,
                    dim_z=dim_z,
                    cell_x=spacing,
                    cell_y=spacing,
                    cell_z=spacing,
                    material=material,
                    jitter=jitter,
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
            wall_order=("left", "right", "back", "front"),
        )
        ground_cfg = newton.ModelBuilder.ShapeConfig(density=0.0, mu=args.boundary_friction)
        ground_cfg.has_shape_collision = False
        ground_cfg.has_particle_collision = True
        ground_cfg.is_visible = False
        mechanism_builder.add_ground_plane(
            height=0.5 * args.wall_thickness,
            cfg=ground_cfg,
            label="sph_hydraulic_turbine_ground",
        )

        wheel_cfg = newton.ModelBuilder.ShapeConfig(density=args.wheel_density, mu=args.collider_friction)
        wheel_cfg.has_shape_collision = False
        wheel_cfg.has_particle_collision = True
        wheel_cfg.is_visible = False
        wheel_cfg.margin = args.collider_margin
        self.wheel_body = mechanism_builder.add_link(label="sph_hydraulic_turbine")
        blade_center = args.hub_radius + 0.5 * args.blade_length
        self.blade_local_q = []
        for blade in range(args.blade_count):
            angle = 2.0 * math.pi * float(blade) / float(args.blade_count)
            blade_q = wp.transform(
                wp.vec3(blade_center * math.cos(angle), blade_center * math.sin(angle), 0.0),
                wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), angle),
            )
            self.blade_local_q.append(blade_q)
            mechanism_builder.add_shape_box(
                self.wheel_body,
                xform=blade_q,
                hx=0.5 * args.blade_length,
                hy=0.5 * args.blade_thickness,
                hz=0.5 * args.blade_width,
                cfg=wheel_cfg,
                color=(0.95, 0.48, 0.08),
                label=f"sph_hydraulic_turbine_blade_{blade}",
            )
        mechanism_builder.add_shape_cylinder(
            self.wheel_body,
            radius=args.hub_radius,
            half_height=0.5 * args.blade_width,
            cfg=wheel_cfg,
            color=(0.24, 0.27, 0.31),
            label="sph_hydraulic_turbine_hub",
        )
        self.wheel_joint = mechanism_builder.add_joint_revolute(
            parent=-1,
            child=self.wheel_body,
            axis=wp.vec3(0.0, 0.0, 1.0),
            parent_xform=wp.transform(wp.vec3(args.wheel_x, args.wheel_y, 0.0), wp.quat_identity()),
            child_xform=wp.transform_identity(),
            damping=args.joint_damping,
            friction=args.joint_friction,
            label="sph_hydraulic_turbine_axle",
        )
        mechanism_builder.add_articulation([self.wheel_joint], label="sph_hydraulic_turbine_articulation")
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

        self.initial_joint_q = float(self.state_0.joint_q.numpy()[0])
        self.max_wheel_rotation = 0.0
        self.fluid_render_radius_scale = 0.46
        self._sph_render_points = None
        self.blade_local_q_wp = wp.array(self.blade_local_q, dtype=wp.transform, device=self.model.device)
        self.blade_world_q_wp = wp.empty_like(self.blade_local_q_wp)
        self.blade_visual_scale = (0.5 * args.blade_length, 0.5 * args.blade_thickness, 0.5 * args.blade_width)
        self.blade_visual_color = wp.array(
            [wp.vec3(1.0, 0.82, 0.16)] + [wp.vec3(0.95, 0.48, 0.08)] * (args.blade_count - 1),
            dtype=wp.vec3,
            device=self.model.device,
        )
        self.blade_visual_material = wp.array(
            [wp.vec4(0.76, 0.16, 0.0, 0.0)] + [wp.vec4(0.72, 0.18, 0.02, 0.0)] * (args.blade_count - 1),
            dtype=wp.vec4,
            device=self.model.device,
        )
        self.hub_visual_color = wp.array([wp.vec3(0.24, 0.27, 0.31)], dtype=wp.vec3, device=self.model.device)
        self.hub_visual_material = wp.array([wp.vec4(0.58, 0.12, 0.0, 0.0)], dtype=wp.vec4, device=self.model.device)
        self.floor_visual_q = wp.array(
            [wp.transform(wp.vec3(0.0), wp.quat_identity())],
            dtype=wp.transform,
            device=self.model.device,
        )
        self.floor_visual_color = wp.array([wp.vec3(0.24, 0.27, 0.30)], dtype=wp.vec3, device=self.model.device)
        self.floor_visual_material = wp.array([wp.vec4(0.72, 0.28, 0.0, 0.0)], dtype=wp.vec4, device=self.model.device)
        self.floor_visual_scale = (0.5 * args.tank_length, 0.5 * args.wall_thickness, 0.5 * args.tank_width)
        self.hub_visual_scale = (args.hub_radius, 0.5 * args.blade_width)

        self.viewer.set_model(self.model)
        self.viewer.show_particles = True
        self.viewer.set_camera(pos=wp.vec3(1.18, 0.76, 1.22), pitch=-21.0, yaw=-135.0)

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.set_defaults(num_frames=150)
        add_sph_timestep_arguments(parser, fps=60.0, substeps=64)
        add_sph_block_dimension_arguments(parser, dim_x=48, dim_y=52, dim_z=32, label="Base fluid particle")
        add_sph_particle_arguments(parser, spacing=0.012, jitter=0.0)
        parser.add_argument(
            "--resolution-scale",
            type=_positive_float,
            default=1.0,
            help="Lattice resolution multiplier; 2.33 gives about one million particles.",
        )
        parser.add_argument("--fluid-x", type=float, default=-0.64, help="Initial reservoir X origin [m].")
        parser.add_argument("--fluid-velocity", type=float, default=0.0, help="Initial fluid velocity along +X [m/s].")
        add_sph_tank_arguments(
            parser,
            tank_length=1.60,
            tank_width=0.58,
            wall_height=2.00,
            wall_thickness=0.035,
            fluid_offset_y=0.035,
        )
        add_sph_solver_config_arguments(
            parser,
            sound_speed=12.0,
            viscosity=0.0015,
            xsph=0.04,
            boundary_friction=0.03,
        )
        parser.add_argument("--wheel-x", type=float, default=0.20, help="Turbine axle X position [m].")
        parser.add_argument("--wheel-y", type=_positive_float, default=0.255, help="Turbine axle height [m].")
        parser.add_argument("--blade-count", type=_positive_int, default=6, help="Number of turbine blades.")
        parser.add_argument("--blade-length", type=_positive_float, default=0.18, help="Radial blade length [m].")
        parser.add_argument(
            "--blade-thickness", type=_positive_float, default=0.028, help="Blade tangential thickness [m]."
        )
        parser.add_argument("--blade-width", type=_positive_float, default=0.40, help="Blade width along the axle [m].")
        parser.add_argument("--hub-radius", type=_positive_float, default=0.045, help="Turbine hub radius [m].")
        parser.add_argument(
            "--wheel-density", type=_positive_float, default=350.0, help="Turbine material density [kg/m^3]."
        )
        parser.add_argument("--initial-angle", type=float, default=0.35, help="Initial turbine angle [rad].")
        parser.add_argument("--joint-damping", type=_non_negative_float, default=0.08, help="Axle load damping.")
        parser.add_argument("--joint-friction", type=_non_negative_float, default=0.01, help="Axle friction.")
        parser.add_argument(
            "--collider-margin", type=_non_negative_float, default=0.0, help="Turbine collider margin [m]."
        )
        parser.add_argument(
            "--collider-friction", type=_non_negative_float, default=0.04, help="Turbine-fluid friction."
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
        rotation = abs(float(self.state_0.joint_q.numpy()[0]) - self.initial_joint_q)
        self.max_wheel_rotation = max(self.max_wheel_rotation, rotation)

    def render(self):
        show_particles = self.viewer.show_particles
        self.viewer.begin_frame(self.sim_time)
        self._sph_render_points = log_sph_fluid_points(
            self.viewer,
            self.fluid_state_0,
            self.fluid_model,
            self.fluid_indices,
            radius_scale=self.fluid_render_radius_scale,
            speed_scale=2.0,
            hidden=not show_particles,
            render_points=self._sph_render_points,
        )
        wp.launch(
            _compose_blade_transforms,
            dim=self.blade_local_q_wp.shape[0],
            inputs=[self.wheel_body, self.state_0.body_q, self.blade_local_q_wp],
            outputs=[self.blade_world_q_wp],
            device=self.model.device,
        )
        self.viewer.log_shapes(
            "/sph_hydraulic_turbine_blades",
            newton.GeoType.BOX,
            self.blade_visual_scale,
            self.blade_world_q_wp,
            self.blade_visual_color,
            self.blade_visual_material,
            backface_culling=False,
        )
        self.viewer.log_shapes(
            "/sph_hydraulic_turbine_hub",
            newton.GeoType.CYLINDER,
            self.hub_visual_scale,
            self.state_0.body_q[self.wheel_body : self.wheel_body + 1],
            self.hub_visual_color,
            self.hub_visual_material,
            backface_culling=False,
        )
        self.viewer.log_shapes(
            "/sph_hydraulic_turbine_floor",
            newton.GeoType.BOX,
            self.floor_visual_scale,
            self.floor_visual_q,
            self.floor_visual_color,
            self.floor_visual_material,
            backface_culling=False,
        )
        self.viewer.end_frame()

    def test_final(self):
        assert_sph_state_finite(self.fluid_state_0, "density", "pressure")
        fluid_q = self.fluid_state_0.particle_q.numpy()[self.fluid_indices]
        joint_q = self.state_0.joint_q.numpy()
        assert np.isfinite(self.state_0.body_q.numpy()).all()
        assert np.isfinite(joint_q).all()
        assert self.sph_solver.collider_body_index.numpy().tolist() == [-1, self.wheel_body]
        assert self.coupling.max_collider_impulse_norm > 0.0
        assert self.max_wheel_rotation > 0.01

        tolerance = 2.0 * self.particle_spacing
        inside_tank = (
            (np.abs(fluid_q[:, 0]) <= self.tank_half_length + tolerance)
            & (fluid_q[:, 1] >= -tolerance)
            & (fluid_q[:, 1] <= self.wall_height + tolerance)
            & (np.abs(fluid_q[:, 2]) <= self.tank_half_width + tolerance)
        )
        assert np.mean(inside_tank) > 0.995
        assert np.count_nonzero(fluid_q[:, 0] > self.wheel_x) > 0


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
