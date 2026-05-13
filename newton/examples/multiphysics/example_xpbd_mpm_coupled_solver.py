# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example XPBD-MPM Coupled Solver
#
# XPBD point particles fall into an implicit-MPM granular cube through
# SolverProxyCoupled's particle proxy path.  The MPM view treats XPBD particles
# as transfer-active proxy particles: they participate in P2G/G2P momentum
# exchange, but they do not contribute material stress.
#
# Command: python -m newton.examples xpbd_mpm_coupled_solver
#
###########################################################################

from __future__ import annotations

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import SolverCoupled, SolverImplicitMPM, SolverProxyCoupled, SolverXPBD


@wp.kernel(enable_backward=False)
def _gather_particles(
    particle_ids: wp.array[int],
    particle_q: wp.array[wp.vec3],
    out_q: wp.array[wp.vec3],
):
    i = wp.tid()
    out_q[i] = particle_q[particle_ids[i]]


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.sim_time = 0.0
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = args.substeps
        self.sim_dt = self.frame_dt / self.sim_substeps

        builder = newton.ModelBuilder()
        builder.default_shape_cfg.mu = 0.35
        builder.add_ground_plane()
        SolverImplicitMPM.register_custom_attributes(builder)

        self.mpm_particles = self._emit_mpm_cube(builder, args)
        self.xpbd_particles = self._emit_xpbd_particles(builder, args)

        builder.color()
        self.model = builder.finalize()
        self.model.particle_mu = args.xpbd_particle_mu
        self.model.particle_cohesion = 0.0
        self.model.particle_adhesion = 0.0
        self.model.particle_max_velocity = args.particle_max_velocity
        self.model.soft_contact_ke = 5.0e4
        self.model.soft_contact_kd = 1.0e-2
        self.model.soft_contact_mu = args.xpbd_ground_mu

        mpm_config = SolverImplicitMPM.Config()
        mpm_config.voxel_size = args.voxel_size
        mpm_config.grid_type = "fixed"
        mpm_config.grid_padding = args.grid_padding
        mpm_config.max_active_cell_count = 1 << 16
        mpm_config.max_iterations = args.mpm_iterations
        mpm_config.critical_fraction = 0.0
        mpm_config.strain_basis = "P0"

        self.solver = SolverProxyCoupled(
            model=self.model,
            entries=[
                SolverCoupled.Entry(
                    name="xpbd",
                    solver=lambda v: SolverXPBD(
                        model=v,
                        **{"iterations": args.xpbd_iterations, "soft_contact_relaxation": args.xpbd_contact_relaxation},
                    ),
                    particles=self.xpbd_particles,
                ),
                SolverCoupled.Entry(
                    name="mpm",
                    solver=lambda v: SolverImplicitMPM(model=v, config=mpm_config),
                    particles=self.mpm_particles,
                    in_place=True,
                ),
            ],
            coupling=SolverProxyCoupled.Config(
                iterations=args.proxy_iterations,
                proxies=[
                    SolverProxyCoupled.Proxy(
                        source="xpbd",
                        destination="mpm",
                        particles=self.xpbd_particles,
                        mass_scale=args.proxy_mass_scale,
                        mode="lagged",
                    )
                ],
            ),
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        self.collision_pipeline = newton.CollisionPipeline(self.solver.view("xpbd"))
        self.contacts = self.collision_pipeline.contacts()

        self.viewer.set_model(self.model)
        if hasattr(self.viewer, "show_particles"):
            self.viewer.show_particles = False
        if hasattr(self.viewer, "set_camera"):
            self.viewer.set_camera(pos=wp.vec3(1.15, -1.65, 0.95), pitch=-22.0, yaw=128.0)
            if hasattr(self.viewer, "camera") and hasattr(self.viewer.camera, "look_at"):
                self.viewer.camera.look_at(wp.vec3(0.0, 0.0, 0.32))

        self.xpbd_particle_ids = wp.array(self.xpbd_particles, dtype=int, device=self.model.device)
        self.mpm_particle_ids = wp.array(self.mpm_particles, dtype=int, device=self.model.device)
        self.xpbd_render_points = wp.empty(len(self.xpbd_particles), dtype=wp.vec3, device=self.model.device)
        self.mpm_render_points = wp.empty(len(self.mpm_particles), dtype=wp.vec3, device=self.model.device)
        self.xpbd_render_radii = wp.full(
            len(self.xpbd_particles), args.xpbd_radius, dtype=float, device=self.model.device
        )
        self.mpm_render_radii = wp.full(
            len(self.mpm_particles), args.voxel_size * 0.32, dtype=float, device=self.model.device
        )
        self.xpbd_render_colors = wp.full(
            len(self.xpbd_particles), wp.vec3(0.10, 0.36, 0.95), dtype=wp.vec3, device=self.model.device
        )
        self.mpm_render_colors = wp.full(
            len(self.mpm_particles), wp.vec3(0.72, 0.59, 0.38), dtype=wp.vec3, device=self.model.device
        )

        self.graph = None
        self.capture()

    def capture(self):
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def test_final(self):
        self._assert_finite_particles()

    def test_post_step(self):
        self._assert_finite_particles()

    def _assert_finite_particles(self):
        particle_q = self.state_0.particle_q.numpy()
        particle_qd = self.state_0.particle_qd.numpy()
        assert np.isfinite(particle_q).all(), f"Particle positions contain NaN or inf values at t={self.sim_time:.4f}"
        assert np.isfinite(particle_qd).all(), f"Particle velocities contain NaN or inf values at t={self.sim_time:.4f}"

        xpbd_q = particle_q[self.xpbd_particles]
        mpm_q = particle_q[self.mpm_particles]
        xpbd_qd = particle_qd[self.xpbd_particles]
        mpm_qd = particle_qd[self.mpm_particles]
        xpbd_min = np.min(xpbd_q, axis=0)
        xpbd_max = np.max(xpbd_q, axis=0)
        mpm_min = np.min(mpm_q, axis=0)
        mpm_max = np.max(mpm_q, axis=0)

        assert np.linalg.norm(xpbd_max - xpbd_min) < 4.0, "XPBD particles exploded"
        assert np.linalg.norm(mpm_max - mpm_min) < 5.0, "MPM cube exploded"
        assert np.max(np.linalg.norm(xpbd_qd, axis=1)) < 40.0, "XPBD particle velocities exploded"
        assert np.max(np.linalg.norm(mpm_qd, axis=1)) < 40.0, "MPM particle velocities exploded"
        assert xpbd_min[2] > -0.20, f"XPBD particles penetrated the ground: z_min={xpbd_min[2]:.4f}"
        assert mpm_min[2] > -0.20, f"MPM particles penetrated the ground: z_min={mpm_min[2]:.4f}"

    def render(self):
        wp.launch(
            _gather_particles,
            dim=len(self.xpbd_particles),
            inputs=[self.xpbd_particle_ids, self.state_0.particle_q, self.xpbd_render_points],
            device=self.model.device,
        )
        wp.launch(
            _gather_particles,
            dim=len(self.mpm_particles),
            inputs=[self.mpm_particle_ids, self.state_0.particle_q, self.mpm_render_points],
            device=self.model.device,
        )

        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.log_points(
            "/xpbd_particles",
            points=self.xpbd_render_points,
            radii=self.xpbd_render_radii,
            colors=self.xpbd_render_colors,
        )
        self.viewer.log_points(
            "/mpm_particles",
            points=self.mpm_render_points,
            radii=self.mpm_render_radii,
            colors=self.mpm_render_colors,
        )
        self.viewer.end_frame()

    def _emit_xpbd_particles(self, builder: newton.ModelBuilder, args) -> list[int]:
        particle_start = builder.particle_count
        builder.add_particle_grid(
            pos=wp.vec3(-0.22, -0.22, 0.78),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, -0.25),
            dim_x=args.xpbd_dim_x,
            dim_y=args.xpbd_dim_y,
            dim_z=args.xpbd_dim_z,
            cell_x=2.08 * args.xpbd_radius,
            cell_y=2.08 * args.xpbd_radius,
            cell_z=2.08 * args.xpbd_radius,
            mass=args.xpbd_mass,
            jitter=0.10 * args.xpbd_radius,
            radius_mean=args.xpbd_radius,
        )
        return list(range(particle_start, builder.particle_count))

    def _emit_mpm_cube(self, builder: newton.ModelBuilder, args) -> list[int]:
        particle_start = builder.particle_count

        particles_per_cell = 1.85
        cube_lo = np.array([-0.34, -0.34, 0.035])
        cube_hi = np.array([0.34, 0.34, 0.43])
        cube_res = np.maximum(1, np.ceil(particles_per_cell * (cube_hi - cube_lo) / args.voxel_size).astype(int))
        cell_size = (cube_hi - cube_lo) / cube_res
        radius = float(np.max(cell_size) * 0.45)
        mass = float(np.prod(cell_size) * args.mpm_density)

        builder.add_particle_grid(
            pos=wp.vec3(cube_lo),
            rot=wp.quat_identity(),
            vel=wp.vec3(args.mpm_initial_velocity),
            dim_x=int(cube_res[0]) + 1,
            dim_y=int(cube_res[1]) + 1,
            dim_z=int(cube_res[2]) + 1,
            cell_x=float(cell_size[0]),
            cell_y=float(cell_size[1]),
            cell_z=float(cell_size[2]),
            mass=mass,
            jitter=0.25 * radius,
            radius_mean=radius,
            custom_attributes={
                "mpm:friction": args.mpm_friction,
                "mpm:yield_pressure": args.mpm_yield_pressure,
                "mpm:tensile_yield_ratio": 0.0,
            },
        )

        return list(range(particle_start, builder.particle_count))

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument(
            "--proxy-iterations",
            help="Number of XPBD/MPM proxy relaxation passes per substep",
            type=int,
            default=3,
        )
        parser.add_argument(
            "--proxy-mass-scale",
            help="Virtual mass scale applied to XPBD proxy particles inside the MPM view",
            type=float,
            default=0.15,
        )
        parser.add_argument(
            "--xpbd-iterations",
            help="XPBD solver iterations per substep",
            type=int,
            default=8,
        )
        parser.add_argument(
            "--xpbd-contact-relaxation",
            help="XPBD particle contact relaxation",
            type=float,
            default=0.55,
        )
        parser.add_argument(
            "--xpbd-radius",
            help="Radius of XPBD point particles",
            type=float,
            default=0.022,
        )
        parser.add_argument(
            "--xpbd-mass",
            help="Mass of each XPBD point particle",
            type=float,
            default=0.018,
        )
        parser.add_argument(
            "--xpbd-dim-x",
            help="Number of XPBD particles along X",
            type=int,
            default=8,
        )
        parser.add_argument(
            "--xpbd-dim-y",
            help="Number of XPBD particles along Y",
            type=int,
            default=8,
        )
        parser.add_argument(
            "--xpbd-dim-z",
            help="Number of XPBD particles along Z",
            type=int,
            default=4,
        )
        parser.add_argument(
            "--xpbd-particle-mu",
            help="XPBD particle-particle friction coefficient",
            type=float,
            default=0.25,
        )
        parser.add_argument(
            "--xpbd-ground-mu",
            help="XPBD particle-ground friction coefficient",
            type=float,
            default=0.35,
        )
        parser.add_argument(
            "--mpm-iterations",
            help="Implicit MPM solver iterations per substep",
            type=int,
            default=40,
        )
        parser.add_argument(
            "--mpm-friction",
            help="Granular friction coefficient for MPM particles",
            type=float,
            default=0.42,
        )
        parser.add_argument(
            "--mpm-density",
            help="MPM material density [kg/m^3]",
            type=float,
            default=800.0,
        )
        parser.add_argument(
            "--mpm-yield-pressure",
            help="MPM compressive yield pressure [Pa]",
            type=float,
            default=1.0e5,
        )
        parser.add_argument(
            "--mpm-initial-velocity",
            help="Initial MPM cube velocity [m/s]",
            type=float,
            nargs=3,
            default=[0.22, 0.0, 0.0],
        )
        parser.add_argument(
            "--voxel-size",
            help="MPM grid voxel size",
            type=float,
            default=0.055,
        )
        parser.add_argument(
            "--grid-padding",
            help="Fixed MPM grid padding in voxels",
            type=int,
            default=38,
        )
        parser.add_argument(
            "--substeps",
            help="Coupled substeps per rendered frame",
            type=int,
            default=2,
        )
        parser.add_argument(
            "--particle-max-velocity",
            help="Velocity clamp shared by XPBD and MPM particles [m/s]",
            type=float,
            default=18.0,
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
