# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example VBD-MPM Coupled Solver
#
# A tetrahedral VBD cube falls into an implicit-MPM granular heap.  The soft
# body's surface is exported to MPM as a prescribed-velocity deformable mesh
# collider through the generic particle proxy hooks.
#
# Command: python -m newton.examples vbd_mpm_coupled_solver
#
###########################################################################

from __future__ import annotations

import numpy as np
import warp as wp
from newton.solvers.experimental.coupled import SolverCoupledProxy

import newton
import newton.examples
from newton.solvers import SolverImplicitMPM, SolverVBD


@wp.kernel(enable_backward=False)
def _gather_particles(
    particle_ids: wp.array[int],
    particle_q: wp.array[wp.vec3],
    out_q: wp.array[wp.vec3],
):
    i = wp.tid()
    out_q[i] = particle_q[particle_ids[i]]


class _VBDMPMProxyCoupled(SolverCoupledProxy):
    """Example-local VBD source / MPM prescribed-collider proxy coupling."""

    def __init__(
        self,
        model: newton.Model,
        soft_particles: list[int],
        mpm_particles: list[int],
        soft_collider_mesh: wp.Mesh,
        mpm_config: SolverImplicitMPM.Config,
        *,
        proxy_iterations: int,
        proxy_mass_scale: float,
        vbd_iterations: int,
        collider_thickness: float,
        collider_friction: float,
    ):
        if proxy_mass_scale <= 0.0:
            raise ValueError("proxy_mass_scale must be positive")

        self.soft_collider_mesh = soft_collider_mesh

        super().__init__(
            model=model,
            entries=[
                SolverCoupledProxy.Entry(
                    name="vbd",
                    solver=lambda v: SolverVBD(
                        model=v,
                        **{
                            "iterations": vbd_iterations,
                            "particle_enable_tile_solve": True,
                            "particle_enable_self_contact": False,
                        },
                    ),
                    particles=soft_particles,
                ),
                SolverCoupledProxy.Entry(
                    name="mpm",
                    solver=lambda v: SolverImplicitMPM(model=v, config=mpm_config),
                    particles=mpm_particles,
                ),
            ],
            coupling=SolverCoupledProxy.Config(
                iterations=max(1, int(proxy_iterations)),
                proxies=[
                    SolverCoupledProxy.Proxy(
                        source="vbd",
                        destination="mpm",
                        particles=soft_particles,
                        proxy_particles=soft_particles,
                        mass_scale=proxy_mass_scale,
                        mode="lagged",
                    )
                ],
            ),
        )

        self.mpm_solver.setup_collider(
            collider_meshes=[self.soft_collider_mesh, None],
            collider_body_ids=[None, -1],
            collider_margins=[collider_thickness, None],
            collider_friction=[collider_friction, 0.8],
            collider_particle_ids=[soft_particles, None],
            model=self.view("mpm"),
        )

    @property
    def vbd_solver(self):
        return self.solver("vbd")

    @property
    def mpm_solver(self):
        return self.solver("mpm")


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.sim_time = 0.0
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 4
        self.sim_dt = self.frame_dt / self.sim_substeps

        builder = newton.ModelBuilder()
        builder.default_shape_cfg.mu = 0.8
        builder.add_ground_plane()
        SolverImplicitMPM.register_custom_attributes(builder)

        self.soft_particles = self._emit_softbody(builder)
        self.mpm_particles = self._emit_mpm_heap(builder, args.voxel_size)

        builder.color()
        self.model = builder.finalize()
        self.model.soft_contact_ke = 5.0e4
        self.model.soft_contact_kd = 1.0e-3 * self.model.soft_contact_ke
        self.model.soft_contact_mu = 0.8

        mpm_config = SolverImplicitMPM.Config()
        mpm_config.voxel_size = args.voxel_size
        mpm_config.grid_type = "fixed"
        mpm_config.grid_padding = 32
        mpm_config.max_active_cell_count = 1 << 14
        mpm_config.max_iterations = args.mpm_iterations
        mpm_config.critical_fraction = 0.0
        mpm_config.strain_basis = "P0"
        mpm_config.collider_velocity_mode = "forward"

        soft_mesh = self._create_soft_collider_mesh()
        self.solver = _VBDMPMProxyCoupled(
            model=self.model,
            soft_particles=self.soft_particles,
            mpm_particles=self.mpm_particles,
            soft_collider_mesh=soft_mesh,
            mpm_config=mpm_config,
            proxy_iterations=args.proxy_iterations,
            proxy_mass_scale=args.proxy_mass_scale,
            vbd_iterations=args.vbd_iterations,
            collider_thickness=args.collider_thickness,
            collider_friction=args.collider_friction,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.collision_pipeline = newton.CollisionPipeline(self.model)
        self.contacts = self.collision_pipeline.contacts()
        self.control = self.model.control()

        newton.examples.configure_coupled_view(self, args)
        if hasattr(self.viewer, "show_particles"):
            self.viewer.show_particles = True

        self.mpm_particle_ids = wp.array(self.mpm_particles, dtype=int, device=self.model.device)
        self.mpm_render_points = wp.empty(len(self.mpm_particles), dtype=wp.vec3, device=self.model.device)
        self.mpm_render_radii = wp.full(
            len(self.mpm_particles), args.voxel_size * 0.35, dtype=float, device=self.model.device
        )
        self.mpm_render_colors = wp.full(
            len(self.mpm_particles), wp.vec3(0.72, 0.60, 0.42), dtype=wp.vec3, device=self.model.device
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
            newton.examples.apply_coupled_viewer_forces(self, self.state_0)
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
        particle_q = self.state_0.particle_q.numpy()
        assert np.isfinite(particle_q).all(), "Particle positions contain NaN or inf values"

        soft_q = particle_q[self.soft_particles]
        mpm_q = particle_q[self.mpm_particles]
        soft_min = np.min(soft_q, axis=0)
        soft_max = np.max(soft_q, axis=0)
        mpm_min = np.min(mpm_q, axis=0)
        mpm_max = np.max(mpm_q, axis=0)

        assert np.linalg.norm(soft_max - soft_min) < 4.0, "Soft body exploded"
        assert np.linalg.norm(mpm_max - mpm_min) < 5.0, "MPM bed exploded"
        assert soft_min[2] > -0.25, f"Soft body penetrated the ground: z_min={soft_min[2]:.4f}"
        assert mpm_min[2] > -0.25, f"MPM particles penetrated the ground: z_min={mpm_min[2]:.4f}"

    def render(self):
        render_state = newton.examples.get_coupled_view_state(self)
        wp.launch(
            _gather_particles,
            dim=len(self.mpm_particles),
            inputs=[self.mpm_particle_ids, render_state.particle_q, self.mpm_render_points],
            device=self.model.device,
        )

        self.viewer.begin_frame(self.sim_time)
        newton.examples.log_coupled_view(self, self.contacts)
        self.viewer.log_points(
            "/mpm",
            points=self.mpm_render_points,
            radii=self.mpm_render_radii,
            colors=self.mpm_render_colors,
            hidden=not getattr(self.viewer, "show_particles", True),
        )
        self.viewer.end_frame()

    def _emit_softbody(self, builder: newton.ModelBuilder) -> list[int]:
        particle_start = builder.particle_count
        k_mu = 2.0e4
        k_lambda = 2.0e4
        edge_ke = 1.0e-2
        builder.add_soft_grid(
            pos=wp.vec3(-0.22, -0.22, 1.02),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, -0.4),
            dim_x=4,
            dim_y=4,
            dim_z=4,
            cell_x=0.11,
            cell_y=0.11,
            cell_z=0.11,
            density=420.0,
            k_mu=k_mu,
            k_lambda=k_lambda,
            k_damp=2.0e-3 * k_mu,
            tri_ke=0.0,
            tri_ka=0.0,
            tri_kd=0.0,
            edge_ke=edge_ke,
            edge_kd=1.0e-3 * edge_ke,
            particle_radius=0.035,
        )
        return list(range(particle_start, builder.particle_count))

    def _emit_mpm_heap(self, builder: newton.ModelBuilder, voxel_size: float) -> list[int]:
        particle_start = builder.particle_count

        particles_per_cell = 1.75
        spacing = voxel_size / particles_per_cell
        domain_half_width = 0.95
        base_depth = 0.20
        heap_radius = 0.82
        heap_height = 0.58
        density = 1800.0
        rng = np.random.default_rng(42 + particle_start)

        positions: list[list[float]] = []
        radii: list[float] = []
        xs = np.arange(-domain_half_width, domain_half_width + 0.5 * spacing, spacing)
        ys = np.arange(-domain_half_width, domain_half_width + 0.5 * spacing, spacing)
        zs = np.arange(0.5 * spacing, base_depth + heap_height + 0.5 * spacing, spacing)
        for x in xs:
            for y in ys:
                r = float(np.hypot(x, y))
                heap_t = max(0.0, 1.0 - (r / heap_radius) ** 2)
                surface_z = base_depth + heap_height * heap_t * heap_t
                for z in zs:
                    if z > surface_z:
                        break
                    jitter = (rng.random(3) - 0.5) * 0.45 * spacing
                    positions.append([float(x + jitter[0]), float(y + jitter[1]), float(z + jitter[2])])
                    radii.append(float(0.35 * voxel_size))

        particle_count = len(positions)
        mass = float(spacing**3 * density)
        builder.add_particles(
            pos=positions,
            vel=[[0.0, 0.0, 0.0]] * particle_count,
            mass=[mass] * particle_count,
            radius=radii,
            custom_attributes={
                "mpm:friction": [0.5] * particle_count,
            },
        )

        return list(range(particle_start, builder.particle_count))

    def _create_soft_collider_mesh(self) -> wp.Mesh:
        soft_local = {particle_id: i for i, particle_id in enumerate(self.soft_particles)}
        tri_indices = self.model.tri_indices.numpy().reshape(-1, 3)
        soft_tri_indices: list[int] = []
        for tri in tri_indices:
            tri_particles = [int(tri[0]), int(tri[1]), int(tri[2])]
            if all(p in soft_local for p in tri_particles):
                soft_tri_indices.extend(soft_local[p] for p in tri_particles)

        if not soft_tri_indices:
            raise RuntimeError("Softbody surface did not produce any collider triangles")

        particle_q = self.model.particle_q.numpy()[self.soft_particles]
        points = wp.array(particle_q, dtype=wp.vec3, device=self.model.device)
        velocities = wp.zeros(len(self.soft_particles), dtype=wp.vec3, device=self.model.device)
        indices = wp.array(soft_tri_indices, dtype=wp.int32, device=self.model.device)
        return wp.Mesh(points=points, indices=indices, velocities=velocities)

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        newton.examples.add_coupled_view_args(parser)
        parser.add_argument(
            "--proxy-iterations",
            help="Number of VBD/MPM proxy relaxation passes per substep",
            type=int,
            default=3,
        )
        parser.add_argument(
            "--proxy-mass-scale",
            help="Scale factor for VBD effective particle mass used by MPM proxies",
            type=float,
            default=0.5,
        )
        parser.add_argument(
            "--vbd-iterations",
            help="VBD solver iterations per substep",
            type=int,
            default=8,
        )
        parser.add_argument(
            "--mpm-iterations",
            help="Implicit MPM solver iterations per substep",
            type=int,
            default=35,
        )
        parser.add_argument(
            "--voxel-size",
            help="MPM grid voxel size",
            type=float,
            default=0.08,
        )
        parser.add_argument(
            "--collider-thickness",
            help="Softbody mesh collider thickness seen by MPM",
            type=float,
            default=0.025,
        )
        parser.add_argument(
            "--collider-friction",
            help="Friction coefficient for the VBD softbody collider in MPM",
            type=float,
            default=0.6,
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
