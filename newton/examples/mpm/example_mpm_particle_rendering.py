# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import SolverImplicitMPM


@wp.kernel
def gather_render_grains(
    grains: wp.array2d[wp.vec3],
    particle_indices: wp.array[wp.int32],
    points: wp.array[wp.vec3],
):
    tid = wp.tid()
    grains_per_particle = grains.shape[1]
    particle_slot = tid // grains_per_particle
    grain_slot = tid - particle_slot * grains_per_particle
    particle_index = particle_indices[particle_slot]
    points[tid] = grains[particle_index, grain_slot]


class Example:
    def __init__(self, viewer, args):
        # setup simulation parameters first
        self.fps = 60.0
        self.frame_dt = 1.0 / self.fps

        # group related attributes by prefix
        self.sim_time = 0.0
        self.sim_substeps = 1
        self.sim_dt = self.frame_dt / self.sim_substeps

        # save a reference to the viewer
        self.viewer = viewer
        self.extrapolate_into_colliders = args.extrapolate_into_colliders
        self.collider_extrapolation_depth = args.collider_extrapolation_depth
        self.collider_extrapolation_onset = args.collider_extrapolation_onset
        builder = newton.ModelBuilder()

        # Register MPM custom attributes before adding particles
        SolverImplicitMPM.register_custom_attributes(builder)

        grain_particles, surface_particles = Example.emit_particles(builder, args)
        builder.add_ground_plane()
        self.model = builder.finalize()

        grain_particles = np.asarray(grain_particles, dtype=np.int32)
        surface_particles = np.asarray(surface_particles, dtype=np.int32)
        self.grain_particle_indices = wp.array(grain_particles, dtype=wp.int32, device=self.model.device)
        surface_flags = np.zeros(self.model.particle_count, dtype=np.int32)
        surface_flags[surface_particles] = int(newton.ParticleFlags.ACTIVE)
        self.surface_flags = wp.array(surface_flags, dtype=wp.int32, device=self.model.device)

        mpm_options = SolverImplicitMPM.Config()
        mpm_options.voxel_size = args.voxel_size

        # Initialize MPM solver
        self.solver = SolverImplicitMPM(self.model, mpm_options)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        # Setup grain rendering for the left particle block.
        self.points_per_particle = args.points_per_particle
        self.grains = self.solver.sample_render_grains(self.state_0, args.points_per_particle)
        grain_radius = args.voxel_size / (3 * args.points_per_particle)
        grain_count = grain_particles.shape[0] * args.points_per_particle
        self.grain_points = wp.empty(grain_count, dtype=wp.vec3, device=self.model.device)
        self.grain_radii = wp.full(grain_count, value=grain_radius, dtype=float, device=self.model.device)
        self.grain_colors = wp.full(grain_count, value=wp.vec3(0.7, 0.6, 0.4), dtype=wp.vec3, device=self.model.device)

        # Setup surface extraction for the right particle block.
        surface_voxel_size = args.surface_voxel_size if args.surface_voxel_size is not None else 0.45 * args.voxel_size
        self.surface_ctx = self.solver.create_particle_surface(
            voxel_size=surface_voxel_size,
            kernel_radius=args.kernel_radius,
            threshold=args.threshold,
            smooth_lambda=args.smooth_lambda,
            anisotropy_ratio=args.anisotropy_ratio,
            anisotropy_scale=args.anisotropy_scale,
            kernel_scale=args.kernel_scale,
            anisotropy_min_neighbors=args.anisotropy_min_neighbors,
            anisotropy_strength=args.anisotropy_strength,
            field_smooth_iterations=args.field_smooth_iterations,
            field_smooth_radius=args.field_smooth_radius,
            surface_method=args.surface_method,
            particle_sdf_radius_scale=args.particle_sdf_radius_scale,
            particle_sdf_band=args.particle_sdf_band,
            field_mode=args.field_mode,
            redistance_iterations=args.redistance_iterations,
            mesh_smooth_iterations=args.mesh_smooth_iterations,
            anisotropic=args.anisotropic,
        )
        self.surface_path = "/model/particle_surface"
        self._empty_surface_points = wp.empty(0, dtype=wp.vec3, device=self.model.device)
        self._empty_surface_indices = wp.empty(0, dtype=wp.int32, device=self.model.device)
        self._empty_surface_normals = wp.empty(0, dtype=wp.vec3, device=self.model.device)

        self.viewer.set_model(self.model)
        self.viewer.show_particles = False
        self.viewer.set_camera(pos=wp.vec3(6.0, -8.0, 4.0), pitch=-21.5, yaw=126.9)

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.solver.step(self.state_0, self.state_1, None, None, self.sim_dt)
            self.solver.project_outside(self.state_1, self.state_1, self.sim_dt)

            # update grains
            self.solver.update_particle_frames(self.state_0, self.state_1, self.sim_dt)
            self.solver.update_render_grains(self.state_0, self.state_1, self.grains, self.sim_dt)

            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt

    def test_final(self):
        newton.examples.test_particle_state(
            self.state_0,
            "all particles are above the ground",
            lambda q, qd: q[2] > -0.05,
        )

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)

        wp.launch(
            gather_render_grains,
            dim=self.grain_points.shape[0],
            inputs=[self.grains, self.grain_particle_indices, self.grain_points],
            device=self.model.device,
        )
        self.viewer.log_points(
            "/model/render_grains",
            points=self.grain_points,
            radii=self.grain_radii,
            colors=self.grain_colors,
            hidden=False,
        )

        verts, indices, normals = self.solver.extract_particle_surface(
            self.state_0,
            self.surface_ctx,
            particle_flags=self.surface_flags,
            extrapolate_into_colliders=self.extrapolate_into_colliders,
            collider_extrapolation_depth=self.collider_extrapolation_depth,
            collider_extrapolation_onset=self.collider_extrapolation_onset,
        )
        if verts is None or verts.shape[0] == 0:
            verts = self._empty_surface_points
            indices = self._empty_surface_indices
            normals = self._empty_surface_normals
            hidden = True
        else:
            hidden = False

        self.viewer.log_mesh(
            self.surface_path,
            verts,
            indices,
            normals,
            hidden=hidden,
            dynamic=True,
            color=(0.45, 0.62, 0.78),
        )
        self.viewer.end_frame()

    @staticmethod
    def emit_particles(builder: newton.ModelBuilder, args):
        voxel_size = args.voxel_size

        particles_per_cell = 3
        grain_lo = np.array([-1.05, -0.45, 0.0])
        grain_hi = np.array([-0.25, 0.45, 1.5])
        surface_lo = np.array([0.25, -0.45, 0.0])
        surface_hi = np.array([1.05, 0.45, 1.5])

        grain_res = np.array(np.ceil(particles_per_cell * (grain_hi - grain_lo) / voxel_size), dtype=int)
        surface_res = np.array(np.ceil(particles_per_cell * (surface_hi - surface_lo) / voxel_size), dtype=int)

        grain_particles = Example._spawn_particles(builder, grain_res, grain_lo, grain_hi, density=2500)
        surface_particles = Example._spawn_particles(builder, surface_res, surface_lo, surface_hi, density=2500)
        return grain_particles, surface_particles

    @staticmethod
    def _spawn_particles(
        builder: newton.ModelBuilder,
        res,
        bounds_lo,
        bounds_hi,
        density,
    ):
        cell_size = (bounds_hi - bounds_lo) / res
        cell_volume = np.prod(cell_size)
        radius = np.max(cell_size) * 0.5
        mass = np.prod(cell_volume) * density

        begin_id = len(builder.particle_q)
        builder.add_particle_grid(
            pos=wp.vec3(bounds_lo),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=res[0] + 1,
            dim_y=res[1] + 1,
            dim_z=res[2] + 1,
            cell_x=cell_size[0],
            cell_y=cell_size[1],
            cell_z=cell_size[2],
            mass=mass,
            jitter=2.0 * radius,
            radius_mean=radius,
            flags=newton.ParticleFlags.ACTIVE,
        )
        end_id = len(builder.particle_q)
        return np.arange(begin_id, end_id, dtype=np.int32)

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--voxel-size", "-dx", type=float, default=0.1)
        parser.add_argument("--points-per-particle", "-ppp", type=int, default=8)
        parser.add_argument(
            "--surface-voxel-size",
            type=float,
            default=None,
            help="Voxel size for the surface grid (default: 0.45 * --voxel-size)",
        )
        parser.add_argument(
            "--kernel-radius",
            type=float,
            default=None,
            help="Splatting kernel radius (default: 3 * surface voxel size)",
        )
        parser.add_argument(
            "--threshold", type=float, default=0.25, help="Isosurface level (field ~1.0 inside, default 0.25)"
        )
        parser.add_argument(
            "--smooth-lambda",
            type=float,
            default=0.5,
            help="Particle center smoothing blend; lower values preserve small droplets (default 0.5)",
        )
        parser.add_argument(
            "--kernel-scale",
            type=float,
            default=0.5,
            help="Kernel radius multiplier for isotropic kernels and anisotropic geometric mean (default 0.5)",
        )
        parser.add_argument(
            "--anisotropy-ratio",
            type=float,
            default=4.0,
            help="Maximum anisotropic kernel axis ratio; higher values allow flatter ellipsoids (default 4)",
        )
        parser.add_argument(
            "--anisotropy-scale",
            type=float,
            default=1.0,
            help="Relative anisotropic kernel scale on top of --kernel-scale (default 1)",
        )
        parser.add_argument(
            "--anisotropy-min-neighbors",
            type=int,
            default=25,
            help="Minimum neighbors required before using an anisotropic kernel (default 25)",
        )
        parser.add_argument(
            "--anisotropy-strength",
            type=float,
            default=1.0,
            help="Blend from isotropic to anisotropic kernels; lower values keep boundary particles linked inward (default 1)",
        )
        parser.add_argument("--field-smooth-iterations", type=int, default=0)
        parser.add_argument("--field-smooth-radius", type=int, default=1)
        parser.add_argument(
            "--surface-method",
            choices=("density", "particle_sdf"),
            default="density",
            help="Surface reconstruction method (default density)",
        )
        parser.add_argument(
            "--particle-sdf-radius-scale",
            type=float,
            default=1.0,
            help="Particle radius multiplier for --surface-method particle_sdf (default 1)",
        )
        parser.add_argument(
            "--particle-sdf-band",
            type=float,
            default=2.0,
            help="Normalized particle SDF update band for --surface-method particle_sdf (default 2)",
        )
        parser.add_argument(
            "--field-mode",
            choices=("density", "sdf"),
            default=None,
            help="Keep the density field or convert it to an SDF after extraction (default: method dependent)",
        )
        parser.add_argument(
            "--redistance-iterations", type=int, default=0, help="Eikonal redistancing passes for --field-mode sdf"
        )
        parser.add_argument(
            "--extrapolate-into-colliders",
            action="store_true",
            help="Mirror-extrapolate SDF field into collider interiors",
        )
        parser.add_argument(
            "--collider-extrapolation-depth",
            type=float,
            default=None,
            help="Maximum collider extrapolation depth (default: 4 * surface voxel size)",
        )
        parser.add_argument(
            "--collider-extrapolation-onset",
            type=float,
            default=0.0,
            help="Collider signed distance where extrapolation starts (default 0, the collider surface)",
        )
        parser.add_argument("--mesh-smooth-iterations", type=int, default=0)
        parser.add_argument("--anisotropic", action="store_true")
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()

    viewer, args = newton.examples.init(parser)

    if args.surface_voxel_size is None:
        args.surface_voxel_size = args.voxel_size * 0.45
    if args.extrapolate_into_colliders:
        if args.field_mode is None:
            args.field_mode = "sdf"
        elif args.field_mode != "sdf":
            parser.error("--extrapolate-into-colliders requires --field-mode sdf")

    # Create example and run
    newton.examples.run(Example(viewer, args), args)
