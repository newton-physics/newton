# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Water-like MPM dam break rendered as an extracted surface mesh."""

import argparse
import warnings

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import SolverImplicitMPM


@wp.kernel
def _project_inside_tank(
    positions: wp.array[wp.vec3],
    velocities: wp.array[wp.vec3],
    particle_flags: wp.array[wp.int32],
    half_extents: wp.vec3,
    floor_height: float,
    projection_threshold: float,
):
    i = wp.tid()
    if (particle_flags[i] & newton.ParticleFlags.ACTIVE) == 0:
        return

    pos = positions[i]
    vel = velocities[i]
    min_x = -half_extents[0] - projection_threshold
    max_x = half_extents[0] + projection_threshold
    min_y = -half_extents[1] - projection_threshold
    max_y = half_extents[1] + projection_threshold
    min_z = floor_height - projection_threshold

    if pos[0] < min_x:
        pos[0] = min_x
        vel[0] = wp.max(vel[0], 0.0)
    elif pos[0] > max_x:
        pos[0] = max_x
        vel[0] = wp.min(vel[0], 0.0)
    if pos[1] < min_y:
        pos[1] = min_y
        vel[1] = wp.max(vel[1], 0.0)
    elif pos[1] > max_y:
        pos[1] = max_y
        vel[1] = wp.min(vel[1], 0.0)
    if pos[2] < min_z:
        pos[2] = min_z
        vel[2] = wp.max(vel[2], 0.0)

    positions[i] = pos
    velocities[i] = vel


class Example:
    def __init__(self, viewer, args):
        self.fps = args.fps
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = args.substeps
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.viewer = viewer

        self.tank_extents = np.asarray(args.tank_extents, dtype=np.float32)

        builder = newton.ModelBuilder()
        SolverImplicitMPM.register_custom_attributes(builder)
        self.initial_water_max_x, self.particle_spacing = self._emit_water(builder, args)
        self._add_tank(builder, args)

        self.model = builder.finalize()
        self.model.set_gravity(args.gravity)

        self.model.mpm.friction.fill_(args.friction)
        self.model.mpm.tensile_yield_ratio.fill_(args.tensile_yield_ratio)
        self.model.mpm.viscosity.fill_(args.viscosity)

        mpm_config = SolverImplicitMPM.Config()
        mpm_config.voxel_size = args.voxel_size
        mpm_config.max_iterations = args.max_iterations
        mpm_config.tolerance = args.tolerance
        mpm_config.warmstart_mode = args.warmstart_mode
        mpm_config.strain_basis = args.strain_basis
        mpm_config.collider_basis = args.collider_basis
        mpm_config.velocity_basis = args.velocity_basis
        self.solver = SolverImplicitMPM(self.model, config=mpm_config)

        self.projection_threshold = (
            args.projection_threshold if args.projection_threshold is not None else 0.25 * args.voxel_size
        )
        self.floor_height = -args.wall_thickness
        if self.projection_threshold < 0.0:
            raise ValueError("projection_threshold must be non-negative")
        self.solver.setup_collider(collider_projection_threshold=[self.projection_threshold])

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.state_0.mpm.particle_Jp.fill_(1.0)
        self.solver.project_outside(self.state_0, self.state_0, self.sim_dt)

        surface_voxel_size = args.surface_voxel_size if args.surface_voxel_size is not None else 0.3 * args.voxel_size
        surface_kernel_radius = (
            args.surface_kernel_radius
            if args.surface_kernel_radius is not None
            else max(3.0 * self.particle_spacing, 1.5 * surface_voxel_size)
        )
        self.surface = self.solver.create_particle_surface(
            voxel_size=surface_voxel_size,
            max_grid_cells=args.surface_max_grid_cells,
            kernel_radius=surface_kernel_radius,
            threshold=args.surface_threshold,
            smooth_lambda=args.surface_smoothing,
            anisotropic=args.anisotropic,
            kernel_scale=args.surface_kernel_scale,
            anisotropy_ratio=args.anisotropy_ratio,
            anisotropy_scale=args.anisotropy_scale,
            anisotropy_min_neighbors=args.anisotropy_min_neighbors,
            anisotropy_strength=args.anisotropy_strength,
            field_smooth_iterations=args.field_smooth_iterations,
            field_smooth_radius=args.field_smooth_radius,
            field_mode="sdf" if args.extrapolate_into_colliders else "density",
            redistance_iterations=1 if args.extrapolate_into_colliders else 0,
            mesh_smooth_iterations=args.mesh_smooth_iterations,
        )
        self.extrapolate_into_colliders = args.extrapolate_into_colliders
        self.collider_extrapolation_depth = min(2.0 * surface_voxel_size, 0.5 * args.wall_thickness)
        self.surface_path = "/model/water_surface"
        self.surface_triangle_count = 0

        self._empty_surface_points = wp.empty(0, dtype=wp.vec3, device=self.model.device)
        self._empty_surface_indices = wp.empty(0, dtype=wp.int32, device=self.model.device)
        self._empty_surface_normals = wp.empty(0, dtype=wp.vec3, device=self.model.device)

        self.viewer.set_model(self.model)
        self.viewer.show_visual = False
        self.viewer.show_particles = args.show_particles
        self.viewer.set_camera(pos=wp.vec3(5.0, -6.0, 4.0), pitch=-20.0, yaw=130.0)
        self._capture_surface_extraction()

    def _extract_surface(self) -> newton.geometry.ParticleSurface.ExtractionMesh:
        return self.solver.extract_particle_surface(
            self.state_0,
            self.surface,
            extrapolate_into_colliders=self.extrapolate_into_colliders,
            collider_extrapolation_depth=self.collider_extrapolation_depth,
        )

    def _capture_surface_extraction(self):
        self.surface_graph = None
        self.surface_mesh = None
        if not self.model.device.is_cuda:
            return
        if self.sim_substeps % 2 != 0:
            warnings.warn("Sim substeps must be even for graph capture of surface extraction", stacklevel=2)
            return

        self.surface_mesh = self._extract_surface()
        with wp.ScopedCapture(device=self.model.device) as capture:
            self.surface_mesh = self._extract_surface()
        self.surface_graph = capture.graph

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.solver.step(self.state_0, self.state_1, None, None, self.sim_dt)
            self.solver.project_outside(self.state_1, self.state_1, self.sim_dt)
            wp.launch(
                _project_inside_tank,
                dim=self.state_1.particle_count,
                inputs=[
                    self.state_1.particle_q,
                    self.state_1.particle_qd,
                    self.model.particle_flags,
                    wp.vec3(self.tank_extents),
                    self.floor_height,
                    self.projection_threshold,
                ],
                device=self.model.device,
            )
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)

        if self.surface_graph is None:
            self.surface_mesh = self._extract_surface()
        else:
            wp.capture_launch(self.surface_graph)
        verts, indices, normals = self.surface_mesh.to_arrays()
        if verts is None:
            verts = self._empty_surface_points
            indices = self._empty_surface_indices
            normals = self._empty_surface_normals
            hidden = True
            self.surface_triangle_count = 0
        else:
            hidden = False
            self.surface_triangle_count = indices.shape[0] // 3

        self.viewer.log_mesh(
            self.surface_path,
            verts,
            indices,
            normals,
            hidden=hidden,
            dynamic=True,
            color=(0.12, 0.42, 0.8),
            roughness=0.15,
            metallic=0.0,
        )
        self.viewer.end_frame()

    def test_final(self):
        positions = self.state_0.particle_q.numpy()
        hx, hy, _hz = self.tank_extents
        tolerance = 2.0 * self.solver.voxel_size
        inside_tank = (
            (positions[:, 0] > -hx - tolerance)
            & (positions[:, 0] < hx + tolerance)
            & (np.abs(positions[:, 1]) < hy + tolerance)
            & (positions[:, 2] > -tolerance)
        )
        if not np.all(inside_tank):
            raise ValueError(f"{np.count_nonzero(~inside_tank)} water particles escaped the tank")
        if self.sim_time >= 0.5 and np.max(positions[:, 0]) < self.initial_water_max_x + 0.1:
            raise ValueError("The water column did not spread after release")
        if self.surface_triangle_count == 0:
            raise ValueError("Water surface extraction produced no triangles")

    @staticmethod
    def _emit_water(builder: newton.ModelBuilder, args) -> tuple[float, float]:
        if args.particles_per_cell <= 0:
            raise ValueError("particles_per_cell must be positive")

        water_lo = np.asarray(args.emit_lo, dtype=np.float32)
        water_hi = np.asarray(args.emit_hi, dtype=np.float32)
        water_extent = water_hi - water_lo
        if np.any(water_extent <= 0.0):
            raise ValueError("emit_hi must be greater than emit_lo on every axis")

        particle_res = np.ceil(args.particles_per_cell * water_extent / args.voxel_size).astype(int)
        cell_size = water_extent / particle_res
        radius = 0.5 * float(np.max(cell_size))

        builder.add_particle_grid(
            # Center jittered samples so the requested volume remains inside the tank.
            pos=wp.vec3(water_lo + 0.5 * cell_size),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=int(particle_res[0]),
            dim_y=int(particle_res[1]),
            dim_z=int(particle_res[2]),
            cell_x=float(cell_size[0]),
            cell_y=float(cell_size[1]),
            cell_z=float(cell_size[2]),
            mass=float(np.prod(cell_size) * args.density),
            jitter=2.0 * radius,
            radius_mean=radius,
            flags=newton.ParticleFlags.ACTIVE,
        )
        return float(water_hi[0]), float(np.max(cell_size))

    @staticmethod
    def _add_tank(builder: newton.ModelBuilder, args):
        half_length, half_width, half_height = args.tank_extents
        vertices, indices = Example._create_hollow_box_mesh(
            half_length,
            half_width,
            half_height,
            args.wall_thickness,
        )
        builder.add_shape_mesh(
            body=-1,
            mesh=newton.Mesh(vertices, indices, compute_inertia=False, is_solid=False),
            cfg=newton.ModelBuilder.ShapeConfig(density=0.0, mu=0.0),
            color=(0.32, 0.35, 0.4),
        )
        builder.add_ground_plane(
            height=-args.wall_thickness,
            cfg=newton.ModelBuilder.ShapeConfig(density=0.0, mu=0.0),
        )

    @staticmethod
    def _create_hollow_box_mesh(
        hx: float,
        hy: float,
        hz: float,
        thickness: float,
        bevel: float | None = None,
        bevel_segments: int = 4,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Create the open-top hollow-box collider from the S26 dam-break scene."""
        if bevel is None:
            bevel = 0.125 * thickness
        top_z = 2.0 * hz
        vertices = []
        triangles = []

        def add_vertex(x, y, z):
            index = len(vertices)
            vertices.append((x, y, z))
            return index

        def add_quad(a, b, c, d):
            triangles.extend((a, b, c, a, c, d))

        arc = np.linspace(0.0, 0.5 * np.pi, bevel_segments + 1)
        ring_count = bevel_segments + 2
        ring_z = np.empty(ring_count)
        ring_inset = np.empty(ring_count)
        for i in range(bevel_segments + 1):
            ring_z[i] = bevel * (1.0 - np.cos(arc[i]))
            ring_inset[i] = bevel * np.cos(arc[i])
        ring_z[-1] = top_z
        ring_inset[-1] = 0.0

        corners = (
            (-hx + bevel, -hy + bevel, np.pi),
            (hx - bevel, -hy + bevel, 1.5 * np.pi),
            (hx - bevel, hy - bevel, 0.0),
            (-hx + bevel, hy - bevel, 0.5 * np.pi),
        )
        perimeter_count = 4 * (bevel_segments + 1)
        ring_indices = []
        for z, inset in zip(ring_z, ring_inset, strict=True):
            ring = []
            for cx, cy, angle_start in corners:
                for segment in range(bevel_segments + 1):
                    angle = angle_start + segment * (0.5 * np.pi) / bevel_segments
                    ring.append(
                        add_vertex(
                            cx + (bevel - inset) * np.cos(angle),
                            cy + (bevel - inset) * np.sin(angle),
                            z,
                        )
                    )
            ring_indices.append(ring)

        for ring in range(ring_count - 1):
            for i in range(perimeter_count):
                j = (i + 1) % perimeter_count
                add_quad(
                    ring_indices[ring][j],
                    ring_indices[ring][i],
                    ring_indices[ring + 1][i],
                    ring_indices[ring + 1][j],
                )

        floor_center = add_vertex(0.0, 0.0, 0.0)
        for i in range(perimeter_count):
            j = (i + 1) % perimeter_count
            triangles.extend((floor_center, ring_indices[0][i], ring_indices[0][j]))

        outer = (
            add_vertex(-hx - thickness, -hy - thickness, -thickness),
            add_vertex(hx + thickness, -hy - thickness, -thickness),
            add_vertex(hx + thickness, hy + thickness, -thickness),
            add_vertex(-hx - thickness, hy + thickness, -thickness),
            add_vertex(-hx - thickness, -hy - thickness, top_z),
            add_vertex(hx + thickness, -hy - thickness, top_z),
            add_vertex(hx + thickness, hy + thickness, top_z),
            add_vertex(-hx - thickness, hy + thickness, top_z),
        )
        add_quad(outer[0], outer[4], outer[7], outer[3])
        add_quad(outer[1], outer[2], outer[6], outer[5])
        add_quad(outer[0], outer[1], outer[5], outer[4])
        add_quad(outer[3], outer[7], outer[6], outer[2])
        add_quad(outer[0], outer[3], outer[2], outer[1])

        inner_top = ring_indices[-1]
        outer_top = []
        for cx, cy, angle_start in corners:
            for segment in range(bevel_segments + 1):
                angle = angle_start + segment * (0.5 * np.pi) / bevel_segments
                px = cx + bevel * np.cos(angle)
                py = cy + bevel * np.sin(angle)
                outer_top.append(
                    add_vertex(
                        np.clip(px, -hx - thickness, hx + thickness),
                        np.clip(py, -hy - thickness, hy + thickness),
                        top_z,
                    )
                )
        for i in range(perimeter_count):
            j = (i + 1) % perimeter_count
            add_quad(inner_top[i], inner_top[j], outer_top[j], outer_top[i])

        return np.asarray(vertices, dtype=np.float32), np.asarray(triangles, dtype=np.int32)

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--fps", type=float, default=60.0)
        parser.add_argument("--substeps", type=int, default=2)
        parser.add_argument("--gravity", type=float, nargs=3, default=[0.0, 0.0, -10.0])
        parser.add_argument("--voxel-size", "-dx", type=float, default=0.05)
        parser.add_argument("--max-iterations", "-it", type=int, default=50)
        parser.add_argument("--tolerance", "-tol", type=float, default=1.0e-3)
        parser.add_argument("--warmstart-mode", type=str, default="particles")
        parser.add_argument("--strain-basis", "-sb", type=str, default="P0")
        parser.add_argument("--collider-basis", "-cb", type=str, default="pic")
        parser.add_argument("--velocity-basis", "-vb", type=str, default="Q1")
        parser.add_argument(
            "--projection-threshold",
            type=float,
            default=None,
            help="Minimum collider penetration before projection [m] (default: 0.25 * --voxel-size)",
        )
        parser.add_argument("--density", type=float, default=1000.0)
        parser.add_argument("--friction", type=float, default=0.0)
        parser.add_argument("--tensile-yield-ratio", type=float, default=1.0)
        parser.add_argument("--viscosity", type=float, default=0.0)
        parser.add_argument(
            "--particles-per-cell",
            type=int,
            default=3,
            help="Particle samples per MPM cell axis; particle count scales cubically",
        )
        parser.add_argument("--emit-lo", type=float, nargs=3, default=[-2.0, -0.5, -0.01])
        parser.add_argument("--emit-hi", type=float, nargs=3, default=[0.0, 0.5, 1.0])
        parser.add_argument(
            "--tank-extents",
            type=float,
            nargs=3,
            default=[2.0, 0.5, 2.0],
            help="Tank interior half-extents; its interior spans z=[0, 2 * hz] [m]",
        )
        parser.add_argument("--wall-thickness", type=float, default=0.15)
        parser.add_argument(
            "--surface-voxel-size",
            type=float,
            default=None,
            help="Surface-grid voxel size [m] (default: 0.3 * --voxel-size)",
        )
        parser.add_argument(
            "--surface-max-grid-cells",
            type=int,
            default=12_000_000,
            help="Maximum logical surface-grid cell count",
        )
        parser.add_argument(
            "--surface-kernel-radius",
            type=float,
            default=None,
            help="Particle reconstruction radius [m] (default: 3 * particle spacing)",
        )
        parser.add_argument(
            "--surface-kernel-scale",
            type=float,
            default=0.6,
            help="Kernel support scale relative to --surface-kernel-radius",
        )
        parser.add_argument("--surface-threshold", type=float, default=0.25)
        parser.add_argument("--surface-smoothing", type=float, default=0.0)
        parser.add_argument("--field-smooth-iterations", type=int, default=1)
        parser.add_argument("--field-smooth-radius", type=int, default=2)
        parser.add_argument(
            "--anisotropy-ratio",
            type=float,
            default=16.0,
            help="Maximum anisotropic kernel axis ratio",
        )
        parser.add_argument(
            "--anisotropy-scale",
            type=float,
            default=2.0,
            help="Anisotropic kernel width multiplier; wider kernels improve continuity",
        )
        parser.add_argument(
            "--anisotropy-min-neighbors",
            type=int,
            default=4,
            help="Minimum neighbors required for anisotropy; sparse particles remain isotropic",
        )
        parser.add_argument(
            "--anisotropy-strength",
            type=float,
            default=0.95,
            help="Blend from isotropic to fully anisotropic kernels",
        )
        parser.add_argument("--mesh-smooth-iterations", type=int, default=1)
        parser.add_argument(
            "--anisotropic",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Use anisotropic kernels for surface reconstruction",
        )
        parser.add_argument(
            "--extrapolate-into-colliders",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Extend the water surface into tank colliders",
        )
        parser.add_argument("--show-particles", action="store_true")
        return parser


if __name__ == "__main__":
    viewer, args = newton.examples.init(Example.create_parser())
    newton.examples.run(Example(viewer, args), args)
