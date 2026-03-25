# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import SolverImplicitMPM


@wp.kernel
def _compute_compression_colors(
    Jp: wp.array(dtype=float),
    colors: wp.array(dtype=wp.vec3),
    Jp_min: float,
    Jp_inv_range: float,
):
    i = wp.tid()
    v = wp.clamp((Jp[i] - Jp_min) * Jp_inv_range, 0.0, 1.0)
    if v < 0.5:
        t = v / 0.5
        colors[i] = wp.vec3(0.0, t, 1.0 - t)
    else:
        t = (v - 0.5) / 0.5
        colors[i] = wp.vec3(t, 1.0 - t, 0.0)


class Example:
    """Snow ball rolling down a heightfield slope with per-particle snow rheology."""

    def __init__(self, viewer, options):
        # setup simulation parameters first
        self.fps = options.fps
        self.frame_dt = 1.0 / self.fps

        # group related attributes by prefix
        self.sim_time = 0.0
        self.sim_substeps = options.substeps
        self.sim_dt = self.frame_dt / self.sim_substeps

        # save a reference to the viewer
        self.viewer = viewer
        builder = newton.ModelBuilder()
        SolverImplicitMPM.register_custom_attributes(builder)

        # Terrain geometry parameters
        # Domain: 5m wide (x), 15m long (y)
        self.L_x = 5.0
        self.L_y = 20.0
        slope_angle_rad = np.radians(45.0)
        self.amplitude = np.tan(slope_angle_rad) * self.L_y / np.pi

        # Create heightfield
        res_x = 50
        res_y = 150

        # Grid coordinates
        # x from -L_x/2 to L_x/2
        # y from -L_y/2 to L_y/2
        self.hf_x = np.linspace(-self.L_x / 2, self.L_x / 2, res_x)
        self.hf_y = np.linspace(-self.L_y / 2, self.L_y / 2, res_y)

        # We want heightfield array of shape (res_x, res_y) corresponding to x and y
        X_hf, Y_hf = np.meshgrid(self.hf_x, self.hf_y, indexing="ij")
        Z_hf = self._get_terrain_z(Y_hf, X_hf)
        terrain_mesh = newton.Mesh.create_heightfield(
            heightfield=Z_hf,
            extent_x=self.L_x,
            extent_y=self.L_y,
            center_x=0.0,
            center_y=0.0,
            ground_z=np.min(Z_hf) - 2.0,
            compute_inertia=False,
        )

        # Add terrain body
        terrain_body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()), label="terrain")
        builder.add_shape_mesh(
            body=terrain_body,
            mesh=terrain_mesh,
            cfg=newton.ModelBuilder.ShapeConfig(mu=options.friction_coeff, density=0.0),  # Static body
        )

        # Emit particles
        self.emit_avalanche_particles(builder, options, Z_hf, self.hf_x, self.hf_y)

        self.model = builder.finalize()
        self.model.set_gravity(options.gravity)

        # Copy all remaining CLI arguments to MPM options
        mpm_options = SolverImplicitMPM.Config()
        for key in vars(options):
            if hasattr(mpm_options, key):
                setattr(mpm_options, key, getattr(options, key))

        # Create MPM model from Newton model
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        # Initialize material properties
        self.init_materials(options, self.model)

        # Initialize MPM solver and add supplemental state variables
        self.solver = SolverImplicitMPM(self.model, mpm_options)

        self.viewer.set_model(self.model)

        # Position camera for an elevated side view showing the slope and rolling ball
        if hasattr(self.viewer, "set_camera"):
            self.viewer.set_camera(pos=wp.vec3(12.0, -8.0, 14.0), pitch=-30.0, yaw=145.0)

        if hasattr(self.viewer, "register_ui_callback"):
            self.viewer.register_ui_callback(self.render_ui, position="side")

        self.viewer.show_particles = True
        self.show_compression = True

        self.capture()

    def _get_terrain_z(self, y, x):
        """Terrain height field"""
        return -self.amplitude * np.sin(np.pi * y / self.L_y) * (1.0 + 0.1 * np.sin(4 * np.pi * y / self.L_y))

    def emit_avalanche_particles(self, builder, args, heightfield, hf_x, hf_y):
        density = args.density
        voxel_size = args.voxel_size

        # We generate particles manually to conform to terrain
        # Thickness of snow pack
        thickness = 0.8  # meter

        particles_per_cell_dim = 2
        spacing = voxel_size / particles_per_cell_dim

        # Create grid of particles
        # We iterate over x and y, interpolate z from heightfield, and stack particles up to thickness

        # Resample for particles
        # Domain bounds
        x_min, x_max = -self.L_x / 2, self.L_x / 2
        y_min, y_max = -self.L_y / 2, self.L_y / 2

        # Margin for particles to avoid exact edge
        margin = 0.1

        px = np.arange(x_min + margin, x_max - margin, spacing)
        py = np.arange(y_min + margin, y_max - margin, spacing)

        # Grid of particle x, y
        PX, PY = np.meshgrid(px, py, indexing="xy")  # Shape (npy, npx)

        # Interpolate height Z from heightfield
        # Given X, Y, find indices in hf_x, hf_y
        # hf_x is linspace
        dx = hf_x[1] - hf_x[0]
        dy = hf_y[1] - hf_y[0]

        # Indices (float)
        ix = (PX - hf_x[0]) / dx
        iy = (PY - hf_y[0]) / dy

        # Bilinear interpolation of Z
        # heightfield shape (res_x, res_y) corresponds to (hf_x, hf_y)
        # So Z[ix, iy]

        # Flatten
        PX = PX.flatten()
        PY = PY.flatten()
        ix = ix.flatten()
        iy = iy.flatten()

        x0 = np.floor(ix).astype(int)
        x1 = x0 + 1
        y0 = np.floor(iy).astype(int)
        y1 = y0 + 1

        # Clip
        x0 = np.clip(x0, 0, heightfield.shape[0] - 1)
        x1 = np.clip(x1, 0, heightfield.shape[0] - 1)
        y0 = np.clip(y0, 0, heightfield.shape[1] - 1)
        y1 = np.clip(y1, 0, heightfield.shape[1] - 1)

        wx = ix - x0
        wy = iy - y0

        # heightfield is indexed [x, y]
        z00 = heightfield[x0, y0]
        z10 = heightfield[x1, y0]
        z01 = heightfield[x0, y1]
        z11 = heightfield[x1, y1]

        z0 = z00 * (1 - wx) + z10 * wx
        z1 = z01 * (1 - wx) + z11 * wx
        PZ_base = z0 * (1 - wy) + z1 * wy

        # Now stack particles
        num_layers = int(thickness / spacing)

        all_pos = []

        for layer in range(num_layers):
            z_offset = (layer + 0.5) * spacing
            pz = PZ_base + z_offset

            # Stack into (N, 3)
            pos_layer = np.stack([PX, PY, pz], axis=1)
            all_pos.append(pos_layer)

        if not all_pos:
            return

        all_pos = np.concatenate(all_pos, axis=0)

        # add jitter: uniformly sample displacement in each axis in [-0.5*radius, 0.5*radius]
        jitter_scale = 1.0 * (spacing / 2.0)  # radius = spacing/2
        rng = np.random.default_rng(seed=423)
        all_pos += rng.uniform(-jitter_scale, jitter_scale, size=all_pos.shape)

        # Calculate mass
        # Total volume approx = Area * thickness
        # Particle volume = spacing^3
        particle_mass = (spacing**3) * density

        print(f"Generating {len(all_pos)} particles...")
        builder.add_particles(
            pos=all_pos.tolist(),
            vel=np.zeros_like(all_pos).tolist(),
            mass=[particle_mass] * len(all_pos),
            radius=[spacing / 2.0] * len(all_pos),
        )

        # Snow ball
        # Emit a sphere of particles
        sphere_radius = 1.0
        sphere_center = np.array([0.0, -5.0, 15.0 + sphere_radius])
        dim = int(2.0 * sphere_radius / spacing + 1)
        px = np.arange(dim) * spacing
        py = np.arange(dim) * spacing
        pz = np.arange(dim) * spacing
        points = np.stack(np.meshgrid(px, py, pz, indexing="ij")).reshape(3, -1).T

        # Offset so the grid is centered, then translate to sphere_center
        points -= np.array([dim - 1, dim - 1, dim - 1]) * spacing * 0.5
        dist = np.linalg.norm(points, axis=1)
        mask = dist <= sphere_radius
        points = points[mask] + sphere_center

        # Add jitter
        rng = np.random.default_rng(42)
        points += (rng.random(points.shape) - 0.5) * spacing

        builder.add_particles(
            pos=points.tolist(),
            vel=np.zeros_like(points).tolist(),
            mass=[particle_mass] * points.shape[0],
            radius=[spacing / 2.0] * points.shape[0],
        )

    def init_materials(self, options, model: newton.Model):
        # Identify particles
        q_np = self.state_0.particle_q.numpy()

        # Boundary particles: sides
        # Boundary particles: mark kinematic if close to domain edge
        boundary_width = 0.2
        boundary_mask = np.logical_or(
            np.abs(q_np[:, 0]) > (self.L_x / 2 - boundary_width),
            np.abs(q_np[:, 1]) > (self.L_y / 2 - boundary_width),
        )

        boundary_indices = wp.array(np.flatnonzero(boundary_mask), dtype=wp.int32, device=model.device)

        # Initialize Jp (plastic deformation gradient determinant or damage variable)
        # 1.0 = fully intact
        self.state_0.mpm.particle_Jp.fill_(0.975)

        # default parameters
        model.mpm.young_modulus.fill_(options.young_modulus)
        model.mpm.poisson_ratio.fill_(options.poisson_ratio)
        model.mpm.friction.fill_(options.friction_coeff)
        model.mpm.damping.fill_(options.damping)
        model.mpm.yield_pressure.fill_(options.yield_pressure)
        model.mpm.tensile_yield_ratio.fill_(options.tensile_yield_ratio)
        model.mpm.yield_stress.fill_(options.yield_stress)
        model.mpm.hardening.fill_(options.hardening)
        model.mpm.dilatancy.fill_(options.dilatancy)

        # Set boundary particles as kinematic (zero mass)
        model.particle_mass[boundary_indices].fill_(0.0)

        self.boundary_indices = boundary_indices

        self.particle_colors = wp.full(shape=model.particle_count, value=wp.vec3(0.8, 0.8, 0.9), device=model.device)

    def capture(self):
        self.graph = None
        if wp.get_device().is_cuda and self.solver.grid_type == "fixed":
            if self.sim_substeps % 2 != 0:
                wp.utils.warn("Sim substeps must be even for graph capture of MPM step")
            else:
                with wp.ScopedCapture() as capture:
                    self.simulate()
                self.graph = capture.graph

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.solver.step(self.state_0, self.state_1, None, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)

        # Color based on Jp
        if self.show_compression:
            Jp_min = 0.5
            Jp_max = 2.0
            wp.launch(
                _compute_compression_colors,
                dim=self.model.particle_count,
                inputs=[self.state_0.mpm.particle_Jp, self.particle_colors, Jp_min, 1.0 / (Jp_max - Jp_min)],
                device=self.model.device,
            )
        else:
            self.particle_colors.fill_(wp.vec3(0.8, 0.8, 0.9))

        self.viewer.log_points(
            name="/model/particles",
            points=self.state_0.particle_q,
            radii=self.model.particle_radius,
            colors=self.particle_colors,
            hidden=not self.show_compression and not self.viewer.show_particles,
        )

        self.viewer.end_frame()

    def test_final(self):
        newton.examples.test_particle_state(
            self.state_0,
            "all particles have finite positions",
            lambda q, qd: wp.length(q) < 1.0e6,
        )
        newton.examples.test_particle_state(
            self.state_0,
            "all particles have finite velocities",
            lambda q, qd: wp.length(qd) < 1.0e6,
        )
        newton.examples.test_particle_state(
            self.state_0,
            "all particles are within the terrain domain",
            lambda q, qd: q[2] > -20.0 and q[2] < 30.0,
        )

    def render_ui(self, imgui):
        _changed, self.show_compression = imgui.checkbox("Show Compression", self.show_compression)

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()

        # Scene configuration
        parser.add_argument("--gravity", type=float, nargs=3, default=[0, 0, -9.81])
        parser.add_argument("--fps", type=float, default=60.0)
        parser.add_argument("--substeps", type=int, default=1)

        # Add MPM-specific arguments
        parser.add_argument("--density", type=float, default=400.0)
        parser.add_argument("--young-modulus", "-ym", type=float, default=1.4e6)
        parser.add_argument("--poisson-ratio", "-nu", type=float, default=0.3)
        parser.add_argument("--friction-coeff", "-mu", type=float, default=0.5)
        parser.add_argument("--damping", type=float, default=0.01)
        parser.add_argument("--yield-pressure", "-yp", type=float, default=1.4e6)
        parser.add_argument("--tensile-yield-ratio", "-tyr", type=float, default=0.2)
        parser.add_argument("--yield-stress", "-ys", type=float, default=0.0e5)
        parser.add_argument("--hardening", type=float, default=5.0)
        parser.add_argument("--dilatancy", type=float, default=1.0)

        parser.add_argument(
            "--solver",
            "-s",
            type=str,
            default="cg+gauss-seidel",
            choices=["gauss-seidel", "jacobi", "cg", "cg+jacobi", "cg+gauss-seidel"],
        )

        parser.add_argument("--strain-basis", "-sb", type=str, default="P0")
        parser.add_argument("--max-iterations", "-it", type=int, default=150)
        parser.add_argument("--tolerance", "-tol", type=float, default=1.0e-4)
        parser.add_argument("--voxel-size", "-dx", type=float, default=0.1)  # Increased voxel size for larger domain

        return parser


if __name__ == "__main__":
    parser = Example.create_parser()

    viewer, args = newton.examples.init(parser)

    example = Example(viewer, args)

    newton.examples.run(example, args)
