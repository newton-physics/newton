# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Change sparse MPM grid resolution and particle support size in a moving bowl.

Requires CUDA for the sparse grid. Runtime changes preserve particle count,
mass, positions, velocities, and constitutive history. Radius changes affect
support volume and density; they do not resample the material.

Command: python -m newton.examples mpm_bowl
Change resolution: python -m newton.examples mpm_bowl --resize-time 3 --resize-voxel-size 0.06
"""

import math
from argparse import BooleanOptionalAction

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import SolverImplicitMPM


@wp.kernel(enable_backward=False)
def move_bowl(
    clock: wp.array[float],
    time_offset: float,
    frequency: float,
    amplitude: float,
    tilt: float,
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
):
    time = clock[0] + time_offset
    omega = 2.0 * wp.pi * frequency
    phase = omega * time
    position = wp.vec3(amplitude * wp.sin(phase), 0.6 * amplitude * wp.sin(0.7 * phase), 0.85)
    rotation = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), tilt * wp.sin(phase))
    linear = wp.vec3(amplitude * omega * wp.cos(phase), 0.42 * amplitude * omega * wp.cos(0.7 * phase), 0.0)
    angular = wp.vec3(0.0, tilt * omega * wp.cos(phase), 0.0)
    body_q[0] = wp.transform(position, rotation)
    body_qd[0] = wp.spatial_vector(linear, angular)


@wp.kernel(enable_backward=False)
def advance_clock(clock: wp.array[float], dt: float):
    clock[0] += dt


def _make_bowl_mesh(radius: float, thickness: float, *, rings: int = 16, segments: int = 64) -> newton.Mesh:
    """Create a watertight hemispherical shell with inward-facing cavity normals."""
    vertices = []
    indices = []
    for surface, r in enumerate((radius, radius + thickness)):
        pole = len(vertices)
        vertices.append((0.0, 0.0, -r))
        for ring in range(1, rings + 1):
            theta = 0.5 * math.pi * ring / rings
            for segment in range(segments):
                phi = 2.0 * math.pi * segment / segments
                vertices.append(
                    (r * math.sin(theta) * math.cos(phi), r * math.sin(theta) * math.sin(phi), -r * math.cos(theta))
                )
        faces = []
        for segment in range(segments):
            next_segment = (segment + 1) % segments
            faces.append((pole, pole + 1 + segment, pole + 1 + next_segment))
        for ring in range(rings - 1):
            lower = pole + 1 + ring * segments
            upper = lower + segments
            for segment in range(segments):
                next_segment = (segment + 1) % segments
                faces.extend(
                    (
                        (lower + segment, upper + segment, upper + next_segment),
                        (lower + segment, upper + next_segment, lower + next_segment),
                    )
                )
        for a, b, c in faces:
            indices.extend((a, b, c) if surface == 0 else (a, c, b))

    inner_rim = 1 + (rings - 1) * segments
    outer_rim = 2 + rings * segments + (rings - 1) * segments
    for segment in range(segments):
        next_segment = (segment + 1) % segments
        indices.extend((inner_rim + segment, outer_rim + segment, outer_rim + next_segment))
        indices.extend((inner_rim + segment, outer_rim + next_segment, inner_rim + next_segment))
    return newton.Mesh(vertices=np.asarray(vertices, dtype=np.float32), indices=np.asarray(indices, dtype=np.int32))


class Example:
    def __init__(self, viewer, args):
        if args.fps <= 0.0 or args.substeps < 1:
            raise ValueError("fps and substeps must be positive.")
        if not 0.0 < args.particle_radius < 0.1 or args.voxel_size <= 0.0:
            raise ValueError("particle-radius must be between 0 and 0.1 m and voxel-size must be positive.")
        if args.iterations < 1:
            raise ValueError("Solver iteration counts must be positive.")
        if args.max_active_cell_count < 128:
            raise ValueError("max-active-cell-count must be at least 128 for the bowl's reserved grid hierarchy.")

        self.viewer = viewer
        self.frame_dt = 1.0 / args.fps
        self.sim_substeps = args.substeps
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.bowl_radius = 0.6
        self.particle_radius = args.particle_radius
        self.motion_frequency = args.motion_frequency
        self.motion_amplitude = args.motion_amplitude
        self.motion_tilt = args.motion_tilt
        self.resize_time = args.resize_time
        self.resize_voxel_size = args.resize_voxel_size
        self._resized = False

        builder = newton.ModelBuilder()
        SolverImplicitMPM.register_custom_attributes(builder)
        bowl = builder.add_body(
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.85), wp.quat_identity()), is_kinematic=True, label="bowl"
        )
        builder.add_shape_mesh(
            bowl,
            mesh=_make_bowl_mesh(self.bowl_radius, 0.04),
            cfg=newton.ModelBuilder.ShapeConfig(mu=0.5, density=0.0, margin=0.002),
            color=(0.25, 0.5, 0.65),
            label="bowl_shell",
        )
        self._emit_particles(builder)
        self.model = builder.finalize()
        if not self.model.device.is_cuda:
            raise RuntimeError("The moving-bowl example requires CUDA for its sparse MPM grid.")
        config = SolverImplicitMPM.Config(
            grid_type="sparse",
            voxel_size=args.voxel_size,
            max_active_cell_count=args.max_active_cell_count,
            max_leaf_node_count=128,
            max_lower_node_count=64,
            max_upper_node_count=16,
            collider_basis="S2",
            solver="gs",
            warmstart_mode="particles",
            max_iterations=args.iterations,
        )
        self.mpm_solver = SolverImplicitMPM(self.model, config=config)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        # Keep start poses and positions available for the swept collision query.
        self.state_prev = self.model.state()
        self.sim_clock = wp.zeros(1, dtype=float, device=self.model.device)
        self._move_bowl(self.state_0, 0.0)
        self._move_bowl(self.state_1, 0.0)

        self.viewer.set_model(self.model)
        self.viewer.show_particles = True
        self.viewer.set_camera(pos=wp.vec3(1.7, -2.1, 2.3), pitch=-30.0, yaw=130.0)
        if hasattr(viewer, "register_ui_callback"):
            viewer.register_ui_callback(self.render_ui, position="side")
        self.use_cuda_graph = args.use_cuda_graph and self.sim_substeps % 2 == 0
        self.graph = None
        self.capture()

    def _emit_particles(self, builder):
        spacing = 2.0 * self.particle_radius
        axis = np.arange(-self.bowl_radius, self.bowl_radius, spacing)
        x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
        points = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
        keep = (np.linalg.norm(points, axis=1) < self.bowl_radius - 2.0 * self.particle_radius) & (points[:, 2] < -0.16)
        points = points[keep]
        if len(points) == 0:
            raise ValueError("The particle radius leaves no particles inside the bowl.")
        points += np.random.default_rng(42).uniform(-0.05 * spacing, 0.05 * spacing, points.shape)
        points[:, 2] += 0.85
        builder.add_particles(
            pos=points.tolist(),
            vel=np.zeros_like(points).tolist(),
            mass=[1500.0 * spacing**3] * len(points),
            radius=[self.particle_radius] * len(points),
        )

    def _move_bowl(self, state, time):
        wp.launch(
            move_bowl,
            dim=1,
            inputs=[
                self.sim_clock,
                time,
                self.motion_frequency,
                self.motion_amplitude,
                self.motion_tilt,
                state.body_q,
                state.body_qd,
            ],
            device=self.model.device,
        )

    def capture(self):
        """Warm up and capture the simulation without advancing the live state."""
        self.graph = None
        if not self.use_cuda_graph or not wp.is_conditional_graph_supported():
            return
        live_states = self.state_0, self.state_1, self.state_prev
        self.state_0, self.state_1, self.state_prev = (self.model.state() for _ in range(3))
        for source, dest in zip(live_states, (self.state_0, self.state_1, self.state_prev), strict=True):
            for name in ("particle_q", "particle_qd", "body_q", "body_qd"):
                wp.copy(getattr(dest, name), getattr(source, name))
            for name, array in vars(source.mpm).items():
                if isinstance(array, wp.array):
                    wp.copy(getattr(dest.mpm, name), array)
        try:
            self.simulate()
        finally:
            self.state_0, self.state_1, self.state_prev = live_states
            self.sim_clock.fill_(self.sim_time)
        self.mpm_solver.reset(self.state_0, flags=newton.StateFlags.BODY_Q)
        with wp.ScopedCapture(device=self.model.device) as capture:
            self.simulate()
        self.graph = capture.graph

    def set_voxel_size(self, voxel_size: float):
        """Rebuild the grid and its graph without restarting particle motion."""
        self.graph = None
        self.mpm_solver.set_voxel_size(voxel_size, state=self.state_0)
        self.capture()

    def set_particle_radius(self, radius: float):
        """Change the uniform support radius, keeping particle masses fixed."""
        if not math.isfinite(radius) or not 0.0 < radius < 0.1:
            raise ValueError("Particle radius must be between 0 and 0.1 m.")
        self.graph = None
        self.mpm_solver.set_particle_radius(radius)
        self.particle_radius = radius
        self.capture()

    def simulate(self):
        for substep in range(self.sim_substeps):
            time = substep * self.sim_dt
            self._move_bowl(self.state_0, time)
            wp.copy(self.state_prev.particle_q, self.state_0.particle_q)
            wp.copy(self.state_prev.body_q, self.state_0.body_q)
            self.state_0.clear_forces()
            self.mpm_solver.step(self.state_0, self.state_1, None, None, self.sim_dt)
            self._move_bowl(self.state_1, time + self.sim_dt)
            self.mpm_solver.project_outside(self.state_1, self.state_1, self.sim_dt, state_prev=self.state_prev)
            self.state_0, self.state_1 = self.state_1, self.state_0
        wp.launch(advance_clock, dim=1, inputs=[self.sim_clock, self.frame_dt], device=self.model.device)

    def step(self):
        if not self._resized and 0.0 <= self.resize_time <= self.sim_time:
            self.set_voxel_size(self.resize_voxel_size)
            self._resized = True
        if self.graph is None:
            self.simulate()
        else:
            wp.capture_launch(self.graph)
        self.mpm_solver.check_sparse_grid_rebuild_status()
        self.sim_time += self.frame_dt

    def test_post_step(self):
        """Check finite state and containment below the bowl rim."""
        positions = self.state_0.particle_q.numpy()
        assert np.isfinite(positions).all(), "Particle positions must remain finite"
        assert np.isfinite(self.state_0.particle_qd.numpy()).all(), "Particle velocities must remain finite"
        pose = self.state_0.body_q.numpy()[0]
        rotation = np.asarray(wp.quat_to_matrix(wp.quat(*pose[3:]))).reshape(3, 3)
        local = (positions - pose[:3]) @ rotation
        below_rim = local[:, 2] < -0.04
        # The mesh approximates the sphere with planar facets.
        assert np.all(np.linalg.norm(local[below_rim], axis=1) < self.bowl_radius + 0.01), (
            "Particles are outside the bowl below its rim"
        )

    def test_final(self):
        """Check final particle state and containment below the bowl rim."""
        self.test_post_step()

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def render_ui(self, imgui):
        imgui.text(f"Particles: {self.model.particle_count:,}")
        changed, voxel_size = imgui.slider_float("Voxel size [m]", self.mpm_solver.voxel_size, 0.06, 0.12)
        if changed:
            self.set_voxel_size(voxel_size)
        changed, radius = imgui.slider_float("Particle radius [m]", self.particle_radius, 0.0144, 0.0176)
        if changed:
            self.set_particle_radius(radius)
        imgui.text("Count and mass stay fixed; radius changes density.")

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--fps", type=float, default=60.0)
        parser.add_argument("--substeps", type=int, default=4)
        parser.add_argument(
            "--use-cuda-graph",
            action=BooleanOptionalAction,
            default=True,
            help="Capture simulation frames; requires an even substep count.",
        )
        parser.add_argument("--voxel-size", type=float, default=0.08)
        parser.add_argument("--max-active-cell-count", type=int, default=8192)
        parser.add_argument("--particle-radius", type=float, default=0.016)
        parser.add_argument("--iterations", type=int, default=100, help="Maximum MPM rheology iterations.")
        parser.add_argument("--motion-frequency", type=float, default=0.3, help="Bowl motion frequency [Hz].")
        parser.add_argument("--motion-amplitude", type=float, default=0.1, help="Bowl translation amplitude [m].")
        parser.add_argument("--motion-tilt", type=float, default=0.1, help="Bowl tilt amplitude [rad].")
        parser.add_argument(
            "--resize-time",
            type=float,
            default=-1.0,
            help="Change grid resolution after this many seconds; negative disables.",
        )
        parser.add_argument("--resize-voxel-size", type=float, default=0.06)
        return parser


if __name__ == "__main__":
    viewer, args = newton.examples.init(Example.create_parser())
    example = Example(viewer, args)
    newton.examples.run(example, args)
