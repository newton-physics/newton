# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import warnings

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import SolverImplicitMPM


@wp.kernel
def apply_twist(
    indices: wp.array[int],
    rel_pos: wp.array[wp.vec3],
    out_pos: wp.array[wp.vec3],
    out_vel: wp.array[wp.vec3],
    out_vel_grad: wp.array[wp.mat33],
    center: wp.vec3,
    angle: float,
    speed: float,
):
    tid = wp.tid()
    idx = indices[tid]
    r = rel_pos[tid]

    # Rotate around X
    s = wp.sin(angle)
    c = wp.cos(angle)

    ry = r[1] * c - r[2] * s
    rz = r[1] * s + r[2] * c

    # Update position
    out_pos[idx] = center + wp.vec3(r[0], ry, rz)

    # Update velocity (v = w x r)
    # w = (speed, 0, 0)
    out_vel[idx] = wp.vec3(0.0, -rz * speed, ry * speed)
    out_vel_grad[idx] = wp.skew(wp.vec3(speed, 0.0, 0.0))


class Example:
    """Elastic beam twisted at one end using kinematic MPM boundary particles."""

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

        # Register MPM custom attributes before adding particles
        SolverImplicitMPM.register_custom_attributes(builder)

        Example.emit_particles(builder, options)

        self.model = builder.finalize()
        self.model.set_gravity(options.gravity)

        # Copy all remaining CLI arguments to MPM options
        mpm_options = SolverImplicitMPM.Config()
        mpm_options.warmstart_mode = "particles"
        for key in vars(options):
            if hasattr(mpm_options, key):
                setattr(mpm_options, key, getattr(options, key))

        # Copy per-particle material options to model custom attributes
        mpm_particle_attrs = [
            "young_modulus",
            "poisson_ratio",
            "damping",
        ]
        for key in mpm_particle_attrs:
            if hasattr(options, key) and hasattr(self.model.mpm, key):
                getattr(self.model.mpm, key).fill_(getattr(options, key))
        self.model.mpm.tensile_yield_ratio.fill_(1.0)

        q_np = self.model.particle_q.numpy()
        fixed_mask = q_np[:, 0] < options.emit_lo[0] + 0.5 * options.voxel_size

        # Clamp right end (twist)
        twist_mask = q_np[:, 0] > options.emit_hi[0] - 0.5 * options.voxel_size

        all_fixed = np.logical_or(fixed_mask, twist_mask)
        fixed_indices = wp.array(np.flatnonzero(all_fixed), dtype=int, device=self.model.device)
        self.model.particle_mass[fixed_indices].fill_(0.0)

        # Setup twist
        self.twist_indices = wp.array(np.flatnonzero(twist_mask), dtype=int, device=self.model.device)
        twist_pos = q_np[twist_mask]
        self.twist_center = np.mean(twist_pos, axis=0)
        self.twist_rel_pos = wp.array(twist_pos - self.twist_center, dtype=wp.vec3, device=self.model.device)

        # Rotate 360 degrees over 1000 frames
        self.twist_frames = 1000
        self.twist_speed = (2.0 * np.pi) / (self.twist_frames * self.frame_dt)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        # Initialize MPM solver
        self.solver = SolverImplicitMPM(self.model, mpm_options)

        self.viewer.set_model(self.model)

        # Position camera for a 3/4 elevated view of the beam
        if hasattr(self.viewer, "set_camera"):
            self.viewer.set_camera(pos=wp.vec3(2.5, -5.0, 2.5), pitch=-15.0, yaw=90.0)

        if isinstance(self.viewer, newton.viewer.ViewerGL):
            self.viewer.register_ui_callback(self.render_ui, position="side")

        self.show_stress = True
        self.viewer.show_particles = True

        self.particle_colors = wp.full(
            shape=self.model.particle_count, value=wp.vec3(0.1, 0.1, 0.2), device=self.model.device
        )

        self.capture()

    def capture(self):
        self.graph = None
        if wp.get_device().is_cuda and self.solver.grid_type == "fixed":
            if self.sim_substeps % 2 != 0:
                warnings.warn("Sim substeps must be even for graph capture of MPM step", stacklevel=2)
            else:
                with wp.ScopedCapture() as capture:
                    self.simulate()
                self.graph = capture.graph

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.solver.step(self.state_0, self.state_1, None, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        frame = int(self.sim_time * self.fps + 0.5)
        if frame < self.twist_frames:
            angle = self.twist_speed * self.sim_time
            wp.launch(
                kernel=apply_twist,
                dim=self.twist_indices.shape[0],
                inputs=[
                    self.twist_indices,
                    self.twist_rel_pos,
                    self.state_0.particle_q,
                    self.state_0.particle_qd,
                    self.state_0.mpm.particle_qd_grad,
                    wp.vec3(float(self.twist_center[0]), float(self.twist_center[1]), float(self.twist_center[2])),
                    angle,
                    self.twist_speed,
                ],
                device=self.model.device,
            )
        elif frame == self.twist_frames:
            angle = self.twist_speed * (self.twist_frames * self.frame_dt)
            wp.launch(
                kernel=apply_twist,
                dim=self.twist_indices.shape[0],
                inputs=[
                    self.twist_indices,
                    self.twist_rel_pos,
                    self.state_0.particle_q,
                    self.state_0.particle_qd,
                    self.state_0.mpm.particle_qd_grad,
                    wp.vec3(float(self.twist_center[0]), float(self.twist_center[1]), float(self.twist_center[2])),
                    angle,
                    0.0,
                ],
                device=self.model.device,
            )

        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)

        if self.show_stress:
            stresses = self.state_0.mpm.particle_stress.numpy()

            pressure = np.trace(stresses, axis1=1, axis2=2) / 3.0
            dev_part = stresses - pressure.reshape(-1, 1, 1) * np.eye(3).reshape(1, 3, 3)
            dev_stress = np.linalg.norm(dev_part, axis=(1, 2))

            s = dev_stress
            s_min, s_max = np.percentile(s, [10, 90])
            s_range = s_max - s_min if s_max > s_min else 1.0
            s_norm = np.clip((s - s_min) / s_range, 0.0, 1.0)

            # Vectorized color mapping: blue->green (v < 0.5), green->red (v >= 0.5)
            colors_np = np.zeros((s_norm.shape[0], 3), dtype=np.float32)

            mask = s_norm < 0.5
            t1 = s_norm[mask] / 0.5
            # Blue to green
            colors_np[mask, 0] = 0.0  # R
            colors_np[mask, 1] = t1  # G: from 0 to 1
            colors_np[mask, 2] = 1.0 - t1  # B: from 1 to 0

            mask2 = ~mask
            t2 = (s_norm[mask2] - 0.5) / 0.5
            # Green to red
            colors_np[mask2, 0] = t2  # R: from 0 to 1
            colors_np[mask2, 1] = 1.0 - t2  # G: from 1 to 0
            colors_np[mask2, 2] = 0.0  # B: stays at 0

            self.particle_colors.assign(colors_np)
        else:
            self.particle_colors.fill_(
                wp.vec3(0.2, 0.2, 0.4),
            )

        self.viewer.log_points(
            name="/model/particles",
            points=self.state_0.particle_q,
            radii=self.model.particle_radius,
            colors=self.particle_colors,
            hidden=not self.show_stress and not self.viewer.show_particles,
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
            "all particles remain near the beam",
            lambda q, qd: wp.length(q) < 10.0,
        )

    def render_ui(self, imgui):
        _changed, self.show_stress = imgui.checkbox("Show Stress", self.show_stress)

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()

        # Scene configuration
        parser.add_argument("--emit-lo", type=float, nargs=3, default=[0.0, 0.0, 0.0])
        parser.add_argument("--emit-hi", type=float, nargs=3, default=[5.0, 1, 1.0])
        parser.add_argument("--gravity", type=float, nargs=3, default=[0, 0, -10])
        parser.add_argument("--fps", type=float, default=240.0)
        parser.add_argument("--substeps", type=int, default=1)

        parser.add_argument("--density", "-rho", type=float, default=1000.0)
        parser.add_argument("--young-modulus", "-ym", type=float, default=5.0e6)
        parser.add_argument("--poisson-ratio", "-nu", type=float, default=0.45)
        parser.add_argument("--damping", "-d", type=float, default=0.001)
        parser.add_argument(
            "--solver",
            "-s",
            type=str,
            default="cr",
        )
        parser.add_argument("--integration-scheme", "-is", type=str, default="pic", choices=["pic", "gimp"])

        parser.add_argument("--strain-basis", "-sb", type=str, default="P1d")
        parser.add_argument("--velocity-basis", "-vb", type=str, default="Q1")

        parser.add_argument("--max-iterations", "-it", type=int, default=250)
        parser.add_argument("--tolerance", "-tol", type=float, default=1.0e-6)
        parser.add_argument("--voxel-size", "-dx", type=float, default=0.25)

        return parser

    @staticmethod
    def emit_particles(builder: newton.ModelBuilder, args):
        density = args.density
        voxel_size = args.voxel_size

        particles_per_cell = 3
        particle_lo = np.array(args.emit_lo)
        particle_hi = np.array(args.emit_hi)
        particle_res = np.array(
            np.ceil(particles_per_cell * (particle_hi - particle_lo) / voxel_size),
            dtype=int,
        )

        cell_size = (particle_hi - particle_lo) / (particle_res + 1)
        cell_volume = np.prod(cell_size)

        radius = np.cbrt(cell_volume) * 0.5
        mass = cell_volume * density

        builder.add_particle_grid(
            pos=wp.vec3(particle_lo),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=particle_res[0] + 1,
            dim_y=particle_res[1] + 1,
            dim_z=particle_res[2] + 1,
            cell_x=cell_size[0],
            cell_y=cell_size[1],
            cell_z=cell_size[2],
            mass=mass,
            jitter=0.0 * radius,
            radius_mean=radius,
        )


if __name__ == "__main__":
    parser = Example.create_parser()

    viewer, args = newton.examples.init(parser)

    example = Example(viewer, args)

    newton.examples.run(example, args)
