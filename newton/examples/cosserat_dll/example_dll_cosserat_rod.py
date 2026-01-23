# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Cosserat rod simulation using DefKit DLL with Newton viewer.

This example demonstrates the iterative Position and Orientation Based
Cosserat Rods solver from the DefKit native library, visualized using
Newton's viewer infrastructure.

Command: uv run python -m newton.examples.cosserat_dll.example_dll_cosserat_rod
"""

import numpy as np
import warp as wp

import newton
import newton.examples

from .rod_state import create_straight_rod
from .simulation import CosseratRodSimulation


class Example:
    """Demo of Cosserat rod using DefKit DLL backend."""

    def __init__(self, viewer, args=None):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.substeps = 4

        self.viewer = viewer
        self.args = args

        # Get DLL path from args
        dll_path = args.dll_path if args and hasattr(args, "dll_path") else "unity_ref"

        # Rod parameters
        self.n_particles = 20
        self.segment_length = 0.05
        self.particle_radius = 0.015

        # Create rod state (horizontal cantilever along X axis)
        self.state = create_straight_rod(
            n_particles=self.n_particles,
            start_pos=(0.0, 0.0, 1.0),
            direction=(1.0, 0.0, 0.0),
            segment_length=self.segment_length,
            fix_first=True,
        )

        # Set stiffness parameters
        self.state.stretch_ks = 1.0
        self.state.bend_twist_ks_mult = 1.0
        self.state.bend_twist_ks[:, :3] = 1.0  # Uniform bending stiffness

        # Rest bend and twist parameters (in radians per segment)
        self.rest_bend_x = 0.0  # Bending around local X axis
        self.rest_bend_y = 0.0  # Bending around local Y axis
        self.rest_twist = 0.0   # Twist around local Z axis

        # Create simulation
        self.sim = CosseratRodSimulation(self.state, dll_path)
        self.sim.constraint_iterations = 8

        # Build Newton model for visualization
        builder = newton.ModelBuilder()
        builder.add_ground_plane()

        for i in range(self.n_particles):
            mass = 0.0 if i == 0 else 1.0
            pos = tuple(self.state.positions[i, :3])
            builder.add_particle(pos=pos, vel=(0.0, 0.0, 0.0), mass=mass, radius=self.particle_radius)

        self.model = builder.finalize()
        self.newton_state = self.model.state()

        self._sync_state()

        self.viewer.set_model(self.model)

        # Enable particle rendering
        if hasattr(self.viewer, "show_particles"):
            self.viewer.show_particles = True

        # Store initial state for reset
        self._initial_positions = self.state.positions.copy()
        self._initial_orientations = self.state.orientations.copy()
        self._initial_rest_darboux = self.state.rest_darboux.copy()

        # Keyboard state
        self._g_key_was_down = False
        self._r_key_was_down = False
        self.gravity_enabled = True

    def _sync_state(self):
        """Sync DLL state to Newton state for visualization."""
        positions_3d = self.state.get_positions_3d().astype(np.float32)
        positions_wp = wp.array(positions_3d, dtype=wp.vec3, device=self.model.device)
        self.newton_state.particle_q.assign(positions_wp)

    def _update_rest_darboux(self):
        """Update rest Darboux vector from bend/twist angles.

        The rest Darboux vector is represented as a quaternion encoding the
        intrinsic curvature between adjacent material frames. For small angles:
        - bend_x: curvature around local X axis (kappa1)
        - bend_y: curvature around local Y axis (kappa2)
        - twist: torsion around local Z axis (tau)

        The quaternion is constructed from the Darboux vector omega = (kappa1, kappa2, tau).
        """
        # Darboux vector components (curvature per unit length)
        kappa1 = self.rest_bend_x
        kappa2 = self.rest_bend_y
        tau = self.rest_twist

        # Convert to quaternion: q = exp(omega/2) for small angles
        # For the Cosserat rod formulation, rest Darboux is stored as quaternion
        # where (x,y,z) = omega/2 and w = sqrt(1 - |omega/2|^2)
        half_omega = np.array([kappa1, kappa2, tau], dtype=np.float32) * 0.5
        half_omega_norm_sq = np.dot(half_omega, half_omega)

        if half_omega_norm_sq < 1.0:
            w = np.sqrt(1.0 - half_omega_norm_sq)
        else:
            # Normalize if too large
            half_omega = half_omega / np.sqrt(half_omega_norm_sq)
            w = 0.0

        # Set rest Darboux for all edges
        for i in range(self.state.n_edges):
            self.state.rest_darboux[i, 0] = half_omega[0]
            self.state.rest_darboux[i, 1] = half_omega[1]
            self.state.rest_darboux[i, 2] = half_omega[2]
            self.state.rest_darboux[i, 3] = w

    def _handle_keyboard(self):
        """Handle keyboard input for interactive controls."""
        if not hasattr(self.viewer, "is_key_down"):
            return

        try:
            import pyglet.window.key as key
        except ImportError:
            return

        # Toggle gravity with G key
        g_down = self.viewer.is_key_down(key.G)
        if g_down and not self._g_key_was_down:
            self.gravity_enabled = not self.gravity_enabled
            if self.gravity_enabled:
                self.sim.set_gravity(0.0, 0.0, -9.81)
            else:
                self.sim.set_gravity(0.0, 0.0, 0.0)
            print(f"Gravity: {'ON' if self.gravity_enabled else 'OFF'}")
        self._g_key_was_down = g_down

        # Reset with R key
        r_down = self.viewer.is_key_down(key.R)
        if r_down and not self._r_key_was_down:
            self._reset()
            print("Reset simulation")
        self._r_key_was_down = r_down

    def _reset(self):
        """Reset simulation to initial state."""
        np.copyto(self.state.positions, self._initial_positions)
        np.copyto(self.state.predicted_positions, self._initial_positions)
        np.copyto(self.state.orientations, self._initial_orientations)
        np.copyto(self.state.predicted_orientations, self._initial_orientations)
        np.copyto(self.state.prev_orientations, self._initial_orientations)
        np.copyto(self.state.rest_darboux, self._initial_rest_darboux)
        self.state.velocities.fill(0)
        self.state.angular_velocities.fill(0)
        self.state.clear_forces()
        self.rest_bend_x = 0.0
        self.rest_bend_y = 0.0
        self.rest_twist = 0.0
        self.sim_time = 0.0
        self._sync_state()

    def step(self):
        self._handle_keyboard()

        sub_dt = self.frame_dt / self.substeps

        for _ in range(self.substeps):
            self.sim.step(sub_dt)

        self._sync_state()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.newton_state)

        # Draw rod segments as lines
        positions_3d = self.state.get_positions_3d().astype(np.float32)
        starts = wp.array(positions_3d[:-1], dtype=wp.vec3, device=self.model.device)
        ends = wp.array(positions_3d[1:], dtype=wp.vec3, device=self.model.device)
        colors = wp.array([[0.2, 0.6, 1.0]] * (self.n_particles - 1), dtype=wp.vec3, device=self.model.device)
        self.viewer.log_lines("/rod", starts, ends, colors)

        # Draw particles as spheres
        if hasattr(self.viewer, "log_spheres"):
            positions_wp = wp.array(positions_3d, dtype=wp.vec3, device=self.model.device)
            radii = wp.array([self.particle_radius] * self.n_particles, dtype=float, device=self.model.device)
            # Color: fixed particle (red) vs dynamic (cyan)
            particle_colors = []
            for i in range(self.n_particles):
                if self.state.inv_masses[i] == 0:
                    particle_colors.append([0.8, 0.2, 0.2])  # Red for fixed
                else:
                    particle_colors.append([0.2, 0.8, 0.8])  # Cyan for dynamic
            colors_wp = wp.array(particle_colors, dtype=wp.vec3, device=self.model.device)
            self.viewer.log_spheres("/particles", positions_wp, radii, colors_wp)

        self.viewer.end_frame()

    def gui(self, ui):
        ui.text("DefKit DLL Cosserat Rod")
        ui.text(f"Particles: {self.n_particles}")
        ui.text(f"Time: {self.sim_time:.2f}s")
        ui.separator()

        _, self.substeps = ui.slider_int("Substeps", self.substeps, 1, 16)
        _, self.sim.constraint_iterations = ui.slider_int("Iterations", self.sim.constraint_iterations, 1, 16)

        ui.separator()
        ui.text("Stiffness:")
        _, self.state.stretch_ks = ui.slider_float("Stretch Ks", self.state.stretch_ks, 0.0, 2.0)
        _, self.state.bend_twist_ks_mult = ui.slider_float("Bend/Twist Ks", self.state.bend_twist_ks_mult, 0.0, 2.0)

        ui.separator()
        ui.text("Rest Shape (intrinsic curvature):")
        changed_bend_x, self.rest_bend_x = ui.slider_float("Rest Bend X", self.rest_bend_x, -1.0, 1.0)
        changed_bend_y, self.rest_bend_y = ui.slider_float("Rest Bend Y", self.rest_bend_y, -1.0, 1.0)
        changed_twist, self.rest_twist = ui.slider_float("Rest Twist", self.rest_twist, -1.0, 1.0)

        # Update rest Darboux if any slider changed
        if changed_bend_x or changed_bend_y or changed_twist:
            self._update_rest_darboux()

        ui.separator()
        ui.text("Damping:")
        _, self.sim.position_damping = ui.slider_float("Position", self.sim.position_damping, 0.0, 0.1)
        _, self.sim.rotation_damping = ui.slider_float("Rotation", self.sim.rotation_damping, 0.0, 0.1)

        ui.separator()
        _, self.gravity_enabled = ui.checkbox("Gravity (G)", self.gravity_enabled)
        if self.gravity_enabled:
            self.sim.set_gravity(0.0, 0.0, -9.81)
        else:
            self.sim.set_gravity(0.0, 0.0, 0.0)

        ui.separator()
        ui.text("Controls:")
        ui.text("  G: Toggle gravity")
        ui.text("  R: Reset simulation")

        ui.separator()
        tip_pos = self.state.positions[-1, :3]
        ui.text(f"Tip: ({tip_pos[0]:.3f}, {tip_pos[1]:.3f}, {tip_pos[2]:.3f})")

        # Show segment length stats
        lengths = []
        for i in range(self.n_particles - 1):
            length = np.linalg.norm(self.state.positions[i + 1, :3] - self.state.positions[i, :3])
            lengths.append(length)
        avg_len = np.mean(lengths)
        max_err = max(abs(l - self.segment_length) / self.segment_length for l in lengths) * 100
        ui.text(f"Avg segment: {avg_len:.4f} (rest: {self.segment_length:.4f})")
        ui.text(f"Max length error: {max_err:.1f}%")

    def test_final(self):
        """Validation after simulation."""
        # Check tip has dropped under gravity
        tip_z = self.state.positions[-1, 2]
        assert tip_z < 0.9, f"Tip should drop below 0.9, got {tip_z}"

        # Check segment lengths are preserved within 20%
        for i in range(self.n_particles - 1):
            actual = np.linalg.norm(self.state.positions[i + 1, :3] - self.state.positions[i, :3])
            error = abs(actual - self.segment_length) / self.segment_length
            assert error < 0.2, f"Segment {i} length error {error * 100:.1f}% exceeds 20%"


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument(
        "--dll-path",
        type=str,
        default="unity_ref",
        help="Path to directory containing DefKit.dll and DefKitAdv.dll",
    )

    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
