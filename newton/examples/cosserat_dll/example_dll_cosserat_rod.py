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

        # Store initial state for reset
        self._initial_positions = self.state.positions.copy()
        self._initial_orientations = self.state.orientations.copy()

        # Keyboard state
        self._g_key_was_down = False
        self._r_key_was_down = False
        self.gravity_enabled = True

    def _sync_state(self):
        """Sync DLL state to Newton state for visualization."""
        positions_3d = self.state.get_positions_3d().astype(np.float32)
        positions_wp = wp.array(positions_3d, dtype=wp.vec3, device=self.model.device)
        self.newton_state.particle_q.assign(positions_wp)

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
        self.state.velocities.fill(0)
        self.state.angular_velocities.fill(0)
        self.state.clear_forces()
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
