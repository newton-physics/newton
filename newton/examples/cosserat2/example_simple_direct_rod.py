# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Minimal direct rod solver demo.

Uses the simplified direct solver (no tree structure, no complex physics).
For debugging the core algorithm.

Command: uv run python newton/examples/cosserat2/example_simple_direct_rod.py
"""

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.cosserat2.reference.direct_solver_simple import SimpleDirectRodSolver


class Example:
    """Simple demo of the simplified direct rod solver."""

    def __init__(self, viewer, args=None):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.substeps = 4  # Multiple substeps improve constraint satisfaction
        self.iterations = 2

        self.viewer = viewer
        self.args = args

        # Rod parameters
        self.n_particles = 10
        self.segment_length = 0.1
        self.particle_radius = 0.02

        # Create horizontal rod along X axis
        positions = np.zeros((self.n_particles, 3), dtype=np.float64)
        for i in range(self.n_particles):
            positions[i] = [i * self.segment_length, 0.0, 1.0]  # Start at z=1

        # Quaternion rotating Z to X (90 degrees around Y)
        q_y90 = np.array([0, np.sin(np.pi / 4), 0, np.cos(np.pi / 4)], dtype=np.float64)
        quaternions = np.tile(q_y90, (self.n_particles - 1, 1))

        rest_lengths = np.full(self.n_particles - 1, self.segment_length, dtype=np.float64)

        # Fix first particle position (cantilever root)
        # Note: edge_inv_mass[0] > 0 allows segment 0 to rotate at the root
        particle_inv_mass = np.ones(self.n_particles, dtype=np.float64)
        particle_inv_mass[0] = 0.0  # Fixed anchor position

        edge_inv_mass = np.ones(self.n_particles - 1, dtype=np.float64)
        # All segments can rotate (cantilever can bend at root)

        # Create simplified solver
        self.solver = SimpleDirectRodSolver(
            n_particles=self.n_particles,
            positions=positions,
            quaternions=quaternions,
            rest_lengths=rest_lengths,
            particle_inv_mass=particle_inv_mass,
            edge_inv_mass=edge_inv_mass,
            stretch_stiffness=1.0,  # Full stretch stiffness
            bend_stiffness=0.5,     # Moderate bending stiffness
        )

        # Gravity
        self.gravity = np.array([0.0, 0.0, -9.81])
        self.gravity_enabled = True

        # Build Newton model for visualization
        builder = newton.ModelBuilder()
        builder.add_ground_plane()

        for i in range(self.n_particles):
            mass = 0.0 if i == 0 else 1.0
            pos = tuple(positions[i])
            builder.add_particle(pos=pos, vel=(0.0, 0.0, 0.0), mass=mass, radius=self.particle_radius)

        self.model = builder.finalize()
        self.state = self.model.state()

        # Sync state
        self._sync_state_from_solver()

        self.viewer.set_model(self.model)
        self.viewer.show_particles = True

        # UI state
        self._gravity_key_was_down = False
        self._reset_key_was_down = False
        self._initial_positions = positions.copy()
        self._initial_quaternions = quaternions.copy()

    def _sync_state_from_solver(self):
        """Copy solver positions to Newton state for visualization."""
        positions_wp = wp.array(self.solver.positions.astype(np.float32), dtype=wp.vec3, device=self.model.device)
        self.state.particle_q.assign(positions_wp)

    def step(self):
        self._handle_keyboard_input()

        # Run simulation substeps
        sub_dt = self.frame_dt / self.substeps
        gravity = self.gravity if self.gravity_enabled else np.zeros(3)

        for _ in range(self.substeps):
            self.solver.step(sub_dt, gravity, iterations=self.iterations)

        # Sync for visualization
        self._sync_state_from_solver()

        self.sim_time += self.frame_dt

    def _handle_keyboard_input(self):
        if not hasattr(self.viewer, "is_key_down"):
            return

        try:
            import pyglet.window.key as key
        except ImportError:
            return

        # Gravity toggle
        g_down = self.viewer.is_key_down(key.G)
        if g_down and not self._gravity_key_was_down:
            self.gravity_enabled = not self.gravity_enabled
            print(f"Gravity: {'ON' if self.gravity_enabled else 'OFF'}")
        self._gravity_key_was_down = g_down

        # Reset
        r_down = self.viewer.is_key_down(key.R)
        if r_down and not self._reset_key_was_down:
            self.solver.positions[:] = self._initial_positions
            self.solver.quaternions[:] = self._initial_quaternions
            self.solver.velocities[:] = 0
            self.solver.lambda_sum[:] = 0
            # Re-sync segments
            for i in range(self.solver.n_seg):
                self.solver.seg_pos[i] = 0.5 * (self.solver.positions[i] + self.solver.positions[i + 1])
                self.solver.seg_quat[i] = self.solver.quaternions[i]
            self._sync_state_from_solver()
            self.sim_time = 0.0
            print("Reset simulation")
        self._reset_key_was_down = r_down

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state)

        # Draw rod segments as lines
        starts = wp.array(self.solver.positions[:-1].astype(np.float32), dtype=wp.vec3, device=self.model.device)
        ends = wp.array(self.solver.positions[1:].astype(np.float32), dtype=wp.vec3, device=self.model.device)
        colors = wp.array([[0.2, 0.6, 1.0]] * (self.n_particles - 1), dtype=wp.vec3, device=self.model.device)
        self.viewer.log_lines("/rod", starts, ends, colors)

        self.viewer.end_frame()

    def gui(self, ui):
        ui.text("Simplified Direct Rod Solver")
        ui.text(f"Particles: {self.n_particles}")
        ui.separator()

        _, self.gravity_enabled = ui.checkbox("Gravity (G)", self.gravity_enabled)
        _, self.substeps = ui.slider_int("Substeps", self.substeps, 1, 16)
        _, self.iterations = ui.slider_int("Iterations", self.iterations, 1, 8)

        ui.separator()
        _, self.solver.stretch_compliance_base = ui.slider_float(
            "Stretch Compliance", self.solver.stretch_compliance_base, 0.0, 0.001
        )
        _, self.solver.bend_compliance_base = ui.slider_float("Bend Compliance", self.solver.bend_compliance_base, 0.0, 100.0)

        ui.separator()
        ui.text("Controls:")
        ui.text("  G: Toggle gravity")
        ui.text("  R: Reset")

        ui.separator()
        ui.text("Debug Info:")
        tip_pos = self.solver.positions[-1]
        ui.text(f"  Tip pos: ({tip_pos[0]:.3f}, {tip_pos[1]:.3f}, {tip_pos[2]:.3f})")

        # Show segment lengths
        lengths = []
        for i in range(self.n_particles - 1):
            l = np.linalg.norm(self.solver.positions[i + 1] - self.solver.positions[i])
            lengths.append(l)
        ui.text(f"  Segment lengths: {[f'{l:.3f}' for l in lengths[:3]]}...")

    def test_final(self):
        """Validation after simulation."""
        # Check tip has dropped under gravity
        tip_z = self.solver.positions[-1, 2]
        assert tip_z < 0.9, f"Tip should drop below 0.9, got {tip_z}"

        # Check segment lengths preserved within 10%
        for i in range(self.n_particles - 1):
            actual = np.linalg.norm(self.solver.positions[i + 1] - self.solver.positions[i])
            error = abs(actual - self.segment_length) / self.segment_length
            assert error < 0.1, f"Segment {i} length error {error*100:.1f}% exceeds 10%"


if __name__ == "__main__":
    viewer, args = newton.examples.init()

    if isinstance(viewer, newton.viewer.ViewerGL):
        viewer.show_particles = True

    example = Example(viewer, args)
    newton.examples.run(example, args)
