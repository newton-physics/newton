# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Direct Position-Based Solver for Stiff Rods using DefKit DLL with Newton viewer.

This example demonstrates the Direct Position-Based Solver for Stiff Rods
(Deul et al.) from the DefKit native library. Unlike the iterative solver,
this uses a global banded matrix solve for better handling of stiff materials.

Command: uv run python -m newton.examples.cosserat_dll.example_dll_direct_cosserat_rod
"""

import numpy as np
import warp as wp

import newton
import newton.examples

from .rod_state import create_straight_rod
from .simulation_direct import DirectCosseratRodSimulation


class Example:
    """Demo of Direct Cosserat rod solver using DefKit DLL backend."""

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
        self.n_particles = 64
        self.segment_length = 0.05
        self.particle_radius = 0.01

        # Create rod state (horizontal cantilever along X axis)
        self.state = create_straight_rod(
            n_particles=self.n_particles,
            start_pos=(0.0, 0.0, 1.0),
            direction=(1.0, 0.0, 0.0),
            segment_length=self.segment_length,
            fix_first=True,
        )

        # Create direct simulation
        self.sim = DirectCosseratRodSimulation(self.state, dll_path)

        # Material parameters (modulus values controlled via GUI sliders)
        self.sim.radius = 0.01

        # Set uniform bend stiffness
        self.sim.set_bend_stiffness(1.0, 1.0, 1.0)

        # Rest curvature parameters (for GUI)
        self.rest_bend_x = 0.0
        self.rest_bend_y = 0.0
        self.rest_twist = 0.0

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

        # Keyboard state
        self._g_key_was_down = False
        self._r_key_was_down = False
        self.gravity_enabled = True

        # Movement/rotation parameters for first particle
        self.move_speed = 0.5  # units per second
        self.rotate_speed = 1.0  # radians per second

    def _sync_state(self):
        """Sync DLL state to Newton state for visualization."""
        positions_3d = self.state.get_positions_3d().astype(np.float32)
        positions_wp = wp.array(positions_3d, dtype=wp.vec3, device=self.model.device)
        self.newton_state.particle_q.assign(positions_wp)

    def _update_rest_curvature(self):
        """Update rest Darboux vector from GUI sliders."""
        self.sim.set_rest_curvature(self.rest_bend_x, self.rest_bend_y, self.rest_twist)

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

        # Move first particle with numpad keys
        move_delta = self.move_speed * self.frame_dt
        rotate_delta = self.rotate_speed * self.frame_dt

        # Numpad 4/6: Move X
        if self.viewer.is_key_down(key.NUM_4):
            self._move_first_particle(-move_delta, 0.0, 0.0)
        if self.viewer.is_key_down(key.NUM_6):
            self._move_first_particle(move_delta, 0.0, 0.0)

        # Numpad 2/8: Move Y
        if self.viewer.is_key_down(key.NUM_2):
            self._move_first_particle(0.0, -move_delta, 0.0)
        if self.viewer.is_key_down(key.NUM_8):
            self._move_first_particle(0.0, move_delta, 0.0)

        # Numpad 9/3: Move Z (up/down)
        if self.viewer.is_key_down(key.NUM_9):
            self._move_first_particle(0.0, 0.0, move_delta)
        if self.viewer.is_key_down(key.NUM_3):
            self._move_first_particle(0.0, 0.0, -move_delta)

        # Numpad 7/1: Rotate around Z axis
        if self.viewer.is_key_down(key.NUM_7):
            self._rotate_first_particle_z(rotate_delta)
        if self.viewer.is_key_down(key.NUM_1):
            self._rotate_first_particle_z(-rotate_delta)

    def _move_first_particle(self, dx: float, dy: float, dz: float):
        """Move the first (fixed) particle by the given delta."""
        self.state.positions[0, 0] += dx
        self.state.positions[0, 1] += dy
        self.state.positions[0, 2] += dz
        self.state.predicted_positions[0, 0] += dx
        self.state.predicted_positions[0, 1] += dy
        self.state.predicted_positions[0, 2] += dz

    def _rotate_first_particle_z(self, angle: float):
        """Rotate the first particle's orientation around its local Z axis (rod tangent)."""
        # Current quaternion (x, y, z, w)
        q = self.state.orientations[0].copy()

        # Rotation quaternion for local Z axis: (0, 0, sin(a/2), cos(a/2))
        half_angle = angle * 0.5
        rz = np.array([0.0, 0.0, np.sin(half_angle), np.cos(half_angle)], dtype=np.float32)

        # Quaternion multiplication: q * rz (post-multiply to rotate in local frame)
        # q1 * q2 = (w1*x2 + x1*w2 + y1*z2 - z1*y2,
        #            w1*y2 - x1*z2 + y1*w2 + z1*x2,
        #            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        #            w1*w2 - x1*x2 - y1*y2 - z1*z2)
        new_q = np.array([
            q[3] * rz[0] + q[0] * rz[3] + q[1] * rz[2] - q[2] * rz[1],
            q[3] * rz[1] - q[0] * rz[2] + q[1] * rz[3] + q[2] * rz[0],
            q[3] * rz[2] + q[0] * rz[1] - q[1] * rz[0] + q[2] * rz[3],
            q[3] * rz[3] - q[0] * rz[0] - q[1] * rz[1] - q[2] * rz[2],
        ], dtype=np.float32)

        # Normalize
        new_q /= np.linalg.norm(new_q)

        # Update orientations
        self.state.orientations[0] = new_q
        self.state.predicted_orientations[0] = new_q
        self.state.prev_orientations[0] = new_q

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
        self.rest_bend_x = 0.0
        self.rest_bend_y = 0.0
        self.rest_twist = 0.0
        self._update_rest_curvature()
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
        colors = wp.array([[0.8, 0.4, 0.1]] * (self.n_particles - 1), dtype=wp.vec3, device=self.model.device)
        self.viewer.log_lines("/rod", starts, ends, colors)

        # Draw particles as spheres
        if hasattr(self.viewer, "log_spheres"):
            positions_wp = wp.array(positions_3d, dtype=wp.vec3, device=self.model.device)
            radii = wp.array([self.particle_radius] * self.n_particles, dtype=float, device=self.model.device)
            # Color: fixed particle (red) vs dynamic (orange)
            particle_colors = []
            for i in range(self.n_particles):
                if self.state.inv_masses[i] == 0:
                    particle_colors.append([0.8, 0.2, 0.2])  # Red for fixed
                else:
                    particle_colors.append([1.0, 0.6, 0.2])  # Orange for dynamic
            colors_wp = wp.array(particle_colors, dtype=wp.vec3, device=self.model.device)
            self.viewer.log_spheres("/particles", positions_wp, radii, colors_wp)

        self.viewer.end_frame()

    def gui(self, ui):
        ui.text("Direct Solver - Stiff Rods")
        ui.text(f"Particles: {self.n_particles}")
        ui.text(f"Time: {self.sim_time:.2f}s")
        ui.separator()

        _, self.substeps = ui.slider_int("Substeps", self.substeps, 1, 16)

        ui.separator()
        ui.text("Material Properties:")
        _, self.sim.young_modulus_mult = ui.slider_float(
            "Young's Mod", self.sim.young_modulus_mult, 0.0, 1.0e6
        )
        _, self.sim.torsion_modulus_mult = ui.slider_float(
            "Torsion Mod", self.sim.torsion_modulus_mult, 0.0, 1.0e6
        )

        ui.separator()
        ui.text("Bend Stiffness:")
        bend_k1 = self.sim.bend_stiffness[0, 0]
        bend_k2 = self.sim.bend_stiffness[0, 1]
        bend_kt = self.sim.bend_stiffness[0, 2]
        changed_k1, bend_k1 = ui.slider_float("Bend K1", bend_k1, 0.0, 2.0)
        changed_k2, bend_k2 = ui.slider_float("Bend K2", bend_k2, 0.0, 2.0)
        changed_kt, bend_kt = ui.slider_float("Twist K", bend_kt, 0.0, 2.0)
        if changed_k1 or changed_k2 or changed_kt:
            self.sim.set_bend_stiffness(bend_k1, bend_k2, bend_kt)

        ui.separator()
        ui.text("Rest Shape (intrinsic curvature):")
        changed_bend_x, self.rest_bend_x = ui.slider_float("Rest Bend X", self.rest_bend_x, -1.0, 1.0)
        changed_bend_y, self.rest_bend_y = ui.slider_float("Rest Bend Y", self.rest_bend_y, -1.0, 1.0)
        changed_twist, self.rest_twist = ui.slider_float("Rest Twist", self.rest_twist, -1.0, 1.0)

        if changed_bend_x or changed_bend_y or changed_twist:
            self._update_rest_curvature()

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
        ui.text("  Numpad 4/6: Move X")
        ui.text("  Numpad 2/8: Move Y")
        ui.text("  Numpad 3/9: Move Z")
        ui.text("  Numpad 1/7: Rotate Z")

        ui.separator()
        base_pos = self.state.positions[0, :3]
        ui.text(f"Base: ({base_pos[0]:.3f}, {base_pos[1]:.3f}, {base_pos[2]:.3f})")
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
        assert tip_z < 0.95, f"Tip should drop below 0.95, got {tip_z}"

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
