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

        # Material parameters (scaled for GUI - actual value = scale * 1e6)
        self.sim.radius = 0.01
        self.young_modulus_scale = 1.0  # x1e6
        self.torsion_modulus_scale = 1.0  # x1e6

        # Bend/twist stiffness (0-1 range)
        self.bend_stiffness = 0.5
        self.twist_stiffness = 0.5
        self.sim.set_bend_stiffness(self.bend_stiffness, self.bend_stiffness, self.twist_stiffness)

        # Rest curvature parameters (for GUI)
        self.rest_bend_x = 0.0
        self.rest_bend_y = 0.0
        self.rest_twist = 0.0

        # Gravity scale
        self.gravity_scale = 1.0

        # Visualization options
        self.show_segments = True
        self.show_directors = False
        self.director_scale = 0.05

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
        self.move_speed = 1.0  # units per second
        self.rotate_speed = 1.0  # radians per second
        self.root_rotation = 0.0  # accumulated rotation angle
        self._root_base_orientation = self.state.orientations[0].copy()

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

        # Numpad 7/1: Rotate around local Z axis (rod tangent)
        rotation_changed = False
        if self.viewer.is_key_down(key.NUM_7):
            self.root_rotation += rotate_delta
            rotation_changed = True
        if self.viewer.is_key_down(key.NUM_1):
            self.root_rotation -= rotate_delta
            rotation_changed = True
        if rotation_changed:
            self._apply_root_rotation()

    def _move_first_particle(self, dx: float, dy: float, dz: float):
        """Move the first (fixed) particle by the given delta."""
        self.state.positions[0, 0] += dx
        self.state.positions[0, 1] += dy
        self.state.positions[0, 2] += dz
        self.state.predicted_positions[0, 0] += dx
        self.state.predicted_positions[0, 1] += dy
        self.state.predicted_positions[0, 2] += dz

    def _apply_root_rotation(self):
        """Apply accumulated rotation around the local Z axis (rod tangent)."""
        # Rotation quaternion for local Z axis: (0, 0, sin(a/2), cos(a/2))
        half_angle = self.root_rotation * 0.5
        rz = np.array([0.0, 0.0, np.sin(half_angle), np.cos(half_angle)], dtype=np.float32)

        # Base orientation
        q = self._root_base_orientation

        # Quaternion multiplication: q * rz (post-multiply to rotate in local frame)
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

    def _rotate_vector_by_quat(self, v: np.ndarray, q: np.ndarray) -> np.ndarray:
        """Rotate a vector by a quaternion."""
        x, y, z, w = q
        vx, vy, vz = v

        tx = 2.0 * (y * vz - z * vy)
        ty = 2.0 * (z * vx - x * vz)
        tz = 2.0 * (x * vy - y * vx)

        return np.array([
            vx + w * tx + y * tz - z * ty,
            vy + w * ty + z * tx - x * tz,
            vz + w * tz + x * ty - y * tx,
        ], dtype=np.float32)

    def _build_director_lines(self):
        """Build line segments for visualizing material frame directors."""
        n_edges = self.n_particles - 1
        positions = self.state.get_positions_3d()
        orientations = self.state.orientations

        starts = np.zeros((n_edges * 3, 3), dtype=np.float32)
        ends = np.zeros((n_edges * 3, 3), dtype=np.float32)
        colors = np.zeros((n_edges * 3, 3), dtype=np.float32)

        for i in range(n_edges):
            midpoint = 0.5 * (positions[i] + positions[i + 1])
            q = orientations[i]

            d1 = self._rotate_vector_by_quat(np.array([1.0, 0.0, 0.0], dtype=np.float32), q)
            d2 = self._rotate_vector_by_quat(np.array([0.0, 1.0, 0.0], dtype=np.float32), q)
            d3 = self._rotate_vector_by_quat(np.array([0.0, 0.0, 1.0], dtype=np.float32), q)

            base = i * 3
            starts[base] = midpoint
            ends[base] = midpoint + d1 * self.director_scale
            colors[base] = [1.0, 0.0, 0.0]  # Red for d1

            starts[base + 1] = midpoint
            ends[base + 1] = midpoint + d2 * self.director_scale
            colors[base + 1] = [0.0, 1.0, 0.0]  # Green for d2

            starts[base + 2] = midpoint
            ends[base + 2] = midpoint + d3 * self.director_scale
            colors[base + 2] = [0.0, 0.0, 1.0]  # Blue for d3 (tangent)

        return starts, ends, colors

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
        self.root_rotation = 0.0
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

        positions_3d = self.state.get_positions_3d().astype(np.float32)

        # Draw rod segments as lines
        if self.show_segments:
            starts = wp.array(positions_3d[:-1], dtype=wp.vec3, device=self.model.device)
            ends = wp.array(positions_3d[1:], dtype=wp.vec3, device=self.model.device)
            colors = wp.array([[0.8, 0.4, 0.1]] * (self.n_particles - 1), dtype=wp.vec3, device=self.model.device)
            self.viewer.log_lines("/rod", starts, ends, colors)

        # Draw material frame directors
        if self.show_directors:
            dir_starts, dir_ends, dir_colors = self._build_director_lines()
            self.viewer.log_lines(
                "/directors",
                wp.array(dir_starts, dtype=wp.vec3, device=self.model.device),
                wp.array(dir_ends, dtype=wp.vec3, device=self.model.device),
                wp.array(dir_colors, dtype=wp.vec3, device=self.model.device),
            )

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
        _, self.sim.position_damping = ui.slider_float("Linear Damping", self.sim.position_damping, 0.0, 0.05)
        _, self.sim.rotation_damping = ui.slider_float("Angular Damping", self.sim.rotation_damping, 0.0, 0.05)

        ui.separator()
        ui.text("Stiffness:")
        changed_bend, self.bend_stiffness = ui.slider_float("Bend Stiffness", self.bend_stiffness, 0.0, 1.0)
        changed_twist_k, self.twist_stiffness = ui.slider_float("Twist Stiffness", self.twist_stiffness, 0.0, 1.0)
        if changed_bend or changed_twist_k:
            self.sim.set_bend_stiffness(self.bend_stiffness, self.bend_stiffness, self.twist_stiffness)

        ui.separator()
        ui.text("Material Moduli:")
        changed_young, self.young_modulus_scale = ui.slider_float(
            "Young Mod (x1e6)", self.young_modulus_scale, 0.01, 100.0
        )
        changed_torsion, self.torsion_modulus_scale = ui.slider_float(
            "Torsion Mod (x1e6)", self.torsion_modulus_scale, 0.01, 100.0
        )
        if changed_young or changed_torsion:
            self.sim.young_modulus_mult = self.young_modulus_scale * 1.0e6
            self.sim.torsion_modulus_mult = self.torsion_modulus_scale * 1.0e6

        ui.separator()
        ui.text("Rest Shape (Darboux Vector):")
        changed_bend_x, self.rest_bend_x = ui.slider_float("Rest Bend d1", self.rest_bend_x, -0.5, 0.5)
        changed_bend_y, self.rest_bend_y = ui.slider_float("Rest Bend d2", self.rest_bend_y, -0.5, 0.5)
        changed_twist, self.rest_twist = ui.slider_float("Rest Twist", self.rest_twist, -0.5, 0.5)
        if changed_bend_x or changed_bend_y or changed_twist:
            self._update_rest_curvature()

        ui.separator()
        gravity_changed, self.gravity_enabled = ui.checkbox("Gravity (G)", self.gravity_enabled)
        scale_changed, self.gravity_scale = ui.slider_float("Gravity Scale", self.gravity_scale, 0.0, 2.0)
        if gravity_changed or scale_changed:
            if self.gravity_enabled:
                self.sim.set_gravity(0.0, 0.0, -9.81 * self.gravity_scale)
            else:
                self.sim.set_gravity(0.0, 0.0, 0.0)

        ui.separator()
        ui.text("Visualization:")
        _, self.show_segments = ui.checkbox("Show Rod Segments", self.show_segments)
        _, self.show_directors = ui.checkbox("Show Directors", self.show_directors)
        _, self.director_scale = ui.slider_float("Director Scale", self.director_scale, 0.01, 0.2)

        ui.separator()
        ui.text("Root Control (Numpad):")
        _, self.move_speed = ui.slider_float("Move Speed", self.move_speed, 0.1, 5.0)
        _, self.rotate_speed = ui.slider_float("Rotate Speed", self.rotate_speed, 0.1, 3.0)
        ui.text(f"  Rotation: {self.root_rotation:.2f} rad")
        ui.text("  4/6: X  2/8: Y  3/9: Z  1/7: Twist")

        ui.separator()
        ui.text("Keyboard: G=Gravity, R=Reset")

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
