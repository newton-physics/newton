# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Direct Position-Based Solver for Stiff Rods - C/C++ vs NumPy comparison.

This example shows two rods side by side:
- Orange rod (Y=0): C/C++ DLL reference implementation
- Cyan rod (Y=1): NumPy implementation (being ported)

Both rods respond to the same UI sliders for comparison.

Command: uv run python -m newton.examples.cosserat_dll.example_dll_direct_cosserat_rod
"""

import numpy as np
import warp as wp

import newton
import newton.examples

from .rod_state import create_straight_rod
from .simulation_direct import DirectCosseratRodSimulation
from .simulation_direct_numpy import DirectCosseratRodSimulationNumPy


class Example:
    """Demo comparing C/C++ and NumPy direct rod solvers side by side."""

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

        # Y-offset for the two rods
        self.y_offset_cpp = 0.0
        self.y_offset_numpy = 1.0

        # =========================================================================
        # Rod 1: C/C++ DLL Reference (orange, at Y=0)
        # =========================================================================
        self.state_cpp = create_straight_rod(
            n_particles=self.n_particles,
            start_pos=(0.0, self.y_offset_cpp, 1.0),
            direction=(1.0, 0.0, 0.0),
            segment_length=self.segment_length,
            fix_first=True,
        )
        self.sim_cpp = DirectCosseratRodSimulation(self.state_cpp, dll_path)

        # =========================================================================
        # Rod 2: NumPy Implementation (cyan, at Y=1)
        # =========================================================================
        self.state_np = create_straight_rod(
            n_particles=self.n_particles,
            start_pos=(0.0, self.y_offset_numpy, 1.0),
            direction=(1.0, 0.0, 0.0),
            segment_length=self.segment_length,
            fix_first=True,
        )
        self.sim_np = DirectCosseratRodSimulationNumPy(self.state_np, dll_path)

        # Enable NumPy implementations by default
        self.sim_np.use_numpy_predict_positions = True
        self.sim_np.use_numpy_predict_rotations = True
        self.sim_np.use_numpy_project_direct = True  # Non-banded solver

        # =========================================================================
        # Shared parameters (controlled by UI)
        # =========================================================================
        self.young_modulus_scale = 1.0  # x1e6
        self.torsion_modulus_scale = 1.0  # x1e6
        self.bend_stiffness = 0.5
        self.twist_stiffness = 0.5
        self.rest_bend_x = 0.0
        self.rest_bend_y = 0.0
        self.rest_twist = 0.0
        self.gravity_scale = 1.0
        self.gravity_enabled = True

        # Apply initial settings to both simulations
        self._sync_sim_parameters()

        # Visualization options
        self.show_segments = True
        self.show_directors = False
        self.director_scale = 0.05

        # Build Newton model for visualization (particles from both rods)
        builder = newton.ModelBuilder()
        builder.add_ground_plane()

        # Add particles for C++ rod
        for i in range(self.n_particles):
            mass = 0.0 if i == 0 else 1.0
            pos = tuple(self.state_cpp.positions[i, :3])
            builder.add_particle(pos=pos, vel=(0.0, 0.0, 0.0), mass=mass, radius=self.particle_radius)

        # Add particles for NumPy rod
        for i in range(self.n_particles):
            mass = 0.0 if i == 0 else 1.0
            pos = tuple(self.state_np.positions[i, :3])
            builder.add_particle(pos=pos, vel=(0.0, 0.0, 0.0), mass=mass, radius=self.particle_radius)

        self.model = builder.finalize()
        self.newton_state = self.model.state()

        self._sync_state()

        self.viewer.set_model(self.model)

        if hasattr(self.viewer, "show_particles"):
            self.viewer.show_particles = True

        # Store initial state for reset
        self._initial_positions_cpp = self.state_cpp.positions.copy()
        self._initial_orientations_cpp = self.state_cpp.orientations.copy()
        self._initial_positions_np = self.state_np.positions.copy()
        self._initial_orientations_np = self.state_np.orientations.copy()

        # Keyboard state
        self._g_key_was_down = False
        self._r_key_was_down = False
        self._b_key_was_down = False

        # Movement/rotation parameters
        self.move_speed = 1.0
        self.rotate_speed = 1.0
        self.root_rotation = 0.0
        self._root_base_orientation_cpp = self.state_cpp.orientations[0].copy()
        self._root_base_orientation_np = self.state_np.orientations[0].copy()

    def _sync_sim_parameters(self):
        """Sync shared parameters to both simulations."""
        # Material moduli
        young_mod = self.young_modulus_scale * 1.0e6
        torsion_mod = self.torsion_modulus_scale * 1.0e6

        self.sim_cpp.young_modulus_mult = young_mod
        self.sim_cpp.torsion_modulus_mult = torsion_mod
        self.sim_np.young_modulus_mult = young_mod
        self.sim_np.torsion_modulus_mult = torsion_mod

        # Bend/twist stiffness
        self.sim_cpp.set_bend_stiffness(self.bend_stiffness, self.bend_stiffness, self.twist_stiffness)
        self.sim_np.set_bend_stiffness(self.bend_stiffness, self.bend_stiffness, self.twist_stiffness)

        # Rest curvature
        self.sim_cpp.set_rest_curvature(self.rest_bend_x, self.rest_bend_y, self.rest_twist)
        self.sim_np.set_rest_curvature(self.rest_bend_x, self.rest_bend_y, self.rest_twist)

        # Gravity
        if self.gravity_enabled:
            g = -9.81 * self.gravity_scale
            self.sim_cpp.set_gravity(0.0, 0.0, g)
            self.sim_np.set_gravity(0.0, 0.0, g)
        else:
            self.sim_cpp.set_gravity(0.0, 0.0, 0.0)
            self.sim_np.set_gravity(0.0, 0.0, 0.0)

    def _sync_state(self):
        """Sync both rod states to Newton state for visualization."""
        # Combine positions from both rods
        positions_cpp = self.state_cpp.get_positions_3d().astype(np.float32)
        positions_np = self.state_np.get_positions_3d().astype(np.float32)
        all_positions = np.vstack([positions_cpp, positions_np])

        positions_wp = wp.array(all_positions, dtype=wp.vec3, device=self.model.device)
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
            self._sync_sim_parameters()
            print(f"Gravity: {'ON' if self.gravity_enabled else 'OFF'}")
        self._g_key_was_down = g_down

        # Reset with R key
        r_down = self.viewer.is_key_down(key.R)
        if r_down and not self._r_key_was_down:
            self._reset()
            print("Reset simulation")
        self._r_key_was_down = r_down

        # Toggle banded/non-banded with B key
        b_down = self.viewer.is_key_down(key.B)
        if b_down and not self._b_key_was_down:
            self._toggle_banded_mode()
        self._b_key_was_down = b_down

        # Move first particle with numpad keys (affects both rods)
        move_delta = self.move_speed * self.frame_dt
        rotate_delta = self.rotate_speed * self.frame_dt

        if self.viewer.is_key_down(key.NUM_4):
            self._move_first_particle(-move_delta, 0.0, 0.0)
        if self.viewer.is_key_down(key.NUM_6):
            self._move_first_particle(move_delta, 0.0, 0.0)
        if self.viewer.is_key_down(key.NUM_2):
            self._move_first_particle(0.0, -move_delta, 0.0)
        if self.viewer.is_key_down(key.NUM_8):
            self._move_first_particle(0.0, move_delta, 0.0)
        if self.viewer.is_key_down(key.NUM_9):
            self._move_first_particle(0.0, 0.0, move_delta)
        if self.viewer.is_key_down(key.NUM_3):
            self._move_first_particle(0.0, 0.0, -move_delta)

        # Rotation
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
        """Move the first (fixed) particle of both rods."""
        for state in [self.state_cpp, self.state_np]:
            state.positions[0, 0] += dx
            state.positions[0, 1] += dy
            state.positions[0, 2] += dz
            state.predicted_positions[0, 0] += dx
            state.predicted_positions[0, 1] += dy
            state.predicted_positions[0, 2] += dz

    def _apply_root_rotation(self):
        """Apply accumulated rotation around the local Z axis to both rods."""
        half_angle = self.root_rotation * 0.5
        rz = np.array([0.0, 0.0, np.sin(half_angle), np.cos(half_angle)], dtype=np.float32)

        for state, base_orient in [(self.state_cpp, self._root_base_orientation_cpp),
                                    (self.state_np, self._root_base_orientation_np)]:
            q = base_orient
            new_q = np.array([
                q[3] * rz[0] + q[0] * rz[3] + q[1] * rz[2] - q[2] * rz[1],
                q[3] * rz[1] - q[0] * rz[2] + q[1] * rz[3] + q[2] * rz[0],
                q[3] * rz[2] + q[0] * rz[1] - q[1] * rz[0] + q[2] * rz[3],
                q[3] * rz[3] - q[0] * rz[0] - q[1] * rz[1] - q[2] * rz[2],
            ], dtype=np.float32)
            new_q /= np.linalg.norm(new_q)

            state.orientations[0] = new_q
            state.predicted_orientations[0] = new_q
            state.prev_orientations[0] = new_q

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

    def _build_director_lines(self, state):
        """Build line segments for visualizing material frame directors."""
        n_edges = self.n_particles - 1
        positions = state.get_positions_3d()
        orientations = state.orientations

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
            colors[base] = [1.0, 0.0, 0.0]

            starts[base + 1] = midpoint
            ends[base + 1] = midpoint + d2 * self.director_scale
            colors[base + 1] = [0.0, 1.0, 0.0]

            starts[base + 2] = midpoint
            ends[base + 2] = midpoint + d3 * self.director_scale
            colors[base + 2] = [0.0, 0.0, 1.0]

        return starts, ends, colors

    def _toggle_banded_mode(self):
        """Toggle between banded and non-banded constraint solving."""
        if self.sim_np.use_numpy_project_direct:
            # Switch to banded mode
            self.sim_np.use_numpy_project_direct = False
            self.sim_np.use_numpy_prepare = True
            self.sim_np.use_numpy_update = True
            self.sim_np.use_numpy_jacobians = True
            self.sim_np.use_numpy_assemble = True
            self.sim_np.use_numpy_solve = True
            print("Switched to BANDED solver (NumPy)")
        else:
            # Switch to non-banded mode
            self.sim_np.use_numpy_project_direct = True
            self.sim_np.use_numpy_prepare = False
            self.sim_np.use_numpy_update = False
            self.sim_np.use_numpy_jacobians = False
            self.sim_np.use_numpy_assemble = False
            self.sim_np.use_numpy_solve = False
            print("Switched to NON-BANDED solver (NumPy)")

    def _reset(self):
        """Reset both simulations to initial state."""
        # Reset C++ rod
        np.copyto(self.state_cpp.positions, self._initial_positions_cpp)
        np.copyto(self.state_cpp.predicted_positions, self._initial_positions_cpp)
        np.copyto(self.state_cpp.orientations, self._initial_orientations_cpp)
        np.copyto(self.state_cpp.predicted_orientations, self._initial_orientations_cpp)
        np.copyto(self.state_cpp.prev_orientations, self._initial_orientations_cpp)
        self.state_cpp.velocities.fill(0)
        self.state_cpp.angular_velocities.fill(0)
        self.state_cpp.clear_forces()

        # Reset NumPy rod
        np.copyto(self.state_np.positions, self._initial_positions_np)
        np.copyto(self.state_np.predicted_positions, self._initial_positions_np)
        np.copyto(self.state_np.orientations, self._initial_orientations_np)
        np.copyto(self.state_np.predicted_orientations, self._initial_orientations_np)
        np.copyto(self.state_np.prev_orientations, self._initial_orientations_np)
        self.state_np.velocities.fill(0)
        self.state_np.angular_velocities.fill(0)
        self.state_np.clear_forces()

        # Reset shared state
        self.rest_bend_x = 0.0
        self.rest_bend_y = 0.0
        self.rest_twist = 0.0
        self.root_rotation = 0.0
        self._sync_sim_parameters()
        self.sim_time = 0.0
        self._sync_state()

    def step(self):
        self._handle_keyboard()

        sub_dt = self.frame_dt / self.substeps

        for _ in range(self.substeps):
            self.sim_cpp.step(sub_dt)
            self.sim_np.step(sub_dt)

        self._sync_state()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.newton_state)

        # =========================================================================
        # C++ Rod (orange)
        # =========================================================================
        positions_cpp = self.state_cpp.get_positions_3d().astype(np.float32)

        if self.show_segments:
            starts = wp.array(positions_cpp[:-1], dtype=wp.vec3, device=self.model.device)
            ends = wp.array(positions_cpp[1:], dtype=wp.vec3, device=self.model.device)
            colors = wp.array([[0.8, 0.4, 0.1]] * (self.n_particles - 1), dtype=wp.vec3, device=self.model.device)
            self.viewer.log_lines("/rod_cpp", starts, ends, colors)

        if self.show_directors:
            dir_starts, dir_ends, dir_colors = self._build_director_lines(self.state_cpp)
            self.viewer.log_lines(
                "/directors_cpp",
                wp.array(dir_starts, dtype=wp.vec3, device=self.model.device),
                wp.array(dir_ends, dtype=wp.vec3, device=self.model.device),
                wp.array(dir_colors, dtype=wp.vec3, device=self.model.device),
            )

        if hasattr(self.viewer, "log_spheres"):
            positions_wp = wp.array(positions_cpp, dtype=wp.vec3, device=self.model.device)
            radii = wp.array([self.particle_radius] * self.n_particles, dtype=float, device=self.model.device)
            particle_colors = []
            for i in range(self.n_particles):
                if self.state_cpp.inv_masses[i] == 0:
                    particle_colors.append([0.8, 0.2, 0.2])  # Red for fixed
                else:
                    particle_colors.append([1.0, 0.6, 0.2])  # Orange for dynamic
            colors_wp = wp.array(particle_colors, dtype=wp.vec3, device=self.model.device)
            self.viewer.log_spheres("/particles_cpp", positions_wp, radii, colors_wp)

        # =========================================================================
        # NumPy Rod (cyan)
        # =========================================================================
        positions_np = self.state_np.get_positions_3d().astype(np.float32)

        if self.show_segments:
            starts = wp.array(positions_np[:-1], dtype=wp.vec3, device=self.model.device)
            ends = wp.array(positions_np[1:], dtype=wp.vec3, device=self.model.device)
            colors = wp.array([[0.1, 0.6, 0.8]] * (self.n_particles - 1), dtype=wp.vec3, device=self.model.device)
            self.viewer.log_lines("/rod_numpy", starts, ends, colors)

        if self.show_directors:
            dir_starts, dir_ends, dir_colors = self._build_director_lines(self.state_np)
            self.viewer.log_lines(
                "/directors_numpy",
                wp.array(dir_starts, dtype=wp.vec3, device=self.model.device),
                wp.array(dir_ends, dtype=wp.vec3, device=self.model.device),
                wp.array(dir_colors, dtype=wp.vec3, device=self.model.device),
            )

        if hasattr(self.viewer, "log_spheres"):
            positions_wp = wp.array(positions_np, dtype=wp.vec3, device=self.model.device)
            radii = wp.array([self.particle_radius] * self.n_particles, dtype=float, device=self.model.device)
            particle_colors = []
            for i in range(self.n_particles):
                if self.state_np.inv_masses[i] == 0:
                    particle_colors.append([0.2, 0.2, 0.8])  # Blue for fixed
                else:
                    particle_colors.append([0.2, 0.8, 0.8])  # Cyan for dynamic
            colors_wp = wp.array(particle_colors, dtype=wp.vec3, device=self.model.device)
            self.viewer.log_spheres("/particles_numpy", positions_wp, radii, colors_wp)

        self.viewer.end_frame()

    def gui(self, ui):
        ui.text("Direct Solver Comparison")
        ui.text("  Orange (Y=0): C/C++ DLL")
        ui.text("  Cyan (Y=1): NumPy")
        ui.text(f"Particles: {self.n_particles} x 2")
        ui.text(f"Time: {self.sim_time:.2f}s")
        ui.separator()

        # NumPy implementation flags
        ui.text("NumPy Implementations:")
        changed_pp, self.sim_np.use_numpy_predict_positions = ui.checkbox(
            "predict_positions", self.sim_np.use_numpy_predict_positions
        )
        changed_pr, self.sim_np.use_numpy_predict_rotations = ui.checkbox(
            "predict_rotations", self.sim_np.use_numpy_predict_rotations
        )

        ui.text("Constraint Solving (B to toggle):")
        use_non_banded = self.sim_np.use_numpy_project_direct
        changed_mode, new_use_non_banded = ui.checkbox("Non-banded solver", use_non_banded)
        if changed_mode and new_use_non_banded != use_non_banded:
            self._toggle_banded_mode()

        # Show banded options only when non-banded is disabled
        if not self.sim_np.use_numpy_project_direct:
            ui.text("  Banded solver steps:")
            changed_prep, self.sim_np.use_numpy_prepare = ui.checkbox(
                "    prepare_constraints", self.sim_np.use_numpy_prepare
            )
            changed_upd, self.sim_np.use_numpy_update = ui.checkbox(
                "    update_constraints", self.sim_np.use_numpy_update
            )
            changed_jac, self.sim_np.use_numpy_jacobians = ui.checkbox(
                "    compute_jacobians", self.sim_np.use_numpy_jacobians
            )
            changed_asm, self.sim_np.use_numpy_assemble = ui.checkbox(
                "    assemble_jmjt", self.sim_np.use_numpy_assemble
            )
            changed_slv, self.sim_np.use_numpy_solve = ui.checkbox(
                "    solve_constraints", self.sim_np.use_numpy_solve
            )
            if self.sim_np.use_numpy_solve:
                _, self.sim_np.use_numpy_solve_spbsv = ui.checkbox(
                    "      use_spbsv_u11", self.sim_np.use_numpy_solve_spbsv
                )
        changed_ip, self.sim_np.use_numpy_integrate_positions = ui.checkbox(
            "integrate_positions", self.sim_np.use_numpy_integrate_positions
        )
        changed_ir, self.sim_np.use_numpy_integrate_rotations = ui.checkbox(
            "integrate_rotations", self.sim_np.use_numpy_integrate_rotations
        )

        ui.separator()
        _, self.substeps = ui.slider_int("Substeps", self.substeps, 1, 16)

        # Sync damping to both
        changed_pd, self.sim_cpp.position_damping = ui.slider_float(
            "Linear Damping", self.sim_cpp.position_damping, 0.0, 0.05
        )
        changed_rd, self.sim_cpp.rotation_damping = ui.slider_float(
            "Angular Damping", self.sim_cpp.rotation_damping, 0.0, 0.05
        )
        if changed_pd:
            self.sim_np.position_damping = self.sim_cpp.position_damping
        if changed_rd:
            self.sim_np.rotation_damping = self.sim_cpp.rotation_damping

        ui.separator()
        ui.text("Stiffness Multipliers (NumPy):")
        changed_stretch, self.sim_np.stretch_stiffness_mult = ui.slider_float(
            "Stretch Mult", self.sim_np.stretch_stiffness_mult, 0.1, 100.0
        )
        changed_shear, self.sim_np.shear_stiffness_mult = ui.slider_float(
            "Shear Mult", self.sim_np.shear_stiffness_mult, 0.1, 100.0
        )
        # Use log scale for bend multiplier (range 1 to 1e9)
        import math
        log_bend_mult = math.log10(max(1.0, self.sim_np.bend_stiffness_mult))
        changed_bend_mult, log_bend_mult = ui.slider_float(
            "Bend Mult (log10)", log_bend_mult, 0.0, 9.0
        )
        if changed_bend_mult:
            self.sim_np.bend_stiffness_mult = 10.0 ** log_bend_mult
        ui.text("Bend/Twist Coefficients:")
        changed_bend, self.bend_stiffness = ui.slider_float("Bend Coeff", self.bend_stiffness, 0.0, 1.0)
        changed_twist_k, self.twist_stiffness = ui.slider_float("Twist Coeff", self.twist_stiffness, 0.0, 1.0)
        if changed_bend or changed_twist_k:
            self._sync_sim_parameters()

        ui.separator()
        ui.text("Material Moduli:")
        changed_young, self.young_modulus_scale = ui.slider_float(
            "Young Mod (x1e6)", self.young_modulus_scale, 0.01, 100.0
        )
        changed_torsion, self.torsion_modulus_scale = ui.slider_float(
            "Torsion Mod (x1e6)", self.torsion_modulus_scale, 0.01, 100.0
        )
        if changed_young or changed_torsion:
            self._sync_sim_parameters()

        ui.separator()
        ui.text("Rest Shape (Darboux Vector):")
        changed_bx, self.rest_bend_x = ui.slider_float("Rest Bend d1", self.rest_bend_x, -0.5, 0.5)
        changed_by, self.rest_bend_y = ui.slider_float("Rest Bend d2", self.rest_bend_y, -0.5, 0.5)
        changed_tw, self.rest_twist = ui.slider_float("Rest Twist", self.rest_twist, -0.5, 0.5)
        if changed_bx or changed_by or changed_tw:
            self._sync_sim_parameters()

        ui.separator()
        gravity_changed, self.gravity_enabled = ui.checkbox("Gravity (G)", self.gravity_enabled)
        scale_changed, self.gravity_scale = ui.slider_float("Gravity Scale", self.gravity_scale, 0.0, 2.0)
        if gravity_changed or scale_changed:
            self._sync_sim_parameters()

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
        ui.text("Keyboard: G=Gravity, R=Reset, B=Banded/Non-banded")

        ui.separator()
        # Show tip positions for both rods
        tip_cpp = self.state_cpp.positions[-1, :3]
        tip_np = self.state_np.positions[-1, :3]
        ui.text(f"C++ Tip: ({tip_cpp[0]:.3f}, {tip_cpp[1]:.3f}, {tip_cpp[2]:.3f})")
        ui.text(f"NP  Tip: ({tip_np[0]:.3f}, {tip_np[1]:.3f}, {tip_np[2]:.3f})")

        # Show difference
        diff = np.linalg.norm(tip_cpp - tip_np)
        ui.text(f"Tip Difference: {diff:.6f}")

    def test_final(self):
        """Validation after simulation."""
        # Check C++ tip has dropped under gravity
        tip_z_cpp = self.state_cpp.positions[-1, 2]
        assert tip_z_cpp < 0.95, f"C++ tip should drop below 0.95, got {tip_z_cpp}"

        # Check NumPy tip has dropped under gravity
        tip_z_np = self.state_np.positions[-1, 2]
        assert tip_z_np < 0.95, f"NumPy tip should drop below 0.95, got {tip_z_np}"

        # Check segment lengths for C++ rod
        for i in range(self.n_particles - 1):
            actual = np.linalg.norm(self.state_cpp.positions[i + 1, :3] - self.state_cpp.positions[i, :3])
            error = abs(actual - self.segment_length) / self.segment_length
            assert error < 0.2, f"C++ Segment {i} length error {error * 100:.1f}% exceeds 20%"


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
