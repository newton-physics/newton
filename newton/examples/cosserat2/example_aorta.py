# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Cosserat rod simulation with pluggable XPBD solver.

This example demonstrates a Cosserat rod (catheter) navigating through
an aorta mesh, with runtime-switchable constraint solving methods:
- Jacobi: Iterative parallel solver (default)
- Thomas: O(n) tridiagonal solver for stretch
- Cholesky: Direct tile-based solver
- Local: Local iterative solver with velocity update (from 02_local_cosserat_rod)

Command: uv run -m newton.examples cosserat2_aorta
"""

import math
import os

import numpy as np
import warp as wp
from pxr import Usd

import newton
import newton.examples
import newton.usd
from newton.examples.cosserat2.cosserat_rod import CosseratRod
from newton.examples.cosserat2.kernels import (
    compute_director_lines_kernel,
    compute_static_tri_aabbs_kernel,
    update_rest_darboux_kernel,
    update_tip_rest_darboux_kernel,
)
from newton.examples.cosserat2.solver_cosserat_xpbd import SolverConfig, SolverCosseratXPBD
from newton.examples.cosserat2.solvers import ConstraintSolverType, FrictionMethod

# Default rod configuration
DEFAULT_NUM_PARTICLES = 32
DEFAULT_PARTICLE_SPACING = 0.025


class Example:
    """Cosserat rod example with pluggable constraint solvers.

    Features:
    - Runtime-switchable solver methods (Jacobi, Thomas, Cholesky, Local)
    - Three internal friction models (not supported by Local solver)
    - Aorta mesh collision
    - Director frame visualization
    - Keyboard control for anchor movement
    """

    def __init__(self, viewer, args=None):
        # Simulation parameters
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 8
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer
        self.args = args

        # Rod parameters
        self.num_particles = DEFAULT_NUM_PARTICLES
        self.num_stretch = self.num_particles - 1
        self.num_bend = self.num_particles - 2

        particle_spacing = DEFAULT_PARTICLE_SPACING
        particle_mass = 0.1
        particle_radius = 0.02
        edge_mass = 0.01
        start_height = 5.0

        # Solver configuration
        self.solver_config = SolverConfig(
            constraint_iterations=2,
            gravity=wp.vec3(0.0, 0.0, 0.0),
            ground_level=0.0,
            solver_type=ConstraintSolverType.JACOBI,
            friction_method=FrictionMethod.NONE,
            stretch_stiffness=1.0,
            shear_stiffness=1.0,
            bend_stiffness=0.5,
            twist_stiffness=0.5,
            velocity_damping=0.99,
            strain_rate_damping=0.1,
            dahl_eps_max=0.01,
            dahl_tau=0.005,
        )

        # Build Newton model
        builder = newton.ModelBuilder()
        builder.add_ground_plane()

        # Load the aorta vessel mesh
        usd_path = os.path.join(os.path.dirname(__file__), "models", "DynamicAorta.usdc")
        usd_stage = Usd.Stage.Open(usd_path)
        mesh_prim = usd_stage.GetPrimAtPath("/root/A4009/A4007/Xueguan_rudong/Dynamic_vessels/Mesh")

        vessel_mesh = newton.usd.get_mesh(mesh_prim)

        # Store mesh data for collision
        self.vessel_vertices_np = np.array(vessel_mesh.vertices, dtype=np.float32)
        self.vessel_indices_np = np.array(vessel_mesh.indices, dtype=np.int32).reshape(-1, 3)
        self.num_vessel_triangles = self.vessel_indices_np.shape[0]

        # Add vessel mesh as static collision shape
        vessel_cfg = newton.ModelBuilder.ShapeConfig(
            ke=1.0e4,
            kd=1.0e2,
            mu=0.1,
            has_shape_collision=False,
            has_particle_collision=True,
        )
        self.mesh_scale = 0.01

        builder.add_shape_mesh(
            body=-1,
            mesh=vessel_mesh,
            scale=(self.mesh_scale, self.mesh_scale, self.mesh_scale),
            xform=wp.transform(
                wp.vec3(0.0, 0.0, 1.0),
                wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), math.pi / 2.0),
            ),
            cfg=vessel_cfg,
        )

        # Target position for the last particle (inside the aorta)
        target_last_pos = np.array([-3.283308, -0.50000024, 1.6833224])
        last_particle_idx = self.num_particles - 1
        current_last_pos = np.array([last_particle_idx * particle_spacing, 0.0, start_height])
        translation_offset = target_last_pos - current_last_pos

        # Create particles
        for i in range(self.num_particles):
            mass = 0.0 if i == 0 else particle_mass
            pos = np.array([i * particle_spacing, 0.0, start_height]) + translation_offset
            builder.add_particle(
                pos=tuple(pos),
                vel=(0.0, 0.0, 0.0),
                mass=mass,
                radius=particle_radius,
            )

        self.model = builder.finalize()
        self.model.soft_contact_ke = 1.0e3
        self.model.soft_contact_kd = 1.0e1
        self.model.soft_contact_mu = 0.5

        # State buffers
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        # Collision pipeline
        self.collision_pipeline = newton.examples.create_collision_pipeline(self.model, self.args)
        self.contacts = self.model.collide(self.state_0, collision_pipeline=self.collision_pipeline)

        device = self.model.device

        # Prepare vessel mesh for collision
        mesh_xform = wp.transform(
            wp.vec3(0.0, 0.0, 1.0),
            wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), math.pi / 2.0),
        )
        scaled_vertices = self.vessel_vertices_np * self.mesh_scale
        transformed_vertices = np.zeros_like(scaled_vertices)
        for i in range(len(scaled_vertices)):
            v = wp.vec3(scaled_vertices[i, 0], scaled_vertices[i, 1], scaled_vertices[i, 2])
            v_transformed = wp.transform_point(mesh_xform, v)
            transformed_vertices[i] = [v_transformed[0], v_transformed[1], v_transformed[2]]

        self.vessel_vertices = wp.array(transformed_vertices, dtype=wp.vec3f, device=device)
        self.vessel_indices = wp.array(self.vessel_indices_np, dtype=wp.int32, device=device)

        # Build BVH for collision
        self.tri_lower_bounds = wp.zeros(self.num_vessel_triangles, dtype=wp.vec3f, device=device)
        self.tri_upper_bounds = wp.zeros(self.num_vessel_triangles, dtype=wp.vec3f, device=device)
        wp.launch(
            kernel=compute_static_tri_aabbs_kernel,
            dim=self.num_vessel_triangles,
            inputs=[self.vessel_vertices, self.vessel_indices],
            outputs=[self.tri_lower_bounds, self.tri_upper_bounds],
            device=device,
        )
        self.vessel_bvh = wp.Bvh(self.tri_lower_bounds, self.tri_upper_bounds)

        # Create CosseratRod data structure
        inv_mass_np = [0.0] + [1.0 / particle_mass] * (self.num_particles - 1)
        particle_inv_mass = wp.array(inv_mass_np, dtype=float, device=device)

        edge_inv_mass_np = [0.0] + [1.0 / edge_mass] * (self.num_stretch - 1)
        edge_inv_mass = wp.array(edge_inv_mass_np, dtype=float, device=device)

        rest_length_np = [particle_spacing] * self.num_stretch
        rest_length = wp.array(rest_length_np, dtype=float, device=device)

        angle = math.pi / 2.0
        q_init = wp.quat(0.0, math.sin(angle / 2.0), 0.0, math.cos(angle / 2.0))
        edge_q_init = wp.array([q_init] * self.num_stretch, dtype=wp.quat, device=device)

        rest_darboux_init = wp.array([wp.quat(0.0, 0.0, 0.0, 1.0)] * self.num_bend, dtype=wp.quat, device=device)

        self.cosserat_rod = CosseratRod(
            num_particles=self.num_particles,
            particle_inv_mass=particle_inv_mass,
            edge_inv_mass=edge_inv_mass,
            rest_length=rest_length,
            edge_q_init=edge_q_init,
            rest_darboux_init=rest_darboux_init,
            device=device,
        )

        # Create main XPBD solver
        self.solver = SolverCosseratXPBD(
            rod=self.cosserat_rod,
            particle_q=self.state_0.particle_q,
            particle_qd=self.state_0.particle_qd,
            particle_radius=self.model.particle_radius,
            config=self.solver_config,
            device=device,
        )

        # Enable mesh collision
        self.solver.enable_mesh_collision(
            vertices=self.vessel_vertices,
            indices=self.vessel_indices,
            bvh=self.vessel_bvh,
        )

        # Setup viewer
        self.viewer.set_model(self.model)
        self.viewer.show_particles = True

        # Director visualization
        num_director_lines = self.num_stretch * 3
        self.director_line_starts = wp.zeros(num_director_lines, dtype=wp.vec3, device=device)
        self.director_line_ends = wp.zeros(num_director_lines, dtype=wp.vec3, device=device)
        self.director_line_colors = wp.zeros(num_director_lines, dtype=wp.vec3, device=device)
        self.show_directors = True
        self.director_scale = 0.03

        # Rest shape parameters
        self.rest_bend_d1 = 0.0
        self.rest_bend_d2 = 0.0
        self.rest_twist = 0.0

        # Tip control
        self.tip_num_particles = 10
        self.tip_rest_bend_d1 = 0.0
        self.tip_bend_speed = 0.3

        # Keyboard control
        self.particle_move_speed = 1.0
        self.particle_rotation_speed = 3.0
        self.first_particle_rotation = 0.0

        # Solver type for UI
        self.current_solver_idx = 0

        # Gravity toggle
        self.gravity_enabled = False
        self.gravity_value = wp.vec3(0.0, 0.0, -9.81)
        self._gravity_key_was_down = False

        # Reset key tracking
        self._reset_key_was_down = False

        # Store initial state for reset
        self._initial_particle_q = self.state_0.particle_q.numpy().copy()
        self._initial_particle_qd = self.state_0.particle_qd.numpy().copy()
        self._initial_edge_q = self.cosserat_rod.edge_q.numpy().copy()

    def _reset_simulation(self):
        """Reset the simulation to initial state."""
        device = self.model.device

        # Reset particle positions and velocities
        self.state_0.particle_q = wp.array(self._initial_particle_q, dtype=wp.vec3, device=device)
        self.state_0.particle_qd = wp.array(self._initial_particle_qd, dtype=wp.vec3, device=device)
        self.state_1.particle_q = wp.array(self._initial_particle_q, dtype=wp.vec3, device=device)
        self.state_1.particle_qd = wp.array(self._initial_particle_qd, dtype=wp.vec3, device=device)

        # Reset edge quaternions
        self.cosserat_rod.edge_q = wp.array(self._initial_edge_q, dtype=wp.quat, device=device)
        self.cosserat_rod.edge_q_new = wp.array(self._initial_edge_q, dtype=wp.quat, device=device)

        # Reset simulation time
        self.sim_time = 0.0

        # Reset control parameters
        self.first_particle_rotation = 0.0
        self.tip_rest_bend_d1 = 0.0
        self.rest_bend_d1 = 0.0
        self.rest_bend_d2 = 0.0
        self.rest_twist = 0.0

        # Update contacts
        self.contacts = self.model.collide(self.state_0, collision_pipeline=self.collision_pipeline)

        print("Simulation reset")

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.solver.substep(
                self.state_0.particle_q,
                self.state_0.particle_qd,
                self.state_1.particle_q,
                self.state_1.particle_qd,
                self.sim_dt,
            )

            # Swap states
            self.state_0, self.state_1 = self.state_1, self.state_0

            # Update contacts for visualization
            self.contacts = self.model.collide(self.state_0, collision_pipeline=self.collision_pipeline)

    def step(self):
        self._handle_keyboard_input()
        self.simulate()
        self.sim_time += self.frame_dt

    def _handle_keyboard_input(self):
        """Move the first locked particle using numpad keys."""
        if not hasattr(self.viewer, "is_key_down"):
            return

        try:
            import pyglet.window.key as key
        except ImportError:
            return

        dx = dy = dz = 0.0

        # Movement controls
        if self.viewer.is_key_down(key.NUM_6) or self.viewer.is_key_down(key.L):
            dx += self.particle_move_speed * self.frame_dt
        if self.viewer.is_key_down(key.NUM_4) or self.viewer.is_key_down(key.J):
            dx -= self.particle_move_speed * self.frame_dt
        if self.viewer.is_key_down(key.NUM_8) or self.viewer.is_key_down(key.I):
            dy += self.particle_move_speed * self.frame_dt
        if self.viewer.is_key_down(key.NUM_2) or self.viewer.is_key_down(key.K):
            dy -= self.particle_move_speed * self.frame_dt
        if self.viewer.is_key_down(key.NUM_9) or self.viewer.is_key_down(key.U):
            dz += self.particle_move_speed * self.frame_dt
        if self.viewer.is_key_down(key.NUM_3) or self.viewer.is_key_down(key.O):
            dz -= self.particle_move_speed * self.frame_dt

        # Rotation control
        rotation_changed = False
        if self.viewer.is_key_down(key.NUM_7) or self.viewer.is_key_down(key.Y):
            self.first_particle_rotation += self.particle_rotation_speed * self.frame_dt
            rotation_changed = True
        if self.viewer.is_key_down(key.NUM_1) or self.viewer.is_key_down(key.T):
            self.first_particle_rotation -= self.particle_rotation_speed * self.frame_dt
            rotation_changed = True

        if rotation_changed:
            self._update_first_edge_rotation()

        # Tip bend control
        tip_bend_changed = False
        if self.viewer.is_key_down(key.NUM_ADD) or self.viewer.is_key_down(key.EQUAL):
            self.tip_rest_bend_d1 += self.tip_bend_speed * self.frame_dt
            self.tip_rest_bend_d1 = min(self.tip_rest_bend_d1, 0.5)
            tip_bend_changed = True
        if self.viewer.is_key_down(key.NUM_SUBTRACT) or self.viewer.is_key_down(key.MINUS):
            self.tip_rest_bend_d1 -= self.tip_bend_speed * self.frame_dt
            self.tip_rest_bend_d1 = max(self.tip_rest_bend_d1, -0.5)
            tip_bend_changed = True

        if tip_bend_changed:
            self._update_tip_rest_darboux()

        # Solver switching with 1, 2, 3, 4 keys
        method_types = [
            ConstraintSolverType.JACOBI,
            ConstraintSolverType.THOMAS,
            ConstraintSolverType.CHOLESKY_SINGLE,
            ConstraintSolverType.LOCAL,
        ]
        method_names = ["Jacobi", "Thomas", "Cholesky", "Local"]

        for i, (k, solver_type, name) in enumerate(
            zip([key._1, key._2, key._3, key._4], method_types, method_names, strict=True)
        ):
            if self.viewer.is_key_down(k) and self.current_solver_idx != i:
                try:
                    self.solver.set_constraint_solver(solver_type)
                    self.current_solver_idx = i
                    print(f"Switched to {name} solver")
                except ValueError as e:
                    print(f"Cannot switch to {name}: {e}")
                break

        # Gravity toggle with G key (edge-triggered)
        g_key_down = self.viewer.is_key_down(key.G)
        if g_key_down and not self._gravity_key_was_down:
            self.gravity_enabled = not self.gravity_enabled
            if self.gravity_enabled:
                self.solver_config.gravity = self.gravity_value
                print("Gravity enabled")
            else:
                self.solver_config.gravity = wp.vec3(0.0, 0.0, 0.0)
                print("Gravity disabled")
        self._gravity_key_was_down = g_key_down

        # Reset simulation with R key (edge-triggered)
        r_key_down = self.viewer.is_key_down(key.R)
        if r_key_down and not self._reset_key_was_down:
            self._reset_simulation()
        self._reset_key_was_down = r_key_down

        # Apply movement
        if dx != 0.0 or dy != 0.0 or dz != 0.0:
            particle_q_np = self.state_0.particle_q.numpy()
            pos = particle_q_np[0]
            new_pos = [pos[0] + dx, pos[1] + dy, max(pos[2] + dz, 0.1)]
            particle_q_np[0] = new_pos
            self.state_0.particle_q = wp.array(particle_q_np, dtype=wp.vec3, device=self.model.device)

    def _update_rest_darboux(self):
        """Update rest Darboux vectors from current slider values."""
        wp.launch(
            kernel=update_rest_darboux_kernel,
            dim=self.num_bend,
            inputs=[self.rest_bend_d1, self.rest_bend_d2, self.rest_twist, self.num_bend],
            outputs=[self.cosserat_rod.rest_darboux],
            device=self.model.device,
        )

    def _update_tip_rest_darboux(self):
        """Update rest Darboux vectors for the tip of the rod."""
        tip_start_idx = max(0, self.num_bend - self.tip_num_particles + 1)
        num_tip_constraints = self.num_bend - tip_start_idx

        if num_tip_constraints > 0:
            wp.launch(
                kernel=update_tip_rest_darboux_kernel,
                dim=num_tip_constraints,
                inputs=[self.tip_rest_bend_d1, tip_start_idx, self.num_bend],
                outputs=[self.cosserat_rod.rest_darboux],
                device=self.model.device,
            )

    def _update_first_edge_rotation(self):
        """Update the first edge quaternion to apply rotation."""
        base_angle = math.pi / 2.0
        q_base = wp.quat(0.0, math.sin(base_angle / 2.0), 0.0, math.cos(base_angle / 2.0))
        twist_angle = self.first_particle_rotation
        q_twist = wp.quat(0.0, 0.0, math.sin(twist_angle / 2.0), math.cos(twist_angle / 2.0))
        q_combined = wp.mul(q_base, q_twist)

        edge_q_np = self.cosserat_rod.edge_q.numpy()
        edge_q_np[0] = [q_combined[0], q_combined[1], q_combined[2], q_combined[3]]
        self.cosserat_rod.edge_q = wp.array(edge_q_np, dtype=wp.quat, device=self.model.device)

    def test_final(self):
        """Validation method run after simulation completes."""
        # Verify anchor particle is stationary
        newton.examples.test_particle_state(
            self.state_0,
            "anchor particle is stationary",
            lambda q, qd: wp.length(qd) < 1e-6,
            indices=[0],
        )

        # Verify all particles are above ground
        newton.examples.test_particle_state(
            self.state_0,
            "particles are above the ground",
            lambda q, qd: q[2] >= -0.01,
        )

        # Verify particles are within bounds
        p_lower = wp.vec3(-5.0, -5.0, -0.1)
        p_upper = wp.vec3(10.0, 5.0, 7.0)
        newton.examples.test_particle_state(
            self.state_0,
            "particles are within reasonable bounds",
            lambda q, qd: newton.utils.vec_inside_limits(q, p_lower, p_upper),
        )

        # Verify quaternions are normalized
        edge_q_np = self.cosserat_rod.edge_q.numpy()
        for i, q in enumerate(edge_q_np):
            norm = (q[0] ** 2 + q[1] ** 2 + q[2] ** 2 + q[3] ** 2) ** 0.5
            assert abs(norm - 1.0) < 0.1, f"Edge quaternion {i} not normalized: norm={norm}"

    def gui(self, ui):
        ui.text("Cosserat Rod with Pluggable Solver")
        ui.text(f"Particles: {self.num_particles}")

        # Constraint Solver Selection
        ui.separator()
        ui.text("Constraint Solving Method")
        methods = ["Jacobi Iteration", "Thomas Algorithm", "Cholesky (Single Tile)", "Local Iterative"]
        method_types = [
            ConstraintSolverType.JACOBI,
            ConstraintSolverType.THOMAS,
            ConstraintSolverType.CHOLESKY_SINGLE,
            ConstraintSolverType.LOCAL,
        ]
        changed, new_idx = ui.combo("Solver Method", self.current_solver_idx, methods)
        if changed:
            self.current_solver_idx = new_idx
            try:
                self.solver.set_constraint_solver(method_types[new_idx])
            except ValueError as e:
                # Cholesky may fail for large rods
                ui.text(f"  Error: {e}")
                self.current_solver_idx = 0
                self.solver.set_constraint_solver(ConstraintSolverType.JACOBI)

        # Method-specific info
        solver_type = self.solver_config.solver_type
        if solver_type == ConstraintSolverType.JACOBI:
            ui.text("  Parallel Jacobi iteration")
        elif solver_type == ConstraintSolverType.THOMAS:
            ui.text("  O(n) direct solve for stretch")
        elif solver_type == ConstraintSolverType.CHOLESKY_SINGLE:
            ui.text("  Tile size: 32x32")
        elif solver_type == ConstraintSolverType.LOCAL:
            ui.text("  Local iteration with velocity update")
        ui.text("  Keys 1/2/3/4: Switch solver")

        # Simulation Parameters
        ui.separator()
        ui.text("Simulation Parameters")
        _, new_substeps = ui.slider_int("Substeps", self.sim_substeps, 1, 32)
        if new_substeps != self.sim_substeps:
            self.sim_substeps = new_substeps
            self.sim_dt = self.frame_dt / self.sim_substeps
        _, self.solver_config.constraint_iterations = ui.slider_int(
            "Constraint Iterations", self.solver_config.constraint_iterations, 1, 16
        )
        changed, self.gravity_enabled = ui.checkbox("Gravity (G key)", self.gravity_enabled)
        if changed:
            if self.gravity_enabled:
                self.solver_config.gravity = self.gravity_value
            else:
                self.solver_config.gravity = wp.vec3(0.0, 0.0, 0.0)

        # Stiffness Parameters
        ui.separator()
        ui.text("Stiffness Parameters")
        _, self.solver_config.stretch_stiffness = ui.slider_float(
            "Stretch Stiffness", self.solver_config.stretch_stiffness, 0.0, 1.0
        )
        _, self.solver_config.shear_stiffness = ui.slider_float(
            "Shear Stiffness", self.solver_config.shear_stiffness, 0.0, 1.0
        )
        _, self.solver_config.bend_stiffness = ui.slider_float(
            "Bend Stiffness", self.solver_config.bend_stiffness, 0.0, 1.0
        )
        _, self.solver_config.twist_stiffness = ui.slider_float(
            "Twist Stiffness", self.solver_config.twist_stiffness, 0.0, 1.0
        )

        # Rest Shape
        ui.separator()
        ui.text("Rest Shape (Darboux Vector)")
        changed_d1, self.rest_bend_d1 = ui.slider_float("Rest Bend d1", self.rest_bend_d1, -0.5, 0.5)
        changed_d2, self.rest_bend_d2 = ui.slider_float("Rest Bend d2", self.rest_bend_d2, -0.5, 0.5)
        changed_twist, self.rest_twist = ui.slider_float("Rest Twist", self.rest_twist, -0.5, 0.5)
        if changed_d1 or changed_d2 or changed_twist:
            self._update_rest_darboux()

        # Tip Control
        ui.separator()
        ui.text("Tip Rest Shape")
        changed_tip, self.tip_rest_bend_d1 = ui.slider_float("Tip Rest Bend d1", self.tip_rest_bend_d1, -0.5, 0.5)
        if changed_tip:
            self._update_tip_rest_darboux()
        ui.text("  Numpad +/-: Increase/decrease tip bend")

        # Internal Friction
        ui.separator()
        ui.text("Internal Friction")
        friction_methods = ["None", "Velocity Damping", "Strain-Rate Damping", "Dahl Hysteresis"]
        _, friction_idx = ui.combo("Friction Method", int(self.solver_config.friction_method), friction_methods)
        self.solver_config.friction_method = FrictionMethod(friction_idx)

        if self.solver_config.friction_method == FrictionMethod.VELOCITY_DAMPING:
            _, self.solver_config.velocity_damping = ui.slider_float(
                "Velocity Damping", self.solver_config.velocity_damping, 0.9, 1.0
            )
        elif self.solver_config.friction_method == FrictionMethod.STRAIN_RATE_DAMPING:
            _, self.solver_config.strain_rate_damping = ui.slider_float(
                "Strain-Rate Damping", self.solver_config.strain_rate_damping, 0.0, 1.0
            )
        elif self.solver_config.friction_method == FrictionMethod.DAHL_HYSTERESIS:
            _, self.solver_config.dahl_eps_max = ui.slider_float("Eps Max", self.solver_config.dahl_eps_max, 0.0, 0.1)
            _, self.solver_config.dahl_tau = ui.slider_float("Tau", self.solver_config.dahl_tau, 0.001, 0.1)

        # Anchor Control
        ui.separator()
        ui.text("Anchor Control (First Particle)")
        ui.text("  Numpad 4/6: Move left/right (X)")
        ui.text("  Numpad 8/2: Move forward/back (Y)")
        ui.text("  Numpad 9/3: Move up/down (Z)")
        ui.text("  Numpad 7/1: Rotate CCW/CW")
        _, self.particle_move_speed = ui.slider_float("Move Speed", self.particle_move_speed, 0.1, 5.0)
        _, self.particle_rotation_speed = ui.slider_float("Rotation Speed", self.particle_rotation_speed, 0.1, 5.0)

        # Visualization
        ui.separator()
        ui.text("Visualization")
        _, self.show_directors = ui.checkbox("Show Directors", self.show_directors)
        _, self.director_scale = ui.slider_float("Director Scale", self.director_scale, 0.01, 0.1)

        # Simulation Controls
        ui.separator()
        ui.text("Simulation Controls")
        ui.text("  R: Reset simulation")
        ui.text("  G: Toggle gravity")

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)

        # Visualize material frames
        if self.show_directors:
            wp.launch(
                kernel=compute_director_lines_kernel,
                dim=self.num_stretch * 3,
                inputs=[
                    self.state_0.particle_q,
                    self.cosserat_rod.edge_q,
                    self.num_stretch,
                    self.director_scale,
                ],
                outputs=[
                    self.director_line_starts,
                    self.director_line_ends,
                    self.director_line_colors,
                ],
                device=self.model.device,
            )
            self.viewer.log_lines(
                "/directors",
                self.director_line_starts,
                self.director_line_ends,
                self.director_line_colors,
            )
        else:
            self.viewer.log_lines("/directors", None, None, None)

        self.viewer.end_frame()


if __name__ == "__main__":
    viewer, args = newton.examples.init()

    if isinstance(viewer, newton.viewer.ViewerGL):
        viewer.show_particles = True

    example = Example(viewer, args)
    newton.examples.run(example, args)
