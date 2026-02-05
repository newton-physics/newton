# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

###########################################################################
# Example Poker Cards Stacking
#
# This simulation demonstrates 52 poker cards (13 ranks x 4 suits) dropping
# and stacking on a cube, then being knocked off by a sphere. The cards use
# high bending stiffness to maintain their rigid shape while still being
# flexible enough to interact naturally.
#
# Standard poker card dimensions:
# - Width: 6.35 cm (2.5 inches) = 0.0635 m
# - Height: 8.89 cm (3.5 inches) = 0.0889 m
# - Resolution: 4x6 cells per card
#
# Command: uv run -m newton.examples multiphysics.example_poker_cards_stacking
#
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples


class Example:
    def __init__(self, viewer, args=None):
        self.viewer = viewer
        self.sim_time = 0.0

        # Simulation parameters
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 20
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.iterations = 10

        # Standard poker card dimensions in meters
        self.card_width = 0.0635  # m (6.35 cm / 2.5 inches)
        self.card_height = 0.0889  # m (8.89 cm / 3.5 inches)

        # Card resolution: 4x6 cells
        self.dim_x = 4  # cells along width
        self.dim_y = 6  # cells along height
        self.cell_x = self.card_width / self.dim_x  # ~0.0159 m
        self.cell_y = self.card_height / self.dim_y  # ~0.0148 m

        # Number of cards: 52 (13 ranks x 4 suits)
        self.num_cards = 52

        # Cube (table/platform) parameters in meters
        self.cube_size = 0.1  # m (5 cm) - half-size of the cube
        self.cube_height = 0.10  # m (10 cm) - height of cube center above ground

        # Card drop parameters in meters
        # Cards drop onto the cube surface (cube_height + cube_size = top of cube)
        self.drop_height_base = self.cube_height + self.cube_size + 0.05  # m
        self.card_spacing_z = 0.001  # m (0.1 cm) - vertical spacing between cards
        self.random_offset_xy = 0.005  # m (0.5 cm) - random XY offset

        # Build the model (using meters)
        builder = newton.ModelBuilder(gravity=-9.8)  # m/s²

        # Add a static cube for cards to stack on
        body_cube = builder.add_body(
            xform=wp.transform(
                p=wp.vec3(0.0, 0.0, self.cube_height),
                q=wp.quat_identity(),
            ),
            key="cube",
        )
        cube_cfg = newton.ModelBuilder.ShapeConfig()
        cube_cfg.density = 0.0  # Static body (infinite mass)
        cube_cfg.ke = 5.0e6  # Contact stiffness
        cube_cfg.kd = 1.0e-4  # Contact damping
        cube_cfg.mu = 0.1  # Friction
        builder.add_shape_box(
            body_cube,
            hx=self.cube_size,
            hy=self.cube_size,
            hz=self.cube_size,
            cfg=cube_cfg,
        )

        # Add a kinematic sphere to knock off the cards
        # Sphere starts to the side and moves toward the card pile
        self.sphere_radius = 0.02  # m (2 cm radius)
        self.sphere_start_x = -0.35  # m - start position to the left
        # Position sphere at card pile height (top of cube + some offset)
        # cube top is at cube_height + cube_size = 0.1 + 0.1 = 0.2m
        self.sphere_height = 0.22  # m - at card pile level
        self.sphere_velocity_x = 0.5  # m/s - velocity toward cards

        body_sphere = builder.add_body(
            xform=wp.transform(
                p=wp.vec3(self.sphere_start_x, 0.0, self.sphere_height),
                q=wp.quat_identity(),
            ),
            key="sphere",
        )
        sphere_cfg = newton.ModelBuilder.ShapeConfig()
        sphere_cfg.density = 0.0  # Kinematic body (not affected by gravity)
        sphere_cfg.ke = 1.0e5  # Contact stiffness
        sphere_cfg.kd = 1.0e-4  # Contact damping
        sphere_cfg.mu = 0.3  # Friction
        builder.add_shape_sphere(body_sphere, radius=self.sphere_radius, cfg=sphere_cfg)

        # Sphere body index for kinematic animation
        self.sphere_body_index = 1  # Second body (after cube)

        # Random generator for reproducible random offsets
        rng = np.random.default_rng(42)

        # Card mass properties
        # Real card: ~1.8g = 0.0018 kg
        # For a 4x6 grid, there are 5x7 = 35 particles
        card_mass_total = 1.8e-3  # kg (1.8 grams)
        num_particles_per_card = (self.dim_x + 1) * (self.dim_y + 1)  # 5 * 7 = 35
        card_mass_per_particle = card_mass_total / num_particles_per_card

        # High bending stiffness for stiff cards
        # tri_ke/tri_ka: in-plane stretch stiffness
        # edge_ke: bending stiffness (key for card rigidity)
        tri_ke = 1.0e4  # High stretch stiffness
        tri_ka = 1.0e4  # High shear stiffness
        tri_kd = 1.0e-4  # Small damping
        edge_ke = 1.0e2  # High bending stiffness for rigid cards
        edge_kd = 1.0e-2  # Bending damping

        # Particle radius for collision (in meters)
        particle_radius = 0.003  # m (0.15 cm)

        # Add 52 cards
        for i in range(self.num_cards):
            # Calculate drop position with slight random offset
            offset_x = rng.uniform(-self.random_offset_xy, self.random_offset_xy)
            offset_y = rng.uniform(-self.random_offset_xy, self.random_offset_xy)

            # Cards drop from different heights
            drop_z = self.drop_height_base + i * self.card_spacing_z

            # Random rotation around Z-axis for natural stacking
            random_angle = rng.uniform(-0.1, 0.1)  # Small random rotation

            # Card center position (offset so card center is at origin)
            pos_x = -self.card_width / 2 + offset_x
            pos_y = -self.card_height / 2 + offset_y
            pos_z = drop_z

            builder.add_cloth_grid(
                pos=wp.vec3(pos_x, pos_y, pos_z),
                rot=wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), random_angle),
                vel=wp.vec3(0.0, 0.0, 0.0),
                dim_x=self.dim_x,
                dim_y=self.dim_y,
                cell_x=self.cell_x,
                cell_y=self.cell_y,
                mass=card_mass_per_particle,
                fix_left=False,
                fix_right=False,
                fix_top=False,
                fix_bottom=False,
                tri_ke=tri_ke,
                tri_ka=tri_ka,
                tri_kd=tri_kd,
                edge_ke=edge_ke,
                edge_kd=edge_kd,
                particle_radius=particle_radius,
            )

        # Add ground plane
        builder.add_ground_plane()

        # Color the mesh for VBD solver (include bending constraints)
        builder.color(include_bending=True)

        # Finalize model
        self.model = builder.finalize()

        # Contact parameters for card-card and card-ground interactions
        self.model.soft_contact_ke = 1.0e5  # Contact stiffness
        self.model.soft_contact_kd = 1.0e-4  # Contact damping
        self.model.soft_contact_mu = 0.3  # Friction coefficient

        # Create VBD solver with self-contact enabled
        self.solver = newton.solvers.SolverVBD(
            model=self.model,
            iterations=self.iterations,
            particle_enable_self_contact=True,
            particle_self_contact_radius=0.001,  # m (0.1 cm)
            particle_self_contact_margin=0.0015,  # m (0.15 cm)
            particle_topological_contact_filter_threshold=2,
            particle_rest_shape_contact_exclusion_radius=0.0,  # m (0.5 cm)
        )

        # Create states
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        # Track sphere position for kinematic animation
        self.sphere_current_x = self.sphere_start_x

        # Create collision pipeline for ground and cube contact
        self.collision_pipeline = newton.CollisionPipelineUnified.from_model(
            self.model,
            broad_phase_mode=newton.BroadPhaseMode.NXN,
            soft_contact_margin=0.005,  # m (0.5 cm)
        )
        self.contacts = self.collision_pipeline.collide(self.model, self.state_0)

        self.viewer.set_model(self.model)

        # Set camera to view the stacking
        self.viewer.set_camera(
            pos=wp.vec3(0.5, -0.5, 0.3),
            pitch=-20.0,
            yaw=90.0,
        )
        if hasattr(self.viewer, "camera") and hasattr(self.viewer.camera, "fov"):
            self.viewer.camera.fov = 60.0

        self.capture()

    def capture(self):
        # Disable CUDA graph capture because we do kinematic animation
        # with numpy operations that require CPU-GPU transfers each frame
        self.graph = None

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()

            # Apply viewer forces (for interactive manipulation)
            self.viewer.apply_forces(self.state_0)

            # Animate kinematic sphere (move it toward the cards)
            self.sphere_current_x += self.sphere_velocity_x * self.sim_dt
            body_q = self.state_0.body_q.numpy()
            # Update sphere position (body_q stores transforms as 7 floats: px, py, pz, qx, qy, qz, qw)
            body_q[self.sphere_body_index][0] = self.sphere_current_x
            body_q[self.sphere_body_index][1] = 0.0
            body_q[self.sphere_body_index][2] = self.sphere_height
            self.state_0.body_q = wp.array(body_q, dtype=wp.transform)

            # Collision detection
            self.contacts = self.collision_pipeline.collide(self.model, self.state_0)

            # Solver step
            self.solver.step(
                self.state_0,
                self.state_1,
                self.control,
                self.contacts,
                self.sim_dt,
            )
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
        self.viewer.end_frame()

    def test_final(self):
        """Verify simulation reached a valid end state."""
        particle_q = self.state_0.particle_q.numpy()
        particle_qd = self.state_0.particle_qd.numpy()

        # Check velocity (cards should be settling)
        max_vel = np.max(np.linalg.norm(particle_qd, axis=1))
        assert max_vel < 0.5, f"Cards moving too fast: max_vel={max_vel:.4f} m/s"

        # Check bbox size is reasonable (not exploding)
        min_pos = np.min(particle_q, axis=0)
        max_pos = np.max(particle_q, axis=0)
        bbox_size = np.linalg.norm(max_pos - min_pos)
        assert bbox_size < 2.0, f"Bounding box exploded: size={bbox_size:.2f}"

        # Check no excessive penetration
        assert min_pos[2] > -0.1, f"Excessive penetration: z_min={min_pos[2]:.4f}"


if __name__ == "__main__":
    # Create parser with base arguments
    parser = newton.examples.create_parser()

    # Parse arguments and initialize viewer
    viewer, args = newton.examples.init(parser)

    # Create example and run
    example = Example(viewer, args)

    newton.examples.run(example, args)
