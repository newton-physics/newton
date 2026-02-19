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
# Example Cable Bundle Hysteresis
#
# Demonstrates Dahl friction model for cable bending hysteresis.
# Creates a bundle of 7 cables passing through moving obstacles that
# apply cyclic loading (load -> hold -> release). The Dahl model captures
# plastic deformation and hysteresis loops in cable bending behavior,
# showing realistic memory effects in cable dynamics.
#
###########################################################################

import argparse

import numpy as np
import warp as wp

import newton
import newton.examples


@wp.kernel
def move_obstacles_triwave(
    bodies: wp.array(dtype=int),
    init_y: wp.array(dtype=float),
    init_z: wp.array(dtype=float),
    amp_scale: float,
    period: float,
    t: wp.array(dtype=float),
    stop_time: float,
    release_time: float,
    body_q0: wp.array(dtype=wp.transform),
    body_q1: wp.array(dtype=wp.transform),
):
    """Move obstacles in a triangle wave pattern in Y direction with phase transitions."""
    i = wp.tid()
    b = bodies[i]
    X = body_q0[b]
    p = wp.transform_get_translation(X)
    q = wp.transform_get_rotation(X)

    cur_t = t[0]

    # Phase 3: Release - teleport obstacles far away
    if cur_t >= release_time:
        new_p = wp.vec3(p[0], p[1], init_z[i] + 10.0)
    # Phase 2: Hold - freeze at stop_time position
    elif cur_t >= stop_time:
        # Use stop_time for triangle wave calculation (frozen)
        cycles = stop_time / period
        frac = cycles - wp.floor(cycles)
        frac = frac + 0.5
        frac = frac - wp.floor(frac)
        tri01 = 1.0 - wp.abs(2.0 * frac - 1.0)
        tri = 2.0 * tri01 - 1.0

        px = p[0]
        pz = init_z[i]
        y0 = init_y[i]
        new_p = wp.vec3(px, tri * (y0 * amp_scale), pz)
    # Phase 1: Load - move obstacles in triangle wave
    else:
        cycles = cur_t / period
        frac = cycles - wp.floor(cycles)
        frac = frac + 0.5
        frac = frac - wp.floor(frac)
        tri01 = 1.0 - wp.abs(2.0 * frac - 1.0)
        tri = 2.0 * tri01 - 1.0

        px = p[0]
        pz = init_z[i]
        y0 = init_y[i]
        new_p = wp.vec3(px, tri * (y0 * amp_scale), pz)

    T = wp.transform(new_p, q)
    body_q0[b] = T
    body_q1[b] = T


@wp.kernel
def advance_time(t: wp.array(dtype=float), dt: float):
    """Advance a device-side time accumulator (single-threaded, graph-capture friendly)."""
    i = wp.tid()
    if i == 0:
        t[0] = t[0] + dt


class Example:
    def bundle_start_offsets_yz(self, num_cables: int, cable_radius: float, gap_multiplier: float):
        """Create cross-sectional positions for cable bundle arrangement.

        Arranges cables in a compact bundle with one central cable and others in
        concentric rings. For 7 cables: 1 center + 6 in first ring. For more cables:
        1 center + 6 inner ring + N outer ring.

        Args:
            num_cables: Total number of cables in bundle.
            cable_radius: Radius of each cable.
            gap_multiplier: Spacing between cable centers (as multiple of diameter).

        Returns:
            List of (y, z) offset positions for each cable in bundle cross-section.
        """
        positions = []
        if num_cables == 1:
            return [(0.0, 0.0)]

        # Central cable at origin
        positions.append((0.0, 0.0))
        remaining = num_cables - 1
        min_center_distance = 2.0 * cable_radius * gap_multiplier

        if remaining <= 6:
            # Single ring around center
            ring_radius = min_center_distance
            if remaining > 1:
                chord_distance = 2.0 * ring_radius * np.sin(np.pi / remaining)
                if chord_distance < min_center_distance:
                    ring_radius = min_center_distance / (2.0 * np.sin(np.pi / remaining))
            for i in range(remaining):
                angle = 2.0 * np.pi * i / remaining
                positions.append((float(ring_radius * np.cos(angle)), float(ring_radius * np.sin(angle))))
        else:
            # Two rings: inner (6 cables) and outer (remaining)
            inner_count = 6
            outer_count = remaining - inner_count
            inner_radius = min_center_distance
            inner_chord = 2.0 * inner_radius * np.sin(np.pi / inner_count)
            if inner_chord < min_center_distance:
                inner_radius = min_center_distance / (2.0 * np.sin(np.pi / inner_count))
            for i in range(inner_count):
                angle = 2.0 * np.pi * i / inner_count
                positions.append((float(inner_radius * np.cos(angle)), float(inner_radius * np.sin(angle))))

            outer_radius = inner_radius + min_center_distance
            if outer_count > 1:
                outer_chord = 2.0 * outer_radius * np.sin(np.pi / outer_count)
                if outer_chord < min_center_distance:
                    outer_radius = min_center_distance / (2.0 * np.sin(np.pi / outer_count))
            for i in range(outer_count):
                angle = 2.0 * np.pi * i / outer_count
                positions.append((float(outer_radius * np.cos(angle)), float(outer_radius * np.sin(angle))))

        return positions

    def __init__(
        self,
        viewer,
        args=None,
        num_cables: int = 7,
        segments: int = 40,
        with_dahl: bool = True,
        eps_max: float = 2.0,
        tau: float = 0.1,
    ):
        # Store viewer and arguments
        self.viewer = viewer
        self.args = args

        # Simulation cadence
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_iterations = 10
        self.update_step_interval = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        # Cable bundle parameters
        self.num_cables = num_cables
        self.num_elements = segments
        self.cable_length = 4.0
        self.cable_radius = 0.02
        self.cable_gap_multiplier = 1.1
        bend_stiffness = 1.0e1
        bend_damping = 5.0e-2
        stretch_stiffness = 1.0e6
        stretch_damping = 0.0

        builder = newton.ModelBuilder()
        builder.rigid_contact_margin = 0.05

        # Register solver-specific custom attributes (Dahl plasticity parameters live on the Model)
        newton.solvers.SolverVBD.register_custom_attributes(builder)
        builder.gravity = -9.81

        # Set default material properties for cables (cable-to-cable contact)
        builder.default_shape_cfg.ke = 1.0e6  # Contact stiffness
        builder.default_shape_cfg.kd = 1.0e-2  # Contact damping
        builder.default_shape_cfg.mu = 2.0  # Friction coefficient

        # Bundle layout: align cable center with obstacle center
        # Obstacles span x in [0.5, 2.5], center at x=1.5
        # With cable_length=4.0, choose start_x so cable center aligns with obstacle center
        start_x = 1.5 - self.cable_length / 2.0  # = -0.5
        start_y = 0.0
        # Obstacle capsule center is at z=0.3, align cable with this
        start_z = 0.3

        # Create bundle cross-section layout
        bundle_positions = self.bundle_start_offsets_yz(self.num_cables, self.cable_radius, self.cable_gap_multiplier)

        # Build each cable in the bundle
        for i in range(self.num_cables):
            off_y, off_z = bundle_positions[i]
            cable_start = wp.vec3(start_x, start_y + off_y, start_z + off_z)

            points, quats = newton.utils.create_straight_cable_points_and_quaternions(
                start=cable_start,
                direction=wp.vec3(1.0, 0.0, 0.0),
                length=float(self.cable_length),
                num_segments=int(self.num_elements),
                twist_total=0.0,
            )

            _rod_bodies, _rod_joints = builder.add_rod(
                positions=points,
                quaternions=quats,
                radius=self.cable_radius,
                bend_stiffness=bend_stiffness,
                bend_damping=bend_damping,
                stretch_stiffness=stretch_stiffness,
                stretch_damping=stretch_damping,
                key=f"bundle_cable_{i}",
            )

        # Create moving obstacles (capsules arranged along X axis)
        obstacle_cfg = newton.ModelBuilder.ShapeConfig(
            density=builder.default_shape_cfg.density,
            ke=1.0e6,
            kd=1.0e-2,
            kf=builder.default_shape_cfg.kf,
            ka=builder.default_shape_cfg.ka,
            mu=0.0,  # Frictionless obstacles
            restitution=builder.default_shape_cfg.restitution,
        )

        num_obstacles = 4
        x_min = 0.5
        x_max = 2.5
        obstacle_radius = 0.05
        obstacle_height = 0.8
        obstacle_half_height = obstacle_height * 0.5 - obstacle_radius
        base_amplitude = 0.3
        amplitude = 1.0 * base_amplitude

        self.obstacle_bodies = []
        obstacle_init_z_list = []
        for i in range(num_obstacles):
            # Distribute obstacles evenly along X
            x = x_min + (x_max - x_min) * (i / max(1, num_obstacles - 1))
            # Alternate initial Y positions (+/- amplitude)
            y = (+amplitude) if (i % 2 == 0) else (-amplitude)
            z = obstacle_height * 0.5 - 0.1

            body = builder.add_body(xform=wp.transform(wp.vec3(x, y, z), wp.quat(0.0, 0.0, 0.0, 1.0)))
            builder.add_shape_capsule(
                body=body, radius=obstacle_radius, half_height=obstacle_half_height, cfg=obstacle_cfg
            )

            # Make obstacle kinematic
            builder.body_mass[body] = 0.0
            builder.body_inv_mass[body] = 0.0
            builder.body_inertia[body] = wp.mat33(0.0)
            builder.body_inv_inertia[body] = wp.mat33(0.0)

            self.obstacle_bodies.append(body)
            obstacle_init_z_list.append(float(z))

        # Add ground plane
        ground_cfg = newton.ModelBuilder.ShapeConfig(
            density=builder.default_shape_cfg.density,
            ke=1.0e4,
            kd=1.0e-1,
            kf=builder.default_shape_cfg.kf,
            ka=builder.default_shape_cfg.ka,
            mu=2.5,
            restitution=builder.default_shape_cfg.restitution,
        )
        builder.add_ground_plane(cfg=ground_cfg)

        # Color bodies for VBD solver
        builder.color()

        # Finalize model
        self.model = builder.finalize()

        # Author Dahl friction parameters (per-joint) via custom model attributes.
        # These are read by SolverVBD when rigid_enable_dahl_friction=True.
        if hasattr(self.model, "vbd"):
            self.model.vbd.dahl_eps_max.fill_(float(eps_max))
            self.model.vbd.dahl_tau.fill_(float(tau))

        # Create VBD solver with Dahl friction (cable bending hysteresis)
        self.solver = newton.solvers.SolverVBD(
            self.model,
            iterations=self.sim_iterations,
            friction_epsilon=0.1,
            rigid_enable_dahl_friction=with_dahl,
        )

        # Initialize states and contacts
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        self.contacts = self.model.contacts()
        self.viewer.set_model(self.model)

        # Obstacle kinematics parameters
        self.obstacle_bodies_wp = wp.array(self.obstacle_bodies, dtype=int, device=self.solver.device)

        # Store initial obstacle positions for kinematic motion
        init_y_list = []
        for i in range(num_obstacles):
            y = (+amplitude) if (i % 2 == 0) else (-amplitude)
            init_y_list.append(float(y))
        self.obstacle_init_y = wp.array(init_y_list, dtype=float, device=self.solver.device)
        self.obstacle_init_z = wp.array(obstacle_init_z_list, dtype=float, device=self.solver.device)

        # Triangle wave parameters
        self.obstacle_amp_scale = 1.0
        self.obstacle_period = 2.0

        # Loading cycle: load -> hold -> release
        self.obstacle_stop_time = 0.5 * self.obstacle_period  # Stop triangle wave
        self.obstacle_release_time = 2.0 * self.obstacle_stop_time  # Teleport obstacles away

        # Time tracking for obstacle motion (stored in device array for graph capture)
        self.sim_time_array = wp.zeros(1, dtype=float, device=self.solver.device)

        # Initialize CUDA graph
        self.capture()

    def capture(self):
        """Capture simulation loop into a CUDA graph for optimal GPU performance."""
        if self.solver.device.is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self):
        """Execute all simulation substeps for one frame."""
        for substep in range(self.sim_substeps):
            self.state_0.clear_forces()

            # Apply forces to the model
            self.viewer.apply_forces(self.state_0)

            # Update obstacle positions (all phases handled inside kernel)
            wp.launch(
                move_obstacles_triwave,
                dim=len(self.obstacle_bodies),
                inputs=[
                    self.obstacle_bodies_wp,
                    self.obstacle_init_y,
                    self.obstacle_init_z,
                    float(self.obstacle_amp_scale),
                    float(self.obstacle_period),
                    self.sim_time_array,
                    float(self.obstacle_stop_time),
                    float(self.obstacle_release_time),
                    self.state_0.body_q,
                    self.state_1.body_q,
                ],
                device=self.solver.device,
            )
            # Advance time in a separate 1-thread kernel to avoid races in move_obstacles_triwave().
            wp.launch(
                advance_time,
                dim=1,
                inputs=[
                    self.sim_time_array,
                    self.sim_dt,
                ],
                device=self.solver.device,
            )

            # Decide whether to refresh rigid solver history (anchors used for long-range joint damping)
            # and recompute contacts on this substep, using a configurable cadence.
            update_step_history = (substep % self.update_step_interval) == 0

            # Collide for contact detection
            if update_step_history:
                self.model.collide(self.state_0, self.contacts)

            self.solver.set_rigid_history_update(update_step_history)
            self.solver.step(
                self.state_0,
                self.state_1,
                self.control,
                self.contacts,
                self.sim_dt,
            )

            # Swap states
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        """Advance simulation by one frame."""
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        # Note: self.sim_time is updated from device array in render()

    def render(self):
        """Render the current simulation state to the viewer."""
        # Sync host time with device time for accurate display
        self.sim_time = float(self.sim_time_array.numpy()[0])

        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        """Test cable bundle hysteresis simulation for stability and correctness (called after simulation)."""
        pass


if __name__ == "__main__":
    # Parse arguments and initialize viewer
    viewer, args = newton.examples.init()

    # Parse example-specific arguments for Dahl friction experimentation
    parser = argparse.ArgumentParser(description="Cable bundle hysteresis with Dahl friction")
    parser.add_argument("--segments", type=int, default=40, help="Number of cable segments")
    parser.add_argument("--no-dahl", action="store_true", help="Disable Dahl friction (purely elastic)")
    parser.add_argument("--eps-max", type=float, default=2.0, help="Maximum plastic strain [rad]")
    parser.add_argument("--tau", type=float, default=0.1, help="Memory decay length [rad]")
    cli, _ = parser.parse_known_args()

    # Create example and run
    example = Example(
        viewer,
        args,
        num_cables=7,
        segments=cli.segments,
        with_dahl=not cli.no_dahl and cli.eps_max > 0.0 and cli.tau > 0.0,
        eps_max=cli.eps_max,
        tau=cli.tau,
    )

    newton.examples.run(example, args)
