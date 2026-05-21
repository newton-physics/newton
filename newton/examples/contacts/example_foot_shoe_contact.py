# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Foot-Shoe Contact Simulation
#
# Simulates a foot mesh interacting with a shoe midsole spring grid
# using calibrated foundation material properties.
#
# Command: python -m newton.examples foot_shoe_contact
#
###########################################################################

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
from projects.digital_instron_v2.foundation import FoundationMaterial, evaluate_foundation_lengths
from projects.digital_instron_v2.geometry import _load_obj_mesh, compute_grid_neighbors
from projects.digital_instron_v2.manifest import load_manifest
from projects.digital_instron_v2.workflow import _load_spring_grid


def _colors_from_compression(compression: np.ndarray, max_compression: float) -> np.ndarray:
    normalized = np.clip(compression / max(max_compression, 1.0e-9), 0.0, 1.0)
    colors = np.empty((len(compression), 3), dtype=np.float32)
    colors[:, 0] = 0.12 + 0.88 * normalized
    colors[:, 1] = 0.55 * (1.0 - normalized) + 0.12
    colors[:, 2] = 0.95 * (1.0 - normalized) + 0.10
    return colors


def _colors_from_pressure(pressure_kpa: np.ndarray, max_pressure_kpa: float) -> np.ndarray:
    normalized = np.clip(pressure_kpa / max(max_pressure_kpa, 1.0e-9), 0.0, 1.0)
    colors = np.empty((len(pressure_kpa), 3), dtype=np.float32)

    # Classic Jet colormap approximation:
    r = np.clip(np.minimum(4.0 * normalized - 1.5, -4.0 * normalized + 4.5), 0.0, 1.0)
    g = np.clip(np.minimum(4.0 * normalized - 0.5, -4.0 * normalized + 3.5), 0.0, 1.0)
    b = np.clip(np.minimum(4.0 * normalized + 0.5, -4.0 * normalized + 2.5), 0.0, 1.0)

    colors[:, 0] = r
    colors[:, 1] = g
    colors[:, 2] = b
    return colors


def _compute_pressures(
    current_lengths: np.ndarray,
    slack_lengths: np.ndarray,
    velocities: np.ndarray,
    material: FoundationMaterial,
    spacing_m: float,
    neighbors: np.ndarray | None = None,
) -> np.ndarray:
    # 1. Strain and Ogden stress
    slack = np.maximum(slack_lengths, 1.0e-6)
    comp = np.maximum(slack - current_lengths, 0.0)
    strain = comp / slack

    lock = max(material.lock_strain, 1.0e-4)
    normalized = np.minimum(strain / lock, 0.999)
    alpha = max(material.ogden_alpha, 1.0e-4)
    ogden_stress = material.stiffness_pa * (np.power(1.0 - normalized, -alpha) - 1.0) / alpha

    # 2. Laplacian (Pasternak shear term) using vectorized lookup
    h2 = spacing_m * spacing_m
    laplacian = np.zeros_like(comp)
    if h2 > 1.0e-12 and neighbors is not None:
        n_indices = neighbors.copy()
        self_indices = np.arange(len(comp))
        for col in range(4):
            mask = n_indices[:, col] == -1
            n_indices[mask, col] = self_indices[mask]

        val_left = comp[n_indices[:, 0]]
        val_right = comp[n_indices[:, 1]]
        val_bottom = comp[n_indices[:, 2]]
        val_top = comp[n_indices[:, 3]]

        laplacian = (val_left + val_right + val_bottom + val_top - 4.0 * comp) / h2

    elastic_stress = ogden_stress - material.pasternak_stiffness_n_per_m * laplacian

    # 3. Viscous stress
    damping_strain = np.maximum(strain, 1.0e-8)
    damping_weight = np.power(damping_strain, max(material.damping_power, 0.0))
    compression_velocity = -velocities
    viscous_stress = material.damping_pa_s * damping_weight * compression_velocity

    pressures_pa = np.maximum(elastic_stress + viscous_stress, 0.0)
    return pressures_pa


class Example:
    """Foot-Shoe interactive ground contact simulation example using foundation materials."""

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.fps = 100
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.test_mode = args.test
        self.kinematic = args.kinematic

        # 1. Load resources from manifest and calibrated v2 cache
        self.manifest = load_manifest(args.manifest)
        self.output_dir = Path(args.output_dir) if args.output_dir else self.manifest.cache_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.spring_grid, self.midsole_vertices = _load_spring_grid(self.manifest, self.output_dir)

        with open(args.material) as f:
            mat_data = json.load(f)["material"]
        self.material = FoundationMaterial(**mat_data)

        self.neighbors = compute_grid_neighbors(self.spring_grid.grid_uv_m, self.spring_grid.spacing_m)
        self.max_display_pressure_kpa = 800.0
        self.min_bottom_m = np.min(self.spring_grid.bottom_m)
        self.start_z = -self.min_bottom_m + 0.005

        # 2. Load and mirror/align the foot model
        foot_v, foot_f = _load_obj_mesh(Path(args.foot_mesh))

        # Mirroring / alignment:
        # Foot: X=length (heel-to-toe), Y=width, Z=height
        # Midsole: X=width, Y=length, Z=height
        # Swapping foot length/width into midsole width/length already reflects the footprint.
        # Keep that reflected orientation for the right-foot-to-left-shoe default; add the
        # lateral sign flip only when explicitly preserving the source foot handedness.
        sign = 1.0 if args.mirror_foot else -1.0

        foot_v_transformed = np.zeros_like(foot_v)
        foot_v_transformed[:, 0] = sign * foot_v[:, 1]  # X_midsole = width
        foot_v_transformed[:, 1] = foot_v[:, 0]  # Y_midsole = length
        foot_v_transformed[:, 2] = foot_v[:, 2]  # Z_midsole = height

        foot_f_transformed = foot_f.copy()
        if args.mirror_foot:
            # The reflected transform changes triangle winding.
            foot_f_transformed[:, [1, 2]] = foot_f_transformed[:, [2, 1]]

        # Center horizontally on the spring grid (X-Y plane)
        foot_center = 0.5 * (np.min(foot_v_transformed, axis=0) + np.max(foot_v_transformed, axis=0))
        midsole_center = 0.5 * (np.min(self.spring_grid.grid_uv_m, axis=0) + np.max(self.spring_grid.grid_uv_m, axis=0))
        foot_v_transformed[:, 0] += midsole_center[0] - foot_center[0]
        foot_v_transformed[:, 1] += midsole_center[1] - foot_center[1]

        # Align foot yaw by rotating mesh coordinates
        foot_yaw = np.radians(args.foot_yaw_deg)
        cos_yaw = np.cos(foot_yaw)
        sin_yaw = np.sin(foot_yaw)

        # Rotate X and Y vertices about local center
        foot_center = 0.5 * (np.min(foot_v_transformed, axis=0) + np.max(foot_v_transformed, axis=0))
        foot_v_rel = foot_v_transformed - foot_center

        x_rot = foot_v_rel[:, 0] * cos_yaw - foot_v_rel[:, 1] * sin_yaw
        y_rot = foot_v_rel[:, 0] * sin_yaw + foot_v_rel[:, 1] * cos_yaw

        foot_v_transformed[:, 0] = x_rot + foot_center[0]
        foot_v_transformed[:, 1] = y_rot + foot_center[1]

        # Align vertically (Z-axis) so foot sole just touches the top surface of midsole
        spacing = self.spring_grid.spacing_m
        z_foot_sole = np.full(len(self.spring_grid.grid_uv_m), np.nan)
        for i, (x_g, y_g) in enumerate(self.spring_grid.grid_uv_m):
            in_cell = (np.abs(foot_v_transformed[:, 0] - x_g) <= spacing * 0.5) & (
                np.abs(foot_v_transformed[:, 1] - y_g) <= spacing * 0.5
            )
            if np.any(in_cell):
                z_foot_sole[i] = np.min(foot_v_transformed[in_cell, 2])

        valid = np.isfinite(z_foot_sole)
        if np.any(valid):
            z_offsets = self.spring_grid.top_m[valid] - z_foot_sole[valid]
            Z_offset = np.max(z_offsets)
            foot_v_transformed[:, 2] += Z_offset
            z_foot_sole[valid] += Z_offset

        self.foot_sole_z_m = z_foot_sole
        self.foot_contact_valid = np.isfinite(self.foot_sole_z_m)

        # Physical parameters for dynamic simulation
        self.mass = 80.0  # kg
        self.gravity = -9.81  # m/s^2

        # 3. Create Model
        builder = newton.ModelBuilder()
        builder.gravity = 0.0
        builder.add_ground_plane()

        # Add foot body (we will set dynamic properties but disable standard rigid collision)
        self.foot_body_id = builder.add_body(
            xform=wp.transform(p=wp.vec3(0.0, 0.0, self.start_z), q=wp.quat_identity()),
            mass=self.mass,
            inertia=wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
            lock_inertia=True,
        )

        foot_mesh = newton.Mesh(foot_v_transformed, foot_f_transformed.reshape(-1))
        cfg = newton.ModelBuilder.ShapeConfig()
        cfg.has_shape_collision = False
        cfg.has_particle_collision = False
        cfg.is_visible = True
        builder.add_shape_mesh(
            self.foot_body_id,
            mesh=foot_mesh,
            cfg=cfg,
            color=wp.vec3(0.65, 0.72, 0.88),
            label="foot_mesh",
        )

        self.model = builder.finalize()
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()
        self.solver = newton.solvers.SolverXPBD(self.model, iterations=1)

        self.device = self.model.device

        # Start height where the bottom of the midsole is 5 mm above the ground
        self.current_z = self.start_z
        self.current_vz = 0.0

        # Initialize body position in state
        body_q = self.state_0.body_q.numpy()
        body_q[self.foot_body_id, :3] = [0.0, 0.0, self.start_z]
        body_q[self.foot_body_id, 3:7] = [0.0, 0.0, 0.0, 1.0]
        self.state_0.body_q.assign(body_q)

        # Hysteresis logging
        self.peak_force_n = 0.0
        self.history_z = []
        self.history_force = []
        self.peak_compression_m = np.zeros_like(self.spring_grid.slack_length_m)
        self.peak_pressure_kpa = np.zeros_like(self.spring_grid.slack_length_m)
        self.peak_foot_top_displacement_m = np.zeros_like(self.spring_grid.slack_length_m)
        self.peak_foot_top_pressure_kpa = np.zeros_like(self.spring_grid.slack_length_m)
        self.peak_ground_bottom_displacement_m = np.zeros_like(self.spring_grid.slack_length_m)
        self.peak_ground_bottom_pressure_kpa = np.zeros_like(self.spring_grid.slack_length_m)

        # Setup viewer
        self.viewer.set_model(self.model)
        self.viewer.set_camera(
            pos=wp.vec3(0.25, -0.45, 0.22),
            pitch=-22.0,
            yaw=145.0,
        )

        self.point_radius = max(float(self.spring_grid.spacing_m) * 0.22, 0.0008)
        self.contact_color = wp.vec3(0.05, 0.95, 0.52)
        self.max_display_compression = 0.02  # 20 mm

    def _ground_bottom_displacement_m(self, body_z_m: float) -> np.ndarray:
        compression = np.maximum(-(body_z_m + self.spring_grid.bottom_m), 0.0)
        return np.minimum(compression, self.spring_grid.slack_length_m)

    def _foot_top_displacement_m(self, body_z_m: float) -> np.ndarray:
        displacement = np.zeros_like(self.spring_grid.slack_length_m)
        if np.any(self.foot_contact_valid):
            top_rest_world = self.start_z + self.spring_grid.top_m[self.foot_contact_valid]
            foot_sole_world = body_z_m + self.foot_sole_z_m[self.foot_contact_valid]
            displacement[self.foot_contact_valid] = np.maximum(top_rest_world - foot_sole_world, 0.0)
        return np.minimum(displacement, self.spring_grid.slack_length_m)

    def _lengths_and_velocities_from_displacement(
        self,
        displacement_m: np.ndarray,
        vertical_velocity_mps: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        displacement = np.minimum(np.maximum(displacement_m, 0.0), self.spring_grid.slack_length_m)
        current_lengths = np.maximum(self.spring_grid.slack_length_m - displacement, 0.0)
        velocities = np.zeros_like(self.spring_grid.slack_length_m)
        velocities[displacement > 1.0e-6] = vertical_velocity_mps
        return current_lengths, velocities

    def _pressure_kpa_from_displacement(
        self,
        displacement_m: np.ndarray,
        vertical_velocity_mps: float,
    ) -> np.ndarray:
        current_lengths, velocities = self._lengths_and_velocities_from_displacement(
            displacement_m,
            vertical_velocity_mps,
        )
        pressures_pa = _compute_pressures(
            current_lengths,
            self.spring_grid.slack_length_m,
            velocities,
            self.material,
            self.spring_grid.spacing_m,
            self.neighbors,
        )
        return pressures_pa / 1000.0

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()

            # 1. Fetch current foot rigid body state
            body_q = self.state_0.body_q.numpy()
            body_qd = self.state_0.body_qd.numpy()

            foot_pos = body_q[self.foot_body_id, :3]
            foot_vel = body_qd[self.foot_body_id, :3]

            # 2. Update position/velocity based on mode
            if self.kinematic:
                # 1 Hz frequency vertical sine trajectory
                omega = 2.0 * np.pi * 1.0
                # Move foot down by up to 25 mm past starting height
                disp = 0.0125 * (1.0 - np.cos(omega * self.sim_time))
                self.current_z = self.start_z - disp
                self.current_vz = -0.0125 * omega * np.sin(omega * self.sim_time)

                foot_pos[2] = self.current_z
                foot_vel[2] = self.current_vz

                body_q[self.foot_body_id, :3] = foot_pos
                body_qd[self.foot_body_id, :3] = foot_vel
                self.state_0.body_q.assign(body_q)
                self.state_0.body_qd.assign(body_qd)
            else:
                # Dynamic mode: foot moves under external weight force and spring feedback
                # External downward force goes up to 1000 N and back down
                omega = 2.0 * np.pi * 1.0
                ext_force = 1000.0 * max(np.sin(omega * self.sim_time * 0.5), 0.0)

                # Compute ground-side spring reaction
                ground_displacement = self._ground_bottom_displacement_m(foot_pos[2])
                current_lengths, velocity_mps = self._lengths_and_velocities_from_displacement(
                    ground_displacement,
                    foot_vel[2],
                )

                res = evaluate_foundation_lengths(
                    self.spring_grid.grid_uv_m,
                    current_lengths,
                    self.spring_grid.slack_length_m,
                    velocity_mps,
                    cell_area_m2=np.full_like(current_lengths, self.spring_grid.cell_area_m2),
                    material=self.material,
                    neighbors=self.neighbors,
                    spacing_m=self.spring_grid.spacing_m,
                    device=self.device,
                )

                # Update physics ODE using body force accumulation
                body_f = np.zeros((self.model.body_count, 6), dtype=np.float32)
                # Apply gravity and external downward force (force along Z: index 2)
                body_f[self.foot_body_id, 2] = self.mass * self.gravity - ext_force + res.force_n
                # Apply torque feedback from Pasternak/foundation
                body_f[self.foot_body_id, 3] = res.wrench[3]
                body_f[self.foot_body_id, 4] = res.wrench[4]

                self.state_0.body_f.assign(body_f)

                self.viewer.apply_forces(self.state_0)
                self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
                self.state_0, self.state_1 = self.state_1, self.state_0

                body_q = self.state_0.body_q.numpy()
                body_qd = self.state_0.body_qd.numpy()
                foot_pos = body_q[self.foot_body_id, :3]
                foot_vel = body_qd[self.foot_body_id, :3]

            # 3. Evaluate spring states for visualization and logging
            ground_displacement = self._ground_bottom_displacement_m(foot_pos[2])
            current_lengths, velocity_mps = self._lengths_and_velocities_from_displacement(
                ground_displacement,
                foot_vel[2],
            )

            res = evaluate_foundation_lengths(
                self.spring_grid.grid_uv_m,
                current_lengths,
                self.spring_grid.slack_length_m,
                velocity_mps,
                cell_area_m2=np.full_like(current_lengths, self.spring_grid.cell_area_m2),
                material=self.material,
                neighbors=self.neighbors,
                spacing_m=self.spring_grid.spacing_m,
                device=self.device,
            )

            self.last_force_n = res.force_n
            if res.force_n > self.peak_force_n:
                self.peak_force_n = res.force_n

            # Track independent peak displacement and pressure on both sides of the midsole.
            foot_top_displacement = self._foot_top_displacement_m(foot_pos[2])
            ground_pressure_kpa = self._pressure_kpa_from_displacement(ground_displacement, foot_vel[2])
            foot_pressure_kpa = self._pressure_kpa_from_displacement(foot_top_displacement, foot_vel[2])
            self.peak_ground_bottom_displacement_m = np.maximum(
                self.peak_ground_bottom_displacement_m,
                ground_displacement,
            )
            self.peak_ground_bottom_pressure_kpa = np.maximum(
                self.peak_ground_bottom_pressure_kpa,
                ground_pressure_kpa,
            )
            self.peak_foot_top_displacement_m = np.maximum(self.peak_foot_top_displacement_m, foot_top_displacement)
            self.peak_foot_top_pressure_kpa = np.maximum(self.peak_foot_top_pressure_kpa, foot_pressure_kpa)

            # Backward-compatible names for the original ground-side heatmap outputs.
            self.peak_compression_m = self.peak_ground_bottom_displacement_m
            self.peak_pressure_kpa = self.peak_ground_bottom_pressure_kpa

            self.history_z.append(foot_pos[2])
            self.history_force.append(res.force_n)

            # Advance simulation time
            self.sim_time += self.sim_dt

    def step(self):
        self.simulate()

    def render(self):
        body_q = self.state_0.body_q.numpy()
        body_qd = self.state_0.body_qd.numpy()
        foot_pos = body_q[self.foot_body_id, :3]
        foot_vel = body_qd[self.foot_body_id, :3]

        z_bottom_world = foot_pos[2] + self.spring_grid.bottom_m
        ground_displacement = self._ground_bottom_displacement_m(foot_pos[2])
        foot_top_displacement = self._foot_top_displacement_m(foot_pos[2])
        ground_pressures_kpa = self._pressure_kpa_from_displacement(ground_displacement, foot_vel[2])
        foot_pressures_kpa = self._pressure_kpa_from_displacement(foot_top_displacement, foot_vel[2])

        # Deformed spring lines
        deformed_top_z = foot_pos[2] + self.spring_grid.top_m
        deformed_bottom_z = np.maximum(z_bottom_world, 0.0)

        spring_starts = np.column_stack(
            (self.spring_grid.grid_uv_m[:, 0], self.spring_grid.grid_uv_m[:, 1], deformed_bottom_z)
        ).astype(np.float32)
        spring_ends = np.column_stack(
            (self.spring_grid.grid_uv_m[:, 0], self.spring_grid.grid_uv_m[:, 1], deformed_top_z)
        ).astype(np.float32)

        colors = _colors_from_compression(ground_displacement, self.max_display_compression)
        spring_colors = colors[ground_displacement > 1.0e-6]
        if len(spring_colors) == 0:
            spring_colors = colors[:1]

        active_starts = spring_starts[ground_displacement > 1.0e-6]
        active_ends = spring_ends[ground_displacement > 1.0e-6]
        if len(active_starts) == 0:
            active_starts = spring_starts[:1]
            active_ends = spring_ends[:1]

        ground_contact_points = spring_starts[ground_displacement > 1.0e-6]
        active_ground_pressures = ground_pressures_kpa[ground_displacement > 1.0e-6]
        if len(ground_contact_points) == 0:
            ground_contact_points = spring_starts[:1]
            ground_contact_colors = np.full((1, 3), [0.0, 0.0, 0.5], dtype=np.float32)
        else:
            ground_contact_colors = _colors_from_pressure(active_ground_pressures, self.max_display_pressure_kpa)

        foot_top_points = spring_ends[foot_top_displacement > 1.0e-6]
        active_foot_pressures = foot_pressures_kpa[foot_top_displacement > 1.0e-6]
        if len(foot_top_points) == 0:
            foot_top_points = spring_ends[:1]
            foot_top_colors = np.full((1, 3), [0.0, 0.0, 0.5], dtype=np.float32)
        else:
            foot_top_colors = _colors_from_pressure(active_foot_pressures, self.max_display_pressure_kpa)

        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_points(
            "/foot_shoe/midsole_bottom_undeformed",
            wp.array(spring_starts, dtype=wp.vec3),
            self.point_radius,
            wp.full(len(spring_starts), wp.vec3(0.24, 0.26, 0.28), dtype=wp.vec3),
        )
        self.viewer.log_lines(
            "/foot_shoe/spring_grid",
            wp.array(active_starts, dtype=wp.vec3),
            wp.array(active_ends, dtype=wp.vec3),
            wp.array(spring_colors, dtype=wp.vec3),
        )
        self.viewer.log_points(
            "/foot_shoe/ground_pressure_surface",
            wp.array(ground_contact_points, dtype=wp.vec3),
            self.point_radius * 1.8,
            wp.array(ground_contact_colors, dtype=wp.vec3),
        )
        self.viewer.log_points(
            "/foot_shoe/foot_pressure_surface",
            wp.array(foot_top_points, dtype=wp.vec3),
            self.point_radius * 1.8,
            wp.array(foot_top_colors, dtype=wp.vec3),
        )
        self.viewer.log_array(
            "/foot_shoe/stats",
            np.asarray(
                [
                    self.sim_time,
                    foot_pos[2],
                    float(self.last_force_n),
                    float(self.peak_force_n),
                    float(np.max(foot_top_displacement)),
                    float(np.max(ground_displacement)),
                ],
                dtype=np.float32,
            ),
        )
        self.viewer.end_frame()

    def test_final(self):
        print(f"Simulation completed. Peak Vertical Force reached: {self.peak_force_n:.2f} N")
        assert self.peak_force_n >= 1000.0, (
            f"Expected peak vertical force to reach 1000 N, but only reached {self.peak_force_n:.2f} N"
        )
        self._plot()

    def _plot(self):
        """Save simulation diagnostics plots to a PNG file and a GIF animation."""
        try:
            import matplotlib.pyplot as plt  # noqa: PLC0415
        except ImportError:
            print("matplotlib is not installed. Skipping plot generation.")
            return

        if len(self.history_z) == 0:
            print("No history recorded. Skipping plot generation.")
            return

        n = len(self.history_z)
        time = np.arange(n, dtype=np.float32) * self.sim_dt
        pos_z = np.array(self.history_z)
        force_n = np.array(self.history_force)

        # Displacement (from start height start_z) in mm
        # Top Surface (Foot sole) moves with pos_z:
        disp_top_mm = (self.start_z - pos_z) * 1000.0

        # Bottom Surface (Outsole) is compressed by contact.
        # Compute bottom surface displacement for each frame:
        disp_bottom_mm = []
        for z in pos_z:
            deformed_bottom_z = np.maximum(z + self.spring_grid.bottom_m, 0.0)
            cell_disp = (self.start_z + self.spring_grid.bottom_m - deformed_bottom_z) * 1000.0
            disp_bottom_mm.append(np.mean(cell_disp))
        disp_bottom_mm = np.array(disp_bottom_mm)

        _fig, axs = plt.subplots(1, 2, figsize=(12, 5))

        # Subplot 1: Force and Displacement vs Time
        color = "tab:red"
        axs[0].set_xlabel("Time [s]")
        axs[0].set_ylabel("Force [N]", color=color)
        axs[0].plot(time, force_n, color=color, linewidth=2, label="Force")
        axs[0].tick_params(axis="y", labelcolor=color)
        axs[0].grid(True)

        axs0_twin = axs[0].twinx()
        axs0_twin.set_ylabel("Displacement [mm]")
        axs0_twin.plot(time, disp_top_mm, color="tab:blue", linewidth=2, label="Top (Foot)")
        axs0_twin.plot(time, disp_bottom_mm, color="tab:cyan", linewidth=2, linestyle="--", label="Bottom (Outsole)")
        axs0_twin.tick_params(axis="y")
        axs0_twin.legend(loc="upper right")
        axs0_twin.set_title("Force & Displacement vs Time")

        # Subplot 2: Force vs Displacement (Hysteresis Loops)
        axs[1].plot(disp_top_mm, force_n, color="purple", linewidth=2.5, label="Top (Foot)")
        axs[1].plot(disp_bottom_mm, force_n, color="green", linewidth=2.0, linestyle="--", label="Bottom (Outsole)")
        axs[1].set_xlabel("Displacement [mm]")
        axs[1].set_ylabel("Force [N]")
        axs[1].set_title("Force-Displacement Hysteresis Loops")
        axs[1].legend()
        axs[1].grid(True)

        plt.tight_layout()
        plot_path = "foot_shoe_hysteresis.png"
        plt.savefig(plot_path, dpi=150)
        print(f"Hysteresis plot saved to {plot_path}")
        plt.close()

        # Generate independent peak heatmaps for the foot-side top surface and ground-side bottom surface.
        fig2, axs2 = plt.subplots(2, 2, figsize=(12, 10))

        sc1 = axs2[0, 0].scatter(
            self.spring_grid.grid_uv_m[:, 0] * 1000.0,
            self.spring_grid.grid_uv_m[:, 1] * 1000.0,
            c=self.peak_foot_top_displacement_m * 1000.0,
            s=18,
            cmap="inferno",
        )
        axs2[0, 0].set_title("Peak Top Displacement from Foot Mesh")
        axs2[0, 0].set_aspect("equal", adjustable="box")
        axs2[0, 0].set_xlabel("Width [mm]")
        axs2[0, 0].set_ylabel("Length [mm]")
        fig2.colorbar(sc1, ax=axs2[0, 0], label="Displacement [mm]")

        sc2 = axs2[0, 1].scatter(
            self.spring_grid.grid_uv_m[:, 0] * 1000.0,
            self.spring_grid.grid_uv_m[:, 1] * 1000.0,
            c=self.peak_foot_top_pressure_kpa,
            s=18,
            cmap="jet",
        )
        axs2[0, 1].set_title("Peak Top Pressure from Foot Mesh")
        axs2[0, 1].set_aspect("equal", adjustable="box")
        axs2[0, 1].set_xlabel("Width [mm]")
        axs2[0, 1].set_ylabel("Length [mm]")
        fig2.colorbar(sc2, ax=axs2[0, 1], label="Pressure [kPa]")

        sc3 = axs2[1, 0].scatter(
            self.spring_grid.grid_uv_m[:, 0] * 1000.0,
            self.spring_grid.grid_uv_m[:, 1] * 1000.0,
            c=self.peak_ground_bottom_displacement_m * 1000.0,
            s=18,
            cmap="inferno",
        )
        axs2[1, 0].set_title("Peak Bottom Displacement from Ground")
        axs2[1, 0].set_aspect("equal", adjustable="box")
        axs2[1, 0].set_xlabel("Width [mm]")
        axs2[1, 0].set_ylabel("Length [mm]")
        fig2.colorbar(sc3, ax=axs2[1, 0], label="Displacement [mm]")

        sc4 = axs2[1, 1].scatter(
            self.spring_grid.grid_uv_m[:, 0] * 1000.0,
            self.spring_grid.grid_uv_m[:, 1] * 1000.0,
            c=self.peak_ground_bottom_pressure_kpa,
            s=18,
            cmap="jet",
        )
        axs2[1, 1].set_title("Peak Bottom Pressure from Ground")
        axs2[1, 1].set_aspect("equal", adjustable="box")
        axs2[1, 1].set_xlabel("Width [mm]")
        axs2[1, 1].set_ylabel("Length [mm]")
        fig2.colorbar(sc4, ax=axs2[1, 1], label="Pressure [kPa]")

        plt.tight_layout()
        heatmap_path = "foot_shoe_peak_heatmap.png"
        fig2.savefig(heatmap_path, dpi=150)
        print(f"Peak heatmaps saved to {heatmap_path}")
        plt.close(fig2)

        # Generate Frame-by-Frame Heatmap GIF Animation
        try:
            from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: PLC0415

            # Subsample frames to keep animation generation fast (around 60 frames max)
            max_anim_frames = 60
            if n > max_anim_frames:
                indices = np.linspace(0, n - 1, max_anim_frames, dtype=np.int64)
            else:
                indices = np.arange(n, dtype=np.int64)

            fig_anim, axs_anim = plt.subplots(2, 2, figsize=(12, 10))
            vels_z = np.gradient(pos_z, time) if n > 1 else np.zeros_like(pos_z)

            def frame_maps(z, v):
                foot_disp = self._foot_top_displacement_m(z)
                ground_disp = self._ground_bottom_displacement_m(z)
                foot_pressure = self._pressure_kpa_from_displacement(foot_disp, v)
                ground_pressure = self._pressure_kpa_from_displacement(ground_disp, v)
                return foot_disp, foot_pressure, ground_disp, ground_pressure

            # Initial frame values
            z_init = pos_z[indices[0]]
            vel_init = vels_z[indices[0]]
            foot_disp_init, foot_pressure_init, ground_disp_init, ground_pressure_init = frame_maps(z_init, vel_init)

            foot_disp_vmax = (
                np.max(self.peak_foot_top_displacement_m * 1000.0)
                if np.max(self.peak_foot_top_displacement_m) > 0.0
                else 1.0
            )
            ground_disp_vmax = (
                np.max(self.peak_ground_bottom_displacement_m * 1000.0)
                if np.max(self.peak_ground_bottom_displacement_m) > 0.0
                else 1.0
            )

            sc_foot_disp = axs_anim[0, 0].scatter(
                self.spring_grid.grid_uv_m[:, 0] * 1000.0,
                self.spring_grid.grid_uv_m[:, 1] * 1000.0,
                c=foot_disp_init * 1000.0,
                s=18,
                cmap="inferno",
                vmin=0.0,
                vmax=foot_disp_vmax,
            )
            axs_anim[0, 0].set_title("Top Displacement from Foot [mm]")
            axs_anim[0, 0].set_aspect("equal", adjustable="box")
            axs_anim[0, 0].set_xlabel("Width [mm]")
            axs_anim[0, 0].set_ylabel("Length [mm]")
            fig_anim.colorbar(sc_foot_disp, ax=axs_anim[0, 0], label="Displacement [mm]")

            sc_foot_pres = axs_anim[0, 1].scatter(
                self.spring_grid.grid_uv_m[:, 0] * 1000.0,
                self.spring_grid.grid_uv_m[:, 1] * 1000.0,
                c=foot_pressure_init,
                s=18,
                cmap="jet",
                vmin=0.0,
                vmax=self.max_display_pressure_kpa,
            )
            axs_anim[0, 1].set_title("Top Pressure from Foot [kPa]")
            axs_anim[0, 1].set_aspect("equal", adjustable="box")
            axs_anim[0, 1].set_xlabel("Width [mm]")
            axs_anim[0, 1].set_ylabel("Length [mm]")
            fig_anim.colorbar(sc_foot_pres, ax=axs_anim[0, 1], label="Pressure [kPa]")

            sc_ground_disp = axs_anim[1, 0].scatter(
                self.spring_grid.grid_uv_m[:, 0] * 1000.0,
                self.spring_grid.grid_uv_m[:, 1] * 1000.0,
                c=ground_disp_init * 1000.0,
                s=18,
                cmap="inferno",
                vmin=0.0,
                vmax=ground_disp_vmax,
            )
            axs_anim[1, 0].set_title("Bottom Displacement from Ground [mm]")
            axs_anim[1, 0].set_aspect("equal", adjustable="box")
            axs_anim[1, 0].set_xlabel("Width [mm]")
            axs_anim[1, 0].set_ylabel("Length [mm]")
            fig_anim.colorbar(sc_ground_disp, ax=axs_anim[1, 0], label="Displacement [mm]")

            sc_ground_pres = axs_anim[1, 1].scatter(
                self.spring_grid.grid_uv_m[:, 0] * 1000.0,
                self.spring_grid.grid_uv_m[:, 1] * 1000.0,
                c=ground_pressure_init,
                s=18,
                cmap="jet",
                vmin=0.0,
                vmax=self.max_display_pressure_kpa,
            )
            axs_anim[1, 1].set_title("Bottom Pressure from Ground [kPa]")
            axs_anim[1, 1].set_aspect("equal", adjustable="box")
            axs_anim[1, 1].set_xlabel("Width [mm]")
            axs_anim[1, 1].set_ylabel("Length [mm]")
            fig_anim.colorbar(sc_ground_pres, ax=axs_anim[1, 1], label="Pressure [kPa]")

            title = fig_anim.suptitle("")

            def update_anim(frame_idx):
                idx = indices[frame_idx]
                z = pos_z[idx]
                v = vels_z[idx]
                t_val = time[idx]
                f_val = force_n[idx]

                foot_disp, foot_pressure, ground_disp, ground_pressure = frame_maps(z, v)

                sc_foot_disp.set_array(foot_disp * 1000.0)
                sc_foot_pres.set_array(foot_pressure)
                sc_ground_disp.set_array(ground_disp * 1000.0)
                sc_ground_pres.set_array(ground_pressure)
                title.set_text(f"Foot-Shoe Impact | t={t_val:.3f} s | Force={f_val:.1f} N")
                return sc_foot_disp, sc_foot_pres, sc_ground_disp, sc_ground_pres, title

            anim_path = "foot_shoe_contact_heatmap.gif"
            anim = FuncAnimation(fig_anim, update_anim, frames=len(indices), interval=100, blit=False)
            anim.save(anim_path, writer=PillowWriter(fps=10), dpi=100)
            print(f"Heatmap video saved to {anim_path}")
            plt.close(fig_anim)
        except Exception as e:
            print(f"Failed to generate animation: {e}")

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--manifest", default="DigitalInstron/manifest_v2.json", help="Path to manifest file")
        parser.add_argument("--output-dir", default=None, help="Directory for cache")
        parser.add_argument(
            "--material",
            default="DigitalInstron/processed/v2_cache/digital_instron_v2_foundation_material.json",
            help="Material JSON path",
        )
        parser.add_argument("--foot-mesh", default="FeetFinder/0002-B.obj", help="Foot OBJ mesh path")
        parser.add_argument(
            "--mirror-foot",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Mirror the right-foot mesh laterally to match the left shoe bed",
        )
        parser.add_argument(
            "--kinematic",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Run kinematic trajectory or dynamic simulation",
        )
        parser.add_argument(
            "--foot-yaw-deg",
            type=float,
            default=0.0,
            help="Rotation angle of foot about Z-axis in degrees",
        )
        return parser


if __name__ == "__main__":
    import argparse

    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
    if not args.test:
        example.test_final()
