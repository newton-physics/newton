# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Selection Deformables
#
# Selects labeled curves, surfaces, and volumes across replicated worlds.
# Demonstrates per-world and flat-group resets plus setup-time topology ranges.
#
# Command: python -m newton.examples selection_deformables
#
###########################################################################

from __future__ import annotations

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.selection import DeformableView


def _add_surface(builder: newton.ModelBuilder, label: str, y: float, z: float) -> None:
    vertices = [
        (-0.25, y - 0.25, z),
        (0.25, y - 0.25, z),
        (0.25, y + 0.25, z),
        (-0.25, y + 0.25, z),
    ]
    builder.add_cloth_mesh(
        pos=wp.vec3(0.0),
        rot=wp.quat_identity(),
        scale=1.0,
        vel=wp.vec3(0.0),
        vertices=vertices,
        indices=[0, 1, 2, 0, 2, 3],
        density=0.1,
        tri_ke=1.0e3,
        tri_ka=1.0e3,
        tri_kd=1.0e-2,
        edge_ke=10.0,
        edge_kd=1.0e-2,
        label=label,
    )


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.world_count = args.world_count
        if self.world_count < 2:
            raise ValueError("selection_deformables requires at least two worlds")

        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 6
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.next_reset = 2.0

        prototype = newton.ModelBuilder()

        # The labels are the public identities used after finalization.
        curve_bodies, _curve_joints = prototype.add_rod_graph(
            node_positions=[
                (-0.8, -0.3, 1.8),
                (-0.8, -0.1, 1.55),
                (-0.8, 0.1, 1.3),
                (-0.8, 0.3, 1.05),
            ],
            edges=[(0, 1), (1, 2), (2, 3)],
            radius=0.025,
            stretch_stiffness=1.0e5,
            bend_stiffness=1.0e2,
            label="hanging_cable",
            wrap_in_articulation=True,
            body_frame_origin="com",
        )
        pinned_body = curve_bodies[0]
        prototype.body_mass[pinned_body] = 0.0
        prototype.body_inv_mass[pinned_body] = 0.0
        prototype.body_inertia[pinned_body] = wp.mat33(0.0)
        prototype.body_inv_inertia[pinned_body] = wp.mat33(0.0)

        _add_surface(prototype, "surface_primary", y=-0.35, z=1.5)
        _add_surface(prototype, "surface_secondary", y=0.35, z=1.8)

        prototype.add_soft_grid(
            pos=wp.vec3(0.55, -0.2, 2.2),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=1,
            dim_y=1,
            dim_z=1,
            cell_x=0.35,
            cell_y=0.35,
            cell_z=0.35,
            density=100.0,
            k_mu=1.0e3,
            k_lambda=1.0e3,
            k_damp=1.0e-2,
            label="soft_cube",
        )

        scene = newton.ModelBuilder()
        scene.add_ground_plane()
        scene.replicate(prototype, world_count=self.world_count, spacing=(2.2, 2.2, 0.0))
        scene.color(balance_colors=False)
        self.model = scene.finalize()

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()
        self.solver = newton.solvers.SolverXPBD(self.model, iterations=5)

        self.curve = DeformableView(self.model, "hanging_cable")
        self.surface = DeformableView(self.model, "surface_primary")
        self.surfaces = DeformableView(self.model, "surface_*", family="surface")
        self.volume = DeformableView(self.model, "soft_cube")

        # These host ranges are sufficient for one-time renderer/Fabric offset setup.
        self.curve_body_ranges = self.curve.ranges("body")
        self.surface_particle_ranges = self.surface.ranges("particle")
        self.volume_particle_ranges = self.volume.ranges("particle")

        # The exact-label view has one primary cloth per world, so its flat group
        # indices coincide with model world IDs.
        positions_primary_default = self.surface.get_particle_positions(self.model).numpy()
        velocities_primary_default = self.surface.get_particle_velocities(self.model).numpy()
        group_indices_primary_host = np.arange(0, self.world_count, 2, dtype=np.int32)
        positions_primary_reset = positions_primary_default[group_indices_primary_host].copy()
        positions_primary_reset[:, :, 2] += 0.25
        self.group_indices_primary = wp.array(
            group_indices_primary_host,
            dtype=wp.int32,
            device=self.model.device,
        )
        self.positions_primary_reset = wp.array(
            positions_primary_reset,
            dtype=wp.vec3,
            device=self.model.device,
        )
        self.velocities_primary_reset = wp.array(
            velocities_primary_default[group_indices_primary_host],
            dtype=wp.vec3,
            device=self.model.device,
        )

        # The wildcard view has two rows per world. Rows 3, 7, ... are the
        # secondary cloths in odd worlds, so they use flat group indices.
        positions_surface_default = self.surfaces.get_particle_positions(self.model).numpy()
        velocities_surface_default = self.surfaces.get_particle_velocities(self.model).numpy()
        group_indices_secondary_host = np.arange(3, 2 * self.world_count, 4, dtype=np.int32)
        positions_secondary_reset = positions_surface_default[group_indices_secondary_host].copy()
        positions_secondary_reset[:, :, 0] += 0.15
        self.group_indices_secondary = wp.array(
            group_indices_secondary_host,
            dtype=wp.int32,
            device=self.model.device,
        )
        self.positions_secondary_reset = wp.array(
            positions_secondary_reset,
            dtype=wp.vec3,
            device=self.model.device,
        )
        self.velocities_secondary_reset = wp.array(
            velocities_surface_default[group_indices_secondary_host],
            dtype=wp.vec3,
            device=self.model.device,
        )

        # Even-world cables return to their authored segment poses and velocities.
        group_indices_cable_host = np.arange(0, self.world_count, 2, dtype=np.int32)
        transforms_cable_default = self.curve.get_body_transforms(self.model).numpy()
        velocities_cable_default = self.curve.get_body_velocities(self.model).numpy()
        self.group_indices_cable = wp.array(
            group_indices_cable_host,
            dtype=wp.int32,
            device=self.model.device,
        )
        self.transforms_cable_reset = wp.array(
            transforms_cable_default[group_indices_cable_host],
            dtype=wp.transform,
            device=self.model.device,
        )
        self.velocities_cable_reset = wp.array(
            velocities_cable_default[group_indices_cable_host],
            dtype=wp.spatial_vector,
            device=self.model.device,
        )

        # Odd-world soft cubes restart half a meter above their authored positions.
        group_indices_volume_host = np.arange(1, self.world_count, 2, dtype=np.int32)
        positions_volume_default = self.volume.get_particle_positions(self.model).numpy()
        velocities_volume_default = self.volume.get_particle_velocities(self.model).numpy()
        positions_volume_reset = positions_volume_default[group_indices_volume_host].copy()
        positions_volume_reset[:, :, 2] += 0.5
        self.group_indices_volume = wp.array(
            group_indices_volume_host,
            dtype=wp.int32,
            device=self.model.device,
        )
        self.positions_volume_reset = wp.array(
            positions_volume_reset,
            dtype=wp.vec3,
            device=self.model.device,
        )
        self.velocities_volume_reset = wp.array(
            velocities_volume_default[group_indices_volume_host],
            dtype=wp.vec3,
            device=self.model.device,
        )

        # Warm the setters before capture and update both alternating simulation states.
        self._apply_resets()
        with wp.ScopedCapture(device=self.model.device) as capture:
            self._apply_resets()
        self.reset_graph = capture.graph
        wp.capture_launch(self.reset_graph)

        positions_primary_expected = positions_primary_default.copy()
        positions_primary_expected[group_indices_primary_host] = positions_primary_reset
        positions_surface_expected = positions_surface_default.copy()
        positions_surface_expected[2 * group_indices_primary_host] = positions_primary_reset
        positions_surface_expected[group_indices_secondary_host] = positions_secondary_reset
        positions_volume_expected = positions_volume_default.copy()
        positions_volume_expected[group_indices_volume_host] = positions_volume_reset

        # Comparing complete views also proves that indexed writes leave every
        # unselected group unchanged.
        self.initial_reset_valid = True
        for state in (self.state_0, self.state_1):
            self.initial_reset_valid &= (
                np.allclose(self.surface.get_particle_positions(state).numpy(), positions_primary_expected)
                and np.allclose(self.surface.get_particle_velocities(state).numpy(), velocities_primary_default)
                and np.allclose(self.surfaces.get_particle_positions(state).numpy(), positions_surface_expected)
                and np.allclose(self.surfaces.get_particle_velocities(state).numpy(), velocities_surface_default)
                and np.allclose(self.curve.get_body_transforms(state).numpy(), transforms_cable_default)
                and np.allclose(self.curve.get_body_velocities(state).numpy(), velocities_cable_default)
                and np.allclose(self.volume.get_particle_positions(state).numpy(), positions_volume_expected)
                and np.allclose(self.volume.get_particle_velocities(state).numpy(), velocities_volume_default)
            )

        self.viewer.set_model(self.model)
        self.viewer.picking_enabled = True
        self.viewer.set_world_offsets((0.0, 0.0, 0.0))
        if args.max_worlds is not None:
            self.viewer.set_visible_worlds(range(min(args.max_worlds, self.world_count)))
        self.viewer.set_camera(pos=wp.vec3(5.0, -6.0, 6.0), pitch=-35.0, yaw=140.0)

        with wp.ScopedCapture(device=self.model.device) as capture:
            self.simulate()
        self.sim_graph = capture.graph

    def _apply_resets(self) -> None:
        for state in (self.state_0, self.state_1):
            self.surface.set_particle_positions(
                state,
                self.positions_primary_reset,
                group_indices=self.group_indices_primary,
            )
            self.surface.set_particle_velocities(
                state,
                self.velocities_primary_reset,
                group_indices=self.group_indices_primary,
            )
            self.surfaces.set_particle_positions(
                state,
                self.positions_secondary_reset,
                group_indices=self.group_indices_secondary,
            )
            self.surfaces.set_particle_velocities(
                state,
                self.velocities_secondary_reset,
                group_indices=self.group_indices_secondary,
            )
            self.curve.set_body_transforms(
                state,
                self.transforms_cable_reset,
                group_indices=self.group_indices_cable,
            )
            self.curve.set_body_velocities(
                state,
                self.velocities_cable_reset,
                group_indices=self.group_indices_cable,
            )
            self.volume.set_particle_positions(
                state,
                self.positions_volume_reset,
                group_indices=self.group_indices_volume,
            )
            self.volume.set_particle_velocities(
                state,
                self.velocities_volume_reset,
                group_indices=self.group_indices_volume,
            )

    def simulate(self) -> None:
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self) -> None:
        if self.sim_time >= self.next_reset:
            wp.capture_launch(self.reset_graph)
            self.next_reset += 2.0

        wp.capture_launch(self.sim_graph)
        self.sim_time += self.frame_dt

    def render(self) -> None:
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def test_final(self) -> None:
        if not self.initial_reset_valid:
            raise ValueError("Indexed deformable resets did not update the selected rows")

        expected_world_starts = np.arange(0, 2 * self.world_count + 1, 2, dtype=np.int32)
        expected_world_ids = np.repeat(np.arange(self.world_count, dtype=np.int32), 2)
        np.testing.assert_array_equal(self.surfaces.world_starts.numpy(), expected_world_starts)
        np.testing.assert_array_equal(self.surfaces.world_ids.numpy(), expected_world_ids)

        expected_counts = {
            (self.curve, "body"): 3,
            (self.curve, "joint"): 2,
            (self.surface, "particle"): 4,
            (self.surface, "triangle"): 2,
            (self.surface, "edge"): 5,
            (self.volume, "particle"): 8,
            (self.volume, "tetrahedron"): 5,
        }
        for (view, kind), expected in expected_counts.items():
            if view.elements_per_group(kind) != expected:
                raise ValueError(f"Unexpected {view.family} {kind} count")

        if not np.all(np.isfinite(self.state_0.particle_q.numpy())):
            raise ValueError("Particle state contains NaN or infinity")
        if not np.all(np.isfinite(self.state_0.body_q.numpy())):
            raise ValueError("Body state contains NaN or infinity")

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        newton.examples.add_world_count_arg(parser)
        newton.examples.add_max_worlds_arg(parser)
        parser.set_defaults(world_count=4, max_worlds=4)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
