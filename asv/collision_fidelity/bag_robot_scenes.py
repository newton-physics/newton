# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Original M1 bag/payload meshes with two original Franka hands.

The bags start in a prepared contact hold. The handle patches move to opposite
shoulders so the two full robot hands have space. All material stays dynamic.
"""

from __future__ import annotations

import numpy as np
import warp as wp

import newton

from .assets import add_cloth_asset, add_solid_asset, cloth_mesh
from .public_vbd import SceneInputs, create_solver, set_command_time
from .robot_gripper import FrankaHandDriver, add_franka_hand

SCENE_IDS = ("m3_franka_tote_empty", "m3_franka_tote_2", "m3_franka_tote_2_spheres", "m3_franka_gusset_4")


@wp.kernel
def _tick(clock: wp.array[float], dt: float):
    clock[0] += dt


class BagRobotScene(SceneInputs):
    def __init__(
        self,
        scene_id,
        *,
        device=None,
        substeps=10,
        iterations=40,
        closed_opening=0.004,
        grip_x=0.04,
        grip_z_offset=0.0,
    ):
        if scene_id not in SCENE_IDS:
            raise ValueError(scene_id)
        if int(substeps) != substeps or not 2 <= substeps <= 10 or substeps % 2:
            raise ValueError("Use an even number of substeps in [2, 10]")
        self.scene_id, self.time = scene_id, 0.0
        self.frame_dt, self.duration = 1.0 / 30.0, 9.0
        self.substeps, self.iterations = int(substeps), int(iterations)
        self.dt = self.frame_dt / self.substeps
        self.table_top, self.particle_radius = 0.18, 0.003
        self.bag_name = "gusset_bag" if "gusset" in scene_id else "tote"
        self.payload_count = 4 if scene_id.endswith("_4") else 2 if "_2" in scene_id else 0
        self.payload_name = "sphere" if scene_id.endswith("_spheres") else "block"
        self.title = f"M3 | Franka hands: {self.bag_name.replace('_', ' ')} | {self.payload_count} payloads"
        self.description = (
            "Original coarse M1 bag and payload geometry, mass, and material. Two original Franka hands "
            "start in a prepared contact hold on opposite handle shoulders; this is not grasp acquisition. "
            "Original 300 mm lift, 220/160 mm carry, hold, place, open, and withdrawal. Every cloth and "
            "payload vertex remains free. The original wide cube jaws are replaced by real 17.5 mm fingertip faces."
        )
        self.asset_ids = [self.bag_name, "cached_franka_fr3_hand"] + [self.payload_name] * self.payload_count
        self.camera_pos, self.camera_target = (1.35, -1.75, 1.52), (0.08, 0.04, 0.57)
        self.history = []
        builder = newton.ModelBuilder()
        builder.default_shape_cfg.ke, builder.default_shape_cfg.kd = 5.0e4, 0.1
        builder.default_shape_cfg.mu = 1.5
        builder.default_shape_cfg.margin, builder.default_shape_cfg.gap = 0.0, 0.002
        builder.default_shape_cfg.configure_sdf(force_sdf=True, max_resolution=64)
        bottom = self.table_top + 0.005
        self.bag = add_cloth_asset(
            builder,
            self.bag_name,
            resolution="coarse",
            pos=(0.0, 0.0, bottom),
            density=0.22,
            particle_radius=self.particle_radius,
            tri_ke=3000.0,
            tri_ka=3000.0,
            tri_kd=0.025,
            edge_ke=0.025,
            edge_kd=0.0005,
        )
        self.payloads = []
        positions = (
            [(-0.06, 0.0), (0.06, 0.0)]
            if self.payload_count == 2
            else [(-0.05, -0.026), (0.05, -0.026), (-0.05, 0.026), (0.05, 0.026)]
        )
        colors = [(0.93, 0.33, 0.19), (0.20, 0.55, 0.85), (0.36, 0.72, 0.40), (0.66, 0.39, 0.83)]
        for i in range(self.payload_count):
            self.payloads.append(
                add_solid_asset(
                    builder,
                    self.payload_name,
                    resolution="coarse",
                    pos=(*positions[i], bottom + 0.012),
                    density=180.0,
                    particle_radius=0.0025,
                    k_mu=8000.0,
                    k_lambda=8000.0,
                    k_damp=0.5,
                    color=colors[i],
                )
            )
        table_cfg = newton.ModelBuilder.ShapeConfig(density=0.0, ke=5.0e4, kd=0.1, mu=0.6)
        builder.add_shape_box(
            body=-1,
            xform=wp.transform(wp.vec3(0.12, 0.07, self.table_top - 0.035), wp.quat_identity()),
            hx=0.57,
            hy=0.44,
            hz=0.035,
            cfg=table_cfg,
            color=(0.43, 0.48, 0.54),
            label="original_fixed_support_and_placement_table",
        )
        builder.add_ground_plane()
        mesh = cloth_mesh(self.bag_name, resolution="coarse")
        width, depth, height = mesh.metadata["bag_body_dimensions_m"]
        # Offset shoulder patches give the complete palms room; the bag mesh is unchanged.
        self.grip_x = float(grip_x)
        self.grip_z_offset = float(grip_z_offset)
        grip_z = bottom + height + 0.14 * np.sqrt(1.0 - (self.grip_x / (width / 4.0)) ** 2) + self.grip_z_offset
        self.grip_centers = [(-self.grip_x, -depth / 2, grip_z), (self.grip_x, depth / 2, grip_z)]
        rotation = (1.0, 0.0, 0.0, 0.0)
        self.closed_opening = float(closed_opening)
        self.hands = [
            add_franka_hand(
                builder, pinch_pos=center, rotation=rotation, opening=self.closed_opening, label=f"bag_franka_{i}"
            )
            for i, center in enumerate(self.grip_centers)
        ]
        builder.color()
        self.model = builder.finalize(device=device)
        self.model.soft_contact_ke, self.model.soft_contact_kd, self.model.soft_contact_mu = 5.0e4, 0.1, 1.5
        self.state_0, self.state_1 = self.model.state(), self.model.state()
        self.control = self.model.control()
        self._clock = wp.zeros(1, dtype=float, device=self.model.device)
        times = [0.0, 0.5, 2.2, 2.5, 4.0, 5.5, 6.8, 7.4, 8.5, 9.0]
        offsets = np.asarray(
            [
                (0, 0, 0),
                (0, 0, 0),
                (0, 0, 0.30),
                (0, 0, 0.30),
                (0.22, 0.16, 0.30),
                (0.22, 0.16, 0.30),
                (0.22, 0.16, 0.02),
                (0.22, 0.16, 0.02),
                (0.22, 0.16, 0.20),
                (0.22, 0.16, 0.20),
            ]
        )
        self.drivers = [
            FrankaHandDriver(
                self.model,
                hand,
                times=times,
                positions=np.asarray(center) + offsets,
                rotations=rotation,
                openings=[self.closed_opening] * 7 + [0.08] * 3,
            )
            for hand, center in zip(self.hands, self.grip_centers, strict=True)
        ]
        for driver in self.drivers:
            driver.update(self._clock, self.state_0, self.state_1)
        self.initial_q = self.state.particle_q.numpy()
        self.radii = self.model.particle_radius.numpy()
        self.initial_geometry = [
            d.geometry_audit(
                self.state, points=self.initial_q, radii=self.radii, table_top=self.table_top, full_surface_contact=True
            )
            for d in self.drivers
        ]
        if any(a["vertices_inside_robot_solids"] for a in self.initial_geometry):
            raise ValueError("Initial bag or payload vertex intersects an original Franka solid")
        self.initial_hand_to_hand = self._hand_separation()
        if min(self.initial_hand_to_hand) < -1.0e-5:
            raise ValueError(f"Initial original robot solids overlap: {self.initial_hand_to_hand}")
        self.solver, self.pipeline = create_solver(
            self.model,
            iterations=self.iterations,
            particle_enable_self_contact=True,
            particle_self_contact_margin=0.005,
            particle_self_contact_gap=0.005,
            particle_vertex_contact_buffer_size=128,
            particle_edge_contact_buffer_size=128,
            friction_epsilon=0.001,
            soft_contact_gap=0.01,
            full_surface_contact=True,
        )
        self.contacts = self.solver.contacts
        tris = self.model.tri_indices.numpy()
        tris = tris[np.all(tris < self.bag.particle_end, axis=1)]
        self.edges = np.unique(
            np.sort(np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]]), axis=1), axis=0
        )
        self.rest_lengths = np.linalg.norm(self.initial_q[self.edges[:, 1]] - self.initial_q[self.edges[:, 0]], axis=1)

    def _hand_separation(self):
        poses, transforms, bodies = (
            self.state.body_q.numpy(),
            self.model.shape_transform.numpy(),
            self.model.shape_body.numpy(),
        )
        minima = []
        for source, target in ((0, 1), (1, 0)):
            samples = []
            for shape, vertices in self.hands[source].collision_vertices.items():
                transform = wp.transform(*poses[bodies[shape]]) * wp.transform(*transforms[shape])
                rotation = np.asarray(wp.quat_to_matrix(wp.transform_get_rotation(transform))).reshape(3, 3)
                samples.append(vertices @ rotation.T + np.asarray(wp.transform_get_translation(transform)))
            audit = self.drivers[target].geometry_audit(self.state, points=np.concatenate(samples), radii=0.0)
            minima.append(audit["minimum_vertex_to_robot_distance_m"])
        return minima

    @property
    def state(self):
        return self.state_0

    @property
    def phase(self):
        for end, name in (
            (0.5, "prepared handle hold"),
            (2.2, "lift"),
            (2.5, "raised pause"),
            (4.0, "carry"),
            (5.5, "hold"),
            (6.8, "place"),
            (7.4, "open fingers"),
            (8.5, "withdraw"),
            (9.1, "released observation"),
        ):
            if self.time < end:
                return name
        return "complete"

    def step_substep(self, command_time_s):
        """Advance one physical substep with prescribed endpoint commands."""
        set_command_time(self._clock, command_time_s + self.dt)
        self.state_0.clear_forces()
        self.state_1.clear_forces()
        for driver in self.drivers:
            driver.update(self._clock, self.state_0)
        self.solver.step(self.state_0, self.state_1, self.control, None, self.dt)
        self.state_0, self.state_1 = self.state_1, self.state_0
        self.time = float(command_time_s + self.dt)
        return self.state


def build_scene(scene_id, **kwargs):
    return BagRobotScene(scene_id, **kwargs)
