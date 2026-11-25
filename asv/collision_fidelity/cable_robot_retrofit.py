# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Full original C8 cable arrangements with an original Franka hand.

Original rod builders are reused verbatim. Only the old jaw boxes are omitted.
VBD integrates dynamic rods internally; the real hand uses prescribed FK.
"""

from __future__ import annotations

import numpy as np
import warp as wp

import newton

from .cable_scenes import _CableScene, _prescribe_bodies
from .public_vbd import create_solver, set_command_time
from .robot_gripper import FrankaHandDriver, add_franka_hand

SCENE_IDS = ("c9_pile_pickup", "c9_bundle_squeeze", "c9_bundle_squeeze_withdraw")


class CableRobotScene(_CableScene):
    def __init__(self, scene_id, *, device=None, substeps=10, iterations=24):
        if scene_id not in SCENE_IDS:
            raise ValueError(scene_id)
        if int(substeps) != substeps or not 2 <= substeps <= 10 or substeps % 2:
            raise ValueError("Use an even number of substeps in [2, 10]")
        self.scene_id, self.time = scene_id, 0.0
        self.is_pile = scene_id == "c9_pile_pickup"
        self.withdraw = scene_id == "c9_bundle_squeeze_withdraw"
        self.frame_dt, self.duration = 1.0 / 60.0, 9.0
        self.substeps, self.iterations = int(substeps), int(iterations)
        self.dt = self.sim_dt = self.frame_dt / self.substeps
        self.radius, self._grasp_x = 0.009, 0.10
        self.cable_bodies, self.cables, self.jaw_bodies = [], [], []
        self.fixture_ids, self.fixture_notes = [], []
        self._moving_ids, self._offsets = [], []
        self._times = np.asarray([0.0, 1.5, 2.5, 4.5, 6.5, 7.5, 9.0], dtype=np.float32)
        self.omitted_original_jaws, self.history = [], []
        builder = newton.ModelBuilder()
        builder.rigid_gap = 0.002
        builder.default_shape_cfg.ke, builder.default_shape_cfg.kd = 1.0e5, 0.0
        builder.default_shape_cfg.mu = 1.5
        builder.default_shape_cfg.margin, builder.default_shape_cfg.gap = 0.0, 0.002
        self._cfg = newton.ModelBuilder.ShapeConfig(density=500.0, ke=1.0e5, kd=0.0, mu=1.0, margin=0.0, gap=0.002)
        ground_cfg = self._cfg.copy()
        ground_cfg.mu = 0.5
        builder.add_ground_plane(cfg=ground_cfg)
        if self.is_pile:
            self._build_pile(builder)
        else:
            self._build_bundle(builder)
        self.original_asset_ids = list(self.asset_ids)
        self.asset_ids = [a for a in self.asset_ids if not a.startswith("G0_")] + ["cached_franka_fr3_hand"]
        self.title = (
            "C9 | Franka: original loose cable pile pickup"
            if self.is_pile
            else "C9 | Franka: original bundle squeeze and strand pull"
        )
        self.description = (
            "Full original C8 loose pile: 720 mm selected strand, long curved strand, open coil, and three-arm branch. "
            "Original Franka hand approaches the settled selected strand, closes real fingers, lifts 450 mm, holds 2 s, "
            "opens and releases. All cable bodies remain dynamic; no cable attachments."
            if self.is_pile
            else "Full original C8 bundle: three 720 mm cables of radius 9 mm. Original Franka fingers squeeze a local "
            "17.5 mm axial patch; one original declared endpoint fixture pulls the orange strand 160 mm, then fingers open. "
            "No cable-to-hand attachments. The single endpoint boundary condition is retained explicitly."
        )
        if self.withdraw:
            self.title = "C9 | Franka: bundle squeeze, strand pull and withdrawal"
            self.description += " The open hand withdraws 150 mm after 7.5 s to separate fully from strands that spread beyond its 80 mm aperture."
        self.fixture_notes = (
            ["No cable endpoints are pinned. Original hand root and finger joints follow prescribed FK."]
            if self.is_pile
            else [
                "Orange last capsule remains the original explicit mass-zero endpoint pulling fixture.",
                "Original hand root and finger joints use prescribed FK. Cable-to-hand coupling is contact only.",
            ]
        )
        self.pinch = np.asarray((self._grasp_x, 0.0, 0.15) if self.is_pile else (0.0, 0.0, 0.013))
        self.closed_opening = 0.0176 if self.is_pile else 0.050
        self.hand = add_franka_hand(builder, pinch_pos=self.pinch, rotation=(1.0, 0.0, 0.0, 0.0), opening=0.08)
        builder.color()
        self.model = builder.finalize(device=device)
        self.state_0, self.state_1 = self.model.state(), self.model.state()
        self.control = self.model.control()
        self._clock = wp.zeros(1, dtype=float, device=self.model.device)
        if self.is_pile:
            times = [0.0, 0.5, 1.5, 2.5, 4.5, 6.5, 7.5, 9.0]
            positions = [(self._grasp_x, 0.0, z) for z in (0.15, 0.15, 0.013, 0.013, 0.463, 0.463, 0.463, 0.463)]
            openings = [0.08] * 3 + [self.closed_opening] * 3 + [0.08] * 2
        else:
            times = self._times.tolist()
            positions = [self.pinch] * len(times)
            openings = [0.08, 0.08, self.closed_opening, self.closed_opening, self.closed_opening, 0.08, 0.08]
            if self.withdraw:
                times = [*self._times[:-1].tolist(), 8.5, 9.0]
                positions = [self.pinch] * 6 + [self.pinch + np.asarray((0.0, 0.0, 0.15))] * 2
                openings = [*openings[:-1], 0.08, 0.08]
        self.driver = FrankaHandDriver(
            self.model, self.hand, times=times, positions=positions, rotations=(1.0, 0.0, 0.0, 0.0), openings=openings
        )
        self.driver.update(self._clock, self.state_0, self.state_1)
        self.solver, self.pipeline = create_solver(
            self.model,
            iterations=self.iterations,
            rigid_body_contact_buffer_size=128,
            rigid_contact_history=True,
            contact_matching="sticky",
            full_surface_contact=False,
        )
        self.contacts = self.solver.contacts
        self._times_wp = wp.array(self._times, dtype=float, device=self.model.device)
        self._moving_ids_wp = wp.array(self._moving_ids, dtype=int, device=self.model.device)
        poses = self.state.body_q.numpy()
        self._base_wp = wp.array(poses[self._moving_ids], dtype=wp.transform, device=self.model.device)
        self._offsets_wp = wp.array(
            np.asarray(self._offsets, dtype=np.float32).reshape((-1, len(self._times), 3)),
            dtype=wp.vec3,
            device=self.model.device,
        )
        self.shape_bodies = self.model.shape_body.numpy()
        scales = self.model.shape_scale.numpy()
        self.half_lengths = np.asarray(
            [scales[np.flatnonzero(self.shape_bodies == body)[0], 1] for body in self.cable_bodies]
        )
        self.initial_cable_q = poses[self.cable_bodies].copy()
        self.initial_geometry = self._geometry_audit()
        if self.initial_geometry["minimum_shell_to_robot_distance_m"] < -1e-6:
            raise ValueError("Initial original cable capsules overlap the real open hand")
        if self.initial_geometry["robot_clearance_above_table_plane_m"] < 0.003:
            raise ValueError("Original robot solids intersect or nearly touch the ground")

    def _jaw(self, builder, position, half_extents, offsets):
        # The inherited geometry builders call this hook. Record, but never add, old boxes.
        self.omitted_original_jaws.append({"position_m": list(position), "half_extents_m": list(half_extents)})
        return -1

    @property
    def phase(self):
        if self.is_pile:
            for end, phase in (
                (0.5, "pile settle"),
                (1.5, "open hand approach"),
                (2.5, "close original fingers"),
                (4.5, "lift"),
                (6.5, "hold"),
                (7.5, "release"),
                (9.1, "fall and settle"),
            ):
                if self.time < end:
                    return phase
        else:
            for end, phase in (
                (1.5, "bundle settle"),
                (2.5, "squeeze"),
                (4.5, "squeeze hold"),
                (6.5, "strand pull"),
                (7.5, "open fingers"),
                (8.5, "open-hand withdrawal" if self.withdraw else "recovery"),
                (9.1, "recovery"),
            ):
                if self.time < end:
                    return phase
        return "complete"

    def _centerline(self):
        q = self.state.body_q.numpy()[self.cable_bodies]
        x, y, z, w = (q[:, i] for i in range(3, 7))
        axis = np.column_stack((2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y)))
        shift = axis * self.half_lengths[:, None]
        return q[:, :3], q[:, :3] - shift, q[:, :3] + shift

    def _geometry_audit(self):
        center, start, end = self._centerline()
        points = np.concatenate([start, (start + center) / 2, center, (center + end) / 2, end])
        return self.driver.geometry_audit(
            self.state, points=points, radii=self.radius, table_top=0.0, full_surface_contact=False
        )

    def step_substep(self, command_time_s):
        """Advance one physical substep with prescribed endpoint commands."""
        set_command_time(self._clock, command_time_s + self.sim_dt)
        self.state_0.clear_forces()
        self.state_1.clear_forces()
        if self._moving_ids:
            wp.launch(
                _prescribe_bodies,
                dim=len(self._moving_ids),
                inputs=[
                    self._clock,
                    self._moving_ids_wp,
                    self._base_wp,
                    self._times_wp,
                    self._offsets_wp,
                    self.state_0.body_q,
                    self.state_1.body_q,
                    self.state_0.body_qd,
                    self.state_1.body_qd,
                ],
                device=self.model.device,
            )
        self.driver.update(self._clock, self.state_0)
        self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
        self.state_0, self.state_1 = self.state_1, self.state_0
        self.time = float(command_time_s + self.sim_dt)
        return self.state


def build_scene(scene_id, **kwargs):
    return CableRobotScene(scene_id, **kwargs)
