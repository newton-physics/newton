# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Contact-only cable pickup with original robot hands and VBD rods.

The hand starts open at a cleared grasp pose. VBD integrates every cable
capsule and rod joint. Only the original robot bodies are kinematic.
"""

from __future__ import annotations

import numpy as np
import warp as wp

import newton

from .allegro_gripper import OPEN_FINGERS, AllegroHandDriver, add_allegro_hand
from .public_vbd import SceneInputs, create_solver, set_command_time
from .robot_gripper import FrankaHandDriver, add_franka_hand

SCENE_IDS = ("m2_franka_cable", "m2_allegro_cable")
__all__ = ["SCENE_IDS", "build_scene"]


@wp.kernel
def _advance_clock(clock: wp.array[float], dt: float):
    clock[0] += dt


class _RobotCableScene(SceneInputs):
    def __init__(self, scene_id, *, device=None, substeps=10, iterations=24):
        if scene_id not in SCENE_IDS:
            raise ValueError(f"Unknown robot cable scene {scene_id!r}")
        if substeps < 2 or substeps > 10 or substeps % 2:
            raise ValueError("Use an even substep count from 2 through 10")
        self.scene_id = scene_id
        self.is_allegro = scene_id == "m2_allegro_cable"
        self.time = 0.0
        self.frame_dt = 1.0 / 30.0
        self.duration = 11.0
        self.substeps, self.iterations = int(substeps), int(iterations)
        self.dt = self.frame_dt / self.substeps
        self.table_top = 0.18
        self.radius = 0.018 if self.is_allegro else 0.006
        self.length = 0.60
        self.segment_count = 15 if self.is_allegro else 25
        self.closed_opening = None if self.is_allegro else 0.0116
        self.closed_fingers = None
        robot_name = "Allegro" if self.is_allegro else "Franka"
        self.title = f"M2 | {robot_name} hand: cable pickup, carry, hold and release"
        self.description = (
            f"The original {robot_name} hand starts open around a supported 600 mm cable, closes its actual fingers, "
            "lifts 380 mm, carries 120 mm in X and 80 mm in Y, holds, places, opens and withdraws. "
            "Every cable capsule is dynamic; VBD solves its rod joints and collision contacts. "
            "No cable attachments or endpoint fixtures. The approach to the initial open pose is not shown. "
            "Robot FK is prescribed; this is not a test of dynamic actuator capacity."
        )
        self.asset_ids = (
            ["C0_straight_flexible_18mm_radius", "cached_allegro_left_hand"]
            if self.is_allegro
            else ["C0_straight_flexible_6mm_radius", "cached_franka_fr3_hand"]
        )
        self.camera_pos = (0.86, -1.1, 0.91)
        self.camera_target = (0.045, 0.025, 0.38)
        self.camera_fov = 42.0
        self.history = []

        builder = newton.ModelBuilder()
        builder.rigid_gap = 0.002
        builder.default_shape_cfg.ke = 1.0e5
        builder.default_shape_cfg.kd = 0.0
        builder.default_shape_cfg.mu = 1.5
        builder.default_shape_cfg.margin = 0.0
        builder.default_shape_cfg.gap = 0.002
        cable_cfg = newton.ModelBuilder.ShapeConfig(density=500.0, ke=1.0e5, kd=0.0, mu=1.0, margin=0.0, gap=0.002)
        points = np.zeros((self.segment_count + 1, 3))
        points[:, 1 if self.is_allegro else 0] = np.linspace(-self.length / 2, self.length / 2, self.segment_count + 1)
        points[:, 2] = self.table_top + self.radius + 0.0005
        self.cable_bodies, self.cable_joints = builder.add_rod(
            rod=newton.Rod([wp.vec3(*p) for p in points], radius=self.radius),
            cfg=cable_cfg,
            stretch_stiffness=5.0e4,
            shear_stiffness=5.0e4,
            stretch_damping=0.1,
            shear_damping=0.1,
            bend_stiffness=0.008,
            twist_stiffness=0.008,
            bend_damping=0.0005,
            twist_damping=0.0005,
            label="dynamic_pickup_cable",
            color=(0.97, 0.42, 0.11),
            body_frame_origin="com",
        )
        self.cable_bodies = np.asarray(self.cable_bodies, dtype=int)
        self.cable_joints = np.asarray(self.cable_joints, dtype=int)
        table_cfg = newton.ModelBuilder.ShapeConfig(density=0.0, ke=1.0e5, kd=0.0, mu=0.5, margin=0.0, gap=0.002)
        builder.add_shape_box(
            body=-1,
            xform=wp.transform((0.06, 0.04, self.table_top - 0.03), wp.quat_identity()),
            hx=0.48,
            hy=0.40 if self.is_allegro else 0.32,
            hz=0.03,
            cfg=table_cfg,
            color=(0.43, 0.48, 0.54),
            label="fixed_support_and_placement_table",
        )
        builder.add_ground_plane(cfg=table_cfg)
        if self.is_allegro:
            self.pinch = np.asarray((-0.015, 0.0, self.table_top + self.radius + 0.0005 + 0.105))
            rotation = (0.0, np.sqrt(0.5), 0.0, np.sqrt(0.5))
            self.closed_fingers = [
                0.07146795171612429,
                0.9387183245389547,
                0.40672494156610906,
                0.655084330229117,
                0.0,
                0.9401318961853883,
                0.42456727085955864,
                0.676473609827385,
                -0.07146795171613404,
                0.9387183182296001,
                0.4067249605927987,
                0.6550843092261311,
                1.1005490672985372,
                0.7349576686875242,
                0.9922936819026352,
                0.0000006689687416481444,
            ]
            self.hand = add_allegro_hand(builder, root_pos=self.pinch, rotation=rotation, finger_positions=OPEN_FINGERS)
        else:
            self.pinch = np.asarray((0.0, 0.0, self.table_top + 0.013))
            rotation = (1.0, 0.0, 0.0, 0.0)
            self.hand = add_franka_hand(builder, pinch_pos=self.pinch, rotation=rotation, opening=0.04)
        builder.color()
        self.model = builder.finalize(device=device)
        self.state_0, self.state_1 = self.model.state(), self.model.state()
        self.control = self.model.control()
        self.clock = wp.zeros(1, dtype=float, device=self.model.device)
        self.times = [0.0, 0.4, 1.2, 1.6, 3.6, 4.8, 6.8, 8.4, 9.0, 10.0, 11.0]
        offsets = np.asarray(
            [(0, 0, 0)] * 4
            + [(0, 0, 0.38), (0.12, 0.08, 0.38), (0.12, 0.08, 0.38)]
            + [(0.12, 0.08, 0.08)] * 2
            + [(0.12, 0.08, 0.24)] * 2
        )
        if self.is_allegro:
            self.driver = AllegroHandDriver(
                self.model,
                self.hand,
                times=self.times,
                positions=self.pinch + offsets,
                rotations=rotation,
                finger_positions=[OPEN_FINGERS] * 2 + [self.closed_fingers] * 6 + [OPEN_FINGERS] * 3,
            )
        else:
            self.driver = FrankaHandDriver(
                self.model,
                self.hand,
                times=self.times,
                positions=self.pinch + offsets,
                rotations=rotation,
                openings=[0.04, 0.04] + [self.closed_opening] * 6 + [0.06] * 3,
            )
        self.driver.update(self.clock, self.state_0, self.state_1)
        self.solver, self.pipeline = create_solver(
            self.model,
            iterations=self.iterations,
            rigid_body_contact_buffer_size=128,
            rigid_contact_history=True,
            contact_matching="sticky",
            full_surface_contact=False,
        )
        self.contacts = self.solver.contacts
        self.shape_bodies = self.model.shape_body.numpy()
        self.half_lengths = np.full(self.segment_count, self.length / self.segment_count / 2)
        self.initial_q = self.state.body_q.numpy()[self.cable_bodies].copy()
        self.initial_geometry = self._geometry_audit()
        if self.initial_geometry["minimum_shell_to_robot_distance_m"] < -1e-6:
            raise ValueError("Cable overlaps the original open hand at the initial pose")
        if self.initial_geometry["robot_clearance_above_table_plane_m"] < 0.003:
            raise ValueError("Original hand is too close to the table")

    @property
    def state(self):
        return self.state_0

    @property
    def phase(self):
        for end, name in (
            (0.4, "supported cable; open hand"),
            (1.2, "close original fingers"),
            (1.6, "closed contact grasp"),
            (3.6, "lift"),
            (4.8, "carry"),
            (6.8, "hold"),
            (8.4, "place"),
            (9.0, "open fingers"),
            (10.0, "withdraw"),
            (11.1, "released observation"),
        ):
            if self.time < end:
                return name
        return "complete"

    def _centerline(self):
        q = self.state.body_q.numpy()[self.cable_bodies]
        x, y, z, w = (q[:, i] for i in range(3, 7))
        axis = np.column_stack((2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y)))
        displacement = axis * self.half_lengths[:, None]
        return q[:, :3], q[:, :3] - displacement, q[:, :3] + displacement

    def _geometry_audit(self):
        center, start, end = self._centerline()
        # Samples along each capsule axis plus its radius. This bounds the
        # capsule-to-plane extent exactly; robot clearance is a sampled audit.
        samples = np.concatenate([start, (start + center) / 2, center, (center + end) / 2, end])
        if self.is_allegro:
            result = self.driver.geometry_audit(self.state, points=samples, radii=self.radius, ground_z=self.table_top)
            result["robot_clearance_above_table_plane_m"] = result["hand_ground_clearance_including_margin_m"]
            result["minimum_shell_to_robot_distance_m"] = result["minimum_shell_to_hand_distance_m"]
            result["physical_tip_aperture_m"] = None
            return result
        return self.driver.geometry_audit(
            self.state, points=samples, radii=self.radius, table_top=self.table_top, full_surface_contact=False
        )

    def step_substep(self, command_time_s):
        """Advance one physical substep with prescribed endpoint commands."""
        set_command_time(self.clock, command_time_s + self.dt)
        self.state_0.clear_forces()
        self.state_1.clear_forces()
        self.driver.update(self.clock, self.state_0)
        self.solver.step(self.state_0, self.state_1, self.control, None, self.dt)
        self.state_0, self.state_1 = self.state_1, self.state_0
        self.time = float(command_time_s + self.dt)
        return self.state


def build_scene(scene_id, **kwargs):
    return _RobotCableScene(scene_id, **kwargs)
