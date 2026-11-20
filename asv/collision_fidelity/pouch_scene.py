# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Original pouch held at its front lip by the cached Franka hand's real fingers.

This is a prepared contact hold, not grasp acquisition. No cloth vertices are
pinned. The 2 mm effective shell thickness fits the rest mesh's narrow sides.
"""

from __future__ import annotations

import numpy as np
import warp as wp

import newton

from .assets import AddedAsset
from .pouch_mesh import build_pouch_mesh
from .public_vbd import SceneInputs, create_solver, set_command_time
from .robot_gripper import FrankaHandDriver, add_franka_hand

SCENE_IDS = ("m2_franka_pouch",)
__all__ = ["SCENE_IDS", "build_scene"]


@wp.kernel
def _tick(clock: wp.array[float], dt: float):
    clock[0] += dt


class _PouchScene(SceneInputs):
    def __init__(self, *, device=None, substeps=10, iterations=40, self_contact_capacity=128):
        self.scene_id = "m2_franka_pouch"
        self.time = 0.0
        self.frame_dt = 1.0 / 30.0
        self.duration = 10.0
        self.substeps = int(substeps)
        self.iterations = int(iterations)
        if self.substeps < 2 or self.substeps % 2:
            raise ValueError("Pouch substeps must be a positive even integer")
        self.dt = self.frame_dt / self.substeps
        self.table_top = 0.18
        self.particle_radius = 0.001
        self.self_contact_margin = 0.002
        self.self_contact_gap = 0.0015
        self.title = "M2 | Franka hand and pouch: prepared rim hold, carry, release"
        self.description = (
            "Original Franka hand collision shapes and finger joints pinch the pouch front lip. "
            "Prepared contact hold; no approach/acquisition claim. The supported pouch is 150 by 200 by 18 mm, "
            "with a 2 mm effective contact thickness and refined front panels. Prescribed robot FK lifts 220 mm, "
            "carries 160 mm in X and 100 mm in Y, holds, places, opens, and withdraws. "
            "VBD computes all pouch motion; every cloth vertex remains free and self-contact stays enabled."
        )
        self.asset_ids = ["pouch_grip_refined_2mm_shell", "cached_franka_fr3_hand"]
        self.camera_pos = (0.72, -0.94, 0.90)
        self.camera_target = (0.055, 0.04, 0.43)
        self.camera_fov = 42.0
        self.history = []
        self.worst_hold_shape = None

        builder = newton.ModelBuilder()
        builder.default_shape_cfg.ke = 5.0e4
        builder.default_shape_cfg.kd = 0.1
        builder.default_shape_cfg.mu = 1.5
        builder.default_shape_cfg.margin = 0.0
        builder.default_shape_cfg.gap = 0.002
        builder.default_shape_cfg.configure_sdf(force_sdf=True, max_resolution=64)
        mesh = build_pouch_mesh()
        particle_start = builder.particle_count
        builder.add_cloth_mesh(
            vertices=mesh.vertices.tolist(),
            indices=mesh.triangles.ravel().tolist(),
            pos=(0.0, 0.0, self.table_top + self.particle_radius + 0.001),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=(0.0, 0.0, 0.0),
            color=mesh.metadata["color_rgb"],
            density=0.22,
            particle_radius=self.particle_radius,
            tri_ke=3000.0,
            tri_ka=3000.0,
            tri_kd=0.025,
            edge_ke=0.025,
            edge_kd=0.0005,
        )
        self.pouch = AddedAsset(mesh, particle_start, builder.particle_count, mesh.landmarks)
        cfg = newton.ModelBuilder.ShapeConfig(density=0.0, ke=5.0e4, kd=0.1, mu=0.6)
        builder.add_shape_box(
            body=-1,
            xform=wp.transform(wp.vec3(0.08, 0.05, self.table_top - 0.03), wp.quat_identity()),
            hx=0.40,
            hy=0.32,
            hz=0.03,
            cfg=cfg,
            color=(0.43, 0.48, 0.54),
            label="fixed_support_and_placement_table",
        )
        builder.add_ground_plane()
        pinch = np.array([0.0, -0.009, self.table_top + self.particle_radius + 0.001 + 0.2 - 0.009])
        rotation = (1.0, 0.0, 0.0, 0.0)
        self.closed_opening = 0.0015
        self.hand = add_franka_hand(
            builder, pinch_pos=pinch, rotation=rotation, opening=self.closed_opening, label="pouch_franka_hand"
        )
        builder.color()
        self.model = builder.finalize(device=device)
        self.model.soft_contact_ke = 5.0e4
        self.model.soft_contact_kd = 0.1
        self.model.soft_contact_mu = 1.5
        self.state_0, self.state_1 = self.model.state(), self.model.state()
        self.control = self.model.control()
        self._clock = wp.zeros(1, dtype=float, device=self.model.device)
        times = [0.0, 0.5, 2.5, 3.0, 4.5, 6.0, 7.5, 8.1, 9.0, 10.0]
        offsets = [
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.22),
            (0.0, 0.0, 0.22),
            (0.16, 0.10, 0.22),
            (0.16, 0.10, 0.22),
            (0.16, 0.10, 0.01),
            (0.16, 0.10, 0.01),
            (0.16, 0.10, 0.16),
            (0.16, 0.10, 0.16),
        ]
        self.driver = FrankaHandDriver(
            self.model,
            self.hand,
            times=times,
            positions=pinch + np.asarray(offsets),
            rotations=rotation,
            openings=[self.closed_opening] * 7 + [0.06] * 3,
        )
        self.driver.update(self._clock, self.state_0, self.state_1)
        self.initial_q = self.state.particle_q.numpy()
        self.initial_geometry = self.driver.geometry_audit(
            self.state,
            points=self.initial_q,
            radii=self.particle_radius,
            table_top=self.table_top,
            full_surface_contact=True,
        )
        if self.initial_geometry["vertices_inside_robot_solids"]:
            raise ValueError("Initial pouch vertices intersect the original Franka collision geometry")
        self.solver, self.pipeline = create_solver(
            self.model,
            iterations=self.iterations,
            particle_enable_self_contact=True,
            particle_self_contact_margin=self.self_contact_margin,
            particle_self_contact_gap=self.self_contact_gap,
            particle_vertex_contact_buffer_size=self_contact_capacity,
            particle_edge_contact_buffer_size=self_contact_capacity,
            friction_epsilon=0.001,
            soft_contact_gap=0.003,
            full_surface_contact=True,
        )
        self.contacts = self.solver.contacts
        self.triangles = self.model.tri_indices.numpy()
        self.edges = np.unique(
            np.sort(
                np.concatenate([self.triangles[:, [0, 1]], self.triangles[:, [1, 2]], self.triangles[:, [2, 0]]]),
                axis=1,
            ),
            axis=0,
        )
        self.rest_lengths = np.linalg.norm(self.initial_q[self.edges[:, 1]] - self.initial_q[self.edges[:, 0]], axis=1)

    @property
    def state(self):
        return self.state_0

    @property
    def phase(self):
        for end, name in (
            (0.5, "prepared front-lip hold"),
            (2.5, "lift"),
            (3.0, "raised pause"),
            (4.5, "carry"),
            (6.0, "hold"),
            (7.5, "place"),
            (8.1, "open fingers"),
            (9.0, "withdraw"),
            (10.1, "released observation"),
        ):
            if self.time < end:
                return name
        return "complete"

    def step_substep(self, command_time_s):
        """Advance one physical substep with prescribed endpoint commands."""
        set_command_time(self._clock, command_time_s + self.dt)
        self.state_0.clear_forces()
        self.state_1.clear_forces()
        self.driver.update(self._clock, self.state_0)
        self.solver.step(self.state_0, self.state_1, self.control, None, self.dt)
        self.state_0, self.state_1 = self.state_1, self.state_0
        self.time = float(command_time_s + self.dt)
        return self.state


def build_scene(scene_id, **kwargs):
    """Build the corrected pouch scene in SI units."""
    if scene_id not in SCENE_IDS:
        raise ValueError(f"Unknown pouch scene {scene_id!r}")
    return _PouchScene(**kwargs)
