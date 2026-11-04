# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The authored 52-card cloth deck with fixed supports and no moving ball."""

from __future__ import annotations

import numpy as np
import warp as wp

import newton

from .public_vbd import SceneInputs, create_solver

__all__ = ["PokerCardsScene", "build_scene"]

_SCENE_ID = "s4_poker_cards_no_ball"
_CARD_COLORS = (
    (68 / 255, 119 / 255, 170 / 255),
    (102 / 255, 204 / 255, 238 / 255),
    (34 / 255, 136 / 255, 51 / 255),
    (204 / 255, 187 / 255, 68 / 255),
    (238 / 255, 102 / 255, 119 / 255),
    (170 / 255, 51 / 255, 119 / 255),
    (238 / 255, 153 / 255, 51 / 255),
    (0 / 255, 153 / 255, 136 / 255),
)


class PokerCardsScene(SceneInputs):
    """Let all cards settle freely, then observe passive motion for 60 seconds.

    Geometry, masses, materials, seed and collision settings come from
    ``newton/examples/cloth/example_cloth_poker_cards.py``. The ball is absent.
    The source's massless platform body becomes an equivalent world-fixed box.
    The benchmark uses ten substeps instead of the source's twenty.
    """

    def __init__(self, name: str, *, device=None, substeps=10, iterations=10):
        if name != _SCENE_ID:
            raise ValueError(f"Unknown poker-card scene: {name}")
        if isinstance(substeps, bool) or not isinstance(substeps, int) or not 1 <= substeps <= 10:
            raise ValueError("substeps must be an integer from 1 to 10")
        if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 1:
            raise ValueError("iterations must be a positive integer")

        self.scene_id = name
        self.time = 0.0
        self._frame = 0
        self.frame_dt = 1.0 / 60.0
        self.substeps = substeps
        self.iterations = iterations
        self.dt = self.frame_dt / self.substeps
        self.duration = 65.0
        self.title = "S4 · poker cards without the ball"
        self.description = (
            "52 free cloth cards settle on a fixed cube and ground. "
            "After a 5 s settling allowance, passive observation lasts 60 s. "
            "There is no ball, prescribed motion, pinned card vertex or interactive force."
        )
        self.camera_pos = (0.5, -0.5, 0.3)
        pitch, yaw = np.deg2rad([-15.0, 140.0])
        direction = np.array([np.cos(yaw) * np.cos(pitch), np.sin(yaw) * np.cos(pitch), np.sin(pitch)])
        self.camera_target = tuple(np.asarray(self.camera_pos) + direction)
        self.camera_fov = 70.0
        self.objects = [(35 * card, 35 * (card + 1)) for card in range(52)]

        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.8))
        cube_cfg = newton.ModelBuilder.ShapeConfig(density=0.0, ke=5e6, kd=1e4, mu=0.1)
        builder.add_shape_box(
            body=-1,
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.1), wp.quat_identity()),
            hx=0.1,
            hy=0.1,
            hz=0.1,
            cfg=cube_cfg,
        )
        rng = np.random.default_rng(42)
        for card in range(52):
            offset_x = rng.uniform(-0.005, 0.005)
            offset_y = rng.uniform(-0.005, 0.005)
            angle = rng.uniform(-0.1, 0.1)
            builder.add_cloth_grid(
                pos=wp.vec3(-0.0635 / 2 + offset_x, -0.0889 / 2 + offset_y, 0.25 + card * 0.001),
                rot=wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), angle),
                vel=wp.vec3(0.0),
                dim_x=4,
                dim_y=6,
                cell_x=0.0635 / 4,
                cell_y=0.0889 / 6,
                mass=0.0018 / 35,
                fix_left=False,
                fix_right=False,
                fix_top=False,
                fix_bottom=False,
                tri_ke=1e4,
                tri_ka=1e4,
                tri_kd=1.0,
                edge_ke=100.0,
                edge_kd=1.0,
                particle_radius=0.003,
                color=_CARD_COLORS[card % len(_CARD_COLORS)],
            )
        builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(ke=1e5, kd=100.0, mu=0.3))
        builder.color(include_bending=True)
        self.model = builder.finalize(device=device)
        self.initial_positions = self.model.particle_q.numpy().copy()
        self.model.soft_contact_ke = 1e5
        self.model.soft_contact_kd = 100.0
        self.model.soft_contact_mu = 0.3
        self.solver, self.pipeline = create_solver(
            self.model,
            iterations=self.iterations,
            particle_enable_self_contact=True,
            particle_self_contact_margin=0.001,
            particle_self_contact_gap=0.0005,
            particle_topological_contact_filter_threshold=2,
            particle_rest_shape_contact_exclusion_radius=0.0,
            rigid_body_particle_contact_buffer_size=1024,
            soft_contact_gap=0.005,
            full_surface_contact=True,
        )
        self.contacts = self.solver.contacts
        self._state0 = self.model.state()
        self._state1 = self.model.state()
        self.control = self.model.control()

    @property
    def state(self):
        """Return the current state after double-buffer swaps."""
        return self._state0

    @property
    def phase(self):
        """Name the observation schedule without claiming that cards settled."""
        return "initial settling allowance" if self.time < 5.0 else "stationary observation"

    def step_substep(self, command_time_s):
        """Advance one physical substep with prescribed endpoint commands."""
        self._state0.clear_forces()
        self.solver.step(self._state0, self._state1, self.control, None, self.dt)
        self._state0, self._state1 = self._state1, self._state0
        self.time = float(command_time_s + self.dt)
        return self.state


def build_scene(name: str, *, device=None, substeps=10, iterations=10):
    """Build the unforced poker deck on the requested Warp device."""
    return PokerCardsScene(name, device=device, substeps=substeps, iterations=iterations)
