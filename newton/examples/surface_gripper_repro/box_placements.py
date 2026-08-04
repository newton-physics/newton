# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
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

"""Auto-placement of the suction-cup example's pick boxes (panel + crates) and their pallets.

Each box and its pallet are seated from the arm's flange pose at the box's pick engagement (forward
kinematics), so they stay under the cups regardless of smoothing, cup layout, or recording. The
example keeps its own geometry constants (``PANEL``, ``CRATE``, ...) and packs the ones needed here
into a :class:`PlacementConfig` passed to :func:`compute_box_placements`.
"""

from dataclasses import dataclass

import numpy as np
import warp as wp

import newton


@dataclass
class PlacementConfig:
    """Geometry :func:`compute_box_placements` needs, packed from the example's module constants.

    ``panel`` / ``crate`` are ``((hx, hy, hz) half-extents [m], mass [kg])``; the ``*_pallet_half`` are
    ``(hx, hy, hz)`` pallet half-extents [m].
    """

    num_arm_dofs: int  # arm joints to write when posing the flange for FK (J1-J6)
    finger_hull_deepest_x: float  # deepest finger-hull reach along the suction axis, flange frame [m]
    gripper_pads: tuple  # cup positions in the flange frame [m]; their mean is the cup center
    panel: tuple  # 1st-engagement box spec
    crate: tuple  # 2nd-engagement box spec (each crate)
    num_crates: int  # number of crates picked after the panel
    panel_pallet_half: tuple  # panel pallet half-extents
    crate_pallet_half: tuple  # shared crate pallet half-extents


@dataclass
class BoxPlacements:
    """How to add every pick box to the builder (index 0 the panel, 1.. the crates), plus the pallets."""

    masses: list  # [B] body mass [kg]
    inertias: list  # [B] body inertia (wp.mat33), solid-box formula
    dims: list  # [B] (hx, hy, hz) half-extents [m]
    pick_poses: list  # [B] pose where the cups grip the box (wp.transform)
    wait_poses: list  # [B] pose where the box is created, before it is moved to its pick pose
    pallet_poses: list  # static pallet poses to add: the panel's, then the one shared crate pallet
    pallet_dims: list  # matching (hx, hy, hz) half-extents [m] for each pallet in pallet_poses


def _solid_box_inertia(spec):
    """Diagonal inertia of a uniform solid box ``spec = ((hx, hy, hz), mass)`` about its center."""
    (hx, hy, hz), mass = spec
    return wp.mat33(
        mass / 3.0 * (hy * hy + hz * hz),
        0.0,
        0.0,
        0.0,
        mass / 3.0 * (hx * hx + hz * hz),
        0.0,
        0.0,
        0.0,
        mass / 3.0 * (hx * hx + hy * hy),
    )


def box_top_world(robot_arm_model, robot_arm_state, ee_body, cup_c, rec_targets, engage_frame, cfg):
    """World position of the box top when the flange is at the given engagement frame's pose.

    FK the arm to the recorded targets at ``engage_frame`` -> flange pose, then offset by the cup
    contact point (finger-hull deepest along the suction axis, at the cup center ``cup_c``) to where
    the cups meet the box top.
    """
    q = robot_arm_state.joint_q.numpy()
    q[: cfg.num_arm_dofs] = rec_targets[engage_frame]
    robot_arm_state.joint_q.assign(q)
    newton.eval_fk(robot_arm_model, robot_arm_state.joint_q, robot_arm_state.joint_qd, robot_arm_state)
    tf = robot_arm_state.body_q.numpy()[ee_body]  # flange world pose [px,py,pz,qx,qy,qz,qw]
    bt = wp.vec3(*tf[:3]) + wp.quat_rotate(
        wp.quat(*tf[3:7]), wp.vec3(cfg.finger_hull_deepest_x, float(cup_c[1]), float(cup_c[2]))
    )
    return float(bt[0]), float(bt[1]), float(bt[2])


def box_and_pallet_poses(robot_arm_model, robot_arm_state, ee_body, cup_c, robot_playback, engage_index, spec, cfg):
    """World center of the picked box body and of the static pallet it rests on, for the box picked at
    the ``engage_index``-th engagement. FK the flange there (:func:`box_top_world`) then drop by the box
    geometry ``spec = ((hx, hy, hz), mass)``."""
    rec_targets = robot_playback.rec_targets_wp.numpy()
    top = box_top_world(
        robot_arm_model, robot_arm_state, ee_body, cup_c, rec_targets, robot_playback.rising[engage_index], cfg
    )
    (_hx, _hy, hz), _mass = spec
    box_center = wp.vec3(top[0], top[1], top[2] - hz)  # body center: half a box below the top face
    pallet_center = wp.vec3(top[0], top[1], top[2] - 2.0 * hz - 0.5)  # pallet directly under the box (half-height 0.5)
    return box_center, pallet_center


def compute_box_placements(robot_arm_builder, robot_playback, ee_body, cfg):
    """Compute how to add every pick box (the panel + all crates) to ``robot_arm_builder``, from the arm's flange
    pose at each pick engagement (``rising[0]`` the panel, ``rising[n+1]`` the nth crate). Finalizes
    ``robot_arm_builder`` into an arm-only model for the FK probes, so call it before adding the boxes. The panel
    is created where it is gripped; each crate is created parked in a line and moved to its grip pose at
    pick time. The panel gets its own pallet; the crates share one. ``cfg`` is a :class:`PlacementConfig`.
    """
    robot_arm_model = robot_arm_builder.finalize()  # arm-only model, for the FK placement probes
    robot_arm_state = robot_arm_model.state()
    cup_c = np.mean(cfg.gripper_pads, axis=0)  # box top seats at the finger-hull deepest, under the cups
    p = BoxPlacements([], [], [], [], [], [], [])

    # Panel: created (and gripped) where the cups meet it at the 1st engagement; its own pallet.
    panel_center, panel_pallet = box_and_pallet_poses(
        robot_arm_model, robot_arm_state, ee_body, cup_c, robot_playback, 0, cfg.panel, cfg
    )
    panel_pose = wp.transform(panel_center, wp.quat_identity())  # panel rests axis-aligned
    p.masses.append(cfg.panel[1])
    p.inertias.append(_solid_box_inertia(cfg.panel))
    p.dims.append(cfg.panel[0])
    p.pick_poses.append(panel_pose)
    p.wait_poses.append(panel_pose)  # panel is created at its pick pose (no separate waiting spot)
    p.pallet_poses.append(wp.transform(panel_pallet, wp.quat_identity()))
    p.pallet_dims.append(cfg.panel_pallet_half)  # pallet snug to the panel

    # Crates: gripped at their own engagement (rising[n+1]); created parked off-scene in a line; the
    # crates share one static pallet under the (common) grip spot.
    crate_quat = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), np.pi / 2.0)  # crates rest rotated 90deg about z
    (_chx, _chy, chz), _crate_mass = cfg.crate
    for n in range(cfg.num_crates):
        grip, crate_pallet = box_and_pallet_poses(
            robot_arm_model, robot_arm_state, ee_body, cup_c, robot_playback, n + 1, cfg.crate, cfg
        )
        if n == 0:
            p.pallet_poses.append(wp.transform(crate_pallet, wp.quat_identity()))  # one shared crate pallet
            p.pallet_dims.append(cfg.crate_pallet_half)  # snug pedestal sized to the rotated crate
        p.masses.append(cfg.crate[1])
        p.inertias.append(_solid_box_inertia(cfg.crate))
        p.dims.append(cfg.crate[0])
        p.pick_poses.append(wp.transform(grip, crate_quat))
        p.wait_poses.append(wp.transform(wp.vec3(-3.0, -3.0 - 0.6 * float(n), chz), crate_quat))  # parked, in a line
    return p
