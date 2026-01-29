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

"""External constraint handlers for Cosserat rod simulation.

This module provides constraint application functions for:
- Track sliding (constraining rod roots to follow a track)
- Concentric constraints (inner/outer rod coupling)
- Tip bending (apply bending torques at tips)
- Mesh collisions (collision with triangle meshes)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import warp as wp

from newton.examples.cosserat_codex.kernels import (
    ConcentricConstraint,
    _warp_apply_concentric_constraint_v2,
    _warp_apply_track_sliding,
    _warp_set_root_on_track,
    warp_concentric_constraint_direct,
)

if TYPE_CHECKING:
    from newton.examples.cosserat_codex.rod.warp_rod import WarpResidentRodState


def apply_track_constraint(
    rod: "WarpResidentRodState",
    track_start: np.ndarray,
    track_end: np.ndarray,
    insertion: float,
    sliding_stiffness: float = 1.0,
    slide_interior: bool = True,
) -> None:
    """Constrain a rod to slide along a track line segment.
    
    Sets the root position on the track based on insertion depth and
    optionally applies sliding constraints to interior particles.
    
    Args:
        rod: The rod state to constrain.
        track_start: 3D start point of the track.
        track_end: 3D end point of the track.
        insertion: Insertion depth along the track.
        sliding_stiffness: Stiffness for interior sliding constraint.
        slide_interior: Whether to apply sliding to interior particles.
    """
    if rod.num_points == 0:
        return

    track_start_wp = wp.vec3(float(track_start[0]), float(track_start[1]), float(track_start[2]))
    track_end_wp = wp.vec3(float(track_end[0]), float(track_end[1]), float(track_end[2]))

    # Set root position on track
    wp.launch(
        _warp_set_root_on_track,
        dim=1,
        inputs=[
            rod.positions_wp,
            rod.predicted_positions_wp,
            rod.velocities_wp,
            track_start_wp,
            track_end_wp,
            float(insertion),
            int(0),  # root_idx
        ],
        device=rod.device,
    )

    # Apply sliding constraint to interior particles
    if slide_interior and rod.num_points > 1:
        wp.launch(
            _warp_apply_track_sliding,
            dim=rod.num_points - 1,
            inputs=[
                rod.positions_wp,
                rod.predicted_positions_wp,
                rod.inv_masses_wp,
                track_start_wp,
                track_end_wp,
                float(sliding_stiffness),
                int(1),  # start_idx
                int(rod.num_points),  # end_idx
            ],
            device=rod.device,
        )


def apply_concentric_constraint(
    outer_rod: "WarpResidentRodState",
    inner_rod: "WarpResidentRodState",
    insertion_diff: float,
    stiffness: float = 1.0,
    weight_inner: float = 1.0,
    weight_outer: float = 1.0,
    start_particle: int = 0,
    end_particle: int = -1,
) -> None:
    """Constrain inner rod to stay on outer rod's centerline.
    
    For each particle in the inner rod (e.g., guidewire), computes its exact
    corresponding point on the outer rod's centerline (e.g., catheter) using
    arc-length parametrization and applies PBD-style bilateral corrections.
    
    The correspondence uses the insertion difference to map inner rod particles
    to outer rod centerline positions. This models a guidewire sliding through
    a catheter based on their relative insertion depths.
    
    Args:
        outer_rod: The outer rod (e.g., catheter) - provides the centerline.
        inner_rod: The inner rod (e.g., guidewire) - constrained to centerline.
        insertion_diff: insertion_inner - insertion_outer. Positive when inner
            rod is more inserted (its root is ahead of outer rod's root).
        stiffness: Constraint stiffness [0, 1].
        weight_inner: Weight for inner rod corrections.
        weight_outer: Weight for outer rod corrections.
        start_particle: First inner particle index to constrain.
        end_particle: Last inner particle index to constrain (-1 = all).
    """
    if inner_rod.num_points == 0 or outer_rod.num_points == 0:
        return

    wp.launch(
        _warp_apply_concentric_constraint_v2,
        dim=inner_rod.num_points,
        inputs=[
            # Outer rod (catheter) - centerline we constrain to
            outer_rod.positions_wp,
            outer_rod.predicted_positions_wp,
            outer_rod.inv_masses_wp,
            int(outer_rod.num_points),
            outer_rod.rest_lengths_wp,
            # Inner rod (guidewire) - constrained to stay on outer centerline
            inner_rod.positions_wp,
            inner_rod.predicted_positions_wp,
            inner_rod.inv_masses_wp,
            int(inner_rod.num_points),
            inner_rod.rest_lengths_wp,
            # Constraint parameters
            float(insertion_diff),
            float(stiffness),
            float(weight_inner),
            float(weight_outer),
            int(start_particle),
            int(end_particle),
        ],
        device=inner_rod.device,
    )


def apply_concentric_constraint_v3(
    outer_rod: "WarpResidentRodState",
    inner_rod: "WarpResidentRodState",
    insertion_diff: float,
    stiffness: float = 1.0,
    weight_inner: float = 1.0,
    weight_outer: float = 1.0,
    start_particle: int = 0,
    end_particle: int = -1,
) -> None:
    """Constrain inner rod to stay on outer rod's centerline (v3 implementation).
    
    This is an improved version of the concentric constraint that uses cleaner
    arc-length parametrization to compute exact correspondences between inner
    rod particles and outer rod centerline points.
    
    For each inner particle at arc-length s_inner from its root:
        1. Compute corresponding outer arc-length: s_outer = s_inner + insertion_diff
        2. Find segment j and t-value where: cumsum[j] <= s_outer < cumsum[j+1]
        3. Compute target point: P = outer[j]*(1-t) + outer[j+1]*t
        4. Apply PBD bilateral correction to minimize ||inner - P||
    
    The insertion_diff parameter models the relative insertion depths:
        - Positive: inner rod is more inserted (extends further into vessel)
        - Negative: outer rod is more inserted
        - Zero: both rods have equal insertion (roots aligned)
    
    Args:
        outer_rod: The outer rod (e.g., catheter/sheath) - provides centerline.
        inner_rod: The inner rod (e.g., guidewire) - constrained to centerline.
        insertion_diff: insertion_inner - insertion_outer.
        stiffness: Constraint stiffness [0, 1].
        weight_inner: Weight multiplier for inner rod corrections.
        weight_outer: Weight multiplier for outer rod corrections.
        start_particle: First inner particle index to constrain.
        end_particle: Last inner particle index to constrain (-1 = all).
    """
    if inner_rod.num_points == 0 or outer_rod.num_points == 0:
        return

    wp.launch(
        warp_concentric_constraint_direct,
        dim=inner_rod.num_points,
        inputs=[
            # Outer rod (catheter) - provides centerline
            outer_rod.positions_wp,
            outer_rod.predicted_positions_wp,
            outer_rod.inv_masses_wp,
            outer_rod.rest_lengths_wp,
            int(outer_rod.num_points),
            # Inner rod (guidewire) - constrained to centerline
            inner_rod.positions_wp,
            inner_rod.predicted_positions_wp,
            inner_rod.inv_masses_wp,
            inner_rod.rest_lengths_wp,
            int(inner_rod.num_points),
            # Constraint parameters
            float(insertion_diff),
            float(stiffness),
            float(weight_inner),
            float(weight_outer),
            int(start_particle),
            int(end_particle),
        ],
        device=inner_rod.device,
    )


def apply_tip_bend(
    rod: "WarpResidentRodState",
    bend_angle_d1: float,
    bend_angle_d2: float,
    num_segments: int = 3,
) -> None:
    """Apply bending to the tip of a rod by modifying rest Darboux vectors.
    
    This creates a curved tip by setting non-zero rest curvature for the
    last few segments of the rod.
    
    Args:
        rod: The rod state to modify.
        bend_angle_d1: Bend angle in d1 direction (radians).
        bend_angle_d2: Bend angle in d2 direction (radians).
        num_segments: Number of segments to apply bending to.
    """
    if rod.num_edges == 0:
        return

    # Modify rest Darboux for the last num_segments edges
    rest_darboux = rod.rest_darboux.copy()
    start_edge = max(0, rod.num_edges - num_segments)

    for i in range(start_edge, rod.num_edges):
        rest_darboux[i, 0] = bend_angle_d1 / rod.segment_length
        rest_darboux[i, 1] = bend_angle_d2 / rod.segment_length
        # Keep twist as is

    # Update host array
    rod.rest_darboux[:] = rest_darboux

    # Sync to device
    rod.rest_darboux_wp.assign(
        wp.array(rest_darboux[:, 0:3], dtype=wp.vec3, device=rod.device)
    )


def apply_floor_collision(
    rod: "WarpResidentRodState",
    floor_z: float,
    restitution: float = 0.0,
) -> None:
    """Apply floor collision constraint to a rod.
    
    Args:
        rod: The rod state to constrain.
        floor_z: Z coordinate of the floor plane.
        restitution: Coefficient of restitution for bouncing.
    """
    rod.apply_floor_collisions(floor_z, restitution)


__all__ = [
    "apply_concentric_constraint",
    "apply_concentric_constraint_v3",
    "apply_floor_collision",
    "apply_tip_bend",
    "apply_track_constraint",
    "ConcentricConstraint",
]
