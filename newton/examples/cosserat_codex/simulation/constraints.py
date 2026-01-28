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
    _warp_apply_concentric_constraint,
    _warp_apply_track_sliding,
    _warp_set_root_on_track,
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
    inner_rod: "WarpResidentRodState",
    outer_rod: "WarpResidentRodState",
    insertion_diff: float,
    stiffness: float = 1.0,
    weight_inner: float = 1.0,
    weight_outer: float = 1.0,
    use_inv_mass_sq: bool = True,
    start_particle: int = 0,
) -> None:
    """Constrain two rods to be concentric (e.g., guidewire inside catheter).
    
    For each particle in the inner rod, computes its corresponding point
    on the outer rod's centerline using arclength-based parametrization
    and applies bilateral corrections.
    
    Args:
        inner_rod: The inner rod (e.g., guidewire).
        outer_rod: The outer rod (e.g., catheter).
        insertion_diff: Difference in insertion depth (inner - outer).
        stiffness: Constraint stiffness [0, 1].
        weight_inner: Weight for inner rod corrections.
        weight_outer: Weight for outer rod corrections.
        use_inv_mass_sq: Whether to use squared weights in denominator.
        start_particle: First particle to apply constraint to.
    """
    if inner_rod.num_points == 0 or outer_rod.num_points == 0:
        return

    wp.launch(
        _warp_apply_concentric_constraint,
        dim=inner_rod.num_points,
        inputs=[
            inner_rod.positions_wp,
            inner_rod.predicted_positions_wp,
            inner_rod.inv_masses_wp,
            int(inner_rod.num_points),
            outer_rod.positions_wp,
            outer_rod.predicted_positions_wp,
            outer_rod.inv_masses_wp,
            int(outer_rod.num_points),
            inner_rod.rest_lengths_wp,
            outer_rod.rest_lengths_wp,
            float(insertion_diff),
            float(stiffness),
            float(weight_inner),
            float(weight_outer),
            int(1 if use_inv_mass_sq else 0),
            int(start_particle),
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
    "apply_floor_collision",
    "apply_tip_bend",
    "apply_track_constraint",
]
