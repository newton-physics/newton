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

"""Concentric constraint kernels for nested Cosserat rods.

This module implements the concentric constraint that keeps an inner rod
(e.g., guidewire) on the centerline of an outer rod (e.g., catheter/sheath).

The constraint uses arc-length parametrization to compute exact correspondence
between inner rod particles and outer rod centerline points based on the known
insertion difference between the two rods.

Physical Model:
    Consider a guidewire sliding through a catheter. Both are inserted into
    a vessel from the same entry point. The "insertion" of each rod is the
    arc-length from the entry point to the rod's tip along the track.
    
    The insertion difference (insertion_inner - insertion_outer) tells us
    how much further the inner rod extends compared to the outer rod.
    
    For each particle on the inner rod at arc-length s_inner from its root,
    the corresponding point on the outer rod is at arc-length:
        s_outer = s_inner + insertion_diff
    
    where insertion_diff = insertion_inner - insertion_outer.

Key Implementation Details:
    1. Arc-length computation: Sum of rest lengths from root to particle
    2. Segment finding: Linear search through cumulative arc-lengths
    3. Interpolation: Barycentric interpolation within segment (t-value)
    4. PBD correction: Bilateral position correction with mass weighting
"""

from __future__ import annotations

import warp as wp


# ==============================================================================
# Helper functions
# ==============================================================================


@wp.func
def _find_segment_from_arclength(
    target_arclength: float,
    rest_lengths: wp.array(dtype=wp.float32),
    num_segments: int,
) -> wp.vec2:
    """Find segment index and parametric t-value from arc-length.
    
    Given a target arc-length along a rod, finds which segment contains that
    point and the parametric position (t ∈ [0, 1]) within that segment.
    
    Args:
        target_arclength: Arc-length from rod root.
        rest_lengths: Rest lengths of each segment.
        num_segments: Number of segments in the rod.
        
    Returns:
        vec2 where x = segment index (as float), y = t-value in [0, 1].
        Returns (-1, 0) if arc-length is negative.
        Returns (num_segments-1, 1) if arc-length exceeds rod length.
    """
    if target_arclength < 0.0:
        return wp.vec2(-1.0, 0.0)
    
    if num_segments <= 0:
        return wp.vec2(-1.0, 0.0)
    
    # Linear search through segments (could use binary search with precomputed cumsum)
    cumulative = float(0.0)
    
    for j in range(num_segments):
        segment_len = rest_lengths[j]
        next_cumulative = cumulative + segment_len
        
        if target_arclength <= next_cumulative:
            # Found the segment containing target_arclength
            if segment_len < 1.0e-10:
                # Degenerate segment
                return wp.vec2(float(j), 0.0)
            t = (target_arclength - cumulative) / segment_len
            t = wp.clamp(t, 0.0, 1.0)
            return wp.vec2(float(j), t)
        
        cumulative = next_cumulative
    
    # Past the end of the rod - clamp to last segment's end
    return wp.vec2(float(num_segments - 1), 1.0)


@wp.func
def _compute_inner_arclength(
    particle_idx: int,
    rest_lengths: wp.array(dtype=wp.float32),
) -> float:
    """Compute arc-length from root to a particle.
    
    The arc-length to particle i is the sum of rest lengths of segments 0..i-1.
    Particle 0 (root) has arc-length 0.
    
    Args:
        particle_idx: Index of the particle.
        rest_lengths: Rest lengths of each segment.
        
    Returns:
        Arc-length from root to the particle.
    """
    arclength = float(0.0)
    for k in range(particle_idx):
        arclength = arclength + rest_lengths[k]
    return arclength


# ==============================================================================
# Concentric Constraint Kernel
# ==============================================================================


@wp.kernel
def warp_concentric_constraint(
    # Outer rod (sheath/catheter) - provides the centerline
    outer_predicted: wp.array(dtype=wp.vec3),
    outer_inv_masses: wp.array(dtype=wp.float32),
    outer_rest_lengths: wp.array(dtype=wp.float32),
    outer_num_points: int,
    # Inner rod (guidewire) - constrained to stay on outer centerline
    inner_predicted: wp.array(dtype=wp.vec3),
    inner_inv_masses: wp.array(dtype=wp.float32),
    inner_rest_lengths: wp.array(dtype=wp.float32),
    inner_num_points: int,
    # Constraint parameters
    insertion_diff: float,  # insertion_inner - insertion_outer
    stiffness: float,  # constraint stiffness in [0, 1]
    # Output: corrections (applied atomically to handle overlapping constraints)
    inner_corrections: wp.array(dtype=wp.vec3),
    outer_corrections: wp.array(dtype=wp.vec3),
):
    """Compute concentric constraint corrections for inner rod particles.
    
    For each inner rod particle, computes the exact corresponding point on
    the outer rod's centerline using arc-length parametrization and insertion
    difference. Outputs position corrections that can be applied atomically.
    
    The constraint equation for particle i on inner rod:
        C_i = ||p_inner[i] - P_outer(s_outer)|| = 0
    
    where:
        s_inner = sum(rest_lengths[0:i])  (arc-length of inner particle)
        s_outer = s_inner + insertion_diff  (corresponding outer arc-length)
        P_outer(s) = lerp(outer[j], outer[j+1], t)  (point on outer centerline)
    
    The segment index j and t-value are found such that:
        cumsum[j] <= s_outer < cumsum[j+1]
        t = (s_outer - cumsum[j]) / (cumsum[j+1] - cumsum[j])
    
    PBD correction:
        Δp_inner = -λ * w_inner * n
        Δp_outer[j] += λ * w_outer[j] * (1-t) * n  
        Δp_outer[j+1] += λ * w_outer[j+1] * t * n
        
    where:
        n = normalize(p_inner - P_outer)  (constraint gradient direction)
        λ = C / w_eff * stiffness
        w_eff = w_inner + w_outer[j]*(1-t)² + w_outer[j+1]*t²
    
    Args:
        outer_predicted: Predicted positions of outer rod particles.
        outer_inv_masses: Inverse masses of outer rod particles.
        outer_rest_lengths: Rest lengths of outer rod segments.
        outer_num_points: Number of particles in outer rod.
        inner_predicted: Predicted positions of inner rod particles.
        inner_inv_masses: Inverse masses of inner rod particles.
        inner_rest_lengths: Rest lengths of inner rod segments.
        inner_num_points: Number of particles in inner rod.
        insertion_diff: insertion_inner - insertion_outer.
        stiffness: Constraint stiffness [0, 1].
        inner_corrections: Output corrections for inner rod (one per particle).
        outer_corrections: Output corrections for outer rod (accumulated atomically).
    """
    tid = wp.tid()
    i = tid  # inner rod particle index
    
    if i >= inner_num_points:
        return
    
    # Get inner particle's inverse mass
    w_inner = inner_inv_masses[i]
    if w_inner <= 0.0:
        # Fixed particle - no correction
        return
    
    # Compute arc-length from inner rod root to particle i
    s_inner = _compute_inner_arclength(i, inner_rest_lengths)
    
    # Corresponding arc-length on outer rod
    # insertion_diff = insertion_inner - insertion_outer
    # If inner is more inserted, its particles map to further positions on outer
    s_outer = s_inner + insertion_diff
    
    # Skip if this particle is before the outer rod's root
    if s_outer < 0.0:
        return
    
    # Find segment and t-value on outer rod
    num_outer_segments = outer_num_points - 1
    if num_outer_segments <= 0:
        return
    
    seg_t = _find_segment_from_arclength(s_outer, outer_rest_lengths, num_outer_segments)
    j = int(seg_t[0])
    t = seg_t[1]
    
    # Skip invalid segments
    if j < 0 or j >= num_outer_segments:
        return
    
    # Skip if at the very tip of outer rod (ease out)
    if j == num_outer_segments - 1 and t > 0.95:
        return
    
    # Compute target point on outer rod centerline via linear interpolation
    p_out_0 = outer_predicted[j]
    p_out_1 = outer_predicted[j + 1]
    target = p_out_0 * (1.0 - t) + p_out_1 * t
    
    # Current inner particle position
    p_inner = inner_predicted[i]
    
    # Constraint violation: delta from inner to target
    delta = p_inner - target
    delta_len = wp.length(delta)
    
    # Skip tiny violations (already satisfied)
    if delta_len < 1.0e-8:
        return
    
    # Constraint gradient direction
    n = delta / delta_len
    
    # Barycentric weights for outer rod vertices
    b0 = 1.0 - t
    b1 = t
    
    # Get inverse masses for outer rod vertices
    w_out_0 = outer_inv_masses[j]
    w_out_1 = outer_inv_masses[j + 1]
    
    # Effective inverse mass (PBD-style with barycentric weighting)
    # The constraint affects inner particle and two outer vertices
    # Each contribution is weighted by the square of the barycentric coordinate
    # This follows from the gradient of the constraint w.r.t. positions
    w_eff = w_inner + w_out_0 * b0 * b0 + w_out_1 * b1 * b1
    
    if w_eff < 1.0e-10:
        return
    
    # PBD Lagrange multiplier (constraint magnitude)
    lambda_val = delta_len / w_eff * stiffness
    
    # Ease out at outer rod tip to avoid instabilities
    if j == num_outer_segments - 1:
        ease = 1.0 - t
        lambda_val = lambda_val * ease
    
    # Compute corrections
    # Inner particle moves toward target (negative direction along n)
    correction_inner = n * lambda_val * w_inner
    inner_corrections[i] = -correction_inner
    
    # Outer rod moves away from inner (positive direction along n)
    # Distributed between vertices based on barycentric weights
    if w_out_0 > 0.0:
        correction_0 = n * lambda_val * w_out_0 * b0
        wp.atomic_add(outer_corrections, j, correction_0)
    
    if w_out_1 > 0.0:
        correction_1 = n * lambda_val * w_out_1 * b1
        wp.atomic_add(outer_corrections, j + 1, correction_1)


@wp.kernel
def warp_apply_corrections(
    positions: wp.array(dtype=wp.vec3),
    predicted: wp.array(dtype=wp.vec3),
    corrections: wp.array(dtype=wp.vec3),
    inv_masses: wp.array(dtype=wp.float32),
):
    """Apply position corrections to particles.
    
    Corrections are only applied to particles with positive inverse mass.
    Updates both current positions and predicted positions.
    
    Args:
        positions: Current particle positions (updated in place).
        predicted: Predicted particle positions (updated in place).
        corrections: Position corrections to apply.
        inv_masses: Inverse masses (corrections skipped if <= 0).
    """
    tid = wp.tid()
    
    if inv_masses[tid] <= 0.0:
        return
    
    corr = corrections[tid]
    new_pos = predicted[tid] + corr
    positions[tid] = new_pos
    predicted[tid] = new_pos


@wp.kernel
def warp_zero_vec3_array(arr: wp.array(dtype=wp.vec3)):
    """Zero out a vec3 array."""
    tid = wp.tid()
    arr[tid] = wp.vec3(0.0, 0.0, 0.0)


# ==============================================================================
# Direct Application Kernel (no atomics, single pass)
# ==============================================================================


@wp.kernel
def warp_concentric_constraint_direct(
    # Outer rod (sheath/catheter) - provides the centerline
    outer_positions: wp.array(dtype=wp.vec3),
    outer_predicted: wp.array(dtype=wp.vec3),
    outer_inv_masses: wp.array(dtype=wp.float32),
    outer_rest_lengths: wp.array(dtype=wp.float32),
    outer_num_points: int,
    # Inner rod (guidewire) - constrained to stay on outer centerline
    inner_positions: wp.array(dtype=wp.vec3),
    inner_predicted: wp.array(dtype=wp.vec3),
    inner_inv_masses: wp.array(dtype=wp.float32),
    inner_rest_lengths: wp.array(dtype=wp.float32),
    inner_num_points: int,
    # Constraint parameters
    insertion_diff: float,  # insertion_inner - insertion_outer
    stiffness: float,  # constraint stiffness in [0, 1]
    weight_inner: float,  # relative weight for inner corrections
    weight_outer: float,  # relative weight for outer corrections
    start_particle: int,  # first inner particle to constrain
    end_particle: int,  # last inner particle to constrain (-1 for all)
):
    """Apply concentric constraint directly without atomic accumulation.
    
    This is a simpler version that directly modifies positions. It's suitable
    when the inner rod particles are sparse enough that race conditions on
    outer rod vertices are rare, or when running multiple constraint iterations.
    
    For each inner rod particle i with arc-length s_inner from root:
        1. Compute s_outer = s_inner + insertion_diff
        2. Find segment j and t-value: cumsum[j] <= s_outer < cumsum[j+1]
        3. Compute target P = outer[j]*(1-t) + outer[j+1]*t
        4. Apply PBD correction to move inner toward P (and optionally outer away)
    
    Args:
        outer_positions: Current positions of outer rod (may be modified).
        outer_predicted: Predicted positions of outer rod (may be modified).
        outer_inv_masses: Inverse masses of outer rod particles.
        outer_rest_lengths: Rest lengths of outer rod segments.
        outer_num_points: Number of particles in outer rod.
        inner_positions: Current positions of inner rod (may be modified).
        inner_predicted: Predicted positions of inner rod (may be modified).
        inner_inv_masses: Inverse masses of inner rod particles.
        inner_rest_lengths: Rest lengths of inner rod segments.
        inner_num_points: Number of particles in inner rod.
        insertion_diff: insertion_inner - insertion_outer.
        stiffness: Constraint stiffness [0, 1].
        weight_inner: Weight multiplier for inner rod corrections.
        weight_outer: Weight multiplier for outer rod corrections.
        start_particle: First inner particle to constrain.
        end_particle: Last inner particle to constrain (-1 = all).
    """
    tid = wp.tid()
    i = tid  # inner rod particle index
    
    if i >= inner_num_points:
        return
    
    # Apply start/end bounds
    if i < start_particle:
        return
    
    effective_end = inner_num_points
    if end_particle >= 0 and end_particle < inner_num_points:
        effective_end = end_particle + 1
    if i >= effective_end:
        return
    
    # Get inner particle's inverse mass
    w_inner = inner_inv_masses[i]
    if w_inner <= 0.0:
        return
    
    # Compute arc-length from inner rod root to particle i
    s_inner = _compute_inner_arclength(i, inner_rest_lengths)
    
    # Corresponding arc-length on outer rod
    s_outer = s_inner + insertion_diff
    
    # Skip if before outer rod's root
    if s_outer < 0.0:
        return
    
    # Find segment and t-value on outer rod
    num_outer_segments = outer_num_points - 1
    if num_outer_segments <= 0:
        return
    
    seg_t = _find_segment_from_arclength(s_outer, outer_rest_lengths, num_outer_segments)
    j = int(seg_t[0])
    t = seg_t[1]
    
    # Skip invalid segments or past-end
    if j < 0 or j >= num_outer_segments:
        return
    
    # Ease out at tip
    if j == num_outer_segments - 1 and t > 0.95:
        return
    
    # Compute target point on outer centerline
    p_out_0 = outer_predicted[j]
    p_out_1 = outer_predicted[j + 1]
    target = p_out_0 * (1.0 - t) + p_out_1 * t
    
    # Current inner position
    p_inner = inner_predicted[i]
    
    # Constraint violation
    delta = p_inner - target
    delta_len = wp.length(delta)
    
    if delta_len < 1.0e-8:
        return
    
    n = delta / delta_len
    
    # Barycentric weights
    b0 = 1.0 - t
    b1 = t
    
    # Outer inverse masses
    w_out_0 = outer_inv_masses[j]
    w_out_1 = outer_inv_masses[j + 1]
    
    # Effective inverse mass with user weights
    w_eff = (w_inner * weight_inner + 
             w_out_0 * b0 * b0 * weight_outer + 
             w_out_1 * b1 * b1 * weight_outer)
    
    if w_eff < 1.0e-10:
        return
    
    # PBD multiplier
    lambda_val = delta_len / w_eff * stiffness
    
    # Ease at tip
    if j == num_outer_segments - 1:
        lambda_val = lambda_val * (1.0 - t)
    
    # Apply corrections directly
    # Inner moves toward target
    if w_inner > 0.0:
        corr_inner = n * lambda_val * w_inner * weight_inner
        new_inner = p_inner - corr_inner
        inner_positions[i] = new_inner
        inner_predicted[i] = new_inner
    
    # Outer moves away from inner
    if w_out_0 > 0.0:
        corr_0 = n * lambda_val * w_out_0 * b0 * weight_outer
        new_0 = p_out_0 + corr_0
        outer_positions[j] = new_0
        outer_predicted[j] = new_0
    
    if w_out_1 > 0.0:
        corr_1 = n * lambda_val * w_out_1 * b1 * weight_outer
        new_1 = p_out_1 + corr_1
        outer_positions[j + 1] = new_1
        outer_predicted[j + 1] = new_1


# ==============================================================================
# Python wrapper for the constraint
# ==============================================================================


class ConcentricConstraint:
    """Concentric constraint manager for nested rods.
    
    This class manages the concentric constraint between an outer rod (e.g.,
    catheter/sheath) and an inner rod (e.g., guidewire). The inner rod is
    constrained to stay on the centerline of the outer rod.
    
    The constraint uses arc-length parametrization and the insertion difference
    between rods to compute exact correspondences.
    
    Example:
        ```python
        constraint = ConcentricConstraint(
            outer_rod=catheter,
            inner_rod=guidewire,
            device=wp.get_device()
        )
        
        # In simulation loop:
        insertion_diff = guidewire_insertion - catheter_insertion
        constraint.apply(insertion_diff, stiffness=1.0)
        ```
    
    Attributes:
        outer_rod: Reference to the outer rod state.
        inner_rod: Reference to the inner rod state.
        device: Warp device for GPU arrays.
        use_atomics: Whether to use atomic accumulation (safer for overlapping).
    """
    
    def __init__(
        self,
        outer_rod,
        inner_rod,
        device=None,
        use_atomics: bool = False,
    ):
        """Initialize the concentric constraint.
        
        Args:
            outer_rod: The outer rod (provides centerline).
            inner_rod: The inner rod (constrained to centerline).
            device: Warp device. Defaults to rod's device or current device.
            use_atomics: If True, use atomic accumulation for outer corrections.
        """
        self.outer_rod = outer_rod
        self.inner_rod = inner_rod
        self.device = device or getattr(outer_rod, 'device', None) or wp.get_device()
        self.use_atomics = use_atomics
        
        if use_atomics:
            # Allocate correction buffers
            self._inner_corrections = wp.zeros(
                inner_rod.num_points, dtype=wp.vec3, device=self.device
            )
            self._outer_corrections = wp.zeros(
                outer_rod.num_points, dtype=wp.vec3, device=self.device
            )
    
    def apply(
        self,
        insertion_diff: float,
        stiffness: float = 1.0,
        weight_inner: float = 1.0,
        weight_outer: float = 1.0,
        start_particle: int = 0,
        end_particle: int = -1,
    ):
        """Apply the concentric constraint.
        
        Args:
            insertion_diff: insertion_inner - insertion_outer.
            stiffness: Constraint stiffness [0, 1].
            weight_inner: Weight multiplier for inner rod corrections.
            weight_outer: Weight multiplier for outer rod corrections.
            start_particle: First inner particle to constrain.
            end_particle: Last inner particle to constrain (-1 = all).
        """
        if self.inner_rod.num_points == 0 or self.outer_rod.num_points == 0:
            return
        
        if self.use_atomics:
            self._apply_with_atomics(
                insertion_diff, stiffness, weight_inner, weight_outer,
                start_particle, end_particle
            )
        else:
            self._apply_direct(
                insertion_diff, stiffness, weight_inner, weight_outer,
                start_particle, end_particle
            )
    
    def _apply_direct(
        self,
        insertion_diff: float,
        stiffness: float,
        weight_inner: float,
        weight_outer: float,
        start_particle: int,
        end_particle: int,
    ):
        """Apply constraint using direct writes (may have race conditions)."""
        wp.launch(
            warp_concentric_constraint_direct,
            dim=self.inner_rod.num_points,
            inputs=[
                self.outer_rod.positions_wp,
                self.outer_rod.predicted_positions_wp,
                self.outer_rod.inv_masses_wp,
                self.outer_rod.rest_lengths_wp,
                int(self.outer_rod.num_points),
                self.inner_rod.positions_wp,
                self.inner_rod.predicted_positions_wp,
                self.inner_rod.inv_masses_wp,
                self.inner_rod.rest_lengths_wp,
                int(self.inner_rod.num_points),
                float(insertion_diff),
                float(stiffness),
                float(weight_inner),
                float(weight_outer),
                int(start_particle),
                int(end_particle),
            ],
            device=self.device,
        )
    
    def _apply_with_atomics(
        self,
        insertion_diff: float,
        stiffness: float,
        weight_inner: float,
        weight_outer: float,
        start_particle: int,
        end_particle: int,
    ):
        """Apply constraint using atomic accumulation (race-condition safe)."""
        # Zero correction buffers
        wp.launch(
            warp_zero_vec3_array,
            dim=self.inner_rod.num_points,
            inputs=[self._inner_corrections],
            device=self.device,
        )
        wp.launch(
            warp_zero_vec3_array,
            dim=self.outer_rod.num_points,
            inputs=[self._outer_corrections],
            device=self.device,
        )
        
        # Compute corrections (inner written directly, outer accumulated atomically)
        wp.launch(
            warp_concentric_constraint,
            dim=self.inner_rod.num_points,
            inputs=[
                self.outer_rod.predicted_positions_wp,
                self.outer_rod.inv_masses_wp,
                self.outer_rod.rest_lengths_wp,
                int(self.outer_rod.num_points),
                self.inner_rod.predicted_positions_wp,
                self.inner_rod.inv_masses_wp,
                self.inner_rod.rest_lengths_wp,
                int(self.inner_rod.num_points),
                float(insertion_diff),
                float(stiffness),
                self._inner_corrections,
                self._outer_corrections,
            ],
            device=self.device,
        )
        
        # Apply inner corrections
        wp.launch(
            warp_apply_corrections,
            dim=self.inner_rod.num_points,
            inputs=[
                self.inner_rod.positions_wp,
                self.inner_rod.predicted_positions_wp,
                self._inner_corrections,
                self.inner_rod.inv_masses_wp,
            ],
            device=self.device,
        )
        
        # Apply outer corrections
        wp.launch(
            warp_apply_corrections,
            dim=self.outer_rod.num_points,
            inputs=[
                self.outer_rod.positions_wp,
                self.outer_rod.predicted_positions_wp,
                self._outer_corrections,
                self.outer_rod.inv_masses_wp,
            ],
            device=self.device,
        )


__all__ = [
    "warp_concentric_constraint",
    "warp_concentric_constraint_direct",
    "warp_apply_corrections",
    "warp_zero_vec3_array",
    "ConcentricConstraint",
]
