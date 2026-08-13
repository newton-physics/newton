# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Hydroelastic contact reduction using hashtable-based tracking.

This module provides hydroelastic-specific contact reduction functionality,
building on the core ``GlobalContactReducer`` from ``contact_reduction_global.py``.

**Hydroelastic Contact Features:**

- Aggregate stiffness calculation: ``c_stiffness = |agg_force| / total_depth`` where
  ``agg_force = sum(area * pressure_func(depth) * normal)`` is in physical force units
- Normal matching: rotates reduced normals to align with aggregate force direction
- Anchor contact: synthetic contact at center of pressure for moment balance

**Usage:**

Use ``HydroelasticContactReduction`` for the high-level API, or call the individual
kernels for more control over the pipeline.

See Also:
    :class:`GlobalContactReducer` in ``contact_reduction_global.py`` for the
    core contact reduction system.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import warp as wp

from newton._src.geometry.hashtable import hashtable_find_or_insert

from .contact_data import ContactData
from .contact_reduction import (
    NUM_NORMAL_BINS,
    NUM_SPATIAL_DIRECTIONS,
    NUM_VOXEL_DEPTH_SLOTS,
    compute_voxel_index,
    get_slot,
    get_spatial_direction_2d,
    project_point_to_plane,
)
from .contact_reduction_global import (
    BETA_THRESHOLD,
    BIN_MASK,
    VALUES_PER_KEY,
    GlobalContactReducer,
    GlobalContactReducerData,
    _unpack_contact_id_det,
    _unpack_contact_id_fast,
    decode_oct,
    export_contact_to_buffer,
    is_contact_already_exported,
    make_contact_key,
    make_contact_value,
    reduction_update_slot,
    unpack_contact_id,
)

# =============================================================================
# Constants for hydroelastic export
# =============================================================================

EPS_LARGE = 1e-8
EPS_SMALL = 1e-20
MIN_FRICTION_SCALE = 1e-2

# =============================================================================
# Deterministic fixed-point accumulation
# =============================================================================
#
# Floating-point addition is not associative, so an unordered ``atomic_add``
# gives results that depend on GPU thread scheduling.  Integer addition *is*
# associative, so in deterministic mode every per-bin aggregate is accumulated
# as int64 fixed-point instead and converted back once the sum is complete.
# This needs no ordering constraint at all, so no sorting, replay or two-pass
# kernel lowering is involved.
#
# The fixed-point grid is chosen per hashtable entry so no magnitude has to be
# assumed: a first pass records ``max(|contribution|)`` as a binary exponent via
# ``atomic_max`` (order-independent for integers), and the accumulating pass
# shifts every contribution onto that entry's grid.  The mantissa width is
# derived from how many terms can reach one entry, so overflow is impossible by
# construction rather than by assumption.  See :func:`_fixed_mantissa_bits`.

# Sentinel exponent meaning "this entry saw no usable contribution".
FIXED_EXP_NONE = wp.constant(-10000)

# Fixed accumulator slots, laid out slot-major as ``slot * ht_capacity + entry``
# to match ``ht_values``.
FIXED_AGG_FORCE = wp.constant(0)  # 3 components
FIXED_WEIGHT_SUM = wp.constant(3)
FIXED_WEIGHTED_POS = wp.constant(4)  # 3 components
FIXED_DEPTH_VOLUME = wp.constant(7)  # 3 components
FIXED_TOTAL_DEPTH = wp.constant(10)
FIXED_TOTAL_NORMAL = wp.constant(11)  # 3 components
FIXED_MOMENT_UNREDUCED = wp.constant(14)
FIXED_MOMENT_REDUCED = wp.constant(15)
FIXED_MOMENT2_REDUCED = wp.constant(16)
NUM_FIXED_SLOTS = 17

# Scale families, laid out ``family * ht_capacity + entry``.  Quantities that
# share a physical scale share a family so they stay mutually consistent.
SCALE_FORCE = wp.constant(0)  # agg_force, weight_sum
SCALE_WEIGHTED_POS = wp.constant(1)
SCALE_DEPTH_VOLUME = wp.constant(2)
SCALE_REDUCED_DEPTH = wp.constant(3)  # total_depth_reduced, total_normal_reduced
SCALE_MOMENT_UNREDUCED = wp.constant(4)
SCALE_MOMENT_REDUCED = wp.constant(5)
SCALE_MOMENT2_REDUCED = wp.constant(6)
NUM_SCALE_FAMILIES = 7

# Accumulation phases for the two-pass (scale, then add) accumulators.
PHASE_SCALE = 0
PHASE_ACCUMULATE = 1

# Finalize groups, ordered by when their inputs become complete.
FINALIZE_UNREDUCED = 0
FINALIZE_REDUCED_DEPTH = 1
FINALIZE_MOMENTS = 2


def _fixed_mantissa_bits(max_terms: int) -> int:
    """Fixed-point mantissa width that cannot overflow for ``max_terms`` terms.

    A contribution equal to the entry maximum scales to just under
    ``2**(bits + 1)``, because ``|x| / 2**exponent`` lies in ``[1, 2)``.  Summing
    ``max_terms`` of those therefore stays at or below ``2**62``, which fits a
    signed int64 in both directions with a bit to spare.

    Args:
        max_terms: Upper bound on how many contributions reach a single entry.
    """
    count_bits = max(int(max_terms - 1), 1).bit_length()
    return 61 - count_bits


@wp.func
def _fixed_exponent(x: float) -> int:
    """Binary exponent of ``|x|``, or :data:`FIXED_EXP_NONE` for ~zero."""
    magnitude = wp.abs(x)
    if magnitude < 1.0e-30:
        return FIXED_EXP_NONE
    return int(wp.floor(wp.log2(magnitude)))


@wp.func
def _max_abs_component(v: wp.vec3) -> float:
    """Largest absolute component of ``v``, used to size its fixed-point grid."""
    return wp.max(wp.max(wp.abs(v[0]), wp.abs(v[1])), wp.abs(v[2]))


@wp.func
def _fixed_shift(exp_ref: int, mantissa_bits: int) -> wp.float64:
    """Power-of-two factor mapping an entry's grid onto the fixed-point range.

    Returns zero for the sentinel exponent so that quantization collapses to
    zero without a branch at every call site.  This is a double-precision
    ``pow``, so callers hoist it out of per-component loops: computing it once
    per entry instead of once per scalar is ~10x cheaper on consumer GPUs.
    """
    if exp_ref == FIXED_EXP_NONE:
        return wp.float64(0.0)
    # Every non-sentinel exponent comes from a finite float32 contribution, so
    # forming the exact shift in float64 covers its full exponent range while
    # preserving the int64 overflow proof in ``_fixed_mantissa_bits``.
    return wp.pow(wp.float64(2.0), wp.float64(mantissa_bits - exp_ref))


@wp.func
def _to_fixed_scaled(x: float, shift: wp.float64) -> wp.int64:
    """Quantize ``x`` onto a grid whose shift was already computed."""
    return wp.int64(wp.round(wp.float64(x) * shift))


@wp.func
def _from_fixed_scaled(v: wp.int64, shift: wp.float64) -> float:
    """Convert a completed fixed-point sum back to float32."""
    if shift == wp.float64(0.0):
        return 0.0
    return wp.float32(wp.float64(v) / shift)


@wp.func
def _to_fixed(x: float, exp_ref: int, mantissa_bits: int) -> wp.int64:
    """Quantize ``x`` onto the fixed-point grid selected for its entry."""
    return _to_fixed_scaled(x, _fixed_shift(exp_ref, mantissa_bits))


@wp.func
def _from_fixed(v: wp.int64, exp_ref: int, mantissa_bits: int) -> float:
    """Convert a completed fixed-point sum back to float32."""
    return _from_fixed_scaled(v, _fixed_shift(exp_ref, mantissa_bits))


@wp.func
def _record_scale(scale: wp.array[wp.int32], family: int, entry_idx: int, ht_capacity: int, value: float):
    """Track the per-entry maximum exponent for a scale family."""
    exponent = _fixed_exponent(value)
    if exponent != FIXED_EXP_NONE:
        wp.atomic_max(scale, family * ht_capacity + entry_idx, exponent)


@wp.func
def _add_fixed(
    accum: wp.array[wp.int64],
    slot: int,
    entry_idx: int,
    ht_capacity: int,
    value: float,
    shift: wp.float64,
):
    """Accumulate a scalar contribution in fixed point."""
    wp.atomic_add(accum, slot * ht_capacity + entry_idx, _to_fixed_scaled(value, shift))


@wp.func
def _add_fixed_vec(
    accum: wp.array[wp.int64],
    slot: int,
    entry_idx: int,
    ht_capacity: int,
    value: wp.vec3,
    shift: wp.float64,
):
    """Accumulate a vector contribution component-wise in fixed point."""
    for i in range(3):
        wp.atomic_add(accum, (slot + i) * ht_capacity + entry_idx, _to_fixed_scaled(value[i], shift))


@wp.func
def _read_fixed_vec(
    accum: wp.array[wp.int64], slot: int, entry_idx: int, ht_capacity: int, shift: wp.float64
) -> wp.vec3:
    """Convert a completed fixed-point vector sum back to float32."""
    return wp.vec3(
        _from_fixed_scaled(accum[slot * ht_capacity + entry_idx], shift),
        _from_fixed_scaled(accum[(slot + 1) * ht_capacity + entry_idx], shift),
        _from_fixed_scaled(accum[(slot + 2) * ht_capacity + entry_idx], shift),
    )


def _unpack_contact_id_variant(deterministic: bool):
    """Select the contact-id unpacking function matching the active packing."""
    return _unpack_contact_id_det if deterministic else _unpack_contact_id_fast


@wp.func
def _compute_normal_matching_rotation(
    selected_normal_sum: wp.vec3,
    agg_force_vec: wp.vec3,
    agg_force_mag: wp.float32,
) -> wp.quat:
    """Compute rotation quaternion that aligns selected_normal_sum with agg_force direction.

    Callers gate reliability on the aggregate depth-volume magnitude; this helper
    only needs ``agg_force_mag`` above ``EPS_SMALL`` so the
    ``agg_force_vec / agg_force_mag`` normalization is well-defined.
    """
    rotation_q = wp.quat_identity()
    selected_mag = wp.length(selected_normal_sum)
    if selected_mag > EPS_LARGE and agg_force_mag > EPS_SMALL:
        selected_dir = selected_normal_sum / selected_mag
        agg_dir = agg_force_vec / agg_force_mag

        cross = wp.cross(selected_dir, agg_dir)
        cross_mag = wp.length(cross)
        dot_val = wp.dot(selected_dir, agg_dir)

        if cross_mag > EPS_LARGE:
            axis = cross / cross_mag
            angle = wp.acos(wp.clamp(dot_val, -1.0, 1.0))
            rotation_q = wp.quat_from_axis_angle(axis, angle)
        elif dot_val < 0.0:
            perp = wp.vec3(1.0, 0.0, 0.0)
            if wp.abs(wp.dot(selected_dir, perp)) > 0.9:
                perp = wp.vec3(0.0, 1.0, 0.0)
            axis = wp.normalize(wp.cross(selected_dir, perp))
            rotation_q = wp.quat_from_axis_angle(axis, 3.14159265359)
    return rotation_q


@wp.func
def _effective_stiffness(k_a: wp.float32, k_b: wp.float32) -> wp.float32:
    denom = k_a + k_b
    if denom <= 0.0:
        return 0.0
    return (k_a * k_b) / denom


# =============================================================================
# Hydroelastic contact buffer function
# =============================================================================


@wp.func
def export_hydroelastic_contact_to_buffer(
    shape_a: int,
    shape_b: int,
    position: wp.vec3,
    normal: wp.vec3,
    depth: float,
    area: float,
    fingerprint: int,
    reducer_data: GlobalContactReducerData,
) -> int:
    """Store a hydroelastic contact in the buffer with face area.

    Extends :func:`export_contact_to_buffer` by storing the face area.
    Per-contact pressure is recomputed on demand by downstream kernels via
    ``pressure_func(depth, shape_b, pressure_data)`` rather than cached here:
    ``depth`` and ``shape_b`` are already in the buffer (``position_depth`` and
    ``shape_pairs``), so a memo would only duplicate the call.

    Args:
        shape_a: First shape index
        shape_b: Second shape index
        position: Contact position in world space
        normal: Contact normal
        depth: Penetration depth (negative = penetrating, standard convention)
        area: Contact surface area
        fingerprint: Stable geometric identifier for deterministic reduction.
        reducer_data: GlobalContactReducerData with all arrays

    Returns:
        Contact ID if successfully stored, -1 if buffer full
    """
    contact_id = export_contact_to_buffer(shape_a, shape_b, position, normal, depth, fingerprint, reducer_data)

    if contact_id >= 0:
        reducer_data.contact_area[contact_id] = area

    return contact_id


@wp.kernel(enable_backward=False)
def _register_hydroelastic_normal_bins_kernel(
    reducer_data: GlobalContactReducerData,
    shape_material_k_hydro: wp.array[wp.float32],
    total_num_threads: int,
):
    """Register normal-bin keys before deterministic aggregate accumulation."""
    tid = wp.tid()
    num_contacts = wp.min(reducer_data.contact_count[0], reducer_data.capacity)
    for buffer_idx in range(tid, num_contacts, total_num_threads):
        contact_id = buffer_idx + 1
        pair = reducer_data.shape_pairs[contact_id]
        normal = decode_oct(reducer_data.normal[contact_id])
        key = make_contact_key(pair[0], pair[1], get_slot(normal))
        entry_idx = hashtable_find_or_insert(key, reducer_data.ht_keys, reducer_data.ht_active_slots)
        reducer_data.contact_nbin_entry[contact_id] = entry_idx
        if entry_idx >= 0:
            reducer_data.entry_k_eff[entry_idx] = _effective_stiffness(
                shape_material_k_hydro[pair[0]], shape_material_k_hydro[pair[1]]
            )
        else:
            wp.atomic_add(reducer_data.ht_insert_failures, 0, 1)


def _create_unreduced_aggregate_kernel(pressure_func, mantissa_bits: int):
    """Create the per-contact unreduced-aggregate accumulation kernel.

    Runs twice: :data:`PHASE_SCALE` records the per-entry fixed-point exponents,
    :data:`PHASE_ACCUMULATE` sums the contributions as int64.

    Args:
        pressure_func: Warp function ``(signed_depth, shape_idx, data) -> pressure``.
        mantissa_bits: Fixed-point mantissa width from :func:`_fixed_mantissa_bits`.
    """

    @wp.kernel(enable_backward=False)
    def accumulate_unreduced_aggregates_kernel(
        reducer_data: GlobalContactReducerData,
        pressure_data: Any,
        fixed_accum: wp.array[wp.int64],
        fixed_scale: wp.array[wp.int32],
        phase: int,
        total_num_threads: int,
    ):
        tid = wp.tid()
        ht_capacity = reducer_data.ht_capacity
        num_contacts = wp.min(reducer_data.contact_count[0], reducer_data.capacity)

        for buffer_idx in range(tid, num_contacts, total_num_threads):
            contact_id = buffer_idx + 1
            entry_idx = reducer_data.contact_nbin_entry[contact_id]
            if entry_idx < 0:
                continue
            pd = reducer_data.position_depth[contact_id]
            depth = pd[3]
            if depth >= 0.0:
                continue

            pair = reducer_data.shape_pairs[contact_id]
            normal = decode_oct(reducer_data.normal[contact_id])
            position = wp.vec3(pd[0], pd[1], pd[2])
            area = reducer_data.contact_area[contact_id]
            force_weight = area * wp.static(pressure_func)(depth, pair[1], pressure_data)
            depth_volume = (area * (-depth)) * normal

            if phase == wp.static(PHASE_SCALE):
                # |normal| == 1, so force_weight already bounds force_weight * normal.
                _record_scale(fixed_scale, SCALE_FORCE, entry_idx, ht_capacity, force_weight)
                _record_scale(
                    fixed_scale,
                    SCALE_WEIGHTED_POS,
                    entry_idx,
                    ht_capacity,
                    force_weight * _max_abs_component(position),
                )
                _record_scale(fixed_scale, SCALE_DEPTH_VOLUME, entry_idx, ht_capacity, _max_abs_component(depth_volume))
                continue

            # One shift per scale family, not per component: see ``_fixed_shift``.
            mb = wp.static(mantissa_bits)
            shift_force = _fixed_shift(fixed_scale[wp.static(SCALE_FORCE) * ht_capacity + entry_idx], mb)
            shift_pos = _fixed_shift(fixed_scale[wp.static(SCALE_WEIGHTED_POS) * ht_capacity + entry_idx], mb)
            shift_vol = _fixed_shift(fixed_scale[wp.static(SCALE_DEPTH_VOLUME) * ht_capacity + entry_idx], mb)
            _add_fixed_vec(fixed_accum, FIXED_AGG_FORCE, entry_idx, ht_capacity, force_weight * normal, shift_force)
            _add_fixed(fixed_accum, FIXED_WEIGHT_SUM, entry_idx, ht_capacity, force_weight, shift_force)
            _add_fixed_vec(fixed_accum, FIXED_WEIGHTED_POS, entry_idx, ht_capacity, force_weight * position, shift_pos)
            _add_fixed_vec(fixed_accum, FIXED_DEPTH_VOLUME, entry_idx, ht_capacity, depth_volume, shift_vol)

    return accumulate_unreduced_aggregates_kernel


def _create_unreduced_moment_kernel(pressure_func, mantissa_bits: int):
    """Create the per-contact unreduced friction-moment accumulation kernel.

    Separate from :func:`_create_unreduced_aggregate_kernel` because the lever
    arm is measured from the center of pressure, which is only known once
    ``weight_sum`` and ``weighted_pos_sum`` have been finalized.
    """

    @wp.kernel(enable_backward=False)
    def accumulate_unreduced_moment_kernel(
        reducer_data: GlobalContactReducerData,
        pressure_data: Any,
        fixed_accum: wp.array[wp.int64],
        fixed_scale: wp.array[wp.int32],
        phase: int,
        total_num_threads: int,
    ):
        tid = wp.tid()
        ht_capacity = reducer_data.ht_capacity
        num_contacts = wp.min(reducer_data.contact_count[0], reducer_data.capacity)

        for buffer_idx in range(tid, num_contacts, total_num_threads):
            contact_id = buffer_idx + 1
            entry_idx = reducer_data.contact_nbin_entry[contact_id]
            if entry_idx < 0:
                continue
            pd = reducer_data.position_depth[contact_id]
            depth = pd[3]
            if depth >= 0.0:
                continue
            weight_sum = reducer_data.weight_sum[entry_idx]
            if weight_sum <= wp.static(EPS_SMALL):
                continue

            pair = reducer_data.shape_pairs[contact_id]
            normal = decode_oct(reducer_data.normal[contact_id])
            position = wp.vec3(pd[0], pd[1], pd[2])
            anchor_pos = reducer_data.weighted_pos_sum[entry_idx] / weight_sum
            lever = wp.length(wp.cross(position - anchor_pos, normal))
            pressure = wp.static(pressure_func)(depth, pair[1], pressure_data)
            moment = reducer_data.contact_area[contact_id] * pressure * lever

            if phase == wp.static(PHASE_SCALE):
                _record_scale(fixed_scale, SCALE_MOMENT_UNREDUCED, entry_idx, ht_capacity, moment)
                continue

            shift = _fixed_shift(
                fixed_scale[wp.static(SCALE_MOMENT_UNREDUCED) * ht_capacity + entry_idx], wp.static(mantissa_bits)
            )
            _add_fixed(fixed_accum, FIXED_MOMENT_UNREDUCED, entry_idx, ht_capacity, moment, shift)

    return accumulate_unreduced_moment_kernel


def _create_finalize_fixed_kernel(mantissa_bits: int):
    """Create the kernel converting completed fixed-point sums back to float32.

    ``group`` selects which families are ready: the unreduced aggregates are
    finalized before the reduction kernel runs, the reduced depth sums before
    the moment accumulator needs them, and the moments last.
    """

    @wp.kernel(enable_backward=False)
    def finalize_fixed_aggregates_kernel(
        reducer_data: GlobalContactReducerData,
        fixed_accum: wp.array[wp.int64],
        fixed_scale: wp.array[wp.int32],
        agg_moment_unreduced: wp.array[wp.float32],
        agg_moment_reduced: wp.array[wp.float32],
        agg_moment2_reduced: wp.array[wp.float32],
        group: int,
    ):
        entry_idx = wp.tid()
        ht_capacity = reducer_data.ht_capacity
        mb = wp.static(mantissa_bits)

        if group == wp.static(FINALIZE_UNREDUCED):
            shift_force = _fixed_shift(fixed_scale[wp.static(SCALE_FORCE) * ht_capacity + entry_idx], mb)
            shift_pos = _fixed_shift(fixed_scale[wp.static(SCALE_WEIGHTED_POS) * ht_capacity + entry_idx], mb)
            shift_vol = _fixed_shift(fixed_scale[wp.static(SCALE_DEPTH_VOLUME) * ht_capacity + entry_idx], mb)
            reducer_data.agg_force[entry_idx] = _read_fixed_vec(
                fixed_accum, FIXED_AGG_FORCE, entry_idx, ht_capacity, shift_force
            )
            reducer_data.weight_sum[entry_idx] = _from_fixed_scaled(
                fixed_accum[wp.static(FIXED_WEIGHT_SUM) * ht_capacity + entry_idx], shift_force
            )
            reducer_data.weighted_pos_sum[entry_idx] = _read_fixed_vec(
                fixed_accum, FIXED_WEIGHTED_POS, entry_idx, ht_capacity, shift_pos
            )
            reducer_data.agg_depth_volume[entry_idx] = _read_fixed_vec(
                fixed_accum, FIXED_DEPTH_VOLUME, entry_idx, ht_capacity, shift_vol
            )
        elif group == wp.static(FINALIZE_REDUCED_DEPTH):
            shift_depth = _fixed_shift(fixed_scale[wp.static(SCALE_REDUCED_DEPTH) * ht_capacity + entry_idx], mb)
            reducer_data.total_depth_reduced[entry_idx] = _from_fixed_scaled(
                fixed_accum[wp.static(FIXED_TOTAL_DEPTH) * ht_capacity + entry_idx], shift_depth
            )
            reducer_data.total_normal_reduced[entry_idx] = _read_fixed_vec(
                fixed_accum, FIXED_TOTAL_NORMAL, entry_idx, ht_capacity, shift_depth
            )
        elif group == wp.static(FINALIZE_MOMENTS) and agg_moment_unreduced.shape[0] > 0:
            agg_moment_unreduced[entry_idx] = _from_fixed(
                fixed_accum[wp.static(FIXED_MOMENT_UNREDUCED) * ht_capacity + entry_idx],
                fixed_scale[wp.static(SCALE_MOMENT_UNREDUCED) * ht_capacity + entry_idx],
                mb,
            )
            agg_moment_reduced[entry_idx] = _from_fixed(
                fixed_accum[wp.static(FIXED_MOMENT_REDUCED) * ht_capacity + entry_idx],
                fixed_scale[wp.static(SCALE_MOMENT_REDUCED) * ht_capacity + entry_idx],
                mb,
            )
            agg_moment2_reduced[entry_idx] = _from_fixed(
                fixed_accum[wp.static(FIXED_MOMENT2_REDUCED) * ht_capacity + entry_idx],
                fixed_scale[wp.static(SCALE_MOMENT2_REDUCED) * ht_capacity + entry_idx],
                mb,
            )

    return finalize_fixed_aggregates_kernel


# =============================================================================
# Hydroelastic reduction kernels
# =============================================================================


def get_reduce_hydroelastic_contacts_kernel(pressure_func: Any, deterministic: bool = False):
    """Create a hydroelastic contact reduction kernel specialized to a pressure callback.

    Args:
        pressure_func: User-supplied Warp function ``(signed_depth, shape_idx, data) ->
            pressure``. Required. Used to weight the unreduced friction-moment
            accumulator by ``area * pressure_func(depth, shape_b, pressure_data)``.
        deterministic: Whether normal bins were pre-registered for deterministic
            fixed-point accumulation.

    Returns:
        A Warp kernel that registers buffered contacts in the hashtable.
    """

    if pressure_func is None:
        raise ValueError("get_reduce_hydroelastic_contacts_kernel requires a non-None pressure_func.")

    @wp.kernel(enable_backward=False)
    def reduce_hydroelastic_contacts_kernel(
        reducer_data: GlobalContactReducerData,
        shape_material_k_hydro: wp.array[wp.float32],
        pressure_data: Any,
        shape_transform: wp.array[wp.transform],
        shape_collision_aabb_lower: wp.array[wp.vec3],
        shape_collision_aabb_upper: wp.array[wp.vec3],
        shape_voxel_resolution: wp.array[wp.vec3i],
        agg_moment_unreduced: wp.array[wp.float32],
        total_num_threads: int,
    ):
        """Register hydroelastic contacts in the hashtable for reduction.

        Populates all hashtable slots (spatial extremes, max-depth, voxel) with
        real contact_ids from the buffer.
        """
        tid = wp.tid()

        num_contacts = reducer_data.contact_count[0]
        if num_contacts == 0:
            return
        num_contacts = wp.min(num_contacts, reducer_data.capacity)

        for i in range(tid, num_contacts, total_num_threads):
            contact_id = i + 1
            pd = reducer_data.position_depth[contact_id]
            normal = decode_oct(reducer_data.normal[contact_id])
            pair = reducer_data.shape_pairs[contact_id]

            position = wp.vec3(pd[0], pd[1], pd[2])
            depth = pd[3]
            shape_a = pair[0]
            shape_b = pair[1]

            aabb_lower = shape_collision_aabb_lower[shape_b]
            aabb_upper = shape_collision_aabb_upper[shape_b]

            ht_capacity = reducer_data.ht_capacity

            # === Part 1: Normal-binned reduction ===
            bin_id = get_slot(normal)
            key = make_contact_key(shape_a, shape_b, bin_id)

            entry_idx = int(-1)
            if wp.static(deterministic):
                # Deterministic aggregate accumulation pre-registers normal
                # bins. Reuse that result so a failed registration is counted
                # once rather than again in this kernel.
                entry_idx = reducer_data.contact_nbin_entry[contact_id]
            else:
                entry_idx = hashtable_find_or_insert(key, reducer_data.ht_keys, reducer_data.ht_active_slots)

            # Cache normal-bin entry index for downstream kernels (avoids repeated hash lookups)
            if reducer_data.contact_nbin_entry.shape[0] > 0:
                reducer_data.contact_nbin_entry[contact_id] = entry_idx

            if entry_idx >= 0:
                # k_eff is constant for a shape pair, so redundant writes are safe.
                reducer_data.entry_k_eff[entry_idx] = _effective_stiffness(
                    shape_material_k_hydro[shape_a], shape_material_k_hydro[shape_b]
                )
                aabb_size = wp.length(aabb_upper - aabb_lower)
                use_beta = depth < wp.static(BETA_THRESHOLD) * aabb_size
                if use_beta:
                    ws = reducer_data.weight_sum[entry_idx]
                    anchor = reducer_data.weighted_pos_sum[entry_idx] / ws
                    pos_2d_centered = project_point_to_plane(bin_id, position - anchor)
                    pen_weight = wp.max(-depth, 0.0)
                    for dir_i in range(wp.static(NUM_SPATIAL_DIRECTIONS)):
                        dir_2d = get_spatial_direction_2d(dir_i)
                        score = wp.dot(pos_2d_centered, dir_2d) * pen_weight
                        value = make_contact_value(
                            score,
                            reducer_data.contact_fingerprints[contact_id],
                            contact_id,
                            reducer_data.deterministic,
                        )
                        reduction_update_slot(entry_idx, dir_i, value, reducer_data.ht_values, ht_capacity)

                max_depth_value = make_contact_value(
                    -depth,
                    reducer_data.contact_fingerprints[contact_id],
                    contact_id,
                    reducer_data.deterministic,
                )
                reduction_update_slot(
                    entry_idx,
                    wp.static(NUM_SPATIAL_DIRECTIONS),
                    max_depth_value,
                    reducer_data.ht_values,
                    ht_capacity,
                )

                if agg_moment_unreduced.shape[0] > 0 and depth < 0.0 and reducer_data.deterministic == 0:
                    ws = reducer_data.weight_sum[entry_idx]
                    if ws > EPS_SMALL:
                        anchor_pos = reducer_data.weighted_pos_sum[entry_idx] / ws
                        lever = wp.length(wp.cross(position - anchor_pos, normal))
                        # Force weight = area * pressure_func(depth). Recomputed
                        # from buffer state (depth + shape_b) rather than cached.
                        # Previously this used area * |depth|, which implicitly
                        # assumed the linear law p = -kh * depth.
                        area_i = reducer_data.contact_area[contact_id]
                        p_i = wp.static(pressure_func)(depth, shape_b, pressure_data)
                        wp.atomic_add(agg_moment_unreduced, entry_idx, area_i * p_i * lever)
            elif not wp.static(deterministic):
                wp.atomic_add(reducer_data.ht_insert_failures, 0, 1)

            # === Part 2: Voxel-based reduction ===
            voxel_res = shape_voxel_resolution[shape_b]
            voxel_idx = compute_voxel_index(position, aabb_lower, aabb_upper, voxel_res)
            voxel_idx = wp.clamp(voxel_idx, 0, wp.static(NUM_VOXEL_DEPTH_SLOTS - 1))

            voxels_per_group = wp.static(NUM_SPATIAL_DIRECTIONS + 1)
            voxel_group = voxel_idx // voxels_per_group
            voxel_local_slot = voxel_idx % voxels_per_group

            voxel_bin_id = wp.static(NUM_NORMAL_BINS) + voxel_group
            voxel_key = make_contact_key(shape_a, shape_b, voxel_bin_id)

            voxel_entry_idx = hashtable_find_or_insert(voxel_key, reducer_data.ht_keys, reducer_data.ht_active_slots)
            if voxel_entry_idx >= 0:
                reducer_data.entry_k_eff[voxel_entry_idx] = _effective_stiffness(
                    shape_material_k_hydro[shape_a], shape_material_k_hydro[shape_b]
                )
                voxel_value = make_contact_value(
                    -depth,
                    reducer_data.contact_fingerprints[contact_id],
                    contact_id,
                    reducer_data.deterministic,
                )
                reduction_update_slot(
                    voxel_entry_idx,
                    voxel_local_slot,
                    voxel_value,
                    reducer_data.ht_values,
                    ht_capacity,
                )
            else:
                wp.atomic_add(reducer_data.ht_insert_failures, 0, 1)

    return reduce_hydroelastic_contacts_kernel


# =============================================================================
# Hydroelastic export kernel factory
# =============================================================================


def _create_accumulate_reduced_depth_kernel(deterministic: bool = False, mantissa_bits: int = 48):
    """Create a kernel that accumulates winning contact depths and normals per normal bin.

    Args:
        deterministic: If True, unpack the deterministic contact-id encoding and
            accumulate in int64 fixed point instead of floating point.
        mantissa_bits: Fixed-point mantissa width from :func:`_fixed_mantissa_bits`.

    Returns:
        A Warp kernel that accumulates ``total_depth_reduced`` and
        ``total_normal_reduced``.
    """
    exported_ids_vec = wp.types.vector(length=VALUES_PER_KEY, dtype=wp.int32)

    @wp.kernel(enable_backward=False)
    def accumulate_reduced_depth_kernel(
        ht_keys: wp.array[wp.uint64],
        ht_values: wp.array[wp.uint64],
        ht_active_slots: wp.array[wp.int32],
        position_depth: wp.array[wp.vec4],
        normal: wp.array[wp.vec2],
        contact_nbin_entry: wp.array[wp.int32],
        total_depth_reduced: wp.array[wp.float32],
        total_normal_reduced: wp.array[wp.vec3],
        fixed_accum: wp.array[wp.int64],
        fixed_scale: wp.array[wp.int32],
        phase: int,
        total_num_threads: int,
    ):
        """Accumulate winning contact depths and normals per normal bin.

        For each active hashtable entry (normal bin or voxel bin), iterates
        over unique winning contacts and atomically adds their penetrating
        depths to the corresponding normal bin's ``total_depth_reduced`` and
        their depth-weighted normals to ``total_normal_reduced``.
        """
        tid = wp.tid()
        ht_capacity = ht_keys.shape[0]
        num_active = ht_active_slots[ht_capacity]
        if num_active == 0:
            return

        for i in range(tid, num_active, total_num_threads):
            entry_idx = ht_active_slots[i]

            # Extract bin_id from the stored key to distinguish normal vs voxel bins.
            stored_key = ht_keys[entry_idx]
            entry_bin_id = int((stored_key >> wp.uint64(55)) & BIN_MASK)

            p1_ids = exported_ids_vec()
            p1_count = int(0)

            for slot in range(wp.static(VALUES_PER_KEY)):
                value = ht_values[slot * ht_capacity + entry_idx]
                if value == wp.uint64(0):
                    continue
                contact_id = wp.static(_unpack_contact_id_variant(deterministic))(value)
                if is_contact_already_exported(contact_id, p1_ids, p1_count):
                    continue
                p1_ids[p1_count] = contact_id
                p1_count = p1_count + 1

                pd = position_depth[contact_id]
                depth = pd[3]
                if depth < 0.0:
                    if entry_bin_id < wp.static(NUM_NORMAL_BINS):
                        nbin_idx = entry_idx
                    else:
                        nbin_idx = contact_nbin_entry[contact_id]
                    if nbin_idx >= 0:
                        pen_mag = -depth
                        contact_normal = decode_oct(normal[contact_id])
                        if wp.static(deterministic):
                            if phase == wp.static(PHASE_SCALE):
                                _record_scale(fixed_scale, SCALE_REDUCED_DEPTH, nbin_idx, ht_capacity, pen_mag)
                            else:
                                shift = _fixed_shift(
                                    fixed_scale[wp.static(SCALE_REDUCED_DEPTH) * ht_capacity + nbin_idx],
                                    wp.static(mantissa_bits),
                                )
                                _add_fixed(fixed_accum, FIXED_TOTAL_DEPTH, nbin_idx, ht_capacity, pen_mag, shift)
                                _add_fixed_vec(
                                    fixed_accum,
                                    FIXED_TOTAL_NORMAL,
                                    nbin_idx,
                                    ht_capacity,
                                    pen_mag * contact_normal,
                                    shift,
                                )
                        else:
                            wp.atomic_add(total_depth_reduced, nbin_idx, pen_mag)
                            wp.atomic_add(total_normal_reduced, nbin_idx, pen_mag * contact_normal)

    return accumulate_reduced_depth_kernel


def _create_accumulate_moments_kernel(
    normal_matching: bool = True, deterministic: bool = False, mantissa_bits: int = 48
):
    """Create a kernel that accumulates unreduced and reduced friction moments per normal bin.

    Args:
        normal_matching: If True, rotate reduced contact normals using the aggregate
            force direction before computing lever arms.
        deterministic: If True, unpack the deterministic contact-id encoding and
            accumulate in int64 fixed point instead of floating point.
        mantissa_bits: Fixed-point mantissa width from :func:`_fixed_mantissa_bits`.

    Returns:
        A Warp kernel that populates ``agg_moment_unreduced``,
        ``agg_moment_reduced``, and ``agg_moment2_reduced``.
    """
    exported_ids_vec = wp.types.vector(length=VALUES_PER_KEY, dtype=wp.int32)

    @wp.kernel(enable_backward=False)
    def accumulate_moments_kernel(
        ht_keys: wp.array[wp.uint64],
        ht_values: wp.array[wp.uint64],
        ht_active_slots: wp.array[wp.int32],
        position_depth: wp.array[wp.vec4],
        normal: wp.array[wp.vec2],
        contact_nbin_entry: wp.array[wp.int32],
        weighted_pos_sum: wp.array[wp.vec3],
        weight_sum: wp.array[wp.float32],
        agg_force: wp.array[wp.vec3],
        agg_depth_volume: wp.array[wp.vec3],
        total_normal_reduced: wp.array[wp.vec3],
        agg_moment_reduced: wp.array[wp.float32],
        agg_moment2_reduced: wp.array[wp.float32],
        fixed_accum: wp.array[wp.int64],
        fixed_scale: wp.array[wp.int32],
        phase: int,
        total_num_threads: int,
    ):
        """Accumulate reduced friction moments per normal bin."""
        tid = wp.tid()
        ht_capacity = ht_keys.shape[0]

        # Reduced moment over winning contacts
        num_active = ht_active_slots[ht_capacity]
        if num_active == 0:
            return

        for i in range(tid, num_active, total_num_threads):
            entry_idx = ht_active_slots[i]

            stored_key = ht_keys[entry_idx]
            entry_bin_id = int((stored_key >> wp.uint64(55)) & BIN_MASK)

            p2_ids = exported_ids_vec()
            p2_count = int(0)

            for slot in range(wp.static(VALUES_PER_KEY)):
                value = ht_values[slot * ht_capacity + entry_idx]
                if value == wp.uint64(0):
                    continue
                contact_id = wp.static(_unpack_contact_id_variant(deterministic))(value)
                if is_contact_already_exported(contact_id, p2_ids, p2_count):
                    continue
                p2_ids[p2_count] = contact_id
                p2_count = p2_count + 1

                pd = position_depth[contact_id]
                depth = pd[3]
                if depth >= 0.0:
                    continue
                pen_mag = -depth
                contact_normal = decode_oct(normal[contact_id])

                # Determine normal-bin index using cached entry
                if entry_bin_id < wp.static(NUM_NORMAL_BINS):
                    nbin_idx = entry_idx
                else:
                    nbin_idx = contact_nbin_entry[contact_id]
                if nbin_idx < 0:
                    continue

                ws = weight_sum[nbin_idx]
                if ws <= EPS_SMALL:
                    continue
                anchor_pos = weighted_pos_sum[nbin_idx] / ws

                # Optionally rotate normal to match aggregate force direction
                rotated_normal = contact_normal
                if wp.static(normal_matching):
                    nbin_agg_force = agg_force[nbin_idx]
                    nbin_agg_mag = wp.length(nbin_agg_force)
                    # Same reliability gate as the export kernel.
                    nbin_dv_mag = wp.length(agg_depth_volume[nbin_idx])
                    if nbin_dv_mag > EPS_LARGE and nbin_agg_mag > EPS_SMALL:
                        nbin_nsum = total_normal_reduced[nbin_idx]
                        rot_q = _compute_normal_matching_rotation(nbin_nsum, nbin_agg_force, nbin_agg_mag)
                        rotated_normal = wp.normalize(wp.quat_rotate(rot_q, contact_normal))

                pos = wp.vec3(pd[0], pd[1], pd[2])
                lever = wp.length(wp.cross(pos - anchor_pos, rotated_normal))
                if wp.static(deterministic):
                    if phase == wp.static(PHASE_SCALE):
                        _record_scale(fixed_scale, SCALE_MOMENT_REDUCED, nbin_idx, ht_capacity, pen_mag * lever)
                        _record_scale(
                            fixed_scale, SCALE_MOMENT2_REDUCED, nbin_idx, ht_capacity, pen_mag * lever * lever
                        )
                    else:
                        mb = wp.static(mantissa_bits)
                        _add_fixed(
                            fixed_accum,
                            FIXED_MOMENT_REDUCED,
                            nbin_idx,
                            ht_capacity,
                            pen_mag * lever,
                            _fixed_shift(fixed_scale[wp.static(SCALE_MOMENT_REDUCED) * ht_capacity + nbin_idx], mb),
                        )
                        _add_fixed(
                            fixed_accum,
                            FIXED_MOMENT2_REDUCED,
                            nbin_idx,
                            ht_capacity,
                            pen_mag * lever * lever,
                            _fixed_shift(fixed_scale[wp.static(SCALE_MOMENT2_REDUCED) * ht_capacity + nbin_idx], mb),
                        )
                else:
                    wp.atomic_add(agg_moment_reduced, nbin_idx, pen_mag * lever)
                    wp.atomic_add(agg_moment2_reduced, nbin_idx, pen_mag * lever * lever)  # second moment

    return accumulate_moments_kernel


def create_export_hydroelastic_reduced_contacts_kernel(
    writer_func: Any,
    margin_contact_area: float,
    normal_matching: bool = True,
    anchor_contact: bool = False,
    moment_matching: bool = False,
    pressure_func: Any = None,
    deterministic_sort_keys: bool = False,
):
    """Create a kernel that exports reduced hydroelastic contacts using a custom writer function.

    Computes contact stiffness using the aggregate stiffness formula:
        c_stiffness = |agg_force| / total_depth_reduced

    where:
    - agg_force = sum(area * pressure_func(depth) * normal) for ALL contacts in the normal bin
    - total_depth_reduced = sum(|depth|) for all winning contacts (normal bin + voxel)
      that map to the normal bin, pre-accumulated by ``accumulate_reduced_depth_kernel``

    This ensures the total contact force from the K reduced contacts equals the
    aggregate force from all original contacts under any user-supplied
    ``pressure_func``. Margin (non-penetrating) contact stiffness still derives
    from the per-pair linear-law harmonic mean stored in ``entry_k_eff`` —
    margin behavior is a constraint regularization, not a physical pressure.

    .. important::

       ``accumulate_reduced_depth_kernel`` (from
       :func:`_create_accumulate_reduced_depth_kernel`) **must** be launched
       before this kernel so that ``total_depth_reduced`` is fully populated.

    Args:
        writer_func: A warp function with signature (ContactData, writer_data, int) -> None
        margin_contact_area: Contact area to use for non-penetrating contacts at the margin
        normal_matching: If True, rotate contact normals so their weighted sum aligns with aggregate force
        anchor_contact: If True, add an anchor contact at the center of pressure for each entry
        moment_matching: If True, adjust per-contact friction scales so that
            the maximum friction moment per normal bin is preserved between
            reduced and unreduced contacts.
        deterministic_sort_keys: Whether to tag normal-bin and voxel-bin
            exports so deterministic contact sort keys remain unique.

    Returns:
        A warp kernel that can be launched to export reduced hydroelastic contacts.
    """
    if pressure_func is None:
        raise ValueError("create_export_hydroelastic_reduced_contacts_kernel requires a non-None pressure_func.")

    # Define vector types for tracking exported contact data
    exported_ids_vec = wp.types.vector(length=VALUES_PER_KEY, dtype=wp.int32)
    exported_depths_vec = wp.types.vector(length=VALUES_PER_KEY, dtype=wp.float32)
    # Cache decoded normals (vec3 per slot) to avoid double decode_oct
    # Stored as 3 separate float vectors (Warp doesn't support vector-of-vec3)
    exported_nx_vec = wp.types.vector(length=VALUES_PER_KEY, dtype=wp.float32)
    exported_ny_vec = wp.types.vector(length=VALUES_PER_KEY, dtype=wp.float32)
    exported_nz_vec = wp.types.vector(length=VALUES_PER_KEY, dtype=wp.float32)

    @wp.kernel(enable_backward=False)
    def export_hydroelastic_reduced_contacts_kernel(
        # Hashtable arrays
        ht_keys: wp.array[wp.uint64],
        ht_values: wp.array[wp.uint64],
        ht_active_slots: wp.array[wp.int32],
        # Aggregate data per entry (from generate kernel)
        agg_force: wp.array[wp.vec3],
        agg_depth_volume: wp.array[wp.vec3],
        weighted_pos_sum: wp.array[wp.vec3],
        weight_sum: wp.array[wp.float32],
        # Contact buffer arrays
        position_depth: wp.array[wp.vec4],
        normal: wp.array[wp.vec2],  # Octahedral-encoded
        shape_pairs: wp.array[wp.vec2i],
        contact_fingerprints: wp.array[wp.int32],
        contact_area: wp.array[wp.float32],
        entry_k_eff: wp.array[wp.float32],
        contact_nbin_entry: wp.array[wp.int32],
        # Pre-accumulated total depth of winning contacts per normal bin
        total_depth_reduced: wp.array[wp.float32],
        # Pre-accumulated depth-weighted normal sum of winning contacts per normal bin
        total_normal_reduced: wp.array[wp.vec3],
        # Pre-accumulated friction moments per normal bin (for moment matching)
        agg_moment_unreduced: wp.array[wp.float32],
        agg_moment_reduced: wp.array[wp.float32],
        agg_moment2_reduced: wp.array[wp.float32],
        # Shape data for margin
        shape_gap: wp.array[float],
        shape_transform: wp.array[wp.transform],
        # User pressure-callback state (recomputed on demand from depth+shape_b)
        pressure_data: Any,
        # Writer data (custom struct)
        writer_data: Any,
        deterministic: int,
        # Grid stride parameters
        total_num_threads: int,
    ):
        """
        Export reduced hydroelastic contacts to the writer with aggregate stiffness.
        """
        tid = wp.tid()

        # Get number of active entries (stored at index = ht_capacity)
        ht_capacity = ht_keys.shape[0]
        num_active = ht_active_slots[ht_capacity]

        # Early exit if no active entries
        if num_active == 0:
            return

        for i in range(tid, num_active, total_num_threads):
            # Get the hashtable entry index
            entry_idx = ht_active_slots[i]
            is_voxel_entry = False
            if wp.static(deterministic_sort_keys):
                bin_id = int((ht_keys[entry_idx] >> wp.uint64(55)) & wp.uint64(0xFF))
                is_voxel_entry = bin_id >= wp.static(NUM_NORMAL_BINS)

            # === First pass: collect unique contacts and compute aggregates ===
            exported_ids = exported_ids_vec()
            exported_depths = exported_depths_vec()
            # Cache decoded normals to avoid double decode_oct in second pass
            cached_nx = exported_nx_vec()
            cached_ny = exported_ny_vec()
            cached_nz = exported_nz_vec()
            num_exported = int(0)
            max_pen_depth = float(0.0)  # Maximum penetration magnitude (positive value)
            k_eff_first = float(0.0)
            shape_a_first = int(0)
            shape_b_first = int(0)

            # Read all value slots for this entry (slot-major layout)
            for slot in range(wp.static(VALUES_PER_KEY)):
                value = ht_values[slot * ht_capacity + entry_idx]

                # Skip empty slots (value = 0)
                if value == wp.uint64(0):
                    continue

                # Extract contact ID from low 32 bits
                contact_id = unpack_contact_id(value, deterministic)

                # Skip if already exported
                if is_contact_already_exported(contact_id, exported_ids, num_exported):
                    continue

                # Unpack contact data (decode oct-normal once, cache for second pass)
                pd = position_depth[contact_id]
                contact_normal = decode_oct(normal[contact_id])
                depth = pd[3]

                # Record this contact, its depth, and cached normal
                exported_ids[num_exported] = contact_id
                exported_depths[num_exported] = depth
                cached_nx[num_exported] = contact_normal[0]
                cached_ny[num_exported] = contact_normal[1]
                cached_nz[num_exported] = contact_normal[2]
                num_exported = num_exported + 1

                # Track max penetration and normal matching (depth < 0 = penetrating)
                if depth < 0.0:
                    pen_magnitude = -depth
                    max_pen_depth = wp.max(max_pen_depth, pen_magnitude)

                # Store first contact's shape pair (same for all contacts in the entry)
                if k_eff_first == 0.0:
                    k_eff_first = entry_k_eff[entry_idx]
                    pair = shape_pairs[contact_id]
                    shape_a_first = pair[0]
                    shape_b_first = pair[1]

            # Skip entries with no contacts
            if num_exported == 0:
                continue

            # === Compute stiffness and optional features based on entry type ===
            # Normal bin entries (bin_id < NUM_NORMAL_BINS): have aggregate force, use aggregate stiffness
            # Voxel bin entries (bin_id >= NUM_NORMAL_BINS): no aggregate force, use per-contact stiffness
            agg_force_vec = agg_force[entry_idx]
            agg_force_mag = wp.length(agg_force_vec)

            # Reliability gate for normal matching / anchor placement. The geometric
            # depth-volume is pressure-law-independent (= |agg_force| / kh for the
            # linear law); the EPS_SMALL term keeps agg_force_vec safe to normalize
            # for the direction even under a degenerate custom pressure law.
            agg_direction_mag = wp.length(agg_depth_volume[entry_idx])
            has_reliable_agg_direction = agg_direction_mag > wp.static(EPS_LARGE) and agg_force_mag > wp.static(
                EPS_SMALL
            )

            # Compute anchor position (center of pressure) for normal bin entries
            anchor_pos = wp.vec3(0.0, 0.0, 0.0)
            add_anchor = 0
            entry_weight_sum = weight_sum[entry_idx]
            if wp.static(anchor_contact) and has_reliable_agg_direction and max_pen_depth > 0.0:
                if entry_weight_sum > wp.static(EPS_SMALL):
                    anchor_pos = weighted_pos_sum[entry_idx] / entry_weight_sum
                    add_anchor = 1

            # Compute total_depth including anchor contribution
            # Use pre-accumulated total_depth_reduced which includes all winning contacts
            # (both normal bin and voxel bin) that map to this normal bin.
            anchor_depth = max_pen_depth  # Anchor uses max penetration depth (positive magnitude)
            entry_total_depth = total_depth_reduced[entry_idx]

            # Compute normal matching rotation quaternion from pre-accumulated
            rotation_q = wp.quat_identity()
            nbin_normal_sum = total_normal_reduced[entry_idx]
            if wp.static(normal_matching) and has_reliable_agg_direction:
                rotation_q = _compute_normal_matching_rotation(nbin_normal_sum, agg_force_vec, agg_force_mag)

            # When normal matching is enabled, use |total_normal_reduced| as the
            # effective depth denominator.  This compensates for the magnitude loss
            # caused by cancellation in the depth-weighted normal sum so that the
            # K reduced contacts together reproduce ``agg_force`` exactly.
            if wp.static(normal_matching):
                effective_depth = wp.length(nbin_normal_sum)
                if effective_depth < wp.static(EPS_LARGE):
                    effective_depth = entry_total_depth
            else:
                effective_depth = entry_total_depth
            total_depth_with_anchor = effective_depth + wp.float32(add_anchor) * anchor_depth

            # Compute shared stiffness so the K reduced contacts reproduce
            # ``agg_force_mag`` exactly. The solver applies
            # ``F = c_stiffness * (-contact_distance) = c_stiffness * 2*|depth|``
            # per reduced contact; summing across reduced contacts gives
            #     sum_F_red = shared_stiffness * 2 * sum(|d_red|)
            #               = shared_stiffness * 2 * total_depth_with_anchor.
            # Setting shared_stiffness = agg_force_mag / (2 * total_depth_with_anchor)
            # makes that sum match agg_force_mag. ``agg_force`` is accumulated
            # as ``area * pressure_func(d) * normal`` in the generate kernel
            # so it is already in physical force units (no ``k_eff_first``
            # factor — that double-counts under a non-linear pressure law).
            shared_stiffness = float(0.0)
            if agg_force_mag > wp.static(EPS_SMALL) and total_depth_with_anchor > 0.0:
                shared_stiffness = agg_force_mag / (2.0 * total_depth_with_anchor)

            # Moment matching: hybrid uniform / per-contact strategy.
            moment_alpha = float(0.0)
            moment_L_avg = float(0.0)
            uniform_friction_scale = float(1.0)
            anchor_friction_scale = float(1.0)
            if wp.static(moment_matching):
                m_unr = agg_moment_unreduced[entry_idx]
                m_red = agg_moment_reduced[entry_idx]  # S1 = sum(pen * lever)
                m_red2 = agg_moment2_reduced[entry_idx]  # S2 = sum(pen * lever^2)
                s0_total = entry_total_depth + wp.float32(add_anchor) * anchor_depth
                if (
                    m_unr > wp.static(EPS_SMALL)
                    and s0_total > wp.static(EPS_SMALL)
                    and m_red > wp.static(EPS_SMALL)
                    and agg_force_mag > wp.static(EPS_SMALL)
                ):
                    m_target = m_unr * total_depth_with_anchor / agg_force_mag
                    if m_target < m_red:
                        # Overshoot: uniform scale down
                        uniform_friction_scale = m_target / m_red
                    else:
                        # Undershoot: per-contact alpha scaling
                        moment_L_avg = m_red / s0_total
                        variance = m_red2 * s0_total - m_red * m_red
                        if variance > wp.static(EPS_SMALL):
                            moment_alpha = wp.clamp((m_target - m_red) * m_red / variance, 0.0, 1.0)
                # Anchor compensation:
                #  - Overshoot: anchor_fs = 1 + (S0/anchor_depth)*(1 - uniform_fs)
                #  - Undershoot: anchor_fs = 1 - alpha
                if add_anchor == 1 and anchor_depth > 0.0:
                    anchor_friction_scale = wp.max(
                        wp.static(MIN_FRICTION_SCALE),
                        1.0 + (entry_total_depth / anchor_depth) * (1.0 - uniform_friction_scale) - moment_alpha,
                    )

            # Get transform and gap sum (same for all contacts in the entry)
            transform_b = shape_transform[shape_b_first]
            gap_a = shape_gap[shape_a_first]
            gap_b = shape_gap[shape_b_first]
            gap_sum = gap_a + gap_b

            # === Second pass: export contacts ===
            for idx in range(num_exported):
                contact_id = exported_ids[idx]
                depth = exported_depths[idx]

                # Read position from buffer; use cached decoded normal
                pd = position_depth[contact_id]
                position = wp.vec3(pd[0], pd[1], pd[2])
                contact_normal = wp.vec3(cached_nx[idx], cached_ny[idx], cached_nz[idx])

                # Get shape pair
                pair = shape_pairs[contact_id]
                shape_a = pair[0]
                shape_b = pair[1]

                # Apply normal matching rotation for penetrating contacts (depth < 0)
                final_normal = contact_normal
                area_i = contact_area[contact_id]

                c_friction_scale = float(1.0)

                if has_reliable_agg_direction:
                    # --- Normal-bin entry ---
                    if wp.static(normal_matching) and depth < 0.0:
                        final_normal = wp.normalize(wp.quat_rotate(rotation_q, contact_normal))
                    c_stiffness = shared_stiffness
                    if shared_stiffness == 0.0:
                        # Normal-bin entry but aggregate stiffness unavailable.
                        # Penetrating: pick c_stiffness so F = c_stiffness*(-d)
                        # equals area * pressure_func(d). Margin: regularization
                        # stays on the linear law (entry_k_eff = harmonic mean).
                        if depth < 0.0:
                            p_i = wp.static(pressure_func)(depth, shape_b, pressure_data)
                            c_stiffness = area_i * p_i / (2.0 * wp.max(-depth, wp.static(EPS_SMALL)))
                        else:
                            c_stiffness = wp.static(margin_contact_area) * k_eff_first

                    # Moment matching friction adjustment
                    if wp.static(moment_matching) and depth < 0.0:
                        if moment_L_avg > wp.static(EPS_SMALL):
                            # Undershoot: per-contact scaling
                            lever_i = wp.length(wp.cross(position - anchor_pos, final_normal))
                            c_friction_scale = wp.max(
                                wp.static(MIN_FRICTION_SCALE),
                                1.0 + moment_alpha * (lever_i - moment_L_avg) / moment_L_avg,
                            )
                        else:
                            # Overshoot: uniform scaling
                            c_friction_scale = uniform_friction_scale
                else:
                    # --- Voxel-bin entry: use cached normal-bin index ---
                    nbin_entry_idx = contact_nbin_entry[contact_id]

                    if nbin_entry_idx >= 0 and depth < 0.0:
                        nbin_agg_force = agg_force[nbin_entry_idx]
                        nbin_agg_mag = wp.length(nbin_agg_force)
                        # Same reliability gate as the aggregate path above.
                        nbin_direction_mag = wp.length(agg_depth_volume[nbin_entry_idx])
                        nbin_dir_reliable = nbin_direction_mag > wp.static(EPS_LARGE) and nbin_agg_mag > wp.static(
                            EPS_SMALL
                        )

                        # Normal matching from the normal bin's rotation
                        if wp.static(normal_matching) and nbin_dir_reliable:
                            voxel_nsum = total_normal_reduced[nbin_entry_idx]
                            voxel_rot_q = _compute_normal_matching_rotation(voxel_nsum, nbin_agg_force, nbin_agg_mag)
                            final_normal = wp.normalize(wp.quat_rotate(voxel_rot_q, contact_normal))

                        # Stiffness from the normal bin's aggregate
                        if wp.static(normal_matching):
                            nbin_effective_depth_no_anchor = wp.length(total_normal_reduced[nbin_entry_idx])
                            if nbin_effective_depth_no_anchor < wp.static(EPS_LARGE):
                                nbin_effective_depth_no_anchor = total_depth_reduced[nbin_entry_idx]
                        else:
                            nbin_effective_depth_no_anchor = total_depth_reduced[nbin_entry_idx]
                        nbin_effective_depth = nbin_effective_depth_no_anchor
                        nbin_anchor_depth = float(0.0)
                        if wp.static(anchor_contact) and nbin_dir_reliable:
                            nbin_max_depth_value = ht_values[
                                wp.static(NUM_SPATIAL_DIRECTIONS) * ht_capacity + nbin_entry_idx
                            ]
                            if nbin_max_depth_value != wp.uint64(0):
                                nbin_max_depth_contact_id = unpack_contact_id(nbin_max_depth_value, deterministic)
                                nbin_max_depth = position_depth[nbin_max_depth_contact_id][3]
                                if nbin_max_depth < 0.0:
                                    nbin_anchor_depth = -nbin_max_depth

                            if weight_sum[nbin_entry_idx] > wp.static(EPS_SMALL) and nbin_anchor_depth > 0.0:
                                nbin_effective_depth = nbin_effective_depth_no_anchor + nbin_anchor_depth

                        if nbin_agg_mag > wp.static(EPS_SMALL) and nbin_effective_depth > 0.0:
                            # Same physical-force argument as the normal-bin path:
                            # ``nbin_agg_mag`` is in force units; the solver
                            # multiplies c_stiffness by 2*|depth|, so divide by
                            # 2*nbin_effective_depth to recover total force.
                            c_stiffness = nbin_agg_mag / (2.0 * nbin_effective_depth)
                        else:
                            p_i = wp.static(pressure_func)(depth, shape_b, pressure_data)
                            c_stiffness = area_i * p_i / (2.0 * wp.max(-depth, wp.static(EPS_SMALL)))

                        # Moment matching friction adjustment (voxel entry)
                        if wp.static(moment_matching):
                            voxel_m_unr = agg_moment_unreduced[nbin_entry_idx]
                            voxel_s1 = agg_moment_reduced[nbin_entry_idx]
                            voxel_s2 = agg_moment2_reduced[nbin_entry_idx]
                            nbin_entry_total_depth = total_depth_reduced[nbin_entry_idx]
                            voxel_s0 = nbin_entry_total_depth + nbin_anchor_depth
                            if (
                                voxel_m_unr > wp.static(EPS_SMALL)
                                and voxel_s0 > wp.static(EPS_SMALL)
                                and voxel_s1 > wp.static(EPS_SMALL)
                                and nbin_agg_mag > wp.static(EPS_SMALL)
                            ):
                                voxel_m_target = voxel_m_unr * nbin_effective_depth / nbin_agg_mag
                                if voxel_m_target < voxel_s1:
                                    # Overshoot: uniform scale down
                                    c_friction_scale = voxel_m_target / voxel_s1
                                else:
                                    # Undershoot: per-contact alpha scaling
                                    voxel_L_avg = voxel_s1 / voxel_s0
                                    voxel_variance = voxel_s2 * voxel_s0 - voxel_s1 * voxel_s1
                                    voxel_alpha = float(0.0)
                                    if voxel_variance > wp.static(EPS_SMALL):
                                        voxel_alpha = wp.clamp(
                                            (voxel_m_target - voxel_s1) * voxel_s1 / voxel_variance, 0.0, 1.0
                                        )
                                    voxel_anchor_pos = wp.vec3(0.0, 0.0, 0.0)
                                    nbin_ws = weight_sum[nbin_entry_idx]
                                    if nbin_ws > wp.static(EPS_SMALL):
                                        voxel_anchor_pos = weighted_pos_sum[nbin_entry_idx] / nbin_ws
                                    voxel_lever = wp.length(wp.cross(position - voxel_anchor_pos, final_normal))
                                    if voxel_L_avg > wp.static(EPS_SMALL):
                                        c_friction_scale = wp.max(
                                            wp.static(MIN_FRICTION_SCALE),
                                            1.0 + voxel_alpha * (voxel_lever - voxel_L_avg) / voxel_L_avg,
                                        )
                    elif depth < 0.0:
                        # Penetrating contact with no normal bin: per-contact
                        # secant from the user pressure law. F_face = area * p
                        # via solver multiplying by 2*|depth| (contact_distance
                        # = 2*depth), so c_stiffness = area * p / (2*|d|).
                        p_i = wp.static(pressure_func)(depth, shape_b, pressure_data)
                        c_stiffness = area_i * p_i / (2.0 * wp.max(-depth, wp.static(EPS_SMALL)))
                    else:
                        # Non-penetrating margin contact: linear-law regularization.
                        c_stiffness = wp.static(margin_contact_area) * k_eff_first

                # Transform contact to world space
                normal_world = wp.transform_vector(transform_b, final_normal)
                pos_world = wp.transform_point(transform_b, position)

                # Create ContactData struct
                # contact_distance = 2 * depth (depth is already negative for penetrating)
                # This gives negative contact_distance for penetrating contacts
                contact_data = ContactData()
                contact_data.contact_point_center = pos_world
                contact_data.contact_normal_a_to_b = normal_world
                contact_data.contact_distance = 2.0 * depth  # depth is negative = penetrating
                contact_data.radius_eff_a = 0.0
                contact_data.radius_eff_b = 0.0
                contact_data.margin_a = 0.0
                contact_data.margin_b = 0.0
                contact_data.shape_a = shape_a
                contact_data.shape_b = shape_b
                contact_data.gap_sum = gap_sum
                contact_data.contact_stiffness = c_stiffness
                contact_data.contact_friction_scale = wp.float32(c_friction_scale)
                if wp.static(deterministic_sort_keys):
                    # Both copies contribute to the force-preserving denominator
                    # when a face wins a normal and voxel entry. Tag their sources
                    # so their exported sort keys remain unique.
                    contact_data.sort_sub_key = (contact_fingerprints[contact_id] << 1) | int(is_voxel_entry)
                else:
                    contact_data.sort_sub_key = contact_fingerprints[contact_id]

                # Call the writer function
                writer_func(contact_data, writer_data, -1)

            # === Export anchor contact if enabled ===
            if add_anchor == 1:
                # Anchor normal is aligned with aggregate force direction
                anchor_normal = wp.normalize(agg_force_vec)
                anchor_normal_world = wp.transform_vector(transform_b, anchor_normal)
                anchor_pos_world = wp.transform_point(transform_b, anchor_pos)

                # Create ContactData for anchor
                # anchor_depth is positive magnitude, so negate for standard convention
                contact_data = ContactData()
                contact_data.contact_point_center = anchor_pos_world
                contact_data.contact_normal_a_to_b = anchor_normal_world
                contact_data.contact_distance = -2.0 * anchor_depth  # anchor_depth is positive magnitude
                contact_data.radius_eff_a = 0.0
                contact_data.radius_eff_b = 0.0
                contact_data.margin_a = 0.0
                contact_data.margin_b = 0.0
                contact_data.shape_a = shape_a_first
                contact_data.shape_b = shape_b_first
                contact_data.gap_sum = gap_sum
                contact_data.contact_stiffness = shared_stiffness
                contact_data.contact_friction_scale = wp.float32(anchor_friction_scale)
                bin_id = int((ht_keys[entry_idx] >> wp.uint64(55)) & wp.uint64(0xFF))
                contact_data.sort_sub_key = 0x400000 | bin_id

                # Call the writer function for anchor
                writer_func(contact_data, writer_data, -1)

    return export_hydroelastic_reduced_contacts_kernel


# =============================================================================
# Hydroelastic Contact Reduction API
# =============================================================================


@dataclass
class HydroelasticReductionConfig:
    """Configuration for hydroelastic contact reduction.

    Attributes:
        normal_matching: If True, rotate reduced contact normals so their weighted
            sum aligns with the aggregate force direction.
        anchor_contact: If True, add an anchor contact at the center of pressure
            for each normal bin. The anchor contact helps preserve moment balance.
        moment_matching: If True, adjust per-contact friction scales so that the
            maximum friction moment per normal bin is preserved between reduced
            and unreduced contacts. Automatically enables ``anchor_contact``.
        margin_contact_area: Contact area used for non-penetrating contacts at the margin.
        hashtable_size_factor: Multiplier applied to the contact buffer capacity
            when allocating the reduction hashtable. Must be positive.
    """

    normal_matching: bool = True
    anchor_contact: bool = False
    moment_matching: bool = False
    margin_contact_area: float = 1e-2
    hashtable_size_factor: float = 0.25


class HydroelasticContactReduction:
    """High-level API for hydroelastic contact reduction.

    This class encapsulates the hydroelastic contact reduction pipeline, providing
    a clean interface that hides the low-level kernel launch details. It manages:

    1. A ``GlobalContactReducer`` for contact storage and hashtable tracking
    2. The reduction kernels for hashtable registration
    3. The export kernel for writing reduced contacts

    **Usage Pattern:**

    The typical usage in a contact generation pipeline is:

    1. Call ``clear()`` at the start of each frame
    2. Write contacts to the buffer using ``export_hydroelastic_contact_to_buffer``
       in your contact generation kernel (use ``get_data_struct()`` to get the data)
    3. Call ``reduce()`` to register contacts in the hashtable
    4. Call ``export()`` to write reduced contacts using the writer function

    Example:

        .. code-block:: python

            # Initialize once
            config = HydroelasticReductionConfig(normal_matching=True)
            reduction = HydroelasticContactReduction(
                capacity=100000,
                device="cuda:0",
                writer_func=my_writer_func,
                config=config,
                pressure_func=my_pressure_func,
                pressure_data=my_pressure_data,
            )

            # Each frame
            reduction.clear()

            # Launch your contact generation kernel that uses:
            # export_hydroelastic_contact_to_buffer(..., reduction.get_data_struct())

            reduction.reduce(shape_material_k_hydro, shape_transform, aabb_lower, aabb_upper, voxel_res, grid_size)
            reduction.export(shape_gap, shape_transform, writer_data, grid_size)

    Attributes:
        reducer: The underlying ``GlobalContactReducer`` instance.
        config: The ``HydroelasticReductionConfig`` for this instance.
        contact_count: Array containing the number of contacts in the buffer.

    See Also:
        :func:`export_hydroelastic_contact_to_buffer`: Warp function for writing
            contacts to the buffer from custom kernels.
        :class:`GlobalContactReducerData`: Struct for passing reducer data to kernels.
    """

    def __init__(
        self,
        capacity: int,
        device: str | None = None,
        writer_func: Any = None,
        config: HydroelasticReductionConfig | None = None,
        pressure_func: Any = None,
        pressure_data: Any = None,
        deterministic: bool = False,
    ):
        """Initialize the hydroelastic contact reduction system.

        Args:
            capacity: Maximum number of contacts to store in the buffer.
            device: Warp device (e.g., "cuda:0", "cpu"). If None, uses default device.
            writer_func: Warp function for writing decoded contacts. Must have signature
                ``(ContactData, writer_data, int) -> None``.
            config: Configuration options. If None, uses default ``HydroelasticReductionConfig``.
            pressure_func: Warp function ``(signed_depth, shape_idx, data) -> pressure``
                used to compute per-contact force throughout the reduction pipeline.
                Required.
            pressure_data: ``@wp.struct`` instance carrying state for ``pressure_func``.
                Threaded through the reduce / export kernel launches. Required.
            deterministic: Whether to use fingerprint-based winner selection and
                int64 fixed-point aggregate accumulation, making results
                independent of GPU thread scheduling.
        """
        if pressure_func is None or pressure_data is None:
            raise ValueError("HydroelasticContactReduction requires pressure_func and pressure_data.")
        if config is None:
            config = HydroelasticReductionConfig()
        # Moment matching requires anchor contact for lever-arm reference
        if config.moment_matching and not config.anchor_contact:
            config.anchor_contact = True
        self.config = config
        self.device = device
        self.pressure_data = pressure_data
        self.deterministic = deterministic

        # Create the underlying reducer with hydroelastic data storage enabled
        self.reducer = GlobalContactReducer(
            capacity=capacity,
            device=device,
            store_hydroelastic_data=True,
            store_moment_data=config.moment_matching,
            deterministic=deterministic,
            hashtable_size_factor=config.hashtable_size_factor,
        )

        # Fixed-point accumulators, used only in deterministic mode.  Unreduced
        # aggregates take one term per buffered contact; the reduced ones take up
        # to VALUES_PER_KEY winners from every hashtable entry, which can exceed
        # the contact capacity.  The wider of the two bounds sizes the grid.
        ht_capacity = self.reducer.hashtable.capacity
        self._mantissa_bits = _fixed_mantissa_bits(max(capacity, ht_capacity * VALUES_PER_KEY))
        if deterministic:
            self._fixed_accum = wp.zeros(NUM_FIXED_SLOTS * ht_capacity, dtype=wp.int64, device=device)
            self._fixed_scale = wp.full(NUM_SCALE_FAMILIES * ht_capacity, FIXED_EXP_NONE, dtype=wp.int32, device=device)
        else:
            self._fixed_accum = wp.zeros(0, dtype=wp.int64, device=device)
            self._fixed_scale = wp.zeros(0, dtype=wp.int32, device=device)

        # Create reduction kernel
        self._reduce_kernel = get_reduce_hydroelastic_contacts_kernel(pressure_func, deterministic=deterministic)
        self._accumulate_depth_kernel = _create_accumulate_reduced_depth_kernel(
            deterministic=deterministic, mantissa_bits=self._mantissa_bits
        )

        # In deterministic mode the generate kernel skips the unreduced aggregate
        # atomics; they are recomputed here in fixed point instead.
        self._unreduced_aggregate_kernel = None
        self._unreduced_moment_kernel = None
        self._finalize_kernel = None
        if deterministic:
            self._unreduced_aggregate_kernel = _create_unreduced_aggregate_kernel(pressure_func, self._mantissa_bits)
            self._finalize_kernel = _create_finalize_fixed_kernel(self._mantissa_bits)
            if config.moment_matching:
                self._unreduced_moment_kernel = _create_unreduced_moment_kernel(pressure_func, self._mantissa_bits)

        # Create moment accumulation kernel (only when moment matching is enabled)
        self._accumulate_moments_kernel = None
        if config.moment_matching:
            self._accumulate_moments_kernel = _create_accumulate_moments_kernel(
                normal_matching=config.normal_matching,
                deterministic=deterministic,
                mantissa_bits=self._mantissa_bits,
            )

        # Create the export kernel with the configured options
        self._export_kernel = create_export_hydroelastic_reduced_contacts_kernel(
            writer_func=writer_func,
            margin_contact_area=config.margin_contact_area,
            normal_matching=config.normal_matching,
            anchor_contact=config.anchor_contact,
            moment_matching=config.moment_matching,
            pressure_func=pressure_func,
            deterministic_sort_keys=deterministic,
        )

    @property
    def contact_count(self) -> wp.array:
        """Array containing the current number of contacts in the buffer."""
        return self.reducer.contact_count

    @property
    def capacity(self) -> int:
        """Maximum number of contacts that can be stored."""
        return self.reducer.capacity

    def get_data_struct(self) -> GlobalContactReducerData:
        """Get the data struct for passing to Warp kernels.

        Returns:
            A ``GlobalContactReducerData`` struct containing all arrays needed
            for contact storage and reduction.
        """
        return self.reducer.get_data_struct()

    def _launch_finalize(self, reducer_data: GlobalContactReducerData, group: int):
        """Convert one group of completed fixed-point sums back to float32."""
        wp.launch(
            kernel=self._finalize_kernel,
            dim=[self.reducer.hashtable.capacity],
            inputs=[
                reducer_data,
                self._fixed_accum,
                self._fixed_scale,
                self.reducer.agg_moment_unreduced,
                self.reducer.agg_moment_reduced,
                self.reducer.agg_moment2_reduced,
                group,
            ],
            device=self.device,
            record_tape=False,
        )

    def clear(self):
        """Clear all contacts and reset for a new frame.

        This efficiently clears only the active hashtable entries and resets
        the contact counter. Call this at the start of each simulation step.
        """
        self.reducer.clear_active()
        if self.deterministic:
            self._fixed_accum.zero_()
            self._fixed_scale.fill_(FIXED_EXP_NONE)

    def reduce(
        self,
        shape_material_k_hydro: wp.array,
        shape_transform: wp.array,
        shape_collision_aabb_lower: wp.array,
        shape_collision_aabb_upper: wp.array,
        shape_voxel_resolution: wp.array,
        grid_size: int,
    ):
        """Register buffered contacts in the hashtable for reduction.

        This launches the reduction kernel that processes all contacts in the
        buffer and registers them in the hashtable based on spatial extremes,
        max-depth per normal bin, and voxel-based slots.

        Aggregate accumulation (``agg_force``, ``weighted_pos_sum``,
        ``weight_sum``, ``agg_depth_volume``) normally happens in the generate
        kernel, leaving this method to do only hashtable slot registration.  In
        deterministic mode the generate kernel skips it and this method
        accumulates instead, in int64 fixed point so the sums do not depend on
        the order threads arrive.

        Args:
            shape_material_k_hydro: Per-shape hydroelastic material stiffness (dtype: float).
            shape_transform: Per-shape world transforms (dtype: wp.transform).
            shape_collision_aabb_lower: Per-shape local AABB lower bounds (dtype: wp.vec3).
            shape_collision_aabb_upper: Per-shape local AABB upper bounds (dtype: wp.vec3).
            shape_voxel_resolution: Per-shape voxel grid resolution (dtype: wp.vec3i).
            grid_size: Number of threads for the kernel launch.
        """
        reducer_data = self.reducer.get_data_struct()
        if self.deterministic:
            # Bin registration must precede aggregate accumulation because the
            # accumulators index by the cached ``contact_nbin_entry``.
            wp.launch(
                kernel=_register_hydroelastic_normal_bins_kernel,
                dim=[grid_size],
                inputs=[reducer_data, shape_material_k_hydro, grid_size],
                device=self.device,
                record_tape=False,
            )
            # Size the fixed-point grid, accumulate onto it, then convert back.
            for phase in (PHASE_SCALE, PHASE_ACCUMULATE):
                wp.launch(
                    kernel=self._unreduced_aggregate_kernel,
                    dim=[grid_size],
                    inputs=[reducer_data, self.pressure_data, self._fixed_accum, self._fixed_scale, phase, grid_size],
                    device=self.device,
                    record_tape=False,
                )
            self._launch_finalize(reducer_data, FINALIZE_UNREDUCED)
            if self._unreduced_moment_kernel is not None:
                # Lever arms need the finalized center of pressure.
                for phase in (PHASE_SCALE, PHASE_ACCUMULATE):
                    wp.launch(
                        kernel=self._unreduced_moment_kernel,
                        dim=[grid_size],
                        inputs=[
                            reducer_data,
                            self.pressure_data,
                            self._fixed_accum,
                            self._fixed_scale,
                            phase,
                            grid_size,
                        ],
                        device=self.device,
                        record_tape=False,
                    )
        wp.launch(
            kernel=self._reduce_kernel,
            dim=[grid_size],
            inputs=[
                reducer_data,
                shape_material_k_hydro,
                self.pressure_data,
                shape_transform,
                shape_collision_aabb_lower,
                shape_collision_aabb_upper,
                shape_voxel_resolution,
                self.reducer.agg_moment_unreduced,
                grid_size,
            ],
            device=self.device,
            record_tape=False,
        )

    def export(
        self,
        shape_gap: wp.array,
        shape_transform: wp.array,
        writer_data: Any,
        grid_size: int,
    ):
        """Export reduced contacts using the writer function.

        This first launches the accumulation kernel so that
        ``total_depth_reduced`` and ``total_normal_reduced`` are fully
        populated before the export kernel reads them (the implicit
        synchronisation between ``wp.launch()`` calls acts as the required
        global memory barrier).

        Args:
            shape_gap: Per-shape contact gap (detection threshold) (dtype: float).
            shape_transform: Per-shape world transforms (dtype: wp.transform).
            writer_data: Data struct for the writer function.
            grid_size: Number of threads for the kernel launch.
        """
        # Deterministic mode runs each accumulator twice: once to size the
        # per-entry fixed-point grid, once to sum onto it.
        phases = (PHASE_SCALE, PHASE_ACCUMULATE) if self.deterministic else (PHASE_ACCUMULATE,)
        # --- accumulate winning-contact depths per normal bin (Phase 1) ---
        for phase in phases:
            wp.launch(
                kernel=self._accumulate_depth_kernel,
                dim=[grid_size],
                inputs=[
                    self.reducer.hashtable.keys,
                    self.reducer.ht_values,
                    self.reducer.hashtable.active_slots,
                    self.reducer.position_depth,
                    self.reducer.normal,
                    self.reducer.contact_nbin_entry,
                    self.reducer.total_depth_reduced,
                    self.reducer.total_normal_reduced,
                    self._fixed_accum,
                    self._fixed_scale,
                    phase,
                    grid_size,
                ],
                device=self.device,
            )
        if self.deterministic:
            # Reduced normals feed the moment accumulator's normal matching.
            self._launch_finalize(self.reducer.get_data_struct(), FINALIZE_REDUCED_DEPTH)
        # --- accumulate reduced friction moments per normal bin (Phase 1.5) ---
        if self._accumulate_moments_kernel is not None:
            for phase in phases:
                wp.launch(
                    kernel=self._accumulate_moments_kernel,
                    dim=[grid_size],
                    inputs=[
                        self.reducer.hashtable.keys,
                        self.reducer.ht_values,
                        self.reducer.hashtable.active_slots,
                        self.reducer.position_depth,
                        self.reducer.normal,
                        self.reducer.contact_nbin_entry,
                        self.reducer.weighted_pos_sum,
                        self.reducer.weight_sum,
                        self.reducer.agg_force,
                        self.reducer.agg_depth_volume,
                        self.reducer.total_normal_reduced,
                        self.reducer.agg_moment_reduced,
                        self.reducer.agg_moment2_reduced,
                        self._fixed_accum,
                        self._fixed_scale,
                        phase,
                        grid_size,
                    ],
                    device=self.device,
                )
        if self.deterministic and self._accumulate_moments_kernel is not None:
            self._launch_finalize(self.reducer.get_data_struct(), FINALIZE_MOMENTS)
        # --- export reduced contacts (Phase 2) ---
        wp.launch(
            kernel=self._export_kernel,
            dim=[grid_size],
            inputs=[
                self.reducer.hashtable.keys,
                self.reducer.ht_values,
                self.reducer.hashtable.active_slots,
                self.reducer.agg_force,
                self.reducer.agg_depth_volume,
                self.reducer.weighted_pos_sum,
                self.reducer.weight_sum,
                self.reducer.position_depth,
                self.reducer.normal,
                self.reducer.shape_pairs,
                self.reducer.contact_fingerprints,
                self.reducer.contact_area,
                self.reducer.entry_k_eff,
                self.reducer.contact_nbin_entry,
                self.reducer.total_depth_reduced,
                self.reducer.total_normal_reduced,
                self.reducer.agg_moment_unreduced,
                self.reducer.agg_moment_reduced,
                self.reducer.agg_moment2_reduced,
                shape_gap,
                shape_transform,
                self.pressure_data,
                writer_data,
                int(self.deterministic),
                grid_size,
            ],
            device=self.device,
            record_tape=False,
        )

    def reduce_and_export(
        self,
        shape_material_k_hydro: wp.array,
        shape_transform: wp.array,
        shape_collision_aabb_lower: wp.array,
        shape_collision_aabb_upper: wp.array,
        shape_voxel_resolution: wp.array,
        shape_gap: wp.array,
        writer_data: Any,
        grid_size: int,
    ):
        """Convenience method to reduce and export in one call.

        Combines ``reduce()`` and ``export()`` into a single method call.

        Args:
            shape_material_k_hydro: Per-shape hydroelastic material stiffness (dtype: float).
            shape_transform: Per-shape world transforms (dtype: wp.transform).
            shape_collision_aabb_lower: Per-shape local AABB lower bounds (dtype: wp.vec3).
            shape_collision_aabb_upper: Per-shape local AABB upper bounds (dtype: wp.vec3).
            shape_voxel_resolution: Per-shape voxel grid resolution (dtype: wp.vec3i).
            shape_gap: Per-shape contact gap (detection threshold) (dtype: float).
            writer_data: Data struct for the writer function.
            grid_size: Number of threads for the kernel launch.
        """
        self.reduce(
            shape_material_k_hydro,
            shape_transform,
            shape_collision_aabb_lower,
            shape_collision_aabb_upper,
            shape_voxel_resolution,
            grid_size,
        )
        self.export(shape_gap, shape_transform, writer_data, grid_size)
