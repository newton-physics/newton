# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Safeguarded Anderson(1) acceleration for LOX projection maps."""

from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp

from ...core.types import mat36f, mat66f, vec6f
from ..padmm.math import project_to_coulomb_cone
from .contact import (
    angular_contact_reaction,
    clamp_contact_reaction,
    linear_contact_reaction,
    pack_contact_reaction,
    pack_spatial_friction,
    scatter_spatial_impulse,
)
from .projection_kernels import is_active_row, scatter_box_impulse, scatter_contact_impulse

if TYPE_CHECKING:
    from .types import AndersonData, LOXProblemData

###
# Module interface
###

__all__ = [
    "AndersonProjection",
    "AndersonWorldState",
    "finalize_anderson_world",
    "mix_box_anderson_row",
    "mix_contact_anderson_row",
    "store_box_anderson_sample",
    "store_contact_anderson_sample",
    "store_contact_map_input",
]


###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Constants
###


_ANDERSON_REGULARIZATION = 1.0e-4


_ANDERSON_COEFFICIENT_LIMIT = 10.0


_ANDERSON_RESIDUAL_GROWTH_LIMIT_SQUARED = 2.25


_ANDERSON_REDUCTION_BLOCK_DIM = 256


###
# Types
###


@wp.struct
class AndersonWorldState:
    """Anderson state of each world and row, for the per-world projection."""

    sample_valid: wp.array[wp.int32]
    direction_valid: wp.array[wp.int32]
    guard_pending: wp.array[wp.int32]
    previous_residual_norm: wp.array[wp.float32]
    coefficient: wp.array[wp.float32]
    box_input: wp.array[wp.float32]
    box_sample_residual: wp.array[wp.float32]
    box_sample_image: wp.array[wp.float32]
    box_direction_residual: wp.array[wp.float32]
    box_direction_image: wp.array[wp.float32]
    contact_input: wp.array[wp.vec3f]
    contact_sample_residual: wp.array[wp.vec3f]
    contact_sample_image: wp.array[wp.vec3f]
    contact_direction_residual: wp.array[wp.vec3f]
    contact_direction_image: wp.array[wp.vec3f]
    angular_input: wp.array[wp.vec3f]
    angular_sample_residual: wp.array[wp.vec3f]
    angular_sample_image: wp.array[wp.vec3f]
    angular_direction_residual: wp.array[wp.vec3f]
    angular_direction_image: wp.array[wp.vec3f]


###
# Functions
###


@wp.func
def _angular_metric(angular_friction: wp.vec2f) -> wp.vec3f:
    # Divide angular impulses by their friction lengths so that they share
    # the linear impulse units; disabled angular rows get a zero weight.
    torsional = wp.float32(0.0)
    rolling = wp.float32(0.0)
    if angular_friction[0] > 0.0:
        torsional = 1.0 / (angular_friction[0] * angular_friction[0])
    if angular_friction[1] > 0.0:
        rolling = 1.0 / (angular_friction[1] * angular_friction[1])
    return wp.vec3f(torsional, rolling, rolling)


@wp.func
def _store_scalar_anderson_sample(
    value: wp.float32,
    map_input: wp.float32,
    previous_residual: wp.float32,
    stored_residual_direction: wp.float32,
    had_sample: wp.bool,
    had_direction: wp.bool,
) -> wp.vec3f:
    residual = value - map_input
    direction = stored_residual_direction
    if had_sample:
        direction = previous_residual - residual
    gram = wp.float32(0.0)
    rhs = wp.float32(0.0)
    if had_sample or had_direction:
        gram = direction * direction
        rhs = -direction * residual
    return wp.vec3f(gram, rhs, residual * residual)


@wp.func
def _store_vector_anderson_sample(
    value: wp.vec3f,
    map_input: wp.vec3f,
    previous_residual: wp.vec3f,
    stored_residual_direction: wp.vec3f,
    had_sample: wp.bool,
    had_direction: wp.bool,
    metric: wp.vec3f,
) -> wp.vec3f:
    residual = value - map_input
    direction = stored_residual_direction
    if had_sample:
        direction = previous_residual - residual
    gram = wp.float32(0.0)
    rhs = wp.float32(0.0)
    if had_sample or had_direction:
        gram = wp.dot(direction, wp.cw_mul(metric, direction))
        rhs = -wp.dot(direction, wp.cw_mul(metric, residual))
    return wp.vec3f(gram, rhs, wp.dot(residual, wp.cw_mul(metric, residual)))


@wp.func
def finalize_anderson_world(
    wid: wp.int32,
    world_active: wp.array[wp.bool],
    world_failed: wp.array[wp.bool],
    gram_value: wp.float32,
    rhs_value: wp.float32,
    norm_value: wp.float32,
    sample_valid: wp.array[wp.int32],
    direction_valid: wp.array[wp.int32],
    guard_pending: wp.array[wp.int32],
    previous_residual_norm: wp.array[wp.float32],
    coefficient: wp.array[wp.float32],
):
    """Set the Anderson coefficient of one world from its secant sums."""
    if not world_active[wid] or world_failed[wid]:
        coefficient[wid] = 0.0
        sample_valid[wid] = 0
        direction_valid[wid] = 0
        guard_pending[wid] = 0
        return
    had_sample = sample_valid[wid] == 1
    had_direction = had_sample or direction_valid[wid] == 1
    residual_grew = (
        guard_pending[wid] == 1 and norm_value > _ANDERSON_RESIDUAL_GROWTH_LIMIT_SQUARED * previous_residual_norm[wid]
    )
    denominator = gram_value + _ANDERSON_REGULARIZATION * wp.max(gram_value, 1.0e-12)
    # Both comparisons are false for NaN, which rejects non-finite secant sums
    okay = had_direction and not residual_grew and denominator > 1.0e-20
    coefficient_value = wp.float32(0.0)
    if okay:
        coefficient_value = rhs_value / denominator
        okay = wp.abs(coefficient_value) <= _ANDERSON_COEFFICIENT_LIMIT
    coefficient[wid] = wp.where(okay, coefficient_value, 0.0)
    direction_valid[wid] = wp.int32(okay)
    sample_valid[wid] = 1
    guard_pending[wid] = wp.int32(okay)
    previous_residual_norm[wid] = norm_value


@wp.func
def store_contact_map_input(
    cid: wp.int32,
    contact_reaction: wp.array[wp.vec3f],
    angular_reaction: wp.array[wp.vec3f],
    contact_input: wp.array[wp.vec3f],
    angular_input: wp.array[wp.vec3f],
):
    """Store the reaction of one contact as the input of the next map."""
    contact_input[cid] = contact_reaction[cid]
    if angular_reaction:
        angular_input[cid] = angular_reaction[cid]


@wp.func
def store_box_anderson_sample(
    row: wp.int32,
    had_sample: wp.bool,
    had_direction: wp.bool,
    box_reaction: wp.array[wp.float32],
    box_input: wp.array[wp.float32],
    box_sample_residual: wp.array[wp.float32],
    box_sample_image: wp.array[wp.float32],
    box_direction_residual: wp.array[wp.float32],
    box_direction_image: wp.array[wp.float32],
) -> wp.vec3f:
    """Store the newest map residual and image of one box row, and return its secant terms."""
    current = box_reaction[row]
    current_residual = current - box_input[row]
    term = _store_scalar_anderson_sample(
        current,
        box_input[row],
        box_sample_residual[row],
        box_direction_residual[row],
        had_sample,
        had_direction,
    )
    if had_sample:
        box_direction_residual[row] = box_sample_residual[row] - current_residual
        box_direction_image[row] = box_sample_image[row] - current
    box_sample_residual[row] = current_residual
    box_sample_image[row] = current
    return term


@wp.func
def store_contact_anderson_sample(
    cid: wp.int32,
    had_sample: wp.bool,
    had_direction: wp.bool,
    contact_reaction: wp.array[wp.vec3f],
    contact_input: wp.array[wp.vec3f],
    contact_sample_residual: wp.array[wp.vec3f],
    contact_sample_image: wp.array[wp.vec3f],
    contact_direction_residual: wp.array[wp.vec3f],
    contact_direction_image: wp.array[wp.vec3f],
    angular_reaction: wp.array[wp.vec3f],
    angular_input: wp.array[wp.vec3f],
    angular_sample_residual: wp.array[wp.vec3f],
    angular_sample_image: wp.array[wp.vec3f],
    angular_direction_residual: wp.array[wp.vec3f],
    angular_direction_image: wp.array[wp.vec3f],
    angular_friction: wp.array[wp.vec2f],
) -> wp.vec3f:
    """Store the newest map residual and image of one contact, and return its secant terms."""
    current = contact_reaction[cid]
    current_residual = current - contact_input[cid]
    term = _store_vector_anderson_sample(
        current,
        contact_input[cid],
        contact_sample_residual[cid],
        contact_direction_residual[cid],
        had_sample,
        had_direction,
        wp.vec3f(1.0),
    )
    if had_sample:
        contact_direction_residual[cid] = contact_sample_residual[cid] - current_residual
        contact_direction_image[cid] = contact_sample_image[cid] - current
    contact_sample_residual[cid] = current_residual
    contact_sample_image[cid] = current
    if angular_reaction:
        angular_current = angular_reaction[cid]
        angular_current_residual = angular_current - angular_input[cid]
        term += _store_vector_anderson_sample(
            angular_current,
            angular_input[cid],
            angular_sample_residual[cid],
            angular_direction_residual[cid],
            had_sample,
            had_direction,
            _angular_metric(angular_friction[cid]),
        )
        if had_sample:
            angular_direction_residual[cid] = angular_sample_residual[cid] - angular_current_residual
            angular_direction_image[cid] = angular_sample_image[cid] - angular_current
        angular_sample_residual[cid] = angular_current_residual
        angular_sample_image[cid] = angular_current
    return term


@wp.func
def mix_box_anderson_row(
    row: wp.int32,
    coefficient: wp.float32,
    box_body_a: wp.array[wp.int32],
    box_body_b: wp.array[wp.int32],
    box_jacobian_a: wp.array[vec6f],
    box_jacobian_b: wp.array[vec6f],
    box_lower: wp.array[wp.float32],
    box_upper: wp.array[wp.float32],
    box_direction_image: wp.array[wp.float32],
    inverse_weight: wp.array[mat66f],
    box_input: wp.array[wp.float32],
    box_reaction: wp.array[wp.float32],
    body_delta: wp.array[vec6f],
):
    """Mix the map image of one box row along the stored direction, clamp it, and scatter the change."""
    current = box_reaction[row]
    if coefficient == 0.0:
        # The image is the next input; the direction may be stale, from a failed solve
        box_input[row] = current
        return
    mixed = wp.clamp(current + coefficient * box_direction_image[row], box_lower[row], box_upper[row])
    box_input[row] = mixed
    box_reaction[row] = mixed
    scatter_box_impulse(
        row,
        box_body_a,
        box_body_b,
        box_jacobian_a,
        box_jacobian_b,
        mixed - current,
        inverse_weight,
        body_delta,
    )


@wp.func
def mix_contact_anderson_row(
    cid: wp.int32,
    coefficient: wp.float32,
    contact_body_a: wp.array[wp.int32],
    contact_body_b: wp.array[wp.int32],
    contact_jacobian_a: wp.array[mat36f],
    contact_jacobian_b: wp.array[mat36f],
    contact_friction: wp.array[wp.float32],
    contact_direction_image: wp.array[wp.vec3f],
    angular_friction: wp.array[wp.vec2f],
    contact_frame: wp.array[wp.mat33f],
    angular_direction_image: wp.array[wp.vec3f],
    inverse_weight: wp.array[mat66f],
    contact_input: wp.array[wp.vec3f],
    contact_reaction: wp.array[wp.vec3f],
    angular_input: wp.array[wp.vec3f],
    angular_reaction: wp.array[wp.vec3f],
    body_delta: wp.array[vec6f],
):
    """Mix the map image of one contact along the stored direction, project it onto its cone, and scatter the change."""
    current = contact_reaction[cid]
    if coefficient == 0.0:
        # The image is the next input; the directions may be stale, from a failed solve
        contact_input[cid] = current
        if angular_reaction:
            angular_input[cid] = angular_reaction[cid]
        return
    trial = current + coefficient * contact_direction_image[cid]
    if angular_reaction:
        angular_current = angular_reaction[cid]
        spatial_mixed = clamp_contact_reaction(
            pack_contact_reaction(trial, angular_current + coefficient * angular_direction_image[cid]),
            pack_spatial_friction(cid, contact_friction, angular_friction),
        )
        mixed = linear_contact_reaction(spatial_mixed)
        angular_mixed = angular_contact_reaction(spatial_mixed)
        angular_input[cid] = angular_mixed
        angular_reaction[cid] = angular_mixed
        contact_input[cid] = mixed
        contact_reaction[cid] = mixed
        scatter_spatial_impulse(
            cid,
            contact_body_a,
            contact_body_b,
            contact_jacobian_a,
            contact_jacobian_b,
            contact_frame,
            spatial_mixed - pack_contact_reaction(current, angular_current),
            inverse_weight,
            body_delta,
        )
        return
    mixed = project_to_coulomb_cone(trial, contact_friction[cid])
    contact_input[cid] = mixed
    contact_reaction[cid] = mixed
    scatter_contact_impulse(
        cid,
        contact_body_a,
        contact_body_b,
        contact_jacobian_a,
        contact_jacobian_b,
        mixed - current,
        inverse_weight,
        body_delta,
    )


###
# Kernels
###


@wp.kernel
def _begin_anderson_outer_iteration(
    # Inputs:
    world_active: wp.array[wp.bool],
    recycle: wp.bool,
    # Outputs:
    sample_valid: wp.array[wp.int32],
    direction_valid: wp.array[wp.int32],
    guard_pending: wp.array[wp.int32],
):
    wid = wp.tid()
    if world_active[wid]:
        sample_valid[wid] = 0
        guard_pending[wid] = 0
        if not recycle:
            direction_valid[wid] = 0


@wp.kernel
def _store_map_input(
    # Inputs:
    box_capacity: wp.int32,
    box_world: wp.array[wp.int32],
    box_local: wp.array[wp.int32],
    world_box_count: wp.array[wp.int32],
    contact_world: wp.array[wp.int32],
    contact_local: wp.array[wp.int32],
    world_contact_count: wp.array[wp.int32],
    world_active: wp.array[wp.bool],
    world_failed: wp.array[wp.bool],
    box_reaction: wp.array[wp.float32],
    contact_reaction: wp.array[wp.vec3f],
    angular_reaction: wp.array[wp.vec3f],
    # Outputs:
    box_input: wp.array[wp.float32],
    contact_input: wp.array[wp.vec3f],
    angular_input: wp.array[wp.vec3f],
):
    tid = wp.tid()
    if tid < box_capacity:
        row = tid
        if is_active_row(box_local[row], box_world[row], world_box_count, world_active, world_failed):
            box_input[row] = box_reaction[row]
        return
    cid = tid - box_capacity
    if is_active_row(contact_local[cid], contact_world[cid], world_contact_count, world_active, world_failed):
        store_contact_map_input(cid, contact_reaction, angular_reaction, contact_input, angular_input)


@wp.kernel
def _store_anderson_samples(
    # Inputs:
    box_capacity: wp.int32,
    global_reduction: wp.bool,
    box_world: wp.array[wp.int32],
    box_local: wp.array[wp.int32],
    world_box_count: wp.array[wp.int32],
    contact_world: wp.array[wp.int32],
    contact_local: wp.array[wp.int32],
    world_contact_count: wp.array[wp.int32],
    world_active: wp.array[wp.bool],
    world_failed: wp.array[wp.bool],
    sample_valid: wp.array[wp.int32],
    direction_valid: wp.array[wp.int32],
    box_reaction: wp.array[wp.float32],
    box_input: wp.array[wp.float32],
    contact_reaction: wp.array[wp.vec3f],
    contact_input: wp.array[wp.vec3f],
    angular_reaction: wp.array[wp.vec3f],
    angular_input: wp.array[wp.vec3f],
    angular_friction: wp.array[wp.vec2f],
    # Outputs:
    box_sample_residual: wp.array[wp.float32],
    box_sample_image: wp.array[wp.float32],
    box_direction_residual: wp.array[wp.float32],
    box_direction_image: wp.array[wp.float32],
    contact_sample_residual: wp.array[wp.vec3f],
    contact_sample_image: wp.array[wp.vec3f],
    contact_direction_residual: wp.array[wp.vec3f],
    contact_direction_image: wp.array[wp.vec3f],
    angular_sample_residual: wp.array[wp.vec3f],
    angular_sample_image: wp.array[wp.vec3f],
    angular_direction_residual: wp.array[wp.vec3f],
    angular_direction_image: wp.array[wp.vec3f],
    terms: wp.array[wp.vec3f],
    gram: wp.array[wp.float32],
    rhs: wp.array[wp.float32],
    residual_norm: wp.array[wp.float32],
):
    """Store the newest map residual and image, and reduce the secant terms per world."""
    tid = wp.tid()
    term = wp.vec3f(0.0)
    wid = wp.int32(0)
    active = False
    if tid < box_capacity:
        row = tid
        wid = box_world[row]
        active = is_active_row(box_local[row], wid, world_box_count, world_active, world_failed)
        if active:
            term = store_box_anderson_sample(
                row,
                sample_valid[wid] == 1,
                direction_valid[wid] == 1,
                box_reaction,
                box_input,
                box_sample_residual,
                box_sample_image,
                box_direction_residual,
                box_direction_image,
            )
    else:
        cid = tid - box_capacity
        wid = contact_world[cid]
        active = is_active_row(contact_local[cid], wid, world_contact_count, world_active, world_failed)
        if active:
            term = store_contact_anderson_sample(
                cid,
                sample_valid[wid] == 1,
                direction_valid[wid] == 1,
                contact_reaction,
                contact_input,
                contact_sample_residual,
                contact_sample_image,
                contact_direction_residual,
                contact_direction_image,
                angular_reaction,
                angular_input,
                angular_sample_residual,
                angular_sample_image,
                angular_direction_residual,
                angular_direction_image,
                angular_friction,
            )
    if global_reduction:
        terms[tid] = term
    elif active:
        wp.atomic_add(gram, wid, term[0])
        wp.atomic_add(rhs, wid, term[1])
        wp.atomic_add(residual_norm, wid, term[2])


@wp.kernel
def _finalize_anderson_coefficients(
    # Inputs:
    world_active: wp.array[wp.bool],
    world_failed: wp.array[wp.bool],
    # Outputs:
    gram: wp.array[wp.float32],
    rhs: wp.array[wp.float32],
    residual_norm: wp.array[wp.float32],
    sample_valid: wp.array[wp.int32],
    direction_valid: wp.array[wp.int32],
    guard_pending: wp.array[wp.int32],
    previous_residual_norm: wp.array[wp.float32],
    coefficient: wp.array[wp.float32],
):
    """Finalize the Anderson coefficient of each world from its secant sums, then clear the sums for the next map."""
    wid = wp.tid()
    finalize_anderson_world(
        wid,
        world_active,
        world_failed,
        gram[wid],
        rhs[wid],
        residual_norm[wid],
        sample_valid,
        direction_valid,
        guard_pending,
        previous_residual_norm,
        coefficient,
    )
    gram[wid] = 0.0
    rhs[wid] = 0.0
    residual_norm[wid] = 0.0


@wp.kernel
def _reduce_and_finalize_anderson_terms(
    # Inputs:
    term_count: wp.int32,
    terms: wp.array[wp.vec3f],
    world_active: wp.array[wp.bool],
    world_failed: wp.array[wp.bool],
    # Outputs:
    sample_valid: wp.array[wp.int32],
    direction_valid: wp.array[wp.int32],
    guard_pending: wp.array[wp.int32],
    previous_residual_norm: wp.array[wp.float32],
    coefficient: wp.array[wp.float32],
):
    tid = wp.tid()
    thread_sum = wp.vec3f(0.0)
    for term in range(tid, term_count, wp.block_dim()):
        thread_sum += terms[term]
    block_sum = wp.tile_sum(wp.tile(thread_sum), axis=1)
    if tid == 0:
        finalize_anderson_world(
            0,
            world_active,
            world_failed,
            block_sum[0],
            block_sum[1],
            block_sum[2],
            sample_valid,
            direction_valid,
            guard_pending,
            previous_residual_norm,
            coefficient,
        )


@wp.kernel
def _mix_anderson_reactions(
    # Inputs:
    box_capacity: wp.int32,
    box_world: wp.array[wp.int32],
    box_local: wp.array[wp.int32],
    world_box_count: wp.array[wp.int32],
    box_body_a: wp.array[wp.int32],
    box_body_b: wp.array[wp.int32],
    box_jacobian_a: wp.array[vec6f],
    box_jacobian_b: wp.array[vec6f],
    box_lower: wp.array[wp.float32],
    box_upper: wp.array[wp.float32],
    box_direction_image: wp.array[wp.float32],
    contact_world: wp.array[wp.int32],
    contact_local: wp.array[wp.int32],
    world_contact_count: wp.array[wp.int32],
    contact_body_a: wp.array[wp.int32],
    contact_body_b: wp.array[wp.int32],
    contact_jacobian_a: wp.array[mat36f],
    contact_jacobian_b: wp.array[mat36f],
    contact_friction: wp.array[wp.float32],
    contact_direction_image: wp.array[wp.vec3f],
    angular_friction: wp.array[wp.vec2f],
    contact_frame: wp.array[wp.mat33f],
    angular_direction_image: wp.array[wp.vec3f],
    world_active: wp.array[wp.bool],
    coefficient: wp.array[wp.float32],
    inverse_weight: wp.array[mat66f],
    # Outputs:
    box_input: wp.array[wp.float32],
    box_reaction: wp.array[wp.float32],
    contact_input: wp.array[wp.vec3f],
    contact_reaction: wp.array[wp.vec3f],
    angular_input: wp.array[wp.vec3f],
    angular_reaction: wp.array[wp.vec3f],
    world_failed: wp.array[wp.bool],
    body_delta: wp.array[vec6f],
):
    """Mix each map image along the stored direction and project it back onto its feasible set."""
    tid = wp.tid()
    if tid < box_capacity:
        row = tid
        wid = box_world[row]
        if is_active_row(box_local[row], wid, world_box_count, world_active, world_failed):
            mix_box_anderson_row(
                row,
                coefficient[wid],
                box_body_a,
                box_body_b,
                box_jacobian_a,
                box_jacobian_b,
                box_lower,
                box_upper,
                box_direction_image,
                inverse_weight,
                box_input,
                box_reaction,
                body_delta,
            )
        return
    cid = tid - box_capacity
    wid = contact_world[cid]
    if is_active_row(contact_local[cid], wid, world_contact_count, world_active, world_failed):
        mix_contact_anderson_row(
            cid,
            coefficient[wid],
            contact_body_a,
            contact_body_b,
            contact_jacobian_a,
            contact_jacobian_b,
            contact_friction,
            contact_direction_image,
            angular_friction,
            contact_frame,
            angular_direction_image,
            inverse_weight,
            contact_input,
            contact_reaction,
            angular_input,
            angular_reaction,
            body_delta,
        )


###
# Interfaces
###


class AndersonProjection:
    """Own one safeguarded, optionally recyclable Anderson(1) direction.

    One projection map is one complete sweep over every unilateral: a single
    mass-split Jacobi sweep, or one pass over every Gauss--Seidel color.
    :meth:`begin_map` stores the input of the first map of a projection; after
    each map, :meth:`finish_map` forms the secant between the two most recent
    map residuals, mixes the map image along the stored direction, and stores
    the mixed reactions as the input of the next map.
    """

    def __init__(self, problem_data: LOXProblemData, data: AndersonData):
        self.problem_data = problem_data
        self.device = problem_data.device
        self._data = data
        self.global_reduction = data.global_reduction
        self.world_state = AndersonWorldState()
        """Anderson state of each world and row, for the per-world projection."""
        for name in ("sample_valid", "direction_valid", "guard_pending", "previous_residual_norm", "coefficient"):
            setattr(self.world_state, name, getattr(data, name))
        for prefix, rows in (("box", data.box), ("contact", data.contact), ("angular", data.angular)):
            if rows is None:
                continue
            setattr(self.world_state, f"{prefix}_input", rows.map_input)
            for name in ("sample_residual", "sample_image", "direction_residual", "direction_image"):
                setattr(self.world_state, f"{prefix}_{name}", getattr(rows, name))

    ###
    # Public API
    ###

    def reset(self) -> None:
        """Invalidate every raw sample and recyclable direction, and clear the secant sums."""
        self._data.sample_valid.zero_()
        self._data.direction_valid.zero_()
        self._data.guard_pending.zero_()
        self._data.coefficient.zero_()
        self._data.previous_residual_norm.zero_()
        self._data.gram.zero_()
        self._data.rhs.zero_()
        self._data.residual_norm.zero_()

    def begin_outer_iteration(self, world_active: wp.array[wp.bool], *, recycle: bool) -> None:
        """Start a new fixed-right-hand-side sample sequence."""
        wp.launch(
            _begin_anderson_outer_iteration,
            dim=self.problem_data.num_worlds,
            inputs=[world_active, recycle],
            outputs=[self._data.sample_valid, self._data.direction_valid, self._data.guard_pending],
            device=self.device,
        )

    def begin_map(self, world_active: wp.array[wp.bool], world_failed: wp.array[wp.bool]) -> None:
        """Store the reaction input of one complete projection map."""
        if self._data.capacity == 0:
            return
        p = self.problem_data
        wp.launch(
            _store_map_input,
            dim=self._data.capacity,
            inputs=[
                p.box_rows.capacity,
                p.box_rows.world,
                p.box_rows.local,
                p.box_rows.world_count,
                p.contact_rows.world,
                p.contact_rows.local,
                p.contact_rows.world_count,
                world_active,
                world_failed,
                p.box_rows.reaction,
                p.contact_rows.reaction,
                p.contact_rows.angular_reaction,
            ],
            outputs=[self._data.box.map_input, self._data.contact.map_input, self._angular_arrays("map_input")],
            device=self.device,
        )

    def finish_map(
        self,
        world_active: wp.array[wp.bool],
        world_failed: wp.array[wp.bool],
        inverse_weight: wp.array[mat66f],
        body_delta: wp.array[vec6f],
    ) -> None:
        """Form one secant, solve the scalar system, and accumulate the mixed image into ``body_delta``."""
        p = self.problem_data
        rows = p.contact_rows
        if self._data.capacity > 0:
            wp.launch(
                _store_anderson_samples,
                dim=self._data.capacity,
                inputs=[
                    p.box_rows.capacity,
                    self.global_reduction,
                    p.box_rows.world,
                    p.box_rows.local,
                    p.box_rows.world_count,
                    p.contact_rows.world,
                    p.contact_rows.local,
                    p.contact_rows.world_count,
                    world_active,
                    world_failed,
                    self._data.sample_valid,
                    self._data.direction_valid,
                    p.box_rows.reaction,
                    self._data.box.map_input,
                    p.contact_rows.reaction,
                    self._data.contact.map_input,
                    rows.angular_reaction,
                    self._angular_arrays("map_input"),
                    rows.angular_friction,
                ],
                outputs=[
                    self._data.box.sample_residual,
                    self._data.box.sample_image,
                    self._data.box.direction_residual,
                    self._data.box.direction_image,
                    self._data.contact.sample_residual,
                    self._data.contact.sample_image,
                    self._data.contact.direction_residual,
                    self._data.contact.direction_image,
                    self._angular_arrays("sample_residual"),
                    self._angular_arrays("sample_image"),
                    self._angular_arrays("direction_residual"),
                    self._angular_arrays("direction_image"),
                    self._data.terms,
                    self._data.gram,
                    self._data.rhs,
                    self._data.residual_norm,
                ],
                device=self.device,
            )
        coefficient_outputs = [
            self._data.sample_valid,
            self._data.direction_valid,
            self._data.guard_pending,
            self._data.previous_residual_norm,
            self._data.coefficient,
        ]
        if self.global_reduction:
            wp.launch(
                _reduce_and_finalize_anderson_terms,
                dim=_ANDERSON_REDUCTION_BLOCK_DIM,
                block_dim=_ANDERSON_REDUCTION_BLOCK_DIM,
                inputs=[self._data.capacity, self._data.terms, world_active, world_failed],
                outputs=coefficient_outputs,
                device=self.device,
            )
        else:
            wp.launch(
                _finalize_anderson_coefficients,
                dim=p.num_worlds,
                inputs=[world_active, world_failed],
                outputs=[self._data.gram, self._data.rhs, self._data.residual_norm, *coefficient_outputs],
                device=self.device,
            )
        if self._data.capacity == 0:
            return
        wp.launch(
            _mix_anderson_reactions,
            dim=self._data.capacity,
            inputs=[
                p.box_rows.capacity,
                p.box_rows.world,
                p.box_rows.local,
                p.box_rows.world_count,
                p.box_rows.body_a,
                p.box_rows.body_b,
                p.box_rows.jacobian_a,
                p.box_rows.jacobian_b,
                p.box_rows.lower,
                p.box_rows.upper,
                self._data.box.direction_image,
                p.contact_rows.world,
                p.contact_rows.local,
                p.contact_rows.world_count,
                p.contact_rows.body_a,
                p.contact_rows.body_b,
                p.contact_rows.jacobian_a,
                p.contact_rows.jacobian_b,
                p.contact_rows.friction,
                self._data.contact.direction_image,
                rows.angular_friction,
                rows.frame,
                self._angular_arrays("direction_image"),
                world_active,
                self._data.coefficient,
                inverse_weight,
            ],
            outputs=[
                self._data.box.map_input,
                p.box_rows.reaction,
                self._data.contact.map_input,
                p.contact_rows.reaction,
                self._angular_arrays("map_input"),
                rows.angular_reaction,
                world_failed,
                body_delta,
            ],
            device=self.device,
        )

    ###
    # Internals
    ###

    def _angular_arrays(self, name: str):
        return getattr(self._data.angular, name) if self._data.angular is not None else None
