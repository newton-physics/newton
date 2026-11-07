# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Row updates and kernels of the LOX unilateral projection.

The unilateral rows come in two families that share the same per-world layout:

- box rows (joint friction and joint limits) with ``lower <= lambda <= upper``;
- contacts, with a normal-last ``(tangent_x, tangent_y, normal_z)`` Coulomb cone.

A row update solves one row against the current projected twist ``p`` and
accumulates ``W^-1 J^T delta_lambda`` into a per-body delta. Bodies shared by
``m`` rows of the same sweep scale their inverse weight by ``m`` (mass
splitting), where ``m = occupancy[body, color]``. Jacobi uses one color and the
body incidence counts; colored Gauss--Seidel uses one column per color.

Each row couples bodies A and B (``body_a``/``body_b``, ``-1`` when absent),
Kamino's base and follower bodies for joint rows and bodies A and B for
contacts. Body twists and Jacobian columns use Kamino's linear-first 6D
convention. See :mod:`.projection` for the schedules that launch these kernels.
"""

from __future__ import annotations

from functools import cache

import warp as wp

from ...core.types import mat36f, mat66f, vec6f
from .contact import (
    SpatialContactStatus,
    angular_contact_reaction,
    clamp_contact_reaction,
    compute_contact_delassus_scale,
    compute_contact_restitution_target,
    compute_contact_scaled_alart_curnier_residual,
    compute_spatial_contact_residual,
    linear_contact_reaction,
    mat55f,
    pack_contact_reaction,
    pack_spatial_friction,
    prepare_spatial_contact,
    scatter_spatial_impulse,
    solve_contact_coulomb_newton,
    solve_spatial_contact,
    spatial_body_velocity,
    spatial_body_wrench,
    spatial_contact_delassus,
    spatial_contact_velocity,
    spatial_contact_velocity_scale,
    spatial_friction_scaling,
    vec5f,
)
from .solver_kernels import atomic_max_nonnegative, compute_body_iteration_residuals
from .system_kernels import ProjectionWeights, apply_body_weight

###
# Module interface
###

__all__ = [
    "_apply_twist_delta",
    "_compute_box_residuals",
    "_prepare_box_blocks",
    "_prepare_contact_blocks",
    "_prepare_physical_spatial_contacts",
    "_prepare_spatial_contacts",
    "_project_box_row",
    "_project_contact_row",
    "_project_spatial_contact_row",
    "_warmstart_box_rows",
    "_warmstart_contacts",
    "_warmstart_spatial_contacts",
    "is_active_row",
    "make_contact_residuals_kernel",
    "make_iteration_residuals_kernel",
    "make_sweep_kernel",
    "row_color",
    "scatter_box_impulse",
    "scatter_contact_impulse",
]

###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Constants
###


_CONTACT_NEGATIVE_EIGENVALUE_TOLERANCE = 1.0e-5
"""Relative tolerance below zero on the smallest eigenvalue of a contact block, before the floor lifts it."""


_CONTACT_EIGENVALUE_FLOOR = 1.0e-6
"""Relative eigenvalue floor used to regularize nearly singular contact blocks."""


_UNRESOLVED_RESIDUAL = 1.0e30
"""Residual reported for a spatial contact whose residual is not finite."""


###
# Functions
###


@wp.func
def _compute_box_delassus(
    jacobian_a: vec6f,
    response_a: vec6f,
    scale_a: wp.float32,
    jacobian_b: vec6f,
    response_b: vec6f,
    scale_b: wp.float32,
) -> wp.float32:
    """Compute the scalar local Delassus coefficient ``J W^-1 J^T`` of a box row from its twist responses.

    Each body side is scaled by its mass-split factor, one for the unsplit coefficient.
    """
    return scale_a * wp.dot(jacobian_a, response_a) + scale_b * wp.dot(jacobian_b, response_b)


@wp.func
def _compute_contact_delassus(
    jacobian_a: mat36f,
    response_a: mat36f,
    scale_a: wp.float32,
    jacobian_b: mat36f,
    response_b: mat36f,
    scale_b: wp.float32,
) -> wp.mat33f:
    """Compute the ``3 x 3`` local Delassus block ``J W^-1 J^T`` of a contact from its transposed twist responses.

    Each body side is scaled by its mass-split factor, one for the unsplit block. Each pair of rows is
    evaluated once, so the block is exactly symmetric.
    """
    delassus = wp.mat33f(0.0)
    for row in range(3):
        for col in range(3):
            if col >= row:
                value = scale_a * wp.dot(response_a[row], jacobian_a[col]) + scale_b * wp.dot(
                    response_b[row], jacobian_b[col]
                )
                delassus[row, col] = value
                delassus[col, row] = value
    return delassus


@wp.func
def _regularize_contact_delassus(delassus: wp.mat33f, bias: wp.vec3f, friction: wp.float32):
    """Lift the eigenvalues of a contact block below a relative floor.

    The block must be exactly symmetric, as :func:`_compute_contact_delassus` assembles it.

    Returns:
        The positive-definite block and whether the inputs were valid.
    """
    if not wp.isfinite(bias) or not wp.isfinite(friction) or friction < 0.0 or not wp.isfinite(delassus):
        return delassus, False

    scale = wp.float32(0.0)
    for row in range(3):
        for col in range(3):
            scale = wp.max(scale, wp.abs(delassus[row, col]))
    if scale <= 0.0:
        return delassus, False

    eigenvectors, eigenvalues = wp.eig3(delassus)
    if not wp.isfinite(eigenvalues):
        return delassus, False
    minimum_eigenvalue = wp.min(eigenvalues)
    if minimum_eigenvalue < -_CONTACT_NEGATIVE_EIGENVALUE_TOLERANCE * scale:
        return delassus, False

    eigenvalue_floor = _CONTACT_EIGENVALUE_FLOOR * scale
    if minimum_eigenvalue < eigenvalue_floor:
        clamped = wp.max(eigenvalues, wp.vec3f(eigenvalue_floor))
        return eigenvectors @ wp.diag(clamped) @ wp.transpose(eigenvectors), True
    return delassus, True


@wp.func
def _project_box_local(
    velocity: wp.float32,
    reaction: wp.float32,
    delassus: wp.float32,
    lower: wp.float32,
    upper: wp.float32,
):
    """Solve one box row exactly: ``lambda = clamp(lambda - v / d, lower, upper)``."""
    free_velocity = velocity - delassus * reaction
    return wp.clamp(-free_velocity / delassus, lower, upper)


@wp.func
def _project_contact_local(
    velocity: wp.vec3f,
    reaction: wp.vec3f,
    delassus: wp.mat33f,
    friction: wp.float32,
):
    """Solve one normal-last Coulomb contact exactly for its local block."""
    return solve_contact_coulomb_newton(delassus, velocity - delassus @ reaction, friction)


@wp.func
def _box_row_velocity(
    row: wp.int32,
    bid_a: wp.int32,
    bid_b: wp.int32,
    jacobian_a: wp.array[vec6f],
    jacobian_b: wp.array[vec6f],
    twist: wp.array[vec6f],
) -> wp.float32:
    """Return ``J v`` of one box row, without its bias."""
    velocity = wp.float32(0.0)
    if bid_a >= 0:
        velocity += wp.dot(jacobian_a[row], twist[bid_a])
    if bid_b >= 0:
        velocity += wp.dot(jacobian_b[row], twist[bid_b])
    return velocity


@wp.func
def _contact_row_velocity(
    cid: wp.int32,
    bid_a: wp.int32,
    bid_b: wp.int32,
    jacobian_a: wp.array[mat36f],
    jacobian_b: wp.array[mat36f],
    twist: wp.array[vec6f],
) -> wp.vec3f:
    """Return ``J v`` of one contact, without its bias."""
    velocity = wp.vec3f(0.0)
    if bid_a >= 0:
        velocity += jacobian_a[cid] @ twist[bid_a]
    if bid_b >= 0:
        velocity += jacobian_b[cid] @ twist[bid_b]
    return velocity


@wp.func
def _contact_normal_compliance(cid: wp.int32, normal_compliance: wp.array[wp.float32]) -> wp.float32:
    """Return the impulse-space normal compliance [1/kg] of one contact, or zero for hard contacts."""
    if normal_compliance:
        return normal_compliance[cid]
    return 0.0


@wp.func
def _contact_restitution_target(
    cid: wp.int32,
    free_normal_velocity: wp.float32,
    restitution: wp.array[wp.vec4f],
) -> wp.float32:
    """Return the in-kernel restitution target of one contact, or zero without in-kernel restitution.

    Args:
        cid: Contact row.
        free_normal_velocity: Mechanical reaction-free normal velocity, without bias [m/s].
        restitution: Restitution inputs ``(gap, begin-step normal velocity, coefficient, dt)`` of each contact.
    """
    if restitution:
        parameters = restitution[cid]
        return compute_contact_restitution_target(
            parameters[0], parameters[1], free_normal_velocity, parameters[2], parameters[3]
        )
    return 0.0


@wp.func
def _contact_law_velocity(
    cid: wp.int32,
    velocity: wp.vec3f,
    reaction: wp.vec3f,
    delassus: wp.mat33f,
    bias: wp.array[wp.vec3f],
    normal_compliance: wp.array[wp.float32],
    restitution: wp.array[wp.vec4f],
) -> wp.vec3f:
    """Add the compliance and in-kernel restitution terms to a biased contact velocity.

    Args:
        cid: Contact row.
        velocity: Biased contact velocity ``J v + bias``.
        reaction: Current normal-last contact impulse.
        delassus: Local Delassus block, including the normal compliance.
        bias: Velocity bias of each contact.
        normal_compliance: Impulse-space normal compliance [1/kg] of each contact, or ``None``.
        restitution: In-kernel restitution inputs of each contact, or ``None``.

    Returns:
        The velocity whose hard Coulomb law is the contact law of the row.
    """
    result = velocity
    # The compliance acts on the reaction through the solve metric only.
    result[2] += _contact_normal_compliance(cid, normal_compliance) * reaction[2]
    if restitution:
        free_normal = result[2] - bias[cid][2] - (delassus @ reaction)[2]
        result[2] -= _contact_restitution_target(cid, free_normal, restitution)
    return result


@wp.func
def _split_factor(bid: wp.int32, color: wp.int32, occupancy: wp.array2d[wp.int32]) -> wp.float32:
    """Return the mass-split factor ``m`` of a body, the number of its rows in the color, or zero if absent."""
    if bid < 0:
        return 0.0
    return wp.float32(wp.max(1, occupancy[bid, color]))


@wp.func
def _split_inverse_weight(
    bid: wp.int32,
    color: wp.int32,
    inverse_weight: wp.array[mat66f],
    occupancy: wp.array2d[wp.int32],
) -> mat66f:
    """Return the mass-split inverse weight ``m W^-1`` of a body, or zero if absent."""
    if bid < 0:
        return mat66f(0.0)
    return _split_factor(bid, color, occupancy) * inverse_weight[bid]


@wp.func
def _accumulate_body_delta(
    bid: wp.int32,
    color: wp.int32,
    delta: vec6f,
    occupancy: wp.array2d[wp.int32],
    twist_delta: wp.array[vec6f],
):
    # A plain add suffices for a body touched by a single row of this sweep.
    if occupancy[bid, color] == 1:
        twist_delta[bid] += delta
    else:
        wp.atomic_add(twist_delta, bid, delta)


@wp.func
def _project_box_row(
    row: wp.int32,
    color: wp.int32,
    body_a: wp.array[wp.int32],
    body_b: wp.array[wp.int32],
    jacobian_a: wp.array[vec6f],
    jacobian_b: wp.array[vec6f],
    bias: wp.array[wp.float32],
    lower: wp.array[wp.float32],
    upper: wp.array[wp.float32],
    delassus: wp.array[wp.float32],
    reaction: wp.array[wp.float32],
    weights: ProjectionWeights,
    occupancy: wp.array2d[wp.int32],
    projected_twist: wp.array[vec6f],
    twist_delta: wp.array[vec6f],
):
    """Project one box row and accumulate its weighted body correction.

    A non-finite reaction reaches the body twists, which fail the world.
    """
    bid_a = body_a[row]
    bid_b = body_b[row]
    if bid_a < 0 and bid_b < 0:
        reaction[row] = 0.0
        return
    # Each Jacobian is read once, for both the row velocity and the body correction
    row_jacobian_a = vec6f(0.0)
    row_jacobian_b = vec6f(0.0)
    velocity = wp.float32(0.0)
    if bid_a >= 0:
        row_jacobian_a = jacobian_a[row]
        velocity += wp.dot(row_jacobian_a, projected_twist[bid_a])
    if bid_b >= 0:
        row_jacobian_b = jacobian_b[row]
        velocity += wp.dot(row_jacobian_b, projected_twist[bid_b])
    velocity = bias[row] + velocity
    reaction_old = reaction[row]
    reaction_new = _project_box_local(velocity, reaction_old, delassus[row], lower[row], upper[row])
    reaction[row] = reaction_new
    delta = reaction_new - reaction_old
    if delta == 0.0:
        return
    if bid_a >= 0:
        correction = weights.box_response[row, 0] * delta
        _accumulate_body_delta(bid_a, color, correction, occupancy, twist_delta)
    if bid_b >= 0:
        correction = weights.box_response[row, 1] * delta
        _accumulate_body_delta(bid_b, color, correction, occupancy, twist_delta)


@wp.func
def _is_zero_vec3(value: wp.vec3f) -> wp.bool:
    return value[0] == 0.0 and value[1] == 0.0 and value[2] == 0.0


@wp.func
def _project_contact_row(
    cid: wp.int32,
    color: wp.int32,
    body_a: wp.array[wp.int32],
    body_b: wp.array[wp.int32],
    jacobian_a: wp.array[mat36f],
    jacobian_b: wp.array[mat36f],
    bias: wp.array[wp.vec3f],
    friction: wp.array[wp.float32],
    delassus: wp.array[wp.mat33f],
    normal_compliance: wp.array[wp.float32],
    restitution: wp.array[wp.vec4f],
    reaction: wp.array[wp.vec3f],
    weights: ProjectionWeights,
    occupancy: wp.array2d[wp.int32],
    projected_twist: wp.array[vec6f],
    twist_delta: wp.array[vec6f],
):
    """Project one contact and accumulate its weighted body correction.

    The local block ``delassus`` includes ``normal_compliance`` on its normal diagonal;
    ``restitution`` enables the trial-dependent in-kernel restitution target. A non-finite
    reaction reaches the body twists, which fail the world.
    """
    bid_a = body_a[cid]
    bid_b = body_b[cid]
    if bid_a < 0 and bid_b < 0:
        reaction[cid] = wp.vec3f(0.0)
        return
    # Each Jacobian is read once, for both the contact velocity and the body correction
    contact_jacobian_a = mat36f(0.0)
    contact_jacobian_b = mat36f(0.0)
    velocity = wp.vec3f(0.0)
    if bid_a >= 0:
        contact_jacobian_a = jacobian_a[cid]
        velocity += contact_jacobian_a @ projected_twist[bid_a]
    if bid_b >= 0:
        contact_jacobian_b = jacobian_b[cid]
        velocity += contact_jacobian_b @ projected_twist[bid_b]
    velocity = bias[cid] + velocity
    reaction_old = reaction[cid]
    local_delassus = delassus[cid]
    velocity = _contact_law_velocity(cid, velocity, reaction_old, local_delassus, bias, normal_compliance, restitution)
    # An unloaded contact that is not approaching stays unloaded.
    if _is_zero_vec3(reaction_old) and velocity[2] >= 0.0:
        return
    reaction_new = _project_contact_local(velocity, reaction_old, local_delassus, friction[cid])
    reaction[cid] = reaction_new
    delta = reaction_new - reaction_old
    if _is_zero_vec3(delta):
        return
    if bid_a >= 0:
        correction = wp.transpose(weights.contact_response[cid, 0]) @ delta
        _accumulate_body_delta(bid_a, color, correction, occupancy, twist_delta)
    if bid_b >= 0:
        correction = wp.transpose(weights.contact_response[cid, 1]) @ delta
        _accumulate_body_delta(bid_b, color, correction, occupancy, twist_delta)


@wp.func
def _normal_target_bias(
    cid: wp.int32,
    free_normal_velocity: wp.float32,
    bias: wp.array[wp.vec3f],
    restitution: wp.array[wp.vec4f],
) -> wp.float32:
    """Return the normal velocity bias, including the in-kernel restitution target."""
    return bias[cid][2] - _contact_restitution_target(cid, free_normal_velocity, restitution)


@wp.func
def _project_spatial_contact_row(
    cid: wp.int32,
    wid: wp.int32,
    color: wp.int32,
    body_a: wp.array[wp.int32],
    body_b: wp.array[wp.int32],
    jacobian_a: wp.array[mat36f],
    jacobian_b: wp.array[mat36f],
    reaction: wp.array[wp.vec3f],
    frame: wp.array[wp.mat33f],
    friction: wp.array[wp.float32],
    angular_friction: wp.array[wp.vec2f],
    bias: wp.array[wp.vec3f],
    normal_compliance: wp.array[wp.float32],
    restitution: wp.array[wp.vec4f],
    delassus: wp.array[mat66f],
    eigenvectors: wp.array[mat55f],
    eigenvalues: wp.array[vec5f],
    angular_reaction: wp.array[wp.vec3f],
    weights: ProjectionWeights,
    occupancy: wp.array2d[wp.int32],
    projected_twist: wp.array[vec6f],
    twist_delta: wp.array[vec6f],
    world_failed: wp.array[wp.bool],
):
    """Project one spatial contact and accumulate its weighted body correction."""
    bid_a = body_a[cid]
    bid_b = body_b[cid]
    if bid_a < 0 and bid_b < 0:
        reaction[cid] = wp.vec3f(0.0)
        return
    old = pack_contact_reaction(reaction[cid], angular_reaction[cid])
    metric = delassus[cid]
    # The compliance acts on the reaction only through the solve metric.
    mechanical = metric
    mechanical[0, 0] -= _contact_normal_compliance(cid, normal_compliance)
    # The frame and each Jacobian are read once, for both the contact velocity and the body corrections
    contact_frame = frame[cid]
    contact_jacobian_a = mat36f(0.0)
    contact_jacobian_b = mat36f(0.0)
    velocity = vec6f(0.0)
    if bid_a >= 0:
        contact_jacobian_a = jacobian_a[cid]
        velocity += spatial_body_velocity(contact_jacobian_a, contact_frame, -1.0, projected_twist[bid_a])
    if bid_b >= 0:
        contact_jacobian_b = jacobian_b[cid]
        velocity += spatial_body_velocity(contact_jacobian_b, contact_frame, 1.0, projected_twist[bid_b])
    free_velocity = velocity - mechanical @ old
    free_velocity[0] += _normal_target_bias(cid, free_velocity[0], bias, restitution)
    reaction_new, status = solve_spatial_contact(
        metric,
        eigenvectors[cid],
        eigenvalues[cid],
        pack_spatial_friction(cid, friction, angular_friction),
        free_velocity,
    )
    if status != SpatialContactStatus.SOLVED:
        world_failed[wid] = True
        return
    reaction[cid] = linear_contact_reaction(reaction_new)
    angular_reaction[cid] = angular_contact_reaction(reaction_new)
    delta = reaction_new - old
    if wp.length_sq(delta) == 0.0:
        return
    if bid_a >= 0:
        wrench = spatial_body_wrench(contact_jacobian_a, contact_frame, -1.0, delta)
        _accumulate_body_delta(
            bid_a, color, apply_body_weight(weights.inverse_weight[bid_a], wrench), occupancy, twist_delta
        )
    if bid_b >= 0:
        wrench = spatial_body_wrench(contact_jacobian_b, contact_frame, 1.0, delta)
        _accumulate_body_delta(
            bid_b, color, apply_body_weight(weights.inverse_weight[bid_b], wrench), occupancy, twist_delta
        )


@wp.func
def row_color(row: wp.int32, row_colors: wp.array[wp.int32]) -> wp.int32:
    if row_colors:
        return row_colors[row]
    return 0


@wp.func
def is_active_row(
    local: wp.int32,
    wid: wp.int32,
    world_row_count: wp.array[wp.int32],
    world_active: wp.array[wp.bool],
    world_failed: wp.array[wp.bool],
) -> wp.bool:
    """Return whether a row is in use by an active, valid world."""
    return local < world_row_count[wid] and world_active[wid] and not world_failed[wid]


@wp.func
def scatter_box_impulse(
    row: wp.int32,
    body_a: wp.array[wp.int32],
    body_b: wp.array[wp.int32],
    jacobian_a: wp.array[vec6f],
    jacobian_b: wp.array[vec6f],
    impulse: wp.float32,
    weights: ProjectionWeights,
    twist_delta: wp.array[vec6f],
):
    """Accumulate the twist change ``W^-1 J^T impulse`` of one box row from its prepared response."""
    if impulse == 0.0:
        return
    bid_a = body_a[row]
    bid_b = body_b[row]
    if bid_a >= 0:
        wp.atomic_add(twist_delta, bid_a, weights.box_response[row, 0] * impulse)
    if bid_b >= 0:
        wp.atomic_add(twist_delta, bid_b, weights.box_response[row, 1] * impulse)


@wp.func
def scatter_contact_impulse(
    cid: wp.int32,
    body_a: wp.array[wp.int32],
    body_b: wp.array[wp.int32],
    jacobian_a: wp.array[mat36f],
    jacobian_b: wp.array[mat36f],
    impulse: wp.vec3f,
    weights: ProjectionWeights,
    twist_delta: wp.array[vec6f],
):
    """Accumulate the twist change ``W^-1 J^T impulse`` of one contact from its prepared response."""
    if _is_zero_vec3(impulse):
        return
    bid_a = body_a[cid]
    bid_b = body_b[cid]
    if bid_a >= 0:
        wp.atomic_add(twist_delta, bid_a, wp.transpose(weights.contact_response[cid, 0]) @ impulse)
    if bid_b >= 0:
        wp.atomic_add(twist_delta, bid_b, wp.transpose(weights.contact_response[cid, 1]) @ impulse)


###
# Kernels
###


@wp.func
def _split_box_delassus(
    jacobian_a: vec6f,
    response_a: vec6f,
    bid_a: wp.int32,
    jacobian_b: vec6f,
    response_b: vec6f,
    bid_b: wp.int32,
    color: wp.int32,
    occupancy: wp.array2d[wp.int32],
):
    """Return the mass-split Delassus coefficient of a box row and whether it is valid, zero without bodies."""
    if bid_a < 0 and bid_b < 0:
        return wp.float32(0.0), True
    value = _compute_box_delassus(
        jacobian_a,
        response_a,
        _split_factor(bid_a, color, occupancy),
        jacobian_b,
        response_b,
        _split_factor(bid_b, color, occupancy),
    )
    return value, wp.isfinite(value) and value > 0.0


@wp.func
def _split_contact_delassus(
    cid: wp.int32,
    jacobian_a: mat36f,
    response_a: mat36f,
    bid_a: wp.int32,
    jacobian_b: mat36f,
    response_b: mat36f,
    bid_b: wp.int32,
    bias: wp.vec3f,
    friction: wp.float32,
    compliance: wp.array[wp.float32],
    color: wp.int32,
    occupancy: wp.array2d[wp.int32],
):
    """Return the regularized mass-split Delassus block of a contact and whether it is valid, zero without bodies.

    A non-finite Jacobian or response makes the block non-finite, which is invalid.
    """
    if bid_a < 0 and bid_b < 0:
        return wp.mat33f(0.0), True
    delassus, valid = _regularize_contact_delassus(
        _compute_contact_delassus(
            jacobian_a,
            response_a,
            _split_factor(bid_a, color, occupancy),
            jacobian_b,
            response_b,
            _split_factor(bid_b, color, occupancy),
        ),
        bias,
        friction,
    )
    delassus[2, 2] += _contact_normal_compliance(cid, compliance)
    return delassus, valid


@wp.kernel
def _prepare_box_blocks(
    # Inputs:
    box_world: wp.array[wp.int32],
    box_local: wp.array[wp.int32],
    world_box_count: wp.array[wp.int32],
    box_color: wp.array[wp.int32],
    box_body_a: wp.array[wp.int32],
    box_body_b: wp.array[wp.int32],
    box_jacobian_a: wp.array[vec6f],
    box_jacobian_b: wp.array[vec6f],
    inverse_weight: wp.array[mat66f],
    occupancy: wp.array2d[wp.int32],
    smoothing_occupancy: wp.array2d[wp.int32],
    # Outputs:
    box_response: wp.array2d[vec6f],
    box_delassus: wp.array[wp.float32],
    box_smoothing_delassus: wp.array[wp.float32],
    physical_delassus: wp.array[wp.float32],
    world_failed: wp.array[wp.bool],
):
    """Prepare the twist response ``W^-1 J^T`` and the Delassus coefficients of each box row.

    The split coefficient divides each body over the rows of the row color in ``occupancy``, the
    optional smoothing coefficient over all rows of the body in ``smoothing_occupancy``, and the
    unsplit coefficient scales the residuals. The response of a missing body is zero.
    """
    row = wp.tid()
    wid = box_world[row]
    if box_local[row] >= world_box_count[wid]:
        return
    bid_a = box_body_a[row]
    bid_b = box_body_b[row]
    jacobian_a = box_jacobian_a[row]
    jacobian_b = box_jacobian_b[row]
    response_a = vec6f(0.0)
    response_b = vec6f(0.0)
    if bid_a >= 0:
        response_a = apply_body_weight(inverse_weight[bid_a], jacobian_a)
    if bid_b >= 0:
        response_b = apply_body_weight(inverse_weight[bid_b], jacobian_b)
    box_response[row, 0] = response_a
    box_response[row, 1] = response_b
    physical_delassus[row] = _compute_box_delassus(jacobian_a, response_a, 1.0, jacobian_b, response_b, 1.0)
    value, valid = _split_box_delassus(
        jacobian_a, response_a, bid_a, jacobian_b, response_b, bid_b, row_color(row, box_color), occupancy
    )
    box_delassus[row] = value
    if box_smoothing_delassus:
        smoothing_value, smoothing_valid = _split_box_delassus(
            jacobian_a, response_a, bid_a, jacobian_b, response_b, bid_b, 0, smoothing_occupancy
        )
        box_smoothing_delassus[row] = smoothing_value
        valid = valid and smoothing_valid
    if not valid:
        world_failed[wid] = True


@wp.kernel
def _prepare_contact_blocks(
    # Inputs:
    contact_world: wp.array[wp.int32],
    contact_local: wp.array[wp.int32],
    world_contact_count: wp.array[wp.int32],
    contact_color: wp.array[wp.int32],
    contact_body_a: wp.array[wp.int32],
    contact_body_b: wp.array[wp.int32],
    contact_jacobian_a: wp.array[mat36f],
    contact_jacobian_b: wp.array[mat36f],
    contact_bias: wp.array[wp.vec3f],
    contact_friction: wp.array[wp.float32],
    contact_compliance: wp.array[wp.float32],
    inverse_weight: wp.array[mat66f],
    occupancy: wp.array2d[wp.int32],
    smoothing_occupancy: wp.array2d[wp.int32],
    # Outputs:
    contact_response: wp.array2d[mat36f],
    contact_delassus: wp.array[wp.mat33f],
    contact_smoothing_delassus: wp.array[wp.mat33f],
    physical_delassus: wp.array[wp.mat33f],
    world_failed: wp.array[wp.bool],
):
    """Prepare the transposed twist response ``J W^-1`` and the compliant Delassus blocks of each contact.

    The split block divides each body over the rows of the contact color in ``occupancy``, the
    optional smoothing block over all rows of the body in ``smoothing_occupancy``, and the unsplit
    block scales the residuals. The response of a missing body is zero.
    """
    cid = wp.tid()
    wid = contact_world[cid]
    if contact_local[cid] >= world_contact_count[wid]:
        return
    bid_a = contact_body_a[cid]
    bid_b = contact_body_b[cid]
    jacobian_a = contact_jacobian_a[cid]
    jacobian_b = contact_jacobian_b[cid]
    response_a = mat36f(0.0)
    response_b = mat36f(0.0)
    for row in range(3):
        if bid_a >= 0:
            response_a[row] = apply_body_weight(inverse_weight[bid_a], jacobian_a[row])
        if bid_b >= 0:
            response_b[row] = apply_body_weight(inverse_weight[bid_b], jacobian_b[row])
    contact_response[cid, 0] = response_a
    contact_response[cid, 1] = response_b
    physical = _compute_contact_delassus(jacobian_a, response_a, 1.0, jacobian_b, response_b, 1.0)
    physical[2, 2] += _contact_normal_compliance(cid, contact_compliance)
    physical_delassus[cid] = physical
    bias = contact_bias[cid]
    friction = contact_friction[cid]
    delassus, valid = _split_contact_delassus(
        cid,
        jacobian_a,
        response_a,
        bid_a,
        jacobian_b,
        response_b,
        bid_b,
        bias,
        friction,
        contact_compliance,
        row_color(cid, contact_color),
        occupancy,
    )
    contact_delassus[cid] = delassus
    if contact_smoothing_delassus:
        smoothing_delassus, smoothing_valid = _split_contact_delassus(
            cid,
            jacobian_a,
            response_a,
            bid_a,
            jacobian_b,
            response_b,
            bid_b,
            bias,
            friction,
            contact_compliance,
            0,
            smoothing_occupancy,
        )
        contact_smoothing_delassus[cid] = smoothing_delassus
        valid = valid and smoothing_valid
    if not valid:
        world_failed[wid] = True


@wp.kernel
def _prepare_spatial_contacts(
    # Inputs:
    contact_world: wp.array[wp.int32],
    contact_local: wp.array[wp.int32],
    world_contact_count: wp.array[wp.int32],
    contact_color: wp.array[wp.int32],
    contact_body_a: wp.array[wp.int32],
    contact_body_b: wp.array[wp.int32],
    contact_jacobian_a: wp.array[mat36f],
    contact_jacobian_b: wp.array[mat36f],
    frame: wp.array[wp.mat33f],
    friction: wp.array[wp.float32],
    angular_friction: wp.array[wp.vec2f],
    normal_compliance: wp.array[wp.float32],
    occupancy: wp.array2d[wp.int32],
    inverse_weight: wp.array[mat66f],
    # Outputs:
    delassus: wp.array[mat66f],
    eigenvectors: wp.array[mat55f],
    eigenvalues: wp.array[vec5f],
    world_failed: wp.array[wp.bool],
):
    """Prepare the mass-split compliant metric and its Schur spectrum."""
    cid = wp.tid()
    wid = contact_world[cid]
    if contact_local[cid] >= world_contact_count[wid]:
        return
    bid_a = contact_body_a[cid]
    bid_b = contact_body_b[cid]
    color = wp.int32(0)
    if contact_color:
        color = contact_color[cid]
    metric = spatial_contact_delassus(
        cid,
        bid_a,
        bid_b,
        contact_jacobian_a,
        contact_jacobian_b,
        frame,
        _split_inverse_weight(bid_a, color, inverse_weight, occupancy),
        _split_inverse_weight(bid_b, color, inverse_weight, occupancy),
    )
    metric[0, 0] += _contact_normal_compliance(cid, normal_compliance)
    delassus[cid] = metric
    contact_eigenvectors, contact_eigenvalues, valid = prepare_spatial_contact(
        metric, pack_spatial_friction(cid, friction, angular_friction)
    )
    eigenvectors[cid] = contact_eigenvectors
    eigenvalues[cid] = contact_eigenvalues
    if not valid:
        world_failed[wid] = True


@wp.kernel
def _prepare_physical_spatial_contacts(
    # Inputs:
    contact_world: wp.array[wp.int32],
    contact_local: wp.array[wp.int32],
    world_contact_count: wp.array[wp.int32],
    contact_body_a: wp.array[wp.int32],
    contact_body_b: wp.array[wp.int32],
    contact_jacobian_a: wp.array[mat36f],
    contact_jacobian_b: wp.array[mat36f],
    frame: wp.array[wp.mat33f],
    normal_compliance: wp.array[wp.float32],
    inverse_weight: wp.array[mat66f],
    # Outputs:
    physical_delassus: wp.array[mat66f],
):
    """Prepare the unsplit compliant metric used by the residuals and convergence checks."""
    cid = wp.tid()
    if contact_local[cid] >= world_contact_count[contact_world[cid]]:
        return
    bid_a = contact_body_a[cid]
    bid_b = contact_body_b[cid]
    inverse_weight_a = mat66f(0.0)
    inverse_weight_b = mat66f(0.0)
    if bid_a >= 0:
        inverse_weight_a = inverse_weight[bid_a]
    if bid_b >= 0:
        inverse_weight_b = inverse_weight[bid_b]
    metric = spatial_contact_delassus(
        cid,
        bid_a,
        bid_b,
        contact_jacobian_a,
        contact_jacobian_b,
        frame,
        inverse_weight_a,
        inverse_weight_b,
    )
    metric[0, 0] += _contact_normal_compliance(cid, normal_compliance)
    physical_delassus[cid] = metric


@wp.kernel
def _warmstart_box_rows(
    # Inputs:
    box_world: wp.array[wp.int32],
    box_local: wp.array[wp.int32],
    world_box_count: wp.array[wp.int32],
    box_body_a: wp.array[wp.int32],
    box_body_b: wp.array[wp.int32],
    box_jacobian_a: wp.array[vec6f],
    box_jacobian_b: wp.array[vec6f],
    box_reaction: wp.array[wp.float32],
    world_active: wp.array[wp.bool],
    world_failed: wp.array[wp.bool],
    weights: ProjectionWeights,
    # Outputs:
    twist_delta: wp.array[vec6f],
):
    row = wp.tid()
    wid = box_world[row]
    if box_local[row] >= world_box_count[wid] or not world_active[wid] or world_failed[wid]:
        return
    scatter_box_impulse(
        row,
        box_body_a,
        box_body_b,
        box_jacobian_a,
        box_jacobian_b,
        box_reaction[row],
        weights,
        twist_delta,
    )


@wp.kernel
def _warmstart_contacts(
    # Inputs:
    contact_world: wp.array[wp.int32],
    contact_local: wp.array[wp.int32],
    world_contact_count: wp.array[wp.int32],
    contact_body_a: wp.array[wp.int32],
    contact_body_b: wp.array[wp.int32],
    contact_jacobian_a: wp.array[mat36f],
    contact_jacobian_b: wp.array[mat36f],
    contact_reaction: wp.array[wp.vec3f],
    world_active: wp.array[wp.bool],
    world_failed: wp.array[wp.bool],
    weights: ProjectionWeights,
    # Outputs:
    twist_delta: wp.array[vec6f],
):
    cid = wp.tid()
    wid = contact_world[cid]
    if contact_local[cid] >= world_contact_count[wid] or not world_active[wid] or world_failed[wid]:
        return
    impulse = contact_reaction[cid]
    if _is_zero_vec3(impulse):
        return
    scatter_contact_impulse(
        cid,
        contact_body_a,
        contact_body_b,
        contact_jacobian_a,
        contact_jacobian_b,
        impulse,
        weights,
        twist_delta,
    )


@wp.kernel
def _warmstart_spatial_contacts(
    # Inputs:
    contact_world: wp.array[wp.int32],
    contact_local: wp.array[wp.int32],
    world_contact_count: wp.array[wp.int32],
    contact_body_a: wp.array[wp.int32],
    contact_body_b: wp.array[wp.int32],
    contact_jacobian_a: wp.array[mat36f],
    contact_jacobian_b: wp.array[mat36f],
    frame: wp.array[wp.mat33f],
    friction: wp.array[wp.float32],
    angular_friction: wp.array[wp.vec2f],
    world_active: wp.array[wp.bool],
    world_failed: wp.array[wp.bool],
    weights: ProjectionWeights,
    # Outputs:
    contact_reaction: wp.array[wp.vec3f],
    angular_reaction: wp.array[wp.vec3f],
    twist_delta: wp.array[vec6f],
):
    """Clamp the warm-start reactions into the cone and apply them to the bodies."""
    cid = wp.tid()
    wid = contact_world[cid]
    if contact_local[cid] >= world_contact_count[wid] or not world_active[wid] or world_failed[wid]:
        return
    reaction = clamp_contact_reaction(
        pack_contact_reaction(contact_reaction[cid], angular_reaction[cid]),
        pack_spatial_friction(cid, friction, angular_friction),
    )
    contact_reaction[cid] = linear_contact_reaction(reaction)
    angular_reaction[cid] = angular_contact_reaction(reaction)
    scatter_spatial_impulse(
        cid,
        contact_body_a,
        contact_body_b,
        contact_jacobian_a,
        contact_jacobian_b,
        frame,
        reaction,
        weights,
        twist_delta,
    )


@wp.kernel
def _apply_twist_delta(
    # Inputs:
    body_world: wp.array[wp.int32],
    world_active: wp.array[wp.bool],
    world_failed: wp.array[wp.bool],
    # Outputs:
    twist_delta: wp.array[vec6f],
    projected_twist: wp.array[vec6f],
):
    bid = wp.tid()
    wid = body_world[bid]
    if world_active[wid] and not world_failed[wid]:
        projected_twist[bid] += twist_delta[bid]
    twist_delta[bid] = vec6f(0.0)


@cache
def make_sweep_kernel(spatial: bool):
    """Build the sweep kernel, projecting contacts as spatial contacts with angular friction."""

    @wp.kernel(module="unique", enable_backward=False)
    def sweep(
        # Inputs:
        worker_count: wp.int32,
        color: wp.int32,
        box_order: wp.array[wp.int32],
        box_color_offset: wp.array[wp.int32],
        box_color_count: wp.array[wp.int32],
        box_world: wp.array[wp.int32],
        box_local: wp.array[wp.int32],
        world_box_count: wp.array[wp.int32],
        box_body_a: wp.array[wp.int32],
        box_body_b: wp.array[wp.int32],
        box_jacobian_a: wp.array[vec6f],
        box_jacobian_b: wp.array[vec6f],
        box_bias: wp.array[wp.float32],
        box_lower: wp.array[wp.float32],
        box_upper: wp.array[wp.float32],
        box_delassus: wp.array[wp.float32],
        contact_order: wp.array[wp.int32],
        contact_color_offset: wp.array[wp.int32],
        contact_color_count: wp.array[wp.int32],
        contact_world: wp.array[wp.int32],
        contact_local: wp.array[wp.int32],
        world_contact_count: wp.array[wp.int32],
        contact_body_a: wp.array[wp.int32],
        contact_body_b: wp.array[wp.int32],
        contact_jacobian_a: wp.array[mat36f],
        contact_jacobian_b: wp.array[mat36f],
        contact_bias: wp.array[wp.vec3f],
        contact_friction: wp.array[wp.float32],
        contact_delassus: wp.array[wp.mat33f],
        contact_compliance: wp.array[wp.float32],
        contact_restitution: wp.array[wp.vec4f],
        # Spatial contacts (with angular friction):
        contact_frame: wp.array[wp.mat33f],
        angular_friction: wp.array[wp.vec2f],
        spatial_delassus: wp.array[mat66f],
        spatial_eigenvectors: wp.array[mat55f],
        spatial_eigenvalues: wp.array[vec5f],
        # Common:
        world_active: wp.array[wp.bool],
        weights: ProjectionWeights,
        occupancy: wp.array2d[wp.int32],
        projected_twist: wp.array[vec6f],
        # Outputs:
        box_reaction: wp.array[wp.float32],
        contact_reaction: wp.array[wp.vec3f],
        angular_reaction: wp.array[wp.vec3f],
        twist_delta: wp.array[vec6f],
        world_failed: wp.array[wp.bool],
    ):
        """Project every row of one color, or every row when no order is given (Jacobi)."""
        tid = wp.tid()

        begin = wp.int32(0)
        end = box_world.shape[0]
        if box_order:
            begin = box_color_offset[color]
            end = begin + box_color_count[color]
        for ordered in range(begin + tid, end, worker_count):
            row = ordered
            if box_order:
                row = box_order[ordered]
            wid = box_world[row]
            if box_local[row] >= world_box_count[wid]:
                continue
            if not world_active[wid] or world_failed[wid]:
                continue
            _project_box_row(
                row,
                color,
                box_body_a,
                box_body_b,
                box_jacobian_a,
                box_jacobian_b,
                box_bias,
                box_lower,
                box_upper,
                box_delassus,
                box_reaction,
                weights,
                occupancy,
                projected_twist,
                twist_delta,
            )

        begin = wp.int32(0)
        end = contact_world.shape[0]
        if contact_order:
            begin = contact_color_offset[color]
            end = begin + contact_color_count[color]
        for ordered in range(begin + tid, end, worker_count):
            cid = ordered
            if contact_order:
                cid = contact_order[ordered]
            wid = contact_world[cid]
            if contact_local[cid] >= world_contact_count[wid]:
                continue
            if not world_active[wid] or world_failed[wid]:
                continue
            if wp.static(spatial):
                _project_spatial_contact_row(
                    cid,
                    wid,
                    color,
                    contact_body_a,
                    contact_body_b,
                    contact_jacobian_a,
                    contact_jacobian_b,
                    contact_reaction,
                    contact_frame,
                    contact_friction,
                    angular_friction,
                    contact_bias,
                    contact_compliance,
                    contact_restitution,
                    spatial_delassus,
                    spatial_eigenvectors,
                    spatial_eigenvalues,
                    angular_reaction,
                    weights,
                    occupancy,
                    projected_twist,
                    twist_delta,
                    world_failed,
                )
            else:
                _project_contact_row(
                    cid,
                    color,
                    contact_body_a,
                    contact_body_b,
                    contact_jacobian_a,
                    contact_jacobian_b,
                    contact_bias,
                    contact_friction,
                    contact_delassus,
                    contact_compliance,
                    contact_restitution,
                    contact_reaction,
                    weights,
                    occupancy,
                    projected_twist,
                    twist_delta,
                )

    return sweep


@wp.func
def _box_row_residual(
    row: wp.int32,
    bid_a: wp.int32,
    bid_b: wp.int32,
    box_jacobian_a: wp.array[vec6f],
    box_jacobian_b: wp.array[vec6f],
    box_bias: wp.array[wp.float32],
    box_lower: wp.array[wp.float32],
    box_upper: wp.array[wp.float32],
    box_reaction: wp.array[wp.float32],
    physical_delassus: wp.array[wp.float32],
    projected_twist: wp.array[vec6f],
    box_velocity: wp.array[wp.float32],
) -> wp.vec2f:
    """Store the velocity ``J p`` of one box row and return its Delassus-scaled natural-map residual and
    Delassus scale.

    The product of both maps the residual back to velocity. A row without bodies has zero velocity and residual.
    """
    if bid_a < 0 and bid_b < 0:
        box_velocity[row] = 0.0
        return wp.vec2f(0.0)
    velocity = _box_row_velocity(row, bid_a, bid_b, box_jacobian_a, box_jacobian_b, projected_twist)
    box_velocity[row] = velocity
    scale = wp.sqrt(physical_delassus[row])
    scaled_reaction = scale * box_reaction[row]
    scaled_velocity = (velocity + box_bias[row]) / scale
    projected = wp.clamp(scaled_reaction - scaled_velocity, scale * box_lower[row], scale * box_upper[row])
    return wp.vec2f(wp.abs(scaled_reaction - projected), scale)


@wp.func
def _contact_row_residual(
    cid: wp.int32,
    bid_a: wp.int32,
    bid_b: wp.int32,
    contact_jacobian_a: wp.array[mat36f],
    contact_jacobian_b: wp.array[mat36f],
    contact_bias: wp.array[wp.vec3f],
    contact_friction: wp.array[wp.float32],
    contact_compliance: wp.array[wp.float32],
    contact_restitution: wp.array[wp.vec4f],
    contact_reaction: wp.array[wp.vec3f],
    physical_delassus: wp.array[wp.mat33f],
    projected_twist: wp.array[vec6f],
    contact_velocity: wp.array[wp.vec3f],
) -> wp.vec2f:
    """Store the velocity ``J p`` of one contact and return its Delassus-scaled Alart--Curnier residual and
    Delassus scale.

    The product of both maps the residual back to velocity. A contact without bodies has zero velocity and
    residual.
    """
    if bid_a < 0 and bid_b < 0:
        contact_velocity[cid] = wp.vec3f(0.0)
        return wp.vec2f(0.0)
    velocity = _contact_row_velocity(cid, bid_a, bid_b, contact_jacobian_a, contact_jacobian_b, projected_twist)
    contact_velocity[cid] = velocity
    delassus = physical_delassus[cid]
    reaction = contact_reaction[cid]
    law_velocity = _contact_law_velocity(
        cid,
        velocity + contact_bias[cid],
        reaction,
        delassus,
        contact_bias,
        contact_compliance,
        contact_restitution,
    )
    residual_vector = compute_contact_scaled_alart_curnier_residual(
        delassus, reaction, law_velocity, contact_friction[cid]
    )
    return wp.vec2f(wp.max(wp.abs(residual_vector)), compute_contact_delassus_scale(delassus))


@wp.func
def _spatial_contact_row_residual(
    cid: wp.int32,
    wid: wp.int32,
    bid_a: wp.int32,
    bid_b: wp.int32,
    contact_friction: wp.vec3f,
    contact_jacobian_a: wp.array[mat36f],
    contact_jacobian_b: wp.array[mat36f],
    frame: wp.array[wp.mat33f],
    contact_reaction: wp.array[wp.vec3f],
    angular_reaction: wp.array[wp.vec3f],
    normal_compliance: wp.array[wp.float32],
    restitution: wp.array[wp.vec4f],
    bias: wp.array[wp.vec3f],
    physical_delassus: wp.array[mat66f],
    projected_twist: wp.array[vec6f],
    contact_velocity: wp.array[wp.vec3f],
    world_failed: wp.array[wp.bool],
) -> wp.vec2f:
    """Store the linear velocity ``J p`` of one spatial contact and return its metric-scaled natural-map
    residual and velocity scale.

    The product of both maps the residual back to velocity; the angular rows use their friction
    lengths. A non-finite residual fails world ``wid`` and is returned as :data:`_UNRESOLVED_RESIDUAL`.
    """
    velocity = spatial_contact_velocity(
        cid, bid_a, bid_b, contact_jacobian_a, contact_jacobian_b, frame, projected_twist
    )
    contact_velocity[cid] = linear_contact_reaction(velocity)
    reaction = pack_contact_reaction(contact_reaction[cid], angular_reaction[cid])
    metric = physical_delassus[cid]
    compliance = _contact_normal_compliance(cid, normal_compliance)
    mechanical = metric
    mechanical[0, 0] -= compliance
    free_normal = velocity[0] - (mechanical @ reaction)[0]
    law_velocity = velocity
    law_velocity[0] += (
        bias[cid][2] - _contact_restitution_target(cid, free_normal, restitution) + compliance * reaction[0]
    )
    value = wp.max(wp.abs(compute_spatial_contact_residual(metric, contact_friction, reaction, law_velocity)))
    if not wp.isfinite(value):
        world_failed[wid] = True
        value = _UNRESOLVED_RESIDUAL
    return wp.vec2f(value, spatial_contact_velocity_scale(metric, contact_friction))


@wp.func
def _spatial_contact_lagged_change(
    cid: wp.int32,
    bid_a: wp.int32,
    bid_b: wp.int32,
    contact_jacobian_a: wp.array[mat36f],
    contact_jacobian_b: wp.array[mat36f],
    frame: wp.array[wp.mat33f],
    contact_friction: wp.vec3f,
    global_twist: wp.array[vec6f],
    projected_twist_previous: wp.array[vec6f],
) -> wp.float32:
    """Return the change ``|J (v - p_prev)|`` of the linear and frictional angular velocities of one spatial contact."""
    lagged = spatial_contact_velocity(
        cid, bid_a, bid_b, contact_jacobian_a, contact_jacobian_b, frame, global_twist
    ) - spatial_contact_velocity(
        cid, bid_a, bid_b, contact_jacobian_a, contact_jacobian_b, frame, projected_twist_previous
    )
    # The angular rows without torsional or rolling friction are free
    scaling = spatial_friction_scaling(contact_friction)
    change = wp.float32(0.0)
    for axis in range(6):
        if axis < 3 or scaling[axis] > 0.0:
            change = wp.max(change, wp.abs(lagged[axis]))
    return change


@wp.kernel
def _compute_box_residuals(
    # Inputs:
    world_failed: wp.array[wp.bool],
    box_world: wp.array[wp.int32],
    box_local: wp.array[wp.int32],
    world_box_count: wp.array[wp.int32],
    box_body_a: wp.array[wp.int32],
    box_body_b: wp.array[wp.int32],
    box_jacobian_a: wp.array[vec6f],
    box_jacobian_b: wp.array[vec6f],
    box_bias: wp.array[wp.float32],
    box_lower: wp.array[wp.float32],
    box_upper: wp.array[wp.float32],
    box_reaction: wp.array[wp.float32],
    physical_delassus: wp.array[wp.float32],
    projected_twist: wp.array[vec6f],
    # Outputs:
    box_velocity: wp.array[wp.float32],
    world_box_residual_max: wp.array[wp.float32],
):
    """Evaluate the Delassus-scaled natural-map residual of each box row, skipping the failed worlds."""
    row = wp.tid()
    wid = box_world[row]
    if box_local[row] >= world_box_count[wid] or world_failed[wid]:
        return
    residual = _box_row_residual(
        row,
        box_body_a[row],
        box_body_b[row],
        box_jacobian_a,
        box_jacobian_b,
        box_bias,
        box_lower,
        box_upper,
        box_reaction,
        physical_delassus,
        projected_twist,
        box_velocity,
    )
    atomic_max_nonnegative(world_box_residual_max, wid, residual[0])


@cache
def make_contact_residuals_kernel(spatial: bool):
    """Create the kernel that evaluates the contact-law residual of each contact, skipping the failed worlds.

    The residual is the Delassus-scaled Alart--Curnier residual of :func:`_contact_row_residual`, or the
    metric-scaled natural-map residual of :func:`_spatial_contact_row_residual` for spatial contacts. The
    ``thread_count`` threads of each world loop over its active contacts, so that the launch size follows
    the number of worlds.

    Args:
        spatial: Whether the contacts are spatial contacts with angular friction.
    """

    @wp.kernel(module="unique", enable_backward=False)
    def compute_contact_residuals(
        # Inputs:
        thread_count: wp.int32,
        world_contact_offset: wp.array[wp.int32],
        world_contact_count: wp.array[wp.int32],
        contact_body_a: wp.array[wp.int32],
        contact_body_b: wp.array[wp.int32],
        contact_jacobian_a: wp.array[mat36f],
        contact_jacobian_b: wp.array[mat36f],
        contact_bias: wp.array[wp.vec3f],
        contact_friction: wp.array[wp.float32],
        contact_compliance: wp.array[wp.float32],
        contact_restitution: wp.array[wp.vec4f],
        contact_reaction: wp.array[wp.vec3f],
        contact_physical_delassus: wp.array[wp.mat33f],
        # Spatial contacts (with angular friction):
        contact_frame: wp.array[wp.mat33f],
        angular_friction: wp.array[wp.vec2f],
        angular_reaction: wp.array[wp.vec3f],
        spatial_physical_delassus: wp.array[mat66f],
        # Common:
        projected_twist: wp.array[vec6f],
        # Outputs:
        world_failed: wp.array[wp.bool],
        contact_velocity: wp.array[wp.vec3f],
        world_contact_residual_max: wp.array[wp.float32],
    ):
        """Reduce the largest contact residual of each world."""
        wid, thread = wp.tid()
        if world_failed[wid]:
            return
        residual_max = wp.float32(0.0)
        local = thread
        while local < world_contact_count[wid]:
            cid = world_contact_offset[wid] + local
            local += thread_count
            bid_a = contact_body_a[cid]
            bid_b = contact_body_b[cid]
            if wp.static(spatial):
                residual = _spatial_contact_row_residual(
                    cid,
                    wid,
                    bid_a,
                    bid_b,
                    pack_spatial_friction(cid, contact_friction, angular_friction),
                    contact_jacobian_a,
                    contact_jacobian_b,
                    contact_frame,
                    contact_reaction,
                    angular_reaction,
                    contact_compliance,
                    contact_restitution,
                    contact_bias,
                    spatial_physical_delassus,
                    projected_twist,
                    contact_velocity,
                    world_failed,
                )
            else:
                residual = _contact_row_residual(
                    cid,
                    bid_a,
                    bid_b,
                    contact_jacobian_a,
                    contact_jacobian_b,
                    contact_bias,
                    contact_friction,
                    contact_compliance,
                    contact_restitution,
                    contact_reaction,
                    contact_physical_delassus,
                    projected_twist,
                    contact_velocity,
                )
            residual_max = wp.max(residual_max, residual[0])
        atomic_max_nonnegative(world_contact_residual_max, wid, residual_max)

    return compute_contact_residuals


@wp.func_native("""
float result = value;
#if defined(__CUDA_ARCH__)
for (int offset = 16; offset > 0; offset /= 2)
    result = fmaxf(result, __shfl_xor_sync(0xffffffff, result, offset));
#endif
return result;
""")
def _warp_max(value: wp.float32) -> wp.float32:
    """Return the largest ``value`` of the 32 threads of a CUDA warp, ignoring NaN; the value itself on the CPU.

    Every thread of the warp must call it.
    """
    ...


@cache
def make_iteration_residuals_kernel(spatial: bool):
    """Create the kernel that advances the splitting dual and reduces the convergence residuals of an iteration.

    The ``thread_count`` threads of each world loop over its bodies, box rows, and active contacts,
    so that one launch covers the work of :func:`_compute_box_residuals`, :func:`make_contact_residuals_kernel`,
    and the body residuals, with the same per-world maxima.
    Each warp reduces its threads before raising the maxima of its world.

    Args:
        spatial: Whether the contacts are spatial contacts with angular friction.
    """

    @wp.kernel(module="unique", enable_backward=False)
    def reduce_iteration_residuals(
        # Inputs:
        thread_count: wp.int32,
        model_time_dt: wp.array[wp.float32],
        position_tolerance: wp.float32,
        rotation_tolerance: wp.float32,
        velocity_tolerance: wp.float32,
        inverse_velocity_tolerance: wp.float32,
        world_active: wp.array[wp.bool],
        world_body_offset: wp.array[wp.int32],
        world_body_count: wp.array[wp.int32],
        global_twist_previous: wp.array[vec6f],
        global_twist: wp.array[vec6f],
        projected_twist_previous: wp.array[vec6f],
        projected_twist: wp.array[vec6f],
        world_box_offset: wp.array[wp.int32],
        world_box_count: wp.array[wp.int32],
        box_body_a: wp.array[wp.int32],
        box_body_b: wp.array[wp.int32],
        box_jacobian_a: wp.array[vec6f],
        box_jacobian_b: wp.array[vec6f],
        box_bias: wp.array[wp.float32],
        box_lower: wp.array[wp.float32],
        box_upper: wp.array[wp.float32],
        box_reaction: wp.array[wp.float32],
        box_physical_delassus: wp.array[wp.float32],
        world_contact_offset: wp.array[wp.int32],
        world_contact_count: wp.array[wp.int32],
        contact_body_a: wp.array[wp.int32],
        contact_body_b: wp.array[wp.int32],
        contact_jacobian_a: wp.array[mat36f],
        contact_jacobian_b: wp.array[mat36f],
        contact_bias: wp.array[wp.vec3f],
        contact_friction: wp.array[wp.float32],
        contact_compliance: wp.array[wp.float32],
        contact_restitution: wp.array[wp.vec4f],
        contact_reaction: wp.array[wp.vec3f],
        contact_physical_delassus: wp.array[wp.mat33f],
        # Spatial contacts (with angular friction):
        contact_frame: wp.array[wp.mat33f],
        angular_friction: wp.array[wp.vec2f],
        angular_reaction: wp.array[wp.vec3f],
        spatial_physical_delassus: wp.array[mat66f],
        # Outputs:
        splitting_dual: wp.array[vec6f],
        world_failed: wp.array[wp.bool],
        box_velocity: wp.array[wp.float32],
        contact_velocity: wp.array[wp.vec3f],
        world_box_residual_max: wp.array[wp.float32],
        world_contact_residual_max: wp.array[wp.float32],
        world_velocity_residual: wp.array[wp.float32],
        world_lagged_residual: wp.array[wp.float32],
        world_residual_change: wp.array[wp.float32],
        world_residual_split: wp.array[wp.float32],
        world_residual_cross_iterate: wp.array[wp.float32],
    ):
        """Advance the splitting dual, fail the worlds with non-finite iterates, and reduce the
        body and unilateral residuals of the active worlds.
        """
        wid, thread = wp.tid()
        if not world_active[wid]:
            return
        # Every thread of a warp reaches the warp reductions, also in the worlds that failed
        body_count = wp.int32(0)
        box_count = wp.int32(0)
        contact_count = wp.int32(0)
        if not world_failed[wid]:
            body_count = world_body_count[wid]
            box_count = world_box_count[wid]
            contact_count = world_contact_count[wid]

        dt = model_time_dt[wid]
        body_max = wp.vec3f(0.0)
        local = thread
        while local < body_count:
            bid = world_body_offset[wid] + local
            local += thread_count
            current = global_twist[bid]
            projected = projected_twist[bid]
            dual = splitting_dual[bid] + projected - current
            splitting_dual[bid] = dual
            # The previous twists were checked as the current ones of the last iteration
            if not (wp.isfinite(current) and wp.isfinite(projected) and wp.isfinite(dual)):
                world_failed[wid] = True
                continue
            body_residuals = compute_body_iteration_residuals(
                dt,
                position_tolerance,
                rotation_tolerance,
                velocity_tolerance,
                global_twist_previous[bid],
                current,
                projected_twist_previous[bid],
                projected,
            )
            body_max = wp.max(body_max, body_residuals)

        box_max = wp.float32(0.0)
        velocity_max = wp.float32(0.0)
        lagged_max = wp.float32(0.0)
        local = thread
        while local < box_count:
            row = world_box_offset[wid] + local
            local += thread_count
            bid_a = box_body_a[row]
            bid_b = box_body_b[row]
            residual = _box_row_residual(
                row,
                bid_a,
                bid_b,
                box_jacobian_a,
                box_jacobian_b,
                box_bias,
                box_lower,
                box_upper,
                box_reaction,
                box_physical_delassus,
                projected_twist,
                box_velocity,
            )
            box_max = wp.max(box_max, residual[0])
            velocity_max = wp.max(velocity_max, residual[0] * residual[1] * inverse_velocity_tolerance)
            lagged = _box_row_velocity(
                row, bid_a, bid_b, box_jacobian_a, box_jacobian_b, global_twist
            ) - _box_row_velocity(row, bid_a, bid_b, box_jacobian_a, box_jacobian_b, projected_twist_previous)
            lagged_max = wp.max(lagged_max, wp.abs(lagged) * inverse_velocity_tolerance)

        contact_max = wp.float32(0.0)
        local = thread
        while local < contact_count:
            cid = world_contact_offset[wid] + local
            local += thread_count
            bid_a = contact_body_a[cid]
            bid_b = contact_body_b[cid]
            if wp.static(spatial):
                friction = pack_spatial_friction(cid, contact_friction, angular_friction)
                contact_residual = _spatial_contact_row_residual(
                    cid,
                    wid,
                    bid_a,
                    bid_b,
                    friction,
                    contact_jacobian_a,
                    contact_jacobian_b,
                    contact_frame,
                    contact_reaction,
                    angular_reaction,
                    contact_compliance,
                    contact_restitution,
                    contact_bias,
                    spatial_physical_delassus,
                    projected_twist,
                    contact_velocity,
                    world_failed,
                )
                change = _spatial_contact_lagged_change(
                    cid,
                    bid_a,
                    bid_b,
                    contact_jacobian_a,
                    contact_jacobian_b,
                    contact_frame,
                    friction,
                    global_twist,
                    projected_twist_previous,
                )
            else:
                contact_residual = _contact_row_residual(
                    cid,
                    bid_a,
                    bid_b,
                    contact_jacobian_a,
                    contact_jacobian_b,
                    contact_bias,
                    contact_friction,
                    contact_compliance,
                    contact_restitution,
                    contact_reaction,
                    contact_physical_delassus,
                    projected_twist,
                    contact_velocity,
                )
                contact_lagged = _contact_row_velocity(
                    cid, bid_a, bid_b, contact_jacobian_a, contact_jacobian_b, global_twist
                ) - _contact_row_velocity(
                    cid, bid_a, bid_b, contact_jacobian_a, contact_jacobian_b, projected_twist_previous
                )
                change = wp.max(wp.abs(contact_lagged))
            contact_max = wp.max(contact_max, contact_residual[0])
            velocity_max = wp.max(velocity_max, contact_residual[0] * contact_residual[1] * inverse_velocity_tolerance)
            lagged_max = wp.max(lagged_max, change * inverse_velocity_tolerance)

        # The threads of a world span whole warps
        change_max = _warp_max(body_max[0])
        split_max = _warp_max(body_max[1])
        cross_iterate_max = _warp_max(body_max[2])
        box_max = _warp_max(box_max)
        contact_max = _warp_max(contact_max)
        velocity_max = _warp_max(velocity_max)
        lagged_max = _warp_max(lagged_max)
        if thread % 32 == 0:
            atomic_max_nonnegative(world_residual_change, wid, change_max)
            atomic_max_nonnegative(world_residual_split, wid, split_max)
            atomic_max_nonnegative(world_residual_cross_iterate, wid, cross_iterate_max)
            atomic_max_nonnegative(world_box_residual_max, wid, box_max)
            atomic_max_nonnegative(world_contact_residual_max, wid, contact_max)
            atomic_max_nonnegative(world_velocity_residual, wid, velocity_max)
            atomic_max_nonnegative(world_lagged_residual, wid, lagged_max)

    return reduce_iteration_residuals
