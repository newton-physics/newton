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
from .solver_kernels import atomic_max_nonnegative
from .system_kernels import apply_body_weight

###
# Module interface
###

__all__ = [
    "_apply_twist_delta",
    "_compute_box_residuals",
    "_compute_contact_residuals",
    "_compute_spatial_contact_residuals",
    "_prepare_box_rows",
    "_prepare_contacts",
    "_prepare_physical_box_rows",
    "_prepare_physical_contacts",
    "_prepare_physical_spatial_contacts",
    "_prepare_spatial_contacts",
    "_project_box_row",
    "_project_contact_row",
    "_project_spatial_contact_row",
    "_warmstart_box_rows",
    "_warmstart_contacts",
    "_warmstart_spatial_contacts",
    "is_active_row",
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
    inverse_weight_a: mat66f,
    jacobian_b: vec6f,
    inverse_weight_b: mat66f,
) -> wp.float32:
    """Compute the scalar local Delassus coefficient ``J W^-1 J^T`` of a box row."""
    return wp.dot(jacobian_a, apply_body_weight(inverse_weight_a, jacobian_a)) + wp.dot(
        jacobian_b, apply_body_weight(inverse_weight_b, jacobian_b)
    )


@wp.func
def _compute_contact_delassus(
    jacobian_a: mat36f,
    inverse_weight_a: mat66f,
    jacobian_b: mat36f,
    inverse_weight_b: mat66f,
) -> wp.mat33f:
    """Compute the ``3 x 3`` local Delassus block ``J W^-1 J^T`` of a contact, evaluating each pair of rows once."""
    delassus = wp.mat33f(0.0)
    for row in range(3):
        weighted_a = apply_body_weight(inverse_weight_a, jacobian_a[row])
        weighted_b = apply_body_weight(inverse_weight_b, jacobian_b[row])
        for col in range(3):
            if col >= row:
                value = wp.dot(weighted_a, jacobian_a[col]) + wp.dot(weighted_b, jacobian_b[col])
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
def _split_inverse_weight(
    bid: wp.int32,
    color: wp.int32,
    inverse_weight: wp.array[mat66f],
    occupancy: wp.array2d[wp.int32],
) -> mat66f:
    """Return the mass-split inverse weight ``m W^-1`` of a body, or zero if absent."""
    if bid < 0:
        return mat66f(0.0)
    return wp.float32(wp.max(1, occupancy[bid, color])) * inverse_weight[bid]


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
    inverse_weight: wp.array[mat66f],
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
        correction = apply_body_weight(inverse_weight[bid_a], row_jacobian_a * delta)
        _accumulate_body_delta(bid_a, color, correction, occupancy, twist_delta)
    if bid_b >= 0:
        correction = apply_body_weight(inverse_weight[bid_b], row_jacobian_b * delta)
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
    inverse_weight: wp.array[mat66f],
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
        correction = apply_body_weight(inverse_weight[bid_a], wp.transpose(contact_jacobian_a) @ delta)
        _accumulate_body_delta(bid_a, color, correction, occupancy, twist_delta)
    if bid_b >= 0:
        correction = apply_body_weight(inverse_weight[bid_b], wp.transpose(contact_jacobian_b) @ delta)
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
    inverse_weight: wp.array[mat66f],
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
        _accumulate_body_delta(bid_a, color, apply_body_weight(inverse_weight[bid_a], wrench), occupancy, twist_delta)
    if bid_b >= 0:
        wrench = spatial_body_wrench(contact_jacobian_b, contact_frame, 1.0, delta)
        _accumulate_body_delta(bid_b, color, apply_body_weight(inverse_weight[bid_b], wrench), occupancy, twist_delta)


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
    inverse_weight: wp.array[mat66f],
    twist_delta: wp.array[vec6f],
):
    """Accumulate the twist change ``W^-1 J^T impulse`` of one box row."""
    if impulse == 0.0:
        return
    bid_a = body_a[row]
    bid_b = body_b[row]
    if bid_a >= 0:
        wp.atomic_add(twist_delta, bid_a, apply_body_weight(inverse_weight[bid_a], jacobian_a[row] * impulse))
    if bid_b >= 0:
        wp.atomic_add(twist_delta, bid_b, apply_body_weight(inverse_weight[bid_b], jacobian_b[row] * impulse))


@wp.func
def scatter_contact_impulse(
    cid: wp.int32,
    body_a: wp.array[wp.int32],
    body_b: wp.array[wp.int32],
    jacobian_a: wp.array[mat36f],
    jacobian_b: wp.array[mat36f],
    impulse: wp.vec3f,
    inverse_weight: wp.array[mat66f],
    twist_delta: wp.array[vec6f],
):
    """Accumulate the twist change ``W^-1 J^T impulse`` of one contact."""
    if _is_zero_vec3(impulse):
        return
    bid_a = body_a[cid]
    bid_b = body_b[cid]
    if bid_a >= 0:
        wp.atomic_add(
            twist_delta, bid_a, apply_body_weight(inverse_weight[bid_a], wp.transpose(jacobian_a[cid]) @ impulse)
        )
    if bid_b >= 0:
        wp.atomic_add(
            twist_delta, bid_b, apply_body_weight(inverse_weight[bid_b], wp.transpose(jacobian_b[cid]) @ impulse)
        )


###
# Kernels
###


@wp.kernel
def _prepare_box_rows(
    # Inputs:
    box_world: wp.array[wp.int32],
    box_local: wp.array[wp.int32],
    world_box_count: wp.array[wp.int32],
    box_color: wp.array[wp.int32],
    box_body_a: wp.array[wp.int32],
    box_body_b: wp.array[wp.int32],
    box_jacobian_a: wp.array[vec6f],
    box_jacobian_b: wp.array[vec6f],
    occupancy: wp.array2d[wp.int32],
    inverse_weight: wp.array[mat66f],
    # Outputs:
    box_delassus: wp.array[wp.float32],
    world_failed: wp.array[wp.bool],
):
    row = wp.tid()
    wid = box_world[row]
    if box_local[row] >= world_box_count[wid]:
        return
    bid_a = box_body_a[row]
    bid_b = box_body_b[row]
    if bid_a < 0 and bid_b < 0:
        box_delassus[row] = 0.0
        return
    color = row_color(row, box_color)
    value = _compute_box_delassus(
        box_jacobian_a[row],
        _split_inverse_weight(bid_a, color, inverse_weight, occupancy),
        box_jacobian_b[row],
        _split_inverse_weight(bid_b, color, inverse_weight, occupancy),
    )
    box_delassus[row] = value
    if not wp.isfinite(value) or value <= 0.0:
        world_failed[wid] = True


@wp.kernel
def _prepare_contacts(
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
    occupancy: wp.array2d[wp.int32],
    inverse_weight: wp.array[mat66f],
    # Outputs:
    contact_delassus: wp.array[wp.mat33f],
    world_failed: wp.array[wp.bool],
):
    cid = wp.tid()
    wid = contact_world[cid]
    if contact_local[cid] >= world_contact_count[wid]:
        return
    bid_a = contact_body_a[cid]
    bid_b = contact_body_b[cid]
    if bid_a < 0 and bid_b < 0:
        contact_delassus[cid] = wp.mat33f(0.0)
        return
    color = row_color(cid, contact_color)
    # A non-finite Jacobian or inverse weight makes the block non-finite, which fails the world
    delassus, valid = _regularize_contact_delassus(
        _compute_contact_delassus(
            contact_jacobian_a[cid],
            _split_inverse_weight(bid_a, color, inverse_weight, occupancy),
            contact_jacobian_b[cid],
            _split_inverse_weight(bid_b, color, inverse_weight, occupancy),
        ),
        contact_bias[cid],
        contact_friction[cid],
    )
    delassus[2, 2] += _contact_normal_compliance(cid, contact_compliance)
    contact_delassus[cid] = delassus
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
def _prepare_physical_box_rows(
    # Inputs:
    box_world: wp.array[wp.int32],
    box_local: wp.array[wp.int32],
    world_box_count: wp.array[wp.int32],
    box_body_a: wp.array[wp.int32],
    box_body_b: wp.array[wp.int32],
    box_jacobian_a: wp.array[vec6f],
    box_jacobian_b: wp.array[vec6f],
    inverse_weight: wp.array[mat66f],
    # Outputs:
    physical_delassus: wp.array[wp.float32],
):
    """Prepare the unsplit Delassus coefficient of each box row, used by the residuals."""
    row = wp.tid()
    if box_local[row] >= world_box_count[box_world[row]]:
        return
    bid_a = box_body_a[row]
    bid_b = box_body_b[row]
    inverse_weight_a = mat66f(0.0)
    inverse_weight_b = mat66f(0.0)
    if bid_a >= 0:
        inverse_weight_a = inverse_weight[bid_a]
    if bid_b >= 0:
        inverse_weight_b = inverse_weight[bid_b]
    physical_delassus[row] = _compute_box_delassus(
        box_jacobian_a[row], inverse_weight_a, box_jacobian_b[row], inverse_weight_b
    )


@wp.kernel
def _prepare_physical_contacts(
    # Inputs:
    contact_world: wp.array[wp.int32],
    contact_local: wp.array[wp.int32],
    world_contact_count: wp.array[wp.int32],
    contact_body_a: wp.array[wp.int32],
    contact_body_b: wp.array[wp.int32],
    contact_jacobian_a: wp.array[mat36f],
    contact_jacobian_b: wp.array[mat36f],
    normal_compliance: wp.array[wp.float32],
    inverse_weight: wp.array[mat66f],
    # Outputs:
    physical_delassus: wp.array[wp.mat33f],
):
    """Prepare the unsplit compliant Delassus block of each contact, used by the residuals."""
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
    delassus = _compute_contact_delassus(
        contact_jacobian_a[cid], inverse_weight_a, contact_jacobian_b[cid], inverse_weight_b
    )
    delassus[2, 2] += _contact_normal_compliance(cid, normal_compliance)
    physical_delassus[cid] = delassus


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
    inverse_weight: wp.array[mat66f],
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
        inverse_weight,
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
    inverse_weight: wp.array[mat66f],
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
        inverse_weight,
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
    inverse_weight: wp.array[mat66f],
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
        inverse_weight,
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
        inverse_weight: wp.array[mat66f],
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
                inverse_weight,
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
                    inverse_weight,
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
                    inverse_weight,
                    occupancy,
                    projected_twist,
                    twist_delta,
                )

    return sweep


@wp.kernel
def _compute_box_residuals(
    # Inputs:
    world_active: wp.array[wp.bool],
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
    global_twist: wp.array[vec6f],
    projected_twist_previous: wp.array[vec6f],
    inverse_velocity_tolerance: wp.float32,
    # Outputs:
    box_velocity: wp.array[wp.float32],
    world_box_residual_max: wp.array[wp.float32],
    world_velocity_residual: wp.array[wp.float32],
    world_lagged_residual: wp.array[wp.float32],
):
    """Evaluate the Delassus-scaled natural-map residual of each box row.

    With ``world_velocity_residual``, also reduce the residual mapped back to
    velocity by the Delassus scale, relative to the velocity tolerance. With
    ``world_lagged_residual``, also reduce the change ``|J (v - p_prev)|`` of the
    row velocity over the last iteration, relative to the velocity tolerance.
    """
    row = wp.tid()
    wid = box_world[row]
    if box_local[row] >= world_box_count[wid] or (world_active and not world_active[wid]) or world_failed[wid]:
        return
    bid_a = box_body_a[row]
    bid_b = box_body_b[row]
    if bid_a < 0 and bid_b < 0:
        box_velocity[row] = 0.0
        return
    velocity = _box_row_velocity(row, bid_a, bid_b, box_jacobian_a, box_jacobian_b, projected_twist)
    box_velocity[row] = velocity
    scale = wp.sqrt(physical_delassus[row])
    scaled_reaction = scale * box_reaction[row]
    scaled_velocity = (velocity + box_bias[row]) / scale
    projected = wp.clamp(scaled_reaction - scaled_velocity, scale * box_lower[row], scale * box_upper[row])
    residual = wp.abs(scaled_reaction - projected)
    atomic_max_nonnegative(world_box_residual_max, wid, residual)
    if world_velocity_residual:
        atomic_max_nonnegative(world_velocity_residual, wid, residual * scale * inverse_velocity_tolerance)
    if world_lagged_residual:
        lagged = _box_row_velocity(row, bid_a, bid_b, box_jacobian_a, box_jacobian_b, global_twist) - _box_row_velocity(
            row, bid_a, bid_b, box_jacobian_a, box_jacobian_b, projected_twist_previous
        )
        atomic_max_nonnegative(world_lagged_residual, wid, wp.abs(lagged) * inverse_velocity_tolerance)


@wp.kernel
def _compute_contact_residuals(
    # Inputs:
    thread_count: wp.int32,
    world_active: wp.array[wp.bool],
    world_failed: wp.array[wp.bool],
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
    physical_delassus: wp.array[wp.mat33f],
    projected_twist: wp.array[vec6f],
    global_twist: wp.array[vec6f],
    projected_twist_previous: wp.array[vec6f],
    inverse_velocity_tolerance: wp.float32,
    # Outputs:
    contact_velocity: wp.array[wp.vec3f],
    world_contact_residual_max: wp.array[wp.float32],
    world_velocity_residual: wp.array[wp.float32],
    world_lagged_residual: wp.array[wp.float32],
):
    """Evaluate the Delassus-scaled Alart--Curnier residual of the contact law of each contact.

    With ``world_velocity_residual``, also reduce the residual mapped back to
    velocity by the Delassus scale, relative to the velocity tolerance. With
    ``world_lagged_residual``, also reduce the change ``|J (v - p_prev)|`` of the
    contact velocity over the last iteration, relative to the velocity tolerance.
    The ``thread_count`` threads of each world loop over its active contacts, so that
    the launch size follows the number of worlds.
    """
    wid, thread = wp.tid()
    if (world_active and not world_active[wid]) or world_failed[wid]:
        return
    residual_max = wp.float32(0.0)
    velocity_residual_max = wp.float32(0.0)
    lagged_max = wp.float32(0.0)
    local = thread
    while local < world_contact_count[wid]:
        cid = world_contact_offset[wid] + local
        local += thread_count
        bid_a = contact_body_a[cid]
        bid_b = contact_body_b[cid]
        if bid_a < 0 and bid_b < 0:
            contact_velocity[cid] = wp.vec3f(0.0)
        else:
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
            residual = wp.max(wp.abs(residual_vector))
            residual_max = wp.max(residual_max, residual)
            if world_velocity_residual:
                velocity_residual = residual * compute_contact_delassus_scale(delassus) * inverse_velocity_tolerance
                velocity_residual_max = wp.max(velocity_residual_max, velocity_residual)
            if world_lagged_residual:
                lagged = _contact_row_velocity(
                    cid, bid_a, bid_b, contact_jacobian_a, contact_jacobian_b, global_twist
                ) - _contact_row_velocity(
                    cid, bid_a, bid_b, contact_jacobian_a, contact_jacobian_b, projected_twist_previous
                )
                lagged_max = wp.max(lagged_max, wp.max(wp.abs(lagged)) * inverse_velocity_tolerance)
    atomic_max_nonnegative(world_contact_residual_max, wid, residual_max)
    if world_velocity_residual:
        atomic_max_nonnegative(world_velocity_residual, wid, velocity_residual_max)
    if world_lagged_residual:
        atomic_max_nonnegative(world_lagged_residual, wid, lagged_max)


@wp.kernel
def _compute_spatial_contact_residuals(
    # Inputs:
    thread_count: wp.int32,
    world_active: wp.array[wp.bool],
    world_contact_offset: wp.array[wp.int32],
    world_contact_count: wp.array[wp.int32],
    contact_body_a: wp.array[wp.int32],
    contact_body_b: wp.array[wp.int32],
    contact_jacobian_a: wp.array[mat36f],
    contact_jacobian_b: wp.array[mat36f],
    contact_reaction: wp.array[wp.vec3f],
    frame: wp.array[wp.mat33f],
    friction: wp.array[wp.float32],
    angular_friction: wp.array[wp.vec2f],
    bias: wp.array[wp.vec3f],
    normal_compliance: wp.array[wp.float32],
    restitution: wp.array[wp.vec4f],
    angular_reaction: wp.array[wp.vec3f],
    physical_delassus: wp.array[mat66f],
    twist: wp.array[vec6f],
    global_twist: wp.array[vec6f],
    projected_twist_previous: wp.array[vec6f],
    inverse_velocity_tolerance: wp.float32,
    # Outputs:
    world_failed: wp.array[wp.bool],
    contact_velocity: wp.array[wp.vec3f],
    world_contact_residual_max: wp.array[wp.float32],
    world_velocity_residual: wp.array[wp.float32],
    world_lagged_residual: wp.array[wp.float32],
):
    """Evaluate the metric-scaled natural-map residual of each spatial contact.

    With ``world_velocity_residual``, also reduce the residual mapped back to
    velocity, relative to the velocity tolerance; the angular rows use their
    friction lengths. With ``world_lagged_residual``, also reduce the change
    ``|J (v - p_prev)|`` of the linear and frictional angular contact velocities
    over the last iteration, relative to the velocity tolerance. The ``thread_count``
    threads of each world loop over its active contacts, so that the launch size
    follows the number of worlds.
    """
    wid, thread = wp.tid()
    if (world_active and not world_active[wid]) or world_failed[wid]:
        return
    residual_max = wp.float32(0.0)
    velocity_residual_max = wp.float32(0.0)
    lagged_max = wp.float32(0.0)
    local = thread
    while local < world_contact_count[wid]:
        cid = world_contact_offset[wid] + local
        local += thread_count
        bid_a = contact_body_a[cid]
        bid_b = contact_body_b[cid]
        reaction = pack_contact_reaction(contact_reaction[cid], angular_reaction[cid])
        contact_friction = pack_spatial_friction(cid, friction, angular_friction)
        velocity = spatial_contact_velocity(cid, bid_a, bid_b, contact_jacobian_a, contact_jacobian_b, frame, twist)
        contact_velocity[cid] = linear_contact_reaction(velocity)
        metric = physical_delassus[cid]
        compliance = _contact_normal_compliance(cid, normal_compliance)
        mechanical = metric
        mechanical[0, 0] -= compliance
        free_normal = velocity[0] - (mechanical @ reaction)[0]
        velocity[0] += (
            bias[cid][2] - _contact_restitution_target(cid, free_normal, restitution) + compliance * reaction[0]
        )
        value = wp.max(wp.abs(compute_spatial_contact_residual(metric, contact_friction, reaction, velocity)))
        if not wp.isfinite(value):
            world_failed[wid] = True
            value = _UNRESOLVED_RESIDUAL
        residual_max = wp.max(residual_max, value)
        if world_velocity_residual:
            velocity_residual = (
                value * spatial_contact_velocity_scale(metric, contact_friction) * inverse_velocity_tolerance
            )
            velocity_residual_max = wp.max(velocity_residual_max, velocity_residual)
        if world_lagged_residual:
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
            lagged_max = wp.max(lagged_max, change * inverse_velocity_tolerance)
    atomic_max_nonnegative(world_contact_residual_max, wid, residual_max)
    if world_velocity_residual:
        atomic_max_nonnegative(world_velocity_residual, wid, velocity_residual_max)
    if world_lagged_residual:
        atomic_max_nonnegative(world_lagged_residual, wid, lagged_max)
