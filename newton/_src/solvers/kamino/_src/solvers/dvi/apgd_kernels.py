# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Warp kernels for the Kamino contact APGD backend."""

from __future__ import annotations

import warp as wp

from ..padmm.math import project_to_coulomb_cone

wp.set_module_options({"enable_backward": False})

float32 = wp.float32
int32 = wp.int32
vec3f = wp.vec3f


# Per-world scalar workspace columns.
_L = 0
_T = 1
_THETA = 2
_BETA = 3
_BEST_RESIDUAL = 4
_OBJ_CANDIDATE = 5
_OBJ_MODEL = 6
_RESIDUAL_SQUARED = 7
_RESTART_DOT = 8
_GDIFF = 9
_GAMMA_NORM_SQUARED = 10
_RAYLEIGH_NORM_SQUARED = 11
_RAYLEIGH_PRODUCT_NORM_SQUARED = 12
_NUM_APGD_SCALARS = 13

# Per-world integer workspace columns.
_UPDATE_BEST = 0
_RESTART = 1
_BACKTRACKS_THIS_ITERATION = 2
_NUM_APGD_FLAGS = 3

# Per-contact partials used by the fixed-order scalar reductions.  APGD's
# Rayleigh, objective, restart, and Res4 sums span a wide dynamic range, so a
# single float32 accumulator is not sufficiently robust for large worlds.
_ACC_RAYLEIGH_NORM_SQUARED = 0
_ACC_RAYLEIGH_PRODUCT_NORM_SQUARED = 1
_ACC_OBJ_CANDIDATE = 2
_ACC_OBJ_MODEL = 3
_ACC_RESIDUAL_SQUARED = 4
_ACC_RESTART_DOT = 5
_ACC_GAMMA_NORM_SQUARED = 6
_NUM_APGD_ACCUMULATORS = 7

# Match the deterministic two-level reduction used by the final newton-dvi
# APGD.  Each block owns a fixed contiguous contact range, and one world thread
# combines the fixed block order.  Integer scheduling therefore cannot change
# the reduction order.
_NUM_APGD_REDUCTION_BLOCKS = 256


@wp.struct
class ContactAPGDConfigStruct:
    """Device-side controls for one world's contact APGD solve."""

    max_iterations: int32
    max_backtrack_iterations: int32
    tolerance: float32
    min_iterations: int32
    early_exit: int32


@wp.struct
class ContactAPGDStatus:
    """Device-resident result of one contact APGD phase."""

    converged: int32
    iterations: int32
    backtracks: int32
    restarts: int32
    residual: float32


@wp.kernel
def reduce_apgd_partials_to_blocks(
    contact_count: wp.array[int32],
    contact_offset: wp.array[int32],
    world_mask: wp.array[bool],
    num_blocks: int32,
    slot_a: int32,
    slot_b: int32,
    partials: wp.array2d[float32],
    block_sums: wp.array2d[float32],
):
    """Reduce fixed contact chunks with float64 accumulators.

    ``block_sums`` has two rows per world. ``slot_b < 0`` requests a
    single-output reduction and leaves the second row unused.
    """
    wid, block = wp.tid()
    if not world_mask[wid]:
        return

    count = contact_count[wid]
    chunk = (count + num_blocks - int32(1)) / num_blocks
    start = block * chunk
    end = wp.min(start + chunk, count)
    partial_offset = contact_offset[wid]
    sum_a = wp.float64(0.0)
    sum_b = wp.float64(0.0)
    for cid in range(start, end):
        sum_a += wp.float64(partials[slot_a, partial_offset + cid])
        if slot_b >= int32(0):
            sum_b += wp.float64(partials[slot_b, partial_offset + cid])

    block_sums[int32(2) * wid, block] = float32(sum_a)
    if slot_b >= int32(0):
        block_sums[int32(2) * wid + int32(1), block] = float32(sum_b)


@wp.kernel
def combine_apgd_block_sums(
    world_mask: wp.array[bool],
    num_blocks: int32,
    slot_b_valid: int32,
    scalar_a: int32,
    scalar_b: int32,
    block_sums: wp.array2d[float32],
    scalars: wp.array2d[float32],
):
    """Combine block partials in fixed order with float64 accumulators."""
    wid = wp.tid()
    if not world_mask[wid]:
        return

    sum_a = wp.float64(0.0)
    sum_b = wp.float64(0.0)
    for block in range(num_blocks):
        sum_a += wp.float64(block_sums[int32(2) * wid, block])
        if slot_b_valid != int32(0):
            sum_b += wp.float64(block_sums[int32(2) * wid + int32(1), block])

    scalars[wid, scalar_a] = float32(sum_a)
    if slot_b_valid != int32(0):
        scalars[wid, scalar_b] = float32(sum_b)


@wp.kernel
def initialize_apgd_worlds(
    contact_count: wp.array[int32],
    contact_row_offset: wp.array[int32],
    phase_mask: wp.array[bool],
    config: wp.array[ContactAPGDConfigStruct],
    scalars: wp.array2d[float32],
    flags: wp.array2d[int32],
    outer_mask: wp.array[bool],
    backtrack_mask: wp.array[bool],
    outer_continue: wp.array[int32],
    status: wp.array[ContactAPGDStatus],
):
    """Initialize per-world APGD state and the outer loop condition."""
    wid = wp.tid()
    capacity = (contact_row_offset[wid + 1] - contact_row_offset[wid]) / int32(3)
    count = contact_count[wid]
    assert count >= int32(0)
    assert count <= capacity

    active = phase_mask[wid] and count > int32(0) and config[wid].max_iterations > int32(0)
    outer_mask[wid] = active
    backtrack_mask[wid] = False

    for column in range(_NUM_APGD_SCALARS):
        scalars[wid, column] = float32(0.0)
    scalars[wid, _THETA] = float32(1.0)
    scalars[wid, _BEST_RESIDUAL] = float32(3.0e38)
    rows = float32(3.0) * float32(count)
    scalars[wid, _GDIFF] = float32(1.0) / wp.max(rows * rows, float32(1.0))

    for column in range(_NUM_APGD_FLAGS):
        flags[wid, column] = int32(0)

    world_status = ContactAPGDStatus()
    world_status.converged = int32(not active)
    world_status.iterations = int32(0)
    world_status.backtracks = int32(0)
    world_status.restarts = int32(0)
    world_status.residual = float32(0.0) if not active else float32(3.0e38)
    status[wid] = world_status

    if active:
        wp.atomic_add(outer_continue, 0, int32(1))


@wp.kernel
def initialize_apgd_vectors(
    contact_count: wp.array[int32],
    contact_row_offset: wp.array[int32],
    phase_mask: wp.array[bool],
    solution: wp.array[float32],
    y: wp.array[float32],
    gamma: wp.array[float32],
    gamma_new: wp.array[float32],
    gamma_best: wp.array[float32],
    rayleigh_vector: wp.array[float32],
):
    """Seed APGD iterates from the incoming warm start and build Rayleigh data."""
    wid, cid = wp.tid()
    row_offset = contact_row_offset[wid]
    capacity = (contact_row_offset[wid + 1] - row_offset) / int32(3)
    if cid >= capacity:
        return

    active = phase_mask[wid] and cid < contact_count[wid]
    row = row_offset + int32(3) * cid
    for component in range(3):
        value = float32(0.0)
        rayleigh_value = float32(0.0)
        if active:
            value = solution[row + component]
            # Match the Project DVI/Chrono APGD seed gamma_0 - gamma_hat_0.
            rayleigh_value = float32(-1.0)
        y[row + component] = value
        gamma[row + component] = value
        gamma_new[row + component] = value
        gamma_best[row + component] = value
        rayleigh_vector[row + component] = rayleigh_value


@wp.kernel
def write_rayleigh_partials(
    contact_count: wp.array[int32],
    contact_row_offset: wp.array[int32],
    contact_offset: wp.array[int32],
    world_mask: wp.array[bool],
    rayleigh_vector: wp.array[float32],
    rayleigh_product: wp.array[float32],
    partials: wp.array2d[float32],
):
    """Write one contact's Rayleigh norm contributions."""
    wid, cid = wp.tid()
    if not world_mask[wid] or cid >= contact_count[wid]:
        return

    row = contact_row_offset[wid] + int32(3) * cid
    norm_w_squared = float32(0.0)
    norm_aw_squared = float32(0.0)
    for component in range(3):
        w = rayleigh_vector[row + component]
        aw = rayleigh_product[row + component]
        norm_w_squared += w * w
        norm_aw_squared += aw * aw
    partial = contact_offset[wid] + cid
    partials[_ACC_RAYLEIGH_NORM_SQUARED, partial] = norm_w_squared
    partials[_ACC_RAYLEIGH_PRODUCT_NORM_SQUARED, partial] = norm_aw_squared


@wp.kernel
def finalize_rayleigh_estimate(
    world_mask: wp.array[bool],
    scalars: wp.array2d[float32],
):
    """Finalize the reduced per-world Rayleigh Lipschitz estimate."""
    wid = wp.tid()
    if not world_mask[wid]:
        return

    lipschitz = float32(1.0e-12)
    norm_w_squared = scalars[wid, _RAYLEIGH_NORM_SQUARED]
    norm_aw_squared = scalars[wid, _RAYLEIGH_PRODUCT_NORM_SQUARED]
    if norm_w_squared > float32(0.0):
        lipschitz = wp.max(wp.sqrt(norm_aw_squared / norm_w_squared), lipschitz)
    scalars[wid, _L] = lipschitz
    scalars[wid, _T] = float32(1.0) / lipschitz


@wp.kernel
def prepare_apgd_iteration(
    outer_mask: wp.array[bool],
    scalars: wp.array2d[float32],
    flags: wp.array2d[int32],
    backtrack_mask: wp.array[bool],
):
    """Clear the accumulators and transient flags for one APGD iteration."""
    wid = wp.tid()
    backtrack_mask[wid] = False
    if not outer_mask[wid]:
        return
    scalars[wid, _OBJ_CANDIDATE] = float32(0.0)
    scalars[wid, _OBJ_MODEL] = float32(0.0)
    scalars[wid, _RESIDUAL_SQUARED] = float32(0.0)
    scalars[wid, _RESTART_DOT] = float32(0.0)
    scalars[wid, _GAMMA_NORM_SQUARED] = float32(0.0)
    flags[wid, _UPDATE_BEST] = int32(0)
    flags[wid, _RESTART] = int32(0)
    flags[wid, _BACKTRACKS_THIS_ITERATION] = int32(0)


@wp.kernel
def compute_apgd_gradient(
    contact_count: wp.array[int32],
    contact_row_offset: wp.array[int32],
    world_mask: wp.array[bool],
    operator_product: wp.array[float32],
    rhs: wp.array[float32],
    gradient: wp.array[float32],
):
    """Compute the contact objective gradient ``A*y - b``."""
    wid, local_row = wp.tid()
    if not world_mask[wid] or local_row >= int32(3) * contact_count[wid]:
        return
    row = contact_row_offset[wid] + local_row
    gradient[row] = operator_product[row] - rhs[row]


@wp.kernel
def project_apgd_step(
    contact_count: wp.array[int32],
    contact_row_offset: wp.array[int32],
    contact_offset: wp.array[int32],
    world_mask: wp.array[bool],
    friction: wp.array[float32],
    y: wp.array[float32],
    gradient: wp.array[float32],
    scalars: wp.array2d[float32],
    gamma_new: wp.array[float32],
):
    """Apply one associated Coulomb-cone projected-gradient step."""
    wid, cid = wp.tid()
    if not world_mask[wid] or cid >= contact_count[wid]:
        return

    row = contact_row_offset[wid] + int32(3) * cid
    step = scalars[wid, _T]
    candidate = vec3f(
        y[row] - step * gradient[row],
        y[row + int32(1)] - step * gradient[row + int32(1)],
        y[row + int32(2)] - step * gradient[row + int32(2)],
    )
    mu = wp.max(friction[contact_offset[wid] + cid], float32(0.0))
    projected = project_to_coulomb_cone(candidate, mu)
    gamma_new[row] = projected[0]
    gamma_new[row + int32(1)] = projected[1]
    gamma_new[row + int32(2)] = projected[2]


@wp.kernel
def reduce_apgd_objectives(
    contact_count: wp.array[int32],
    contact_row_offset: wp.array[int32],
    contact_offset: wp.array[int32],
    world_mask: wp.array[bool],
    rhs: wp.array[float32],
    y: wp.array[float32],
    gradient: wp.array[float32],
    gamma_new: wp.array[float32],
    operator_gamma_new: wp.array[float32],
    scalars: wp.array2d[float32],
    partials: wp.array2d[float32],
):
    """Write one contact's objective and backtracking-model contributions."""
    wid, cid = wp.tid()
    if not world_mask[wid] or cid >= contact_count[wid]:
        return

    row = contact_row_offset[wid] + int32(3) * cid
    lipschitz = scalars[wid, _L]
    objective = float32(0.0)
    model = float32(0.0)
    for component in range(3):
        vector_row = row + component
        candidate = gamma_new[vector_row]
        momentum = y[vector_row]
        b = rhs[vector_row]
        grad = gradient[vector_row]
        ay = grad + b
        difference = candidate - momentum
        objective += float32(0.5) * candidate * operator_gamma_new[vector_row] - candidate * b
        model += (
            float32(0.5) * momentum * ay
            - momentum * b
            + grad * difference
            + float32(0.5) * lipschitz * difference * difference
        )
    partial = contact_offset[wid] + cid
    partials[_ACC_OBJ_CANDIDATE, partial] = objective
    partials[_ACC_OBJ_MODEL, partial] = model


@wp.kernel
def update_backtracking_condition(
    world_mask: wp.array[bool],
    config: wp.array[ContactAPGDConfigStruct],
    scalars: wp.array2d[float32],
    flags: wp.array2d[int32],
    backtrack_mask: wp.array[bool],
    backtrack_continue: wp.array[int32],
    status: wp.array[ContactAPGDStatus],
):
    """Grow ``L`` for violated descent bounds and update the inner-loop mask."""
    wid = wp.tid()
    needs_backtrack = False
    if world_mask[wid]:
        objective = scalars[wid, _OBJ_CANDIDATE]
        model = scalars[wid, _OBJ_MODEL]
        scale = wp.max(wp.abs(objective), wp.abs(model))
        comparison_tolerance = float32(1.0e-6) * scale + float32(1.0e-12)
        world_status = status[wid]
        violated = not wp.isfinite(objective) or not wp.isfinite(model) or objective > model + comparison_tolerance
        backtracks = flags[wid, _BACKTRACKS_THIS_ITERATION]
        max_backtracks = config[wid].max_backtrack_iterations
        if violated and backtracks < max_backtracks:
            backtracks += int32(1)
            flags[wid, _BACKTRACKS_THIS_ITERATION] = backtracks
            world_status.backtracks += int32(1)
            lipschitz = wp.max(float32(2.0) * scalars[wid, _L], float32(1.0e-12))
            scalars[wid, _L] = lipschitz
            scalars[wid, _T] = float32(1.0) / lipschitz
            status[wid] = world_status
            # The initial descent check is pass one, matching final
            # newton-dvi.  A doubling on the last allowed pass updates L for
            # the next outer iteration but does not launch another projection.
            needs_backtrack = backtracks < max_backtracks

    backtrack_mask[wid] = needs_backtrack
    if needs_backtrack:
        wp.atomic_add(backtrack_continue, 0, int32(1))


@wp.kernel
def reduce_restart_and_res4(
    contact_count: wp.array[int32],
    contact_row_offset: wp.array[int32],
    partial_contact_offset: wp.array[int32],
    contact_offset: wp.array[int32],
    world_mask: wp.array[bool],
    friction: wp.array[float32],
    rhs: wp.array[float32],
    gamma: wp.array[float32],
    gamma_new: wp.array[float32],
    gradient: wp.array[float32],
    operator_gamma_new: wp.array[float32],
    scalars: wp.array2d[float32],
    partials: wp.array2d[float32],
):
    """Write one contact's restart, Res4, and iterate-norm contributions."""
    wid, cid = wp.tid()
    if not world_mask[wid] or cid >= contact_count[wid]:
        return

    row = contact_row_offset[wid] + int32(3) * cid
    gdiff = scalars[wid, _GDIFF]
    inv_gdiff = float32(1.0) / gdiff
    grad_new = vec3f(
        operator_gamma_new[row] - rhs[row],
        operator_gamma_new[row + int32(1)] - rhs[row + int32(1)],
        operator_gamma_new[row + int32(2)] - rhs[row + int32(2)],
    )
    candidate = vec3f(gamma_new[row], gamma_new[row + int32(1)], gamma_new[row + int32(2)])
    mu = wp.max(friction[contact_offset[wid] + cid], float32(0.0))
    projected = project_to_coulomb_cone(candidate - gdiff * grad_new, mu)
    residual_vector = inv_gdiff * (candidate - projected)

    old = vec3f(gamma[row], gamma[row + int32(1)], gamma[row + int32(2)])
    grad_at_y = vec3f(gradient[row], gradient[row + int32(1)], gradient[row + int32(2)])
    partial = partial_contact_offset[wid] + cid
    partials[_ACC_RESIDUAL_SQUARED, partial] = wp.dot(residual_vector, residual_vector)
    partials[_ACC_RESTART_DOT, partial] = wp.dot(grad_at_y, candidate - old)
    partials[_ACC_GAMMA_NORM_SQUARED, partial] = wp.dot(candidate, candidate)


@wp.kernel
def update_apgd_worlds(
    outer_mask: wp.array[bool],
    config: wp.array[ContactAPGDConfigStruct],
    scalars: wp.array2d[float32],
    flags: wp.array2d[int32],
    status: wp.array[ContactAPGDStatus],
):
    """Advance Nesterov state, latch convergence, and recover the step size."""
    wid = wp.tid()
    if not outer_mask[wid]:
        return

    world_status = status[wid]
    world_status.iterations += int32(1)
    residual = wp.sqrt(scalars[wid, _RESIDUAL_SQUARED])
    best_residual = scalars[wid, _BEST_RESIDUAL]
    if wp.isfinite(residual) and residual < best_residual:
        best_residual = residual
        scalars[wid, _BEST_RESIDUAL] = residual
        flags[wid, _UPDATE_BEST] = int32(1)
    world_status.residual = best_residual

    if world_status.iterations >= config[wid].min_iterations and best_residual <= config[wid].tolerance:
        world_status.converged = int32(1)

    restart = scalars[wid, _RESTART_DOT] > float32(0.0)
    if restart:
        flags[wid, _RESTART] = int32(1)
        world_status.restarts += int32(1)

    theta = scalars[wid, _THETA]
    theta_new = float32(0.5) * (-theta * theta + theta * wp.sqrt(theta * theta + float32(4.0)))
    beta = theta * (float32(1.0) - theta) / (theta * theta + theta_new)
    if restart:
        theta_new = float32(1.0)
        beta = float32(0.0)
    scalars[wid, _THETA] = theta_new
    scalars[wid, _BETA] = beta

    lipschitz = wp.max(float32(0.9) * scalars[wid, _L], float32(1.0e-12))
    scalars[wid, _L] = lipschitz
    scalars[wid, _T] = float32(1.0) / lipschitz
    status[wid] = world_status


@wp.kernel
def update_apgd_vectors(
    contact_count: wp.array[int32],
    contact_row_offset: wp.array[int32],
    outer_mask: wp.array[bool],
    scalars: wp.array2d[float32],
    flags: wp.array2d[int32],
    gamma_new: wp.array[float32],
    y: wp.array[float32],
    gamma: wp.array[float32],
    gamma_best: wp.array[float32],
):
    """Store the best iterate and apply Nesterov momentum or restart."""
    wid, local_row = wp.tid()
    if not outer_mask[wid] or local_row >= int32(3) * contact_count[wid]:
        return

    row = contact_row_offset[wid] + local_row
    candidate = gamma_new[row]
    if flags[wid, _UPDATE_BEST] != int32(0):
        gamma_best[row] = candidate

    previous = gamma[row]
    if flags[wid, _RESTART] != int32(0):
        y[row] = candidate
    else:
        y[row] = candidate + scalars[wid, _BETA] * (candidate - previous)
    gamma[row] = candidate


@wp.kernel
def update_outer_condition(
    phase_mask: wp.array[bool],
    config: wp.array[ContactAPGDConfigStruct],
    outer_mask: wp.array[bool],
    outer_continue: wp.array[int32],
    status: wp.array[ContactAPGDStatus],
):
    """Update per-world activity and the batch APGD loop condition."""
    wid = wp.tid()
    world_status = status[wid]
    active = (
        phase_mask[wid]
        and (config[wid].early_exit == int32(0) or world_status.converged == int32(0))
        and world_status.iterations < config[wid].max_iterations
    )
    outer_mask[wid] = active
    if active:
        wp.atomic_add(outer_continue, 0, int32(1))


@wp.kernel
def copy_best_to_solution(
    contact_count: wp.array[int32],
    contact_row_offset: wp.array[int32],
    phase_mask: wp.array[bool],
    gamma_best: wp.array[float32],
    solution: wp.array[float32],
):
    """Copy the minimum-Res4 contact iterate to the caller-owned solution."""
    wid, local_row = wp.tid()
    if not phase_mask[wid] or local_row >= int32(3) * contact_count[wid]:
        return
    row = contact_row_offset[wid] + local_row
    solution[row] = gamma_best[row]


@wp.kernel
def dense_contact_matvec(
    problem_dim: wp.array[int32],
    problem_mio: wp.array[int32],
    problem_vio: wp.array[int32],
    problem_nc: wp.array[int32],
    problem_ccgo: wp.array[int32],
    contact_row_offset: wp.array[int32],
    world_mask: wp.array[bool],
    matrix: wp.array[float32],
    represented_compliance: wp.array[float32],
    x: wp.array[float32],
    y: wp.array[float32],
):
    """Apply the represented dense contact block ``D_CC + E_hat_CC``."""
    wid, local_row = wp.tid()
    row_count = int32(3) * problem_nc[wid]
    if not world_mask[wid] or local_row >= row_count:
        return

    dimension = problem_dim[wid]
    matrix_offset = problem_mio[wid]
    contact_group_offset = problem_ccgo[wid]
    compact_offset = contact_row_offset[wid]
    full_row = contact_group_offset + local_row
    value = float32(0.0)
    for local_column in range(row_count):
        full_column = contact_group_offset + local_column
        value += matrix[matrix_offset + dimension * full_row + full_column] * x[compact_offset + local_column]
    value += represented_compliance[problem_vio[wid] + full_row] * x[compact_offset + local_row]
    y[compact_offset + local_row] = value


@wp.kernel
def build_dense_contact_rhs(
    problem_dim: wp.array[int32],
    problem_mio: wp.array[int32],
    problem_vio: wp.array[int32],
    problem_nc: wp.array[int32],
    problem_ccgo: wp.array[int32],
    contact_row_offset: wp.array[int32],
    world_mask: wp.array[bool],
    matrix: wp.array[float32],
    free_velocity: wp.array[float32],
    full_solution: wp.array[float32],
    rhs: wp.array[float32],
):
    """Build ``b_C = -(v_f,C + D_C,notC*lambda_notC)`` exactly once."""
    wid, local_row = wp.tid()
    contact_row_count = int32(3) * problem_nc[wid]
    if not world_mask[wid] or local_row >= contact_row_count:
        return

    dimension = problem_dim[wid]
    matrix_offset = problem_mio[wid]
    vector_offset = problem_vio[wid]
    contact_begin = problem_ccgo[wid]
    contact_end = contact_begin + contact_row_count
    full_row = contact_begin + local_row
    value = free_velocity[vector_offset + full_row]
    for column in range(dimension):
        if column < contact_begin or column >= contact_end:
            value += matrix[matrix_offset + dimension * full_row + column] * full_solution[vector_offset + column]
    rhs[contact_row_offset[wid] + local_row] = -value


@wp.kernel
def gather_dense_contact_solution(
    problem_vio: wp.array[int32],
    problem_nc: wp.array[int32],
    problem_ccgo: wp.array[int32],
    contact_row_offset: wp.array[int32],
    world_mask: wp.array[bool],
    full_solution: wp.array[float32],
    compact_solution: wp.array[float32],
):
    """Gather contact rows from Kamino's unified impulse vector."""
    wid, local_row = wp.tid()
    if not world_mask[wid] or local_row >= int32(3) * problem_nc[wid]:
        return
    full_row = problem_vio[wid] + problem_ccgo[wid] + local_row
    compact_solution[contact_row_offset[wid] + local_row] = full_solution[full_row]


@wp.kernel
def scatter_dense_contact_solution(
    problem_vio: wp.array[int32],
    problem_nc: wp.array[int32],
    problem_ccgo: wp.array[int32],
    contact_row_offset: wp.array[int32],
    world_mask: wp.array[bool],
    compact_solution: wp.array[float32],
    full_solution: wp.array[float32],
):
    """Scatter solved contact rows into Kamino's unified impulse vector."""
    wid, local_row = wp.tid()
    if not world_mask[wid] or local_row >= int32(3) * problem_nc[wid]:
        return
    full_row = problem_vio[wid] + problem_ccgo[wid] + local_row
    full_solution[full_row] = compact_solution[contact_row_offset[wid] + local_row]
