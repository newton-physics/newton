# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Kernels of the LOX joint penalty seed.

Each thread handles one factor block and sums its structural rows in a fixed order, so
the seed is deterministic and identical blocks receive identical seeds.
"""

import warp as wp

from ...core.types import vec6f

###
# Module interface
###

__all__ = [
    "SEED_STATUS_NONFINITE",
    "SEED_STATUS_NOT_REPRESENTABLE",
    "SEED_STATUS_NO_POSITIVE_EIGENVALUE",
    "SEED_STATUS_VALID",
    "_apply_structural_transpose",
    "_estimate_block_scales",
    "_exclude_inertia_free_dofs",
    "_fill_random_vector",
    "_lanczos_step",
    "_start_lanczos",
    "_write_body_scales",
]

###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Constants
###

SEED_STATUS_VALID = wp.constant(0)
"""The seed of the factor block is valid."""

SEED_STATUS_NONFINITE = wp.constant(1)
"""The structural Delassus of the factor block is not finite."""

SEED_STATUS_NO_POSITIVE_EIGENVALUE = wp.constant(2)
"""The structural Delassus of the factor block has no resolvable positive eigenvalue."""

SEED_STATUS_NOT_REPRESENTABLE = wp.constant(3)
"""The seed of the factor block exceeds the float32 range."""

_BISECTION_STEPS = wp.constant(64)
"""Bisection steps for each extreme eigenvalue of a Lanczos tridiagonal."""

_FLOAT32_EPSILON = wp.constant(wp.float64(1.1920928955078125e-07))
_FLOAT32_MAX = wp.constant(wp.float64(3.4028234663852886e38))


###
# Functions
###


@wp.func
def _add_row_side(
    body: wp.int32,
    jacobian: vec6f,
    value: wp.float32,
    body_vector_index: wp.array[wp.int32],
    dof_excluded: wp.array[wp.int32],
    body_vector: wp.array[wp.float32],
):
    """Add ``value * jacobian`` to the packed vector of a dynamic body, over its DOFs with inertia."""
    if body >= 0:
        index = body_vector_index[body]
        if index >= 0:
            for k in range(6):
                if dof_excluded[index + k] == 0:
                    body_vector[index + k] += value * jacobian[k]


@wp.func
def _row_side_velocity(
    body: wp.int32,
    jacobian: vec6f,
    body_vector_index: wp.array[wp.int32],
    dof_excluded: wp.array[wp.int32],
    body_vector: wp.array[wp.float32],
) -> wp.float32:
    """Return ``jacobian . x`` over the DOFs with inertia of a dynamic body, from its packed vector ``x``."""
    velocity = wp.float32(0.0)
    if body >= 0:
        index = body_vector_index[body]
        if index >= 0:
            for k in range(6):
                if dof_excluded[index + k] == 0:
                    velocity += jacobian[k] * body_vector[index + k]
    return velocity


@wp.func
def _row_velocity(
    row: wp.int32,
    body_a: wp.array[wp.int32],
    body_b: wp.array[wp.int32],
    body_vector_index: wp.array[wp.int32],
    jacobian_a: wp.array[vec6f],
    jacobian_b: wp.array[vec6f],
    effective_mass: wp.array[wp.float32],
    dof_excluded: wp.array[wp.int32],
    body_vector: wp.array[wp.float32],
) -> wp.float32:
    """Return the entry of ``M^1/2 J x`` of a structural row."""
    velocity = _row_side_velocity(
        body_a[row], jacobian_a[row], body_vector_index, dof_excluded, body_vector
    ) + _row_side_velocity(body_b[row], jacobian_b[row], body_vector_index, dof_excluded, body_vector)
    return wp.sqrt(effective_mass[row]) * velocity


@wp.func
def _count_eigenvalues_below(
    alpha: wp.array2d[wp.float32],
    beta: wp.array2d[wp.float32],
    block: wp.int32,
    size: wp.int32,
    x: wp.float64,
) -> wp.int32:
    """Count the eigenvalues below ``x`` of the Lanczos tridiagonal of a block with its Sturm sequence."""
    below = wp.int32(0)
    pivot = wp.float64(1.0)
    for i in range(size):
        coupling = wp.float64(0.0)
        if i > 0:
            off_diagonal = wp.float64(beta[block, i - 1])
            coupling = off_diagonal * off_diagonal / pivot
        pivot = wp.float64(alpha[block, i]) - x - coupling
        # A vanishing pivot counts its eigenvalue below x
        if pivot == wp.float64(0.0):
            pivot = wp.float64(-1.0e-300)
        if pivot < wp.float64(0.0):
            below += 1
    return below


###
# Kernels
###


@wp.kernel
def _exclude_inertia_free_dofs(
    # Inputs:
    dim: wp.array[wp.int32],
    mio: wp.array[wp.int32],
    vio: wp.array[wp.int32],
    # Outputs:
    matrix: wp.array[wp.float32],
    dof_excluded: wp.array[wp.int32],
):
    """Give each DOF without inertia a unit pivot and exclude it from the structural Jacobians.

    A zero diagonal entry of the positive semi-definite smooth matrix makes its whole row and column
    zero, so the DOF is an exact null direction, which the pseudo-inverse of the matrix leaves out.
    """
    block, i = wp.tid()
    n = dim[block]
    if i >= n:
        return
    diagonal = mio[block] + i * n + i
    excluded = wp.where(matrix[diagonal] == 0.0, 1, 0)
    if excluded != 0:
        matrix[diagonal] = 1.0
    dof_excluded[vio[block] + i] = excluded


@wp.kernel
def _fill_random_vector(
    # Inputs:
    block_row_offset: wp.array[wp.int32],
    block_rows: wp.array[wp.int32],
    seed: wp.int32,
    # Outputs:
    row_vector: wp.array[wp.float32],
):
    """Fill the rows of each block with pseudo-random values keyed by their index within the block.

    Blocks with identical rows thus start from identical vectors and receive identical seeds.
    """
    block = wp.tid()
    first = block_row_offset[block]
    for k in range(first, block_row_offset[block + 1]):
        state = wp.rand_init(seed, k - first)
        row_vector[block_rows[k]] = wp.randf(state) - 0.5


@wp.kernel
def _apply_structural_transpose(
    # Inputs:
    block_row_offset: wp.array[wp.int32],
    block_rows: wp.array[wp.int32],
    dim: wp.array[wp.int32],
    vio: wp.array[wp.int32],
    body_a: wp.array[wp.int32],
    body_b: wp.array[wp.int32],
    body_vector_index: wp.array[wp.int32],
    jacobian_a: wp.array[vec6f],
    jacobian_b: wp.array[vec6f],
    effective_mass: wp.array[wp.float32],
    dof_excluded: wp.array[wp.int32],
    row_vector: wp.array[wp.float32],
    # Outputs:
    body_vector: wp.array[wp.float32],
):
    """Write ``J^T M^1/2 v`` over the structural rows of each factor block into its packed vector.

    One thread sums the rows of a block in a fixed order, so the result is deterministic.
    """
    block = wp.tid()
    start = vio[block]
    for i in range(dim[block]):
        body_vector[start + i] = 0.0
    for k in range(block_row_offset[block], block_row_offset[block + 1]):
        row = block_rows[k]
        value = wp.sqrt(effective_mass[row]) * row_vector[row]
        _add_row_side(body_a[row], jacobian_a[row], value, body_vector_index, dof_excluded, body_vector)
        _add_row_side(body_b[row], jacobian_b[row], value, body_vector_index, dof_excluded, body_vector)


@wp.kernel
def _start_lanczos(
    # Inputs:
    block_row_offset: wp.array[wp.int32],
    block_rows: wp.array[wp.int32],
    body_a: wp.array[wp.int32],
    body_b: wp.array[wp.int32],
    body_vector_index: wp.array[wp.int32],
    jacobian_a: wp.array[vec6f],
    jacobian_b: wp.array[vec6f],
    effective_mass: wp.array[wp.float32],
    dof_excluded: wp.array[wp.int32],
    solution: wp.array[wp.float32],
    # Outputs:
    lanczos_vector: wp.array[wp.float32],
    previous_vector: wp.array[wp.float32],
    steps: wp.array[wp.int32],
    active: wp.array[wp.int32],
    any_active: wp.array[wp.int32],
):
    """Start the Lanczos recurrence of each block from the normalized ``S v``, which lies in the range of ``S``.

    ``solution`` holds ``A^-1 J^T M^1/2 v``. A block whose ``S v`` vanishes stays inactive; the
    other blocks set ``any_active``.
    """
    block = wp.tid()
    first = block_row_offset[block]
    last = block_row_offset[block + 1]
    norm_sq = wp.float32(0.0)
    for k in range(first, last):
        row = block_rows[k]
        value = _row_velocity(
            row, body_a, body_b, body_vector_index, jacobian_a, jacobian_b, effective_mass, dof_excluded, solution
        )
        lanczos_vector[row] = value
        previous_vector[row] = 0.0
        norm_sq += value * value
    steps[block] = 0
    active[block] = 0
    if norm_sq > 0.0:
        inverse_norm = 1.0 / wp.sqrt(norm_sq)
        for k in range(first, last):
            row = block_rows[k]
            lanczos_vector[row] = lanczos_vector[row] * inverse_norm
        active[block] = 1
        any_active[0] = 1


@wp.kernel
def _lanczos_step(
    # Inputs:
    block_row_offset: wp.array[wp.int32],
    block_rows: wp.array[wp.int32],
    body_a: wp.array[wp.int32],
    body_b: wp.array[wp.int32],
    body_vector_index: wp.array[wp.int32],
    jacobian_a: wp.array[vec6f],
    jacobian_b: wp.array[vec6f],
    effective_mass: wp.array[wp.float32],
    dof_excluded: wp.array[wp.int32],
    solution: wp.array[wp.float32],
    steps_per_row: wp.int32,
    # Outputs:
    lanczos_vector: wp.array[wp.float32],
    previous_vector: wp.array[wp.float32],
    work_vector: wp.array[wp.float32],
    alpha: wp.array2d[wp.float32],
    beta: wp.array2d[wp.float32],
    steps: wp.array[wp.int32],
    active: wp.array[wp.int32],
    any_active: wp.array[wp.int32],
):
    """Advance the Lanczos recurrence of each active block by one step.

    ``solution`` holds ``A^-1 J^T M^1/2 q`` of the current Lanczos vector ``q``. Each block
    counts its own steps, so every launch takes the same arguments, as a captured loop requires.
    A block stops after ``steps_per_row`` steps per row, or when ``beta`` vanishes at an
    invariant subspace; the blocks that continue set ``any_active``.
    """
    block = wp.tid()
    if active[block] == 0:
        return
    step = steps[block]
    first = block_row_offset[block]
    last = block_row_offset[block + 1]
    diagonal = wp.float32(0.0)
    for k in range(first, last):
        row = block_rows[k]
        value = _row_velocity(
            row, body_a, body_b, body_vector_index, jacobian_a, jacobian_b, effective_mass, dof_excluded, solution
        )
        work_vector[row] = value
        diagonal += lanczos_vector[row] * value
    previous_off_diagonal = wp.float32(0.0)
    if step > 0:
        previous_off_diagonal = beta[block, step - 1]
    norm_sq = wp.float32(0.0)
    for k in range(first, last):
        row = block_rows[k]
        value = work_vector[row] - diagonal * lanczos_vector[row] - previous_off_diagonal * previous_vector[row]
        work_vector[row] = value
        norm_sq += value * value
    off_diagonal = wp.sqrt(norm_sq)
    alpha[block, step] = diagonal
    beta[block, step] = off_diagonal
    steps[block] = step + 1
    breakdown = off_diagonal <= 4.0 * wp.float32(_FLOAT32_EPSILON) * (wp.abs(diagonal) + previous_off_diagonal)
    if step + 1 >= steps_per_row * (last - first) or breakdown:
        active[block] = 0
        return
    inverse_off_diagonal = 1.0 / off_diagonal
    for k in range(first, last):
        row = block_rows[k]
        previous_vector[row] = lanczos_vector[row]
        lanczos_vector[row] = work_vector[row] * inverse_off_diagonal
    any_active[0] = 1


@wp.kernel
def _estimate_block_scales(
    # Inputs:
    block_row_offset: wp.array[wp.int32],
    alpha: wp.array2d[wp.float32],
    beta: wp.array2d[wp.float32],
    steps: wp.array[wp.int32],
    condition: wp.float64,
    # Outputs:
    block_scale: wp.array[wp.float32],
    status: wp.array[wp.int32],
):
    """Turn the extreme eigenvalues of the Lanczos tridiagonal of each block into its penalty scale.

    Bisection on the Sturm count gives ``mu_max`` and the smallest eigenvalue ``mu_min`` above the
    resolution ``eps * rows * mu_max`` of the float32 system. The scale is
    ``min(1 / mu_min, (condition - 1) / mu_max)``.
    """
    block = wp.tid()
    status[block] = SEED_STATUS_VALID
    row_count = block_row_offset[block + 1] - block_row_offset[block]
    if row_count == 0:
        return
    size = steps[block]
    if size == 0:
        status[block] = SEED_STATUS_NO_POSITIVE_EIGENVALUE
        return

    # Gershgorin interval of the tridiagonal
    lower = wp.float64(0.0)
    upper = wp.float64(0.0)
    for i in range(size):
        radius = wp.float64(0.0)
        if i > 0:
            radius += wp.abs(wp.float64(beta[block, i - 1]))
        if i + 1 < size:
            radius += wp.abs(wp.float64(beta[block, i]))
        center = wp.float64(alpha[block, i])
        if i == 0:
            lower = center - radius
            upper = center + radius
        else:
            lower = wp.min(lower, center - radius)
            upper = wp.max(upper, center + radius)
    if not (wp.isfinite(lower) and wp.isfinite(upper)):
        status[block] = SEED_STATUS_NONFINITE
        return

    # Largest eigenvalue: the smallest x with every eigenvalue below it
    low = lower
    high = upper
    for _ in range(_BISECTION_STEPS):
        middle = wp.float64(0.5) * (low + high)
        if _count_eigenvalues_below(alpha, beta, block, size, middle) >= size:
            high = middle
        else:
            low = middle
    mu_max = high
    threshold = _FLOAT32_EPSILON * wp.float64(row_count) * mu_max
    unresolved = _count_eigenvalues_below(alpha, beta, block, size, threshold)
    if mu_max <= wp.float64(0.0) or unresolved >= size:
        status[block] = SEED_STATUS_NO_POSITIVE_EIGENVALUE
        return

    # Smallest resolved eigenvalue: the smallest x with one more eigenvalue below it than below the threshold
    low = threshold
    high = mu_max
    for _ in range(_BISECTION_STEPS):
        middle = wp.float64(0.5) * (low + high)
        if _count_eigenvalues_below(alpha, beta, block, size, middle) > unresolved:
            high = middle
        else:
            low = middle
    mu_min = high

    scale = wp.min(wp.float64(1.0) / mu_min, (condition - wp.float64(1.0)) / mu_max)
    if not wp.isfinite(scale) or scale > _FLOAT32_MAX:
        status[block] = SEED_STATUS_NOT_REPRESENTABLE
        return
    block_scale[block] = wp.float32(scale)


@wp.kernel
def _write_body_scales(
    # Inputs:
    body_block: wp.array[wp.int32],
    block_row_offset: wp.array[wp.int32],
    block_scale: wp.array[wp.float32],
    block_status: wp.array[wp.int32],
    min_scale: wp.float32,
    # Outputs:
    scales: wp.array[wp.float32],
):
    """Give each body of a block with structural rows the larger of the minimum scale and its block seed.

    Blocks whose seed failed take the minimum scale.
    """
    body = wp.tid()
    block = body_block[body]
    if block >= 0 and block_row_offset[block + 1] > block_row_offset[block]:
        if block_status[block] == SEED_STATUS_VALID:
            scales[body] = wp.max(min_scale, block_scale[block])
        else:
            scales[body] = min_scale
