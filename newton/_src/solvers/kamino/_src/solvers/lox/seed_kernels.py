# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Kernels of the LOX joint penalty seed.

The seed runs a Golub-Kahan bidiagonalization of ``Q = J^T M^1/2``, which maps the structural rows of a
factor block to its body forces, with the ``A^-1`` inner product on the forces. Its adjoint is
``M^1/2 J A^-1``, so ``Q^* Q`` is the effective-mass-normalized structural Delassus
``S = M^1/2 J A^-1 J^T M^1/2``, and the squared singular values ``sigma^2`` of the bidiagonal ``B``
approximate the eigenvalues ``mu`` of ``S``. Rounding moves the singular values of ``B`` by about
``eps * sigma_max``, so the null space of ``S`` stays near ``eps^2 * mu_max`` in ``mu``. A Ritz value
sets the seed once its residual bound separates its singular value from that floor.

Each thread block of a step handles one factor block, and each body sums its incident structural rows
in a fixed order, so the seed is deterministic and identical blocks receive identical seeds.
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
    "_bidiagonal_step",
    "_estimate_block_scales",
    "_exclude_inertia_free_dofs",
    "_fill_random_vector",
    "_normalize_force",
    "_start_bidiagonalization",
    "_stop_bounded_blocks",
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
"""Bisection steps for each eigenvalue of ``B^T B``."""

_CERTIFICATION_CANDIDATES = wp.constant(32)
"""Smallest Ritz values above the singular floor of a block that a certification examines."""

_LARGEST_MARGIN = wp.constant(wp.float64(1.0e-3))
"""Relative margin of the bound on the largest Ritz value that the stop checks reuse."""

_NO_BOUND = wp.constant(wp.float64(1.0e308))
"""Bound of a Ritz value whose residual leaves its singular value unresolved."""

_FLOAT32_EPSILON = wp.constant(wp.float64(1.1920928955078125e-07))
_FLOAT64_EPSILON = wp.constant(wp.float64(2.220446049250313e-16))
_FLOAT32_MAX = wp.constant(wp.float64(3.4028234663852886e38))


###
# Functions
###


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
def _gram_diagonal(
    alpha: wp.array2d[wp.float32],
    beta: wp.array2d[wp.float32],
    block: wp.int32,
    i: wp.int32,
) -> wp.float64:
    """Return the diagonal entry ``i`` of ``B^T B`` for the upper bidiagonal ``B`` of a block.

    ``B`` has the diagonal ``alpha`` and the superdiagonal ``beta``.
    """
    a = wp.float64(alpha[block, i])
    value = a * a
    if i > 0:
        b = wp.float64(beta[block, i - 1])
        value += b * b
    return value


@wp.func
def _gram_off_diagonal(
    alpha: wp.array2d[wp.float32],
    beta: wp.array2d[wp.float32],
    block: wp.int32,
    i: wp.int32,
) -> wp.float64:
    """Return the entry ``(i, i + 1)`` of ``B^T B`` for the upper bidiagonal ``B`` of a block."""
    return wp.float64(alpha[block, i]) * wp.float64(beta[block, i])


@wp.func
def _count_eigenvalues_below(
    alpha: wp.array2d[wp.float32],
    beta: wp.array2d[wp.float32],
    block: wp.int32,
    size: wp.int32,
    x: wp.float64,
) -> wp.int32:
    """Count the eigenvalues below ``x`` of ``B^T B`` of a block with its Sturm sequence."""
    below = wp.int32(0)
    pivot = wp.float64(1.0)
    for i in range(size):
        coupling = wp.float64(0.0)
        if i > 0:
            off_diagonal = _gram_off_diagonal(alpha, beta, block, i - 1)
            coupling = off_diagonal * off_diagonal / pivot
        pivot = _gram_diagonal(alpha, beta, block, i) - x - coupling
        # A vanishing pivot counts its eigenvalue below x
        if pivot == wp.float64(0.0):
            pivot = wp.float64(-1.0e-300)
        if pivot < wp.float64(0.0):
            below += 1
    return below


@wp.func
def _largest_eigenvalue(
    alpha: wp.array2d[wp.float32],
    beta: wp.array2d[wp.float32],
    block: wp.int32,
    size: wp.int32,
) -> wp.float64:
    """Return the largest eigenvalue of ``B^T B`` of a block, or a non-finite value for a non-finite bidiagonal."""
    # Gershgorin interval of B^T B
    lower = wp.float64(0.0)
    upper = wp.float64(0.0)
    for i in range(size):
        radius = wp.float64(0.0)
        if i > 0:
            radius += wp.abs(_gram_off_diagonal(alpha, beta, block, i - 1))
        if i + 1 < size:
            radius += wp.abs(_gram_off_diagonal(alpha, beta, block, i))
        center = _gram_diagonal(alpha, beta, block, i)
        if i == 0:
            lower = center - radius
            upper = center + radius
        else:
            lower = wp.min(lower, center - radius)
            upper = wp.max(upper, center + radius)
    if not (wp.isfinite(lower) and wp.isfinite(upper)):
        return lower + upper

    # The smallest x with every eigenvalue below it
    low = lower
    high = upper
    for _ in range(_BISECTION_STEPS):
        middle = wp.float64(0.5) * (low + high)
        if _count_eigenvalues_below(alpha, beta, block, size, middle) >= size:
            high = middle
        else:
            low = middle
    return high


@wp.func
def _eigenvalue(
    alpha: wp.array2d[wp.float32],
    beta: wp.array2d[wp.float32],
    block: wp.int32,
    size: wp.int32,
    index: wp.int32,
    upper: wp.float64,
) -> wp.float64:
    """Return the eigenvalue ``index`` in increasing order of ``B^T B`` of a block, given a bound ``upper`` of its spectrum.

    ``B^T B`` is positive semi-definite up to rounding, so ``-upper`` bounds its spectrum from below.
    """
    low = -upper
    high = upper
    for _ in range(_BISECTION_STEPS):
        middle = wp.float64(0.5) * (low + high)
        if _count_eigenvalues_below(alpha, beta, block, size, middle) > index:
            high = middle
        else:
            low = middle
    return high


@wp.func
def _ritz_residual(
    alpha: wp.array2d[wp.float32],
    beta: wp.array2d[wp.float32],
    block: wp.int32,
    size: wp.int32,
    mu: wp.float64,
) -> wp.float64:
    """Return the residual bound ``|beta_k x_k|`` of the singular triplet of ``B`` with ``sigma = sqrt(mu)``.

    The right singular vector ``y`` is the eigenvector of ``B^T B``, whose squared last component is
    ``-1 / d'`` for the last Sturm pivot ``d`` at ``mu``, and the left singular vector has
    ``x_k = alpha_k y_k / sigma``. The derivatives follow the pivot recurrence
    ``d_i' = -1 + (c_i / d_{i-1})^2 d_{i-1}'`` with the off-diagonal entries ``c_i``. A non-finite
    derivative gives the bound ``|beta_k|`` of a unit ``x``.
    """
    last_beta = wp.abs(wp.float64(beta[block, size - 1]))
    sigma = wp.sqrt(wp.max(mu, wp.float64(0.0)))
    diagonal = _gram_diagonal(alpha, beta, block, 0)
    pivot = diagonal - mu
    growth = wp.float64(1.0)
    for i in range(1, size):
        if pivot == wp.float64(0.0):
            pivot = -_FLOAT64_EPSILON * (wp.abs(diagonal) + wp.abs(mu))
        off_diagonal = _gram_off_diagonal(alpha, beta, block, i - 1)
        ratio = off_diagonal / pivot
        growth = wp.float64(1.0) + ratio * ratio * growth
        diagonal = _gram_diagonal(alpha, beta, block, i)
        pivot = diagonal - mu - off_diagonal * ratio
    if not wp.isfinite(growth) or sigma == wp.float64(0.0):
        return last_beta
    last = wp.abs(wp.float64(alpha[block, size - 1])) / (sigma * wp.sqrt(growth))
    return last_beta * wp.min(last, wp.float64(1.0))


@wp.func
def _singular_floor(mu_max: wp.float64, row_count: wp.int32) -> wp.float64:
    """Return the singular value of ``B`` that the rounding of the float32 bidiagonalization of a block reaches.

    The rounding errors of the row sums of a block grow as the square root of its row count.
    """
    return _FLOAT32_EPSILON * wp.sqrt(wp.float64(row_count)) * wp.sqrt(mu_max)


@wp.func
def _certified_bound(
    alpha: wp.array2d[wp.float32],
    beta: wp.array2d[wp.float32],
    block: wp.int32,
    size: wp.int32,
    mu: wp.float64,
    floor: wp.float64,
) -> wp.float64:
    """Return the bound ``(sigma + r)^2`` on a positive eigenvalue of ``S`` that the Ritz value ``mu`` certifies.

    The singular triplet with residual ``r`` lies within ``r`` of a singular value of ``Q``, which is
    positive once ``sigma - r`` exceeds the singular floor. Other Ritz values give :data:`_NO_BOUND`.
    """
    sigma = wp.sqrt(wp.max(mu, wp.float64(0.0)))
    residual = _ritz_residual(alpha, beta, block, size, mu)
    if sigma - residual > floor:
        bound = sigma + residual
        return bound * bound
    return _NO_BOUND


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
    body_row_offset: wp.array[wp.int32],
    body_row_sides: wp.array[wp.int32],
    body_vector_index: wp.array[wp.int32],
    body_block: wp.array[wp.int32],
    jacobian_a: wp.array[vec6f],
    jacobian_b: wp.array[vec6f],
    effective_mass: wp.array[wp.float32],
    dof_excluded: wp.array[wp.int32],
    row_vector: wp.array[wp.float32],
    steps: wp.array[wp.int32],
    beta: wp.array2d[wp.float32],
    force_previous: wp.array[wp.float32],
    # Outputs:
    body_vector: wp.array[wp.float32],
):
    """Write ``J^T M^1/2 v - beta p`` over the structural rows of each factor block into its packed vector.

    ``beta`` is the last superdiagonal entry of the bidiagonal of the block and ``p`` its previous
    force, so a block without steps writes ``J^T M^1/2 v``. Each thread gathers the incident rows of
    one body in a fixed order, so the result is deterministic. ``body_row_sides`` holds
    ``2 * row + side`` for each incident row, with side 0 for body A and 1 for body B.
    """
    body = wp.tid()
    index = body_vector_index[body]
    if index < 0:
        return
    impulse = vec6f(0.0)
    for k in range(body_row_offset[body], body_row_offset[body + 1]):
        row = body_row_sides[k] // 2
        value = wp.sqrt(effective_mass[row]) * row_vector[row]
        if body_row_sides[k] % 2 == 0:
            impulse += value * jacobian_a[row]
        else:
            impulse += value * jacobian_b[row]
    block = body_block[body]
    step = steps[block]
    previous = wp.float32(0.0)
    if step > 0:
        previous = beta[block, step - 1]
    for k in range(6):
        force = impulse[k] - previous * force_previous[index + k]
        body_vector[index + k] = wp.where(dof_excluded[index + k] == 0, force, 0.0)


@wp.kernel
def _start_bidiagonalization(
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
    steps: wp.array[wp.int32],
    active: wp.array[wp.int32],
    any_active: wp.array[wp.int32],
):
    """Start the bidiagonalization of each block from the normalized ``S v``, which lies in the range of ``S``.

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
def _normalize_force(
    # Inputs:
    dim: wp.array[wp.int32],
    vio: wp.array[wp.int32],
    steps: wp.array[wp.int32],
    beta: wp.array2d[wp.float32],
    right_hand_side: wp.array[wp.float32],
    # Outputs:
    solution: wp.array[wp.float32],
    force_previous: wp.array[wp.float32],
    alpha: wp.array2d[wp.float32],
    active: wp.array[wp.int32],
):
    """Normalize the force ``p`` of each active block in the ``A^-1`` norm, which sets the diagonal entry ``alpha``.

    ``right_hand_side`` holds ``p`` and ``solution`` holds ``A^-1 p``, so ``alpha^2 = p . A^-1 p``.
    The normalized force becomes the previous force of the next step, and ``solution`` the
    normalized ``A^-1 p``. A block stops when ``alpha`` vanishes at an invariant subspace.

    One thread block handles each factor block, and the tile reduction sums the thread partials in a fixed order.
    """
    block, tid = wp.tid()
    if active[block] == 0:
        return
    step = steps[block]
    first = vio[block]
    last = first + dim[block]
    partial = wp.float64(0.0)
    for k in range(first + tid, last, wp.block_dim()):
        partial += wp.float64(right_hand_side[k]) * wp.float64(solution[k])
    norm = wp.float32(wp.sqrt(wp.max(wp.tile_sum(wp.tile(partial))[0], wp.float64(0.0))))
    previous = wp.float32(0.0)
    if step > 0:
        previous = beta[block, step - 1]
    breakdown = norm <= 4.0 * wp.float32(_FLOAT32_EPSILON) * previous
    if tid == 0:
        alpha[block, step] = norm
        if breakdown:
            active[block] = 0
    if breakdown:
        return
    inverse_norm = 1.0 / norm
    for k in range(first + tid, last, wp.block_dim()):
        force_previous[k] = right_hand_side[k] * inverse_norm
        solution[k] = solution[k] * inverse_norm


@wp.kernel
def _bidiagonal_step(
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
    alpha: wp.array2d[wp.float32],
    steps_per_row: wp.int32,
    # Outputs:
    lanczos_vector: wp.array[wp.float32],
    beta: wp.array2d[wp.float32],
    steps: wp.array[wp.int32],
    active: wp.array[wp.int32],
    any_active: wp.array[wp.int32],
):
    """Complete one bidiagonalization step of each active block with the next Lanczos vector.

    ``solution`` holds the normalized ``A^-1 p`` of the current force ``p``, so ``M^1/2 J A^-1 p``
    is its adjoint image, and ``w = M^1/2 J A^-1 p - alpha v`` sets the superdiagonal entry
    ``beta = |w|`` and the next Lanczos vector ``w / beta``. Each block counts its own steps, so
    every launch takes the same arguments, as a captured loop requires. A block stops after
    ``steps_per_row`` steps per row, or when ``beta`` vanishes at an invariant subspace; the blocks
    that continue set ``any_active``.

    One thread block handles each factor block. Each thread visits the same rows in every pass,
    and the tile reduction sums the thread partials in a fixed order.
    """
    block, tid = wp.tid()
    if active[block] == 0:
        return
    step = steps[block]
    first = block_row_offset[block]
    last = block_row_offset[block + 1]
    diagonal = alpha[block, step]
    partial = wp.float64(0.0)
    for k in range(first + tid, last, wp.block_dim()):
        row = block_rows[k]
        value = (
            _row_velocity(
                row, body_a, body_b, body_vector_index, jacobian_a, jacobian_b, effective_mass, dof_excluded, solution
            )
            - diagonal * lanczos_vector[row]
        )
        lanczos_vector[row] = value
        partial += wp.float64(value) * wp.float64(value)
    off_diagonal = wp.float32(wp.sqrt(wp.tile_sum(wp.tile(partial))[0]))
    breakdown = off_diagonal <= 4.0 * wp.float32(_FLOAT32_EPSILON) * diagonal
    stop = step + 1 >= steps_per_row * (last - first) or breakdown
    if tid == 0:
        beta[block, step] = off_diagonal
        steps[block] = step + 1
        if stop:
            active[block] = 0
        else:
            any_active[0] = 1
    if stop:
        return
    inverse_off_diagonal = 1.0 / off_diagonal
    for k in range(first + tid, last, wp.block_dim()):
        row = block_rows[k]
        lanczos_vector[row] = lanczos_vector[row] * inverse_off_diagonal


@wp.kernel
def _stop_bounded_blocks(
    # Inputs:
    block_row_offset: wp.array[wp.int32],
    alpha: wp.array2d[wp.float32],
    beta: wp.array2d[wp.float32],
    steps: wp.array[wp.int32],
    max_ratio: wp.float64,
    check_interval: wp.int32,
    # Outputs:
    largest: wp.array[wp.float64],
    next_check: wp.array[wp.int32],
    active: wp.array[wp.int32],
):
    """Stop the bidiagonalization of each block whose seed already takes the ratio cap.

    A block checks once its step count reaches ``next_check``, and checks again after
    ``max(check_interval, steps // 16)`` more steps. It stops once a Ritz value certifies a positive
    eigenvalue of ``S`` below the cap level ``mu_max / max_ratio``, or once its bidiagonal is
    not finite. The threads of a block share the Ritz values between the singular floor and the cap level.

    ``largest`` holds an upper bound of the largest Ritz value of each block from its previous check,
    initially zero. The largest Ritz value grows with the steps and lies within a factor
    ``1 + _LARGEST_MARGIN`` below a bound set by bisection, so the bound is reused while it holds,
    and the check uses the floor of the upper bound and the cap level of the lower bound. A block
    that stops thus takes the cap in :func:`_estimate_block_scales`.
    """
    block, tid = wp.tid()
    size = steps[block]
    if active[block] == 0 or size == 0 or size < next_check[block]:
        return
    upper = largest[block]
    if _count_eigenvalues_below(alpha, beta, block, size, upper) < size:
        upper = (wp.float64(1.0) + _LARGEST_MARGIN) * _largest_eigenvalue(alpha, beta, block, size)
    level = upper / ((wp.float64(1.0) + _LARGEST_MARGIN) * max_ratio)
    smallest = _NO_BOUND
    if wp.isfinite(upper) and upper > wp.float64(0.0):
        floor = _singular_floor(upper, block_row_offset[block + 1] - block_row_offset[block])
        first = _count_eigenvalues_below(alpha, beta, block, size, floor * floor)
        band = wp.min(_count_eigenvalues_below(alpha, beta, block, size, level) - first, _CERTIFICATION_CANDIDATES)
        for j in range(tid, band, wp.block_dim()):
            mu = _eigenvalue(alpha, beta, block, size, first + j, upper)
            smallest = wp.min(smallest, _certified_bound(alpha, beta, block, size, mu, floor))
    smallest = wp.tile_min(wp.tile(smallest))[0]
    if tid == 0:
        largest[block] = upper
        next_check[block] = size + wp.max(check_interval, size // 16)
        if not wp.isfinite(upper) or smallest <= level:
            active[block] = 0


@wp.kernel
def _estimate_block_scales(
    # Inputs:
    block_row_offset: wp.array[wp.int32],
    alpha: wp.array2d[wp.float32],
    beta: wp.array2d[wp.float32],
    steps: wp.array[wp.int32],
    min_ratio: wp.float64,
    max_ratio: wp.float64,
    # Outputs:
    block_scale: wp.array[wp.float32],
    status: wp.array[wp.int32],
):
    """Turn the bidiagonal of each block into its penalty scale.

    Bisection on the Sturm count of ``B^T B`` gives ``mu_max`` and the Ritz values above the singular
    floor. Each Ritz value whose residual separates its singular value from the floor certifies a
    positive eigenvalue of ``S`` below ``(sigma + r)^2``, and the smallest such bound ``mu_min`` sets
    the scale ``s = clamp(mu_max / mu_min, min_ratio, max_ratio) / mu_max``.
    The threads of a block share the
    :data:`_CERTIFICATION_CANDIDATES` smallest Ritz values above the floor, and the largest Ritz
    value also bounds ``mu_min``.
    """
    block, tid = wp.tid()
    row_count = block_row_offset[block + 1] - block_row_offset[block]
    if row_count == 0:
        if tid == 0:
            status[block] = SEED_STATUS_VALID
        return
    size = steps[block]
    result = SEED_STATUS_VALID
    mu_max = wp.float64(0.0)
    if size == 0:
        result = SEED_STATUS_NO_POSITIVE_EIGENVALUE
    else:
        mu_max = _largest_eigenvalue(alpha, beta, block, size)
        if not wp.isfinite(mu_max):
            result = SEED_STATUS_NONFINITE
        elif mu_max <= wp.float64(0.0):
            result = SEED_STATUS_NO_POSITIVE_EIGENVALUE

    smallest = _NO_BOUND
    if result == SEED_STATUS_VALID:
        floor = _singular_floor(mu_max, row_count)
        upper = (wp.float64(1.0) + _LARGEST_MARGIN) * mu_max
        first = _count_eigenvalues_below(alpha, beta, block, size, floor * floor)
        count = wp.min(size - first, _CERTIFICATION_CANDIDATES)
        for j in range(tid, count, wp.block_dim()):
            mu = _eigenvalue(alpha, beta, block, size, first + j, upper)
            smallest = wp.min(smallest, _certified_bound(alpha, beta, block, size, mu, floor))
        if tid == 0:
            smallest = wp.min(smallest, _certified_bound(alpha, beta, block, size, mu_max, floor))
    smallest = wp.tile_min(wp.tile(smallest))[0]
    if tid != 0:
        return

    if result == SEED_STATUS_VALID:
        if smallest >= _NO_BOUND:
            result = SEED_STATUS_NO_POSITIVE_EIGENVALUE
        else:
            scale = wp.clamp(mu_max / smallest, min_ratio, max_ratio) / mu_max
            if not wp.isfinite(scale) or scale > _FLOAT32_MAX:
                result = SEED_STATUS_NOT_REPRESENTABLE
            else:
                block_scale[block] = wp.float32(scale)
    status[block] = result


@wp.kernel
def _write_body_scales(
    # Inputs:
    body_block: wp.array[wp.int32],
    block_row_offset: wp.array[wp.int32],
    block_scale: wp.array[wp.float32],
    block_status: wp.array[wp.int32],
    fallback_scale: wp.float32,
    # Outputs:
    scales: wp.array[wp.float32],
):
    """Give each body of a block with structural rows its block seed.

    Blocks whose seed failed take the fallback scale.
    """
    body = wp.tid()
    block = body_block[body]
    if block >= 0 and block_row_offset[block + 1] > block_row_offset[block]:
        if block_status[block] == SEED_STATUS_VALID:
            scales[body] = block_scale[block]
        else:
            scales[body] = fallback_scale
