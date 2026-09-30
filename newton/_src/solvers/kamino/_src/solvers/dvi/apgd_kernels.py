# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Device operations for unilateral APGD with frozen De Saxce corrections.

Vectors use the full constraint layout: bilaterals, bounds, limits, then
contact triplets ``[t0, t1, n]``. Only unilateral rows are updated. Per-world
masks select active correction solves, inner iterations, and trial steps.
"""

import warp as wp

from ..padmm.math import project_to_coulomb_cone
from .types import DVIConfigStruct, DVIStatus

wp.set_module_options({"enable_backward": False})


@wp.struct
class APGDConfig:
    """Per-world budgets and tolerances for the nested solve."""

    max_iterations: wp.int32
    """Maximum accepted steps in each frozen-correction QP."""
    max_backtracks: wp.int32
    """Maximum trial steps per inner iteration, including the initial trial."""
    max_nonlinear_corrections: wp.int32
    """Maximum frozen-correction QPs per unilateral phase."""
    tolerance: wp.float32
    """Absolute infinity-norm tolerance for both natural-map residuals."""
    relaxation: wp.float32
    """Damping applied after each frozen-correction solve."""


@wp.struct
class APGDState:
    """Per-world acceleration and termination state."""

    lipschitz: wp.float32
    """Quadratic curvature estimate; its reciprocal is the trial step size."""
    theta: wp.float32
    """Nesterov acceleration parameter, reset for each frozen-correction QP."""
    iterations: wp.int32
    """Accepted steps in the current frozen-correction QP."""
    corrections: wp.int32
    """Correction solves started in the current unilateral phase."""
    backtracks: wp.int32
    """Rejected trial steps in the current inner iteration."""
    failed: wp.int32
    """Whether a line search failed during the current unilateral phase."""


@wp.func
def project_unilateral(
    row: wp.int32,
    nbc: wp.int32,
    nl: wp.int32,
    bcio: wp.int32,
    cio: wp.int32,
    mu: wp.array[wp.float32],
    lower: wp.array[wp.float32],
    upper: wp.array[wp.float32],
    value: wp.vec3f,
) -> wp.vec3f:
    """Project a scalar bound/limit or a complete contact triplet."""
    if row < nbc:
        return wp.vec3f(wp.clamp(value.x, lower[bcio + row], upper[bcio + row]), 0.0, 0.0)
    if row < nbc + nl:
        return wp.vec3f(wp.max(value.x, 0.0), 0.0, 0.0)
    return project_to_coulomb_cone(value, mu[cio + (row - nbc - nl) // 3])


@wp.kernel
def initialize_phase(
    dim: wp.array[wp.int32],
    njc: wp.array[wp.int32],
    config: wp.array[DVIConfigStruct],
    block_iteration: wp.int32,
    phase: wp.array[wp.bool],
    active: wp.array[wp.bool],
    state: wp.array[APGDState],
    condition: wp.array[wp.int32],
):
    """Select the worlds participating in this unilateral phase."""
    wid = wp.tid()
    enabled = dim[wid] > njc[wid] and (block_iteration < 0 or block_iteration < config[wid].max_alternating_iterations)
    phase[wid] = enabled
    active[wid] = enabled
    entry = APGDState()
    entry.lipschitz = 1.0
    state[wid] = entry
    if enabled:
        wp.atomic_add(condition, 0, 1)


@wp.kernel
def copy_unilateral(
    dim: wp.array[wp.int32],
    vio: wp.array[wp.int32],
    njc: wp.array[wp.int32],
    source: wp.array[wp.float32],
    target: wp.array[wp.float32],
):
    """Copy active unilateral entries and zero bilateral entries."""
    wid, row = wp.tid()
    if row < dim[wid]:
        i = vio[wid] + row
        target[i] = source[i] if row >= njc[wid] else 0.0


@wp.kernel
def build_phase_bias(
    dim: wp.array[wp.int32],
    vio: wp.array[wp.int32],
    njc: wp.array[wp.int32],
    full_product: wp.array[wp.float32],
    free_velocity: wp.array[wp.float32],
    unilateral_product: wp.array[wp.float32],
    bias: wp.array[wp.float32],
):
    """Hold the current bilateral contribution fixed, or eliminate it through Schur."""
    wid, row = wp.tid()
    if njc[wid] <= row and row < dim[wid]:
        i = vio[wid] + row
        bias[i] = full_product[i] + free_velocity[i] - unilateral_product[i]


@wp.kernel
def begin_correction(
    active: wp.array[wp.bool],
    state: wp.array[APGDState],
    inner: wp.array[wp.bool],
    condition: wp.array[wp.int32],
):
    """Restart acceleration for a new frozen-correction QP."""
    wid = wp.tid()
    inner[wid] = active[wid]
    if active[wid]:
        entry = state[wid]
        entry.theta = 1.0
        entry.iterations = 0
        entry.corrections += 1
        state[wid] = entry
        wp.atomic_add(condition, 0, 1)


@wp.kernel
def freeze_correction(
    dim: wp.array[wp.int32],
    vio: wp.array[wp.int32],
    njc: wp.array[wp.int32],
    ccgo: wp.array[wp.int32],
    cio: wp.array[wp.int32],
    mu: wp.array[wp.float32],
    active: wp.array[wp.bool],
    product: wp.array[wp.float32],
    bias: wp.array[wp.float32],
    x: wp.array[wp.float32],
    shift: wp.array[wp.float32],
    y: wp.array[wp.float32],
):
    """Freeze the De Saxce shift at the current outer impulse.

    ``product + bias`` is the velocity ``A x + b``. Store
    ``mu * norm(v_t)`` on contact normal rows, zero all other shifts, and
    initialize the extrapolated iterate ``y`` from ``x``.
    """
    wid, row = wp.tid()
    if not active[wid] or row < njc[wid] or row >= dim[wid]:
        return
    i = vio[wid] + row
    value = wp.float32(0.0)
    if row >= ccgo[wid] and (row - ccgo[wid]) % 3 == 2:
        vt = wp.vec2f(product[i - 2] + bias[i - 2], product[i - 1] + bias[i - 1])
        value = mu[cio[wid] + (row - ccgo[wid]) // 3] * wp.length(vt)
    shift[i] = value
    y[i] = x[i]


@wp.kernel
def begin_iteration(
    inner: wp.array[wp.bool],
    state: wp.array[APGDState],
    searching: wp.array[wp.bool],
    condition: wp.array[wp.int32],
):
    """Start a backtracking search for each active QP."""
    wid = wp.tid()
    searching[wid] = inner[wid]
    if inner[wid]:
        entry = state[wid]
        entry.backtracks = 0
        state[wid] = entry
        wp.atomic_add(condition, 0, 1)


@wp.kernel
def projected_step(
    dim: wp.array[wp.int32],
    vio: wp.array[wp.int32],
    njc: wp.array[wp.int32],
    nbc: wp.array[wp.int32],
    nl: wp.array[wp.int32],
    bcio: wp.array[wp.int32],
    cio: wp.array[wp.int32],
    mu: wp.array[wp.float32],
    lower: wp.array[wp.float32],
    upper: wp.array[wp.float32],
    searching: wp.array[wp.bool],
    state: wp.array[APGDState],
    y: wp.array[wp.float32],
    product: wp.array[wp.float32],
    bias: wp.array[wp.float32],
    shift: wp.array[wp.float32],
    candidate: wp.array[wp.float32],
):
    """Project ``y - (A y + b + s) / L`` onto the unilateral feasible set.

    ``product`` contains ``A y`` and ``shift`` contains the fixed correction
    ``s``. A backtracking retry changes ``L`` while retaining ``y`` and ``s``.
    """
    wid, row = wp.tid()
    nu = dim[wid] - njc[wid]
    scalar_rows = nbc[wid] + nl[wid]
    if not searching[wid] or row >= nu or (row >= scalar_rows and (row - scalar_rows) % 3 != 0):
        return
    count = 1 if row < scalar_rows else 3
    i = vio[wid] + njc[wid] + row
    value = wp.vec3f(0.0)
    for j in range(count):
        value[j] = y[i + j] - (product[i + j] + bias[i + j] + shift[i + j]) / state[wid].lipschitz
    value = project_unilateral(row, nbc[wid], nl[wid], bcio[wid], cio[wid], mu, lower, upper, value)
    for j in range(count):
        candidate[i + j] = value[j]


@wp.kernel
def check_descent(
    dim: wp.array[wp.int32],
    vio: wp.array[wp.int32],
    njc: wp.array[wp.int32],
    config: wp.array[APGDConfig],
    y: wp.array[wp.float32],
    product_y: wp.array[wp.float32],
    candidate: wp.array[wp.float32],
    product_candidate: wp.array[wp.float32],
    state: wp.array[APGDState],
    searching: wp.array[wp.bool],
    inner: wp.array[wp.bool],
    active: wp.array[wp.bool],
    condition: wp.array[wp.int32],
    status: wp.array[DVIStatus],
):
    """Check the quadratic upper bound and reject exhausted searches.

    For ``d = candidate - y``, require ``d.T A d <= L * norm(d)**2``.
    The fixed linear term ``b + s`` cancels from this test. This checks the
    frozen QP's curvature, not convergence of the nonlinear contact law.
    Non-finite curvature or bounds fail the search even if an infinity
    comparison would otherwise accept the trial.
    """
    wid = wp.tid()
    if not searching[wid]:
        return
    curvature = wp.float64(0.0)
    norm = wp.float64(0.0)
    for row in range(njc[wid], dim[wid]):
        i = vio[wid] + row
        delta = wp.float64(candidate[i] - y[i])
        curvature += delta * wp.float64(product_candidate[i] - product_y[i])
        norm += delta * delta
    entry = state[wid]
    bound = wp.float64(entry.lipschitz) * norm
    finite = wp.isfinite(curvature) and wp.isfinite(bound)
    accepted = finite and (
        curvature <= bound + wp.float64(1.0e-6) * wp.max(wp.abs(curvature), bound) + wp.float64(1.0e-20)
    )
    searching[wid] = False
    if not accepted:
        info = status[wid]
        info.apgd_backtracks += 1
        status[wid] = info
        entry.backtracks += 1
        entry.lipschitz *= 2.0
        if entry.backtracks >= config[wid].max_backtracks or not finite or not wp.isfinite(entry.lipschitz):
            entry.failed = 1
            inner[wid] = False
            active[wid] = False
        else:
            searching[wid] = True
            wp.atomic_add(condition, 0, 1)
    state[wid] = entry


@wp.func
def natural_residual(
    dim: wp.int32,
    vio: wp.int32,
    njc: wp.int32,
    nbc: wp.int32,
    nl: wp.int32,
    bcio: wp.int32,
    cio: wp.int32,
    mu: wp.array[wp.float32],
    lower: wp.array[wp.float32],
    upper: wp.array[wp.float32],
    x: wp.array[wp.float32],
    product: wp.array[wp.float32],
    bias: wp.array[wp.float32],
    shift: wp.array[wp.float32],
    nonlinear: wp.bool,
) -> wp.float32:
    """Evaluate the infinity norm of the unit-step unilateral natural map.

    With ``nonlinear=False``, use the frozen ``shift`` to measure inner QP
    convergence. Otherwise recompute the De Saxce shift from ``product +
    bias = A x + b`` to measure the nonlinear contact law at ``x``.
    """
    residual = wp.float32(0.0)
    row = wp.int32(0)
    while row < dim - njc:
        count = 1 if row < nbc + nl else 3
        i = vio + njc + row
        impulse = wp.vec3f(0.0)
        velocity = wp.vec3f(0.0)
        for j in range(count):
            impulse[j] = x[i + j]
            velocity[j] = product[i + j] + bias[i + j]
            if not nonlinear:
                velocity[j] += shift[i + j]
        if nonlinear and count == 3:
            velocity.z += mu[cio + (row - nbc - nl) // 3] * wp.length(wp.vec2f(velocity.x, velocity.y))
        projected = project_unilateral(row, nbc, nl, bcio, cio, mu, lower, upper, impulse - velocity)
        for j in range(count):
            difference = wp.abs(impulse[j] - projected[j])
            if not wp.isfinite(difference):
                difference = 3.0e38
            residual = wp.max(residual, difference)
        row += count
    return residual


@wp.kernel
def accept_iteration(
    dim: wp.array[wp.int32],
    vio: wp.array[wp.int32],
    njc: wp.array[wp.int32],
    nbc: wp.array[wp.int32],
    nl: wp.array[wp.int32],
    bcio: wp.array[wp.int32],
    cio: wp.array[wp.int32],
    mu: wp.array[wp.float32],
    lower: wp.array[wp.float32],
    upper: wp.array[wp.float32],
    config: wp.array[APGDConfig],
    product: wp.array[wp.float32],
    bias: wp.array[wp.float32],
    shift: wp.array[wp.float32],
    candidate: wp.array[wp.float32],
    x: wp.array[wp.float32],
    y: wp.array[wp.float32],
    state: wp.array[APGDState],
    inner: wp.array[wp.bool],
    condition: wp.array[wp.int32],
    status: wp.array[DVIStatus],
):
    """Accept the step, restart unhelpful momentum, and test the inner QP residual."""
    wid = wp.tid()
    if not inner[wid]:
        return
    entry = state[wid]
    restart = wp.float64(0.0)
    for row in range(njc[wid], dim[wid]):
        i = vio[wid] + row
        restart += wp.float64(y[i] - candidate[i]) * wp.float64(candidate[i] - x[i])
    theta = 0.5 * (1.0 + wp.sqrt(1.0 + 4.0 * entry.theta * entry.theta))
    beta = (entry.theta - 1.0) / theta
    if restart > wp.float64(0.0):
        theta = 1.0
        beta = 0.0
    for row in range(njc[wid], dim[wid]):
        i = vio[wid] + row
        y[i] = candidate[i] + beta * (candidate[i] - x[i])
        x[i] = candidate[i]
    entry.theta = theta
    entry.iterations += 1
    state[wid] = entry
    info = status[wid]
    info.iterations += 1
    status[wid] = info
    residual = natural_residual(
        dim[wid],
        vio[wid],
        njc[wid],
        nbc[wid],
        nl[wid],
        bcio[wid],
        cio[wid],
        mu,
        lower,
        upper,
        x,
        product,
        bias,
        shift,
        False,
    )
    inner[wid] = residual > config[wid].tolerance and entry.iterations < config[wid].max_iterations
    if inner[wid]:
        wp.atomic_add(condition, 0, 1)


@wp.kernel
def relax_correction(
    dim: wp.array[wp.int32],
    vio: wp.array[wp.int32],
    njc: wp.array[wp.int32],
    active: wp.array[wp.bool],
    config: wp.array[APGDConfig],
    previous: wp.array[wp.float32],
    x: wp.array[wp.float32],
):
    """Damp the nonlinear fixed-point update without leaving the convex feasible set."""
    wid, row = wp.tid()
    if active[wid] and njc[wid] <= row and row < dim[wid]:
        i = vio[wid] + row
        x[i] = previous[i] + config[wid].relaxation * (x[i] - previous[i])


@wp.kernel
def finish_correction(
    dim: wp.array[wp.int32],
    vio: wp.array[wp.int32],
    njc: wp.array[wp.int32],
    nbc: wp.array[wp.int32],
    nl: wp.array[wp.int32],
    bcio: wp.array[wp.int32],
    cio: wp.array[wp.int32],
    mu: wp.array[wp.float32],
    lower: wp.array[wp.float32],
    upper: wp.array[wp.float32],
    config: wp.array[APGDConfig],
    x: wp.array[wp.float32],
    product: wp.array[wp.float32],
    bias: wp.array[wp.float32],
    shift: wp.array[wp.float32],
    state: wp.array[APGDState],
    active: wp.array[wp.bool],
    condition: wp.array[wp.int32],
    status: wp.array[DVIStatus],
):
    """Stop only on the recomputed nonlinear residual or an explicit work limit."""
    wid = wp.tid()
    if not active[wid]:
        return
    residual = natural_residual(
        dim[wid],
        vio[wid],
        njc[wid],
        nbc[wid],
        nl[wid],
        bcio[wid],
        cio[wid],
        mu,
        lower,
        upper,
        x,
        product,
        bias,
        shift,
        True,
    )
    info = status[wid]
    info.apgd_residual = residual
    info.apgd_corrections += 1
    status[wid] = info
    active[wid] = residual > config[wid].tolerance and state[wid].corrections < config[wid].max_nonlinear_corrections
    if active[wid]:
        wp.atomic_add(condition, 0, 1)


@wp.kernel
def scatter_solution(
    dim: wp.array[wp.int32],
    vio: wp.array[wp.int32],
    njc: wp.array[wp.int32],
    phase: wp.array[wp.bool],
    x: wp.array[wp.float32],
    solution: wp.array[wp.float32],
):
    """Export unilateral impulses while preserving bilateral and masked-world entries."""
    wid, row = wp.tid()
    if phase[wid] and njc[wid] <= row and row < dim[wid]:
        i = vio[wid] + row
        solution[i] = x[i]


@wp.kernel
def finish_phase(phase: wp.array[wp.bool], state: wp.array[APGDState], status: wp.array[DVIStatus]):
    """Expose line-search failure even when the previous iterate was feasible."""
    wid = wp.tid()
    if phase[wid] and state[wid].failed != 0:
        info = status[wid]
        info.apgd_line_search_failed = 1
        info.apgd_residual = 3.0e38
        status[wid] = info


@wp.kernel
def guard_convergence(status: wp.array[DVIStatus]):
    """Prevent a failed APGD line search from being reported as converged."""
    wid = wp.tid()
    info = status[wid]
    if info.apgd_line_search_failed != 0:
        info.converged = 0
    status[wid] = info


@wp.kernel
def dense_matvec(
    dim: wp.array[wp.int32],
    mio: wp.array[wp.int32],
    vio: wp.array[wp.int32],
    matrix: wp.array[wp.float32],
    mask: wp.array[wp.bool],
    x: wp.array[wp.float32],
    y: wp.array[wp.float32],
):
    """Apply the existing dense Delassus matrix to active worlds."""
    wid, row = wp.tid()
    if not mask[wid] or row >= dim[wid]:
        return
    value = wp.float32(0.0)
    for column in range(dim[wid]):
        value += matrix[mio[wid] + row * dim[wid] + column] * x[vio[wid] + column]
    y[vio[wid] + row] = value


@wp.kernel
def copy_active(
    dim: wp.array[wp.int32],
    vio: wp.array[wp.int32],
    mask: wp.array[wp.bool],
    source: wp.array[wp.float32],
    target: wp.array[wp.float32],
):
    """Preserve completed worlds when a sparse product clears its output buffer."""
    wid, row = wp.tid()
    if mask[wid] and row < dim[wid]:
        i = vio[wid] + row
        target[i] = source[i]


@wp.kernel
def build_response_rhs(
    vio: wp.array[wp.int32],
    njc: wp.array[wp.int32],
    bvio: wp.array[wp.int32],
    scale: wp.array[wp.float32],
    product: wp.array[wp.float32],
    mask: wp.array[wp.bool],
    rhs: wp.array[wp.float32],
    active_dim: wp.array[wp.int32],
):
    """Build the scaled bilateral response to a unilateral search vector."""
    wid, row = wp.tid()
    if row == 0:
        active_dim[wid] = njc[wid] if mask[wid] else 0
    if row < njc[wid]:
        i = bvio[wid] + row
        rhs[i] = -scale[i] * product[vio[wid] + row] if mask[wid] else 0.0


@wp.kernel
def assemble_response(
    dim: wp.array[wp.int32],
    vio: wp.array[wp.int32],
    njc: wp.array[wp.int32],
    bvio: wp.array[wp.int32],
    scale: wp.array[wp.float32],
    response: wp.array[wp.float32],
    x: wp.array[wp.float32],
    full: wp.array[wp.float32],
):
    """Combine unilateral input with its eliminated bilateral response."""
    wid, row = wp.tid()
    if row < dim[wid]:
        value = x[vio[wid] + row]
        if row < njc[wid]:
            i = bvio[wid] + row
            value = scale[i] * response[i]
        full[vio[wid] + row] = value
