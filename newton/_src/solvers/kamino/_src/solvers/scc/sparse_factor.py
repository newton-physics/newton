# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Kamino assembly and response adapters for the sparse LDL transpose backend."""

import warp as wp

from ...core.math import FLOAT32_EPS
from ...core.types import vec6f
from .kernels import BILATERAL_DIAGONAL_FLOOR, _compact_schur_fits

wp.set_module_options({"enable_backward": False})


def joint_block_pattern(model, capacities):
    """Partition bilateral rows by joint and connect groups sharing a body."""
    joints = model.joints
    origins = model.info.total_cts_offset.numpy()
    worlds = joints.wid.numpy()
    dynamic = joints.num_dynamic_cts.numpy()
    kinematic = joints.num_kinematic_cts.numpy()
    dynamic_offset = joints.dynamic_cts_offset_total_cts.numpy()
    kinematic_offset = joints.kinematic_cts_offset_total_cts.numpy()
    body_b, body_f = joints.bid_B.numpy(), joints.bid_F.numpy()
    groups = [[] for _ in capacities]
    incidence = [{} for _ in capacities]
    for joint, world in enumerate(worlds):
        rows = list(
            range(
                int(dynamic_offset[joint] - origins[world]),
                int(dynamic_offset[joint] - origins[world] + dynamic[joint]),
            )
        )
        rows += list(
            range(
                int(kinematic_offset[joint] - origins[world]),
                int(kinematic_offset[joint] - origins[world] + kinematic[joint]),
            )
        )
        if not rows:
            continue
        group = len(groups[world])
        groups[world].append(rows)
        for body in {int(body_b[joint]), int(body_f[joint])}:
            if body >= 0:
                incidence[world].setdefault(body, []).append(group)
    edges = []
    for world, bodies in enumerate(incidence):
        if not groups[world]:
            groups[world] = [[0]]  # The existing bilateral metadata uses one dummy row for empty worlds.
        edges.append(sorted({(i, j) for adjacent in bodies.values() for i in adjacent for j in adjacent if i < j}))
    return groups, edges


@wp.kernel
def _dense_values(
    dim: wp.array[wp.int32],
    njc: wp.array[wp.int32],
    mio: wp.array[wp.int32],
    vio: wp.array[wp.int32],
    worlds: wp.array[wp.int32],
    rows: wp.array2d[wp.int32],
    block_i: wp.array[wp.int32],
    block_j: wp.array[wp.int32],
    matrix: wp.array[wp.float32],
    scale: wp.array[wp.float32],
    values: wp.array3d[wp.float32],
):
    """Gather and diagonally normalize dense bilateral blocks, regularizing their diagonal."""
    block, r, c = wp.tid()
    i, j = block_i[block], block_j[block]
    world = worlds[i]
    row, col = rows[i, r], rows[j, c]
    n = njc[world]
    value = wp.float32(0.0)
    if row >= 0 and col >= 0 and row < n and col < n:
        stride, start = dim[world], mio[world]
        pr = wp.sqrt(1.0 / (wp.abs(matrix[start + row * stride + row]) + FLOAT32_EPS))
        pc = wp.sqrt(1.0 / (wp.abs(matrix[start + col * stride + col]) + FLOAT32_EPS))
        value = pr * matrix[start + row * stride + col] * pc
        if row == col:
            value += BILATERAL_DIAGONAL_FLOOR
            scale[vio[world] + row] = pr
    elif i == j and r == c:
        value = wp.float32(1.0)
    values[block, r, c] = value


def factor_dense(solver, problem, scale):
    """Gather the same normalized bilateral operator as the dense LLT path."""
    wp.launch(
        _dense_values,
        dim=solver.values.shape,
        inputs=[
            problem.data.dim,
            problem.data.njc,
            problem.data.mio,
            solver.operator.info.vio,
            solver.worlds,
            solver.rows,
            solver.block_i,
            solver.block_j,
            problem.data.D,
            scale,
            solver.values,
        ],
        device=solver.device,
    )
    solver.factorize_values()


@wp.kernel
def _sparse_diagonal(
    njc: wp.array[wp.int32],
    pvio: wp.array[wp.int32],
    bvio: wp.array[wp.int32],
    worlds: wp.array[wp.int32],
    rows: wp.array2d[wp.int32],
    diagonal: wp.array[wp.float32],
    scale: wp.array[wp.float32],
    values: wp.array3d[wp.float32],
):
    """Initialize normalized bilateral diagonals and record row scales, including identity padding."""
    pivot, r = wp.tid()
    world, row = worlds[pivot], rows[pivot, r]
    value = wp.float32(1.0)
    if row >= 0 and row < njc[world]:
        diag = wp.abs(diagonal[pvio[world] + row])
        p = wp.sqrt(1.0 / (diag + FLOAT32_EPS))
        scale[bvio[world] + row] = p
        value = p * diag * p + BILATERAL_DIAGONAL_FLOOR
    values[pivot, r, r] = value


@wp.kernel
def _sparse_entries(
    inv_m: wp.array[wp.float32],
    inv_i: wp.array[wp.mat33],
    starts: wp.array[wp.int32],
    worlds: wp.array[wp.int32],
    rows: wp.array[wp.int32],
    cols: wp.array[wp.int32],
    bodies: wp.array[wp.int32],
    pair_i: wp.array[wp.int32],
    pair_j: wp.array[wp.int32],
    jacobian: wp.array[vec6f],
    vio: wp.array[wp.int32],
    scale: wp.array[wp.float32],
    targets: wp.array[wp.int32],
    pivots: int,
    values: wp.array[wp.float32],
):
    """Assemble normalized off-diagonal bilateral entries from shared-body Jacobian products."""
    entry = wp.tid()
    first = starts[entry]
    value = wp.float32(0.0)
    for pair in range(first, starts[entry + 1]):
        a, b = jacobian[pair_i[pair]], jacobian[pair_j[pair]]
        linear_a, linear_b = wp.vec3(a[0], a[1], a[2]), wp.vec3(b[0], b[1], b[2])
        angular_a, angular_b = wp.vec3(a[3], a[4], a[5]), wp.vec3(b[3], b[4], b[5])
        body = bodies[pair]
        value += inv_m[body] * wp.dot(linear_a, linear_b) + wp.dot(angular_a, inv_i[body] @ angular_b)
    start = vio[worlds[first]]
    value = scale[start + rows[first]] * value * scale[start + cols[first]]
    target = targets[entry]
    values[target] = value
    if target // 36 < pivots:
        values[(target // 36) * 36 + (target % 6) * 6 + (target % 36) // 6] = value


def prepare_sparse(path):
    """Map the existing structural body pairs into sparse factor storage."""
    first = path.bilateral_entry_starts.numpy()[:-1]
    world, row, col = [a.numpy()[first] for a in path.bilateral_nzb_pairs[:3]]
    path.bilateral_entry_target = wp.array(
        path.bilateral_solver.entry_offsets(world, row, col), dtype=wp.int32, device=path.device
    )


def factor_sparse(path, problem):
    """Assemble sparse entries without allocating a dense bilateral matrix."""
    solver, state = path.bilateral_solver, path.data.state
    solver.values.zero_()
    state.bilateral_preconditioner.zero_()
    problem.delassus.diagonal(state.scratch)
    wp.launch(
        _sparse_diagonal,
        dim=(solver.pivot_count, 6),
        inputs=[
            problem.data.njc,
            problem.data.vio,
            solver.operator.info.vio,
            solver.worlds,
            solver.rows,
            state.scratch,
            state.bilateral_preconditioner,
            solver.values,
        ],
        device=path.device,
    )
    wp.launch(
        _sparse_entries,
        dim=path.bilateral_entry_target.size,
        inputs=[
            path.model.bodies.inv_m_i,
            path.model_data.bodies.inv_I_i,
            path.bilateral_entry_starts,
            *path.bilateral_nzb_pairs,
            problem.delassus.constraint_jacobian.nzb_values,
            solver.operator.info.vio,
            state.bilateral_preconditioner,
            path.bilateral_entry_target,
            solver.pivot_count,
            solver.values.flatten(),
        ],
        device=path.device,
    )
    solver.factorize_values()


@wp.kernel
def _prepare_response(
    dim: wp.array[wp.int32],
    njc: wp.array[wp.int32],
    vio: wp.array[wp.int32],
    offsets: wp.array[wp.int32],
    capacity: wp.array[wp.int32],
    scale: wp.array[wp.float32],
    source: wp.array[wp.float32],
    dense: bool,
    source_mio: wp.array[wp.int32],
    compact: bool,
    origins: wp.array[wp.int32],
    strides: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    full_columns: wp.array[wp.int32],
    whitened_columns: wp.array[wp.int32],
    result: wp.array[wp.float32],
):
    """Scale bilateral coupling columns and select the columns each response pass must solve.

    Dense responses start at the first unilateral column of the projected operator.
    Worlds using compact Schur keep only the whitened forward columns, so they skip
    the full-response diagonal and backward passes.
    """
    world, row, column = wp.tid()
    n, nu = njc[world], dim[world] - njc[world]
    stride = capacity[world]
    origin = offsets[world]
    whitened = False
    if dense:
        stride = dim[world]
        origin += n
    elif compact and _compact_schur_fits(n, nu, stride):
        stride = nu
        whitened = True
    if row == 0 and column == 0:
        origins[world] = origin
        strides[world] = stride
        columns[world] = nu
        full_columns[world] = 0 if whitened else nu
        whitened_columns[world] = nu if whitened else 0
    if row >= n or column >= nu:
        return
    value = wp.float32(0.0)
    if dense:
        value = source[source_mio[world] + row * stride + n + column]
    else:
        value = source[origin + row * stride + column]
    result[origin + row * stride + column] = scale[vio[world] + row] * value


@wp.kernel
def _finish_response(
    njc: wp.array[wp.int32],
    vio: wp.array[wp.int32],
    origins: wp.array[wp.int32],
    strides: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    whitened_columns: wp.array[wp.int32],
    scale: wp.array[wp.float32],
    forward: wp.array[wp.float32],
    full: wp.array[wp.float32],
):
    """Store whitened compact responses or undo normalization on full inverse responses."""
    world, row, column = wp.tid()
    if row >= njc[world] or column >= columns[world]:
        return
    index = origins[world] + row * strides[world] + column
    if whitened_columns[world] > 0:
        full[index] = forward[index]
    else:
        full[index] *= scale[vio[world] + row]


def allocate_response(state, world_count, device):
    """Allocate per-world response layout arrays before graph capture."""
    state.bilateral_response_origin = wp.zeros(world_count, dtype=wp.int32, device=device)
    state.bilateral_response_active_stride = wp.zeros_like(state.bilateral_response_origin)
    state.bilateral_response_columns = wp.zeros_like(state.bilateral_response_origin)
    state.bilateral_response_full_columns = wp.zeros_like(state.bilateral_response_origin)
    state.bilateral_response_whitened_columns = wp.zeros_like(state.bilateral_response_origin)


def _response_layout(state, problem, columns, max_columns):
    """Return the shared response RHS layout for one column selection."""
    return {
        "offsets": state.bilateral_response_origin,
        "strides": state.bilateral_response_active_stride,
        "columns": columns,
        "max_columns": max_columns,
        "active_dim": problem.data.njc,
    }


def response_sparse(path, problem, compact):
    """Build full responses, or whitened columns for worlds using compact Schur."""
    solver, state = path.bilateral_solver, path.data.state
    max_columns = max(path.unilateral_strides_host, default=0)
    dims = (path.size.num_worlds, max(1, path.size.max_of_num_bilateral_joint_cts), max(1, max_columns))
    wp.launch(
        _prepare_response,
        dim=dims,
        inputs=[
            problem.data.dim,
            problem.data.njc,
            solver.operator.info.vio,
            state.bilateral_response_mio,
            state.bilateral_response_stride,
            state.bilateral_preconditioner,
            state.bilateral_coupling,
            False,
            problem.data.mio,
            compact,
            state.bilateral_response_origin,
            state.bilateral_response_active_stride,
            state.bilateral_response_columns,
            state.bilateral_response_full_columns,
            state.bilateral_response_whitened_columns,
            state.bilateral_response_factor,
        ],
        device=path.device,
    )
    # Keep forward-only columns separately because some worlds may exceed compact capacity.
    solver.solve_triangular(
        state.bilateral_response_factor,
        state.bilateral_response_factor,
        forward=True,
        **_response_layout(state, problem, state.bilateral_response_columns, max_columns),
    )
    full = _response_layout(state, problem, state.bilateral_response_full_columns, max_columns)
    solver.solve_diagonal(state.bilateral_response_factor, state.bilateral_response, **full)
    solver.solve_triangular(state.bilateral_response, state.bilateral_response, forward=False, **full)
    if compact:
        # Z = D^-1/2 L^-1 B gives Z^T Z = B^T A^-1 B for the compact Schur product.
        solver.solve_diagonal(
            state.bilateral_response_factor,
            state.bilateral_response_factor,
            inverse_sqrt=True,
            **_response_layout(state, problem, state.bilateral_response_whitened_columns, max_columns),
        )
    wp.launch(
        _finish_response,
        dim=dims,
        inputs=[
            problem.data.njc,
            solver.operator.info.vio,
            state.bilateral_response_origin,
            state.bilateral_response_active_stride,
            state.bilateral_response_columns,
            state.bilateral_response_whitened_columns,
            state.bilateral_preconditioner,
            state.bilateral_response_factor,
            state.bilateral_response,
        ],
        device=path.device,
    )


def response_dense(owner, problem):
    """Write full normalized responses for the unilateral columns of the dense projected operator."""
    solver, state = owner._bilateral_solver, owner.data.state
    max_columns = owner._max_unilateral_rows
    wp.launch(
        _prepare_response,
        dim=(owner.size.num_worlds, max(1, owner.size.max_of_num_bilateral_joint_cts), max(1, max_columns)),
        inputs=[
            problem.data.dim,
            problem.data.njc,
            solver.operator.info.vio,
            state.projected_mio,
            problem.data.dim,
            state.bilateral_preconditioner,
            problem.data.D,
            True,
            problem.data.mio,
            False,
            state.bilateral_response_origin,
            state.bilateral_response_active_stride,
            state.bilateral_response_columns,
            state.bilateral_response_full_columns,
            state.bilateral_response_whitened_columns,
            state.projected_D,
        ],
        device=owner.device,
    )
    solver.solve_matrix(
        state.projected_D,
        state.projected_D,
        **_response_layout(state, problem, state.bilateral_response_full_columns, max_columns),
    )
