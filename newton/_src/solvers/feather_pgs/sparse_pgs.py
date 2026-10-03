# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Serial PGS over ancestor-sparse, mass-whitened constraint rows."""

from functools import cache

import warp as wp

from .friction import FRICTION_PAIR_CUDA
from .kernels import (
    PGS_CONSTRAINT_TYPE_CONTACT,
    PGS_CONSTRAINT_TYPE_FRICTION,
    PGS_CONSTRAINT_TYPE_JOINT_LIMIT,
    PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT,
)


@cache
def _get_pgs_solve_sparse_kernel(
    max_constraints: int,
    max_world_dofs: int,
    max_row_dofs: int,
    *,
    mf_max_constraints: int = 0,
    free_row_dofs: int = 0,
) -> wp.Kernel:
    """Build a CUDA sweep over factored articulations and physical free bodies.

    Rows contain unique factor-coordinate indices, padded with -1. Both
    tangent rows of a contact must have identical, ordered support. The caller
    supplies ``row_incident = J v_hat``. Factor blocks store ``Z = J P^T L^-T``
    once; physical free blocks store J and their response separately. All
    coordinates accumulate deltas from v_hat; only factor blocks need decoding.

    This uses the existing interleaved row order and paired friction law,
    including linked patch normal loads. The caller admits only augmented
    drives, immediate response, and no warm start, torsion or ordinary joint
    velocity-limit rows. Seeded patch impulses retain the ordinary delta semantics.
    Independent free-body rows follow the articulated rows in every sweep,
    retaining their own response, capacity, friction law, and impulse buffer.

    Launch with ``ceil(world_count / 2)`` tiles and 64 threads per tile.
    """
    M, D, S = int(max_constraints), int(max_world_dofs), int(max_row_dofs)
    F = int(mf_max_constraints)
    P = int(free_row_dofs)
    if M <= 0 or D <= 0 or S <= 0 or S > D or F < 0 or P < 0 or P > S:
        raise ValueError("sparse PGS requires positive capacities and max_row_dofs <= max_world_dofs")
    W = 2
    Q = (S + 31) // 32
    contact_type = int(PGS_CONSTRAINT_TYPE_CONTACT)
    friction_type = int(PGS_CONSTRAINT_TYPE_FRICTION)
    limit_type = int(PGS_CONSTRAINT_TYPE_JOINT_LIMIT)
    velocity_limit_type = int(PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT)
    mf_storage = mf_initialize = mf_sweep = mf_writeback = ""
    if F:
        mf_storage = f"""
    __shared__ float mf_impulse_storage[{W * F}];
    __shared__ float mf_incident_storage[{W * F}];
    const int mf_offset = world * {F};
    const int mf_count = min(mf_constraint_count.data[world], {F});
    float* mf_impulse = mf_impulse_storage + slot * {F};
    float* mf_incident = mf_incident_storage + slot * {F};
"""
        mf_initialize = f"""
    for (int i = lane; i < mf_count; i += 32) {{
        const int global_row = mf_offset + i;
        mf_impulse[i] = mf_impulses.data[global_row];
        const int dofs = mf_meta.data[global_row * 4];
        const int first = dofs >> 16;
        const int second = (dofs << 16) >> 16;
        float incident = 0.0f;
        for (int k = 0; k < 6; ++k) {{
            if (first >= 0)
                incident += mf_J_a.data[global_row * 6 + k]
                    * v_hat.data[world_dof_indices.data[world * {D} + first + k]];
            if (second >= 0)
                incident += mf_J_b.data[global_row * 6 + k]
                    * v_hat.data[world_dof_indices.data[world * {D} + second + k]];
        }}
        mf_incident[i] = incident;
    }}
"""
        mf_sweep = f"""
        for (int row = 0; row < mf_count; ++row) {{
            const int global_row = mf_offset + row;
            const int4 meta = *reinterpret_cast<const int4*>(&mf_meta.data[global_row * 4]);
            const int kind = meta.w & 0xffff;
            const bool friction = kind == {friction_type};
            if (friction && global_iteration < friction_start_iteration) {{
                if (lane == 0) mf_impulse[row] = 0.0f;
                __syncwarp(MASK);
                continue;
            }}
            const int parent = meta.w >> 16;
            if (friction && row != parent + 1) continue;
            const int sibling = friction ? parent + 2 : -1;
            const float inverse_diagonal = __int_as_float(meta.y);
            if (!friction && inverse_diagonal <= 0.0f) continue;
            const int first = meta.x >> 16;
            const int second = (meta.x << 16) >> 16;
            int index = -1;
            if (lane < 6 && first >= 0) index = first + lane;
            if (lane >= 6 && lane < 12 && second >= 0) index = second + lane - 6;
            float left = 0.0f, right = 0.0f, sibling_left = 0.0f, sibling_right = 0.0f;
            if (index >= 0 && lane < 6) {{
                left = mf_J_a.data[global_row * 6 + lane];
                right = mf_MiJt_a.data[global_row * 6 + lane];
                if (friction) {{
                    sibling_left = mf_J_a.data[(mf_offset + sibling) * 6 + lane];
                    sibling_right = mf_MiJt_a.data[(mf_offset + sibling) * 6 + lane];
                }}
            }}
            if (index >= 0 && lane >= 6 && lane < 12) {{
                left = mf_J_b.data[global_row * 6 + lane - 6];
                right = mf_MiJt_b.data[global_row * 6 + lane - 6];
                if (friction) {{
                    sibling_left = mf_J_b.data[(mf_offset + sibling) * 6 + lane - 6];
                    sibling_right = mf_MiJt_b.data[(mf_offset + sibling) * 6 + lane - 6];
                }}
            }}
            // A coordinate has one writer, including coincident endpoint blocks.
            if (first >= 0 && first == second) {{
                if (lane < 6) {{
                    left += mf_J_b.data[global_row * 6 + lane];
                    right += mf_MiJt_b.data[global_row * 6 + lane];
                    if (friction) {{
                        sibling_left += mf_J_b.data[(mf_offset + sibling) * 6 + lane];
                        sibling_right += mf_MiJt_b.data[(mf_offset + sibling) * 6 + lane];
                    }}
                }} else {{
                    index = -1;
                    left = right = sibling_left = sibling_right = 0.0f;
                }}
            }}
            float dot = index >= 0 ? left * delta[index] : 0.0f;
            float sibling_dot = index >= 0 ? sibling_left * delta[index] : 0.0f;
            float cross = left * sibling_right;
            for (int shift = 16; shift > 0; shift >>= 1) {{
                dot += __shfl_down_sync(MASK, dot, shift);
                sibling_dot += __shfl_down_sync(MASK, sibling_dot, shift);
                cross += __shfl_down_sync(MASK, cross, shift);
            }}
            const float residual = __shfl_sync(MASK, dot, 0)
                + mf_incident[row] + __int_as_float(meta.z);
            const float old_impulse = mf_impulse[row];
            float new_impulse = old_impulse;
            float sibling_change = 0.0f;
            if (friction) {{
                float normal_load = mf_impulse[parent];
                for (int next = mf_meta.data[(mf_offset + parent) * 4 + 3] >> 16;
                     next >= 0 && next != parent; next = mf_meta.data[(mf_offset + next) * 4 + 3] >> 16)
                    normal_load += mf_impulse[next];
                const float radius = fmaxf(mf_row_mu.data[global_row] * normal_load, 0.0f);
                const float old_sibling = mf_impulse[sibling];
                float2 pair = make_float2(0.0f, 0.0f);
                if (radius > 0.0f) {{
                    const float sibling_inverse = __int_as_float(mf_meta.data[(mf_offset + sibling) * 4 + 1]);
                    const float sibling_residual = __shfl_sync(MASK, sibling_dot, 0)
                        + mf_incident[sibling]
                        + __int_as_float(mf_meta.data[(mf_offset + sibling) * 4 + 2]);
                    pair = friction_pair_candidate(inverse_diagonal > 0.0f ? 1.0f / inverse_diagonal : 0.0f,
                        __shfl_sync(MASK, cross, 0), sibling_inverse > 0.0f ? 1.0f / sibling_inverse : 0.0f,
                        residual, sibling_residual, old_impulse, old_sibling, radius, omega);
                }}
                const float magnitude = sqrtf(pair.x * pair.x + pair.y * pair.y);
                const float scale = magnitude > radius ? radius / magnitude : 1.0f;
                new_impulse = pair.x * scale;
                const float new_sibling = pair.y * scale;
                sibling_change = new_sibling - old_sibling;
                if (lane == 0) mf_impulse[sibling] = new_sibling;
            }} else if (kind == {velocity_limit_type}) {{
                // Velocity limits apply a stateless, unrelaxed correction.
                new_impulse = residual < 0.0f ? -residual * inverse_diagonal : 0.0f;
            }} else {{
                new_impulse = old_impulse - omega * residual * inverse_diagonal;
                if (kind == {contact_type}) new_impulse = fmaxf(new_impulse, 0.0f);
            }}
            const float change = kind == {velocity_limit_type} ? new_impulse : new_impulse - old_impulse;
            if (lane == 0) mf_impulse[row] = new_impulse;
            if (change != 0.0f || sibling_change != 0.0f) {{
                changed = 1;
                if (index >= 0) {{
                    float value = delta[index];
                    if (sibling_change != 0.0f) value += sibling_right * sibling_change;
                    if (change != 0.0f) value += right * change;
                    delta[index] = value;
                }}
            }}
            __syncwarp(MASK);
        }}
"""
        mf_writeback = """
    for (int i = lane; i < mf_count; i += 32) mf_impulses.data[mf_offset + i] = mf_impulse[i];
"""
    snippet = (
        "#if defined(__CUDA_ARCH__)\n"
        + FRICTION_PAIR_CUDA
        + f"""
    __shared__ float delta_storage[{W * D}];
    __shared__ float impulse_storage[{W * M}];
    __shared__ float rhs_storage[{W * M}];
    __shared__ float diagonal_storage[{W * M}];
    __shared__ unsigned char type_storage[{W * M}];
    constexpr unsigned MASK = 0xffffffffu;
    const int lane = threadIdx.x & 31;
    const int slot = threadIdx.x >> 5;
    const int offset = world * {M};
    const int row_base = offset * {S};
    const int count = min(world_constraint_count.data[world], {M});
    float* delta = delta_storage + slot * {D};
    float* impulse = impulse_storage + slot * {M};
    float* rhs = rhs_storage + slot * {M};
    float* diagonal = diagonal_storage + slot * {M};
    unsigned char* type = type_storage + slot * {M};
{mf_storage}

    for (int d = lane; d < {D}; d += 32) delta[d] = 0.0f;
    for (int i = lane; i < count; i += 32) {{
        impulse[i] = world_impulses.data[offset + i];
        rhs[i] = rhs_bias.data[offset + i] + row_incident.data[offset + i];
        diagonal[i] = world_diag.data[offset + i];
        type[i] = static_cast<unsigned char>(world_row_type.data[offset + i]);
    }}
{mf_initialize}
    __syncwarp(MASK);

    for (int iteration = 0; iteration < iterations; ++iteration) {{
        const int global_iteration = iteration_offset + iteration;
        int changed = 0;
        for (int row = 0; row < count; ++row) {{
            const int row_type = static_cast<int>(type[row]);
            const bool friction = row_type == {friction_type};
            if (friction && global_iteration < friction_start_iteration) {{
                if (lane == 0) impulse[row] = 0.0f;
                __syncwarp(MASK);
                continue;
            }}
            const int parent = friction ? world_row_parent.data[offset + row] : -1;
            if (friction && row != parent + 1) continue;
            const int sibling = friction ? parent + 2 : -1;
            const float denominator = diagonal[row];
            if (!friction && denominator <= 0.0f) continue;

            int indices[{Q}];
            float responses[{Q}];
            float sibling_responses[{Q}];
            float dot = 0.0f;
            float sibling_dot = 0.0f;
            float cross = 0.0f;
            #pragma unroll
            for (int q = 0; q < {Q}; ++q) {{
                const int entry = lane + q * 32;
                const int index = entry < {S} ? row_dof.data[row_base + row * {S} + entry] : -1;
                const float value = index >= 0 ? row_factor.data[row_base + row * {S} + entry] : 0.0f;
                const float sibling_value = friction && index >= 0
                    ? row_factor.data[row_base + sibling * {S} + entry] : 0.0f;
                float response = value;
                float sibling_response = sibling_value;
                if ({str(bool(P)).lower()} && index >= 0 && world_free_dof_mask.data[world * {D} + index] != 0) {{
                    const int width = row_free_response.shape[2];
                    response = row_free_response.data[(offset + row) * width + entry];
                    if (friction)
                        sibling_response = row_free_response.data[(offset + sibling) * width + entry];
                }}
                indices[q] = index;
                responses[q] = response;
                sibling_responses[q] = sibling_response;
                if (index >= 0) {{
                    const float velocity = delta[index];
                    dot += value * velocity;
                    sibling_dot += sibling_value * velocity;
                    cross += value * sibling_response;
                }}
            }}
            for (int shift = 16; shift > 0; shift >>= 1) {{
                dot += __shfl_down_sync(MASK, dot, shift);
                sibling_dot += __shfl_down_sync(MASK, sibling_dot, shift);
                cross += __shfl_down_sync(MASK, cross, shift);
            }}
            const float residual = __shfl_sync(MASK, dot, 0) + rhs[row];
            const float old_impulse = impulse[row];
            float new_impulse = old_impulse;
            float sibling_change = 0.0f;
            if (friction) {{
                float normal_load = impulse[parent];
                for (int next = world_row_parent.data[offset + parent]; next >= 0 && next != parent;
                     next = world_row_parent.data[offset + next])
                    normal_load += impulse[next];
                const float radius = fmaxf(world_row_mu.data[offset + row] * normal_load, 0.0f);
                const float old_sibling = impulse[sibling];
                float2 pair = make_float2(0.0f, 0.0f);
                if (radius > 0.0f) {{
                    const float sibling_residual = __shfl_sync(MASK, sibling_dot, 0) + rhs[sibling];
                    pair = friction_pair_candidate(denominator, __shfl_sync(MASK, cross, 0),
                        diagonal[sibling], residual, sibling_residual, old_impulse, old_sibling, radius, omega);
                }}
                const float magnitude = sqrtf(pair.x * pair.x + pair.y * pair.y);
                const float scale = magnitude > radius ? radius / magnitude : 1.0f;
                new_impulse = pair.x * scale;
                const float new_sibling = pair.y * scale;
                sibling_change = new_sibling - old_sibling;
                if (lane == 0) impulse[sibling] = new_sibling;
            }} else {{
                new_impulse = old_impulse - omega * residual / denominator;
                if (row_type == {contact_type} || row_type == {limit_type})
                    new_impulse = fmaxf(new_impulse, 0.0f);
            }}
            const float change = new_impulse - old_impulse;
            if (lane == 0) impulse[row] = new_impulse;
            if (change != 0.0f || sibling_change != 0.0f) {{
                changed = 1;
                #pragma unroll
                for (int q = 0; q < {Q}; ++q) {{
                    if (indices[q] >= 0) {{
                        // Match the existing paired-tangent response order.
                        float value = delta[indices[q]];
                        if (sibling_change != 0.0f) value += sibling_responses[q] * sibling_change;
                        if (change != 0.0f) value += responses[q] * change;
                        delta[indices[q]] = value;
                    }}
                }}
            }}
            __syncwarp(MASK);
        }}
{mf_sweep}
        if (global_iteration >= friction_start_iteration && __ballot_sync(MASK, changed != 0) == 0u) break;
    }}
    for (int d = lane; d < {D}; d += 32) factor_velocity_delta.data[world * {D} + d] = delta[d];
    for (int i = lane; i < count; i += 32) world_impulses.data[offset + i] = impulse[i];
{mf_writeback}
#endif
"""
    )

    @wp.func_native(snippet)
    def solve_native(
        world: int,
        world_constraint_count: wp.array[int],
        rhs_bias: wp.array2d[float],
        world_diag: wp.array2d[float],
        world_impulses: wp.array2d[float],
        row_dof: wp.array3d[int],
        row_factor: wp.array3d[float],
        row_free_response: wp.array3d[float],
        row_incident: wp.array2d[float],
        world_row_type: wp.array2d[int],
        world_row_parent: wp.array2d[int],
        world_row_mu: wp.array2d[float],
        world_free_dof_mask: wp.array2d[int],
        world_dof_indices: wp.array2d[int],
        v_hat: wp.array[float],
        mf_constraint_count: wp.array[int],
        mf_meta: wp.array2d[int],
        mf_impulses: wp.array2d[float],
        mf_J_a: wp.array3d[float],
        mf_J_b: wp.array3d[float],
        mf_MiJt_a: wp.array3d[float],
        mf_MiJt_b: wp.array3d[float],
        mf_row_mu: wp.array2d[float],
        iterations: int,
        omega: float,
        friction_start_iteration: int,
        iteration_offset: int,
        factor_velocity_delta: wp.array2d[float],
    ): ...

    def solve(
        world_count: int,
        world_constraint_count: wp.array[int],
        rhs_bias: wp.array2d[float],
        world_diag: wp.array2d[float],
        world_impulses: wp.array2d[float],
        row_dof: wp.array3d[int],
        row_factor: wp.array3d[float],
        row_free_response: wp.array3d[float],
        row_incident: wp.array2d[float],
        world_row_type: wp.array2d[int],
        world_row_parent: wp.array2d[int],
        world_row_mu: wp.array2d[float],
        world_free_dof_mask: wp.array2d[int],
        world_dof_indices: wp.array2d[int],
        v_hat: wp.array[float],
        mf_constraint_count: wp.array[int],
        mf_meta: wp.array2d[int],
        mf_impulses: wp.array2d[float],
        mf_J_a: wp.array3d[float],
        mf_J_b: wp.array3d[float],
        mf_MiJt_a: wp.array3d[float],
        mf_MiJt_b: wp.array3d[float],
        mf_row_mu: wp.array2d[float],
        iterations: int,
        omega: float,
        friction_start_iteration: int,
        iteration_offset: int,
        factor_velocity_delta: wp.array2d[float],
    ):
        block, lane = wp.tid()
        world = block * W + lane // 32
        if world < world_count:
            solve_native(
                world,
                world_constraint_count,
                rhs_bias,
                world_diag,
                world_impulses,
                row_dof,
                row_factor,
                row_free_response,
                row_incident,
                world_row_type,
                world_row_parent,
                world_row_mu,
                world_free_dof_mask,
                world_dof_indices,
                v_hat,
                mf_constraint_count,
                mf_meta,
                mf_impulses,
                mf_J_a,
                mf_J_b,
                mf_MiJt_a,
                mf_MiJt_b,
                mf_row_mu,
                iterations,
                omega,
                friction_start_iteration,
                iteration_offset,
                factor_velocity_delta,
            )

    solve.__name__ = f"pgs_solve_sparse_{M}_{D}_{S}_{F}_{P}"
    solve.__qualname__ = solve.__name__
    return wp.kernel(enable_backward=False, module="unique")(solve)
