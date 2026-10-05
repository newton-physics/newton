# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Build contact responses in the sparse mass factor's coordinates."""

from functools import cache

import warp as wp

from .friction_patches import FrictionPatches
from .kernels import (
    PGS_CONSTRAINT_TYPE_CONTACT,
    PGS_CONSTRAINT_TYPE_JOINT_LIMIT,
    contact_restitution_fires,
    contact_tangent_basis,
)


@cache
def _get_sparse_contact_response_kernel(
    dof_count: int, *, lanes_per_contact: int = 8, warps_per_block: int = 4
) -> wp.Kernel:
    """Project contact rows with a lane group per contact and no global Jacobian.

    Contact directions share endpoint geometry, support and inverse-factor
    loads. Each endpoint Jacobian is evaluated once in shared scratch. Free
    bodies retain their physical coordinates and existing mass metric.

    All input/output arrays must be contiguous for the native flat accesses.
    Launch total_num_workers * lanes_per_contact threads in blocks of
    warps_per_block * 32. The CPU implementation uses one lane per group.
    """
    n, lanes, warps = int(dof_count), int(lanes_per_contact), int(warps_per_block)
    if not 1 <= n <= 64 or lanes not in (8, 16, 32) or not 1 <= warps <= 32:
        raise ValueError("Invalid sparse contact dimensions or lane group")
    groups = warps * 32 // lanes
    scratch_size = max(n, 6) + 6
    snippet = """
#if defined(__CUDA_ARCH__)
    const int lane = tid & ($LANES - 1);
    const int stride = $LANES;
    const int group = threadIdx.x / $LANES;
    const unsigned MASK = (0xffffffffu >> (32 - $LANES))
        << ((threadIdx.x & 31) & ~($LANES - 1));
    __shared__ float storage[$STORAGE];
    float* jacobian = storage + group * $SCRATCH;
    auto angular_a_shared = angular_a;
    auto angular_b_shared = angular_b;
    auto directions_shared = directions;
#pragma unroll
    for (int row = 0; row < 3; ++row) {
        if (row >= row_count) continue;
#pragma unroll
        for (int k = 0; k < 3; ++k) {
            angular_a_shared.data[row][k] = __shfl_sync(MASK, angular_a.data[row][k], 0, $LANES);
            angular_b_shared.data[row][k] = __shfl_sync(MASK, angular_b.data[row][k], 0, $LANES);
            directions_shared.data[row][k] = __shfl_sync(MASK, directions.data[row][k], 0, $LANES);
        }
    }
#else
    if ((tid & ($LANES - 1)) != 0) return;
    const int lane = 0;
    const int stride = 1;
    float jacobian[$SCRATCH];
    const auto angular_a_shared = angular_a;
    const auto angular_b_shared = angular_b;
    const auto directions_shared = directions;
#endif
    float* free_factor = jacobian + $JACOBIAN;
    const auto bit_count = [](unsigned long long mask) {
#if defined(__CUDA_ARCH__)
        return __popcll(mask);
#else
        int count = 0;
        while (mask != 0) { mask &= mask - 1; ++count; }
        return count;
#endif
    };
    const auto first_bit = [](unsigned long long mask) {
#if defined(__CUDA_ARCH__)
        return __ffsll(mask) - 1;
#else
        int column = 0;
        while ((mask & 1ull) == 0) { mask >>= 1; ++column; }
        return column;
#endif
    };
    const auto jacobian_value = [&](const auto& motion, const auto& angular_wrenches, int row) {
        const auto linear = wp::vec3(motion.c[0], motion.c[1], motion.c[2]);
        const auto angular = wp::vec3(motion.c[3], motion.c[4], motion.c[5]);
        const auto direction = wp::vec3(directions_shared.data[row][0],
            directions_shared.data[row][1], directions_shared.data[row][2]);
        const auto angular_wrench = wp::vec3(angular_wrenches.data[row][0],
            angular_wrenches.data[row][1], angular_wrenches.data[row][2]);
        return wp::dot(direction, linear) + wp::dot(angular_wrench, angular);
    };
    unsigned long long mask_a = 0, mask_b = 0;
    if (body_a >= 0 && art_a >= 0) mask_a = body_dof_mask.data[body_a];
    if (body_b >= 0 && art_b >= 0) mask_b = body_dof_mask.data[body_b];
    const int row_offset = world * row_dof.shape[1] + slot;
    const int support_capacity = row_dof.shape[2];
    int count = 0;
    auto norm = wp::vec3(0.0f);
    auto incident = wp::vec3(0.0f);
    // Physical blocks remain a short prefix; ordinary free/free contacts use
    // native MF storage, so production rows have at most one such block.
    int first_endpoint = 0;
    if (art_b >= 0 && is_free_rigid.data[art_b] != 0
        && (art_a < 0 || is_free_rigid.data[art_a] == 0)) first_endpoint = 1;
    for (int endpoint = 0; endpoint < 2; ++endpoint) {
        const int side = (first_endpoint + endpoint) % 2;
        const int art = side == 0 ? art_a : art_b;
        if (art < 0 || (side == 1 && art == art_a)
            || articulation_response_dof_count.data[art] <= 0) continue;
        const int group = art_group_idx.data[art];
        const int start = articulation_dof_start.data[art];
        const int offset = articulation_world_dof_offset.data[art];
        if (is_free_rigid.data[art] != 0) {
            for (int node = lane; node < 6; node += stride) {
                const auto motion = joint_S_s.data[start + node];
                const float velocity = v_hat.data[start + node];
#pragma unroll
                for (int row = 0; row < 3; ++row) {
                    if (row >= row_count) continue;
                    float value = 0.0f;
                    if (art == art_a) value += jacobian_value(motion, angular_a_shared, row);
                    if (art == art_b) value -= jacobian_value(motion, angular_b_shared, row);
                    jacobian[row * $PLANE + node] = value;
                    incident.c[row] += value * velocity;
                }
            }
#if defined(__CUDA_ARCH__)
            __syncwarp(MASK);
#endif
            const int factor_base = group * 36;
            for (int node = lane; node < 6; node += stride) {
                auto value = wp::vec3(0.0f);
                for (int column = 0; column <= node; ++column) {
                    const float factor = free_inverse_factor.data[factor_base + node * 6 + column];
#pragma unroll
                    for (int row = 0; row < 3; ++row) {
                        if (row < row_count) value.c[row] += factor * jacobian[row * $PLANE + column];
                    }
                }
#pragma unroll
                for (int row = 0; row < 3; ++row) {
                    if (row < row_count) free_factor[row * $PLANE + node] = value.c[row];
                }
            }
#if defined(__CUDA_ARCH__)
            __syncwarp(MASK);
#endif
            for (int node = lane; node < 6; node += stride) {
                auto response = wp::vec3(0.0f);
                for (int column = node; column < 6; ++column) {
                    const float factor = free_inverse_factor.data[factor_base + column * 6 + node];
#pragma unroll
                    for (int row = 0; row < 3; ++row) {
                        if (row < row_count) response.c[row] += factor * free_factor[row * $PLANE + column];
                    }
                }
#pragma unroll
                for (int row = 0; row < 3; ++row) {
                    if (row >= row_count) continue;
                    const int response_offset = (row_offset + row) * support_capacity;
                    const float value = jacobian[row * $PLANE + node];
                    row_dof.data[response_offset + count + node] = offset + node;
                    row_factor.data[response_offset + count + node] = value;
                    row_free_response.data[(row_offset + row) * row_free_response.shape[2] + count + node] = response.c[row];
                    norm.c[row] += value * response.c[row];
                }
            }
            count += 6;
        } else {
            unsigned long long mask = 0;
            if (art == art_a) mask |= mask_a;
            if (art == art_b) mask |= mask_b;
            for (int node = lane; node < $N; node += stride) {
                const unsigned long long bit = 1ull << node;
                if ((mask & bit) == 0) continue;
                const int physical = permutation.data[node];
                const auto motion = joint_S_s.data[start + physical];
                const float velocity = v_hat.data[start + physical];
#pragma unroll
                for (int row = 0; row < 3; ++row) {
                    if (row >= row_count) continue;
                    float value = 0.0f;
                    if (art == art_a && (mask_a & bit) != 0)
                        value += jacobian_value(motion, angular_a_shared, row);
                    if (art == art_b && (mask_b & bit) != 0)
                        value -= jacobian_value(motion, angular_b_shared, row);
                    jacobian[row * $PLANE + node] = value;
                    incident.c[row] += value * velocity;
                }
            }
#if defined(__CUDA_ARCH__)
            __syncwarp(MASK);
#endif
            const int factor_base = group * inverse_factor.shape[1];
            for (int node = lane; node < $N; node += stride) {
                const unsigned long long bit = 1ull << node;
                if ((mask & bit) == 0) continue;
                auto value = wp::vec3(0.0f);
                const int begin = factor_row_offsets.data[node];
                const int first = node - (factor_row_offsets.data[node + 1] - begin) + 1;
                // Reverse DFS packs this row's columns into [first, node].
                // Visit only supporting columns, including bit 63 without a shift by 64.
                unsigned long long pending = mask & (~0ull << first) & (~0ull >> (63 - node));
                while (pending != 0) {
                    const int column = first_bit(pending);
                    const int entry = begin + column - first;
                    const float factor = inverse_factor.data[factor_base + entry];
#pragma unroll
                    for (int row = 0; row < 3; ++row) {
                        if (row < row_count) value.c[row] += factor * jacobian[row * $PLANE + column];
                    }
                    pending &= pending - 1;
                }
                const int index = count + bit_count(mask & (bit - 1ull));
#pragma unroll
                for (int row = 0; row < 3; ++row) {
                    if (row >= row_count) continue;
                    const int response_offset = (row_offset + row) * support_capacity;
                    row_dof.data[response_offset + index] = offset + node;
                    row_factor.data[response_offset + index] = value.c[row];
                    norm.c[row] += value.c[row] * value.c[row];
                }
            }
            count += bit_count(mask);
        }
        // The next endpoint reuses the same group-local Jacobian storage.
#if defined(__CUDA_ARCH__)
        __syncwarp(MASK);
#endif
    }
#pragma unroll
    for (int row = 0; row < 3; ++row) {
        if (row >= row_count) continue;
        const int response_offset = (row_offset + row) * support_capacity;
        for (int index = count + lane; index < support_capacity; index += stride) {
            row_dof.data[response_offset + index] = -1;
            row_factor.data[response_offset + index] = 0.0f;
        }
#if defined(__CUDA_ARCH__)
        for (int shift = $LANES / 2; shift > 0; shift >>= 1) {
            norm.c[row] += __shfl_down_sync(MASK, norm.c[row], shift, $LANES);
            incident.c[row] += __shfl_down_sync(MASK, incident.c[row], shift, $LANES);
        }
#endif
        if (lane == 0) {
            row_incident.data[row_offset + row] = incident.c[row];
            diagonal.data[row_offset + row] = norm.c[row];
        }
    }
"""
    snippet = (
        snippet.replace("$STORAGE", str(groups * 3 * scratch_size))
        .replace("$LANES", str(lanes))
        .replace("$SCRATCH", str(3 * scratch_size))
        .replace("$PLANE", str(scratch_size))
        .replace("$JACOBIAN", str(max(n, 6)))
        .replace("$N", str(n))
    )

    @wp.func_native(snippet)
    def project_native(
        tid: int,
        body_a: int,
        body_b: int,
        art_a: int,
        art_b: int,
        world: int,
        slot: int,
        row_count: int,
        angular_a: wp.mat33,
        angular_b: wp.mat33,
        directions: wp.mat33,
        art_group_idx: wp.array[int],
        articulation_response_dof_count: wp.array[int],
        is_free_rigid: wp.array[int],
        articulation_world_dof_offset: wp.array[int],
        articulation_dof_start: wp.array[int],
        body_dof_mask: wp.array[wp.uint64],
        joint_S_s: wp.array[wp.spatial_vector],
        permutation: wp.array[int],
        factor_row_offsets: wp.array[int],
        inverse_factor: wp.array2d[float],
        free_inverse_factor: wp.array3d[float],
        v_hat: wp.array[float],
        row_dof: wp.array3d[int],
        row_factor: wp.array3d[float],
        row_free_response: wp.array3d[float],
        row_incident: wp.array2d[float],
        diagonal: wp.array2d[float],
    ): ...

    def contact_kernel(
        contact_count: wp.array[int],
        total_num_workers: int,
        contact_point0: wp.array[wp.vec3],
        contact_point1: wp.array[wp.vec3],
        contact_normal: wp.array[wp.vec3],
        contact_shape0: wp.array[int],
        contact_shape1: wp.array[int],
        contact_thickness0: wp.array[float],
        contact_thickness1: wp.array[float],
        contact_world: wp.array[int],
        contact_slot: wp.array[int],
        contact_art_a: wp.array[int],
        contact_art_b: wp.array[int],
        contact_path: wp.array[int],
        contact_slots_needed: wp.array[int],
        art_group_idx: wp.array[int],
        articulation_response_dof_count: wp.array[int],
        is_free_rigid: wp.array[int],
        articulation_world_dof_offset: wp.array[int],
        articulation_dof_start: wp.array[int],
        articulation_origin: wp.array[wp.vec3],
        body_dof_mask: wp.array[wp.uint64],
        joint_S_s: wp.array[wp.spatial_vector],
        shape_body: wp.array[int],
        body_q: wp.array[wp.transform],
        contact_friction_shared_anchor: int,
        friction_patches: FrictionPatches,
        contact_shared_anchor: int,
        permutation: wp.array[int],
        factor_row_offsets: wp.array[int],
        inverse_factor: wp.array2d[float],
        free_inverse_factor: wp.array3d[float],
        v_hat: wp.array[float],
        row_dof: wp.array3d[int],
        row_factor: wp.array3d[float],
        row_free_response: wp.array3d[float],
        row_incident: wp.array2d[float],
        diagonal: wp.array2d[float],
    ):
        tid = wp.tid()
        worker = tid // lanes
        total_contacts = wp.min(contact_count[0], contact_point0.shape[0])
        for c in range(worker, total_contacts, total_num_workers):
            row_count = wp.min(contact_slots_needed[c], 3)
            if contact_path[c] != 0 or contact_slot[c] < 0 or row_count <= 0:
                continue
            body_a = int(-1)
            body_b = int(-1)
            if contact_shape0[c] >= 0:
                body_a = shape_body[contact_shape0[c]]
            if contact_shape1[c] >= 0:
                body_b = shape_body[contact_shape1[c]]
            art_a = contact_art_a[c]
            art_b = contact_art_b[c]
            angular_a = wp.mat33(0.0)
            angular_b = wp.mat33(0.0)
            directions = wp.mat33(0.0)
            if tid % lanes == 0:
                normal = -contact_normal[c]
                point_a = contact_point0[c] - contact_thickness0[c] * normal
                point_b = contact_point1[c] + contact_thickness1[c] * normal
                if body_a >= 0:
                    point_a = wp.transform_point(body_q[body_a], contact_point0[c]) - contact_thickness0[c] * normal
                if body_b >= 0:
                    point_b = wp.transform_point(body_q[body_b], contact_point1[c]) + contact_thickness1[c] * normal
                midpoint = 0.5 * (point_a + point_b)
                if contact_shared_anchor != 0:
                    point_a = midpoint
                    point_b = midpoint
                friction_point_a = point_a
                friction_point_b = point_b
                tangent0 = wp.vec3(0.0)
                tangent1 = wp.vec3(0.0)
                if row_count > 1:
                    tangent0, tangent1 = contact_tangent_basis(normal)
                    if contact_friction_shared_anchor != 0:
                        friction_point_a = midpoint
                        friction_point_b = midpoint
                    if friction_patches.enabled != 0:
                        friction_point_a = friction_patches.point_a[c]
                        friction_point_b = friction_patches.point_b[c]
                for row in range(3):
                    if row >= row_count:
                        continue
                    direction = normal
                    anchor_a = point_a
                    anchor_b = point_b
                    if row != 0:
                        direction = tangent0
                        if row == 2:
                            direction = tangent1
                        anchor_a = friction_point_a
                        anchor_b = friction_point_b
                    # J = [direction, (point - origin) x direction] dot motion.
                    wrench_a = wp.vec3(0.0)
                    wrench_b = wp.vec3(0.0)
                    if art_a >= 0 and articulation_response_dof_count[art_a] > 0:
                        wrench_a = wp.cross(anchor_a - articulation_origin[art_a], direction)
                    if art_b >= 0 and articulation_response_dof_count[art_b] > 0:
                        wrench_b = wp.cross(anchor_b - articulation_origin[art_b], direction)
                    for axis in range(3):
                        angular_a[row, axis] = wrench_a[axis]
                        angular_b[row, axis] = wrench_b[axis]
                        directions[row, axis] = direction[axis]
            project_native(
                tid,
                body_a,
                body_b,
                art_a,
                art_b,
                contact_world[c],
                contact_slot[c],
                row_count,
                angular_a,
                angular_b,
                directions,
                art_group_idx,
                articulation_response_dof_count,
                is_free_rigid,
                articulation_world_dof_offset,
                articulation_dof_start,
                body_dof_mask,
                joint_S_s,
                permutation,
                factor_row_offsets,
                inverse_factor,
                free_inverse_factor,
                v_hat,
                row_dof,
                row_factor,
                row_free_response,
                row_incident,
                diagonal,
            )

    contact_kernel.__name__ = f"populate_sparse_contact_response_{n}_{lanes}_{warps}"
    contact_kernel.__qualname__ = contact_kernel.__name__
    return wp.kernel(enable_backward=False, module="unique")(contact_kernel)


@wp.func_native(
    """
    const int group = tid >> 5;
    if (group >= group_to_art.shape[0]) return;
#if defined(__CUDA_ARCH__)
    constexpr unsigned MASK = 0xffffffffu;
    const int lane = tid & 31;
    const int width = 32;
#else
    if ((tid & 31) != 0) return;
    const int lane = 0;
    const int width = 1;
#endif
    const auto bit_count = [](unsigned long long mask) {
#if defined(__CUDA_ARCH__)
        return __popcll(mask);
#else
        int count = 0;
        while (mask != 0) { mask &= mask - 1; ++count; }
        return count;
#endif
    };
    const int art = group_to_art.data[group];
    const int world = art_to_world.data[art];
    const int start = articulation_dof_start.data[art];
    const int dofs = inverse_permutation.shape[0];
    const int capacity = row_type.shape[1];
    const int support_capacity = row_dof.shape[2];
    const int factor_base = group * inverse_factor.shape[1];
    for (int base = 0; base < 2 * dofs; base += width) {
        const int candidate = base + lane;
        const int local_dof = candidate >> 1;
        const int side = candidate & 1;
        const int dof = start + local_dof;
        const int q_index = candidate < 2 * dofs ? limit_q_index.data[dof] : -1;
        float gap = 0.0f;
        int active = 0;
        if (q_index >= 0) {
            const float position = joint_q.data[q_index];
            const float bound = side == 0 ? joint_limit_lower.data[dof] : joint_limit_upper.data[dof];
            gap = side == 0 ? position - bound : bound - position;
            active = wp::isfinite(bound)
                && (side == 0 ? position <= bound + activation_gap : position >= bound - activation_gap);
        }
#if defined(__CUDA_ARCH__)
        const unsigned active_mask = __ballot_sync(MASK, active != 0);
        const int active_count = __popc(active_mask);
        int first_slot = 0;
        if (lane == 0 && active_count != 0)
            first_slot = atomicAdd(&slot_counter.data[world], active_count);
        first_slot = __shfl_sync(MASK, first_slot, 0);
#else
        const unsigned active_mask = active != 0 ? 1u : 0u;
        const int first_slot = slot_counter.data[world];
        slot_counter.data[world] += active;
#endif
        // Visit the ballot in lane order, matching the existing lower/upper DOF order.
        unsigned pending = active_mask;
        int rank = 0;
        while (pending != 0) {
            const int row = first_slot + rank;
            // The reservation above still counts every over-capacity row.
            if (row >= capacity) break;
#if defined(__CUDA_ARCH__)
            const int source = __ffs(pending) - 1;
            const int selected_dof = __shfl_sync(MASK, local_dof, source);
            const int selected_side = __shfl_sync(MASK, side, source);
            const float selected_gap = __shfl_sync(MASK, gap, source);
#else
            const int selected_dof = local_dof;
            const int selected_side = side;
            const float selected_gap = gap;
#endif
            const float sign = selected_side == 0 ? 1.0f : -1.0f;
            const int column = inverse_permutation.data[selected_dof];
            const unsigned long long support_mask = dof_mask.data[selected_dof];
            const int row_offset = world * capacity + row;
            const int response_offset = row_offset * support_capacity;
            float norm = 0.0f;
            for (int node = lane; node < dofs; node += width) {
                const unsigned long long bit = 1ull << node;
                if ((support_mask & bit) == 0) continue;
                const int index = bit_count(support_mask & (bit - 1ull));
                const int entry = factor_lookup.data[node * dofs + column];
                const float value = entry >= 0 ? sign * inverse_factor.data[factor_base + entry] : 0.0f;
                row_dof.data[response_offset + index] = articulation_world_dof_offset.data[art] + node;
                row_factor.data[response_offset + index] = value;
                norm += value * value;
            }
            const int support_count = bit_count(support_mask);
            for (int index = support_count + lane; index < support_capacity; index += width) {
                row_dof.data[response_offset + index] = -1;
                row_factor.data[response_offset + index] = 0.0f;
            }
#if defined(__CUDA_ARCH__)
            for (int shift = 16; shift > 0; shift >>= 1) norm += __shfl_down_sync(MASK, norm, shift);
#endif
            if (lane == 0) {
                row_incident.data[row_offset] = sign * v_hat.data[start + selected_dof];
                diagonal.data[row_offset] = norm;
                row_type.data[row_offset] = $LIMIT_TYPE;
                row_parent.data[row_offset] = -1;
                row_mu.data[row_offset] = 0.0f;
                row_beta.data[row_offset] = pgs_beta;
                row_cfm.data[row_offset] = pgs_cfm;
                phi.data[row_offset] = selected_gap;
                target_velocity.data[row_offset] = 0.0f;
            }
            pending &= pending - 1;
            ++rank;
        }
    }
""".replace("$LIMIT_TYPE", str(PGS_CONSTRAINT_TYPE_JOINT_LIMIT))
)
def _build_sparse_joint_limit_rows(
    tid: int,
    group_to_art: wp.array[int],
    art_to_world: wp.array[int],
    articulation_world_dof_offset: wp.array[int],
    articulation_dof_start: wp.array[int],
    limit_q_index: wp.array[int],
    joint_limit_lower: wp.array[float],
    joint_limit_upper: wp.array[float],
    joint_q: wp.array[float],
    activation_gap: float,
    pgs_beta: float,
    pgs_cfm: float,
    dof_mask: wp.array[wp.uint64],
    inverse_permutation: wp.array[int],
    factor_lookup: wp.array2d[int],
    inverse_factor: wp.array2d[float],
    v_hat: wp.array[float],
    slot_counter: wp.array[int],
    row_type: wp.array2d[int],
    row_parent: wp.array2d[int],
    row_mu: wp.array2d[float],
    row_beta: wp.array2d[float],
    row_cfm: wp.array2d[float],
    phi: wp.array2d[float],
    target_velocity: wp.array2d[float],
    row_dof: wp.array3d[int],
    row_factor: wp.array3d[float],
    row_incident: wp.array2d[float],
    diagonal: wp.array2d[float],
): ...


@wp.kernel(enable_backward=False)
def build_sparse_joint_limit_rows(
    group_to_art: wp.array[int],
    art_to_world: wp.array[int],
    articulation_world_dof_offset: wp.array[int],
    articulation_dof_start: wp.array[int],
    limit_q_index: wp.array[int],
    joint_limit_lower: wp.array[float],
    joint_limit_upper: wp.array[float],
    joint_q: wp.array[float],
    activation_gap: float,
    pgs_beta: float,
    pgs_cfm: float,
    dof_mask: wp.array[wp.uint64],
    inverse_permutation: wp.array[int],
    factor_lookup: wp.array2d[int],
    inverse_factor: wp.array2d[float],
    v_hat: wp.array[float],
    slot_counter: wp.array[int],
    row_type: wp.array2d[int],
    row_parent: wp.array2d[int],
    row_mu: wp.array2d[float],
    row_beta: wp.array2d[float],
    row_cfm: wp.array2d[float],
    phi: wp.array2d[float],
    target_velocity: wp.array2d[float],
    row_dof: wp.array3d[int],
    row_factor: wp.array3d[float],
    row_incident: wp.array2d[float],
    diagonal: wp.array2d[float],
):
    """Build ordered limit rows and sparse responses with one warp per articulation.

    Launch ``32 * group_count`` threads in blocks divisible by 32. Active row
    reservations include over-capacity demand; writes stay within row capacity.
    Reservations retain each articulation's lower/upper DOF order.
    """
    _build_sparse_joint_limit_rows(
        wp.tid(),
        group_to_art,
        art_to_world,
        articulation_world_dof_offset,
        articulation_dof_start,
        limit_q_index,
        joint_limit_lower,
        joint_limit_upper,
        joint_q,
        activation_gap,
        pgs_beta,
        pgs_cfm,
        dof_mask,
        inverse_permutation,
        factor_lookup,
        inverse_factor,
        v_hat,
        slot_counter,
        row_type,
        row_parent,
        row_mu,
        row_beta,
        row_cfm,
        phi,
        target_velocity,
        row_dof,
        row_factor,
        row_incident,
        diagonal,
    )


@wp.kernel(enable_backward=False)
def apply_sparse_contact_restitution(
    constraint_count: wp.array[int],
    phi: wp.array2d[float],
    row_type: wp.array2d[int],
    target_velocity: wp.array2d[float],
    row_restitution: wp.array2d[float],
    row_incident: wp.array2d[float],
    dt: float,
    restitution_velocity_threshold: float,
    rhs: wp.array2d[float],
):
    """Use the already projected incident velocity for the unchanged impact law."""
    world, row = wp.tid()
    if row >= constraint_count[world] or row_type[world, row] != PGS_CONSTRAINT_TYPE_CONTACT:
        return
    restitution = row_restitution[world, row]
    relative_incident = row_incident[world, row] - target_velocity[world, row]
    if restitution > 0.0 and contact_restitution_fires(
        phi[world, row], relative_incident, dt, restitution_velocity_threshold
    ):
        rhs[world, row] = -target_velocity[world, row] + restitution * relative_incident


@wp.kernel(enable_backward=False)
def apply_sparse_factor_velocity(
    group_to_art: wp.array[int],
    art_to_world: wp.array[int],
    articulation_world_dof_offset: wp.array[int],
    articulation_dof_start: wp.array[int],
    permutation: wp.array[int],
    factor_lookup: wp.array2d[int],
    inverse_factor: wp.array2d[float],
    factor_velocity_delta: wp.array2d[float],
    v_hat: wp.array[float],
    v_out: wp.array[float],
):
    """Decode the factor-coordinate impulse once per world after its sweeps."""
    group, column = wp.tid()
    art = group_to_art[group]
    world = art_to_world[art]
    delta = float(0.0)
    for row in range(column, permutation.shape[0]):
        entry = factor_lookup[row, column]
        if entry >= 0:
            delta += (
                inverse_factor[group, entry] * factor_velocity_delta[world, articulation_world_dof_offset[art] + row]
            )
    dof = articulation_dof_start[art] + permutation[column]
    v_out[dof] = v_hat[dof] + delta


@wp.kernel(enable_backward=False)
def apply_sparse_free_velocity(
    group_to_art: wp.array[int],
    art_to_world: wp.array[int],
    articulation_world_dof_offset: wp.array[int],
    articulation_dof_start: wp.array[int],
    factor_velocity_delta: wp.array2d[float],
    v_hat: wp.array[float],
    v_out: wp.array[float],
):
    """Publish physical free-body deltas without a coordinate conversion."""
    group, column = wp.tid()
    art = group_to_art[group]
    world = art_to_world[art]
    offset = articulation_world_dof_offset[art]
    dof = articulation_dof_start[art] + column
    v_out[dof] = v_hat[dof] + factor_velocity_delta[world, offset + column]
