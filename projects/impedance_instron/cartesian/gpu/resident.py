# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Device-resident shared-controller Cartesian equilibrium optimizer.

All coefficient generation, bounds evaluation, residual Jacobians, normal
equations, damped Gauss-Newton linear solves, line searches, stochastic
exploration proposals, and candidate selections execute entirely on CUDA via
Warp float64 kernels.

No fitting-array transfers or CPU solves occur inside the search loop. When
loss-plateau early stopping is enabled, a single 4-byte stop status integer is
copied from device to host after each completed poll+trial iteration to detect
early termination; fitting arrays, residuals, and losses remain entirely on device.
Device completion waits between full batches measure actual wall time, not enqueue
time. The saved winner and compact history are read only after search ends.
"""

from __future__ import annotations

import sys
from copy import deepcopy
from time import perf_counter
from typing import TYPE_CHECKING

import numpy as np
import warp as wp

from ..spline import _uniform_knots
from ..trajectory import Spline

if TYPE_CHECKING:
    from .engine import Engine

wp.set_module_options({"enable_backward": False, "fuse_fp": False})

_IMPROVEMENT_TOLERANCE = 1.0e-10
_DAMPING = (0.01, 0.1, 1.0, 10.0)
_LINE_FACTORS = (1.0, 0.5, 0.25)
_DIAGONAL_FLOOR = 1.0e-12
_TRUST_MULTIPLIER = 2.0

_CONTROL_COUNT = 12
_WORLD_COUNT_BY_CHANNELS = {4: 128, 6: 192}


def canonical_knot_scales() -> tuple[np.ndarray, np.ndarray]:
    """Compute canonical float64 knot derivative scales for the 12-point controller."""
    knots = _uniform_knots(_CONTROL_COUNT, 3)
    first_scale = 3.0 / (knots[4 : _CONTROL_COUNT + 3] - knots[1:_CONTROL_COUNT])
    derivative_knots = knots[1:-1]
    second_scale = 2.0 / (derivative_knots[3 : _CONTROL_COUNT + 1] - derivative_knots[1 : _CONTROL_COUNT - 1])
    return np.asarray(first_scale, dtype=np.float64), np.asarray(second_scale, dtype=np.float64)


@wp.func
def _check_bounds(
    coeffs: wp.array3d[wp.float64],
    world: int,
    duration_s: wp.float64,
    lower: wp.array[wp.float64],
    upper: wp.array[wp.float64],
    rate_limit: wp.array[wp.float64],
    acc_limit: wp.array[wp.float64],
    first_scale: wp.array[wp.float64],
    second_scale: wp.array[wp.float64],
    control_count: int,
) -> int:
    """Evaluate position, rate, and acceleration control-polygon bounds."""
    # 1. Position bounds
    channels = coeffs.shape[2]
    for i in range(control_count):
        for ch in range(channels):
            value = coeffs[world, i, ch]
            if not wp.isfinite(value) or value < lower[ch] or value > upper[ch]:
                return 0

    # 2. Derivative control-polygon bounds
    for ch in range(channels):
        for row in range(control_count - 1):
            first = first_scale[row] * (coeffs[world, row + 1, ch] - coeffs[world, row, ch])
            if wp.abs(first) > rate_limit[ch] * duration_s:
                return 0
            if row < control_count - 2:
                next_first = first_scale[row + 1] * (coeffs[world, row + 2, ch] - coeffs[world, row + 1, ch])
                second = second_scale[row] * (next_first - first)
                if wp.abs(second) > acc_limit[ch] * (duration_s * duration_s):
                    return 0

    return 1


@wp.kernel
def _generate_poll_coefficients_kernel(
    island_incumbent: wp.array3d[wp.float64],
    island_fraction: wp.array[wp.float64],
    scale: wp.array[wp.float64],
    rng_states: wp.array[wp.uint32],
    duration_s: wp.float64,
    lower: wp.array[wp.float64],
    upper: wp.array[wp.float64],
    rate_limit: wp.array[wp.float64],
    acc_limit: wp.array[wp.float64],
    first_scale: wp.array[wp.float64],
    second_scale: wp.array[wp.float64],
    coefficients: wp.array3d[wp.float64],
    real_mask: wp.array[int],
    proposed_mask: wp.array[int],
    padding_mask: wp.array[int],
    active: wp.array[int],
    islands: int,
    parameter_count: int,
    control_count: int,
    worlds_per_island: int,
):
    """Populate poll: 1 baseline + 2*P coordinate candidates and remaining exploration slots."""
    tid = wp.tid()
    if active[0] == 0:
        real_mask[tid] = 0
        proposed_mask[tid] = 0
        padding_mask[tid] = 1
        return
    island = tid // worlds_per_island
    slot = tid % worlds_per_island
    frac = island_fraction[island]
    channels = coefficients.shape[2]
    stochastic = int(0)
    if slot == 0:
        for i in range(control_count):
            for ch in range(channels):
                coefficients[tid, i, ch] = island_incumbent[island, i, ch]
    elif slot <= 2 * parameter_count:
        coord_idx = (slot - 1) // 2
        sign = wp.float64(1.0)
        if (slot - 1) % 2 == 1:
            sign = wp.float64(-1.0)
        ctrl = coord_idx // channels
        ch_idx = coord_idx % channels
        for i in range(control_count):
            for ch in range(channels):
                val = island_incumbent[island, i, ch]
                if i == ctrl and ch == ch_idx:
                    val = val + sign * frac * scale[coord_idx]
                coefficients[tid, i, ch] = val
    else:
        stochastic = 1
        state = rng_states[tid]
        trust_bound = wp.float64(_TRUST_MULTIPLIER) * frac
        for c in range(parameter_count):
            ctrl = c // channels
            ch = c % channels
            state = state * wp.uint32(1664525) + wp.uint32(1013904223)
            u = wp.float64(state) / wp.float64(4294967295.0)
            coefficients[tid, ctrl, ch] = (
                island_incumbent[island, ctrl, ch] + (u * wp.float64(2.0) - wp.float64(1.0)) * trust_bound * scale[c]
            )
        rng_states[tid] = state
    proposed_mask[tid] = 1
    valid = _check_bounds(
        coefficients, tid, duration_s, lower, upper, rate_limit, acc_limit, first_scale, second_scale, control_count
    )
    if valid == 0 and stochastic == 1:
        for _ in range(8):
            if valid == 1:
                break
            for i in range(control_count):
                for ch in range(channels):
                    incumbent = island_incumbent[island, i, ch]
                    coefficients[tid, i, ch] = incumbent + wp.float64(0.5) * (coefficients[tid, i, ch] - incumbent)
            valid = _check_bounds(
                coefficients,
                tid,
                duration_s,
                lower,
                upper,
                rate_limit,
                acc_limit,
                first_scale,
                second_scale,
                control_count,
            )
    if valid == 1:
        real_mask[tid] = 1
        padding_mask[tid] = 0
    else:
        for i in range(control_count):
            for ch in range(channels):
                coefficients[tid, i, ch] = island_incumbent[island, i, ch]
        real_mask[tid] = 0
        padding_mask[tid] = 1


@wp.kernel
def _evaluate_usable_kernel(
    real_mask: wp.array[int],
    failure_code: wp.array[int],
    integrated_steps: wp.array[int],
    loss: wp.array[wp.float64],
    rmse: wp.array2d[wp.float64],
    costs: wp.array2d[wp.float64],
    residual: wp.array2d[wp.float64],
    expected_steps: int,
    residual_dim: int,
    usable_mask: wp.array[int],
    completed_mask: wp.array[int],
    active: wp.array[int],
):
    """Mark candidates as completed and usable matching _complete_scores."""
    world = wp.tid()
    complete = 1
    if failure_code[world] != 0:
        complete = 0
    if integrated_steps[world] != expected_steps:
        complete = 0
    l_val = loss[world]
    if not wp.isfinite(l_val) or l_val < wp.float64(0.0):
        complete = 0
    for ch in range(6):
        if not wp.isfinite(rmse[world, ch]):
            complete = 0
    for c in range(3):
        if not wp.isfinite(costs[world, c]):
            complete = 0
    # Keep this check independent of objective-side score validation.
    for k in range(residual_dim):
        if not wp.isfinite(residual[k, world]):
            complete = 0

    completed_mask[world] = complete
    if complete == 1 and real_mask[world] == 1 and active[0] != 0:
        usable_mask[world] = 1
    else:
        usable_mask[world] = 0


@wp.kernel
def _select_initial_diagnostic_kernel(
    usable_mask: wp.array[int],
    winner: wp.array[int],
    initial_seen: wp.array[int],
):
    """Preserve world zero as a diagnostic when the mandatory baseline fails."""
    if wp.tid() == 0:
        winner[0] = 0 if initial_seen[0] == 0 and usable_mask[0] == 0 else -1


@wp.kernel
def _mark_initial_device_state_kernel(
    usable_mask: wp.array[int],
    loss: wp.array[wp.float64],
    active: wp.array[int],
    initial_loss: wp.array[wp.float64],
    initial_seen: wp.array[int],
):
    """Commit the mandatory baseline state entirely on device."""
    if wp.tid() != 0 or initial_seen[0] != 0:
        return
    initial_seen[0] = 1
    if usable_mask[0] == 0:
        active[0] = 0
    else:
        initial_loss[0] = loss[0]


@wp.kernel
def _select_global_candidate_kernel(
    usable_mask: wp.array[int],
    loss: wp.array[wp.float64],
    coefficients: wp.array3d[wp.float64],
    active: wp.array[int],
    best_loss: wp.array[wp.float64],
    best_coeffs: wp.array2d[wp.float64],
    winner: wp.array[int],
    world_count: int,
    parameter_count: int,
    control_count: int,
):
    """Retain a strictly better usable candidate and its current world index."""
    if wp.tid() != 0:
        return
    winner[0] = -1
    if active[0] == 0:
        return
    candidate = int(-1)
    candidate_loss = best_loss[0]
    for world in range(world_count):
        if usable_mask[world] == 1 and wp.isfinite(loss[world]):
            if loss[world] < candidate_loss - wp.float64(_IMPROVEMENT_TOLERANCE):
                candidate = world
                candidate_loss = loss[world]
    if candidate >= 0:
        best_loss[0] = candidate_loss
        winner[0] = candidate
        for i in range(control_count):
            for ch in range(coefficients.shape[2]):
                best_coeffs[i, ch] = coefficients[candidate, i, ch]


@wp.kernel
def _accumulate_batch_counts_kernel(
    proposed_mask: wp.array[int],
    real_mask: wp.array[int],
    padding_mask: wp.array[int],
    completed_mask: wp.array[int],
    integrated_steps: wp.array[int],
    counters: wp.array[wp.int64],
):
    """Add batch contributions into cumulative search counters entirely on device."""
    world = wp.tid()
    wp.atomic_add(counters, 0, wp.int64(1))  # physical world launched
    wp.atomic_add(counters, 1, wp.int64(proposed_mask[world]))
    wp.atomic_add(counters, 2, wp.int64(real_mask[world]))
    wp.atomic_add(counters, 3, wp.int64(padding_mask[world]))
    wp.atomic_add(counters, 4, wp.int64(completed_mask[world]))
    wp.atomic_add(counters, 5, wp.int64(completed_mask[world] * real_mask[world]))
    wp.atomic_add(counters, 6, wp.int64(integrated_steps[world]))


@wp.kernel
def _assemble_jacobian_kernel(
    residual: wp.array2d[wp.float64],
    usable: wp.array[int],
    jacobian: wp.array2d[wp.float64],
    step_fraction: wp.array[wp.float64],
    residual_dim: int,
    parameter_count: int,
    worlds_per_island: int,
):
    """Assemble finite-difference Jacobian entries in parallel."""
    tid = wp.tid()
    column_count = jacobian.shape[1]
    entry = tid
    k = entry // column_count
    q = entry - k * column_count
    if k >= residual_dim:
        return
    island = q // parameter_count
    coord = q - island * parameter_count
    base = island * worlds_per_island
    plus = base + 1 + 2 * coord
    minus = base + 2 + 2 * coord
    frac = step_fraction[island]
    value = wp.float64(0.0)
    if usable[base] == 0:
        jacobian[k, q] = value
        return
    if usable[plus] == 1 and usable[minus] == 1:
        value = (residual[k, plus] - residual[k, minus]) / (wp.float64(2.0) * frac)
    elif usable[plus] == 1:
        value = (residual[k, plus] - residual[k, base]) / frac
    elif usable[minus] == 1:
        value = (residual[k, base] - residual[k, minus]) / frac
    jacobian[k, q] = value


@wp.kernel
def _assemble_normal_equations_kernel(
    residual: wp.array2d[wp.float64],
    jacobian: wp.array2d[wp.float64],
    usable: wp.array[int],
    H_base: wp.array3d[wp.float64],
    residual_dim: int,
    parameter_count: int,
    worlds_per_island: int,
):
    """Assemble each upper-triangular normal-equation entry independently."""
    tid = wp.tid()
    entry_count = parameter_count * (parameter_count + 1)
    island = tid // entry_count
    entry = tid - island * entry_count
    i = entry // (parameter_count + 1)
    j = entry - i * (parameter_count + 1)
    if i >= parameter_count:
        return
    base = island * worlds_per_island
    total = wp.float64(0.0)
    if usable[base] == 0:
        H_base[island, i, j] = total
        return
    for k in range(residual_dim):
        if j == parameter_count:
            total = total + jacobian[k, island * parameter_count + i] * residual[k, base]
        elif i <= j:
            total = total + jacobian[k, island * parameter_count + i] * jacobian[k, island * parameter_count + j]
        else:
            total = total + jacobian[k, island * parameter_count + j] * jacobian[k, island * parameter_count + i]
    H_base[island, i, j] = total


@wp.kernel
def _compute_derivatives_and_solve_kernel(
    residual: wp.array2d[wp.float64],  # (residual_dim, world_count)
    loss: wp.array[wp.float64],  # (world_count,)
    usable: wp.array[int],  # (world_count,)
    step_fraction: wp.array[wp.float64],  # (islands,)
    H_base: wp.array3d[wp.float64],  # (islands, P, P + 1)
    aug_mat: wp.array3d[wp.float64],  # (islands * 4, P, P + 1)
    gn_directions: wp.array3d[wp.float64],  # (islands, 4, P)
    gn_valid: wp.array2d[int],  # (islands, 4)
    net_pattern: wp.array2d[wp.float64],  # (islands, P)
    net_valid: wp.array[int],  # (islands,)
    residual_dim: int,
    parameter_count: int,
    preassembled: int,
    worlds_per_island: int,
):
    """Solve damped PxP systems; optionally consume parallel H entries."""
    island = wp.tid()
    base_world = island * worlds_per_island
    frac = step_fraction[island]

    # Check baseline usability
    if usable[base_world] == 0:
        net_valid[island] = 0
        for d in range(4):
            gn_valid[island, d] = 0
        return

    has_active_jacobian = int(0)
    has_pattern = int(0)
    base_loss = loss[base_world]
    tol = wp.float64(_IMPROVEMENT_TOLERANCE)

    for c in range(parameter_count):
        plus_w = base_world + 1 + 2 * c
        minus_w = base_world + 2 + 2 * c
        u_plus = usable[plus_w]
        u_minus = usable[minus_w]

        pat_val = wp.float64(0.0)
        if u_plus == 1 and u_minus == 1:
            # Preserve the established direction tie-break: retain the + direction.
            if loss[plus_w] <= loss[minus_w]:
                if loss[plus_w] < base_loss - tol:
                    pat_val = frac
                    has_pattern = 1
            elif loss[minus_w] < base_loss - tol:
                pat_val = -frac
                has_pattern = 1
        elif u_plus == 1:
            if loss[plus_w] < base_loss - tol:
                pat_val = frac
                has_pattern = 1
        elif u_minus == 1:
            if loss[minus_w] < base_loss - tol:
                pat_val = -frac
                has_pattern = 1
        net_pattern[island, c] = pat_val
    net_valid[island] = has_pattern

    # Build derivatives first. Usable columns can still be identically zero;
    # they must not make a singular all-zero normal equation look active.
    if preassembled == 0:
        for i in range(parameter_count):
            for j in range(parameter_count):
                H_base[island, i, j] = wp.float64(0.0)
            H_base[island, i, parameter_count] = wp.float64(0.0)

    inv_frac = wp.float64(1.0) / frac
    inv_2frac = wp.float64(0.5) * inv_frac

    if preassembled == 0:
        for k in range(residual_dim):
            r0 = residual[k, base_world]
            for i in range(parameter_count):
                p_w = base_world + 1 + 2 * i
                m_w = base_world + 2 + 2 * i
                up = usable[p_w]
                um = usable[m_w]
                Ji = wp.float64(0.0)
                if up == 1 and um == 1:
                    Ji = (residual[k, p_w] - residual[k, m_w]) * inv_2frac
                elif up == 1:
                    Ji = (residual[k, p_w] - r0) * inv_frac
                elif um == 1:
                    Ji = (r0 - residual[k, m_w]) * inv_frac
                if wp.abs(Ji) > wp.float64(1.0e-15):
                    has_active_jacobian = 1

                H_base[island, i, parameter_count] = H_base[island, i, parameter_count] + Ji * r0

                for j in range(i, parameter_count):
                    p_wj = base_world + 1 + 2 * j
                    m_wj = base_world + 2 + 2 * j
                    upj = usable[p_wj]
                    umj = usable[m_wj]
                    Jj = wp.float64(0.0)
                    if upj == 1 and umj == 1:
                        Jj = (residual[k, p_wj] - residual[k, m_wj]) * inv_2frac
                    elif upj == 1:
                        Jj = (residual[k, p_wj] - r0) * inv_frac
                    elif umj == 1:
                        Jj = (r0 - residual[k, m_wj]) * inv_frac

                    H_base[island, i, j] = H_base[island, i, j] + Ji * Jj

    else:
        # Parallel assembly has already populated H_base; inspect its diagonal
        # to reject a batch whose usable columns are all numerically zero.
        for i in range(parameter_count):
            if H_base[island, i, i] > wp.float64(1.0e-30):
                has_active_jacobian = 1
    if has_active_jacobian == 0:
        for d in range(4):
            gn_valid[island, d] = 0
        return

    max_diag = wp.float64(0.0)
    for i in range(parameter_count):
        d_val = H_base[island, i, i]
        if d_val > max_diag:
            max_diag = d_val
        for j in range(i + 1, parameter_count):
            H_base[island, j, i] = H_base[island, i, j]

    diag_floor = wp.float64(_DIAGONAL_FLOOR)
    if wp.float64(_DIAGONAL_FLOOR) * max_diag > diag_floor:
        diag_floor = wp.float64(_DIAGONAL_FLOOR) * max_diag

    # Solve for 4 dampings: 0.01, 0.1, 1.0, 10.0
    for d in range(4):
        slot_d = island * 4 + d
        damp = wp.float64(0.01)
        if d == 1:
            damp = wp.float64(0.1)
        elif d == 2:
            damp = wp.float64(1.0)
        elif d == 3:
            damp = wp.float64(10.0)

        for i in range(parameter_count):
            for j in range(parameter_count):
                aug_mat[slot_d, i, j] = H_base[island, i, j]
            reg = H_base[island, i, i]
            if reg < diag_floor:
                reg = diag_floor
            aug_mat[slot_d, i, i] = aug_mat[slot_d, i, i] + damp * reg
            aug_mat[slot_d, i, parameter_count] = -H_base[island, i, parameter_count]

        # Gauss-Jordan elimination
        valid = int(1)
        for i in range(parameter_count):
            pivot = int(i)
            max_v = wp.abs(aug_mat[slot_d, i, i])
            for r in range(i + 1, parameter_count):
                val = wp.abs(aug_mat[slot_d, r, i])
                if val > max_v:
                    max_v = val
                    pivot = r
            if max_v < wp.float64(1.0e-15):
                valid = 0
            if pivot != i:
                for c_idx in range(i, parameter_count + 1):
                    tmp = aug_mat[slot_d, i, c_idx]
                    aug_mat[slot_d, i, c_idx] = aug_mat[slot_d, pivot, c_idx]
                    aug_mat[slot_d, pivot, c_idx] = tmp
            piv_val = aug_mat[slot_d, i, i]
            for c_idx in range(i, parameter_count + 1):
                aug_mat[slot_d, i, c_idx] = aug_mat[slot_d, i, c_idx] / piv_val
            for r in range(parameter_count):
                if r != i:
                    factor = aug_mat[slot_d, r, i]
                    for c_idx in range(i, parameter_count + 1):
                        aug_mat[slot_d, r, c_idx] = aug_mat[slot_d, r, c_idx] - factor * aug_mat[slot_d, i, c_idx]

        max_mag = wp.float64(0.0)
        has_nonzero = int(0)
        for i in range(parameter_count):
            xi = aug_mat[slot_d, i, parameter_count]
            if not wp.isfinite(xi):
                valid = 0
            abs_xi = wp.abs(xi)
            if abs_xi > max_mag:
                max_mag = abs_xi
            if abs_xi > wp.float64(1.0e-15):
                has_nonzero = 1

        if has_nonzero == 0:
            valid = 0

        gn_valid[island, d] = valid
        if valid == 1:
            trust_limit = wp.float64(_TRUST_MULTIPLIER) * frac
            mult = wp.float64(1.0)
            if max_mag > trust_limit:
                mult = trust_limit / max_mag
            for i in range(parameter_count):
                gn_directions[island, d, i] = aug_mat[slot_d, i, parameter_count] * mult
        else:
            for i in range(parameter_count):
                gn_directions[island, d, i] = wp.float64(0.0)


@wp.kernel
def _generate_trial_coefficients_kernel(
    island_incumbent: wp.array3d[wp.float64],
    island_origin: wp.array3d[wp.float64],
    island_fraction: wp.array[wp.float64],
    scale: wp.array[wp.float64],
    gn_directions: wp.array3d[wp.float64],
    gn_valid: wp.array2d[int],
    net_pattern: wp.array2d[wp.float64],
    net_valid: wp.array[int],
    rng_states: wp.array[wp.uint32],
    duration_s: wp.float64,
    lower: wp.array[wp.float64],
    upper: wp.array[wp.float64],
    rate_limit: wp.array[wp.float64],
    acc_limit: wp.array[wp.float64],
    first_scale: wp.array[wp.float64],
    second_scale: wp.array[wp.float64],
    coefficients: wp.array3d[wp.float64],
    real_mask: wp.array[int],
    proposed_mask: wp.array[int],
    padding_mask: wp.array[int],
    active: wp.array[int],
    islands: int,
    parameter_count: int,
    control_count: int,
    worlds_per_island: int,
):
    """Generate trial candidates and stochastic exploration proposals."""
    tid = wp.tid()
    if active[0] == 0:
        real_mask[tid] = 0
        proposed_mask[tid] = 0
        padding_mask[tid] = 1
        return
    island = tid // worlds_per_island
    slot = tid % worlds_per_island
    frac = island_fraction[island]
    channels = coefficients.shape[2]

    # Slot 0: incumbent
    if slot == 0:
        for i in range(control_count):
            for ch in range(channels):
                coefficients[tid, i, ch] = island_incumbent[island, i, ch]
        real_mask[tid] = 1
        proposed_mask[tid] = 1
        padding_mask[tid] = 0
        return

    # Slots 1..12: 4 dampings x 3 line factors (1.0, 0.5, 0.25)
    # Slots 13..15: net coordinate pattern x 3 line factors
    is_stochastic = int(0)

    if slot >= 1 and slot <= 12:
        gn_idx = (slot - 1) // 3
        factor_idx = (slot - 1) % 3
        if gn_valid[island, gn_idx] == 1:
            factor = wp.float64(1.0)
            if factor_idx == 1:
                factor = wp.float64(0.5)
            elif factor_idx == 2:
                factor = wp.float64(0.25)

            for c in range(parameter_count):
                ctrl = c // channels
                ch = c % channels
                dir_val = gn_directions[island, gn_idx, c]
                val = island_origin[island, ctrl, ch] + factor * dir_val * scale[c]
                coefficients[tid, ctrl, ch] = val
        else:
            is_stochastic = 1
    elif slot >= 13 and slot <= 15:
        if net_valid[island] == 1:
            factor = wp.float64(1.0)
            if slot == 14:
                factor = wp.float64(0.5)
            elif slot == 15:
                factor = wp.float64(0.25)

            for c in range(parameter_count):
                ctrl = c // channels
                ch = c % channels
                pat_val = net_pattern[island, c]
                val = island_origin[island, ctrl, ch] + factor * pat_val * scale[c]
                coefficients[tid, ctrl, ch] = val
        else:
            is_stochastic = 1
    else:
        is_stochastic = 1

    if is_stochastic == 1:
        state = rng_states[tid]
        trust_bound = wp.float64(_TRUST_MULTIPLIER) * frac
        for c in range(parameter_count):
            ctrl = c // channels
            ch = c % channels
            state = state * wp.uint32(1664525) + wp.uint32(1013904223)
            u = wp.float64(state) / wp.float64(4294967295.0)
            rand_step = (u * wp.float64(2.0) - wp.float64(1.0)) * trust_bound * scale[c]
            coefficients[tid, ctrl, ch] = island_incumbent[island, ctrl, ch] + rand_step
        rng_states[tid] = state

    proposed_mask[tid] = 1
    valid = _check_bounds(
        coefficients, tid, duration_s, lower, upper, rate_limit, acc_limit, first_scale, second_scale, control_count
    )
    if valid == 0 and is_stochastic == 1:
        for _ in range(8):
            if valid == 1:
                break
            for i in range(control_count):
                for ch in range(channels):
                    incumbent = island_incumbent[island, i, ch]
                    coefficients[tid, i, ch] = incumbent + wp.float64(0.5) * (coefficients[tid, i, ch] - incumbent)
            valid = _check_bounds(
                coefficients,
                tid,
                duration_s,
                lower,
                upper,
                rate_limit,
                acc_limit,
                first_scale,
                second_scale,
                control_count,
            )
    if valid == 1:
        real_mask[tid] = 1
        padding_mask[tid] = 0
    else:
        for i in range(control_count):
            for ch in range(channels):
                coefficients[tid, i, ch] = island_incumbent[island, i, ch]
        real_mask[tid] = 0
        padding_mask[tid] = 1


@wp.kernel
def _update_island_incumbents_kernel(
    usable_mask: wp.array[int],
    loss: wp.array[wp.float64],
    coefficients: wp.array3d[wp.float64],
    minimum_step_fraction: wp.float64,
    island_incumbent: wp.array3d[wp.float64],
    island_loss: wp.array[wp.float64],
    island_fraction: wp.array[wp.float64],
    island_origin: wp.array3d[wp.float64],
    best_world_idx: wp.array[int],
    best_loss_val: wp.array[wp.float64],
    adapt_fraction: int,
    islands: int,
    parameter_count: int,
    control_count: int,
    worlds_per_island: int,
):
    """Select a batch winner, optionally adapt state after poll and trial."""
    island = wp.tid()
    base_world = island * worlds_per_island

    curr_loss = island_loss[island]
    best_w = int(-1)
    best_l = wp.float64(curr_loss)

    for s in range(worlds_per_island):
        w = base_world + s
        if usable_mask[w] == 1:
            l = loss[w]
            if l < best_l - wp.float64(_IMPROVEMENT_TOLERANCE):
                best_l = l
                best_w = w

    improved = int(0)
    if best_w >= 0 and best_l < curr_loss - wp.float64(_IMPROVEMENT_TOLERANCE):
        improved = 1
        island_loss[island] = best_l
        for i in range(control_count):
            for ch in range(coefficients.shape[2]):
                island_incumbent[island, i, ch] = coefficients[best_w, i, ch]

    # Poll selection must not alter the origin or fraction used by the
    # derivative and trial phase. Adapt both after the trial batch only.
    if adapt_fraction == 1:
        if improved == 0:
            f = island_fraction[island] * wp.float64(0.5)
            if f < minimum_step_fraction:
                f = minimum_step_fraction
            island_fraction[island] = f

        for i in range(control_count):
            for ch in range(coefficients.shape[2]):
                island_origin[island, i, ch] = island_incumbent[island, i, ch]


@wp.kernel
def _begin_iteration(
    incumbent: wp.array3d[wp.float64],
    origin: wp.array3d[wp.float64],
    loss: wp.array[wp.float64],
    before: wp.array[wp.float64],
    control_count: int,
):
    island = wp.tid()
    before[island] = loss[island]
    for row in range(control_count):
        for channel in range(incumbent.shape[2]):
            origin[island, row, channel] = incumbent[island, row, channel]


@wp.kernel
def _latch_poll_baseline(
    loss: wp.array[wp.float64],
    usable: wp.array[int],
    before: wp.array[wp.float64],
    worlds_per_island: int,
):
    island = wp.tid()
    base = island * worlds_per_island
    if not wp.isfinite(before[island]) and usable[base] != 0:
        before[island] = loss[base]


@wp.kernel
def _gate_usable(active: wp.array[int], usable: wp.array[int]):
    if active[0] == 0:
        usable[wp.tid()] = 0


@wp.kernel
def _finish_iteration(
    active: wp.array[int],
    loss: wp.array[wp.float64],
    before: wp.array[wp.float64],
    fraction: wp.array[wp.float64],
    minimum: wp.float64,
):
    island = wp.tid()
    if active[0] != 0 and not (loss[island] < before[island] - wp.float64(_IMPROVEMENT_TOLERANCE)):
        fraction[island] = wp.max(minimum, fraction[island] * wp.float64(0.5))


@wp.kernel
def _record_history(
    batch: int,
    best_loss: wp.array[wp.float64],
    counters: wp.array[wp.int64],
    loss_history: wp.array[wp.float64],
    count_history: wp.array2d[wp.int64],
):
    loss_history[batch] = best_loss[0]
    for i in range(7):
        count_history[batch, i] = counters[i]


@wp.kernel
def _update_plateau_kernel(
    active: wp.array[int],
    best_loss_val: wp.array[wp.float64],
    anchor_loss: wp.array[wp.float64],
    stale_count: wp.array[int],
    stop_flag: wp.array[int],
    patience: int,
    rtol: wp.float64,
):
    """Track best loss anchor and count iterations without relative improvement."""
    if wp.tid() != 0:
        return
    if active[0] == 0:
        return
    current_best = best_loss_val[0]
    if not wp.isfinite(current_best):
        return
    anchor = anchor_loss[0]
    if not wp.isfinite(anchor):
        anchor_loss[0] = current_best
        stale_count[0] = 0
        return
    scale = wp.max(wp.abs(anchor), wp.float64(1.0e-12))
    threshold = rtol * scale
    if current_best <= anchor - threshold:
        anchor_loss[0] = current_best
        stale_count[0] = 0
    else:
        stale_count[0] += 1
        if stale_count[0] >= patience:
            stop_flag[0] = 1


def fit_resident(
    engine: Engine,
    initial: Spline,
    *,
    max_iterations: int = 200,
    max_wall_s: float = 3600.0,
    initial_step_fraction: float = 0.05,
    minimum_step_fraction: float = 0.005,
    seed: int = 17,
    plateau_patience: int | None = 20,
    plateau_rtol: float = 1e-4,
) -> tuple[Spline, dict, dict, dict]:
    """Use every world to optimize one shared equilibrium controller on device.

    One incumbent, one coordinate poll, and one step fraction are shared across
    all 128 worlds. Extra worlds explore bounded local updates to that incumbent.

    When ``plateau_patience`` is None, early stopping is disabled
    and search runs until iteration or wall budget exhaustion without any
    intermediate device-to-host transfers. When enabled, a device kernel tracks
    best loss improvement relative to an anchor loss initialized from the baseline
    rollout. If ``plateau_patience`` consecutive completed poll+trial iterations
    fail to produce cumulative improvement >= ``plateau_rtol * max(abs(anchor), 1e-12)``,
    search terminates with reason ``"loss_plateau"``. To detect this condition,
    a single 4-byte stop-flag integer is copied from device to host after each
    completed iteration; all fitting arrays, candidate coefficients, and losses
    remain strictly on device.

    Args:
        engine: GPU engine configured with exactly 128 worlds.
        initial: Bounded starting spline matching engine control count and duration.
        max_iterations: Maximum number of search iterations.
        max_wall_s: Maximum wall time budget in seconds (checked between iterations).
        initial_step_fraction: Starting dimensionless step fraction.
        minimum_step_fraction: Smallest poll fraction allowed (does not stop search).
        seed: Deterministic integer seed for local proposals.
        plateau_patience: Number of consecutive completed iterations without relative
            improvement before early stopping. None disables plateau stopping.
        plateau_rtol: Relative improvement tolerance against anchor loss.

    Returns:
        Best spline, saved best trace, saved best run, and fitting summary.
    """
    started = perf_counter()
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int) or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")
    if plateau_patience is not None:
        if isinstance(plateau_patience, bool) or not isinstance(plateau_patience, int) or plateau_patience < 1:
            raise ValueError("plateau_patience must be a positive integer or None")
    if (
        isinstance(plateau_rtol, bool)
        or not isinstance(plateau_rtol, (int, float))
        or not np.isfinite(plateau_rtol)
        or plateau_rtol <= 0
    ):
        raise ValueError("plateau_rtol must be finite and positive")
    islands = 1
    for name, value in (
        ("max_wall_s", max_wall_s),
        ("initial_step_fraction", initial_step_fraction),
        ("minimum_step_fraction", minimum_step_fraction),
    ):
        if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if minimum_step_fraction > initial_step_fraction:
        raise ValueError("minimum_step_fraction must not exceed initial_step_fraction")

    control_count = getattr(engine.settings, "control_count", None)
    channels = initial.coefficients.shape[1] if initial.coefficients.ndim == 2 else 0
    if control_count != _CONTROL_COUNT or channels not in (4, 6):
        raise ValueError("Engine and initial spline must use 12 control points and four or six channels")
    if not np.isclose(initial.duration_s, engine.duration, rtol=0, atol=1e-12):
        raise ValueError("Initial spline duration must match the engine duration")

    expected_worlds = _WORLD_COUNT_BY_CHANNELS[channels]
    if engine.world_count != expected_worlds:
        raise ValueError(f"A {channels}-channel engine must use exactly {expected_worlds} worlds")
    w_per_island = expected_worlds
    parameter_count = initial.coefficients.size

    channel_scale = np.asarray(engine.settings.parameter_scale, dtype=np.float64)
    if channel_scale.shape != (channels,) or not np.isfinite(channel_scale).all() or np.any(channel_scale <= 0):
        raise ValueError(f"parameter_scale must contain {channels} finite positive channel scales")
    scale = np.tile(channel_scale, control_count)

    limits = tuple(
        engine.profile[key]
        for key in (
            "equilibrium_lower",
            "equilibrium_upper",
            "equilibrium_rate_limit",
            "equilibrium_acceleration_limit",
        )
    )
    if not initial.bounds(*limits):
        raise ValueError("Initial spline violates position, rate, or acceleration bounds")

    first_scale_np, second_scale_np = canonical_knot_scales()

    device = engine.device
    lower_vec = wp.array(engine.profile["equilibrium_lower"], dtype=wp.float64, device=device)
    upper_vec = wp.array(engine.profile["equilibrium_upper"], dtype=wp.float64, device=device)
    rate_vec = wp.array(engine.profile["equilibrium_rate_limit"], dtype=wp.float64, device=device)
    acc_vec = wp.array(engine.profile["equilibrium_acceleration_limit"], dtype=wp.float64, device=device)
    duration_s = float(engine.duration)

    # Initialize the shared controller before transferring it to the device.
    rng = np.random.default_rng(seed)
    island_starts = np.asarray(initial.coefficients[None], dtype=np.float64).copy()

    # 2. Allocate device arrays
    island_incumbent = wp.array(island_starts, dtype=wp.float64, device=device)
    island_origin = wp.array(island_starts, dtype=wp.float64, device=device)
    island_fraction = wp.full(islands, initial_step_fraction, dtype=wp.float64, device=device)
    island_loss = wp.full(islands, np.inf, dtype=wp.float64, device=device)
    iteration_loss_before = wp.empty(islands, dtype=wp.float64, device=device)
    loss_history = wp.empty(2 * max_iterations, dtype=wp.float64, device=device)
    count_history = wp.empty((2 * max_iterations, 7), dtype=wp.int64, device=device)

    scale_wp = wp.array(scale, dtype=wp.float64, device=device)
    first_scale_wp = wp.array(first_scale_np, dtype=wp.float64, device=device)
    second_scale_wp = wp.array(second_scale_np, dtype=wp.float64, device=device)

    # Masks and counters
    real_mask = wp.zeros(expected_worlds, dtype=wp.int32, device=device)
    proposed_mask = wp.zeros(expected_worlds, dtype=wp.int32, device=device)
    padding_mask = wp.zeros(expected_worlds, dtype=wp.int32, device=device)
    usable_mask = wp.zeros(expected_worlds, dtype=wp.int32, device=device)
    completed_mask = wp.zeros(expected_worlds, dtype=wp.int32, device=device)

    # RNG states per world
    init_rng_seeds = rng.integers(1, 2**31 - 1, size=expected_worlds, dtype=np.uint32)
    rng_states = wp.array(init_rng_seeds, dtype=wp.uint32, device=device)

    # Gauss-Newton solver work buffers
    residual_dim = engine.objective.residual_dim
    H_base = wp.zeros((islands, parameter_count, parameter_count + 1), dtype=wp.float64, device=device)
    jacobian = wp.empty((residual_dim, islands * parameter_count), dtype=wp.float64, device=device)
    aug_mat = wp.zeros((islands * 4, parameter_count, parameter_count + 1), dtype=wp.float64, device=device)
    gn_directions = wp.zeros((islands, 4, parameter_count), dtype=wp.float64, device=device)
    gn_valid = wp.zeros((islands, 4), dtype=wp.int32, device=device)
    net_pattern = wp.zeros((islands, parameter_count), dtype=wp.float64, device=device)
    net_valid = wp.zeros(islands, dtype=wp.int32, device=device)
    batch_counters = wp.zeros(7, dtype=wp.int64, device=device)
    search_active = wp.full(1, 1, dtype=wp.int32, device=device)
    initial_seen = wp.zeros(1, dtype=wp.int32, device=device)
    initial_loss_device = wp.full(1, np.inf, dtype=wp.float64, device=device)

    best_world_idx = wp.full(1, -1, dtype=wp.int32, device=device)
    best_loss_val = wp.full(1, np.inf, dtype=wp.float64, device=device)
    best_coeffs = wp.array(initial.coefficients, dtype=wp.float64, device=device)

    # Loss-plateau early stopping buffers
    plateau_enabled = plateau_patience is not None
    if plateau_enabled:
        plateau_anchor = wp.full(1, np.inf, dtype=wp.float64, device=device)
        plateau_stale = wp.zeros(1, dtype=wp.int32, device=device)
        plateau_stop = wp.zeros(1, dtype=wp.int32, device=device)
        plateau_stop_h = wp.zeros(1, dtype=wp.int32, device="cpu")

    counts = {
        "batches": 0,
        "physics_worlds": 0,
        "real_candidates": 0,
        "unique_candidates": 0,
        "proposed_candidates": 0,
        "bound_rejections": 0,
        "padding_worlds": 0,
        "completed_physics_worlds": 0,
        "completed_real_candidates": 0,
        "integrated_steps_including_padding": 0,
    }

    if not hasattr(engine, "create_snapshot") or not hasattr(engine, "snapshot_device"):
        raise RuntimeError("resident fit requires Engine resident snapshot support")
    best_snapshot = engine.create_snapshot()

    warmup_started = perf_counter()
    wp.load_module(module=sys.modules[__name__], device=device)

    # Pre-capture engine if not captured
    initial_batch = np.repeat(island_starts[0][None], expected_worlds, axis=0)
    if getattr(engine, "graph", None) is None:
        engine.capture(initial_batch)
    elif hasattr(engine, "capture_resident"):
        engine.capture_resident()
    else:
        raise RuntimeError("Engine does not provide resident graph capture")
    wp.synchronize_device(device)
    warmup_wall_s = perf_counter() - warmup_started

    def _launch_forward():
        if hasattr(engine, "foundation") and hasattr(engine.foundation, "enabled"):
            wp.copy(engine.foundation.enabled, real_mask)
        if hasattr(engine, "evaluate_device"):
            engine.evaluate_device()
        else:
            engine._reset()
            for _ in range(max(1, engine.steps // engine.chunk_steps)):
                wp.capture_launch(engine.graph)
            if engine.tail_graph is not None:
                wp.capture_launch(engine.tail_graph)
            engine._prepare()
            engine.objective.launch(engine.states, engine.forces, engine.integrated, engine.failure)

    search_started = perf_counter()
    iterations_run = 0
    iterations_started = 0
    batch_walls = []
    termination = "iteration_budget_exhausted"

    for _iteration in range(max_iterations):
        wp.synchronize_device(device)
        elapsed = perf_counter() - search_started
        next_batch_allowance = 1.1 * max(batch_walls) + 0.05 if batch_walls else 0.0
        if _iteration > 0 and elapsed + next_batch_allowance >= max_wall_s:
            termination = "wall_budget_exhausted"
            break

        iterations_started += 1
        batch_started = perf_counter()
        wp.launch(
            _begin_iteration,
            dim=islands,
            inputs=[island_incumbent, island_origin, island_loss, iteration_loss_before, control_count],
            device=device,
        )
        # ------------------ Phase 1: Poll Batch ------------------
        wp.launch(
            _generate_poll_coefficients_kernel,
            dim=expected_worlds,
            inputs=[
                island_incumbent,
                island_fraction,
                scale_wp,
                rng_states,
                duration_s,
                lower_vec,
                upper_vec,
                rate_vec,
                acc_vec,
                first_scale_wp,
                second_scale_wp,
                engine.coefficients,
                real_mask,
                proposed_mask,
                padding_mask,
                search_active,
                islands,
                parameter_count,
                control_count,
                w_per_island,
            ],
            device=device,
        )

        _launch_forward()

        wp.launch(
            _evaluate_usable_kernel,
            dim=expected_worlds,
            inputs=[
                real_mask,
                engine.failure,
                engine.integrated,
                engine.objective.loss,
                engine.objective.rmse,
                engine.objective.costs,
                engine.objective.residual,
                engine.steps,
                residual_dim,
                usable_mask,
                completed_mask,
                search_active,
            ],
            device=device,
        )

        wp.launch(
            _accumulate_batch_counts_kernel,
            dim=expected_worlds,
            inputs=[
                proposed_mask,
                real_mask,
                padding_mask,
                completed_mask,
                engine.integrated,
                batch_counters,
            ],
            device=device,
        )
        wp.launch(
            _select_initial_diagnostic_kernel,
            dim=1,
            inputs=[usable_mask, best_world_idx, initial_seen],
            device=device,
        )
        engine.snapshot_device(best_world_idx, best_snapshot)
        wp.launch(
            _mark_initial_device_state_kernel,
            dim=1,
            inputs=[
                usable_mask,
                engine.objective.loss,
                search_active,
                initial_loss_device,
                initial_seen,
            ],
            device=device,
        )
        wp.launch(_gate_usable, dim=expected_worlds, inputs=[search_active, usable_mask], device=device)
        wp.launch(
            _latch_poll_baseline,
            dim=islands,
            inputs=[engine.objective.loss, usable_mask, iteration_loss_before, w_per_island],
            device=device,
        )
        wp.launch(
            _select_global_candidate_kernel,
            dim=1,
            inputs=[
                usable_mask,
                engine.objective.loss,
                engine.coefficients,
                search_active,
                best_loss_val,
                best_coeffs,
                best_world_idx,
                expected_worlds,
                parameter_count,
                control_count,
            ],
            device=device,
        )
        engine.snapshot_device(best_world_idx, best_snapshot)

        # Update incumbents from poll batch
        wp.launch(
            _update_island_incumbents_kernel,
            dim=islands,
            inputs=[
                usable_mask,
                engine.objective.loss,
                engine.coefficients,
                wp.float64(minimum_step_fraction),
                island_incumbent,
                island_loss,
                island_fraction,
                island_origin,
                best_world_idx,
                best_loss_val,
                0,
                islands,
                parameter_count,
                control_count,
                w_per_island,
            ],
            device=device,
        )

        wp.launch(
            _record_history,
            dim=1,
            inputs=[counts["batches"], best_loss_val, batch_counters, loss_history, count_history],
            device=device,
        )
        counts["batches"] += 1
        wp.synchronize_device(device)
        batch_walls.append(perf_counter() - batch_started)
        next_batch_allowance = 1.1 * max(batch_walls) + 0.05
        if perf_counter() - search_started + next_batch_allowance >= max_wall_s:
            termination = "wall_budget_exhausted"
            break
        batch_started = perf_counter()
        # ------------------ Phase 2: Solve & Trial Batch ------------------
        wp.launch(
            _assemble_jacobian_kernel,
            dim=residual_dim * islands * parameter_count,
            inputs=[
                engine.objective.residual,
                usable_mask,
                jacobian,
                island_fraction,
                residual_dim,
                parameter_count,
                w_per_island,
            ],
            device=device,
        )
        wp.launch(
            _assemble_normal_equations_kernel,
            dim=islands * parameter_count * (parameter_count + 1),
            inputs=[
                engine.objective.residual,
                jacobian,
                usable_mask,
                H_base,
                residual_dim,
                parameter_count,
                w_per_island,
            ],
            device=device,
        )
        wp.launch(
            _compute_derivatives_and_solve_kernel,
            dim=islands,
            inputs=[
                engine.objective.residual,
                engine.objective.loss,
                usable_mask,
                island_fraction,
                H_base,
                aug_mat,
                gn_directions,
                gn_valid,
                net_pattern,
                net_valid,
                residual_dim,
                parameter_count,
                1,
                w_per_island,
            ],
            device=device,
        )

        wp.launch(
            _generate_trial_coefficients_kernel,
            dim=expected_worlds,
            inputs=[
                island_incumbent,
                island_origin,
                island_fraction,
                scale_wp,
                gn_directions,
                gn_valid,
                net_pattern,
                net_valid,
                rng_states,
                duration_s,
                lower_vec,
                upper_vec,
                rate_vec,
                acc_vec,
                first_scale_wp,
                second_scale_wp,
                engine.coefficients,
                real_mask,
                proposed_mask,
                padding_mask,
                search_active,
                islands,
                parameter_count,
                control_count,
                w_per_island,
            ],
            device=device,
        )

        _launch_forward()

        wp.launch(
            _evaluate_usable_kernel,
            dim=expected_worlds,
            inputs=[
                real_mask,
                engine.failure,
                engine.integrated,
                engine.objective.loss,
                engine.objective.rmse,
                engine.objective.costs,
                engine.objective.residual,
                engine.steps,
                residual_dim,
                usable_mask,
                completed_mask,
                search_active,
            ],
            device=device,
        )

        wp.launch(
            _select_global_candidate_kernel,
            dim=1,
            inputs=[
                usable_mask,
                engine.objective.loss,
                engine.coefficients,
                search_active,
                best_loss_val,
                best_coeffs,
                best_world_idx,
                expected_worlds,
                parameter_count,
                control_count,
            ],
            device=device,
        )
        engine.snapshot_device(best_world_idx, best_snapshot)

        wp.launch(
            _accumulate_batch_counts_kernel,
            dim=expected_worlds,
            inputs=[
                proposed_mask,
                real_mask,
                padding_mask,
                completed_mask,
                engine.integrated,
                batch_counters,
            ],
            device=device,
        )

        # Update incumbents from trial batch
        wp.launch(
            _update_island_incumbents_kernel,
            dim=islands,
            inputs=[
                usable_mask,
                engine.objective.loss,
                engine.coefficients,
                wp.float64(minimum_step_fraction),
                island_incumbent,
                island_loss,
                island_fraction,
                island_origin,
                best_world_idx,
                best_loss_val,
                0,
                islands,
                parameter_count,
                control_count,
                w_per_island,
            ],
            device=device,
        )
        wp.launch(
            _finish_iteration,
            dim=islands,
            inputs=[search_active, island_loss, iteration_loss_before, island_fraction, minimum_step_fraction],
            device=device,
        )
        wp.launch(
            _record_history,
            dim=1,
            inputs=[counts["batches"], best_loss_val, batch_counters, loss_history, count_history],
            device=device,
        )
        counts["batches"] += 1
        iterations_run += 1

        if plateau_enabled:
            if _iteration == 0:
                wp.copy(plateau_anchor, initial_loss_device)
            wp.launch(
                _update_plateau_kernel,
                dim=1,
                inputs=[
                    search_active,
                    best_loss_val,
                    plateau_anchor,
                    plateau_stale,
                    plateau_stop,
                    int(plateau_patience),
                    wp.float64(plateau_rtol),
                ],
                device=device,
            )
            wp.copy(plateau_stop_h, plateau_stop)

        wp.synchronize_device(device)
        batch_walls.append(perf_counter() - batch_started)

        if plateau_enabled and plateau_stop_h.numpy()[0] != 0:
            termination = "loss_plateau"
            break

    wp.synchronize_device(device)
    search_wall_s = perf_counter() - search_started

    counter_values = np.asarray(batch_counters.numpy(), dtype=np.int64)
    counts["physics_worlds"] = int(counter_values[0])
    counts["proposed_candidates"] = int(counter_values[1])
    counts["real_candidates"] = int(counter_values[2])
    counts["padding_worlds"] = int(counter_values[3])
    counts["completed_physics_worlds"] = int(counter_values[4])
    counts["completed_real_candidates"] = int(counter_values[5])
    counts["integrated_steps_including_padding"] = int(counter_values[6])
    counts["unique_candidates"] = None
    counts["unique_candidates_tracked"] = False
    counts["bound_rejections"] = counts["proposed_candidates"] - counts["real_candidates"]
    counts["inactive_worlds"] = counts["padding_worlds"] - counts["bound_rejections"]

    saved_losses = loss_history.numpy()[: counts["batches"]]
    saved_counts = count_history.numpy()[: counts["batches"]]
    final_fractions = island_fraction.numpy()
    history = [
        {
            "batch": batch,
            "iteration": batch // 2,
            "phase": "poll" if batch % 2 == 0 else "trial",
            "best_loss": float(saved_losses[batch]) if np.isfinite(saved_losses[batch]) else None,
            "batch_wall_s": batch_walls[batch],
            "cumulative_device_counts": saved_counts[batch].tolist(),
        }
        for batch in range(counts["batches"])
    ]
    best_loss = float(best_loss_val.numpy()[0])
    initial_loss_value = float(initial_loss_device.numpy()[0])
    if not np.isfinite(initial_loss_value):
        termination = "initial_rollout_incomplete"
    best_coefficients = best_coeffs.numpy().copy()
    best_spline = Spline(engine.duration, best_coefficients)

    snapshot = best_snapshot
    snapshot_result = snapshot.trace()
    if isinstance(snapshot_result, tuple) and len(snapshot_result) == 2:
        best_trace, best_run = snapshot_result
    else:
        best_trace, best_run = snapshot_result, None
    final_scores = snapshot.score()

    if isinstance(final_scores, tuple):
        final_scores = final_scores[0]
    final_loss = float(np.asarray(final_scores["loss"])[0])
    if np.isfinite(best_loss) and final_loss != best_loss:
        raise RuntimeError(f"resident winner score mismatch: retained={best_loss:.17g}, snapshot={final_loss:.17g}")

    metrics = {}
    costs = {}
    if np.isfinite(initial_loss_value):
        for block, (name, unit) in enumerate((("hip", "m"), ("joint", "rad"), ("force", "n"))):
            pair = slice(2 * block, 2 * block + 2)
            metrics[f"{name}_rmse_{unit}"] = np.asarray(final_scores["rmse"])[0, pair].tolist()
            if "maximum_error" in final_scores:
                metrics[f"{name}_maximum_error_{unit}"] = np.asarray(final_scores["maximum_error"])[0, pair].tolist()
            costs[name] = float(final_scores["costs"][0, block])
        diagnostic_names = (
            "hip_velocity_rmse_x_m_s",
            "hip_velocity_rmse_z_m_s",
            "maximum_hip_speed_m_s",
            "maximum_hip_spring_force_n",
            "maximum_hip_damping_force_n",
            "maximum_coulomb_equivalent_ratio",
            "fraction_contact_samples_near_coulomb_limit",
            "contact_sample_fraction",
            "minimum_hip_spring_vertical_force_in_flight_n",
            "hip_spring_work_in_flight_j",
            "hip_damping_work_in_flight_j",
        )
        if "diagnostics" in final_scores:
            diagnostics = np.asarray(final_scores["diagnostics"])[0]
            metrics["rollout_diagnostics"] = {
                name: float(value) for name, value in zip(diagnostic_names, diagnostics, strict=True)
            }
    else:
        final_loss = float("inf")

    final_complete = bool(
        np.isfinite(initial_loss_value)
        and np.isfinite(final_loss)
        and int(np.asarray(final_scores["failure_code"])[0]) == 0
        and int(np.asarray(final_scores["integrated_steps"])[0]) == engine.steps
    )
    best_record = {
        "batch": counts["batches"],
        "iteration": iterations_run,
        "phase": "resident_search_best",
        "world": None,
        "snapshot_world": 0,
        "kind": "shared_controller_winner",
        "coefficients": best_coefficients.tolist(),
        "loss": final_loss,
        "metrics": metrics,
        "objective_components": costs,
        "complete": final_complete,
        "integrated_steps": int(np.asarray(final_scores["integrated_steps"])[0]),
        "failure_code": int(final_scores["failure_code"][0]),
        "coefficients_changed": not np.array_equal(best_coefficients, initial.coefficients),
        "selection": "resident_shared_controller_winner",
    }

    if final_complete:
        if termination == "loss_plateau":
            run_status = "loss_plateau"
        else:
            run_status = "budget_exhausted"
    else:
        run_status = "incomplete"

    summary = {
        "schema": "cartesian_gpu_equilibrium_fit_1",
        "status": run_status,
        "termination": termination,
        "complete": final_complete,
        "converged": False,
        "loss": final_loss if final_complete else None,
        "initial_loss": initial_loss_value if np.isfinite(initial_loss_value) else None,
        "metrics": deepcopy(metrics),
        "objective_components": deepcopy(costs),
        "best": deepcopy(best_record) if final_complete else None,
        "objective": deepcopy(engine.objective.description),
        "counts": counts,
        "history": history,
        "iterations_started": iterations_started,
        "iterations_completed": iterations_run,
        "island_iterations_completed": iterations_run * islands,
        "iterations_per_second": iterations_run / search_wall_s,
        "completed_candidates_per_second": counts["completed_real_candidates"] / search_wall_s,
        "optimizer": {
            "name": "fit_resident",
            "search_mode": "shared_controller",
            "batch_size_policy": "fixed",
            "independent_controllers": islands,
            "islands": islands,
            "seed": seed,
            "max_iterations": max_iterations,
            "max_wall_s": float(max_wall_s),
            "initial_step_fraction": float(initial_step_fraction),
            "minimum_step_fraction": float(minimum_step_fraction),
            "plateau_patience": plateau_patience,
            "plateau_rtol": float(plateau_rtol),
            "plateau_criterion": (
                "stop after plateau_patience consecutive iterations without cumulative loss improvement "
                ">= plateau_rtol * max(abs(anchor), 1e-12), anchor initialized from initial baseline rollout"
                if plateau_enabled
                else "disabled"
            ),
            "final_step_fractions": final_fractions.tolist(),
            "worlds_per_island": w_per_island,
            "worlds": expected_worlds,
            "control_count": control_count,
            "parameter_count": parameter_count,
            "parameter_scale": channel_scale.tolist(),
            "damping": list(_DAMPING),
            "line_factors": list(_LINE_FACTORS),
            "trust_multiplier": _TRUST_MULTIPLIER,
            "absolute_improvement_tolerance": _IMPROVEMENT_TOLERANCE,
            "regularization": "lambda * diag(max(diag(J.T@J), max(1e-12, 1e-12*max(diag(J.T@J)))))",
            "search_backend": "GPU device-resident shared-controller Warp float64 kernels",
            "candidate_allocation_per_controller": {
                "poll_baselines": 1,
                "poll_coordinate_probes": 2 * parameter_count,
                "poll_local_exploration": w_per_island - 1 - 2 * parameter_count,
                "trial_baselines": 1,
                "trial_gauss_newton_slots": 12,
                "trial_pattern_slots": 3,
                "trial_local_exploration": w_per_island - 16,
            },
            "wall_budget": "soft; admit next batch only with 1.1*slowest_completed_batch+0.05 s remaining; "
            "first poll mandatory; no convergence stop",
            "next_batch_safety_factor": 1.1,
            "next_batch_safety_margin_s": 0.05,
            "padding": "bound-rejected or disabled slots remain allocated but are not integrated; other unused slots explore",
            "physics_worlds_scope": "allocated launch slots; completed worlds and integrated steps count actual work",
            "candidate_count_scope": "real proposal slots including baseline reevaluations; global unique count not tracked",
            "failed_initial": "latched on device; later selection disabled; diagnostic returned at final unload",
            "history_count_order": [
                "physics_worlds",
                "proposed_candidates",
                "real_candidates",
                "padding_worlds",
                "completed_physics_worlds",
                "completed_real_candidates",
                "integrated_steps",
            ],
            "final_replay": "none (GPU snapshot; no host full-world replay)",
        },
        "timers": {
            "warmup_wall_s": warmup_wall_s,
            "optimizer_setup_and_warmup_wall_s": search_started - started,
            "search_wall_s": search_wall_s,
            "unload_wall_s": perf_counter() - search_started - search_wall_s,
            "total_wall_s": perf_counter() - started,
        },
        "refinement": {"performed": False, "reason": "caller must run frozen half-step check separately"},
        "wall_s": search_wall_s,
        "qualification": "Numerical search only. No convergence, measured acceptance, or physical validation claim.",
    }

    return best_spline, best_trace, best_run, summary
