# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Differentiate the unchanged measured residuals using their existing sampling maps."""

from __future__ import annotations

import warp as wp

from .mechanics import Vec5

wp.set_module_options({"enable_backward": True, "fuse_fp": False})


@wp.kernel
def _residuals(
    states: wp.array2d[Vec5],
    forces: wp.array2d[wp.vec2d],
    motion_count: int,
    lower: wp.array[int],
    upper: wp.array[int],
    fraction: wp.array[wp.float64],
    targets: wp.array[wp.vec2d],
    weights: wp.array[wp.float64],
    scales: wp.array[wp.float64],
    ground_angle: int,
    ground_offset: wp.float64,
    residual: wp.array2d[wp.float64],
):
    """Evaluate the same native-grid sample pairs independently for reverse mode."""
    k, w = wp.tid()
    sample = k // 2
    channel = k % 2
    lo = lower[sample]
    hi = upper[sample]
    alpha = fraction[sample]
    block = int(2)
    predicted = wp.float64(0.0)
    if sample < 2 * motion_count:
        coordinate = channel
        block = 0
        if sample >= motion_count:
            coordinate = channel + 3
            block = 1
        predicted = states[lo, w][coordinate] * (wp.float64(1.0) - alpha) + states[hi, w][coordinate] * alpha
        if block == 1 and channel == 1 and ground_angle != 0:
            state_lo = states[lo, w]
            state_hi = states[hi, w]
            lo_pitch = ((state_lo[2] + state_lo[3]) + state_lo[4]) + ground_offset
            hi_pitch = ((state_hi[2] + state_hi[3]) + state_hi[4]) + ground_offset
            predicted = lo_pitch * (wp.float64(1.0) - alpha) + hi_pitch * alpha
    else:
        predicted = forces[lo, w][channel] * (wp.float64(1.0) - alpha) + forces[hi, w][channel] * alpha
    residual[k, w] = (predicted - targets[sample][channel]) / scales[block] * weights[sample]


@wp.func
def _block_sum(residual: wp.array2d[wp.float64], w: int, offset: int, count: int) -> wp.float64:
    """Sum residual squares in the original sample/channel order."""
    value = wp.float64(0.0)
    for k in range(offset, offset + count):
        r = residual[k, w]
        value += r * r
    return value


@wp.func_grad(_block_sum)
def _adj_block_sum(residual: wp.array2d[wp.float64], w: int, offset: int, count: int, adj_value: wp.float64):
    """Apply the exact sum-of-squares VJP without relying on dynamic-loop replay."""
    for k in range(offset, offset + count):
        wp.atomic_add(wp.adjoint[residual], k, w, wp.float64(2.0) * residual[k, w] * adj_value)


@wp.kernel
def _costs(residual: wp.array2d[wp.float64], motion_count: int, force_count: int, costs: wp.array2d[wp.float64]):
    """Keep separate block reductions before the final measured-loss sum."""
    w, block = wp.tid()
    count = 2 * motion_count
    if block == 2:
        count = 2 * force_count
    costs[w, block] = _block_sum(residual, w, 2 * block * motion_count, count)


@wp.kernel
def _loss(costs: wp.array2d[wp.float64], loss: wp.array[wp.float64]):
    """Accumulate the original three block costs without an extra one-half factor."""
    w = wp.tid()
    total = wp.float64(0.0)
    for block in range(3):
        total += costs[w, block]
    wp.atomic_add(loss, 0, total)


class ObjectiveAdjoint:
    """Reuse a MeasuredObjective's maps, targets, and normalization for a VJP.

    Callers must validate the full trajectory before differentiating this scalar.
    A sum over worlds is returned; controller optimization initially uses one world.
    """

    def __init__(self, source):
        self.source = source
        self.residual = wp.zeros_like(source.residual, requires_grad=True)
        self.costs = wp.zeros_like(source.costs, requires_grad=True)

    def launch(self, states, forces, loss):
        """Record residual evaluation and exact block reductions on the current Tape."""
        s = self.source
        wp.launch(
            _residuals,
            dim=(s.residual_dim, s.world_count),
            inputs=[
                states,
                forces,
                s.motion_sample_count,
                s._lower,
                s._upper,
                s._fraction,
                s._targets,
                s._residual_weights,
                s._scales,
                s.ground_angle,
                wp.float64(s.ground_offset),
                self.residual,
            ],
            device=s.device,
        )
        wp.launch(
            _costs,
            dim=(s.world_count, 3),
            inputs=[self.residual, s.motion_sample_count, s.force_sample_count, self.costs],
            device=s.device,
        )
        wp.launch(_loss, dim=s.world_count, inputs=[self.costs, loss], device=s.device)
