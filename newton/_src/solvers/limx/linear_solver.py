# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Preconditioned conjugate-gradient solver for LIMX linear systems."""

from __future__ import annotations

from typing import Any

import warp as wp

from .operator import CompositeLinearOperator

_DOT_BLOCK_DIM = 256


@wp.kernel
def _subtract(
    lhs: wp.array[wp.vec3],
    rhs: wp.array[wp.vec3],
    output: wp.array[wp.vec3],
):
    particle = wp.tid()
    output[particle] = lhs[particle] - rhs[particle]


@wp.kernel
def _apply_preconditioner(
    inverse_diagonal: wp.array[wp.mat33],
    residual: wp.array[wp.vec3],
    output: wp.array[wp.vec3],
):
    particle = wp.tid()
    output[particle] = inverse_diagonal[particle] * residual[particle]


@wp.func
def _dot_vec3(lhs: wp.vec3, rhs: wp.vec3) -> float:
    return wp.dot(lhs, rhs)


@wp.kernel
def _reduce_dot(lhs: wp.array[wp.vec3], rhs: wp.array[wp.vec3], output: wp.array[float]):
    block, lane = wp.tid()
    offset = block * _DOT_BLOCK_DIM
    lhs_tile = wp.tile_load(lhs, shape=_DOT_BLOCK_DIM, offset=offset)
    rhs_tile = wp.tile_load(rhs, shape=_DOT_BLOCK_DIM, offset=offset)
    block_sum = wp.tile_sum(wp.tile_map(_dot_vec3, lhs_tile, rhs_tile))
    if lane == 0:
        wp.atomic_add(output, 0, block_sum[0])


@wp.kernel
def _update_direction(
    preconditioned_residual: wp.array[wp.vec3],
    rz: wp.array[float],
    rz_previous: wp.array[float],
    direction: wp.array[wp.vec3],
):
    particle = wp.tid()
    denominator = rz_previous[0]
    beta = float(0.0)
    if wp.isfinite(denominator) and wp.abs(denominator) > 1.0e-30:
        beta = rz[0] / denominator
    if not wp.isfinite(beta):
        beta = 0.0
    direction[particle] = preconditioned_residual[particle] + beta * direction[particle]


@wp.kernel
def _update_solution_and_residual(
    rz: wp.array[float],
    p_ap: wp.array[float],
    direction: wp.array[wp.vec3],
    operator_direction: wp.array[wp.vec3],
    solution: wp.array[wp.vec3],
    residual: wp.array[wp.vec3],
):
    particle = wp.tid()
    denominator = p_ap[0]
    alpha = float(0.0)
    if wp.isfinite(denominator) and wp.abs(denominator) > 1.0e-30:
        alpha = rz[0] / denominator
    if not wp.isfinite(alpha):
        alpha = 0.0
    solution[particle] += alpha * direction[particle]
    residual[particle] -= alpha * operator_direction[particle]


class PcgSolver:
    """Allocation-free block-Jacobi PCG after construction."""

    def __init__(self, dimension: int, device: Any):
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self.dimension = dimension
        self.device = wp.get_device(device)
        self.residual = wp.empty(dimension, dtype=wp.vec3, device=self.device)
        self.preconditioned_residual = wp.empty_like(self.residual)
        self.direction = wp.empty_like(self.residual)
        self.operator_direction = wp.empty_like(self.residual)
        self._rz = wp.zeros(1, dtype=float, device=self.device)
        self._rz_previous = wp.zeros_like(self._rz)
        self._p_ap = wp.zeros_like(self._rz)
        self._residual_squared = wp.zeros_like(self._rz)
        self._dot_block_count = (dimension + _DOT_BLOCK_DIM - 1) // _DOT_BLOCK_DIM

    def solve(
        self,
        operator: CompositeLinearOperator,
        rhs: wp.array[wp.vec3],
        solution: wp.array[wp.vec3],
        iterations: int,
        zero_initial_guess: bool = True,
        tolerance: float | None = None,
        check_interval: int = 1,
    ) -> int:
        """Solve ``operator * solution = rhs``.

        With no tolerance, this executes a fixed number of iterations without a
        device-to-host convergence check. A tolerance enables a debug-oriented
        residual check every ``check_interval`` iterations.

        Returns:
            Number of PCG iterations executed.
        """
        if operator.particle_count != self.dimension or operator.device != self.device:
            raise ValueError("Operator dimensions and device must match the PCG solver")
        self._validate_vector(rhs, "rhs")
        self._validate_vector(solution, "solution")
        if iterations < 0:
            raise ValueError("iterations must not be negative")
        if tolerance is not None and tolerance < 0.0:
            raise ValueError("tolerance must not be negative")
        if check_interval <= 0:
            raise ValueError("check_interval must be positive")

        if zero_initial_guess:
            solution.zero_()
            wp.copy(self.residual, rhs)
        else:
            operator.multiply(solution, self.operator_direction)
            wp.launch(
                _subtract,
                dim=self.dimension,
                inputs=[rhs, self.operator_direction],
                outputs=[self.residual],
                device=self.device,
            )

        self._rz_previous.zero_()
        for iteration in range(iterations):
            wp.launch(
                _apply_preconditioner,
                dim=self.dimension,
                inputs=[operator.inverse_diagonal, self.residual],
                outputs=[self.preconditioned_residual],
                device=self.device,
            )
            self._dot(self.residual, self.preconditioned_residual, self._rz)

            if iteration == 0:
                wp.copy(self.direction, self.preconditioned_residual)
            else:
                wp.launch(
                    _update_direction,
                    dim=self.dimension,
                    inputs=[self.preconditioned_residual, self._rz, self._rz_previous],
                    outputs=[self.direction],
                    device=self.device,
                )

            operator.multiply(self.direction, self.operator_direction)
            self._dot(self.direction, self.operator_direction, self._p_ap)
            wp.launch(
                _update_solution_and_residual,
                dim=self.dimension,
                inputs=[self._rz, self._p_ap, self.direction, self.operator_direction],
                outputs=[solution, self.residual],
                device=self.device,
            )
            wp.copy(self._rz_previous, self._rz)

            executed = iteration + 1
            if tolerance is not None and executed % check_interval == 0:
                self._dot(self.residual, self.residual, self._residual_squared)
                if float(self._residual_squared.numpy()[0]) <= tolerance * tolerance:
                    return executed

        return iterations

    def _dot(
        self,
        lhs: wp.array[wp.vec3],
        rhs: wp.array[wp.vec3],
        output: wp.array[float],
    ) -> None:
        self._validate_vector(lhs, "lhs")
        self._validate_vector(rhs, "rhs")
        if len(output) != 1 or output.device != self.device:
            raise ValueError("Dot-product output must be one scalar on the solver device")

        output.zero_()
        wp.launch_tiled(
            _reduce_dot,
            dim=self._dot_block_count,
            inputs=[lhs, rhs],
            outputs=[output],
            block_dim=_DOT_BLOCK_DIM,
            device=self.device,
        )

    def _validate_vector(self, vector: wp.array[wp.vec3], name: str) -> None:
        if len(vector) != self.dimension:
            raise ValueError(f"{name} must contain {self.dimension} vectors")
        if vector.device != self.device:
            raise ValueError(f"{name} must use device {self.device}")
