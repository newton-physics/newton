# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Native affine-block and heterogeneous PCG solver primitives for LIMX."""

from __future__ import annotations

from typing import Any

import warp as wp

from .affine_types import mat1212, vec12

_AFFINE_DIMENSION = 12
_DOT_BLOCK_DIM = 256


@wp.func
def _factor_affine_block(
    block: mat1212,
    diagonal_shift: float,
    pivot_floor: float,
) -> tuple[mat1212, int]:
    factor = mat1212(0.0)
    valid = 1

    for column in range(_AFFINE_DIMENSION):
        for row in range(column, _AFFINE_DIMENSION):
            value = 0.5 * (block[row, column] + block[column, row])
            if not wp.isfinite(value):
                valid = 0
                value = 0.0
            if row == column:
                value += diagonal_shift

            for inner in range(column):
                value -= factor[row, inner] * factor[column, inner]

            if row == column:
                if not wp.isfinite(value) or value < pivot_floor:
                    valid = 0
                factor[row, column] = wp.sqrt(wp.max(value, pivot_floor))
            else:
                factor[row, column] = value / factor[column, column]

    return factor, valid


@wp.kernel
def _factor_affine_diagonal(
    diagonal: wp.array[mat1212],
    factors: wp.array[mat1212],
    regularization: wp.array[int],
):
    """Factor affine diagonal blocks with one regularized retry."""
    row = wp.tid()
    block = diagonal[row]
    trace = float(0.0)
    for component in range(_AFFINE_DIMENSION):
        trace += block[component, component]
    pivot_floor = wp.max(1.0e-8, 1.0e-6 * trace / float(_AFFINE_DIMENSION))
    if not wp.isfinite(pivot_floor):
        pivot_floor = 1.0e-8

    factor, valid = _factor_affine_block(block, 0.0, pivot_floor)
    regularization[row] = 0
    if valid == 0:
        factor, valid = _factor_affine_block(block, pivot_floor, pivot_floor)
        regularization[row] = 1
        if valid == 0:
            factor = mat1212(0.0)
            for component in range(_AFFINE_DIMENSION):
                factor[component, component] = wp.sqrt(pivot_floor)
    factors[row] = factor


@wp.kernel
def _apply_affine_preconditioner(
    factors: wp.array[mat1212],
    residual: wp.array[vec12],
    output: wp.array[vec12],
):
    """Apply native affine Cholesky factors through triangular solves."""
    row = wp.tid()
    factor = factors[row]
    intermediate = vec12(0.0)
    solution = vec12(0.0)

    for component in range(_AFFINE_DIMENSION):
        value = residual[row][component]
        for inner in range(component):
            value -= factor[component, inner] * intermediate[inner]
        intermediate[component] = value / factor[component, component]

    for reverse_component in range(_AFFINE_DIMENSION):
        component = _AFFINE_DIMENSION - 1 - reverse_component
        value = intermediate[component]
        for inner in range(component + 1, _AFFINE_DIMENSION):
            value -= factor[inner, component] * solution[inner]
        solution[component] = value / factor[component, component]

    output[row] = solution


@wp.kernel
def _subtract_particle(
    lhs: wp.array[wp.vec3],
    rhs: wp.array[wp.vec3],
    output: wp.array[wp.vec3],
):
    row = wp.tid()
    output[row] = lhs[row] - rhs[row]


@wp.kernel
def _subtract_affine(
    lhs: wp.array[vec12],
    rhs: wp.array[vec12],
    output: wp.array[vec12],
):
    row = wp.tid()
    output[row] = lhs[row] - rhs[row]


@wp.kernel
def _apply_particle_preconditioner(
    inverse_diagonal: wp.array[wp.mat33],
    residual: wp.array[wp.vec3],
    output: wp.array[wp.vec3],
):
    row = wp.tid()
    output[row] = inverse_diagonal[row] * residual[row]


@wp.func
def _dot_vec3(lhs: wp.vec3, rhs: wp.vec3) -> float:
    return wp.dot(lhs, rhs)


@wp.func
def _dot_vec12(lhs: vec12, rhs: vec12) -> float:
    value = float(0.0)
    for component in range(_AFFINE_DIMENSION):
        value += lhs[component] * rhs[component]
    return value


@wp.kernel
def _reduce_particle_dot(lhs: wp.array[wp.vec3], rhs: wp.array[wp.vec3], output: wp.array[float]):
    block, lane = wp.tid()
    offset = block * _DOT_BLOCK_DIM
    lhs_tile = wp.tile_load(lhs, shape=_DOT_BLOCK_DIM, offset=offset)
    rhs_tile = wp.tile_load(rhs, shape=_DOT_BLOCK_DIM, offset=offset)
    block_sum = wp.tile_sum(wp.tile_map(_dot_vec3, lhs_tile, rhs_tile))
    if lane == 0:
        wp.atomic_add(output, 0, block_sum[0])


@wp.kernel
def _reduce_affine_dot(lhs: wp.array[vec12], rhs: wp.array[vec12], output: wp.array[float]):
    block, lane = wp.tid()
    offset = block * _DOT_BLOCK_DIM
    lhs_tile = wp.tile_load(lhs, shape=_DOT_BLOCK_DIM, offset=offset)
    rhs_tile = wp.tile_load(rhs, shape=_DOT_BLOCK_DIM, offset=offset)
    block_sum = wp.tile_sum(wp.tile_map(_dot_vec12, lhs_tile, rhs_tile))
    if lane == 0:
        wp.atomic_add(output, 0, block_sum[0])


@wp.kernel
def _update_particle_direction(
    preconditioned_residual: wp.array[wp.vec3],
    rz: wp.array[float],
    rz_previous: wp.array[float],
    direction: wp.array[wp.vec3],
):
    row = wp.tid()
    denominator = rz_previous[0]
    beta = float(0.0)
    if wp.isfinite(denominator) and wp.abs(denominator) > 1.0e-30:
        beta = rz[0] / denominator
    if not wp.isfinite(beta):
        beta = 0.0
    direction[row] = preconditioned_residual[row] + beta * direction[row]


@wp.kernel
def _update_affine_direction(
    preconditioned_residual: wp.array[vec12],
    rz: wp.array[float],
    rz_previous: wp.array[float],
    direction: wp.array[vec12],
):
    row = wp.tid()
    denominator = rz_previous[0]
    beta = float(0.0)
    if wp.isfinite(denominator) and wp.abs(denominator) > 1.0e-30:
        beta = rz[0] / denominator
    if not wp.isfinite(beta):
        beta = 0.0
    direction[row] = preconditioned_residual[row] + beta * direction[row]


@wp.kernel
def _update_particle_solution_and_residual(
    rz: wp.array[float],
    p_ap: wp.array[float],
    direction: wp.array[wp.vec3],
    operator_direction: wp.array[wp.vec3],
    solution: wp.array[wp.vec3],
    residual: wp.array[wp.vec3],
):
    row = wp.tid()
    denominator = p_ap[0]
    alpha = float(0.0)
    if wp.isfinite(denominator) and wp.abs(denominator) > 1.0e-30:
        alpha = rz[0] / denominator
    if not wp.isfinite(alpha):
        alpha = 0.0
    solution[row] += alpha * direction[row]
    residual[row] -= alpha * operator_direction[row]


@wp.kernel
def _update_affine_solution_and_residual(
    rz: wp.array[float],
    p_ap: wp.array[float],
    direction: wp.array[vec12],
    operator_direction: wp.array[vec12],
    solution: wp.array[vec12],
    residual: wp.array[vec12],
):
    row = wp.tid()
    denominator = p_ap[0]
    alpha = float(0.0)
    if wp.isfinite(denominator) and wp.abs(denominator) > 1.0e-30:
        alpha = rz[0] / denominator
    if not wp.isfinite(alpha):
        alpha = 0.0
    solution[row] += alpha * direction[row]
    residual[row] -= alpha * operator_direction[row]


class MixedPcgSolver:
    """Allocation-free split-vector block-Jacobi PCG after construction."""

    def __init__(self, particle_count: int, affine_count: int, device: Any):
        if particle_count < 0 or affine_count < 0:
            raise ValueError("particle_count and affine_count must not be negative")
        self.particle_count = particle_count
        self.affine_count = affine_count
        self.device = wp.get_device(device)
        self.particle_residual = wp.empty(particle_count, dtype=wp.vec3, device=self.device)
        self.particle_preconditioned_residual = wp.empty_like(self.particle_residual)
        self.particle_direction = wp.empty_like(self.particle_residual)
        self.particle_operator_direction = wp.empty_like(self.particle_residual)
        self.affine_residual = wp.empty(affine_count, dtype=vec12, device=self.device)
        self.affine_preconditioned_residual = wp.empty_like(self.affine_residual)
        self.affine_direction = wp.empty_like(self.affine_residual)
        self.affine_operator_direction = wp.empty_like(self.affine_residual)
        self._rz = wp.zeros(1, dtype=float, device=self.device)
        self._rz_previous = wp.zeros_like(self._rz)
        self._p_ap = wp.zeros_like(self._rz)
        self._residual_squared = wp.zeros_like(self._rz)
        self._particle_dot_block_count = (particle_count + _DOT_BLOCK_DIM - 1) // _DOT_BLOCK_DIM
        self._affine_dot_block_count = (affine_count + _DOT_BLOCK_DIM - 1) // _DOT_BLOCK_DIM

    def solve(
        self,
        operator: Any,
        rhs: Any,
        solution: Any,
        iterations: int,
        zero_initial_guess: bool = True,
        tolerance: float | None = None,
        check_interval: int = 1,
    ) -> int:
        """Solve a prepared heterogeneous linear system.

        With no tolerance, this executes a fixed number of iterations without a
        device-to-host convergence check. A tolerance enables a debug-oriented
        residual check every ``check_interval`` iterations.

        Returns:
            Number of PCG iterations executed.
        """
        if operator.particle_count != self.particle_count or operator.affine_count != self.affine_count:
            raise ValueError("Operator dimensions must match the mixed PCG solver")
        if operator.device != self.device:
            raise ValueError("Operator and mixed PCG solver must use the same device")
        self._validate_vectors(rhs, "rhs")
        self._validate_vectors(solution, "solution")
        if iterations < 0:
            raise ValueError("iterations must not be negative")
        if tolerance is not None and tolerance < 0.0:
            raise ValueError("tolerance must not be negative")
        if check_interval <= 0:
            raise ValueError("check_interval must be positive")

        if zero_initial_guess:
            solution.particle.zero_()
            solution.affine.zero_()
            wp.copy(self.particle_residual, rhs.particle)
            wp.copy(self.affine_residual, rhs.affine)
        else:
            operator.multiply(
                solution.particle,
                solution.affine,
                self.particle_operator_direction,
                self.affine_operator_direction,
            )
            self._subtract(
                rhs.particle,
                rhs.affine,
                self.particle_operator_direction,
                self.affine_operator_direction,
                self.particle_residual,
                self.affine_residual,
            )

        self._rz_previous.zero_()
        for iteration in range(iterations):
            self._apply_preconditioner(operator)
            self._dot(
                self.particle_residual,
                self.affine_residual,
                self.particle_preconditioned_residual,
                self.affine_preconditioned_residual,
                self._rz,
            )

            if iteration == 0:
                wp.copy(self.particle_direction, self.particle_preconditioned_residual)
                wp.copy(self.affine_direction, self.affine_preconditioned_residual)
            else:
                self._update_direction()

            operator.multiply(
                self.particle_direction,
                self.affine_direction,
                self.particle_operator_direction,
                self.affine_operator_direction,
            )
            self._dot(
                self.particle_direction,
                self.affine_direction,
                self.particle_operator_direction,
                self.affine_operator_direction,
                self._p_ap,
            )
            self._update_solution_and_residual(solution)
            wp.copy(self._rz_previous, self._rz)

            executed = iteration + 1
            if tolerance is not None and executed % check_interval == 0:
                self._dot(
                    self.particle_residual,
                    self.affine_residual,
                    self.particle_residual,
                    self.affine_residual,
                    self._residual_squared,
                )
                if float(self._residual_squared.numpy()[0]) <= tolerance * tolerance:
                    return executed

        return iterations

    def _apply_preconditioner(self, operator: Any) -> None:
        if self.particle_count:
            wp.launch(
                _apply_particle_preconditioner,
                dim=self.particle_count,
                inputs=[operator.particle_operator.inverse_diagonal, self.particle_residual],
                outputs=[self.particle_preconditioned_residual],
                device=self.device,
            )
        if self.affine_count:
            wp.launch(
                _apply_affine_preconditioner,
                dim=self.affine_count,
                inputs=[operator.affine_factors, self.affine_residual],
                outputs=[self.affine_preconditioned_residual],
                device=self.device,
            )

    def _subtract(
        self,
        particle_lhs: wp.array[wp.vec3],
        affine_lhs: wp.array[vec12],
        particle_rhs: wp.array[wp.vec3],
        affine_rhs: wp.array[vec12],
        particle_output: wp.array[wp.vec3],
        affine_output: wp.array[vec12],
    ) -> None:
        if self.particle_count:
            wp.launch(
                _subtract_particle,
                dim=self.particle_count,
                inputs=[particle_lhs, particle_rhs],
                outputs=[particle_output],
                device=self.device,
            )
        if self.affine_count:
            wp.launch(
                _subtract_affine,
                dim=self.affine_count,
                inputs=[affine_lhs, affine_rhs],
                outputs=[affine_output],
                device=self.device,
            )

    def _update_direction(self) -> None:
        if self.particle_count:
            wp.launch(
                _update_particle_direction,
                dim=self.particle_count,
                inputs=[self.particle_preconditioned_residual, self._rz, self._rz_previous],
                outputs=[self.particle_direction],
                device=self.device,
            )
        if self.affine_count:
            wp.launch(
                _update_affine_direction,
                dim=self.affine_count,
                inputs=[self.affine_preconditioned_residual, self._rz, self._rz_previous],
                outputs=[self.affine_direction],
                device=self.device,
            )

    def _update_solution_and_residual(self, solution: Any) -> None:
        if self.particle_count:
            wp.launch(
                _update_particle_solution_and_residual,
                dim=self.particle_count,
                inputs=[self._rz, self._p_ap, self.particle_direction, self.particle_operator_direction],
                outputs=[solution.particle, self.particle_residual],
                device=self.device,
            )
        if self.affine_count:
            wp.launch(
                _update_affine_solution_and_residual,
                dim=self.affine_count,
                inputs=[self._rz, self._p_ap, self.affine_direction, self.affine_operator_direction],
                outputs=[solution.affine, self.affine_residual],
                device=self.device,
            )

    def _dot(
        self,
        particle_lhs: wp.array[wp.vec3],
        affine_lhs: wp.array[vec12],
        particle_rhs: wp.array[wp.vec3],
        affine_rhs: wp.array[vec12],
        output: wp.array[float],
    ) -> None:
        self._validate_particle_vector(particle_lhs, "particle_lhs")
        self._validate_particle_vector(particle_rhs, "particle_rhs")
        self._validate_affine_vector(affine_lhs, "affine_lhs")
        self._validate_affine_vector(affine_rhs, "affine_rhs")
        if len(output) != 1 or output.device != self.device:
            raise ValueError("Dot-product output must be one scalar on the solver device")

        output.zero_()
        if self._particle_dot_block_count:
            wp.launch_tiled(
                _reduce_particle_dot,
                dim=self._particle_dot_block_count,
                inputs=[particle_lhs, particle_rhs],
                outputs=[output],
                block_dim=_DOT_BLOCK_DIM,
                device=self.device,
            )
        if self._affine_dot_block_count:
            wp.launch_tiled(
                _reduce_affine_dot,
                dim=self._affine_dot_block_count,
                inputs=[affine_lhs, affine_rhs],
                outputs=[output],
                block_dim=_DOT_BLOCK_DIM,
                device=self.device,
            )

    def _validate_vectors(self, vectors: Any, name: str) -> None:
        self._validate_particle_vector(vectors.particle, f"{name}.particle")
        self._validate_affine_vector(vectors.affine, f"{name}.affine")

    def _validate_particle_vector(self, vector: wp.array[wp.vec3], name: str) -> None:
        if len(vector) != self.particle_count:
            raise ValueError(f"{name} must contain {self.particle_count} vectors")
        if vector.device != self.device:
            raise ValueError(f"{name} must use device {self.device}")

    def _validate_affine_vector(self, vector: wp.array[vec12], name: str) -> None:
        if len(vector) != self.affine_count:
            raise ValueError(f"{name} must contain {self.affine_count} vectors")
        if vector.device != self.device:
            raise ValueError(f"{name} must use device {self.device}")
