# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Native affine-block linear solver primitives for LIMX."""

import warp as wp

from .affine_types import mat1212, vec12

_AFFINE_DIMENSION = 12


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
