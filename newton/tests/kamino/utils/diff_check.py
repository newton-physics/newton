# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
KAMINO: UNIT TESTS: Utils for running derivative checks with finite differences
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

###
# Module interface
###

__all__ = ["central_finite_differences", "diff_check"]


def central_finite_differences(fun: Callable, eval_point: float | np.ndarray[float], epsilon: float = 1e-5):
    """
    Evaluates central finite differences of the given function at the given point
    Supports scalar/vector-valued functions, of scalar/vector-valued variables

    Parameters
    ----------
    fun: function
        function to take the derivative of with finite differences, accepting a scalar (float) or a vector (1D numpy array)
        and returning a scalar or a vector
    eval_point: float | np.ndarray
        evaluation point, scalar or vector depending on the signature of function
    epsilon: float, optional
        step size for the central differences, i.e. we evaluate (f(x + epsilon) - f(x - epsilon)) / (2 * epsilon)
        (default: 1e-5)

    Returns
    -------
    derivative: float | np.ndarray
        (approximate) derivative, a scalar for scalar functions of scalars, a 1D vector for vector functions of scalars or
        scalar functions of vectors; a 2D Jacobian for vector functions of vectors
    """
    dim_in = 0
    if isinstance(eval_point, np.ndarray):
        dim_in = len(eval_point)
    if dim_in == 0:
        return (fun(eval_point + epsilon) - fun(eval_point - epsilon)) / (2.0 * epsilon)

    y = np.array(eval_point, copy=True)
    for i in range(dim_in):
        y[i] += epsilon
        v_plus = fun(y)
        y[i] -= 2 * epsilon
        v_minus = fun(y)
        y[i] += epsilon
        fd_val = (v_plus - v_minus) / (2 * epsilon)

        if i == 0:
            dim_out = 0
            if isinstance(fd_val, np.ndarray):
                dim_out = len(fd_val)
            res = np.zeros(dim_in) if dim_out == 0 else np.zeros((dim_out, dim_in))

        if dim_out == 0:
            res[i] = fd_val
        else:
            res[:, i] = fd_val
    return res


def diff_check(
    fun: Callable,
    derivative: float | np.ndarray[float],
    eval_point: float | np.ndarray[float],
    epsilon: float = 1e-5,
    tolerance_rel: float = 1e-6,
    tolerance_abs: float = 1e-6,
):
    """
    Checks the derivative of a function against central differences

    Parameters:
    -----------
    fun: function
        function to check the derivative of, accepting a scalar (float) or a vector (1D numpy array)
        and returning a scalar or a vector
    derivative: float | np.ndarray(dtype=float)
        derivative to check against central differences. A scalar for scalar functions of scalars, a 1D vector for vector
        functions of scalars or scalar functions of vectors; a 2D Jacobian for vector functions of vectors
    eval_point: float | np.ndarray
        evaluation point, scalar or vector depending on the signature of function
    epsilon: float, optional
        step size for the central differences (default: 1e-5)
    tolerance_rel: float, optional
        relative tolerance (default: 1e-6)
    tolerance_abs: float, optional
        absolute tolerance (default: 1e-6)

    Returns:
    --------
    success: bool
        whether the central differences derivative was close to the derivative to check. More specifically, whether
        the absolute error or the relative error was below tolerance

    """
    derivative_fd = central_finite_differences(fun, eval_point, epsilon)
    error = derivative_fd - derivative
    abs_test = np.max(np.abs(error)) <= tolerance_abs
    rel_test = np.linalg.norm(error) <= tolerance_rel * np.linalg.norm(derivative_fd)

    success = abs_test or rel_test

    if not success:
        print("DIFF CHECK FAILED")
        print("DERIVATIVE: ")
        print(derivative)
        print("DERIVATIVE_FD")
        print(derivative_fd)
        if not abs_test:
            print(f"Absolute test failed, error={np.max(np.abs(error))}, tolerance={tolerance_abs}")
        if not rel_test:
            print(
                f"Relative test failed, error={np.linalg.norm(error)}, tolerance={tolerance_rel * np.linalg.norm(derivative_fd)}"
            )

    return success
