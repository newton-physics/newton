# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
KAMINO: UNIT TESTS: Utils for running derivative checks on single-joint examples
"""

from __future__ import annotations

import os
from collections.abc import Callable

import numpy as np
from warp.context import Devicelike

from newton._src.solvers.kamino.models import get_testing_usd_assets_path
from newton._src.solvers.kamino.utils.io.usd import USDImporter

###
# Module interface
###

__all__ = ["central_finite_differences", "diff_check", "run_test_single_joint_examples"]


def run_test_single_joint_examples(
    test_fun: Callable,
    test_name: str = "test",
    unary_joints: bool = True,
    binary_joints: bool = True,
    passive_joints: bool = True,
    actuators: bool = True,
    device: Devicelike = None,
):
    """
    Runs a test function over all or a subset of the single-joint examples (e.g. to check some derivatives for all joint types)

    Parameters
    ----------
    test_fun: function
        test function to run on each example, with signature kamino.core.Model -> bool, returning a success flag
    test_name: str, optional
        a name for the test to print as part of the error message upon failure (default: "test")
    unary_joints: bool, optional
        whether to include unary joint examples (NOTE: currently unsupported)
    binary_joints: bool, optional
        whether to include binary joint examples
    passive_joints: bool, optional
        whether to include passive joint examples
    actuators: bool, optional
        whether to include actuator examples
    device: DeviceLike, optional
        device on which to allocate the test models (default: None)

    Returns
    -------
    success: bool
        whether all tests succeeded
    """

    # Resolve path of data folder
    data_dir = os.path.join(get_testing_usd_assets_path(), "joints")

    # List file paths of examples
    file_paths = []
    if unary_joints and passive_joints:
        file_paths.extend(
            [
                os.path.join(data_dir, "test_joint_cartesian_passive_unary.usda"),
                os.path.join(data_dir, "test_joint_cylindrical_passive_unary.usda"),
                os.path.join(data_dir, "test_joint_fixed_unary.usda"),
                os.path.join(data_dir, "test_joint_prismatic_passive_unary.usda"),
                os.path.join(data_dir, "test_joint_revolute_passive_unary.usda"),
                os.path.join(data_dir, "test_joint_spherical_unary.usda"),
                os.path.join(data_dir, "test_joint_universal_passive_unary.usda"),
            ]
        )
    if binary_joints and passive_joints:
        file_paths.extend(
            [
                os.path.join(data_dir, "test_joint_cartesian_passive.usda"),
                os.path.join(data_dir, "test_joint_cylindrical_passive.usda"),
                os.path.join(data_dir, "test_joint_fixed.usda"),
                os.path.join(data_dir, "test_joint_prismatic_passive.usda"),
                os.path.join(data_dir, "test_joint_revolute_passive.usda"),
                os.path.join(data_dir, "test_joint_spherical.usda"),
                os.path.join(data_dir, "test_joint_universal_passive.usda"),
            ]
        )
    if unary_joints and actuators:
        file_paths.extend(
            [
                os.path.join(data_dir, "test_joint_cartesian_actuated_unary.usda"),
                os.path.join(data_dir, "test_joint_cylindrical_actuated_unary.usda"),
                os.path.join(data_dir, "test_joint_prismatic_actuated_unary.usda"),
                os.path.join(data_dir, "test_joint_revolute_actuated_unary.usda"),
                os.path.join(data_dir, "test_joint_universal_actuated_unary.usda"),
                # Note: missing actuated spherical and free
            ]
        )
    if binary_joints and actuators:
        file_paths.extend(
            [
                os.path.join(data_dir, "test_joint_cartesian_actuated.usda"),
                os.path.join(data_dir, "test_joint_cylindrical_actuated.usda"),
                os.path.join(data_dir, "test_joint_prismatic_actuated.usda"),
                os.path.join(data_dir, "test_joint_revolute_actuated.usda"),
                os.path.join(data_dir, "test_joint_universal_actuated.usda"),
                # Note: missing actuated spherical and free
            ]
        )

    # Load and test all examples
    success = True
    for file_path in file_paths:
        importer = USDImporter()
        builder = importer.import_from(source=file_path)
        file_stem_split = os.path.basename(file_path).split(".")[0].split("_")
        unary_binary_str = "unary" if file_stem_split[-1] == "unary" else "binary"
        passive_actuated_str = (
            "actuated" if len(file_stem_split) > 3 and file_stem_split[3] == "actuated" else "passive"
        )
        joint_type_str = file_stem_split[2]

        # Run test
        model = builder.finalize(device=device, requires_grad=False, base_auto=False)
        single_test_success = test_fun(model)
        success &= single_test_success
        if not single_test_success:
            print(f"{test_name} failed for {unary_binary_str} {passive_actuated_str} {joint_type_str} joint")
    return success


def central_finite_differences(fun: Callable, eval_point: float | np.ndarray(dtype=float), epsilon: float = 1e-5):
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
    derivative: float | np.ndarray(dtype=float),
    eval_point: float | np.ndarray(dtype=float),
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
