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

import warp as wp
from warp.context import assert_conditional_graph_support

from .download_assets import clear_git_cache, download_asset
from .render import RendererOpenGL, RendererUsd, SimRenderer
from .topology import topological_sort


@wp.func
def transform_inertia(t: wp.transform, I: wp.spatial_matrix):
    """
    Transform a spatial inertia tensor to a new coordinate frame.

    This computes the change of coordinates for a spatial inertia tensor under a rigid-body
    transformation `t`. The result is mathematically equivalent to:

        adj_t^-T * I * adj_t^-1

    where `adj_t` is the adjoint transformation matrix of `t`, and `I` is the spatial inertia
    tensor in the original frame. This operation is described in Frank & Park, "Modern Robotics",
    Section 8.2.3 (pg. 290).

    Args:
        t (wp.transform): The rigid-body transform (destination ← source).
        I (wp.spatial_matrix): The spatial inertia tensor in the source frame.

    Returns:
        wp.spatial_matrix: The spatial inertia tensor expressed in the destination frame.
    """
    t_inv = wp.transform_inverse(t)

    q = wp.transform_get_rotation(t_inv)
    p = wp.transform_get_translation(t_inv)

    r1 = wp.quat_rotate(q, wp.vec3(1.0, 0.0, 0.0))
    r2 = wp.quat_rotate(q, wp.vec3(0.0, 1.0, 0.0))
    r3 = wp.quat_rotate(q, wp.vec3(0.0, 0.0, 1.0))

    R = wp.matrix_from_cols(r1, r2, r3)
    S = wp.mul(wp.skew(p), R)

    T = wp.spatial_adjoint(R, S)

    return wp.mul(wp.mul(wp.transpose(T), I), T)


@wp.func
def boltzmann(a: float, b: float, alpha: float):
    """
    Compute the Boltzmann-weighted average of two values.

    This function returns a smooth interpolation between `a` and `b` using a Boltzmann (softmax-like) weighting,
    controlled by the parameter `alpha`. As `alpha` increases, the result approaches `max(a, b)`;
    as `alpha` decreases, the result approaches the mean of `a` and `b`.

    Args:
        a (float): The first value.
        b (float): The second value.
        alpha (float): The sharpness parameter. Higher values make the function more "max-like".

    Returns:
        float: The Boltzmann-weighted average of `a` and `b`.
    """
    e1 = wp.exp(alpha * a)
    e2 = wp.exp(alpha * b)
    return (a * e1 + b * e2) / (e1 + e2)


@wp.func
def smooth_max(a: float, b: float, eps: float):
    """
    Compute a smooth approximation of the maximum of two values.

    This function returns a value close to `max(a, b)`, but is differentiable everywhere.
    The `eps` parameter controls the smoothness: larger values make the transition smoother.

    Args:
        a (float): The first value.
        b (float): The second value.
        eps (float): Smoothing parameter (should be small and positive).

    Returns:
        float: A smooth approximation of `max(a, b)`.
    """
    d = a - b
    return 0.5 * (a + b + wp.sqrt(d * d + eps))


@wp.func
def smooth_min(a: float, b: float, eps: float):
    """
    Compute a smooth approximation of the minimum of two values.

    This function returns a value close to `min(a, b)`, but is differentiable everywhere.
    The `eps` parameter controls the smoothness: larger values make the transition smoother.

    Args:
        a (float): The first value.
        b (float): The second value.
        eps (float): Smoothing parameter (should be small and positive).

    Returns:
        float: A smooth approximation of `min(a, b)`.
    """
    d = a - b
    return 0.5 * (a + b - wp.sqrt(d * d + eps))


@wp.func
def leaky_max(a: float, b: float):
    """
    Compute a numerically stable, differentiable approximation of `max(a, b)`.

    This is equivalent to `smooth_max(a, b, 1e-5)`.

    Args:
        a (float): The first value.
        b (float): The second value.

    Returns:
        float: A smooth, "leaky" maximum of `a` and `b`.
    """
    return smooth_max(a, b, 1e-5)


@wp.func
def leaky_min(a: float, b: float):
    """
    Compute a numerically stable, differentiable approximation of `min(a, b)`.

    This is equivalent to `smooth_min(a, b, 1e-5)`.

    Args:
        a (float): The first value.
        b (float): The second value.

    Returns:
        float: A smooth, "leaky" minimum of `a` and `b`.
    """
    return smooth_min(a, b, 1e-5)


@wp.func
def vec_min(a: wp.vec3, b: wp.vec3):
    """
    Compute the elementwise minimum of two 3D vectors.

    Args:
        a (wp.vec3): The first vector.
        b (wp.vec3): The second vector.

    Returns:
        wp.vec3: The elementwise minimum.
    """
    return wp.vec3(wp.min(a[0], b[0]), wp.min(a[1], b[1]), wp.min(a[2], b[2]))


@wp.func
def vec_max(a: wp.vec3, b: wp.vec3):
    """
    Compute the elementwise maximum of two 3D vectors.

    Args:
        a (wp.vec3): The first vector.
        b (wp.vec3): The second vector.

    Returns:
        wp.vec3: The elementwise maximum.
    """
    return wp.vec3(wp.max(a[0], b[0]), wp.max(a[1], b[1]), wp.max(a[2], b[2]))


@wp.func
def vec_leaky_min(a: wp.vec3, b: wp.vec3):
    """
    Compute the elementwise "leaky" minimum of two 3D vectors.

    This uses `leaky_min` for each component.

    Args:
        a (wp.vec3): The first vector.
        b (wp.vec3): The second vector.

    Returns:
        wp.vec3: The elementwise leaky minimum.
    """
    return wp.vec3(leaky_min(a[0], b[0]), leaky_min(a[1], b[1]), leaky_min(a[2], b[2]))


@wp.func
def vec_leaky_max(a: wp.vec3, b: wp.vec3):
    """
    Compute the elementwise "leaky" maximum of two 3D vectors.

    This uses `leaky_max` for each component.

    Args:
        a (wp.vec3): The first vector.
        b (wp.vec3): The second vector.

    Returns:
        wp.vec3: The elementwise leaky maximum.
    """
    return wp.vec3(leaky_max(a[0], b[0]), leaky_max(a[1], b[1]), leaky_max(a[2], b[2]))


@wp.func
def vec_abs(a: wp.vec3):
    """
    Compute the elementwise absolute value of a 3D vector.

    Args:
        a (wp.vec3): The input vector.

    Returns:
        wp.vec3: The elementwise absolute value.
    """
    return wp.vec3(wp.abs(a[0]), wp.abs(a[1]), wp.abs(a[2]))


def check_conditional_graph_support():
    """
    Check if conditional graph support is available in the current environment.

    Returns:
        bool: True if conditional graph support is available, False otherwise.
    """
    try:
        assert_conditional_graph_support()
    except Exception:
        return False
    return True


__all__ = [
    "RendererOpenGL",
    "RendererUsd",
    "SimRenderer",
    "boltzmann",
    "check_conditional_graph_support",
    "clear_git_cache",
    "download_asset",
    "leaky_max",
    "leaky_min",
    "smooth_max",
    "smooth_min",
    "topological_sort",
    "vec_abs",
    "vec_leaky_max",
    "vec_leaky_min",
    "vec_max",
    "vec_min",
]
