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

from typing import Any

import warp as wp


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


@wp.func
def vec_allclose(a: Any, b: Any, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
    """
    Check if two Warp vectors are all close.
    """
    for i in range(wp.static(len(a))):
        if wp.abs(a[i] - b[i]) > atol + rtol * wp.abs(b[i]):
            return False
    return True


@wp.func
def vec_inside_limits(a: Any, lower: Any, upper: Any) -> bool:
    """
    Check if a Warp vector is inside the given limits.
    """
    for i in range(wp.static(len(a))):
        if a[i] < lower[i] or a[i] > upper[i]:
            return False
    return True


@wp.func
def orthonormal_basis(n: wp.vec3):
    r"""Build an orthonormal basis from a normal vector.

    Given a (typically unit-length) normal vector ``n``, this returns two
    tangent vectors ``b1`` and ``b2`` such that:

    .. math::
        b_1 \cdot n = 0,\quad b_2 \cdot n = 0,\quad
        b_1 \cdot b_2 = 0,\quad \|b_1\|=\|b_2\|=1.

    Args:
        n: Normal vector (assumed to be close to unit length).

    Returns:
        Tuple[wp.vec3, wp.vec3]: Orthonormal tangent vectors ``(b1, b2)``.
    """
    b1 = wp.vec3()
    b2 = wp.vec3()
    if n[2] < 0.0:
        a = 1.0 / (1.0 - n[2])
        b = n[0] * n[1] * a
        b1[0] = 1.0 - n[0] * n[0] * a
        b1[1] = -b
        b1[2] = n[0]

        b2[0] = b
        b2[1] = n[1] * n[1] * a - 1.0
        b2[2] = -n[1]
    else:
        a = 1.0 / (1.0 + n[2])
        b = -n[0] * n[1] * a
        b1[0] = 1.0 - n[0] * n[0] * a
        b1[1] = b
        b1[2] = -n[0]

        b2[0] = b
        b2[1] = 1.0 - n[1] * n[1] * a
        b2[2] = -n[1]

    return b1, b2


EPSILON = 1e-15


@wp.func
def safe_div(x: Any, y: Any, eps: float = EPSILON) -> Any:
    """Safe division that avoids division-by-zero."""
    return x / wp.where(y != 0.0, y, eps)


@wp.func
def normalize_with_norm(x: Any):
    """Normalize a vector and return both the normalized vector and the original norm."""
    norm = wp.length(x)
    if norm == 0.0:
        return x, 0.0
    return x / norm, norm


__all__ = [
    "boltzmann",
    "leaky_max",
    "leaky_min",
    "normalize_with_norm",
    "orthonormal_basis",
    "safe_div",
    "smooth_max",
    "smooth_min",
    "vec_abs",
    "vec_allclose",
    "vec_inside_limits",
    "vec_leaky_max",
    "vec_leaky_min",
    "vec_max",
    "vec_min",
]
