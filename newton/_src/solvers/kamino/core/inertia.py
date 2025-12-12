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
Provides functions to compute moments
of inertia for solid geometric bodies.
"""

from .types import mat33f

###
# Module interface
###

__all__ = [
    "solid_cone_body_moment_of_inertia",
    "solid_cuboid_body_moment_of_inertia",
    "solid_cylinder_body_moment_of_inertia",
    "solid_ellipsoid_body_moment_of_inertia",
    "solid_sphere_body_moment_of_inertia",
]


###
# Functions
###


def solid_sphere_body_moment_of_inertia(m: float, r: float) -> mat33f:
    Ia = 0.4 * m * r * r
    i_I_i = mat33f([[Ia, 0.0, 0.0], [0.0, Ia, 0.0], [0.0, 0.0, Ia]])
    return i_I_i


def solid_cylinder_body_moment_of_inertia(m: float, r: float, h: float) -> mat33f:
    Ia = 1.0 / 12.0 * m * (3.0 * r * r + h * h)
    Ib = 0.5 * m * r * r
    i_I_i = mat33f([[Ia, 0.0, 0.0], [0.0, Ia, 0.0], [0.0, 0.0, Ib]])
    return i_I_i


def solid_cone_body_moment_of_inertia(m: float, r: float, h: float) -> mat33f:
    Ia = 0.15 * m * (r * r + 0.25 * h * h)
    Ib = 0.3 * m * r * r
    i_I_i = mat33f([[Ia, 0.0, 0.0], [0.0, Ia, 0.0], [0.0, 0.0, Ib]])
    return i_I_i


def solid_ellipsoid_body_moment_of_inertia(m: float, a: float, b: float, c: float) -> mat33f:
    Ia = 0.2 * m * (b * b + c * c)
    Ib = 0.2 * m * (a * a + c * c)
    Ic = 0.2 * m * (a * a + b * b)
    i_I_i = mat33f([[Ia, 0.0, 0.0], [0.0, Ib, 0.0], [0.0, 0.0, Ic]])
    return i_I_i


def solid_cuboid_body_moment_of_inertia(m: float, w: float, h: float, d: float) -> mat33f:
    Ia = (1.0 / 12.0) * m * (h * h + d * d)
    Ib = (1.0 / 12.0) * m * (w * w + d * d)
    Ic = (1.0 / 12.0) * m * (w * w + h * h)
    i_I_i = mat33f([[Ia, 0.0, 0.0], [0.0, Ib, 0.0], [0.0, 0.0, Ic]])
    return i_I_i
