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
KAMINO: Collision Detection: Math Operations
"""

from __future__ import annotations

import warp as wp

from newton._src.solvers.kamino.core.math import COS_PI_6, UNIT_X, UNIT_Y
from newton._src.solvers.kamino.core.types import mat33f, vec3f

###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Functions
###


@wp.func
def all_same(v0: vec3f, v1: vec3f) -> wp.bool:
    dx = abs(v0[0] - v1[0])
    dy = abs(v0[1] - v1[1])
    dz = abs(v0[2] - v1[2])
    return (
        (dx <= 1.0e-9 or dx <= max(abs(v0[0]), abs(v1[0])) * 1.0e-9)
        and (dy <= 1.0e-9 or dy <= max(abs(v0[1]), abs(v1[1])) * 1.0e-9)
        and (dz <= 1.0e-9 or dz <= max(abs(v0[2]), abs(v1[2])) * 1.0e-9)
    )


@wp.func
def any_different(v0: vec3f, v1: vec3f) -> wp.bool:
    dx = abs(v0[0] - v1[0])
    dy = abs(v0[1] - v1[1])
    dz = abs(v0[2] - v1[2])
    return (
        (dx > 1.0e-9 and dx > max(abs(v0[0]), abs(v1[0])) * 1.0e-9)
        or (dy > 1.0e-9 and dy > max(abs(v0[1]), abs(v1[1])) * 1.0e-9)
        or (dz > 1.0e-9 and dz > max(abs(v0[2]), abs(v1[2])) * 1.0e-9)
    )


@wp.func
def gjk_normalize(a: vec3f):
    norm = wp.length(a)
    if norm > 1e-8 and norm < 1e12:
        return a / norm, True
    return a, False


@wp.func
def orthonormal(normal: vec3f) -> vec3f:
    if wp.abs(normal[0]) < wp.abs(normal[1]) and wp.abs(normal[0]) < wp.abs(normal[2]):
        dir = vec3f(1.0 - normal[0] * normal[0], -normal[0] * normal[1], -normal[0] * normal[2])
    elif wp.abs(normal[1]) < wp.abs(normal[2]):
        dir = vec3f(-normal[1] * normal[0], 1.0 - normal[1] * normal[1], -normal[1] * normal[2])
    else:
        dir = vec3f(-normal[2] * normal[0], -normal[2] * normal[1], 1.0 - normal[2] * normal[2])
        dir, _ = gjk_normalize(dir)
    return dir


@wp.func
def make_contact_frame_znorm(n: vec3f) -> mat33f:
    n = wp.normalize(n)
    if wp.abs(wp.dot(n, UNIT_X)) < COS_PI_6:
        e = UNIT_X
    else:
        e = UNIT_Y
    o = wp.normalize(wp.cross(n, e))
    t = wp.normalize(wp.cross(o, n))
    return mat33f(t.x, o.x, n.x, t.y, o.y, n.y, t.z, o.z, n.z)


@wp.func
def make_contact_frame_xnorm(n: vec3f) -> mat33f:
    n = wp.normalize(n)
    if wp.abs(wp.dot(n, UNIT_X)) < COS_PI_6:
        e = UNIT_X
    else:
        e = UNIT_Y
    o = wp.normalize(wp.cross(n, e))
    t = wp.normalize(wp.cross(o, n))
    return mat33f(n.x, t.x, o.x, n.y, t.y, o.y, n.z, t.z, o.z)
