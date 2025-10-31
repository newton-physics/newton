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
KAMINO: Gravity Model Module
"""

from __future__ import annotations

import warp as wp

from .types import Descriptor, vec3f, vec4f

###
# Module interface
###

__all__ = [
    "GRAVITY_ACCEL_DEFAULT",
    "GRAVITY_DIREC_DEFAULT",
    "GRAVITY_NAME_DEFAULT",
    "GravityDescriptor",
    "GravityModel",
]


###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Constants
###

GRAVITY_NAME_DEFAULT = "Earth"
"""The name of the default gravity descriptor."""

GRAVITY_ACCEL_DEFAULT = 9.8067
"""The default gravitational acceleration in m/s^2."""

GRAVITY_DIREC_DEFAULT = [0.0, 0.0, -1.0]
"""The default direction of gravity, also defining the global default `up-axis` as +Z."""


###
# Containers
###


class GravityDescriptor(Descriptor):
    """
    A container to describe a world's gravity.
    """

    def __init__(self, name: str = GRAVITY_NAME_DEFAULT):
        super().__init__(name)
        self._enabled: bool = True
        self._acceleration: float = float(GRAVITY_ACCEL_DEFAULT)
        self._direction: vec3f = vec3f(GRAVITY_DIREC_DEFAULT)

    def __repr__(self):
        return (
            f"GravityDescriptor(\n"
            f"name={self.name},\n"
            f"uid={self.uid},\n"
            f"enabled={self.enabled},\n"
            f"acceleration={self.acceleration},\n"
            f"direction={self.direction}\n"
            f")"
        )

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, on: bool):
        self._enabled = on

    @property
    def acceleration(self) -> float:
        return self._acceleration

    @acceleration.setter
    def acceleration(self, g: float):
        self._acceleration = g

    @property
    def direction(self) -> vec3f:
        return self._direction

    @direction.setter
    def direction(self, dir: vec3f):
        self._direction = wp.normalize(dir)

    def dir_accel(self) -> vec4f:
        return vec4f([self.direction[0], self.direction[1], self.direction[2], self.acceleration])

    def vector(self) -> vec4f:
        g = vec3f(self.acceleration * self.direction)
        return vec4f([g[0], g[1], g[2], float(self.enabled)])


class GravityModel:
    """
    A container to hold the time-invariant gravity model data.
    """

    def __init__(self):
        self.g_dir_acc: wp.array(dtype=vec4f) | None = None
        """
        The gravity direction and acceleration vector.\n
        Shape of ``(num_worlds,)`` and type :class:`vec4f`.
        """

        self.vector: wp.array(dtype=vec4f) | None = None
        """
        The gravity vector defined as ``[g_x, g_y, g_z, enabled]``.\n
        Shape of ``(num_worlds,)`` and type :class:`vec4f`.
        """
