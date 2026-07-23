# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The gravity descriptor and model used throughout Kamino"""

from __future__ import annotations

from dataclasses import dataclass

import warp as wp

from .....core.types import override
from .....sim.model import Model
from ..utils import logger as msg
from .types import ArrayLike, Descriptor

###
# Module interface
###

__all__ = [
    "GRAVITY_ACCEL_DEFAULT",
    "GRAVITY_DIREC_DEFAULT",
    "GRAVITY_NAME_DEFAULT",
    "GravityDescriptor",
    "GravityModel",
    "convert_model_gravity",
]


###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Constants
###

GRAVITY_NAME_DEFAULT = "Earth"
"""The default gravity descriptor name, set as 'Earth'."""

GRAVITY_ACCEL_DEFAULT = 9.8067
"""
The default gravitational acceleration in m/s^2.
Equal to Earth's standard gravity of approximately 9.8067 m/s^2.
"""

GRAVITY_DIREC_DEFAULT = wp.constant(wp.vec3f(wp.float32(0.0), wp.float32(0.0), wp.float32(-1.0)))
"""The default direction of gravity defined as -Z."""


###
# Containers
###


class GravityDescriptor(Descriptor):
    """
    A container to describe a world's gravity.

    Attributes:
        name: The name of the gravity descriptor.
        uid: The unique identifier of the gravity descriptor.
        enabled: Whether gravity is enabled.
        acceleration: The gravitational acceleration magnitude [m/s²].
        direction: The normalized direction vector of gravity.
    """

    def __init__(
        self,
        enabled: bool = True,
        acceleration: float = GRAVITY_ACCEL_DEFAULT,
        direction: ArrayLike = GRAVITY_DIREC_DEFAULT,
        name: str = GRAVITY_NAME_DEFAULT,
        uid: str | None = None,
    ):
        """
        Initialize the gravity descriptor.

        Args:
            enabled: Whether gravity is enabled.
                Defaults to `True` to enable gravity by default.
            acceleration: The gravitational acceleration magnitude in m/s^2.
                Defaults to 9.8067 m/s^2 (Earth's gravity).
            direction: The normalized direction vector of gravity.
                Defaults to pointing down the -Z axis.
            name: The name of the gravity descriptor.
            uid: Optional unique identifier of the gravity descriptor.
        """
        super().__init__(name, uid)
        self._enabled: bool = enabled
        self._acceleration: float = acceleration
        self._direction: wp.vec3f = wp.normalize(wp.vec3f(direction))

    @override
    def __repr__(self):
        """Returns a human-readable string representation of the GravityDescriptor."""
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
        """Returns whether gravity is enabled."""
        return self._enabled

    @enabled.setter
    def enabled(self, on: bool):
        """Sets whether gravity is enabled."""
        self._enabled = on

    @property
    def acceleration(self) -> float:
        """Returns the gravitational acceleration."""
        return self._acceleration

    @acceleration.setter
    def acceleration(self, g: float):
        """Sets the gravitational acceleration."""
        self._acceleration = g

    @property
    def direction(self) -> wp.vec3f:
        """Returns the normalized direction vector of gravity."""
        return self._direction

    @direction.setter
    def direction(self, direction: wp.vec3f):
        """Sets the normalized direction vector of gravity."""
        self._direction = wp.normalize(direction)

    def dir_accel(self) -> wp.vec4f:
        """Returns the gravity direction and acceleration as compactly as a :class:`wp.vec4f`."""
        return wp.vec4f([self.direction[0], self.direction[1], self.direction[2], self.acceleration])

    def vector(self) -> wp.vec4f:
        """Returns the effective gravity vector and enabled flag compactly as a :class:`wp.vec4f`."""
        g = wp.vec3f(self.acceleration * self.direction)
        return wp.vec4f([g[0], g[1], g[2], float(self.enabled)])


@dataclass
class GravityModel:
    """
    A container to hold the time-invariant gravity model data.

    Attributes:
        g_dir_acc: The gravity direction and acceleration vector as ``[g_dir_x, g_dir_y, g_dir_z, g_accel]``.
            Shape of ``(num_worlds,)``.
        vector: The gravity vector defined as ``[g_x, g_y, g_z, enabled]``.
            Shape of ``(num_worlds,)``.
    """

    g_dir_acc: wp.array[wp.vec4f] | None = None
    """
    The gravity direction and acceleration vector.
    Shape of ``(num_worlds,)``.
    """

    vector: wp.array[wp.vec4f] | None = None
    """
    The gravity vector defined as ``[g_x, g_y, g_z, enabled]``.
    Shape of ``(num_worlds,)``.
    """

    ###
    # Operations
    ###

    @staticmethod
    def from_newton(model_in: Model) -> GravityModel:
        return convert_model_gravity(model_in)


###
#  Utilities
###


@wp.kernel
def _convert_model_gravity_kernel(
    gravity: wp.array[wp.vec3f],
    g_dir_acc: wp.array[wp.vec4f],
    vector: wp.array[wp.vec4f],
):
    world = wp.tid()
    gravity_vector = gravity[world]
    acceleration = wp.length(gravity_vector)
    direction = GRAVITY_DIREC_DEFAULT
    enabled = 0.0
    if acceleration > 0.0:
        direction = gravity_vector / acceleration
        enabled = 1.0
    g_dir_acc[world] = wp.vec4f(direction[0], direction[1], direction[2], acceleration)
    vector[world] = wp.vec4f(gravity_vector[0], gravity_vector[1], gravity_vector[2], enabled)


def convert_model_gravity(model_in: Model, gravity_out: GravityModel | None = None) -> GravityModel:
    """
    Converts the gravity representation from the Newton model to the Kamino format.

    Args:
        model_in: The input Newton model containing the gravity information to be converted.
        gravity_out: The output GravityModel instance where the converted gravity data will be stored.
            If `None`, a new GravityModel instance will be created and returned.
            If the arrays within `gravity_out` are not already allocated
            with the appropriate shapes, this function will allocate them.
    """
    if gravity_out is None:
        with wp.ScopedDevice(model_in.device):
            gravity_out = GravityModel(
                g_dir_acc=wp.empty(model_in.world_count, dtype=wp.vec4f),
                vector=wp.empty(model_in.world_count, dtype=wp.vec4f),
            )
    else:
        if gravity_out.g_dir_acc is None or gravity_out.g_dir_acc.shape != (model_in.world_count,):
            msg.warning("Output `GravityModel.g_dir_acc` array does not have matching shape. Allocating a new array.")
            gravity_out.g_dir_acc = wp.empty(model_in.world_count, dtype=wp.vec4f, device=model_in.device)
        if gravity_out.vector is None or gravity_out.vector.shape != (model_in.world_count,):
            msg.warning("Output `GravityModel.vector` array does not have matching shape. Allocating a new array.")
            gravity_out.vector = wp.empty(model_in.world_count, dtype=wp.vec4f, device=model_in.device)

    wp.launch(
        kernel=_convert_model_gravity_kernel,
        dim=model_in.world_count,
        inputs=[
            # Inputs:
            model_in.gravity,
            # Outputs:
            gravity_out.g_dir_acc,
            gravity_out.vector,
        ],
        device=model_in.device,
    )

    return gravity_out
