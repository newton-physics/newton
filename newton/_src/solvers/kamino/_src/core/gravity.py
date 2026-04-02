# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The gravity descriptor and model used throughout Kamino"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

from .....sim.model import Model
from ..utils import logger as msg
from .types import ArrayLike, Descriptor, override, vec3f, vec4f

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

GRAVITY_DIREC_DEFAULT = [0.0, 0.0, -1.0]
"""The default direction of gravity defined as -Z."""


###
# Containers
###


class GravityDescriptor(Descriptor):
    """
    A container to describe a world's gravity.

    Attributes:
        name (str): The name of the gravity descriptor.
        uid (str): The unique identifier of the gravity descriptor.
        enabled (bool): Whether gravity is enabled.
        acceleration (float): The gravitational acceleration magnitude in m/s^2.
        direction (vec3f): The normalized direction vector of gravity.
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
            enabled (bool): Whether gravity is enabled.\n
                Defaults to `True` to enable gravity by default.
            acceleration (float): The gravitational acceleration magnitude in m/s^2.\n
                Defaults to 9.8067 m/s^2 (Earth's gravity).
            direction (vec3f): The normalized direction vector of gravity.\n
                Defaults to pointing down the -Z axis.
            name (str): The name of the gravity descriptor.
            uid (str | None): Optional unique identifier of the gravity descriptor.
        """
        super().__init__(name, uid)
        self._enabled: bool = enabled
        self._acceleration: float = acceleration
        self._direction: vec3f = wp.normalize(vec3f(direction))

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
    def direction(self) -> vec3f:
        """Returns the normalized direction vector of gravity."""
        return self._direction

    @direction.setter
    def direction(self, dir: vec3f):
        """Sets the normalized direction vector of gravity."""
        self._direction = wp.normalize(dir)

    def dir_accel(self) -> vec4f:
        """Returns the gravity direction and acceleration as compactly as a :class:`vec4f`."""
        return vec4f([self.direction[0], self.direction[1], self.direction[2], self.acceleration])

    def vector(self) -> vec4f:
        """Returns the effective gravity vector and enabled flag compactly as a :class:`vec4f`."""
        g = vec3f(self.acceleration * self.direction)
        return vec4f([g[0], g[1], g[2], float(self.enabled)])


@dataclass
class GravityModel:
    """
    A container to hold the time-invariant gravity model data.

    Attributes:
        g_dir_acc (wp.array): The gravity direction and acceleration vector as ``[g_dir_x, g_dir_y, g_dir_z, g_accel]``.
        vector (wp.array): The gravity vector defined as ``[g_x, g_y, g_z, enabled]``.
    """

    g_dir_acc: wp.array | None = None
    """
    The gravity direction and acceleration vector.\n
    Shape of ``(num_worlds,)`` and type :class:`vec4f`.
    """

    vector: wp.array | None = None
    """
    The gravity vector defined as ``[g_x, g_y, g_z, enabled]``.\n
    Shape of ``(num_worlds,)`` and type :class:`vec4f`.
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


# TODO: Re-implement using kernels
def convert_model_gravity(model_in: Model, gravity_out: GravityModel | None = None) -> GravityModel:
    """
    Converts the gravity representation from the Newton model to the Kamino format.

    Args:
        model_in (Model):
            The input Newton model containing the gravity information to be converted.
        gravity_out (GravityModel, optional):
            The output GravityModel instance where the converted gravity data will be stored.\n
            If `None`, a new GravityModel instance will be created and returned.\n
            If the arrays within `gravity_out` are not already allocated
            with the appropriate shapes, this function will allocate them.
    """
    # Capture the necessary properties from source model
    gravity_np = model_in.gravity.numpy().copy()

    # Allocate data for the conversion
    g_dir_acc_np = np.zeros((model_in.world_count, 4), dtype=np.float32)
    vector_np = np.zeros((model_in.world_count, 4), dtype=np.float32)

    # Convert each world's gravity vector into direction
    # and acceleration, and pack into the output arrays
    for w in range(model_in.world_count):
        g_vec = gravity_np[w, :]
        accel = float(np.linalg.norm(g_vec))
        if accel > 0.0:
            direction = g_vec / accel
        else:
            direction = np.array([0.0, 0.0, -1.0])
        g_dir_acc_np[w, :3] = direction
        g_dir_acc_np[w, 3] = accel
        vector_np[w, :3] = g_vec
        vector_np[w, 3] = 1.0 if accel > 0.0 else 0.0

    # If the output gravity model is not provided, create a new one with allocated arrays;
    if gravity_out is None:
        with wp.ScopedDevice(model_in.device):
            gravity_out = GravityModel(
                g_dir_acc=wp.array(g_dir_acc_np, dtype=vec4f),
                vector=wp.array(vector_np, dtype=vec4f),
            )

    # Otherwise, ensure the provided model has allocated arrays of the
    # correct shape and type, and copy the converted data into them.
    else:
        # Ensure that the output GravityModel has allocated arrays of the correct shape and type
        if gravity_out.g_dir_acc is None or gravity_out.g_dir_acc.shape != (model_in.world_count, 4):
            msg.warning("Output `GravityModel.g_dir_acc` array does not have matching shape. Allocating a new array.")
            gravity_out.g_dir_acc = wp.array(g_dir_acc_np, dtype=vec4f)
        else:
            gravity_out.g_dir_acc.assign(g_dir_acc_np)
        if gravity_out.vector is None or gravity_out.vector.shape != (model_in.world_count, 4):
            msg.warning("Output `GravityModel.vector` array does not have matching shape. Allocating a new array.")
            gravity_out.vector = wp.array(vector_np, dtype=vec4f)
        else:
            gravity_out.vector.assign(vector_np)

    # Return the output gravity model
    return gravity_out
