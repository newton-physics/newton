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
Defines containers for time-keeping across heterogeneous worlds simulated in parallel.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from .types import float32, int32

###
# Module interface
###

__all__ = [
    "TimeData",
    "TimeModel",
    "advance_time",
]


###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Containers Containers
###


class TimeModel:
    """
    A container to hold the time-invariant gravity model data.
    """

    def __init__(self):
        self.dt: wp.array | None = None
        """
        The discrete time-step size of each world.\n
        Shape of ``(num_worlds,)`` and type :class:`float`.
        """

        self.inv_dt: wp.array | None = None
        """
        The inverse of the discrete time-step size of each world.\n
        Shape of ``(num_worlds,)`` and type :class:`float`.
        """

    def set_uniform_timestep(self, dt: float):
        """
        Sets a uniform discrete time-step for all worlds.

        Args:
            dt (float): The time-step size to set.
        """
        # Ensure that the provided time-step is a floating-point value
        if not isinstance(dt, float):
            raise TypeError(f"Invalid dt type: {type(dt)}. Expected: float.")

        # Ensure that the provided time-step is positive
        if dt <= 0.0:
            raise ValueError(f"Invalid dt value: {dt}. Expected: positive float.")

        # Assign the target time-step uniformly to all worlds
        self.dt.fill_(dt)
        self.inv_dt.fill_(1.0 / dt)

    def set_timesteps(self, dt: list[float] | np.ndarray):
        """
        Sets the discrete time-step of each world explicitly.

        Args:
            dt (list[float] | np.ndarray): An iterable collection of time-steps over all worlds.
        """
        # Ensure that the length of the input matches the number of worlds
        if len(dt) != self.dt.size:
            raise ValueError(f"Invalid dt size: {len(dt)}. Expected: {self.dt.size}.")

        # If the input is a list, convert it to a numpy array
        if isinstance(dt, list):
            dt = np.array(dt, dtype=np.float32)

        # Ensure that the input is a numpy array of the correct dtype
        if not isinstance(dt, np.ndarray):
            raise TypeError(f"Invalid dt type: {type(dt)}. Expected: np.ndarray.")
        if dt.dtype != np.float32:
            raise TypeError(f"Invalid dt dtype: {dt.dtype}. Expected: np.float32.")

        # Assign the values to the internal arrays
        self.dt.assign(dt)
        self.inv_dt.assign(1.0 / dt)


class TimeData:
    """
    A container to hold the time-invariant gravity model data.
    """

    def __init__(self):
        self.steps: wp.array | None = None
        """
        The current number of simulation steps of each world.\n
        Shape of ``(num_worlds,)`` and type :class:`int`.
        """

        self.time: wp.array | None = None
        """
        The current simulation time of each world.\n
        Shape of ``(num_worlds,)`` and type :class:`float`.
        """

    def zero(self):
        """
        Reset the time state to zero.
        """
        self.steps.fill_(0)
        self.time.fill_(0.0)


###
# Kernels
###


@wp.kernel
def _advance_time(
    # Inputs
    dt: wp.array(dtype=float32),
    # Outputs
    steps: wp.array(dtype=int32),  # TODO: Make this uint64
    time: wp.array(dtype=float32),
):
    """
    Reset the current state to the initial state defined in the model.
    """
    # Retrieve the thread index as the world index
    wid = wp.tid()

    # Update the time and step count
    steps[wid] += 1
    time[wid] += dt[wid]


###
# Launchers
###


def advance_time(model: TimeModel, data: TimeData):
    # Ensure the model is valid
    if model is None:
        raise ValueError("'model' must be initialized, is None.")
    elif not isinstance(model, TimeModel):
        raise TypeError("'model' must be an instance of TimeModel.")
    if model.dt is None:
        raise ValueError("'model' must has a `model.dt` array, is None.")

    # Ensure the state is valid
    if data.steps is None:
        raise ValueError("'data' must has a `data.steps` array, is None.")
    elif not isinstance(data, TimeData):
        raise TypeError("'data' must be an instance of TimeData.")
    if data.time is None:
        raise ValueError("'data' must has a `data.time` array, is None.")

    # Retrieve the number of worlds from an array size
    # NOTE: It is assumed that all arrays are of the same size.
    num_worlds = model.dt.size

    #
    wp.launch(
        _advance_time,
        dim=num_worlds,
        inputs=[
            # Inputs:
            model.dt,
            # Outputs:
            data.steps,
            data.time,
        ],
    )
