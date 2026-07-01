# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
Defines containers for time-keeping across heterogeneous worlds simulated in parallel.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

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
# Containers
###


@dataclass
class TimeModel:
    """
    A container to hold heterogeneous model time-step.

    Attributes:
        dt: The discrete time-step size of each world.
            Shape of ``(num_worlds,)``.
        inv_dt: The inverse of the discrete time-step size of each world.
            Shape of ``(num_worlds,)``.
    """

    dt: wp.array[wp.float32] | None = None
    """
    The discrete time-step size of each world.
    Shape of ``(num_worlds,)``.
    """

    inv_dt: wp.array[wp.float32] | None = None
    """
    The inverse of the discrete time-step size of each world.
    Shape of ``(num_worlds,)``.
    """

    def set_uniform_timestep(self, dt: float):
        """
        Sets a uniform discrete time-step for all worlds.

        Args:
            dt: The time-step size to set.
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
            dt: An iterable collection of time-steps over all worlds.
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


@dataclass
class TimeData:
    """
    A container to hold heterogeneous model time-keeping data.

    Attributes:
        steps: The current number of simulation steps of each world.
            Shape of ``(num_worlds,)``.
        time: The current simulation time of each world.
            Shape of ``(num_worlds,)``.
    """

    steps: wp.array[wp.int32] | None = None
    """
    The current number of simulation steps of each world.
    Shape of ``(num_worlds,)``.
    """

    time: wp.array[wp.float32] | None = None
    """
    The current simulation time of each world.
    Shape of ``(num_worlds,)``.
    """

    def reset(self):
        """
        Resets the time state to zero.
        """
        self.steps.fill_(0)
        self.time.fill_(0.0)


###
# Kernels
###


@wp.kernel
def _advance_time(
    # Inputs
    dt: wp.array[wp.float32],
    # Outputs
    steps: wp.array[wp.int32],  # TODO: Make this wp.uint64
    time: wp.array[wp.float32],
):
    """
    Advances the time-keeping state of each world by one time-step.

    For each world index ``wid``, this kernel increments the step counter
    ``steps[wid]`` by 1 and increases the simulation time ``time[wid]``
    by the corresponding time increment ``dt[wid]``.
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
    """
    Advances the time-keeping state of each world by one time-step.

    For each world index ``wid``, this kernel increments the step counter
    ``steps[wid]`` by 1 and increases the simulation time ``time[wid]``
    by the corresponding time increment ``dt[wid]``.

    Args:
        model: The time model containing the time-step information.
        data: The time data containing the current time-keeping state.
    """
    # Ensure the model is valid
    if model is None:
        raise ValueError("'model' must be initialized, is None.")
    elif not isinstance(model, TimeModel):
        raise TypeError("'model' must be an instance of TimeModel.")
    if model.dt is None:
        raise ValueError("'model' must contain a `model.dt` array, is None.")

    # Ensure the state is valid
    if data is None:
        raise ValueError("'data' must be initialized, is None.")
    elif not isinstance(data, TimeData):
        raise TypeError("'data' must be an instance of TimeData.")
    if data.steps is None:
        raise ValueError("'data' must contain a `data.steps` array, is None.")

    # Launch the kernel to advance the time state of each world by one step
    wp.launch(
        _advance_time,
        dim=model.dt.size,
        inputs=[
            # Inputs:
            model.dt,
            # Outputs:
            data.steps,
            data.time,
        ],
        device=data.time.device,
    )
