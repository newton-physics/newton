# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
Provides utilities for generating random control
inputs for testing and benchmarking purposes.

See this link for relevant details:
https://nvidia.github.io/warp/user_guide/runtime.html#random-number-generation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import get_args

import numpy as np
import warp as wp

from ...core.control import ControlKamino
from ...core.joints import JointActuationType
from ...core.math import FLOAT32_MAX
from ...core.model import ModelKamino
from ...core.time import TimeData
from ...core.types import FloatArrayLike, IntArrayLike, float32, int32

###
# Module interface
###

__all__ = [
    "RandomJointController",
    "RandomJointControllerData",
]


###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Types
###


@dataclass
class RandomJointControllerData:
    """Data container for randomized control reference."""

    seed: int = 0
    """Seed for random number generation."""

    scale: wp.array | None = None
    """
    Scaling applied to randomly generated control inputs.

    Shape of `(sum_of_num_actuated_joint_dofs,)` and dtype of :class:`float32`.
    """

    decimation: wp.array | None = None
    """
    Control decimation of each world expressed as a multiple of simulation steps.

    Values greater than `1` result in a zero-order hold of the control
    inputs, meaning that they will change only every `decimation` steps.

    Shape of `(num_worlds,)` and dtype of :class:`int32`.
    """


###
# Kernels
###


@wp.kernel
def _generate_random_control_inputs(
    # Inputs
    controller_seed: int,
    controller_decimation: wp.array[int32],
    controller_scale: wp.array[float32],
    model_joints_wid: wp.array[int32],
    model_joints_act_type: wp.array[int32],
    model_joints_dofs_offset: wp.array[int32],
    model_joints_tau_j_max: wp.array[float32],
    state_time_steps: wp.array[int32],
    # Outputs
    # TODO: Add support for other control types
    # (e.g. position and velocity targets)
    control_tau_j: wp.array[float32],
):
    """
    A kernel to generate random control inputs for testing and benchmarking purposes.
    """
    # Retrieve the the joint index from the thread indices
    jid = wp.tid()

    # Retrieve the total number of joints from the size of the input arrays
    num_joints = model_joints_act_type.shape[0]

    # Retrieve the joint actuation type
    act_type = model_joints_act_type[jid]

    # Retrieve the world index from the thread indices
    wid = model_joints_wid[jid]

    # Retrieve the current simulation step
    step = state_time_steps[wid]

    # Retrieve the control decimation for the world
    decimation = controller_decimation[wid]

    # Only proceed for force actuated joints and at
    # simulation steps matching the control decimation
    if act_type == JointActuationType.PASSIVE or step % decimation != 0:
        return

    # Retrieve the number of DoFs and offset of the joint
    dofs_start = model_joints_dofs_offset[jid]
    num_dofs_j = model_joints_dofs_offset[jid + 1] - dofs_start

    # Iterate over the DoFs of the joint
    for dof in range(num_dofs_j):
        # Compute the DoF index in the global DoF vector
        joint_dof_index = dofs_start + dof

        # Retrieve the maximum limit of the generalized actuator forces
        tau_j_max = model_joints_tau_j_max[joint_dof_index]
        if tau_j_max == FLOAT32_MAX:
            tau_j_max = 1.0

        # Retrieve the scaling factor for the joint DoF
        scale_j = controller_scale[joint_dof_index]

        # Initialize a random number generator based on the
        # seed, current step, joint index, and DoF index
        rng_j_dof = wp.rand_init(controller_seed, (step + 1) * (num_joints * jid + dof))

        # Generate a random control input for the joint DoF
        tau_j_c = scale_j * wp.randf(rng_j_dof, -1.0, 1.0)

        # Clamp the control input to the maximum limits of the actuator
        tau_j_c = wp.clamp(tau_j_c, -tau_j_max, tau_j_max)

        # Store the updated integrator state and actuator control forces
        control_tau_j[joint_dof_index] = tau_j_c


###
# Interfaces
###


class RandomJointController:
    """
    Provides a simple interface for generating random
    control inputs for testing and benchmarking purposes.
    """

    def __init__(
        self,
        model: ModelKamino | None = None,
        decimation: int | IntArrayLike | None = None,
        scale: float | FloatArrayLike | None = None,
        seed: int | None = None,
    ):
        """
        Instantiates a new `RandomJointController` and allocates
        on-device data arrays if a model instance is provided.

        Args:
            model (`ModelKamino`, optional):
                The model container describing the system to be simulated.\n
                If `None`, a call to ``finalize()`` must be made later.
            decimation (`int` or `IntArrayLike`, optional):
                Control decimation for each world expressed as a multiple of simulation steps.\n
                Defaults to `1` for all worlds if `None`.
            scale (`float` or `FloatArrayLike`, optional):
                Scaling applied to randomly generated control inputs.\n
                Can be specified per-DoF as an array of shape `(sum_of_num_actuated_joint_dofs,)`
                and dtype of `float32`, or as a single float value applied uniformly across all DoFs.\n
                Defaults to `1.0` if `None`.
            seed (`int`, optional):
                Seed for random number generation. If `None`, it will default to `0`.
        """
        # Declare a local reference to the model
        # for which this controller is created
        self._model: ModelKamino | None = None

        # Declare the device cache
        self._device: wp.DeviceLike = None

        # Cache constructor arguments for potential later
        self._decimation: int | IntArrayLike | None = decimation
        self._scale: float | FloatArrayLike | None = scale
        self._seed: int = seed

        # Declare the internal controller data
        self._data: RandomJointControllerData | None = None

        # If a model is provided, allocate the controller data
        if model is not None:
            self.finalize(model=model, seed=seed, decimation=decimation, scale=scale)

    ###
    # Properties
    ###

    @property
    def device(self) -> wp.DeviceLike:
        """The device used for allocations and execution."""
        if self._data is None:
            raise RuntimeError("Controller data is not allocated. Call finalize() first.")
        return self._data.decimation.device

    @property
    def seed(self) -> int:
        """The seed used for random number generation."""
        if self._data is None:
            raise RuntimeError("Controller data is not allocated. Call finalize() first.")
        return self._data.seed

    @seed.setter
    def seed(self, s: int):
        """Sets the seed used for random number generation."""
        if self._data is None:
            raise RuntimeError("Controller data is not allocated. Call finalize() first.")
        self._data.seed = s

    @property
    def model(self) -> ModelKamino:
        """The model for which this controller is created."""
        if self._model is None:
            raise RuntimeError("Controller is not finalized with a model. Call finalize() first.")
        return self._model

    @property
    def data(self) -> RandomJointControllerData:
        """The internal controller data."""
        if self._data is None:
            raise RuntimeError("Controller data is not allocated. Call finalize() first.")
        return self._data

    ###
    # Operations
    ###

    def finalize(
        self,
        model: ModelKamino,
        seed: int | None = None,
        decimation: int | IntArrayLike | None = None,
        scale: float | FloatArrayLike | None = None,
    ):
        """
        Finalizes the random controller by allocating
        on-device data arrays based on the provided model.

        Args:
            model (`ModelKamino`):
                The model container describing the system to be simulated.
            decimation (`int` or `IntArrayLike`, optional):
                Control decimation for each world expressed as a multiple of simulation steps.\n
                Defaults to `1` for all worlds if `None`.
            scale (`float` or `FloatArrayLike`, optional):
                Scaling applied to randomly generated control inputs.\n
                Can be specified per-DoF as an array of shape `(sum_of_num_actuated_joint_dofs,)`
                and dtype of `float32`, or as a single float value applied uniformly across all DoFs.\n
                Defaults to `1.0` if `None`.
            seed (`int`, optional):
                Seed for random number generation. If `None`, it will default to `0`.

        Raises:
            ValueError: If the model has no actuated DoFs.
            ValueError: If the length of the decimation array does not match the number of worlds.
        """
        # Ensure the model is valid and assign it to the controller
        if model is None:
            raise ValueError("ModelKamino must be provided to finalize the controller.")
        elif not isinstance(model, ModelKamino):
            raise ValueError(f"Expected model to be of type ModelKamino, but got {type(model)}.")

        # Cache the model reference for use in the compute function
        self._model = model

        # Check that the model has joint DoFs
        num_joint_dofs = model.size.sum_of_num_joint_dofs
        if num_joint_dofs == 0:
            raise ValueError("The provided model has no joint DoFs to generate control inputs for.")

        # Validate and process the constructor arguments
        self._decimation, self._scale, self._seed = self._validate_arguments(
            num_worlds=model.size.num_worlds,
            num_joint_dofs=num_joint_dofs,
            decimation=decimation if decimation is not None else self._decimation,
            scale=scale if scale is not None else self._scale,
            seed=seed if seed is not None else self._seed,
        )

        # Use the model's device
        self._device = model.device

        # Allocate the controller data
        with wp.ScopedDevice(self._device):
            self._data = RandomJointControllerData(
                seed=self._seed,
                decimation=wp.array(self._decimation, dtype=int32),
                scale=wp.array(self._scale, dtype=float32),
            )

    def compute(self, time: TimeData, control: ControlKamino):
        """
        Generate randomized generalized control forces to apply to the system.

        Each random values is generated based on the seed, current simulation step,
        joint index, and local DoF index to ensure reproducibility across runs.

        Args:
            model (ModelKamino):
                The input model container holding the time-invariant parameters of the simulation.
            time (TimeData):
                The input time data container holding the current simulation time and steps.
            control (ControlKamino):
                The output control container where the computed control torques will be stored.
        """
        # Ensure a model has been assigned and finalized
        if self._model is None or self._data is None:
            raise RuntimeError("Controller is not finalized with a model. Call finalize() first.")

        # Launch the kernel to compute the random control inputs
        wp.launch(
            _generate_random_control_inputs,
            dim=self._model.size.sum_of_num_joints,
            inputs=[
                # Inputs
                self._data.seed,
                self._data.decimation,
                self._data.scale,
                self._model.joints.wid,
                self._model.joints.act_type,
                self._model.joints.dofs_offset,
                self._model.joints.tau_j_max,
                time.steps,
                # Outputs
                # TODO: Add support for other control types
                # (e.g. position and velocity targets)
                control.tau_j,
            ],
            device=self._device,
        )

    ###
    # Internals
    ###

    def _validate_arguments(
        self,
        num_worlds: int,
        num_joint_dofs: int,
        decimation: int | IntArrayLike | None,
        scale: float | FloatArrayLike | None,
        seed: int | None,
    ):
        # Check if the decimation argument is specified, and validate it accordingly
        if decimation is not None:
            if isinstance(decimation, int):
                _decimation = np.full(num_worlds, decimation, dtype=np.int32)
            elif isinstance(decimation, get_args(IntArrayLike)):
                decsize = len(decimation)
                if decsize != num_worlds:
                    raise ValueError(f"Expected decimation `IntArrayLike` of length {num_worlds}, but has {decsize}.")
                _decimation = np.array(decimation, dtype=np.int32)
            else:
                raise ValueError(f"Expected decimation of type `int` or `IntArrayLike`, but got {type(decimation)}.")
        # Otherwise, set it to the default value of 1 for all worlds
        else:
            _decimation = np.ones(num_worlds, dtype=np.int32)

        # Check if the scale argument is specified, and validate it accordingly
        if scale is not None:
            if isinstance(scale, (int, float)):
                _scale = np.full(num_joint_dofs, float(scale), dtype=np.float32)
            elif isinstance(scale, get_args(FloatArrayLike)):
                if len(scale) != num_joint_dofs:
                    raise ValueError(
                        f"Expected scale `FloatArrayLike` of length {num_joint_dofs}, but has {len(scale)}"
                    )
                _scale = np.array(scale, dtype=np.float32)
            else:
                raise ValueError(f"Expected scale of type `float` or `FloatArrayLike`, but got {type(scale)}.")
        # Otherwise, set it to the default value of 1.0 for all DoFs
        else:
            _scale = np.full(num_joint_dofs, 1.0, dtype=np.float32)

        # Check if the seed argument is specified, and set it accordingly
        if seed is not None:
            if not isinstance(seed, int):
                raise ValueError(f"Expected seed of type `int`, but got {type(seed)}.")
            _seed = int(seed)
        else:
            _seed = int(0)

        # Return the validated and processed arguments
        return _decimation, _scale, _seed
