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

"""Containers and interfaces for animation reference tracking."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp
from scipy.interpolate import interp1d
from warp.context import Devicelike

from ..core.model import Model
from ..core.time import TimeData
from ..core.types import float32, int32

###
# Module interface
###


__all__ = [
    "AnimationJointReference",
    "AnimationJointReferenceData",
]


###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Types
###


@dataclass
class AnimationJointReferenceData:
    """
    Container of animation references for actuated joints.

    By default, the animation reference is allocated such that all worlds share
    the same reference data, but can progress and/or loop independently.

    The animation references are organized as 2D arrays where each column corresponds to
    an actuated joint DoF and each row corresponds to a frame in the animation sequence.

    Progression through the animation sequence is controlled via the ``frame`` attribute,
    which indicates the current frame index that is active for each world, from which
    actuator joint coordinates and velocities are extracted. In addition, to control the
    progression along the reference sequence, each world has its own ``rate`` and
    ``decimation`` attributes. The ``rate`` attribute (defaults to 1) determines the number
    of frames by which to advance the active frame at each step, while the ``decimation``
    attribute determines how many steps to wait until the frame index should be updated.
    These attributes effectively allow for both slowing down and speeding up the
    animation's progression relative to the simulation time-step. Finally, the ``loop``
    attribute specifies whether the animation should restart from the beginning after
    reaching the end, or stop at the last frame.

    Attributes:
        num_actuated_joint_dofs: wp.array
            The number of actuated joint DoFs per world.
        actuated_joint_dofs_offset: wp.array
            The offset indices for the actuated joint DoFs per world.
        q_j_ref: wp.array
            The reference actuator joint positions.
        dq_j_ref: wp.array
            The reference actuator joint velocities.
        loop: wp.array
            Flag indicating whether the animation should loop.
        rate: wp.array
            The rate at which to progress the animation sequence.
        decimation: wp.array
            The decimation factor for extracting references from the animation sequence.
        length: wp.array
            The length of the animation sequence.
        frame: wp.array
            The current frame index in the animation sequence that is active.
    """

    num_actuated_joint_dofs: wp.array | None = None
    """
    Number of actuated joint DoFs per world.\n
    Shape is ``(num_worlds,)`` and dtype is :class:`int32`.
    """

    actuated_joint_dofs_offset: wp.array | None = None
    """
    Offset indices for the actuated joint DoFs per world.\n
    Shape is ``(num_worlds,)`` and dtype is :class:`int32`.
    """

    q_j_ref: wp.array | None = None
    """
    Sequence of reference joint actuator positions.\n
    Shape is ``(max_of_num_actuated_joint_coords, sequence_length)`` and dtype is :class:`float32`.
    """

    dq_j_ref: wp.array | None = None
    """
    Sequence of reference joint actuator velocities.\n
    Shape is ``(max_of_num_actuated_joint_dofs, sequence_length)`` and dtype is :class:`float32`.
    """

    length: wp.array | None = None
    """
    Integer length of the animation sequence.\n
    Shape is ``(num_worlds,)`` and dtype is :class:`int32`.
    """

    decimation: wp.array | None = None
    """
    Integer decimation factor by which references are extracted from the animation sequence.\n
    Shape is ``(num_worlds,)`` and dtype is :class:`int32`.
    """

    rate: wp.array | None = None
    """
    Integer rate by which to progress the active frame of the animation sequence at each step.\n
    Shape is ``(num_worlds,)`` and dtype is :class:`int32`.
    """

    loop: wp.array | None = None
    """
    Integer flag to indicate if the animation should loop.\n
    Shape is ``(num_worlds,)`` and dtype is :class:`int32`.\n
    If `1`, the animation will restart from the beginning after reaching the end.\n
    If `0`, the animation will stop at the last frame.
    """

    frame: wp.array | None = None
    """
    Integer index indicating the active frame of the animation sequence.\n
    Shape is ``(num_worlds,)`` and dtype is :class:`int32`.
    """


###
# Kernels
###


@wp.kernel
def _advance_animation_frame(
    # Inputs
    time_steps: wp.array(dtype=int32),
    animation_length: wp.array(dtype=int32),
    animation_decimation: wp.array(dtype=int32),
    animation_rate: wp.array(dtype=int32),
    animation_loop: wp.array(dtype=int32),
    # Outputs
    animation_frame: wp.array(dtype=int32),
):
    """
    A kernel to compute joint-space PID control torques for force-actuated joints.
    """
    # Retrieve the the world index from the thread indices
    wid = wp.tid()

    # Retrieve the animation sequence info
    length = animation_length[wid]
    decimation = animation_decimation[wid]
    rate = animation_rate[wid]
    loop = animation_loop[wid]

    # Retrieve the current step (i.e. discrete-time index) for this world
    step = time_steps[wid]

    # Check if we need to advance the animation frame
    if step % decimation != 0:
        return

    # Retrieve the current frame index for this world
    frame = animation_frame[wid]

    # Advance the frame index
    frame += rate

    # If looping is enabled, wrap the frame index around
    if loop:
        frame %= length
    # Otherwise, clamp the frame index to the last frame
    else:
        if frame >= length:
            frame = length - 1

    # Update the active reference arrays
    animation_frame[wid] = frame


# TODO: Make the 2D arrays as flattened 1D arrays to handle arbitrary layouts
@wp.kernel
def _extract_animation_references(
    # Inputs
    num_actuated_joint_dofs: wp.array(dtype=int32),
    actuated_joint_dofs_offset: wp.array(dtype=int32),
    animation_frame: wp.array(dtype=int32),
    animation_q_j_ref: wp.array2d(dtype=float32),
    animation_dq_j_ref: wp.array2d(dtype=float32),
    # Outputs
    q_j_ref_active: wp.array(dtype=float32),
    dq_j_ref_active: wp.array(dtype=float32),
):
    """
    A kernel to compute joint-space PID control torques for force-actuated joints.
    """
    # Retrieve the the world and DoF index from the thread indices
    wid, qid = wp.tid()

    # Retrieve the number of actuated DoFs and offset for this world
    num_ajq = num_actuated_joint_dofs[wid]
    ajq_offset = actuated_joint_dofs_offset[wid]

    # Ensure we are within the valid range of actuated DoFs for this world
    if qid >= num_ajq:
        return

    # Retrieve the current step index for this world
    frame = animation_frame[wid]

    # Compute the global DoF index
    actuator_dof_index = ajq_offset + qid

    # Update the active reference arrays
    q_j_ref_active[actuator_dof_index] = animation_q_j_ref[frame, qid]
    dq_j_ref_active[actuator_dof_index] = animation_dq_j_ref[frame, qid]


###
# Interfaces
###


class AnimationJointReference:
    """
    A module for managing and operating joint-space references from an animation.
    """

    def __init__(
        self,
        model: Model | None = None,
        data: np.ndarray | None = None,
        data_dt: float | None = None,
        target_dt: float | None = None,
        decimation: int | list[int] = 1,
        rate: int | list[int] = 1,
        loop: bool | list[bool] = True,
        use_fd: bool = False,
        device: Devicelike = None,
    ):
        """
        Initialize the animation joint reference interface.

        Args:
            model (Model | None): The model container used to determine the required allocation sizes.
                If None, calling ``allocate()`` later can be used for deferred allocation.
            data (np.ndarray | None): The input animation reference data as a 2D numpy array.
                If None, calling ``allocate()`` later can be used for deferred allocation.
            data_dt (float | None): The time-step between frames in the input data.
            target_dt (float | None): The desired time-step between frames in the animation reference.
                If None, defaults to ``data_dt``.
            decimation (int | list[int]): Decimation factor(s) defining the rate at which the animation
                frame index is updated w.r.t the simulation step. If a list of integers, then frame
                progression can proceed independently in each world. Defaults to 1 for all worlds.
            rate (int | list[int]): Rate(s) by which to progress the animation frame index each time
                the simulation step matches the set decimation. Defaults to 1 for all worlds.
            loop (bool | list[bool]): Flag(s) indicating whether the animation should loop.
            use_fd (bool): Whether to compute finite-difference velocities from the input coordinates.
            device (Devicelike | None): Device to use for allocations and execution.
        """

        # Cache the device
        self._device: Devicelike = device

        # Declare the model dimensions meta-data
        self._num_worlds: int = 0
        self._max_of_num_actuated_dofs: int = 0
        self._sequence_length: int = 0

        # Declare the internal controller data
        self._data: AnimationJointReferenceData | None = None

        # If a model is provided, allocate the controller data
        if model is not None:
            self.allocate(
                model=model,
                data=data,
                data_dt=data_dt,
                target_dt=target_dt,
                decimation=decimation,
                rate=rate,
                loop=loop,
                use_fd=use_fd,
                device=device,
            )

    ###
    # Properties
    ###

    @property
    def device(self) -> Devicelike | None:
        """The device used for allocations and execution."""
        return self._device

    @property
    def sequence_length(self) -> int:
        """The length of the animation sequence."""
        return self._sequence_length

    @property
    def data(self) -> AnimationJointReferenceData:
        """The internal animation reference data."""
        self._assert_has_data()
        return self._data

    ###
    # Internals
    ###

    def _assert_has_data(self) -> None:
        """Check if the internal animation data has been allocated."""
        if self._data is None:
            raise ValueError("Animation reference data is not allocated. Call allocate() first.")

    @staticmethod
    def _upsample_reference_coordinates(
        q_ref: np.ndarray,
        dt_in: float,
        dt_out: float,
        t0: float = 0.0,
        t_start: float | None = None,
        t_end: float | None = None,
        extrapolate: bool = False,
    ) -> np.ndarray:
        """
        Upsample the given reference joint coordinates from the input time-step to the output time-step.

        Args:
            q_ref (np.ndarray): Reference joint positions of shape (sequence_length, num_actuated_dofs).
            dt_in (float): Input time step between frames.
            dt_out (float): Output time step between frames.
            t0 (float): Initial time corresponding to the first frame.
            t_start (float | None): Start time for the up-sampled reference. If None, uses t0.
            t_end (float | None): End time for the up-sampled reference. If None, uses the last input frame time.
            extrapolate (bool): Whether to allow extrapolation beyond the input time range.

        Returns:
            np.ndarray: Up-sampled reference joint positions of shape (new_sequence_length, num_actuated_dofs).
        """

        # Extract the number of samples
        num_samples, _ = q_ref.shape
        if t_start is None:
            t_start = t0
        if t_end is None:
            t_end = t0 + (num_samples - 1) * dt_in

        # Construct time-sample sequences for the original and new references
        t_original = t0 + dt_in * np.arange(num_samples)
        num_samples_new = int(np.floor((t_end - t_start) / dt_out)) + 1
        t_new = t_start + dt_out * np.arange(num_samples_new)

        # Create the up-sampling interpolation function
        upsample_func = interp1d(
            t_original,
            q_ref,
            axis=0,
            kind="linear",
            bounds_error=False,
            fill_value=("extrapolate" if extrapolate else (q_ref[0], q_ref[-1])),
        )

        # Evaluate the up-sampling function at the new time samples
        # to compute the up-sampled joint coordinate references
        return upsample_func(t_new)

    @staticmethod
    def _compute_finite_difference_velocities(q_ref: np.ndarray, dt: float) -> np.ndarray:
        """
        Compute finite-difference velocities for the given reference positions.

        Args:
            q_ref (np.ndarray): Reference joint positions of shape (sequence_length, num_actuated_dofs).
            dt (float): Time step between frames.

        Returns:
            np.ndarray: Reference joint velocities of shape (sequence_length, num_actuated_dofs).
        """
        # TODO: Try this instead (it might be more robust):
        # _compute_finite_difference_velocities = staticmethod(lambda q_ref_np, dt: np.gradient(q_ref_np, dt, axis=0))

        # First allocate and initialize the output array
        dq_j_ref = np.zeros_like(q_ref)

        # Guard against single-frame animations (constant set-points)
        num_samples = q_ref.shape[0]
        if num_samples < 2:
            return dq_j_ref

        # Compute forward finite-difference velocities for the reference positions
        dq_j_ref[1:] = np.diff(q_ref, axis=0) / dt

        # Set the first velocity to match the second
        dq_j_ref[0] = dq_j_ref[1]

        # Apply a simple moving average filter to smooth out the velocities
        kernel_size = 5
        kernel = np.ones(kernel_size) / kernel_size
        for i in range(q_ref.shape[1]):
            dq_j_ref[:, i] = np.convolve(dq_j_ref[:, i], kernel, mode="same")

        # Return the computed reference joint velocities
        return dq_j_ref

    ###
    # Operations
    ###

    def allocate(
        self,
        model: Model,
        data: np.ndarray,
        data_dt: float,
        target_dt: float | None = None,
        decimation: int | list[int] = 1,
        rate: int | list[int] = 1,
        loop: bool | list[bool] = True,
        use_fd: bool = False,
        device: Devicelike = None,
    ) -> None:
        """
        Allocate the animation joint reference data.

        Args:
            model (Model): The model container used to determine the required allocation sizes.
            data (np.ndarray): The input animation reference data as a 2D numpy array.
            data_dt (float): The time-step between frames in the input data.
            target_dt (float | None): The desired time-step between frames in the animation reference.
                If None, defaults to ``data_dt``.
            decimation (int | list[int]): Decimation factor(s) defining the rate at which the animation
                frame index is updated w.r.t the simulation step. If a list of integers, then frame
                progression can proceed independently in each world. Defaults to 1 for all worlds.
            rate (int | list[int]): Rate(s) by which to progress the animation frame index each time
                the simulation step matches the set decimation. Defaults to 1 for all worlds.
            loop (bool | list[bool]): Flag(s) indicating whether the animation should loop.
            use_fd (bool): Whether to compute finite-difference velocities from the input coordinates.
            device (Devicelike | None): Device to use for allocations and execution.

        Raises:
            ValueError: If the model is not valid or actuated DoFs are not properly configured.
            ValueError: If the input data is not a valid 2D numpy array.

        Note:
            The model must have only 1-DoF actuated joints for this controller to be compatible.
        """
        # Ensure the model is valid
        if model is None or model.size is None:
            raise ValueError("Model is not valid. Cannot allocate controller data.")

        # Retrieve the shape of the input data
        if data is None:
            raise ValueError("Input data must be provided for allocation.")

        # Ensure input array is valid
        if not isinstance(data, np.ndarray):
            raise ValueError("Input data must be a numpy array.")
        if data.ndim != 2:
            raise ValueError("Input data must be a 2D numpy array.")

        # Get the number of actuated coordinates and DoFs
        total_num_actuated_coords = model.size.sum_of_num_actuated_joint_coords
        total_num_actuated_dofs = model.size.sum_of_num_actuated_joint_dofs
        max_num_actuated_dofs = model.size.max_of_num_actuated_joint_dofs

        # Check if there are any actuated DoFs
        if total_num_actuated_dofs == 0:
            raise ValueError("Model has no actuated DoFs.")

        # Ensure the model has only 1-DoF actuated joints
        if total_num_actuated_coords != total_num_actuated_dofs:
            raise ValueError(
                f"Model has {total_num_actuated_coords} actuated coordinates but {total_num_actuated_dofs} actuated "
                "DoFs. AnimationJointReference is currently incompatible with multi-DoF actuated joints."
            )

        # Check that input data matches the number of actuated DoFs
        if data.shape[1] != max_num_actuated_dofs and data.shape[0] != max_num_actuated_dofs:
            raise ValueError(
                f"Input data has shape {data.shape} which does not match the "
                f"per-world number of actuated DoFs ({max_num_actuated_dofs})."
            )

        # We assume the input is organized as (sequence_length, num_actuated_dofs)
        # Transpose the input if necessary in order to match the assumed shape
        if data.shape[0] == max_num_actuated_dofs or data.shape[0] == 2 * max_num_actuated_dofs:
            data = data.T

        # Ensure the target time-step is valid
        if data_dt <= 0.0:
            raise ValueError("Target time-step must be positive.")

        # Check the target time-step input and set it to the animation dt if not provided
        if target_dt is None:
            target_dt = data_dt

        # Ensure decimation, rate, and loop are lists matching the number of worlds
        if isinstance(decimation, int):
            decimation = [decimation] * model.size.num_worlds
        if isinstance(rate, int):
            rate = [rate] * model.size.num_worlds
        if isinstance(loop, bool):
            loop = [loop] * model.size.num_worlds

        # Optionally upsample the input data with linearly-interpolation to match the target time-step
        if target_dt < data_dt:
            data = self._upsample_reference_coordinates(
                q_ref=data,
                dt_in=data_dt,
                dt_out=target_dt,
                extrapolate=False,
            )

        # Cache the model dimensions meta-data
        self._num_worlds = model.size.num_worlds
        self._max_of_num_actuated_dofs = max_num_actuated_dofs
        self._sequence_length = data.shape[0]

        # Extract the reference joint positions and velocities
        q_j_ref_np = data[:, :max_num_actuated_dofs].astype(np.float32)
        if data.shape[1] >= 2 * max_num_actuated_dofs:
            dq_j_ref_np = data[:, max_num_actuated_dofs : 2 * max_num_actuated_dofs].astype(np.float32)
        else:
            # Optionally use finite-differences to estimate velocities if requested
            if use_fd:
                dq_j_ref_np = self._compute_finite_difference_velocities(q_j_ref_np, target_dt)
            # Otherwise, default to zero velocities
            else:
                dq_j_ref_np = np.zeros_like(q_j_ref_np)

        # Create the rate and loop arrays
        length_np = np.array([q_j_ref_np.shape[0]] * self._num_worlds, dtype=np.int32)
        decimation_np = np.array(decimation, dtype=np.int32)
        rate_np = np.array(rate, dtype=np.int32)
        loop_np = np.array([1 if _l else 0 for _l in loop], dtype=np.int32)

        # Override the device if provided
        if device is not None:
            self._device = device

        # Allocate the controller data
        with wp.ScopedDevice(self._device):
            self._data = AnimationJointReferenceData(
                num_actuated_joint_dofs=model.info.num_actuated_joint_dofs,
                actuated_joint_dofs_offset=model.info.joint_actuated_dofs_offset,
                q_j_ref=wp.array(q_j_ref_np, dtype=float32),
                dq_j_ref=wp.array(dq_j_ref_np, dtype=float32),
                length=wp.array(length_np, dtype=int32),
                decimation=wp.array(decimation_np, dtype=int32),
                rate=wp.array(rate_np, dtype=int32),
                loop=wp.array(loop_np, dtype=int32),
                frame=wp.zeros(self._num_worlds, dtype=int32),
            )

    def plot(self, path: str | None = None, show: bool = False) -> None:
        from matplotlib import pyplot as plt  # noqa: PLC0415

        # Extract numpy arrays for plotting
        q_j_ref_np = self._data.q_j_ref.numpy()
        dq_j_ref_np = self._data.dq_j_ref.numpy()

        # Plot the input data for verification
        _, axs = plt.subplots(2, 1, figsize=(10, 6))
        for i in range(self._max_of_num_actuated_dofs):
            axs[0].plot(q_j_ref_np[:, i], label=f"Joint {i}")
            axs[1].plot(dq_j_ref_np[:, i], label=f"Joint {i}")
        axs[0].set_title("Reference Joint Positions")
        axs[0].set_xlabel("Frame")
        axs[0].set_ylabel("Position")
        axs[0].legend()
        axs[1].set_title("Reference Joint Velocities")
        axs[1].set_xlabel("Frame")
        axs[1].set_ylabel("Velocity")
        axs[1].legend()
        plt.tight_layout()

        # Save the figure if a path is provided
        if path is not None:
            plt.savefig(path, dpi=300)

        # Show the figure if requested
        # NOTE: This will block execution until the plot window is closed
        if show:
            plt.show()

        # Close the plot to free up resources
        plt.close()

    def loop(self, enabled: bool | list[bool] = True) -> None:
        """
        Enable or disable looping of the animation sequence.

        Args:
            enabled (bool | list[bool]): If True, enable looping. If False, disable looping.
        """
        # Ensure the animation data container is allocated
        if self._data is None:
            raise ValueError("Controller data is not allocated. Call allocate() first.")

        # Check if a single value or list is provided and set the loop flags accordingly
        if isinstance(enabled, list):
            if len(enabled) != self._num_worlds:
                raise ValueError("Length of 'enabled' list must match the number of worlds.")
            enabled_array = np.array([1 if e else 0 for e in enabled], dtype=np.int32)
            self._data.loop.assign(enabled_array)
        else:
            self._data.loop = wp.array([1 if enabled else 0] * self._num_worlds, dtype=int32)

    def advance(self, time: TimeData) -> None:
        """
        Advances the animation sequence frame index according to the configured
        decimation and rate, in accordance with the current simulation time-step.

        Args:
            time (TimeData): The time data container holding the current simulation time.
        """
        self._assert_has_data()
        wp.launch(
            _advance_animation_frame,
            dim=self._num_worlds,
            inputs=[
                # Inputs:
                time.steps,
                self._data.length,
                self._data.decimation,
                self._data.rate,
                self._data.loop,
                # Outputs:
                self._data.frame,
            ],
            device=self._device,
        )

    def extract(self, q_j_ref_out: wp.array, dq_j_ref_out: wp.array) -> None:
        """
        Extract the reference arrays from the animation sequence at the current frame index.

        Args:
            q_j_ref_out (wp.array): Output array for the reference joint positions.
            dq_j_ref_out (wp.array): Output array for the reference joint velocities.
        """
        self._assert_has_data()
        wp.launch(
            _extract_animation_references,
            dim=(self._num_worlds, self._max_of_num_actuated_dofs),
            inputs=[
                # Inputs:
                self._data.num_actuated_joint_dofs,
                self._data.actuated_joint_dofs_offset,
                self._data.frame,
                self._data.q_j_ref,
                self._data.dq_j_ref,
                # Outputs:
                q_j_ref_out,
                dq_j_ref_out,
            ],
            device=self._device,
        )

    def reset(self, q_j_ref_out: wp.array, dq_j_ref_out: wp.array) -> None:
        """
        Reset the active frame index of the animation sequence to zero
        and sets the extracts the initial references into the output arrays.

        Args:
            q_j_ref_out (wp.array): Output array for the reference joint positions.
            dq_j_ref_out (wp.array): Output array for the reference joint velocities.
        """
        self._assert_has_data()
        self._data.frame.fill_(0)
        self.extract(q_j_ref_out, dq_j_ref_out)

    def step(self, time: TimeData, q_j_ref_out: wp.array, dq_j_ref_out: wp.array) -> None:
        """
        Advances the animation sequence by the configured decimation and
        rate, and extracts the reference arrays at the active frame index.

        This is a convenience method that effectively combines
        ``advance()`` and ``extract()`` into a single operation.

        Args:
            time (TimeData): The time data container holding the current simulation time.
            q_j_ref_out (wp.array): Output array for the reference joint positions.
            dq_j_ref_out (wp.array): Output array for the reference joint velocities.
        """
        self.advance(time)
        self.extract(q_j_ref_out, dq_j_ref_out)
