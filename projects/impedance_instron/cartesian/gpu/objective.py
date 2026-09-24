# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Evaluate recorded hip, joint, and native force errors without host transfers."""

from __future__ import annotations

import numpy as np
import warp as wp

from ..fit import FitConfig, _mapping, _Objective, _weights
from .mechanics import Vec5

wp.set_module_options({"enable_backward": False, "fuse_fp": False})


@wp.kernel
def _evaluate(
    states: wp.array2d[Vec5],
    forces: wp.array2d[wp.vec2d],
    integrated_steps: wp.array[int],
    failure_code: wp.array[int],
    steps: int,
    motion_count: int,
    force_count: int,
    lower: wp.array[int],
    upper: wp.array[int],
    fraction: wp.array[wp.float64],
    targets: wp.array[wp.vec2d],
    weights: wp.array[wp.float64],
    residual_weights: wp.array[wp.float64],
    scales: wp.array[wp.float64],
    ground_angle: int,
    ground_offset: wp.float64,
    loss: wp.array[wp.float64],
    rmse: wp.array2d[wp.float64],
    maximum_error: wp.array2d[wp.float64],
    costs: wp.array2d[wp.float64],
    residual: wp.array2d[wp.float64],
):
    """Reduce each complete world in fixed sample and channel order."""
    world = wp.tid()
    complete = integrated_steps[world] == steps and failure_code[world] == 0
    # Match the CPU completion gate, including samples outside interpolation maps
    # and the unactuated thigh coordinate. No partial trajectory can earn a score.
    if complete:
        for step in range(steps + 1):
            state = states[step, world]
            for channel in range(5):
                if not wp.isfinite(state[channel]):
                    complete = False
        for step in range(steps):
            force = forces[step, world]
            if not wp.isfinite(force[0]) or not wp.isfinite(force[1]):
                complete = False

    total = wp.float64(0.0)
    if complete:
        for block in range(3):
            sample_count = motion_count
            if block == 2:
                sample_count = force_count
            offset = block * motion_count
            cost = wp.float64(0.0)
            squared_error = wp.vec2d(wp.float64(0.0))
            maximum = wp.vec2d(wp.float64(0.0))
            for sample in range(sample_count):
                index = offset + sample
                lo = lower[index]
                hi = upper[index]
                alpha = fraction[index]
                predicted = wp.vec2d(wp.float64(0.0))
                if block == 2:
                    predicted = forces[lo, world] * (wp.float64(1.0) - alpha) + forces[hi, world] * alpha
                else:
                    state_lo = states[lo, world]
                    state_hi = states[hi, world]
                    state_channel = int(0)
                    if block == 1:
                        state_channel = 3
                    for channel in range(2):
                        coordinate = state_channel + channel
                        predicted[channel] = (
                            state_lo[coordinate] * (wp.float64(1.0) - alpha) + state_hi[coordinate] * alpha
                        )
                    if block == 1 and ground_angle != 0:
                        lo_pitch = ((state_lo[2] + state_lo[3]) + state_lo[4]) + ground_offset
                        hi_pitch = ((state_hi[2] + state_hi[3]) + state_hi[4]) + ground_offset
                        predicted[1] = lo_pitch * (wp.float64(1.0) - alpha) + hi_pitch * alpha
                error = predicted - targets[index]
                for channel in range(2):
                    value = error[channel] / scales[block] * residual_weights[index]
                    residual[2 * index + channel, world] = value
                    cost += value * value
                    squared_error[channel] += weights[index] * (error[channel] * error[channel])
                    maximum[channel] = wp.max(maximum[channel], wp.abs(error[channel]))
                    if not wp.isfinite(value):
                        complete = False
            costs[world, block] = cost
            total += cost
            for channel in range(2):
                rmse[world, 2 * block + channel] = wp.sqrt(squared_error[channel])
                maximum_error[world, 2 * block + channel] = maximum[channel]
        if not wp.isfinite(total):
            complete = False

    if complete:
        loss[world] = total
    else:
        # Overwrite all diagnostics so repeated batches cannot retain a valid
        # previous residual or expose a truncated candidate to a GPU optimizer.
        infinity = wp.float64(wp.inf)
        loss[world] = infinity
        for channel in range(6):
            rmse[world, channel] = infinity
            maximum_error[world, channel] = infinity
        for block in range(3):
            costs[world, block] = infinity
        for row in range(4 * motion_count + 2 * force_count):
            residual[row, world] = infinity


class MeasuredObjective:
    """Keep native-grid measured residuals and reductions on the device.

    Motion uses the full state clock, including its terminal sample. Forces
    use only the preintegration clock. The residual has the CPU objective's
    flattened order: hip sample pairs, knee/ankle sample pairs, then native
    force sample pairs. Each block has two equally weighted channels and
    normalized trapezoidal weights on its own recorded grid.

    Args:
        reference: Recorded Cartesian reference arrays. Only ``hip_target_m``,
            ``joint_target_rad``, and ``grf_target_n`` enter the residual.
        settings: Measured tolerances; hip [m], joint [rad], and force [N].
        time_s: Full simulation clock [s], including the terminal sample,
            produced by ``np.linspace(0, duration, steps + 1)``.
        world_count: Number of independent candidates in a batch.
        device: Warp device for persistent inputs and outputs.
    """

    def __init__(self, reference: dict, settings: FitConfig, time_s: np.ndarray, world_count: int, device):
        time = np.array(time_s, dtype=np.float64, copy=True)
        if time.ndim != 1 or len(time) < 2 or not np.isfinite(time).all() or np.any(np.diff(time) <= 0):
            raise ValueError("time_s must contain at least two finite, strictly increasing samples")
        if isinstance(world_count, bool) or not isinstance(world_count, int) or world_count < 1:
            raise ValueError("world_count must be a positive integer")
        duration = float(reference["time_s"][-1])
        if time[0] != 0.0 or not np.isclose(time[-1], duration, rtol=0, atol=1e-12):
            raise ValueError("Simulation time_s must cover the entire reference duration from zero")
        motion_time = np.asarray(reference["time_s"])
        native_time = np.asarray(reference["grf_time_s"])
        force_mask = (native_time >= time[0]) & (native_time <= time[-2])
        if not np.any(force_mask):
            raise ValueError("No native GRF samples lie within simulated force support; reduce dt_s")
        motion_map = _mapping(time, motion_time)
        force_map = _mapping(time[:-1], native_time[force_mask])
        motion_weights = _weights(motion_time)
        force_weights = _weights(native_time[force_mask])
        maps = (motion_map, motion_map, force_map)
        weights = np.concatenate((motion_weights, motion_weights, force_weights))
        angle_targets = np.array(reference["joint_target_rad"], copy=True)
        self.ground_angle = int("foot_ground_target_rad" in reference)
        self.ground_offset = np.pi / 2 - float(reference.get("shoe_static_pitch_rad", 0.0))
        if self.ground_angle:
            angle_targets[:, 1] = reference["foot_ground_target_rad"]
        targets = np.concatenate(
            (
                reference["hip_target_m"],
                angle_targets,
                np.asarray(reference["grf_target_n"])[force_mask],
            )
        )
        if targets.shape != (len(weights), 2) or not np.isfinite(targets).all():
            raise ValueError("Measured targets must contain finite sample pairs on their recorded grids")
        self.device = wp.get_device(device)
        self.world_count = world_count
        self.steps = len(time) - 1
        self.time_s = time
        self.motion_sample_count = len(motion_time)
        self.force_sample_count = int(force_mask.sum())
        self.residual_dim = 4 * self.motion_sample_count + 2 * self.force_sample_count
        self.force_mask = force_mask.copy()
        self.description = _Objective(reference, settings).description
        self.description.update(
            motion_sample_count=self.motion_sample_count,
            force_sample_count=self.force_sample_count,
            force_interval_s=[float(native_time[force_mask][0]), float(native_time[force_mask][-1])],
            native_force_samples_outside_simulated_support=int(np.count_nonzero(~force_mask)),
        )
        self._lower = wp.array(np.concatenate([item[0] for item in maps]), dtype=wp.int32, device=self.device)
        self._upper = wp.array(np.concatenate([item[1] for item in maps]), dtype=wp.int32, device=self.device)
        self._fraction = wp.array(
            np.concatenate([item[2].ravel() for item in maps]), dtype=wp.float64, device=self.device
        )
        self._targets = wp.array(targets, dtype=wp.vec2d, device=self.device)
        self._weights = wp.array(weights, dtype=wp.float64, device=self.device)
        self._residual_weights = wp.array(np.sqrt(weights / 2), dtype=wp.float64, device=self.device)
        self._scales = wp.array(
            [settings.hip_tolerance_m, settings.joint_tolerance_rad, settings.force_tolerance_n],
            dtype=wp.float64,
            device=self.device,
        )
        self.loss = wp.empty(world_count, dtype=wp.float64, device=self.device)
        """Dimensionless measured loss, shape [world_count]."""
        self.rmse = wp.empty((world_count, 6), dtype=wp.float64, device=self.device)
        """Channel RMSE [m, m, rad, rad, N, N], shape [world_count, 6]."""
        self.maximum_error = wp.empty((world_count, 6), dtype=wp.float64, device=self.device)
        """Channel maximum absolute error [m, m, rad, rad, N, N]."""
        self.costs = wp.empty((world_count, 3), dtype=wp.float64, device=self.device)
        """Dimensionless hip, joint, and force costs, shape [world_count, 3]."""
        self.residual = wp.empty((self.residual_dim, world_count), dtype=wp.float64, device=self.device)
        """Dimensionless weighted residual, shape [residual_dim, world_count]."""

    def launch(
        self,
        states: wp.array2d[Vec5],
        forces: wp.array2d[wp.vec2d],
        integrated_steps: wp.array[int],
        failure_code: wp.array[int],
    ) -> None:
        """Launch device-only reductions without synchronization or readback.

        Args:
            states: Positions [m, m, rad, rad, rad], shape [steps + 1, worlds].
            forces: Preintegration forces [N, N], shape [steps, worlds].
            integrated_steps: Completed integration counts, shape [worlds].
            failure_code: Zero for success, nonzero for failure, shape [worlds].
        """
        wp.launch(
            _evaluate,
            dim=self.world_count,
            inputs=[
                states,
                forces,
                integrated_steps,
                failure_code,
                self.steps,
                self.motion_sample_count,
                self.force_sample_count,
                self._lower,
                self._upper,
                self._fraction,
                self._targets,
                self._weights,
                self._residual_weights,
                self._scales,
                self.ground_angle,
                wp.float64(self.ground_offset),
            ],
            outputs=[self.loss, self.rmse, self.maximum_error, self.costs, self.residual],
            device=self.device,
        )

    def read(self) -> dict[str, np.ndarray]:
        """Copy batch diagnostics to the host for an explicit reporting boundary.

        This method is not part of integration or device-side optimization.
        Invalid worlds contain infinity in every output, not partial fit errors.
        """
        return {
            "loss": self.loss.numpy(),
            "rmse": self.rmse.numpy(),
            "maximum_error": self.maximum_error.numpy(),
            "costs": self.costs.numpy(),
            "residual": self.residual.numpy(),
        }
