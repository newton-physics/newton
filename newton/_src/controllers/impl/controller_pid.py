# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""ControllerPID — stateful PID controller with anti-windup."""

from __future__ import annotations

from dataclasses import dataclass

import warp as wp

from ..base import Controller
from ..utils import _normalize_port


@wp.kernel
def _pid_kernel(
    measurement: wp.array[float],
    measurement_indices: wp.array[wp.uint32],
    measurement_rate: wp.array[float],
    measurement_rate_indices: wp.array[wp.uint32],
    setpoint: wp.array[float],
    setpoint_indices: wp.array[wp.uint32],
    setpoint_rate: wp.array[float],
    setpoint_rate_indices: wp.array[wp.uint32],
    kp: wp.array[float],
    kp_indices: wp.array[wp.uint32],
    ki: wp.array[float],
    ki_indices: wp.array[wp.uint32],
    kd: wp.array[float],
    kd_indices: wp.array[wp.uint32],
    integral_max: wp.array[float],
    integral_max_indices: wp.array[wp.uint32],
    output_indices: wp.array[wp.uint32],
    dt: float,
    current_integral: wp.array[float],
    output: wp.array[float],
    next_integral: wp.array[float],
):
    i = wp.tid()
    pos_err = setpoint[setpoint_indices[i]] - measurement[measurement_indices[i]]
    vel_err = setpoint_rate[setpoint_rate_indices[i]] - measurement_rate[measurement_rate_indices[i]]

    imax = integral_max[integral_max_indices[i]]
    integral = wp.clamp(current_integral[i] + pos_err * dt, -imax, imax)

    contribution = kp[kp_indices[i]] * pos_err + ki[ki_indices[i]] * integral + kd[kd_indices[i]] * vel_err
    out_idx = output_indices[i]
    output[out_idx] = output[out_idx] + contribution
    next_integral[i] = integral


@wp.kernel
def _pid_masked_reset_kernel(
    integral: wp.array[float],
    target_integral: wp.array[float],
    mask: wp.array[wp.bool],
):
    i = wp.tid()
    if mask[i]:
        integral[i] = target_integral[i]


class ControllerPID(Controller):
    """Stateful PID controller with symmetric anti-windup clamping.

    Independent per-DOF: ``output[i]`` depends only on the ``i``-th entries of
    its bound ports. Each compute step adds the following to the bound
    ``output`` array:

    .. code-block:: text

        contribution[i] = kp[i] * (setpoint - measurement)
                        + ki[i] * clamp(integral + (setpoint - measurement) * dt,
                                        -integral_max, +integral_max)
                        + kd[i] * (setpoint_rate - measurement_rate)

    Per-DOF port forms follow the unified rule: bare ``wp.array`` uses
    ``indices`` as the lookup; ``(array, port_indices)`` uses ``port_indices``.

    Args:
        indices: Global DOF indices this controller writes to. Length ``N``.
        measurement: Process variable (per-DOF port).
        measurement_rate: Time derivative of the measurement (per-DOF port).
        setpoint: Target value for the measurement (per-DOF port).
        setpoint_rate: Time derivative of the setpoint (per-DOF port).
        kp: Proportional gain (per-DOF port).
        ki: Integral gain (per-DOF port).
        kd: Derivative gain (per-DOF port).
        integral_max: Symmetric saturation bound on the integrator
            (per-DOF port). Use ``wp.full(N, float('inf'))`` to disable
            clamping.
        output: Destination array (per-DOF port). ``compute()`` accumulates
            ``+=`` into the bound slots.
    """

    @dataclass
    class State(Controller.State):
        integral: wp.array[float] | None = None
        """Accumulated integral term, shape ``[N]``."""

    def __init__(
        self,
        *,
        indices: wp.array[wp.uint32],
        measurement,
        measurement_rate,
        setpoint,
        setpoint_rate,
        kp,
        ki,
        kd,
        integral_max,
        output,
    ):
        if not isinstance(indices, wp.array):
            raise TypeError(f"ControllerPID: indices must be wp.array[wp.uint32], got {type(indices).__name__}.")
        self.indices = indices
        self._measurement = _normalize_port(measurement, indices, name="measurement")
        self._measurement_rate = _normalize_port(measurement_rate, indices, name="measurement_rate")
        self._setpoint = _normalize_port(setpoint, indices, name="setpoint")
        self._setpoint_rate = _normalize_port(setpoint_rate, indices, name="setpoint_rate")
        self._kp = _normalize_port(kp, indices, name="kp")
        self._ki = _normalize_port(ki, indices, name="ki")
        self._kd = _normalize_port(kd, indices, name="kd")
        self._integral_max = _normalize_port(integral_max, indices, name="integral_max")
        self._output = _normalize_port(output, indices, name="output")

    def finalize(self, device: wp.Device, num_outputs: int) -> None:
        self.reset_state = self.state(num_outputs, device)

    def is_stateful(self) -> bool:
        return True

    def is_graphable(self) -> bool:
        return True

    def state(self, num_outputs: int, device: wp.Device) -> ControllerPID.State:
        return ControllerPID.State(
            integral=wp.zeros(num_outputs, dtype=wp.float32, device=device),
        )

    def outputs(self) -> list[tuple[wp.array, wp.array[wp.uint32]]]:
        return [self._output]

    def compute(
        self,
        state: ControllerPID.State,
        next_state: ControllerPID.State,
        dt: float,
    ) -> None:
        meas, meas_idx = self._measurement
        meas_rate, mrate_idx = self._measurement_rate
        sp, sp_idx = self._setpoint
        sp_rate, sp_rate_idx = self._setpoint_rate
        kp, kp_idx = self._kp
        ki, ki_idx = self._ki
        kd, kd_idx = self._kd
        imax, imax_idx = self._integral_max
        out, out_idx = self._output
        wp.launch(
            _pid_kernel,
            dim=len(self.indices),
            inputs=[
                meas,
                meas_idx,
                meas_rate,
                mrate_idx,
                sp,
                sp_idx,
                sp_rate,
                sp_rate_idx,
                kp,
                kp_idx,
                ki,
                ki_idx,
                kd,
                kd_idx,
                imax,
                imax_idx,
                out_idx,
                dt,
                state.integral,
            ],
            outputs=[out, next_state.integral],
        )

    def reset(
        self,
        state: ControllerPID.State,
        mask: wp.array[wp.bool],
    ) -> None:
        wp.launch(
            _pid_masked_reset_kernel,
            dim=len(self.indices),
            inputs=[state.integral, self.reset_state.integral, mask],
        )
