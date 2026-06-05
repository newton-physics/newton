# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""ControlLawPID — stateful PID controller with anti-windup."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import warp as wp

from ..control_law import ControlLaw
from ..utils import _normalize_port, _resolve_input_array


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


class ControlLawPID(ControlLaw):
    """Stateful PID controller with symmetric anti-windup clamping.

    Independent per-DOF: ``output[i]`` depends only on the ``i``-th entries of
    its declared ports. Each compute step adds the following to the output
    array:

    .. code-block:: text

        contribution[i] = kp[i] * (setpoint - measurement)
                        + ki[i] * clamp(integral + (setpoint - measurement) * dt,
                                        -integral_max, +integral_max)
                        + kd[i] * (setpoint_rate - measurement_rate)

    Every port is a **string** giving the attribute name on the ``input`` /
    ``output`` object passed to :meth:`Controller.step` — e.g.
    ``measurement="joint_q"`` resolves to ``input.joint_q`` at step time.
    Per-DOF ports may also be passed as ``(attr_name, port_indices)`` if
    the source array's layout differs from this controller's ``indices``.

    Args:
        indices: Global DOF indices this controller writes to. Length ``N``.
        measurement: Per-DOF read port. Process variable.
        measurement_rate: Per-DOF read port. Time derivative of measurement.
        setpoint: Per-DOF read port. Target for the measurement.
        setpoint_rate: Per-DOF read port. Time derivative of the setpoint.
        kp: Per-DOF read port. Proportional gain.
        ki: Per-DOF read port. Integral gain.
        kd: Per-DOF read port. Derivative gain.
        integral_max: Per-DOF read port. Symmetric saturation bound on the
            integrator. Use ``wp.full(N, float('inf'))`` to disable clamping.
        output: Per-DOF write port. ``compute()`` accumulates ``+=`` into
            the slots ``output[output_indices[i]]``.
    """

    @dataclass
    class State(ControlLaw.State):
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
            raise TypeError(f"ControlLawPID: indices must be wp.array[wp.uint32], got {type(indices).__name__}.")
        self.indices = indices
        self._measurement_attr, self._measurement_port_indices = _normalize_port(
            measurement, indices, name="measurement"
        )
        self._measurement_rate_attr, self._measurement_rate_port_indices = _normalize_port(
            measurement_rate, indices, name="measurement_rate"
        )
        self._setpoint_attr, self._setpoint_port_indices = _normalize_port(setpoint, indices, name="setpoint")
        self._setpoint_rate_attr, self._setpoint_rate_port_indices = _normalize_port(
            setpoint_rate, indices, name="setpoint_rate"
        )
        self._kp_attr, self._kp_port_indices = _normalize_port(kp, indices, name="kp")
        self._ki_attr, self._ki_port_indices = _normalize_port(ki, indices, name="ki")
        self._kd_attr, self._kd_port_indices = _normalize_port(kd, indices, name="kd")
        self._integral_max_attr, self._integral_max_port_indices = _normalize_port(
            integral_max, indices, name="integral_max"
        )
        self._output_attr, self._output_port_indices = _normalize_port(output, indices, name="output")

    def finalize(self, device: wp.Device, num_outputs: int, requires_grad: bool = False) -> None:
        self.reset_state = self.state(num_outputs, device, requires_grad=requires_grad)

    def is_stateful(self) -> bool:
        return True

    def is_graphable(self) -> bool:
        return True

    def state(self, num_outputs: int, device: wp.Device, requires_grad: bool = False) -> ControlLawPID.State:
        return ControlLawPID.State(
            integral=wp.zeros(num_outputs, dtype=wp.float32, device=device, requires_grad=requires_grad),
        )

    def outputs(self) -> list[tuple[str, wp.array[wp.uint32]]]:
        return [(self._output_attr, self._output_port_indices)]

    def compute(
        self,
        input: Any,
        output: Any,
        state: ControlLawPID.State,
        next_state: ControlLawPID.State,
        dt: float,
    ) -> None:
        meas = _resolve_input_array(input, self._measurement_attr, name="measurement")
        meas_rate = _resolve_input_array(input, self._measurement_rate_attr, name="measurement_rate")
        sp = _resolve_input_array(input, self._setpoint_attr, name="setpoint")
        sp_rate = _resolve_input_array(input, self._setpoint_rate_attr, name="setpoint_rate")
        kp = _resolve_input_array(input, self._kp_attr, name="kp")
        ki = _resolve_input_array(input, self._ki_attr, name="ki")
        kd = _resolve_input_array(input, self._kd_attr, name="kd")
        imax = _resolve_input_array(input, self._integral_max_attr, name="integral_max")
        out = _resolve_input_array(output, self._output_attr, name="output")
        wp.launch(
            _pid_kernel,
            dim=len(self.indices),
            inputs=[
                meas,
                self._measurement_port_indices,
                meas_rate,
                self._measurement_rate_port_indices,
                sp,
                self._setpoint_port_indices,
                sp_rate,
                self._setpoint_rate_port_indices,
                kp,
                self._kp_port_indices,
                ki,
                self._ki_port_indices,
                kd,
                self._kd_port_indices,
                imax,
                self._integral_max_port_indices,
                self._output_port_indices,
                dt,
                state.integral,
            ],
            outputs=[out, next_state.integral],
        )

    def reset(
        self,
        state: ControlLawPID.State,
        mask: wp.array[wp.bool],
    ) -> None:
        wp.launch(
            _pid_masked_reset_kernel,
            dim=len(self.indices),
            inputs=[state.integral, self.reset_state.integral, mask],
        )
