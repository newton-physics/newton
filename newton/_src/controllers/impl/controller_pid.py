# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""ControlLawPID — stateful PID controller with anti-windup."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import warp as wp

from ..control_law import ControlLaw
from ..utils import HardwareInterface, _normalize_port, _resolve_input_array


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


class ControlLawPID(ControlLaw):
    """Stateful PID controller with symmetric anti-windup clamping.

    Independent per-DOF: ``output[i]`` depends only on the ``i``-th
    entries of its declared ports. Each compute step adds the following
    to the output array:

    .. code-block:: text

        contribution[i] = kp[i] * (setpoint - measurement)
                        + ki[i] * clamp(integral + (setpoint - measurement) * dt,
                                        -integral_max, +integral_max)
                        + kd[i] * (setpoint_rate - measurement_rate)

    Every port is bound to a :class:`ControlSignal` and a
    ``port_indices`` array at construction. The composing
    :class:`Controller` resolves each signal to its attribute name on
    the runtime ``input`` / ``output`` object via its
    :class:`HardwareInterface`.

    Args:
        measurement: Per-DOF read port for the process variable.
        measurement_rate: Per-DOF read port for the derivative of the
            measurement.
        setpoint: Per-DOF read port for the target value.
        setpoint_rate: Per-DOF read port for the derivative of the
            setpoint.
        kp: Per-DOF read port for the proportional gain.
        ki: Per-DOF read port for the integral gain.
        kd: Per-DOF read port for the derivative gain.
        integral_max: Per-DOF read port for the symmetric integrator
            clamp. Pass a constant ``wp.full(N, float('inf'))`` array on
            the input object to disable clamping.
        output: Per-DOF write port. ``compute()`` accumulates ``+=``
            into the slots ``output[output_indices[i]]``.

    Every port spec is a 2-tuple ``(ControlSignal, wp.array[wp.uint32])``.
    The output port's ``port_indices`` length defines ``num_outputs``;
    every other port's ``port_indices`` is cross-checked against it.
    """

    INPUT_PORTS = frozenset(
        {
            "measurement",
            "measurement_rate",
            "setpoint",
            "setpoint_rate",
            "kp",
            "ki",
            "kd",
            "integral_max",
        }
    )
    OUTPUT_PORTS = frozenset({"output"})

    @dataclass
    class State(ControlLaw.State):
        integral: wp.array[float] | None = None
        """Accumulated integral term, shape ``[num_outputs]``."""

    def __init__(
        self,
        *,
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
        # Output binding defines num_outputs; everything else aligns.
        self._output_signal, self._output_port_indices = _normalize_port(output, name="output")
        self._num_outputs = self._output_port_indices.shape[0]

        def _norm_input(spec, name):
            signal, port_indices = _normalize_port(spec, name=name)
            if port_indices.shape != self._output_port_indices.shape:
                raise ValueError(
                    f"Port '{name}': port_indices shape {port_indices.shape} must match "
                    f"the output port's shape {self._output_port_indices.shape}."
                )
            return signal, port_indices

        self._measurement_signal, self._measurement_port_indices = _norm_input(measurement, "measurement")
        self._measurement_rate_signal, self._measurement_rate_port_indices = _norm_input(
            measurement_rate, "measurement_rate"
        )
        self._setpoint_signal, self._setpoint_port_indices = _norm_input(setpoint, "setpoint")
        self._setpoint_rate_signal, self._setpoint_rate_port_indices = _norm_input(setpoint_rate, "setpoint_rate")
        self._kp_signal, self._kp_port_indices = _norm_input(kp, "kp")
        self._ki_signal, self._ki_port_indices = _norm_input(ki, "ki")
        self._kd_signal, self._kd_port_indices = _norm_input(kd, "kd")
        self._integral_max_signal, self._integral_max_port_indices = _norm_input(integral_max, "integral_max")

        self._used_inputs = frozenset(
            {
                self._measurement_signal,
                self._measurement_rate_signal,
                self._setpoint_signal,
                self._setpoint_rate_signal,
                self._kp_signal,
                self._ki_signal,
                self._kd_signal,
                self._integral_max_signal,
            }
        )
        self._used_outputs = frozenset({self._output_signal})

        # Resolved attribute names — filled in by _resolve().
        self._measurement_attr: str | None = None
        self._measurement_rate_attr: str | None = None
        self._setpoint_attr: str | None = None
        self._setpoint_rate_attr: str | None = None
        self._kp_attr: str | None = None
        self._ki_attr: str | None = None
        self._kd_attr: str | None = None
        self._integral_max_attr: str | None = None
        self._output_attr: str | None = None

    def _resolve(self, hw: HardwareInterface) -> None:
        self._measurement_attr = hw.inputs[self._measurement_signal]
        self._measurement_rate_attr = hw.inputs[self._measurement_rate_signal]
        self._setpoint_attr = hw.inputs[self._setpoint_signal]
        self._setpoint_rate_attr = hw.inputs[self._setpoint_rate_signal]
        self._kp_attr = hw.inputs[self._kp_signal]
        self._ki_attr = hw.inputs[self._ki_signal]
        self._kd_attr = hw.inputs[self._kd_signal]
        self._integral_max_attr = hw.inputs[self._integral_max_signal]
        self._output_attr = hw.outputs[self._output_signal]

    def finalize(self, device: wp.Device, requires_grad: bool = False) -> None:
        # No private buffers — state allocated on demand via state().
        self._device = device
        self._requires_grad = requires_grad

    def is_stateful(self) -> bool:
        return True

    def is_graphable(self) -> bool:
        return True

    def state(self, device: wp.Device, requires_grad: bool = False) -> ControlLawPID.State:
        return ControlLawPID.State(
            integral=wp.zeros(self._num_outputs, dtype=wp.float32, device=device, requires_grad=requires_grad),
        )

    def inputs(self) -> list[tuple[str, wp.array[wp.uint32]]]:
        return [
            (self._measurement_attr, self._measurement_port_indices),
            (self._measurement_rate_attr, self._measurement_rate_port_indices),
            (self._setpoint_attr, self._setpoint_port_indices),
            (self._setpoint_rate_attr, self._setpoint_rate_port_indices),
            (self._kp_attr, self._kp_port_indices),
            (self._ki_attr, self._ki_port_indices),
            (self._kd_attr, self._kd_port_indices),
            (self._integral_max_attr, self._integral_max_port_indices),
        ]

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
            dim=self._num_outputs,
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
