# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""ControllerPID — stateful PID with anti-windup."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import warp as wp

from ..controller import Controller
from ..utils import (
    _allocate_namespace,
    _normalize_indices,
    _normalize_parameter_port,
)


@wp.kernel
def _pid_kernel(
    measurement: wp.array[float],
    measurement_indices: wp.array[wp.uint32],
    measurement_rate: wp.array[float],
    measurement_rate_indices: wp.array[wp.uint32],
    joint_target: wp.array[float],
    joint_target_indices: wp.array[wp.uint32],
    joint_target_rate: wp.array[float],
    joint_target_rate_indices: wp.array[wp.uint32],
    kp: wp.array[float],
    ki: wp.array[float],
    kd: wp.array[float],
    integral_max: wp.array[float],
    output_indices: wp.array[wp.uint32],
    dt: float,
    current_integral: wp.array[float],
    output: wp.array[float],
    next_integral: wp.array[float],
):
    i = wp.tid()
    pos_err = joint_target[joint_target_indices[i]] - measurement[measurement_indices[i]]
    vel_err = joint_target_rate[joint_target_rate_indices[i]] - measurement_rate[measurement_rate_indices[i]]
    imax = integral_max[i]
    integral = wp.clamp(current_integral[i] + pos_err * dt, -imax, imax)
    contribution = kp[i] * pos_err + ki[i] * integral + kd[i] * vel_err
    output[output_indices[i]] = contribution
    next_integral[i] = integral


class ControllerPID(Controller):
    """Stateful PID controller with symmetric anti-windup clamping.

    Per-DOF: ``output[i]`` depends only on the ``i``-th entries of its
    declared ports. Each :meth:`compute` step writes::

        output[output_idx[i]] = kp[i] * (joint_target - joint_measured)
                              + ki[i] * clamp(integral + pos_err * dt,
                                              -integral_max[i], +integral_max[i])
                              + kd[i] * (joint_target_rate - joint_measured_rate)

    where the bracketed differences read ``arr[idx[i]]`` for each live port.

    **Gain ports** (``kp``, ``kd``, ``ki``, ``integral_max``) take either:

    - a :class:`wp.array` — *baked*: the controller stores its own copy at
      construction; mutating the user's original after construction has no
      effect. Must be length ``default_dof_indices.size`` and dtype
      ``wp.float32``.
    - a ``str`` — *live*: at step time the controller resolves
      ``getattr(input_struct, value)`` and reads from that array in natural
      order (``arr[i]``).

    **Live-data ports** take an ``attr_name`` string plus an optional
    ``_idx`` array. When ``_idx`` is ``None`` the controller's
    ``default_dof_indices`` is used; otherwise the kernel reads
    ``arr[_idx[i]]``.

    Args:
        kp: Proportional gain.
        kd: Derivative gain.
        ki: Integral gain.
        integral_max: Symmetric integrator clamp. Use
            ``wp.full(N, float('inf'))`` to disable clamping.
        default_dof_indices: Default indices for any live-data port whose
            ``*_idx`` is ``None``. Length ``num_outputs``.
        joint_measured_attr: Live read port — process variable (joint
            positions).
        joint_measured_idx: Override indices for ``joint_measured_attr``,
            or ``None`` to use ``default_dof_indices``.
        joint_measured_rate_attr: Live read port — derivative of the
            process variable (joint velocities).
        joint_measured_rate_idx: Override indices.
        joint_target_attr: Live read port — target (joint positions).
        joint_target_idx: Override indices.
        joint_target_rate_attr: Live read port — target derivative.
        joint_target_rate_idx: Override indices.
        output_attr: Live write port — controller output (effort).
        output_idx: Override indices.
        device: Warp device for internal buffers. Defaults to
            :func:`wp.get_device`.
        requires_grad: If ``True``, internally-allocated buffers are
            created with gradient support so :meth:`compute` is
            transparent to :class:`wp.Tape`.
    """

    @dataclass
    class State(Controller.State):
        integral: wp.array[float] | None = None
        """Accumulated integral term, shape ``[num_outputs]``."""

    def __init__(
        self,
        default_dof_indices: wp.array,
        kp: wp.array | str | None = None,
        kd: wp.array | str | None = None,
        ki: wp.array | str | None = None,
        integral_max: wp.array | str | None = None,
        joint_measured_attr: str = "joint_q",
        joint_measured_idx: wp.array | None = None,
        joint_measured_rate_attr: str = "joint_qd",
        joint_measured_rate_idx: wp.array | None = None,
        joint_target_attr: str = "joint_target_q",
        joint_target_idx: wp.array | None = None,
        joint_target_rate_attr: str = "joint_target_qd",
        joint_target_rate_idx: wp.array | None = None,
        output_attr: str = "joint_f",
        output_idx: wp.array | None = None,
        device: Any = None,
        requires_grad: bool = False,
    ):
        if not isinstance(default_dof_indices, wp.array):
            raise TypeError(f"default_dof_indices must be wp.array[uint32], got {type(default_dof_indices).__name__}.")

        self._device = device if device is not None else wp.get_device()
        self._requires_grad = requires_grad
        self._default_dof_indices = default_dof_indices
        self._num_outputs = int(default_dof_indices.size)

        # I/O-data ports.
        self._measurement_attr = joint_measured_attr
        self._measurement_idx = _normalize_indices(joint_measured_idx, default_dof_indices, name="joint_measured")

        self._measurement_rate_attr = joint_measured_rate_attr
        self._measurement_rate_idx = _normalize_indices(
            joint_measured_rate_idx, default_dof_indices, name="joint_measured_rate"
        )

        self._target_attr = joint_target_attr
        self._target_idx = _normalize_indices(joint_target_idx, default_dof_indices, name="joint_target")

        self._target_rate_attr = joint_target_rate_attr
        self._target_rate_idx = _normalize_indices(joint_target_rate_idx, default_dof_indices, name="joint_target_rate")

        self._output_attr = output_attr
        self._output_idx = _normalize_indices(output_idx, default_dof_indices, name="output")

        # Gain ports (wp.array | str).
        if not kp:
            kp = wp.zeros(
                shape=self._num_outputs, 
                dtype=float, 
                device=self.device, 
                requires_grad=self.requires_grad
            )
        self._kp_attr, self._kp_baked = _normalize_parameter_port(
            kp, self._num_outputs, wp.float32, self._device, requires_grad, name="kp"
        )

        if not kd:
            kd = wp.zeros(
                shape=self._num_outputs, 
                dtype=float, 
                device=self.device, 
                requires_grad=self.requires_grad
            )
        self._kd_attr, self._kd_baked = _normalize_parameter_port(
            kd, self._num_outputs, wp.float32, self._device, requires_grad, name="kd"
        )

        if not ki:
            ki = wp.zeros(
                shape=self._num_outputs, 
                dtype=float, 
                device=self.device, 
                requires_grad=self.requires_grad
            )
        self._ki_attr, self._ki_baked = _normalize_parameter_port(
            ki, self._num_outputs, wp.float32, self._device, requires_grad, name="ki"
        )

        if not integral_max:
            # default to no max:
            integral_max = wp.full(
                shape=self._num_outputs,
                value=wp.inf,
                dtype=float, 
                device=self.device, 
                requires_grad=self.requires_grad
            )
        self._imax_attr, self._imax_baked = _normalize_parameter_port(
            integral_max, self._num_outputs, wp.float32, self._device, requires_grad, name="integral_max"
        )

        # Live read ports + any live gain ports --> fields on input_struct().
        self._input_specs: list[tuple[str, Any, int]] = [
            (self._measurement_attr, wp.float32, _idx_max(self._measurement_idx)),
            (self._measurement_rate_attr, wp.float32, _idx_max(self._measurement_rate_idx)),
            (self._target_attr, wp.float32, _idx_max(self._target_idx)),
            (self._target_rate_attr, wp.float32, _idx_max(self._target_rate_idx)),
        ]
        for attr in (self._kp_attr, self._kd_attr, self._ki_attr, self._imax_attr):
            if attr is not None:
                self._input_specs.append((attr, wp.float32, self._num_outputs))

        self._output_specs: list[tuple[str, Any, int]] = [
            (self._output_attr, wp.float32, _idx_max(self._output_idx)),
        ]

    @property
    def num_outputs(self) -> int:
        return self._num_outputs

    @property
    def device(self):
        return self._device

    @property
    def requires_grad(self) -> bool:
        return self._requires_grad

    def is_stateful(self) -> bool:
        return True

    def is_graphable(self) -> bool:
        return True

    def state(self) -> ControllerPID.State:
        return ControllerPID.State(
            integral=wp.zeros(
                self._num_outputs, dtype=wp.float32, device=self._device, requires_grad=self._requires_grad
            ),
        )

    def input_struct(self):
        return _allocate_namespace(self._input_specs, self._device, self._requires_grad)

    def output_struct(self):
        return _allocate_namespace(self._output_specs, self._device, self._requires_grad)

    def compute(
        self,
        input_struct: Any,
        output_struct: Any,
        controller_state_now: ControllerPID.State,
        controller_state_next: ControllerPID.State,
        time_step: float,
    ) -> None:
        meas = getattr(input_struct, self._measurement_attr)
        meas_rate = getattr(input_struct, self._measurement_rate_attr)
        target = getattr(input_struct, self._target_attr)
        target_rate = getattr(input_struct, self._target_rate_attr)
        out = getattr(output_struct, self._output_attr)

        kp = self._kp_baked or getattr(input_struct, self._kp_attr)
        kd = self._kd_baked or getattr(input_struct, self._kd_attr)
        ki = self._ki_baked or getattr(input_struct, self._ki_attr)
        imax = self._imax_baked or getattr(input_struct, self._imax_attr)

        wp.launch(
            _pid_kernel,
            dim=self._num_outputs,
            inputs=[
                meas,
                self._measurement_idx,
                meas_rate,
                self._measurement_rate_idx,
                target,
                self._target_idx,
                target_rate,
                self._target_rate_idx,
                kp,
                ki,
                kd,
                imax,
                self._output_idx,
                time_step,
                controller_state_now.integral,
            ],
            outputs=[out, controller_state_next.integral],
            device=self._device,
        )


def _idx_max(idx: wp.array) -> int:
    """Smallest backing-array size that satisfies ``arr[idx[i]]`` reads.

    Reads ``idx.numpy()`` once at construction — cheap relative to the rest
    of controller setup, and lets :meth:`input_struct` size every field
    minimally for the controller's view.
    """
    return int(np.max(idx.numpy())) + 1
