# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""ControllerDifferentialDrive — vectorized unicycle differential-drive."""

from __future__ import annotations

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
def _diff_drive_kernel(
    linear_speed: wp.array[float],
    linear_speed_indices: wp.array[wp.uint32],
    angular_speed: wp.array[float],
    angular_speed_indices: wp.array[wp.uint32],
    wheel_radius: wp.array[float],
    wheel_base: wp.array[float],
    max_linear_speed: wp.array[float],
    max_angular_speed: wp.array[float],
    max_wheel_speed: wp.array[float],
    output_indices: wp.array[wp.uint32],
    output: wp.array[float],
):
    """Per-robot: convert [v, omega] to (left, right) wheel angular velocities.

    ``output_indices`` is laid out ``[r0_left, r0_right, r1_left, r1_right, ...]``
    — slot ``2*r`` is robot r's left wheel, slot ``2*r + 1`` is its right wheel.
    """
    r = wp.tid()
    v_max = max_linear_speed[r]
    w_max = max_angular_speed[r]
    v = wp.clamp(linear_speed[linear_speed_indices[r]], -v_max, v_max)
    w = wp.clamp(angular_speed[angular_speed_indices[r]], -w_max, w_max)

    radius = wheel_radius[r]
    base = wheel_base[r]
    # omega_L = (2V - omega*b)/(2r), omega_R = (2V + omega*b)/(2r)
    left = (2.0 * v - w * base) / (2.0 * radius)
    right = (2.0 * v + w * base) / (2.0 * radius)

    wheel_max = max_wheel_speed[r]
    left = wp.clamp(left, -wheel_max, wheel_max)
    right = wp.clamp(right, -wheel_max, wheel_max)

    output[output_indices[2 * r + 0]] = left
    output[output_indices[2 * r + 1]] = right


class ControllerDifferentialDrive(Controller):
    """Vectorized unicycle differential-drive controller.

    For each of N wheeled robots, converts a per-robot ``(linear_speed,
    angular_speed)`` command into the pair of left / right wheel angular
    velocities that realise it under the standard differential-drive
    kinematic model::

        omega_L = (2 * v - omega * wheel_base) / (2 * wheel_radius)
        omega_R = (2 * v + omega * wheel_base) / (2 * wheel_radius)

    The body-frame command is clamped to ``[-max_linear_speed,
    +max_linear_speed]`` / ``[-max_angular_speed, +max_angular_speed]``
    before the conversion; the resulting wheel rates are clamped to
    ``[-max_wheel_speed, +max_wheel_speed]`` after.

    Stateless. The output ports are slot-replacing writes (``=``) into the
    caller's ``joint_target_qd``-style array; slots outside the controller's
    ``2 * num_robots`` outputs are untouched.

    **Parameter ports** (``wheel_radius``, ``wheel_base``, ``max_linear_speed``,
    ``max_angular_speed``, ``max_wheel_speed``) accept either a length-N
    :class:`wp.array` (baked; stored by copy at construction) or a ``str``
    (live; resolved from the input struct each step). All natural per-robot
    order; no ``_idx`` override.

    Args:
        num_robots: Number of differential-drive robots managed.
        wheel_radius: Per-robot wheel radius [m].
        wheel_base: Per-robot lateral wheel-to-wheel distance [m].
        default_dof_indices: Output indices for each robot's two wheel
            DOFs, length ``2 * num_robots``, laid out
            ``[r0_left, r0_right, r1_left, r1_right, ...]``.
        max_linear_speed: Per-robot forward/backward speed limit [m/s].
            Defaults to ``+inf`` per robot.
        max_angular_speed: Per-robot yaw-rate limit [rad/s]. Defaults to
            ``+inf`` per robot.
        max_wheel_speed: Per-robot wheel-rate clamp [rad/s] applied after
            the kinematic conversion. Defaults to ``+inf`` per robot.
        linear_speed_attr: Live read port — body-frame forward speed
            command [m/s]. Length ``num_robots``.
        linear_speed_idx: Override per-robot indices, or ``None`` to use
            natural order ``wp.arange(num_robots)``.
        angular_speed_attr: Live read port — body-frame yaw-rate command
            [rad/s]. Length ``num_robots``.
        angular_speed_idx: Override per-robot indices.
        joint_target_qd_attr: Live write port — commanded wheel angular
            velocities [rad/s].
        joint_target_qd_idx: Override per-DOF indices, or ``None`` to use
            ``default_dof_indices``.
        device: Warp device for internal buffers. Defaults to
            :func:`wp.get_device`.
        requires_grad: If ``True``, internal buffers are allocated with
            gradient support so :meth:`compute` is transparent to
            :class:`wp.Tape`.
    """

    def __init__(
        self,
        num_robots: int,
        wheel_radius: wp.array | str,
        wheel_base: wp.array | str,
        default_dof_indices: wp.array,
        max_linear_speed: wp.array | str | None = None,
        max_angular_speed: wp.array | str | None = None,
        max_wheel_speed: wp.array | str | None = None,
        linear_speed_attr: str = "linear_speed_command",
        linear_speed_idx: wp.array | None = None,
        angular_speed_attr: str = "angular_speed_command",
        angular_speed_idx: wp.array | None = None,
        joint_target_qd_attr: str = "joint_target_qd",
        joint_target_qd_idx: wp.array | None = None,
        device: Any = None,
        requires_grad: bool = False,
    ):
        if num_robots < 1:
            raise ValueError(f"num_robots must be >= 1, got {num_robots}.")
        if not isinstance(default_dof_indices, wp.array) or default_dof_indices.dtype != wp.uint32:
            raise TypeError("default_dof_indices must be wp.array[uint32].")
        if int(default_dof_indices.size) != 2 * num_robots:
            raise ValueError(
                f"default_dof_indices length {default_dof_indices.size} must equal 2 * num_robots = {2 * num_robots}."
            )

        self._num_robots = num_robots
        self._num_outputs = 2 * num_robots
        self._device = device if device is not None else wp.get_device()
        self._requires_grad = requires_grad
        self._default_dof_indices = default_dof_indices

        # Per-DOF output port.
        self._output_attr = joint_target_qd_attr
        self._output_idx = _normalize_indices(joint_target_qd_idx, default_dof_indices, name="joint_target_qd")

        # Per-robot live command ports.
        default_robot_idx = wp.array(np.arange(num_robots, dtype=np.uint32), device=self._device)
        self._default_robot_idx = default_robot_idx
        self._linear_speed_attr = linear_speed_attr
        self._linear_speed_idx = _normalize_indices(linear_speed_idx, default_robot_idx, name="linear_speed")
        self._angular_speed_attr = angular_speed_attr
        self._angular_speed_idx = _normalize_indices(angular_speed_idx, default_robot_idx, name="angular_speed")

        # Per-robot parameter ports. Defaults: +inf for the three clamps.
        if max_linear_speed is None:
            max_linear_speed = wp.full(num_robots, value=wp.inf, dtype=wp.float32, device=self._device)
        if max_angular_speed is None:
            max_angular_speed = wp.full(num_robots, value=wp.inf, dtype=wp.float32, device=self._device)
        if max_wheel_speed is None:
            max_wheel_speed = wp.full(num_robots, value=wp.inf, dtype=wp.float32, device=self._device)

        self._wheel_radius_attr, self._wheel_radius_baked = _normalize_parameter_port(
            wheel_radius, num_robots, wp.float32, self._device, requires_grad, name="wheel_radius"
        )
        self._wheel_base_attr, self._wheel_base_baked = _normalize_parameter_port(
            wheel_base, num_robots, wp.float32, self._device, requires_grad, name="wheel_base"
        )
        self._max_lin_attr, self._max_lin_baked = _normalize_parameter_port(
            max_linear_speed, num_robots, wp.float32, self._device, requires_grad, name="max_linear_speed"
        )
        self._max_ang_attr, self._max_ang_baked = _normalize_parameter_port(
            max_angular_speed, num_robots, wp.float32, self._device, requires_grad, name="max_angular_speed"
        )
        self._max_wheel_attr, self._max_wheel_baked = _normalize_parameter_port(
            max_wheel_speed, num_robots, wp.float32, self._device, requires_grad, name="max_wheel_speed"
        )

        # input_struct fields: command ports + any live parameter ports.
        self._input_specs: list[tuple[str, Any, int]] = [
            (self._linear_speed_attr, wp.float32, _idx_max(self._linear_speed_idx)),
            (self._angular_speed_attr, wp.float32, _idx_max(self._angular_speed_idx)),
        ]
        for attr in (
            self._wheel_radius_attr,
            self._wheel_base_attr,
            self._max_lin_attr,
            self._max_ang_attr,
            self._max_wheel_attr,
        ):
            if attr is not None:
                self._input_specs.append((attr, wp.float32, num_robots))

        self._output_specs: list[tuple[str, Any, int]] = [
            (self._output_attr, wp.float32, _idx_max(self._output_idx)),
        ]

    @property
    def num_robots(self) -> int:
        return self._num_robots

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
        return False

    def is_graphable(self) -> bool:
        return True

    def state(self) -> None:
        return None

    def input_struct(self):
        return _allocate_namespace(self._input_specs, self._device, self._requires_grad)

    def output_struct(self):
        return _allocate_namespace(self._output_specs, self._device, self._requires_grad)

    def compute(
        self,
        input_struct: Any,
        output_struct: Any,
        controller_state_now: None,
        controller_state_next: None,
        time_step: float,
    ) -> None:
        linear_speed = getattr(input_struct, self._linear_speed_attr)
        angular_speed = getattr(input_struct, self._angular_speed_attr)
        wheel_radius = self._wheel_radius_baked or getattr(input_struct, self._wheel_radius_attr)
        wheel_base = self._wheel_base_baked or getattr(input_struct, self._wheel_base_attr)
        max_lin = self._max_lin_baked or getattr(input_struct, self._max_lin_attr)
        max_ang = self._max_ang_baked or getattr(input_struct, self._max_ang_attr)
        max_wheel = self._max_wheel_baked or getattr(input_struct, self._max_wheel_attr)
        out = getattr(output_struct, self._output_attr)

        wp.launch(
            _diff_drive_kernel,
            dim=self._num_robots,
            inputs=[
                linear_speed,
                self._linear_speed_idx,
                angular_speed,
                self._angular_speed_idx,
                wheel_radius,
                wheel_base,
                max_lin,
                max_ang,
                max_wheel,
                self._output_idx,
            ],
            outputs=[out],
            device=self._device,
        )


def _idx_max(idx: wp.array) -> int:
    return int(np.max(idx.numpy())) + 1
