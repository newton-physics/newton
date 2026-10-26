# Copyright 2025 Marc Duclusaud & Grégoire Passault

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Union

import numpy as np
from .testbench import Testbench
from bam.parameter import Parameter

if TYPE_CHECKING:
    import torch

# Anything the control law / torque equations can operate on elementwise. The
# concrete backend (numpy or torch, see :class:`Backend`) decides how ``clamp``
# behaves; the arithmetic itself broadcasts identically over scalars, numpy
# arrays and torch tensors.
ArrayLike = Union[float, np.ndarray, "torch.Tensor"]


class Backend:
    """Abstracts the array library so the actuator math is vectorization-agnostic.

    Only operations that differ between numpy and torch (currently ``clamp``)
    live here; plain arithmetic (``+``, ``*``, ``**``, ``np.sign`` …) broadcasts
    the same way for both, so it is written directly in the actuator methods.
    """

    def clamp(self, x: ArrayLike, low: ArrayLike, high: ArrayLike) -> ArrayLike:
        raise NotImplementedError

    def sign(self, x: ArrayLike) -> ArrayLike:
        raise NotImplementedError


class NumpyBackend(Backend):
    def clamp(self, x: ArrayLike, low: ArrayLike, high: ArrayLike) -> ArrayLike:
        return np.clip(x, low, high)

    def sign(self, x: ArrayLike) -> ArrayLike:
        return np.sign(x)


class TorchBackend(Backend):
    def clamp(self, x: ArrayLike, low: ArrayLike, high: ArrayLike) -> ArrayLike:
        import torch

        return torch.clamp(x, low, high)

    def sign(self, x: ArrayLike) -> ArrayLike:
        import torch

        return torch.sign(x)


class Actuator:
    """Abstract base class for all BAM actuator models.

    Subclasses implement the firmware control law and the motor torque equation
    for a specific actuator family.  An actuator is always attached to a
    :class:`~bam.model.Model` via :meth:`~bam.model.Model.set_actuator`, which
    calls :meth:`initialize` to create the motor parameters (``kt``, ``R``, …)
    on the model.

    :param testbench_class: Class used to instantiate the testbench when a log
        is loaded (typically :class:`~bam.testbench.Pendulum`).
    """

    #: Whether :meth:`compute_control` carries an internal state from one call to
    #: the next (e.g. a firmware that rate-limits its internal target position).
    #: Such an actuator must be called exactly once per timestep, in
    #: chronological order. Callers that drive several *independent* simulations
    #: with the same model instance must give each of them its own state, by
    #: saving and restoring it around the call with :meth:`get_state` /
    #: :meth:`set_state` (this is what :class:`bam.mujoco.MujocoController` does).
    stateful: bool = False

    def __init__(self, testbench_class: Testbench):
        self.testbench_class = testbench_class
        self.testbench: Testbench | None = None
        self.backend: Backend = NumpyBackend()

    def set_model(self, model):
        """Attach this actuator to a model and run :meth:`initialize`.

        :param model: :class:`~bam.model.Model` instance to attach to.
        """
        self.model = model
        self.initialize()

    def reset(self, env_ids=...) -> None:
        """Reset the actuator's internal state (see :attr:`stateful`).

        When the actuator is evaluated over a batch of environments (torch
        backend), only part of them may be reset at a time. Because the state is
        expressed in terms of the environment state (positions, velocities …),
        which is not known here, the reset is expected to be *deferred*: the
        actuator records which environments are pending and applies it at the
        next :meth:`compute_control` call.

        :param env_ids: Index of the environments to reset, in any form the
            backend accepts (integer array/tensor, slice, mask). Defaults to
            ``...``, which selects all of them.
        """
        pass

    def get_state(self):
        """Return the internal state of the control law, ``None`` if stateless.

        The value is opaque and only meant to be handed back to
        :meth:`set_state`. Only relevant when :attr:`stateful` is ``True``.
        """
        return None

    def set_state(self, state) -> None:
        """Restore an internal state previously returned by :meth:`get_state`."""
        pass

    def load_log(self, log: dict):
        """Called when a log is loaded.

        Instantiates the testbench from the log metadata (mass, length, …)
        and updates actuator settings (kp, vin) from the log.

        :param log: Log dict as loaded from a processed JSON file.
        """
        self.testbench = self.testbench_class(log)

    def initialize(self):
        """Create motor parameters on the attached model.

        Called automatically by :meth:`set_model`. Must create at minimum
        ``self.model.kt`` and ``self.model.R`` as :class:`~bam.parameter.Parameter`
        instances.
        """
        raise NotImplementedError

    def control_unit(self) -> str:
        """Return the physical unit of the control signal (e.g. ``"volts"``, ``"amps"``)."""
        raise NotImplementedError

    def compute_control(
        self, q_target: ArrayLike, q: ArrayLike, dq: ArrayLike, dt: float
    ) -> ArrayLike | None:
        """Compute the control signal from the current state and target.

        The state arguments (``q_target``, ``q``, ``dq``) may be Python floats,
        numpy arrays or torch tensors; the computation is elementwise and the
        result matches their (broadcast) type. Use :class:`TorchBackend` for a
        fully vectorized, autograd-friendly evaluation.

        :param q_target: Target joint angle(s) [rad].
        :param q: Current joint angle(s) [rad].
        :param dq: Current joint velocity(ies) [rad/s].
        :param dt: Timestep [s].
        :returns: Control signal in the unit given by :meth:`control_unit`.
        """
        raise NotImplementedError

    def compute_torque(
        self,
        control: ArrayLike | None,
        torque_enable: bool,
        q: ArrayLike,
        dq: ArrayLike,
    ) -> ArrayLike:
        """Compute the motor torque from the control signal and current state.

        As with :meth:`compute_control`, the array arguments accept floats, numpy
        arrays or torch tensors and the result follows their (broadcast) type.

        :param control: Control signal (volts, amps, or Nm depending on the actuator).
        :param torque_enable: Whether the actuator is powered.
        :param q: Current joint angle(s) [rad].
        :param dq: Current joint velocity(ies) [rad/s].
        :returns: Motor torque [Nm].
        """
        raise NotImplementedError

    def get_extra_inertia(self) -> float:
        """Return the actuator's apparent inertia added to the load [kg·m²]."""
        raise NotImplementedError


class DCMotorActuator(Actuator):
    """Base class for DC motor actuators with a supply voltage and a P-gain.

    Loads ``kp`` and ``vin`` from the log metadata when :meth:`load_log` is called.

    :param testbench_class: Testbench class (see :class:`Actuator`).
    :param vin: Default supply voltage [V].
    :param kp: Default firmware proportional gain.
    """

    def __init__(
        self,
        testbench_class: Testbench,
        vin: float,
        kp: float,
    ):
        super().__init__(testbench_class)
        self.vin = vin
        self.kp = kp

    def load_log(self, log: dict):
        super().load_log(log)
        self.kp = log["kp"]
        if "vin" in log:
            self.vin = log["vin"]

    def initialize(self):
        """Create ``kt`` and ``R`` parameters on the attached model."""
        # Torque constant [Nm/A] or [V/(rad/s)]
        self.model.kt = Parameter(1.0, 0.0, 10.0)

        # Motor resistance [Ohm]
        self.model.R = Parameter(1.0, 0.0, 10.0)


class VoltageControlledActuator(DCMotorActuator):
    """Voltage-controlled servo with a firmware P-position controller.

    Implements the control law
    ``duty_cycle = clip(error_gain * kp * Δq, -max_pwm, max_pwm)``
    and the DC motor torque equation with back-EMF.

    :param testbench_class: Testbench class (see :class:`Actuator`).
    :param vin: Supply voltage [V].
    :param kp: Firmware proportional gain.
    :param error_gain: Converts ``kp * Δq`` to a duty cycle in [−1, 1].
        Depends on the servo's internal encoder resolution and gain scaling.
    :param max_pwm: Maximum duty cycle magnitude (default 1.0).
    :param max_current: Firmware current limit [A]. If not None, the firmware
        limiter is modelled in :meth:`compute_control` as a constraint on the PWM
        duty cycle that *attempts* to keep the motor current within
        ``[-max_current, max_current]``. Because the firmware can only bound the
        duty cycle (not synthesize arbitrary voltage), the limit is only reached
        when the battery voltage allows it; at high speed the back-EMF can make
        it unreachable. ``None`` (default) → no current limiting.
    """

    def __init__(
        self,
        testbench_class: Testbench,
        vin: float,
        kp: float,
        error_gain: float = 1.0,
        max_pwm: float = 1.0,
        max_current: float | None = None,
    ):
        super().__init__(testbench_class, vin, kp)
        self.error_gain = error_gain
        self.max_pwm = max_pwm
        self.max_current = max_current

    def control_unit(self) -> str:
        return "volts"

    def compute_control(
        self, q_target: ArrayLike, q: ArrayLike, dq: ArrayLike, dt: float
    ) -> ArrayLike | None:
        """Compute the voltage command from position error.

        When ``max_current`` is set, the firmware current limiter is modelled as
        a constraint on the duty cycle rather than a clamp on the output torque.
        The firmware can only act on the PWM duty cycle, so the achievable
        current is bounded by the battery voltage: solving :math:`|I| \\le
        I_\\text{max}` for :math:`I = (\\text{duty} \\cdot v_\\text{in} - k_t
        \\dot{q}) / R` gives the duty window

        .. math::

            \\frac{k_t \\dot{q} - R\\,I_\\text{max}}{v_\\text{in}}
            \\le \\text{duty} \\le
            \\frac{k_t \\dot{q} + R\\,I_\\text{max}}{v_\\text{in}}

        The commanded duty is clamped to this window (the limiter *attempt*) and
        then to the physical ``[-max_pwm, max_pwm]`` range (the battery reality),
        applied last. When back-EMF is large the window can fall outside the
        physical range, so the limiter saturates without actually holding the
        current at ``max_current`` — exactly as the real firmware behaves.

        :param q_target: Target joint angle(s) [rad].
        :param q: Current joint angle(s) [rad].
        :param dq: Current joint velocity(ies) [rad/s]. Used by the current
            limiter (back-EMF term); otherwise unused.
        :param dt: Timestep [s] (unused here).
        :returns: Voltage [V] sent to the motor.
        """
        duty_cycle = (q_target - q) * self.kp * self.error_gain

        # Firmware current limiter: bound the duty cycle so the motor current
        # I = (duty * vin - kt * dq) / R stays within [-max_current, max_current].
        # This is only an attempt: the physical PWM clamp below is applied last,
        # so if the required duty falls outside [-max_pwm, max_pwm] the current
        # limit is not actually reached (the battery cannot supply the voltage).
        if self.max_current is not None:
            back_emf = self.model.kt.value * dq
            duty_span = self.model.R.value * self.max_current / self.vin
            duty_center = back_emf / self.vin
            duty_cycle = self.backend.clamp(
                duty_cycle, duty_center - duty_span, duty_center + duty_span
            )

        # Physical PWM limit (voltage bounded by the battery) — applied last.
        duty_cycle = self.backend.clamp(duty_cycle, -self.max_pwm, self.max_pwm)
        self.duty_cycle = duty_cycle  # for logging

        return self.vin * duty_cycle

    def compute_torque(
        self,
        control: ArrayLike | None,
        torque_enable: bool,
        q: ArrayLike,
        dq: ArrayLike,
    ) -> ArrayLike:
        """Compute motor torque using the DC motor equation with back-EMF.

        :math:`\\tau = k_t V / R - k_t^2 \\dot{q} / R`

        The firmware current limit is *not* applied here: it is modelled as a
        duty-cycle constraint in :meth:`compute_control`, so the voltage
        ``control`` already reflects the (possibly saturated) current limiter and
        the torque follows directly from the DC motor equation.

        :param control: Voltage(s) [V].
        :param torque_enable: If ``False``, returns zero torque.
        :param q: Current joint angle(s) [rad] (unused here).
        :param dq: Current joint velocity(ies) [rad/s].
        :returns: Motor torque [Nm].
        """
        volts = control
        torque = self.model.kt.value * volts / self.model.R.value
        torque -= (self.model.kt.value**2) * dq / self.model.R.value
        return torque * torque_enable


class CurrentControlledActuator(DCMotorActuator):
    """Current-controlled servo with a firmware P-position controller.

    The controller outputs a target current proportional to position error,
    clipped by a user-configurable current cap, then by what the H-bridge can
    physically deliver.

    :param testbench_class: Testbench class (see :class:`Actuator`).
    :param vin: Supply voltage [V], used to compute the current saturation.
    :param kp: Firmware proportional gain.
    :param error_gain: Converts ``kp * Δq`` to a target current [A].
    :param max_pwm: Maximum duty cycle magnitude (default 1.0).
    """

    def __init__(
        self,
        testbench_class: Testbench,
        vin: float,
        kp: float,
        error_gain: float = 1.0,
        max_pwm: float = 1.0,
    ):
        super().__init__(testbench_class, vin, kp)
        self.error_gain = error_gain
        self.max_pwm = max_pwm

    def control_unit(self) -> str:
        return "amps"

    def initialize(self):
        super().initialize()
        self.model.current_limit = Parameter(1.5, 0, 3)

    def compute_control(
        self, q_target: ArrayLike, q: ArrayLike, dq: ArrayLike, dt: float
    ) -> ArrayLike | None:
        """Compute the current command from position error.

        Clips the P-controller output by the ``current_limit`` parameter, then
        by what the H-bridge can deliver given the supply voltage.

        :param q_target: Target joint angle(s) [rad].
        :param q: Current joint angle(s) [rad].
        :param dq: Current joint velocity(ies) [rad/s].
        :param dt: Timestep [s] (unused here).
        :returns: Target current [A].
        """
        kt = self.model.kt.value
        R = self.model.R.value

        # Target current using simple P controller
        current = (q_target - q) * self.kp * self.error_gain

        # Maximum current allowed by the user to avoid heating
        current = self.backend.clamp(
            current, -self.model.current_limit.value, self.model.current_limit.value
        )

        # Physical PWM limit (voltage bounded by the battery) — applied last, since
        # the firmware can't limit the current beyond it. The duty cycle feeding
        # the current, from I = (duty * vin - kt * dq) / R, is clamped and the
        # current recomputed from it.
        duty_cycle = (R * current + kt * dq) / self.vin
        duty_cycle = self.backend.clamp(duty_cycle, -self.max_pwm, self.max_pwm)
        self.duty_cycle = duty_cycle  # for logging (and the battery drop model)

        return (duty_cycle * self.vin - kt * dq) / R

    def compute_torque(
        self,
        control: ArrayLike | None,
        torque_enable: bool,
        q: ArrayLike,
        dq: ArrayLike,
    ) -> ArrayLike:
        """Compute motor torque from current command.

        :math:`\\tau = k_t I`

        :param control: Current command(s) [A].
        :param torque_enable: If ``False``, returns zero torque.
        :param q: Current joint angle(s) [rad] (unused here).
        :param dq: Current joint velocity(ies) [rad/s] (unused here).
        :returns: Motor torque [Nm].
        """
        torque = self.model.kt.value * control
        return torque * torque_enable
