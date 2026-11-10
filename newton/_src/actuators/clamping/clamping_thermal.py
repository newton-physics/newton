# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import warp as wp

from .base import ClampingBase

_ARRAY_PARAMS = (
    "max_effort",
    "torque_constant",
    "winding_resistance",
    "resistance_temperature_coefficient",
    "iron_loss_coefficient",
    "heat_capacity_winding",
    "heat_capacity_housing",
    "thermal_resistance_winding_housing",
    "thermal_resistance_housing_ambient",
    "heat_flow_housing",
    "temperature_ambient",
    "temperature_reference",
    "temperature_derating_start",
    "temperature_derating_end",
    "temperature_cutoff",
    "temperature_release",
)


@wp.func
def _evaluate_thermal_clamp(
    value: wp.float64,
    q: wp.float64,
    qd: wp.float64,
    params: wp.array2d[float],
    i: int,
    base: int,
) -> wp.float64:
    """Implicit-solve entry point; params row is ``[effort_limit]``."""
    limit = wp.float64(params[i, base])
    return wp.clamp(value, -limit, limit)


@wp.kernel(enable_backward=False)
def _thermal_effort_limit_kernel(
    max_effort: wp.array[float],
    temperature_derating_start: wp.array[float],
    temperature_derating_end: wp.array[float],
    temperature_cutoff: wp.array[float],
    temperature_winding: wp.array[float],
    cutoff_active: wp.array[wp.bool],
    effort_limit: wp.array[float],
):
    """Effort bound from the winding temperature: linear derating, zero while cut off.

    A winding at or above the cut-off temperature counts as cut off even before
    the latch updates, e.g. right after a reset or a direct temperature write.
    """
    i = wp.tid()
    t_w = temperature_winding[i]
    limit = max_effort[i]
    t_end = temperature_derating_end[i]
    if t_end < wp.inf:
        scale = wp.clamp((t_end - t_w) / (t_end - temperature_derating_start[i]), 0.0, 1.0)
        if scale > 0.0:
            limit = limit * scale
        else:
            limit = 0.0
    if cutoff_active[i] or t_w >= temperature_cutoff[i]:
        limit = 0.0
    effort_limit[i] = limit


@wp.kernel
def _thermal_clamp_kernel(
    effort_limit: wp.array[float],
    src: wp.array[float],
    dst: wp.array[float],
):
    """Clamp src efforts to ±effort_limit, write to dst."""
    i = wp.tid()
    dst[i] = wp.clamp(src[i], -effort_limit[i], effort_limit[i])


@wp.kernel(enable_backward=False)
def _thermal_update_kernel(
    applied_forces: wp.array[float],
    velocities: wp.array[float],
    vel_indices: wp.array[wp.uint32],
    torque_constant: wp.array[float],
    winding_resistance: wp.array[float],
    resistance_temperature_coefficient: wp.array[float],
    iron_loss_coefficient: wp.array[float],
    heat_capacity_winding: wp.array[float],
    heat_capacity_housing: wp.array[float],
    thermal_resistance_winding_housing: wp.array[float],
    thermal_resistance_housing_ambient: wp.array[float],
    heat_flow_housing: wp.array[float],
    temperature_ambient: wp.array[float],
    temperature_reference: wp.array[float],
    temperature_cutoff: wp.array[float],
    temperature_release: wp.array[float],
    dt: float,
    temperature_winding: wp.array[float],
    temperature_housing: wp.array[float],
    cutoff_active: wp.array[wp.bool],
    next_temperature_winding: wp.array[float],
    next_temperature_housing: wp.array[float],
    next_cutoff_active: wp.array[wp.bool],
):
    """Advance the two-node network one step with backward Euler on the conduction terms.

    The losses are evaluated at the current state. The linear network is then
    solved implicitly, so the update is stable for any ``dt``.
    """
    i = wp.tid()
    t_w = temperature_winding[i]
    t_h = temperature_housing[i]
    qd = velocities[vel_indices[i]]

    current = applied_forces[i] / torque_constant[i]
    resistance = winding_resistance[i] * (
        1.0 + resistance_temperature_coefficient[i] * (t_w - temperature_reference[i])
    )
    power = wp.max(resistance, 0.0) * current * current + iron_loss_coefficient[i] * qd * qd

    a = heat_capacity_winding[i] / dt
    b = heat_capacity_housing[i] / dt
    g = 1.0 / thermal_resistance_winding_housing[i]
    h = 1.0 / thermal_resistance_housing_ambient[i]
    rhs_w = a * t_w + power
    rhs_h = b * t_h + h * temperature_ambient[i] + heat_flow_housing[i]
    det = (a + g) * (b + g + h) - g * g
    t_w_next = (rhs_w * (b + g + h) + g * rhs_h) / det
    t_h_next = ((a + g) * rhs_h + g * rhs_w) / det

    active = cutoff_active[i]
    if t_w_next >= temperature_cutoff[i]:
        active = True
    elif t_w_next <= temperature_release[i]:
        active = False

    next_temperature_winding[i] = t_w_next
    next_temperature_housing[i] = t_h_next
    next_cutoff_active[i] = active


@wp.kernel(enable_backward=False)
def _thermal_reset_kernel(
    temperature: wp.array[float],
    mask: wp.array[wp.bool],
    temperature_winding: wp.array[float],
    temperature_housing: wp.array[float],
    cutoff_active: wp.array[wp.bool],
):
    i = wp.tid()
    if mask:
        if not mask[i]:
            return
    temperature_winding[i] = temperature[i]
    temperature_housing[i] = temperature[i]
    cutoff_active[i] = False


class ClampingThermal(ClampingBase):
    r"""Thermal motor model that derates and cuts off actuator effort.

    Each actuator carries a two-node lumped thermal network, a winding and
    a housing, heated by copper and iron losses of the effort it applied::

        I    = effort / torque_constant
        P    = winding_resistance * (1 + alpha * (T_w - temperature_reference)) * I**2
               + iron_loss_coefficient * qd**2
        C_w dT_w/dt = P - (T_w - T_h) / R_wh
        C_h dT_h/dt = (T_w - T_h) / R_wh - (T_h - temperature_ambient) / R_ha
                      + heat_flow_housing

    with ``alpha = resistance_temperature_coefficient``, ``C_w, C_h`` the
    heat capacities and ``R_wh, R_ha`` the thermal resistances. The losses are
    evaluated at the start of the step and the conduction terms are
    integrated with backward Euler, which is stable for any timestep.

    The effort is clamped to ``±effort_limit``, which is ``max_effort`` scaled
    linearly from 1 at ``temperature_derating_start`` to 0 at
    ``temperature_derating_end`` and is zero while the cut-off is active. The
    cut-off engages when the winding reaches ``temperature_cutoff`` and
    releases once it cools to ``temperature_release``. With the default
    infinite thresholds the clamp tracks temperature and limits the effort to
    ``max_effort`` only.

    Place it after :class:`ClampingDCMotor` to combine the effort-speed
    envelope with the thermal limit. The heat comes from the effort the
    actuator finally applied, after every clamp, so a clamp placed after this
    one, e.g. :class:`ClampingDCMotor` braking above its velocity limit, can
    apply effort while the cut-off is active.

    Temperatures are not differentiated: with ``requires_grad=True``,
    gradients pass through the effort limit as a constant of the step.

    Every parameter is a per-actuator array that may be rewritten between
    steps, e.g. ``temperature_ambient`` for per-world randomization or
    ``thermal_resistance_housing_ambient`` for fan or airflow cooling.
    Temperatures are in degrees Celsius; only differences enter the heat
    balance, so any offset scale works if used consistently.

    .. note::
        Inter-actuator heat exchange is not modeled; subclass and extend
        :meth:`update_state` to add it.
    """

    @dataclass
    class State(ClampingBase.State):
        """Thermal state of every actuator managed by a :class:`ClampingThermal`."""

        temperature_winding: wp.array[float] | None = None
        """Winding temperature [°C]. Shape ``(N,)``."""
        temperature_housing: wp.array[float] | None = None
        """Housing temperature [°C]. Shape ``(N,)``."""
        cutoff_active: wp.array[wp.bool] | None = None
        """Whether the thermal cut-off holds the effort at zero. Shape ``(N,)``."""
        temperature_initial: wp.array[float] | None = None
        """Temperature [°C] :meth:`reset` writes to both nodes. Refers to the
        clamp's ``temperature_ambient`` array; :meth:`assign` does not copy it."""

        def reset(self, mask: wp.array[wp.bool] | None = None) -> None:
            """Return both nodes to :attr:`temperature_initial` and release the cut-off.

            Args:
                mask: Boolean mask of length N. ``True`` entries are reset.
                    ``None`` resets all.
            """
            n = len(self.temperature_winding)
            if mask is not None and len(mask) != n:
                raise ValueError(f"mask length {len(mask)} must match the state length {n}")
            wp.launch(
                _thermal_reset_kernel,
                dim=n,
                inputs=[self.temperature_initial, mask],
                outputs=[self.temperature_winding, self.temperature_housing, self.cutoff_active],
                device=self.temperature_winding.device,
            )

        def assign(self, other: ClampingThermal.State) -> None:
            """Copy the temperatures and cut-off flags of *other* into this state.

            Args:
                other: State to copy from.
            """
            self.temperature_winding.assign(other.temperature_winding)
            self.temperature_housing.assign(other.temperature_housing)
            self.cutoff_active.assign(other.cutoff_active)

    @classmethod
    def resolve_arguments(cls, args: dict[str, Any]) -> dict[str, Any]:
        resolved = {
            "max_effort": args.get("max_effort", math.inf),
            "torque_constant": args.get("torque_constant", 1.0),
            "winding_resistance": args.get("winding_resistance", 0.0),
            "resistance_temperature_coefficient": args.get("resistance_temperature_coefficient", 0.00393),
            "iron_loss_coefficient": args.get("iron_loss_coefficient", 0.0),
            "heat_capacity_winding": args.get("heat_capacity_winding", 1.0),
            "heat_capacity_housing": args.get("heat_capacity_housing", 1.0),
            "thermal_resistance_winding_housing": args.get("thermal_resistance_winding_housing", 1.0),
            "thermal_resistance_housing_ambient": args.get("thermal_resistance_housing_ambient", 1.0),
            "heat_flow_housing": args.get("heat_flow_housing", 0.0),
            "temperature_ambient": args.get("temperature_ambient", 25.0),
            "temperature_reference": args.get("temperature_reference", 25.0),
            "temperature_derating_start": args.get("temperature_derating_start", math.inf),
            "temperature_derating_end": args.get("temperature_derating_end", math.inf),
            "temperature_cutoff": args.get("temperature_cutoff", math.inf),
        }
        resolved["temperature_release"] = args.get("temperature_release", resolved["temperature_cutoff"])

        if resolved["max_effort"] < 0:
            raise ValueError(f"max_effort must be non-negative, got {resolved['max_effort']}")
        if resolved["torque_constant"] <= 0:
            raise ValueError(f"torque_constant must be positive, got {resolved['torque_constant']}")
        for name in ("winding_resistance", "iron_loss_coefficient"):
            if resolved[name] < 0:
                raise ValueError(f"{name} must be non-negative, got {resolved[name]}")
        for name in (
            "heat_capacity_winding",
            "heat_capacity_housing",
            "thermal_resistance_winding_housing",
            "thermal_resistance_housing_ambient",
        ):
            if not resolved[name] > 0:
                raise ValueError(f"{name} must be positive, got {resolved[name]}")
        for name, value in resolved.items():
            if math.isnan(value):
                raise ValueError(f"{name} must not be NaN")
        for name in ("temperature_ambient", "temperature_reference", "heat_flow_housing"):
            if not math.isfinite(resolved[name]):
                raise ValueError(f"{name} must be finite, got {resolved[name]}")
        start, end = resolved["temperature_derating_start"], resolved["temperature_derating_end"]
        if math.isinf(start) != math.isinf(end):
            raise ValueError("temperature_derating_start and temperature_derating_end must both be set")
        if not math.isinf(end):
            if not start < end:
                raise ValueError(f"temperature_derating_start ({start}) must be below temperature_derating_end ({end})")
            if not math.isfinite(resolved["max_effort"]):
                raise ValueError("max_effort must be finite when derating is enabled")
        if resolved["temperature_release"] > resolved["temperature_cutoff"]:
            raise ValueError(
                f"temperature_release ({resolved['temperature_release']}) must not exceed "
                f"temperature_cutoff ({resolved['temperature_cutoff']})"
            )
        return resolved

    def __init__(
        self,
        max_effort: wp.array[float],
        torque_constant: wp.array[float],
        winding_resistance: wp.array[float],
        resistance_temperature_coefficient: wp.array[float],
        iron_loss_coefficient: wp.array[float],
        heat_capacity_winding: wp.array[float],
        heat_capacity_housing: wp.array[float],
        thermal_resistance_winding_housing: wp.array[float],
        thermal_resistance_housing_ambient: wp.array[float],
        heat_flow_housing: wp.array[float],
        temperature_ambient: wp.array[float],
        temperature_reference: wp.array[float],
        temperature_derating_start: wp.array[float],
        temperature_derating_end: wp.array[float],
        temperature_cutoff: wp.array[float],
        temperature_release: wp.array[float],
    ):
        """Initialize the thermal clamp.

        Args:
            max_effort: Effort limit of a cold actuator [N or N·m]. Shape ``(N,)``.
            torque_constant: Joint-side effort per winding current, including
                gearing and the number of motors driving the joint
                [N·m/A or N/A]. Shape ``(N,)``.
            winding_resistance: Electrical winding resistance at
                ``temperature_reference`` [Ω]. Shape ``(N,)``.
            resistance_temperature_coefficient: Relative change of the winding
                resistance per degree [1/K]; 0.00393 for copper. Shape ``(N,)``.
            iron_loss_coefficient: Speed-dependent loss per squared joint
                speed [W·s²/rad² or W·s²/m²]. Shape ``(N,)``.
            heat_capacity_winding: Heat capacity of the winding [J/K]. Shape ``(N,)``.
            heat_capacity_housing: Heat capacity of the housing [J/K]. Shape ``(N,)``.
            thermal_resistance_winding_housing: Thermal resistance from winding
                to housing [K/W]. Shape ``(N,)``.
            thermal_resistance_housing_ambient: Thermal resistance from housing
                to ambient air [K/W]; ``inf`` for an insulated housing. Shape ``(N,)``.
            heat_flow_housing: Constant heat into the housing, e.g. from the
                motor driver [W]. Shape ``(N,)``.
            temperature_ambient: Ambient temperature [°C]. Shape ``(N,)``.
            temperature_reference: Temperature at which ``winding_resistance``
                is specified [°C]. Shape ``(N,)``.
            temperature_derating_start: Winding temperature at which the effort
                limit starts to fall [°C]; ``inf`` disables derating. Shape ``(N,)``.
            temperature_derating_end: Winding temperature at which the effort
                limit reaches zero [°C]. Shape ``(N,)``.
            temperature_cutoff: Winding temperature that cuts the effort to zero
                [°C]; ``inf`` disables the cut-off. Shape ``(N,)``.
            temperature_release: Winding temperature below which a cut-off
                releases [°C]. Shape ``(N,)``.
        """
        values = locals()
        shape = max_effort.shape
        if max_effort.ndim != 1:
            raise ValueError(f"max_effort must be one-dimensional, got shape {shape}")
        for name in _ARRAY_PARAMS:
            array = values[name]
            if array.shape != shape:
                raise ValueError(f"{name} shape {array.shape} must match max_effort shape {shape}")
            setattr(self, name, array)
        self.effort_limit: wp.array[float] = wp.clone(max_effort)
        """Effort bound applied in the last step [N or N·m]: ``max_effort``
        after derating and cut-off. Refreshed from the state at the start of
        every step. Shape ``(N,)``."""

    def finalize(self, device: wp.Device, num_actuators: int) -> None:
        if len(self.max_effort) != num_actuators:
            raise ValueError(f"ClampingThermal has {len(self.max_effort)} entries but the actuator has {num_actuators}")

    def is_stateful(self) -> bool:
        return True

    def state(self, num_actuators: int, device: wp.Device) -> ClampingThermal.State:
        state = ClampingThermal.State(
            temperature_winding=wp.zeros(num_actuators, dtype=float, device=device),
            temperature_housing=wp.zeros(num_actuators, dtype=float, device=device),
            cutoff_active=wp.zeros(num_actuators, dtype=wp.bool, device=device),
            temperature_initial=self.temperature_ambient,
        )
        state.reset()
        return state

    evaluate_clamp = _evaluate_thermal_clamp

    def param_width(self) -> int:
        return 1

    def bind_params(self, block: wp.array2d[float]) -> None:
        block[:, 0].assign(self.effort_limit)
        self.effort_limit = block[:, 0]

    def prepare(
        self,
        state: ClampingThermal.State,
        positions: wp.array[float],
        velocities: wp.array[float],
        pos_indices: wp.array[wp.uint32],
        vel_indices: wp.array[wp.uint32],
        dt: float,
        device: wp.Device | None = None,
    ) -> None:
        wp.launch(
            _thermal_effort_limit_kernel,
            dim=len(self.max_effort),
            inputs=[
                self.max_effort,
                self.temperature_derating_start,
                self.temperature_derating_end,
                self.temperature_cutoff,
                state.temperature_winding,
                state.cutoff_active,
            ],
            outputs=[self.effort_limit],
            device=device,
        )

    def modify_forces(
        self,
        src_forces: wp.array[float],
        dst_forces: wp.array[float],
        positions: wp.array[float],
        velocities: wp.array[float],
        pos_indices: wp.array[wp.uint32],
        vel_indices: wp.array[wp.uint32],
        device: wp.Device | None = None,
    ) -> None:
        wp.launch(
            _thermal_clamp_kernel,
            dim=len(src_forces),
            inputs=[self.effort_limit, src_forces],
            outputs=[dst_forces],
            device=device,
        )

    def update_state(
        self,
        current_state: ClampingThermal.State,
        next_state: ClampingThermal.State,
        applied_forces: wp.array[float],
        positions: wp.array[float],
        velocities: wp.array[float],
        pos_indices: wp.array[wp.uint32],
        vel_indices: wp.array[wp.uint32],
        dt: float,
        device: wp.Device | None = None,
    ) -> None:
        wp.launch(
            _thermal_update_kernel,
            dim=len(applied_forces),
            inputs=[
                applied_forces,
                velocities,
                vel_indices,
                self.torque_constant,
                self.winding_resistance,
                self.resistance_temperature_coefficient,
                self.iron_loss_coefficient,
                self.heat_capacity_winding,
                self.heat_capacity_housing,
                self.thermal_resistance_winding_housing,
                self.thermal_resistance_housing_ambient,
                self.heat_flow_housing,
                self.temperature_ambient,
                self.temperature_reference,
                self.temperature_cutoff,
                self.temperature_release,
                float(dt),
                current_state.temperature_winding,
                current_state.temperature_housing,
                current_state.cutoff_active,
            ],
            outputs=[
                next_state.temperature_winding,
                next_state.temperature_housing,
                next_state.cutoff_active,
            ],
            device=device,
        )
