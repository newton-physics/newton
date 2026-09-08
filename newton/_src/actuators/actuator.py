# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import warnings
from dataclasses import dataclass, fields
from typing import Any

import numpy as np
import warp as wp

from .clamping.base import ClampingBase
from .delay import Delay
from .drives.base import DriveBase
from .effort_mode_explicit import _EffortModeExplicit
from .effort_mode_implicit import ImplicitOptions, ResponseOracle, _EffortModeImplicit

_DEPRECATED_UNSET = object()
_CONTROLLER_KEYWORD_DEPRECATION_MSG = (
    "Actuator(controller=...) is deprecated in Newton 1.6; use Actuator(drive=...) instead."
)
_CONTROLLER_ATTRIBUTE_DEPRECATION_MSG = "Actuator.controller is deprecated in Newton 1.6; use Actuator.drive instead."
_CONTROLLER_STATE_DEPRECATION_MSG = (
    "Actuator.State.controller_state is deprecated in Newton 1.6; use drive_state instead."
)


@wp.kernel
def _scatter_add_kernel(
    forces: wp.array[float],
    computed_forces: wp.array[float],
    indices: wp.array[wp.uint32],
    output: wp.array[float],
    computed_output: wp.array[float],
):
    """Scatter-add effort into output; optionally scatter computed effort too."""
    i = wp.tid()
    idx = indices[i]
    output[idx] = output[idx] + forces[i]
    if computed_output:
        computed_output[idx] = computed_output[idx] + computed_forces[i]


def _assign_state_value(dst: Any, src: Any, name: str) -> None:
    """Copy a supported state value without replacing its storage."""
    if dst is None and src is None:
        return
    if dst is None or src is None:
        raise ValueError(f"Cannot assign '{name}': present in one state and missing in the other.")

    dst_is_warp = isinstance(dst, wp.array)
    src_is_warp = isinstance(src, wp.array)
    if dst_is_warp or src_is_warp:
        if not (dst_is_warp and src_is_warp):
            raise ValueError(f"Cannot assign '{name}': a Warp array in one state and not in the other.")
        dst.assign(src)
        return

    dst_is_torch = type(dst).__module__.startswith("torch")
    src_is_torch = type(src).__module__.startswith("torch")
    if dst_is_torch or src_is_torch:
        if not (dst_is_torch and src_is_torch):
            raise ValueError(f"Cannot assign '{name}': a Torch tensor in one state and not in the other.")
        if dst.shape != src.shape:
            raise ValueError(f"Cannot assign '{name}': tensor shapes differ ({dst.shape} and {src.shape}).")
        import torch

        with torch.inference_mode():
            dst.copy_(src)
        return

    raise ValueError(f"Cannot assign '{name}': expected Warp arrays or Torch tensors.")


def _assign_component_state(dst: Any, src: Any, name: str) -> None:
    """Copy one actuator component state from *src* into *dst*.

    Args:
        dst: Component state to copy into.
        src: Component state to copy from.
        name: Component name, used in error messages.

    Raises:
        ValueError: The two actuator states have incompatible components or
            fields.
        NotImplementedError: A custom state is not a dataclass and does not
            implement ``assign()``.
    """
    if dst is None and src is None:
        return
    if dst is None or src is None:
        raise ValueError(f"Cannot assign '{name}': one state has it allocated and the other does not.")
    if type(dst) is not type(src):
        raise ValueError(f"Cannot assign '{name}': state types differ ({type(dst).__name__} and {type(src).__name__}).")

    custom_assign = getattr(dst, "assign", None)
    if custom_assign is not None:
        custom_assign(src)
        return

    if "__dataclass_fields__" not in type(dst).__dict__:
        raise NotImplementedError(f"{type(dst).__qualname__} must be decorated with @dataclass or implement assign")

    state_fields = fields(dst)
    field_names = {field.name for field in state_fields}
    attributes = set(getattr(dst, "__dict__", ())) | set(getattr(src, "__dict__", ()))
    undeclared = attributes - field_names
    if undeclared:
        names = ", ".join(sorted(undeclared))
        raise ValueError(f"Cannot assign '{name}': undeclared state attributes: {names}.")

    for field in state_fields:
        _assign_state_value(getattr(dst, field.name), getattr(src, field.name), f"{name}.{field.name}")


class Actuator:
    """Composed actuator: delay → drive → clamping.

    An actuator reads from simulation state/control arrays, optionally
    delays command inputs, computes effort via a drive, applies clamping
    (effort limits, saturation, etc.), and **accumulates** the
    result into the output array (scatter-add).  The caller must zero the
    output array before stepping actuators.

    Usage::

        actuator = Actuator(
            indices=indices,
            drive=DrivePD(kp=kp, kd=kd),
            delay=Delay(delay_steps=wp.array([5, 5], dtype=wp.int32), max_delay=5),
            clamping=[ClampingMaxEffort(max_effort=max_effort)],
        )

        # Simulation loop
        actuator.step(sim_state, sim_control, state_a, state_b, dt=0.01)

    Effort is computed explicitly by default (control law evaluated at the
    current state, zero-order hold over the step).
    """

    @dataclass
    class State:
        """Composed state for an :class:`Actuator`.

        Holds the delay state (if a delay is present) and the drive
        state. Clamping objects are stateless.
        """

        delay_state: Delay.State | None = None
        """Delay buffer state, or ``None`` if no delay is used."""
        drive_state: DriveBase.State | None = None
        """Drive-specific state, or ``None`` if stateless."""

        def __init__(
            self,
            delay_state: Delay.State | None = None,
            drive_state: DriveBase.State | object | None = _DEPRECATED_UNSET,
            *,
            controller_state: DriveBase.State | object | None = _DEPRECATED_UNSET,
        ) -> None:
            """Initialize composed actuator state.

            Args:
                delay_state: Delay buffer state, or ``None`` if no delay is used.
                drive_state: Drive-specific state, or ``None`` if stateless.
                controller_state: Deprecated in Newton 1.6; use ``drive_state``.
            """
            if controller_state is not _DEPRECATED_UNSET:
                if drive_state is not _DEPRECATED_UNSET:
                    raise TypeError("Specify only one of 'drive_state' and deprecated 'controller_state'.")
                warnings.warn(_CONTROLLER_STATE_DEPRECATION_MSG, DeprecationWarning, stacklevel=2)
                drive_state = controller_state

            self.delay_state = delay_state
            self.drive_state = None if drive_state is _DEPRECATED_UNSET else drive_state

        @property
        def controller_state(self) -> DriveBase.State | None:
            """Deprecated alias for :attr:`drive_state`.

            .. deprecated:: 1.6
                Use :attr:`drive_state` instead.
            """
            warnings.warn(_CONTROLLER_STATE_DEPRECATION_MSG, DeprecationWarning, stacklevel=2)
            return self.drive_state

        @controller_state.setter
        def controller_state(self, value: DriveBase.State | None) -> None:
            warnings.warn(_CONTROLLER_STATE_DEPRECATION_MSG, DeprecationWarning, stacklevel=2)
            self.drive_state = value

        def reset(self, mask: wp.array[wp.bool] | None = None) -> None:
            """Reset composed state.

            Args:
                mask: Boolean mask of length N. ``True`` entries are reset.
                    ``None`` resets all.
            """
            if self.delay_state is not None:
                self.delay_state.reset(mask)
            if self.drive_state is not None:
                self.drive_state.reset(mask)

        def assign(self, other: Actuator.State) -> None:
            """Copy the state held by *other* into this one.

            A CUDA graph records buffer addresses rather than the caller's
            Python names. Assigning at the boundary of an odd-length captured
            region, in place of its final state swap, preserves the advanced
            state for the next replay::

                for i in range(steps):
                    control.joint_f.zero_()
                    actuator.step(state, control, state_0, state_1, dt=0.01)
                    if steps % 2 == 1 and i == steps - 1:
                        state_0.assign(state_1)
                    else:
                        state_0, state_1 = state_1, state_0

            Args:
                other: State to copy from.

            Raises:
                ValueError: The two states do not hold the same components.
                NotImplementedError: A custom state does not implement
                    assignment.
            """
            _assign_component_state(self.delay_state, other.delay_state, "delay_state")
            _assign_component_state(self.drive_state, other.drive_state, "drive_state")

    def __init__(
        self,
        indices: wp.array[wp.uint32],
        drive: DriveBase | None = None,
        delay: Delay | None = None,
        clamping: list[ClampingBase] | None = None,
        pos_indices: wp.array[wp.uint32] | None = None,
        target_pos_indices: wp.array[wp.uint32] | None = None,
        effort_indices: wp.array[wp.uint32] | None = None,
        state_pos_attr: str = "joint_q",
        state_vel_attr: str = "joint_qd",
        control_target_pos_attr: str | None = "joint_target_q",
        control_target_vel_attr: str | None = "joint_target_qd",
        control_feedforward_attr: str | None = "joint_act",
        control_output_attr: str = "joint_f",
        control_computed_output_attr: str | None = None,
        requires_grad: bool = False,
        *,
        controller: DriveBase | object | None = _DEPRECATED_UNSET,
    ):
        """Initialize actuator.

        Args:
            indices: DOF indices into velocity-shaped arrays (velocities,
                velocity targets, feedforward, effort output). Shape ``(N,)``.
            drive: Drive that computes raw effort.
            delay: Optional Delay instance for input delay.
            clamping: List of Clamping objects (post-drive effort bounds).
            pos_indices: Indices into coordinate-shaped arrays (positions =
                ``state.joint_q``). Defaults to *indices*. Differs from
                *indices* when position and velocity arrays have different
                layouts (e.g. floating-base or ball-joint articulations).
            target_pos_indices: Indices into ``control.joint_target_q``.
                Defaults to *pos_indices* when
                :attr:`newton.use_coord_layout_targets` is ``True`` (coord
                layout), otherwise to *indices* (legacy DOF layout). The flag is
                read once here, so toggling ``newton.use_coord_layout_targets``
                after construction does not change ``target_pos_indices``.
            effort_indices: DOF indices into effort output arrays. Defaults to
                *indices*. Differs from *indices* for coupled transmissions
                or tendon-driven joints.
            state_pos_attr: Attribute on sim_state for positions.
            state_vel_attr: Attribute on sim_state for velocities.
            control_target_pos_attr: Attribute on sim_control for target positions.
                ``None`` selects the default ``"joint_target_q"``.
            control_target_vel_attr: Attribute on sim_control for target velocities.
                ``None`` selects the default ``"joint_target_qd"``.
            control_feedforward_attr: Attribute on sim_control for feedforward effort. None to skip.
            control_output_attr: Attribute on sim_control for clamped output effort.
            control_computed_output_attr: Attribute on sim_control for raw (pre-clamp)
                effort. None to skip writing computed effort.
            requires_grad: Allocate intermediate arrays with gradient support
                for differentiable simulation.
            controller: Deprecated in Newton 1.6; use ``drive`` instead.
        """
        if controller is not _DEPRECATED_UNSET:
            if drive is not None:
                raise TypeError("Specify only one of 'drive' and deprecated 'controller'.")
            warnings.warn(_CONTROLLER_KEYWORD_DEPRECATION_MSG, DeprecationWarning, stacklevel=2)
            drive = controller
        if drive is None:
            raise TypeError("Actuator() missing required argument: 'drive'")

        self.indices = indices
        self.pos_indices = pos_indices if pos_indices is not None else indices
        if target_pos_indices is not None:
            self.target_pos_indices = target_pos_indices
        else:
            import newton  # noqa: PLC0415

            self.target_pos_indices = self.pos_indices if newton.use_coord_layout_targets else indices
        self.effort_indices = effort_indices if effort_indices is not None else indices
        if self.pos_indices.shape != indices.shape:
            raise ValueError(f"pos_indices shape {self.pos_indices.shape} must match indices shape {indices.shape}")
        if self.target_pos_indices.shape != indices.shape:
            raise ValueError(
                f"target_pos_indices shape {self.target_pos_indices.shape} must match indices shape {indices.shape}"
            )
        if self.effort_indices.shape != indices.shape:
            raise ValueError(
                f"effort_indices shape {self.effort_indices.shape} must match indices shape {indices.shape}"
            )
        self.drive = drive
        self.delay = delay
        self.clamping = clamping or []
        self.num_actuators = len(indices)

        self.state_pos_attr = state_pos_attr
        self.state_vel_attr = state_vel_attr
        # These used to default to None and resolve against the target layout.
        # Normalize so callers still passing None explicitly keep working
        # instead of tripping getattr() with a non-string name in step().
        self.control_target_pos_attr = "joint_target_q" if control_target_pos_attr is None else control_target_pos_attr
        self.control_target_vel_attr = "joint_target_qd" if control_target_vel_attr is None else control_target_vel_attr
        self.control_feedforward_attr = control_feedforward_attr
        self.control_output_attr = control_output_attr
        self.control_computed_output_attr = control_computed_output_attr

        self.device = indices.device
        self.requires_grad = requires_grad
        self._sequential_indices = wp.array(np.arange(self.num_actuators, dtype=np.uint32), device=self.device)
        self._computed_forces = wp.zeros(
            self.num_actuators, dtype=wp.float32, device=self.device, requires_grad=requires_grad
        )
        self._applied_forces = wp.zeros(
            self.num_actuators, dtype=wp.float32, device=self.device, requires_grad=requires_grad
        )

        drive.finalize(self.device, self.num_actuators)
        if delay is not None:
            delay.finalize(self.device, self.num_actuators, requires_grad=requires_grad)
        for clamp in self.clamping:
            clamp.finalize(self.device, self.num_actuators)

        self._effort_mode = _EffortModeExplicit(drive, self.clamping, self.device)

    @property
    def controller(self) -> DriveBase:
        """Deprecated alias for :attr:`drive`.

        .. deprecated:: 1.6
            Use :attr:`drive` instead.
        """
        warnings.warn(_CONTROLLER_ATTRIBUTE_DEPRECATION_MSG, DeprecationWarning, stacklevel=2)
        return self.drive

    @controller.setter
    def controller(self, value: DriveBase) -> None:
        warnings.warn(_CONTROLLER_ATTRIBUTE_DEPRECATION_MSG, DeprecationWarning, stacklevel=2)
        self.drive = value

    # To achieve public API Actuator.ImplicitOptions.
    # Defining ImplicitOptions inside Actuator would create a circular import issue.
    ImplicitOptions = ImplicitOptions

    def set_effort_mode_implicit(
        self,
        response: ResponseOracle,
        options: Actuator.ImplicitOptions | None = None,
    ) -> None:
        """Switch effort computation to implicit mode.

        The control law is solved against the predicted end-of-step state
        before the solver runs. See :ref:`effort-modes` for details on the
        computation of effort in the implicit mode, its caveats, and its expected use.

        Args:
            response: :class:`~newton.actuators.ResponseOracle` supplying the
                coupled effective inverse mass [1/kg or 1/(kg·m²)]. Refresh it
                once per step before :meth:`step`.
            options: Solver options; defaults to :class:`Actuator.ImplicitOptions`.

        Raises:
            NotImplementedError: The actuator was built with ``requires_grad=True``.
                The implicit solve is not differentiable.
        """
        if self.requires_grad:
            raise NotImplementedError(
                "Implicit actuation is not differentiable: the Newton solve has no adjoint, "
                "and the neural drives open their own wp.Tape, which cannot nest inside "
                "an outer tape. Build the Actuator with requires_grad=False."
            )
        self._effort_mode = _EffortModeImplicit(
            self.drive,
            self.clamping,
            response,
            options,
            self.num_actuators,
            self.device,
            self.indices,
        )

    def set_effort_mode_explicit(self) -> None:
        """Switch effort computation back to the default explicit mode."""
        self._effort_mode = _EffortModeExplicit(self.drive, self.clamping, self.device)

    def is_stateful(self) -> bool:
        """Return True if the delay or drive maintains internal state."""
        return self.delay is not None or self.drive.is_stateful()

    def is_graphable(self) -> bool:
        """Return True if all components can be captured in a CUDA graph."""
        return self._effort_mode.is_graphable()

    def state(self) -> Actuator.State | None:
        """Return a new composed state, or None if fully stateless."""
        if not self.is_stateful():
            return None
        return Actuator.State(
            delay_state=(self.delay.state(self.num_actuators, self.device) if self.delay is not None else None),
            drive_state=(self.drive.state(self.num_actuators, self.device) if self.drive.is_stateful() else None),
        )

    def step(
        self,
        sim_state: Any,
        sim_control: Any,
        current_act_state: Actuator.State | None = None,
        next_act_state: Actuator.State | None = None,
        dt: float | None = None,
    ) -> None:
        """Execute one control step.

        1. **Delay read** — read per-DOF delayed targets from
           ``current_state`` (falls back to current targets when
           the buffer is empty).
        2. **Effort** — raw effort into ``_computed_forces`` (explicit control
           law, or the implicit end-of-step solve).
        3. **Clamping** — bounded effort into ``_applied_forces``. Explicit
           clamps after the drive law; implicit enforces them inside the
           solve.
        4. **Scatter-add** — *accumulate* applied (and optionally computed)
           effort into the output array.  The caller must zero the output
           (e.g. ``control.joint_f.zero_()``) before looping over actuators.
        5. **State updates** — drive state update, then delay
           buffer write (push current targets into ``next_state``).

        Args:
            sim_state: Simulation state with position/velocity arrays.
            sim_control: Control structure with target/output arrays.
            current_act_state: Current composed state (None if stateless).
            next_act_state: Next composed state (None if stateless).
            dt: Timestep [s].
        """
        if self.is_stateful() and (current_act_state is None or next_act_state is None):
            raise ValueError(
                "Stateful actuator requires both current_act_state and next_act_state; create them via actuator.state()"
            )

        positions = getattr(sim_state, self.state_pos_attr)
        velocities = getattr(sim_state, self.state_vel_attr)

        orig_target_pos = getattr(sim_control, self.control_target_pos_attr)
        orig_target_vel = getattr(sim_control, self.control_target_vel_attr)
        orig_feedforward = None
        if self.control_feedforward_attr is not None:
            orig_feedforward = getattr(sim_control, self.control_feedforward_attr, None)

        target_pos = orig_target_pos
        target_vel = orig_target_vel
        feedforward = orig_feedforward
        target_pos_indices = self.target_pos_indices
        target_vel_indices = self.indices

        # --- 1. Delay read (from current_state) ---
        if self.delay is not None:
            target_pos, target_vel, feedforward = self.delay.get_delayed_targets(
                orig_target_pos,
                orig_target_vel,
                orig_feedforward,
                self.target_pos_indices,
                self.indices,
                current_act_state.delay_state,
            )
            target_pos_indices = self._sequential_indices
            target_vel_indices = self._sequential_indices

        # --- 2+3. Effort mode: compute raw effort and clamp ---
        drive_state = current_act_state.drive_state if current_act_state else None
        output_forces = self._effort_mode.compute_force(
            sim_state,
            positions,
            velocities,
            target_pos,
            target_vel,
            feedforward,
            self.pos_indices,
            self.indices,
            target_pos_indices,
            target_vel_indices,
            self._computed_forces,
            self._applied_forces,
            drive_state,
            dt,
        )

        # --- 4. Scatter-add to output ---
        applied_output = getattr(sim_control, self.control_output_attr)
        computed_output = None
        if (
            self.control_computed_output_attr is not None
            and self.control_computed_output_attr != self.control_output_attr
        ):
            computed_output = getattr(sim_control, self.control_computed_output_attr)
        wp.launch(
            kernel=_scatter_add_kernel,
            dim=self.num_actuators,
            inputs=[output_forces, self._computed_forces, self.effort_indices],
            outputs=[applied_output, computed_output],
            device=self.device,
        )

        # --- 5. State updates (write to next_state) ---
        if self.drive.is_stateful():
            self.drive.update_state(
                current_act_state.drive_state,
                next_act_state.drive_state,
            )
        if self.delay is not None:
            self.delay.update_state(
                orig_target_pos,
                orig_target_vel,
                orig_feedforward,
                self.target_pos_indices,
                self.indices,
                current_act_state.delay_state,
                next_act_state.delay_state,
            )
