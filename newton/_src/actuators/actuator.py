# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import warnings
from dataclasses import dataclass, fields
from typing import Any

import warp as wp

from .battery import Battery
from .clamping.base import ClampingBase
from .drives.base import DriveBase
from .effort_mode_explicit import _EffortModeExplicit
from .effort_mode_implicit import ImplicitOptions, JointSpaceResponse, _EffortModeImplicit
from .input_processors.base import InputProcessorBase
from .input_processors.input_processor_delay import InputProcessorDelay

_DEPRECATED_UNSET = object()
_DELAY_KEYWORD_DEPRECATION_MSG = (
    "Actuator(delay=...) is deprecated in Newton 1.7; use Actuator(input_processors=[delay]) instead."
)
_DELAY_ATTRIBUTE_DEPRECATION_MSG = "Actuator.delay is deprecated in Newton 1.7; use Actuator.input_processors instead."
_DELAY_UPDATE_STATE_OVERRIDE_DEPRECATION_MSG = (
    "Overriding InputProcessorDelay.update_state with the (target_pos, target_vel, feedforward, pos_indices, "
    "vel_indices, current_state, next_state) signature is deprecated in Newton 1.7; override "
    "update_state(inputs, current_state, next_state) instead."
)
_DELAY_STATE_DEPRECATION_MSG = (
    "Actuator.State.delay_state is deprecated in Newton 1.7; use input_processor_states instead."
)
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


def _legacy_delay_updates(processors: list[InputProcessorBase]) -> list[InputProcessorDelay]:
    """Return the delays whose ``update_state`` override still takes the deprecated 7-argument form."""
    legacy = [p for p in processors if isinstance(p, InputProcessorDelay) and p._overrides_legacy_update_state()]
    if legacy:
        warnings.warn(_DELAY_UPDATE_STATE_OVERRIDE_DEPRECATION_MSG, DeprecationWarning, stacklevel=3)
    return legacy


class Actuator:
    """Composed actuator: input processors → drive → clamping.

    An actuator reads from simulation state/control arrays, optionally
    transforms them with input processors (e.g. a command delay), computes
    effort via a drive, applies clamping (effort limits, saturation, etc.),
    and **accumulates** the result into the output array (scatter-add).  The
    caller must zero the output array before stepping actuators.

    Usage::

        actuator = Actuator(
            indices=indices,
            drive=DrivePD(kp=kp, kd=kd),
            input_processors=[InputProcessorDelay(delay_steps=wp.array([5, 5], dtype=wp.int32), max_delay=5)],
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

        Holds one state per input processor and the drive state.
        Clamping objects are stateless.
        """

        input_processor_states: list[InputProcessorBase.State | None] | None = None
        """One state per input processor, in processor order (``None`` for
        stateless processors), or ``None`` if there are no input processors."""
        drive_state: DriveBase.State | None = None
        """Drive-specific state, or ``None`` if stateless."""

        def __init__(
            self,
            delay_state: InputProcessorDelay.State | None = None,
            drive_state: DriveBase.State | object | None = _DEPRECATED_UNSET,
            *,
            input_processor_states: list[InputProcessorBase.State | None] | None = None,
            controller_state: DriveBase.State | object | None = _DEPRECATED_UNSET,
        ) -> None:
            """Initialize composed actuator state.

            Args:
                delay_state: Deprecated in Newton 1.7; use ``input_processor_states``.
                drive_state: Drive-specific state, or ``None`` if stateless.
                input_processor_states: One state per input processor, in
                    processor order (``None`` for stateless processors).
                controller_state: Deprecated in Newton 1.6; use ``drive_state``.
            """
            if controller_state is not _DEPRECATED_UNSET:
                if drive_state is not _DEPRECATED_UNSET:
                    raise TypeError("Specify only one of 'drive_state' and deprecated 'controller_state'.")
                warnings.warn(_CONTROLLER_STATE_DEPRECATION_MSG, DeprecationWarning, stacklevel=2)
                drive_state = controller_state
            if delay_state is not None:
                if input_processor_states is not None:
                    raise TypeError("Specify only one of 'input_processor_states' and deprecated 'delay_state'.")
                warnings.warn(_DELAY_STATE_DEPRECATION_MSG, DeprecationWarning, stacklevel=2)
                input_processor_states = [delay_state]

            self.input_processor_states = input_processor_states
            self.drive_state = None if drive_state is _DEPRECATED_UNSET else drive_state

        @property
        def delay_state(self) -> InputProcessorDelay.State | None:
            """Deprecated alias for the first :class:`InputProcessorDelay` state in :attr:`input_processor_states`.

            .. deprecated:: 1.7
                Use :attr:`input_processor_states` instead.
            """
            warnings.warn(_DELAY_STATE_DEPRECATION_MSG, DeprecationWarning, stacklevel=2)
            return next(
                (s for s in self.input_processor_states or () if isinstance(s, InputProcessorDelay.State)), None
            )

        @delay_state.setter
        def delay_state(self, value: InputProcessorDelay.State | None) -> None:
            warnings.warn(_DELAY_STATE_DEPRECATION_MSG, DeprecationWarning, stacklevel=2)
            states = self.input_processor_states or []
            for i, s in enumerate(states):
                if isinstance(s, InputProcessorDelay.State):
                    states[i] = value
                    return
            if value is not None:
                self.input_processor_states = [value, *states]

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
            for state in self.input_processor_states or ():
                if state is not None:
                    state.reset(mask)
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
            own_states = self.input_processor_states or []
            other_states = other.input_processor_states or []
            if len(own_states) != len(other_states):
                raise ValueError(
                    f"Cannot assign 'input_processor_states': {len(own_states)} and {len(other_states)} entries."
                )
            for i, (dst, src) in enumerate(zip(own_states, other_states, strict=True)):
                _assign_component_state(dst, src, f"input_processor_states[{i}]")
            _assign_component_state(self.drive_state, other.drive_state, "drive_state")

    def __init__(
        self,
        indices: wp.array[wp.uint32],
        drive: DriveBase | None = None,
        delay: InputProcessorDelay | None = None,
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
        input_processors: list[InputProcessorBase] | None = None,
        controller: DriveBase | object | None = _DEPRECATED_UNSET,
    ):
        """Initialize actuator.

        Args:
            indices: DOF indices into velocity-shaped arrays (velocities,
                velocity targets, feedforward, effort output). Shape ``(N,)``.
            drive: Drive that computes raw effort.
            delay: Deprecated in Newton 1.7; pass the delay in
                ``input_processors`` instead.
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
            input_processors: Input processors applied in list order to the
                state and command inputs before the drive (e.g.
                :class:`InputProcessorDelay`).
            controller: Deprecated in Newton 1.6; use ``drive`` instead.
        """
        if controller is not _DEPRECATED_UNSET:
            if drive is not None:
                raise TypeError("Specify only one of 'drive' and deprecated 'controller'.")
            warnings.warn(_CONTROLLER_KEYWORD_DEPRECATION_MSG, DeprecationWarning, stacklevel=2)
            drive = controller
        if drive is None:
            raise TypeError("Actuator() missing required argument: 'drive'")
        input_processors = list(input_processors or [])
        if delay is not None:
            warnings.warn(_DELAY_KEYWORD_DEPRECATION_MSG, DeprecationWarning, stacklevel=2)
            input_processors.insert(0, delay)

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
        self.input_processors = input_processors
        """Input processors, applied in list order before the drive."""
        self._legacy_delay_updates = _legacy_delay_updates(input_processors)
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
        self._computed_forces = wp.zeros(
            self.num_actuators, dtype=wp.float32, device=self.device, requires_grad=requires_grad
        )
        self._applied_forces = wp.zeros(
            self.num_actuators, dtype=wp.float32, device=self.device, requires_grad=requires_grad
        )

        drive.finalize(self.device, self.num_actuators)
        for processor in self.input_processors:
            processor.finalize(self.device, self.num_actuators, requires_grad=requires_grad)
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

    @property
    def delay(self) -> InputProcessorDelay | None:
        """Deprecated alias for the first :class:`InputProcessorDelay` in :attr:`input_processors`.

        .. deprecated:: 1.7
            Use :attr:`input_processors` instead.
        """
        warnings.warn(_DELAY_ATTRIBUTE_DEPRECATION_MSG, DeprecationWarning, stacklevel=2)
        return next((p for p in self.input_processors if isinstance(p, InputProcessorDelay)), None)

    @delay.setter
    def delay(self, value: InputProcessorDelay | None) -> None:
        warnings.warn(_DELAY_ATTRIBUTE_DEPRECATION_MSG, DeprecationWarning, stacklevel=2)
        processors = [p for p in self.input_processors if not isinstance(p, InputProcessorDelay)]
        if value is not None:
            value.finalize(self.device, self.num_actuators, requires_grad=self.requires_grad)
            processors.insert(0, value)
        self.input_processors = processors
        self._legacy_delay_updates = _legacy_delay_updates(processors)

    # To achieve public API Actuator.ImplicitOptions.
    # Defining ImplicitOptions inside Actuator would create a circular import issue.
    ImplicitOptions = ImplicitOptions

    def set_effort_mode_implicit(
        self,
        response: JointSpaceResponse,
        options: Actuator.ImplicitOptions | None = None,
    ) -> None:
        """Switch effort computation to implicit mode.

        The control law is solved against the predicted end-of-step state
        before the solver runs. See :ref:`effort-modes` for details on the
        computation of effort in the implicit mode, its caveats, and its expected use.

        Args:
            response: :class:`~newton.actuators.JointSpaceResponse` supplying the
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
        """Return True if an input processor or the drive maintains internal state."""
        return any(p.is_stateful() for p in self.input_processors) or self.drive.is_stateful()

    def is_graphable(self) -> bool:
        """Return True if all components can be captured in a CUDA graph."""
        return all(p.is_graphable() for p in self.input_processors) and self._effort_mode.is_graphable()

    def state(self) -> Actuator.State | None:
        """Return a new composed state, or None if fully stateless."""
        if not self.is_stateful():
            return None
        return Actuator.State(
            input_processor_states=(
                [p.state(self.num_actuators, self.device) if p.is_stateful() else None for p in self.input_processors]
                or None
            ),
            drive_state=(self.drive.state(self.num_actuators, self.device) if self.drive.is_stateful() else None),
        )

    def step(
        self,
        sim_state: Any,
        sim_control: Any,
        current_act_state: Actuator.State | None = None,
        next_act_state: Actuator.State | None = None,
        dt: float | None = None,
        *,
        battery: Battery | None = None,
    ) -> None:
        """Execute one control step.

        1. **Input processors** — transform the state and command inputs
           in list order (e.g. read delayed targets from ``current_state``),
           and write each processor's ``next_state`` (e.g. push the current
           targets into the delay buffer).
        2. **Effort** — raw effort into ``_computed_forces`` (explicit control
           law, or the implicit end-of-step solve).
        3. **Clamping** — bounded effort into ``_applied_forces``. Explicit
           clamps after the drive law; implicit enforces them inside the
           solve.
        4. **Scatter-add** — *accumulate* applied (and optionally computed)
           effort into the output array.  The caller must zero the output
           (e.g. ``control.joint_f.zero_()``) before looping over actuators.
        5. **Drive state update** — write the drive's ``next_state``.

        Args:
            sim_state: Simulation state with position/velocity arrays.
            sim_control: Control structure with target/output arrays.
            current_act_state: Current composed state (None if stateless).
            next_act_state: Next composed state (None if stateless).
            dt: Timestep [s].
            battery: Optional :class:`Battery` for drives that use one (see
                :meth:`DriveBase.uses_battery`). Other drives ignore it.
        """
        if self.is_stateful() and (current_act_state is None or next_act_state is None):
            raise ValueError(
                "Stateful actuator requires both current_act_state and next_act_state; create them via actuator.state()"
            )

        positions = getattr(sim_state, self.state_pos_attr)
        velocities = getattr(sim_state, self.state_vel_attr)

        target_pos = getattr(sim_control, self.control_target_pos_attr)
        target_vel = getattr(sim_control, self.control_target_vel_attr)
        feedforward = None
        if self.control_feedforward_attr is not None:
            feedforward = getattr(sim_control, self.control_feedforward_attr, None)

        no_states = [None] * len(self.input_processors)
        current_states = (current_act_state and current_act_state.input_processor_states) or no_states
        next_states = (next_act_state and next_act_state.input_processor_states) or no_states

        # --- 1. Input processors (read current_state, write next_state) ---
        inputs = InputProcessorBase.Inputs(
            positions=positions,
            velocities=velocities,
            target_pos=target_pos,
            target_vel=target_vel,
            feedforward=feedforward,
            pos_indices=self.pos_indices,
            vel_indices=self.indices,
            target_pos_indices=self.target_pos_indices,
            target_vel_indices=self.indices,
            sim_positions=positions,
            sim_velocities=velocities,
        )
        for processor, current, nxt in zip(self.input_processors, current_states, next_states, strict=True):
            processed = processor.process(inputs, current, dt)
            if any(processor is legacy for legacy in self._legacy_delay_updates):
                processor.update_state(
                    inputs.target_pos,
                    inputs.target_vel,
                    inputs.feedforward,
                    inputs.target_pos_indices,
                    inputs.target_vel_indices,
                    current,
                    nxt,
                )
            else:
                processor.update_state(inputs, current, nxt)
            inputs = processed

        # --- 2+3. Effort mode: compute raw effort and clamp ---
        battery_kwargs = None
        if battery is not None and self.drive.uses_battery():
            battery_kwargs = {"battery": battery, "dof_indices": self.indices}
        drive_state = current_act_state.drive_state if current_act_state else None
        output_forces = self._effort_mode.compute_force(
            sim_state,
            inputs.positions,
            inputs.velocities,
            inputs.target_pos,
            inputs.target_vel,
            inputs.feedforward,
            inputs.pos_indices,
            inputs.vel_indices,
            inputs.target_pos_indices,
            inputs.target_vel_indices,
            self._computed_forces,
            self._applied_forces,
            drive_state,
            dt,
            battery_kwargs,
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
