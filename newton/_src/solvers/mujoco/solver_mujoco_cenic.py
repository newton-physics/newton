# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CENIC Adaptive-Step MuJoCo Solver.

Per-world adaptive time-stepping via step doubling. Each step() runs three MuJoCo
evaluations (full dt, dt/2, dt/2), estimates per-world RMS integration error, and
accepts or rejects each world on the GPU with no CPU-GPU transfers in the hot path.
"""

from __future__ import annotations

import warp as wp

from ...core.types import override
from ...sim import Contacts, Control, Model, State
from ...utils.benchmark import event_scope
from .solver_mujoco import SolverMuJoCo


@wp.kernel
def _apply_dt_cap(
    ideal_dt: wp.array(dtype=wp.float32),
    dt_min: float,
    dt_max: float,
    dt: wp.array(dtype=wp.float32),
    dt_half: wp.array(dtype=wp.float32),
):
    """Clamp ideal_dt to [dt_min, dt_max] to produce the actual dt used this step.

    Keeping ideal_dt separate prevents a boundary cap from corrupting the
    controller state, so dt recovers quickly after contact-dense phases.
    """
    i = wp.tid()
    actual = wp.clamp(ideal_dt[i], dt_min, dt_max)
    dt[i] = actual
    dt_half[i] = actual * wp.float32(0.5)


@wp.kernel
def _error_control_kernel(
    joint_q_full: wp.array(dtype=wp.float32),
    joint_qd_full: wp.array(dtype=wp.float32),
    joint_q_double: wp.array(dtype=wp.float32),
    joint_qd_double: wp.array(dtype=wp.float32),
    coords_per_world: int,
    dofs_per_world: int,
    tol: float,
    dt_safety: float,
    dt_min: float,
    dt_max: float,
    dt: wp.array(dtype=wp.float32),
    sim_time: wp.array(dtype=wp.float32),
    ideal_dt: wp.array(dtype=wp.float32),
    dt_half: wp.array(dtype=wp.float32),
    accepted: wp.array(dtype=wp.bool),
    last_error: wp.array(dtype=wp.float32),
):
    """Per-world step-doubling error control (Drake IntegratorBase::CalcAdjustedStepSize).

    * NaN/Inf guard — diverged worlds get error=1e10, guaranteeing rejection and aggressive shrink.
    * Accept-at-floor — worlds stuck at dt_min always advance rather than stalling.
    * Growth hysteresis (1.2×) — dt only grows when the gain is ≥ 20%.
    * Rejection shrink — genuine rejections enforce at least 10% shrink.
    * Accept shrink suppression — accepted steps never shrink dt.
    """
    world = wp.tid()

    q_start = world * coords_per_world
    qd_start = world * dofs_per_world

    error_sq = float(0.0)
    for i in range(coords_per_world):
        d = joint_q_double[q_start + i] - joint_q_full[q_start + i]
        error_sq += d * d
    for i in range(dofs_per_world):
        d = joint_qd_double[qd_start + i] - joint_qd_full[qd_start + i]
        error_sq += d * d

    n = float(coords_per_world + dofs_per_world)
    error = wp.sqrt(error_sq / n)

    if wp.isnan(error) or wp.isinf(error):
        error = float(1.0e10)

    last_error[world] = error

    old_dt = dt[world]

    true_acceptance = error <= tol
    at_floor = old_dt <= dt_min * float(1.001)  # fp-safe floor comparison
    is_accepted = true_acceptance or at_floor
    accepted[world] = is_accepted

    if is_accepted:
        sim_time[world] = sim_time[world] + old_dt

    ratio = tol / wp.max(error, float(1.0e-10))
    factor = wp.clamp(dt_safety * wp.sqrt(ratio), float(0.1), float(5.0))

    new_dt = old_dt * factor

    if new_dt > old_dt and new_dt < old_dt * float(1.2):
        new_dt = old_dt

    if new_dt < old_dt:
        if true_acceptance:
            new_dt = old_dt
        else:
            new_dt = wp.min(new_dt, old_dt * float(0.9))

    ideal_dt[world] = new_dt
    dt[world] = wp.clamp(new_dt, dt_min, dt_max)
    dt_half[world] = wp.clamp(new_dt, dt_min, dt_max) * wp.float32(0.5)


@wp.kernel
def _select_float_kernel(
    candidate: wp.array(dtype=wp.float32),
    fallback: wp.array(dtype=wp.float32),
    accepted: wp.array(dtype=wp.bool),
    stride: int,
    out: wp.array(dtype=wp.float32),
):
    """Select candidate for accepted worlds, fallback for rejected worlds."""
    i = wp.tid()
    world = i // stride
    if accepted[world]:
        out[i] = candidate[i]
    else:
        out[i] = fallback[i]


@wp.kernel
def _select_transform_kernel(
    candidate: wp.array(dtype=wp.transform),
    fallback: wp.array(dtype=wp.transform),
    accepted: wp.array(dtype=wp.bool),
    stride: int,
    out: wp.array(dtype=wp.transform),
):
    """Select body pose from accepted or fallback state."""
    i = wp.tid()
    world = i // stride
    if accepted[world]:
        out[i] = candidate[i]
    else:
        out[i] = fallback[i]


@wp.kernel
def _select_spatial_vector_kernel(
    candidate: wp.array(dtype=wp.spatial_vector),
    fallback: wp.array(dtype=wp.spatial_vector),
    accepted: wp.array(dtype=wp.bool),
    stride: int,
    out: wp.array(dtype=wp.spatial_vector),
):
    """Select body velocity from accepted or fallback state."""
    i = wp.tid()
    world = i // stride
    if accepted[world]:
        out[i] = candidate[i]
    else:
        out[i] = fallback[i]


@wp.kernel
def _boundary_reset(flag: wp.array(dtype=wp.int32)):
    """Set flag[0] = 1 (all-reached)."""
    flag[0] = 1


@wp.kernel
def _boundary_check(
    sim_time: wp.array(dtype=wp.float32),
    target: wp.array(dtype=wp.float32),
    flag: wp.array(dtype=wp.int32),
):
    """Clear flag to 0 if any world has not yet reached target."""
    i = wp.tid()
    if sim_time[i] < target[i]:
        wp.atomic_min(flag, 0, 0)


@wp.kernel
def _boundary_advance(arr: wp.array(dtype=wp.float32), delta: float):
    """Increment arr[i] by delta."""
    i = wp.tid()
    arr[i] = arr[i] + delta


@wp.kernel
def _status_sentinel_reset(out: wp.array(dtype=wp.float32)):
    """Reset 6-element summary buffer: [min_sim_time, max_sim_time, max_error, accept_count, min_dt, max_dt]."""
    out[0] = float(1.0e38)
    out[1] = float(0.0)
    out[2] = float(0.0)
    out[3] = float(0.0)
    out[4] = float(1.0e38)
    out[5] = float(0.0)


@wp.kernel
def _status_summary_kernel(
    sim_time: wp.array(dtype=wp.float32),
    last_error: wp.array(dtype=wp.float32),
    dt: wp.array(dtype=wp.float32),
    accepted: wp.array(dtype=wp.bool),
    out: wp.array(dtype=wp.float32),
):
    """Reduce per-world arrays to 6 summary scalars via atomics."""
    i = wp.tid()
    wp.atomic_min(out, 0, sim_time[i])
    wp.atomic_max(out, 1, sim_time[i])
    wp.atomic_max(out, 2, last_error[i])
    if accepted[i]:
        wp.atomic_add(out, 3, wp.float32(1.0))
    wp.atomic_min(out, 4, dt[i])
    wp.atomic_max(out, 5, dt[i])


@wp.kernel
def _error_control_boundary_kernel(
    joint_q_full: wp.array(dtype=wp.float32),
    joint_qd_full: wp.array(dtype=wp.float32),
    joint_q_double: wp.array(dtype=wp.float32),
    joint_qd_double: wp.array(dtype=wp.float32),
    coords_per_world: int,
    dofs_per_world: int,
    tol: float,
    dt_safety: float,
    dt_min: float,
    dt_max: float,
    dt: wp.array(dtype=wp.float32),
    sim_time: wp.array(dtype=wp.float32),
    ideal_dt: wp.array(dtype=wp.float32),
    next_time: wp.array(dtype=wp.float32),
    boundary_flag: wp.array(dtype=wp.int32),
    dt_half: wp.array(dtype=wp.float32),
    accepted: wp.array(dtype=wp.bool),
    last_error: wp.array(dtype=wp.float32),
):
    """_error_control_kernel fused with boundary check.

    Atomically clears boundary_flag[0] to 0 if this world has not yet reached
    next_time[world], eliminating the separate _boundary_check kernel launch.
    """
    world = wp.tid()

    q_start = world * coords_per_world
    qd_start = world * dofs_per_world

    error_sq = float(0.0)
    for i in range(coords_per_world):
        d = joint_q_double[q_start + i] - joint_q_full[q_start + i]
        error_sq += d * d
    for i in range(dofs_per_world):
        d = joint_qd_double[qd_start + i] - joint_qd_full[qd_start + i]
        error_sq += d * d

    n = float(coords_per_world + dofs_per_world)
    error = wp.sqrt(error_sq / n)

    if wp.isnan(error) or wp.isinf(error):
        error = float(1.0e10)

    last_error[world] = error

    old_dt = dt[world]

    true_acceptance = error <= tol
    at_floor = old_dt <= dt_min * float(1.001)  # fp-safe floor comparison
    is_accepted = true_acceptance or at_floor
    accepted[world] = is_accepted

    if is_accepted:
        sim_time[world] = sim_time[world] + old_dt

    ratio = tol / wp.max(error, float(1.0e-10))
    factor = wp.clamp(dt_safety * wp.sqrt(ratio), float(0.1), float(5.0))

    new_dt = old_dt * factor

    if new_dt > old_dt and new_dt < old_dt * float(1.2):
        new_dt = old_dt

    if new_dt < old_dt:
        if true_acceptance:
            new_dt = old_dt
        else:
            new_dt = wp.min(new_dt, old_dt * float(0.9))

    ideal_dt[world] = new_dt
    dt[world] = wp.clamp(new_dt, dt_min, dt_max)
    dt_half[world] = wp.clamp(new_dt, dt_min, dt_max) * wp.float32(0.5)

    if sim_time[world] < next_time[world]:
        wp.atomic_min(boundary_flag, 0, 0)


class SolverMuJoCoCENIC(SolverMuJoCo):
    """Adaptive-step MuJoCo solver for high-accuracy dataset generation.

    Uses step doubling to estimate per-world integration error and adapt the
    timestep entirely on the GPU — no CPU-GPU transfers in the simulation loop.

    Each call to :meth:`step` runs three MuJoCo evaluations (one full step and
    two half-steps) and either advances a world's state or leaves it unchanged,
    depending on whether the local error is within ``tol``.  Per-world ``dt``
    grows for easy regions and shrinks for stiff or nonlinear ones, so the
    resulting dataset is naturally denser where the dynamics are hardest.

    Note:
        Timesteps are managed internally by the error controller.  Set the initial
        value via ``dt_inner_init`` and query current values via :attr:`dt`.

    Example:

    .. code-block:: python

        solver = newton.solvers.SolverMuJoCoCENIC(model, tol=1e-3, dt_inner_init=0.01)
        state_0, state_1 = model.state(), model.state()

        while collecting_data:
            solver.step(state_0, state_1, control, contacts=None)
            record(solver.sim_time, state_1)
            state_0, state_1 = state_1, state_0
    """

    def __init__(
        self,
        model: Model,
        *,
        tol: float = 1e-3,
        dt_inner_init: float = 0.01,
        dt_inner_min: float = 1e-6,
        dt_inner_max: float | None = None,
        dt_safety: float = 0.9,
        **kwargs,
    ):
        """
        Args:
            model: The model to simulate.
            tol: RMS integration error tolerance per world [same units as joint_q/qd].
                Worlds with error > tol are rejected and retry with a smaller dt.
            dt_inner_init: Initial inner (adaptive physics) timestep [s] for all worlds.
            dt_inner_min: Minimum allowed inner timestep [s]. Near-zero floor so the
                adaptive stepper has full dynamic range; accept-at-floor is a last resort.
            dt_inner_max: Maximum allowed inner timestep [s]. If None, clamped to the
                ``dt_outer`` argument of each :meth:`step_dt` call automatically so the
                inner step never overshoots the outer control boundary.
            dt_safety: Safety factor (< 1) applied to the step-doubling dt_inner controller.
                Smaller values are more conservative.
            **kwargs: Forwarded to :class:`SolverMuJoCo`.
        """
        # Compute generous njmax/nconmax defaults so the user never sees
        # "nefc overflow" errors.  Each contact generates up to 5 constraint
        # rows (1 normal + 4 pyramidal friction).
        shapes_per_world = model.shape_count // model.world_count
        if "nconmax" not in kwargs:
            kwargs["nconmax"] = shapes_per_world * shapes_per_world
        if "njmax" not in kwargs:
            kwargs["njmax"] = kwargs["nconmax"] * 5
        if "iterations" not in kwargs:
            kwargs["iterations"] = 50
        if "ls_iterations" not in kwargs:
            kwargs["ls_iterations"] = 10
        if "ccd_iterations" not in kwargs:
            kwargs["ccd_iterations"] = 1000
        if "ccd_tolerance" not in kwargs:
            kwargs["ccd_tolerance"] = 1e-4

        super().__init__(model, separate_worlds=True, use_mujoco_cpu=False, **kwargs)

        world_count = model.world_count
        device = model.device

        self._dt = wp.full(world_count, dt_inner_init, dtype=wp.float32, device=device)
        self._ideal_dt = wp.full(world_count, dt_inner_init, dtype=wp.float32, device=device)
        self._dt_half = wp.full(world_count, dt_inner_init * 0.5, dtype=wp.float32, device=device)
        self._sim_time = wp.zeros(world_count, dtype=wp.float32, device=device)
        self._accepted = wp.zeros(world_count, dtype=wp.bool, device=device)
        self._last_error = wp.zeros(world_count, dtype=wp.float32, device=device)

        self._tol = float(tol)
        self._dt_min = float(dt_inner_min)
        # None means "clamp to dt_outer boundary automatically".
        # Store inf so the kernel always receives a valid float.
        self._dt_max = float(dt_inner_max) if dt_inner_max is not None else float("inf")
        self._dt_safety = float(dt_safety)

        self._scratch_full = model.state()
        self._scratch_mid = model.state()
        self._scratch_double = model.state()

        self._coords_per_world = model.joint_coord_count // world_count
        self._dofs_per_world = model.joint_dof_count // world_count
        self._bodies_per_world = model.body_count // world_count

        self._next_time = wp.zeros(world_count, dtype=wp.float32, device=device)
        self._boundary_flag = wp.zeros(1, dtype=wp.int32, device=device)
        self._status_scalars = wp.zeros(6, dtype=wp.float32, device=device)

    def _run_substep(
        self,
        state_in: State,
        state_out: State,
        contacts: Contacts,
        dt_array: wp.array,
    ) -> None:
        """Run one MuJoCo step from state_in → state_out.

        Always syncs mjw_data from state_in before stepping, because the three
        substeps each start from a different initial state.
        """
        self._update_mjc_data(self.mjw_data, self.model, state_in)
        self.mjw_model.opt.timestep = dt_array

        with wp.ScopedDevice(self.model.device):
            if self.mjw_model.opt.run_collision_detection:
                self._mujoco_warp_step()
            else:
                self._convert_contacts_to_mjwarp(self.model, state_in, contacts)
                self._mujoco_warp_step()

        self._update_newton_state(self.model, state_out, self.mjw_data)

    def _error_control_and_select_state(self, state_in: State, state_out: State) -> None:
        """Estimate error, adapt dt, and write the correct state to state_out."""
        model = self.model
        device = model.device

        wp.launch(
            _error_control_kernel,
            dim=model.world_count,
            inputs=[
                self._scratch_full.joint_q,
                self._scratch_full.joint_qd,
                self._scratch_double.joint_q,
                self._scratch_double.joint_qd,
                self._coords_per_world,
                self._dofs_per_world,
                self._tol,
                self._dt_safety,
                self._dt_min,
                self._dt_max,
                self._dt,
                self._sim_time,
                self._ideal_dt,
            ],
            outputs=[self._dt_half, self._accepted, self._last_error],
            device=device,
        )

        wp.launch(
            _select_float_kernel,
            dim=model.joint_coord_count,
            inputs=[
                self._scratch_double.joint_q,
                state_in.joint_q,
                self._accepted,
                self._coords_per_world,
            ],
            outputs=[state_out.joint_q],
            device=device,
        )

        wp.launch(
            _select_float_kernel,
            dim=model.joint_dof_count,
            inputs=[
                self._scratch_double.joint_qd,
                state_in.joint_qd,
                self._accepted,
                self._dofs_per_world,
            ],
            outputs=[state_out.joint_qd],
            device=device,
        )

        if state_out.body_q is not None:
            wp.launch(
                _select_transform_kernel,
                dim=model.body_count,
                inputs=[
                    self._scratch_double.body_q,
                    state_in.body_q,
                    self._accepted,
                    self._bodies_per_world,
                ],
                outputs=[state_out.body_q],
                device=device,
            )

        if state_out.body_qd is not None:
            wp.launch(
                _select_spatial_vector_kernel,
                dim=model.body_count,
                inputs=[
                    self._scratch_double.body_qd,
                    state_in.body_qd,
                    self._accepted,
                    self._bodies_per_world,
                ],
                outputs=[state_out.body_qd],
                device=device,
            )

    def _error_control_and_select_state_with_boundary(
        self, state_in: State, state_out: State
    ) -> None:
        """Like _error_control_and_select_state but with fused boundary check.

        The caller must reset self._boundary_flag[0] = 1 via _boundary_reset before calling.
        """
        model = self.model
        device = model.device

        wp.launch(
            _error_control_boundary_kernel,
            dim=model.world_count,
            inputs=[
                self._scratch_full.joint_q,
                self._scratch_full.joint_qd,
                self._scratch_double.joint_q,
                self._scratch_double.joint_qd,
                self._coords_per_world,
                self._dofs_per_world,
                self._tol,
                self._dt_safety,
                self._dt_min,
                self._dt_max,
                self._dt,
                self._sim_time,
                self._ideal_dt,
                self._next_time,
                self._boundary_flag,
            ],
            outputs=[self._dt_half, self._accepted, self._last_error],
            device=device,
        )

        wp.launch(
            _select_float_kernel,
            dim=model.joint_coord_count,
            inputs=[
                self._scratch_double.joint_q,
                state_in.joint_q,
                self._accepted,
                self._coords_per_world,
            ],
            outputs=[state_out.joint_q],
            device=device,
        )

        wp.launch(
            _select_float_kernel,
            dim=model.joint_dof_count,
            inputs=[
                self._scratch_double.joint_qd,
                state_in.joint_qd,
                self._accepted,
                self._dofs_per_world,
            ],
            outputs=[state_out.joint_qd],
            device=device,
        )

        if state_out.body_q is not None:
            wp.launch(
                _select_transform_kernel,
                dim=model.body_count,
                inputs=[
                    self._scratch_double.body_q,
                    state_in.body_q,
                    self._accepted,
                    self._bodies_per_world,
                ],
                outputs=[state_out.body_q],
                device=device,
            )

        if state_out.body_qd is not None:
            wp.launch(
                _select_spatial_vector_kernel,
                dim=model.body_count,
                inputs=[
                    self._scratch_double.body_qd,
                    state_in.body_qd,
                    self._accepted,
                    self._bodies_per_world,
                ],
                outputs=[state_out.body_qd],
                device=device,
            )

    def _step_with_boundary(
        self,
        state_in: State,
        state_out: State,
        control: Control,
        contacts: Contacts,
    ) -> State:
        """Like step() but fuses the boundary check into the error-control kernel.

        The caller must reset _boundary_flag[0] = 1 via _boundary_reset before calling.
        """
        self._apply_mjc_control(self.model, state_in, control, self.mjw_data)
        self._enable_rne_postconstraint(state_out)

        self._run_substep(state_in, self._scratch_full, contacts, self._dt)
        self._run_substep(state_in, self._scratch_mid, contacts, self._dt_half)
        self._run_substep(self._scratch_mid, self._scratch_double, contacts, self._dt_half)

        self._error_control_and_select_state_with_boundary(state_in, state_out)

        self._step += 1
        return state_out

    @event_scope
    @override
    def step(
        self,
        state_in: State,
        state_out: State,
        control: Control,
        contacts: Contacts,
    ) -> State:
        """Advance each world by one adaptive step using step doubling.

        Runs three MuJoCo evaluations on the GPU with no CPU-GPU transfers.

        Args:
            state_in:  Input state.
            state_out: Output state (written in place).
            control:   Control inputs, applied identically to all three evaluations.
            contacts:  Contact data (only used when ``use_mujoco_contacts`` is False).

        Returns:
            state_out
        """
        # Control is written to mjw_data.ctrl, which _update_mjc_data does not reset,
        # so the same control is active throughout all three substep evaluations.
        self._apply_mjc_control(self.model, state_in, control, self.mjw_data)
        self._enable_rne_postconstraint(state_out)

        self._run_substep(state_in, self._scratch_full, contacts, self._dt)
        self._run_substep(state_in, self._scratch_mid, contacts, self._dt_half)
        self._run_substep(self._scratch_mid, self._scratch_double, contacts, self._dt_half)

        self._error_control_and_select_state(state_in, state_out)

        self._step += 1
        return state_out

    def step_dt(
        self,
        dt_outer: float,
        state_0: State,
        state_1: State,
        control: Control,
        apply_forces=None,
    ) -> tuple[State, State]:
        """Advance all worlds by exactly ``dt_outer`` of simulation time.

        Fast path (``dt_inner_min >= dt_outer``): runs exactly one :meth:`step` with no GPU→CPU
        transfer. General path (``dt_inner_min < dt_outer``): loops with one ``int32`` read-back
        per iteration until all worlds have crossed the boundary.

        Args:
            dt_outer: Outer control/render period to advance [s].
            state_0: Current state (input).
            state_1: Scratch state (output).
            control: Control inputs.
            apply_forces: Optional callable ``fn(state)`` invoked before each substep.

        Returns:
            ``(new_current, new_scratch)``
        """
        device = self.model.device
        n = self.model.world_count

        effective_dt_max = min(self._dt_max, dt_outer)
        _saved_dt_max = self._dt_max
        self._dt_max = effective_dt_max

        # Recompute dt_inner and dt_half from ideal_dt with the new effective_dt_max so the
        # controller state recovers from boundary caps without permanently reducing ideal_dt.
        wp.launch(
            _apply_dt_cap,
            dim=n,
            inputs=[self._ideal_dt, self._dt_min, effective_dt_max, self._dt, self._dt_half],
            device=device,
        )

        wp.launch(_boundary_advance, dim=n, inputs=[self._next_time, dt_outer], device=device)

        if self._dt_min >= dt_outer:
            # Fast path: dt_inner_min >= dt_outer guarantees one step crosses the boundary.
            state_0.clear_forces()
            if apply_forces is not None:
                apply_forces(state_0)
            self.step(state_0, state_1, control, contacts=None)
            self._dt_max = _saved_dt_max
            return state_1, state_0

        while True:
            state_0.clear_forces()
            if apply_forces is not None:
                apply_forces(state_0)

            wp.launch(_boundary_reset, dim=1, inputs=[self._boundary_flag], device=device)
            self._step_with_boundary(state_0, state_1, control, contacts=None)
            state_0, state_1 = state_1, state_0

            if self._boundary_flag.numpy()[0]:
                break

        self._dt_max = _saved_dt_max
        return state_0, state_1

    @property
    def sim_time(self) -> wp.array:
        """Per-world simulation time [s], shape ``[world_count]``, float32, on device.

        Only advances for accepted steps. Useful for tagging dataset entries
        with the correct simulation timestamp.
        """
        return self._sim_time

    @property
    def dt(self) -> wp.array:
        """Current per-world timestep [s], shape ``[world_count]``, float32, on device.

        Updated after every :meth:`step` call by the step-doubling controller.
        """
        return self._dt

    @property
    def last_error(self) -> wp.array:
        """RMS integration error from the most recent step, shape ``[world_count]``, float32, on device.

        Values above ``tol`` indicate that world's step was rejected.
        """
        return self._last_error

    @property
    def accepted(self) -> wp.array:
        """Per-world accept flags from the most recent step, shape ``[world_count]``, bool, on device."""
        return self._accepted

    def get_status_summary(self) -> dict[str, float]:
        """Compact per-world status with a single GPU transfer, O(1) in world count.

        Returns:
            A dict with keys ``sim_time_min``, ``sim_time_max``, ``error_max``,
            ``accept_count``, ``dt_min``, ``dt_max``.
        """
        device = self.model.device
        n = self.model.world_count

        wp.launch(_status_sentinel_reset, dim=1, inputs=[self._status_scalars], device=device)

        wp.launch(
            _status_summary_kernel,
            dim=n,
            inputs=[self._sim_time, self._last_error, self._dt, self._accepted, self._status_scalars],
            device=device,
        )

        scalars = self._status_scalars.numpy()
        return {
            "sim_time_min": float(scalars[0]),
            "sim_time_max": float(scalars[1]),
            "error_max":    float(scalars[2]),
            "accept_count": int(scalars[3]),
            "dt_min":       float(scalars[4]),
            "dt_max":       float(scalars[5]),
        }
