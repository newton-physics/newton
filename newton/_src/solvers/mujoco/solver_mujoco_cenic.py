"""CENIC adaptive-step MuJoCo solver (CUDA-graph-fused).

Per-world adaptive time-stepping via step doubling, captured as a single
CUDA graph.  Step controller follows Drake's CalcAdjustedStepSize.
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
    """Clamp ideal_dt to [dt_min, dt_max], preserving ideal_dt for controller recovery."""
    i = wp.tid()
    actual = wp.clamp(ideal_dt[i], dt_min, dt_max)
    dt[i] = actual
    dt_half[i] = actual * wp.float32(0.5)



@wp.kernel
def _inf_norm_q_error_kernel(
    joint_q_full: wp.array(dtype=wp.float32),
    joint_q_double: wp.array(dtype=wp.float32),
    coords_per_world: int,
    error_out: wp.array(dtype=wp.float32),
):
    """Inf-norm (max absolute difference) on joint_q between full-step and doubled half-step.

    Diverged sims get error = 1e10.
    """
    world = wp.tid()
    q_start = world * coords_per_world

    max_err = float(0.0)
    for i in range(coords_per_world):
        d = wp.abs(joint_q_double[q_start + i] - joint_q_full[q_start + i])
        max_err = wp.max(max_err, d)

    if wp.isnan(max_err) or wp.isinf(max_err):
        max_err = float(1.0e10)

    error_out[world] = max_err


# Drake CalcAdjustedStepSize constants (err_order=2 for step doubling).
_DRAKE_SAFETY = wp.constant(wp.float32(0.9))
_DRAKE_MIN_SHRINK = wp.constant(wp.float32(0.1))
_DRAKE_MAX_GROW = wp.constant(wp.float32(5.0))
_DRAKE_HYSTERESIS_HIGH = wp.constant(wp.float32(1.2))


@wp.kernel
def _calc_adjusted_step(
    err: wp.array(dtype=wp.float32),
    dt: wp.array(dtype=wp.float32),
    ideal_dt: wp.array(dtype=wp.float32),
    accepted: wp.array(dtype=wp.bool),
    tol: float,
    dt_min: float,
):
    """Per-world Drake CalcAdjustedStepSize for step doubling (err_order=2).

    dt_max clamping is deferred to _apply_dt_cap so ideal_dt is preserved.
    """
    world = wp.tid()
    e = err[world]
    step = dt[world]

    # Boundary-stalled worlds (dt clamped to 0): accept without touching ideal_dt
    # so the next interval inherits a good dt instead of ramping from dt_min.
    if step <= wp.float32(0.0):
        accepted[world] = True
        return

    if wp.isnan(e) or wp.isinf(e):
        accepted[world] = False
        ideal_dt[world] = _DRAKE_MIN_SHRINK * step
        return

    # At the floor we must accept to avoid stalling.
    if step <= dt_min * wp.float32(1.001) and e > tol:
        accepted[world] = True
        ideal_dt[world] = dt_min
        return

    new_step = _DRAKE_SAFETY * step * wp.sqrt(tol / wp.max(e, wp.float32(1.0e-30)))

    # Hysteresis: suppress tiny grows.
    if new_step > step and new_step < _DRAKE_HYSTERESIS_HIGH * step:
        new_step = step

    # Don't shrink an already-good step.
    if new_step < step and e <= tol:
        new_step = step

    new_step = wp.clamp(new_step, _DRAKE_MIN_SHRINK * step, _DRAKE_MAX_GROW * step)

    accepted[world] = e <= tol or new_step >= step
    ideal_dt[world] = new_step


@wp.kernel
def _advance_sim_time(
    sim_time: wp.array(dtype=wp.float32),
    dt: wp.array(dtype=wp.float32),
    accepted: wp.array(dtype=wp.bool),
    error: wp.array(dtype=wp.float32),
    accepted_error: wp.array(dtype=wp.float32),
):
    """Advance sim_time[i] by dt[i] and snapshot error for accepted worlds only."""
    i = wp.tid()
    if accepted[i]:
        sim_time[i] = sim_time[i] + dt[i]
        accepted_error[i] = error[i]


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
    """Set flag[0] = 0 (assume all worlds reached the boundary)."""
    flag[0] = 0


@wp.kernel
def _boundary_check(
    sim_time: wp.array(dtype=wp.float32),
    target: wp.array(dtype=wp.float32),
    flag: wp.array(dtype=wp.int32),
):
    """Set flag to 1 if any world has not yet reached target."""
    i = wp.tid()
    if sim_time[i] < target[i]:
        wp.atomic_max(flag, 0, 1)


@wp.kernel
def _boundary_advance(arr: wp.array(dtype=wp.float32), delta: float):
    """Increment arr[i] by delta."""
    i = wp.tid()
    arr[i] = arr[i] + delta


@wp.kernel
def _clamp_dt_to_boundary(
    dt: wp.array(dtype=wp.float32),
    dt_half: wp.array(dtype=wp.float32),
    sim_time: wp.array(dtype=wp.float32),
    next_time: wp.array(dtype=wp.float32),
):
    """Clamp dt so worlds don't overshoot their boundary target.

    Worlds already at or past the boundary get dt=0 (no-op step).
    """
    i = wp.tid()
    remaining = next_time[i] - sim_time[i]
    if remaining <= wp.float32(0.0):
        dt[i] = wp.float32(0.0)
        dt_half[i] = wp.float32(0.0)
    elif dt[i] > remaining:
        dt[i] = remaining
        dt_half[i] = remaining * wp.float32(0.5)


@wp.kernel
def _iter_count_increment(count: wp.array(dtype=wp.int32)):
    """Increment iteration counter (dim=1, single thread)."""
    count[0] = count[0] + 1


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


class SolverMuJoCoCENIC(SolverMuJoCo):
    """Adaptive-step MuJoCo solver for high-accuracy dataset generation.

    Uses step doubling (3 MuJoCo evals per attempt) to estimate per-world
    integration error and adapt the timestep on the GPU.  The boundary loop
    launches kernels directly via ``wp.launch()`` each iteration, checking
    a 4-byte flag via ``.numpy()`` to detect when all worlds have reached
    the target time.

    Timesteps are managed internally by the error controller.  Set the
    initial value via ``dt_inner_init`` and query current values via
    :attr:`dt`.

    Example:

    .. code-block:: python

        solver = newton.solvers.SolverMuJoCoCENIC(model, tol=1e-3)
        state_0, state_1 = model.state(), model.state()

        while viewer.is_running():
            state_0, state_1 = solver.step_dt(DT, state_0, state_1, control,
                                               apply_forces=viewer.apply_forces)
            viewer.render(state_0, solver.sim_time.numpy().min())
    """

    def __init__(
        self,
        model: Model,
        *,
        tol: float = 1e-3,
        dt_inner_init: float = 0.01,
        dt_inner_min: float = 1e-6,
        dt_inner_max: float | None = None,
        **kwargs,
    ):
        """
        Args:
            model: The model to simulate.
            tol: Inf-norm error tolerance on joint_q per world [m or rad, depending on joint type].
                Worlds with error > tol are rejected and retry with a smaller dt.
            dt_inner_init: Initial inner (adaptive physics) timestep [s].
            dt_inner_min: Minimum allowed inner timestep [s].
            dt_inner_max: Maximum allowed inner timestep [s].  If None, clamped
                to the ``dt_outer`` argument of each :meth:`step_dt` call
                automatically so the inner step never overshoots the boundary.
            **kwargs: Forwarded to :class:`SolverMuJoCo`.
        """
        super().__init__(model, separate_worlds=True, use_mujoco_cpu=False, **kwargs)

        world_count = model.world_count
        device = model.device

        self._dt = wp.full(world_count, dt_inner_init, dtype=wp.float32, device=device)
        self._ideal_dt = wp.full(world_count, dt_inner_init, dtype=wp.float32, device=device)
        self._dt_half = wp.full(world_count, dt_inner_init * 0.5, dtype=wp.float32, device=device)
        self._sim_time = wp.zeros(world_count, dtype=wp.float32, device=device)
        self._accepted = wp.zeros(world_count, dtype=wp.bool, device=device)
        self._last_error = wp.zeros(world_count, dtype=wp.float32, device=device)
        self._accepted_error = wp.zeros(world_count, dtype=wp.float32, device=device)

        self._tol = float(tol)
        self._dt_min = float(dt_inner_min)
        self._dt_max = float(dt_inner_max) if dt_inner_max is not None else float("inf")

        self._scratch_full = model.state()
        self._scratch_mid = model.state()
        self._scratch_double = model.state()

        # Internal state buffers for the iteration body.
        self._state_cur = model.state()
        self._state_saved = model.state()

        self._coords_per_world = model.joint_coord_count // world_count
        self._dofs_per_world = model.joint_dof_count // world_count
        self._bodies_per_world = model.body_count // world_count

        self._next_time = wp.zeros(world_count, dtype=wp.float32, device=device)
        self._boundary_flag = wp.zeros(1, dtype=wp.int32, device=device)
        self._status_scalars = wp.zeros(6, dtype=wp.float32, device=device)

        self._iteration_count_buf = wp.zeros(1, dtype=wp.int32, device=device)

        # Stable buffer for opt.timestep; updated via wp.copy() per substep.
        self._timestep_buf = wp.full(world_count, dt_inner_init, dtype=wp.float32, device=device)
        self.mjw_model.opt.timestep = self._timestep_buf



    def _run_substep(
        self,
        state_in: State,
        state_out: State,
        contacts: Contacts,
        dt_array: wp.array,
    ) -> None:
        """Run one MuJoCo step: sync state_in, set timestep, step, write state_out."""
        self._update_mjc_data(self.mjw_data, self.model, state_in)
        wp.copy(self.mjw_model.opt.timestep, dt_array)

        with wp.ScopedDevice(self.model.device):
            if self.mjw_model.opt.run_collision_detection:
                self._mujoco_warp_step()
            else:
                self._convert_contacts_to_mjwarp(self.model, state_in, contacts)
                self._mujoco_warp_step()

        self._update_newton_state(self.model, state_out, self.mjw_data)

    def _run_iteration_body(self, effective_dt_max: float) -> None:
        """One step-doubling iteration: 3-eval + error control + dt cap + boundary check."""
        model = self.model
        n = model.world_count
        dev = model.device

        wp.launch(_iter_count_increment, dim=1, inputs=[self._iteration_count_buf], device=dev)

        # Clamp dt so no world overshoots its boundary target.
        wp.launch(
            _clamp_dt_to_boundary,
            dim=n,
            inputs=[self._dt, self._dt_half, self._sim_time, self._next_time],
            device=dev,
        )

        # Snapshot for rollback on rejection.
        wp.copy(self._state_saved.joint_q, self._state_cur.joint_q)
        wp.copy(self._state_saved.joint_qd, self._state_cur.joint_qd)
        if self._state_cur.body_q is not None and self._state_saved.body_q is not None:
            wp.copy(self._state_saved.body_q, self._state_cur.body_q)
        if self._state_cur.body_qd is not None and self._state_saved.body_qd is not None:
            wp.copy(self._state_saved.body_qd, self._state_cur.body_qd)

        # 3 MuJoCo evals: full dt, half dt, half dt.
        self._run_substep(self._state_cur, self._scratch_full, None, self._dt)
        self._run_substep(self._state_cur, self._scratch_mid, None, self._dt_half)
        self._run_substep(self._scratch_mid, self._scratch_double, None, self._dt_half)

        wp.launch(
            _inf_norm_q_error_kernel,
            dim=n,
            inputs=[
                self._scratch_full.joint_q,
                self._scratch_double.joint_q,
                self._coords_per_world,
            ],
            outputs=[self._last_error],
            device=dev,
        )

        wp.launch(
            _calc_adjusted_step,
            dim=n,
            inputs=[self._last_error, self._dt, self._ideal_dt, self._accepted,
                    self._tol, self._dt_min],
            device=dev,
        )

        # State select: accepted worlds get scratch_double, rejected get saved.
        wp.launch(
            _select_float_kernel,
            dim=model.joint_coord_count,
            inputs=[self._scratch_double.joint_q, self._state_saved.joint_q,
                    self._accepted, self._coords_per_world],
            outputs=[self._state_cur.joint_q],
            device=dev,
        )
        wp.launch(
            _select_float_kernel,
            dim=model.joint_dof_count,
            inputs=[self._scratch_double.joint_qd, self._state_saved.joint_qd,
                    self._accepted, self._dofs_per_world],
            outputs=[self._state_cur.joint_qd],
            device=dev,
        )
        if self._state_cur.body_q is not None:
            wp.launch(
                _select_transform_kernel,
                dim=model.body_count,
                inputs=[self._scratch_double.body_q, self._state_saved.body_q,
                        self._accepted, self._bodies_per_world],
                outputs=[self._state_cur.body_q],
                device=dev,
            )
        if self._state_cur.body_qd is not None:
            wp.launch(
                _select_spatial_vector_kernel,
                dim=model.body_count,
                inputs=[self._scratch_double.body_qd, self._state_saved.body_qd,
                        self._accepted, self._bodies_per_world],
                outputs=[self._state_cur.body_qd],
                device=dev,
            )

        wp.launch(
            _advance_sim_time,
            dim=n,
            inputs=[self._sim_time, self._dt, self._accepted,
                    self._last_error, self._accepted_error],
            device=dev,
        )

        # Cap dt for the next iteration.
        wp.launch(
            _apply_dt_cap,
            dim=n,
            inputs=[self._ideal_dt, self._dt_min, effective_dt_max,
                    self._dt, self._dt_half],
            device=dev,
        )

        # Boundary check: sets _boundary_flag to 0 (done) or 1 (continue).
        wp.launch(_boundary_reset, dim=1, inputs=[self._boundary_flag], device=dev)
        wp.launch(
            _boundary_check,
            dim=n,
            inputs=[self._sim_time, self._next_time, self._boundary_flag],
            device=dev,
        )

    @event_scope
    @override
    def step(
        self,
        state_in: State,
        state_out: State,
        control: Control,
        contacts: Contacts,
    ) -> State:
        """Advance each world by one adaptive step (non-graph path).

        For the CUDA-graph-optimized path use :meth:`step_dt`.

        Args:
            state_in: Input state.
            state_out: Output state (written in place).
            control: Control inputs.
            contacts: Contact data (when ``use_mujoco_contacts`` is False).

        Returns:
            state_out
        """
        model = self.model
        device = model.device
        n = model.world_count

        self._apply_mjc_control(model, state_in, control, self.mjw_data)
        self._enable_rne_postconstraint(state_out)

        self._run_substep(state_in, self._scratch_full, contacts, self._dt)
        self._run_substep(state_in, self._scratch_mid, contacts, self._dt_half)
        self._run_substep(self._scratch_mid, self._scratch_double, contacts, self._dt_half)

        wp.launch(
            _inf_norm_q_error_kernel,
            dim=n,
            inputs=[
                self._scratch_full.joint_q,
                self._scratch_double.joint_q,
                self._coords_per_world,
            ],
            outputs=[self._last_error],
            device=device,
        )
        wp.launch(
            _calc_adjusted_step,
            dim=n,
            inputs=[self._last_error, self._dt, self._ideal_dt, self._accepted,
                    self._tol, self._dt_min],
            device=device,
        )
        wp.launch(
            _apply_dt_cap,
            dim=n,
            inputs=[self._ideal_dt, self._dt_min, self._dt_max, self._dt, self._dt_half],
            device=device,
        )

        wp.launch(
            _select_float_kernel,
            dim=model.joint_coord_count,
            inputs=[self._scratch_double.joint_q, state_in.joint_q,
                    self._accepted, self._coords_per_world],
            outputs=[state_out.joint_q],
            device=device,
        )
        wp.launch(
            _select_float_kernel,
            dim=model.joint_dof_count,
            inputs=[self._scratch_double.joint_qd, state_in.joint_qd,
                    self._accepted, self._dofs_per_world],
            outputs=[state_out.joint_qd],
            device=device,
        )
        if state_out.body_q is not None:
            wp.launch(
                _select_transform_kernel,
                dim=model.body_count,
                inputs=[self._scratch_double.body_q, state_in.body_q,
                        self._accepted, self._bodies_per_world],
                outputs=[state_out.body_q],
                device=device,
            )
        if state_out.body_qd is not None:
            wp.launch(
                _select_spatial_vector_kernel,
                dim=model.body_count,
                inputs=[self._scratch_double.body_qd, state_in.body_qd,
                        self._accepted, self._bodies_per_world],
                outputs=[state_out.body_qd],
                device=device,
            )

        wp.launch(
            _advance_sim_time,
            dim=n,
            inputs=[self._sim_time, self._dt, self._accepted,
                    self._last_error, self._accepted_error],
            device=device,
        )

        self._step += 1
        return state_out

    @event_scope
    @override
    def step_dt(
        self,
        dt_outer: float,
        state_0: State,
        state_1: State,
        control: Control,
        apply_forces=None,
    ) -> tuple[State, State]:
        """Advance all worlds by exactly ``dt_outer`` seconds of simulation time.

        The 3-eval step-doubling block is captured as a CUDA graph and replayed
        once per iteration via ``wp.capture_launch()``.  The dt cap and boundary
        check run as direct ``wp.launch()`` calls outside the graph, with a
        single ``.numpy()`` read-back (4 bytes) per iteration to check
        termination.

        Args:
            dt_outer: Outer control/render period [s].
            state_0: Current state (input/output).
            state_1: Scratch state (unused; returned unchanged).
            control: Control inputs (applied once, persists across substeps).
            apply_forces: Optional ``fn(state)`` for external forces.

        Returns:
            ``(state_0, state_1)`` with ``state_0`` updated.
        """
        device = self.model.device
        n = self.model.world_count

        effective_dt_max = min(self._dt_max, dt_outer)

        wp.launch(
            _apply_dt_cap,
            dim=n,
            inputs=[self._ideal_dt, self._dt_min, effective_dt_max,
                    self._dt, self._dt_half],
            device=device,
        )

        wp.copy(self._state_cur.joint_q, state_0.joint_q)
        wp.copy(self._state_cur.joint_qd, state_0.joint_qd)
        if state_0.body_q is not None and self._state_cur.body_q is not None:
            wp.copy(self._state_cur.body_q, state_0.body_q)
        if state_0.body_qd is not None and self._state_cur.body_qd is not None:
            wp.copy(self._state_cur.body_qd, state_0.body_qd)

        self._apply_mjc_control(self.model, state_0, control, self.mjw_data)
        if apply_forces is not None:
            apply_forces(state_0)

        self._enable_rne_postconstraint(self._state_cur)

        wp.launch(_boundary_advance, dim=n,
                  inputs=[self._next_time, dt_outer], device=device)

        self._iteration_count_buf.fill_(0)
        self._boundary_flag.fill_(1)

        while True:
            self._run_iteration_body(effective_dt_max)
            if self._boundary_flag.numpy()[0] == 0:
                break

        wp.copy(state_0.joint_q, self._state_cur.joint_q)
        wp.copy(state_0.joint_qd, self._state_cur.joint_qd)
        if state_0.body_q is not None and self._state_cur.body_q is not None:
            wp.copy(state_0.body_q, self._state_cur.body_q)
        if state_0.body_qd is not None and self._state_cur.body_qd is not None:
            wp.copy(state_0.body_qd, self._state_cur.body_qd)

        return state_0, state_1

    @property
    def iteration_count(self) -> wp.array:
        """Iteration count from the most recent ``step_dt``, shape ``[1]``, int32, on device."""
        return self._iteration_count_buf

    @property
    def sim_time(self) -> wp.array:
        """Per-world simulation time [s], shape ``[world_count]``, float32, on device.

        Only advances for accepted steps.
        """
        return self._sim_time

    @property
    def dt(self) -> wp.array:
        """Current per-world timestep [s], shape ``[world_count]``, float32, on device."""
        return self._dt

    @property
    def last_error(self) -> wp.array:
        """Inf-norm on q from the most recent accepted step, shape ``[world_count]``, float32, on device."""
        return self._accepted_error

    @property
    def last_raw_error(self) -> wp.array:
        """Inf-norm on q from the most recent attempt (accepted or rejected), shape ``[world_count]``, float32, on device."""
        return self._last_error

    @property
    def accepted(self) -> wp.array:
        """Per-world accept flags from the most recent step, shape ``[world_count]``, bool, on device."""
        return self._accepted

    def get_status_summary(self) -> dict[str, float]:
        """Reduce per-world arrays to a 6-scalar summary via one GPU transfer."""
        device = self.model.device
        n = self.model.world_count

        wp.launch(_status_sentinel_reset, dim=1, inputs=[self._status_scalars], device=device)
        wp.launch(
            _status_summary_kernel,
            dim=n,
            inputs=[self._sim_time, self._accepted_error, self._dt, self._accepted,
                    self._status_scalars],
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

