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

"""CENIC Adaptive-Step MuJoCo Solver
=====================================
Per-world adaptive time-stepping via **step doubling**, with all decision
logic in Warp kernels so no data crosses the PCIe bus during the hot path.

Three-evaluation pipeline per outer ``step()`` call
----------------------------------------------------
::

    Eval 1 — full step    :  state_in   ─── dt ───►  scratch_full
    Eval 2 — first half   :  state_in   ─── dt/2 ──► scratch_mid
    Eval 3 — second half  :  scratch_mid ── dt/2 ──► scratch_double

The RMS difference between ``scratch_double`` (two half-steps, more accurate)
and ``scratch_full`` (one full step) is used as the per-world error estimate.
A single Warp kernel then:

* **Accepts** worlds where error ≤ tol → ``state_out = scratch_double``
* **Rejects** worlds where error > tol  → ``state_out = state_in`` (no advance)
* Adapts per-world ``dt`` via the step-doubling controller
* Advances per-world ``sim_time`` only for accepted steps

No ``numpy`` arrays or ``.numpy()`` calls appear in the hot path.
"""

from __future__ import annotations

import warp as wp

from ...core.types import override
from ...sim import Contacts, Control, Model, State
from ...utils.benchmark import event_scope
from .solver_variable_step_mujoco import SolverVariableStepMuJoCo

# ---------------------------------------------------------------------------
# Warp kernels — module-level so Warp compiles them once at import time.
# ---------------------------------------------------------------------------


@wp.kernel
def _error_control_kernel(
    # Full-step and double-half-step joint states for error estimation
    joint_q_full: wp.array(dtype=wp.float32),
    joint_qd_full: wp.array(dtype=wp.float32),
    joint_q_double: wp.array(dtype=wp.float32),
    joint_qd_double: wp.array(dtype=wp.float32),
    # Per-world layout sizes
    coords_per_world: int,
    dofs_per_world: int,
    # Step-doubling control parameters
    tol: float,
    dt_safety: float,
    dt_min: float,
    dt_max: float,
    # Per-world state — read then updated in-place (each thread owns its slot)
    dt: wp.array(dtype=wp.float32),
    sim_time: wp.array(dtype=wp.float32),
    # Pure outputs
    dt_half: wp.array(dtype=wp.float32),
    accepted: wp.array(dtype=wp.bool),
    last_error: wp.array(dtype=wp.float32),
):
    """One thread per world.

    Computes RMS integration error between the full step and the double
    half-step, then updates dt and sim_time for that world.

    Guards and hysteresis (mirroring Drake ``IntegratorBase``):

    * **NaN/Inf guard** — diverged simulations produce ``error = 1e10`` so
      ``dt`` shrinks aggressively instead of becoming NaN
      (Drake ``CalcAdjustedStepSize``, L280–283).
    * **Accept-at-floor** — worlds stuck at ``dt_min`` always advance so they
      produce data rather than stalling indefinitely
      (Drake ``at_minimum_step_size`` flag).
    * **Growth hysteresis** (``kHysteresisHigh = 1.2``) — ``dt`` only grows if
      the computed new value is ≥ 20% larger, preventing jitter near the
      tolerance boundary.
    * **Shrink hysteresis** (``kHysteresisLow``) — genuinely accepted steps
      (error ≤ tol) never shrink ``dt``.
    """
    world = wp.tid()

    q_start = world * coords_per_world
    qd_start = world * dofs_per_world

    # --- RMS error over joint positions and velocities ---
    error_sq = float(0.0)
    for i in range(coords_per_world):
        d = joint_q_double[q_start + i] - joint_q_full[q_start + i]
        error_sq += d * d
    for i in range(dofs_per_world):
        d = joint_qd_double[qd_start + i] - joint_qd_full[qd_start + i]
        error_sq += d * d

    n = float(coords_per_world + dofs_per_world)
    error = wp.sqrt(error_sq / n)

    # NaN/Inf → simulation diverged. Force rejection and aggressive dt shrink.
    # (Drake IntegratorBase::CalcAdjustedStepSize, L280–283)
    if wp.isnan(error) or wp.isinf(error):
        error = float(1.0e10)

    last_error[world] = error

    old_dt = dt[world]

    true_acceptance = error <= tol
    at_floor = old_dt <= dt_min * float(1.001)  # fp-safe floor comparison
    is_accepted = true_acceptance or at_floor
    accepted[world] = is_accepted

    # Advance simulation time only when the step is accepted.
    if is_accepted:
        sim_time[world] = sim_time[world] + old_dt

    # Step-doubling controller for a 1st-order method:
    #   dt_new = dt * safety * sqrt(tol / error)
    # The sqrt comes from exponent 1/(p+1) with p=1 (Euler).
    # The clamp prevents runaway growth or collapse in a single step.
    # kMaxGrow = 5.0 matches Drake's IntegratorBase default.
    ratio = tol / wp.max(error, float(1.0e-10))
    factor = wp.clamp(dt_safety * wp.sqrt(ratio), float(0.1), float(5.0))

    new_dt = wp.clamp(old_dt * factor, dt_min, dt_max)

    # Growth hysteresis: only grow dt when the gain is meaningful.
    # (Drake kHysteresisHigh = 1.2 — avoids jitter near the tolerance)
    if new_dt > old_dt and new_dt < old_dt * float(1.2):
        new_dt = old_dt

    # Shrink hysteresis: genuine accepts never reduce dt.
    # (Drake kHysteresisLow logic)
    if true_acceptance and new_dt < old_dt:
        new_dt = old_dt

    dt[world] = new_dt
    dt_half[world] = new_dt * wp.float32(0.5)


@wp.kernel
def _select_float_kernel(
    candidate: wp.array(dtype=wp.float32),
    fallback: wp.array(dtype=wp.float32),
    accepted: wp.array(dtype=wp.bool),
    stride: int,
    out: wp.array(dtype=wp.float32),
):
    """One thread per element (joint coordinate or DOF).

    Writes ``candidate[i]`` for accepted worlds, ``fallback[i]`` for rejected.
    """
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
    """One thread per body. Selects body pose from accepted or fallback state."""
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
    """One thread per body. Selects body velocity from accepted or fallback state."""
    i = wp.tid()
    world = i // stride
    if accepted[world]:
        out[i] = candidate[i]
    else:
        out[i] = fallback[i]


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------


class SolverMuJoCoCENIC(SolverVariableStepMuJoCo):
    """Adaptive-step MuJoCo solver for high-accuracy dataset generation.

    Uses step doubling to estimate per-world integration error and adapt the
    timestep entirely on the GPU — no CPU-GPU transfers in the simulation loop.

    Each call to :meth:`step` runs three MuJoCo evaluations (one full step and
    two half-steps) and either advances a world's state or leaves it unchanged,
    depending on whether the local error is within ``tol``.  Per-world ``dt``
    grows for easy regions and shrinks for stiff or nonlinear ones, so the
    resulting dataset is naturally denser where the dynamics are hardest.

    Note:
        The ``dt`` parameter accepted by the parent :class:`SolverVariableStepMuJoCo`
        is not exposed here — timesteps are managed internally.  Set the initial
        value via ``dt_init`` and query current values via :attr:`dt`.

    Example:

    .. code-block:: python

        solver = newton.solvers.SolverMuJoCoCENIC(model, tol=1e-3, dt_init=0.01)
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
        dt_init: float = 0.01,
        dt_min: float = 1e-4,
        dt_max: float = 0.1,
        dt_safety: float = 0.9,
        **kwargs,
    ):
        """
        Args:
            model: The model to simulate.
            tol: RMS integration error tolerance per world [same units as joint_q/qd].
                Worlds with error > tol are rejected and retry with a smaller dt.
            dt_init: Initial timestep [s] for all worlds.
            dt_min: Minimum allowed timestep [s].
            dt_max: Maximum allowed timestep [s].
            dt_safety: Safety factor (< 1) applied to the step-doubling dt controller.
                Smaller values are more conservative.
            **kwargs: Forwarded to :class:`SolverVariableStepMuJoCo`.
        """
        super().__init__(model, **kwargs)

        world_count = model.world_count
        device = model.device

        # --- Per-world adaptive timestep state (all on GPU, never transferred) ---
        self._dt = wp.full(world_count, dt_init, dtype=wp.float32, device=device)
        self._dt_half = wp.full(world_count, dt_init * 0.5, dtype=wp.float32, device=device)
        self._sim_time = wp.zeros(world_count, dtype=wp.float32, device=device)
        self._accepted = wp.zeros(world_count, dtype=wp.bool, device=device)
        self._last_error = wp.zeros(world_count, dtype=wp.float32, device=device)

        # Error tolerance and controller parameters
        self._tol = float(tol)
        self._dt_min = float(dt_min)
        self._dt_max = float(dt_max)
        self._dt_safety = float(dt_safety)

        # --- Scratch states for the three-evaluation pipeline ---
        # These live on the GPU and are reused every step — no allocations in the loop.
        self._scratch_full = model.state()  # result of one full step (dt)
        self._scratch_mid = model.state()  # result of the first half-step (dt/2)
        self._scratch_double = model.state()  # result of two half-steps (2 × dt/2)

        # Per-world layout — used to map flat array indices to world indices.
        self._coords_per_world = model.joint_coord_count // world_count
        self._dofs_per_world = model.joint_dof_count // world_count
        self._bodies_per_world = model.body_count // world_count

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_substep(
        self,
        state_in: State,
        state_out: State,
        contacts: Contacts,
        dt_array: wp.array,
    ) -> None:
        """Run one MuJoCo step from ``state_in`` → ``state_out``.

        Always syncs ``mjw_data`` from ``state_in`` before stepping, because
        the three substeps each start from a different initial state.
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
        """Estimate error, adapt dt, and write the correct state to ``state_out``.

        All work happens in Warp kernels on the GPU:
          1. ``_error_control_kernel``   — per world, compute RMS error, update dt and sim_time
          2. ``_select_*_kernel`` calls  — per element, copy from scratch_double or state_in
        """
        model = self.model
        device = model.device

        # --- Step 1: error estimation, dt adaptation, time advance ---
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
                self._dt,  # read old dt, written with new dt in-place
                self._sim_time,  # written in-place for accepted worlds
            ],
            outputs=[self._dt_half, self._accepted, self._last_error],
            device=device,
        )

        # --- Step 2: state selection (separate kernel per dtype) ---

        # Joint positions — float32, stride = coords per world
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

        # Joint velocities — float32, stride = DOFs per world
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

        # Body poses — wp.transform, stride = bodies per world
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

        # Body velocities — wp.spatial_vector, stride = bodies per world
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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

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

        The three MuJoCo evaluations and the error control kernel all run
        on the GPU.  No data is transferred to the CPU.

        Args:
            state_in:  Input state.
            state_out: Output state (written in place).
            control:   Control inputs, applied identically to all three evaluations.
            contacts:  Contact data (only used when ``use_mujoco_contacts`` is False).

        Returns:
            state_out
        """
        # Apply control once. It is written to mjw_data.ctrl, which is not
        # reset by _update_mjc_data (that only resets qpos/qvel), so the same
        # control is active throughout all three substep evaluations.
        self._apply_mjc_control(self.model, state_in, control, self.mjw_data)
        self._enable_rne_postconstraint(state_out)

        # ── Eval 1: full step ──────────────────────────────────────────────
        #   state_in ──────── dt ────────► scratch_full
        self._run_substep(state_in, self._scratch_full, contacts, self._dt)

        # ── Eval 2: first half-step ────────────────────────────────────────
        #   state_in ──────── dt/2 ──────► scratch_mid
        self._run_substep(state_in, self._scratch_mid, contacts, self._dt_half)

        # ── Eval 3: second half-step ───────────────────────────────────────
        #   scratch_mid ───── dt/2 ──────► scratch_double   (2 × dt/2 total)
        self._run_substep(self._scratch_mid, self._scratch_double, contacts, self._dt_half)

        # ── Error control + state selection (GPU only) ────────────────────
        #   Accepted worlds → state_out = scratch_double, sim_time += dt
        #   Rejected worlds → state_out = state_in,       sim_time unchanged
        self._error_control_and_select_state(state_in, state_out)

        self._step += 1
        return state_out

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
