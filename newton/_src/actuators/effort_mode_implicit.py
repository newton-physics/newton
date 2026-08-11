# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Coupled implicit effort mode for actuators.

For each articulation, the mode solves for the actuator impulse ``p`` at the
predicted end-of-step state:

    ``r(p) = p - h g(q(p), qd(p)) = 0``

    ``qd(p) = qd + A p``

    ``q(p) = q + h qd(p)``

Here ``h`` is the timestep, ``g`` is the controller force law with clamping,
and ``A`` is the coupled inverse-mass response supplied by
:class:`ResponseOracle`. Controller integration is defined by
:class:`Controller`; solver configuration is provided by
:class:`ActuatorImplicitOptions`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import warp as wp

from ..sim import JointType
from .response_oracle import ResponseOracle

__all__ = ["ActuatorImplicitOptions", "ResponseOracle"]


# ---------------------------------------------------------------------------
# Solver options
# ---------------------------------------------------------------------------


@dataclass
class ActuatorImplicitOptions:
    """Configuration for implicit actuation; see :meth:`Actuator.set_effort_mode_implicit`.

    Args:
        max_iters: Maximum Newton iterations per articulation group.
        residual_tol: Stop when the residual vector norm falls below this
            [N·s or N·m·s]. The residual is an impulse.
        update_tol: Stop when the impulse-update vector norm falls below this
            [N·s or N·m·s].
        fd_epsilon: Relative forward finite-difference step in impulse space
            (dimensionless; scaled by ``1 + |p|``).
        derivative_floor: Smallest Jacobian pivot used during elimination and
            back-substitution (dimensionless: the Jacobian is d(impulse)/d(impulse)).
        warm_start: Initial impulse guess: ``"explicit"`` starts from the
            explicit force impulse, ``"zero"`` starts from zero.
    """

    max_iters: int = 4
    residual_tol: float = 1.0e-5
    update_tol: float = 1.0e-5
    fd_epsilon: float = 1.0e-4
    derivative_floor: float = 1.0e-8
    warm_start: str = "explicit"


# ---------------------------------------------------------------------------
# Clamp chain: fold every implicit-capable clamp into one @wp.func
# ---------------------------------------------------------------------------


@wp.func
def _identity_clamp_chain(
    value: wp.float64, q: wp.float64, qd: wp.float64, params: wp.array2d[float], i: int
) -> wp.float64:
    """Chain used when the actuator has no implicit-capable clamp."""
    return value


def _compose_clamps(entries: tuple) -> wp.Function:
    """Compose ``(evaluate_clamp, base_column)`` entries into one ``@wp.func``.

    Application order matches the actuator's clamping list. ``base_column``
    is the clamp's offset into the packed clamp-params array.
    """
    chain = _identity_clamp_chain
    for func, base in entries:
        chain = _chain_clamp(chain, func, base)
    return chain


def _chain_clamp(inner: wp.Function, func: wp.Function, base: int) -> wp.Function:
    """Wrap *inner* with one more clamp; closes over the function and its offset."""

    @wp.func
    def chained(value: wp.float64, q: wp.float64, qd: wp.float64, params: wp.array2d[float], i: int) -> wp.float64:
        return func(inner(value, q, qd, params, i), q, qd, params, i, wp.static(base))

    return chained


@wp.kernel(enable_backward=False)
def _gather_slot_response_kernel(
    inverse_blocks: wp.array3d[float],
    slot_art: wp.array[wp.int32],
    slot_local: wp.array[wp.int32],
    slot_response: wp.array[float],
):
    """Per-slot diagonal response A_ii, for laws that need their own inverse mass."""
    i = wp.tid()
    li = slot_local[i]
    slot_response[i] = inverse_blocks[slot_art[i], li, li]


# ---------------------------------------------------------------------------
# Coupled solve kernel
# ---------------------------------------------------------------------------

_coupled_kernel_cache: dict[tuple[Any, Any], wp.Kernel] = {}


def _build_coupled_solve_kernel(evaluate_force: wp.Function, clamp_chain: wp.Function, cache_key: tuple):
    """Build or reuse the coupled solve kernel for a force law and clamp chain.

    One thread handles each articulation group. It predicts the group state
    with its inverse-mass response, evaluates the force laws and clamps, forms
    a finite-difference Jacobian, and applies a dense Newton update.

    Float64 avoids loss of precision when finite differencing stiff residuals.
    """
    cached = _coupled_kernel_cache.get(cache_key)
    if cached is not None:
        return cached

    @wp.kernel
    def solve(
        group_size: wp.array[wp.int32],
        group_art: wp.array[wp.int32],
        group_slot: wp.array2d[wp.int32],
        group_local: wp.array2d[wp.int32],
        inverse_blocks: wp.array3d[float],
        pos_indices: wp.array[wp.uint32],
        vel_indices: wp.array[wp.uint32],
        target_pos_indices: wp.array[wp.uint32],
        target_vel_indices: wp.array[wp.uint32],
        positions: wp.array[float],
        velocities: wp.array[float],
        target_pos: wp.array[float],
        target_vel: wp.array[float],
        feedforward: wp.array[float],
        params: wp.array2d[float],
        h: float,
        max_iters: int,
        residual_tol: float,
        update_tol: float,
        fd_epsilon: float,
        derivative_floor: float,
        warm_zero: int,
        clamp_params: wp.array2d[float],
        pbuf: wp.array2d[wp.float64],
        rbuf: wp.array2d[wp.float64],
        jbuf: wp.array3d[wp.float64],
        computed_efforts: wp.array[float],
        applied_efforts: wp.array[float],
    ):
        g = wp.tid()
        ng = group_size[g]
        art = group_art[g]
        hd = wp.float64(h)
        fd_eps = wp.float64(fd_epsilon)
        res_tol = wp.float64(residual_tol)
        upd_tol = wp.float64(update_tol)
        dfloor = wp.float64(derivative_floor)

        # Warm start: the clamped explicit impulse, or zero.
        for i in range(ng):
            si = group_slot[g, i]
            if warm_zero != 0:
                pbuf[g, i] = wp.float64(0.0)
            else:
                q0 = wp.float64(positions[pos_indices[si]])
                qd0 = wp.float64(velocities[vel_indices[si]])
                tq = wp.float64(target_pos[target_pos_indices[si]])
                tqd = wp.float64(target_vel[target_vel_indices[si]])
                ff = wp.float64(0.0)
                if feedforward:
                    ff = wp.float64(feedforward[target_vel_indices[si]])
                tau0 = evaluate_force(q0, qd0, tq, tqd, ff, params, si)
                pbuf[g, i] = hd * clamp_chain(tau0, q0, qd0, clamp_params, si)

        for _ in range(max_iters):
            # Residual at the current impulse guess (state coupled via A_g).
            rn = wp.float64(0.0)
            for i in range(ng):
                si = group_slot[g, i]
                li = group_local[g, i]
                qd_i = wp.float64(velocities[vel_indices[si]])
                for jj in range(ng):
                    qd_i += wp.float64(inverse_blocks[art, li, group_local[g, jj]]) * pbuf[g, jj]
                q_i = wp.float64(positions[pos_indices[si]]) + hd * qd_i
                tq = wp.float64(target_pos[target_pos_indices[si]])
                tqd = wp.float64(target_vel[target_vel_indices[si]])
                ff = wp.float64(0.0)
                if feedforward:
                    ff = wp.float64(feedforward[target_vel_indices[si]])
                f_i = clamp_chain(evaluate_force(q_i, qd_i, tq, tqd, ff, params, si), q_i, qd_i, clamp_params, si)
                ri = pbuf[g, i] - hd * f_i
                rbuf[g, i] = ri
                rn += ri * ri
            if rn < res_tol * res_tol:
                break

            # Jacobian columns by forward finite difference: perturb p_c (which
            # shifts every predicted state through A_g) and re-form the residual.
            for c in range(ng):
                psave = pbuf[g, c]
                eps = fd_eps * (wp.float64(1.0) + wp.abs(psave))
                pbuf[g, c] = psave + eps
                for i in range(ng):
                    si = group_slot[g, i]
                    li = group_local[g, i]
                    qd_i = wp.float64(velocities[vel_indices[si]])
                    for jj in range(ng):
                        qd_i += wp.float64(inverse_blocks[art, li, group_local[g, jj]]) * pbuf[g, jj]
                    q_i = wp.float64(positions[pos_indices[si]]) + hd * qd_i
                    tq = wp.float64(target_pos[target_pos_indices[si]])
                    tqd = wp.float64(target_vel[target_vel_indices[si]])
                    ff = wp.float64(0.0)
                    if feedforward:
                        ff = wp.float64(feedforward[target_vel_indices[si]])
                    f_i = clamp_chain(evaluate_force(q_i, qd_i, tq, tqd, ff, params, si), q_i, qd_i, clamp_params, si)
                    r_pert = pbuf[g, i] - hd * f_i
                    jbuf[g, i, c] = (r_pert - rbuf[g, i]) / eps
                pbuf[g, c] = psave

            # Dense Newton step: solve J dp = -r by Gauss elimination, update p.
            for i in range(ng):
                rbuf[g, i] = -rbuf[g, i]
            for k in range(ng):
                piv = jbuf[g, k, k]
                if wp.abs(piv) < dfloor:
                    piv = wp.where(piv < wp.float64(0.0), -dfloor, dfloor)
                    # Back-substitution divides by this diagonal, so floor it too.
                    jbuf[g, k, k] = piv
                for i in range(k + 1, ng):
                    fac = jbuf[g, i, k] / piv
                    for j in range(k, ng):
                        jbuf[g, i, j] -= fac * jbuf[g, k, j]
                    rbuf[g, i] -= fac * rbuf[g, k]
            dpn = wp.float64(0.0)
            for kk in range(ng):
                i = ng - 1 - kk
                s = rbuf[g, i]
                for j in range(i + 1, ng):
                    s -= jbuf[g, i, j] * rbuf[g, j]
                dv = s / jbuf[g, i, i]
                rbuf[g, i] = dv
                pbuf[g, i] += dv
                dpn += dv * dv
            if dpn < upd_tol * upd_tol:
                break

        # Re-clamp at the final predicted state and write effort.
        for i in range(ng):
            si = group_slot[g, i]
            li = group_local[g, i]
            qd_i = wp.float64(velocities[vel_indices[si]])
            for jj in range(ng):
                qd_i += wp.float64(inverse_blocks[art, li, group_local[g, jj]]) * pbuf[g, jj]
            q_i = wp.float64(positions[pos_indices[si]]) + hd * qd_i
            tq = wp.float64(target_pos[target_pos_indices[si]])
            tqd = wp.float64(target_vel[target_vel_indices[si]])
            ff = wp.float64(0.0)
            if feedforward:
                ff = wp.float64(feedforward[target_vel_indices[si]])
            raw_effort = evaluate_force(q_i, qd_i, tq, tqd, ff, params, si)
            computed_efforts[si] = wp.float32(raw_effort)
            applied_efforts[si] = wp.float32(clamp_chain(pbuf[g, i] / hd, q_i, qd_i, clamp_params, si))

    _coupled_kernel_cache[cache_key] = solve
    return solve


# ---------------------------------------------------------------------------
# The mode object installed by Actuator.set_effort_mode_implicit
# ---------------------------------------------------------------------------


class _EffortModeImplicit:
    """Implicit effort mode and in-kernel solver.

    Groups actuator DOFs by articulation and solves each group using the
    response provided by :class:`ResponseOracle`. The generated kernel
    combines the controller force law, controller parameters, and clamps.

    Before each solve, :meth:`compute_force` calls the controller's
    :meth:`~Controller.prepare_implicit` hook to update state-dependent
    controller parameters.
    """

    def __init__(
        self,
        controller,
        clamping,
        effective_inv_mass: ResponseOracle | None,
        options: ActuatorImplicitOptions | None,
        num_actuators: int,
        device: wp.Device,
        vel_indices: wp.array[wp.uint32] | None = None,
    ):
        self._options = options or ActuatorImplicitOptions()
        if self._options.warm_start not in ("explicit", "zero"):
            raise ValueError(f"warm_start must be 'explicit' or 'zero', got {self._options.warm_start!r}")
        self._num_actuators = num_actuators
        self._device = device
        if not isinstance(effective_inv_mass, ResponseOracle):
            raise ValueError(
                "Implicit actuation requires effective_inv_mass to be a ResponseOracle; "
                "build one with newton.actuators.ResponseOracle(model)."
            )
        self._response = effective_inv_mass
        self._controller = controller
        self._init_solver(controller, clamping)
        # Up front: this reads to host and allocates, both illegal during graph capture.
        if vel_indices is not None:
            self._build_groups(vel_indices)

    def _resolve_force_law(self, controller):
        """Validate the controller's in-kernel force law and adopt its params.

        ``bind_params`` builds the pack and re-points the controller's
        parameter attributes to views into it, so later writes stay live.
        """
        # Check first: bind_params() re-points the controller's parameter arrays.
        if controller.evaluate_force is None:
            raise NotImplementedError(
                f"{type(controller).__name__} does not support implicit actuation "
                "(Controller.evaluate_force unavailable)"
            )
        params = controller.bind_params()
        if params is None:
            raise NotImplementedError(
                f"{type(controller).__name__} does not support implicit actuation "
                "in this configuration (Controller.bind_params() returned None)"
            )
        self._params = params

    def _pack_clamps(self, clamping):
        """Allocate one packed clamp-param array, bind each clamp to its slice.

        Sets :attr:`_clamp_params` and returns ``(chain, entries)`` for the
        solve-kernel cache key. ``bind_params`` fills each slice and re-points
        the clamp's parameter attributes at it, so user writes (e.g.
        ``clamp.max_effort``) stay visible to the solve kernel.
        """
        entries: list[tuple[wp.Function, int]] = []
        widths: list[int] = []
        col = 0
        for clamp in clamping or []:
            func = clamp.evaluate_clamp
            if func is None:
                raise NotImplementedError(
                    f"{type(clamp).__name__} does not support implicit actuation (Clamping.evaluate_clamp unavailable)"
                )
            width = clamp.param_width()
            entries.append((func, col))
            widths.append(width)
            col += width
        self._clamp_params = wp.zeros((self._num_actuators, max(col, 1)), dtype=float, device=self._device)
        for clamp, (_func, base), width in zip(clamping or [], entries, widths, strict=True):
            clamp.bind_params(self._clamp_params[:, base : base + width])
        return _compose_clamps(tuple(entries)), tuple(entries)

    def _init_solver(self, controller, clamping) -> None:
        """Build the coupled in-kernel solve from the controller and clamps."""
        self._resolve_force_law(controller)
        chain, entries = self._pack_clamps(clamping)
        key = (controller.evaluate_force, entries)
        self._kernel = _build_coupled_solve_kernel(controller.evaluate_force, chain, key)
        self._groups_built = False

    def _build_groups(self, vel_indices) -> None:
        """Map actuator DOFs to (articulation, local index) and group by articulation."""
        model = self._response.model
        dofs = vel_indices.numpy().astype(np.int64)
        joint_qd_start = model.joint_qd_start.numpy()
        art_start = model.articulation_start.numpy()
        art_end = model.articulation_end.numpy()

        art_base = []
        art_ndof = []
        for a in range(model.articulation_count):
            base = int(joint_qd_start[int(art_start[a])])
            end = int(joint_qd_start[int(art_end[a])])
            art_base.append(base)
            art_ndof.append(end - base)

        # DOF ranges are contiguous per articulation, so one sorted search maps them all.
        base_arr = np.asarray(art_base, dtype=np.int64)
        ndof_arr = np.asarray(art_ndof, dtype=np.int64)
        order = np.argsort(base_arr)
        found = order[np.clip(np.searchsorted(base_arr[order], dofs, side="right") - 1, 0, None)]
        in_range = (np.searchsorted(base_arr[order], dofs, side="right") > 0) & (
            dofs < base_arr[found] + ndof_arr[found]
        )
        if not np.all(in_range):
            bad = int(dofs[~in_range][0])
            raise ValueError(f"Implicit actuation: DOF {bad} is not in an articulation")

        # q + h*qd only integrates 1-DOF axes; BALL/FREE store quaternion components.
        joint_type = model.joint_type.numpy()
        joint_qd_start_all = model.joint_qd_start.numpy()
        scalar_types = (int(JointType.REVOLUTE), int(JointType.PRISMATIC))
        dof_ok = np.zeros(int(model.joint_dof_count), dtype=bool)
        for jt, lo, hi in zip(joint_type, joint_qd_start_all[:-1], joint_qd_start_all[1:], strict=True):
            if int(jt) in scalar_types:
                dof_ok[lo:hi] = True
        bad_dofs = [int(d) for d in dofs if not dof_ok[int(d)]]
        if bad_dofs:
            raise ValueError(
                f"Implicit actuation supports REVOLUTE and PRISMATIC joints only; "
                f"DOF {bad_dofs[0]} belongs to another joint type"
            )

        groups: dict[int, list[tuple[int, int]]] = {}  # art -> [(slot, local_dof)]
        for slot, (dof, a) in enumerate(zip(dofs, found, strict=True)):
            groups.setdefault(int(a), []).append((slot, int(dof) - art_base[int(a)]))

        arts = sorted(groups)
        num_groups = len(arts)
        max_ng = max(len(groups[a]) for a in arts)
        device = self._device

        size = np.zeros(num_groups, dtype=np.int32)
        art_id = np.zeros(num_groups, dtype=np.int32)
        slot = np.zeros((num_groups, max_ng), dtype=np.int32)
        local = np.zeros((num_groups, max_ng), dtype=np.int32)
        for gi, a in enumerate(arts):
            size[gi] = len(groups[a])
            art_id[gi] = a
            for i, (s, ld) in enumerate(groups[a]):
                slot[gi, i] = s
                local[gi, i] = ld

        self._group_size = wp.array(size, dtype=wp.int32, device=device)
        self._group_art = wp.array(art_id, dtype=wp.int32, device=device)
        self._group_slot = wp.array(slot, dtype=wp.int32, device=device)
        self._group_local = wp.array(local, dtype=wp.int32, device=device)
        self._pbuf = wp.zeros((num_groups, max_ng), dtype=wp.float64, device=device)
        self._rbuf = wp.zeros((num_groups, max_ng), dtype=wp.float64, device=device)
        self._jbuf = wp.zeros((num_groups, max_ng, max_ng), dtype=wp.float64, device=device)
        slot_art = np.zeros(self._num_actuators, dtype=np.int32)
        slot_local = np.zeros(self._num_actuators, dtype=np.int32)
        for a in arts:
            for slot_idx, ld in groups[a]:
                slot_art[slot_idx] = a
                slot_local[slot_idx] = ld
        self._slot_art = wp.array(slot_art, dtype=wp.int32, device=device)
        self._slot_local = wp.array(slot_local, dtype=wp.int32, device=device)
        self._slot_response = wp.zeros(self._num_actuators, dtype=float, device=device)
        self._num_groups = num_groups
        self._groups_built = True

    def is_graphable(self) -> bool:
        return True

    def compute_force(
        self,
        sim_state: Any,
        positions: wp.array[float],
        velocities: wp.array[float],
        target_pos: wp.array[float],
        target_vel: wp.array[float],
        feedforward: wp.array[float] | None,
        pos_indices: wp.array[wp.uint32],
        vel_indices: wp.array[wp.uint32],
        target_pos_indices: wp.array[wp.uint32],
        target_vel_indices: wp.array[wp.uint32],
        computed_forces: wp.array[float],
        applied_forces: wp.array[float] | None,
        ctrl_state: Any,
        dt: float | None,
    ) -> wp.array[float]:
        """Solve implicit effort and return the applied-effort buffer.

        The controller law at the final predicted state is written to
        *computed_forces*. Clamps are enforced inside the solve against that
        state, and the solved effort is written to *applied_forces*.
        """
        if dt is None:
            raise ValueError("Implicit actuation requires dt")
        if dt <= 0.0:
            raise ValueError(f"Implicit actuation requires dt > 0, got {dt}")
        if applied_forces is None:
            raise RuntimeError("Implicit actuation requires an applied-effort buffer")
        # Update state-dependent controller parameters before solving.
        if not self._groups_built:
            self._build_groups(vel_indices)
        wp.launch(
            _gather_slot_response_kernel,
            dim=self._num_actuators,
            inputs=[self._response.inverse_blocks, self._slot_art, self._slot_local],
            outputs=[self._slot_response],
            device=self._device,
        )
        self._controller.prepare_implicit(
            positions,
            velocities,
            target_pos,
            target_vel,
            pos_indices,
            vel_indices,
            target_pos_indices,
            target_vel_indices,
            ctrl_state,
            float(dt),
            self._slot_response,
            self._device,
        )
        inverse_blocks = self._response.inverse_blocks

        opts = self._options
        wp.launch(
            self._kernel,
            dim=self._num_groups,
            inputs=[
                self._group_size,
                self._group_art,
                self._group_slot,
                self._group_local,
                inverse_blocks,
                pos_indices,
                vel_indices,
                target_pos_indices,
                target_vel_indices,
                positions,
                velocities,
                target_pos,
                target_vel,
                feedforward,
                self._params,
                float(dt),
                int(opts.max_iters),
                float(opts.residual_tol),
                float(opts.update_tol),
                float(opts.fd_epsilon),
                float(opts.derivative_floor),
                1 if opts.warm_start == "zero" else 0,
                self._clamp_params,
                self._pbuf,
                self._rbuf,
                self._jbuf,
            ],
            outputs=[computed_forces, applied_forces],
            device=self._device,
        )
        return applied_forces
