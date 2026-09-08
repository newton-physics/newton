# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Coupled implicit effort mode for actuators.

For each articulation, the mode solves for the actuator impulse ``p`` at the
predicted end-of-step state:

    ``r(p) = p - h g(q(p), qd(p)) = 0``

    ``qd(p) = qd + A p``

    ``q(p) = q + h qd(p)``

Here ``h`` is the timestep, ``g`` is the drive force law with clamping,
and ``A`` is the coupled inverse-mass response supplied by
:class:`ResponseOracle`. Options: :class:`ImplicitOptions`.

``qd(p)`` advances the step-start velocity by this actuator's own impulse alone.
Gravity, any other applied force, other actuators on the same articulation, and
joint drive applied without the actuator do not enter the prediction.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import warp as wp

from ..sim import JointType
from .drives.base import DriveBase
from .response_oracle import ResponseOracle

__all__ = ["ImplicitOptions", "ResponseOracle"]


# ---------------------------------------------------------------------------
# Solver options
# ---------------------------------------------------------------------------


@dataclass
class ImplicitOptions:
    """Configuration for implicit actuation; see :meth:`Actuator.set_effort_mode_implicit`."""

    class WarmStart(str, Enum):
        """Initial impulse guess for the Newton solve."""

        EXPLICIT = "explicit"
        """Start from the clamped explicit force impulse."""

        ZERO = "zero"
        """Start from zero impulse."""

    max_iters: int = 4
    """Maximum Newton iterations per articulation group."""

    residual_tol: float = 1.0e-5
    """Stop when the residual vector norm falls below this [N·s or N·m·s]."""

    update_tol: float = 1.0e-5
    """Stop when the impulse-update vector norm falls below this [N·s or N·m·s]."""

    fd_epsilon: float = 1.0e-4
    """Relative forward finite-difference step in velocity space (dimensionless)."""

    derivative_floor: float = 1.0e-8
    """Smallest Jacobian pivot used during elimination and back-substitution
    (dimensionless: the Jacobian is d(impulse)/d(impulse))."""

    warm_start: WarmStart = WarmStart.EXPLICIT
    """Initial impulse guess."""


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

    @wp.func
    def predict_qd(
        velocities: wp.array[float],
        vel_indices: wp.array[wp.uint32],
        inverse_blocks: wp.array3d[float],
        group_local: wp.array2d[wp.int32],
        pbuf: wp.array2d[wp.float64],
        art: wp.int32,
        g: wp.int32,
        i: wp.int32,
        ng: wp.int32,
        si: wp.int32,
    ) -> wp.float64:
        """End-of-step velocity of row *i*: ``qd + (A p)_i``."""
        qd_i = wp.float64(velocities[vel_indices[si]])
        li = group_local[g, i]
        for jj in range(ng):
            qd_i += wp.float64(inverse_blocks[art, li, group_local[g, jj]]) * pbuf[g, jj]
        return qd_i

    @wp.func
    def force_at(
        q_i: wp.float64,
        qd_i: wp.float64,
        target_pos: wp.array[float],
        target_vel: wp.array[float],
        feedforward: wp.array[float],
        target_pos_indices: wp.array[wp.uint32],
        target_vel_indices: wp.array[wp.uint32],
        params: wp.array2d[float],
        clamp_params: wp.array2d[float],
        si: wp.int32,
    ) -> wp.float64:
        """Clamped control law of row *i* at a predicted state."""
        tq = wp.float64(target_pos[target_pos_indices[si]])
        tqd = wp.float64(target_vel[target_vel_indices[si]])
        ff = wp.float64(0.0)
        if feedforward:
            ff = wp.float64(feedforward[target_vel_indices[si]])
        return clamp_chain(evaluate_force(q_i, qd_i, tq, tqd, ff, params, si), q_i, qd_i, clamp_params, si)

    @wp.kernel(enable_backward=False)
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
        sbuf: wp.array2d[wp.float64],
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
                pbuf[g, i] = hd * force_at(
                    q0,
                    qd0,
                    target_pos,
                    target_vel,
                    feedforward,
                    target_pos_indices,
                    target_vel_indices,
                    params,
                    clamp_params,
                    si,
                )

        for _ in range(max_iters):
            # Residual at the current impulse guess (state coupled via A_g), plus
            # the row slope s_i = h*df_i/dq + df_i/dqd by one forward difference.
            rn = wp.float64(0.0)
            for i in range(ng):
                si = group_slot[g, i]
                qd_i = predict_qd(velocities, vel_indices, inverse_blocks, group_local, pbuf, art, g, i, ng, si)
                q_i = wp.float64(positions[pos_indices[si]]) + hd * qd_i
                f_i = force_at(
                    q_i,
                    qd_i,
                    target_pos,
                    target_vel,
                    feedforward,
                    target_pos_indices,
                    target_vel_indices,
                    params,
                    clamp_params,
                    si,
                )
                ri = pbuf[g, i] - hd * f_i
                rbuf[g, i] = ri
                rn += ri * ri

                eps = fd_eps * (wp.float64(1.0) + wp.abs(qd_i))
                f_p = force_at(
                    q_i + hd * eps,
                    qd_i + eps,
                    target_pos,
                    target_vel,
                    feedforward,
                    target_pos_indices,
                    target_vel_indices,
                    params,
                    clamp_params,
                    si,
                )
                sbuf[g, i] = (f_p - f_i) / eps
            if rn < res_tol * res_tol:
                break

            # Row i depends on p only through u_i = (A p)_i, so the Jacobian is
            # exactly dr_i/dp_c = delta_ic - h * s_i * A[li, lc].
            for i in range(ng):
                li = group_local[g, i]
                si_slope = sbuf[g, i]
                for c in range(ng):
                    jij = -hd * si_slope * wp.float64(inverse_blocks[art, li, group_local[g, c]])
                    if i == c:
                        jij += wp.float64(1.0)
                    jbuf[g, i, c] = jij

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
            qd_i = predict_qd(velocities, vel_indices, inverse_blocks, group_local, pbuf, art, g, i, ng, si)
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
    combines the drive force law, drive parameters, and clamps.

    Before each solve, :meth:`compute_force` calls the drive's
    :meth:`~DriveBase.prepare_implicit` hook to update state-dependent drive
    parameters.
    """

    def __init__(
        self,
        drive,
        clamping,
        response: ResponseOracle,
        options: ImplicitOptions | None,
        num_actuators: int,
        device: wp.Device,
        vel_indices: wp.array[wp.uint32],
    ):
        self._options = options or ImplicitOptions()
        try:
            self._options.warm_start = ImplicitOptions.WarmStart(self._options.warm_start)
        except ValueError:
            valid = ", ".join(repr(w.value) for w in ImplicitOptions.WarmStart)
            raise ValueError(f"warm_start must be one of {valid}, got {self._options.warm_start!r}") from None
        if self._options.fd_epsilon <= 0.0:
            # The Jacobian divides by this, so zero yields NaN.
            raise ValueError(f"fd_epsilon must be positive, got {self._options.fd_epsilon}")
        self._num_actuators = num_actuators
        self._device = device
        if not isinstance(response, ResponseOracle):
            raise ValueError(
                "Implicit actuation requires response to be a ResponseOracle; "
                "build one with newton.actuators.ResponseOracle(model)."
            )
        self._response = response
        self._drive = drive
        # Set for drives that require per-step preparation ahead of the implicit
        # solve, such as advancing an integral term or relinearizing a network.
        self._needs_prepare = type(drive).prepare_implicit is not DriveBase.prepare_implicit
        self._init_solver(drive, clamping)
        # Up front: this reads to host and allocates, both illegal during graph capture.
        self._build_groups(vel_indices)

    def _resolve_force_law(self, drive):
        """Validate the drive's in-kernel force law and adopt its params.

        ``bind_params`` builds the pack and re-points the drive's
        parameter attributes to views into it, so later writes stay live.
        """
        # Check first: bind_params() re-points the drive's parameter arrays.
        if drive.evaluate_force is None:
            raise NotImplementedError(
                f"{type(drive).__name__} does not support implicit actuation (DriveBase.evaluate_force unavailable)"
            )
        params = drive.bind_params()
        if params is None:
            raise NotImplementedError(
                f"{type(drive).__name__} does not support implicit actuation "
                "in this configuration (DriveBase.bind_params() returned None)"
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
                    f"{type(clamp).__name__} does not support implicit actuation "
                    "(ClampingBase.evaluate_clamp unavailable)"
                )
            width = clamp.param_width()
            entries.append((func, col))
            widths.append(width)
            col += width
        self._clamp_params = wp.zeros((self._num_actuators, max(col, 1)), dtype=float, device=self._device)
        for clamp, (_func, base), width in zip(clamping or [], entries, widths, strict=True):
            owner = getattr(clamp, "_bound_owner", None)
            if owner is not None and owner is not clamping:
                raise ValueError(
                    f"{type(clamp).__name__} is already bound to another actuator: binding it again "
                    "would detach the first actuator's parameters. Give each actuator its own instance."
                )
            clamp.bind_params(self._clamp_params[:, base : base + width])
            clamp._bound_owner = clamping
        return _compose_clamps(tuple(entries)), tuple(entries)

    def _init_solver(self, drive, clamping) -> None:
        """Build the coupled in-kernel solve from the drive and clamps."""
        self._resolve_force_law(drive)
        chain, entries = self._pack_clamps(clamping)
        key = (drive.evaluate_force, entries)
        self._kernel = _build_coupled_solve_kernel(drive.evaluate_force, chain, key)

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

        # q + h*qd needs one scalar coordinate per DOF; derive that from the
        # joint's layout rather than its type.
        joint_type = model.joint_type.numpy()
        joint_qd_start_all = model.joint_qd_start.numpy()
        joint_q_start_all = model.joint_q_start.numpy()
        dof_ok = np.zeros(int(model.joint_dof_count), dtype=bool)
        joint_of_dof = np.zeros(int(model.joint_dof_count), dtype=np.int64)
        for j, (qd_lo, qd_hi, q_lo, q_hi) in enumerate(
            zip(
                joint_qd_start_all[:-1],
                joint_qd_start_all[1:],
                joint_q_start_all[:-1],
                joint_q_start_all[1:],
                strict=True,
            )
        ):
            joint_of_dof[qd_lo:qd_hi] = j
            dof_ok[qd_lo:qd_hi] = (qd_hi - qd_lo) == (q_hi - q_lo)
        bad_dofs = [int(d) for d in dofs if not dof_ok[int(d)]]
        if bad_dofs:
            bad = bad_dofs[0]
            name = JointType(int(joint_type[joint_of_dof[bad]])).name
            raise ValueError(
                f"Implicit actuation requires one position coordinate per DOF; DOF {bad} belongs to a {name} joint"
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
        self._sbuf = wp.zeros((num_groups, max_ng), dtype=wp.float64, device=device)
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

    def is_graphable(self) -> bool:
        return self._drive.is_graphable()

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
        applied_forces: wp.array[float],
        drive_state: Any,
        dt: float | None,
    ) -> wp.array[float]:
        """Solve implicit effort and return the applied-effort buffer.

        The drive law at the final predicted state is written to
        *computed_forces*. Clamps are enforced inside the solve against that
        state, and the solved effort is written to *applied_forces*.
        """
        if dt is None:
            raise ValueError("Implicit actuation requires dt")
        if dt <= 0.0:
            raise ValueError(f"Implicit actuation requires dt > 0, got {dt}")
        if self._needs_prepare:
            wp.launch(
                _gather_slot_response_kernel,
                dim=self._num_actuators,
                inputs=[self._response.inverse_blocks, self._slot_art, self._slot_local],
                outputs=[self._slot_response],
                device=self._device,
            )
            self._drive.prepare_implicit(
                positions,
                velocities,
                target_pos,
                target_vel,
                pos_indices,
                vel_indices,
                target_pos_indices,
                target_vel_indices,
                drive_state,
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
                1 if opts.warm_start is ImplicitOptions.WarmStart.ZERO else 0,
                self._clamp_params,
                self._pbuf,
                self._rbuf,
                self._sbuf,
                self._jbuf,
            ],
            outputs=[computed_forces, applied_forces],
            device=self._device,
        )
        return applied_forces
