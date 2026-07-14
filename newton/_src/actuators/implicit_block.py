# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Block implicit effort strategy for coupled DOF groups.

Same solve as the scalar strategy — the controller's ``evaluate_force``
``@wp.func`` with a finite-difference Newton iteration and the clamp chain
composed into the residual — but coupled: each thread solves a whole group of
DOFs that share an articulation, predicting the end-of-step state through the
group's response block ``A_g = S_g M^{-1} S_g^T`` (a submatrix of
``oracle.inverse_blocks``) so the off-diagonal inertial cross terms are kept.

Per Newton iteration: form the ``n_g``-vector residual
``r(p) = p - h C(g(q(p), qd(p)))``, its ``n_g x n_g`` Jacobian by forward
finite differences, and take a dense Gauss-elimination Newton step. Reduces to
the scalar solve for ``n_g = 1``. Requires ``oracle.refresh(state,
blocks=True)`` each step.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import warp as wp

from .implicit import _EffortImplicit

_block_kernel_cache: dict[tuple[Any, Any], wp.Kernel] = {}


def _build_block_solve_kernel(evaluate_force: wp.Function, clamp_chain: wp.Function, cache_key: tuple):
    """Build (or reuse) the block solve kernel for one (force law, clamp chain) pair.

    One thread solves one coupled group with Newton's method: predict the
    group's end-of-step state through the block response ``A_g``, evaluate the
    clamped force law per DOF, form the block residual and its finite-difference
    Jacobian, and take a dense Newton step. Float64 for the same reason as the
    scalar solve — the FD slope is a small difference of large numbers.
    """
    cached = _block_kernel_cache.get(cache_key)
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
        efforts: wp.array[float],
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
            efforts[si] = wp.float32(clamp_chain(pbuf[g, i] / hd, q_i, qd_i, clamp_params, si))

    _block_kernel_cache[cache_key] = solve
    return solve


class _EffortImplicitBlock(_EffortImplicit):
    """Block implicit (Stable-PD) effort strategy for coupled DOF groups.

    Groups the actuator's DOFs by articulation and solves each group as a
    coupled block through the oracle's inverse mass block. Uses the same
    ``evaluate_force`` law and clamp chain as the scalar strategy.
    """

    def _init_solver(self, controller, clamping) -> None:
        self._resolve_force_law(controller)
        chain, entries = self._pack_clamps(clamping)
        self._kernel = _build_block_solve_kernel(controller.evaluate_force, chain, (controller.evaluate_force, entries))
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

        def find_art(dof):
            for a in range(model.articulation_count):
                if art_base[a] <= dof < art_base[a] + art_ndof[a]:
                    return a
            return -1

        groups: dict[int, list[tuple[int, int]]] = {}  # art -> [(slot, local_dof)]
        for slot, dof in enumerate(dofs):
            a = find_art(int(dof))
            if a < 0:
                raise ValueError(f"Block implicit actuation: DOF {int(dof)} is not in an articulation")
            groups.setdefault(a, []).append((slot, int(dof) - art_base[a]))

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
        self._num_groups = num_groups
        self._groups_built = True

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
        """Solve each coupled group as a block; clamps run inside the solve."""
        if dt is None:
            raise ValueError("Implicit actuation requires dt")
        if not self._groups_built:
            self._build_groups(vel_indices)
        inverse_blocks = self._response.inverse_blocks
        if inverse_blocks is None:
            raise ValueError("Block implicit actuation requires oracle.refresh(state, blocks=True) before step()")

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
            outputs=[computed_forces],
            device=self._device,
        )
        return computed_forces
