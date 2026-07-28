# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Implicit (Stable-PD style) effort strategy for actuators.

One solve runs per actuated 1-DOF joint (``REVOLUTE`` / ``PRISMATIC``). The
unknown is the impulse ``p = h * effort``. The solve finds the root of the
residual

    ``r(p) = p - h * g(q(p), qd(p))``

where ``g`` is the force law and the predicted end-of-step state is

    ``qd(p) = qd + alpha * p`` — velocity after applying the impulse,
    ``q(p) = q + h * qd(p)`` — position after moving with that velocity.

``alpha`` is the joint's effective inverse mass: how much its velocity
changes per unit of impulse. It is the only quantity the solve needs from
the simulator and enters as a :class:`ResponseOracle` through the
``effective_inv_mass`` argument of :meth:`Actuator.set_strategy_implicit`.
The oracle owns one global per-DOF buffer shared by any number of actuators;
keep it current either by calling ``oracle.refresh(state)`` once per step
(before the actuators) or by writing values into ``oracle.alpha`` directly.

:class:`_EffortImplicit` solves in-kernel force laws: the controller provides
its law as a ``@wp.func`` (:attr:`Controller.evaluate_force`) with an opaque
parameter pack (:meth:`Controller.force_params`), and each thread of one
generated kernel Newton-iterates on its slot's residual with a
finite-difference slope. Clamps provide their own ``@wp.func``
(:attr:`Clamping.evaluate_clamp`) and are composed into the residual, so
limits are enforced against the predicted state — a DC-motor envelope, for
example, sees the end-of-step velocity.

With ``options.block_solve`` the same class instead groups the actuator's DOFs
by articulation and solves each group as a coupled block, predicting the
group's end-of-step state through the oracle's inverse-mass block so the
inertial cross terms are kept. Reduces to the scalar solve for isolated DOFs;
requires ``oracle.refresh(state, blocks=True)`` each step.

Neural-network controllers also solve in-kernel: they cannot run their law
inside the kernel, so each step they linearize the network about the current
state (:meth:`Controller.prepare_implicit`) and expose the linear force law
``tau = c + a*q + b*qd`` through the same :attr:`~Controller.evaluate_force`
/ :meth:`~Controller.force_params` interface as PD.

See the design report: https://reports.mmacklin.com/implicit-actuation-newton/
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import warp as wp

from .response_oracle import ResponseOracle

__all__ = ["ActuatorImplicitOptions", "ResponseOracle"]


# ---------------------------------------------------------------------------
# Solver options
# ---------------------------------------------------------------------------


@dataclass
class ActuatorImplicitOptions:
    """Configuration for implicit actuation; see :meth:`Actuator.set_strategy_implicit`.

    Args:
        max_iters: Maximum Newton iterations per DOF (in-kernel strategy).
        residual_tol: Stop when the residual magnitude falls below this.
        update_tol: Stop when the impulse update magnitude falls below this.
        fd_epsilon: Relative forward finite-difference step in impulse space.
        derivative_floor: Stop when ``|dr/dp|`` falls below this (flat residual).
        warm_start: Initial impulse guess: ``"explicit"`` starts from the
            explicit force impulse, ``"zero"`` starts from zero.
        block_solve: Solve coupled DOF groups (DOFs sharing an articulation)
            as a block instead of independent scalars, recovering the
            inertial cross terms. Requires ``oracle.refresh(state,
            blocks=True)`` each step.
    """

    max_iters: int = 4
    residual_tol: float = 1.0e-5
    update_tol: float = 1.0e-5
    fd_epsilon: float = 1.0e-4
    derivative_floor: float = 1.0e-8
    warm_start: str = "explicit"
    block_solve: bool = False


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


# ---------------------------------------------------------------------------
# Solve kernel
# ---------------------------------------------------------------------------

_solve_kernel_cache: dict[tuple[Any, Any], wp.Kernel] = {}


def _build_solve_kernel(evaluate_force: wp.Function, clamp_chain: wp.Function, cache_key: tuple):
    """Build (or reuse) the solve kernel for one (force law, clamp chain) pair.

    One thread solves one actuator slot with Newton's method:

    1. Predict the end-of-step state for the current impulse guess ``p``.
    2. Evaluate the clamped force law there and form the residual
       ``r(p) = p - h * g(q(p), qd(p))``.
    3. Estimate the slope ``dr/dp`` with a forward finite difference.
    4. Take the Newton step ``p -= r / (dr/dp)``; repeat until converged.

    Why float64: at stiff gains the residual is a small difference of large
    numbers; in float32 the finite-difference slope would drown in rounding
    error.
    """
    cached = _solve_kernel_cache.get(cache_key)
    if cached is not None:
        return cached

    @wp.kernel
    def solve(
        pos_indices: wp.array[wp.uint32],
        vel_indices: wp.array[wp.uint32],
        target_pos_indices: wp.array[wp.uint32],
        target_vel_indices: wp.array[wp.uint32],
        inv_mass: wp.array[float],
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
        efforts: wp.array[float],
    ):
        i = wp.tid()

        # Gather this slot's inputs and promote to float64.
        a = wp.float64(inv_mass[vel_indices[i]])
        q0 = wp.float64(positions[pos_indices[i]])
        qd_free = wp.float64(velocities[vel_indices[i]])
        tq = wp.float64(target_pos[target_pos_indices[i]])
        tqd = wp.float64(target_vel[target_vel_indices[i]])
        ff = wp.float64(0.0)
        if feedforward:
            ff = wp.float64(feedforward[target_vel_indices[i]])
        hd = wp.float64(h)

        res_tol = wp.float64(residual_tol)
        upd_tol = wp.float64(update_tol)
        fd_eps = wp.float64(fd_epsilon)
        deriv_floor = wp.float64(derivative_floor)

        # Warm start: the clamped explicit impulse, or zero.
        if warm_zero != 0:
            p = wp.float64(0.0)
        else:
            tau0 = evaluate_force(q0, qd_free, tq, tqd, ff, params, i)
            p = hd * clamp_chain(tau0, q0, qd_free, clamp_params, i)

        for _ in range(max_iters):
            # Residual at the current impulse guess.
            qd_n = qd_free + a * p
            q_n = q0 + hd * qd_n
            f_n = clamp_chain(evaluate_force(q_n, qd_n, tq, tqd, ff, params, i), q_n, qd_n, clamp_params, i)
            r = p - hd * f_n
            if wp.abs(r) < res_tol:
                break

            # Residual at a perturbed impulse; the step scales with |p|.
            eps = fd_eps * (wp.float64(1.0) + wp.abs(p))
            pe = p + eps
            qd_ne = qd_free + a * pe
            q_ne = q0 + hd * qd_ne
            f_ne = clamp_chain(evaluate_force(q_ne, qd_ne, tq, tqd, ff, params, i), q_ne, qd_ne, clamp_params, i)
            re = pe - hd * f_ne

            # Newton step on the finite-difference slope.
            drdp = (re - r) / eps
            if wp.abs(drdp) < deriv_floor:
                break
            dp = -r / drdp
            p = p + dp
            if wp.abs(dp) < upd_tol:
                break

        # Re-clamp at the final predicted state: a no-op at convergence, and a
        # hard-limit guarantee when max_iters cut the iteration short.
        qd_f = qd_free + a * p
        q_f = q0 + hd * qd_f
        tau_out = clamp_chain(p / hd, q_f, qd_f, clamp_params, i)

        efforts[i] = wp.float32(tau_out)

    _solve_kernel_cache[cache_key] = solve
    return solve


# ---------------------------------------------------------------------------
# Block solve kernel (coupled DOF groups)
# ---------------------------------------------------------------------------

_block_kernel_cache: dict[tuple[Any, Any], wp.Kernel] = {}


def _build_block_solve_kernel(evaluate_force: wp.Function, clamp_chain: wp.Function, cache_key: tuple):
    """Build (or reuse) the block solve kernel for one (force law, clamp chain) pair.

    One thread solves one coupled group with Newton's method: predict the
    group's end-of-step state through the block response ``A_g`` (a submatrix of
    ``oracle.inverse_blocks``, keeping the off-diagonal inertial cross terms),
    evaluate the clamped force law per DOF, form the block residual and its
    finite-difference Jacobian, and take a dense Gauss-elimination Newton step.
    Reduces to the scalar solve for a group size of one. Float64 for the same
    reason as the scalar solve — the FD slope is a small difference of large
    numbers.
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


# ---------------------------------------------------------------------------
# The strategy object installed by Actuator.set_strategy_implicit
# ---------------------------------------------------------------------------


class _EffortImplicit:
    """Implicit (Stable-PD) effort strategy and in-kernel solver.

    Owns everything the fused solve needs: the controller's force-law
    ``@wp.func`` and parameter pack, the compiled solve kernel, the resolved
    effective-inverse-mass buffer, and the composed in-solve clamp chain.

    ``options.block_solve`` selects between two solves that share the force law
    and clamp chain: the default per-DOF scalar solve, and a coupled block
    solve that groups the actuator's DOFs by articulation and predicts each
    group's state through the oracle's inverse-mass block, keeping the
    off-diagonal inertial cross terms (see :meth:`_compute_force_block`).

    Before each solve, :meth:`compute_force` calls the controller's
    :meth:`~Controller.prepare_implicit` hook so state-dependent laws (a
    network linearized about the current state) can refresh their parameter
    pack; parameter-static laws like PD leave it a no-op.
    """

    def __init__(
        self,
        controller,
        clamping,
        effective_inv_mass: ResponseOracle | None,
        options: ActuatorImplicitOptions | None,
        num_actuators: int,
        device: wp.Device,
    ):
        self._options = options or ActuatorImplicitOptions()
        self._num_actuators = num_actuators
        self._device = device
        if not isinstance(effective_inv_mass, ResponseOracle):
            raise ValueError(
                "Implicit actuation requires effective_inv_mass to be a ResponseOracle; "
                "build one with newton.actuators.ResponseOracle(model). For constant "
                "values, write them into oracle.alpha instead of calling refresh()."
            )
        self._inv_mass = effective_inv_mass.alpha
        self._response = effective_inv_mass  # oracle; block solves also read .inverse_blocks
        self._controller = controller
        self._block = bool(self._options.block_solve)
        self._init_solver(controller, clamping)

    def _resolve_force_law(self, controller):
        """Validate the controller's in-kernel force law and bind its params."""
        params = controller.force_params()
        if controller.evaluate_force is None or params is None:
            raise NotImplementedError(
                f"{type(controller).__name__} does not support implicit actuation "
                "(Controller.evaluate_force / force_params() unavailable)"
            )
        self._params = params
        # The controller re-binds its parameter attributes as views into the
        # pack, keeping user writes live.
        controller.bind_params(self._params)

    def _pack_clamps(self, clamping):
        """Pack every clamp's params side by side, bind views, compose one @wp.func.

        Sets :attr:`_clamp_params` and returns ``(chain, entries)`` for the
        solve-kernel cache key. Each clamp re-binds its parameter attributes as
        views into its slice, so user writes (e.g. ``clamp.max_effort``) stay
        visible to the solve kernel.
        """
        entries: list[tuple[wp.Function, int]] = []
        blocks = []
        col = 0
        for clamp in clamping or []:
            func = clamp.evaluate_clamp
            if func is None:
                raise NotImplementedError(
                    f"{type(clamp).__name__} does not support implicit actuation "
                    "(Clamping.evaluate_clamp / clamp_params() unavailable)"
                )
            block = clamp.clamp_params().numpy().reshape(self._num_actuators, -1).astype(np.float32)
            entries.append((func, col))
            col += block.shape[1]
            blocks.append(block)
        if blocks:
            self._clamp_params = wp.array(np.hstack(blocks), dtype=float, device=self._device)
            for clamp, (_func, base), block in zip(clamping, entries, blocks, strict=True):
                clamp.bind_params(self._clamp_params[:, base : base + block.shape[1]])
        else:
            self._clamp_params = wp.zeros((self._num_actuators, 1), dtype=float, device=self._device)
        return _compose_clamps(tuple(entries)), tuple(entries)

    def _init_solver(self, controller, clamping) -> None:
        """Build the in-kernel solve from the controller and clamps.

        ``options.block_solve`` selects the coupled block kernel; otherwise the
        per-DOF scalar kernel. Both share the force law and clamp chain.
        """
        self._resolve_force_law(controller)
        chain, entries = self._pack_clamps(clamping)
        key = (controller.evaluate_force, entries)
        if self._block:
            self._kernel = _build_block_solve_kernel(controller.evaluate_force, chain, key)
            self._groups_built = False
        else:
            self._kernel = _build_solve_kernel(controller.evaluate_force, chain, key)

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
        """Solve the implicit effort into *computed_forces* and return it.

        Clamps are enforced inside the solve against the predicted
        end-of-step state.
        """
        if dt is None:
            raise ValueError("Implicit actuation requires dt")
        # Parameter-static laws (PD) no-op here; a network relinearizes about
        # the current state and rewrites its parameter pack in place.
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
            self._device,
        )
        if self._block:
            return self._compute_force_block(
                positions,
                velocities,
                target_pos,
                target_vel,
                feedforward,
                pos_indices,
                vel_indices,
                target_pos_indices,
                target_vel_indices,
                computed_forces,
                float(dt),
            )
        opts = self._options
        wp.launch(
            self._kernel,
            dim=self._num_actuators,
            inputs=[
                pos_indices,
                vel_indices,
                target_pos_indices,
                target_vel_indices,
                self._inv_mass,
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
            ],
            outputs=[computed_forces],
            device=self._device,
        )
        return computed_forces

    def _compute_force_block(
        self,
        positions,
        velocities,
        target_pos,
        target_vel,
        feedforward,
        pos_indices,
        vel_indices,
        target_pos_indices,
        target_vel_indices,
        computed_forces,
        dt: float,
    ) -> wp.array[float]:
        """Coupled-block solve path (``options.block_solve``)."""
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
                dt,
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
