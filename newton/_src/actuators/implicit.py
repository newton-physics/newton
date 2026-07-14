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

Controllers whose law cannot run in-kernel (neural networks) are solved by
the :class:`_EffortImplicitNetwork` subclass instead; see
:mod:`implicit_network`.

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
        newton_iters: Newton iterations for the network strategy (neural
            controllers): each iteration evaluates the network at the
            previous step's predicted end-of-step state and takes one Newton
            step. Fixed count, so CUDA-graph capture stays possible.
        block_solve: Solve coupled DOF groups (DOFs sharing an articulation)
            as a block instead of independent scalars, recovering the
            inertial cross terms. Requires ``oracle.refresh(state,
            blocks=True)`` each step. Supported for PD controllers; ignored
            for network controllers.
    """

    max_iters: int = 4
    residual_tol: float = 1.0e-5
    update_tol: float = 1.0e-5
    fd_epsilon: float = 1.0e-4
    derivative_floor: float = 1.0e-8
    warm_start: str = "explicit"
    newton_iters: int = 1
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
# The strategy object installed by Actuator.set_strategy_implicit
# ---------------------------------------------------------------------------


class _EffortImplicit:
    """Implicit (Stable-PD) effort strategy; base class and in-kernel solver.

    Owns everything the fused solve needs: the controller's force-law
    ``@wp.func`` and parameter pack, the compiled solve kernel, the resolved
    effective-inverse-mass buffer, and the composed in-solve clamp chain.

    Subclasses for force laws that cannot run in-kernel override
    :meth:`_init_solver` and :meth:`compute_force` (see
    :class:`_EffortImplicitNetwork`).
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
        """Build the in-kernel scalar solve from the controller and clamps."""
        self._resolve_force_law(controller)
        chain, entries = self._pack_clamps(clamping)
        self._kernel = _build_solve_kernel(controller.evaluate_force, chain, (controller.evaluate_force, entries))

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
