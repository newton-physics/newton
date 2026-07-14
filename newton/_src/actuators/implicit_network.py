# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Implicit effort strategy for network force laws.

Controllers whose law cannot run inside a kernel expose a force-and-gradients
hook (:meth:`Controller.implicit_force_grad`). This :class:`_EffortImplicit`
subclass solves the same residual as the base class, but with a fixed-count
Newton loop at launch level:

    p = 0
    repeat newton_iters times:
        predict end-of-step state from p                 (kernel)
        tau, d(tau)/dq, d(tau)/d(qd) at that state       (hook: forward + autodiff)
        p -= r / (dr/dp)                                 (kernel)
    effort = p / h

The fixed iteration count (no convergence check) keeps the loop CUDA-graph
capturable. Clamps are applied to the solved effort afterwards, not inside
the solve.
"""

from __future__ import annotations

from typing import Any

import warp as wp

from .implicit import _EffortImplicit


@wp.kernel(enable_backward=False)
def _gather_slot_state_kernel(
    positions: wp.array[float],
    velocities: wp.array[float],
    target_pos: wp.array[float],
    target_vel: wp.array[float],
    pos_indices: wp.array[wp.uint32],
    vel_indices: wp.array[wp.uint32],
    target_pos_indices: wp.array[wp.uint32],
    target_vel_indices: wp.array[wp.uint32],
    q: wp.array[float],
    qd: wp.array[float],
    tq: wp.array[float],
    tqd: wp.array[float],
):
    i = wp.tid()
    q[i] = positions[pos_indices[i]]
    qd[i] = velocities[vel_indices[i]]
    tq[i] = target_pos[target_pos_indices[i]]
    tqd[i] = target_vel[target_vel_indices[i]]


@wp.kernel(enable_backward=False)
def _predict_state_kernel(
    q0: wp.array[float],
    qd0: wp.array[float],
    inv_mass: wp.array[float],
    vel_indices: wp.array[wp.uint32],
    impulse: wp.array[float],
    h: float,
    q_pred: wp.array[float],
    qd_pred: wp.array[float],
):
    """End-of-step state implied by ``impulse``: qd(p) = qd + alpha*p, q(p) = q + h*qd(p)."""
    i = wp.tid()
    qd = qd0[i] + inv_mass[vel_indices[i]] * impulse[i]
    qd_pred[i] = qd
    q_pred[i] = q0[i] + h * qd


@wp.kernel(enable_backward=False)
def _newton_update_kernel(
    tau: wp.array[float],
    dtau_dq: wp.array[float],
    dtau_dqd: wp.array[float],
    inv_mass: wp.array[float],
    vel_indices: wp.array[wp.uint32],
    h: float,
    impulse: wp.array[float],
):
    """One Newton step on ``r(p) = p - h*tau(state(p))``.

    ``d(tau)/dp = (d(tau)/dq * h + d(tau)/d(qd)) * alpha`` via the predicted
    state, so ``dr/dp = 1 - h * d(tau)/dp``.
    """
    i = wp.tid()
    a = inv_mass[vel_indices[i]]
    r = impulse[i] - h * tau[i]
    drdp = 1.0 - h * a * (h * dtau_dq[i] + dtau_dqd[i])
    impulse[i] = impulse[i] - r / drdp


@wp.kernel(enable_backward=False)
def _impulse_to_effort_kernel(impulse: wp.array[float], h: float, forces: wp.array[float]):
    i = wp.tid()
    forces[i] = impulse[i] / h


class _EffortImplicitNetwork(_EffortImplicit):
    """Implicit (Stable-PD) effort strategy for network force laws.

    Overrides the in-kernel solve of :class:`_EffortImplicit` with a
    launch-level Newton loop: the strategy owns the loop state (impulse and
    per-slot scratch buffers); the controller's hook owns the force law and
    its gradients.
    """

    def _init_solver(self, controller, clamping) -> None:
        self._force_grad = controller.implicit_force_grad()
        if self._force_grad is None:
            raise NotImplementedError(
                f"{type(controller).__name__} does not support implicit actuation "
                "in this configuration (Controller.implicit_force_grad() returned None)"
            )
        self._clamps = clamping or []

        n = self._num_actuators
        device = self._device
        self._impulse = wp.zeros(n, dtype=wp.float32, device=device)
        self._q0 = wp.zeros(n, dtype=wp.float32, device=device)
        self._qd0 = wp.zeros(n, dtype=wp.float32, device=device)
        self._tq = wp.zeros(n, dtype=wp.float32, device=device)
        self._tqd = wp.zeros(n, dtype=wp.float32, device=device)
        self._q_pred = wp.zeros(n, dtype=wp.float32, device=device)
        self._qd_pred = wp.zeros(n, dtype=wp.float32, device=device)
        self._tau = wp.zeros(n, dtype=wp.float32, device=device)
        self._dtau_dq = wp.zeros(n, dtype=wp.float32, device=device)
        self._dtau_dqd = wp.zeros(n, dtype=wp.float32, device=device)

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
        """Solve the implicit effort into *computed_forces* and apply clamps.

        Newton on ``r(p) = p - h*tau(state(p))``, warm-started at ``p = 0``,
        evaluating the controller's hook each iteration.
        """
        if dt is None:
            raise ValueError("Implicit actuation requires dt")
        h = float(dt)
        n = self._num_actuators
        device = self._device

        wp.launch(
            _gather_slot_state_kernel,
            dim=n,
            inputs=[
                positions,
                velocities,
                target_pos,
                target_vel,
                pos_indices,
                vel_indices,
                target_pos_indices,
                target_vel_indices,
            ],
            outputs=[self._q0, self._qd0, self._tq, self._tqd],
            device=device,
        )
        self._impulse.zero_()
        for _ in range(max(1, int(self._options.newton_iters))):
            wp.launch(
                _predict_state_kernel,
                dim=n,
                inputs=[self._q0, self._qd0, self._inv_mass, vel_indices, self._impulse, h],
                outputs=[self._q_pred, self._qd_pred],
                device=device,
            )
            self._force_grad(self._q_pred, self._qd_pred, self._tq, self._tqd, self._tau, self._dtau_dq, self._dtau_dqd)
            wp.launch(
                _newton_update_kernel,
                dim=n,
                inputs=[self._tau, self._dtau_dq, self._dtau_dqd, self._inv_mass, vel_indices, h],
                outputs=[self._impulse],
                device=device,
            )
        wp.launch(_impulse_to_effort_kernel, dim=n, inputs=[self._impulse, h], outputs=[computed_forces], device=device)

        forces = computed_forces
        for clamp in self._clamps:
            clamp.modify_forces(
                forces, applied_forces, positions, velocities, pos_indices, vel_indices, device=self._device
            )
            forces = applied_forces
        return forces
