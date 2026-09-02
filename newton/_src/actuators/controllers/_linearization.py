# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared linearization for the neural controllers under implicit actuation.

A network is not an in-kernel force law, so each step it is linearized about the
current state and enters the implicit solve as ``tau0 + a (q - q0) + b (qd - qd0)``.
Both :class:`ControllerNeuralMLP` and :class:`ControllerNeuralLSTM` use this;
only the forward and gradient evaluation differs between them.
"""

from __future__ import annotations

import warp as wp

IMPLICIT_JACOBIAN_MARGIN: float = 0.1
"""Lower bound kept on the solve's Jacobian ``J = 1 - dt*alpha*(a*dt + b)``."""


@wp.func
def _linear_force(
    q: wp.float64,
    qd: wp.float64,
    target_q: wp.float64,
    target_qd: wp.float64,
    feedforward: wp.float64,
    params: wp.array2d[float],
    i: wp.int32,
) -> wp.float64:
    """Force law from the network's linearization about the current state.

    ``tau(q, qd) = tau0 + a (q - q0) + b (qd - qd0)`` with ``a = d(tau)/dq`` and
    ``b = d(tau)/dqd`` taken at the expansion point ``(q0, qd0)`` (see
    :meth:`ControllerNeuralMLP.prepare_implicit`). Keeping the expansion point
    rather than a collapsed offset avoids cancelling two large float32 terms to
    recover ``tau0`` near that point, where the solve is most sensitive.
    """
    tau0 = wp.float64(params[i, 0])
    a = wp.float64(params[i, 1])
    b = wp.float64(params[i, 2])
    q0 = wp.float64(params[i, 3])
    qd0 = wp.float64(params[i, 4])
    return tau0 + a * (q - q0) + b * (qd - qd0)


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
    q0: wp.array[float],
    qd0: wp.array[float],
    tq0: wp.array[float],
    tqd0: wp.array[float],
):
    """Gather per-slot current state and targets for the linearization point."""
    i = wp.tid()
    q0[i] = positions[pos_indices[i]]
    qd0[i] = velocities[vel_indices[i]]
    tq0[i] = target_pos[target_pos_indices[i]]
    tqd0[i] = target_vel[target_vel_indices[i]]


@wp.kernel(enable_backward=False)
def _assemble_linear_params_kernel(
    tau0: wp.array[float],
    dtau_dq: wp.array[float],
    dtau_dqd: wp.array[float],
    q0: wp.array[float],
    qd0: wp.array[float],
    inv_mass: wp.array[float],
    dt: float,
    margin: float,
    params: wp.array2d[float],
):
    """Pack the linearization into ``[tau0, a, b, q0, qd0]`` for the implicit force law.

    The solve's Jacobian is ``J = 1 - dt*alpha*(a*dt + b)``. Once ``a*dt + b``
    reaches ``1/(dt*alpha)`` the problem is singular and the solution flips sign.
    Both slopes are scaled down together, preserving their ratio, until
    ``J >= margin``. Laws with ``a, b <= 0`` are never touched.
    """
    i = wp.tid()
    a = dtau_dq[i]
    b = dtau_dqd[i]
    alpha = inv_mass[i]
    slope = a * dt + b
    limit = (1.0 - margin) / (dt * wp.max(alpha, 1.0e-12))
    if slope > limit and slope > 0.0:
        scale = limit / slope
        a *= scale
        b *= scale
    params[i, 0] = tau0[i]
    params[i, 1] = a
    params[i, 2] = b
    params[i, 3] = q0[i]
    params[i, 4] = qd0[i]
