# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""LuGre friction as a Newton actuator controller.

Model::

    dz/dt = qd - sigma0*|qd|*z/g(qd)
    g(qd) = coulomb + (static_friction - coulomb)*exp(-(qd/stribeck)^2)
    force = -(sigma0*z + sigma1*dz/dt + sigma2*qd)

The deflection z is advanced once per step in :meth:`prepare_implicit`, at the
step-start velocity, and the part of the force that does not vary with velocity
is frozen for the solve. The sigma1 and sigma2 terms stay live, so the solve
sees the same velocity derivative MuJoCo uses (`bias_vel -= sigma1`, plus
sigma2 through the actuator damping path).

This prototype exposes two discretization choices for comparison:

    z_method: "be" (backward Euler) or "zoh" (the exponential step MuJoCo uses).
    force_z:  "new" builds the force from the deflection just advanced,
              "old" from the value before the update, as MuJoCo does.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import warp as wp

from newton.actuators import Controller

_COL_CONST = 0  # velocity-independent friction, rewritten every step
_COL_SIGMA1 = 1
_COL_SIGMA2 = 2
_PARAM_WIDTH = 3


@wp.func
def _stribeck_g(qd: wp.float64, coulomb: wp.float64, static: wp.float64, stribeck: wp.float64) -> wp.float64:
    """Velocity-dependent friction level: `static` near zero, `coulomb` when sliding."""
    ratio = qd / stribeck
    return coulomb + (static - coulomb) * wp.exp(-ratio * ratio)


@wp.func
def _lugre_evaluate_force(
    q: wp.float64,
    qd: wp.float64,
    target_q: wp.float64,
    target_qd: wp.float64,
    feedforward: wp.float64,
    params: wp.array2d[float],
    i: wp.int32,
) -> wp.float64:
    """Force law seen by the implicit solve; affine in qd, so it solves in one step."""
    return (
        feedforward
        - wp.float64(params[i, _COL_CONST])
        - (wp.float64(params[i, _COL_SIGMA1]) + wp.float64(params[i, _COL_SIGMA2])) * qd
    )


def _make_kernels(z_method: str, force_z: str):
    """Build the prepare and explicit kernels for one pair of choices.

    Kernels are specialized at build time via ``wp.static``, so each variant
    gets its own compiled code with no runtime branch.
    """
    use_be = z_method == "be"
    use_new = force_z == "new"

    @wp.func
    def advance(qd: wp.float64, z: wp.float64, h: wp.float64, decay: wp.float64) -> wp.float64:
        if wp.static(use_be):
            return (z + h * qd) / (wp.float64(1.0) + h * decay)
        if decay > wp.float64(1.0e-300):
            e = wp.exp(-decay * h)
            return e * z + (qd / decay) * (wp.float64(1.0) - e)
        return z + h * qd

    @wp.func
    def const_term(
        qd: wp.float64, z_old: wp.float64, h: wp.float64, s0: wp.float64, s1: wp.float64, decay: wp.float64
    ) -> wp.float64:
        """Friction contribution that does not vary with velocity.

        sigma1*zdot = sigma1*(qd - decay*z_used); the qd part is left live for
        the solve, so only -sigma1*decay*z_used belongs here.
        """
        z_new = advance(qd, z_old, h, decay)
        z_used = z_new
        if wp.static(not use_new):
            z_used = z_old
        return z_used * (s0 - s1 * decay)

    @wp.kernel(enable_backward=False)
    def prepare(
        velocities: wp.array[float],
        vel_indices: wp.array[wp.uint32],
        z_current: wp.array[float],
        sigma0: wp.array[float],
        sigma1: wp.array[float],
        coulomb: wp.array[float],
        static_friction: wp.array[float],
        stribeck: wp.array[float],
        dt: float,
        params: wp.array2d[float],
        z_next: wp.array[float],
    ):
        i = wp.tid()
        qd = wp.float64(velocities[vel_indices[i]])
        z = wp.float64(z_current[i])
        h = wp.float64(dt)
        s0 = wp.float64(sigma0[i])
        decay = (
            s0
            * wp.abs(qd)
            / _stribeck_g(qd, wp.float64(coulomb[i]), wp.float64(static_friction[i]), wp.float64(stribeck[i]))
        )
        params[i, _COL_CONST] = float(const_term(qd, z, h, s0, wp.float64(sigma1[i]), decay))
        z_next[i] = float(advance(qd, z, h, decay))

    @wp.kernel
    def explicit(
        velocities: wp.array[float],
        feedforward: wp.array[float],
        vel_indices: wp.array[wp.uint32],
        target_vel_indices: wp.array[wp.uint32],
        z_current: wp.array[float],
        sigma0: wp.array[float],
        sigma1: wp.array[float],
        sigma2: wp.array[float],
        coulomb: wp.array[float],
        static_friction: wp.array[float],
        stribeck: wp.array[float],
        dt: float,
        efforts: wp.array[float],
        z_next: wp.array[float],
    ):
        i = wp.tid()
        qd = wp.float64(velocities[vel_indices[i]])
        z = wp.float64(z_current[i])
        h = wp.float64(dt)
        s0 = wp.float64(sigma0[i])
        s1 = wp.float64(sigma1[i])
        decay = (
            s0
            * wp.abs(qd)
            / _stribeck_g(qd, wp.float64(coulomb[i]), wp.float64(static_friction[i]), wp.float64(stribeck[i]))
        )
        ff = wp.float64(0.0)
        if feedforward:
            ff = wp.float64(feedforward[target_vel_indices[i]])
        efforts[i] = float(ff - const_term(qd, z, h, s0, s1, decay) - (s1 + wp.float64(sigma2[i])) * qd)
        z_next[i] = float(advance(qd, z, h, decay))

    return prepare, explicit


@wp.kernel(enable_backward=False)
def _masked_zero(data: wp.array[float], mask: wp.array[wp.bool]):
    i = wp.tid()
    if mask[i]:
        data[i] = 0.0


class ControllerLuGre(Controller):
    """LuGre friction with per-step bristle-state preparation."""

    @dataclass
    class State(Controller.State):
        """Contact deflection, one entry per actuated degree of freedom."""

        z: wp.array[float] | None = None

        def reset(self, mask: wp.array[wp.bool] | None = None) -> None:
            if self.z is None:
                return
            if mask is None:
                self.z.zero_()
                return
            wp.launch(_masked_zero, dim=len(mask), inputs=[self.z, mask], device=self.z.device)

    @classmethod
    def resolve_arguments(cls, args: dict[str, Any]) -> dict[str, Any]:
        return {
            "sigma0": args.get("sigma0", 1.0e5),
            "sigma1": args.get("sigma1", 0.0),
            "sigma2": args.get("sigma2", 0.0),
            "coulomb_friction": args.get("coulomb_friction", 1.0),
            "static_friction": args.get("static_friction", 1.5),
            "stribeck_velocity": args.get("stribeck_velocity", 1.0e-3),
        }

    def __init__(
        self,
        sigma0: wp.array[float],
        sigma1: wp.array[float],
        sigma2: wp.array[float],
        coulomb_friction: wp.array[float],
        static_friction: wp.array[float],
        stribeck_velocity: wp.array[float],
        z_method: str = "zoh",
        force_z: str = "old",
    ):
        """Initialize.

        Args:
            sigma0: Contact stiffness [N·m/rad].
            sigma1: Contact damping [N·m·s/rad].
            sigma2: Viscous friction [N·m·s/rad].
            coulomb_friction: Sliding friction level [N·m].
            static_friction: Stiction level [N·m].
            stribeck_velocity: Velocity scale of the transition [rad/s].
            z_method: ``"be"`` or ``"zoh"``.
            force_z: ``"new"`` or ``"old"``.
        """
        self.sigma0 = sigma0
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.coulomb_friction = coulomb_friction
        self.static_friction = static_friction
        self.stribeck_velocity = stribeck_velocity
        self._prepare_kernel, self._explicit_kernel = _make_kernels(z_method, force_z)
        self._param_pack: wp.array2d[float] | None = None
        self._next_z: wp.array[float] | None = None

    def finalize(self, device: wp.Device, num_actuators: int) -> None:
        self._next_z = wp.zeros(num_actuators, dtype=wp.float32, device=device)

    def is_stateful(self) -> bool:
        return True

    def is_graphable(self) -> bool:
        return True

    def state(self, num_actuators: int, device: wp.Device) -> ControllerLuGre.State:
        return ControllerLuGre.State(z=wp.zeros(num_actuators, dtype=wp.float32, device=device))

    def update_state(self, current_state, next_state) -> None:
        """Copy the scratch deflection into the next state."""
        wp.copy(next_state.z, self._next_z)

    def compute(
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
        forces,
        state,
        dt,
        device=None,
    ) -> None:
        wp.launch(
            self._explicit_kernel,
            dim=len(forces),
            inputs=[
                velocities,
                feedforward,
                vel_indices,
                target_vel_indices,
                state.z,
                self.sigma0,
                self.sigma1,
                self.sigma2,
                self.coulomb_friction,
                self.static_friction,
                self.stribeck_velocity,
                float(dt),
            ],
            outputs=[forces, self._next_z],
            device=device or self.sigma0.device,
        )

    evaluate_force = _lugre_evaluate_force

    def bind_params(self) -> wp.array2d[float]:
        """Build the pack once; `prepare_implicit` writes into this same array."""
        if self._param_pack is not None:
            return self._param_pack
        pack = wp.zeros((len(self.sigma0), _PARAM_WIDTH), dtype=float, device=self.sigma0.device)
        pack[:, _COL_SIGMA1].assign(self.sigma1)
        pack[:, _COL_SIGMA2].assign(self.sigma2)
        self.sigma1 = pack[:, _COL_SIGMA1]
        self.sigma2 = pack[:, _COL_SIGMA2]
        self._param_pack = pack
        return pack

    def prepare_implicit(
        self,
        positions,
        velocities,
        target_pos,
        target_vel,
        pos_indices,
        vel_indices,
        target_pos_indices,
        target_vel_indices,
        ctrl_state,
        dt,
        inv_mass=None,
        device=None,
    ) -> None:
        if ctrl_state is None:
            raise RuntimeError("Implicit ControllerLuGre requires controller state (contact deflection z)")
        wp.launch(
            self._prepare_kernel,
            dim=len(self._next_z),
            inputs=[
                velocities,
                vel_indices,
                ctrl_state.z,
                self.sigma0,
                self.sigma1,
                self.coulomb_friction,
                self.static_friction,
                self.stribeck_velocity,
                float(dt),
            ],
            outputs=[self._param_pack, self._next_z],
            device=device or self.sigma0.device,
        )
