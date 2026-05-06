# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
Provides an implementation of a Semi-Implicit Moreau-Jean
mid-point integration scheme for non-smooth dynamical systems.
"""

from __future__ import annotations

from collections.abc import Callable

import warp as wp

from .....core.types import override
from ..core.control import ControlKamino
from ..core.data import DataKamino
from ..core.math import (
    compute_body_pose_update_with_logmap,
    compute_body_twist_update_with_eom,
    screw,
    screw_angular,
    screw_linear,
)
from ..core.model import ModelKamino
from ..core.state import StateKamino
from ..core.types import (
    float32,
    int32,
    mat33f,
    transformf,
    vec3f,
    vec4f,
    vec6f,
)
from ..geometry.contacts import ContactsKamino
from ..geometry.detector import CollisionDetector
from ..kinematics.limits import LimitsKamino
from .integrator import IntegratorBase

###
# Module interface
###


__all__ = ["IntegratorMoreauJean"]


###
# Module configs
###


wp.set_module_options({"enable_backward": False})


###
# Functions
###


@wp.func
def moreau_jean_semi_implicit_with_logmap(
    alpha: float32,
    dt: float32,
    g: vec3f,
    inv_m_i: float32,
    I_i: mat33f,
    inv_I_i: mat33f,
    p_i: transformf,
    u_i: vec6f,
    w_i: vec6f,
) -> tuple[transformf, vec6f]:
    # Integrate the body twist using the maximal coordinate forward dynamics equations
    v_i_n, omega_i_n = compute_body_twist_update_with_eom(
        dt=dt,
        g=g,
        inv_m_i=inv_m_i,
        I_i=I_i,
        inv_I_i=inv_I_i,
        u_i=u_i,
        w_i=w_i,
    )

    # Apply damping to angular velocity
    omega_i_n *= 1.0 - alpha * dt

    # Integrate the body pose using the updated twist
    p_i_n = compute_body_pose_update_with_logmap(
        dt=0.5 * dt,
        p_i=p_i,
        v_i=v_i_n,
        omega_i=omega_i_n,
    )

    # Return the new pose and twist
    return p_i_n, screw(v_i_n, omega_i_n)


###
# Kernels
###


@wp.kernel
def _integrate_moreau_jean_first_inplace(
    # Inputs:
    model_dt: wp.array[float32],
    model_bodies_wid: wp.array[int32],
    bodies_u: wp.array[vec6f],
    # Outputs:
    bodies_q: wp.array[transformf],
):
    # Retrieve the thread index
    tid = wp.tid()

    # Retrieve the world index
    wid = model_bodies_wid[tid]

    # Retrieve the configured time-step of the corresponding world
    dt = model_dt[wid]

    # Retrieve the current state of the body
    q_i = bodies_q[tid]
    u_i = bodies_u[tid]

    # Integrate the body pose using the updated twist
    q_i_m = compute_body_pose_update_with_logmap(
        dt=0.5 * dt,
        p_i=q_i,
        v_i=screw_linear(u_i),
        omega_i=screw_angular(u_i),
    )

    # Store the computed next pose and twist
    bodies_q[tid] = q_i_m


@wp.kernel
def _integrate_moreau_jean_second_inplace(
    # Inputs:
    alpha: float,
    model_dt: wp.array[float32],
    model_gravity: wp.array[vec4f],
    model_bodies_wid: wp.array[int32],
    model_bodies_inv_m: wp.array[float32],
    model_bodies_I: wp.array[mat33f],
    model_bodies_inv_I: wp.array[mat33f],
    state_bodies_w: wp.array[vec6f],
    # Outputs:
    state_bodies_q: wp.array[transformf],
    state_bodies_u: wp.array[vec6f],
):
    # Retrieve the thread index
    tid = wp.tid()

    # Retrieve the world index
    wid = model_bodies_wid[tid]

    # Retrieve the configured time-step and the
    # gravity vector of the corresponding world
    dt = model_dt[wid]
    gv = model_gravity[wid]
    g = gv.w * vec3f(gv.x, gv.y, gv.z)

    # Retrieve the model data
    inv_m_i = model_bodies_inv_m[tid]
    I_i = model_bodies_I[tid]
    inv_I_i = model_bodies_inv_I[tid]

    # Retrieve the current state of the body
    q_i_m = state_bodies_q[tid]
    u_i = state_bodies_u[tid]
    w_i = state_bodies_w[tid]

    # Compute the next pose and twist
    q_i_n, u_i_n = moreau_jean_semi_implicit_with_logmap(
        alpha,
        dt,
        g,
        inv_m_i,
        I_i,
        inv_I_i,
        q_i_m,
        u_i,
        w_i,
    )

    # Store the computed next pose and twist
    state_bodies_q[tid] = q_i_n
    state_bodies_u[tid] = u_i_n


###
# Interfaces
###


class IntegratorMoreauJean(IntegratorBase):
    """
    Provides an implementation of a semi-implicit Moreau-Jean
    time-stepping integrator for non-smooth dynamical systems.

    Effectively, the Moreau-Jean scheme involves the following three steps:

    1. An initial explicit forward integration of the generalized coordinates
       using the generalized velocities at the start of the time-step to render
       an intermediate configuration at the mid-point of the time-step.

    2. An implicit solve of the forward dynamics using the generalized coordinates
       evaluated at the mid-point of the discrete time-step together with the initial
       generalized velocities to render constraint reactions.

    3. A final explicit forward integration of the generalized coordinates and velocities
       using the constraint reactions computed at the mid-point of the time-step to render
       the next state of the system at the end of the time-step.

    These steps can be summarized by the following equations:
    ```
    1: q_m = q_p + 1/2 * dt * G(q_p) * u_p
    2: lambdas = f_fd(q_m, u_p, tau_j)
    3: u_n = u_p + M(q_m)^{-1} * ( dt * h(q_m, u_p) + dt * J_a(q_m)^T * tau_j + J_c(q_m)^T * lambdas )
    4: q_n = q_m + 1/2 * dt * G(q_m) @ u_n
    ```

    where `q_p` and `u_p` are the generalized coordinates and velocities at the start of the
    time-step, `q_m` is the intermediate configuration at the mid-point of the time-step,
    `q_n` and `u_n` are the generalized coordinates and velocities at the end of the time-step.

    `M(q_m)` is the generalized mass matrix, `h(q_m, u_p)` is the vector of generalized
    non-linear forces, `J_a(q_m)` is the actuation Jacobian matrix, `J_c(q_m)` is the
    constraint Jacobian matrix, all evaluated at the mid-point configuration `q_m`.

    `tau_j` is the vector of generalized forces provided at the start of the
    time-step, and `lambdas` are the resulting constraint reactions computed
    at the mid-point from the forward dynamics sub-problem.
    """

    def __init__(self, model: ModelKamino, alpha: float | None = None):
        """
        Initializes the semi-implicit Moreau-Jean integrator with the given :class:`ModelKamino` instance.

        Args:
            model (`ModelKamino`):
                The model container holding the time-invariant parameters of the system being simulated.
            alpha (`float`, optional):
                The angular damping coefficient. Defaults to 0.0 if `None` is provided.
        """
        super().__init__(model)

        self._alpha: float = alpha if alpha is not None else 0.0
        """
        Damping coefficient for angular velocity used to improve numerical stability of the integrator.\n
        Defaults to `0.0`, corresponding to no damping being applied.
        """

    ###
    # Operations
    ###

    @override
    def integrate(
        self,
        forward: Callable,
        model: ModelKamino,
        data: DataKamino,
        state_in: StateKamino,
        state_out: StateKamino,
        control: ControlKamino,
        limits: LimitsKamino | None = None,
        contacts: ContactsKamino | None = None,
        detector: CollisionDetector | None = None,
    ):
        """
        Solves the time integration sub-problem using a semi-implicit Moreau-Jean
        scheme to integrate the current state of the system over a single time-step.

        Args:
            forward (`Callable`):
                An operator that calls the underlying solver for the forward dynamics sub-problem.
            model (`ModelKamino`):
                The model container holding the time-invariant parameters of the system being simulated.
            data (`DataKamino`):
                The data container holding the time-varying parameters of the system being simulated.
            state_in (`StateKamino`):
                The state of the system at the current time-step.
            state_out (`StateKamino`):
                The state of the system at the next time-step.
            control (`ControlKamino`):
                The control inputs applied to the system at the current time-step.
            limits (`LimitsKamino`, optional):
                The joint limits of the system at the current time-step.
                If `None`, no joint limits are considered for the current time-step.
            contacts (`ContactsKamino`, optional):
                The set of active contacts of the system at the current time-step.
                If `None`, no contacts are considered for the current time-step.
            detector (`CollisionDetector`, optional):
                The collision detector to use for generating the set of active contacts at the current time-step.\n
                If `None`, no collision detection is performed for the current time-step,
                and active contacts must be provided via the `contacts` argument.
        """

        # Take the first semi-step until the mid-point of the step
        # NOTE: We only integrate on configuration level
        # q_M = q_i + 1/2 * dt * G(q_i) * u_i
        self._integrate1(model=model, data=data)

        # Solve the forward dynamics sub-problem to compute the
        # constraint reactions at the mid-point of the step
        forward(
            state_in=state_in,
            state_out=state_out,
            control=control,
            limits=limits,
            contacts=contacts,
            detector=detector,
        )

        # Take the second semi-step until the end of the step
        # u_E = u_S + dt * M(q_M)^{-1} * (dt * h(q_M, u_S) + dt * J_a^T(q_M) * tau_j + J_c^T(q_M) * lambda)
        # q_E = q_M + 1/2 * dt * G(q_M) * u_E
        self._integrate2(model=model, data=data)

    ###
    # Operations
    ###

    def _integrate1(self, model: ModelKamino, data: DataKamino):
        """
        Executes the first semi-step of the Moreau-Jean scheme to
        integrate the generalized coordinates of the system from
        the start of the time-step to the mid-point of the time-step
        using the initial generalized velocities of the system.

        Args:
            model (`ModelKamino`):
                The model container holding the time-invariant parameters of the system being simulated.
            data (`DataKamino`):
                The data container holding the time-varying parameters of the system being simulated.
        """
        wp.launch(
            _integrate_moreau_jean_first_inplace,
            dim=model.size.sum_of_num_bodies,
            inputs=[
                # Inputs:
                model.time.dt,
                model.bodies.wid,
                data.bodies.u_i,
                # Outputs:
                data.bodies.q_i,
            ],
            device=model.device,
        )

    def _integrate2(self, model: ModelKamino, data: DataKamino):
        """
        Executes the second semi-step of the Moreau-Jean scheme to
        integrate the generalized coordinates and velocities of the
        system from the mid-point the end of the time-step using
        the constraint reactions computed from the forward dynamics.

        Args:
            model (`ModelKamino`):
                The model container holding the time-invariant parameters of the system being simulated.
            data (`DataKamino`):
                The data container holding the time-varying parameters of the system being simulated.
        """
        wp.launch(
            _integrate_moreau_jean_second_inplace,
            dim=model.size.sum_of_num_bodies,
            inputs=[
                # Inputs:
                self._alpha,
                model.time.dt,
                model.gravity.vector,
                model.bodies.wid,
                model.bodies.inv_m_i,
                data.bodies.I_i,
                data.bodies.inv_I_i,
                data.bodies.w_i,
                # Outputs:
                data.bodies.q_i,
                data.bodies.u_i,
            ],
            device=model.device,
        )
