# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
Provides an implementation of a Semi-Implicit Euler time-integrator.
"""

from __future__ import annotations

from collections.abc import Callable

import warp as wp

from .....core.types import override
from ..core.control import ControlKamino
from ..core.data import DataKamino
from ..core.math import compute_body_pose_update_with_logmap, compute_body_twist_update_with_eom
from ..core.model import ModelKamino
from ..core.state import StateKamino
from ..geometry.contacts import ContactsKamino
from ..geometry.detector import CollisionDetector
from ..kinematics.limits import LimitsKamino
from .integrator import IntegratorBase

###
# Module interface
###


__all__ = ["IntegratorEuler"]


###
# Module configs
###


wp.set_module_options({"enable_backward": False, "default_grid_stride": False})


###
# Functions
###


@wp.func
def euler_semi_implicit_with_logmap(
    alpha: wp.float32,
    dt: wp.float32,
    g: wp.vec3f,
    inv_m_i: wp.float32,
    I_i: wp.mat33f,
    inv_I_i: wp.mat33f,
    p_i: wp.transformf,
    u_i: wp.spatial_vectorf,
    w_i: wp.spatial_vectorf,
) -> tuple[wp.transformf, wp.spatial_vectorf]:
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
        dt=dt,
        p_i=p_i,
        v_i=v_i_n,
        omega_i=omega_i_n,
    )

    # Return the new pose and twist
    return p_i_n, wp.spatial_vectorf(*v_i_n, *omega_i_n)


###
# Kernels
###


@wp.kernel
def _integrate_semi_implicit_euler_inplace(
    # Inputs:
    alpha: float,
    model_dt: wp.array[wp.float32],
    model_gravity: wp.array[wp.vec4f],
    model_bodies_wid: wp.array[wp.int32],
    model_bodies_inv_m: wp.array[wp.float32],
    model_bodies_I: wp.array[wp.mat33f],
    model_bodies_inv_I: wp.array[wp.mat33f],
    state_bodies_w: wp.array[wp.spatial_vectorf],
    # Outputs:
    state_bodies_q: wp.array[wp.transformf],
    state_bodies_u: wp.array[wp.spatial_vectorf],
):
    # Retrieve the thread index
    tid = wp.tid()

    # Retrieve the world index
    wid = model_bodies_wid[tid]

    # Retrieve the time step and gravity vector
    dt = model_dt[wid]
    gv = model_gravity[wid]
    g = gv.w * wp.vec3f(gv.x, gv.y, gv.z)

    # Retrieve the model data
    inv_m_i = model_bodies_inv_m[tid]
    I_i = model_bodies_I[tid]
    inv_I_i = model_bodies_inv_I[tid]

    # Retrieve the current state of the body
    q_i = state_bodies_q[tid]
    u_i = state_bodies_u[tid]
    w_i = state_bodies_w[tid]

    # Compute the next pose and twist
    q_i_n, u_i_n = euler_semi_implicit_with_logmap(
        alpha,
        dt,
        g,
        inv_m_i,
        I_i,
        inv_I_i,
        q_i,
        u_i,
        w_i,
    )

    # Store the computed next pose and twist
    state_bodies_q[tid] = q_i_n
    state_bodies_u[tid] = u_i_n


###
# Launchers
###


def integrate_euler_semi_implicit(model: ModelKamino, data: DataKamino, alpha: float = 0.0):
    wp.launch(
        _integrate_semi_implicit_euler_inplace,
        dim=model.size.sum_of_num_bodies,
        inputs=[
            # Inputs:
            alpha,  # alpha: angular damping
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


###
# Interfaces
###


class IntegratorEuler(IntegratorBase):
    """
    Provides an implementation of a Semi-Implicit Euler time-stepping integrator.

    Effectively, the Semi-Implicit Euler scheme involves an implicit solve of the
    forward dynamics to render constraint reactions at the start of the time-step,
    followed by an explicit forward integration step to compute the next state:

    ```
    lambda = f_fd(q_p, u_p, tau_j)
    u_n = u_p + M^{-1} * ( dt * h(q_p, u_p) + dt * J_a(q_p)^T * tau_j + J_c(q_p)^T * lambda )
    q_n = q_p + dt * G(q_p) @ u_n
    ```

    where `q_p` and `u_p` are the generalized coordinates and velocities at the start of the
    time-step, `q_n` and `u_n` are the generalized coordinates and velocities at the end of
    the time-step, `M` is the generalized mass matrix, `h` is the vector of generalized
    non-linear forces, `J_a` is the actuation Jacobian matrix, `tau_j` is the vector of
    generalized forces, `J_c` is the constraint Jacobian matrix, and `lambda` are the
    constraint reactions.
    """

    def __init__(self, model: ModelKamino, alpha: float | None = None):
        """
        Initializes the Semi-Implicit Euler integrator with the given :class:`ModelKamino` instance.

        Args:
            model: The model container holding the time-invariant parameters of the system being simulated.
            alpha: The angular damping coefficient. Defaults to 0.0 if `None` is provided.
        """
        super().__init__(model)

        self._alpha: float = alpha if alpha is not None else 0.0
        """
        Damping coefficient for angular velocity used to improve numerical stability of the integrator.
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
        Solves the time integration sub-problem using a Semi-Implicit Euler scheme
        to integrate the current state of the system over a single time-step.

        Args:
            forward: An operator that calls the underlying solver for the forward dynamics sub-problem.
            model: The model container holding the time-invariant parameters of the system being simulated.
            data: The data container holding the time-varying parameters of the system being simulated.
            state_in: The state of the system at the current time-step.
            state_out: The state of the system at the next time-step.
            control: The control inputs applied to the system at the current time-step.
            limits: The joint limits of the system at the current time-step.
                If `None`, no joint limits are considered for the current time-step.
            contacts: The set of active contacts of the system at the current time-step.
                If `None`, no contacts are considered for the current time-step.
            detector: The collision detector to use for generating the set of active contacts at the current time-step.
                If `None`, no collision detection is performed for the current time-step,
                and active contacts must be provided via the `contacts` argument.
        """
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

        # Perform forward integration to compute the next state of the system
        integrate_euler_semi_implicit(model=model, data=data, alpha=self._alpha)
