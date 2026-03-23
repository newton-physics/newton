# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Defines the state container of Kamino."""

from __future__ import annotations

from dataclasses import dataclass

import warp as wp

from .....sim import Model, State
from .bodies import convert_body_com_to_origin, convert_body_origin_to_com
from .size import SizeKamino
from .types import vec6f

###
# Module interface
###

__all__ = [
    "StateKamino",
]


###
# Types
###


@dataclass
class StateKamino:
    """
    Represents the time-varying state of a :class:`ModelKamino` in a simulation.

    The :class:`StateKamino` object holds all dynamic quantities that change over time during
    simulation, such as rigid body poses, twists, and wrenches, as well as joint coordinates,
    velocities, and constraint forces.

    :class:`StateKamino` objects are typically created via :meth:`kamino.ModelKamino.state()`
    and are used to store and update the simulation's current configuration and derived data.

    For constrained rigid multi-body system, the state is defined formally using either:
    1. maximal-coordinates, as the absolute poses and twists of all bodies expressed in world coordinates, or
    2. minimal-coordinates, as the joint coordinates and velocities along with the
       pose and twist of a base body when it is a so-called "floating-base" system.

    In Kamino, we formally adopt the maximal-coordinate formulation in order to compute the physics of the
    system, but we are also interested in the state of the joints for the purposes of control and analysis.

    Thus, this container incorporates the data of both representations, and in addition also includes the per-body
    total (i.e. net) wrenches expressed in world coordinates, as well as the joint constraint forces. Thus, it
    provides a complete description of the dynamic state of the constrained rigid multi-body system.

    We adopt the following notational conventions for the state attributes:
    - Generalized coordinates, whether maximal or minimal, are universally denoted by ``q``
    - Generalized velocities for bodies are denoted by ``u`` since they are twists
    - Generalized velocities for joints are denoted by ``dq`` since they are time-derivatives of ``q``
    - Wrenches (forces + torques) are denoted by ``w``
    - Constraint forces are denoted by ``lambda`` since they are effectively Lagrange multipliers
    - Subscripts ``_i`` denote body-indexed quantities, e.g. :attr:`q_i`, :attr:`u_i`, :attr:`w_i`.
    - Subscripts ``_j`` denote joint-indexed quantities, e.g. :attr:`q_j`, :attr:`dq_j`, :attr:`lambda_j`.
    """

    ###
    # Attributes
    ###

    q_i: wp.array | None = None
    """
    Array of absolute body CoM poses expressed in world coordinates.\n
    Each element is a 7D transform consisting of a 3D position + 4D unit quaternion.\n
    Shape is ``(num_bodies,)`` and dtype is :class:`transformf`.
    """

    u_i: wp.array | None = None
    """
    Array of absolute body CoM twists expressed in world coordinates.\n
    Each element is a 6D vector comprising a 3D linear + 3D angular components.\n
    Shape is ``(num_bodies,)`` and dtype is :class:`vec6f`.
    """

    w_i: wp.array | None = None
    """
    Array of total body CoM wrenches expressed in world coordinates.\n
    Each element is a 6D vector comprising a 3D linear + 3D angular components.\n
    Shape is ``(num_bodies,)`` and dtype is :class:`vec6f`.
    """

    w_i_e: wp.array | None = None
    """
    Array of external body CoM wrenches expressed in world coordinates.\n
    Each element is a 6D vector comprising a 3D linear + 3D angular components.\n
    Shape is ``(num_bodies,)`` and dtype is :class:`vec6f`.
    """

    q_j: wp.array | None = None
    """
    Array of generalized joint coordinates.\n
    Shape is ``(sum_of_num_joint_coords,)`` and dtype is :class:`float32`.
    """

    q_j_p: wp.array | None = None
    """
    Array of previous generalized joint coordinates.\n
    Shape is ``(sum_of_num_joint_coords,)`` and dtype is :class:`float32`.
    """

    dq_j: wp.array | None = None
    """
    Array of generalized joint velocities.\n
    Shape is ``(sum_of_num_joint_dofs,)`` and dtype is :class:`float32`.
    """

    lambda_j: wp.array | None = None
    """
    Array of generalized joint constraint forces.\n
    Shape is ``(sum_of_num_joint_cts,)`` and dtype is :class:`float32`.
    """

    ###
    # Operations
    ###

    def copy_to(self, other: StateKamino) -> None:
        """
        Copy the current data to another :class:`StateKamino` object.

        Args:
            other: The target :class:`StateKamino` object to copy data into.
        """
        if other is None:
            raise ValueError("A StateKamino instance must be provided to copy to.")
        if not isinstance(other, StateKamino):
            raise TypeError(f"Expected state of type StateKamino, but got {type(other)}.")

        other.copy_from(self)

    def copy_from(self, other: StateKamino) -> None:
        """
        Copy the data from another :class:`StateKamino` object into the current.

        Args:
            other: The source :class:`StateKamino` object to copy data from.
        """
        if other is None:
            raise ValueError("A StateKamino instance must be provided to copy from.")
        if not isinstance(other, StateKamino):
            raise TypeError(f"Expected state of type StateKamino, but got {type(other)}.")
        if self.q_i is None or other.q_i is None:
            raise ValueError("Error copying from/to uninitialized StateKamino")

        wp.copy(self.q_i, other.q_i)
        wp.copy(self.u_i, other.u_i)
        wp.copy(self.w_i, other.w_i)
        wp.copy(self.w_i_e, other.w_i_e)
        wp.copy(self.q_j, other.q_j)
        wp.copy(self.q_j_p, other.q_j_p)
        wp.copy(self.dq_j, other.dq_j)
        wp.copy(self.lambda_j, other.lambda_j)

    def convert_to_body_com_state(
        self,
        model: Model,
        world_mask: wp.array | None = None,
        body_wid: wp.array | None = None,
    ) -> None:
        """
        Convert the body-frame state to body center-of-mass (CoM)
        state using the provided body center-of-mass offsets.

        Args:
            model (Model):
                The model container holding the time-invariant parameters of the simulation.
            world_mask: optional per-world mask selecting which worlds to process.
            body_wid: body-to-world index mapping, required when ``world_mask`` is given.
        """
        # Ensure the model is valid
        if model is None:
            raise ValueError("Model must be provided to convert to body CoM state.")
        if not isinstance(model, Model):
            raise TypeError(f"Expected model of type Model, but got {type(model)}.")
        if model.body_com is None:
            raise ValueError("Model must have body_com defined to convert to body CoM state.")

        convert_body_origin_to_com(
            body_com=model.body_com,
            body_q=self.q_i,
            body_q_com=self.q_i,
            body_wid=body_wid,
            world_mask=world_mask,
        )

    def convert_to_body_frame_state(
        self,
        model: Model,
        world_mask: wp.array | None = None,
        body_wid: wp.array | None = None,
    ) -> None:
        """
        Convert the body center-of-mass (CoM) state to body-frame
        state using the provided body center-of-mass offsets.

        Args:
            model (Model):
                The model container holding the time-invariant parameters of the simulation.
            world_mask: optional per-world mask selecting which worlds to process.
            body_wid: body-to-world index mapping, required when ``world_mask`` is given.
        """
        # Ensure the model is valid
        if model is None:
            raise ValueError("Model must be provided to convert to body CoM state.")
        if not isinstance(model, Model):
            raise TypeError(f"Expected model of type Model, but got {type(model)}.")
        if model.body_com is None:
            raise ValueError("Model must have body_com defined to convert to body CoM state.")

        convert_body_com_to_origin(
            body_com=model.body_com,
            body_q_com=self.q_i,
            body_q=self.q_i,
            body_wid=body_wid,
            world_mask=world_mask,
        )

    @staticmethod
    def from_newton(
        size: SizeKamino,
        model: Model,
        state: State,
        initialize_state_prev: bool = False,
        convert_to_com_frame: bool = False,
    ) -> StateKamino:
        """
        Constructs a :class:`kamino.StateKamino` object from a :class:`newton.State` object.

        This operation serves only as an adaptor-like constructor to interface a
        :class:`newton.State`, effectively creating an alias without copying data.

        Args:
            state (State):
                The source :class:`newton.State` object to be adapted.
            initialize_state_prev (bool):
                If True, initialize the `joint_q_prev` attribute to
                match the current `joint_q` from the newton.State.
            convert_to_com_frame (bool):
                If True, convert body poses to local center-of-mass frames.

        Returns:
            A :class:`kamino.StateKamino` object that aliases the data of the input :class:`newton.State` object.
        """
        # Ensure the state is valid
        if state is None:
            raise ValueError("A State instance must be provided to convert to StateKamino.")
        if not isinstance(state, State):
            raise TypeError(f"Expected state of type State, but got {type(state)}.")

        # Retrieve the device of the state container
        device = None
        if state.body_q is not None:
            device = state.body_q.device
        elif state.joint_q is not None:
            device = state.joint_q.device
        else:
            raise ValueError("State must have at least body_q or joint_q defined to determine device for StateKamino.")

        # If the state contains the Kamino-specific `body_f_total` custom attribute,
        # capture a reference to it; otherwise, create a new array for it.
        if hasattr(state, "body_f_total"):
            body_f_total = state.body_f_total
        else:
            body_f_total = wp.zeros_like(state.body_f)
            state.body_f_total = body_f_total

        # If the state contains the Kamino-specific `joint_q_prev` custom attribute,
        # capture a reference to it; otherwise, create a new array for it.
        if hasattr(state, "joint_q_prev"):
            joint_q_prev = state.joint_q_prev
        else:
            joint_q_prev = wp.clone(state.joint_q)
            state.joint_q_prev = joint_q_prev

        # If the state contains the Kamino-specific `joint_lambdas` custom attribute,
        # capture a reference to it; otherwise, create a new array for it.
        is_joint_lambdas_valid = (
            hasattr(state, "joint_lambdas")
            and state.joint_lambdas is not None
            and state.joint_lambdas.shape == (size.sum_of_num_joint_cts,)
        )
        if is_joint_lambdas_valid:
            joint_lambdas = state.joint_lambdas
        else:
            joint_lambdas = wp.zeros(shape=(size.sum_of_num_joint_cts,), dtype=wp.float32, device=device)
            state.joint_lambdas = joint_lambdas

        # Optionally initialize the `joint_q_prev` array to match the current `joint_q`
        if initialize_state_prev:
            wp.copy(joint_q_prev, state.joint_q)

        # Create a new StateKamino object, aliasing the relevant data from the input newton.State
        state_kamino = StateKamino(
            q_i=state.body_q,
            u_i=state.body_qd.view(dtype=vec6f),  # TODO: change to wp.spatial_vector
            w_i=body_f_total.view(dtype=vec6f),  # TODO: change to wp.spatial_vector
            w_i_e=state.body_f.view(dtype=vec6f),  # TODO: change to wp.spatial_vector
            q_j=state.joint_q,
            q_j_p=joint_q_prev,
            dq_j=state.joint_qd,
            lambda_j=joint_lambdas,
        )

        # Optionally convert body poses to CoM frame
        if convert_to_com_frame:
            state_kamino.convert_to_body_com_state(model)

        # Return the StateKamino object, aliasing the
        # relevant data from the input newton.State
        return state_kamino

    @staticmethod
    def to_newton(model: Model, state: StateKamino, convert_to_body_frame: bool = False) -> State:
        """
        Constructs a :class:`newton.State` object from a :class:`kamino.StateKamino` object.

        This operation serves only as an adaptor-like constructor to interface a
        :class:`kamino.StateKamino`, effectively creating an alias without copying data.

        Args:
            state (StateKamino):
                The source :class:`kamino.StateKamino` object to be adapted.
            convert_to_body_frame (bool):
                If True, convert body poses to body-local frames.

        Returns:
            A :class:`newton.State` object that aliases the data of the input :class:`kamino.StateKamino` object.
        """
        # Ensure the model is valid
        if model is None:
            raise ValueError("A Model instance must be provided to convert to StateKamino.")
        if not isinstance(model, Model):
            raise TypeError(f"Expected model of type Model, but got {type(model)}.")

        # Ensure the state is valid
        if state is None:
            raise ValueError("A StateKamino instance must be provided to convert to State.")
        if not isinstance(state, StateKamino):
            raise TypeError(f"Expected state of type StateKamino, but got {type(state)}.")

        # Optionally convert body poses to body frame
        if convert_to_body_frame:
            state.convert_to_body_frame_state(model)

        # Create a new State object, aliasing the relevant
        # data from the input kamino.StateKamino
        state_newton = State()
        state_newton.body_q = state.q_i
        state_newton.body_qd = state.u_i.view(dtype=wp.spatial_vectorf)
        state_newton.body_f = state.w_i_e.view(dtype=wp.spatial_vectorf)
        state_newton.joint_q = state.q_j
        state_newton.joint_qd = state.dq_j

        # Add Kamino-specific custom attributes to the newton.State object
        state_newton.body_f_total = state.w_i.view(dtype=wp.spatial_vectorf)
        state_newton.joint_q_prev = state.q_j_p
        state_newton.joint_lambdas = state.lambda_j

        # Return the new newton.State object
        return state_newton
