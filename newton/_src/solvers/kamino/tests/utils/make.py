# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
Utilities for constructing test problem containers and data.
"""

import math
from collections.abc import Callable

import numpy as np
import warp as wp

from ..._src.core.bodies import update_body_inertias
from ..._src.core.builder import ModelBuilderKamino
from ..._src.core.data import DataKamino
from ..._src.core.math import quat_exp, screw, screw_angular, screw_linear
from ..._src.core.model import ModelKamino
from ..._src.core.state import StateKamino
from ..._src.core.types import float32, int32, mat33f, transformf, vec3f, vec6f
from ..._src.geometry.contacts import ContactsKamino
from ..._src.geometry.detector import CollisionDetector
from ..._src.kinematics.constraints import make_unilateral_constraints_info, update_constraints_info
from ..._src.kinematics.jacobians import DenseSystemJacobians, SparseSystemJacobians, SystemJacobiansType
from ..._src.kinematics.joints import compute_joints_data
from ..._src.kinematics.limits import LimitsKamino
from ..._src.models.builders import basics as _model_basics
from ..._src.models.builders import utils as _model_utils
from .print import (
    print_data_info,
    print_model_constraint_info,
)

###
# Module interface
###

__all__ = [
    "make_containers",
    "make_generalized_mass_matrices",
    "make_inverse_generalized_mass_matrices",
    "make_test_problem",
    "make_test_problem_fourbar",
    "update_containers",
]


###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# NumPy Reference Data Generators
###


def make_generalized_mass_matrices(model: ModelKamino, data: DataKamino) -> list[np.ndarray]:
    # Extract the masses and inertias as numpy arrays
    m_i = model.bodies.m_i.numpy()
    I_i = data.bodies.I_i.numpy()

    # Initialize a list to hold the generalized mass matrices
    M_np: list[np.ndarray] = []

    # Iterate over each world in the model and construct the generalized mass matrix
    num_worlds = model.info.num_worlds
    bodies_offset = model.info.bodies_offset.numpy().tolist()
    for w in range(num_worlds):
        bio = bodies_offset[w]
        nb = bodies_offset[w + 1] - bio
        M = np.zeros((6 * nb, 6 * nb), dtype=np.float32)
        for i in range(nb):
            start = 6 * i
            M[start : start + 3, start : start + 3] = m_i[bio + i] * np.eye(3)  # Linear part
            M[start + 3 : start + 6, start + 3 : start + 6] = I_i[bio + i]  # Angular part
        M_np.append(M)

    # Return the list of generalized mass matrices
    return M_np


def make_inverse_generalized_mass_matrices(model: ModelKamino, data: DataKamino) -> list[np.ndarray]:
    # Extract the inverse masses and inertias as numpy arrays
    inv_m_i = model.bodies.inv_m_i.numpy()
    inv_I_i = data.bodies.inv_I_i.numpy()

    # Initialize a list to hold the inverse generalized mass matrices
    invM_np: list[np.ndarray] = []

    # Iterate over each world in the model and construct the inverse generalized mass matrix
    num_worlds = model.info.num_worlds
    bodies_offset = model.info.bodies_offset.numpy().tolist()
    for w in range(num_worlds):
        bio = bodies_offset[w]
        nb = bodies_offset[w + 1] - bio
        invM = np.zeros((6 * nb, 6 * nb), dtype=np.float32)
        for i in range(nb):
            start = 6 * i
            invM[start : start + 3, start : start + 3] = inv_m_i[bio + i] * np.eye(3)  # Linear part
            invM[start + 3 : start + 6, start + 3 : start + 6] = inv_I_i[bio + i]  # Angular part
        invM_np.append(invM)

    # Return the list of inverse generalized mass matrices
    return invM_np


###
# Test Problem Scaffolding
###


def make_containers(
    builder: ModelBuilderKamino,
    device: wp.DeviceLike = None,
    max_world_contacts: int = 0,
    sparse: bool = True,
    dt: float = 0.001,
) -> tuple[ModelKamino, DataKamino, StateKamino, LimitsKamino, CollisionDetector, SystemJacobiansType]:
    # Create the model from the builder
    model = builder.finalize(device=device)

    # Configure model time-steps
    model.time.dt.fill_(wp.float32(dt))
    model.time.inv_dt.fill_(wp.float32(1.0 / dt))

    # Create a data and state container
    data = model.data()
    state = model.state()

    # Create the limits container
    limits = LimitsKamino(model=model)

    # Create the collision detector
    config = CollisionDetector.Config(max_contacts_per_world=max_world_contacts, pipeline="primitive")
    detector = CollisionDetector(model=model, config=config)

    # Construct the unilateral constraints members in the model info
    make_unilateral_constraints_info(model, data, limits, detector.contacts)

    # Create the Jacobians container
    if sparse:
        jacobians = SparseSystemJacobians(model=model, limits=limits, contacts=detector.contacts)
    else:
        jacobians = DenseSystemJacobians(model=model, limits=limits, contacts=detector.contacts)

    # Return the model, data, detector, and jacobians
    return model, data, state, limits, detector, jacobians


def update_containers(
    model: ModelKamino,
    data: DataKamino,
    state: StateKamino,
    limits: LimitsKamino | None = None,
    detector: CollisionDetector | None = None,
    jacobians: SystemJacobiansType | None = None,
):
    # Update body inertias according to the current state of the bodies
    update_body_inertias(model=model.bodies, data=data.bodies)
    wp.synchronize()

    # Update joint states according to the state of the bodies
    compute_joints_data(model=model, data=data, q_j_p=wp.zeros_like(data.joints.q_j))
    wp.synchronize()

    # Run joint-limit detection to generate active limits
    if limits is not None:
        limits.detect(q_j=data.joints.q_j)
        wp.synchronize()

    # Run collision detection to generate active contacts
    if detector is not None:
        detector.collide(data=data, state=state)
        wp.synchronize()

    # Update the constraint state info
    update_constraints_info(model=model, data=data)
    wp.synchronize()

    # Build the dense system Jacobians
    if jacobians is not None:
        ldata = limits.data if limits is not None else None
        cdata = detector.contacts.data if detector is not None else None
        jacobians.build(model=model, data=data, limits=ldata, contacts=cdata)
        wp.synchronize()


def make_test_problem(
    builder: ModelBuilderKamino,
    set_state_fn: Callable[[ModelKamino, DataKamino], None] | None = None,
    device: wp.DeviceLike = None,
    max_world_contacts: int = 12,
    with_limits: bool = False,
    with_contacts: bool = False,
    dt: float = 0.001,
    verbose: bool = False,
) -> tuple[ModelKamino, DataKamino, StateKamino, LimitsKamino | None, ContactsKamino | None]:
    # Create the model from the builder
    model = builder.finalize(device=device)

    # Configure model time-steps
    model.time.dt.fill_(wp.float32(dt))
    model.time.inv_dt.fill_(wp.float32(1.0 / dt))

    # Create a model state container
    data = model.data(device=device)
    state = model.state(device=device)

    # Construct and allocate the limits container
    limits = None
    if with_limits:
        limits = LimitsKamino(model=model)

    # Create the collision detector
    contacts = None
    if with_contacts:
        config = CollisionDetector.Config(max_contacts_per_world=max_world_contacts, pipeline="primitive")
        detector = CollisionDetector(model=model, config=config)
        contacts = detector.contacts

    # Create the constraints info
    make_unilateral_constraints_info(
        model=model,
        data=data,
        limits=limits,
        contacts=contacts,
    )
    if verbose:
        print("")  # Add a newline for better readability
        print_model_constraint_info(model)
        print_data_info(data)
        print("\n")  # Add a newline for better readability

    # If a set-state callback is provided, perturb the system
    # NOTE: This is done to potentially trigger
    # joint limits and contacts to become active
    if set_state_fn is not None:
        set_state_fn(model=model, data=data)
        wp.synchronize()
    if verbose:
        print("data.bodies.q_i:\n", data.bodies.q_i)
        print("data.bodies.u_i:\n\n", data.bodies.u_i)

    # Compute the joints state
    compute_joints_data(model=model, data=data, q_j_p=wp.zeros_like(data.joints.q_j))
    wp.synchronize()
    if verbose:
        print("data.joints.p_j:\n", data.joints.p_j)
        print("data.joints.r_j:\n", data.joints.r_j)
        print("data.joints.dr_j:\n", data.joints.dr_j)
        print("data.joints.q_j:\n", data.joints.q_j)
        print("data.joints.dq_j:\n\n", data.joints.dq_j)

    # Run limit detection to generate active limits
    if with_limits:
        limits.detect(q_j=data.joints.q_j)
        wp.synchronize()
        if verbose:
            print(f"limits.world_active_limits: {limits.world_active_limits}")
            print(f"data.info.num_limits: {data.info.num_limits}\n\n")

    # Run collision detection to generate active contacts
    if with_contacts:
        detector.collide(data, state)
        wp.synchronize()
        if verbose:
            print(f"contacts.world_active_contacts: {detector.contacts.world_active_contacts}")
            print(f"data.info.num_contacts: {data.info.num_contacts}\n\n")

    # Update the constraints info
    update_constraints_info(model=model, data=data)
    if verbose:
        print("")  # Add a newline for better readability
        print_data_info(data)
        print("\n")  # Add a newline for better readability
    wp.synchronize()

    # Return the problem data containers
    return model, data, state, limits, contacts


def make_constraint_multiplier_arrays(model: ModelKamino) -> tuple[wp.array, wp.array]:
    with wp.ScopedDevice(model.device):
        lambdas = wp.zeros(model.size.sum_of_max_total_cts, dtype=float32)
    return model.info.total_cts_offset, lambdas


###
# Fourbar
#
# Generates a problem using basics.boxes_fourbar model with a specific
# state configuration that induces active joint limits and contacts.
###

Q_X_J = 0.3 * math.pi
THETA_Y_J = 0.0
THETA_Z_J = 0.0
J_DR_J = vec3f(0.0)
J_DV_J = vec3f(0.0)
J_DOMEGA_J = vec3f(0.0)


@wp.kernel
def _set_fourbar_body_states(
    model_joint_bid_B: wp.array[int32],
    model_joint_bid_F: wp.array[int32],
    model_joint_B_r_Bj: wp.array[vec3f],
    model_joint_F_r_Fj: wp.array[vec3f],
    model_joint_X_j: wp.array[mat33f],
    state_body_q_i: wp.array[transformf],
    state_body_u_i: wp.array[vec6f],
):
    """
    Set the state of the bodies to a certain values in order to check computations of joint states.
    """
    # Retrieve the thread index as the joint index
    jid = wp.tid()

    # Retrieve the joint parameters
    bid_B = model_joint_bid_B[jid]
    bid_F = model_joint_bid_F[jid]
    B_r_Bj = model_joint_B_r_Bj[jid]
    F_r_Fj = model_joint_F_r_Fj[jid]
    X_j = model_joint_X_j[jid]

    # Retrieve the current state of the Base body
    p_B = state_body_q_i[bid_B]
    u_B = state_body_u_i[bid_B]

    # Extract the position and orientation of the Base body
    r_B = wp.transform_get_translation(p_B)
    q_B = wp.transform_get_rotation(p_B)
    R_B = wp.quat_to_matrix(q_B)

    # Extract the linear and angular velocity of the Base body
    v_B = screw_linear(u_B)
    omega_B = screw_angular(u_B)

    # Define the joint rotation offset
    # NOTE: X_j projects quantities into the joint frame
    # NOTE: X_j^T projects quantities into the outer frame (world or body)
    q_x_j = Q_X_J * wp.pow(-1.0, float(jid))  # Alternate sign for each joint
    theta_y_j = THETA_Y_J
    theta_z_j = THETA_Z_J
    j_dR_j = vec3f(q_x_j, theta_y_j, theta_z_j)  # Joint offset as rotation vector
    q_jq = quat_exp(j_dR_j)  # Joint offset as rotation quaternion
    R_jq = wp.quat_to_matrix(q_jq)  # Joint offset as rotation matrix

    # Define the joint translation offset
    j_dr_j = J_DR_J

    # Define the joint twist offset
    j_dv_j = J_DV_J
    j_domega_j = J_DOMEGA_J

    # Follower body rotation via the Base and joint frames
    R_B_X_j = R_B @ X_j
    R_F_new = R_B_X_j @ R_jq @ wp.transpose(X_j)
    q_F_new = wp.quat_from_matrix(R_F_new)

    # Follower body position via the Base and joint frames
    r_Fj = R_F_new @ F_r_Fj
    r_F_new = r_B + R_B @ B_r_Bj + R_B_X_j @ j_dr_j - r_Fj

    # Follower body twist via the Base and joint frames
    r_Bj = R_B @ B_r_Bj
    r_Fj = R_F_new @ F_r_Fj
    omega_F_new = R_B_X_j @ j_domega_j + omega_B
    v_F_new = R_B_X_j @ j_dv_j + v_B + wp.cross(omega_B, r_Bj) - wp.cross(omega_F_new, r_Fj)

    # Offset the pose of the body by a fixed amount
    state_body_q_i[bid_F] = wp.transformation(r_F_new, q_F_new, dtype=float32)
    state_body_u_i[bid_F] = screw(v_F_new, omega_F_new)


def set_fourbar_body_states(model: ModelKamino, data: DataKamino):
    wp.launch(
        _set_fourbar_body_states,
        dim=3,  # Set to three because we only need to set the first three joints
        inputs=[
            model.joints.bid_B,
            model.joints.bid_F,
            model.joints.B_r_Bj,
            model.joints.F_r_Fj,
            model.joints.X_j,
            data.bodies.q_i,
            data.bodies.u_i,
        ],
        device=model.device,
    )


def make_test_problem_fourbar(
    device: wp.DeviceLike = None,
    max_world_contacts: int = 12,
    num_worlds: int = 1,
    with_limits: bool = False,
    with_contacts: bool = False,
    with_implicit_joints: bool = True,
    verbose: bool = False,
) -> tuple[ModelKamino, DataKamino, StateKamino, LimitsKamino | None, ContactsKamino | None]:
    # Define the problem using the ModelBuilderKamino
    builder: ModelBuilderKamino = _model_utils.make_homogeneous_builder(
        num_worlds=num_worlds,
        build_fn=_model_basics.build_boxes_fourbar,
        dynamic_joints=with_implicit_joints,
        implicit_pd=with_implicit_joints,
    )

    # Generate the problem containers using the builder
    return make_test_problem(
        builder=builder,
        set_state_fn=set_fourbar_body_states,
        device=device,
        max_world_contacts=max_world_contacts,
        with_limits=with_limits,
        with_contacts=with_contacts,
        verbose=verbose,
    )


def make_test_problem_heterogeneous(
    device: wp.DeviceLike = None,
    max_world_contacts: int = 12,
    with_limits: bool = False,
    with_contacts: bool = False,
    with_implicit_joints: bool = True,
    verbose: bool = False,
) -> tuple[ModelKamino, DataKamino, StateKamino, LimitsKamino | None, ContactsKamino | None]:
    # Define the problem using the ModelBuilderKamino
    builder: ModelBuilderKamino = _model_basics.make_basics_heterogeneous_builder(
        dynamic_joints=with_implicit_joints,
        implicit_pd=with_implicit_joints,
    )

    # Generate the problem containers using the builder
    return make_test_problem(
        builder=builder,
        device=device,
        max_world_contacts=max_world_contacts,
        with_limits=with_limits,
        with_contacts=with_contacts,
        verbose=verbose,
    )
