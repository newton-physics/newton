# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Provides a set of conversion utilities to bridge Kamino and Newton."""

import numpy as np
import warp as wp

from .....sim.model import Model
from ..utils import logger as msg
from .joints import JointDoFType, JointsModel
from .shapes import max_contacts_for_shape_pair

###
# Module interface
###

__all__ = [
    "compute_required_contact_capacity",
    "convert_entity_local_transforms",
    "convert_model_joint_transforms",
]

###
# Functions
###


def convert_entity_local_transforms(model: Model) -> dict[str, np.ndarray]:
    """
    Converts all entity-local transforms (i.e. of bodies, joints and shapes) in the
    given  Newton model to a format that is compatible with Kamino's constraint system.

    This involves absorbing any non-identity :attr:`Model.joint_X_c`
    rotations into the child body frames, and updating all downstream
    :attr:`Model.joint_X_p` and :attr:`Model.shape_transform` accordingly.

    Args:
        model (Model): Newton model to be modified in-place.
    """
    # ---------------------------------------------------------------------------
    # Pre-processing: absorb non-identity joint_X_c rotations into child body
    # frames so that Kamino sees aligned joint frames on both sides.
    #
    # Kamino's constraint system assumes a single joint frame X_j valid for both
    # the base (parent) and follower (child) bodies.  At q = 0 it requires
    #   q_base^{-1} * q_follower = identity
    #
    # Newton, however, allows different parent / child joint-frame orientations
    # via joint_X_p and joint_X_c.  At q = 0 Newton's FK gives:
    #   q_follower = q_parent * q_pj * inv(q_cj)
    # so q_base^{-1} * q_follower = q_pj * inv(q_cj) which is generally not
    # identity.
    #
    # To fix this we apply a per-body correction rotation q_corr = q_cj * inv(q_pj)
    # (applied on the right) to each child body's frame:
    #   q_body_new = q_body_old * q_corr
    #
    # This makes q_base^{-1} * q_follower_new = identity at q = 0, and the joint
    # rotation axis R(q_pj) * axis is preserved.
    #
    # All body-local quantities (CoM, inertia, shapes) are re-expressed in the
    # rotated frame, and downstream joint_X_p transforms are updated to account
    # for the parent body's frame change.
    # ---------------------------------------------------------------------------

    def _to_wpq(q):
        return wp.quat(float(q[0]), float(q[1]), float(q[2]), float(q[3]))

    def _from_wpq(q):
        return np.array([q[0], q[1], q[2], q[3]], dtype=np.float64)

    def _quat_is_identity(q, tol=1e-5):
        return abs(abs(q[3]) - 1.0) < tol

    def _quat_mul(a, b):
        return _from_wpq(_to_wpq(a) * _to_wpq(b))

    def _quat_inv(q):
        return _from_wpq(wp.quat_inverse(_to_wpq(q)))

    def _quat_rotate_vec(q, v):
        r = wp.quat_rotate(_to_wpq(q), wp.vec3(float(v[0]), float(v[1]), float(v[2])))
        return np.array([r[0], r[1], r[2]], dtype=np.float64)

    def _quat_to_mat33(q):
        return np.array(wp.quat_to_matrix(_to_wpq(q)), dtype=np.float64).reshape(3, 3)

    # Work on copies so the original Newton model is not mutated
    body_com_np: np.ndarray = model.body_com.numpy().copy()
    body_q_np: np.ndarray = model.body_q.numpy().copy()
    body_qd_np: np.ndarray = model.body_qd.numpy().copy()
    body_inertia_np: np.ndarray = model.body_inertia.numpy().copy()
    body_inv_inertia_np: np.ndarray = model.body_inv_inertia.numpy().copy()
    joint_parent_np: np.ndarray = model.joint_parent.numpy().copy()
    joint_child_np: np.ndarray = model.joint_child.numpy().copy()
    joint_X_p_np: np.ndarray = model.joint_X_p.numpy().copy()
    joint_X_c_np: np.ndarray = model.joint_X_c.numpy().copy()
    shape_transform_np: np.ndarray = model.shape_transform.numpy().copy()
    shape_body_np: np.ndarray = model.shape_body.numpy().copy()

    # Process joints in tree order (Newton stores them parent-before-child).
    # For each joint whose q_pj * inv(q_cj) is not identity, we apply a
    # correction q_corr to the child body's frame and immediately propagate
    # to all downstream joints that reference the corrected body as parent.
    body_corr: dict[int, np.ndarray] = {}  # body_index -> cumulative q_corr

    for j in range(model.joint_count):
        parent = int(joint_parent_np[j])
        child = int(joint_child_np[j])

        # If the parent body was previously corrected, first update this
        # joint's parent-side transform to the new parent frame.
        if parent >= 0 and parent in body_corr:
            q_par_corr_inv = _quat_inv(body_corr[parent])
            p_pos = joint_X_p_np[j, :3].astype(np.float64)
            joint_X_p_np[j, :3] = _quat_rotate_vec(q_par_corr_inv, p_pos)
            p_quat = joint_X_p_np[j, 3:7].astype(np.float64)
            joint_X_p_np[j, 3:7] = _quat_mul(q_par_corr_inv, p_quat)

        # Now compute the correction for this joint's child body
        q_cj = joint_X_c_np[j, 3:7].astype(np.float64)
        q_pj = joint_X_p_np[j, 3:7].astype(np.float64)
        q_corr = _quat_mul(q_cj, _quat_inv(q_pj))

        if child < 0 or _quat_is_identity(q_corr):
            continue

        if child in body_corr:
            msg.warning(
                "Body %d is the child of multiple joints requiring joint_X_c "
                "correction. The previous correction will be overwritten, which "
                "may produce incorrect joint constraints for loop-closing joints.",
                child,
            )
        body_corr[child] = q_corr.copy()

        # Update child-side joint transform: rotation becomes identity,
        # position re-expressed in the new child frame
        q_corr_inv = _quat_inv(q_corr)
        c_pos = joint_X_c_np[j, :3].astype(np.float64)
        joint_X_c_np[j, :3] = _quat_rotate_vec(q_corr_inv, c_pos)
        joint_X_c_np[j, 3:7] = [0.0, 0.0, 0.0, 1.0]

        # Rotate the child body's local quantities
        R_inv_corr = _quat_to_mat33(q_corr_inv)

        q_old = body_q_np[child, 3:7].astype(np.float64)
        body_q_np[child, 3:7] = _quat_mul(q_old, q_corr)

        body_com_np[child] = _quat_rotate_vec(q_corr_inv, body_com_np[child].astype(np.float64))

        body_inertia_np[child] = R_inv_corr @ body_inertia_np[child].astype(np.float64) @ R_inv_corr.T
        body_inv_inertia_np[child] = R_inv_corr @ body_inv_inertia_np[child].astype(np.float64) @ R_inv_corr.T

        # TODO: Do these need be converted? Aren't they already computed at body CoM?
        body_qd_np[child, :3] = R_inv_corr @ body_qd_np[child, :3].astype(np.float64)
        body_qd_np[child, 3:6] = R_inv_corr @ body_qd_np[child, 3:6].astype(np.float64)

        for s in range(model.shape_count):
            if int(shape_body_np[s]) != child:
                continue
            s_pos = shape_transform_np[s, :3].astype(np.float64)
            s_quat = shape_transform_np[s, 3:7].astype(np.float64)
            shape_transform_np[s, :3] = _quat_rotate_vec(q_corr_inv, s_pos)
            shape_transform_np[s, 3:7] = _quat_mul(q_corr_inv, s_quat)

    if body_corr:
        msg.debug(
            "Absorbed joint_X_c rotations for %d child bodies: %s",
            len(body_corr),
            list(body_corr.keys()),
        )

    # Return the converted transforms as numpy arrays
    # to be used for constructing the Kamino model
    return {
        "body_q": body_q_np,
        "body_qd": body_qd_np,
        "body_com": body_com_np,
        "body_inertia": body_inertia_np,
        "body_inv_inertia": body_inv_inertia_np,
        "shape_transform": shape_transform_np,
        "joint_X_p": joint_X_p_np,
        "joint_X_c": joint_X_c_np,
    }


def compute_required_contact_capacity(
    model: Model,
    max_contacts_per_pair: int | None = None,
    max_contacts_per_world: int | None = None,
) -> tuple[int, list[int]]:
    """
    Computes the required contact capacity for a given Newton model.

    The outputs are used to determine the minimum number of contacts
    to be allocated, according to the shapes present in the model.

    Args:
        model (Model):
            The Newton model for which to compute the required contact capacity.
        max_contacts_per_pair (int, optional):
            Optional maximum number of contacts to allocate per shape pair.
            If `None`, no per-pair limit is applied.
        max_contacts_per_world (int, optional):
            Optional maximum number of contacts to allocate per world.
            If `None`, no per-world limit is applied, otherwise it will
            override the computed per-world requirements if it is larger.

    Returns:
        (model_required_contacts, world_required_contacts):
            A tuple containing:
            - `model_required_contacts` (int):
                The total number of contacts required for the entire model.
            - `world_required_contacts` (list[int]):
                A list of required contacts per world, where the length of the
                list is equal to `model.world_count` and each entry corresponds
                to the required contacts for that world.

    """
    # Retrieve the shape types, worlds and collision candidate pairs from the model
    shape_type_np = model.shape_type.numpy()
    shape_world_np = model.shape_world.numpy()
    shape_contact_pairs = model.shape_contact_pairs.numpy().tolist()

    # First check if there are any collision geometries
    if model.shape_count == 0:
        return 0, [0] * model.world_count

    # Compute the maximum possible number of geom pairs per world.
    # Global shapes (world=-1, e.g. ground plane) can collide with shapes
    # from any world — attribute their contacts to the non-global shape's world.
    world_max_contacts = [0] * model.world_count
    for shape_pair in shape_contact_pairs:
        s1 = int(shape_pair[0])
        s2 = int(shape_pair[1])
        type1 = int(shape_type_np[s1])
        type2 = int(shape_type_np[s2])
        if type1 > type2:
            s1, s2 = s2, s1
            type1, type2 = type2, type1
        num_contacts_a, num_contacts_b = max_contacts_for_shape_pair(
            type_a=type1,
            type_b=type2,
        )
        num_contacts = num_contacts_a + num_contacts_b
        # Determine the world for this pair — fall back to other shape if one is global
        w1 = int(shape_world_np[s1])
        w2 = int(shape_world_np[s2])
        wid = w1 if w1 >= 0 else w2
        if wid < 0:
            continue  # Both shapes are global — skip
        if max_contacts_per_pair is not None:
            world_max_contacts[wid] += min(num_contacts, max_contacts_per_pair)
        else:
            world_max_contacts[wid] += num_contacts

    # Override the per-world maximum contacts if specified in the settings
    if max_contacts_per_world is not None:
        for w in range(model.world_count):
            world_max_contacts[w] = min(world_max_contacts[w], max_contacts_per_world)

    # Return the per-world maximum contacts list
    return sum(world_max_contacts), world_max_contacts


# TODO: Re-implement this using a kernel to run in parallel on the GPU if possible
def convert_model_joint_transforms(model: Model, joints: JointsModel) -> None:
    """
    Converts the joint model parameterization of Newton's to Kamino's format.

    This essentially involves computing the B_r_Bj, F_r_Fj and
    X_j arrays from the joint_X_p and joint_X_c transforms.

    Args:
    - model (Model):
        The input Newton model containing the joint information to be converted.
    - joints (JointsModel):
        The output JointsModel instance where the converted joint data will be stored.
        This function modifies the `joints` object in-place.
    """
    joint_X_p_np = model.joint_X_p.numpy()
    joint_X_c_np = model.joint_X_c.numpy()
    body_com_np = model.body_com.numpy()
    joint_parent_np = model.joint_parent.numpy()
    joint_child_np = model.joint_child.numpy()
    joint_axis_np = model.joint_axis.numpy()
    joint_dof_dim_np = model.joint_dof_dim.numpy()
    joint_qd_start_np = model.joint_qd_start.numpy()
    joint_limit_lower_np = model.joint_limit_lower.numpy()
    joint_limit_upper_np = model.joint_limit_upper.numpy()
    dof_type_np = joints.dof_type.numpy()

    n_joints = model.joint_count
    B_r_Bj_np = np.zeros((n_joints, 3), dtype=np.float32)
    F_r_Fj_np = np.zeros((n_joints, 3), dtype=np.float32)
    X_j_np = np.zeros((n_joints, 9), dtype=np.float32)

    for j in range(n_joints):
        dof_type_j = JointDoFType(int(dof_type_np[j]))
        dof_dim_j = (int(joint_dof_dim_np[j][0]), int(joint_dof_dim_np[j][1]))
        dofs_start_j = int(joint_qd_start_np[j])
        ndofs_j = dof_type_j.num_dofs
        joint_axes_j = joint_axis_np[dofs_start_j : dofs_start_j + ndofs_j]
        joint_q_min_j = joint_limit_lower_np[dofs_start_j : dofs_start_j + ndofs_j]
        joint_q_max_j = joint_limit_upper_np[dofs_start_j : dofs_start_j + ndofs_j]
        R_axis_j = JointDoFType.from_newton(dof_type_j, dof_dim_j, joint_axes_j, joint_q_min_j, joint_q_max_j)

        parent_bid = int(joint_parent_np[j])
        p_r_p_com = wp.vec3f(body_com_np[parent_bid]) if parent_bid >= 0 else wp.vec3f(0.0)
        c_r_c_com = wp.vec3f(body_com_np[int(joint_child_np[j])])

        X_p_j = wp.transformf(*joint_X_p_np[j, :])
        X_c_j = wp.transformf(*joint_X_c_np[j, :])
        q_p_j = wp.transform_get_rotation(X_p_j)
        p_r_p_j = wp.transform_get_translation(X_p_j)
        c_r_c_j = wp.transform_get_translation(X_c_j)

        B_r_Bj_np[j, :] = p_r_p_j - p_r_p_com
        F_r_Fj_np[j, :] = c_r_c_j - c_r_c_com
        X_j_np[j, :] = wp.quat_to_matrix(q_p_j) @ R_axis_j

    joints.B_r_Bj.assign(B_r_Bj_np)
    joints.F_r_Fj.assign(F_r_Fj_np)
    joints.X_j.assign(X_j_np.reshape((n_joints, 3, 3)))
