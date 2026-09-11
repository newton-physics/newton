# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import warnings

import warp as wp

from .enums import JointType
from .model import Model
from .state import State

_SUPPORTED_JOINT_TYPES = {int(JointType.PRISMATIC), int(JointType.REVOLUTE), int(JointType.D6)}
_MAX_REPORTED_UNSUPPORTED_JOINTS = 10


@wp.kernel
def eval_mimic_joints(
    joint_mimic_joint: wp.array[int],
    joint_mimic_coeffs: wp.array[wp.vec2],
    joint_q_start: wp.array[int],
    joint_qd_start: wp.array[int],
    # outputs
    joint_q: wp.array[float],
    joint_qd: wp.array[float],
):
    """Apply joint-owned mimic relationships to generalized coordinates."""
    joint = wp.tid()
    reference_joint = joint_mimic_joint[joint]
    if reference_joint < 0:
        return

    coeffs = joint_mimic_coeffs[joint]
    q_start = joint_q_start[joint]
    reference_q_start = joint_q_start[reference_joint]
    for coordinate in range(joint_q_start[joint + 1] - q_start):
        joint_q[q_start + coordinate] = coeffs[0] + coeffs[1] * joint_q[reference_q_start + coordinate]

    qd_start = joint_qd_start[joint]
    reference_qd_start = joint_qd_start[reference_joint]
    for dof in range(joint_qd_start[joint + 1] - qd_start):
        joint_qd[qd_start + dof] = coeffs[1] * joint_qd[reference_qd_start + dof]


def eval_mimic(model: Model, state_in: State, state_out: State | None = None) -> None:
    """Update follower joint coordinates from their reference joints.

    For each follower, this function reads every position and velocity
    coordinate of the reference joint, then writes the matching follower
    coordinates according to :attr:`Model.joint_mimic_coeffs`. Independent
    joints are left unchanged. Only :attr:`State.joint_q` and
    :attr:`State.joint_qd` are written.

    If ``state_out`` is omitted, ``state_in`` is updated in place. Otherwise,
    all joint coordinates are first copied from ``state_in`` to ``state_out``
    and the followers are updated in ``state_out``.

    Args:
        model: Model containing the joint mimic metadata.
        state_in: State providing the input joint coordinates.
        state_out: State receiving the updated joint coordinates. If ``None``,
            update ``state_in`` in place.

    Raises:
        ValueError: If either state does not contain joint coordinate arrays.
    """
    if state_in.joint_q is None or state_in.joint_qd is None:
        raise ValueError("state_in must contain joint_q and joint_qd arrays")

    if state_out is None:
        state_out = state_in
    elif state_out.joint_q is None or state_out.joint_qd is None:
        raise ValueError("state_out must contain joint_q and joint_qd arrays")
    elif state_out is not state_in:
        state_out.joint_q.assign(state_in.joint_q)
        state_out.joint_qd.assign(state_in.joint_qd)

    if model.joint_count == 0:
        return

    wp.launch(
        kernel=eval_mimic_joints,
        dim=model.joint_count,
        inputs=[
            model.joint_mimic_joint,
            model.joint_mimic_coeffs,
            model.joint_q_start,
            model.joint_qd_start,
        ],
        outputs=[state_out.joint_q, state_out.joint_qd],
        device=model.device,
    )


def has_supported_joint_mimics(model: Model, solver_name: str) -> bool:
    """Return whether a model has supported mimics and warn about unsupported ones."""
    joint_mimic_joint = model.joint_mimic_joint.numpy()
    joint_type = model.joint_type.numpy()
    has_supported = False
    unsupported_count = 0
    unsupported_sample = []
    for follower, reference in enumerate(joint_mimic_joint):
        if reference < 0:
            continue
        if int(joint_type[follower]) in _SUPPORTED_JOINT_TYPES and int(joint_type[reference]) in _SUPPORTED_JOINT_TYPES:
            has_supported = True
            continue
        unsupported_count += 1
        if len(unsupported_sample) < _MAX_REPORTED_UNSUPPORTED_JOINTS:
            unsupported_sample.append(follower)

    if unsupported_count:
        omitted_count = unsupported_count - len(unsupported_sample)
        omitted_suffix = f"; {omitted_count} additional indices omitted" if omitted_count else ""
        warnings.warn(
            f"{solver_name} ignores joint-owned mimic relationships unless both joints are PRISMATIC, "
            f"REVOLUTE, or D6; unsupported follower joint indices: {unsupported_sample}{omitted_suffix}.",
            stacklevel=3,
        )
    return has_supported
