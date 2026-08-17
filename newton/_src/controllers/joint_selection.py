# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

""":func:`select_joints` and its result type — a helper for computing the
index arrays :class:`~newton.controllers.ControllerJointImpedance` and
:class:`~newton.controllers.ControllerJointImpedanceModelFree` take as
constructor arguments (``default_dof_indices``, ``joint_q_idx``, ``joint_qd_idx``).

:func:`select_joints` is a pure helper: it does not construct a controller and
is never passed to one. It only resolves a set of joints against a
:class:`~newton.Model` into the flat index arrays those controllers expect.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

from newton import JointType
from newton._src.sim.model import Model

# Joints whose position error ``q_des - q`` is a well-defined scalar subtraction.
_SCALAR_JOINT_TYPES = (int(JointType.REVOLUTE), int(JointType.PRISMATIC))


@dataclass(frozen=True)
class JointSelection:
    """Index arrays addressing a set of controlled joints in a :class:`~newton.Model`.

    Returned by :func:`select_joints`. Each controlled DOF carries both a
    coordinate index into :attr:`newton.State.joint_q` and a DOF index into
    :attr:`newton.State.joint_qd`, since the two spaces differ once any
    uncontrolled joint upstream spans more coordinates than DOFs.

    Controlled DOFs are ordered by articulation, then by model joint index,
    matching the ``(robot 0's indices first, then robot 1's, ...)`` layout
    :class:`~newton.controllers.ControllerJointImpedance` expects.
    """

    q_idx: wp.array[wp.uint32]
    """Model coordinate index of each controlled DOF, shape [controlled_dof_count]."""

    qd_idx: wp.array[wp.uint32]
    """Model DOF index of each controlled DOF, shape [controlled_dof_count]."""

    dofs_per_robot: wp.array[wp.int32]
    """Controlled DOF count of each robot, shape [robot_count]."""

    robot_count: int
    """Number of articulations contributing at least one controlled DOF."""

    max_dofs: int
    """Largest controlled DOF count of any single robot."""


def select_joints(
    model: Model,
    *,
    articulations: list[int] | None = None,
    joints: list[int] | None = None,
) -> JointSelection:
    """Resolve a set of joints to control into the index arrays a controller needs.

    Args:
        model: Model to select from.
        articulations: Articulation indices to control. ``None`` selects all.
        joints: Model joint indices to control within the selected
            articulations. ``None`` selects every Revolute/Prismatic joint of
            each selected articulation; every other joint (Fixed, or any
            multi-DOF type) is skipped rather than controlled. Passed
            explicitly, every joint must be Revolute or Prismatic.

    Returns:
        Index tables addressing the selected DOFs, in the layout
        :class:`~newton.controllers.ControllerJointImpedance` expects for
        ``default_dof_indices`` and its ``*_idx`` overrides.

    Raises:
        ValueError: If the model has no articulations, an articulation index
            is out of range, or an explicitly listed joint is not 1-DOF
            revolute or prismatic.

    Example:
        .. code-block:: python

            selection = select_joints(model, joints=[shoulder, elbow, wrist])
            controller = ControllerJointImpedance(
                model,
                default_dof_indices=selection.q_idx,
                joint_qd_idx=selection.qd_idx,
                stiffness=kp,
                damping=kd,
            )
    """
    if model.articulation_count == 0:
        raise ValueError("model contains no articulations; nothing can be controlled.")

    art_start = model.articulation_start.numpy()
    art_end = model.articulation_end.numpy()
    joint_type = model.joint_type.numpy()
    q_start = model.joint_q_start.numpy()
    qd_start = model.joint_qd_start.numpy()

    selected_arts = range(model.articulation_count) if articulations is None else articulations
    for art in selected_arts:
        if not 0 <= art < model.articulation_count:
            raise ValueError(
                f"articulation index {art} is out of range for a model with {model.articulation_count} articulations."
            )

    q_idx_chunks: list[np.ndarray] = []
    qd_idx_chunks: list[np.ndarray] = []
    dofs_per_robot: list[int] = []

    for art in selected_arts:
        art_joints = np.arange(art_start[art], art_end[art])
        if joints is None:
            robot_joints = art_joints[np.isin(joint_type[art_joints], _SCALAR_JOINT_TYPES)]
        else:
            robot_joints = np.asarray([j for j in joints if art_start[art] <= j < art_end[art]], dtype=np.int64)
            unsupported = [
                (int(j), JointType(joint_type[j]).name)
                for j in robot_joints
                if joint_type[j] not in _SCALAR_JOINT_TYPES
            ]
            if unsupported:
                raise ValueError(f"select_joints only supports 1-DOF revolute or prismatic joints; got {unsupported}.")
        if robot_joints.size == 0:
            continue
        q_idx_chunks.append(q_start[robot_joints])
        qd_idx_chunks.append(qd_start[robot_joints])
        dofs_per_robot.append(int(robot_joints.size))

    if not dofs_per_robot:
        raise ValueError("selection resolved to zero controlled joints.")

    device = model.device
    return JointSelection(
        q_idx=wp.array(np.concatenate(q_idx_chunks), dtype=wp.uint32, device=device),
        qd_idx=wp.array(np.concatenate(qd_idx_chunks), dtype=wp.uint32, device=device),
        dofs_per_robot=wp.array(np.array(dofs_per_robot, dtype=np.int32), dtype=wp.int32, device=device),
        robot_count=len(dofs_per_robot),
        max_dofs=max(dofs_per_robot),
    )
