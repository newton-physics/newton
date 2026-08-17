# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

""":func:`select_joints` and its result type — a helper for computing the
``joint_q_idx`` / ``joint_qd_idx`` pair that
:class:`~newton.controllers.ControllerJointImpedance` requires.

:func:`select_joints` is a pure helper: it does not construct a controller and
is never passed to one. It only resolves a set of joints against a
:class:`~newton.Model` into the flat index arrays that controller expects.
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

    Controlled DOFs are grouped by articulation, matching the ``(robot 0's
    indices first, then robot 1's, ...)`` layout
    :class:`~newton.controllers.ControllerJointImpedance` requires. Within an
    articulation the order follows the ``joints`` argument when one is given,
    and model joint index otherwise.

    Both arrays are ``int32`` so they can be used directly as Warp indexed-view
    subscripts (``sim_array[selection.qd_idx]``), which is how ports are bound
    to simulation-sized arrays.
    """

    q_idx: wp.array[wp.int32]
    """Model coordinate index of each controlled DOF, shape [controlled_dof_count]."""

    qd_idx: wp.array[wp.int32]
    """Model DOF index of each controlled DOF, shape [controlled_dof_count]."""


def select_joints(
    model: Model,
    *,
    articulations: list[int] | list[str] | None = None,
    joints: list[int] | list[str] | None = None,
) -> JointSelection:
    """Resolve a set of joints to control into the index arrays a controller needs.

    Integers match exactly. Labels (:attr:`~newton.Model.articulation_label`,
    :attr:`~newton.Model.joint_label`) match by string equality and select
    every match, so a label shared by several robots selects one joint on
    each of them. Every entry must match at least one thing, or the call
    raises; for ``joints``, a label only needs to match in one selected
    articulation, so it is safe to use a label that exists on some robots of
    a heterogeneous fleet but not others.

    Args:
        model: Model to select from.
        articulations: Articulation indices or labels to control. ``None``
            selects all. Duplicates — whether repeated indices or an index and
            a label that resolve to the same articulation — are collapsed, so
            no joint is ever selected twice.
        joints: Model joint indices or labels to control within the selected
            articulations. ``None`` selects every Revolute/Prismatic joint of
            each selected articulation; every other joint (Fixed, or any
            multi-DOF type) is skipped rather than controlled. Passed
            explicitly, joints are taken as-is; whether each one is
            controllable is checked by the controller, not here.

    Returns:
        The matched coordinate/DOF index pair addressing the selected DOFs, in
        the grouped-by-articulation layout
        :class:`~newton.controllers.ControllerJointImpedance` expects for
        ``joint_q_idx`` and ``joint_qd_idx``.

    Raises:
        ValueError: If the model has no articulations, an entry of
            ``articulations`` or ``joints`` matches nothing, or the selection
            resolves to zero joints.

    Example:
        .. code-block:: python

            selection = select_joints(model, joints=["shoulder", "elbow", "wrist"])
            controller = ControllerJointImpedance(
                model,
                joint_q_idx=selection.q_idx,
                joint_qd_idx=selection.qd_idx,
                stiffness=kp,
                damping=kd,
            )
            outputs = controller.output()
            # Scatter the compact torque command straight into the simulation.
            outputs.joint_f = control.joint_f[selection.qd_idx]
    """
    if model.articulation_count == 0:
        raise ValueError("model contains no articulations; nothing can be controlled.")

    art_start = model.articulation_start.numpy()
    art_end = model.articulation_end.numpy()
    joint_type = model.joint_type.numpy()
    joint_label = model.joint_label
    q_start = model.joint_q_start.numpy()
    qd_start = model.joint_qd_start.numpy()

    if articulations is None:
        selected_arts = list(range(model.articulation_count))
    else:
        matched_arts: list[int] = []
        for entry in articulations:
            if isinstance(entry, str):
                matches = [i for i, label in enumerate(model.articulation_label) if label == entry]
                if not matches:
                    raise ValueError(f"articulation label {entry!r} matches no articulation in the model.")
                matched_arts.extend(matches)
            else:
                if not 0 <= entry < model.articulation_count:
                    raise ValueError(
                        f"articulation index {entry} is out of range for a model with "
                        f"{model.articulation_count} articulations."
                    )
                matched_arts.append(entry)
        # An index and a label can name the same articulation; selecting it
        # twice would duplicate every one of its joints in the output.
        selected_arts = sorted(dict.fromkeys(matched_arts))

    robot_joints_by_art: dict[int, list[int]] = {art: [] for art in selected_arts}
    if joints is None:
        for art in selected_arts:
            art_joints = np.arange(art_start[art], art_end[art])
            robot_joints_by_art[art] = art_joints[np.isin(joint_type[art_joints], _SCALAR_JOINT_TYPES)].tolist()
    else:
        for entry in joints:
            if isinstance(entry, str):
                matched_any = False
                for art in selected_arts:
                    matches = [j for j in range(art_start[art], art_end[art]) if joint_label[j] == entry]
                    if matches:
                        matched_any = True
                        robot_joints_by_art[art].extend(matches)
                if not matched_any:
                    raise ValueError(f"joint label {entry!r} matches no joint in the selected articulations.")
            else:
                owning_art = next((art for art in selected_arts if art_start[art] <= entry < art_end[art]), None)
                if owning_art is None:
                    raise ValueError(f"joint index {entry} is not a joint of any selected articulation.")
                robot_joints_by_art[owning_art].append(entry)

    q_idx_chunks: list[np.ndarray] = []
    qd_idx_chunks: list[np.ndarray] = []
    for art in selected_arts:
        robot_joints = np.asarray(robot_joints_by_art[art], dtype=np.int64)
        if robot_joints.size == 0:
            continue
        q_idx_chunks.append(q_start[robot_joints])
        qd_idx_chunks.append(qd_start[robot_joints])

    if not q_idx_chunks:
        raise ValueError("selection resolved to zero controlled joints.")

    device = model.device
    return JointSelection(
        q_idx=wp.array(np.concatenate(q_idx_chunks), dtype=wp.int32, device=device),
        qd_idx=wp.array(np.concatenate(qd_idx_chunks), dtype=wp.int32, device=device),
    )
