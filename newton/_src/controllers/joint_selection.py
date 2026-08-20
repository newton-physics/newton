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

import re
from dataclasses import dataclass

import numpy as np
import warp as wp

from newton._src.sim.model import Model

from ..utils.selection import get_name_from_label, match_labels


@dataclass(frozen=True)
class JointSelection:
    """Index arrays addressing a set of controlled joints in a :class:`~newton.Model`.

    Returned by :func:`select_joints`. Each controlled DOF carries both a
    coordinate index into :attr:`~newton.State.joint_q` and a DOF index into
    :attr:`~newton.State.joint_qd`, since the two spaces differ once any
    uncontrolled joint upstream spans more coordinates than DOFs.

    Controlled DOFs are grouped by robot, matching the ``(robot 0's
    indices first, then robot 1's, ...)`` layout
    :class:`~newton.controllers.ControllerJointImpedance` requires. Within a
    robot the order follows the ``joints`` argument when one is given,
    and model joint index otherwise.

    Both arrays are ``int32`` so they can be used directly as Warp indexed-view
    subscripts (``sim_array[selection.qd_idx]``), which is how ports are bound
    to simulation-sized arrays.
    """

    q_idx: wp.array[wp.int32]
    """Model coordinate index of each controlled DOF, shape [controlled_dof_count]."""

    qd_idx: wp.array[wp.int32]
    """Model DOF index of each controlled DOF, shape [controlled_dof_count]."""


def _resolve_joint_entry(
    entry: int | str | re.Pattern[str],
    joint_names: list[str],
    joint_to_art: np.ndarray,
    selected_arts_set: set[int],
) -> list[tuple[int, int]]:
    """Resolve one ``joints`` entry to its ``(articulation, joint)`` contributions."""
    if isinstance(entry, int):
        owning_art = joint_to_art[entry] if 0 <= entry < len(joint_to_art) else None
        if owning_art is None or owning_art not in selected_arts_set:
            raise ValueError(f"joint index {entry} is not a joint of any selected articulation.")
        return [(owning_art, entry)]

    matched = match_labels(joint_names, entry)
    contributions = [(joint_to_art[j], j) for j in matched if joint_to_art[j] in selected_arts_set]
    if not contributions:
        raise ValueError(f"joint pattern {entry!r} matches no joint in the selected articulations.")
    return contributions


def select_joints(
    model: Model,
    *,
    articulations: list[int | str | re.Pattern[str]] | str | re.Pattern[str] | None = None,
    joints: list[int | str | re.Pattern[str]] | str | re.Pattern[str] | None = None,
) -> JointSelection:
    """Resolve a set of joints to control into the index arrays a controller needs.

    Integers match exactly. ``articulations`` patterns are matched against the
    full :attr:`~newton.Model.articulation_label` following
    :ref:`label-matching`. ``joints`` patterns are matched against the leaf
    component of :attr:`~newton.Model.joint_label` (the part after the last
    ``/``), so a prefix added by :meth:`~newton.ModelBuilder.add_builder`
    (e.g. ``"panda_0/shoulder"``) does not need to be repeated in the
    pattern — a pattern shared by several robots selects one joint on each of
    them.

    An entry that matches nothing at all raises. For ``joints``, matching
    anywhere in the selection is enough, so one joint list can serve a
    heterogeneous fleet: asking for ``"wrist"`` across two robots when only one
    has a wrist selects that one and leaves the other with fewer controlled
    DOFs.

    Args:
        model: Model to select from.
        articulations: Articulation indices or label patterns to control, as a
            list or as a single pattern. ``None`` selects all. Duplicates —
            whether repeated indices or an index and a pattern that resolve to
            the same articulation — are collapsed, so no joint is ever selected
            twice.
        joints: Model joint indices or label patterns to control within the
            selected articulations, as a list or as a single pattern. ``None``
            selects every single-coordinate, single-DOF joint of each selected
            articulation; every other joint (Fixed, or any multi-DOF type) is
            skipped rather than controlled. Passed explicitly, joints are taken
            as-is; whether each one is controllable is checked by the
            controller, not here. Duplicates are collapsed, as they are for
            ``articulations``.

    Returns:
        The matched coordinate/DOF index pair addressing the selected DOFs, in
        the grouped-by-robot layout
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

    # A lone pattern is a selection of one; without this it would iterate as
    # characters.
    if isinstance(articulations, str | re.Pattern):
        articulations = [articulations]
    if isinstance(joints, str | re.Pattern):
        joints = [joints]

    art_start = model.articulation_start.numpy()
    art_end = model.articulation_end.numpy()
    joint_label = model.joint_label
    q_start = model.joint_q_start.numpy()
    qd_start = model.joint_qd_start.numpy()

    if articulations is None:
        selected_arts = list(range(model.articulation_count))
    else:
        matched_arts: list[int] = []
        for entry in articulations:
            if not isinstance(entry, int):
                matches = match_labels(model.articulation_label, entry)
                if not matches:
                    raise ValueError(f"articulation pattern {entry!r} matches no articulation in the model.")
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

    # A joint is controllable when its position error is a scalar subtraction,
    # i.e. it spans exactly one coordinate and one DOF.
    is_scalar = (np.diff(q_start) == 1) & (np.diff(qd_start) == 1)

    robot_joints_by_art: dict[int, list[int]] = {art: [] for art in selected_arts}
    if joints is None:
        for art in selected_arts:
            art_joints = np.arange(art_start[art], art_end[art])
            robot_joints_by_art[art] = art_joints[is_scalar[art_joints]].tolist()
    else:
        # Maps a joint index to its owning articulation, so a joint's art can be
        # looked up directly instead of scanning ``selected_arts`` per joint.
        joint_to_art = np.searchsorted(art_end, np.arange(model.joint_count), side="right")
        selected_arts_set = set(selected_arts)
        # Match against leaf names, not full labels, so a pattern like "shoulder"
        # selects that joint on every robot regardless of its add_builder prefix.
        joint_names = [get_name_from_label(label) for label in joint_label]
        for entry in joints:
            for art, j in _resolve_joint_entry(entry, joint_names, joint_to_art, selected_arts_set):
                robot_joints_by_art[art].append(j)

    q_idx_chunks: list[np.ndarray] = []
    qd_idx_chunks: list[np.ndarray] = []
    for art in selected_arts:
        # A joint named twice — repeated in ``joints``, or matched by both an
        # index and a label — would otherwise be controlled twice, aliasing two
        # controlled slots onto one simulation DOF. Order is preserved.
        robot_joints = np.asarray(list(dict.fromkeys(robot_joints_by_art[art])), dtype=np.int64)
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
