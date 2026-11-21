# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

""":func:`select_joints` and :func:`resolve_joint_selection` — internal
helpers every model-based controller uses to resolve its
``articulations``/``joints`` constructor arguments into index arrays.

Both are pure helpers: neither constructs a controller itself.
:func:`select_joints` only resolves a set of joints against a
:class:`~newton.Model` into a :class:`JointSelection`; :func:`resolve_joint_selection`
additionally validates that selection and packs it by robot, since every
model-based controller (:class:`~newton.controllers.ControllerDifferentialIK`,
:class:`~newton.controllers.ControllerOperationalSpace`,
:class:`~newton.controllers.ControllerJointImpedance`) needs the same checks
and bookkeeping.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import warp as wp

from newton import JointType
from newton._src.sim.model import Model

from ..utils.selection import get_name_from_label, match_labels
from .utils import _validate_array


@dataclass(frozen=True)
class JointSelection:
    """Index arrays addressing a set of controlled joints in a :class:`~newton.Model`.

    Returned by :func:`select_joints`. Each entry is a controlled joint's
    *starting* coordinate index into :attr:`~newton.State.joint_q` and
    starting DOF index into :attr:`~newton.State.joint_qd` — not one entry
    per coordinate or DOF. To find a joint's end instead, use the model's own
    start arrays: ``model.joint_q_start[j + 1]`` (and
    ``model.joint_qd_start[j + 1]``) give the exclusive end of joint ``j``'s
    coordinate (DOF) span.

    Controlled DOFs are grouped by robot, matching the ``(robot 0's
    indices first, then robot 1's, ...)`` layout.
    Within a robot the order follows the ``joints`` argument when one is given,
    and model joint index otherwise.
    """

    q_start: wp.array[wp.int32]
    """Model starting coordinate index of each selected joint, shape [selected_joint_count]."""

    qd_start: wp.array[wp.int32]
    """Model starting DOF index of each selected joint, shape [selected_joint_count]."""


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
    """Resolve a set of joints to control into the starting index arrays a controller needs.

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
            records the starting coordinate/DOF index of every *controllable*
            joint — one spanning exactly one coordinate and one DOF — in each
            selected articulation; any other joint (Fixed, or a multi-DOF type
            such as a floating base) is silently left out, so a model mixing
            controllable and uncontrollable joints does not need to be pruned
            by hand. This filtering applies only to the default: a joint named
            explicitly is not screened for controllability — Fixed still
            contributes no entry, since it has no starting index to give, but
            any other uncontrollable joint is left in for the controller to
            reject, so naming one by name or pattern still raises rather than
            being silently dropped. Every selected joint contributes exactly
            one entry, its starting coordinate/DOF index — not one entry per
            coordinate or DOF. Duplicates are collapsed, as they are for
            ``articulations``.

    Returns:
        The matched starting coordinate/DOF index pair for each selected
        joint, in the grouped-by-robot layout
        :class:`~newton.controllers.ControllerJointImpedance` builds
        internally from its ``articulations``/``joints`` arguments.

    Raises:
        ValueError: If the model has no articulations, an entry of
            ``articulations`` or ``joints`` matches nothing, or the selection
            resolves to zero joints.
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

    robot_joints_by_art: dict[int, list[int]] = {art: [] for art in selected_arts}
    if joints is None:
        for art in selected_arts:
            robot_joints_by_art[art] = np.arange(art_start[art], art_end[art]).tolist()
    else:
        joint_to_art = model.joint_articulation.numpy()
        selected_arts_set = set(selected_arts)
        # Match against leaf names, not full labels, so a pattern like "shoulder"
        # selects that joint on every robot regardless of its add_builder prefix.
        joint_names = [get_name_from_label(label) for label in joint_label]
        for entry in joints:
            for art, j in _resolve_joint_entry(entry, joint_names, joint_to_art, selected_arts_set):
                robot_joints_by_art[art].append(j)

    q_start_chunks: list[np.ndarray] = []
    qd_start_chunks: list[np.ndarray] = []
    for art in selected_arts:
        # A joint named twice — repeated in ``joints``, or matched by both an
        # index and a label — would otherwise be controlled twice, aliasing two
        # controlled slots onto one simulation DOF. Order is preserved.
        robot_joints = np.asarray(list(dict.fromkeys(robot_joints_by_art[art])), dtype=np.int64)
        # A joint with zero coordinates and zero DOFs (Fixed) has no starting
        # index of its own: q_start/qd_start alias the next joint's, or run
        # past the end of the model's arrays entirely if it is the last
        # joint. q_start/qd_start carry a trailing sentinel (length
        # joint_count + 1), so q_start[j + 1] - q_start[j] is that joint's
        # coordinate count without a separate end array.
        coord_count = q_start[robot_joints + 1] - q_start[robot_joints]
        dof_count = qd_start[robot_joints + 1] - qd_start[robot_joints]
        if joints is None:
            # Only the default is narrowed to what the controller can actually
            # use; a joint named explicitly is left in even if not 1x1, so the
            # controller raises instead of the joint silently disappearing.
            eligible = (coord_count == 1) & (dof_count == 1)
        else:
            eligible = (coord_count > 0) & (dof_count > 0)
        robot_joints = robot_joints[eligible]
        if robot_joints.size == 0:
            continue
        q_start_chunks.append(q_start[robot_joints])
        qd_start_chunks.append(qd_start[robot_joints])

    if not q_start_chunks:
        raise ValueError("selection resolved to zero controlled joints.")

    device = model.device
    return JointSelection(
        q_start=wp.array(np.concatenate(q_start_chunks), dtype=wp.int32, device=device),
        qd_start=wp.array(np.concatenate(qd_start_chunks), dtype=wp.int32, device=device),
    )


@dataclass(frozen=True)
class ResolvedJointSelection:
    """Validated, packed-robot-grouped form of a :class:`JointSelection`.

    Returned by :func:`resolve_joint_selection`. ``q_idx``/``qd_idx`` are the
    (cloned) input arrays; the rest are derived from them.
    """

    q_idx: wp.array[wp.int32]
    qd_idx: wp.array[wp.int32]
    q_idx_np: np.ndarray
    qd_idx_np: np.ndarray
    owning_joint: np.ndarray
    """Model joint index owning each controlled coordinate/DOF, shape [total_controlled_dofs]."""
    model_robot_index_np: np.ndarray
    """Ascending model articulation index of each controlled (packed) robot slot."""
    controlled_dofs_per_robot_np: np.ndarray
    controlled_dofs_per_robot: wp.array[wp.int32]
    controlled_robot_count: int
    max_controlled_dofs: int
    total_controlled_dofs: int
    model_robot_index: wp.array[wp.int32]
    controlled_robot_mask: wp.array[wp.bool]
    """model_robot_count-length mask, true for every controlled articulation."""


def resolve_joint_selection(
    model: Model,
    *,
    articulations: list[int | str | re.Pattern[str]] | str | re.Pattern[str] | None,
    joints: list[int | str | re.Pattern[str]] | str | re.Pattern[str] | None,
    device: wp.DeviceLike,
    controller_name: str,
    ownerless_joint_reason: str,
) -> ResolvedJointSelection:
    """Call :func:`select_joints`, validate its result, and pack it by robot.

    Shared by every model-based controller
    (:class:`~newton.controllers.ControllerDifferentialIK`,
    :class:`~newton.controllers.ControllerOperationalSpace`,
    :class:`~newton.controllers.ControllerJointImpedance`): each takes the
    same ``articulations``/``joints`` arguments and needs the same
    validation and packed-robot bookkeeping, only the wording of two error
    messages differs.

    Args:
        model: Model to select from.
        articulations: Forwarded to :func:`select_joints`.
        joints: Forwarded to :func:`select_joints`.
        device: Device the returned arrays live on (the caller's own
            ``self._device``, which may differ from ``model.device``).
        controller_name: Class name, used in the "unsupported joint" message.
        ownerless_joint_reason: Clause appended to the "belongs to no robot"
            message, naming what the caller cannot compute without an owner
            (e.g. "so such a joint has no Jacobian.").

    Raises:
        ValueError: If the selection fails any of the checks documented on
            :class:`ResolvedJointSelection`'s fields.
    """
    joint_selection = select_joints(model, articulations=articulations, joints=joints)
    joint_q_idx = joint_selection.q_start
    joint_qd_idx = joint_selection.qd_start

    # ------------------------------------------------------------------
    # Validate the two model-space index arrays select_joints returns:
    #   1. type/dtype/shape of q_start, then qd_start against its length
    #   2. non-empty
    #   3. both index within the model's coordinate/DOF space
    #   4. qd_start has no duplicate DOF
    #   5. q_start[i]/qd_start[i] name the same joint for every i
    #   6. every addressed joint spans a single coordinate and single DOF
    #   7. every joint belongs to a robot (articulation)
    #   8. joints are grouped by robot, ascending
    # ------------------------------------------------------------------
    if not isinstance(joint_q_idx, wp.array):
        raise TypeError(f"joint_selection.q_start must be a wp.array, got {type(joint_q_idx).__name__}.")
    _validate_array(
        array=joint_q_idx,
        name="joint_selection.q_start",
        dtype=wp.int32,
        shape=(joint_q_idx.size,),
        device=device,
    )
    total_controlled_dofs = int(joint_q_idx.size)
    if total_controlled_dofs < 1:
        raise ValueError("joint_selection.q_start is empty; there is nothing to control.")
    _validate_array(
        array=joint_qd_idx,
        name="joint_selection.qd_start",
        dtype=wp.int32,
        shape=(total_controlled_dofs,),
        device=device,
    )

    q_idx_np = joint_q_idx.numpy()
    qd_idx_np = joint_qd_idx.numpy()
    coord_count = int(model.joint_coord_count)
    dof_count = int(model.joint_dof_count)
    for name, idx_np, limit, space in (
        ("joint_selection.q_start", q_idx_np, coord_count, "coordinate"),
        ("joint_selection.qd_start", qd_idx_np, dof_count, "DOF"),
    ):
        if idx_np.min() < 0 or idx_np.max() >= limit:
            raise ValueError(
                f"{name} must index the model's {space} space [0, {limit}), got "
                f"range [{int(idx_np.min())}, {int(idx_np.max())}]."
            )

    if np.unique(qd_idx_np).size != qd_idx_np.size:
        duplicate = int(np.bincount(qd_idx_np).argmax())
        raise ValueError(
            f"joint_selection.qd_start contains DOF {duplicate} more than once; two controlled slots "
            f"cannot map to the same simulation DOF."
        )

    owning_joint = np.searchsorted(model.joint_q_start.numpy(), q_idx_np, side="right") - 1
    owning_joint_qd = np.searchsorted(model.joint_qd_start.numpy(), qd_idx_np, side="right") - 1
    if not np.array_equal(owning_joint, owning_joint_qd):
        mismatched = int(np.flatnonzero(owning_joint != owning_joint_qd)[0])
        raise ValueError(
            f"joint_selection.q_start and joint_selection.qd_start disagree at entry {mismatched}: "
            f"coordinate {int(q_idx_np[mismatched])} belongs to joint {int(owning_joint[mismatched])} "
            f"but DOF {int(qd_idx_np[mismatched])} belongs to joint {int(owning_joint_qd[mismatched])}. "
            f"Did you swap the two arrays?"
        )

    # A joint is controllable when its DOF maps to exactly one Jacobian
    # column, i.e. it spans exactly one coordinate and one DOF.
    joint_type_np = model.joint_type.numpy()
    coord_span = np.diff(model.joint_q_start.numpy())[owning_joint]
    dof_span = np.diff(model.joint_qd_start.numpy())[owning_joint]
    unsupported = sorted(
        {
            (int(j), JointType(joint_type_np[j]).name)
            for j, coords, dofs in zip(owning_joint, coord_span, dof_span, strict=True)
            if coords != 1 or dofs != 1
        }
    )
    if unsupported:
        raise ValueError(
            f"{controller_name} only supports controlling joints that span a single coordinate and a "
            f"single DOF; joint_selection addresses unsupported joints: {unsupported}"
        )

    owning_robot = model.joint_articulation.numpy()[owning_joint]
    loose = np.flatnonzero(owning_robot < 0)
    if loose.size:
        raise ValueError(
            f"joint_selection addresses joint {int(owning_joint[loose[0]])}, which belongs to no "
            f"robot. {ownerless_joint_reason}"
        )
    if np.any(np.diff(owning_robot) < 0):
        raise ValueError(
            "joint_selection.q_start/qd_start must be grouped by robot (robot 0's DOFs first, "
            f"then robot 1's, ...); got robot order {owning_robot.tolist()}."
        )

    model_robot_index_np, controlled_dofs_per_robot_np = np.unique(owning_robot, return_counts=True)
    model_robot_index_np = model_robot_index_np.astype(np.int32)
    controlled_dofs_per_robot_np = controlled_dofs_per_robot_np.astype(np.int32)
    controlled_robot_count = int(model_robot_index_np.size)
    controlled_dofs_per_robot = wp.array(controlled_dofs_per_robot_np, dtype=wp.int32, device=device)
    max_controlled_dofs = int(controlled_dofs_per_robot_np.max())
    model_robot_index = wp.array(model_robot_index_np, dtype=wp.int32, device=device)
    mask_np = np.zeros(int(model.articulation_count), dtype=bool)
    mask_np[model_robot_index_np] = True
    controlled_robot_mask = wp.array(mask_np, dtype=wp.bool, device=device)

    return ResolvedJointSelection(
        q_idx=wp.clone(joint_q_idx),
        qd_idx=wp.clone(joint_qd_idx),
        q_idx_np=q_idx_np,
        qd_idx_np=qd_idx_np,
        owning_joint=owning_joint,
        model_robot_index_np=model_robot_index_np,
        controlled_dofs_per_robot_np=controlled_dofs_per_robot_np,
        controlled_dofs_per_robot=controlled_dofs_per_robot,
        controlled_robot_count=controlled_robot_count,
        max_controlled_dofs=max_controlled_dofs,
        total_controlled_dofs=total_controlled_dofs,
        model_robot_index=model_robot_index,
        controlled_robot_mask=controlled_robot_mask,
    )
