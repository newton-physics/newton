# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

""":func:`resolve_tool_sites` — an internal helper shared by every model-based
controller that takes a ``tool_sites`` argument
(:class:`~newton.controllers.ControllerDifferentialIK`,
:class:`~newton.controllers.ControllerOperationalSpace`): resolves the Newton
*sites* matched on each controlled robot into the flat, per-frame
body/transform/Jacobian-row-index triplet the controller needs each step.

A pure helper: it does not construct a controller itself.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import warp as wp

from newton._src.geometry.flags import ShapeFlags
from newton._src.sim.model import Model

from ..utils.selection import get_name_from_label, match_labels


@dataclass(frozen=True)
class ResolvedToolSites:
    """One or more tool frames per controlled robot, resolved by :func:`resolve_tool_sites`.

    Every array is flat and frame-indexed, robot 0's frames first, then
    robot 1's, matching :attr:`frames_per_robot` — the same layout
    :class:`~newton.controllers.ControllerDifferentialIKModelFree` uses for
    its own ``frames_per_robot``/per-frame ports. Within a robot, frames are
    ordered by ``tool_sites``'s own entry order, not by model site index —
    see :func:`resolve_tool_sites`.
    """

    tool_body: wp.array[wp.int32]
    """Model body index of each frame's tool site, shape [total_frame_count]."""

    tool_transform_body: wp.array[wp.transform]
    """Tool site's fixed transform relative to :attr:`tool_body`, shape [total_frame_count]."""

    robot_link_idx: wp.array[wp.int32]
    """Tool site's row-block index within its own articulation's eval_jacobian output, shape [total_frame_count]."""

    frames_per_robot: wp.array[wp.int32]
    """Matched tool-frame count for each controlled robot, shape [controlled_robot_count]."""


def resolve_tool_sites(
    model: Model,
    *,
    model_robot_index_np: np.ndarray,
    tool_sites: list[int | str | re.Pattern[str]] | str | re.Pattern[str],
    device: wp.DeviceLike,
) -> ResolvedToolSites:
    """Match ``tool_sites`` to one or more sites per robot in ``model_robot_index_np``.

    ``model_robot_index_np`` is expected to be the already-ordered set of
    controlled articulations a :func:`~newton.controllers.resolve_joint_selection`
    call resolved, so there is no second, independent articulation resolution
    to keep in sync with the first.

    Within a robot, matched frames are ordered to match ``tool_sites``'s own
    entry order (an entry that itself matches several sites on one robot,
    e.g. a shared pattern, contributes them in ascending model site index),
    not by model site index overall -- mirroring
    :func:`~newton.controllers.select_joints`'s "order follows the ``joints``
    argument when one is given".

    Args:
        model: Model to select from.
        model_robot_index_np: Ascending model articulation index of each
            controlled (packed) robot slot.
        tool_sites: Site indices or label patterns selecting each controlled
            robot's tool frame(s), as a list or as a single pattern/index.
        device: Device the returned arrays live on (the caller's own
            ``self._device``, which may differ from ``model.device``).

    Raises:
        ValueError: If the model has no sites, an entry of ``tool_sites``
            matches nothing, a robot has zero matching sites, a matched site
            is attached to no body (a world-fixed reference frame), or a
            matched site's body is not moved by any joint.
    """
    joint_child_np = model.joint_child.numpy()
    joint_articulation_np = model.joint_articulation.numpy()
    body_to_articulation_np = np.full(model.body_count, -1, dtype=np.int32)
    body_to_articulation_np[joint_child_np] = joint_articulation_np

    shape_flags_np = model.shape_flags.numpy()
    shape_body_np = model.shape_body.numpy()
    shape_transform_np = model.shape_transform.numpy()
    site_indices_np = np.flatnonzero((shape_flags_np & ShapeFlags.SITE) != 0)
    if site_indices_np.size == 0:
        raise ValueError("model contains no sites; add one with ModelBuilder.add_site for the tool frame.")
    # A site attached to no body (ModelBuilder.add_site(-1, ...), a
    # world-fixed reference frame) has no articulation. Resolved explicitly
    # rather than via body_to_articulation_np[-1], which would silently
    # alias onto whatever articulation the model's last body happens to
    # belong to.
    site_body_np = shape_body_np[site_indices_np]
    site_articulation_np = np.full(site_indices_np.size, -1, dtype=np.int32)
    site_has_body = site_body_np >= 0
    site_articulation_np[site_has_body] = body_to_articulation_np[site_body_np[site_has_body]]
    site_names = [get_name_from_label(model.shape_label[s]) for s in site_indices_np]

    tool_entries = [tool_sites] if isinstance(tool_sites, (int, str, re.Pattern)) else tool_sites
    matched_sites: list[int] = []
    for entry in tool_entries:
        if isinstance(entry, int):
            if entry not in site_indices_np:
                raise ValueError(f"tool_sites index {entry} is not a site in the model.")
            matched_sites.append(entry)
        else:
            local_matches = match_labels(site_names, entry)
            if not local_matches:
                raise ValueError(f"tool_sites pattern {entry!r} matches no site in the model.")
            matched_sites.extend(int(site_indices_np[m]) for m in local_matches)
    # Order-preserving de-duplication: a robot's frames end up ordered by
    # tool_sites's own entry order (mirroring select_joints's "order
    # follows the joints argument when one is given"), not by model site
    # index -- a caller listing ["hand_left", "hand_right", "torso"] gets
    # frames in exactly that order, regardless of which site was added to
    # the model first.
    matched_sites_ordered = list(dict.fromkeys(matched_sites))

    site_index_to_articulation = dict(zip(site_indices_np.tolist(), site_articulation_np.tolist(), strict=True))
    tool_body_list: list[int] = []
    tool_transform_body: list[wp.transform] = []
    frames_per_robot_np = np.zeros(model_robot_index_np.size, dtype=np.int32)
    for robot_slot, art in enumerate(model_robot_index_np.tolist()):
        sites_on_robot = [s for s in matched_sites_ordered if site_index_to_articulation[s] == art]
        if len(sites_on_robot) == 0:
            raise ValueError(f"tool_sites matches no site on articulation {art}.")
        frames_per_robot_np[robot_slot] = len(sites_on_robot)
        for site in sites_on_robot:
            body = int(shape_body_np[site])
            if body < 0:
                raise ValueError(
                    f"tool_sites matches site {site} ('{get_name_from_label(model.shape_label[site])}') on "
                    f"articulation {art}, but that site is attached to no body (added with "
                    f"ModelBuilder.add_site(-1, ...), a world-fixed reference frame); a tool site must be "
                    f"attached to a moving body."
                )
            tool_body_list.append(body)
            tool_transform_body.append(wp.transform(*shape_transform_np[site]))
    tool_body_np = np.array(tool_body_list, dtype=np.int32)

    # robot_link_idx: each frame's tool site's row-block index within its own
    # articulation's eval_jacobian output. eval_jacobian writes link i's rows
    # at [i*6 : i*6+6], where i is the position, within its articulation's
    # own joint range, of the joint that moves the tool site's body -- so
    # this is (that joint's index) minus (the articulation's first joint
    # index).
    body_to_joint_np = np.full(model.body_count, -1, dtype=np.int32)
    body_to_joint_np[joint_child_np] = np.arange(joint_child_np.size, dtype=np.int32)
    tool_site_joint_np = body_to_joint_np[tool_body_np]
    unmoved = np.flatnonzero(tool_site_joint_np < 0)
    if unmoved.size:
        raise ValueError(
            f"tool_sites resolves to body {int(tool_body_np[unmoved[0]])}, which is not moved by any "
            f"joint (not a joint's child body), so it has no Jacobian row to use as a tool frame."
        )
    frame_robot_idx_np = np.repeat(np.arange(model_robot_index_np.size, dtype=np.int32), frames_per_robot_np)
    articulation_start_np = model.articulation_start.numpy()[model_robot_index_np][frame_robot_idx_np]
    robot_link_idx_np = (tool_site_joint_np - articulation_start_np).astype(np.int32)

    return ResolvedToolSites(
        tool_body=wp.array(tool_body_np, dtype=wp.int32, device=device),
        tool_transform_body=wp.array(tool_transform_body, dtype=wp.transform, device=device),
        robot_link_idx=wp.array(robot_link_idx_np, dtype=wp.int32, device=device),
        frames_per_robot=wp.array(frames_per_robot_np, dtype=wp.int32, device=device),
    )
